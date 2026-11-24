"""Generate the committed KPI tables, held-out metrics, and scientific figures."""

import argparse
from collections import defaultdict
from dataclasses import replace
import csv
from hashlib import sha256
from itertools import combinations
from pathlib import Path
import json
import os
import re

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import t

from design import (BAND_SCENARIOS, FACTOR_PAIRS, FIT_SEEDS, MODES, PROFILES,
                    blind_jobs, kernel_jobs, mtag, pair_catalog, pair_jobs, pair_tag)
from contexts import available_model
from rcwa_runner import REGISTRY, Setting
from readout import (ScanOperator, Spectrum, align_spectrum, comparison, curves, evaluate,
                     linear_apd, native_fields, source_grid)
from response_geometry import PITCH, ROOT, Profile, load_mark
from response_model import OUTPUT, open_response

WQ_LIMIT=.005
APD_LIMIT=1.
TRACE_LIMIT=.01
NUM_WQ=.001
NUM_APD=.2


def save_csv(name,rows):
    rows=list(rows)
    if not rows:
        return 0
    path=OUTPUT/f"{name}.csv"
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    temp=path.parent/f".{name}_{os.getpid()}.csv"
    with temp.open("w",encoding="utf-8-sig",newline="") as handle:
        w=csv.DictWriter(handle,keys)
        w.writeheader()
        w.writerows(rows)
    temp.replace(path)
    return len(rows)


class Results:
    def __init__(self,reveal=False):
        self.receipts=[]
        self.marks={}
        self.ops={}
        self.ports={}
        self.ref={}
        self.kpis={}
        self.reveal=reveal
        for file in sorted(REGISTRY.glob("*/*/done.json")):
            data=json.loads(file.read_text(encoding="utf-8"))
            if data["case"].startswith("blind_") and not reveal:
                continue
            data["folder"]=file.parent
            data["id"]=str(file.parent.relative_to(ROOT))
            data["lu_um"]=data.get("requested_lu_um",data["lu_um"])
            self.receipts.append(data)
        self.smooth=[]
        for receipt in self.receipts:
            if ("smooth" in receipt["case"] or "reference" in receipt["case"] or
                    receipt["case"].startswith("pbase_")):
                mark=self.geometry(receipt)
                if mark.info.get("kind")=="smooth" and mark.rigid_nm==0:
                    self.smooth.append(receipt)

    def geometry(self,receipt):
        name=receipt["case"]
        if name not in self.marks:
            m=load_mark(ROOT/"geometry"/f"{name}.npz")
            if m.signature()!=receipt["geometry_hash"]:
                raise ValueError(f"Geometry hash mismatch: {name}")
            self.marks[name]=m
        return self.marks[name]

    def reference(self,receipt,nominal=False):
        key=(receipt["id"],nominal)
        if key not in self.ref:
            s=receipt["setting"]
            mark=self.geometry(receipt)
            profile=Profile() if nominal else mark.profile
            candidates=[]
            for other in self.smooth:
                q=other["setting"]
                if (other["lu_um"]!=receipt["lu_um"] or
                    any(q[k]!=s[k] for k in ("ku","angles","du_nm","li"))):
                    continue
                base=self.geometry(other)
                if base.profile!=profile:
                    continue
                same_length=other["lv_um"]==receipt["lv_um"]
                same_dv=q["dv_nm"]==s["dv_nm"]
                same_h=q["slices"]==s["slices"]
                if profile.shaped and not same_h:
                    continue
                candidates.append((same_length,same_dv,same_h,-q["kv"],other["id"],other))
            self.ref[key]=max(candidates)[-1] if candidates else None
        return self.ref[key]

    def collect_separated(self,spec):
        margin=np.max(abs(spec.incoming))
        q_outer=.6*2*np.pi/.643+margin+1e-9
        if margin*2>=min(2*np.pi/spec.lu,2*np.pi/spec.lv):
            return None
        lx=np.arange(-int(q_outer*spec.lu/(2*np.pi)),int(q_outer*spec.lu/(2*np.pi))+1)
        lz=np.arange(-int(q_outer*spec.lv/(2*np.pi)),int(q_outer*spec.lv/(2*np.pi))+1)
        key=(spec.lu,spec.lv,len(spec.incoming))
        if key not in self.ops:
            self.ops[key]=ScanOperator(spec.lu,spec.lv,lx,lz,spec.incoming)
        data=align_spectrum(spec,spec.lu,spec.lv,lx,lz,spec.incoming)
        return self.ops[key],data

    def read(self,receipt):
        name=receipt["id"]
        if name in self.ports:
            return self.ports[name]
        spec=native_fields(receipt["folder"])
        separated=self.collect_separated(spec)
        result=separated[0](separated[1]) if separated is not None else curves(spec,quadrature=14)
        self.ports[name]=result
        return result

    def kpi(self,receipt):
        if receipt["id"] not in self.kpis:
            mark=self.geometry(receipt)
            ref=self.reference(receipt)
            ideal=self.reference(receipt,nominal=True)
            if ref is None:
                ref=ideal
            if ref is None:
                return None
            data=self.read(receipt)
            own=evaluate(data,self.read(ref),mark.rigid_nm)
            if ideal is not None:
                total=evaluate(data,self.read(ideal),mark.rigid_nm)
            else:
                total={}
            self.kpis[receipt["id"]]=(own,total,ref)
        return self.kpis[receipt["id"]]


def direct_rows(results,extra_groups=()):
    include={"original_curve","smooth_reference","chart_reference","chart_library_reference",
             "mode_reference","chart_nested","chart_library_check","chart_convergence","chart_convergence_ref",
             "chart_band_convergence","chart_band_reference","strict_anchor","strict_anchor_ref",
             "strict_angular","strict_angular_ref","known_translation","known_translation_geometry",
             "amplitude_curve","mode_bank","phase_control","legacy_reproduction","joint_train",
             "blind_new_seed","blind_new_combo","stat_factor_cross","profile_axis","profile_cross",
             "profile_reference","long_cell","long_reference","nested_band_screen","random_band_refinement",
             "random_band_refinement_ref","angular_blind_reference","angular_blind_smooth",
             "height_split_convergence","policy_direct","blind_resolution_diagnosis",
             "blind_resolution_reference","factor_refinement","profile_refinement",
             "interaction_refinement_reference","fine_blind_reference",
             "fine_blind_test","fine_blind_diagnostic"}
    include.update(extra_groups)
    rows=[]
    for receipt in results.receipts:
        if receipt["group"] not in include:
            continue
        if receipt["group"]=="mode_bank":
            term=results.geometry(receipt).info["terms"][0]
            if term[1]!="+" or term[2] not in (1,2,4,8,16):
                continue
        datum=results.kpi(receipt)
        if datum is None:
            continue
        own,total,ref=datum
        m=results.geometry(receipt)
        row=dict(run=receipt["id"],group=receipt["group"],case=receipt["case"],
                 cutoff_um=m.cutoff,Lx_um=m.lu,Lz_um=m.length,setting=receipt["settings_key"],
                 reference=ref["id"],reference_matched_profile=results.geometry(ref).profile==m.profile,
                 seconds=receipt["solve_s"],**receipt["setting"])
        if "native_equivalent_run" in receipt:
            row["native_equivalent_run"]=receipt["native_equivalent_run"]
        row.update({k:v for k,v in m.info.items() if not isinstance(v,(list,dict))})
        profile=m.profile
        row.update(cd_offset_nm=profile.cd_nm,height_nm=profile.height_nm,swa_deg=profile.swa_deg,
                   vertical_shear_nm=profile.shear_nm,footing_nm=profile.footing_nm)
        row["master_hash"]=replace(m,cutoff=1.).signature()
        row.update({k:v for k,v in own.items()})
        row.update(wq_nominal=total.get("wq"),apd_nominal_nm=total.get("apd_nm"))
        if m.rigid_nm==0 and own["position_valid"]:
            sample,template=results.read(receipt),results.read(ref)
            linear=linear_apd(sample[0],sample[1],template[1])
            row.update(apd_linear_nm=linear,apd_linear_difference_nm=optional_difference(linear,own["apd_nm"]))
        rows.append(row)
    return rows


def in_budget(wq,apd):
    return apd is not None and abs(wq)<=WQ_LIMIT and abs(apd)<=APD_LIMIT


def convergence_rows(direct):
    group=defaultdict(list)
    for row in direct:
        group[row["case"]].append(row)
    out=[]
    settings=("ku","kv","du_nm","dv_nm","angles","slices")
    for case,rows in group.items():
        for a,b in combinations(rows,2):
            delta=[k for k in settings if a[k]!=b[k]]
            if not delta or (len(delta)>1 and set(delta)!={"du_nm","dv_nm"}):
                continue
            dim="+".join(delta)
            dwq=b["wq"]-a["wq"]
            dapd=None if None in (a["apd_nm"],b["apd_nm"]) else b["apd_nm"]-a["apd_nm"]
            out.append(dict(case=case,axis=dim,first=a["setting"],second=b["setting"],
                            angle_count_a=a["angles"],angle_count_b=b["angles"],
                            delta_wq_percent=dwq*100,delta_apd_nm=dapd,
                            within_num_budget=dapd is not None and abs(dwq)<=NUM_WQ and abs(dapd)<=NUM_APD))
    return out


def band_pairs(direct,results=None):
    cases=defaultdict(dict)
    receipts={r["id"]:r for r in results.receipts} if results is not None else {}
    for row in direct:
        name=row["case"]
        if name.startswith("chart_s1427") and row["group"]!="known_translation_geometry":
            family="chart_s1427"
        elif row["group"]=="random_band_refinement":
            parameters=tuple(row[k] for k in ("sigma_nm","xi_um","hurst","rho_lr"))
            family=f"band_{BAND_SCENARIOS.index(parameters)}_s{row['seed']}"
        elif row["group"]=="nested_band_screen":
            family=name.rsplit("_b",1)[0]
        elif row["group"] in ("joint_train","stat_factor_cross","factor_refinement","blind_new_seed","blind_new_combo"):
            signatures=(row[k] for k in ("sigma_nm","xi_um","hurst","rho_lr","rho_line","rho_height"))
            family="random_"+"_".join(f"{x:g}" for x in signatures)+f"_s{row['seed']}"
        else:
            continue
        cases[family,row["master_hash"],row["setting"]][row["cutoff_um"]]=row
    out=[]
    for (name,master,setting),data in cases.items():
        full=min(data)
        a=data[full]
        for cutoff,b in sorted(data.items()):
            if cutoff==full:
                continue
            dwq=b["wq"]-a["wq"]
            dapd=None if None in (a["apd_nm"],b["apd_nm"]) else b["apd_nm"]-a["apd_nm"]
            record=dict(geometry=name,master_hash=master,setting=setting,angle_count=a["angles"],
                        full_um=full,filtered_um=cutoff,seed=a.get("seed"),
                        delta_wq_percent=100*dwq,delta_apd_nm=dapd,
                        passes_budget_at_this_setting=in_budget(dwq,dapd),
                        full_run=a["run"],filtered_run=b["run"])
            if receipts:
                trace=comparison(results.read(receipts[b["run"]]),results.read(receipts[a["run"]]))
                record.update(sum_nrmse=trace[0],diff_nrmse=trace[1],passes_scan_at_this_setting=max(trace)<=TRACE_LIMIT)
                first=results.collect_separated(native_fields(receipts[a["run"]]["folder"]))
                second=results.collect_separated(native_fields(receipts[b["run"]]["folder"]))
                if first is not None and second is not None:
                    operator,full_field=first
                    filtered_field=second[1]
                    full_power,full_cross=operator.components(full_field)
                    _,filtered_cross=operator.components(filtered_field)
                    error_power,_=operator.components(filtered_field-full_field)
                    carrier=np.argmin(abs(operator.gx-2*np.pi/PITCH))
                    cross_error=filtered_cross-full_cross
                    record.update(pupil_field_abs_rms=float(np.sqrt(max(error_power,0.))),
                                  pupil_field_relative_rms=float(np.sqrt(max(error_power,0.)/max(full_power,1e-30))),
                                  rsi_overlap_delta_l2=float(np.linalg.norm(cross_error)),
                                  rsi_carrier_delta_real=float(cross_error[carrier].real),
                                  rsi_carrier_delta_imag=float(cross_error[carrier].imag))
            out.append(record)
    return out


def band_convergence(pairs,direct):
    runs={r["run"]:r for r in direct}
    grouped=defaultdict(list)
    for r in pairs:
        grouped[r["geometry"],r["master_hash"],r["full_um"],r["filtered_um"]].append(r)
    axes=("ku","kv","du_nm","dv_nm","angles","slices","li")
    out=[]
    for (name,master,lo,hi),rows in grouped.items():
        for a,b in combinations(rows,2):
            af,bf=[runs[r["full_run"]] for r in (a,b)]
            delta=[k for k in axes if af[k]!=bf[k]]
            if not delta or (len(delta)>1 and set(delta)!={"du_nm","dv_nm"}):
                continue
            if tuple(af[k] for k in delta)>tuple(bf[k] for k in delta):
                a,b,af,bf=b,a,bf,af
            al,bl=[runs[r["filtered_run"]] for r in (a,b)]
            dwq=b["delta_wq_percent"]-a["delta_wq_percent"]
            da=optional_difference(b["delta_apd_nm"],a["delta_apd_nm"])
            fwq=100*(bf["wq"]-af["wq"])
            lwq=100*(bl["wq"]-al["wq"])
            fa=optional_difference(bf["apd_nm"],af["apd_nm"])
            la=optional_difference(bl["apd_nm"],al["apd_nm"])
            out.append(dict(geometry=name,master_hash=master,full_um=lo,filtered_um=hi,
                            axis="+".join(delta),first=a["setting"],second=b["setting"],
                            first_delta_wq_percent=a["delta_wq_percent"],second_delta_wq_percent=b["delta_wq_percent"],
                            first_delta_apd_nm=a["delta_apd_nm"],second_delta_apd_nm=b["delta_apd_nm"],
                            delta_wq_contrast_percent=dwq,delta_apd_contrast_nm=da,
                            full_wq_shift_percent=fwq,full_apd_shift_nm=fa,
                            filtered_wq_shift_percent=lwq,filtered_apd_shift_nm=la,
                            contrast_within_num_budget=within_numerical_budget(dwq,da),
                            both_direct_within_num_budget=(within_numerical_budget(fwq,fa) and
                                                          within_numerical_budget(lwq,la))))
    return out


def optional_difference(after,before):
    return after-before if after is not None and before is not None else None


def within_numerical_budget(wq_percent,apd_nm):
    return apd_nm is not None and abs(wq_percent)<=100*NUM_WQ and abs(apd_nm)<=NUM_APD


def paired_statistics(pairs):
    """Summarize fixed-condition, same-master cutoff contrasts over seeds."""
    groups=defaultdict(dict)
    for r in pairs:
        scenario=re.sub(r"_s[0-9]+$","",r["geometry"])
        key=(scenario,r["setting"],r["angle_count"],r["full_um"],r["filtered_um"])
        groups[key][r["seed"]]=r

    def statistics(values,metric):
        x=np.asarray([v for v in values if v is not None])
        if not len(x):
            return {}
        mean=float(np.mean(x))
        sd=float(np.std(x,ddof=1)) if len(x)>1 else None
        ci=t.ppf(.975,len(x)-1)*sd/np.sqrt(len(x)) if sd is not None else None
        return {f"mean_{metric}":mean,f"sd_{metric}":sd,
                f"rms_{metric}":float(np.sqrt(np.mean(x*x))),
                f"max_abs_{metric}":float(np.max(abs(x))),
                f"mean_95ci_low_{metric}":mean-ci if ci is not None else None,
                f"mean_95ci_high_{metric}":mean+ci if ci is not None else None}

    output=[]
    for (scenario,setting,angles,full,filtered),group in sorted(groups.items()):
        rows=list(group.values())
        traces=[max(r["sum_nrmse"],r["diff_nrmse"]) for r in rows if "sum_nrmse" in r]
        output.append(dict(scenario=scenario,setting=setting,angle_count=angles,full_um=full,
                           filtered_um=filtered,seeds=len(rows),passed=sum(r["passes_budget_at_this_setting"] for r in rows),
                           passed_kpi_and_scan=sum(r["passes_budget_at_this_setting"] and r.get("passes_scan_at_this_setting",False) for r in rows),
                           max_scan_nrmse=max(traces,default=None),
                           **statistics([r["delta_wq_percent"] for r in rows],"wq_percent"),
                           **statistics([r["delta_apd_nm"] for r in rows],"apd_nm")))
    return output


def profile_response_curves(direct):
    rows=[]
    for r in direct:
        if r["group"] not in ("profile_axis","profile_cross","profile_refinement","profile_reference"):
            continue
        tag="pbase" if r["group"]=="profile_reference" else "profile"
        if tag=="pbase" and not r["case"].endswith("_u2"):
            continue
        for axis,levels in PROFILES.items():
            prefix=f"{tag}_{axis}_"
            if not r["case"].startswith(prefix):
                continue
            level=int(r["case"].removeprefix(prefix).split("_")[0])
            nominal=r["wq_nominal"]
            matched=r["reference_matched_profile"]
            rows.append(dict(profile_axis=axis,value=levels[level],sigma_nm=r.get("sigma_nm",0.),
                             seed=r.get("seed"),setting=r["setting"],
                             wq_delta_from_matched_profile_percent=100*(r["wq"]-1) if matched else None,
                             apd_from_matched_profile_nm=r["apd_nm"] if matched else None,
                             wq_delta_from_nominal_percent=None if nominal is None else 100*(nominal-1),
                             apd_from_nominal_nm=r["apd_nominal_nm"],run=r["run"],reference=r["reference"],
                             reference_matched_profile=matched))
            break
    return rows


def matched_parameter_changes(direct):
    """Find solved one-parameter changes of the same Gaussian draw."""
    factors=("sigma_nm","xi_um","hurst","rho_lr","rho_line","rho_height","cutoff_um")
    profile=("cd_offset_nm","height_nm","swa_deg","vertical_shear_nm","footing_nm")
    contexts=defaultdict(dict)
    for r in direct:
        if r.get("kind")!="random" or r["group"] not in ("joint_train","blind_new_seed","blind_new_combo",
                                                         "stat_factor_cross","factor_refinement"):
            continue
        context=(r["setting"],r["Lx_um"],r["Lz_um"],r["seed"],r["master_min_um"],*(r[k] for k in profile))
        levels=tuple(r[k] for k in factors)
        contexts[context].setdefault(levels,r)
    changes=[]
    for (setting,lx,lz,seed,_,*_profile),domain in contexts.items():
        for (a,first),(b,second) in combinations(sorted(domain.items()),2):
            varied=[k for k,x,y in zip(factors,a,b) if x!=y]
            if len(varied)!=1:
                continue
            factor=varied[0]
            fixed={f"fixed_{k}":first[k] for k in factors if k!=factor}
            wq=second["wq"]-first["wq"]
            apd=None if None in (first["apd_nm"],second["apd_nm"]) else second["apd_nm"]-first["apd_nm"]
            changes.append(dict(factor=factor,low=first[factor],high=second[factor],seed=seed,
                                setting=setting,Lx_um=lx,Lz_um=lz,delta_wq_percent=100*wq,delta_apd_nm=apd,
                                passes_same_setting=in_budget(wq,apd),run_low=first["run"],run_high=second["run"],**fixed))
    return changes


def interaction_rows(direct):
    source=defaultdict(dict)
    profiles=defaultdict(dict)
    for row in direct:
        if row["group"] in ("stat_factor_cross","factor_refinement"):
            source[row["setting"]][row["case"]]=row
        if row["group"] in ("profile_cross","profile_refinement"):
            profiles[row["setting"]][row["case"]]=row
    rows=[]

    def record(cases,left,right,seed,setting,metrics):
        if any(r is None for r in cases):
            return
        for index,metric in enumerate(metrics):
            if any(r[metric] is None for r in cases):
                continue
            origin,a,b,joint=(r[metric] for r in cases)
            da,db=a-origin,b-origin
            effect=joint-a-b+origin
            units,scale=("percent of smooth",100.) if index==0 else ("nm",1.)
            rows.append(dict(factor_a=left,factor_b=right,seed=seed,kpi=metric,
                             unit=units,alone_a=scale*da,alone_b=scale*db,
                             combined=scale*(joint-origin),interaction=scale*effect,
                             interaction_above_both_main=abs(effect)>max(abs(da),abs(db)),
                             setting=setting,max_fidelity="matched RCWA numerical setting"))

    for id,catalog in source.items():
        for seed in FIT_SEEDS:
            for left,right in FACTOR_PAIRS:
                base=f"factor_{left}_{right}"
                cases=[catalog.get(f"{base}_{tag}_s{seed}") for tag in ("00","10","01","11")]
                record(cases,left,right,seed,id,("wq","apd_nm"))
    for id,catalog in profiles.items():
        for factor in PROFILES:
            for seed in FIT_SEEDS[:2]:
                cases=[catalog.get(f"profile_{factor}_{p}_r{r}_s{seed}") for p,r in ((0,5),(1,5),(0,20),(1,20))]
                record(cases,factor,"sigma_nm",seed,id,("wq_nominal","apd_nominal_nm"))
    return rows


def interaction_ensembles(rows):
    groups=defaultdict(dict)
    for r in rows:
        key=(r["factor_a"],r["factor_b"],r["kpi"],r["setting"],r["unit"])
        groups[key][r["seed"]]=r
    summary=[]
    for (a,b,kpi,setting,unit),group in sorted(groups.items()):
        values=np.array([r["interaction"] for r in group.values()])
        mean=float(np.mean(values))
        sd=float(np.std(values,ddof=1)) if len(values)>1 else None
        ci=t.ppf(.975,len(values)-1)*sd/np.sqrt(len(values)) if sd is not None else None
        limit=100*WQ_LIMIT if kpi.startswith("wq") else APD_LIMIT
        summary.append(dict(factor_a=a,factor_b=b,kpi=kpi,setting=setting,unit=unit,seeds=len(values),
                            signed_mean=mean,sd=sd,rms=float(np.sqrt(np.mean(values**2))),
                            mean_abs=float(np.mean(abs(values))),max_abs=float(np.max(abs(values))),
                            mean_95ci_low=mean-ci if ci is not None else None,
                            mean_95ci_high=mean+ci if ci is not None else None,
                            exceeds_kpi_budget=sum(abs(v)>limit for v in values),
                            exceeds_both_main_effects=sum(r["interaction_above_both_main"] for r in group.values())))
    return summary


def interaction_convergence(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[r["factor_a"],r["factor_b"],r["kpi"],r["seed"]].append(r)
    output=[]
    for (a,b,kpi,seed),comparisons in sorted(groups.items()):
        limit=100*NUM_WQ if kpi.startswith("wq") else NUM_APD
        for first,second in combinations(comparisons,2):
            start,end=first["setting"],second["setting"]
            fields_a=start.split("_")
            fields_b=end.split("_")
            if fields_a[1:]!=fields_b[1:]:
                continue
            modal_a=fields_a[0].split("x")
            modal_b=fields_b[0].split("x")
            dims=[column for column,x,y in zip(("ku","kv"),modal_a,modal_b) if x!=y]
            if len(dims)!=1:
                continue
            delta=second["interaction"]-first["interaction"]
            output.append(dict(factor_a=a,factor_b=b,kpi=kpi,seed=seed,unit=first["unit"],
                               first=start,second=end,axis=dims[0],
                               contrast_first=first["interaction"],contrast_second=second["interaction"],
                               delta=delta,stable_under_axis=abs(delta)<=limit))
    return output


def deterministic_interactions(model):
    angle_count=len_categories(model.angles)
    ku=model.meta.get("ku_full",10)
    solo={j.mark.name:j for j in kernel_jobs(angle_count,ku) if j.group=="mode_bank"}
    pairs={j.mark.name:j for j in pair_jobs(angle_count,ku)}

    def aligned(job):
        return align_spectrum(native_fields(job.folder),6.4,8.,model.l,model.n,model.angles)

    op=model.operator(2)
    reference=op(model.base)
    rows=[]
    for i,n,j,m,angle in pair_catalog():
        labels=(f"solo_{mtag(i)}_n{n}_a5",f"solo_{mtag(j)}_n{m}_a5",
                pair_tag(i,n,j,m,angle)+"_pp")
        jobs=(solo.get(labels[0]),solo.get(labels[1]),pairs.get(labels[2]))
        if any(job is None or not (job.folder/"done.json").exists() for job in jobs):
            continue
        a,b,ab=map(aligned,jobs)
        b*=np.exp(1j*angle*model.n/m)[None,None,:,None]
        trio=[evaluate(op(x),reference) for x in (a,b,ab)]
        additive=add_scan(op(a),op(b),reference)
        no_cross=evaluate(op(a+b-model.base),reference)
        add=evaluate(additive,reference)
        for key,base,scale in (("wq",1.,100.),("apd_nm",0.,1.)):
            ai,bi,both=(row[key] for row in trio)
            if None in (ai,bi,both,no_cross[key],add[key]):
                continue
            ia,ib,joint=(scale*(v-base) for v in (ai,bi,both))
            excess=joint-ia-ib
            rows.append(dict(mode_a=mtag(i),mode_b=mtag(j),n_a=n,n_b=m,phase_deg=angle*180/np.pi,
                             kpi=key,alone_a=ia,alone_b=ib,combined=joint,interaction=excess,
                             interaction_above_both_main=abs(excess)>max(abs(ia),abs(ib)),
                             excess_after_coherent_optics=scale*(both-no_cross[key]),
                             sum_of_isolated_error=scale*(both-add[key]),
                             native_joint=str(jobs[2].folder.relative_to(ROOT))))
    return rows


def modal_kpi_coefficients(model):
    """Estimate odd and even KPI responses per 5 nm single-edge peak."""
    operator=model.operator(2)
    base=operator(model.base)
    flat=evaluate(base,base)
    amplitude=float(model.meta["amplitude_nm"])
    rows=[]
    for key in sorted(model.linear):
        measured=[]
        for phase in (1.,-1j):
            criteria=[]
            for sign in (1.,-1.):
                odd,even,_=model.single(key,sign*amplitude*phase)
                metrics=evaluate(operator(model.base+odd+even),base)
                criteria.append((100*(metrics["wq"]-flat["wq"]),metrics["apd_nm"]-flat["apd_nm"]))
            measured.append(criteria)
        values=np.asarray(measured,dtype=float)
        gradient=(values[:,0]-values[:,1])/(2*amplitude)
        curvature=np.mean(values,axis=(0,1))/(amplitude**2)
        c,p,n=key
        rows.append(dict(mode=mtag((c,p)),frequency_index=n,period_um=8/n,peak_nm=amplitude,
                         wq_gradient_cos_percent_per_nm=gradient[0,0],
                         wq_gradient_sin_percent_per_nm=gradient[1,0],
                         wq_even_percent_per_nm2=curvature[0],
                         apd_gradient_cos_nm_per_nm=gradient[0,1],
                         apd_gradient_sin_nm_per_nm=gradient[1,1],
                         apd_even_nm_per_nm2=curvature[1]))
    return rows


def psd_forecast(coefficient_rows):
    """Propagate the target PSD through responses at the nominal section."""
    n=np.arange(1,161)
    spectra={}
    named=("wq_even_percent_per_nm2","apd_gradient_cos_nm_per_nm",
           "apd_gradient_sin_nm_per_nm","apd_even_nm_per_nm2")
    for mode in MODES:
        data=[r for r in coefficient_rows if r["mode"]==mtag(mode)]
        if not data or data[-1]["frequency_index"]<n[-1]:
            return []
        x=[r["frequency_index"] for r in data]
        spectra[mode]=np.stack([np.interp(n,x,[r[k] for r in data]) for k in named])
    rows=[]
    for i,(sigma,xi,hurst,rho_lr) in enumerate(BAND_SCENARIOS):
        q=2*np.pi*n/8.
        psd=(1+(q*xi)**2)**(-hurst-.5)
        energy=2*sigma**2*psd/psd.sum()
        effects=np.zeros((3,len(n)))
        for (component,_),transfer in spectra.items():
            # The four center/width and line-symmetry components are independent
            # at rho_line=0 under this stationary generator.
            fraction=(1+rho_lr*(1-2*component))/4
            wq,apd_c,apd_s,apd_even=transfer
            effects+=fraction*energy*np.stack((wq,.5*(apd_c**2+apd_s**2),apd_even))
        for cutoff in (.05,.1,.2,.5,1.,2.):
            keep=n<=round(8/cutoff)
            wq,apd_var,apd_bias=effects[:,keep].sum(axis=1)
            tail_wq,tail_var,tail_bias=effects[:,~keep].sum(axis=1)
            rows.append(dict(scenario=f"band_{i}",sigma_nm=sigma,xi_um=xi,hurst=hurst,rho_lr=rho_lr,
                             retained_period_um=cutoff,kept_wq_second_order_mean_percent=wq,
                             kept_apd_first_order_rms_nm=np.sqrt(apd_var),kept_apd_second_order_mean_nm=apd_bias,
                             omitted_wq_second_order_mean_percent=tail_wq,
                             omitted_apd_first_order_rms_nm=np.sqrt(tail_var),
                             omitted_apd_second_order_mean_nm=tail_bias))
    return rows


def add_scan(first,second,base):
    return (base[0],*(first[i]+second[i]-base[i] for i in (1,2,3)))


def original_fine_transfer(results,model):
    """Keep the frozen perturbation and use the independently higher-order smooth field."""
    variants=("linear","self_second","mixed_second","calibrated")
    if "joint_calibration" not in model.meta:
        variants=variants[:-1]
    rows=[]
    for r in results.receipts:
        if r["case"]!="chart_s1427" or r["settings_key"] not in ("k10x12_d2x5_a3_h4","k10x12_d2x5_a5_h4"):
            continue
        reference=results.reference(r)
        if reference is None:
            continue
        mark=results.geometry(r)
        _,_,fields,_=model.predict(mark,variants)
        truth=native_fields(r["folder"])
        incoming=truth.incoming
        args=(2*PITCH,mark.length,model.l,model.n,incoming)
        full=align_spectrum(truth,*args)
        base=align_spectrum(native_fields(reference["folder"]),*args)
        operator=ScanOperator(*args)
        standard=operator(base)
        observed=operator(full)
        observed_kpi=evaluate(observed,standard)
        perturbations={"smooth":np.zeros_like(full)}
        for name,fields_in in fields.items():
            perturbation=fields_in-model.base
            if np.array_equal(incoming,model.angles):
                perturbations[name]=perturbation
                continue
            source=Spectrum(model.l,model.n,2*PITCH,mark.length,model.angles,perturbation)
            ax,az,values=source_grid(source)
            interpolate=RegularGridInterpolator((ax,az),values,bounds_error=False,fill_value=None)
            perturbations[name]=interpolate(incoming)
        for name,perturbation in perturbations.items():
            estimated=operator(base+perturbation)
            predicted=evaluate(estimated,standard)
            wave=comparison(estimated,observed)
            dwq=predicted["wq"]-observed_kpi["wq"]
            dapd=optional_difference(predicted["apd_nm"],observed_kpi["apd_nm"])
            rows.append(dict(case=r["case"],setting=r["settings_key"],model=name,
                             native_wq=observed_kpi["wq"],native_apd_nm=observed_kpi["apd_nm"],
                             response_wq=predicted["wq"],response_apd_nm=predicted["apd_nm"],
                             wq_error_percent=100*dwq,apd_error_nm=dapd,
                             sum_nrmse=wave[0],diff_nrmse=wave[1],
                             high_setting_reference=reference["id"],
                             same_angle_nodes=np.array_equal(incoming,model.angles)))
    return rows


def score(results,model,selected_runs=None):
    table=[]
    curves_out={}
    contexts={}
    library_digest=model.meta.get("digest")
    for r in results.receipts:
        name=r["case"]
        is_chart=(r["group"] in ("chart_nested","chart_library_check") and
                  name in ("chart_s1427","chart_s1427_cut0.5","chart_s1427_cut1"))
        selected=(is_chart or r["group"] in ("legacy_reproduction","joint_train","blind_new_seed",
                                               "blind_new_combo","profile_cross"))
        if selected_runs is not None:
            selected=r["id"] in selected_runs
        if not selected or r["setting"]["angles"]!=len_categories(model.angles):
            continue
        info=results.kpi(r)
        if info is None or results.geometry(r).length!=8:
            continue
        own,total,ref_receipt=info
        mark=results.geometry(r)
        local_ref,predicted,spectra,domain=model.predict(mark)
        target=results.read(r)
        matched=results.read(ref_receipt)
        reference_nrmse,reference_diff= comparison(local_ref,matched)
        predicted={"smooth":local_ref,**predicted}
        references={name:local_ref for name in predicted}
        domains={name:domain for name in predicted}
        predicted_fields={"smooth":model.base,**spectra}
        predicted_bases={name:model.base for name in predicted_fields}
        true_field,true_base=(align_spectrum(native_fields(a["folder"]),6.4,8.,model.l,model.n,model.angles)
                              for a in (r,ref_receipt))
        perturbation=true_field-true_base
        field_norm=np.linalg.norm(true_field)
        perturbation_norm=np.linalg.norm(perturbation)
        context=available_model(mark,OUTPUT/"conditional")
        if context is not None and model.meta.get("ku_full",10)==10:
            if context not in contexts:
                contexts[context]=open_response(context)
            conditional=contexts[context]
            if not conditional.meta["partial"]:
                ctx,ctx_ports,ctx_fields,ctx_domain=conditional.predict(mark,("self_second",))
                predicted["conditioned_second"]=ctx_ports["self_second"]
                references["conditioned_second"]=ctx
                domains["conditioned_second"]=ctx_domain
                predicted_fields["conditioned_second"]=ctx_fields["self_second"]
                predicted_bases["conditioned_second"]=conditional.base
        is_blind=name.startswith("blind_")
        for kind,port in predicted.items():
            metrics=evaluate(port,references[kind],mark.rigid_nm)
            e1,e2=comparison(port,target)
            wc=metrics["wq"]-own["wq"]
            ac=None if None in (metrics["apd_nm"],own["apd_nm"]) else metrics["apd_nm"]-own["apd_nm"]
            coverage=domains[kind]
            abs_field_error=relative_field_error=relative_perturbation_error=None
            if kind in predicted_fields:
                prediction=predicted_fields[kind]
                delta=prediction-true_field
                absolute=np.linalg.norm(delta)
                abs_field_error=float(absolute/np.sqrt(delta.size))
                relative_field_error=float(absolute/max(field_norm,1e-30))
                if perturbation_norm>1e-12:
                    roughness_error=(prediction-predicted_bases[kind])-perturbation
                    relative_perturbation_error=float(np.linalg.norm(roughness_error)/perturbation_norm)
            table.append(dict(case=name,group=r["group"],native_setting=r["settings_key"],model=kind,
                              seed=mark.info.get("seed"),sigma_nm=mark.info.get("sigma_nm"),
                              xi_um=mark.info.get("xi_um"),hurst=mark.info.get("hurst"),
                              rho_lr=mark.info.get("rho_lr"),rho_line=mark.info.get("rho_line"),
                              rho_height=mark.info.get("rho_height"),retained_min_um=mark.cutoff,
                              direct_wq=own["wq"],pred_wq=metrics["wq"],error_wq_percent=100*wc,
                              direct_apd_nm=own["apd_nm"],pred_apd_nm=metrics["apd_nm"],error_apd_nm=ac,
                              sum_nrmse=e1,diff_nrmse=e2,reference_basis_sum_nrmse=reference_nrmse,
                              rms_complex_field_error=abs_field_error,
                              relative_complex_field_error=relative_field_error,
                              relative_roughness_field_error=relative_perturbation_error,
                              pass_kpi=in_budget(wc,ac),pass_scan=max(e1,e2)<=TRACE_LIMIT,
                              supported_terms=coverage["terms"]-len(coverage["unsupported"]),
                              missing_terms=len(coverage["unsupported"]),height_varying=coverage["height_dependent"],
                              conditioned_height=coverage["vertical_condition"],
                              applied_cross_terms=coverage["quadratic_pairs_used"],
                              amplitude_adjusted_modes=coverage["amplitude_adjusted_modes"],
                              amplitude_caps=coverage["amplitude_caps"],
                              library_hash=library_digest,is_blind=is_blind))
        if is_chart or is_blind:
            key=name+"_"+r["settings_key"]
            cur={"positions_um":target[0],"sum_direct":target[1],"diff_direct":target[2],
                 "sum_ideal_for_direct":matched[1],"diff_ideal_for_direct":matched[2]}
            for kind,port in predicted.items():
                cur["sum_"+kind]=port[1]
                cur["diff_"+kind]=port[2]
            curves_out[key]=cur
    return table,curves_out


def summarize_holdouts(rows):
    groups=defaultdict(list)
    for row in rows:
        if row["is_blind"]:
            for key in ("all",row["group"]):
                groups[key,row["model"]].append(row)
    output=[]
    for (key,model),observed in sorted(groups.items()):
        wq=np.asarray([row["error_wq_percent"] for row in observed])
        apd=np.asarray([row["error_apd_nm"] for row in observed if row["error_apd_nm"] is not None])
        output.append(dict(group=key,model=model,count=len(observed),valid_apd=len(apd),
                           pass_both_kpis=sum(row["pass_kpi"] for row in observed),
                           pass_all=sum(row["pass_kpi"] and row["pass_scan"] for row in observed),
                           median_wq_error_percent=float(np.median(abs(wq))),
                           max_wq_error_percent=float(np.max(abs(wq))),
                           median_apd_error_nm=float(np.median(abs(apd))) if len(apd) else None,
                           max_apd_error_nm=float(np.max(abs(apd))) if len(apd) else None,
                           mean_apd_bias_nm=float(np.mean(apd)) if len(apd) else None,
                           median_sum_nrmse=float(np.median([row["sum_nrmse"] for row in observed]))))
    return output


def corrected_bands(results,model,band_records):
    by_id={r["id"]:r for r in results.receipts}
    rows=[]
    for pair in band_records:
        if pair["angle_count"]!=len_categories(model.angles):
            continue
        full,filtered=(by_id[pair[key]] for key in ("full_run","filtered_run"))
        mark_f,mark_s=map(results.geometry,(full,filtered))
        if mark_f.length!=8. or mark_f.vertically_varying:
            continue
        variants=("linear","self_second","mixed_second")
        _,pred_f,_,shape_f=model.predict(mark_f,variants)
        _,pred_s,_,shape_s=model.predict(mark_s,variants)
        if shape_f["unsupported"] or shape_s["unsupported"]:
            continue
        observed=results.read(filtered)
        direct=results.read(full)
        actual,_,baseline=results.kpi(full)
        reference=results.read(baseline)
        raw_residual=comparison(observed,direct)
        raw_pass=pair["passes_budget_at_this_setting"] and max(raw_residual)<=TRACE_LIMIT
        record={**pair,"raw_sum_nrmse":raw_residual[0],"raw_diff_nrmse":raw_residual[1],
                "uncorrected_pass_at_same_settings":raw_pass}
        for kind in variants:
            f,s=(v[kind] for v in (pred_f,pred_s))
            corrected=(observed[0],*(observed[i]+f[i]-s[i] for i in (1,2,3)))
            estimated=evaluate(corrected,reference)
            if None in (estimated["apd_nm"],actual["apd_nm"]):
                continue
            wq=estimated["wq"]-actual["wq"]
            apd=estimated["apd_nm"]-actual["apd_nm"]
            residual=comparison(corrected,direct)
            prefix="" if kind=="mixed_second" else kind+"_"
            record.update({prefix+"correction_error_wq_percent":100*wq,
                           prefix+"correction_error_apd_nm":apd,prefix+"corrected_sum_nrmse":residual[0],
                           prefix+"corrected_diff_nrmse":residual[1],
                           prefix+"corrected_pass_at_same_settings":in_budget(wq,apd) and max(residual)<=TRACE_LIMIT})
        if "corrected_pass_at_same_settings" in record:
            rows.append(record)
    return rows


def len_categories(angles):
    return int(round(len(angles)**.5))


def figures(direct,scores,waves,interactions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"]="DejaVu Sans"
    if waves:
        chart_names=[k for k in waves if k.startswith("chart_s1427_k")]
        pick=next((n for n in chart_names if "k5x8" in n),chart_names[0] if chart_names else None)
        if pick:
            w=waves[pick]
            setting=pick.split("chart_s1427_",1)[-1]
            native=next(r for r in direct if r["case"]=="chart_s1427" and r["setting"]==setting)
            nx,nz=2*native["ku"]+1,2*native["kv"]+1
            nodes=native["angles"]
            x=w["positions_um"]
            fig,axes=plt.subplots(2,2,figsize=(12,6),layout="constrained",sharex=True)
            colors={"additive":"#a96b2a","linear":"#738194","self_second":"#855190",
                    "mixed_second":"#137c93","calibrated":"#258045","amplitude_response":"#8f4160"}
            labels={"additive":"Sum of isolated scans","linear":"Linear field","self_second":"Self quadratic",
                    "mixed_second":"With mixed quadratic","calibrated":"Jointly calibrated",
                    "amplitude_response":"Amplitude-dependent"}
            for col,port in enumerate(("sum","diff")):
                raw=w[f"{port}_direct"]
                axes[0,col].plot(x,raw,label="Direct RCWA",lw=2.2,color="#252d35")
                for kind,label in labels.items():
                    key=f"{port}_{kind}"
                    if key not in w:
                        continue
                    y=w[key]
                    axes[0,col].plot(x,y,label=label,lw=1.,alpha=.9,color=colors[kind])
                    axes[1,col].plot(x,1e4*(y-raw),color=colors[kind],lw=1.)
                axes[0,col].set_title(port.upper())
                axes[1,col].axhline(0,c=".6",lw=.7)
                axes[1,col].set_xlabel("Stage position (µm)")
            fig.suptitle(f"Original 80-mode edge · {nx} × {nz} field harmonics, {nodes} × {nodes} angles")
            axes[0,0].set_ylabel("Signal / incident power")
            axes[1,0].set_ylabel("Prediction residual (×10⁻⁴)")
            axes[0,0].legend(ncol=2,fontsize=8)
            fig.savefig(OUTPUT/"original_edge_prediction.png",dpi=165)
            plt.close(fig)
            labels=["positions_um",*(key for key in w if key!="positions_um")]
            np.savetxt(OUTPUT/"original_scans.csv",np.column_stack([w[key] for key in labels]),
                       delimiter=",",header=",".join(labels),comments="")
    blind=[r for r in scores if r["is_blind"]]
    if blind:
        models=list(dict.fromkeys(row["model"] for row in blind))
        fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout="constrained")
        for ax,metric,budget,title in ((axes[0],"error_wq_percent",.5,"WQ error (%)"),
                                       (axes[1],"error_apd_nm",1.,"APD error (nm)")):
            vals=[[abs(row[metric]) for row in blind if row["model"]==m and row[metric] is not None] for m in models]
            ax.boxplot(vals,tick_labels=models)
            ax.axhline(budget,c="#b24e31",ls="--",lw=1.)
            ax.set_ylabel(title)
            ax.tick_params(axis="x",rotation=27)
        fig.savefig(OUTPUT/"heldout_errors.png",dpi=165)
        plt.close(fig)
        candidates=[(k,value) for k,value in sorted(waves.items()) if k.startswith("blind_")]
        fig,axes=plt.subplots(4,4,figsize=(14,10),layout="constrained",sharex=True)
        for ax in axes.ravel():
            ax.set_visible(False)
        for idx,(ax,(name,signal)) in enumerate(zip(axes.ravel(),candidates)):
            ax.set_visible(True)
            x=signal["positions_um"]
            y=signal["sum_direct"]
            ax.plot(x,y,color="#404954",lw=1.7,label="Direct RCWA" if idx==0 else None)
            ax.plot(x,signal["sum_smooth"],color="#b3b2af",lw=.8,label="Smooth" if idx==0 else None)
            ax.plot(x,signal.get("sum_calibrated",signal["sum_mixed_second"]),color="#137c93",lw=1.,
                    label="Response prediction" if idx==0 else None)
            tokens=name.split("_")
            family="New parameter mix" if tokens[2]=="combo" else "New seed"
            text=f"{family} {int(tokens[3])+1} · {tokens[4][1:]}"
            ax.set_title(text,fontsize=9)
        if candidates:
            axes[0,0].legend(fontsize=7)
        fig.supxlabel("Stage position (µm)")
        fig.supylabel("SUM / incident power")
        fig.savefig(OUTPUT/"heldout_scans.png",dpi=150)
        plt.close(fig)
    amplitude=[r for r in direct if r["group"] in ("amplitude_curve","mode_bank") and r["angles"]==3
               and r["ku"]==5]
    if amplitude:
        fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout="constrained")
        for ax,kind,title in zip(axes,("cp","wp"),("Center motion","Width modulation")):
            for n in (1,2,4,8,16):
                prefix=f"solo_{kind}_n{n}_a"
                curve=[(float(r["case"].split("_a")[1]),(r["wq"]-1)*100)
                       for r in amplitude if r["case"].startswith(prefix)]
                if curve:
                    ax.plot(*zip(*sorted([(0.,0.),*curve])),marker="o",label=f"n={n}")
            ax.set_title(title)
            ax.set_xlabel("Single-edge cosine amplitude (nm)")
            ax.legend()
        axes[0].set_ylabel("WQ change (% of smooth)")
        fig.savefig(OUTPUT/"single_mode_response.png",dpi=155)
        plt.close(fig)
    if interactions:
        fig,axes=plt.subplots(1,2,figsize=(12,6),layout="constrained")
        for ax,keys,title in ((axes[0],("wq","wq_nominal"),"WQ"),
                               (axes[1],("apd_nm","apd_nominal_nm"),"APD")):
            grouped=defaultdict(list)
            for row in interactions:
                if row["kpi"] in keys and row["setting"]=="k10x8_d4x20_a3_h4":
                    grouped[row["factor_a"],row["factor_b"]].append(abs(row["interaction"]))
            order=sorted(grouped,key=lambda k:np.mean(grouped[k]))
            ax.barh([" × ".join(k) for k in order],[np.mean(grouped[k]) for k in order],color="#2a8297")
            ax.set_xlabel("Mean |four-corner contrast| ("+("% of smooth" if title=="WQ" else "nm")+")")
        fig.savefig(OUTPUT/"parameter_interactions.png",dpi=155)
        plt.close(fig)


def band_figure(bands):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    screening=defaultdict(list)
    for row in bands:
        parts=row["geometry"].split("_")
        if parts[0]=="band" and row["setting"]=="k10x8_d4x3.125_a3_h4":
            screening[int(parts[1]),row["filtered_um"]].append(row)
    if not screening:
        return
    plots=(
        ("Absolute WQ change (%)", "delta_wq_percent", WQ_LIMIT*100, .015),
        ("Absolute APD change (nm)", "delta_apd_nm", APD_LIMIT, .05),
        ("Larger port NRMSE (%)", "trace", TRACE_LIMIT*100, .05),
    )
    colors=("#236990","#bc4d39","#487a4c","#795497")
    styles=("o","s","D","^")
    fig,axes=plt.subplots(1,3,figsize=(12.4,4.7),layout="constrained")
    for ax,(label,metric,limit,threshold) in zip(axes,plots):
        for i,(sigma,xi,hurst,rho) in enumerate(BAND_SCENARIOS):
            data=[]
            for (series,period),rows in sorted(screening.items()):
                if series!=i:
                    continue
                values=([100*max(r["sum_nrmse"],r["diff_nrmse"]) for r in rows] if metric=="trace"
                        else [abs(r[metric]) for r in rows if r[metric] is not None])
                if values:
                    data.append((period,max(values)))
                    ax.scatter([period]*len(values),values,marker=styles[i],color=colors[i],alpha=.21,s=17)
            if data:
                periods,maximum=zip(*data)
                ax.plot(periods,maximum,marker=styles[i],color=colors[i],lw=1.5,ms=4.5,
                        label=f"{sigma:g} nm  · ξ {xi:g} μm  · H {hurst:g}  · ρ {rho:g}")
        ax.axhline(limit,color="#333f41",linestyle="--",lw=1,label="Error budget")
        ax.set_xscale("log")
        ax.set_xticks([.1,.2,.5,1,2],labels=["0.1","0.2","0.5","1","2"])
        ax.set_yscale("symlog",linthresh=threshold)
        ax.margins(y=.12)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Shortest retained period (μm)")
        ax.set_ylabel(label)
        ax.grid(True,linestyle=":",alpha=.35)
    axes[0].legend(loc="best",fontsize=7.2,frameon=True)
    fig.suptitle("Paired direct RCWA against each 0.05 μm master | faint points are individual seeds",fontsize=12)
    fig.savefig(OUTPUT/"bandwidth_native_response.png",dpi=165)
    plt.close(fig)


def strong_width_figure():
    path=OUTPUT/"strong_width_high_scans.npz"
    if not path.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(path) as scan:
        x=scan["positions_um"]
        fig,axes=plt.subplots(2,2,figsize=(10.3,6.),sharex=True,layout="constrained")
        for column,port in enumerate(("sum","diff")):
            exact=scan[f"{port}_direct"]
            scale=max(np.ptp(exact),1e-12)
            axes[0,column].plot(x,exact,color="#20292d",label="Direct RCWA",lw=1.9)
            for label,name,color,style in (("Linear field","linear","#ad503c","--"),
                                            ("Including modal quadratic response","self_second","#2d7894",":")):
                predicted=scan[f"{port}_{name}"]
                axes[0,column].plot(x,predicted,color=color,label=label,linestyle=style,lw=1.6)
                axes[1,column].plot(x,100*(predicted-exact)/scale,color=color,linestyle=style,lw=1.5)
            axes[0,column].set_title(port.upper())
            axes[1,column].axhline(0,color="#596368",lw=.65)
            axes[1,column].set_xlabel("Scan position (μm)")
        axes[0,0].set_ylabel("Normalized port signal")
        axes[1,0].set_ylabel("Residual (% of direct peak to peak)")
        for ax in axes.flat:
            ax.grid(True,linestyle=":",alpha=.35)
        axes[0,0].legend(loc="best",fontsize=8)
        fig.suptitle("Strong width-only holdout | matching native and response bases at $K_x=60$")
        fig.savefig(OUTPUT/"strong_width_high_comparison.png",dpi=170)
        plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",type=Path)
    p.add_argument("--blind",action="store_true")
    p.add_argument("--no-figures",action="store_true")
    args=p.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    model=open_response(args.model) if args.model else None
    if model:
        model.meta["digest"]=sha256(args.model.read_bytes()).hexdigest()
    if args.blind:
        if model is None or model.meta["partial"]:
            raise ValueError("Reveal requires the frozen, complete response library")
        waiting=[j.mark.name for j in blind_jobs() if not (j.folder/"done.json").is_file()]
        if waiting:
            raise ValueError(f"Blind direct references are incomplete ({len(waiting)} waiting)")
    results=Results(reveal=args.blind)
    physical=direct_rows(results)
    numerical=convergence_rows(physical)
    bands=band_pairs(physical,results)
    pair=interaction_rows(physical)
    tables={"direct_kpis":physical,"numerical_convergence":numerical,
            "band_pairs_native":bands,"band_seed_statistics":paired_statistics(bands),
            "band_pair_convergence":band_convergence(bands,physical),
            "profile_response_curves":profile_response_curves(physical),
            "matched_parameter_direct":matched_parameter_changes(physical),
            "factor_interactions":pair,"factor_seed_summary":interaction_ensembles(pair),
            "factor_interaction_convergence":interaction_convergence(pair)}
    waves={}
    scores=[]
    if model:
        scores,waves=score(results,model)
        tables["heldout_prediction" if args.blind else "development_prediction"]=scores
        if args.blind:
            tables["heldout_summary"]=summarize_holdouts(scores)
            height_cases={(r["case"],r["native_setting"]) for r in scores
                          if r["is_blind"] and r["height_varying"] and r["model"]=="conditioned_second"}
            height_models=("smooth","self_second","calibrated","conditioned_second")
            tables["height_conditioned_validation"]=[r for r in scores
                                                    if (r["case"],r["native_setting"]) in height_cases and r["model"] in height_models]
        tables["band_quadratic_correction"]=corrected_bands(results,model,bands)
        tables["original_fine_transfer"]=original_fine_transfer(results,model)
        tables["deterministic_interactions"]=deterministic_interactions(model)
        coefficients=modal_kpi_coefficients(model)
        tables["modal_kpi_frequency_response"]=coefficients
        tables["band_psd_prediction"]=psd_forecast(coefficients)
    print({k:save_csv(k,v) for k,v in tables.items()})
    if waves:
        for name,arrays in waves.items():
            label="heldout" if name.startswith("blind_") else "comparison"
            folder=OUTPUT/label
            folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/f"{name}.npz",**arrays)
    if not args.no_figures:
        figures(physical,scores,waves,pair)
        band_figure(bands)
        strong_width_figure()


if __name__=="__main__":
    main()

