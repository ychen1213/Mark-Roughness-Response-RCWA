"""Parameter response curves and same-master reduction tests."""

import argparse
import csv
from collections import defaultdict
from dataclasses import replace
from hashlib import sha256
from itertools import combinations, product
from pathlib import Path

import numpy as np
from scipy.stats import t

from design import CONTROL, LEVELS, domain_points
from rcwa_runner import Job, Setting, baseline, run_jobs
from readout import comparison, evaluate
from report_analysis import APD_LIMIT, NUM_APD, NUM_WQ, TRACE_LIMIT, WQ_LIMIT, Results, direct_rows, save_csv
from response_geometry import ROOT, random_mark
from response_model import OUTPUT, open_response

SEEDS=tuple(range(24319,24351))
REDUCTIONS=("center","width","rms","xi","hurst","rho_lr","rho_line","rho_height","band_1","band_2")
REF_LEVEL={"rms":("sigma_nm",0.),"xi":("xi_um",.2),
           "hurst":("hurst",.5),"rho_lr":("rho_lr",0.),"rho_line":("rho_line",0.),
           "rho_height":("rho_height",1.)}


def estimate(model,mark,variant):
    ref,x,_,info=model.predict(mark,(variant,))
    row=evaluate(x[variant],ref)
    row["missing_modes"]=len(info["unsupported"])
    return row


def reduce_geometry(mark,reduction):
    name=f"{mark.name}_{reduction}"
    if reduction in ("center","width"):
        a=mark.coeff.copy()
        a[:,:,0 if reduction=="center" else 1,:]=0
        return replace(mark,name=name,coeff=a)
    if reduction.startswith("band_"):
        cutoff=float(reduction.split("_")[1])
        return replace(mark,name=name,cutoff=max(mark.cutoff,cutoff))
    key,value=REF_LEVEL[reduction]
    opts={k:mark.info[k] for k in ("seed","sigma_nm","xi_um","hurst","rho_lr","rho_line","rho_height")}
    opts[key]=value
    return random_mark(name,**opts,cutoff=mark.cutoff,length=mark.length,stripes=mark.stripes,
                       profile=mark.profile,master_min=mark.info["master_min_um"])


def parameter_curves(model):
    rows=[]
    variant="calibrated" if "joint_calibration" in model.meta else "mixed_second"
    for factor,levels in LEVELS.items():
        if factor=="rho_height":
            continue
        if factor=="sigma_nm":
            levels=(0,*levels)
        for level in levels:
            values=[]
            for seed in SEEDS:
                args={**CONTROL,factor:level}
                g=random_mark(f"axis_{factor}_{level:g}_s{seed}",seed=seed,**args)
                values.append(estimate(model,g,variant))
            apds=np.array([r["apd_nm"] for r in values if r["apd_nm"] is not None])
            wq=np.array([r["wq"] for r in values])
            mean_wq=100*(float(wq.mean())-1)
            sd_wq=float(np.std(wq,ddof=1)*100)
            margin=t.ppf(.975,len(wq)-1)*sd_wq/np.sqrt(len(wq))
            rows.append(dict(factor=factor,level=level,seeds=len(values),valid_apd=len(apds),model=variant,
                             mean_wq_percent=mean_wq,sd_wq_percent=sd_wq,
                             mean_wq_95ci_low_percent=mean_wq-margin,mean_wq_95ci_high_percent=mean_wq+margin,
                             mean_apd_nm=float(apds.mean()) if len(apds) else None,
                             rms_apd_nm=float(np.sqrt(np.mean(apds*apds))) if len(apds) else None,
                             p90_abs_apd_nm=float(np.quantile(abs(apds),.9)) if len(apds) else None,
                             max_abs_apd_nm=float(np.max(abs(apds))) if len(apds) else None,
                             max_missing_modes=max(r["missing_modes"] for r in values)))
    return rows


def joint_surfaces(model):
    variant="calibrated" if "joint_calibration" in model.meta else "mixed_second"
    axes=[name for name in LEVELS if name!="rho_height"]
    rows=[]
    for a,b in combinations(axes,2):
        samples={}
        la,lb=LEVELS[a],LEVELS[b]
        for x,y in product(la,lb):
            args={**CONTROL,a:x,b:y}
            predicted=[estimate(model,random_mark(f"surface_{a}_{b}_s{seed}",seed=seed,**args),variant)
                       for seed in SEEDS[:16]]
            wq=np.array([r["wq"] for r in predicted])
            apd=np.array([np.nan if r["apd_nm"] is None else r["apd_nm"] for r in predicted])
            samples[x,y]=(wq,apd,max(r["missing_modes"] for r in predicted))
        x0,y0=la[0],lb[0]
        for (x,y),(wq,apd,missing) in samples.items():
            a_wq,a_apd,_=samples[x,y0]
            b_wq,b_apd,_=samples[x0,y]
            ref_wq,ref_apd,_=samples[x0,y0]
            iw=100*(wq-a_wq-b_wq+ref_wq)
            ia=apd-a_apd-b_apd+ref_apd
            ia=ia[np.isfinite(ia)]
            valid=apd[np.isfinite(apd)]
            rows.append(dict(factor_a=a,factor_b=b,level_a=x,level_b=y,
                             contrast_origin_a=x0,contrast_origin_b=y0,seeds=len(wq),valid_apd=len(valid),
                             mean_wq_percent=100*(wq.mean()-1),sd_wq_percent=wq.std(ddof=1)*100,
                             rms_apd_nm=np.sqrt(np.mean(valid**2)) if len(valid) else None,
                             mean_interaction_wq_percent=float(np.mean(iw)),
                             rms_interaction_wq_percent=float(np.sqrt(np.mean(iw**2))),
                             rms_interaction_apd_nm=float(np.sqrt(np.mean(ia**2))) if len(ia) else None,
                             missing_modes=missing,model=variant))
    return rows


def plots(surfaces):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    specs=(("sigma_nm","rho_lr"),("xi_um","hurst"),("rho_lr","rho_line"),("xi_um","cutoff"))
    display={"sigma_nm":"Single-edge RMS σ (nm)","xi_um":"Correlation length ξ (µm)",
             "hurst":"Hurst exponent H","rho_lr":"Left-right correlation",
             "rho_line":"Adjacent-stripe correlation","cutoff":"Minimum retained period (µm)"}
    fig,axes=plt.subplots(2,len(specs),figsize=(13,6.3),layout="constrained")
    for col,(factor_a,factor_b) in enumerate(specs):
        selected=[r for r in surfaces if (r["factor_a"],r["factor_b"])==(factor_a,factor_b)]
        xx=sorted({r["level_a"] for r in selected})
        yy=sorted({r["level_b"] for r in selected})
        for row,(k,label) in enumerate((("mean_wq_percent","Mean WQ change (%)"),("rms_apd_nm","RMS APD (nm)"))):
            vals=np.zeros((len(yy),len(xx)))
            for v in selected:
                vals[yy.index(v["level_b"]),xx.index(v["level_a"])]=v[k]
            ax=axes[row,col]
            image=ax.imshow(vals,aspect="auto",origin="lower",cmap="viridis")
            ax.set_xticks(range(len(xx)),[f"{x:g}" for x in xx],rotation=25)
            ax.set_yticks(range(len(yy)),[f"{y:g}" for y in yy])
            ax.set_xlabel(display[factor_a])
            ax.set_ylabel(display[factor_b])
            fig.colorbar(image,ax=ax,label=label,shrink=.76)
    fig.savefig(OUTPUT/"joint_parameter_response.png",dpi=145)
    plt.close(fig)


def risk_screen(model):
    variant="calibrated" if "joint_calibration" in model.meta else "mixed_second"
    cases=[]
    for i,p in enumerate(domain_points(32,20261125)):
        p.update(rho_height=1.,cutoff=.5)
        for seed in SEEDS[:4]:
            g=random_mark(f"omit_{i:02d}_s{seed}",seed=seed,**p)
            ref=estimate(model,g,variant)
            for reduction in REDUCTIONS:
                if reduction=="rho_height":
                    continue
                altered=reduce_geometry(g,reduction)
                if g.signature()==altered.signature():
                    continue
                out=estimate(model,altered,variant)
                if None in (ref["apd_nm"],out["apd_nm"]):
                    continue
                dw=out["wq"]-ref["wq"]
                da=out["apd_nm"]-ref["apd_nm"]
                cases.append((reduction,g,altered,dw,da,max(abs(dw)/WQ_LIMIT,abs(da)/APD_LIMIT)))
    return cases


def summarize_risks(cases,model_hash):
    groups=defaultdict(list)
    for reduction,base,altered,wq,apd,severity in cases:
        groups[reduction].append((base.name,100*abs(wq),abs(apd),severity))
    rows=[]
    for name,items in sorted(groups.items()):
        wq=np.array([v[1] for v in items])
        apd=np.array([v[2] for v in items])
        rows.append(dict(reduction=name,cases=len(items),
                         cases_over_either_budget=sum(v[3]>1. for v in items),
                         p90_abs_wq_percent=np.quantile(wq,.9),max_abs_wq_percent=max(wq),
                         max_wq_case=items[int(np.argmax(wq))][0],
                         p90_abs_apd_nm=np.quantile(apd,.9),max_abs_apd_nm=max(apd),
                         max_apd_case=items[int(np.argmax(apd))][0],
                         model_sha256=model_hash))
    return rows


def choose_jobs(cases):
    groups=defaultdict(list)
    for case in cases:
        groups[case[0]].append(case)
    selected=[]
    for k,items in groups.items():
        worst=max(items,key=lambda x:x[-1])
        near=min(items,key=lambda x:abs(x[-1]-1))
        for label,choice in (("max_risk",worst),("near_budget",near)):
            if label=="near_budget" and choice[1].name==worst[1].name:
                continue
            selected.append((label,*choice))
    return selected


def make_jobs(selected,angles,check):
    settings=[Setting(10,8,4,20,angles)]
    if check:
        settings.extend((Setting(20,8,4,20,angles),Setting(10,16,4,20,angles),
                         Setting(10,8,2,10,angles),Setting(10,8,4,20,5)))
    for setting in settings:
        for label,reduction,base,filtered,*_ in selected:
            yield Job(baseline(base),replace(setting,kv=1),"policy_reference")
            for g in (base,filtered):
                meta={**g.info,"policy_parent":base.name,"policy_reduction":reduction,"policy_rank":label}
                yield Job(replace(g,info=meta),setting,"policy_direct")


def direct_policy_pairs(direct):
    cases=defaultdict(dict)
    for row in direct:
        if "policy_parent" in row:
            cases[row["policy_parent"],row["setting"]][row["case"]]=row
    for (parent,_),rows in cases.items():
        if parent not in rows:
            continue
        full=rows[parent]
        for name,filtered in rows.items():
            if name!=parent and None not in (full["apd_nm"],filtered["apd_nm"]):
                yield full,filtered


def direct_pair_details(pairs,results):
    receipts={r["id"]:r for r in results.receipts}
    rows=[]
    for a,b in pairs:
        dwq=b["wq"]-a["wq"]
        dapd=b["apd_nm"]-a["apd_nm"]
        e1,e2=comparison(results.read(receipts[b["run"]]),results.read(receipts[a["run"]]))
        pass_kpis=abs(dwq)<=WQ_LIMIT and abs(dapd)<=APD_LIMIT
        pass_scan=max(e1,e2)<=TRACE_LIMIT
        rows.append(dict(base=a["case"],filtered=b["case"],setting=a["setting"],
                         reduction=b["case"][len(a["case"])+1:],
                         delta_wq_percent=100*dwq,delta_apd_nm=dapd,sum_nrmse=e1,diff_nrmse=e2,
                         pass_kpis=pass_kpis,pass_scan=pass_scan,pass_all=pass_kpis and pass_scan,
                         base_run=a["run"],filtered_run=b["run"]))
    return rows


def assess_screen(details):
    manifest=OUTPUT/"reduction_targets.csv"
    if not manifest.is_file():
        return []
    with manifest.open(encoding="utf-8-sig",newline="") as source:
        selected={(r["base"],r["filtered"]):r for r in csv.DictReader(source)}
    rows=[]
    for direct in details:
        key=(direct["base"],direct["filtered"])
        if key not in selected or direct["setting"]!="k10x8_d4x20_a3_h4":
            continue
        model=selected[key]
        prediction_wq=float(model["delta_wq_percent"])
        prediction_apd=float(model["delta_apd_nm"])
        model_pass=abs(prediction_wq)<=100*WQ_LIMIT and abs(prediction_apd)<=APD_LIMIT
        direct_pass=direct["pass_kpis"]
        rows.append(dict(**direct,selection=model["selection"],model_wq_delta_percent=prediction_wq,
                         model_apd_delta_nm=prediction_apd,model_wq_error_percent=prediction_wq-direct["delta_wq_percent"],
                         model_apd_error_nm=prediction_apd-direct["delta_apd_nm"],
                         model_pass_kpis=model_pass,correct_budget_class=model_pass==direct_pass,
                         false_exemption=model_pass and not direct_pass,
                         model_sha256=model["model_sha256"]))
    return rows


def certify(direct,details):
    traces={(r["base"],r["filtered"],r["setting"]):max(r["sum_nrmse"],r["diff_nrmse"]) for r in details}
    grouping=defaultdict(list)
    for full,filtered in direct_policy_pairs(direct):
        changes=np.array([filtered["wq"]-full["wq"],filtered["apd_nm"]-full["apd_nm"]])
        grouping[full["case"],filtered["case"]].append((changes,full,filtered))
    rows=[]
    for (parent,filtered),runs in grouping.items():
        dk=[p[0] for p in runs]
        waves=[traces[parent,filtered,full["setting"]] for _,full,_ in runs]
        core=next((v for v,p,q in runs if p["ku"]==10 and p["kv"]==8 and
                   p["du_nm"]==4 and p["angles"]==3),None)
        x=next((v for v,p,q in runs if p["ku"]==20 and p["kv"]==8),None)
        z=next((v for v,p,q in runs if p["ku"]==10 and p["kv"]==16),None)
        raster=next((v for v,p,q in runs if p["du_nm"]==2 and p["angles"]==3),None)
        angle=next((v for v,p,q in runs if p["angles"]==5),None)
        span=np.ptp(dk,axis=0)
        peak=np.max(np.abs(dk),axis=0)
        stable=all(a is not None for a in (core,x,z)) and all(span<=[NUM_WQ,NUM_APD])
        qualified=stable and raster is not None and angle is not None
        if stable and (np.any(peak-span>[WQ_LIMIT,APD_LIMIT]) or min(waves)>TRACE_LIMIT):
            verdict="retain_in_this_geometry" if qualified else "retain_on_verified_modal_axes"
        elif stable and np.all(peak+span<=[WQ_LIMIT,APD_LIMIT]) and max(waves)<=TRACE_LIMIT:
            verdict="within_budget_in_this_geometry" if qualified else "within_budget_on_verified_modal_axes"
        else:
            verdict="unresolved"
        rows.append(dict(base=parent,filtered=filtered,reduction=filtered[len(parent)+1:],
                         direct_settings=len(runs),max_abs_wq_percent=100*peak[0],max_abs_apd_nm=peak[1],
                         max_scan_nrmse=max(waves),
                         modal_wq_spread_percent=100*span[0],modal_apd_spread_nm=span[1],
                         modal_status=verdict,full_qualification=qualified))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",type=Path,default=OUTPUT/"response_model.npz")
    p.add_argument("--direct",action="store_true")
    p.add_argument("--refine",action="store_true")
    p.add_argument("--summarize",action="store_true")
    p.add_argument("--threads",type=int,default=5)
    args=p.parse_args()
    if args.summarize:
        results=Results()
        results.receipts=[r for r in results.receipts if r["group"]=="policy_direct"]
        rows=direct_rows(results,extra_groups={"policy_direct"})
        details=direct_pair_details(direct_policy_pairs(rows),results)
        print("Direct policy pairs:",save_csv("reduction_direct_checks",certify(rows,details)))
        print("Direct policy settings:",save_csv("reduction_pair_details",details))
        print("Screen classification checks:",save_csv("reduction_screen_validation",assess_screen(details)))
        return
    model=open_response(args.model)
    print("Parameter-axis response curves:",save_csv("parameter_response_curves",parameter_curves(model)),flush=True)
    surfaces=joint_surfaces(model)
    print("Joint parameter response grid:",save_csv("joint_parameter_response",surfaces),flush=True)
    plots(surfaces)
    risks=risk_screen(model)
    selection=choose_jobs(risks)
    manifest=[]
    digest=sha256(args.model.read_bytes()).hexdigest()
    print("Reduction screening statistics:",save_csv("reduction_model_screen",summarize_risks(risks,digest)),flush=True)
    for label,key,base,filtered,wq,apd,severity in selection:
        manifest.append(dict(selection=label,reduction=key,base=base.name,filtered=filtered.name,
                             delta_wq_percent=100*wq,delta_apd_nm=apd,normalized_error=severity,
                             model_sha256=digest))
    print("Reduction targets:",save_csv("reduction_targets",manifest),flush=True)
    if args.direct:
        seen=set()
        actual=[]
        for j in make_jobs(selection,model.meta["angles"],args.refine):
            if j.folder not in seen:
                actual.append(j)
                seen.add(j.folder)
        run_jobs(actual,"reductions",args.threads)


if __name__=="__main__":
    main()

