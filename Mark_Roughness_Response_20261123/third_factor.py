"""Confirm the strongest measured two-factor interactions under a third factor."""

import argparse
from collections import defaultdict
from itertools import product
from pathlib import Path
import json

import numpy as np
from scipy.stats import t

from design import CONTROL, FIT_SEEDS, PAIR_LEVELS, PROFILES, factor_jobs, low_setting
from rcwa_runner import Job, run_jobs
from report_analysis import Results, direct_rows, interaction_rows, save_csv
from response_geometry import ROOT, Profile, random_mark
from response_model import OUTPUT

LEVEL={**PAIR_LEVELS,**PROFILES}
OPTIONS=("sigma_nm","rho_lr","xi_um","cutoff","hurst","rho_line","rho_height")


def evaluate_candidates(rows):
    groups=defaultdict(list)
    for row in rows:
        if row["setting"]!="k10x8_d4x20_a3_h4":
            continue
        kpi=row["kpi"]
        budget=.5 if kpi.startswith("wq") else 1.
        groups[row["factor_a"],row["factor_b"],kpi].append(abs(row["interaction"])/budget)
    factors=defaultdict(list)
    for (a,b,k),change in groups.items():
        if len(change)>=2:
            factors[a,b].append(float(np.mean(change)))
    rank=sorted(factors,key=lambda k:(max(factors[k]),k),reverse=True)
    selected=[]
    used=set()
    for a,b in rank:
        c=next((c for c in OPTIONS if c not in (a,b) and frozenset((a,b,c)) not in used),None)
        if c is None:
            continue
        used.add(frozenset((a,b,c)))
        selected.append((a,b,c))
        if len(selected)==3:
            break
    return selected


def jobs_for(keys):
    for a,b,c in keys:
        for seed in FIT_SEEDS:
            for flag in product((0,1),repeat=3):
                tag="".join(map(str,flag))
                values={k:LEVEL[k][i] for k,i in zip((a,b,c),flag)}
                profile=Profile(**{k:v for k,v in values.items() if k in PROFILES})
                kwargs={**CONTROL,**{k:v for k,v in values.items() if k in CONTROL}}
                mark=random_mark(f"triple_{a}_{b}_{c}_{tag}_s{seed}",seed=seed,profile=profile,**kwargs)
                mark.info.update(triple_a=a,triple_b=b,triple_c=c,triple_corner=tag)
                yield Job(mark,low_setting(mark),"third_interaction")


def summarize():
    catalogue=Results()
    catalogue.receipts=[r for r in catalogue.receipts if r["group"]=="third_interaction"]
    runs=direct_rows(catalogue,extra_groups=("third_interaction",))
    groups=defaultdict(dict)
    corners=[]
    for row in runs:
        if row["group"]=="third_interaction":
            groups[row["triple_a"],row["triple_b"],row["triple_c"],row["seed"]][row["triple_corner"]]=row
            corners.append({key:row.get(key) for key in ("triple_a","triple_b","triple_c","triple_corner",
                              "seed","setting","wq_nominal","apd_nominal_nm","run","reference")})
    save_csv("third_factor_native_kpis",corners)
    scores=[]
    for (a,b,c,seed),group in groups.items():
        if len(group)!=8:
            continue
        for name,unit in (("wq_nominal",100.),("apd_nominal_nm",1.)):
            if any(v[name] is None for v in group.values()):
                continue
            cross=[]
            for last in ("0","1"):
                v=[group[i+last][name] for i in ("00","10","01","11")]
                cross.append(unit*(v[3]-v[2]-v[1]+v[0]))
            scores.append(dict(factor_a=a,factor_b=b,factor_c=c,seed=seed,kpi=name,
                               pair_at_c_low=cross[0],pair_at_c_high=cross[1],
                               third_difference=cross[1]-cross[0],setting=group["000"]["setting"]))
    ensembles=defaultdict(list)
    for row in scores:
        key=tuple(row[k] for k in ("factor_a","factor_b","factor_c","kpi","setting"))
        ensembles[key].append(row["third_difference"])
    statistics=[]
    for (a,b,c,kpi,setting),values in ensembles.items():
        mean=float(np.mean(values))
        sd=float(np.std(values,ddof=1)) if len(values)>1 else None
        width=t.ppf(.975,len(values)-1)*sd/np.sqrt(len(values)) if sd is not None else None
        statistics.append(dict(factor_a=a,factor_b=b,factor_c=c,kpi=kpi,setting=setting,seeds=len(values),
                               signed_mean=mean,rms=float(np.sqrt(np.mean(np.square(values)))),sd=sd,
                               min_signed=min(values),max_signed=max(values),
                               mean_95ci_low=mean-width if width is not None else None,
                               mean_95ci_high=mean+width if width is not None else None))
    save_csv("third_factor_seed_summary",statistics)
    return save_csv("third_factor_interactions",scores)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary",action="store_true")
    p.add_argument("--threads",type=int,default=4)
    args=p.parse_args()
    if args.summary:
        print("Third-factor contrasts:",summarize())
        return
    frozen=ROOT/"queue"/"third_factor_selection.json"
    if not frozen.exists():
        waiting=[j for j in factor_jobs() if not (j.folder/"done.json").exists()]
        if waiting:
            raise RuntimeError(f"Two-factor reference incomplete ({len(waiting)} waiting)")
        data=direct_rows(Results())
        matched=evaluate_candidates(interaction_rows(data))
        if len(matched)<3:
            raise RuntimeError("Three measured two-factor pairs are required")
        frozen.write_text(json.dumps(matched,indent=2),encoding="utf-8")
    keys=json.loads(frozen.read_text(encoding="utf-8"))
    waiting=[j for j in jobs_for(keys) if not (j.folder/"done.json").exists()]
    run_jobs(waiting,"third_factor",args.threads)
    print("Third-factor contrasts:",summarize())


if __name__=="__main__":
    main()

