"""Frozen design and restartable RCWA lanes for the roughness response study."""

import argparse
from dataclasses import asdict, replace
from itertools import combinations, product
import json
import math
from pathlib import Path
import re

import numpy as np
from scipy.stats import qmc

from rcwa_runner import Job, PREVIOUS, Setting, baseline, run_jobs
from response_geometry import ROOT, Profile, chart_mark, from_legacy, modal_mark, random_mark, smooth

# Modes denote center or halfwidth and common or differential across the two stripes.
MODES=((0,"+"),(0,"-"),(1,"+"),(1,"-"))
FIT_SEEDS=(1427,2441,3547,4639)
BLIND_SEEDS=(9101,9109,9127,9151,9209,9221,9239,9241)
LEVELS={
    "sigma_nm":(2,5,10,20), "xi_um":(.05,.2,.5,1.), "hurst":(.5,.65,.8),
    "rho_lr":(-1.,-.5,0.,.5,1.), "rho_line":(0.,.5,1.),
    "rho_height":(0.,.5,1.), "cutoff":(.5,1.,2.),
}
CONTROL=dict(sigma_nm=10,xi_um=.2,hurst=.5,rho_lr=0.,rho_line=0.,rho_height=1.,cutoff=.5)
# Two levels for matched combinations, even if a factor has a small marginal effect.
PAIR_LEVELS={
    "sigma_nm":(2,20), "xi_um":(.05,1.),"hurst":(.5,.8),"rho_lr":(-1,1),
    "rho_line":(0.,1.),"rho_height":(0.,1.),"cutoff":(.5,1.),
}
FACTOR_PAIRS=(("sigma_nm","xi_um"),("sigma_nm","hurst"),("sigma_nm","rho_lr"),
              ("xi_um","hurst"),("rho_lr","rho_line"),("rho_lr","rho_height"),
              ("cutoff","xi_um"),("cutoff","hurst"),("sigma_nm","rho_height"))
PROFILES={
    "cd_nm":(-40,40), "height_nm":(270,330), "swa_deg":(90,85),
    "shear_nm":(-25,25), "footing_nm":(0,30),
}
HIGH_N=(20,24,32,40,56,80,120,160)
MID_N=(4,8,16,2,1)
BAND_SCENARIOS=((5,.2,.5,.5),(20,.2,.5,-1),(20,.05,.5,0),(20,1.,.8,.5))


def low_setting(mark,angles=3):
    ku=5*mark.stripes
    kv=1 if mark.along_invariant else min(8,max(2,math.ceil(mark.length/mark.cutoff)))
    stacks = 4
    return Setting(ku,kv,4,20,angles,stacks)


def modal_setting(mark,angles=3,ku_full=10,max_full_kv=16):
    nshort=mark.active.shape[-1]-1
    divisor=mark.info["primitive_divisor"]
    modes=max(1,min(nshort+2,max_full_kv//divisor))
    du,dv=(4,20) if ku_full<20 else (2,10)
    return Setting(ku_full*mark.stripes//2,modes,du,min(dv,1000*mark.cutoff/16),angles,4)


def mtag(mode):
    return ("c" if mode[0]==0 else "w")+("p" if mode[1]=="+" else "m")


def solo(mode,n,amplitude=5.,phase=0.,*,height=1,name=None,profile=Profile(),compact=True):
    name=name or f"solo_{mtag(mode)}_n{n}_a{amplitude:g}"
    return modal_mark(name,[(*mode,n,amplitude,phase,height)],profile=profile,compact=compact)


def kernel_jobs(angles=3,ku=10):
    main=[]
    for n in (*MID_N,*(i for i in range(1,17) if i not in MID_N),*HIGH_N):
        for m in MODES:
            g=solo(m,n)
            main.append(Job(g,modal_setting(g,angles,ku),"mode_bank"))
    for n,a,mode in product(MID_N,(2.5,10.,20.),((0,"+"),(1,"+"))):
        g=solo(mode,n,a)
        main.append(Job(g,modal_setting(g,angles,ku),"amplitude_curve"))
    for n,mode,tag,phase in [(4,(0,"-"),"minus",np.pi),(8,(1,"+"),"sine",np.pi/2),
                             (1,(0,"+"),"minus",np.pi),(16,(1,"-"),"sine",np.pi/2)]:
        g=solo(mode,n,phase=phase,name=f"control_{mtag(mode)}_n{n}_{tag}")
        main.append(Job(g,modal_setting(g,angles,ku),"phase_control"))
    du,dv=(4,20) if ku<20 else (2,10)
    return [Job(smooth(f"reference_u{i}",8.,i),Setting(ku*i//2,1,du,dv,angles),
                "mode_reference") for i in (1,2)]+main


def pair_catalog():
    # Equal-frequency pairs need a quadrature case to identify the zero-order phase dependence.
    for n in (4,8,16,2):
        for i,j in combinations(MODES,2):
            for phase in (0.,np.pi/2):
                yield i,n,j,n,phase
    # The gcd of the two frequencies permits shorter cells even for two short waves.
    for n,m in ((2,4),(4,8),(8,16),(2,16),(36,40),(56,64),(80,88),(120,160)):
        for i,j in (((0,"+"),(1,"+")),((1,"+"),(1,"+")),((0,"-"),(1,"-"))):
            yield i,n,j,m,0.


def pair_tag(i,n,j,m,phase):
    return f"cross_{mtag(i)}{n}_{mtag(j)}{m}_q{int(round(2*phase/np.pi))}"


def pair_jobs(angles=3,ku=10):
    jobs=[]
    for i,n,j,m,phase in pair_catalog():
        tag=pair_tag(i,n,j,m,phase)
        divisor=math.gcd(n,m)
        parity=(n//divisor%2,m//divisor%2)
        # Two explicitly solved signs plus verified exact half-cell translations.
        other=(1,-1) if parity[0] else (-1,1)
        for a,b in ((1,1),other):
            name=f"{tag}_{'p' if a>0 else 'm'}{'p' if b>0 else 'm'}"
            terms=[(*i,n,5*a,0.,1),(*j,m,5*b,phase,1)]
            g=modal_mark(name,terms)
            setting=modal_setting(g,angles,ku)
            if max(n,m)>16:
                # Keep both short-wave channels in the eigenbasis for their beat.
                setting=replace(setting,kv=max(setting.kv,max(n,m)//divisor+2))
            jobs.append(Job(g,setting,"mode_cross"))
    return jobs


def domain_points(count,scramble,avoid=()):
    dims=tuple(LEVELS)
    limit={tuple(d[k] for k in dims) for d in avoid}
    sampler=qmc.Sobol(d=len(dims),scramble=True,seed=scramble)
    points=[]
    for p in sampler.random_base2(8):
        theta={key:LEVELS[key][min(int(v*len(LEVELS[key])),len(LEVELS[key])-1)] for key,v in zip(dims,p)}
        # Shape-dependent height correlation gets an explicit block and a matched slice sweep.
        if len(points)%4:
            theta["rho_height"]=1.
        ident=tuple(theta[k] for k in dims)
        if ident not in limit:
            points.append(theta)
            limit.add(ident)
        if len(points)==count:
            return points
    raise AssertionError("The joint design did not produce enough distinct cases")


def reference_jobs(angles=3):
    jobs=[]
    for i,theta in enumerate(domain_points(32,20261123)):
        for seed in FIT_SEEDS[:2]:
            name=f"train_{i:02d}_s{seed}"
            g=random_mark(name,seed=seed,**theta)
            jobs.append(Job(g,low_setting(g,angles),"joint_train"))
    return jobs


def blind_jobs(angles=3):
    training=domain_points(32,20261123)
    familiar=[training[i] for i in (1,2,3,5,7,11,15,23)]
    novel=domain_points(8,20261124,training)
    jobs=[]
    for family,points in (("new_seed",familiar),("new_combo",novel)):
        for i,theta in enumerate(points):
            seed=BLIND_SEEDS[i]
            g=random_mark(f"blind_{family}_{i:02d}_s{seed}",seed=seed,**theta)
            jobs.append(Job(g,low_setting(g,angles),"blind_"+family))
    return jobs


def corner_mark(pair,tag,seed,levels=PAIR_LEVELS,*,controls=None):
    overrides=dict(CONTROL if controls is None else controls)
    overrides.update({key:levels[key][int(value)] for key,value in zip(pair,tag)})
    return random_mark(f"factor_{pair[0]}_{pair[1]}_{tag}_s{seed}",seed=seed,**overrides)


def factor_jobs(angles=3):
    for seed in FIT_SEEDS:
        for pair in FACTOR_PAIRS:
            for tag in ("00","10","01","11"):
                g=corner_mark(pair,tag,seed)
                yield Job(g,low_setting(g,angles),"stat_factor_cross")


def profile_jobs(angles=3):
    for factor,levels in PROFILES.items():
        for i,value in enumerate(levels):
            p=Profile(**{factor:value})
            for sigma in (0,5,20):
                if sigma==0:
                    g=smooth(f"profile_{factor}_{i}_smooth",8.,2,p)
                    yield Job(g,Setting(20,1,4,20,angles,4),"profile_axis")
                else:
                    for seed in FIT_SEEDS[:2]:
                        g=random_mark(f"profile_{factor}_{i}_r{sigma}_s{seed}",seed=seed,sigma_nm=sigma,
                                      xi_um=.2,profile=p)
                        yield Job(g,low_setting(g,angles),"profile_cross")


def band_jobs(angles=3):
    cutoff=(.05,.1,.2,.5,1.,2.)
    for seed in FIT_SEEDS:
        for j,(sigma,xi,h,rho) in enumerate(BAND_SCENARIOS):
            g=random_mark(f"band_{j}_s{seed}",seed=seed,sigma_nm=sigma,xi_um=xi,hurst=h,rho_lr=rho,cutoff=.05)
            for band in cutoff:
                selected=replace(g,name=f"{g.name}_b{band:g}",cutoff=band)
                yield Job(selected,Setting(10,8,4,3.125,angles),"nested_band_screen")
    for xi,length in product((2.,5.),(24.,48.)):
        g=random_mark(f"long_x{xi:g}_l{length:g}_s1427",seed=1427,sigma_nm=20,xi_um=xi,length=length,cutoff=2.)
        yield Job(g,Setting(10,8 if length==24 else 16,4,20,angles),"long_cell")
        yield Job(baseline(g),Setting(10,1,4,20,angles),"long_reference")


def chart_jobs(angles=3):
    cfg=Setting(9,12,2,5,angles)
    original=chart_mark()
    ref=baseline(original)
    yield Job(ref,Setting(cfg.ku,1,cfg.du_nm,cfg.dv_nm,angles),"chart_reference")
    for b in (.1,.5,1.,2.,.2):
        yield Job(chart_mark(b),cfg,"chart_nested")
    # A rigid translation on the original geometry tests phase calibration and template sign.
    for shift in (-100.,100.):
        g=replace(ref,name=f"chart_shift_{shift:+g}",rigid_nm=shift)
        yield Job(g,Setting(cfg.ku,1,cfg.du_nm,cfg.dv_nm,angles),"known_translation")
    for name in ("center_1427_r5_x0.2_b0.5","width_1427_r20_x0.2_b0.5"):
        old=from_legacy(name,PREVIOUS/"geometry"/f"{name}.npz")
        yield Job(old,low_setting(old,angles),"legacy_reproduction")


def high_jobs(angles=1):
    g=chart_mark()
    for ku,kv,du,dv in ((15,16,2,5),(20,12,2,5),(15,24,2,5),(20,16,2,5),
                        (15,12,1,2.5),(15,32,2,5),(20,24,2,5)):
        yield Job(g,Setting(ku,kv,du,dv,angles),"chart_convergence")
        yield Job(baseline(g),Setting(ku,1,du,dv,angles),"chart_convergence_ref")
    width=from_legacy("width_1427_r20_x0.2_b0.5",PREVIOUS/"geometry/width_1427_r20_x0.2_b0.5.npz")
    for mark in (width,*[j.mark for j in blind_jobs(angles)[:4]]):
        for ku,kv in ((20,8),(30,8),(20,16),(30,16)):
            yield Job(mark,Setting(ku,kv,2,10,angles,4),"strict_anchor")
            yield Job(baseline(mark),Setting(ku,1,2,10,angles),"strict_anchor_ref")
    for mark in (chart_mark(.5),chart_mark(.1)):
        for ku,kv,n in ((10,12,3),(10,12,5),(15,12,3)):
            if angles==1:
                yield Job(mark,Setting(ku,kv,2,5,n),"strict_angular")
                yield Job(baseline(mark),Setting(ku,1,2,5,n),"strict_angular_ref")


def unique(jobs):
    seen=set()
    for j in jobs:
        if str(j.folder) not in seen:
            yield j
            seen.add(str(j.folder))


def named_lanes(angles=3,ku=10):
    return {
        "chart":list(chart_jobs(angles)),
        "modes":kernel_jobs(angles,ku),
        "pairs":pair_jobs(angles,ku),
        "joint":reference_jobs(angles),
        "blind":blind_jobs(angles),
        "factors":list(factor_jobs(angles)),
        "profiles":list(profile_jobs(angles)),
        "bands":list(band_jobs(angles)),
        "strict":list(high_jobs(1)),
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lane",choices=("prepare","chart","modes","pairs","joint","blind","factors",
                                       "profiles","bands","strict","overnight"))
    parser.add_argument("--angles",type=int,choices=(1,3,5),default=3)
    parser.add_argument("--ku",type=int,choices=(10,20,30),default=10)
    parser.add_argument("--threads",type=int,default=5)
    parser.add_argument("--start",type=int,default=0)
    parser.add_argument("--limit",type=int)
    args=parser.parse_args()
    lanes=named_lanes(args.angles,args.ku)
    if args.lane=="prepare":
        directory=ROOT/"queue"
        directory.mkdir(parents=True,exist_ok=True)
        contract={lane:[dict(case=j.mark.name,group=j.group,settings=asdict(j.setting),
                             folder=str(j.folder.relative_to(ROOT)),geometry_hash=j.mark.signature(),
                             profile=asdict(j.mark.profile),inputs=j.mark.info,cutoff=j.mark.cutoff)
                        for j in unique(jobs)] for lane,jobs in lanes.items()}
        path=directory/f"contract_a{args.angles}_ku{args.ku}_v3.json"
        text=json.dumps(contract,ensure_ascii=False,indent=2,sort_keys=True)
        if path.is_file() and path.read_text(encoding="utf-8")!=text:
            raise ValueError(f"Frozen design has changed {path.name}")
        path.write_text(text,encoding="utf-8")
        print({key:len(value) for key,value in contract.items()})
        return
    selected=lanes.get(args.lane)
    if args.lane=="overnight":
        selected=[job for lane in ("chart","modes","blind","pairs","joint","strict","factors","bands","profiles")
                  for job in lanes[lane]]
    run_jobs(list(unique(selected))[args.start:],args.lane,args.threads,max_cases=args.limit)


if __name__=="__main__":
    main()

