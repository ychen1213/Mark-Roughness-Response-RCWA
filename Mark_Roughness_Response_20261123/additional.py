"""Additional verified references, height kernels and profile-conditioned kernels."""

import argparse
import csv
from dataclasses import replace
from itertools import product
import json

from design import MODES, FIT_SEEDS, PROFILES, blind_jobs, low_setting, modal_setting, mtag, solo, unique
from rcwa_runner import Job, Setting, baseline, run_jobs
from response_geometry import ROOT, Profile, chart_mark, load_mark, modal_mark, random_mark, smooth


def matched_chart(angles=3):
    base=smooth("chart_library_reference",8.,1)
    yield Job(base,Setting(5,1,4,5,angles),"chart_library_reference")
    for b in (.1,.5,1.,2.):
        mark=chart_mark(b)
        yield Job(mark,Setting(5,8,4,5,angles),"chart_library_check")
    # Independent rigid translations and a direct sign corner at matched raster.
    for shift in (-100.,100.):
        mark=replace(chart_mark(),name=f"chart_actual_shift_{shift:+g}",rigid_nm=shift)
        yield Job(mark,Setting(5,8,4,5,angles),"known_translation_geometry")
    for a,b in ((-1,1),(-1,-1)):
        mark=modal_mark(f"direct_corner_cw4_{a:+g}_{b:+g}",[(0,"+",4,5*a,0.,1),(1,"-",4,5*b,0.,1)])
        yield Job(mark,modal_setting(mark,angles),"native_corner_control")


def height_kernels(angles=3):
    # Even and odd vertical profiles use the same per-height PSD normalization
    # as the continuously interpolated random geometry.
    for rho in (.5,0.):
        for n,mode,sign in product((4,8,2,1,16),MODES,(1,-1)):
            name=f"hmode_r{rho:g}_{mtag(mode)}_n{n}_t{sign:+g}"
            mark=solo(mode,n,height=sign,name=name)
            mark=replace(mark,info={**mark.info,"rho_height":rho,"height_envelope":True})
            yield Job(mark,modal_setting(mark,angles),"height_kernel")
    for rho,n,mode,sign in product((.5,0.),(4,8),((0,"+"),(1,"+")),(1,-1)):
        name=f"hmode_r{rho:g}_{mtag(mode)}_n{n}_t{sign:+g}"
        mark=replace(solo(mode,n,height=sign,name=name),
                     info={"kind":"mode","rho_height":rho,"height_envelope":True,
                           "terms":[(*mode,n,5.,0.,sign)],"primitive_divisor":n,"full_length_um":8.})
        yield Job(mark,replace(modal_setting(mark,angles),slices=8),"height_split_convergence")


def height_blind_context(angles=3):
    mark=next(j.mark for j in blind_jobs(angles) if j.mark.vertically_varying)
    rho=mark.info["rho_height"]
    for job in height_kernels(angles):
        if job.group=="height_kernel" and job.mark.info["rho_height"]==rho and not (job.folder/"done.json").exists():
            yield job


def conditional_profiles(angles=3):
    for factor,levels in PROFILES.items():
        for i,value in enumerate(levels):
            profile=Profile(**{factor:value})
            for stripes in (1,2):
                mark=smooth(f"pbase_{factor}_{i}_u{stripes}",8.,stripes,profile)
                yield Job(mark,Setting(5*stripes,1,4,20,angles),"profile_reference")
            for n,mode in product((4,8,2,1,16),MODES):
                name=f"pmode_{factor}_{i}_{mtag(mode)}_n{n}"
                mark=solo(mode,n,profile=profile,name=name)
                yield Job(mark,modal_setting(mark,angles),"profile_kernel")


def profile_references(angles=3):
    for job in conditional_profiles(angles):
        if job.group=="profile_reference" and job.mark.stripes==2:
            yield job


def strict_pairs(angles=1):
    original=chart_mark()
    for cutoff in (.1,.5,1.,2.):
        g=chart_mark(cutoff)
        for kx,kz in ((15,16),(15,24),(20,16)):
            s=Setting(kx,kz,2,5,angles)
            yield Job(g,s,"chart_band_convergence")
            yield Job(baseline(g),replace(s,kv=1),"chart_band_reference")
    for j in blind_jobs(angles)[:4]:
        g=j.mark
        for kx,kz in ((20,8),(20,16)):
            s=Setting(kx,kz,2,10,angles)
            yield Job(g,s,"angular_blind_reference")
            yield Job(baseline(g),replace(s,kv=1),"angular_blind_smooth")


def strict_factors(angles=3):
    # Replay the widest input and its decision cutoffs on independent numerical axes.
    grids=(Setting(20,8,4,3.125,angles),Setting(10,16,4,3.125,angles),
           Setting(10,8,2,1.5625,angles),Setting(10,8,4,3.125,5))
    for cutoffs in ((.05,.5,1.),(.1,.2,2.)):
        for seed,sigma,xi,rho in ((1427,20,.2,-1.),(2441,5,.2,.5)):
            master=random_mark(f"qual_band_s{seed}",seed=seed,sigma_nm=sigma,xi_um=xi,rho_lr=rho,cutoff=.05)
            for s in grids:
                yield Job(baseline(master),replace(s,kv=1),"random_band_refinement_ref")
                for cutoff in cutoffs:
                    mark=replace(master,name=f"{master.name}_b{cutoff:g}",cutoff=cutoff)
                    yield Job(mark,s,"random_band_refinement")


def blind_diagnosis(angles=3):
    # This sample exceeded the frozen model's error limits. Its extra solves
    # are diagnostic and remain outside training and the original score.
    mark=load_mark(ROOT/"geometry"/"blind_new_combo_05_s9221.npz")
    for ku,kv,du,dv,n in ((20,4,4,20,angles),(10,8,4,20,angles),
                          (10,4,2,10,angles),(30,4,4,20,angles),
                          (10,4,4,20,5),(20,8,4,20,angles)):
        settings=Setting(ku,kv,du,dv,n)
        yield Job(baseline(mark),replace(settings,kv=1),"blind_resolution_reference")
        yield Job(mark,settings,"blind_resolution_diagnosis")


def blind_x_continuation(angles=3):
    """Continue the observed X-harmonic oscillation with fixed physical geometry."""
    mark=load_mark(ROOT/"geometry"/"blind_new_combo_05_s9221.npz")
    for kx,kz in ((40,4),(50,4),(40,8)):
        cfg=Setting(kx,kz,4,20,angles)
        yield Job(baseline(mark),replace(cfg,kv=1),"blind_resolution_reference")
        yield Job(mark,cfg,"blind_resolution_diagnosis")


def interaction_refinement(angles=3):
    profile_marks=[load_mark(ROOT/"geometry"/f"profile_cd_nm_{side}_r{sigma}_s1427.npz")
                   for side,sigma in product((0,1),(5,20))]
    factor_marks=[load_mark(ROOT/"geometry"/f"factor_sigma_nm_rho_lr_{tag}_s1427.npz")
                  for tag in ("00","10","01","11")]
    for ku,kv in ((20,8),(10,16),(20,16)):
        s=Setting(ku,kv,4,20,angles)
        nominal=smooth("reference_u2",8.,2)
        yield Job(nominal,replace(s,kv=1),"interaction_refinement_reference")
        for mark in profile_marks:
            yield Job(baseline(mark),replace(s,kv=1),"interaction_refinement_reference")
            yield Job(mark,s,"profile_refinement")
        for mark in factor_marks:
            yield Job(mark,s,"factor_refinement")


def sigma_length_refinement(angles=3):
    """Replay the four size-by-correlation-length corners on each harmonic axis."""
    prefix=ROOT/"geometry"
    corners=[load_mark(prefix/f"factor_sigma_nm_xi_um_{tag}_s1427.npz")
             for tag in ("00","10","01","11")]
    for ku,kv in ((20,8),(10,16)):
        setting=Setting(ku,kv,4,20,angles)
        yield Job(baseline(corners[0]),replace(setting,kv=1),"interaction_refinement_reference")
        for mark in corners:
            yield Job(mark,setting,"factor_refinement")


def priority_reduction_checks(angles=3):
    """Refine the largest screened center, width and RMS omissions first."""
    manifest=ROOT/"analysis"/"reduction_targets.csv"
    with manifest.open(encoding="utf-8-sig",newline="") as stream:
        records=list(csv.DictReader(stream))
    selected=[r for r in records if r["selection"]=="max_risk" and
              r["reduction"] in ("center","width","rms")]
    marks={name:load_mark(ROOT/"geometry"/f"{name}.npz")
           for row in selected for name in (row["base"],row["filtered"])}
    if len(selected)!=3:
        raise ValueError("Expected the frozen center, width and RMS targets")
    for ku,kv in ((20,8),(10,16)):
        setting=Setting(ku,kv,4,20,angles)
        for mark in marks.values():
            yield Job(baseline(mark),replace(setting,kv=1),"policy_reference")
            yield Job(mark,setting,"policy_direct")


def fine_blind(angles=3):
    held=blind_jobs(angles)
    # Four prespecified points spanning the two holdout families, plus the
    # separately labeled failed case for a numerical diagnosis.
    for i in (13,0,3,8,11):
        g=held[i].mark
        base=smooth("reference_u2",8.,2)
        kv=4 if g.cutoff>=2. else 8
        for v in (kv,2*kv):
            s=Setting(20,v,2,10,angles)
            yield Job(base,replace(s,kv=1),"fine_blind_reference")
            yield Job(g,s,"fine_blind_diagnostic" if i==13 else "fine_blind_test")


def main():
    builders={"chart_match":matched_chart,"height":height_kernels,"height_blind_context":height_blind_context,
              "profile_modes":conditional_profiles,
              "profile_refs":profile_references,
              "strict_pairs":strict_pairs,"strict_bands":strict_factors,"blind_diagnosis":blind_diagnosis,
              "blind_x_continuation":blind_x_continuation,"interaction_refinement":interaction_refinement,
              "sigma_length_refinement":sigma_length_refinement,"fine_blind":fine_blind,
              "priority_reduction_checks":priority_reduction_checks}
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("lane",choices=builders)
    p.add_argument("--angles",type=int,choices=(1,3,5),default=3)
    p.add_argument("--threads",type=int,default=5)
    p.add_argument("--limit",type=int)
    p.add_argument("--start",type=int,default=0)
    args=p.parse_args()
    jobs=list(unique(builders[args.lane](args.angles)))
    out=ROOT/"queue"/f"extra_{args.lane}_a{args.angles}.json"
    compact=[dict(case=j.mark.name,folder=str(j.folder.relative_to(ROOT)),
                  hash=j.mark.signature(),group=j.group,settings=j.setting.key) for j in jobs]
    out.write_text(json.dumps(compact,ensure_ascii=False,indent=2),encoding="utf-8")
    run_jobs(jobs[args.start:],args.lane,args.threads,max_cases=args.limit)


if __name__=="__main__":
    main()

