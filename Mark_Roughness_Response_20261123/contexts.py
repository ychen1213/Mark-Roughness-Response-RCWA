"""Conditioned response libraries for height-dependent edges and nonideal profiles."""

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from additional import conditional_profiles, height_kernels
from rcwa_runner import Job, Setting
from readout import align_spectrum, native_fields
from response_geometry import ROOT, Profile, smooth
from response_model import OUTPUT, Response, analytic_field_orders, has_data, open_response


def slug(profile):
    if profile==Profile():
        return "nominal"
    return f"cd{profile.cd_nm:g}_h{profile.height_nm:g}_a{profile.swa_deg:g}_s{profile.shear_nm:g}_f{profile.footing_nm:g}"


def context_name(mark):
    if mark.profile!=Profile():
        return "profile_"+slug(mark.profile)
    rho=float(mark.info.get("rho_height",1.))
    return f"height_r{rho:g}" if rho!=1 else None


def available_model(mark,directory):
    name=context_name(mark)
    path=Path(directory)/f"{name}.npz"
    return path if name is not None and path.is_file() else None


def build_response(recipe,reference,settings,destination,partial=False):
    if not has_data(reference):
        return False
    l,n=analytic_field_orders()
    ref=native_fields(reference.folder)
    def embed(job):
        return align_spectrum(native_fields(job.folder),6.4,8.,l,n,ref.incoming)
    base=embed(reference)
    L={}
    D={}
    for job in recipe:
        if not has_data(job):
            continue
        c,p,k,_,_,sign=job.mark.info["terms"][0]
        code=(c+(2 if sign<0 else 0),p,k)
        measured=embed(job)
        L[code]=np.where((abs(n)==k)[None,None,:,None],measured,0.)
        D[code]=np.where(((n==0)|(abs(n)==2*k))[None,None,:,None],measured-base,0.)
    if not L or (not partial and len(L)<len(recipe)):
        return False
    model=Response(l,n,ref.incoming,base,L,D,{},dict(angles=reference.setting.angles,
                   partial=len(L)<len(recipe),interpolate_low=True,known_modes=len(L),
                   intended_modes=len(recipe),**settings,
                   source_runs=[str(j.folder.relative_to(ROOT)) for j in [reference,*recipe] if has_data(j)]))
    model.save(destination)
    return True


def compile_contexts(angles=3,partial=False):
    output=OUTPUT/"conditional"
    bins=defaultdict(list)
    for job in height_kernels(angles):
        if job.group=="height_kernel":
            bins[job.mark.info["rho_height"]].append(job)
    completed=[]
    for rho,jobs in bins.items():
        label=f"height_r{rho:g}"
        ref=Job(smooth("reference_u2",8.,2),Setting(10,1,4,20,angles),"mode_reference")
        if build_response(jobs,ref,dict(vertical=True,rho_height=rho),
                          output/f"{label}.npz",partial):
            completed.append(label)
    profiles=defaultdict(dict)
    for job in conditional_profiles(angles):
        current=profiles[job.mark.profile]
        if job.group=="profile_reference":
            if job.mark.stripes==2:
                current["baseline"]=job
        else:
            current.setdefault("cases",[]).append(job)
    for p,data in profiles.items():
        label=f"profile_{slug(p)}"
        if build_response(data.get("cases",[]),data["baseline"],dict(vertical=False,profile=asdict(p)),
                          output/f"{label}.npz",partial):
            completed.append(label)
    return completed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--angles",type=int,default=3)
    p.add_argument("--partial",action="store_true")
    args=p.parse_args()
    print("Conditioned libraries:",compile_contexts(args.angles,args.partial))


if __name__=="__main__":
    main()

