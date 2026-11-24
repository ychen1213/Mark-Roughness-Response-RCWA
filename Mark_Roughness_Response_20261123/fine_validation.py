"""Score the finer X-harmonic library against independent direct replays."""

import argparse
from hashlib import sha256
import time

from additional import fine_blind
import numpy as np

from design import kernel_jobs, pair_jobs, solo
from rcwa_runner import Job, Setting, baseline, run_jobs
from readout import align_spectrum, comparison, curves, evaluate, native_fields
from report_analysis import Results, save_csv, score
from response_geometry import ROOT, load_mark
from response_model import OUTPUT, Response, analytic_field_orders, build_response, has_data, keys, open_response


def waiting():
    base=kernel_jobs(3,20)
    modes=[j for j in base if j.group in ("mode_bank","mode_reference")]
    cross=[j for j in pair_jobs(3,20) if max(t[2] for t in j.mark.info["terms"])<=16]
    return {"mode_runs":sum(not has_data(j) for j in modes),
            "cross_runs":sum(not has_data(j) for j in cross),
            "direct_runs":sum(not has_data(j) for j in fine_blind())}


def strong_width_check(threads):
    """Rebuild the needed self responses on the refined X basis for one diagnostic geometry."""
    mark=load_mark(ROOT/"geometry"/"blind_new_combo_05_s9221.npz")
    target=Job(mark,Setting(60,4,4,20,3),"blind_resolution_diagnosis")
    reference=Job(baseline(mark),Setting(60,1,4,20,3),"blind_resolution_reference")
    terms=keys(mark)
    if not terms or any(c!=1 or n>4 for c,_,n in terms):
        raise ValueError("The diagnostic mark no longer matches the selected width-only grid")
    jobs={}
    for c,p,n in sorted(terms):
        primitive=solo((c,p),n)
        settings=Setting(30*primitive.stripes,max(1,4//n),4,20,3)
        jobs[c,p,n]=Job(primitive,settings,"high_width_kernel")
    run_jobs([reference,*jobs.values()],"high_width_kernel",threads)
    wait=[j.mark.name for j in (reference,target,*jobs.values()) if not has_data(j)]
    if wait:
        print("Waiting for refined native results:",wait,flush=True)
        return
    spec=native_fields(reference.folder)
    l,n=analytic_field_orders()

    def to_basis(job):
        return align_spectrum(native_fields(job.folder),6.4,8.,l,n,spec.incoming)

    base=to_basis(reference)
    linear={}
    diagonal={}
    for key,job in jobs.items():
        field=to_basis(job)
        k=key[2]
        linear[key]=np.where((abs(n)==k)[None,None,:,None],field,0.)
        diagonal[key]=np.where(((n==0)|(abs(n)==2*k))[None,None,:,None],field-base,0.)
    model=Response(l,n,spec.incoming,base,linear,diagonal,{},dict(angles=3,ku_full=60,partial=True,amplitude_nm=5.))
    smooth,estimates,_,coverage=model.predict(mark,("linear","self_second"))
    exact=curves(native_fields(target.folder))
    initial=evaluate(exact,smooth)
    records=[]
    payload=dict(positions_um=exact[0],sum_direct=exact[1],diff_direct=exact[2])
    for name,prediction in estimates.items():
        predicted=evaluate(prediction,smooth)
        errors=comparison(prediction,exact)
        records.append(dict(case=mark.name,setting=target.setting.key,model=name,
                            direct_wq=initial["wq"],pred_wq=predicted["wq"],
                            wq_error_percent=100*(predicted["wq"]-initial["wq"]),
                            direct_apd_nm=initial["apd_nm"],pred_apd_nm=predicted["apd_nm"],
                            apd_error_nm=predicted["apd_nm"]-initial["apd_nm"],
                            sum_nrmse=errors[0],diff_nrmse=errors[1],width_modes=len(terms),
                            missing_modes=len(coverage["unsupported"]),max_mode_peak_nm=coverage["maximum_solo_nm"]))
        payload.update({f"sum_{name}":prediction[1],f"diff_{name}":prediction[2]})
    np.savez_compressed(OUTPUT/"strong_width_high_scans.npz",**payload)
    print("High-basis dominant width diagnosis:",save_csv("strong_width_high_validation",records),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wait",action="store_true")
    p.add_argument("--strong-width",action="store_true",help="Rebuild the X=60 width response for the diagnostic mark")
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--single-mode-preview",action="store_true",
                   help="Score completed replays with fully covered linear and self terms")
    args=p.parse_args()
    if args.strong_width:
        strong_width_check(args.threads)
        return
    jobs={str(j.folder.relative_to(ROOT)) for j in fine_blind()
          if j.group in ("fine_blind_test","fine_blind_diagnostic")}
    if args.single_mode_preview:
        model=build_response(3,20,partial=True)
        rows,waves=score(Results(reveal=True),model,selected_runs=jobs)
        present=[r for r in rows if r["missing_terms"]==0 and r["model"] in ("smooth","linear","self_second")]
        print("Fine single-mode checks:",save_csv("fine_single_mode_preview",present))
        checked={r["case"]+"_"+r["native_setting"] for r in present if r["model"]=="self_second"}
        directory=OUTPUT/"fine_validation"
        directory.mkdir(parents=True,exist_ok=True)
        for key in sorted(checked):
            keep={"positions_um"}
            for port in ("sum","diff"):
                keep.update(f"{port}_{kind}" for kind in ("direct","ideal_for_direct","smooth","linear","self_second"))
            np.savez_compressed(directory/f"{key}_single_modes.npz",**{k:v for k,v in waves[key].items() if k in keep})
        return
    while True:
        keys=waiting()
        if not any(keys.values()):
            break
        if not args.wait:
            print("Fine validation awaiting:",keys)
            return
        time.sleep(40)
    frozen=OUTPUT/"response_fine.npz"
    if not frozen.exists():
        build_response(3,20).save(frozen)
    model=open_response(frozen)
    model.meta["digest"]=sha256(frozen.read_bytes()).hexdigest()
    rows,waves=score(Results(reveal=True),model,selected_runs=jobs)
    print("Fine selected comparisons:",save_csv("fine_response_validation",rows))
    if waves:
        directory=OUTPUT/"fine_validation"
        directory.mkdir(parents=True,exist_ok=True)
        for key,v in waves.items():
            np.savez_compressed(directory/f"{key}.npz",**v)


if __name__=="__main__":
    main()
