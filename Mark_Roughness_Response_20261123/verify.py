"""Measured closure checks for the model coordinates, native sign corners, and APD."""

import json
from hashlib import sha256
from pathlib import Path
import subprocess

import numpy as np

from design import blind_jobs, kernel_jobs
from rcwa_runner import REGISTRY
from readout import (Spectrum, ScanOperator, align_spectrum, coherent_scan, comparison,
                     curves, evaluate, native_fields)
from response_geometry import ROOT, chart_mark
from response_model import OUTPUT, open_response

WIDGET=Path(r"C:\Users\Ychen\.codex\visualizations\2026\11\22\01a203cb-b69c-7d42-8d61-3a1cc18acd41\roughness-edge-modes.html")


def metric(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-20))


def widget():
    html=WIDGET.read_text(encoding="utf-8")
    functions=html[html.index("  const length = 8;"):html.index("  function evaluate()")]
    nodes=np.asarray(json.loads(subprocess.run(
        ["node","-e",functions+"\nconsole.log(JSON.stringify(makeBasis(.25,1427).points));"],
        check=True,capture_output=True,text=True).stdout))
    lo,hi=chart_mark().bounds(np.linspace(-4,4,len(nodes)))
    expected_l=np.sqrt(.75)*nodes[:,0]-.5*nodes[:,1]
    expected_r=np.sqrt(.75)*nodes[:,0]+.5*nodes[:,1]
    err=max(np.max(abs(1000*(lo[0]+.8)-expected_l)),np.max(abs(1000*(hi[0]-.8)-expected_r)))
    return dict(points=len(nodes),fourier_modes=80,max_edge_error_nm=float(err))


def native_checks():
    output={"chart_reconstruction":widget()}
    original=REGISTRY/"reference_u2"/"k10x1_d4x20_a3_h4"
    compact=REGISTRY/"reference_u1"/"k5x1_d4x20_a3_h4"
    a,b=map(native_fields,(original,compact))
    repeated=align_spectrum(b,a.lu,a.lv,a.l,a.n,a.incoming)
    op=ScanOperator(a.lu,a.lv,a.l,a.n,a.incoming)
    wave=curves(a)
    normal=coherent_scan(a,quadrature=18)
    n=np.arange(-7,8)
    longer=Spectrum(a.l,n,a.lu,24.,a.incoming,align_spectrum(a,a.lu,24.,a.l,n,a.incoming))
    long=coherent_scan(longer,quadrature=24)
    output.update(compact_smooth_relative_complex_error=metric(repeated,a.electric),
                  quadratic_readout_nrmse=comparison(op(a.electric),wave),
                  coherent_disjoint_nrmse=comparison(normal,wave),
                  overlapping_empty_blocks_nrmse=comparison(long,wave))
    jobs=kernel_jobs()
    observed={j.mark.name:j for j in jobs}
    translations=[]
    for j in jobs:
        if j.group!="phase_control" or not (j.folder/"done.json").exists():
            continue
        c,p,n,amp,phase,_=j.mark.info["terms"][0]
        base_name=j.mark.name.replace("control_","solo_").rsplit("_",1)[0]+f"_a{amp:g}"
        src=native_fields(observed[base_name].folder)
        real=native_fields(j.folder)
        expected=src.electric*np.exp(1j*real.n[None,None,:,None]*phase)
        # The compact Z cell's fundamental is the physical Fourier carrier.
        translations.append(dict(case=j.mark.name,complex_relative=float(metric(real.electric,expected))))
    output["independent_native_phase_controls"]=translations
    key="k10x4_d4x20_a3_h4"
    double=REGISTRY/"train_03_s1427"/key
    single=REGISTRY/"one_cell_equivalence_train_03_s1427"/key
    if all((p/"done.json").exists() for p in (double,single)):
        pair,primitive=map(native_fields,(double,single))
        repeated=align_spectrum(primitive,pair.lu,pair.lv,pair.l,pair.n,pair.incoming)
        expanded=Spectrum(pair.l,pair.n,pair.lu,pair.lv,pair.incoming,repeated)
        output["repeated_roughness_cell"]={
            "complex_relative":metric(repeated,pair.electric),
            "scan_nrmse":comparison(curves(expanded),curves(pair)),
            "two_stripe_run":str(double.relative_to(ROOT)),
            "primitive_run":str(single.relative_to(ROOT))}
    corners=[]
    for along,exact in (("pp","_-1_-1"),("pm","_-1_+1")):
        key="k10x3_d4x20_a3_h4"
        reported=REGISTRY/f"cross_cp4_wm4_q0_{along}"/key
        emitted=REGISTRY/f"direct_corner_cw4{exact}"/key
        if all((p/"done.json").exists() for p in (reported,emitted)):
            first,second=map(native_fields,(reported,emitted))
            expected=first.electric*((-1.)**first.n)[None,None,:,None]
            corners.append(dict(solved=emitted.parent.name,translated=reported.parent.name,
                                complex_relative=metric(expected,second.electric)))
    output["independent_native_sign_corners"]=corners
    strict=REGISTRY/"chart_library_reference"/"k5x1_d4x5_a3_h4"
    origin=REGISTRY/"chart_s1427"/"k5x8_d4x5_a3_h4"
    if (strict/"done.json").exists() and (origin/"done.json").exists():
        ref=curves(native_fields(strict))
        f=curves(native_fields(origin))
        known0=evaluate(f,ref)["apd_nm"]
        shifts=[]
        for d in (-100,100):
            path=REGISTRY/f"chart_actual_shift_{d:+g}"/"k5x8_d4x5_a3_h4"
            if (path/"done.json").exists():
                kpi=evaluate(curves(native_fields(path)),ref,d)
                shifts.append(dict(known_nm=d,measured_nm=kpi["position_nm"],
                                   roughness_APD_nm=kpi["apd_nm"],relative_recovery_error_nm=kpi["apd_nm"]-known0))
        output["rough_geometry_translation"]=shifts
    return output


def holdout_isolation():
    path=OUTPUT/"response_model.npz"
    if not path.is_file():
        return None
    model=open_response(path)
    sources=model.meta["source_runs"]
    training=model.meta["joint_calibration"]["training_folders"]
    receipts=[]
    for folder in set(sources+training):
        receipt=json.loads((ROOT/folder/"done.json").read_text(encoding="utf-8"))
        receipts.append(receipt)
    held=blind_jobs()
    names={j.mark.name for j in held}
    signatures={j.mark.signature() for j in held}
    overlap=[r["case"] for r in receipts if r["case"] in names or r["geometry_hash"] in signatures]
    assert not overlap, overlap
    assert len(signatures)==len(held)
    return dict(model_sha256=sha256(path.read_bytes()).hexdigest(),kernel_inputs=len(sources),
                joint_training_inputs=len(training),unique_heldout_geometries=len(signatures),
                identical_fit_or_kernel_inputs=len(overlap))


if __name__=="__main__":
    OUTPUT.mkdir(parents=True,exist_ok=True)
    data=native_checks()
    assert data["chart_reconstruction"]["max_edge_error_nm"]<1e-9
    assert data["compact_smooth_relative_complex_error"]<1e-8
    assert max(data["quadratic_readout_nrmse"])<1e-10
    assert max(data["coherent_disjoint_nrmse"])<1e-8
    assert max(data["overlapping_empty_blocks_nrmse"])<1e-5
    if "repeated_roughness_cell" in data:
        assert data["repeated_roughness_cell"]["complex_relative"]<1e-8
    assert all(r["complex_relative"]<1e-8 for r in data["independent_native_sign_corners"])
    assert all(abs(row["relative_recovery_error_nm"])<.01 for row in data.get("rough_geometry_translation",[]))
    data["holdout_isolation"]=holdout_isolation()
    (OUTPUT/"coordinate_validation.json").write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(data,indent=2,ensure_ascii=False))

