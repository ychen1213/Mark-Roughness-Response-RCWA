"""Replay separate incident-angle blocks and assemble the unchanged native RCWA table."""

import argparse
from contextlib import ExitStack
from datetime import datetime
import json
import os
import time

import numpy as np

from rcwa_runner import Job, layout, native, physical_cell, run_one, save_native_scene, solver_slot
from recover_limited import settings
from response_geometry import ROOT, load_mark
from response_model import OUTPUT


def merge_blocks(blocks):
    joined={}
    names=set(blocks[0].files)
    if not all(set(block.files)==names for block in blocks):
        raise ValueError("Native angle blocks expose different result channels")
    for key in blocks[0].files:
        arrays=[block[key] for block in blocks]
        parameter=key.rsplit("_",1)[-1]
        if parameter in ("theta","phi"):
            joined[key]=np.concatenate(arrays,axis=0)
        elif key.startswith("sub_") or parameter in ("lambda","f","n","m"):
            if not all(np.array_equal(arrays[0],array) for array in arrays[1:]):
                raise ValueError(f"Native static coordinate differs between batches: {key}")
            joined[key]=arrays[0]
        else:
            joined[key]=np.concatenate(arrays,axis=1)
    return joined


def run_batches(job,destination,size,threads,min_free,indices=None,assemble_only=False):
    ready=destination/"done.json"
    if ready.exists():
        run_one(None,job,output_folder=destination)
        return
    all_angles=native.incident_table(job.setting.angles)
    full_range=range(len(all_angles))
    chunks=[]
    for start in full_range[::size]:
        if indices is not None and start not in indices:
            continue
        rows=tuple(full_range[start:start+size])
        path=destination/f"angles_{rows[0]:02d}_{rows[-1]:02d}"
        if not (path/"done.json").exists():
            if assemble_only:
                raise ValueError(f"The independently solved angle block is still missing: {path.name}")
            while True:
                if native.free_ram_gib()<min_free:
                    print("Waiting for memory for a native angle block",flush=True)
                    while native.free_ram_gib()<min_free:
                        time.sleep(25)
                with solver_slot():
                    if native.free_ram_gib()<min_free:
                        continue
                    sim=native.lumapi.FDTD(hide=True,serverArgs={"threads":str(threads)})
                    try:
                        receipt,_=run_one(sim,job,output_folder=path,angular_rows=rows)
                    finally:
                        sim.close()
                    break
        else:
            receipt,_=run_one(None,job,output_folder=path,angular_rows=rows)
        if not np.array_equal(receipt["angle_table_deg"],[all_angles[i] for i in rows]):
            raise ValueError(f"Saved result has different incidence angles: {path}")
        chunks.append((path,receipt))
    if indices is not None:
        print(f"Saved or checked {len(chunks)} assigned incidence angles.",flush=True)
        return
    with ExitStack() as stack:
        combined=merge_blocks([stack.enter_context(np.load(path/"raw.npz")) for path,_ in chunks])
    native_angles=np.column_stack((combined["field_theta"].ravel(),combined["field_phi"].ravel()))
    if combined["field_ekz_fs"].shape[1]!=len(all_angles) or not np.allclose(native_angles,all_angles,atol=1e-10,rtol=0):
        raise ValueError("Angular blocks do not populate the full native incident-angle table")
    if not (destination/"model.fsp").exists():
        solved,actual=physical_cell(job.mark,job.setting)
        script,_,_=layout(solved,actual)
        with solver_slot():
            sim=native.lumapi.FDTD(hide=True,serverArgs={"threads":str(threads)})
            try:
                save_native_scene(sim,script,destination)
            finally:
                sim.close()
    raw=destination/f".raw_{os.getpid()}.npz"
    np.savez_compressed(raw,**combined)
    raw.replace(destination/"raw.npz")
    metrics=dict(chunks[0][1]["results"])
    for name,combine in (("peak_memory",max),("simulation_run_time",sum)):
        values=[r["results"][name] for _,r in chunks if name in r["results"]]
        if len(values)==len(chunks):
            metrics[name]=combine(values)
    info={**chunks[0][1],"angle_table_deg":all_angles,"solve_s":sum(r["solve_s"] for _,r in chunks),
          "results":metrics,
          "angular_native_blocks":[str(p.relative_to(ROOT)) for p,_ in chunks],
          "assembled_at":datetime.now().astimezone().isoformat()}
    native.json_write(ready,info)
    print(f"Assembled {len(all_angles)} original incident angles in {len(chunks)} independent native blocks.")


def comparison(reference,assembled):
    with np.load(reference/"raw.npz") as a,np.load(assembled/"raw.npz") as b:
        if set(a.files)!=set(b.files):
            raise ValueError("Combined and native reads expose different channels")
        for k in a.files:
            if a[k].shape!=b[k].shape or not np.allclose(a[k],b[k],rtol=1e-8,atol=1e-10,equal_nan=True):
                raise ValueError(f"Combined and full-angle native results differ in {k}")
        error=np.linalg.norm(a["field_ekz_fs"]-b["field_ekz_fs"])/np.linalg.norm(a["field_ekz_fs"])
        channels=len(a.files)
    native.json_write(OUTPUT/"angular_batch_verification.json",dict(setting=reference.name,
                      geometry=reference.parent.name,relative_complex_error=float(error),compared_channels=channels))
    print(f"Full native vs blocked native relative complex error {error:.3g}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    parser.add_argument("setting")
    parser.add_argument("--size",type=int,choices=(1,3,5),default=3)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--min-free",type=float,default=8.)
    parser.add_argument("--indices",type=int,nargs="+",help="Solve only these zero-based native angle rows")
    parser.add_argument("--assemble",action="store_true",help="Assemble blocks already solved in separate processes")
    parser.add_argument("--control",action="store_true")
    args=parser.parse_args()
    mark=load_mark(ROOT/"geometry"/f"{args.case}.npz")
    key=settings(ROOT/args.setting)
    if args.indices is not None:
        if args.size!=1 or args.assemble or args.control or not set(args.indices).issubset(range(key.angles**2)):
            parser.error("Use --size 1 and valid native indices for partial solves")
    job=Job(mark,key,"fine_blind_test")
    destination=OUTPUT/"angular_batch_control"/f"size_{args.size}"/args.case/args.setting if args.control else job.folder
    run_batches(job,destination,args.size,args.threads,args.min_free,args.indices,args.assemble)
    if args.control:
        comparison(job.folder,destination)


if __name__=="__main__":
    main()
