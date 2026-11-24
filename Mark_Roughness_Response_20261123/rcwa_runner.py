"""Native RCWA in Lumerical's design environment. This study launches no time-domain solver."""

import argparse
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from itertools import groupby
import json
import os
from pathlib import Path
from shutil import copy2
import sys
import time
import traceback

import numpy as np

from response_geometry import ROOT, YPIN, Mark, chart_mark, load_mark, read_edge_csv, smooth

PREVIOUS = Path(os.environ.get("ROUGHNESS_BASE_PROJECT",r"E:\Mark_Roughness_50Duty_20261123"))
sys.path.append(str(PREVIOUS))
import native

os.environ["PATH"]=str(native.INSTALL/"bin")+os.pathsep+os.environ["PATH"]
REGISTRY=ROOT/"runs"


@dataclass(frozen=True)
class Setting:
    ku: int
    kv: int
    du_nm: float = 4.
    dv_nm: float = 20.
    angles: int = 1
    slices: int = 4
    li: bool = False

    @property
    def key(self):
        label=f"k{self.ku}x{self.kv}_d{self.du_nm:g}x{self.dv_nm:g}_a{self.angles}_h{self.slices}"
        return label+("_li" if self.li else "")

    @property
    def modes(self):
        return (2*self.ku+1)*(2*self.kv+1)


@dataclass(frozen=True)
class Job:
    mark: Mark
    setting: Setting
    group: str

    @property
    def folder(self):
        return REGISTRY/self.mark.name/self.setting.key


class ResourceLimit(RuntimeError):
    pass


@contextmanager
def solver_slot(exclusive=False):
    """Bound concurrent CAD processes on the measured 32 GiB workstation."""
    if sys.platform!="win32":
        yield
        return
    win=ctypes.WinDLL("kernel32",use_last_error=True)
    win.CreateMutexW.argtypes=(wintypes.LPVOID,wintypes.BOOL,wintypes.LPCWSTR)
    win.CreateMutexW.restype=wintypes.HANDLE
    win.WaitForMultipleObjects.argtypes=(wintypes.DWORD,ctypes.POINTER(wintypes.HANDLE),
                                        wintypes.BOOL,wintypes.DWORD)
    win.WaitForMultipleObjects.restype=wintypes.DWORD
    win.ReleaseMutex.argtypes=(wintypes.HANDLE,)
    win.CloseHandle.argtypes=(wintypes.HANDLE,)
    count=int(os.environ.get("ROUGHNESS_RCWA_SLOTS","2"))
    if not 1<=count<=8:
        raise ValueError("ROUGHNESS_RCWA_SLOTS must be between 1 and 8")
    if exclusive:
        count=max(3,count)
    handles=(wintypes.HANDLE*count)()
    selected=None
    try:
        for i in range(count):
            handles[i]=win.CreateMutexW(None,False,f"Local\\RoughnessResponse20261123_{i}")
            if not handles[i]:
                raise ctypes.WinError(ctypes.get_last_error())
        waiting=False
        while selected is None:
            state=win.WaitForMultipleObjects(count,handles,exclusive,40000)
            if 0<=state<count or 128<=state<128+count:
                selected=state if state<count else state-128
            elif state==258:
                if not waiting:
                    print("Waiting for a study RCWA worker slot",flush=True)
                    waiting=True
            else:
                raise ctypes.WinError(ctypes.get_last_error())
        yield
    finally:
        if selected is not None:
            for index in range(count) if exclusive else (selected,):
                win.ReleaseMutex(handles[index])
        for handle in handles:
            if handle:
                win.CloseHandle(handle)


def intervals(mark, setting):
    if not mark.vertically_varying:
        return np.array([0.,mark.profile.height])
    bounds=np.linspace(0,mark.profile.height,setting.slices+1)
    if mark.profile.footing_nm and mark.profile.height>.05:
        bounds=np.unique(np.r_[bounds,.05])
    return bounds


def layout(mark,setting,angular_rows=None):
    """Use precisely the same interface sampling for matching reference geometries."""
    position=native.position
    n_samples=1+int(np.ceil((mark.length+.2)/min(.01,setting.dv_nm/1000,mark.cutoff/12)))
    z=np.linspace(-mark.length/2-.1,mark.length/2+.1,n_samples)
    material=native.MATCHED
    boundaries=intervals(mark,setting)
    script=[
        'addrect;set("name","substrate");'+position("x span",mark.lu+2)+
        position("z span",mark.length+2)+position("y min",-2)+position("y max",0)+
        f'set("material","{material}");'
    ]
    for layer,(bottom,top) in enumerate(zip(boundaries[:-1],boundaries[1:])):
        middle=(bottom+top)/2
        left,right=mark.bounds(z,middle/mark.profile.height)
        for i in range(mark.stripes):
            name=f"ridge_{layer}_{i}"
            if mark.along_invariant:
                script.append(f'addrect;set("name","{name}");'+
                    position("x",(left[i,0]+right[i,0])/2)+position("x span",right[i,0]-left[i,0])+
                    position("z span",mark.length+.2)+position("y min",bottom)+position("y max",top)+
                    f'set("material","{material}");')
            else:
                poly=np.vstack((np.c_[right[i],z],np.c_[left[i,::-1],z[::-1]]))
                vertices=";\n".join(f"{u:.14g},{v:.14g}" for u,v in poly)
                script.append(f'addpoly;set("name","{name}");set("vertices",[{vertices}]*1e-6);'+
                              position("y",middle)+position("z span",top-bottom)+
                              'set("first axis","x");set("rotation 1",90);'+f'set("material","{material}");')
    theta=native.incident_table(setting.angles)
    if angular_rows is not None:
        theta=[theta[index] for index in angular_rows]
    table=";".join(f"{t:.14g},{p:.14g}" for t,p in theta)
    interface=";".join(f"{v:.14g}" for v in boundaries)
    region="2D Z-normal" if setting.kv==0 else "3D"
    script.extend((
        'addrcwa;set("name","RCWA");',
        f'set("simulation region","{region}");set("propagation axis","y");'
        'set("propagation direction","backward");',
        position("x span",mark.lu),
        position("z span",mark.length) if setting.kv else position("z",0),
        position("y min",-.15),position("y max",max(YPIN+.1,mark.profile.height+.1)),
        f'set("interface position","absolute");set("interface absolute positions",[{interface}]*1e-6);',
        'set("k vectors domain","rectangular");',
        f'set("max number ku",{setting.ku});set("max number kv",{max(1,setting.kv)});',
        'set("mesh type","custom");set("mesh refinement","staircase");',
        f'set("mesh cells x",{round(mark.lu*1000/setting.du_nm)});',
        f'set("mesh cells z",{round(mark.length*1000/setting.dv_nm)});' if setting.kv else "",
        f'set("use li factorization",{int(setting.li)});' if not setting.kv else "",
        'set("fit materials with multi-coefficient model",0);',
        f'set("sample spacing","custom");set("custom frequency samples",{native.FREQUENCY:.14g});',
        f'set("incident angle","table");set("incident angle table",[{table}]);',
        'set("report grating orders",1);set("report grating power",1);',
        'set("report grating characterization",1);set("report field amplitudes",1);'
    ))
    return "\n".join(script),theta,boundaries


def estimate_memory(setting,layer_count):
    dense=1+5.6*(setting.modes/1353)**2
    # A layered eigenproblem retains additional basis transforms during composition.
    return dense*(1+.095*max(layer_count-1,0))


def physical_cell(mark,setting):
    """Reduce exactly repeated stripe pairs without changing physical cutoffs."""
    repeat=(mark.info.get("kind")=="random" and mark.stripes==2 and setting.ku%2==0 and
            np.array_equal(mark.active[:,0],mark.active[:,1]) and
            round(mark.lu*1000/setting.du_nm)%2==0 and
            round(mark.lu*1000/setting.du_nm)==2*round(mark.lu*500/setting.du_nm))
    if repeat:
        return replace(mark,coeff=mark.coeff[:,:1]),replace(setting,ku=setting.ku//2)
    return mark,setting


def save_native_scene(sim,script,folder):
    path=folder/"layout.lsf"
    path.write_text(script,encoding="ascii")
    sim.eval("switchtolayout;newproject;redrawoff;")
    fit=native.silicon(sim)
    native.evaluate_layout(sim,script,path)
    sim.save((folder/"model.fsp").as_posix())
    return fit


def run_one(sim, job, *, output_folder=None, angular_rows=None):
    mark,setting=job.mark,job.setting
    folder=job.folder if output_folder is None else output_folder
    ready=folder/"done.json"
    if ready.is_file():
        receipt=json.loads(ready.read_text(encoding="utf-8"))
        if receipt["geometry_hash"]!=mark.signature() or not (folder/"raw.npz").is_file():
            raise ValueError(f"Inconsistent saved RCWA result {folder}")
        return receipt,True
    folder.mkdir(parents=True,exist_ok=True)
    solved,actual=physical_cell(mark,setting)
    script,angles,boundaries=layout(solved,actual,angular_rows)
    estimate=estimate_memory(actual,len(boundaries)-1)
    available=native.free_ram_gib()
    preflight=dict(field_modes=actual.modes,estimated_GiB=estimate,available_GiB=available,
                   layers=len(boundaries)-1,solved_Lx_um=solved.lu)
    native.json_write(folder/"resource.json",preflight)
    if estimate+1.5>available:
        raise ResourceLimit(f"{estimate:.2f} GiB estimated, {available:.2f} GiB free")
    mark.save()
    fit=save_native_scene(sim,script,folder)
    start=time.monotonic()
    print(f"RCWA {job.group} {mark.name} {setting.key}, modes {actual.modes}",flush=True)
    sim.eval("run;")
    seconds=time.monotonic()-start
    payload={}
    for key,label in (("field","field_amplitudes"),("ord","grating_orders"),
                      ("power","grating_power"),("energy","total_energy"),("sub","substrate")):
        payload.update({f"{key}_{k}":v for k,v in native.dataset(sim,label).items()})
    field=payload.get("field_ekz_fs")
    if field is None or len(field.shape)!=4 or not np.isfinite(field).all():
        raise ValueError("RCWA did not return finite native complex fields")
    tmp=folder/f".raw_{os.getpid()}.npz"
    np.savez_compressed(tmp,**payload)
    tmp.replace(folder/"raw.npz")
    extracted={}
    for key in ("num_k","peak_memory","simulation_run_time","total_threads"):
        try:
            extracted[key]=sim.getresult("RCWA",key)
        except native.lumapi.LumApiError:
            continue
    info=dict(case=mark.name,geometry_hash=mark.signature(),group=job.group,
              setting=asdict(setting),settings_key=setting.key,angle_table_deg=angles,
              reference_y_um=mark.profile.height,lu_um=solved.lu,lv_um=mark.length,
              requested_lu_um=mark.lu,native_setting=asdict(actual),
              fit_nk=fit,interface_y_um=boundaries,solve_s=seconds,results=extracted)
    native.json_write(ready,info)
    print(f"Saved {mark.name} {setting.key} in {seconds:.1f} s",flush=True)
    return info,False


def exact_results():
    """Index finished identical coefficient sets at the same solver setting."""
    sources={}
    for receipt_path in REGISTRY.glob("*/*/done.json"):
        receipt=json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["case"].startswith("blind_") or "native_equivalent_run" in receipt:
            continue
        folder=receipt_path.parent
        if (folder/"raw.npz").is_file():
            sources.setdefault((receipt["geometry_hash"],receipt["settings_key"]),(folder,receipt))
    return sources


def replay_exact(job,source):
    """Reuse an identical continuous geometry and preserve the source solve."""
    source_dir,receipt=source
    job.mark.save()
    job.folder.mkdir(parents=True,exist_ok=True)
    for name in ("raw.npz","model.fsp","layout.lsf","resource.json"):
        origin=source_dir/name
        if origin.is_file():
            temporary=job.folder/f".{name}.{os.getpid()}.tmp"
            copy2(origin,temporary)
            temporary.replace(job.folder/name)
    translated={**receipt,"case":job.mark.name,"group":job.group,
                "native_equivalent_run":str(source_dir.relative_to(ROOT))}
    native.json_write(job.folder/"done.json",translated)
    return translated


def substantial_native_job(job):
    mark,setting=job.mark,job.setting
    raster_x=round(mark.lu*1000/setting.du_nm)
    raster_z=round(mark.length*1000/setting.dv_nm) if setting.kv else 1
    return setting.modes>=450 or mark.vertically_varying or raster_x*raster_z>=2_000_000


def run_jobs(jobs,lane,threads=8,*,max_cases=None):
    planned=list(jobs)
    if max_cases is not None:
        planned=planned[:max_cases]
    failures=0
    for heavy,group in groupby(planned,key=substantial_native_job):
        jobs=list(group)
        batch_size=1 if heavy else 8
        for start in range(0,len(jobs),batch_size):
            batch=jobs[start:start+batch_size]
            exclusive=False
            if heavy:
                job=batch[0]
                _,actual=physical_cell(job.mark,job.setting)
                exclusive=actual.modes>=1300 and (job.mark.vertically_varying or actual.angles>1)
            with solver_slot(exclusive):
                failures+=execute_jobs(batch,lane,threads)
    return failures


def execute_jobs(jobs,lane,threads):
    """Log each outcome, with one fresh-session retry on a solver failure."""
    out=ROOT/"queue"
    out.mkdir(parents=True,exist_ok=True)
    filename=out/f"{lane}_{os.getpid()}.jsonl"
    print(f"{lane}: next {len(jobs)} cases, queue log {filename.name}",flush=True)
    failures=0
    sources=exact_results()
    sim=None
    try:
        with filename.open("a",encoding="utf-8") as log:
            for job in jobs:
                row=dict(case=job.mark.name,setting=job.setting.key,group=job.group,
                         folder=str(job.folder.relative_to(ROOT)))
                for attempt in range(2):
                    try:
                        key=(job.mark.signature(),job.setting.key)
                        if (job.folder/"done.json").is_file():
                            receipt,_=run_one(sim,job)
                            row.update(result="cached",seconds=receipt["solve_s"])
                        elif key in sources and not job.mark.name.startswith("blind_"):
                            receipt=replay_exact(job,sources[key])
                            row.update(result="cached_equivalent",seconds=receipt["solve_s"],
                                       source=receipt["native_equivalent_run"])
                        else:
                            if sim is None:
                                sim=native.lumapi.FDTD(hide=True,serverArgs={"threads":str(threads)})
                            receipt,_=run_one(sim,job)
                            row.update(result="complete",seconds=receipt["solve_s"])
                            if not job.mark.name.startswith("blind_"):
                                sources[key]=(job.folder,receipt)
                            k=receipt.get("native_setting",receipt["setting"])
                            modes=(2*k["ku"]+1)*(2*k["kv"]+1)
                            if modes>=450 or len(receipt["interface_y_um"])>3:
                                sim.close()
                                sim=None
                        break
                    except ResourceLimit as exc:
                        row.update(result="resource_limit",error=str(exc))
                        failures+=1
                        print(f"Resource limit {job.mark.name} {job.setting.key}: {exc}",flush=True)
                        break
                    except Exception as exc:
                        message=str(exc)
                        print(f"RCWA attempt {attempt+1} {job.mark.name}: {message}",flush=True)
                        (job.folder/"last_failure.txt").parent.mkdir(parents=True,exist_ok=True)
                        (job.folder/"last_failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
                        if sim is not None:
                            try:
                                sim.close()
                            except Exception:
                                pass
                            sim=None
                        if attempt:
                            failures+=1
                            row.update(result="failed",error=message)
                row["time"]=datetime.now().astimezone().isoformat()
                log.write(json.dumps(row,ensure_ascii=False)+"\n")
                log.flush()
    finally:
        if sim is not None:
            sim.close()
    print(f"{lane}: complete, {failures} cases unavailable",flush=True)
    return failures


def baseline(mark):
    if mark.profile==smooth().profile:
        name=f"smooth_l{mark.length:g}_u{mark.stripes}"
    else:
        p=mark.profile
        name=f"smooth_l{mark.length:g}_u{mark.stripes}_cd{p.cd_nm:g}_h{p.height_nm:g}_a{p.swa_deg:g}_s{p.shear_nm:g}_f{p.footing_nm:g}"
    return smooth(name,mark.length,mark.stripes,mark.profile)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("case",help="chart, saved geometry NPZ, or periodic sampled edge CSV")
    p.add_argument("--ku",type=int,default=15)
    p.add_argument("--kv",type=int,default=12)
    p.add_argument("--du",type=float,default=2.)
    p.add_argument("--dv",type=float,default=5.)
    p.add_argument("--angles",type=int,choices=(1,3,5),default=1)
    p.add_argument("--slices",type=int,default=4)
    p.add_argument("--threads",type=int,default=8)
    p.add_argument("--cutoff",type=float,default=.1)
    p.add_argument("--length",type=float,default=8.,help="Z period of sampled CSV")
    p.add_argument("--origin",type=float,default=0.,help="Optical Z origin in the sampled CSV")
    p.add_argument("--with-reference",action="store_true")
    args=p.parse_args()
    if args.case=="chart":
        mark=chart_mark(args.cutoff)
    elif Path(args.case).suffix.lower()==".csv":
        mark=read_edge_csv(args.case,length=args.length,origin=args.origin,cutoff=args.cutoff)
        mark=replace(mark,name=f"{mark.name}_b{mark.cutoff:g}_{mark.signature()[:10]}")
    else:
        mark=load_mark(args.case)
    cfg=Setting(args.ku,args.kv,args.du,args.dv,args.angles,args.slices)
    jobs=[Job(mark,cfg,"original_curve")]
    if args.with_reference:
        jobs.append(Job(baseline(mark),Setting(cfg.ku,1,cfg.du_nm,cfg.dv_nm,cfg.angles,cfg.slices),
                        "smooth_reference"))
    raise SystemExit(bool(run_jobs(jobs,"original_curve",args.threads)))


if __name__=="__main__":
    main()

