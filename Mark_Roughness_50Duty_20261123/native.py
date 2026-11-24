"""Native Ansys Lumerical solves and raw result export."""

import argparse
import ctypes
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

from geometry import Geometry, HEIGHT, REFERENCE_Y, ROOT, WAVELENGTH, WIDTH, cases


INSTALL = Path(os.environ.get("LUMERICAL_ROOT", r"D:\Ansys Inc\v252\Lumerical"))
sys.path.insert(0, str(INSTALL / "api" / "python"))
import lumapi

MATERIAL = "Si (Silicon) - Palik"
MATCHED = "Si_643nm_matched"
CONDUCTIVE = "Si_643nm_conductive"
C0 = 299792458.
FREQUENCY = C0 / (WAVELENGTH * 1e-6)
F_MIN = C0 / 663e-9
F_MAX = C0 / 623e-9
MESHES = ((20., 20., 10.), (16., 16., 8.), (12.5, 12.5, 6.25))


def free_ram_gib():
    """Available physical RAM on this Windows solver host."""
    class Status(ctypes.Structure):
        _fields_ = [("length",ctypes.c_uint32),("load",ctypes.c_uint32)] + [
            (name,ctypes.c_uint64) for name in
            ("total","available","page_total","page_available","virtual_total","virtual_available","extended")
        ]
    if sys.platform!="win32":
        return float("inf")
    status=Status()
    status.length=ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return float("inf")
    return status.available / 2**30


def json_write(path, value):
    tmp = path.with_suffix(".partial.json")
    tmp.write_text(json.dumps(value, indent=2, default=lambda obj: np.asarray(obj).tolist()), encoding="utf-8")
    tmp.replace(path)


def numeric(v):
    return f"{v:.14g}"


def position(prop, um):
    return f'set("{prop}",{numeric(um*1e-6)});'


def evaluate_layout(session, script, path):
    session.eval('setup_error="";try {\n'+script+'\n} catch(setup_error);')
    error = session.getv("setup_error")
    if error:
        raise RuntimeError(f"{error}\nLayout source: {path}")


def selected(name):
    catalog = cases()
    if name in catalog:
        return catalog[name]
    if name == "air":
        return Geometry("air")
    raise ValueError(f"Unknown geometry {name}")


def geometry_script(g, material, domain_u, domain_v, finite=None, offset=0.):
    substrate_y = HEIGHT if g.mode == "flat" else 0.
    substrate = (f'addrect;set("name","substrate");{position("x span",domain_u+2)}'
                 f'{position("z span",domain_v+2)}{position("y min",-2)}'
                 f'{position("y max",substrate_y)}set("material","{material}");')
    lines = [] if g.mode == "air" else [substrate]
    if g.mode in ("air", "flat"):
        return "\n".join(lines)
    end = finite / 2 if finite is not None else domain_v / 2 + .1
    v = np.linspace(-end, end, round(2 * end / .01) + 1)
    left, right = g.bounds(v, offset)
    for i, center in enumerate(g.centers):
        if g.mode == "smooth":
            lines.append(f'addrect;set("name","line_{i}");{position("x",center+offset)}'
                         f'{position("x span",WIDTH)}{position("z span",2*end)}'
                         f'{position("y min",0)}{position("y max",HEIGHT)}set("material","{material}");')
            continue
        polygon = np.vstack((np.column_stack((right[i], v)),
                             np.column_stack((left[i, ::-1], v[::-1]))))
        vertices = ";\n".join(f"{numeric(u)},{numeric(z)}" for u, z in polygon)
        lines.append(f'addpoly;set("name","line_{i}");set("vertices",[{vertices}]*1e-6);'
                     f'{position("y",HEIGHT/2)}{position("z span", HEIGHT)}'
                     f'set("first axis","x");set("rotation 1",90);set("material","{material}");')
    return "\n".join(lines)


def incident_table(count, range_k=.24):
    if count == 1:
        return np.array([[0., 0.]])
    k0 = 2 * np.pi / WAVELENGTH
    nodes = np.linspace(-range_k, range_k, count)
    ordered = [(0., 0.)] + [(kx, kz) for kx in nodes for kz in nodes if kx or kz]
    return np.array([(np.degrees(np.arcsin(np.hypot(kx,kz) / k0)),
                      np.degrees(np.arctan2(kz,kx)) % 360) for kx,kz in ordered])


def rcwa_script(g, ku, kv, du, dv, angles, spectrum_grid, factor):
    two_dim = kv == 1
    region = "2D Z-normal" if two_dim else "3D"
    table = ";".join(f"{numeric(theta)},{numeric(phi)}" for theta, phi in angles)
    positions = "0.3" if g.mode == "flat" else "0;0.3"
    parts = [
        geometry_script(g, MATCHED, g.lu, g.length_um),
        'addrcwa;set("name","RCWA");',
        f'set("simulation region","{region}");set("propagation axis","y");'
        'set("propagation direction","backward");',
        position("x span", g.lu), position("y min", -.1), position("y max", .7),
        'set("z",0);' if two_dim else position("z span", g.length_um),
        f'set("interface position","absolute");set("interface absolute positions",[{positions}]*1e-6);',
        'set("k vectors domain","rectangular");',
        f'set("max number ku",{ku});set("max number kv",{kv});',
        'set("mesh type","custom");',
        f'set("mesh cells x",{round(g.lu*1000/du)});',
        "" if two_dim else f'set("mesh cells z",{round(g.length_um*1000/dv)});',
        'set("mesh refinement","staircase");',
        f'set("use li factorization",{int(factor)});' if two_dim else "",
        'set("fit materials with multi-coefficient model",0);',
        f'set("sample spacing","custom");set("custom frequency samples",{numeric(FREQUENCY)});',
        f'set("incident angle","table");set("incident angle table",[{table}]);',
        'set("report grating orders",1);set("report grating power",1);'
        'set("report grating characterization",1);set("report field amplitudes",1);',
        'set("report index",1);' if spectrum_grid else '',
    ]
    return "\n".join(parts)


def silicon(session):
    fit = complex(np.asarray(session.getfdtdindex(MATERIAL, FREQUENCY, F_MIN, F_MAX)).item())
    name = session.addmaterial("Sampled data")
    session.setmaterial(name, "name", MATCHED)
    sample_freqs = np.linspace(F_MIN, F_MAX, 5)
    session.setmaterial(MATCHED, "sampled data",
                        np.column_stack((sample_freqs, np.full(sample_freqs.shape, fit**2))))
    measured = complex(np.asarray(session.getindex(MATCHED, FREQUENCY)).item())
    if abs(fit - measured) > 1e-8:
        raise ValueError(f"Material mismatch: {fit}, {measured}")
    return [fit.real, fit.imag]


def conductive_silicon(session):
    """Same complex permittivity at the target frequency, single Ohmic term."""
    fit = complex(np.asarray(session.getfdtdindex(MATERIAL,FREQUENCY,F_MIN,F_MAX)).item())
    eps = fit**2
    eps0 = 8.8541878128e-12
    mat = session.addmaterial("Conductive")
    session.setmaterial(mat,"name",CONDUCTIVE)
    session.setmaterial(CONDUCTIVE,"Permittivity",eps.real)
    session.setmaterial(CONDUCTIVE,"Conductivity",2*np.pi*FREQUENCY*eps0*eps.imag)
    check = complex(np.asarray(session.getfdtdindex(CONDUCTIVE,FREQUENCY,F_MIN,F_MAX)).item())
    if abs(check-fit) > 1e-7:
        raise ValueError(f"Conductive FDTD material mismatch: {check}, Palik fit {fit}")
    return check


def dataset(session, name):
    result = session.getresult("RCWA", name)
    return {key: np.asarray(val) for key, val in result.items()
            if key != "Lumerical_dataset" and np.asarray(val).dtype.kind in "buifc"}


def index_raster(session):
    # The v252 Python rectilinear translator counts interface coordinates as
    # layer samples. Fetch individual LSF arrays to preserve the native shape.
    session.eval('rr_preview=getresult("RCWA","index");'
                 'rr_x=rr_preview.x;rr_y=rr_preview.y;rr_z=rr_preview.z;rr_n=rr_preview.index_y;')
    return {key:np.asarray(session.getv("rr_"+value))
            for key,value in (("x","x"),("y","y"),("z","z"),("index_y","n"))}


def rcwa_solve(session, g, args):
    suffix = f"k{args.ku}x{args.kv}_d{args.du:g}x{args.dv:g}_a{args.angles}"
    if args.li:
        suffix += "_li"
    run = ROOT / "runs" / "rcwa" / g.name / suffix
    run.mkdir(parents=True, exist_ok=True)
    if (run / "done.json").exists():
        return run
    count=(2*args.ku+1)*(1 if args.kv==1 else 2*args.kv+1)
    if count>1353:
        # Based on the observed 4.9 GiB CAD footprint with 1353 modes,
        # including a reserve for other processes on the shared workstation.
        estimate=.65+4.3*(count/1353)**2
        available=free_ram_gib()
        if estimate+2>available:
            json_write(run/"resource_limit.json",dict(mode_count=count,estimated_GiB=estimate,available_GiB=available))
            raise MemoryError(f"RCWA {count} modes estimate {estimate:.2f} GiB with {available:.2f} GiB free")
    g.save(ROOT / "geometry")
    started = time.monotonic()
    session.eval("switchtolayout;newproject;redrawoff;")
    nk = silicon(session)
    table = incident_table(args.angles)
    script = rcwa_script(g, args.ku, args.kv, args.du, args.dv, table, args.index, args.li)
    (run / "build.lsf").write_text(script, encoding="ascii")
    evaluate_layout(session, script, run / "build.lsf")
    session.save((run / "model.fsp").as_posix())
    geometry_started = time.monotonic()
    print(f"RCWA {g.name} {suffix}", flush=True)
    session.eval("run;")
    solve_s = time.monotonic() - geometry_started
    payload = {}
    for key, name in (("char","grating_characterization"), ("ord","grating_orders"),
                      ("power","grating_power"), ("field","field_amplitudes"),
                      ("energy","total_energy"), ("sub","substrate")):
        payload.update({key+"_"+label: value for label,value in dataset(session, name).items()})
    np.savez_compressed(run / "raw.npz", **payload)
    log = {}
    for key in ("num_k", "peak_memory", "simulation_run_time", "total_threads"):
        try:
            log[key] = session.getresult("RCWA", key)
        except lumapi.LumApiError:
            continue
    if args.index:
        try:
            np.savez_compressed(run / "index.npz", **index_raster(session))
        except lumapi.LumApiError as exc:
            log["index_export_error"] = str(exc)
    for key in ("actual mesh cells x", "actual mesh cells z", "interface actual positions"):
        try:
            log[key] = session.getnamed("RCWA", key)
        except lumapi.LumApiError:
            continue
    json_write(run / "done.json", dict(case=g.name, ku=args.ku, kv=args.kv, du_nm=args.du,
                                     dv_nm=args.dv, angle_table_deg=table, fit_nk=nk, solve_s=solve_s,
                                     wall_s=time.monotonic()-started, results=log, lu_um=g.lu,
                                     lv_um=g.length_um, reference_y_um=.3, li=args.li))
    print(f"RCWA done {g.name} {suffix}: {solve_s:.2f} s", flush=True)
    return run


def fdtd_script(g, args, mesh):
    du, dv, dw = mesh
    finite = args.finite
    lu, lv = ((16., 16.) if args.compact else (20.,20.)) if finite else (g.lu,g.length_um)
    window = 14.0 if args.compact else 17.6
    bc = "PML" if finite else "Periodic"
    text = [
        geometry_script(g, CONDUCTIVE if args.conductive else MATERIAL, lu, lv, finite, args.offset),
        'addfdtd;set("dimension","3D");',
        position("x span",lu), position("z span",lv), position("y min",-.65), position("y max",1.3),
        'set("mesh type","auto non-uniform");set("mesh accuracy",2);'
        'set("mesh refinement","conformal variant 0");',
        f'set("x min bc","{bc}");set("x max bc","{bc}");'
        f'set("z min bc","{bc}");set("z max bc","{bc}");'
        'set("y min bc","PML");set("y max bc","PML");'
        'set("pml profile",1);set("pml layers",16);set("dt stability factor",.90);'
        f'set("auto shutoff min",{args.shutoff:.12g});'
        f'set("simulation time",{args.fs}e-15);',
    ]
    # Fine masks are keyed to the entire reference footprint, with matching mesh in
    # smooth/rough comparisons and in the two finite end-length comparisons.
    for name, x_span, z_span, y_low, y_high, axes in (
            ("lateral_grid", g.lu+.5 if finite else lu+1, 12.6 if finite else lv+1, -2., 2., (1,0,1)),
            ("surface_grid", g.lu+.5 if finite else lu+1, 12.6 if finite else lv+1, -.3, .5, (0,1,0))):
        text.extend((f'addmesh;set("name","{name}");', position("x span", x_span),
                     position("z span",z_span), position("y min",y_low),position("y max",y_high),
                     f'set("override x mesh",{axes[0]});set("override y mesh",{axes[1]});'
                     f'set("override z mesh",{axes[2]});',
                     f'set("dx",{du}e-9);' if axes[0] else "",
                     f'set("dy",{dw}e-9);' if axes[1] else "",
                     f'set("dz",{dv}e-9);' if axes[2] else ""))
    if finite:
        text.extend(('addgaussian;set("injection axis","y-axis");set("use scalar approximation",1);'
                     'set("waist radius w0",3.2e-6);set("distance from waist",-.35e-6);',
                     position("x",-args.shift), position("x span",window), position("z span",window)))
    else:
        text.extend(('addplane;set("injection axis","y-axis");', position("x span", lu+.2), position("z span",lv+.2)))
    text.extend(('set("name","illumination");set("direction","backward");',
                 position("y",.95), 'set("polarization angle",0);set("amplitude",1);'
                 'set("wavelength start",623e-9);set("wavelength stop",663e-9);',
                 'adddftmonitor;set("name","reflected_plane");set("monitor type","2D Y-normal");',
                 position("x span",window if finite else lu),position("z span",window if finite else lv),
                 position("y",REFERENCE_Y),
                 'set("override global monitor settings",1);set("sample spacing","custom");'
                 'set("custom frequency samples",c/643e-9);set("spatial interpolation","specified position");'))
    if finite:
        text.append('set("down sample X",2);set("down sample Z",2);')
    return "\n".join(text)


def fdtd_solve(g, args, sim):
    mesh = MESHES[args.mesh]
    mesh = tuple(value * args.mesh_scale for value in mesh)
    mesh = tuple(override or value for override,value in zip((args.du,args.dv,args.dw),mesh))
    if args.finite:
        g = replace(g,stripes=args.stripes or 4)
    suffix = f"m{'x'.join(f'{value:g}' for value in mesh)}_t{args.fs:g}_o{args.offset:g}"
    if args.conductive:
        suffix += "_conductive"
    if args.shutoff != 1e-8:
        suffix += f"_shut{args.shutoff:g}"
    if args.finite:
        suffix += f"_v{args.finite:g}_s{args.shift:+g}"
        if args.compact:
            suffix +="_compact"
    run = ROOT / "runs" / "fdtd" / g.name / suffix
    run.mkdir(parents=True, exist_ok=True)
    if (run / "done.json").exists():
        return run
    g.save(ROOT / "geometry")
    fsp = run / "model.fsp"
    script = fdtd_script(g, args, mesh)
    (run / "build.lsf").write_text(script, encoding="ascii")
    sim.eval("switchtolayout;newproject;redrawoff;")
    if args.conductive:
        conductive_silicon(sim)
    evaluate_layout(sim, script, run / "build.lsf")
    sim.save(fsp.as_posix())
    devices = ("GPU", "CPU") if args.device == "auto" else (args.device,)
    for device in devices:
        try:
            check = sim.runsystemcheck("FDTD", device)
            json_write(run / (device.lower()+"_estimate.json"), check)
            print("FDTD system check", g.name, device, budget_estimate(run,device), "GiB", flush=True)
        except lumapi.LumApiError as exc:
            print("System check unavailable", device, str(exc), flush=True)
    if args.check:
        return run
    engine = INSTALL / "bin" / "fdtd-engine.exe"
    log = run / "model_p0.log"
    start = time.monotonic()
    device_used = None
    for device in devices:
        parsed = budget_estimate(run, device)
        cap = args.gpu_limit if device == "GPU" else min(args.cpu_limit,max(free_ram_gib()-2,0))
        if parsed is not None and parsed > cap:
            print(f"FDTD {device} estimate {parsed:.2f} GiB exceeds {cap:.2f} GiB", flush=True)
            continue
        command = [str(engine)] + (["-gpu"] if device=="GPU" else [])
        command += ["-t", str(args.threads), "-fullinfo", str(fsp)]
        print("FDTD", g.name, suffix, device, flush=True)
        with (run / f"engine_{device.lower()}.txt").open("wb") as stream:
            process = subprocess.run(command, cwd=run, stdout=stream, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW, check=False)
        raw = log.read_text(errors="replace") if log.exists() else ""
        ended = re.search(r"Completed\s+(\d+) iterations, or ([0-9.eE+-]+)s", raw)
        bad = re.search(r"(?im)^\s*ERROR\s*:|fields are diverging|simulation diverged", raw)
        if process.returncode == 0 and ended and not bad:
            device_used = device
            break
        if log.exists():
            (run / f"rejected_{device.lower()}.log").write_text(raw, encoding="utf-8")
        print(f"FDTD {device} did not complete, return code {process.returncode}", flush=True)
    if not device_used:
        raise RuntimeError(f"No successful FDTD device: {run}")
    levels = [float(value) for value in re.findall(r"Auto Shutoff:\s*([0-9.eE+-]+)", raw)]
    quiet = bool(re.search(r"(?:reached|early).*?auto shutoff|auto shutoff.*?threshold reached", raw, re.I))
    if not quiet and (not levels or levels[-1] > 2e-6):
        raise RuntimeError(f"FDTD transient had not decayed: {run}; sampled levels {levels[-3:]}")
    sim.load(fsp.as_posix())
    e = sim.getresult("reflected_plane","E")
    h = sim.getresult("reflected_plane","H")
    output = {f"{axis}_um": np.ravel(e[axis]) * 1e6 for axis in ("x","y","z")}
    output.update(E=np.squeeze(e["E"]), H=np.squeeze(h["H"]))
    output.update({f"mesh_{axis}_um":np.ravel(sim.getresult("FDTD",axis))*1e6 for axis in ("x","y","z")})
    if not np.isfinite(output["E"]).all() or not np.isfinite(output["H"]).all():
        raise ValueError("Non-finite field")
    material = CONDUCTIVE if args.conductive else MATERIAL
    fit = complex(np.asarray(sim.getfdtdindex(material,FREQUENCY,F_MIN,F_MAX)).item())
    np.savez_compressed(run / "raw.npz", **output)
    json_write(run / "done.json", dict(case=g.name, device=device_used, mesh_nm=mesh, fit_nk=[fit.real,fit.imag],
                                    lu_um=g.lu,lv_um=g.length_um,reference_y_um=float(output["y_um"][0]),
                                    finite_um=args.finite, stage_shift_um=args.shift, offset_um=args.offset,
                                    material=material, requested_shutoff=args.shutoff, compact=args.compact,
                                    solve_and_export_s=time.monotonic()-start, iterations=int(ended[1]),
                                    time_fs=1e15*float(ended[2]), last_logged_decay=levels[-1] if levels else None))
    print(f"FDTD done {g.name} {suffix}", flush=True)
    return run


def budget_estimate(run, device):
    path = run / (device.lower()+"_estimate.json")
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if device=="GPU":
        size = result.get("Approximate_GPU_Memory_Requirements",{}).get("Maximum_Bytes")
    else:
        size = result.get("Memory_Recommended_Bytes")
    return size/2**30 if size is not None else None


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    subs = root.add_subparsers(dest="solver", required=True)
    rc = subs.add_parser("rcwa")
    rc.add_argument("names", nargs="+")
    rc.add_argument("--ku", type=int, default=20, help="Native half-extent along X; 20 requests 41 harmonics")
    rc.add_argument("--kv", type=int, default=16, help="Native half-extent along Z; 16 requests 33 harmonics; 1 uses 2D")
    rc.add_argument("--du", type=float, default=4)
    rc.add_argument("--dv", type=float, default=20)
    rc.add_argument("--angles", type=int, choices=(1,3,5), default=1)
    rc.add_argument("--threads", type=int, default=8)
    rc.add_argument("--length", type=float)
    rc.add_argument("--stripes", type=int)
    rc.add_argument("--index", action="store_true")
    rc.add_argument("--li", action="store_true")
    fd = subs.add_parser("fdtd")
    fd.add_argument("names", nargs="+")
    fd.add_argument("--mesh", type=int, choices=(0,1,2), default=0)
    fd.add_argument("--mesh-scale", type=float, default=1)
    fd.add_argument("--du",type=float)
    fd.add_argument("--dv",type=float)
    fd.add_argument("--dw",type=float)
    fd.add_argument("--length", type=float)
    fd.add_argument("--stripes", type=int)
    fd.add_argument("--conductive", action="store_true")
    fd.add_argument("--fs", type=float, default=1000)
    fd.add_argument("--shutoff",type=float,default=1e-8)
    fd.add_argument("--threads", type=int, default=8)
    fd.add_argument("--device", choices=("auto","CPU","GPU"), default="auto")
    fd.add_argument("--gpu-limit", type=float, default=4.2)
    fd.add_argument("--cpu-limit", type=float, default=12)
    fd.add_argument("--finite", type=float, choices=(8.,12.))
    fd.add_argument("--compact",action="store_true")
    fd.add_argument("--shift", type=float, default=0.)
    fd.add_argument("--offset", type=float, default=0.)
    fd.add_argument("--check", action="store_true")
    return root


def main():
    args = parser().parse_args()
    os.environ["PATH"] = str(INSTALL / "bin") + os.pathsep + os.environ["PATH"]
    geometry = [selected(name) for name in args.names]
    if args.length or args.stripes:
        geometry = [replace(entry,length_um=args.length or entry.length_um,
                            stripes=args.stripes or entry.stripes) for entry in geometry]
    if args.solver == "rcwa":
        if args.kv == 1 and any(g.mode not in ("smooth","flat") for g in geometry):
            raise ValueError("2D RCWA is restricted to the z-invariant controls")
        with lumapi.FDTD(hide=True, serverArgs={"threads":str(args.threads)}) as session:
            for entry in geometry:
                rcwa_solve(session, entry, args)
    else:
        with lumapi.FDTD(hide=True,serverArgs={"threads":str(args.threads)}) as session:
            for entry in geometry:
                fdtd_solve(entry, args, session)


if __name__ == "__main__":
    main()
