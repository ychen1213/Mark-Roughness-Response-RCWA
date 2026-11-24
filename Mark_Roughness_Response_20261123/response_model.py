"""A calibrated Fourier response library and single-geometry SUM/DIFF predictor."""

import argparse
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import json
import math
import os

import numpy as np

from design import MODES, kernel_jobs, pair_catalog, pair_tag, pair_jobs, reference_jobs
from rcwa_runner import REGISTRY
from readout import ScanOperator, align_spectrum, evaluate, native_fields
from response_geometry import PITCH, ROOT, chart_mark, load_mark, read_edge_csv

OUTPUT=ROOT/"analysis"


def add_components(a,b,scale=1.):
    return (a[0]+scale*b[0],a[1]+scale*b[1])


def analytic_field_orders():
    # Include the aperture-edge third parent order under 38 µm illumination.
    return np.arange(-6,7),np.arange(-8,9)


def keys(mark,vertical=False):
    """Physical exposure amplitudes in nm, complex phase relative to Z=0."""
    if mark.stripes not in (1,2):
        raise ValueError("The library accepts one or two stripes")
    if vertical:
        fields=((mark.active[0]+mark.active[1])/2,(mark.active[0]-mark.active[1])/2)
    else:
        fields=(mark.height_coeff(.5),)
    items={}
    for height,coeff in enumerate(fields):
        if mark.stripes==1:
            c=(coeff[0],np.zeros_like(coeff[0]))
        else:
            c=((coeff[0]+coeff[1])/2,(coeff[0]-coeff[1])/2)
        for component,pattern in MODES:
            for n,value in enumerate(c[0 if pattern=="+" else 1][component,1:],1):
                if abs(value)>1e-12:
                    items[component+2*height,pattern,n]=value*2000
    return items


class Response:
    def __init__(self,x_orders,z_orders,angles,base,linear,diagonal,mixed,metadata,amplitude_curves=None):
        self.l,self.n=np.asarray(x_orders),np.asarray(z_orders)
        self.angles,self.base=angles,base
        self.linear,self.diagonal,self.mixed=linear,diagonal,mixed
        self.meta=metadata
        self.amplitude_curves={} if amplitude_curves is None else amplitude_curves
        self._operators={}

    def operator(self,stripes):
        if stripes not in self._operators:
            order=self.l if stripes==2 else self.l[self.l%2==0]//2
            self._operators[stripes]=ScanOperator(PITCH*stripes,8.,order,self.n,self.angles)
        return self._operators[stripes]

    def cell(self,field,stripes):
        if stripes==2:
            return field
        order=self.l[self.l%2==0]//2
        # Undo the +P/2 shift used to align a primitive with the large cell.
        return field[:,self.l%2==0]*((-1.)**order)[None,:,None,None]

    def single(self,key,value):
        component,pattern,n=key
        amp=abs(value)/5.
        phase=np.angle(value)
        if key in self.linear:
            orient=np.exp(1j*phase*self.n/n)[None,None,:,None]
            return amp*self.linear[key]*orient,amp*amp*self.diagonal[key]*orient,True
        # For these short periods ±n and ±2n are outside the pupil. Interpolate
        # the retained n=0 second-order correction, not a scattered power curve.
        nodes=sorted(k[2] for k in self.diagonal if k[:2]==key[:2])
        if not nodes or n>nodes[-1] or n<nodes[0]:
            return np.zeros_like(self.base),np.zeros_like(self.base),False
        if not self.meta.get("interpolate_low") and n<=max(abs(self.n)):
            return np.zeros_like(self.base),np.zeros_like(self.base),False
        high=int(np.searchsorted(nodes,n))
        if high==0:
            return np.zeros_like(self.base),np.zeros_like(self.base),False
        lo,hi=nodes[high-1],nodes[high]
        weight=(n-lo)/(hi-lo)
        first=np.zeros_like(self.base)
        second=np.zeros_like(self.base)
        mid=len(self.n)//2
        left=self.diagonal[component,pattern,lo]
        right=self.diagonal[component,pattern,hi]
        second[:,:,mid]=(1-weight)*left[:,:,mid]+weight*right[:,:,mid]
        for order,tensor,source in ((1,first,self.linear),(2,second,self.diagonal)):
            for sign in (-1,1):
                k=mid+sign*n*order
                if 0<=k<len(self.n):
                    a,b=mid+sign*lo*order,mid+sign*hi*order
                    if 0<=a<len(self.n) and 0<=b<len(self.n):
                        rotated=(1-weight)*source[component,pattern,lo][:,:,a]+weight*source[component,pattern,hi][:,:,b]
                        tensor[:,:,k]=rotated*np.exp(1j*phase*sign*order)
        return amp*first,amp*amp*second,True

    def amplitude_delta(self,key,value,linear,diagonal):
        nodes,layers,curves=self.amplitude_curves[key]
        magnitude=abs(value)
        measured=np.clip(magnitude,nodes[0],nodes[-1])
        i=int(np.clip(np.searchsorted(nodes,measured),1,len(nodes)-1))
        weight=(measured**2-nodes[i-1]**2)/(nodes[i]**2-nodes[i-1]**2)
        first=(1-weight)*layers[i-1]+weight*layers[i]
        second=(1-weight)*curves[i-1]+weight*curves[i]
        orient=np.exp(1j*np.angle(value)*self.n/key[2])[None,None,:,None]
        scaled=magnitude/5.
        return orient*(scaled*first+scaled**2*second)-linear-diagonal

    def predict(self,mark,variants=("additive","linear","self_second","mixed_second")):
        if not np.isclose(mark.length,8.):
            raise ValueError("Library domain is an 8 µm along-stripe supercell")
        if np.any(abs(mark.active[...,0])>1e-12):
            raise ValueError("Set static offsets through the profile or rigid displacement")
        amplitudes=keys(mark,self.meta.get("vertical",False))
        weights=self.meta.get("joint_calibration",{}).get("scales")
        if variants==("additive","linear","self_second","mixed_second"):
            if weights is not None:
                variants=(*variants,"calibrated")
            if self.amplitude_curves:
                variants=(*variants,"amplitude_response")
        use_amplitudes="amplitude_response" in variants
        amplitude_adjustment=np.zeros_like(self.base) if use_amplitudes else None
        adjusted_modes=0
        capped_modes=0
        unsupported=[]
        op=self.operator(mark.stripes)
        field=lambda x:self.cell(x,mark.stripes)
        base_port=op.components(field(self.base))
        curve=base_port
        one=np.zeros_like(self.base)
        two=np.zeros_like(self.base)
        for key,value in amplitudes.items():
            lin,diag,valid=self.single(key,value)
            if not valid:
                unsupported.append(key)
                continue
            one+=lin
            two+=diag
            if use_amplitudes and key in self.amplitude_curves:
                amplitude_adjustment+=self.amplitude_delta(key,value,lin,diag)
                adjusted_modes+=1
                capped_modes+=int(abs(value)>self.amplitude_curves[key][0][-1])
            if "additive" in variants:
                curve=add_components(curve,op.components(field(self.base+lin+diag)))
                curve=add_components(curve,base_port,-1)
        joined=np.zeros_like(self.base)
        present=0
        for (i,j,si,sj),tensor in self.mixed.items():
            if i not in amplitudes or j not in amplitudes:
                continue
            a,b=amplitudes[i],amplitudes[j]
            ra=a if si==1 else a.conjugate()
            rb=b if sj==1 else b.conjugate()
            joined+=tensor*(ra*rb/25.)
            present+=1
        powers=dict(linear=self.base+one,self_second=self.base+one+two,
                    mixed_second=self.base+one+two+joined)
        if weights is not None:
            powers["calibrated"]=self.base+weights[0]*one+weights[1]*two+weights[2]*joined
        if use_amplitudes:
            powers["amplitude_response"]=powers["mixed_second"]+amplitude_adjustment
        result={}
        spectra={}
        translation=np.exp(-1j*2*np.pi*self.l*mark.rigid_nm/(1000*6.4))[None,:,None,None]
        translated_overlap=np.exp(-2j*op.gx*mark.rigid_nm/1000)
        for name in variants:
            if name=="additive":
                result[name]=op.ports((curve[0],curve[1]*translated_overlap))
            elif name in powers:
                spectra[name]=powers[name]*translation
                result[name]=op(field(spectra[name]))
            else:
                raise ValueError(f"Unknown response {name}")
        info=dict(terms=len(amplitudes),unsupported=[list(k) for k in unsupported],
                  quadratic_pairs_used=present,maximum_solo_nm=max(map(abs,amplitudes.values()),default=0.),
                  amplitude_adjusted_modes=adjusted_modes,amplitude_caps=capped_modes,
                  height_dependent=bool(mark.vertically_varying),
                  vertical_condition=self.meta.get("rho_height"),
                  profile_in_domain=mark.profile==type(mark.profile)())
        return op.ports(base_port),result,spectra,info

    def save(self,path):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        params=dict(self.meta)
        params["single_keys"]=[list(k) for k in self.linear]
        params["mixed_keys"]=[[list(i),list(j),si,sj] for i,j,si,sj in self.mixed]
        params["amplitude_keys"]=[list(k) for k in self.amplitude_curves]
        payload=dict(l=self.l,n=self.n,angles=self.angles,base=self.base,
                     metadata=json.dumps(params,ensure_ascii=False))
        for k,(name,value) in enumerate(self.linear.items()):
            payload[f"L{k}"]=value
            payload[f"D{k}"]=self.diagonal[name]
        for k,value in enumerate(self.mixed.values()):
            payload[f"C{k}"]=value
        for j,(nodes,linear,diag) in enumerate(self.amplitude_curves.values()):
            payload[f"A{j}"]=nodes
            payload[f"AL{j}"]=linear
            payload[f"AD{j}"]=diag
        temp=path.parent/f".{path.stem}_{os.getpid()}.npz"
        np.savez_compressed(temp,**payload)
        temp.replace(path)


def open_response(path):
    with np.load(path) as data:
        meta=json.loads(str(data["metadata"].item()))
        s=[tuple(k) for k in meta.pop("single_keys")]
        p=[(tuple(i),tuple(j),si,sj) for i,j,si,sj in meta.pop("mixed_keys")]
        a=[tuple(k) for k in meta.pop("amplitude_keys",[])]
        return Response(data["l"],data["n"],data["angles"],data["base"],
                        {key:data[f"L{i}"] for i,key in enumerate(s)},
                        {key:data[f"D{i}"] for i,key in enumerate(s)},
                        {key:data[f"C{i}"] for i,key in enumerate(p)},meta,
                        {key:(data[f"A{i}"],data[f"AL{i}"],data[f"AD{i}"]) for i,key in enumerate(a)})


def has_data(job):
    return (job.folder/"done.json").is_file()


def build_response(angles=3,ku=10,*,partial=False):
    singles=kernel_jobs(angles,ku)
    pair_set=pair_jobs(angles,ku)
    x,z=analytic_field_orders()
    reference_job=singles[1]
    if not has_data(reference_job):
        raise ValueError("The two-line reference has not completed")
    ref=native_fields(reference_job.folder)
    fix=ref.incoming
    sources=[str(reference_job.folder.relative_to(ROOT))]
    def read(job):
        sources.append(str(job.folder.relative_to(ROOT)))
        return align_spectrum(native_fields(job.folder),6.4,8.,x,z,fix)
    base=align_spectrum(ref,6.4,8.,x,z,fix)
    if has_data(singles[0]):
        primitive=read(singles[0])
        baseline_delta=float(np.linalg.norm(primitive-base)/np.linalg.norm(base))
    else:
        baseline_delta=None
    linear={}
    diagonal={}
    for job in singles:
        if job.group!="mode_bank" or not has_data(job):
            continue
        c,p,n,*_=job.mark.info["terms"][0]
        observed=read(job)
        k=(c,p,n)
        odd=(abs(z)==n)[None,None,:,None]
        even=((z==0)|(abs(z)==2*n))[None,None,:,None]
        linear[k]=np.where(odd,observed,0)
        diagonal[k]=np.where(even,observed-base,0)
    main={(c,p,n) for c,p in MODES for n in range(1,17)}
    if not partial and not main.issubset(linear):
        raise ValueError(f"{len(main-linear.keys())} core single-mode responses are missing")
    # Central mixed differences, using two native corner calculations and
    # reconstruction of the other two by exact common Z translation.
    groups=defaultdict(list)
    for job in pair_set:
        tag,sign=job.mark.name.rsplit("_",1)
        groups[tag].append((sign,job))
    cross={}
    tail=[]
    for i,n,j,m,phase in pair_catalog():
        tag=pair_tag(i,n,j,m,phase)
        pair=groups[tag]
        if len(pair)!=2 or not all(has_data(job) for _,job in pair):
            tail.append(tag)
            continue
        divisor=math.gcd(n,m)
        parity=(n//divisor%2,m//divisor%2)
        transfer=np.exp(1j*np.pi*z/divisor)[None,None,:,None]
        result=np.zeros_like(base)
        for signs,job in pair:
            a,b=[1 if c=="p" else -1 for c in signs]
            source=read(job)
            result+=a*b*source
            result+=a*b*(-1)**sum(parity)*source*transfer
        cross[i,n,j,m,phase]=result/4
    core_tail=[pair_tag(i,n,j,m,phase) for i,n,j,m,phase in pair_catalog()
               if max(n,m)<=16 and (i,n,j,m,phase) not in cross]
    if not partial and core_tail:
        raise ValueError(f"{len(core_tail)} core cross terms awaiting calibration")
    mixed={}
    for (i,n,j,m,phase),reference in cross.items():
        if phase!=0:
            continue
        first,second=(*i,n),(*j,m)
        for a,b in ((1,1),(1,-1),(-1,1),(-1,-1)):
            shifted=a*n+b*m
            if shifted not in z:
                continue
            use=reference
            if n==m and shifted==0:
                quadrature=cross.get((i,n,j,m,np.pi/2))
                if quadrature is None:
                    continue
                use=(reference+(-b)*1j*quadrature)/2
            mixed[first,second,a,b]=np.where((z==shifted)[None,None,:,None],use,0.)
    samples={}
    for job in singles:
        if job.group!="amplitude_curve" or not has_data(job):
            continue
        component,pattern,frequency,peak,*_=job.mark.info["terms"][0]
        key=(component,pattern,frequency)
        if key not in linear:
            continue
        values=samples.setdefault(key,{5.:(linear[key],diagonal[key])})
        observed=read(job)
        first=np.where((abs(z)==frequency)[None,None,:,None],observed,0.)*5./peak
        second=np.where(((z==0)|(abs(z)==2*frequency))[None,None,:,None],observed-base,0.)*(5./peak)**2
        values[peak]=(first,second)
    amplitude_curves={}
    for key,values in samples.items():
        if len(values)!=4:
            continue
        nodes=sorted(values)
        linear_values,second_values=zip(*(values[n] for n in nodes))
        amplitude_curves[key]=(np.asarray(nodes),np.stack(linear_values),np.stack(second_values))
    metadata=dict(library_version=2,angles=angles,ku_full=ku,
                  partial=partial,source_runs=sorted(set(sources)),
                  core_missing=len(main-linear.keys()),cross_missing=len(tail),core_cross_missing=len(core_tail),
                  compact_smooth_field_error=baseline_delta,
                  amplitude_nm=5.,training_domain="8 um periodic, nominal cross section, height-invariant")
    return Response(x,z,fix,base,linear,diagonal,mixed,metadata,amplitude_curves)


def calibrate_joint(model,*,partial=False):
    planned=reference_jobs(model.meta["angles"])
    # Training metadata and responses only. Height-dependent profiles have their own basis.
    selected=[j for j in planned if not j.mark.vertically_varying]
    rows=[j for j in selected if has_data(j)]
    if len(rows)<len(selected) and not partial:
        raise ValueError(f"Joint calibration needs {len(selected)-len(rows)} remaining height-invariant training cases")
    normal=np.zeros((3,3))
    target=np.zeros(3)
    chosen=[]
    for job in rows:
        spec=native_fields(job.folder)
        real=align_spectrum(spec,6.4,8.,model.l,model.n,model.angles)-model.base
        _,_,response,info=model.predict(job.mark,variants=("linear","self_second","mixed_second"))
        if info["unsupported"]:
            continue
        first,second,third=(response[name] for name in ("linear","self_second","mixed_second"))
        matrix=np.stack((first-model.base,second-first,third-second)).reshape(3,-1)
        normal+=np.real(matrix.conj()@matrix.T)
        target+=(matrix.conj()@real.ravel()).real
        chosen.append(str(job.folder.relative_to(ROOT)))
    if not chosen:
        raise ValueError("No joint training data with the fitted basis")
    penalty=.05*np.maximum(np.diag(normal),1e-14)
    estimate=np.linalg.solve(normal+np.diag(penalty),target+penalty)
    model.meta["joint_calibration"]=dict(scales=np.clip(estimate,0,3).tolist(),
                                         training_folders=chosen,partial=partial)
    return model


def save_prediction(mark,model,path):
    reference,ports,spectra,info=model.predict(mark)
    records={name:evaluate(port,reference,mark.rigid_nm) for name,port in ports.items()}
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    payload={"positions_um":reference[0],"sum_reference":reference[1],"diff_reference":reference[2],
             "result":json.dumps({"case":mark.name,"geometry_hash":mark.signature(),"kpi":records,
                                  "domain":info},ensure_ascii=False)}
    for key,output in ports.items():
        payload[f"sum_{key}"]=output[1]
        payload[f"diff_{key}"]=output[2]
    np.savez_compressed(path,**payload)
    return dict(case=mark.name,quality=records,domain=info)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    s=p.add_subparsers(dest="command",required=True)
    fit=s.add_parser("fit")
    fit.add_argument("--angles",type=int,choices=(1,3,5),default=3)
    fit.add_argument("--ku",type=int,default=10)
    fit.add_argument("--partial",action="store_true")
    fit.add_argument("--calibrate",action="store_true")
    fit.add_argument("--out",type=Path,default=OUTPUT/"response_model.npz")
    pred=s.add_parser("predict")
    pred.add_argument("geometry",help="chart, saved geometry NPZ, or periodic sampled edge CSV")
    pred.add_argument("--model",type=Path,default=OUTPUT/"response_model.npz")
    pred.add_argument("--out",type=Path,default=OUTPUT/"predicted_curve.npz")
    pred.add_argument("--length",type=float,default=8.)
    pred.add_argument("--origin",type=float,default=0.)
    pred.add_argument("--cutoff",type=float,default=.5)
    args=p.parse_args()
    if args.command=="fit":
        if args.out.is_file() and not args.partial:
            raise FileExistsError(f"Frozen library already exists: {args.out}")
        response=build_response(args.angles,args.ku,partial=args.partial)
        if args.calibrate:
            if args.ku!=10:
                raise ValueError("Joint calibration requires matched native X harmonics")
            calibrate_joint(response,partial=args.partial)
        response.save(args.out)
        digest=sha256(args.out.read_bytes()).hexdigest()
        print(f"Response library with {len(response.linear)} single modes and {len(response.mixed)} mixed terms")
        print(f"SHA-256 {digest} {args.out}")
    else:
        if args.geometry=="chart":
            mark=chart_mark()
        elif Path(args.geometry).suffix.lower()==".csv":
            mark=read_edge_csv(args.geometry,length=args.length,origin=args.origin,cutoff=args.cutoff)
        else:
            mark=load_mark(args.geometry)
        print(json.dumps(save_prediction(mark,open_response(args.model),args.out),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()

