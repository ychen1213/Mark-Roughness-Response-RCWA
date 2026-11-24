"""Continuous edge geometry for the RCWA response experiment. Internal lengths are µm."""

from dataclasses import asdict, dataclass, field, replace
import csv
from hashlib import sha256
import json
from pathlib import Path
import math
import os

import numpy as np

ROOT = Path(__file__).resolve().parent
PITCH = 3.2
WIDTH = 1.6
WAVELENGTH = .643
YPIN = .6
MASTER_MIN = .05


@dataclass(frozen=True)
class Profile:
    cd_nm: float = 0
    height_nm: float = 300
    swa_deg: float = 90
    shear_nm: float = 0
    footing_nm: float = 0

    @property
    def height(self):
        return self.height_nm / 1000

    @property
    def shaped(self):
        return self.swa_deg != 90 or self.shear_nm != 0 or self.footing_nm != 0


@dataclass(frozen=True)
class Mark:
    """Positive Fourier coefficients and DC, ordered (bottom/top, line, center/halfwidth, n)."""
    name: str
    length: float
    coeff: np.ndarray
    cutoff: float
    profile: Profile = field(default_factory=Profile)
    info: dict = field(default_factory=dict)
    rigid_nm: float = 0

    def __post_init__(self):
        a = np.asarray(self.coeff, dtype=np.complex128)
        if self.length <= 0 or self.cutoff <= 0 or self.profile.height <= 0:
            raise ValueError("Invalid length, cutoff or height")
        if a.ndim != 4 or a.shape[0] != 2 or a.shape[1] < 1 or a.shape[2] != 2 or a.shape[3] < 1:
            raise ValueError("Expected (bottom/top, line, center/halfwidth, Fourier mode)")
        if not np.isfinite(a).all() or np.any(a[...,0].imag):
            raise ValueError("Fourier coefficients must be finite with real DC")
        object.__setattr__(self,"coeff",a)

    @property
    def stripes(self):
        return self.coeff.shape[1]

    @property
    def lu(self):
        return PITCH * self.stripes

    @property
    def centers(self):
        return (np.arange(self.stripes)+.5-self.stripes/2)*PITCH

    @property
    def q(self):
        return 2*np.pi*np.arange(self.coeff.shape[-1])/self.length

    @property
    def active(self):
        n = min(self.coeff.shape[-1], 1+math.floor(self.length/self.cutoff+1e-9))
        return self.coeff[...,:n]

    @property
    def along_invariant(self):
        return not np.any(self.active[...,1:])

    @property
    def vertically_varying(self):
        return (self.profile.shaped or bool(self.info.get("height_envelope")) or
                not np.array_equal(self.active[0],self.active[1]))

    def height_coeff(self, height_fraction):
        if np.array_equal(self.active[0],self.active[1]) and not self.info.get("height_envelope"):
            return self.active[0]
        rho = float(self.info.get("rho_height",1))
        t = height_fraction
        rms = math.sqrt((1-t)**2+t*t+2*rho*t*(1-t))
        return ((1-t)*self.active[0]+t*self.active[1])/rms

    def bounds(self, z, t=.5):
        coef = self.height_coeff(t)
        wave = np.exp(1j*self.q[1:coef.shape[-1],None]*np.asarray(z)[None,:])
        dc = coef[...,0,None].real
        ripple = dc + 2*np.einsum("jcn,nv->jcv",coef[...,1:],wave).real
        center = (self.centers[:,None]+ripple[:,0]+self.rigid_nm/1000+
                  self.profile.shear_nm*(t-.5)/1000)
        slope = math.tan(math.radians(90-self.profile.swa_deg))
        foot = self.profile.footing_nm/1000 * max(1-t*self.profile.height/.05,0)**2
        half = (WIDTH/2+self.profile.cd_nm/2000+slope*self.profile.height*(.5-t)+foot+
                ripple[:,1])
        return center-half,center+half

    def signature(self):
        desc = (self.length, self.cutoff, asdict(self.profile), self.rigid_nm,
                self.info.get("rho_height",1))
        ids = sha256(json.dumps(desc,sort_keys=True).encode())
        if self.info.get("height_envelope"):
            ids.update(b"height_envelope")
        ids.update(np.ascontiguousarray(self.coeff).tobytes())
        return ids.hexdigest()

    def save(self,directory=ROOT/"geometry"):
        """Round trip the master and verify the polygon dimensions used by the solver."""
        directory = Path(directory)
        directory.mkdir(parents=True,exist_ok=True)
        dest = directory/f"{self.name}.npz"
        signature = self.signature()
        if dest.exists():
            with np.load(dest) as original:
                if str(original["signature"].item()) != signature:
                    raise ValueError(f"Existing input {dest.name} has a different geometry")
            return dest
        z = np.linspace(-self.length/2,self.length/2,1+math.ceil(self.length/min(.01,self.cutoff/12)))
        checks=[]
        for t in (0.,.5,1.):
            lo,hi = self.bounds(z,t)
            gap = np.vstack((lo[1:]-hi[:-1],lo[:1]+self.lu-hi[-1:]))
            if min((hi-lo).min(),gap.min()) <= 0:
                raise ValueError(f"Invalid stripe or gap in {self.name} at height fraction {t}")
            if max(np.max(abs(lo[:,0]-lo[:,-1])),np.max(abs(hi[:,0]-hi[:,-1])))>1e-10:
                raise ValueError(f"Periodic seam does not close in {self.name}")
            checks.append(dict(t=t,min_width_nm=float((hi-lo).min()*1000),
                               min_gap_nm=float(gap.min()*1000)))
        left,right = self.bounds(z)
        ridge_center = (left[:,:-1]+right[:,:-1])/2
        d = ((right-left)[:,:-1])/2
        meta = dict(name=self.name, length_um=self.length, cutoff_um=self.cutoff,
                    profile=asdict(self.profile), info=self.info, rigid_nm=self.rigid_nm,
                    checks=checks, center_rms_nm=(1000*ridge_center.std(axis=-1)).tolist(),
                    halfwidth_rms_nm=(1000*d.std(axis=-1)).tolist(),
                    left_rms_nm=(1000*left[:,:-1].std(axis=-1)).tolist(),
                    right_rms_nm=(1000*right[:,:-1].std(axis=-1)).tolist())
        tmp = directory/f".{self.name}.{os.getpid()}.npz"
        np.savez_compressed(tmp, coeff=self.coeff, z_um=z, left_um=left,right_um=right,
                            metadata=json.dumps(meta),signature=signature)
        tmp.replace(dest)
        return dest


def load_mark(path):
    with np.load(path) as d:
        m=json.loads(str(d["metadata"].item()))
        return Mark(m["name"], m["length_um"], d["coeff"], m["cutoff_um"],
                    Profile(**m["profile"]), m["info"], m["rigid_nm"])


def read_edge_csv(path,*,length=8.,origin=0.,cutoff=.5):
    """Fit arbitrary sampled periodic edges at explicitly declared physical frequencies."""
    path=Path(path)
    with path.open(encoding="utf-8-sig",newline="") as handle:
        rows=list(csv.DictReader(handle))
    if not rows:
        raise ValueError("The input edge table is empty")
    stripes=2 if "left2_um" in rows[0] else 1
    columns=[key for j in range(1,stripes+1) for key in (f"left{j}_um",f"right{j}_um")]
    z=np.array([float(r["z_um"])-origin for r in rows])
    xy=np.array([[float(r[key]) for key in columns] for r in rows])
    if not np.isfinite(z).all() or not np.isfinite(xy).all() or np.any(np.diff(z)<=0):
        raise ValueError("Use finite edge values with strictly ascending Z")
    if np.isclose(z[-1]-z[0],length,rtol=0,atol=1e-8):
        if not np.allclose(xy[0],xy[-1],atol=1e-7,rtol=0):
            raise ValueError("The supplied period endpoints do not match")
        z,xy=z[:-1],xy[:-1]
    if np.ptp(z)>=length or cutoff<=0:
        raise ValueError("Supply exactly one ordered period")
    n=int(math.floor(length/cutoff+1e-10))
    q=2*np.pi*np.arange(1,n+1)/length
    phase=z[:,None]*q
    design=np.c_[np.ones(len(z)),np.cos(phase),np.sin(phase)]
    lo,hi=xy[:,::2],xy[:,1::2]
    if np.any(hi<=lo):
        raise ValueError("Edge samples contain a crossed stripe")
    c=(lo+hi)/2-(np.arange(stripes)+.5-stripes/2)*PITCH
    d=(hi-lo)/2-WIDTH/2
    geometry=np.stack((c,d),axis=-1).reshape(len(z),2*stripes)
    beta,_,rank,sv=np.linalg.lstsq(design,geometry,rcond=None)
    if rank!=design.shape[1] or sv[0]/sv[-1]>1e6:
        raise ValueError("Sampling cannot resolve the requested periodic Fourier basis")
    coeff=beta[0].reshape(stripes,2)
    a=np.zeros((2,stripes,2,n+1),complex)
    a[...,1:]=(beta[1:n+1]-1j*beta[n+1:]).T.reshape(stripes,2,n)/2
    rounded=coeff.mean(axis=0)
    offsets=coeff-rounded
    a[...,0]=np.where(abs(offsets)>1e-10,offsets,0.)
    mean_cd=2*rounded[1]*1000
    profile=Profile(cd_nm=0. if abs(mean_cd)<1e-6 else float(mean_cd))
    return Mark(path.stem,length,a,cutoff,profile,
                dict(kind="sampled",source=str(path),fit_rms_nm=float(np.std(geometry-design@beta)*1000),
                     origin_um=origin,samples=len(z)),
                float(rounded[0]*1000))


def smooth(name="nominal",length=8.,stripes=2,profile=Profile(),shift_nm=0):
    return Mark(name,length,np.zeros((2,stripes,2,1)),length,profile,
                {"kind":"smooth"},shift_nm)


def chart_mark(cutoff=.1):
    """Reproduce the captured browser widget (LCG and Box–Muller, no sample rescaling)."""
    state = 1427
    def uniform():
        nonlocal state
        state = (1664525*state+1013904223) & 0xffffffff
        return (state+.5)/4294967296
    def gauss():
        r=math.sqrt(-2*math.log(uniform()))
        p=2*math.pi*uniform()
        return complex(r*math.cos(p),-r*math.sin(p))
    rho,xi=.5,.25
    n=np.arange(1,81)
    q=2*np.pi*n/8
    w=1/(1+(q*xi)**2)
    draws = np.array([[gauss(),gauss()] for _ in n]).T
    a = np.zeros((2,1,2,81),complex)
    # Widget z=4 µm is the simulation origin. Match its cosine/sine samples.
    c = ((.005/2)*np.sqrt(w/w.sum())*(-1.)**n)[None,:]*draws
    c *= np.sqrt((1+rho*np.array([1,-1]))/2)[:,None]
    a[:,0,:,1:] = c
    suffix="" if cutoff==.1 else f"_cut{cutoff:g}"
    return Mark("chart_s1427"+suffix,8.,a,cutoff,
                info=dict(kind="chart",seed=1427,sigma_nm=5.,xi_um=xi,hurst=.5,rho_lr=rho,
                          rho_line=1.,rho_height=1.,source_modes=80,chart_z_origin_um=4.))


def random_mark(name,*,seed,sigma_nm,xi_um,hurst=.5,rho_lr=0.,rho_line=0.,rho_height=1.,
                cutoff=.5,length=8.,stripes=2,profile=Profile(),master_min=MASTER_MIN):
    """Draw the same Gaussian master for every parameter combination with this seed."""
    if sigma_nm < 0 or xi_um <= 0 or not -1 <= rho_lr <= 1 or not -1 <= rho_line <= 1 or not 0 <= rho_height <= 1:
        raise ValueError("Invalid roughness inputs")
    n=int(math.floor(length/master_min+1e-10))
    q=2*np.pi*np.arange(1,n+1)/length
    w=(1+(q*xi_um)**2)**(-hurst-.5)
    rng=np.random.default_rng(seed)
    noise=rng.normal(size=(2,stripes,2,n,2))
    u=(noise[...,0]+1j*noise[...,1])/np.sqrt(2)
    # Two-stripe common/differential directions are stable even when |rho|=1.
    if stripes==2:
        cm=math.sqrt((1+rho_line)/2)*u[:,0]
        dm=math.sqrt((1-rho_line)/2)*u[:,1]
        u=np.stack((cm+dm,cm-dm),axis=1)
    elif stripes>2:
        distance=np.abs(np.arange(stripes)[:,None]-np.arange(stripes))
        eig,vec=np.linalg.eigh(rho_line**distance)
        u=np.einsum("ij,tjcn->ticn",(vec*np.sqrt(np.maximum(0,eig)))@vec.T,u)
    u[1]=rho_height*u[0]+math.sqrt(max(0,1-rho_height*rho_height))*u[1]
    phase_scale=np.sqrt((1+rho_lr*np.array([1,-1]))/2)
    weights=sigma_nm/1000*np.sqrt(w/(2*w.sum()))
    a=np.zeros((2,stripes,2,n+1),complex)
    a[...,1:]=u*phase_scale[None,None,:,None]*weights
    info=dict(kind="random",seed=seed,sigma_nm=sigma_nm,xi_um=xi_um,hurst=hurst,
              rho_lr=rho_lr,rho_line=rho_line,rho_height=rho_height,master_min_um=master_min)
    return Mark(name,length,a,cutoff,profile,info)


def modal_mark(name, terms, *, length=8., stripes=2, profile=Profile(), compact=True):
    """terms are (component, line pattern, full-cell n, cosine nm, phase rad, top sign)."""
    periods=[int(p[2]) for p in terms]
    if not periods or any(p<1 for p in periods):
        raise ValueError("A modal mark needs at least one positive spatial frequency")
    primitive=math.gcd(*periods) if compact else 1
    if stripes==2 and all(t[1]=="+" for t in terms) and compact:
        stripes=1
    a=np.zeros((2,stripes,2,max(periods)//primitive+1),complex)
    for component,pattern,n,nm,phase,top_sign in terms:
        vec=np.ones(stripes)
        if pattern=="-":
            if stripes!=2:
                raise ValueError("Differential mode requires a two-line cell")
            vec[1]=-1
        a[0,:,component,n//primitive] += nm/2000*np.exp(1j*phase)*vec
        a[1,:,component,n//primitive] += nm/2000*np.exp(1j*phase)*vec*top_sign
    info=dict(kind="mode",terms=terms,primitive_divisor=primitive,full_length_um=length,
              rho_height=1.)
    return Mark(name,length/primitive,a,length/max(periods),profile,info)


def from_legacy(name,path):
    with np.load(path) as d:
        source=d["master_um"]
        z=d["v_um"]
        length=float(z[-1]-z[0])
    a=np.zeros((2,source.shape[0],2,source.shape[1]+1),complex)
    a[:,:,int(name.startswith("width")),1:]=source
    return Mark("legacy_"+name,length,a,.5,info={"kind":"legacy","source":str(path)})

