"""RSI readout for phase-calibrated native fields and response predictions."""

from functools import lru_cache
from pathlib import Path
import json

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize_scalar
from scipy.special import erf

from rcwa_runner import PREVIOUS
from response_geometry import PITCH
from optics import K0, NA, STOP, Spectrum, RSI_ROTATION, load_rcwa, normal_component, pupil_nodes
from optics import metrics as scan_metrics
from optics import scan as separated_scan


@lru_cache(maxsize=48)
def load_spectrum(folder):
    folder=Path(folder)
    md=json.loads((folder/"done.json").read_text(encoding="utf-8"))
    count=round(len(md["angle_table_deg"])**.5)
    flat_name="k81x1_d2x20_a1_li" if count==1 else f"k2x2_d4x20_a{count}"
    flat=PREVIOUS/"runs"/"rcwa"/"flat"/flat_name
    source=load_rcwa(folder,flat)
    requested=md.get("requested_lu_um",source.lu)
    if requested==source.lu:
        return source
    ku=md["setting"]["ku"]
    l=np.arange(-ku,ku+1)
    fields=align_spectrum(source,requested,source.lv,l,source.n,source.incoming)
    return Spectrum(l,source.n,requested,source.lv,source.incoming,fields)


def native_fields(folder):
    """Copy normalized incident and outgoing coordinate arrays from cached native data."""
    return load_spectrum(Path(folder))


def source_grid(spectrum):
    q=np.round(spectrum.incoming,9)
    ax=np.unique(q[:,0])
    az=np.unique(q[:,1])
    if len(ax)<3 or len(ax)*len(az)!=len(q):
        raise ValueError("A complete incident angular table is required for a focused beam")
    order=np.lexsort((q[:,1],q[:,0]))
    field=spectrum.electric[order].reshape(len(ax),len(az),*spectrum.electric.shape[1:])
    return ax,az,field


def shared_pupil_nodes(gx,gz,ax,az,rule):
    """Integrate in output coordinates, partitioning at every input-block knot."""
    x,w=rule
    outer=NA*K0
    knots=(gz[:,None]+az[None,:]).ravel().tolist()
    for rad in (STOP,outer):
        for kx in gx+ax:
            if abs(kx)<rad:
                z=np.sqrt(rad*rad-kx*kx)
                knots.extend((-z,z))
    cuts=np.unique(np.clip(knots+[-outer,outer],-outer,outer))
    nodes=[]
    weights=[]
    for a,b in zip(cuts[:-1],cuts[1:]):
        midway=(a+b)/2
        if not np.any((az[0]<=midway-gz)&(midway-gz<=az[-1])):
            continue
        for kz,wz in zip(midway+x*(b-a)/2,w*(b-a)/2):
            radius=np.sqrt(outer*outer-kz*kz)
            left=max(ax[0],-gx-radius)
            right=min(ax[-1],-gx+radius)
            if abs(kz)<STOP:
                inner=np.sqrt(STOP*STOP-kz*kz)
                intervals=((left,min(right,-gx-inner)),(max(left,-gx+inner),right))
            else:
                intervals=((left,right),)
            for lo,hi in intervals:
                if hi<=lo:
                    continue
                boundaries=np.r_[lo,ax[(lo<ax)&(ax<hi)],hi]
                for x0,x1 in zip(boundaries[:-1],boundaries[1:]):
                    nodes.append(np.c_[(x0+x1)/2+x*(x1-x0)/2,np.full(len(x),kz)])
                    weights.append(w*wz*(x1-x0)/2)
    if not nodes:
        return np.empty((0,2)),np.empty(0)
    return np.concatenate(nodes),np.concatenate(weights)


def coherent_scan(spectrum,points=1024,beam=38.,quadrature=12):
    """Sum all contributing incoming plane waves at the same output k before detection.

    X diffraction blocks are separated in the configured 3.2 and 6.4 µm cells.
    Z blocks may overlap in the long-cell study. Grids are symmetric about k=0.
    """
    ax,az,field=source_grid(spectrum)
    gx=2*np.pi*spectrum.l/spectrum.lu
    gz=2*np.pi*spectrum.n/spectrum.lv
    if np.ptp(ax)>=2*np.pi/spectrum.lu-1e-10:
        raise ValueError("The optical implementation requires separated cross-stripe blocks")
    sigma=2/beam
    norm=2*np.pi*sigma*sigma*erf(ax[-1]/(np.sqrt(2)*sigma))*erf(az[-1]/(np.sqrt(2)*sigma))
    collection=0.
    overlap=np.zeros(len(gx),complex)
    rule=np.polynomial.legendre.leggauss(quadrature)
    active_z=np.where(abs(gz)<=NA*K0+max(abs(az)))[0]

    def superpose(i,qout):
        summed=np.zeros((len(qout),2),complex)
        for j in active_z:
            qz=qout[:,1]-gz[j]
            valid=(az[0]<=qz)&(qz<=az[-1])
            if not np.any(valid):
                continue
            q=np.c_[qout[valid,0],qz[valid]]
            radius2=np.einsum("ij,ij->i",q,q)
            source=np.exp(-radius2/(4*sigma*sigma))/np.sqrt(norm*np.sqrt(1-radius2/K0**2))
            vector=RegularGridInterpolator((ax,az),field[:,:,i,j][...,[0,2]],
                                           bounds_error=False,fill_value=None)(q)
            summed[valid]+=vector*source[:,None]
        return summed

    for i,offset in enumerate(gx):
        if abs(offset)>NA*K0+max(abs(ax)):
            continue
        q,weights=shared_pupil_nodes(offset,gz[active_z],ax,az,rule)
        if not len(q):
            continue
        kx,kz=q[:,0]+offset,q[:,1]
        root=np.sqrt(1-(kx*kx+kz*kz)/K0**2)
        forward=normal_component(superpose(i,q),kx,kz,root,1)
        mirror=normal_component(superpose(len(gx)-1-i,-q),-kx,-kz,root,1)*RSI_ROTATION
        weights*=root
        collection+=np.sum(weights*np.sum(abs(forward)**2,axis=1))
        overlap[i]=np.sum(weights*np.einsum("ij,ij->i",mirror.conj(),forward))
    positions=np.arange(points)*spectrum.lu/points-spectrum.lu/2
    interference=(overlap@np.exp(-2j*gx[:,None]*positions)).real
    plus=(collection+interference)/2
    minus=(collection-interference)/2
    if min(plus.min(),minus.min()) < -1e-10:
        raise ValueError("Coherent output ports are not positive")
    return positions,plus,minus,collection


def curves(spectrum,points=1024,quadrature=18,force_coherent=False):
    if len(spectrum.incoming)==1:
        return separated_scan(spectrum,points=points)
    angular_span=np.ptp(spectrum.incoming,axis=0)
    if force_coherent or max(angular_span)>=min(2*np.pi/spectrum.lu,2*np.pi/spectrum.lv)-1e-10:
        return coherent_scan(spectrum,points=points,quadrature=quadrature)
    return separated_scan(spectrum,points=points,beam=38.,quadrature=quadrature)


@lru_cache(maxsize=12)
def position_grid(count,length,capture_nm):
    frequencies=2*np.pi*np.fft.fftfreq(count,d=length/count)/1000
    grid=np.linspace(-capture_nm,capture_nm,141)
    scan=np.exp(1j*grid[:,None]*frequencies)
    return frequencies,grid,scan


def apd(positions,signal,template,true_nm=0.,capture_nm=350.):
    """Shift of periodic S0(x+d), fitting a positive gain and a constant offset."""
    y=np.asarray(signal)-np.mean(signal)
    r=np.asarray(template)-np.mean(template)
    if np.linalg.norm(r)<=1e-12 or np.linalg.norm(y)<=1e-4*np.linalg.norm(r):
        return dict(apd_nm=None,position_nm=None,position_valid=False,position_error="weak_modulation")
    length=round((positions[1]-positions[0])*len(positions),12)
    frequencies,grid,scans=position_grid(len(positions),length,capture_nm)
    cross=np.conj(np.fft.fft(y))*np.fft.fft(r)
    scale=len(positions)*np.linalg.norm(y)*np.linalg.norm(r)
    def correlation(d):
        return float(np.real(cross@np.exp(1j*frequencies*d))/scale)
    score=np.real(scans@cross)/scale
    i=int(score.argmax())
    if i in (0,len(grid)-1) or score[i]<=0:
        return dict(apd_nm=None,position_nm=None,position_valid=False,position_error="capture_bound")
    result=minimize_scalar(lambda d:-correlation(d),bounds=(grid[i-1],grid[i+1]),
                           method="bounded",options={"xatol":1e-6})
    shift=float(result.x)
    return dict(apd_nm=shift-true_nm,position_nm=shift,position_valid=bool(result.success),
                template_nrmse=float(np.sqrt(max(0.,1-correlation(shift)**2))))


def linear_apd(positions,signal,template):
    """Project a small distortion onto the shift Jacobian at zero translation."""
    y=np.asarray(signal)-np.mean(signal)
    r=np.asarray(template)-np.mean(template)
    energy=np.dot(r,r)
    if energy<=1e-24:
        return None
    gain=np.dot(y,r)/energy
    if gain<=0:
        return None
    frequency=2*np.pi*np.fft.fftfreq(len(positions),d=1000*(positions[1]-positions[0]))
    slope=np.fft.ifft(1j*frequency*np.fft.fft(r)).real
    slope-=r*np.dot(slope,r)/energy
    weight=np.dot(slope,slope)
    return float(np.dot(y-gain*r,slope)/(gain*weight)) if weight>1e-24 else None


def evaluate(port,reference,true_nm=0.):
    x,s,d,power=port
    tx,ts,td,_=reference
    if not np.allclose(x,tx,rtol=0,atol=1e-10):
        raise ValueError("Reference and sample scan coordinates differ")
    result=scan_metrics(x,s)
    ref=scan_metrics(tx,ts)
    if ref["carrier_abs"]<=1e-12:
        raise ValueError("Insufficient reference carrier")
    result["wq"]=result["carrier_abs"]/ref["carrier_abs"]
    result["collection"]=power
    result.update(apd(x,s,ts,true_nm))
    return result


def comparison(a,b):
    """Compare two ports on the same stage coordinates, using strict peak-to-peak scaling."""
    x,plus,minus,_=a
    y,rp,rm,_=b
    if not np.allclose(x,y,rtol=0,atol=1e-10):
        raise ValueError("Cannot compare different scan coordinates")
    return tuple(float(np.sqrt(np.mean((s-t)**2))/max(np.ptp(t),1e-12))
                 for s,t in ((plus,rp),(minus,rm)))


def align_spectrum(source, lu, lv, x_orders, z_orders, angles):
    """Embed a primitive geometry in the corresponding larger periodic cell."""
    divisor_x=lu/source.lu
    divisor_z=lv/source.lv
    ix=int(round(divisor_x))
    iz=int(round(divisor_z))
    if min(ix,iz)<1 or not np.allclose((divisor_x,divisor_z),(ix,iz),atol=1e-8):
        raise ValueError("Source and target cells are not integer subdivisions")
    if len(source.incoming)!=len(angles) or not np.allclose(source.incoming,angles,atol=1e-8):
        raise ValueError("Angular grids differ")
    ux=source.l*ix
    uz=source.n*iz
    keep_x=np.isin(ux,x_orders)
    keep_z=np.isin(uz,z_orders)
    x_idx=np.searchsorted(x_orders,ux[keep_x])
    z_idx=np.searchsorted(z_orders,uz[keep_z])
    field=np.zeros((len(angles),len(x_orders),len(z_orders),3),complex)
    # A centered primitive ridge moves by half its period in an even repeat count.
    phase=(-1.)**(source.l[keep_x]*(ix-1))
    field[:,x_idx[:,None],z_idx[None,:],:]=source.electric[:,keep_x][:,:,keep_z]*phase[None,:,None,None]
    return field


def interpolation_weights(q,ax,az):
    ix=np.clip(np.searchsorted(ax,q[:,0],side="right")-1,0,len(ax)-2)
    iz=np.clip(np.searchsorted(az,q[:,1],side="right")-1,0,len(az)-2)
    tx=(q[:,0]-ax[ix])/(ax[ix+1]-ax[ix])
    tz=(q[:,1]-az[iz])/(az[iz+1]-az[iz])
    weights=np.zeros((len(q),len(ax)*len(az)))
    rows=np.arange(len(q))
    for x,z,weight in ((0,0,(1-tx)*(1-tz)),(1,0,tx*(1-tz)),(0,1,(1-tx)*tz),(1,1,tx*tz)):
        weights[rows,(ix+x)*len(az)+iz+z]=weight
    return weights


class ScanOperator:
    """An exactly equivalent integrated quadratic form for separated angular blocks."""
    def __init__(self,lu,lv,x_orders,z_orders,angles,points=1024,quadrature=18):
        self.gx=2*np.pi*np.asarray(x_orders)/lu
        self.gz=2*np.pi*np.asarray(z_orders)/lv
        self.lu,self.lv=lu,lv
        self.angles=np.round(angles,9)
        self.points=points
        self.x=np.arange(points)*lu/points-lu/2
        self.shift=np.exp(-2j*self.gx[:,None]*self.x)
        self.sort=np.lexsort((self.angles[:,1],self.angles[:,0]))
        self.weight=self.build_weight(quadrature)

    def build_weight(self,quadrature):
        count=len(self.angles)
        q=np.round(self.angles[self.sort],9)
        ax,az=np.unique(q[:,0]),np.unique(q[:,1])
        if count>1 and (len(ax)*len(az)!=count or max(np.ptp(ax),np.ptp(az))>=min(2*np.pi/self.lu,2*np.pi/self.lv)):
            raise ValueError("Use coherent input-angle remapping for overlapping blocks")
        if count>1 and (not np.allclose(ax,-ax[::-1]) or not np.allclose(az,-az[::-1])):
            raise ValueError("ScanOperator assumes a symmetric angular table")
        w=np.zeros((len(self.gx),len(self.gz),2*count,2*count))
        sigma=2/38.
        rule=np.polynomial.legendre.leggauss(quadrature)
        norm=2*np.pi*sigma*sigma if count==1 else (
            2*np.pi*sigma*sigma*erf(ax[-1]/(np.sqrt(2)*sigma))*erf(az[-1]/(np.sqrt(2)*sigma)))
        for i,gx in enumerate(self.gx):
            for j,gz in enumerate(self.gz):
                if count==1:
                    radius2=gx*gx+gz*gz
                    if STOP*STOP<=radius2<=(K0*NA)**2:
                        r=np.sqrt(1-radius2/K0**2)
                        direction=np.array([gx,gz])/(K0*r)
                        w[i,j]=r*(np.eye(2)+np.outer(direction,direction))
                    continue
                points,weights=pupil_nodes(gx,gz,ax,az,rule)
                if not len(points):
                    continue
                tangential=points+np.array([gx,gz])
                q2=np.einsum("ka,ka->k",points,points)
                ky=np.sqrt(K0**2-np.einsum("ka,ka->k",tangential,tangential))
                weights*=np.exp(-q2/(2*sigma*sigma))*ky/(np.sqrt(K0**2-q2)*norm)
                basis=interpolation_weights(points,ax,az)
                u=tangential/ky[:,None]
                for a in (0,1):
                    for b in (0,1):
                        metric=weights*((a==b)+u[:,a]*u[:,b])
                        w[i,j,a::2,b::2]=basis.T@(metric[:,None]*basis)
        return w

    def components(self,field):
        t=field[self.sort][...,[0,2]]
        # Dimensions after reshape are (X order, Z order, (angle, component)).
        flat=t.transpose(1,2,0,3).reshape(*t.shape[1:3],-1)
        adj=t[::-1,::-1,::-1].transpose(1,2,0,3).reshape(*t.shape[1:3],-1)
        mixed=np.einsum("ijab,ijb->ija",self.weight,flat,optimize=True)
        power=float(np.vdot(flat,mixed).real)
        cross=np.einsum("ija,ija->i",adj.conj(),mixed)
        return power,cross

    def ports(self,components):
        power,cross=components
        s=(cross@self.shift).real
        return self.x,(power+s)/2,(power-s)/2,power

    def __call__(self,field):
        return self.ports(self.components(field))

