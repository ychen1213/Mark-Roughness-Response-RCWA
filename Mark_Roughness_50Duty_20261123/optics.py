"""Common flux-normalized angular spectrum, RSI detection, and KPI extraction."""

import json
from dataclasses import dataclass

import numpy as np
from scipy.special import erf
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import CZT

from geometry import PITCH, REFERENCE_Y, WAVELENGTH

K0 = 2 * np.pi / WAVELENGTH
NA = .6
STOP = .2
Z0 = 376.730313668
RSI_ROTATION = np.array([1., -1., 1.])


@dataclass
class Spectrum:
    l: np.ndarray
    n: np.ndarray
    lu: float
    lv: float
    incoming: np.ndarray       # angle, (kx,kz); rad/um
    electric: np.ndarray       # angle, l, n, (Ex,Ey,Ez); per unit incident E


def order_axes(d):
    l = d["field_n"].ravel().astype(int)
    n = d["field_m"].ravel().astype(int)
    ix, iz = int(np.where(l == 0)[0][0]), int(np.where(n == 0)[0][0])
    incident = K0 * np.column_stack((d["field_kx"][0,:,ix,iz], d["field_kz"][0,:,ix,iz]))
    return l, n, incident


def fresnel_tangential(kappa, substrate):
    b = np.asarray(kappa) / K0
    radial = b @ b
    y1, y2 = np.sqrt(1-radial), np.sqrt(substrate**2-radial)
    rs = (y1-y2) / (y1+y2)
    rp = (1/y1-substrate**2/y2) / (1/y1+substrate**2/y2)
    if not radial:
        return rs * np.eye(2)
    return rs * np.eye(2) + (rp-rs) * np.outer(b,b) / radial


def flat_calibration(path):
    with np.load(path / "raw.npz") as d:
        l, n, incident = order_axes(d)
        ix, iz = np.where(l == 0)[0][0], np.where(n == 0)[0][0]
        substrate = complex(d["sub_n_lower"].item())
        basis = np.stack([np.stack([d[f"field_ek{axis}_f{pol}"][0,:,ix,iz] for axis in "xz"],axis=-1)
                          for pol in "sp"],axis=-1)
    incoming_basis = np.stack([np.linalg.solve(fresnel_tangential(k,substrate),b)
                               for k,b in zip(incident,basis)])
    q = incident / K0
    b_y = (q[:,0,None] * incoming_basis[:,0] + q[:,1,None] * incoming_basis[:,1]) / np.sqrt(1-np.sum(q**2,axis=1))[:,None]
    b = np.stack([incoming_basis[:,0],b_y,incoming_basis[:,1]],axis=1)
    gram = np.einsum("aip,aiq->apq",b.conj(),b)
    error = float(np.max(np.abs(gram-np.eye(2))))
    if error > 1e-5:
        raise ValueError(f"Native s/p incident basis failed flat-interface normalization check: {error}")
    return incident, incoming_basis, error


def load_rcwa(path, flat_path):
    """Calibrate incident basis on a planar interface; use outgoing E, not powers."""
    metadata = json.loads((path / "done.json").read_text(encoding="utf-8"))
    with np.load(path / "raw.npz") as d:
        l, n, incident = order_axes(d)
        field = np.stack([np.stack([d[f"field_ek{axis}_f{pol}"][0] for axis in "xyz"],axis=-1)
                          for pol in "sp"],axis=-1)
    # In the native reflected amplitudes only the s channel uses a reversed normal component.
    field[...,1,0] *= -1
    cal_k, cal_basis, _ = flat_calibration(flat_path)
    mapping = np.abs(incident[:,None,:]-cal_k[None,:,:]).max(axis=2)
    indices = mapping.argmin(axis=1)
    if mapping[np.arange(len(incident)),indices].max() > 1e-7:
        raise ValueError("Flat calibration does not contain the excitation directions")
    q = incident / K0
    k_inc = np.column_stack((q[:,0], -np.sqrt(1-np.sum(q**2,axis=1)), q[:,1]))
    input_e = np.array([0.,0.,1.]) - k_inc * k_inc[:,2,None]
    input_e /= np.linalg.norm(input_e,axis=1)[:,None]
    source_coefficients = np.linalg.solve(cal_basis[indices],input_e[:,[0,2],None])[...,0]
    field = np.einsum("alncp,ap->alnc",field,source_coefficients)
    lu, lv = metadata["lu_um"], metadata["lv_um"]
    kx = incident[:,0,None,None] + (2*np.pi*l/lu)[None,:,None]
    kz = incident[:,1,None,None] + (2*np.pi*n/lv)[None,None,:]
    ky = np.sqrt((K0**2-kx**2-kz**2).astype(complex))
    propagating = ky.imag==0
    divergence = kx*field[...,0]+ky*field[...,1]+kz*field[...,2]
    if np.linalg.norm(divergence[propagating]) > 1e-8*K0*np.linalg.norm(field[propagating]):
        raise ValueError("RCWA reflected field did not match its physical wavevector")
    reference = (REFERENCE_Y-metadata["reference_y_um"])
    phase = np.exp(1j * reference * (ky-K0*k_inc[:,1,None,None]))
    # The native 2025 R2 field amplitudes use the cell corner as lateral origin.
    phase *= (-1.)**(l[:,None]+n[None,:])
    return Spectrum(l,n,lu,lv,incident,field * phase[...,None])


def normal_component(tangential, kx, kz, root, direction):
    qx, qz = kx / K0, kz / K0
    e_y = -direction * (qx*tangential[...,0]+qz*tangential[...,1]) / root
    return np.stack((tangential[...,0],e_y,tangential[...,1]),axis=-1)


def split_air(e, h, kx, kz):
    """Separate up/down waves on an air monitor with matching E/H coordinates."""
    qx, qz = np.broadcast_arrays(kx / K0,kz / K0)
    q = np.stack((qx,qz),axis=-1)
    root = np.sqrt(np.maximum(1-qx*qx-qz*qz,1e-14))
    jt_h = Z0 * np.stack((-h[...,2], h[...,0]),axis=-1)
    directional = (jt_h - q * np.sum(q*jt_h,axis=-1,keepdims=True)) / root[...,None]
    tangential = e[...,[0,2]]
    up = normal_component((tangential+directional)/2,kx,kz,root,1)
    dn = normal_component((tangential-directional)/2,kx,kz,root,-1)
    return up, dn


def periodic_integral_axis(nodes, period, modes, array, axis):
    grid = np.asarray(nodes)
    if abs(grid[-1]-grid[0]-period) < 1e-6:
        grid = grid[:-1]
        array = np.take(array, np.arange(len(grid)), axis=axis)
    before = np.r_[grid[-1]-period,grid[:-1]]
    after = np.r_[grid[1:],grid[0]+period]
    weights = (after-before)/(2*period)
    if np.any(weights <= 0) or not np.isclose(weights.sum(),1):
        raise ValueError("Bad periodic monitor quadrature")
    basis = np.exp(-2j*np.pi*np.asarray(modes)[:,None]*grid/period)*weights
    return np.moveaxis(np.tensordot(basis,array,axes=(1,axis)),0,axis)


def load_fdtd(path):
    info = json.loads((path / "done.json").read_text(encoding="utf-8"))
    lu, lv = info["lu_um"],info["lv_um"]
    a, b = int(lu/WAVELENGTH),int(lv/WAVELENGTH)
    l, n = np.arange(-a,a+1),np.arange(-b,b+1)
    with np.load(path / "raw.npz") as d:
        spectra = []
        for key in ("E","H"):
            x = periodic_integral_axis(d["x_um"],lu,l,d[key],axis=0)
            spectra.append(periodic_integral_axis(d["z_um"],lv,n,x,axis=1))
    kx, kz = (2*np.pi*l/lu)[:,None],(2*np.pi*n/lv)[None,:]
    outgoing, incoming = split_air(*spectra,kx,kz)
    amp = incoming[a,b,2]
    if abs(amp) < .25 or abs(incoming[a,b,0]) > .01 * abs(amp):
        raise ValueError(f"Unexpected FDTD incident basis or amplitude: {incoming[a,b]}")
    mask = kx*kx + kz*kz < K0*K0
    field = np.where(mask[...,None],outgoing/amp,0)
    offset = info.get("offset_um",0.)
    field *= np.exp(1j*kx*offset)[...,None]
    return Spectrum(l,n,lu,lv,np.zeros((1,2)),field[None])


def flux(directions, electric, x, z):
    ky = np.sqrt(np.maximum(0.,1-(x/K0)**2-(z/K0)**2))
    ky_inc = np.sqrt(1-np.sum((directions/K0)**2,axis=1))
    return electric * np.sqrt(ky/ky_inc[:,None,None])[...,None]


def pupil_nodes(gu, gv, ax, az, quadrature):
    """Integrate continuous incident angle inside the circular pupil and stop."""
    abscissa, weights = quadrature
    outer = NA*K0
    corners = az.tolist()
    for radius in (STOP,outer):
        for ray in np.r_[gu+ax,0]:
            if abs(ray) < radius:
                root = np.sqrt(radius**2-ray**2)
                corners.extend(value for value in (-gv-root,-gv+root) if az[0]<value<az[-1])
    corners = np.unique(corners)
    nodes,out_weights=[],[]
    for za,zb in zip(corners[:-1],corners[1:]):
        z_values = (za+zb)/2+(zb-za)/2*abscissa
        for zv,zw in zip(z_values,weights*(zb-za)/2):
            y = gv+zv
            if abs(y) >= outer:
                continue
            radius = np.sqrt(outer**2-y**2)
            broad = max(ax[0],-gu-radius),min(ax[-1],-gu+radius)
            if abs(y) < STOP:
                blocked = np.sqrt(STOP**2-y**2)
                intervals = ((broad[0],min(broad[1],-gu-blocked)),
                             (max(broad[0],-gu+blocked),broad[1]))
            else:
                intervals = (broad,)
            for low,high in intervals:
                if high <= low:
                    continue
                xbounds = np.r_[low,ax[(ax>low)&(ax<high)],high]
                for a,b in zip(xbounds[:-1],xbounds[1:]):
                    x = (a+b)/2+(b-a)/2*abscissa
                    nodes.append(np.column_stack((x,np.full(len(x),zv))))
                    out_weights.append(weights*(b-a)/2*zw)
    if not nodes:
        return np.empty((0,2)),np.empty(0)
    return np.concatenate(nodes),np.concatenate(out_weights)


def gaussian_ports(spectrum, gu, gv, values, beam, nodes):
    angles = np.round(spectrum.incoming,9)
    ax,az = np.unique(angles[:,0]),np.unique(angles[:,1])
    if min(len(ax),len(az)) < 3 or len(ax)*len(az) != len(angles):
        raise ValueError(f"Expected a complete Cartesian grid of incidence, got {angles}")
    if not np.allclose(ax,-ax[::-1]) or not np.allclose(az,-az[::-1]):
        raise ValueError("Incident angle domain is not symmetric")
    if max(np.ptp(ax),np.ptp(az)) >= min(2*np.pi/spectrum.lu,2*np.pi/spectrum.lv):
        raise ValueError("Incident blocks overlap; use coherent remapping for this supercell")
    order = np.lexsort((angles[:,1],angles[:,0]))
    grid = values[order].reshape(len(ax),len(az),*values.shape[1:])
    sigma = 2/beam
    a,b = ax[-1],az[-1]
    normalization = (2*np.pi*sigma**2 * erf(a/(np.sqrt(2)*sigma)) * erf(b/(np.sqrt(2)*sigma)))
    rule = np.polynomial.legendre.leggauss(nodes)
    power = 0.
    overlap = np.zeros(len(gu),complex)
    for i,gx in enumerate(gu):
        if min(abs(gx+ax[0]),abs(gx+ax[-1])) > NA*K0 and gx:
            continue
        for j,gz in enumerate(gv):
            q,w = pupil_nodes(gx,gz,ax,az,rule)
            if not len(q):
                continue
            kx,kz = q[:,0]+gx,q[:,1]+gz
            r2 = kx*kx+kz*kz
            root = np.sqrt(1-r2/K0**2)
            tangential = RegularGridInterpolator((ax,az),grid[:,:,i,j][...,[0,2]])(q)
            mirror_tangential = RegularGridInterpolator((ax,az),grid[:,:,len(gu)-1-i,len(gv)-1-j][...,[0,2]])(-q)
            field = normal_component(tangential,kx,kz,root,1)
            mirror = normal_component(mirror_tangential,-kx,-kz,root,1)
            flux_ratio = np.sqrt((K0**2-r2)/(K0**2-np.sum(q*q,axis=1)))
            w *= np.exp(-np.sum(q*q,axis=1)/(2*sigma**2))*flux_ratio / normalization
            power += float(np.sum(w*np.sum(abs(field)**2,axis=1)))
            overlap[i] += np.sum(w*np.sum(field*np.conj(mirror*RSI_ROTATION),axis=1))
    return power,overlap


def scan(spectrum, points=512, beam=None, quadrature=16, omit_parent_third=False):
    """Full-pupil SUM/DIFF for periodic structure and one coherent beam."""
    l0,n0 = spectrum.l,spectrum.n
    gu0,gv0 = 2*np.pi*l0/spectrum.lu,2*np.pi*n0/spectrum.lv
    radius = K0*NA + np.max(np.abs(spectrum.incoming))
    keep_u,keep_v = np.abs(gu0) <= radius,np.abs(gv0) <= radius
    l,n,gu,gv = l0[keep_u],n0[keep_v],gu0[keep_u],gv0[keep_v]
    values = spectrum.electric[:,keep_u][:,:,keep_v]
    if omit_parent_third:
        values[:,np.isclose(abs(gu),6*np.pi/PITCH)] = 0
    if beam is not None:
        power,by_l = gaussian_ports(spectrum,gu,gv,values,beam,quadrature)
    else:
        normal = int(np.linalg.norm(spectrum.incoming,axis=1).argmin())
        field = values[normal:normal+1]
        kappa = np.zeros((1,2))
        kx = gu[None,:,None]
        kz = gv[None,None,:]
        r2 = kx*kx+kz*kz
        pupil = (STOP**2 <= r2) & (r2 <= (NA*K0)**2)
        tray = np.where(pupil[...,None],flux(kappa,field,kx,kz),0.)
        power = float(np.sum(abs(tray)**2))
        overlap = np.sum(tray*np.conj(tray[:,::-1,::-1]*RSI_ROTATION),axis=-1)
        by_l = np.sum(overlap,axis=(0,2))
    s = np.arange(points)*spectrum.lu/points-spectrum.lu/2
    term = np.real(by_l @ np.exp(-2j*gu[:,None]*s))
    plus,minus = (power+term)/2,(power-term)/2
    if min(plus.min(),minus.min()) < -1e-10:
        raise ValueError("Negative port power")
    return s,plus,minus,power


def metrics(s, signal):
    lo,hi = float(np.min(signal)),float(np.max(signal))
    if lo+hi <= 1e-12:
        raise ValueError("Power too small for DOM")
    carrier = 2*np.mean(signal*np.exp(-4j*np.pi*s/PITCH))
    return dict(carrier_abs=float(abs(carrier)),carrier_re=float(carrier.real),carrier_im=float(carrier.imag),
                dom=(hi-lo)/(hi+lo),minimum=lo,maximum=hi,mean=float(np.mean(signal)))


def finite_port(path, sample=.04, aperture=8.0, dk=.02):
    """Full-vector air-plane transform sampled densely at the pupil and stop."""
    info = json.loads((path / "done.json").read_text(encoding="utf-8"))
    k = np.arange(-int(K0*NA/dk)-1,int(K0*NA/dk)+2)*dk
    with np.load(path / "raw.npz") as d:
        aperture = min(aperture,float(min(-d["x_um"][0],d["x_um"][-1],-d["z_um"][0],d["z_um"][-1]))-.04)
        n = round(2*aperture/sample)
        dx = 2*aperture/n
        coordinate = -aperture+dx*np.arange(n)
        x,z = np.meshgrid(coordinate,coordinate,indexing="ij")
        czt = CZT(n,len(k),w=np.exp(-1j*dk*dx),a=np.exp(1j*k[0]*dx))
        input_arrays = []
        for key in ("E","H"):
            regrid = RegularGridInterpolator((d["x_um"],d["z_um"]),d[key])
            array = regrid(np.stack((x,z),axis=-1))
            transform = czt(czt(array,axis=0),axis=1)
            input_arrays.append(transform*dx**2)
    kx,kz = k[:,None],k[None,:]
    outgoing,incoming = split_air(*input_arrays,kx,kz)
    r2 = kx*kx+kz*kz
    root = np.sqrt(np.maximum(0,1-r2/K0**2))
    fraction = np.zeros_like(r2)
    for sx in np.arange(-2,3)*dk/5:
        for sz in np.arange(-2,3)*dk/5:
            shifted = (kx+sx)**2+(kz+sz)**2
            fraction += (shifted >= STOP**2) & (shifted <= (NA*K0)**2)
    fraction /= 25
    input_power = float(np.sum(np.sum(np.abs(incoming)**2,axis=-1)*root) * dk**2/(2*np.pi)**2)
    phase = np.exp(1j*(kx+kz)*aperture - 1j*kx*info["stage_shift_um"])
    field = outgoing * (phase*np.sqrt(root*fraction))[...,None]
    pair = field[::-1,::-1]*RSI_ROTATION
    common = dk**2 / ((2*np.pi)**2 * input_power)
    ports = [float(common*np.sum(np.abs(field+sign*pair)**2)/4) for sign in (1,-1)]
    return dict(input_power_equivalent=input_power,monitor_half_width_um=aperture,angle_step_rad_um=dk,
                sum=ports[0],diff=ports[1],collection=sum(ports),**info)
