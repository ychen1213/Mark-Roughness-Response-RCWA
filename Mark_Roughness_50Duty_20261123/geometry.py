"""Continuous, reproducible rough stripe geometry. Lengths are in micrometres."""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PITCH = 3.2
WIDTH = 1.6
HEIGHT = .3
WAVELENGTH = .643
REFERENCE_Y = .6
SEEDS = (1427, 2441, 3547, 4639)
BANDS = (.5, 1., 2.)
MASTER_BAND = .5


@dataclass(frozen=True)
class Geometry:
    mode: str
    seed: int = 0
    sigma_nm: float = 0.
    xi_um: float = .2
    band_um: float = .5
    length_um: float = 8.
    stripes: int = 2

    @property
    def lu(self):
        return self.stripes * PITCH

    @property
    def centers(self):
        return (np.arange(self.stripes) + .5 - self.stripes / 2) * PITCH

    @property
    def name(self):
        suffix = f"_v{self.length_um:g}_u{self.stripes}"
        if self.mode in ("flat", "air", "smooth", "sine"):
            return self.mode + (suffix if suffix != "_v8_u2" else "")
        return (f"{self.mode}_{self.seed}_r{self.sigma_nm:g}_"
                f"x{self.xi_um:g}_b{self.band_um:g}" + (suffix if suffix != "_v8_u2" else ""))

    def master(self):
        count = int(np.floor(self.length_um / MASTER_BAND + 1e-10))
        q = 2 * np.pi * np.arange(1, count + 1) / self.length_um
        a = np.zeros((self.stripes, count), complex)
        if self.mode == "sine":
            a[:, round(self.length_um / 2) - 1] = -.01j
        if self.mode in ("center", "width"):
            weight = 1 / (1 + (q * self.xi_um)**2)
            scale = self.sigma_nm / 1000 / np.sqrt(2 * weight.sum())
            rng = np.random.default_rng(self.seed)
            # Draw one complex series per line. The two pure modes share it.
            noise = rng.normal(size=(self.stripes, count, 2))
            a = scale * np.sqrt(weight) * (noise[..., 0] + 1j * noise[..., 1]) / np.sqrt(2)
        return q, a

    def offsets(self, v):
        q, master = self.master()
        qmax = 2 * np.pi / self.band_um
        active = (q <= qmax * (1 + 1e-12))
        return 2 * (master[:, active] @ np.exp(1j * q[active, None] * np.asarray(v))).real

    def bounds(self, v, shift=0.):
        h = self.offsets(v)
        c = self.centers[:, None] + shift
        return c - WIDTH / 2 + (-h if self.mode == "width" else h), c + WIDTH / 2 + h

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        v = np.linspace(-self.length_um / 2, self.length_um / 2, round(self.length_um / .01) + 1)
        left, right = self.bounds(v)
        width = right - left
        gaps = np.vstack((left[1:] - right[:-1], left[:1] + self.lu - right[-1:]))
        if np.min(width) <= 0 or np.min(gaps) <= 0:
            raise ValueError(f"Crossed boundaries in {self.name}")
        if not np.allclose(left[:, 0], left[:, -1], atol=1e-12) or not np.allclose(right[:, 0], right[:, -1], atol=1e-12):
            raise ValueError(f"Non-periodic seam in {self.name}")
        q, master = self.master()
        dst = directory / f"{self.name}.npz"
        payload = directory / f".{self.name}.{os.getpid()}.npz"
        np.savez_compressed(payload, v_um=v, left_um=left, right_um=right, q_rad_um=q, master_um=master)
        payload.replace(dst)
        audit = dict(asdict(self), case=self.name, path=str(dst.relative_to(ROOT)),
                     minimum_width_nm=float(1000 * width.min()), minimum_gap_nm=float(1000 * gaps.min()),
                     edge_rms_nm=(1000 * np.std(self.offsets(v[:-1]), axis=1)).tolist(),
                     dc_width_um=np.mean(width[:, :-1], axis=1).tolist())
        note = directory / f".{self.name}.{os.getpid()}.json"
        note.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        note.replace(directory / f"{self.name}.json")
        return audit


def cases():
    entries = [Geometry("smooth"), Geometry("sine"), Geometry("flat")]
    entries.extend(Geometry(kind, seed, sigma, .2, band)
                   for kind in ("center", "width") for sigma in (5., 20.)
                   for seed in SEEDS for band in BANDS)
    entries.extend(Geometry(kind, SEEDS[0], 20., .8, band)
                   for kind in ("center", "width") for band in BANDS)
    return {case.name: case for case in entries}


def material_harmonics(geom, lu_orders, lv_orders, step=.002):
    """Exact u integral for each stripe, quadrature only along v."""
    v = -geom.length_um / 2 + np.arange(round(geom.length_um / step)) * step
    left, right = geom.bounds(v)
    width, center = right - left, (right + left) / 2
    m = np.asarray(lu_orders) / geom.lu
    n = np.asarray(lv_orders) / geom.length_um
    cross = np.sum(width[None, ...] * np.sinc(m[:, None, None] * width[None, ...]) *
                   np.exp(-2j * np.pi * m[:, None, None] * center[None, ...]), axis=1)
    return cross @ np.exp(-2j * np.pi * v[:, None] * n[None, :]) / (len(v) * geom.lu)


if __name__ == "__main__":
    for selection in cases().values():
        selection.save(ROOT / "geometry")
    print(f"Saved {len(cases())} geometries in {ROOT}")
