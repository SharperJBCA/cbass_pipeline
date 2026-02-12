from __future__ import annotations
from dataclasses import dataclass, field
import os, h5py, numpy as np, healpy as hp
from typing import Optional, Dict, Any, Tuple, Iterable
from astropy.io import fits

try:
    from scipy.spatial import cKDTree  # optional
    _HAS_KDTREE = True
except Exception:
    _HAS_KDTREE = False

# --- small helpers -----------------------------------------------------------

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)

def rotate(x_deg: np.ndarray, y_deg: np.ndarray, coord: Iterable[str] = ('G','C')) -> Tuple[np.ndarray, np.ndarray]:
    rot = hp.rotator.Rotator(coord=list(coord))
    yr, xr = rot((90 - y_deg) * np.pi/180.0, x_deg * np.pi/180.0)
    return xr * 180/np.pi, (np.pi/2 - yr) * 180/np.pi

def _unit_vector_thetaphi(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    return np.column_stack((st*cp, st*sp, ct))

# --- base class --------------------------------------------------------------

@dataclass
class Catalogue:
    name   : str = ''
    glon   : np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    glat   : np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    flux   : np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    eflux  : np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    flag   : np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    source : np.ndarray = field(default_factory=lambda: np.empty(0, dtype='S20'))
    min_flux: float = 0.0  # cut in subclass on load

    # ---- core protocol
    def run(self, fitsfile: str) -> None:
        """Subclasses fill glon/glat/flux/eflux/flag/source here."""
        raise NotImplementedError

    def __call__(self, fitsfile: str) -> None:
        self.run(fitsfile)
        self.clean_nan()
        self.remove_flagged()

    # ---- properties / dunder
    def __len__(self) -> int: return self.size
    @property
    def size(self) -> int: return int(self.glon.size)
    @property
    def thetaphi(self) -> Tuple[np.ndarray, np.ndarray]:
        return (90 - self.glat) * np.pi/180.0, self.glon * np.pi/180.0

    def __add__(self, other: "Catalogue") -> "Catalogue":
        new = Catalogue(name=f"{self.name}_{other.name}")
        new.glon  = np.concatenate([self.glon,  other.glon ])
        new.glat  = np.concatenate([self.glat,  other.glat ])
        new.flux  = np.concatenate([self.flux,  other.flux ])
        new.eflux = np.concatenate([self.eflux, other.eflux])
        new.flag  = np.concatenate([self.flag,  other.flag ])
        # normalize SOURCE dtype to fixed-length bytes
        s1 = self.source.astype('S20', copy=False)
        s2 = other.source.astype('S20', copy=False)
        new.source = np.concatenate([s1, s2])
        return new

    def __repr__(self) -> str:
        return f"Catalogue({self.name}): N={self.size}, flagged={int(self.flag.sum())}"

    # ---- maintenance
    def clean_nan(self) -> None:
        m = (
            ~np.isfinite(self.glon) | ~np.isfinite(self.glat) |
            ~np.isfinite(self.flux) | ~np.isfinite(self.eflux) |
            (self.flux <= 0)
        )
        keep = ~m
        self._apply_mask_inplace(keep)

    def remove_flagged(self) -> None:
        self._apply_mask_inplace(~self.flag)

    def remove_sources(self, mask) -> None:
        self._apply_mask_inplace(mask)

    def _apply_mask_inplace(self, keep: np.ndarray) -> None:
        self.glon  = np.ascontiguousarray(self.glon [keep])
        self.glat  = np.ascontiguousarray(self.glat [keep])
        self.flux  = np.ascontiguousarray(self.flux [keep])
        self.eflux = np.ascontiguousarray(self.eflux[keep])
        self.flag  = np.ascontiguousarray(self.flag [keep])
        self.source= np.ascontiguousarray(self.source[keep])

    def update_sources(self,mask,flux,eflux):
        """Update fluxes of sources"""
        self.flux[mask] = flux
        self.eflux[mask] = eflux

    # ---- masking/cuts
    def mask_map(self, mask_map: np.ndarray, flux: Optional[np.ndarray]=None, lower_limit: float=0.0) -> None:
        """
        Remove sources in masked pixels.

        If ``flux`` is provided, only remove masked sources with ``flux < lower_limit``.
        This is used for tiered threshold masking where deeper masks remove progressively
        fainter sources from a given catalogue.
        """
        theta, phi = self.thetaphi
        pix = hp.ang2pix(hp.npix2nside(mask_map.size), theta, phi)
        if flux is None:
            kill = mask_map[pix].astype(bool)
        else:
            kill = mask_map[pix].astype(bool) & (flux < float(lower_limit))
        self._apply_mask_inplace(~kill)

    def mask_declinations(self, declination_min: float=-10, declination_max: float=90) -> None:
        theta, phi = self.thetaphi
        rot = hp.rotator.Rotator(coord=['G','C'])
        tC, _ = rot(theta, phi)
        dec = 90.0 - tC * 180/np.pi
        keep = (dec > declination_min) & (dec < declination_max)
        self._apply_mask_inplace(keep)

    # ---- persistence
    def write_file(self, filename: str) -> None:
        _ensure_dir(filename)
        # HDF5 (compressed)
        with h5py.File(filename, "w") as h:
            grp = h.create_group(self.name)
            grp.attrs["schema"] = "Catalogue/v1"
            grp.create_dataset("GLON", data=self.glon,  compression="gzip", shuffle=True)
            grp.create_dataset("GLAT", data=self.glat,  compression="gzip", shuffle=True)
            grp.create_dataset("FLUX", data=self.flux,  compression="gzip", shuffle=True)
            grp.create_dataset("eFLUX",data=self.eflux, compression="gzip", shuffle=True)
            grp.create_dataset("FLAG", data=self.flag,  compression="gzip", shuffle=True)
            grp.create_dataset("SOURCE", data=self.source.astype('S20'))

        # FITS mirror
        cols = [
            fits.Column(name='GLON',  format='D',   array=self.glon),
            fits.Column(name='GLAT',  format='D',   array=self.glat),
            fits.Column(name='FLUX',  format='D',   array=self.flux),
            fits.Column(name='eFLUX', format='D',   array=self.eflux),
            fits.Column(name='FLAG',  format='L',   array=self.flag),
            fits.Column(name='SOURCE',format='20A', array=self.source)
        ]
        fits.BinTableHDU.from_columns(cols).writeto(filename.replace(".hdf5",".fits"), overwrite=True)

    def load_file(self, filename: str, name: str='none') -> None:
        with h5py.File(filename, "r") as h:
            grp = h[name]
            self.glon  = grp['GLON'][...]
            self.glat  = grp['GLAT'][...]
            self.flux  = grp['FLUX'][...]
            self.eflux = grp['eFLUX'][...]
            self.flag  = grp['FLAG'][...].astype(bool)
            s = grp['SOURCE'][...]
            # bytes -> str (safe)
            try:
                self.source = np.array([x.decode('ascii') for x in s], dtype='S20')
            except Exception:
                self.source = s.astype('S20')
        self.name = name

    # ---- convenience
    def pixels(self, nside: int) -> np.ndarray:
        return hp.ang2pix(nside, *self.thetaphi)

    def as_table(self) -> Dict[str, np.ndarray]:
        return dict(GLON=self.glon, GLAT=self.glat, FLUX=self.flux, eFLUX=self.eflux,
                    FLAG=self.flag, SOURCE=self.source)

    # helper for subclasses
    def _from_fits_table(self, hdu: fits.BinTableHDU, mappings: Dict[str, str],
                         source_name: str, min_flux_ok: bool=True,
                         flags: Optional[np.ndarray]=None) -> None:
        d = hdu.data
        g = np.asarray(d[mappings['glon']], dtype=float)
        b = np.asarray(d[mappings['glat']], dtype=float)
        f = np.asarray(d[mappings['flux']], dtype=float)
        ef= np.asarray(d[mappings['eflux']], dtype=float)
        mflag = np.zeros_like(f, dtype=bool) if flags is None else flags.astype(bool).copy()
        if min_flux_ok:
            mflag |= (f < self.min_flux)
        self.glon, self.glat, self.flux, self.eflux, self.flag = g, b, f, ef, mflag
        self.source = np.array([source_name]*self.size, dtype='S20')

# --- concrete catalogues -----------------------------------------------------

@dataclass
class Mingaliev(Catalogue):
    name: str = "Mingaliev"
    def run(self, fitsfile: str) -> None:
        with fits.open(fitsfile, memmap=False) as hdul:
            d = hdul[1].data
            gl = np.asarray(d['_Glon'], dtype=float)
            gb = np.asarray(d['_Glat'], dtype=float)
            # Scale 3.9->4.76 GHz using spectral index
            sidx = np.asarray(d['Sp-Index'], dtype=float)
            f39  = np.asarray(d['S3_9GHz'], dtype=float)
            ef39 = np.asarray(d['e_S3_9GHz'], dtype=float)
            scale = (3.9/4.76)**sidx
            self.glon, self.glat = gl, gb
            self.flux  = f39  * scale
            self.eflux = ef39 * scale
            self.flag  = (self.flux < self.min_flux)
            self.source= np.array(['Mingaliev']*self.size, dtype='S20')

@dataclass
class GB6(Catalogue):
    name: str = "GB6"
    def run(self, fitsfile: str) -> None:
        with fits.open(fitsfile, memmap=False) as hdul:
            d = hdul[1].data
            gl = np.asarray(d['_Glon'], dtype=float)
            gb = np.asarray(d['_Glat'], dtype=float)
            f  = np.asarray(d['Flux'], dtype=float)*1e-3
            ef = np.asarray(d['e_Flux'], dtype=float)*1e-3
            # Eflag == 'E' means extended -> flag it
            efld = np.asarray(d['Eflag']).astype('U1')
            bad = (efld == 'E')
            bad |= (f < self.min_flux)
            self.glon, self.glat, self.flux, self.eflux = gl, gb, f, ef
            self.flag = bad.astype(bool)
            self.source = np.array(['GB6']*self.size, dtype='S20')

@dataclass
class PMN(Catalogue):
    name: str = "PMN"
    def run(self, fitsfile: str) -> None:
        with fits.open(fitsfile, memmap=False) as hdul:
            d = hdul[1].data
            gl = np.asarray(d['_Glon'], dtype=float)
            gb = np.asarray(d['_Glat'], dtype=float)
            f  = np.asarray(d['Flux'], dtype=float)*1e-3
            ef = np.asarray(d['e_Flux'], dtype=float)*1e-3
            xflag = np.asarray(d['Xflag']).astype('U1')
            bad = (xflag == 'X') | (f < self.min_flux)
            self.glon, self.glat, self.flux, self.eflux = gl, gb, f, ef
            self.flag = bad.astype(bool)
            self.source = np.array(['PMN']*self.size, dtype='S20')

@dataclass
class CBASS(Catalogue):
    name: str = "CBASS"
    def run(self, txtfile: str) -> None:
        # Columns (skip header): 3=glon,4=glat,7=flux(Jy),8=eflux(Jy)
        data = np.loadtxt(txtfile, skiprows=1, usecols=[3,4,7,8])
        gl, gb, f, ef = data.T
        bad = (f < self.min_flux)
        self.glon, self.glat, self.flux, self.eflux = gl, gb, f, ef
        self.flag = bad.astype(bool)
        self.source = np.array(['CBASS']*self.size, dtype='S20')

# --- cross-matching utilities -----------------------------------------------

def haversine(glat1, glon1, glat2, glon2) -> np.ndarray:
    """Great-circle distance in radians (vectorized, inputs deg)."""
    t1, p1 = np.radians(glat1), np.radians(glon1)
    t2, p2 = np.radians(glat2), np.radians(glon2)
    dt, dp = t2 - t1, p2 - p1
    a = np.sin(dt/2)**2 + np.cos(t1)*np.cos(t2)*np.sin(dp/2)**2
    return 2*np.arcsin(np.sqrt(a))

def common_sources(cat1: Catalogue, cat2: Catalogue, radius_arcmin: float=1.0) -> Tuple[Catalogue, Catalogue]:
    """
    Merge duplicates within radius_arcmin between cat1 and cat2 by inverse-variance average.
    Returns (cat1_trimmed, cat2_updated).
    """
    r_rad = np.deg2rad(radius_arcmin / 60.0)
    if _HAS_KDTREE:
        # unit vectors on sphere, search within chord distance ~ 2 sin(r/2)
        v1 = _unit_vector_thetaphi(*cat1.thetaphi)
        v2 = _unit_vector_thetaphi(*cat2.thetaphi)
        tree = cKDTree(v2)
        # chord radius
        eps = 2*np.sin(r_rad/2)
        idxs = tree.query_ball_point(v1, r=eps)
        keep1 = np.ones(cat1.size, dtype=bool)
        keep2 = np.ones(cat2.size, dtype=bool)
        f2new = cat2.flux.copy()
        ef2new= cat2.eflux.copy()
        for i, js in enumerate(idxs):
            if not js: continue
            # choose nearest by true angular separation
            j = min(js, key=lambda k: haversine(cat1.glat[i], cat1.glon[i], cat2.glat[k], cat2.glon[k]))
            f1, e1 = cat1.flux[i], cat1.eflux[i]
            f2, e2 = cat2.flux[j], cat2.eflux[j]
            w = 1/(e1**2) + 1/(e2**2)
            f2new[j] = (f1/(e1**2) + f2/(e2**2)) / w
            ef2new[j]= np.sqrt(1/w)
            keep1[i] = False
            keep2[j] = False  # mark updated; we’ll keep it with new vals
        cat1.remove_sources(keep1)  # drop merged ones
        cat2.update_sources(~keep2, f2new[~keep2], ef2new[~keep2])
        return cat1, cat2
    else:
        # fall back to original (O(N²) for clustered regions)
        keep1 = np.ones(cat1.size, dtype=bool)
        keep2 = np.ones(cat2.size, dtype=bool)
        f2new = cat2.flux.copy()
        ef2new= cat2.eflux.copy()
        for i in range(cat1.size):
            dist = haversine(cat1.glat[i], cat1.glon[i], cat2.glat, cat2.glon)
            j = np.argmin(dist)
            if dist[j] < r_rad:
                f1, e1 = cat1.flux[i], cat1.eflux[i]
                f2, e2 = cat2.flux[j], cat2.eflux[j]
                w = 1/(e1**2) + 1/(e2**2)
                f2new[j] = (f1/(e1**2) + f2/(e2**2)) / w
                ef2new[j]= np.sqrt(1/w)
                keep1[i] = False
                keep2[j] = False
        cat1.remove_sources(keep1)
        cat2.update_sources(~keep2, f2new[~keep2], ef2new[~keep2])
        return cat1, cat2

# --- legacy loader/saver convenience (kept for compatibility) ----------------

def save_catalogues(filename: str, catalogues: Dict[str, Dict[str, np.ndarray]]) -> None:
    _ensure_dir(filename)
    with h5py.File(filename, "a") as h:
        for catname, cat in catalogues.items():
            grp = h.require_group(catname)
            for dname, data in cat.items():
                if dname in grp: del grp[dname]
                try:
                    grp.create_dataset(dname, data=data, compression="gzip", shuffle=True)
                except TypeError:
                    continue

def load_catalogues(filename: str) -> Dict[str, Dict[str, np.ndarray]]:
    with h5py.File(filename, "r") as h:
        out = {}
        for catname, grp in h.items():
            out[catname] = {d: grp[d][...] for d in grp.keys()}
    return out
