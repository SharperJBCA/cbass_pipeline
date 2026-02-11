# postprocess/plotting/quicklook.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

@dataclass
class QuicklookResult:
    fullsky_figs: Dict[str, str]
    zoom_figs: Dict[str, str]

# -------------------- helpers --------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _finite_mask(arr: np.ndarray) -> np.ndarray:
    m = np.isfinite(arr) & (arr != hp.UNSEEN)
    return m

def _robust_vlim(arr: np.ndarray, p_lo=1.0, p_hi=99.0) -> Tuple[float, float]:
    m = _finite_mask(arr)
    if not np.any(m):
        return -1.0, 1.0
    v = arr[m]
    lo, hi = np.percentile(v, [p_lo, p_hi])
    # widen slightly
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    return lo - pad, hi + pad

def _moll(
    m: np.ndarray,
    title: str,
    outpng: str,
    cmap="viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    coord: str = "G",
) -> str:
    plt.figure(figsize=(10, 6))
    # hist normalization plays nice with large dynamic range maps, but we prefer percentile control
    if vmin is None or vmax is None:
        vmin, vmax = _robust_vlim(m)
    hp.mollview(m, title=title, cmap=cmap, min=vmin, max=vmax, coord=coord,norm='hist')
    hp.graticule(dpar=30, dmer=30, alpha=0.3)
    _ensure_dir(os.path.dirname(outpng) or ".")
    plt.savefig(outpng, dpi=180, bbox_inches="tight")
    plt.close()
    return outpng

def _moll_div(
    m: np.ndarray, title: str, outpng: str, coord: str = "G"
) -> str:
    # Diverging for Q/U and cross-covariances
    vmin, vmax = _robust_vlim(m)
    vmax = max(abs(vmin), abs(vmax))
    vmin = -vmax
    return _moll(m, title, outpng, cmap="RdBu_r", vmin=vmin, vmax=vmax, coord=coord)

def _gnom(
    m: np.ndarray,
    l_deg: float,
    b_deg: float,
    outpng: str,
    title: str = "",
    coord_map: str = "G",
    fov_deg: float = 12.0,
    reso_arcmin: float = 6.0,
    diverging: bool = False,
):
    """
    Small postage-stamp around (l,b) in Galactic by default.
    """
    plt.figure(figsize=(7, 6))
    # healpy expects rot=[lon, lat] in the map's native frame
    rot = [l_deg, b_deg]
    # autoscale with robust percentiles in the cutout
    # Quick hack: use global robust limits (works fine); gnomview lacks easy mask pre-calc.
    vmin, vmax = _robust_vlim(m)
    if diverging:
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax
        cmap = "RdBu_r"
    else:
        cmap = "viridis"
    hp.gnomview(
        m,
        rot=rot,
        xsize=int((fov_deg * 60.0) / reso_arcmin),
        reso=reso_arcmin,
        title=title,
        min=vmin,
        max=vmax,
        cmap=cmap,
        coord=coord_map,
        no_plot=False,
        norm='hist'
    )
    hp.graticule()
    _ensure_dir(os.path.dirname(outpng) or ".")
    plt.savefig(outpng, dpi=180, bbox_inches="tight")
    plt.close()
    return outpng

# -------------------- public API --------------------

def plot_fullsky_IQU(
    IQU: np.ndarray,
    fig_root: str,
    coord: str = "G",
    tag: str = "map",
) -> Dict[str, str]:
    """
    IQU: array with shape (3, npix) or (npix,) for I-only.
    """
    _ensure_dir(fig_root)
    out: Dict[str, str] = {}
    if IQU.ndim == 1:
        I = IQU
        out["I"] = _moll(I, f"I (full sky) [{tag}]", os.path.join(fig_root, f"{tag}_I_fullsky.png"), coord=coord)
        return out

    I, Q, U = IQU
    out["I"] = _moll(I, f"I (full sky) [{tag}]", os.path.join(fig_root, f"{tag}_I_fullsky.png"), coord=coord)
    out["Q"] = _moll_div(Q, f"Q (full sky) [{tag}]", os.path.join(fig_root, f"{tag}_Q_fullsky.png"), coord=coord)
    out["U"] = _moll_div(U, f"U (full sky) [{tag}]", os.path.join(fig_root, f"{tag}_U_fullsky.png"), coord=coord)
    return out

def plot_fullsky_cov(
    C: np.ndarray,
    fig_root: str,
    coord: str = "G",
    tag: str = "cov",
) -> Dict[str, str]:
    """
    C: covariance “maps”, shape (6, npix): [II, QQ, UU, IQ, IU, QU]
    """
    _ensure_dir(fig_root)
    out: Dict[str, str] = {}
    names = ["II", "QQ", "UU", "IQ", "IU", "QU"]
    cmaps = {
        "II": "magma",
        "QQ": "magma",
        "UU": "magma",
        "IQ": "RdBu_r",
        "IU": "RdBu_r",
        "QU": "RdBu_r",
    }
    for i, name in enumerate(names):
        arr = C[i]
        if name in ("II", "QQ", "UU"):
            out[name] = _moll(arr, f"{name} (full sky) [{tag}]", os.path.join(fig_root, f"{tag}_{name}_fullsky.png"),
                              coord=coord, cmap=cmaps[name])
        else:
            out[name] = _moll_div(arr, f"{name} (full sky) [{tag}]", os.path.join(fig_root, f"{tag}_{name}_fullsky.png"),
                                  coord=coord)
    return out

def plot_zooms_standard(
    IQU: np.ndarray,
    fig_root: str,
    coord: str = "G",
    tag: str = "map",
) -> Dict[str, str]:
    """
    Zooms: Tau A, Cas A, quiet patches at (l,b)=(90,30) and (210,45).
    Uses I only for the zoom by default (can extend to Q/U if you like).
    """
    _ensure_dir(fig_root)
    if IQU.ndim == 1:
        I = IQU
    else:
        I, _, _ = IQU

    targets = [
        ("Bad_Source_1", 183, -8),
        ("TauA", 184.5551, -5.7877),
        ("CasA", 111.734751, -2.129568),
        ("Quiet1_l90_b30", 90.0, 30.0),
        ("Quiet2_l210_b45", 210.0, 45.0),
    ]
    out: Dict[str, str] = {}
    for name, l, b in targets:
        out[name] = _gnom(
            I,
            l_deg=l,
            b_deg=b,
            outpng=os.path.join(fig_root, f"{tag}_zoom_{name}.png"),
            title=f"{name} [{tag}]",
            coord_map=coord,
            fov_deg=12.0,
            reso_arcmin=6.0,
            diverging=False,
        )
    return out

def run_quicklook(
    bundle,        # MapBundle-like (has .map, optional .cov, .coords, .nside)
    fig_root: Optional[str] = None,
    tag: str = "quicklook",
) -> QuicklookResult:
    """
    Generate a compact set of diagnostic plots and return paths.
    """
    fig_root = fig_root or "figures"
    _ensure_dir(fig_root)
    coord = getattr(bundle, "coords", "G") or "G"

    fullsky: Dict[str, str] = {}
    zooms: Dict[str, str] = {}

    # IQU
    if bundle.map is not None:
        fullsky |= plot_fullsky_IQU(np.asarray(bundle.map), fig_root, coord=coord, tag=tag)
        zooms   |= plot_zooms_standard(np.asarray(bundle.map), fig_root, coord=coord, tag=tag)

    # Covariances if present
    if getattr(bundle, "cov", None) is not None:
        C = np.asarray(bundle.cov)
        if C.ndim == 2 and C.shape[0] == 6:
            fullsky |= plot_fullsky_cov(C, fig_root, coord=coord, tag=f"{tag}_cov")

    return QuicklookResult(fullsky_figs=fullsky, zoom_figs=zooms)
