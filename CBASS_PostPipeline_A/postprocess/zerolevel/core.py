# postprocess/zerolevel/core.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import healpy as hp
from matplotlib import pyplot as plt


@dataclass
class ZeroLevelResult:
    map_I: np.ndarray                  # processed intensity map (npix,)
    metrics: Dict[str, Any]
    figures: Tuple[str, ...] = ()


def _ensure_dir(p: Optional[str]) -> None:
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def dipole_map(nside: int, amplitude_K: float, lon_deg: float, lat_deg: float) -> np.ndarray:
    """Geometric dipole: A * (v ⋅ n). lon/lat in degrees (Galactic or Equatorial)."""
    npix = hp.nside2npix(nside)
    v = np.array(hp.pix2vec(nside, np.arange(npix)))               # (3,npix)
    n = hp.ang2vec(lon_deg, lat_deg, lonlat=True)                  # (3,)
    n = n / np.linalg.norm(n)
    cosang = (v.T @ n)
    return amplitude_K * cosang


def declination_mask(nside: int, coord: str, min_dec_deg: float = 0.0) -> np.ndarray:
    """1 where Dec >= min_dec_deg, 0 elsewhere. Accepts coord='G' or 'C'."""
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    if coord.upper() == 'G':
        rot = hp.Rotator(coord=['G', 'C'])
        theta, phi = rot(theta, phi)
    dec = -np.rad2deg(theta) + 90.0
    mask = np.zeros(npix, dtype=np.bool_)
    mask[dec >= min_dec_deg] = True
    return mask


def subtract_cmb_dipole(I: np.ndarray, coord: str, amplitude_mK: float,
                        fig_dir: Optional[str] = None) -> np.ndarray:
    """
    Subtracts the CMB dipole in the given coordinate frame.

    Planck 2018 dipole direction:
      Galactic: (l,b) = (264.021, 48.253) deg
      Equatorial: (RA,Dec) = (167.942, -6.944) deg  [included for reference]
    """
    nside = hp.get_nside(I)
    amp_K = amplitude_mK * 1e-3
    if coord.upper() == 'G':
        dmap = dipole_map(nside, amp_K, lon_deg=264.021, lat_deg=48.253)
    else:
        # Using the provided Equatorial coordinates for completeness
        dmap = dipole_map(nside, amp_K, lon_deg=167.942, lat_deg=-6.944)

    out = I.copy()
    mask = (I != hp.UNSEEN) & np.isfinite(I)
    out[mask] = I[mask] - dmap[mask]

    if fig_dir:
        _ensure_dir(fig_dir)
        hp.mollview(dmap, title="CMB Dipole (K)")
        plt.savefig(os.path.join(fig_dir, "Mono_CMB_dipole.png"))
        plt.close()
    return out


def set_zerolevel(I: np.ndarray, coord: str, offset_mK: float,
                  min_dec_deg: float = 0.0,
                  fig_dir: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Shift map so: (1) subtract mean above min_dec, (2) add offset (mK) globally.
    Returns adjusted map and summary metrics.
    """
    nside = hp.get_nside(I)
    out = I.copy()
    mask_valid = (out != hp.UNSEEN) & np.isfinite(out)

    dec_mask = declination_mask(nside, coord, min_dec_deg=min_dec_deg)
    use = mask_valid & dec_mask
    mean_above = float(np.nanmean(out[use])) if np.any(use) else 0.0

    # subtract mean, then add offset (convert mK→K)
    out[mask_valid] = out[mask_valid] - mean_above + (offset_mK * 1e-3)

    # metrics
    med_above = float(np.nanmedian(out[dec_mask & (out != hp.UNSEEN)]))
    min_above = float(np.nanmin(out[dec_mask & (out != hp.UNSEEN)]))
    metrics = dict(
        mean_before_K=mean_above,
        mean_after_K=float(np.nanmean(out[use])) if np.any(use) else 0.0,
        median_above0dec_mK=med_above * 1e3,
        min_above0dec_mK=min_above * 1e3,
        applied_offset_mK=float(offset_mK),
    )

    if fig_dir:
        _ensure_dir(fig_dir)
        hp.mollview(dec_mask.astype(float), title=f"Declination mask ≥ {min_dec_deg}°", cmap="Greys")
        plt.savefig(os.path.join(fig_dir, f"Mono_mask_dec_ge_{int(min_dec_deg)}.png"))
        plt.close()

    return out, metrics


def run_zero_level(
    input_map: np.ndarray,
    coord: str,
    remove_cmb: bool = True,
    dipole_amp_mK: float = 3.36208,
    offset_mK: float = 42.0,
    min_dec_deg: float = 0.0,
    fig_dir: Optional[str] = None,
) -> ZeroLevelResult:
    """
    Core pipeline for intensity (Stokes I) zero-level:
      - optional CMB dipole subtraction
      - zero-level set to desired offset above a declination cut
    """
    I = input_map.copy()

    figs = []

    if fig_dir:
        _ensure_dir(fig_dir)
        hp.mollview(I, norm="hist", title="ZeroLevel: input (I)")
        p0 = os.path.join(fig_dir, "Mono_InputMap.png")
        plt.savefig(p0); plt.close(); figs.append(p0)

    if remove_cmb:
        I = subtract_cmb_dipole(I, coord=coord, amplitude_mK=dipole_amp_mK, fig_dir=fig_dir)
        if fig_dir:
            hp.mollview(I, norm="hist", title="ZeroLevel: after CMB dipole subtraction")
            p1 = os.path.join(fig_dir, "Mono_CMBDipoleSubtractedMap.png")
            plt.savefig(p1); plt.close(); figs.append(p1)

    I_out, extra = set_zerolevel(I, coord=coord, offset_mK=offset_mK, min_dec_deg=min_dec_deg, fig_dir=fig_dir)

    if fig_dir:
        hp.mollview(I_out, norm="hist", title="ZeroLevel: calibrated (I)")
        p2 = os.path.join(fig_dir, "ZeroLevelCalibratedMap.png")
        plt.savefig(p2); plt.close(); figs.append(p2)

    metrics = dict(
        cmb_dipole_removed=bool(remove_cmb),
        dipole_amp_mK=float(dipole_amp_mK),
        dipole_glon_deg=264.021,
        dipole_glat_deg=48.253,
        offset_mK=float(offset_mK),
        **extra,
    )

    return ZeroLevelResult(map_I=I_out, metrics=metrics, figures=tuple(figs))
