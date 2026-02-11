# postprocess/sources/mapper.py
from __future__ import annotations
import os
import numpy as np
import healpy as hp
from typing import Optional, Tuple

# Optional: if you keep your GSL fast-path
try:
    from . import gsl_funcs 
    _HAS_GSL = True
except Exception:
    _HAS_GSL = False

# ---- constants (SI) ----
KB = 1.380649e-23
C  = 2.99792458e8

def _jy_per_sr_to_K(nu_hz: float) -> float:
    """Return factor X such that T_K = I[Jy/sr] / X."""
    # Rayleigh–Jeans: T = (c^2 / (2 k_B nu^2)) * I_nu, with I in W m^-2 Hz^-1 sr^-1.
    # Convert Jy/sr -> W m^-2 Hz^-1 sr^-1 by *1e-26. So divide by 2k (nu/c)^2 * 1e26.
    return 2 * KB * (nu_hz / C)**2 * 1e26

def _load_beam(
    beam_model: Optional[str] = None,
    bl: Optional[np.ndarray] = None,
    lmax: Optional[int] = None,
    normalize: bool = True
) -> Tuple[np.ndarray, Optional[float]]:
    """
    Returns (B_ell, beam_solid_angle_sr_or_None).

    - If `bl` is given: use it (pad/crop to lmax). Solid angle is None.
    - Else if beam_theta_deg_and_Btheta is given: compute B_ell with healpy.beam2bl and
      return also the real-space beam solid angle.
    """
    if bl is not None:
        bl = np.asarray(bl, dtype=float)
        if lmax is not None:
            # pad/crop
            if bl.size < (lmax + 1):
                bl = np.pad(bl, (0, lmax + 1 - bl.size), mode="edge")
            elif bl.size > (lmax + 1):
                bl = bl[:lmax + 1]
        if normalize and bl[0] != 0.0:
            bl = bl / bl[0]
        return bl, None

    if beam_model is None:
        raise ValueError("Must provide either bl or beam_theta_deg_and_Btheta")

    tab = np.asarray(beam_model, dtype=float)
    if tab.ndim != 2 or tab.shape[1] < 2:
        raise ValueError("beam_theta_deg_and_Btheta must be shape (N,2) = [theta_deg, B(theta)]")
    theta = np.deg2rad(tab[:, 0])
    Bth   = tab[:, 1]
    if lmax is None:
        raise ValueError("lmax is required when providing B(theta) table")
    bl = hp.beam2bl(Bth, theta, lmax=lmax)
    # solid angle for arbitrary circularly symmetric beam: ∫ B(θ) 2π sinθ dθ
    dtheta = np.abs(theta[1] - theta[0])
    omega = float(2 * np.pi * np.sum(Bth * np.sin(theta)) * dtheta)
    if normalize and bl[0] != 0.0:
        bl = bl / bl[0]
    return bl, omega

def _safe_bl(bl: np.ndarray, spin2_zero: bool = False) -> np.ndarray:
    """Clamp tiny/negative values to avoid numerical blow-ups; zero ℓ<2 for spin-2 if requested."""
    out = np.asarray(bl, dtype=float).copy()
    # Guard against negative/zero tails (can occur with noisy beam tables)
    floor = max(out[0] * 7e-3, 1e-12)  # consistent with your earlier cut
    out = np.maximum(out, floor)
    if spin2_zero and out.size >= 2:
        out[:2] = 0.0
    return out

def pixel_space(
    flux_jy: np.ndarray,
    glon_deg: np.ndarray,
    glat_deg: np.ndarray,
    *,
    nside: int = 1024,
    lmax: Optional[int] = None,
    fwhm_deg: float = 1.0,
    frequency_ghz: float = 4.76,
    beam_model: Optional[np.ndarray] = None,
    bl: Optional[np.ndarray] = None,
    use_beam_model: bool = True,
) -> np.ndarray:
    """
    Bin point-source fluxes (Jy) into a HEALPix map and convolve with a beam.

    Returns: map in Kelvin (RJ).

    Pipeline-friendly, no plotting, no disk I/O.

    Notes on units:
    - After binning, map is Jy per pixel.
    - After smoothing with a unit-normalized beam window, still Jy per pixel.
    - Convert to K via: (Jy/pixel)/(Jy/sr per K) where Jy/sr per K = 2 k (nu/c)^2 1e26,
      but we must divide the per-pixel Jy by pixel solid angle to get Jy/sr.
    """
    flux_jy = np.asarray(flux_jy, dtype=float)
    glon_deg = np.asarray(glon_deg, dtype=float)
    glat_deg = np.asarray(glat_deg, dtype=float)
    assert flux_jy.shape == glon_deg.shape == glat_deg.shape

    if lmax is None:
        lmax = 3 * nside - 1

    # Beam window
    if use_beam_model:
        B_ell, omega_sr = _load_beam(beam_model, bl, lmax=lmax, normalize=True)
    else:
        # simple Gaussian
        B_ell = hp.gauss_beam(np.deg2rad(fwhm_deg), lmax=lmax)
        B_ell /= B_ell[0]
        omega_sr = 1.133 * np.deg2rad(fwhm_deg)**2  # approx FWHM->Ω

    B_ell = _safe_bl(B_ell)

    # Bin sources: Jy per pixel
    theta = np.deg2rad(90.0 - glat_deg)
    phi   = np.deg2rad(glon_deg % 360.0)
    pix = hp.ang2pix(nside, theta, phi)
    # bincount is faster than histogram for 0..npix-1 bins
    npix = hp.nside2npix(nside)
    jy_per_pix = np.bincount(pix, weights=flux_jy, minlength=npix).astype(float)

    # Smooth in harmonic space
    alm = hp.map2alm(jy_per_pix, lmax=lmax)
    alm = hp.almxfl(alm, B_ell)
    jy_per_pix = hp.alm2map(alm, nside, verbose=False)

    # Convert Jy/pixel -> Jy/sr -> K
    pix_sr = hp.nside2pixarea(nside)
    jy_per_sr = jy_per_pix / pix_sr
    conv = _jy_per_sr_to_K(frequency_ghz * 1e9)
    T_K = jy_per_sr / conv
    return T_K

def alm_space(
    flux_jy: np.ndarray,
    glon_deg: np.ndarray,
    glat_deg: np.ndarray,
    *,
    nside: int = 1024,
    lmax: Optional[int] = None,
    fwhm_deg: float = 1.0,
    frequency_ghz: float = 4.76,
    beam_theta_deg_and_Btheta: Optional[np.ndarray] = None,
    bl: Optional[np.ndarray] = None,
    use_beam_model: bool = True,
    use_gsl: Optional[bool] = None,
) -> np.ndarray:
    """
    Build the sky in harmonic space by summing the spherical harmonics of point sources.

    Returns: map in Kelvin (RJ).

    Two paths:
      1) Fast-path with gsl_funcs.precomp_harmonics if available and use_gsl is True.
      2) Fallback to pixel binning (calls pixel_space) if GSL is unavailable.

    Units:
      - We assemble a map that corresponds to surface brightness: Jy/sr after smoothing.
      - Convert to Kelvin via RJ factor.
    """
    if lmax is None:
        lmax = 3 * nside - 1

    if use_gsl is None:
        use_gsl = _HAS_GSL

    if not use_gsl:
        # Fall back to pixel method (robust and usually fast enough)
        return pixel_space(
            flux_jy, glon_deg, glat_deg,
            nside=nside, lmax=lmax, fwhm_deg=fwhm_deg,
            frequency_ghz=frequency_ghz,
            beam_theta_deg_and_Btheta=beam_theta_deg_and_Btheta,
            bl=bl, use_beam_model=use_beam_model,
        )

    # --- GSL path: build alms from source list directly ---
    if use_beam_model:
        B_ell, _ = _load_beam(beam_theta_deg_and_Btheta, bl, lmax=lmax, normalize=True)
    else:
        B_ell = hp.gauss_beam(np.deg2rad(fwhm_deg), lmax=lmax)
        B_ell /= B_ell[0]
    B_ell = _safe_bl(B_ell)

    # Prepare source array for GSL: [phi(rad), theta(rad), flux(Jy)]
    src = np.column_stack([
        np.deg2rad(glon_deg % 360.0),
        np.deg2rad(90.0 - glat_deg),
        np.asarray(flux_jy, dtype=float),
    ])

    # Compute alms (real/imag from GSL helper)
    Nalm = hp.Alm.getsize(lmax)
    idx = np.arange(Nalm, dtype=int)
    # gsl_funcs.precomp_harmonics returns shape (Nalm, 2): [Re, Im]? Your code used [:,0]-i[:,1]
    d_alms = gsl_funcs.precomp_harmonics(src, idx.size, lmax)
    alms = d_alms[:, 0] - 1j * d_alms[:, 1]
    alms[~np.isfinite(alms)] = 0.0 + 0.0j

    # Apply beam window
    alms = hp.almxfl(alms, B_ell)

    # Back to map space
    jy_per_sr_map = hp.alm2map(alms, nside, verbose=False)
    # Convert to K
    conv = _jy_per_sr_to_K(frequency_ghz * 1e9)
    T_K = jy_per_sr_map / conv
    return T_K
