import numpy as np 
import healpy as hp 
from . import inpainting

BL_FLOOR_FRAC = 7e-3 

def load_bl_any(
    beam_filename: str,
    lmax: int,
    beam_format: str = "theta_beam",   #  "theta_beam" | "ell_bl"
    beam_units: str | None = None,  # "deg" | "rad" (only for theta_beam)
    normalise: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (bl_spin0, bl_spin2) of length lmax+1 from either B(theta) or (ell,Bl) file.
    Spin-2 is copied from spin-0 with l=0,1 forced to zero (matches your current usage).
    """
    data = np.loadtxt(beam_filename)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{beam_filename}: expected 2+ columns")

    x = data[:, 0].astype(float)
    y = data[:, 1:].astype(float)

    # Header hints
    fmt = (beam_format or "auto").upper()
    if fmt == "THETA_BEAM":
        units = (beam_units or "DEG").upper()
        theta = x if units == "RAD" else np.radians(x)
        bl0 = hp.beam2bl(y, theta, lmax=lmax)
        if normalise and bl0[0] != 0:
            bl0 = bl0 / bl0[0]
        # Build spin-2
        bl2 = bl0.copy()
        bl2[:2] = 0.0
    elif fmt == "ELL_BL":
        ell_in = x
        bl0_in,bl2_in = y.T
        if not np.all(np.diff(ell_in) > 0):
            idx = np.argsort(ell_in)
            ell_in = ell_in[idx]; bl0_in = bl0_in[idx]; bl2_in = bl2_in[idx]
        # Interpolate onto 0..lmax, pad safely at ends
        ell = np.arange(lmax + 1, dtype=int)
        # Fill outside range with edge values
        bl0 = np.interp(ell, ell_in, bl0_in, left=bl0_in[0], right=bl0_in[-1])
        bl2 = np.interp(ell, ell_in, bl2_in, left=0, right=0)
        # Normalise if requested
        if normalise and bl0[0] != 0:
            bl2 = bl2 / bl0[0]
            bl0 = bl0 / bl0[0]
    else:
        raise ValueError(f"Unknown beam_format: {beam_format}")


    return bl0, bl2

def gen_pixel_window(nside, lmax=None):
    """Approximate pixel window function"""

    ell = np.arange(lmax + 1) 

    # Okay, so we are approximating the high-ell values of the pixel
    # window function by transforming a circular top beam into l-space.
    # This is pretty close. 
    theta = np.linspace(0,np.pi,10000) 
    # top hat beam 
    beam = np.zeros_like(theta) 
    pixel_area = hp.nside2pixarea(nside, degrees=True)
    beam[theta<np.radians(3.6/np.pi*0.5*pixel_area**0.5)] = 1
    bl = hp.beam2bl(beam, theta, lmax=lmax) 

    return  bl/bl[0] 

def build_transfer_functions(beam_filename, output_fwhm_deg, nside_in, nside_out, lmax, beam_normalise=False, beam_format="auto", beam_units=None,apply_transfer_function=False):
    # Pixel window ratio (nside_out / nside_in)
    if nside_out < nside_in:
        pw_out = gen_pixel_window(nside_out, lmax=lmax)
        pw_in  = gen_pixel_window(nside_in,  lmax=lmax)
        pixwin = pw_out / pw_in
    else:
        pixwin = np.ones(lmax + 1)

    if beam_filename:
        if apply_transfer_function: 
            bl0, bl2 = load_bl_any(
                beam_filename, lmax,
                beam_format=beam_format,
                beam_units=beam_units,
                normalise=beam_normalise,
            )
        else: # flat transfer functions so we just apply the pixel window function
            bl0 = np.ones(lmax + 1)
            bl2 = np.ones(lmax + 1)
            bl2[0:2] = 0.0

        g0 = hp.gauss_beam(np.radians(output_fwhm_deg), lmax=lmax)
        # R_l = G_l / B_l
        if beam_format == 'ELL_BL':
            R0 = bl0 
            R2 = bl2
        else:
            R0 = g0 / bl0
            R2 = np.zeros_like(R0); R2[2:] = g0[2:] / bl2[2:]
    else:
        # No-beam path: just reconvolve to target FWHM? (or identity)
        # If you want identity when no beam: set R to ones
        g0 = hp.gauss_beam(np.radians(output_fwhm_deg), lmax=lmax)
        R0 = g0
        R2 = np.zeros_like(R0); R2[2:] = g0[2:]

    return R0, R2, pixwin


def apply_transfer_to_maps(
    I: np.ndarray, Q: np.ndarray, U: np.ndarray, coord: str,
    R0: np.ndarray, R2: np.ndarray, pixwin: np.ndarray,
    lmax: int, nside_out: int, apodise_inpaint : bool=False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply harmonic-domain transfer functions to I,Q,U.

    Parameters
    ----------
    I, Q, U : 1D HEALPix maps (NSIDE_in)
        Input sky maps. UNSEEN is respected.
    R0 : array-like, shape (>= lmax+1,)
        Transfer function for spin-0 (e.g., G_l / B_l with any tapers).
    R2 : array-like, shape (>= lmax+1,)
        Transfer function for spin-2 (same convention; l<2 typically zeroed).
    pixwin : array-like, shape (>= lmax+1,)
        Pixel-window *ratio* to apply (usually pw_out / pw_in). If you’ve
        already folded pixel windows into R0/R2 upstream, pass an array of ones.
    lmax : int
        Maximum multipole for transforms.
    nside_out : int
        Output NSIDE (may equal input).

    Returns
    -------
    dI, dQ, dU : np.ndarray
        Transformed maps at NSIDE_out with UNSEEN propagated.
    """
    # --- sanity & prep
    I = np.asarray(I, dtype=float)
    Q = np.asarray(Q, dtype=float)
    U = np.asarray(U, dtype=float)
    I[I < -1e20] = hp.UNSEEN
    Q[Q < -1e20] = hp.UNSEEN
    U[U < -1e20] = hp.UNSEEN

    nside_in = hp.get_nside(I)
    if nside_out is None:
        nside_out = nside_in

    # Unified valid mask: require all three to be finite and not UNSEEN
    valid = np.isfinite(I) & np.isfinite(Q) & np.isfinite(U) \
            & (I != hp.UNSEEN) & (Q != hp.UNSEEN) & (U != hp.UNSEEN)

    # If nothing valid, just return UNSEENs of right length
    if not np.any(valid):
        npix_out = 12 * nside_out**2
        unseen = np.full(npix_out, hp.UNSEEN, dtype=float)
        return unseen, unseen.copy(), unseen.copy()

    # Work on copies to avoid modifying caller arrays
    Ii = I.copy()
    Qi = Q.copy()
    Ui = U.copy()

    # Subtract a robust offset from I to mitigate ringing; add back later
    # (use only valid pixels for the statistic)
    i_offset = np.median(Ii[valid]) if np.any(valid) else 0.0
    Ii[valid] -= i_offset

    # Apply Gilles inpainting and apodisation scheme 
    if apodise_inpaint:
        dec_apo, sm_apo, k, n_iter = -15.6 - 6, 4, 2, 200
        Ii[~valid] = hp.UNSEEN
        Qi[~valid] = hp.UNSEEN
        Ui[~valid] = hp.UNSEEN
        Ii = inpainting.inpaint_and_apodise(Ii, coord, dec_apo, sm_apo, n_iter=n_iter, k=k, set_0=False)
        Qi = inpainting.inpaint_and_apodise(Qi, coord, dec_apo, sm_apo, n_iter=n_iter, k=k, set_0=True)
        Ui = inpainting.inpaint_and_apodise(Ui, coord, dec_apo, sm_apo, n_iter=n_iter, k=k, set_0=True)
    else:
        # Replace invalids with 0 to avoid NaNs in spherical transforms
        Ii[~valid] = 0.0
        Qi[~valid] = 0.0
        Ui[~valid] = 0.0

    # Ensure there are no invalid pixels left for spherical transforms.
    for arr in (Ii, Qi, Ui):
        arr[~np.isfinite(arr) | (arr == hp.UNSEEN)] = 0.0
        
    # map2alm for T,E,B in one call (spin-2 decomposition for Q/U)
    almT, almE, almB = hp.map2alm([Ii, Qi, Ui], lmax=lmax, pol=True, use_weights=False)

    # Combine transfer with pixel-window ratio
    # (R* = R*_in × (pw_out/pw_in); caller can pass pixwin=zeros_like if already applied)
    R0_eff = np.asarray(R0, dtype=float)[:lmax+1] * np.asarray(pixwin, dtype=float)[:lmax+1]
    R2_eff = np.asarray(R2, dtype=float)[:lmax+1] * np.asarray(pixwin, dtype=float)[:lmax+1]

    # Apply filters
    almT = hp.almxfl(almT, R0_eff)
    almE = hp.almxfl(almE, R2_eff)
    almB = hp.almxfl(almB, R2_eff)

    # Back to maps at NSIDE_out
    dI, dQ, dU = hp.alm2map((almT, almE, almB), nside=nside_out, pol=True, verbose=False)

    # Restore I offset on valid output pixels
    # Build output validity by downgrading the original valid mask
    valid_out = hp.ud_grade(valid.astype(float), nside_out) > 0.5
    dI[valid_out] += i_offset

    # Propagate UNSEEN to invalid-out pixels
    dI[~valid_out] = hp.UNSEEN
    dQ[~valid_out] = hp.UNSEEN
    dU[~valid_out] = hp.UNSEEN

    return dI, dQ, dU


def apply_transfer_to_cov(
    covII: np.ndarray, covQQ: np.ndarray, covUU: np.ndarray,
    covQU: np.ndarray,
    R0: np.ndarray, R2: np.ndarray, pixwin: np.ndarray,
    lmax: int, nside_out: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministically propagate per-pixel covariance maps through a linear,
    isotropic transfer (R0 for I, R2 for Q/U). Operates in harmonic space.

    Parameters
    ----------
    covII, covQQ, covUU, covIQ, covIU, covQU : 1D HEALPix maps (NSIDE_in)
        Pixel-space covariance components. UNSEEN respected.
    R0, R2 : array-like (>= lmax+1)
        Transfer functions for spin-0 and spin-2 (as used on the sky maps).
    pixwin : array-like (>= lmax+1)
        Pixel-window *ratio* W_out / W_in. If R0/R2 already include this, pass ones.
    lmax : int
        Max multipole to use.
    nside_out : int
        Output NSIDE.

    Returns
    -------
    dII, dQQ, dUU, dIQ, dIU, dQU : np.ndarray
        Transformed covariance components at NSIDE_out with UNSEEN propagated.
    """
    # Ensure arrays
    covII = np.asarray(covII, dtype=float)
    covQQ = np.asarray(covQQ, dtype=float)
    covUU = np.asarray(covUU, dtype=float)
    covQU = np.asarray(covQU, dtype=float)

    nside_in = hp.get_nside(covII)
    if nside_out is None:
        nside_out = nside_in

    # Valid mask: wherever any of the covariance entries are valid we try to propagate;
    # but for safety, use covII’s validity as the base (common in your files)
    base_valid = (covII != hp.UNSEEN) & np.isfinite(covII)
    if not np.any(base_valid):
        npix_out = 12 * nside_out**2
        unseen = np.full(npix_out, hp.UNSEEN, dtype=float)
        return unseen, unseen.copy(), unseen.copy(), unseen.copy(), unseen.copy(), unseen.copy()

    # Replace invalids with 0 so harmonics behave
    for m in (covII, covQQ, covUU, covQU):
        bad = ~np.isfinite(m) | (m == hp.UNSEEN)
        m[bad] = 0.0

    # Build effective kernels for covariance components
    R0 = np.asarray(R0, dtype=float)[:lmax+1]
    PW = np.asarray(pixwin, dtype=float)[:lmax+1]

    theta = np.linspace(0,np.pi,10800)
    beam = hp.bl2beam(R0*pixwin, theta) 
    K0 = hp.beam2bl(beam**2, theta, lmax) 


    # Helper to transform a scalar map with kernel K_ell
    def _transform_scalar(m: np.ndarray, K: np.ndarray) -> np.ndarray:
        alm = hp.map2alm(m, lmax=lmax, pol=False, use_weights=False)
        alm = hp.almxfl(alm, K)
        out = hp.alm2map(alm, nside=nside_out, verbose=False)
        pixel_area = hp.nside2pixarea(nside_in)
        return out*pixel_area

    # Apply per-component
    dII = _transform_scalar(covII, K0)
    dQQ = _transform_scalar(covQQ, K0)
    dUU = _transform_scalar(covUU, K0)
    dQU = _transform_scalar(covQU, K0)

    # Build output validity by downgrading the base_valid mask
    valid_out = hp.ud_grade(base_valid.astype(float), nside_out) > 0.5
    for m in (dII, dQQ, dUU, dQU):
        m[~valid_out] = hp.UNSEEN
        # tiny negative clamp from numerical noise
        neg = (m != hp.UNSEEN) & (m < 0)
        if np.any(neg):
            m[neg] = 0.0

    return dII, dQQ, dUU, dQU

def dec_mask(nside: int, coord: str, min_dec: float) -> np.ndarray:
    """
    Build a 1/0 mask selecting pixels with declination >= min_dec (degrees).
    If the input maps are in Galactic ('G'), convert pixel coordinates to
    Celestial to compute declination.

    Parameters
    ----------
    nside : int
        HEALPix NSIDE.
    coord : str
        'G' for Galactic or 'C' for Celestial (case-insensitive).
    min_dec : float
        Minimum declination in degrees to keep (>=).

    Returns
    -------
    mask : np.ndarray
        Float array of shape (12*nside^2,), values in {0.0, 1.0}.
    """
    coord = (coord or "G").upper()
    npix = 12 * nside**2

    # Angles for all pixels in the map's native frame
    theta, phi = hp.pix2ang(nside, np.arange(npix, dtype=int))  # native frame

    # Convert to Celestial angles if needed to compute declination
    if coord == "G":
        rot = hp.Rotator(coord=["G", "C"])
        theta, phi = rot(theta, phi)
    elif coord != "C":
        raise ValueError(f"dec_mask: unknown coord '{coord}' (use 'G' or 'C').")

    # Declination (deg): dec = 90° - theta_deg
    dec = 90.0 - np.degrees(theta)

    mask = (dec >= float(min_dec))
    return mask
