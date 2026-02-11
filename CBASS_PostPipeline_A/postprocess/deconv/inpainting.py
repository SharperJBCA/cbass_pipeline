from astropy.coordinates import SkyCoord
from astropy import units as u
from tqdm import tqdm
import numpy as np 
import healpy as hp 


def pix_under_dec(nside, lat_cut_deg):
    
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    lon = np.degrees(phi)
    lat = 90.0 - np.degrees(theta)
    gal = SkyCoord(l=lon*u.deg, b=lat*u.deg, frame="galactic")
    icrs = gal.icrs
    dec = icrs.dec.deg
    return np.where(dec < lat_cut_deg)[0]


def inpaint_and_apodise(hpmap, decapo=-21.6, smapo=4, n_iter=200, k=2, set_0=False):
    """
    Iteratively inpaint outside the survey
    
    Parameters
    ----------
    hpmap : input I, Q or U map.
    decapo : declination where transition from inpainting to median (or 0) 
    smapo : smoothness of transition  
    k : how many SEEN neighbour pixels to inpaint an UNSEEN pixel (k=1: inpainting along Q & U, k=2: inpainting along normal direction)   
    n_iter : how many iterations of the diffusion inpainting are performed (more needed at high Nside or when decapo far from edge) 
    """ 
    
    mask = (hpmap != hp.UNSEEN)
    map_inpainted = hpmap.copy()
    map_inpainted[map_inpainted == hp.UNSEEN] = np.nan
    nside = hp.npix2nside(len(hpmap))
    npix = 12*nside**2
    neighbors = hp.get_all_neighbours(nside, np.arange(len(hpmap)))
    mask = mask.copy()
    
    # iteratively inpaints
    for _ in tqdm(range(n_iter)):    
        boundary = (~mask) & (np.sum(mask[neighbors] , axis=0)>=k)
        for pix in np.where(boundary)[0]:
            neighs = neighbors[:, pix]
            neighs = neighs[neighs >= 0]  # drop invalid indices
            valid_neighs = neighs[mask[neighs]]
            if len(valid_neighs) > 0:
                map_inpainted[pix] = np.nanmean(map_inpainted[valid_neighs])

        mask[boundary] = True
        
    # what's not inpainted is set to median or to 0
    if not set_0:
        finite_seen = np.isfinite(hpmap) & (hpmap != hp.UNSEEN)
        complete_value = np.nanmedian(hpmap[finite_seen]) if np.any(finite_seen) else 0.0
    else:
        complete_value = 0
    map_inpainted[np.isnan(map_inpainted)] = complete_value

    # the transition is made smoothed
    list_pix_drop = pix_under_dec(nside, decapo)
    mask_bin = np.ones((12*nside**2))
    mask_bin[list_pix_drop] = 0
    if smapo!=0: 
        apo_mask = hp.smoothing(mask_bin, smapo/180*np.pi)
        apo_mask[apo_mask<0.001] = 0
        apo_mask[apo_mask>0.999] = 1
    else: 
        apo_mask = mask_bin
    map_inpainted_apo = map_inpainted * apo_mask + complete_value * (1-apo_mask)

    map_inpainted_apo[np.isnan(map_inpainted_apo)] = hp.UNSEEN
    return map_inpainted_apo
