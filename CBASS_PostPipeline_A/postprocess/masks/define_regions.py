import numpy as np
import healpy as hp
from astropy.io import fits

def smooth_mask(pixels, nside=128, fwhm=1,**kwargs):

    mtmp = np.zeros(12*nside**2)
    mtmp[pixels] = 1
    mtmp = hp.smoothing(mtmp,fwhm=fwhm*np.pi/180.)
    return np.where(mtmp > 0.5)[0]

def tothetaphi(lon,lat):
    return np.radians(90-lat), np.radians(lon)

def planck_source_catalogue(pixels, nside=128, fwhm=1,min_latitude=0, catalogue='none', hdu_index=1, flux_threshold=1,**kwargs):

    hdu = fits.open(catalogue,memmap=False)
    data = hdu[hdu_index]
    gl, gb = data.data['GLON'], data.data['GLAT']
    gaussian_flux = data.data['PSFFLUX']*1e-3 # Jy 
    hdu.close() 

    if min_latitude > 0:
        select = np.where(np.abs(gb) > min_latitude)[0]
        gl = gl[select]
        gb = gb[select]
        gaussian_flux = gaussian_flux[select]

    select = (gaussian_flux > flux_threshold)
    gl = gl[select]
    gb = gb[select]
    gaussian_flux = gaussian_flux[select]

    theta, phi = tothetaphi(gl,gb)
    pixels = hp.ang2pix(nside,theta,phi)

    mask = np.zeros(12*nside**2,dtype=bool)
    for i in range(pixels.size):
        mask[hp.query_disc(nside,hp.ang2vec(theta[i],phi[i]),fwhm*np.pi/180.)] = True

    return np.where(mask)[0]


def query_arc_pixels(pixels,nside=128, ra=0, dec=0, major=1, minor=None, width=1, pa=0, radians_input = False,theta_start=0,theta_end=2*np.pi,res=12,**kwargs):
    """Query pixels within an arc region.

    Builds up an arc by querying convex 4 sided polygons (rhombi?) along the arc

    Parameters
    ----------
    pixels : array-like
        Dummy pixels to be overwritten.
    nside : int
        Healpix nside.
    ra : float
        Right ascension of the center of the arc (degrees).
    dec : float
        Declination of the center of the arc (degrees).
    major : float
        Major axis of the arc (degrees).
    minor : float
        Minor axis of the arc (degrees).
    with : float
        Width of the arc (degrees).
    pa : float
        Position angle of the arc (degrees).

    Returns
    -------
    pixels : array-like
        Healpix pixels within the arc.
    """
    # Convert to radians
    if minor is None:
        minor = major

    if not radians_input:
        ra = np.radians(ra)
        dec = np.radians(dec)
        major = np.radians(major)
        minor = np.radians(minor)
        pa = np.radians(pa)
        width = np.radians(width)

    # Define the central line of the arc
    theta = np.linspace(theta_start, theta_end, res+1, endpoint=True)
    x = np.cos(theta) * minor
    y = np.sin(theta) * major
    # Now we need to define the boundaries of each rhombus. These will be determined by lines perpendicular to the arc line at each point.
    edge_x = []
    edge_y = []
    for i in range(x.size):
        # Define the perpendicular line
        x1 = x[i] + np.cos(theta[i])*width/2.
        y1 = y[i] + np.sin(theta[i])*width/2.
        x2 = x[i] - np.cos(theta[i])*width/2.
        y2 = y[i] - np.sin(theta[i])*width/2.
        edge_x.append(np.array([x1,x2]))
        edge_y.append(np.array([y1,y2]))
    edge_x = np.array(edge_x)
    edge_y = np.array(edge_y)
    edge_y = np.pi/2. - edge_y

    # Now rotate the arc line to the ra, dec, pa location 
    rot = hp.rotator.Rotator(rot=[np.degrees(ra),np.degrees(dec),np.degrees(pa)], inv=True) 
    edge_theta, edge_phi = rot(edge_y.flatten(), edge_x.flatten())
    edge_theta = np.reshape(edge_theta,edge_x.shape)
    edge_phi = np.reshape(edge_phi,edge_x.shape)

    pixel_rhombi = []
    # Find all pixels bound by arc by creating rhombi along arc using query_polygon, the number of rhombi is determined by res 
    for i in range(res):
        vec = [hp.ang2vec(edge_theta[i,0],edge_phi[i,0]),hp.ang2vec(edge_theta[i,1],edge_phi[i,1]),hp.ang2vec(edge_theta[i+1,1],edge_phi[i+1,1]),hp.ang2vec(edge_theta[i+1,0],edge_phi[i+1,0])]
        pixel_rhombi += [hp.query_polygon(nside,vec)] 

    # Find all pixels bound by ellipse 
    pixels = np.concatenate(pixel_rhombi)
    return pixels


def query_elliptical_pixels(pixels,nside=128, ra=0, dec=0, major=1, minor=None, pa=0, radians_input = False,theta_start=0,theta_end=2*np.pi,res=12,**kwargs):
    """Query pixels within an elliptical region.

    Parameters
    ----------
    pixels : array-like
        Dummy pixels to be overwritten.
    nside : int
        Healpix nside.
    ra : float
        Right ascension of the center of the ellipse (degrees).
    dec : float
        Declination of the center of the ellipse (degrees).
    major : float
        Major axis of the ellipse (degrees).
    minor : float
        Minor axis of the ellipse (degrees).
    pa : float
        Position angle of the ellipse (degrees).

    Returns
    -------
    pixels : array-like
        Healpix pixels within the ellipse.
    """
    # Convert to radians
    if minor is None:
        minor = major

    if not radians_input:
        ra = np.radians(ra)
        dec = np.radians(dec)
        major = np.radians(major)
        minor = np.radians(minor)
        pa = np.radians(pa)

    # Define the ellipse
    theta = np.linspace(theta_start, theta_end, res, endpoint=False)
    x = np.cos(theta) * minor
    y = np.pi/2. - np.sin(theta) * major

    if (theta_start != 0) | (theta_end != 2*np.pi):
        x = np.insert(x,0,0)
        y = np.insert(y,0,0)

    # Now rotate the ellipse to the ra, dec, pa location 
    rot = hp.rotator.Rotator(rot=[np.degrees(ra),np.degrees(dec),np.degrees(pa)], inv=True) 
    theta, phi = rot(y, x)

    # Find all pixels bound by ellipse 
    pixels = hp.query_polygon(nside, hp.ang2vec(theta[::-1], phi[::-1]))
    
    return pixels

def threshold_mask(pixels, map=np.zeros(1), threshold=None, threshold_pc=None,**kwargs):
    """Threshold a map and return the pixels above the threshold.

    Parameters
    ----------
    pixels : array-like
        Healpix pixels to threshold.
    map : array-like
        Healpix map to threshold.
    threshold : float
        Threshold value.

    Returns
    -------
    pixels : array-like
        Healpix pixels above threshold.
    """
    if threshold is None and threshold_pc is None:
        raise ValueError('Must specify threshold or threshold_pc.')
    elif threshold is not None and threshold_pc is not None:
        raise ValueError('Must specify only one of threshold or threshold_pc.')
    elif threshold_pc is not None:
        z = map[pixels] 
        z = z[z != hp.UNSEEN]
        threshold = np.percentile(z, threshold_pc)

    return pixels[np.where((map[pixels] > threshold) & (map[pixels] != hp.UNSEEN))[0]]

def remove_pixels_with_masked_neighbour(pixels,nside=128,**kwargs):
    """Remove pixels that have a masked neighbour.

    Parameters
    ----------
    pixels : array-like
        Healpix pixels to threshold.

    Returns
    -------
    pixels : array-like
        Healpix pixels above threshold.
    """
    # Get the neighbours of the pixels
    m = np.zeros(12*nside**2)
    m[pixels] = 1

    neighbours = hp.get_all_neighbours(nside, pixels)

    # Remove the pixels that have a masked neighbour
    return pixels[np.where(np.sum(m[neighbours], axis=0) > 6)[0]]

def remove_overlap_pixels(pixels_main,nside=128, pixels_cut=None):

    m = np.zeros(12*nside**2)
    m[pixels_main] = 1
    m[pixels_cut] += 2

    return np.where(m == 1)[0]

def subtract_elliptical_pixels(pixels,nside=128, ra=0, dec=0, major=1, minor=None, pa=0, radians_input = False,theta_start=0,theta_end=2*np.pi,res=12,**kwargs):
    pixels_to_subtract = query_elliptical_pixels(pixels,nside=nside, ra=ra, dec=dec, major=major, minor=minor, pa=pa, radians_input = radians_input,theta_start=theta_start,theta_end=theta_end,res=res)

    pixels = remove_overlap_pixels(pixels,pixels_cut=pixels_to_subtract,nside=nside)
    return pixels 

def latitude_cut(pixels, nside=128, latitude=0, model='>',**kwargs):
    """Cut pixels based on latitude.

    Parameters
    ----------
    pixels : array-like
        Healpix pixels to threshold.
    latitude : float
        Latitude to cut at (degrees).
    model : str
        Cut model. Can be '>' or '<'.

    Returns
    -------
    pixels : array-like
        Healpix pixels above threshold.
    """
    theta, phi = hp.pix2ang(nside, pixels)
    theta = np.pi/2. - theta
    latitude = np.radians(latitude)
    if model == '<':
        return pixels[np.where(theta > latitude)[0]]
    elif model == '>':
        return pixels[np.where(theta < latitude)[0]]
    else:
        raise ValueError('Invalid model.')
    
def remove_bad_data_mask(pixels,map=None,**kwargs):
    """Mask bad data.

    Parameters
    ----------
    m : array-like
        Healpix map.
    pixels : array-like
        Healpix pixels to threshold.

    Returns
    -------
    pixels : array-like
        Healpix pixels above threshold.
    """
    return pixels[(map[pixels] != hp.UNSEEN)]

def bad_data_mask(pixels,map=None,**kwargs):
    """Mask bad data.

    Parameters
    ----------
    m : array-like
        Healpix map.
    pixels : array-like
        Healpix pixels to threshold.

    Returns
    -------
    pixels : array-like
        Healpix pixels above threshold.
    """
    return pixels[(map[pixels] == hp.UNSEEN)]
