#!/usr/bin/env python

import sys
import os.path
import numpy as np

import astropy.io.fits as fits
from astropy.visualization import scale_image
import astropy.units as u
from astropy.constants import c
# import aplpy

import matplotlib.pyplot as plt
import matplotlib.cm as cm


# may need to make it read-only, but now I trust everybody!
linelist = {'OII3726'  : 3727.09,
            'OII3729'  : 3729.88,
            'H12'      : 3751.22,
            'H11'      : 3771.70,
            'H10'      : 3798.98,
            'H9'       : 3836.48,
            'NeIII3869': 3869.81,
            'H8'       : 3890.15,
            'NeIII3967': 3968.53,
            'Hepsilon' : 3971.19,
            'Hdelta'   : 4102.92,
            'Hgamma'   : 4341.69,
            'Hbeta'    : 4862.69,
            'OIII4959' : 4960.30,
            'OIII5007' : 5008.24,
            'NII6548'  : 6549.84,
            'NII6584'  : 6585.23,
            'Halpha'   : 6564.61,
            'SII6717'  : 6718.32,
            'SII6731'  : 6732.71,
            'NII5755'  : 5756.24,
            'OIII4363' : 4364.44,
            'OI6300'   : 6302.04,
            'HeI3889'  : 3889.75,
            'HeI5876'  : 5877.30,
            'HeI6678'  : 6679.996,
            # duplicated keys...
            'NII6583'  : 6585.23
            }



def get_wavelength(hdu, axis=None):

    if axis==None:
        print("need to specify axis (NAXISn)")
        sys.exit()

    h = hdu[1].header

    if 'CD%i_%i' % (axis,axis) in h:
        if h['CD%i_%i' % (axis,axis)]==1:
            cdelt = h['CDELT%i' % axis]
        else:
            cdelt = h['CD%i_%i' % (axis,axis)]
    elif 'CDELT%i' % axis in h:
        cdelt = h['CDELT%i' % axis]
    else:
        print("CD%i_%i or CDELT%i not found in the header. Exit." % (axis,axis,axis))
        sys.exit()

    w = h['CRVAL%i' % axis] + cdelt*(np.arange(h['NAXIS%i' % axis])-h['CRPIX%i' % axis]+1)

    return(w)


def search_nearest_index(x, x0):
    return(np.argmin(np.abs(x-x0)))

def create_whitelight_image(infile, prefix_out,
                            is_save=True, is_plot=True,
                            wi_scale='linear', wi_percent=99.,
                            wi_cmap=cm.Greys_r):
    """Create MUSE white light image and save it to FITS file if specified.

    Simple white light image is produced for the input MUSE cube.  
    Some keywords for plotting is accepted.

    Args:
        infile: An input MUSE cube
        prefix_out: prefix for the output FITS file and PDF image if requested.
        wi_scale: Scaling function for plotting
                  (see astropy.visualization.scale_image(); default: linear)
        wi_percent: Percentile for clipping for plotting
                  (see astropy.visualization.scale_image(); default: 99)
        wi_cmap: Colormap for plotting (default: cm.Greys_r)
        is_plot: Flag if plot is required (default: True)
        is_save: Flag if white light image will be saved to a FITS file (default: True)

    Returns:
        A 2D numpy array of white image with a dimension of (NAXIS2, NAXIS1)
    """

    hdu = fits.open(infile)

    wi = np.nansum(hdu[1].data, axis=0)

    if is_save==True:
        fits.writeto(prefix_out+'.fits', wi, hdu[1].header, clobber=True)

    if is_plot==True:
        wi_cmap.set_bad('white')
        wi_scaled = scale_image(wi, scale=wi_scale, percent=wi_percent)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.imshow(wi_scaled,
                origin='lower',
                interpolation='none',
                cmap=wi_cmap)
        ax.set_xlabel('X [pixel]')
        ax.set_ylabel('Y [pixel]')
        plt.savefig(prefix_out+'.pdf', bbox_inches='tight')

    return(wi)


def create_narrowband_image(hducube, wcenter, dw=None, vel=None, vdisp=None, nsig=3.):
    """Create a narrow band image

    Args:
        hducube: HDU object of astropy.io.fits for the input cube
        wcenter: central wavelength in angstrom
        dw: if specified, narrow-band image will be extracted wcenter+/-dw
        vel: input velocity (either scaler or 2D map with a same spatial dimension with the input cube)
                If specfied, wcenter is considered as the rest-frame wavelength and
                will be shited accordingly to the observed frame for the extraction.
        vdisp: velocity dispersion in km/s (either scalar or 2D map like velmap).
        nsig: narrow-band extract is carried out out to nsig times the velocity dispersion.

    Returns:
        2D numpy array with a shape of (NAXIS2, NAXIS1)
    """

    h = hducube[1].header

    if 'CD3_3' in h:
        cdelt3 = h['CD3_3']
    elif 'CDELT3' in h:
        cdelt3 = h['CDELT3']
    else:
        print("CD3_3 or CDELT3 not found in the header. Exit.")
        sys.exit()

    wcube = h['CRVAL3'] + cdelt3*(np.arange(h['NAXIS3'])-h['CRPIX3']+1)
    nbimg = np.empty((h['NAXIS2'],h['NAXIS1']))
    maskimg = np.nanmax(hducube[1].data, axis=0)
    maskimg[np.isfinite(maskimg)] = 1.

    if vel==None:
        # velmap = np.zeros_like(nbimg)
        zz = 1.
        ww = wcenter * np.ones_like(nbimg)
    else:
        if type(vel) != np.ndarray:
            vel = np.ones_like(nbimg) * vel
        zz = (1.+vel*u.km/u.s/c.to('km/s'))
        ww = wcenter * zz

    if dw == None:
        if type(vdisp) != np.ndarray:
            vdisp = np.ones_like(nbimg) * vdisp
        dwave = (vdisp*u.km/u.s) / c.to('km/s') * ww * nsig
    elif vdisp==None:
        dwave = dw * np.ones_like(nbimg)
        zz = 0.
    else:
        print("One of 'dw' or 'velmap/sigmap' must be specified! Exit.")
        sys.exit()

    wmin, wmax = ww-dwave, ww+dwave

    # FIXME: stupid looping is slow. There must be more efficient way.
    #        Need to calm down and relax to think about it!
    for ix in xrange(h['NAXIS1']):
        for iy in xrange(h['NAXIS2']):
            if np.isnan(maskimg[iy,ix])==True:
                continue
            idx_wmin = search_nearest_index(wcube, wmin[iy,ix])
            idx_wmax = search_nearest_index(wcube, wmax[iy,ix])
            tmpspec = hducube[1].data[idx_wmin:idx_wmax+1, iy, ix]
            nbimg[iy,ix] = np.nansum(tmpspec)

    nbimg[np.isnan(maskimg)] = np.nan

    return(nbimg)



def create_narrowband_image_simple(hducube, wcenter, dw):
    """Create a narrow band image

    Args:
        hducube: HDU object of astropy.io.fits for the input cube
        wcenter: central wavelength in angstrom
        dw: if specified, narrow-band image will be extracted wcenter+/-dw

    Returns:
        2D numpy array with a shape of (NAXIS2, NAXIS1)
    """

    h = hducube[1].header

    if 'CD3_3' in h:
        cdelt3 = h['CD3_3']
    elif 'CDELT3' in h:
        cdelt3 = h['CDELT3']
    else:
        print("CD3_3 or CDELT3 not found in the header. Exit.")
        sys.exit()

    wcube = h['CRVAL3'] + cdelt3*(np.arange(h['NAXIS3'])-h['CRPIX3']+1)
    nbimg = np.empty((h['NAXIS2'],h['NAXIS1']))
    maskimg = np.nanmax(hducube[1].data, axis=0)
    maskimg[np.isfinite(maskimg)] = 1.

    wmin, wmax = wcenter-dw, wcenter+dw

    idx_wmin = search_nearest_index(wcube, wmin)
    idx_wmax = search_nearest_index(wcube, wmax)
    tmpspec = hducube[1].data[idx_wmin:idx_wmax+1, :, :]
    nbimg = np.nansum(tmpspec, axis=0)

    nbimg[np.isnan(maskimg)] = np.nan

    return(nbimg)



def per_pixel_to_arcsec(pixscale=0.2):
    """Convert area from sq. pixel to sq. arcsec.

    Args:
        pixscale: pixel scale in arcsec/pixel (default: 0.2)

    Outputs:
        Scalar to convert from per sq. pixel to per sq. arcsec.
        For instance, one can convert flux f in erg/s/cm^2/A to in erg/s/cm^2/A/arcsec^2
        by f*per_pixel_to_arcsec()**2
    """
    a = 1./pixscale
    return(a)


def per_pixel_to_physical(distance, scale='kpc', pixscale=0.2):
    """Convert area from sq. pixel to per physical area (kpc^2 or pc^2)

    Args:
        distance: a distance to the object in Mpc (astropy.unit instance is recommended)
        scale: unit to be converted either per 'kpc' or 'pc'. (Other units will actually work)
        pixscale: pixel scale in arcsec/pixel (default: 0.2)

    Outputs:
        Scalar to convert from per sq. pixel to per sq. physical angular size with the given unit.
        For instance, one can convert flux f in erg/s/cm^2/A to in erg/s/cm^2/A/kpc^2
        by f*per_pixel_to_physical(distance, scale='kpc')**2
    """

    try:
        dummy = distance.unit.name
    except AttributeError:
        distance *= u.Mpc
        print("Warning: Distance is forced to be in astropy.units.Mpc")

    if scale == 'pc':
        pixscale2physical = distance.to('pc') * (pixscale*u.arcsec).to('radian') / u.radian
    elif scale == 'kpc':
        pixscale2physical = distance.to('kpc') * (pixscale*u.arcsec).to('radian') / u.radian

    return(1./pixscale2physical.value)


if __name__=='__main__':
    print('do nothing')
    print(per_pixel_to_arcsec())
    print(per_pixel_to_physical(20.))
    print(per_pixel_to_physical(20.*u.Mpc, scale='pc'))
