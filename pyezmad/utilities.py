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
linelist_vac = {'OII3726'  : 3727.09,
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

linelist_air = {'OII3726'  : 3726.03,
                'OII3729'  : 3728.82,
                'H12'      : 3750.15,
                'H11'      : 3770.63,
                'H10'      : 3797.90,
                'H9'       : 3835.39,
                'NeIII3869': 3868.71,
                'H8'       : 3889.05,
                'NeIII3967': 3967.41,
                'Hepsilon' : 3970.07,
                'Hdelta'   : 4101.76,
                'Hgamma'   : 4340.47,
                'Hbeta'    : 4861.33,
                'OIII4959' : 4958.92,
                'OIII5007' : 5006.84,
                'NII6548'  : 6548.03,
                'NII6584'  : 6583.41,
                'Halpha'   : 6562.80,
                'SII6717'  : 6716.47,
                'SII6731'  : 6730.85,
                'NII5755'  : 5754.64,
                'OIII4363' : 4363.21,
                'OI6300'   : 6300.30,
                'HeI3889'  : 3888.65,
                'HeI5876'  : 5875.67,
                'HeI6678'  : 6678.152,
                # duplicated keys...
                # 'NII6549'  : 6585.23,
                'NII6583'  : 6583.41
                }


linelist = linelist_air



def get_wavelength(hdu, ext=1, axis=None):
    """Create a wavelength array from an input Astropy HDU object.

    Parameters
    ----------
    hdu : astropy.io.fits.HDUList
        HDUList object.
    axis : int
        Index of axis in which the wavelength coordinate is stored
        (3 for a standard MUSE cube).
    ext : int or string, optional
        Index of the extention from which wavelength info will be extracted
        (1 or 'DATA' or 2 or 'STAT' is a standard for a MUSE cube; Defalt 1).

    Returns
    -------
    wavelength : ndarray
        Wavelength array reconstructed from the header information.

    """

    if axis==None:
        print("need to specify axis (NAXISn)")
        sys.exit()

    h = hdu[ext].header

    if 'CD%i_%i' % (axis,axis) in h:
        if h['CD%i_%i' % (axis,axis)]==1:
            cdelt = h['CDELT%i' % axis]
        else:
            cdelt = h['CD%i_%i' % (axis,axis)]
    elif 'CDELT%i' % axis in h:
        cdelt = h['CDELT%i' % axis]
    else:
        print("Neither CD%i_%i nor CDELT%i found in the header. Exit." % (axis,axis,axis))
        sys.exit()

    w = h['CRVAL%i' % axis] + cdelt*(np.arange(h['NAXIS%i' % axis])-h['CRPIX%i' % axis]+1)

    return(w)


def search_nearest_index(x, x0):
    """Search nearest index to the input value.

    Parameters
    ----------
    x : array_like
        An array to be searched.
    x0 : float
        A value for which the nearest index will be searched.

    Returns
    -------
    index : int
       Index value for which (x-x0) is minimized.
    """
    return(np.argmin(np.abs(x-x0)))


def create_whitelight_image(infile, prefix_out,
                            is_save=True, is_plot=True,
                            wi_scale='linear', wi_percent=99.,
                            wi_cmap=cm.Greys_r,
                            w_begin=4750., w_end=9350., ext=1):
    """Create MUSE white light image and save it to FITS file if specified.

    Simple white light image is produced for the input MUSE cube.  
    Some keywords for plotting is accepted.

    Parameters
    ----------
    infile : str
        An input MUSE cube. 
    prefix_out : str
        A prefix for the output FITS file and PDF image if requested.
    is_plot : bool, optional
        Make a plot of the white light image if True.  The default is `True`.
        The plot will be saved as prefix_out.pdf.
    is_save : bool, optional
        Save the white light image as a FITS file. The default is `True`.
        The name of the FITS file will be prefix_out.pdf.
    wi_scale: {'linear', 'sqrt', 'power', 'log', 'asinh'}, optional
        Scaling function for plotting.
        See ``astropy.visualization.scale_image``. The default is ``linear``.
    wi_percent: int or float, optional
        Percentile for clipping for plotting. 
        See ``astropy.visualization.scale_image``. The default is ``99``.
    wi_cmap: matplotlib colormap object, optional
        Colormap for plotting. The default is `matplotlib.cm.Greys_r`.
    w_begin: int or float, optional
        Starting wavelength of white light image in angstrom. The default is 4750.
    w_end: int or float, optional
        End wavelength of white light image in angstrom. The default is 9350.
    ext: int or str, optional
        FITS Extention where the data is stored. The default is 1.

    Returns
    -------
    image : ndarray
        A white light image with a shape of (NAXIS2, NAXIS1).
    """

    hdu = fits.open(infile)

    w = get_wavelength(hdu, axis=3)

    iw_begin = search_nearest_index(w, w_begin)
    iw_end = search_nearest_index(w, w_end)

    wi = np.nansum(hdu[ext].data[iw_begin:iw_end,:,:], axis=0)

    if is_save==True:
        fits.writeto(prefix_out+'.fits', wi, hdu[ext].header, clobber=True)

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
    """Create a narrow band image.  Possibiilty to input velocity structures. 

    Parameters
    ----------
    hducube : HDU object
        Input HDU object.
    wcenter : float or int
        Central wavelength in angstrom.
    dw : float or int, optional
        If specified, narrow-band image will be extracted ``wcenter+/-dw``.
    vel : float, int, or array_like
        Input velocity in km/s
        (either scaler or 2D array with the same spatial dimension as that of the input cube).
        If specfied, wcenter is considered as the rest-frame wavelength and
        will be shited accordingly to the observed frame for the extraction.
    vdisp : float, int, or array_like
        Velocity dispersion in km/s (either scalar or 2D map like velmap).
    nsig : int or float
        Narrow-band extraction is carried out to ``nsig`` times the velocity dispersion.

    Returns
    -------
    image : ndarray
        Narrow-band image with a shape of (NAXIS2, NAXIS1).
    """

    h = hducube[1].header

    wcube = get_wavelength(hducube, axis=3)
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

    # FIXME: Looping is very slow in Python. There must be more efficient way.
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
    """Create a narrow band image in a simple way. 

    Parameters
    ----------
    hducube : HDU object
        Input HDU object for a cube.
    wcenter : float or int
        Central wavelength in angstrom.
    dw :
        Extraction width in angstrom.
        Narrow-band image will be extracted for ``wcenter+/-dw``.

    Returns
    -------
    image : ndarray
        Narrow-band image with a shape of (NAXIS2, NAXIS1).
    """

    h = hducube[1].header

    wcube = get_wavelength(hducube, axis=3)

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
    """A factor to convert from per pixel to per arcsec.

    Parameters
    ----------
    pixscale : float, optional
        Pixel scale in arcsec/pixel. The default is 0.2.

    Returns
    -------
    float :
        Scalar to convert from per sq. pixel to per sq. arcsec.
        For instance, one can convert flux f in :math:`erg/s/cm^2/A/pix`
        to in :math:`erg/s/cm^2/A/arcsec^2`
        by ``f * per_pixel_to_arcsec()**2``.
    """
    a = 1./pixscale
    return(a)


def per_pixel_to_physical(distance, scale='kpc', pixscale=0.2):
    """A factor to convert from per pixel to per physical length (kpc or pc).

    Parameters
    ----------
    distance : float
        a distance to the object in Mpc (astropy.unit instance is recommended)
    scale : {'kpc', 'pc'}, optional
        Unit to be converted either per 'kpc' or 'pc' (Other units may work).
        The default is 'kpc'.
    pixscale: float
        Pixel scale in arcsec/pixel The default is 0.2.

    Returns
    -------
    float :
        Scalar to convert from per pixel to per physical angular size with the given unit.
        For instance, one can convert flux f in :math:`erg/s/cm^2/A/pix^2` to
        in :math:`erg/s/cm^2/A/kpc^2`
        by ``f * per_pixel_to_physical(distance, scale='kpc')**2``.
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
