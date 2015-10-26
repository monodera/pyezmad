#!/usr/bin/env python

import os.path
import numpy as np
import numpy.polynomial.polynomial as polynomial

import astropy.io.fits as fits
from astropy.visualization import scale_image
import astropy.units as u
from astropy.constants import c

from astroquery.ned import Ned

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pyezmad

# maybe there is a better way to set this number...
# sigma2fwhm = 2.3548200
sigma2fwhm = 2. * np.sqrt(2. * np.log(2.))


def error_fraction(x1, x2, ex1, ex2):
    """Compute error of x1/x2 when errors of x1 and x2 are given.
    """

    err2 = (ex1 / x2)**2 + (x1 * ex2 / x2**2)**2
    err = np.sqrt(err2)

    return(err)


def error_log10_fraction(x1, x2, ex1, ex2):
    """Compute error of log10(x1/x2)
    when errors of x1 and x2 are given.
    """

    # err2 = (ex1 / x2)**2 + (x1 * ex2 / x2**2)**2
    # err = np.sqrt(err2)

    err = error_fraction(x1, x2, ex1, ex2) / (x1 / x2) / np.log(10.)

    return(err)


def read_emission_linelist(wavelength='air'):
    path_to_database = os.path.join(pyezmad.__path__[0],
                                    'database/emission_lines.dat')
    # print(path_to_database)
    linelist_all = np.genfromtxt(path_to_database,
                                 dtype=None,
                                 names=['line', 'w_air', 'w_vac'])
    linelist_air = {}
    linelist_vac = {}
    for i in range(linelist_all['line'].size):
        linelist_air[linelist_all['line'][i]] = linelist_all['w_air'][i]
        linelist_vac[linelist_all['line'][i]] = linelist_all['w_vac'][i]

    if wavelength == 'air':
        return(linelist_air)
    elif wavelength == 'vacuum':
        return(linelist_vac)
    else:
        raise(ValueError("wavelength=%s (air or vacuum) is not supported." %
                         wavelength))


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

    if axis is None:
        raise(ValueError("axis (NAXISn) is not defined."))

    h = hdu[ext].header

    if 'CD%i_%i' % (axis, axis) in h:
        if h['CD%i_%i' % (axis, axis)] == 1:
            cdelt = h['CDELT%i' % axis]
        else:
            cdelt = h['CD%i_%i' % (axis, axis)]
    elif 'CDELT%i' % axis in h:
        cdelt = h['CDELT%i' % axis]
    else:
        raise(ValueError(
            "Neither CD%i_%i nor CDELT%i found in the header. Exit." %
            (axis, axis, axis)))

    w = h['CRVAL%i' % axis] + cdelt * (np.arange(h['NAXIS%i' % axis]) -
                                       h['CRPIX%i' % axis] + 1)

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
    return(np.argmin(np.abs(x - x0)))


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
        Starting wavelength of white light image in angstrom.
        The default is 4750.
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

    wi = np.nansum(hdu[ext].data[iw_begin:iw_end, :, :], axis=0)

    if is_save is True:
        fits.writeto(prefix_out + '.fits', wi, hdu[ext].header, clobber=True)

    if is_plot is True:
        wi_cmap.set_bad('white')
        wi_scaled = scale_image(wi, scale=wi_scale, percent=wi_percent)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(wi_scaled,
                  origin='lower',
                  interpolation='none',
                  cmap=wi_cmap)
        ax.set_xlabel('X [pixel]')
        ax.set_ylabel('Y [pixel]')
        plt.savefig(prefix_out + '.pdf', bbox_inches='tight')

    return(wi)


def create_narrowband_image(hducube,
                            wcenter, dw=None,
                            vel=None, vdisp=None, nsig=3.):
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
        (either scaler or 2D array with the same spatial dimension
        as that of the input cube).
        If specfied, wcenter is considered as the rest-frame wavelength and
        will be shited accordingly to the observed frame for the extraction.
    vdisp : float, int, or array_like
        Velocity dispersion in km/s (either scalar or 2D map like velmap).
    nsig : int or float
        Narrow-band extraction is carried out to ``nsig``
        times the velocity dispersion.

    Returns
    -------
    image : ndarray
        Narrow-band image with a shape of (NAXIS2, NAXIS1).
    """

    h = hducube[1].header

    wcube = get_wavelength(hducube, axis=3)
    nbimg = np.empty((h['NAXIS2'], h['NAXIS1']))
    maskimg = np.nanmax(hducube[1].data, axis=0)
    maskimg[np.isfinite(maskimg)] = 1.

    if vel is None:
        # velmap = np.zeros_like(nbimg)
        zz = 1.
        ww = wcenter * np.ones_like(nbimg)
    else:
        if isinstance(vel, np.ndarray) is False:
            vel = np.ones_like(nbimg) * vel
        zz = (1. + vel * u.km / u.s / c.to('km / s'))
        ww = wcenter * zz

    if dw is None:
        if isinstance(vdisp, np.ndarray) is False:
            vdisp = np.ones_like(nbimg) * vdisp
        dwave = (vdisp * u.km / u.s) / c.to('km / s') * ww * nsig
    elif vdisp is None:
        dwave = dw * np.ones_like(nbimg)
        zz = 0.
    else:
        raise(ValueError(
            "One of 'dw' or 'velmap/sigmap' must be specified! Exit."))

    wmin, wmax = ww - dwave, ww + dwave

    # FIXME: Looping is very slow in Python. There must be more efficient way.
    for ix in xrange(h['NAXIS1']):
        for iy in xrange(h['NAXIS2']):
            if np.isnan(maskimg[iy, ix]) is True:
                continue
            idx_wmin = search_nearest_index(wcube, wmin[iy, ix])
            idx_wmax = search_nearest_index(wcube, wmax[iy, ix])
            tmpspec = hducube[1].data[idx_wmin:idx_wmax + 1, iy, ix]
            nbimg[iy, ix] = np.nansum(tmpspec)

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

    nbimg = np.empty((h['NAXIS2'], h['NAXIS1']))
    maskimg = np.nanmax(hducube[1].data, axis=0)
    maskimg[np.isfinite(maskimg)] = 1.

    wmin, wmax = wcenter - dw, wcenter + dw

    idx_wmin = search_nearest_index(wcube, wmin)
    idx_wmax = search_nearest_index(wcube, wmax)
    tmpspec = hducube[1].data[idx_wmin:idx_wmax + 1, :, :]
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
    return(1. / pixscale)


def per_pixel_to_physical(distance, scale='kpc', pixscale=0.2):
    """A factor to convert from per pixel to per physical length (kpc or pc).

    Parameters
    ----------
    distance : float
        Distance to the object in Mpc (astropy.unit instance is recommended)
    scale : {'kpc', 'pc'}, optional
        Unit to be converted either per 'kpc' or 'pc' (Other units may work).
        The default is 'kpc'.
    pixscale: float
        Pixel scale in arcsec/pixel The default is 0.2.

    Returns
    -------
    float :
        Scalar to convert from per pixel to per physical
        angular size with the given unit.
        For instance, one can convert flux f in :math:`erg/s/cm^2/A/pix^2` to
        in :math:`erg/s/cm^2/A/kpc^2`
        by ``f * per_pixel_to_physical(distance, scale='kpc')**2``.
    """

    if isinstance(distance, u.quantity.Quantity) is False:
        distance *= u.Mpc
        print("Warning: Distance is forced to be in astropy.units.Mpc")

    pixscale2physical = (pixscale * u.arcsec).to('radian') / u.radian

    if scale == 'pc':
        pixscale2physical *= distance.to('pc')
    elif scale == 'kpc':
        pixscale2physical *= distance.to('kpc')

    return(1. / pixscale2physical.value)


def get_ned_velocity(name=None):
    """A wrapper for Astroquery NED function to obtain
    the heliocentric (I believe) line-of-sight velocity
    of the specified galaxy.

    Parameters
    ----------
    name : str
        Name of a galaxy.

    Returns
    -------
    vel : float
        Line-of-sight velocity of the galaxy.
    """
    if name is None:
        raise(NameError("Variable name is not defined."))
    else:
        res = Ned.query_object(name)
        vel = res['Velocity'][0]
        print("Velocity of %s is retrived from NED as %6.1f km/s." %
              (name, vel))
        return(vel)


def get_ned_distance(name=None):
    """A wrapper for Astroquery NED function to obtain
    the distance to the specified galaxy.
    **Not yet implemented because ``astroquery`` does not have
    a function to retrieve distance entries from NED.**

    Parameters
    ----------
    name : str
        Name of a galaxy.

    Returns
    -------
    d : float
        Distance to the galaxy
    """

    raise(Exception("Sorry, not implemented yet."))

    if name is None:
        raise(NameError("Variable name is not defined."))
    else:
        res = Ned.query_object(name)
        d = res['Distance'][0]
        print("Distance to %s is retrived from NED as %6.1f Mpc." %
              (name, d))
        return(d)


def muse_fwhm(w, deg=2):
    """Return an array_like object with MUSE FWHM at given wavelengths.

    Parameters
    ----------
    w : array_like
        Wavelength in angstrom.
    deg : int, optional
        Polynomial degree for the best-fit FWHM(lambda) relation.

    Returns
    -------
    fwhm : array_like
        FWHM at given wavelength vector.
    """

    # set pivot wavelength as 7000A
    pivot_wavelength = 7000.
    ww = w - pivot_wavelength

    if deg == 2:
        coeff = np.array([2.30127694e+00, -5.69354340e-05, 6.37774915e-08])
    elif deg == 3:
        coeff = np.array([2.29582574e+00, -1.09992646e-04,
                          6.71840715e-08, 1.49922507e-11])
    else:
        raise(ValueError("deg must be 2 or 3"))

    return(polynomial.polyval(ww, coeff))


def map_pixel_major_axis(x, y, xc, yc, theta=0., ellip=0.):
    """Convert (x, y) to elliptical radius
    (equivalent to the semi-major axis length).

    Parameters
    ----------
    x, y : array_like
        Pixel coodinates to be converted.
    xc, yc : int or float
        Pixel coordinates of the central pixel.
    theta : float
        Position angle in degrees measured counter-clockwise
        from the north (or Y) axis.
    ellip : float
        Ellipticity defined as :math:`1-b/a` where
        :math:`a` and :math:`b` are major and minor axis length, respectively.
    """

    raise(Exception("Not yet implemented!"))

    if isinstance(x, float) or isinstance(x, int):
        x = np.array([x], dtype=np.float)
    if isinstance(y, float) or isinstance(y, int):
        y = np.array([y], dtype=np.float)

    radiant = (theta * u.deg).to('radian')
    axis_ratio = (1. - ellip)

    first_term = (x - xc) * np.cos(radiant) - (y - yc) * np.sin(radiant)
    second_term = (x - xc) * np.sin(radiant) + (y - yc) * np.cos(radiant)

    # first_term = (x - xc) * np.sin(radiant) - (y - yc) * np.cos(radiant)
    # second_term = (x - xc) * np.cos(radiant) + (y - yc) * np.sin(radiant)

    # rad_major_square = axis_ratio**2 * first_term**2 + second_term**2
    rad_major_square = (first_term*axis_ratio)**2 + (second_term)**2
    rad_major = np.sqrt(rad_major_square)

    return(rad_major.value)


if __name__ == '__main__':

    print('do nothing')
    print(per_pixel_to_arcsec())
    print(per_pixel_to_physical(20.))
    print(per_pixel_to_physical(20. * u.Mpc, scale='pc'))
    print(get_ned_velocity('NGC 7552'))
    print(get_ned_distance('NGC 7552'))
    # get_wavelength(fits.open('../test/ngc4980_voronoi_stack_spec_sn50_mwcorr.fits'))
