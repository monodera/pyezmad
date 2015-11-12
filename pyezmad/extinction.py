#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import sys
# import warnings
import numpy as np
from numpy.polynomial.polynomial import polyval
from scipy import interpolate
from astroquery.irsa_dust import IrsaDust

from .utilities import get_wavelength


def correct_dust_binspec(w, spec, ebv, extcurve='CCM'):
    """Correct dust extinction for binned spectra.

    TODO: Need to handle error. There are more room for improvement.

    Parameters
    ----------
    w : :py:class:`~numpy.ndarray`
        Wavelength in angstrom
    spec : :py:class:`~numpy.ndarray`
        Input spectra. Now, the 2nd dimension must be wavelength.
    ebv : :py:class:`~numpy.ndarray`
        E(B-V)
    extcuve : str, optional
        Extinction curve. "CCM" or "Calzetti".

    Returns
    -------
    spec_unred : :py:class:`~numpy.ndarray`
        Extinction corrected spectra with the same shape as the input.
    """

    alam_av, rv = get_alam_av(w, extcurve=extcurve)

    expo_corr_tile = np.empty_like(spec)
    for i in range(spec.shape[0]):
        expo_corr = 0.4 * ebv[i] * alam_av * rv
        expo_corr_tile[i, :] = expo_corr

    ext_factor = np.power(10., expo_corr_tile)
    spec_unred = spec * ext_factor

    return(spec_unred)


def get_alam_av(wave, extcurve=None):

    if extcurve in {'CCM', 'ccm'}:
        alam_av1, rv = extinction_ccm89(wave, ret_rv=True)
    elif extcurve in {'calzetti', 'Calzetti', 'Calz', 'calz'}:
        alam_av1, rv = extinction_calz(wave, ret_rv=True)
    elif extcurve is None:
        alam_av1, rv = 0., 0.
    else:
        raise(ValueError("%s curve is not yet implemented." % extcurve))

    return(alam_av1, rv)


def ebv_balmer_decrement(r_obs, err=None,
                         line1='Halpha', line2='Hbeta',
                         extcurve=None, clip=True):
    """Compute E(B-V) toward HII regions from the Balmer decrement.
    """
    balmer_wave = {'Halpha': 6564.62,
                   'Hbeta': 4862.69,
                   'Hgamma': 4341.69,
                   'Hdelta': 4102.92}
    balmer_flux = {'Halpha': 2.863,
                   'Hbeta': 1.0,
                   'Hgamma': 0.468,
                   'Hdelta': 0.259}

    if line1 in balmer_wave:
        wave1 = balmer_wave[line1]

    if line2 in balmer_wave:
        wave2 = balmer_wave[line2]

    if not (line1 in balmer_wave) or not (line2 in balmer_wave):
        raise(ValueError("combination of %s and %s are not supported." %
                         (line1, line2)))

    r_theo = balmer_flux[line1] / balmer_flux[line2]

    if extcurve in {'CCM', 'ccm'}:
        alam_av1, rv = extinction_ccm89(wave1, ret_rv=True)
        alam_av2 = extinction_ccm89(wave2)
    elif extcurve in {'calzetti', 'Calzetti', 'Calz', 'calz'}:
        alam_av1, rv = extinction_calz(wave1, ret_rv=True)
        alam_av2 = extinction_calz(wave2)
    else:
        raise(ValueError("%s curve is not yet implemented." % extcurve))

    ebv_factor = 2.5 / (alam_av2 - alam_av1) / rv
    ebv = ebv_factor * np.log10(r_obs / r_theo)

    if err is not None:
        errebv = np.power(ebv_factor * (1. / np.log(10.) * err / r_obs), 2)
        errebv = np.sqrt(errebv)
    else:
        errebv = None

    if clip is True:
        ebv = np.clip(ebv, 0., None)

    return(ebv, errebv)


def extinction_ccm89(w, rv=3.1, ret_rv=False):
    """Cardelli, Clayton, and Mathis (1989, ApJ, 345, 245) extinction curve.
    Now, it only supports MUSE wavelength range as rest-frame,
    but in principle, it's easy to implement the whole curve
    including an update for the near-UV by O'Donnell (1994, ApJ, 422, 158).
    """

    if isinstance(w, float) or isinstance(w, int):
        w = np.array([w])

    x = 1.e4 / w  # convert to um^{-1}
    y = x - 1.82

    a = np.zeros(w.size)
    b = np.zeros(w.size)

    idx_opt = np.where(np.logical_and(x > 1.1, x < 3.3))
    idx_nir = np.where(np.logical_and(x > 0.3, x <= 1.1))

    if idx_opt[0].size + idx_nir[0].size != w.size:
        raise(ValueError("Some wavelength out of the range."))

    if idx_opt[0].size > 0:
        # New coefficients from O'Donnell (1994)
        coeff_a = np.array([1., 0.104, -0.609, 0.701, 1.137,
                            -1.718, -0.827, 1.647, -0.505])
        coeff_b = np.array([0., 1.952, 2.908, -3.989, -7.985,
                            11.102, 5.491, -10.805, 3.347 ])
        a[idx_opt] = polyval(y[idx_opt], coeff_a)
        b[idx_opt] = polyval(y[idx_opt], coeff_b)

    if idx_nir[0].size > 0.:
        a[idx_nir] = 0.574 * np.power(x[idx_nir], 1.61)
        b[idx_nir] = -0.527 * np.power(x[idx_nir], 1.61)

    alam_av = a + b / rv

    if ret_rv is True:
        return(alam_av, rv)
    else:
        return(alam_av)


def extinction_calz(w, rv=4.05, ret_rv=False):
    """Calzetti (2000) extinction curve"""

    if isinstance(w, float) is True:
        w = np.array([w])

    x = w * 1.e-4  # angstrom to micron

    idx1 = np.logical_and(x > 0.63, x < 2.2)
    idx2 = np.logical_and(x > 0.12, x <= 0.63)

    klam = np.zeros(w.size)

    xinv = 1. / x

    coeff_1 = np.array([-1.857, 1.040])
    coeff_2 = np.array([-2.156, 1.509, -0.198, 0.011])

    klam[idx1] = 2.659 * polyval(xinv[idx1], coeff_1) + rv
    klam[idx2] = 2.659 * polyval(xinv[idx2], coeff_2) + rv

    alam_av = klam / rv

    if ret_rv is True:
        return(alam_av, rv)
    else:
        return(alam_av)


def extinction_f99(w):
    '''Fitzpatrick (1999, PASP, 111, 63) extinction curve.

    Rv=3.1 is assumed at this moment to correct the Galactic extinction
    with E(B-V) from Schlafly & Finkbeiner (2011, ApJ, 737, 103).

    Parameters
    ----------
    w : array_like
        Wavelength in angstrom.

    Returns
    -------
    alamebv : array_like
        A(lambda)/E(B-V)

    References
    ----------

    `Fitzpatrick 1999 <http://adsabs.harvard.edu/abs/1999PASP..111...63F>`_

    `Schalafly and Finkbeiner <http://adsabs.harvard.edu/abs/2011ApJ...737..103S>`_
    '''

    w_anchor_f99 = np.array([np.inf, 26500., 12200., 6000., 5470.,
                             4670., 4110., 2700., 2600.])  # angstrom
    wnumber_anchor_f99 = 1. / (w_anchor_f99 * 1.e-4)  # 1/micron
    alam_ebv_anchor_f99 = np.array([0., 0.265, 0.829, 2.688, 3.055,
                                    3.806, 4.315, 6.265, 6.591])
    tck_f99 = interpolate.splrep(wnumber_anchor_f99, alam_ebv_anchor_f99, k=3)

    alam_ebv_f99 = interpolate.splev(1. / (w * 1e-4), tck_f99, ext=2)

    return(alam_ebv_f99)


def extract_galactic_ebv(objname):
    """Obtain Galactic extinction E(B-V) from an input object name.

    Parameters
    ----------
    objname : str
        Name of a galaxy for which Galactic E(B-V) is searched.
        The object should be searchable in NED.

    Returns
    -------
    ebv : float
        E(B-V) for Schlafly & Finkbeiner (2011, ApJ, 737, 103)
        Galactic extinction map.
    """

    tb_ebv = IrsaDust.get_query_table(objname, section='ebv')
    ebv = tb_ebv['ext SandF mean'][0]

    return(ebv)


def correct_mwdust_cube(hdu, ebv=0., debug=False):
    """Correct Galactic extinction for a cube.

    Parameters
    ----------
    hdu : HDUList
        Input file (must be a standard MUSE cube)
    ebv : float, optional
        E(B-V) to be applied. The default is 0.
    debug : bool
        Toggle debug mode if True.

    Returns
    -------
    hdu_corr : HDUList
        HDUList object contains an extinction corrected cube.
    """

    hdu_corr = hdu

    w = get_wavelength(hdu_corr, axis=3)

    alam_ebv = extinction_f99(w)

    extfac = np.power(10., 0.4 * alam_ebv * ebv)
    extfac_cube = np.tile(extfac,
                          (hdu_corr[1].data.shape[2],
                           hdu_corr[1].data.shape[1],
                           1)).T

    if debug is True:
        print("Shape of extinciton vector: ", extfac.shape)
        print("Shape of input cube: ", hdu_corr[1].data.shape)
        print("Shape of extinction curve cube: ", extfac_cube.shape)

    hdu_corr[1].data = hdu_corr[1].data.copy() * extfac_cube
    hdu_corr[2].data = hdu_corr[2].data.copy() * extfac_cube**2

    hdu_corr[0].header['HIERARCH MAD MW EBV'] = ('E(B-V)=%5.3f' % ebv,
                                                 'Galactic extinction E(B-V)')
    hdu_corr[0].header['HIERARCH MAD MW EXTCURVE'] = ('Fitzpatrick (1999)',
                                                      'Extinction curve to correct MW extinction')

    return(hdu_corr)


def correct_mwdust_binspec(hdu, ebv=0., debug=False):
    """**Not implemented yet** Correct Galactic extinction for binned spectra.

    Parameters
    ----------
    hdu : HDU object
        Input HDU with a format of Voronoi-binned spectra
        from the pyezmad.voronoi package.
    ebv : float, optional
        E(B-V) to be applied. The default is 0.
    debug : bool
        Toggle debug mode if True.

    Returns
    -------
    hdu_corr : HDUList
        HDUList object contains extinction corrected binned-spectra.
    """

    hdu_corr = hdu

    w = get_wavelength(hdu_corr, axis=1)

    alam_ebv = extinction_f99(w)

    extfac = np.power(10., 0.4 * alam_ebv * ebv)
    extfac_tile = np.tile(extfac, (hdu[1].data.shape[0], 1))

    if debug is True:
        print("Shape of extinciton vector: ", extfac.shape)
        print("Shape of input cube: ", hdu_corr[1].data.shape)
        print("Shape of extinction curve cube: ", extfac_tile.shape)

    hdu_corr[1].data = hdu_corr[1].data.copy() * extfac_tile
    hdu_corr[2].data = hdu_corr[2].data.copy() * extfac_tile**2

    hdu_corr[0].header['HIERARCH MAD MW EBV'] = ('E(B-V)=%5.3f' % ebv,
                                                 'Galactic extinction E(B-V)')
    hdu_corr[0].header['HIERARCH MAD MW EXTCURVE'] = ('Fitzpatrick (1999)',
                                                      'Extinction curve to correct MW extinction')

    return(hdu_corr)


def correct_mwdust(hdu, is_cube=False, is_binspec=False,
                   debug=False, obj=None):
    """Unified interface for Correct Galactic extinction

    Parameters
    ----------
    hdu : HDUList
        Input HDU with a shape of either a standard MUSE cube
        or :py:meth:`~pyezmad.voronoi spectra`.
    is_cube : bool
        `True` if the input HDUList is a cube.
    is_binspec : bool
        `True` if the input HDUList is a binned spectra.
        One of ``is_cube`` or ``is_binspec`` must be `True`.
    debug : bool
        Toggle debug mode (print a few more lines).
    obj : str
        Object name in the case OBJECT key is not present in the header.
        When both this parameter and OBJECT key present,
        the OBJECT key is used.

    Returns
    -------
    hdu_corr : HDUList
        HDUList object contains extinction corrected spectra
        with the same format as the input.
    """

    if (is_cube is True) and (is_binspec is True):
        raise(ValueError("Both 'is_cube' and 'is_binspec' cannot be set True at the same time."))
    if (is_cube is False) and (is_binspec is False):
        raise(ValueError("Both 'is_cube' and 'is_binspec' cannot be set False at the same time."))

    # Get E(B-V) from 'OBJECT' key from the header
    if (obj is not None) and ('OBJECT' not in hdu[0].header):
        hdu[0].header['OBJECT'] = obj
    elif (obj is not None) and ('OBJECT' in hdu[0].header):
        print("Warning: the option obj='%s' is ignored as 'OBJECT' %s is already present in the header." %
              (obj, hdu[0].header['OBJECT']))

    try:
        objname = hdu[0].header['OBJECT']
        ebv = extract_galactic_ebv(objname)
        print('Object name %s is found' % objname)
        print('E(B-V)=%5.3f is obtained from NED' % ebv)
    except:
        raise(
            KeyError(
                'OBJECT key not found in the header of 0th extention! Exit.'))

    if is_cube is True:
        hdu_corr = correct_mwdust_cube(hdu, ebv=ebv, debug=debug)
    elif is_binspec is True:
        hdu_corr = correct_mwdust_binspec(hdu, ebv=ebv, debug=debug)

    return(hdu_corr)


if __name__ == '__main__':
    exit()
