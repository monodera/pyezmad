#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import sys
import warnings
import numpy as np
from scipy import interpolate
from astroquery.irsa_dust import IrsaDust
import astropy.io.fits as fits

# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import matplotlib

from .utilities import get_wavelength

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
                             4670., 4110., 2700., 2600.]) # angstrom
    wnumber_anchor_f99 = 1./(w_anchor_f99 * 1.e-4) # 1/micron
    alam_ebv_anchor_f99 = np.array([0., 0.265, 0.829, 2.688, 3.055,
                                    3.806, 4.315, 6.265, 6.591])
    tck_f99 = interpolate.splrep(wnumber_anchor_f99, alam_ebv_anchor_f99, k=3)

    alam_ebv_f99 = interpolate.splev(1./(w*1e-4), tck_f99, ext=2)

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
        E(B-V) for Schlafly & Finkbeiner (2011, ApJ, 737, 103) Galactic extinction map.
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

    extfac = np.power(10., 0.4*alam_ebv*ebv)
    extfac_cube = np.tile(extfac, (hdu_corr[1].data.shape[2], hdu_corr[1].data.shape[1], 1)).T

    if debug == True:
        print("Shape of extinciton vector: ", extfac.shape)
        print("Shape of input cube: ", hdu_corr[1].data.shape)
        print("Shape of extinction curve cube: ", extfac_cube.shape)

    hdu_corr[1].data = hdu_corr[1].data.copy() * extfac_cube
    hdu_corr[2].data = hdu_corr[2].data.copy() * extfac_cube**2

    hdu_corr[0].header['HIERARCH MAD MW EBV'] = ('E(B-V)=%5.3f' % ebv, 'Galactic extinction E(B-V)')
    hdu_corr[0].header['HIERARCH MAD MW EXTCURVE'] = ('Fitzpatrick (1999)', 'Extinction curve to correct MW extinction')

    return(hdu_corr)


def correct_mwdust_binspec(hdu, ebv=0., debug=False):
    """**Not implemented yet** Correct Galactic extinction for binned spectra. 

    Parameters
    ----------
    hdu : HDU object
        Input HDU with a format of Voronoi-binned spectra from the pyezmad.voronoi package.
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

    extfac = np.power(10., 0.4*alam_ebv*ebv)
    extfac_tile = np.tile(extfac, (hdu[1].data.shape[0], 1))

    if debug == True:
        print("Shape of extinciton vector: ", extfac.shape)
        print("Shape of input cube: ", hdu_corr[1].data.shape)
        print("Shape of extinction curve cube: ", extfac_tile.shape)

    hdu_corr[1].data = hdu_corr[1].data.copy() * extfac_tile
    hdu_corr[2].data = hdu_corr[2].data.copy() * extfac_tile**2

    hdu_corr[0].header['HIERARCH MAD MW EBV'] = ('E(B-V)=%5.3f' % ebv, 'Galactic extinction E(B-V)')
    hdu_corr[0].header['HIERARCH MAD MW EXTCURVE'] = ('Fitzpatrick (1999)', 'Extinction curve to correct MW extinction')

    return(hdu_corr)


def correct_mwdust(hdu, is_cube=False, is_binspec=False, debug=False, obj=None):
    """Unified interface for Correct Galactic extinction

    Parameters
    ----------
    hdu : HDUList
        Input HDU with a shape of either a standard MUSE cube or ~pyezmad.voronoi spectra.
    is_cube : bool
        `True` if the input HDUList is a cube.
    is_binspec : bool
        `True` if the input HDUList is a binned spectra. One of ``is_cube`` or ``is_binspec`` must be `True`.
    debug : bool
        Toggle debug mode (print a few more lines).
    obj : str
        Object name in the case OBJECT key is not present in the header.
        When both this parameter and OBJECT key present, the OBJECT key is used.

    Returns
    -------
    hdu_corr : HDUList
        HDUList object contains extinction corrected spectra with the same format as the input.
    """

    if is_cube == True and is_binspec == True:
        raise(ValueError("Both 'is_cube' and 'is_binspec' cannot be set True at the same time."))
    if is_cube == False and is_binspec == False:
        raise(ValueError("Both 'is_cube' and 'is_binspec' cannot be set False at the same time."))

    # Get E(B-V) from 'OBJECT' key from the header
    if (obj != None) and ('OBJECT' not in hdu[0].header):
        hdu[0].header['OBJECT'] = obj
    elif (obj != None) and ('OBJECT' in hdu[0].header):
        print("Warning: the option obj='%s' is ignored as 'OBJECT' %s is already present in the header." % (obj, hdu[0].header['OBJECT']))

    try:
        objname = hdu[0].header['OBJECT']
        ebv = extract_galactic_ebv(objname)
        print('Object name %s is found' % objname)
        print('E(B-V)=%5.3f is obtained from NED' % ebv)
    except:
        raise KeyError('OBJECT key not found in the header of 0th extention! Exit.')

    if is_cube == True:
        hdu_corr = correct_mwdust_cube(hdu, ebv=ebv, debug=debug)
    elif is_binspec == True:
        hdu_corr = correct_mwdust_binspec(hdu, ebv=ebv, debug=debug)

    return(hdu_corr)

if __name__ == '__main__':
    exit()
