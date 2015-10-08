#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from scipy import interpolate

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

from .utilities import get_wavelength

def extinction_f99(w):
    '''
    Fitzpatrick (1999, PASP, 111, 63) extinction curve
    Rv=3.1 is assumed at this moment to correct the Galactic extinction
    with E(B-V) from Schlafly & Finkbeiner (2011, ApJ, 737, 103)

    Input:
        w: wavelength in angstrom
    Output:
        A(lambda)/E(B-V)
    '''

    w_anchor_f99 = np.array([np.inf, 26500., 12200., 6000., 5470.,
                             4670., 4110., 2700., 2600.]) # angstrom
    wnumber_anchor_f99 = 1./(w_anchor_f99 * 1.e-4) # 1/micron
    alam_ebv_anchor_f99 = np.array([0., 0.265, 0.829, 2.688, 3.055,
                                    3.806, 4.315, 6.265, 6.591])
    tck_f99 = interpolate.splrep(wnumber_anchor_f99, alam_ebv_anchor_f99, k=3)

    alam_ebv_f99 = interpolate.splev(1./(w*1e-4), tck_f99, ext=2)

    return(alam_ebv_f99)

def extract_galactic_ebv(gal_name):

    tb_ebv = IrsaDust.get_query_table(gal_name, section='ebv')
    ebv = tb_ebv['ext SandF mean'][0]

    return ebv

def correct_mwdust_cube(infile, debug=False):

    hdu_corr = fits.open(infile)
    gal_name=hdu_corr[0].header['OBJECT']
    ebv = extract_galactic_ebv(gal_name)
    w = get_wavelength(hdu, axis=3)
    alam_ebv = extinction_f99(w)
    extfac = np.power(10., 0.4*alam_ebv*ebv)
    extfac_cube = np.tile(extfac, (hdu[1].data.shape[2], hdu[1].data.shape[1], 1)).T

    if debug == True:
        print("Shape of extinciton vector: ", extfac.shape)
        print("Shape of input cube: ", hdu[1].data.shape)
        print("Shape of extinction curve cube: ", extfac_cube.shape)

    hdu_corr[1].data = hdu[1].data * extfac_cube
    hdu_corr[2].data = hdu[2].data * extfac_cube**2

    hdu_corr[0].header['comment'] = 'Galactic extinction was corrected with E(B-V)=%5.3f' % ebv
    hdu_corr[0].header['EBV_MW'] = 'E(B-V)=%5.3f' % ebv
    hdu_corr[0].header['EXTCURVE'] = 'Fitzpatrick (1999)'

    return(hdu_corr)


def correct_mwdust_binspec(hdu, ebv, debug=False):
    hdu_corr = hdu
    # w = get_wavelength(hdu, axis=1)
    # kxv = mw_f99(w)
    # extfac = np.power(10., 0.4*kxv*ebv)
    # # FIXME: absolutely wrong
    # extfac_cube = np.tile(extfac, (1, hdu[1].data.shape[1], hdu[1].data.shape[2]))
    # hdu_corr[1].data = hdu[1].data * extfac_cube
    # hdu_corr[2].data = hdu[2].data * extfac_cube**2
    # hdu_corr[0].header['comment'] = 'Galactic extinction was corrected with E(B-V)=%5.3f' % ebv
    # hdu_corr[0].header['E(B-V)MW'] = 'E(B-V)=%5.3f' % ebv
    # hdu_corr[0].header['EXTCURVE'] = 'Fitzpatrick (1999)'
    return(hdu_corr)

def correct_mwdust(hdu, ebv, is_cube=False, is_binspec=False, debug=False):
    if is_cube == True and is_binspec == True:
        raise(ValueError("Both 'is_cube' and 'is_binspec' cannot be set True at the same time."))
    if is_cube == False and is_binspec == False:
        raise(ValueError("Both 'is_cube' and 'is_binspec' cannot be set False at the same time."))

    if is_cube == True:
        hdu_corr = correct_mwdust_cube(hdu, ebv, debug=debug)
    elif is_binspec == True:
        hdu_corr = correct_mwdust_binspec(hdu, ebv, debug=debug)

    return(hdu_corr)

if __name__ == '__main__':
    exit()
