#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

from .utilities import get_wavelength

def mw_f99(w, rv=3.1, gamma=0.99, x0=4.596, c1=None, c2=None, c3=3.23, c4=0.41):

    if c2==None:
        c2 = -0.824 + 4.717/rv
    if c1==None:
        c1 = 2.030 - 3.007*c2

    x = 1.e4 / w # convert to inverse micron
    x2 = np.power(x,2)

    d = x2 / ((x2-x0**2)**2 + x2*gamma**2)

    f = 0.5392 * (x-5.9)**2 + 0.05644 * (x-5.9)**3
    f[x<5.9] = 0.

    kxv = c1 + c2*x + c3*d + c4*f

    return(kxv)


def correct_mwdust_cube(hdu, ebv):

    hdu_corr = hdu.copy()

    w = get_wavelength(hdu, axis=3)

    kxv = mw_f99(w)
    extfac = np.power(10., 0.4*kxv*ebv)

    extfac_cube = np.tile(extfac, (1, hdu[1].data.shape[1], hdu[1].data.shape[2]))

    hdu_corr[1].data = hdu[1].data * extfac_cube
    hdu_corr[2].data = hdu[2].data * extfac_cube**2

    hdu_corr[0].header['comment'] = 'Galactic extinction was corrected with E(B-V)=%5.3f' % ebv
    hdu_corr[0].header['E(B-V)MW'] = 'E(B-V)=%5.3f' % ebv
    hdu_corr[0].header['EXTCURVE'] = 'Fitzpatrick (1999)'

    return(hdu_corr)


def correct_mwdust_binspec(hdu, ebv):
    hdu_corr = hdu.copy()
    w = get_wavelength(hdu, axis=1)
    kxv = mw_f99(w)
    extfac = np.power(10., 0.4*kxv*ebv)

    # FIXME: absolutely wrong
    extfac_cube = np.tile(extfac, (1, hdu[1].data.shape[1], hdu[1].data.shape[2]))

    hdu_corr[1].data = hdu[1].data * extfac_cube
    hdu_corr[2].data = hdu[2].data * extfac_cube**2

    hdu_corr[0].header['comment'] = 'Galactic extinction was corrected with E(B-V)=%5.3f' % ebv
    hdu_corr[0].header['E(B-V)MW'] = 'E(B-V)=%5.3f' % ebv
    hdu_corr[0].header['EXTCURVE'] = 'Fitzpatrick (1999)'

    return(hdu_corr)




if __name__ == '__main__':

    w = np.logspace(3, 4, 100) # wavelength array in angstrom

    kxv = mw_f99(w)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(1e4/w, kxv, '-')

    plt.show()
