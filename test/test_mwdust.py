#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, os.path
import numpy as np
import astropy.io.fits as fits
import pyezmad.extinction as extinction

def main_cube():
    indir = '/net/astrogate/export/astro/shared/MAD/MUSE/P95/reduceddata/'
    infile = os.path.join(indir, 'NGC4980/NGC4980_FINAL.fits')

    hdu = fits.open(infile)

    ebv = 0.3

    hdu_corr = extinction.correct_mwdust(hdu, ebv, debug=True, is_cube=True)
    hdu_corr.writeto('ngc4980_final_mwcorr.fits', clobber=True)

    return

def main_binspec():
    return

if __name__ == '__main__':
    main_cube()
    main_binspec()

