#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, os.path
import numpy as np
import astropy.io.fits as fits
import pyezmad.extinction as extinction

import matplotlib.pyplot as plt

def main_cube():
    indir = '/net/astrogate/export/astro/shared/MAD/MUSE/P95/reduceddata/'
    infile = os.path.join(indir, 'NGC4980/NGC4980_FINAL.fits')

    hdu = fits.open(infile)

    hdu_corr = extinction.correct_mwdust(hdu, debug=True, is_cube=True, obj='NGC4980')
    hdu_corr.writeto('ngc4980_final_mwcorr.fits', clobber=True)

    plt.ion()
    plt.plot(hdu_corr[1].data[:,150,150] / hdu[1].data[:,150,150])

    return

def main_binspec():
    infile = '../../../mad_work/ngc4980/stacking/ngc4980_voronoi_stack_spec_sn50.fits'

    hdu = fits.open(infile)

    hdu_corr = extinction.correct_mwdust(hdu, debug=True, is_binspec=True, obj='NGC4980')
    hdu_corr.writeto('ngc4980_voronoi_stack_spec_sn50_mwcorr.fits')
    return

if __name__ == '__main__':
    main_cube()
    # main_binspec()

