#!/usr/bin/env python

import os, os.path
import numpy as np

import astropy.io.fits as fits
from astropy.visualization import scale_image

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def make_whiteimage(infile, prefix_out, wi_scale='linear', wi_percent=99., wi_cmap=cm.Greys_r, is_plot=True):
    hdu = fits.open(infile)

    wi = np.nansum(hdu[1].data, axis=0)

    fits.writeto(prefix_out+'.fits', wi, hdu[1].header, clobber=True)

    if is_plot:
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

if __name__=='__main__':
	print('do nothing')