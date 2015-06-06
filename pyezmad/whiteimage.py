#!/usr/bin/env python

import os, os.path
import numpy as np

import astropy.io.fits as fits
from astropy.visualization import scale_image

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def make_whiteimage(infile, prefix_out,
                    wi_scale='linear', wi_percent=99.,
                    wi_cmap=cm.Greys_r, is_plot=True):
    """Make MUSE white image and safe it to FITS file.

    Simple white light image is produced for the input MUSE cube.  
    Some keywords for plotting is accepted.

    Args:
        infile: An input MUSE cube
        prefix_out: prefix for the output FITS file and PDF image if requested.
        wi_scale: (Optional) Scaling function for plotting
                  (see astropy.visualization.scale_image(); default: linear)
        wi_percent: (Optional) Percentile for clipping for plotting
                  (see astropy.visualization.scale_image(); default: 99)
        wi_cmap: (Optional) Colormap for plotting (default: cm.Greys_r)
        is_plot: (Optional) flag if plot is required (default: True)

    Returns:
        A 2D numpy array of white image with a dimension of (NAXIS2, NAXIS1)
    """

    hdu = fits.open(infile)

    # wi = np.nansum(hdu[1].data, axis=0)
    wi = np.sum(hdu[1].data, axis=0)

    fits.writeto(prefix_out+'.fits', wi, hdu[1].header, clobber=True)

    if is_plot:
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

if __name__=='__main__':
	print('do nothing')
