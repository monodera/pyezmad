#!/usr/bin/env python

import numpy as np

import astropy.io.fits as fits
from astropy.visualization import scale_image

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def create(infile, prefix_out,
           is_save=True, is_plot=True,
           wi_scale='linear', wi_percent=99.,
           wi_cmap=cm.Greys_r):
    """Create MUSE white light image and save it to FITS file if specified.

    Simple white light image is produced for the input MUSE cube.  
    Some keywords for plotting is accepted.

    Args:
        infile: An input MUSE cube
        prefix_out: prefix for the output FITS file and PDF image if requested.
        wi_scale: Scaling function for plotting
                  (see astropy.visualization.scale_image(); default: linear)
        wi_percent: Percentile for clipping for plotting
                  (see astropy.visualization.scale_image(); default: 99)
        wi_cmap: Colormap for plotting (default: cm.Greys_r)
        is_plot: Flag if plot is required (default: True)
        is_save: Flag if white light image will be saved to a FITS file (default: True)

    Returns:
        A 2D numpy array of white image with a dimension of (NAXIS2, NAXIS1)
    """

    hdu = fits.open(infile)

    wi = np.sum(hdu[1].data, axis=0)

    if is_save==True:
        fits.writeto(prefix_out+'.fits', wi, hdu[1].header, clobber=True)

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

if __name__=='__main__':
	print('do nothing')
