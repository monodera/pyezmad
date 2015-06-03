#!/usr/bin/env python

import os, os.path
import time
import numpy as np
from astropy.table import Table
import astropy.io.fits as fits
from voronoi_2d_binning import voronoi_2d_binning
from mpdaf.obj import Cube

def make_snlist_to_voronoi(infile, wave_center=5750., dwave=25.):
    """Make arrays needed for Voronoi binning routine by Cappellari & Coppin 

    Args:
        infile: input FITS MUSE cube
        wave_center: central wavelenth to compute signal and noise
        dwave: signal and noise are computed within +/- dwave from wave_center

    Returns:
        (x, y, signal, noise, snmap): 
        x and y are, respectively, x and y pixel coordinates. 
        signal and noise are signal and noise per angstrom at each pixel.  
        All of the above arrays are with a size of NAXIS1*NAXIS2.
        snmap is a pixel-by-pixel S/N map.
    """

    cube = Cube(infile)

    subcube = cube.get_lambda(lbda_min=wave_center-dwave, lbda_max=wave_center+dwave)

    # mask = subcube.data.mask

    # compute **average** signal and noise within the specified wavelength range
    signal = np.trapz(subcube.data, x=subcube.wave.coord(), axis=0)
    signal /= (subcube.wave.get_end()-subcube.wave.get_start())

    var = np.trapz(subcube.var, x=subcube.wave.coord(), axis=0)
    var /= (subcube.wave.get_end()-subcube.wave.get_start())

    x = np.arange(subcube.shape[2])
    y = np.arange(subcube.shape[1])

    xx, yy = np.meshgrid(x, y)

    snmap = signal/np.sqrt(var)

    return(np.ravel(xx), np.ravel(yy), np.ravel(signal), np.ravel(np.sqrt(var)), snmap)



def run_voronoi_binning(infile, outprefix, wave_center=5750, dwave=25., target_sn=50.):
    """
    infile: input MUSE cube
    outprefix: prefix for the outputs
    wave_center: central wavelength to compute signal and noise for Voronoi binning
    dwave: signal and noise are computed +/- dwave from wave_center
    target_sn: target S/N for Voronoi binning
    """

    #
    # Compute signal and noise at each pixel
    #
    t_begin = time.time()
    x, y, signal, noise, snmap = make_snlist_to_voronoi(infile, wave_center, dwave)
    t_end = time.time()
    print("Time Elapsed for Signal and Noise Computation: %.2f [seconds]" % (t_end-t_begin))

    fits.writeto(outprefix+'_prebin_snmap.fits', snmap.data, fits.getheader(infile), clobber=True)

    # exit()

    # select indices of valid pixels
    idx_valid = np.logical_and(np.isfinite(signal), np.isfinite(noise))
    # idx_valid = np.logical_and(idx_valid, noise>0.)

    #
    # Voronoi binning
    #
    t_begin = time.time()
    binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(
        x[idx_valid], y[idx_valid], signal[idx_valid], noise[idx_valid],
        target_sn, plot=False, quiet=False)
    t_end = time.time()
    print("Time Elapsed for Voronoi Binning: %.2f [seconds]" % (t_end-t_begin))


    #
    # write output into 2 files
    #
    tb_xyinfo = Table([x[idx_valid], y[idx_valid], binNum],
                      names=['x', 'y', 'bin'])
    tb_bininfo = Table([np.arange(xNode.size, dtype=np.int),
                        xNode, yNode, xBar, yBar,
                        sn, np.array(nPixels, dtype=np.int), scale],
                       names=['bin', 'xnode', 'ynode', 'xcen', 'ycen', 'sn', 'npix', 'scale'])

    tb_xyinfo.write(outprefix+'_xy2bin_sn%i.fits' % int(target_sn), overwrite=True)
    tb_xyinfo.write(outprefix+'_xy2bin_sn%i.dat' % int(target_sn), format='ascii.fixed_width')
    tb_bininfo.write(outprefix+'_bininfo_sn%i.fits' % int(target_sn), overwrite=True)
    tb_bininfo.write(outprefix+'_bininfo_sn%i.dat' % int(target_sn), format='ascii.fixed_width')

    print("Writing... %s_xy2bin_sn%i.fits" % (outprefix, int(target_sn)))
    print("Writing... %s_xy2bin_sn%i.dat" % (outprefix, int(target_sn)))
    print("Writing... %s_bininfo_sn%i.fits" % (outprefix, int(target_sn)))
    print("Writing... %s_bininfo_sn%i.dat" % (outprefix, int(target_sn)))

    return(outprefix+'_xy2bin_sn%i.fits' % int(target_sn),
           outprefix+'_bininfo_sn%i.fits' % int(target_sn))


def make_voronoi_segmentation_image(tb_xy_fits, refimg_fits):
    """
    tb_xy_fits: FITS output table from run_voronoi_binning
    refimg_fits: FITS image for the reference (white light image is expected at the moment)
    """

    tb_xy = Table.read(tb_xy_fits)
    hdu = fits.open(refimg_fits)

    binimg = np.empty_like(hdu[0].data) + np.nan

    bin_min = np.min(tb_xy['bin'])
    bin_max = np.max(tb_xy['bin'])

    xbin, ybin = tb_xy['x'], tb_xy['y']

    # this is not efficient, but should finish in an acceptable computational time
    for i in xrange(bin_min, bin_max+1):
        idx = np.where(tb_xy['bin']==i)
        xbin = tb_xy['x'][idx]
        ybin = tb_xy['y'][idx]
        for j in xrange(xbin.size):
            binimg[ybin[j],xbin[j]] = i

    hdu.close()
    return(binimg)


def make_voronoi_value_image(segmentation_image, value):
    """
    segmentation_image: 2D numpy array of Voronoi segmentation map (output from make_voronoi_segmentation_image)
    value: a numpy array of value to be reconstructed as an output image (the length must be identical to the number of bins)
    """

    valimg = np.empty_like(segmentation_image) + np.nan

    for i in xrange(value.size):
        idx=np.where(segmentation_image==i)
        valimg[idx] = value[i]

    return(valimg)
