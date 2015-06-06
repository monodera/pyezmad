#!/usr/bin/env python

# import os.path
import time
import numpy as np
from astropy.table import Table
import astropy.io.fits as fits
from voronoi_2d_binning import voronoi_2d_binning
from mpdaf.obj import Cube

def make_snlist_to_voronoi(infile, wave_center=5750., dwave=25.):
    """Make arrays needed for Voronoi binning routine by Cappellari & Coppin 

    Args:
        infile: input MUSE cube file (FITS format)
        wave_center: central wavelenth in angstrom to compute signal and noise
        dwave: signal and noise are computed within +/- dwave (unit: angstrom) from wave_center

    Returns: (x, y, signal, noise, snmap)
        x     : x pixel coordinates of the input cube (size NAXIS1*NAXIS2)
        y     : y pixel coordinates of the input cube (size NAXIS1*NAXIS2)
        signal: signal per angstrom at each pixel (size NAXIS1*NAXIS2) 
        noise : 1sigma noise per angstrom at each pixel (size NAXIS1*NAXIS2) 
        snmap : pixel-by-pixel S/N map
    """

    # read MUSE cube with MPDAF
    cube = Cube(infile)

    # cut subcube [wc-dw:wc+dw]
    subcube = cube.get_lambda(lbda_min=wave_center-dwave, lbda_max=wave_center+dwave)

    # compute **average** signal and noise within the specified wavelength range
    signal = np.trapz(subcube.data, x=subcube.wave.coord(), axis=0)
    signal /= (subcube.wave.get_end()-subcube.wave.get_start())

    var = np.trapz(subcube.var, x=subcube.wave.coord(), axis=0)
    var /= (subcube.wave.get_end()-subcube.wave.get_start())

    # create arrays of x and y coordinates
    x = np.arange(subcube.shape[2])
    y = np.arange(subcube.shape[1])

    # make the x, y to 2D coordinates
    xx, yy = np.meshgrid(x, y)

    snmap = signal/np.sqrt(var)

    return(np.ravel(xx), np.ravel(yy), np.ravel(signal), np.ravel(np.sqrt(var)), snmap)



def run_voronoi_binning(infile, outprefix, wave_center=5750, dwave=25., target_sn=50.):
    """All-in-one function to run Voronoi 2D binning

    Args:
        infile: input MUSE cube
        outprefix: prefix for the outputs
        wave_center: central wavelength to compute signal and noise for Voronoi binning
        dwave: signal and noise are computed +/- dwave from wave_center
        target_sn: target S/N for Voronoi binning

    Returns:
        file_xy2bin: a FITS file storing infomation of (x, y, bin ID)
        file_bininfo: a FITS file stroing information in each bin, including
                      (bin, xnode, ynode, xcen, ycen, sn, npix, scale)
    """

    #
    # Compute signal and noise at each pixel
    #
    t_begin = time.time()
    x, y, signal, noise, snmap = make_snlist_to_voronoi(infile, wave_center, dwave)
    t_end = time.time()
    print("Time Elapsed for Signal and Noise Computation: %.2f [seconds]" % (t_end-t_begin))

    fits.writeto(outprefix+'_prebin_snmap.fits', snmap.data, fits.getheader(infile), clobber=True)

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


def create_segmentation_image(tb_xy_fits, refimg_fits):
    """Create a segmentation image based on the output from `run_voronoi_binning`

    Args:
        tb_xy_fits: FITS output table from run_voronoi_binning
        refimg_fits: FITS image for the reference
                     (white light image is expected at the moment)

    Returns:
        binimg: a 2D numpy array with the same shape as `refimg_fits`
                storing bin ID at each pixel.
    """

    tb_xy = Table.read(tb_xy_fits)
    hdu = fits.open(refimg_fits)

    binimg = np.empty_like(hdu[0].data) + np.nan

    bin_min = np.min(tb_xy['bin'])
    bin_max = np.max(tb_xy['bin'])

    xbin, ybin = tb_xy['x'], tb_xy['y']

    # FIXME: this is not efficient, but should finish in an acceptable computational time.
    for i in xrange(bin_min, bin_max+1):
        idx = np.where(tb_xy['bin']==i)
        xbin = tb_xy['x'][idx]
        ybin = tb_xy['y'][idx]
        for j in xrange(xbin.size):
            binimg[ybin[j],xbin[j]] = i

    hdu.close()

    return(binimg)


def create_value_image(segmentation_image, value):
    """Create an 2D Voronoi segmented image filled with a given value at each Voronoi bin

    Args:
        segmentation_image: 2D numpy array of Voronoi segmentation map
                            (output from `create_segmentation_image`)
        value: a numpy array of value to be reconstructed as an output image.
               The length must be identical to the number of bins.

    Returns:
        valimg: a 2D numpy array with the same shape as `refimg_fits`
                storing value at each Voronoi bin.
    """

    valimg = np.empty(segmentation_image.shape) + np.nan

    for i in xrange(value.size):
        idx=np.where(segmentation_image==i)
        valimg[idx] = value[i]

    return(valimg)


def read_stacked_spectra(infile):
    """Read Voronoi binned spectra into array.

    Args:
        infile: input FITS file to be stored. Should have a shape of (nbin, nwave).

    Returns: (wave, data, noise)
        wave: wavelenth array with a length of NAXIS1
        data: stacked spectra with a shape of (NAXIS2, NAXIS1) or (nbin, nwave)
        noise: noise spectra with a shape of (NAXIS2, NAXIS1) or (nbin, nwave)
    """

    hdu = fits.open(infile)
    h_obj = hdu['FLUX'].header
    # h_var = hdu['VAR'].header
    wave = h_obj['CRVAL1'] + h_obj['CDELT1']*(np.arange(h_obj['NAXIS1'])-h_obj['CRPIX1']+1)
    return(wave, hdu['FLUX'].data, np.sqrt(hdu['VAR'].data))

def stacking(fcube, ftable, fout):
    """Stack spectra based on Voronoi binning

    Args:
        fcube: FITS file of MUSE cube
        ftable: 'xy2bin' FITS file created by the binning process
        fout: Output FITS file

    Returns:
        None
    """

    tb_xy2bin = Table.read(ftable)
    cube = Cube(fcube)

    bin_start = tb_xy2bin['bin'].min()
    bin_end = tb_xy2bin['bin'].max()

    nbin = bin_end-bin_start+1

    nwave = cube.wave.coord().size
    stack_spec = np.empty((nbin,nwave), dtype=np.float) + np.nan
    stack_var  = np.empty((nbin,nwave), dtype=np.float) + np.nan

    for i in xrange(nbin):
    # for i in xrange(20):

        idx = tb_xy2bin['bin']==i
        xbin = tb_xy2bin['x'][idx]
        ybin = tb_xy2bin['y'][idx]

        stack_data_bin = np.array([cube.data[:,iy,ix] for iy,ix in zip(ybin,xbin)])
        stack_data_bin = np.nansum(stack_data_bin, axis=0)

        stack_var_bin = np.array([cube.var[:,iy,ix] for iy,ix in zip(ybin,xbin)])
        stack_var_bin = np.nansum(stack_var_bin, axis=0)

        # subcube_data_test = np.zeros(nwave)
        # subcube_var_test = np.zeros(nwave)
        # for j in xrange(tb_xy2bin['bin'][idx].size):
        #     subcube_data_test += cube.data[:,tb_xy2bin['y'][idx][j],tb_xy2bin['x'][idx][j]]
        #     subcube_var_test += cube.var[:,tb_xy2bin['y'][idx][j],tb_xy2bin['x'][idx][j]]
        # ratio_data = subcube_data_test/stack_data_bin
        # ratio_var = subcube_var_test/stack_var_bin
        # print(i, tb_xy2bin['bin'][idx].size, np.median(ratio_data), np.median(ratio_var))

        stack_data_bin /= tb_xy2bin['bin'][idx].size
        stack_var_bin /= tb_xy2bin['bin'][idx].size**2

        stack_spec[i,:] = stack_data_bin
        stack_var[i,:] = stack_var_bin

    prihdu = fits.PrimaryHDU()
    data_hdu = fits.ImageHDU(data=stack_spec, name='FLUX')
    var_hdu = fits.ImageHDU(data=stack_var, name='VAR')

    for hdu in [data_hdu, var_hdu]:
        hdu.header['CRPIX1'] = cube.data_header['CRPIX3']
        hdu.header['CRVAL1'] = cube.data_header['CRVAL3']
        hdu.header['CDELT1'] = cube.data_header['CDELT3']
        hdu.header['CTYPE1'] = cube.data_header['CTYPE3']
        hdu.header['CUNIT1'] = cube.data_header['CUNIT3']
        hdu.header['BUNIT'] = cube.data_header['BUNIT']
        hdu.header['FSCALE'] = cube.data_header['FSCALE']

    hdulist = fits.HDUList([prihdu, data_hdu, var_hdu])
    hdulist.writeto(fout, clobber=True)

    return()
