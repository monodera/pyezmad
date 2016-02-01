#!/usr/bin/env python

import os.path
import time
import numpy as np
from astropy.table import Table
import astropy.io.fits as fits
import matplotlib.pyplot as plt

from voronoi_2d_binning import voronoi_2d_binning
from mpdaf.obj import Cube

# from .emission_line_fitting import gaussian
from .emission_line_fitting import gaussian, search_lines
from .utilities import (get_wavelength, read_emission_linelist,
                        muse_fwhm, sigma2fwhm)


class Voronoi:
    def __init__(self, voronoi_xy=None, voronoi_bininfo=None):
        if (voronoi_xy is None) or (voronoi_bininfo is None):
            raise(
                ValueError(
                    "Both voronoi_ xy and voronoi_bininfo \
                    must be set togather."))
        else:
            self.load_data(voronoi_xy, voronoi_bininfo)

    def load_data(self, voronoi_xy, voronoi_bininfo):
        self.__xy = Table.read(voronoi_xy)
        self.__bininfo = Table.read(voronoi_bininfo)

    @property
    def xy(self):
        return(self.__xy)

    @property
    def bininfo(self):
        return(self.__bininfo)


def make_snlist_to_voronoi(infile, wave_center=5750., dwave=25.):
    """Make arrays needed for Voronoi binning routine by Cappellari & Coppin.

    Parameters
    ----------
    infile : str
        Input MUSE cube file (a standard MUSE cube format).
    wave_center : float
        Central wavelenth in angstrom to compute signal and noise.
    dwave : float
        Signal and noise are computed within ``wave_center+/-dwave``.

    Returns
    -------
    x, y : :py:class:`numpy.ndarray`
        Pixel coordinates of the input cube (shape of NAXIS1*NAXIS2).
    signal : :py:class:`numpy.ndarray`
        Signal per angstrom at each pixel (shape of NAXIS1*NAXIS2).
    noise : :py:class:`numpy.ndarray`
        1 sigma noise per angstrom at each pixel (shape of NAXIS1*NAXIS2).
    snmap : :py:class:`numpy.ndarray`
        Pixel-by-pixel S/N map. The shape of the map is ``(NAIX2, NAXIS1)``.
    """

    # read MUSE cube with MPDAF
    cube = Cube(infile)

    # cut subcube [wc-dw:wc+dw]
    subcube = cube.get_lambda(lbda_min=wave_center - dwave,
                              lbda_max=wave_center + dwave)

    # compute *average* signal and noise within the specified wavelength range
    signal = np.trapz(subcube.data, x=subcube.wave.coord(), axis=0)
    signal /= (subcube.wave.get_end() - subcube.wave.get_start())

    var = np.trapz(subcube.var, x=subcube.wave.coord(), axis=0)
    var /= (subcube.wave.get_end() - subcube.wave.get_start())

    # create arrays of x and y coordinates
    x = np.arange(subcube.shape[2])
    y = np.arange(subcube.shape[1])

    # make the x, y to 2D coordinates
    xx, yy = np.meshgrid(x, y)

    snmap = signal / np.sqrt(var)

    return(np.ravel(xx), np.ravel(yy), np.ravel(signal),
           np.ravel(np.sqrt(var)), snmap)


def run_voronoi_binning(infile, outprefix,
                        wave_center=5750, dwave=25.,
                        target_sn=50.,
                        maskfile=None,
                        invert_mask=None,
                        min_sn=None,
                        quiet=False):
    """All-in-one function to run Voronoi 2D binning.

    Parameters
    ----------
    infile : str
        Input MUSE cube.
    outprefix : str
        Prefix for the outputs.
    wave_center : float, optional
        Central wavelength to compute signal and noise for Voronoi binning.
        The default is 5750 angstrom.
    dwave : float, optional
        Signal and noise are computed ``+/-dwave`` from ``wave_center``.
        The default is 25 angstrom.
    target_sn : float, optional
        Target S/N per pixel for Voronoi binning. The default is 50.
    quiet : bool, optional
        Toggle ``quiet`` option in :py:func:`voronoi.voronoi_2d_binning.
    maskfile : str or list, optional
        File defining the object mask. It has to be a single FITS file or
        a list of FITS files. In the mask image,
        pixels with 0 will be regarded as valid.
        (1: mask, 0: valid)
    invert_mask: bool or list
        Flag to determine whether the mask should be inverted or not.
    min_sn : float, optional
        Minimum S/N per pixel to be binned.

    Returns
    -------
    file_xy2bin : str
        A FITS file storing infomation of ``(x, y, bin ID)``.
    file_bininfo : str
        A FITS file stroing information in each bin, including
        ``(bin, xnode, ynode, xcen, ycen, sn, npix, scale)``.
    """

    #
    # Compute signal and noise at each pixel
    #
    t_begin = time.time()
    x, y, signal, noise, snmap = make_snlist_to_voronoi(infile, wave_center,
                                                        dwave)
    t_end = time.time()
    print("Time Elapsed for Signal and Noise Computation: %.2f [seconds]" %
          (t_end - t_begin))

    fits.writeto(outprefix + '_prebin_snmap.fits', snmap.data,
                 fits.getheader(infile), clobber=True)

    # select indices of valid pixels
    idx_valid = np.logical_and(np.isfinite(signal), np.isfinite(noise))

    # mask
    # it's a bit complicated as we defined to use 1 for masked pixels...
    if maskfile is not None:
        if isinstance(maskfile, str):
            mask = np.array(fits.getdata(maskfile), dtype=np.bool)
            if invert_mask is True:
                mask = np.logical_not(mask)
            idx_valid = np.logical_and(idx_valid,
                                       np.ravel(np.logical_not(mask)))
        elif isinstance(maskfile, list):
            if invert_mask is None:
                invert_mask = np.zeros(len(maskfile), dtype=np.bool)
            elif not isinstance(invert_mask, list):
                raise(TypeError(
                    "invert_mask must be a list when mask is a list."))
            for imask, mask in enumerate(maskfile):
                mask = np.array(fits.getdata(mask), dtype=np.bool)
                if invert_mask[imask] is True:
                    mask = np.logical_not(mask)
                idx_valid = np.logical_and(idx_valid,
                                           np.ravel(np.logical_not(mask)))

    # set min S/N
    if min_sn is not None:
        idx_valid = np.logical_and(np.ravel(snmap) >= min_sn, idx_valid)

    #
    # Voronoi binning
    #
    t_begin = time.time()
    binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(
        x[idx_valid], y[idx_valid], signal[idx_valid], noise[idx_valid],
        target_sn, plot=False, quiet=quiet)
    t_end = time.time()
    print("Time Elapsed for Voronoi Binning: %.2f [seconds]" %
          (t_end - t_begin))


    #
    # write output into 2 files
    #
    tb_xyinfo = Table([x[idx_valid], y[idx_valid], binNum],
                      names=['x', 'y', 'bin'])
    tb_bininfo = Table([np.arange(xNode.size, dtype=np.int),
                        xNode, yNode, xBar, yBar,
                        sn, np.array(nPixels, dtype=np.int), scale],
                       names=['bin', 'xnode', 'ynode', 'xcen', 'ycen',
                              'sn', 'npix', 'scale'])

    tb_xyinfo.write(outprefix + '_xy2bin_sn%i.fits' % int(target_sn),
                    overwrite=True)
    tb_xyinfo.write(outprefix + '_xy2bin_sn%i.dat' % int(target_sn),
                    format='ascii.fixed_width')
    tb_bininfo.write(outprefix + '_bininfo_sn%i.fits' % int(target_sn),
                     overwrite=True)
    tb_bininfo.write(outprefix + '_bininfo_sn%i.dat' % int(target_sn),
                     format='ascii.fixed_width')

    print("Writing... %s_xy2bin_sn%i.fits" % (outprefix, int(target_sn)))
    print("Writing... %s_xy2bin_sn%i.dat" % (outprefix, int(target_sn)))
    print("Writing... %s_bininfo_sn%i.fits" % (outprefix, int(target_sn)))
    print("Writing... %s_bininfo_sn%i.dat" % (outprefix, int(target_sn)))

    return(outprefix + '_xy2bin_sn%i.fits' % int(target_sn),
           outprefix + '_bininfo_sn%i.fits' % int(target_sn))


def create_segmentation_image(tb_xy_fits, refimg_fits):
    """Create a segmentation image based on the output
    from :py:func:`pyezmad.voronoi.run_voronoi_binning`.

    Parameters
    ----------
    tb_xy_fits : str
        FITS output table from :py:func:`pyezmad.voronoi.run_voronoi_binning`.
    refimg_fits : str
        Reference FITS image (white-light image is just fine).

    Returns
    -------
    binimg : :py:class:`~numpy.ndarray`
        A 2D numpy array with the same shape as the reference image
        storing Voronoi bin ID at each pixel.
    """

    tb_xy = Table.read(tb_xy_fits)
    hdu = fits.open(refimg_fits)

    binimg = np.empty_like(hdu[0].data) + np.nan

    bin_min = np.min(tb_xy['bin'])
    bin_max = np.max(tb_xy['bin'])

    xbin, ybin = tb_xy['x'], tb_xy['y']

    # FIXME: this is not efficient,
    # but should finish in an acceptable computational time.
    for i in xrange(bin_min, bin_max + 1):
        idx = np.where(tb_xy['bin'] == i)
        xbin = tb_xy['x'][idx]
        ybin = tb_xy['y'][idx]
        for j in xrange(xbin.size):
            binimg[ybin[j], xbin[j]] = i

    hdu.close()

    return(binimg)


def create_value_image(segmentation_image, value):
    """Create an 2D Voronoi segmented image filled with
    given values at each Voronoi bin.

    Parameters
    ----------
    segmentation_image : :py:class:`~numpy.ndarray`
        A 2D numpy array of Voronoi segmentation map
        (i.e., output from :py:meth:`create_segmentation_image`).
    value : array_like
        A numpy array of values to be reconstructed as an output image.
        The length of the array must be identical to the number of bins.

    Returns
    -------
    valimg : :py:class:`~numpy.ndarray`
        A 2D numpy array with the same shape as the segmentation image
        storing ``value`` at each Voronoi bin.
    """

    valimg = np.empty(segmentation_image.shape) + np.nan

    for i in xrange(value.size):
        idx = np.where(segmentation_image == i)
        valimg[idx] = value[i]

    return(valimg)


def read_stacked_spectra(infile, ext_data=1, ext_var=2):
    """Read Voronoi binned spectra into array.

    Parameters
    ----------
    infile : str
        Input FITS file to be processed.
        It must have a shape of ``(nbin, nwave)``.
    ext_data : int or str, optional
        Data extension
    ext_var : int or str, optional
        Variance extenstion

    Returns
    -------
    wave : :py:class:`~numpy.ndarray`
        Wavelenth array with a length of ``NAXIS1``.
    data : :py:class:`~numpy.ndarray`
        Stacked spectra with a shape of ``(NAXIS2, NAXIS1) = (nbin, nwave)``.
    noise: :py:class:`~numpy.ndarray`
        Noise spectra with a shape of ``(NAXIS2, NAXIS1) = (nbin, nwave)``.
    """

    hdu = fits.open(infile)
    wave = get_wavelength(hdu, ext='FLUX', axis=1)
    return(wave, hdu[ext_data].data, np.sqrt(hdu[ext_var].data))


def stacking(fcube, ftable, fout):
    """Stack spectra from a cube based on Voronoi binning.

    The spectra is stacked by averaging pixels in a bin.
    So, the resulting spectra have an unit of
    :math:`10^{-20} erg/s/cm^2/A/pixel^2`.

    Parameters
    ----------
    fcube : str
        FITS file of MUSE cube.
    ftable : str
        ``xy2bin`` FITS file created by the Voronoi binning process.
    fout : str
        Output FITS file. The output FITS file has a format of
        ``(NAXIS2, NAXIS2) = (nbin, nwave)`` for 1st and 2nd extentions.
        The stacked spectra and corresponding noise are
        stored in the 1st (``FLUX``) and 2nd (``VAR``) extensions, repectively.

    """

    tb_xy2bin = Table.read(ftable)
    cube = Cube(fcube)

    bin_start = tb_xy2bin['bin'].min()
    bin_end = tb_xy2bin['bin'].max()

    nbin = bin_end - bin_start + 1

    nwave = cube.wave.coord().size
    stack_spec = np.empty((nbin, nwave), dtype=np.float) + np.nan
    stack_var  = np.empty((nbin, nwave), dtype=np.float) + np.nan

    # for i in xrange(20):
    for i in xrange(nbin):

        idx = tb_xy2bin['bin'] == i
        xbin = tb_xy2bin['x'][idx]
        ybin = tb_xy2bin['y'][idx]

        stack_data_bin = np.array([cube.data[:, iy, ix]
                                   for iy, ix in zip(ybin, xbin)])
        stack_data_bin = np.nansum(stack_data_bin, axis=0)

        stack_var_bin = np.array([cube.var[:, iy, ix]
                                  for iy, ix in zip(ybin, xbin)])
        stack_var_bin = np.nansum(stack_var_bin, axis=0)

        stack_data_bin /= tb_xy2bin['bin'][idx].size
        stack_var_bin /= tb_xy2bin['bin'][idx].size**2

        stack_spec[i, :] = stack_data_bin
        stack_var[i, :] = stack_var_bin

    prihdu = fits.PrimaryHDU()
    data_hdu = fits.ImageHDU(data=stack_spec, name='FLUX')
    var_hdu = fits.ImageHDU(data=stack_var, name='VAR')

    for hdu in [data_hdu, var_hdu]:
        hdu.header['CRPIX1'] = cube.data_header['CRPIX3']
        hdu.header['CRVAL1'] = cube.data_header['CRVAL3']
        if 'CDELT3' in cube.data_header:
            hdu.header['CDELT1'] = cube.data_header['CDELT3']
        elif 'CD3_3' in cube.data_header:
            hdu.header['CDELT1'] = cube.data_header['CD3_3']
        hdu.header['CTYPE1'] = cube.data_header['CTYPE3']
        hdu.header['CUNIT1'] = cube.data_header['CUNIT3']

        if 'BUNIT' in cube.data_header:
            hdu.header['BUNIT'] = cube.data_header['BUNIT']
        if 'FSCALE' in cube.data_header:
            hdu.header['FSCALE'] = cube.data_header['FSCALE']

    hdulist = fits.HDUList([prihdu, data_hdu, var_hdu])
    hdulist.writeto(fout, clobber=True)

    return()


def subtract_ppxf_continuum_simple(voronoi_binspec_file, ppxf_npy_dir,
                                   ppxf_npy_prefix, outfile):
    """Subtract continuum defined as the best-fit of pPXF
    from the Voronoi binned spectra.
    Here, 'simple' means that the Voronoi binning is assumed to be
    identical for stellar continuum fitting and parent binned spectra.

    Parameters
    ----------
    voronoi_binspec_file : str
        Input file of Voronoi binned stacked spectra (FITS format).
    ppxf_npy_dir : str
        Name of a directory storing pPXF results (i.e., ``.npy`` files).
    ppxf_npy_prefix : str
        Prefix of the ``.npy`` files.
    outfile : str
        Output file name (FITS format). The shape will be ``(nbins, nwave)``.

    Returns
    -------
    wave : array_like
        Wavelength.
    flux : :py:class:`~numpy.ndarray`
        Continuum subtracted 2D array of Voronoi binned spectra
        with a shape of ``(nbins, nwave)``.
    var : :py:class:`~numpy.ndarray`
        2D array of Voronoi binned variance with a shape of ``(nbins, nwave)``.
    header : :py:class:`~astropy.io.fits.Header`
        FITS header object copied from the input file.
    """

    wave, flux, var = read_stacked_spectra(voronoi_binspec_file)

    nbins = flux.shape[0]

    flux_cont_sub = np.empty_like(flux) + np.nan
    var_cont_sub = np.empty_like(var) + np.nan

    for i in range(nbins):
        pp_npy = os.path.join(ppxf_npy_dir, ppxf_npy_prefix + '_%06i.npy' % i)
        pp = np.load(pp_npy)[0]

        best = np.interp(wave, pp.lam, pp.bestfit, left=np.nan, right=np.nan)

        flux_cont_sub[i, :] = flux[i, :] - best
        var_cont_sub[i, np.isfinite(best)] = var[i, np.isfinite(best)]

    hdu = fits.open(voronoi_binspec_file)
    hdu['FLUX'].data = flux_cont_sub
    hdu['VAR'].data = var_cont_sub

    hdu.writeto(outfile, clobber=True)

    return(wave, flux_cont_sub, var_cont_sub,
           fits.getheader(voronoi_binspec_file, 'FLUX'))


def subtract_emission_line(voronoi_binspec_file,
                           emission_line_file,
                           linename=None):
    """Subtract emission lines from observed (binned) spectra by using
    the best-fit parameters.

    Parameters
    ----------
    voronoi_binspec_file : str
        Input file of Voronoi binned stacked spectra (FITS format).
    emission_line_file : str
        Input file storing the best-fit emission line fitting parameters.
    linename : list, optional
        List of emission line names to be subtracted.
        By default, it scans headers of input FITS file
        and subtract all found lines.

    Returns
    -------
    hdulist : :py:class:`~astropy.io.fits.HDUList`
        HDUList object containging emission line subtracted spectra, variance,
        and emission line models in 1st, 2nd, and 3rd extensions, respectively.
        Headers are copied from the input FITS file.
    wave : :py:class:`~numpy.ndarray`
        Wavelength array reconstructed from the input header information.
    spec_out : :py:class:`~numpy.ndarray`
        Emission line subtracted spectra.
    var : :py:class:`~numpy.ndarray`
        Variance spectra, copied from the input cube
        (i.e., assuming noise less emission line models).
    emspec : :py:class:`~numpy.ndarray`
        Reconstruncted emission line spectra.
    """
    hdu = fits.open(voronoi_binspec_file)

    wave, flux, var = read_stacked_spectra(voronoi_binspec_file)

    hdu_em = fits.open(emission_line_file)

    # cont_model = hdu_em[0].header['MAD EMFIT CONT_MODEL']

    nbins = flux.shape[0]

    master_linelist = read_emission_linelist()
    if linename is None:
        linelist = master_linelist
    else:
        linelist = {}
        for k in linename:
            linelist[k] = master_linelist[k]

    emspec = np.zeros_like(flux)

    extname, keyname = search_lines(hdu_em, linelist)

    # MO: This loop may be slow as it scans
    #     all emission lines in the master list.
    for ibin in range(nbins):
        if ibin % 100 == 0:
            print("Emission line subtracted for %i/%i spectra."
                  % (ibin, nbins))
        for k in extname.keys():
            tmp_spec = gaussian(wave,
                                hdu_em[extname[k]].data['f_' + k][ibin],
                                hdu_em[extname[k]].data['vel'][ibin],
                                hdu_em[extname[k]].data['sig'][ibin],
                                linelist[k])
            emspec[ibin, :] += tmp_spec

    spec_out = flux - emspec

    hdulist = fits.HDUList([
        fits.PrimaryHDU(header=hdu[0].header),
        fits.ImageHDU(data=spec_out,
                      header=hdu[1].header, name='FLUX'),
        fits.ImageHDU(data=var,
                      header=hdu[2].header, name='VAR'),
        fits.ImageHDU(data=emspec,
                      header=hdu[1].header, name='EMSPEC')])

    return(hdulist, wave, spec_out, var, emspec)


def create_kinematics_image(hdu_segimg, tb_vel,
                            output_fits_name, is_save=True):
    """Create kinematics map based on a table object.

    Parameters
    ----------
    hdu_segimg : :py:class:`~astropy.io.fits.HDUList`
        HDUList of the segmentation image.
    tb_vel : :py:class:`~astropy.table.Table`
        Table object containing information on kinematics.
        The table must contain ``vel``, ``sig``, ``errvel``,
        and ``errsig`` keys.
    output_file_name : str
        Output file name (FITS format).
    is_save : bool
        Flag to save the resulting image into a multi-extension FITS image.
        Extension 1, 2, 3, and 4 correspond to velocity, velocity error,
        velocity dispersion, and velocity dispersion error, respectively.

    Returns
    -------
    velimg : :py:class:`~numpy.ndarray`
        Velocity image.
    errvelimg : :py:class:`~numpy.ndarray`
        Velocity error image.
    sigimg : :py:class:`~numpy.ndarray`
        Velocity dispersion image.
    errsigimg : :py:class:`~numpy.ndarray`
        Velocity dispersion error image.

    These images have the identical shape to the input segmentation image.
    """

    segimg = hdu_segimg[0].data

    velimg = create_value_image(segimg, tb_vel['vel'])
    sigimg = create_value_image(segimg, tb_vel['sig'])
    errvelimg = create_value_image(segimg, tb_vel['errvel'])
    errsigimg = create_value_image(segimg, tb_vel['errsig'])

    prihdu = fits.PrimaryHDU()
    vel_hdu = fits.ImageHDU(data=velimg, name='VEL')
    evel_hdu = fits.ImageHDU(data=errvelimg, name='ERRVEL')
    sig_hdu = fits.ImageHDU(data=sigimg, name='SIG')
    esig_hdu = fits.ImageHDU(data=errsigimg, name='ERRSIG')

    refheader = hdu_segimg[0].header

    for k in ['CRVAL3', 'CRPIX3', 'CDELT3', 'CTYPE3', 'CUNIT3']:
        if k in refheader:
            refheader.pop(k, None)

    for hdu in [vel_hdu, evel_hdu, sig_hdu, esig_hdu]:
        for k, v in refheader.items():
            if k != 'EXTNAME':
                hdu.header[k] = v

    hdulist = fits.HDUList([prihdu, vel_hdu, evel_hdu, sig_hdu, esig_hdu])
    hdulist.writeto(output_fits_name, clobber=True)

    return(velimg, errvelimg, sigimg, errsigimg)


def ql_binspec(ibin, infile, ext_data=1, ext_var=2):
    w, s, n = read_stacked_spectra(infile,
                                   ext_data=ext_data,
                                   ext_var=ext_var)

    plt.figure()
    plt.plot(w, s[ibin, :], '-', color='0.2')
    plt.plot(w, n[ibin, :], '-', color='0.2')

    plt.axhline(y=0, xmin=w[0], xmax=w[-1], linestyle='dashed')

    plt.xlim(w[0], w[-1])

    plt.show()
