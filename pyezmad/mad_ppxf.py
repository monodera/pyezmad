#!/usr/bin/env python
from __future__ import print_function

import time
import os, os.path

import astropy.io.fits as fits
from astropy.constants import c
from scipy import ndimage
import numpy as np

from multiprocessing import Process

from ppxf import ppxf
import ppxf_util as util

from .voronoi import read_stacked_spectra

def determine_goodpixels(logLam, lamRangeTemp, z, dv_mask=800.):
    """Generates a list of goodpixels to mask a given set of gas emission lines.
       This is meant to be used as input for PPXF.

    Args:
        logLam: logarithmically rebinned wavelength grid (Note: it's natural log)
        LamRangeTemp:minimum and maximum wavelength of template spectra
        z: redshift to be applied to properly mask the emission lines.
        dv_mask: size of the mask in velocity space (km/s). +/-dv_mask will be masked.

    Returns:
        An array containing indices of valid pixels. 

    """
#                     -----[OII]-----    Hdelta   Hgamma   Hbeta   -----[OIII]-----   [OI]    -----[NII]-----   Halpha   -----[SII]-----
    # lines = np.array([3726.03, 3728.82, 4101.76, 4340.47, 4861.33, 4958.92, 5006.84, 6300.30, 6548.03, 6583.41, 6562.80, 6716.47, 6730.85])
    #                 Hbeta    -----[OIII]-----  ------ Na I ------  [OI]     -----[NII]-----   Halpha   -----[SII]-----
    lines = np.array([4862.69, 4960.30, 5008.24, 5891.583, 5897.558, 6302.04, 6548.03, 6583.41, 6562.80, 6716.47, 6730.85])

    dv = np.ones(lines.size)*dv_mask # width/2 of masked gas emission region in km/s

    flag = logLam < 0  # empy mask
    for j in range(lines.size):
        flag |= (np.exp(logLam) > lines[j]*(1 + z)*(1 - dv[j]/c.to('km/s').value)) \
              & (np.exp(logLam) < lines[j]*(1 + z)*(1 + dv[j]/c.to('km/s').value))

    flag |= np.exp(logLam) > lamRangeTemp[1]*(1 + z)*(1 - 900/c.to('km/s').value)   # Mask edges of
    flag |= np.exp(logLam) < lamRangeTemp[0]*(1 + z)*(1 + 900/c.to('km/s').value)   # stellar library

    return np.where(flag==0)[0]

def setup_spectral_library(file_template_list, velscale, FWHM_gal, FWHM_tem):
    """Set-up spectral library

    Args:
        file_template_list: a file contains a list of template files.
                            Now a template file is assumed to be a 1D FITS file
                            containing flux at each pixel.  Wavelength inforamtion
                            must be stored in the FITS header with (CRPIX1, CRVAL1, CDELT1).
        velscale: velscale parameter derived from logRebin
        FWHM_gal: spectral resolution of the instrument, FWHM in angstrom 
        FWHM_tem: spectral resolution of the templates, FWHM in angstrom

    Returns: (templates, lamRange_temp, logLam_temp)
        templates: template array with a shape of (nwave, ntemplate)
        lamRange_temp: wavelength range of templates
        logLam_temp: natural logarithmically rebinned wavelength array for templates
    """

    # file_template_list = 'miles_ssp_padova_all.list'
    template_list = np.genfromtxt(file_template_list, dtype=None)

    # FWHM_tem = 2.51 # Vazdekis+10 spectra have a resolution FWHM of 2.51A.

    hdu = fits.open(template_list[0])
    ssp = hdu[0].data
    h2 = hdu[0].header
    lamRange_temp = h2['CRVAL1'] + np.array([0.,h2['CDELT1']*(h2['NAXIS1']-1)])
    sspNew, logLam_temp, velscale = util.log_rebin(lamRange_temp, ssp, velscale=velscale)

    templates = np.empty((sspNew.size,template_list.size))

    FWHM_dif = np.sqrt(FWHM_gal**2 - FWHM_tem**2)
    sigma = FWHM_dif/2.355/h2['CDELT1'] # Sigma difference in pixels

    for i in range(template_list.size):
        hdu = fits.open(template_list[i])
        ssp = hdu[0].data
        if sigma>0.:
            ssp = ndimage.gaussian_filter1d(ssp,sigma)
        sspNew, logLam2, velscale = util.log_rebin(lamRange_temp, ssp, velscale=velscale)
        templates[:,i] = sspNew # Templates are *not* normalized here

    return templates, lamRange_temp, logLam_temp

#------------------------------------------------------------------------------


# def ppxf_muse_voronoi_stack_all(infile, npy_prefix, npy_dir='.', temp_list=None,

def run_voronoi_stacked_spectra_all(infile, npy_prefix, npy_dir='.', temp_list=None,
                                    vel_init=1000., sigma_init=50., dv_mask=200.,
                                    wmin_fit=4800, wmax_fit=7000., n_thread=12,
                                    FWHM_muse=2.51, FWHM_tem=2.51):
    """Run pPXF for all Voronoi binned spectra formatted within this package scheme

    Args:
        infile: input file, voronoi binned spectra with a shape of (nbins,nwave)
        npy_prefix: prefix for output .npy files (output will be `npy_prefx_<bin ID>.npy`
        npy_dir: directory to store .npy files
        temp_list: list of templates
        vel_init: initial guess of line-of-sight velocity
        sigma_init: initial guess of velocity dispersion
        dv_mask: velocity width to be masked for emission lines
        wmin_fit: minimum wavelength to be fit
        wmax_fit: maximum wavelength ot be fit
        n_thread: number of processes to be executed in parallel
        FWHM_muse: instrumental resolution in angstrom
        FWHM_temp: template resolution in angstrom
    Returns: None
        .npy files will be saved in npy_dir.  They will be used for post-processing.
    """

    if not os.path.exists(npy_dir):
        os.mkdir(npy_dir)

    wave0, galaxy0, noise0 = read_stacked_spectra(infile)
    nbins = galaxy0.shape[0]

    # ispec = 0 # absorption dominated spectra

    # Only use the wavelength range in common between galaxy and stellar library.
    mask = np.logical_and(wave0>wmin_fit, wave0<wmax_fit)
    wave = wave0[mask]
    lamRange_galaxy = np.array([wave[0], wave[-1]])
    # FWHM_muse = 2.52 # MUSE instrumental resolution FWHM (to be confirmed)

    # dummy operation to get logLam_galaxy and velscale
    galaxy, logLam_galaxy, velscale = util.log_rebin(lamRange_galaxy, galaxy0[0,mask])

    #------------------- Setup templates -----------------------

    stars_templates, lamRange_temp, logLam_temp = setup_spectral_library(temp_list, velscale, FWHM_muse, FWHM_tem)
    stars_templates /= np.median(stars_templates) # Normalizes stellar templates by a scalar

    #-----------------------------------------------------------

    dv = (np.log(lamRange_temp[0])-np.log(wave[0]))*c.to('km/s').value # km/s
    # vel_init, sigma_init = 1400., 40.

    # goodPixels = util.determine_goodpixels(logLam_galaxy, lamRange_temp, vel/c)
    goodPixels = determine_goodpixels(logLam_galaxy, lamRange_temp, vel_init/c.to('km/s').value, dv_mask=dv_mask)

    def run_ppxf_multiprocess(bins_begin, bins_end):

        for ibin in np.arange(bins_begin, bins_end):
            if (ibin-bins_begin+1)%10==0:
                print("%f %% finished [%i:%i] " % ((ibin-bins_begin)*1./((bins_end-bins_begin)*1.)*100., bins_begin, bins_end))

            galaxy, logLam_galaxy, velscale = util.log_rebin(lamRange_galaxy, galaxy0[ibin,mask])
            noise2, logLam_galaxy, velscale = util.log_rebin(lamRange_galaxy, noise0[ibin,mask]**2)
            noise = np.sqrt(noise2)

            pp = ppxf(stars_templates, galaxy, noise, velscale,
                      start=[vel_init, sigma_init],
                      lam=np.exp(logLam_galaxy),
                      moments=4, degree=4, mdegree=4,
                      goodpixels=goodPixels, plot=False,
                      vsyst=dv, clean=True, quiet=False)

            pp.star = None
            pp.star_rfft = None
            pp.matrix = None

            np.save(os.path.join(npy_dir, npy_prefix+'_%06i.npy' % (ibin)), np.array([pp], dtype=np.object))

        # print("Formal errors:")
        # print("     dV    dsigma   dh3      dh4")
        # print("".join("%8.2g" % f for f in pp.error*np.sqrt(pp.chi2)))
        # # When the two Delta Chi^2 below are the same, the solution is the smoothest
        # # consistent with the observed spectrum.
        # #
        # print('Desired Delta Chi^2: %.4g' % np.sqrt(2*galaxy.size))
        # print('Current Delta Chi^2: %.4g' % ((pp.chi2 - 1)*galaxy.size))
        # print('Elapsed time in PPXF: %.2f s' % (clock() - t))


    ##
    ## parallelization
    ##
    # n_thread = 12
    nobj_per_proc = nbins/n_thread
    ispec_start, ispec_end = 0, nbins
    bins_begin = np.arange(ispec_start, ispec_end, nobj_per_proc)
    bins_end = bins_begin + nobj_per_proc
    bins_end[-1] = ispec_end
    print(bins_begin, bins_end)

    processes = [Process(target=run_ppxf_multiprocess, args=(bins_begin[i], bins_end[i])) for i in range(bins_begin.size)]

    t_start = time.time()

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    t_end = time.time()

    print("Time Elapsed for pPXF run: %f [seconds]" % (t_end-t_start))

#------------------------------------------------------------------------------

if __name__ == '__main__':
    infile = '../stacking/ngc4980_voronoi_stack_spec_sn100.fits'
    npy_prefix = 'ngc4980_pp'
    npy_dir = 'pp_npy_out'
    file_template_list = 'miles_ssp_padova_all.list'

    run_voronoi_stacked_spectra_all(infile, npy_prefix, npy_dir=npy_dir,
                                temp_list=file_template_list,
                                vel_init=1400., sigma_init=40., dv_mask=200.,
                                wmin_fit=4800, wmax_fit=7000., n_thread=12)
