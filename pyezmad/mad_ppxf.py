#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import warnings
import time
import os
import os.path
import sys

import astropy.io.fits as fits
from astropy.table import Table
from astropy.constants import c
# from scipy import ndimage
import numpy as np

from multiprocessing import Process

import matplotlib.pyplot as plt

from tqdm import trange

from ppxf import ppxf
import ppxf_util as util

from .voronoi import read_stacked_spectra, create_value_image
from .utilities import get_wavelength, read_emission_linelist,\
    muse_fwhm, sigma2fwhm
# from .analysis_binned_spectra import BinSpecAnalysis
from .stellar_population import compute_ppxf_stellar_population


class Ppxf():
    def __init__(self, ppxf=None, segimg=None, mask=None, mask_img=None,
                 ppxf_dir1=None, ppxf_prefix1=None,
                 ppxf_dir2=None, ppxf_prefix2=None):
        if ppxf is not None:
            self.__segimg = segimg
            self.__mask = mask
            self.__mask_img = mask_img
            self.dir1 = ppxf_dir1
            self.prefix1 = ppxf_prefix1
            self.dir2 = ppxf_dir2
            self.prefix2 = ppxf_prefix2

            self.load_data(ppxf)

    def load_data(self, ppxf):
        self.__tb = Table.read(ppxf)

        self.__chisq1 = np.empty(self.__tb['vel'].size) + np.nan
        self.__chisq2 = np.empty(self.__tb['vel'].size) + np.nan

        for i in range(self.__tb['vel'].size):

            if self.dir1 is not None:
                file_pp = os.path.join(self.dir1,
                                       self.prefix1 + '_%06i.npy' % i)
                pp = np.load(file_pp)[0]
                if pp is not None:
                    self.__chisq1[i] = pp.chi2
            if self.dir2 is not None:
                file_pp = os.path.join(self.dir2,
                                       self.prefix2 + '_%06i.npy' % i)
                pp = np.load(file_pp)[0]
                if pp is not None:
                    self.__chisq2[i] = pp.chi2

        if self.__mask is not None:
            self.__tb['vel'][self.__mask] = np.nan
            self.__tb['errvel'][self.__mask] = np.nan
            self.__tb['sig'][self.__mask] = np.nan
            self.__tb['errsig'][self.__mask] = np.nan

        self.__vel = self.__tb['vel']
        self.__e_vel = self.__tb['errvel']
        self.__sig = self.__tb['sig']
        self.__e_sig = self.__tb['errsig']

    @property
    def tb(self):
        return(self.__tb)

    @property
    def vel(self):
        return(self.__vel)

    @property
    def e_vel(self):
        return(self.__e_vel)

    @property
    def sig(self):
        return(self.__sig)

    @property
    def e_sig(self):
        return(self.__e_sig)

    @property
    def segimg(self):
        return(self.__segimg)

    @property
    def chisq1(self):
        return(self.__chisq1)

    @property
    def chisq2(self):
        return(self.__chisq2)

    @property
    def mask(self):
        return(self.__mask)

    @property
    def mask_img(self):
        return(self.__mask_img)

    # create various maps
    def make_kinematics_img(self, vc=None):
        if vc is None:
            vc = np.nanmedian(self.__tb['vel'])
            warnings.warn(
                'Set velocity zero point by median(velocity(x,y)): %.2f' % vc)

        self.__vel_img = create_value_image(self.__segimg,
                                            self.__tb['vel']) - vc
        self.__sig_img = create_value_image(self.__segimg,
                                            self.__tb['sig'])
        self.__e_vel_img = create_value_image(self.__segimg,
                                              self.__tb['errvel'])
        self.__e_sig_img = create_value_image(self.__segimg,
                                              self.__tb['errsig'])

        if self.__mask_img is not None:
            self.__vel_img[self.__mask_img] = np.nan
            self.__e_vel_img[self.__mask_img] = np.nan
            self.__sig_img[self.__mask_img] = np.nan
            self.__e_sig_img[self.__mask_img] = np.nan

    @property
    def vel_img(self):
        return(self.__vel_img)

    @property
    def sig_img(self):
        return(self.__sig_img)

    @property
    def e_vel_img(self):
        return(self.__e_vel_img)

    @property
    def e_sig_img(self):
        return(self.__e_sig_img)

    def calc_stellarpop(self,
                        distance,
                        pp_dir=None,
                        pp_prefix=None,
                        parfile=None):

        if parfile is None:
            raise(
                ValueError(
                    "A file listing log Age/Gyr, [Z/H], "
                    "and stellar mass must be provided."))

        if (pp_dir is None) and (self.dir2 is not None):
            pp_dir = self.dir2
        if (pp_prefix is None) and (self.prefix2 is not None):
            pp_prefix = self.prefix2

        self.__lage, self.__lmetal, self.__smd, self.__lsmd \
            = compute_ppxf_stellar_population(self.__tb['vel'].size,
                                              distance,
                                              pp_dir,
                                              pp_prefix,
                                              parfile)

        self.__lage_img = create_value_image(self.__segimg,
                                             self.__lage)
        self.__lmetal_img = create_value_image(self.__segimg,
                                               self.__lmetal)
        self.__smd_img = create_value_image(self.__segimg,
                                            self.__smd)
        self.__lsmd_img = create_value_image(self.__segimg,
                                             self.__lsmd)

        if self.__mask is not None:
            self.__lage[self.__mask] = np.nan
            self.__lmetal[self.__mask] = np.nan
            self.__smd[self.__mask] = np.nan
            self.__lsmd[self.__mask] = np.nan

        if self.__mask_img is not None:
            self.__lage_img[self.__mask_img] = np.nan
            self.__lmetal_img[self.__mask_img] = np.nan
            self.__smd_img[self.__mask_img] = np.nan
            self.__lsmd_img[self.__mask_img] = np.nan

    @property
    def lage(self):
        return(self.__lage)

    @property
    def lmetal(self):
        return(self.__lmetal)

    @property
    def smd(self):
        return(self.__smd)

    @property
    def lsmd(self):
        return(self.__lsmd)

    @property
    def lage_img(self):
        return(self.__lage_img)

    @property
    def lmetal_img(self):
        return(self.__lmetal_img)

    @property
    def smd_img(self):
        return(self.__smd_img)

    @property
    def lsmd_img(self):
        return(self.__lsmd_img)


def _gaussian_filter1d(spec, sig):
    """
    Convolve a spectrum by a Gaussian with different sigma for every
    pixel, given by the vector "sigma" with the same size as "spec".
    If al sigma are the same this routine produces the same output as
    scipy.ndimage.gaussian_filter1d, except for the border treatment.
    Here the first/last p pixels are filled with zeros.
    When creating  template library for SDSS data, this implementation
    is 60x faster than the naive loop over pixels.

    Parameters
    ----------
    spec : :py:class:`~numpy:numpy.ndarray`
        Input spectrum. It must be 1D.
    sig : float
        Gaussian sigma for a convolution kernel.

    Returns
    -------
    conv_spectrum : :py:class:`~numpy:numpy.ndarray`
        Convolved spectrum with the same dimension as the input.

    Notes
    -----
    This function is copied from ppxf_utils.py from Michele Cappellari's
    original distribution.  Only modification over the original version
    is to use :py:func:`~numpy:numpy.nansum`
    instead of :py:func:`~numpy:numpy.sum`.
    """

    sig = sig.clip(0.01)  # forces zero sigmas to have 0.01 pixels
    p = int(np.ceil(np.max(3 * sig)))
    m = 2 * p + 1  # kernel size
    x2 = np.linspace(-p, p, m)**2

    n = spec.size
    a = np.zeros((m, n))
    for j in range(m):   # Loop over the small size of the kernel
        a[j, p:-p] = spec[j:n - m + j + 1]

    gau = np.exp(-x2[:, None] / (2 * sig**2))
    gau /= np.nansum(gau, 0)[None, :]

    conv_spectrum = np.nansum(a * gau, 0)

    return conv_spectrum


def determine_goodpixels(logLam, lamRangeTemp, z, dv_mask=800.,
                         linelist=None, is_mask_telluric=True):
    """Generates a list of goodpixels to mask a given set of gas emission lines.
       This is meant to be used as input for PPXF.

    Parameters
    ----------
    logLam : :py:class:`~numpy:numpy.ndarray`
        logarithmically rebinned wavelength grid (Note: it's natural log).
    LamRangeTemp : array_like
        Minimum and maximum wavelengths of template spectra.
    z : float
        Redshift to be applied to properly mask the emission lines.
    dv_mask : float
        Size of the mask in velocity space (km/s).
        ``v(z)+/-dv_mask`` will be masked.
    linelist : list
        List of emission line names as defined in
        `database/emission_lines.dat <../pyezmad/database/emission_lines.dat>`_
        in the repository.
        The default set is ["Hbeta", "OIII4959", "OIII5007",
        "NaI5890", "NaI5896", "OI6300", "NII6548", "NII6583",
        "Halpha", "SII6716", "SII6731"].
    is_mask_telluric : bool
        Mask telluric absorption bands (A, B, and gamma).

    Returns
    -------
    goodpixels : :py:class:`~numpy:numpy.ndarray`
        An array containing indices of valid pixels.
    """

    if linelist is None:
        # set a default linelist
        lines = np.array([4862.69,   # Hbeta
                          4960.30,   # [OIII]
                          5008.24,   # [OIII]
                          5891.583,  # Na I
                          5897.558,  # Na I
                          6302.04,   # [OI]
                          6548.03,   # [NII]
                          6583.41,   # [NII]
                          6562.80,   # Halpha
                          6716.47,   # [SII]
                          6730.85])  # [SII]
    else:
        emission_lines = read_emission_linelist()
        lines = np.array([emission_lines[lname] for lname in linelist])

    # width/2 of masked gas emission region in km/s
    dv = np.ones(lines.size) * dv_mask

    flag = logLam < 0  # empy mask
    for j in range(lines.size):
        flag |= (np.exp(logLam) >
                 lines[j] * (1 + z) * (1 - dv[j] / c.to('km/s').value)) & \
                (np.exp(logLam) <
                 lines[j] * (1 + z) * (1 + dv[j] / c.to('km/s').value))

    # mask telluric absorption bands
    if is_mask_telluric is True:
        # A, B, and gamma bands, respectively
        flag |= ((np.exp(logLam) > 7590.) & (np.exp(logLam) < 7720.))
        flag |= ((np.exp(logLam) > 6860.) & (np.exp(logLam) < 6950.))
        flag |= ((np.exp(logLam) > 6280.) & (np.exp(logLam) < 6340.))

    # Mask edges of stellar library
    flag |= np.exp(logLam) > (lamRangeTemp[1] *
                              (1 + z) * (1 - 900. / c.to('km/s').value))
    flag |= np.exp(logLam) < (lamRangeTemp[0] *
                              (1 + z) * (1 + 900. / c.to('km/s').value))

    return(np.where(flag == 0)[0])


def setup_spectral_library(file_template_list, velscale,
                           FWHM_inst, FWHM_templ,
                           wmin=None, wmax=None):
    """Set-up spectral library

    Parameters
    ----------
    file_template_list : str
        A file contains a list of template files.
        Now each template file is assumed to be a 1D FITS file
        containing flux at each pixel.  Wavelength inforamtion
        must be stored in the FITS header with (CRPIX1, CRVAL1, CDELT1).
    velscale : array_like
        ``velscale`` parameter derived from ``logRebin``.
    FWHM_inst : float
        Instrumental spectral resolution, FWHM in angstrom.
    FWHM_templ : float
        Spectral resolution of the templates, FWHM in angstrom.
    wmin : float, optional
        Minimum wavelength to consider for templates.
    wmax : float, optional
        Maximum wavelength to consider for templates.

    Returns
    -------
    templates : :py:class:`~numpy:numpy.ndarray`
        template array with a shape of ``(nwave, ntemplate)``.
    lamRange_temp : array_like
        Wavelength range of templates.
    logLam_temp : array_like
        Natural logarithmically rebinned wavelength array for templates.
    """

    # file_template_list = 'miles_ssp_padova_all.list'
    template_list = np.genfromtxt(file_template_list, dtype=None)

    # FWHM_tem = 2.51 # Vazdekis+10 spectra have a resolution FWHM of 2.51A.

    hdu = fits.open(template_list[0])
    ssp = hdu[0].data
    h2 = hdu[0].header
    wtempl = get_wavelength(hdu, ext=0, axis=1)

    if wmin is None:
        wmin = wtempl[0]
    else:
        wmin = np.max([wtempl[0], wmin])
    if wmax is None:
        wmax = wtempl[-1]
    else:
        wmax = np.min([wtempl[-1], wmax])

    idx_lam = np.logical_and(wtempl >= wmin, wtempl <= wmax)
    lamRange_temp = np.array([wtempl[idx_lam][0], wtempl[idx_lam][-1]])
    sspNew, logLam_temp, velscale = util.log_rebin(lamRange_temp,
                                                   ssp[idx_lam],
                                                   velscale=velscale)

    templates = np.empty((sspNew.size, template_list.size))

    if FWHM_inst is None:
        FWHM_inst_array = muse_fwhm(wtempl, deg=2)
    else:
        if isinstance(np.float(FWHM_inst), float):
            FWHM_inst_array = np.ones(wtempl.size) * FWHM_inst
        elif isinstance(FWHM_inst, np.ndarray):
            FWHM_inst_array = FWHM_inst
        else:
            raise(TypeError("FWHM_inst must be a scalar or numpy.ndarray."))

    FWHM_diff = np.sqrt((FWHM_inst_array**2 - FWHM_templ**2).clip(0.))

    # Sigma difference in pixels
    sigma = FWHM_diff / sigma2fwhm / h2['CDELT1']

    # norm_templ = np.empty(template_list.size)

    for i in trange(template_list.size, desc='Making Templates', leave=True):
        # if i % 100 == 0:
        #     print("....%4i/%4i templates are processed." %
        #           (i, template_list.size))
        hdu = fits.open(template_list[i])
        ssp = hdu[0].data
        # perform a convolution with wavelength-dependent sigma
        if np.any(sigma > 0.):
            # print("Convolving to instrumental resolution...")
            ssp = _gaussian_filter1d(ssp, sigma)
            # ssp = util.gaussian_filter1d(ssp, sigma)
        # if sigma > 0.:  # old version with scipy
        #     ssp = ndimage.gaussian_filter1d(ssp, sigma)
        sspNew, logLam2, velscale = util.log_rebin(lamRange_temp, ssp[idx_lam],
                                                   velscale=velscale)

        # norm = np.nanmedian(sspNew)
        # sspNew /= norm
        # norm_templ[i] = norm

        templates[:, i] = sspNew

    print("%i/%i templates are processed." %
          (template_list.size, template_list.size))

    return(templates, lamRange_temp, logLam_temp)  # , norm_templ


def run_voronoi_stacked_spectra_all(infile, npy_prefix, npy_dir='.',
                                        temp_list=None,
                                        vel_init=1000., sigma_init=50.,
                                        dv_mask=200.,
                                        wmin_fit=4800, wmax_fit=7000.,
                                        dw_edge=100.,
                                        n_thread=12,
                                        ext_data=1, ext_var=2,
                                        FWHM_inst=None, FWHM_tem=2.51,
                                        ppxf_kwargs=None, linelist=None,
                                        is_mask_telluric=True):
    """Run pPXF for all Voronoi binned spectra made with
    :py:meth:`pyezmad.voronoi.stack`.

    This function runs pPXF in parallel for Voronoi-binned spectra.
    It will save the results as ``pp`` object in a numpy binary form
    as ``.npy`` files in ``npy_dir``.
    These output files will be post-processed in the further analysis.
    It's a bit inconvenient, but at this moment, I'd like to keep as it is
    because we don't really know what best-fit parameters can be used later
    (and the shared memory functionality of Python's :py:mod:`multiprocessing`
    and the numpy array are not very friendly).

    Parameters
    ----------
    infile : str
        Input file, voronoi binned spectra with a shape of ``(nbins,nwave)``.
    npy_prefix : str
        Prefix for the output ``.npy`` files
        (output files will be ``npy_prefx_<bin ID>.npy``).
    npy_dir : str
        Directory to store the output ``.npy`` files.
    temp_list : str
        A file containing a list of template files.
    vel_init : floatm, optional
        Initial guess for line-of-sight velocities.
    sigma_init : float, optional
        Initial guess for line-of-sight  velocity dispersions.
    dv_mask : float, optional
        Velocity width to be masked for emission lines.
    wmin_fit : float, optional
        Minimum wavelength to run pPXF.
    wmax_fit : float, optional
        Maximum wavelength to run pPXF.
    dw_edge : float, optional
        Wavelengths additionally considered for templates.
        Templates from `wmin_fit-dw_edge` to `wmax_fit+dw_edge`
        will be considered. The default is 100 A.
    n_thread : int, optional
        Number of processes to be executed in parallel.
    ext_data : int or str, optional
        Data extension.
    ext_variance : int or str, optional
        Variance extension.
    FWHM_inst : array_like, optional
        Instrumental resolution in angstrom in FWHM.
        If it's ``None``, the MUSE resolution will be computed
        by :py:func:`pyezmad.mad_ppxf.muse_fwhm`.
    FWHM_temp : float, optional
        Template resolution in angstrom in FWHM.
    ppxf_kwargs : dict, optional
        Additional pPXF options.
    linelist : list, optional
        List of emission line names to be masked,
        e.g., ``['Halpha', 'Hbeta', 'OIII5007']``.
    is_mask_telluric : bool, optional
        Flag to determine whether to mask telluric absorption band or not.
    """

    if not os.path.exists(npy_dir):
        os.mkdir(npy_dir)

    wave0, galaxy0, noise0 = read_stacked_spectra(infile,
                                                  ext_data=ext_data,
                                                  ext_var=ext_var)
    nbins = galaxy0.shape[0]

    # ispec = 0 # absorption dominated spectra

    # Only use the lambda range in common between galaxy and stellar library.
    mask = np.logical_and(wave0 > wmin_fit, wave0 < wmax_fit)
    wave = wave0[mask]
    lamRange_galaxy = np.array([wave[0], wave[-1]])

    # dummy operation to get logLam_galaxy and velscale
    galaxy, logLam_galaxy, velscale \
        = util.log_rebin(lamRange_galaxy, galaxy0[0, mask])

    # ------------------- Setup templates -----------------------
    print("Preparing templates")
    stars_templates, lamRange_temp, logLam_temp \
        = setup_spectral_library(temp_list, velscale, FWHM_inst, FWHM_tem,
                                 wmin=wmin_fit - dw_edge,
                                 wmax=wmax_fit + dw_edge)
    # -----------------------------------------------------------

    dv = (np.log(lamRange_temp[0]) - np.log(wave[0])) * c.to('km/s').value

    goodPixels = determine_goodpixels(logLam_galaxy, lamRange_temp,
                                      vel_init / c.to('km/s').value,
                                      dv_mask=dv_mask,
                                      linelist=linelist,
                                      is_mask_telluric=is_mask_telluric)

    ppxf_keydic = dict(moments=4, degree=4, mdegree=4,
                       plot=False, clean=True, quiet=False)

    if ppxf_kwargs is not None:
        for k, v in ppxf_kwargs.items():
            ppxf_keydic[k] = v

    # for multiprocessing
    def run_ppxf_multiprocess(bins):

        for ibin in bins:
            if (ibin - bins[0]) % 10 == 0:
                print("....%6.3f %% finished [%i:%i] " %
                      ((ibin - bins[0]) * 1. /
                       ((bins[-1] - bins[0]) * 1.) * 100.,
                       bins[0], bins[-1]))

            galaxy, logLam_galaxy, velscale \
                = util.log_rebin(lamRange_galaxy, galaxy0[ibin, mask])
            noise2, logLam_galaxy, velscale \
                = util.log_rebin(lamRange_galaxy, noise0[ibin, mask]**2)
            noise = np.sqrt(noise2)

            # norm_galaxy = np.nanmedian(galaxy)
            # galaxy /= norm_galaxy
            # noise /= norm_galaxy

            pp_out = os.path.join(npy_dir, npy_prefix + '_%06i.npy' % (ibin))

            t_begin_each = time.time()
            try:
                pp = ppxf(stars_templates, galaxy, noise, velscale,
                          start=[vel_init, sigma_init, 0., 0.],
                          lam=np.exp(logLam_galaxy),
                          goodpixels=goodPixels, vsyst=dv,
                          **ppxf_keydic)
                t_end_each = time.time()
                if not ppxf_keydic['quiet']:
                    print("....Time elapsed for %i-th bin: %.2f [seconds]"
                          % (ibin, t_end_each - t_begin_each))

                # pp.weights *= (norm_galaxy * norm_templ)
                # pp.bestfit *= norm_galaxy
                # pp.galaxy *= norm_galaxy
                # pp.noise *= norm_galaxy

                pp.star = None
                pp.star_rfft = None
                pp.matrix = None

                np.save(pp_out, np.array([pp], dtype=np.object))
            except:
                print("Unexpected error:", sys.exc_info())
                print("pPXF failed at bin %i" % ibin)
                print("An array containing None is saved as %s" % pp_out)
                np.save(pp_out, np.array([None], dtype=np.object))

    #
    # parallelization
    #
    # nobj_per_proc = nbins / n_thread
    # ispec_start, ispec_end = 0, nbins
    # bins_begin = np.arange(ispec_start, ispec_end, nobj_per_proc)
    # bins_end = bins_begin + nobj_per_proc
    # bins_end[-1] = ispec_end

    bins_parallel = np.array_split(range(nbins), n_thread)
    # bins_begin = [bins_parallel[ib][0] for ib in range(n_thread)]
    # bins_end = [bins_parallel[ib][-1] for ib in range(n_thread)]

    print("pPXF will run on %i threads in parallel." % n_thread)
    print("Bins are split as follows.")
    print("=============")
    print(" Start    End")
    print("-------------")
    for b in bins_parallel:
        print("%6i %6i" % (b[0], b[-1]))
    print("-------------")

    processes = [Process(target=run_ppxf_multiprocess,
                         args=(bins_parallel[i],))
                 for i in range(len(bins_parallel))]

    t_start = time.time()

    die = (lambda vals:
           [val.terminate() for val in vals if val.exitcode is None])
    # die = (lambda vals:
    #        [val.terminate() for val in vals])

    try:
        for p in processes:
            p.start()
            # - have some sleep before submitting next process.
            # - not sure this helps, but sometimes it make pipe error,
            #   so I'm trying to do it. Maximum n_thread*0.5 sec loss, should be no problem.
            time.sleep(0.5)

        for p in processes:
            p.join()

        t_end = time.time()

    except Exception as e:
        die(processes)
        raise(e)
    except KeyboardInterrupt as e:
        die(processes)
        raise(e)

    print("Time Elapsed for pPXF run: %f [seconds]" % (t_end - t_start))


def ppxf_npy2array(ppxf_npy_dir, ppxf_npy_prefix):
    """Extract information of kinematics at each Voronoi bin
    from pPXF output ``.npy`` files.

    Parameters
    ----------
    ppxf_npy_dir : str
        Directory where pPXF's output ``.npy`` files are located.
    ppxf_npy_prefix : str
        File prefix for the ``.npy`` file from the pPXF run.

    Returns
    -------
    table : astropy.table.Table
        Table object containing
        ``(bin ID, velocity, sigma, velocity error, sigma error)``.
    binid : :py:class:`~numpy:numpy.ndarray`
        Voronoi bin ID.
    vel : :py:class:`~numpy:numpy.ndarray`
        Best velocity (km/s) solution.
    sig : :py:class:`~numpy:numpy.ndarray`
        Best sigma (km/s) solution.
    errvel : :py:class:`~numpy:numpy.ndarray`
        Formal error in velocity (km/s) corrected to :math:`\\chi^2_\\nu=1`.
    errsig : :py:class:`~numpy:numpy.ndarray`
        Formal error in velocity dispersion (km/s)
        corrected to :math:`\\chi^2_\\nu=1`.
    """

    ibin = 0

    vel, errvel = [], []
    sig, errsig = [], []

    while True:
        pp_npy = os.path.join(ppxf_npy_dir,
                              ppxf_npy_prefix + '_%06i.npy' % ibin)
        if os.path.exists(pp_npy) is True:
            pp = np.load(pp_npy)[0]
            if pp is None:
                vel.append(np.nan)
                sig.append(np.nan)
                errvel.append(np.nan)
                errsig.append(np.nan)
            else:
                vel.append(pp.sol[0])
                sig.append(pp.sol[1])
                errvel.append(pp.error[0] * np.sqrt(pp.chi2))
                errsig.append(pp.error[1] * np.sqrt(pp.chi2))
        else:
            break
        ibin += 1

    nbin = len(vel)

    tb_out = Table([np.arange(nbin),
                    np.array(vel), np.array(errvel),
                    np.array(sig), np.array(errsig)],
                   names=['bin', 'vel', 'errvel', 'sig', 'errsig'])

    return(tb_out,
           np.arange(nbin),
           np.array(vel), np.array(sig),
           np.array(errvel), np.array(errsig))


def show_output(ibin, voronoi_binspec_file, ppxf_npy_dir, ppxf_npy_prefix):
    """Quick-look at the output from pPXF run.

    Parameters
    ----------
    ibin : int
        Voronoi bin ID.
    voronoi_binspec_file : str
        Input FITS file containing Voronoi binned spectra.
    ppxf_npy_dir : str
        Directory where ``.npy`` files are stored.
    ppxf_npy_prefix : str
        File prefix for ``.npy`` files.

    """

    pp_npy = os.path.join(ppxf_npy_dir,
                          ppxf_npy_prefix + '_%06i.npy' % ibin)

    try:
        pp = np.load(pp_npy)[0]
    except IOError:
        print("File %s does not exist." % pp_npy)
        sys.exit()
    #
    # Print results
    #
    print("=======================================================================")
    print("    Best Fit:       V     sigma        h3        h4        h5        h6")
    print("-----------------------------------------------------------------------")
    print("    Values    ", "".join("%10.3g" % f for f in pp.sol))
    print("    Errors    ", "".join("%10.3g" % f for f in pp.error*np.sqrt(pp.chi2)))
    print("    chi2/DOF         : %.4g" % pp.chi2)
    print("    Nonzero Templates: %i / %i" % (np.sum(pp.weights > 0), pp.weights.size))
    print("-----------------------------------------------------------------------")

    #
    # Plotting
    #
    residual = pp.galaxy - pp.bestfit

    is_goodpixels = np.zeros(pp.galaxy.size, dtype=np.bool)
    is_goodpixels[pp.goodpixels] = True

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(pp.lam, pp.galaxy, '-', color='k')
    ax.plot(pp.lam, pp.bestfit, '-', color='tomato', lw=2, alpha=0.7)
    ax.fill_between(pp.lam, pp.noise, y2=0, color='dodgerblue', alpha=0.7)
    # ax.plot(pp.lam, residual, '-', color='k')

    residual_fit = residual.copy()
    residual_masked = residual.copy()
    residual_fit[~is_goodpixels] = np.nan
    residual_masked[is_goodpixels] = np.nan

    ax.plot(pp.lam, residual, '-', color='0.5')
    ax.plot(pp.lam, residual_masked, '-', color='seagreen', lw=1.5)

    ax.axhline(0, linestyle='dashed', color='k')

    favg = np.nanmedian(pp.galaxy)

    ax.set_xlim(pp.lam[0], pp.lam[-1])
    ax.set_ylim(-0.1 * favg, 1.5 * favg)
    ax.set_xlabel('Wavelength [angstrom]')
    ax.set_ylabel('Flux')


if __name__ == '__main__':
    print("do nothing")
