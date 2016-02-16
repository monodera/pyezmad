#!/usr/bin/env python

from __future__ import print_function

import time
import os.path
import numpy as np

from astropy.table import Table
from astropy.constants import c
import astropy.io.fits as fits
from astropy import log
import matplotlib.pyplot as plt
import lmfit
from pyspeckit.parallel_map import parallel_map

from .utilities import read_emission_linelist, muse_fwhm, sigma2fwhm


def search_lines(hdu, line_list, verbose=True):
    """Search the extension ID and keys for the input list.

    Parameters
    ----------
    hdu : HDUList
        HDUList object of FITS file output from
        the emission_line_fitting function
    line_list : list
        A list containing names of spectral features (emission lines)
        to be searched from the master list.

    Returns
    -------
    extname : dict
        A python dictionary containing information which extension of
        the input hdu each emission line can be found.
    keyname : dict
        A python dictionary containing information
        about which key one has to use to access
        the wavelength stored in the header.

    Examples
    --------

    Here is an example. ::

        >>> from pyezmad.emission_line_fitting import search_lines
        >>> import astropy.io.fits as fits
        >>> hdu = fits.open('ngc4980_voronoi_out_emprop_sn50.fits')
        >>> line_list = ['Hbeta', 'OIII5007', 'Halpha', 'NII6583']
        >>> extname, keyname = search_lines(hdu, line_list)
        >>> print(extname)
        {'NII6583': 'GROUP1', 'OIII5007': 'GROUP0',
         'Hbeta': 'GROUP0', 'Halpha': 'GROUP1'}

        >>> print(keyname)
        {'NII6583': 'LINE1', 'OIII5007': 'LINE2',
        'Hbeta': 'LINE0', 'Halpha': 'LINE2'}

        >>> print(hdu[extname['OIII5007']].data['f_OIII5007'])
        [ 344.87768249  252.25565277  268.18625201 ...,   29.67768105
           17.77837443  134.66131437]

    The format is a bit messy, so any suggestions are welcome!
    """

    extname = {}
    keyname = {}

    for k in line_list:

        iext = 0
        iline = 0
        icomp = 0

        while True:
            try:
                maxcomp = hdu[iext].header['MAXCOMP']
            except:
                maxcomp = 1
            try:
                nline = hdu[iext].header['NLINE']
            except:
                nline = 99  # max number of lines is hard-coded as 99.

            if iext == len(hdu):
                break
            if icomp == maxcomp:
                break
            if iline == nline:
                icomp += 1
                iline = 0

            key_line = 'LINE%i_%i' % (iline, icomp)
            key_line_old = 'LINE%i' % (iline)  # for backward compatibility

            if (key_line in hdu[iext].header):
                if (hdu[iext].header[key_line] == k):
                    extname[k] = hdu[iext].header['EXTNAME']
                    keyname[k] = key_line
                    if verbose is True:
                        log.info('Key found for %s' % k)
                    break
                else:
                    iline += 1
            elif (key_line_old in hdu[iext].header):
                if (hdu[iext].header[key_line_old] == k):
                    extname[k] = hdu[iext].header['EXTNAME']
                    keyname[k] = key_line_old
                    if verbose is True:
                        log.info('Key found for %s' % k)
                    break
                else:
                    iline += 1
            else:
                iext += 1
                iline = 0
                icomp = 0
    return(extname, keyname)


def gaussian(x, flux, vel, sig, lam):
    """Gaussian function.

    Parameters
    ----------
    x : array_like
        Wavelength vector.
    flux : float or int
        Total flux.
    vel : float or int
        Line-of-sight velocity in km/s.
    sig : float or int
        Intrinsic line-of-sight velocity dispersion in km/s,
        i.e., instrumental resolution subtracted one.
        MUSE resolution is assumed now.
    lam : float or int
        Rest-frame wavelength in angstrom.

    Returns
    -------
    gaussian : array_like
        Gaussian function for a given parameter set.

    Notes
    -----

    .. math::

        G(x, f, v, \\sigma_v, \\lambda_\\text{in}) =
            \\frac{f}{\\sqrt{2\pi} \\sigma_\\lambda}
            \\exp\\left(-\\frac{(x-\\lambda_c)^2}{2\\sigma_\\lambda}\\right)

        \\lambda_c = \\lambda_\\text{in} \\left( 1 + \\frac{v}{c} \\right)

        \\sigma_\\lambda = \\lambda_c \\frac{\\sigma_v}{c}

    """

    lcen = lam * (1. + vel / c.to('km/s').value)  # angstrom
    # km/s
    sig_inst = c.to('km/s').value * (muse_fwhm(lcen) / sigma2fwhm) / lcen
    sig_tot = np.sqrt(sig**2 + sig_inst**2)  # km/s
    # lsig = lcen * sig / c.to('km/s').value
    lsig = lcen * sig_tot / c.to('km/s').value  # angstrom

    g = (flux / np.sqrt(2. * np.pi) / lsig) \
        * np.exp(-(x - lcen)**2 / 2. / lsig**2)

    return(g)


def fit_single_spec(wave, flux, var, vel_star,
                    linelist_name,
                    components=None,
                    dwfit=100., maxdv=None,
                    cont_model='linear', is_checkeach=False,
                    instrument=None, linelist=None, verbose=True, maxfev=None):
    """Fit Gaussian(s) to a single spectrum.

    Parameters
    ----------
    wave : array_like
        Wavelength array.
    flux : array_like
        Flux array.
    var : array_like
        Variance array.
    vel_star : float
        Velocity of stellar component in km/s.
        This will be used as an initial guess.
    linelist_name : list
        A list containing names of emission lines fit simultaneously.
    components : list, optional
        A list containing integers telling how many component in each set of
        `linelist_name` will be used.
    dwfit : float, optional
        Fitting will be done +/-dwfit angstrom
        from the line center in rest-frame.
        The default is 100.
    maxdv : float, optional
        Maximum allowed velocity offset from the ininital velocity input in km/s.
    is_checkeach : bool, optional
        If `True`, a plot showing the observed and best-fit spectra
        wiil be shown.  The default is `False`.
    instrument : str
        Name of the instrument to subtract instrumental resolution.
        Only MUSE is supported now.
    linelist : dictionary
        Master line list. Each entry in ``linelist_name``
        must be present in the ``linelist`` to
        extract the wavelength of it.
    cont_model : str, optional
        Continuum model. Either ``linear`` or ``quadratic`` is supported.
    maxfev : int
        Maximum number of iteratin for :py:func:`scipy.optimize.leastsq`.
        The default is set to be 1000, but not sure what is the best
        (cf. the default of :py:meth:`lmfit.Model.fit` is 2000*(nvar+1)
        where ``nvar`` is the number of variables).

    Returns
    -------
    outfitting : :py:class:`numpy.ndarray`
        ndarray of dictionaries containing fitting results.
        Each dictionary contains
        ``{'flux', 'eflux', 'vel', 'sig', 'evel', 'esig', 'cont_a', 'cont_b', 'cont_c', 'chi2'}``.

    """

    out_fitting = np.empty(len(linelist_name), dtype=np.object)

    if maxdv is None:
        vmin, vmax = None, None
    else:
        vmin, vmax = vel_star - maxdv, vel_star + maxdv

    # loop on each group of emission lines
    for j in range(len(linelist_name)):

        nline = len(linelist_name[j])

        ncomp = components[j]

        # initialize the dictionary to contain the results
        out_fitting_group = {'flux': {},
                             'eflux': {},
                             'cont_slope': np.nan,
                             'cont_intercept': np.nan,
                             'cont_a': np.nan,
                             'cont_b': np.nan,
                             'cont_c': np.nan,
                             'chi2': np.nan}
        for icomp in range(np.max(ncomp)):
            out_fitting_group['vel_%i' % icomp] = np.nan
            out_fitting_group['sig_%i' % icomp] = np.nan
            out_fitting_group['errvel_%i' % icomp] = np.nan
            out_fitting_group['errsig_%i' % icomp] = np.nan

        idx_fit = np.zeros(wave.size, dtype=np.bool)

        for k in range(nline):
            idx_fit_k = np.logical_and(
                wave >= linelist[linelist_name[j][k]] - dwfit,
                wave <= linelist[linelist_name[j][k]] + dwfit)
            idx_fit = np.logical_or(idx_fit, idx_fit_k)

        idx_fit = np.logical_and(idx_fit, np.isfinite(var))

        idx_cont1 = np.logical_and(wave >= wave[idx_fit].min() - dwfit,
                                   wave < wave[idx_fit].min())
        idx_cont2 = np.logical_and(wave > wave[idx_fit].max(),
                                   wave <= wave[idx_fit].max() + dwfit)
        idx_cont = np.logical_or(idx_cont1, idx_cont2)
        idx_cont = np.logical_and(idx_cont, np.isfinite(var))
        idx_cont = np.logical_and(idx_cont, np.isfinite(1. / var))

        x = wave[idx_fit]
        y = flux[idx_fit]
        w = 1. / np.sqrt(var[idx_fit])

        models = {}

        if cont_model == 'const':
            models['cont'] = lmfit.models.ConstantModel(name='cont',
                                                        prefix='cont_',
                                                        missing='drop')
        elif cont_model == 'linear':
            models['cont'] = lmfit.models.LinearModel(name='cont',
                                                      prefix='cont_',
                                                      missing='drop')
        elif cont_model == 'quadratic':
            models['cont'] = lmfit.models.QuadraticModel(name='cont',
                                                         prefix='cont_',
                                                         missing='drop')
        else:
            raise(
                ValueError(
                    "cont_model must be 'const', 'linear' or 'quadratic'."))

        # pars = models['cont'].guess(y * 0., x=x)
        pars = models['cont'].guess(flux[idx_cont], x=wave[idx_cont])

        model = models['cont']

        init_flux = np.trapz(y - np.nanmedian(flux[idx_cont]), x=x) / nline

        lname_common = []

        for icomp in range(np.max(ncomp)):

            iline = 0

            for k in range(nline):

                if icomp < ncomp[k]:
                    lname = linelist_name[j][k] + '_%i' % icomp
                    init_lam = linelist[linelist_name[j][k]]
                    models[lname] = lmfit.models.Model(gaussian,
                                                       prefix=lname + '_',
                                                       missing='drop',
                                                       name=lname)
                    pars.update(models[lname].make_params())
                    # MO: here, I force the flux of secondary or higher component
                    #     to be 10x weaker than the primary component.
                    #     Maybe, it's not a good assumption in some cases...
                    if icomp > 0:
                        init_flux /= 5.
                    pars[lname + '_flux'].set(init_flux, min=0.)
                    pars[lname + '_lam'].set(init_lam, vary=False)

                    if iline > 0:
                        # relate velocity and sigma to the 1st line in the group
                        # Now sigma = 40km/s and v=v_star is assumed,
                        # but may be make them input
                        pars[lname + '_vel'].set(vel_star,
                                                 expr='%s_vel' % (lname_common[icomp]))
                        pars[lname + '_sig'].set(40.,
                                                 min=0.,
                                                 expr='%s_sig' % (lname_common[icomp]))
                        iline += 1
                    else:
                        pars[lname + '_vel'].set(vel_star, min=vmin, max=vmax)
                        pars[lname + '_sig'].set(40., min=0.)
                        lname_common.append(lname)
                        iline += 1

                    model += models[lname]

        res_fit = model.fit(y, pars, x=x, weights=w,
                            fit_kws=dict(maxfev=maxfev),
                            verbose=verbose)

        if verbose is True:
            log.info(res_fit.fit_report(show_correl=False))
            log.info("res_fit.nfev = ", res_fit.nfev, end='\n')
            log.info("res_fit.success = ", res_fit.success, end='\n')
            log.info("res_fit.ier = %i" % res_fit.ier, end='\n')
            log.info("res_fit.lmdif_message: %s" % res_fit.lmdif_message, end='\n')

        for icomp in range(np.max(ncomp)):
            out_fitting_group['vel_%i' % icomp ] = res_fit.best_values[lname_common[icomp] + '_vel']
            out_fitting_group['sig_%i' % icomp] = res_fit.best_values[lname_common[icomp] + '_sig']
            out_fitting_group['errvel_%i' % icomp] = res_fit.params[lname_common[icomp] + '_vel'].stderr
            out_fitting_group['errsig_%i' % icomp] = res_fit.params[lname_common[icomp] + '_sig'].stderr

        if cont_model == 'const':
            out_fitting_group['cont_c'] = res_fit.best_values['cont_c']
        elif cont_model == 'linear':
            out_fitting_group['cont_slope'] \
                = res_fit.best_values['cont_slope']
            out_fitting_group['cont_intercept'] \
                = res_fit.best_values['cont_intercept']
        elif cont_model == 'quadratic':
            out_fitting_group['cont_a'] = res_fit.best_values['cont_a']
            out_fitting_group['cont_b'] = res_fit.best_values['cont_b']
            out_fitting_group['cont_c'] = res_fit.best_values['cont_c']

        out_fitting_group['redchi'] = res_fit.redchi
        out_fitting_group['chisqr'] = res_fit.chisqr

        out_fitting_group['nfev'] = res_fit.nfev
        out_fitting_group['success'] = res_fit.success
        out_fitting_group['ier'] = res_fit.ier
        out_fitting_group['lmdif_message'] = res_fit.lmdif_message

        for icomp in range(np.max(ncomp)):
            for k in range(len(linelist_name[j])):
                lname = linelist_name[j][k] + '_%i' % icomp
                try:
                    out_fitting_group['flux'][lname] \
                      = res_fit.best_values[lname + '_flux']
                    out_fitting_group['eflux'][lname] \
                      = res_fit.params[lname + '_flux'].stderr
                except:
                    out_fitting_group['flux'][lname] = np.nan
                    out_fitting_group['eflux'][lname] = np.nan

        out_fitting[j] = out_fitting_group

        # plt.ion()
        if is_checkeach is True:
            comps = res_fit.eval_components(x=x)
            for key, comp in comps.items():
                plt.plot(x, comp, '--', c='0.2', lw=1, label=key)
            plt.plot(x, y, 'k-')
            plt.fill_between(x, 1. / w, color='0.5', alpha=0.5)
            plt.plot(x, res_fit.best_fit, 'r-', lw=2, alpha=0.7)
            plt.show()

    return(out_fitting)


def emission_line_fitting(voronoi_binspec_file,
                          ppxf_output_file,
                          outfile,
                          linelist_name,
                          components=None,
                          dwfit=100.,
                          maxdv=None,
                          velocity=None,
                          cont_model='linear',
                          maxfev=1000,
                          is_checkeach=False,
                          instrument='MUSE',
                          master_linelist=None,
                          n_thread=12,
                          verbose=True):
    """Fit emission lines to (continuum-subtracted) spectra
    with Voronoi output format.

    Parameters
    ----------
    voronoi_binspec_file : str
        Input FITS file with a shape of Voronoi binned stacked spectra,
        i.e., (nbins, nwave).
        This supposed to be continuum subtracted, but may work for continuum
        non-subtracted spectra.
    ppxf_output_file : str
        File name of FITS binary table which stores pPXF outputs
        (use stellar line-of-sight velocity as an initial guess).
        Stellar line-of-sight velocity should be stored with the key 'vel'.
    outfile : str
        Output file name (FITS file)
    velocity : int, float, or numpy.ndarray, optional
        Velocity to be applied if `ppxf_output_file=None`.
        This will be ignored when `ppxf_output_file` is provided.
    linelist_name : list
        List of emission lines to be fit.
        When grouped with square brackets,
        they will be fit simultaneously by assuming
        identical line-of-sight velocity and velocity dispersion.
    components : list, optional
        A list containing integers telling how many component in each set of
        `linelist_name` will be used.
    dwfit : float, optional
        Wavelength +/- dwfit angstrom will be considered for the fitting.
        The Default value is 100.
    maxdv : float, optional
        Maximum allowed velocity offset from the ininital velocity input in km/s.
    is_checkeach : bool, optional
        Plot after each fitting.  The default is False.
    cont_model : str, optional
        Continuum model.
        Either ``const``, ``linear`` or ``quadratic`` is supported.
    instrument : str, optional
        Name of the instrument to subtract instrumental resolution.
        Only MUSE is supported now.
    master_linelist : dictionary, optional
        Master linelist for emission lines.
        By default, the list is loaded from
        a file in the ``database`` directory
        of the ``pyezmad`` distribution.
    n_thread : int
        Number of processes to be parallelized. The default is 12.
    verbose : bool, optional
        Print more details.

    Returns
    -------
    tbhdulist: HDUList object
        Output table as aHDUList object. It's very messy format...
        Each HDU contains the fitting result for each emission line group.
    """

    t_begin = time.time()

    if instrument != "MUSE":
        raise(ValueError("Only MUSE is supported as instrument."))

    if master_linelist is None:
        master_linelist = read_emission_linelist()

    if not isinstance(linelist_name, list):
        raise(TypeError("'linelist_name' must be a list even if there is only one component."))

    if not isinstance(linelist_name[0], list):
        raise(TypeError("Each element of 'linelist_name' must be a list even if there is only one component."))

    if (components is not None) and (len(linelist_name) != len(components)):
        raise(TypeError("Length of 'components' must be identical to that of 'linelist_name'."))

    if components is not None:
        for i in range(len(components)):
            if len(linelist_name[i]) != len(components[i]):
                raise(TypeError("Length of %i-th element of 'linelist_name' and 'components' do not match." % i))
    elif components is None:
        components = [np.ones_like(llist, dtype=np.int) for llist in linelist_name]

    # need to import here for some reason...
    from .voronoi import read_stacked_spectra

    wave, flux, var = read_stacked_spectra(voronoi_binspec_file)

    if ppxf_output_file is None:
        if isinstance(velocity, int) or isinstance(velocity, float):
            vel_init = np.ones(flux.shape[0], dtype=np.float) * velocity
        elif isinstance(velocity, np.ndarray):
            vel_init = velocity
    else:
        tb_ppxf = Table.read(ppxf_output_file)
        vel_init = tb_ppxf['vel']

    res_fitting = np.empty(flux.shape[0], dtype=np.object)

    t_start = time.time()

    def fit_parallel(tup):
        """Wrapper function for parallel processing.

        Parameters
        ----------
        tup : tuple
            A tuple containing the index of the bin to be fit.

        Returns
        -------
        ibin : int
            Index of the bin which will be used to check
            if the output array is properly sorted.
        tmp_res_fitting : numpy.ndarray
            Numpy array containing the output of the fitting.

        """
        ibin = tup
        tmp_res_fitting = fit_single_spec(wave,
                                          flux[ibin, :],
                                          var[ibin, :],
                                          # tb_ppxf['vel'][i],
                                          vel_init[ibin],
                                          linelist_name,
                                          components=components,
                                          dwfit=dwfit,
                                          maxdv=maxdv,
                                          cont_model=cont_model,
                                          maxfev=maxfev,
                                          is_checkeach=is_checkeach,
                                          instrument=instrument,
                                          linelist=master_linelist,
                                          verbose=verbose)
        if ibin % 100 == 0:
            log.info("Finished fit %i-th bin (%.1f%%). Elapsed time is %.2f minutes." %
                     (ibin, ((ibin + 1) * 1.) / flux.shape[0] * 100., (time.time() - t_start) / 60.))

        return(ibin, tmp_res_fitting)

    # Parallelization
    seq = [(i) for i in range(flux.shape[0])]
    result_all = parallel_map(fit_parallel, seq, numcores=n_thread)

    # Extract indices of bins
    idx_result0 = [result_all[i][0] for i in range(len(seq))]

    # Check if the output is sorted (as expected)
    if not np.all(sorted(idx_result0) == idx_result0):
        raise(ValueError("Index of index array after parallel processing is not sorted."))

    res_fitting = [result_all[i][1] for i in range(len(seq))]

    log.info("Finished fitting all bins.  Writing the output.")

    prihdu = fits.PrimaryHDU()
    prihdu.header['MAD EMFIT CONT_MODEL'] \
        = (cont_model, "continuum model for fitting.")
    prihdu.header['MAD EMFIT MAX_FEV'] \
        = (maxfev, "max iteration for fitting")

    hdu_arr = []

    for j in range(len(linelist_name)):
        cols = []
        for icomp in range(np.max(components[j])):
            for k in range(len(linelist_name[j])):
                lname = linelist_name[j][k] + '_%i' % icomp
                col_val = fits.Column(
                    name='f_' + lname, format='D',
                    array=np.array([res_fitting[i][j]['flux'][lname]
                                        for i in xrange(flux.shape[0])]))
                col_err = fits.Column(
                    name='ef_' + lname, format='D',
                    array=np.array([res_fitting[i][j]['eflux'][lname]
                                        for i in xrange(flux.shape[0])]))
                cols.append(col_val)
                cols.append(col_err)

            for key in ['vel', 'sig', 'errvel', 'errsig']:
                keykey = key + '_%i' % icomp
                cols.append(
                    fits.Column(
                        name=keykey, format='D',
                        array=np.array([res_fitting[i][j][keykey]
                                            for i in xrange(flux.shape[0])])))

        for key in ['cont_a', 'cont_b', 'cont_c',
                    'cont_intercept', 'cont_slope',
                    'redchi', 'chisqr']:
            cols.append(
                fits.Column(
                    name=key, format='D',
                    array=np.array([res_fitting[i][j][key]
                                        for i in xrange(flux.shape[0])])))

        cols.append(
            fits.Column(
                name='nfev', format='I',
                array=np.array([res_fitting[i][j]['nfev']
                                for i in xrange(flux.shape[0])])))
        cols.append(
            fits.Column(
                name='success', format='L',
                array=np.array([res_fitting[i][j]['success']
                                for i in xrange(flux.shape[0])])))
        cols.append(
            fits.Column(
                name='ier', format='I',
                array=np.array([res_fitting[i][j]['ier']
                                for i in xrange(flux.shape[0])])))

        tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols),
                                              name='GROUP%i' % j)

        for icomp in range(np.max(components[j])):
            for k in range(len(linelist_name[j])):
                lname = linelist_name[j][k] + '_%i' % icomp
                tbhdu.header['LINE%i_%i' % (k, icomp)] = lname
                tbhdu.header['WAVE%i_%i' % (k, icomp)] = master_linelist[linelist_name[j][k]]

        tbhdu.header['MAXCOMP'] = np.max(components[j])
        tbhdu.header['NLINE'] = len(components[j])

        hdu_arr.append(tbhdu)

    tbhdulist = fits.HDUList([prihdu] + hdu_arr)
    tbhdulist.writeto(outfile, clobber=True)

    return(tbhdulist)


if __name__ == '__main__':
    print("Nothing done.")
    exit()
