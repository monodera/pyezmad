#!/usr/bin/env python

from __future__ import print_function

import time
import os.path
import numpy as np

from astropy.table import Table
from astropy.constants import c
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import lmfit

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
        while True:
            if iext == len(hdu):
                break
            key_line = 'LINE%i' % iline
            if key_line in hdu[iext].header:
                if (hdu[iext].header[key_line] == k):
                    extname[k] = hdu[iext].header['EXTNAME']
                    keyname[k] = key_line
                    if verbose is True:
                        print('Key found for %s' % k)
                    break
                else:
                    iline += 1

            else:
                iext += 1
                iline = 0
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


def fit_single_spec(wave, flux, var, vel_star, linelist_name,
                    dwfit=100., cont_model='linear', is_checkeach=False,
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
    dwfit : float, optional
        Fitting will be done +/-dwfit angstrom
        from the line center in rest-frame.
        The default is 100.
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

    for j in range(len(linelist_name)):

        out_fitting_group = {'flux': {},
                             'eflux': {},
                             'vel': np.nan, 'sig': np.nan,
                             'evel': np.nan, 'esig': np.nan,
                             'cont_slope': np.nan,
                             'cont_intercept': np.nan,
                             'cont_a': np.nan,
                             'cont_b': np.nan,
                             'cont_c': np.nan,
                             'chi2': np.nan}

        idx_fit = np.zeros(wave.size, dtype=np.bool)

        for k in range(len(linelist_name[j])):
            idx_fit_k = np.logical_and(
                wave >= linelist[linelist_name[j][k]] - dwfit,
                wave <= linelist[linelist_name[j][k]] + dwfit)
            idx_fit = np.logical_or(idx_fit, idx_fit_k)

        idx_fit = np.logical_and(idx_fit, np.isfinite(var))

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

        pars = models['cont'].guess(y * 0., x=x)

        model = models['cont']

        for k in range(len(linelist_name[j])):
            lname = linelist_name[j][k]
            models[lname] = lmfit.models.Model(gaussian,
                                               prefix=lname + '_',
                                               missing='drop',
                                               name=lname)
            pars.update(models[lname].make_params())
            pars[lname + '_flux'].set(250., min=0.)
            pars[lname + '_lam'].set(linelist[lname], vary=False)

            if k > 0:
                # relate velocity and sigma to the 1st line in the group
                # Now sigma = 40km/s and v=v_star is assumed,
                # but may be make them input
                pars[lname + '_vel'].set(vel_star,
                                         expr='%s_vel' % linelist_name[j][0])
                pars[lname + '_sig'].set(40.,
                                         min=0.,
                                         expr='%s_sig' % linelist_name[j][0])
            else:
                pars[lname + '_vel'].set(vel_star)
                pars[lname + '_sig'].set(40., min=0.)

            model += models[lname]

        res_fit = model.fit(y, pars, x=x, weights=w,
                            fit_kws=dict(maxfev=maxfev),
                            verbose=verbose)

        # print(res_fit.fit_report(min_correl=0.5))
        print(res_fit.fit_report(show_correl=False))
        print("res_fit.nfev = ", res_fit.nfev, end='\n')
        print("res_fit.success = ", res_fit.success, end='\n')
        print("res_fit.ier = %i" % res_fit.ier, end='\n')
        print("res_fit.lmdif_message: %s" % res_fit.lmdif_message, end='\n')

        # NOTE: slightly confusing...
        # lname can be carried over to this point
        # as only one vel and sig values in each
        # group of emission lines.
        out_fitting_group['vel'] = res_fit.best_values[lname + '_vel']
        out_fitting_group['sig'] = res_fit.best_values[lname + '_sig']
        out_fitting_group['errvel'] = res_fit.params[lname + '_vel'].stderr
        out_fitting_group['errsig'] = res_fit.params[lname + '_sig'].stderr

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

        for k in range(len(linelist_name[j])):
            lname = linelist_name[j][k]
            out_fitting_group['flux'][lname] \
                = res_fit.best_values[lname + '_flux']
            out_fitting_group['eflux'][lname] \
                = res_fit.params[lname + '_flux'].stderr

        out_fitting[j] = out_fitting_group

        # plt.ion()
        if is_checkeach is True:
            plt.plot(x, y, 'k-')
            plt.plot(x, res_fit.best_fit, 'r-', lw=2, alpha=0.7)
            plt.show()

    return(out_fitting)


def emission_line_fitting(voronoi_binspec_file,
                          ppxf_output_file,
                          outfile,
                          linelist_name,
                          cont_model='linear',
                          maxfev=1000.,
                          is_checkeach=False,
                          instrument='MUSE',
                          master_linelist=None,
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
    linelist_name : list
        List of emission lines to be fit.
        When grouped with square brackets,
        they will be fit simultaneously by assuming
        identical line-of-sight velocity and velocity dispersion.
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

    # need to import here for some reason...
    from .voronoi import read_stacked_spectra

    wave, flux, var = read_stacked_spectra(voronoi_binspec_file)
    tb_ppxf = Table.read(ppxf_output_file)

    res_fitting = np.empty(flux.shape[0], dtype=np.object)

    # It can be parallelized here.  Any volunteers?
    for i in xrange(flux.shape[0]):

        if i % 100 == 0:
            t_mid = time.time()
            print("# \n" +
                  "# %i-th spectra is going to be fit\n" % (i) +
                  "%f minutes elapsed for the emission line fitting."
                  % ((t_mid - t_begin) / 60.))

        res_fitting[i] = fit_single_spec(wave,
                                         flux[i, :],
                                         var[i, :],
                                         tb_ppxf['vel'][i],
                                         linelist_name,
                                         cont_model=cont_model,
                                         maxfev=maxfev,
                                         is_checkeach=is_checkeach,
                                         instrument=instrument,
                                         linelist=master_linelist,
                                         verbose=verbose)
        print("# \n" +
              "# %i-th spectra has just finished\n" % i, end='\n')

    prihdu = fits.PrimaryHDU()
    prihdu.header['MAD EMFIT CONT_MODEL'] \
        = (cont_model, "continuum model for fitting.")
    prihdu.header['MAD EMFIT MAX_FEV'] \
        = (maxfev, "max iteration for fitting")

    hdu_arr = []

    for j in range(len(linelist_name)):
        cols = []
        for k in range(len(linelist_name[j])):
            lname = linelist_name[j][k]
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

        for key in ['vel', 'sig', 'errvel', 'errsig',
                    'cont_a', 'cont_b', 'cont_c',
                    'cont_intercept', 'cont_slope',
                    'redchi', 'chisqr']:
            cols.append(
                fits.Column(
                    name=key, format='D',
                    array=np.array([res_fitting[i][j][key]
                                    for i in xrange(flux.shape[0])])))

        cols.append(
            fits.Column(
                name=key, format='I',
                array=np.array([res_fitting[i][j]['nfev']
                                for i in xrange(flux.shape[0])])))
        cols.append(
            fits.Column(
                name=key, format='L',
                array=np.array([res_fitting[i][j]['sccucess']
                                for i in xrange(flux.shape[0])])))
        cols.append(
            fits.Column(
                name=key, format='I',
                array=np.array([res_fitting[i][j]['ier']
                                for i in xrange(flux.shape[0])])))


        tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols),
                                              name='GROUP%i' % j)

        for k in range(len(linelist_name[j])):
            tbhdu.header['LINE%i' % k] = linelist_name[j][k]
            tbhdu.header['WAVE%i' % k] = master_linelist[linelist_name[j][k]]

        hdu_arr.append(tbhdu)

    tbhdulist = fits.HDUList([prihdu] + hdu_arr)
    tbhdulist.writeto(outfile, clobber=True)

    return(tbhdulist)


if __name__ == '__main__':
    print("Nothing done.")
    exit()
