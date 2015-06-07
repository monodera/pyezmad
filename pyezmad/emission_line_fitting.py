#!/usr/bin/env python

from __future__ import print_function

# import os, os.path
import numpy as np

from astropy.table import Table
from astropy.constants import c
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import lmfit

from .ppxf_kinematics import read_voronoi_stack
from .utilities import linelist

def search_lines(hdu, line_list):
    """Search extension ID and keys for the input list.

    Args:
        hdu: HDUList object of FITS file output from the emission_line_fitting function
        line_list: list of emission line to be searched

    Returns:
        extname: a python dictionary containing information which extension of
                the input hdu each emission line can be found.
        keyname: In the extension, which key one has to use to access the wavelength stored in the header.

    Note:
        If you linelist is  ['Hbeta', 'OIII5007', 'Halpha', 'NII6583'], it will return
            extname={'NII6583': 'GROUP1', 'OIII5007': 'GROUP0', 'Hbeta': 'GROUP0', 'Halpha': 'GROUP1'}
            keyname={'NII6583': 'LINE1', 'OIII5007': 'LINE2', 'Hbeta': 'LINE0', 'Halpha': 'LINE2'}
        Then, one can access, for example the flux of [OIII]5007 by
            flux = hdu[extname['OIII5007']].data['f_OIII5007']
        It's messy, so any suggestions to improve the accessiblitiy are welcome.
    """
    extname = {}
    keyname = {}

    for k in line_list:
        iext = 0
        iline = 0
        while True:
            # print(iext, iline)
            key_line = 'LINE%i' % iline
            if key_line in hdu[iext].header:
                if (hdu[iext].header[key_line]==k):
                    extname[k] = hdu[iext].header['EXTNAME']
                    keyname[k] = key_line
                    print('Key found for %s' % k)
                    break
                else:
                    iline += 1

            else:
                iext += 1
                iline = 0
    print(extname)
    print(keyname)
    return(extname, keyname)


def gaussian(x, flux, vel, sig, lam):
    lcen = lam * (1. + vel/c.to('km/s').value)
    lsig = lcen * sig / c.to('km/s').value
    return(flux/np.sqrt(2.*np.pi)/lsig * np.exp(-(x-lcen)**2/lsig**2))


def fit_single_spec(wave, flux, var, vel_star, linelist_name, dwfit=100., is_checkeach=False):

    out_fitting = np.empty(len(linelist_name), dtype=np.object)

    for j in range(len(linelist_name)):

        out_fitting_group = {'flux': {},
                             'eflux': {},
                             'vel': 0., 'sig': 0.,
                             'evel': 0., 'esig': 0.,
                             'cont_a': 0., 'cont_b': 0., 'cont_c': 0.,
                             'chi2': 0.}

        idx_fit = np.zeros(wave.size, dtype=np.bool)

        for k in range(len(linelist_name[j])):
            idx_fit_k = np.logical_and(wave>=linelist[linelist_name[j][k]]-dwfit,
                                       wave<=linelist[linelist_name[j][k]]+dwfit)
            idx_fit = np.logical_or(idx_fit, idx_fit_k)

        idx_fit = np.logical_and(idx_fit, np.isfinite(var))

        x = wave[idx_fit]
        y = flux[idx_fit]
        w = 1./var[idx_fit]

        models = {}

        # models['cont'] = lmfit.models.LinearModel(name='cont')
        models['cont'] = lmfit.models.QuadraticModel(name='cont', prefix='cont_')

        pars = models['cont'].guess(y*0., x=x)

        model = models['cont']

        for k in range(len(linelist_name[j])):
            lname = linelist_name[j][k]
            models[lname] = lmfit.models.Model(gaussian, prefix=lname+'_', name=lname)
            pars.update(models[lname].make_params())
            pars[lname+'_flux'].set(500., min=0.)
            pars[lname+'_lam'].set(linelist[lname], vary=False)

            if k > 0:
                # relate velocity and sigma to the 1st line in the group
                # Now sigma = 40km/s and v=v_star is assumed, but may be make them input
                pars[lname+'_vel'].set(vel_star, expr='%s_vel' % linelist_name[j][0])
                pars[lname+'_sig'].set(40., min=0., expr='%s_sig' % linelist_name[j][0])
            else:
                pars[lname+'_vel'].set(vel_star)
                pars[lname+'_sig'].set(40., min=0.)

            model += models[lname]

        # init = model.eval(pars, x=x)

        res_fit = model.fit(y, pars, x=x, weights=w)

        print(res_fit.fit_report(min_correl=0.5))

        out_fitting_group['vel'] = res_fit.best_values[lname+'_vel']
        out_fitting_group['sig'] = res_fit.best_values[lname+'_sig']
        out_fitting_group['errvel'] = res_fit.params[lname+'_vel'].stderr
        out_fitting_group['errsig'] = res_fit.params[lname+'_sig'].stderr
        out_fitting_group['cont_a'] = res_fit.best_values['cont_a']
        out_fitting_group['cont_b'] = res_fit.best_values['cont_b']
        out_fitting_group['cont_c'] = res_fit.best_values['cont_c']
        out_fitting_group['redchi'] = res_fit.redchi
        out_fitting_group['chisqr'] = res_fit.chisqr

        for k in range(len(linelist_name[j])):
            lname = linelist_name[j][k]
            # print(lname, res_fit.best_values[lname+'_flux'], res_fit.params[lname+'_flux'].stderr)
            out_fitting_group['flux'][lname] = res_fit.best_values[lname+'_flux']
            out_fitting_group['eflux'][lname] = res_fit.params[lname+'_flux'].stderr
        out_fitting[j] = out_fitting_group


        # plt.ion()
        if is_checkeach==True:
            plt.plot(x,y, 'k-')
            plt.plot(x, res_fit.best_fit, 'r-', lw=2, alpha=0.7)
            plt.show()

    return(out_fitting)

def emission_line_fitting(voronoi_binspec_file, ppxf_output_file, outfile, linelist_name, is_checkeach=False):
    """Fit emission lines to (continuum-subtracted) spectra with Voronoi output format.
    Args:
        voronoi_binspec_file: Input FITS file formatted as Voronoi binned stack, i.e., (nbins, nwave).
                              This supposed to be continuum subtracted, but may work for continuum
                              non-subtracted spectra. 
        ppxf_output_file: File name of FITS binary table which stores pPXF outputs (use velocity as an initial guess).
        outfile: Output file name (FITS file)
        linelist_name: List of emission lines to be fit.
                       When grouped with square brackets, they will be fit simultaneously by assuming
                       identical line-of-sight velocity and gaussian sigma.

    Returns:
       tbhdulist: HDUList object of output table. It's very messy format...
    """

    wave, flux, var = read_voronoi_stack(voronoi_binspec_file)
    tb_ppxf = Table.read(ppxf_output_file)

    res_fitting = np.empty(flux.shape[0], dtype=np.object)


    # It can be parallelized here.
    for i in xrange(flux.shape[0]):
        res_fitting[i] = fit_single_spec(wave, flux[i,:], var[i,:], tb_ppxf['vel'][i],
                                         linelist_name, is_checkeach=is_checkeach)

    prihdu = fits.PrimaryHDU()
    hdu_arr = []
    for j in range(len(linelist_name)):
        cols = []
        for k in range(len(linelist_name[j])):
            lname = linelist_name[j][k]
            col_val = fits.Column(name='f_'+lname, format='D',
                                  array=np.array([res_fitting[i][j]['flux'][lname] for i in xrange(flux.shape[0])]))
            col_err = fits.Column(name='ef_'+lname, format='D',
                                  array=np.array([res_fitting[i][j]['eflux'][lname] for i in xrange(flux.shape[0])]))
            cols.append(col_val)
            cols.append(col_err)

        for key in ['vel', 'sig', 'errvel', 'errsig', 'cont_a', 'cont_b', 'cont_c', 'redchi', 'chisqr']:
            cols.append(fits.Column(name=key, format='D',
                                    array=np.array([res_fitting[i][j][key] for i in xrange(flux.shape[0])])))

        tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols), name='GROUP%i' % j)

        for k in range(len(linelist_name[j])):
            tbhdu.header['LINE%i' % k] = linelist_name[j][k]
            tbhdu.header['WAVE%i' % k] = linelist[linelist_name[j][k]]

        hdu_arr.append(tbhdu)

    tbhdulist = fits.HDUList([prihdu]+hdu_arr)
    tbhdulist.writeto(outfile, clobber=True)

    return(tbhdulist)

if __name__ == '__main__':

    linelist_name = [['Hbeta', 'OIII4959', 'OIII5007'], # these 3 lines are fit together
                     ['NII6548', 'NII6583', 'Halpha'],
                     ['SII6717', 'SII6731']]


    # sn = [100,50,25]
    sn = [25]

    for i in range(len(sn)):
        voronoi_binspec_file = '../ppxf_run/ngc4980_voronoi_stack_sn%i_emspec.fits' % sn[i]
        ppxf_output_file = '../ppxf_run/ngc4980_ppxf_vel_sn%i.fits' % sn[i]
        outfile = 'ngc4980_voronoi_out_emprop_sn%i.fits' % sn[i]

        tbhdulist = emission_line_fitting(voronoi_binspec_file,
                                          ppxf_output_file,
                                          outfile,
                                          linelist_name,
                                          is_checkeach=False)
