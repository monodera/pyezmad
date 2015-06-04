#!/usr/bin/env python
from __future__ import print_function

import sys
import os, os.path
import numpy as np
import matplotlib.pyplot as plt

import astropy.io.fits as fits
from astropy.table import Table
from pyezmad.voronoi import make_voronoi_value_image
from pyezmad.ppxf_kinematics import read_voronoi_stack

def subtract_continuum_simple(voronoi_binspec_file, ppxf_npy_dir, ppxf_npy_prefix, outfile):

    wave, flux, var = read_voronoi_stack(voronoi_binspec_file)

    nbins = flux.shape[0]
    # print(nbins)

    flux_cont_sub = np.empty_like(flux) + np.nan
    var_cont_sub = np.empty_like(var) + np.nan

    for i in range(nbins):
        pp_npy = os.path.join(ppxf_npy_dir, ppxf_npy_prefix+'_%06i.npy' % i)
        pp = np.load(pp_npy)[0]

        best = np.interp(wave, pp.lam, pp.bestfit, left=np.nan, right=np.nan)

        flux_cont_sub[i,:] = flux[i,:] - best
        var_cont_sub[i,np.isfinite(best)] = var[i,np.isfinite(best)]

    hdu = fits.open(voronoi_binspec_file)
    hdu['FLUX'].data = flux_cont_sub
    hdu['VAR'].data = var_cont_sub

    hdu.writeto(outfile, clobber=True)

    return(wave, flux_cont_sub, var_cont_sub,
           fits.open(voronoi_binspec_file)['FLUX'].header)

def ppxf_npy2array(ppxf_npy_dir, ppxf_npy_prefix):

    ibin = 0

    vel, errvel = [], []
    sig, errsig = [], []

    while True:

        pp_npy = os.path.join(ppxf_npy_dir, ppxf_npy_prefix+'_%06i.npy' % ibin)

        if os.path.exists(pp_npy) == True:
            pp = np.load(pp_npy)[0]
            vel.append(pp.sol[0])
            sig.append(pp.sol[1])
            errvel.append(pp.error[0]*np.sqrt(pp.chi2))
            errsig.append(pp.error[1]*np.sqrt(pp.chi2))
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


def make_voronoi_kinematics_image(hdu_segimg, tb_vel, output_fits_name):

    segimg = hdu_segimg[0].data

    velimg = make_voronoi_value_image(segimg, tb_vel['vel'])
    sigimg = make_voronoi_value_image(segimg, tb_vel['sig'])
    errvelimg = make_voronoi_value_image(segimg, tb_vel['errvel'])
    errsigimg = make_voronoi_value_image(segimg, tb_vel['errsig'])

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
        for k,v in refheader.items():
            if k!='EXTNAME':
                hdu.header[k] = v

    hdulist = fits.HDUList([prihdu, vel_hdu, evel_hdu, sig_hdu, esig_hdu])
    hdulist.writeto(output_fits_name, clobber=True)

    return(velimg, errvelimg, sigimg, errsigimg)


def show_ppxf_output(ibin, voronoi_binspec_file, ppxf_npy_dir, ppxf_npy_prefix):
    pp_npy = os.path.join(ppxf_npy_dir, ppxf_npy_prefix+'_%06i.npy' % ibin)
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
    ax = fig.add_subplot(1,1,1)
    ax.plot(pp.lam, pp.galaxy, '-', color='k')
    ax.plot(pp.lam, pp.bestfit, '-', color='tomato', lw=2, alpha=0.7)
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
    ax.set_ylim(-0.1*favg, 1.5*favg)
    ax.set_xlabel('Wavelength [angstrom]')
    ax.set_ylabel('Flux')

    # plt.show()



if __name__ == '__main__':

    ppxf_npy_dir = 'pp_npy_out_sn100'
    ppxf_npy_prefix = 'ngc4980_pp'
    voronoi_binspec_file = '../stacking/ngc4980_voronoi_stack_spec_sn50.fits'
    ibin = 61
    show_ppxf_output(ibin, voronoi_binspec_file, ppxf_npy_dir, ppxf_npy_prefix)
