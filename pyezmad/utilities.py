#!/usr/bin/env python

import sys
import os.path
import numpy as np

import astropy.io.fits as fits
from astropy.visualization import scale_image
import astropy.units as u
from astropy.constants import c
# import aplpy

import matplotlib.pyplot as plt
import matplotlib.cm as cm


# may need to make it read-only, but now I trust everybody!
linelist = {'OII3726'  : 3727.09,
            'OII3729'  : 3729.88,
            'H12'      : 3751.22,
            'H11'      : 3771.70,
            'H10'      : 3798.98,
            'H9'       : 3836.48,
            'NeIII3869': 3869.81,
            'H8'       : 3890.15,
            'NeIII3967': 3968.53,
            'Hepsilon' : 3971.19,
            'Hdelta'   : 4102.92,
            'Hgamma'   : 4341.69,
            'Hbeta'    : 4862.69,
            'OIII4959' : 4960.30,
            'OIII5007' : 5008.24,
            'NII6548'  : 6549.84,
            'NII6584'  : 6585.23,
            'Halpha'   : 6564.61,
            'SII6717'  : 6718.32,
            'SII6731'  : 6732.71,
            'NII5755'  : 5756.24,
            'OIII4363' : 4364.44,
            'OI6300'   : 6302.04,
            'HeI3889'  : 3889.75,
            'HeI5876'  : 5877.30,
            'HeI6678'  : 6679.996,
            # duplicated keys...
            'NII6583'  : 6585.23
            }



def search_nearest_index(x, x0):
    return(np.argmin(np.abs(x-x0)))

def create_whitelight_image(infile, prefix_out,
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


def create_narrowband_image(hducube, wcenter, dw=None, vel=None, vdisp=None, nsig=3.):
    """Create a narrow band image

    Args:
        hducube: HDU object of astropy.io.fits for the input cube
        wcenter: central wavelength in angstrom
        dw: if specified, narrow-band image will be extracted wcenter+/-dw
        vel: input velocity (either scaler or 2D map with a same spatial dimension with the input cube)
                If specfied, wcenter is considered as the rest-frame wavelength and
                will be shited accordingly to the observed frame for the extraction.
        vdisp: velocity dispersion in km/s (either scalar or 2D map like velmap).
        nsig: narrow-band extract is carried out out to nsig times the velocity dispersion.

    Returns:
        2D numpy array with a shape of (NAXIS2, NAXIS1)
    """

    h = hducube[1].header
    wcube = h['CRVAL3'] + h['CDELT3']*(np.arange(h['NAXIS3'])-h['CRPIX3']+1)
    nbimg = np.empty((h['NAXIS2'],h['NAXIS1']))
    maskimg = np.nanmax(hducube[1].data, axis=0)
    maskimg[np.isfinite(maskimg)] = 1.

    if vel==None:
        # velmap = np.zeros_like(nbimg)
        zz = 1.
        ww = wcenter * np.ones_like(nbimg)
    else:
        zz = (1.+vel*u.km/u.s/c.to('km/s'))
        ww = wcenter * zz

    if dw == None:
        dwave = (vdisp*u.km/u.s) / c.to('km/s') * ww * nsig
    elif vdisp==None:
        dwave = dw * np.ones_like(nbimg)
        zz = 0.
    else:
        print("One of 'dw' or 'velmap/sigmap' must be specified! Exit.")
        sys.exit()

    wmin, wmax = ww-dwave, ww+dwave

    # FIXME: stupid looping is slow. There must be more efficient way.
    #        Need to calm down and relax to think about it!
    for ix in xrange(h['NAXIS1']):
        for iy in xrange(h['NAXIS2']):
            if np.isnan(maskimg[iy,ix])==True:
                continue
            idx_wmin = search_nearest_index(wcube, wmin[iy,ix])
            idx_wmax = search_nearest_index(wcube, wmax[iy,ix])
            tmpspec = hducube[1].data[idx_wmin:idx_wmax+1, iy, ix]
            nbimg[iy,ix] = np.nansum(tmpspec)

    nbimg[np.isnan(maskimg)] = np.nan

    return(nbimg)




if __name__=='__main__':
    print('do nothing')
    indir = '/net/astrogate/export/astro/shared/MAD/MUSE/P95/reduceddata/'
    infile = os.path.join(indir, 'NGC4980/NGC4980_FINAL.fits')

    # ha_img = make_narrowband_image(fits.open(infile),
    #                                wcenter=6564.61,
    #                                dw=None,
    #                                velmap=fits.getdata('../emission_line_fitting/ngc4980_em_kinematics_img_sn50.fits', 1),
    #                                sigmap=fits.getdata('../emission_line_fitting/ngc4980_em_kinematics_img_sn50.fits', 3))

    # o3_img = make_narrowband_image(fits.open(infile),
    #                                wcenter=5008.24,
    #                                dw=None,
    #                                velmap=fits.getdata('../emission_line_fitting/ngc4980_em_kinematics_img_sn50.fits', 1),
    #                                sigmap=fits.getdata('../emission_line_fitting/ngc4980_em_kinematics_img_sn50.fits', 3))

    # vband_img = make_narrowband_image(fits.open(infile),
    #                                   wcenter=5400,
    #                                   # dw=200.,
    #                                   velmap=fits.getdata('../emission_line_fitting/ngc4980_em_kinematics_img_sn50.fits', 1),
    #                                   sigmap=fits.getdata('../emission_line_fitting/ngc4980_em_kinematics_img_sn50.fits', 3))

    # iband_img = make_narrowband_image(fits.open(infile),
    #                                   wcenter=8000.,
    #                                   # dw=200.,
    #                                   velmap=fits.getdata('../emission_line_fitting/ngc4980_em_kinematics_img_sn50.fits', 1),
    #                                   sigmap=fits.getdata('../emission_line_fitting/ngc4980_em_kinematics_img_sn50.fits', 3))

    # fits.writeto('test_ngc4980_nbimg_halpha.fits', ha_img,
    #              fits.getheader('../emission_line_fitting/ngc4980_em_kinematics_img_sn50.fits', 1),
    #              clobber=True)
    # fits.writeto('test_ngc4980_nbimg_o3.fits', o3_img,
    #              fits.getheader('../emission_line_fitting/ngc4980_em_kinematics_img_sn50.fits', 1),
    #              clobber=True)

    # fits.writeto('test_ngc4980_nbimg_vband.fits', vband_img,
    #              fits.getheader('../emission_line_fitting/ngc4980_em_kinematics_img_sn50.fits', 1),
    #              clobber=True)

    # fits.writeto('test_ngc4980_nbimg_iband.fits', iband_img,
    #              fits.getheader('../emission_line_fitting/ngc4980_em_kinematics_img_sn50.fits', 1),
    #              clobber=True)

    # aplpy.make_rgb_image(['test_ngc4980_nbimg_halpha.fits',
    #                       'test_ngc4980_nbimg_cont.fits',
    #                       'test_ngc4980_nbimg_o3.fits'],
    #                       'test_ngc4980_colorimg.png',
    #                       stretch_r='arcsinh',
    #                       stretch_g='arcsinh',
    #                       stretch_b='arcsinh',
    #                       # pmin_r=0.1, pmax_r=99.9,
    #                       # pmin_g=0.1, pmax_g=99.9,
    #                       # pmin_b=0.1, pmax_b=99.9,
    #                       vmin_r=-10, vmax_r=8e3,
    #                       vmin_g=-10, vmax_g=5e3,
    #                       vmin_b=-10, vmax_b=1e4,
    #                       make_nans_transparent=True)
# INFO: Making alpha layer [aplpy.rgb]
# INFO: Red: [aplpy.rgb]
# INFO: vmin = -2.374e+01 (auto) [aplpy.rgb]
# INFO: vmax =  3.425e+04 (auto) [aplpy.rgb]
# INFO: Green: [aplpy.rgb]f
# INFO: vmin = -4.244e+01 (auto) [aplpy.rgb]
# INFO: vmax =  1.265e+03 (auto) [aplpy.rgb]
# INFO: Blue: [aplpy.rgb]
# INFO: vmin = -3.667e+01 (auto) [aplpy.rgb]
# INFO: vmax =  4.978e+04 (auto) [aplpy.rgb]

    aplpy.make_rgb_image(['test_ngc4980_nbimg_halpha.fits',
                          'test_ngc4980_nbimg_iband.fits',
                          'test_ngc4980_nbimg_vband.fits'],
                          'test_ngc4980_colorimg.png',
                          stretch_r='arcsinh',
                          stretch_g='arcsinh',
                          stretch_b='arcsinh',
                          # pmin_r=0.1, pmax_r=99.9,
                          # pmin_g=0.1, pmax_g=99.9,
                          # pmin_b=0.1, pmax_b=99.9,
                          vmin_r=-5, vmax_r=2e4,
                          vmin_g=-5, vmax_g=3e3,
                          vmin_b=-5, vmax_b=3e3,
                          make_nans_transparent=True)

# INFO: Making alpha layer [aplpy.rgb]
# INFO: Red: [aplpy.rgb]
# INFO: vmin = -1.121e+01 (auto) [aplpy.rgb]
# INFO: vmax =  1.521e+04 (auto) [aplpy.rgb]
# INFO: Green: [aplpy.rgb]
# INFO: vmin = -2.441e+01 (auto) [aplpy.rgb]
# INFO: vmax =  1.139e+03 (auto) [aplpy.rgb]
# INFO: Blue: [aplpy.rgb]
# INFO: vmin = -2.138e+01 (auto) [aplpy.rgb]
# INFO: vmax =  1.057e+03 (auto) [aplpy.rgb]
