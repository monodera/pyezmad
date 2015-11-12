#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path
import numpy as np

import astropy.units as u
import astropy.constants as const

from .utilities import per_pixel_to_physical
from .emission_line_fitting import (search_lines,
                                    gaussian,
                                    read_emission_linelist)


def compute_ppxf_stellar_population(nbin,
                                    distance,
                                    pp_dir,
                                    pp_prefix,
                                    parfile):
    """Compute stellar population parameters based on pPXF results.

    Parameters
    ----------
    nbin : int
    distance : :py:class:`astropy.units`
    pp_dir : str
    pp_prefix : str
    parfile : str

    Returns
    -------
    lage : :py:class:`~numpy.ndarray`
        :math:`\\langle \\log Age/Gyr \\rangle `
    lmetal : :py:class:`~numpy.ndarray`
        :math:`\\langle [Z/H] \\rangle`
    smd: :py:class:`~numpy.ndarray`
        :math:`\\Sigma(M_\\odot\\,yr^{-1},pc^{-1})`
    lsmd: :py:class:`~numpy.ndarray`
        :math:`\\log \\Sigma/M_\\odot\\,yr^{-1},pc^{-1}`
    """

    tb_par = np.genfromtxt(parfile, dtype=None,
                           names=['ssp', 'lage', 'lmetal', 'mstar'])

    lage = np.empty(nbin) + np.nan
    lmetal = np.empty(nbin) + np.nan
    smd = np.empty(nbin) + np.nan
    lsmd = np.empty(nbin) + np.nan

    for i in range(nbin):

        pp_npy = os.path.join(pp_dir, pp_prefix + '_%06i.npy' % i)
        pp = np.load(pp_npy)[0]

        if pp is not None:
            # mass_weights = pp.weights / tb_par['mstar']
            mass_weights = pp.weights * tb_par['mstar']

            lage[i] = np.average(tb_par['lage'], weights=mass_weights)
            lmetal[i] = np.average(tb_par['lmetal'], weights=mass_weights)

            # convert to per solar mass weight
            tmp_smd = np.sum(mass_weights)

            smd[i] = tmp_smd

    # scaled to luminosity
    smd *= (4 * np.pi * distance.to('cm').value**2) * 1e-20
    # normalization
    smd /= u.Lsun.to('erg/s')
    smd *= per_pixel_to_physical(distance=distance, scale='pc')**2

    lsmd = np.log10(smd)

    return(lage, lmetal, smd, lsmd)


def compute_equivalent_width(line=None,
                             ppxf_npy_dir=None,
                             ppxf_npy_prefix=None,
                             hdu_em=None,
                             dw=100.):

        linelist = read_emission_linelist()

        if isinstance(line, str) is not True:
            raise(TypeError("'line' must be a string."))

        extname, keyname = search_lines(hdu_em, [line])

        nbin = hdu_em[extname[line]].data['f_' + line].size

        eqw = np.empty(nbin) + np.nan

        for i in range(nbin):

            pp_npy = os.path.join(ppxf_npy_dir,
                                  ppxf_npy_prefix + '_%06i.npy' % i)

            pp = np.load(pp_npy)[0]

            if pp is not None:
                g = gaussian(pp.lam,
                             hdu_em[extname[line]].data['f_' + line][i],
                             hdu_em[extname[line]].data['vel'][i],
                             hdu_em[extname[line]].data['sig'][i],
                             linelist[line])

                zz = 1. + (hdu_em[extname[line]].data['vel'][i] *
                           u.km / u.s / const.c.to('km/s'))
                idx = np.logical_and(pp.lam > (linelist[line] - dw) * zz,
                                     pp.lam < (linelist[line] + dw) * zz)
                eqtmp = np.trapz(g[idx] / pp.bestfit[idx], x=pp.lam[idx])
                # print(eqtmp, np.any(np.isnan(pp.bestfit[idx])),
                #       np.all(np.isfinite(pp.bestfit[idx])), zz,
                #       g[idx].size,
                #       np.mean(pp.lam[idx]))
                eqw[i] = eqtmp

        return(eqw)
