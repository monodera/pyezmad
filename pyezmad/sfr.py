#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import astropy.units as u

from .extinction import (extinction_ccm89,
                         extinction_calz)

from .utilities import (per_pixel_to_arcsec,
                        per_pixel_to_physical)


def sfr_halpha(f_obs, err=None, ebv=None, e_ebv=None,
               wave=None, extcurve=None, distance=None,
               scale=None):

        if extcurve in {'CCM', 'ccm'}:
            alam_av1, rv = extinction_ccm89(wave, ret_rv=True)
        elif extcurve in {'calzetti', 'Calzetti', 'Calz', 'calz'}:
            alam_av1, rv = extinction_calz(wave, ret_rv=True)
        elif extcurve is None:
            alam_av1, rv = 0., 0.
        else:
            raise(ValueError("%s curve is not yet implemented." % extcurve))

        expo_corr = 0.4 * ebv * alam_av1 * rv
        f_corr = f_obs * np.power(10., expo_corr)

        if not isinstance(distance, u.Quantity):
            distance *= u.Mpc

        # luminosity per pix (0.2*0.2 arcsec^2)
        lum = 4. * np.pi * distance.to('cm')**2 * f_corr
        lum *= u.erg / u.s / u.cm**2

        if scale in {'kpc', 'pc'}:
            pix_scaling = per_pixel_to_physical(distance, scale=scale)
        elif scale in {'arcsec'}:
            pix_scaling = per_pixel_to_arcsec()
        else:
            pix_scaling = 1.

        lum *= pix_scaling**2

        #
        # Use a calibration from Kennicutt and Evans (2012, ARAA, 50, 531)
        # This assumes Kroupa IMF (right?)
        #
        sfr = lum * np.power(10., -41.27)
        sfr *= (u.solMass / u.yr) / (u.erg / u.s)

        if (err is not None) and (e_ebv is not None):
            e_sfr = sfr * np.sqrt((err / f_obs)**2 +
                                  (np.log(10.) * expo_corr * e_ebv)**2)
        else:
            e_sfr = None

        lsfr = np.log10(sfr.value)

        if e_sfr is not None:
            e_lsfr = np.abs(e_sfr / sfr / np.log(10.))

        return(sfr, e_sfr, lsfr, e_lsfr)
