#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import numpy as np
import astropy.io.fits as fits

from .emission_line_fitting import search_lines
from .extinction import ebv_balmer_decrement
from .sfr import sfr_halpha
from .voronoi import create_value_image
from .utilities import (error_fraction,
                        read_emission_linelist)


class EmissionLine:
    def __init__(self, emfit=None, segimg=None):
        if emfit is not None:
            self.load_data(emfit)
            self.__segimg = segimg

    def load_data(self, emfit):
        self.__hdu = fits.open(emfit)

    @property
    def hdu(self):
        return(self.__hdu)

    @property
    def segimg(self):
        return(self.__segimg)

    def make_kinematics_img(self, vc=None, refline='Halpha'):
        extname, keyname = search_lines(self.__hdu, [refline])
        self.vel = self.__hdu[extname[refline]].data['vel']
        self.sig = self.__hdu[extname[refline]].data['sig']
        self.e_vel = self.__hdu[extname[refline]].data['errvel']
        self.e_sig = self.__hdu[extname[refline]].data['errsig']

        if vc is None:
            vc = np.nanmedian(self.vel)
            warnings.warn(
                'Set velocity zero point by median(velocity(x,y)): %.2f' % vc)

        self.__vel_img = create_value_image(self.__segimg,
                                            self.vel) - vc
        self.__sig_img = create_value_image(self.__segimg,
                                            self.sig)
        self.__e_vel_img = create_value_image(self.__segimg,
                                              self.e_vel)
        self.__e_sig_img = create_value_image(self.__segimg,
                                              self.e_sig)

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

    def calc_ebv(self, line1=None, line2=None, extcurve='CCM', clip=True):

        extname, keyname = search_lines(self.__hdu, [line1, line2])
        f1 = self.__hdu[extname[line1]].data['f_' + line1]
        f2 = self.__hdu[extname[line2]].data['f_' + line2]
        ef1 = self.__hdu[extname[line1]].data['ef_' + line1]
        ef2 = self.__hdu[extname[line2]].data['ef_' + line2]
        r_obs = f1 / f2
        err_r_obs = error_fraction(f1, f2, ef1, ef2)

        (self.__ebv,
         self.__e_ebv) = ebv_balmer_decrement(r_obs,
                                              err=err_r_obs,
                                              line1=line1,
                                              line2=line2,
                                              extcurve=extcurve,
                                              clip=clip)

        self.__ebv_img = create_value_image(self.__segimg, self.__ebv)
        self.__e_ebv_img = create_value_image(self.__segimg, self.__e_ebv)

    @property
    def ebv(self):
        return(self.__ebv)

    @property
    def e_ebv(self):
        return(self.__e_ebv)

    @property
    def ebv_img(self):
        return(self.__ebv_img)

    @property
    def e_ebv_img(self):
        return(self.__e_ebv_img)

    def calc_sfr_density(self,
                         line='Halpha',
                         extcurve='CCM',
                         distance=None,
                         scale='kpc'):

        if line != 'Halpha':
            raise(
                ValueError(
                    "Sorry, an input line other than Halpha is not supposed."))

        extname, keyname = search_lines(self.__hdu, [line])
        linelist = read_emission_linelist()
        wave1 = linelist[line]

        f_obs = self.__hdu[extname[line]].data['f_' + line] * 1e-20
        ef_obs = self.__hdu[extname[line]].data['ef_' + line] * 1e-20

        (self.__sfr_density,
         self.__e_sfr_density,
         self.__lsfr_density,
         self.__e_lsfr_density) = sfr_halpha(f_obs,
                                             err=ef_obs,
                                             ebv=self.__ebv,
                                             e_ebv=self.__e_ebv,
                                             wave=wave1,
                                             extcurve=extcurve,
                                             distance=distance,
                                             scale=scale)

        self.__sfr_density_img = create_value_image(self.__segimg,
                                                    self.__sfr_density)
        self.__e_sfr_density_img = create_value_image(self.__segimg,
                                                      self.__e_sfr_density)
        self.__lsfr_density_img = create_value_image(self.__segimg,
                                                     self.__lsfr_density)
        self.__e_lsfr_density_img = create_value_image(self.__segimg,
                                                       self.__e_lsfr_density)

    @property
    def sfr_density(self):
        return(self.__sfr_density)

    @property
    def e_sfr_density(self):
        return(self.__e_sfr_density)

    @property
    def lsfr_density(self):
        return(self.__lsfr_density)

    @property
    def e_lsfr_density(self):
        return(self.__e_lsfr_density)

    @property
    def sfr_density_img(self):
        return(self.__sfr_density_img)

    @property
    def e_sfr_density_img(self):
        return(self.__e_sfr_density_img)

    @property
    def lsfr_density_img(self):
        return(self.__lsfr_density_img)

    @property
    def e_lsfr_density_img(self):
        return(self.__e_lsfr_density_img)
