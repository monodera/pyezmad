#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import numpy as np
import astropy.io.fits as fits

from .emission_line_fitting import search_lines
from .extinction import ebv_balmer_decrement
from .metallicity import compute_metallicity
from .sfr import sfr_halpha
from .voronoi import create_value_image
from .utilities import (error_fraction,
                        read_emission_linelist)


class EmissionLine:
    def __init__(self, emfit=None, segimg=None):

        self.__linelist = read_emission_linelist()

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
        self.__vel = self.__hdu[extname[refline]].data['vel']
        self.__sig = self.__hdu[extname[refline]].data['sig']
        self.__e_vel = self.__hdu[extname[refline]].data['errvel']
        self.__e_sig = self.__hdu[extname[refline]].data['errsig']

        if vc is None:
            vc = np.nanmedian(self.__vel)
            warnings.warn(
                'Set velocity zero point by median(velocity(x,y)): %.2f' % vc)

        self.__vel_img = create_value_image(self.__segimg,
                                            self.__vel) - vc
        self.__sig_img = create_value_image(self.__segimg,
                                            self.__sig)
        self.__e_vel_img = create_value_image(self.__segimg,
                                              self.__e_vel)
        self.__e_sig_img = create_value_image(self.__segimg,
                                              self.__e_sig)

    @property
    def vel(self):
        return(self.__vel)

    @property
    def sig(self):
        return(self.__sig)

    @property
    def e_vel(self):
        return(self.__e_vel)

    @property
    def e_sig(self):
        return(self.__e_sig)

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
        wave1 = self.__linelist[line]

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

    def calc_metallicity(self, calib=None, unred=False,
                         ebv=None, e_ebv=None,
                         extcurve='CCM'):

        if ebv is None:
            ebv = self.__ebv
        if e_ebv is None:
            e_ebv = self.__e_ebv

        extname, keyname = search_lines(self.__hdu, self.__linelist)

        fdic = dict()
        efdic = dict()

        for k in self.__linelist.keys():
            if k in keyname:
                fdic[k] = self.__hdu[extname[k]].data['f_' + k]
                efdic[k] = self.__hdu[extname[k]].data['ef_' + k]

        self.__oh12, self.__e_oh12 \
            = compute_metallicity(fdic, efdic=efdic, calib=calib,
                                  unred=unred, ebv=ebv, e_ebv=e_ebv,
                                  extcurve=extcurve)

        self.__oh12_img = create_value_image(self.__segimg,
                                             self.__oh12)
        self.__e_oh12_img = create_value_image(self.__segimg,
                                               self.__e_oh12)

    @property
    def oh12(self):
        return(self.__oh12)

    @property
    def e_oh12(self):
        return(self.__e_oh12)

    @property
    def oh12_img(self):
        return(self.__oh12_img)

    @property
    def e_oh12_img(self):
        return(self.__e_oh12_img)
