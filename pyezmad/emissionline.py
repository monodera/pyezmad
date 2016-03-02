#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import numpy as np
import astropy.io.fits as fits

import pyneb as pn

from .emission_line_fitting import search_lines
from .extinction import ebv_balmer_decrement
from .metallicity import compute_metallicity
from .sfr import sfr_halpha
from .voronoi import create_value_image
from .utilities import (error_fraction,
                        read_emission_linelist)


class EmissionLine:
    def __init__(self, emfit=None, segimg=None, mask=None, mask_img=None):

        self.__linelist = read_emission_linelist()

        if emfit is not None:
            self.__segimg = segimg
            self.__mask = mask
            self.__mask_img = mask_img

            self.load_data(emfit)

    def load_data(self, emfit):
        self.__hdu = fits.open(emfit)
        self.__ebv = np.zeros(self.__hdu[1].header['NAXIS2'])
        self.__e_ebv = np.zeros(self.__hdu[1].header['NAXIS2'])

    @property
    def hdu(self):
        return(self.__hdu)

    @property
    def segimg(self):
        return(self.__segimg)

    def get_flux(self, line=None, err=True):
        extname, keyname = search_lines(self.__hdu, [line])
        f = self.__hdu[extname[line]].data['f_' + line]
        ef = self.__hdu[extname[line]].data['ef_' + line]
        if err is True:
            return(f, ef)
        else:
            return(f)

    def make_kinematics_img(self, vc=None, refline='Halpha', component=None):
        v, ev, s, es = 'vel', 'errvel', 'sig', 'errsig'
        if component is not None:
            refline = refline + '_%i' % component
            v, ev = 'vel_%i' % component, 'errvel_%i' % component
            s, es = 'sig_%i' % component, 'errsig_%i' % component
        extname, keyname = search_lines(self.__hdu, [refline])
        self.__vel = self.__hdu[extname[refline]].data[v]
        self.__sig = self.__hdu[extname[refline]].data[s]
        self.__e_vel = self.__hdu[extname[refline]].data[ev]
        self.__e_sig = self.__hdu[extname[refline]].data[es]

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

        if self.__mask is not None:
            self.__vel[self.__mask] = np.nan
            self.__e_vel[self.__mask] = np.nan
            self.__sig[self.__mask] = np.nan
            self.__e_sig[self.__mask] = np.nan
        if self.__mask_img is not None:
            self.__vel_img[self.__mask_img] = np.nan
            self.__e_vel_img[self.__mask_img] = np.nan
            self.__sig_img[self.__mask_img] = np.nan
            self.__e_sig_img[self.__mask_img] = np.nan

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

    def calc_ebv(self, line1=None, line2=None, extcurve='CCM', component=None, clip=True):

        if line1 is None:
            line1 = 'Halpha'
        if line2 is None:
            line2 = 'Hbeta'

        if component is None:
            line1_i = line1
            line2_i = line2
        else:
            line1_i = line1 + '_%i' % component
            line2_i = line2 + '_%i' % component

        extname, keyname = search_lines(self.__hdu, [line1_i, line2_i])
        f1 = self.__hdu[extname[line1_i]].data['f_' + line1_i]
        f2 = self.__hdu[extname[line2_i]].data['f_' + line2_i]
        ef1 = self.__hdu[extname[line1_i]].data['ef_' + line1_i]
        ef2 = self.__hdu[extname[line2_i]].data['ef_' + line2_i]
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

        if self.__mask is not None:
            self.__ebv[self.__mask] = np.nan
            self.__e_ebv[self.__mask] = np.nan
        if self.__mask_img is not None:
            self.__ebv_img[self.__mask_img] = np.nan
            self.__e_ebv_img[self.__mask_img] = np.nan

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
                         component=None,
                         distance=None,
                         scale='kpc',
                         bunit=1e-20):

        if not ('Halpha' in line):
            # line != 'Halpha':
            raise(
                ValueError(
                    "Sorry, an input line other than Halpha is not supported."))

        if component is not None:
            line_i = line + '_%i' % component
        else:
            line_i = line

        extname, keyname = search_lines(self.__hdu, [line_i])
        wave1 = self.__linelist[line]

        f_obs = self.__hdu[extname[line_i]].data['f_' + line_i] * bunit
        ef_obs = self.__hdu[extname[line_i]].data['ef_' + line_i] * bunit

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

        if self.__mask is not None:
            self.__sfr_density[self.__mask] = np.nan
            self.__lsfr_density[self.__mask] = np.nan
            self.__e_sfr_density[self.__mask] = np.nan
            self.__e_lsfr_density[self.__mask] = np.nan
        if self.__mask_img is not None:
            self.__sfr_density_img[self.__mask_img] = np.nan
            self.__lsfr_density_img[self.__mask_img] = np.nan
            self.__e_sfr_density_img[self.__mask_img] = np.nan
            self.__e_lsfr_density_img[self.__mask_img] = np.nan

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
                         extcurve='CCM', component=None):

        if ebv is None:
            ebv = self.__ebv
        if e_ebv is None:
            e_ebv = self.__e_ebv

        if component is not None:
            linelist_i = {}
            for k, v in self.__linelist.items():
                linelist_i[k + '_%i' % component] = v
        else:
            linelist_i = self.__linelist

        extname, keyname = search_lines(self.__hdu, linelist_i)

        fdic = dict()
        efdic = dict()

        for k in self.__linelist.keys():
            kk = k + '_%i' % component
            if kk in keyname:
                fdic[k] = self.__hdu[extname[kk]].data['f_' + kk]
                efdic[k] = self.__hdu[extname[kk]].data['ef_' + kk]

        self.__oh12, self.__e_oh12 \
            = compute_metallicity(fdic, efdic=efdic, calib=calib,
                                  unred=unred, ebv=ebv, e_ebv=e_ebv,
                                  extcurve=extcurve)

        self.__oh12_img = create_value_image(self.__segimg,
                                             self.__oh12)
        self.__e_oh12_img = create_value_image(self.__segimg,
                                               self.__e_oh12)

        if self.__mask is not None:
            self.__oh12[self.__mask] = np.nan
            self.__e_oh12[self.__mask] = np.nan
        if self.__mask_img is not None:
            self.__oh12_img[self.__mask_img] = np.nan
            self.__e_oh12_img[self.__mask_img] = np.nan

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

    def calc_electron_density(self, line1=None, line2=None, tem=1e4, component=None):

        line1_i, line2_i = line1, line2
        if component is not None:
            line1_i = line1 + '_%i' % component
            line2_i = line2 + '_%i' % component

        if line1[0:3] == 'SII':
            extname, keyname = search_lines(self.__hdu, [line1_i, line2_i])
            ratio = (self.__hdu[extname[line1_i]].data['f_' + line1_i] /
                     self.__hdu[extname[line2_i]].data['f_' + line2_i])
            atom = pn.Atom('S', 2)
            self.__ne = compute_electron_density(ratio,
                                                 wave1=self.__linelist[line1],
                                                 wave2=self.__linelist[line2],
                                                 tem=tem,
                                                 atom=atom)
            self.__ne_img = create_value_image(self.__segimg,
                                               self.__ne)

            if self.__mask is not None:
                self.__ne[self.__mask] = np.nan
            if self.__mask_img is not None:
                self.__ne_img[self.__mask_img] = np.nan
        else:
            raise(ValueError("Only 'SII6717' and 'SII6731' are supported."))

    @property
    def ne(self):
        return(self.__ne)

    @property
    def ne_img(self):
        return(self.__ne_img)


def compute_electron_density(ratio,
                             wave1=None,
                             wave2=None,
                             tem=1e4,
                             atom=None):

    ne = atom.getTemDen(int_ratio=ratio, tem=tem, wave1=wave1, wave2=wave2)

    return(ne)
