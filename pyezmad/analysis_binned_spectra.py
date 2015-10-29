#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import warnings
import numpy as np
import astropy.io.fits as fits
# import astropy.units as u
# from astropy.table import Table

from .utilities import (get_wavelength,
                        map_pixel_major_axis)
#                         error_fraction,
#                         read_emission_linelist)
# from .voronoi import Voronoi, create_value_image
# from .emission_line_fitting import search_lines
# from .extinction import ebv_balmer_decrement
# from .sfr import sfr_halpha

from .emissionline import EmissionLine
from .mad_ppxf import Ppxf
from .voronoi import Voronoi, create_value_image
from .stellar_population import compute_equivalent_width


class BinSpecAnalysis:
    """A class to work with Voronoi binned spectra."""
    def __init__(self,
                 binspec=None,
                 voronoi_xy=None,
                 voronoi_bininfo=None,
                 segimg=None,
                 ppxf=None,
                 ppxf_dir1=None,
                 ppxf_prefix1=None,
                 ppxf_dir2=None,
                 ppxf_prefix2=None,
                 emfit=None,
                 max_npix=None,
                 ):

        if binspec is not None:
            self.read_binspec(binspec)

        if (voronoi_xy is not None) and (voronoi_bininfo is not None):
            self.read_voronoi(voronoi_xy, voronoi_bininfo)

        if segimg is not None:
            self.read_segimg(segimg)

        # create a mask to reject bins with too many pixels (>max_npix)
        self.__mask = np.zeros_like(self.voronoi.bininfo['npix'],
                                    dtype=np.bool)
        self.__mask_img = np.zeros_like(self.__segimg, dtype=np.bool)

        if (voronoi_xy is not None) and (segimg is not None):
            self.__npix_img = create_value_image(self.__segimg,
                                                 self.voronoi.bininfo['npix'])

            if max_npix is not None:
                self.__mask[np.where(self.voronoi.bininfo['npix'] >
                                     max_npix)] = True
                self.__mask_img[np.where(self.__npix_img > max_npix)] = True

        if ppxf is not None:
            self.read_ppxf(ppxf,
                           ppxf_dir1, ppxf_prefix1,
                           ppxf_dir2, ppxf_prefix2)

        if emfit is not None:
            self.read_emfit(emfit)

    # data readers
    def read_binspec(self, binspec):
        self.hdu = fits.open(binspec)
        self.wave = get_wavelength(self.hdu, axis=1, ext='FLUX')
        self.spec = self.hdu['FLUX'].data
        self.var = self.hdu['VAR'].data

    def read_voronoi(self, voronoi_xy, voronoi_bininfo):
        self.voronoi = Voronoi(voronoi_xy, voronoi_bininfo)

    def read_segimg(self, segimg):
        self.hdu_segimg = fits.open(segimg)
        self.__segimg = self.hdu_segimg[0].data

    def read_ppxf(self, ppxf,
                  ppxf_dir1, ppxf_prefix1,
                  ppxf_dir2, ppxf_prefix2):
        self.ppxf = Ppxf(ppxf,
                         segimg=self.segimg,
                         mask=self.__mask, mask_img=self.__mask_img,
                         ppxf_dir1=ppxf_dir1,
                         ppxf_prefix1=ppxf_prefix1,
                         ppxf_dir2=ppxf_dir2,
                         ppxf_prefix2=ppxf_prefix2)

    def read_emfit(self, emfit):
        self.em = EmissionLine(emfit, self.segimg,
                               self.__mask, self.__mask_img)

    @property
    def segimg(self):
        return(self.__segimg)

    def calc_elliptical_radius(self, xc, yc, pa, ellip):
        self.__r_ell = map_pixel_major_axis(self.voronoi.bininfo['xcen'],
                                            self.voronoi.bininfo['ycen'],
                                            xc, yc, pa, ellip)

    @property
    def r_ell(self):
        return(self.__r_ell)

    def calc_equivalent_width(self,
                              line=None,
                              ppxf_npy_dir=None,
                              ppxf_npy_prefix=None):

        self.__eqw = {}
        self.__eqw_img = {}

        if (ppxf_npy_dir is None) and (self.ppxf.dir2 is not None):
            ppxf_npy_dir = self.ppxf.dir2
        if (ppxf_npy_prefix is None) and (self.ppxf.prefix2 is not None):
            ppxf_npy_prefix = self.ppxf.prefix2

        if isinstance(line, str) is True:
            line = [line]

        for lname in line:
            self.__eqw[lname] \
                = compute_equivalent_width(
                    line=lname,
                    ppxf_npy_dir=ppxf_npy_dir,
                    ppxf_npy_prefix=ppxf_npy_prefix,
                    hdu_em=self.em.hdu)
            self.__eqw_img[lname] = create_value_image(self.segimg,
                                                       self.__eqw[lname])
            if self.__mask is not None:
                self.__eqw[lname][self.__mask] = np.nan
            if self.__mask_img is not None:
                self.__eqw_img[lname][self.__mask_img] = np.nan

    @property
    def eqw(self):
        return(self.__eqw)

    @property
    def eqw_img(self):
        return(self.__eqw_img)

    # def get_eqw(self, line=None):
    #     return(self.__eqw[line])
