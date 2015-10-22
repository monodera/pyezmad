#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import warnings
# import numpy as np
import astropy.io.fits as fits
# import astropy.units as u
# from astropy.table import Table

from .utilities import get_wavelength
#                         error_fraction,
#                         read_emission_linelist)
# from .voronoi import Voronoi, create_value_image
# from .emission_line_fitting import search_lines
# from .extinction import ebv_balmer_decrement
# from .sfr import sfr_halpha

from .emissionline import EmissionLine
from .mad_ppxf import Ppxf
from .voronoi import Voronoi


class BinSpecAnalysis:
    """A class to work with Voronoi binned spectra."""
    def __init__(self,
                 binspec=None,
                 voronoi_xy=None,
                 voronoi_bininfo=None,
                 segimg=None,
                 ppxf=None,
                 emfit=None,):

        if binspec is not None:
            self.read_binspec(binspec)

        if (voronoi_xy is not None) and (voronoi_bininfo is not None):
            self.read_voronoi(voronoi_xy, voronoi_bininfo)

        if segimg is not None:
            self.read_segimg(segimg)

        if ppxf is not None:
            self.read_ppxf(ppxf)

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

    def read_ppxf(self, ppxf):
        self.ppxf = Ppxf(ppxf, self.segimg)

    def read_emfit(self, emfit):
        self.em = EmissionLine(emfit, self.segimg)

    @property
    def segimg(self):
        return(self.__segimg)
