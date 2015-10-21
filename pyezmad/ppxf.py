#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import numpy as np
from astropy.table import Table

from .voronoi import create_value_image


class Ppxf:
    def __init__(self, ppxf=None, segimg=None):
        if ppxf is not None:
            self.load_data(ppxf)
            self.__segimg = segimg

    def load_data(self, ppxf):
        self.__tb = Table.read(ppxf)

    @property
    def tb(self):
        return(self.__tb)

    @property
    def segimg(self):
        return(self.__segimg)

    # create various maps
    def make_kinematics_img(self, vc=None):
        if vc is None:
            vc = np.nanmedian(self.__tb['vel'])
            warnings.warn(
                'Set velocity zero point by median(velocity(x,y)): %.2f' % vc)

        self.__vel_img = create_value_image(self.__segimg,
                                            self.__tb['vel']) - vc
        self.__sig_img = create_value_image(self.__segimg,
                                            self.__tb['sig'])
        self.__e_vel_img = create_value_image(self.__segimg,
                                              self.__tb['errvel'])
        self.__e_sig_img = create_value_image(self.__segimg,
                                              self.__tb['errsig'])

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
