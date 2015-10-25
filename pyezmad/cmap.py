#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.cm as cm
import seaborn.apionly as sns

cmap_ebv = sns.cubehelix_palette(light=1, as_cmap=True)

cmap_dict = {'white': cm.Greys_r,
             'npix': cm.Greys_r,
             'oh12': cm.YlGn,
             'ebv': cmap_ebv,
             'sfr': cm.YlOrBr,
             'vel': cm.Spectral,
             'sigma': cm.YlOrRd,
             'age_ha': cm.Oranges,
             'ew': cm.YlGn_r,
             'mass': cm.Purples,
             'age_star': cm.Oranges,
             'metal_star': cm.PuBu}

white = cmap_dict['white']
npix = cmap_dict['npix']
oh12 = cmap_dict['oh12']
ebv = cmap_dict['ebv']
sfr = cmap_dict['sfr']
vel = cmap_dict['vel']
sigma = cmap_dict['sigma']
age_ha = cmap_dict['age_ha']
ew = cmap_dict['ew']
mass = cmap_dict['mass']
age_star = cmap_dict['age_star']
metal_star = cmap_dict['metal_star']


def get_cmap(key, n=None):
    return(cm.get_cmap(name=cmap_dict[key], lut=n))
