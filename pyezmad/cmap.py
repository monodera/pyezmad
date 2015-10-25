#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.cm as cm
import seaborn.apionly as sns

_cmap_ebv = sns.cubehelix_palette(light=1, as_cmap=True)

cmap_dict = {'white': cm.Greys_r,
             'npix': cm.Greys_r,
             'oh12': cm.YlGn,
             'ebv': _cmap_ebv,
             'sfr': cm.YlOrBr,
             'vel': cm.Spectral,
             'sigma': cm.YlOrRd,
             'sig': cm.YlOrRd,
             'age_ha': cm.Oranges,
             'ew': cm.YlGn_r,
             'mass': cm.Purples,
             'age_star': cm.Oranges,
             'metal_star': cm.PuBu}

white = cm.Greys_r
npix = cm.Greys_r
oh12 = cm.YlGn
ebv = _cmap_ebv
sfr = cm.YlOrBr
vel = cm.Spectral
sigma = cm.YlOrRd
sig = sigma
age_ha = cm.Oranges
ew = cm.YlGn_r
mass = cm.Purples
age_star = cm.Oranges
metal_star = cm.PuBu


# def get_cmap(key, n=None):
#     return(cm.get_cmap(name=cmap_dict[key], lut=n))
