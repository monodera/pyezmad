#!/usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.version import LooseVersion, StrictVersion

import matplotlib
import matplotlib.cm as cm
import seaborn.apionly as sns


_cmap_ebv = sns.cubehelix_palette(light=1, as_cmap=True)

_cmap_mass = sns.cubehelix_palette(start=2.5, as_cmap=True)

_cmap_dict = {'white': cm.Greys_r,
              'npix': cm.Greys_r,
              'oh12': cm.YlGn,
              'ebv': _cmap_ebv,
              'sfr': cm.YlOrBr,
              'vel': cm.Spectral,
              'sigma': cm.YlOrRd,
              'sig': cm.YlOrRd,
              'age_ha': cm.Oranges,
              'ew': cm.YlGn_r,
              # 'mass': cm.Purples,
              'mass': _cmap_mass,
              'metal_star': cm.PuBu}

if LooseVersion(matplotlib.__version__) > LooseVersion('1.5'):
    _cmap_dict['age_star'] = cm.magma
else:
    _cmap_dict['age_star'] = cm.Oranges


white = _cmap_dict['white']
npix = _cmap_dict['npix']
oh12 = _cmap_dict['oh12']
ebv = _cmap_dict['ebv']
sfr = _cmap_dict['sfr']
vel = _cmap_dict['vel']
sigma = _cmap_dict['sigma']
sig = sigma
age_ha = _cmap_dict['age_ha']
ew = _cmap_dict['ew']
eqw = _cmap_dict['ew']
# mass = cm.Purples
mass = _cmap_dict['mass']
age_star = _cmap_dict['age_star']
metal_star = _cmap_dict['metal_star']


# def get_cmap(key, n=None):
#     return(cm.get_cmap(name=cmap_dict[key], lut=n))
