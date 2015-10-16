# -*- coding: utf-8 -*-

import matplotlib.cm as cm

cmap_dic = {'white': cm.Greys_r,
            'npix': cm.Greys_r,
            'oh12': cm.YlGn,
            'ebv': cm.rainbow,
            'sfr': cm.YlOrBr,
            'vel': cm.rainbow,
            'sigma': cm.rainbow,
            'age_ha': cm.Oranges,
            'ew': cm.YlGn_r,
            'mass': cm.Purples,
            'age_star': cm.Oranges,
            'metal_star': cm.PuBu}


def get_cmap(key, n=None):
    return(cm.get_cmap(name=cmap_dic[key], lut=n))
