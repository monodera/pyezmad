#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def compute_metallicity(fdic, efdic=None, calib=None):
    """Interface to compute 12+log(O/H) with various calibrations.
    Currently, (planned) supported calibrations are the following.
    - Marino et al. (2013): N2, O3N2
    - Pettini and Pagel (2004): N2, O3N2
    - Denicolo et al. (2002): N2
    - Maiolino et al. (2008): N2+[OIII]/Hbeta
    """

    if calib == "M13_N2":
        func_oh12 = oh12_marino2013_n2
    elif calib == "M13_O3N2":
        func_oh12 = oh12_marino2013_o3n2
    elif calib == 'PP04_N2':
        func_oh12 = oh12_pp03_n2
    elif calib == 'PP04_O3N2':
        func_oh12 = oh12_pp03_o3n2
