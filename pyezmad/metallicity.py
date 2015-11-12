#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from pyezmad.utilities import (read_emission_linelist,
                               error_log10_fraction)
from pyezmad.extinction import get_alam_av


def compute_metallicity(fdic, efdic=None, calib=None,
                        unred=False, ebv=None, e_ebv=None,
                        extcurve='CCM'):
    """Interface to compute 12+log(O/H) with various calibrations.
    Currently, (planned) supported calibrations are the following.
    - Marino et al. (2013): N2, O3N2 **Implemented**
    - Pettini and Pagel (2004): N2, O3N2 **Implemented**
    - Denicolo et al. (2002): N2
    - Maiolino et al. (2008): N2+[OIII]/Hbeta

    Parameters
    ----------
    fdic : dict
    efdic : dict, optional
    calib : str
    unred : bool
    ebv : :py:class:`~numpy.ndarray`
    e_ebv : :py:class:`~numpy.ndarray`
    extcurve : str

    Returns
    -------
    oh12 : :py:class:`~numpy.ndarray`
    e_oh12 : :py:class:`~numpy.ndarray`
    """

    linelist = read_emission_linelist()

    if calib is None:
        raise(ValueError("""
        Calibration must be specified.
        Currently, "M13_N2", "M13_O3N2", "PP04_N2",
        and "PP04_O3N2" are supported."""))

    if calib == "M13_N2":
        func_oh12 = oh12_marino2013_n2
    elif calib == "M13_O3N2":
        func_oh12 = oh12_marino2013_o3n2
    elif calib == 'PP04_N2':
        func_oh12 = oh12_pp04_n2
    elif calib == 'PP04_O3N2':
        func_oh12 = oh12_pp04_o3n2

    # reddening correction if needed
    if unred is True:
        for k, v in fdic.items():
            alam_av, rv = get_alam_av(linelist[k], extcurve)
            expo_corr = 0.4 * ebv * alam_av * rv

            if efdic is not None:
                if e_ebv is not None:
                    efdic[k] = (fdic[k] * np.power(10., expo_corr) *
                                np.sqrt((efdic[k] / fdic[k])**2 +
                                        (np.log(10.) * expo_corr * e_ebv)**2))
                else:
                    efdic[k] = (fdic[k] * np.power(10., expo_corr) *
                                np.sqrt((efdic[k] / fdic[k])**2))

            fdic[k] *= np.power(10., expo_corr)

    oh12, e_oh12 = func_oh12(fdic, efdic)

    return(oh12, e_oh12)


def oh12_marino2013_n2(fdic, efdic):

    oh12, e_oh12 = None, None

    n2 = np.log10(fdic['NII6583'] / fdic['Halpha'])
    oh12 = 8.743 + 0.462 * n2

    if efdic is not None:
        e_n2 = error_log10_fraction(fdic['NII6583'], fdic['Halpha'],
                                    efdic['NII6583'], efdic['Halpha'])
        e_oh12 = 0.462 * e_n2

    return(oh12, e_oh12)


def oh12_marino2013_o3n2(fdic, efdic):

    oh12, e_oh12 = None, None

    o3n2 = np.log10((fdic['OIII5007'] / fdic['Hbeta']) /
                    (fdic['NII6583'] / fdic['Halpha']))
    oh12 = 8.533 - 0.214 * o3n2

    if efdic is not None:
        e_o3n2 = np.sqrt(
            error_log10_fraction(fdic['OIII5007'], fdic['Hbeta'],
                                 efdic['OIII5007'], efdic['Hbeta'])**2 +
            error_log10_fraction(fdic['NII6583'], fdic['Halpha'],
                                 efdic['NII6583'], efdic['Halpha'])**2)

        e_oh12 = 0.214 * e_o3n2

    return(oh12, e_oh12)


def oh12_pp04_n2(fdic, efdic):

    oh12, e_oh12 = None, None

    n2 = np.log10(fdic['NII6583'] / fdic['Halpha'])
    oh12 = 8.90 + 0.57 * n2

    if efdic is not None:
        e_n2 = error_log10_fraction(fdic['NII6583'], fdic['Halpha'],
                                    efdic['NII6583'], efdic['Halpha'])
        e_oh12 = 0.57 * e_n2

    return(oh12, e_oh12)


def oh12_pp04_o3n2(fdic, efdic):

    oh12, e_oh12 = None, None

    o3n2 = np.log10((fdic['OIII5007'] / fdic['Hbeta']) /
                    (fdic['NII6583'] / fdic['Halpha']))
    oh12 = 8.73 - 0.32 * o3n2

    if efdic is not None:
        e_o3n2 = np.sqrt(
            error_log10_fraction(fdic['OIII5007'], fdic['Hbeta'],
                                 efdic['OIII5007'], efdic['Hbeta'])**2 +
            error_log10_fraction(fdic['NII6583'], fdic['Halpha'],
                                 efdic['NII6583'], efdic['Halpha'])**2)

        e_oh12 = 0.32 * e_o3n2

    return(oh12, e_oh12)
