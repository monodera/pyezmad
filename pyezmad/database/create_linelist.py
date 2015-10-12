#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import astropy.units as u
from astropy.table import Table

from astroquery.atomic import AtomicLineList, Transition

def obtain_table(element_spectrum=None, wmin=None, wmax=None, transitions=None):
    wrange = (wmin, wmax)
    tb_air = AtomicLineList.query_object(wrange, wavelength_type='Air', element_spectrum=element_spectrum, transitions=transitions)
    tb_vac = AtomicLineList.query_object(wrange, wavelength_type='Vacuum', element_spectrum=element_spectrum, transitions=transitions)

    tbout = Table([tb_air['SPECTRUM'], tb_air['LAMBDA AIR ANG'], tb_vac['LAMBDA VAC ANG']])
    return(tbout, tb_air, tb_vac)

if __name__ == '__main__':


    # tb_balmer, tb_balmer_air, tb_balmer_vac = obtain_table(element_spectrum='H', wmin=3700.*u.angstrom, wmax=6800.*u.angstrom)

    # tb_paschen, tb_paschen_air, tb_paschen_vac = obtain_table(element_spectrum='H', wmin=8350*u.angstrom, wmax=10000.*u.angstrom)

    # tb_oxygen, tb_oxygen_air, tb_oxygen_vac = obtain_table(element_spectrum='O I-III', wmin=4700*u.angstrom, wmax=10000.*u.angstrom,
    #                                                        transitions=Transition.nebular)


    # tb_nitrogen, tb_nitrogen_air, tb_nitrogen_vac = obtain_table(element_spectrum='N I-III', wmin=4700*u.angstrom, wmax=10000.*u.angstrom,
    #                                                        transitions=Transition.nebular)


    # tb_sulfur, tb_sulfur_air, tb_sulfur_vac = obtain_table(element_spectrum='S I-III', wmin=4700*u.angstrom, wmax=10000.*u.angstrom,
    #                                                        transitions=Transition.nebular)


    # # Too many transitions...
    # # tb_iron, tb_iron_air, tb_iron_vac = obtain_table(element_spectrum='Fe I-III', wmin=4700*u.angstrom, wmax=10000.*u.angstrom,
    # #                                                        transitions=Transition.nebular)
    # # print(tb_iron)


    # tb_argon, tb_argon_air, tb_argon_vac = obtain_table(element_spectrum='Ar I-III', wmin=4700*u.angstrom, wmax=10000.*u.angstrom,
    #                                                        transitions=Transition.nebular)


    # tb_cl, tb_cl_air, tb_cl_vac = obtain_table(element_spectrum='Cl I-III', wmin=4700*u.angstrom, wmax=10000.*u.angstrom,
    #                                                        transitions=Transition.nebular)
    # print(tb_balmer)
    # print(tb_paschen)
    # print(tb_oxygen)
    # print(tb_nitrogen)
    # print(tb_sulfur)
    # print(tb_argon)
    # print(tb_cl)


    # # Too many transitions...
    # # tb_ni, tb_ni_air, tb_ni_vac = obtain_table(element_spectrum='Ni I-III', wmin=4700*u.angstrom, wmax=10000.*u.angstrom,
    # #                                                        transitions=Transition.nebular)
    # # print(tb_ni)


    # for tb in [tb_balmer, tb_paschen, tb_oxygen, tb_nitrogen, tb_sulfur, tb_argon, tb_cl]:
    #     for i in range(tb['SPECTRUM'].size):
    #         print("%10s%4i  %9.4f  %9.4f" % (tb['SPECTRUM'][i].replace(" ", ""), round(tb['LAMBDA AIR ANG'][i]),
    #                                       tb['LAMBDA AIR ANG'][i], tb['LAMBDA VAC ANG'][i]))


    # tb_sodium, tb_sodium_air, tb_sodium_vac = obtain_table(element_spectrum='Na I', wmin=5500*u.angstrom, wmax=6000.*u.angstrom)
    # print(tb_sodium_air)
    # print(tb_sodium_vac)
    # print(tb_sodium)

    # for tb in [tb_sodium]:
    #     for i in range(tb['SPECTRUM'].size):
    #         print("%10s%4i  %9.4f  %9.4f" % (tb['SPECTRUM'][i].replace(" ", ""), round(tb['LAMBDA AIR ANG'][i]),
    #                                          tb['LAMBDA AIR ANG'][i], tb['LAMBDA VAC ANG'][i]))


    # tb_helium, tb_helium_air, tb_helium_vac = obtain_table(element_spectrum='He', wmin=4700*u.angstrom, wmax=10000.*u.angstrom,
    tb_helium, tb_helium_air, tb_helium_vac = obtain_table(element_spectrum='He I-II', wmin=4700*u.angstrom, wmax=10000.*u.angstrom,
                                                           transitions=Transition.nebular)
    print(tb_helium)
    # print(tb_helium_air)

    for tb in [tb_helium]:
        for i in range(tb['SPECTRUM'].size):
            print("%10s%4i  %9.4f  %9.4f" % (tb['SPECTRUM'][i].replace(" ", ""), round(tb['LAMBDA AIR ANG'][i]),
                                             tb['LAMBDA AIR ANG'][i], tb['LAMBDA VAC ANG'][i]))


