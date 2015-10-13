#!/usr/bin/env python

from setuptools import setup

setup(name='pyezmad',
      version='0.1.10',
      description='Python MAD Analysis Tools',
      url='https://bitbucket.org/monodera/pyezmad',
      author='Masato Onodera',
      author_email='monodera@phys.ethz.ch',
      # license='BSD',
      packages=['pyezmad'],
      package_data={'pyezmad': ['database/emission_lines.dat']},
      zip_safe=False,

      # install_requires=['numpy', 'scipy', 'matplotlib', 'astropy', 'lmfit']
      )
