#!/usr/bin/env python

from setuptools import setup

# Generate version.py
__version__ = None
with open('pyezmad/version.py') as f:
    exec(f.read())

setup(name='pyezmad',
      version=__version__,
      description='Python MAD Analysis Tools',
      url='https://bitbucket.org/monodera/pyezmad',
      author='Masato Onodera',
      author_email='monodera@phys.ethz.ch',
      license='MIT',
      packages=['pyezmad'],
      package_data={'pyezmad': ['database/emission_lines.dat']},
      zip_safe=False,
      install_requires=['numpy', 'scipy', 'astropy',
                        'astroquery', 'lmfit', 'mpdaf', 'tqdm',
                        'matplotlib', 'seaborn', 'pyneb'],
      )
