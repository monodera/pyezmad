
Introduction
============


Code
----

The code is hosted in a private repository on
`Bitbucket <https://bitbucket.org/monodera/pyezmad>`_.
If you want to use it please let me know,
I will add you as a member of the repository.


Install
-------
``pyezmad`` requres the following packages and their dependencies.

* `NumPy <http://numpy.org>`_
* `SciPy <http://scipy.org>`_
* `Matplotlib <http://matplotlib.org>`_
* `Astropy <http://astropy.org>`_
* `lmfit <http://cars9.uchicago.edu/software/python/lmfit/>`_
* `mpdaf CoreLib <http://urania1.univ-lyon1.fr/mpdaf/chrome/site/DocCoreLib/index.html>`_

All of them can be installed with ``pip``, e.g., ``pip install numpy``.
If you don't have a root permission on the computer you are working,
you can install it with ``pip install --user numpy``
to your user specific directory.

Note that the resulting binary files included in these packages
could be imcompatible between different host computerss
(e.g, ``finvarra.ethz.ch`` and ``theia.ethz.ch``).
What I'm doing is to make a virtual environment for different host
and install the up-to-date libraries.  If you want to know more about this
option, please contact me.

In addition to the above dependencies,
``pyezmad`` requires pPXF and 2D Voronoi binning codes developed by
Michele Cappellari.  You can download the Python version of them
from `his code repository <http://www-astro.physics.ox.ac.uk/~mxc/software/>`_.
You need to put the files into directories where your Python can import.
Since third-parties are not allowed to re-distribute his programs,
they are not included in the ``pyezmad`` repository.

Once the dependencies are installed, you can install ``pyezmad`` as follows. ::

  git clone git@bitbucket.org:monodera/pyezmad.git
  cd pyezmad
  python ./setup.py install


If you failed to download the code by a permission issue, please make sure to generate and register your SSH key to bitbucket
(see `the instruction <https://confluence.atlassian.com/bitbucket/add-an-ssh-key-to-an-account-302811853.html>`_).

You can check if ``pyezmad`` is installed at a proper location. ::

  import pyezmad
  print(pyezmad.__version__)



(Optional) If you are lazy to install anything by yourself, you can try my Python setup (assuming you use ``bash`` as login shell). ::

    # on finvarra
    . /net/astrogate/export/astro/shared/MUSE/user/monodera/python_venv_finvarra/bin/activate
    
    # on theia
    . /net/astrogate/export/astro/shared/MUSE/user/monodera/python_venv_theia/bin/activate




Known issues (mainly excuses)
-----------------------------

* The code is optimized for my data analysis at this momemnt, i.e. NGC 4980.
  Your suggestions and contributions are greatly appreciated!
* No error handling
* Inconsistent naming scheme
* Inconsistent input/output formats (e.g., fits or hdu object, etc.)
* Documentation is crappy
* Crappy English in documentation
* Still subject to be a major structural change
