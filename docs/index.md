# Welcome to Python MAD Analysis Tools

Python MAD Analysis Tools (`pyezmad`) is a tool intended 
to help analyse MAD (MUSE Atlas for Disks) data 
without going mad.



## Code

The code is hosted in a private repository on 
[Bitbucket](https://bitbucket.org/monodera/pyezmad).
If you want to use it please let me know, 
I will add you as a member of the repository.


## Install

`pyezmad` requres the following packages and their dependencies. 

* [NumPy](http://numpy.org)
* [SciPy](http://scipy.org)
* [Matplotlib](http://matplotlib.org)
* [Astropy](http://astropy.org)
* [lmfit](http://cars9.uchicago.edu/software/python/lmfit/)
* [mpdaf CoreLib](http://urania1.univ-lyon1.fr/mpdaf/chrome/site/DocCoreLib/index.html)

All of them can be installed with `pip`, e.g., `pip install numpy`.
If you don't have a root permission on the computer you are working, 
you can install it with `pip install --user numpy` 
to your user specific directory. 

Note that the resulting binary files included in these packages 
could be imcompatible between different host computerss 
(e.g, `finvarra` and `theia`). 
What I'm doing is to make a virtual environment for different host 
and install the up-to-date libraries.  If you want to know more about this
option, please contact me. 

In addition to the above dependencies, 
`pyezmad` requires pPXF and 2D Voronoi binning codes developed by 
Michele Cappellari.  You can download the Python version of them 
from [his code repository](http://www-astro.physics.ox.ac.uk/~mxc/software/).
You need to put the files into directories where your Python can import.
Since third-parties are not allowed to re-distribute his programs, 
they are not included in the `pyezmad` repository.

Once the dependencies are installed, you can install `pyezmad` as follows.

```shell
git clone git@bitbucket.org:monodera/pyezmad.git
cd pyezmad
python ./setup.py install
```

You can check if `pyezmad` is installed at a proper location. 

```python
import pyezmad
print(pyezmad.__version__)
```


## Known issues (mainly excuses)

* The code is optimized for my data analysis at this momemnt, i.e. NGC 4980. Your suggestions and contributions are greatly appreciated!
* No error handling
* Inconsistent naming scheme
* Inconsistent input/output formats
* Documentation is crappy
* Crappy English in documentation
* Still subject to be a major structural change

