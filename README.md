mirnylib
========

Installation
------------
The easiest way is to use pip, but see notes below:

`$pip install https://bitbucket.org/mirnylab/mirnylib/get/tip.tar.gz`

Installation Requirements
-------------------------

Python 2.6 or 2.7

It is probably a good idea to upgrade to the latest version of setuptools and pip. Follow the PyPA's [guide](http://python-packaging-user-guide.readthedocs.org/en/latest/).

### Python dependencies

Required:

Getting the basic Scientific Python stack (numpy/scipy/matplotlib) can be trickier on some platforms than others. For more details, see the [instructions on scipy.org](http://www.scipy.org/install.html). You should already have these dependencies installed and running correctly before attempting to install this package.

- numpy (1.6+)
- scipy
- matplotlib


The following dependencies are automatically installed by setuptools if missing:

- biopython
- joblib (0.6.3+)
- bx-python (preferably from the [bitbucket repo](https://bitbucket.org/james_taylor/bx-python/wiki/Home) by james_taylor)
- h5py (see note below)

### Non-Python dependencies
- For the h5py package to install properly, you need to have the shared library and development headers for HDF5 1.8.4 or newer installed (`libhdf5-dev` or similar). See the [h5py docs](http://docs.h5py.org/en/latest/build.html) for more information.

### Build dependencies
- Cython (0.16+) to build Cython extensions from .pyx source (recommended but optional; C files are provided)

- A C compiler to compile the C extensions