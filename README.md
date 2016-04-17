mirnylib
========

Installation
------------
Use pip. There is also an install script (Linux only).

Get the latest:

`$ pip install https://bitbucket.org/mirnylab/mirnylib/get/tip.tar.gz`

Or clone the repo and install in development mode:

```sh
$ hg clone https://bitbucket.org/mirnylab/mirnylib
$ cd mirnylib
$ pip install -e .
```

Installation Requirements
-------------------------

Python 2.7/3.3+

### Dependencies

For the h5py package to install properly, you need to have the shared library and development headers for HDF5 1.8.4 or newer installed (`libhdf5-dev` or similar). See the [h5py docs](http://docs.h5py.org/en/latest/build.html) for more information.

Required:

- numpy (1.6+)
- scipy
- matplotlib
- h5py
- biopython
- joblib (0.6.3+)
- bx-python (preferably from the [bitbucket repo](https://bitbucket.org/james_taylor/bx-python/wiki/Home) by james_taylor)


### Build dependencies

- Cython (0.16+) to build Cython extensions from .pyx source (C files no longer provided)

### Notes

We highly recommend using the [conda](http://conda.pydata.org/miniconda.html) package/environment manager if you have trouble building the core scientific Python packages.

`$ conda install numpy scipy matplotlib h5py cython`

### See also

[hiclib](https://bitbucket.org/mirnylab/hiclib)