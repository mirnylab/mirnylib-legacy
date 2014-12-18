from setuptools import setup, Extension
import os
import numpy

ext_modules = []
cmdclass = {}

try:
    from Cython.Distutils import build_ext
    ext_modules += [
        Extension("mirnylib.numutils_new", [ "mirnylib/numutils_new.pyx" ],
            language = "c++",
                  include_dirs=[numpy.get_include()]),
    ]
    cmdclass.update( {'build_ext': build_ext} )
except ImportError:
    if not os.path.isfile('mirnylib/numutils_new.c'):
        raise RuntimeError("Cython is required to build extension modules for mirnylib.")
    ext_modules += [
        Extension("mirnylib.numutils_new", [ "mirnylib/numutils_new.c" ],
                  include_dirs=[numpy.get_include()]),
    ]
        

setup(
    name='mirnylib',
    url='https://bitbucket.org/mirnylab/mirnylib/',
    description=('Libraries, shared between different mirnylab projects.'),
    packages=['mirnylib',"mirnylib/h5dictUtils"],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=[
        'biopython',
        'pysam',
        'joblib>=0.6.3',
        'h5py',
        'bx-python',
    ],
    dependency_links=[
        'https://bitbucket.org/james_taylor/bx-python/get/tip.tar.bz2'
    ],
)
