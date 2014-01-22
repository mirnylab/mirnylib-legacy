from distutils.core import setup
from distutils.extension import Extension
import os
import numpy

cmdclass = {}
ext_modules = []

try:
    from Cython.Distutils import build_ext
    ext_modules += [
        Extension("mirnylib.numutils_new", [ "mirnylib/numutils_new.pyx" ],
                  include_dirs=[numpy.get_include()]),
    ]
    cmdclass.update( {'build_ext': build_ext} )

except ImportError:
    if not os.path.isfile('mirnylib/numutils_new.c'): raise
    ext_modules += [
        Extension("mirnylib.numutils_new", [ "mirnylib/numutils_new.c" ],
                  include_dirs=[numpy.get_include()]),
    ]


setup(
    name='mirnylib',
    url='https://bitbucket.org/mirnylab/mirnylib/',
    description=('Libraries, shared between different mirnylab projects.'),
    packages=['mirnylib'],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)