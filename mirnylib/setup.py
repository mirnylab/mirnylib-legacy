from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("numutils_new", ["numutils_new.pyx"])]


setup(
  name = 'New numutils',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
