from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
  name = 'f2v',
  ext_modules = cythonize("feat2vec_inner.pyx"),
  include_dirs=[numpy.get_include()]
)
