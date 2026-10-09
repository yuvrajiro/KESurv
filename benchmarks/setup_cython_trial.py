"""Build the optional benchmark extension with `python ... build_ext --inplace`."""

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(
        [Extension("_cython_trial", ["benchmarks/_cython_trial.pyx"],
                   include_dirs=[np.get_include()])],
        compiler_directives={"language_level": "3"},
    )
)
