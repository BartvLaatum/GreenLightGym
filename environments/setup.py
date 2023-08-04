from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(["GreenLightCy.pyx"],
                            compiler_directives={'language_level' : "3"},
                            annotate=True),
                include_dirs=[np.get_include()],
)