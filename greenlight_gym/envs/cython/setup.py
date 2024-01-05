from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(["greenlight_cy.pyx"],
                            compiler_directives={'language_level' : "3"},
                            annotate=False),
                include_dirs=[np.get_include()],
)