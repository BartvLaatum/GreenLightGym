from setuptools import setup, Extension, Command
import numpy as np
import os
from Cython.Build import cythonize
from setuptools.command.build_ext import build_ext as _build_ext

# Define the path for the Cython module
cython_module_path = "greenlight_gym/envs/cython/greenlight_cy.pyx"

# Define the extension module
extensions = [
    Extension(
        "greenlight_gym.envs.cython.greenlight_cy",  # Full module path
        [cython_module_path],
        include_dirs=[np.get_include()],
    )
]

# Custom build_ext class to change the output directory
class build_ext(_build_ext):
    def build_extensions(self):
        # Ensure the output directory exists
        os.makedirs(self.build_lib, exist_ok=True)
        # Call the original build_extensions method
        super().build_extensions()

# Custom command to build only the Cython extensions
class BuildCythonOnlyCommand(Command):
    description = "Build only the Cython extensions"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # Build the Cython extensions
        self.run_command('build_ext')

# Function to read the basic requirements file
def read_requirements():
    with open('requirements.txt') as req_file:
        return req_file.read().splitlines()

# Setup function
setup(
    name="greenlight_gym",
    version="0.1",
    description="A custom gym environment with Cython optimizations",
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"},
        annotate=False,
    ),
    include_dirs=[np.get_include()],
    cmdclass={
        'build_ext': build_ext,          # Use the custom build_ext class
        'build_cython_only': BuildCythonOnlyCommand,  # Command to build only Cython extensions
    },
    options={
        'build_ext': {
            'build_lib': 'greenlight_gym/envs/cython'  # Specify the output directory
        }
    },
    install_requires=read_requirements(),  # Basic dependencies
)
