from setuptools import setup, Extension, find_packages
import numpy

# Define the extension module located in the subdirectory
c_module = Extension(
    name='ehrlich.operations._find_closest',  # Full Python package path for the C extension
    sources=['ehrlich/operations/find_closest.c'],  # Path to your C file
    include_dirs=[numpy.get_include()],  # Include the NumPy headers
)

# Setup configuration
setup(
    name='ehrlich',
    version='1.0',
    description='Tool for comparing protein surface parameters',
    packages=find_packages(),
    ext_modules=[c_module],
    install_requires=['numpy'],
)