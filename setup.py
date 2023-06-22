#!/usr/bin/env python

import glob
import sys
import numpy as np
import os

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

hdf5_dir = os.environ.get('HDF5_DIR') or os.environ.get('HDF5_HOME')

ext_modules = cythonize([
    Extension("openmc.data.data_wrapper", ["openmc/data/data_wrapper.pyx", "src/material.cpp","src/nuclide.cpp",
              "src/message_passing.cpp","src/settings.cpp","src/error.cpp","src/distribution.cpp","src/thermal.cpp",
              "src/simulation.cpp","src/event.cpp","src/timer.cpp","src/photon.cpp","src/physics_mg.cpp","src/particle.cpp",
              "src/particle_restart.cpp","src/tallies/tally.cpp","src/tallies/filter_delayedgroup.cpp","src/tallies/filter.cpp",
              "src/tallies/filter_universe.cpp","src/tallies/filter_mu.cpp","src/tallies/filter_polar.cpp","src/tallies/filter_cellfrom.cpp",
              "src/tallies/filter_cell.cpp","src/tallies/filter_energy.cpp","src/tallies/filter_particle.cpp","src/tallies/filter_time.cpp",
              "src/tallies/filter_energyfunc.cpp","src/tallies/filter_meshsurface.cpp","src/tallies/filter_mesh.cpp","src/tallies/filter_legendre.cpp",
              "src/tallies/filter_zernike.cpp","src/tallies/filter_azimuthal.cpp","src/tallies/filter_collision.cpp","src/cross_sections.cpp",
              "src/mgxs_interface.cpp","src/finalize.cpp","src/endf.cpp","src/tallies/filter_cellborn.cpp","src/geometry_aux.cpp","src/surface.cpp",
              "src/boundary_condition.cpp","src/tallies/derivative.cpp","src/universe.cpp","src/tallies/filter_sptl_legendre.cpp","src/hdf5_interface.cpp"],
              language="c++",
              include_dirs=[np.get_include(), 'include', '/usr/include/hdf5/serial','include/openmc']
              library_dirs=[hdf5_dir],  # And this line (change to your hdf5 lib dir)
              extra_compile_args=["-std=c++11", "-O3", "-w"],  # Add necessary compiler flags here
              extra_link_args=["-lhdf5"]  # Add necessary linker flags here
              ), 
    "openmc/data/*.pyx"
])

# Determine shared library suffix
if sys.platform == 'darwin':
    suffix = 'dylib'
else:
    suffix = 'so'

# Get version information from __init__.py. This is ugly, but more reliable than
# using an import.
with open('openmc/__init__.py', 'r') as f:
    version = f.readlines()[-1].split()[-1].strip("'")

kwargs = {
    'name': 'openmc',
    'version': version,
    'packages': find_packages(exclude=['tests*']),
    'scripts': glob.glob('scripts/openmc-*'),
    'ext_modules': ext_modules,

    # Data files and libraries
    'package_data': {
        'openmc.lib': ['libopenmc.{}'.format(suffix)],
        'openmc.data': ['mass_1.mas20.txt', 'BREMX.DAT', 'half_life.json', '*.h5'],
        'openmc.data.effective_dose': ['*.txt']
    },

    # Metadata
    'author': 'The OpenMC Development Team',
    'author_email': 'openmc@anl.gov',
    'description': 'OpenMC',
    'url': 'https://openmc.org',
    'download_url': 'https://github.com/openmc-dev/openmc/releases',
    'project_urls': {
        'Issue Tracker': 'https://github.com/openmc-dev/openmc/issues',
        'Documentation': 'https://docs.openmc.org',
        'Source Code': 'https://github.com/openmc-dev/openmc',
    },
    'classifiers': [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering'
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],

    # Dependencies
    'python_requires': '>=3.7',
    'install_requires': [
        'numpy>=1.9', 'h5py', 'scipy', 'ipython', 'matplotlib',
        'pandas', 'lxml', 'uncertainties'
    ],
    'extras_require': {
        'depletion-mpi': ['mpi4py'],
        'docs': ['sphinx', 'sphinxcontrib-katex', 'sphinx-numfig', 'jupyter',
                 'sphinxcontrib-svg2pdfconverter', 'sphinx-rtd-theme'],
        'test': ['pytest', 'pytest-cov', 'colorama'],
        'vtk': ['vtk'],
    }
}

setup(**kwargs)
