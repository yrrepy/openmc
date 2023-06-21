#!/usr/bin/env python

import glob
import sys
import numpy as np

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

ext_modules = cythonize([
    Extension("openmc.data.data_wrapper", ["openmc/data/data_wrapper.pyx", "src/material.cpp","src/nuclide.cpp","src/message_passing.cpp","src/settings.cpp","src/error.cpp","src/distribution.cpp","src/thermal.cpp","src/simulation.cpp","src/event.cpp","src/timer.cpp","src/photon.cpp","src/physics_mg.cpp","src/particle.cpp","src/particle_restart.cpp","src/tallies/tally.cpp","src/tallies/filter_delayedgroup.cpp","src/tallies/filter.cpp","src/tallies/filter_universe.cpp","src/tallies/filter_mu.cpp"],
              language="c++",
              include_dirs=[np.get_include(), 'include', '/usr/include/hdf5/serial','include/openmc']),
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
