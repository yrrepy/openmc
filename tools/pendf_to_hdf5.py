#!/usr/bin/env python
"""Preprocess a directory of PENDF files into an OpenMC PENDF HDF5 library.

Thin command-line wrapper around
:meth:`openmc.data.PendfLibrary.from_endf_directory`.
"""

import argparse
from pathlib import Path

from openmc.data import PendfLibrary

parser = argparse.ArgumentParser(description='Convert a directory of PENDF files into a PENDF HDF5 library.')
parser.add_argument('pendf_dir',        type=Path,                    help='Directory of PENDF files (uses _manifest.tsv if present)')
parser.add_argument('out',              type=Path,                    help='Output .h5 file')
parser.add_argument('--library',        type=str,   default=None,     help='Name of the source data library (e.g. TENDL-2017)')
parser.add_argument('--temperature',    type=float, default=None,     help='Library temperature in K; each file must agree within 0.1 K')
parser.add_argument('--keep-extra-mts', action='store_true',          help='Retain non-activation MF=3 reactions (heating, particle production, ...)')
parser.add_argument('--mapping',        type=str,   default='none',   help="Product-mapping mode: 'none' (only 'none' implemented)")
parser.add_argument('--decay-file',     type=Path,  default=None,     help='Decay data for product mapping (when --mapping is not none)')
args = parser.parse_args()

lib = PendfLibrary.from_endf_directory(
    args.pendf_dir, args.out, library=args.library, temperature=args.temperature,
    keep_extra_mts=args.keep_extra_mts, mapping=args.mapping,
    decay_file=args.decay_file)

print(f"Wrote {len(lib.nuclides)} nuclides to {args.out} "
      f"({lib.library}, {lib.temperature} K).")
