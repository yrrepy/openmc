#!/usr/bin/env python
"""Preprocess a directory of PENDF files into an OpenMC PENDF HDF5 library.

Thin command-line wrapper around
:meth:`openmc.data.PendfLibrary.from_endf_directory`.
"""

import argparse
from pathlib import Path

from openmc.data import PendfLibrary
from openmc.data.isomeric import ELIS_ATOL, ELIS_RTOL

parser = argparse.ArgumentParser(description='Convert a directory of PENDF files into a PENDF HDF5 library.')
parser.add_argument('pendf_dir',        type=Path,                      help='Directory of PENDF files (uses _manifest.tsv if present)')
parser.add_argument('out',              type=Path,                      help='Output .h5 file')
parser.add_argument('--library',        type=str,   default=None,       help='Name of the source data library (e.g. TENDL-2017)')
parser.add_argument('--temperature',    type=float, default=None,       help='Library temperature in K; each file must agree within 0.1 K')
parser.add_argument('--keep-extra-mts', action='store_true',            help='Retain non-activation MF=3 reactions (heating, particle production, ...)')
parser.add_argument('--mapping',        type=str,   default='none',     help="Product-mapping mode: 'none', 'elis', or 'lfs_order'")
parser.add_argument('--decay-file',     type=Path,  default=None,       help='Decay data (dir or file) for product mapping; required if --mapping is not none')
parser.add_argument('--elis-rtol',      type=float, default=ELIS_RTOL,  help='Relative ELFS/ELIS match tolerance for --mapping elis')
parser.add_argument('--elis-atol',      type=float, default=ELIS_ATOL,  help='Absolute ELFS/ELIS match tolerance in eV for --mapping elis')
args = parser.parse_args()

lib = PendfLibrary.from_endf_directory(
    args.pendf_dir, args.out, library=args.library, temperature=args.temperature,
    keep_extra_mts=args.keep_extra_mts, mapping=args.mapping,
    decay_file=args.decay_file, elis_rtol=args.elis_rtol, elis_atol=args.elis_atol)

print(f"Wrote {len(lib.nuclides)} nuclides to {args.out} "
      f"({lib.library}, {lib.temperature} K).")
