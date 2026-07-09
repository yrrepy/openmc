#!/usr/bin/env python
"""Preprocess a directory of PENDF files into an OpenMC PENDF HDF5 library.

Thin command-line wrapper around
:meth:`openmc.data.PendfLibrary.from_endf_directory`. The library is written
source-faithful (raw MF=10 IZAP/LFS/QM/QI/ELFS attributes only); the
isomer<->LFS product mapping is carried on the depletion chain, built separately
with ``tools/add_pendf_isomeric_branching_to_chain.py``.
"""

import argparse
from pathlib import Path

from openmc.data import PendfLibrary

def main():
    parser = argparse.ArgumentParser(description='Convert a directory of PENDF files into a PENDF HDF5 library.')
    parser.add_argument('pendf_dir',        type=Path,                      help='Directory of PENDF files (uses _manifest.tsv if present)')
    parser.add_argument('out',              type=Path,                      help='Output .h5 file')
    parser.add_argument('--library',        type=str,   default=None,       help='Name of the source data library (e.g. TENDL-2017)')
    parser.add_argument('--temperature',    type=float, default=None,       help='Library temperature in K; each file must agree within 0.1 K')
    parser.add_argument('--keep-extra-mts', action='store_true',            help='Retain non-activation MF=3 reactions (heating, particle production, ...)')
    args = parser.parse_args()

    lib = PendfLibrary.from_endf_directory(
        args.pendf_dir, args.out, library=args.library, temperature=args.temperature,
        keep_extra_mts=args.keep_extra_mts)

    print(f"Wrote {len(lib.nuclides)} nuclides to {args.out} "
          f"({lib.library}, {lib.temperature} K).")


if __name__ == '__main__':
    main()
