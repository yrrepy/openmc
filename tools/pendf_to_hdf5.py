#!/usr/bin/env python
"""Preprocess a directory of PENDF files into an OpenMC PENDF HDF5 library.

Thin command-line wrapper around
:meth:`openmc.data.PendfLibrary.from_endf_directory`. The library is written
source-faithful (raw MF=10 IZAP/LFS/QM/QI/ELFS attributes only); the
isomer<->LFS product mapping is carried on the depletion chain, built separately
with ``tools/add_pendf_isomeric_branching_to_chain.py``.

A reaction whose MF=10 section has no MF=3 sibling is stored with its total
summed from its own MF=10 partials (stamped ``total_source='sum-mf10'``, counted
in the ``mf10_only_totals`` root attribute); ``--no-mf10-only-totals`` drops that
class instead, as builds before the feature did. MT=5 and MT=18 are never stored
this way.

The ``--identity-only`` mode does not rebuild: it opens an existing ``out`` h5
in place and (re)writes only its ``source_identity`` root attr from the tapes in
``pendf_dir`` -- a cheap restamp of a library built before the attr existed.
"""

import argparse
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np

from openmc.data import PendfLibrary
from openmc.data.pendf import tape_identity

def main():
    parser = argparse.ArgumentParser(description='Convert a directory of PENDF files into a PENDF HDF5 library.')
    parser.add_argument('pendf_dir',             type=Path,                                        help='Directory of PENDF files (uses _manifest.tsv if present)')
    parser.add_argument('out',                   type=Path,                                        help='Output .h5 file')
    parser.add_argument('--library',             type=str,                default=None,            help='Name of the source data library (e.g. TENDL-2017)')
    parser.add_argument('--temperature',         type=float,              default=None,            help='Library temperature in K; each file must agree within 0.1 K')
    parser.add_argument('--keep-extra-mts',      action='store_true',                              help='Retain non-activation MF=3 reactions (heating, particle production, ...)')
    parser.add_argument('--identity-only',       action='store_true',                              help='Restamp only: write source_identity (from the tapes) into an existing --out h5, no rebuild')
    parser.add_argument('--log-file',            type=Path,               default=None,            help='Write a warning-summary log (counts + top-10 MF=10/MF=3 offenders) here')
    parser.add_argument('--no-mf10-only-totals', action='store_false',    dest='mf10_only_totals', help='Drop MF=10 sections that have no MF=3 sibling instead of storing them with a total summed from their partials (MT=5/MT=18 are dropped either way)')
    args = parser.parse_args()

    # Restamp mode: touch only the source_identity root attr of an existing h5.
    if args.identity_only:
        identity = tape_identity(args.pendf_dir)
        with h5py.File(args.out, 'r+') as h5:
            if identity is None:
                print(f"No tape identity derivable from {args.pendf_dir}; "
                      f"{args.out} left unchanged.")
            else:
                h5.attrs['source_identity'] = np.bytes_(identity)
                print(f"Stamped source_identity={identity!r} into {args.out}.")
        return

    # Capture build warnings only when a log is requested; otherwise the hook is
    # never installed so default behaviour is unchanged.
    if args.log_file is not None:
        from pendf_warning_log import WarningCapture, write_warning_log
        capture = WarningCapture()
    else:
        capture = None

    with (capture if capture is not None else nullcontext()):
        lib = PendfLibrary.from_endf_directory(
            args.pendf_dir, args.out, library=args.library, temperature=args.temperature,
            keep_extra_mts=args.keep_extra_mts, mf10_only_totals=args.mf10_only_totals)

    print(f"Wrote {len(lib.nuclides)} nuclides to {args.out} "
          f"({lib.library}, {lib.temperature} K).")

    if capture is not None:
        write_warning_log(args.log_file, capture)


if __name__ == '__main__':
    main()
