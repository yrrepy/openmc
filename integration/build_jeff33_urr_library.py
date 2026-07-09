#!/usr/bin/env python
"""Build the JEFF-3.3 URR self-shielding PENDF libraries (pointwise + grouped).

Runs the MT=153-aware PENDF parser (:meth:`PendfLibrary.from_endf_directory`)
over ONLY the flagged URR nuclides in the JEFF-3.3 point library, producing a
compact pointwise HDF5 that carries ``<nuclide>/urr`` probability tables, then
group-bins it onto CCFE-709 (via ``tools/pendf_group_bin.py``) so the grouped
library keeps ``/urr`` verbatim. Both libraries feed the URR material-dilution
self-shielding correction.

Run from the clone root so the local (modified) ``openmc`` shadows any
site-installed copy::

    cd OpenMC_pendf-urr
    python integration/build_jeff33_urr_library.py
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

# Ensure the local (modified) openmc in the clone root wins over any
# site-installed copy, regardless of the directory this script is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import openmc.data
from openmc.data.pendf import PendfLibrary
from openmc.mgxs import GROUP_STRUCTURES
from tools.pendf_group_bin import bin_pendf_library, chain_relevant_mts

# Flagged URR nuclides (all confirmed to carry MF=2 MT=152/153 in JEFF-3.3).
DEFAULT_FLAGGED = [
    'W180', 'W182', 'W183', 'W184', 'W186', 'Ta181', 'Re185', 'Re187',
    'Hf174', 'Hf176', 'Hf177', 'Hf178', 'Hf179', 'Hf180',
    'Os186', 'Os187', 'Os188', 'Os189', 'Os190', 'Os192',
    'U235', 'U238', 'Pu239', 'Pu240',
]


def _jeff33_filename(gnds):
    """Map a GNDS name to its JEFF-3.3 point-library filename.

    ``'W182'`` -> ``'74-W-182g.jeff33.pendf'`` (ground state; the flagged set is
    all ground-state nuclides).
    """
    z, a, m = openmc.data.zam(gnds)
    if m != 0:
        raise ValueError(f"{gnds}: flagged set is ground-state only (m={m}).")
    sym = openmc.data.ATOMIC_SYMBOL[z]
    return f'{z}-{sym}-{a}g.jeff33.pendf'


def build(pendf_dir, out_pointwise, out_grouped, groups, nuclides, library):
    """Build the pointwise then grouped JEFF-3.3 URR libraries."""
    pendf_dir = Path(pendf_dir)
    out_pointwise = Path(out_pointwise)
    out_grouped = Path(out_grouped)
    out_pointwise.parent.mkdir(parents=True, exist_ok=True)
    out_grouped.parent.mkdir(parents=True, exist_ok=True)

    if groups not in GROUP_STRUCTURES:
        raise SystemExit(f"unknown group structure {groups!r}; available: "
                         f"{', '.join(sorted(GROUP_STRUCTURES))}")
    edges = np.asarray(GROUP_STRUCTURES[groups], dtype=np.float64)

    # Symlink only the flagged files into a temp dir so from_endf_directory
    # (which scans a whole directory) processes exactly the flagged subset.
    tmpdir = Path(tempfile.mkdtemp(prefix='jeff33_urr_'))
    try:
        missing = []
        for gnds in nuclides:
            src = pendf_dir / _jeff33_filename(gnds)
            if not src.is_file():
                missing.append(str(src))
                continue
            os.symlink(src, tmpdir / src.name)
        if missing:
            raise SystemExit("missing JEFF-3.3 PENDF files:\n  "
                             + "\n  ".join(missing))

        print(f"Parsing {len(nuclides)} flagged nuclides from {pendf_dir} ...")
        lib = PendfLibrary.from_endf_directory(
            tmpdir, out_pointwise, library=library)
        n_urr = sum(lib.has_ptables(n) for n in lib.nuclides)
        print(f"  pointwise: {out_pointwise} "
              f"({len(lib.nuclides)} nuclides, {n_urr} with /urr, "
              f"{lib.temperature} K)")
        lib.close()
    finally:
        for p in tmpdir.iterdir():
            p.unlink()
        tmpdir.rmdir()

    # The URR material-dilution correction needs each diluter's group TOTAL
    # sigma_t,g to build the sigma_0 background (mat_ssf: xs_g(j, 1) on a grouped
    # library, and the resonant nuclide's smooth total at the URR nodes for
    # factor-form tables). MT=1 is not a depletion-activation reaction, so the
    # default group-bin MT set (chain_relevant_mts) drops it; add it explicitly
    # so the grouped URR library can supply the total. (An extra ~1 dataset per
    # nuclide; the flag-off collapse never requests MT=1 so it is unaffected.)
    mts = chain_relevant_mts() | {1}
    print(f"Group-binning onto {groups} ({len(edges) - 1} groups) "
          f"with MT=1 (total) included for URR sigma_0 ...")
    bin_pendf_library(out_pointwise, out_grouped, edges, mts=mts,
                      nuclides=nuclides)
    print(f"  grouped: {out_grouped}")


def main():
    default_data = Path('/home/perry/Projects/OMC_Development/PENDF/data')
    parser = argparse.ArgumentParser(description='Build the JEFF-3.3 URR PENDF libraries (pointwise + grouped).')
    parser.add_argument('--pendf-dir',     type=Path, default=Path('/home/perry/NukeData/Activation/PENDF/Point_JEFF33'), help='Directory of JEFF-3.3 point PENDF files')
    parser.add_argument('--out-pointwise', type=Path, default=default_data / 'jeff33_urr_pendf_294K.h5',                  help='Output pointwise PENDF HDF5 (carries /urr)')
    parser.add_argument('--out-grouped',   type=Path, default=default_data / 'jeff33_urr_pendf_294K_ccfe709.h5',         help='Output grouped PENDF HDF5 (CCFE-709, /urr verbatim)')
    parser.add_argument('--groups',        type=str,  default='CCFE-709',                                                help='Named group structure for the grouped library')
    parser.add_argument('--library',       type=str,  default='jeff-3.3',                                                help='Library name recorded in the HDF5 root attrs')
    parser.add_argument('--nuclides',      type=str,  default=None, nargs='+',                                           help='Override the flagged nuclide list (default: 24 flagged)')
    args = parser.parse_args()

    nuclides = args.nuclides if args.nuclides is not None else DEFAULT_FLAGGED
    build(args.pendf_dir, args.out_pointwise, args.out_grouped,
          args.groups, nuclides, args.library)


if __name__ == '__main__':
    main()
