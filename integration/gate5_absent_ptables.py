#!/usr/bin/env python
"""Gate 5 - URR self-shielding with an absent probability table.

Copies the grouped JEFF-3.3 URR library, deletes ``W186/urr`` from the copy, and
runs the gate-4 collapse (flag on) against it. A flagged nuclide with no ``/urr``
group must warn exactly once (naming it), be left at infinite dilution (its rows
identical to the flag-off collapse), while the other W isotopes still shield and
the run completes.

Run from the clone root::

    python integration/gate5_absent_ptables.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

# Local (modified) openmc must win; also expose the sibling gate-4 helpers.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

import h5py
import numpy as np

from openmc.data.pendf_grouped import GroupedPendfLibrary
from openmc.deplete.microxs import MicroXS
from gate4_end_to_end import (parse_hfr, rebin_integral, one_group,
                              NAT_W, W_ISOTOPES, PRODUCTS, REACTIONS)

ABSENT = 'W186'   # delete this nuclide's /urr from the working copy


def main():
    root = Path('/home/perry/Projects/OMC_Development/PENDF')
    parser = argparse.ArgumentParser(description='Gate 5 - absent-ptable URR self-shielding path.')
    parser.add_argument('--grouped', type=Path, default=root / 'data/jeff33_urr_pendf_294K_ccfe709.h5', help='Grouped JEFF-3.3 URR library (CCFE-709)')
    parser.add_argument('--hfr',     type=Path, default=root / 'URR/616_HFR-low.txt',                   help='LLNL-616 HFR reference spectrum')
    args = parser.parse_args()

    # Copy the library and strip W186/urr from the copy.
    scratch = Path(tempfile.mkdtemp(prefix='gate5_'))
    work = scratch / 'grouped_no_W186_urr.h5'
    shutil.copy(args.grouped, work)
    with h5py.File(work, 'r+') as f:
        assert f'{ABSENT}/urr' in f, f'{ABSENT}/urr not present to delete'
        del f[f'{ABSENT}/urr']
    print(f'Deleted {ABSENT}/urr from working copy: {work}')

    glib = GroupedPendfLibrary(work)
    assert not glib.has_ptables(ABSENT), f'{ABSENT} still reports ptables'
    edges = np.asarray(glib.group_edges, dtype=float)

    src_edges, src_flux = parse_hfr(args.hfr)
    flux = rebin_integral(src_edges, src_flux, edges)
    flux_norm = flux / flux.sum()

    nucs = W_ISOTOPES + PRODUCTS
    kw = dict(energies=edges, multigroup_flux=flux_norm, nuclides=nucs,
              reactions=REACTIONS, pendf_library=glib)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        off = MicroXS.from_multigroup_flux(**kw)

    completed = True
    err = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            on = MicroXS.from_multigroup_flux(
                urr_material_dilution=True, densities=NAT_W, **kw)
        except Exception as exc:                       # pragma: no cover
            completed = False
            err = exc

    glib.close()
    shutil.rmtree(scratch, ignore_errors=True)

    # --- checks ---
    ptable_warns = [str(w.message) for w in caught
                    if 'probability tables' in str(w.message)]
    w186_warns = [m for m in ptable_warns if ABSENT in m]

    check_warn = (len(ptable_warns) == 1 and len(w186_warns) == 1)

    if completed:
        w186_off = one_group(off, ABSENT, '(n,gamma)')
        w186_on = one_group(on, ABSENT, '(n,gamma)')
        check_identical = (w186_off == w186_on)
        # Other W isotopes must still shield (strictly lower with flag on).
        others = [n for n in W_ISOTOPES if n != ABSENT]
        shielded = {n: (one_group(on, n, '(n,gamma)')
                        < one_group(off, n, '(n,gamma)')) for n in others}
        check_others = all(shielded.values())
    else:
        check_identical = check_others = False
        shielded = {}

    all_ok = check_warn and check_identical and check_others and completed

    print('=' * 78)
    print('Gate 5 - absent-ptable URR self-shielding path')
    print('=' * 78)
    print(f'run completed without error:                    '
          f'{"PASS" if completed else "FAIL"}'
          + (f'  ({type(err).__name__}: {err})' if err else ''))
    print(f'exactly one absent-ptable warning naming {ABSENT}:  '
          f'{"PASS" if check_warn else "FAIL"}  '
          f'(got {len(ptable_warns)} ptable warning(s))')
    for m in ptable_warns:
        print(f'    warning: {m}')
    if completed:
        print(f'{ABSENT} (n,gamma) identical flag-on vs flag-off:    '
              f'{"PASS" if check_identical else "FAIL"}  '
              f'(off={w186_off:.6e}, on={w186_on:.6e})')
        print(f'other W isotopes still shielded (on < off):     '
              f'{"PASS" if check_others else "FAIL"}  {shielded}')
    print('=' * 78)
    print(f'GATE 5: {"PASS" if all_ok else "FAIL"}')
    print('=' * 78)
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
