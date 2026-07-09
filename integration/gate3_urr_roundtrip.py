#!/usr/bin/env python
"""Gate 3 - URR probability-table round-trip.

Verifies that the ``/urr`` probability tables written into the JEFF-3.3 PENDF
libraries (pointwise and grouped) read back -- via the C2 ``ptables()``
accessor and ``ProbabilityTables.from_hdf5`` -- identically to a FRESH,
INDEPENDENT parse of MF=2 MT=153 from the source ENDF file. The reference table
is built here from scratch (energy verbatim; column 0 = cumsum of the raw
per-band probabilities; columns 1-5 verbatim), so a bug in the parser's
reshape/cumsum/attr handling shows up as a mismatch.

Checks, for W182, U238, Ta181, against BOTH libraries:
  * energy grid exactly equal,
  * table equal to ~1e-12 rtol (plus an exact-equality report),
  * attrs interpolation=2, inelastic=-1, absorption=-1, multiply_smooth=False.
Also checks that a nuclide absent from the library answers has_ptables()=False
and ptables()=None gracefully (no exception).

Run from the clone root::

    python integration/gate3_urr_roundtrip.py
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

# Local (modified) openmc must win over any site-installed copy.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import openmc.data
from openmc.data.endf import Evaluation, get_head_record, get_list_record
from openmc.data.pendf import PendfLibrary
from openmc.data.pendf_grouped import GroupedPendfLibrary

TEST_NUCLIDES = ['W182', 'U238', 'Ta181']
ABSENT_NUCLIDE = 'Xe135'  # not in the flagged-only URR library
RTOL = 1e-12

# The 24 flagged JEFF-3.3 URR nuclides and their band convention. Only these 5
# carry ABSOLUTE bands (MT=153 LSSF=0 -> multiply_smooth=False); the other 19
# are FACTOR-form (LSSF=1 -> multiply_smooth=True). Empirically established from
# the MT=153 LIST L1 field, cross-checked against the data (prob-weighted
# band-total ~1 iff factor). The parser must set this per nuclide.
FLAGGED_NUCLIDES = [
    'W180', 'W182', 'W183', 'W184', 'W186', 'Ta181', 'Re185', 'Re187',
    'Hf174', 'Hf176', 'Hf177', 'Hf178', 'Hf179', 'Hf180',
    'Os186', 'Os187', 'Os188', 'Os189', 'Os190', 'Os192',
    'U235', 'U238', 'Pu239', 'Pu240',
]
ABSOLUTE_NUCLIDES = {'W182', 'W183', 'W184', 'W186', 'Ta181'}


def _jeff33_filename(gnds):
    """GNDS name -> JEFF-3.3 point-library filename (ground state)."""
    z, a, m = openmc.data.zam(gnds)
    sym = openmc.data.ATOMIC_SYMBOL[z]
    return f'{z}-{sym}-{a}g.jeff33.pendf'


def reference_ptable(src_path):
    """Independently parse MF=2 MT=153 -> (energy, table, attrs dict).

    Mirrors the verified JEFF-3.3 layout but is written from scratch so it is a
    genuine oracle for the parser under test: column-major bands, column 0 the
    CUMSUM of the raw probability column.
    """
    ev = Evaluation(src_path)
    fo = io.StringIO(ev.section[2, 153])
    _za, _awr, _l1, _l2, _n1, nband = get_head_record(fo)
    # LIST L1 is the LSSF flag (verified): 0 -> absolute bands, 1 -> factor bands.
    (temp, _c2, lssf, _ll2, npl, nunr), values = get_list_record(fo)
    per_energy = 1 + 6 * nband
    assert npl == nunr * per_energy, (
        f"{src_path.name}: NPL={npl} != NUNR*({per_energy}) for NUNR={nunr}")
    block = np.asarray(values, dtype=np.float64).reshape(nunr, per_energy)
    energy = np.array(block[:, 0], dtype=np.float64)
    table = block[:, 1:].reshape(nunr, 6, nband).copy()
    table[:, 0, :] = np.cumsum(table[:, 0, :], axis=1)
    attrs = dict(interpolation=2, inelastic=-1, absorption=-1,
                 multiply_smooth=bool(lssf), temp=float(temp),
                 nunr=int(nunr), nband=int(nband))
    return energy, table, attrs


def _rel_dev(ref, got):
    """Max abs and max rel deviation between two equal-shape arrays."""
    diff = np.abs(ref - got)
    max_abs = float(diff.max()) if diff.size else 0.0
    nz = ref != 0.0
    max_rel = float((diff[nz] / np.abs(ref[nz])).max()) if nz.any() else 0.0
    return max_abs, max_rel


def check_one(kind, lib, nuc, ref_energy, ref_table, ref_attrs, results):
    """Compare a library's ptables(nuc) against the reference; record result."""
    ok = True
    msgs = []

    assert lib.has_ptables(nuc), f"{kind}: has_ptables({nuc}) is False"
    pt = lib.ptables(nuc)
    assert pt is not None, f"{kind}: ptables({nuc}) is None"

    # Energy grid: exact equality required.
    energy_exact = np.array_equal(ref_energy, np.asarray(pt.energy))
    if not energy_exact:
        ok = False
        msgs.append('energy grid not exactly equal')

    # Table: tight rtol + exact-equality report; capture deviations.
    got_table = np.asarray(pt.table)
    shape_ok = got_table.shape == ref_table.shape
    if not shape_ok:
        ok = False
        msgs.append(f'table shape {got_table.shape} != {ref_table.shape}')
        max_abs = max_rel = float('nan')
        table_exact = table_close = False
    else:
        table_exact = np.array_equal(ref_table, got_table)
        table_close = np.allclose(ref_table, got_table, rtol=RTOL, atol=0.0)
        max_abs, max_rel = _rel_dev(ref_table, got_table)
        if not table_close:
            ok = False
            msgs.append(f'table not within rtol={RTOL:.0e} '
                        f'(max_abs={max_abs:.3e}, max_rel={max_rel:.3e})')

    # Attributes.
    attr_checks = {
        'interpolation': (pt.interpolation, ref_attrs['interpolation']),
        'inelastic': (pt.inelastic_flag, ref_attrs['inelastic']),
        'absorption': (pt.absorption_flag, ref_attrs['absorption']),
        'multiply_smooth': (pt.multiply_smooth, ref_attrs['multiply_smooth']),
    }
    for k, (got, exp) in attr_checks.items():
        if got != exp:
            ok = False
            msgs.append(f'attr {k}={got!r} != {exp!r}')

    results.append(dict(kind=kind, nuc=nuc, ok=ok, energy_exact=energy_exact,
                        table_exact=table_exact, table_close=table_close,
                        max_abs=max_abs, max_rel=max_rel,
                        nunr=ref_attrs['nunr'], nband=ref_attrs['nband'],
                        msgs=msgs))


def main():
    default_data = Path('/home/perry/Projects/OMC_Development/PENDF/data')
    parser = argparse.ArgumentParser(description='Gate 3 - URR probability-table round-trip validator.')
    parser.add_argument('--pointwise', type=Path, default=default_data / 'jeff33_urr_pendf_294K.h5',          help='Pointwise PENDF HDF5 built by build_jeff33_urr_library.py')
    parser.add_argument('--grouped',   type=Path, default=default_data / 'jeff33_urr_pendf_294K_ccfe709.h5', help='Grouped PENDF HDF5 built by build_jeff33_urr_library.py')
    parser.add_argument('--pendf-dir', type=Path, default=Path('/home/perry/NukeData/Activation/PENDF/Point_JEFF33'), help='JEFF-3.3 point PENDF source directory (for the independent parse)')
    parser.add_argument('--nuclides',  type=str,  default=None, nargs='+',                                    help='Override the test nuclides (default: W182 U238 Ta181)')
    args = parser.parse_args()

    nuclides = args.nuclides if args.nuclides is not None else TEST_NUCLIDES

    references = {}
    for nuc in nuclides:
        src = args.pendf_dir / _jeff33_filename(nuc)
        references[nuc] = reference_ptable(src)

    results = []
    plib = PendfLibrary(args.pointwise)
    glib = GroupedPendfLibrary(args.grouped)
    try:
        for nuc in nuclides:
            ref_energy, ref_table, ref_attrs = references[nuc]
            check_one('pointwise', plib, nuc, ref_energy, ref_table,
                      ref_attrs, results)
            check_one('grouped', glib, nuc, ref_energy, ref_table,
                      ref_attrs, results)

        # Absent-nuclide graceful path (both libraries).
        absent_ok = True
        absent_msgs = []
        for kind, lib in (('pointwise', plib), ('grouped', glib)):
            if lib.has_ptables(ABSENT_NUCLIDE):
                absent_ok = False
                absent_msgs.append(f'{kind}: has_ptables({ABSENT_NUCLIDE}) True')
            if lib.ptables(ABSENT_NUCLIDE) is not None:
                absent_ok = False
                absent_msgs.append(f'{kind}: ptables({ABSENT_NUCLIDE}) not None')

        # Per-nuclide multiply_smooth (LSSF) split: exactly the 5 ABSOLUTE
        # nuclides carry multiply_smooth=False; the other 19 carry True. Checked
        # against BOTH libraries for all 24 flagged nuclides.
        ms_ok = True
        ms_msgs = []
        n_abs = n_fac = 0
        for kind, lib in (('pointwise', plib), ('grouped', glib)):
            for nuc in FLAGGED_NUCLIDES:
                pt = lib.ptables(nuc)
                if pt is None:
                    ms_ok = False
                    ms_msgs.append(f'{kind}: {nuc} has no ptables')
                    continue
                expected = nuc not in ABSOLUTE_NUCLIDES  # True == factor-form
                if kind == 'pointwise':
                    n_fac += int(expected)
                    n_abs += int(not expected)
                if bool(pt.multiply_smooth) != expected:
                    ms_ok = False
                    ms_msgs.append(
                        f'{kind}: {nuc} multiply_smooth='
                        f'{bool(pt.multiply_smooth)} expected {expected}')
    finally:
        plib.close()
        glib.close()

    # --- Summary ---
    print('=' * 78)
    print('Gate 3 - URR probability-table round-trip')
    print('=' * 78)
    hdr = (f'{"library":9s} {"nuclide":8s} {"NUNR":>4s} {"NBAND":>5s} '
           f'{"E==":>4s} {"tbl==":>5s} {"~1e-12":>7s} '
           f'{"max_abs":>10s} {"max_rel":>10s}  result')
    print(hdr)
    print('-' * 78)
    all_ok = True
    for r in results:
        all_ok &= r['ok']
        print(f'{r["kind"]:9s} {r["nuc"]:8s} {r["nunr"]:4d} {r["nband"]:5d} '
              f'{"Y" if r["energy_exact"] else "N":>4s} '
              f'{"Y" if r["table_exact"] else "N":>5s} '
              f'{"Y" if r["table_close"] else "N":>7s} '
              f'{r["max_abs"]:10.3e} {r["max_rel"]:10.3e}  '
              f'{"PASS" if r["ok"] else "FAIL"}')
        for m in r['msgs']:
            print(f'    ! {m}')
    print('-' * 78)
    print(f'absent-nuclide ({ABSENT_NUCLIDE}) graceful False/None: '
          f'{"PASS" if absent_ok else "FAIL"}')
    for m in absent_msgs:
        print(f'    ! {m}')
    all_ok &= absent_ok

    print('-' * 78)
    print(f'multiply_smooth LSSF split (both libs, 24 flagged): '
          f'{n_abs} absolute / {n_fac} factor per lib '
          f'(expect 5/19): {"PASS" if ms_ok else "FAIL"}')
    for m in ms_msgs:
        print(f'    ! {m}')
    all_ok &= ms_ok

    worst_abs = max((r['max_abs'] for r in results
                     if not np.isnan(r['max_abs'])), default=0.0)
    worst_rel = max((r['max_rel'] for r in results
                     if not np.isnan(r['max_rel'])), default=0.0)
    print('-' * 78)
    print(f'worst max_abs deviation: {worst_abs:.3e}')
    print(f'worst max_rel deviation: {worst_rel:.3e}')
    print('=' * 78)
    print(f'GATE 3: {"PASS" if all_ok else "FAIL"}')
    print('=' * 78)
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
