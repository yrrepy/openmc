#!/usr/bin/env python3
"""Gate 2 -- infinite-dilution sanity and monotonicity of the URR fold.

Three checks on W182, U238, Ta181 (mixing absolute and factor band forms):

(a) Node level -- at sigma_0 = 1e10 the fold factor ``f = sx_eff/sx_inf`` equals
    1 within ``|f - 1| < 1e-6`` (amendment A6) at every URR node.
(b) Group level -- ``mat_ssf_factors`` with sigma_0 = 1e10 in every group, on the
    CCFE-709 collapse structure, returns ``f_g == 1`` within 1e-6 for ALL groups
    (inside the URR span through the fold, outside through the inf_g==0 branch).
(c) Monotonicity -- the URR-integrated capture factor decreases as sigma_0 falls
    through the MT=152 grid [1e10, 1e4, 1e3, 1e2, 1e1, 1e0] for all three
    nuclides (self-shielding strengthens as the background thins).

Factor (multiply_smooth) nuclides such as U238 are absolutized with the smooth
MF=3 total/capture cross sections at the URR nodes; absolute nuclides ignore
them. Run from the clone root::

    python integration/gate2_infinite_dilution.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# clone root first so the local openmc shadows any site install; then the
# integration dir so the shared gate-1 parse helpers import.
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np

from openmc.data.endf import Evaluation
from openmc.mgxs import GROUP_STRUCTURES
from openmc.deplete.mat_ssf import _fold_node, mat_ssf_factors
from gate1_bondarenko_oracle import (
    DATA_DIR, NUCLIDES, parse_ptable, parse_oracle, smooth_xs,
    PTAB_CAPTURE_COL, ORACLE_CAPTURE_ROW, MT_CAPTURE)

TOL = 1e-6          # amendment A6
GROUP_NAME = "CCFE-709"
SIGMA0_INF = 1e10


def load(nuc):
    """Return (ptab, smooth_total, smooth_capture) for a gate nuclide."""
    ev = Evaluation(str(DATA_DIR / NUCLIDES[nuc]))
    ptab, _temp, smooth_tot = parse_ptable(ev)
    smooth_cap = smooth_xs(ev, MT_CAPTURE, ptab.energy)
    energy_o, sigma0_grid, rxns = parse_oracle(ev)
    return ptab, smooth_tot, smooth_cap, sigma0_grid, rxns


def check_a_node_infinite_dilution(data):
    """(a) node-level f(sigma0=1e10) == 1 within TOL for every node."""
    print("(a) node-level f(sigma_0=1e10) == 1 within "
          f"{TOL:.0e}:")
    ok = True
    for nuc, (ptab, st, sx, _grid, _rxns) in data.items():
        worst = 0.0
        for i in range(len(ptab.energy)):
            eff, inf = _fold_node(ptab, i, SIGMA0_INF, PTAB_CAPTURE_COL,
                                  st[i], sx[i])
            worst = max(worst, abs(eff / inf - 1.0))
        passed = worst < TOL
        ok &= passed
        print(f"    {nuc:>6}: max |f-1| = {worst:.2e}  "
              f"-> {'PASS' if passed else 'FAIL'}")
    return ok


def check_b_group_infinite_dilution(data, group_edges):
    """(b) mat_ssf_factors == 1 within TOL for ALL CCFE-709 groups."""
    n_groups = len(group_edges) - 1
    sigma0_g = np.full(n_groups, SIGMA0_INF)
    print(f"(b) group-level f_g(sigma_0=1e10) == 1 within {TOL:.0e} "
          f"on {GROUP_NAME} ({n_groups} groups):")
    ok = True
    for nuc, (ptab, st, sx, _grid, _rxns) in data.items():
        f_g = mat_ssf_factors(ptab, sigma0_g, group_edges, '(n,gamma)',
                              smooth_total=st, smooth_rxn=sx)
        worst = np.max(np.abs(f_g - 1.0))
        worst_g = int(np.argmax(np.abs(f_g - 1.0)))
        n_urr = int(np.sum(np.abs(f_g - 1.0) > 0))
        passed = worst < TOL
        ok &= passed
        print(f"    {nuc:>6}: max |f_g-1| = {worst:.2e} "
              f"(group {worst_g}, {n_urr} groups touched by URR span) "
              f"-> {'PASS' if passed else 'FAIL'}")
    return ok


def _integrated_factor(ptab, st, sx, sigma0):
    """URR-integrated capture self-shielding factor at scalar sigma0."""
    num = den = 0.0
    for i in range(len(ptab.energy)):
        eff, inf = _fold_node(ptab, i, sigma0, PTAB_CAPTURE_COL, st[i], sx[i])
        num += eff
        den += inf
    return num / den


def check_c_monotonicity(data):
    """(c) URR-integrated capture factor decreases as sigma_0 falls."""
    print("(c) monotonic decrease of capture f as sigma_0 falls "
          "through the MT=152 grid:")
    ok = True
    for nuc, (ptab, st, sx, sigma0_grid, _rxns) in data.items():
        # MT=152 grid is descending [1e10, ..., 1e0]
        fs = [_integrated_factor(ptab, st, sx, s0) for s0 in sigma0_grid]
        # strictly decreasing (allow tiny float slack)
        diffs = np.diff(fs)
        passed = np.all(diffs <= 1e-9)
        ok &= passed
        seq = " > ".join(f"{v:.4f}" for v in fs)
        print(f"    {nuc:>6}: f = [{seq}]  -> "
              f"{'PASS (monotone dec.)' if passed else 'FAIL'}")
    return ok


def main():
    if GROUP_NAME not in GROUP_STRUCTURES:
        raise SystemExit(f"group structure {GROUP_NAME!r} not available")
    group_edges = np.asarray(GROUP_STRUCTURES[GROUP_NAME], dtype=float)

    data = {nuc: load(nuc) for nuc in NUCLIDES}

    print("=" * 64)
    print("GATE 2 -- infinite-dilution sanity + monotonicity")
    print("=" * 64)
    a = check_a_node_infinite_dilution(data)
    print()
    b = check_b_group_infinite_dilution(data, group_edges)
    print()
    c = check_c_monotonicity(data)
    print("-" * 64)
    print(f"(a) node infinite-dilution : {'PASS' if a else 'FAIL'}")
    print(f"(b) group infinite-dilution: {'PASS' if b else 'FAIL'}")
    print(f"(c) monotonicity           : {'PASS' if c else 'FAIL'}")
    all_pass = a and b and c
    print(f"GATE 2: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
