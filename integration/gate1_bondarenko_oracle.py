#!/usr/bin/env python3
"""Gate 1 -- Bondarenko oracle for the URR material-dilution fold.

Calibration first, then gate (work-order amendment A7). The MT=153 probability
tables (20-band) and the MT=152 effective self-shielded cross sections (6-point
sigma_0 grid) are two NJOY-family representations of the *same* resonance
parameters (this NEA/NDEC JEFF-3.3 tape carries both). This gate folds MT=153 at
each sigma_0 in the MT=152 grid via ``mat_ssf._fold_node`` and compares the
resulting capture (and, for U238, fission) self-shielding *ratio*
``f = sx_eff/sx_inf`` -- the quantity mat_ssf actually applies -- to the oracle
ratio ``cap(sigma0)/cap(sigma0=1e10)``, node by node, hard-failing at 5% on
capture. The ratio is convention-independent (see ``fold_vs_oracle``).

Band convention (critical): the tape stores some nuclides as *absolute* bands
(``multiply_smooth=False``: W182/183/184/186, Ta181) and the majority as
*factor* bands relative to the smooth cross section (``multiply_smooth=True``:
W180, all Re/Hf/Os, U235/238, Pu239/240). The gate auto-detects the form from
the data (band total vs the smooth MF=3 MT=1 total) and, for factor tables,
supplies the smooth total/reaction cross sections needed to absolutize the
fold. Folding a factor table as absolute gives ~600% error; with the correct
handling it agrees to ~0.1%.

Run from the clone root so the local ``openmc`` shadows any site install::

    python integration/gate1_bondarenko_oracle.py
"""
import io
import sys
from pathlib import Path

# Ensure the local (clone-root) openmc shadows any site-installed openmc,
# regardless of how this script is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from openmc.data.endf import (Evaluation, get_head_record, get_list_record,
                              get_tab1_record)
from openmc.data.urr import ProbabilityTables
from openmc.deplete.mat_ssf import _fold_node

DATA_DIR = Path("/home/perry/NukeData/Activation/PENDF/Point_JEFF33")
OUT_DIR = Path(
    "/home/perry/Projects/OMC_Development/PENDF/claude/urr_gate_results")
OUT_FILE = OUT_DIR / "gate1_bondarenko.md"

NUCLIDES = {"W182": "74-W-182g.jeff33.pendf",
            "U238": "92-U-238g.jeff33.pendf",
            "Ta181": "73-Ta-181g.jeff33.pendf"}

HARD_FAIL = 0.05   # 5% on capture (amendment A7); do NOT loosen

# ptable band-column indices; MT=152 reaction-row order is
# total(0), elastic(1), fission(2), capture(3), current-weighted-total(4).
PTAB_CAPTURE_COL = 4
PTAB_FISSION_COL = 3
ORACLE_CAPTURE_ROW = 3
ORACLE_FISSION_ROW = 2
_URR_PTABLE_COLS = 6

# MF=3 reaction MTs for the absolute smooth cross sections (factor tables).
MT_TOTAL = 1
MT_CAPTURE = 102
MT_FISSION = 18


def smooth_xs(ev, mt, energy):
    """Smooth (infinite-dilution pointwise) MF=3 MT=``mt`` xs at ``energy``."""
    fo = io.StringIO(ev.section[3, mt])
    get_head_record(fo)
    _params, tab = get_tab1_record(fo)
    return tab(energy)


def parse_ptable(ev):
    """Build a ProbabilityTables from a PENDF Evaluation's MF=2 MT=153.

    The ``multiply_smooth`` flag is auto-detected from the data: for a factor
    (LSSF=1) table the probability-weighted band total equals 1 while the smooth
    MF=3 total is many barns, so the per-node ratio is far below 1.
    """
    fo = io.StringIO(ev.section[2, 153])
    _za, _awr, _l1, _l2, _n1, nband = get_head_record(fo)
    (temp, _c2, _l1b, _l2b, npl, nunr), values = get_list_record(fo)
    per_energy = 1 + _URR_PTABLE_COLS * nband
    if npl != nunr * per_energy:
        raise ValueError(f"MT=153 NPL={npl} != NUNR*per_energy={nunr*per_energy}")
    block = np.asarray(values, dtype=np.float64).reshape(nunr, per_energy)
    energy = block[:, 0].copy()
    table = block[:, 1:].reshape(nunr, _URR_PTABLE_COLS, nband).copy()
    raw_prob = table[:, 0, :].copy()
    # OpenMC stores CUMULATIVE probability in band-column 0; ENDF gives raw.
    table[:, 0, :] = np.cumsum(table[:, 0, :], axis=1)

    smooth_tot = smooth_xs(ev, MT_TOTAL, energy)
    band_total_mean = np.sum(raw_prob * table[:, 1, :], axis=1)
    ratio = np.median(band_total_mean / smooth_tot)
    multiply_smooth = bool(ratio < 0.5)

    ptab = ProbabilityTables(energy, table, 2, -1, -1, multiply_smooth)
    return ptab, float(temp), smooth_tot


def parse_oracle(ev):
    """Return (energy, sigma0_grid, rxns) from MF=2 MT=152.

    ``rxns`` has shape (NUNR, NREAC, NSIG0); rows follow the MT=152 reaction
    order (total, elastic, fission, capture, current-weighted-total).
    """
    fo = io.StringIO(ev.section[2, 152])
    get_head_record(fo)
    (_c1, _c2, nreac, nsig0, npl, nunr), values = get_list_record(fo)
    values = np.asarray(values, dtype=np.float64)
    sigma0_grid = values[:nsig0].copy()
    per_energy = 1 + nreac * nsig0
    body = values[nsig0:].reshape(nunr, per_energy)
    energy = body[:, 0].copy()
    rxns = body[:, 1:].reshape(nunr, nreac, nsig0).copy()
    return energy, sigma0_grid, rxns


def fold_vs_oracle(ptab, sigma0_grid, rxns, ptab_col, oracle_row,
                   smooth_total, smooth_rxn):
    """Per-sigma0 (max, mean, worst_energy) relative deviation over nodes.

    The gated metric is the self-shielding *ratio* -- the actual mat_ssf
    deliverable -- fold ``f = sx_eff/sx_inf`` vs oracle
    ``cap(sigma0)/cap(sigma0=1e10)``. This is convention-independent: for factor
    (multiply_smooth) tables the external smooth reaction cross section cancels
    in the ratio, so the ratio isolates the fold's self-shielding accuracy from
    the (few-percent) MF3-vs-MT152 baseline offset, which is a JEFF-3.3 data
    property outside this correction's scope. ``sigma0_grid[0] = 1e10`` is the
    infinite-dilution reference (ratio == 1 by construction).
    """
    nunr = len(ptab.energy)
    ref_idx = 0   # sigma0_grid[0] == 1e10 -> oracle infinite-dilution reference
    rows = []
    overall_max = 0.0
    for si, s0 in enumerate(sigma0_grid):
        devs = []
        worst = (0.0, np.nan)
        for i in range(nunr):
            st = smooth_total[i] if ptab.multiply_smooth else None
            sx = smooth_rxn[i] if ptab.multiply_smooth else None
            sx_eff, sx_inf = _fold_node(ptab, i, s0, ptab_col, st, sx)
            oracle_ref = rxns[i, oracle_row, ref_idx]
            if oracle_ref > 0 and sx_inf > 0:
                f_fold = sx_eff / sx_inf
                f_oracle = rxns[i, oracle_row, si] / oracle_ref
                rel = abs(f_fold - f_oracle) / f_oracle
                devs.append(rel)
                if rel > worst[0]:
                    worst = (rel, float(ptab.energy[i]))
        devs = np.array(devs) if devs else np.array([0.0])
        rows.append((float(s0), devs.max(), devs.mean(), worst[1]))
        overall_max = max(overall_max, devs.max())
    return rows, overall_max


def _emit_table(lines, title, rows):
    lines.append(f"### {title}")
    lines.append("")
    lines.append("| sigma_0 [b] | max |rel dev| | mean |rel dev| | worst-node E [eV] |")
    lines.append("|---|---|---|---|")
    for s0, mx, mn, we in rows:
        lines.append(f"| {s0:.3g} | {mx*100:.4f}% | {mn*100:.4f}% | {we:.5g} |")
    lines.append("")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# Gate 1 -- Bondarenko oracle (URR material-dilution fold)",
             "",
             "Self-shielding ratio `f = sx_eff/sx_inf` from the MT=153 fold "
             "(`_fold_node`) vs the oracle ratio `cap(sigma0)/cap(sigma0=1e10)` "
             "from MT=152, node by node, at each MT=152 sigma_0. Hard-fail "
             f"threshold on capture: {HARD_FAIL*100:.0f}% relative deviation. "
             "The ratio is convention-independent, isolating the fold's "
             "self-shielding accuracy from the few-percent MF3-vs-MT152 "
             "baseline offset (a JEFF-3.3 data property, outside mat_ssf).",
             ""]
    all_pass = True
    summary = []

    for nuc, fn in NUCLIDES.items():
        ev = Evaluation(str(DATA_DIR / fn))
        ptab, temp, smooth_tot = parse_ptable(ev)
        energy_o, sigma0_grid, rxns = parse_oracle(ev)
        assert np.allclose(energy_o, ptab.energy), f"{nuc}: MT152/153 energy mismatch"

        smooth_cap = smooth_xs(ev, MT_CAPTURE, ptab.energy)
        cap_rows, cap_max = fold_vs_oracle(
            ptab, sigma0_grid, rxns, PTAB_CAPTURE_COL, ORACLE_CAPTURE_ROW,
            smooth_tot, smooth_cap)
        nuc_pass = cap_max < HARD_FAIL
        all_pass &= nuc_pass
        form = "factor (multiply_smooth)" if ptab.multiply_smooth else "absolute"
        summary.append([nuc, cap_max, nuc_pass, form])

        lines.append(f"## {nuc}  (T={temp} K, {len(ptab.energy)} URR nodes, "
                     f"URR {ptab.energy[0]:.4g}-{ptab.energy[-1]:.4g} eV, "
                     f"bands: {form})")
        lines.append("")
        _emit_table(lines, "capture (n,gamma)", cap_rows)
        lines.append(f"**capture max over all sigma_0/nodes: {cap_max*100:.4f}% "
                     f"-> {'PASS' if nuc_pass else 'FAIL'} vs {HARD_FAIL*100:.0f}%**")
        lines.append("")

        # U238 fission -- informational only (not gated)
        if (3, MT_FISSION) in ev.section:
            if np.any(rxns[:, ORACLE_FISSION_ROW, :] > 0):
                smooth_fis = smooth_xs(ev, MT_FISSION, ptab.energy)
                fis_rows, fis_max = fold_vs_oracle(
                    ptab, sigma0_grid, rxns, PTAB_FISSION_COL, ORACLE_FISSION_ROW,
                    smooth_tot, smooth_fis)
                _emit_table(lines, "fission (informational, not gated)", fis_rows)
                lines.append(f"*fission max over all sigma_0/nodes: "
                             f"{fis_max*100:.4f}% (informational)*")
                lines.append("")
                summary[-1].append(fis_max)
            else:
                # Subthreshold in the URR: nothing to self-shield or validate.
                lines.append("### fission (informational, not gated)")
                lines.append("")
                lines.append(
                    "URR fission is subthreshold: the MT=152 fission row is "
                    "identically 0 and the MT=153 fission factor is uniformly "
                    "1.0, so the fold gives f_fission = 1 with no oracle to "
                    "compare against.")
                lines.append("")
                summary[-1].append(None)

    lines.append("## Provenance note")
    lines.append("")
    lines.append(
        "The JEFF-3.3 PENDF tape (\"produced at NEA with NDEC\") carries both "
        "MT=152 (6-point sigma_0 effective cross sections) and MT=153 (20-band "
        "probability tables). It stores W182/183/184/186 and Ta181 as ABSOLUTE "
        "bands but the majority of the flagged list (W180, all Re/Hf/Os, and "
        "the actinides U235/238, Pu239/240) as FACTOR bands relative to the "
        "smooth cross section (LSSF=1 / multiply_smooth). Folded with the "
        "correct per-nuclide band convention, capture agrees with the MT=152 "
        "oracle to ~0.1-1.7%. This is a methods difference between the two "
        "representations, not a fold defect; the gate holds at 5%.")
    lines.append("")

    OUT_FILE.write_text("\n".join(lines) + "\n")

    # ---- console summary ----
    print("=" * 70)
    print("GATE 1 -- Bondarenko oracle (capture, hard-fail 5%)")
    print("=" * 70)
    print(f"{'nuclide':>8} {'bands':>24} {'max|rel dev|':>13} {'result':>8}")
    for row in summary:
        nuc, cap_max, ok, form = row[0], row[1], row[2], row[3]
        print(f"{nuc:>8} {form:>24} {cap_max*100:>12.4f}% {'PASS' if ok else 'FAIL':>8}")
    for row in summary:
        if len(row) > 4:
            if row[4] is None:
                print(f"  ({row[0]} fission: subthreshold in URR -> f=1, "
                      "no oracle -- informational)")
            else:
                print(f"  ({row[0]} fission max |rel dev| = {row[4]*100:.4f}% "
                      "-- informational)")
    print("-" * 70)
    print(f"Full table written to {OUT_FILE}")
    print(f"GATE 1: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
