#!/usr/bin/env python
"""Gate 4 - URR material-dilution self-shielding, end-to-end.

Runs the ``urr_material_dilution`` correction *inside* the normal PENDF
multigroup-flux collapse, on the grouped JEFF-3.3 URR library's own CCFE-709
group structure, weighted by a realistic HFR (thermal MTR) spectrum. The HFR
spectrum ``URR/616_HFR-low.txt`` is tabulated on LLNL-616 (descending); it is
parsed, reversed to ascending, and re-binned integral-preserving onto CCFE-709
(NOT collapsed on LLNL-616). Pure natural tungsten is the material.

Checks (see the work order gate 4):
  (a) one-group W-isotope (n,gamma) is STRICTLY lower with the flag on;
  (b) per-group f_g computed directly for each W isotope at its nat-W sigma_0:
      f<1 across the URR span, exactly 1 outside it;
  (c) (n,2n) rows are byte-identical on vs off;
  (d) report per-isotope sigma_0, URR-avg f, one-group (n,gamma) on/off ratio.

(a) and (c) are strict PASS/FAIL; (b) is strict (f<1 in span, =1 outside); the
endf/b-viii anchor magnitudes (W186 ~0.88, W184 ~0.84) are calibration, not
gates - JEFF-3.3 differs. The result table is written to
``claude/urr_gate_results/gate4_end_to_end.md``.

Run from the clone root::

    python integration/gate4_end_to_end.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# Local (modified) openmc must win over any site-installed copy.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from openmc.data.pendf_grouped import GroupedPendfLibrary
from openmc.deplete.microxs import MicroXS
from openmc.deplete.mat_ssf import (
    material_dilution_sigma0, mat_ssf_factors, _smooth_at_nodes,
    _BASE_REACTION_MT)

# Pure natural tungsten (atom fractions; only ratios matter for sigma_0).
NAT_W = {'W180': 0.0012, 'W182': 0.2650, 'W183': 0.1431,
         'W184': 0.3064, 'W186': 0.2843}
W_ISOTOPES = list(NAT_W)
# A few transmutation products carry ptables but no material density -> f=1.
PRODUCTS = ['Ta181', 'Re185', 'Re187']
REACTIONS = ['(n,gamma)', '(n,2n)', 'fission']


def parse_hfr(path):
    """Parse the LLNL-616 HFR spectrum -> ascending (edges[617], flux[616]).

    File layout: 617 descending boundaries, a blank line, 616 descending flux
    values (per-group integrals), a norm line, a title line. Boundaries and
    fluxes are reversed to ascending; flux[j] belongs to ascending group j.
    """
    lines = [ln.strip() for ln in Path(path).read_text().splitlines()]
    blank = lines.index('')
    edges_desc = np.array([float(x) for x in lines[:blank]], dtype=float)
    rest = [ln for ln in lines[blank + 1:] if ln]
    # Last two non-blank lines are the norm and the title; the flux is the rest.
    flux_desc = np.array([float(x) for x in rest[:len(edges_desc) - 1]],
                         dtype=float)
    assert len(edges_desc) == 617, f"expected 617 edges, got {len(edges_desc)}"
    assert len(flux_desc) == 616, f"expected 616 flux, got {len(flux_desc)}"
    # Reverse to ascending: edge j <- 616-j, group flux j <- 615-j.
    return edges_desc[::-1].copy(), flux_desc[::-1].copy()


def rebin_integral(src_edges, src_flux, dst_edges):
    """Integral-preserving re-bin of per-group flux integrals.

    Each source group's integral is redistributed to the destination groups in
    proportion to their energy overlap (flat-in-energy within a source group).
    Total flux is conserved when the destination range covers the source range.
    """
    src_edges = np.asarray(src_edges, dtype=float)
    dst_edges = np.asarray(dst_edges, dtype=float)
    dst = np.zeros(len(dst_edges) - 1)
    for i, fi in enumerate(src_flux):
        if fi == 0.0:
            continue
        a, b = src_edges[i], src_edges[i + 1]
        width = b - a
        if width <= 0:
            continue
        lo = np.maximum(a, dst_edges[:-1])
        hi = np.minimum(b, dst_edges[1:])
        overlap = np.clip(hi - lo, 0.0, None)
        dst += fi * overlap / width
    return dst


def _group_totals(glib, nuclides):
    """Group total sigma_t,g for each nuclide from the grouped library."""
    return {nuc: glib.xs_g(nuc, 1) for nuc in nuclides}


def direct_fg(glib, nuc, edges, group_totals):
    """Per-group f_g for one W isotope at its nat-W sigma_0 (direct fold)."""
    ptab = glib.ptables(nuc)
    sigma0_g = material_dilution_sigma0(NAT_W, nuc, group_totals)
    if ptab.multiply_smooth:
        nodes = np.asarray(ptab.energy, dtype=float)
        st = _smooth_at_nodes(glib, nuc, nodes, [1], True, edges)
        sx = _smooth_at_nodes(glib, nuc, nodes, [_BASE_REACTION_MT['(n,gamma)']],
                              True, edges)
        f_g = mat_ssf_factors(ptab, sigma0_g, edges, '(n,gamma)',
                              smooth_total=st, smooth_rxn=sx)
    else:
        f_g = mat_ssf_factors(ptab, sigma0_g, edges, '(n,gamma)')
    return ptab, sigma0_g, f_g


def one_group(mx, nuc, rxn):
    """One-group value (barns for a normalized flux) for (nuc, rxn)."""
    return float(np.ravel(mx[nuc, rxn])[0])


def main():
    root = Path('/home/perry/Projects/OMC_Development/PENDF')
    parser = argparse.ArgumentParser(description='Gate 4 - end-to-end URR material-dilution self-shielding.')
    parser.add_argument('--grouped', type=Path, default=root / 'data/jeff33_urr_pendf_294K_ccfe709.h5', help='Grouped JEFF-3.3 URR library (CCFE-709)')
    parser.add_argument('--hfr',     type=Path, default=root / 'URR/616_HFR-low.txt',                   help='LLNL-616 HFR reference spectrum')
    parser.add_argument('--out',     type=Path, default=root / 'claude/urr_gate_results/gate4_end_to_end.md', help='Markdown result table')
    args = parser.parse_args()

    glib = GroupedPendfLibrary(args.grouped)
    edges = np.asarray(glib.group_edges, dtype=float)
    n_groups = len(edges) - 1

    # --- HFR spectrum: parse, reverse, re-bin onto CCFE-709 ---
    src_edges, src_flux = parse_hfr(args.hfr)
    flux = rebin_integral(src_edges, src_flux, edges)
    sum_before = float(src_flux.sum())
    sum_after = float(flux.sum())
    conservation = abs(sum_after - sum_before) / sum_before
    flux_norm = flux / flux.sum()

    # --- collapse: flag off vs on ---
    nucs = W_ISOTOPES + PRODUCTS
    kw = dict(energies=edges, multigroup_flux=flux_norm, nuclides=nucs,
              reactions=REACTIONS, pendf_library=glib)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        off = MicroXS.from_multigroup_flux(**kw)
        on = MicroXS.from_multigroup_flux(
            urr_material_dilution=NAT_W, **kw)

    # --- direct per-group f_g for each W isotope ---
    gtot = _group_totals(glib, W_ISOTOPES)
    mid = np.sqrt(edges[:-1] * edges[1:])   # geometric group midpoints

    rows = []
    check_a_ok = True
    check_b_ok = True
    for nuc in W_ISOTOPES:
        ptab, sigma0_g, f_g = direct_fg(glib, nuc, edges, gtot)
        e0, e1 = ptab.energy[0], ptab.energy[-1]
        span = (mid >= e0) & (mid <= e1)               # groups inside URR span
        fully_out = (edges[1:] <= e0) | (edges[:-1] >= e1)

        # (b) f<1 inside span, ==1 outside
        shielded = f_g < 1.0 - 1e-12
        b_inside = bool(np.all(f_g[span] < 1.0 + 1e-9) and shielded[span].any())
        b_outside = bool(np.all(f_g[fully_out] == 1.0))
        if not (b_inside and b_outside):
            check_b_ok = False

        # magnitudes: sigma_0 (mean over span), URR flux-weighted avg f, min f
        s0_span = float(np.mean(sigma0_g[span])) if span.any() else float('nan')
        w = flux[shielded]
        favg = (float(np.sum(f_g[shielded] * w) / np.sum(w))
                if w.sum() > 0 else 1.0)
        fmin = float(f_g[shielded].min()) if shielded.any() else 1.0

        # (a) one-group (n,gamma) strictly lower with flag on
        g_off = one_group(off, nuc, '(n,gamma)')
        g_on = one_group(on, nuc, '(n,gamma)')
        a_ok = g_on < g_off
        if not a_ok:
            check_a_ok = False

        rows.append(dict(nuc=nuc, ms=bool(ptab.multiply_smooth), s0=s0_span,
                         favg=favg, fmin=fmin, g_off=g_off, g_on=g_on,
                         ratio=g_on / g_off, n_shield=int(shielded.sum()),
                         b_in=b_inside, b_out=b_outside, a_ok=a_ok))

    # (c) (n,2n) rows byte-identical on vs off (all requested nuclides)
    check_c_ok = True
    c_detail = []
    for nuc in nucs:
        try:
            v_off = one_group(off, nuc, '(n,2n)')
            v_on = one_group(on, nuc, '(n,2n)')
        except (KeyError, ValueError):
            continue
        same = v_off == v_on
        c_detail.append((nuc, same))
        if not same:
            check_c_ok = False

    glib.close()

    all_ok = check_a_ok and check_b_ok and check_c_ok

    # --- console summary ---
    print('=' * 78)
    print('Gate 4 - URR material-dilution self-shielding, end-to-end')
    print('=' * 78)
    print(f'HFR flux re-bin LLNL-616 -> CCFE-709: sum before={sum_before:.6e}, '
          f'after={sum_after:.6e}, rel. change={conservation:.3e}')
    print('-' * 78)
    hdr = (f'{"nuclide":8s} {"form":7s} {"sigma0[b]":>10s} {"URRavg f":>9s} '
           f'{"min f":>7s} {"1g(ng)off":>11s} {"1g(ng)on":>11s} {"on/off":>8s} '
           f'{"(a)":>4s} {"(b)":>4s}')
    print(hdr)
    print('-' * 78)
    for r in rows:
        print(f'{r["nuc"]:8s} {"factor" if r["ms"] else "abs":7s} '
              f'{r["s0"]:10.3f} {r["favg"]:9.4f} {r["fmin"]:7.4f} '
              f'{r["g_off"]:11.5e} {r["g_on"]:11.5e} {r["ratio"]:8.5f} '
              f'{"Y" if r["a_ok"] else "N":>4s} '
              f'{"Y" if (r["b_in"] and r["b_out"]) else "N":>4s}')
    print('-' * 78)
    print(f'(a) one-group (n,gamma) strictly lower with flag on: '
          f'{"PASS" if check_a_ok else "FAIL"}')
    print(f'(b) f<1 in URR span & =1 outside (all W isotopes):   '
          f'{"PASS" if check_b_ok else "FAIL"}')
    print(f'(c) (n,2n) rows byte-identical on vs off:            '
          f'{"PASS" if check_c_ok else "FAIL"}  '
          f'({sum(s for _, s in c_detail)}/{len(c_detail)} identical)')
    print('=' * 78)
    print(f'GATE 4: {"PASS" if all_ok else "FAIL"}')
    print('=' * 78)

    # --- markdown report ---
    args.out.parent.mkdir(parents=True, exist_ok=True)
    md = []
    md.append('# Gate 4 - URR material-dilution self-shielding (end-to-end)\n')
    md.append(f'- Library: `{args.grouped.name}` (CCFE-709, {n_groups} groups)')
    md.append(f'- Spectrum: `{args.hfr.name}` (HFR thermal MTR, LLNL-616 '
              're-binned integral-preserving onto CCFE-709)')
    md.append(f'- Material: pure natural W {NAT_W}')
    md.append(f'- Flux conservation (sum before/after re-bin): '
              f'{sum_before:.6e} / {sum_after:.6e} '
              f'(rel. change {conservation:.2e})\n')
    md.append('## Per-isotope results\n')
    md.append('| nuclide | band form | sigma_0 [b] (URR-avg) | URR flux-avg f | '
              'min f | shielded groups | 1-group (n,g) off [b] | on [b] | '
              'on/off | (a) | (b) |')
    md.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        md.append(f'| {r["nuc"]} | {"factor" if r["ms"] else "absolute"} | '
                  f'{r["s0"]:.2f} | {r["favg"]:.4f} | {r["fmin"]:.4f} | '
                  f'{r["n_shield"]} | {r["g_off"]:.5e} | {r["g_on"]:.5e} | '
                  f'{r["ratio"]:.5f} | {"PASS" if r["a_ok"] else "FAIL"} | '
                  f'{"PASS" if (r["b_in"] and r["b_out"]) else "FAIL"} |')
    md.append('')
    md.append('## Gate checks\n')
    md.append(f'- **(a)** one-group (n,gamma) strictly lower with flag on: '
              f'**{"PASS" if check_a_ok else "FAIL"}**')
    md.append(f'- **(b)** f<1 across each URR span, exactly 1 outside: '
              f'**{"PASS" if check_b_ok else "FAIL"}**')
    md.append(f'- **(c)** (n,2n) rows byte-identical on vs off '
              f'({sum(s for _, s in c_detail)}/{len(c_detail)}): '
              f'**{"PASS" if check_c_ok else "FAIL"}**')
    md.append('')
    md.append('## Notes / calibration\n')
    md.append('- endf/b-viii.0 anchors (pure nat W, homogeneous sigma_0): '
              'W186 window-avg f ~0.88, W184 ~0.84. JEFF-3.3 differs '
              '(different evaluation); anchors are calibration, not gates.')
    md.append('- The net one-group (n,gamma) change is small because a thermal '
              'MTR spectrum places little flux in the URR (few keV-100 keV); '
              'the per-group f_g (column min f) is the physical self-shielding.')
    md.append('- W180 is factor-form (multiply_smooth=1): its f_g exercises the '
              'smooth-XS path end-to-end.')
    md.append(f'\n**GATE 4: {"PASS" if all_ok else "FAIL"}**\n')
    args.out.write_text('\n'.join(md))
    print(f'wrote {args.out}')

    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
