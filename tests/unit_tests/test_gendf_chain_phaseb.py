"""Phase B tests: isomeric-metadata gating, export round-trip fidelity, and
deterministic trailing nuclide order.

Covers review items M3 (gate ``keep_isomeric_siblings`` on chain metadata),
M4 (lossless load->export->load for flag-only and embedded branching), and the
two minor fixes (defensive ``gendf_lfs`` parsing; deterministic sort key).
"""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import openmc.deplete
from openmc.deplete.chain import Chain
from openmc.deplete.openmc_operator import OpenMCOperator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plain_sibling_chain():
    """Chain where Cd111 -> Cd112 (ground) and Cd112_m1 is an unbranched sibling."""
    chain = openmc.deplete.Chain()

    cd111 = openmc.deplete.Nuclide('Cd111')
    cd111.add_reaction('(n,gamma)', 'Cd112', Q=5e6, branching_ratio=1.0)
    chain.add_nuclide(cd111)

    cd112 = openmc.deplete.Nuclide('Cd112')
    chain.add_nuclide(cd112)

    cd112_m1 = openmc.deplete.Nuclide('Cd112_m1')
    cd112_m1.half_life = 1000.0
    chain.add_nuclide(cd112_m1)

    chain._build_isomeric_families_cache()
    return chain


def _branched_sibling_chain():
    """Same topology, but Cd111 (n,gamma) branches to Cd112 and Cd112_m1."""
    chain = _plain_sibling_chain()
    chain.isomeric_branching_targets = {
        'Cd111': {'(n,gamma)': ['Cd112', 'Cd112_m1']}
    }
    chain._build_isomeric_families_cache()
    return chain


# ---------------------------------------------------------------------------
# M3: gate keep_isomeric_siblings on chain isomeric metadata
# ---------------------------------------------------------------------------

def test_m3_no_metadata_vanilla_parity():
    """Without metadata, keep_isomeric_siblings=True matches False (upstream)."""
    kept = _plain_sibling_chain().reduce(['Cd111'], level=1,
                                         keep_isomeric_siblings=True)
    dropped = _plain_sibling_chain().reduce(['Cd111'], level=1,
                                            keep_isomeric_siblings=False)
    kept_names = {n.name for n in kept.nuclides}
    dropped_names = {n.name for n in dropped.nuclides}
    assert kept_names == dropped_names == {'Cd111', 'Cd112'}
    assert 'Cd112_m1' not in kept_names


def test_m3_with_metadata_keeps_siblings():
    """With metadata, True keeps the branched sibling; False drops it."""
    kept = _branched_sibling_chain().reduce(['Cd111'], level=1,
                                            keep_isomeric_siblings=True)
    dropped = _branched_sibling_chain().reduce(['Cd111'], level=1,
                                               keep_isomeric_siblings=False)
    kept_names = {n.name for n in kept.nuclides}
    dropped_names = {n.name for n in dropped.nuclides}
    assert kept_names == {'Cd111', 'Cd112', 'Cd112_m1'}
    assert 'Cd112_m1' not in dropped_names
