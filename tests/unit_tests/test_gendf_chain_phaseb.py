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


def _embedded_equal(a, b):
    """Numpy-aware equality for isomeric_branching_embedded dicts."""
    if a is None or b is None:
        return a is b
    if set(a) != set(b):
        return False
    for key in a:
        da, db = a[key], b[key]
        if da['targets'] != db['targets']:
            return False
        if not np.array_equal(da['energies'], db['energies']):
            return False
        if set(da['branching_ratios']) != set(db['branching_ratios']):
            return False
        for t in da['branching_ratios']:
            if not np.array_equal(da['branching_ratios'][t],
                                  db['branching_ratios'][t]):
                return False
    return True


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


# ---------------------------------------------------------------------------
# M4: lossless load -> export -> load round-trip
# ---------------------------------------------------------------------------

_FLAGS_XML = """<?xml version="1.0"?>
<depletion_chain>
  <nuclide name="Nb93" reactions="1">
    <reaction type="(n,2n)" Q="-8000000.0">
      <isomeric_branching targets="Nb92 Nb92_m1" gendf_lfs="0 1" Q="-8.0e6 -8.1e6"/>
    </reaction>
  </nuclide>
  <nuclide name="Nb92" reactions="0"/>
  <nuclide name="Nb92_m1" reactions="0"/>
</depletion_chain>
"""

_EMBEDDED_XML = """<?xml version="1.0"?>
<depletion_chain>
  <nuclide name="Ag107" reactions="1">
    <reaction type="(n,2n)" target="Ag106" Q="-9000000.0"/>
    <reaction type="(n,gamma)" target="Ag108" Q="7000000.0">
      <isomeric_yields type="energy_dependent">
        <energies>1.000000e-05 2.000000e+07</energies>
        <targets>Ag108 Ag108_m1</targets>
        <branching_ratios>
          9.000000e-01 8.000000e-01
          1.000000e-01 2.000000e-01
        </branching_ratios>
      </isomeric_yields>
    </reaction>
  </nuclide>
  <nuclide name="Ag106" reactions="0"/>
  <nuclide name="Ag108" reactions="0"/>
  <nuclide name="Ag108_m1" reactions="0"/>
</depletion_chain>
"""


def test_m4_flags_roundtrip(tmp_path):
    """Flag-only chain round-trips targets and gendf_lfs."""
    src = tmp_path / "flags_in.xml"
    src.write_text(_FLAGS_XML)
    chain = Chain.from_xml(src)

    out = tmp_path / "flags_out.xml"
    chain.export_to_xml(out)
    reloaded = Chain.from_xml(out)

    assert reloaded.isomeric_branching_targets == chain.isomeric_branching_targets
    assert reloaded.isomeric_branching_lfs == chain.isomeric_branching_lfs
    assert reloaded.isomeric_branching_lfs == {'Nb93': {'(n,2n)': [0, 1]}}


def test_m4_embedded_roundtrip(tmp_path):
    """Embedded energy-dependent ratios round-trip (previously silently lost)."""
    src = tmp_path / "emb_in.xml"
    src.write_text(_EMBEDDED_XML)
    chain = Chain.from_xml(src)
    assert chain.isomeric_branching_embedded is not None

    out = tmp_path / "emb_out.xml"
    chain.export_to_xml(out)

    # The exported XML must carry the lossless legacy form, not bare flags.
    assert '<isomeric_yields' in out.read_text()

    reloaded = Chain.from_xml(out)
    assert _embedded_equal(reloaded.isomeric_branching_embedded,
                           chain.isomeric_branching_embedded)
    # Scalar reaction target/Q survive so the transmutation path still works.
    gamma = next(r for r in reloaded['Ag107'].reactions
                 if r.type == '(n,gamma)')
    assert gamma.target == 'Ag108'


# ---------------------------------------------------------------------------
# Minor #2: malformed gendf_lfs must not abort Chain.from_xml
# ---------------------------------------------------------------------------

def test_minor2_malformed_gendf_lfs(tmp_path):
    """A non-integer gendf_lfs token warns and is skipped, chain still loads."""
    bad = _FLAGS_XML.replace('gendf_lfs="0 1"', 'gendf_lfs="0 x"')
    src = tmp_path / "bad_lfs.xml"
    src.write_text(bad)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chain = Chain.from_xml(src)

    assert chain.isomeric_branching_targets == {'Nb93': {'(n,2n)': ['Nb92', 'Nb92_m1']}}
    assert chain.isomeric_branching_lfs is None
    assert any("Malformed gendf_lfs" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# Minor #11: deterministic trailing (non-chain) nuclide order
# ---------------------------------------------------------------------------

def _fake_operator(chain_nuclides, model_nuclides):
    """Minimal stand-in exercising OpenMCOperator._get_burnable_mats sorting."""
    mat = SimpleNamespace(get_nuclides=lambda: list(model_nuclides),
                          depletable=True, id=1, name='m',
                          volume=1.0, fissionable_mass=0.0)
    return SimpleNamespace(
        materials=[mat],
        nuclides_with_data=set(model_nuclides),
        _decay_nucs=set(),
        chain=SimpleNamespace(
            nuclide_dict={n: i for i, n in enumerate(chain_nuclides)}),
        heavy_metal=0.0)


def test_minor11_trailing_order_deterministic_and_alphabetical():
    """Non-chain nuclides append in alphabetical order regardless of set order."""
    chain_nuclides = ['H1', 'He4', 'U235']
    # Non-chain nuclides deliberately not alphabetical in the input set
    model_nuclides = {'U235', 'Zr90', 'Ba140', 'Kr85', 'H1'}

    _, _, nuclides = OpenMCOperator._get_burnable_mats(
        _fake_operator(chain_nuclides, model_nuclides))

    # Chain nuclides come first, in chain order
    assert nuclides[:3] == ['H1', 'He4', 'U235']
    # Trailing non-chain nuclides are alphabetical (deterministic)
    trailing = nuclides[3:]
    assert trailing == sorted(trailing)
    assert trailing == ['Ba140', 'Kr85', 'Zr90']


def test_minor11_stable_across_set_orderings():
    """Result is identical for different input set iteration orders."""
    chain_nuclides = ['H1', 'U235']
    names = ['Zr90', 'Ba140', 'Kr85', 'Xe135', 'Cs137']

    results = []
    for shift in range(len(names)):
        rotated = set(names[shift:] + names[:shift])
        _, _, nuclides = OpenMCOperator._get_burnable_mats(
            _fake_operator(chain_nuclides, rotated))
        results.append(nuclides)

    assert all(r == results[0] for r in results)
    assert results[0] == ['H1', 'U235', 'Ba140', 'Cs137', 'Kr85', 'Xe135', 'Zr90']
