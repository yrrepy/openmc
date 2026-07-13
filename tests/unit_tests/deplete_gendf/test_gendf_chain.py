"""Consolidated GENDF chain tests: LFS metadata, XML round-trip, reduce pruning,
and isomeric sibling expansion.

Merges ``test_chain_reduce_isomeric``, ``test_chain_reduce_isomeric_siblings``,
``test_gendf_chain_phaseb``, and the LFS / round-trip / reduce subset of
``test_phase0_phase1_validation``. Single-parent isomeric chains use the shared
``make_isomeric_chain`` factory; the multi-parent / IT-decay sibling builders and
the ``_get_burnable_mats`` stand-in stay file-local (see conftest_api mapping).
"""

import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import openmc.deplete
from openmc.deplete import ReactionRates
from openmc.deplete.chain import Chain
from openmc.deplete.openmc_operator import OpenMCOperator

from .gendf_testing import make_isomeric_chain


# ---------------------------------------------------------------------------
# Single-parent chain builders (shared factory)
# ---------------------------------------------------------------------------

def _simple_isomeric_chain():
    """Ag109 (n,gamma) -> Ag110 (+ Ag110_m1 sibling)."""
    return make_isomeric_chain(
        'Ag109', {'(n,gamma)': ['Ag110', 'Ag110_m1']},
        half_lives={'Ag110': 88560, 'Ag110_m1': 2.157e7})


def _multi_reaction_chain():
    """Ag109 with (n,gamma) and (n,2n), each branching to a metastable (no LFS)."""
    return make_isomeric_chain(
        'Ag109',
        {'(n,gamma)': ['Ag110', 'Ag110_m1'], '(n,2n)': ['Ag108', 'Ag108_m1']},
        half_lives={'Ag110_m1': 2.157e7, 'Ag108_m1': 1.382e10},
        q={'(n,gamma)': 6.8e6, '(n,2n)': -9.5e6})


def _three_target_chain():
    """Cd110 (n,gamma) branching to Cd111 and two metastables."""
    return make_isomeric_chain(
        'Cd110', {'(n,gamma)': ['Cd111', 'Cd111_m1', 'Cd111_m2']},
        half_lives={'Cd111_m1': 1000, 'Cd111_m2': 10000}, q=5e6)


def _lfs_chain():
    """Ir191 (n,gamma) branching to Ir192/m1/m2 with gendf LFS [0, 3, 15]."""
    return make_isomeric_chain(
        'Ir191', {'(n,gamma)': ['Ir192', 'Ir192_m1', 'Ir192_m2']},
        half_lives=dict.fromkeys(['Ir192', 'Ir192_m1', 'Ir192_m2'], 1e6),
        lfs={'(n,gamma)': [0, 3, 15]}, q=6.2e6)


def _no_lfs_chain():
    """Ag109 (n,gamma) branching to Ag110/m1 with no LFS metadata."""
    return make_isomeric_chain(
        'Ag109', {'(n,gamma)': ['Ag110', 'Ag110_m1']},
        half_lives=dict.fromkeys(['Ag110', 'Ag110_m1'], 1e5))


def _multi_reaction_lfs_chain():
    """Ag109 with two reactions, both carrying LFS metadata."""
    return make_isomeric_chain(
        'Ag109',
        {'(n,gamma)': ['Ag110', 'Ag110_m1'], '(n,2n)': ['Ag108', 'Ag108_m1']},
        half_lives={'Ag110': 1e5, 'Ag110_m1': 2.5e7,
                    'Ag108': 1e5, 'Ag108_m1': 1.4e10},
        lfs={'(n,gamma)': [0, 5], '(n,2n)': [0, 2]},
        q={'(n,gamma)': 6.8e6, '(n,2n)': -9.5e6})


# ---------------------------------------------------------------------------
# File-local sibling builders (multi-parent / IT decay -- deliberately bespoke)
# ---------------------------------------------------------------------------

def create_chain_with_siblings():
    """Ag108 -> Ag109 (ground); Ag109_m1 reachable only via Pd108 isomeric branch."""
    chain = openmc.deplete.Chain()

    ag108 = openmc.deplete.Nuclide('Ag108')
    ag108.add_reaction('(n,gamma)', 'Ag109', Q=5e6, branching_ratio=1.0)
    chain.add_nuclide(ag108)

    ag109 = openmc.deplete.Nuclide('Ag109')
    ag109.half_life = 100.0
    chain.add_nuclide(ag109)

    ag109_m1 = openmc.deplete.Nuclide('Ag109_m1')
    ag109_m1.half_life = 4.9
    ag109_m1.add_decay_mode('IT', 'Ag109', 1.0)
    chain.add_nuclide(ag109_m1)

    pd108 = openmc.deplete.Nuclide('Pd108')
    pd108.add_reaction('(n,p)', 'Ag109', Q=-2e6, branching_ratio=1.0)
    chain.add_nuclide(pd108)

    chain.isomeric_branching_targets = {'Pd108': {'(n,p)': ['Ag109', 'Ag109_m1']}}
    return chain


def create_complex_sibling_chain():
    """Two isomeric families: Ag109->Ag110/m1 and Ag110->Ag111/m1."""
    chain = openmc.deplete.Chain()

    ag109 = openmc.deplete.Nuclide('Ag109')
    ag109.add_reaction('(n,gamma)', 'Ag110', Q=6.8e6, branching_ratio=1.0)
    chain.add_nuclide(ag109)

    ag110 = openmc.deplete.Nuclide('Ag110')
    ag110.half_life = 24.6
    ag110.add_reaction('(n,gamma)', 'Ag111', Q=5.5e6, branching_ratio=1.0)
    chain.add_nuclide(ag110)

    ag110_m1 = openmc.deplete.Nuclide('Ag110_m1')
    ag110_m1.half_life = 249.79 * 86400
    ag110_m1.add_reaction('(n,gamma)', 'Ag111_m1', Q=5.5e6, branching_ratio=1.0)
    chain.add_nuclide(ag110_m1)

    ag111 = openmc.deplete.Nuclide('Ag111')
    ag111.half_life = 7.45 * 86400
    chain.add_nuclide(ag111)

    ag111_m1 = openmc.deplete.Nuclide('Ag111_m1')
    ag111_m1.half_life = 64.8
    chain.add_nuclide(ag111_m1)

    chain.isomeric_branching_targets = {
        'Ag109': {'(n,gamma)': ['Ag110', 'Ag110_m1']},
        'Ag110': {'(n,gamma)': ['Ag111', 'Ag111_m1']},
    }
    return chain


def _plain_sibling_chain():
    """Cd111 -> Cd112 (ground); Cd112_m1 is an unbranched sibling."""
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


# ===========================================================================
# LFS attributes
# ===========================================================================

def test_lfs_initialized_none():
    """Default chain has isomeric_branching_lfs == None."""
    chain = openmc.deplete.Chain()
    assert chain.isomeric_branching_lfs is None


def test_lfs_set_correctly():
    """LFS dict mirrors the target structure."""
    chain = _lfs_chain()
    assert chain.isomeric_branching_lfs is not None
    assert 'Ir191' in chain.isomeric_branching_lfs
    assert chain.isomeric_branching_lfs['Ir191']['(n,gamma)'] == [0, 3, 15]


def test_lfs_parallel_to_targets():
    """LFS list length matches the target list length."""
    chain = _lfs_chain()
    targets = chain.isomeric_branching_targets['Ir191']['(n,gamma)']
    lfs = chain.isomeric_branching_lfs['Ir191']['(n,gamma)']
    assert len(targets) == len(lfs)


# ===========================================================================
# XML round-trip
# ===========================================================================

def test_roundtrip_with_lfs():
    """Chain with LFS data round-trips through XML."""
    chain = _lfs_chain()
    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "chain.xml"
        chain.export_to_xml(xml_path)
        reloaded = openmc.deplete.Chain.from_xml(xml_path)

        assert reloaded.isomeric_branching_targets is not None
        assert reloaded.isomeric_branching_lfs is not None
        assert (reloaded.isomeric_branching_targets['Ir191']['(n,gamma)'] ==
                chain.isomeric_branching_targets['Ir191']['(n,gamma)'])
        assert (reloaded.isomeric_branching_lfs['Ir191']['(n,gamma)'] ==
                chain.isomeric_branching_lfs['Ir191']['(n,gamma)'])


def test_roundtrip_without_lfs():
    """Chain without LFS round-trips with lfs == None (targets preserved)."""
    chain = _no_lfs_chain()
    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "chain_nolfs.xml"
        chain.export_to_xml(xml_path)
        reloaded = openmc.deplete.Chain.from_xml(xml_path)

        assert reloaded.isomeric_branching_targets is not None
        assert 'Ag109' in reloaded.isomeric_branching_targets
        assert reloaded.isomeric_branching_lfs is None


def test_roundtrip_multi_reaction():
    """Multiple reactions with LFS all survive round-trip."""
    chain = _multi_reaction_lfs_chain()
    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "chain_multi.xml"
        chain.export_to_xml(xml_path)
        reloaded = openmc.deplete.Chain.from_xml(xml_path)

        for rx in ['(n,gamma)', '(n,2n)']:
            assert (reloaded.isomeric_branching_targets['Ag109'][rx] ==
                    chain.isomeric_branching_targets['Ag109'][rx])
            assert (reloaded.isomeric_branching_lfs['Ag109'][rx] ==
                    chain.isomeric_branching_lfs['Ag109'][rx])


def test_xml_contains_gendf_lfs_attribute():
    """Exported XML carries the gendf_lfs attribute '0 3 15'."""
    import lxml.etree as ET

    chain = _lfs_chain()
    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "chain_raw.xml"
        chain.export_to_xml(xml_path)

        root = ET.parse(str(xml_path)).getroot()
        found_lfs = False
        for nuc_elem in root.findall('nuclide'):
            for rx_elem in nuc_elem.findall('reaction'):
                iso_elem = rx_elem.find('isomeric_branching')
                if iso_elem is not None and iso_elem.get('gendf_lfs'):
                    assert iso_elem.get('gendf_lfs') == '0 3 15'
                    found_lfs = True

        assert found_lfs, "gendf_lfs attribute not found in exported XML"


def test_lfs_count_mismatch_in_xml():
    """A gendf_lfs token count != target count makes LFS load as None."""
    import lxml.etree as ET

    chain = _lfs_chain()
    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "chain.xml"
        chain.export_to_xml(xml_path)

        tree = ET.parse(str(xml_path))
        for nuc_elem in tree.getroot().findall('nuclide'):
            for rx_elem in nuc_elem.findall('reaction'):
                iso_elem = rx_elem.find('isomeric_branching')
                if iso_elem is not None:
                    iso_elem.set('gendf_lfs', '0 3')  # 2 tokens, 3 targets

        corrupted = Path(tmpdir) / "corrupted.xml"
        tree.write(str(corrupted), encoding='utf-8')
        reloaded = openmc.deplete.Chain.from_xml(corrupted)

        assert reloaded.isomeric_branching_targets is not None
        assert reloaded.isomeric_branching_lfs is None


def test_chain_no_isomeric_data():
    """Plain chain with no isomeric data round-trips to None/None."""
    chain = openmc.deplete.Chain()
    chain.add_nuclide(openmc.deplete.Nuclide('U235'))

    assert chain.isomeric_branching_targets is None
    assert chain.isomeric_branching_lfs is None

    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "plain.xml"
        chain.export_to_xml(xml_path)
        reloaded = openmc.deplete.Chain.from_xml(xml_path)
        assert reloaded.isomeric_branching_targets is None
        assert reloaded.isomeric_branching_lfs is None


def test_export_import_roundtrip_with_isomeric():
    """Isomeric targets survive export and reload."""
    chain = _simple_isomeric_chain()
    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "chain.xml"
        chain.export_to_xml(xml_path)
        reloaded = openmc.deplete.Chain.from_xml(xml_path)

        assert reloaded.isomeric_branching_targets is not None
        assert reloaded.isomeric_branching_targets['Ag109']['(n,gamma)'] == \
            ['Ag110', 'Ag110_m1']


def test_export_reduced_chain_roundtrip():
    """Reduced chain with pruned targets exports/reloads; pruned list is transient."""
    chain = _simple_isomeric_chain()
    reduced = chain.reduce(['Ag109', 'Ag110'], keep_isomeric_siblings=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "reduced.xml"
        reduced.export_to_xml(xml_path)
        reloaded = openmc.deplete.Chain.from_xml(xml_path)

        assert reloaded.isomeric_branching_targets['Ag109']['(n,gamma)'] == ['Ag110']
        assert reloaded.reduce_pruned_targets is None


# --- Phase B M4/minor: flag-only and embedded round-trip fidelity -----------

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
    gamma = next(r for r in reloaded['Ag107'].reactions if r.type == '(n,gamma)')
    assert gamma.target == 'Ag108'


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


# ===========================================================================
# reduce (target + LFS pruning)
# ===========================================================================

def test_reduce_preserves_targets_all_retained():
    """All targets retained -> target list preserved, no pruning."""
    chain = _simple_isomeric_chain()
    reduced = chain.reduce(['Ag109', 'Ag110', 'Ag110_m1'])

    assert reduced.isomeric_branching_targets is not None
    assert reduced.isomeric_branching_targets['Ag109']['(n,gamma)'] == \
        ['Ag110', 'Ag110_m1']
    assert reduced.reduce_pruned_targets is None


def test_reduce_drops_parent_excluded():
    """Parent excluded -> isomeric entry dropped."""
    chain = _simple_isomeric_chain()
    reduced = chain.reduce(['Ag110', 'Ag110_m1'])

    if reduced.isomeric_branching_targets is not None:
        assert 'Ag109' not in reduced.isomeric_branching_targets
    else:
        assert reduced.isomeric_branching_targets is None


def test_reduce_drops_all_targets_excluded():
    """level=0 keeps only the parent and drops its isomeric entry."""
    chain = _simple_isomeric_chain()
    reduced = chain.reduce(['Ag109'], level=0, keep_isomeric_siblings=False)

    assert len(reduced.nuclides) == 1
    assert reduced.nuclides[0].name == 'Ag109'
    if (reduced.isomeric_branching_targets is not None and
            'Ag109' in reduced.isomeric_branching_targets):
        assert '(n,gamma)' not in reduced.isomeric_branching_targets['Ag109']


def test_reduce_prunes_partial_exclusion():
    """Excluded targets are pruned and tracked in reduce_pruned_targets."""
    chain = _simple_isomeric_chain()
    reduced = chain.reduce(['Ag109', 'Ag110'], keep_isomeric_siblings=False)

    assert reduced.isomeric_branching_targets['Ag109']['(n,gamma)'] == ['Ag110']
    assert reduced.reduce_pruned_targets is not None
    assert reduced.reduce_pruned_targets['Ag109']['(n,gamma)'] == ['Ag110_m1']


def test_reduce_prunes_two_of_three():
    """Two of three targets kept; the third is pruned."""
    chain = _three_target_chain()
    reduced = chain.reduce(['Cd110', 'Cd111', 'Cd111_m2'],
                           keep_isomeric_siblings=False)

    assert reduced.isomeric_branching_targets['Cd110']['(n,gamma)'] == \
        ['Cd111', 'Cd111_m2']
    assert reduced.reduce_pruned_targets['Cd110']['(n,gamma)'] == ['Cd111_m1']


def test_reduce_handles_multiple_reactions():
    """Multiple reactions prune independently."""
    chain = _multi_reaction_chain()
    reduced = chain.reduce(['Ag109', 'Ag110', 'Ag108', 'Ag108_m1'],
                           keep_isomeric_siblings=False)

    assert reduced.isomeric_branching_targets['Ag109']['(n,gamma)'] == ['Ag110']
    assert set(reduced.isomeric_branching_targets['Ag109']['(n,2n)']) == \
        {'Ag108', 'Ag108_m1'}


def test_reduce_backward_compatible_no_isomeric():
    """reduce works for chains without isomeric data."""
    chain = openmc.deplete.Chain()
    u235 = openmc.deplete.Nuclide('U235')
    u235.add_reaction('(n,gamma)', 'U236', Q=6.5e6, branching_ratio=1.0)
    chain.add_nuclide(u235)
    chain.add_nuclide(openmc.deplete.Nuclide('U236'))
    chain.isomeric_branching_targets = None

    reduced = chain.reduce(['U235', 'U236'])

    assert reduced.isomeric_branching_targets is None
    assert reduced.reduce_pruned_targets is None


def test_reduce_level_zero():
    """level=0 reduction drops targets when siblings excluded."""
    chain = _simple_isomeric_chain()
    reduced = chain.reduce(['Ag109'], level=0, keep_isomeric_siblings=False)

    assert len(reduced.nuclides) == 1
    assert reduced.nuclides[0].name == 'Ag109'
    if (reduced.isomeric_branching_targets is not None and
            'Ag109' in reduced.isomeric_branching_targets):
        assert '(n,gamma)' not in reduced.isomeric_branching_targets['Ag109']


def test_form_matrix_after_reduce_with_isomeric():
    """form_matrix works after reduce with runtime isomeric branching."""
    chain = _simple_isomeric_chain()
    reduced = chain.reduce(['Ag109', 'Ag110'], keep_isomeric_siblings=False)

    nuclides = [n.name for n in reduced.nuclides]
    rates = ReactionRates(['mat1'], nuclides, list(reduced.reactions))
    rates[:] = 1e-10

    isomeric_branching = {'Ag109': {'(n,gamma)': {'Ag110': 1.0}}}
    matrix = reduced.form_matrix(rates[0], isomeric_branching=isomeric_branching)
    assert matrix is not None
    assert matrix.shape[0] == len(nuclides)


def test_reduce_all_retained():
    """All targets retained -> LFS unchanged."""
    reduced = _lfs_chain().reduce(['Ir191', 'Ir192', 'Ir192_m1', 'Ir192_m2'])

    assert reduced.isomeric_branching_lfs is not None
    assert reduced.isomeric_branching_lfs['Ir191']['(n,gamma)'] == [0, 3, 15]


def test_reduce_one_target_removed():
    """Removing Ir192_m2 drops its LFS=15 from the parallel list."""
    reduced = _lfs_chain().reduce(['Ir191', 'Ir192', 'Ir192_m1'],
                                  keep_isomeric_siblings=False)

    assert reduced.isomeric_branching_targets['Ir191']['(n,gamma)'] == \
        ['Ir192', 'Ir192_m1']
    assert reduced.isomeric_branching_lfs['Ir191']['(n,gamma)'] == [0, 3]


def test_reduce_two_targets_removed():
    """Removing both metastables leaves only the ground-state LFS=0."""
    reduced = _lfs_chain().reduce(['Ir191', 'Ir192'],
                                  keep_isomeric_siblings=False)

    assert reduced.isomeric_branching_targets['Ir191']['(n,gamma)'] == ['Ir192']
    assert reduced.isomeric_branching_lfs['Ir191']['(n,gamma)'] == [0]


def test_reduce_parent_excluded():
    """Excluding the parent removes its entry from targets and LFS."""
    reduced = _lfs_chain().reduce(['Ir192', 'Ir192_m1', 'Ir192_m2'])

    if reduced.isomeric_branching_targets is not None:
        assert 'Ir191' not in reduced.isomeric_branching_targets
    if reduced.isomeric_branching_lfs is not None:
        assert 'Ir191' not in reduced.isomeric_branching_lfs


def test_reduce_no_lfs_data():
    """Partial prune on a no-LFS chain works and leaves lfs == None."""
    reduced = _no_lfs_chain().reduce(['Ag109', 'Ag110'],
                                     keep_isomeric_siblings=False)

    assert reduced.isomeric_branching_targets['Ag109']['(n,gamma)'] == ['Ag110']
    assert reduced.isomeric_branching_lfs is None


def test_reduce_multi_reaction_partial():
    """Multi-reaction partial prune keeps LFS aligned with targets."""
    reduced = _multi_reaction_lfs_chain().reduce(
        ['Ag109', 'Ag110', 'Ag110_m1'], keep_isomeric_siblings=False)

    # (n,gamma): both targets retained (in initial list)
    assert reduced.isomeric_branching_targets['Ag109']['(n,gamma)'] == \
        ['Ag110', 'Ag110_m1']
    assert reduced.isomeric_branching_lfs['Ag109']['(n,gamma)'] == [0, 5]

    # (n,2n): Ag108 retained (followed via reaction path), Ag108_m1 pruned
    assert reduced.isomeric_branching_targets['Ag109']['(n,2n)'] == ['Ag108']
    assert reduced.isomeric_branching_lfs['Ag109']['(n,2n)'] == [0]


def test_reduce_roundtrip_after_prune():
    """A reduced chain with pruned LFS survives an XML round-trip."""
    reduced = _lfs_chain().reduce(['Ir191', 'Ir192', 'Ir192_m1'],
                                  keep_isomeric_siblings=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "reduced.xml"
        reduced.export_to_xml(xml_path)
        reloaded = openmc.deplete.Chain.from_xml(xml_path)

        assert reloaded.isomeric_branching_lfs['Ir191']['(n,gamma)'] == [0, 3]
        assert reloaded.isomeric_branching_targets['Ir191']['(n,gamma)'] == \
            ['Ir192', 'Ir192_m1']


# ===========================================================================
# Sibling expansion (keep_isomeric_siblings)
# ===========================================================================

def test_reduce_siblings_policy_false():
    """policy=False does not expand siblings (original behavior)."""
    chain = create_chain_with_siblings()
    reduced = chain.reduce(['Ag108'], level=1, keep_isomeric_siblings=False)

    assert len(reduced.nuclides) == 2
    nuclide_names = {n.name for n in reduced.nuclides}
    assert nuclide_names == {'Ag108', 'Ag109'}
    assert 'Ag109_m1' not in nuclide_names


def test_reduce_siblings_policy_true():
    """policy=True includes the isomeric sibling (Ag109_m1)."""
    chain = create_chain_with_siblings()
    reduced = chain.reduce(['Ag108'], level=1, keep_isomeric_siblings=True)

    assert len(reduced.nuclides) == 3
    assert {n.name for n in reduced.nuclides} == {'Ag108', 'Ag109', 'Ag109_m1'}


def test_reduce_siblings_true_no_branching():
    """policy=True is a no-op without isomeric branching metadata (vanilla parity)."""
    chain = openmc.deplete.Chain()

    cd111 = openmc.deplete.Nuclide('Cd111')
    cd111.add_reaction('(n,gamma)', 'Cd112', Q=5e6, branching_ratio=1.0)
    chain.add_nuclide(cd111)

    cd112 = openmc.deplete.Nuclide('Cd112')
    chain.add_nuclide(cd112)

    cd112_m1 = openmc.deplete.Nuclide('Cd112_m1')
    cd112_m1.half_life = 1000
    chain.add_nuclide(cd112_m1)

    chain.isomeric_branching_targets = None

    reduced = chain.reduce(['Cd111'], level=1, keep_isomeric_siblings=True)
    nuclide_names = {n.name for n in reduced.nuclides}
    assert nuclide_names == {'Cd111', 'Cd112'}
    assert 'Cd112_m1' not in nuclide_names


def test_reduce_siblings_invalid_type():
    """Non-bool keep_isomeric_siblings raises TypeError."""
    chain = create_chain_with_siblings()

    with pytest.raises(TypeError, match="keep_isomeric_siblings must be bool"):
        chain.reduce(['Ag108'], level=1, keep_isomeric_siblings='invalid')

    with pytest.raises(TypeError, match="keep_isomeric_siblings must be bool"):
        chain.reduce(['Ag108'], level=1,
                     keep_isomeric_siblings='isomeric_branch_siblings')


def test_siblings_pathways_followed():
    """Pathways from added siblings are followed (Ag110_m1 -> Ag111_m1)."""
    chain = create_complex_sibling_chain()
    reduced = chain.reduce(['Ag109'], level=2, keep_isomeric_siblings=True)

    nuclide_names = {n.name for n in reduced.nuclides}
    assert nuclide_names == {'Ag109', 'Ag110', 'Ag110_m1', 'Ag111', 'Ag111_m1'}

    ag110_m1 = reduced['Ag110_m1']
    assert any(r.target == 'Ag111_m1' for r in ag110_m1.reactions)


def test_siblings_with_depth_limit():
    """Sibling expansion pulls isomeric targets in even beyond the depth limit."""
    chain = create_complex_sibling_chain()
    reduced = chain.reduce(['Ag109'], level=1, keep_isomeric_siblings=True)

    nuclide_names = {n.name for n in reduced.nuclides}
    assert nuclide_names == {'Ag109', 'Ag110', 'Ag110_m1', 'Ag111', 'Ag111_m1'}


def test_siblings_isomeric_data_preserved():
    """Isomeric branching data is preserved for expanded siblings."""
    chain = create_complex_sibling_chain()
    reduced = chain.reduce(['Ag109'], level=2, keep_isomeric_siblings=True)

    assert reduced.isomeric_branching_targets is not None
    assert 'Ag109' in reduced.isomeric_branching_targets
    assert 'Ag110' in reduced.isomeric_branching_targets
    assert set(reduced.isomeric_branching_targets['Ag110']['(n,gamma)']) == \
        {'Ag111', 'Ag111_m1'}


def test_siblings_export_import_roundtrip():
    """Chains with expanded siblings export and reload with all nuclides intact."""
    chain = create_chain_with_siblings()
    reduced = chain.reduce(['Ag108'], level=1, keep_isomeric_siblings=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "sibling_chain.xml"
        reduced.export_to_xml(xml_path)
        reloaded = openmc.deplete.Chain.from_xml(xml_path)

        assert {n.name for n in reduced.nuclides} == \
            {n.name for n in reloaded.nuclides}
        if reduced.isomeric_branching_targets:
            assert reloaded.isomeric_branching_targets is not None
            for parent in reduced.isomeric_branching_targets:
                assert parent in reloaded.isomeric_branching_targets


def test_default_keeps_all_siblings():
    """Default keep_isomeric_siblings=True keeps all siblings."""
    chain = create_chain_with_siblings()
    reduced = chain.reduce(['Ag108'], level=1)

    assert {n.name for n in reduced.nuclides} == {'Ag108', 'Ag109', 'Ag109_m1'}


def test_siblings_form_matrix():
    """form_matrix works with expanded siblings."""
    chain = create_chain_with_siblings()
    reduced = chain.reduce(['Ag108'], level=1, keep_isomeric_siblings=True)

    nuclide_names = [n.name for n in reduced.nuclides]
    rates = ReactionRates(['mat1'], nuclide_names, list(reduced.reactions))
    rates[:] = 1e-10

    matrix = reduced.form_matrix(rates[0], isomeric_branching=None)
    assert matrix.shape[0] == len(reduced.nuclides)


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


# ===========================================================================
# Trailing (non-chain) nuclide ordering
# ===========================================================================

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
    model_nuclides = {'U235', 'Zr90', 'Ba140', 'Kr85', 'H1'}

    _, _, nuclides = OpenMCOperator._get_burnable_mats(
        _fake_operator(chain_nuclides, model_nuclides))

    assert nuclides[:3] == ['H1', 'He4', 'U235']
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
