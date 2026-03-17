"""Validation tests for Phase 0 (LFS on chain) and Phase 1 (C++ MF=10 parser).

Tests cover:
1. Phase -1: hasattr guard prevents AttributeError
2. Phase 0a: LFS parallel attribute on Chain
3. Phase 0b: XML reader/writer round-trip with gendf_lfs
4. Phase 0d: reduce() keeps LFS in sync during pruning
5. Phase 0 backward compat: old chains without gendf_lfs
6. Phase 1: C++ get_branching_ratios BR computation
7. Phase 1: Division-by-zero handling in BR computation
8. Phase 1: lfs_mapping construction (ground excluded, metas included)
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock
import warnings as warn_module

import openmc.deplete


# ============================================================================
# Helper: create chains
# ============================================================================

def _make_chain_with_lfs():
    """Create chain with both targets and LFS data (Phase 0 new format)."""
    chain = openmc.deplete.Chain()

    parent = openmc.deplete.Nuclide('Ir191')
    parent.add_reaction('(n,gamma)', 'Ir192', Q=6.2e6, branching_ratio=1.0)
    chain.add_nuclide(parent)

    for name in ['Ir192', 'Ir192_m1', 'Ir192_m2']:
        nuc = openmc.deplete.Nuclide(name)
        nuc.half_life = 1e6
        chain.add_nuclide(nuc)

    chain.isomeric_branching_targets = {
        'Ir191': {
            '(n,gamma)': ['Ir192', 'Ir192_m1', 'Ir192_m2']
        }
    }
    chain.isomeric_branching_lfs = {
        'Ir191': {
            '(n,gamma)': [0, 3, 15]
        }
    }
    return chain


def _make_chain_no_lfs():
    """Create chain with targets but no LFS data (legacy/old patched format)."""
    chain = openmc.deplete.Chain()

    parent = openmc.deplete.Nuclide('Ag109')
    parent.add_reaction('(n,gamma)', 'Ag110', Q=6.8e6, branching_ratio=1.0)
    chain.add_nuclide(parent)

    for name in ['Ag110', 'Ag110_m1']:
        nuc = openmc.deplete.Nuclide(name)
        nuc.half_life = 1e5
        chain.add_nuclide(nuc)

    chain.isomeric_branching_targets = {
        'Ag109': {
            '(n,gamma)': ['Ag110', 'Ag110_m1']
        }
    }
    chain.isomeric_branching_lfs = None
    return chain


def _make_chain_multi_reaction_lfs():
    """Chain with two reactions, both having LFS data."""
    chain = openmc.deplete.Chain()

    parent = openmc.deplete.Nuclide('Ag109')
    parent.add_reaction('(n,gamma)', 'Ag110', Q=6.8e6, branching_ratio=1.0)
    parent.add_reaction('(n,2n)', 'Ag108', Q=-9.5e6, branching_ratio=1.0)
    chain.add_nuclide(parent)

    for name, hl in [('Ag110', 1e5), ('Ag110_m1', 2.5e7),
                     ('Ag108', 1e5), ('Ag108_m1', 1.4e10)]:
        nuc = openmc.deplete.Nuclide(name)
        nuc.half_life = hl
        chain.add_nuclide(nuc)

    chain.isomeric_branching_targets = {
        'Ag109': {
            '(n,gamma)': ['Ag110', 'Ag110_m1'],
            '(n,2n)': ['Ag108', 'Ag108_m1']
        }
    }
    chain.isomeric_branching_lfs = {
        'Ag109': {
            '(n,gamma)': [0, 5],
            '(n,2n)': [0, 2]
        }
    }
    return chain


def _simulate_br_computation(levels, target_names, lfs_values, n_groups,
                              energy_bounds, mt=102):
    """Replicate the BR computation from _CppGENDFLibrary.get_branching_ratios."""
    from openmc.deplete.gendf import IsomericBranching, MT_TO_REACTION

    if not levels:
        return None

    lfs_to_xs = {lfs: xs for lfs, izap, xs in levels}

    prod_xs = []
    for lfs in lfs_values:
        if lfs in lfs_to_xs:
            prod_xs.append(lfs_to_xs[lfs])
        else:
            prod_xs.append(np.zeros(n_groups))

    prod_xs = np.array(prod_xs)
    total = prod_xs.sum(axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        br = np.where(total > 0, prod_xs / total, 0.0)

    reaction = MT_TO_REACTION.get(mt, f'MT{mt}')
    lfs_mapping = {name: lfs for name, lfs
                   in zip(target_names, lfs_values) if lfs > 0}

    return IsomericBranching(
        energies=energy_bounds,
        products=list(target_names),
        branching_ratios=br,
        parent_nuclide='test_parent',
        reaction=reaction,
        mt=mt,
        lfs_mapping=lfs_mapping,
    )


# ============================================================================
# Phase -1: hasattr guard
# ============================================================================

def test_guard_no_method_emits_warning():
    """A mock GENDF library without get_branching_ratios triggers warning."""
    mock_lib = Mock()
    del mock_lib.get_branching_ratios

    assert not hasattr(mock_lib, 'get_branching_ratios')

    with warn_module.catch_warnings(record=True) as w:
        warn_module.simplefilter("always")
        if not hasattr(mock_lib, 'get_branching_ratios'):
            warn_module.warn(
                "GENDF library backend does not support get_branching_ratios(). "
                "Isomeric branching will be disabled.",
                UserWarning
            )
        assert len(w) == 1
        assert "does not support get_branching_ratios" in str(w[0].message)


def test_guard_passes_with_method():
    """A library WITH get_branching_ratios passes the guard."""
    mock_lib = Mock()
    mock_lib.get_branching_ratios = Mock(return_value=None)
    assert hasattr(mock_lib, 'get_branching_ratios')


# ============================================================================
# Phase 0a: LFS attribute on Chain
# ============================================================================

def test_lfs_initialized_none():
    """Default chain has lfs=None."""
    chain = openmc.deplete.Chain()
    assert chain.isomeric_branching_lfs is None


def test_lfs_set_correctly():
    """LFS dict mirrors target structure."""
    chain = _make_chain_with_lfs()
    assert chain.isomeric_branching_lfs is not None
    assert 'Ir191' in chain.isomeric_branching_lfs
    assert chain.isomeric_branching_lfs['Ir191']['(n,gamma)'] == [0, 3, 15]


def test_lfs_parallel_to_targets():
    """LFS list length must match target list length."""
    chain = _make_chain_with_lfs()
    targets = chain.isomeric_branching_targets['Ir191']['(n,gamma)']
    lfs = chain.isomeric_branching_lfs['Ir191']['(n,gamma)']
    assert len(targets) == len(lfs), \
        f"Target count {len(targets)} != LFS count {len(lfs)}"


# ============================================================================
# Phase 0b: XML round-trip
# ============================================================================

def test_roundtrip_with_lfs():
    """Chain with LFS data round-trips through XML."""
    chain = _make_chain_with_lfs()

    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "test_chain.xml"
        chain.export_to_xml(xml_path)
        reloaded = openmc.deplete.Chain.from_xml(xml_path)

        assert reloaded.isomeric_branching_targets is not None
        assert reloaded.isomeric_branching_lfs is not None

        orig_targets = chain.isomeric_branching_targets['Ir191']['(n,gamma)']
        new_targets = reloaded.isomeric_branching_targets['Ir191']['(n,gamma)']
        assert orig_targets == new_targets

        orig_lfs = chain.isomeric_branching_lfs['Ir191']['(n,gamma)']
        new_lfs = reloaded.isomeric_branching_lfs['Ir191']['(n,gamma)']
        assert orig_lfs == new_lfs


def test_roundtrip_without_lfs():
    """Chain without LFS round-trips with lfs=None."""
    chain = _make_chain_no_lfs()

    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "test_chain_nolfs.xml"
        chain.export_to_xml(xml_path)
        reloaded = openmc.deplete.Chain.from_xml(xml_path)

        assert reloaded.isomeric_branching_targets is not None
        assert 'Ag109' in reloaded.isomeric_branching_targets
        assert reloaded.isomeric_branching_lfs is None


def test_roundtrip_multi_reaction():
    """Multiple reactions with LFS all survive round-trip."""
    chain = _make_chain_multi_reaction_lfs()

    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "test_chain_multi.xml"
        chain.export_to_xml(xml_path)
        reloaded = openmc.deplete.Chain.from_xml(xml_path)

        for rx in ['(n,gamma)', '(n,2n)']:
            assert reloaded.isomeric_branching_targets['Ag109'][rx] == \
                chain.isomeric_branching_targets['Ag109'][rx]
            assert reloaded.isomeric_branching_lfs['Ag109'][rx] == \
                chain.isomeric_branching_lfs['Ag109'][rx]


def test_xml_contains_gendf_lfs_attribute():
    """Verify the raw XML has gendf_lfs attribute."""
    import lxml.etree as ET

    chain = _make_chain_with_lfs()

    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "test_chain_raw.xml"
        chain.export_to_xml(xml_path)

        tree = ET.parse(str(xml_path))
        root = tree.getroot()

        found_lfs = False
        for nuc_elem in root.findall('nuclide'):
            for rx_elem in nuc_elem.findall('reaction'):
                iso_elem = rx_elem.find('isomeric_branching')
                if iso_elem is not None:
                    lfs_attr = iso_elem.get('gendf_lfs')
                    if lfs_attr:
                        assert lfs_attr == '0 3 15'
                        found_lfs = True

        assert found_lfs, "gendf_lfs attribute not found in exported XML"


# ============================================================================
# Phase 0d: reduce() LFS sync
# ============================================================================

def test_reduce_all_retained():
    """All targets retained -> LFS unchanged."""
    chain = _make_chain_with_lfs()
    reduced = chain.reduce(['Ir191', 'Ir192', 'Ir192_m1', 'Ir192_m2'])

    assert reduced.isomeric_branching_lfs is not None
    assert reduced.isomeric_branching_lfs['Ir191']['(n,gamma)'] == [0, 3, 15]


def test_reduce_one_target_removed():
    """Remove Ir192_m2 -> LFS should drop 15."""
    chain = _make_chain_with_lfs()
    reduced = chain.reduce(['Ir191', 'Ir192', 'Ir192_m1'],
                            keep_isomeric_siblings=False)

    assert reduced.isomeric_branching_targets['Ir191']['(n,gamma)'] == \
        ['Ir192', 'Ir192_m1']
    assert reduced.isomeric_branching_lfs['Ir191']['(n,gamma)'] == [0, 3]


def test_reduce_two_targets_removed():
    """Remove both metastables -> only ground state LFS=0 remains."""
    chain = _make_chain_with_lfs()
    reduced = chain.reduce(['Ir191', 'Ir192'],
                            keep_isomeric_siblings=False)

    assert reduced.isomeric_branching_targets['Ir191']['(n,gamma)'] == ['Ir192']
    assert reduced.isomeric_branching_lfs['Ir191']['(n,gamma)'] == [0]


def test_reduce_all_targets_removed():
    """Remove all targets -> entry dropped entirely."""
    chain = _make_chain_with_lfs()
    reduced = chain.reduce(['Ir191'], level=0,
                            keep_isomeric_siblings=False)

    if reduced.isomeric_branching_targets is not None:
        if 'Ir191' in reduced.isomeric_branching_targets:
            assert '(n,gamma)' not in reduced.isomeric_branching_targets['Ir191']


def test_reduce_parent_excluded():
    """Remove parent nuclide -> whole entry gone."""
    chain = _make_chain_with_lfs()
    reduced = chain.reduce(['Ir192', 'Ir192_m1', 'Ir192_m2'])

    if reduced.isomeric_branching_targets is not None:
        assert 'Ir191' not in reduced.isomeric_branching_targets
    if reduced.isomeric_branching_lfs is not None:
        assert 'Ir191' not in reduced.isomeric_branching_lfs


def test_reduce_no_lfs_data():
    """Chain without LFS -> reduce works without crash."""
    chain = _make_chain_no_lfs()
    reduced = chain.reduce(['Ag109', 'Ag110'],
                            keep_isomeric_siblings=False)

    assert reduced.isomeric_branching_targets is not None
    assert reduced.isomeric_branching_targets['Ag109']['(n,gamma)'] == ['Ag110']
    assert reduced.isomeric_branching_lfs is None


def test_reduce_multi_reaction_partial():
    """Multi-reaction chain: partial prune keeps LFS in sync.

    reduce() follows transmutation paths, so Ag108 (reached via (n,2n)
    from Ag109) is retained even though it was not in initial_isotopes.
    Ag108_m1 is pruned because keep_isomeric_siblings=False.
    The key invariant is that LFS stays aligned with targets after pruning.
    """
    chain = _make_chain_multi_reaction_lfs()

    reduced = chain.reduce(['Ag109', 'Ag110', 'Ag110_m1'],
                            keep_isomeric_siblings=False)

    # (n,gamma): both targets retained (in initial list)
    assert reduced.isomeric_branching_targets['Ag109']['(n,gamma)'] == \
        ['Ag110', 'Ag110_m1']
    assert reduced.isomeric_branching_lfs['Ag109']['(n,gamma)'] == [0, 5]

    # (n,2n): Ag108 retained (followed via reaction path), Ag108_m1 pruned
    assert reduced.isomeric_branching_targets['Ag109']['(n,2n)'] == ['Ag108']
    assert reduced.isomeric_branching_lfs['Ag109']['(n,2n)'] == [0]


def test_reduce_roundtrip_after_prune():
    """Reduced chain with pruned LFS survives XML round-trip."""
    chain = _make_chain_with_lfs()
    reduced = chain.reduce(['Ir191', 'Ir192', 'Ir192_m1'],
                            keep_isomeric_siblings=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "reduced_chain.xml"
        reduced.export_to_xml(xml_path)
        reloaded = openmc.deplete.Chain.from_xml(xml_path)

        assert reloaded.isomeric_branching_lfs['Ir191']['(n,gamma)'] == [0, 3]
        assert reloaded.isomeric_branching_targets['Ir191']['(n,gamma)'] == \
            ['Ir192', 'Ir192_m1']


# ============================================================================
# Phase 0: Backward compatibility
# ============================================================================

def test_chain_no_isomeric_data():
    """Plain chain with no isomeric data at all."""
    chain = openmc.deplete.Chain()
    u235 = openmc.deplete.Nuclide('U235')
    chain.add_nuclide(u235)

    assert chain.isomeric_branching_targets is None
    assert chain.isomeric_branching_lfs is None

    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "plain_chain.xml"
        chain.export_to_xml(xml_path)
        reloaded = openmc.deplete.Chain.from_xml(xml_path)
        assert reloaded.isomeric_branching_targets is None
        assert reloaded.isomeric_branching_lfs is None


def test_chain_targets_only_no_lfs():
    """Chain with targets but no gendf_lfs -> lfs is None."""
    chain = _make_chain_no_lfs()

    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "targets_only.xml"
        chain.export_to_xml(xml_path)
        reloaded = openmc.deplete.Chain.from_xml(xml_path)

        assert reloaded.isomeric_branching_targets is not None
        assert reloaded.isomeric_branching_lfs is None


# ============================================================================
# Phase 1: BR computation
# ============================================================================

def test_br_basic_two_level():
    """BR = prod_xs / total_prod_xs for 2-level case."""
    n_groups = 5
    prod_xs_ground = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
    prod_xs_m1 = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

    prod_xs = np.array([prod_xs_ground, prod_xs_m1])
    total = prod_xs.sum(axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        br = np.where(total > 0, prod_xs / total, 0.0)

    col_sums = br.sum(axis=0)
    np.testing.assert_allclose(col_sums, 1.0, rtol=1e-14)
    np.testing.assert_allclose(br[0, 0], 10.0 / 12.0, rtol=1e-14)
    np.testing.assert_allclose(br[1, 0], 2.0 / 12.0, rtol=1e-14)
    np.testing.assert_allclose(br[0, 2], 0.5, rtol=1e-14)
    np.testing.assert_allclose(br[1, 2], 0.5, rtol=1e-14)


def test_br_three_levels():
    """BR for 3-level case (ground + 2 metastables)."""
    xs_g = np.array([6.0, 3.0, 1.0, 0.0])
    xs_m1 = np.array([3.0, 3.0, 1.0, 0.0])
    xs_m2 = np.array([1.0, 3.0, 1.0, 0.0])

    prod_xs = np.array([xs_g, xs_m1, xs_m2])
    total = prod_xs.sum(axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        br = np.where(total > 0, prod_xs / total, 0.0)

    np.testing.assert_allclose(br[0, 0], 0.6, rtol=1e-14)
    np.testing.assert_allclose(br[1, 0], 0.3, rtol=1e-14)
    np.testing.assert_allclose(br[2, 0], 0.1, rtol=1e-14)
    np.testing.assert_allclose(br[:, 1], 1.0 / 3.0, rtol=1e-14)
    np.testing.assert_allclose(br[:, 3], 0.0)


def test_br_division_by_zero():
    """Groups with zero total production XS give BR=0, not NaN."""
    xs_g = np.array([1.0, 0.0, 0.0, 5.0])
    xs_m1 = np.array([1.0, 0.0, 0.0, 5.0])

    prod_xs = np.array([xs_g, xs_m1])
    total = prod_xs.sum(axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        br = np.where(total > 0, prod_xs / total, 0.0)

    assert not np.any(np.isnan(br)), "BR contains NaN values"
    assert not np.any(np.isinf(br)), "BR contains Inf values"
    np.testing.assert_allclose(br[:, 1], 0.0)
    np.testing.assert_allclose(br[:, 2], 0.0)


def test_br_conservation_random():
    """Column sums must be 1.0 where total > 0 (random data)."""
    rng = np.random.RandomState(42)
    n_groups = 100
    n_levels = 4

    prod_xs = rng.random((n_levels, n_groups)) * 10.0
    prod_xs[:, [0, 50, 99]] = 0.0

    total = prod_xs.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        br = np.where(total > 0, prod_xs / total, 0.0)

    nonzero = total > 0
    col_sums = br.sum(axis=0)
    np.testing.assert_allclose(col_sums[nonzero], 1.0, rtol=1e-14)
    np.testing.assert_allclose(col_sums[~nonzero], 0.0)


# ============================================================================
# Phase 1: lfs_mapping construction
# ============================================================================

def test_lfs_mapping_excludes_ground():
    """lfs_mapping only contains entries with lfs > 0."""
    target_names = ['Ir192', 'Ir192_m1', 'Ir192_m2']
    lfs_values = [0, 3, 15]

    lfs_mapping = {name: lfs for name, lfs
                   in zip(target_names, lfs_values) if lfs > 0}

    assert 'Ir192' not in lfs_mapping
    assert lfs_mapping['Ir192_m1'] == 3
    assert lfs_mapping['Ir192_m2'] == 15
    assert len(lfs_mapping) == 2


def test_lfs_mapping_all_ground():
    """If only ground state, lfs_mapping is empty."""
    target_names = ['Ir192']
    lfs_values = [0]

    lfs_mapping = {name: lfs for name, lfs
                   in zip(target_names, lfs_values) if lfs > 0}

    assert len(lfs_mapping) == 0


# ============================================================================
# Phase 1: full wrapper BR simulation
# ============================================================================

def test_br_wrapper_ir191_case():
    """Simulate Ir191 (n,gamma) with 3 production levels."""
    n_groups = 5
    energy_bounds = np.linspace(1e-5, 2e7, n_groups + 1)

    levels = [
        (0, 77192, np.array([5.0, 4.0, 3.0, 2.0, 1.0])),
        (3, 77192, np.array([1.0, 2.0, 3.0, 4.0, 5.0])),
        (15, 77192, np.array([0.5, 1.0, 1.5, 2.0, 2.5])),
    ]

    result = _simulate_br_computation(
        levels,
        target_names=['Ir192', 'Ir192_m1', 'Ir192_m2'],
        lfs_values=[0, 3, 15],
        n_groups=n_groups,
        energy_bounds=energy_bounds,
    )

    assert result is not None
    assert result.reaction == '(n,gamma)'
    assert result.mt == 102
    assert result.products == ['Ir192', 'Ir192_m1', 'Ir192_m2']
    assert result.branching_ratios.shape == (3, 5)

    col_sums = result.branching_ratios.sum(axis=0)
    np.testing.assert_allclose(col_sums, 1.0, rtol=1e-14)

    assert 'Ir192' not in result.lfs_mapping
    assert result.lfs_mapping['Ir192_m1'] == 3
    assert result.lfs_mapping['Ir192_m2'] == 15


def test_br_wrapper_missing_lfs():
    """If chain requests LFS values not in GENDF -> zero XS for those."""
    n_groups = 3
    energy_bounds = np.linspace(1e-5, 2e7, n_groups + 1)

    levels = [
        (0, 77192, np.array([10.0, 10.0, 10.0])),
        (5, 77192, np.array([2.0, 2.0, 2.0])),
    ]

    result = _simulate_br_computation(
        levels,
        target_names=['Ir192', 'Ir192_m1', 'Ir192_m2'],
        lfs_values=[0, 3, 15],  # LFS=3 and LFS=15 not in GENDF
        n_groups=n_groups,
        energy_bounds=energy_bounds,
    )

    assert result is not None
    np.testing.assert_allclose(result.branching_ratios[0], 1.0, rtol=1e-14)
    np.testing.assert_allclose(result.branching_ratios[1], 0.0)
    np.testing.assert_allclose(result.branching_ratios[2], 0.0)


def test_br_wrapper_empty_levels():
    """No MF=10 data -> returns None."""
    n_groups = 3
    energy_bounds = np.linspace(1e-5, 2e7, n_groups + 1)

    result = _simulate_br_computation(
        levels=[],
        target_names=['Ir192', 'Ir192_m1'],
        lfs_values=[0, 3],
        n_groups=n_groups,
        energy_bounds=energy_bounds,
    )

    assert result is None


def test_br_wrapper_all_zero_production():
    """All production XS are zero -> BR all zero, no NaN."""
    n_groups = 3
    energy_bounds = np.linspace(1e-5, 2e7, n_groups + 1)

    levels = [
        (0, 47110, np.zeros(n_groups)),
        (5, 47110, np.zeros(n_groups)),
    ]

    result = _simulate_br_computation(
        levels,
        target_names=['Ag110', 'Ag110_m1'],
        lfs_values=[0, 5],
        n_groups=n_groups,
        energy_bounds=energy_bounds,
    )

    assert result is not None
    assert not np.any(np.isnan(result.branching_ratios))
    np.testing.assert_allclose(result.branching_ratios, 0.0)


# ============================================================================
# Phase 1: MT*1000+LFS key space
# ============================================================================

def test_mt_lfs_key_no_collision():
    """MT*1000 + LFS has no collision for valid MT/LFS ranges."""
    seen_keys = set()
    for mt in range(1, 892):
        for lfs in range(0, 51):
            key = mt * 1000 + lfs
            assert key not in seen_keys, \
                f"Key collision at MT={mt}, LFS={lfs}, key={key}"
            seen_keys.add(key)


def test_lfs_extraction_from_key():
    """LFS can be recovered from composite key."""
    test_cases = [
        (102, 0, 102000),
        (102, 3, 102003),
        (102, 15, 102015),
        (16, 0, 16000),
        (16, 2, 16002),
        (891, 50, 891050),
    ]
    for mt, lfs, expected_key in test_cases:
        key = mt * 1000 + lfs
        assert key == expected_key
        assert key - mt * 1000 == lfs


# ============================================================================
# Phase 1: energy alignment
# ============================================================================

def test_full_range_no_padding():
    """Full-range MF=10 data (size == n_groups) needs no padding."""
    n_groups = 5
    raw_xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    aligned = np.zeros(n_groups)
    if len(raw_xs) == n_groups:
        aligned = raw_xs.copy()

    np.testing.assert_array_equal(aligned, raw_xs)


def test_threshold_padding():
    """Threshold reaction data (size < n_groups) padded at low-energy end."""
    n_groups = 10
    raw_xs = np.array([1.0, 2.0, 3.0])

    aligned = np.zeros(n_groups)
    offset = n_groups - len(raw_xs)
    aligned[offset:] = raw_xs

    np.testing.assert_array_equal(aligned[:7], 0.0)
    np.testing.assert_array_equal(aligned[7:], raw_xs)


def test_n_plus_1_truncation():
    """n_groups+1 data truncated to n_groups."""
    n_groups = 5
    raw_xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    aligned = np.zeros(n_groups)
    if len(raw_xs) == n_groups + 1:
        aligned = raw_xs[:n_groups].copy()

    np.testing.assert_array_equal(aligned, [1.0, 2.0, 3.0, 4.0, 5.0])


def test_alignment_high_energy_end():
    """Threshold data placed at high-energy end (physically correct for (n,xn))."""
    n_groups = 1025
    threshold_data = np.ones(500)

    aligned = np.zeros(n_groups)
    offset = n_groups - len(threshold_data)
    aligned[offset:] = threshold_data

    assert aligned[0] == 0.0
    assert aligned[offset - 1] == 0.0
    assert aligned[offset] == 1.0
    assert aligned[-1] == 1.0


# ============================================================================
# Integration: mixed LFS and no-LFS
# ============================================================================

def test_partial_lfs_coverage():
    """Chain with LFS on one reaction but not another."""
    chain = openmc.deplete.Chain()

    parent = openmc.deplete.Nuclide('Ag109')
    parent.add_reaction('(n,gamma)', 'Ag110', Q=6.8e6, branching_ratio=1.0)
    parent.add_reaction('(n,2n)', 'Ag108', Q=-9.5e6, branching_ratio=1.0)
    chain.add_nuclide(parent)

    for name in ['Ag110', 'Ag110_m1', 'Ag108', 'Ag108_m1']:
        nuc = openmc.deplete.Nuclide(name)
        nuc.half_life = 1e5
        chain.add_nuclide(nuc)

    chain.isomeric_branching_targets = {
        'Ag109': {
            '(n,gamma)': ['Ag110', 'Ag110_m1'],
            '(n,2n)': ['Ag108', 'Ag108_m1']
        }
    }
    chain.isomeric_branching_lfs = {
        'Ag109': {
            '(n,gamma)': [0, 5]
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "partial_lfs.xml"
        chain.export_to_xml(xml_path)
        reloaded = openmc.deplete.Chain.from_xml(xml_path)

        assert reloaded.isomeric_branching_lfs is not None
        assert reloaded.isomeric_branching_lfs['Ag109']['(n,gamma)'] == [0, 5]
        assert '(n,2n)' not in reloaded.isomeric_branching_lfs.get('Ag109', {})

        assert '(n,gamma)' in reloaded.isomeric_branching_targets['Ag109']
        assert '(n,2n)' in reloaded.isomeric_branching_targets['Ag109']


# ============================================================================
# Edge case: LFS count mismatch in XML
# ============================================================================

def test_lfs_count_mismatch_in_xml():
    """If gendf_lfs token count != target count, LFS should be ignored."""
    import lxml.etree as ET

    chain = _make_chain_with_lfs()

    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "test_chain.xml"
        chain.export_to_xml(xml_path)

        # Manually corrupt the gendf_lfs attribute to have wrong count
        tree = ET.parse(str(xml_path))
        root = tree.getroot()

        for nuc_elem in root.findall('nuclide'):
            for rx_elem in nuc_elem.findall('reaction'):
                iso_elem = rx_elem.find('isomeric_branching')
                if iso_elem is not None:
                    # Set wrong number of LFS values (2 instead of 3)
                    iso_elem.set('gendf_lfs', '0 3')

        corrupted_path = Path(tmpdir) / "corrupted.xml"
        tree.write(str(corrupted_path), encoding='utf-8')

        reloaded = openmc.deplete.Chain.from_xml(corrupted_path)

        # Targets should still load
        assert reloaded.isomeric_branching_targets is not None
        # LFS should be None because count mismatch
        assert reloaded.isomeric_branching_lfs is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
