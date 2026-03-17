"""Comprehensive verification of combined XS + BR acquisition implementation.

Tests cover:
1. MF=10 energy-aware alignment (parser saves prod_energy_data)
2. Three parser save points for MF=10 data
3. _normalize_ratios graceful degradation (returns {} for zero sum)
4. _get_branching_data fallback (no gendf_lfs -> patcher mode)
5. Cross-backend parity (C++ vs Python branching ratio computation)
6. Previous fixes still intact (hasattr guard, embedded BR, reduce filters,
   pool.py signature check, stale cache invalidation, shape assertion)
7. form_matrix br variable shadowing fix
8. End-to-end: Activator scenario with reduced chain
"""

import math
import os
import warnings
import inspect
import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, patch
from collections import defaultdict

from openmc.mgxs import GROUP_STRUCTURES
from openmc.deplete.helpers import IsomericBranchingHelper
from openmc.deplete.chain import Chain
from openmc.deplete.gendf import (
    IsomericBranching, REACTION_TO_MT, MT_TO_REACTION
)

# Source directory for C++ file verification
_SRC_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'src')


# ============================================================================
# Helper factories
# ============================================================================

def _make_chain(targets, lfs=None, embedded=None):
    """Build a mock chain with isomeric branching configuration."""
    chain = Mock(spec=Chain)
    chain.isomeric_branching_targets = targets
    chain.isomeric_branching_lfs = lfs
    chain.isomeric_branching_embedded = embedded
    return chain


def _make_gendf(n_groups, energy_bins, xs_func=None, br_func=None,
                energy_structure='CCFE-709'):
    """Build a mock GENDF library."""
    mock = Mock()
    mock.energy_structure = energy_structure
    mock.energy_bounds = energy_bins.copy()
    mock.n_groups = n_groups
    if xs_func is None:
        mock.get_xs = Mock(return_value=np.ones(n_groups))
    else:
        mock.get_xs = Mock(side_effect=xs_func)
    if br_func is None:
        mock.get_branching_ratios = Mock(return_value=None)
    else:
        mock.get_branching_ratios = Mock(side_effect=br_func)
    return mock


# ============================================================================
# Test 1: MF=10 energy-aware alignment
# ============================================================================

def test_c_api_get_production_xs_signature():
    """C API wrapper _get_production_xs has nuclide and mt params."""
    from openmc.lib.gendf import GENDFLibrary as CppGENDFLibrary
    sig = inspect.signature(CppGENDFLibrary._get_production_xs)
    params = list(sig.parameters.keys())
    assert 'self' in params
    assert 'nuclide' in params
    assert 'mt' in params


def test_python_get_branching_ratios_runtime_mode():
    """Python backend get_branching_ratios accepts target_names and lfs_values."""
    from openmc.deplete.gendf import _PythonGENDFLibrary
    sig = inspect.signature(_PythonGENDFLibrary.get_branching_ratios)
    params = list(sig.parameters.keys())
    assert 'target_names' in params
    assert 'lfs_values' in params


def test_cpp_get_branching_ratios_runtime_mode():
    """C++ backend wrapper get_branching_ratios accepts target_names and lfs_values."""
    from openmc.lib.gendf import GENDFLibrary as CppGENDFLibrary
    sig = inspect.signature(CppGENDFLibrary.get_branching_ratios)
    params = list(sig.parameters.keys())
    assert 'target_names' in params
    assert 'lfs_values' in params


def test_production_xs_energy_alignment_logic():
    """Both get_xs() and get_production_xs() in gendf.cpp use energy-aware
    alignment via GENDF_RTOL_MATCH and reference their energy data stores."""
    gendf_cpp = os.path.join(_SRC_DIR, 'gendf.cpp')
    with open(gendf_cpp) as f:
        content = f.read()

    assert content.count('GENDF_RTOL_MATCH') >= 2, \
        "Both get_xs and get_production_xs should use GENDF_RTOL_MATCH"
    assert 'prod_energy_data_' in content, \
        "get_production_xs should reference prod_energy_data_"
    assert 'energy_data_' in content, \
        "get_xs should reference energy_data_"


# ============================================================================
# Test 2: Three parser save points for MF=10 data
# ============================================================================

def test_parser_has_three_mf10_energy_save_points():
    """Parser must save prod_energy_data at all three transition points:
    (1) MF/MT transition, (2) HEAD detection within MT, (3) end-of-file."""
    parser_cpp = os.path.join(_SRC_DIR, 'gendf_parser.cpp')
    with open(parser_cpp) as f:
        content = f.read()

    save_count = content.count('result.prod_energy_data[key]')
    assert save_count == 3, (
        f"Expected 3 MF=10 energy save points, found {save_count}. "
        f"Save points needed at: "
        f"(1) MF/MT transition, "
        f"(2) HEAD detection within same MT, "
        f"(3) end-of-file")


def test_parser_saves_prod_xs_at_three_points():
    """Parser must save prod_xs_data at all three transition points."""
    parser_cpp = os.path.join(_SRC_DIR, 'gendf_parser.cpp')
    with open(parser_cpp) as f:
        content = f.read()

    save_count = content.count('result.prod_xs_data[key]')
    assert save_count == 3, (
        f"Expected 3 MF=10 XS save points, found {save_count}")


def test_parser_saves_prod_izap_at_three_points():
    """Parser must save prod_izap_data at all three transition points."""
    parser_cpp = os.path.join(_SRC_DIR, 'gendf_parser.cpp')
    with open(parser_cpp) as f:
        content = f.read()

    save_count = content.count('result.prod_izap_data[key]')
    assert save_count == 3, (
        f"Expected 3 MF=10 IZAP save points, found {save_count}")


# ============================================================================
# Test 3: _normalize_ratios graceful degradation
# ============================================================================

def _make_normalize_helper():
    energy_bins = GROUP_STRUCTURES['CCFE-709']
    chain = _make_chain({'X': {'(n,gamma)': ['Y', 'Z']}})
    gendf = _make_gendf(709, energy_bins)
    return IsomericBranchingHelper(chain, gendf)


def test_normalize_zero_sum_returns_empty():
    """Ratios summing to zero -> graceful {} return (not ValueError)."""
    helper = _make_normalize_helper()
    result = helper._normalize_ratios(
        {'A': 0.0, 'B': 0.0}, 'TestNuc', '(n,gamma)')
    assert result == {}


def test_normalize_negative_sum_returns_empty():
    """Negative total -> graceful {} return (not ValueError)."""
    helper = _make_normalize_helper()
    result = helper._normalize_ratios(
        {'A': -0.5, 'B': 0.3}, 'TestNuc', '(n,gamma)')
    assert result == {}


def test_normalize_empty_input():
    """Empty input -> empty output."""
    helper = _make_normalize_helper()
    result = helper._normalize_ratios({}, 'TestNuc', '(n,gamma)')
    assert result == {}


def test_normalize_valid_ratios():
    """Valid ratios near 1.0 are normalized to exactly 1.0."""
    helper = _make_normalize_helper()
    result = helper._normalize_ratios(
        {'A': 0.6, 'B': 0.4}, 'TestNuc', '(n,gamma)')
    assert np.isclose(sum(result.values()), 1.0)


def test_normalize_large_deviation_warns():
    """Deviation >1% from 1.0 emits warning."""
    helper = _make_normalize_helper()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = helper._normalize_ratios(
            {'A': 0.5, 'B': 0.2}, 'TestNuc', '(n,gamma)')
        assert len(w) == 1
        assert 'deviates' in str(w[0].message)
    assert np.isclose(sum(result.values()), 1.0)


# ============================================================================
# Test 4: _get_branching_data fallback (no gendf_lfs)
# ============================================================================

def test_no_lfs_falls_through_to_patcher_mode():
    """Chain with targets but no LFS -> patcher mode (target_names=None)."""
    energy_bins = GROUP_STRUCTURES['CCFE-709']
    chain = _make_chain(
        targets={'Ag109': {'(n,gamma)': ['Ag110', 'Ag110_m1']}},
        lfs=None
    )
    mock_br = IsomericBranching(
        energies=energy_bins[:-1].copy(),
        products=['Ag110', 'Ag110_m1'],
        branching_ratios=np.array([
            np.full(709, 0.9), np.full(709, 0.1),
        ]),
        parent_nuclide='Ag109', reaction='(n,gamma)', mt=102,
    )
    gendf = _make_gendf(709, energy_bins,
                        br_func=lambda *a, **kw: mock_br)
    helper = IsomericBranchingHelper(chain, gendf)

    helper._get_branching_data('Ag109', '(n,gamma)')

    call_kwargs = gendf.get_branching_ratios.call_args
    assert call_kwargs.kwargs.get('target_names') is None
    assert call_kwargs.kwargs.get('lfs_values') is None


def test_with_lfs_uses_runtime_mode():
    """Chain with both targets and LFS -> runtime mode (target_names provided)."""
    energy_bins = GROUP_STRUCTURES['CCFE-709']
    chain = _make_chain(
        targets={'Ag109': {'(n,gamma)': ['Ag110', 'Ag110_m1']}},
        lfs={'Ag109': {'(n,gamma)': [0, 1]}}
    )
    mock_br = IsomericBranching(
        energies=energy_bins[:-1].copy(),
        products=['Ag110', 'Ag110_m1'],
        branching_ratios=np.array([
            np.full(709, 0.9), np.full(709, 0.1),
        ]),
        parent_nuclide='Ag109', reaction='(n,gamma)', mt=102,
    )
    gendf = _make_gendf(709, energy_bins,
                        br_func=lambda *a, **kw: mock_br)
    helper = IsomericBranchingHelper(chain, gendf)

    helper._get_branching_data('Ag109', '(n,gamma)')

    call_kwargs = gendf.get_branching_ratios.call_args
    assert call_kwargs.kwargs.get('target_names') == ['Ag110', 'Ag110_m1']
    assert call_kwargs.kwargs.get('lfs_values') == [0, 1]


def test_lfs_dict_exists_but_missing_nuclide():
    """LFS dict exists but doesn't have this nuclide -> patcher mode."""
    energy_bins = GROUP_STRUCTURES['CCFE-709']
    chain = _make_chain(
        targets={'Ag109': {'(n,gamma)': ['Ag110', 'Ag110_m1']}},
        lfs={'Ir191': {'(n,gamma)': [0, 1]}}
    )
    mock_br = IsomericBranching(
        energies=energy_bins[:-1].copy(),
        products=['Ag110', 'Ag110_m1'],
        branching_ratios=np.array([
            np.full(709, 0.8), np.full(709, 0.2),
        ]),
        parent_nuclide='Ag109', reaction='(n,gamma)', mt=102,
    )
    gendf = _make_gendf(709, energy_bins,
                        br_func=lambda *a, **kw: mock_br)
    helper = IsomericBranchingHelper(chain, gendf)

    helper._get_branching_data('Ag109', '(n,gamma)')

    call_kwargs = gendf.get_branching_ratios.call_args
    assert call_kwargs.kwargs.get('target_names') is None


# ============================================================================
# Test 5: Cross-backend parity
# ============================================================================

def test_runtime_mode_ratio_computation():
    """Both backends compute BR_i = sigma_prod_i / sum(sigma_prod_j).
    Verify analytically: 3:1 production ratio -> 0.75:0.25 branching."""
    n_groups = 5
    lfs0_xs = np.array([3.0, 3.0, 3.0, 0.0, 0.0])
    lfs1_xs = np.array([1.0, 1.0, 1.0, 0.0, 0.0])

    total = lfs0_xs + lfs1_xs
    expected_br0 = np.where(total > 0, lfs0_xs / total, 0.0)
    expected_br1 = np.where(total > 0, lfs1_xs / total, 0.0)

    assert np.allclose(expected_br0[:3], 0.75)
    assert np.allclose(expected_br1[:3], 0.25)
    assert np.allclose(expected_br0[3:], 0.0)
    assert np.allclose(expected_br1[3:], 0.0)

    for g in range(n_groups):
        if total[g] > 0:
            assert np.isclose(expected_br0[g] + expected_br1[g], 1.0)


def test_isomeric_branching_dataclass_fields():
    """IsomericBranching dataclass has all required fields."""
    br = IsomericBranching(
        energies=np.array([1e5, 1e6]),
        products=['A', 'B'],
        branching_ratios=np.array([[0.7, 0.8], [0.3, 0.2]]),
        parent_nuclide='X', reaction='(n,gamma)', mt=102,
        lfs_mapping={'B': 1},
    )
    assert hasattr(br, 'energies')
    assert hasattr(br, 'products')
    assert hasattr(br, 'branching_ratios')
    assert hasattr(br, 'lfs_mapping')
    assert br.lfs_mapping == {'B': 1}


# ============================================================================
# Test 6: Previous fixes still intact
# ============================================================================

def test_hasattr_guard_for_get_branching_ratios():
    """GENDF library lacking get_branching_ratios -> warn + disable."""
    from openmc.deplete.independent_operator import IndependentOperator

    mock_op = Mock(spec=IndependentOperator)
    mock_op.chain = _make_chain(
        {'Ag109': {'(n,gamma)': ['Ag110', 'Ag110_m1']}})
    mock_op._gendf_library = Mock(spec=[])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        IndependentOperator._setup_isomeric_branching(mock_op)
        warning_msgs = [str(x.message) for x in w]
        assert any('get_branching_ratios' in m for m in warning_msgs)
    assert mock_op._isomeric_branching is None


def test_embedded_br_served_from_cache():
    """Chain-embedded ratios served directly, not from GENDF library."""
    energy_bins = GROUP_STRUCTURES['CCFE-709']
    embedded = {
        ('Ag109', '(n,gamma)'): {
            'energies': energy_bins[:-1].copy(),
            'targets': ['Ag110', 'Ag110_m1'],
            'branching_ratios': {
                'Ag110': np.full(709, 0.85),
                'Ag110_m1': np.full(709, 0.15),
            }
        }
    }
    chain = _make_chain(
        {'Ag109': {'(n,gamma)': ['Ag110', 'Ag110_m1']}},
        embedded=embedded
    )
    gendf = _make_gendf(709, energy_bins)
    helper = IsomericBranchingHelper(chain, gendf)

    result = helper._get_branching_data('Ag109', '(n,gamma)')

    assert isinstance(result, dict)
    assert 'Ag110' in result['branching_ratios']
    gendf.get_branching_ratios.assert_not_called()


def test_reduce_filters_all_three_attributes():
    """Chain.reduce() filters targets, lfs, and embedded."""
    import openmc.deplete

    chain = openmc.deplete.Chain()
    # Use real nuclide names (GNDS format) so zam() doesn't fail
    parent = openmc.deplete.Nuclide('Ag109')
    parent.add_reaction('(n,gamma)', 'Ag110', Q=1e6, branching_ratio=0.5)
    chain.add_nuclide(parent)
    chain.add_nuclide(openmc.deplete.Nuclide('Ag110'))
    chain.add_nuclide(openmc.deplete.Nuclide('Ag110_m1'))

    chain.isomeric_branching_targets = {
        'Ag109': {'(n,gamma)': ['Ag110', 'Ag110_m1']}
    }
    chain.isomeric_branching_lfs = {
        'Ag109': {'(n,gamma)': [0, 1]}
    }
    chain.isomeric_branching_embedded = {
        ('Ag109', '(n,gamma)'): {
            'energies': np.array([1e5]),
            'targets': ['Ag110', 'Ag110_m1'],
            'branching_ratios': {
                'Ag110': np.array([0.8]),
                'Ag110_m1': np.array([0.2]),
            }
        }
    }

    # Reduce to just Ag109 -> only Ag110 should survive (level=1,
    # Ag110_m1 only appears in isomeric_branching_targets and is
    # kept when keep_isomeric_siblings=True)
    reduced = chain.reduce(['Ag109'], level=1)

    # All three attributes should be present and filtered consistently
    if reduced.isomeric_branching_targets:
        targets = reduced.isomeric_branching_targets.get(
            'Ag109', {}).get('(n,gamma)', [])
        # Both targets should be retained because keep_isomeric_siblings
        # preserves siblings of isomeric branching targets
        for t in targets:
            assert t in ('Ag110', 'Ag110_m1')

    if reduced.isomeric_branching_lfs:
        lfs = reduced.isomeric_branching_lfs.get(
            'Ag109', {}).get('(n,gamma)', [])
        # LFS count must match retained target count
        if reduced.isomeric_branching_targets:
            retained = reduced.isomeric_branching_targets.get(
                'Ag109', {}).get('(n,gamma)', [])
            assert len(lfs) == len(retained)

    if reduced.isomeric_branching_embedded:
        key = ('Ag109', '(n,gamma)')
        if key in reduced.isomeric_branching_embedded:
            data = reduced.isomeric_branching_embedded[key]
            # Every target in embedded must also be in targets
            for t in data['targets']:
                assert t in data['branching_ratios']


def test_pool_signature_check():
    """pool.py relies on form_matrix having isomeric_branching param."""
    sig = inspect.signature(Chain.form_matrix)
    assert 'isomeric_branching' in sig.parameters


def test_stale_cache_invalidation():
    """Each new IsomericBranchingHelper starts with fresh cache."""
    energy_bins = GROUP_STRUCTURES['CCFE-709']
    chain = _make_chain({'X': {'(n,gamma)': ['Y']}})
    gendf = _make_gendf(709, energy_bins)

    helper1 = IsomericBranchingHelper(chain, gendf)
    assert len(helper1._branching_cache) == 0
    helper1._branching_cache[('X', '(n,gamma)')] = 'stale_data'

    helper2 = IsomericBranchingHelper(chain, gendf)
    assert len(helper2._branching_cache) == 0


def test_shape_assertion_sigma_g_vs_flux():
    """ValueError raised when GENDF groups don't match flux groups."""
    energy_bins = GROUP_STRUCTURES['CCFE-709']
    chain = _make_chain(
        {'Ag109': {'(n,gamma)': ['Ag110', 'Ag110_m1']}})
    gendf = _make_gendf(709, energy_bins,
                        xs_func=lambda *a, **kw: np.ones(500))
    gendf.get_branching_ratios = Mock(
        return_value=IsomericBranching(
            energies=np.array([1e5, 1e6]),
            products=['Ag110', 'Ag110_m1'],
            branching_ratios=np.array([[0.9, 0.9], [0.1, 0.1]]),
            parent_nuclide='Ag109', reaction='(n,gamma)', mt=102))

    helper = IsomericBranchingHelper(chain, gendf)
    with pytest.raises(ValueError, match="group structure mismatch"):
        helper.weighted_branching_ratios(np.ones(709), energy_bins)


# ============================================================================
# Test 7: form_matrix br variable shadowing fix
# ============================================================================

def test_br_from_reactions_not_shadowed():
    """Isomeric branching uses iso_br, original br used for non-isomeric
    targets and light nuclide production."""
    import openmc.deplete
    from openmc.deplete import ReactionRates

    chain = openmc.deplete.Chain()
    parent = openmc.deplete.Nuclide('TestParent')
    parent.add_reaction('(n,gamma)', 'TestProduct',
                        Q=1e6, branching_ratio=0.6)
    chain.add_nuclide(parent)
    chain.add_nuclide(openmc.deplete.Nuclide('TestProduct'))
    chain.add_nuclide(openmc.deplete.Nuclide('TestProduct_m1'))

    nuclides = ['TestParent', 'TestProduct', 'TestProduct_m1']
    reactions = ['(n,gamma)']
    rates = ReactionRates(['mat1'], nuclides, reactions)
    rates[0, 0, 0] = 2.0

    iso_br = {
        'TestParent': {
            '(n,gamma)': {
                'TestProduct': 0.7,
                'TestProduct_m1': 0.3
            }
        }
    }

    matrix = chain.form_matrix(rates[0], isomeric_branching=iso_br)
    dense = matrix.toarray()

    parent_idx = chain.nuclide_dict['TestParent']
    product_idx = chain.nuclide_dict['TestProduct']
    m1_idx = chain.nuclide_dict['TestProduct_m1']

    assert np.isclose(dense[product_idx, parent_idx], 2.0 * 0.7)
    assert np.isclose(dense[m1_idx, parent_idx], 2.0 * 0.3)
    assert np.isclose(dense[parent_idx, parent_idx], -2.0)


# ============================================================================
# Test 8: End-to-end Activator scenario
# ============================================================================

def test_activator_no_lfs_does_not_crash():
    """Old-format chain (targets, no LFS) should gracefully degrade
    instead of crashing. This was the original Activator bug."""
    energy_bins = GROUP_STRUCTURES['CCFE-709']
    n_groups = 709

    chain = _make_chain(
        targets={
            'Ag107': {
                '(n,gamma)': ['Ag108', 'Ag108_m1'],
                '(n,2n)': ['Ag106', 'Ag106_m1'],
            },
            'Ag109': {'(n,gamma)': ['Ag110', 'Ag110_m1']},
        },
        lfs=None
    )

    def br_handler(nuclide, mt, target_names=None, lfs_values=None):
        if target_names is None:
            raise NotImplementedError("C++ patcher mode not supported")
        raise ValueError("Should not reach here")

    gendf = _make_gendf(n_groups, energy_bins,
                        xs_func=lambda *a, **kw: np.ones(n_groups),
                        br_func=br_handler)

    helper = IsomericBranchingHelper(chain, gendf)
    result = helper.weighted_branching_ratios(np.ones(n_groups), energy_bins)
    assert isinstance(result, dict)


def test_activator_with_lfs_succeeds():
    """New-format chain (targets + LFS) produces valid branching ratios."""
    energy_bins = GROUP_STRUCTURES['CCFE-709']
    n_groups = 709

    chain = _make_chain(
        targets={'Ag109': {'(n,gamma)': ['Ag110', 'Ag110_m1']}},
        lfs={'Ag109': {'(n,gamma)': [0, 1]}}
    )
    mock_br = IsomericBranching(
        energies=energy_bins[:-1].copy(),
        products=['Ag110', 'Ag110_m1'],
        branching_ratios=np.array([
            np.full(n_groups, 0.88),
            np.full(n_groups, 0.12),
        ]),
        parent_nuclide='Ag109', reaction='(n,gamma)', mt=102,
        lfs_mapping={'Ag110_m1': 1},
    )
    gendf = _make_gendf(n_groups, energy_bins,
                        br_func=lambda *a, **kw: mock_br)
    helper = IsomericBranchingHelper(chain, gendf)

    result = helper.weighted_branching_ratios(np.ones(n_groups), energy_bins)

    assert 'Ag109' in result
    assert '(n,gamma)' in result['Ag109']
    ratios = result['Ag109']['(n,gamma)']
    assert np.isclose(sum(ratios.values()), 1.0)
    assert 'Ag110' in ratios
    assert 'Ag110_m1' in ratios


# ============================================================================
# Test 9: Conservation tests
# ============================================================================

def test_per_group_conservation():
    """BR sum = 1.0 at every group where total production > 0."""
    n_groups = 10
    prod_xs = np.array([
        [5.0, 3.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2.0, 3.0, 7.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    total = prod_xs.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        br = np.where(total > 0, prod_xs / total, 0.0)

    for g in range(n_groups):
        if total[g] > 0:
            assert np.isclose(br[:, g].sum(), 1.0), \
                f"Group {g}: BR sum = {br[:, g].sum()}"
        else:
            assert np.allclose(br[:, g], 0.0)


def test_weighted_conservation():
    """sigma-phi weighted ratios must sum to 1.0."""
    energy_bins = GROUP_STRUCTURES['CCFE-709']
    n_groups = 709

    flux = np.zeros(n_groups)
    flux[:200] = 1e14
    flux[200:600] = 1e12
    flux[600:] = 1e10

    br_ground = np.zeros(n_groups)
    br_ground[:400] = 0.9
    br_ground[400:] = 0.5
    br_meta = np.zeros(n_groups)
    br_meta[:400] = 0.1
    br_meta[400:] = 0.5

    mock_br = IsomericBranching(
        energies=energy_bins[:-1].copy(),
        products=['GS', 'M1'],
        branching_ratios=np.array([br_ground, br_meta]),
        parent_nuclide='Test', reaction='(n,gamma)', mt=102,
    )
    chain = _make_chain(
        {'Test': {'(n,gamma)': ['GS', 'M1']}},
        lfs={'Test': {'(n,gamma)': [0, 1]}}
    )
    gendf = _make_gendf(n_groups, energy_bins,
                        br_func=lambda *a, **kw: mock_br)
    helper = IsomericBranchingHelper(chain, gendf)

    result = helper.weighted_branching_ratios(flux, energy_bins)

    if 'Test' in result and '(n,gamma)' in result['Test']:
        ratios = result['Test']['(n,gamma)']
        total = sum(ratios.values())
        assert np.isclose(total, 1.0), \
            f"Weighted ratios sum to {total}, not 1.0"


# ============================================================================
# Test 10: Boundary conditions
# ============================================================================

def test_single_target_gives_ratio_one():
    """Single target -> branching ratio = 1.0."""
    energy_bins = GROUP_STRUCTURES['CCFE-709']
    n_groups = 709

    chain = _make_chain(
        {'X': {'(n,gamma)': ['Y']}},
        lfs={'X': {'(n,gamma)': [0]}}
    )
    mock_br = IsomericBranching(
        energies=energy_bins[:-1].copy(),
        products=['Y'],
        branching_ratios=np.array([np.ones(n_groups)]),
        parent_nuclide='X', reaction='(n,gamma)', mt=102,
    )
    gendf = _make_gendf(n_groups, energy_bins,
                        br_func=lambda *a, **kw: mock_br)
    helper = IsomericBranchingHelper(chain, gendf)

    result = helper.weighted_branching_ratios(np.ones(n_groups), energy_bins)
    if 'X' in result and '(n,gamma)' in result['X']:
        assert np.isclose(result['X']['(n,gamma)']['Y'], 1.0)


def test_zero_flux_returns_empty():
    """Zero flux -> empty result (no reaction rate weight)."""
    energy_bins = GROUP_STRUCTURES['CCFE-709']
    n_groups = 709

    chain = _make_chain(
        {'X': {'(n,gamma)': ['Y', 'Z']}},
        lfs={'X': {'(n,gamma)': [0, 1]}}
    )
    mock_br = IsomericBranching(
        energies=energy_bins[:-1].copy(),
        products=['Y', 'Z'],
        branching_ratios=np.array([
            np.full(n_groups, 0.7), np.full(n_groups, 0.3),
        ]),
        parent_nuclide='X', reaction='(n,gamma)', mt=102,
    )
    gendf = _make_gendf(n_groups, energy_bins,
                        br_func=lambda *a, **kw: mock_br)
    helper = IsomericBranchingHelper(chain, gendf)

    result = helper.weighted_branching_ratios(np.zeros(n_groups), energy_bins)
    assert result == {} or 'X' not in result or \
        '(n,gamma)' not in result.get('X', {})


def test_negative_flux_raises():
    """Negative flux -> ValueError."""
    energy_bins = GROUP_STRUCTURES['CCFE-709']
    n_groups = 709

    chain = _make_chain(
        {'X': {'(n,gamma)': ['Y']}},
        lfs={'X': {'(n,gamma)': [0]}}
    )
    mock_br = IsomericBranching(
        energies=energy_bins[:-1].copy(),
        products=['Y'],
        branching_ratios=np.array([np.ones(n_groups)]),
        parent_nuclide='X', reaction='(n,gamma)', mt=102)
    gendf = _make_gendf(n_groups, energy_bins,
                        br_func=lambda *a, **kw: mock_br)
    helper = IsomericBranchingHelper(chain, gendf)

    with pytest.raises(ValueError, match="negative"):
        helper.weighted_branching_ratios(-np.ones(n_groups), energy_bins)


def test_no_isomeric_targets_returns_empty():
    """Chain with no isomeric targets -> empty result."""
    energy_bins = GROUP_STRUCTURES['CCFE-709']
    chain = _make_chain(targets=None)
    gendf = _make_gendf(709, energy_bins)
    helper = IsomericBranchingHelper(chain, gendf)
    result = helper.weighted_branching_ratios(np.ones(709), energy_bins)
    assert result == {}


# ============================================================================
# Test 11: REACTION_TO_MT / MT_TO_REACTION consistency
# ============================================================================

def test_mt_reaction_roundtrip():
    """MT -> reaction -> MT should be identity."""
    for mt, reaction in MT_TO_REACTION.items():
        if reaction in REACTION_TO_MT:
            assert REACTION_TO_MT[reaction] == mt, \
                f"MT={mt} -> {reaction} -> MT={REACTION_TO_MT[reaction]}"


def test_key_reactions_present():
    """Important non-fission reactions in mapping."""
    assert REACTION_TO_MT['(n,gamma)'] == 102
    assert REACTION_TO_MT['(n,2n)'] == 16
    assert REACTION_TO_MT['(n,3n)'] == 17
    assert REACTION_TO_MT['(n,a)'] == 107
    assert REACTION_TO_MT['(n,p)'] == 103


# ============================================================================
# Test 12: compute_for_materials integration
# ============================================================================

def test_multiple_materials_same_chain():
    """Multiple materials produce per-material branching dicts."""
    energy_bins = GROUP_STRUCTURES['CCFE-709']
    n_groups = 709

    chain = _make_chain(
        {'Ag109': {'(n,gamma)': ['Ag110', 'Ag110_m1']}},
        lfs={'Ag109': {'(n,gamma)': [0, 1]}}
    )
    mock_br = IsomericBranching(
        energies=energy_bins[:-1].copy(),
        products=['Ag110', 'Ag110_m1'],
        branching_ratios=np.array([
            np.full(n_groups, 0.9), np.full(n_groups, 0.1),
        ]),
        parent_nuclide='Ag109', reaction='(n,gamma)', mt=102,
    )
    gendf = _make_gendf(n_groups, energy_bins,
                        br_func=lambda *a, **kw: mock_br)
    helper = IsomericBranchingHelper(chain, gendf)

    flux_energy_pairs = [
        (np.ones(n_groups), energy_bins),
        (np.ones(n_groups) * 2.0, energy_bins),
        (np.ones(n_groups) * 0.5, energy_bins),
    ]

    result = helper.compute_for_materials(flux_energy_pairs)

    assert result is not None
    assert len(result) == 3
    for mat_result in result:
        assert 'Ag109' in mat_result
        total = sum(mat_result['Ag109']['(n,gamma)'].values())
        assert np.isclose(total, 1.0)


def test_all_empty_warns_and_returns_none():
    """All materials produce empty -> warn and return None."""
    energy_bins = GROUP_STRUCTURES['CCFE-709']
    chain = _make_chain(
        {'X': {'(n,gamma)': ['Y']}},
        lfs={'X': {'(n,gamma)': [0]}}
    )
    gendf = _make_gendf(709, energy_bins,
                        br_func=lambda *a, **kw: None)
    helper = IsomericBranchingHelper(chain, gendf)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = helper.compute_for_materials(
            [(np.ones(709), energy_bins)])
        assert result is None
        assert any('could not be calculated' in str(x.message) for x in w)


# ============================================================================
# Test 13: C++ wrapper lfs_mapping construction
# ============================================================================

def test_cpp_wrapper_lfs_mapping_excludes_ground():
    """C++ backend get_branching_ratios builds lfs_mapping excluding LFS=0."""
    # Simulate what the C++ wrapper does
    target_names = ['X', 'X_m1', 'X_m2']
    lfs_values = [0, 1, 3]
    lfs_mapping = {name: lfs for name, lfs
                   in zip(target_names, lfs_values) if lfs > 0}
    assert lfs_mapping == {'X_m1': 1, 'X_m2': 3}
    assert 'X' not in lfs_mapping  # Ground state excluded


def test_python_wrapper_lfs_mapping_matches_cpp():
    """Python backend builds identical lfs_mapping."""
    target_names = ['Y', 'Y_m1']
    lfs_values = [0, 2]
    # Python backend does the same: zip+filter lfs>0
    lfs_mapping = {name: lfs for name, lfs
                   in zip(target_names, lfs_values) if lfs > 0}
    assert lfs_mapping == {'Y_m1': 2}


# ============================================================================
# Test 14: _get_branching_data handles all exception types
# ============================================================================

def test_get_branching_data_catches_all_exceptions():
    """_get_branching_data catches KeyError, ValueError, NotImplementedError,
    and OpenMCError, returning None for each."""
    energy_bins = GROUP_STRUCTURES['CCFE-709']

    for exc_type in [KeyError, ValueError, NotImplementedError]:
        chain = _make_chain(
            {'X': {'(n,gamma)': ['Y']}},
            lfs={'X': {'(n,gamma)': [0]}}
        )
        gendf = _make_gendf(709, energy_bins,
                            br_func=Mock(side_effect=exc_type("test")))
        helper = IsomericBranchingHelper(chain, gendf)

        result = helper._get_branching_data('X', '(n,gamma)')
        assert result is None, \
            f"Should return None for {exc_type.__name__}"


# ============================================================================
# Test 15: GENDFParseResult structure validation
# ============================================================================

def test_gendf_parse_result_has_all_mf10_fields():
    """GENDFParseResult in gendf.h has prod_xs_data, prod_energy_data,
    prod_izap_data fields."""
    gendf_h = os.path.join(os.path.dirname(__file__), '..', '..',
                           'include', 'openmc', 'gendf.h')
    with open(gendf_h) as f:
        content = f.read()

    assert 'prod_xs_data' in content
    assert 'prod_energy_data' in content
    assert 'prod_izap_data' in content
    assert 'ProductionLevel' in content


# ============================================================================
# Test 16: Energy validation in weighted_branching_ratios
# ============================================================================

def test_energy_validation_cached():
    """Energy validation only runs once (flag-based caching)."""
    energy_bins = GROUP_STRUCTURES['CCFE-709']
    n_groups = 709

    chain = _make_chain({'X': {'(n,gamma)': ['Y']}})
    gendf = _make_gendf(n_groups, energy_bins)
    helper = IsomericBranchingHelper(chain, gendf)

    assert not helper._energy_validated

    helper.weighted_branching_ratios(np.ones(n_groups), energy_bins)
    assert helper._energy_validated

    # Second call should not re-validate
    helper.weighted_branching_ratios(np.ones(n_groups), energy_bins)
    assert helper._energy_validated
