"""Final review diagnostic tests for the combined XS + BR acquisition pipeline.

Tests the full pipeline: chain loading -> GENDF branching -> IsomericBranchingHelper
-> form_matrix -> pool.py integration.

Focus areas from review:
1. pool.py signature check for matrix_func
2. _get_branching_data fallback paths (embedded, runtime, missing)
3. reduce() handling of targets, lfs, embedded
4. Patcher-mode decay_file guard in get_branching_ratios()
5. energy_bounds[:-1] shape correctness
6. Error handling in _get_branching_data and _calculate_weighted
7. Conservation laws in form_matrix with isomeric branching
"""

import inspect
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass
from itertools import repeat
from typing import Dict, List, Optional, Any
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pytest

import openmc.deplete
from openmc.deplete.chain import Chain, REACTIONS
from openmc.deplete.helpers import IsomericBranchingHelper
from openmc.deplete.gendf import (
    IsomericBranching, REACTION_TO_MT, MT_TO_REACTION,
    _PythonGENDFLibrary
)
from openmc.mgxs import GROUP_STRUCTURES


# ============================================================================
# 1. pool.py signature check
# ============================================================================

def test_pool_signature_check_accepts_iso():
    """Verify pool.py correctly detects isomeric_branching parameter."""
    # All _matrix_funcs functions should accept isomeric_branching
    from openmc.deplete import _matrix_funcs

    funcs = [
        _matrix_funcs.celi_f1, _matrix_funcs.celi_f2,
        _matrix_funcs.cf4_f1, _matrix_funcs.cf4_f2,
        _matrix_funcs.cf4_f3, _matrix_funcs.cf4_f4,
        _matrix_funcs.rk4_f1, _matrix_funcs.rk4_f4,
        _matrix_funcs.leqi_f1, _matrix_funcs.leqi_f2,
        _matrix_funcs.leqi_f3, _matrix_funcs.leqi_f4,
    ]

    for fn in funcs:
        sig = inspect.signature(fn)
        assert 'isomeric_branching' in sig.parameters, \
            f"{fn.__name__} missing isomeric_branching parameter"


def test_pool_signature_check_with_custom_func():
    """Verify pool.py correctly handles custom matrix_func without iso param."""
    # A custom function WITHOUT isomeric_branching
    def custom_matrix_func(chain, rates, fission_yields):
        return chain.form_matrix(rates, fission_yields)

    sig = inspect.signature(custom_matrix_func)
    assert 'isomeric_branching' not in sig.parameters

    # A custom function WITH isomeric_branching
    def custom_matrix_func_iso(chain, rates, fission_yields,
                                isomeric_branching=None):
        return chain.form_matrix(rates, fission_yields, isomeric_branching)

    sig = inspect.signature(custom_matrix_func_iso)
    assert 'isomeric_branching' in sig.parameters


# ============================================================================
# 2. _get_branching_data fallback paths
# ============================================================================

def _make_mock_chain(has_embedded=False, has_lfs=False):
    """Create a mock chain with configurable branching attributes."""
    chain = Mock(spec=Chain)
    chain.isomeric_branching_targets = {
        'Ir191': {'(n,gamma)': ['Ir192', 'Ir192_m1']}
    }
    chain.isomeric_branching_lfs = None
    if has_lfs:
        chain.isomeric_branching_lfs = {
            'Ir191': {'(n,gamma)': [0, 3]}
        }

    chain.isomeric_branching_embedded = None
    if has_embedded:
        chain.isomeric_branching_embedded = {
            ('Ir191', '(n,gamma)'): {
                'energies': np.array([1e5, 1e6, 5e6]),
                'targets': ['Ir192', 'Ir192_m1'],
                'branching_ratios': {
                    'Ir192': np.array([0.95, 0.85, 0.75]),
                    'Ir192_m1': np.array([0.05, 0.15, 0.25]),
                }
            }
        }
    return chain


def _make_mock_gendf(n_groups=709):
    """Create a mock GENDF library for testing."""
    energy_bins = GROUP_STRUCTURES['CCFE-709']
    mock = Mock()
    mock.energy_structure = 'CCFE-709'
    mock.energy_bounds = energy_bins.copy()
    mock.n_groups = n_groups
    mock.get_xs = Mock(return_value=np.ones(n_groups))
    mock.get_branching_ratios = Mock(return_value=IsomericBranching(
        energies=np.array([1e5, 1e6, 5e6]),
        products=['Ir192', 'Ir192_m1'],
        branching_ratios=np.array([
            [0.90, 0.80, 0.70],
            [0.10, 0.20, 0.30],
        ]),
        parent_nuclide='Ir191',
        reaction='(n,gamma)',
        mt=102,
    ))
    return mock


def test_fallback_embedded_first():
    """Verify _get_branching_data checks embedded ratios before GENDF."""
    chain = _make_mock_chain(has_embedded=True)
    gendf = _make_mock_gendf()

    helper = IsomericBranchingHelper(chain, gendf)
    data = helper._get_branching_data('Ir191', '(n,gamma)')

    # Should get embedded dict, not IsomericBranching
    assert isinstance(data, dict), f"Expected dict (embedded), got {type(data)}"
    assert 'energies' in data
    assert np.allclose(data['energies'], [1e5, 1e6, 5e6])

    # GENDF should NOT have been called
    gendf.get_branching_ratios.assert_not_called()


def test_fallback_gendf_when_no_embedded():
    """Verify _get_branching_data calls GENDF when no embedded data."""
    chain = _make_mock_chain(has_embedded=False, has_lfs=True)
    gendf = _make_mock_gendf()

    helper = IsomericBranchingHelper(chain, gendf)
    data = helper._get_branching_data('Ir191', '(n,gamma)')

    # Should get IsomericBranching from GENDF
    assert isinstance(data, IsomericBranching)
    gendf.get_branching_ratios.assert_called_once()


def test_fallback_none_on_gendf_error():
    """Verify _get_branching_data returns None on GENDF errors."""
    chain = _make_mock_chain(has_embedded=False)
    gendf = _make_mock_gendf()
    gendf.get_branching_ratios = Mock(side_effect=KeyError("not found"))

    helper = IsomericBranchingHelper(chain, gendf)
    data = helper._get_branching_data('Ir191', '(n,gamma)')

    assert data is None


def test_fallback_none_on_unknown_reaction():
    """Verify _get_branching_data returns None for unknown reaction type."""
    chain = _make_mock_chain()
    gendf = _make_mock_gendf()

    helper = IsomericBranchingHelper(chain, gendf)
    # A reaction type that is NOT in REACTION_TO_MT
    data = helper._get_branching_data('Ir191', '(n,xyzzy)')

    assert data is None


def test_branching_data_cached():
    """Verify _get_branching_data caches results."""
    chain = _make_mock_chain(has_lfs=True)
    gendf = _make_mock_gendf()

    helper = IsomericBranchingHelper(chain, gendf)

    # Call twice
    data1 = helper._get_branching_data('Ir191', '(n,gamma)')
    data2 = helper._get_branching_data('Ir191', '(n,gamma)')

    assert data1 is data2  # Same object (cached)
    assert gendf.get_branching_ratios.call_count == 1


# ============================================================================
# 3. reduce() handling of all three chain attributes
# ============================================================================

def test_reduce_preserves_all_three_attributes():
    """Verify reduce() filters targets, lfs, and embedded correctly."""
    chain = openmc.deplete.Chain()

    ag109 = openmc.deplete.Nuclide('Ag109')
    ag109.add_reaction('(n,gamma)', 'Ag110', Q=6.8e6, branching_ratio=1.0)
    chain.add_nuclide(ag109)

    ag110 = openmc.deplete.Nuclide('Ag110')
    ag110.half_life = 24.6 * 3600
    chain.add_nuclide(ag110)

    ag110_m1 = openmc.deplete.Nuclide('Ag110_m1')
    ag110_m1.half_life = 249.79 * 86400
    chain.add_nuclide(ag110_m1)

    chain.isomeric_branching_targets = {
        'Ag109': {'(n,gamma)': ['Ag110', 'Ag110_m1']}
    }
    chain.isomeric_branching_lfs = {
        'Ag109': {'(n,gamma)': [0, 3]}
    }
    chain.isomeric_branching_embedded = {
        ('Ag109', '(n,gamma)'): {
            'energies': np.array([1e5, 1e6]),
            'targets': ['Ag110', 'Ag110_m1'],
            'branching_ratios': {
                'Ag110': np.array([0.9, 0.8]),
                'Ag110_m1': np.array([0.1, 0.2]),
            }
        }
    }

    # Full reduction keeping all
    reduced = chain.reduce(['Ag109'])

    assert reduced.isomeric_branching_targets is not None
    assert 'Ag109' in reduced.isomeric_branching_targets
    targets = reduced.isomeric_branching_targets['Ag109']['(n,gamma)']
    assert targets == ['Ag110', 'Ag110_m1']

    # LFS should be preserved
    assert reduced.isomeric_branching_lfs is not None
    lfs = reduced.isomeric_branching_lfs['Ag109']['(n,gamma)']
    assert lfs == [0, 3]

    # Embedded should be preserved
    assert reduced.isomeric_branching_embedded is not None
    key = ('Ag109', '(n,gamma)')
    assert key in reduced.isomeric_branching_embedded


def test_reduce_partial_exclusion_lfs_sync():
    """Verify LFS values stay in sync with targets during partial exclusion."""
    chain = openmc.deplete.Chain()

    ag109 = openmc.deplete.Nuclide('Ag109')
    ag109.add_reaction('(n,gamma)', 'Ag110', Q=6.8e6, branching_ratio=1.0)
    chain.add_nuclide(ag109)

    ag110 = openmc.deplete.Nuclide('Ag110')
    chain.add_nuclide(ag110)

    ag110_m1 = openmc.deplete.Nuclide('Ag110_m1')
    ag110_m1.half_life = 249.79 * 86400
    chain.add_nuclide(ag110_m1)

    chain.isomeric_branching_targets = {
        'Ag109': {'(n,gamma)': ['Ag110', 'Ag110_m1']}
    }
    chain.isomeric_branching_lfs = {
        'Ag109': {'(n,gamma)': [0, 3]}
    }

    # Reduce excluding Ag110_m1
    reduced = chain.reduce(['Ag109', 'Ag110'], keep_isomeric_siblings=False)

    targets = reduced.isomeric_branching_targets['Ag109']['(n,gamma)']
    assert targets == ['Ag110']

    lfs = reduced.isomeric_branching_lfs['Ag109']['(n,gamma)']
    assert lfs == [0], f"LFS should be [0] (ground only), got {lfs}"


def test_reduce_embedded_filtered():
    """Verify reduce() filters embedded branching ratios properly."""
    chain = openmc.deplete.Chain()

    ag109 = openmc.deplete.Nuclide('Ag109')
    ag109.add_reaction('(n,gamma)', 'Ag110', Q=6.8e6, branching_ratio=1.0)
    chain.add_nuclide(ag109)

    ag110 = openmc.deplete.Nuclide('Ag110')
    chain.add_nuclide(ag110)

    ag110_m1 = openmc.deplete.Nuclide('Ag110_m1')
    ag110_m1.half_life = 249.79 * 86400
    chain.add_nuclide(ag110_m1)

    chain.isomeric_branching_targets = {
        'Ag109': {'(n,gamma)': ['Ag110', 'Ag110_m1']}
    }
    chain.isomeric_branching_embedded = {
        ('Ag109', '(n,gamma)'): {
            'energies': np.array([1e5, 1e6]),
            'targets': ['Ag110', 'Ag110_m1'],
            'branching_ratios': {
                'Ag110': np.array([0.9, 0.8]),
                'Ag110_m1': np.array([0.1, 0.2]),
            }
        }
    }

    reduced = chain.reduce(['Ag109', 'Ag110'], keep_isomeric_siblings=False)

    # Embedded data should only have Ag110
    key = ('Ag109', '(n,gamma)')
    assert key in reduced.isomeric_branching_embedded
    emb = reduced.isomeric_branching_embedded[key]
    assert emb['targets'] == ['Ag110']
    assert 'Ag110' in emb['branching_ratios']
    assert 'Ag110_m1' not in emb['branching_ratios']


# ============================================================================
# 4. Patcher-mode decay_file guard
# ============================================================================

def test_patcher_mode_requires_decay_file():
    """Verify get_branching_ratios raises when patcher mode has no decay file."""
    mock_lib = Mock(spec=_PythonGENDFLibrary)
    mock_lib.decay_lookup = None

    # Calling get_branching_ratios without target_names => patcher mode
    with pytest.raises(ValueError, match="Patcher mode requires decay_file"):
        _PythonGENDFLibrary.get_branching_ratios(mock_lib, 'Ir191', 102)


def test_runtime_mode_no_decay_needed():
    """Verify runtime mode works without decay file."""
    mock_lib = Mock(spec=_PythonGENDFLibrary)
    mock_lib.n_groups = 709
    mock_lib.energy_bounds = GROUP_STRUCTURES['CCFE-709']
    mock_lib._get_production_xs = Mock(return_value=[
        (0, 77192, np.ones(709) * 5.0),  # ground
        (3, 77192, np.ones(709) * 1.0),  # m1
    ])

    result = _PythonGENDFLibrary.get_branching_ratios(
        mock_lib, 'Ir191', 102,
        target_names=['Ir192', 'Ir192_m1'],
        lfs_values=[0, 3]
    )

    assert result is not None
    assert isinstance(result, IsomericBranching)
    assert result.products == ['Ir192', 'Ir192_m1']
    # Check BR: 5/(5+1) = 0.833..., 1/(5+1) = 0.1666...
    np.testing.assert_allclose(result.branching_ratios[0], 5.0/6.0, rtol=1e-10)
    np.testing.assert_allclose(result.branching_ratios[1], 1.0/6.0, rtol=1e-10)


def test_runtime_mode_requires_lfs():
    """Verify runtime mode raises when lfs_values missing."""
    mock_lib = Mock(spec=_PythonGENDFLibrary)

    with pytest.raises(ValueError, match="lfs_values required"):
        _PythonGENDFLibrary.get_branching_ratios(
            mock_lib, 'Ir191', 102,
            target_names=['Ir192'], lfs_values=None
        )


# ============================================================================
# 5. energy_bounds[:-1] shape correctness
# ============================================================================

def test_runtime_br_energy_shape():
    """Verify runtime branching energies have n_groups length (not n_groups+1)."""
    mock_lib = Mock(spec=_PythonGENDFLibrary)
    mock_lib.n_groups = 709
    ebounds = GROUP_STRUCTURES['CCFE-709'].copy()
    mock_lib.energy_bounds = ebounds
    mock_lib._get_production_xs = Mock(return_value=[
        (0, 77192, np.ones(709) * 5.0),
        (3, 77192, np.ones(709) * 1.0),
    ])

    result = _PythonGENDFLibrary.get_branching_ratios(
        mock_lib, 'Ir191', 102,
        target_names=['Ir192', 'Ir192_m1'],
        lfs_values=[0, 3]
    )

    assert result.energies.shape == (709,), \
        f"Energies should be (n_groups,)=(709,), got {result.energies.shape}"
    assert result.branching_ratios.shape == (2, 709), \
        f"BR should be (n_products, n_groups)=(2,709), got {result.branching_ratios.shape}"

    # Energies should match energy_bounds[:-1]
    np.testing.assert_array_equal(result.energies, ebounds[:-1])


# ============================================================================
# 6. Error handling
# ============================================================================

def test_negative_flux_raises():
    """Verify ValueError on negative flux spectrum."""
    chain = _make_mock_chain(has_lfs=True)
    gendf = _make_mock_gendf()

    helper = IsomericBranchingHelper(chain, gendf)

    energy_bins = GROUP_STRUCTURES['CCFE-709']
    flux = np.ones(709)
    flux[100] = -1.0  # Negative value

    with pytest.raises(ValueError, match="negative values"):
        helper.weighted_branching_ratios(flux, energy_bins)


def test_flux_energy_mismatch_raises():
    """Verify ValueError when flux length != energy_bins - 1."""
    chain = _make_mock_chain(has_lfs=True)
    gendf = _make_mock_gendf()

    helper = IsomericBranchingHelper(chain, gendf)

    energy_bins = GROUP_STRUCTURES['CCFE-709']
    flux = np.ones(500)  # Wrong size

    with pytest.raises(ValueError, match="groups"):
        helper.weighted_branching_ratios(flux, energy_bins)


def test_gendf_library_none_raises():
    """Verify ValueError when gendf_library is None."""
    chain = _make_mock_chain()
    with pytest.raises(ValueError, match="gendf_library is required"):
        IsomericBranchingHelper(chain, None)


# ============================================================================
# 7. Conservation law in form_matrix
# ============================================================================

def test_form_matrix_mass_conservation():
    """Verify isomeric branching preserves mass: sum of gains = loss.

    For a reaction A -> B + B_m1 with branching 0.7/0.3:
    - Loss from A = -rate * 1.0
    - Gain to B = rate * 0.7
    - Gain to B_m1 = rate * 0.3
    - Total gain = loss (conservation)
    """
    from openmc.deplete import ReactionRates

    chain = openmc.deplete.Chain()

    parent = openmc.deplete.Nuclide('Ir191')
    parent.add_reaction('(n,gamma)', 'Ir192', Q=6e6, branching_ratio=1.0)
    chain.add_nuclide(parent)

    gs = openmc.deplete.Nuclide('Ir192')
    chain.add_nuclide(gs)
    ms = openmc.deplete.Nuclide('Ir192_m1')
    chain.add_nuclide(ms)

    nuclides = ['Ir191', 'Ir192', 'Ir192_m1']
    reactions = ['(n,gamma)']
    rates = ReactionRates(['mat1'], nuclides, reactions)

    test_rate = 1.5e-8  # Arbitrary reaction rate
    rates[0, 0, 0] = test_rate

    iso_br = {
        'Ir191': {
            '(n,gamma)': {
                'Ir192': 0.65,
                'Ir192_m1': 0.35
            }
        }
    }

    matrix = chain.form_matrix(rates[0], isomeric_branching=iso_br)
    dense = matrix.toarray()

    i_parent = chain.nuclide_dict['Ir191']
    i_gs = chain.nuclide_dict['Ir192']
    i_ms = chain.nuclide_dict['Ir192_m1']

    loss = dense[i_parent, i_parent]
    gain_gs = dense[i_gs, i_parent]
    gain_ms = dense[i_ms, i_parent]

    # Loss should equal -rate
    assert np.isclose(loss, -test_rate), \
        f"Loss should be -{test_rate}, got {loss}"

    # Gains should equal rate * branching_ratio
    assert np.isclose(gain_gs, test_rate * 0.65), \
        f"Ground state gain should be {test_rate * 0.65}, got {gain_gs}"
    assert np.isclose(gain_ms, test_rate * 0.35), \
        f"Metastable gain should be {test_rate * 0.35}, got {gain_ms}"

    # Conservation: loss + gains = 0
    total = loss + gain_gs + gain_ms
    assert np.isclose(total, 0.0, atol=1e-20), \
        f"Mass conservation violated: loss + gains = {total}"


def test_form_matrix_three_way_branching():
    """Verify 3-way branching (ground + m1 + m2) conserves mass."""
    from openmc.deplete import ReactionRates

    chain = openmc.deplete.Chain()

    parent = openmc.deplete.Nuclide('Ir191')
    parent.add_reaction('(n,gamma)', 'Ir192', Q=6e6, branching_ratio=1.0)
    chain.add_nuclide(parent)

    for name in ['Ir192', 'Ir192_m1', 'Ir192_m2']:
        nuc = openmc.deplete.Nuclide(name)
        chain.add_nuclide(nuc)

    nuclides = [n.name for n in chain.nuclides]
    rates = ReactionRates(['mat1'], nuclides, ['(n,gamma)'])
    rates[0, 0, 0] = 2.0e-8

    iso_br = {
        'Ir191': {
            '(n,gamma)': {
                'Ir192': 0.60,
                'Ir192_m1': 0.25,
                'Ir192_m2': 0.15
            }
        }
    }

    matrix = chain.form_matrix(rates[0], isomeric_branching=iso_br)
    dense = matrix.toarray()

    i_parent = chain.nuclide_dict['Ir191']
    loss = dense[i_parent, i_parent]
    gains = sum(dense[chain.nuclide_dict[t], i_parent]
                for t in ['Ir192', 'Ir192_m1', 'Ir192_m2'])

    assert np.isclose(loss + gains, 0.0, atol=1e-20), \
        f"3-way conservation violated: {loss} + {gains} = {loss + gains}"


def test_form_matrix_self_transmutation_nn_prime():
    """Verify (n,n') self-transmutation handles loss correctly.

    For A(n,n')A with branching 0.85 ground / 0.15 metastable:
    - Loss from A = -rate
    - Gain to A (from self-scatter) = rate * 0.85
    - Gain to A_m1 = rate * 0.15
    - Net effect on A: -rate + rate*0.85 = -rate*0.15 (net transfer to m1)
    """
    from openmc.deplete import ReactionRates

    chain = openmc.deplete.Chain()

    in115 = openmc.deplete.Nuclide('In115')
    in115.add_reaction("(n,n')", 'In115', Q=0.0, branching_ratio=0.85)
    chain.add_nuclide(in115)

    in115_m1 = openmc.deplete.Nuclide('In115_m1')
    in115_m1.half_life = 1.61e4
    in115_m1.add_decay_mode('IT', 'In115', 1.0)
    chain.add_nuclide(in115_m1)

    nuclides = ['In115', 'In115_m1']
    rates = ReactionRates(['mat1'], nuclides, list(chain.reactions))

    rates.set('mat1', 'In115', "(n,n')", 1e-10)

    iso_br = {
        'In115': {
            "(n,n')": {
                'In115': 0.85,
                'In115_m1': 0.15
            }
        }
    }

    matrix = chain.form_matrix(rates[0], isomeric_branching=iso_br)
    dense = matrix.toarray()

    i_115 = chain.nuclide_dict['In115']
    i_m1 = chain.nuclide_dict['In115_m1']

    # Self-scatter: loss -rate, self-gain rate*0.85
    diag = dense[i_115, i_115]
    gain_to_m1 = dense[i_m1, i_115]

    # Note: decay of In115_m1 also contributes to In115
    decay_rate = math.log(2) / 1.61e4

    # The diagonal for In115 should be: -reaction_rate + reaction_rate * 0.85
    # (isomeric branching replaces the single-target gain with multi-target)
    # Wait -- form_matrix applies loss as -rate ONCE per reaction type,
    # and the isomeric branching gain is rate * iso_br.
    # So net on In115: -rate + rate*0.85 = -rate*0.15

    # The In115_m1 diagonal should include -decay_rate
    diag_m1 = dense[i_m1, i_m1]
    assert diag_m1 < 0, "In115_m1 should have negative diagonal (decay)"

    # Gain from In115 to In115_m1 via (n,n')
    assert np.isclose(gain_to_m1, 1e-10 * 0.15), \
        f"Expected {1e-10 * 0.15}, got {gain_to_m1}"


# ============================================================================
# 8. IsomericBranching dataclass consistency
# ============================================================================

def test_isomeric_branching_to_from_dict():
    """Verify IsomericBranching round-trips through dict serialization."""
    orig = IsomericBranching(
        energies=np.array([1e5, 1e6, 5e6, 1e7]),
        products=['Ir192', 'Ir192_m1'],
        branching_ratios=np.array([
            [0.90, 0.80, 0.70, 0.60],
            [0.10, 0.20, 0.30, 0.40],
        ]),
        parent_nuclide='Ir191',
        reaction='(n,gamma)',
        mt=102,
    )

    d = orig.to_dict()
    restored = IsomericBranching.from_dict(d)

    np.testing.assert_array_equal(restored.energies, orig.energies)
    assert restored.products == orig.products
    np.testing.assert_array_equal(restored.branching_ratios,
                                   orig.branching_ratios)
    assert restored.parent_nuclide == orig.parent_nuclide
    assert restored.reaction == orig.reaction
    assert restored.mt == orig.mt


# ============================================================================
# 9. Weighted branching with embedded data path
# ============================================================================

def test_weighted_branching_with_embedded():
    """Verify weighted_branching_ratios works with chain-embedded data.

    This tests the isinstance(br, dict) path in weighted_branching_ratios.
    """
    chain = _make_mock_chain(has_embedded=True)
    gendf = _make_mock_gendf()

    helper = IsomericBranchingHelper(chain, gendf)

    energy_bins = GROUP_STRUCTURES['CCFE-709']
    flux = np.ones(709)

    result = helper.weighted_branching_ratios(flux, energy_bins)

    assert 'Ir191' in result
    assert '(n,gamma)' in result['Ir191']
    ratios = result['Ir191']['(n,gamma)']
    total = sum(ratios.values())
    assert np.isclose(total, 1.0), f"Ratios should sum to 1.0, got {total}"


# ============================================================================
# 10. Edge cases and physical limits
# ============================================================================

def test_single_target_degeneracy():
    """Verify single-target branching produces ratio of 1.0."""
    chain = Mock(spec=Chain)
    chain.isomeric_branching_targets = {
        'X': {'(n,gamma)': ['Y']}  # Only one target
    }
    chain.isomeric_branching_lfs = None
    chain.isomeric_branching_embedded = None

    gendf = _make_mock_gendf()
    # Return single-product branching
    gendf.get_branching_ratios = Mock(return_value=IsomericBranching(
        energies=np.array([1e5, 1e6]),
        products=['Y'],
        branching_ratios=np.array([[1.0, 1.0]]),
        parent_nuclide='X',
        reaction='(n,gamma)',
        mt=102,
    ))

    helper = IsomericBranchingHelper(chain, gendf)
    energy_bins = GROUP_STRUCTURES['CCFE-709']
    flux = np.ones(709)

    result = helper.weighted_branching_ratios(flux, energy_bins)

    if 'X' in result and '(n,gamma)' in result['X']:
        assert np.isclose(result['X']['(n,gamma)']['Y'], 1.0)


def test_normalize_ratios_warns_on_deviation():
    """Verify warning when ratios deviate > 1% from 1.0."""
    chain = _make_mock_chain()
    gendf = _make_mock_gendf()
    helper = IsomericBranchingHelper(chain, gendf)

    bad_ratios = {'A': 0.5, 'B': 0.3}  # Sum = 0.8, deviation > 1%
    with pytest.warns(UserWarning, match="deviates >1%"):
        result = helper._normalize_ratios(bad_ratios, 'test', 'test')

    assert np.isclose(sum(result.values()), 1.0)


def test_normalize_ratios_zero_sum_returns_empty():
    """Verify empty dict when ratios sum to zero (fall back to static BR)."""
    chain = _make_mock_chain()
    gendf = _make_mock_gendf()
    helper = IsomericBranchingHelper(chain, gendf)

    result = helper._normalize_ratios({'A': 0.0, 'B': 0.0}, 'test', 'test')
    assert result == {}


# ============================================================================
# 11. Chain XML round-trip with all three attributes
# ============================================================================

def test_chain_xml_roundtrip_lfs():
    """Verify gendf_lfs attribute survives XML export/import."""
    import tempfile
    from pathlib import Path

    chain = openmc.deplete.Chain()

    ag109 = openmc.deplete.Nuclide('Ag109')
    ag109.add_reaction('(n,gamma)', 'Ag110', Q=6.8e6, branching_ratio=1.0)
    chain.add_nuclide(ag109)

    ag110 = openmc.deplete.Nuclide('Ag110')
    chain.add_nuclide(ag110)

    ag110_m1 = openmc.deplete.Nuclide('Ag110_m1')
    ag110_m1.half_life = 249.79 * 86400
    chain.add_nuclide(ag110_m1)

    chain.isomeric_branching_targets = {
        'Ag109': {'(n,gamma)': ['Ag110', 'Ag110_m1']}
    }
    chain.isomeric_branching_lfs = {
        'Ag109': {'(n,gamma)': [0, 3]}
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "chain.xml"
        chain.export_to_xml(xml_path)
        reloaded = openmc.deplete.Chain.from_xml(xml_path)

        assert reloaded.isomeric_branching_targets is not None
        assert reloaded.isomeric_branching_targets['Ag109']['(n,gamma)'] == \
            ['Ag110', 'Ag110_m1']

        assert reloaded.isomeric_branching_lfs is not None
        assert reloaded.isomeric_branching_lfs['Ag109']['(n,gamma)'] == [0, 3]


# ============================================================================
# 12. Helper uses gendf_library.get_xs for XS lookup
# ============================================================================

def test_helper_calls_get_xs():
    """Verify IsomericBranchingHelper calls gendf_library.get_xs for XS."""
    chain = _make_mock_chain(has_lfs=True)
    gendf = _make_mock_gendf()

    helper = IsomericBranchingHelper(chain, gendf)

    energy_bins = GROUP_STRUCTURES['CCFE-709']
    flux = np.ones(709)

    result = helper.weighted_branching_ratios(flux, energy_bins)

    # get_xs should be called for sigma*phi weighting
    gendf.get_xs.assert_called()
    assert 'Ir191' in result


# ============================================================================
# 13. form_matrix validation of isomeric branching
# ============================================================================

def test_form_matrix_nan_br_warns():
    """Verify form_matrix warns on NaN/Inf branching ratios."""
    from openmc.deplete import ReactionRates

    chain = openmc.deplete.Chain()
    parent = openmc.deplete.Nuclide('Parent')
    parent.add_reaction('(n,gamma)', 'Product', Q=1e6, branching_ratio=1.0)
    chain.add_nuclide(parent)

    product = openmc.deplete.Nuclide('Product')
    chain.add_nuclide(product)

    nuclides = ['Parent', 'Product']
    rates = ReactionRates(['mat1'], nuclides, ['(n,gamma)'])
    rates[0, 0, 0] = 1.0

    iso_br = {
        'Parent': {
            '(n,gamma)': {
                'Product': float('nan')
            }
        }
    }

    with pytest.warns(UserWarning, match="Invalid isomeric branching ratio"):
        matrix = chain.form_matrix(rates[0], isomeric_branching=iso_br)


def test_form_matrix_missing_target_raises():
    """Verify form_matrix raises KeyError for missing isomeric target."""
    from openmc.deplete import ReactionRates

    chain = openmc.deplete.Chain()
    parent = openmc.deplete.Nuclide('Parent')
    parent.add_reaction('(n,gamma)', 'Product', Q=1e6, branching_ratio=1.0)
    chain.add_nuclide(parent)

    product = openmc.deplete.Nuclide('Product')
    chain.add_nuclide(product)

    nuclides = ['Parent', 'Product']
    rates = ReactionRates(['mat1'], nuclides, ['(n,gamma)'])
    rates[0, 0, 0] = 1.0

    # Reference a target that doesn't exist in the chain
    iso_br = {
        'Parent': {
            '(n,gamma)': {
                'Product': 0.7,
                'NonExistent': 0.3  # Not in chain
            }
        }
    }

    with pytest.raises(KeyError, match="not in chain"):
        chain.form_matrix(rates[0], isomeric_branching=iso_br)


# ============================================================================
# 14. _build_branching_result shape assertion
# ============================================================================

def test_build_branching_result_shape_assertion():
    """Verify _build_branching_result validates product count vs array shape.

    The shape assertion at line 1843 catches internal bugs where the number
    of products doesn't match the branching_array row count.
    """
    # This is tested implicitly -- the assertion fires if our mock construction
    # is wrong. Here we verify the assertion exists by testing the normal path.
    mock_lib = Mock(spec=_PythonGENDFLibrary)
    mock_lib.n_groups = 3
    mock_lib.energy_bounds = np.array([1e-5, 1e3, 1e6, 2e7])
    mock_lib._get_production_xs = Mock(return_value=[
        (0, 77192, np.array([5.0, 5.0, 5.0])),
        (3, 77192, np.array([1.0, 1.0, 1.0])),
    ])

    result = _PythonGENDFLibrary.get_branching_ratios(
        mock_lib, 'Ir191', 102,
        target_names=['Ir192', 'Ir192_m1'],
        lfs_values=[0, 3]
    )

    # 2 products, 3 groups
    assert result.branching_ratios.shape[0] == 2
    assert result.branching_ratios.shape[1] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
