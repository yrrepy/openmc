"""Tests for σ×φ-weighted isomeric branching ratios.

Tests the correct implementation of:
- No extrapolation of BR outside data range
- σ×φ weighting instead of φ-only weighting
- Runtime GENDF lookup with caching
"""

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from openmc.mgxs import GROUP_STRUCTURES
from openmc.deplete.helpers import IsomericBranchingHelper
from openmc.deplete.chain import Chain
from openmc.deplete.gendf import IsomericBranching


def create_mock_chain_with_isomeric_targets():
    """Create a mock chain with isomeric branching targets."""
    chain = Mock(spec=Chain)
    chain.isomeric_branching_targets = {
        'Ag109': {
            '(n,gamma)': ['Ag110', 'Ag110_m1']
        }
    }
    chain.isomeric_branching_lfs = None
    chain.isomeric_branching_embedded = None
    return chain


def create_mock_isomeric_branching():
    """Create an IsomericBranching dataclass for Ag109(n,gamma)."""
    return IsomericBranching(
        energies=np.array([1e5, 1e6, 5e6, 1e7]),
        products=['Ag110', 'Ag110_m1'],
        branching_ratios=np.array([
            [0.9, 0.8, 0.7, 0.6],    # Ag110
            [0.1, 0.2, 0.3, 0.4],    # Ag110_m1
        ]),
        parent_nuclide='Ag109',
        reaction='(n,gamma)',
        mt=102,
    )


def create_mock_gendf_library(n_groups, energy_bins, in_range_mask=None,
                              energy_structure='CCFE-709'):
    """Create a mock GENDF library for testing."""
    mock_gendf = Mock()
    mock_gendf.energy_structure = energy_structure
    mock_gendf.energy_bounds = energy_bins.copy()
    mock_gendf.n_groups = n_groups

    if in_range_mask is None:
        e_min = 1e5
        e_max = 1e7
        in_range_mask = (energy_bins[:-1] >= e_min) & (energy_bins[:-1] < e_max)

    gendf_xs = np.zeros(n_groups)
    gendf_xs[in_range_mask] = 1.0
    mock_gendf.get_xs = Mock(return_value=gendf_xs)

    # Return IsomericBranching dataclass from get_branching_ratios
    mock_gendf.get_branching_ratios = Mock(
        return_value=create_mock_isomeric_branching()
    )

    return mock_gendf


# ============================================================================
# Phase 14A Tests: No extrapolation outside data range
# ============================================================================

def test_no_extrapolation_below_threshold():
    """Verify BR array is zero below isomeric data range."""
    chain = create_mock_chain_with_isomeric_targets()
    mock_gendf = Mock()
    mock_gendf.energy_structure = 'CCFE-709'
    mock_gendf.energy_bounds = GROUP_STRUCTURES['CCFE-709'].copy()
    mock_gendf.get_branching_ratios = Mock(
        return_value=create_mock_isomeric_branching()
    )
    helper = IsomericBranchingHelper(chain, mock_gendf)

    n_groups = 10
    ratios = np.array([0.9, 0.8, 0.7])
    iso_indices = np.array([0, 0, 0, 0, 0, 1, 1, 2, 2, 2])
    below_range = np.array([True, True, True, True, False, False, False, False, False, False])
    above_range = np.array([False, False, False, False, False, False, False, False, False, False])
    in_range = ~below_range & ~above_range

    br_array = helper._build_branching_array(
        ratios, iso_indices, below_range, above_range, in_range, n_groups
    )

    assert np.all(br_array[below_range] == 0.0), "Below-range groups should be zero"
    assert np.any(br_array[in_range] > 0), "In-range groups should have non-zero values"


def test_no_extrapolation_above_range():
    """Verify BR array is zero above isomeric data range."""
    chain = create_mock_chain_with_isomeric_targets()
    mock_gendf = Mock()
    mock_gendf.energy_structure = 'CCFE-709'
    mock_gendf.energy_bounds = GROUP_STRUCTURES['CCFE-709'].copy()
    mock_gendf.get_branching_ratios = Mock(
        return_value=create_mock_isomeric_branching()
    )
    helper = IsomericBranchingHelper(chain, mock_gendf)

    n_groups = 10
    ratios = np.array([0.9, 0.8, 0.7])
    iso_indices = np.array([0, 0, 1, 1, 2, 2, 2, 2, 2, 2])
    below_range = np.array([False, False, False, False, False, False, False, False, False, False])
    above_range = np.array([False, False, False, False, False, False, True, True, True, True])
    in_range = ~below_range & ~above_range

    br_array = helper._build_branching_array(
        ratios, iso_indices, below_range, above_range, in_range, n_groups
    )

    assert np.all(br_array[above_range] == 0.0), "Above-range groups should be zero"


def test_in_range_values_preserved():
    """Verify in-range BR values are correctly mapped."""
    chain = create_mock_chain_with_isomeric_targets()
    mock_gendf = Mock()
    mock_gendf.energy_structure = 'CCFE-709'
    mock_gendf.energy_bounds = GROUP_STRUCTURES['CCFE-709'].copy()
    mock_gendf.get_branching_ratios = Mock(
        return_value=create_mock_isomeric_branching()
    )
    helper = IsomericBranchingHelper(chain, mock_gendf)

    n_groups = 5
    ratios = np.array([0.9, 0.8, 0.7])
    iso_indices = np.array([0, 0, 1, 2, 2])
    below_range = np.array([True, False, False, False, False])
    above_range = np.array([False, False, False, False, True])
    in_range = ~below_range & ~above_range

    br_array = helper._build_branching_array(
        ratios, iso_indices, below_range, above_range, in_range, n_groups
    )

    expected_in_range = ratios[iso_indices[in_range]]
    assert np.allclose(br_array[in_range], expected_in_range)


# ============================================================================
# Phase 14B Tests: σ×φ weighting with runtime GENDF lookup
# ============================================================================

def test_sigma_phi_weighting_basic():
    """Verify σ×φ weighting produces results and sums to 1.0."""
    chain = create_mock_chain_with_isomeric_targets()

    energy_bins = GROUP_STRUCTURES['CCFE-709']
    n_groups = 709

    mock_gendf = create_mock_gendf_library(n_groups, energy_bins)

    helper = IsomericBranchingHelper(chain, mock_gendf)

    flux_spectrum = np.ones(n_groups)

    result = helper.weighted_branching_ratios(flux_spectrum, energy_bins)

    assert 'Ag109' in result, f"Ag109 should be in result, got: {result}"
    assert '(n,gamma)' in result['Ag109']

    ratios = result['Ag109']['(n,gamma)']
    total = sum(ratios.values())
    assert np.isclose(total, 1.0), f"Ratios should sum to 1.0, got {total}"


def test_gendf_branching_ratios_called():
    """Verify get_branching_ratios is called for GENDF runtime lookup."""
    chain = create_mock_chain_with_isomeric_targets()

    energy_bins = GROUP_STRUCTURES['CCFE-709']
    n_groups = 709

    mock_gendf = create_mock_gendf_library(n_groups, energy_bins)

    helper = IsomericBranchingHelper(chain, mock_gendf)

    flux_spectrum = np.ones(n_groups)
    helper.weighted_branching_ratios(flux_spectrum, energy_bins)

    # No LFS on chain → patcher mode (target_names=None)
    mock_gendf.get_branching_ratios.assert_called_once_with(
        'Ag109', 102,
        target_names=None,
        lfs_values=None)


def test_branching_cache():
    """Verify GENDF branching data is cached after first lookup."""
    chain = create_mock_chain_with_isomeric_targets()

    energy_bins = GROUP_STRUCTURES['CCFE-709']
    n_groups = 709

    mock_gendf = create_mock_gendf_library(n_groups, energy_bins)

    helper = IsomericBranchingHelper(chain, mock_gendf)

    flux_spectrum = np.ones(n_groups)

    # Call twice
    helper.weighted_branching_ratios(flux_spectrum, energy_bins)
    helper.weighted_branching_ratios(flux_spectrum, energy_bins)

    # get_branching_ratios should only be called once (cached)
    assert mock_gendf.get_branching_ratios.call_count == 1


def test_skip_when_nuclide_not_in_gendf():
    """Verify empty dict returned when nuclide not in GENDF library."""
    chain = create_mock_chain_with_isomeric_targets()

    energy_bins = GROUP_STRUCTURES['CCFE-709']
    n_groups = 709

    flux_spectrum = np.ones(n_groups)

    mock_gendf = Mock()
    mock_gendf.energy_structure = 'CCFE-709'
    mock_gendf.energy_bounds = energy_bins.copy()
    mock_gendf.get_branching_ratios = Mock(return_value=None)
    mock_gendf.get_xs = Mock(side_effect=KeyError("Ag109 not found"))

    helper = IsomericBranchingHelper(chain, mock_gendf)

    result = helper.weighted_branching_ratios(flux_spectrum, energy_bins)

    assert result == {} or 'Ag109' not in result


def test_gendf_library_required():
    """Verify ValueError when gendf_library is None at construction."""
    chain = create_mock_chain_with_isomeric_targets()

    with pytest.raises(ValueError, match="gendf_library is required"):
        IsomericBranchingHelper(chain, None)


def test_with_gendf_succeeds():
    """Verify GENDF library path works correctly."""
    chain = create_mock_chain_with_isomeric_targets()

    energy_bins = GROUP_STRUCTURES['CCFE-709']
    n_groups = 709

    mock_gendf = create_mock_gendf_library(n_groups, energy_bins)

    helper = IsomericBranchingHelper(chain, mock_gendf)

    flux_spectrum = np.ones(n_groups)

    result = helper.weighted_branching_ratios(flux_spectrum, energy_bins)

    # Verify GENDF was called
    mock_gendf.get_xs.assert_called()
    call_args = mock_gendf.get_xs.call_args
    assert call_args[0][0] == 'Ag109'
    assert call_args[0][1] == 102

    assert 'Ag109' in result
    assert '(n,gamma)' in result['Ag109']

    ratios = result['Ag109']['(n,gamma)']
    total = sum(ratios.values())
    assert np.isclose(total, 1.0), f"Ratios should sum to 1.0, got {total}"


def test_error_on_group_mismatch():
    """Verify ValueError raised when GENDF groups don't match flux."""
    chain = create_mock_chain_with_isomeric_targets()

    energy_bins = GROUP_STRUCTURES['CCFE-709']
    n_groups = 709

    flux_spectrum = np.ones(n_groups)

    mock_gendf = Mock()
    mock_gendf.energy_structure = 'CCFE-709'
    mock_gendf.energy_bounds = energy_bins.copy()
    mock_gendf.get_branching_ratios = Mock(
        return_value=create_mock_isomeric_branching()
    )
    mock_gendf.get_xs = Mock(return_value=np.ones(500))

    helper = IsomericBranchingHelper(chain, mock_gendf)

    with pytest.raises(ValueError, match="group structure mismatch"):
        helper.weighted_branching_ratios(flux_spectrum, energy_bins)


def test_zero_weight_sum_returns_empty():
    """Verify empty dict when all reaction rate is below threshold."""
    chain = create_mock_chain_with_isomeric_targets()

    energy_bins = GROUP_STRUCTURES['CCFE-709']
    n_groups = 709

    flux_spectrum = np.zeros(n_groups)
    flux_spectrum[:100] = 1.0

    gendf_xs = np.zeros(n_groups)
    gendf_xs[650:] = 1.0

    mock_gendf = Mock()
    mock_gendf.energy_structure = 'CCFE-709'
    mock_gendf.energy_bounds = energy_bins.copy()
    mock_gendf.get_branching_ratios = Mock(
        return_value=create_mock_isomeric_branching()
    )
    mock_gendf.get_xs = Mock(return_value=gendf_xs)

    helper = IsomericBranchingHelper(chain, mock_gendf)

    result = helper.weighted_branching_ratios(flux_spectrum, energy_bins)

    assert result == {} or '(n,gamma)' not in result.get('Ag109', {})


def test_target_filtering_with_reduced_chain():
    """Verify helper filters GENDF products to chain's target list."""
    chain = Mock(spec=Chain)
    # Reduced chain only has ground state
    chain.isomeric_branching_targets = {
        'Ag109': {
            '(n,gamma)': ['Ag110']  # Ag110_m1 was pruned
        }
    }
    chain.isomeric_branching_lfs = None
    chain.isomeric_branching_embedded = None

    energy_bins = GROUP_STRUCTURES['CCFE-709']
    n_groups = 709

    mock_gendf = create_mock_gendf_library(n_groups, energy_bins)

    helper = IsomericBranchingHelper(chain, mock_gendf)

    flux_spectrum = np.ones(n_groups)

    result = helper.weighted_branching_ratios(flux_spectrum, energy_bins)

    if 'Ag109' in result and '(n,gamma)' in result['Ag109']:
        ratios = result['Ag109']['(n,gamma)']
        # Should only have Ag110, not Ag110_m1
        assert 'Ag110' in ratios
        assert 'Ag110_m1' not in ratios
        # Should be renormalized to 1.0
        assert np.isclose(ratios['Ag110'], 1.0)


# ============================================================================
# CoupledOperator Tests
# ============================================================================

def test_coupled_operator_isomeric_disabled_without_gendf():
    """Verify CoupledOperator has no isomeric branching without GENDF library."""
    from openmc.deplete.coupled_operator import CoupledOperator

    mock_coupled = Mock(spec=CoupledOperator)
    mock_coupled._gendf_library = None

    CoupledOperator._setup_isomeric_branching(mock_coupled)

    assert mock_coupled._isomeric_helper is None
    assert mock_coupled._isomeric_branching is None

    result = CoupledOperator._calculate_isomeric_branching(mock_coupled)
    assert result is None


def test_coupled_operator_isomeric_disabled_without_direct_with_flux():
    """Verify CoupledOperator disables isomeric branching without DirectWithFluxHelper."""
    from openmc.deplete.coupled_operator import CoupledOperator
    from openmc.deplete.helpers import DirectReactionRateHelper

    mock_coupled = Mock(spec=CoupledOperator)
    mock_coupled._gendf_library = Mock()
    mock_coupled._rate_helper = Mock(spec=DirectReactionRateHelper)
    mock_coupled.chain = create_mock_chain_with_isomeric_targets()

    with pytest.warns(UserWarning, match="flux spectrum"):
        CoupledOperator._setup_isomeric_branching(mock_coupled)

    assert mock_coupled._isomeric_helper is None
    assert mock_coupled._isomeric_branching is None


def test_coupled_operator_isomeric_enabled_with_gendf():
    """Verify CoupledOperator enables isomeric branching with GENDF + direct_with_flux."""
    from openmc.deplete.coupled_operator import CoupledOperator
    from openmc.deplete.helpers import DirectWithFluxHelper, IsomericBranchingHelper

    mock_coupled = Mock(spec=CoupledOperator)
    mock_gendf = Mock()
    mock_gendf.energy_structure = 'CCFE-709'
    mock_gendf.energy_bounds = GROUP_STRUCTURES['CCFE-709'].copy()
    mock_gendf.get_branching_ratios = Mock(
        return_value=create_mock_isomeric_branching()
    )
    mock_coupled._gendf_library = mock_gendf
    mock_coupled._rate_helper = Mock(spec=DirectWithFluxHelper)
    mock_coupled._rate_helper.energies = GROUP_STRUCTURES['CCFE-709']
    mock_coupled.chain = create_mock_chain_with_isomeric_targets()

    CoupledOperator._setup_isomeric_branching(mock_coupled)

    assert mock_coupled._isomeric_helper is not None
    assert isinstance(mock_coupled._isomeric_helper, IsomericBranchingHelper)


def test_coupled_operator_calculate_isomeric_branching():
    """Verify CoupledOperator calculates isomeric branching using flux from tally."""
    from openmc.deplete.coupled_operator import CoupledOperator
    from openmc.deplete.helpers import DirectWithFluxHelper, IsomericBranchingHelper

    mock_coupled = Mock(spec=CoupledOperator)
    mock_coupled.local_mats = ['mat1', 'mat2']

    n_groups = 709
    energy_bins = GROUP_STRUCTURES['CCFE-709']

    mock_gendf = create_mock_gendf_library(n_groups, energy_bins)

    chain = create_mock_chain_with_isomeric_targets()
    mock_coupled._isomeric_helper = IsomericBranchingHelper(chain, mock_gendf)

    mock_rate_helper = Mock(spec=DirectWithFluxHelper)
    mock_rate_helper.energies = energy_bins
    mock_rate_helper.get_flux_spectrum = Mock(return_value=np.ones(n_groups))
    mock_coupled._rate_helper = mock_rate_helper

    result = CoupledOperator._calculate_isomeric_branching(mock_coupled)

    assert result is not None
    assert len(result) == 2

    for mat_result in result:
        assert 'Ag109' in mat_result
        assert '(n,gamma)' in mat_result['Ag109']
        ratios = mat_result['Ag109']['(n,gamma)']
        total = sum(ratios.values())
        assert np.isclose(total, 1.0)


# ============================================================================
# form_matrix() Tests (unchanged interface — runtime parameter)
# ============================================================================

def test_form_matrix_uses_isomeric_br():
    """Verify form_matrix uses path_rate * br from isomeric data."""
    import openmc.deplete
    from openmc.deplete import ReactionRates

    chain = openmc.deplete.Chain()

    parent = openmc.deplete.Nuclide('Parent')
    parent.add_reaction('(n,gamma)', 'Product', Q=1e6, branching_ratio=0.5)
    chain.add_nuclide(parent)

    product1 = openmc.deplete.Nuclide('Product')
    chain.add_nuclide(product1)
    product2 = openmc.deplete.Nuclide('Product_m1')
    chain.add_nuclide(product2)

    nuclides = ['Parent', 'Product', 'Product_m1']
    reactions = ['(n,gamma)']
    rates = ReactionRates(['mat1'], nuclides, reactions)
    rates[0, 0, 0] = 1.0

    isomeric_branching = {
        'Parent': {
            '(n,gamma)': {
                'Product': 0.7,
                'Product_m1': 0.3
            }
        }
    }

    rates_2d = rates[0]
    matrix = chain.form_matrix(rates_2d, isomeric_branching=isomeric_branching)

    dense = matrix.toarray()

    parent_idx = chain.nuclide_dict['Parent']
    product_idx = chain.nuclide_dict['Product']
    product_m1_idx = chain.nuclide_dict['Product_m1']

    gain_product = dense[product_idx, parent_idx]
    gain_product_m1 = dense[product_m1_idx, parent_idx]

    assert np.isclose(gain_product, 0.7), \
        f"Expected gain term 0.7, got {gain_product}."

    assert np.isclose(gain_product_m1, 0.3), \
        f"Expected gain term 0.3, got {gain_product_m1}."

    loss_parent = dense[parent_idx, parent_idx]
    assert loss_parent < 0, "Loss term should be negative"


def test_form_matrix_without_isomeric_uses_chain_br():
    """Verify form_matrix uses chain's br when no isomeric branching is provided."""
    import openmc.deplete
    from openmc.deplete import ReactionRates

    chain = openmc.deplete.Chain()

    parent = openmc.deplete.Nuclide('Parent')
    parent.add_reaction('(n,gamma)', 'Product', Q=1e6, branching_ratio=0.6)
    chain.add_nuclide(parent)

    product = openmc.deplete.Nuclide('Product')
    chain.add_nuclide(product)

    nuclides = ['Parent', 'Product']
    reactions = ['(n,gamma)']
    rates = ReactionRates(['mat1'], nuclides, reactions)
    rates[0, 0, 0] = 2.0

    rates_2d = rates[0]
    matrix = chain.form_matrix(rates_2d, isomeric_branching=None)

    dense = matrix.toarray()

    parent_idx = chain.nuclide_dict['Parent']
    product_idx = chain.nuclide_dict['Product']

    gain_product = dense[product_idx, parent_idx]

    assert np.isclose(gain_product, 1.2), \
        f"Expected gain term 1.2 (rate * br), got {gain_product}."
