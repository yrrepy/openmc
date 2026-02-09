"""Tests for Phase 14: σ×φ-weighted isomeric branching ratios.

Tests the correct implementation of:
- Phase 14A: No extrapolation of BR outside data range
- Phase 14B: σ×φ weighting instead of φ-only weighting
"""

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock

from openmc.mgxs import GROUP_STRUCTURES
from openmc.deplete.helpers import IsomericBranchingHelper
from openmc.deplete.chain import Chain


def create_mock_chain_with_isomeric_data():
    """Create a mock chain with isomeric branching data for testing."""
    chain = Mock(spec=Chain)

    # Isomeric branching data for Ag109 (n,gamma) -> Ag110/Ag110_m1
    # Energy range: 1e5 to 1e7 eV (subset of full range)
    chain.isomeric_branching = {
        'Ag109': {
            '(n,gamma)': {
                'energies': np.array([1e5, 1e6, 5e6, 1e7]),  # 4 energy points
                'targets': ['Ag110', 'Ag110_m1'],
                'branching_ratios': {
                    'Ag110': np.array([0.9, 0.8, 0.7, 0.6]),
                    'Ag110_m1': np.array([0.1, 0.2, 0.3, 0.4])
                }
            }
        }
    }
    return chain


def create_mock_gendf_library(n_groups, energy_bins, in_range_mask=None,
                              energy_structure='CCFE-709'):
    """Create a mock GENDF library for testing.

    Parameters
    ----------
    n_groups : int
        Number of energy groups
    energy_bins : numpy.ndarray
        Energy bin boundaries
    in_range_mask : numpy.ndarray, optional
        Boolean mask for groups with non-zero XS. If None, uses isomeric
        data energy range (1e5 to 1e7 eV).
    energy_structure : str, optional
        Energy structure name. Default is 'CCFE-709'.
    """
    mock_gendf = Mock()
    mock_gendf.energy_structure = energy_structure
    mock_gendf.energy_bounds = energy_bins.copy()
    mock_gendf.n_groups = n_groups

    if in_range_mask is None:
        # Isomeric data energy range is 1e5 to 1e7 eV
        e_min = 1e5
        e_max = 1e7
        in_range_mask = (energy_bins[:-1] >= e_min) & (energy_bins[:-1] < e_max)

    # GENDF returns multigroup XS (non-zero in isomeric data range)
    gendf_xs = np.zeros(n_groups)
    gendf_xs[in_range_mask] = 1.0
    mock_gendf.get_xs = Mock(return_value=gendf_xs)

    return mock_gendf


# ============================================================================
# Phase 14A Tests: No extrapolation outside data range
# ============================================================================

def test_no_extrapolation_below_threshold():
    """Verify BR array is zero below isomeric data range."""
    chain = create_mock_chain_with_isomeric_data()
    mock_gendf = Mock()
    mock_gendf.energy_structure = 'CCFE-709'
    mock_gendf.energy_bounds = GROUP_STRUCTURES['CCFE-709'].copy()
    helper = IsomericBranchingHelper(chain, mock_gendf)

    # Create arrays simulating energy ranges
    n_groups = 10
    ratios = np.array([0.9, 0.8, 0.7])  # BR values in data range
    iso_indices = np.array([0, 0, 0, 0, 0, 1, 1, 2, 2, 2])
    below_range = np.array([True, True, True, True, False, False, False, False, False, False])
    above_range = np.array([False, False, False, False, False, False, False, False, False, False])
    in_range = ~below_range & ~above_range

    br_array = helper._build_branching_array(
        ratios, iso_indices, below_range, above_range, in_range, n_groups
    )

    # Below-range groups should be ZERO (not extrapolated)
    assert np.all(br_array[below_range] == 0.0), "Below-range groups should be zero"
    # In-range groups should have actual values
    assert np.any(br_array[in_range] > 0), "In-range groups should have non-zero values"


def test_no_extrapolation_above_range():
    """Verify BR array is zero above isomeric data range."""
    chain = create_mock_chain_with_isomeric_data()
    mock_gendf = Mock()
    mock_gendf.energy_structure = 'CCFE-709'
    mock_gendf.energy_bounds = GROUP_STRUCTURES['CCFE-709'].copy()
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

    # Above-range groups should be ZERO (not extrapolated)
    assert np.all(br_array[above_range] == 0.0), "Above-range groups should be zero"


def test_in_range_values_preserved():
    """Verify in-range BR values are correctly mapped."""
    chain = create_mock_chain_with_isomeric_data()
    mock_gendf = Mock()
    mock_gendf.energy_structure = 'CCFE-709'
    mock_gendf.energy_bounds = GROUP_STRUCTURES['CCFE-709'].copy()
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

    # Check in-range values are correctly mapped
    expected_in_range = ratios[iso_indices[in_range]]
    assert np.allclose(br_array[in_range], expected_in_range)


# ============================================================================
# Phase 14B Tests: σ×φ weighting
# ============================================================================

def test_sigma_phi_weighting_basic():
    """Verify σ×φ weighting produces results and sums to 1.0."""
    chain = create_mock_chain_with_isomeric_data()

    # Use CCFE-709 structure (709 groups)
    energy_bins = GROUP_STRUCTURES['CCFE-709']
    n_groups = 709

    # Create mock GENDF library
    mock_gendf = create_mock_gendf_library(n_groups, energy_bins)

    helper = IsomericBranchingHelper(chain, mock_gendf)

    # Create flux spectrum - uniform
    flux_spectrum = np.ones(n_groups)

    # Calculate weighted branching
    result = helper.weighted_branching_ratios(flux_spectrum, energy_bins)

    # With σ×φ weighting, only groups where σ>0 contribute
    assert 'Ag109' in result, f"Ag109 should be in result, got: {result}"
    assert '(n,gamma)' in result['Ag109']

    # Check that ratios sum to 1.0
    ratios = result['Ag109']['(n,gamma)']
    total = sum(ratios.values())
    assert np.isclose(total, 1.0), f"Ratios should sum to 1.0, got {total}"


def test_skip_when_nuclide_not_in_gendf():
    """Verify empty dict returned when nuclide not in GENDF library."""
    chain = create_mock_chain_with_isomeric_data()

    energy_bins = GROUP_STRUCTURES['CCFE-709']
    n_groups = 709

    flux_spectrum = np.ones(n_groups)

    # GENDF library that raises KeyError for Ag109
    mock_gendf = Mock()
    mock_gendf.energy_structure = 'CCFE-709'
    mock_gendf.energy_bounds = energy_bins.copy()
    mock_gendf.get_xs = Mock(side_effect=KeyError("Ag109 not found"))

    helper = IsomericBranchingHelper(chain, mock_gendf)

    result = helper.weighted_branching_ratios(flux_spectrum, energy_bins)

    # Should return empty dict (nuclide not in GENDF)
    assert result == {} or 'Ag109' not in result


def test_gendf_library_required():
    """Verify ValueError when gendf_library is None at construction."""
    chain = create_mock_chain_with_isomeric_data()

    # gendf_library=None should raise ValueError at construction
    with pytest.raises(ValueError, match="gendf_library is required"):
        IsomericBranchingHelper(chain, None)


def test_with_gendf_succeeds():
    """Verify GENDF library path works correctly.

    This is the primary path for both IndependentOperator and CoupledOperator
    isomeric branching calculations.
    """
    chain = create_mock_chain_with_isomeric_data()

    energy_bins = GROUP_STRUCTURES['CCFE-709']
    n_groups = 709

    # Create mock GENDF library
    mock_gendf = create_mock_gendf_library(n_groups, energy_bins)

    helper = IsomericBranchingHelper(chain, mock_gendf)

    flux_spectrum = np.ones(n_groups)

    # Should succeed by fetching from GENDF
    result = helper.weighted_branching_ratios(flux_spectrum, energy_bins)

    # Verify GENDF was called with correct MT (102 for n,gamma)
    mock_gendf.get_xs.assert_called()
    call_args = mock_gendf.get_xs.call_args
    assert call_args[0][0] == 'Ag109'  # nuclide
    assert call_args[0][1] == 102      # MT for (n,gamma)

    # Should have valid results
    assert 'Ag109' in result
    assert '(n,gamma)' in result['Ag109']

    # Ratios should sum to 1.0
    ratios = result['Ag109']['(n,gamma)']
    total = sum(ratios.values())
    assert np.isclose(total, 1.0), f"Ratios should sum to 1.0, got {total}"


def test_error_on_group_mismatch():
    """Verify ValueError raised when GENDF groups don't match flux."""
    chain = create_mock_chain_with_isomeric_data()

    energy_bins = GROUP_STRUCTURES['CCFE-709']
    n_groups = 709

    flux_spectrum = np.ones(n_groups)

    # GENDF library that returns wrong number of groups
    mock_gendf = Mock()
    mock_gendf.energy_structure = 'CCFE-709'
    mock_gendf.energy_bounds = energy_bins.copy()
    mock_gendf.get_xs = Mock(return_value=np.ones(500))  # Wrong number!

    helper = IsomericBranchingHelper(chain, mock_gendf)

    with pytest.raises(ValueError, match="group structure mismatch"):
        helper.weighted_branching_ratios(flux_spectrum, energy_bins)


def test_zero_weight_sum_returns_empty():
    """Verify empty dict when all reaction rate is below threshold."""
    chain = create_mock_chain_with_isomeric_data()

    energy_bins = GROUP_STRUCTURES['CCFE-709']
    n_groups = 709

    # Flux only in low energy region
    flux_spectrum = np.zeros(n_groups)
    flux_spectrum[:100] = 1.0  # Only thermal flux

    # GENDF XS only in high energy region (no overlap with flux)
    gendf_xs = np.zeros(n_groups)
    gendf_xs[650:] = 1.0  # Only fast XS

    mock_gendf = Mock()
    mock_gendf.energy_structure = 'CCFE-709'
    mock_gendf.energy_bounds = energy_bins.copy()
    mock_gendf.get_xs = Mock(return_value=gendf_xs)

    helper = IsomericBranchingHelper(chain, mock_gendf)

    result = helper.weighted_branching_ratios(flux_spectrum, energy_bins)

    # Should return empty (no reaction rate in isomeric data range)
    assert result == {} or '(n,gamma)' not in result.get('Ag109', {})


# ============================================================================
# CoupledOperator Tests
# ============================================================================

def test_coupled_operator_isomeric_disabled_without_gendf():
    """Verify CoupledOperator has no isomeric branching without GENDF library."""
    from openmc.deplete.coupled_operator import CoupledOperator

    # Create minimal mock to test the methods
    mock_coupled = Mock(spec=CoupledOperator)
    mock_coupled._gendf_library = None  # No GENDF library

    # Call the actual methods on a patched instance
    # The _setup_isomeric_branching should set both to None
    CoupledOperator._setup_isomeric_branching(mock_coupled)

    assert mock_coupled._isomeric_helper is None
    assert mock_coupled._isomeric_branching is None

    # _calculate_isomeric_branching should return None
    result = CoupledOperator._calculate_isomeric_branching(mock_coupled)
    assert result is None


def test_coupled_operator_isomeric_disabled_without_direct_with_flux():
    """Verify CoupledOperator disables isomeric branching without DirectWithFluxHelper."""
    from openmc.deplete.coupled_operator import CoupledOperator
    from openmc.deplete.helpers import DirectReactionRateHelper

    # Create minimal mock to test the methods
    mock_coupled = Mock(spec=CoupledOperator)
    mock_coupled._gendf_library = Mock()  # Has GENDF library
    mock_coupled._rate_helper = Mock(spec=DirectReactionRateHelper)  # Wrong helper type
    mock_coupled._isomeric_energy_structure = 'CCFE-709'

    # Should disable because not using DirectWithFluxHelper
    with pytest.warns(UserWarning, match="not 'direct_with_flux'"):
        CoupledOperator._setup_isomeric_branching(mock_coupled)

    assert mock_coupled._isomeric_helper is None
    assert mock_coupled._isomeric_branching is None


def test_coupled_operator_isomeric_enabled_with_gendf():
    """Verify CoupledOperator enables isomeric branching with GENDF + direct_with_flux."""
    from openmc.deplete.coupled_operator import CoupledOperator
    from openmc.deplete.helpers import DirectWithFluxHelper, IsomericBranchingHelper

    # Create minimal mock to test the methods
    mock_coupled = Mock(spec=CoupledOperator)
    mock_gendf = Mock()
    mock_gendf.energy_structure = 'CCFE-709'
    mock_gendf.energy_bounds = GROUP_STRUCTURES['CCFE-709'].copy()
    mock_coupled._gendf_library = mock_gendf
    mock_coupled._rate_helper = Mock(spec=DirectWithFluxHelper)  # Correct helper type
    mock_coupled.chain = create_mock_chain_with_isomeric_data()

    # Should enable isomeric branching
    CoupledOperator._setup_isomeric_branching(mock_coupled)

    # Helper should be created (not None)
    assert mock_coupled._isomeric_helper is not None
    assert isinstance(mock_coupled._isomeric_helper, IsomericBranchingHelper)


def test_coupled_operator_calculate_isomeric_branching():
    """Verify CoupledOperator calculates isomeric branching using flux from tally."""
    from openmc.deplete.coupled_operator import CoupledOperator
    from openmc.deplete.helpers import DirectWithFluxHelper, IsomericBranchingHelper
    from openmc.mgxs import GROUP_STRUCTURES

    # Create mock objects
    mock_coupled = Mock(spec=CoupledOperator)
    mock_coupled.local_mats = ['mat1', 'mat2']

    n_groups = 709
    energy_bins = GROUP_STRUCTURES['CCFE-709']

    # Create mock GENDF library
    mock_gendf = create_mock_gendf_library(n_groups, energy_bins)

    # Create actual helper (not mock)
    chain = create_mock_chain_with_isomeric_data()
    mock_coupled._isomeric_helper = IsomericBranchingHelper(chain, mock_gendf)

    # Mock rate helper with flux spectrum
    mock_rate_helper = Mock(spec=DirectWithFluxHelper)
    mock_rate_helper.energies = energy_bins
    mock_rate_helper.get_flux_spectrum = Mock(return_value=np.ones(n_groups))
    mock_coupled._rate_helper = mock_rate_helper

    # Calculate isomeric branching
    result = CoupledOperator._calculate_isomeric_branching(mock_coupled)

    # Should return list of dicts (one per material)
    assert result is not None
    assert len(result) == 2  # Two materials

    # Each result should have Ag109 data
    for mat_result in result:
        assert 'Ag109' in mat_result
        assert '(n,gamma)' in mat_result['Ag109']
        # Ratios should sum to 1.0
        ratios = mat_result['Ag109']['(n,gamma)']
        total = sum(ratios.values())
        assert np.isclose(total, 1.0)


# ============================================================================
# Phase 14 Step 3 Tests: Unified BR interface in form_matrix()
# ============================================================================

def test_form_matrix_uses_isomeric_br():
    """Verify form_matrix uses path_rate * br from isomeric data.

    This test ensures that when isomeric branching is provided:
    1. The br values from isomeric data are used directly (they sum to 1.0)
    2. The chain's static br is NOT used (isomeric br replaces it)

    The br from flux-weighted isomeric calculation represents the complete
    branching distribution, so it replaces the chain's default br.
    """
    import openmc.deplete
    from openmc.deplete import ReactionRates

    # Create a simple chain with known br values
    chain = openmc.deplete.Chain()

    # Parent nuclide with chain br=0.5 (not 1.0 - this exposes if chain br is incorrectly used)
    parent = openmc.deplete.Nuclide('Parent')
    # Set chain's branching ratio to 0.5 - if form_matrix uses this, results will differ
    parent.add_reaction('(n,gamma)', 'Product', Q=1e6, branching_ratio=0.5)
    chain.add_nuclide(parent)

    # Target nuclides
    product1 = openmc.deplete.Nuclide('Product')
    chain.add_nuclide(product1)
    product2 = openmc.deplete.Nuclide('Product_m1')
    chain.add_nuclide(product2)

    # Create reaction rates
    nuclides = ['Parent', 'Product', 'Product_m1']
    reactions = ['(n,gamma)']
    rates = ReactionRates(['mat1'], nuclides, reactions)
    rates[0, 0, 0] = 1.0  # 1.0 reaction rate for Parent (n,gamma)

    # Isomeric branching with known br values
    # br values sum to 1.0 (complete branching distribution)
    isomeric_branching = {
        'Parent': {
            '(n,gamma)': {
                'Product': 0.7,     # 70% to ground state
                'Product_m1': 0.3   # 30% to metastable
            }
        }
    }

    # Form matrix with isomeric branching
    rates_2d = rates[0]
    matrix = chain.form_matrix(rates_2d, isomeric_branching=isomeric_branching)

    # Convert to dense for inspection
    dense = matrix.toarray()

    # Get nuclide indices
    parent_idx = chain.nuclide_dict['Parent']
    product_idx = chain.nuclide_dict['Product']
    product_m1_idx = chain.nuclide_dict['Product_m1']

    # Check the gain terms in the matrix
    # Matrix[product_idx, parent_idx] should be path_rate * br = 1.0 * 0.7 = 0.7
    # If chain br was also used: 1.0 * 0.5 * 0.7 = 0.35 (WRONG - double counting)
    gain_product = dense[product_idx, parent_idx]
    gain_product_m1 = dense[product_m1_idx, parent_idx]

    # Verify isomeric br is used directly (not multiplied by chain br)
    assert np.isclose(gain_product, 0.7), \
        f"Expected gain term 0.7, got {gain_product}. " \
        "form_matrix may be incorrectly using chain br instead of isomeric br."

    assert np.isclose(gain_product_m1, 0.3), \
        f"Expected gain term 0.3, got {gain_product_m1}. " \
        "form_matrix may be incorrectly using chain br instead of isomeric br."

    # Also verify the loss term (should still use full reaction rate)
    loss_parent = dense[parent_idx, parent_idx]
    assert loss_parent < 0, "Loss term should be negative"


def test_form_matrix_without_isomeric_uses_chain_br():
    """Verify form_matrix uses chain's br when no isomeric branching is provided.

    This ensures backward compatibility - without isomeric branching data,
    the chain's static branching ratio should be used.
    """
    import openmc.deplete
    from openmc.deplete import ReactionRates

    chain = openmc.deplete.Chain()

    # Parent with br=0.6
    parent = openmc.deplete.Nuclide('Parent')
    parent.add_reaction('(n,gamma)', 'Product', Q=1e6, branching_ratio=0.6)
    chain.add_nuclide(parent)

    # Target
    product = openmc.deplete.Nuclide('Product')
    chain.add_nuclide(product)

    # Create reaction rates
    nuclides = ['Parent', 'Product']
    reactions = ['(n,gamma)']
    rates = ReactionRates(['mat1'], nuclides, reactions)
    rates[0, 0, 0] = 2.0  # 2.0 reaction rate

    # Form matrix WITHOUT isomeric branching
    rates_2d = rates[0]
    matrix = chain.form_matrix(rates_2d, isomeric_branching=None)

    dense = matrix.toarray()

    parent_idx = chain.nuclide_dict['Parent']
    product_idx = chain.nuclide_dict['Product']

    # Gain term should be path_rate * br = 2.0 * 0.6 = 1.2
    gain_product = dense[product_idx, parent_idx]

    assert np.isclose(gain_product, 1.2), \
        f"Expected gain term 1.2 (rate * br), got {gain_product}. " \
        "form_matrix should use chain br when no isomeric branching."
