"""Tests for Phase 14: σ×φ-weighted isomeric branching ratios.

Tests the correct implementation of:
- Phase 14A: No extrapolation of BR outside data range
- Phase 14B: σ×φ weighting instead of φ-only weighting
"""

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock

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


class MockMicroXS:
    """Mock MicroXS class for testing."""

    def __init__(self, nuclides_reactions):
        self._data = nuclides_reactions

    def __getitem__(self, index):
        nuc, rx = index
        if (nuc, rx) in self._data:
            return self._data[(nuc, rx)]
        raise KeyError(f"{nuc} {rx} not found")


def create_mock_microxs(nuclides_reactions, n_groups):
    """Create a mock MicroXS object.

    Parameters
    ----------
    nuclides_reactions : dict
        Dict of {(nuclide, reaction): xs_array}
    n_groups : int
        Number of energy groups (unused, for documentation)
    """
    return MockMicroXS(nuclides_reactions)


# ============================================================================
# Phase 14A Tests: No extrapolation outside data range
# ============================================================================

def test_no_extrapolation_below_threshold():
    """Verify BR array is zero below isomeric data range."""
    chain = create_mock_chain_with_isomeric_data()
    helper = IsomericBranchingHelper(chain, energy_structure='CCFE-709')

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
    helper = IsomericBranchingHelper(chain, energy_structure='CCFE-709')

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
    helper = IsomericBranchingHelper(chain, energy_structure='CCFE-709')

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
    helper = IsomericBranchingHelper(chain, energy_structure='CCFE-709')

    # Use CCFE-709 structure (709 groups)
    from openmc.mgxs import GROUP_STRUCTURES
    energy_bins = GROUP_STRUCTURES['CCFE-709']
    n_groups = 709

    # Isomeric data energy range is 1e5 to 1e7 eV
    # Find the group indices for this range
    e_min = 1e5
    e_max = 1e7
    in_range_mask = (energy_bins[:-1] >= e_min) & (energy_bins[:-1] < e_max)

    # Create flux spectrum - uniform
    flux_spectrum = np.ones(n_groups)

    # Create MicroXS - non-zero in the isomeric data energy range
    sigma_g = np.zeros(n_groups)
    sigma_g[in_range_mask] = 1.0  # XS in range of isomeric data

    micro_xs = create_mock_microxs(
        {('Ag109', '(n,gamma)'): sigma_g},
        n_groups
    )

    # Calculate weighted branching
    result = helper.weighted_branching_ratios(flux_spectrum, energy_bins, micro_xs)

    # With σ×φ weighting, only groups where σ>0 contribute
    assert 'Ag109' in result, f"Ag109 should be in result, got: {result}"
    assert '(n,gamma)' in result['Ag109']

    # Check that ratios sum to 1.0
    ratios = result['Ag109']['(n,gamma)']
    total = sum(ratios.values())
    assert np.isclose(total, 1.0), f"Ratios should sum to 1.0, got {total}"


def test_skip_when_nuclide_not_in_microxs():
    """Verify empty dict returned when nuclide not in MicroXS."""
    chain = create_mock_chain_with_isomeric_data()
    helper = IsomericBranchingHelper(chain, energy_structure='CCFE-709')

    from openmc.mgxs import GROUP_STRUCTURES
    energy_bins = GROUP_STRUCTURES['CCFE-709']
    n_groups = 709

    flux_spectrum = np.ones(n_groups)

    # MicroXS with NO Ag109 data
    micro_xs = create_mock_microxs({}, n_groups)

    result = helper.weighted_branching_ratios(flux_spectrum, energy_bins, micro_xs)

    # Should return empty dict (nuclide not in MicroXS)
    assert result == {} or 'Ag109' not in result


def test_single_group_microxs_skipped_without_gendf():
    """Verify single-group MicroXS returns empty dict when no GENDF library.

    Phase 14C changed the behavior: instead of raising ValueError, we now
    return an empty dict (graceful degradation) when single-group MicroXS
    is detected and no GENDF library is available for on-the-fly lookup.
    """
    chain = create_mock_chain_with_isomeric_data()
    # No gendf_library provided
    helper = IsomericBranchingHelper(chain, energy_structure='CCFE-709')

    from openmc.mgxs import GROUP_STRUCTURES
    energy_bins = GROUP_STRUCTURES['CCFE-709']
    n_groups = 709

    flux_spectrum = np.ones(n_groups)

    # Single-group MicroXS (collapsed)
    sigma_g = np.array([1.0])  # Only 1 group!
    micro_xs = create_mock_microxs(
        {('Ag109', '(n,gamma)'): sigma_g},
        1
    )

    # Should return empty dict (skip isomeric branching for this reaction)
    result = helper.weighted_branching_ratios(flux_spectrum, energy_bins, micro_xs)
    assert result == {} or 'Ag109' not in result


def test_single_group_microxs_with_gendf_succeeds():
    """Verify single-group MicroXS with GENDF library fetches multigroup XS.

    Phase 14C: When MicroXS has single-group (collapsed) data but a GENDF
    library is available, the helper should fetch multigroup XS from GENDF
    and successfully calculate weighted branching ratios.
    """
    chain = create_mock_chain_with_isomeric_data()

    from openmc.mgxs import GROUP_STRUCTURES
    energy_bins = GROUP_STRUCTURES['CCFE-709']
    n_groups = 709

    # Mock GENDF library that returns multigroup XS
    mock_gendf = Mock()

    # Isomeric data energy range is 1e5 to 1e7 eV
    e_min = 1e5
    e_max = 1e7
    in_range_mask = (energy_bins[:-1] >= e_min) & (energy_bins[:-1] < e_max)

    # GENDF returns multigroup XS (non-zero in isomeric data range)
    gendf_xs = np.zeros(n_groups)
    gendf_xs[in_range_mask] = 1.0
    mock_gendf.get_xs = Mock(return_value=gendf_xs)

    helper = IsomericBranchingHelper(
        chain, energy_structure='CCFE-709', gendf_library=mock_gendf
    )

    flux_spectrum = np.ones(n_groups)

    # Single-group MicroXS (collapsed) - should trigger GENDF lookup
    sigma_g = np.array([1.0])  # Only 1 group!
    micro_xs = create_mock_microxs(
        {('Ag109', '(n,gamma)'): sigma_g},
        1
    )

    # Should succeed by fetching from GENDF
    result = helper.weighted_branching_ratios(flux_spectrum, energy_bins, micro_xs)

    # Verify GENDF was called with correct MT (102 for n,gamma)
    mock_gendf.get_xs.assert_called()

    # Should have valid results
    assert 'Ag109' in result
    assert '(n,gamma)' in result['Ag109']

    # Ratios should sum to 1.0
    ratios = result['Ag109']['(n,gamma)']
    total = sum(ratios.values())
    assert np.isclose(total, 1.0), f"Ratios should sum to 1.0, got {total}"


def test_no_microxs_with_gendf_succeeds():
    """Verify micro_xs=None with GENDF library works (for CoupledOperator).

    Phase 14C: When micro_xs is None but gendf_library is provided,
    cross-sections should be fetched directly from GENDF library.
    This is the primary path for CoupledOperator isomeric branching.
    """
    chain = create_mock_chain_with_isomeric_data()

    from openmc.mgxs import GROUP_STRUCTURES
    energy_bins = GROUP_STRUCTURES['CCFE-709']
    n_groups = 709

    # Mock GENDF library that returns multigroup XS
    mock_gendf = Mock()

    # Isomeric data energy range is 1e5 to 1e7 eV
    e_min = 1e5
    e_max = 1e7
    in_range_mask = (energy_bins[:-1] >= e_min) & (energy_bins[:-1] < e_max)

    # GENDF returns multigroup XS (non-zero in isomeric data range)
    gendf_xs = np.zeros(n_groups)
    gendf_xs[in_range_mask] = 1.0
    mock_gendf.get_xs = Mock(return_value=gendf_xs)

    helper = IsomericBranchingHelper(
        chain, energy_structure='CCFE-709', gendf_library=mock_gendf
    )

    flux_spectrum = np.ones(n_groups)

    # Pass micro_xs=None - should fetch directly from GENDF
    result = helper.weighted_branching_ratios(flux_spectrum, energy_bins, micro_xs=None)

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


def test_no_microxs_without_gendf_raises():
    """Verify ValueError when micro_xs=None and no GENDF library."""
    chain = create_mock_chain_with_isomeric_data()
    # No gendf_library provided
    helper = IsomericBranchingHelper(chain, energy_structure='CCFE-709')

    from openmc.mgxs import GROUP_STRUCTURES
    energy_bins = GROUP_STRUCTURES['CCFE-709']
    n_groups = 709

    flux_spectrum = np.ones(n_groups)

    # micro_xs=None and no GENDF - should raise ValueError
    with pytest.raises(ValueError, match="no gendf_library was provided"):
        helper.weighted_branching_ratios(flux_spectrum, energy_bins, micro_xs=None)


def test_error_on_group_mismatch():
    """Verify ValueError raised when MicroXS groups don't match flux."""
    chain = create_mock_chain_with_isomeric_data()
    helper = IsomericBranchingHelper(chain, energy_structure='CCFE-709')

    from openmc.mgxs import GROUP_STRUCTURES
    energy_bins = GROUP_STRUCTURES['CCFE-709']
    n_groups = 709

    flux_spectrum = np.ones(n_groups)

    # MicroXS with wrong number of groups
    sigma_g = np.ones(500)  # Wrong number!
    micro_xs = create_mock_microxs(
        {('Ag109', '(n,gamma)'): sigma_g},
        500
    )

    with pytest.raises(ValueError, match="group structure mismatch"):
        helper.weighted_branching_ratios(flux_spectrum, energy_bins, micro_xs)


def test_zero_weight_sum_returns_empty():
    """Verify empty dict when all reaction rate is below threshold."""
    chain = create_mock_chain_with_isomeric_data()
    helper = IsomericBranchingHelper(chain, energy_structure='CCFE-709')

    from openmc.mgxs import GROUP_STRUCTURES
    energy_bins = GROUP_STRUCTURES['CCFE-709']
    n_groups = 709

    # Flux only in low energy region
    flux_spectrum = np.zeros(n_groups)
    flux_spectrum[:100] = 1.0  # Only thermal flux

    # XS only in high energy region (no overlap with flux)
    sigma_g = np.zeros(n_groups)
    sigma_g[650:] = 1.0  # Only fast XS

    micro_xs = create_mock_microxs(
        {('Ag109', '(n,gamma)'): sigma_g},
        n_groups
    )

    result = helper.weighted_branching_ratios(flux_spectrum, energy_bins, micro_xs)

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
    mock_coupled._gendf_library = Mock()  # Has GENDF library
    mock_coupled._rate_helper = Mock(spec=DirectWithFluxHelper)  # Correct helper type
    mock_coupled._isomeric_energy_structure = 'CCFE-709'
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

    # Mock GENDF library
    mock_gendf = Mock()
    n_groups = 709
    energy_bins = GROUP_STRUCTURES['CCFE-709']

    # GENDF returns multigroup XS (non-zero in isomeric data range)
    e_min = 1e5
    e_max = 1e7
    in_range_mask = (energy_bins[:-1] >= e_min) & (energy_bins[:-1] < e_max)
    gendf_xs = np.zeros(n_groups)
    gendf_xs[in_range_mask] = 1.0
    mock_gendf.get_xs = Mock(return_value=gendf_xs)

    # Create actual helper (not mock)
    chain = create_mock_chain_with_isomeric_data()
    mock_coupled._isomeric_helper = IsomericBranchingHelper(
        chain, energy_structure='CCFE-709', gendf_library=mock_gendf
    )

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
