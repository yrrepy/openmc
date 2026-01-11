"""Tests for GENDF nuclide filtering consistency with CE workflow.

These tests verify that the GENDF workflow filters nuclides at MicroXS creation
time, matching the CE HDF5 workflow pattern where only nuclides with actual
cross-section data appear in nuclides_with_data.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from openmc.deplete import MicroXS


# --- Mock classes for testing ---

class MockGENDFLibrary:
    """Mock GENDF library for testing nuclide filtering."""

    def __init__(self, available_nuclides):
        """
        Parameters
        ----------
        available_nuclides : set
            Set of nuclide names that have GENDF data
        """
        self._available = set(available_nuclides)
        self.energy_structure = 'CCFE-709'
        # CCFE-709 has 710 energy boundaries (709 groups)
        self.energy_bounds = np.logspace(-5, np.log10(2e7), 710)

    def available_nuclides_set(self):
        return self._available

    def get_xs(self, nuclide, mt, energies):
        """Return mock cross-section data."""
        if nuclide not in self._available:
            raise KeyError(f"Nuclide {nuclide} not in GENDF library")
        # Return 709 group values (for CCFE-709)
        return np.ones(709) * 1.0  # 1 barn for all groups


class MockChain:
    """Mock chain with configurable nuclides."""

    def __init__(self, nuclide_names):
        self.nuclides = [Mock(name=n) for n in nuclide_names]
        for nuc, name in zip(self.nuclides, nuclide_names):
            nuc.name = name
        self.reactions = ['(n,gamma)', 'fission']


def _create_microxs_with_mocks(chain_nuclides, gendf_nuclides, user_nuclides=None):
    """Helper to create MicroXS with mocked chain and GENDF type check."""
    mock_gendf = MockGENDFLibrary(gendf_nuclides)
    mock_chain = MockChain(chain_nuclides)

    flux = np.ones(709)  # CCFE-709
    energies = mock_gendf.energy_bounds

    # Patch chain loading and add MockGENDFLibrary to valid GENDF types
    with patch('openmc.deplete.microxs._get_chain', return_value=mock_chain):
        import openmc.deplete.microxs as microxs_mod
        original_types = microxs_mod._GENDF_TYPES
        try:
            microxs_mod._GENDF_TYPES = (MockGENDFLibrary,) + original_types
            micro_xs = MicroXS.from_multigroup_flux_with_gendf(
                energies=energies,
                multigroup_flux=flux,
                gendf_library=mock_gendf,
                chain_file='dummy_chain.xml',
                nuclides=user_nuclides
            )
        finally:
            microxs_mod._GENDF_TYPES = original_types

    return micro_xs


# --- Test functions ---

def test_microxs_only_contains_gendf_nuclides():
    """MicroXS should only include nuclides present in GENDF library."""
    # Setup: Chain has 5 nuclides, but only 3 have GENDF data
    chain_nuclides = ['U235', 'U238', 'Pu239', 'Am241', 'Cm244']
    gendf_nuclides = {'U235', 'U238', 'Pu239'}  # Am241, Cm244 missing

    micro_xs = _create_microxs_with_mocks(chain_nuclides, gendf_nuclides)

    # Verify: Only nuclides with GENDF data are in MicroXS
    assert set(micro_xs.nuclides) == gendf_nuclides
    assert len(micro_xs.nuclides) == 3

    # Verify: Nuclides without GENDF data are NOT in MicroXS
    assert 'Am241' not in micro_xs.nuclides
    assert 'Cm244' not in micro_xs.nuclides


def test_microxs_excludes_nuclides_without_gendf():
    """Nuclides without GENDF data should not appear in MicroXS."""
    chain_nuclides = ['H1', 'He4', 'Li6', 'Be9', 'B10']
    gendf_nuclides = {'Li6', 'B10'}  # Only 2 of 5 available

    micro_xs = _create_microxs_with_mocks(chain_nuclides, gendf_nuclides)

    # Only Li6 and B10 should be present
    assert len(micro_xs.nuclides) == 2
    assert 'Li6' in micro_xs.nuclides
    assert 'B10' in micro_xs.nuclides

    # H1, He4, Be9 should NOT be present
    for missing in ['H1', 'He4', 'Be9']:
        assert missing not in micro_xs.nuclides


def test_user_provided_nuclides_filtered():
    """User-provided nuclide list should be filtered to GENDF availability."""
    chain_nuclides = ['U235', 'U238', 'Pu239']
    gendf_nuclides = {'U235', 'Pu239'}  # U238 NOT in GENDF

    # User explicitly requests nuclides, including one not in GENDF
    user_nuclides = ['U235', 'U238', 'Pu239']

    micro_xs = _create_microxs_with_mocks(chain_nuclides, gendf_nuclides, user_nuclides)

    # U238 should be filtered out
    assert 'U238' not in micro_xs.nuclides
    assert set(micro_xs.nuclides) == {'U235', 'Pu239'}


def test_empty_result_if_no_gendf_nuclides():
    """If no chain nuclides have GENDF data, MicroXS should be empty."""
    chain_nuclides = ['Og294', 'Ts293', 'Lv292']  # Fictional/rare nuclides
    gendf_nuclides = set()  # Empty GENDF library

    micro_xs = _create_microxs_with_mocks(chain_nuclides, gendf_nuclides)

    # Should have zero nuclides
    assert len(micro_xs.nuclides) == 0


def test_all_nuclides_included_when_all_have_data():
    """When all chain nuclides have GENDF data, all should be included."""
    chain_nuclides = ['U235', 'U238', 'Pu239']
    gendf_nuclides = {'U235', 'U238', 'Pu239'}  # All available

    micro_xs = _create_microxs_with_mocks(chain_nuclides, gendf_nuclides)

    # All nuclides should be present
    assert set(micro_xs.nuclides) == gendf_nuclides
    assert len(micro_xs.nuclides) == 3


def test_nuclide_order_preserved():
    """Nuclide order from chain should be preserved (minus filtered ones)."""
    chain_nuclides = ['A', 'B', 'C', 'D', 'E']
    gendf_nuclides = {'A', 'C', 'E'}  # B, D filtered out

    micro_xs = _create_microxs_with_mocks(chain_nuclides, gendf_nuclides)

    # Order should be A, C, E (chain order, with B, D removed)
    assert micro_xs.nuclides == ['A', 'C', 'E']


def test_array_size_matches_filtered_nuclides():
    """MicroXS data array should only have rows for filtered nuclides."""
    chain_nuclides = ['U235', 'U238', 'Pu239', 'Am241', 'Cm244']
    gendf_nuclides = {'U235', 'Pu239'}  # Only 2 of 5

    micro_xs = _create_microxs_with_mocks(chain_nuclides, gendf_nuclides)

    # Array should have shape (2 nuclides, 2 reactions, 1 group)
    assert micro_xs.data.shape[0] == 2  # Only 2 nuclides
    assert micro_xs.data.shape[0] == len(micro_xs.nuclides)


def test_no_zero_rows_from_missing_nuclides():
    """There should be no all-zero rows from missing nuclides."""
    chain_nuclides = ['U235', 'U238', 'Pu239']
    gendf_nuclides = {'U235', 'Pu239'}  # U238 missing

    micro_xs = _create_microxs_with_mocks(chain_nuclides, gendf_nuclides)

    # Every row should have at least one non-zero value
    # (since mock returns 1.0 for all XS)
    for i, nuc in enumerate(micro_xs.nuclides):
        row_sum = micro_xs.data[i, :, :].sum()
        assert row_sum > 0, f"Row for {nuc} is all zeros"
