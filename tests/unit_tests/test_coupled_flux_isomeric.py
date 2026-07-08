"""Tests for isomeric branching support in CoupledOperator flux modes.

Covers the FluxCollapseHelper flux-spectrum surface added for isomeric
branching and the _setup_isomeric_branching validation, exercised on a bare
CoupledOperator instance (no transport, no openmc.lib session).
"""

import warnings

import numpy as np
import pytest
from unittest.mock import Mock

from openmc.deplete import CoupledOperator
from openmc.deplete.helpers import (
    DirectReactionRateHelper, FluxCollapseHelper, GENDFFluxCollapseHelper)
from openmc.mgxs import GROUP_STRUCTURES


ENERGIES = GROUP_STRUCTURES['CCFE-709']
NG = len(ENERGIES) - 1


class MockGENDFLibrary:
    energy_structure = 'CCFE-709'
    energy_bounds = ENERGIES
    n_groups = NG

    def available_nuclides_set(self):
        return frozenset()

    def get_branching_ratios(self, nuclide, mt, target_names=None,
                             lfs_values=None):
        raise KeyError(nuclide)


def _bare_operator(rate_helper, targets):
    """CoupledOperator skeleton with just the state isomeric setup needs."""
    op = CoupledOperator.__new__(CoupledOperator)
    op._gendf_library = MockGENDFLibrary()
    op._rate_helper = rate_helper
    op.chain = Mock(isomeric_branching_targets=targets)
    op._isomeric_helper = None
    op._isomeric_branching = None
    return op


def test_flux_collapse_helper_spectrum_surface():
    """FluxCollapseHelper exposes energies and per-material flux spectra."""
    helper = FluxCollapseHelper(2, 2, ENERGIES)
    helper._materials = [Mock(), Mock()]
    flux = np.vstack([np.full(NG, 2.0), np.full(NG, 7.0)])
    helper._flux_tally_means_cache = flux.reshape(-1, 1)

    assert np.array_equal(helper.energies, ENERGIES)
    assert np.allclose(helper.get_flux_spectrum(0), 2.0)
    assert np.allclose(helper.get_flux_spectrum(1), 7.0)


def test_isomeric_enabled_with_matching_energies():
    """flux mode with the GENDF group structure enables isomeric branching."""
    helper = FluxCollapseHelper(1, 1, ENERGIES)
    op = _bare_operator(helper, targets={'Al27': {'(n,gamma)': ['Al28']}})
    op._setup_isomeric_branching()
    assert op._isomeric_helper is not None


def test_isomeric_enabled_with_gendf_flux_helper():
    """gendf-flux mode passes the flux-spectrum gate."""
    helper = GENDFFluxCollapseHelper(1, 1, MockGENDFLibrary())
    op = _bare_operator(helper, targets={'Al27': {'(n,gamma)': ['Al28']}})
    op._setup_isomeric_branching()
    assert op._isomeric_helper is not None


def test_mismatched_energies_raises_with_targets():
    """Wrong flux group structure is a hard error when the chain has data."""
    helper = FluxCollapseHelper(1, 1, np.array([0.0, 1e6, 2e7]))
    op = _bare_operator(helper, targets={'Al27': {'(n,gamma)': ['Al28']}})
    with pytest.raises(ValueError, match='CCFE-709'):
        op._setup_isomeric_branching()


def test_mismatched_energies_disabled_without_targets():
    """Without chain isomeric data, a mismatch just disables the helper."""
    helper = FluxCollapseHelper(1, 1, np.array([0.0, 1e6, 2e7]))
    op = _bare_operator(helper, targets={})
    op._setup_isomeric_branching()
    assert op._isomeric_helper is None


def test_direct_helper_disables_with_warning():
    """Helpers without a flux spectrum disable isomeric branching."""
    helper = DirectReactionRateHelper(1, 1)
    op = _bare_operator(helper, targets={'Al27': {'(n,gamma)': ['Al28']}})
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        op._setup_isomeric_branching()
    assert op._isomeric_helper is None
    assert any('flux spectrum' in str(w.message) for w in record)


def test_calculate_isomeric_branching_uses_helper_surface():
    """Flux/energy pairs are built from the rate helper via duck typing."""
    rate_helper = Mock(spec=['energies', 'get_flux_spectrum'])
    rate_helper.energies = ENERGIES
    rate_helper.get_flux_spectrum = lambda i: np.full(NG, float(i + 1))

    op = CoupledOperator.__new__(CoupledOperator)
    op._rate_helper = rate_helper
    op.local_mats = ['1', '2']
    op._isomeric_helper = Mock()
    op._isomeric_helper.compute_for_materials.return_value = {'ok': True}

    result = op._calculate_isomeric_branching()
    assert result == {'ok': True}
    pairs = op._isomeric_helper.compute_for_materials.call_args[0][0]
    assert len(pairs) == 2
    assert np.allclose(pairs[0][0], 1.0)
    assert np.allclose(pairs[1][0], 2.0)
    assert np.array_equal(pairs[0][1], ENERGIES)
