"""Tests for GENDFFluxCollapseHelper (gendf-flux reaction rate mode).

The helper is exercised without an initialized openmc.lib session by setting
the state that generate_tallies would normally create (materials, scores,
MTs, tally mean caches) directly, following the mock pattern of
test_gendf_nuclide_filtering.py.
"""

import warnings

import numpy as np
import pytest
from unittest.mock import Mock

from openmc.deplete.helpers import GENDFFluxCollapseHelper
from openmc.mgxs import GROUP_STRUCTURES


ENERGIES = GROUP_STRUCTURES['CCFE-709']
NG = len(ENERGIES) - 1


class MockGENDFLibrary:
    """Mock GENDF library with per-nuclide MT -> xs arrays."""

    energy_structure = 'CCFE-709'
    n_groups = NG

    def __init__(self, xs_by_nuclide):
        self._xs = xs_by_nuclide

    def available_nuclides_set(self):
        return frozenset(self._xs)

    def get_all_xs(self, nuclide, mts=None, strict_alignment=True):
        if nuclide not in self._xs:
            raise KeyError(nuclide)
        all_xs = self._xs[nuclide]
        if mts is not None:
            mts_set = set(mts)
            return {mt: xs for mt, xs in all_xs.items() if mt in mts_set}
        return all_xs


def _make_helper(gendf, nuclides, scores, mts, fluxes, reactions=()):
    """Build a helper with generate_tallies state injected manually."""
    n_mats = fluxes.shape[0]
    helper = GENDFFluxCollapseHelper(
        len(nuclides), len(scores), gendf, reactions=list(reactions))
    helper._materials = [Mock() for _ in range(n_mats)]
    helper._scores = list(scores)
    helper._mts = list(mts)
    helper._nuclides = list(nuclides)
    helper._flux_tally_means_cache = fluxes.reshape(-1, 1)
    return helper


def test_collapse_matches_hand_calculation():
    """Rates must equal the hand-computed sigma_g . phi_g dot product."""
    rng = np.random.default_rng(42)
    xs_ng = rng.random(NG)
    xs_n2n = rng.random(NG)
    xs_nnp = rng.random(NG)
    gendf = MockGENDFLibrary({
        'Al27': {102: xs_ng, 4: xs_nnp},
        'Fe56': {102: 3 * xs_ng, 16: xs_n2n},
    })
    flux = rng.random((2, NG))

    helper = _make_helper(
        gendf, ['Al27', 'Fe56'], ['(n,gamma)', "(n,n')", '(n,2n)'],
        [102, 4, 16], flux)

    rates = helper.get_material_rates(1, [0, 1], [0, 1, 2])
    expected = np.array([
        [xs_ng @ flux[1], xs_nnp @ flux[1], 0.0],
        [3 * xs_ng @ flux[1], 0.0, xs_n2n @ flux[1]],
    ])
    assert np.allclose(rates, expected)


def test_missing_nuclide_zero_rates_and_single_warning():
    """Nuclides absent from GENDF get zero rates and one warning total."""
    gendf = MockGENDFLibrary({'Al27': {102: np.ones(NG)}})
    flux = np.ones((1, NG))
    helper = _make_helper(gendf, ['Al27', 'Co59'], ['(n,gamma)'], [102], flux)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        rates = helper.get_material_rates(0, [0, 1], [0])
        helper.get_material_rates(0, [0, 1], [0])

    assert rates[0, 0] == pytest.approx(NG)
    assert rates[1, 0] == 0.0
    messages = [str(w.message) for w in record if 'GENDF' in str(w.message)]
    assert len(messages) == 1
    assert 'Co59' in messages[0]


def test_direct_tally_overrides_collapse():
    """Directly tallied scores replace the GENDF collapse values."""
    gendf = MockGENDFLibrary({'U235': {102: np.full(NG, 2.0),
                                       18: np.full(NG, 5.0)}})
    flux = np.ones((1, NG))
    helper = _make_helper(
        gendf, ['U235', 'Co59'], ['(n,gamma)', 'fission'], [102, 18], flux,
        reactions=['fission'])

    helper._rate_tally = Mock()
    helper._rate_tally.nuclides = ['U235', 'Co59']
    helper._rate_tally_means_cache = np.array([[999.0, 777.0]])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rates = helper.get_material_rates(0, [0, 1], [0, 1])

    # (n,gamma) from GENDF collapse; fission from the direct tally, even for
    # the nuclide missing from the GENDF library
    assert rates[0, 0] == pytest.approx(2.0 * NG)
    assert rates[0, 1] == pytest.approx(999.0)
    assert rates[1, 0] == 0.0
    assert rates[1, 1] == pytest.approx(777.0)


def test_isomeric_branching_surface():
    """energies and get_flux_spectrum match DirectWithFluxHelper's surface."""
    gendf = MockGENDFLibrary({'Al27': {102: np.ones(NG)}})
    flux = np.vstack([np.full(NG, 1.5), np.full(NG, 4.0)])
    helper = _make_helper(gendf, ['Al27'], ['(n,gamma)'], [102], flux)

    assert np.array_equal(helper.energies, ENERGIES)
    assert np.allclose(helper.get_flux_spectrum(0), 1.5)
    assert np.allclose(helper.get_flux_spectrum(1), 4.0)


def test_table_rebuild_on_nuclide_growth():
    """Adding nuclides between steps triggers a table rebuild."""
    gendf = MockGENDFLibrary({'Al27': {102: np.full(NG, 2.0)},
                              'Fe56': {102: np.full(NG, 3.0)}})
    flux = np.ones((1, NG))
    helper = _make_helper(gendf, ['Al27'], ['(n,gamma)'], [102], flux)

    rates = helper.get_material_rates(0, [0], [0])
    assert rates[0, 0] == pytest.approx(2.0 * NG)

    helper._nuclides = ['Al27', 'Fe56']
    helper._results_cache = np.empty((2, 1))
    rates = helper.get_material_rates(0, [0, 1], [0])
    assert rates[0, 0] == pytest.approx(2.0 * NG)
    assert rates[1, 0] == pytest.approx(3.0 * NG)


def test_undetected_energy_structure_raises():
    """A library without a detected group structure is rejected."""
    gendf = Mock()
    gendf.energy_structure = None
    with pytest.raises(ValueError, match='energy group structure'):
        GENDFFluxCollapseHelper(1, 1, gendf)
