"""Tests for the FluxCollapseHelper per-temperature group cross section cache.

Split out of ``test_deplete_activation.py`` so the upstream-owned file stays
free of branch-added tests.
"""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

import openmc.deplete.microxs as microxs_mod
import openmc.lib
from openmc.data import REACTION_MT
from openmc.deplete.helpers import FluxCollapseHelper
from openmc.mgxs import GROUP_STRUCTURES


CASMO40 = np.asarray(GROUP_STRUCTURES['CASMO-40'], dtype=float)
N_GROUPS = CASMO40.size - 1
TEMPERATURE = 293.6


def _wire_helper(nuclides, scores, flux, n_nucs=None,
                 reactions_direct=None, nuclides_direct=None):
    """Wire a FluxCollapseHelper for one material without building tallies."""
    helper = FluxCollapseHelper(
        n_nucs or len(nuclides), len(scores), CASMO40,
        reactions=reactions_direct, nuclides=nuclides_direct)
    helper._materials = [SimpleNamespace(temperature=TEMPERATURE)]
    helper._xs_tables = {}
    helper._mts = [REACTION_MT[s] for s in scores]
    helper._scores = list(scores)
    helper._flux_tally_means_cache = np.asarray(flux, dtype=float)
    helper.nuclides = list(nuclides)
    return helper


def test_flux_collapse_helper_cache_lifecycle(monkeypatch):
    """The FluxCollapseHelper per-temperature cache lifecycle: flux-path rates
    equal collapse_rate; the table is built once per temperature; re-setting the
    same nuclides (as happens each timestep) does not rebuild it; and growing the
    nuclide set clears the cache, so the table is rebuilt and the newly-added
    nuclide gets correct (non-stale) rates.
    """
    nuclides = ['U235', 'U238', 'O16']
    grown = nuclides + ['Pu239']  # larger set, added later
    scores = ['fission', '(n,gamma)', '(n,2n)']  # (n,2n) is a threshold reaction
    flux = np.random.default_rng(0).random(N_GROUPS)
    react_index = list(range(len(scores)))

    build_spy = mock.MagicMock(wraps=microxs_mod._build_xs_table_ce)
    monkeypatch.setattr(microxs_mod, '_build_xs_table_ce', build_spy)

    with openmc.lib.TemporarySession():
        # Size the result cache for the larger (grown) set from the start
        helper = _wire_helper(nuclides, scores, flux, n_nucs=len(grown))
        rates = helper.get_material_rates(
            0, list(range(len(nuclides))), react_index).copy()

        # Flux-path rate equals collapse_rate for every (nuclide, score) ...
        for i, name in enumerate(nuclides):
            nuc = openmc.lib.nuclides[name]
            for j, s in enumerate(scores):
                expected = nuc.collapse_rate(
                    REACTION_MT[s], TEMPERATURE, CASMO40, flux)
                assert rates[i, j] == pytest.approx(expected, rel=1e-10)

        # ... built once for the single temperature ...
        assert build_spy.call_count == 1

        # ... and re-setting the same nuclide list (each step) does not rebuild
        helper.nuclides = list(nuclides)
        helper.get_material_rates(0, list(range(len(nuclides))), react_index)
        assert build_spy.call_count == 1

        # Growing the set clears the cache -> exactly one rebuild at the same T
        helper.nuclides = list(grown)
        rates = helper.get_material_rates(
            0, list(range(len(grown))), react_index).copy()
        assert build_spy.call_count == 2

        # The newly-added nuclide gets correct rates, not stale zeros
        pu = openmc.lib.nuclides['Pu239']
        for j, s in enumerate(scores):
            expected = pu.collapse_rate(
                REACTION_MT[s], TEMPERATURE, CASMO40, flux)
            assert rates[len(nuclides), j] == pytest.approx(expected, rel=1e-10)
        assert rates[len(nuclides), 0] > 0.0  # Pu239 fission nonzero -> not stale


def test_flux_collapse_helper_direct_override():
    """A direct-tally (nuclide, reaction) pair overrides the flux-collapsed value."""
    nuclides = ['U235', 'U238']
    scores = ['fission', '(n,gamma)']
    flux = np.random.default_rng(1).random(N_GROUPS)

    with openmc.lib.TemporarySession():
        # Direct-tally U235 fission; everything else via flux collapse
        helper = _wire_helper(nuclides, scores, flux,
            reactions_direct=['fission'], nuclides_direct=['U235'])
        direct_value = 1.2345e-3
        helper._rate_tally = SimpleNamespace(nuclides=['U235'])
        helper._rate_tally_means_cache = np.array([[direct_value]])

        rates = helper.get_material_rates(
            0, list(range(len(nuclides))), list(range(len(scores)))).copy()

        # U235 fission comes from the direct tally, not the flux collapse
        assert rates[0, 0] == pytest.approx(direct_value)

        # U235 (n,gamma) still comes from the flux collapse
        nuc = openmc.lib.nuclides['U235']
        expected = nuc.collapse_rate(
            REACTION_MT['(n,gamma)'], TEMPERATURE, CASMO40, flux)
        assert rates[0, 1] == pytest.approx(expected, rel=1e-10)
