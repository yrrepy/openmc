"""Tests for the opt-in (n,n') MT=4-via-GENDF fallback in get_microxs_and_flux.

CE HDF5 libraries typically lack the lumped MT=4 reaction, so the (n,n')
column of a MicroXS is silently zero. With gendf_mt4_fallback=True the column
is filled from GENDF MT=4 data, restricted to nuclides present in BOTH the
CE and GENDF libraries. _apply_gendf_mt4_fallback is exercised directly on
hand-built MicroXS objects (no transport).
"""

import numpy as np
import pytest
from unittest.mock import Mock

import openmc.deplete.microxs as microxs_mod
from openmc.deplete.microxs import (
    MicroXS, _apply_gendf_mt4_fallback, get_microxs_and_flux)
from openmc.mgxs import GROUP_STRUCTURES


ENERGIES = GROUP_STRUCTURES['CCFE-709']
NG = len(ENERGIES) - 1
SIGMA4_G = np.linspace(0.1, 2.0, NG)

NUCLIDES = ['Al27', 'Co59', 'Fe56']  # Co59: CE-only, Fe56: GENDF-only
CE_NUCLIDES = {'Al27', 'Co59'}
REACTIONS = ['(n,gamma)', "(n,n')"]


class MockGENDFLibrary:
    energy_structure = 'CCFE-709'
    energy_bounds = ENERGIES
    n_groups = NG

    def available_nuclides_set(self):
        return frozenset({'Al27', 'Fe56'})

    def get_all_xs(self, nuclide, mts=None, strict_alignment=True):
        return {4: SIGMA4_G.copy()}


def _micro(n_grps):
    """MicroXS with (n,gamma) = 1.0 sentinel and (n,n') = 0 as in CE data."""
    data = np.zeros((len(NUCLIDES), len(REACTIONS), n_grps))
    data[:, 0, :] = 1.0
    return MicroXS(data, list(NUCLIDES), list(REACTIONS))


def test_direct_mode_substitutes_per_group():
    """Multigroup micros get sigma4_g verbatim; other columns untouched."""
    micro = _micro(NG)
    _apply_gendf_mt4_fallback(
        [micro], [np.full(NG, 2.0)], MockGENDFLibrary(), CE_NUCLIDES)
    assert np.array_equal(micro.data[0, 1, :], SIGMA4_G)
    assert np.all(micro.data[:, 0, :] == 1.0)


def test_flux_mode_collapses_to_one_group():
    """1-group micros get the flux-weighted collapse of sigma4_g."""
    micro = _micro(1)
    flux = np.random.default_rng(42).random(NG)
    _apply_gendf_mt4_fallback([micro], [flux], MockGENDFLibrary(), CE_NUCLIDES)
    assert micro.data[0, 1, 0] == pytest.approx(SIGMA4_G @ flux / flux.sum())


def test_fairness_filter():
    """CE-only and GENDF-only nuclides keep a zero (n,n') column."""
    for n_grps in (NG, 1):
        micro = _micro(n_grps)
        _apply_gendf_mt4_fallback(
            [micro], [np.ones(NG)], MockGENDFLibrary(), CE_NUCLIDES)
        assert np.all(micro.data[1, 1, :] == 0.0)  # Co59 not in GENDF
        assert np.all(micro.data[2, 1, :] == 0.0)  # Fe56 not in CE


def test_zero_flux_domain_stays_zero():
    micro = _micro(1)
    _apply_gendf_mt4_fallback(
        [micro], [np.zeros(NG)], MockGENDFLibrary(), CE_NUCLIDES)
    assert micro.data[0, 1, 0] == 0.0


def test_noop_without_nn_reaction(capsys):
    """Chains without (n,n') leave data and stdout untouched."""
    micro = MicroXS(np.ones((3, 1, NG)), list(NUCLIDES), ['(n,gamma)'])
    _apply_gendf_mt4_fallback(
        [micro], [np.ones(NG)], MockGENDFLibrary(), CE_NUCLIDES)
    assert np.all(micro.data == 1.0)
    assert capsys.readouterr().out == ''


def test_one_time_message(capsys):
    """One setup message across all domains; every domain substituted."""
    micros = [_micro(1), _micro(1)]
    _apply_gendf_mt4_fallback(
        micros, [np.ones(NG), np.ones(NG)], MockGENDFLibrary(), CE_NUCLIDES)
    out = capsys.readouterr().out
    assert out.count('GENDF MT=4') == 1
    assert '1 nuclides' in out
    assert micros[1].data[0, 1, 0] == pytest.approx(SIGMA4_G.mean())


def test_flag_requires_gendf_library():
    with pytest.raises(ValueError, match='gendf_library'):
        get_microxs_and_flux(None, [], gendf_mt4_fallback=True)


def test_mismatched_energies_raise(monkeypatch):
    """Explicit energies not matching the GENDF structure fail fast."""
    monkeypatch.setattr(microxs_mod, '_GENDF_TYPES', (MockGENDFLibrary,))
    model = Mock()
    model.tallies = []
    with pytest.raises(ValueError, match='CCFE-709'):
        get_microxs_and_flux(model, [], energies=[0.0, 1e6, 2e7],
                             gendf_library=MockGENDFLibrary(),
                             gendf_mt4_fallback=True)
