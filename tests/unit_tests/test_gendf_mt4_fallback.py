"""Tests for the opt-in (n,n') MT=4-via-GENDF fallback in CE modes.

CE HDF5 libraries typically lack the lumped MT=4 reaction, so direct (n,n')
tallies are silently zero. With gendf_mt4_fallback=True, the (n,n') column
is filled by collapsing the helper's own flux tally with GENDF MT=4 cross
sections. Helpers are exercised without an openmc.lib session by injecting
the state generate_tallies would create.
"""

import numpy as np
import pytest
from unittest.mock import Mock

import openmc.lib
from openmc.deplete import CoupledOperator
from openmc.deplete.helpers import DirectWithFluxHelper, FluxCollapseHelper
from openmc.mgxs import GROUP_STRUCTURES


ENERGIES = GROUP_STRUCTURES['CCFE-709']
NG = len(ENERGIES) - 1
SIGMA4 = 0.5


class MockGENDFLibrary:
    energy_structure = 'CCFE-709'
    energy_bounds = ENERGIES
    n_groups = NG

    def available_nuclides_set(self):
        return frozenset({'Al27'})

    def get_all_xs(self, nuclide, mts=None, strict_alignment=True):
        return {4: np.full(NG, SIGMA4)}


def _dwf_helper(fallback, gendf):
    """DirectWithFluxHelper with post-generate_tallies state injected."""
    helper = DirectWithFluxHelper(
        2, 2, ENERGIES, gendf_library=gendf, gendf_mt4_fallback=fallback)
    helper._scores = ['(n,gamma)', "(n,n')"]
    helper._materials = [Mock()]
    helper._flux_tally = Mock(mean=np.full((NG, 1), 2.0))
    helper._nuclides = ['Al27', 'Co59']
    # direct tally: (n,gamma) nonzero, (n,n') silently zero as in CE data
    helper._direct_helper._nuclides = ['Al27', 'Co59']
    helper._direct_helper._rate_tally_means_cache = np.array(
        [[10.0, 0.0, 20.0, 0.0]])
    return helper


def test_fallback_off_by_default():
    """Without the flag the (n,n') column keeps the direct tally zeros."""
    helper = _dwf_helper(False, MockGENDFLibrary())
    rates = helper.get_material_rates(0, [0, 1], [0, 1])
    assert np.allclose(rates, [[10.0, 0.0], [20.0, 0.0]])


def test_fallback_fills_nn_column(capsys):
    """Opt-in fills (n,n') from the GENDF collapse; other columns untouched."""
    helper = _dwf_helper(True, MockGENDFLibrary())
    rates = helper.get_material_rates(0, [0, 1], [0, 1])

    # Al27: sigma4 . phi with phi = 2.0 per group; Co59 not in GENDF -> zero
    expected_nn = SIGMA4 * 2.0 * NG
    assert np.allclose(rates, [[10.0, expected_nn], [20.0, 0.0]])

    # One-time rank-0 setup message
    helper.get_material_rates(0, [0, 1], [0, 1])
    out = capsys.readouterr().out
    assert out.count('GENDF MT=4') == 1
    assert '1 nuclides' in out


def test_fallback_noop_without_nn_score():
    """Chains without (n,n') leave rates and message untouched."""
    helper = _dwf_helper(True, MockGENDFLibrary())
    helper._scores = ['(n,gamma)', '(n,2n)']
    rates = helper.get_material_rates(0, [0, 1], [0, 1])
    assert np.allclose(rates, [[10.0, 0.0], [20.0, 0.0]])
    assert helper._mt4_map is None


def test_flux_collapse_helper_fallback(monkeypatch):
    """FluxCollapseHelper overrides only the (n,n') CE collapse values."""
    mock_nuc = Mock()
    mock_nuc.collapse_rate = Mock(return_value=7.7)
    monkeypatch.setattr(openmc.lib, 'nuclides',
                        {'Al27': mock_nuc, 'Co59': mock_nuc})

    helper = FluxCollapseHelper(
        2, 2, ENERGIES, gendf_library=MockGENDFLibrary(),
        gendf_mt4_fallback=True)
    helper._materials = [Mock(temperature=294.0)]
    helper._scores = ['(n,gamma)', "(n,n')"]
    helper._mts = [102, 4]
    helper._nuclides = ['Al27', 'Co59']
    helper._flux_tally_means_cache = np.full((NG, 1), 2.0)

    rates = helper.get_material_rates(0, [0, 1], [0, 1])

    expected_nn = SIGMA4 * 2.0 * NG
    assert rates[0, 0] == pytest.approx(7.7)          # CE collapse kept
    assert rates[0, 1] == pytest.approx(expected_nn)  # GENDF override
    assert rates[1, 0] == pytest.approx(7.7)
    assert rates[1, 1] == pytest.approx(7.7)          # Co59 not in GENDF


def _mock_model():
    model = Mock(spec=['materials', 'geometry', 'settings', 'plots'])
    model.materials.cross_sections = 'dummy_cross_sections.xml'
    return model


def test_operator_flag_requires_gendf_library():
    with pytest.raises(ValueError, match='gendf_library'):
        CoupledOperator(_mock_model(), reaction_rate_mode='direct_with_flux',
                        gendf_mt4_fallback=True)


def test_operator_flag_rejects_direct_mode():
    with pytest.raises(ValueError, match='flux-tallying'):
        CoupledOperator(_mock_model(), gendf_library=MockGENDFLibrary(),
                        gendf_mt4_fallback=True)


def _bare_operator(mode, opts, has_isomeric):
    op = CoupledOperator.__new__(CoupledOperator)
    op._gendf_library = MockGENDFLibrary()
    op._gendf_mt4_fallback = True
    op.chain = Mock()
    op.chain.isomeric_branching_targets = (
        {'Al27': {'(n,gamma)': ['Al28']}} if has_isomeric else {})
    op.chain.nuclides = []
    op.reaction_rates = Mock(n_nuc=2, n_react=2)
    op.model = Mock()
    op._get_helper_classes({
        'reaction_rate_mode': mode,
        'normalization_mode': 'source-rate',
        'fission_yield_mode': 'constant',
        'reaction_rate_opts': opts,
        'fission_yield_opts': {},
    })
    return op


def test_direct_with_flux_helper_created_without_isomeric_data():
    """The flag alone is enough to get the flux-tallying helper."""
    op = _bare_operator('direct_with_flux', {}, has_isomeric=False)
    assert isinstance(op._rate_helper, DirectWithFluxHelper)
    assert op._rate_helper._mt4_gendf is op._gendf_library


def test_flux_mode_wires_fallback_with_default_energies():
    op = _bare_operator('flux', {}, has_isomeric=False)
    assert isinstance(op._rate_helper, FluxCollapseHelper)
    assert op._rate_helper._mt4_gendf is op._gendf_library
    assert np.array_equal(op._rate_helper.energies, ENERGIES)


def test_flux_mode_rejects_mismatched_energies():
    opts = {'energies': np.array([0.0, 1e6, 2e7])}
    with pytest.raises(ValueError, match='gendf_mt4_fallback'):
        _bare_operator('flux', opts, has_isomeric=False)
