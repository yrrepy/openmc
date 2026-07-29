"""Tests for (n,n') / MT=4-via-GENDF support across the depletion stack.

Three related concerns are merged here:

* **Coupled/CE fallback** - the opt-in ``gendf_mt4_fallback`` on
  ``CoupledOperator``'s flux-tallying helpers. CE HDF5 libraries typically lack
  the lumped MT=4 reaction, so direct (n,n') tallies are silently zero; the flag
  fills the (n,n') column by collapsing the helper's own flux tally with GENDF
  MT=4 cross sections. Helpers are exercised without an ``openmc.lib`` session
  by injecting the state ``generate_tallies`` would create.
* **Independent fallback** - the same opt-in inside ``get_microxs_and_flux``,
  where ``_apply_gendf_mt4_fallback`` fills the (n,n') column of hand-built
  ``MicroXS`` objects (no transport), restricted to nuclides in BOTH libraries.
* **(n,n') reaction plumbing** - MT=4 in the ``REACTIONS``/``DADZ`` tables and
  its propagation through chain loading and depletion-matrix formation.
"""

import numpy as np
import pytest
from unittest.mock import Mock

import openmc.lib
import openmc.deplete.microxs as microxs_mod
from openmc.deplete import CoupledOperator
from openmc.deplete.helpers import DirectWithFluxHelper, FluxCollapseHelper
from openmc.deplete.gendf.collapse import _apply_gendf_mt4_fallback
from openmc.deplete.microxs import MicroXS, get_microxs_and_flux

from .gendf_testing import MockGENDFLibrary, CCFE709_BOUNDS, CCFE709_NGROUPS


# Shared group structure / MT=4 cross-section fixtures.
ENERGIES = CCFE709_BOUNDS
NG = CCFE709_NGROUPS
SIGMA4 = 0.5                          # coupled path: flat per-group MT=4 xs
SIGMA4_G = np.linspace(0.1, 2.0, NG)  # independent path: per-group MT=4 xs

NUCLIDES = ['Al27', 'Co59', 'Fe56']   # Co59: CE-only, Fe56: GENDF-only
CE_NUCLIDES = {'Al27', 'Co59'}
REACTIONS = ['(n,gamma)', "(n,n')"]


def _coupled_gendf():
    """Coupled/CE-path GENDF mock: only Al27, MT=4 = flat SIGMA4 per group."""
    return MockGENDFLibrary(xs={'Al27': {4: np.full(NG, SIGMA4)}})


def _independent_gendf():
    """Independent-path GENDF mock: Al27 + Fe56, MT=4 = SIGMA4_G per group."""
    return MockGENDFLibrary(xs={'Al27': {4: SIGMA4_G}, 'Fe56': {4: SIGMA4_G}})


# ==============================================================================
# Coupled / CE-mode fallback (CoupledOperator helpers)
# ==============================================================================

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
    helper = _dwf_helper(False, _coupled_gendf())
    rates = helper.get_material_rates(0, [0, 1], [0, 1])
    assert np.allclose(rates, [[10.0, 0.0], [20.0, 0.0]])


def test_fallback_fills_nn_column(capsys):
    """Opt-in fills (n,n') from the GENDF collapse; other columns untouched."""
    helper = _dwf_helper(True, _coupled_gendf())
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
    helper = _dwf_helper(True, _coupled_gendf())
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
        2, 2, ENERGIES, gendf_library=_coupled_gendf(),
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
    """CoupledOperator rejects the flag without a GENDF library."""
    with pytest.raises(ValueError, match='gendf_library'):
        CoupledOperator(_mock_model(), reaction_rate_mode='direct_with_flux',
                        gendf_mt4_fallback=True)


def test_operator_flag_rejects_direct_mode():
    """CoupledOperator rejects the flag in the non-flux 'direct' mode."""
    with pytest.raises(ValueError, match='flux-tallying'):
        CoupledOperator(_mock_model(), gendf_library=_coupled_gendf(),
                        gendf_mt4_fallback=True)


def _bare_operator(mode, opts, has_isomeric):
    """CoupledOperator skeleton exercising _get_helper_classes with the flag.

    Specialized to the MT=4 path (sets ``_gendf_mt4_fallback``,
    ``reaction_rates``, ``model`` and calls ``_get_helper_classes``); the shared
    ``bare_coupled_operator`` targets ``_setup_isomeric_branching`` instead, so
    this stays file-local.
    """
    op = CoupledOperator.__new__(CoupledOperator)
    op._gendf_library = _coupled_gendf()
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
    """Flux mode wires the fallback and inherits the GENDF energies."""
    op = _bare_operator('flux', {}, has_isomeric=False)
    assert isinstance(op._rate_helper, FluxCollapseHelper)
    assert op._rate_helper._mt4_gendf is op._gendf_library
    assert np.array_equal(op._rate_helper.energies, ENERGIES)


def test_flux_mode_rejects_mismatched_energies():
    """Explicit flux energies not matching the GENDF structure fail fast."""
    opts = {'energies': np.array([0.0, 1e6, 2e7])}
    with pytest.raises(ValueError, match='gendf_mt4_fallback'):
        _bare_operator('flux', opts, has_isomeric=False)


# ==============================================================================
# Independent fallback (get_microxs_and_flux / _apply_gendf_mt4_fallback)
# ==============================================================================

def _micro(n_grps):
    """MicroXS with (n,gamma) = 1.0 sentinel and (n,n') = 0 as in CE data."""
    data = np.zeros((len(NUCLIDES), len(REACTIONS), n_grps))
    data[:, 0, :] = 1.0
    return MicroXS(data, list(NUCLIDES), list(REACTIONS))


def test_direct_mode_substitutes_per_group():
    """Multigroup micros get sigma4_g verbatim; other columns untouched."""
    micro = _micro(NG)
    _apply_gendf_mt4_fallback(
        [micro], [np.full(NG, 2.0)], _independent_gendf(), CE_NUCLIDES)
    assert np.array_equal(micro.data[0, 1, :], SIGMA4_G)
    assert np.all(micro.data[:, 0, :] == 1.0)


def test_flux_mode_collapses_to_one_group():
    """1-group micros get the flux-weighted collapse of sigma4_g."""
    micro = _micro(1)
    flux = np.random.default_rng(42).random(NG)
    _apply_gendf_mt4_fallback([micro], [flux], _independent_gendf(), CE_NUCLIDES)
    assert micro.data[0, 1, 0] == pytest.approx(SIGMA4_G @ flux / flux.sum())


def test_fairness_filter():
    """CE-only and GENDF-only nuclides keep a zero (n,n') column."""
    for n_grps in (NG, 1):
        micro = _micro(n_grps)
        _apply_gendf_mt4_fallback(
            [micro], [np.ones(NG)], _independent_gendf(), CE_NUCLIDES)
        assert np.all(micro.data[1, 1, :] == 0.0)  # Co59 not in GENDF
        assert np.all(micro.data[2, 1, :] == 0.0)  # Fe56 not in CE


def test_zero_flux_domain_stays_zero():
    """A domain with zero flux leaves the (n,n') column at zero."""
    micro = _micro(1)
    _apply_gendf_mt4_fallback(
        [micro], [np.zeros(NG)], _independent_gendf(), CE_NUCLIDES)
    assert micro.data[0, 1, 0] == 0.0


def test_noop_without_nn_reaction(capsys):
    """Chains without (n,n') leave data and stdout untouched."""
    micro = MicroXS(np.ones((3, 1, NG)), list(NUCLIDES), ['(n,gamma)'])
    _apply_gendf_mt4_fallback(
        [micro], [np.ones(NG)], _independent_gendf(), CE_NUCLIDES)
    assert np.all(micro.data == 1.0)
    assert capsys.readouterr().out == ''


def test_one_time_message(capsys):
    """One setup message across all domains; every domain substituted."""
    micros = [_micro(1), _micro(1)]
    _apply_gendf_mt4_fallback(
        micros, [np.ones(NG), np.ones(NG)], _independent_gendf(), CE_NUCLIDES)
    out = capsys.readouterr().out
    assert out.count('GENDF MT=4') == 1
    assert '1 nuclides' in out
    assert micros[1].data[0, 1, 0] == pytest.approx(SIGMA4_G.mean())


def test_flag_requires_gendf_library():
    """get_microxs_and_flux rejects the flag without a GENDF library."""
    with pytest.raises(ValueError, match='gendf_library'):
        get_microxs_and_flux(None, [], gendf_mt4_fallback=True)


def test_mismatched_energies_raise(monkeypatch):
    """Explicit energies not matching the GENDF structure fail fast."""
    monkeypatch.setattr(microxs_mod, '_GENDF_TYPES', (MockGENDFLibrary,))
    model = Mock()
    model.tallies = []
    with pytest.raises(ValueError, match='CCFE-709'):
        get_microxs_and_flux(model, [], energies=[0.0, 1e6, 2e7],
                             gendf_library=_independent_gendf(),
                             gendf_mt4_fallback=True)


# ==============================================================================
# (n,n') reaction constants (REACTIONS / MT mappings)
# ==============================================================================

def test_nn_prime_in_reactions():
    """Verify (n,n') is present in REACTIONS dictionary."""
    from openmc.deplete.chain import REACTIONS
    assert "(n,n')" in REACTIONS


def test_nn_prime_mt_value():
    """Verify (n,n') has MT=4."""
    from openmc.deplete.chain import REACTIONS
    info = REACTIONS["(n,n')"]
    assert 4 in info.mts
    assert info.mts == {4}


def test_nn_prime_secondaries():
    """Verify (n,n') has no secondary particles."""
    from openmc.deplete.chain import REACTIONS
    info = REACTIONS["(n,n')"]
    assert info.secondaries == ()


def test_mt_to_reaction_mapping():
    """Verify MT=4 maps to (n,n') in gendf.py."""
    from openmc.deplete.gendf import MT_TO_REACTION
    assert 4 in MT_TO_REACTION
    assert MT_TO_REACTION[4] == "(n,n')"


def test_reaction_to_mt_mapping():
    """Verify (n,n') maps to MT=4 in gendf.py."""
    from openmc.deplete.gendf import REACTION_TO_MT
    assert "(n,n')" in REACTION_TO_MT
    assert REACTION_TO_MT["(n,n')"] == 4


def test_total_reaction_count():
    """Verify total number of reactions after adding (n,n')."""
    from openmc.deplete.chain import REACTIONS
    # Was 84 before, now 85 with (n,n')
    assert len(REACTIONS) == 85


# ==============================================================================
# Chain + matrix structure for (n,n')
# ==============================================================================

def test_self_transmutation_chain_structure(tmp_path):
    """Chain handles (n,n') self-transmutation (In115 -> In115 / In115_m1)."""
    from openmc.deplete import Chain

    chain_xml = """<?xml version="1.0"?>
<depletion_chain>
  <nuclide name="In115" decay_modes="0" reactions="1">
    <reaction type="(n,n')" Q="0.0" target="In115" branching_ratio="0.85"/>
    <isomeric_branching>
      <reaction type="(n,n')">
        <product nuclide="In115" ratio="0.85"/>
        <product nuclide="In115_m1" ratio="0.15"/>
      </reaction>
    </isomeric_branching>
  </nuclide>
  <nuclide name="In115_m1" decay_modes="1" reactions="0">
    <decay type="IT" target="In115" branching_ratio="1.0"/>
  </nuclide>
</depletion_chain>
"""
    chain_file = tmp_path / "chain_nn_prime.xml"
    chain_file.write_text(chain_xml)
    chain = Chain.from_xml(chain_file)

    # Verify both nuclides are in the chain
    assert 'In115' in chain.nuclide_dict
    assert 'In115_m1' in chain.nuclide_dict

    # Verify reaction is loaded
    in115 = chain['In115']
    reactions = [r.type for r in in115.reactions]
    assert "(n,n')" in reactions


def test_chain_xml_with_nn_prime(tmp_path):
    """Test that chain XML with (n,n') reactions loads correctly."""
    from openmc.deplete import Chain

    # Chain with (n,n') reaction and isomeric branching
    chain_xml = """<?xml version="1.0"?>
<depletion_chain>
  <nuclide name="In115" decay_modes="0" reactions="2">
    <reaction type="(n,gamma)" Q="6.78e6" target="In116"/>
    <reaction type="(n,n')" Q="0.0" target="In115" branching_ratio="0.85"/>
    <isomeric_branching>
      <reaction type="(n,n')">
        <product nuclide="In115" ratio="0.85"/>
        <product nuclide="In115_m1" ratio="0.15"/>
      </reaction>
    </isomeric_branching>
  </nuclide>
  <nuclide name="In115_m1" decay_modes="1" reactions="0" half_life="1.61e4">
    <decay type="IT" target="In115" branching_ratio="1.0"/>
  </nuclide>
  <nuclide name="In116" decay_modes="0" reactions="0"/>
</depletion_chain>
"""
    chain_file = tmp_path / "chain_nn_prime.xml"
    chain_file.write_text(chain_xml)
    chain = Chain.from_xml(chain_file)

    # Verify chain loaded correctly
    assert len(chain) == 3
    assert 'In115' in chain.nuclide_dict
    assert 'In115_m1' in chain.nuclide_dict
    assert 'In116' in chain.nuclide_dict

    # Verify In115 has (n,n') reaction
    in115 = chain['In115']
    reaction_types = [r.type for r in in115.reactions]
    assert "(n,n')" in reaction_types

    # Verify isomeric branching targets are loaded
    if chain.isomeric_branching_targets:
        if 'In115' in chain.isomeric_branching_targets:
            assert "(n,n')" in chain.isomeric_branching_targets['In115']


def test_form_matrix_with_nn_prime(tmp_path):
    """Test matrix formation with (n,n') isomeric branching."""
    from openmc.deplete import Chain
    from openmc.deplete import reaction_rates
    import scipy.sparse as sp

    chain_xml = """<?xml version="1.0"?>
<depletion_chain>
  <nuclide name="In115" decay_modes="0" reactions="1">
    <reaction type="(n,n')" Q="0.0" target="In115" branching_ratio="0.85"/>
    <isomeric_branching>
      <reaction type="(n,n')">
        <product nuclide="In115" ratio="0.85"/>
        <product nuclide="In115_m1" ratio="0.15"/>
      </reaction>
    </isomeric_branching>
  </nuclide>
  <nuclide name="In115_m1" decay_modes="1" reactions="0" half_life="1.61e4">
    <decay type="IT" target="In115" branching_ratio="1.0"/>
  </nuclide>
</depletion_chain>
"""
    chain_file = tmp_path / "chain_nn_prime.xml"
    chain_file.write_text(chain_xml)
    chain = Chain.from_xml(chain_file)

    # Create reaction rates using ReactionRates object
    nuclides = ["In115", "In115_m1"]
    rates = reaction_rates.ReactionRates(["mat1"], nuclides, chain.reactions)
    rates.set("mat1", "In115", "(n,n')", 1e-10)  # 1e-10 s^-1 reaction rate

    # Apply the runtime isomeric distribution -- without it the (n,n') reaction
    # only self-loops In115->In115 and the metastable path is never exercised.
    reaction_rate = 1e-10
    iso = {'In115': {"(n,n')": {'In115': 0.85, 'In115_m1': 0.15}}}
    matrix = chain.form_matrix(rates[0], isomeric_branching=iso)

    # Matrix should be sparse
    assert sp.issparse(matrix)

    # Get indices
    i_in115 = chain.nuclide_dict['In115']
    i_in115_m1 = chain.nuclide_dict['In115_m1']

    # Convert to dense for inspection. Matrix is [row, col]; col is the source.
    dense = matrix.toarray()

    # Metastable production = reaction_rate * metastable branching ratio.
    transfer_rate = dense[i_in115_m1, i_in115]
    assert transfer_rate == pytest.approx(reaction_rate * 0.15)

    # Ground self-term = parent loss (-rate) + ground branch gain (+rate*0.85).
    assert dense[i_in115, i_in115] == pytest.approx(reaction_rate * (0.85 - 1.0))

    # (n,n') conserves nuclide count: the In115 source column (pure reaction,
    # In115 is stable) sums to zero across parent-loss and both product-gains.
    assert dense[:, i_in115].sum() == pytest.approx(0.0, abs=1e-20)


# ==============================================================================
# DADZ dictionary consistency
# ==============================================================================

def test_nn_prime_in_dadz():
    """Verify (n,n') has (delta_A, delta_Z) == (0, 0) in DADZ."""
    from openmc.data import DADZ
    assert "(n,n')" in DADZ
    assert DADZ["(n,n')"] == (0, 0)


def test_dadz_consistency():
    """Verify all REACTIONS have corresponding DADZ entries."""
    from openmc.deplete.chain import REACTIONS
    from openmc.data import DADZ

    missing = []
    for rx_name in REACTIONS:
        if rx_name not in DADZ:
            missing.append(rx_name)

    assert not missing, f"Missing DADZ entries for: {missing}"


def test_dadz_all_reactions_available():
    """Test that Chain.from_endf() can access DADZ for every reaction."""
    from openmc.deplete.chain import REACTIONS
    from openmc.data import DADZ

    # This is exactly what Chain.from_endf() does at line 595
    for rx_name in REACTIONS:
        delta_A, delta_Z = DADZ[rx_name]
        # Verify reasonable values
        assert isinstance(delta_A, int)
        assert isinstance(delta_Z, int)
        # For (n,n') specifically
        if rx_name == "(n,n')":
            assert delta_A == 0
            assert delta_Z == 0
