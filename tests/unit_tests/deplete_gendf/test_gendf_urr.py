"""URR self-shielding by material dilution (CALENDF) for the GENDF workflow.

Organized by section:

* Physics of the fold - the total-XS self-shielding factor
  (:func:`mat_ssf_total_factors_gendf`) and the Jacobi mutual-shielding sigma_0
  fixed point (:func:`iterate_material_dilution_sigma0_gendf`) on synthetic
  band tables (no data dependency): the permanent G-A (analytic convergence)
  and G-B (mono / inert-diluter no-op) gates. GENDF sibling of
  ``test_deplete_mat_ssf.py`` (PENDF side).
* Entry-point validation - the ``urr_material_dilution`` toggle of
  ``MicroXS.from_multigroup_flux_with_gendf`` (Material, mapping or off) and of
  ``get_gendfxs_and_flux`` (bool only; Material or single-Material-filled Cell
  domains). Bad input raises before the transport solve.
* Scaler contract - ``_CalendfRowScaler`` applied row by row in the streaming
  collapse: one scaler per distinct material, each domain shielded by its own
  composition, library rows never modified, the same result as the table
  applicator ``_apply_mat_ssf_gendf``, and each ``.tpe`` file read once per
  call.
"""
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

import openmc
import openmc.deplete.gendf.calendf as calendf_mod
import openmc.deplete.gendf.collapse as collapse_mod
import openmc.deplete.microxs as microxs_mod
from openmc.data import REACTION_MT
from openmc.deplete import MicroXS
from openmc.deplete.gendf.calendf import (
    SIGMA0_ITER_MAX, SIGMA0_ITER_TOL,
    TpeTable, _BandGroup,
    material_dilution_sigma0, mat_ssf_factors_gendf,
    mat_ssf_total_factors_gendf, iterate_material_dilution_sigma0_gendf,
    _CalendfRowScaler, _apply_mat_ssf_gendf)
from openmc.deplete.gendf.collapse import (_gendf_dilution_material,
                                           get_gendfxs_and_flux)
from openmc.deplete.microxs import (_build_sparse_xs_table,
                                    _collapse_gendf_streaming,
                                    _normalize_flux_batch)

from .gendf_testing import CCFE709_NGROUPS, MockGENDFLibrary


# ===========================================================================
# Physics of the fold
# ===========================================================================

# Synthetic three-group library with a single band-carrying (URR) group in the
# middle; groups 0 and 2 carry no bands, so R_tot == 1 there.
_N_GROUPS = 3
_MID = 1


def _band_tpe(total_lo, total_hi, index=_MID,
              capture_lo=0.0, capture_hi=0.0):
    """A synthetic two-equiprobable-band CALENDF table, band totals [lo, hi]."""
    prob = np.array([0.5, 0.5])
    total = np.array([float(total_lo), float(total_hi)])
    capture = np.array([float(capture_lo), float(capture_hi)])
    zeros = np.zeros(2)
    g = _BandGroup(ig=0, index=index, elo=1.0, ehi=2.0, prob=prob,
                   total=total, elastic=zeros.copy(),
                   capture=capture, fission=zeros.copy())
    return TpeTable(za=0, mat=0, teff=294.0, ip=1,
                    group_structure='CCFE-709', span=(1.0, 2.0),
                    groups={index: g})


def _dense_fixed_point(densities, group_totals, coupling, n_groups, passes=500):
    """Independent high-iteration reference solution of the sigma_0 fixed point.

    Runs the same Jacobi map far past the production ``SIGMA0_ITER_MAX`` cap, so
    the result is the true fixed point to solve against (contraction guarantees
    convergence).
    """
    d = {p: material_dilution_sigma0(densities, p, group_totals,
                                     n_groups=n_groups) for p in coupling}
    for _ in range(passes):
        R = {q: mat_ssf_total_factors_gendf(coupling[q], d[q], n_groups)
             for q in coupling}
        d_new = {}
        for p in coupling:
            acc = np.zeros(n_groups)
            for j, n_j in densities.items():
                if j == p or n_j <= 0 or j not in group_totals:
                    continue
                r_j = R[j] if j in R else 1.0
                acc = acc + n_j * np.asarray(group_totals[j]) * r_j
            d_new[p] = acc / densities[p]
        d = d_new
    return d


def test_mat_ssf_total_factor_gendf_limits():
    """R_tot matches the analytic band harmonic/arithmetic ratio and -> 1 dilute."""
    tpe = _band_tpe(10.0, 100.0)   # p=[.5,.5], sigma_t,inf = 55 b
    # sigma_0 = 0: sigma_t,eff = 1 / (0.5/10 + 0.5/100) = 18.1818 b -> R = 0.33058
    r0 = mat_ssf_total_factors_gendf(tpe, np.zeros(_N_GROUPS), _N_GROUPS)
    assert r0[_MID] == pytest.approx((1.0 / 0.055) / 55.0, rel=1e-9)
    # infinite dilution: R -> 1 exactly; groups without bands: R == 1 (ceiling)
    r_inf = mat_ssf_total_factors_gendf(
        tpe, np.full(_N_GROUPS, 1.0e12), _N_GROUPS)
    assert r_inf[_MID] == pytest.approx(1.0, rel=1e-6)
    assert r0[0] == 1.0 and r0[2] == 1.0
    # monotone increasing in sigma_0 (more dilution -> less shielding)
    rs = [mat_ssf_total_factors_gendf(tpe, np.full(_N_GROUPS, s), _N_GROUPS)[_MID]
          for s in (0.0, 5.0, 55.0, 1.0e3)]
    assert np.all(np.diff(rs) > 0)


def test_mat_ssf_total_factor_gendf_ceiling_and_fallback():
    """Uncovered groups keep R=1 (ceiling); an all-zero total falls back to partials."""
    # Bands only in the mid group -> R == 1 in the uncovered groups 0, 2
    # (the TENDL-2017 W-ceiling behaviour: no band record -> no shielding).
    tpe = _band_tpe(20.0, 80.0)
    R = mat_ssf_total_factors_gendf(tpe, np.zeros(_N_GROUPS), _N_GROUPS)
    assert R[0] == 1.0 and R[2] == 1.0 and R[_MID] < 1.0

    # Defensive no-total fallback: a group whose total column is all zero is
    # reconstructed from elastic + capture + fission (never fires on real .tpe).
    prob = np.array([0.5, 0.5])
    g = _BandGroup(ig=0, index=_MID, elo=1.0, ehi=2.0, prob=prob,
                   total=np.zeros(2), elastic=np.array([10.0, 100.0]),
                   capture=np.zeros(2), fission=np.zeros(2))
    tpe_zero_total = TpeTable(za=0, mat=0, teff=294.0, ip=1,
                              group_structure='CCFE-709', span=(1.0, 2.0),
                              groups={_MID: g})
    R2 = mat_ssf_total_factors_gendf(tpe_zero_total, np.zeros(_N_GROUPS), _N_GROUPS)
    # Reconstructed total [10, 100] gives the same ratio as the limit test.
    assert R2[_MID] == pytest.approx((1.0 / 0.055) / 55.0, rel=1e-9)


def test_sigma0_iteration_gendf_converges_to_fixed_point():
    """G-A: two synthetic resonant nuclides -> iteration reaches the analytic d*."""
    coupling = {'A': _band_tpe(20.0, 80.0), 'B': _band_tpe(30.0, 70.0)}
    densities = {'A': 0.6, 'B': 0.4}
    group_totals = {'A': np.full(_N_GROUPS, 50.0),
                    'B': np.full(_N_GROUPS, 50.0)}

    sigma0, info = iterate_material_dilution_sigma0_gendf(
        densities, group_totals, coupling, _N_GROUPS)

    # Converged within the production bound.
    assert info['converged']
    assert info['n_iter'] <= SIGMA0_ITER_MAX

    ref = _dense_fixed_point(densities, group_totals, coupling, _N_GROUPS)
    R = {q: mat_ssf_total_factors_gendf(coupling[q], sigma0[q], _N_GROUPS)
         for q in coupling}
    for p in ('A', 'B'):
        # Matches the independent high-iteration reference fixed point.
        assert sigma0[p][_MID] == pytest.approx(ref[p][_MID], rel=SIGMA0_ITER_TOL)
        # Satisfies the fixed-point equation d(p) = (1/f_p) sum_{q!=p} f_q s_q R_q.
        rhs = sum(densities[j] * group_totals[j][_MID] * R[j][_MID]
                  for j in coupling if j != p) / densities[p]
        assert abs(sigma0[p][_MID] - rhs) / sigma0[p][_MID] < SIGMA0_ITER_TOL
        # Mutual shielding strictly lowers sigma_0 from the first approximation.
        assert sigma0[p][_MID] < info['trajectory'][0][p][_MID]

    # Contraction: the per-pass update shrinks monotonically (no oscillation).
    assert np.all(np.diff(info['max_rel']) < 0)


def test_sigma0_iteration_gendf_warns_on_nonconvergence(monkeypatch):
    """Loop exhausted below tolerance -> UserWarning + converged False."""
    coupling = {'A': _band_tpe(20.0, 80.0), 'B': _band_tpe(30.0, 70.0)}
    densities = {'A': 0.6, 'B': 0.4}
    group_totals = {'A': np.full(_N_GROUPS, 50.0),
                    'B': np.full(_N_GROUPS, 50.0)}

    monkeypatch.setattr('openmc.deplete.gendf.calendf.SIGMA0_ITER_MAX', 1)
    with pytest.warns(UserWarning, match="did not converge"):
        _, info = iterate_material_dilution_sigma0_gendf(
            densities, group_totals, coupling, _N_GROUPS)
    assert not info['converged']


def test_sigma0_iteration_gendf_jacobi_order_independent():
    """G-A companion: the simultaneous update is independent of nuclide order."""
    tA, tB = _band_tpe(20.0, 80.0), _band_tpe(30.0, 70.0)
    gt = {'A': np.full(_N_GROUPS, 50.0), 'B': np.full(_N_GROUPS, 50.0)}
    d1, _ = iterate_material_dilution_sigma0_gendf(
        {'A': 0.6, 'B': 0.4}, gt, {'A': tA, 'B': tB}, _N_GROUPS)
    d2, _ = iterate_material_dilution_sigma0_gendf(
        {'B': 0.4, 'A': 0.6}, gt, {'B': tB, 'A': tA}, _N_GROUPS)
    assert np.array_equal(d1['A'], d2['A'])
    assert np.array_equal(d1['B'], d2['B'])


def test_sigma0_iteration_gendf_mono_noop():
    """G-B: a single resonant nuclide (no diluters) -> bit-identical no-op."""
    densities = {'X': 1.0}
    group_totals = {'X': np.full(_N_GROUPS, 50.0)}
    coupling = {'X': _band_tpe(20.0, 80.0)}

    d0 = material_dilution_sigma0(densities, 'X', group_totals, n_groups=_N_GROUPS)
    sigma0, info = iterate_material_dilution_sigma0_gendf(
        densities, group_totals, coupling, _N_GROUPS)
    assert np.array_equal(d0, np.zeros(_N_GROUPS))   # pure absorber, sigma_0 = 0
    assert np.array_equal(sigma0['X'], d0)           # iteration leaves it untouched
    assert info['converged']

    # capture self-shielding factor is bit-identical to the non-iterated fold
    tpe = _band_tpe(20.0, 80.0, capture_lo=1.0, capture_hi=4.0)
    f_d0 = mat_ssf_factors_gendf(tpe, d0, _N_GROUPS, 'capture')
    f_it = mat_ssf_factors_gendf(tpe, sigma0['X'], _N_GROUPS, 'capture')
    assert np.array_equal(f_d0, f_it)


def test_sigma0_iteration_gendf_inert_diluter_noop():
    """G-B: one resonant nuclide in a non-ptable diluter -> iteration no-op."""
    densities = {'X': 1.0, 'M': 4.0}                 # M inert (no .tpe, R=1)
    group_totals = {'X': np.full(_N_GROUPS, 50.0),
                    'M': np.full(_N_GROUPS, 6.0)}
    coupling = {'X': _band_tpe(20.0, 80.0)}          # only X carries a table

    d0 = material_dilution_sigma0(densities, 'X', group_totals, n_groups=_N_GROUPS)
    sigma0, _ = iterate_material_dilution_sigma0_gendf(
        densities, group_totals, coupling, _N_GROUPS)
    # X sees only the inert diluter: sigma_0 = n_M * s_M / n_X = 24 b, unchanged.
    assert sigma0['X'][_MID] == pytest.approx(24.0)
    assert np.array_equal(sigma0['X'], d0)


# ===========================================================================
# Entry-point validation and scaler contract
# ===========================================================================
#
# Transport is stubbed: ``_get_chain`` and ``StatePoint`` are patched in
# ``openmc.deplete.gendf.collapse`` so the flux read path returns canned
# per-domain flux, and ``model.run`` is a Mock that records whether the solve
# was reached. Every patch goes through ``monkeypatch`` (undone per test).

_CALENDF = 'cal/tp-709-294'


def _mock_chain(nuclide_names):
    """Chain stand-in: named nuclides, capture and fission reactions."""
    chain = Mock()
    chain.nuclides = [Mock() for _ in nuclide_names]
    for nuc, name in zip(chain.nuclides, nuclide_names):
        nuc.name = name
    chain.reactions = ['(n,gamma)', 'fission']
    return chain


def _register_mock_library(monkeypatch):
    """Accept MockGENDFLibrary as a GENDF backend (collapse.py reads the
    type tuple through ``openmc.deplete.microxs``)."""
    if MockGENDFLibrary not in microxs_mod._GENDF_TYPES:
        monkeypatch.setattr(microxs_mod, '_GENDF_TYPES',
                            (MockGENDFLibrary,) + microxs_mod._GENDF_TYPES)


def _material(nuclides, fracs, density):
    m = openmc.Material()
    for n, f in zip(nuclides, fracs):
        m.add_nuclide(n, f)
    m.set_density('atom/b-cm', density)
    return m


def _install_stub_scaler(monkeypatch):
    """Replace ``_CalendfRowScaler`` in collapse.py by a stub, no .tpe needed.

    The stub multiplies capture rows by a composition-dependent constant
    ``1 / (1 + sum of densities)`` and leaves every other row. Returns the list
    of constructor records (``densities``, ``calendf_path``), one per scaler
    built.
    """
    built = []

    class _StubScaler:
        def __init__(self, gendf_library, calendf_path, densities,
                     mat_ssf_nuclides=None, tpe_cache=None):
            built.append({'densities': dict(densities),
                          'calendf_path': calendf_path})
            self.factor = 1.0 / (1.0 + sum(float(v)
                                           for v in densities.values()))

        def scale(self, nuc, rxn, row):
            if rxn == '(n,gamma)':
                row *= self.factor

    monkeypatch.setattr(collapse_mod, '_CalendfRowScaler', _StubScaler)
    return built


def _run_gendf_wrapper(monkeypatch, domains, canned_flux, *,
                       urr_material_dilution=False, calendf_path=None,
                       mat_ssf_nuclides=None, gendf_library=None,
                       gendf_nuclides=('U235', 'U238'),
                       chain_nuclides=('U235', 'U238')):
    """Drive get_gendfxs_and_flux with transport stubbed by canned per-domain
    flux (shape ``(n_domains, n_groups)``). Returns ``((fluxes, micros),
    model)`` so callers can assert on ``model.run`` call state."""
    if gendf_library is None:
        gendf_library = MockGENDFLibrary(set(gendf_nuclides))
    reshaped = np.asarray(canned_flux, dtype=float)[:, :, np.newaxis, np.newaxis]
    tally = Mock()
    tally.get_reshaped_data.return_value = reshaped
    statepoint = MagicMock()
    statepoint.__enter__.return_value = statepoint
    statepoint.tallies.__getitem__.return_value = tally

    model = Mock()
    model.tallies = []
    model.run.return_value = 'statepoint.dummy.h5'

    with monkeypatch.context() as m:
        m.setattr(collapse_mod, '_get_chain',
                  Mock(return_value=_mock_chain(list(chain_nuclides))))
        m.setattr(collapse_mod, 'StatePoint', Mock(return_value=statepoint))
        m.setattr(openmc.lib, 'is_initialized', False)
        _register_mock_library(m)
        result = get_gendfxs_and_flux(
            model, domains, gendf_library,
            urr_material_dilution=urr_material_dilution,
            calendf_path=calendf_path,
            mat_ssf_nuclides=mat_ssf_nuclides,
        )
    return result, model


def _direct_gendf(monkeypatch, flux, *, urr_material_dilution=False,
                  calendf_path=None, gendf_library=None,
                  gendf_nuclides=('U235', 'U238'),
                  chain_nuclides=('U235', 'U238')):
    """Direct MicroXS.from_multigroup_flux_with_gendf over the same mocks, for
    G-EQ / flag-off comparison against the wrapper."""
    if gendf_library is None:
        gendf_library = MockGENDFLibrary(set(gendf_nuclides))
    with monkeypatch.context() as m:
        m.setattr(collapse_mod, '_get_chain',
                  Mock(return_value=_mock_chain(list(chain_nuclides))))
        _register_mock_library(m)
        return MicroXS.from_multigroup_flux_with_gendf(
            multigroup_flux=np.asarray(flux, dtype=float),
            gendf_library=gendf_library,
            chain_file='dummy_chain.xml',
            urr_material_dilution=urr_material_dilution,
            calendf_path=calendf_path,
        )


# Seeded four-nuclide library with fake CALENDF tables for the W isotopes.
_W_NUCLIDES = ['W182', 'W184', 'Fe56', 'H1']
_URR_GROUP = 400          # band-carrying library group of the fake .tpe tables


@pytest.fixture
def urr_library(monkeypatch):
    """Mock library (MT 1/102/18/16 rows) plus fake ``.tpe`` tables.

    ``find_tpe`` / ``read_tpe`` in calendf.py are patched: W182 and W184 carry
    a ``_band_tpe()`` table, Fe56 and H1 carry none. Returns ``(library,
    reads)``; ``reads`` counts the ``read_tpe`` calls per file name.
    """
    n = CCFE709_NGROUPS
    rng = np.random.default_rng(20260924)
    shape = np.linspace(0.5, 2.0, n)
    xs = {nuc: {mt: shape * rng.uniform(0.1, 10.0, n)
                for mt in (1, 102, 18, 16)}
          for nuc in _W_NUCLIDES}
    tpe = {'W182': _band_tpe(5.0, 400.0, index=_URR_GROUP, capture_lo=0.5,
                             capture_hi=80.0),
           'W184': _band_tpe(8.0, 250.0, index=_URR_GROUP, capture_lo=0.3,
                             capture_hi=40.0)}
    reads = Counter()

    def fake_find_tpe(calendf_path, nuclide):
        return Path(f'{nuclide}-294.tpe') if nuclide in tpe else None

    def fake_read_tpe(path, group_structure='CCFE-709'):
        reads[Path(path).name] += 1
        return tpe[Path(path).name.split('-')[0]]

    monkeypatch.setattr(calendf_mod, 'find_tpe', fake_find_tpe)
    monkeypatch.setattr(calendf_mod, 'read_tpe', fake_read_tpe)
    return MockGENDFLibrary(xs=xs), reads


# --- MicroXS.from_multigroup_flux_with_gendf toggle ---

def test_urr_material_dilution_true_raises():
    """Bare True is under-specified: no composition for the sigma0_mat background."""
    with pytest.raises(ValueError, match='under-specified'):
        MicroXS.from_multigroup_flux_with_gendf(
            multigroup_flux=np.ones(709),
            gendf_library=MockGENDFLibrary({'U235'}),
            chain_file='dummy_chain.xml',
            urr_material_dilution=True, calendf_path='calendf')


def test_urr_material_dilution_empty_mapping_raises():
    """An empty composition mapping must be loud, not silently degrade to f=1."""
    with pytest.raises(ValueError, match='empty'):
        MicroXS.from_multigroup_flux_with_gendf(
            multigroup_flux=np.ones(709),
            gendf_library=MockGENDFLibrary({'U235'}),
            chain_file='dummy_chain.xml',
            urr_material_dilution={}, calendf_path='calendf')


def test_urr_material_dilution_material_matches_dict(monkeypatch):
    """An openmc.Material and its get_nuclide_atom_densities() dict feed the same
    densities to the CALENDF scaler and yield the same collapsed MicroXS."""
    mat = openmc.Material()
    mat.add_nuclide('U235', 1.0)
    mat.add_nuclide('U238', 2.0)
    mat.set_density('atom/b-cm', 3.0)
    dens = mat.get_nuclide_atom_densities()

    built = _install_stub_scaler(monkeypatch)

    micro_mat = _direct_gendf(monkeypatch, np.ones(709),
                              urr_material_dilution=mat, calendf_path='calendf')
    dens_from_mat = built.pop()['densities']

    micro_dict = _direct_gendf(monkeypatch, np.ones(709),
                               urr_material_dilution=dens, calendf_path='calendf')
    dens_from_dict = built.pop()['densities']

    assert dens_from_mat == dens_from_dict == dict(dens)
    assert np.array_equal(micro_mat.data, micro_dict.data)
    assert micro_mat.nuclides == micro_dict.nuclides


# --- get_gendfxs_and_flux: scaler contract ---

def test_wrapper_ssf_called_once_per_domain(monkeypatch):
    """One scaler per distinct Material.id, built with THAT material's
    densities and the supplied calendf_path; results come back in domain
    order."""
    mat_a = _material(['U235'], [1.0], 1.0)
    mat_b = _material(['U235', 'U238'], [1.0, 2.0], 3.0)
    built = _install_stub_scaler(monkeypatch)

    (_fluxes, micros), model = _run_gendf_wrapper(
        monkeypatch, [mat_a, mat_b, mat_a], np.ones((3, 709)),
        urr_material_dilution=True, calendf_path=_CALENDF)

    assert len(built) == 2
    assert built[0]['densities'] == dict(mat_a.get_nuclide_atom_densities())
    assert built[1]['densities'] == dict(mat_b.get_nuclide_atom_densities())
    assert built[0]['calendf_path'] == built[1]['calendf_path'] == _CALENDF
    # Domain order A, B, A: the stub's capture factor differs per material.
    assert len(micros) == 3
    assert np.array_equal(micros[0].data, micros[2].data)
    assert not np.array_equal(micros[0].data, micros[1].data)
    model.run.assert_called_once()


def test_wrapper_ssf_not_called_when_off(monkeypatch):
    """Flag off: no CALENDF scaler is built."""
    mat = _material(['U235'], [1.0], 1.0)
    built = _install_stub_scaler(monkeypatch)

    _run_gendf_wrapper(monkeypatch, [mat], np.ones((1, 709)),
                       urr_material_dilution=False)
    assert built == []


def test_wrapper_off_matches_direct_collapse(monkeypatch):
    """Flag-off regression: wrapper result identical to the direct collapse
    (from_multigroup_flux_with_gendf with the flag off)."""
    mat = _material(['U235', 'U238'], [1.0, 2.0], 3.0)
    flux = np.linspace(1.0, 2.0, 709)
    (_fluxes, micros), _model = _run_gendf_wrapper(
        monkeypatch, [mat], flux[np.newaxis, :], urr_material_dilution=False)
    direct = _direct_gendf(monkeypatch, flux, urr_material_dilution=False)
    assert np.array_equal(micros[0].data, direct.data)
    assert micros[0].nuclides == direct.nuclides


def test_wrapper_geq_matches_direct_with_dilution(monkeypatch):
    """G-EQ: transport-coupled == flux-supplied. Each domain's diluted MicroXS
    equals the direct from_multigroup_flux_with_gendf call with that domain's
    composition and the same flux."""
    mat1 = _material(['U235'], [1.0], 1.0)
    mat2 = _material(['U235', 'U238'], [1.0, 2.0], 3.0)
    canned = np.vstack([np.ones(709), np.linspace(1.0, 2.0, 709)])
    built = _install_stub_scaler(monkeypatch)

    (_fluxes, micros), _model = _run_gendf_wrapper(
        monkeypatch, [mat1, mat2], canned, urr_material_dilution=True,
        calendf_path='cal')

    d1 = _direct_gendf(monkeypatch, canned[0], urr_material_dilution=mat1,
                       calendf_path='cal')
    d2 = _direct_gendf(monkeypatch, canned[1], urr_material_dilution=mat2,
                       calendf_path='cal')

    assert len(built) == 4          # two in the wrapper, one per direct call
    np.testing.assert_allclose(micros[0].data, d1.data, rtol=1e-13, atol=0)
    np.testing.assert_allclose(micros[1].data, d2.data, rtol=1e-13, atol=0)


def test_wrapper_domain_isolation_deepcopy_guard(monkeypatch, urr_library):
    """Two different-composition domains do not contaminate each other, and the
    library's rows are unchanged after the shielded collapse."""
    lib, _reads = urr_library
    n = lib.n_groups
    mts = (1, 102, 18, 16)
    before = {nuc: {mt: lib.get_xs(nuc, mt) for mt in mts}
              for nuc in _W_NUCLIDES}
    mat_w = _material(_W_NUCLIDES, [0.3, 0.2, 0.5, 0.1], 0.08)
    mat_no_w = _material(['Fe56', 'H1'], [0.5, 0.1], 0.06)
    canned = np.vstack([np.ones(n), np.linspace(1.0, 2.0, n)])

    (_fluxes, micros), _model = _run_gendf_wrapper(
        monkeypatch, [mat_w, mat_no_w], canned, urr_material_dilution=True,
        calendf_path=_CALENDF, gendf_library=lib, chain_nuclides=_W_NUCLIDES)

    unshielded_w = _direct_gendf(monkeypatch, canned[0], gendf_library=lib,
                                 chain_nuclides=_W_NUCLIDES)
    unshielded_no_w = _direct_gendf(monkeypatch, canned[1], gendf_library=lib,
                                    chain_nuclides=_W_NUCLIDES)

    iw = micros[0].nuclides.index('W182')
    ig = micros[0].reactions.index('(n,gamma)')
    # The W182-free domain sees no shielding, whatever the other domain did.
    assert np.array_equal(micros[1].data, unshielded_no_w.data)
    # The W domain is shielded.
    assert micros[0].data[iw, ig, 0] != unshielded_w.data[iw, ig, 0]
    # The library's own rows are never modified.
    for nuc in _W_NUCLIDES:
        for mt in mts:
            assert np.array_equal(lib.get_xs(nuc, mt), before[nuc][mt])


def test_wrapper_dilution_cell_with_material_fill_resolves_composition(monkeypatch):
    """A Cell filled with a single Material passes the dilution domain check and
    shields with that fill's composition (the tally domain stays the Cell)."""
    mat = _material(['U235', 'U238'], [1.0, 2.0], 3.0)
    cell = openmc.Cell(fill=mat)

    # The composition resolver returns the cell's fill Material; a bare Material
    # resolves to itself (mixed Material/Cell sequences are allowed).
    assert _gendf_dilution_material(cell) is mat
    assert _gendf_dilution_material(mat) is mat

    # End to end: the material-filled Cell drives the collapse without raising,
    # and the scaler sees the FILL's composition (per-cell flux, but the fill
    # supplies the sigma0_mat background).
    built = _install_stub_scaler(monkeypatch)
    (_fluxes, micros), model = _run_gendf_wrapper(
        monkeypatch, [cell], np.ones((1, 709)), urr_material_dilution=True,
        calendf_path='cal')

    assert [b['densities'] for b in built] == [dict(mat.get_nuclide_atom_densities())]
    assert len(micros) == 1
    model.run.assert_called_once()

    (_fluxes, micros_mat), _model = _run_gendf_wrapper(
        monkeypatch, [mat], np.ones((1, 709)), urr_material_dilution=True,
        calendf_path='cal')
    assert np.array_equal(micros[0].data, micros_mat[0].data)


# --- get_gendfxs_and_flux: validation before the transport solve ---

def test_wrapper_dilution_requires_calendf_path(monkeypatch):
    """True without calendf_path raises before the transport solve."""
    mat = _material(['U235'], [1.0], 1.0)
    model = Mock()
    model.run.return_value = 'sp'
    _register_mock_library(monkeypatch)
    with pytest.raises(ValueError, match='calendf_path'):
        get_gendfxs_and_flux(model, [mat], MockGENDFLibrary({'U235'}),
                             urr_material_dilution=True)
    model.run.assert_not_called()


def test_wrapper_dilution_requires_material_domains(monkeypatch):
    """True with a non-Material domain raises before the transport solve."""
    cell = openmc.Cell()
    model = Mock()
    model.run.return_value = 'sp'
    _register_mock_library(monkeypatch)
    with pytest.raises(ValueError, match='openmc.Material'):
        get_gendfxs_and_flux(model, [cell], MockGENDFLibrary({'U235'}),
                             urr_material_dilution=True, calendf_path='cal')
    model.run.assert_not_called()


def test_wrapper_dilution_cell_with_nonmaterial_fill_raises(monkeypatch):
    """A Cell whose fill is not a single Material (void or a Universe) has no
    single composition -> raise before the transport solve, with a clear message
    naming the offending cell's fill."""
    void_cell = openmc.Cell()                       # no fill -> void
    universe_cell = openmc.Cell(fill=openmc.Universe())
    _register_mock_library(monkeypatch)
    for bad in (void_cell, universe_cell):
        model = Mock()
        model.run.return_value = 'sp'
        with pytest.raises(ValueError,
                           match='filled with a single openmc.Material'):
            get_gendfxs_and_flux(
                model, [bad], MockGENDFLibrary({'U235'}),
                urr_material_dilution=True, calendf_path='cal')
        model.run.assert_not_called()


def test_wrapper_non_bool_dilution_raises(monkeypatch):
    """A non-bool urr_material_dilution raises with the redirect message,
    before the transport solve."""
    mat = _material(['U235'], [1.0], 1.0)
    model = Mock()
    model.run.return_value = 'sp'
    _register_mock_library(monkeypatch)
    with pytest.raises(ValueError,
                       match='must be a bool') as exc:
        get_gendfxs_and_flux(model, [mat], MockGENDFLibrary({'U235'}),
                             urr_material_dilution=mat, calendf_path='cal')
    assert 'from_multigroup_flux_with_gendf' in str(exc.value)
    model.run.assert_not_called()


# --- Streaming scaler vs table applicator; per-call .tpe cache ---

def test_streaming_scaler_matches_table_applicator(urr_library):
    """The streaming collapse with a real scaler (several row blocks) equals the
    table applicator followed by the table collapse, to 1e-13 relative."""
    lib, _reads = urr_library
    reactions = ['(n,gamma)', 'fission', '(n,2n)']
    mts = [REACTION_MT[r] for r in reactions]
    densities = {'W182': 0.03, 'W184': 0.02, 'Fe56': 0.05, 'H1': 0.01}
    n = lib.n_groups
    phi = _normalize_flux_batch(
        [np.random.default_rng(1).uniform(0.0, 1.0, n)], 0, n)

    scaler = _CalendfRowScaler(lib, _CALENDF, densities)
    stream = _collapse_gendf_streaming(lib, _W_NUCLIDES, reactions, mts, phi,
                                       block_rows=3, scaler=scaler)[0]
    table = _build_sparse_xs_table(lib, _W_NUCLIDES, reactions, mts)
    _apply_mat_ssf_gendf(table, lib, _CALENDF, densities)
    np.testing.assert_allclose(stream, table.collapse(phi[0]),
                               rtol=1e-13, atol=0)

    unshielded = _build_sparse_xs_table(
        lib, _W_NUCLIDES, reactions, mts).collapse(phi[0])
    ig = reactions.index('(n,gamma)')
    iw, ife = _W_NUCLIDES.index('W182'), _W_NUCLIDES.index('Fe56')
    assert stream[iw, ig] != pytest.approx(unshielded[iw, ig], rel=1e-12)
    assert stream[ife, ig] == pytest.approx(unshielded[ife, ig], rel=1e-13)


def test_wrapper_factor_sets_cached_per_material(monkeypatch, urr_library):
    """Three domains over two materials read each .tpe file once per call."""
    lib, reads = urr_library
    n = lib.n_groups
    mat_a = _material(_W_NUCLIDES, [0.3, 0.2, 0.5, 0.1], 0.08)
    mat_b = _material(_W_NUCLIDES, [0.05, 0.01, 0.9, 0.2], 0.09)

    (_fluxes, micros), _model = _run_gendf_wrapper(
        monkeypatch, [mat_a, mat_b, mat_a], np.ones((3, n)),
        urr_material_dilution=True, calendf_path=_CALENDF,
        gendf_library=lib, chain_nuclides=_W_NUCLIDES)

    assert len(micros) == 3
    assert reads == {'W182-294.tpe': 1, 'W184-294.tpe': 1}
