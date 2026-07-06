"""Unit tests for :func:`openmc.deplete.get_pendf_microxs_and_flux`.

The transport-coupled PENDF wrapper runs one flux-only transport solve and then
collapses group cross sections out of a ``pendf_library`` per domain. These tests
never touch real transport: they patch :meth:`openmc.Model.run` (and
``StatePoint``) to prove the tally set-up, the pre-run validation, and -- the
load-bearing gate -- that given identical flux each returned :class:`MicroXS`
equals the direct :meth:`MicroXS.from_multigroup_flux` call for that domain
(gate G-EQ), with the flag-off path equal to the direct ``False`` call
(gate G-OFF).

The duck-typed PENDF library carrying a probability table is reused from
``test_deplete_mat_ssf`` -- no data files are required.
"""
from unittest.mock import Mock, patch

import numpy as np
import pytest

import openmc
from openmc.deplete import get_pendf_microxs_and_flux
from openmc.deplete.microxs import MicroXS

# Reuse the fused-API duck fixtures (pointwise PENDF lib + probability table).
from tests.unit_tests.test_deplete_mat_ssf import (
    _FakePendfURR, EDGES, _fake_library,
)

N_GROUPS = len(EDGES) - 1  # 3


# ---------------------------------------------------------------------------
# Test doubles for the transport solve / statepoint read
# ---------------------------------------------------------------------------

class _FakeFluxTally:
    """Stand-in for a statepoint flux tally with canned reshaped data.

    ``get_reshaped_data`` returns a ``(n_domains, n_groups, 1, 1)`` array, the
    exact shape the wrapper's energy-filtered flux tally yields before its
    ``moveaxis(..., 1, -1).squeeze((1, 2))``.
    """

    def __init__(self, data):
        self._data = np.asarray(data, dtype=float)

    def _read_results(self):
        pass

    def get_reshaped_data(self):
        return self._data


class _FakeTallies:
    """``sp.tallies[id]`` always hands back the single canned flux tally."""

    def __init__(self, tally):
        self._tally = tally

    def __getitem__(self, key):
        return self._tally


class _FakeStatePoint:
    """Context-manager stand-in for :class:`openmc.StatePoint`."""

    def __init__(self, tally):
        self.tallies = _FakeTallies(tally)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _GroupedDuck:
    """Minimal grouped-library duck exposing only ``group_edges``.

    Enough for the ``energies=None`` group-structure default and the flux-tally
    build; the collapse is never reached in the tests that use it.
    """

    def __init__(self, group_edges):
        self.group_edges = np.asarray(group_edges, dtype=float)


def _bare_model():
    return openmc.Model()


def _uo2_material(o16_density):
    """U238 + O16 material; ``n_O16/n_U238 = o16_density`` sets sigma_0."""
    mat = openmc.Material()
    mat.add_nuclide("U238", 1.0)
    mat.add_nuclide("O16", o16_density)
    mat.set_density("atom/b-cm", 1.0)
    return mat


# ---------------------------------------------------------------------------
# Tally set-up (StopIteration pattern) -- flux-only, no reaction-rate tally
# ---------------------------------------------------------------------------

def test_tally_setup_flux_only_explicit_energies():
    """One flux tally on the given edges; no reaction-rate tally is built."""
    model = _bare_model()
    mat = _uo2_material(12.5)
    lib = _fake_library()

    captured = {}

    def capture_run(**kwargs):
        captured['tallies'] = list(model.tallies)
        raise StopIteration

    with patch.object(model, 'run', side_effect=capture_run):
        with pytest.raises(StopIteration):
            get_pendf_microxs_and_flux(
                model, [mat], pendf_library=lib,
                nuclides=["U238"], reactions=["(n,gamma)"],
                energies=EDGES,
            )

    tallies = captured['tallies']
    names = [t.name for t in tallies]
    assert 'MicroXS flux 0' in names
    # Flux-only: no reaction-rate tally exists.
    assert not any(n.startswith('MicroXS RR') for n in names)
    assert len(tallies) == 1

    flux_tally = tallies[0]
    assert flux_tally.scores == ['flux']
    assert flux_tally.nuclides == []
    ef = next(f for f in flux_tally.filters if isinstance(f, openmc.EnergyFilter))
    np.testing.assert_allclose(ef.values, EDGES)


def test_tally_setup_energies_default_from_grouped_group_edges():
    """energies=None defaults the tally edges to a grouped library's group_edges."""
    model = _bare_model()
    mat = _uo2_material(12.5)
    lib = _GroupedDuck(EDGES)

    captured = {}

    def capture_run(**kwargs):
        captured['tallies'] = list(model.tallies)
        raise StopIteration

    with patch.object(model, 'run', side_effect=capture_run):
        with pytest.raises(StopIteration):
            get_pendf_microxs_and_flux(
                model, [mat], pendf_library=lib,
                nuclides=["U238"], reactions=["(n,gamma)"],
                energies=None,
            )

    flux_tally = captured['tallies'][0]
    ef = next(f for f in flux_tally.filters if isinstance(f, openmc.EnergyFilter))
    np.testing.assert_allclose(ef.values, EDGES)


# ---------------------------------------------------------------------------
# Pre-run validation (all raise BEFORE model.run -- proven by a run spy)
# ---------------------------------------------------------------------------

def test_dilution_true_with_cell_domains_raises_before_run():
    """dilution=True requires Material domains; a Cell domain raises pre-run."""
    model = _bare_model()
    cell = openmc.Cell()
    lib = _fake_library()
    run_spy = Mock()

    with patch.object(model, 'run', run_spy):
        with pytest.raises(ValueError,
                           match="every domain to be an openmc.Material"):
            get_pendf_microxs_and_flux(
                model, [cell], pendf_library=lib,
                nuclides=["U238"], reactions=["(n,gamma)"],
                energies=EDGES, urr_material_dilution=True,
            )

    run_spy.assert_not_called()


def test_non_bool_dilution_raises_redirect_before_run():
    """A Material passed as the wrapper-level toggle gets the redirect message."""
    model = _bare_model()
    mat = _uo2_material(12.5)
    lib = _fake_library()
    run_spy = Mock()

    with patch.object(model, 'run', run_spy):
        with pytest.raises(
                ValueError,
                match=r"must be a bool for get_pendf_microxs_and_flux"):
            get_pendf_microxs_and_flux(
                model, [mat], pendf_library=lib,
                nuclides=["U238"], reactions=["(n,gamma)"],
                energies=EDGES, urr_material_dilution=mat,
            )

    run_spy.assert_not_called()


def test_non_bool_dilution_redirect_message_verbatim():
    """The redirect message is the canonical wording, verbatim."""
    model = _bare_model()
    mat = _uo2_material(12.5)
    lib = _fake_library()

    expected = (
        "urr_material_dilution must be a bool for "
        "get_pendf_microxs_and_flux(); True uses each domain's own "
        "composition automatically. To supply an explicit composition "
        "(openmc.Material or {nuclide: density} mapping), call "
        "MicroXS.from_multigroup_flux directly."
    )
    with patch.object(model, 'run', Mock()):
        with pytest.raises(ValueError) as excinfo:
            get_pendf_microxs_and_flux(
                model, [mat], pendf_library=lib,
                nuclides=["U238"], reactions=["(n,gamma)"],
                energies=EDGES, urr_material_dilution=mat,
            )
    assert str(excinfo.value) == expected


def test_pointwise_library_energies_none_raises_before_run():
    """A pointwise library has no group_edges; energies=None must raise pre-run."""
    model = _bare_model()
    mat = _uo2_material(12.5)
    lib = _fake_library()  # pointwise: no group_edges
    run_spy = Mock()

    with patch.object(model, 'run', run_spy):
        with pytest.raises(ValueError, match="energies must be provided"):
            get_pendf_microxs_and_flux(
                model, [mat], pendf_library=lib,
                nuclides=["U238"], reactions=["(n,gamma)"],
                energies=None,
            )

    run_spy.assert_not_called()


# ---------------------------------------------------------------------------
# G-EQ / G-OFF: transport-coupled == flux-supplied (exact array equality)
# ---------------------------------------------------------------------------

def _run_wrapper_with_canned_flux(model, domains, lib, canned, **kwargs):
    """Drive the wrapper with a canned per-domain flux (no real transport)."""
    fake_tally = _FakeFluxTally(canned)
    fake_sp = _FakeStatePoint(fake_tally)
    with patch.object(model, 'run', return_value='sp.h5'), \
            patch('openmc.deplete.microxs.StatePoint', return_value=fake_sp):
        return get_pendf_microxs_and_flux(
            model, domains, pendf_library=lib,
            nuclides=["U238"], reactions=["(n,gamma)"],
            energies=EDGES, **kwargs)


def test_geq_dilution_on_equals_direct_per_domain():
    """Each domain's MicroXS equals the direct dilution call with that material."""
    lib = _fake_library()
    mat0 = _uo2_material(12.5)   # sigma_0 = 50 b
    mat1 = _uo2_material(50.0)   # sigma_0 = 200 b (less shielding)
    flux0 = [1.0, 2.0, 3.0]
    flux1 = [4.0, 5.0, 6.0]
    canned = np.array([flux0, flux1]).reshape(2, N_GROUPS, 1, 1)

    model = _bare_model()
    fluxes, micros = _run_wrapper_with_canned_flux(
        model, [mat0, mat1], lib, canned, urr_material_dilution=True)

    # Raw tallied flux magnitudes are returned unmodified, one per domain.
    assert len(fluxes) == 2
    np.testing.assert_array_equal(fluxes[0], np.array(flux0))
    np.testing.assert_array_equal(fluxes[1], np.array(flux1))

    direct0 = MicroXS.from_multigroup_flux(
        energies=EDGES, multigroup_flux=flux0,
        nuclides=["U238"], reactions=["(n,gamma)"],
        pendf_library=_fake_library(), urr_material_dilution=mat0)
    direct1 = MicroXS.from_multigroup_flux(
        energies=EDGES, multigroup_flux=flux1,
        nuclides=["U238"], reactions=["(n,gamma)"],
        pendf_library=_fake_library(), urr_material_dilution=mat1)

    np.testing.assert_array_equal(micros[0].data, direct0.data)
    np.testing.assert_array_equal(micros[1].data, direct1.data)
    assert micros[0].nuclides == direct0.nuclides == ["U238"]
    assert micros[0].reactions == direct0.reactions == ["(n,gamma)"]

    # The two compositions actually give different self-shielding (proves the
    # per-domain composition -- not a shared one -- drives each collapse).
    assert not np.array_equal(micros[0].data, micros[1].data)


def test_goff_dilution_off_equals_direct_false_per_domain():
    """Flag-off is identical to the direct from_multigroup_flux(False) call."""
    lib = _fake_library()
    mat0 = _uo2_material(12.5)
    mat1 = _uo2_material(50.0)
    flux0 = [1.0, 2.0, 3.0]
    flux1 = [4.0, 5.0, 6.0]
    canned = np.array([flux0, flux1]).reshape(2, N_GROUPS, 1, 1)

    model = _bare_model()
    _, micros_off = _run_wrapper_with_canned_flux(
        model, [mat0, mat1], lib, canned, urr_material_dilution=False)

    direct0 = MicroXS.from_multigroup_flux(
        energies=EDGES, multigroup_flux=flux0,
        nuclides=["U238"], reactions=["(n,gamma)"],
        pendf_library=_fake_library(), urr_material_dilution=False)
    direct1 = MicroXS.from_multigroup_flux(
        energies=EDGES, multigroup_flux=flux1,
        nuclides=["U238"], reactions=["(n,gamma)"],
        pendf_library=_fake_library(), urr_material_dilution=False)

    np.testing.assert_array_equal(micros_off[0].data, direct0.data)
    np.testing.assert_array_equal(micros_off[1].data, direct1.data)

    # Flag off must differ from flag on for the shielded domain (correction fires).
    _, micros_on = _run_wrapper_with_canned_flux(
        model, [mat0, mat1], lib, canned, urr_material_dilution=True)
    assert not np.array_equal(micros_off[0].data, micros_on[0].data)
