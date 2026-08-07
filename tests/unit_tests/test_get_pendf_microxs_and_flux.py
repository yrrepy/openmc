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
from pathlib import Path
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

# The PENDF collapse now always needs a chain. The fake library exposes no MF=10
# pathways, so the chain is never consulted for naming -- any loadable chain
# satisfies the requirement.
CHAIN_FILE = Path(__file__).parents[1] / "chain_simple.xml"

# The canonical redirect wording for a non-bool wrapper-level dilution toggle.
_REDIRECT_MSG = (
    "urr_material_dilution must be a bool for "
    "get_pendf_microxs_and_flux(); True uses each domain's own "
    "composition automatically. To supply an explicit composition "
    "(openmc.Material or {nuclide: density} mapping), call "
    "MicroXS.from_multigroup_flux directly."
)


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


def _capture_tallies(model, lib, **kwargs):
    """Drive the wrapper until the tallies are built, capture them, abort the run.

    Returns the list of tallies present on ``model`` at the moment ``model.run``
    would fire -- proving what the wrapper set up without a real transport solve.
    """
    captured = {}

    def capture_run(**_):
        captured['tallies'] = list(model.tallies)
        raise StopIteration

    with patch.object(model, 'run', side_effect=capture_run):
        with pytest.raises(StopIteration):
            get_pendf_microxs_and_flux(
                model, [_uo2_material(12.5)], pendf_library=lib,
                chain_file=CHAIN_FILE, nuclides=["U238"],
                reactions=["(n,gamma)"], **kwargs)
    return captured['tallies']


def _assert_raises_before_run(domains, lib, match, **kwargs):
    """The wrapper raises ``ValueError(match)`` BEFORE ``model.run`` is called.

    A ``run`` spy proves the failure is pre-run; the raised exception is returned
    so the caller can additionally check its verbatim text.
    """
    model = _bare_model()
    run_spy = Mock()
    with patch.object(model, 'run', run_spy):
        with pytest.raises(ValueError, match=match) as excinfo:
            get_pendf_microxs_and_flux(
                model, domains, pendf_library=lib,
                nuclides=["U238"], reactions=["(n,gamma)"], **kwargs)
    run_spy.assert_not_called()
    return excinfo


def _run_wrapper_with_canned_flux(model, domains, lib, canned, **kwargs):
    """Drive the wrapper with a canned per-domain flux (no real transport)."""
    fake_tally = _FakeFluxTally(canned)
    fake_sp = _FakeStatePoint(fake_tally)
    with patch.object(model, 'run', return_value='sp.h5'), \
            patch('openmc.deplete.pendf.collapse.StatePoint', return_value=fake_sp):
        return get_pendf_microxs_and_flux(
            model, domains, pendf_library=lib, chain_file=CHAIN_FILE,
            nuclides=["U238"], reactions=["(n,gamma)"],
            energies=EDGES, **kwargs)


# ---------------------------------------------------------------------------
# Tally set-up (StopIteration pattern) -- flux-only, no reaction-rate tally
# ---------------------------------------------------------------------------

def test_tally_setup_flux_only_and_energies_default():
    """One flux-only tally is built (no reaction-rate tally), and ``energies=None``
    defaults the tally edges to a grouped library's ``group_edges``."""
    # Explicit energies, pointwise library: exactly one flux tally, no RR tally.
    tallies = _capture_tallies(_bare_model(), _fake_library(), energies=EDGES)
    names = [t.name for t in tallies]
    assert 'MicroXS flux 0' in names
    assert not any(n.startswith('MicroXS RR') for n in names)
    assert len(tallies) == 1

    flux_tally = tallies[0]
    assert flux_tally.scores == ['flux']
    assert flux_tally.nuclides == []
    ef = next(f for f in flux_tally.filters if isinstance(f, openmc.EnergyFilter))
    np.testing.assert_allclose(ef.values, EDGES)

    # energies=None + grouped library: tally edges default to group_edges.
    tallies = _capture_tallies(_bare_model(), _GroupedDuck(EDGES), energies=None)
    ef = next(f for f in tallies[0].filters if isinstance(f, openmc.EnergyFilter))
    np.testing.assert_allclose(ef.values, EDGES)


# ---------------------------------------------------------------------------
# Pre-run validation (all raise BEFORE model.run -- proven by a run spy)
# ---------------------------------------------------------------------------

def test_pre_run_validation_raises_before_run():
    """Every wrapper-level input error raises before any transport: a void
    dilution domain, a non-bool dilution toggle (with the verbatim redirect
    message), and ``energies=None`` for a pointwise library."""
    lib = _fake_library()
    mat = _uo2_material(12.5)

    # dilution=True on a void (fill-less) Cell has no single composition.
    _assert_raises_before_run(
        [openmc.Cell()], lib, "filled with a single openmc.Material",
        chain_file=CHAIN_FILE, energies=EDGES, urr_material_dilution=True)

    # A Material passed as the bool toggle gets the redirect message, verbatim.
    excinfo = _assert_raises_before_run(
        [mat], lib, r"must be a bool for get_pendf_microxs_and_flux",
        energies=EDGES, urr_material_dilution=mat)
    assert str(excinfo.value) == _REDIRECT_MSG

    # A pointwise library has no group_edges, so energies=None cannot default.
    _assert_raises_before_run(
        [mat], lib, "energies must be provided", energies=None)


def test_dilution_material_resolution_and_cell_fill_domain():
    """``_pendf_dilution_material`` resolves a domain's shielding composition, and
    a Cell filled with a single Material drives the collapse -- its MicroXS equals
    the direct dilution call with that fill (the tally domain stays the Cell)."""
    from openmc.deplete.pendf.collapse import _pendf_dilution_material

    mat = _uo2_material(12.5)
    cell = openmc.Cell(fill=mat)

    # The resolver returns a Cell's fill Material, and a bare Material itself.
    assert _pendf_dilution_material(cell) is mat
    assert _pendf_dilution_material(mat) is mat

    # End to end: a material-filled Cell domain collapses without raising and
    # equals the direct dilution call with that fill.
    lib = _fake_library()
    flux0 = [1.0, 2.0, 3.0]
    canned = np.array([flux0]).reshape(1, N_GROUPS, 1, 1)
    _, micros = _run_wrapper_with_canned_flux(
        _bare_model(), [cell], lib, canned, urr_material_dilution=True)

    direct = MicroXS.from_multigroup_flux(
        energies=EDGES, multigroup_flux=flux0, chain_file=CHAIN_FILE,
        nuclides=["U238"], reactions=["(n,gamma)"],
        pendf_library=_fake_library(), urr_material_dilution=mat)
    np.testing.assert_array_equal(micros[0].data, direct.data)


# ---------------------------------------------------------------------------
# G-EQ / G-OFF: transport-coupled == flux-supplied (exact array equality)
# ---------------------------------------------------------------------------

def test_geq_goff_equal_direct_per_domain():
    """Transport-coupled MicroXS equals the direct ``from_multigroup_flux`` call
    for each domain with dilution ON (gate G-EQ) and OFF (gate G-OFF); the two
    disagree for the shielded domain, proving the per-domain correction fires."""
    lib = _fake_library()
    mat0 = _uo2_material(12.5)   # sigma_0 = 50 b
    mat1 = _uo2_material(50.0)   # sigma_0 = 200 b (less shielding)
    flux0 = [1.0, 2.0, 3.0]
    flux1 = [4.0, 5.0, 6.0]
    canned = np.array([flux0, flux1]).reshape(2, N_GROUPS, 1, 1)

    # G-EQ: each domain's MicroXS equals the direct dilution call with its material.
    fluxes, micros_on = _run_wrapper_with_canned_flux(
        _bare_model(), [mat0, mat1], lib, canned, urr_material_dilution=True)

    # Raw tallied flux magnitudes are returned unmodified, one per domain.
    assert len(fluxes) == 2
    np.testing.assert_array_equal(fluxes[0], np.array(flux0))
    np.testing.assert_array_equal(fluxes[1], np.array(flux1))

    direct0 = MicroXS.from_multigroup_flux(
        energies=EDGES, multigroup_flux=flux0, chain_file=CHAIN_FILE,
        nuclides=["U238"], reactions=["(n,gamma)"],
        pendf_library=_fake_library(), urr_material_dilution=mat0)
    direct1 = MicroXS.from_multigroup_flux(
        energies=EDGES, multigroup_flux=flux1, chain_file=CHAIN_FILE,
        nuclides=["U238"], reactions=["(n,gamma)"],
        pendf_library=_fake_library(), urr_material_dilution=mat1)
    np.testing.assert_array_equal(micros_on[0].data, direct0.data)
    np.testing.assert_array_equal(micros_on[1].data, direct1.data)
    assert micros_on[0].nuclides == direct0.nuclides == ["U238"]
    assert micros_on[0].reactions == direct0.reactions == ["(n,gamma)"]

    # Different compositions give different self-shielding (per-domain, not shared).
    assert not np.array_equal(micros_on[0].data, micros_on[1].data)

    # G-OFF: flag-off is identical to the direct from_multigroup_flux(False) call.
    _, micros_off = _run_wrapper_with_canned_flux(
        _bare_model(), [mat0, mat1], lib, canned, urr_material_dilution=False)
    off0 = MicroXS.from_multigroup_flux(
        energies=EDGES, multigroup_flux=flux0, chain_file=CHAIN_FILE,
        nuclides=["U238"], reactions=["(n,gamma)"],
        pendf_library=_fake_library(), urr_material_dilution=False)
    off1 = MicroXS.from_multigroup_flux(
        energies=EDGES, multigroup_flux=flux1, chain_file=CHAIN_FILE,
        nuclides=["U238"], reactions=["(n,gamma)"],
        pendf_library=_fake_library(), urr_material_dilution=False)
    np.testing.assert_array_equal(micros_off[0].data, off0.data)
    np.testing.assert_array_equal(micros_off[1].data, off1.data)

    # Flag off must differ from flag on for the shielded domain (correction fires).
    assert not np.array_equal(micros_off[0].data, micros_on[0].data)
