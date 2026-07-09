"""Unit tests for the fused ``urr_material_dilution`` API of
:meth:`MicroXS.from_multigroup_flux`.

The toggle and its sigma_0 composition are a single parameter: an
``openmc.Material`` or a ``{nuclide: density-or-fraction}`` mapping turns the URR
material-dilution self-shielding correction on, ``False`` (default) or ``None``
leaves it off, and the impossible "on with no composition" states (bare ``True``,
an empty mapping) raise ``ValueError``. These tests exercise the normalization
and the end-to-end correction against a duck-typed PENDF library that carries a
probability table (no data files required).
"""
import numpy as np
import pytest

from openmc.deplete.microxs import MicroXS


class _FakePtab:
    """Duck stand-in for openmc.data.urr.ProbabilityTables.

    mat_ssf reads only ``energy`` (URR node energies, eV), ``table`` (shape
    ``(n_nodes, 6, n_bands)``; dim-1 index 0 cumulative prob, 1 total, 3 fission,
    4 capture) and ``multiply_smooth``.
    """

    def __init__(self, energy, table, multiply_smooth=False):
        self.energy = np.asarray(energy, dtype=float)
        self.table = np.asarray(table, dtype=float)
        self.multiply_smooth = multiply_smooth


class _FakePendfURR:
    """Duck stand-in for a pointwise PendfLibrary that also carries ptables."""

    def __init__(self, data, ptabs=None):
        # data: {nuclide: {mt: (energy, xs)}}
        self._data = data
        self._ptabs = ptabs or {}

    @property
    def nuclides(self):
        return list(self._data)

    def reactions(self, nuclide):
        return list(self._data[nuclide])

    def xs(self, nuclide, mt):
        return self._data[nuclide][mt]

    def ptables(self, nuclide):
        return self._ptabs.get(nuclide)


# Group structure with a single URR-overlapping middle group ([1e3, 1e5]); the
# probability table nodes below both fall in it, so the correction lands there
# and groups 0/2 stay at f=1.
EDGES = [0.0, 1.0e3, 1.0e5, 2.0e7]
FLUX = [1.0, 1.0, 1.0]
_EFULL = np.array([0.0, 2.0e7])

# Two equiprobable bands, one low- and one high-total. Absolute (multiply_smooth
# False), capture correlated with total. sigma_x,inf = 55 b; at sigma_0 = 50 b
# the flux weight biases toward the low-total band -> f = 0.649351.
_NODE = np.array([
    [0.5, 1.0],        # 0 cumulative probability
    [10.0, 100.0],     # 1 total
    [0.0, 0.0],        # 2 elastic
    [0.0, 0.0],        # 3 fission
    [10.0, 100.0],     # 4 (n,gamma)
    [0.0, 0.0],        # 5 heating
])
_PTAB_U238 = _FakePtab([5.0e3, 5.0e4], np.stack([_NODE, _NODE]))

# Expected self-shielding factor in the middle group and the resulting one-group
# collapse of a flat 3 b capture over the three equal-flux groups.
_F = (10.0 / 120.0 + 100.0 / 300.0) / (1.0 / 120.0 + 1.0 / 300.0) / 55.0
_ON_EXPECT = (3.0 + 3.0 * _F + 3.0) / 3.0


def _fake_library():
    # U238 (flagged, resonant) carries a flat 3 b capture and the ptable; O16 is
    # the diluter, a flat 4 b total. sigma_0 = (n_O16/n_U238) * 4 b.
    return _FakePendfURR(
        {
            "U238": {102: (_EFULL, np.array([3.0, 3.0]))},
            "O16": {1: (_EFULL, np.array([4.0, 4.0]))},
        },
        {"U238": _PTAB_U238},
    )


def _capture(mx):
    return float(np.ravel(mx["U238", "(n,gamma)"])[0])


def _collapse(**dilution):
    return MicroXS.from_multigroup_flux(
        energies=EDGES, multigroup_flux=FLUX,
        nuclides=["U238"], reactions=["(n,gamma)"],
        pendf_library=_fake_library(), **dilution)


# ---------------------------------------------------------------------------
# Normalization: impossible states are rejected
# ---------------------------------------------------------------------------

def test_urr_material_dilution_true_raises():
    """Bare ``True`` has no composition and is rejected."""
    with pytest.raises(ValueError, match="under-specified"):
        _collapse(urr_material_dilution=True)


def test_urr_material_dilution_empty_mapping_raises():
    """An empty composition would silently degrade to f=1 everywhere."""
    with pytest.raises(ValueError, match="empty"):
        _collapse(urr_material_dilution={})


def test_urr_material_dilution_requires_pendf_library():
    """Dilution is undefined on the continuous-energy path."""
    with pytest.raises(ValueError, match="pendf_library"):
        MicroXS.from_multigroup_flux(
            energies=EDGES, multigroup_flux=FLUX,
            nuclides=["U238"], reactions=["(n,gamma)"],
            urr_material_dilution={"U238": 1.0, "O16": 12.5})


# ---------------------------------------------------------------------------
# Off paths leave the collapse untouched
# ---------------------------------------------------------------------------

def test_urr_material_dilution_off_unchanged():
    """``False`` (default) and ``None`` both leave the collapse unchanged."""
    base = _capture(_collapse())
    assert _capture(_collapse(urr_material_dilution=False)) == base
    assert _capture(_collapse(urr_material_dilution=None)) == base
    assert base == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# On path: correction fires, Material and dict agree
# ---------------------------------------------------------------------------

def test_urr_material_dilution_material_matches_dict():
    """An openmc.Material and its atom-density dict give the same result."""
    import openmc

    mat = openmc.Material()
    mat.add_nuclide("U238", 1.0)
    mat.add_nuclide("O16", 12.5)   # n_O16/n_U238 = 12.5 -> sigma_0 = 50 b
    mat.set_density("atom/b-cm", 1.0)
    dens = mat.get_nuclide_atom_densities()

    off = _capture(_collapse())
    on_dict = _capture(_collapse(urr_material_dilution=dens))
    on_mat = _capture(_collapse(urr_material_dilution=mat))

    # The correction actually shields (on < off) and lands on the expected value.
    assert on_dict < off
    assert on_dict == pytest.approx(_ON_EXPECT, rel=1e-9)
    # Material and equivalent dict are byte-identical.
    assert on_mat == pytest.approx(on_dict, rel=0.0, abs=0.0)
