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


# ---------------------------------------------------------------------------
# Mutual-shielding sigma_0 iteration (C2): total-XS factor + Jacobi fixed point
# ---------------------------------------------------------------------------
from openmc.deplete.mat_ssf import (              # noqa: E402
    SIGMA0_ITER_MAX, SIGMA0_ITER_TOL,
    material_dilution_sigma0, mat_ssf_total_factors,
    iterate_material_dilution_sigma0)

# One URR-overlapping group ([1e3, 1e5]) with many nodes inside it, so
# ``_group_average`` can integrate the folded pointwise total (a single node
# would average to 0 and leave R=1).
_ITER_EDGES = np.array([0.0, 1.0e3, 1.0e5, 2.0e7])
_ITER_NODES = np.geomspace(2.0e3, 8.0e4, 12)
_MID = 1  # index of the URR-overlapping group


def _band_ptab(total_lo, total_hi):
    """A synthetic two-equiprobable-band absolute table, totals [lo, hi]."""
    band = np.array([
        [0.5, 1.0],              # 0 cumulative probability
        [total_lo, total_hi],    # 1 total (drives R_tot)
        [0.0, 0.0],              # 2 elastic
        [0.0, 0.0],              # 3 fission
        [0.0, 0.0],              # 4 (n,gamma) -- irrelevant to the total factor
        [0.0, 0.0],              # 5 heating
    ])
    return _FakePtab(_ITER_NODES, np.stack([band] * len(_ITER_NODES)))


def _dense_fixed_point(densities, group_totals, coupling, edges, passes=500):
    """Independent high-iteration reference solution of the sigma_0 fixed point.

    Runs the same Jacobi map far past the production ``SIGMA0_ITER_MAX`` cap, so
    the result is the true fixed point to solve against (contraction guarantees
    convergence).
    """
    n_groups = len(edges) - 1
    d = {p: material_dilution_sigma0(densities, p, group_totals) for p in coupling}
    for _ in range(passes):
        R = {q: mat_ssf_total_factors(coupling[q][0], d[q], edges,
                                      smooth_total=coupling[q][1])
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


def test_mat_ssf_total_factor_limits():
    """R_tot matches the analytic band harmonic/arithmetic ratio and -> 1 dilute."""
    ptab = _band_ptab(10.0, 100.0)   # p=[.5,.5], sigma_t,inf = 55 b
    # sigma_0 = 0: sigma_t,eff = 1 / (0.5/10 + 0.5/100) = 18.1818 b -> R = 0.33058
    r0 = mat_ssf_total_factors(ptab, np.zeros(3), _ITER_EDGES)
    assert r0[_MID] == pytest.approx((1.0 / 0.055) / 55.0, rel=1e-9)
    # infinite dilution: R -> 1 exactly; outside the URR span: R == 1
    r_inf = mat_ssf_total_factors(ptab, np.full(3, 1.0e12), _ITER_EDGES)
    assert r_inf[_MID] == pytest.approx(1.0, rel=1e-6)
    assert r0[0] == 1.0 and r0[2] == 1.0
    # monotone increasing in sigma_0 (more dilution -> less shielding)
    rs = [mat_ssf_total_factors(ptab, np.full(3, s), _ITER_EDGES)[_MID]
          for s in (0.0, 5.0, 55.0, 1.0e3)]
    assert np.all(np.diff(rs) > 0)


def test_sigma0_iteration_converges_to_fixed_point():
    """G-A: two synthetic resonant nuclides -> iteration reaches the analytic d*."""
    edges = _ITER_EDGES
    coupling = {'A': (_band_ptab(20.0, 80.0), None),
                'B': (_band_ptab(30.0, 70.0), None)}
    densities = {'A': 0.6, 'B': 0.4}
    group_totals = {'A': np.full(3, 50.0), 'B': np.full(3, 50.0)}

    sigma0, info = iterate_material_dilution_sigma0(
        densities, group_totals, coupling, edges)

    # Converged within the production bound.
    assert info['converged']
    assert info['n_iter'] <= SIGMA0_ITER_MAX

    ref = _dense_fixed_point(densities, group_totals, coupling, edges)
    R = {q: mat_ssf_total_factors(coupling[q][0], sigma0[q], edges)
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


def test_sigma0_iteration_jacobi_order_independent():
    """G-A companion: the simultaneous update is independent of nuclide order."""
    edges = _ITER_EDGES
    pA, pB = _band_ptab(20.0, 80.0), _band_ptab(30.0, 70.0)
    gt = {'A': np.full(3, 50.0), 'B': np.full(3, 50.0)}
    d1, _ = iterate_material_dilution_sigma0(
        {'A': 0.6, 'B': 0.4}, gt, {'A': (pA, None), 'B': (pB, None)}, edges)
    d2, _ = iterate_material_dilution_sigma0(
        {'B': 0.4, 'A': 0.6}, gt, {'B': (pB, None), 'A': (pA, None)}, edges)
    assert np.array_equal(d1['A'], d2['A'])
    assert np.array_equal(d1['B'], d2['B'])


def test_sigma0_iteration_mono_noop():
    """G-B: a single resonant nuclide (no diluters) -> bit-identical no-op."""
    edges = _ITER_EDGES
    densities = {'X': 1.0}
    group_totals = {'X': np.full(3, 50.0)}
    coupling = {'X': (_band_ptab(20.0, 80.0), None)}

    d0 = material_dilution_sigma0(densities, 'X', group_totals)
    sigma0, info = iterate_material_dilution_sigma0(
        densities, group_totals, coupling, edges)
    assert np.array_equal(d0, np.zeros(3))         # pure absorber, sigma_0 = 0
    assert np.array_equal(sigma0['X'], d0)         # iteration leaves it untouched
    assert info['converged']


def test_sigma0_iteration_inert_diluter_noop():
    """G-B: one resonant nuclide in a non-ptable diluter -> iteration no-op."""
    edges = _ITER_EDGES
    densities = {'X': 1.0, 'M': 4.0}               # M inert (no ptable, R=1)
    group_totals = {'X': np.full(3, 50.0), 'M': np.full(3, 6.0)}
    coupling = {'X': (_band_ptab(20.0, 80.0), None)}

    d0 = material_dilution_sigma0(densities, 'X', group_totals)
    sigma0, _ = iterate_material_dilution_sigma0(
        densities, group_totals, coupling, edges)
    # X sees only the inert diluter: sigma_0 = n_M * s_M / n_X = 24 b, unchanged.
    assert sigma0['X'][_MID] == pytest.approx(24.0)
    assert np.array_equal(sigma0['X'], d0)


# ---------------------------------------------------------------------------
# D3 rider: warn when a diluter "total" is reconstructed without MT=1 or MT=2
# ---------------------------------------------------------------------------
from openmc.deplete.mat_ssf import _library_group_total   # noqa: E402

_D3_E = np.array([0.0, 2.0e7])
_D3_EDGES = np.array([0.0, 1.0e3, 1.0e5, 2.0e7])


def _mt_lib(mts):
    return _FakePendfURR({'D': {mt: (_D3_E, np.array([1.0, 1.0])) for mt in mts}})


def test_library_group_total_warns_only_without_total_or_elastic():
    """MT=1 or MT=2 present -> silent; capture-only reconstruction -> warns."""
    import warnings

    # MT=1 present, or MT=2 present: no warning.
    for mts in ([1, 102], [2, 102]):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert _library_group_total(_mt_lib(mts), 'D', _D3_EDGES, False) \
                is not None

    # Neither MT=1 nor MT=2: the elastic channel is missing -> warn.
    with pytest.warns(UserWarning, match="neither MT=1 .* nor MT=2"):
        _library_group_total(_mt_lib([102, 18]), 'D', _D3_EDGES, False)
