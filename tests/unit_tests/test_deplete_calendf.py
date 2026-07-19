"""Unit tests for the CALENDF mutual-shielding sigma_0 iteration (C3).

GENDF sibling of ``test_deplete_mat_ssf.py`` (PENDF side). Covers the C3
total-XS self-shielding factor (:func:`mat_ssf_total_factors_gendf`) and the
Jacobi mutual-shielding fixed point (:func:`iterate_material_dilution_sigma0_gendf`)
with synthetic band tables (no data dependency) -- the permanent G-A (analytic
convergence) and G-B (mono / inert-diluter no-op) gates.
"""
import numpy as np
import pytest

from openmc.deplete.calendf import (
    SIGMA0_ITER_MAX, SIGMA0_ITER_TOL,
    TpeTable, _BandGroup,
    material_dilution_sigma0, mat_ssf_factors_gendf,
    mat_ssf_total_factors_gendf, iterate_material_dilution_sigma0_gendf)

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

    monkeypatch.setattr('openmc.deplete.calendf.SIGMA0_ITER_MAX', 1)
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
