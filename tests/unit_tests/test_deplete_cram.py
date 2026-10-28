"""Tests for cram.py.

Compares a few Mathematica matrix exponentials to CRAM16/CRAM48.
Tests substep accuracy against self-converged reference solutions.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from pytest import approx
from openmc.deplete.cram import (CRAM16, CRAM48, Cram16Solver, Cram48Solver,
                                  IPFCramSolver)


def test_CRAM16():
    """Test 16-term CRAM."""
    x = np.array([1.0, 1.0])
    mat = sp.csr_matrix([[-1.0, 0.0], [-2.0, -3.0]])
    dt = 0.1

    z = CRAM16(mat, x, dt)

    # Solution from mathematica
    z0 = np.array((0.904837418035960, 0.576799023327476))

    assert z == approx(z0)


def test_CRAM48():
    """Test 48-term CRAM."""
    x = np.array([1.0, 1.0])
    mat = sp.csr_matrix([[-1.0, 0.0], [-2.0, -3.0]])
    dt = 0.1

    z = CRAM48(mat, x, dt)

    # Solution from mathematica
    z0 = np.array((0.904837418035960, 0.576799023327476))

    assert z == approx(z0)


def test_substeps2_matches_two_half_steps():
    """substeps=2 must match two independent CRAM calls with dt/2."""
    x = np.array([1.0, 1.0])
    mat = sp.csr_matrix([[-1.0, 0.0], [-2.0, -3.0]])
    dt = 1.0

    # Two manual half-steps, each a single-step call
    z_half = CRAM48(mat, x, dt / 2)
    z_two = CRAM48(mat, z_half, dt / 2)

    # Single call with substeps=2
    z_sub2 = CRAM48(mat, x, dt, substeps=2)

    assert z_sub2 == approx(z_two, rel=1e-12)


@pytest.mark.parametrize("substeps", [0, -1])
def test_invalid_substeps(substeps):
    """substeps must be a positive integer at call time."""
    x = np.array([1.0, 1.0])
    mat = sp.csr_matrix([[-1.0, 0.0], [-2.0, -3.0]])

    with pytest.raises(ValueError, match="substeps"):
        CRAM48(mat, x, 0.1, substeps=substeps)


def test_substeps_self_convergence():
    """Increasing substeps converges toward reference solution.

    Uses CRAM16 (alpha0 ~ 2e-16) where substep convergence is visible.
    CRAM48 (alpha0 ~ 2e-47) is already near machine precision for small
    systems; its correctness is verified by the other substep tests.
    """
    mat = sp.csr_matrix([[-1.0, 0.0], [-2.0, -3.0]])
    x = np.array([1.0, 1.0])
    dt = 50  # lambda*dt = 50 and 150, stresses CRAM16

    n_ref = CRAM16(mat, x, dt, substeps=128)

    prev_err = np.inf
    for s in [1, 2, 4, 8, 16]:
        n_s = CRAM16(mat, x, dt, substeps=s)
        err = np.linalg.norm(n_s - n_ref) / np.linalg.norm(n_ref)
        assert err < prev_err, \
            f"substeps={s} error {err:.2e} not less than previous {prev_err:.2e}"
        prev_err = err


def _burnup_matrix(nuclides, transitions):
    """Burnup matrix from (parent, daughter, rate) transitions."""
    index = {name: i for i, name in enumerate(nuclides)}
    mat = np.zeros((len(nuclides), len(nuclides)))
    for parent, daughter, rate in transitions:
        mat[index[parent], index[parent]] -= rate
        mat[index[daughter], index[parent]] += rate
    return sp.csr_matrix(mat)


@pytest.mark.parametrize("substeps", [1, 4])
def test_CRAM48_irradiation_and_cooling(substeps):
    """Precision of Mo activation: 10 h irradiation, then 100 d cooling.

    Nb98 is checked against its analytic secular equilibrium with Mo98, and
    Nb97 and Nb97_m1 must have decayed out after cooling. With row pivoting
    in the sparse LU solve, the first check fails with SuperLU's default
    COLAMD ordering and the second with natural ordering.
    """
    # Nuclides in chain order; rates in 1/s
    nuclides = ['Y98', 'Zr96', 'Zr97', 'Zr98', 'Nb97', 'Nb97_m1', 'Nb98',
                'Mo97', 'Mo98']
    lam_nb98 = 0.2
    r_p, r_np, r_2p, r_he3 = 1e-16, 2e-17, 1e-29, 1e-28  # Mo98 reactions
    decay = [
        ('Y98', 'Zr98', 1.0),           # beta-
        ('Y98', 'Zr97', 3e-3),          # beta-,n
        ('Zr97', 'Nb97_m1', 1e-5),      # beta-
        ('Zr98', 'Nb98', 0.02),         # beta-
        ('Nb97_m1', 'Nb97', 0.01),      # isomeric transition
        ('Nb97', 'Mo97', 1e-4),         # beta-
        ('Nb98', 'Mo98', lam_nb98),     # beta-
    ]
    reactions = [
        ('Zr96', 'Zr97', 1e-17),        # (n,gamma)
        ('Mo97', 'Zr96', 1e-24),        # (n,2p)
        ('Mo98', 'Nb97', r_np),         # (n,np)
        ('Mo98', 'Nb98', r_p),          # (n,p)
        ('Mo98', 'Zr97', r_2p),         # (n,2p)
        ('Mo98', 'Zr96', r_he3),        # (n,3He)
    ]
    x = np.array([0, 0, 0, 0, 0, 0, 0, 1e21, 2e21])  # Mo97 and Mo98 only
    dt = 36000.0

    z = CRAM48(_burnup_matrix(nuclides, decay + reactions), x, dt,
               substeps=substeps)
    z_cool = CRAM48(_burnup_matrix(nuclides, decay), z, 8.64e6,
                    substeps=substeps)

    # Nb98 (2.9 s half-life) is in secular equilibrium with Mo98, whose
    # (n,p) losses return through Nb98 decay
    loss = r_np + r_2p + r_he3
    nb98 = r_p * x[8] * np.exp(-loss * dt) / (lam_nb98 - loss)
    assert z[6] == approx(nb98, rel=1e-12), \
        f"Nb98 relative error {abs(z[6] / nb98 - 1):.1e}"

    # All sources of Nb97 and Nb97_m1 decay with rates >= 1e-5/s, so after
    # 100 d both are below 1e-60 of the Mo inventory. CRAM48 itself leaves
    # about 1e-47 of the initial Nb97 (either sign).
    bound = 1e-40 * z_cool.max()
    for i in (4, 5):
        assert abs(z_cool[i]) <= bound, \
            f"{nuclides[i]} = {z_cool[i]:.1e} after cooling, above {bound:.0e}"


@pytest.mark.parametrize("dt, substeps", [(86400.0, 1), (30 * 86400.0, 8)])
def test_CRAM48_stable_nuclide_decay(dt, substeps):
    """Pure decay of stable Sn118 must not create In118 states.

    Only Sn118 is present at t=0, so the exact solution is unchanged. With
    row pivoting in the sparse LU solve, SuperLU's default settings create
    In118_m1 and In118_m2 atoms (thousands, either sign) out of rounding
    error. Half-lives and branching ratios are from UKAEA DECAY2012 data.
    """
    # Nuclides in chain order; decay constants in 1/s
    nuclides = ['In118', 'In118_m1', 'In118_m2', 'Sn118']
    lam = {name: np.log(2) / t for name, t in
           [('In118', 5.0), ('In118_m1', 267.0), ('In118_m2', 8.5)]}
    decay = [
        ('In118', 'Sn118', lam['In118']),                   # beta-
        ('In118_m1', 'Sn118', lam['In118_m1']),             # beta-
        ('In118_m2', 'In118_m1', 0.985 * lam['In118_m2']),  # IT
        ('In118_m2', 'Sn118', 0.015 * lam['In118_m2']),     # beta-
    ]
    x = np.array([0.0, 0.0, 0.0, 1e22])

    z = CRAM48(_burnup_matrix(nuclides, decay), x, dt, substeps=substeps)

    assert z[3] == approx(x[3], rel=1e-14)
    for i, name in enumerate(nuclides[:3]):
        assert abs(z[i]) <= 1e-30 * x[3], f"{name} = {z[i]:.1e} atoms"
