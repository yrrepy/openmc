"""Regression tests for threshold-reaction MF=10 alignment (Python backend).

Real libraries carry partial-range MF=10 sections for threshold reactions
((n,2n), (n,p), (n,alpha), ...): the section's energy grid starts at the
reaction threshold, not at the bottom of the library group structure. The
Python backend must align these bands onto the full group grid by energy
(mirroring the C++ backend fix in commit 6e66d2e18). Left-placing the raw
values puts high-energy data in the lowest groups, which crashes
IsomericBranchingHelper._calculate_weighted with an IndexError, or -- when
ground/metastable levels have different thresholds (ragged lengths) --
raises a ValueError that silently disables branching for the reaction.

Pure Python: no C++ library or external data files required.
"""

import types

import numpy as np
import pytest

from openmc.mgxs import GROUP_STRUCTURES
from openmc.deplete.gendf import _PythonGENDFLibrary
from openmc.deplete.helpers import IsomericBranchingHelper

BOUNDS = GROUP_STRUCTURES['CCFE-709']
N_GROUPS = len(BOUNDS) - 1


class _Tab1D:
    """Minimal Tabulated1D stand-in with .x/.y like the endf package."""
    def __init__(self, x, y):
        self.x = np.asarray(x)
        self.y = np.asarray(y)


def _make_lib(mf10_levels):
    """Bare Python-backend library with a stubbed MF=10 loader."""
    lib = _PythonGENDFLibrary.__new__(_PythonGENDFLibrary)
    lib.energy_bounds = BOUNDS
    lib.n_groups = N_GROUPS
    lib.energy_structure = 'CCFE-709'
    lib._load_mf10_data = lambda nuc, mt: ({'levels': mf10_levels}, None)
    return lib


def _threshold_level(lfs, izap, start, value_lo, value_hi):
    """Partial-range MF=10 band spanning groups [start, N_GROUPS)."""
    x = BOUNDS[start:]
    y = np.linspace(value_lo, value_hi, len(x))
    return {'LFS': lfs, 'IZAP': izap, 'sigma': _Tab1D(x, y)}


def test_production_xs_aligned_to_full_grid():
    """Partial-range band lands in its energy groups, not at index 0."""
    start = N_GROUPS - 50
    lib = _make_lib([_threshold_level(1, 27058, start, 2.0, 2.0)])

    levels = lib._get_production_xs('Co59', 16)

    assert len(levels) == 1
    lfs, izap, xs = levels[0]
    assert lfs == 1
    assert izap == 27058
    assert xs.shape == (N_GROUPS,)
    assert np.all(xs[:start] == 0.0)
    np.testing.assert_allclose(xs[start:], 2.0)


def test_threshold_branching_ratios_full_grid():
    """Runtime BR array spans the full grid with data in the threshold band."""
    start = N_GROUPS - 50
    lib = _make_lib([
        _threshold_level(0, 27058, start, 1.0, 0.5),    # ground
        _threshold_level(1, 27058, start, 0.5, 0.25),   # metastable = g/2
    ])

    br = lib.get_branching_ratios(
        'Co59', 16, target_names=['Co58', 'Co58_m1'], lfs_values=[0, 1])

    assert br.branching_ratios.shape == (2, N_GROUPS)
    # Below threshold: no production -> zero ratios
    assert np.all(br.branching_ratios[:, :start] == 0.0)
    # In-band: m/(g+m) = 1/3 everywhere since m = g/2 pointwise
    np.testing.assert_allclose(br.branching_ratios[0, start:], 2.0 / 3.0)
    np.testing.assert_allclose(br.branching_ratios[1, start:], 1.0 / 3.0)


def test_ragged_thresholds_do_not_raise():
    """Different ground/metastable thresholds must not disable branching."""
    g_start = N_GROUPS - 60
    m_start = N_GROUPS - 50
    lib = _make_lib([
        _threshold_level(0, 27058, g_start, 1.0, 1.0),
        _threshold_level(1, 27058, m_start, 1.0, 1.0),
    ])

    br = lib.get_branching_ratios(
        'Co59', 16, target_names=['Co58', 'Co58_m1'], lfs_values=[0, 1])

    assert br.branching_ratios.shape == (2, N_GROUPS)
    # Between the thresholds only the ground band produces
    np.testing.assert_allclose(br.branching_ratios[0, g_start:m_start], 1.0)
    np.testing.assert_allclose(br.branching_ratios[1, g_start:m_start], 0.0)
    # Above both thresholds: equal XS -> 50/50
    np.testing.assert_allclose(br.branching_ratios[0, m_start:], 0.5)
    np.testing.assert_allclose(br.branching_ratios[1, m_start:], 0.5)


def test_weighting_helper_threshold_no_index_error():
    """End-to-end through _calculate_weighted: no IndexError, correct BR."""
    start = N_GROUPS - 50
    lib = _make_lib([
        _threshold_level(0, 27058, start, 1.0, 0.5),
        _threshold_level(1, 27058, start, 0.5, 0.25),
    ])
    br = lib.get_branching_ratios(
        'Co59', 16, target_names=['Co58', 'Co58_m1'], lfs_values=[0, 1])

    # Fake MF=3 sigma: zero below threshold, 1 barn in-band
    sigma_g = np.zeros(N_GROUPS)
    sigma_g[start:] = 1.0
    lib.get_xs = lambda nuc, mt, e=None, **kwargs: sigma_g

    helper = IsomericBranchingHelper.__new__(IsomericBranchingHelper)
    helper.gendf_library = lib

    data = {
        'energies': br.energies,
        'targets': list(br.products),
        'branching_ratios': {p: br.branching_ratios[i]
                             for i, p in enumerate(br.products)},
    }
    weighted = helper._calculate_weighted(
        data, np.ones(N_GROUPS), BOUNDS, 'Co59', '(n,2n)')

    assert np.isclose(sum(weighted.values()), 1.0)
    assert np.isclose(weighted['Co58'], 2.0 / 3.0)
    assert np.isclose(weighted['Co58_m1'], 1.0 / 3.0)


def test_branching_failure_warns_not_silent():
    """A failed GENDF branching lookup warns instead of silently disabling."""
    helper = IsomericBranchingHelper.__new__(IsomericBranchingHelper)
    helper._branching_cache = {}
    helper.chain = types.SimpleNamespace(
        isomeric_branching_embedded=None,
        isomeric_branching_targets={'Co59': {'(n,2n)': ['Co58', 'Co58_m1']}},
        isomeric_branching_lfs={'Co59': {'(n,2n)': [0, 1]}},
    )

    def _raise(*args, **kwargs):
        raise ValueError("bad MF=10 data")
    helper.gendf_library = types.SimpleNamespace(get_branching_ratios=_raise)

    with pytest.warns(UserWarning,
                      match=r"Isomeric branching disabled for Co59"):
        result = helper._get_branching_data('Co59', '(n,2n)')
    assert result is None


def test_full_range_band_unchanged():
    """Full-range MF=10 (e.g. (n,gamma)) keeps its original behavior."""
    lib = _make_lib([
        {'LFS': 0, 'IZAP': 47110, 'sigma': _Tab1D(BOUNDS, np.full(len(BOUNDS), 3.0))},
        {'LFS': 1, 'IZAP': 47110, 'sigma': _Tab1D(BOUNDS, np.full(len(BOUNDS), 1.0))},
    ])

    br = lib.get_branching_ratios(
        'Ag109', 102, target_names=['Ag110', 'Ag110_m1'], lfs_values=[0, 1])

    assert br.branching_ratios.shape == (2, N_GROUPS)
    np.testing.assert_allclose(br.branching_ratios[0], 0.75)
    np.testing.assert_allclose(br.branching_ratios[1], 0.25)
