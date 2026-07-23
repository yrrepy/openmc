"""MF=10-only Σ-fallback in the Python GENDF backend (M2).

EAF-2010 stores isomer-producing reactions (e.g. Al27(n,a), (n,2n)) with no
MF=3 section -- only MF=10 per-final-state partials. ``get_xs`` / ``get_all_xs``
fall back to Σ(MF=10 partials) as the total for requested MTs missing from
MF=3. Synthetic MF=3 + MF=10 sections on a 3-group grid; no real data files.
"""
import types

import numpy as np
import pytest

from openmc.deplete.gendf import _PythonGENDFLibrary

from .gendf_testing import Tab1D

BOUNDS = np.array([1.0, 1e3, 1e6, 1e9])   # 3-group grid


def _mf3(y):
    """Full-range MF=3 section dict on the 3-group grid."""
    return {'sigma': Tab1D(BOUNDS, np.asarray(y, dtype=float))}


def _level(lfs, izap, y):
    """One full-range MF=10 level on the 3-group grid."""
    return {'LFS': lfs, 'IZAP': izap, 'sigma': Tab1D(BOUNDS, np.asarray(y, float))}


def _make_lib(mf3=None, mf10=None):
    """Bare ``_PythonGENDFLibrary`` with stubbed ``_load_material`` / MF=10 loader.

    ``mf3``  : ``{mt: y}`` MF=3 sections present on the material.
    ``mf10`` : ``{mt: [levels]}`` MF=10 levels keyed by MT (``None`` otherwise).
    """
    lib = _PythonGENDFLibrary.__new__(_PythonGENDFLibrary)
    lib.energy_bounds = BOUNDS
    lib.n_groups = len(BOUNDS) - 1
    lib.energy_structure = 'test-3g'
    lib._energy_validated = True
    section_data = {(3, mt): _mf3(y) for mt, y in (mf3 or {}).items()}
    lib._load_material = lambda nuc, require_full_parser=False: \
        types.SimpleNamespace(section_data=section_data)
    mf10 = mf10 or {}
    lib._load_mf10_data = lambda nuc, mt: \
        (({'levels': mf10[mt]}, None) if mt in mf10 else None)
    return lib


def test_mf10_fallback_sums_partials():
    """get_xs / get_all_xs serve Σ(MF=10 partials); IZAP=0 skipped; MF=3 kept."""
    lib = _make_lib(
        mf3={102: [0.5, 0.4, 0.3, 0.0]},
        mf10={107: [_level(0, 110240, [1.0, 2.0, 3.0, 0.0]),
                    _level(1, 110240, [0.1, 0.2, 0.3, 0.0]),
                    _level(0, 0,      [9.0, 9.0, 9.0, 0.0])]})  # IZAP=0 -> skip

    res = lib.get_all_xs('X', mts=[102, 107])
    assert set(res) == {102, 107}
    np.testing.assert_array_equal(res[102], [0.5, 0.4, 0.3])   # MF=3 unchanged
    np.testing.assert_array_equal(res[107], [1.1, 2.2, 3.3])   # Σ MF=10

    np.testing.assert_array_equal(lib.get_xs('X', 107), [1.1, 2.2, 3.3])
    np.testing.assert_array_equal(lib.get_xs('X', 102), [0.5, 0.4, 0.3])

    # mts=None keeps the MF=3-only behavior (no fallback for un-requested MTs)
    assert set(lib.get_all_xs('X')) == {102}


def test_mf10_fallback_missing_everywhere():
    """A MT absent from both MF=3 and MF=10: KeyError (get_xs), omitted (all)."""
    lib = _make_lib(mf3={102: [0.5, 0.4, 0.3, 0.0]}, mf10={})
    with pytest.raises(KeyError):
        lib.get_xs('X', 999)
    assert lib.get_all_xs('X', mts=[999]) == {}


def test_mf10_fallback_load_failure_degrades():
    """A full-parser load crash degrades to today's behavior, not a new error."""
    lib = _make_lib(mf3={102: [0.5, 0.4, 0.3, 0.0]},
                    mf10={107: [_level(0, 110240, [1.0, 2.0, 3.0, 0.0])]})

    def boom(nuc, mt):
        raise RuntimeError("Failed to load GENDF file")   # mimics int_endf('')
    lib._load_mf10_data = boom

    assert set(lib.get_all_xs('X', mts=[102, 107])) == {102}   # 107 omitted
    with pytest.raises(KeyError):
        lib.get_xs('X', 107)
