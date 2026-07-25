"""MF=3 / MF=10 interaction in the Python GENDF backend.

Two shapes, both with synthetic sections on a 3-group grid (no data files):

* MF=10-only Σ-fallback (M2): EAF-2010 stores isomer-producing reactions (e.g.
  Al27(n,a), (n,2n)) with no MF=3 section -- only MF=10 per-final-state
  partials, so ``get_xs`` / ``get_all_xs`` serve Σ(MF=10 partials) as the total.
* Ground-absent MF=10 repair (policy 3(a)): a radioactive-products-only file
  omits a (quasi-)stable ground (e.g. JEFF-4.0 In115 MT=4), so the patcher
  synthesizes it from the MF=3 total as the clamped remainder.
"""
import types

import numpy as np
import pytest

from openmc.deplete.decay_elis import ELIS_ATOL, ELIS_RTOL, DecayState
from openmc.deplete.gendf import _PythonGENDFLibrary

from .gendf_testing import Tab1D

BOUNDS = np.array([1.0, 1e3, 1e6, 1e9])   # 3-group grid


def _mf3(y):
    """Full-range MF=3 section dict on the 3-group grid."""
    return {'sigma': Tab1D(BOUNDS, np.asarray(y, dtype=float))}


def _level(lfs, izap, y, **qs):
    """One full-range MF=10 level on the 3-group grid (``QM``/``QI`` optional)."""
    return {'LFS': lfs, 'IZAP': izap,
            'sigma': Tab1D(BOUNDS, np.asarray(y, float)), **qs}


def _make_lib(mf3=None, mf10=None, decay_lookup=None):
    """Bare ``_PythonGENDFLibrary`` with stubbed ``_load_material`` / MF=10 loader.

    ``mf3``  : ``{mt: y}`` MF=3 sections present on the material.
    ``mf10`` : ``{mt: [levels]}`` MF=10 levels keyed by MT (``None`` otherwise).
    ``decay_lookup`` : enables the patcher (ELIS-mapping) lane.
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
    lib.decay_lookup = decay_lookup
    lib._mapping_mode = 'elis'
    lib._elis_rtol = ELIS_RTOL
    lib._elis_atol = ELIS_ATOL
    lib._skip_zero_elis_metastables = True
    lib._processing_errors = []
    lib._ground_repaired = []
    return lib


def test_mf10_fallback_sums_partials():
    """get_xs / get_all_xs serve Σ(MF=10 partials), anonymous levels included.

    R1-61: an IZAP=0 level is retained (keyed by LFS) and therefore contributes
    to the Σ total -- dropping it used to under-count the reaction XS.
    """
    lib = _make_lib(
        mf3={102: [0.5, 0.4, 0.3, 0.0]},
        mf10={107: [_level(0, 110240, [1.0, 2.0, 3.0, 0.0]),
                    _level(1, 110240, [0.1, 0.2, 0.3, 0.0]),
                    _level(2, 0,      [9.0, 9.0, 9.0, 0.0])]})  # IZAP=0 -> kept

    res = lib.get_all_xs('X', mts=[102, 107])
    assert set(res) == {102, 107}
    np.testing.assert_array_equal(res[102], [0.5, 0.4, 0.3])   # MF=3 unchanged
    np.testing.assert_array_equal(res[107], [10.1, 11.2, 12.3])  # Σ MF=10

    np.testing.assert_array_equal(lib.get_xs('X', 107), [10.1, 11.2, 12.3])
    np.testing.assert_array_equal(lib.get_xs('X', 102), [0.5, 0.4, 0.3])

    # mts=None keeps the MF=3-only behavior (no fallback for un-requested MTs)
    assert set(lib.get_all_xs('X')) == {102}


def test_mf10_duplicate_lfs_drops_both():
    """A repeated LFS within one MT is ambiguous: both subsections are dropped
    (never last-wins) with one warning; unaffected levels survive."""
    lib = _make_lib(
        mf3={102: [0.5, 0.4, 0.3, 0.0]},
        mf10={107: [_level(0, 110240, [1.0, 2.0, 3.0, 0.0]),
                    _level(0, 110241, [5.0, 5.0, 5.0, 0.0]),   # collides
                    _level(1, 110240, [0.1, 0.2, 0.3, 0.0])]})

    with pytest.warns(UserWarning, match='more than one subsection'):
        levels = lib._get_production_xs('X', 107)
    assert [(lfs, izap) for lfs, izap, _ in levels] == [(1, 110240)]
    np.testing.assert_array_equal(lib.get_xs('X', 107), [0.1, 0.2, 0.3])


def test_mf10_mt5_not_served():
    """MT=5 (lumped, many products per LFS) is never keyed by LFS, so it has no
    production levels and no Σ fallback (C++ parity)."""
    lib = _make_lib(mf3={102: [0.5, 0.4, 0.3, 0.0]},
                    mf10={5: [_level(0, 1, [1.0, 2.0, 3.0, 0.0]),
                              _level(0, 0, [9.0, 9.0, 9.0, 0.0])]})
    assert lib._get_production_xs('X', 5) == []
    with pytest.raises(KeyError):
        lib.get_xs('X', 5)


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


# =============================================================================
# Ground-absent MF=10 -> clamped MF=3 remainder (policy 3(a), patcher lane)
# =============================================================================

# In115 MT=4 shape: MF=10 carries only the isomer (ground In115 is stable).
IN115_DECAY = {(49, 115): [DecayState(z=49, a=115, elis=0.0, liso=0),
                           DecayState(z=49, a=115, elis=336244.0, liso=1)]}
IN115_META = _level(1, 49115, [0.25, 0.5, 0.0, 0.0], QM=0.0, QI=-336240.0)


def test_ground_absent_repaired_from_mf3():
    """Metastable-only MF=10 + a true MF=3: ground = clamped MF=3 remainder.

    Group 1 has sigma_meta (0.5) > sigma_MF3 (0.2), so the remainder clamps to 0
    and the isomer takes the whole channel -- never a negative ground ratio.
    """
    lib = _make_lib(mf3={4: [1.0, 0.2, 0.5, 0.0]}, mf10={4: [IN115_META]},
                    decay_lookup=IN115_DECAY)

    with pytest.warns(UserWarning, match='synthesized from the MF=3 total'):
        br = lib.get_branching_ratios('In115', 4)

    assert br.products == ['In115', 'In115_m1']
    assert br.lfs_mapping == {'In115_m1': 1}
    np.testing.assert_allclose(br.branching_ratios[0], [0.75, 0.0, 1.0])
    np.testing.assert_allclose(br.branching_ratios[1], [0.25, 1.0, 0.0])
    assert [(r['nuclide'], r['mt'], r['ground_product'])
            for r in lib.ground_repaired] == [('In115', 4, 'In115')]


def test_ground_remainder_uses_histogram_left_value():
    """A metastable point off the MF=3 grid takes the enclosing group's total.

    An exact-match lookup returned 0 there, making the remainder 0 and handing
    the isomer a silent BR = 1.0 at that energy.
    """
    off_grid = {'LFS': 1, 'IZAP': 49115, 'QM': 0.0, 'QI': -336240.0,
                'sigma': Tab1D(np.array([1.0, 500.0, 1e3, 1e6, 1e9]),
                               np.array([0.25, 0.25, 0.5, 0.0, 0.0]))}
    lib = _make_lib(mf3={4: [1.0, 0.2, 0.5, 0.0]}, mf10={4: [off_grid]},
                    decay_lookup=IN115_DECAY)

    with pytest.warns(UserWarning, match='synthesized from the MF=3 total'):
        br = lib.get_branching_ratios('In115', 4)

    # 500 eV sits in the first MF=3 group (total 1.0): ground = 1.0 - 0.25.
    assert list(br.energies) == [1.0, 500.0, 1e3, 1e6]
    np.testing.assert_allclose(br.branching_ratios[1],
                               [0.25, 0.25, 1.0, 0.0])


def test_ground_absent_unrepairable_is_classified_skip():
    """Anonymous levels, no MF=3, or several IZAPs: None + a recorded reason."""
    lib = _make_lib(mf10={4: [IN115_META]}, decay_lookup=IN115_DECAY)
    assert lib.get_branching_ratios('In115', 4) is None

    ambiguous = _make_lib(
        mf3={4: [1.0, 0.2, 0.5, 0.0]},
        mf10={4: [IN115_META, _level(1, 49113, [0.1, 0.1, 0.0, 0.0])]},
        decay_lookup=IN115_DECAY)
    assert ambiguous.get_branching_ratios('In115', 4) is None

    # R1-61: with an anonymous (IZAP=0) level the ground may be UNNAMED rather
    # than absent, so the named levels are a subset -- repairing would decorate
    # that subset and invert the branching. Declines even though MF=3 is there.
    anonymous = _make_lib(
        mf3={4: [1.0, 0.2, 0.5, 0.0]},
        mf10={4: [_level(0, 0, [0.5, 0.1, 0.3, 0.0]), IN115_META]},
        decay_lookup=IN115_DECAY)
    with pytest.warns(UserWarning, match='Invalid IZAP=0'):
        assert anonymous.get_branching_ratios('In115', 4) is None

    assert [u['reason'] for u in lib.unmatched_mts] == \
        ['mf10_metastable_only_no_mf3']
    assert [u['reason'] for u in ambiguous.unmatched_mts] == \
        ['mf10_ambiguous_izap']
    assert [u['reason'] for u in anonymous.unmatched_mts] == \
        ['mf10_anonymous_levels']
    assert (lib.ground_repaired == [] and ambiguous.ground_repaired == []
            and anonymous.ground_repaired == [])
