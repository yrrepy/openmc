"""MF=3 / MF=10 interaction in the Python GENDF backend.

Two shapes, both with synthetic sections on a 3-group grid (no data files):

* MF=10-only Σ-fallback (M2): EAF-2010 stores isomer-producing reactions (e.g.
  Al27(n,a), (n,2n)) with no MF=3 section -- only MF=10 per-final-state
  partials, so ``get_xs`` / ``get_all_xs`` serve Σ(MF=10 partials) as the total.
* Ground-absent MF=10 repair (policy 3(a)): a radioactive-products-only file
  omits a (quasi-)stable ground (e.g. JEFF-4.0 In115 MT=4), so the patcher
  synthesizes it from the MF=3 total as the clamped remainder.

A third section covers the hybrid ELIS-LFS_order mapping mode on the same
synthetic machinery: ELIS matching first, positional fallback for what it
cannot identify, and the placeholder-LFS (40/99) always-bind rule.
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


def _make_lib(mf3=None, mf10=None, decay_lookup=None, mapping_mode='elis'):
    """Bare ``_PythonGENDFLibrary`` with stubbed ``_load_material`` / MF=10 loader.

    ``mf3``  : ``{mt: y}`` MF=3 sections present on the material.
    ``mf10`` : ``{mt: [levels]}`` MF=10 levels keyed by MT (``None`` otherwise).
    ``decay_lookup`` : enables the patcher (ELIS-mapping) lane.
    ``mapping_mode`` : 'elis', 'lfs_order' or the hybrid 'elis_lfs_order'.
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
    lib._mapping_mode = mapping_mode
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

    Group 1 has sigma_meta (0.5) > sigma_MF3 (0.2), so the isomer takes the
    whole channel there. The ratios alone cannot show the clamp (a negative
    ground is re-clamped downstream in ``_build_branching_result``); the repair
    record's clamped-point count is what makes it observable.
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
    rec = lib.ground_repaired[0]
    assert (rec['clamped_points'], rec['total_points']) == (1, 4)


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


# =============================================================================
# Hybrid ELIS-LFS_order mapping (mapping_mode='elis_lfs_order')
# =============================================================================

# Bi212 (UKDD-12): m1 at 250 keV, m2 at 3.93 MeV. TENDL-2017b writes the Bi212
# levels as LFS=5 (ELFS = 250 keV) and LFS=12 (ELFS = 1.91 MeV) -- for LFS=12
# the NEAREST decay state is m1, 1.66 MeV away, so ELIS refuses it.
BI212_DECAY = {(83, 212): [DecayState(z=83, a=212, elis=0.0, liso=0,
                                      half_life=3632.4),
                           DecayState(z=83, a=212, elis=250000.0, liso=1,
                                      half_life=1500.0),
                           DecayState(z=83, a=212, elis=3930000.0, liso=2,
                                      half_life=540.0)]}
BI212_GROUND = _level(0, 83212, [1.0, 1.0, 1.0, 0.0], QM=0.0, QI=0.0)


def _meta(lfs, elfs, y=(0.2, 0.2, 0.2, 0.0), izap=83212):
    """One Bi212 metastable level: ELFS = QM - QI with QM = 0 (TENDL shape)."""
    return _level(lfs, izap, list(y), QM=0.0, QI=-float(elfs))


def test_hybrid_mixes_elis_match_and_positional_fallback():
    """One reaction, both phases -- and an order crossing is allowed.

    LFS=3 lands on m2 by energy (rank 1 -> LISO 2); LFS=9 is 1.66 MeV from its
    nearest state, so it re-derives positionally onto the one metastable left.
    ELIS wins outright, even where the pairing inverts the level order.
    """
    lib = _make_lib(mf10={102: [BI212_GROUND, _meta(3, 3930000.0),
                                _meta(9, 1910000.0)]},
                    decay_lookup=BI212_DECAY, mapping_mode='elis_lfs_order')

    br = lib.get_branching_ratios('Bi211', 102)

    assert br.products == ['Bi212', 'Bi212_m2', 'Bi212_m1']
    assert br.lfs_mapping == {'Bi212_m2': 3, 'Bi212_m1': 9}
    assert br.elis_mapping['Bi212_m2']['method'] == 'elis'
    fb = br.elis_mapping['Bi212_m1']
    assert (fb['method'], fb['fallback_reason']) == ('lfs_order_fallback',
                                                     'elis_tol_exceeded')
    assert (fb['position'], fb['liso']) == (2, 1)   # rank 2 -> m1: crossing


def test_hybrid_fallback_skips_the_liso_elis_claimed():
    """Bi212 trap: the fallback must not land on a state Phase 1 already took.

    LFS=5 binds m1 by energy. LFS=12's nearest decay state is that same m1, so
    a naive nearest-state or independent positional pass would collide there;
    collision-aware Phase 2 leaves m1 alone and takes m2.
    """
    lib = _make_lib(mf10={102: [BI212_GROUND, _meta(5, 250000.0),
                                _meta(12, 1910000.0)]},
                    decay_lookup=BI212_DECAY, mapping_mode='elis_lfs_order')

    br = lib.get_branching_ratios('Bi211', 102)

    assert br.lfs_mapping == {'Bi212_m1': 5, 'Bi212_m2': 12}
    assert br.elis_mapping['Bi212_m1']['method'] == 'elis'
    fb = br.elis_mapping['Bi212_m2']
    assert (fb['method'], fb['liso']) == ('lfs_order_fallback', 2)
    assert fb['phase1_claimed_lisos'] == [1]
    # The refused ELIS candidate WAS the claimed m1 -- the collision avoided.
    routed = [e for e in lib._processing_errors if e['type'] == 'hybrid_fallback']
    assert [(e['lfs'], e['nearest_liso'], e['fallback_reason']) for e in routed] \
        == [(12, 1, 'elis_tol_exceeded')]


# In122 (UKDD-12): m1 and m2 both sit at 200 keV, so no energy can tell them
# apart. TENDL-2017b writes LFS=1 (ELFS = 40 keV) and LFS=5 (ELFS = 290 keV).
IN122_DECAY = {(49, 122): [DecayState(z=49, a=122, elis=0.0, liso=0,
                                      half_life=1.5),
                           DecayState(z=49, a=122, elis=200000.0, liso=1,
                                      half_life=10.8),
                           DecayState(z=49, a=122, elis=200000.0, liso=2,
                                      half_life=10.8)]}


def _in122_lib(mode):
    return _make_lib(
        mf10={102: [_level(0, 49122, [1.0, 1.0, 1.0, 0.0], QM=0.0, QI=0.0),
                    _meta(1, 40000.0, izap=49122),
                    _meta(5, 290000.0, izap=49122)]},
        decay_lookup=IN122_DECAY, mapping_mode=mode)


def test_hybrid_abstains_when_two_decay_states_are_degenerate():
    """In122: an ambiguous ELIS match is no evidence, so the hybrid abstains.

    Pure ELIS mode accepts LFS=5 -> m1 on a coin toss (m2 is just as close and
    the ratios then renormalize over that single survivor). The hybrid refuses
    the match and re-derives both levels positionally instead.
    """
    br = _in122_lib('elis_lfs_order').get_branching_ratios('In122', 102)

    assert br.lfs_mapping == {'In122_m1': 1, 'In122_m2': 5}
    assert [br.elis_mapping[p]['method'] for p in ('In122_m1', 'In122_m2')] \
        == ['lfs_order_fallback', 'lfs_order_fallback']
    assert br.elis_mapping['In122_m2']['fallback_reason'] == 'elis_ambiguous'
    assert br.elis_mapping['In122_m1']['fallback_reason'] == 'elis_tol_exceeded'

    # Contrast: the same data in pure ELIS mode still takes the arbitrary m1
    # and drops LFS=1 entirely.
    elis = _in122_lib('elis').get_branching_ratios('In122', 102)
    assert elis.lfs_mapping == {'In122_m1': 5}


# Hf178: m1 at 1147.4 keV, m2 at 2446.1 keV (JEFF-4.0 blank-QI shape).
HF178_DECAY = {(72, 178): [DecayState(z=72, a=178, elis=0.0, liso=0),
                           DecayState(z=72, a=178, elis=1147420.0, liso=1,
                                      half_life=4.0),
                           DecayState(z=72, a=178, elis=2446090.0, liso=2,
                                      half_life=9.8e8)]}
HF178_GROUND = _level(0, 72178, [1.0, 1.0, 1.0, 0.0], QM=0.0, QI=0.0)
# Placeholder LFS=40 = "unidentified excited state". QI=0 with QM>0 is the
# blank-QI disguise: ELFS looks like QM (2.4 MeV, close enough to m2 to pass
# tolerance) but the energy is in fact UNKNOWN, so it must never be evidence.
HF178_PLACEHOLDER = _level(40, 72178, [0.1, 0.1, 0.1, 0.0],
                           QM=2400000.0, QI=0.0)


def test_hybrid_placeholder_lfs_binds_last_and_never_displaces():
    """Placeholder LFS 40/99: no rank, no energy claim, bound by elimination.

    First shape: the blank-QI placeholder skips Phase 1 (its ELFS would have
    passed for m2) and binds the lowest unclaimed metastable afterwards.
    Second shape: a real level needs that same m2 positionally and gets it --
    the placeholder is left report-only. Neither shape may emit an _m40 name.
    """
    lib = _make_lib(mf10={102: [HF178_GROUND, _meta(1, 1147420.0, izap=72178),
                                HF178_PLACEHOLDER]},
                    decay_lookup=HF178_DECAY, mapping_mode='elis_lfs_order')
    with pytest.warns(UserWarning, match='PLACEHOLDER_LFS_BOUND'):
        br = lib.get_branching_ratios('Hf177', 102)

    assert br.lfs_mapping == {'Hf178_m1': 1, 'Hf178_m2': 40}
    bound = br.elis_mapping['Hf178_m2']
    assert (bound['method'], bound['fallback_reason']) == ('placeholder_bound',
                                                           'energy_unknown')
    assert bound['position'] is None            # never ranks among real levels

    # A real level outranks the placeholder for the last free state.
    crowded = _make_lib(
        mf10={102: [HF178_GROUND, _meta(1, 1147420.0, izap=72178),
                    _meta(3, 5000000.0, izap=72178), HF178_PLACEHOLDER]},
        decay_lookup=HF178_DECAY, mapping_mode='elis_lfs_order')
    with pytest.warns(UserWarning, match='PLACEHOLDER_LFS_UNMAPPED'):
        br2 = crowded.get_branching_ratios('Hf177', 102)

    assert br2.lfs_mapping == {'Hf178_m1': 1, 'Hf178_m2': 3}
    assert [e['type'] for e in crowded._processing_errors
            if e['type'].startswith('placeholder')] == ['placeholder_unmapped']
    assert not any(p.endswith(('_m40', '_m99')) for p in br2.products)


def test_hybrid_routes_missing_qm_to_fallback_elis_records_it():
    """No QM on the subsection head: ELIS = QM - QI is incomputable.

    The hybrid keeps the level and derives it positionally; pure ELIS mode
    drops it, but now records the loss instead of skipping in silence.
    """
    no_qm = _level(4, 83212, [0.2, 0.2, 0.2, 0.0], QI=-250000.0)   # no QM

    hybrid = _make_lib(mf10={102: [BI212_GROUND, no_qm]},
                       decay_lookup=BI212_DECAY,
                       mapping_mode='elis_lfs_order')
    br = hybrid.get_branching_ratios('Bi211', 102)
    assert br.lfs_mapping == {'Bi212_m1': 4}
    info = br.elis_mapping['Bi212_m1']
    assert (info['method'], info['fallback_reason']) == ('lfs_order_fallback',
                                                         'qm_absent')

    elis = _make_lib(mf10={102: [BI212_GROUND, no_qm]},
                     decay_lookup=BI212_DECAY, mapping_mode='elis')
    with pytest.warns(UserWarning, match='ELIS_INCOMPUTABLE'):
        dropped = elis.get_branching_ratios('Bi211', 102)
    assert dropped.products == ['Bi212']        # ground only, level lost
    assert [(e['type'], e['lfs'], e['omitted'])
            for e in elis._processing_errors] == [('elis_incomputable', 4, True)]
