"""Unit tests for openmc.data.isomeric ELIS/LFS product mapping (no data files)."""

import pytest

from openmc.data.isomeric import (
    DecayState, ELIS_ATOL, ELIS_RTOL, elis_match, lookup_liso, map_lfs_to_liso,
)


@pytest.fixture
def ir192_lookup():
    # Ir-192: ground + m1 (56.72 keV) + m2 (168.14 keV); LFS in the transport
    # file is {0, 3, 15} -- a level index, not the isomer ordinal.
    return {(77, 192): [
        DecayState(77, 192, 0.0, 0),
        DecayState(77, 192, 56720.0, 1),
        DecayState(77, 192, 168140.0, 2),
    ]}


def test_elis_match():
    # Reference is the decay-library value; tolerance scales with it.
    assert elis_match(56720.0, 56720.0)
    assert elis_match(70000.0, 56720.0)          # 23% off < 50%
    assert not elis_match(140000.0, 56720.0)     # way off
    # Absolute floor
    assert elis_match(100.0, 0.0, rtol=0.0, atol=150.0)
    assert not elis_match(100.0, 0.0, rtol=0.0, atol=50.0)
    # Defaults are the re-homed GENDF values
    assert ELIS_RTOL == 0.50 and ELIS_ATOL == 0.0


def test_lookup_liso_matched(ir192_lookup):
    r = lookup_liso(77, 192, 56720.0, ir192_lookup)
    assert r['status'] == 'matched' and r['liso'] == 1 and r['dk_elis'] == 56720.0
    r = lookup_liso(77, 192, 168000.0, ir192_lookup)
    assert r['status'] == 'matched' and r['liso'] == 2


def test_lookup_liso_nearest_and_no_match(ir192_lookup):
    # 300 eV is nowhere near either metastable -> outside tolerance
    r = lookup_liso(77, 192, 300.0, ir192_lookup, return_nearest=True)
    assert r['status'] == 'nearest' and r['liso'] == 1
    r = lookup_liso(77, 192, 300.0, ir192_lookup, return_nearest=False)
    assert r['status'] == 'no_match'


def test_lookup_liso_missing_and_ground_only():
    assert lookup_liso(1, 1, 0.0, {})['status'] == 'no_decay_data'
    ground_only = {(50, 120): [DecayState(50, 120, 0.0, 0)]}
    assert lookup_liso(50, 120, 0.0, ground_only)['status'] == 'no_metastables'


def test_lookup_liso_zero_elis_metastable():
    # Metastable exists but decay library carries ELIS=0 (data-quality issue)
    lookup = {(43, 102): [
        DecayState(43, 102, 0.0, 0),
        DecayState(43, 102, 0.0, 1, half_life=261.0),
    ]}
    r = lookup_liso(43, 102, 120100.0, lookup)
    assert r['status'] == 'zero_elis_only'
    # With skipping disabled the zero-ELIS metastable becomes usable
    r = lookup_liso(43, 102, 100.0, lookup, atol=1000.0,
                    skip_zero_elis_metastables=False)
    assert r['status'] == 'matched' and r['liso'] == 1


def test_map_elis_ground_and_skipping_lfs(ir192_lookup):
    partials = [
        {'lfs': 0, 'izap': 77192, 'elfs': 0.0},
        {'lfs': 3, 'izap': 77192, 'elfs': 56720.0},
        {'lfs': 15, 'izap': 77192, 'elfs': 168140.0},
    ]
    assert map_lfs_to_liso(partials, ir192_lookup, mode='elis') == {0: 0, 3: 1, 15: 2}


def test_map_lfs_order(ir192_lookup):
    partials = [
        {'lfs': 0, 'izap': 77192, 'elfs': 0.0},
        {'lfs': 3, 'izap': 77192, 'elfs': 56720.0},
        {'lfs': 15, 'izap': 77192, 'elfs': 168140.0},
    ]
    # Positions 1,2 -> LISO 1,2 (matches ELIS here)
    assert map_lfs_to_liso(partials, ir192_lookup, mode='lfs_order') == {0: 0, 3: 1, 15: 2}


def test_map_lfs_order_drops_excess():
    # Decay library knows only one metastable; a second LFS must be dropped.
    lookup = {(77, 192): [
        DecayState(77, 192, 0.0, 0),
        DecayState(77, 192, 56720.0, 1),
    ]}
    partials = [
        {'lfs': 0, 'izap': 77192, 'elfs': 0.0},
        {'lfs': 3, 'izap': 77192, 'elfs': 56720.0},
        {'lfs': 15, 'izap': 77192, 'elfs': 168140.0},
    ]
    with pytest.warns(UserWarning, match='LFS_ORDER_DROPPED'):
        assert map_lfs_to_liso(partials, lookup, mode='lfs_order') == {0: 0, 3: 1}


def test_map_elis_tol_exceeded_is_skipped(ir192_lookup):
    # A metastable partial whose ELFS matches nothing is warned and omitted.
    partials = [
        {'lfs': 0, 'izap': 77192, 'elfs': 0.0},
        {'lfs': 3, 'izap': 77192, 'elfs': 5000.0},   # far from 56720/168140
    ]
    with pytest.warns(UserWarning, match='ELIS_TOL_EXCEEDED'):
        assert map_lfs_to_liso(partials, ir192_lookup, mode='elis') == {0: 0}


def test_map_elis_duplicate_keeps_closest(ir192_lookup):
    # Two metastable LFS both match m1; the closer one wins.
    partials = [
        {'lfs': 3, 'izap': 77192, 'elfs': 56720.0},   # exact
        {'lfs': 5, 'izap': 77192, 'elfs': 60000.0},   # also within tol of m1
    ]
    with pytest.warns(UserWarning, match='DUPLICATE_MAPPING'):
        result = map_lfs_to_liso(partials, ir192_lookup, mode='elis')
    assert result == {3: 1}


def test_map_elis_no_decay_data_skipped():
    partials = [{'lfs': 1, 'izap': 93236, 'elfs': 160000.0}]
    with pytest.warns(UserWarning, match='NO_METASTABLE_DECAY_DATA'):
        assert map_lfs_to_liso(partials, {}, mode='elis') == {}


def test_map_invalid_mode(ir192_lookup):
    with pytest.raises(ValueError, match='mode'):
        map_lfs_to_liso([{'lfs': 1, 'izap': 77192, 'elfs': 1.0}],
                        ir192_lookup, mode='bogus')
