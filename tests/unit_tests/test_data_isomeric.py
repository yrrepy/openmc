"""Unit tests for openmc.data.isomeric ELIS/LFS product mapping (no data files)."""

import pytest

from openmc.data.isomeric import (
    DecayState, ELIS_ATOL, ELIS_RTOL, elis_match, lookup_liso, map_lfs_to_liso,
    parse_decay_isomeric_levels,
)


def _endf_cont(fields, mat, mf, mt, ns=0):
    """Format one 80-column ENDF CONT/HEAD record from six field strings."""
    body = ''.join(f'{s:>11}' for s in fields)
    return f'{body}{mat:>4}{mf:>2}{mt:>3}{ns:>5}\n'


def _malformed_decay_tape():
    """A one-material MF=1/451 decay tape that defeats ``get_evaluations``.

    Reproduces the EASY-II/FISPACT failure mode: the tape ends with a MEND
    record (MAT=0) but omits the ENDF TEND record (MAT=-1), so the general
    reader parses a spurious material past end-of-file and raises
    ``ValueError: invalid literal for int() with base 10: ''``.  Encodes
    Z=47, A=100, ELIS=15500.5 eV, LISO=1.
    """
    mat = 4710
    lines = [
        # TPID line (skipped by the reader), non-standard columns like the real
        # decay files.
        f'{"EASY-II decay test tape":<66}{"":>4}{"":>2}{"":>3}{0:>5}\n',
        # HEAD: ZA, AWR, LRP=-1, LFI=1, NLIB=2, NMOD=0
        _endf_cont([' 4.710000+4', ' 9.900000+1', '-1', '1', '2', '0'], mat, 1, 451, 1),
        # CONT 1: ELIS, STA, LIS, LISO, 0, NFOR
        _endf_cont([' 1.550050+4', ' 1.000000+0', '1', '1', '0', '6'], mat, 1, 451, 2),
        # CONT 2: AWI, EMAX, LREL, 0, NSUB=4 (decay), NVER
        _endf_cont([' 0.000000+0', ' 0.000000+0', '0', '0', '4', '22'], mat, 1, 451, 3),
        # CONT 3: TEMP, 0, LDRV, 0, NWD=1, NXC=1
        _endf_cont([' 0.000000+0', ' 0.000000+0', '0', '0', '1', '1'], mat, 1, 451, 4),
        # NWD=1 text record
        f'{"AG-100M   DECAY":<66}{mat:>4}{1:>2}{451:>3}{5:>5}\n',
        # NXC=1 directory record (blank C1/C2, MF, MT, NC, MOD)
        _endf_cont(['', '', '1', '451', '5', '0'], mat, 1, 451, 6),
        _endf_cont(['0', '0', '0', '0', '0', '0'], mat, 1, 0, 99999),  # SEND
        _endf_cont(['0', '0', '0', '0', '0', '0'], mat, 0, 0),         # FEND
        _endf_cont(['0', '0', '0', '0', '0', '0'], 0, 0, 0),           # MEND (no TEND)
    ]
    return ''.join(lines)


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


def test_malformed_tape_defeats_get_evaluations(tmp_path):
    # Guard: the fixture really does reproduce the EOF int('') failure mode,
    # so the fallback path is genuinely exercised.
    from openmc.data.endf import get_evaluations
    tape = tmp_path / 'Ag100m'
    tape.write_text(_malformed_decay_tape())
    with pytest.raises(ValueError, match="invalid literal for int"):
        get_evaluations(tape)


def test_manual_fallback_recovers_states(tmp_path):
    # parse_decay_isomeric_levels must fall back to the fixed-column reader and
    # recover ZA/ELIS/LISO, emitting a 'manual-parse' note (counts stay visible).
    tape = tmp_path / 'Ag100m'
    tape.write_text(_malformed_decay_tape())
    with pytest.warns(UserWarning, match='manual-parse'):
        lookup = parse_decay_isomeric_levels(tape)
    assert (47, 100) in lookup
    (state,) = lookup[(47, 100)]
    assert state.z == 47 and state.a == 100
    assert state.liso == 1
    assert state.elis == pytest.approx(15500.5)
