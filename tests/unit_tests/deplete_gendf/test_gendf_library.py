"""Unit tests for the Python GENDF library internals.

Merges three former modules:

* ELIS-based isomeric-state mapping (``DecayState``, ``elis_match``,
  ``parse_decay_isomeric_levels``, ``lookup_liso``, ``IsomericBranching``
  serialization) -- from ``test_gendf_elis_mapping``.
* Threshold-reaction MF=10 grid alignment in the real ``_PythonGENDFLibrary``
  backend -- from ``test_gendf_threshold_alignment``.
* Partial-LFS XML round-trip coverage -- lifted from
  ``test_phase0_phase1_validation``.

Shared mocks/factories/fixtures come from ``gendf_testing`` / ``conftest``.
"""

import tempfile
import types
from pathlib import Path

import numpy as np
import pytest

import openmc.deplete
from openmc.deplete.gendf import (
    DecayState,
    GENDFLibrary,
    IsomericBranching,
    _PythonGENDFLibrary,
    elis_match,
    lookup_liso,
    parse_decay_isomeric_levels,
)
from openmc.deplete.helpers import IsomericBranchingHelper

from .gendf_testing import (
    CCFE709_BOUNDS,
    CCFE709_NGROUPS,
    Tab1D,
    make_python_gendf_lib,
    threshold_level,
)

BOUNDS = CCFE709_BOUNDS
N_GROUPS = CCFE709_NGROUPS


# =============================================================================
# ELIS mapping -- DecayState dataclass
# =============================================================================

def test_decay_state_creation():
    """Test basic DecayState creation."""
    state = DecayState(z=77, a=192, elis=56720.0, liso=1)
    assert state.z == 77
    assert state.a == 192
    assert state.elis == 56720.0
    assert state.liso == 1
    assert state.half_life is None


def test_decay_state_with_half_life():
    """Test DecayState with half-life."""
    state = DecayState(z=77, a=192, elis=56720.0, liso=1, half_life=86.4)
    assert state.half_life == 86.4


def test_decay_state_ground():
    """Test DecayState for ground state (LISO=0, ELIS=0)."""
    state = DecayState(z=77, a=192, elis=0.0, liso=0)
    assert state.liso == 0
    assert state.elis == 0.0


# =============================================================================
# ELIS mapping -- elis_match() tolerance function
# =============================================================================

def test_elis_match_exact():
    """Test exact ELIS match."""
    assert elis_match(56720.0, 56720.0) is True


def test_elis_match_within_absolute_tolerance():
    """Test ELIS match within absolute tolerance (100 eV default)."""
    # 50 eV difference should match (within 100 eV atol)
    assert elis_match(56720.0, 56770.0) is True
    assert elis_match(56720.0, 56670.0) is True
    # 99 eV difference
    assert elis_match(56720.0, 56819.0) is True


def test_elis_match_within_relative_tolerance():
    """Test ELIS match within relative tolerance (1% default)."""
    # For ELIS=100000 eV, 1% = 1000 eV
    # With atol=100, total tolerance = 100 + 0.01*100000 = 1100 eV
    assert elis_match(100000.0, 101000.0) is True
    assert elis_match(100000.0, 99000.0) is True


def test_elis_match_outside_tolerance():
    """Test ELIS values that don't match."""
    # Ir192_m1 (56720 eV) vs Ir192_m2 (168140 eV)
    assert elis_match(56720.0, 168140.0) is False
    # Large difference
    assert elis_match(1000.0, 10000.0) is False


def test_elis_match_zero():
    """Test ELIS matching with zero values.

    With dk_elis-based tolerance formula, rtol only applies to dk_elis.
    When dk_elis is small, tolerance is correspondingly small.
    """
    # Zero vs zero (both zero, diff=0)
    assert elis_match(0.0, 0.0) is True

    # With dk_elis=0, tolerance = atol only (since rtol*0=0)
    # Default atol=0, so only exact match works
    assert elis_match(50.0, 0.0) is False  # diff=50, tol=0
    assert elis_match(0.0, 0.0) is True    # diff=0, tol=0

    # With non-zero dk_elis, tolerance = rtol * dk_elis
    # For dk_elis=100, rtol=0.50 -> tolerance=50
    assert elis_match(0.0, 100.0) is False    # diff=100, tol=50 -> FAIL
    assert elis_match(60.0, 100.0) is True    # diff=40, tol=50 -> OK
    assert elis_match(140.0, 100.0) is True   # diff=40, tol=50 -> OK
    assert elis_match(200.0, 100.0) is False  # diff=100, tol=50 -> FAIL


def test_elis_match_custom_tolerances():
    """Test ELIS matching with custom tolerances."""
    # Tighter tolerance
    assert elis_match(56720.0, 56770.0, rtol=0.0, atol=10.0) is False
    assert elis_match(56720.0, 56725.0, rtol=0.0, atol=10.0) is True

    # Looser tolerance
    assert elis_match(56720.0, 60000.0, rtol=0.1, atol=100.0) is True


# =============================================================================
# ELIS mapping -- lookup_liso() function
# =============================================================================

def test_lookup_liso_exact_match(ir192_decay_lookup):
    """Test LISO lookup with exact ELIS match."""
    result = lookup_liso(77, 192, 56720.0, ir192_decay_lookup)
    assert result['status'] == 'matched'
    assert result['liso'] == 1
    assert result['dk_elis'] == 56720.0

    result = lookup_liso(77, 192, 168140.0, ir192_decay_lookup)
    assert result['status'] == 'matched'
    assert result['liso'] == 2
    assert result['dk_elis'] == 168140.0


def test_lookup_liso_tolerance_match(ir192_decay_lookup):
    """Test LISO lookup with tolerance matching."""
    # Slight deviation within tolerance
    result = lookup_liso(77, 192, 56750.0, ir192_decay_lookup)
    assert result['status'] == 'matched'
    assert result['liso'] == 1
    assert result['dk_elis'] == 56720.0  # DK_ELIS (actual value in lookup)

    result = lookup_liso(77, 192, 168100.0, ir192_decay_lookup)
    assert result['status'] == 'matched'
    assert result['liso'] == 2
    assert result['dk_elis'] == 168140.0  # DK_ELIS (actual value in lookup)


def test_lookup_liso_no_match(ir192_decay_lookup):
    """Test LISO lookup with no matching state."""
    # Invalid ELIS (doesn't match any state) - use return_nearest=False
    # With return_nearest=False, it returns status='no_match'
    result = lookup_liso(77, 192, 100000.0, ir192_decay_lookup, return_nearest=False)
    assert result['status'] == 'no_match'


def test_lookup_liso_missing_nuclide(ir192_decay_lookup):
    """Test LISO lookup for nuclide not in lookup table."""
    result = lookup_liso(78, 195, 56720.0, ir192_decay_lookup)
    assert result['status'] == 'no_decay_data'


def test_lookup_liso_ground_state_not_returned(ir192_decay_lookup):
    """Test that ground state (LISO=0) is never returned."""
    # Even if ELIS=0 matches ground state, lookup_liso only searches metastables.
    # With return_nearest=False, returns status='no_match' when no metastable matches.
    result = lookup_liso(77, 192, 0.0, ir192_decay_lookup, return_nearest=False)
    assert result['status'] == 'no_match'


def test_lookup_liso_multiple_states_best_match():
    """Test that best match (smallest diff) is returned when multiple match."""
    # Create states with overlapping tolerances
    lookup = {
        (77, 192): [
            DecayState(z=77, a=192, elis=0.0, liso=0),
            DecayState(z=77, a=192, elis=50000.0, liso=1),
            DecayState(z=77, a=192, elis=50500.0, liso=2),
        ]
    }
    # 50200 is within tolerance of both m1 (200 diff) and m2 (300 diff)
    # Should return m1 as best match
    result = lookup_liso(77, 192, 50200.0, lookup, rtol=0.01, atol=500.0)
    assert result['status'] == 'matched'
    assert result['liso'] == 1  # Best match
    assert result['dk_elis'] == 50000.0


def test_lookup_liso_zero_elis_only():
    """Test that zero-ELIS metastables are flagged as data quality issue."""
    # Create lookup with zero-ELIS metastable (like Tc102 in UKDD-12)
    lookup = {
        (43, 102): [
            DecayState(z=43, a=102, elis=0.0, liso=0),     # Ground
            DecayState(z=43, a=102, elis=0.0, liso=1, half_life=261.0),  # m1 with ELIS=0!
        ]
    }
    # Should return 'zero_elis_only' status
    result = lookup_liso(43, 102, 120100.0, lookup)
    assert result['status'] == 'zero_elis_only'
    assert 'skipped_states' in result
    assert len(result['skipped_states']) == 1
    assert result['skipped_states'][0][0] == 1  # liso
    assert result['skipped_states'][0][2] == 261.0  # half_life


# =============================================================================
# ELIS mapping -- parse_decay_isomeric_levels()
# =============================================================================

def test_parse_decay_nonexistent_path():
    """Test that FileNotFoundError is raised for nonexistent path."""
    with pytest.raises(FileNotFoundError):
        parse_decay_isomeric_levels('/nonexistent/path')


def test_parse_decay_empty_directory(tmp_path):
    """Test parsing empty directory returns empty dict."""
    result = parse_decay_isomeric_levels(tmp_path)
    assert result == {}


def test_parse_decay_directory_format_detection(tmp_path):
    """Test that directory path triggers directory parsing."""
    # Just verify it doesn't crash on empty directory
    result = parse_decay_isomeric_levels(tmp_path)
    assert isinstance(result, dict)


def test_file_index_collision_raises(tmp_path):
    """Two files normalizing to the same nuclide name raise, not silently overwrite (R1-9)."""
    (tmp_path / 'Al027g.asc').write_text('')
    (tmp_path / 'Al27g.asc').write_text('')
    lib = _PythonGENDFLibrary.__new__(_PythonGENDFLibrary)
    lib.library_path = tmp_path
    with pytest.raises(ValueError, match="two files"):
        lib._build_file_index()


# =============================================================================
# ELIS mapping -- integration tests (real data; skip when unavailable)
# =============================================================================

def test_parse_decay_directory_jeff33(jeff33_decay_path):
    """Test parsing JEFF33 decay directory."""
    decay_lookup = parse_decay_isomeric_levels(jeff33_decay_path)

    # Should have many nuclides
    assert len(decay_lookup) > 100

    # Check Ir-192 specifically (known to have multiple metastables)
    ir192_states = decay_lookup.get((77, 192), [])
    assert len(ir192_states) >= 2  # At least ground + m1

    # Check that we have both metastables
    lisos = [s.liso for s in ir192_states]
    assert 0 in lisos  # Ground
    assert 1 in lisos  # m1


def test_parse_single_file_ukdd12(ukdd12_path):
    """Test parsing UKDD12 single-file decay library.

    Tests the single-file concatenated ENDF format parser which handles
    JEFF-4.0, JEFF311RDD, UKDD12, and EAF2010-all.txt formats.
    """
    decay_lookup = parse_decay_isomeric_levels(ukdd12_path)

    # Should have many nuclides
    assert len(decay_lookup) > 100

    # Check that we got reasonable data
    total_states = sum(len(states) for states in decay_lookup.values())
    assert total_states > 200


def test_gendf_library_with_elis_mapping(jeff33_gendf_path, jeff33_decay_path):
    """Test GENDFLibrary with ELIS-based mapping."""
    # Create library with ELIS mapping
    lib = GENDFLibrary(
        jeff33_gendf_path,
        decay_file=jeff33_decay_path,
    )

    # Verify decay_lookup is populated
    assert lib.decay_lookup is not None
    assert len(lib.decay_lookup) > 0


def test_library_without_decay_file(jeff33_gendf_path):
    """Test that GENDFLibrary works without decay_file but has no decay_lookup."""
    lib = GENDFLibrary(jeff33_gendf_path)

    # Without decay_file, decay_lookup is None
    assert lib.decay_lookup is None


def test_library_with_decay_file_works(jeff33_gendf_path, jeff33_decay_path):
    """Test that GENDFLibrary works with decay_file."""
    lib = GENDFLibrary(
        jeff33_gendf_path,
        decay_file=jeff33_decay_path,
    )

    # decay_lookup should be populated
    assert lib.decay_lookup is not None
    assert len(lib.decay_lookup) > 0

    # XS retrieval should work
    xs = lib.get_xs('Ir191', 102, lib.energy_bounds)
    assert xs is not None


# =============================================================================
# ELIS mapping -- IsomericBranching to_dict/from_dict
# =============================================================================

def test_isomeric_branching_to_dict():
    """Test IsomericBranching.to_dict() serialization."""
    branching = IsomericBranching(
        energies=np.array([1e5, 1e6, 1e7]),
        products=['Ir192', 'Ir192_m1'],
        branching_ratios=np.array([[0.7, 0.6, 0.5], [0.3, 0.4, 0.5]]),
        parent_nuclide='Ir191',
        reaction='(n,gamma)',
        mt=102
    )

    d = branching.to_dict()

    assert d['energies'] == [1e5, 1e6, 1e7]
    assert d['products'] == ['Ir192', 'Ir192_m1']
    assert len(d['branching_ratios']) == 2
    assert d['parent_nuclide'] == 'Ir191'
    assert d['reaction'] == '(n,gamma)'
    assert d['mt'] == 102


def test_isomeric_branching_from_dict():
    """Test IsomericBranching.from_dict() deserialization."""
    d = {
        'energies': [1e5, 1e6, 1e7],
        'products': ['Ir192', 'Ir192_m1'],
        'branching_ratios': [[0.7, 0.6, 0.5], [0.3, 0.4, 0.5]],
        'parent_nuclide': 'Ir191',
        'reaction': '(n,gamma)',
        'mt': 102
    }

    branching = IsomericBranching.from_dict(d)

    assert np.allclose(branching.energies, [1e5, 1e6, 1e7])
    assert branching.products == ['Ir192', 'Ir192_m1']
    assert branching.branching_ratios.shape == (2, 3)
    assert branching.parent_nuclide == 'Ir191'
    assert branching.reaction == '(n,gamma)'
    assert branching.mt == 102


def test_isomeric_branching_roundtrip():
    """Test IsomericBranching roundtrip (to_dict -> from_dict)."""
    original = IsomericBranching(
        energies=np.array([1e5, 1e6, 1e7]),
        products=['Ag110', 'Ag110_m1'],
        branching_ratios=np.array([[0.95, 0.90, 0.85], [0.05, 0.10, 0.15]]),
        parent_nuclide='Ag109',
        reaction='(n,gamma)',
        mt=102
    )

    # Roundtrip
    d = original.to_dict()
    reconstructed = IsomericBranching.from_dict(d)

    # Verify equivalence
    assert np.allclose(reconstructed.energies, original.energies)
    assert reconstructed.products == original.products
    assert np.allclose(reconstructed.branching_ratios, original.branching_ratios)
    assert reconstructed.parent_nuclide == original.parent_nuclide
    assert reconstructed.reaction == original.reaction
    assert reconstructed.mt == original.mt


# =============================================================================
# Grid alignment -- threshold MF=10 on the real _PythonGENDFLibrary backend
# =============================================================================

def test_production_xs_aligned_to_full_grid():
    """Partial-range band lands in its energy groups, not at index 0."""
    start = N_GROUPS - 50
    lib = make_python_gendf_lib([threshold_level(1, 27058, start, 2.0, 2.0)])

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
    lib = make_python_gendf_lib([
        threshold_level(0, 27058, start, 1.0, 0.5),    # ground
        threshold_level(1, 27058, start, 0.5, 0.25),   # metastable = g/2
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
    lib = make_python_gendf_lib([
        threshold_level(0, 27058, g_start, 1.0, 1.0),
        threshold_level(1, 27058, m_start, 1.0, 1.0),
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
    lib = make_python_gendf_lib([
        threshold_level(0, 27058, start, 1.0, 0.5),
        threshold_level(1, 27058, start, 0.5, 0.25),
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


def test_process_library_surfaces_unexpected_errors():
    """Unexpected per-nuclide errors raise a summarizing RuntimeError; expected
    NO_METASTABLE_DECAY_DATA skips are recorded silently (K13/R1-2)."""
    from openmc.deplete.gendf import _PythonGENDFLibrary

    def _make(nuclides, brancher):
        lib = _PythonGENDFLibrary.__new__(_PythonGENDFLibrary)
        lib.available_nuclides = lambda: nuclides
        lib.get_branching_ratios = brancher
        return lib

    # An unexpected ValueError must not vanish -- it surfaces by nuclide name.
    def _bad(nuc, mt):
        if nuc == 'BadNuc':
            raise ValueError("corrupted MF=10 data")
        return None
    with pytest.raises(RuntimeError, match=r"BadNuc"):
        _make(['GoodNuc', 'BadNuc'], _bad).process_library_for_branching(
            mt_list=[102], verbose=False)

    # An expected metastable-decay skip is recorded, not raised.
    def _skip(nuc, mt):
        raise ValueError("WARNING: NO_METASTABLE_DECAY_DATA: SkipNuc ...")
    lib = _make(['SkipNuc'], _skip)
    result = lib.process_library_for_branching(mt_list=[102], verbose=False)
    assert result == {}
    assert [e['type'] for e in lib.processing_errors] == \
        ['no_metastable_decay_data']


def test_full_range_band_unchanged():
    """Full-range MF=10 (e.g. (n,gamma)) keeps its original behavior."""
    lib = make_python_gendf_lib([
        {'LFS': 0, 'IZAP': 47110, 'sigma': Tab1D(BOUNDS, np.full(len(BOUNDS), 3.0))},
        {'LFS': 1, 'IZAP': 47110, 'sigma': Tab1D(BOUNDS, np.full(len(BOUNDS), 1.0))},
    ])

    br = lib.get_branching_ratios(
        'Ag109', 102, target_names=['Ag110', 'Ag110_m1'], lfs_values=[0, 1])

    assert br.branching_ratios.shape == (2, N_GROUPS)
    np.testing.assert_allclose(br.branching_ratios[0], 0.75)
    np.testing.assert_allclose(br.branching_ratios[1], 0.25)

    # Real _align_to_group_grid truncates a full-length (N_GROUPS+1) MF=10 band
    # to N_GROUPS, dropping the last point and preserving the rest in order.
    # Real-backend cover for the dropped inline full-range-no-op / n+1
    # truncation arithmetic tests.
    ramp = np.arange(len(BOUNDS), dtype=float)
    lib2 = make_python_gendf_lib(
        [{'LFS': 0, 'IZAP': 47110, 'sigma': Tab1D(BOUNDS, ramp)}])
    xs = lib2._get_production_xs('Ag109', 102)[0][2]
    assert xs.shape == (N_GROUPS,)
    np.testing.assert_array_equal(xs, ramp[:N_GROUPS])


# =============================================================================
# Partial-LFS XML round-trip (from test_phase0_phase1_validation)
# =============================================================================

def test_partial_lfs_coverage():
    """Chain with LFS on one reaction but not another."""
    chain = openmc.deplete.Chain()

    parent = openmc.deplete.Nuclide('Ag109')
    parent.add_reaction('(n,gamma)', 'Ag110', Q=6.8e6, branching_ratio=1.0)
    parent.add_reaction('(n,2n)', 'Ag108', Q=-9.5e6, branching_ratio=1.0)
    chain.add_nuclide(parent)

    for name in ['Ag110', 'Ag110_m1', 'Ag108', 'Ag108_m1']:
        nuc = openmc.deplete.Nuclide(name)
        nuc.half_life = 1e5
        chain.add_nuclide(nuc)

    chain.isomeric_branching_targets = {
        'Ag109': {
            '(n,gamma)': ['Ag110', 'Ag110_m1'],
            '(n,2n)': ['Ag108', 'Ag108_m1']
        }
    }
    chain.isomeric_branching_lfs = {
        'Ag109': {
            '(n,gamma)': [0, 5]
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "partial_lfs.xml"
        chain.export_to_xml(xml_path)
        reloaded = openmc.deplete.Chain.from_xml(xml_path)

        assert reloaded.isomeric_branching_lfs is not None
        assert reloaded.isomeric_branching_lfs['Ag109']['(n,gamma)'] == [0, 5]
        assert '(n,2n)' not in reloaded.isomeric_branching_lfs.get('Ag109', {})

        assert '(n,gamma)' in reloaded.isomeric_branching_targets['Ag109']
        assert '(n,2n)' in reloaded.isomeric_branching_targets['Ag109']
