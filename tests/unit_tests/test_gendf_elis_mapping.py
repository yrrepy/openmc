"""Unit tests for ELIS-based isomeric state mapping in GENDFLibrary.

This module tests the ELIS (excitation energy) matching functionality that maps
GENDF MF=10 metastable products to OpenMC ``_m{n}`` naming using decay library data.

Key features tested:
- DecayState dataclass
- elis_match() tolerance function
- parse_decay_isomeric_levels() for directory and single-file formats
- lookup_liso() for ELIS-based LISO lookup
- get_branching_ratios() with ELIS mapping

.. versionadded:: 0.15.3
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from openmc.deplete.gendf import (
    DecayState,
    elis_match,
    parse_decay_isomeric_levels,
    lookup_liso,
    ELIS_RTOL,
    ELIS_ATOL
)


# =============================================================================
# Tests for DecayState dataclass
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
# Tests for elis_match() tolerance function
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
# Tests for lookup_liso() function
# =============================================================================

@pytest.fixture
def ir192_decay_lookup():
    """Create a mock decay lookup for Ir-192 states."""
    return {
        (77, 192): [
            DecayState(z=77, a=192, elis=0.0, liso=0),       # Ground
            DecayState(z=77, a=192, elis=56720.0, liso=1),   # m1
            DecayState(z=77, a=192, elis=168140.0, liso=2),  # m2
        ]
    }


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
# Tests for parse_decay_isomeric_levels()
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


# =============================================================================
# Integration tests (require real data)
# =============================================================================

@pytest.fixture
def jeff33_decay_path():
    """Get path to JEFF33 decay library."""
    path = Path('/home/perry/NukeData/Activation/FISPACT/JEFF33data/decay/')
    if not path.exists():
        pytest.skip("JEFF33 decay data not available")
    return path


@pytest.fixture
def jeff33_gendf_path():
    """Get path to JEFF33 GENDF library."""
    path = Path('/home/perry/NukeData/Activation/FISPACT/JEFF33data/jeff33-n/gxs-709/')
    if not path.exists():
        pytest.skip("JEFF33 GENDF data not available")
    return path


@pytest.fixture
def ukdd12_path():
    """Get path to UKDD12 decay file."""
    path = Path('/home/perry/NukeData/Activation/ukdd-12_decay.dat')
    if not path.exists():
        pytest.skip("UKDD12 decay data not available")
    return path


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
    from openmc.deplete.gendf import GENDFLibrary

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
    from openmc.deplete.gendf import GENDFLibrary

    lib = GENDFLibrary(jeff33_gendf_path)

    # Without decay_file, decay_lookup is None
    assert lib.decay_lookup is None


def test_library_with_decay_file_works(jeff33_gendf_path, jeff33_decay_path):
    """Test that GENDFLibrary works with decay_file."""
    from openmc.deplete.gendf import GENDFLibrary

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
# Tests for IsomericBranching to_dict/from_dict
# =============================================================================

def test_isomeric_branching_to_dict():
    """Test IsomericBranching.to_dict() serialization."""
    from openmc.deplete.gendf import IsomericBranching

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
    from openmc.deplete.gendf import IsomericBranching

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
    from openmc.deplete.gendf import IsomericBranching

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
