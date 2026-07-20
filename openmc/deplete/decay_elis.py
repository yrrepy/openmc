"""ELIS-based isomeric state mapping for decay data.

This module provides functions to parse decay libraries and perform
excitation energy (ELIS) based mapping of isomeric states for use
with GENDF cross-section data in depletion calculations.

The key function is :func:`parse_decay_isomeric_levels` which builds a
lookup table from decay library data. This table is then used by
:func:`lookup_liso` to map GENDF MF=10 metastable products to the correct
OpenMC ``_m{n}`` naming based on excitation energy matching.
"""

from collections import defaultdict
from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import endf

from openmc.data.endf import py_float_endf

# Type alias for path-like objects
PathLike = Union[str, Path]

# Logger for this module (use logging.getLogger(__name__) pattern)
_logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

# Default tolerances for ELIS matching
# 50% relative tolerance handles typical evaluation differences between libraries
# while rejecting clearly wrong matches (>50% difference)
ELIS_RTOL = 0.50   # 50% relative tolerance
ELIS_ATOL = 0.0    # No absolute tolerance (rtol-only)

# Dedup store for once-per-(Z, A) ELIS ambiguity warnings.
# Reset at the start of every parse_decay_isomeric_levels() call so each
# library load gets its warnings again (a long-lived process that loads
# several libraries is not silenced after the first).
_WARNED_ELIS_AMBIGUITY: set = set()


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DecayState:
    """Represents a nuclear state from decay library data.

    This class stores information extracted from ENDF decay files (MF=1, MT=451)
    needed to identify isomeric states based on excitation energy (ELIS).

    Attributes
    ----------
    z : int
        Atomic number (protons)
    a : int
        Mass number (protons + neutrons)
    elis : float
        Excitation energy in eV (0.0 for ground state)
    liso : int
        Isomeric state number (0=ground, 1=m1, 2=m2, etc.)
    half_life : float, optional
        Half-life in seconds, if available from MF=8, MT=457

    Notes
    -----
    The ELIS (Excitation Energy of LISo state) value is the key identifier
    for matching GENDF MF=10 product levels to the correct OpenMC `_m{n}` naming.

    In ENDF decay files:
    - ELIS is stored in MF=1, MT=451 second record
    - LISO directly gives the isomeric state number (0, 1, 2, ...)

    In GENDF MF=10:
    - ELIS is calculated as QM - QI (Q-value difference)
    - LFS is an internal level flag that may not correspond to LISO

    Examples
    --------
    >>> # Ir-192 ground state
    >>> ground = DecayState(z=77, a=192, elis=0.0, liso=0)
    >>> # Ir-192m1 (first metastable, 56.7 keV)
    >>> m1 = DecayState(z=77, a=192, elis=56720.0, liso=1)
    >>> # Ir-192m2 (second metastable, 168.1 keV)
    >>> m2 = DecayState(z=77, a=192, elis=168140.0, liso=2)
    """
    z: int
    a: int
    elis: float
    liso: int
    half_life: Optional[float] = None


# ============================================================================
# ELIS Matching Functions
# ============================================================================

def elis_match(
    gendf_elis: float,
    dk_elis: float,
    rtol: float = ELIS_RTOL,
    atol: float = ELIS_ATOL
) -> bool:
    """Check if GENDF-derived ELIS matches decay library ELIS within tolerance.

    Uses decay library ELIS as the reference (ground truth), consistent with
    NumPy's ``np.isclose()`` semantics where the second argument is the reference:
    ``abs(gendf_elis - dk_elis) <= atol + rtol * abs(dk_elis)``

    This approach treats the decay library as the authoritative source for
    excitation energies, since:
    1. Decay library ELIS values are directly measured/evaluated
    2. GENDF ELIS is derived (QM - QI) and may have evaluation differences
    3. Consistent with NumPy convention for tolerance comparisons

    The default 50% relative tolerance handles typical evaluation differences
    between nuclear data libraries while rejecting clearly wrong matches.

    Parameters
    ----------
    gendf_elis : float
        Excitation energy from GENDF (QM - QI) in eV
    dk_elis : float
        Excitation energy from decay library (reference) in eV
    rtol : float, optional
        Relative tolerance applied to dk_elis (default: 0.50 = 50%)
    atol : float, optional
        Absolute tolerance in eV (default: 0.0, not used)

    Returns
    -------
    bool
        True if GENDF ELIS is within tolerance of decay library ELIS

    Notes
    -----
    The tolerance formula ``rtol * abs(dk_elis)`` means:
    - Large dk_elis values allow larger absolute differences
    - Question asked: "Is GENDF ELIS within X% of the known isomeric state?"

    This differs from the symmetric ``max()``-based formula which could
    inflate tolerance when GENDF ELIS is larger than decay library ELIS.

    Examples
    --------
    >>> # Ir192_m1: GENDF 56720 eV vs decay 56720 eV (exact)
    >>> elis_match(56720.0, 56720.0)
    True

    >>> # Within 50% of decay library value
    >>> elis_match(70000.0, 56720.0)  # diff=13280, tol=28360 -> OK
    True

    >>> # Outside 50% of decay library value
    >>> elis_match(140000.0, 81200.0)  # diff=58800, tol=40600 -> FAIL
    False
    """
    return abs(gendf_elis - dk_elis) <= atol + rtol * abs(dk_elis)


def _warn_elis_ambiguity(z, a, target_elis, assigned, other, rtol):
    """Warn once per (Z, A) when two decay levels both match the GENDF ELIS."""
    key = (z, a)
    if key in _WARNED_ELIS_AMBIGUITY:
        return
    _WARNED_ELIS_AMBIGUITY.add(key)
    warnings.warn(
        f"Ambiguous ELIS match for Z={z} A={a} (GENDF ELIS={target_elis:.1f} "
        f"eV): assigned LISO={assigned[0]} (ELIS={assigned[1]:.1f} eV), but "
        f"LISO={other[0]} (ELIS={other[1]:.1f} eV) also passes rtol={rtol}. "
        f"Possible m1/m2 mis-assignment.", UserWarning)


def lookup_liso(
    z: int,
    a: int,
    target_elis: float,
    decay_lookup: Dict[Tuple[int, int], List[DecayState]],
    rtol: float = ELIS_RTOL,
    atol: float = ELIS_ATOL,
    skip_zero_elis_metastables: bool = True,
    return_nearest: bool = True
) -> Dict[str, Any]:
    """Find LISO (isomeric state number) for given excitation energy.

    Searches the decay lookup table for a metastable state (LISO > 0) whose
    ELIS matches the target within tolerance. Always returns information about
    the best/nearest match to support logging of tolerance-exceeded cases.

    Parameters
    ----------
    z : int
        Atomic number of product nuclide
    a : int
        Mass number of product nuclide
    target_elis : float
        Excitation energy in eV (from GENDF MF=10: QM - QI)
    decay_lookup : dict
        Decay state lookup table from :func:`parse_decay_isomeric_levels`
    rtol : float, optional
        Relative tolerance for ELIS matching (default: 0.50 = 50%)
    atol : float, optional
        Absolute tolerance in eV for ELIS matching (default: 0.0 eV)
    skip_zero_elis_metastables : bool, optional
        If True (default), skip metastable states (LISO > 0) that have ELIS=0.0
        in the decay library. This is a physics constraint - metastable states
        are excited states and must have ELIS > 0. ELIS=0 for a metastable
        indicates invalid data in the decay library (e.g., 32 such cases in
        JEFF-4.0). Set to False only for debugging.
    return_nearest : bool, optional
        If True (default), always return the nearest match even when tolerance
        is exceeded, with status indicating 'nearest'. If False, return
        status='no_match' without nearest info when outside tolerance.

    Returns
    -------
    dict
        Dictionary with 'status' key and additional fields depending on status:

        - **'matched'**: Match found within tolerance
          ``{'status': 'matched', 'liso': int, 'dk_elis': float}``

        - **'nearest'**: Nearest match found, but outside tolerance (return_nearest=True)
          ``{'status': 'nearest', 'liso': int, 'dk_elis': float, 'diff_pct': float}``

        - **'no_match'**: Outside tolerance and return_nearest=False
          ``{'status': 'no_match'}``

        - **'no_decay_data'**: Nuclide (Z, A) not in decay library at all
          ``{'status': 'no_decay_data'}``

        - **'no_metastables'**: Nuclide exists but has no metastable states (only ground)
          ``{'status': 'no_metastables'}``

        - **'zero_elis_only'**: Metastable states exist but all have ELIS=0 (data quality issue)
          ``{'status': 'zero_elis_only', 'skipped_states': [(liso, elis, half_life), ...]}``

    Notes
    -----
    **Matching Logic**:

    1. Skip ground states (LISO=0) since we're matching metastables
    2. Track zero-ELIS metastables separately (data quality issue)
    3. Find the metastable state with smallest absolute ELIS difference
    4. If within tolerance, return with status='matched'
    5. If outside tolerance, return with status='nearest' (if return_nearest=True)

    If target_elis is close to 0 (< 100 eV), this likely indicates the
    ground state, which should be handled separately.

    Examples
    --------
    >>> # Ir192_m1: ELIS = 56720 eV -> LISO = 1 (within tolerance)
    >>> decay_lookup = parse_decay_isomeric_levels('/path/to/decay/')
    >>> result = lookup_liso(77, 192, 56720.0, decay_lookup)
    >>> print(result)
    {'status': 'matched', 'liso': 1, 'dk_elis': 56710.0}
    >>>
    >>> # Ga73 with ELIS mismatch (nearest but outside tolerance)
    >>> result = lookup_liso(31, 73, 300.0, decay_lookup)
    >>> print(result)
    {'status': 'nearest', 'liso': 1, 'dk_elis': 13500.0, 'diff_pct': 4400.0}
    >>>
    >>> # Tc102 with zero-ELIS metastable in UKDD-12
    >>> result = lookup_liso(43, 102, 120100.0, decay_lookup)
    >>> print(result)
    {'status': 'zero_elis_only', 'skipped_states': [(1, 0.0, 261.0)]}

    See Also
    --------
    parse_decay_isomeric_levels : Build decay lookup table
    elis_match : Tolerance check function
    """
    # Get decay states for this nuclide
    decay_states = decay_lookup.get((z, a), [])

    if not decay_states:
        return {'status': 'no_decay_data'}

    # Categorize states
    metastables_valid = []      # LISO > 0, ELIS > 0 (valid for matching)
    metastables_zero_elis = []  # LISO > 0, ELIS = 0 (data quality issue)

    for state in decay_states:
        if state.liso == 0:
            continue  # Skip ground state

        if state.elis == 0.0:
            # Track zero-ELIS metastables (data quality issue)
            metastables_zero_elis.append((state.liso, state.elis, state.half_life))
        else:
            metastables_valid.append(state)

    # Check for zero-ELIS-only case (metastables exist but all have ELIS=0)
    if not metastables_valid and metastables_zero_elis:
        if skip_zero_elis_metastables:
            return {
                'status': 'zero_elis_only',
                'skipped_states': metastables_zero_elis
            }
        # If not skipping, treat zero-ELIS metastables as valid (for debugging)
        metastables_valid = [
            DecayState(z=z, a=a, elis=0.0, liso=liso, half_life=hl)
            for liso, _, hl in metastables_zero_elis
        ]

    # No metastable states at all (only ground state exists)
    if not metastables_valid:
        return {'status': 'no_metastables'}

    # Find nearest and second-nearest matches by absolute ELIS difference
    nearest_match = None
    nearest_diff = float('inf')
    second_match = None
    second_diff = float('inf')

    for state in metastables_valid:
        diff = abs(target_elis - state.elis)
        if diff < nearest_diff:
            second_diff, second_match = nearest_diff, nearest_match
            nearest_diff = diff
            nearest_match = (state.liso, state.elis)
        elif diff < second_diff:
            second_diff = diff
            second_match = (state.liso, state.elis)

    liso, dk_elis = nearest_match

    # Check if within tolerance
    if elis_match(target_elis, dk_elis, rtol, atol):
        # Ambiguity: a second level also passes tolerance -> possible m1/m2
        # mis-assignment. Warn once per (Z, A).
        if second_match is not None and \
                elis_match(target_elis, second_match[1], rtol, atol):
            _warn_elis_ambiguity(z, a, target_elis, nearest_match,
                                 second_match, rtol)
        return {'status': 'matched', 'liso': liso, 'dk_elis': dk_elis}
    else:
        # Outside tolerance
        if return_nearest:
            # Calculate percentage difference
            if dk_elis != 0:
                diff_pct = abs(target_elis - dk_elis) / abs(dk_elis) * 100
            else:
                diff_pct = float('inf')
            return {
                'status': 'nearest',
                'liso': liso,
                'dk_elis': dk_elis,
                'diff_pct': diff_pct
            }
        else:
            return {'status': 'no_match'}


# ============================================================================
# Decay Library Parsing
# ============================================================================

def parse_decay_isomeric_levels(
    decay_path: PathLike
) -> Dict[Tuple[int, int], List[DecayState]]:
    """Parse decay library to build ELIS lookup table for isomeric state identification.

    This function reads ENDF decay files and extracts excitation energy (ELIS)
    and isomeric state number (LISO) for each nuclear state. The resulting
    lookup table is used to map GENDF MF=10 metastable products to the correct
    OpenMC `_m{n}` naming.

    Supports two formats:
    1. **Directory of files**: FISPACT-style directories with one file per nuclide
       (e.g., ``Ir192``, ``Ir192m``, ``Ir192n`` for ground, m1, m2)
    2. **Single concatenated file**: Multiple materials in one file
       (e.g., ``ukdd-12_decay.dat``, ``JEFF311RDD_ALL.OUT``, ``EAF2010-all.txt``)

    Parameters
    ----------
    decay_path : path-like
        Path to decay library. Can be:
        - Directory containing individual ENDF decay files
        - Single file containing multiple materials

    Returns
    -------
    dict
        Dictionary mapping ``(Z, A)`` tuple to list of :class:`DecayState` objects.
        Each nuclide may have multiple states (ground + metastables).

    Raises
    ------
    FileNotFoundError
        If decay_path does not exist

    Examples
    --------
    >>> # Parse JEFF33 decay directory
    >>> decay_lookup = parse_decay_isomeric_levels('/path/to/JEFF33data/decay/')
    >>> # Get Ir-192 states
    >>> ir192_states = decay_lookup[(77, 192)]
    >>> for state in ir192_states:
    ...     print(f"LISO={state.liso}: ELIS={state.elis:.0f} eV")
    LISO=0: ELIS=0 eV
    LISO=1: ELIS=56720 eV
    LISO=2: ELIS=168140 eV

    See Also
    --------
    lookup_liso : Find LISO for given excitation energy
    DecayState : Data class for nuclear state information
    """
    # New library load: re-arm the once-per-(Z, A) ambiguity warnings so a
    # second load in the same process is not silently deduped against the first.
    _WARNED_ELIS_AMBIGUITY.clear()

    decay_path = Path(decay_path)

    if not decay_path.exists():
        raise FileNotFoundError(f"Decay library path not found: {decay_path}")

    # Determine format based on path type
    if decay_path.is_dir():
        return _parse_decay_directory(decay_path)
    else:
        return _parse_decay_single_file(decay_path)


def _parse_decay_directory(
    decay_dir: Path
) -> Dict[Tuple[int, int], List[DecayState]]:
    """Parse a directory of individual decay files.

    Internal function for :func:`parse_decay_isomeric_levels`.

    Handles two cases:
    1. Directory with multiple individual files (FISPACT style, e.g., Ir192, Ir192m)
    2. Directory with a single concatenated file (JEFF-4.0 style)
    """
    decay_data = defaultdict(list)

    # Get all files in directory
    files = [f for f in decay_dir.iterdir() if f.is_file()]

    # Check for single concatenated file case (e.g., JEFF-4.0)
    if len(files) <= 2:
        large_files = [f for f in files if f.stat().st_size > 1_000_000]
        if large_files:
            for large_file in large_files:
                single_file_data = _parse_decay_single_file(large_file)
                for key, states in single_file_data.items():
                    decay_data[key].extend(states)
            return dict(decay_data)

    for filepath in files:
        try:
            mat = endf.Material(str(filepath))

            if (1, 451) not in mat.sections:
                continue

            info = mat.section_data[1, 451]
            za = int(info.get('ZA', 0))
            if za == 0:
                continue

            z = za // 1000
            a = za % 1000

            elis = float(info.get('ELIS', 0.0))
            liso = int(info.get('LISO', 0))

            half_life = None
            if (8, 457) in mat.sections:
                hl_data = mat.section_data[8, 457]
                t12 = hl_data.get('T1/2', None)
                if t12 is not None:
                    # T1/2 may be returned as (value, uncertainty) tuple
                    if isinstance(t12, tuple):
                        half_life = float(t12[0])
                    else:
                        half_life = float(t12)

            state = DecayState(z=z, a=a, elis=elis, liso=liso, half_life=half_life)
            decay_data[(z, a)].append(state)

        except Exception as e:
            _logger.debug("Failed to parse decay file %s: %s", filepath, e)
            continue

    return dict(decay_data)


def _parse_decay_single_file(
    filepath: Path
) -> Dict[Tuple[int, int], List[DecayState]]:
    """Parse a single file containing multiple decay materials.

    Internal function for :func:`parse_decay_isomeric_levels`.

    Handles ENDF-6 formatted files where multiple materials are concatenated
    in one file (e.g., JEFF-4.0, ukdd-12_decay.dat, JEFF311RDD_ALL.OUT, EAF2010-all.txt).
    """
    decay_data = defaultdict(list)

    try:
        for mat in endf.get_materials(filepath):
            try:
                if (1, 451) not in mat.section_data:
                    continue

                info = mat.section_data[(1, 451)]
                za = int(info.get('ZA', 0))

                if za < 1001:
                    continue

                z = za // 1000
                a = za % 1000

                elis = float(info.get('ELIS', 0.0))
                liso = int(info.get('LISO', 0))

                half_life = None
                if (8, 457) in mat.section_data:
                    try:
                        mf8_457 = mat.section_data[(8, 457)]
                        if isinstance(mf8_457, dict):
                            t12 = mf8_457.get('T1/2', None)
                            if t12 is not None:
                                # T1/2 may be returned as (value, uncertainty) tuple
                                if isinstance(t12, tuple):
                                    half_life = float(t12[0])
                                else:
                                    half_life = float(t12)
                    except Exception as e:
                        _logger.debug("Failed to parse T1/2 for ZA=%s: %s", za, e)

                state = DecayState(z=z, a=a, elis=elis, liso=liso, half_life=half_life)
                decay_data[(z, a)].append(state)

            except Exception as e:
                _logger.debug("Failed to parse material ZA=%s: %s", za if 'za' in dir() else '?', e)
                continue

    except ValueError as e:
        # ValueError indicates formatting issues (e.g., EAF2010-all.txt).
        # Try fallback manual parser.
        decay_data = _parse_decay_single_file_manual(filepath)
        if decay_data:
            return decay_data
        warnings.warn(
            f"Error parsing decay file {filepath}: {e}. "
            f"Manual fallback parser also failed.",
            UserWarning
        )
    except Exception as e:
        warnings.warn(f"Error parsing decay file {filepath}: {e}", UserWarning)

    return dict(decay_data)


def _parse_decay_single_file_manual(
    filepath: Path
) -> Dict[Tuple[int, int], List[DecayState]]:
    """Manual parser for concatenated ENDF decay files.

    Fallback parser for files that fail with endf.get_materials() due to
    non-standard formatting (e.g., EAF2010-all.txt with comment lines
    like "----JEFF-311" in the data fields).
    """
    decay_data = defaultdict(list)

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        warnings.warn(f"Cannot read decay file {filepath}: {e}", UserWarning)
        return dict(decay_data)

    current_mf = None
    current_mt = None
    mf1_451_lines = []

    for line in lines:
        if len(line) < 75:
            continue

        try:
            mat_str = line[66:70].strip()
            mf_str = line[70:72].strip()
            mt_str = line[72:75].strip()

            if not mat_str or not mf_str or not mt_str:
                continue

            try:
                mat = int(mat_str)
                mf = int(mf_str)
                mt = int(mt_str)
            except ValueError:
                continue

            if mat == 0:
                if mf1_451_lines:
                    _process_mf1_451_section(mf1_451_lines, decay_data)
                    mf1_451_lines = []
                continue

            if mf != current_mf or mt != current_mt:
                if mf1_451_lines:
                    _process_mf1_451_section(mf1_451_lines, decay_data)
                    mf1_451_lines = []
                current_mf = mf
                current_mt = mt

            if mf == 1 and mt == 451:
                mf1_451_lines.append(line)

        except Exception as e:
            _logger.debug("Failed to parse line in manual parser: %s", e)
            continue

    if mf1_451_lines:
        _process_mf1_451_section(mf1_451_lines, decay_data)

    return dict(decay_data)


def _process_mf1_451_section(
    lines: List[str],
    decay_data: Dict[Tuple[int, int], List[DecayState]]
) -> None:
    """Process a collected MF=1 MT=451 section to extract decay data."""
    if len(lines) < 2:
        return

    try:
        line1 = lines[0]
        za_str = line1[0:11].strip()
        if not za_str:
            return

        za = int(py_float_endf(za_str))

        if za < 1001:
            return

        z = za // 1000
        a = za % 1000

        line2 = lines[1]
        elis_str = line2[0:11].strip()
        liso_str = line2[33:44].strip()

        elis = py_float_endf(elis_str) if elis_str else 0.0
        liso = int(py_float_endf(liso_str)) if liso_str else 0

        state = DecayState(z=z, a=a, elis=elis, liso=liso, half_life=None)
        decay_data[(z, a)].append(state)

    except Exception as e:
        _logger.debug("Failed to process MF1/MT451 section: %s", e)


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'DecayState',
    'elis_match',
    'parse_decay_isomeric_levels',
    'lookup_liso',
    'ELIS_RTOL',
    'ELIS_ATOL',
]
