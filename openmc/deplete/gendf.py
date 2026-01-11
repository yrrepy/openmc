"""GENDF Cross-Section Library Module

This module provides functionality for reading and using FISPACT GENDF
(Group-averaged ENDF) cross-section libraries for depletion calculations.
GENDF files contain pre-processed, group-averaged cross-sections optimized
for activation and burnup calculations.

The module supports:
- Loading GENDF libraries (ENDF-6 format .asc files)
- Extracting cross-sections for specific nuclides and reactions
- Validating energy group structures (CCFE-709, UKAEA-1102)
- Caching for efficient repeated access
- Automatic selection of C++ (fast) or Python (fallback) backend

.. versionadded:: 0.15.3
"""

from __future__ import annotations
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
import os

import numpy as np
import endf

from openmc.checkvalue import check_type, check_value, PathLike
from openmc.mgxs import GROUP_STRUCTURES
from openmc.data import REACTION_MT
from openmc.deplete.chain import REACTIONS

# Import ELIS/decay functions from dedicated module
from openmc.deplete.decay_elis import (
    DecayState,
    elis_match,
    parse_decay_isomeric_levels,
    lookup_liso,
    ELIS_RTOL,
    ELIS_ATOL,
)


# Supported energy group structures for GENDF libraries
SUPPORTED_GROUP_STRUCTURES = {'CCFE-709', 'UKAEA-1102'}

# H1 fix: Unified tolerance constants (must match C++ gendf.h)
# Relative tolerance for energy boundary matching
GENDF_RTOL_MATCH = 1.0e-6
# Relative tolerance for energy boundary mismatch warnings
GENDF_RTOL_WARN = 1.0e-4
# Absolute tolerance for energy matching (only used when values are near zero)
GENDF_ATOL = 0.0

# Atomic symbols for nuclide identification
ATOMIC_SYMBOL = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
    11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca',
    21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
    31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr',
    41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
    51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd',
    61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb',
    71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg',
    81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th',
    91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm'
}

def _build_mt_to_reaction():
    """Build MT to reaction name mapping from chain.py REACTIONS dict.

    For reactions with multiple MTs (e.g., (n,2n) has MT 16 and 875-891),
    we use only the minimum (primary) MT since MF=10 data uses primary MTs.

    Returns
    -------
    dict
        Mapping from primary MT number to reaction name string
    """
    mt_to_reaction = {}
    for reaction_name, info in REACTIONS.items():
        # Use minimum MT as the "primary" MT
        primary_mt = min(info.mts)
        mt_to_reaction[primary_mt] = reaction_name
    return mt_to_reaction

# Build MT to reaction mapping from chain.py::REACTIONS - automatically stays in sync
MT_TO_REACTION = _build_mt_to_reaction()

# Reverse mapping for convenience (reaction name -> MT)
REACTION_TO_MT = {v: k for k, v in MT_TO_REACTION.items()}

# Try to import C++ backend (via openmc.lib.gendf)
# If available, it will be used automatically for better performance
_CPP_BACKEND_AVAILABLE = False
_CppGENDFLibrary = None

try:
    from openmc.lib import gendf as cpp_gendf
    _CppGENDFLibrary = cpp_gendf.GENDFLibrary
    _CPP_BACKEND_AVAILABLE = True
except (ImportError, AttributeError):
    # C++ backend not available - will use Python implementation
    pass


# ============================================================================
# Utility Functions for GENDF File Interaction
# ============================================================================

def get_target_name(material: endf.Material) -> str:
    """Extract target nuclide name from ENDF material.

    Parses the ENDF MF=1, MT=451 metadata to determine the target nuclide
    name in OpenMC format.

    Parameters
    ----------
    material : endf.Material
        ENDF material object

    Returns
    -------
    str
        Nuclide name in OpenMC format (e.g., 'U235', 'Am242_m1')

    Examples
    --------
    >>> mat = endf.Material('U235.asc')
    >>> get_target_name(mat)
    'U235'
    """
    metadata = material.section_data[1, 451]
    Z, A = divmod(metadata['ZA'], 1000)
    symbol = ATOMIC_SYMBOL[Z]

    if metadata['LISO'] == 0:
        return f"{symbol}{A}"
    else:
        return f"{symbol}{A}_m{metadata['LISO']}"


def get_product_name(izap: int, lfs: int) -> Optional[str]:
    """Construct product nuclide name from IZAP and LFS.

    Converts ENDF IZAP (isotope identifier) and LFS (level) values to
    OpenMC nuclide naming convention.

    Parameters
    ----------
    izap : int
        ENDF IZAP value (Z*1000 + A)
    lfs : int
        Level number (0=ground state, >0=metastable state)

    Returns
    -------
    str or None
        Product nuclide name in OpenMC format, or None if IZAP is invalid

    Notes
    -----
    IZAP=0 indicates the product nuclide is not specified in the ENDF file.
    This is a data quality issue in some GENDF libraries (e.g., Al27, Am241
    in JEFF33). When encountered, this function returns None to allow graceful
    handling rather than crashing.

    Examples
    --------
    >>> get_product_name(92235, 0)  # U-235 ground state
    'U235'
    >>> get_product_name(95242, 1)  # Am-242 first metastable
    'Am242_m1'
    >>> get_product_name(0, 0)      # Invalid IZAP
    None
    """
    # Handle invalid IZAP values (data quality issue)
    if izap == 0:
        # Product not specified in ENDF file - return None
        return None

    Z, A = divmod(izap, 1000)

    # Validate Z is in valid range
    if Z not in ATOMIC_SYMBOL:
        warnings.warn(
            f"Invalid atomic number Z={Z} from IZAP={izap}. "
            f"Valid range is 1-100.",
            UserWarning
        )
        return None

    symbol = ATOMIC_SYMBOL[Z]

    if lfs == 0:
        return f"{symbol}{A}"
    else:
        return f"{symbol}{A}_m{lfs}"


# Note: ELIS-based isomeric state mapping functions (DecayState, elis_match,
# parse_decay_isomeric_levels, lookup_liso) have been moved to decay_elis.py


def detect_energy_structure(library_path: PathLike) -> str:
    """Auto-detect energy group structure from GENDF library.

    Reads sample GENDF files from the library and compares their energy grids
    against known group structures to identify which one is being used.
    Uses a smart prioritization strategy to find files most likely to have
    full energy range (avoiding threshold reactions).

    Parameters
    ----------
    library_path : path-like
        Path to directory containing GENDF .asc files

    Returns
    -------
    str
        Name of detected energy structure ('CCFE-709' or 'UKAEA-1102')

    Raises
    ------
    FileNotFoundError
        If no GENDF files found in directory
    ValueError
        If energy structure cannot be determined from any file

    Notes
    -----
    The function tries files in priority order to maximize chances of finding
    a full energy range reaction:
    1. Common actinides (U235, Pu239) - usually have full range
    2. Hydrogen isotopes - lightest nuclides
    3. All other files - systematic search

    This avoids failures when alphabetically-first files contain only
    threshold reactions with partial energy coverage.

    Examples
    --------
    >>> structure = detect_energy_structure('/path/to/GENDF/')
    >>> print(f"Detected: {structure}")
    Detected: UKAEA-1102
    """
    library_path = Path(library_path)

    # Find all GENDF files
    all_files = list(library_path.glob('*.asc'))
    if not all_files:
        raise FileNotFoundError(f"No .asc files found in {library_path}")

    # Smart prioritization: Try files most likely to have full energy range
    # Threshold reactions (e.g., n,2n) have partial coverage and cause false negatives
    priority_patterns = [
        '*U235*.asc',      # Common actinide, full range
        '*Pu239*.asc',     # Common actinide, full range
        '*H1*.asc',        # Lightest nuclide
        '*H2*.asc',        # Deuterium
        '*.asc'            # Fallback: all files
    ]

    # Collect files to try, avoiding duplicates
    files_to_try = []
    seen = set()

    for pattern in priority_patterns:
        candidates = sorted(library_path.glob(pattern))
        for file in candidates[:5]:  # Try up to 5 files per pattern
            if file not in seen:
                files_to_try.append(file)
                seen.add(file)

    # Track failures for better error reporting
    failures = []

    for sample_file in files_to_try:
        try:
            # Use endf library to load the file
            material = endf.Material(str(sample_file))

            # Get any MF=3 section to check energy grid
            mf3_sections = [(mf, mt) for mf, mt in material.section_data.keys() if mf == 3]
            if not mf3_sections:
                failures.append((sample_file.name, "No MF=3 data"))
                continue

            # Extract energy grid
            mf, mt = mf3_sections[0]
            xs_data = material.section_data[mf, mt]

            if 'sigma' not in xs_data:
                failures.append((sample_file.name, f"No sigma data in MT={mt}"))
                continue

            sigma = xs_data['sigma']
            if not hasattr(sigma, 'x'):
                failures.append((sample_file.name, "Cannot extract energy grid"))
                continue

            file_energies = sigma.x
            n_groups = len(file_energies)

            # Compare against known structures
            for structure_name in SUPPORTED_GROUP_STRUCTURES:
                ref_energies = GROUP_STRUCTURES[structure_name]

                # Check if number of points matches
                if len(ref_energies) == n_groups:
                    # Check if energy values match (H1: use unified tolerance)
                    if np.allclose(file_energies, ref_energies, rtol=GENDF_RTOL_MATCH):
                        return structure_name

            # No match for this file
            failures.append((sample_file.name,
                           f"Has {n_groups} groups, expected {len(GROUP_STRUCTURES['CCFE-709'])} or {len(GROUP_STRUCTURES['UKAEA-1102'])}"))

        except Exception as e:
            failures.append((sample_file.name, str(e)))
            continue

    # If we get here, no file matched
    # Provide detailed error report
    error_msg = (
        f"Could not detect energy structure from {len(files_to_try)} files in {library_path}.\n"
        f"Supported structures: {SUPPORTED_GROUP_STRUCTURES}\n"
        f"Files tried:\n"
    )
    for filename, reason in failures[:10]:  # Show first 10 failures
        error_msg += f"  - {filename}: {reason}\n"

    if len(failures) > 10:
        error_msg += f"  ... and {len(failures) - 10} more files\n"

    raise ValueError(error_msg)


# ============================================================================
# IsomericBranching Data Class
# ============================================================================

@dataclass
class IsomericBranching:
    """Container for energy-dependent isomeric branching data.

    This class stores branching ratios that vary with incident neutron energy,
    typically extracted from ENDF MF=10 (production cross-sections) data.

    Attributes
    ----------
    energies : np.ndarray
        Energy points in eV where branching ratios are defined
    products : List[str]
        Product nuclide names in OpenMC format (e.g., ['Ir192', 'Ir192_m1', 'Ir192_m2'])
    branching_ratios : np.ndarray
        2D array of shape [n_products, n_energies] containing branching
        fractions. Each column sums to 1.0 at a given energy.
    parent_nuclide : str
        Parent (target) nuclide name
    reaction : str
        Reaction type in OpenMC notation (e.g., '(n,2n)', '(n,gamma)')
    mt : int
        ENDF MT number for the reaction
    lfs_mapping : Dict[str, int], optional
        Mapping from product name to original ENDF LFS value. Useful for logging
        the ELIS-based remapping (e.g., {'Ir192_m1': 3, 'Ir192_m2': 15} indicates
        LFS=3 was mapped to _m1 and LFS=15 was mapped to _m2).
    elis_mapping : Dict[str, Dict[str, Any]], optional
        Mapping from product name to isomeric mapping information. Contains details
        about how each metastable product was mapped, including:
        - 'method': 'elis' (matched via excitation energy), 'lfs_order' (positional
          mapping based on sorted LFS values), or 'unmatched' (product omitted)
        - 'elis': Excitation energy in eV (from GENDF QM-QI calculation)
        - 'liso': Isomeric state number (1 for _m1, 2 for _m2, etc.)
        - For 'lfs_order' mode, additional fields: 'lfs' (original LFS value),
          'position' (sorted position), 'elis_ref_status' ('ok', 'mismatch', or
          'wrong_liso' indicating ELIS check result for reference)
        Example: {'Ir192_m1': {'method': 'elis', 'elis': 56720.0, 'liso': 1}}

    Examples
    --------
    >>> branching = IsomericBranching(
    ...     energies=np.array([1e5, 1e6, 1e7]),
    ...     products=['Rh104', 'Rh104_m1'],
    ...     branching_ratios=np.array([[0.7, 0.6, 0.5], [0.3, 0.4, 0.5]]),
    ...     parent_nuclide='Rh103',
    ...     reaction='(n,gamma)',
    ...     mt=102
    ... )
    >>> branching.get_branching_at_energy(5e5)
    {'Rh104': 0.65, 'Rh104_m1': 0.35}
    """
    energies: np.ndarray
    products: List[str]
    branching_ratios: np.ndarray
    parent_nuclide: str
    reaction: str
    mt: int
    lfs_mapping: Optional[Dict[str, int]] = None
    elis_mapping: Optional[Dict[str, Dict[str, Any]]] = None

    def get_branching_at_energy(self, energy: float) -> Dict[str, float]:
        """Get branching ratios at specific energy with linear interpolation.

        Parameters
        ----------
        energy : float
            Energy in eV

        Returns
        -------
        dict
            Dictionary mapping product names to branching fractions
        """
        result = {}
        for i, product in enumerate(self.products):
            # Linear interpolation
            ratio = np.interp(energy, self.energies, self.branching_ratios[i, :])
            result[product] = float(ratio)
        return result

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.

        Returns
        -------
        dict
            Dictionary representation suitable for JSON serialization
        """
        return {
            'energies': self.energies.tolist(),
            'products': self.products,
            'branching_ratios': self.branching_ratios.tolist(),
            'parent_nuclide': self.parent_nuclide,
            'reaction': self.reaction,
            'mt': self.mt
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'IsomericBranching':
        """Create IsomericBranching from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with keys matching IsomericBranching attributes

        Returns
        -------
        IsomericBranching
            New instance created from dictionary data
        """
        return cls(
            energies=np.array(data['energies']),
            products=data['products'],
            branching_ratios=np.array(data['branching_ratios']),
            parent_nuclide=data['parent_nuclide'],
            reaction=data['reaction'],
            mt=data['mt']
        )


class _PythonGENDFLibrary:
    """Python implementation of GENDF cross-section library.

    This class provides access to pre-processed group-averaged cross-sections
    from FISPACT GENDF libraries. It handles loading ENDF-6 formatted .asc files,
    caching materials, and extracting cross-section data for specific nuclides
    and reactions.

    .. note::
        Users should use the :class:`GENDFLibrary` factory function instead of
        instantiating this class directly. The factory automatically selects
        the fastest available backend (C++ or Python).

    Parameters
    ----------
    library_path : path-like
        Path to directory containing GENDF .asc files
    energy_structure : str, optional
        Name of the energy group structure. Must be one of 'CCFE-709' or
        'UKAEA-1102'. Default is 'UKAEA-1102'.
    validate_energy_grid : bool, optional
        If True, validate that each GENDF file's energy grid matches the
        specified energy structure. Default is True.
    use_fast_parser : bool, optional
        If True, use optimized MF=3-only parser that skips covariance data,
        providing 3-4x speedup. If False, use full endf.Material parser.
        Default is True.
    decay_file : path-like, optional
        Path to ENDF decay library for ELIS-based isomeric state mapping.
        Can be a directory of decay files or a single concatenated file.
        When provided, enables accurate mapping of GENDF MF=10 metastable
        products to OpenMC ``_m{n}`` naming based on excitation energy
        matching. **Highly recommended** for isomeric branching workflows.
    elis_rtol : float, optional
        Relative tolerance for ELIS matching (default: 0.01 = 1%).
        Used with ``elis_atol`` to determine if GENDF and decay ELIS values match.
    elis_atol : float, optional
        Absolute tolerance in eV for ELIS matching (default: 100.0 eV).
        Provides a floor for matching low excitation energies.
    skip_zero_elis_metastables : bool, optional
        If True (default), skip metastable states (LISO > 0) that have ELIS=0.0
        in the decay library when performing ELIS matching. This is a physics
        constraint - metastable states are excited states and must have ELIS > 0.
        ELIS=0 for a metastable indicates invalid data in the decay library
        (e.g., 32 such cases in JEFF-4.0 including Np242_m1, Ta178_m1).
        Set to False only for debugging.

    Attributes
    ----------
    library_path : pathlib.Path
        Path to the GENDF library directory
    energy_structure : str
        Name of the energy group structure
    energy_bounds : numpy.ndarray
        Energy group boundaries in eV
    n_groups : int
        Number of energy groups
    decay_lookup : dict or None
        Decay state lookup table if decay_file was provided, None otherwise.
        Maps (Z, A) tuples to lists of :class:`DecayState` objects.

    Examples
    --------
    >>> # Basic usage for cross-section retrieval (no decay file needed)
    >>> gendf_lib = GENDFLibrary('/path/to/JEFF40-GENDF/', 'UKAEA-1102')
    >>> xs = gendf_lib.get_xs('Ac225', 102, gendf_lib.energy_bounds)
    >>>
    >>> # With ELIS-based isomeric mapping (recommended)
    >>> gendf_lib = GENDFLibrary(
    ...     '/path/to/JEFF40-GENDF/',
    ...     decay_file='/path/to/JEFF40-decay/'
    ... )
    >>> branching = gendf_lib.get_branching_ratios('Ir191', 102)
    >>> print(branching.products)  # Correctly mapped: ['Ir192', 'Ir192_m1', 'Ir192_m2']
    >>>
    >>> # Cross-library usage with relaxed tolerance
    >>> gendf_lib = GENDFLibrary(
    ...     '/path/to/TENDL2017-GENDF/',
    ...     decay_file='/path/to/UKDD12_decay.dat',
    ...     elis_rtol=0.10  # Allow 10% tolerance for cross-library mismatches
    ... )

    Notes
    -----
    GENDF files use naming convention with suffixes:
    - 'g' for ground state (e.g., Ac225g.asc)
    - 'm' for metastable state (e.g., Ac225mg.asc)

    The energy grid in GENDF files represents group boundaries. Cross-section
    values are group-averaged (integrated over each group).

    **ELIS-Based Mapping** (when decay_file is provided):

    Instead of assuming LFS order equals LISO order, the library uses excitation
    energy (ELIS) matching:

    1. For each GENDF MF=10 metastable product, calculate ELIS = QM - QI
    2. Look up matching metastable state in decay library by ELIS
    3. Use decay library's LISO value for ``_m{n}`` naming

    This handles cases like Ir191(n,gamma)->Ir192 where GENDF uses LFS=3,15
    but decay library correctly identifies these as LISO=1,2 (m1, m2).

    .. warning::
        **Thread Safety (H5 fix)**: This Python backend is NOT thread-safe.
        The internal caches (`_material_cache`, `_file_index`, `_pending_metastable`)
        are modified without synchronization. For thread-safe access, either:

        1. Use the C++ backend (GENDFLibrary with backend='cpp'), which uses
           proper `std::shared_mutex` read/write locking
        2. Create separate library instances per thread
        3. Externally synchronize access (e.g., using `threading.Lock`)

        The C++ backend is recommended for multi-threaded applications.

    See Also
    --------
    parse_decay_isomeric_levels : Parse decay library for ELIS data
    lookup_liso : Find LISO for given excitation energy
    DecayState : Data class for nuclear state information
    """

    def __init__(
        self,
        library_path: PathLike,
        energy_structure: str = 'UKAEA-1102',
        validate_energy_grid: bool = True,
        use_fast_parser: bool = True,
        decay_file: Optional[PathLike] = None,
        elis_rtol: float = ELIS_RTOL,
        elis_atol: float = ELIS_ATOL,
        skip_zero_elis_metastables: bool = True,
        mapping_mode: str = 'elis'
    ):
        # Validate inputs
        check_type('library_path', library_path, (str, Path))
        check_value('energy_structure', energy_structure, SUPPORTED_GROUP_STRUCTURES)
        check_type('validate_energy_grid', validate_energy_grid, bool)
        check_type('use_fast_parser', use_fast_parser, bool)
        check_type('elis_rtol', elis_rtol, float)
        check_type('elis_atol', elis_atol, float)
        check_type('skip_zero_elis_metastables', skip_zero_elis_metastables, bool)
        check_value('mapping_mode', mapping_mode, ('elis', 'lfs_order'))

        self.library_path = Path(library_path)
        if not self.library_path.exists():
            raise FileNotFoundError(
                f"GENDF library path does not exist: {self.library_path}")
        if not self.library_path.is_dir():
            raise ValueError(
                f"GENDF library path must be a directory: {self.library_path}")

        self.energy_structure = energy_structure
        self.energy_bounds = GROUP_STRUCTURES[energy_structure].copy()
        self.n_groups = len(self.energy_bounds) - 1
        self._validate_energy_grid = validate_energy_grid
        self._use_fast_parser = use_fast_parser

        # Isomeric state mapping parameters
        self._mapping_mode = mapping_mode
        self._elis_rtol = elis_rtol
        self._elis_atol = elis_atol
        self._skip_zero_elis_metastables = skip_zero_elis_metastables

        # Parse decay file for ELIS lookup if provided
        self.decay_lookup = None
        self._decay_file = None
        if decay_file is not None:
            check_type('decay_file', decay_file, (str, Path))
            self._decay_file = Path(decay_file)
            self.decay_lookup = parse_decay_isomeric_levels(self._decay_file)
            n_nuclides = len(self.decay_lookup)
            n_states = sum(len(states) for states in self.decay_lookup.values())

        # Cache for loaded ENDF materials
        # Key: nuclide name (OpenMC format), Value: endf.Material object OR dict
        self._material_cache = {}

        # Processing errors list for tracking ELIS mismatches, missing metastables, etc.
        self._processing_errors = []

        # Build index of available files
        self._build_file_index()

    @property
    def energy_bins(self):
        """Alias for energy_bounds (backwards compatibility).

        Returns
        -------
        numpy.ndarray
            Energy group boundaries in eV
        """
        return self.energy_bounds

    @property
    def mapping_mode(self):
        """Isomeric state mapping mode.

        Returns
        -------
        str
            Either 'elis' (ELIS-based matching) or 'lfs_order' (FISPACT-like
            positional mapping)
        """
        return self._mapping_mode

    def _build_file_index(self):
        """Build index of available GENDF files in the library.

        Creates mapping from OpenMC nuclide names to GENDF filenames.
        Handles naming conventions like 'Al027g.asc' → 'Al27', 'Ac225g.asc' → 'Ac225'

        FISPACT metastable naming convention:
        - 'g'  = ground state → Element{A}
        - 'mg' = m1 (1st metastable) → Element{A}_m1
        - 'ng' = m2 (2nd metastable) → Element{A}_m2
        - 'og' = m3 (3rd metastable) → Element{A}_m3
        - 'pg' = m4 (4th metastable) → Element{A}_m4
        - 'qg' = m5 (5th metastable) → Element{A}_m5

        For metastable nuclides, uses filename heuristics initially and defers
        loading MF=1, MT=451 metadata until first access for performance.
        """
        import re
        self._file_index = {}

        # Track metastable files that need validation on first access
        # Maps preliminary name -> (filepath, needs_validation)
        self._pending_metastable = {}

        # FISPACT suffix to OpenMC metastable level mapping
        # Order matters: check longer suffixes first
        METASTABLE_SUFFIXES = {
            'mg': '_m1',  # 1st metastable
            'ng': '_m2',  # 2nd metastable
            'og': '_m3',  # 3rd metastable
            'pg': '_m4',  # 4th metastable
            'qg': '_m5',  # 5th metastable
        }

        # Find all .asc files in library directory
        asc_files = list(self.library_path.glob('*.asc'))

        if not asc_files:
            warnings.warn(
                f"No .asc files found in GENDF library: {self.library_path}",
                UserWarning)

        for filepath in asc_files:
            filename = filepath.stem  # Remove .asc extension

            # Skip macOS metadata files
            if filename.startswith('._'):
                continue

            # Check for metastable suffixes (must check before single 'g')
            metastable_suffix = None
            openmc_suffix = None
            for fispact_suffix, omc_suffix in METASTABLE_SUFFIXES.items():
                if filename.endswith(fispact_suffix):
                    metastable_suffix = fispact_suffix
                    openmc_suffix = omc_suffix
                    break

            if metastable_suffix is not None:
                # Metastable nuclide: extract base name and add OpenMC suffix
                raw_name = filename[:-len(metastable_suffix)]
                match = re.match(r'([A-Z][a-z]?)(\d+)', raw_name)
                if match:
                    element = match.group(1)
                    mass = int(match.group(2))
                    preliminary_name = f"{element}{mass}{openmc_suffix}"
                else:
                    preliminary_name = raw_name + openmc_suffix

                # Store filepath and mark as needing validation
                self._file_index[preliminary_name] = filepath
                self._pending_metastable[preliminary_name] = True

            else:
                # Ground state nuclides: use fast filename parsing
                if filename.endswith('g'):
                    raw_name = filename[:-1]  # Remove 'g' suffix
                else:
                    raw_name = filename  # No suffix

                # Parse and normalize to OpenMC format (remove leading zeros)
                # e.g., 'Al027' → 'Al27'
                match = re.match(r'([A-Z][a-z]?)(\d+)', raw_name)
                if match:
                    element = match.group(1)
                    mass = int(match.group(2))  # Convert to int to remove leading zeros
                    nuclide_name = f"{element}{mass}"
                else:
                    # If parsing fails, use raw name
                    nuclide_name = raw_name

                self._file_index[nuclide_name] = filepath

    def _parse_gendf_mf3_only(self, filepath: Path) -> dict:
        """Fast parser that only extracts MF=3 (cross-section) data.

        This parser skips all non-MF=3 sections (especially MF=33 covariance),
        providing 3-4x speedup compared to full ENDF parsing.

        Parameters
        ----------
        filepath : Path
            Path to GENDF .asc file

        Returns
        -------
        dict
            Dictionary with structure matching endf.Material.section_data
            Keys: (MF, MT) tuples
            Values: dict with 'sigma' key containing Tabulated1D object
        """
        from openmc.data.function import Tabulated1D
        try:
            from endf.records import float_endf, int_endf
        except ImportError:
            # Fallback if endf C extensions not available
            def float_endf(s):
                s = s.strip()
                if not s:
                    return 0.0
                # Simple ENDF float parser (handles +/- exponent format)
                import re
                s = re.sub(r'([+-])(\s*\d)', r'\1\2', s)
                s = s.replace('+', 'e+').replace('-', 'e-')
                if s.startswith('e'):
                    s = '1' + s
                return float(s)

            def int_endf(s):
                s = s.strip()
                return 0 if not s or s.isspace() else int(s)

        section_data = {}

        with open(filepath, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i]

            # ENDF-6 format: columns 71-72 contain MF, 73-75 contain MT
            # (using 0-based Python indexing: 70:72 for MF, 72:75 for MT)
            if len(line) < 75:
                i += 1
                continue

            try:
                mf = int_endf(line[70:72])  # Columns 71-72 (1-based)
                mt = int_endf(line[72:75])  # Columns 73-75 (1-based)
            except (ValueError, IndexError):
                i += 1
                continue

            # Only process MF=3 (cross-sections)
            if mf == 3 and mt > 0:
                # Parse TAB1 record for this cross-section
                try:
                    # Read HEAD record (C1, C2, L1, L2, N1, N2)
                    n1 = int_endf(line[44:55])  # Number of interpolation regions
                    n2 = int_endf(line[55:66])  # Number of points

                    i += 1

                    # Skip interpolation table if present (N1 regions)
                    n_interp_lines = (2 * n1 + 5) // 6  # 6 integers per line
                    i += n_interp_lines

                    # Read data points (energy and cross-section)
                    n_data_lines = (2 * n2 + 5) // 6  # 6 floats per line

                    energies = []
                    xs_values = []

                    for _ in range(n_data_lines):
                        if i >= len(lines):
                            break
                        data_line = lines[i]

                        # Parse up to 6 values per line (each 11 characters)
                        for j in range(0, min(66, len(data_line)), 11):
                            value_str = data_line[j:j+11]
                            if value_str.strip():
                                try:
                                    val = float_endf(value_str)
                                    if len(energies) <= len(xs_values):
                                        energies.append(val)
                                    else:
                                        xs_values.append(val)
                                except (ValueError, TypeError):
                                    pass

                        i += 1

                    # Create Tabulated1D object (compatible with endf.Material format)
                    if len(energies) == len(xs_values) and len(energies) > 0:
                        sigma = Tabulated1D(
                            x=np.array(energies),
                            y=np.array(xs_values)
                        )
                        section_data[(3, mt)] = {'sigma': sigma}

                except (ValueError, IndexError):
                    # Skip malformed section
                    i += 1
                    continue
            else:
                # Skip non-MF=3 sections quickly
                i += 1

        return section_data

    def _find_gendf_file(self, nuclide_name: str) -> Path:
        """Find GENDF file for a nuclide, trying multiple naming conventions.

        Parameters
        ----------
        nuclide_name : str
            Nuclide name in OpenMC format (e.g., 'Al27', 'Ag110m')

        Returns
        -------
        pathlib.Path
            Path to the GENDF file

        Raises
        ------
        KeyError
            If file cannot be found with any naming convention
        """
        # Parse nuclide name (e.g., 'Al27', 'Ag110m' -> element='Al'/'Ag', mass=27/110, meta=''/m)
        import re
        match = re.match(r'([A-Z][a-z]?)(\d+)(m\d*)?', nuclide_name)
        if not match:
            raise ValueError(f"Cannot parse nuclide name: {nuclide_name}")

        element = match.group(1)
        mass = match.group(2)
        meta = match.group(3) or ''

        # Try multiple naming patterns
        patterns = [
            # Pattern 1: Leading zeros, 'g' suffix (ENDF-B8, TENDL, JEFF-3.x)
            f"{element}{int(mass):03d}{meta}g.asc",
            # Pattern 2: No leading zeros, 'g' suffix (JEFF-4.0)
            f"{element}{mass}{meta}g.asc",
            # Pattern 3: Leading zeros, no 'g' suffix (alternative)
            f"{element}{int(mass):03d}{meta}.asc",
            # Pattern 4: No leading zeros, no 'g' suffix (alternative)
            f"{element}{mass}{meta}.asc",
        ]

        # Try each pattern
        for pattern in patterns:
            filepath = self.library_path / pattern
            if filepath.exists():
                return filepath

        # If not found, raise error
        raise KeyError(
            f"Nuclide '{nuclide_name}' not found in GENDF library. "
            f"Tried patterns: {patterns}")

    def _validate_metastable_name(self, preliminary_name: str):
        """Validate and correct metastable nuclide naming using MF=1 MT=451 metadata.

        This is called lazily on first access to a metastable nuclide.

        Parameters
        ----------
        preliminary_name : str
            Preliminary name from filename heuristics (e.g., 'Co62_m1')

        Returns
        -------
        str
            Correct nuclide name based on LISO value (e.g., 'Co62_m1' or 'Co62_m2')
        """
        # H5 note: This method modifies _file_index, _material_cache, and
        # _pending_metastable without synchronization. See class docstring
        # for thread-safety warnings.
        if preliminary_name not in self._pending_metastable:
            # Not a pending metastable or already validated
            return preliminary_name

        filepath = self._file_index[preliminary_name]

        try:
            # Load material to get accurate naming from metadata
            # Use full parser to ensure MF=1, MT=451 is available
            material = endf.Material(str(filepath))
            correct_name = get_target_name(material)

            # Update index if name changed
            if correct_name != preliminary_name:
                # Remove old entry
                del self._file_index[preliminary_name]
                # Add with correct name
                self._file_index[correct_name] = filepath

                # If we already had this in cache under wrong name, update it
                if preliminary_name in self._material_cache:
                    self._material_cache[correct_name] = self._material_cache[preliminary_name]
                    del self._material_cache[preliminary_name]

            # Cache the loaded material to avoid reloading
            self._material_cache[correct_name] = material

            # Mark as validated
            del self._pending_metastable[preliminary_name]

            return correct_name

        except Exception as e:
            # If validation fails, keep using preliminary name
            warnings.warn(
                f"Could not validate metastable naming for {filepath.name}: {e}. "
                f"Using preliminary name '{preliminary_name}'.",
                UserWarning)
            # Mark as validated (even if failed) to avoid repeated attempts
            del self._pending_metastable[preliminary_name]
            return preliminary_name

    def _load_material(self, nuclide_name: str, require_full_parser: bool = False):
        """Load ENDF material for a nuclide.

        Parameters
        ----------
        nuclide_name : str
            Nuclide name in OpenMC format (e.g., 'Ac225', 'Ag110m')
        require_full_parser : bool, optional
            If True, forces use of the full endf.Material parser even when
            fast parser is enabled. Required for accessing MF=10 data
            (isomeric branching). Default is False.

        Returns
        -------
        endf.Material or dict
            Loaded ENDF material object (if using full parser) or
            dict with section_data (if using fast parser)

        Raises
        ------
        KeyError
            If nuclide is not available in the GENDF library
        """
        # For metastable nuclides, validate naming on first access
        if nuclide_name in self._pending_metastable:
            correct_name = self._validate_metastable_name(nuclide_name)
            if correct_name != nuclide_name:
                # Name was corrected, use the correct name going forward
                nuclide_name = correct_name

        # Check cache first
        # For require_full_parser, we need to check if cached material has full data
        cache_key = nuclide_name
        if nuclide_name in self._material_cache:
            cached = self._material_cache[nuclide_name]
            # If we need full parser but cached is from fast parser, reload
            if require_full_parser and not isinstance(cached, endf.Material):
                # Need to reload with full parser
                pass
            else:
                return cached

        # Find the file (tries multiple naming patterns)
        filepath = None

        # First try direct lookup
        if nuclide_name in self._file_index:
            filepath = self._file_index[nuclide_name]
        else:
            # Try _find_gendf_file which handles multiple naming patterns
            try:
                filepath = self._find_gendf_file(nuclide_name)
            except KeyError:
                # For metastable nuclides, check if we have it under a different metastable level
                import re
                if '_m' in nuclide_name:
                    match = re.match(r'([A-Z][a-z]?\d+)_m\d+', nuclide_name)
                    if match:
                        base = match.group(1)
                        # Check all pending metastables with this base
                        for pending_name in list(self._pending_metastable.keys()):
                            if pending_name.startswith(base + '_m'):
                                correct_name = self._validate_metastable_name(pending_name)
                                if correct_name == nuclide_name:
                                    # Found it after validation
                                    filepath = self._file_index[correct_name]
                                    break

                if not filepath:
                    raise KeyError(
                        f"Nuclide '{nuclide_name}' not found in GENDF library. "
                        f"Available nuclides: {sorted(list(self._file_index.keys())[:10])}...")

        # Load material from file
        try:
            # Use full parser if requested OR if fast parser is disabled
            use_full = require_full_parser or not self._use_fast_parser

            if not use_full:
                # Use optimized MF=3-only parser (3-4x faster)
                section_data = self._parse_gendf_mf3_only(filepath)

                # Create a simple object to store section_data
                # (compatible with the rest of the code)
                class _FastMaterial:
                    def __init__(self, data):
                        self.section_data = data

                material = _FastMaterial(section_data)

                # Fast parser doesn't need energy validation (already filtered)
            else:
                # Use full endf.Material parser (slower but complete)
                # Required for MF=10 (isomeric branching) data
                material = endf.Material(str(filepath))

                # Validate energy grid if requested
                if self._validate_energy_grid:
                    self._validate_material_energy_grid(material, nuclide_name)

        except Exception as e:
            raise RuntimeError(
                f"Failed to load GENDF file for {nuclide_name}: {filepath}\n"
                f"Error: {e}")

        # Cache and return
        self._material_cache[nuclide_name] = material
        return material

    def _validate_material_energy_grid(
        self,
        material: endf.Material,
        nuclide_name: str
    ):
        """Validate that material's energy grid matches expected structure.

        Parameters
        ----------
        material : endf.Material
            ENDF material to validate
        nuclide_name : str
            Nuclide name (for error messages)

        Raises
        ------
        ValueError
            If energy grid does not match expected structure
        """
        # Try to find any MF=3 section to check energy grid
        mf3_sections = [(mf, mt) for mf, mt in material.section_data.keys() if mf == 3]

        if not mf3_sections:
            warnings.warn(
                f"No MF=3 (cross-section) data found for {nuclide_name}",
                UserWarning)
            return

        # Check first available cross-section
        mf, mt = mf3_sections[0]
        xs_data = material.section_data[mf, mt]

        if 'sigma' not in xs_data:
            warnings.warn(
                f"No 'sigma' data in MF={mf}, MT={mt} for {nuclide_name}",
                UserWarning)
            return

        sigma = xs_data['sigma']
        if not hasattr(sigma, 'x'):
            return

        gendf_energies = sigma.x

        # Compare with expected energy structure
        if len(gendf_energies) != len(self.energy_bounds):
            raise ValueError(
                f"Energy grid mismatch for {nuclide_name}: "
                f"GENDF has {len(gendf_energies)} points, "
                f"{self.energy_structure} has {len(self.energy_bounds)} boundaries")

        if not np.allclose(gendf_energies, self.energy_bounds):
            max_diff = np.max(np.abs(gendf_energies - self.energy_bounds))
            raise ValueError(
                f"Energy grid values do not match for {nuclide_name}. "
                f"Maximum difference: {max_diff} eV")

    def get_xs(
        self,
        nuclide_name: str,
        mt: int,
        energy_bounds: np.ndarray,
        strict_alignment: bool = True
    ) -> np.ndarray:
        """Get group-averaged cross-section for a nuclide and reaction.

        Parameters
        ----------
        nuclide_name : str
            Nuclide name in OpenMC format (e.g., 'U235', 'Ac225')
        mt : int
            ENDF MT number for the reaction
        energy_bounds : numpy.ndarray
            Energy group boundaries in eV. Must match the library's energy structure.
        strict_alignment : bool, optional
            If True (default), raise ValueError when GENDF energy grid for
            threshold reactions cannot be exactly aligned with library energy
            structure. If False, use nearest-group alignment and issue a warning.
            This only affects threshold reactions (n,2n), (n,3n), etc. that have
            partial energy coverage.

        Returns
        -------
        numpy.ndarray
            Group-averaged cross-section in barns for each energy group.
            Array has length len(energy_bounds) - 1.

        Raises
        ------
        KeyError
            If nuclide or reaction not found in GENDF library
        ValueError
            If energy bounds don't match library structure, or if strict_alignment
            is True and threshold reaction energies cannot be exactly aligned.

        Examples
        --------
        >>> lib = GENDFLibrary('/path/to/gendf/')
        >>> energy = lib.energy_bounds
        >>> xs_ngamma = lib.get_xs('Ac225', 102, energy)  # (n,gamma)
        >>> xs_n2n = lib.get_xs('Ac225', 16, energy)      # (n,2n)

        Notes
        -----
        For threshold reactions like (n,2n) with threshold ~6 MeV, GENDF files
        only contain data above the threshold. The start and end energies must
        align exactly with library group boundaries for accurate placement.
        """
        # Validate energy bounds
        # H1: Use unified tolerance constants (same as C++ backend)
        # NOTE: May need to relax GENDF_RTOL_MATCH if too strict for some FISPACT GENDF files
        if not np.allclose(energy_bounds, self.energy_bounds,
                          rtol=GENDF_RTOL_MATCH, atol=GENDF_ATOL):
            raise ValueError(
                f"Provided energy bounds do not match library energy structure "
                f"'{self.energy_structure}'")

        # Load material
        material = self._load_material(nuclide_name)

        # Check if reaction is available
        if (3, mt) not in material.section_data:
            raise KeyError(
                f"Reaction MT={mt} not found for {nuclide_name} in GENDF library. "
                f"Available reactions: {[mt for mf,mt in material.section_data.keys() if mf==3]}")

        # Extract cross-section data
        xs_data = material.section_data[3, mt]

        if 'sigma' not in xs_data:
            raise ValueError(
                f"No 'sigma' data in MF=3, MT={mt} for {nuclide_name}")

        sigma = xs_data['sigma']

        # Handle threshold reactions that may have partial energy coverage
        # GENDF files for threshold reactions (like n,2n) may only contain data
        # for energies above the threshold, resulting in fewer groups than the full structure
        gendf_energies = sigma.x
        gendf_xs = sigma.y

        # Check if this is a full or partial energy range
        if len(gendf_energies) == len(self.energy_bounds):
            # Full energy range - use first n_groups values directly
            # Return view instead of copy for performance (values are read-only in usage)
            xs_values = gendf_xs[:self.n_groups]
        else:
            # Partial energy range (threshold reaction)
            # Find where the GENDF energies fit in the full energy structure
            xs_values = np.zeros(self.n_groups)

            start_energy = gendf_energies[0]
            end_energy = gendf_energies[-1]

            # Try exact match for start boundary (H1: use unified tolerance)
            start_matches = np.where(np.isclose(
                self.energy_bounds, start_energy,
                rtol=GENDF_RTOL_MATCH, atol=GENDF_ATOL))[0]

            if len(start_matches) == 1:
                start_idx = start_matches[0]
            elif len(start_matches) > 1:
                # Ambiguous match - should not happen with unique energy bounds
                warnings.warn(
                    f"Multiple energy boundary matches for {nuclide_name} MT={mt} "
                    f"start energy {start_energy:.6e} eV. Using first match.",
                    UserWarning)
                start_idx = start_matches[0]
            else:
                # No exact match found
                nearest_idx = np.argmin(np.abs(self.energy_bounds - start_energy))
                nearest_energy = self.energy_bounds[nearest_idx]
                relative_diff = abs(start_energy - nearest_energy) / max(start_energy, 1e-10)

                if strict_alignment:
                    raise ValueError(
                        f"Cannot align GENDF energy grid for {nuclide_name} MT={mt}. "
                        f"GENDF starts at {start_energy:.6e} eV, "
                        f"nearest library boundary is {nearest_energy:.6e} eV "
                        f"(relative difference: {relative_diff:.2e}). "
                        f"This may indicate incompatible energy structures. "
                        f"Set strict_alignment=False to use nearest-group alignment.")
                else:
                    warnings.warn(
                        f"Energy alignment uncertainty for {nuclide_name} MT={mt}: "
                        f"GENDF starts at {start_energy:.6e} eV, "
                        f"using nearest boundary {nearest_energy:.6e} eV "
                        f"(relative difference: {relative_diff:.2e}). "
                        f"Cross-section placement may be off by one energy group.",
                        UserWarning)
                    start_idx = nearest_idx

            # Calculate number of GENDF groups and end index
            # GENDF energies are boundaries, so n_groups = len(energies) - 1
            n_gendf_groups = len(gendf_energies) - 1
            end_idx = min(start_idx + n_gendf_groups, self.n_groups)

            # Validate end boundary also matches (sanity check for contiguous data)
            # H1: Use slightly looser tolerance (10x) for end boundary due to
            # potential accumulated numerical error across many energy groups
            if end_idx < self.n_groups:
                expected_end = self.energy_bounds[end_idx]
                end_rtol = GENDF_RTOL_MATCH * 10  # 1e-5 for end boundary sanity check
                if not np.isclose(expected_end, end_energy, rtol=end_rtol, atol=GENDF_ATOL):
                    relative_diff_end = abs(end_energy - expected_end) / max(end_energy, 1e-10)
                    if strict_alignment:
                        raise ValueError(
                            f"GENDF energy range for {nuclide_name} MT={mt} does not "
                            f"align with library structure. End energy mismatch: "
                            f"GENDF {end_energy:.6e} eV vs expected {expected_end:.6e} eV "
                            f"(relative difference: {relative_diff_end:.2e})")
                    else:
                        warnings.warn(
                            f"End energy mismatch for {nuclide_name} MT={mt}: "
                            f"GENDF {end_energy:.6e} eV vs expected {expected_end:.6e} eV "
                            f"(relative difference: {relative_diff_end:.2e})",
                            UserWarning)

            # Fill in the cross-section values for the available energy range
            n_values = min(end_idx - start_idx, len(gendf_xs))
            xs_values[start_idx:start_idx + n_values] = gendf_xs[:n_values]

        return xs_values

    def has_nuclide(self, nuclide_name: str) -> bool:
        """Check if nuclide is available in library.

        Parameters
        ----------
        nuclide_name : str
            Nuclide name in OpenMC format (e.g., 'U235', 'Pu239')

        Returns
        -------
        bool
            True if nuclide is available
        """
        # First check if it's directly in the index
        if nuclide_name in self._file_index:
            return True

        # For metastable nuclides, check if we might have it under a different name
        # (e.g., user requests 'Co62_m2' but we have it as 'Co62_m1' pending validation)
        import re
        if '_m' in nuclide_name:
            # Extract base nuclide (e.g., 'Co62' from 'Co62_m2')
            match = re.match(r'([A-Z][a-z]?\d+)_m\d+', nuclide_name)
            if match:
                base = match.group(1)
                # Check if we have any metastable variant that hasn't been validated yet
                for pending_name in self._pending_metastable:
                    if pending_name.startswith(base + '_m'):
                        # Validate it now to see if it matches
                        correct_name = self._validate_metastable_name(pending_name)
                        if correct_name == nuclide_name:
                            return True
        return False

    def available_nuclides(self) -> list[str]:
        """Get list of nuclides available in the GENDF library.

        Returns
        -------
        list of str
            Sorted list of nuclide names in OpenMC format

        Note
        ----
        For performance reasons, metastable nuclide names may show preliminary
        names until they are accessed and validated.
        """
        return sorted(list(self._file_index.keys()))

    def available_nuclides_set(self) -> frozenset:
        """Get set of all available nuclides for O(1) lookup.

        This is more efficient than `available_nuclides()` when checking
        nuclide availability in a loop.

        Returns
        -------
        frozenset of str
            Immutable set of nuclide names

        Examples
        --------
        >>> lib = GENDFLibrary('/path/to/GENDF/', 'UKAEA-1102')
        >>> available = lib.available_nuclides_set()
        >>> if 'U235' in available:  # O(1) lookup
        ...     xs = lib.get_xs('U235', 102)
        """
        if not hasattr(self, '_nuclides_set_cache') or self._nuclides_set_cache is None:
            self._nuclides_set_cache = frozenset(self._file_index.keys())
        return self._nuclides_set_cache

    def available_reactions(self, nuclide_name: str) -> list[int]:
        """Get list of available reaction MT numbers for a nuclide.

        Parameters
        ----------
        nuclide_name : str
            Nuclide name in OpenMC format

        Returns
        -------
        list of int
            Sorted list of MT numbers available for this nuclide

        Raises
        ------
        KeyError
            If nuclide not found in library
        """
        material = self._load_material(nuclide_name)
        mt_numbers = [mt for mf, mt in material.section_data.keys() if mf == 3]
        return sorted(mt_numbers)

    def get_branching_ratios(
        self,
        nuclide_name: str,
        mt: int
    ) -> Optional[IsomericBranching]:
        """Extract energy-dependent isomeric branching ratios from MF=10.

        Reads ENDF MF=10 (production cross-sections) data to determine
        how products are distributed among ground and metastable states
        as a function of incident neutron energy.

        Parameters
        ----------
        nuclide_name : str
            Nuclide name in OpenMC format (e.g., 'Rh103', 'U235')
        mt : int
            ENDF MT number for the reaction (e.g., 102 for n,gamma, 16 for n,2n)

        Returns
        -------
        IsomericBranching or None
            Energy-dependent branching data if isomeric branching exists,
            None if only ground state is produced or MF=10 data is absent

        Raises
        ------
        KeyError
            If nuclide not found in library
        ValueError
            If MF=10 data has metastable state but no ground state
            (indicates corrupted GENDF data)

        Notes
        -----
        **Metastable Product Mapping**:

        The mapping of GENDF MF=10 metastable products to OpenMC ``_m{n}`` naming
        depends on whether a decay file was provided:

        **With decay_file (ELIS-based mapping, recommended)**:

        - Calculate excitation energy (ELIS) from GENDF: ``ELIS = QM - QI``
        - Look up LISO in decay library by matching ELIS within tolerance
        - Use decay library's LISO for naming (e.g., LISO=1 -> ``_m1``)
        - If no match within rtol/atol: skip product, warn, renormalize remaining
        - If no metastable states in decay library: skip product, warn, renormalize

        **Note**: decay_file is required for isomeric branching. Without it,
        get_branching_ratios() will raise ValueError.

        Example with ELIS mapping (Ir191(n,gamma) -> Ir192):
        - GENDF: LFS=3, QM-QI=56720 eV; LFS=15, QM-QI=168140 eV
        - Decay: Ir192m (LISO=1, ELIS=56720); Ir192n (LISO=2, ELIS=168140)
        - Result: LFS=3 -> Ir192_m1, LFS=15 -> Ir192_m2 (correctly matched by ELIS)

        Examples
        --------
        >>> # ELIS-based mapping (decay_file required)
        >>> lib = GENDFLibrary(
        ...     '/path/to/JEFF40-GENDF/',
        ...     decay_file='/path/to/JEFF40-decay/'
        ... )
        >>> branching = lib.get_branching_ratios('Ir191', 102)
        >>> print(branching.products)  # ['Ir192', 'Ir192_m1', 'Ir192_m2']

        See Also
        --------
        parse_decay_isomeric_levels : Parse decay library for ELIS data
        lookup_liso : Find LISO for given excitation energy
        """
        # Load material with full parser (required for MF=10 data)
        material = self._load_material(nuclide_name, require_full_parser=True)

        # Check if MF=10 data exists for this reaction
        if (10, mt) not in material.section_data:
            return None

        mf10_data = material.section_data[10, mt]

        # Get QM from the MF=10 section data (Q-value for reaction)
        # Note: In ENDF-6 format, QM may be at section level or in first level
        qm_section = mf10_data.get('QM', None)

        # Extract cross sections for each product level
        ground_data = None
        ground_product = None
        # Track all metastable states: (lfs, izap, sigma, level_data)
        all_meta_levels = []

        for level in mf10_data['levels']:
            lfs = level['LFS']
            izap = level['IZAP']
            sigma = level['sigma']

            # Get product name (without metastable mapping - just base name)
            # We'll handle metastable naming separately
            if izap == 0:
                warnings.warn(
                    f"Skipping MF=10 level in {nuclide_name} MT={mt}: "
                    f"Invalid IZAP={izap} (product not specified in GENDF file). "
                    f"This is a data quality issue in the source GENDF library.",
                    UserWarning
                )
                continue

            # Extract Z, A from IZAP
            z_prod = izap // 1000
            a_prod = izap % 1000

            if z_prod not in ATOMIC_SYMBOL:
                warnings.warn(
                    f"Invalid atomic number Z={z_prod} from IZAP={izap}. "
                    f"Skipping this level.",
                    UserWarning
                )
                continue

            symbol = ATOMIC_SYMBOL[z_prod]
            base_product = f"{symbol}{a_prod}"

            if lfs == 0:
                ground_data = sigma
                ground_product = base_product
            else:
                # Store level data for ELIS calculation
                all_meta_levels.append({
                    'lfs': lfs,
                    'izap': izap,
                    'z': z_prod,
                    'a': a_prod,
                    'sigma': sigma,
                    'level': level,
                    'base_product': base_product
                })

        # If only ground state, no branching
        if len(all_meta_levels) == 0:
            return None

        # Sort metastables by LFS ascending (for consistent ordering)
        all_meta_levels.sort(key=lambda x: x['lfs'])

        # Extract base nuclide name for product naming
        if ground_product:
            base_nuclide = ground_product
        else:
            base_nuclide = all_meta_levels[0]['base_product']

        # ============================================================
        # Metastable Mapping: ELIS-based or LFS-order
        # ============================================================

        mapped_meta_levels = []  # [(lfs, mapped_name, sigma, elis_info), ...]
        lfs_mapping = {}  # {mapped_name: original_lfs}
        reaction_name = MT_TO_REACTION.get(mt, f'MT{mt}')

        if self._mapping_mode == 'elis':
            # ELIS-based mapping using decay library
            # Two-pass approach to avoid conflicts:
            # 1. First pass: Try ELIS matching for all metastables, collect results
            # 2. Second pass: Assign fallback for unmatched, avoiding conflicts

            elis_results = []  # [(meta_dict, elis, liso_or_none), ...]

            # First pass: Calculate ELIS and try lookup
            for meta in all_meta_levels:
                level = meta['level']

                # Calculate ELIS from QM and QI
                qm = level.get('QM', qm_section)
                qi = level.get('QI', 0.0)

                if qm is None:
                    elis = None
                else:
                    elis = qm - qi

                # Try ELIS lookup - returns dict with 'status' key:
                #   'matched': Match within tolerance
                #   'nearest': Outside tolerance (with nearest match info)
                #   'no_decay_data': Nuclide not in decay library
                #   'no_metastables': Nuclide exists but only ground state
                #   'zero_elis_only': Metastables exist but all have ELIS=0
                liso_result = None
                if elis is not None:
                    liso_result = lookup_liso(
                        meta['z'], meta['a'], elis, self.decay_lookup,
                        rtol=self._elis_rtol, atol=self._elis_atol,
                        skip_zero_elis_metastables=self._skip_zero_elis_metastables,
                        return_nearest=True  # Always get nearest match info
                    )

                elis_results.append((meta, elis, liso_result))

            # ============================================================
            # Duplicate Mapping Detection: Multiple LFS → Same LISO
            # ============================================================
            # When multiple GENDF LFS values have ELIS within tolerance of
            # the same decay library LISO, keep only the closest match.

            # Group matches by LISO to detect duplicates
            liso_to_matches = {}  # {liso: [(idx, meta, elis, liso_result, elis_diff), ...]}
            for idx, (meta, elis, liso_result) in enumerate(elis_results):
                if liso_result is not None and liso_result.get('status') == 'matched':
                    liso = liso_result['liso']
                    dk_elis = liso_result['dk_elis']
                    elis_diff = abs(elis - dk_elis) if elis is not None else float('inf')
                    if liso not in liso_to_matches:
                        liso_to_matches[liso] = []
                    liso_to_matches[liso].append((idx, meta, elis, liso_result, elis_diff))

            # Resolve duplicates - keep only closest match
            indices_to_skip = set()
            for liso, matches in liso_to_matches.items():
                if len(matches) > 1:
                    # Multiple LFS mapped to same LISO - keep closest
                    matches.sort(key=lambda x: x[4])  # Sort by elis_diff (ascending)
                    keeper = matches[0]
                    discards = matches[1:]

                    keeper_idx, keeper_meta, keeper_elis, keeper_result, keeper_diff = keeper
                    keeper_lfs = keeper_meta['lfs']
                    z = keeper_meta['z']
                    a = keeper_meta['a']
                    dk_elis = keeper_result['dk_elis']

                    # Collect ALL GENDF LFS levels for this reaction (from all_meta_levels + ground)
                    gendf_all_lfs = []
                    if ground_data is not None:
                        gendf_all_lfs.append({'lfs': 0, 'elis': 0.0})
                    for meta_item in all_meta_levels:
                        qm = meta_item['level'].get('QM', qm_section)
                        qi = meta_item['level'].get('QI', 0.0)
                        meta_elis = (qm - qi) if qm is not None else None
                        gendf_all_lfs.append({
                            'lfs': meta_item['lfs'],
                            'elis': meta_elis
                        })

                    # Collect ALL decay library LISO levels for this (Z, A)
                    decay_states = self.decay_lookup.get((z, a), [])
                    decay_all_liso = []
                    for ds in decay_states:
                        decay_all_liso.append({
                            'liso': ds.liso,
                            'elis': ds.elis,
                            'half_life': ds.half_life
                        })

                    # Build discarded list
                    discarded_list = []
                    for discard in discards:
                        d_idx, d_meta, d_elis, d_result, d_diff = discard
                        discarded_list.append({
                            'lfs': d_meta['lfs'],
                            'elis': d_elis,
                            'diff': d_diff
                        })
                        indices_to_skip.add(d_idx)

                    # Emit warning
                    discarded_str = ', '.join(
                        f"LFS={d['lfs']} (ELIS={d['elis']:.0f}eV, diff={d['diff']:.0f}eV)"
                        for d in discarded_list
                    )
                    warnings.warn(
                        f"DUPLICATE_MAPPING: {nuclide_name}({reaction_name})->{base_nuclide}_m{liso}: "
                        f"Multiple LFS map to same LISO. Keeping LFS={keeper_lfs} "
                        f"(ELIS={keeper_elis:.0f}eV, diff={keeper_diff:.0f}eV). "
                        f"Discarding: {discarded_str}",
                        UserWarning
                    )

                    # Store in processing errors for detailed logging
                    self._processing_errors.append({
                        'type': 'duplicate_mapping',
                        'nuclide': nuclide_name,
                        'reaction': reaction_name,
                        'mt': mt,
                        'liso': liso,
                        'base_nuclide': base_nuclide,
                        'kept_lfs': keeper_lfs,
                        'kept_elis': keeper_elis,
                        'kept_diff': keeper_diff,
                        'dk_elis': dk_elis,
                        'discarded': discarded_list,
                        'gendf_all_lfs': gendf_all_lfs,
                        'decay_all_liso': decay_all_liso,
                        'target_z': z,
                        'target_a': a,
                    })

            # Filter out duplicate mappings
            if indices_to_skip:
                elis_results = [
                    item for idx, item in enumerate(elis_results)
                    if idx not in indices_to_skip
                ]

            # Collect all ELIS-matched LISO values first (only 'matched', not 'nearest')
            assigned_lisos = set()
            for _, _, liso_result in elis_results:
                if liso_result is not None and liso_result.get('status') == 'matched':
                    assigned_lisos.add(liso_result['liso'])
                    # 'nearest' matches (beyond rtol) are skipped, not used

            # Second pass: Assign names, using fallback for unmatched
            for order_idx, (meta, elis, liso_result) in enumerate(elis_results, start=1):
                lfs = meta['lfs']
                sigma = meta['sigma']
                z = meta['z']
                a = meta['a']
                elis_str = f"{elis:.1f}" if elis is not None else "N/A"

                # Find available alternatives in decay library for logging
                decay_states = self.decay_lookup.get((z, a), [])
                available_metas = [s for s in decay_states if s.liso > 0]
                alternatives = [f"_m{s.liso} (ELIS={s.elis:.0f}eV)" for s in available_metas]

                if liso_result is None:
                    # Should not happen with return_nearest=True, but handle defensively
                    continue

                status = liso_result.get('status')

                if status == 'matched':
                    # ELIS match within tolerance - use LISO for naming
                    liso = liso_result['liso']
                    dk_elis = liso_result['dk_elis']
                    mapped_name = f"{base_nuclide}_m{liso}"
                    elis_info = {
                        'method': 'elis',
                        'elis': elis,       # GENDF-calculated ELIS
                        'dk_elis': dk_elis, # Decay library ELIS
                        'liso': liso
                    }

                elif status == 'nearest':
                    # Tolerance exceeded - skip product
                    liso = liso_result['liso']
                    dk_elis = liso_result['dk_elis']
                    diff_percent = liso_result.get('diff_pct', 0.0)

                    warnings.warn(
                        f"WARNING: ELIS_TOL_EXCEEDED: {nuclide_name}({reaction_name})->"
                        f"{base_nuclide} LFS={lfs} ELIS={elis_str} eV. "
                        f"Nearest _m{liso}: {dk_elis:.0f}eV ({diff_percent:.1f}% diff). "
                        f"Product skipped; branching will be renormalized.",
                        UserWarning
                    )
                    # Find half_life for the closest match
                    closest_state = next((s for s in available_metas if s.liso == liso), None)
                    closest_half_life = closest_state.half_life if closest_state else None
                    # Store for logging
                    self._processing_errors.append({
                        'type': 'elis_tol_exceeded',
                        'nuclide': nuclide_name,
                        'parent': nuclide_name,
                        'reaction': reaction_name,
                        'mt': mt,
                        'lfs': lfs,
                        'elis': elis,
                        'dk_elis': dk_elis,
                        'liso': liso,
                        'diff_percent': diff_percent,
                        'base_nuclide': base_nuclide,
                        'target_z': z,
                        'target_a': a,
                        'omitted': True,
                        'half_life': closest_half_life,
                    })
                    continue  # Skip adding to mapped_meta_levels

                elif status == 'zero_elis_only':
                    # Metastable states exist but all have ELIS=0 (data quality issue)
                    skipped_states = liso_result.get('skipped_states', [])
                    skipped_str = ', '.join(
                        f"_m{liso} (ELIS=0, T1/2={hl:.1f}s)" if hl else f"_m{liso} (ELIS=0)"
                        for liso, _, hl in skipped_states
                    )
                    warnings.warn(
                        f"WARNING: ZERO_ELIS_METASTABLES: {nuclide_name}({reaction_name})->"
                        f"{base_nuclide}_m? LFS={lfs} ELIS={elis_str} eV. "
                        f"Metastable state(s) exist but have ELIS=0.0 (data quality issue): "
                        f"{skipped_str}. "
                        f"Consider using a decay library with complete ELIS data. "
                        f"Product skipped; branching will be renormalized.",
                        UserWarning
                    )
                    # Store for logging
                    self._processing_errors.append({
                        'type': 'zero_elis_metastables',
                        'nuclide': nuclide_name,
                        'parent': nuclide_name,
                        'reaction': reaction_name,
                        'mt': mt,
                        'lfs': lfs,
                        'elis': elis,
                        'base_nuclide': base_nuclide,
                        'target_z': z,
                        'target_a': a,
                        'omitted': True,
                        'skipped_states': skipped_states,
                    })
                    continue  # Skip adding to mapped_meta_levels

                elif status in ('no_decay_data', 'no_metastables', 'no_match'):
                    # No usable metastable data:
                    # - no_decay_data: Nuclide not in decay library at all
                    # - no_metastables: Nuclide exists but only ground state
                    # - no_match: Outside tolerance (shouldn't happen with return_nearest=True)
                    if status == 'no_decay_data':
                        reason = f"Nuclide (Z={z}, A={a}) not found in decay library"
                    elif status == 'no_metastables':
                        reason = f"No metastable states in decay library for (Z={z}, A={a})"
                    else:
                        reason = f"No matching metastable state found"

                    warnings.warn(
                        f"WARNING: NO_METASTABLE_DECAY_DATA: {nuclide_name}({reaction_name})->"
                        f"{base_nuclide}_m? LFS={lfs} ELIS={elis_str} eV. "
                        f"{reason}. "
                        f"Product skipped; branching will be renormalized.",
                        UserWarning
                    )

                    # Store in processing errors for logging
                    self._processing_errors.append({
                        'type': 'no_metastable_decay_data',
                        'subtype': status,
                        'nuclide': nuclide_name,
                        'parent': nuclide_name,
                        'reaction': reaction_name,
                        'mt': mt,
                        'lfs': lfs,
                        'elis': elis,
                        'base_nuclide': base_nuclide,
                        'target_z': z,
                        'target_a': a,
                        'omitted': True,
                        'available_alternatives': alternatives,
                    })
                    continue  # Skip adding to mapped_meta_levels

                else:
                    # Unknown status - defensive handling
                    warnings.warn(
                        f"WARNING: Unknown ELIS lookup status '{status}' for "
                        f"{nuclide_name}({reaction_name})->{base_nuclide}_m? LFS={lfs}",
                        UserWarning
                    )
                    continue

                mapped_meta_levels.append((lfs, mapped_name, sigma, elis_info))
                lfs_mapping[mapped_name] = lfs

        elif self._mapping_mode == 'lfs_order':
            # ============================================================
            # LFS-ORDER MAPPING: FISPACT-like positional mapping
            # ============================================================
            # Map by sorted LFS position: 1st LFS → _m1, 2nd LFS → _m2, etc.
            # Count validation against decay library LISO count.

            # Get decay library metastable count for validation
            z = all_meta_levels[0]['z']
            a = all_meta_levels[0]['a']
            decay_states = self.decay_lookup.get((z, a), [])
            # Sort by LISO to ensure _m1, _m2, _m3 order
            dk_meta_states = sorted(
                [s for s in decay_states if s.liso > 0],
                key=lambda s: s.liso
            )
            dk_meta_count = len(dk_meta_states)
            gendf_meta_count = len(all_meta_levels)

            # Track dropped LFS states and orphan DK states
            dropped_lfs_states = []
            orphan_dk_states = []

            # Map by position with count validation
            for position, meta in enumerate(all_meta_levels, start=1):
                lfs = meta['lfs']
                sigma = meta['sigma']
                level = meta['level']

                # Calculate GENDF ELIS for reference logging
                qm = level.get('QM', qm_section)
                qi = level.get('QI', 0.0)
                gendf_elis = (qm - qi) if qm is not None else None

                if position > dk_meta_count:
                    # DROP this LFS - exceeds DK-Lib metastable count
                    dropped_lfs_states.append({
                        'lfs': lfs,
                        'position': position,
                        'would_be_liso': position,
                        'gendf_elis': gendf_elis,
                    })
                    self._processing_errors.append({
                        'type': 'lfs_order_dropped',
                        'nuclide': nuclide_name,
                        'parent': nuclide_name,
                        'reaction': reaction_name,
                        'mt': mt,
                        'lfs': lfs,
                        'position': position,
                        'would_be_liso': position,
                        'gendf_elis': gendf_elis,
                        'dk_meta_count': dk_meta_count,
                        'gendf_meta_count': gendf_meta_count,
                        'base_nuclide': base_nuclide,
                        'target_z': z,
                        'target_a': a,
                    })
                    warnings.warn(
                        f"LFS_ORDER_DROPPED: {nuclide_name}({reaction_name})->"
                        f"{base_nuclide}_m{position} LFS={lfs}. "
                        f"DK-Lib has only {dk_meta_count} metastable state(s). "
                        f"Product skipped; branching will be renormalized.",
                        UserWarning
                    )
                    continue

                # Map by position: position → LISO
                liso = position
                mapped_name = f"{base_nuclide}_m{liso}"

                # Get DK-Lib ELIS for reference (for logging)
                dk_elis = None
                dk_half_life = None
                if liso <= len(dk_meta_states):
                    # dk_meta_states is sorted by liso in parse_decay_isomeric_levels
                    dk_state = dk_meta_states[liso - 1]  # 0-indexed
                    dk_elis = dk_state.elis
                    dk_half_life = dk_state.half_life

                # Check ELIS reference for Ag116-type warnings
                elis_ref_status = None
                elis_ref_diff = None
                if gendf_elis is not None and dk_elis is not None:
                    elis_ref_diff = abs(gendf_elis - dk_elis)
                    # Check if ELIS would have matched via ELIS mode
                    if elis_match(gendf_elis, dk_elis, self._elis_rtol, self._elis_atol):
                        elis_ref_status = 'ok'
                    else:
                        elis_ref_status = 'mismatch'
                        # Check if ELIS would have matched a different LISO
                        elis_lookup_result = lookup_liso(
                            z, a, gendf_elis, self.decay_lookup,
                            rtol=self._elis_rtol, atol=self._elis_atol,
                            skip_zero_elis_metastables=self._skip_zero_elis_metastables,
                            return_nearest=True
                        )
                        if (elis_lookup_result is not None and
                                elis_lookup_result.get('status') == 'matched'):
                            elis_liso = elis_lookup_result['liso']
                            elis_dk_elis = elis_lookup_result['dk_elis']
                            if elis_liso != liso:
                                # Ag116-type case: ELIS would map to different LISO
                                elis_ref_status = 'wrong_liso'
                                warnings.warn(
                                    f"LFS_ORDER_ELIS_MISMATCH: {nuclide_name}({reaction_name})->"
                                    f"{mapped_name}: LFS-order maps LFS={lfs} to _m{liso}, "
                                    f"but ELIS matching would map to _m{elis_liso}. "
                                    f"(GENDF ELIS={gendf_elis:.0f}eV, DK _m{liso} ELIS={dk_elis:.0f}eV, "
                                    f"DK _m{elis_liso} ELIS={elis_dk_elis:.0f}eV). "
                                    f"Consider using mapping_mode='elis' for production.",
                                    UserWarning
                                )

                elis_info = {
                    'method': 'lfs_order',
                    'lfs': lfs,
                    'liso': liso,
                    'position': position,
                    'elis': gendf_elis,  # For consistency with ELIS mode
                    'gendf_elis': gendf_elis,  # Also keep explicit name
                    'dk_elis': dk_elis,
                    'dk_half_life': dk_half_life,
                    'dk_meta_count': dk_meta_count,
                    'gendf_meta_count': gendf_meta_count,
                    'elis_ref_status': elis_ref_status,
                    'elis_ref_diff': elis_ref_diff,
                }

                mapped_meta_levels.append((lfs, mapped_name, sigma, elis_info))
                lfs_mapping[mapped_name] = lfs

            # Report orphan DK-Lib states (DK has more metastables than GENDF)
            if dk_meta_count > gendf_meta_count:
                for i in range(gendf_meta_count, dk_meta_count):
                    dk_state = dk_meta_states[i]
                    orphan_dk_states.append({
                        'liso': dk_state.liso,
                        'dk_elis': dk_state.elis,
                        'dk_half_life': dk_state.half_life,
                    })
                    self._processing_errors.append({
                        'type': 'lfs_order_orphan_dk',
                        'nuclide': nuclide_name,
                        'parent': nuclide_name,
                        'reaction': reaction_name,
                        'mt': mt,
                        'liso': dk_state.liso,
                        'dk_elis': dk_state.elis,
                        'dk_half_life': dk_state.half_life,
                        'dk_meta_count': dk_meta_count,
                        'gendf_meta_count': gendf_meta_count,
                        'base_nuclide': base_nuclide,
                        'target_z': z,
                        'target_a': a,
                    })

        # Validate ground state exists (required for branching ratio calculation)
        if ground_data is None:
            meta_products_str = ', '.join(name for _, name, _, _ in mapped_meta_levels)
            raise ValueError(
                f"MF=10 data for {nuclide_name} MT={mt} has metastable state(s) "
                f"({meta_products_str}) but no ground state (LFS=0). This indicates "
                f"corrupted GENDF data - isomeric branching requires both ground "
                f"and metastable cross-sections."
            )

        # Use union of energy grids from ground + ALL metastable states
        all_energy_sets = [set(ground_data.x)]
        for _, _, sigma, _ in mapped_meta_levels:
            all_energy_sets.append(set(sigma.x))
        all_energies = sorted(set.union(*all_energy_sets))

        # Create lookup dictionaries for ground and all metastables
        ground_energy_to_xs = dict(zip(ground_data.x, ground_data.y))
        meta_energy_to_xs_list = []
        for _, _, sigma, _ in mapped_meta_levels:
            meta_energy_to_xs_list.append(dict(zip(sigma.x, sigma.y)))

        # Build energy-dependent branching ratios for N products
        # Products: [ground, meta1, meta2, ...]
        products = [ground_product] + [mapped for _, mapped, _, _ in mapped_meta_levels]
        n_products = len(products)

        energies_list = []
        # ratios_per_product[i] = list of ratios for product i
        ratios_per_product = [[] for _ in range(n_products)]

        for energy in all_energies:
            # Get cross-sections for ground and all metastables
            ground_xs = ground_energy_to_xs.get(energy, 0.0)
            meta_xs_values = [lookup.get(energy, 0.0) for lookup in meta_energy_to_xs_list]

            # H4 fix: Clamp negative XS values (numerical noise) to zero
            if ground_xs < 0:
                ground_xs = 0.0
            meta_xs_values = [max(0.0, xs) for xs in meta_xs_values]

            total_xs = ground_xs + sum(meta_xs_values)

            if total_xs > 0:
                energies_list.append(float(energy))
                ratios_per_product[0].append(ground_xs / total_xs)
                for i, meta_xs in enumerate(meta_xs_values, start=1):
                    ratios_per_product[i].append(meta_xs / total_xs)

        # Convert to arrays
        energies_array = np.array(energies_list)
        branching_array = np.array(ratios_per_product)  # Shape: [n_products, n_energies]

        # Validate computed branching ratios
        if len(energies_list) == 0:
            warnings.warn(
                f"No valid energy points with positive cross-section for "
                f"{nuclide_name} MT={mt}. Cannot compute branching ratios.",
                UserWarning
            )
            return None

        # Check for NaN/Inf values (indicates numerical issues)
        if not np.all(np.isfinite(branching_array)):
            nan_count = np.sum(~np.isfinite(branching_array))
            warnings.warn(
                f"Invalid values (NaN/Inf) found in branching ratios for "
                f"{nuclide_name} MT={mt}: {nan_count} out of "
                f"{branching_array.size} values. Replacing with 0.0.",
                UserWarning
            )
            branching_array = np.nan_to_num(branching_array, nan=0.0,
                                           posinf=0.0, neginf=0.0)

        # Validate sum to 1.0 at each energy point (internal consistency check)
        column_sums = branching_array.sum(axis=0)
        if not np.allclose(column_sums, 1.0, rtol=1e-10):
            # This should never happen given our computation logic
            # If it does, it indicates a bug in the code above
            bad_indices = np.where(~np.isclose(column_sums, 1.0, rtol=1e-10))[0]
            raise AssertionError(
                f"Internal error: Branching ratios for {nuclide_name} MT={mt} "
                f"do not sum to 1.0 at {len(bad_indices)} energy points. "
                f"Sum range: [{column_sums.min():.6f}, {column_sums.max():.6f}]. "
                f"This indicates a bug in the branching ratio calculation."
            )

        # Get reaction name
        reaction_name = MT_TO_REACTION.get(mt, f'MT{mt}')

        # lfs_mapping is already built during the mapping section above

        # Build elis_mapping from mapped_meta_levels (extract 4th tuple element)
        # Only include metastable products (not ground state)
        elis_mapping = {}
        for _, mapped_name, _, elis_info in mapped_meta_levels:
            elis_mapping[mapped_name] = elis_info

        return IsomericBranching(
            energies=energies_array,
            products=products,
            branching_ratios=branching_array,
            parent_nuclide=nuclide_name,
            reaction=reaction_name,
            mt=mt,
            lfs_mapping=lfs_mapping,
            elis_mapping=elis_mapping if elis_mapping else None
        )

    def process_library_for_branching(
        self,
        mt_list: Optional[List[int]] = None,
        progress_callback: Optional[callable] = None,
        verbose: bool = False,
        chain: Optional['Chain'] = None
    ) -> Dict[str, Dict[str, IsomericBranching]]:
        """Process entire GENDF library for isomeric branching data.

        Scans all nuclides in the library and extracts energy-dependent
        branching ratios for specified reactions that have MF=10 data.

        Parameters
        ----------
        mt_list : list of int, optional
            MT values to process. Default: all MTs from MT_TO_REACTION
            (all reactions defined in chain.py::REACTIONS)
        progress_callback : callable, optional
            Function called with (current, total, nuclide_name) after
            processing each nuclide
        verbose : bool, optional
            Print progress messages. Default is False.
        chain : openmc.deplete.Chain, optional
            If provided, only process reactions that exist in the chain for
            each nuclide. This ensures consistency between GENDF branching data
            and chain reactions. Unmatched MTs are tracked in `unmatched_mts`.

        Returns
        -------
        dict
            Nested dictionary with structure:
            {nuclide: {reaction: IsomericBranching}}
            Only includes nuclides and reactions with branching data.

        Examples
        --------
        >>> lib = GENDFLibrary('/path/to/JEFF40-GENDF/', 'UKAEA-1102')
        >>> branching_data = lib.process_library_for_branching(
        ...     mt_list=[16, 102],  # n,2n and n,gamma only
        ...     verbose=True
        ... )
        >>> print(f"Found branching for {len(branching_data)} nuclides")
        """
        if mt_list is None:
            mt_list = sorted(MT_TO_REACTION.keys())

        # Build chain reaction lookup if chain provided
        chain_reactions = None
        if chain is not None:
            chain_reactions = self._build_chain_reaction_lookup(chain)

        available_nuclides = self.available_nuclides()
        total = len(available_nuclides)

        if verbose:
            print(f"Processing {total} nuclides for branching data")
            if chain is not None:
                print(f"Chain-aware filtering enabled: {len(chain.nuclides)} nuclides in chain")
            print("=" * 60)

        all_branching_data = {}
        self._processing_errors = []  # Capture ELIS mismatch errors (missing metastable products)
        self._unmatched_mts = []  # Track reaction TYPES not in chain (nuclide/reaction missing)

        for idx, nuclide_name in enumerate(available_nuclides, start=1):
            if progress_callback:
                progress_callback(idx, total, nuclide_name)

            if verbose:
                print(f"\n[{idx}/{total}] Processing {nuclide_name}")
                print("-" * 40)

            try:
                has_branching = False
                temp_reactions = {}

                for mt in mt_list:
                    if mt not in MT_TO_REACTION:
                        continue

                    reaction_name = MT_TO_REACTION[mt]

                    # Chain-aware filtering: skip if reaction not in chain
                    if chain_reactions is not None:
                        if not self._reaction_exists_in_chain(
                            nuclide_name, reaction_name, chain_reactions
                        ):
                            # Track unmatched MT for logging
                            self._unmatched_mts.append({
                                'nuclide': nuclide_name,
                                'mt': mt,
                                'reaction': reaction_name,
                                'reason': 'not_in_chain'
                            })
                            continue

                    branching = self.get_branching_ratios(nuclide_name, mt)

                    if branching:
                        temp_reactions[reaction_name] = branching
                        has_branching = True
                        if verbose:
                            n_energies = len(branching.energies)
                            # Determine mapping method for logging
                            method_tag = ""
                            if branching.elis_mapping:
                                methods = set(m.get('method', 'unknown')
                                              for m in branching.elis_mapping.values())
                                if methods == {'elis'}:
                                    method_tag = " [ELIS]"
                                elif 'unmatched' in methods:
                                    method_tag = " [ELIS+RENORM]"
                                elif methods == {'order'}:
                                    method_tag = " [ORDER]"
                            print(f"  {nuclide_name} {reaction_name}: "
                                  f"{n_energies} energy points{method_tag}")
                            print(f"    Products: {branching.products}")

                # Only add to result if at least one branching reaction found
                if has_branching:
                    all_branching_data[nuclide_name] = temp_reactions

            except Exception as e:
                error_str = str(e)
                if verbose:
                    print(f"  {nuclide_name}: {e}")
                # Capture no_metastable_decay_data errors for logging
                if 'NO_METASTABLE_DECAY_DATA' in error_str:
                    self._processing_errors.append({
                        'nuclide': nuclide_name,
                        'error': error_str,
                        'type': 'no_metastable_decay_data'
                    })
                continue

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Found branching data for {len(all_branching_data)} nuclides")
            total_reactions = sum(len(rxs) for rxs in all_branching_data.values())
            print(f"Total reactions with branching: {total_reactions}")

        return all_branching_data

    @property
    def processing_errors(self) -> list:
        """Get ELIS mismatch errors from last process_library_for_branching() call.

        Returns
        -------
        list of dict
            Each dict contains 'nuclide', 'error' (full error message), and 'type'
            ('elis_mismatch' for ELIS matching failures).
        """
        return getattr(self, '_processing_errors', [])

    @property
    def unmatched_mts(self) -> list:
        """Get unmatched reaction TYPES from last process_library_for_branching().

        When chain-aware filtering is enabled (chain parameter provided),
        this property returns reaction TYPES (MTs) where GENDF has isomeric
        branching data but the chain doesn't have that reaction for that
        nuclide. This occurs when either:
        - The nuclide is not in the chain at all, OR
        - The nuclide is in the chain but doesn't have that specific reaction

        NOTE: This is different from processing_errors which tracks reactions
        that EXIST in the chain but have missing metastable product states.

        Returns
        -------
        list of dict
            Each dict contains 'nuclide', 'mt', 'reaction', and 'reason'.
        """
        return getattr(self, '_unmatched_mts', [])

    def _build_chain_reaction_lookup(self, chain) -> Dict[str, set]:
        """Build lookup dict mapping nuclide names to their reaction types.

        Parameters
        ----------
        chain : openmc.deplete.Chain
            Depletion chain to build lookup from

        Returns
        -------
        dict
            Mapping from nuclide name to set of reaction type strings
        """
        lookup = {}
        for nuclide in chain.nuclides:
            lookup[nuclide.name] = set()
            for reaction in nuclide.reactions:
                lookup[nuclide.name].add(reaction.type)
        return lookup

    def _reaction_exists_in_chain(
        self,
        nuclide_name: str,
        reaction_name: str,
        chain_reactions: Dict[str, set]
    ) -> bool:
        """Check if a reaction exists for a nuclide in the chain.

        Parameters
        ----------
        nuclide_name : str
            Name of the nuclide (e.g., 'U235')
        reaction_name : str
            Reaction type string (e.g., '(n,gamma)')
        chain_reactions : dict
            Lookup dict from _build_chain_reaction_lookup()

        Returns
        -------
        bool
            True if reaction exists in chain for this nuclide
        """
        if nuclide_name not in chain_reactions:
            return False
        return reaction_name in chain_reactions[nuclide_name]

    def __repr__(self) -> str:
        """String representation of the library."""
        n_nuclides = len(self._file_index)
        return (f"_PythonGENDFLibrary('{self.library_path}', "
                f"energy_structure='{self.energy_structure}', "
                f"n_nuclides={n_nuclides}, n_groups={self.n_groups})")

def GENDFLibrary(
    library_path: PathLike,
    energy_structure: str = 'UKAEA-1102',
    validate_energy_grid: bool = True,
    use_fast_parser: bool = True,
    backend: str = 'auto',
    decay_file: Optional[PathLike] = None,
    elis_rtol: float = ELIS_RTOL,
    elis_atol: float = ELIS_ATOL,
    skip_zero_elis_metastables: bool = True,
    mapping_mode: str = 'elis'
):
    """Create a GENDF cross-section library with automatic backend selection.

    This factory function automatically selects the best available backend
    for GENDF cross-section loading:
    - **C++ backend**: Fast implementation (2-5x speedup) if OpenMC was built
      with C++ GENDF support
    - **Python backend**: Pure Python fallback implementation

    Parameters
    ----------
    library_path : path-like
        Path to directory containing GENDF .asc files
    energy_structure : str, optional
        Name of the energy group structure. Must be one of 'CCFE-709' or
        'UKAEA-1102'. Default is 'UKAEA-1102'.
    validate_energy_grid : bool, optional
        If True, validate that each GENDF file's energy grid matches the
        specified energy structure. Default is True. Only used with Python
        backend.
    use_fast_parser : bool, optional
        If True, use optimized MF=3-only parser that skips covariance data.
        Default is True. Only used with Python backend.
    backend : {'auto', 'cpp', 'python'}, optional
        Backend selection:
        - 'auto': Automatically use C++ if available, otherwise Python (default)
        - 'cpp': Force C++ backend (raises error if not available)
        - 'python': Force Python backend
    decay_file : path-like, optional
        Path to ENDF decay library for ELIS-based isomeric state mapping.
        Can be a directory of decay files or a single concatenated file.
        When provided, enables accurate mapping of GENDF MF=10 metastable
        products to OpenMC ``_m{n}`` naming based on excitation energy
        matching. **Highly recommended** for isomeric branching workflows.
        Only used with Python backend.
    elis_rtol : float, optional
        Relative tolerance for ELIS matching (default: 0.01 = 1%).
        Only used with Python backend when decay_file is provided.
    elis_atol : float, optional
        Absolute tolerance in eV for ELIS matching (default: 100.0 eV).
        Only used with Python backend when decay_file is provided.
    skip_zero_elis_metastables : bool, optional
        If True, skip metastable states with ELIS=0 in decay library (likely
        data errors). Default is True. Only used with Python backend when
        decay_file is provided.
    mapping_mode : {'elis', 'lfs_order'}, optional
        Isomeric state mapping mode:
        - 'elis' (default): Use excitation energy (ELIS) matching between
          GENDF MF=10 products and decay library. Most accurate method.
        - 'lfs_order': Use FISPACT-like positional mapping where the 1st
          metastable LFS maps to _m1, 2nd to _m2, etc. Useful for validation
          testing against FISPACT-II. **Warning**: LFS order may not match
          LISO order for some nuclides (e.g., Ag116).
        Only used with Python backend.

    Returns
    -------
    GENDFLibrary
        Library instance (either C++ or Python implementation)

    Raises
    ------
    ValueError
        If requested backend is not available
    FileNotFoundError
        If library_path does not exist
    ValueError
        If energy_structure is not supported

    Examples
    --------
    >>> # Auto-select fastest backend (without ELIS mapping)
    >>> lib = GENDFLibrary('/path/to/JEFF40-GENDF/', 'UKAEA-1102')
    >>> xs = lib.get_xs('U235', 102, lib.energy_bounds)
    >>>
    >>> # With ELIS-based isomeric mapping (recommended for branching)
    >>> lib = GENDFLibrary(
    ...     '/path/to/JEFF40-GENDF/',
    ...     decay_file='/path/to/JEFF40-decay/'
    ... )
    >>> branching = lib.get_branching_ratios('Ir191', 102)
    >>> print(branching.products)  # ['Ir192', 'Ir192_m1', 'Ir192_m2']
    >>>
    >>> # Cross-library usage with relaxed tolerance
    >>> lib = GENDFLibrary(
    ...     '/path/to/TENDL2017-GENDF/',
    ...     decay_file='/path/to/UKDD12_decay.dat',
    ...     elis_rtol=0.10  # Allow 10% tolerance for cross-library mismatches
    ... )
    >>>
    >>> # Force Python backend
    >>> lib_py = GENDFLibrary('/path/to/JEFF40-GENDF/', backend='python')
    >>>
    >>> # Force C++ backend (error if not available)
    >>> lib_cpp = GENDFLibrary('/path/to/JEFF40-GENDF/', backend='cpp')

    Notes
    -----
    The C++ backend provides significant performance improvements:
    - 2-5x faster GENDF file parsing
    - Lower memory overhead
    - Better caching performance

    Both backends provide identical interfaces and results, so code using
    GENDFLibrary does not need to change when switching backends.

    **ELIS-Based Mapping** (Python backend only, when decay_file is provided):

    Instead of assuming LFS order equals LISO order, the library uses excitation
    energy (ELIS) matching:

    1. For each GENDF MF=10 metastable product, calculate ELIS = QM - QI
    2. Look up matching metastable state in decay library by ELIS
    3. Use decay library's LISO value for ``_m{n}`` naming

    This handles cases like Ir191(n,gamma)->Ir192 where GENDF uses LFS=3,15
    but decay library correctly identifies these as LISO=1,2 (m1, m2).

    .. versionadded:: 0.15.3
        C++ backend and automatic selection

    .. versionadded:: 0.15.3
        ELIS-based isomeric state mapping (decay_file, elis_rtol, elis_atol)

    See Also
    --------
    parse_decay_isomeric_levels : Parse decay library for ELIS data
    lookup_liso : Find LISO for given excitation energy
    DecayState : Data class for nuclear state information
    """
    # Validate backend choice
    if backend not in ('auto', 'cpp', 'python'):
        raise ValueError(
            f"Invalid backend '{backend}'. Must be 'auto', 'cpp', or 'python'")

    # Validate mapping_mode
    if mapping_mode not in ('elis', 'lfs_order'):
        raise ValueError(
            f"Invalid mapping_mode '{mapping_mode}'. Must be 'elis' or 'lfs_order'")

    # decay_file is optional - only needed for isomeric branching (MF=10)
    # Library can still be used for available_nuclides() and XS retrieval without it

    # Determine which backend to use
    use_cpp = False
    if backend == 'cpp':
        if not _CPP_BACKEND_AVAILABLE:
            raise ValueError(
                "C++ backend requested but not available. "
                "Ensure OpenMC was built with C++ GENDF support.")
        use_cpp = True
    elif backend == 'auto':
        # If decay_file is provided, force Python backend for ELIS mapping
        # C++ backend doesn't support ELIS mapping yet
        if decay_file is not None:
            use_cpp = False
        else:
            use_cpp = _CPP_BACKEND_AVAILABLE
    # else backend == 'python', use_cpp remains False

    # Warn if C++ backend requested but isomeric mapping parameters provided
    if use_cpp and decay_file is not None:
        warnings.warn(
            "C++ backend does not support isomeric state mapping (neither 'elis' "
            "nor 'lfs_order' mode). decay_file and mapping_mode parameters will "
            "be ignored. Use backend='python' to enable isomeric state mapping.",
            UserWarning
        )

    # Create appropriate backend
    if use_cpp:
        # Use C++ backend
        energy_bounds = GROUP_STRUCTURES[energy_structure]
        return _CppGENDFLibrary(
            str(library_path),
            energy_bounds,
            energy_structure
        )
    else:
        # Use Python backend
        return _PythonGENDFLibrary(
            library_path,
            energy_structure,
            validate_energy_grid,
            use_fast_parser,
            decay_file,
            elis_rtol,
            elis_atol,
            skip_zero_elis_metastables,
            mapping_mode
        )


# Export public interface and backend classes (for type checking)
__all__ = [
    'GENDFLibrary',
    '_PythonGENDFLibrary',
    '_CppGENDFLibrary',
    'IsomericBranching',
    'DecayState',
    'get_target_name',
    'get_product_name',
    'detect_energy_structure',
    'parse_decay_isomeric_levels',
    'lookup_liso',
    'elis_match',
    'ATOMIC_SYMBOL',
    'MT_TO_REACTION',
    'ELIS_RTOL',
    'ELIS_ATOL'
]
