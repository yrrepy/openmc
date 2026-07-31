"""GENDF Cross-Section Library Module

This module provides functionality for reading and using GENDF cross-section
libraries. GENDF files contain pre-processed, group-averaged cross-sections
optimized for activation calculations.

The module supports:
- Loading GENDF libraries (ENDF-6 format files)
- Extracting cross-sections for specific nuclides and reactions
- Validating energy group structures (CCFE-709, UKAEA-1102)
- Caching for efficient repeated access
- Automatic selection of C++ (fast) or Python (fallback) backend

.. versionadded:: 0.15.4
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING
from dataclasses import dataclass
import io
import itertools
import warnings

import numpy as np
import endf

from openmc.checkvalue import check_type, check_value, PathLike
from openmc.exceptions import OpenMCError
from openmc.mgxs import GROUP_STRUCTURES
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

if TYPE_CHECKING:
    from openmc.deplete.chain import Chain


# Supported energy group structures for GENDF libraries
SUPPORTED_GROUP_STRUCTURES = {'CCFE-709', 'UKAEA-1102'}

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

def _parse_endf_material(filepath):
    """Parse an ENDF-format file, suppressing benign MF/MT warnings.

    WORKAROUND -- MF=40 (covariance) lines are stripped before parsing.

    This is a workaround for a bug in the ``endf`` package: ``parse_mf40``
    (endf/mf40.py) loops exactly ``NS`` times, where ``NS`` is the subsection
    count declared in the section HEAD record, with no SEND-record or EOF
    guard. When a file's HEAD over-declares ``NS``, the parser reads past the
    end of the section and eventually crashes with
    ``ValueError: invalid literal for int() with base 10: ''``.

    The EAF-2010 GENDF files trigger this: 97 of 816 files (e.g. Ag105g.asc,
    Am242mg.asc) have a final MF=40 section whose HEAD over-declares ``NS``.
    Because ``endf.Material.__init__`` has no per-section error isolation, that
    single bad section aborts the whole file load -- taking the fully intact
    MF=1/3/8/10 sections we actually need down with it.

    MF=40 covariance data is never used by this module, so dropping those lines
    is lossless here (and marginally speeds up parsing). Lines are identified by
    the ENDF column MF field, ``line[70:72]``. Line 0 (the TPID record) is always
    kept because ``Material`` unconditionally discards the first line.

    REMOVE THIS WORKAROUND once the ``endf`` package is fixed -- either by
    bounding the ``parse_mf40`` loop with a SEND/EOF check, or by adding a
    ``sections=`` MF-filter parameter to ``Material.__init__`` so unwanted
    sections are never parsed.
    """
    with open(filepath, 'r') as fh:
        lines = fh.readlines()

    filtered = ''.join(
        line for i, line in enumerate(lines)
        if i == 0 or line[70:72] != '40'
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"MF=\d+, MT=\d+ ignored", category=UserWarning
        )
        return endf.Material(io.StringIO(filtered))


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

# If C++ backend available, use it
# Try C++ backend, fall back to Python
try:
    from openmc.lib.gendf import GENDFLibrary as _CppGENDFLibrary
except (ImportError, AttributeError):
    _CppGENDFLibrary = None

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
        Level number (0=ground state, >0=excited state)

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
            "Valid range is 1-100.",
            UserWarning
        )
        return None

    symbol = ATOMIC_SYMBOL[Z]

    if lfs == 0:
        return f"{symbol}{A}"
    else:
        return f"{symbol}{A}_m{lfs}"


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
    2. Hydrogen isotopes              - lightest nuclides
    3. All other files                - systematic search

    This avoids failures when alphabetically-first files contain only
    threshold reactions with partial energy coverage.

    """
    library_path = Path(library_path)

    # Priority: actinides and H have full energy range (avoid threshold reactions)
    patterns = ['*U235*.asc', '*Pu239*.asc', '*H1*.asc', '*H2*.asc', '*.asc']

    seen = set()
    files_to_try = []
    for pattern in patterns:
        for f in sorted(library_path.glob(pattern))[:5]:
            if f not in seen:
                files_to_try.append(f)
                seen.add(f)

    if not files_to_try:
        raise FileNotFoundError(f"No .asc files found in {library_path}")

    for sample_file in files_to_try:
        try:
            material = _parse_endf_material(sample_file)
            mf3 = [(mf, mt) for mf, mt in material.section_data if mf == 3]
            if not mf3:
                continue

            # Extract energy grid from sigma data
            sigma = material.section_data[mf3[0]].get('sigma')
            if sigma is None or not hasattr(sigma, 'x'):
                continue

            n_groups = len(sigma.x)
            for name in SUPPORTED_GROUP_STRUCTURES:
                ref = GROUP_STRUCTURES[name]
                if len(ref) == n_groups and np.allclose(sigma.x, ref, rtol=GENDF_RTOL_MATCH):
                    return name
        except Exception:
            continue

    raise ValueError(
        f"Could not detect energy structure from {len(files_to_try)} files in {library_path}. "
        f"Supported: {SUPPORTED_GROUP_STRUCTURES}"
    )


@dataclass
class IsomericBranching:
    """Class for energy-dependent isomeric branching data.

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
        - 'method': 'elis' (matched via excitation energy) or 'lfs_order'
          (positional mapping based on sorted LFS values)
        - 'elis': Excitation energy in eV (from GENDF QM-QI calculation)
        - 'liso': Isomeric state number (1 for _m1(m), 2 for _m2(n), etc.)
        - 'qm', 'qi': that LEVEL's own MF=10 Q pair in eV (ELIS = QM - QI), read
          off its TAB1 head. The chain patcher writes the per-pathway Q straight
          from these: QI for a metastable slot. Both are always present in
          'elis' mode (ELIS is computed from them, and a level with no ELIS
          never maps); in 'lfs_order' mode they are carried verbatim and may be
          None if a stand-in parser omits them.
        - For 'lfs_order' mode, additional fields: 'lfs' (original LFS value),
          'position' (sorted position), 'elis_ref_status' ('ok', 'mismatch', or
          'wrong_liso' indicating ELIS check result for reference)
        Example: {'Ir192_m1': {'method': 'elis', 'elis': 56720.0, 'liso': 1,
        'qm': 6198000.0, 'qi': 6141280.0}}
    ground_qm : float, optional
        QM of the section's OWN LFS=0 subsection, in eV -- the Q the patcher
        writes into the ground (LFS=0) pathway slot. QM is per-subsection, not
        per-section: evaluations do disagree between the ground and metastable
        subsections of one MT (EAF-2010 metastable-parent MT=4 puts +E(parent)
        on LFS=0 and 0.0 on the metastable levels), so a sibling's QM is not a
        substitute. None when the section has no LFS=0 subsection at all (the
        ground was synthesized from the MF=3 remainder, or the file is
        metastable-only); the patcher then falls back to a mapped level's QM.

    """
    energies: np.ndarray
    products: list[str]
    branching_ratios: np.ndarray
    parent_nuclide: str
    reaction: str
    mt: int
    lfs_mapping: Optional[dict[str, int]] = None
    elis_mapping: Optional[dict[str, dict[str, Any]]] = None
    ground_qm: Optional[float] = None


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
            'mt': self.mt,
            'lfs_mapping': self.lfs_mapping,
            'elis_mapping': self.elis_mapping,
            'ground_qm': self.ground_qm
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
            mt=data['mt'],
            lfs_mapping=data.get('lfs_mapping'),
            elis_mapping=data.get('elis_mapping'),
            ground_qm=data.get('ground_qm')
        )


# Dedup store for once-per-key runtime-branching warnings
_WARNED_RUNTIME_BRANCHING: set = set()

# Dedup store for once-per-(nuclide, MT) MF=10 duplicate-LFS warnings
_WARNED_MF10_DUPLICATE_LFS: set = set()

# Warn-once identities for libraries with no path (stubs, in-memory doubles);
# a monotonic counter, mirroring src/gendf.cpp's g_next_lib_id
_ANONYMOUS_LIBRARY_IDS = itertools.count()


def _library_warn_key(library):
    """Stable warn-once identity for a library: its path, else a counter.

    Deliberately not ``id()``: CPython reuses an address after GC, so a later
    library inherits a freed one's key and loses its warning. Keying on the path
    also collapses the fresh-library-per-call pattern in
    :mod:`openmc.deplete.microxs` to a single warning, while two libraries with
    different paths always both warn.
    """
    key = getattr(library, '_warn_key', None)
    if key is None:
        path = getattr(library, 'library_path', None)
        key = (str(path) if path is not None
               else f"anonymous#{next(_ANONYMOUS_LIBRARY_IDS)}")
        library._warn_key = key
    return key


def _warn_runtime_branching(key, message):
    """Emit a runtime-branching UserWarning once per key."""
    if key not in _WARNED_RUNTIME_BRANCHING:
        _WARNED_RUNTIME_BRANCHING.add(key)
        warnings.warn(message, UserWarning)


def build_runtime_branching(levels, target_names, lfs_values, energy_bounds,
                            nuclide_name, mt, total_xs=None, library_key=None):
    """Build runtime-mode IsomericBranching from aligned production XS.

    Shared by the Python and C++ backends: both fetch per-level MF=10
    production XS aligned to the full group grid and the reaction's rate cross
    section from ``get_xs``, and feed both here, so the branching-ratio
    computation cannot diverge between backends.

    The denominator sums production XS over the chain-requested LFS levels only
    (plus the file's ground when not requested), so yield to any untracked file
    levels is reattributed onto the tracked isomers -- conserving total reaction
    rate into the chain and matching the patcher convention.

    Parameters
    ----------
    levels : list of (int, int, numpy.ndarray)
        (lfs, izap, xs) tuples with xs aligned to the full group grid
    target_names : list of str
        Product names from chain, ordered as lfs_values
    lfs_values : list of int
        LFS values corresponding to target_names
    energy_bounds : numpy.ndarray
        Full group-structure boundaries in eV, length n_groups + 1
    nuclide_name : str
        Parent nuclide name
    mt : int
        ENDF MT number
    total_xs : numpy.ndarray, optional
        The reaction's rate cross section (MF=3 total, or Sigma(MF=10) for
        MF=10-only reactions), per group. Used only to synthesize a missing
        requested ground row as the clamped remainder (policy 3(a)): the
        remainder is taken against the same array the rate uses, so BR*rate
        reproduces the per-state MF=10 XS exactly -- and inversion cannot occur
        -- under two conditions: (1) the requested LFS set covers every
        metastable level present in the file (an unrequested level's yield is
        reattributed onto the tracked isomers, inflating them), and (2) no
        group clamps (Sigma sigma_MF10_m <= total_xs everywhere). A clamped
        group gives the metastables 100% of that group's rate. When MF=3 is
        absent, ``get_xs`` serves Sigma(MF=10) and the remainder is zero in
        EVERY group: the ground gets no cross section at all and the whole
        reaction rate branches to the metastable level(s) (warned distinctly).
    library_key : hashable, optional
        Stable identity of the calling library -- its path, see
        :func:`_library_warn_key` -- folded into the warn-once dedup keys so a
        second, different library in the same process still warns. None (direct
        helper calls) keeps the plain (kind, nuclide, mt) key.

    Returns
    -------
    IsomericBranching or None
        None when no production levels are usable or the requested ground
        state is absent from the file and no rate cross section is available to
        repair it (runtime branching disabled, warned)
    """
    def _key(kind):
        return ((kind, nuclide_name, mt) if library_key is None
                else (kind, nuclide_name, mt, library_key))

    if not levels:
        _warn_runtime_branching(
            _key('no_levels'),
            f"{nuclide_name} MT={mt}: no usable MF=10 production levels; "
            "runtime isomeric branching is unavailable, falling back to the "
            "chain's static branching.")
        return None

    # NJOY noise can leave tiny negative MF=10 production values; clamp to 0 so
    # they never enter the numerator/denominator (patcher and C++ clamp too).
    lfs_to_xs = {lfs: np.maximum(xs, 0.0) for lfs, izap, xs in levels}
    n_groups = len(energy_bounds) - 1

    # Ground requested but absent from the file. A radioactive-products-only
    # MF=10 omits a (quasi-)stable ground, so when the rate cross section is
    # available the ground channel is its clamped remainder (policy 3(a));
    # without it a zeros row would hand 100% of the rate to the metastable, so
    # runtime branching is disabled instead (R1-61).
    if 0 in lfs_values and 0 not in lfs_to_xs:
        if total_xs is None:
            _warn_runtime_branching(
                _key('ground_missing'),
                f"{nuclide_name} MT={mt}: chain requested the ground state "
                "(LFS=0) but the GENDF file has no LFS=0 production level; "
                "proceeding would send 100% of the reaction rate to the "
                "metastable, so runtime isomeric branching is disabled for this "
                "reaction, falling back to the chain's static branching.")
            return None
        # Every metastable in the FILE is subtracted, not only the requested
        # ones: yield to untracked levels is reattributed onto the tracked
        # isomers (the patcher convention), never left in the ground channel.
        file_meta = [xs for lfs, xs in lfs_to_xs.items() if lfs != 0]
        remainder = np.asarray(total_xs, dtype=float)
        if file_meta:
            remainder = remainder - np.sum(file_meta, axis=0)
        ground_row = np.maximum(remainder, 0.0)
        lfs_to_xs[0] = ground_row
        n_clamped = int(np.count_nonzero(np.asarray(remainder) < 0))
        clamp_note = (
            f"; remainder clamped to zero in {n_clamped} of "
            f"{ground_row.size} groups (Sigma sigma_MF10_m exceeds the "
            "rate cross section there; metastables receive 100% in those "
            "groups)" if n_clamped else "")
        if np.all(ground_row == 0):
            # Two data shapes reach an all-zero remainder, so the message states
            # the observable and names both: no MF=3 (get_xs serves Sigma(MF=10),
            # zero by construction), or an MF=3 that under-sums its partials.
            _warn_runtime_branching(
                _key('ground_missing_zero_remainder'),
                f"{nuclide_name} MT={mt}: the GENDF file has no LFS=0 "
                "production level, and the reaction's rate cross section is at "
                "or below the sum of its MF=10 metastable levels in EVERY "
                "group, so the synthesized ground receives ZERO cross section "
                "and 100% of the reaction rate branches to the metastable "
                "level(s). Typical of MF=10-only reactions where get_xs serves "
                "Sigma(MF=10); also occurs when a present MF=3 under-sums its "
                f"MF=10 partials{clamp_note}.")
        else:
            _warn_runtime_branching(
                _key('ground_missing_repaired'),
                f"{nuclide_name} MT={mt}: the GENDF file has no LFS=0 "
                "production level (radioactive-products-only MF=10), so the "
                "requested ground state is synthesized from the reaction's "
                "rate cross section as "
                f"max(0, sigma_total - sum sigma_MF10_m){clamp_note}.")

    # Anonymous levels are keyed by LFS alone; the chain names the products.
    used_lfs = set(lfs_values) | ({0} if 0 in lfs_to_xs else set())
    anonymous = sorted(lfs for lfs, izap, _ in levels
                       if izap == 0 and lfs in used_lfs)
    if anonymous:
        _warn_runtime_branching(
            _key('anonymous_izap'),
            f"{nuclide_name} MT={mt}: isomeric branching is served by "
            f"anonymous (IZAP=0) MF=10 level(s) LFS={anonymous}, keyed by LFS "
            "alone; product identities are taken from the chain.")

    prod_xs = []
    for lfs in lfs_values:
        if lfs in lfs_to_xs:
            prod_xs.append(lfs_to_xs[lfs])
        else:
            # Chain requested a level the GENDF file lacks -> BR=0. Warn once.
            prod_xs.append(np.zeros(n_groups))
            _warn_runtime_branching(
                _key('lfs_missing'),
                f"{nuclide_name} MT={mt}: chain requested LFS={lfs} but the "
                "GENDF file has no such production level; its branching ratio "
                "is set to 0.")

    prod_xs = np.array(prod_xs)  # (n_targets, n_groups)

    # Branching ratios: BR_i = σ_prod_i / Σ σ_prod_j. The denominator must
    # include the ground state (LFS=0), which the chain almost always
    # requests; a metastable-only request would otherwise normalize the
    # subset to 1.0 -> wrong physics.
    total = prod_xs.sum(axis=0)
    if 0 not in lfs_values:
        if 0 in lfs_to_xs:
            total = total + lfs_to_xs[0]
        else:
            _warn_runtime_branching(
                _key('no_ground'),
                f"{nuclide_name} MT={mt}: no ground-state (LFS=0) production in "
                "the GENDF file and none requested; branching ratios normalize "
                "over metastables only and may not reflect absolute yields.")
    with np.errstate(divide='ignore', invalid='ignore'):
        br = np.where(total > 0, prod_xs / total, 0.0)

    reaction = MT_TO_REACTION.get(mt, f'MT{mt}')
    lfs_mapping = {name: lfs for name, lfs
                   in zip(target_names, lfs_values) if lfs > 0}

    return IsomericBranching(
        energies=energy_bounds[:-1].copy(),
        products=list(target_names),
        branching_ratios=br,
        parent_nuclide=nuclide_name,
        reaction=reaction,
        mt=mt,
        lfs_mapping=lfs_mapping,
    )


class _PythonGENDFLibrary:
    """Python implementation of GENDF cross-section library.

    This class provides access to pre-processed group-averaged cross-sections
    from GENDF libraries. It handles loading ENDF-6 files, caching materials,
    and extracting cross-section data for specific nuclides and reactions.

    .. note::
        Users should use the :class:`GENDFLibrary` factory function instead of
        instantiating this class directly. The factory automatically selects
        the fastest available backend (C++ or Python).

    Parameters
    ----------
    library_path : path-like
        Path to directory containing GENDF .asc files
    validate_energy_grid : bool, optional
        If True, validate that each GENDF file's energy grid matches the
        auto-detected energy structure. Default is True.
    decay_file : path-like, optional
        Path to ENDF decay library for ELIS-based isomeric state mapping.
        Can be a directory of decay files or a single concatenated file.
        When provided, enables accurate mapping of GENDF MF=10 metastable
        products to OpenMC ``_m{n}`` naming based on excitation energy
        matching. **Highly recommended** for isomeric branching workflows.
    elis_rtol : float, optional
        Relative tolerance for ELIS matching (default: 0.50 = 50%).
        Used with ``elis_atol`` to determine if GENDF and decay ELIS values match.
    elis_atol : float, optional
        Absolute tolerance in eV for ELIS matching (default: 0.0, rtol-only).
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

    Notes
    -----
    GENDF files use naming convention with suffixes:
    - 'g' for ground state (e.g., Ir192g.asc)
    - 'm' for first metastable state (e.g., Ir192mg.asc)
    - 'n' for second metastable state (e.g., Ir192ng.asc)
    - and so on...

    The energy grid in GENDF files represents group boundaries. Cross-section
    values are group-averaged (integrated over each group).

    ELIS-Based Mapping:
    Decay_file library is provided.
    Use excitation energy (GENDF-ELIF to DK-ELIS) matching.
    (FISPACT assumes LFS order equals LISO order.)

    1. For each GENDF MF=10 metastable product, calculate ELIS = QM - QI
    2. Look up matching metastable state in decay library by ELIS
    3. Use decay library's LISO value for ``_m{n}`` naming

    This handles cases like Ir191(n,gamma)->Ir192 where GENDF uses LFS=3,15
    but decay library identifies these as LISO=1,2 (m1, m2).

    .. warning::
        Python backend is NOT thread-safe, is used for build isomeric chains.
        The C++ backend is recommended for multi-threaded applications and
        on-the-fly cross-section retrieval during activation.

    See Also
    --------
    parse_decay_isomeric_levels : Parse decay library for ELIS data
    lookup_liso : Find LISO for given excitation energy
    DecayState : Data class for nuclear state information
    """

    def __init__(
        self,
        library_path: PathLike,
        validate_energy_grid: bool = True,
        decay_file: Optional[PathLike] = None,
        elis_rtol: float = ELIS_RTOL,
        elis_atol: float = ELIS_ATOL,
        skip_zero_elis_metastables: bool = True,
        mapping_mode: str = 'elis',
        _energy_structure: Optional[str] = None
    ):
        check_type('library_path', library_path, (str, Path))
        check_type('validate_energy_grid', validate_energy_grid, bool)
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

        if _energy_structure is not None:
            energy_structure = _energy_structure
        else:
            energy_structure = detect_energy_structure(self.library_path)

        self.energy_structure = energy_structure
        self.energy_bounds = GROUP_STRUCTURES[energy_structure].copy()
        self.n_groups = len(self.energy_bounds) - 1
        self._validate_energy_grid = validate_energy_grid

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

        # Skip redundant energy validation after first successful check
        self._energy_validated = False

        # Cache for loaded ENDF materials
        # Key: nuclide name (OpenMC format), Value: endf.Material object OR dict
        self._material_cache = {}

        # Negative cache for files that failed to parse, keyed by
        # (nuclide, require_full_parser): the fast MF=3 scanner and the full
        # endf parser are different code paths, so a full-parser crash must not
        # condemn the fast read (and vice versa). Values are the failure message;
        # every hit raises a fresh RuntimeError, never a stored exception object.
        self._material_load_failures = {}

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

        GENDF metastable naming convention:
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

        # GENDF suffix to OpenMC metastable level mapping
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
            for gendf_suffix, omc_suffix in METASTABLE_SUFFIXES.items():
                if filename.endswith(gendf_suffix):
                    metastable_suffix = gendf_suffix
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
                self._register_file(preliminary_name, filepath)
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

                self._register_file(nuclide_name, filepath)

    def _register_file(self, name: str, filepath: Path):
        """Index name->filepath, raising if two files map to the same nuclide."""
        existing = self._file_index.get(name)
        if existing is not None and existing != filepath:
            raise ValueError(
                f"GENDF library at {self.library_path} maps two files to "
                f"nuclide '{name}': '{existing.name}' and '{filepath.name}'. "
                "Remove or rename one so the cross sections are unambiguous.")
        self._file_index[name] = filepath

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
        from endf.records import float_endf, int_endf

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
        if preliminary_name not in self._pending_metastable:
            # Not a pending metastable or already validated
            return preliminary_name

        filepath = self._file_index[preliminary_name]

        try:
            # Load material to get accurate naming from metadata
            # Use full parser to ensure MF=1, MT=451 is available
            material = _parse_endf_material(filepath)
            correct_name = get_target_name(material)

            # Update index if name changed
            if correct_name != preliminary_name:
                # A LISO-corrected name that lands on an existing entry is the
                # same collision as in _build_file_index: raise rather than
                # silently overwrite another file's cross sections.
                self._register_file(correct_name, filepath)
                # Remove old (preliminary) entry
                del self._file_index[preliminary_name]
                # Invalidate nuclides set cache (keys changed)
                self._nuclides_set_cache = None

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
            Nuclide name in OpenMC format (e.g., 'Ac225', 'Ag110_m1')
        require_full_parser : bool, optional
            If True, forces use of the full endf.Material parser.
            Required for accessing MF=10 data (isomeric branching).
            Default is False.

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

        # Positive cache first: a cached full endf.Material satisfies BOTH
        # request kinds, so it must outrank a stale fast-parse failure marker
        # left by an earlier attempt on the same file.
        if nuclide_name in self._material_cache:
            cached = self._material_cache[nuclide_name]
            # A fast-parser dict cannot serve a full-parser request; reload.
            if not (require_full_parser
                    and not isinstance(cached, endf.Material)):
                return cached

        # A parse that already failed is not retried: per-MT isolation would
        # otherwise re-parse a crasher file once per chain MT. Only the message
        # is stored -- re-raising one exception object would append propagation
        # frames on every retry and pin the parser stack for the library's life.
        if not hasattr(self, '_material_load_failures'):
            self._material_load_failures = {}
        failed = self._material_load_failures.get(
            (nuclide_name, require_full_parser))
        if failed is not None:
            raise RuntimeError(failed)

        # Resolve the file from the exact-name index built by _build_file_index
        filepath = None

        if nuclide_name in self._file_index:
            filepath = self._file_index[nuclide_name]
        elif '_m' in nuclide_name:
            # Metastable rescue: the requested level may still be pending
            # validation under a different preliminary _m level.
            import re
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
            if require_full_parser:
                # Use full endf.Material parser (slower but complete)
                # Required for MF=10 (isomeric branching) data
                material = _parse_endf_material(filepath)

                # Validate energy grid if requested
                if self._validate_energy_grid:
                    self._validate_material_energy_grid(material, nuclide_name)
            else:
                # Use optimized MF=3-only parser (3-4x faster)
                section_data = self._parse_gendf_mf3_only(filepath)

                # Create a simple object to store section_data
                # (compatible with the rest of the code)
                class _FastMaterial:
                    def __init__(self, data):
                        self.section_data = data

                material = _FastMaterial(section_data)

        except Exception as e:
            message = (
                f"Failed to load GENDF file for {nuclide_name}: {filepath}\n"
                f"Error: {e}")
            self._material_load_failures[nuclide_name, require_full_parser] = message
            raise RuntimeError(message)


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
        energy_bounds: Optional[np.ndarray] = None,
        strict_alignment: bool = True
    ) -> np.ndarray:
        """Get group-averaged cross-section for a nuclide and reaction.

        Parameters
        ----------
        nuclide_name : str
            Nuclide name in OpenMC format (e.g., 'U235', 'Ac225')
        mt : int
            ENDF MT number for the reaction
        energy_bounds : numpy.ndarray, optional
            Energy group boundaries in eV. If None, uses the library's
            energy structure. If provided, must match the library's structure.
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

        Notes
        -----
        For threshold reactions like (n,2n) with threshold ~6 MeV, GENDF files
        only contain data above the threshold. The start and end energies must
        align exactly with library group boundaries for accurate placement.
        """
        if energy_bounds is None:
            energy_bounds = self.energy_bounds
        elif not self._energy_validated:
            if not np.allclose(energy_bounds, self.energy_bounds,
                              rtol=GENDF_RTOL_MATCH, atol=GENDF_ATOL):
                raise ValueError(
                    "Provided energy bounds do not match library energy "
                    f"structure '{self.energy_structure}'")
            self._energy_validated = True

        # Load material
        material = self._load_material(nuclide_name)

        # Normal path is MF=3; if absent, fall back to Σ(MF=10 partials), the
        # total for reactions stored MF=10-only (EAF-2010 isomer producers).
        if (3, mt) not in material.section_data:
            levels = self._production_levels(nuclide_name, mt)
            if levels:
                return np.sum([xs for _, _, xs in levels], axis=0)
            raise KeyError(
                f"Reaction MT={mt} not found for {nuclide_name} in GENDF library. "
                f"Available reactions: {[mt for mf,mt in material.section_data.keys() if mf==3]}")

        xs_data = material.section_data[3, mt]
        return self._extract_xs(xs_data, nuclide_name, mt, strict_alignment)

    def get_all_xs(
        self,
        nuclide_name: str,
        *,
        strict_alignment: bool = True,
        mts: Optional[list[int]] = None
    ) -> dict[int, np.ndarray]:
        """Get all available cross-sections for a nuclide.

        Loads the material once and extracts all MF=3 reactions in a single
        pass. More efficient than calling :meth:`get_xs` per reaction.

        Parameters
        ----------
        nuclide_name : str
            Nuclide name in OpenMC format (e.g., 'U235', 'Ac225')
        strict_alignment : bool, optional
            If True (default), raise ValueError for threshold alignment
            failures. See :meth:`get_xs` for details.
        mts : list of int, optional
            If provided, only extract these MT numbers. Default: all MF=3.

        Returns
        -------
        dict of int to numpy.ndarray
            Mapping of MT number to group-averaged cross-section array.
            Each array has length n_groups.
        """
        material = self._load_material(nuclide_name)
        mts_set = set(mts) if mts is not None else None
        result = {}
        for (mf, mt), xs_data in material.section_data.items():
            if mf != 3:
                continue
            if mts_set is not None and mt not in mts_set:
                continue
            if 'sigma' not in xs_data:
                continue
            result[mt] = self._extract_xs(
                xs_data, nuclide_name, mt, strict_alignment)

        # MF=10-only fallback: for explicitly requested MTs with no MF=3 section
        # (EAF-2010 isomer producers), serve Σ(MF=10 partials) as the total.
        if mts_set is not None:
            for mt in mts_set:
                if mt in result:
                    continue
                levels = self._production_levels(nuclide_name, mt)
                if levels:
                    result[mt] = np.sum([xs for _, _, xs in levels], axis=0)
        return result

    def _rate_total_xs(self, nuclide_name, mt):
        """The reaction's rate XS (MF=3 total, else Σ(MF=10)); None on failure.

        Deliberately get_xs(): the missing-ground remainder must be taken
        against the very array the rate machinery serves, so the repair stays
        self-consistent and matches the C++ backend on the same file.
        """
        try:
            return self.get_xs(nuclide_name, mt, strict_alignment=False)
        except (KeyError, ValueError, OpenMCError) as exc:
            _warn_runtime_branching(
                ('rate_xs_unavailable', nuclide_name, mt,
                 _library_warn_key(self)),
                f"{nuclide_name} MT={mt}: could not obtain the rate cross "
                f"section ({exc!r}); runtime ground repair unavailable.")
            return None

    def _production_levels(self, nuclide_name, mt):
        """Σ-fallback MF=10 levels for a missing MT; [] on any load failure."""
        try:
            return self._get_production_xs(nuclide_name, mt)
        except Exception:
            return []

    def _extract_xs(self, xs_data, nuclide_name, mt, strict_alignment):
        """Extract and align XS from a single MF=3 section. Returns a copy."""
        if 'sigma' not in xs_data:
            raise ValueError(
                f"No 'sigma' data in MF=3, MT={mt} for {nuclide_name}")

        sigma = xs_data['sigma']
        xs = self._align_to_group_grid(
            sigma.x, sigma.y, f"{nuclide_name} MT={mt}", strict_alignment)
        # Clamp negative values to zero, matching the C++ parser
        negative = xs < 0
        if negative.any():
            warnings.warn(
                f"Clamping {negative.sum()} negative XS values to zero in "
                f"MF=3, MT={mt} for {nuclide_name}", UserWarning)
            xs[negative] = 0.0
        return xs

    def _align_to_group_grid(self, gendf_energies, gendf_xs, context,
                             strict_alignment):
        """Align a section's (energies, xs) onto the full group grid. Returns a copy."""
        # Full energy range
        if len(gendf_energies) == len(self.energy_bounds):
            return gendf_xs[:self.n_groups].copy()

        # Partial energy range (threshold reaction)
        xs_values = np.zeros(self.n_groups)

        start_energy = gendf_energies[0]
        end_energy = gendf_energies[-1]

        # Try exact match for start boundary
        start_matches = np.where(np.isclose(
            self.energy_bounds, start_energy,
            rtol=GENDF_RTOL_MATCH, atol=GENDF_ATOL))[0]

        if len(start_matches) == 1:
            start_idx = start_matches[0]
        elif len(start_matches) > 1:
            warnings.warn(
                f"Multiple energy boundary matches for {context} "
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
                    f"Cannot align GENDF energy grid for {context}. "
                    f"GENDF starts at {start_energy:.6e} eV, "
                    f"nearest library boundary is {nearest_energy:.6e} eV "
                    f"(relative difference: {relative_diff:.2e}). "
                    "This may indicate incompatible energy structures. "
                    "Set strict_alignment=False to use nearest-group alignment.")
            else:
                warnings.warn(
                    f"Energy alignment uncertainty for {context}: "
                    f"GENDF starts at {start_energy:.6e} eV, "
                    f"using nearest boundary {nearest_energy:.6e} eV "
                    f"(relative difference: {relative_diff:.2e}). "
                    "Cross-section placement may be off by one energy group.",
                    UserWarning)
                start_idx = nearest_idx

        # Calculate number of GENDF groups and end index
        n_gendf_groups = len(gendf_energies) - 1
        end_idx = min(start_idx + n_gendf_groups, self.n_groups)

        # Validate end boundary (sanity check for contiguous data)
        if end_idx < self.n_groups:
            expected_end = self.energy_bounds[end_idx]
            end_rtol = GENDF_RTOL_MATCH * 10  # 1e-5 for end boundary
            if not np.isclose(expected_end, end_energy, rtol=end_rtol, atol=GENDF_ATOL):
                relative_diff_end = abs(end_energy - expected_end) / max(end_energy, 1e-10)
                if strict_alignment:
                    raise ValueError(
                        f"GENDF energy range for {context} does not "
                        "align with library structure. End energy mismatch: "
                        f"GENDF {end_energy:.6e} eV vs expected {expected_end:.6e} eV "
                        f"(relative difference: {relative_diff_end:.2e})")
                else:
                    warnings.warn(
                        f"End energy mismatch for {context}: "
                        f"GENDF {end_energy:.6e} eV vs expected {expected_end:.6e} eV "
                        f"(relative difference: {relative_diff_end:.2e})",
                        UserWarning)

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

        """
        if not hasattr(self, '_nuclides_set_cache') or self._nuclides_set_cache is None:
            self._nuclides_set_cache = frozenset(self._file_index.keys())
        return self._nuclides_set_cache

    def _record_processing_error(
        self,
        error_type: str,
        nuclide: str,
        reaction: str,
        mt: int,
        **kwargs
    ) -> None:
        """Record a processing error for later logging.

        This helper consolidates the common error recording pattern used
        throughout isomeric branching extraction.

        Parameters
        ----------
        error_type : str
            Error category (e.g., 'elis_tol_exceeded', 'duplicate_mapping')
        nuclide : str
            Parent nuclide name
        reaction : str
            Reaction name (e.g., '(n,gamma)')
        mt : int
            ENDF MT number
        **kwargs : dict
            Additional error-specific fields
        """
        error = {
            'type': error_type,
            'nuclide': nuclide,
            'parent': nuclide,
            'reaction': reaction,
            'mt': mt,
            **kwargs
        }
        self._processing_errors.append(error)

    def _map_via_elis(
        self,
        all_meta_levels: list,
        ground_data,
        nuclide_name: str,
        reaction_name: str,
        mt: int,
        base_nuclide: str
    ) -> tuple:
        """Map metastable products using ELIS-based matching against decay library.

        Parameters
        ----------
        all_meta_levels : list
            List of metastable level dicts from _categorize_mf10_levels()
        ground_data : Tabulated1D or _RemainderXS or None
            Ground state data for context in duplicate detection
        nuclide_name : str
            Parent nuclide name
        reaction_name : str
            Reaction name (e.g., '(n,gamma)')
        mt : int
            ENDF MT number
        base_nuclide : str
            Base product nuclide name (e.g., 'Ir192')

        Returns
        -------
        tuple
            (mapped_meta_levels, lfs_mapping)
        """
        mapped_meta_levels = []
        lfs_mapping = {}

        # ==============================================================
        # First pass: Calculate ELIS and try lookup for all metastables
        # ==============================================================
        elis_results = []  # [(meta_dict, elis, liso_or_none), ...]

        for meta in all_meta_levels:
            level = meta['level']

            # Calculate ELIS from this level's own QM/QI TAB1 head
            qm = level.get('QM')
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

        # ==============================================================
        # Duplicate Mapping Detection: Multiple LFS -> Same LISO
        # ==============================================================
        # When multiple GENDF LFS values have ELIS within tolerance of
        # the same decay library LISO, keep only the closest match.

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

                # Collect ALL GENDF LFS levels for this reaction. A synthesized
                # ground is not a file level, so it must not appear here.
                gendf_all_lfs = []
                if ground_data is not None and \
                        not isinstance(ground_data, self._RemainderXS):
                    gendf_all_lfs.append({'lfs': 0, 'elis': 0.0})
                for meta_item in all_meta_levels:
                    qm = meta_item['level'].get('QM')
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
                self._record_processing_error(
                    'duplicate_mapping', nuclide_name, reaction_name, mt,
                    liso=liso, base_nuclide=base_nuclide,
                    kept_lfs=keeper_lfs, kept_elis=keeper_elis,
                    kept_diff=keeper_diff, dk_elis=dk_elis,
                    discarded=discarded_list, gendf_all_lfs=gendf_all_lfs,
                    decay_all_liso=decay_all_liso, target_z=z, target_a=a
                )

        # Filter out duplicate mappings
        if indices_to_skip:
            elis_results = [
                item for idx, item in enumerate(elis_results)
                if idx not in indices_to_skip
            ]

        # Collect all ELIS-matched LISO values (only 'matched', not 'nearest')
        assigned_lisos = set()
        for _, _, liso_result in elis_results:
            if liso_result is not None and liso_result.get('status') == 'matched':
                assigned_lisos.add(liso_result['liso'])

        # ==============================================================
        # Second pass: Assign names based on status
        # ==============================================================
        for meta, elis, liso_result in elis_results:
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

            # Handle each status type
            result = self._handle_elis_status(
                status, liso_result, nuclide_name, reaction_name, mt,
                base_nuclide, lfs, sigma, elis, elis_str, z, a,
                available_metas, alternatives
            )

            if result is not None:
                # Keep the level's MF=10 Q pair alongside its ELIS: the patcher
                # writes the per-pathway Q from QI/QM directly (ELIS = QM - QI).
                level = meta['level']
                result[3]['qm'] = level.get('QM')
                result[3]['qi'] = level.get('QI', 0.0)
                mapped_meta_levels.append(result)
                lfs_mapping[result[1]] = lfs  # result[1] is mapped_name

        return (mapped_meta_levels, lfs_mapping)

    def _handle_elis_status(
        self,
        status: str,
        liso_result: dict,
        nuclide_name: str,
        reaction_name: str,
        mt: int,
        base_nuclide: str,
        lfs: int,
        sigma,
        elis: float,
        elis_str: str,
        z: int,
        a: int,
        available_metas: list,
        alternatives: list
    ):
        """Handle ELIS lookup status and return mapped level or None.

        Returns
        -------
        tuple or None
            (lfs, mapped_name, sigma, elis_info) if successful, None if skipped
        """
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
            return (lfs, mapped_name, sigma, elis_info)

        elif status == 'nearest':
            # Tolerance exceeded - skip product
            liso = liso_result['liso']
            dk_elis = liso_result['dk_elis']
            diff_percent = liso_result.get('diff_pct', 0.0)

            warnings.warn(
                f"WARNING: ELIS_TOL_EXCEEDED: {nuclide_name}({reaction_name})->"
                f"{base_nuclide} LFS={lfs} ELIS={elis_str} eV. "
                f"Nearest _m{liso}: {dk_elis:.0f}eV ({diff_percent:.1f}% diff). "
                "Product skipped; branching will be renormalized.",
                UserWarning
            )
            # Find half_life for the closest match
            closest_state = next((s for s in available_metas if s.liso == liso), None)
            closest_half_life = closest_state.half_life if closest_state else None
            # Store for logging
            self._record_processing_error(
                'elis_tol_exceeded', nuclide_name, reaction_name, mt,
                lfs=lfs, elis=elis, dk_elis=dk_elis, liso=liso,
                diff_percent=diff_percent, base_nuclide=base_nuclide,
                target_z=z, target_a=a, omitted=True,
                half_life=closest_half_life
            )
            return None  # Skip

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
                "Metastable state(s) exist but have ELIS=0.0 (data quality issue): "
                f"{skipped_str}. "
                "Consider using a decay library with complete ELIS data. "
                "Product skipped; branching will be renormalized.",
                UserWarning
            )
            # Store for logging
            self._record_processing_error(
                'zero_elis_metastables', nuclide_name, reaction_name, mt,
                lfs=lfs, elis=elis, base_nuclide=base_nuclide,
                target_z=z, target_a=a, omitted=True,
                skipped_states=skipped_states
            )
            return None  # Skip

        elif status in ('no_decay_data', 'no_metastables', 'no_match'):
            # No usable metastable data
            if status == 'no_decay_data':
                reason = f"Nuclide (Z={z}, A={a}) not found in decay library"
            elif status == 'no_metastables':
                reason = f"No metastable states in decay library for (Z={z}, A={a})"
            else:
                reason = "No matching metastable state found"

            warnings.warn(
                f"WARNING: NO_METASTABLE_DECAY_DATA: {nuclide_name}({reaction_name})->"
                f"{base_nuclide}_m? LFS={lfs} ELIS={elis_str} eV. "
                f"{reason}. "
                "Product skipped; branching will be renormalized.",
                UserWarning
            )

            # Store in processing errors for logging
            self._record_processing_error(
                'no_metastable_decay_data', nuclide_name, reaction_name, mt,
                subtype=status, lfs=lfs, elis=elis, base_nuclide=base_nuclide,
                target_z=z, target_a=a, omitted=True,
                available_alternatives=alternatives
            )
            return None  # Skip

        else:
            # Unknown status - defensive handling
            warnings.warn(
                f"WARNING: Unknown ELIS lookup status '{status}' for "
                f"{nuclide_name}({reaction_name})->{base_nuclide}_m? LFS={lfs}",
                UserWarning
            )
            return None

    def _map_via_lfs_order(
        self,
        all_meta_levels: list,
        nuclide_name: str,
        reaction_name: str,
        mt: int,
        base_nuclide: str
    ) -> tuple:
        """Map metastable products using LFS-order (FISPACT-like) positional mapping.

        Parameters
        ----------
        all_meta_levels : list
            List of metastable level dicts from _categorize_mf10_levels()
        nuclide_name : str
            Parent nuclide name
        reaction_name : str
            Reaction name (e.g., '(n,gamma)')
        mt : int
            ENDF MT number
        base_nuclide : str
            Base product nuclide name (e.g., 'Ir192')

        Returns
        -------
        tuple
            (mapped_meta_levels, lfs_mapping)
        """
        mapped_meta_levels = []
        lfs_mapping = {}

        # Get decay library metastable count for validation
        z = all_meta_levels[0]['z']
        a = all_meta_levels[0]['a']
        decay_states = self.decay_lookup.get((z, a), [])
        dk_meta_states = sorted(
            [s for s in decay_states if s.liso > 0],
            key=lambda s: s.liso
        )
        dk_meta_count = len(dk_meta_states)
        gendf_meta_count = len(all_meta_levels)

        # Map by position with count validation
        for position, meta in enumerate(all_meta_levels, start=1):
            lfs = meta['lfs']
            sigma = meta['sigma']
            level = meta['level']

            # Calculate GENDF ELIS for reference logging
            qm = level.get('QM')
            qi = level.get('QI', 0.0)
            gendf_elis = (qm - qi) if qm is not None else None

            if position > dk_meta_count:
                # DROP this LFS - exceeds DK-Lib metastable count
                self._record_processing_error(
                    'lfs_order_dropped', nuclide_name, reaction_name, mt,
                    lfs=lfs, position=position, would_be_liso=position,
                    gendf_elis=gendf_elis, dk_meta_count=dk_meta_count,
                    gendf_meta_count=gendf_meta_count, base_nuclide=base_nuclide,
                    target_z=z, target_a=a
                )
                warnings.warn(
                    f"LFS_ORDER_DROPPED: {nuclide_name}({reaction_name})->"
                    f"{base_nuclide}_m{position} LFS={lfs}. "
                    f"DK-Lib has only {dk_meta_count} metastable state(s). "
                    "Product skipped; branching will be renormalized.",
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
                dk_state = dk_meta_states[liso - 1]
                dk_elis = dk_state.elis
                dk_half_life = dk_state.half_life

            # Check ELIS reference for Ag116-type warnings
            elis_ref_status = None
            elis_ref_diff = None
            if gendf_elis is not None and dk_elis is not None:
                elis_ref_diff = abs(gendf_elis - dk_elis)
                if elis_match(gendf_elis, dk_elis, self._elis_rtol, self._elis_atol):
                    elis_ref_status = 'ok'
                else:
                    elis_ref_status = 'mismatch'
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
                            elis_ref_status = 'wrong_liso'
                            warnings.warn(
                                f"LFS_ORDER_ELIS_MISMATCH: {nuclide_name}({reaction_name})->"
                                f"{mapped_name}: LFS-order maps LFS={lfs} to _m{liso}, "
                                f"but ELIS matching would map to _m{elis_liso}. "
                                f"(GENDF ELIS={gendf_elis:.0f}eV, DK _m{liso} ELIS={dk_elis:.0f}eV, "
                                f"DK _m{elis_liso} ELIS={elis_dk_elis:.0f}eV). "
                                "Consider using mapping_mode='elis' for production.",
                                UserWarning
                            )

            elis_info = {
                'method': 'lfs_order',
                'lfs': lfs,
                'liso': liso,
                'position': position,
                'elis': gendf_elis,
                'qm': qm,          # MF=10 Q pair; the patcher writes the
                'qi': qi,          # per-pathway Q from it (ELIS = QM - QI)
                'gendf_elis': gendf_elis,
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
                self._record_processing_error(
                    'lfs_order_orphan_dk', nuclide_name, reaction_name, mt,
                    liso=dk_state.liso, dk_elis=dk_state.elis,
                    dk_half_life=dk_state.half_life, dk_meta_count=dk_meta_count,
                    gendf_meta_count=gendf_meta_count, base_nuclide=base_nuclide,
                    target_z=z, target_a=a
                )

        return (mapped_meta_levels, lfs_mapping)

    def _build_branching_result(
        self,
        ground_data,
        ground_product: str,
        mapped_meta_levels: list,
        lfs_mapping: dict,
        nuclide_name: str,
        mt: int,
        ground_qm=None
    ) -> Optional[IsomericBranching]:
        """Build IsomericBranching result from ground and mapped metastable data.

        Parameters
        ----------
        ground_data : Tabulated1D or _RemainderXS or None
            Ground state cross-section data (file section or the synthesized
            MF=3 remainder from :meth:`_synthesize_ground_from_mf3`)
        ground_product : str
            Ground state product name (e.g., 'Ir192')
        mapped_meta_levels : list
            List of (lfs, mapped_name, sigma, elis_info) tuples
        lfs_mapping : dict
            Mapping of product names to LFS values
        nuclide_name : str
            Parent nuclide name (for warnings)
        mt : int
            ENDF MT number
        ground_qm : float or None
            QM of the LFS=0 subsection itself; None when the ground was
            synthesized (no such subsection exists)

        Returns
        -------
        IsomericBranching or None
            Branching data, or None if no valid energy points, no ground state
            (the caller repairs a repairable ground before this point), or a
            synthesized ground left as the only product
        """
        # A ground state is required for the ratios; the ground-absent MF=10
        # pattern is repaired (or classified as a skip) by the caller.
        if ground_data is None:
            return None

        # Synthesized ground + no surviving metastable = a one-product
        # decoration: no branching information, and its presence would keep an
        # (n,n') self-loop alive through the pruner's has_branching test.
        if isinstance(ground_data, self._RemainderXS) and not mapped_meta_levels:
            return None

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

        if len(energies_list) > 0 and branching_array.shape[0] != n_products:
            raise AssertionError(
                f"Product count mismatch for {nuclide_name} MT={mt}: "
                f"{n_products} products but branching_array has "
                f"{branching_array.shape[0]} rows")

        # Validate computed branching ratios
        if len(energies_list) == 0:
            warnings.warn(
                "No valid energy points with positive cross-section for "
                f"{nuclide_name} MT={mt}. Cannot compute branching ratios.",
                UserWarning
            )
            return None

        # Check for NaN/Inf values (indicates numerical issues)
        if not np.all(np.isfinite(branching_array)):
            nan_count = np.sum(~np.isfinite(branching_array))
            warnings.warn(
                "Invalid values (NaN/Inf) found in branching ratios for "
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
                "This indicates a bug in the branching ratio calculation."
            )

        # Get reaction name
        reaction_name = MT_TO_REACTION.get(mt, f'MT{mt}')

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
            elis_mapping=elis_mapping if elis_mapping else None,
            ground_qm=ground_qm
        )

    def _categorize_mf10_levels(
        self,
        mf10_data: dict,
        nuclide_name: str,
        mt: int
    ) -> Optional[tuple]:
        """Categorize MF=10 levels into ground state and metastable levels.

        Parameters
        ----------
        mf10_data : dict
            MF=10 section data from ENDF material
        nuclide_name : str
            Nuclide name (for warning messages)
        mt : int
            ENDF MT number (for warning messages)

        Returns
        -------
        tuple or None
            (ground_data, ground_product, ground_qm, all_meta_levels,
            base_nuclide, n_anonymous) if metastable levels exist, None if only
            ground state. ``ground_qm`` is the LFS=0 subsection's OWN QM (None
            if there is no LFS=0 subsection) -- QM is a per-subsection TAB1
            head, not a section constant, so a metastable's QM cannot stand in
            for it. ``n_anonymous`` counts the dropped IZAP=0 levels, so callers
            can tell an absent ground from an unnamed one (R1-61).
        """
        ground_data = None
        ground_product = None
        ground_qm = None
        all_meta_levels = []
        n_anonymous = 0

        for level in mf10_data['levels']:
            lfs = level['LFS']
            izap = level['IZAP']
            sigma = level['sigma']

            # Validate IZAP
            if izap == 0:
                n_anonymous += 1
                warnings.warn(
                    f"Skipping MF=10 level in {nuclide_name} MT={mt}: "
                    f"Invalid IZAP={izap} (product not specified in GENDF file). "
                    "This is a data quality issue in the source GENDF library.",
                    UserWarning
                )
                continue

            # Extract Z, A from IZAP
            z_prod = izap // 1000
            a_prod = izap % 1000

            if z_prod not in ATOMIC_SYMBOL:
                warnings.warn(
                    f"Invalid atomic number Z={z_prod} from IZAP={izap}. "
                    "Skipping this level.",
                    UserWarning
                )
                continue

            symbol = ATOMIC_SYMBOL[z_prod]
            base_product = f"{symbol}{a_prod}"

            if lfs == 0:
                ground_data = sigma
                ground_product = base_product
                ground_qm = level.get('QM')
            else:
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

        # Determine base nuclide name
        base_nuclide = ground_product if ground_product else all_meta_levels[0]['base_product']

        return (ground_data, ground_product, ground_qm, all_meta_levels,
                base_nuclide, n_anonymous)

    def _record_unmatched_mt(self, nuclide_name, mt, reaction_name, reason):
        """Record a classified (nuclide, MT) skip in the unmatched-MT bookkeeping."""
        if not hasattr(self, '_unmatched_mts'):
            self._unmatched_mts = []
        self._unmatched_mts.append({
            'nuclide': nuclide_name,
            'mt': mt,
            'reaction': reaction_name,
            'reason': reason,
        })

    @dataclass(eq=False)
    class _RemainderXS:
        """Bare (x, y) container for the synthesized ground level.

        Intentionally NOT a Tabulated1D: the remainder is histogram group data
        and must never be lin-lin callable. Consumers touch only ``.x``/``.y``,
        and the type doubles as the "this ground is synthesized" marker.
        ``eq=False``: numpy fields make a generated ``__eq__`` raise.
        """

        x: np.ndarray
        y: np.ndarray
        n_clamped: int = 0   # union-grid points where the raw remainder was < 0
        n_points: int = 0    # union-grid points total

    def _synthesize_ground_from_mf3(self, all_meta_levels, nuclide_name, mt,
                                    n_anonymous):
        """Synthesize the absent ground level as the clamped MF=3 remainder.

        A radioactive-products-only MF=10 omits a (quasi-)stable ground state
        (e.g. JEFF-4.0 In115 MT=4 stores only In115_m1). MF=3 carries the
        reaction total, so ground = max(0, sigma_MF3 - sum sigma_MF10_m)
        per energy point (policy 3(a)). Patcher lane only: the decoration is
        written solely against a true MF=3 section, so an MF=10-only reaction
        stays a classified skip here (the runtime lane unifies on get_xs).

        Returns
        -------
        tuple or None
            (ground_data, ground_product), or None when unrepairable (any
            anonymous level, no true MF=3 section, or metastables from more
            than one product IZAP); the skip is classified in ``unmatched_mts``.
        """
        reaction_name = MT_TO_REACTION.get(mt, f'MT{mt}')

        # An anonymous (IZAP=0) level means the ground may be UNNAMED rather
        # than absent, so the named levels are a subset of the real final
        # states: repairing would decorate that subset and invert the branching
        # (R1-61 all-or-nothing). Checked first -- it outranks the other
        # reasons.
        if n_anonymous:
            self._record_unmatched_mt(nuclide_name, mt, reaction_name,
                                      'mf10_anonymous_levels')
            return None

        izaps = {meta['izap'] for meta in all_meta_levels}
        if len(izaps) != 1:
            self._record_unmatched_mt(nuclide_name, mt, reaction_name,
                                      'mf10_ambiguous_izap')
            return None
        ground_product = get_product_name(izaps.pop(), 0)

        # get_xs() serves Sigma(MF=10) when MF=3 is absent, which would make the
        # remainder identically zero -- test the section itself.
        material = self._load_material(nuclide_name, require_full_parser=True)
        mf3_section = material.section_data.get((3, mt))
        mf3_sigma = mf3_section.get('sigma') if mf3_section else None
        if mf3_sigma is None or not hasattr(mf3_sigma, 'x'):
            self._record_unmatched_mt(nuclide_name, mt, reaction_name,
                                      'mf10_metastable_only_no_mf3')
            return None

        meta_lookups = [dict(zip(meta['sigma'].x, meta['sigma'].y))
                        for meta in all_meta_levels]
        mf3_x = np.asarray(mf3_sigma.x, dtype=float)
        mf3_y = np.asarray(mf3_sigma.y, dtype=float)
        energies = sorted(set(mf3_x.tolist()).union(*meta_lookups))
        # INT=1 group data: on [x[i-1], x[i]) the total is y[i-1]. An exact
        # dict lookup would return 0 at a metastable grid point the MF=3 grid
        # lacks -> a silent BR_m = 1.0 there.
        idx = np.searchsorted(mf3_x, energies, side='right')
        inside = (idx > 0) & (idx < len(mf3_x))
        mf3_at = np.where(inside, mf3_y[np.clip(idx - 1, 0, None)], 0.0)
        raw = [
            total - sum(max(0.0, lookup.get(e, 0.0)) for lookup in meta_lookups)
            for e, total in zip(energies, mf3_at)
        ]
        remainder = [max(0.0, r) for r in raw]
        n_clamped = sum(1 for r in raw if r < 0.0)

        return (self._RemainderXS(np.array(energies), np.array(remainder),
                                  n_clamped, len(energies)),
                ground_product)

    def _record_ground_repair(self, nuclide_name, mt, reaction_name,
                              ground_product, all_meta_levels,
                              clamped_points=0, total_points=0):
        """Record a synthesized-ground repair; warns once per (nuclide, MT, library)."""
        if not hasattr(self, '_ground_repaired'):
            self._ground_repaired = []
        self._ground_repaired.append({
            'nuclide': nuclide_name,
            'parent': nuclide_name,
            'mt': mt,
            'reaction': reaction_name,
            'ground_product': ground_product,
            'metastable_lfs': [meta['lfs'] for meta in all_meta_levels],
            'clamped_points': clamped_points,
            'total_points': total_points,
        })
        clamp_note = (
            f"; remainder clamped to zero in {clamped_points} of "
            f"{total_points} points (sum sigma_MF10_m exceeds the MF=3 total "
            "there; metastables receive 100% at those points)"
            if clamped_points else "")
        # Shared warn-once store, own key prefix: a patch-then-run process must
        # see both the patcher message and the later runtime one. The library's
        # path is in the key so a second library still warns.
        _warn_runtime_branching(
            ('patcher_ground_repaired', nuclide_name, mt,
             _library_warn_key(self)),
            f"{nuclide_name} MT={mt}: MF=10 has metastable level(s) but no "
            "LFS=0 subsection (radioactive-products-only file); the ground "
            f"channel {ground_product} is synthesized from the MF=3 total as "
            f"max(0, sigma_MF3 - sum sigma_MF10_m){clamp_note}.")

    def _load_mf10_data(
        self,
        nuclide_name: str,
        mt: int
    ) -> Optional[tuple]:
        """Load MF=10 section data for isomeric branching extraction.

        Parameters
        ----------
        nuclide_name : str
            Nuclide name in OpenMC format
        mt : int
            ENDF MT number

        Returns
        -------
        tuple or None
            (mf10_data dict, qm_section float or None) if MF=10 exists,
            None if no MF=10 data for this reaction. ENDF-6 puts QM on every
            subsection's TAB1 head and never at section level, so ``qm_section``
            is None for endf-parsed data; callers read QM per level.
        """
        # Load material with full parser (required for MF=10 data)
        material = self._load_material(nuclide_name, require_full_parser=True)

        # Check if MF=10 data exists for this reaction
        if (10, mt) not in material.section_data:
            return None

        mf10_data = material.section_data[10, mt]

        return (mf10_data, mf10_data.get('QM'))

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

    def _get_production_xs(self, nuclide_name: str, mt: int):
        """Get raw MF=10 production XS without name mapping.

        Returns list of (lfs, izap, xs_array) sorted by LFS ascending.
        Each xs_array is aligned to the full group grid (threshold
        reactions carry partial-range MF=10 sections).
        """
        # MT=5 (lumped) carries many products sharing an LFS, so LFS-keyed
        # retention is meaningless and no depletion pathway consumes it.
        if mt == 5:
            return []

        mf10_result = self._load_mf10_data(nuclide_name, mt)
        if mf10_result is None:
            return []
        mf10_data, _ = mf10_result

        levels = []
        for level in mf10_data['levels']:
            lfs = level['LFS']
            izap = level['IZAP']
            sigma = level['sigma']
            if hasattr(sigma, 'x'):
                # Energy-aware alignment onto the full group grid
                # (mirrors the C++ backend, commit 6e66d2e18)
                xs = self._align_to_group_grid(
                    np.asarray(sigma.x), np.asarray(sigma.y),
                    f"{nuclide_name} MT={mt} LFS={lfs} (MF=10)",
                    strict_alignment=False)
            else:
                # No energy grid available — assume threshold data at the
                # high-energy end (C++ backend fallback)
                raw = np.asarray(sigma)
                xs = np.zeros(self.n_groups)
                n = min(len(raw), self.n_groups)
                xs[self.n_groups - n:] = raw[:n]
            levels.append((lfs, izap, xs))

        # Reaction+LFS is a complete key, so a repeated LFS is ambiguous: drop
        # every colliding subsection rather than letting one of them win.
        seen = [lfs for lfs, _, _ in levels]
        duplicated = sorted({lfs for lfs in seen if seen.count(lfs) > 1})
        if duplicated:
            levels = [lv for lv in levels if lv[0] not in duplicated]
            key = (nuclide_name, mt)
            if key not in _WARNED_MF10_DUPLICATE_LFS:
                _WARNED_MF10_DUPLICATE_LFS.add(key)
                warnings.warn(
                    f"{nuclide_name} MT={mt}: MF=10 LFS {duplicated} appear in "
                    "more than one subsection; all colliding subsections are "
                    "dropped, so those levels are unavailable for isomeric "
                    "branching and for cross sections.", UserWarning)

        levels.sort(key=lambda x: x[0])
        return levels

    def get_branching_ratios(
        self,
        nuclide_name: str,
        mt: int,
        target_names=None,
        lfs_values=None
    ) -> Optional[IsomericBranching]:
        """Extract energy-dependent isomeric branching ratios from MF=10.

        Two modes:
        - Patcher mode (target_names=None): ELIS/LFS-order mapping using
          decay file. Used at chain-creation time.
        - Runtime mode (target_names + lfs_values provided): fetch
          production XS by LFS, pair with pre-resolved names. No decay
          file needed.

        Parameters
        ----------
        nuclide_name : str
            Nuclide name in OpenMC format
        mt : int
            ENDF MT number
        target_names : list of str, optional
            Product names from chain (runtime mode)
        lfs_values : list of int, optional
            LFS values from chain (runtime mode)

        Returns
        -------
        IsomericBranching or None
            None when the reaction has no MF=10 data, no usable levels, or an
            absent ground state that could not be repaired or whose products
            did not survive mapping.

        Notes
        -----
        Patcher mode has side effects: an absent ground level may be
        synthesized from the MF=3 remainder (recorded in ``ground_repaired``
        and warned once), and every classified skip is recorded in
        ``unmatched_mts``.
        """
        # Runtime mode — no decay file needed
        if target_names is not None:
            if lfs_values is None:
                raise ValueError("lfs_values required with target_names")

            levels = self._get_production_xs(nuclide_name, mt)
            # The rate XS only feeds the missing-ground repair, so fetch it
            # lazily and leave every other reaction's path untouched.
            total_xs = None
            if levels and 0 in lfs_values and 0 not in {lfs for lfs, _, _ in levels}:
                total_xs = self._rate_total_xs(nuclide_name, mt)
            return build_runtime_branching(
                levels, target_names, lfs_values, self.energy_bounds,
                nuclide_name, mt, total_xs=total_xs,
                library_key=_library_warn_key(self))

        # Patcher mode — ELIS/LFS-order mapping (requires decay file)
        if self.decay_lookup is None:
            raise ValueError(
                "Patcher mode requires decay_file for ELIS/LFS-order mapping. "
                "Either pass decay_file to GENDFLibrary(), or use runtime mode "
                "with target_names and lfs_values from a patched chain.")

        mf10_result = self._load_mf10_data(nuclide_name, mt)
        if mf10_result is None:
            return None
        # QM lives on each subsection's TAB1 head, never at section level, so
        # the second element is discarded here and read per level instead.
        mf10_data, _ = mf10_result

        levels_result = self._categorize_mf10_levels(mf10_data, nuclide_name, mt)
        if levels_result is None:
            return None
        (ground_data, ground_product, ground_qm, all_meta_levels, base_nuclide,
         n_anonymous) = levels_result

        # Ground absent with metastables present is a legitimate evaluator
        # pattern (stable ground in a radioactive-products-only file), repaired
        # from MF=3 before the normal mapping/decoration path runs.
        ground_synthesized = ground_data is None
        if ground_synthesized:
            repaired = self._synthesize_ground_from_mf3(
                all_meta_levels, nuclide_name, mt, n_anonymous)
            if repaired is None:
                return None
            ground_data, ground_product = repaired
            synthesized_ground = ground_data

        reaction_name = MT_TO_REACTION.get(mt, f'MT{mt}')

        if self._mapping_mode == 'elis':
            mapped_meta_levels, lfs_mapping = self._map_via_elis(
                all_meta_levels, ground_data,
                nuclide_name, reaction_name, mt, base_nuclide
            )
        elif self._mapping_mode == 'lfs_order':
            mapped_meta_levels, lfs_mapping = self._map_via_lfs_order(
                all_meta_levels, nuclide_name, reaction_name, mt, base_nuclide
            )

        result = self._build_branching_result(
            ground_data, ground_product, mapped_meta_levels,
            lfs_mapping, nuclide_name, mt, ground_qm=ground_qm
        )
        # Only a decoration that actually ships counts as a repair; a synthesis
        # that lost every metastable at mapping is a classified skip, not a
        # silent nothing. The other None -- no union-grid point with positive
        # total XS -- keeps its own loud warning and is NOT a mapping loss, so
        # it records nothing here.
        if ground_synthesized:
            if result is not None:
                self._record_ground_repair(
                    nuclide_name, mt, reaction_name, ground_product,
                    all_meta_levels, synthesized_ground.n_clamped,
                    synthesized_ground.n_points)
            elif not mapped_meta_levels:
                self._record_unmatched_mt(
                    nuclide_name, mt, reaction_name,
                    'ground_absent_all_metastables_unmapped')
        return result

    def process_library_for_branching(
        self,
        mt_list: Optional[list[int]] = None,
        progress_callback: Optional[callable] = None,
        verbose: bool = False,
        chain: Optional['Chain'] = None,
        skip_reactions: Optional[set] = None
    ) -> dict[str, dict[str, IsomericBranching]]:
        """Process entire GENDF library for isomeric branching data.

        Scans all nuclides in the library and extracts energy-dependent
        branching ratios for specified reactions that have MF=10 data.
        Extraction is isolated per (nuclide, MT): one surprising channel costs
        that channel only, never the nuclide's other reactions. Data-integrity
        failures are collected and raised together at the end of the scan.
        Resets ``ground_repaired``, ``unmatched_mts`` and ``processing_errors``.

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
        skip_reactions : set of tuple, optional
            ``(nuclide, reaction)`` pairs to leave unextracted. Used by the
            patcher for MF=10 sections holding unrepaired anonymous (IZAP=0)
            subsections: their decoration is pruned anyway, and the named
            levels are only a subset of the real final states, so extracting
            them would invert the branching (R1-61).

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
        self._ground_repaired = []  # Reactions whose ground was synthesized from MF=3
        unexpected_errors = []  # Data-integrity failures that must abort the scan

        def _record_failure(nuclide_name, exc, **where):
            """Classify a failure, record it, and flag the unexpected ones."""
            error_str = str(exc)
            # NO_METASTABLE_DECAY_DATA is an expected skip on healthy libraries.
            # A whole-file load failure (tracked in _material_load_failures) is
            # a known-unparseable file -- e.g. the 97 EAF-2010 endf-crasher
            # files -- whose channels are unextractable, not a data defect in
            # parsed content: skip the nuclide, warn in the summary below.
            # Every other exception is a data-integrity failure that must not
            # be swallowed. Record all of them (typed) so the scan completes,
            # then raise below on the unexpected ones.
            if 'NO_METASTABLE_DECAY_DATA' in error_str:
                error_type = 'no_metastable_decay_data'
            elif ('Failed to load GENDF file' in error_str
                    and any(k[0] == nuclide_name
                            for k in self._material_load_failures)):
                error_type = 'material_load_failure'
            else:
                error_type = 'unexpected_processing_error'
            entry = {
                'nuclide': nuclide_name,
                **where,
                'error': error_str,
                'exception_class': type(exc).__name__,
                'type': error_type,
            }
            self._processing_errors.append(entry)
            if error_type == 'unexpected_processing_error':
                unexpected_errors.append(entry)

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

                    # Per-MT isolation: one surprising channel must never cost
                    # the nuclide its other channels.
                    try:
                        # Caller-supplied exclusion: a knowingly unextractable
                        # section, not an unmatched MT and not an error.
                        if skip_reactions and (nuclide_name,
                                               reaction_name) in skip_reactions:
                            continue

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
                                print(f"  {nuclide_name} {reaction_name}: "
                                      f"{n_energies} energy points{method_tag}")
                                print(f"    Products: {branching.products}")
                    except Exception as e:
                        if verbose:
                            print(f"  {nuclide_name} MT={mt}: {e}")
                        _record_failure(nuclide_name, e, mt=mt,
                                        reaction=reaction_name)
                        continue

                # Only add to result if at least one branching reaction found
                if has_branching:
                    all_branching_data[nuclide_name] = temp_reactions

            except Exception as e:
                # Defensive net: all known raise sites are inside the per-MT
                # try; retained for future edits / iterator failures.
                if verbose:
                    print(f"  {nuclide_name}: {e}")
                _record_failure(nuclide_name, e)
                continue

        # Known-unparseable files are skipped whole, but never silently.
        load_failed = sorted({e['nuclide'] for e in self._processing_errors
                              if e['type'] == 'material_load_failure'})
        if load_failed:
            preview = ', '.join(load_failed[:5])
            warnings.warn(
                f"{len(load_failed)} nuclide(s) skipped entirely: GENDF file "
                f"failed to parse (endf-crasher files): {preview}"
                + (', ...' if len(load_failed) > 5 else ''))

        # Fail loud on data-integrity errors; expected skips never trigger this.
        if unexpected_errors:
            n = len(unexpected_errors)
            # Entries are per-(nuclide, MT); the window is grouped by NUCLIDE so
            # one flooding nuclide cannot hide the other 3780 failures.
            by_nuclide = {}
            for e in unexpected_errors:
                by_nuclide.setdefault(e['nuclide'], []).append(e)
            shown = list(by_nuclide.items())[:10]
            lines = []
            for nuc, entries in shown:
                first = entries[0]
                # One indented line per nuclide: a multi-line exception message
                # would otherwise push the trailer onto an unindented line.
                text = str(first['error']).splitlines() or ['']
                head = text[0] + ('...' if len(text) > 1 else '')
                lines.append(
                    f"  {nuc}"
                    + (f" MT={first['mt']}" if 'mt' in first else "")
                    + f": {first['exception_class']}: {head}"
                    + (f" [+{len(entries) - 1} more error(s)]"
                       if len(entries) > 1 else ""))
            if len(by_nuclide) > len(shown):
                lines.append(f"  ... and {len(by_nuclide) - len(shown)} more "
                             "nuclide(s)")
            raise RuntimeError(
                f"process_library_for_branching encountered {n} unexpected "
                f"error(s) across {len(by_nuclide)} nuclide(s) during isomeric "
                "branching extraction:\n"
                + "\n".join(lines)
            )

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Found branching data for {len(all_branching_data)} nuclides")
            total_reactions = sum(len(rxs) for rxs in all_branching_data.values())
            print(f"Total reactions with branching: {total_reactions}")

        return all_branching_data

    @property
    def processing_errors(self) -> list:
        """Get processing errors from last process_library_for_branching() call.

        Returns
        -------
        list of dict
            Heterogeneous entries from two recorders, so always use ``.get()``:

            - mapping errors (:meth:`_record_processing_error`): 'type'
              ('elis_tol_exceeded', 'no_metastable_decay_data',
              'zero_elis_metastables', 'duplicate_mapping',
              'lfs_order_dropped', 'lfs_order_orphan_dk', ...), 'nuclide',
              'parent', 'reaction', 'mt', plus error-specific fields. No
              'exception_class'.
            - scan failures (``_record_failure``): 'nuclide', 'error' (the
              exception's message), 'exception_class', 'type'
              ('no_metastable_decay_data' or 'unexpected_processing_error'),
              and 'mt'/'reaction' only when the failure happened inside the
              per-MT loop (the outer defensive net records neither).
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
            Ground-absent MF=10 skips carry 'mf10_anonymous_levels',
            'mf10_metastable_only_no_mf3', 'mf10_ambiguous_izap' or
            'ground_absent_all_metastables_unmapped' (ground synthesized, but
            no metastable survived ELIS/decay-data mapping).
        """
        return getattr(self, '_unmatched_mts', [])

    @property
    def ground_repaired(self) -> list:
        """Reactions whose absent ground level was synthesized from MF=3.

        Returns
        -------
        list of dict
            Each dict contains 'nuclide'/'parent', 'mt', 'reaction',
            'ground_product', 'metastable_lfs', and the clamp counters
            'clamped_points' / 'total_points' (union-grid points where the raw
            remainder went negative, out of the total).
        """
        return getattr(self, '_ground_repaired', [])

    def _build_chain_reaction_lookup(self, chain) -> dict[str, set]:
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
        chain_reactions: dict[str, set]
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
    validate_energy_grid: bool = True,
    decay_file: Optional[PathLike] = None,
    elis_rtol: float = ELIS_RTOL,
    elis_atol: float = ELIS_ATOL,
    skip_zero_elis_metastables: bool = True,
    mapping_mode: str = 'elis'
):
    """Create a GENDF cross-section library with automatic backend selection.

    Energy structure is auto-detected from the GENDF files. The backend
    (C++ or Python) is selected automatically: C++ if available and no
    decay_file is provided, otherwise Python.

    Parameters
    ----------
    library_path : path-like
        Path to directory containing GENDF .asc files
    validate_energy_grid : bool, optional
        If True, validate that each GENDF file's energy grid matches the
        detected energy structure. Default is True. Only used with Python
        backend.
    decay_file : path-like, optional
        Path to ENDF decay library for ELIS-based isomeric state mapping.
        Can be a directory of decay files or a single concatenated file.
        When provided, enables accurate mapping of GENDF MF=10 metastable
        products to OpenMC ``_m{n}`` naming based on excitation energy
        matching. **Highly recommended** for isomeric branching workflows.
    elis_rtol : float, optional
        Relative tolerance for ELIS matching (default: 0.50 = 50%).
    elis_atol : float, optional
        Absolute tolerance in eV for ELIS matching (default: 0.0, rtol-only).
    skip_zero_elis_metastables : bool, optional
        If True, skip metastable states with ELIS=0 in decay library (likely
        data errors). Default is True.
    mapping_mode : {'elis', 'lfs_order'}, optional
        Isomeric state mapping mode:

        - 'elis' (default): Use excitation energy (ELIS) matching between
          GENDF MF=10 products and decay library. Most accurate method.
        - 'lfs_order': Use FISPACT-like positional mapping where the 1st
          metastable LFS maps to _m1, 2nd to _m2, etc. Useful for validation
          testing against FISPACT-II.

    Returns
    -------
    GENDFLibrary
        Library instance (either C++ or Python implementation)

    Raises
    ------
    FileNotFoundError
        If library_path does not exist
    ValueError
        If energy structure cannot be detected from GENDF files

    Examples
    --------
    >>> lib = GENDFLibrary('/path/to/JEFF40-GENDF/')
    >>> xs = lib.get_xs('U235', 102, lib.energy_bounds)
    >>>
    >>> # With ELIS-based isomeric mapping (recommended for branching)
    >>> lib = GENDFLibrary(
    ...     '/path/to/JEFF40-GENDF/',
    ...     decay_file='/path/to/JEFF40-decay/'
    ... )

    See Also
    --------
    parse_decay_isomeric_levels : Parse decay library for ELIS data
    lookup_liso : Find LISO for given excitation energy
    DecayState : Data class for nuclear state information
    """
    library_path = Path(library_path)
    if not library_path.exists():
        raise FileNotFoundError(
            f"GENDF library path does not exist: {library_path}")
    if not library_path.is_dir():
        raise ValueError(
            f"GENDF library path must be a directory: {library_path}")

    energy_structure = detect_energy_structure(library_path)

    use_cpp = _CppGENDFLibrary is not None and decay_file is None

    if use_cpp:
        if mapping_mode != 'elis':
            warnings.warn(
                f"mapping_mode='{mapping_mode}' ignored for C++ backend. "
                "C++ backend uses runtime mode with pre-resolved LFS values "
                "from the chain. Pass decay_file to use Python backend with "
                f"'{mapping_mode}' mapping.",
                UserWarning
            )
        energy_bounds = GROUP_STRUCTURES[energy_structure]
        return _CppGENDFLibrary(
            str(library_path),
            energy_bounds,
            energy_structure
        )
    else:
        return _PythonGENDFLibrary(
            library_path,
            validate_energy_grid,
            decay_file,
            elis_rtol,
            elis_atol,
            skip_zero_elis_metastables,
            mapping_mode,
            _energy_structure=energy_structure
        )


# Export public interface and backend classes (for type checking)
# Public GENDF API re-exported into the openmc.deplete namespace via
# ``from .gendf import *``. Kept deliberately narrow: internal helpers and
# generic data constants (ATOMIC_SYMBOL, MT_TO_REACTION, the ELIS_* helpers,
# etc.) stay importable via ``openmc.deplete.gendf`` but no longer leak into
# openmc.deplete, where names like ATOMIC_SYMBOL would shadow openmc.data.
__all__ = [
    'GENDFLibrary',
    'IsomericBranching',
    'build_runtime_branching',
    'detect_energy_structure',
]
