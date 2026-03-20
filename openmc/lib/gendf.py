"""
GENDF library interface via OpenMC C API

This module provides Python access to the C++ GENDF library implementation
through OpenMC's C API bindings.
"""

from ctypes import c_int, c_int32, c_double, c_char_p, c_bool, POINTER, pointer, cast
from typing import List, Optional
import numpy as np

from . import _dll
from .error import _error_handler
from openmc.exceptions import OpenMCError


# =============================================================================
# C API function signatures
# =============================================================================

# Library creation/destruction
_dll.openmc_gendf_library_create.argtypes = [
    c_char_p,                    # library_path
    c_int,                       # n_energy_bounds
    POINTER(c_double),           # energy_bounds
    c_char_p,                    # energy_structure_name
    POINTER(c_int32)             # lib_id (output)
]
_dll.openmc_gendf_library_create.restype = c_int
_dll.openmc_gendf_library_create.errcheck = _error_handler

_dll.openmc_gendf_library_free.argtypes = [c_int32]
_dll.openmc_gendf_library_free.restype = c_int
_dll.openmc_gendf_library_free.errcheck = _error_handler

# Library queries
_dll.openmc_gendf_library_get_n_groups.argtypes = [
    c_int32,                     # lib_id
    POINTER(c_int)               # n_groups (output)
]
_dll.openmc_gendf_library_get_n_groups.restype = c_int
_dll.openmc_gendf_library_get_n_groups.errcheck = _error_handler

_dll.openmc_gendf_library_get_energy_bounds.argtypes = [
    c_int32,                     # lib_id
    POINTER(POINTER(c_double)),  # bounds (output pointer)
    POINTER(c_int)               # n (output)
]
_dll.openmc_gendf_library_get_energy_bounds.restype = c_int
_dll.openmc_gendf_library_get_energy_bounds.errcheck = _error_handler

_dll.openmc_gendf_library_has_nuclide.argtypes = [
    c_int32,                     # lib_id
    c_char_p,                    # nuclide
    POINTER(c_bool)              # has (output)
]
_dll.openmc_gendf_library_has_nuclide.restype = c_int
_dll.openmc_gendf_library_has_nuclide.errcheck = _error_handler

_dll.openmc_gendf_library_available_nuclides.argtypes = [
    c_int32,                     # lib_id
    POINTER(POINTER(c_char_p)),  # nuclides (output pointer to array)
    POINTER(c_int)               # n (output)
]
_dll.openmc_gendf_library_available_nuclides.restype = c_int
_dll.openmc_gendf_library_available_nuclides.errcheck = _error_handler

# Cross-section retrieval
_dll.openmc_gendf_get_xs.argtypes = [
    c_int32,                     # lib_id
    c_char_p,                    # nuclide
    c_int32,                     # mt
    c_int,                       # n_energy_bounds
    POINTER(c_double),           # energy_bounds
    POINTER(POINTER(c_double)),  # xs_data (output pointer)
    POINTER(c_int)               # n_groups (output)
]
_dll.openmc_gendf_get_xs.restype = c_int
_dll.openmc_gendf_get_xs.errcheck = _error_handler

_dll.openmc_gendf_free_xs.argtypes = [POINTER(c_double)]
_dll.openmc_gendf_free_xs.restype = None

_dll.openmc_gendf_free_nuclides.argtypes = [POINTER(c_char_p), c_int]
_dll.openmc_gendf_free_nuclides.restype = None

# MF=10 production cross-section retrieval
_dll.openmc_gendf_get_production_xs.argtypes = [
    c_int32,                     # lib_id
    c_char_p,                    # nuclide
    c_int32,                     # mt
    POINTER(c_int),              # n_levels (output)
    POINTER(c_int),              # n_groups (output)
    POINTER(POINTER(c_int)),     # lfs_out (output)
    POINTER(POINTER(c_int)),     # izap_out (output)
    POINTER(POINTER(c_double)),  # xs_out (output)
]
_dll.openmc_gendf_get_production_xs.restype = c_int
_dll.openmc_gendf_get_production_xs.errcheck = _error_handler

_dll.openmc_gendf_free_production_xs.argtypes = [
    POINTER(c_int), POINTER(c_int), POINTER(c_double)]
_dll.openmc_gendf_free_production_xs.restype = None


# =============================================================================
# Python wrapper class
# =============================================================================

class GENDFLibrary:
    """
    Python wrapper for C++ GENDFLibrary class.

    Provides access to GENDF (Group-averaged ENDF) cross-section libraries
    through OpenMC's C API, enabling high-performance cross-section retrieval
    for depletion calculations.

    Parameters
    ----------
    library_path : str
        Path to directory containing GENDF files
    energy_bounds : array-like
        Energy group boundaries in eV (length n_groups + 1)
    energy_structure_name : str, optional
        Name of energy structure (e.g., 'CCFE-709', 'UKAEA-1102')

    Attributes
    ----------
    _lib_id : int
        C API library instance ID
    _energy_bounds : np.ndarray
        Energy boundaries array

    Examples
    --------
    >>> import numpy as np
    >>> from openmc.lib import gendf
    >>>
    >>> # Load GENDF library
    >>> energy_bounds = np.logspace(-5, 9, 710)  # 709 groups
    >>> lib = gendf.GENDFLibrary(
    ...     '/path/to/gendf/library',
    ...     energy_bounds,
    ...     'CCFE-709'
    ... )
    >>>
    >>> # Check available nuclides
    >>> nuclides = lib.available_nuclides()
    >>> print(f"Library contains {len(nuclides)} nuclides")
    >>>
    >>> # Get cross-section
    >>> if lib.has_nuclide('U235'):
    ...     xs = lib.get_xs('U235', 102, energy_bounds)  # n-gamma
    ...     print(f"U-235 (n,gamma): {xs.shape} groups")
    >>>
    >>> # Cleanup
    >>> del lib
    """

    def __init__(self, library_path: str, energy_bounds: np.ndarray,
                 energy_structure_name: Optional[str] = None):
        """
        Initialize GENDF library.

        Parameters
        ----------
        library_path : str
            Path to directory containing GENDF files
        energy_bounds : array-like
            Energy group boundaries in eV (length n_groups + 1)
        energy_structure_name : str, optional
            Name of energy structure (e.g., 'CCFE-709')
        """
        # Convert inputs
        self._energy_bounds = np.asarray(energy_bounds, dtype=np.float64)
        self._energy_structure = energy_structure_name
        n_bounds = len(self._energy_bounds)

        # Prepare C arrays
        path_bytes = library_path.encode('utf-8')
        bounds_array = self._energy_bounds.ctypes.data_as(POINTER(c_double))

        if energy_structure_name:
            name_bytes = energy_structure_name.encode('utf-8')
        else:
            name_bytes = None

        # Create library via C API
        lib_id = c_int32()
        _dll.openmc_gendf_library_create(
            path_bytes,
            n_bounds,
            bounds_array,
            name_bytes,
            pointer(lib_id)
        )

        self._lib_id = lib_id.value

        # Initialize caches for available_nuclides()
        self._nuclides_cache = None
        self._nuclides_set = None

    def __del__(self):
        """Cleanup library resources when object is destroyed."""
        if hasattr(self, '_lib_id'):
            try:
                _dll.openmc_gendf_library_free(self._lib_id)
            except Exception:
                pass  # Ignore errors during cleanup

    @property
    def n_groups(self) -> int:
        """
        Number of energy groups.

        Returns
        -------
        int
            Number of energy groups
        """
        n = c_int()
        _dll.openmc_gendf_library_get_n_groups(self._lib_id, pointer(n))
        return n.value

    @property
    def energy_bounds(self) -> np.ndarray:
        """
        Energy group boundaries in eV.

        Returns
        -------
        np.ndarray
            Energy boundaries array (length n_groups + 1)
        """
        return self._energy_bounds.copy()

    @property
    def energy_bins(self) -> np.ndarray:
        """
        Alias for energy_bounds (backwards compatibility).

        Returns
        -------
        np.ndarray
            Energy boundaries array (length n_groups + 1)
        """
        return self.energy_bounds

    @property
    def energy_structure(self) -> Optional[str]:
        """Name of the energy group structure (e.g., 'CCFE-709', 'UKAEA-1102')."""
        return self._energy_structure

    @property
    def decay_lookup(self):
        """C++ backend does not support decay file lookup."""
        return None

    def has_nuclide(self, nuclide: str) -> bool:
        """
        Check if nuclide is available in library.

        Parameters
        ----------
        nuclide : str
            Nuclide name (e.g., 'U235', 'Pu239')

        Returns
        -------
        bool
            True if nuclide is available
        """
        nuclide_bytes = nuclide.encode('utf-8')
        has = c_bool()
        _dll.openmc_gendf_library_has_nuclide(
            self._lib_id,
            nuclide_bytes,
            pointer(has)
        )
        return has.value

    def available_nuclides(self) -> List[str]:
        """
        Get list of all available nuclides.

        This method caches the result for performance since the nuclide
        list is immutable for a given library instance.

        Returns
        -------
        list of str
            List of nuclide names

        Notes
        -----
        The returned list is a copy of the cached list to prevent external
        modification of the cache.
        """
        # Return cached value if available
        if self._nuclides_cache is not None:
            return list(self._nuclides_cache)  # Return copy to prevent modification

        # Fetch from C++
        nuclides_ptr = POINTER(c_char_p)()
        n = c_int()

        _dll.openmc_gendf_library_available_nuclides(
            self._lib_id,
            pointer(nuclides_ptr),
            pointer(n)
        )

        # Convert C string array to Python list
        nuclide_list = []
        for i in range(n.value):
            nuclide_list.append(nuclides_ptr[i].decode('utf-8'))

        # Free C-allocated memory (FIXES MEMORY LEAK)
        _dll.openmc_gendf_free_nuclides(nuclides_ptr, n.value)

        # Cache the result (use tuple for immutability)
        self._nuclides_cache = tuple(nuclide_list)

        # Also create set for O(1) lookups
        self._nuclides_set = frozenset(nuclide_list)

        return list(self._nuclides_cache)  # Return mutable copy

    def available_nuclides_set(self) -> frozenset:
        """
        Get set of all available nuclides for O(1) lookup.

        This is more efficient than `available_nuclides()` when checking
        nuclide availability in a loop.

        Returns
        -------
        frozenset of str
            Immutable set of nuclide names

        Examples
        --------
        >>> lib = GENDFLibrary('/path/to/gendf', energy_bounds, 'CCFE-709')
        >>> available = lib.available_nuclides_set()
        >>> if 'U235' in available:  # O(1) lookup
        ...     xs = lib.get_xs('U235', 102, lib.energy_bounds)
        """
        # Ensure cache is populated
        if self._nuclides_set is None:
            self.available_nuclides()  # Populates both caches
        return self._nuclides_set

    def get_xs(self, nuclide: str, mt: int,
               energy_bounds: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Retrieve cross-section data for a nuclide and reaction.

        Parameters
        ----------
        nuclide : str
            Nuclide name (e.g., 'U235', 'Pu239')
        mt : int
            ENDF MT reaction number (e.g., 102 for n-gamma, 18 for fission)
        energy_bounds : array-like, optional
            Energy boundaries for interpolation. If None, uses library's
            energy structure.

        Returns
        -------
        np.ndarray
            Cross-section values in barns for each energy group

        Raises
        ------
        RuntimeError
            If nuclide or reaction not found
        """
        # Use library energy bounds if not specified
        if energy_bounds is None:
            energy_bounds = self._energy_bounds
        else:
            energy_bounds = np.asarray(energy_bounds, dtype=np.float64)

        n_bounds = len(energy_bounds)
        bounds_array = energy_bounds.ctypes.data_as(POINTER(c_double))

        nuclide_bytes = nuclide.encode('utf-8')

        # Prepare output pointers
        xs_ptr = POINTER(c_double)()
        n_groups = c_int()

        # Call C API
        _dll.openmc_gendf_get_xs(
            self._lib_id,
            nuclide_bytes,
            mt,
            n_bounds,
            bounds_array,
            pointer(xs_ptr),
            pointer(n_groups)
        )

        # Copy data to numpy array (must copy before freeing C memory)
        xs_array = np.ctypeslib.as_array(xs_ptr, shape=(n_groups.value,)).copy()

        # Free C-allocated memory
        _dll.openmc_gendf_free_xs(xs_ptr)

        return xs_array

    def get_all_xs(self, nuclide: str,
                   mts: Optional[list[int]] = None) -> dict[int, np.ndarray]:
        """Get cross-sections for all requested reactions for a nuclide.

        Parameters
        ----------
        nuclide : str
            Nuclide name (e.g., 'U235', 'Pu239')
        mts : list of int, optional
            MT numbers to retrieve. If None, raises ValueError (C++ backend
            cannot discover available reactions).

        Returns
        -------
        dict of int to numpy.ndarray
            Mapping of MT number to cross-section array. MTs not found
            in the library are silently skipped.
        """
        if mts is None:
            raise ValueError(
                "C++ backend requires explicit mts list; cannot discover "
                "available reactions. Use Python backend or pass mts.")
        result = {}
        for mt in mts:
            try:
                result[mt] = self.get_xs(nuclide, mt)
            except OpenMCError:
                pass
        return result

    def _get_production_xs(self, nuclide: str, mt: int):
        """Get raw MF=10 production XS for all levels of a reaction."""
        nuclide_bytes = nuclide.encode('utf-8')
        n_levels = c_int()
        n_groups = c_int()
        lfs_ptr = POINTER(c_int)()
        izap_ptr = POINTER(c_int)()
        xs_ptr = POINTER(c_double)()

        _dll.openmc_gendf_get_production_xs(
            self._lib_id, nuclide_bytes, mt,
            pointer(n_levels), pointer(n_groups),
            pointer(lfs_ptr), pointer(izap_ptr), pointer(xs_ptr))

        nl = n_levels.value
        ng = n_groups.value

        if nl == 0:
            return []

        result = []
        xs_flat = np.ctypeslib.as_array(xs_ptr, shape=(nl * ng,)).copy()
        for i in range(nl):
            result.append((
                lfs_ptr[i],
                izap_ptr[i],
                xs_flat[i * ng : (i + 1) * ng]
            ))

        _dll.openmc_gendf_free_production_xs(lfs_ptr, izap_ptr, xs_ptr)
        return result

    def get_branching_ratios(self, nuclide: str, mt: int,
                             target_names=None, lfs_values=None):
        """Get isomeric branching ratios from MF=10 production cross-sections.

        Parameters
        ----------
        nuclide : str
            Nuclide name
        mt : int
            ENDF MT number
        target_names : list of str, optional
            Product names from chain. Required for runtime mode.
        lfs_values : list of int, optional
            LFS values corresponding to target_names. Required for runtime mode.

        Returns
        -------
        IsomericBranching or None
        """
        if target_names is None:
            raise NotImplementedError(
                "C++ backend does not support ELIS mapping. Use Python "
                "backend with decay_file for chain patching.")

        if lfs_values is None:
            raise ValueError("lfs_values required with target_names")

        from openmc.deplete.gendf import IsomericBranching, MT_TO_REACTION

        levels = self._get_production_xs(nuclide, mt)
        if not levels:
            return None

        # Build LFS→XS lookup from C++ results
        lfs_to_xs = {lfs: xs for lfs, izap, xs in levels}

        # Filter to requested LFS values
        n_groups = self.n_groups
        prod_xs = []
        for lfs in lfs_values:
            if lfs in lfs_to_xs:
                prod_xs.append(lfs_to_xs[lfs])
            else:
                # LFS not found in GENDF — zero production
                prod_xs.append(np.zeros(n_groups))

        prod_xs = np.array(prod_xs)  # (n_targets, n_groups)

        # Compute branching ratios: BR_i = σ_prod_i / Σ σ_prod_j
        total = prod_xs.sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            br = np.where(total > 0, prod_xs / total, 0.0)

        reaction = MT_TO_REACTION.get(mt, f'MT{mt}')
        lfs_mapping = {name: lfs for name, lfs
                       in zip(target_names, lfs_values) if lfs > 0}

        return IsomericBranching(
            energies=self.energy_bounds,
            products=list(target_names),
            branching_ratios=br,
            parent_nuclide=nuclide,
            reaction=reaction,
            mt=mt,
            lfs_mapping=lfs_mapping,
        )

    def __repr__(self):
        return (f"GENDFLibrary(lib_id={self._lib_id}, "
                f"n_groups={self.n_groups})")
