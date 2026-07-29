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

# Package roof for the GENDF stack. ``library`` is the only submodule imported
# here: the others (``collapse``, ``helpers``, ``chain_io``, ``operators``)
# import from ``openmc.deplete`` themselves and would cycle if pulled in at
# package-init time. They are reached as ordinary submodules
# (``openmc.deplete.gendf.collapse``), so this file never needs to list them.
from . import library as _library
from .library import *

# ``__all__`` is taken verbatim from the library module so that
# ``from .gendf import *`` in ``openmc/deplete/__init__.py`` re-exports exactly
# the same narrow public set as the pre-package module did.
from .library import __all__

# Names that consumers import from ``openmc.deplete.gendf`` but that are not in
# the narrow ``__all__`` (private helpers, backend classes, data maps, warn-once
# stores -- the stores are mutated in place, so re-binding them here is safe).
from .library import (
    DecayState,
    MT_TO_REACTION,
    REACTION_TO_MT,
    _CppGENDFLibrary,
    _PythonGENDFLibrary,
    _WARNED_MF10_DUPLICATE_LFS,
    _WARNED_RUNTIME_BRANCHING,
    _library_warn_key,
    _warn_runtime_branching,
    elis_match,
    get_product_name,
    lookup_liso,
    parse_decay_isomeric_levels,
)


def __getattr__(name):
    """Forward any remaining former module-level name to :mod:`.library`."""
    try:
        return getattr(_library, name)
    except AttributeError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}") from None
