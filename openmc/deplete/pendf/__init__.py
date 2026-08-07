"""PENDF Depletion Support Module

This package holds the PENDF depletion support extracted from
:mod:`openmc.deplete.microxs`:

- ``collapse`` -- pointwise-PENDF flux collapse
- ``ground`` -- ground-pathway physics
- ``chain_check`` -- chain/stamp consistency checks

.. versionadded:: 0.15.4
"""

# Package roof for the PENDF stack. NO submodule is imported here: all three
# (``chain_check``, ``ground``, ``collapse``) import from
# ``openmc.deplete.microxs`` themselves and would cycle if pulled in at
# ``openmc.deplete`` package-init time. They are reached as ordinary submodules
# (``openmc.deplete.pendf.collapse``, and so on). This boundary is fixed: this
# file is never edited again.
