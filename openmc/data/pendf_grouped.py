"""Reader for pre-binned grouped PENDF HDF5 libraries.

A grouped PENDF library stores group-averaged cross sections on a fixed energy
group structure, so the depletion collapse can skip the runtime flat-weighting
of a pointwise :class:`~openmc.data.PendfLibrary`. The file schema mirrors the
pointwise layout one level deeper: ``/<Nuclide>/MT<mt>/xs_g`` holds the MF=3
group cross section and ``/<Nuclide>/MT<mt>/LFS<l>/xs_g`` each MF=10 partial,
all length ``n_groups`` on the shared ``/group_edges`` grid. The class exposes
the same duck-typed accessor names as the pointwise reader
(``nuclides``, ``reactions``, ``pathways``, ``product``) plus the grouped
accessors :meth:`xs_g` and :meth:`pathway_xs_g`, so
:func:`openmc.deplete.microxs._build_xs_table_pendf` can source rows from either
library without rebinning.

.. versionadded:: 0.15.4
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from openmc.checkvalue import PathLike

#: ``format`` root attribute identifying a grouped PENDF library.
GROUPED_FORMAT = 'pendf-grouped'

#: Highest grouped-schema ``version`` root attribute this reader understands.
#: 1 -- MF=10 subgroups always named ``LFS<l>``; 2 -- a shared LFS may be
#: named ``LFS<l>_ZAP<izap>``. A file stamped higher was written by a newer
#: OpenMC and is rejected.
GROUPED_VERSION = 2


def _decode(value):
    """Return a str for a bytes/np.bytes_ attribute, else the value itself."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.bytes_):
        return value.decode()
    return value


class GroupedPendfLibrary:
    """Pre-binned grouped PENDF library backed by an HDF5 file.

    .. versionadded:: 0.15.4

    Parameters
    ----------
    path : path-like
        Path to a grouped PENDF HDF5 file written by ``tools/pendf_group_bin.py``
        (root attribute ``format='pendf-grouped'``).

    Attributes
    ----------
    group_edges : numpy.ndarray
        Ascending energy group boundaries in [eV], length ``n_groups + 1``.
    nuclides : list of str
        GNDS nuclide names present in the library.
    """

    def __init__(self, path: PathLike):
        self._path = Path(path)
        self._file = h5py.File(self._path, 'r')
        # Any failure while validating/reading the just-opened file must close
        # the handle before propagating, otherwise a caller that catches the
        # error leaks the open HDF5 file.
        try:
            fmt = _decode(self._file.attrs.get('format'))
            if fmt != GROUPED_FORMAT:
                raise ValueError(
                    f"{self._path} is not a grouped PENDF library "
                    f"(format={fmt!r}, expected {GROUPED_FORMAT!r}).")
            # Forward-compat: a file stamped a newer schema version than this
            # reader supports was written by a newer OpenMC. A missing attr or a
            # version <= supported is an older (valid) file.
            file_version = self._file.attrs.get('version')
            if file_version is not None and file_version > GROUPED_VERSION:
                raise ValueError(
                    f"{self._path}: grouped PENDF version {int(file_version)} "
                    f"is newer than the supported version {GROUPED_VERSION}; "
                    f"this file was written by a newer OpenMC.")
            if 'group_edges' not in self._file:
                raise ValueError(
                    f"{self._path} is not a grouped PENDF library "
                    f"(missing 'group_edges' dataset).")
            self._group_edges = np.asarray(
                self._file['group_edges'][()], dtype=np.float64)
            self._nuclides = [
                name for name, obj in self._file.items()
                if isinstance(obj, h5py.Group)]
        except Exception:
            self._file.close()
            raise

    def close(self):
        """Close the backing HDF5 file."""
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def group_edges(self) -> np.ndarray:
        return self._group_edges

    @property
    def nuclides(self) -> list[str]:
        return self._nuclides

    def reactions(self, nuclide: str) -> list[int]:
        """Return the MT numbers with grouped MF=3 data for ``nuclide``."""
        return sorted(
            int(name[2:]) for name in self._file[nuclide]
            if name.startswith('MT'))

    def pathways(self, nuclide: str, mt: int) -> list[tuple[int, int]]:
        """Return the ``(LFS, IZAP)`` pairs of the MF=10 partials for a reaction.

        One pair per stored partial, sorted by ``(lfs, izap)``. A lumped
        reaction repeats an LFS: an LFS shared by several product nuclides is
        stored as ``LFS<l>_ZAP<izap>`` subgroups, one per product.
        """
        group = self._file[f'{nuclide}/MT{mt}']
        return sorted(
            (int(group[name].attrs['LFS']), int(group[name].attrs['IZAP']))
            for name in group if name.startswith('LFS'))

    def _partial(self, nuclide: str, mt: int, lfs: int, izap):
        """Return the single MF=10 subgroup matching ``lfs`` (and ``izap``).

        Scans the reaction's ``LFS*`` subgroups for one whose ``LFS`` attribute
        equals ``lfs`` and, when ``izap`` is not ``None``, whose ``IZAP`` equals
        it. Raises ``KeyError`` if none match and ``ValueError`` if an LFS shared
        by several products is requested without an ``izap`` to disambiguate.
        """
        group = self._file[f'{nuclide}/MT{mt}']
        matches = [group[name] for name in group
                   if name.startswith('LFS')
                   and int(group[name].attrs['LFS']) == lfs
                   and (izap is None or int(group[name].attrs['IZAP']) == izap)]
        if not matches:
            extra = f', IZAP={izap}' if izap is not None else ''
            raise KeyError(
                f"Nuclide {nuclide!r} MT={mt} has no MF=10 partial "
                f"LFS={lfs}{extra}.")
        if len(matches) > 1:
            izaps = sorted(int(g.attrs['IZAP']) for g in matches)
            raise ValueError(
                f"Nuclide {nuclide!r} MT={mt} LFS={lfs} is shared by products "
                f"with IZAP {izaps}; pass izap to select one.")
        return matches[0]

    def product(self, nuclide: str, mt: int, lfs: int, izap=None):
        """Return the baked GNDS product name for a partial, or ``None``.

        ``izap`` selects among the products of a lumped LFS; omit it for a
        non-lumped reaction (a unique LFS).
        """
        prod = self._partial(nuclide, mt, lfs, izap).attrs.get('product')
        return _decode(prod) if prod is not None else None

    def xs_g(self, nuclide: str, mt: int) -> np.ndarray:
        """Return the MF=3 group cross section [b], length ``n_groups``."""
        return np.asarray(
            self._file[f'{nuclide}/MT{mt}/xs_g'][()], dtype=np.float64)

    def pathway_xs_g(self, nuclide: str, mt: int, lfs: int,
                     izap=None) -> np.ndarray:
        """Return an MF=10 partial group cross section [b], length ``n_groups``.

        ``izap`` selects among the products of a lumped LFS; omit it for a
        non-lumped reaction (a unique LFS).
        """
        return np.asarray(
            self._partial(nuclide, mt, lfs, izap)['xs_g'][()], dtype=np.float64)
