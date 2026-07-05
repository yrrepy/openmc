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

    def pathways(self, nuclide: str, mt: int) -> list[int]:
        """Return the ascending MF=10 LFS levels stored for ``(nuclide, mt)``.

        A lumped reaction may repeat an LFS: disambiguated ``LFS<l>_ZAP<izap>``
        subgroups yield that level once per product.
        """
        group = self._file[f'{nuclide}/MT{mt}']
        return sorted(
            int(group[name].attrs['LFS']) for name in group if name.startswith('LFS'))

    def product(self, nuclide: str, mt: int, lfs: int):
        """Return the baked GNDS product name for a partial, or ``None``."""
        attrs = self._file[f'{nuclide}/MT{mt}/LFS{lfs}'].attrs
        prod = attrs.get('product')
        return _decode(prod) if prod is not None else None

    def xs_g(self, nuclide: str, mt: int) -> np.ndarray:
        """Return the MF=3 group cross section [b], length ``n_groups``."""
        return np.asarray(
            self._file[f'{nuclide}/MT{mt}/xs_g'][()], dtype=np.float64)

    def pathway_xs_g(self, nuclide: str, mt: int, lfs: int) -> np.ndarray:
        """Return an MF=10 partial group cross section [b], length ``n_groups``."""
        return np.asarray(
            self._file[f'{nuclide}/MT{mt}/LFS{lfs}/xs_g'][()], dtype=np.float64)
