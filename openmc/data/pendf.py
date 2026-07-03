"""Preprocessing and access for pointwise PENDF cross-section libraries.

PENDF files are ENDF-6 formatted evaluations in which the resonance region has
been reconstructed and Doppler broadened to a single temperature (via NJOY
RECONR/BROADR). This module extracts the reconstructed pointwise cross sections
(MF=3) and isomeric production cross sections (MF=10) into a compact HDF5
representation, and provides a light-weight reader (:class:`PendfLibrary`) with
lazy access to the tabulated data.

The HDF5 schema is documented in the PENDF-MGB implementation plan (§4.1) and is
frozen. Cross sections are stored per nuclide, per reaction (MT), with MF=10
partial cross sections nested under their reaction as ``LFS{lfs}`` subgroups.

.. versionadded:: 0.15.4
"""

import io
import re
from datetime import date
from pathlib import Path
from warnings import warn

import h5py
import numpy as np

import openmc
from .data import gnds_name
from .endf import Evaluation, get_head_record, get_tab1_record
from .isomeric import (ELIS_ATOL, ELIS_RTOL, map_lfs_to_liso,
                       parse_decay_isomeric_levels)

__all__ = ['PendfLibrary']

# Version of the PENDF HDF5 format written/read by this module
_FORMAT_VERSION = 1

# MF=3 reactions retained only when ``keep_extra_mts=True``: resonance
# parameters/derived (151-153), particle production (203-207), average
# secondary quantities (251-253), and HEATR heating/damage (301-450).
_EXTRA_MTS = frozenset(
    {151, 152, 153} | set(range(203, 208)) | {251, 252, 253} | set(range(301, 451))
)

# Filename conventions understood by :func:`_discover_pendf_files`
_TENDL_RE = re.compile(r'^n-([A-Za-z]+)(\d+)([mn]?)\.pendf$')
_ENDFB_RE = re.compile(r'^ZA(\d{3})(\d{3})(?:\.(\d+))?$')
_JEFF_RE = re.compile(r'^(?:0[kK]|293[kK])-\d+-([A-Za-z]+)-(\d+)([gmn]?)_p\.asc$')

# Isomer suffix -> LISO (isomeric state) implied by a filename
_SUFFIX_LISO = {'': 0, 'g': 0, 'm': 1, 'n': 2}


def _attr_str(attrs, key):
    """Return an HDF5 string attribute as a ``str``."""
    value = attrs[key]
    return value.decode() if isinstance(value, bytes) else str(value)


def _write_xy(group, x, y):
    """Write an (energy, xs) pair as chunked, gzip-compressed float64 datasets."""
    group.create_dataset('energy', data=np.asarray(x, dtype=np.float64),
                         compression='gzip', shuffle=True)
    group.create_dataset('xs', data=np.asarray(y, dtype=np.float64),
                         compression='gzip', shuffle=True)


def _check_lin_lin(name, mf, mt, tab):
    """Raise unless a TAB1 record is a single lin-lin (NR=1, INT=2) region."""
    if len(tab.breakpoints) != 1 or int(tab.interpolation[0]) != 2:
        raise ValueError(
            f"{name} MF={mf} MT={mt}: expected a single lin-lin region "
            f"(NR=1, INT=2), got NR={len(tab.breakpoints)}, "
            f"INT={list(tab.interpolation)}."
        )


def _discover_pendf_files(pendf_dir):
    """Find PENDF files in a directory and the isomeric state implied by name.

    A ``_manifest.tsv`` (TENDL convention, columns ``Z El A m url fname``) is
    used when present; otherwise the directory is scanned for the TENDL, ENDF/B,
    and JEFF filename conventions.

    Parameters
    ----------
    pendf_dir : pathlib.Path
        Directory to scan.

    Returns
    -------
    list of (pathlib.Path, int)
        Each PENDF file and the isomeric state (LISO) implied by its filename.
        The implied state is only used to validate the header-derived identity.

    """
    pendf_dir = Path(pendf_dir)
    manifest = pendf_dir / '_manifest.tsv'
    files = []

    if manifest.is_file():
        with open(manifest) as fh:
            header = fh.readline().rstrip('\n').split('\t')
            try:
                i_m = header.index('m')
                i_f = header.index('fname')
            except ValueError:
                i_m, i_f = 3, 5
            for line in fh:
                cols = line.rstrip('\n').split('\t')
                if len(cols) <= max(i_m, i_f):
                    continue
                path = pendf_dir / cols[i_f]
                if path.is_file():
                    files.append((path, _SUFFIX_LISO.get(cols[i_m].strip(), 0)))
        return files

    for path in sorted(pendf_dir.iterdir()):
        name = path.name
        m = _TENDL_RE.match(name)
        if m is not None:
            files.append((path, _SUFFIX_LISO[m.group(3)]))
            continue
        m = _ENDFB_RE.match(name)
        if m is not None:
            if int(m.group(1)) == 0 and int(m.group(2)) == 1:
                continue  # skip ZA000001 (free-neutron placeholder)
            files.append((path, int(m.group(3)) if m.group(3) else 0))
            continue
        m = _JEFF_RE.match(name)
        if m is not None:
            files.append((path, _SUFFIX_LISO[m.group(3)]))
    return files


class PendfLibrary:
    """Pointwise PENDF cross-section library backed by preprocessed HDF5.

    .. versionadded:: 0.15.4

    Parameters
    ----------
    path : str or path-like
        Either a single ``.h5`` file written by :meth:`from_endf_directory`, or
        a directory containing such files (one per nuclide or a single combined
        file). Datasets are read lazily.

    Attributes
    ----------
    nuclides : list of str
        GNDS names of the nuclides in the library.
    temperature : float
        Temperature of the library in kelvin.
    mapping : str
        Product-mapping mode used at preprocessing time
        (``'none'``, ``'elis'``, or ``'lfs_order'``).
    library : str
        Name of the source data library.

    """

    def __init__(self, path):
        path = Path(path)
        if path.is_dir():
            paths = sorted(path.glob('*.h5'))
            if not paths:
                raise ValueError(f"No .h5 files found in directory {path}")
        elif path.is_file():
            paths = [path]
        else:
            raise FileNotFoundError(str(path))

        self._files = []
        self._groups = {}
        self.library = None
        self.temperature = None
        self.mapping = None
        for p in paths:
            f = h5py.File(p, 'r')
            self._files.append(f)
            if self.temperature is None:
                self.library = _attr_str(f.attrs, 'library')
                self.temperature = float(f.attrs['temperature'])
                self.mapping = _attr_str(f.attrs, 'mapping')
            for name, group in f.items():
                if name in self._groups:
                    warn(f"Nuclide {name} appears in more than one file; "
                         f"using the first occurrence.")
                    continue
                self._groups[name] = group
        self.nuclides = sorted(self._groups)

    def __repr__(self):
        return (f"<PendfLibrary: {len(self.nuclides)} nuclides, "
                f"{self.library}, {self.temperature} K>")

    def _nuclide(self, nuclide):
        try:
            return self._groups[nuclide]
        except KeyError:
            raise KeyError(f"Nuclide {nuclide!r} not in library.")

    def _reaction(self, nuclide, mt):
        group = self._nuclide(nuclide).get(f'MT{mt}')
        if group is None:
            raise KeyError(f"Nuclide {nuclide!r} has no MF=3 reaction MT={mt}.")
        return group

    def reactions(self, nuclide):
        """Return the MTs with MF=3 cross sections for a nuclide.

        Parameters
        ----------
        nuclide : str
            GNDS name of the nuclide.

        Returns
        -------
        list of int
            Sorted reaction MT numbers.

        """
        return sorted(int(k[2:]) for k in self._nuclide(nuclide)
                      if k.startswith('MT'))

    def pathways(self, nuclide, mt):
        """Return the MF=10 partial (LFS) values available for a reaction.

        Parameters
        ----------
        nuclide : str
            GNDS name of the nuclide.
        mt : int
            Reaction MT number.

        Returns
        -------
        list of int
            Sorted LFS (final-level) indices; empty if the reaction has no
            MF=10 partials.

        """
        return sorted(int(k[3:]) for k in self._reaction(nuclide, mt)
                      if k.startswith('LFS'))

    def xs(self, nuclide, mt):
        """Return the MF=3 cross section for a reaction.

        Parameters
        ----------
        nuclide : str
            GNDS name of the nuclide.
        mt : int
            Reaction MT number.

        Returns
        -------
        tuple of numpy.ndarray
            Energy grid (eV) and cross section (barn).

        """
        group = self._reaction(nuclide, mt)
        return group['energy'][()], group['xs'][()]

    def pathway_xs(self, nuclide, mt, lfs):
        """Return the MF=10 partial cross section for a reaction/final level.

        Parameters
        ----------
        nuclide : str
            GNDS name of the nuclide.
        mt : int
            Reaction MT number.
        lfs : int
            Final-level index (LFS).

        Returns
        -------
        tuple of numpy.ndarray
            Energy grid (eV) and partial cross section (barn).

        """
        group = self._reaction(nuclide, mt).get(f'LFS{lfs}')
        if group is None:
            raise KeyError(
                f"Nuclide {nuclide!r} MT={mt} has no MF=10 partial LFS={lfs}.")
        return group['energy'][()], group['xs'][()]

    def product(self, nuclide, mt, lfs):
        """Return the baked product name for an MF=10 partial, if mapped.

        Parameters
        ----------
        nuclide : str
            GNDS name of the nuclide.
        mt : int
            Reaction MT number.
        lfs : int
            Final-level index (LFS).

        Returns
        -------
        str or None
            GNDS product name if the library was written with a product
            mapping, otherwise ``None``.

        """
        group = self._reaction(nuclide, mt).get(f'LFS{lfs}')
        if group is None:
            raise KeyError(
                f"Nuclide {nuclide!r} MT={mt} has no MF=10 partial LFS={lfs}.")
        if 'product' in group.attrs:
            return _attr_str(group.attrs, 'product')
        return None

    def close(self):
        """Close the underlying HDF5 file handles."""
        for f in self._files:
            f.close()
        self._files = []
        self._groups = {}

    @staticmethod
    def from_endf_directory(pendf_dir, out, library=None, temperature=None,
                            keep_extra_mts=False, mapping='none',
                            decay_file=None, elis_rtol=ELIS_RTOL,
                            elis_atol=ELIS_ATOL):
        """Preprocess a directory of PENDF files into an HDF5 library.

        Nuclide identity (Z, A, isomeric state) is always taken from the
        MF=1/MT=451 header and validated against the isomeric state implied by
        the filename (a mismatch warns; the header is trusted). MF=3 cross
        sections are stored per reaction; MF=10 isomeric production cross
        sections are stored as ``LFS{lfs}`` subgroups of their reaction.

        Parameters
        ----------
        pendf_dir : str or path-like
            Directory of PENDF files (TENDL, ENDF/B, or JEFF filename
            conventions; a ``_manifest.tsv`` is used when present).
        out : str or path-like
            Output ``.h5`` file (one file per library and temperature).
        library : str, optional
            Name of the source data library recorded in the file. Defaults to
            ``'unknown'``.
        temperature : float, optional
            Library temperature in kelvin. If given, every file's own
            temperature must agree with it to within 0.1 K. If omitted, the
            temperature of the first file is adopted and enforced on the rest.
        keep_extra_mts : bool
            Retain non-activation MF=3 reactions (particle production, HEATR
            heating/damage, average secondary quantities, resonance
            parameters). Default is ``False``.
        mapping : {'none', 'elis', 'lfs_order'}
            Product-mapping mode for MF=10 partials. ``'none'`` stores only the
            raw IZAP/LFS/ELFS attributes. ``'elis'`` bakes a product GNDS name
            onto each partial by matching ELFS (= QM - QI) to decay-library
            excitation energies (:mod:`openmc.data.isomeric`); ``'lfs_order'``
            uses the positional FISPACT-like fallback. Both require
            ``decay_file``.
        decay_file : str or path-like, optional
            Decay data used for product mapping (required when ``mapping`` is not
            ``'none'``). A directory of per-nuclide ENDF decay files or a single
            concatenated decay file.
        elis_rtol, elis_atol : float
            Relative and absolute tolerances for ELFS/ELIS matching
            (``mapping='elis'``). Default to the values in
            :mod:`openmc.data.isomeric`.

        Returns
        -------
        PendfLibrary
            Reader for the file just written.

        """
        if mapping not in ('none', 'elis', 'lfs_order'):
            raise ValueError(
                f"mapping must be 'none', 'elis', or 'lfs_order', got "
                f"{mapping!r}.")

        pendf_dir = Path(pendf_dir)
        out = Path(out)
        entries = _discover_pendf_files(pendf_dir)
        if not entries:
            raise ValueError(f"No PENDF files found in {pendf_dir}.")

        decay_lookup = None
        if mapping != 'none':
            if decay_file is None:
                raise ValueError(
                    f"mapping={mapping!r} requires decay_file.")
            decay_lookup = parse_decay_isomeric_levels(decay_file)

        lib_temperature = temperature
        with h5py.File(out, 'w') as h5:
            for path, implied_liso in entries:
                ev = Evaluation(path)
                Z = ev.target['atomic_number']
                A = ev.target['mass_number']
                liso = ev.target['isomeric_state']
                name = gnds_name(Z, A, liso)
                if implied_liso != liso:
                    warn(f"{path.name}: filename implies isomeric state "
                         f"{implied_liso} but MF=1/451 header gives {liso}; "
                         f"trusting header ({name}).")

                T = ev.target['temperature']
                if lib_temperature is None:
                    lib_temperature = T
                elif abs(T - lib_temperature) > 0.1:
                    raise ValueError(
                        f"{path.name}: file temperature {T} K disagrees with "
                        f"library temperature {lib_temperature} K by more than "
                        "0.1 K.")

                if name in h5:
                    warn(f"{path.name}: duplicate nuclide {name}; skipping.")
                    continue

                nuc = h5.create_group(name)
                nuc.attrs['ZA'] = 1000 * Z + A
                nuc.attrs['AWR'] = ev.target['mass']
                nuc.attrs['LIS'] = ev.target['state']
                nuc.attrs['LISO'] = liso
                nuc.attrs['ELIS'] = ev.target['excitation_energy']
                nuc.attrs['MAT'] = ev.material
                nuc.attrs['source_file'] = np.bytes_(path.name)

                for (mf, mt), text in sorted(ev.section.items()):
                    if mf != 3:
                        continue
                    if mt in _EXTRA_MTS and not keep_extra_mts:
                        continue
                    fo = io.StringIO(text)
                    get_head_record(fo)                 # MF=3 HEAD (discarded)
                    (QM, QI, _l1, _lr), tab = get_tab1_record(fo)
                    _check_lin_lin(name, 3, mt, tab)
                    mtg = nuc.create_group(f'MT{mt}')
                    mtg.attrs['QM'] = QM
                    mtg.attrs['QI'] = QI
                    _write_xy(mtg, tab.x, tab.y)

                    # MF=10 isomeric production partials for this reaction
                    if (10, mt) in ev.section:
                        fo = io.StringIO(ev.section[10, mt])
                        _, _, _lis, _liso, ns, _ = get_head_record(fo)
                        partials = []
                        for _ in range(ns):
                            (pqm, pqi, izap, lfs), ptab = get_tab1_record(fo)
                            _check_lin_lin(name, 10, mt, ptab)
                            partials.append((pqm, pqi, izap, lfs, ptab))

                        lfs_to_liso = {}
                        if mapping != 'none':
                            lfs_to_liso = map_lfs_to_liso(
                                [{'lfs': lfs, 'izap': izap, 'elfs': pqm - pqi}
                                 for pqm, pqi, izap, lfs, _pt in partials],
                                decay_lookup, mode=mapping,
                                rtol=elis_rtol, atol=elis_atol,
                                context=f"{name} MT={mt}")

                        for pqm, pqi, izap, lfs, ptab in partials:
                            lg = mtg.create_group(f'LFS{lfs}')
                            lg.attrs['QM'] = pqm
                            lg.attrs['QI'] = pqi
                            lg.attrs['IZAP'] = izap
                            lg.attrs['LFS'] = lfs
                            lg.attrs['ELFS'] = pqm - pqi
                            if lfs in lfs_to_liso:
                                lg.attrs['product'] = np.bytes_(gnds_name(
                                    izap // 1000, izap % 1000, lfs_to_liso[lfs]))
                            _write_xy(lg, ptab.x, ptab.y)

            h5.attrs['format_version'] = _FORMAT_VERSION
            h5.attrs['library'] = np.bytes_(
                'unknown' if library is None else str(library))
            h5.attrs['temperature'] = float(lib_temperature)
            h5.attrs['source_path'] = np.bytes_(str(pendf_dir))
            h5.attrs['created'] = np.bytes_(date.today().isoformat())
            h5.attrs['mapping'] = np.bytes_(mapping)
            if mapping != 'none':
                h5.attrs['decay_file'] = np.bytes_(str(decay_file))
                if mapping == 'elis':
                    h5.attrs['elis_rtol'] = float(elis_rtol)
                    h5.attrs['elis_atol'] = float(elis_atol)
            h5.attrs['openmc_version'] = np.bytes_(openmc.__version__)

        return PendfLibrary(out)
