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
import os
import re
import tempfile
from collections import defaultdict
from datetime import date
from pathlib import Path
from warnings import warn

import h5py
import numpy as np

import openmc
import openmc.checkvalue as cv
from .data import gnds_name
from .endf import Evaluation, get_head_record, get_list_record, get_tab1_record
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

# Filename conventions understood by :func:`_discover_pendf_files`. Every pattern
# is fully anchored (``^...$``) so a name is matched in whole or not at all; this
# keeps recognition deterministic and prevents, e.g., the ``.tendl20NN`` infix
# names below from being partially accepted by the plain TENDL patterns.
_TENDL_RE = re.compile(r'^n-([A-Za-z]+)(\d+)([mn]?)\.pendf$')
_ENDFB_RE = re.compile(r'^ZA(\d{3})(\d{3})(?:\.(\d+))?$')
_JEFF_RE = re.compile(r'^(?:0[kK]|293[kK])-\d+-([A-Za-z]+)-(\d+)([gmn]?)_p\.asc$')
# TENDL-2015 (nuclide-first, reversed vs TENDL-2017), plain or carrying a
# ``.tendl20NN`` version infix: ``Ag107-n.pendf``, ``Ac222m-n.pendf``,
# ``Ag107-n.tendl2015.pendf``.
_TENDL2015_RE = re.compile(r'^([A-Za-z]+)(\d+)([mn]?)-n(?:\.tendl20\d\d)?\.pendf$')
# TENDL-2017 projectile-first stem carrying a ``.tendl20NN`` version infix; the
# loose-file companion to the frozen ``_TENDL_RE`` (which cannot absorb the
# optional infix): ``n-Ag107.tendl2017.pendf``.
_TENDL2017_INFIX_RE = re.compile(r'^n-([A-Za-z]+)(\d+)([mn]?)\.tendl20\d\d\.pendf$')
# TENDL-2019 native pendf: ``Ag107p.asc``, ``Ac222mp.asc``.
_TENDL2019_RE = re.compile(r'^([A-Za-z]+)(\d+)([mn]?)p\.asc$')
# JEFF-3.3: ``47-Ag-107g.jeff33.pendf`` (any two-digit ``.jeffNN`` suffix).
_JEFF33_RE = re.compile(r'^\d+-([A-Za-z]+)-(\d+)([gmn]?)\.jeff\d{2}\.pendf$')
# JENDL-5: ``n_047-Ag-107_300K.dat`` with a free-form temperature token
# (``300K``, ``293.6K``, ...) and an ``m<digit>`` isomer index that doubles as
# the implied LISO (``n_052-Te-123m1_300K.dat`` -> LISO 1).
_JENDL5_RE = re.compile(r'^n_\d{3}-([A-Za-z]+)-(\d+)(?:m(\d))?_[^_]*K\.dat$')

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
    """Raise unless every interpolation region of a TAB1 is lin-lin (INT=2).

    NJOY PENDF is single-region, but a multi-region TAB1 in which every region
    is lin-lin (INT=2) is equivalent to a single lin-lin table, so it is
    accepted; only a region with INT != 2 (a genuinely non-lin-lin scheme) is
    rejected.
    """
    if any(int(i) != 2 for i in tab.interpolation):
        raise ValueError(
            f"{name} MF={mf} MT={mt}: expected lin-lin interpolation (INT=2) "
            f"in every region, got NR={len(tab.breakpoints)}, "
            f"INT={list(tab.interpolation)}."
        )


def _discover_pendf_files(pendf_dir):
    """Find PENDF files in a directory and the isomeric state implied by name.

    A ``_manifest.tsv`` (TENDL convention, columns ``Z El A m url fname``) is
    used when present; otherwise the directory is scanned for the TENDL-2017,
    TENDL-2015, TENDL-2019, ENDF/B PREPRO, JEFF-4.0, JEFF-3.3, and JENDL-5
    filename conventions (including loose ``.tendl20NN``-infixed TENDL names).
    Every convention contributes only the isomeric state (LISO) implied by the
    name; nuclide identity is always taken from the MF=1/451 header afterward.

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
            continue
        m = _TENDL2015_RE.match(name)
        if m is not None:
            files.append((path, _SUFFIX_LISO[m.group(3)]))
            continue
        m = _TENDL2017_INFIX_RE.match(name)
        if m is not None:
            files.append((path, _SUFFIX_LISO[m.group(3)]))
            continue
        m = _TENDL2019_RE.match(name)
        if m is not None:
            files.append((path, _SUFFIX_LISO[m.group(3)]))
            continue
        m = _JEFF33_RE.match(name)
        if m is not None:
            files.append((path, _SUFFIX_LISO[m.group(3)]))
            continue
        m = _JENDL5_RE.match(name)
        if m is not None:
            files.append((path, int(m.group(3)) if m.group(3) else 0))
            continue
    return files


def _write_mf10_partials(mtg, ev, mt, name, path, mapping, decay_lookup,
                         elis_rtol, elis_atol):
    """Write a reaction's MF=10 isomeric production partials as ``LFS`` subgroups.

    Each MF=10 partial for reaction ``mt`` is stored as a subgroup of ``mtg``
    with its IZAP/LFS/QM/QI/ELFS attributes (and, when ``mapping`` is active, a
    baked ``product`` name). A partial whose LFS is unique within the reaction
    keeps the bare ``LFS{lfs}`` name (byte-identical to non-lumped libraries);
    an LFS shared by several distinct IZAP -- different product nuclides, as in
    lumped TENDL MT=5 -- is disambiguated as ``LFS{lfs}_ZAP{izap}``. A true
    ``(IZAP, LFS)`` duplicate warns and is skipped.
    """
    if (10, mt) not in ev.section:
        return

    fo = io.StringIO(ev.section[10, mt])
    _, _, _lis, _liso, ns, _ = get_head_record(fo)
    partials = []
    for _ in range(ns):
        (pqm, pqi, izap, lfs), ptab = get_tab1_record(fo)
        _check_lin_lin(name, 10, mt, ptab)
        partials.append((pqm, pqi, izap, lfs, ptab))

    # Drop true (IZAP, LFS) duplicates, keeping the first occurrence. Uniqueness
    # of an LFS is decided from the distinct (IZAP, LFS) pairs below, so a
    # duplicate never makes the surviving partial's LFS look shared.
    seen = set()
    unique = []
    for pqm, pqi, izap, lfs, ptab in partials:
        if (izap, lfs) in seen:
            warn(f"{path.name}: duplicate MF=10 "
                 f"partial (IZAP={izap}, LFS={lfs}) in {name} "
                 f"MT={mt}; skipping.")
            continue
        seen.add((izap, lfs))
        unique.append((pqm, pqi, izap, lfs, ptab))

    # An LFS carried by more than one product nuclide must be disambiguated by
    # IZAP in the subgroup name; a unique LFS keeps the plain ``LFS{lfs}`` name.
    lfs_izaps = defaultdict(set)
    for _pqm, _pqi, izap, lfs, _pt in unique:
        lfs_izaps[lfs].add(izap)

    # Resolve products per IZAP so a shared LFS bakes the right product for each
    # nuclide; the result is keyed by (IZAP, LFS). With a single IZAP (the
    # common, non-lumped case) this is one call with the same partials as before.
    lfs_to_liso = {}
    if mapping != 'none':
        by_izap = defaultdict(list)
        for pqm, pqi, izap, lfs, _pt in unique:
            by_izap[izap].append(
                {'lfs': lfs, 'izap': izap, 'elfs': pqm - pqi})
        for izap, plist in by_izap.items():
            for lfs, liso in map_lfs_to_liso(
                    plist, decay_lookup, mode=mapping,
                    rtol=elis_rtol, atol=elis_atol,
                    context=f"{name} MT={mt}").items():
                lfs_to_liso[izap, lfs] = liso

    for pqm, pqi, izap, lfs, ptab in unique:
        gname = (f'LFS{lfs}_ZAP{izap}' if len(lfs_izaps[lfs]) > 1
                 else f'LFS{lfs}')
        lg = mtg.create_group(gname)
        lg.attrs['QM'] = pqm
        lg.attrs['QI'] = pqi
        lg.attrs['IZAP'] = izap
        lg.attrs['LFS'] = lfs
        lg.attrs['ELFS'] = pqm - pqi
        if (izap, lfs) in lfs_to_liso:
            lg.attrs['product'] = np.bytes_(gnds_name(
                izap // 1000, izap % 1000,
                lfs_to_liso[izap, lfs]))
        _write_xy(lg, ptab.x, ptab.y)


# MF=2 MT=153 (NJOY PURR) probability-table layout: each URR energy carries a
# leading energy value followed by six band columns -- probability, total,
# elastic, fission, capture, heating -- so NPL = NUNR * (1 + 6*NBAND).
_URR_PTABLE_COLS = 6


def _write_urr_ptables(nuc, ev, name, path):
    """Ingest MF=2 MT=153 probability tables into <nuclide>/urr/<Tkey>/."""
    if (2, 153) not in ev.section:
        return
    fo = io.StringIO(ev.section[2, 153])
    # HEAD: N1 = #xs columns (5), N2 = #bands (NBAND=20)
    _za, _awr, _l1, _l2, _n1, nband = get_head_record(fo)
    # LIST: C1 = temperature [K], L1 = LSSF flag, N2 = #URR energies (NUNR).
    # LSSF (self-shielding flag, carried through from MF=2 MT=151) governs the
    # band convention: 0 -> bands are ABSOLUTE cross sections; 1 -> bands are
    # FACTORS relative to the smooth (infinite-dilution) MF=3 cross section.
    # Empirically verified across all 24 JEFF-3.3 flagged nuclides (LIST L1 is
    # the only record field matching the 5/19 absolute/factor split): LSSF=0 for
    # W182/183/184/186 & Ta181, LSSF=1 for the other 19. Cross-checked against
    # the data (prob-weighted band-total ~1 iff factor-form). The fold needs this
    # to know whether to multiply the bands by the smooth XS (mat_ssf.py).
    (temp, _c2, lssf, _ll2, npl, nunr), values = get_list_record(fo)
    per_energy = 1 + _URR_PTABLE_COLS * nband
    if nunr <= 0 or nband <= 0 or npl != nunr * per_energy:
        warn(f"{path.name}: {name} MF=2 MT=153 NPL={npl} incompatible with "
             f"NUNR={nunr}, NBAND={nband}; skipping probability tables.")
        return
    block = np.asarray(values, dtype=np.float64).reshape(nunr, per_energy)
    energy = np.array(block[:, 0], dtype=np.float64)                    # eV
    table = block[:, 1:].reshape(nunr, _URR_PTABLE_COLS, nband).copy()  # col-major
    # OpenMC stores CUMULATIVE probability in column 0; ENDF gives raw per-band.
    table[:, 0, :] = np.cumsum(table[:, 0, :], axis=1)
    grp = nuc.create_group(f'urr/{round(float(temp))}K')
    grp.attrs['interpolation'] = 2
    grp.attrs['inelastic'] = -1
    grp.attrs['absorption'] = -1
    grp.attrs['multiply_smooth'] = int(lssf)
    grp.create_dataset('energy', data=energy)
    grp.create_dataset('table', data=table)


def _select_urr_tkey(tkeys, temperature, default_temperature):
    """Pick the ``<nuclide>/urr`` temperature key to read.

    A single-temperature PENDF library stores exactly one ``<Tkey>`` (e.g.
    ``'294K'``), which is returned regardless of ``temperature``. With several,
    the key whose temperature is nearest to ``temperature`` is chosen (falling
    back to ``default_temperature`` when ``temperature`` is ``None``). Returns
    ``None`` for an empty ``urr`` group.
    """
    if not tkeys:
        return None
    if len(tkeys) == 1:
        return tkeys[0]
    target = default_temperature if temperature is None else temperature
    if target is None:
        return sorted(tkeys)[0]

    def _temp(k):
        return float(k[:-1]) if k.endswith('K') else float(k)

    return min(tkeys, key=lambda k: abs(_temp(k) - float(target)))


class _TemperatureMismatchError(ValueError):
    """A tape's temperature disagrees with the library temperature.

    Library-level inconsistency: unlike a single unparseable tape, this must
    abort the whole conversion rather than be skipped with a warning.
    """


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
            library = _attr_str(f.attrs, 'library')
            temperature = float(f.attrs['temperature'])
            mapping = _attr_str(f.attrs, 'mapping')
            if self.temperature is None:
                self.library = library
                self.temperature = temperature
                self.mapping = mapping
            else:
                # Directory mode: every file must share the first file's
                # library identity, temperature, and product mapping, or data
                # would be served silently under mismatched metadata. Compare
                # root attributes only (no dataset reads).
                mismatches = []
                if library != self.library:
                    mismatches.append(
                        f"library {library!r} != {self.library!r}")
                if abs(temperature - self.temperature) > 0.1:
                    mismatches.append(
                        f"temperature {temperature} K != {self.temperature} K")
                if mapping != self.mapping:
                    mismatches.append(
                        f"mapping {mapping!r} != {self.mapping!r}")
                if mismatches:
                    for handle in self._files:
                        handle.close()
                    raise ValueError(
                        f"{Path(p).name}: root metadata disagrees with "
                        f"{Path(paths[0]).name}: {'; '.join(mismatches)}.")
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
        rx = self._reaction(nuclide, mt)
        return sorted(int(rx[k].attrs['LFS'])
                      for k in rx if k.startswith('LFS'))

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

    def has_ptables(self, nuclide):
        """Return whether the nuclide carries URR probability tables.

        Parameters
        ----------
        nuclide : str
            GNDS name of the nuclide.

        Returns
        -------
        bool
            ``True`` if a ``<nuclide>/urr`` group is present (ingested from
            MF=2 MT=153 at build time), otherwise ``False``. A nuclide absent
            from the library answers ``False`` gracefully (the group is not
            present) rather than raising.

        """
        group = self._groups.get(nuclide)
        return group is not None and 'urr' in group

    def ptables(self, nuclide, temperature=None):
        """Return the URR probability tables for a nuclide, or ``None``.

        Parameters
        ----------
        nuclide : str
            GNDS name of the nuclide.
        temperature : float, optional
            Requested temperature in kelvin. When a nuclide carries a single
            temperature (the usual case for a single-temperature PENDF library)
            it is returned regardless of this value; when several are present
            the nearest ``<Tkey>`` is chosen. Defaults to the library
            temperature.

        Returns
        -------
        openmc.data.ProbabilityTables or None
            Probability tables read from ``<nuclide>/urr/<Tkey>``, or ``None``
            if the nuclide has no ``/urr`` group (including a nuclide absent
            from the library).

        """
        group = self._groups.get(nuclide)
        if group is None or 'urr' not in group:
            return None
        urr = group['urr']
        tkey = _select_urr_tkey(
            list(urr.keys()), temperature, self.temperature)
        if tkey is None:
            return None
        return openmc.data.urr.ProbabilityTables.from_hdf5(urr[tkey])

    def close(self):
        """Close the underlying HDF5 file handles."""
        for f in self._files:
            f.close()
        self._files = []
        self._groups = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

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
            Directory of PENDF files (TENDL, ENDF/B, JEFF, or JENDL filename
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
        cv.check_value('mapping', mapping, ('none', 'elis', 'lfs_order'))

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
        n_converted = 0
        n_skipped = 0

        # Convert into a temporary file in the destination directory and
        # atomically replace ``out`` only on success, so an unparseable file
        # mid-run never leaves a partially written (and unreadable) library
        # behind. Files that fail to parse are skipped with a warning.
        fd, tmp_name = tempfile.mkstemp(dir=str(out.parent), suffix='.h5.tmp')
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            with h5py.File(tmp, 'w') as h5:
                for path, implied_liso in entries:
                    name = None
                    created_here = False
                    try:
                        ev = Evaluation(path)
                        Z = ev.target['atomic_number']
                        A = ev.target['mass_number']
                        liso = ev.target['isomeric_state']
                        name = gnds_name(Z, A, liso)
                        if implied_liso != liso:
                            warn(f"{path.name}: filename implies isomeric state "
                                 f"{implied_liso} but MF=1/451 header gives "
                                 f"{liso}; trusting header ({name}).")

                        T = ev.target['temperature']
                        if lib_temperature is None:
                            lib_temperature = T
                        elif abs(T - lib_temperature) > 0.1:
                            raise _TemperatureMismatchError(
                                f"file temperature {T} K disagrees with library "
                                f"temperature {lib_temperature} K by more than "
                                "0.1 K.")

                        if name in h5:
                            warn(f"{path.name}: duplicate nuclide {name}; "
                                 "skipping.")
                            continue

                        nuc = h5.create_group(name)
                        created_here = True
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
                            get_head_record(fo)             # MF=3 HEAD (discarded)
                            (QM, QI, _l1, _lr), tab = get_tab1_record(fo)
                            _check_lin_lin(name, 3, mt, tab)
                            mtg = nuc.create_group(f'MT{mt}')
                            mtg.attrs['QM'] = QM
                            mtg.attrs['QI'] = QI
                            _write_xy(mtg, tab.x, tab.y)

                            # MF=10 isomeric production partials for this reaction
                            _write_mf10_partials(mtg, ev, mt, name, path,
                                                 mapping, decay_lookup,
                                                 elis_rtol, elis_atol)

                        # MF=10 partials are written only alongside their MF=3
                        # sibling (loop above). Warn about any MF=10 reaction
                        # with no MF=3 section so the silent drop is visible; a
                        # total is not synthesized from the partials (that would
                        # invent data the format requires from MF=3).
                        mf3_mts = {mt for (mf, mt) in ev.section if mf == 3}
                        mf10_mts = {mt for (mf, mt) in ev.section if mf == 10}
                        for mt in sorted(mf10_mts - mf3_mts):
                            warn(f"{path.name}: {name} MF=10 MT={mt} has no "
                                 f"MF=3 section; isomeric partials dropped.")

                        # MF=2 MT=153 probability tables (URR). Ingested
                        # unconditionally (independent of keep_extra_mts, which
                        # only gates the MF=3 loop) so a rebuilt library always
                        # carries /urr for downstream self-shielding.
                        _write_urr_ptables(nuc, ev, name, path)

                        n_converted += 1
                    except _TemperatureMismatchError:
                        # Library-level inconsistency, not a bad file: abort
                        # (the temp-file cleanup leaves no partial output).
                        raise
                    except Exception as exc:
                        # Drop any partially written group for this file so a
                        # later file for the same nuclide can still convert.
                        if created_here and name in h5:
                            del h5[name]
                        warn(f"skipping {path}: {exc}")
                        n_skipped += 1
                        continue

                if n_converted == 0:
                    raise ValueError(
                        f"No PENDF files in {pendf_dir} could be converted "
                        f"({n_skipped} skipped).")

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

            os.replace(tmp, out)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

        return PendfLibrary(out)
