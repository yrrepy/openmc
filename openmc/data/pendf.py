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

This module also provides :class:`GroupedPendfLibrary`, a reader for pre-binned
grouped PENDF libraries that store group-averaged cross sections on a fixed
energy group structure, so the depletion collapse can skip the runtime
flat-weighting of a pointwise :class:`PendfLibrary`. The grouped schema mirrors
the pointwise layout one level deeper (``/<Nuclide>/MT<mt>/xs_g`` for the MF=3
group cross section and ``/<Nuclide>/MT<mt>/LFS<l>/xs_g`` for each MF=10 partial,
all length ``n_groups`` on a shared ``/group_edges`` grid) and exposes the same
duck-typed accessors so the collapse can source rows from either reader without
rebinning.

.. versionadded:: 0.15.4
"""

from __future__ import annotations

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
from openmc.checkvalue import PathLike
from .data import gnds_name
from .endf import Evaluation, get_head_record, get_list_record, get_tab1_record
from .urr import ProbabilityTables

__all__ = ['PendfLibrary', 'GroupedPendfLibrary']

# Version of the PENDF HDF5 format written/read by this module.
#   1 -- MF=10 subgroups are always named ``LFS{lfs}``.
#   2 -- an LFS shared by >=2 distinct product IZAPs is named
#        ``LFS{lfs}_ZAP{izap}`` (a unique LFS keeps the bare ``LFS{lfs}``).
# The library is written source-faithful (raw IZAP/LFS/QM/QI/ELFS attrs only);
# the isomer<->LFS product mapping lives on the depletion chain, not baked here.
# Files written by older OpenMC carried baked ``product``/``mapping`` attrs; the
# reader accepts them and ignores those attrs (see :class:`PendfLibrary`).
_FORMAT_VERSION = 2

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


def _iter_mf10_partials(ev, mt, name):
    """Yield the MF=10 isomeric-production partials of reaction ``mt``.

    Each yielded tuple is ``(QM, QI, IZAP, LFS, tab)`` -- the partial's mass-
    difference Q, level Q, product ZA identifier, final-level index, and the
    TAB1 (energy, xs) record. Yields nothing when the evaluation has no MF=10
    section for ``mt``. Every partial is checked for lin-lin interpolation.

    Shared by :func:`_write_mf10_partials` (HDF5 build) and the ASC-tape source
    adapter in ``tools/add_pendf_isomeric_branching_to_chain.py`` so the ENDF-6
    record parsing lives in one place.
    """
    if (10, mt) not in ev.section:
        return
    fo = io.StringIO(ev.section[10, mt])
    _, _, _lis, _liso, ns, _ = get_head_record(fo)
    for _ in range(ns):
        (pqm, pqi, izap, lfs), ptab = get_tab1_record(fo)
        _check_lin_lin(name, 10, mt, ptab)
        yield pqm, pqi, izap, lfs, ptab


def _write_mf10_partials(mtg, ev, mt, name, path):
    """Write a reaction's MF=10 isomeric production partials as ``LFS`` subgroups.

    Each MF=10 partial for reaction ``mt`` is stored as a subgroup of ``mtg``
    with its raw IZAP/LFS/QM/QI/ELFS attributes (source-faithful; no product
    name is baked -- the isomer<->LFS mapping lives on the depletion chain). A
    partial whose LFS is unique within the reaction keeps the bare ``LFS{lfs}``
    name (byte-identical to non-lumped libraries); an LFS shared by several
    distinct IZAP -- different product nuclides, as in lumped TENDL MT=5 -- is
    disambiguated as ``LFS{lfs}_ZAP{izap}``. A true ``(IZAP, LFS)`` duplicate
    warns and is skipped.
    """
    partials = list(_iter_mf10_partials(ev, mt, name))
    if not partials:
        return

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

    for pqm, pqi, izap, lfs, ptab in unique:
        gname = (f'LFS{lfs}_ZAP{izap}' if len(lfs_izaps[lfs]) > 1
                 else f'LFS{lfs}')
        lg = mtg.create_group(gname)
        lg.attrs['QM'] = pqm
        lg.attrs['QI'] = pqi
        lg.attrs['IZAP'] = izap
        lg.attrs['LFS'] = lfs
        lg.attrs['ELFS'] = pqm - pqi
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
    mapping : str or None
        Legacy product-mapping mode read from an old baked file's root attr
        (``'none'``, ``'elis'``, or ``'lfs_order'``), or ``None`` for a de-baked
        (source-faithful) file. Informational only -- the isomer<->LFS product
        mapping is now carried on the depletion chain, not the library.
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
            # Forward-compat: a file stamped a newer format than this reader
            # understands was written by a newer OpenMC. A missing attr or a
            # version <= supported is an older (valid) file. Checked once per
            # file, before any dataset access.
            file_version = f.attrs.get('format_version')
            if file_version is not None and file_version > _FORMAT_VERSION:
                for handle in self._files:
                    handle.close()
                raise ValueError(
                    f"{p}: PENDF format_version {int(file_version)} is newer "
                    f"than the supported version {_FORMAT_VERSION}; this file "
                    f"was written by a newer OpenMC.")
            library = _attr_str(f.attrs, 'library')
            temperature = float(f.attrs['temperature'])
            # ``mapping`` is a legacy root attr from the old baked-product build
            # path; de-baked files omit it. Read it when present (informational
            # only -- product naming is now chain-sourced), else ``None``.
            mapping = _attr_str(f.attrs, 'mapping') if 'mapping' in f.attrs \
                else None
            if self.temperature is None:
                self.library = library
                self.temperature = temperature
                self.mapping = mapping
            else:
                # Directory mode: every file must share the first file's
                # library identity and temperature, or data would be served
                # silently under mismatched metadata. Compare root attributes
                # only (no dataset reads). ``mapping`` is no longer authoritative
                # (raw attrs are always stored), so it is not cross-checked.
                mismatches = []
                if library != self.library:
                    mismatches.append(
                        f"library {library!r} != {self.library!r}")
                if abs(temperature - self.temperature) > 0.1:
                    mismatches.append(
                        f"temperature {temperature} K != {self.temperature} K")
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

    def _mf10_group(self, nuclide, mt, lfs, izap=None):
        """Resolve the single MF=10 partial subgroup for ``mt``/``lfs``.

        The reaction's ``LFS*`` subgroups are matched by their ``LFS`` (and,
        when ``izap`` is given, ``IZAP``) attributes rather than by name, so a
        lumped reaction -- one whose LFS is shared by several product nuclides
        (distinct IZAP) -- is disambiguated. Raises ``KeyError`` if nothing
        matches and ``ValueError`` if a shared LFS needs ``izap`` to pick one.
        """
        rx = self._reaction(nuclide, mt)
        matches = [rx[k] for k in rx if k.startswith('LFS')
                   and int(rx[k].attrs['LFS']) == lfs
                   and (izap is None or int(rx[k].attrs['IZAP']) == izap)]
        if not matches:
            which = (f"LFS={lfs}" if izap is None
                     else f"LFS={lfs}, IZAP={izap}")
            raise KeyError(
                f"Nuclide {nuclide!r} MT={mt} has no MF=10 partial {which}.")
        if len(matches) > 1:
            izaps = sorted(int(g.attrs['IZAP']) for g in matches)
            raise ValueError(
                f"Nuclide {nuclide!r} MT={mt} LFS={lfs} is shared by IZAP "
                f"values {izaps}; pass izap= to disambiguate.")
        return matches[0]

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
        """Return the MF=10 partials available for a reaction.

        Parameters
        ----------
        nuclide : str
            GNDS name of the nuclide.
        mt : int
            Reaction MT number.

        Returns
        -------
        list of tuple of int
            One ``(lfs, izap)`` pair per stored MF=10 partial, sorted by
            ``(lfs, izap)``; empty if the reaction has no MF=10 partials. A
            lumped reaction (a shared LFS produced by several nuclides) repeats
            an ``lfs`` with different ``izap`` values.

        """
        rx = self._reaction(nuclide, mt)
        return sorted((int(rx[k].attrs['LFS']), int(rx[k].attrs['IZAP']))
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

    def pathway_xs(self, nuclide, mt, lfs, izap=None):
        """Return the MF=10 partial cross section for a reaction/final level.

        Parameters
        ----------
        nuclide : str
            GNDS name of the nuclide.
        mt : int
            Reaction MT number.
        lfs : int
            Final-level index (LFS).
        izap : int, optional
            Product IZAP (``1000*Z + A``) selecting one partial when ``lfs`` is
            shared by several product nuclides (a lumped reaction such as
            MT=5). Not needed when the LFS is unique.

        Returns
        -------
        tuple of numpy.ndarray
            Energy grid (eV) and partial cross section (barn).

        """
        group = self._mf10_group(nuclide, mt, lfs, izap)
        return group['energy'][()], group['xs'][()]

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
                            keep_extra_mts=False):
        """Preprocess a directory of PENDF files into an HDF5 library.

        Nuclide identity (Z, A, isomeric state) is always taken from the
        MF=1/MT=451 header and validated against the isomeric state implied by
        the filename (a mismatch warns; the header is trusted). MF=3 cross
        sections are stored per reaction; MF=10 isomeric production cross
        sections are stored source-faithfully as ``LFS{lfs}`` subgroups of their
        reaction (raw IZAP/LFS/QM/QI/ELFS attributes only). No product name is
        baked onto the partials: the isomer<->LFS product mapping is carried on
        the depletion chain (built with
        ``tools/add_pendf_isomeric_branching_to_chain.py``), which is where the
        collapse sources isomeric row names from.

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

        Returns
        -------
        PendfLibrary
            Reader for the file just written.

        """
        pendf_dir = Path(pendf_dir)
        out = Path(out)
        entries = _discover_pendf_files(pendf_dir)
        if not entries:
            raise ValueError(f"No PENDF files found in {pendf_dir}.")

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
                            _write_mf10_partials(mtg, ev, mt, name, path)

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
                h5.attrs['openmc_version'] = np.bytes_(openmc.__version__)

            os.replace(tmp, out)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

        return PendfLibrary(out)


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
    library : str or None
        Name of the source data library (root ``library`` attr), or ``None``.
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

    @property
    def library(self) -> str | None:
        """Source data library name (root ``library`` attr), or ``None``.

        Copied from the pointwise source at bin time (``tools/pendf_group_bin.py``)
        so a grouped library carries the same provenance identity as a
        :class:`PendfLibrary`; the chain-provenance stamp check reads it.
        """
        if 'library' in self._file.attrs:
            return _attr_str(self._file.attrs, 'library')
        return None

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

    def has_ptables(self, nuclide: str) -> bool:
        """Return whether the nuclide carries URR probability tables.

        Grouped libraries copy the pointwise ``<nuclide>/urr`` group through
        verbatim (probability tables are energy-pointwise), so this mirrors
        :meth:`openmc.data.PendfLibrary.has_ptables`. A nuclide absent from the
        library answers ``False`` gracefully rather than raising.
        """
        return nuclide in self._file and 'urr' in self._file[nuclide]

    def ptables(self, nuclide: str, temperature=None):
        """Return the URR probability tables for a nuclide, or ``None``.

        Parameters
        ----------
        nuclide : str
            GNDS name of the nuclide.
        temperature : float, optional
            Requested temperature in kelvin. A nuclide with a single stored
            temperature returns it regardless; with several the nearest
            ``<Tkey>`` is chosen. Defaults to the library temperature.

        Returns
        -------
        openmc.data.ProbabilityTables or None
            Probability tables read from ``<nuclide>/urr/<Tkey>``, or ``None``
            if the nuclide has no ``/urr`` group (including a nuclide absent
            from the library).
        """
        if nuclide not in self._file:
            return None
        group = self._file[nuclide]
        if 'urr' not in group:
            return None
        urr = group['urr']
        temp_attr = self._file.attrs.get('temperature')
        default_temp = None if temp_attr is None else float(temp_attr)
        tkey = _select_urr_tkey(list(urr.keys()), temperature, default_temp)
        if tkey is None:
            return None
        return ProbabilityTables.from_hdf5(urr[tkey])

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
