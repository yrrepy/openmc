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

For cross-validation this module also provides :class:`PendfTapeLibrary`, a
testing adapter that presents a directory of raw ASC PENDF tapes through the
same collapse-facing accessors as :class:`PendfLibrary`, reading MF=3 totals and
MF=10 partials straight from the tapes. It applies the identical structural rules
as the HDF5 build, so a collapse against the tape directory is bit-identical to a
collapse against a pointwise HDF5 built from the same tapes -- the HDF5 remains
the production format; the adapter validates the build/collapse against source.
:func:`open_pendf_library` sniffs a path (grouped ``.h5``, pointwise ``.h5``, or
ASC tape directory) and returns the matching reader.

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
from .endf import (Evaluation, get_cont_record, get_head_record,
                   get_list_record, get_tab1_record)
from .urr import ProbabilityTables

__all__ = ['PendfLibrary', 'GroupedPendfLibrary', 'PendfTapeLibrary',
           'open_pendf_library']

# Version of the PENDF HDF5 format written/read by this module.
#   1 -- MF=10 subgroups are always named ``LFS{lfs}``.
#   2 -- an LFS shared by >=2 distinct product IZAPs is named
#        ``LFS{lfs}_ZAP{izap}`` (a unique LFS keeps the bare ``LFS{lfs}``).
# The library is written source-faithful (raw IZAP/LFS/QM/QI/ELFS attrs only);
# the isomer<->LFS product mapping lives on the depletion chain, not baked here.
# Files written by older OpenMC carried baked ``product``/``mapping`` attrs; the
# reader accepts them and ignores those attrs (see :class:`PendfLibrary`).
# The MF=10-without-MF=3 totals (root ``mf10_only_totals`` census, per-reaction
# ``total_source='sum-mf10'``) are additive within version 2: the stored value is
# a valid reaction total either way, so a reader that predates the stamp consumes
# it safely and no version bump is warranted.
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


def _identity_from_evaluation(ev) -> str | None:
    """Derive a source identity from an ENDF MF=1/451 evaluation.

    Formats the NLIB library tuple as ``'<LIBRARY>-<NVER>'`` and appends the
    sublibrary description when available, e.g. ``('JEFF', 40, 0)`` +
    ``'Radioactive decay data'`` -> ``'JEFF-40 Radioactive decay data'``. Returns
    ``None`` when the evaluation carries no library information.
    """
    lib = ev.info.get('library')
    if lib is None:
        return None
    name, version, _release = lib
    identity = f"{name}-{version}"
    sublibrary = ev.info.get('sublibrary')
    if sublibrary:
        identity = f"{identity} {sublibrary}"
    return identity


def _tape_file_identity(path) -> str | None:
    """Identity of a single ENDF/PENDF tape file, or ``None`` if unreadable.

    Reads the TPID record (the first line's columns 0:66); if its stripped text
    is non-empty it is returned verbatim. A blank TPID (common on decay tapes)
    falls back to the MF=1/451-derived identity (see
    :func:`_identity_from_evaluation`).
    """
    try:
        with open(path) as fh:
            first_line = fh.readline()
    except OSError:
        return None
    tpid = first_line[:66].strip()
    if tpid:
        return tpid
    try:
        ev = Evaluation(str(path))
    except Exception:
        return None
    return _identity_from_evaluation(ev)


def tape_identity(path: PathLike) -> str | None:
    """Return a human-readable identity string for a PENDF/decay tape source.

    The identity stamps the provenance of an HDF5 PENDF library (root
    ``source_identity`` attr) and of a patched depletion chain, so a chain and a
    library built from the same tapes are recognizable as belonging together even
    when their user-supplied ``library`` labels differ.

    Parameters
    ----------
    path : str or path-like
        A single ENDF/PENDF tape file, or a directory of them.

    Returns
    -------
    str or None
        For a single tape: the TPID text (first line, columns 0:66) when
        non-empty, else the MF=1/451-derived ``'<LIBRARY>-<NVER> <sublibrary>'``
        identity. For a directory: the identity of the first of up to three
        sampled tapes (PENDF tapes discovered by :func:`_discover_pendf_files`,
        or the directory's sorted files otherwise). Divergent sampled TPIDs are
        per-material processing strings, not a library identity, so the
        MF=1/451-derived identity is used instead (with a warning); only if no
        451 identity can be derived is the first divergent string returned.
        ``None`` when no identity can be read (unreadable / non-ENDF / missing
        source).
    """
    path = Path(path)
    if not path.is_dir():
        return _tape_file_identity(path)

    entries = _discover_pendf_files(path)
    if entries:
        tapes = sorted(p for p, _liso in entries)
    else:
        tapes = sorted(p for p in path.iterdir() if p.is_file())

    identities: list[str] = []
    for tape in tapes:
        identity = _tape_file_identity(tape)
        if identity is not None:
            identities.append(identity)
        if len(identities) >= 3:
            break
    if not identities:
        return None
    first = identities[0]
    distinct = list(dict.fromkeys(identities))
    if len(distinct) > 1:
        # Divergent TPIDs are per-material processing strings (e.g. NJOY's
        # 'pendf for material 1125'), useless as a library identity. Fall back
        # to the MF=1/451-derived identity, which is per-library and stable
        # across the tape set.
        for tape in tapes:
            try:
                ev = Evaluation(str(tape))
            except Exception:
                continue
            derived = _identity_from_evaluation(ev)
            if derived is not None:
                warn(f"tape_identity: sampled tapes in {path} report "
                     f"per-material TPIDs ({', '.join(repr(i) for i in distinct)}); "
                     f"using the MF=1/451-derived identity {derived!r}.")
                return derived
        warn(f"tape_identity: sampled tapes in {path} report different "
             f"identities ({', '.join(repr(i) for i in distinct)}) and no "
             f"MF=1/451 identity could be derived; using {first!r}.")
    return first


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


def _dedupe_mf10_partials(partials, source_name, name, mt):
    """Drop true ``(IZAP, LFS)`` duplicate MF=10 partials, keeping the first.

    Shared by :func:`_write_mf10_partials` (the HDF5 build) and
    :class:`PendfTapeLibrary` (the ASC-tape collapse adapter) so both surface an
    identical set of partials for a reaction -- the single-source-of-truth
    behind the ASC-vs-h5 bit-identical collapse. ``partials`` is the raw
    ``(QM, QI, IZAP, LFS, tab)`` sequence from :func:`_iter_mf10_partials`;
    ``source_name`` names the offending tape in the skip warning. Uniqueness of
    an LFS is decided from the surviving distinct ``(IZAP, LFS)`` pairs, so a
    duplicate never makes the retained partial's LFS look shared.
    """
    seen = set()
    unique = []
    for pqm, pqi, izap, lfs, ptab in partials:
        if (izap, lfs) in seen:
            warn(f"{source_name}: duplicate MF=10 "
                 f"partial (IZAP={izap}, LFS={lfs}) in {name} "
                 f"MT={mt}; skipping.")
            continue
        seen.add((izap, lfs))
        unique.append((pqm, pqi, izap, lfs, ptab))
    return unique


# How far an LFS=0 partial's own QI may sit from its section QM (eV) before the
# ground-state invariant ELFS = QM - QI = 0 counts as violated. The gate is
# tight because a clean subsection head carries the SAME number in both fields
# -- one record, no cross-file rounding to absorb (every JEFF-4.0 MT=4 LFS=0
# stores ELFS exactly 0.0). The smallest genuine defect in the TENDL-2019 fleet
# is U235_m1 MT=4 at 76.77 eV, the U-235 first level standing in as parent, so a
# looser gate would miss a whole shape. Deliberately NOT the GENDF twin's 1 keV
# C4 tolerance: that one measures QM - QI against the DECAY library's ELIS, a
# cross-source comparison with real rounding to tolerate.
_LFS0_QI_QM_TOL_EV = 1.0


def _synthesize_mf10_total(unique, context=None):
    """Synthesize a reaction total from its MF=10 partials alone.

    Used for an MF=10 section that has **no MF=3 sibling**: each event of the
    reaction ends in exactly one final state, so the sum over the (aligned)
    MF=10 partials *is* the reaction total. ``unique`` is the deduplicated
    ``(QM, QI, IZAP, LFS, tab)`` sequence from :func:`_dedupe_mf10_partials`;
    ``context`` labels the validation warning with the tape/nuclide/MT.

    Returns ``(energy, xs, QM, QI)``: the union of every partial's energy grid,
    the pointwise sum of the partials interpolated onto it with zero fill
    outside their own range (exact -- every partial is lin-lin, see
    :func:`_check_lin_lin`), and the section-sourced Q values. QM is the
    section QM read off the LFS=0 partial when the section has one (the ground
    subsection is the section's zero reference -- the same head the patcher's
    ground-route Q and pathway-Q gate probe read) and off the first partial
    otherwise; QI is that same QM by the ground-state definition
    ELFS = QM - QI = 0. The LFS=0 partial's own QM/QI pair is VALIDATED
    (:data:`_LFS0_QI_QM_TOL_EV`, warned on violation) and its QI never
    inherited -- TENDL-2019 MT=4 writes the reaction QI there on ground targets
    and a blank QI on isomer targets, either of which would put the ground route
    on a different energy zero than the level it is the base of. **No Q value is
    ever fabricated**, there being no MF=3 HEAD to read them from.

    Shared verbatim by the HDF5 build (:meth:`PendfLibrary.from_endf_directory`)
    and :meth:`PendfTapeLibrary._load`, so a synthesized total is bit-identical
    between the two source forms.
    """
    grids = [np.asarray(ptab.x, dtype=np.float64)
             for _pqm, _pqi, _izap, _lfs, ptab in unique]
    energy = np.unique(np.concatenate(grids))
    xs = np.zeros_like(energy)
    for _pqm, _pqi, _izap, _lfs, ptab in unique:
        xs = xs + np.interp(energy, np.asarray(ptab.x, dtype=np.float64),
                            np.asarray(ptab.y, dtype=np.float64),
                            left=0.0, right=0.0)

    lfs0 = next(((float(pqm), float(pqi))
                 for pqm, pqi, _izap, lfs, _pt in unique if lfs == 0), None)
    qm = lfs0[0] if lfs0 is not None else float(unique[0][0])
    if lfs0 is not None and abs(lfs0[0] - lfs0[1]) > _LFS0_QI_QM_TOL_EV:
        warn(f"{context or 'MF=10 total synthesis'}: LFS=0 partial QI="
             f"{lfs0[1]} disagrees with section QM={qm}; using QM as the "
             f"ground-route Q, a ground-state subsection having ELFS = 0 "
             f"(TENDL-2019 MT=4 convention: reaction QI on ground targets, "
             f"blank QI on isomer targets).")
    return energy, xs, qm, qm


def _write_mf10_partials(mtg, ev, mt, name, path, unique=None):
    """Write a reaction's MF=10 isomeric production partials as ``LFS`` subgroups.

    Each MF=10 partial for reaction ``mt`` is stored as a subgroup of ``mtg``
    with its raw IZAP/LFS/QM/QI/ELFS attributes (source-faithful; no product
    name is baked -- the isomer<->LFS mapping lives on the depletion chain). A
    partial whose LFS is unique within the reaction keeps the bare ``LFS{lfs}``
    name (byte-identical to non-lumped libraries); an LFS shared by several
    distinct IZAP -- different product nuclides, as in lumped TENDL MT=5 -- is
    disambiguated as ``LFS{lfs}_ZAP{izap}``. A true ``(IZAP, LFS)`` duplicate
    warns and is skipped.

    ``unique`` optionally supplies the already parsed and deduplicated partials
    (the :func:`_dedupe_mf10_partials` output), so a caller that needed them
    first -- the MF=10-without-MF=3 branch, which synthesizes the reaction total
    from them -- neither re-parses the section nor re-emits its duplicate
    warnings. Omit it to parse and deduplicate here.
    """
    if unique is None:
        partials = list(_iter_mf10_partials(ev, mt, name))
        if not partials:
            return

        # Drop true (IZAP, LFS) duplicates, keeping the first occurrence (shared
        # with the ASC tape adapter so the two see identical partials).
        unique = _dedupe_mf10_partials(partials, path.name, name, mt)

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


def _mf9_backed_mts(ev):
    """Return ``{mt: n_subsections}`` for MF=8 sections that declare LMF=9.

    Scans each MF=8 section of ``ev`` for product subsections whose LMF pointer
    (the subsection header's ``L1`` field) is 9 -- i.e. the isomeric-production
    data for that channel lives in MF=9 (branching *multiplicities*) rather than
    MF=10 (partial cross sections). This module folds isomeric branching from
    MF=10 only, so an LMF=9 channel's partials would be silently absent.

    The MF=8 HEAD carries ``NS`` (subsection count) and ``NO``: ``NO=0`` -> each
    subsection is a LIST with inline decay data, ``NO=1`` -> each is a single
    CONT record (decay chain deferred to MT=457). Both layouts put LMF in the
    subsection header's ``L1`` field, so the header is read with
    :func:`get_list_record` (which also consumes the LIST body) or
    :func:`get_cont_record` accordingly -- the same walk-by-section idiom used to
    read MF=10 in :func:`_iter_mf10_partials`. The special MF=8 fission-yield /
    decay sections (MT 454/457/459) use a different record layout and carry no
    LMF pointer, so they are skipped. Any parse hiccup on a section is swallowed:
    this scan is advisory and must never perturb ingestion.

    Returns an empty dict when no MF=8 section declares LMF=9 (the expected
    result for producer-folded PENDF tapes, whose MF=8 carries LMF=10 only) and
    when ``ev`` has no MF=8 section at all.
    """
    mf9_mts = {}
    for (mf, mt) in ev.section:
        if mf != 8 or mt in (454, 457, 459):
            continue
        count = 0
        try:
            fo = io.StringIO(ev.section[8, mt])
            _za, _awr, _lis, _liso, ns, no = get_head_record(fo)
            for _ in range(ns):
                if no == 0:
                    (_zap, _elfs, lmf, _lfs, _npl, _n2), _vals = \
                        get_list_record(fo)
                else:
                    _zap, _elfs, lmf, _lfs, _npl, _n2 = get_cont_record(fo)
                if lmf == 9:
                    count += 1
        except Exception:
            # Advisory-only: a malformed / unexpected MF=8 section must never
            # break tape ingestion, so a parse failure just skips this section.
            continue
        if count:
            mf9_mts[mt] = count
    return mf9_mts


def _warn_if_mf9_backed(ev, name, source_name):
    """Warn once per tape if it declares MF=9-backed isomeric production.

    Our PENDF ASC tapes are producer-folded: a channel that originally stored
    isomeric branching as MF=9 multiplicities arrives with its MF=8 LMF pointer
    rewritten 9 -> 10 and a synthesized MF=10 partial cross section, which this
    reader consumes. A tape that *still* declares ``LMF=9`` in MF=8 carries MF=9
    content this MF=10-only reader does not fold, so those channels' isomeric
    partials would be silently missing. A correctly folded tape has zero LMF=9
    (and a tape with no MF=8 section trivially so), for which this is silent.

    Advisory only: it never changes what is parsed, stored, or returned -- it
    just surfaces one diagnostic per tape when the never-expected condition is
    met. Shared by the HDF5 build (:meth:`PendfLibrary.from_endf_directory`) and
    the tape-direct collapse (:meth:`PendfTapeLibrary._load`).
    """
    mf9_mts = _mf9_backed_mts(ev)
    if not mf9_mts:
        return
    detail = ', '.join(
        f"MT={mt} ({n} subsection{'s' if n != 1 else ''})"
        for mt, n in sorted(mf9_mts.items()))
    warn(f"{source_name}: {name} declares MF=9-backed isomeric production "
         f"(MF=8 LMF=9) for {detail}, which this reader does not fold; "
         f"MF=10-only reads will silently miss these channels' partials. "
         f"Expected zero for producer-folded PENDF tapes; check the tape's "
         f"processing chain.")


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
        Name of the source data library (user-supplied label).
    source_identity : str or None
        Tape-derived provenance identity of the source directory (root
        ``source_identity`` attr), or ``None`` for a file written before this
        attr was added. Used by the chain provenance-stamp check.
    mf10_only_totals : int or None
        Number of reactions stored with a total synthesized from their MF=10
        partials because they have no MF=3 section (root ``mf10_only_totals``
        attr, stamped ``total_source='sum-mf10'`` on each such reaction group).
        ``None`` -- for a file predating the feature, built with it disabled, or
        a directory in which any file lacks the attribute -- means the library
        cannot serve that class at all, which is distinct from a count of 0
        (nothing in the source tapes needed it).

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
        self.source_identity = None
        # Summed over the files of a directory library; ``None`` as soon as one
        # file lacks the attr -- that file cannot serve the MF=10-only class, so
        # neither can the library as a whole.
        self.mf10_only_totals = 0
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
            if 'mf10_only_totals' not in f.attrs:
                self.mf10_only_totals = None
            elif self.mf10_only_totals is not None:
                self.mf10_only_totals += int(f.attrs['mf10_only_totals'])
            if self.temperature is None:
                self.library = library
                self.temperature = temperature
                self.mapping = mapping
                self.source_identity = (
                    _attr_str(f.attrs, 'source_identity')
                    if 'source_identity' in f.attrs else None)
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
        """Return the MTs with a stored total cross section for a nuclide.

        Parameters
        ----------
        nuclide : str
            GNDS name of the nuclide.

        Returns
        -------
        list of int
            Sorted reaction MT numbers. These are the MF=3 reactions plus any
            reaction stored with a total synthesized from its MF=10 partials
            (stamped ``total_source='sum-mf10'``; see
            :meth:`from_endf_directory`).

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
        """Return the stored total cross section for a reaction.

        The reaction's MF=3 cross section, or -- for a reaction stored without
        an MF=3 section (group attr ``total_source='sum-mf10'``) -- the total
        the builder synthesized as the sum of its MF=10 partials.

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
                            keep_extra_mts=False, mf10_only_totals=True):
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

        A reaction whose MF=10 section has **no MF=3 sibling** is stored with
        its total synthesized from the partials (``mf10_only_totals``, on by
        default), stamped ``total_source='sum-mf10'`` on the reaction group and
        counted in the ``mf10_only_totals`` root attribute.

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
        mf10_only_totals : bool
            Store a reaction whose MF=10 section has no MF=3 sibling, using the
            union-grid sum of its MF=10 partials as the reaction total (each
            event ends in exactly one final state, so the sum *is* the total).
            The reaction group is stamped ``total_source='sum-mf10'`` and the
            root ``mf10_only_totals`` attribute records how many such reactions
            the file carries (written even when 0). MT=5 (lumped channel) and
            MT=18 (sub-actinide fission placeholders) are never stored this way
            -- they only warn, as they always did. Default is ``True``; ``False``
            restores the pre-feature build, in which every MF=10 section without
            an MF=3 sibling warns and is dropped and no root attribute is
            written.

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
        n_mf10_only_totals = 0

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
                    mf10_only_here = 0
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

                        # An MF=10 section with no MF=3 sibling carries partials
                        # the loop above cannot reach. Unless disabled, such a
                        # reaction is stored with its total SYNTHESIZED from the
                        # partials (each event ends in exactly one final state,
                        # so their sum is the total) and stamped
                        # ``total_source='sum-mf10'``. MT=5 (lumped channel,
                        # never consumed by chain/collapse) and MT=18 (JEFF-4.0
                        # sub-actinide fission placeholders, which the grouped
                        # binner's whitelist would carry into every grouped
                        # library) stay excluded and only warn.
                        mf3_mts = {mt for (mf, mt) in ev.section if mf == 3}
                        mf10_mts = {mt for (mf, mt) in ev.section if mf == 10}
                        for mt in sorted(mf10_mts - mf3_mts):
                            if not mf10_only_totals or mt in (5, 18):
                                warn(f"{path.name}: {name} MF=10 MT={mt} has no "
                                     f"MF=3 section; isomeric partials dropped.")
                                continue
                            unique = _dedupe_mf10_partials(
                                list(_iter_mf10_partials(ev, mt, name)),
                                path.name, name, mt)
                            if not unique:
                                warn(f"{path.name}: {name} MF=10 MT={mt} has no "
                                     f"MF=3 section and no usable partial; "
                                     f"nothing stored.")
                                continue
                            energy, xs, QM, QI = _synthesize_mf10_total(
                                unique, f"{path.name}: {name} MT={mt}")
                            mtg = nuc.create_group(f'MT{mt}')
                            mtg.attrs['QM'] = QM
                            mtg.attrs['QI'] = QI
                            mtg.attrs['total_source'] = np.bytes_('sum-mf10')
                            _write_xy(mtg, energy, xs)
                            _write_mf10_partials(mtg, ev, mt, name, path,
                                                 unique=unique)
                            mf10_only_here += 1

                        # Advisory guard: warn if this tape declares MF=9-backed
                        # isomeric production (MF=8 LMF=9) that this MF=10-only
                        # reader does not fold. Silent for producer-folded tapes
                        # (zero LMF=9); changes nothing written to the h5.
                        _warn_if_mf9_backed(ev, name, path.name)

                        # MF=2 MT=153 probability tables (URR). Ingested
                        # unconditionally (independent of keep_extra_mts, which
                        # only gates the MF=3 loop) so a rebuilt library always
                        # carries /urr for downstream self-shielding.
                        _write_urr_ptables(nuc, ev, name, path)

                        n_converted += 1
                        # Counted only once the nuclide is committed: a failure
                        # above deletes its group, and the census must match
                        # what the file actually carries.
                        n_mf10_only_totals += mf10_only_here
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
                # Census of the MF=10-without-MF=3 reactions stored with a
                # synthesized total. Written whenever the feature is enabled --
                # even as 0 -- so its PRESENCE marks a file that can serve that
                # class at all; a file built with ``mf10_only_totals=False``
                # omits it and is otherwise identical to a pre-feature build.
                if mf10_only_totals:
                    h5.attrs['mf10_only_totals'] = int(n_mf10_only_totals)
                # Tape-derived provenance identity (distinct from the
                # user-supplied ``library`` label): the source directory's TPID /
                # MF=1/451 identity, used by the chain provenance-stamp check.
                source_identity = tape_identity(pendf_dir)
                if source_identity is not None:
                    h5.attrs['source_identity'] = np.bytes_(source_identity)

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
    source_identity : str or None
        Tape-derived provenance identity carried from the pointwise source (root
        ``source_identity`` attr), or ``None``.
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

    @property
    def source_identity(self) -> str | None:
        """Tape-derived source identity (root ``source_identity`` attr), or None.

        Copied from the pointwise source at bin time (``tools/pendf_group_bin.py``)
        so a grouped library carries the same tape-derived provenance as a
        :class:`PendfLibrary`; the chain provenance-stamp check reads it.
        """
        if 'source_identity' in self._file.attrs:
            return _attr_str(self._file.attrs, 'source_identity')
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


class PendfTapeLibrary:
    """Raw ASC PENDF tape directory presented through the collapse interface.

    .. versionadded:: 0.15.4

    A **cross-validation / testing adapter**. It exposes a directory of raw
    ENDF-6 PENDF tapes (the ``.pendf``/``.asc`` files recognized by
    :func:`_discover_pendf_files`) through the same duck-typed, collapse-facing
    interface as :class:`PendfLibrary` -- ``nuclides``, ``reactions(nuclide)``,
    ``xs(nuclide, mt)``, ``pathways(nuclide, mt)`` and
    ``pathway_xs(nuclide, mt, lfs, izap=None)`` -- reading MF=3 totals and MF=10
    isomeric-production partials straight from the tapes instead of from a
    preprocessed HDF5 file. Collapsing a domain against this adapter yields a
    **bit-identical** :class:`~openmc.deplete.MicroXS` to collapsing against a
    pointwise :class:`PendfLibrary` built from the same tapes by
    :meth:`PendfLibrary.from_endf_directory` with default options: it applies the
    identical MF=3 extra-MT filtering, ``lin-lin`` check, MF=10 duplicate-partial
    drop and MF=10-without-MF=3 handling (a named MT gets the same synthesized
    total, MT=5/MT=18 are excluded alike), and returns the same float64 arrays
    (the source tapes are parsed the same way, so the group averages match to the
    bit).

    The pointwise HDF5 (:class:`PendfLibrary`) remains the production format;
    this adapter exists to validate the HDF5 build/collapse against the source
    tapes (and to collapse directly from tapes without a build step). Like a
    pointwise :class:`PendfLibrary`, and unlike a :class:`GroupedPendfLibrary`,
    it carries no group structure of its own, so the collapse ``energies`` must
    be supplied by the caller. URR self-shielding probability tables are **not**
    served (:meth:`has_ptables` is always ``False``); the collapse entry points
    reject ``urr_material_dilution`` on a tape adapter -- build a pointwise HDF5
    for URR work.

    Tapes are parsed **lazily**: the constructor reads each tape's MF=1/451
    header once to resolve its GNDS identity (validated against the isomeric
    state implied by the filename, exactly as
    :meth:`PendfLibrary.from_endf_directory`), but a nuclide's pointwise cross
    section arrays are parsed only on first access and cached one nuclide at a
    time -- the collapse consumes a nuclide's reactions back-to-back, so only the
    tape under collapse is held in memory (the ``_AscSource`` pattern from
    ``tools/add_pendf_isomeric_branching_to_chain.py``).

    Parameters
    ----------
    path : str or path-like
        Directory of raw ASC PENDF tapes.
    library : str, optional
        Name of the source data library recorded as ``library``. Defaults to
        ``'unknown'``; the tape-derived ``source_identity`` carries the
        provenance the chain-stamp check verifies against.
    temperature : float, optional
        Library temperature in kelvin. If given, every tape's own temperature
        must agree to within 0.1 K; if omitted, the first tape's temperature is
        adopted and enforced on the rest (mirrors
        :meth:`PendfLibrary.from_endf_directory`).
    keep_extra_mts : bool
        Retain non-activation MF=3 reactions (particle production, HEATR
        heating/damage, average secondary quantities, resonance parameters).
        The default ``False`` matches the default HDF5 build, so the two collapse
        bit-identically.
    mf10_only_totals : bool
        Serve a reaction whose MF=10 section has no MF=3 sibling with a total
        synthesized from its MF=10 partials, exactly as the HDF5 build stores it
        (:meth:`PendfLibrary.from_endf_directory`). The default ``True`` matches
        that build's default, so the two stay bit-identical; ``False`` restores
        the pre-feature contract in which such a section is dropped entirely.

    Attributes
    ----------
    nuclides : list of str
        GNDS names of the nuclides discovered in the directory.
    temperature : float or None
        Library temperature in kelvin.
    library : str
        Source data library label (user-supplied or ``'unknown'``).
    mapping : None
        Always ``None`` (source-faithful; the isomer<->LFS mapping lives on the
        chain). Present for :class:`PendfLibrary` parity.
    source_identity : str or None
        Tape-derived provenance identity of the directory
        (:func:`tape_identity`), or ``None`` if none could be read.
    mf10_only_totals : bool
        Whether MF=10-without-MF=3 reactions are served with a synthesized
        total. The tape analog of the HDF5 root ``mf10_only_totals`` census: the
        adapter synthesizes lazily, per nuclide, so it carries the *mode* rather
        than a count (``False`` is the analog of the attribute being absent from
        an h5). Each such reaction is stamped ``total_source='sum-mf10'`` in the
        in-memory reaction record, mirroring the h5 group attribute.

    """

    #: Duck marker: the collapse entry points reject URR self-shielding on a tape
    #: adapter (no probability tables are served) by testing this attribute.
    is_tape_source = True

    def __init__(self, path, library=None, temperature=None,
                 keep_extra_mts=False, mf10_only_totals=True):
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(
                f"PENDF tape directory does not exist: {self._path}")
        if not self._path.is_dir():
            raise NotADirectoryError(
                f"PENDF tape source is not a directory: {self._path}")
        entries = _discover_pendf_files(self._path)
        if not entries:
            raise ValueError(
                f"No recognizable PENDF tapes found in {self._path}.")

        self._keep_extra_mts = bool(keep_extra_mts)
        self.mf10_only_totals = bool(mf10_only_totals)
        self._tapes: dict[str, Path] = {}
        # One-nuclide (energy, xs) cache; see _load.
        self._cache_name = None
        self._cache = None

        # Eager header scan: resolve each tape's GNDS identity once (the ASC
        # source pattern). Only the name->path map and metadata are retained --
        # never the pointwise arrays, which _load parses lazily.
        lib_temperature = temperature
        for tape_path, implied_liso in entries:
            try:
                ev = Evaluation(tape_path)
                Z = ev.target['atomic_number']
                A = ev.target['mass_number']
                liso = ev.target['isomeric_state']
                name = gnds_name(Z, A, liso)
                T = ev.target['temperature']
            except Exception as exc:
                warn(f"skipping {tape_path}: {exc}")
                continue
            if implied_liso != liso:
                warn(f"{tape_path.name}: filename implies isomeric state "
                     f"{implied_liso} but MF=1/451 header gives {liso}; "
                     f"trusting header ({name}).")
            if lib_temperature is None:
                lib_temperature = T
            elif abs(T - lib_temperature) > 0.1:
                raise _TemperatureMismatchError(
                    f"{tape_path.name}: temperature {T} K disagrees with "
                    f"library temperature {lib_temperature} K by more than "
                    "0.1 K.")
            if name in self._tapes:
                warn(f"{tape_path.name}: duplicate nuclide {name}; skipping.")
                continue
            self._tapes[name] = tape_path

        if not self._tapes:
            raise ValueError(f"No PENDF tapes in {self._path} could be read.")

        self.nuclides = sorted(self._tapes)
        self.temperature = (None if lib_temperature is None
                            else float(lib_temperature))
        self.library = 'unknown' if library is None else str(library)
        self.mapping = None
        # Tape-derived provenance identity (matches what the chain-stamp check
        # and an h5 built from these tapes report).
        self.source_identity = tape_identity(self._path)

    def __repr__(self):
        return (f"<PendfTapeLibrary: {len(self.nuclides)} nuclides, "
                f"{self.library}, {self.temperature} K>")

    def _load(self, nuclide):
        """Parse one tape into the collapse-shaped reaction map (cached).

        Returns ``{mt: {'xs': (energy, xs), 'partials': {(lfs, izap): (energy,
        xs)}}}`` for the requested nuclide, cached one nuclide at a time (a
        second access to the same nuclide never re-parses; touching a different
        nuclide evicts the previous one). A reaction whose total was synthesized
        from its MF=10 partials additionally carries ``'total_source':
        'sum-mf10'``, the in-memory twin of the h5 group attribute.

        Applies the identical structural rules as
        :meth:`PendfLibrary.from_endf_directory`, in the identical
        ``sorted(ev.section)`` order: an MF=3 reaction in :data:`_EXTRA_MTS` is
        dropped unless ``keep_extra_mts``; every retained MF=3 TAB1 is lin-lin
        checked; MF=10 partials are attached to their MF=3 sibling, with true
        ``(IZAP, LFS)`` duplicates removed via :func:`_dedupe_mf10_partials`. An
        MF=10 section with no MF=3 sibling is served with the union-grid sum of
        its partials as the total (:func:`_synthesize_mf10_total`, exactly what
        the h5 build stores) unless ``mf10_only_totals`` is ``False``; MT=5 and
        MT=18 are dropped either way, as in the h5 build. Arrays are returned as
        float64, matching the ``_write_xy`` dtype the h5 round-trips, so the two
        collapse to the bit.
        """
        if self._cache_name == nuclide:
            return self._cache
        try:
            path = self._tapes[nuclide]
        except KeyError:
            raise KeyError(f"Nuclide {nuclide!r} not in library.")

        ev = Evaluation(path)
        name = gnds_name(ev.target['atomic_number'],
                         ev.target['mass_number'],
                         ev.target['isomeric_state'])
        # Advisory guard (shared with from_endf_directory): warn once per tape
        # if it declares MF=9-backed isomeric production the MF=10-only read
        # silently misses. Silent for producer-folded tapes; parses unchanged.
        _warn_if_mf9_backed(ev, name, path.name)
        reactions: dict = {}
        for (mf, mt), text in sorted(ev.section.items()):
            if mf != 3:
                continue
            if mt in _EXTRA_MTS and not self._keep_extra_mts:
                continue
            fo = io.StringIO(text)
            get_head_record(fo)                 # MF=3 HEAD (discarded)
            (_qm, _qi, _l1, _lr), tab = get_tab1_record(fo)
            _check_lin_lin(name, 3, mt, tab)
            xs = (np.asarray(tab.x, dtype=np.float64),
                  np.asarray(tab.y, dtype=np.float64))
            partials: dict = {}
            raw = list(_iter_mf10_partials(ev, mt, name))
            for _pqm, _pqi, izap, lfs, ptab in _dedupe_mf10_partials(
                    raw, path.name, name, mt):
                partials[(int(lfs), int(izap))] = (
                    np.asarray(ptab.x, dtype=np.float64),
                    np.asarray(ptab.y, dtype=np.float64))
            reactions[mt] = dict(xs=xs, partials=partials)

        # MF=10 sections with no MF=3 sibling: same rule as the h5 build --
        # MT=5/MT=18 dropped (lumped channel / fission placeholders), every other
        # MT served with the total synthesized from its own partials.
        if self.mf10_only_totals:
            mf3_mts = {mt for (mf, mt) in ev.section if mf == 3}
            mf10_mts = {mt for (mf, mt) in ev.section if mf == 10}
            for mt in sorted(mf10_mts - mf3_mts):
                if mt in (5, 18):
                    continue
                unique = _dedupe_mf10_partials(
                    list(_iter_mf10_partials(ev, mt, name)), path.name, name, mt)
                if not unique:
                    continue
                energy, xs, _qm, _qi = _synthesize_mf10_total(
                    unique, f"{path.name}: {name} MT={mt}")
                partials = {}
                for _pqm, _pqi, izap, lfs, ptab in unique:
                    partials[(int(lfs), int(izap))] = (
                        np.asarray(ptab.x, dtype=np.float64),
                        np.asarray(ptab.y, dtype=np.float64))
                reactions[mt] = dict(xs=(energy, xs), partials=partials,
                                     total_source='sum-mf10')

        self._cache_name = nuclide
        self._cache = reactions
        return reactions

    def reactions(self, nuclide):
        """Return the sorted MTs with a total cross section for a nuclide.

        MF=3 reactions plus, when ``mf10_only_totals`` is set, the reactions
        whose total is synthesized from their MF=10 partials.
        """
        return sorted(self._load(nuclide))

    def xs(self, nuclide, mt):
        """Return ``(energy, xs)`` of the total cross section (eV, barn).

        The MF=3 cross section, or the sum of the reaction's MF=10 partials for
        a reaction that has no MF=3 section (see :meth:`_load`).
        """
        rx = self._load(nuclide).get(mt)
        if rx is None:
            raise KeyError(f"Nuclide {nuclide!r} has no reaction MT={mt}.")
        return rx['xs']

    def pathways(self, nuclide, mt):
        """Return the sorted ``(lfs, izap)`` MF=10 partials of a reaction.

        Empty if the reaction has no MF=10 partials; a lumped reaction (a shared
        LFS produced by several nuclides) repeats an ``lfs`` with different
        ``izap`` values, exactly as :meth:`PendfLibrary.pathways`.
        """
        rx = self._load(nuclide).get(mt)
        if rx is None:
            raise KeyError(f"Nuclide {nuclide!r} has no MF=3 reaction MT={mt}.")
        return sorted(rx['partials'])

    def pathway_xs(self, nuclide, mt, lfs, izap=None):
        """Return ``(energy, xs)`` of one MF=10 isomeric-production partial.

        ``izap`` selects one product of a shared/lumped LFS; omit it for a
        unique LFS.
        """
        rx = self._load(nuclide).get(mt)
        if rx is None:
            raise KeyError(f"Nuclide {nuclide!r} MT={mt} has no MF=10 partials.")
        parts = rx['partials']
        if izap is not None:
            try:
                return parts[(lfs, izap)]
            except KeyError:
                raise KeyError(
                    f"Nuclide {nuclide!r} MT={mt} has no MF=10 partial "
                    f"LFS={lfs}, IZAP={izap}.")
        matches = [v for (level, _z), v in parts.items() if level == lfs]
        if not matches:
            raise KeyError(
                f"Nuclide {nuclide!r} MT={mt} has no MF=10 partial LFS={lfs}.")
        if len(matches) > 1:
            izaps = sorted(z for (level, z) in parts if level == lfs)
            raise ValueError(
                f"Nuclide {nuclide!r} MT={mt} LFS={lfs} is shared by IZAP "
                f"values {izaps}; pass izap= to disambiguate.")
        return matches[0]

    def has_ptables(self, nuclide):
        """Return ``False`` -- the tape adapter serves no URR probability tables.

        URR self-shielding is a production feature of the pointwise
        :class:`PendfLibrary`; the tape adapter targets the deterministic
        collapse cross-validation, and the collapse entry points reject
        ``urr_material_dilution`` on it (see
        :meth:`openmc.deplete.MicroXS.from_multigroup_flux`).
        """
        return False

    def ptables(self, nuclide, temperature=None):
        """Return ``None`` -- no URR probability tables served (see
        :meth:`has_ptables`)."""
        return None

    def close(self):
        """Release the one-nuclide cache (tapes hold no persistent handle)."""
        self._cache_name = None
        self._cache = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def open_pendf_library(path: PathLike, library=None):
    """Open a PENDF source path as the reader the depletion collapse expects.

    Sniffs ``path`` and returns the matching duck-typed reader:

    * a single ``.h5`` file -> :class:`GroupedPendfLibrary` when its root
      ``format`` attribute is ``'pendf-grouped'``, else :class:`PendfLibrary`;
    * a directory containing ``.h5`` files -> :class:`PendfLibrary`
      (directory mode);
    * a directory of raw ASC PENDF tapes -> :class:`PendfTapeLibrary`.

    Only a grouped ``.h5`` carries its own group structure; the pointwise
    :class:`PendfLibrary` and the :class:`PendfTapeLibrary` both require the
    collapse ``energies`` to be supplied by the caller. This is the collapse-side
    counterpart of ``open_pendf_source`` in
    ``tools/add_pendf_isomeric_branching_to_chain.py`` (which maps the same
    source kinds to the chain patcher's adapters).

    Parameters
    ----------
    path : str or path-like
        A ``.h5`` PENDF library file, a directory of such files, or a directory
        of raw ASC PENDF tapes.
    library : str, optional
        Source library label, forwarded to :class:`PendfTapeLibrary` for the ASC
        directory case (ignored for the HDF5 readers, which read it from file).

    Returns
    -------
    PendfLibrary or GroupedPendfLibrary or PendfTapeLibrary
        A reader exposing the duck-typed collapse interface.

    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PENDF source path does not exist: {path}")
    if path.is_file():
        try:
            with h5py.File(path, 'r') as f:
                fmt = _decode(f.attrs.get('format'))
        except OSError as exc:
            raise ValueError(
                f"{path} is not an HDF5 PENDF library; a raw PENDF tape source "
                f"must be a directory of ASC tapes, not a single file") from exc
        if fmt == GROUPED_FORMAT:
            return GroupedPendfLibrary(path)
        return PendfLibrary(path)
    # Directory: prefer a preprocessed .h5 library (pointwise directory mode)
    # over raw ASC tapes, mirroring open_pendf_source's precedence.
    if any(path.glob('*.h5')):
        return PendfLibrary(path)
    if not _discover_pendf_files(path):
        raise ValueError(
            f"PENDF source directory {path} contains no .h5 libraries and no "
            f"recognizable ASC PENDF tapes")
    return PendfTapeLibrary(path, library=library)
