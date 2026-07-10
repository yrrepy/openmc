"""
PENDF Isomeric Branching Chain Patcher (v1)

Adds isomeric branching pathways derived from PENDF MF=10 isomeric production
cross sections to an OpenMC depletion chain. For every reaction whose MF=10
data carries a metastable final level, a product-qualified pathway
(``(n,gamma)_m1`` -> ``In116_m1``) is added alongside the ground pathway. The
resulting groups are serialized by the chain's Phase-1 refold writer as the
canonical type-only ``<reaction><isomeric_branching .../></reaction>`` form.

Two mapping modes are supported:
- 'elis' (default): ELIS-based mapping for accurate LFS->LISO conversion,
  matching each partial's excitation energy (ELFS = QM - QI) against the
  decay-library excitation energies (:mod:`openmc.deplete.decay_elis`).
- 'lfs_order': FISPACT-like positional mapping for validation testing.

IMPORTANT: a decay file is REQUIRED for both modes.

This is the PENDF cousin of ``add_gendf_isomeric_branching_to_chain.py``; the
console output, statistics block, and mapping-log format mirror that tool with
the GENDF-* columns renamed PENDF-*.
"""

import argparse
import re
import sys
from collections import defaultdict, Counter
from itertools import combinations
from pathlib import Path

import numpy as np

import openmc.data
from openmc.data import gnds_name, zam, ATOMIC_SYMBOL, DADZ
from openmc.deplete import Chain
from openmc.deplete.nuclide import ReactionTuple
from openmc.deplete.chain import REACTIONS
from openmc.deplete.decay_elis import (
    parse_decay_isomeric_levels, lookup_liso, ELIS_RTOL, ELIS_ATOL,
)
from openmc.deplete.microxs import (
    _partials_total_max_deviation, CONSISTENCY_ABS_FLOOR, CONSISTENCY_RTOL,
)

# ``numpy.trapz`` was renamed to ``numpy.trapezoid`` in NumPy 2.0; fall back so
# the audit's integral ratio works on either.
_TRAPEZOID = getattr(np, 'trapezoid', getattr(np, 'trapz', None))


# =============================================================================
# Library configurations for CLI
# =============================================================================

_BYPERRY = '/home/perry/Projects/OMC_Development/PENDF/data/byPerry_v0/'
_DEPLETION = '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/'

LIBRARY_CONFIGS = {
    'jeff40': {
        'description':   'JEFF-4.0 (IST) - PENDF MF=10 isomeric branching',
        'base_chain':    _DEPLETION + 'JEFF40/Chain_JEFF40.xml',
        'pendf':         _BYPERRY + 'JEFF40-IST.elis_mapped.293K.PENDF.h5',
        'decay_file':    _DEPLETION + 'JEFF40/jeff-4.0-endf/decay/Radioactive_Decay_Data_JEFF-40.txt',
        'output_dir':    _BYPERRY,
        'output_prefix': 'Chain_JEFF40-IST',
        'log_prefix':    'JEFF40_pendf_isomer_mapping',
    },
    'endfb81': {
        'description':   'ENDF/B-8.1 (IST) - PENDF MF=10 isomeric branching',
        'base_chain':    _DEPLETION + 'ENDFB81/Chain_ENDFB81.xml',
        'pendf':         _BYPERRY + 'ENDFB81-IST.elis_mapped.293K.PENDF.h5',
        'decay_file':    _DEPLETION + 'ENDFB81/decay',
        'output_dir':    _BYPERRY,
        'output_prefix': 'Chain_ENDFB81-IST',
        'log_prefix':    'ENDFB81_pendf_isomer_mapping',
    },
    'tendl2019': {
        'description':   'TENDL-2019 (IST, decay2020) - PENDF MF=10 isomeric branching',
        'base_chain':    _DEPLETION + 'TENDL2019/Chain_TENDL2019.xml',
        'pendf':         _BYPERRY + 'TENDL2019-IST_decay2020.elis_mapped.293K.PENDF.h5',
        'decay_file':    _DEPLETION + 'TENDL2019/tendl-2019-endf/decay/decay_2020',
        'output_dir':    _BYPERRY,
        'output_prefix': 'Chain_TENDL2019-IST_decay2020',
        'log_prefix':    'TENDL2019_pendf_isomer_mapping',
    },
}


def build_parser():
    """Build argument parser for CLI."""
    class CustomFormatter(argparse.RawDescriptionHelpFormatter):
        pass

    lib_help_lines = ["Available library presets:"]
    for key, config in LIBRARY_CONFIGS.items():
        lib_help_lines.append(f"  {key:<12} {config['description']}")
    epilog = "\n".join(lib_help_lines)

    parser = argparse.ArgumentParser(description='PENDF Isomeric Branching Chain Patcher v1 -- adds MF=10 isomeric branching from a PENDF library to an OpenMC chain.', epilog=epilog, formatter_class=CustomFormatter)
    parser.add_argument('-l', '--library',      choices=list(LIBRARY_CONFIGS), metavar='LIB', default=None,     help='Library preset (see list below); provides defaults for the paths')
    parser.add_argument('--base-chain',         type=Path,                                    default=None,     help='Base OpenMC chain XML (overrides preset)')
    parser.add_argument('--pendf',              type=Path,                                    default=None,     help='PENDF source: an .h5 library file OR a directory of ASC .pendf/.asc tapes (overrides preset)')
    parser.add_argument('--decay-file',         type=Path,                                    default=None,     help='ENDF decay library file or directory (overrides preset)')
    parser.add_argument('--output-chain',       type=Path,                                    default=None,     help='Output chain XML (overrides preset-derived name)')
    parser.add_argument('--log-file',           type=Path,                                    default=None,     help='Isomer mapping log file (overrides preset-derived name)')
    parser.add_argument('-m', '--map',          choices=['elis', 'lfs_order'],                default='elis',   help="Mapping mode: 'elis' (production, default) or 'lfs_order' (FISPACT validation)")
    parser.add_argument('-r', '--rtol',         type=float,                                   default=0.50,     help='Relative tolerance for ELIS matching (default: 0.50 = 50%%)')
    parser.add_argument('-a', '--atol',         type=float,                                   default=0.0,      help='Absolute tolerance for ELIS matching in eV (default: 0.0)')
    parser.add_argument('--mf10-reject-rtol',   type=float,                                   default=None,     help='Leave a reaction stock (no isomeric branching) when its MF=10-vs-MF=3 audit max rel dev exceeds X (default: None = audit only, reject nothing)')
    parser.add_argument('-v', '--verbose',      action='store_true',                          default=True,     help='Enable verbose output (default: True)')
    parser.add_argument('-q', '--quiet',        action='store_true',                          default=False,    help='Disable verbose output')
    return parser


# =============================================================================
# Reaction / MT helpers
# =============================================================================

# Reverse MT -> canonical chain reaction name, built once (first name wins).
_MT_TO_NAME = {}
for _name, _info in REACTIONS.items():
    for _mt in _info.mts:
        _MT_TO_NAME.setdefault(_mt, _name)

_ISOMER_SUFFIX = re.compile(r'_m\d+$')


def _ground_product(z, a, r_name):
    """Return the DADZ ground-state product GNDS name for reaction ``r_name``.

    Returns ``None`` when the (z, a) shift pushes the product below Z=1.
    """
    delta_a, delta_z = DADZ[r_name]
    zp = z + delta_z
    if zp not in ATOMIC_SYMBOL:
        return None
    return gnds_name(zp, a + delta_a, 0)


# =============================================================================
# PENDF source adapters (h5 and ASC), one interface
# =============================================================================

class _H5Source:
    """PENDF HDF5 library backend built on :class:`openmc.data.PendfLibrary`.

    Public metadata (``nuclides``, ``library``, ``mapping``) comes from the
    reader; per-partial ``QI``/``QM``/``ELFS`` attributes are read directly off
    the HDF5 groups (they have no public accessor -- documented in
    ``openmc/data/pendf.py``).
    """

    kind = 'h5'

    def __init__(self, path):
        self._lib = openmc.data.PendfLibrary(path)
        self.nuclides = list(self._lib.nuclides)
        self.library = self._lib.library or 'unknown'
        self.mapping = self._lib.mapping

    def reactions(self, nuclide):
        grp = self._lib._groups[nuclide]
        out = {}
        for mtk in grp:
            if not mtk.startswith('MT'):
                continue
            mt = int(mtk[2:])
            mtg = grp[mtk]
            partials = []
            for sk in mtg:
                if not sk.startswith('LFS'):
                    continue
                attrs = mtg[sk].attrs
                partials.append(dict(
                    lfs=int(attrs['LFS']), izap=int(attrs['IZAP']),
                    qi=float(attrs['QI']), qm=float(attrs['QM']),
                    elfs=float(attrs['ELFS'])))
            out[mt] = dict(qm=float(mtg.attrs['QM']), qi=float(mtg.attrs['QI']),
                           partials=partials)
        return out

    def total_xs(self, nuclide, mt):
        """(energy, xs) of the MF=3 total cross section (barn vs eV)."""
        return self._lib.xs(nuclide, mt)

    def pathway_xs(self, nuclide, mt, lfs, izap=None):
        """(energy, xs) of one MF=10 isomeric-production partial."""
        return self._lib.pathway_xs(nuclide, mt, lfs, izap)

    def close(self):
        self._lib.close()


class _AscSource:
    """ASC PENDF tape backend (directory of ``.pendf``/``.asc`` tapes).

    Reuses the MF=8/10 metadata extraction primitives from
    ``openmc.data.pendf`` (``_discover_pendf_files``, ``_iter_mf10_partials``)
    rather than duplicating the ENDF-6 record parsing. No HDF5 is written.
    """

    kind = 'asc'

    def __init__(self, path, library=None):
        from openmc.data.pendf import _discover_pendf_files, _iter_mf10_partials
        from openmc.data.endf import (Evaluation, get_head_record,
                                       get_tab1_record)
        import io

        self.library = library or 'unknown'
        self.mapping = None
        self._data = {}
        self._tapes = {}                 # GNDS name -> tape path
        self._xs_cache_name = None       # single-nuclide (energy, xs) cache
        self._xs_cache = None
        for tape, _implied in _discover_pendf_files(Path(path)):
            try:
                ev = Evaluation(tape)
                z = ev.target['atomic_number']
                a = ev.target['mass_number']
                liso = ev.target['isomeric_state']
                name = gnds_name(z, a, liso)
            except Exception as exc:
                print(f"  WARNING: skipping {tape}: {exc}", file=sys.stderr)
                continue
            reactions = {}
            mf10_mts = {mt for (mf, mt) in ev.section if mf == 10}
            for mt in sorted(mf10_mts):
                if (3, mt) in ev.section:
                    fo = io.StringIO(ev.section[3, mt])
                    get_head_record(fo)
                    (qm, qi, _l1, _lr), _tab = get_tab1_record(fo)
                else:
                    qm = qi = 0.0
                partials = []
                for pqm, pqi, izap, lfs, _ptab in _iter_mf10_partials(
                        ev, mt, name):
                    partials.append(dict(
                        lfs=int(lfs), izap=int(izap), qi=float(pqi),
                        qm=float(pqm), elfs=float(pqm - pqi)))
                reactions[mt] = dict(qm=float(qm), qi=float(qi),
                                     partials=partials)
            self._data[name] = reactions
            self._tapes[name] = tape
        self.nuclides = sorted(self._data)

    def reactions(self, nuclide):
        return self._data[nuclide]

    def _load_xs(self, nuclide):
        """Parse one tape's MF=3 totals + MF=10 partial (energy, xs) arrays.

        Cached one nuclide at a time: ``map_library`` audits a parent's
        reactions back-to-back before moving on, so only the tape currently
        under audit is held in memory -- never the whole library.
        """
        if self._xs_cache_name == nuclide:
            return self._xs_cache
        from openmc.data.pendf import _iter_mf10_partials
        from openmc.data.endf import (Evaluation, get_head_record,
                                       get_tab1_record)
        import io

        ev = Evaluation(self._tapes[nuclide])
        z = ev.target['atomic_number']
        a = ev.target['mass_number']
        liso = ev.target['isomeric_state']
        name = gnds_name(z, a, liso)
        cache = {}
        mf10_mts = {mt for (mf, mt) in ev.section if mf == 10}
        for mt in mf10_mts:
            total = None
            if (3, mt) in ev.section:
                fo = io.StringIO(ev.section[3, mt])
                get_head_record(fo)
                (_qm, _qi, _l1, _lr), tab = get_tab1_record(fo)
                total = (np.asarray(tab.x, dtype=float),
                         np.asarray(tab.y, dtype=float))
            partials = {}
            for pqm, pqi, izap, lfs, ptab in _iter_mf10_partials(ev, mt, name):
                partials[(int(lfs), int(izap))] = (
                    np.asarray(ptab.x, dtype=float),
                    np.asarray(ptab.y, dtype=float))
            cache[mt] = dict(total=total, partials=partials)
        self._xs_cache_name = nuclide
        self._xs_cache = cache
        return cache

    def total_xs(self, nuclide, mt):
        """(energy, xs) of the MF=3 total cross section (barn vs eV)."""
        rx = self._load_xs(nuclide).get(mt)
        if rx is None or rx['total'] is None:
            raise KeyError(f"{nuclide!r} MT={mt} has no MF=3 total.")
        return rx['total']

    def pathway_xs(self, nuclide, mt, lfs, izap=None):
        """(energy, xs) of one MF=10 isomeric-production partial."""
        rx = self._load_xs(nuclide).get(mt)
        if rx is None:
            raise KeyError(f"{nuclide!r} MT={mt} has no MF=10 partials.")
        parts = rx['partials']
        if izap is not None:
            return parts[(lfs, izap)]
        matches = [v for (l, _z), v in parts.items() if l == lfs]
        if not matches:
            raise KeyError(f"{nuclide!r} MT={mt} has no MF=10 partial LFS={lfs}.")
        if len(matches) > 1:
            raise ValueError(f"{nuclide!r} MT={mt} LFS={lfs} shared by several "
                             f"IZAP; pass izap= to disambiguate.")
        return matches[0]

    def close(self):
        pass


def open_pendf_source(path, library=None):
    """Return the PENDF source adapter for ``path`` (h5 file or ASC directory)."""
    path = Path(path)
    if path.is_file() and path.suffix == '.h5':
        return _H5Source(path)
    if path.is_dir() and any(path.glob('*.h5')):
        return _H5Source(path)
    if path.is_dir():
        return _AscSource(path, library=library)
    if path.is_file():
        return _H5Source(path)
    raise FileNotFoundError(str(path))


# =============================================================================
# Pointwise MF=10-vs-MF=3 consistency audit
# =============================================================================

def _audit_reaction(source, parent, mt, partials):
    """Pointwise consistency of a reaction's MF=10 partials against its MF=3 total.

    Every MF=10 partial (ground + metastable) is interpolated lin-lin onto the
    MF=3 energy grid (``np.interp(..., left=0, right=0)`` -- PENDF is lin-lin;
    outside a partial's tabulated range it contributes nothing) and summed. The
    summed partials are compared to the MF=3 total with the same
    :func:`_partials_total_max_deviation` used by the collapse, so the audit and
    the runtime warning share one definition of "consistent" (including the
    ``CONSISTENCY_ABS_FLOOR`` both-sides floor-dust exemption).

    Returns ``None`` when the MF=3 total is unavailable (nothing to compare
    against), else a dict with ``worst_dev`` (max relative deviation),
    ``energy`` (eV at that group; ``None`` when only floor dust qualified),
    ``sum_partials``/``total`` (barn there), and ``integral_ratio``
    (``int Sum(partials) / int total`` over the MF=3 grid).
    """
    try:
        mf3_e, mf3_xs = source.total_xs(parent, mt)
    except Exception:
        return None
    mf3_e = np.asarray(mf3_e, dtype=float)
    mf3_xs = np.asarray(mf3_xs, dtype=float)
    if mf3_e.size == 0:
        return None

    part_sum = np.zeros_like(mf3_xs)
    for p in partials:
        try:
            pe, pxs = source.pathway_xs(parent, mt, p['lfs'], p['izap'])
        except Exception:
            continue
        part_sum = part_sum + np.interp(mf3_e, np.asarray(pe, dtype=float),
                                        np.asarray(pxs, dtype=float),
                                        left=0.0, right=0.0)

    worst, idx = _partials_total_max_deviation(mf3_xs, part_sum)

    int_total = float(_TRAPEZOID(mf3_xs, mf3_e))
    int_part = float(_TRAPEZOID(part_sum, mf3_e))
    ratio = (int_part / int_total) if int_total != 0.0 else float('inf')

    return dict(
        worst_dev=worst,
        energy=(float(mf3_e[idx]) if idx >= 0 else None),
        sum_partials=(float(part_sum[idx]) if idx >= 0 else None),
        total=(float(mf3_xs[idx]) if idx >= 0 else None),
        integral_ratio=ratio)


# =============================================================================
# Mapping core -- classify each MF=10 metastable partial of a reaction
# =============================================================================

def _classify_metastables(parent, mt, r_name, metastables, decay_lookup,
                          chain_names, mode, rtol, atol):
    """Classify each metastable (LFS>0) MF=10 partial of one reaction.

    Returns a list of records (one per metastable partial) with a ``bucket``
    key drawn from: ``matched`` (mapped and product in chain),
    ``product_not_in_chain`` (mapped but target absent from chain),
    ``rtol_exceeded``, ``no_dk`` (no product/metastable in decay library),
    ``zero_elis`` (decay metastable carries ELIS=0), ``duplicate`` (a closer
    LFS mapped to the same LISO), and ``lfs_order_dropped`` (lfs_order mode).
    """
    records = []
    if mode == 'lfs_order':
        return _classify_lfs_order(parent, mt, r_name, metastables,
                                   decay_lookup, chain_names, rtol, atol)

    # elis mode: first pass -- resolve each metastable's LISO by ELFS
    results = []
    for p in metastables:
        z, a = p['izap'] // 1000, p['izap'] % 1000
        res = lookup_liso(z, a, p['elfs'], decay_lookup, rtol=rtol, atol=atol)
        results.append((p, z, a, res))

    # Detect several LFS matching the same LISO; keep closest in ELFS.
    liso_to_matches = defaultdict(list)
    for idx, (p, _z, _a, res) in enumerate(results):
        if res['status'] == 'matched':
            diff = abs(p['elfs'] - res['dk_elis'])
            liso_to_matches[res['liso']].append((diff, idx, p))
    discarded = {}
    for liso, matches in liso_to_matches.items():
        if len(matches) > 1:
            matches.sort(key=lambda m: m[0])
            keeper = matches[0][2]
            for _d, idx, _p in matches[1:]:
                discarded[idx] = dict(liso=liso, kept_lfs=keeper['lfs'])

    for idx, (p, z, a, res) in enumerate(results):
        rec = dict(lfs=p['lfs'], izap=p['izap'], elfs=p['elfs'], qi=p['qi'],
                   z=z, a=a, parent=parent, mt=mt, reaction=r_name)
        status = res['status']
        if idx in discarded:
            rec.update(bucket='duplicate', liso=discarded[idx]['liso'],
                       kept_lfs=discarded[idx]['kept_lfs'],
                       dk_elis=res.get('dk_elis'))
        elif status == 'matched':
            liso = res['liso']
            product = gnds_name(z, a, liso)
            rec.update(liso=liso, product=product, dk_elis=res['dk_elis'])
            rec['bucket'] = ('matched' if product in chain_names
                             else 'product_not_in_chain')
        elif status == 'nearest':
            rec.update(bucket='rtol_exceeded', liso=res['liso'],
                       dk_elis=res['dk_elis'], diff_pct=res.get('diff_pct'))
        elif status == 'zero_elis_only':
            rec.update(bucket='zero_elis',
                       skipped_states=res.get('skipped_states', []))
        else:  # no_decay_data / no_metastables / no_match
            rec.update(bucket='no_dk', status=status)
        records.append(rec)
    return records


def _classify_lfs_order(parent, mt, r_name, metastables, decay_lookup,
                        chain_names, rtol, atol):
    """Positional (FISPACT-like) mapping of metastable partials."""
    records = []
    ordered = sorted(metastables, key=lambda p: p['lfs'])
    if not ordered:
        return records
    z, a = ordered[0]['izap'] // 1000, ordered[0]['izap'] % 1000
    dk_meta = sum(1 for s in decay_lookup.get((z, a), []) if s.liso > 0)
    for position, p in enumerate(ordered, start=1):
        zp, ap = p['izap'] // 1000, p['izap'] % 1000
        rec = dict(lfs=p['lfs'], izap=p['izap'], elfs=p['elfs'], qi=p['qi'],
                   z=zp, a=ap, parent=parent, mt=mt, reaction=r_name)
        res = lookup_liso(zp, ap, p['elfs'], decay_lookup, rtol=rtol, atol=atol)
        rec['dk_elis'] = res.get('dk_elis')
        if position > dk_meta:
            rec.update(bucket='lfs_order_dropped', liso=position)
        else:
            product = gnds_name(zp, ap, position)
            rec.update(liso=position, product=product)
            rec['bucket'] = ('matched' if product in chain_names
                             else 'product_not_in_chain')
        records.append(rec)
    return records


def map_library(source, chain, decay_lookup, mode, rtol, atol, verbose=True,
                reject_rtol=None):
    """Map every PENDF nuclide's MF=10 metastable partials against the chain.

    Returns ``(branching, stats)`` where ``branching`` is
    ``{parent: {r_name: {'mt', 'ground', 'metastables'}}}`` restricted to the
    matched-and-in-chain pathways, and ``stats`` carries the counters and the
    per-parent log records.

    Every reaction carrying >=1 metastable pathway is run through the pointwise
    MF=10-vs-MF=3 consistency audit (:func:`_audit_reaction`) regardless of
    ``reject_rtol``. When ``reject_rtol`` is not ``None``, a reaction whose audit
    max relative deviation exceeds it is left stock -- no ``<isomeric_branching>``
    is created -- and recorded in ``stats['rejected']``.
    """
    chain_names = set(chain.nuclide_dict)
    branching = defaultdict(dict)

    isomer_mappings = []          # matched-in-chain rows (per-parent tables)
    elis_errors = []              # rtol_exceeded / no_dk / zero_elis
    products_not_in_chain = []    # matched but target absent from chain
    duplicate_errors = []         # kept-closest, others discarded
    lfs_order_dropped = []

    audit_offenders = []          # reactions with worst_dev > CONSISTENCY_RTOL
    audit_clean = 0               # auditable reactions within CONSISTENCY_RTOL
    rejected = []                 # audit-rejected (left stock) when flag set
    absent_status = {}            # base GNDS name -> lookup_liso status (no_dk)

    nuclides_with_branching = set()
    total_lfs = 0
    counts = Counter()
    ground_only = 0

    for parent in source.nuclides:
        try:
            reactions = source.reactions(parent)
        except Exception as exc:
            if verbose:
                print(f"  WARNING: {parent}: {exc}", file=sys.stderr)
            continue
        z, a, _ = zam(parent)
        parent_in_chain = parent in chain_names

        for mt, rxinfo in sorted(reactions.items()):
            r_name = _MT_TO_NAME.get(mt)
            if r_name is None:
                continue  # MT not a depletion reaction (e.g. MT=5 lumped)
            partials = rxinfo['partials']
            if not partials:
                continue
            metastables = [p for p in partials if p['lfs'] != 0]
            if not metastables:
                ground_only += 1
                continue

            total_lfs += len(metastables)

            # Pointwise MF=10-vs-MF=3 consistency audit. Always runs (the
            # decoration candidates are exactly the reactions with >=1
            # metastable pathway); rejection is a separate, opt-in gate below.
            audit = _audit_reaction(source, parent, mt, partials)
            reject_this = False
            if audit is not None:
                if audit['worst_dev'] > CONSISTENCY_RTOL:
                    audit_offenders.append(dict(
                        parent=parent, mt=mt, reaction=r_name, **audit))
                else:
                    audit_clean += 1
                if reject_rtol is not None and audit['worst_dev'] > reject_rtol:
                    reject_this = True
                    rejected.append(dict(
                        parent=parent, mt=mt, reaction=r_name,
                        threshold=reject_rtol, **audit))

            records = _classify_metastables(
                parent, mt, r_name, metastables, decay_lookup,
                chain_names, mode, rtol, atol)

            mapped = []           # matched + in chain
            dup_by_liso = defaultdict(list)
            for rec in records:
                bucket = rec['bucket']
                counts[bucket] += 1
                if bucket == 'matched':
                    nuclides_with_branching.add(parent)
                    hl = _half_life(chain, rec['product'])
                    mrow = dict(parent=parent, reaction=r_name, mt=mt,
                                product=rec['product'], lfs=rec['lfs'],
                                liso=rec['liso'], elis=rec['elfs'],
                                dk_elis=rec['dk_elis'],
                                method=('lfs_order' if mode == 'lfs_order'
                                        else 'elis'),
                                half_life=hl if hl is not None else 'stable',
                                target_z=rec['z'], target_a=rec['a'])
                    isomer_mappings.append(mrow)
                    mapped.append(rec)
                elif bucket == 'product_not_in_chain':
                    nuclides_with_branching.add(parent)
                    products_not_in_chain.append(dict(
                        type='no_metastable_decay_data', parent=parent, mt=mt,
                        reaction=r_name, lfs=rec['lfs'], elis=rec['elfs'],
                        base_nuclide=gnds_name(rec['z'], rec['a'], 0),
                        target_z=rec['z'], target_a=rec['a'],
                        product=rec['product'], liso=rec.get('liso'),
                        note='Product not in chain'))
                elif bucket == 'rtol_exceeded':
                    nuclides_with_branching.add(parent)
                    elis_errors.append(dict(
                        type='elis_tol_exceeded', parent=parent, mt=mt,
                        reaction=r_name, lfs=rec['lfs'], elis=rec['elfs'],
                        dk_elis=rec['dk_elis'], liso=rec.get('liso'),
                        diff_percent=rec.get('diff_pct', 0.0),
                        base_nuclide=gnds_name(rec['z'], rec['a'], 0),
                        target_z=rec['z'], target_a=rec['a'], omitted=True))
                elif bucket == 'zero_elis':
                    nuclides_with_branching.add(parent)
                    elis_errors.append(dict(
                        type='zero_elis_metastables', parent=parent, mt=mt,
                        reaction=r_name, lfs=rec['lfs'], elis=rec['elfs'],
                        base_nuclide=gnds_name(rec['z'], rec['a'], 0),
                        target_z=rec['z'], target_a=rec['a'],
                        skipped_states=rec.get('skipped_states', [])))
                elif bucket == 'no_dk':
                    nuclides_with_branching.add(parent)
                    base_nuc = gnds_name(rec['z'], rec['a'], 0)
                    # Status (no_decay_data / no_metastables / no_match) is a
                    # property of (Z, A) + decay library, so one base name maps
                    # to one status regardless of which reaction surfaced it.
                    absent_status.setdefault(base_nuc,
                                             rec.get('status', 'no_decay_data'))
                    elis_errors.append(dict(
                        type='no_metastable_decay_data', parent=parent, mt=mt,
                        reaction=r_name, lfs=rec['lfs'], elis=rec['elfs'],
                        base_nuclide=base_nuc,
                        target_z=rec['z'], target_a=rec['a']))
                elif bucket == 'duplicate':
                    dup_by_liso[rec['liso']].append(rec)
                elif bucket == 'lfs_order_dropped':
                    lfs_order_dropped.append(dict(
                        parent=parent, mt=mt, reaction=r_name, lfs=rec['lfs'],
                        would_be_liso=rec['liso'], gendf_elis=rec['elfs'],
                        dk_meta_count=None))

            for liso, dups in dup_by_liso.items():
                base_nuc = gnds_name(dups[0]['z'], dups[0]['a'], 0)
                duplicate_errors.append(dict(
                    nuclide=parent, reaction=r_name, mt=mt, liso=liso,
                    base_nuclide=base_nuc, kept_lfs=dups[0].get('kept_lfs'),
                    dk_elis=dups[0].get('dk_elis'),
                    target_z=dups[0]['z'], target_a=dups[0]['a'],
                    discarded=[dict(lfs=d['lfs'], elis=d['elfs'],
                                    diff=abs(d['elfs'] - (d.get('dk_elis') or 0.0)))
                               for d in dups]))

            # Record the branching for decoration (parent must be in chain).
            # A reaction the audit rejected (reject_this) is skipped here: no
            # <isomeric_branching> is created, the base reaction stays stock, and
            # the collapse's MF=3 fallback routes the total to the ground target.
            if mapped and parent_in_chain and not reject_this:
                ground = next((p for p in partials if p['lfs'] == 0), None)
                branching[parent][r_name] = dict(
                    mt=mt, ground=ground, qm=rxinfo['qm'], qi=rxinfo['qi'],
                    metastables=mapped)
            elif not mapped:
                # MF=10 metastables present but none mapped into the chain:
                # the base reaction is left stock.
                ground_only += 1

    # Unique base names absent from the decay library, grouped by lookup_liso
    # status (no_decay_data / no_metastables / no_match).
    absent_by_status = defaultdict(list)
    for base, status in absent_status.items():
        absent_by_status[status].append(base)
    absent_by_status = {status: sorted(names)
                        for status, names in absent_by_status.items()}

    stats = dict(
        pendf_nuclides_total=len(source.nuclides),
        nuclides_with_branching=len(nuclides_with_branching),
        total_lfs=total_lfs,
        matched=counts['matched'],
        products_not_in_chain=counts['product_not_in_chain'],
        rtol_exceeded=counts['rtol_exceeded'],
        no_dk=counts['no_dk'],
        zero_elis=counts['zero_elis'],
        duplicate_discarded=counts['duplicate'],
        duplicate_resolved=len(duplicate_errors),
        lfs_order_dropped=counts['lfs_order_dropped'],
        ground_only=ground_only,
        isomer_mappings=isomer_mappings,
        elis_errors=elis_errors,
        products_not_in_chain_errors=products_not_in_chain,
        duplicate_errors=duplicate_errors,
        lfs_order_dropped_list=lfs_order_dropped,
        # MF=10 consistency audit + threshold-gated rejection + decay-gap log.
        reject_rtol=reject_rtol,
        audit_offenders=len(audit_offenders),
        audit_clean=audit_clean,
        audit_offenders_list=audit_offenders,
        rejected_count=len(rejected),
        rejected=rejected,
        absent_by_status=absent_by_status,
        absent_unique_count=len(absent_status),
    )
    return branching, stats


def _half_life(chain, name):
    if name in chain.nuclide_dict:
        return chain[name].half_life
    return None


# =============================================================================
# Chain decoration -- work at the Chain-object level (never raw XML)
# =============================================================================

def decorate_chain(chain, branching):
    """Add ground + qualified metastable ReactionTuples to the chain in place.

    Every tuple carries a ``pendf_lfs`` (ground = 0, metastable = tape LFS) so
    the Phase-1 refold writer folds the group into an ``<isomeric_branching>``
    element instead of falling back to stock reaction elements. Returns the
    number of reactions synthesized for MTs absent from the base chain.
    """
    reactions_added = 0
    for parent, reactions in branching.items():
        nuc = chain[parent]
        z, a, _ = zam(parent)
        folded_members = {}
        for r_name, info in reactions.items():
            metastables = info['metastables']
            if not metastables:
                continue
            existing = next((rx for rx in nuc.reactions if rx.type == r_name),
                            None)
            if existing is not None:
                ground = ReactionTuple(r_name, existing.target, existing.Q,
                                       1.0, 0)
            else:
                # MT absent from base chain -- synthesize the ground pathway.
                if r_name == "(n,n')":
                    ground = ReactionTuple(r_name, parent, 0.0, 1.0, 0)
                else:
                    daughter = _ground_product(z, a, r_name)
                    if daughter is None or daughter not in chain.nuclide_dict:
                        # No usable ground target: emit a metastable-only group
                        # (still folds -- Phase 1 tolerates no LFS 0 entry).
                        ground = None
                    else:
                        gq = info['ground']['qi'] if info['ground'] else info['qm']
                        ground = ReactionTuple(r_name, daughter, float(gq),
                                               1.0, 0)
                reactions_added += 1
            members = []
            if ground is not None:
                members.append(ground)
            for m in metastables:
                members.append(ReactionTuple(
                    f"{r_name}_m{m['liso']}", m['product'], float(m['qi']),
                    1.0, m['lfs']))
            folded_members[r_name] = members

        # Rebuild the reaction list: replace each decorated base (and any of its
        # pre-existing _m qualifiers) in place; append newly-added MTs at end.
        new_reactions = []
        emitted = set()
        for rx in nuc.reactions:
            base = _ISOMER_SUFFIX.sub('', rx.type)
            if base in folded_members:
                if base not in emitted:
                    emitted.add(base)
                    new_reactions.extend(folded_members[base])
            else:
                new_reactions.append(rx)
        for base, members in folded_members.items():
            if base not in emitted:
                new_reactions.extend(members)
        nuc.reactions = new_reactions

    # Rebuild the top-level reaction-type list (first-appearance order) so the
    # in-memory chain matches what a from_xml round-trip would produce.
    chain.reactions = []
    for nuc in chain.nuclides:
        for rx in nuc.reactions:
            if rx.type not in chain.reactions:
                chain.reactions.append(rx.type)
    return reactions_added


# =============================================================================
# Console statistics block
# =============================================================================

def print_stats(stats, mode):
    total_lfs = (stats['matched'] + stats['products_not_in_chain']
                 + stats['rtol_exceeded'] + stats['no_dk'] + stats['zero_elis']
                 + stats['duplicate_discarded'] + stats['lfs_order_dropped'])
    print(f"\n                      nuclides in PENDF library: {stats['pendf_nuclides_total']:5d}")
    print(f"               nuclides in PENDF with branching: {stats['nuclides_with_branching']:5d}")
    print("-" * 52)
    print(f"                          Total PENDF-LFS found: {total_lfs:5d}")
    if mode == 'elis':
        print(f"                                   ELIS matched: {stats['matched']:5d}")
    else:
        print(f"                              LFS-order mapped: {stats['matched']:5d}")
    if stats['rtol_exceeded']:
        print(f"                             ELIS rtol exceeded: {stats['rtol_exceeded']:5d}")
    if stats['no_dk']:
        print(f"                           No product in DK-Lib: {stats['no_dk']:5d}")
    if stats['zero_elis']:
        print(f"                       Zero-ELIS in DK-Lib (QA): {stats['zero_elis']:5d}")
    if stats['duplicate_resolved']:
        print(f"                    Duplicate mappings resolved: {stats['duplicate_resolved']:5d} ({stats['duplicate_discarded']} LFS discarded)")
    if stats['lfs_order_dropped']:
        print(f"                 LFS dropped (exceeds DK count): {stats['lfs_order_dropped']:5d}")
    print(f"                        Products not in chain: {stats['products_not_in_chain']:5d}")
    print(f"                      Reactions added to chain: {stats['reactions_added']:5d}")
    print(f"                    Ground-only MF=10 reactions: {stats['ground_only']:5d}")
    print(f"                          MF=10 audit offenders: {stats['audit_offenders']:5d}")
    print(f"                                 MF=10 rejected: {stats['rejected_count']:5d}")
    print(f"             Unique nuclides absent from DK-Lib: {stats['absent_unique_count']:5d}")


# =============================================================================
# Isomer mapping log (mirrors the GENDF tool with PENDF-* columns)
# =============================================================================

def _z_to_element(z):
    return ATOMIC_SYMBOL.get(z, f'Z{z}')


def _write_mapping_row(f, m):
    """Write a single mapping row (per-parent table)."""
    mt = m.get('mt', '?')
    reaction = m.get('reaction', '?')
    lfs = m.get('lfs')
    product = m.get('product', '?')
    liso = m.get('liso')
    half_life = m.get('half_life', '-')
    elis = m.get('elis')
    dk_elis = m.get('dk_elis')
    method = m.get('method', 'unknown')
    notes = m.get('notes', '')

    err_type = m.get('type')
    if err_type == 'elis_tol_exceeded':
        method = 'not-mapped'
        notes = 'ELIS rtol exceeded. Closest shown.'
        if product == '?':
            product = m.get('base_nuclide', '?')
    elif err_type == 'no_metastable_decay_data':
        method = 'not-mapped'
        notes = m.get('note', 'Product not in DK-Lib')
        if product == '?':
            product = m.get('base_nuclide', '?')
    elif err_type == 'zero_elis_metastables':
        method = 'not-mapped'
        skipped = m.get('skipped_states', [])
        if skipped:
            skipped_str = ', '.join(f"_m{liso_}" for liso_, _, _ in skipped)
            notes = f'DK-Lib ELIS=0 ({skipped_str})'
        else:
            notes = 'DK-Lib has ELIS=0 (data quality)'
        if product == '?':
            product = m.get('base_nuclide', '?')
    elif err_type == 'duplicate_mapping':
        method = 'not-mapped'

    mt_str = str(mt) if mt is not None else "?"
    lfs_str = str(lfs) if lfs is not None else "?"
    base_nuc = str(product).split('_')[0] if '_' in str(product) else str(product)
    pendf_product = f"{base_nuc}_m{lfs}" if lfs is not None else base_nuc
    liso_str = str(liso) if liso is not None else "-"

    if method == 'not-mapped' and 'Closest' in notes:
        chain_product = f"{base_nuc}_m{liso}" if liso is not None else base_nuc
    elif method == 'not-mapped' and err_type == 'no_metastable_decay_data':
        chain_product = m.get('product', '-') if 'not in chain' in notes else '-'
    elif method == 'not-mapped' and err_type == 'zero_elis_metastables':
        chain_product = '-'
    elif method == 'not-mapped':
        chain_product = 'OMITTED'
    else:
        chain_product = product

    if isinstance(half_life, (int, float)):
        half_life_str = f"{half_life:.3e}"
    else:
        half_life_str = str(half_life)

    elis_str = f"{elis:.1f}" if elis is not None else "N/A"

    if err_type == 'no_metastable_decay_data' and 'not in chain' not in notes:
        dk_elis_str = "-"
        disc_str = "-"
    elif err_type == 'zero_elis_metastables':
        dk_elis_str = "0.0 (invalid)"
        disc_str = "N/A"
    else:
        dk_elis_str = f"{dk_elis:.1f}" if dk_elis is not None else "N/A"
        if elis is not None and dk_elis is not None and dk_elis != 0:
            abs_diff = abs(elis - dk_elis)
            rel_diff = abs_diff / abs(dk_elis) * 100
            disc_str = f"{abs_diff:.0f}eV ({rel_diff:.2f}%)"
        else:
            disc_str = "N/A"

    if method == 'elis':
        method_display = 'ELIS'
    elif method == 'lfs_order':
        method_display = 'LFS_ORDER'
    else:
        method_display = method

    f.write(f"{mt_str:>5}  {reaction:<12}  {lfs_str:>9}  {pendf_product:<15}  "
            f"{liso_str:>7}  {chain_product:<15}  {half_life_str:>12}  "
            f"{elis_str:>14}  {dk_elis_str:>14}  {disc_str:>22}    {method_display:<12}    {notes:<30}\n")


def _write_proximity_check(f, isomer_mappings, rtol):
    """ISOMERIC STATE PROXIMITY CHECK section."""
    by_base = defaultdict(dict)
    for m in isomer_mappings:
        product = m.get('product', '')
        liso = m.get('liso')
        if '_m' in product and liso is not None:
            base = product.split('_m')[0]
            if liso not in by_base[base]:
                by_base[base][liso] = m

    pairs = []
    for base, states in by_base.items():
        if len(states) < 2:
            continue
        for (l1, s1), (l2, s2) in combinations(states.items(), 2):
            e1, e2 = s1.get('elis'), s2.get('elis')
            if e1 is not None and e2 is not None and max(e1, e2) > 0:
                rel = abs(e1 - e2) / max(e1, e2)
                pairs.append(dict(base=base, lfs1=s1.get('lfs'), elis1=e1,
                                  lfs2=s2.get('lfs'), elis2=e2, rel=rel,
                                  overlap=rel < rtol))
    pairs.sort(key=lambda p: p['rel'])

    f.write("\n\n" + "=" * 220 + "\n")
    f.write("ISOMERIC STATE PROXIMITY CHECK\n")
    f.write("=" * 220 + "\n\n")
    f.write(f"Checking for isomeric states with PENDF-ELFS values within "
            f"rtol={rtol*100:.0f}% of each other.\n")
    f.write("This could indicate potential mapping ambiguity.\n\n")
    header = (f"{'Base':<12}  {'LFS1':>5}  {'PENDF-ELFS1':>14}  {'LFS2':>5}  "
              f"{'PENDF-ELFS2':>14}  {'Rel-Diff':>10}  {'Status':<20}")
    overlaps = [p for p in pairs if p['overlap']]
    f.write("PENDF-ELFS PROXIMITY:\n")
    f.write("-" * 120 + "\n")

    def fmt(v):
        return str(v) if v is not None else '?'

    if overlaps:
        f.write(f"\nPOTENTIAL OVERLAPS DETECTED: {len(overlaps)}\n\n")
        f.write(header + "\n")
        f.write("-" * 90 + "\n")
        for p in overlaps:
            status = "OVERLAP" if p['rel'] < rtol * 0.5 else "WARNING: Near rtol"
            f.write(f"{p['base']:<12}  {fmt(p['lfs1']):>5}  {p['elis1']:>14.1f}  "
                    f"{fmt(p['lfs2']):>5}  {p['elis2']:>14.1f}  "
                    f"{p['rel']*100:>9.1f}%  {status:<20}\n")
    else:
        f.write("\nNo overlaps detected. Closest pairs (highest risk):\n\n")
        f.write(header + "\n")
        f.write("-" * 90 + "\n")
        for p in pairs[:10]:
            f.write(f"{p['base']:<12}  {fmt(p['lfs1']):>5}  {p['elis1']:>14.1f}  "
                    f"{fmt(p['lfs2']):>5}  {p['elis2']:>14.1f}  "
                    f"{p['rel']*100:>9.1f}%  {'OK':<20}\n")
        if not pairs:
            f.write("  (No multi-state nuclides found)\n")


def _write_duplicate_section(f, duplicate_errors):
    """DUPLICATE MAPPING CONFLICTS section."""
    f.write("\n\n" + "=" * 220 + "\n")
    f.write("DUPLICATE MAPPING CONFLICTS\n")
    f.write("=" * 220 + "\n\n")
    f.write("When multiple PENDF LFS values have ELFS within tolerance of the "
            "same decay library LISO,\n")
    f.write("only the closest match is kept. Discarded LFS values are shown "
            "below with full context.\n\n")
    if not duplicate_errors:
        f.write("No duplicate mapping conflicts detected.\n")
        return
    f.write(f"Total conflicts: {len(duplicate_errors)}\n")
    f.write(f"Total LFS discarded: "
            f"{sum(len(e.get('discarded', [])) for e in duplicate_errors)}\n\n")
    for err in duplicate_errors:
        f.write("-" * 120 + "\n")
        f.write(f"{err.get('nuclide', '?')} {err.get('reaction', '?')} "
                f"(MT={err.get('mt', '?')}) -> {err.get('base_nuclide', '?')}"
                f"_m{err.get('liso', '?')}\n")
        f.write("-" * 120 + "\n\n")
        f.write(f"  RESOLUTION: Kept LFS={err.get('kept_lfs')} "
                f"(DK-ELIS for _m{err.get('liso')}: "
                f"{(err.get('dk_elis') or 0.0):.0f}eV)\n\n")
        f.write(f"  DISCARDED ({len(err.get('discarded', []))} LFS):\n")
        for d in err.get('discarded', []):
            f.write(f"    LFS={d['lfs']}: ELFS={d['elis']:.0f}eV "
                    f"(diff={d['diff']:.0f}eV)\n")
        f.write("\n")


def _write_consistency_audit_section(f, offenders, audit_clean):
    """MF=10 CONSISTENCY AUDIT section: offenders (worst_dev > rtol) worst-first."""
    f.write("\n\n" + "=" * 220 + "\n")
    f.write("MF=10 CONSISTENCY AUDIT\n")
    f.write("=" * 220 + "\n\n")
    f.write("Pointwise Sum(MF=10 partials) interpolated onto the MF=3 energy "
            "grid, compared against the MF=3 total, for every reaction carrying "
            ">=1 metastable pathway.\n")
    f.write(f"Groups where BOTH sides sit below {CONSISTENCY_ABS_FLOOR:.0e} b "
            "(evaluator floor dust) are exempt -- their relative deviation is "
            "meaningless.\n")
    f.write(f"Offenders (max rel dev > {CONSISTENCY_RTOL:.0e}) are listed "
            f"worst-first; {audit_clean} audited reaction(s) are clean.\n\n")
    if not offenders:
        f.write("No offenders: all audited reactions agree within "
                f"{CONSISTENCY_RTOL:.0e}.\n")
        return
    f.write(f"Total offenders: {len(offenders)}\n\n")
    header = (f"{'Parent':<12}  {'MT':>5}  {'Reaction':<12}  {'MaxRelDev':>12}  "
              f"{'E[eV]@max':>14}  {'Sum-part[b]':>14}  {'Total[b]':>14}  "
              f"{'IntRatio':>12}")
    sep = "-" * 120
    f.write(header + "\n" + sep + "\n")
    for o in sorted(offenders, key=lambda x: x['worst_dev'], reverse=True):
        e = o.get('energy')
        sp = o.get('sum_partials')
        tot = o.get('total')
        ratio = o.get('integral_ratio')
        e_str = f"{e:.4e}" if e is not None else "-"
        sp_str = f"{sp:.4e}" if sp is not None else "-"
        tot_str = f"{tot:.4e}" if tot is not None else "-"
        ratio_str = f"{ratio:.4f}" if ratio not in (None, float('inf')) else "inf"
        f.write(f"{o['parent']:<12}  {o['mt']:>5}  {o['reaction']:<12}  "
                f"{o['worst_dev']:>12.4e}  {e_str:>14}  {sp_str:>14}  "
                f"{tot_str:>14}  {ratio_str:>12}\n")


def _write_rejected_section(f, rejected, reject_rtol):
    """MF=10 REJECTED REACTIONS section: audit-gated reactions left stock."""
    f.write("\n\n" + "=" * 220 + "\n")
    f.write("MF=10 REJECTED REACTIONS\n")
    f.write("=" * 220 + "\n\n")
    if reject_rtol is None:
        f.write("rejection disabled (audit only): --mf10-reject-rtol was not "
                "set, so no reaction was rejected on audit grounds.\n")
        return
    f.write(f"Threshold: --mf10-reject-rtol = {reject_rtol:.3e}\n")
    f.write("Criterion: a reaction whose MF=10-vs-MF=3 audit max relative "
            f"deviation exceeds {reject_rtol:.3e} is left stock (no "
            "<isomeric_branching> child).\n")
    f.write("Consequence: MF=3 total routes to the ground target; isomeric "
            "branching discarded.\n\n")
    if not rejected:
        f.write(f"No reactions exceeded the threshold {reject_rtol:.3e}.\n")
        return
    f.write(f"Total rejected: {len(rejected)}\n\n")
    header = (f"{'Parent':<12}  {'MT':>5}  {'Reaction':<12}  {'MaxRelDev':>12}  "
              f"{'Threshold':>12}  {'E[eV]@max':>14}  {'Sum-part[b]':>14}  "
              f"{'Total[b]':>14}  {'Consequence':<52}")
    sep = "-" * 160
    f.write(header + "\n" + sep + "\n")
    consequence = "left stock: MF=3 total -> ground target; branching discarded"
    for o in sorted(rejected, key=lambda x: x['worst_dev'], reverse=True):
        e = o.get('energy')
        sp = o.get('sum_partials')
        tot = o.get('total')
        e_str = f"{e:.4e}" if e is not None else "-"
        sp_str = f"{sp:.4e}" if sp is not None else "-"
        tot_str = f"{tot:.4e}" if tot is not None else "-"
        f.write(f"{o['parent']:<12}  {o['mt']:>5}  {o['reaction']:<12}  "
                f"{o['worst_dev']:>12.4e}  {o['threshold']:>12.3e}  "
                f"{e_str:>14}  {sp_str:>14}  {tot_str:>14}  {consequence:<52}\n")


_ABSENT_STATUS_LABELS = (
    ('no_decay_data', 'ABSENT ENTIRELY (no decay data for this Z,A)'),
    ('no_metastables', 'PRESENT BUT NO METASTABLE DATA (only a ground state)'),
    ('no_match', 'NO ELIS MATCH (metastables exist; none within tolerance)'),
)


def _write_absent_decay_section(f, absent_by_status):
    """NUCLIDES ABSENT FROM DECAY LIBRARY section: unique base names by status."""
    f.write("\n\n" + "=" * 220 + "\n")
    f.write("NUCLIDES ABSENT FROM DECAY LIBRARY\n")
    f.write("=" * 220 + "\n\n")
    f.write("Unique product base nuclides (GNDS ground name) whose MF=10 "
            "metastable partials could not be mapped because the decay library "
            "carries no usable metastable data,\n")
    f.write("grouped by the reason lookup_liso returned. Each name is listed "
            "once regardless of how many reactions produced it.\n\n")
    total = sum(len(absent_by_status.get(s, [])) for s, _ in _ABSENT_STATUS_LABELS)
    # Include any status not in the fixed label set (defensive).
    other = {s: n for s, n in absent_by_status.items()
             if s not in {s0 for s0, _ in _ABSENT_STATUS_LABELS}}
    total += sum(len(n) for n in other.values())
    if total == 0:
        f.write("No product nuclides were absent from the decay library.\n")
        return
    f.write(f"Total unique nuclides absent: {total}\n")
    for status, label in _ABSENT_STATUS_LABELS:
        names = absent_by_status.get(status, [])
        f.write(f"\n{label} [{status}]: {len(names)}\n")
        f.write("-" * 120 + "\n")
        if names:
            for i in range(0, len(names), 8):
                f.write("  " + "  ".join(f"{n:<12}" for n in names[i:i + 8])
                        + "\n")
        else:
            f.write("  (none)\n")
    for status, names in sorted(other.items()):
        f.write(f"\n{status}: {len(names)}\n")
        f.write("-" * 120 + "\n")
        for i in range(0, len(names), 8):
            f.write("  " + "  ".join(f"{n:<12}" for n in names[i:i + 8]) + "\n")


def write_isomer_mapping_log(log_file, stats, source_stats, mode, rtol, atol):
    """Write the comprehensive PENDF isomer mapping log."""
    isomer_mappings = stats['isomer_mappings']
    elis_errors = list(stats['elis_errors'])
    products_not_in_chain = stats['products_not_in_chain_errors']
    duplicate_errors = stats['duplicate_errors']

    with open(log_file, 'w') as f:
        f.write("=" * 220 + "\n")
        f.write("ISOMER MAPPING LOG\n")
        f.write("=" * 220 + "\n\n")

        f.write("MAPPING MODE\n")
        f.write("-" * 70 + "\n")
        if mode == 'lfs_order':
            f.write("MODE: LFS_ORDER (FISPACT-like positional mapping)\n\n")
            f.write("  Maps PENDF LFS by sorted position: 1st LFS -> _m1, 2nd -> _m2, etc.\n")
            f.write("  WARNING: May produce incorrect results for nuclides where\n")
            f.write("           LFS order does not match LISO order (e.g., Ag116).\n")
            f.write("           Use 'elis' mode for production calculations.\n")
        else:
            f.write("MODE: ELIS (decay library excitation energy matching)\n\n")
            f.write("  Maps PENDF MF=10 products to OpenMC _m{n} naming based on\n")
            f.write("  excitation energy (ELFS = QM - QI) matching with decay library.\n")
        f.write("\n\n")

        f.write("SOURCE FILES\n")
        f.write("-" * 70 + "\n")
        f.write(f"OpenMC Chain:    {source_stats['base_chain']}\n")
        f.write(f"PENDF Library:   {source_stats['pendf']}\n")
        f.write(f"Decay Library:   {source_stats['decay_file']}\n")
        f.write("-" * 70 + "\n")
        f.write(f"Output Chain:    {source_stats['output_chain']}\n")

        f.write("\n\nBASE CHAIN DETAILS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total nuclides in chain: {source_stats['chain_nuclides']}\n")

        f.write("\n\nMATCHING SETTINGS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Mapping mode:    {mode}\n")
        f.write(f"ELIS tolerance:  rtol={rtol} ({rtol*100:.0f}%), atol={atol} eV\n")
        if mode == 'elis':
            f.write("Products beyond tolerance or not in decay library are skipped.\n")
            f.write("Products mapped to nuclides absent from the chain are skipped and logged.\n\n")
        else:
            f.write("In LFS-order mode: tolerance used for ELIS reference warnings only.\n\n")

        total_lfs = (stats['matched'] + stats['products_not_in_chain']
                     + stats['rtol_exceeded'] + stats['no_dk']
                     + stats['zero_elis'] + stats['duplicate_discarded']
                     + stats['lfs_order_dropped'])
        f.write(f"                      nuclides in PENDF library: {stats['pendf_nuclides_total']:5d}\n")
        f.write(f"               nuclides in PENDF with branching: {stats['nuclides_with_branching']:5d}\n")
        f.write("-" * 52 + "\n")
        f.write(f"                          Total PENDF-LFS found: {total_lfs:5d}\n")
        if mode == 'elis':
            f.write(f"                                   ELIS matched: {stats['matched']:5d}\n")
        else:
            f.write(f"                              LFS-order mapped: {stats['matched']:5d}\n")
        if stats['rtol_exceeded']:
            f.write(f"                             ELIS rtol exceeded: {stats['rtol_exceeded']:5d}\n")
        if stats['no_dk']:
            f.write(f"                           No product in DK-Lib: {stats['no_dk']:5d}\n")
        if stats['zero_elis']:
            f.write(f"                       Zero-ELIS in DK-Lib (QA): {stats['zero_elis']:5d}\n")
        if stats['duplicate_resolved']:
            f.write(f"                    Duplicate mappings resolved: {stats['duplicate_resolved']:5d} ({stats['duplicate_discarded']} LFS discarded)\n")
        if stats['lfs_order_dropped']:
            f.write(f"                 LFS dropped (exceeds DK count): {stats['lfs_order_dropped']:5d}\n")
        f.write(f"                          Products not in chain: {stats['products_not_in_chain']:5d}\n")
        f.write(f"                       Reactions added to chain: {stats['reactions_added']:5d}\n")
        f.write(f"                     Ground-only MF=10 reactions: {stats['ground_only']:5d}\n")
        f.write(f"                          MF=10 audit offenders: {stats['audit_offenders']:5d}\n")
        f.write(f"                                 MF=10 rejected: {stats['rejected_count']:5d}\n")
        f.write(f"             Unique nuclides absent from DK-Lib: {stats['absent_unique_count']:5d}\n")
        f.write("\n")

        # Column definitions
        f.write("COLUMN DEFINITIONS\n")
        f.write("-" * 93 + "\n")
        f.write("MT              = ENDF reaction type number\n\n")
        f.write("Reaction        = Reaction name (e.g., (n,g), (n,2n))\n\n")
        f.write("PENDF-LFS       = Level number of the state of ZAP formed by the neutron interaction.\n")
        f.write("                  Indicator to specify the level number of the nuclide (ZAP) (as defined in\n")
        f.write("                  File 8) produced in the reaction (MT number).\n\n")
        f.write("PENDF-Product   = Product nuclide from PENDF reaction (using _m{LFS} naming)\n\n")
        f.write("DK-LISO         = Decay library isomeric state number (_m1=1, _m2=2, etc.).\n")
        f.write("                  (Only isomeric levels)\n\n")
        f.write("CHAIN-Product   = Product nuclide name in chain (after ELIS mapping) (using _m{LISO} naming)\n\n")
        f.write("CHAIN-t1/2      = Half-life from chain (originally from decay library)\n\n")
        f.write("PENDF-ELFS[eV]  = Excitation energy of final state calculated from PENDF MF=10 (QM - QI).\n")
        f.write("                  Excitation energy of the reaction product.\n\n")
        f.write("DK-ELIS[eV]     = Excitation energy from decay library MF=1 MT=451 (matched within tolerance).\n")
        f.write("                  Excitation energy of the target nucleus relative to 0.0 for the ground state.\n\n")
        f.write("D(ELFS-ELIS)    = Discrepancy between PENDF-ELFS and DK-ELIS.\n")
        f.write("                  Shows absolute difference [eV] and relative difference [%].\n\n")
        f.write("Method          = 'ELIS' (matched via excitation energy), 'not-mapped' (failed to match)\n\n")
        f.write("Notes           = Additional information (skip reason / renormalization).\n\n")
        f.write("Further info:\n")
        f.write("  | LIS/LFS | Level number          | All excited states (short-lived + long-lived) |\n")
        f.write("  | LISO    | Isomeric state number | Only Isomeric (metastable/long-lived) states  |\n\n")
        f.write("=" * 93 + "\n\n")

        # Per-parent tables
        by_parent = defaultdict(list)
        for m in isomer_mappings:
            by_parent[m['parent']].append(m)
        for e in elis_errors:
            by_parent[e.get('parent', '?')].append(e)
        for e in products_not_in_chain:
            by_parent[e.get('parent', '?')].append(e)

        header = (f"{'MT':>5}  {'Reaction':<12}  {'PENDF-LFS':>9}  {'PENDF-Product':<15}  "
                  f"{'DK-LISO':>7}  {'CHAIN-Product':<15}  {'CHAIN-t1/2[s]':>12}  "
                  f"{'PENDF-ELFS[eV]':>14}  {'DK-ELIS[eV]':>14}  {'D(ELFS-ELIS)':>22}    {'Method':<12}    {'Notes':<30}")
        sep = "-" * 220
        for parent in sorted(by_parent):
            f.write(f"\n{parent}:\n{sep}\n{header}\n{sep}\n")
            for m in sorted(by_parent[parent],
                            key=lambda x: (x.get('mt') or 0, x.get('method', ''))):
                _write_mapping_row(f, m)

        # HIGH DISCREPANCY + FAILED MAPPINGS
        f.write("\n\n" + "=" * 220 + "\n")
        f.write("HIGH ELIS DISCREPANCY (>50%) + FAILED MAPPINGS\n")
        f.write("=" * 220 + "\n\n")
        high = []
        for m in isomer_mappings:
            elis, dk = m.get('elis'), m.get('dk_elis')
            if elis is not None and dk is not None and dk != 0:
                rel = abs(elis - dk) / abs(dk) * 100
                if rel > 50.0:
                    high.append({**m, 'rel_diff': rel})
        for e in elis_errors:
            high.append({**e, 'rel_diff': e.get('diff_percent', float('inf')),
                         'method': 'not-mapped'})
        for e in products_not_in_chain:
            high.append({**e, 'rel_diff': float('inf'), 'method': 'not-mapped'})
        high.sort(key=lambda x: x.get('rel_diff', 0), reverse=True)
        if not high:
            f.write("No entries with >50% discrepancy or failed mappings.\n")
        else:
            mapped = sum(1 for h in high if h.get('method') != 'not-mapped')
            f.write(f"Total: {len(high)} (mapped: {mapped}, not-mapped: {len(high)-mapped})\n\n")
            f.write(f"{sep}\n{'Parent':<12}  {header}\n{sep}\n")
            for m in high:
                f.write(f"{m.get('parent', '?'):<12}  ")
                _write_mapping_row(f, m)

        _write_proximity_check(f, isomer_mappings, rtol)
        _write_duplicate_section(f, duplicate_errors)
        _write_consistency_audit_section(
            f, stats.get('audit_offenders_list', []),
            stats.get('audit_clean', 0))
        _write_rejected_section(
            f, stats.get('rejected', []), stats.get('reject_rtol'))
        _write_absent_decay_section(f, stats.get('absent_by_status', {}))

    print(f"Isomer mapping log written to: {log_file}")


# =============================================================================
# Main workflow
# =============================================================================

def main(base_chain_file, pendf_path, decay_file, output_chain_file,
         log_file=None, mapping_mode='elis', elis_rtol=ELIS_RTOL,
         elis_atol=ELIS_ATOL, verbose=True, library=None, reject_rtol=None):
    """Patch a chain with PENDF MF=10 isomeric branching. Returns the Chain."""
    if decay_file is None:
        raise ValueError("decay_file is required for isomeric branching.")
    if mapping_mode not in ('elis', 'lfs_order'):
        raise ValueError(f"Invalid mapping_mode {mapping_mode!r}.")

    print("=" * 60)
    print("PENDF Isomeric Branching Chain Patcher v1")
    print("=" * 60)
    print(f"\nMAPPING MODE: {mapping_mode.upper()}")
    if mapping_mode == 'lfs_order':
        print("  (FISPACT-like positional mapping for validation)")
    else:
        print("  (ELIS-based matching - production recommended)")

    print("\nStep 1: Loading base chain...")
    chain = Chain.from_xml(base_chain_file)
    print(f"  Loaded {len(chain.nuclides)} nuclides")

    print("\nStep 2: Opening PENDF source...")
    source = open_pendf_source(pendf_path, library=library)
    print(f"  Backend: {source.kind}; nuclides: {len(source.nuclides)}; "
          f"library: {source.library}")
    if source.kind == 'h5' and source.mapping not in (None, 'elis', 'lfs_order'):
        print(f"  NOTE: PENDF library mapping attr = {source.mapping!r}")

    print("\nStep 3: Loading decay library...")
    decay_lookup = parse_decay_isomeric_levels(decay_file)
    print(f"  Decay states for {len(decay_lookup)} (Z, A) nuclides")

    print("\nStep 4: Mapping MF=10 isomeric branching...")
    print(f"  ELIS tolerance: rtol={elis_rtol} ({elis_rtol*100:.0f}%), "
          f"atol={elis_atol} eV")
    if reject_rtol is not None:
        print(f"  MF=10 audit rejection: worst rel dev > {reject_rtol:.3e} "
              f"-> reaction left stock")
    else:
        print("  MF=10 audit: detection + logging only (no rejection)")
    branching, stats = map_library(
        source, chain, decay_lookup, mapping_mode, elis_rtol, elis_atol,
        verbose=verbose, reject_rtol=reject_rtol)

    print("\nStep 5: Decorating chain...")
    reactions_added = decorate_chain(chain, branching)
    stats['reactions_added'] = reactions_added

    print_stats(stats, mapping_mode)

    print("\nStep 6: Exporting folded chain XML...")
    chain.export_to_xml(output_chain_file)
    print(f"  Chain written to: {output_chain_file}")

    if log_file:
        source_stats = dict(
            base_chain=str(base_chain_file), pendf=str(pendf_path),
            decay_file=str(decay_file), output_chain=str(output_chain_file),
            chain_nuclides=len(chain.nuclides))
        write_isomer_mapping_log(log_file, stats, source_stats, mapping_mode,
                                 elis_rtol, elis_atol)

    try:
        source.close()
    except Exception:
        pass
    return chain


def _resolve_paths(args):
    """Resolve preset + explicit overrides into concrete paths."""
    config = LIBRARY_CONFIGS.get(args.library, {}) if args.library else {}

    base_chain = args.base_chain or config.get('base_chain')
    pendf = args.pendf or config.get('pendf')
    decay_file = args.decay_file or config.get('decay_file')

    suffix = '.elis_mapped' if args.map == 'elis' else '.lfs_order_mapped'
    if args.output_chain is not None:
        output_chain = args.output_chain
    elif config:
        output_chain = (Path(config['output_dir'])
                        / f"{config['output_prefix']}{suffix}.xml")
    else:
        output_chain = None

    if args.log_file is not None:
        log_file = args.log_file
    elif config:
        log_file = (Path(config['output_dir'])
                    / f"{config['log_prefix']}{suffix}.txt")
    else:
        log_file = None

    missing = [n for n, v in (('base-chain', base_chain), ('pendf', pendf),
                              ('decay-file', decay_file),
                              ('output-chain', output_chain)) if v is None]
    if missing:
        raise SystemExit(
            "Missing required path(s): " + ', '.join(missing) +
            ". Pass -l/--library for a preset or provide the explicit "
            "--base-chain/--pendf/--decay-file/--output-chain overrides.")
    return base_chain, pendf, decay_file, output_chain, log_file


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    verbose = not args.quiet

    base_chain, pendf, decay_file, output_chain, log_file = _resolve_paths(args)

    print("=" * 70)
    print("PENDF Isomeric Branching Chain Patcher v1")
    print("=" * 70)
    if args.library:
        print(f"\nLibrary:      {args.library} - "
              f"{LIBRARY_CONFIGS[args.library]['description']}")
    print(f"Mapping mode: {args.map}")
    print(f"Tolerances:   rtol={args.rtol}, atol={args.atol}")
    if args.mf10_reject_rtol is not None:
        print(f"MF=10 reject: worst rel dev > {args.mf10_reject_rtol}")
    else:
        print("MF=10 reject: off (audit only)")
    print(f"\nInput chain:  {base_chain}")
    print(f"PENDF source: {pendf}")
    print(f"Decay lib:    {decay_file}")
    print(f"Output chain: {output_chain}")
    print(f"Mapping log:  {log_file}")
    print()

    main(base_chain_file=str(base_chain), pendf_path=str(pendf),
         decay_file=str(decay_file), output_chain_file=str(output_chain),
         log_file=str(log_file) if log_file else None,
         mapping_mode=args.map, elis_rtol=args.rtol, elis_atol=args.atol,
         verbose=verbose, library=args.library,
         reject_rtol=args.mf10_reject_rtol)

    print("\n" + "=" * 70)
    print("Done. Chain saved to:", output_chain)
    print("=" * 70)
