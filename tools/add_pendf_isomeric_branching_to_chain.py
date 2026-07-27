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
from openmc.data.pendf import tape_identity
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
# LFS sentinel values ("unspecified isomer" conventions) -- report-only guard
# =============================================================================

# Some evaluations tag a reaction product whose final-state LEVEL could not be
# resolved with a SENTINEL LFS instead of a true level index. These sentinels
# are NOT level ordinals and must NEVER be interpreted as isomer ordinals -- a
# sentinel must never become ``_m99`` / ``_m40``. ELIS mapping is unaffected
# (it matches the real ELFS excitation energy, QM - QI); only a positional
# ``lfs_order`` consumer would mis-name them. Detection below is REPORT-ONLY:
# no mapping decision and no byte of the output chain depends on it.
SENTINEL_LFS = {
    99: 'ENDF/JEFF convention: isomer of unspecified level',
    40: 'TENDL convention: isomer of unspecified level',
}

# Human-readable outcome per classification bucket, for the sentinel report.
_SENTINEL_OUTCOME = {
    'matched':              'mapped (in chain)',
    'product_not_in_chain': 'mapped (product NOT in chain)',
    'rtol_exceeded':        'ELIS rtol exceeded (unmapped)',
    'no_dk':                'no DK-Lib data (unmapped)',
    'zero_elis':            'DK-Lib ELIS=0 (unmapped)',
    'duplicate':            'duplicate LFS (discarded)',
    'lfs_order_dropped':    'lfs_order dropped (unmapped)',
}


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
    parser.add_argument('-l', '--library',          choices=list(LIBRARY_CONFIGS), metavar='LIB', default=None,   help='Library preset (see list below); provides defaults for the paths')
    parser.add_argument('--base-chain',             type=Path,                                    default=None,   help='Base OpenMC chain XML (overrides preset)')
    parser.add_argument('--pendf',                  type=Path,                                    default=None,   help='PENDF source: an .h5 library file OR a directory of ASC .pendf/.asc tapes (overrides preset)')
    parser.add_argument('--decay-file',             type=Path,                                    default=None,   help='ENDF decay library file or directory (overrides preset)')
    parser.add_argument('--output-chain',           type=Path,                                    default=None,   help='Output chain XML (overrides preset-derived name)')
    parser.add_argument('--log-file',               type=Path,                                    default=None,   help='Isomer mapping log file (overrides preset-derived name)')
    parser.add_argument('-m', '--map',              choices=['elis', 'lfs_order'],                default='elis', help="Mapping mode: 'elis' (production, default) or 'lfs_order' (FISPACT validation)")
    parser.add_argument('-r', '--rtol',             type=float,                                   default=0.50,   help='Relative tolerance for ELIS matching (default: 0.50 = 50%%)')
    parser.add_argument('-a', '--atol',             type=float,                                   default=0.0,    help='Absolute tolerance for ELIS matching in eV (default: 0.0)')
    parser.add_argument('--audit-emax',             type=float,                                   default=2.0e7,  help='Cap the MF=10-vs-MF=3 audit at E <= this many eV (default: 2.0e7 = application group cap; MF=10 partials legitimately stop near 30 MeV while MF=3 runs to 200 MeV)')
    parser.add_argument('--mf10-reject-rtol',       type=float,                                   default=None,   help="Leave a reaction stock (no isomeric branching) when its MF=10-vs-MF=3 audit max rel dev exceeds X (self-loop-ground reactions are EXEMPT -- a metastable-only (n,n') has dev pinned at 1.0) (default: None = audit only, reject nothing)")
    parser.add_argument('--mf10-reject-band-ratio', type=float,                                   default=None,   help='Leave a reaction stock when any DEFINED lethargy-weighted band ratio has ratio-1 > X (over-summing ONLY; under-summing never rejects -- the collapse silence-fill and Class-4 policy own it) (self-loop-ground reactions are EXEMPT -- their ground route is a depletion-matrix no-op) (default: None = off; None-ratio bands never trigger)')
    parser.add_argument('--emit-mf10-only-reactions', action='store_true',                        default=False,  help='Emit chain reactions for MTs the library carries as MF=10 partials with NO MF=3 total (their total is the partial sum). Needs an h5 built with MF=10-only totals (tools/pendf_to_hdf5.py, on by default) or an ASC tape source; MT=5/MT=18 are never emitted. Default: off -- the class stays invisible in BOTH source forms.')
    parser.add_argument('-v', '--verbose',          action='store_true',                          default=True,   help='Enable verbose output (default: True)')
    parser.add_argument('-q', '--quiet',            action='store_true',                          default=False,  help='Disable verbose output')
    parser.add_argument('--prune-nn-prime-self-loops', action='store_true',                       default=False,  help="Remove (n,n') reactions with no isomeric branching whose target is EXACTLY the parent -- ground-parent self-loops that are an exact no-op in the depletion matrix. A metastable parent's (n,n') to ground is real isomer burnup and is KEPT. Default: keep all (n,n') reactions.")
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


def _safe_gnds_name(z, a, liso=0):
    """GNDS name for ``(z, a, liso)``, or a ``Z<z>-A<a>`` placeholder off-table.

    ``gnds_name`` raises ``KeyError`` for a Z absent from ``ATOMIC_SYMBOL``
    (a corrupt IZAP record, Z > 118). Every caller here is building a LOG or
    WARNING record, where one malformed tape record must not abort a
    whole-library run; the placeholder can never match a chain nuclide, so such
    a record is reported rather than silently mapped.
    """
    if z not in ATOMIC_SYMBOL:
        return f"Z{z}-A{a}" if liso == 0 else f"Z{z}-A{a}_m{liso}"
    return gnds_name(z, a, liso)


def _ground_product(z, a, r_name):
    """Return the DADZ ground-state product GNDS name for reaction ``r_name``.

    Returns ``None`` when the (z, a) shift pushes the product below Z=1.
    """
    delta_a, delta_z = DADZ[r_name]
    zp = z + delta_z
    if zp not in ATOMIC_SYMBOL:
        return None
    return gnds_name(zp, a + delta_a, 0)


def _self_loop_ground(parent, r_name, partials, chain):
    """True when reaction ``r_name``'s GROUND pathway returns to ``parent``.

    Such a ground route -- e.g. ``(n,n')`` on a ground-state parent, whose
    LFS=0 partial produces the parent itself -- is a transmutation-matrix
    self-loop no-op (loss and gain both land on the diagonal and cancel), so
    the partial-sum-vs-total completeness the band audit measures cannot affect
    the chain: only the metastable partials carry real isomer production.
    Reactions flagged here are therefore exempt from the band-ratio rejection
    gate (rejecting would destroy valid isomer production -- e.g. JEFF In113
    ``(n,n')`` has Fast=0.13 because MF=10 enumerates only ~13% of inelastic,
    yet its m1 partial is the physically wanted cross section).

    The ground product name is taken from the LFS=0 partial's IZAP
    (``gnds_name(z, a, 0)``); when the reaction carries no LFS=0 partial, it
    falls back to the base chain's existing target for this reaction on
    ``parent``, and failing that to what :func:`decorate_chain` would
    SYNTHESIZE for the missing base reaction -- ``(n,n')`` with target ==
    parent on a GROUND-state parent (a METASTABLE parent's synthesized ground is
    the true ground, ``In115_m1 -> In115``, with Q = +ELIS(parent)).
    Without that last step the audit and the decoration would disagree
    about what a self-loop is for exactly the reactions the decoration
    synthesizes. The match must be EXACT: for a metastable parent (e.g.
    ``In115_m1 (n,n') -> In115`` ground) the ground route is a real
    isomer-burnup transition, NOT a self-loop, so ``parent`` never equals the
    ground name and the reaction stays rejectable -- including in the
    synthesized case, which is why the fallback requires a ground-state parent.
    """
    ground = next((p for p in partials if p['lfs'] == 0), None)
    if ground is not None:
        z, a = ground['izap'] // 1000, ground['izap'] % 1000
        if z not in ATOMIC_SYMBOL:
            return False
        return gnds_name(z, a, 0) == parent
    # No LFS=0 partial: fall back to the base chain's existing target.
    if parent in chain.nuclide_dict:
        existing = next((rx for rx in chain[parent].reactions
                         if rx.type == r_name), None)
        if existing is not None:
            return existing.target == parent
    # No base-chain entry either: decorate_chain synthesizes a GROUND-state
    # parent's "(n,n')" ground as target == parent, so its ground route really is
    # the diagonal no-op the exemption is about. A metastable parent's (n,n') to
    # ground is isomer burnup (synthesized as parent -> true ground), never a
    # self-loop.
    return r_name == "(n,n')" and not _ISOMER_SUFFIX.search(parent)


# =============================================================================
# PENDF source adapters (h5 and ASC), one interface
# =============================================================================

def _h5_attr_text(attrs, key):
    """Return HDF5 string attribute ``key`` as ``str``, or ``None`` if absent."""
    if key not in attrs:
        return None
    value = attrs[key]
    return value.decode() if isinstance(value, bytes) else str(value)


def _mf10_only_qm_qi(partials):
    """``(QM, QI)`` of a reaction whose Q values must come from MF=10 itself.

    An MF=10 section with no MF=3 sibling has no HEAD record to read Q values
    from, so they are taken from the partials, matching what the HDF5 builder
    stores (``openmc.data.pendf._synthesize_mf10_total``): QM is the partials'
    shared section QM and QI is the LFS=0 partial's QI when the section has one
    (ELFS = 0 there, so QI == QM), else QM. No Q value is ever fabricated.
    """
    qm = float(partials[0]['qm'])
    qi = next((float(p['qi']) for p in partials if p['lfs'] == 0), qm)
    return qm, qi


class _H5Source:
    """PENDF HDF5 library backend built on :class:`openmc.data.PendfLibrary`.

    Public metadata (``nuclides``, ``library``, ``mapping``) comes from the
    reader; per-partial ``QI``/``QM``/``ELFS`` attributes are read directly off
    the HDF5 groups (they have no public accessor -- documented in
    ``openmc/data/pendf.py``).

    A reaction the builder stored with a total SYNTHESIZED from its MF=10
    partials (group attr ``total_source='sum-mf10'`` -- an MT with no MF=3
    section) is served only when ``emit_mf10_only_reactions`` is set. With the
    flag off it is filtered out and booked in ``mf10_without_mf3``, exactly as
    the ASC adapter excludes the same tape sections, so rebuilding an h5 with
    the feature on cannot silently change a flag-off chain and the two source
    forms keep enumerating identical reactions.
    """

    kind = 'h5'

    def __init__(self, path, emit_mf10_only_reactions=False):
        self._lib = openmc.data.PendfLibrary(path)
        self.nuclides = list(self._lib.nuclides)
        # For provenance stamping prefer the h5's tape-derived source_identity
        # (matches what the collapse verifies against), falling back to the
        # user-supplied ``library`` label on files that predate it.
        self.library = (self._lib.source_identity or self._lib.library
                        or 'unknown')
        self.mapping = self._lib.mapping
        self.emit_mf10_only = bool(emit_mf10_only_reactions)
        # ``None`` (no root attr on some file) means the library predates the
        # feature and cannot serve the class at all -- distinct from a count of
        # 0 (feature on, nothing in the tapes needed it).
        self.mf10_only_totals = self._lib.mf10_only_totals
        self.serve_mf10_only = (self.emit_mf10_only
                                and self.mf10_only_totals is not None)
        self.mf10_without_mf3 = []       # excluded, undecorable MT groups
        self._recorded = set()           # (nuclide, mt) already booked
        if self.emit_mf10_only and self.mf10_only_totals is None:
            # Warned once, at open time: nothing in this file is stamped, so the
            # class is served exactly as if the flag were off.
            print("  WARNING: --emit-mf10-only-reactions: h5 predates MF=10-only "
                  "totals; rebuild with tools/pendf_to_hdf5.py to serve this "
                  "class. Nothing will be emitted for it.", file=sys.stderr)

    def reactions(self, nuclide):
        grp = self._lib._groups[nuclide]
        out = {}
        for mtk in grp:
            if not mtk.startswith('MT'):
                continue
            mt = int(mtk[2:])
            mtg = grp[mtk]
            # MF=10-only reactions (no MF=3 sibling, total = Sum(MF=10)) are
            # served only under the flag; MT=5 (lumped channel) and MT=18
            # (fission placeholders) never are -- the builder stores neither,
            # the guard is defensive.
            mf3_less = _h5_attr_text(mtg.attrs, 'total_source') == 'sum-mf10'
            if mf3_less and (not self.serve_mf10_only or mt in (5, 18)):
                self._record_mf10_without_mf3(nuclide, mt, mtg)
                continue
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
                           partials=partials, mf3_less=mf3_less)
        return out

    def _record_mf10_without_mf3(self, nuclide, mt, mtg):
        """Book one excluded MF=10-without-MF=3 reaction for the mapping log.

        The h5 twin of :meth:`_AscSource._record_mf10_without_mf3`, producing
        the same record shape from the stored group attrs so an h5-sourced run
        and a tape-sourced run report the same class. Only NAMED transmutation
        MTs are recorded (MT=5/MT=18 have no chain reaction name and were never
        candidates), and each (nuclide, MT) is booked once however often
        :meth:`reactions` is called.
        """
        r_name = _MT_TO_NAME.get(mt)
        if r_name is None or (nuclide, mt) in self._recorded:
            return
        self._recorded.add((nuclide, mt))
        lfs, products = [], []
        for sk in sorted(mtg, key=lambda k: int(mtg[k].attrs['LFS'])
                         if k.startswith('LFS') else -1):
            if not sk.startswith('LFS'):
                continue
            attrs = mtg[sk].attrs
            lfs.append(int(attrs['LFS']))
            izap = int(attrs['IZAP'])
            product = _safe_gnds_name(izap // 1000, izap % 1000)
            if product not in products:
                products.append(product)
        self.mf10_without_mf3.append(dict(
            parent=nuclide, mt=mt, reaction=r_name, lfs=lfs, products=products,
            metastable=any(v != 0 for v in lfs)))

    def total_xs(self, nuclide, mt):
        """(energy, xs) of the reaction's stored total (barn vs eV).

        The MF=3 cross section, or -- for an MF=10-only MT served under
        ``--emit-mf10-only-reactions`` -- the total the builder synthesized as
        the sum of that reaction's MF=10 partials.
        """
        return self._lib.xs(nuclide, mt)

    def pathway_xs(self, nuclide, mt, lfs, izap=None):
        """(energy, xs) of one MF=10 isomeric-production partial."""
        return self._lib.pathway_xs(nuclide, mt, lfs, izap)

    def nuclide_elis(self, nuclide):
        """Target excitation energy [eV] (MF=1/451 ELIS), or ``None``.

        Stored per nuclide group by the h5 builder (``openmc/data/pendf.py``:
        ``nuc.attrs['ELIS'] = ev.target['excitation_energy']``); ``None`` only on
        a file written before that attr existed.
        """
        attrs = self._lib._groups[nuclide].attrs
        return float(attrs['ELIS']) if 'ELIS' in attrs else None

    def close(self):
        self._lib.close()


class _AscSource:
    """ASC PENDF tape backend (directory of ``.pendf``/``.asc`` tapes).

    Reuses the MF=8/10 metadata extraction primitives from
    ``openmc.data.pendf`` (``_discover_pendf_files``, ``_iter_mf10_partials``)
    rather than duplicating the ENDF-6 record parsing. No HDF5 is written.

    An MF=10 section with no MF=3 sibling is EXCLUDED from :meth:`reactions`
    (see :meth:`_record_mf10_without_mf3`) and booked in ``mf10_without_mf3``
    for the log, so a tape-sourced run enumerates exactly the reactions an
    h5-sourced run does. With ``emit_mf10_only_reactions`` set, a NAMED MT of
    that class is served instead -- Q values from the MF=10 section itself and
    the total synthesized as the union-grid sum of the partials, exactly what an
    h5 built with MF=10-only totals stores -- while MT=5 and MT=18 stay
    excluded either way.
    """

    kind = 'asc'

    def __init__(self, path, library=None, emit_mf10_only_reactions=False):
        from openmc.data.pendf import _discover_pendf_files, _iter_mf10_partials
        from openmc.data.endf import (Evaluation, get_head_record,
                                       get_tab1_record)
        import io

        # Provenance identity from the tapes themselves (TPID / MF=1/451), so the
        # stamp matches what the collapse verifies against; 'unknown' only when
        # no tape identity can be read.
        self.library = tape_identity(Path(path)) or 'unknown'
        self.mapping = None
        self.emit_mf10_only = bool(emit_mf10_only_reactions)
        self._data = {}
        self._tapes = {}                 # GNDS name -> tape path
        self._elis = {}                  # GNDS name -> MF=1/451 ELIS [eV]
        self._xs_cache_name = None       # single-nuclide (energy, xs) cache
        self._xs_cache = None
        self.mf10_without_mf3 = []       # excluded, undecorable tape sections
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
                # An MF=10 section with no MF=3 sibling carries no total of its
                # own. Unless --emit-mf10-only-reactions is set it is EXCLUDED
                # here -- before any candidate enumeration or stats counting --
                # because with the flag off no source form serves it (a new h5
                # filters its stamped groups the same way), so decorating one
                # would put a chain row that collapses to a silent zero. Under
                # the flag a NAMED MT is served with the union-grid partial sum
                # as its total, matching the h5 build; MT=5 (lumped channel) and
                # MT=18 (fission placeholders) stay excluded either way.
                mf3_less = (3, mt) not in ev.section
                if mf3_less and (not self.emit_mf10_only or mt in (5, 18)):
                    self._record_mf10_without_mf3(ev, name, mt)
                    continue
                qm = qi = None
                if not mf3_less:
                    fo = io.StringIO(ev.section[3, mt])
                    get_head_record(fo)
                    (qm, qi, _l1, _lr), _tab = get_tab1_record(fo)
                partials = []
                for pqm, pqi, izap, lfs, _ptab in _iter_mf10_partials(
                        ev, mt, name):
                    partials.append(dict(
                        lfs=int(lfs), izap=int(izap), qi=float(pqi),
                        qm=float(pqm), elfs=float(pqm - pqi)))
                if mf3_less:
                    if not partials:
                        # Nothing to synthesize a total from: undecorable after
                        # all, booked like a flag-off exclusion.
                        self._record_mf10_without_mf3(ev, name, mt)
                        continue
                    # No MF=3 HEAD to read Q from: MF=10 sources them.
                    qm, qi = _mf10_only_qm_qi(partials)
                reactions[mt] = dict(qm=float(qm), qi=float(qi),
                                     partials=partials, mf3_less=mf3_less)
            self._data[name] = reactions
            self._tapes[name] = tape
            # The target's own MF=1/451 excitation energy (ELIS, eV); ``None``
            # when the header lacks the record -- never guessed downstream.
            self._elis[name] = ev.target.get('excitation_energy')
        self.nuclides = sorted(self._data)

    def _record_mf10_without_mf3(self, ev, name, mt):
        """Book one excluded MF=10-without-MF=3 section for the mapping log.

        Only NAMED transmutation MTs are recorded: MT=5 (lumped residual) and
        MT=18 (fission) have no entry in ``_MT_TO_NAME``, were never decoration
        candidates, and stay silent exactly as before. Products come from the
        partials' IZAP (ground base names -- the LFS->LISO isomer mapping is not
        run for a section that is being discarded).
        """
        r_name = _MT_TO_NAME.get(mt)
        if r_name is None:
            return
        from openmc.data.pendf import _iter_mf10_partials
        lfs, products = [], []
        for _pqm, _pqi, izap, plfs, _ptab in _iter_mf10_partials(ev, mt, name):
            lfs.append(int(plfs))
            product = _safe_gnds_name(int(izap) // 1000, int(izap) % 1000)
            if product not in products:
                products.append(product)
        self.mf10_without_mf3.append(dict(
            parent=name, mt=mt, reaction=r_name, lfs=lfs, products=products,
            metastable=any(v != 0 for v in lfs)))

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
            if (total is None and self.emit_mf10_only and mt not in (5, 18)
                    and partials):
                # Served MF=10-only reaction: its total is the union-grid sum of
                # its own partials, the same arithmetic the h5 build stores
                # (``openmc.data.pendf._synthesize_mf10_total``), so the audit
                # sees identical numbers from either source form. The sum equals
                # the partials by construction, which is why such a reaction can
                # never be an audit offender or be band-rejected.
                grids = [pe for pe, _pxs in partials.values()]
                e = np.unique(np.concatenate(grids))
                xs = np.zeros_like(e)
                for pe, pxs in partials.values():
                    xs = xs + np.interp(e, pe, pxs, left=0.0, right=0.0)
                total = (e, xs)
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

    def nuclide_elis(self, nuclide):
        """Target excitation energy [eV] (MF=1/451 ELIS), or ``None``.

        Read off the tape's own MF=1/451 header at discovery time
        (``Evaluation.target['excitation_energy']``) -- the same record the h5
        builder stores as the nuclide group's ``ELIS`` attr.
        """
        return self._elis.get(nuclide)

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


def open_pendf_source(path, library=None, emit_mf10_only_reactions=False):
    """Return the PENDF source adapter for ``path`` (h5 file or ASC directory).

    ``emit_mf10_only_reactions`` is handed to whichever adapter is built, so the
    MF=10-without-MF=3 class is visible -- or invisible -- in BOTH source forms
    under one switch.
    """
    path = Path(path)
    emit = emit_mf10_only_reactions
    if path.is_file() and path.suffix == '.h5':
        return _H5Source(path, emit_mf10_only_reactions=emit)
    if path.is_dir() and any(path.glob('*.h5')):
        return _H5Source(path, emit_mf10_only_reactions=emit)
    if path.is_dir():
        return _AscSource(path, library=library,
                          emit_mf10_only_reactions=emit)
    if path.is_file():
        return _H5Source(path, emit_mf10_only_reactions=emit)
    raise FileNotFoundError(str(path))


def _nuclide_elis(source, nuclide):
    """``nuclide``'s own excitation energy [eV] from the source, or ``None``.

    Both adapters serve the target's MF=1/451 ELIS record (h5: the nuclide
    group's ``ELIS`` attr; ASC: ``Evaluation.target['excitation_energy']``); an
    adapter without the accessor, a file predating the attr, or a lookup failure
    yields ``None``, which callers must treat as "unknown" and never guess
    around -- :func:`decorate_chain` leaves the reaction stock instead.
    """
    getter = getattr(source, 'nuclide_elis', None)
    if getter is None:
        return None
    try:
        elis = getter(nuclide)
    except Exception:
        return None
    return None if elis is None else float(elis)


def _pendf_source_label(path) -> str:
    """Short human-readable label of the PENDF source for the chain stamp.

    A directory uses its last two path components (e.g. ``'jeff40-n/pendf'``) so
    the label distinguishes sibling projection dirs; a single file uses its bare
    basename. Informational only -- never a mismatch trigger.
    """
    p = Path(path)
    if p.is_dir():
        parts = p.parts
        return '/'.join(parts[-2:]) if len(parts) >= 2 else p.name
    return p.name


# =============================================================================
# Pointwise MF=10-vs-MF=3 consistency audit
# =============================================================================

# Lethargy band edges (eV): thermal [grid_min, 0.625), epithermal [0.625, 1e5),
# intermediate [1e5, 1e6), fast [1e6, emax]. The 0.625 eV thermal/epithermal
# split is the cadmium cutoff. The four ratios diagnose WHERE the MF=10 partials
# depart from the MF=3 total, which the single full-range number hides.
_BAND_THERMAL_HI = 0.625
_BAND_EPITHERMAL_HI = 1.0e5
_BAND_INTERMEDIATE_HI = 1.0e6

# Pathway-Q sanity gate (eV, absolute): how far a non-MT=4 reaction's own MF=10
# ground QM may sit from the base chain's scalar Q before the file's absolute Q
# scale is refused and the metastable pathways keep chain-anchored values. 1 eV
# separates the populations cleanly: benign file jitter on agreeing sections is
# <= 0.31 eV, the smallest genuine divergence is Am241 (n,gamma) at 3350 eV
# (where the chain scalar is 344x closer to AME than the file), and the corrupt
# ENDF/B-8.1 (n,alpha) sections are 7.9-12.0 MeV out, sign included. Sub-eV
# agreement is not luck: both numbers descend from the same mass evaluation.
PATHWAY_Q_CHAIN_TOL = 1.0

# Reverted pathway Q values are rounded to this many decimals: ``Q - ELFS`` is a
# float subtraction that leaves artefacts (-13286354.999999998) the file-sourced
# path never produces. Applied to REVERTED values only, so a run in which the
# gate never fires serializes byte-identically to one without it.
_PATHWAY_Q_REVERT_DECIMALS = 4


def _lethargy_integral(e, xs):
    """Trapezoidal integral of the lethargy-weighted cross section (int sigma/E dE).

    ``sigma/E`` is evaluated with a guarded divide so a non-positive grid edge
    (never expected for a physical energy grid) contributes zero rather than a
    NaN/inf.
    """
    leth = np.divide(xs, e, out=np.zeros_like(xs, dtype=float), where=e > 0.0)
    return float(_TRAPEZOID(leth, e))


def _band_ratio(e, total, part, lo, hi, inclusive_hi):
    """Lethargy-weighted ``int part/E dE`` / ``int total/E dE`` over one band.

    Selects the grid points falling in ``[lo, hi)`` (or ``[lo, hi]`` when
    ``inclusive_hi``). Returns ``(ratio, part_nonzero_but_total_zero)``:
    ``ratio`` is ``None`` when the band holds fewer than 2 grid points, its
    total integral is zero, or the band fails the significance floor below; the
    second flag is ``True`` only in the degenerate case where the partials
    integrate to something while the total is zero (a genuine inconsistency the
    caller surfaces in the row notes).

    Significance floor: a band whose MF=3 total never rises above
    ``CONSISTENCY_ABS_FLOOR`` (``max(total_in_band) < CONSISTENCY_ABS_FLOOR``)
    is treated as undefined (``None``, no flag), same as the <2-points /
    zero-integral cases. Such a band is all below-threshold evaluator dust of a
    threshold reaction; its sparse points yield spurious ratios (~2.0 from a
    two-point trapezoid over near-zero values), which accounted for ~2/3 of a
    full-library JEFF band-reject set as false positives on sigma < 1e-9 b
    reactions.
    """
    sel = (e >= lo) & (e <= hi) if inclusive_hi else (e >= lo) & (e < hi)
    if int(np.count_nonzero(sel)) < 2:
        return None, False
    eb, tb, pb = e[sel], total[sel], part[sel]
    if float(np.max(tb)) < CONSISTENCY_ABS_FLOOR:
        return None, False
    int_total = _lethargy_integral(eb, tb)
    if int_total == 0.0:
        int_part = _lethargy_integral(eb, pb)
        return None, (int_part != 0.0)
    return _lethargy_integral(eb, pb) / int_total, False


def _audit_reaction(source, parent, mt, partials, emax=2.0e7):
    """Pointwise consistency of a reaction's MF=10 partials against its MF=3 total.

    Every MF=10 partial (ground + metastable) is interpolated lin-lin onto the
    MF=3 energy grid (``np.interp(..., left=0, right=0)`` -- PENDF is lin-lin;
    outside a partial's tabulated range it contributes nothing) and summed. The
    summed partials are compared to the MF=3 total with the same
    :func:`_partials_total_max_deviation` used by the collapse, so the audit and
    the runtime warning share one definition of "consistent" (including the
    ``CONSISTENCY_ABS_FLOOR`` both-sides floor-dust exemption).

    The MF=3 grid is truncated to ``E <= emax`` BEFORE everything (worst-dev
    scan, E-at-max, values, and all integrals). This suppresses the known
    policy artifact where MF=10 partials legitimately end near 30 MeV (TENDL
    lumping into MT=5) while the MF=3 total runs to 200 MeV: over the full tape
    range nearly every reaction shows ``dev=1.0`` somewhere and a plain
    ``int sigma dE`` ratio is dominated by the >30 MeV tail. ``emax`` defaults
    to the 20 MeV application group-structure cap.

    Returns ``None`` when the MF=3 total is unavailable (nothing to compare
    against) or fewer than 2 grid points survive the cap, else a dict with
    ``worst_dev`` (max relative deviation), ``energy`` (eV at that group;
    ``None`` when only floor dust qualified), ``sum_partials``/``total`` (barn
    there), ``integral_ratio`` (full-range, capped, LETHARGY-weighted
    ``int Sum(partials)/E dE`` / ``int total/E dE`` -- in v1 this was an
    unweighted ``dE`` ratio), the four per-band lethargy ratios
    ``ratio_thermal``/``ratio_epithermal``/``ratio_intermediate``/``ratio_fast``
    (``None`` for a band
    with <2 grid points, zero total integral, or a below-threshold total whose
    ``max < CONSISTENCY_ABS_FLOOR`` -- see :func:`_band_ratio`), and ``notes``
    (a string flagging any band whose partials integrate nonzero against a zero
    total).
    """
    try:
        mf3_e, mf3_xs = source.total_xs(parent, mt)
    except Exception:
        return None
    mf3_e = np.asarray(mf3_e, dtype=float)
    mf3_xs = np.asarray(mf3_xs, dtype=float)
    if mf3_e.size == 0:
        return None

    # Cap the grid at emax before anything else.
    keep = mf3_e <= emax
    mf3_e = mf3_e[keep]
    mf3_xs = mf3_xs[keep]
    if mf3_e.size < 2:
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

    int_total = _lethargy_integral(mf3_e, mf3_xs)
    int_part = _lethargy_integral(mf3_e, part_sum)
    ratio = (int_part / int_total) if int_total != 0.0 else float('inf')

    ratio_thermal, flag_th = _band_ratio(
        mf3_e, mf3_xs, part_sum, 0.0, _BAND_THERMAL_HI, False)
    ratio_epithermal, flag_ep = _band_ratio(
        mf3_e, mf3_xs, part_sum, _BAND_THERMAL_HI, _BAND_EPITHERMAL_HI, False)
    ratio_intermediate, flag_in = _band_ratio(
        mf3_e, mf3_xs, part_sum, _BAND_EPITHERMAL_HI, _BAND_INTERMEDIATE_HI,
        False)
    ratio_fast, flag_fa = _band_ratio(
        mf3_e, mf3_xs, part_sum, _BAND_INTERMEDIATE_HI, emax, True)

    flagged = [name for name, flag in
               (('thermal', flag_th), ('epithermal', flag_ep),
                ('intermediate', flag_in), ('fast', flag_fa))
               if flag]
    notes = (f"partials nonzero vs zero total in {', '.join(flagged)}"
             if flagged else '')

    return dict(
        worst_dev=worst,
        energy=(float(mf3_e[idx]) if idx >= 0 else None),
        sum_partials=(float(part_sum[idx]) if idx >= 0 else None),
        total=(float(mf3_xs[idx]) if idx >= 0 else None),
        integral_ratio=ratio,
        ratio_thermal=ratio_thermal,
        ratio_epithermal=ratio_epithermal,
        ratio_intermediate=ratio_intermediate,
        ratio_fast=ratio_fast,
        notes=notes)


def _append_note(row, marker):
    """Append ``marker`` to a log row's ``notes`` (semicolon-separated).

    Serves both the audit rows and the per-parent mapping rows (which carry no
    ``notes`` key until one is needed). Append-only: the notes field is the last
    column of either table, so extra markers never disturb the column layout of
    the rows before it.
    """
    row['notes'] = f"{row['notes']}; {marker}" if row.get('notes') else marker


# =============================================================================
# MF=10-only emission bookkeeping (--emit-mf10-only-reactions)
# =============================================================================

# Terminal buckets of one examined MF=10-only MT. Every examined reaction lands
# in exactly one of these or in an ``emitted_*`` counter, which is what makes
# the reconciliation identity (examined = emitted + Sum(skips)) a real check.
_MF10_ONLY_SKIPS = (
    ('skipped_no_name',              'MT maps to no chain reaction'),
    ('skipped_parent_not_in_chain',  'parent not in chain'),
    ('skipped_no_target',            'ground target not in chain'),
    ('skipped_already_present',      'reaction type already in chain'),
    ('skipped_superseded',           'superseded by another MT'),
    ('skipped_no_mapped_metastable', 'no metastable mapped into the chain'),
    ('skipped_rejected',             'audit-rejected (left stock)'),
    ('skipped_no_ground_partial',    'no usable LFS=0 partial'),
)

_MF10_ONLY_OUTCOMES = dict(
    [('emitted_ground_only', 'EMITTED plain (ground-only)'),
     ('emitted_branched', 'EMITTED branched (isomeric_branching)')]
    + [(key, f'skipped: {label}') for key, label in _MF10_ONLY_SKIPS])


def _new_mf10_only_book():
    """Fresh counter/record book for the MF=10-only emission pass."""
    book = dict(examined=0, emitted_ground_only=0, emitted_branched=0, rows=[])
    for key, _label in _MF10_ONLY_SKIPS:
        book[key] = 0
    return book


def _mf10_only_open_row(book, parent, mt, r_name, partials):
    """Open (and count) one examined MF=10-only reaction's log row."""
    lfs = sorted(int(p['lfs']) for p in partials)
    products = []
    for p in partials:
        izap = int(p['izap'])
        product = _safe_gnds_name(izap // 1000, izap % 1000)
        if product not in products:
            products.append(product)
    shape = ('ground-only' if all(v == 0 for v in lfs)
             else ('g+m' if 0 in lfs else 'm-only'))
    row = dict(parent=parent, mt=mt, reaction=r_name, lfs=lfs,
               products=products, shape=shape, target='-', q=None,
               outcome='(pending)')
    book['examined'] += 1
    book['rows'].append(row)
    return row


def _mf10_only_note(book, row, key, target=None, q=None, note=None):
    """Terminate one examined MF=10-only reaction in bucket ``key``.

    ``note`` is a short parenthetical appended to the row's outcome text (e.g.
    the MT that superseded this one); it never changes the bucket counted.
    """
    if book is None:
        return
    book[key] = book.get(key, 0) + 1
    if row is not None:
        row['outcome'] = _MF10_ONLY_OUTCOMES.get(key, key)
        if note:
            row['outcome'] = f"{row['outcome']} ({note})"
        if target is not None:
            row['target'] = target
        if q is not None:
            row['q'] = q


def _mf10_only_row_for(book, parent, mt):
    """The open row of ``(parent, mt)``, or ``None``."""
    for row in book['rows']:
        if row['parent'] == parent and row['mt'] == mt:
            return row
    return None


def _mf10_only_reconciliation(book):
    """``(examined, emitted, skipped)`` of the MF=10-only emission pass."""
    emitted = book['emitted_ground_only'] + book['emitted_branched']
    skipped = sum(book[key] for key, _label in _MF10_ONLY_SKIPS)
    return book['examined'], emitted, skipped


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
            product = _safe_gnds_name(z, a, liso)
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
            product = _safe_gnds_name(zp, ap, position)
            rec.update(liso=position, product=product)
            rec['bucket'] = ('matched' if product in chain_names
                             else 'product_not_in_chain')
        records.append(rec)
    return records


def map_library(source, chain, decay_lookup, mode, rtol, atol, verbose=True,
                reject_rtol=None, audit_emax=2.0e7, reject_band_ratio=None):
    """Map every PENDF nuclide's MF=10 metastable partials against the chain.

    Returns ``(branching, stats)`` where ``branching`` is
    ``{parent: {r_name: {'mt', 'ground', 'metastables', 'parent_elis'}}}``
    restricted to the matched-and-in-chain pathways, and ``stats`` carries the
    counters and the per-parent log records.

    Every reaction carrying >=1 metastable pathway is run through the pointwise
    MF=10-vs-MF=3 consistency audit (:func:`_audit_reaction`, capped at
    ``audit_emax`` eV) regardless of the rejection gates. A reaction is left
    stock -- no ``<isomeric_branching>`` is created, and it is recorded in
    ``stats['rejected']`` with a ``criterion`` -- when EITHER gate fires:

    * ``reject_rtol`` is set and the audit max relative deviation exceeds it, OR
    * ``reject_band_ratio`` is set and any DEFINED lethargy band ratio has
      ``ratio - 1 > reject_band_ratio`` -- ONE-SIDED, over-summing only
      (``None`` band ratios never trigger; under-summing never rejects, since
      deep silence is the collapse silence-fill's job and a live under-sum is
      Class-4 source-faithful by policy).

    Self-loop-ground exemption: a reaction whose GROUND product is the parent
    itself (see :func:`_self_loop_ground`) is EXEMPT from BOTH gates -- its
    ground pathway is a transmutation-matrix self-loop no-op, so partial-sum
    incompleteness cannot affect the chain and rejecting would only destroy
    valid metastable isomer production. The rtol gate needs the exemption most:
    a metastable-only self-loop (the In113/In115 ``(n,n')`` class) has NO ground
    partial to cover the energies below its metastable threshold, so
    ``worst_dev`` is pinned at exactly 1.0 there and ANY ``reject_rtol`` below
    1.0 would reject the whole class and destroy its m1 branching. A suppressed
    rejection marks the audit row's ``notes`` (``self-loop ground: rtol exempt``
    / ``self-loop ground: band-reject exempt``) and is counted in
    ``stats['rtol_reject_exempt']`` / ``stats['band_reject_exempt']``.

    Metastable-only observability: a reaction whose MF=10 carries >=1 metastable
    partial but NO LFS=0 (the In113/In115 MT=4 class -- a stable ground product
    means the evaluation legitimately tabulates no ground partial) is counted in
    ``stats['metastable_only']`` and, when auditable, marked
    ``metastable-only MF=10 (no LFS=0)`` in its audit row's ``notes``. Purely a
    report: no rejection or decoration decision depends on it.

    Sections the SOURCE excluded as undecorable -- an MF=10 with no MF=3 sibling
    that the source is not serving -- are carried through to
    ``stats['mf10_without_mf3']`` / ``['mf10_without_mf3_list']`` for the console
    and log so the class stays greppable; they never reach the counters above.

    When the source serves that class (``--emit-mf10-only-reactions``; both
    adapters expose it as ``source.emit_mf10_only``), its reactions arrive marked
    ``mf3_less`` and are decorated like any other -- a ground-only one is queued
    with an empty metastable set so :func:`decorate_chain` emits a plain
    reaction. Every examined MF=10-only MT is booked in ``stats['mf10_only']``,
    which reconciles as ``examined = emitted + Sum(skips)``.
    """
    chain_names = set(chain.nuclide_dict)
    branching = defaultdict(dict)

    # MF=10-only emission book (empty and unreported unless the flag is on).
    mf10_only_enabled = bool(getattr(source, 'emit_mf10_only', False))
    mf10_only = _new_mf10_only_book()

    isomer_mappings = []          # matched-in-chain rows (per-parent tables)
    elis_errors = []              # rtol_exceeded / no_dk / zero_elis
    products_not_in_chain = []    # matched but target absent from chain
    duplicate_errors = []         # kept-closest, others discarded
    lfs_order_dropped = []
    lfs_sentinels = []            # report-only: classified partials whose LFS
                                  # is a library "unspecified level" sentinel

    audit_offenders = []          # reactions with worst_dev > CONSISTENCY_RTOL
    audit_clean = 0               # auditable reactions within CONSISTENCY_RTOL
    rejected = []                 # audit-rejected (left stock) when flag set
    band_reject_exempt = 0        # self-loop-ground reactions spared band reject
    rtol_reject_exempt = 0        # self-loop-ground reactions spared rtol reject
    absent_status = {}            # base GNDS name -> lookup_liso status (no_dk)

    nuclides_with_branching = set()
    total_lfs = 0
    counts = Counter()
    ground_only = 0
    metastable_only = 0           # MF=10 with metastable partial(s) but no LFS=0

    for parent in source.nuclides:
        try:
            reactions = source.reactions(parent)
        except Exception as exc:
            if verbose:
                print(f"  WARNING: {parent}: {exc}", file=sys.stderr)
            continue
        z, a, _ = zam(parent)
        parent_in_chain = parent in chain_names
        # The parent's OWN excitation energy (MF=1/451 ELIS, eV), carried on
        # every branching record: decorate_chain needs it as the Q of a
        # synthesized metastable-parent (n,n') ground (In115_m1 -> In115), and
        # source data is the only admissible origin for it.
        parent_elis = _nuclide_elis(source, parent)

        for mt, rxinfo in sorted(reactions.items()):
            r_name = _MT_TO_NAME.get(mt)
            partials = rxinfo['partials']
            if not partials:
                continue
            # An MF=10-only reaction (no MF=3 sibling, total = Sum(MF=10)) is
            # visible only under --emit-mf10-only-reactions; when it is, every
            # such MT is booked so the emission pass reconciles.
            mf3_less = bool(rxinfo.get('mf3_less'))
            if r_name is None:
                # MT not a depletion reaction (e.g. MT=5 lumped): invisible to
                # the chain, but still booked when it reached us as MF=10-only.
                if mf3_less:
                    _mf10_only_note(
                        mf10_only,
                        _mf10_only_open_row(mf10_only, parent, mt, f'MT{mt}',
                                            partials),
                        'skipped_no_name')
                continue
            mf10_row = (_mf10_only_open_row(mf10_only, parent, mt, r_name,
                                            partials) if mf3_less else None)
            metastables = [p for p in partials if p['lfs'] != 0]
            if not metastables:
                ground_only += 1
                if mf3_less:
                    # Ground-only MF=10-only MT (the Cr (n,3n) class): queued
                    # with an empty metastable set, which decorate_chain emits
                    # as a PLAIN <reaction type Q target> -- no branching child.
                    ground = next((p for p in partials if p['lfs'] == 0), None)
                    if ground is None:
                        _mf10_only_note(mf10_only, mf10_row,
                                        'skipped_no_ground_partial')
                    elif not parent_in_chain:
                        _mf10_only_note(mf10_only, mf10_row,
                                        'skipped_parent_not_in_chain')
                    elif r_name in branching[parent]:
                        # Another MT of this parent already claimed the reaction
                        # name (e.g. MT=103 and MT=600 are both '(n,p)').
                        _mf10_only_note(mf10_only, mf10_row,
                                        'skipped_already_present')
                    else:
                        branching[parent][r_name] = dict(
                            mt=mt, ground=ground, qm=rxinfo['qm'],
                            qi=rxinfo['qi'], metastables=[],
                            parent_elis=parent_elis, mf3_less=True)
                continue

            total_lfs += len(metastables)

            # Metastable-only MF=10: >=1 metastable partial but NO LFS=0 -- the
            # In113/In115 MT=4 class, where the ground product is stable so the
            # evaluation legitimately tabulates no ground partial. Observability
            # only (no decision depends on it): counted for every candidate
            # reaction here, and marked on the audit row below so the class is
            # greppable in the log instead of surfacing only when a gate fires.
            metastable_only_rxn = not any(p['lfs'] == 0 for p in partials)
            if metastable_only_rxn:
                metastable_only += 1

            # Pointwise MF=10-vs-MF=3 consistency audit. Always runs (the
            # decoration candidates are exactly the reactions with >=1
            # metastable pathway); rejection is a separate, opt-in gate below.
            audit = _audit_reaction(source, parent, mt, partials,
                                    emax=audit_emax)
            reject_this = False
            if audit is not None:
                if metastable_only_rxn:
                    _append_note(audit, 'metastable-only MF=10 (no LFS=0)')

                # Rejection gates: worst-dev rtol OR any defined band ratio.
                rtol_fired = (reject_rtol is not None
                              and audit['worst_dev'] > reject_rtol)
                band_fired = []
                if reject_band_ratio is not None:
                    for key, band in (('ratio_thermal', 'thermal'),
                                      ('ratio_epithermal', 'epithermal'),
                                      ('ratio_intermediate', 'intermediate'),
                                      ('ratio_fast', 'fast')):
                        r = audit.get(key)
                        # One-sided: reject only OVER-summing bands (partials
                        # exceed the total -- MF=10 corruption, e.g. the 0.00859
                        # eV spike class). Under-summing never rejects: deep
                        # silence is the collapse silence-fill's job and a live
                        # under-sum is Class-4 source-faithful by policy.
                        if r is not None and (r - 1.0) > reject_band_ratio:
                            band_fired.append(f'band_ratio:{band}')

                # Self-loop-ground exemption: when the reaction's ground pathway
                # returns to the parent itself, its ground route is a
                # transmutation-matrix no-op, so partial-sum incompleteness
                # cannot affect the chain -- only the metastable partials carry
                # real isomer production. Both gates are suppressed. The rtol
                # gate matters here because a metastable-only self-loop has no
                # ground partial below its metastable threshold, pinning
                # worst_dev at exactly 1.0, so ANY reject_rtol < 1.0 would take
                # out the whole In113/In115 (n,n') class. A metastable parent's
                # ground route (e.g. In115_m1 -> In115) is a real transition,
                # NOT a self-loop, and stays rejectable (see _self_loop_ground).
                # Evaluated at most once per reaction, and only if a gate fired.
                self_loop = False
                if rtol_fired or band_fired:
                    self_loop = _self_loop_ground(parent, r_name, partials,
                                                  chain)

                fired = []
                if rtol_fired:
                    if self_loop:
                        rtol_reject_exempt += 1
                        _append_note(audit, 'self-loop ground: rtol exempt')
                    else:
                        fired.append('worst_dev')
                if band_fired and self_loop:
                    # The exemption spares a rejection that the band criterion
                    # would otherwise have fired: mark the audit row and count
                    # it. `fired` is necessarily empty here -- a self-loop's
                    # worst_dev is exempted above, so it never fills `fired`.
                    band_reject_exempt += 1
                    _append_note(audit, 'self-loop ground: band-reject exempt')
                else:
                    fired.extend(band_fired)

                if audit['worst_dev'] > CONSISTENCY_RTOL:
                    audit_offenders.append(dict(
                        parent=parent, mt=mt, reaction=r_name, **audit))
                else:
                    audit_clean += 1

                if fired:
                    reject_this = True
                    rejected.append(dict(
                        parent=parent, mt=mt, reaction=r_name,
                        threshold=reject_rtol, band_threshold=reject_band_ratio,
                        criterion=', '.join(fired), **audit))

            records = _classify_metastables(
                parent, mt, r_name, metastables, decay_lookup,
                chain_names, mode, rtol, atol)

            # Report-only LFS sentinel scan: flag any classified partial whose
            # LFS is a library "unspecified level" sentinel (99 / 40). This
            # runs regardless of the mapping outcome or the rejection gates and
            # never alters a decision -- it only records the occurrence so a
            # sentinel LFS is never silently consumed as an isomer ordinal.
            for rec in records:
                if rec['lfs'] not in SENTINEL_LFS:
                    continue
                zr, ar = rec['z'], rec['a']
                base = (gnds_name(zr, ar, 0) if zr in ATOMIC_SYMBOL
                        else f"Z{zr}-A{ar}")
                outcome = _SENTINEL_OUTCOME.get(rec['bucket'], rec['bucket'])
                prod = rec.get('product')
                if prod and rec['bucket'] in ('matched', 'product_not_in_chain'):
                    outcome = f"{outcome} -> {prod}"
                lfs_sentinels.append(dict(
                    parent=parent, mt=mt, reaction=r_name, lfs=rec['lfs'],
                    description=SENTINEL_LFS[rec['lfs']],
                    convention=SENTINEL_LFS[rec['lfs']].split(':')[0],
                    target_z=zr, target_a=ar, base_nuclide=base,
                    elfs=rec['elfs'], bucket=rec['bucket'], product=prod,
                    method=('lfs_order' if mode == 'lfs_order' else 'elis'),
                    outcome=outcome))

            mapped = []           # matched + in chain
            dup_by_liso = defaultdict(list)
            for rec in records:
                bucket = rec['bucket']
                if bucket != 'matched':
                    counts[bucket] += 1
                if bucket == 'matched':
                    hl = _half_life(chain, rec['product'])
                    mrow = dict(parent=parent, reaction=r_name, mt=mt,
                                product=rec['product'], lfs=rec['lfs'],
                                liso=rec['liso'], elis=rec['elfs'],
                                dk_elis=rec['dk_elis'],
                                method=('lfs_order' if mode == 'lfs_order'
                                        else 'elis'),
                                half_life=hl if hl is not None else 'stable',
                                target_z=rec['z'], target_a=rec['a'])
                    # A matched pathway on an audit-rejected reaction is left
                    # stock (no <isomeric_branching>): it is booked separately,
                    # NOT credited as ELIS-matched, and its parent gains no
                    # branching. The row is still listed -- tagged REJECTED, to
                    # cross-reference the MF=10 REJECTED REACTIONS section.
                    if reject_this:
                        counts['matched_rejected'] += 1
                        mrow['notes'] = 'REJECTED (band)'
                    else:
                        counts['matched'] += 1
                        nuclides_with_branching.add(parent)
                    isomer_mappings.append(mrow)
                    mapped.append(rec)
                elif bucket == 'product_not_in_chain':
                    nuclides_with_branching.add(parent)
                    products_not_in_chain.append(dict(
                        type='no_metastable_decay_data', parent=parent, mt=mt,
                        reaction=r_name, lfs=rec['lfs'], elis=rec['elfs'],
                        base_nuclide=_safe_gnds_name(rec['z'], rec['a']),
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
                        base_nuclide=_safe_gnds_name(rec['z'], rec['a']),
                        target_z=rec['z'], target_a=rec['a'], omitted=True))
                elif bucket == 'zero_elis':
                    nuclides_with_branching.add(parent)
                    elis_errors.append(dict(
                        type='zero_elis_metastables', parent=parent, mt=mt,
                        reaction=r_name, lfs=rec['lfs'], elis=rec['elfs'],
                        base_nuclide=_safe_gnds_name(rec['z'], rec['a']),
                        target_z=rec['z'], target_a=rec['a'],
                        skipped_states=rec.get('skipped_states', [])))
                elif bucket == 'no_dk':
                    nuclides_with_branching.add(parent)
                    base_nuc = _safe_gnds_name(rec['z'], rec['a'])
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
                base_nuc = _safe_gnds_name(dups[0]['z'], dups[0]['a'])
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
                prev = branching[parent].get(r_name)
                if prev is not None and prev.get('mf3_less'):
                    # This reaction name was already queued by ANOTHER MF=10-only
                    # MT of the same parent (several MTs share one chain reaction
                    # name -- MT=16 and MT=875..891 are all '(n,2n)'). The
                    # newcomer overwrites it below (last wins, the pre-existing
                    # idiom for name-colliding MTs); terminate the DISPLACED
                    # entry's row here -- whatever its shape -- so the emission
                    # reconciliation identity stays exact instead of leaving a
                    # '(pending)' row behind.
                    _mf10_only_note(
                        mf10_only,
                        _mf10_only_row_for(mf10_only, parent, prev['mt']),
                        'skipped_superseded', note=f'MT={mt}')
                branching[parent][r_name] = dict(
                    mt=mt, ground=ground, qm=rxinfo['qm'], qi=rxinfo['qi'],
                    metastables=mapped, parent_elis=parent_elis,
                    mf3_less=mf3_less)
            elif not mapped:
                # MF=10 metastables present but none mapped into the chain:
                # the base reaction is left stock.
                ground_only += 1
                if mf3_less:
                    _mf10_only_note(mf10_only, mf10_row,
                                    'skipped_no_mapped_metastable')
            elif mf3_less and not parent_in_chain:
                _mf10_only_note(mf10_only, mf10_row,
                                'skipped_parent_not_in_chain')
            elif mf3_less:
                # Mapped and in chain, so the audit gate is what stopped it.
                _mf10_only_note(mf10_only, mf10_row, 'skipped_rejected')

    # Sections the source excluded as undecorable (MF=10 with no MF=3). Both
    # adapters report the class: the tape adapter fills its list eagerly while
    # parsing the tapes, the h5 adapter lazily as ``reactions()`` walks each
    # nuclide's stamped ``total_source='sum-mf10'`` groups -- so the snapshot is
    # taken AFTER the loop above, never before it.
    mf10_without_mf3 = list(getattr(source, 'mf10_without_mf3', []))

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
        matched_rejected=counts['matched_rejected'],
        products_not_in_chain=counts['product_not_in_chain'],
        rtol_exceeded=counts['rtol_exceeded'],
        no_dk=counts['no_dk'],
        zero_elis=counts['zero_elis'],
        duplicate_discarded=counts['duplicate'],
        duplicate_resolved=len(duplicate_errors),
        lfs_order_dropped=counts['lfs_order_dropped'],
        ground_only=ground_only,
        metastable_only=metastable_only,
        mf10_without_mf3=len(mf10_without_mf3),
        mf10_without_mf3_list=mf10_without_mf3,
        # True only for an h5 built WITH MF=10-only totals (root attr present):
        # such a library can carry the class, which the log section says. An
        # older h5 -- like a tape run -- keeps the original wording.
        mf10_without_mf3_from_h5=(
            getattr(source, 'kind', None) == 'h5'
            and getattr(source, 'mf10_only_totals', None) is not None),
        # MF=10-only emission (--emit-mf10-only-reactions); the counters and the
        # log section are reported ONLY when the flag is on, so a flag-off run's
        # console and log stay byte-identical to a run without the feature.
        mf10_only_enabled=mf10_only_enabled,
        mf10_only=mf10_only,
        isomer_mappings=isomer_mappings,
        elis_errors=elis_errors,
        products_not_in_chain_errors=products_not_in_chain,
        duplicate_errors=duplicate_errors,
        lfs_order_dropped_list=lfs_order_dropped,
        # MF=10 consistency audit + threshold-gated rejection + decay-gap log.
        reject_rtol=reject_rtol,
        reject_band_ratio=reject_band_ratio,
        audit_emax=audit_emax,
        audit_offenders=len(audit_offenders),
        audit_clean=audit_clean,
        audit_offenders_list=audit_offenders,
        rejected_count=len(rejected),
        rejected=rejected,
        band_reject_exempt=band_reject_exempt,
        rtol_reject_exempt=rtol_reject_exempt,
        lfs_sentinels=lfs_sentinels,
        lfs_sentinel_count=len(lfs_sentinels),
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

def _note_mg_ground(stats, parent, r_name, marker, skipped=False):
    """Count + log one metastable-parent ``(n,n')`` ground synthesis or skip.

    The counters (``mg_ground_synthesized`` / ``mg_ground_skipped``) print
    unconditionally in the console and log summary blocks; the marker is
    appended to every mapping-log row of ``parent``'s reaction ``r_name``, so a
    synthesized -- or refused -- m->g ground is visible in the Notes column next
    to the pathways it belongs to, not only in the summary. No-op when
    :func:`decorate_chain` is called without a ``stats`` sink.
    """
    if stats is None:
        return
    key = 'mg_ground_skipped' if skipped else 'mg_ground_synthesized'
    stats[key] = stats.get(key, 0) + 1
    for row in stats.get('isomer_mappings', []):
        if row.get('parent') == parent and row.get('reaction') == r_name:
            _append_note(row, marker)


# Per-row marker for a gate-refused reaction. The core string matches the
# mapping-log section title (and the GENDF patcher's) so one grep finds both.
_PATHWAY_Q_REJECT_MARKER = 'PATHWAY-Q FILE-QM REJECTED'


def _pathway_q_file_ground_qm(info):
    """The file's own ground QM for the pathway-Q gate, or ``None``.

    The probe is the LFS=0 partial's OWN QM -- the same subsection whose QI the
    metastable partials are offset from. A reaction with no LFS=0 partial (the
    metastable-only class) has no such value, so a mapped metastable's QM stands
    in, reconstructed as ``QI + ELFS``: the classified records carry no ``qm``
    key, and ELFS is defined as ``QM - QI``, so the sum is exact.

    The MF=3 head QM (``info['qm']``) is NEVER the probe: it is a different
    quantity, and comparing the file against its own MF=3 head would validate
    nothing about the MF=10 absolute scale the gate exists to check.
    """
    ground = info.get('ground')
    if ground is not None and ground.get('qm') is not None:
        return float(ground['qm'])
    for m in info.get('metastables') or []:
        if m.get('qi') is not None and m.get('elfs') is not None:
            return float(m['qi']) + float(m['elfs'])
    return None


def _note_pathway_q_reject(stats, parent, r_name, record):
    """Count + log one pathway-Q gate refusal.

    ``record['q_kept']`` holds one entry per member, so the slot counter credits
    the METASTABLE slots only -- the ground slot is chain-anchored by
    construction and does not move. The reaction count is ``len`` of the record
    list, which is the number the GENDF patcher reports for the same gate. No-op
    when :func:`decorate_chain` is called without a ``stats`` sink.
    """
    if stats is None:
        return
    stats.setdefault('pathway_q_rejected', []).append(record)
    stats['q_chain_anchored'] = (stats.get('q_chain_anchored', 0)
                                 + record['reverted_slots'])
    for row in stats.get('isomer_mappings', []):
        if row.get('parent') == parent and row.get('reaction') == r_name:
            _append_note(row, _PATHWAY_Q_REJECT_MARKER)


def _emit_mf10_only_ground(nuc, chain, z, a, r_name, info, book, row):
    """Append the PLAIN ground reaction of a ground-only MF=10-only MT.

    The Cr ``(n,3n)`` class: an MT the library serves with a total synthesized
    from MF=10 partials, whose only partial is LFS=0. It carries no isomeric
    branching, so it is appended as a stock ``<reaction type Q target>`` --
    ``pendf_lfs`` left ``None`` so the writer never folds it into an
    ``<isomeric_branching>`` element.

    Target resolution follows the existing synthesized-ground path (the
    reaction's ``(dA, dZ)`` shift, :func:`_ground_product`) and Q is the LFS=0
    partial's QI -- section-sourced, never fabricated. Returns 1 when a reaction
    was appended, else 0 (an already-present reaction type or an unresolvable /
    off-chain target is skipped and counted).
    """
    if any(rx.type == r_name for rx in nuc.reactions):
        # Idempotency: re-running the patcher on an already-patched chain, or a
        # base chain that harvested this MT from another source.
        _mf10_only_note(book, row, 'skipped_already_present')
        return 0
    daughter = _ground_product(z, a, r_name)
    if daughter is None or daughter not in chain.nuclide_dict:
        _mf10_only_note(book, row, 'skipped_no_target',
                        target=(daughter or '-'))
        return 0
    q = float(info['ground']['qi'])
    nuc.add_reaction(r_name, daughter, q, 1.0)
    _mf10_only_note(book, row, 'emitted_ground_only', target=daughter, q=q)
    return 1


def decorate_chain(chain, branching, stats=None):
    """Add ground + qualified metastable ReactionTuples to the chain in place.

    Every tuple carries a ``pendf_lfs`` (ground = 0, metastable = tape LFS) so
    the Phase-1 refold writer folds the group into an ``<isomeric_branching>``
    element instead of falling back to stock reaction elements. Returns the
    number of reactions synthesized for MTs absent from the base chain.

    ``stats`` (the :func:`map_library` dict) is an optional sink for the
    metastable-parent ``(n,n')`` ground bookkeeping -- see
    :func:`_note_mg_ground` -- for the pathway-Q gate ledger (see
    :func:`_note_pathway_q_reject`) and for the MF=10-only emission counters
    (``stats['mf10_only']``). The decoration itself never depends on it: the
    pathway-Q revert happens with or without a stats sink, only its record does
    not.

    A reaction marked ``mf3_less`` (served only under
    ``--emit-mf10-only-reactions``) takes one of three shapes:

    * **ground-only** -- no metastable pathway at all: a PLAIN
      ``<reaction type Q target>`` is appended (``pendf_lfs`` unset, so the
      writer never folds it), with the target from the reaction's ``(dA, dZ)``
      shift and Q from the LFS=0 partial's QI.
    * **ground + metastable(s)** -- the ordinary added-reaction path, unchanged;
      its Q values are now MF=10-section-sourced rather than MF=3-sourced.
    * **metastable-only** -- folded with ``ground = None`` (a branched element
      whose members are all metastable): there is neither an LFS=0 partial nor
      an MF=3 total to serve a ground row from, and the metastable partials
      already sum to the reaction's synthesized total. One exception, by
      precedence: the two ``(n,n')`` branches below are taken first, so an
      MF=10-only ``(n,n')`` (MT=4 or a level MT) still receives a synthesized
      ground member -- the m->g transition for a metastable parent, the
      ``target == parent`` self-loop otherwise. Harmless in both cases: a
      self-loop is a depletion-matrix no-op, and the collapse serves that
      ground row by balance as identically zero (the metastable partials
      already sum to the synthesized total).
    """
    if stats is not None:
        stats.setdefault('mg_ground_synthesized', 0)
        stats.setdefault('mg_ground_skipped', 0)
        stats.setdefault('pathway_q_rejected', [])
        stats.setdefault('q_chain_anchored', 0)
    book = stats.get('mf10_only') if stats is not None else None
    rows = ({(r['parent'], r['mt']): r for r in book['rows']}
            if book is not None else {})
    reactions_added = 0
    for parent, reactions in branching.items():
        nuc = chain[parent]
        z, a, _ = zam(parent)
        folded_members = {}
        for r_name, info in reactions.items():
            metastables = info['metastables']
            mf3_less = bool(info.get('mf3_less'))
            row = rows.get((parent, info.get('mt')))
            if not metastables:
                # Ground-only MF=10-only MT: a plain reaction, no branching.
                if mf3_less:
                    reactions_added += _emit_mf10_only_ground(
                        nuc, chain, z, a, r_name, info, book, row)
                continue
            existing = next((rx for rx in nuc.reactions if rx.type == r_name),
                            None)

            # Pathway-Q sanity gate. The metastable slots below transcribe the
            # MF=10 subsection's own ABSOLUTE QI, which nothing validates: the
            # ENDF/B-8.1 (n,alpha) sections are wrong by up to 12 MeV, sign
            # included. The ground slot of a chain-present reaction is the base
            # chain's scalar Q -- AME/evaluation-derived -- so a corrupt file
            # would leave the two halves of one pathway set on DIFFERENT energy
            # zeros. Where the file's own ground QM fails to corroborate that
            # scalar, the metastable slots revert to the chain-anchored
            # ``Q_chain - ELFS``, which consumes only the QM-QI DIFFERENCE and is
            # immune to an absolute-scale error; the whole set then shares the
            # chain's zero. The refused file values are ledgered and written
            # nowhere. MT=4 is exempt by construction: there the chain scalar is
            # QI(MF=3) = -E(level), a different quantity than QM, so
            # "disagreement" is expected rather than evidence against the file.
            # A reaction absent from the chain has no anchor to check against.
            chain_q = existing.Q if existing is not None else None
            file_ground_qm = _pathway_q_file_ground_qm(info)
            q_chain_reject = (
                info.get('mt') != 4 and chain_q is not None
                and file_ground_qm is not None
                and abs(file_ground_qm - chain_q) > PATHWAY_Q_CHAIN_TOL)

            if existing is not None:
                ground = ReactionTuple(r_name, existing.target, existing.Q,
                                       1.0, 0)
            else:
                # MT absent from base chain -- synthesize the ground pathway.
                if r_name == "(n,n')" and _ISOMER_SUFFIX.search(parent):
                    # METASTABLE parent: the LFS=0 route is super-elastic
                    # de-excitation to the TRUE ground (In115_m1 -> In115), a
                    # real off-diagonal isomer-burnup transition -- NOT the
                    # self-transition ``target == parent`` would encode. Q is
                    # the parent isomer's own excitation energy (+ELIS, eV,
                    # exothermic), which is also the convention the base chains
                    # carry for their stock m->g (n,n') rows (e.g. Co58_m1 ->
                    # Co58, Q = +24950.03 eV). The audit agrees: such a reaction
                    # is never self-loop-exempt (see _self_loop_ground).
                    # ELIS is SOURCE data, never guessed: with no usable ELIS on
                    # the parent, or with its ground absent from the chain,
                    # nothing is synthesized -- the reaction is left stock and
                    # the skip is counted and noted in the mapping log. A
                    # metastable parent tabulated with ELIS = 0 is a broken
                    # MF=1/451 header, not a zero-energy isomer, and counts as
                    # unusable (mirroring the decay side's zero_elis policy) --
                    # Q = 0 here would silently reinstate the self-transition
                    # no-op this branch exists to remove.
                    target = _ISOMER_SUFFIX.sub('', parent)
                    elis = info.get('parent_elis')
                    if not elis or target not in chain.nuclide_dict:
                        reason = ('parent MF=1/451 ELIS absent or 0' if not elis
                                  else f'ground {target} not in chain')
                        _note_mg_ground(
                            stats, parent, r_name,
                            f"m->g (n,n') left stock: {reason}", skipped=True)
                        if mf3_less:
                            _mf10_only_note(book, row, 'skipped_no_target')
                        continue
                    ground = ReactionTuple(r_name, target, float(elis), 1.0, 0)
                    _note_mg_ground(
                        stats, parent, r_name,
                        f"synthesized m->g (n,n') ground: -> {target}, "
                        f"Q=+{float(elis):.1f} eV")
                elif r_name == "(n,n')":
                    ground = ReactionTuple(r_name, parent, 0.0, 1.0, 0)
                elif mf3_less and info['ground'] is None:
                    # METASTABLE-ONLY MF=10-only MT (no LFS=0 partial, and no
                    # MF=3 total either): nothing can serve a ground row, and
                    # the metastable partials already sum to the reaction's
                    # synthesized total. Fold metastable-only rather than
                    # inventing a ground pathway the library cannot serve.
                    ground = None
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
            q_refused = ([] if ground is None
                         else ['n/a (ground already chain-anchored)'])
            for m in metastables:
                if q_chain_reject:
                    q = round(chain_q - float(m['elfs']),
                              _PATHWAY_Q_REVERT_DECIMALS)
                    q_refused.append(float(m['qi']))
                else:
                    q = float(m['qi'])
                members.append(ReactionTuple(
                    f"{r_name}_m{m['liso']}", m['product'], q, 1.0, m['lfs']))
            folded_members[r_name] = members

            if q_chain_reject:
                # Ledgered here and nowhere else: a refused reaction's written Q
                # values are indistinguishable from a clean section's, so
                # without this record a 12 MeV file defect leaves no trace.
                _note_pathway_q_reject(stats, parent, r_name, dict(
                    parent=parent, reaction=r_name, mt=info.get('mt'),
                    targets=[rx.target for rx in members],
                    lfs=[rx.pendf_lfs for rx in members],
                    file_ground_qm=file_ground_qm, chain_q=chain_q,
                    delta=file_ground_qm - chain_q, q_file=q_refused,
                    q_kept=[rx.Q for rx in members],
                    reverted_slots=len(metastables)))
            if mf3_less:
                _mf10_only_note(
                    book, row, 'emitted_branched',
                    target=(ground.target if ground is not None
                            else members[0].target),
                    q=(ground.Q if ground is not None else members[0].Q))

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


def _prune_nn_prime_self_loops(chain):
    """Remove stock ``(n,n')`` ground self-loops (target == parent) in place.

    A stock ``(n,n')`` whose target is EXACTLY the parent nuclide is a
    transmutation-matrix no-op (loss and gain both land on the diagonal and
    cancel), so dropping it changes no depletion result while trimming the
    matrix. The match is EXACT on three conditions, all required:

    * ``rx.type == "(n,n')"`` (never a qualified ``(n,n')_m<n>`` variant), AND
    * ``rx.pendf_lfs is None`` -- the reaction carries no isomeric branching. A
      branched ``(n,n')``'s ground member has ``pendf_lfs == 0`` (part of a
      folded ``<isomeric_branching>`` set) and is NEVER removed; removing it
      would break the fold precondition and flip the collapse to MF=3 fallback,
      AND
    * ``rx.target == nuc.name`` EXACTLY (no base-name stripping). A metastable
      parent's ground route (e.g. ``In115_m1 (n,n') -> In115``) is real isomer
      burnup, an off-diagonal transition, and stays -- its target never equals
      the metastable parent name.

    Returns a list of pruned records ``{'nuclide', 'reaction', 'target'}``. The
    per-nuclide ``reactions=`` XML count is recomputed from ``len(reactions)``
    at export time, so no manual count fix-up is needed here.
    """
    pruned = []
    for nuc in chain.nuclides:
        kept = []
        removed_any = False
        for rx in nuc.reactions:
            if (rx.type == "(n,n')" and rx.pendf_lfs is None
                    and rx.target == nuc.name):
                pruned.append(dict(nuclide=nuc.name, reaction=rx.type,
                                   target=rx.target))
                removed_any = True
            else:
                kept.append(rx)
        if removed_any:
            nuc.reactions = kept

    # Rebuild the top-level reaction-type list (first-appearance order) so the
    # in-memory chain stays consistent, mirroring decorate_chain's final step.
    if pruned:
        chain.reactions = []
        for nuc in chain.nuclides:
            for rx in nuc.reactions:
                if rx.type not in chain.reactions:
                    chain.reactions.append(rx.type)
    return pruned


# =============================================================================
# Console statistics block
# =============================================================================

def print_stats(stats, mode):
    total_lfs = (stats['matched'] + stats['matched_rejected']
                 + stats['products_not_in_chain']
                 + stats['rtol_exceeded'] + stats['no_dk'] + stats['zero_elis']
                 + stats['duplicate_discarded'] + stats['lfs_order_dropped'])
    print(f"\n                      nuclides in PENDF library: {stats['pendf_nuclides_total']:5d}")
    print(f"               nuclides in PENDF with branching: {stats['nuclides_with_branching']:5d}")
    print("-" * 52)
    print(f"                          Total PENDF-LFS found: {total_lfs:5d}")
    if mode == 'elis':
        print(f"                                   ELIS matched: {stats['matched']:5d}")
    else:
        print(f"                               LFS-order mapped: {stats['matched']:5d}")
    if stats['matched_rejected']:
        print(f"         matched but band-rejected (left stock): {stats['matched_rejected']:5d}")
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
    print(f"                          Products not in chain: {stats['products_not_in_chain']:5d}")
    print(f"                       Reactions added to chain: {stats['reactions_added']:5d}")
    print(f"                Synthesized m->g (n,n') grounds: {stats.get('mg_ground_synthesized', 0):5d}")
    print(f"           Skipped m->g (n,n') (no ELIS/ground): {stats.get('mg_ground_skipped', 0):5d}")
    print(f"                    Ground-only MF=10 reactions: {stats['ground_only']:5d}")
    print(f"                Metastable-only MF=10 reactions: {stats['metastable_only']:5d}")
    print(f"             MF=10 without MF=3 (not decorable): {stats.get('mf10_without_mf3', 0):5d}")
    if stats.get('mf10_only_enabled'):
        # Only with --emit-mf10-only-reactions, so a flag-off run's console and
        # log blocks stay byte-identical to a run without the feature.
        book = stats['mf10_only']
        examined, emitted, skipped = _mf10_only_reconciliation(book)
        print(f"               MF=10-only emitted (ground-only): {book['emitted_ground_only']:5d}")
        print(f"                  MF=10-only emitted (branched): {book['emitted_branched']:5d}")
        print(f"                MF=10-only skipped (left stock): {skipped:5d}")
        print(f"          MF=10-only examined (= emitted+skips): {examined:5d}"
              f"   [{emitted} + {skipped}"
              f"{'' if examined == emitted + skipped else ' -- MISMATCH'}]")
    print(f"                          MF=10 audit offenders: {stats['audit_offenders']:5d}")
    print(f"                                 MF=10 rejected: {stats['rejected_count']:5d}")
    print(f"                 Band-reject exempt (self-loop): {stats['band_reject_exempt']:5d}")
    print(f"                 Rtol-reject exempt (self-loop): {stats['rtol_reject_exempt']:5d}")
    print(f"        Pathway-Q file QM rejected (reactions): {len(stats.get('pathway_q_rejected', [])):5d}")
    print(f"         Pathway-Q slots chain-anchored (gate): {stats.get('q_chain_anchored', 0):5d}")
    print(f"                       LFS sentinel occurrences: {stats['lfs_sentinel_count']:5d}")
    print(f"             Unique nuclides absent from DK-Lib: {stats['absent_unique_count']:5d}")


def _print_lfs_sentinel_warning(sentinels, mode):
    """Loud console warning + per-value breakdown when LFS sentinels were seen.

    Report-only: no mapping decision or chain output depends on this. Under
    'elis' mapping the sentinel rows are matched on their real ELFS excitation
    energy and are SAFE; the risk is only a downstream POSITIONAL consumer
    (FISPACT-parity lfs_order tooling) that would mis-name them ``_m99`` /
    ``_m40``. Under 'lfs_order' the tool IS doing positional naming, so the
    warning is emphatic.
    """
    if not sentinels:
        return
    by_val = Counter(s['lfs'] for s in sentinels)
    bar = "!" * 70
    print("\n" + bar)
    print(f"WARNING: {len(sentinels)} LFS SENTINEL occurrence(s) detected "
          "(unspecified-level isomer tags)")
    print(bar)
    for val in sorted(by_val):
        print(f"  LFS={val:<3d} x{by_val[val]:<4d} {SENTINEL_LFS[val]}")
    print("  Sentinel LFS values are NOT level ordinals; they must never be")
    print("  interpreted as isomer ordinals (e.g. _m99 / _m40).")
    if mode == 'lfs_order':
        print("  MODE=lfs_order is ACTIVE: positional product naming is "
              "UNRELIABLE for these")
        print("  rows -- use ELIS mapping for production calculations.")
    else:
        print("  ELIS mapping is active, so these rows are matched on real "
              "excitation energy")
        print("  and are SAFE here; the risk is only if the chain is later "
              "consumed ORDINALLY")
        print("  (e.g. FISPACT-parity lfs_order tooling).")
    print("  See the 'LFS SENTINEL VALUES' section of the mapping log for "
          "every occurrence.")
    print(bar)


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


def _fmt_ratio(ratio):
    """Render a band/full ratio: 'n/a' for None, 'inf' for inf, else 4dp."""
    if ratio is None:
        return "n/a"
    if ratio == float('inf'):
        return "inf"
    return f"{ratio:.4f}"


def _write_consistency_audit_section(f, offenders, audit_clean, emax=2.0e7):
    """MF=10 CONSISTENCY AUDIT section: offenders (worst_dev > rtol) worst-first."""
    f.write("\n\n" + "=" * 220 + "\n")
    f.write("MF=10 CONSISTENCY AUDIT\n")
    f.write("=" * 220 + "\n\n")
    f.write("Pointwise Sum(MF=10 partials) interpolated onto the MF=3 energy "
            f"grid (capped at E <= {emax:.3e} eV), compared against the MF=3 "
            "total, for every reaction carrying >=1 metastable pathway.\n")
    f.write("The cap suppresses the >30 MeV MT=5 lumping artifact (MF=10 "
            "partials stop near 30 MeV while the MF=3 total runs to 200 MeV).\n")
    f.write(f"Groups where BOTH sides sit below {CONSISTENCY_ABS_FLOOR:.0e} b "
            "(evaluator floor dust) are exempt -- their relative deviation is "
            "meaningless.\n")
    f.write("IntRatio and the Thermal/Epithermal/Intermed/Fast ratios are "
            "LETHARGY-weighted (int sigma/E dE) partials/total; bands are "
            f"thermal [grid_min, {_BAND_THERMAL_HI:g} eV) "
            f"({_BAND_THERMAL_HI:g} eV = Cd cutoff), epithermal "
            f"[{_BAND_THERMAL_HI:g} eV, {_BAND_EPITHERMAL_HI:.0e} eV), "
            f"intermediate [{_BAND_EPITHERMAL_HI:.0e} eV, "
            f"{_BAND_INTERMEDIATE_HI:.0e} eV), fast "
            f"[{_BAND_INTERMEDIATE_HI:.0e} eV, {emax:.3e} eV]. "
            "'n/a' = band has <2 grid points, a zero total integral, or a "
            f"below-threshold total (max < {CONSISTENCY_ABS_FLOOR:.0e} b, "
            "evaluator dust -- spurious ratios suppressed).\n")
    f.write(f"Offenders (max rel dev > {CONSISTENCY_RTOL:.0e}) are listed "
            f"worst-first; {audit_clean} audited reaction(s) are clean.\n\n")
    if not offenders:
        f.write("No offenders: all audited reactions agree within "
                f"{CONSISTENCY_RTOL:.0e}.\n")
        return
    f.write(f"Total offenders: {len(offenders)}\n\n")
    header = (f"{'Parent':<12}  {'MT':>5}  {'Reaction':<12}  {'MaxRelDev':>12}  "
              f"{'E[eV]@max':>14}  {'Sum-part[b]':>14}  {'Total[b]':>14}  "
              f"{'IntRatio':>12}  {'Thermal':>10}  {'Epithermal':>10}  "
              f"{'Intermed':>10}  {'Fast':>10}  {'Notes':<40}")
    sep = "-" * 195
    f.write(header + "\n" + sep + "\n")
    for o in sorted(offenders, key=lambda x: x['worst_dev'], reverse=True):
        e = o.get('energy')
        sp = o.get('sum_partials')
        tot = o.get('total')
        e_str = f"{e:.4e}" if e is not None else "-"
        sp_str = f"{sp:.4e}" if sp is not None else "-"
        tot_str = f"{tot:.4e}" if tot is not None else "-"
        f.write(f"{o['parent']:<12}  {o['mt']:>5}  {o['reaction']:<12}  "
                f"{o['worst_dev']:>12.4e}  {e_str:>14}  {sp_str:>14}  "
                f"{tot_str:>14}  {_fmt_ratio(o.get('integral_ratio')):>12}  "
                f"{_fmt_ratio(o.get('ratio_thermal')):>10}  "
                f"{_fmt_ratio(o.get('ratio_epithermal')):>10}  "
                f"{_fmt_ratio(o.get('ratio_intermediate')):>10}  "
                f"{_fmt_ratio(o.get('ratio_fast')):>10}  "
                f"{o.get('notes', ''):<40}\n")


def _write_rejected_section(f, rejected, reject_rtol, reject_band_ratio=None):
    """MF=10 REJECTED REACTIONS section: audit-gated reactions left stock."""
    f.write("\n\n" + "=" * 220 + "\n")
    f.write("MF=10 REJECTED REACTIONS\n")
    f.write("=" * 220 + "\n\n")
    if reject_rtol is None and reject_band_ratio is None:
        f.write("rejection disabled (audit only): neither --mf10-reject-rtol "
                "nor --mf10-reject-band-ratio was set, so no reaction was "
                "rejected on audit grounds.\n")
        return
    rtol_str = (f"{reject_rtol:.3e}" if reject_rtol is not None
                else "off")
    band_str = (f"{reject_band_ratio:.3e}" if reject_band_ratio is not None
                else "off")
    f.write(f"Thresholds: --mf10-reject-rtol = {rtol_str}; "
            f"--mf10-reject-band-ratio = {band_str}\n")
    f.write("Criterion: a reaction is left stock (no <isomeric_branching> "
            "child) when its MF=10-vs-MF=3 audit max relative deviation exceeds "
            "the rtol threshold, OR any DEFINED lethargy band ratio has "
            "ratio-1 exceeding the band threshold (one-sided: over-summing "
            "only; under-summing never rejects).\n")
    f.write("Consequence: MF=3 total routes to the ground target; isomeric "
            "branching discarded.\n\n")
    if not rejected:
        f.write("No reactions exceeded the active threshold(s).\n")
        return
    f.write(f"Total rejected: {len(rejected)}\n\n")
    header = (f"{'Parent':<12}  {'MT':>5}  {'Reaction':<12}  {'MaxRelDev':>12}  "
              f"{'E[eV]@max':>14}  {'Sum-part[b]':>14}  {'Total[b]':>14}  "
              f"{'Criterion':<28}  {'Consequence':<52}")
    sep = "-" * 190
    f.write(header + "\n" + sep + "\n")
    consequence = "left stock: MF=3 total -> ground target; branching discarded"
    for o in sorted(rejected, key=lambda x: x['worst_dev'], reverse=True):
        e = o.get('energy')
        sp = o.get('sum_partials')
        tot = o.get('total')
        e_str = f"{e:.4e}" if e is not None else "-"
        sp_str = f"{sp:.4e}" if sp is not None else "-"
        tot_str = f"{tot:.4e}" if tot is not None else "-"
        crit = o.get('criterion', '-')
        f.write(f"{o['parent']:<12}  {o['mt']:>5}  {o['reaction']:<12}  "
                f"{o['worst_dev']:>12.4e}  {e_str:>14}  {sp_str:>14}  "
                f"{tot_str:>14}  {crit:<28}  {consequence:<52}\n")


def _write_pathway_q_rejected_section(f, records):
    """PATHWAY-Q FILE-QM REJECTED section: reactions whose file Q scale failed.

    A refused reaction is INVISIBLE everywhere else -- it writes exactly the
    values an ungated run would have written -- so the refused file values are
    printed here or not at all.
    """
    f.write("\n\n" + "=" * 220 + "\n")
    f.write("PATHWAY-Q FILE-QM REJECTED (CHAIN-ANCHORED VALUES RETAINED)\n")
    f.write("=" * 220 + "\n\n")
    f.write("Sanity gate: for a non-MT=4 reaction present in the base chain, "
            "the file's own MF=10 ground QM must agree with the chain's scalar "
            f"Q to within {PATHWAY_Q_CHAIN_TOL} eV\n")
    f.write("(PATHWAY_Q_CHAIN_TOL) before ANY of the reaction's per-pathway "
            "file Q values are adopted. The chain scalar is AME/evaluation-"
            "derived; an absolute MF=10 QM is\n")
    f.write("unvalidated and can be badly wrong (ENDF/B-8.1 (n,alpha): 7.9-12.0 "
            "MeV out, sign included). On failure each metastable pathway keeps "
            "Q_chain - ELFS, which\n")
    f.write("consumes only the QM-QI difference and is immune to an absolute-"
            "scale error; the refused file values are listed here and written "
            "nowhere.\n\n")
    f.write("The ground slot is NOT listed as reverted because it is chain-"
            "anchored by construction: a chain-present reaction's ground "
            "pathway already carries the base\n")
    f.write("chain's scalar Q. The refusal therefore restores the property the "
            "gate exists to protect -- every pathway of one reaction sharing "
            "ONE energy zero -- while only\n")
    f.write("the metastable slots move numerically. MT=4 is exempt: there the "
            "chain scalar is QI(MF=3) = -E(level), a different quantity than "
            "QM, so the mismatch is\n")
    f.write("expected rather than evidence against the file. A reaction absent "
            "from the base chain has no anchor and is exempt too.\n\n")
    f.write("On name-colliding MT families ((n,p) = MT 103/600-649, (n,2n) = "
            "MT 16/875-891) the MT below is the surviving level MT while the "
            "chain anchor may come from\n")
    f.write("the family total -- benign, because MF=10 QM is the ground-to-"
            "ground reaction Q and is MT-invariant within a family.\n\n")
    if not records:
        f.write("No reaction's file ground QM diverged from the chain's "
                "scalar Q beyond the tolerance.\n")
        return
    slots = sum(r['reverted_slots'] for r in records)
    f.write(f"Total rejected: {len(records)} reaction(s), {slots} metastable "
            f"slot(s) chain-anchored\n\n")
    header = (f"{'Parent':<12}  {'Reaction':<12}  {'MT':>5}  "
              f"{'File ground QM[eV]':>18}  {'Chain Q[eV]':>16}  "
              f"{'Delta[eV]':>16}  {'Refused -> kept per pathway':<80}")
    sep = "-" * len(header)
    f.write(header + "\n" + sep + "\n")
    for r in sorted(records, key=lambda x: abs(x['delta']), reverse=True):
        pathways = "  ".join(
            f"{t}[LFS={l}]={_fmt_pathway_q(qf)} (kept {_fmt_pathway_q(qk)})"
            for t, l, qf, qk in zip(r['targets'], r['lfs'], r['q_file'],
                                    r['q_kept']))
        f.write(f"{r['parent']:<12}  {r['reaction']:<12}  {r['mt']:>5}  "
                f"{r['file_ground_qm']:>18.4f}  {r['chain_q']:>16.4f}  "
                f"{r['delta']:>+16.4f}  {pathways:<80}\n")


def _fmt_pathway_q(value):
    """Format one pathway-Q ledger entry (float, or the ground slot's text)."""
    return f"{value:.4f}" if isinstance(value, (int, float)) else str(value)


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


def _write_lfs_sentinel_section(f, sentinels):
    """LFS SENTINEL VALUES section: every classified partial with a sentinel LFS.

    Report-only. Lists all occurrences (all nuclides/targets) so a sentinel LFS
    is never silently consumed as an isomer ordinal. Printed unconditionally
    (mirrors the other audit sections), with a 'none found' line when empty.
    """
    f.write("\n\n" + "=" * 220 + "\n")
    f.write('LFS SENTINEL VALUES ("unspecified isomer" conventions)\n')
    f.write("=" * 220 + "\n\n")
    f.write("Some evaluations tag a product whose final-state LEVEL could not "
            "be resolved with a SENTINEL LFS instead of a true level index:\n")
    for val, desc in sorted(SENTINEL_LFS.items()):
        f.write(f"    LFS={val:<3d} = {desc}\n")
    f.write("These sentinels are NOT level ordinals. ELIS mapping (this tool's "
            "default) is unaffected -- it matches the real ELFS excitation "
            "energy, so the CHAIN-Product\n")
    f.write("column carries the correct _m<liso> even though the raw "
            "PENDF-Product column shows the misleading _m<sentinel>. A "
            "positional/lfs_order consumer, however, would mis-name\n")
    f.write("these rows _m99 / _m40. They are reported here so such misuse is "
            "caught; mapping decisions and the output chain are UNCHANGED.\n\n")
    if not sentinels:
        f.write("No LFS sentinel values found.\n")
        return
    by_val = Counter(s['lfs'] for s in sentinels)
    breakdown = ", ".join(f"LFS={v}: {by_val[v]}" for v in sorted(by_val))
    f.write(f"Total sentinel occurrences: {len(sentinels)}  ({breakdown})\n\n")
    header = (f"{'Parent':<12}  {'MT':>5}  {'Reaction':<12}  {'LFS':>4}  "
              f"{'Convention':<22}  {'Target':<12}  {'PENDF-ELFS[eV]':>14}  "
              f"{'Method':<10}  {'Outcome':<44}")
    sep = "-" * len(header)
    f.write(header + "\n" + sep + "\n")
    for s in sorted(sentinels, key=lambda x: (x['parent'], x.get('mt') or 0,
                                              x['lfs'])):
        elfs = s.get('elfs')
        elfs_str = f"{elfs:.1f}" if elfs is not None else "N/A"
        f.write(f"{s['parent']:<12}  {s['mt']:>5}  {s['reaction']:<12}  "
                f"{s['lfs']:>4}  {s.get('convention', ''):<22}  "
                f"{s.get('base_nuclide', '?'):<12}  {elfs_str:>14}  "
                f"{s.get('method', ''):<10}  {s.get('outcome', ''):<44}\n")


def _write_mf10_without_mf3_section(f, records, emit_enabled=False,
                                    from_h5=False):
    """MF=10 WITHOUT MF=3 section: sections excluded as not decorable.

    With ``--emit-mf10-only-reactions`` off (``emit_enabled=False``) and nothing
    recorded from an HDF5 source (``from_h5=False``) the wording is the original
    one, so such a run's log is byte-identical to a run without the feature. The
    two flags select the wording that is true of the run actually made: an h5
    built with MF=10-only totals DOES carry the class (its stamped groups are
    filtered out and recorded here), and with the flag on the class is served
    and reported in the MF=10-ONLY EMISSION section instead.

    Listed unconditionally, like the other audit sections, so the exclusion is
    greppable rather than silent.
    """
    f.write("\n\n" + "=" * 220 + "\n")
    f.write("MF=10 WITHOUT MF=3 (NOT DECORABLE)\n")
    f.write("=" * 220 + "\n\n")
    if emit_enabled or from_h5:
        f.write("Library sections carrying MF=10 isomeric production for an MT "
                "with NO MF=3 total. Such an MT has no total of its own unless "
                "one is synthesized from\n")
        f.write("its partials, which the HDF5 builder does (stamping the "
                "reaction group 'total_source=sum-mf10') and which the chain "
                "patcher consumes only under\n")
        f.write("--emit-mf10-only-reactions. The sections listed here are the "
                "ones this run did NOT decorate: with the flag off that is the "
                "whole class, in either source\n")
        f.write("form (an h5's stamped groups are filtered out exactly as the "
                "tape adapter excludes the same sections, so the two stay "
                "identical); with the flag on only MT=5\n")
        f.write("and MT=18 remain excluded, and the served ones are reported "
                "in the MF=10-ONLY EMISSION section. Only NAMED transmutation "
                "MTs are listed: MT=5 (lumped\n")
        f.write("residual) and MT=18 (fission) map to no chain reaction and "
                "were never candidates.\n\n")
    else:
        f.write("Tape sections carrying MF=10 isomeric production for an MT "
                "with NO MF=3 total. Every PENDF library source form the "
                "collapse can read -- pointwise HDF5,\n")
        f.write("grouped HDF5, and the ASC-tape adapter -- drops such an MT (a "
                "total is never synthesized from the partials), so a chain "
                "reaction decorated from one would\n")
        f.write("collapse to a silent zero row. They are therefore excluded "
                "from the decoration candidates BEFORE any counting above, "
                "making a tape-sourced run identical to an\n")
        f.write("h5-sourced one. Only NAMED transmutation MTs are listed: MT=5 "
                "(lumped residual) and MT=18 (fission) map to no chain reaction "
                "and were never candidates.\n\n")
    if not records:
        if emit_enabled:
            f.write("No MF=10 sections without an MF=3 total were left "
                    "undecorated (--emit-mf10-only-reactions is ON).\n")
        elif from_h5:
            f.write("No MF=10 sections without an MF=3 total "
                    "(this HDF5 library carries none).\n")
        else:
            f.write("No MF=10 sections without an MF=3 total "
                    "(an HDF5 source can never carry any).\n")
        return
    meta = sum(1 for r in records if r.get('metastable'))
    f.write(f"Total sections excluded: {len(records)}  "
            f"(metastable-bearing: {meta})\n\n")
    header = (f"{'Parent':<12}  {'MT':>5}  {'Reaction':<12}  {'LFS':<16}  "
              f"{'Product(s)':<28}  {'Metastable':<10}")
    sep = "-" * len(header)
    f.write(header + "\n" + sep + "\n")
    for r in sorted(records, key=lambda x: (x['parent'], x['mt'])):
        lfs = ", ".join(str(v) for v in r.get('lfs', []))
        products = ", ".join(r.get('products', []))
        f.write(f"{r['parent']:<12}  {r['mt']:>5}  {r['reaction']:<12}  "
                f"{lfs:<16}  {products:<28}  "
                f"{'yes' if r.get('metastable') else 'no':<10}\n")


def _write_mf10_only_section(f, book):
    """MF=10-ONLY EMISSION section: every MT served without an MF=3 total.

    Written only when ``--emit-mf10-only-reactions`` is on, so a flag-off log is
    byte-identical to one from a build without the feature. Carries the counter
    block, the reconciliation identity, and one row per examined MT (emitted or
    skipped, with the reason).
    """
    examined, emitted, skipped = _mf10_only_reconciliation(book)
    f.write("\n\n" + "=" * 220 + "\n")
    f.write("MF=10-ONLY EMISSION (--emit-mf10-only-reactions)\n")
    f.write("=" * 220 + "\n\n")
    f.write("MTs whose library data is MF=10 isomeric production with NO MF=3 "
            "total. Their total is the union-grid SUM of their own partials "
            "(each event ends in exactly\n")
    f.write("one final state, so the sum IS the total) -- synthesized by the "
            "HDF5 builder and stamped 'total_source=sum-mf10', and reproduced "
            "identically by the ASC-tape\n")
    f.write("adapter. Q values come from the MF=10 section itself (QM = "
            "section QM; ground Q = the LFS=0 partial's QI), never fabricated. "
            "MT=5 (lumped residual) and MT=18\n")
    f.write("(fission placeholders) are never emitted; they stay in the MF=10 "
            "WITHOUT MF=3 section.\n\n")
    f.write("Emission shapes: a ground-only MT becomes a PLAIN reaction (no "
            "isomeric_branching child); a ground+metastable MT decorates "
            "normally; a metastable-only MT\n")
    f.write("folds with metastable members only (no ground row exists to "
            "serve). Products are always ELIS-mapped -- an MF=10 LFS is a level "
            "index, never an isomer ordinal.\n\n")
    f.write("Audit note: these MTs are VACUOUS to the MF=10-vs-MF=3 audit BY "
            "CONSTRUCTION -- their total is the partial sum, so every band "
            "ratio is 1.0 and the worst\n")
    f.write("deviation is 0 (up to float round-off). They can never be audit "
            "offenders and can never be band-rejected; that silence is "
            "expected, not a gap.\n\n")
    counters = [('Emitted plain (ground-only)', book['emitted_ground_only']),
                ('Emitted branched (g+m / m-only)', book['emitted_branched'])]
    counters += [(f'Skipped: {label}', book[key])
                 for key, label in _MF10_ONLY_SKIPS]
    width = max(len(label) for label, _n in counters)
    for label, n in counters:
        f.write(f"  {label:<{width}}  {n:6d}\n")
    f.write("-" * (width + 10) + "\n")
    f.write(f"  Reconciliation: examined MF=10-only MTs = emitted + skips = "
            f"{examined} = {emitted} + {skipped}"
            f"{'' if examined == emitted + skipped else '   *** MISMATCH ***'}"
            "\n\n")
    if not book['rows']:
        f.write("No MF=10-only MTs were examined (the library carries none, or "
                "it predates MF=10-only totals).\n")
        return
    header = (f"{'Parent':<12}  {'MT':>5}  {'Reaction':<12}  {'Shape':<12}  "
              f"{'LFS':<16}  {'Product(s)':<28}  {'Target':<12}  "
              f"{'Q[eV]':>16}  {'Outcome':<44}")
    sep = "-" * len(header)
    f.write(header + "\n" + sep + "\n")
    for r in sorted(book['rows'], key=lambda x: (x['parent'], x['mt'])):
        lfs = ", ".join(str(v) for v in r.get('lfs', []))
        products = ", ".join(r.get('products', []))
        q = r.get('q')
        q_str = f"{q:.4e}" if q is not None else "-"
        f.write(f"{r['parent']:<12}  {r['mt']:>5}  {r['reaction']:<12}  "
                f"{r['shape']:<12}  {lfs:<16}  {products:<28}  "
                f"{r.get('target', '-'):<12}  {q_str:>16}  "
                f"{r.get('outcome', ''):<44}\n")


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

        total_lfs = (stats['matched'] + stats['matched_rejected']
                     + stats['products_not_in_chain']
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
            f.write(f"                               LFS-order mapped: {stats['matched']:5d}\n")
        if stats['matched_rejected']:
            f.write(f"         matched but band-rejected (left stock): {stats['matched_rejected']:5d}\n")
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
        f.write(f"                Synthesized m->g (n,n') grounds: {stats.get('mg_ground_synthesized', 0):5d}\n")
        f.write(f"           Skipped m->g (n,n') (no ELIS/ground): {stats.get('mg_ground_skipped', 0):5d}\n")
        f.write(f"                    Ground-only MF=10 reactions: {stats['ground_only']:5d}\n")
        f.write(f"                Metastable-only MF=10 reactions: {stats['metastable_only']:5d}\n")
        f.write(f"             MF=10 without MF=3 (not decorable): {stats.get('mf10_without_mf3', 0):5d}\n")
        if stats.get('mf10_only_enabled'):
            book = stats['mf10_only']
            examined, emitted, skipped = _mf10_only_reconciliation(book)
            f.write(f"               MF=10-only emitted (ground-only): {book['emitted_ground_only']:5d}\n")
            f.write(f"                  MF=10-only emitted (branched): {book['emitted_branched']:5d}\n")
            f.write(f"                MF=10-only skipped (left stock): {skipped:5d}\n")
            f.write(f"          MF=10-only examined (= emitted+skips): {examined:5d}"
                    f"   [{emitted} + {skipped}"
                    f"{'' if examined == emitted + skipped else ' -- MISMATCH'}]\n")
        f.write(f"                          MF=10 audit offenders: {stats['audit_offenders']:5d}\n")
        f.write(f"                                 MF=10 rejected: {stats['rejected_count']:5d}\n")
        f.write(f"                 Band-reject exempt (self-loop): {stats['band_reject_exempt']:5d}\n")
        f.write(f"                 Rtol-reject exempt (self-loop): {stats['rtol_reject_exempt']:5d}\n")
        f.write(f"        Pathway-Q file QM rejected (reactions): {len(stats.get('pathway_q_rejected', [])):5d}\n")
        f.write(f"         Pathway-Q slots chain-anchored (gate): {stats.get('q_chain_anchored', 0):5d}\n")
        if stats.get('nn_prime_prune_enabled'):
            f.write(f"                       Pruned (n,n') self-loops: {stats.get('nn_prime_pruned_count', 0):5d}\n")
        f.write(f"                       LFS sentinel occurrences: {stats['lfs_sentinel_count']:5d}\n")
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
            stats.get('audit_clean', 0), stats.get('audit_emax', 2.0e7))
        _write_rejected_section(
            f, stats.get('rejected', []), stats.get('reject_rtol'),
            stats.get('reject_band_ratio'))
        _write_pathway_q_rejected_section(
            f, stats.get('pathway_q_rejected', []))
        _write_absent_decay_section(f, stats.get('absent_by_status', {}))
        _write_lfs_sentinel_section(f, stats.get('lfs_sentinels', []))
        _write_mf10_without_mf3_section(
            f, stats.get('mf10_without_mf3_list', []),
            emit_enabled=bool(stats.get('mf10_only_enabled')),
            from_h5=bool(stats.get('mf10_without_mf3_from_h5')))
        if stats.get('mf10_only_enabled'):
            _write_mf10_only_section(f, stats['mf10_only'])

    print(f"Isomer mapping log written to: {log_file}")


# =============================================================================
# Main workflow
# =============================================================================

def main(base_chain_file, pendf_path, decay_file, output_chain_file,
         log_file=None, mapping_mode='elis', elis_rtol=ELIS_RTOL,
         elis_atol=ELIS_ATOL, verbose=True, library=None, reject_rtol=None,
         audit_emax=2.0e7, reject_band_ratio=None,
         prune_nn_prime_self_loops=False, emit_mf10_only_reactions=False):
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
    source = open_pendf_source(
        pendf_path, library=library,
        emit_mf10_only_reactions=emit_mf10_only_reactions)
    print(f"  Backend: {source.kind}; nuclides: {len(source.nuclides)}; "
          f"library: {source.library}")
    if emit_mf10_only_reactions:
        # Printed only under the flag, so a flag-off console log is unchanged.
        print("  MF=10-only emission: ON (MTs with MF=10 partials but no MF=3 "
              "total are served with their partial sum as the total; "
              "MT=5/MT=18 excluded)")
    if source.kind == 'h5' and source.mapping not in (None, 'elis', 'lfs_order'):
        print(f"  NOTE: PENDF library mapping attr = {source.mapping!r}")

    print("\nStep 3: Loading decay library...")
    decay_lookup = parse_decay_isomeric_levels(decay_file)
    print(f"  Decay states for {len(decay_lookup)} (Z, A) nuclides")

    print("\nStep 4: Mapping MF=10 isomeric branching...")
    print(f"  ELIS tolerance: rtol={elis_rtol} ({elis_rtol*100:.0f}%), "
          f"atol={elis_atol} eV")
    print(f"  MF=10 audit cap: E <= {audit_emax:.3e} eV")
    if reject_rtol is None and reject_band_ratio is None:
        print("  MF=10 audit: detection + logging only (no rejection)")
    else:
        rtol_msg = (f"worst rel dev > {reject_rtol:.3e}"
                    if reject_rtol is not None else "off")
        band_msg = (f"band ratio - 1 > {reject_band_ratio:.3e} (over-sum only)"
                    if reject_band_ratio is not None else "off")
        print(f"  MF=10 audit rejection: rtol {rtol_msg}; band {band_msg} "
              f"-> reaction left stock")
    branching, stats = map_library(
        source, chain, decay_lookup, mapping_mode, elis_rtol, elis_atol,
        verbose=verbose, reject_rtol=reject_rtol, audit_emax=audit_emax,
        reject_band_ratio=reject_band_ratio)

    print("\nStep 5: Decorating chain...")
    reactions_added = decorate_chain(chain, branching, stats)
    stats['reactions_added'] = reactions_added

    # Optionally prune stock (n,n') ground self-loops (target == parent). Runs
    # AFTER decoration -- branched (n,n') groups already carry their pendf_lfs=0
    # ground member and are exempt -- and BEFORE the export/stamp step.
    if prune_nn_prime_self_loops:
        print("  Pruning (n,n') self-loops without isomeric branching...")
        nn_prime_pruned = _prune_nn_prime_self_loops(chain)
    else:
        nn_prime_pruned = []
    stats['nn_prime_prune_enabled'] = prune_nn_prime_self_loops
    stats['nn_prime_pruned'] = nn_prime_pruned
    stats['nn_prime_pruned_count'] = len(nn_prime_pruned)

    print_stats(stats, mapping_mode)
    _print_lfs_sentinel_warning(stats.get('lfs_sentinels', []), mapping_mode)
    q_rejected = stats.get('pathway_q_rejected', [])
    if q_rejected:
        print(f"\nWARNING: {len(q_rejected)} reaction(s) had a file ground QM "
              f"more than {PATHWAY_Q_CHAIN_TOL} eV from the chain's scalar Q; "
              "their file Q values were REFUSED and the chain-anchored values "
              f"kept ({stats.get('q_chain_anchored', 0)} metastable slot(s)); "
              "see PATHWAY-Q FILE-QM REJECTED in the mapping log",
              file=sys.stderr)
    if nn_prime_pruned:
        print(f"\nPruned (n,n') self-loops: {len(nn_prime_pruned)}")

    print("\nStep 6: Exporting folded chain XML...")
    # Stamp the exported chain's root element with the PENDF source's identity so
    # a wrong/stale chain paired with a library is self-detecting at collapse time
    # (openmc.deplete.microxs._verify_pendf_chain_stamp). The tape-derived
    # ``pendf_library`` string and the nuclide count are the mismatch triggers;
    # ``pendf_source`` (dir last-two-components / file basename) is informational
    # only -- a rename must not false-alarm. The ``decay_*`` attrs are pure
    # provenance (never verified at collapse), recording the decay library the
    # isomer mapping was resolved against.
    pendf_source = _pendf_source_label(pendf_path)
    decay_source = Path(decay_file).name
    decay_library = tape_identity(Path(decay_file)) or 'unknown'
    chain.root_attrs = {
        'pendf_source': pendf_source,
        'pendf_library': source.library,
        'pendf_nuclides': str(len(source.nuclides)),
        'decay_source': decay_source,
        'decay_library': decay_library,
    }
    chain.export_to_xml(output_chain_file)
    print(f"  Chain written to: {output_chain_file}")
    print(f"  Provenance stamp: library={source.library!r}, "
          f"nuclides={len(source.nuclides)}, source={pendf_source!r}")
    print(f"  Decay provenance: source={decay_source!r}, "
          f"library={decay_library!r}")

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
    if args.prune_nn_prime_self_loops:
        print("Prune (n,n') self-loops: ENABLED")
    if args.emit_mf10_only_reactions:
        print("Emit MF=10-only reactions: ENABLED")
    print(f"Audit emax:   {args.audit_emax:.3e} eV")
    if args.mf10_reject_rtol is None and args.mf10_reject_band_ratio is None:
        print("MF=10 reject: off (audit only)")
    else:
        rtol_msg = (f"worst rel dev > {args.mf10_reject_rtol}"
                    if args.mf10_reject_rtol is not None else "off")
        band_msg = (f"band ratio - 1 > {args.mf10_reject_band_ratio} "
                    "(over-sum only)"
                    if args.mf10_reject_band_ratio is not None else "off")
        print(f"MF=10 reject: rtol {rtol_msg}; band {band_msg}")
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
         reject_rtol=args.mf10_reject_rtol, audit_emax=args.audit_emax,
         reject_band_ratio=args.mf10_reject_band_ratio,
         prune_nn_prime_self_loops=args.prune_nn_prime_self_loops,
         emit_mf10_only_reactions=args.emit_mf10_only_reactions)

    print("\n" + "=" * 70)
    print("Done. Chain saved to:", output_chain)
    print("=" * 70)
