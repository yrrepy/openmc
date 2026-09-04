"""
GENDF Isomeric Branching Chain Patcher (v12)

Adds energy-dependent isomeric branching from GENDF MF=10 data to OpenMC chains.
Three mapping modes:
- 'elis_lfs_order' (default): hybrid - ELIS energy matching first, positional
  fallback for the levels left over, orphan states kept instead of dropped
- 'elis': ELIS-based mapping only; unmatched levels are dropped and the sibling
  branching renormalized
- 'lfs_order': FISPACT-like positional mapping for validation testing

A product with no partner in the chain (an unidentified excited state, or a
whole nuclide the decay library never carried) is handled by --orphan-policy
{add-stable, renorm, reattribute}, orthogonal to the mapping mode.

IMPORTANT: decay_file is REQUIRED for every mode (for count validation and logging).

IMPORTANT: the input base_chain MUST be an UNPATCHED chain. A rerun's output
chain and mapping log are unsupported: the first pass consumes the scalar Q
(moving it onto the <isomeric_branching> element), so the rerun's ledger
reconstructs a legacy value from nothing and reports phantom corrections. Since
48bf86e24 the pathway Q values themselves are stable across a rerun (they come
from the file's QM/QI, not from the consumed scalar; measured 0/3833 changed) --
only slots that fell back to the scalar Q would zero. Always start from a clean,
unpatched chain.

v12 Changes:
- 'elis_lfs_order' hybrid mapping mode, now the default; -r/--rtol defaults per
  mode (0.15 hybrid, 0.50 legacy)
- --orphan-policy {add-stable, renorm, reattribute} with add-stable the default:
  a product missing from the chain is added as a stable nuclide and its branch
  kept, instead of being dropped and renormalized away
- --output-dir / --log-dir keep verification runs out of the production outputs
- --reattribute-mf10-noIZAP recovers MF=10 subsections written with IZAP=0
  (evidence gate C1-C4); OFF or a gate failure prunes the reaction's ENTIRE
  isomeric decoration, all-or-nothing
- Added single-target isomeric yields detection and logging
- Always logs single-target cases with reason (GENDF_SINGLE_LFS, NO_DECAY_DATA, etc.)
- Added --suppress-single-target-yields flag to optionally suppress redundant yields
- Single-target section in mapping log file
- --prune-nn-prime-self-loops now prunes exact (n,n') self-loops only (X -> X); metastable-parent de-excitation (X_m1 (n,n') X) is preserved

v11 Changes:
- Added mapping_mode parameter ('elis' or 'lfs_order')
- Added COUNT MISMATCH REPORT section for LFS-order mode
- Log header clearly shows the mapping mode
- ELIS reference warnings for LFS-order mode (Ag116-type detection)
"""

import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from enum import Enum
from xml.dom import minidom

import numpy as np

import openmc.data
from openmc.deplete import Chain
from openmc.deplete.chain import REACTIONS
from openmc.deplete.nuclide import Nuclide
from openmc.deplete.decay_elis import lookup_liso
from openmc.deplete.gendf import (
    GENDFLibrary, REACTION_TO_MT, MT_TO_REACTION, PLACEHOLDER_LFS_VALUES,
    get_product_name
)


# =============================================================================
# LFS placeholder values ("unspecified isomer" conventions) -- report-only guard
# =============================================================================

# Some evaluations tag a reaction product whose final-state LEVEL could not be
# resolved with a PLACEHOLDER LFS instead of a true level index: unidentified
# excited states. Placeholders are NOT level ordinals and must NEVER be
# interpreted as isomer ordinals -- a placeholder must never become ``_m99`` /
# ``_m40``. ELIS mapping is unaffected (it matches the real GENDF-ELFS
# excitation energy, QM - QI); only a positional ``lfs_order`` consumer would
# mis-name them. Detection below is REPORT-ONLY: no mapping decision and no
# byte of the output chain depends on it. The hybrid ``elis_lfs_order`` mode
# carries its own placeholder binding rule, implemented library-side.
_PLACEHOLDER_DESCRIPTIONS = {
    99: 'ENDF/JEFF convention: isomer of unspecified level',
    40: 'TENDL convention: isomer of unspecified level',
}
# Keys derive from the library-side value set so the two can never drift.
PLACEHOLDER_LFS = {v: _PLACEHOLDER_DESCRIPTIONS[v]
                   for v in sorted(PLACEHOLDER_LFS_VALUES)}


# =============================================================================
# Mapping modes, orphan dispositions and their tolerance defaults
# =============================================================================

MAPPING_MODES = ('elis', 'lfs_order', 'elis_lfs_order')

# What happens to a reaction product with no partner in the chain (an orphan):
# a state the mapper could not identify, or a whole product nuclide the decay
# library never carried. All three are selectable at patch time and all three
# are reported in the ORPHAN DISPOSITION section of the mapping log.
ORPHAN_POLICIES = ('add-stable', 'renorm', 'reattribute')

# Mode-dependent -r/--rtol default (an explicit -r always wins). The hybrid
# trusts a Phase-1 ELIS accept absolutely -- it may cross the positional order
# -- so a wrong match costs far more there than a demotion to the positional
# fallback, and 0.15 sits in the empty valley of the accepted-difference
# distribution while still accepting the tightest genuine sibling pair on file
# (Lu166 m1/m2, 34.4 / 42.9 keV). The legacy modes keep 0.50, which reproduces
# every earlier production run byte for byte.
DEFAULT_ELIS_RTOL = {'elis': 0.50, 'lfs_order': 0.50, 'elis_lfs_order': 0.15}

# Hybrid bookkeeping records (lib.processing_errors types). None of these is a
# drop: they describe HOW a level was mapped, so they never feed the not-mapped
# tables. 'elis_incomputable' is the exception -- an elis-mode omission.
HYBRID_ERROR_TYPES = ('hybrid_fallback', 'placeholder_bound',
                      'placeholder_unmapped', 'hybrid_orphan_dk')

# Why a level left Phase 1, grouped for the counter block.
HYBRID_UNUSABLE_REASONS = ('qm_absent', 'elfs_unusable', 'dk_elis_zero',
                           'elis_tol_exceeded', 'no_dk_partner')

# Plain-language reasons for the log (the mapper's own labels are terse).
HYBRID_REASON_LABELS = {
    'qm_absent':         'QM absent: ELFS = QM - QI cannot be computed',
    'elfs_unusable':     'ELFS zero, negative or not a number',
    'dk_elis_zero':      'every decay state of the product carries ELIS = 0',
    'elis_tol_exceeded': 'nearest decay state lies outside the tolerance',
    'no_dk_partner':     'product has no metastable state in the decay library',
    'elis_ambiguous':    'two decay states within tolerance: abstained',
    'duplicate_loser':   'a closer level claimed the same decay state',
    'energy_unknown':    'placeholder LFS: the excitation energy is unknown',
}

# EAF-2010 (n,n') channels stored as TWO metastable levels with no ground
# subsection: emitting one static target would hand it the whole rate, and
# there is no MF=3 total to synthesize the ground from, so the channel is not
# emitted at all. Listed in the log as a standing exclusion class; the fix
# (ground-less two-isomer split emission) belongs to the full patcher run.
EAF_DUAL_METASTABLE_NN = (
    'Nb90', 'Sb124', 'Sb126', 'Eu152', 'Tb154', 'Lu177', 'Hf177', 'Hf178',
    'Hf179', 'Ir191', 'Ir194', 'Au194', 'Au196', 'Pb203',
)


def resolve_elis_rtol(mapping_mode, rtol=None):
    """Apply the mode-dependent -r/--rtol default (explicit value wins)."""
    return DEFAULT_ELIS_RTOL[mapping_mode] if rtol is None else rtol


# =============================================================================
# Anonymous (IZAP=0) MF=10 subsections -- R1-61 evidence gate
# =============================================================================

# A few evaluations (JEFF-3.3 Am241 MT=102, Al27 MT=16/107) tag an MF=10
# production subsection with IZAP=0, leaving the product nuclide unnamed while
# the level itself stays keyed by LFS. The residual can be RE-DERIVED from the
# reaction's deterministic (dA, dZ) shift, but only under the
# --reattribute-mf10-noIZAP flag (default: on) and only when EVERY anonymous subsection of the
# reaction clears the C1-C4 evidence gate below. Otherwise the reaction's ENTIRE
# isomeric decoration is pruned and the plain MF=3 route to the default target
# is kept: decorating the attributed subset alone would hand 100% of the rate to
# the metastable (the R1-61 inversion).

# MT -> (dA, dZ) residual shift, assembled from the two upstream tables so no
# reaction map is hand-written here: chain.REACTIONS names each MT and
# data.DADZ gives that name's shift. MT=5 (lumped) and MT=18 (fission) are
# absent from REACTIONS, so C1 excludes them by construction.
MT_TO_DADZ = {mt: openmc.data.DADZ[name]
              for name, info in REACTIONS.items() for mt in info.mts}

# C4 Q-consistency tolerances, one per leg. LFS>0 matches QM-QI to a decay
# level's ELIS across two independent sources (the real JEFF-3.3 cases sit
# 30-95 eV off: 1 keV). LFS=0 compares QM and QI inside one TAB1 head, where
# both descend from the same mass evaluation, so 1 eV -- wide enough for
# rounding, tight enough to reject the defective TENDL-2019/JEFF-3.3 ground
# heads (Q-55: QI = -E(level) or blank, 76.77 eV at the smallest) that a
# shared 1 keV gate would pass.
REATTRIB_Q_TOL_EV = 1.0e3
REATTRIB_Q_TOL_GROUND_EV = 1.0

# Pathway-Q sanity gate (eV, absolute): how far a non-MT=4 section's own ground
# QM may sit from the chain's scalar Q before the file value is refused and the
# whole reaction keeps the chain-anchored legacy arithmetic. 1 eV separates the
# populations cleanly: benign file jitter on agreeing sections is <= 0.31 eV
# after 4-dp rounding, the smallest genuine divergence is Am241 (n,gamma) at
# 3 350 eV (where the chain scalar is 344x closer to AME than the file), and the
# corrupt ENDF/B-8.1 (n,alpha) sections are 7.9-12.0 MeV out. Sub-eV agreement is
# not luck: both numbers descend from the same mass evaluation.
PATHWAY_Q_CHAIN_TOL = 1.0

# Marker: the MF=10 section could not be read (endf crasher). Distinct from
# ``None`` (fully attributed) so an unreadable section is never counted as clean.
MF10_LOAD_ERROR = 'load_error'


def _gate_anonymous_level(lib, parent, mt, level, lfs_counts,
                          q_tol=REATTRIB_Q_TOL_EV,
                          q_tol_ground=REATTRIB_Q_TOL_GROUND_EV):
    """Evidence gate C1-C4 for ONE anonymous (IZAP=0) MF=10 subsection.

    Returns ``(izap, verdicts)``: the DADZ-derived IZAP when every condition
    passes (``None`` otherwise), plus the ordered ``(condition, ok, detail)``
    verdicts for the report. ``lfs_counts`` is the section's LFS histogram (C2).
    Level identity is NEVER assumed from LFS (EAF-2010 keys Am242m as LFS=1,
    other libraries as LFS=2) -- C4 matches the real excitation energy.
    """
    verdicts = []

    # C1: deterministic-residual depletion reaction (MT=5/18 excluded here).
    dadz = MT_TO_DADZ.get(mt)
    verdicts.append(('C1_reaction', dadz is not None,
                     f"{MT_TO_REACTION.get(mt, '?')} dA,dZ={dadz}" if dadz else
                     f"MT={mt} is not a deterministic-residual depletion reaction"))
    if dadz is None:
        return None, verdicts

    # C2: valid LFS, unique within the section (a repeated LFS is ambiguous).
    try:
        lfs = int(level.get('LFS'))
    except (TypeError, ValueError):
        lfs = -1
    n_same = lfs_counts.get(lfs, 0)
    ok = lfs >= 0 and n_same == 1
    verdicts.append(('C2_lfs', ok, f"LFS={level.get('LFS')} x{n_same} in section"))
    if not ok:
        return None, verdicts

    # C3: the DADZ-derived residual exists in the decay library.
    d_a, d_z = dadz
    z_parent, a_parent, _ = openmc.data.zam(parent)
    z_prod, a_prod = z_parent + d_z, a_parent + d_a
    izap = z_prod * 1000 + a_prod
    decay_lookup = getattr(lib, 'decay_lookup', None) or {}
    ok = z_prod > 0 and a_prod > 0 and (z_prod, a_prod) in decay_lookup
    verdicts.append(('C3_product', ok, f"Z={z_prod} A={a_prod} IZAP={izap}"))
    if not ok:
        return None, verdicts

    # C4: Q consistency. LFS=0 needs QM == QI (within-head: q_tol_ground);
    # LFS>0 needs QM-QI to land on a decay-library level's ELIS (cross-source:
    # q_tol). ELIS lives at patch time only.
    qm, qi = level.get('QM'), level.get('QI')
    if qm is None or qi is None:
        verdicts.append(('C4_q', False, 'QM/QI missing from the MF=10 subsection'))
        return None, verdicts
    elfs = float(qm) - float(qi)
    if lfs == 0:
        ok = abs(elfs) <= q_tol_ground
        detail = f"QM-QI={elfs:.3f} eV (ground, tol {q_tol_ground:.1f} eV)"
    else:
        match = lookup_liso(z_prod, a_prod, elfs, decay_lookup,
                            rtol=0.0, atol=q_tol)
        ok = match.get('status') == 'matched'
        dk_elis = match.get('dk_elis')
        detail = (f"QM-QI={elfs/1e3:.3f} keV vs ELIS={dk_elis/1e3:.3f} keV "
                  f"(_m{match.get('liso')})" if ok else
                  f"QM-QI={elfs/1e3:.3f} keV: {match.get('status')}" +
                  (f", nearest ELIS={dk_elis/1e3:.3f} keV"
                   if dk_elis is not None else ""))
    verdicts.append(('C4_q', ok, detail))
    return (izap if ok else None), verdicts


def classify_mf10_attribution(lib, parent, mt, reattribute=False,
                              q_tol=REATTRIB_Q_TOL_EV,
                              q_tol_ground=REATTRIB_Q_TOL_GROUND_EV):
    """Classify -- and optionally repair -- one MF=10 section's IZAP attribution.

    Returns ``None`` when the section is fully attributed (nothing to report),
    ``MF10_LOAD_ERROR`` when it could not be read at all, else a record whose
    ``status`` is one of:

    * ``'excluded'``     -- every anonymous level fails C1 (MT=5/18 and friends;
      these MTs never carry isomeric decoration, so nothing is lost)
    * ``'reattributed'`` -- ``reattribute`` is on and every anonymous level
      cleared the gate; the derived IZAP is written straight into the parsed
      level, so product naming, branching extraction, the emission pre-pass and
      the consistency audit all see an ordinary attributed subsection
    * ``'anon_single'``  -- unrecovered, but the section holds a single final
      state, so there is no branching to decorate either way
    * ``'pruned'``       -- flag off, or at least one anonymous level failed the
      gate; ALL-OR-NOTHING, so no level of the section is repaired
    """
    try:
        mf10_result = lib._load_mf10_data(parent, mt)
    except Exception:
        return MF10_LOAD_ERROR
    if mf10_result is None:
        return None
    levels = mf10_result[0].get('levels', []) or []
    anonymous = [lv for lv in levels if int(lv.get('IZAP', 0) or 0) == 0]
    if not anonymous:
        return None

    lfs_counts = Counter(int(lv.get('LFS', 0) or 0) for lv in levels)
    recovered, failures = [], []
    for lv in anonymous:
        izap, verdicts = _gate_anonymous_level(lib, parent, mt, lv, lfs_counts,
                                               q_tol=q_tol,
                                               q_tol_ground=q_tol_ground)
        (failures if izap is None else recovered).append((lv, izap, verdicts))

    record = {
        'parent': parent, 'mt': mt, 'reaction': MT_TO_REACTION.get(mt),
        'n_levels': len(levels), 'n_anonymous': len(anonymous),
        'single_level': len(levels) == 1,
        'recovered': [{'lfs': int(lv.get('LFS', 0) or 0), 'izap': izap,
                       'evidence': verdicts[-1][2]}
                      for lv, izap, verdicts in recovered],
        'failed': [{'lfs': lv.get('LFS'),
                    'condition': next(v[0] for v in verdicts if not v[1]),
                    'detail': next(v[2] for v in verdicts if not v[1])}
                   for lv, _izap, verdicts in failures],
    }

    if not failures and reattribute:
        for lv, izap, _verdicts in recovered:
            lv['IZAP'] = izap
        record['status'] = 'reattributed'
        record['reason'] = None
        return record

    conditions = {f['condition'] for f in record['failed']}
    record['reason'] = sorted(conditions)[0] if conditions else 'flag_off'
    if conditions == {'C1_reaction'}:
        # Not a depletion channel at all (MT=5/18): reported, never decorated.
        record['status'] = 'excluded'
    elif record['single_level']:
        # One final state -- there is no branching to prune either way.
        record['status'] = 'anon_single'
    else:
        record['status'] = 'pruned'
    return record


def scan_mf10_attribution(lib, chain, reattribute=False,
                          q_tol=REATTRIB_Q_TOL_EV,
                          q_tol_ground=REATTRIB_Q_TOL_GROUND_EV):
    """Gate every MF=10 section of every chain nuclide for anonymous levels.

    Must run BEFORE the emission pre-pass and the branching extraction so a
    repaired IZAP is visible to both. Returns ``(records, pruned, counts)``:
    the per-section records that carry anonymous levels, the
    ``{(parent, reaction_name)}`` set whose isomeric decoration must be pruned,
    and the classification counters.
    """
    available = lib.available_nuclides_set()
    counts = Counter()
    records = []
    pruned = set()

    for parent in sorted({nuc.name for nuc in chain.nuclides} & set(available)):
        # Full parser needed for MF=10; the endf int_endf('') bug crashes ~97 of
        # the 816 EAF files -- absorb, count, and keep going (as the emission
        # pre-pass does), never abort the run.
        try:
            material = lib._load_material(parent, require_full_parser=True)
        except Exception:
            counts['load_error'] += 1
            continue
        for mt in sorted(mt for (mf, mt) in material.section_data if mf == 10):
            record = classify_mf10_attribution(lib, parent, mt,
                                               reattribute=reattribute,
                                               q_tol=q_tol,
                                               q_tol_ground=q_tol_ground)
            if record is MF10_LOAD_ERROR:
                counts['load_error'] += 1
                continue
            if record is None:
                counts['attributed'] += 1
                continue
            records.append(record)
            counts[record['status']] += 1
            counts['anonymous_levels'] += record['n_anonymous']
            if record['status'] == 'pruned' and record['reaction']:
                pruned.add((parent, record['reaction']))

    return records, pruned, counts


def unrepaired_anonymous_mts(records):
    """``{(parent, mt)}`` of scanned sections whose IZAP=0 levels stay unnamed.

    Everything the scan recorded except the repaired ones. The emission passes
    consult this: a section with an unrepaired anonymous level may not yield a
    pathway of its own, since the levels it CAN name are only a subset of the
    real final states (R1-61 all-or-nothing).
    """
    return {(r['parent'], r['mt']) for r in records
            if r['status'] != 'reattributed'}


def census_lfs0_ground_heads(lib, chain, tol=REATTRIB_Q_TOL_GROUND_EV):
    """Q-55 census: MF=10 LFS=0 heads whose QM and QI disagree (report-only).

    A ground product's level energy is zero by definition, so a LFS=0 TAB1
    head must carry QM == QI. TENDL-2019 and JEFF-3.3 instead write the
    reaction QI on ground-target tapes (QM=0, QI=-E(level)) or a blank QI on
    isomer-target tapes (QI=0, QM=+ELIS). Nothing in this tool consumes an
    LFS=0 QI -- every ground-route Q write is QM-sourced -- so violating
    heads are logged for observability, never repaired or acted on. Runs
    after ``scan_mf10_attribution`` so every material is already cached.
    Returns ``(rows, counts)``.
    """
    counts = Counter()
    rows = []
    for parent in sorted({nuc.name for nuc in chain.nuclides}
                         & set(lib.available_nuclides_set())):
        try:
            material = lib._load_material(parent, require_full_parser=True)
        except Exception:
            counts['load_error'] += 1
            continue
        for mt in sorted(mt for (mf, mt) in material.section_data if mf == 10):
            counts['sections'] += 1
            levels = material.section_data[10, mt].get('levels', []) or []
            for lv in levels:
                try:
                    if int(lv.get('LFS')) != 0:
                        continue
                except (TypeError, ValueError):
                    continue
                qm, qi = lv.get('QM'), lv.get('QI')
                if qm is None or qi is None:
                    counts['q_missing'] += 1
                    continue
                counts['lfs0_heads'] += 1
                qm, qi = float(qm), float(qi)
                elfs = qm - qi
                if abs(elfs) <= tol:
                    continue
                if qm == 0.0:
                    shape = 'ground-target'   # QM=0, QI=-E(level): reaction QI
                elif qi == 0.0:
                    shape = 'blank-QI'        # QI=0, QM=+ELIS
                else:
                    shape = 'other'
                counts[shape] += 1
                rows.append({'parent': parent, 'mt': mt,
                             'reaction': MT_TO_REACTION.get(mt),
                             'izap': int(lv.get('IZAP', 0) or 0),
                             'qm': qm, 'qi': qi, 'elfs': elfs, 'shape': shape})
    return rows, counts


def prune_unattributed_decoration(branching_data, pruned):
    """Drop the isomeric decoration of every reaction flagged by the scan.

    All-or-nothing (R1-61): the chain keeps its plain MF=3-backed reaction to
    the default target. Applies to both writer modes, since ``flags_only`` and
    ``embedded`` are both written from ``branching_data``.
    """
    for parent, reaction_name in pruned:
        branching_data.get(parent, {}).pop(reaction_name, None)
    return {p: rxns for p, rxns in branching_data.items() if rxns}


# =============================================================================
# Single-target reason codes
# =============================================================================

class SingleTargetReason(Enum):
    """Reasons why a reaction ended up with only a single target."""
    GENDF_SINGLE_LFS = "gendf_single_lfs"
    ELIS_TOL_EXCEEDED = "elis_tol_exceeded"
    NO_DECAY_DATA = "no_decay_data"
    ZERO_ELIS_DECAY = "zero_elis_metastables"
    DUPLICATE_MAPPING = "duplicate_mapping"
    PRODUCTS_NOT_IN_CHAIN = "products_not_in_chain"
    LFS_ORDER_DROPPED = "lfs_order_dropped"
    UNKNOWN = "unknown"

# =============================================================================
# Library configurations for CLI
# =============================================================================

LIBRARY_CONFIGS = {
    'jendl50': {
        'description': 'JENDL-5.0 (Native pairing) - UKAEA-1102',
        'endf_gxs_dir': '/home/perry/NukeData/Activation/FISPACT/JENDL5data/jendl5-n/gxs-1102/',
        'decay_file':   '/home/perry/NukeData/Activation/DecayData/jendl5dd/',
        'base_chain':   '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/JENDL50/Chain_JENDL50.xml',
        'output_dir':   '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/JENDL50/GENDF/',
        'output_prefix': 'Chain_JENDL50-IsoFlagGENDF',
        'log_prefix': 'JENDL50.IsoFlagGENDF.isomer_mapping',
    },
    'cendl32': {
        'description': 'CENDL-3.2 + ENDF/B-8.0 decay - UKAEA-1102',
        'endf_gxs_dir': '/home/perry/NukeData/Activation/FISPACT/CENDL32data/gendf-1102/',
        'decay_file': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/ENDFB80/endf-b8.0-endf/decay/ENDF-B-VIII.0_decay/',
        'base_chain': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/endf-b8.0/chain_endf_b8.0.xml',
        'output_dir': '/home/perry/NukeData/Activation/OMC/Perry-made/Isomeric-Chains/',
        'output_prefix': 'chain_endfCENDL32_dkENDF80_isoCENDL32gendf.mt4.',
        'log_prefix': 'CENDL32.IsoFlagGENDF.isomer_mapping',
    },
    'endfb71_decay2012': {
        'description': 'ENDF/B-7.1 + decay2012 - CCFE-709',
        'endf_gxs_dir': '/home/perry/NukeData/Activation/FISPACT/ENDFB71data/endfb71-n/gxs-709/',
        'decay_file': '/home/perry/NukeData/Activation/DecayData/ukdd-12_decay.dat',
        'base_chain': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/ENDFB71_decay2012/Chain_ENDFB71_decay2012.xml',
        'output_dir': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/ENDFB71_decay2012/GENDF/',
        'output_prefix': 'Chain_ENDFB71_decay2012-IsoFlagGENDF',
        'log_prefix': 'ENDFB71-dk2012.IsoFlagGENDF.isomer_mapping',
    },
    'endfb80': {
        'description': 'ENDF/B-8.0 (Native pairing) - CCFE-709',
        'endf_gxs_dir': '/home/perry/NukeData/Activation/FISPACT/ENDFB80data/endfb80-n/gxs-709/',
        'decay_file': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/ENDFB80/endf-b8.0-endf/decay/ENDF-B-VIII.0_decay/',
        'base_chain': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/ENDFB80/Chain_ENDFB80.xml',
        'output_dir': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/ENDFB80/GENDF/',
        'output_prefix': 'Chain_ENDFB80-IsoFlagGENDF',
        'log_prefix': 'ENDFB80.IsoFlagGENDF.isomer_mapping',
    },
    'endfb81': {
        'description': 'ENDF/B-8.1 (Native pairing) - UKAEA-1102',
        'endf_gxs_dir': '/home/perry/NukeData/Activation/FISPACT/ENDFB81data/endfb81-n/gxs-1102/',
        'decay_file': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/ENDFB81/decay/',
        'base_chain': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/ENDFB81/Chain_ENDFB81.xml',
        'output_dir': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/ENDFB81/GENDF/',
        'output_prefix': 'Chain_ENDFB81-IsoFlagGENDF',
        'log_prefix': 'ENDFB81.IsoFlagGENDF.isomer_mapping',
    },
    'tendl2017a': {
        'description': 'TENDL-2017 + decay2012 - CCFE-709',
        'endf_gxs_dir': '/home/perry/NukeData/Activation/FISPACT/TENDL2017data/tal2017-n/gxs-709/',
        'decay_file': '/home/perry/NukeData/Activation/DecayData/ukdd-12_decay.dat',
        'base_chain': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/TENDL2017a/Chain_TENDL2017.xml',
        'output_dir': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/TENDL2017a/GENDF/',
        'output_prefix': 'Chain_TENDL2017a-IsoFlagGENDF',
        'log_prefix': 'TENDL2017a.IsoFlagGENDF.isomer_mapping',
    },
    'tendl2017b': {
        'description': 'TENDL-2017 + decay2012 - CCFE-709',
        'endf_gxs_dir': '/home/perry/Projects/OMC_Development/PREPRO_production-pipeline_GENDF-PENDF/Pipeline-Produced/ref-IST/tendl2017b/gxs-709/',
        'decay_file': '/home/perry/NukeData/Activation/DecayData/ukdd-12_decay.dat',
        'base_chain': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/TENDL2017b/Chain_TENDL2017.xml',
        'output_dir': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/TENDL2017b/GENDF/',
        'output_prefix': 'Chain_TENDL2017b-IsoFlagGENDF',
        'log_prefix': 'TENDL2017b.IsoFlagGENDF.isomer_mapping',
    },
    'tendl2019': {
        'description': 'TENDL-2019 + decay2020 - UKAEA-1102',
        'endf_gxs_dir': '/home/perry/NukeData/Activation/FISPACT/TENDL2019data/gendf-1102/',
        'decay_file': '/home/perry/NukeData/Activation/DecayData/decay_2020/',
        'base_chain': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/TENDL2019/Chain_TENDL2019.xml',
        'output_dir': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/TENDL2019/GENDF/',
        'output_prefix': 'Chain_TENDL2019-IsoFlagGENDF',
        'log_prefix': 'TENDL2019.IsoFlagGENDF.isomer_mapping',
    },
    'tendl2021': {
        'description': 'TENDL-2021 + decay2020 - UKAEA-1102',
        'endf_gxs_dir': '/home/perry/NukeData/Activation/FISPACT/TENDL2021data/tal2021-n/gendf-1102/',
        'decay_file': '/home/perry/NukeData/Activation/DecayData/decay_2020/',
        'base_chain': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/TENDL2021/Chain_TENDL2021.xml',
        'output_dir': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/TENDL2021/GENDF/',
        'output_prefix': 'Chain_TENDL2021-IsoFlagGENDF',
        'log_prefix': 'TENDL2021.IsoFlagGENDF.isomer_mapping',
    },
    'jeff33': {
        'description': 'JEFF-3.3 (Native pairing) - CCFE-709',
        'endf_gxs_dir': '/home/perry/NukeData/Activation/FISPACT/JEFF33data/jeff33-n/gxs-709/',
        'decay_file': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/JEFF33/jeff-3.3-endf/decay/',
        'base_chain': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/JEFF33/Chain_JEFF33.xml',
        'output_dir': '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/JEFF33/GENDF/',
        'output_prefix': 'Chain_JEFF33-IsoFlagGENDF',
        'log_prefix': 'JEFF33.IsoFlagGENDF.isomer_mapping',
    },
    'jeff40': {
        'description':   'JEFF-4.0 (Native pairing) - UKAEA-1102',
        'endf_gxs_dir':  '/home/perry/NukeData/Activation/FISPACT/JEFF40data/jeff40-n/gxs-1102/',
        'decay_file':    '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/JEFF40/jeff-4.0-endf/decay/Radioactive_Decay_Data_JEFF-40.txt',
        'base_chain':    '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/JEFF40/Chain_JEFF40.xml',
        'output_dir':    '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/JEFF40/GENDF/',
        'output_prefix': 'Chain_JEFF40-IsoFlagGENDF',
        'log_prefix':    'JEFF40.IsoFlagGENDF.isomer_mapping',
    },
    'eaf2010': {
        'description':   'EAF-2010 (Native pairing) - CCFE-709',
        'endf_gxs_dir':  '/home/perry/NukeData/Activation/FISPACT/EAF2010data/eaf2010-n/gxs-709/',
        'decay_file':    '/home/perry/NukeData/Activation/DecayData/JEFF311RDD_ALL.OUT',
        'base_chain':    '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/EAF2010/Chain_EAF2010.xml',
        'output_dir':    '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/EAF2010/GENDF/',
        'output_prefix': 'Chain_EAF2010-IsoFlagGENDF',
        'log_prefix':    'EAF2010.IsoFlagGENDF.isomer_mapping',
    },
    'scale631': {
        'description':   'SCALE-6.3.1: EAF-2010 (JEFF3.1/A+) + ENDF/B-7.1 - CCFE-709',
        'endf_gxs_dir':  '/home/perry/NukeData/Activation/FISPACT/EAF2010data/eaf2010-n/gxs-709/',
        'decay_file':    '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/SCALE613/jeff-SCALE-6.1.3-endf/decay/decay/',
        'base_chain':    '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/SCALE613/Chain_SCALE613.xml',
        'output_dir':    '/home/perry/NukeData/openmc_data/src/openmc_data/depletion/SCALE613/GENDF/',
        'output_prefix': 'Chain_SCALE613-IsoFlagGENDF',
        'log_prefix':    'SCALE613.IsoFlagGENDF.isomer_mapping',
    },
}


def build_parser():
    """Build argument parser for CLI."""
    # Custom formatter to show library descriptions in help
    class CustomFormatter(argparse.RawDescriptionHelpFormatter):
        pass

    # Build library choices description for help
    lib_help_lines = ["Available library pairings:"]
    for key, config in LIBRARY_CONFIGS.items():
        lib_help_lines.append(f"  {key:<12} {config['description']}")

    epilog = "\n".join(lib_help_lines)

    parser = argparse.ArgumentParser(
        description='GENDF Isomeric Branching Chain Patcher v12\n\n'
                    'Adds energy-dependent isomeric branching from GENDF MF=10 data to OpenMC chains.\n\n'
                    'IMPORTANT: the input base_chain must be an UNPATCHED chain. A rerun on an '
                    'already-patched chain produces an unsupported output chain and ledger: the '
                    'scalar Q it reconstructs the legacy comparison from was consumed by the first '
                    'pass, so the ledger reports phantom corrections. The pathway Q values are now '
                    'stable across a rerun (sourced from the file QM/QI); only scalar-Q fallback '
                    'slots would zero. Always start from a clean, unpatched chain.',
        epilog=epilog,
        formatter_class=CustomFormatter
    )

    parser.add_argument(
        '-l', '--library',
        choices=list(LIBRARY_CONFIGS.keys()),
        required=True,
        metavar='LIB',
        help='Library pairing to use (see list below)'
    )

    parser.add_argument('-m', '--map',      choices=MAPPING_MODES,    default='elis_lfs_order', help="Mapping mode. 'elis_lfs_order' (hybrid, DEFAULT): match each GENDF level to a decay-library state by excitation energy first (unambiguous matches only), pair the levels left over positionally with the decay states left over, and keep any level with no decay partner at all as an ORPHAN state (disposition set by --orphan-policy) instead of discarding it. 'elis': excitation-energy matching only; unmatched levels are dropped and their sibling branching renormalized. 'lfs_order': FISPACT-like positional mapping (1st LFS -> _m1, 2nd -> _m2, ...), for validation runs.")
    parser.add_argument('--orphan-policy',  choices=ORPHAN_POLICIES,  default='add-stable',     help="What to do with a reaction product that has no partner in the chain -- an excited state the mapper could not identify, or a whole product nuclide the decay library never carried. 'add-stable' (DEFAULT): keep the branch and add the product to the output chain as a new nuclide with no decay data (a stable pure sink). Its name is the lowest free _mN of that Z/A, allocated against the decay-library LISO indices and the existing chain names -- it is a placeholder ordinal, NOT a decay-library LISO; several parents feeding the same unidentified state share one nuclide. 'renorm': drop the product and redistribute its share pro-rata over the surviving targets (the historical behaviour). 'reattribute': fold the orphan's share DOWN into the kept isomer of the same product at the nearest lower rank (ground when none is below it) -- unrelated to --reattribute-mf10-noIZAP, which re-derives an unnamed IZAP=0 product. Placeholder LFS levels (unidentified excited states, 99/40) are never orphan-added under any policy. Every disposition is listed in the ORPHAN DISPOSITION and ORPHAN NUCLIDES ADDED TO CHAIN sections of the mapping log.")
    parser.add_argument('-r', '--rtol',     type=float,               default=None,             help='Relative tolerance for ELIS matching. Default depends on the mapping mode: 0.15 (15%%) for elis_lfs_order, 0.50 (50%%) for elis and lfs_order. An explicit value always wins, so a tolerance study stays separable from a mode study.')
    parser.add_argument('-a', '--atol',     type=float,               default=0.0,              help='Absolute tolerance for ELIS matching in eV (default: 0.0)')

    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        default=False,
        help='Disable the verbose progress output (on by default)'
    )

    parser.add_argument(
        '--prune-nn-prime-self-loops',
        action='store_true',
        default=True,
        help="Remove (n,n') reactions that are exact self-loops (target == parent "
             "nuclide) with no isomeric branching (Bateman diagonal no-ops with no "
             "effect on depletion results). Metastable-parent de-excitation "
             "(X_m1 (n,n') X) is always preserved. Default: keep all (n,n') reactions."
    )

    parser.add_argument(
        '--suppress-single-target-yields',
        action='store_true',
        default=True,
        help="Suppress redundant single-target isomeric yields where the sole product equals "
             "the original reaction target with all 1.0 branching ratios. These add XML bloat "
             "without physics impact. Single-target cases are ALWAYS logged with reasons "
             "regardless of this flag. Default: keep all isomeric yields."
    )

    parser.add_argument(
        '--mode',
        choices=['flags_only', 'embedded'],
        default='flags_only',
        help="Output mode: 'flags_only' (default) writes target names only as "
             "<isomeric_branching targets='...'/> elements. 'embedded' writes full "
             "energy-dependent ratios as <isomeric_yields> (legacy/informational)."
    )

    parser.add_argument('--audit-emax',               type=float,          default=2.0e7, help='Cap the MF=10-vs-MF=3 consistency audit at E <= this many eV (default: 2.0e7; MF=10 partials legitimately stop near 30 MeV while MF=3 runs higher)')
    parser.add_argument('--mf10-reject-band-ratio',   type=float,          default=None,  help='Leave a reaction stock (no isomeric branching) when any DEFINED lethargy band has ratio-1 > X, over-summing ONLY; under-summing never rejects (it is the radioactive-products-only MF=10 signature: an absent stable ground or anonymous levels missing from the partials). Default: None = audit only, reject nothing. GENDF-SPECIFIC NOTE: band-ratio deviations are HARMLESS if common-mode (small BR-spread) on the GENDF ratio path, since the runtime applies partial/Sum(partials) ratios to an MF=3 rate; this gate stays OFF by default.')
    parser.add_argument('--emit-mf10-only-reactions', action='store_true', default=True,  help='Emit plain <reaction> elements for GENDF MF=10-only channels (residual has a tabulated isomer => stored in MF=8/10 with no MF=3, e.g. EAF-2010 Al27(n,a)Na24) before isomeric decoration. Default: on; general-purpose libraries (TENDL/JEFF/ENDF) have none, so the pass emits nothing there.')
    parser.add_argument('--output-dir',               type=str,            default=None,  help='Write the output chain XML into this directory instead of the library pairing\'s own output directory. Use it for verification runs so they never overwrite a production chain. Default: the pairing\'s output_dir.')
    parser.add_argument('--log-dir',                  type=str,            default=None,  help='Write the mapping log into this directory instead of the library pairing\'s own output directory. Use it for verification runs so they never overwrite a production log. Default: the pairing\'s output_dir.')
    parser.add_argument('--reattribute-mf10-noIZAP',  action='store_true', default=True,  help='Recover MF=10 subsections written with IZAP=0 (product nuclide unnamed) by re-deriving the residual from the reaction dA/dZ, gated on evidence C1 (deterministic-residual depletion MT), C2 (valid, section-unique LFS), C3 (derived product in the decay library) and C4 (Q consistency: QM==QI within 1 eV for LFS=0, QM-QI == a decay level ELIS within 1 keV). Default: on. ANY anonymous subsection failing the gate prunes that reaction\'s ENTIRE isomeric decoration (all-or-nothing) and keeps the plain MF=3 route -- decorating the attributed subset alone would invert the branching. Only JEFF-3.3 needs this (Am241, Al27); a no-op elsewhere. NOT related to "--orphan-policy reattribute": this one NAMES a product the file left unnamed, that one FOLDS an identified-but-unmatchable state into a kept sibling.')

    return parser


# =============================================================================
# Helper functions
# =============================================================================

def _z_to_element(z):
    """Convert atomic number to element symbol."""
    elements = [
        'n', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg',
        'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
        'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
        'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
        'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
        'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
        'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po',
        'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
        'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs',
        'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
    ]
    return elements[z] if 0 <= z < len(elements) else f'Z{z}'


# =============================================================================
# MF=10-vs-MF=3 consistency audit  (GROUP-SPACE port of the PENDF patcher)
# =============================================================================
#
# This is the multigroup cousin of the pointwise audit in
# ``add_pendf_isomeric_branching_to_chain.py``. For every decorated-candidate
# reaction it compares the summed MF=10 isomeric-production partials against the
# MF=3 total, group by group, and reports a per-band lethargy-weighted ratio so
# it is clear WHERE (thermal / epithermal / intermediate / fast) any departure
# lives. Column names, section titles, the ``--mf10-reject-band-ratio`` knob and
# the console phrasing mirror the PENDF tool so downstream tooling reads both.
#
# DELIBERATE GENDF/PENDF SEMANTIC DIFFERENCE -- read before using rejection:
# the GENDF depletion runtime consumes branching as partial/Sum(partials)
# RATIOS applied to an MF=3-total reaction rate. A COMMON-MODE inflation of all
# partials (the same factor in every group -- e.g. the known JEFF-4.0 sub-thermal
# lin-lin chord class) therefore CANCELS in the ratio and is HARMLESS on this
# path; only DIFFERENTIAL defects (partials disagreeing with each other by
# energy) bias results. Consequently rejection stays OFF by default, and the
# audit adds a GENDF-specific ``BR-spread`` column that separates the two:
# near-constant branching fractions (small spread) + large band ratios => benign
# common-mode; a large spread => genuine differential suspicion.

# Constants mirror ``openmc.deplete.microxs`` in the PENDF fork (this GENDF
# fork's microxs.py does not define them). RTOL is the "offender" threshold for
# the always-on table; ABS_FLOOR is the both-sides evaluator floor-dust exempt.
CONSISTENCY_RTOL = 1e-5
CONSISTENCY_ABS_FLOOR = 1e-15

# Lethargy band edges (eV): thermal [grid_min, 0.625), epithermal [0.625, 1e5),
# intermediate [1e5, 1e6), fast [1e6, emax]. 0.625 eV is the cadmium cutoff.
_BAND_THERMAL_HI = 0.625
_BAND_EPITHERMAL_HI = 1.0e5
_BAND_INTERMEDIATE_HI = 1.0e6

# ``unmatched_mts`` reasons meaning "MF=10 had metastables but no usable ground"
# (policy 3(a) declined, or repaired and then lost at mapping). Explicit
# membership, not a prefix test: a future 'mf10_*' reason must not silently join
# the GROUND-ABSENT report.
GROUND_ABSENT_SKIP_REASONS = {
    'mf10_anonymous_levels',
    'mf10_metastable_only_no_mf3',
    'mf10_ambiguous_izap',
    'ground_absent_all_metastables_unmapped',
}

# Readable labels for the GROUND-ABSENT report; anything else prints verbatim.
GROUND_ABSENT_SKIP_LABELS = {
    'ground_absent_all_metastables_unmapped':
        'ground synthesized, but no metastable survived mapping',
}


def _partials_total_max_deviation(total_g, part_sum):
    """Max relative deviation of summed MF=10 partials from the MF=3 total.

    Group-space definition, ported verbatim from the PENDF fork's
    ``openmc.deplete.microxs``: returns ``(worst, group_idx)`` over groups with
    nonzero total, skipping groups where BOTH sides are below
    ``CONSISTENCY_ABS_FLOOR`` (evaluator floor dust). ``(0.0, -1)`` when no group
    qualifies.
    """
    nz = (total_g != 0.0) & (
        (np.abs(total_g) >= CONSISTENCY_ABS_FLOOR)
        | (np.abs(part_sum) >= CONSISTENCY_ABS_FLOOR))
    if not nz.any():
        return 0.0, -1
    dev = np.abs(part_sum[nz] - total_g[nz]) / np.abs(total_g[nz])
    worst = float(dev.max())
    return worst, int(np.nonzero(nz)[0][dev.argmax()])


def _band_lethargy_weights(bounds, lo, hi):
    """Per-group lethargy overlap Delta-u_g of each group with band ``[lo, hi]``.

    ``bounds`` is the ascending energy-boundary array (length n_groups+1); group
    g spans ``[bounds[g], bounds[g+1]]``. Returns a length-n_groups array of
    ``ln(min(bounds[g+1], hi) / max(bounds[g], lo))`` clipped at 0 -- the portion
    of the group's lethargy that falls inside the band. A group straddling a band
    edge is thereby apportioned by its lethargy overlap (exact for multigroup
    data, where the group cross section is constant so
    ``int sigma/E dE = sigma * Delta-u``).
    """
    elo = np.asarray(bounds[:-1], dtype=float)
    ehi = np.asarray(bounds[1:], dtype=float)
    top = np.minimum(ehi, hi)
    bot = np.maximum(elo, lo)
    with np.errstate(divide='ignore', invalid='ignore'):
        du = np.where((top > bot) & (bot > 0.0), np.log(top / bot), 0.0)
    return du


def _band_ratio_gendf(bounds, total_pg, part_pg, lo, hi):
    """Lethargy-weighted band ratio ``Sum part_g du_g / Sum tot_g du_g``.

    ``du_g`` is each group's lethargy overlap with ``[lo, hi]`` (band-edge groups
    apportioned by overlap). Returns ``(ratio, part_nonzero_vs_zero_total)``.
    ``ratio`` is ``None`` when the band has no contributing group, a zero total
    integral, or fails the significance floor -- a band whose contributing MF=3
    totals never rise above ``CONSISTENCY_ABS_FLOOR`` is below-threshold
    evaluator dust and cannot produce a meaningful ratio (mirrors the PENDF
    tool). The second flag is ``True`` only in the degenerate case of nonzero
    partials against a zero total.
    """
    du = _band_lethargy_weights(bounds, lo, hi)
    sel = du > 0.0
    if not sel.any():
        return None, False
    if float(np.max(total_pg[sel])) < CONSISTENCY_ABS_FLOOR:
        return None, False
    int_total = float(np.sum(total_pg * du))
    if int_total == 0.0:
        int_part = float(np.sum(part_pg * du))
        return None, (int_part != 0.0)
    return float(np.sum(part_pg * du)) / int_total, False


def _branching_spread(bounds, meta_partials, part_sum, band_lo, band_hi, emax):
    """GENDF-specific BR-spread: branching-fraction spread WITHIN one band.

    For each non-ground partial ``m`` compute the per-group branching fraction
    ``r_m,g = part_m,g / Sum(parts)_g`` over the SIGNIFICANT groups that overlap
    the band ``[band_lo, band_hi]`` (lethargy overlap > 0, ``Sum(parts)_g >
    CONSISTENCY_ABS_FLOOR``, ``E < emax``), then take ``max_g r - min_g r``.
    BR-spread is the max of that over the non-ground partials -- one number.

    The band is deliberately the WORST-deviating band (largest ``|ratio - 1|``;
    chosen by the caller), i.e. where the MF10-vs-MF3 anomaly lives. Measuring
    the spread THERE -- not over the full range -- is what separates the two
    failure modes: a sub-thermal common-mode chord scales all partials together
    so the branching fraction is CONSTANT across the anomalous band (spread ~ 0),
    whereas a differential defect makes the partials disagree by energy WITHIN
    the anomalous band (large spread). Full-range spread would instead be
    dominated by the reaction's legitimate fast-region branching variation and
    could not tell a benign chord from a real defect (verified empirically:
    JEFF-4.0 chord class -> full-range median ~0.4 but worst-band ~0.000).
    Returns ``None`` when no non-ground partial or no qualifying group.
    """
    if not meta_partials:
        return None
    du = _band_lethargy_weights(bounds, band_lo, band_hi)
    elo = np.asarray(bounds[:-1], dtype=float)
    sig = (du > 0.0) & (part_sum > CONSISTENCY_ABS_FLOOR) & (elo < emax)
    if not sig.any():
        return None
    denom = part_sum[sig]
    max_spread = 0.0
    for _lfs, pg in meta_partials:
        r = pg[sig] / denom
        spread = float(r.max() - r.min())
        if spread > max_spread:
            max_spread = spread
    return max_spread


def _self_loop_ground_gendf(parent, branching):
    """True when the reaction's GROUND product is the parent itself.

    Group-space analog of the PENDF self-loop-ground exemption: an
    ``(n,n')``-type ground route back to the parent is a transmutation-matrix
    no-op (loss and gain both land on the diagonal and cancel), so partial-sum
    band incompleteness cannot affect the chain -- only the metastable partials
    carry real isomer production, and rejecting would destroy it. The ground
    product is the branching's first product (``products[0]`` -- the LFS=0 entry
    the flags-only writer emits as ground). A metastable parent's ground route
    (e.g. In115_m1 -> In115) is a real transition, NOT a self-loop, and stays
    rejectable because ``products[0]`` then differs from ``parent``.
    """
    products = list(getattr(branching, 'products', None) or [])
    return bool(products) and products[0] == parent


def _audit_reaction_gendf(lib, parent, mt, emax=2.0e7, ground_repaired=False):
    """Group-space MF=10-vs-MF=3 consistency for one reaction.

    Reads the MF=3 total (per-group) and every MF=10 production partial
    (``IZAP != 0``, ground + metastable), each aligned to the library group grid
    via the reader's ``_extract_xs`` -- which handles a threshold partial's group
    offset. (``_get_production_xs`` now performs the same energy-aware alignment
    of partial-range MF=10 bands; ``_extract_xs`` is kept here simply because the
    audit already holds each raw level from ``_load_mf10_data``.) Groups whose low
    edge is at or above ``emax`` are dropped from the worst-deviation scan; band
    ratios cap via the fast band's lethargy overlap at ``emax``.

    Returns ``None`` when MF=3 or MF=10 is unavailable, else a dict with
    ``worst_dev`` and the worst group's energy band / values, the capped
    lethargy-weighted ``integral_ratio``, the four per-band ratios, the
    GENDF-specific ``br_spread``, and ``notes``.
    """
    try:
        mf3 = np.asarray(lib.get_xs(parent, mt), dtype=float)
    except Exception:
        return None
    try:
        mf10 = lib._load_mf10_data(parent, mt)
    except Exception:
        return None
    if mf10 is None:
        return None
    mf10_data, _ = mf10

    bounds = np.asarray(lib.energy_bounds, dtype=float)
    n_groups = int(lib.n_groups)
    if mf3.size != n_groups or bounds.size != n_groups + 1:
        return None

    part_sum = np.zeros(n_groups)
    meta_partials = []              # (lfs, per-group array) for non-ground
    n_anonymous = 0                 # IZAP=0: no product, excluded from partials
    for level in mf10_data.get('levels', []):
        if int(level.get('IZAP', 0)) == 0:
            n_anonymous += 1
            continue
        try:
            pg = np.asarray(lib._extract_xs({'sigma': level['sigma']}, parent,
                                            mt, False), dtype=float)
        except Exception:
            continue
        if pg.size != n_groups:
            continue
        part_sum = part_sum + pg
        if int(level.get('LFS', 0)) != 0:
            meta_partials.append((int(level['LFS']), pg))

    # emax cap: keep groups whose LOW edge is below emax for the worst-dev scan.
    keep = bounds[:-1] < emax
    if not keep.any():
        return None
    mf3_k = np.where(keep, mf3, 0.0)
    part_k = np.where(keep, part_sum, 0.0)

    worst, gidx = _partials_total_max_deviation(mf3_k, part_k)
    if gidx >= 0:
        energy_lo = float(bounds[gidx])
        energy_hi = float(bounds[gidx + 1])
        sum_at = float(part_sum[gidx])
        total_at = float(mf3[gidx])
    else:
        energy_lo = energy_hi = sum_at = total_at = None

    # Full-range (capped) lethargy-weighted integral ratio.
    du_full = _band_lethargy_weights(bounds, 0.0, emax)
    int_total = float(np.sum(mf3 * du_full))
    int_part = float(np.sum(part_sum * du_full))
    ratio = (int_part / int_total) if int_total != 0.0 else float('inf')

    r_th, f_th = _band_ratio_gendf(bounds, mf3, part_sum, 0.0, _BAND_THERMAL_HI)
    r_ep, f_ep = _band_ratio_gendf(bounds, mf3, part_sum, _BAND_THERMAL_HI,
                                   _BAND_EPITHERMAL_HI)
    r_in, f_in = _band_ratio_gendf(bounds, mf3, part_sum, _BAND_EPITHERMAL_HI,
                                   _BAND_INTERMEDIATE_HI)
    r_fa, f_fa = _band_ratio_gendf(bounds, mf3, part_sum, _BAND_INTERMEDIATE_HI,
                                   emax)

    flagged = [name for name, flag in (('thermal', f_th), ('epithermal', f_ep),
               ('intermediate', f_in), ('fast', f_fa)) if flag]
    notes = (f"partials nonzero vs zero total in {', '.join(flagged)}"
             if flagged else '')
    # Anonymous levels are missing from Sum(partials), so say so rather than let
    # the resulting deficit read as an MF=10-vs-MF=3 inconsistency.
    if n_anonymous:
        anon_note = f"{n_anonymous} anonymous (IZAP=0) level(s) not in partials"
        notes = f"{notes}; {anon_note}" if notes else anon_note
    # A repaired reaction has no MF=10 ground at all -- Sum(partials) is the
    # metastables alone, so the MF=3 deficit is the synthesized ground, by
    # construction, not an MF=10-vs-MF=3 inconsistency. Informational only: the
    # one-sided reject gate already ignores every deficit.
    if ground_repaired:
        repair_note = ("ground synthesized from MF=3: partials are metastables "
                       "only")
        notes = f"{notes}; {repair_note}" if notes else repair_note

    # BR-spread is measured WITHIN the worst-deviating defined band (max
    # |ratio - 1|) -- where the MF10-vs-MF3 anomaly lives -- so it reports
    # whether THAT anomaly is common-mode (constant branching -> small) or
    # differential (varying -> large). See _branching_spread.
    bands = [('thermal', 0.0, _BAND_THERMAL_HI, r_th),
             ('epithermal', _BAND_THERMAL_HI, _BAND_EPITHERMAL_HI, r_ep),
             ('intermediate', _BAND_EPITHERMAL_HI, _BAND_INTERMEDIATE_HI, r_in),
             ('fast', _BAND_INTERMEDIATE_HI, emax, r_fa)]
    defined = [b for b in bands if b[3] is not None]
    if defined:
        wname, wlo, whi, _wr = max(defined, key=lambda b: abs(b[3] - 1.0))
        br_spread = _branching_spread(bounds, meta_partials, part_sum, wlo, whi,
                                      emax)
        br_spread_band = wname
    else:
        br_spread = None
        br_spread_band = None

    return dict(
        worst_dev=worst, group_index=gidx,
        energy_lo=energy_lo, energy_hi=energy_hi,
        sum_partials=sum_at, total=total_at,
        integral_ratio=ratio, ratio_thermal=r_th, ratio_epithermal=r_ep,
        ratio_intermediate=r_in, ratio_fast=r_fa, br_spread=br_spread,
        br_spread_band=br_spread_band, anonymous_levels=n_anonymous,
        notes=notes)


def run_mf10_consistency_audit(lib, branching_data, emax=2.0e7,
                               reject_band_ratio=None):
    """Audit every decorated-candidate reaction and (optionally) gate rejection.

    ``branching_data`` is ``{parent: {reaction_name: IsomericBranching}}`` -- the
    exact set that would receive an ``<isomeric_branching>`` child. Every one is
    run through :func:`_audit_reaction_gendf` (always -- the table is
    always-on). When ``reject_band_ratio`` is set, a reaction whose
    ``band ratio - 1`` exceeds it on ANY defined band is REJECTED (returned so
    the caller can leave it stock), UNLESS it is a self-loop-ground reaction
    (exempt -- see :func:`_self_loop_ground_gendf`). The test is ONE-SIDED
    (over-summing only): under-summing is the radioactive-products-only MF=10
    signature -- an absent stable ground, or anonymous levels missing from the
    partials -- and never rejects. Rejection is OFF by default; band-ratio
    deviations are harmless if common-mode on the GENDF ratio path (see the
    module note and the ``BR-spread`` column).

    Returns ``(audit_rows, offender_count, clean_count, rejected_rows,
    exempt_count)``.
    """
    audit_rows = []
    rejected_rows = []
    offenders = clean = exempt = 0
    repaired = {(r['nuclide'], r['mt'])
                for r in getattr(lib, 'ground_repaired', [])}

    for parent in sorted(branching_data):
        for r_name, branching in branching_data[parent].items():
            mt = getattr(branching, 'mt', None)
            if mt is None:
                continue
            is_repaired = (parent, mt) in repaired
            audit = _audit_reaction_gendf(lib, parent, mt, emax=emax,
                                          ground_repaired=is_repaired)
            if audit is None:
                continue
            row = dict(parent=parent, mt=mt, reaction=r_name, **audit)
            audit_rows.append(row)
            if audit['worst_dev'] > CONSISTENCY_RTOL:
                offenders += 1
            else:
                clean += 1

            if reject_band_ratio is None:
                continue
            band_fired = []
            for key, band in (('ratio_thermal', 'thermal'),
                              ('ratio_epithermal', 'epithermal'),
                              ('ratio_intermediate', 'intermediate'),
                              ('ratio_fast', 'fast')):
                r = audit.get(key)
                if r is not None and (r - 1.0) > reject_band_ratio:
                    band_fired.append(f'band_ratio:{band}')
            if not band_fired:
                continue
            if _self_loop_ground_gendf(parent, branching):
                exempt += 1
                marker = 'self-loop ground: band-reject exempt'
                row['notes'] = (f"{row['notes']}; {marker}"
                                if row['notes'] else marker)
                continue
            rejected_rows.append(dict(parent=parent, mt=mt, reaction=r_name,
                                      criterion=', '.join(band_fired), **audit))

    return audit_rows, offenders, clean, rejected_rows, exempt


def _parse_no_metastable_decay_data_error(error_str):
    """
    Parse 'no_metastable_decay_data' error string to extract details.

    Example error string:
    "WARNING: NO_METASTABLE_DECAY_DATA: Cd112((n,pa))->Rh108_m? LFS=4 ELIS=215000.0 eV. No metastable states in decay library for (Z=45, A=108). Reaction Isomeric Branching not-mapped."

    Returns dict with: parent, reaction, lfs, elis, target_z, target_a, product
    """
    result = {
        'parent': '?', 'reaction': '?', 'lfs': None, 'elis': None,
        'target_z': 0, 'target_a': 0, 'product': '?'
    }

    # Parse parent and reaction: "Cd112((n,pa))->"
    parent_rx_match = re.search(r'(\w+)\(\(([^)]+)\)\)->', error_str)
    if parent_rx_match:
        result['parent'] = parent_rx_match.group(1)
        result['reaction'] = f"({parent_rx_match.group(2)})"

    # Parse LFS: "(LFS=4,"
    lfs_match = re.search(r'LFS=(\d+)', error_str)
    if lfs_match:
        result['lfs'] = int(lfs_match.group(1))

    # Parse ELIS: "ELIS=215000.0 eV"
    elis_match = re.search(r'ELIS=([\d.]+)', error_str)
    if elis_match:
        result['elis'] = float(elis_match.group(1))

    # Parse Z, A: "(Z=45, A=108)"
    za_match = re.search(r'Z=(\d+),\s*A=(\d+)', error_str)
    if za_match:
        result['target_z'] = int(za_match.group(1))
        result['target_a'] = int(za_match.group(2))

    # Build product name from Z, A, LFS
    if result['target_z'] > 0:
        elem = _z_to_element(result['target_z'])
        mass = result['target_a']
        lfs = result['lfs']
        result['product'] = f"{elem}{mass}_m{lfs}" if lfs else f"{elem}{mass}_m?"

    return result


def _collect_lfs_placeholders(branching_data, elis_errors, duplicate_mapping_errors,
                           lfs_order_dropped, mapping_mode):
    """Report-only scan for LFS PLACEHOLDER values (see :data:`PLACEHOLDER_LFS`).

    Returns a list of occurrence records drawn from BOTH the successfully mapped
    products (``branching_data`` -> ``IsomericBranching.lfs_mapping``) and the
    unmapped/skipped error channels (``lib.processing_errors``). No mapping
    decision depends on this; it exists so a placeholder LFS -- which the
    flags-only writer would otherwise emit verbatim in ``gendf_lfs`` -- is never
    silently consumed as an isomer ordinal (``_m99`` / ``_m40``).
    """
    placeholders = []
    method = 'lfs_order' if mapping_mode == 'lfs_order' else 'elis'

    def _base(product, z, a):
        if product and '_m' in str(product):
            return str(product).split('_m')[0]
        if z and a:
            return f"{_z_to_element(z)}{a}"
        return '?'

    # 1) Mapped products (these are what the flags-only writer emits to the
    #    chain's gendf_lfs attribute -- the primary ordinal-misuse risk).
    for parent, reactions in (branching_data or {}).items():
        for reaction_type, branching in (reactions or {}).items():
            lfs_map = getattr(branching, 'lfs_mapping', None) or {}
            elis_map = getattr(branching, 'elis_mapping', None) or {}
            mt = getattr(branching, 'mt', None)
            for product, lfs in lfs_map.items():
                if lfs not in PLACEHOLDER_LFS:
                    continue
                info = elis_map.get(product, {}) if isinstance(elis_map, dict) else {}
                z, a = info.get('target_z'), info.get('target_a')
                placeholders.append(dict(
                    parent=parent, mt=mt, reaction=reaction_type, lfs=lfs,
                    description=PLACEHOLDER_LFS[lfs], target_z=z, target_a=a,
                    base_nuclide=_base(product, z, a), elis=info.get('elis'),
                    product=product, method=info.get('method', method),
                    outcome=f"mapped -> {product}"))

    # 2) Unmapped / skipped channels (completeness -- these are NOT written to
    #    the chain, but are reported so every placeholder occurrence is visible).
    for err in (elis_errors or []):
        lfs = err.get('lfs')
        parent = err.get('parent')
        reaction = err.get('reaction')
        if lfs is None and err.get('error'):
            parsed = _parse_no_metastable_decay_data_error(err.get('error', ''))
            lfs = parsed.get('lfs')
            parent = parent or parsed.get('parent')
            reaction = reaction or parsed.get('reaction')
            err = {**parsed, **{k: v for k, v in err.items() if v is not None}}
        if lfs not in PLACEHOLDER_LFS:
            continue
        etype = err.get('type', '')
        outcome = {
            'elis_tol_exceeded':        'ELIS rtol exceeded (unmapped)',
            'no_metastable_decay_data': 'no DK-Lib data (unmapped)',
            'zero_elis_metastables':    'DK-Lib ELIS=0 (unmapped)',
        }.get(etype, (etype or 'unmapped'))
        z, a = err.get('target_z'), err.get('target_a')
        placeholders.append(dict(
            parent=parent, mt=err.get('mt'), reaction=reaction, lfs=lfs,
            description=PLACEHOLDER_LFS[lfs], target_z=z, target_a=a,
            base_nuclide=err.get('base_nuclide') or _base(None, z, a),
            elis=err.get('elis'), product=None, method=method, outcome=outcome))

    for err in (duplicate_mapping_errors or []):
        z, a = err.get('target_z'), err.get('target_a')
        base = err.get('base_nuclide') or _base(None, z, a)
        for d in err.get('discarded', []):
            if d.get('lfs') not in PLACEHOLDER_LFS:
                continue
            placeholders.append(dict(
                parent=err.get('nuclide'), mt=err.get('mt'),
                reaction=err.get('reaction'), lfs=d.get('lfs'),
                description=PLACEHOLDER_LFS[d.get('lfs')], target_z=z, target_a=a,
                base_nuclide=base, elis=d.get('elis'), product=None,
                method=method,
                outcome=f"duplicate LFS discarded (kept LFS={err.get('kept_lfs')})"))
        if err.get('kept_lfs') in PLACEHOLDER_LFS:
            placeholders.append(dict(
                parent=err.get('nuclide'), mt=err.get('mt'),
                reaction=err.get('reaction'), lfs=err.get('kept_lfs'),
                description=PLACEHOLDER_LFS[err.get('kept_lfs')], target_z=z,
                target_a=a, base_nuclide=base, elis=err.get('kept_elis'),
                product=None, method=method,
                outcome=f"duplicate resolved: kept -> _m{err.get('liso')}"))

    for err in (lfs_order_dropped or []):
        lfs = err.get('lfs')
        if lfs not in PLACEHOLDER_LFS:
            continue
        z, a = err.get('target_z'), err.get('target_a')
        placeholders.append(dict(
            parent=err.get('parent'), mt=err.get('mt'),
            reaction=err.get('reaction'), lfs=lfs,
            description=PLACEHOLDER_LFS[lfs], target_z=z, target_a=a,
            base_nuclide=err.get('base_nuclide') or _base(None, z, a),
            elis=err.get('gendf_elis'), product=None, method='lfs_order',
            outcome='lfs_order dropped (unmapped)'))

    return placeholders


def _print_lfs_placeholder_warning(placeholders, mode):
    """Loud console warning + per-value breakdown when LFS placeholders were seen.

    Report-only: no mapping decision or chain output depends on this. Under
    'elis' mapping the placeholder rows are matched on their real GENDF-ELFS
    excitation energy and are SAFE; the risk is only a downstream POSITIONAL
    consumer (FISPACT-parity lfs_order tooling) that would mis-name them
    ``_m99`` / ``_m40``. Under 'lfs_order' the tool IS doing positional naming,
    so the warning is emphatic.
    """
    if not placeholders:
        return
    by_val = Counter(s['lfs'] for s in placeholders)
    bar = "!" * 70
    print("\n" + bar)
    print(f"WARNING: {len(placeholders)} LFS PLACEHOLDER occurrence(s) detected "
          "(unspecified-level isomer tags)")
    print(bar)
    for val in sorted(by_val):
        print(f"  LFS={val:<3d} x{by_val[val]:<4d} {PLACEHOLDER_LFS[val]}")
    print("  Placeholder LFS values are NOT level ordinals; they must never be")
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
    print("  See the 'LFS PLACEHOLDER VALUES' section of the mapping log for "
          "every occurrence.")
    print(bar)


def _write_lfs_placeholder_section(f, placeholders):
    """LFS PLACEHOLDER VALUES section: every GENDF partial carrying a placeholder LFS.

    Report-only. Lists all occurrences (mapped + unmapped, all nuclides/targets)
    so a placeholder LFS is never silently consumed as an isomer ordinal. Printed
    unconditionally (mirrors the other log sections), with a 'none found' line
    when empty.
    """
    f.write("\n\n" + "=" * 220 + "\n")
    f.write('LFS PLACEHOLDER VALUES (unidentified excited states)\n')
    f.write("=" * 220 + "\n\n")
    f.write("Some evaluations tag a product whose final-state LEVEL could not "
            "be resolved with a PLACEHOLDER LFS instead of a true level index:\n")
    for val, desc in sorted(PLACEHOLDER_LFS.items()):
        f.write(f"    LFS={val:<3d} = {desc}\n")
    f.write("These placeholders are NOT level ordinals. ELIS mapping (this tool's "
            "default) is unaffected -- it matches the real GENDF-ELFS "
            "excitation energy (QM - QI), so the\n")
    f.write("CHAIN-Product carries the correct _m<liso>. A positional/lfs_order "
            "consumer, however, would mis-name these rows _m99 / _m40. They are "
            "reported here so such misuse is\n")
    f.write("caught; mapping decisions and the output chain XML are "
            "UNCHANGED.\n\n")
    if not placeholders:
        f.write("No LFS placeholder values found.\n")
        return
    by_val = Counter(s['lfs'] for s in placeholders)
    breakdown = ", ".join(f"LFS={v}: {by_val[v]}" for v in sorted(by_val))
    f.write(f"Total placeholder occurrences: {len(placeholders)}  ({breakdown})\n\n")
    header = (f"{'Parent':<12}  {'MT':>5}  {'Reaction':<12}  {'LFS':>4}  "
              f"{'Convention':<22}  {'Target':<12}  {'GENDF-ELFS[eV]':>14}  "
              f"{'Method':<10}  {'Outcome':<46}")
    sep = "-" * len(header)
    f.write(header + "\n" + sep + "\n")

    def _k(x):
        return (str(x.get('parent') or ''), x.get('mt') or 0, x.get('lfs') or 0)

    for s in sorted(placeholders, key=_k):
        elis = s.get('elis')
        elis_str = f"{elis:.1f}" if isinstance(elis, (int, float)) else "N/A"
        conv = (s.get('description') or '').split(':')[0]
        mt = s.get('mt')
        mt_str = str(mt) if mt is not None else "?"
        f.write(f"{str(s.get('parent') or '?'):<12}  {mt_str:>5}  "
                f"{str(s.get('reaction') or '?'):<12}  {s['lfs']:>4}  "
                f"{conv:<22}  {str(s.get('base_nuclide') or '?'):<12}  "
                f"{elis_str:>14}  {str(s.get('method') or ''):<10}  "
                f"{str(s.get('outcome') or ''):<46}\n")


def _check_isomeric_state_proximity(isomer_mappings, rtol):
    """
    Check if different isomeric states of the same nuclide have ELIS values
    that are too close together (within rtol of each other).

    Parameters
    ----------
    isomer_mappings : list of dict
        Isomer mapping entries with 'product', 'elis', 'dk_elis', 'liso' fields
    rtol : float
        Relative tolerance threshold (e.g., 0.5 for 50%)

    Returns
    -------
    dict with 'gendf_overlaps', 'dk_overlaps' lists, each containing:
        {'base': str, 'state1': str, 'elis1': float, 'state2': str, 'elis2': float,
         'rel_diff': float, 'is_overlap': bool}
    """
    from itertools import combinations

    # Group by base nuclide
    by_base = defaultdict(list)
    for m in isomer_mappings:
        product = m.get('product', '')
        if '_m' in product:
            base = product.split('_m')[0]
            liso = m.get('liso')
            gendf_elis = m.get('elis')
            dk_elis = m.get('dk_elis')
            if liso is not None:
                by_base[base].append({
                    'liso': liso,
                    'lfs': m.get('lfs'),
                    'state': f"_m{liso}",
                    'gendf_elis': gendf_elis,
                    'dk_elis': dk_elis
                })

    gendf_pairs = []
    dk_pairs = []

    for base, states in by_base.items():
        # Get unique states by liso
        unique_states = {}
        for s in states:
            liso = s['liso']
            if liso not in unique_states:
                unique_states[liso] = s
            else:
                # If we have multiple entries for same liso, prefer one with values
                if s['gendf_elis'] is not None and unique_states[liso]['gendf_elis'] is None:
                    unique_states[liso] = s

        if len(unique_states) < 2:
            continue

        # Check all pairs
        for (liso1, s1), (liso2, s2) in combinations(unique_states.items(), 2):
            # GENDF-ELFS check
            e1, e2 = s1['gendf_elis'], s2['gendf_elis']
            if e1 is not None and e2 is not None and max(e1, e2) > 0:
                rel_diff = abs(e1 - e2) / max(e1, e2)
                gendf_pairs.append({
                    'base': base,
                    'state1': s1['state'], 'lfs1': s1.get('lfs'), 'elis1': e1,
                    'state2': s2['state'], 'lfs2': s2.get('lfs'), 'elis2': e2,
                    'rel_diff': rel_diff,
                    'is_overlap': rel_diff < rtol
                })

            # DK-ELIS check
            d1, d2 = s1['dk_elis'], s2['dk_elis']
            if d1 is not None and d2 is not None and max(d1, d2) > 0:
                rel_diff = abs(d1 - d2) / max(d1, d2)
                dk_pairs.append({
                    'base': base,
                    'state1': s1['state'], 'elis1': d1,
                    'state2': s2['state'], 'elis2': d2,
                    'rel_diff': rel_diff,
                    'is_overlap': rel_diff < rtol
                })

    # Sort by rel_diff (smallest first = highest risk)
    gendf_pairs.sort(key=lambda x: x['rel_diff'])
    dk_pairs.sort(key=lambda x: x['rel_diff'])

    return {'gendf_pairs': gendf_pairs, 'dk_pairs': dk_pairs}


def _write_isomeric_proximity_check(f, isomer_mappings, rtol):
    """Write isomeric state proximity check section to log file."""
    results = _check_isomeric_state_proximity(isomer_mappings, rtol)

    f.write("\n\n" + "=" * 220 + "\n")
    f.write("ISOMERIC STATE PROXIMITY CHECK\n")
    f.write("=" * 220 + "\n\n")
    f.write(f"Checking for isomeric states with ELIS values within rtol={rtol*100:.0f}% of each other.\n")
    f.write("This could indicate potential mapping ambiguity.\n\n")

    # GENDF-ELFS check (ELFS = excitation energy from GENDF Q-values: QM - QI)
    gendf_overlaps = [p for p in results['gendf_pairs'] if p['is_overlap']]
    f.write(f"GENDF-ELFS PROXIMITY:\n")
    f.write("-" * 120 + "\n")

    # Helper to format LFS value
    def fmt_lfs(lfs):
        return str(lfs) if lfs is not None else '?'

    header = f"{'Base':<12}  {'LFS1':>5}  {'GENDF-ELFS1':>14}  {'LFS2':>5}  {'GENDF-ELFS2':>14}  {'Rel-Diff':>10}  {'Status':<20}"

    if gendf_overlaps:
        f.write(f"\nPOTENTIAL OVERLAPS DETECTED: {len(gendf_overlaps)}\n\n")
        f.write(header + "\n")
        f.write("-" * 90 + "\n")
        for p in gendf_overlaps:
            status = "OVERLAP" if p['rel_diff'] < rtol * 0.5 else "WARNING: Near rtol"
            f.write(f"{p['base']:<12}  {fmt_lfs(p.get('lfs1')):>5}  {p['elis1']:>14.1f}  {fmt_lfs(p.get('lfs2')):>5}  {p['elis2']:>14.1f}  {p['rel_diff']*100:>9.1f}%  {status:<20}\n")
    else:
        f.write("\nNo overlaps detected. Closest pairs (highest risk):\n\n")
        f.write(header + "\n")
        f.write("-" * 90 + "\n")
        for p in results['gendf_pairs'][:10]:
            f.write(f"{p['base']:<12}  {fmt_lfs(p.get('lfs1')):>5}  {p['elis1']:>14.1f}  {fmt_lfs(p.get('lfs2')):>5}  {p['elis2']:>14.1f}  {p['rel_diff']*100:>9.1f}%  {'OK':<20}\n")
        if not results['gendf_pairs']:
            f.write("  (No multi-state nuclides found)\n")

    # DK-ELIS check (ELIS = excitation energy from decay library)
    dk_overlaps = [p for p in results['dk_pairs'] if p['is_overlap']]
    f.write(f"\n\nDK-ELIS PROXIMITY:\n")
    f.write("-" * 120 + "\n")

    # Helper to strip leading underscore from state name (_m1 -> m1)
    def strip_state(s):
        return s.lstrip('_') if s.startswith('_') else s

    header = f"{'Nuclide':<12}  {'LISO1':<6}  {'DK-ELIS1':>14}  {'LISO2':<6}  {'DK-ELIS2':>14}  {'Rel-Diff':>10}  {'Status':<20}"

    if dk_overlaps:
        f.write(f"\nPOTENTIAL OVERLAPS DETECTED: {len(dk_overlaps)}\n\n")
        f.write(header + "\n")
        f.write("-" * 120 + "\n")
        for p in dk_overlaps:
            status = "OVERLAP" if p['rel_diff'] < rtol * 0.5 else "WARNING: Near rtol"
            f.write(f"{p['base']:<12}  {strip_state(p['state1']):<6}  {p['elis1']:>14.1f}  {strip_state(p['state2']):<6}  {p['elis2']:>14.1f}  {p['rel_diff']*100:>9.1f}%  {status:<20}\n")
    else:
        f.write("\nNo overlaps detected. Closest pairs (highest risk):\n\n")
        f.write(header + "\n")
        f.write("-" * 120 + "\n")
        for p in results['dk_pairs'][:10]:
            f.write(f"{p['base']:<12}  {strip_state(p['state1']):<6}  {p['elis1']:>14.1f}  {strip_state(p['state2']):<6}  {p['elis2']:>14.1f}  {p['rel_diff']*100:>9.1f}%  {'OK':<20}\n")
        if not results['dk_pairs']:
            f.write("  (No multi-state nuclides found)\n")


def _write_duplicate_mapping_section(f, duplicate_mapping_errors):
    """Write duplicate mapping conflicts section to log file.

    When multiple GENDF LFS values map to the same decay library LISO
    (within tolerance), only the closest ELIS match is kept.
    This section shows all such conflicts with full context.
    """
    f.write("\n\n" + "=" * 220 + "\n")
    f.write("DUPLICATE MAPPING CONFLICTS\n")
    f.write("=" * 220 + "\n\n")

    f.write("When multiple GENDF LFS values have ELIS within tolerance of the same decay library LISO,\n")
    f.write("only the closest match is kept. Discarded LFS values are shown below with full context.\n\n")

    if not duplicate_mapping_errors:
        f.write("No duplicate mapping conflicts detected.\n")
        return

    f.write(f"Total conflicts: {len(duplicate_mapping_errors)}\n")
    f.write(f"Total LFS discarded: {sum(len(e.get('discarded', [])) for e in duplicate_mapping_errors)}\n\n")

    for err in duplicate_mapping_errors:
        nuclide = err.get('nuclide', '?')
        reaction = err.get('reaction', '?')
        mt = err.get('mt', '?')
        liso = err.get('liso', '?')
        base_nuc = err.get('base_nuclide', '?')
        kept_lfs = err.get('kept_lfs')
        kept_elis = err.get('kept_elis', 0)
        kept_diff = err.get('kept_diff', 0)
        dk_elis = err.get('dk_elis', 0)

        f.write("-" * 120 + "\n")
        f.write(f"{nuclide} {reaction} (MT={mt}) -> {base_nuc}_m{liso}\n")
        f.write("-" * 120 + "\n\n")

        # Resolution
        f.write(f"  RESOLUTION: Kept LFS={kept_lfs} (ELIS={kept_elis:.0f}eV, diff={kept_diff:.0f}eV)\n")
        f.write(f"              DK-ELIS for _m{liso}: {dk_elis:.0f}eV\n\n")

        # Discarded
        discarded = err.get('discarded', [])
        f.write(f"  DISCARDED ({len(discarded)} LFS):\n")
        for d in discarded:
            f.write(f"    LFS={d['lfs']}: ELIS={d['elis']:.0f}eV (diff={d['diff']:.0f}eV)\n")
        f.write("\n")

        # All GENDF LFS levels for this reaction
        gendf_all = err.get('gendf_all_lfs', [])
        discarded_lfs_set = {d['lfs'] for d in discarded}
        f.write(f"  ALL GENDF LFS LEVELS (this reaction):\n")
        f.write(f"    {'LFS':>5}  {'ELIS[eV]':>14}  {'Status':<20}\n")
        for g in gendf_all:
            elis_val = g.get('elis')
            elis_str = f"{elis_val:.0f}" if elis_val is not None else "N/A"
            if g['lfs'] in discarded_lfs_set:
                status = "****not-mapped****"
            elif g['lfs'] == kept_lfs:
                status = f"-> _m{liso} (kept)"
            elif g['lfs'] == 0:
                status = "-> ground"
            else:
                status = ""
            f.write(f"    {g['lfs']:>5}  {elis_str:>14}  {status:<20}\n")
        f.write("\n")

        # All decay library LISO levels for this (Z, A)
        decay_all = err.get('decay_all_liso', [])
        target_z = err.get('target_z', '?')
        target_a = err.get('target_a', '?')
        f.write(f"  ALL DECAY LIBRARY LISO LEVELS (Z={target_z}, A={target_a}):\n")
        f.write(f"    {'LISO':>5}  {'ELIS[eV]':>14}  {'T1/2[s]':>14}  {'Status':<20}\n")
        for d in decay_all:
            half_life = d.get('half_life')
            if isinstance(half_life, (int, float)):
                hl_str = f"{half_life:.3e}"
            else:
                hl_str = str(half_life) if half_life else "stable"
            if d['liso'] == liso:
                status = "<- mapped (conflict resolved)"
            else:
                status = ""
            f.write(f"    {d['liso']:>5}  {d['elis']:>14.0f}  {hl_str:>14}  {status:<20}\n")
        f.write("\n")


# =============================================================================
# Single-target reason detection
# =============================================================================

def _determine_single_target_reason(branching, valid_products, missing_products, all_products):
    """
    Determine why a reaction ended up with only a single target.

    Parameters
    ----------
    branching : IsomericBranching
        The branching data object from GENDF
    valid_products : list
        Products that passed chain validation
    missing_products : list
        Products that were filtered out (not in chain)
    all_products : list
        All original products from branching

    Returns
    -------
    tuple
        (SingleTargetReason, detail_string)
    """
    # Check if GENDF only had one LFS to begin with
    if branching.lfs_mapping and len(branching.lfs_mapping) == 1:
        return SingleTargetReason.GENDF_SINGLE_LFS, "GENDF MF=10 has only one LFS value"

    # Check ELIS mapping errors if available
    if hasattr(branching, 'elis_mapping') and branching.elis_mapping:
        errors = branching.elis_mapping.get('errors', [])
        if errors:
            error_types = {e.get('error_type') for e in errors if e.get('error_type')}

            if 'elis_tol_exceeded' in error_types:
                details = [e for e in errors if e.get('error_type') == 'elis_tol_exceeded']
                return SingleTargetReason.ELIS_TOL_EXCEEDED, f"ELIS tolerance exceeded for {len(details)} product(s)"

            if 'no_metastable_decay_data' in error_types:
                details = [e for e in errors if e.get('error_type') == 'no_metastable_decay_data']
                return SingleTargetReason.NO_DECAY_DATA, f"No decay library data for {len(details)} product(s)"

            if 'zero_elis_metastables' in error_types:
                return SingleTargetReason.ZERO_ELIS_DECAY, "Decay library has no metastable states for this product"

            if 'duplicate_mapping' in error_types:
                return SingleTargetReason.DUPLICATE_MAPPING, "Multiple LFS values mapped to same product"

            if 'lfs_order_dropped' in error_types:
                details = [e for e in errors if e.get('error_type') == 'lfs_order_dropped']
                return SingleTargetReason.LFS_ORDER_DROPPED, f"LFS-order mode dropped {len(details)} metastable(s)"

    # Check if products were filtered out (not in chain)
    if missing_products:
        return SingleTargetReason.PRODUCTS_NOT_IN_CHAIN, f"Filtered out: {', '.join(missing_products)}"

    return SingleTargetReason.UNKNOWN, "Unknown reason (single product but no error recorded)"


# =============================================================================
# XML manipulation
# =============================================================================

def _q_str(value):
    """Format a Q value (eV) for the chain XML at 4 dp.

    Values come straight off the MF=10 QM/QI heads (or, in the ELIS fallback,
    from ``Q_ground - ELFS``, a float subtraction that can leave e.g.
    ``-13286354.999999998``). Rounding to 4 dp normalises both -- the clean
    ``-13286355.0`` -- while preserving genuine sub-eV level energies. Q is a
    heating quantity (never enters the transmutation matrix), so this precision
    is ample.
    """
    return str(round(float(value), 4))


def _q_lists_differ(new, old):
    """True if two Q token lists differ NUMERICALLY (not just in formatting).

    Both sides are produced by ``_q_str`` or copied verbatim from the chain's
    scalar Q attribute, so the same value can reach them in two spellings
    (``1e5`` vs ``100000.0``) and fabricate a correction row. Non-numeric
    tokens (a malformed chain Q) fall back to the string compare.
    """
    try:
        return [float(v) for v in new] != [float(v) for v in old]
    except ValueError:
        return list(new) != list(old)


def _attribution_line(record):
    """One compact per-reaction line for the IZAP=0 classification report."""
    head = (f"  {record['status'].upper():<14} {record['parent']:<10} "
            f"{record['reaction'] or '?':<12} MT={record['mt']:<4} "
            f"{record['n_anonymous']}/{record['n_levels']} anonymous")
    if record['status'] == 'reattributed':
        body = "; ".join(f"LFS={r['lfs']}->IZAP={r['izap']} [{r['evidence']}]"
                         for r in record['recovered'])
    elif record['failed']:
        body = "; ".join(f"LFS={f['lfs']} {f['condition']} FAILED: {f['detail']}"
                         for f in record['failed'])
    elif record['status'] == 'anon_single':
        body = "single final state: no isomeric decoration either way"
    else:
        body = "flag off (--reattribute-mf10-noIZAP recovers it)"
    return f"{head}  {body}"


def _print_attribution_summary(records, counts, reattribute):
    """Loud console block for the anonymous (IZAP=0) MF=10 classification.

    Printed whether or not --reattribute-mf10-noIZAP is on, so a silently lost
    isomeric decoration is impossible. Excluded rows (MT=5/18 -- never decorated)
    are counted only; the per-reaction lines cover the decoration-relevant ones.
    """
    if not records:
        print("  No anonymous (IZAP=0) MF=10 subsections found.")
        return
    bar = "!" * 70
    print("\n" + bar)
    print(f"WARNING: {len(records)} MF=10 section(s) carry anonymous (IZAP=0) "
          f"subsections ({counts['anonymous_levels']} level(s))")
    print(bar)
    print(f"  Fully attributed sections:           {counts['attributed']:6d}")
    print(f"  Re-attributed (gate passed):         {counts['reattributed']:6d}")
    print(f"  Pruned (no isomeric decoration):     {counts['pruned']:6d}")
    print(f"  Excluded (MT=5/18, never decorated): {counts['excluded']:6d}")
    print(f"  Anonymous single-level sections:     {counts['anon_single']:6d}")
    if counts['load_error']:
        print(f"  Load errors (endf crashers):         {counts['load_error']:6d}")
    if not reattribute and counts['pruned']:
        print("  Re-attribution is OFF: each pruned reaction keeps its plain "
              "MF=3 route to the")
        print("  default target and loses ALL isomeric branching "
              "(--reattribute-mf10-noIZAP).")
    for record in sorted(records, key=lambda r: (r['parent'], r['mt'])):
        if record['status'] != 'excluded':
            print(_attribution_line(record))
    print(bar)


def _write_attribution_section(f, records, counts, reattribute):
    """ANONYMOUS (IZAP=0) MF=10 SUBSECTIONS section of the mapping log."""
    f.write("\n\n" + "=" * 220 + "\n")
    f.write("ANONYMOUS (IZAP=0) MF=10 SUBSECTIONS\n")
    f.write("=" * 220 + "\n\n")
    f.write("An MF=10 subsection written with IZAP=0 names no product nuclide, "
            "yet the level stays keyed by LFS. --reattribute-mf10-noIZAP "
            "re-derives the residual from the\n")
    f.write("reaction's deterministic (dA, dZ) shift under a four-part "
            "evidence gate: C1 deterministic-residual depletion MT, C2 valid "
            "and section-unique LFS, C3 derived product present in the\n")
    f.write("decay library, C4 Q consistency (QM == QI within "
            f"{REATTRIB_Q_TOL_GROUND_EV:.1f} eV for LFS=0; QM-QI equal "
            f"to a decay level's ELIS within {REATTRIB_Q_TOL_EV/1e3:.1f} keV "
            "for LFS>0 -- level identity comes from ELIS, never from an "
            "LFS<->m-number\n")
    f.write("assumption). With the flag OFF, or when ANY anonymous subsection "
            "of a reaction fails the gate, that reaction's ENTIRE isomeric "
            "decoration is pruned (all-or-nothing) and the plain\n")
    f.write("MF=3 route to the default target is kept: decorating the "
            "attributed subset alone would hand 100% of the reaction rate to "
            "the metastable.\n")
    f.write("Note: --suppress-single-target-yields classifications can flip "
            "single-target to two-target on a re-attributed reaction; that is "
            "expected.\n\n")
    f.write(f"Re-attribution: {'ON' if reattribute else 'OFF'}\n")
    f.write(f"  Fully attributed sections:           {counts['attributed']:6d}\n")
    f.write(f"  Re-attributed (gate passed):         {counts['reattributed']:6d}\n")
    f.write(f"  Pruned (no isomeric decoration):     {counts['pruned']:6d}\n")
    f.write(f"  Excluded (MT=5/18, never decorated): {counts['excluded']:6d}\n")
    f.write(f"  Anonymous single-level sections:     {counts['anon_single']:6d}\n")
    f.write(f"  Anonymous levels seen:               {counts['anonymous_levels']:6d}\n")
    f.write(f"  Load errors (endf crashers):         {counts['load_error']:6d}\n\n")
    if not records:
        f.write("No anonymous (IZAP=0) MF=10 subsections found.\n")
        return
    for record in sorted(records, key=lambda r: (r['parent'], r['mt'])):
        f.write(_attribution_line(record) + "\n")


def _write_q55_ground_head_section(f, rows, counts):
    """MF=10 LFS=0 GROUND-HEAD Q CONSISTENCY (Q-55) section of the mapping log."""
    rows = rows or []
    counts = counts or Counter()
    f.write("\n\n" + "=" * 220 + "\n")
    f.write("MF=10 LFS=0 GROUND-HEAD Q CONSISTENCY (Q-55)\n")
    f.write("=" * 220 + "\n\n")
    f.write("A ground product's level energy is zero by definition, so an "
            "MF=10 LFS=0 TAB1 head must carry QM == QI (ELFS = QM - QI = 0). "
            "TENDL-2019 and JEFF-3.3 instead write\n")
    f.write("the reaction QI on ground-target tapes (QM=0, QI=-E(level)) or a "
            "blank QI on isomer-target tapes (QI=0, QM=+ELIS) -- either way "
            "the ground head reads as excited.\n")
    f.write("Report-only: no code path consumes an LFS=0 QI (every "
            "ground-route Q write is QM-sourced), so mapping decisions and "
            "the output chain XML are UNCHANGED. Rows are\n")
    f.write("listed so a defective library announces itself instead of "
            "carrying the defect silently into a future consumer.\n\n")
    f.write(f"  MF=10 sections scanned:              {counts['sections']:6d}\n")
    f.write(f"  LFS=0 heads with QM and QI:          {counts['lfs0_heads']:6d}\n")
    f.write(f"  LFS=0 heads missing QM or QI:        {counts['q_missing']:6d}\n")
    f.write(f"  Violations (|QM - QI| > "
            f"{REATTRIB_Q_TOL_GROUND_EV:.1f} eV):     {len(rows):6d}"
            f"  (ground-target: {counts['ground-target']}, "
            f"blank-QI: {counts['blank-QI']}, other: {counts['other']})\n")
    f.write(f"  Load errors (endf crashers):         {counts['load_error']:6d}\n\n")
    if not rows:
        f.write("No violating LFS=0 ground heads found.\n")
        return
    devs = sorted(abs(r['elfs']) for r in rows)
    f.write(f"Deviation span: {devs[0]:.2f} eV to {devs[-1]:.2f} eV\n\n")
    header = (f"{'Parent':<12}  {'MT':>5}  {'Reaction':<12}  {'IZAP':>6}  "
              f"{'QM[eV]':>14}  {'QI[eV]':>14}  {'QM-QI[eV]':>14}  "
              f"{'Shape':<14}")
    f.write(header + "\n" + "-" * len(header) + "\n")
    for r in sorted(rows, key=lambda x: (x['parent'], x['mt'])):
        f.write(f"{r['parent']:<12}  {r['mt']:>5}  "
                f"{str(r['reaction'] or '?'):<12}  {r['izap']:>6}  "
                f"{r['qm']:>14.4f}  {r['qi']:>14.4f}  {r['elfs']:>14.4f}  "
                f"{r['shape']:<14}\n")


def _print_emission_summary(summary, output_file):
    """Console log section + reconciliation for the MF=10-only emission passes.

    Called after BOTH the ground pre-pass and the metastable-direct post-pass, so
    every deferred channel has terminated in exactly one counter.
    """
    print("\n" + "=" * 60)
    print("MF=10-ONLY EMISSION (--emit-mf10-only-reactions)")
    print("=" * 60)
    print(f"  Emitted ground reactions:        {summary['emitted']:6d}")
    print(f"    of which ground-only (NS=1):   {summary['emit_ground_only']:6d}")
    print(f"  Emitted metastable-direct (post):{summary['emitted_metastable_direct']:6d}")
    print(f"  Skipped (no MT_TO_REACTION):     {summary['emit_skipped_no_name']:6d}")
    print(f"  Skipped (target not in chain):   {summary['emit_skipped_no_target']:6d}")
    print(f"  Skipped (orphan-added target):   {summary.get('emit_skipped_orphan_target', 0):6d}")
    print(f"  Skipped (multi-metastable):      {summary['emit_skipped_multi_metastable']:6d}")
    print(f"  Skipped (unrepaired IZAP=0):     {summary['emit_skipped_anonymous']:6d}")
    print(f"  Skipped (type already present):  {summary['emit_skipped_exists']:6d}")
    print(f"  Load errors (endf crashers):     {summary['emit_skipped_load_error']:6d}")
    print(f"  Anonymous (IZAP=0) levels left:  {summary['emit_anonymous_levels']:6d}")
    examined = (summary['emitted'] + summary['emitted_metastable_direct']
                + summary['emit_skipped_no_name']
                + summary['emit_skipped_no_target']
                + summary.get('emit_skipped_orphan_target', 0)
                + summary['emit_skipped_multi_metastable']
                + summary['emit_skipped_anonymous']
                + summary['emit_skipped_exists'])
    print("-" * 60)
    print(f"  Reconciliation: examined MF=10-only MTs = "
          f"emitted + skips = {examined}")
    print(f"    = {summary['emitted']} ground + {summary['emitted_metastable_direct']} meta-direct "
          f"+ {summary['emit_skipped_no_name']} no-name "
          f"+ {summary['emit_skipped_no_target']} no-target "
          f"+ {summary.get('emit_skipped_orphan_target', 0)} orphan-target "
          f"+ {summary['emit_skipped_multi_metastable']} multi-metastable "
          f"+ {summary['emit_skipped_anonymous']} unrepaired-IZAP=0 "
          f"+ {summary['emit_skipped_exists']} already-present "
          f"(load-error nuclides not examined)")
    if summary['load_error_nuclides']:
        names = ', '.join(n for n, _ in summary['load_error_nuclides'])
        print(f"  endf-parser crashers ({len(summary['load_error_nuclides'])}): {names}")
    print(f"  Output chain written:   {output_file}")
    if summary['emitted_details']:
        print("\n  Ground-emitted (parent, type, target, Q):")
        for parent, rtype, target, q in summary['emitted_details']:
            print(f"    {parent:<10} {rtype:<12} -> {target:<10} Q={q}")
    if summary['metastable_direct_details']:
        print("\n  Metastable-direct post-pass (parent, type, target, Q):")
        for parent, rtype, target, q in summary['metastable_direct_details']:
            print(f"    {parent:<10} {rtype:<12} -> {target:<10} Q={q}")
    if summary['skipped_multi_metastable_details']:
        print("\n  Not emitted -- multi-metastable, no LFS=0 ground (parent, MT, LFS levels):")
        for parent, mt, lfs_levels in summary['skipped_multi_metastable_details']:
            print(f"    {parent:<10} MT={mt:<4} LFS={lfs_levels}")
    if summary.get('skipped_orphan_target_details'):
        print("\n  Not emitted -- target exists only as an orphan state added "
              "by --orphan-policy add-stable (parent, MT, target):")
        for parent, mt, target in summary['skipped_orphan_target_details']:
            print(f"    {parent:<10} MT={mt:<4} -> {target}")
    if summary['skipped_anonymous_details']:
        print("\n  Not emitted -- unrepaired anonymous (IZAP=0) level, ground "
              "unnamed (parent, MT):")
        for parent, mt in summary['skipped_anonymous_details']:
            print(f"    {parent:<10} MT={mt:<4}  "
                  "(--reattribute-mf10-noIZAP recovers it)")


def emit_mf10_only_prepass(lib, chain, base_chain_file, output_base_file,
                           verbose=True, unrepaired_anonymous=None):
    """Emit plain <reaction> elements for GENDF MF=10-only channels (D2/D3).

    EAF-2010 stores any reaction whose residual has a tabulated isomer ONLY in
    MF=8/10 (no MF=3), so ``Chain.from_endf`` never harvested them. For each
    chain nuclide with a GENDF file, scan its full-parser material for MTs
    present in MF=10 but absent from MF=3 and append a plain reaction (target =
    ground residual from IZAP, Q = the LFS=0 subsection's QM in eV). The
    augmented base chain is written to ``output_base_file``; the caller reloads
    the Chain from it so Gate 1 (chain object) and the XML writer (tree) stay
    consistent (D4 single source of truth).

    ``unrepaired_anonymous`` is the scan's ``{(parent, mt)}`` set of sections
    whose IZAP=0 levels stay unnamed (R1-61); those sections never reach the
    metastable-direct deferral.
    """
    unrepaired = unrepaired_anonymous or set()
    tree = ET.parse(base_chain_file)
    root = tree.getroot()
    nuclide_elems = {n.get('name'): n for n in root.findall('nuclide')}
    chain_names = {nuc.name for nuc in chain.nuclides}   # tool:1272-1273 membership
    available = lib.available_nuclides_set()

    summary = {
        'emitted': 0,                       # ground pre-pass emissions
        'emitted_metastable_direct': 0,     # post-pass emissions (filled later)
        'emit_skipped_no_name': 0,
        'emit_skipped_no_target': 0,
        # Post-pass only: the deferred target is a state the writer ADDED under
        # --orphan-policy add-stable (a placeholder ordinal, not a decay-library
        # state), so no static reaction is pointed at it.
        'emit_skipped_orphan_target': 0,
        'skipped_orphan_target_details': [],
        'emit_skipped_load_error': 0,
        'emit_ground_only': 0,
        'emit_skipped_exists': 0,     # idempotency: reaction type already present
        # metastable-only MF=10 (no LFS=0 subsection) with >1 metastable level:
        # would need branching between metastables -> not emitted.
        'emit_skipped_multi_metastable': 0,
        # Metastable-only channels whose section ALSO carries an unrepaired
        # anonymous level: the ground is unnamed, not absent, so a static
        # metastable emission would hand it 100% of the rate (R1-61).
        'emit_skipped_anonymous': 0,
        # Anonymous (IZAP=0) levels still unnamed at this point: the
        # re-attribution pass either recovered them (flag on + gate passed, so
        # they never reach here) or left them excluded. Counted, never silent.
        'emit_anonymous_levels': 0,
        'anonymous_details': [],
        'emitted_details': [],
        'metastable_direct_details': [],    # post-pass (parent, type, target, Q)
        # Single-metastable no-ground channels deferred to the post-pass (D2/D3):
        # one final state => no branching decoration needed.
        'metastable_deferred': [],
        'load_error_nuclides': [],
        'skipped_no_name_details': [],
        'skipped_no_target_details': [],
        'skipped_multi_metastable_details': [],
        'skipped_anonymous_details': [],
    }

    for nuc_name, nuc_elem in nuclide_elems.items():
        if nuc_name not in available:
            continue
        # Full parser needed for MF=10; ~97 of 816 EAF files crash the endf
        # int_endf('') bug -- absorb, count, and keep going (never abort the run).
        try:
            material = lib._load_material(nuc_name, require_full_parser=True)
        except Exception as exc:
            summary['emit_skipped_load_error'] += 1
            summary['load_error_nuclides'].append((nuc_name, type(exc).__name__))
            continue

        keys = material.section_data.keys()
        mf3_mts = {mt for (mf, mt) in keys if mf == 3}
        mf10_mts = {mt for (mf, mt) in keys if mf == 10}
        existing_types = {rx.get('type') for rx in nuc_elem.findall('reaction')}

        for mt in sorted(mf10_mts - mf3_mts):
            name = MT_TO_REACTION.get(mt)
            if name is None:
                summary['emit_skipped_no_name'] += 1
                summary['skipped_no_name_details'].append((nuc_name, mt))
                continue
            if name in existing_types:   # re-run / patched-chain idempotency
                summary['emit_skipped_exists'] += 1
                continue

            levels = material.section_data[10, mt].get('levels', [])
            valid = [lv for lv in levels if int(lv.get('IZAP', 0)) != 0]
            if len(valid) != len(levels):
                n_anon = len(levels) - len(valid)
                summary['emit_anonymous_levels'] += n_anon
                summary['anonymous_details'].append((nuc_name, mt, n_anon))
            if not valid:
                summary['emit_skipped_no_target'] += 1
                summary['skipped_no_target_details'].append((nuc_name, mt, None))
                continue

            ground = next((lv for lv in valid if int(lv.get('LFS', 0)) == 0), None)
            if ground is None:
                # Metastable-only MF=10 (no LFS=0 ground subsection; every EAF
                # case is MT=4 (n,n')->X_m1). A single metastable final state has
                # NO branching, so decoration is unnecessary: with no MF=3
                # section the ground cannot be synthesized either, and
                # get_branching_ratios classifies the channel as a skip
                # ('mf10_metastable_only_no_mf3'). Defer to the post-pass,
                # which emits a plain static-target reaction AFTER the pipeline so
                # Gate 1 never sees it. >1 metastable would need branching between
                # metastables -> not emitted.
                #
                # R1-61 all-or-nothing: with an unrepaired anonymous level in
                # the section the LFS=0 ground is unnamed rather than absent,
                # so `valid` is a metastable-only SUBSET of the real final
                # states. Emitting it as a plain static target would give the
                # metastable 100% of the rate -- the inversion this guards.
                # Checked first: it is the reason that survives re-attribution.
                if (nuc_name, mt) in unrepaired:
                    summary['emit_skipped_anonymous'] += 1
                    summary['skipped_anonymous_details'].append((nuc_name, mt))
                    continue
                if len(valid) > 1:
                    summary['emit_skipped_multi_metastable'] += 1
                    summary['skipped_multi_metastable_details'].append(
                        (nuc_name, mt, [int(lv.get('LFS', 0)) for lv in valid]))
                    continue
                lv = valid[0]
                summary['metastable_deferred'].append({
                    'nuclide': nuc_name, 'mt': mt,
                    'izap': int(lv['IZAP']), 'lfs': int(lv.get('LFS', 0)),
                    'qi': lv.get('QI', lv.get('QM')),
                })
                continue
            # Ground residual by construction (LFS=0), so this membership test
            # can never involve an orphan name: those are metastable ordinals
            # minted later, in the writer, under --orphan-policy add-stable.
            # The metastable-direct post-pass, which CAN meet one, screens for
            # them explicitly.
            target = get_product_name(int(ground['IZAP']), 0)
            if target is None or target not in chain_names:
                summary['emit_skipped_no_target'] += 1
                summary['skipped_no_target_details'].append((nuc_name, mt, target))
                continue

            # NS=1 (ground-only) MTs are emitted too; they simply never gain a
            # branching child later.
            has_meta = any(int(lv.get('LFS', 0)) != 0 for lv in valid)
            if not has_meta:
                summary['emit_ground_only'] += 1

            q_str = _q_str(ground['QM'])
            rx = ET.SubElement(nuc_elem, 'reaction')
            rx.set('type', name)
            rx.set('Q', q_str)
            rx.set('target', target)
            existing_types.add(name)
            summary['emitted'] += 1
            summary['emitted_details'].append((nuc_name, name, target, q_str))

    tree.write(output_base_file)

    # The summary is NOT printed here: metastable-direct emissions happen in the
    # post-pass (emit_metastable_direct_postpass), which runs after the branching
    # pipeline. main() prints the unified summary once both passes are done.
    return summary


def emit_metastable_direct_postpass(root, emit_summary, orphan_names=None):
    """Emit plain static-target reactions for single-metastable MF=10-only
    channels deferred by the pre-pass (D2/D3).

    A metastable-only channel (no LFS=0 subsection) has exactly ONE final state,
    so it needs no branching decoration. This runs AFTER the branching pipeline
    and XML decoration but BEFORE the reactions= recount and the final write, so
    Gate 1 (which sees only the in-memory chain) never encounters these reactions
    and cannot ask for a branching the file cannot supply. Mutates the writer tree
    ``root`` in place and updates ``emit_summary`` counters; guards (name, target
    membership, idempotency) mirror the pre-pass but check the tree being
    mutated so re-runs on already-patched chains stay safe.

    ``orphan_names`` are the states the writer has just ADDED under
    ``--orphan-policy add-stable``. They are placeholder ordinals, not decay
    library states, so they are not legitimate targets for a static emission:
    the channel is skipped and reported rather than pointed at a name that
    exists only because another reaction could not identify its product.
    """
    nuclide_map = {n.get('name'): n for n in root.findall('nuclide')}
    orphan_names = orphan_names or set()
    chain_names = set(nuclide_map) - set(orphan_names)
    emit_summary.setdefault('emit_skipped_orphan_target', 0)
    emit_summary.setdefault('skipped_orphan_target_details', [])

    for entry in emit_summary['metastable_deferred']:
        nuc_elem = nuclide_map.get(entry['nuclide'])
        if nuc_elem is None:
            emit_summary['emit_skipped_no_target'] += 1
            continue
        name = MT_TO_REACTION.get(entry['mt'])
        if name is None:
            emit_summary['emit_skipped_no_name'] += 1
            continue
        # Target is the metastable product itself (e.g. In115(n,n')->In115_m1).
        target = get_product_name(entry['izap'], entry['lfs'])
        if target is not None and target in orphan_names:
            emit_summary['emit_skipped_orphan_target'] += 1
            emit_summary['skipped_orphan_target_details'].append(
                (entry['nuclide'], entry['mt'], target))
            continue
        if target is None or target not in chain_names:
            emit_summary['emit_skipped_no_target'] += 1
            continue
        # Idempotency against the WRITER tree at post-pass time.
        existing = {rx.get('type') for rx in nuc_elem.findall('reaction')}
        if name in existing:
            emit_summary['emit_skipped_exists'] += 1
            continue
        # Q = the metastable subsection's QI (state-corrected Q, identical to what
        # a decorated child would carry: Q_meta = Q_ground - (QM - QI) = QI).
        q_str = _q_str(entry['qi'])
        rx = ET.SubElement(nuc_elem, 'reaction')
        rx.set('type', name)
        rx.set('Q', q_str)
        rx.set('target', target)
        emit_summary['emitted_metastable_direct'] += 1
        emit_summary['metastable_direct_details'].append(
            (entry['nuclide'], name, target, q_str))

    return emit_summary


def _add_stable_nuclide_element(root, nuclide_map, chain, chain_names, name,
                                summary, source):
    """Add an orphan product to the output chain as a stable nuclide.

    ``<nuclide name="X_m3" reactions="0"/>`` carries no decay data, so the
    chain reader treats it as STABLE: the branch is kept and its mass is
    conserved, but the state's own activity is not modelled. Idempotent by
    name -- several parents feeding the same unidentified state share one
    nuclide and each records its own source row. The element is inserted after
    the last sibling of the same base nuclide so the family stays grouped.
    Registration in ``nuclide_map`` / ``chain_names`` / ``chain`` makes the new
    name visible to every later reaction of this run.
    """
    summary['orphan_nuclides_added'].setdefault(name, []).append(source)
    if name in chain_names or name in nuclide_map:
        chain_names.add(name)
        return False

    elem = ET.Element('nuclide')
    elem.set('name', name)
    elem.set('reactions', '0')
    base = name.split('_')[0]
    children = list(root)
    insert_at = len(children)
    for i, child in enumerate(children):
        if child.tag == 'nuclide' and child.get('name', '').split('_')[0] == base:
            insert_at = i + 1
    root.insert(insert_at, elem)

    nuclide_map[name] = elem
    chain_names.add(name)
    chain.add_nuclide(Nuclide(name))
    return True


def _reattribution_recipient(orphan, valid_products, elis_map, lfs_map):
    """Pick the kept isomer an orphan branch folds DOWN into.

    The recipient is the kept isomeric sibling of the same product at the
    nearest LOWER rank, ground when nothing is below it. Rank -- not energy --
    decides: an orphan only reaches this point because its excitation energy
    failed to identify it, so that energy is exactly the datum not to trust
    again, and it is identically zero in the EAF/SCALE evaluations. A level
    high in the band cascades DOWN into the isomer below it, never up.

    Returns ``(recipient, siblings)`` where ``siblings`` lists every kept
    isomer of the same product with its rank and energies (logged so a
    large-energy fold stays visible), and ``recipient`` is None when the
    reaction has no kept product at all to receive the share.
    """
    base = orphan.split('_')[0]
    orphan_rank = (elis_map.get(orphan) or {}).get('position')

    siblings, ground = [], None
    for p in valid_products:
        if p.split('_')[0] != base:
            continue
        info = elis_map.get(p) or {}
        lfs = lfs_map.get(p)
        if lfs == 0 or '_m' not in p:
            ground = ground or p          # the reaction's ground slot
            continue
        if lfs in PLACEHOLDER_LFS:
            continue                      # unidentified state: never a rank
        siblings.append({
            'product': p, 'position': info.get('position'),
            'elis': info.get('elis'), 'dk_elis': info.get('dk_elis'),
        })

    below = [s for s in siblings
             if s['position'] is not None and orphan_rank is not None
             and s['position'] < orphan_rank]
    if below:
        return max(below, key=lambda s: s['position']), siblings
    if ground is not None:
        return {'product': ground, 'position': 0, 'elis': None,
                'dk_elis': None}, siblings
    return None, siblings


def add_branching_to_xml(original_xml_file, branching_data, output_xml_file,
                         chain, verbose=True, prune_nn_prime_self_loops=False,
                         suppress_single_target_yields=False,
                         mode='flags_only', mf10_emit_summary=None,
                         orphan_policy='renorm',
                         repaired_keys=None):
    """Add branching data to chain XML.

    Parameters
    ----------
    original_xml_file : str
        Path to input chain XML file
    branching_data : dict
        Isomeric branching data keyed by nuclide name
    output_xml_file : str
        Path to output chain XML file
    chain : openmc.deplete.Chain
        Chain object for nuclide validation
    verbose : bool, optional
        Enable verbose output (default True)
    prune_nn_prime_self_loops : bool, optional
        If True, remove (n,n') reactions that have no isomeric branching.
        These self-loop reactions (target=parent, no metastable production)
        have no net effect on depletion and add computational overhead.
        Default is False.
    suppress_single_target_yields : bool, optional
        If True, suppress redundant single-target isomeric yields where
        the single target equals the original reaction target and all
        branching ratios are 1.0. Default is False (keep all).
    orphan_policy : {'add-stable', 'renorm', 'reattribute'}, optional
        What happens to a product that is not in the chain -- an excited state
        the mapper could not identify, or a whole product nuclide the decay
        library never carried. 'add-stable' adds it to the output chain as a
        stable nuclide with no decay data and keeps the branch; 'renorm'
        drops it and redistributes its share pro-rata over the surviving
        targets; 'reattribute' folds its share into the kept isomer at the
        nearest lower rank (ground when nothing is below it). Orthogonal to
        the mapping mode: all three apply in every mode. Default here is
        'renorm', the historical behaviour, as with the other writer switches;
        the CLI's own default is ``--orphan-policy add-stable`` and ``main()``
        always passes its choice explicitly.
    repaired_keys : dict, optional
        ``{(parent, reaction_type): ground_product}`` for reactions whose ground
        level was synthesized from the MF=3 remainder. The decoration is dropped
        only when the chain-membership filter leaves the synthesized GROUND as
        the sole survivor (a self-loop no-op carrying no branching information);
        a surviving metastable keeps its one-product decoration, which is
        informative. Non-repaired reactions are unaffected.

    Returns
    -------
    dict
        Summary with keys: 'added', 'added_keys' (the (nuclide, reaction_type)
        pairs actually written), 'skipped', 'errors', 'skipped_details',
        'elis_mappings', 'renormalizations', 'nuclides_with_branching_added',
        'nn_prime_self_loops_pruned', 'single_target_cases',
        'single_target_suppressed', 'repaired_dropped_no_metastable',
        'pathway_q_corrections', 'pathway_q_qm_disagreements',
        'pathway_q_rejected', 'q_from_mf10', 'q_from_elis', 'q_replicated',
        'q_chain_anchored', 'q_zero_anchor', 'orphan_policy',
        'orphan_nuclides_added', 'orphan_products_kept', 'reattributions',
        'reaction_type_missing', 'embedded_zero_rows'
    """
    tree = ET.parse(original_xml_file)
    root = tree.getroot()
    root.set('version', '1.0-branching')

    summary = {
        'added': 0, 'skipped': 0, 'errors': [], 'skipped_details': [],
        'added_keys': set(),  # (nuclide, reaction_type) actually written
        'repaired_dropped_no_metastable': 0,
        'elis_mappings': [], 'renormalizations': [],
        'nuclides_with_branching_added': set(),
        'nn_prime_self_loops_pruned': [],  # Track pruned (n,n') self-loops
        'single_target_cases': [],  # Track all single-target cases
        'single_target_suppressed': 0,  # Count of suppressed single-target cases
        'q_from_mf10': 0,   # pathway Q slots taken from the MF=10 QM/QI pair
        'q_from_elis': 0,   # ... from the legacy gq - ELIS (no QM/QI pair)
        'q_replicated': 0,  # pathway Q slots replicated (no QM/QI, no ELIS)
        'q_chain_anchored': 0,  # ... kept legacy because the gate refused
        'q_zero_anchor': 0,  # reactions whose gate was skipped (chain Q = 0)
        'pathway_q_corrections': [],  # slots that differ from the legacy fold
        'pathway_q_qm_disagreements': [],  # non-uniform QM within one section
        'pathway_q_rejected': [],  # reactions whose file QM failed the gate
        'orphan_policy': orphan_policy,
        'orphan_nuclides_added': {},  # name -> [source reaction record, ...]
        'orphan_products_kept': 0,    # branches kept by add-stable
        'reattributions': [],         # orphan shares folded into a kept isomer
        'reaction_type_missing': [],  # reaction absent from the chain nuclide
        'embedded_zero_rows': 0,      # embedded rows with no surviving target
    }

    nuclide_map = {nuc.get('name'): nuc for nuc in root.findall('nuclide')}
    # One membership set for the whole write: the per-product test below used
    # to walk every chain nuclide, and add-stable needs a set it can register
    # newly added names in so later reactions and other parents see them.
    chain_names = {nuc.name for nuc in chain.nuclides}

    for nuclide_name, nuclide_reactions in branching_data.items():
        if nuclide_name not in nuclide_map:
            summary['errors'].append(f"Nuclide {nuclide_name} not in chain")
            summary['skipped'] += len(nuclide_reactions)
            continue

        nuc_elem = nuclide_map[nuclide_name]
        # Multimap: official chains represent a branched reaction as multiple
        # same-type elements (one per static pathway)
        reaction_map = {}
        for rx in nuc_elem.findall('reaction'):
            reaction_map.setdefault(rx.get('type'), []).append(rx)

        for reaction_type, branching in nuclide_reactions.items():
            if reaction_type not in reaction_map:
                # The GENDF file has the channel, the chain nuclide does not
                # carry that reaction at all -- nothing to decorate. Recorded
                # with parent and MT so the drop is chaseable (F5e).
                summary['skipped'] += 1
                summary['reaction_type_missing'].append({
                    'nuclide': nuclide_name, 'reaction': reaction_type,
                    'mt': branching.mt,
                })
                continue

            # Convert to energy_yields format
            energy_yields = {}
            for i, energy in enumerate(branching.energies):
                energy_yields[float(energy)] = {
                    product: float(branching.branching_ratios[j, i])
                    for j, product in enumerate(branching.products)
                }

            # Validate products
            all_products = list(branching.products)
            valid_products = [p for p in all_products if p in chain_names]
            missing_products = [p for p in all_products if p not in valid_products]

            # Orphan disposition (mode-agnostic). Runs BEFORE the "nothing
            # left" skip and the repaired-ground check, so add-stable also
            # rescues a reaction whose every product was missing.
            lfs_map_all = branching.lfs_mapping or {}
            elis_map_all = getattr(branching, 'elis_mapping', None) or {}
            if not isinstance(elis_map_all, dict):
                elis_map_all = {}

            if orphan_policy == 'add-stable':
                # A state an EARLIER reaction already added is no longer
                # "missing", but this parent still feeds it: record the source
                # so the roll call shows every contributing pathway and the
                # ELFS spread between them (wide spread = the parents do not
                # agree on which state it is).
                shared = [p for p in all_products
                          if p not in missing_products
                          and p in summary['orphan_nuclides_added']]
                for orphan in list(missing_products) + shared:
                    info = elis_map_all.get(orphan) or {}
                    source = {
                        'parent': nuclide_name, 'reaction': reaction_type,
                        'mt': branching.mt, 'lfs': lfs_map_all.get(orphan),
                        'elis': info.get('elis'),
                        'position': info.get('position'),
                        'method': info.get('method')}
                    source['created'] = _add_stable_nuclide_element(
                        root, nuclide_map, chain, chain_names, orphan, summary,
                        source)
                if missing_products:
                    summary['orphan_products_kept'] += len(missing_products)
                    valid_products = list(all_products)
                    missing_products = []

            elif missing_products and orphan_policy == 'reattribute':
                # Fold each orphan's per-energy share DOWN into a kept isomer;
                # the sums stay 1, so no renormalization follows. An orphan
                # with no recipient at all stays in missing_products and takes
                # the pro-rata path below (reported loudly).
                stranded = []
                for orphan in missing_products:
                    kept = [p for p in all_products
                            if p in chain_names and p != orphan]
                    recipient, siblings = _reattribution_recipient(
                        orphan, kept, elis_map_all, lfs_map_all)
                    orphan_info = elis_map_all.get(orphan) or {}
                    record = {
                        'nuclide': nuclide_name, 'reaction': reaction_type,
                        'mt': branching.mt, 'orphan': orphan,
                        'lfs': lfs_map_all.get(orphan),
                        'position': orphan_info.get('position'),
                        'elis': orphan_info.get('elis'),
                        'recipient': None, 'rank_distance': None,
                        'delta_elfs': None, 'siblings': siblings,
                    }
                    if recipient is None:
                        stranded.append(orphan)
                        summary['reattributions'].append(record)
                        continue
                    for products_dict in energy_yields.values():
                        products_dict[recipient['product']] = (
                            products_dict.get(recipient['product'], 0.0)
                            + products_dict.pop(orphan, 0.0))
                    o_rank, r_rank = record['position'], recipient['position']
                    o_elfs, r_elfs = record['elis'], recipient['elis']
                    record.update({
                        'recipient': recipient['product'],
                        'recipient_position': r_rank,
                        'rank_distance': (o_rank - r_rank
                                          if None not in (o_rank, r_rank)
                                          else None),
                        'delta_elfs': (o_elfs - r_elfs
                                       if None not in (o_elfs, r_elfs)
                                       else None),
                    })
                    summary['reattributions'].append(record)
                missing_products = stranded

            if not valid_products:
                summary['skipped'] += 1
                missing_with_lfs = [{'name': mp, 'lfs': branching.lfs_mapping.get(mp) if branching.lfs_mapping else None}
                                   for mp in missing_products]
                summary['skipped_details'].append({
                    'nuclide': nuclide_name, 'reaction': reaction_type,
                    'mt': branching.mt, 'missing_with_lfs': missing_with_lfs
                })
                continue

            # A synthesized ground left alone carries no branching at all --
            # writing it would ship a ground-only self-loop no-op that the
            # pruner then has to keep. A surviving METASTABLE is informative
            # (100% of the channel to the isomer), so that decoration ships.
            ground_product = (repaired_keys or {}).get(
                (nuclide_name, reaction_type))
            if ground_product is not None and (not valid_products
                                               or valid_products == [ground_product]):
                summary['repaired_dropped_no_metastable'] += 1
                summary['skipped'] += 1
                summary['skipped_details'].append({
                    'nuclide': nuclide_name, 'reaction': reaction_type,
                    'mt': branching.mt, 'ground_product': ground_product,
                    'reason': 'repaired_ground_only_survivor',
                })
                continue

            # Renormalize if some products missing
            if missing_products:
                new_yields = {}
                for energy, products_dict in energy_yields.items():
                    valid_sum = sum(products_dict.get(p, 0.0) for p in valid_products)
                    if valid_sum > 0:
                        new_yields[energy] = {p: products_dict[p] / valid_sum for p in valid_products}
                    else:
                        new_yields[energy] = {p: 0.0 for p in valid_products}
                energy_yields = new_yields

                summary['renormalizations'].append({
                    'nuclide': nuclide_name, 'reaction': reaction_type,
                    'mt': branching.mt, 'valid_products': valid_products,
                    'dropped_products': missing_products
                })

            # Collect ELIS mappings
            if hasattr(branching, 'elis_mapping') and branching.elis_mapping:
                for product, elis_info in branching.elis_mapping.items():
                    half_life = next((nuc.half_life for nuc in chain.nuclides if nuc.name == product), None)
                    lfs = branching.lfs_mapping.get(product) if branching.lfs_mapping else None
                    summary['elis_mappings'].append({
                        'parent': nuclide_name, 'reaction': reaction_type,
                        'mt': branching.mt, 'product': product, 'lfs': lfs,
                        'elis': elis_info.get('elis'), 'dk_elis': elis_info.get('dk_elis'),
                        'liso': elis_info.get('liso'), 'method': elis_info.get('method', 'elis'),
                        'half_life': half_life if half_life else 'stable',
                        'target_z': elis_info.get('target_z'), 'target_a': elis_info.get('target_a'),
                        # Hybrid context: rank among the product's real levels,
                        # why the level left Phase 1, and which decay states
                        # Phase 1 had already claimed when Phase 2 ran.
                        'position': elis_info.get('position'),
                        'fallback_reason': elis_info.get('fallback_reason'),
                        'phase1_claimed_lisos': elis_info.get('phase1_claimed_lisos'),
                    })

            # Detect single-target cases - ALWAYS log regardless of suppress flag
            # For multi-entry reactions keep the ground-target element as the
            # survivor (duplicates are folded away at the write step below)
            rx_elems = reaction_map[reaction_type]
            rx_elem = next((e for e in rx_elems
                            if e.get('target') == valid_products[0]), rx_elems[0])
            original_target = rx_elem.get('target')

            if len(valid_products) == 1:
                single_product = valid_products[0]
                all_ratios_are_one = all(
                    abs(energy_yields[e].get(single_product, 0.0) - 1.0) < 1e-10
                    for e in energy_yields.keys()
                )

                # Determine reason
                reason, reason_detail = _determine_single_target_reason(
                    branching, valid_products, missing_products, all_products
                )

                # Check if redundant (single target == original reaction target)
                is_redundant = (single_product == original_target) and all_ratios_are_one

                # A KEPT single-GROUND decoration has no elis entry for its one
                # slot, so it is written from the LFS=0 subsection's own QM like
                # any other ground slot (0.0 for MT=4); only a section with no
                # QM anywhere would replicate the scalar Q there. Canonical runs
                # suppress these cases outright, so none of them ship.
                # Decide action based on flag (DEFAULT: KEEP)
                if suppress_single_target_yields and is_redundant:
                    action = "suppressed"
                elif is_redundant:
                    action = "kept (redundant)"
                else:
                    action = "kept (non-redundant)"  # e.g., single metastable differs from original

                # ALWAYS log the case
                summary['single_target_cases'].append({
                    'nuclide': nuclide_name,
                    'reaction': reaction_type,
                    'mt': branching.mt,
                    'single_target': single_product,
                    'original_target': original_target,
                    'reason': reason.value,
                    'reason_detail': reason_detail,
                    'is_redundant': is_redundant,
                    'action': action,
                    'n_energies': len(energy_yields),
                    'missing_products': missing_products
                })

                # Skip XML writing if suppressed
                if action == "suppressed":
                    summary['single_target_suppressed'] += 1
                    summary['skipped'] += 1
                    continue

            # Fold duplicate same-type elements into the survivor so exactly
            # one isomeric element is emitted per (nuclide, reaction type).
            # The child carries the complete distribution, so the static split
            # (including the survivor's branching_ratio) no longer applies.
            if len(rx_elems) > 1:
                for extra in rx_elems:
                    if extra is not rx_elem:
                        nuc_elem.remove(extra)
                rx_elem.attrib.pop('branching_ratio', None)

            # Remove any existing isomeric elements
            for tag in ('isomeric_yields', 'isomeric_branching'):
                existing = rx_elem.find(tag)
                if existing is not None:
                    rx_elem.remove(existing)

            products = list(valid_products)
            energies = sorted(energy_yields.keys())

            if mode == 'flags_only':
                # Folded new form:
                #   <reaction type="(n,2n)">
                #     <isomeric_branching targets="A B" gendf_lfs="0 1"
                #                         Q="q_ground q_meta"/>
                #   </reaction>
                # The per-pathway target/LFS/Q live ONLY on the child; the scalar
                # target/Q are dropped from the <reaction> element (below).
                iso_elem = ET.SubElement(rx_elem, 'isomeric_branching')
                iso_elem.set('targets', ' '.join(products))
                # Write LFS values from branching.lfs_mapping. Slot 0 is read
                # from the mapping too: the chain-membership filter can promote
                # a metastable into it, and a hardcoded 0 would mislabel it.
                lfs_map = branching.lfs_mapping or {}
                lfs_values = [lfs_map.get(products[0], 0)]
                for p in products[1:]:
                    lfs = lfs_map.get(p)
                    if lfs is None:
                        raise ValueError(
                            f"No LFS mapping for product {p} of "
                            f"{nuclide_name} {reaction_type}")
                    lfs_values.append(lfs)
                iso_elem.set('gendf_lfs', ' '.join(str(v) for v in lfs_values))

                # Per-pathway Q parallel-list. Every slot takes the Q of ITS OWN
                # MF=10 subsection: a metastable slot gets that level's QI, the
                # LFS=0 slot gets the LFS=0 subsection's own QM. QM is a
                # per-subsection TAB1 head, NOT a section constant -- evaluations
                # do disagree between the ground and metastable subsections of
                # one MT (EAF-2010 metastable-parent MT=4: +E(parent) on LFS=0,
                # 0.0 on the metastable levels; ENDF/B-8.1 Ta180_m1: eV-level
                # spreads), so a sibling's QM is only a fallback for a
                # SYNTHESIZED ground, where no LFS=0 subsection exists at all.
                # Deriving Q from QM/QI instead of the old ``Q_chain - ELFS``
                # matters for MT=4: the chain's scalar Q is QI(MF=3) =
                # -E(level), not QM, so that subtraction charged the level
                # energy twice. Nothing is invented: a slot with neither a Q
                # pair nor an ELIS replicates the scalar Q and is counted and
                # reported. The reaction's own scalar Q is untouched here (it is
                # popped with the target below).
                scalar_q = rx_elem.get('Q')
                q_default = scalar_q if scalar_q is not None else '0.0'
                gq = float(scalar_q) if scalar_q is not None else 0.0
                elis_map = getattr(branching, 'elis_mapping', None) or {}
                if not isinstance(elis_map, dict):
                    elis_map = {}
                # Sibling QM: elis_map holds mapped METASTABLES only, so this is
                # never the ground's own value -- synthesized-ground fallback.
                meta_qms = [i['qm'] for i in elis_map.values()
                            if isinstance(i, dict) and i.get('qm') is not None]
                section_qm = meta_qms[0] if meta_qms else None
                ground_qm = getattr(branching, 'ground_qm', None)

                # Sanity gate: file values are adopted only where the file's own
                # ground QM corroborates the chain's scalar Q. The chain scalar
                # is AME/evaluation-derived and usually the more accurate of the
                # two, while nothing validates an absolute QM -- ENDF/B-8.1 ships
                # (n,alpha) sections wrong by up to 12 MeV, sign included. The
                # legacy fold consumed only the QM-QI DIFFERENCE and so was
                # immune to that; transcribing absolutes is not. When the two
                # disagree beyond PATHWAY_Q_CHAIN_TOL the whole reaction reverts
                # to the chain-anchored legacy arithmetic (all slots, so the
                # written offsets stay mutually consistent) and the refused file
                # values are ledgered. MT=4 is exempt by construction: there the
                # scalar Q is QI(MF=3) = -E(level), a different quantity than QM
                # (0.0 for every ground-parent MT=4 section, 424/424), so
                # "disagreement" is the very defect being fixed. A reaction with
                # no scalar Q has nothing to check against and keeps the file.
                # An explicit Q="0.0" is the evaluators' missing-Q placeholder,
                # never a physical transmutation Q: it anchors nothing, so the
                # gate is skipped (file values kept) and the reaction counted.
                file_ground_qm = ground_qm if ground_qm is not None else section_qm
                q_zero_anchor = (
                    branching.mt != 4 and scalar_q is not None
                    and gq == 0.0 and file_ground_qm is not None)
                q_chain_reject = (
                    branching.mt != 4 and scalar_q is not None
                    and gq != 0.0 and file_ground_qm is not None
                    and abs(file_ground_qm - gq) > PATHWAY_Q_CHAIN_TOL)
                if q_zero_anchor:
                    summary['q_zero_anchor'] += 1

                q_values, q_legacy, q_sources = [], [], []
                q_refused = []  # file values the gate threw away (ledger only)
                for slot, (p, lfs) in enumerate(zip(products, lfs_values)):
                    info = elis_map.get(p)
                    info = info if isinstance(info, dict) else {}
                    elfs = info.get('elis')

                    # What the pre-fix fold wrote -- reported, never written.
                    q_legacy.append(_q_str(gq - float(elfs))
                                    if elfs is not None and (slot or lfs)
                                    else q_default)

                    if lfs:
                        qi, qm = info.get('qi'), info.get('qm')
                        if qi is not None and qm is not None:
                            q_new, source = qi, 'QI'
                        elif elfs is not None:
                            # No Q pair (pre-48bf86e24 serialized input): keep
                            # the computable legacy correction rather than
                            # dropping to the undifferentiated scalar.
                            q_new, source = gq - float(elfs), 'ELIS'
                        else:
                            q_new, source = None, 'QI'
                    else:
                        # True ground slot: its own subsection's QM, else (only
                        # for a synthesized ground) a sibling's.
                        q_new = ground_qm if ground_qm is not None else section_qm
                        source = 'QM'
                    if q_chain_reject:
                        # Whole-reaction revert: every slot keeps the legacy
                        # chain-anchored value, so the ground and metastable
                        # offsets stay mutually consistent (a half-reverted
                        # reaction would mix two energy zeros). The file value
                        # is carried to the ledger and nowhere else ('n/a' =
                        # that slot had no file value to refuse).
                        q_refused.append('n/a' if q_new is None
                                         else _q_str(q_new))
                        q_values.append(q_legacy[slot])
                        q_sources.append('chain')
                        summary['q_chain_anchored'] += 1
                    elif q_new is None:
                        q_values.append(q_default)
                        q_sources.append('scalar')
                        summary['q_replicated'] += 1
                    else:
                        q_values.append(_q_str(q_new))
                        q_sources.append(source)
                        summary['q_from_elis' if source == 'ELIS'
                                else 'q_from_mf10'] += 1
                iso_elem.set('Q', ' '.join(q_values))

                if q_chain_reject:
                    summary['pathway_q_rejected'].append({
                        'nuclide': nuclide_name, 'reaction': reaction_type,
                        'mt': branching.mt, 'targets': list(products),
                        'lfs': list(lfs_values), 'file_ground_qm': file_ground_qm,
                        'chain_q': gq, 'delta': file_ground_qm - gq,
                        'q_file': q_refused, 'q_kept': list(q_values),
                    })

                # Intra-section QM disagreement: the ground slot's value is only
                # as trustworthy as the section is uniform. Record when the
                # mapped levels disagree among themselves, or when the sibling
                # fallback would have written something else than the LFS=0 QM.
                # Recorded independently of the gate -- it describes the FILE,
                # not what was written.
                distinct_meta_qms = sorted(set(meta_qms))
                if len(distinct_meta_qms) > 1 or (
                        ground_qm is not None and section_qm is not None
                        and ground_qm != section_qm):
                    summary['pathway_q_qm_disagreements'].append({
                        'nuclide': nuclide_name, 'reaction': reaction_type,
                        'mt': branching.mt, 'ground_qm': ground_qm,
                        'meta_qms': distinct_meta_qms,
                    })

                # A gate-rejected reaction wrote q_legacy verbatim, so this is
                # False there by construction: nothing changed, nothing to
                # correct -- it is listed in the REJECTED section instead.
                if _q_lists_differ(q_values, q_legacy):
                    summary['pathway_q_corrections'].append({
                        'nuclide': nuclide_name, 'reaction': reaction_type,
                        'mt': branching.mt, 'targets': list(products),
                        'lfs': list(lfs_values), 'q': q_values,
                        'source': q_sources, 'q_legacy': q_legacy,
                    })

                # Fold: drop the scalar target/Q now that the child carries the
                # full per-pathway lists. Unbranched reactions are untouched.
                rx_elem.attrib.pop('target', None)
                rx_elem.attrib.pop('Q', None)
            else:
                # Write full <isomeric_yields> (embedded/informational).
                # An energy where every surviving target is zero carries no
                # branching at all (the whole rate went to products that left
                # with the orphan policy or the membership filter); such a row
                # would only invite a divide-by-zero downstream, so it is
                # dropped -- counted once, never row by row.
                kept_energies = [
                    e for e in energies
                    if any(energy_yields[e].get(p, 0.0) for p in products)]
                if kept_energies and len(kept_energies) != len(energies):
                    summary['embedded_zero_rows'] += len(energies) - len(kept_energies)
                    energies = kept_energies

                yields_elem = ET.SubElement(rx_elem, 'isomeric_yields')
                yields_elem.set('type', 'energy_dependent')

                energies_elem = ET.SubElement(yields_elem, 'energies')
                energies_elem.text = ' '.join(f'{e:.6e}' for e in energies)

                targets_elem = ET.SubElement(yields_elem, 'targets')
                targets_elem.text = ' '.join(products)

                ratios_elem = ET.SubElement(yields_elem, 'branching_ratios')
                ratios_lines = []
                for product in products:
                    ratios = [energy_yields[e].get(product, 0.0) for e in energies]
                    ratios_lines.append('          ' + ' '.join(f'{r:.6e}' for r in ratios))
                ratios_elem.text = '\n' + '\n'.join(ratios_lines) + '\n        '

            summary['added'] += 1
            summary['added_keys'].add((nuclide_name, reaction_type))
            summary['nuclides_with_branching_added'].add(nuclide_name)

    # Prune (n,n') self-loops without isomeric branching if requested
    if prune_nn_prime_self_loops:
        for nuc_elem in root.findall('nuclide'):
            nuc_name = nuc_elem.get('name')

            reactions_to_remove = []
            for rx_elem in nuc_elem.findall('reaction'):
                rx_type = rx_elem.get('type')
                target = rx_elem.get('target')

                # Exact self-loop only: X -> X. A metastable parent's ground
                # route (X_m1 -> X) is a real de-excitation transition and
                # must stay.
                if rx_type == "(n,n')" and target == nuc_name:
                    # Check if no isomeric branching element exists
                    has_branching = (rx_elem.find('isomeric_branching') is not None
                                     or rx_elem.find('isomeric_yields') is not None)
                    if not has_branching:
                        reactions_to_remove.append((rx_elem, rx_type, target))

            # Remove the identified reactions
            for rx_elem, rx_type, target in reactions_to_remove:
                nuc_elem.remove(rx_elem)
                summary['nn_prime_self_loops_pruned'].append({
                    'nuclide': nuc_name,
                    'reaction': rx_type,
                    'target': target
                })

    # Metastable-direct post-pass: emit single-metastable MF=10-only channels
    # (e.g. In115(n,n')->In115_m1) deferred by the pre-pass. Runs AFTER decoration
    # and prune so Gate 1 never sees them (never asked for a branching the file
    # cannot supply), and BEFORE the recount below so their counts are included.
    # A direct X->X_m1 is not an exact self-loop, so the prune above (already
    # done) leaves it alone.
    if mf10_emit_summary is not None:
        emit_metastable_direct_postpass(
            root, mf10_emit_summary,
            orphan_names=set(summary['orphan_nuclides_added']))

    # Recount 'reactions' per unfolded pathway to match the PENDF chains: a
    # branched reaction counts once per target (ground + each metastable), not
    # once per <reaction> element. Runs after pruning.
    for nuc_elem in root.findall('nuclide'):
        if 'reactions' not in nuc_elem.attrib:
            continue
        n_pathways = 0
        for rx_elem in nuc_elem.findall('reaction'):
            iso = rx_elem.find('isomeric_branching')
            if iso is not None and iso.get('targets'):
                n_pathways += len(iso.get('targets').split())
                continue
            yld = rx_elem.find('isomeric_yields')
            tgt = yld.find('targets') if yld is not None else None
            if tgt is not None and tgt.text:
                n_pathways += len(tgt.text.split())
                continue
            n_pathways += 1
        nuc_elem.set('reactions', str(n_pathways))

    # Write output
    xml_str = ET.tostring(root, encoding='unicode')
    pretty = minidom.parseString(xml_str).toprettyxml(indent='  ')
    lines = [l for l in pretty.split('\n') if l.strip()]
    with open(output_xml_file, 'w') as f:
        f.write('\n'.join(lines))

    return summary


# =============================================================================
# Logging functions
# =============================================================================

def _fmt_ratio(ratio):
    """Render a band/full ratio: 'n/a' for None, 'inf' for inf, else 4dp."""
    if ratio is None:
        return "n/a"
    if ratio == float('inf'):
        return "inf"
    return f"{ratio:.4f}"


def _fmt_erange(o):
    """Worst-deviation group's energy band as a compact 'elo-ehi' range (eV).

    This is the group-space analog of the PENDF pointwise 'E@max': a multigroup
    reaction's worst deviation lives in a GROUP, which spans an energy range.
    """
    lo = o.get('energy_lo')
    hi = o.get('energy_hi')
    if lo is None or hi is None:
        return "-"
    return f"{lo:.3e}-{hi:.3e}"


def _fmt_spread(v):
    """Render the BR-spread column: compact 4dp, or 'n/a' when undefined."""
    return f"{v:.4f}" if v is not None else "n/a"


def _write_consistency_audit_section(f, offenders, audit_clean, emax=2.0e7):
    """MF=10 CONSISTENCY AUDIT section (group-space): offenders worst-first.

    Mirrors the PENDF tool's section (title, columns, phrasing) with two
    group-space adaptations: E[eV]@max carries the worst group's energy RANGE,
    and a GENDF-specific BR-spread column separates common-mode (benign) from
    differential (suspect) band-ratio departures.
    """
    f.write("\n\n" + "=" * 240 + "\n")
    f.write("MF=10 CONSISTENCY AUDIT\n")
    f.write("=" * 240 + "\n\n")
    f.write("Group-by-group Sum(MF=10 partials) vs the MF=3 total (each MF=10 "
            "partial aligned to the library group grid; capped at "
            f"E <= {emax:.3e} eV), for every reaction carrying >=1 metastable "
            "pathway.\n")
    f.write("The cap suppresses the >30 MeV MT=5 lumping artifact (MF=10 "
            "partials stop near 30 MeV while the MF=3 total runs higher).\n")
    f.write(f"Groups where BOTH sides sit below {CONSISTENCY_ABS_FLOOR:.0e} b "
            "(evaluator floor dust) are exempt from the deviation scan.\n")
    f.write("IntRatio and the Thermal/Epithermal/Intermed/Fast ratios are "
            "LETHARGY-weighted (Sum sigma_g*du_g) partials/total; bands are "
            f"thermal [grid_min, {_BAND_THERMAL_HI:g} eV) "
            f"({_BAND_THERMAL_HI:g} eV = Cd cutoff), epithermal "
            f"[{_BAND_THERMAL_HI:g}, {_BAND_EPITHERMAL_HI:.0e} eV), intermediate "
            f"[{_BAND_EPITHERMAL_HI:.0e}, {_BAND_INTERMEDIATE_HI:.0e} eV), fast "
            f"[{_BAND_INTERMEDIATE_HI:.0e} eV, {emax:.3e} eV]; a band-edge group "
            "is apportioned by lethargy overlap. 'n/a' = no contributing group, "
            f"zero total, or below-threshold dust (max < "
            f"{CONSISTENCY_ABS_FLOOR:.0e} b).\n")
    f.write("E[eV]@max = ENERGY RANGE of the worst-deviation group (group-space "
            "analog of the pointwise E@max).\n")
    f.write("BR-spread [GENDF-specific] = max over non-ground partials of "
            "(max_g - min_g) of the per-group branching fraction "
            "part_m,g / Sum(parts)_g, measured over the significant groups "
            "WITHIN the worst-deviating band (max |ratio-1|, where the anomaly "
            "lives). It separates the two failure modes on the GENDF RATIO "
            "path:\n")
    f.write("  * SMALL BR-spread + large band ratios => COMMON-MODE (all "
            "partials scaled together, branching fraction constant across the "
            "anomalous band): HARMLESS here -- the runtime applies "
            "partial/Sum(partials) ratios to an MF=3 rate, so the uniform factor "
            "cancels.\n")
    f.write("  * LARGE BR-spread => DIFFERENTIAL defect (partials disagree by "
            "energy within the anomalous band): genuinely biases the "
            "branching.\n")
    f.write("  (Measuring the spread over the FULL range instead would be "
            "dominated by the reaction's legitimate fast-region branching "
            "variation and could not tell a benign chord from a real defect.)\n")
    f.write(f"Offenders (max rel dev > {CONSISTENCY_RTOL:.0e}) are listed "
            f"worst-first; {audit_clean} audited reaction(s) are clean.\n\n")
    if not offenders:
        f.write("No offenders: all audited reactions agree within "
                f"{CONSISTENCY_RTOL:.0e}.\n")
        return
    f.write(f"Total offenders: {len(offenders)}\n\n")
    header = (f"{'Parent':<12}  {'MT':>5}  {'Reaction':<12}  {'MaxRelDev':>12}  "
              f"{'E[eV]@max':>21}  {'Sum-part[b]':>14}  {'Total[b]':>14}  "
              f"{'IntRatio':>12}  {'Thermal':>10}  {'Epithermal':>10}  "
              f"{'Intermed':>10}  {'Fast':>10}  {'BR-spread':>10}  {'Notes':<40}")
    sep = "-" * len(header)
    f.write(header + "\n" + sep + "\n")
    for o in sorted(offenders, key=lambda x: x['worst_dev'], reverse=True):
        sp = o.get('sum_partials')
        tot = o.get('total')
        sp_str = f"{sp:.4e}" if sp is not None else "-"
        tot_str = f"{tot:.4e}" if tot is not None else "-"
        f.write(f"{o['parent']:<12}  {o['mt']:>5}  {o['reaction']:<12}  "
                f"{o['worst_dev']:>12.4e}  {_fmt_erange(o):>21}  {sp_str:>14}  "
                f"{tot_str:>14}  {_fmt_ratio(o.get('integral_ratio')):>12}  "
                f"{_fmt_ratio(o.get('ratio_thermal')):>10}  "
                f"{_fmt_ratio(o.get('ratio_epithermal')):>10}  "
                f"{_fmt_ratio(o.get('ratio_intermediate')):>10}  "
                f"{_fmt_ratio(o.get('ratio_fast')):>10}  "
                f"{_fmt_spread(o.get('br_spread')):>10}  "
                f"{o.get('notes', ''):<40}\n")


def _write_rejected_section(f, rejected, reject_band_ratio):
    """MF=10 REJECTED REACTIONS section: band-ratio-gated reactions left stock."""
    f.write("\n\n" + "=" * 240 + "\n")
    f.write("MF=10 REJECTED REACTIONS\n")
    f.write("=" * 240 + "\n\n")
    if reject_band_ratio is None:
        f.write("rejection disabled (audit only): --mf10-reject-band-ratio was "
                "not set, so no reaction was left stock on audit grounds.\n")
        f.write("NOTE: on the GENDF ratio path a band-ratio deviation is "
                "harmless when it is COMMON-MODE (small BR-spread), so rejection "
                "is OFF by default.\n")
        return
    f.write(f"Threshold: --mf10-reject-band-ratio = {reject_band_ratio:.3e}\n")
    f.write("Criterion: a reaction is left stock (no <isomeric_branching> child) "
            "when any DEFINED lethargy band has ratio-1 exceeding the threshold "
            "(over-summing ONLY; under-summing never rejects), EXCEPT "
            "self-loop-ground reactions (exempt).\n")
    f.write("Consequence: the base reaction keeps its scalar target/Q; the MF=3 "
            "total routes to the ground target and isomeric branching is "
            "discarded.\n")
    f.write("CAUTION: a common-mode band-ratio deviation (small BR-spread) is "
            "HARMLESS on the GENDF path -- inspect BR-spread before trusting a "
            "rejection.\n\n")
    if not rejected:
        f.write("No reactions exceeded the active threshold.\n")
        return
    f.write(f"Total rejected: {len(rejected)}\n\n")
    header = (f"{'Parent':<12}  {'MT':>5}  {'Reaction':<12}  {'MaxRelDev':>12}  "
              f"{'E[eV]@max':>21}  {'Thermal':>10}  {'Epithermal':>10}  "
              f"{'Intermed':>10}  {'Fast':>10}  {'BR-spread':>10}  "
              f"{'Criterion':<28}  {'Consequence':<40}")
    sep = "-" * len(header)
    f.write(header + "\n" + sep + "\n")
    consequence = "left stock: MF=3 total -> ground target"
    for o in sorted(rejected, key=lambda x: x['worst_dev'], reverse=True):
        f.write(f"{o['parent']:<12}  {o['mt']:>5}  {o['reaction']:<12}  "
                f"{o['worst_dev']:>12.4e}  {_fmt_erange(o):>21}  "
                f"{_fmt_ratio(o.get('ratio_thermal')):>10}  "
                f"{_fmt_ratio(o.get('ratio_epithermal')):>10}  "
                f"{_fmt_ratio(o.get('ratio_intermediate')):>10}  "
                f"{_fmt_ratio(o.get('ratio_fast')):>10}  "
                f"{_fmt_spread(o.get('br_spread')):>10}  "
                f"{o.get('criterion', '-'):<28}  {consequence:<40}\n")


def _elfs_str(value, width=14):
    """Right-aligned energy in eV, 'N/A' when the file gave none."""
    return (f"{value:>{width}.1f}" if isinstance(value, (int, float))
            else f"{'N/A':>{width}}")


def _hybrid_crossings(isomer_mappings):
    """Mark the Phase-2 pairings whose level order is inverted.

    A crossing is not an error: a Phase-1 energy match is trusted absolutely,
    so the level below it can legitimately end up in a HIGHER decay state than
    a level above it. Detected by comparing rank against LISO across every
    level of one product, and flagged only so an audit can look at them.
    """
    by_product = defaultdict(list)
    for m in isomer_mappings:
        product = str(m.get('product') or '')
        base = product.split('_m')[0]
        by_product[(m.get('parent'), m.get('mt'), base)].append(m)

    crossed = set()
    for rows in by_product.values():
        for r in rows:
            rank, liso = r.get('position'), r.get('liso')
            if rank is None or liso is None:
                continue
            for s in rows:
                s_rank, s_liso = s.get('position'), s.get('liso')
                if s is r or s_rank is None or s_liso is None:
                    continue
                if ((s_rank < rank and s_liso > liso)
                        or (s_rank > rank and s_liso < liso)):
                    crossed.add(id(r))
                    break
    return crossed


def _write_hybrid_mapping_section(f, hybrid_records, isomer_mappings):
    """HYBRID MAPPING: how each level reached its state, phase by phase."""
    fallbacks = [r for r in hybrid_records if r.get('type') == 'hybrid_fallback']
    abstained = [r for r in fallbacks
                 if r.get('fallback_reason') == 'elis_ambiguous']
    requeued = [r for r in fallbacks
                if r.get('fallback_reason') == 'duplicate_loser']
    unusable = [r for r in fallbacks
                if r.get('fallback_reason') in HYBRID_UNUSABLE_REASONS]
    bound = [r for r in hybrid_records if r.get('type') == 'placeholder_bound']
    unmapped = [r for r in hybrid_records
                if r.get('type') == 'placeholder_unmapped']
    phase2 = [m for m in isomer_mappings
              if m.get('method') == 'lfs_order_fallback']

    f.write("\n\n" + "=" * 220 + "\n")
    f.write("HYBRID MAPPING (elis_lfs_order) -- PHASE DETAIL\n")
    f.write("=" * 220 + "\n\n")
    f.write("Phase 1 binds a level to the decay state whose excitation energy "
            "matches its GENDF-ELFS, and only when that state is the single\n")
    f.write("candidate within tolerance. Phase 2 takes the levels left over, "
            "in level order, and pairs them with the decay metastables left\n")
    f.write("over, in LISO order -- energies play no part there. Nothing on "
            "this page was discarded: every row below is either mapped by\n")
    f.write("position or reported as a placeholder that could not be placed.\n\n")

    f.write("PHASE-1 ABSTENTIONS (two decay states within tolerance)\n")
    f.write("-" * 150 + "\n")
    if not abstained:
        f.write("None.\n")
    else:
        f.write("The level's energy fits two states, so binding either one "
                "would be a coin toss; the level is re-derived by position "
                "instead.\n")
        f.write(f"{'Parent':<12}  {'MT':>5}  {'Reaction':<12}  {'LFS':>4}  "
                f"{'GENDF-ELFS[eV]':>14}  {'Nearest':<10}  "
                f"{'DK-ELIS[eV]':>14}  {'Second':<10}  {'DK-ELIS[eV]':>14}\n")
        f.write("-" * 150 + "\n")
        for r in abstained:
            f.write(f"{str(r.get('parent')):<12}  {str(r.get('mt')):>5}  "
                    f"{str(r.get('reaction')):<12}  {str(r.get('lfs')):>4}  "
                    f"{_elfs_str(r.get('elis'))}  "
                    f"{'_m' + str(r.get('nearest_liso')):<10}  "
                    f"{_elfs_str(r.get('nearest_dk_elis'))}  "
                    f"{'_m' + str(r.get('second_liso')):<10}  "
                    f"{_elfs_str(r.get('second_dk_elis'))}\n")
    f.write("\n")

    f.write("PHASE-1 LEVELS ROUTED TO PHASE 2 (energy unusable or unmatched)\n")
    f.write("-" * 150 + "\n")
    if not (unusable or requeued):
        f.write("None.\n")
    else:
        f.write("None of these is a loss: each level is handed to the "
                "positional phase, which is what the lfs_order mode would have "
                "done with it from the start.\n")
        f.write(f"{'Parent':<12}  {'MT':>5}  {'Reaction':<12}  {'LFS':>4}  "
                f"{'Rank':>4}  {'GENDF-ELFS[eV]':>14}  {'Reason':<60}\n")
        f.write("-" * 150 + "\n")
        for r in sorted(unusable + requeued,
                        key=lambda x: (str(x.get('parent')), x.get('mt') or 0,
                                       x.get('lfs') or 0)):
            reason = HYBRID_REASON_LABELS.get(r.get('fallback_reason'),
                                              str(r.get('fallback_reason')))
            rank = r.get('position')
            f.write(f"{str(r.get('parent')):<12}  {str(r.get('mt')):>5}  "
                    f"{str(r.get('reaction')):<12}  {str(r.get('lfs')):>4}  "
                    f"{('-' if rank is None else str(rank)):>4}  "
                    f"{_elfs_str(r.get('elis'))}  {reason:<60}\n")
    f.write("\n")

    f.write("PHASE-2 POSITIONAL ASSIGNMENTS\n")
    f.write("-" * 190 + "\n")
    if not phase2:
        f.write("None.\n")
    else:
        f.write("Rank = the level's place among the product's real levels "
                "(placeholder LFS values never take a rank). 'Phase-1 held' "
                "lists the decay states already\n")
        f.write("claimed by an energy match when this pairing was made -- "
                "those are skipped, which is what keeps a fallback off a state "
                "that is already spoken for. 'Crossing?'\n")
        f.write("flags a pairing whose level order is inverted against an "
                "energy-matched sibling: allowed by design (an energy match is "
                "trusted absolutely), shown for audit.\n")
        f.write(f"{'Parent':<12}  {'MT':>5}  {'Reaction':<12}  {'LFS':>4}  "
                f"{'Rank':>4}  {'CHAIN-Product':<15}  {'GENDF-ELFS[eV]':>14}  "
                f"{'DK-ELIS[eV]':>14}  {'Δ(ELFS-ELIS)':>22}  "
                f"{'Phase-1 held':<16}  {'Crossing?':<9}  {'Routed by':<50}\n")
        f.write("-" * 190 + "\n")
        crossed = _hybrid_crossings(isomer_mappings)
        for m in sorted(phase2, key=lambda x: (str(x.get('parent')),
                                               x.get('mt') or 0,
                                               x.get('lfs') or 0)):
            elis, dk_elis = m.get('elis'), m.get('dk_elis')
            if isinstance(elis, (int, float)) and dk_elis:
                delta = f"{abs(elis - dk_elis):.0f}eV ({abs(elis - dk_elis)/abs(dk_elis)*100:.2f}%)"
            else:
                delta = "N/A"
            claimed = m.get('phase1_claimed_lisos') or []
            claimed_str = ", ".join(f"_m{i}" for i in claimed) or 'none'
            rank = m.get('position')
            reason = HYBRID_REASON_LABELS.get(m.get('fallback_reason'),
                                              str(m.get('fallback_reason')))
            f.write(f"{str(m.get('parent')):<12}  {str(m.get('mt')):>5}  "
                    f"{str(m.get('reaction')):<12}  {str(m.get('lfs')):>4}  "
                    f"{('-' if rank is None else str(rank)):>4}  "
                    f"{str(m.get('product')):<15}  {_elfs_str(elis)}  "
                    f"{_elfs_str(dk_elis)}  {delta:>22}  {claimed_str:<16}  "
                    f"{('YES' if id(m) in crossed else 'no'):<9}  "
                    f"{reason:<50}\n")
    f.write("\n")

    f.write("PLACEHOLDER LFS (unidentified excited states)\n")
    f.write("-" * 150 + "\n")
    f.write("A placeholder LFS says 'an isomer, level unidentified'. It never "
            "takes a rank among the real levels and never displaces one. With "
            "its energy unknown or\n")
    f.write("unmatched it binds, after Phase 2, to the lowest decay state "
            "still unclaimed; with no state left it is reported only, and the "
            "share it would have carried\n")
    f.write("stays with the reaction's other targets.\n")
    if not (bound or unmapped):
        f.write("None found.\n")
    else:
        surviving = defaultdict(list)
        for m in isomer_mappings:
            surviving[(m.get('parent'), m.get('mt'))].append(str(m.get('product')))
        for r in bound:
            f.write(f"  BOUND      {str(r.get('parent')):<10} "
                    f"{str(r.get('reaction')):<12} MT={str(r.get('mt')):<4} "
                    f"LFS={r.get('lfs')} -> {r.get('base_nuclide')}_m{r.get('liso')} "
                    f"({'sole' if r.get('n_unclaimed') == 1 else 'lowest of ' + str(r.get('n_unclaimed'))}"
                    f" unclaimed state, DK-ELIS={_elfs_str(r.get('dk_elis'), 1).strip()} eV, "
                    f"{HYBRID_REASON_LABELS.get(r.get('energy_status'), r.get('energy_status'))})\n")
        for r in unmapped:
            kept = surviving.get((r.get('parent'), r.get('mt')), [])
            dest = ", ".join(sorted(set(kept))) or 'the ground product'
            f.write(f"  UNMAPPED   {str(r.get('parent')):<10} "
                    f"{str(r.get('reaction')):<12} MT={str(r.get('mt')):<4} "
                    f"LFS={r.get('lfs')} -> no state left to bind "
                    f"({HYBRID_REASON_LABELS.get(r.get('energy_status'), r.get('energy_status'))}); "
                    f"its share spreads pro-rata over ground + {dest}\n")
    f.write("\n")


def _write_orphan_disposition_section(f, orphan_levels, orphan_nuclides_added,
                                      reattributions, renormalizations,
                                      orphan_policy):
    """ORPHAN DISPOSITION: every product that had no partner in the chain.

    Written in every mapping mode. An orphan is a reaction product the decay
    library cannot account for: an excited state whose identity did not
    resolve, or a whole product nuclide the library never carried.
    """
    levels = {(o['parent'], o['reaction'], o['name']): o
              for o in (orphan_levels or [])}
    rows = {}

    def _row(parent, reaction, mt, product):
        key = (parent, reaction, product)
        detail = levels.get(key, {})
        return rows.setdefault(key, {
            'parent': parent, 'reaction': reaction, 'mt': mt,
            'product': product, 'lfs': detail.get('lfs'),
            'position': detail.get('position'), 'elis': detail.get('elis'),
            'reason': detail.get('fallback_reason'), 'disposition': None,
            'extra': [],
        })

    for name, sources in (orphan_nuclides_added or {}).items():
        for s in sources:
            row = _row(s['parent'], s['reaction'], s['mt'], name)
            row['disposition'] = (
                f"ADDED to chain as {name} (stable, no decay)"
                if s.get('created', True) else
                f"kept -- shares the orphan nuclide {name} added for another "
                "pathway")
            if row['lfs'] is None:
                row['lfs'] = s.get('lfs')
            if row['elis'] is None:
                row['elis'] = s.get('elis')
            if row['position'] is None:
                row['position'] = s.get('position')

    for r in (reattributions or []):
        row = _row(r['nuclide'], r['reaction'], r['mt'], r['orphan'])
        if row['lfs'] is None:
            row['lfs'] = r.get('lfs')
        if row['elis'] is None:
            row['elis'] = r.get('elis')
        if row['position'] is None:
            row['position'] = r.get('position')
        if r.get('recipient') is None:
            row['disposition'] = ("NO RECIPIENT -- share spread pro-rata "
                                  "(renorm for this reaction)")
        else:
            dist = r.get('rank_distance')
            delta = r.get('delta_elfs')
            row['disposition'] = (
                f"folded into {r['recipient']} "
                f"(rank distance {'?' if dist is None else dist}, "
                f"ΔELFS {'n/a' if delta is None else f'{delta:.0f} eV'})")
        for s in r.get('siblings', []):
            row['extra'].append(
                f"kept isomer {s['product']} rank="
                f"{'-' if s['position'] is None else s['position']} "
                f"GENDF-ELFS={_elfs_str(s['elis'], 1).strip()} eV "
                f"DK-ELIS={_elfs_str(s['dk_elis'], 1).strip()} eV")

    for r in (renormalizations or []):
        for dropped in r.get('dropped_products', []):
            row = _row(r['nuclide'], r['reaction'], r['mt'], dropped)
            kept = ", ".join(r.get('valid_products', [])) or 'nothing'
            row['disposition'] = f"dropped, share pro-rata over {kept}"

    for key, o in levels.items():
        row = _row(o['parent'], o['reaction'], o['mt'], o['name'])
        if row['disposition'] is None:
            row['disposition'] = ("not written -- the reaction lost its "
                                  "decoration elsewhere")

    f.write("\n\n" + "=" * 220 + "\n")
    f.write("ORPHAN DISPOSITION\n")
    f.write("=" * 220 + "\n\n")
    f.write("An ORPHAN is a reaction product with no partner in the chain: an "
            "excited state whose identity did not resolve against the decay "
            "library, or a whole\n")
    f.write("product nuclide that library never carried. The reaction rate "
            "into it is real either way, so the patcher never drops it "
            "silently -- what happens to its\n")
    f.write(f"share is chosen at patch time with --orphan-policy (this run: "
            f"{orphan_policy}):\n\n")
    f.write("  add-stable   the product is added to the output chain with no "
            "decay data and the branch is kept. Mass is conserved and the "
            "pathway stays visible;\n")
    f.write("               the state's own activity is not modelled (see the "
            "next section).\n")
    f.write("  renorm       the product is dropped and its share is spread "
            "pro-rata over the surviving targets of that reaction. The share "
            "is not lost, but it is\n")
    f.write("               attributed to states that did not produce it.\n")
    f.write("  reattribute  the share is folded into the kept isomer of the "
            "same product at the nearest LOWER rank (ground when nothing sits "
            "below it). Rank, not\n")
    f.write("               energy, decides: an orphan is here precisely "
            "because its energy failed to identify it, and a high-lying state "
            "cascades down, never up.\n\n")
    f.write("Wording note for a mapping-only (ratio-free) chain: 'share' "
            "means the reaction rate into that state. Where a chain carries no "
            "branching ratios at all,\n")
    f.write("'dropped' means the partial is never collapsed and the parent "
            "under-burns by exactly that fraction -- a different outcome from "
            "the pro-rata spread here.\n\n")

    if not rows:
        f.write("No orphan products: every reaction product had a partner in "
                "the chain.\n")
        return

    f.write(f"Total orphan pathways: {len(rows)}\n\n")
    f.write(f"{'Parent':<12}  {'MT':>5}  {'Reaction':<12}  {'LFS':>4}  "
            f"{'Rank':>4}  {'GENDF-ELFS[eV]':>14}  {'Orphan product':<15}  "
            f"{'Why orphaned':<52}  {'Disposition'}\n")
    f.write("-" * 220 + "\n")
    for key in sorted(rows, key=lambda k: (str(k[0]), str(k[1]), str(k[2]))):
        row = rows[key]
        reason = HYBRID_REASON_LABELS.get(row['reason'],
                                          row['reason'] or
                                          'product not in the chain')
        rank = row['position']
        f.write(f"{str(row['parent']):<12}  {str(row['mt']):>5}  "
                f"{str(row['reaction']):<12}  "
                f"{('-' if row['lfs'] is None else str(row['lfs'])):>4}  "
                f"{('-' if rank is None else str(rank)):>4}  "
                f"{_elfs_str(row['elis'])}  {row['product']:<15}  "
                f"{reason:<52}  {row['disposition']}\n")
        for line in row['extra']:
            f.write(f"{'':<12}  {'':>5}  {'':<12}  -> {line}\n")
    f.write("\n")


def _write_orphan_nuclides_section(f, orphan_nuclides_added):
    """ORPHAN NUCLIDES ADDED TO CHAIN (--orphan-policy add-stable)."""
    f.write("\n\n" + "=" * 220 + "\n")
    f.write("ORPHAN NUCLIDES ADDED TO CHAIN (--orphan-policy add-stable)\n")
    f.write("=" * 220 + "\n\n")
    if not orphan_nuclides_added:
        f.write("No nuclides were added to the chain.\n")
        return

    f.write("Each nuclide below was written into the output chain as "
            "<nuclide name=\"...\" reactions=\"0\"/> -- no decay data, so the "
            "chain reads it as STABLE.\n")
    f.write("What that buys and what it costs, plainly:\n")
    f.write("  KEPT      the reaction pathway into the state survives, with "
            "its own branching share; nothing is renormalized away.\n")
    f.write("  SINK      the state never decays in the calculation. If the "
            "real state is short-lived, its daughters are never produced and "
            "its decay radiation\n")
    f.write("            is missing from the results -- the inventory holds "
            "material at a level that should have moved on.\n")
    f.write("  NAME      the _mN suffix is a free ordinal for that Z/A, "
            "allocated against the decay-library LISO indices and the existing "
            "chain names. It is NOT a\n")
    f.write("            decay-library isomeric state number and must not be "
            "read as one.\n")
    f.write("  SHARED    several parents feeding the same unidentified state "
            "share one nuclide. A wide spread of GENDF-ELFS among those "
            "parents means they do not\n")
    f.write("            agree on which state it is; a narrow spread means "
            "they do, and only the decay library is missing it.\n\n")
    f.write("A documented limitation: the decay library is the ONLY authority "
            "on state identity here. The chain filters names, it never "
            "testifies to identity -- a\n")
    f.write("state absent from the chain may be absent from the decay library, "
            "or present under a different index. So an added nuclide can be "
            "wrong in BOTH\n")
    f.write("directions: a genuine isomer whose identity was lost (its real "
            "decay is now missing) or a prompt, sub-second level held up as if "
            "it were long-lived\n")
    f.write("(mass parked where it should have flowed on within the first time "
            "step). The remedy is a decay library that carries the state, not "
            "a better guess here.\n\n")

    f.write(f"Nuclides added: {len(orphan_nuclides_added)}\n\n")
    for name in sorted(orphan_nuclides_added):
        sources = orphan_nuclides_added[name]
        energies = [s['elis'] for s in sources
                    if isinstance(s.get('elis'), (int, float))]
        if len(energies) > 1:
            spread = (f"GENDF-ELFS {min(energies):.1f} .. {max(energies):.1f} eV "
                      f"(spread {max(energies) - min(energies):.1f} eV)")
        elif energies:
            spread = f"GENDF-ELFS {energies[0]:.1f} eV"
        else:
            spread = "no GENDF-ELFS on file"
        f.write(f"  {name:<14} from {len(sources)} pathway(s); {spread}\n")
        for s in sorted(sources, key=lambda x: (str(x['parent']),
                                                x.get('mt') or 0)):
            f.write(f"      {str(s['parent']):<10} {str(s['reaction']):<12} "
                    f"MT={str(s['mt']):<4} LFS="
                    f"{'-' if s.get('lfs') is None else s['lfs']:<4} "
                    f"ELFS={_elfs_str(s.get('elis'), 1).strip():<12} "
                    f"mapper={s.get('method') or '-'}\n")
    f.write("\n")


def write_isomer_mapping_log(isomer_mappings, log_file, stats=None, elis_errors=None,
                             duplicate_mapping_errors=None, lfs_order_dropped=None,
                             lfs_order_orphan_dk=None, single_target_cases=None,
                             lfs_placeholders=None, audit_rows=None,
                             rejected_rows=None, attribution_records=None,
                             attribution_counts=None,
                             reattribute_mf10_noizap=False,
                             hybrid_records=None, orphan_levels=None,
                             orphan_nuclides_added=None, reattributions=None,
                             orphan_policy='renorm',
                             reaction_type_missing=None,
                             q55_rows=None, q55_counts=None):
    """Write comprehensive isomer mapping log."""
    if elis_errors is None:
        elis_errors = []
    if lfs_placeholders is None:
        lfs_placeholders = []
    if duplicate_mapping_errors is None:
        duplicate_mapping_errors = []
    if lfs_order_dropped is None:
        lfs_order_dropped = []
    if lfs_order_orphan_dk is None:
        lfs_order_orphan_dk = []
    if single_target_cases is None:
        single_target_cases = []
    hybrid_records = hybrid_records or []
    orphan_levels = orphan_levels or []
    orphan_nuclides_added = orphan_nuclides_added or {}
    reattributions = reattributions or []
    reaction_type_missing = reaction_type_missing or []

    # A level handed to the positional phase, and a duplicate loser requeued
    # with it, are MAPPED -- they must never appear in a not-mapped table or
    # trigger a renormalization note, whatever the caller passed in.
    elis_errors = [e for e in elis_errors if not e.get('routed_to_fallback')]
    duplicate_mapping_errors = [e for e in duplicate_mapping_errors
                                if not e.get('requeued')]

    mapping_mode = stats.get('mapping_mode', 'elis') if stats else 'elis'
    hybrid = mapping_mode == 'elis_lfs_order'
    renormalizations = stats.get('renormalizations', []) if stats else []

    with open(log_file, 'w') as f:
        f.write("=" * 220 + "\n")
        f.write("ISOMER MAPPING LOG\n")
        f.write("=" * 220 + "\n\n")

        # Mapping mode header
        f.write("MAPPING MODE\n")
        f.write("-" * 70 + "\n")
        if mapping_mode == 'lfs_order':
            f.write("MODE: LFS_ORDER (FISPACT-like positional mapping)\n")
            f.write("\n")
            f.write("  Maps GENDF LFS by sorted position: 1st LFS → _m1, 2nd → _m2, etc.\n")
            f.write("  WARNING: May produce incorrect results for nuclides where\n")
            f.write("           LFS order does not match LISO order (e.g., Ag116).\n")
            f.write("           Use 'elis' mode for production calculations.\n")
            f.write("\n")
            f.write("  ELIS data shown below is for REFERENCE ONLY - not used for mapping.\n")
        elif hybrid:
            f.write("MODE: ELIS_LFS_ORDER (excitation energy first, then position)\n")
            f.write("\n")
            f.write("  Phase 1 -- energy. A level is bound to the decay state whose\n")
            f.write("    excitation energy matches its GENDF-ELFS (QM - QI) within the\n")
            f.write("    tolerance below, and ONLY when that state is the single\n")
            f.write("    candidate within it. The level abstains, and waits for Phase 2,\n")
            f.write("    when two decay states both fit (energy-degenerate isomers), when\n")
            f.write("    the file gives no usable ELFS (no QM, zero or negative energy),\n")
            f.write("    when every decay state of the product carries ELIS = 0, when the\n")
            f.write("    nearest state lies outside tolerance, or when a closer level\n")
            f.write("    claimed the same state (the loser is requeued, never dropped).\n")
            f.write("  Phase 2 -- position. The levels left over, in level order, pair\n")
            f.write("    with the decay metastables left over, in LISO order. States\n")
            f.write("    already claimed in Phase 1 are skipped, so no pairing collides.\n")
            f.write("    Energies play no part here; a large energy gap is flagged for\n")
            f.write("    audit, not acted on.\n")
            f.write("  Crossings are permitted. A Phase-1 energy match is trusted\n")
            f.write("    absolutely, so a Phase-2 pairing may end up order-inverted\n")
            f.write("    against it (level 5 to _m2 while level 10 takes _m1). The\n")
            f.write("    PHASE-2 table marks those rows.\n")
            f.write("  Placeholder LFS values (99 ENDF/JEFF, 40 TENDL) mean 'an isomer,\n")
            f.write("    level unidentified'. They never take a rank among real levels\n")
            f.write("    and never displace one. With a usable, distinct energy a\n")
            f.write("    placeholder competes in Phase 1 like any level; otherwise it\n")
            f.write("    binds AFTER Phase 2 to the lowest decay state still unclaimed,\n")
            f.write("    and is reported only when none is left. A placeholder is never\n")
            f.write("    made into an orphan nuclide and never named _m99 / _m40.\n")
            f.write("  Phase 3 -- orphans. A level with no decay state left is kept as an\n")
            f.write("    orphan; --orphan-policy decides its fate (ORPHAN DISPOSITION).\n")
        else:
            f.write("MODE: ELIS (decay library excitation energy matching)\n")
            f.write("\n")
            f.write("  Maps GENDF MF=10 products to OpenMC _m{n} naming based on\n")
            f.write("  excitation energy (ELIS) matching with decay library.\n")
        f.write("\n")
        f.write(f"ORPHAN POLICY: {orphan_policy}\n")
        f.write("\n")

        # What still leaves a pathway out, even at the most permissive
        # settings. Named here so 'nothing is dropped' is never read as a
        # blanket claim.
        f.write("NOT COVERED BY THE ORPHAN POLICY -- pathways that still leave "
                "the chain\n")
        f.write("-" * 70 + "\n")
        f.write("  1. Parent nuclide absent from the chain: the whole GENDF "
                "nuclide has\n")
        f.write("     nowhere to attach.\n")
        f.write("  2. Reaction absent from the chain nuclide: the GENDF file "
                "has the\n")
        f.write("     channel, the chain nuclide carries no such reaction. "
                "Listed below\n")
        f.write("     by parent and MT.\n")
        f.write("  3. Unrepaired anonymous (IZAP=0) MF=10 subsection: the "
                "reaction loses\n")
        f.write("     its ENTIRE isomeric decoration and keeps the plain MF=3 "
                "route --\n")
        f.write("     all-or-nothing, because decorating the named subset alone "
                "would\n")
        f.write("     invert the branching (--reattribute-mf10-noIZAP recovers "
                "most).\n")
        f.write("  4. MF=10-vs-MF=3 band rejection "
                "(--mf10-reject-band-ratio, off by\n")
        f.write("     default): the reaction is left stock.\n")
        f.write("  5. Placeholder LFS with no decay state left to bind: "
                "report-only. Its\n")
        f.write("     share stays with the reaction's other targets, "
                "pro-rata.\n")
        f.write("  6. MF=10-only channels with TWO metastable levels and no "
                "ground\n")
        f.write("     subsection: not emitted at all -- one static target "
                "would take the\n")
        f.write("     whole rate, and with no MF=3 total the ground cannot be "
                "synthesized.\n")
        f.write("     The EAF-2010 (n,n') set: "
                + ", ".join(EAF_DUAL_METASTABLE_NN) + ".\n")
        if reaction_type_missing:
            f.write("\n  Case 2 in this run "
                    f"({len(reaction_type_missing)} reaction(s)):\n")
            for rec in sorted(reaction_type_missing,
                              key=lambda r: (str(r['nuclide']), r['mt'] or 0)):
                f.write(f"    {str(rec['nuclide']):<10} "
                        f"{str(rec['reaction']):<12} MT={rec['mt']}\n")
        f.write("\n\n")

        # Source files
        if stats:
            f.write("SOURCE FILES\n")
            f.write("-" * 70 + "\n")
            f.write(f"OpenMC Chain:    {stats.get('base_chain_file', 'N/A')}\n")
            f.write(f"GENDF Library:   {stats.get('gendf_dir', 'N/A')}\n")
            f.write(f"Decay Library:   {stats.get('decay_file', 'N/A')}\n")
            f.write("-" * 70 + "\n")
            f.write(f"Output Chain:    {stats.get('output_chain_file', 'N/A')}\n")

            f.write("\n\nBASE CHAIN DETAILS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total nuclides in chain: {stats.get('chain_nuclides_total', 0)}\n")

            f.write("\n\nMATCHING SETTINGS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Mapping mode:    {mapping_mode}\n")
            f.write(f"Orphan policy:   {orphan_policy}\n")
            f.write(f"ELIS tolerance:  rtol={stats.get('elis_rtol')} ({stats.get('elis_rtol', 0)*100:.0f}%), atol={stats.get('elis_atol')} eV\n")
            if hybrid:
                f.write("Default tolerance in this mode is 0.15; a Phase-1 match is trusted\n")
                f.write("absolutely, so a level that misses it is re-derived by position\n")
                f.write("rather than lost.\n\n")
            elif mapping_mode == 'elis':
                f.write("Products beyond tolerance or not in decay library are skipped.\n")
                f.write("Skipped products trigger renormalization for isomeric branching to remaining isomers (constant reaction rate).\n\n")
            else:
                f.write("In LFS-order mode: tolerance used for ELIS reference warnings only.\n")
                f.write("Products dropped if GENDF LFS count > DK-Lib LISO count.\n\n")

        # Summary counts
        err_types = ('elis_tol_exceeded', 'no_metastable_decay_data',
                     'zero_elis_metastables', 'elis_incomputable')

        def _n_method(name):
            return sum(1 for m in isomer_mappings if m.get('method') == name)

        def _n_hybrid(err_type, reason=None):
            return sum(1 for r in hybrid_records
                       if r.get('type') == err_type
                       and (reason is None
                            or r.get('fallback_reason') == reason))

        elis_matched_count = _n_method('elis')
        lfs_order_mapped_count = _n_method('lfs_order')
        # Method census: a fallback, placeholder-bound or orphan row is MAPPED.
        fallback_count = _n_method('lfs_order_fallback')
        placeholder_bound_count = _n_method('placeholder_bound')
        orphan_added_count = _n_method('orphan_added')
        mapped_count = (elis_matched_count + lfs_order_mapped_count
                        + fallback_count + placeholder_bound_count
                        + orphan_added_count)
        abstained_count = _n_hybrid('hybrid_fallback', 'elis_ambiguous')
        requeued_count = _n_hybrid('hybrid_fallback', 'duplicate_loser')
        unusable_count = sum(_n_hybrid('hybrid_fallback', r)
                             for r in HYBRID_UNUSABLE_REASONS)
        placeholder_unmapped_count = _n_hybrid('placeholder_unmapped')
        incomputable_count = sum(1 for e in elis_errors
                                 if e.get('type') == 'elis_incomputable')
        elis_exceeded_count = sum(1 for e in elis_errors if e.get('type') == 'elis_tol_exceeded')
        missing_meta_count = sum(1 for e in elis_errors if e.get('type') == 'no_metastable_decay_data')
        zero_elis_count = sum(1 for e in elis_errors if e.get('type') == 'zero_elis_metastables')
        dup_mapping_count = len(duplicate_mapping_errors)
        dup_discarded_count = sum(len(e.get('discarded', [])) for e in duplicate_mapping_errors)
        dropped_count = len(lfs_order_dropped)
        orphan_count = len(lfs_order_orphan_dk)

        # Count unique nuclides with branching
        nuclides_with_branching = set()
        for m in isomer_mappings:
            nuclides_with_branching.add(m.get('parent'))
        for e in elis_errors:
            if e.get('type') in err_types:
                nuclides_with_branching.add(e.get('parent'))
        for e in lfs_order_dropped:
            nuclides_with_branching.add(e.get('parent'))
        for e in hybrid_records:
            nuclides_with_branching.add(e.get('parent'))

        gendf_total = stats.get('gendf_nuclides_total', 0) if stats else 0

        # Calculate totals based on mode
        if mapping_mode == 'elis':
            total = (mapped_count + elis_exceeded_count + missing_meta_count
                     + zero_elis_count + incomputable_count)
            total_gendf_lfs = total + dup_discarded_count
        elif hybrid:
            # Every level lands in exactly one of: Phase 1, Phase 2,
            # placeholder-bound, orphan, placeholder-unmapped. The abstention
            # and requeue counters below are memo lines: those levels are
            # already counted in Phase 2, not alongside it.
            total = mapped_count + placeholder_unmapped_count
            total_gendf_lfs = total
        else:
            total = mapped_count + dropped_count
            total_gendf_lfs = total

        def _count(label, value):
            """Counter line, label right-aligned to the block's column."""
            f.write(f"{label:>47}: {value:5d}\n")

        f.write(f"                      nuclides in GENDF library: {gendf_total:5d}\n")
        f.write(f"               nuclides in GENDF with branching: {len(nuclides_with_branching):5d}\n")
        f.write("-" * 52 + "\n")
        f.write(f"                          Total GENDF-LFS found: {total_gendf_lfs:5d}\n")
        if mapping_mode == 'elis':
            f.write(f"                                   ELIS matched: {mapped_count:5d}\n")
        elif hybrid:
            _count('Phase-1 ELIS matched (unique)', elis_matched_count)
            _count('Phase-2 positional fallback-mapped', fallback_count)
            _count('Placeholder LFS bound', placeholder_bound_count)
            _count('Phase-3 orphan levels', orphan_added_count)
            _count('Placeholder LFS unmapped (report-only)',
                   placeholder_unmapped_count)
            f.write(f"{'(the five lines above sum to the total)':>54}\n")
            _count('memo: Phase-1 abstained (ambiguous)', abstained_count)
            _count('memo: Phase-1 unusable (QM/ELIS=0/rtol)', unusable_count)
            _count('memo: duplicates requeued to Phase 2', requeued_count)
            _count('Orphan nuclides ADDED to chain',
                   len(orphan_nuclides_added))
            _count('Branches reattributed (fold-down)',
                   sum(1 for r in reattributions if r.get('recipient')))
            _count('Branches renormalized (pro-rata)', len(renormalizations))
        else:
            f.write(f"                              LFS-order mapped: {mapped_count:5d}\n")
        if not hybrid:
            if orphan_nuclides_added:
                _count('Orphan nuclides ADDED to chain',
                       len(orphan_nuclides_added))
            if reattributions:
                _count('Branches reattributed (fold-down)',
                       sum(1 for r in reattributions if r.get('recipient')))
            if renormalizations:
                _count('Branches renormalized (pro-rata)',
                       len(renormalizations))
        if incomputable_count:
            _count('ELIS incomputable (no QM on the level)',
                   incomputable_count)
        if elis_exceeded_count:
            f.write(f"                             ELIS rtol exceeded: {elis_exceeded_count:5d}\n")
        if missing_meta_count:
            f.write(f"                           No product in DK-Lib: {missing_meta_count:5d}\n")
        if zero_elis_count:
            f.write(f"                       Zero-ELIS in DK-Lib (QA): {zero_elis_count:5d}\n")
        if dropped_count:
            f.write(f"                 LFS dropped (exceeds DK count): {dropped_count:5d}\n")
        if orphan_count:
            f.write(f"                     Orphan DK states (no LFS): {orphan_count:5d}\n")
        if dup_mapping_count:
            f.write(f"                    Duplicate mappings resolved: {dup_mapping_count:5d} ({dup_discarded_count} LFS discarded)\n")
        if lfs_placeholders:
            f.write(f"                    LFS placeholder occurrences: {len(lfs_placeholders):5d}\n")
        if audit_rows is not None and stats is not None:
            f.write(f"                          MF=10 audit offenders: {stats.get('audit_offenders', 0):5d}\n")
            f.write(f"                                 MF=10 rejected: {stats.get('mf10_rejected', 0):5d}\n")
            f.write(f"          Band-reject exempt (self-loop ground): {stats.get('band_reject_exempt', 0):5d}\n")
        ground_repaired = stats.get('ground_repaired', []) if stats else []
        ground_absent_skipped = stats.get('ground_absent_skipped', []) if stats else []
        ground_repaired_dropped = (stats.get('ground_repaired_dropped', [])
                                   if stats else [])
        if ground_repaired:
            f.write(f"                   Ground synthesized from MF=3: {len(ground_repaired):5d}\n")
        if ground_repaired_dropped:
            f.write(f"        Ground synthesized, decoration dropped: {len(ground_repaired_dropped):5d}\n")
        if ground_absent_skipped:
            # Not "unrepairable": the set also holds repaired-then-unmapped ones.
            f.write(f"                   Ground absent, not decorated: {len(ground_absent_skipped):5d}\n")
        pathway_q = stats.get('pathway_q_corrections', []) if stats else []
        qm_disagree = (stats.get('pathway_q_qm_disagreements', [])
                       if stats else [])
        q_rejected = stats.get('pathway_q_rejected', []) if stats else []
        if pathway_q:
            f.write(f"                          Pathway-Q QM/QI-corrected: {len(pathway_q):5d}\n")
        if q_rejected:
            f.write(f"             Pathway-Q file QM rejected (reactions): {len(q_rejected):5d}\n")
        if stats is not None:
            # Both counters always: a fallback-heavy run must be visible here,
            # not only by its absence from the corrections list.
            f.write(f"                   Pathway-Q slots from MF=10 QM/QI: {stats.get('q_from_mf10', 0):5d}\n")
            if stats.get('q_from_elis'):
                f.write(f"               Pathway-Q slots from ELIS arithmetic: {stats['q_from_elis']:5d}\n")
            if stats.get('q_chain_anchored'):
                f.write(f"Pathway-Q slots chain-anchored (gate, incl. ground): {stats['q_chain_anchored']:5d}\n")
            # Unconditional: a skipped gate is a silent non-event otherwise.
            f.write(f"       Pathway-Q gate skipped (zero-Q chain anchor): {stats.get('q_zero_anchor', 0):5d}\n")
            f.write(f"              Pathway-Q slots replicated (scalar Q): {stats.get('q_replicated', 0):5d}\n")
        if qm_disagree:
            f.write(f"                 Pathway-Q QM disagreement sections: {len(qm_disagree):5d}\n")
        f.write("\n")

        # Policy 3(a): metastable-only MF=10 (stable ground omitted by the
        # evaluator) with a true MF=3 total -> ground = clamped remainder.
        if ground_repaired or ground_absent_skipped or ground_repaired_dropped:
            f.write("GROUND-ABSENT MF=10 (radioactive-products-only files)\n")
            f.write("-" * 93 + "\n")
            for rec in ground_repaired:
                clamped = rec.get('clamped_points', 0)
                clamp = (f" [clamped {clamped}/{rec.get('total_points', 0)} pts]"
                         if clamped else "")
                f.write(f"  REPAIRED  {rec['nuclide']:<10} {rec['reaction']:<12} "
                        f"MT={rec['mt']:<4} ground {rec['ground_product']} = "
                        f"max(0, MF3 - sum MF10 LFS{rec['metastable_lfs']})"
                        f"{clamp}\n")
            for rec in ground_repaired_dropped:
                f.write(f"  DROPPED   {rec['nuclide']:<10} {rec['reaction']:<12} "
                        f"MT={rec['mt']:<4} ground {rec['ground_product']} "
                        "synthesized, but it was the only product left in the "
                        "chain; decoration not written\n")
            for rec in ground_absent_skipped:
                label = GROUND_ABSENT_SKIP_LABELS.get(rec['reason'])
                f.write(f"  SKIPPED   {rec['nuclide']:<10} {rec['reaction']:<12} "
                        f"MT={rec['mt']:<4} {label or rec['reason']}\n")
            f.write("\n")

        # Per-pathway Q now read off the MF=10 QM/QI pair instead of being
        # derived as (chain scalar Q - ELFS), which double-charged the level
        # energy for MT=4. Every slot whose written value differs from that
        # legacy arithmetic is listed.
        if pathway_q:
            f.write("PATHWAY-Q (QM/QI-SOURCED)\n")
            f.write("-" * 93 + "\n")
            f.write("Convention: pathway Q = MF=10 QI of that level; the LFS=0 slot = the\n")
            f.write("LFS=0 subsection's own QM; the reaction's scalar Q is untouched. Listed\n")
            f.write("below: reactions whose written per-slot Q differs from the legacy fold\n")
            f.write("(Q_scalar - ELFS). This re-sourcing applies to EVERY MT, not only MT=4.\n")
            f.write("The (n,n') rows are the MT=4 double-subtraction fix -- there the chain's\n")
            f.write("scalar Q is QI(MF=3) = -E(level 1), not QM, so subtracting ELFS charged\n")
            f.write("the level energy twice (and left the ground slot at -E(level 1)). Every\n")
            f.write("other row is ordinary re-sourcing of a ground or metastable slot, where\n")
            f.write("the file's QM/QI simply differs from the chain's scalar Q.\n")
            f.write("Sources: QM/QI = the level's own MF=10 pair; ELIS = legacy Q_scalar -\n")
            f.write("ELFS (no Q pair in the input); scalar = the chain Q replicated.\n")
            f.write("File values are adopted only where they are consistent with the chain:\n")
            f.write("outside MT=4 the file's ground QM must match the chain's scalar Q to\n")
            f.write(f"within {_q_str(PATHWAY_Q_CHAIN_TOL)} eV, or the whole reaction keeps the legacy values --\n")
            f.write("see PATHWAY-Q FILE-QM REJECTED below for the reactions that failed that.\n")
            f.write("\n")
            for rec in pathway_q:
                slots = "  ".join(
                    f"{t}[LFS={l}] {q} ({s}, was {old})"
                    for t, l, q, s, old in zip(rec['targets'], rec['lfs'],
                                               rec['q'], rec['source'],
                                               rec['q_legacy']))
                f.write(f"  {rec['nuclide']:<10} {rec['reaction']:<12} "
                        f"MT={rec['mt']:<4} {slots}\n")
            f.write("\n")

        # The sanity gate's ledger. A rejected reaction is INVISIBLE in the
        # section above (it writes the legacy values verbatim, so it is not a
        # "correction"), which is exactly why the refused file values have to be
        # printed here -- otherwise a 12 MeV file defect would leave no trace.
        if q_rejected:
            f.write("PATHWAY-Q FILE-QM REJECTED (chain-anchored values retained)\n")
            f.write("-" * 93 + "\n")
            f.write("Sanity gate: for a non-MT=4 reaction that has a scalar Q in the chain, the\n")
            f.write(f"file's own ground QM must agree with that scalar to within {_q_str(PATHWAY_Q_CHAIN_TOL)} eV\n")
            f.write("(PATHWAY_Q_CHAIN_TOL) before ANY of the reaction's per-slot file values\n")
            f.write("are adopted. The chain scalar is AME/evaluation-derived; an absolute\n")
            f.write("MF=10 QM is unvalidated and can be badly wrong (ENDF/B-8.1 (n,alpha):\n")
            f.write("7.9-12.0 MeV out, sign included). On failure the WHOLE reaction keeps the\n")
            f.write("legacy fold (Q_scalar - ELFS), which consumed only the QM-QI difference\n")
            f.write("and is immune to an absolute-scale error; the refused file values are\n")
            f.write("listed here and written nowhere. MT=4 is exempt -- there the scalar Q is\n")
            f.write("QI(MF=3) = -E(level), not QM, so the mismatch IS the defect being fixed.\n")
            f.write("\n")
            for rec in q_rejected:
                delta = _q_str(rec['delta'])
                delta = f"+{delta}" if rec['delta'] >= 0 else delta
                slots = "  ".join(
                    f"{t}[LFS={l}]={qf} (kept {qk})"
                    for t, l, qf, qk in zip(rec['targets'], rec['lfs'],
                                            rec['q_file'], rec['q_kept']))
                f.write(f"  {rec['nuclide']:<10} {rec['reaction']:<12} "
                        f"MT={rec['mt']:<4} file QM={_q_str(rec['file_ground_qm'])} "
                        f"vs chain Q={_q_str(rec['chain_q'])} (delta {delta})  "
                        f"refused: {slots}\n")
            f.write("\n")

        # QM is per-subsection: where the subsections of one MT disagree, the
        # ground slot's value depends on WHICH subsection it came from, so the
        # section is listed for inspection (the written value is always the
        # LFS=0 subsection's own QM, or a mapped level's for a synthesized
        # ground).
        if qm_disagree:
            f.write("PATHWAY-Q INTRA-SECTION QM DISAGREEMENT\n")
            f.write("-" * 93 + "\n")
            f.write("MF=10 subsections of one MT carrying different QM values. 'ground QM' is\n")
            f.write("the LFS=0 subsection's own (n/a = no LFS=0 subsection; the ground was\n")
            f.write("synthesized and the first metastable QM below was used instead). This\n")
            f.write("describes the FILE; a section listed here may also have been refused\n")
            f.write("outright by the sanity gate, in which case none of its QM was written.\n")
            f.write("\n")
            for rec in qm_disagree:
                gqm = ('n/a' if rec['ground_qm'] is None
                       else _q_str(rec['ground_qm']))
                metas = ", ".join(_q_str(q) for q in rec['meta_qms']) or 'none'
                f.write(f"  {rec['nuclide']:<10} {rec['reaction']:<12} "
                        f"MT={rec['mt']:<4} ground QM={gqm}  "
                        f"metastable QM={metas}\n")
            f.write("\n")

        # Column descriptions
        f.write("COLUMN DEFINITIONS\n")
        f.write("-" * 93 + "\n")
        f.write("MT              = ENDF reaction type number\n")
        f.write("\n")
        f.write("Reaction        = Reaction name (e.g., (n,g), (n,2n))\n")
        f.write("\n")
        f.write("GENDF-LFS       = Level number of the state of ZAP formed by the neutron interaction.\n")
        f.write("                  Indicator to specify the level number of the nuclide (ZAP) (as defined in\n")
        f.write("                  File 8) produced in the reaction (MT number).\n")
        f.write("\n")
        f.write("GENDF-Product   = Product nuclide from GENDF reaction (using _m{LFS} naming)\n")
        f.write("\n")
        f.write("DK-LISO         = Decay library isomeric state number (_m1=1, _m2=2, etc.).\n")
        f.write("                  (Only isomeric levels)\n")
        f.write("\n")
        f.write("CHAIN-Product   = Product nuclide name in chain (after ELIS mapping) (using _m{LISO} naming)\n")
        f.write("\n")
        f.write("CHAIN-t1/2      = Half-life from chain (originally from decay library)\n")
        f.write("\n")
        f.write("GENDF-ELFS[eV]  = Excitation energy of final state calculated from GENDF MF=10 (QM - QI).\n")
        f.write("                  Excitation energy of the reaction product.\n")
        f.write("\n")
        f.write("DK-ELIS[eV]     = Excitation energy from decay library MF=1 MT=451 (matched within tolerance).\n")
        f.write("                  Excitation energy of the target nucleus relative to 0.0 for the ground state.\n")
        f.write("\n")
        f.write("Δ(ELFS-ELIS)    = Discrepancy between GENDF-ELFS and DK-ELIS.\n")
        f.write("                  Shows absolute difference [eV] and relative difference [%].\n")
        f.write("\n")
        f.write("Method          = How this level got its chain product:\n")
        f.write("                  - 'ELIS'        = matched on excitation energy (Phase 1 in the hybrid mode)\n")
        f.write("                  - 'LFS_ORDER'   = positional mapping (lfs_order mode)\n")
        f.write("                  - 'LFS-ORD(fb)' = hybrid Phase 2: paired by position after the energy match failed\n")
        f.write("                  - 'PLACEHOLDER' = placeholder LFS (unidentified excited state) bound to the lowest unclaimed state\n")
        f.write("                  - 'ORPHAN+'     = no decay partner at all; state kept under --orphan-policy add-stable\n")
        f.write("                  - 'not-mapped'  = the level was not written to the chain\n")
        f.write("\n")
        f.write("Notes           = Additional information:\n")
        f.write("                  - 'Mass renorm (X omit→pro-rata)' = a sibling was omitted; its share went pro-rata to the rest\n")
        f.write("                  - 'ELIS rtol exceeded' = Product skipped, closest match shown\n")
        f.write("                  - 'Product not in DK-Lib' = No metastable data in decay library\n")
        f.write("                  - 'No QM: ELFS incomputable' = the level's TAB1 head carries no QM, so QM - QI cannot be formed\n")
        f.write("\n")
        f.write("Further info:\n")
        f.write("  | LIS/LFS | Level number          | All excited states (short-lived + long-lived) |\n")
        f.write("  | LISO    | Isomeric state number | Only Isomeric (metastable/long-lived) states  |\n")
        f.write("\n")
        f.write("=" * 93 + "\n\n")

        # Build map of (parent, mt) → skipped products for renormalization notes
        skipped_by_reaction = defaultdict(list)
        for error in elis_errors:
            err_type = error.get('type')
            if err_type == 'elis_tol_exceeded':
                parent = error.get('parent', '?')
                mt = error.get('mt')
                lfs = error.get('lfs')
                liso = error.get('liso')
                base_nuc = error.get('base_nuclide', '?')
                # Build product name from LISO (closest match shown)
                skipped_name = f"{base_nuc}_m{liso}" if liso else f"{base_nuc}_m{lfs}"
                skipped_by_reaction[(parent, mt)].append({
                    'name': skipped_name,
                    'reason': 'rtol exceeded'
                })
            elif err_type == 'no_metastable_decay_data':
                # For no_metastable_decay_data, try to get info from error dict or parse
                parent = error.get('parent')
                mt = error.get('mt')
                lfs = error.get('lfs')
                base_nuc = error.get('base_nuclide', '?')
                if parent and mt:
                    skipped_name = f"{base_nuc}_m{lfs}" if lfs else f"{base_nuc}_m?"
                    skipped_by_reaction[(parent, mt)].append({
                        'name': skipped_name,
                        'reason': 'no DK-Lib data'
                    })
                else:
                    # Parse from error string
                    parsed = _parse_no_metastable_decay_data_error(error.get('error', ''))
                    parent = parsed.get('parent', '?')
                    reaction = parsed.get('reaction', '?')
                    mt = REACTION_TO_MT.get(reaction)
                    skipped_name = parsed.get('product', '?')
                    if parent and mt:
                        skipped_by_reaction[(parent, mt)].append({
                            'name': skipped_name,
                            'reason': 'no DK-Lib data'
                        })

        # Add duplicate_mapping errors to skipped_by_reaction
        for error in duplicate_mapping_errors:
            parent = error.get('nuclide')
            mt = error.get('mt')
            base_nuc = error.get('base_nuclide', '?')
            kept_lfs = error.get('kept_lfs')
            conflict_liso = error.get('liso')  # The LISO that multiple LFS mapped to
            for d in error.get('discarded', []):
                skipped_name = f"LFS={d['lfs']}"
                skipped_by_reaction[(parent, mt)].append({
                    'name': skipped_name,
                    'reason': f'duplicate mapping to _m{conflict_liso} (closer: LFS={kept_lfs})',
                    'lfs': d['lfs'],
                    'liso': conflict_liso  # The LISO they were trying to map to
                })

        # Group by parent and add renormalization notes to successful mappings
        by_parent = defaultdict(list)
        for m in isomer_mappings:
            parent = m['parent']
            mt = m.get('mt')
            # Check if any siblings were skipped for this reaction
            skipped = skipped_by_reaction.get((parent, mt), [])
            if skipped:
                # Add renormalization note - mass renorm means pro-rata redistribution
                skipped_names = [s['name'] for s in skipped]
                m = dict(m)  # Make a copy to avoid modifying original
                m['notes'] = f"Mass renorm ({', '.join(skipped_names)} omit→pro-rata)"
            by_parent[parent].append(m)

        # Add errors (including no_metastable_decay_data with parsed details)
        for error in elis_errors:
            err_type = error.get('type')
            if err_type == 'elis_tol_exceeded':
                # elis_tol_exceeded errors have 'parent' directly in dict
                by_parent[error.get('parent', '?')].append(error)
            elif err_type == 'no_metastable_decay_data':
                # Check if we have structured data or need to parse
                if error.get('parent') and error.get('mt'):
                    by_parent[error.get('parent', '?')].append(error)
                else:
                    # Parse the error string to extract details
                    parsed = _parse_no_metastable_decay_data_error(error.get('error', ''))
                    parsed['type'] = 'no_metastable_decay_data'
                    # Look up MT from reaction name using REACTION_TO_MT
                    reaction = parsed.get('reaction', '')
                    parsed['mt'] = REACTION_TO_MT.get(reaction)
                    by_parent[parsed['parent']].append(parsed)
            elif err_type == 'zero_elis_metastables':
                # Zero-ELIS metastables: structured data from gendf.py
                by_parent[error.get('parent', '?')].append(error)
            elif err_type == 'elis_incomputable':
                # No QM on the level's TAB1 head, so ELFS = QM - QI cannot be
                # formed at all: nothing to match on in this mode.
                by_parent[error.get('parent', '?')].append(error)

        # Add duplicate_mapping discarded entries to by_parent
        for error in duplicate_mapping_errors:
            parent = error.get('nuclide')
            mt = error.get('mt')
            reaction = error.get('reaction', '?')
            base_nuc = error.get('base_nuclide', '?')
            conflict_liso = error.get('liso')
            kept_lfs = error.get('kept_lfs')
            dk_elis = error.get('dk_elis')

            for d in error.get('discarded', []):
                by_parent[parent].append({
                    'type': 'duplicate_mapping',
                    'parent': parent,
                    'mt': mt,
                    'reaction': reaction,
                    'lfs': d['lfs'],
                    'elis': d['elis'],
                    'liso': conflict_liso,
                    'dk_elis': dk_elis,
                    'product': f"{base_nuc}_m{conflict_liso}" if conflict_liso else base_nuc,
                    'base_nuclide': base_nuc,
                    'kept_lfs': kept_lfs,
                    'half_life': '-',
                    'method': 'not-mapped',
                    'notes': f'Dup map to _m{conflict_liso} (LFS={kept_lfs} closer)'
                })

        # Table header
        header = (f"{'MT':>5}  {'Reaction':<12}  {'GENDF-LFS':>9}  {'GENDF-Product':<15}  "
                  f"{'DK-LISO':>7}  {'CHAIN-Product':<15}  {'CHAIN-t½[s]':>12}  "
                  f"{'GENDF-ELFS[eV]':>14}  {'DK-ELIS[eV]':>14}  {'Δ(ELFS-ELIS)':>22}    {'Method':<12}    {'Notes':<30}")
        sep = "-" * 220

        for parent in sorted(by_parent.keys()):
            f.write(f"\n{parent}:\n{sep}\n{header}\n{sep}\n")

            for m in sorted(by_parent[parent], key=lambda x: (x.get('mt') or 0, x.get('method', ''))):
                _write_mapping_row(f, m)

        # HIGH ELIS DISCREPANCY + FAILED MAPPINGS section
        f.write("\n\n" + "=" * 220 + "\n")
        f.write("HIGH ELIS DISCREPANCY (>50%) + FAILED MAPPINGS\n")
        f.write("=" * 220 + "\n\n")

        high_disc = []
        for m in isomer_mappings:
            elis, dk_elis = m.get('elis'), m.get('dk_elis')
            if elis is not None and dk_elis is not None and dk_elis != 0:
                rel_diff = abs(elis - dk_elis) / abs(dk_elis) * 100
                if rel_diff > 50.0:
                    high_disc.append({**m, 'rel_diff': rel_diff, 'notes': ''})

        for err in elis_errors:
            if err.get('type') == 'elis_tol_exceeded' and err.get('omitted', True):
                # Build closest match product name using LISO (not LFS)
                elem = _z_to_element(err.get('target_z', 0))
                mass = err.get('target_a', '?')
                liso = err.get('liso')
                closest_product = f"{elem}{mass}_m{liso}" if liso else f"{elem}{mass}"
                high_disc.append({
                    'parent': err.get('parent'), 'mt': err.get('mt'),
                    'reaction': err.get('reaction'), 'lfs': err.get('lfs'),
                    'product': closest_product,  # Closest match product name
                    'liso': liso, 'elis': err.get('elis'),
                    'dk_elis': err.get('dk_elis'),
                    'half_life': err.get('half_life', '-'),  # From error if available
                    'method': 'not-mapped', 'rel_diff': err.get('diff_percent', 0),
                    'notes': 'ELIS rtol exceeded. Closest shown.'
                })
            elif err.get('type') == 'no_metastable_decay_data':
                # Two error sources: structured (with parent info) or unstructured (exception)
                if 'parent' in err and err.get('parent') is not None:
                    # Structured error with parent info
                    elem = _z_to_element(err.get('target_z', 0))
                    mass = err.get('target_a', '?')
                    lfs = err.get('lfs')
                    high_disc.append({
                        'parent': err.get('parent'), 'mt': err.get('mt'),
                        'reaction': err.get('reaction'), 'lfs': lfs,
                        'product': f"{elem}{mass}_m{lfs}" if lfs else f"{elem}{mass}_m?",
                        'liso': '-', 'elis': err.get('elis'), 'dk_elis': None,
                        'half_life': '-', 'method': 'not-mapped',
                        'rel_diff': float('inf'), 'notes': 'No product in DK-Lib'
                    })
                else:
                    # Unstructured error from exception - parse error string
                    parsed = _parse_no_metastable_decay_data_error(err.get('error', ''))
                    reaction = parsed['reaction']
                    high_disc.append({
                        'parent': parsed['parent'], 'mt': REACTION_TO_MT.get(reaction),
                        'reaction': reaction, 'lfs': parsed['lfs'],
                        'product': parsed['product'],
                        'liso': '-', 'elis': parsed['elis'], 'dk_elis': None,
                        'half_life': '-', 'method': 'not-mapped',
                        'rel_diff': float('inf'), 'notes': 'No product in DK-Lib',
                        'type': 'no_metastable_decay_data'
                    })
            elif err.get('type') == 'elis_incomputable':
                elem = _z_to_element(err.get('target_z', 0))
                mass = err.get('target_a', '?')
                lfs = err.get('lfs')
                high_disc.append({
                    'parent': err.get('parent'), 'mt': err.get('mt'),
                    'reaction': err.get('reaction'), 'lfs': lfs,
                    'product': f"{elem}{mass}_m{lfs}" if lfs else f"{elem}{mass}_m?",
                    'liso': '-', 'elis': None, 'dk_elis': None,
                    'half_life': '-', 'method': 'not-mapped',
                    'rel_diff': float('inf'),
                    'notes': 'No QM: ELFS incomputable',
                    'type': 'elis_incomputable'
                })
            elif err.get('type') == 'zero_elis_metastables':
                # Zero-ELIS metastables: structured data from gendf.py
                elem = _z_to_element(err.get('target_z', 0))
                mass = err.get('target_a', '?')
                lfs = err.get('lfs')
                skipped = err.get('skipped_states', [])
                skipped_str = ', '.join(f"_m{liso}" for liso, _, _ in skipped)
                high_disc.append({
                    'parent': err.get('parent'), 'mt': err.get('mt'),
                    'reaction': err.get('reaction'), 'lfs': lfs,
                    'product': f"{elem}{mass}_m{lfs}" if lfs else f"{elem}{mass}_m?",
                    'liso': '-', 'elis': err.get('elis'), 'dk_elis': 0.0,
                    'half_life': '-', 'method': 'not-mapped',
                    'rel_diff': float('inf'),
                    'notes': f'DK-Lib ELIS=0 ({skipped_str})',
                    'type': 'zero_elis_metastables',
                    'skipped_states': skipped
                })

        high_disc.sort(key=lambda x: x.get('rel_diff', 0), reverse=True)

        if not high_disc:
            f.write("No entries with >50% discrepancy or failed mappings.\n")
        else:
            mapped = sum(1 for h in high_disc if h.get('method') != 'not-mapped')
            failed = len(high_disc) - mapped
            f.write(f"Total: {len(high_disc)} (mapped: {mapped}, not-mapped: {failed})\n\n")

            f.write(f"{sep}\n")
            f.write(f"{'Parent':<12}  {header}\n")
            f.write(f"{sep}\n")

            for m in high_disc:
                f.write(f"{m.get('parent', '?'):<12}  ")
                _write_mapping_row(f, m)

        # ISOMERIC STATE PROXIMITY CHECK section
        rtol = stats.get('elis_rtol', 0.5) if stats else 0.5
        _write_isomeric_proximity_check(f, isomer_mappings, rtol)

        # DUPLICATE MAPPING CONFLICTS section
        _write_duplicate_mapping_section(f, duplicate_mapping_errors)

        # COUNT MISMATCH REPORT: GENDF level count vs decay state count. Keyed
        # on the records themselves -- the hybrid reports leftover decay states
        # the same way, it simply never drops a level to get there.
        orphan_dk_rows = list(lfs_order_orphan_dk) + [
            r for r in hybrid_records if r.get('type') == 'hybrid_orphan_dk']
        if lfs_order_dropped or orphan_dk_rows:
            f.write("\n\n" + "=" * 220 + "\n")
            f.write("COUNT MISMATCH REPORT (GENDF levels vs DK-Lib states)\n")
            f.write("=" * 220 + "\n\n")

            if lfs_order_dropped:
                f.write("DROPPED LFS STATES (GENDF LFS count > DK-Lib LISO count):\n")
                f.write("-" * 100 + "\n")
                f.write(f"{'Parent':<12}  {'MT':>5}  {'Reaction':<12}  {'GENDF-LFS':>9}  "
                        f"{'Would-Be':>10}  {'GENDF-ELFS':>14}  {'Reason':<40}\n")
                f.write("-" * 100 + "\n")
                for err in lfs_order_dropped:
                    parent = err.get('parent', '?')
                    mt = err.get('mt', '?')
                    reaction = err.get('reaction', '?')
                    lfs = err.get('lfs', '?')
                    would_be = f"_m{err.get('would_be_liso', '?')}"
                    gendf_elis = err.get('gendf_elis')
                    elis_str = f"{gendf_elis:.1f}" if gendf_elis is not None else "N/A"
                    dk_count = err.get('dk_meta_count', '?')
                    reason = f"DK-Lib has only {dk_count} metastable(s)"
                    f.write(f"{parent:<12}  {mt:>5}  {reaction:<12}  {lfs:>9}  "
                            f"{would_be:>10}  {elis_str:>14}  {reason:<40}\n")
                f.write("\n")

            if orphan_dk_rows:
                f.write("ORPHAN DK-Lib STATES (DK-Lib LISO count > GENDF LFS count):\n")
                f.write("Informational: the nuclide is already in the chain, it "
                        "simply gains no production from this reaction.\n")
                f.write("-" * 100 + "\n")
                f.write(f"{'Parent':<12}  {'MT':>5}  {'Reaction':<12}  {'DK-LISO':>10}  "
                        f"{'DK-ELIS':>14}  {'Reason':<40}\n")
                f.write("-" * 100 + "\n")
                for err in orphan_dk_rows:
                    parent = err.get('parent', '?')
                    mt = err.get('mt', '?')
                    reaction = err.get('reaction', '?')
                    liso = f"_m{err.get('liso', '?')}"
                    dk_elis = err.get('dk_elis')
                    elis_str = f"{dk_elis:.1f}" if dk_elis is not None else "N/A"
                    gendf_count = err.get('gendf_meta_count', '?')
                    reason = f"GENDF has only {gendf_count} metastable(s)"
                    f.write(f"{parent:<12}  {mt:>5}  {reaction:<12}  {liso:>10}  "
                            f"{elis_str:>14}  {reason:<40}\n")
                f.write("\n")

        # SINGLE-TARGET ISOMERIC YIELDS section
        if single_target_cases:
            f.write("\n\n" + "=" * 220 + "\n")
            f.write("SINGLE-TARGET ISOMERIC YIELDS\n")
            f.write("=" * 220 + "\n\n")

            suppressed = sum(1 for c in single_target_cases if 'suppressed' in c.get('action', ''))
            kept = len(single_target_cases) - suppressed

            f.write(f"Total: {len(single_target_cases)} reactions (suppressed: {suppressed}, kept: {kept})\n\n")
            f.write("These reactions have only a single target product for isomeric branching.\n")
            f.write("This can be redundant if the single target equals the original reaction target.\n\n")

            # Breakdown by reason
            reason_counts = Counter(c['reason'] for c in single_target_cases)
            f.write("Breakdown by reason:\n")
            for reason, count in reason_counts.most_common():
                pct = 100 * count / len(single_target_cases)
                f.write(f"  {reason}: {count} ({pct:.1f}%)\n")
            f.write("\n")

            f.write("-" * 140 + "\n")
            f.write(f"{'Nuclide':<12} {'MT':<6} {'Reaction':<14} {'Target':<16} "
                    f"{'Orig-Target':<16} {'Redundant':<10} {'Action':<16} {'Reason'}\n")
            f.write("-" * 140 + "\n")

            for case in single_target_cases:
                redundant = 'Yes' if case.get('is_redundant') else 'No'
                f.write(f"{case['nuclide']:<12} {case['mt']:<6} {case['reaction']:<14} "
                        f"{case['single_target']:<16} {case.get('original_target', '-'):<16} "
                        f"{redundant:<10} {case['action'].upper():<16} {case['reason']}\n")

                # Show missing products if any
                if case.get('missing_products'):
                    for mp in case['missing_products']:
                        f.write(f"             Filtered out: {mp}\n")

            f.write("\n")

        # MAPPING CONFLICTS section (expanded tabular format)
        f.write("\n\n" + "=" * 220 + "\n")
        f.write("MAPPING CONFLICTS\n")
        f.write("=" * 220 + "\n\n")

        f.write("N.B.:\n")
        f.write("ONE-TO-MANY and MANY-TO-ONE Mapping not necessarily bad\n")
        f.write("LFS is Arbitrary Per-Reaction\n\n")
        f.write("  LFS (Level number of final excited state) in GENDF MF=10 is just an index assigned\n")
        f.write("  within each reaction's data section. It's not globally consistent across the library.\n\n")
        f.write("  Example - Hf178_m2 production:\n\n")
        f.write("  | Parent | Reaction | LFS | ELFS (eV) | Physical State |\n")
        f.write("  |--------|----------|-----|-----------|----------------|\n")
        f.write("  | Hf177  | (n,g)    | 10  | 2,446,090 | Hf178_m2       |\n")
        f.write("  | Lu179  | (n,p)    | 99  | 2,446,090 | Hf178_m2       |\n")
        f.write("  | Ta181  | (n,a)    | 3   | 2,446,090 | Hf178_m2       |\n\n")
        f.write("  All three reactions produce the same physical state (Hf178_m2 at 2.446 MeV excitation),\n")
        f.write("  but they use different LFS values (10, 99, 3).\n\n")

        conflicts = _detect_mapping_conflicts(isomer_mappings)
        _write_conflicts_table(f, conflicts, isomer_mappings, header, sep)

        # MF=10 CONSISTENCY AUDIT + REJECTED sections (group-space band audit)
        audit_emax = stats.get('audit_emax', 2.0e7) if stats else 2.0e7
        reject_band_ratio = (stats.get('mf10_reject_band_ratio')
                             if stats else None)
        audit_clean = stats.get('audit_clean', 0) if stats else 0
        offenders = [r for r in (audit_rows or [])
                     if r.get('worst_dev', 0.0) > CONSISTENCY_RTOL]
        _write_consistency_audit_section(f, offenders, audit_clean, audit_emax)
        _write_rejected_section(f, rejected_rows or [], reject_band_ratio)

        # HYBRID MAPPING section (elis_lfs_order only): the phase detail behind
        # every mapped row above.
        if hybrid:
            _write_hybrid_mapping_section(f, hybrid_records, isomer_mappings)

        # ORPHAN sections. The disposition table is mode-agnostic (any mode can
        # meet a product the chain does not carry); the added-nuclide roll call
        # only exists under add-stable.
        _write_orphan_disposition_section(f, orphan_levels,
                                          orphan_nuclides_added,
                                          reattributions, renormalizations,
                                          orphan_policy)
        if orphan_policy == 'add-stable':
            _write_orphan_nuclides_section(f, orphan_nuclides_added)

        # LFS PLACEHOLDER VALUES section (report-only safeguard)
        _write_lfs_placeholder_section(f, lfs_placeholders)

        # ANONYMOUS (IZAP=0) MF=10 classification (R1-61)
        _write_attribution_section(f, attribution_records or [],
                                   Counter(attribution_counts or {}),
                                   reattribute_mf10_noizap)

        # MF=10 LFS=0 ground-head Q consistency census (Q-55, report-only)
        _write_q55_ground_head_section(f, q55_rows, q55_counts)

    print(f"Isomer mapping log written to: {log_file}")


def _write_mapping_row(f, m):
    """Write a single mapping row."""
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

    # Handle error entries - use base_nuclide if product not set
    err_type = m.get('type')
    if err_type == 'elis_tol_exceeded':
        method = 'not-mapped'
        notes = 'ELIS rtol exceeded. Closest shown.'
        # Use base_nuclide from error dict if product is '?'
        if product == '?':
            product = m.get('base_nuclide', '?')
    elif err_type == 'no_metastable_decay_data':
        method = 'not-mapped'
        notes = 'Product not in DK-Lib'
        # Use base_nuclide from error dict if product is '?'
        if product == '?':
            product = m.get('base_nuclide', '?')
    elif err_type == 'zero_elis_metastables':
        method = 'not-mapped'
        # Format skipped states info
        skipped = m.get('skipped_states', [])
        if skipped:
            skipped_str = ', '.join(f"_m{liso}" for liso, _, _ in skipped)
            notes = f'DK-Lib ELIS=0 ({skipped_str})'
        else:
            notes = 'DK-Lib has ELIS=0 (data quality)'
        if product == '?':
            product = m.get('base_nuclide', '?')
    elif err_type == 'elis_incomputable':
        method = 'not-mapped'
        notes = 'No QM: ELFS incomputable'
        if product == '?':
            product = m.get('base_nuclide', '?')
    elif err_type == 'duplicate_mapping':
        # Duplicate mapping - this LFS was discarded because another LFS was closer
        method = 'not-mapped'
        # Notes already set when adding to by_parent

    mt_str = str(mt) if mt is not None else "?"
    lfs_str = str(lfs) if lfs is not None else "?"
    base_nuc = product.split('_')[0] if '_' in str(product) else str(product)
    gendf_product = f"{base_nuc}_m{lfs}" if lfs is not None else base_nuc
    liso_str = str(liso) if liso is not None else "-"

    # For "closest shown" entries, show actual product; otherwise OMITTED for not-mapped
    if method == 'not-mapped' and 'Closest' in notes:
        chain_product = f"{base_nuc}_m{liso}" if liso is not None else base_nuc
    elif method == 'not-mapped' and err_type == 'no_metastable_decay_data':
        chain_product = '-'  # No product exists in decay library at all
    elif method == 'not-mapped' and err_type == 'zero_elis_metastables':
        chain_product = '-'  # Product has ELIS=0 in decay library (data quality issue)
    elif method == 'not-mapped' and err_type == 'elis_incomputable':
        chain_product = '-'  # No ELFS to match on: the level was never placed
    elif method == 'not-mapped':
        chain_product = 'OMITTED'
    else:
        chain_product = product

    if isinstance(half_life, (int, float)):
        half_life_str = f"{half_life:.3e}"
    else:
        half_life_str = str(half_life)

    elis_str = f"{elis:.1f}" if elis is not None else "N/A"

    # For missing_metastable or zero_elis, show '-' or '0.0' to indicate data issues
    if err_type == 'no_metastable_decay_data':
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

    # Display method name appropriately
    method_display = {
        'elis': 'ELIS',
        'lfs_order': 'LFS_ORDER',
        'lfs_order_fallback': 'LFS-ORD(fb)',   # hybrid Phase 2
        'placeholder_bound': 'PLACEHOLDER',    # unidentified excited state
        'orphan_added': 'ORPHAN+',             # no decay partner at all
    }.get(method, method)

    f.write(f"{mt_str:>5}  {reaction:<12}  {lfs_str:>9}  {gendf_product:<15}  "
            f"{liso_str:>7}  {chain_product:<15}  {half_life_str:>12}  "
            f"{elis_str:>14}  {dk_elis_str:>14}  {disc_str:>22}    {method_display:<12}    {notes:<30}\n")


def _detect_mapping_conflicts(isomer_mappings):
    """Detect LFS->LISO mapping conflicts."""
    valid = [m for m in isomer_mappings if m.get('liso') is not None and m.get('lfs') is not None]

    lfs_to_liso = defaultdict(list)
    for m in valid:
        base = m.get('product', '').split('_')[0]
        lfs_to_liso[(base, m['lfs'])].append(m)

    one_to_many = []
    for (base, lfs), mappings in lfs_to_liso.items():
        lisos = set(m['liso'] for m in mappings)
        if len(lisos) > 1:
            one_to_many.append({'base': base, 'lfs': lfs, 'mappings': mappings})

    liso_to_lfs = defaultdict(list)
    for m in valid:
        base = m.get('product', '').split('_')[0]
        liso_to_lfs[(base, m['liso'])].append(m)

    many_to_one = []
    for (base, liso), mappings in liso_to_lfs.items():
        lfs_vals = set(m['lfs'] for m in mappings)
        if len(lfs_vals) > 1:
            many_to_one.append({'base': base, 'liso': liso, 'mappings': mappings})

    return {'one_to_many': one_to_many, 'many_to_one': many_to_one}


def _write_conflicts_table(f, conflicts, all_mappings, header, sep):
    """Write mapping conflicts in tabular format."""
    one_to_many = conflicts['one_to_many']
    many_to_one = conflicts['many_to_one']

    if not one_to_many and not many_to_one:
        f.write("No mapping conflicts detected.\n")
        return

    if one_to_many:
        f.write("ONE-TO-MANY (Same LFS → Different LISO) - CRITICAL:\n")
        f.write("Same GENDF LFS maps to different decay library LISO values.\n\n")

        for conflict in one_to_many:
            base, lfs = conflict['base'], conflict['lfs']
            mappings = conflict['mappings']
            lisos = sorted(set(m['liso'] for m in mappings))
            f.write(f"  {base} LFS={lfs} → LISO values: {lisos}\n")
            f.write(f"  {sep}\n")
            f.write(f"  {'Parent':<12}  {header}\n")
            f.write(f"  {sep}\n")
            for m in mappings:
                f.write(f"  {m.get('parent', '?'):<12}  ")
                _write_mapping_row(f, m)
            f.write("\n")

    if many_to_one:
        f.write("\nMANY-TO-ONE (Different LFS → Same LISO):\n")
        f.write("Multiple GENDF LFS values map to same decay library LISO.\n\n")

        for conflict in many_to_one:
            base, liso = conflict['base'], conflict['liso']
            mappings = conflict['mappings']
            lfs_vals = sorted(set(m['lfs'] for m in mappings))
            f.write(f"  {base} LISO={liso} ← LFS values: {lfs_vals}\n")
            f.write(f"  {sep}\n")
            f.write(f"  {'Parent':<12}  {header}\n")
            f.write(f"  {sep}\n")
            for m in mappings:
                f.write(f"  {m.get('parent', '?'):<12}  ")
                _write_mapping_row(f, m)
            f.write("\n")


def write_renormalization_log(renormalizations, log_file):
    """Write renormalization log."""
    with open(log_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RENORMALIZATION LOG\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total renormalized reactions: {len(renormalizations)}\n\n")

        for r in renormalizations:
            f.write(f"{r['nuclide']} {r['reaction']} (MT={r['mt']}):\n")
            f.write(f"  Valid: {r['valid_products']}\n")
            f.write(f"  Dropped: {r['dropped_products']}\n\n")

    print(f"Renormalization log written to: {log_file}")


# =============================================================================
# Main workflow
# =============================================================================

def main(endf_gxs_dir, base_chain_file, output_chain_file,
         decay_file,
         mapping_mode='elis_lfs_order',
         orphan_policy='add-stable',
         elis_rtol=None, elis_atol=0.0,
         skip_zero_elis_metastables=True,
         mt_list=None, verbose=True,
         isomer_mapping_log_file=None,
         renormalization_log_file=None,
         prune_nn_prime_self_loops=False,
         suppress_single_target_yields=False,
         emit_mf10_only_reactions=False,
         reattribute_mf10_noizap=False,
         mode='flags_only',
         audit_emax=2.0e7,
         mf10_reject_band_ratio=None):
    """
    Main workflow: GENDF MF=10 → OpenMC chain with isomeric branching.

    Parameters
    ----------
    endf_gxs_dir : str
        Directory containing GENDF files
    base_chain_file : str
        Base OpenMC chain XML file
    output_chain_file : str
        Output chain XML file
    decay_file : str
        ENDF decay library for ELIS mapping (REQUIRED for all modes)
    mapping_mode : str
        Isomeric state mapping mode:
        - 'elis_lfs_order' (default): hybrid - ELIS matching first, positional
          fallback for the levels left over, orphan states kept
        - 'elis': ELIS-based matching only (unmatched levels dropped)
        - 'lfs_order': FISPACT-like positional mapping (validation)
    orphan_policy : str
        Disposition of a product with no partner in the chain: 'add-stable'
        (default, add it to the chain as a stable pure sink and keep the
        branch), 'renorm' (drop it, redistribute pro-rata) or 'reattribute'
        (fold its share into the kept isomer at the nearest lower rank).
    elis_rtol : float or None
        Relative tolerance for ELIS matching. None (default) takes the
        mode-dependent default: 0.15 for elis_lfs_order, 0.50 for the other
        two. In ELIS mode products beyond rtol are skipped; in the hybrid they
        fall through to the positional phase; in LFS-order mode the tolerance
        drives ELIS reference warnings only.
    elis_atol : float
        Absolute tolerance in eV (default 0.0)
    reattribute_mf10_noizap : bool
        Recover MF=10 subsections written with IZAP=0 when the C1-C4 evidence
        gate passes. Default False; OFF (or a gate failure) prunes the whole
        reaction's isomeric decoration.
    """
    # Validate required parameters
    if decay_file is None:
        raise ValueError(
            "decay_file is required for isomeric branching (both modes)."
        )
    if mapping_mode not in MAPPING_MODES:
        raise ValueError(
            f"Invalid mapping_mode '{mapping_mode}'. Must be one of "
            f"{', '.join(MAPPING_MODES)}."
        )
    if orphan_policy not in ORPHAN_POLICIES:
        raise ValueError(
            f"Invalid orphan_policy '{orphan_policy}'. Must be one of "
            f"{', '.join(ORPHAN_POLICIES)}."
        )
    # Mode-dependent tolerance default, resolved before the library is built so
    # the mapper, the log header and the proximity check all see one value.
    elis_rtol = resolve_elis_rtol(mapping_mode, elis_rtol)

    # Create output directories up front (first-run libraries have no GENDF/ dir)
    for out_file in (output_chain_file, isomer_mapping_log_file,
                     renormalization_log_file):
        if out_file:
            out_dir = os.path.dirname(out_file)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("GENDF Isomeric Branching Chain Patcher v12")
    print("=" * 60)
    print(f"\nMAPPING MODE: {mapping_mode.upper()}")
    if mapping_mode == 'lfs_order':
        print("  (FISPACT-like positional mapping for validation)")
        print("  WARNING: May produce incorrect results for nuclides where")
        print("           LFS order ≠ LISO order (e.g., Ag116)")
    elif mapping_mode == 'elis_lfs_order':
        print("  (hybrid: excitation-energy matching first, then position)")
        print("    Phase 1: levels whose ELFS matches exactly one decay state")
        print("             within tolerance are bound to it; an ambiguous or")
        print("             unusable energy abstains instead of guessing")
        print("    Phase 2: the levels left over are paired, in level order,")
        print("             with the decay metastables left over (no energy")
        print("             preference, states already claimed are skipped)")
        print("    Phase 3: a level with no decay state left becomes an ORPHAN")
        print(f"             state -- disposition: --orphan-policy {orphan_policy}")
    else:
        print("  (ELIS-based matching only - unmatched levels dropped)")
    print(f"ORPHAN POLICY: {orphan_policy}")

    # Step 1: Load chain
    print("\nStep 1: Loading base chain...")
    chain = Chain.from_xml(base_chain_file)
    print(f"  Loaded {len(chain.nuclides)} nuclides")

    # Step 2: Detect energy structure
    # Step 2-3: Load GENDF library (energy structure auto-detected)
    print("\nStep 2: Loading GENDF library...")
    lib_kwargs = {
        'validate_energy_grid': False,
        'skip_zero_elis_metastables': skip_zero_elis_metastables,
        'decay_file': decay_file,
        'elis_rtol': elis_rtol,
        'elis_atol': elis_atol,
        'mapping_mode': mapping_mode,
    }
    print(f"  Mapping mode: {mapping_mode}")
    print(f"  ELIS tolerance: rtol={elis_rtol} ({elis_rtol*100:.0f}%), atol={elis_atol} eV")

    lib = GENDFLibrary(endf_gxs_dir, **lib_kwargs)
    print(f"  Energy structure: {lib.energy_structure}")
    print(f"  Available nuclides: {len(lib.available_nuclides())}")

    # Step 3a: anonymous (IZAP=0) MF=10 classification. Runs BEFORE the emission
    # pre-pass and the branching extraction so a recovered IZAP -- written into
    # the cached parsed section -- is visible to product naming, the emission
    # pass, the branching extraction and the consistency audit alike.
    print("\nStep 3a: Classifying anonymous (IZAP=0) MF=10 subsections...")
    if reattribute_mf10_noizap:
        print("  Re-attribution ENABLED (evidence gate C1-C4, Q tolerance "
              f"{REATTRIB_Q_TOL_GROUND_EV:.1f} eV ground / "
              f"{REATTRIB_Q_TOL_EV/1e3:.1f} keV isomer ELIS)")
    else:
        print("  Re-attribution OFF: affected reactions lose ALL isomeric "
              "decoration (--reattribute-mf10-noIZAP)")
    attrib_records, attrib_pruned, attrib_counts = scan_mf10_attribution(
        lib, chain, reattribute=reattribute_mf10_noizap)
    _print_attribution_summary(attrib_records, attrib_counts,
                               reattribute_mf10_noizap)
    attrib_anonymous = unrepaired_anonymous_mts(attrib_records)

    # Q-55 census (report-only): materials are cached by the scan above, so
    # this second walk costs only dictionary lookups.
    q55_rows, q55_counts = census_lfs0_ground_heads(lib, chain)
    if q55_rows:
        print(f"  WARNING: {len(q55_rows)} of {q55_counts['lfs0_heads']} MF=10 "
              f"LFS=0 ground heads violate QM == QI (Q-55) -- see mapping log")
    else:
        print(f"  MF=10 LFS=0 ground heads: all {q55_counts['lfs0_heads']} "
              f"consistent (|QM - QI| <= {REATTRIB_Q_TOL_GROUND_EV:.1f} eV)")

    # MF=10-only channel emission (opt-in). EAF-2010 stores isomer-daughter
    # reactions only in MF=8/10 (no MF=3), so Chain.from_endf never harvested
    # them. The ENTIRE pre-pass is gated on this flag, so a flag-off run is
    # byte-identical to the unmodified tool.
    writer_base = base_chain_file
    emit_summary = None
    if emit_mf10_only_reactions:
        print("\nStep 3b: Emitting GENDF MF=10-only reactions "
              "(EAF isomer-daughter channels)...")
        writer_base = f"{output_chain_file}.mf10_emitted_base.xml"
        emit_summary = emit_mf10_only_prepass(
            lib, chain, base_chain_file, writer_base, verbose=verbose,
            unrepaired_anonymous=attrib_anonymous)
        # D4: reload the Chain from the emitted XML so Gate 1 (chain object) and
        # the XML writer (tree) both see the new reactions (single source).
        chain = Chain.from_xml(writer_base)

    print("\nStep 4: Extracting MF=10 branching data...")
    # Pruned reactions are excluded from extraction outright: their named levels
    # are only a subset of the real final states, so building branching from
    # them would decorate that subset alone and invert the branching (R1-61).
    branching_data = lib.process_library_for_branching(
        mt_list=mt_list, verbose=verbose, chain=chain,
        skip_reactions=attrib_pruned
    )

    # All-or-nothing prune (R1-61): a reaction with an unrecovered anonymous
    # subsection keeps its plain MF=3 route and loses its isomeric decoration
    # entirely -- decorating the attributed subset alone would invert the
    # branching. Belt-and-braces after the extraction skip, and the point where
    # every downstream count, the mapping log and the output XML see the
    # reduced set (as the band-reject does).
    if attrib_pruned:
        branching_data = prune_unattributed_decoration(branching_data,
                                                       attrib_pruned)
        print(f"  Pruned {len(attrib_pruned)} reaction(s) with unrecovered "
              "anonymous (IZAP=0) MF=10 subsections (excluded from extraction)")

    # Capture errors. A hybrid record describes HOW a level was mapped, not a
    # loss: a level routed to the positional fallback (and a duplicate loser
    # requeued with it) is still mapped, so it must never reach the not-mapped
    # tables or trigger a renormalization note.
    elis_errors = []
    duplicate_mapping_errors = []
    lfs_order_dropped = []
    lfs_order_orphan_dk = []
    hybrid_records = []
    for err in lib.processing_errors:
        err_type = err.get('type')
        if err_type in HYBRID_ERROR_TYPES:
            hybrid_records.append(err)
        elif err.get('routed_to_fallback') or err.get('requeued'):
            hybrid_records.append(err)
        elif err_type in ('elis_tol_exceeded', 'no_metastable_decay_data',
                          'zero_elis_metastables', 'elis_incomputable'):
            elis_errors.append(err)
        elif err_type == 'duplicate_mapping':
            duplicate_mapping_errors.append(err)
        elif err_type == 'lfs_order_dropped':
            lfs_order_dropped.append(err)
        elif err_type == 'lfs_order_orphan_dk':
            lfs_order_orphan_dk.append(err)
    orphan_levels = lib.orphan_levels if hasattr(lib, 'orphan_levels') else []

    # Step 4b: MF=10-vs-MF=3 consistency audit (group-space). Always builds the
    # audit table; --mf10-reject-band-ratio (OFF by default) additionally leaves
    # a band-inconsistent reaction stock. Rejection is applied by dropping the
    # reaction from branching_data BEFORE decoration, so every downstream count,
    # the mapping log, and the output XML all see the reduced (post-reject) set.
    print("\nStep 4b: MF=10-vs-MF=3 consistency audit "
          f"(cap E <= {audit_emax:.3e} eV)...")
    if mf10_reject_band_ratio is None:
        print("  Audit: detection + logging only (no rejection)")
    else:
        print(f"  Audit rejection: band ratio - 1 > "
              f"{mf10_reject_band_ratio:.3e} (over-summing only) -> reaction "
              "left stock (self-loop-ground exempt)")
    (audit_rows, audit_offenders, audit_clean, rejected_rows,
     band_reject_exempt) = run_mf10_consistency_audit(
        lib, branching_data, emax=audit_emax,
        reject_band_ratio=mf10_reject_band_ratio)
    for row in rejected_rows:
        branching_data.get(row['parent'], {}).pop(row['reaction'], None)
    branching_data = {p: rxns for p, rxns in branching_data.items() if rxns}
    print(f"  Audited: {len(audit_rows)} reaction(s); "
          f"offenders (dev > {CONSISTENCY_RTOL:.0e}): {audit_offenders}; "
          f"rejected: {len(rejected_rows)}; band-reject exempt: "
          f"{band_reject_exempt}")

    # Policy 3(a) repairs: a radioactive-products-only MF=10 (metastable levels,
    # no LFS=0) gets its ground channel synthesized from the MF=3 total. Listed
    # per reaction -- it is a data-shape decision worth eyeballing. Snapshotted
    # HERE, after the R1-61 prune and the Step-4b rejection; the writer can drop
    # a decoration too (chain-membership filter, single-target suppression), so
    # this list is filtered AGAIN against what add_branching_to_xml actually
    # wrote before anything reports it.
    ground_repaired = [r for r in lib.ground_repaired
                       if r['reaction'] in branching_data.get(r['nuclide'], {})]
    ground_absent_skipped = [u for u in lib.unmatched_mts
                             if u['reason'] in GROUND_ABSENT_SKIP_REASONS]
    if ground_absent_skipped:
        # "not decorated", not "unrepairable": the set also holds reactions whose
        # ground WAS synthesized and then lost every metastable at mapping.
        print(f"  Ground absent, not decorated: {len(ground_absent_skipped)}")
        for rec in ground_absent_skipped:
            label = GROUND_ABSENT_SKIP_LABELS.get(rec['reason'], rec['reason'])
            print(f"    {rec['nuclide']} {rec['reaction']} (MT={rec['mt']}): "
                  f"{label}")

    # Calculate counts for summary
    # branching_data structure: {nuclide: {reaction_name: IsomericBranching}}
    mapped_count = sum(
        sum(1 for p in branching.products if '_m' in p)
        for nuclide_reactions in branching_data.values()
        for branching in nuclide_reactions.values()
    )
    rtol_exceeded = sum(1 for e in elis_errors if e.get('type') == 'elis_tol_exceeded')
    no_dk_data = sum(1 for e in elis_errors if e.get('type') == 'no_metastable_decay_data')
    zero_elis = sum(1 for e in elis_errors if e.get('type') == 'zero_elis_metastables')
    incomputable = sum(1 for e in elis_errors if e.get('type') == 'elis_incomputable')
    dup_mappings = len(duplicate_mapping_errors)
    # Count discarded LFS values (each duplicate_mapping error has 'discarded' list)
    dup_discarded = sum(len(e.get('discarded', [])) for e in duplicate_mapping_errors)
    dropped_count = len(lfs_order_dropped)
    orphan_count = len(lfs_order_orphan_dk)
    hybrid_fallback_count = sum(1 for r in hybrid_records
                                if r.get('type') == 'hybrid_fallback')
    placeholder_unmapped = sum(1 for r in hybrid_records
                               if r.get('type') == 'placeholder_unmapped')

    if mapping_mode == 'elis':
        total_mappings = mapped_count + rtol_exceeded + no_dk_data + incomputable
        total_gendf_lfs = total_mappings + dup_discarded
    elif mapping_mode == 'elis_lfs_order':
        # Nothing is dropped except a placeholder with no state left to bind:
        # every other level is written (Phase 1, Phase 2, placeholder-bound or
        # orphan), so mapped_count already carries it.
        total_mappings = mapped_count + placeholder_unmapped
        total_gendf_lfs = total_mappings
    else:
        total_mappings = mapped_count + dropped_count
        total_gendf_lfs = total_mappings

    print(f"\n                      nuclides in GENDF library: {len(lib.available_nuclides()):5d}")
    print(f"               nuclides in GENDF with branching: {len(branching_data):5d}")
    print("-" * 52)
    print(f"                          Total GENDF-LFS found: {total_gendf_lfs:5d}")
    if mapping_mode == 'elis':
        print(f"                                   ELIS matched: {mapped_count:5d}")
    elif mapping_mode == 'elis_lfs_order':
        print(f"                       Levels mapped (all phases): {mapped_count:5d}")
        print(f"                Phase-2 positional fallback pool: {hybrid_fallback_count:5d}")
        print(f"                          Phase-3 orphan levels: {len(orphan_levels):5d}")
        if placeholder_unmapped:
            print(f"          Placeholder LFS unmapped (report-only): {placeholder_unmapped:5d}")
    else:
        print(f"                              LFS-order mapped: {mapped_count:5d}")
    if incomputable:
        print(f"                    ELIS incomputable (no QM): {incomputable:5d}")
    if rtol_exceeded:
        print(f"                             ELIS rtol exceeded: {rtol_exceeded:5d}")
    if no_dk_data:
        print(f"                           No product in DK-Lib: {no_dk_data:5d}")
    if dropped_count:
        print(f"                 LFS dropped (exceeds DK count): {dropped_count:5d}")
    if orphan_count:
        print(f"                     Orphan DK states (no LFS): {orphan_count:5d}")
    if dup_mappings:
        print(f"                    Duplicate mappings resolved: {dup_mappings:5d} ({dup_discarded} LFS discarded)")
    print(f"                          MF=10 audit offenders: {audit_offenders:5d}")
    print(f"                                 MF=10 rejected: {len(rejected_rows):5d}")
    print(f"          Band-reject exempt (self-loop ground): {band_reject_exempt:5d}")

    # Report-only LFS placeholder scan (see PLACEHOLDER_LFS): flag products/partials
    # whose GENDF MF=10 LFS is a library "unspecified level" placeholder (99 / 40)
    # so it is never silently consumed as an isomer ordinal. The flags-only
    # writer would otherwise emit the placeholder verbatim in gendf_lfs. This
    # changes no mapping decision and no byte of the output chain XML.
    lfs_placeholders = _collect_lfs_placeholders(
        branching_data, elis_errors, duplicate_mapping_errors,
        lfs_order_dropped, mapping_mode)
    if lfs_placeholders:
        print(f"                    LFS placeholder occurrences: {len(lfs_placeholders):5d}")
    _print_lfs_placeholder_warning(lfs_placeholders, mapping_mode)

    # Step 5: Add to XML
    print("\nStep 5: Adding branching to chain XML...")
    if prune_nn_prime_self_loops:
        print("  Pruning (n,n') self-loops without isomeric branching...")
    summary = add_branching_to_xml(
        original_xml_file=writer_base,
        branching_data=branching_data,
        output_xml_file=output_chain_file,
        chain=chain,
        verbose=verbose,
        prune_nn_prime_self_loops=prune_nn_prime_self_loops,
        suppress_single_target_yields=suppress_single_target_yields,
        mode=mode,
        mf10_emit_summary=emit_summary,
        orphan_policy=orphan_policy,
        repaired_keys={(r['nuclide'], r['reaction']): r['ground_product']
                       for r in ground_repaired}
    )

    # Orphan disposition: what the writer did with the products that had no
    # partner in the chain (see the ORPHAN sections of the mapping log).
    if summary['orphan_nuclides_added']:
        n_sources = sum(len(v) for v in summary['orphan_nuclides_added'].values())
        print(f"\n  Orphan nuclides ADDED to the chain: "
              f"{len(summary['orphan_nuclides_added'])} "
              f"(from {n_sources} reaction pathway(s)); each is stable with no "
              "decay data -- it conserves the branch but models no activity")
        for name in sorted(summary['orphan_nuclides_added']):
            sources = summary['orphan_nuclides_added'][name]
            parents = ', '.join(sorted({s['parent'] for s in sources}))
            print(f"    {name:<12} <- {parents}")
    if summary['reattributions']:
        folded = [r for r in summary['reattributions'] if r['recipient']]
        stranded = len(summary['reattributions']) - len(folded)
        print(f"  Branches reattributed (folded into a kept isomer): "
              f"{len(folded)}")
        if stranded:
            print(f"  WARNING: {stranded} orphan branch(es) had no recipient at "
                  "all and were renormalized instead; see ORPHAN DISPOSITION "
                  "in the mapping log")
    if summary['embedded_zero_rows']:
        print(f"  Embedded mode: {summary['embedded_zero_rows']} energy row(s) "
              "skipped -- every surviving target was zero there")

    # Only decorations the writer actually shipped count as repairs.
    ground_repaired = [r for r in ground_repaired
                       if (r['nuclide'], r['reaction']) in summary['added_keys']]
    if ground_repaired:
        print(f"\n  Ground synthesized from MF=3 remainder: {len(ground_repaired)}")
        for rec in ground_repaired:
            clamped = rec.get('clamped_points', 0)
            clamp = (f" [clamped {clamped}/{rec.get('total_points', 0)} pts]"
                     if clamped else "")
            print(f"    {rec['nuclide']} {rec['reaction']} (MT={rec['mt']}): "
                  f"{rec['ground_product']} = max(0, MF3 - sum MF10 LFS"
                  f"{rec['metastable_lfs']}){clamp}")
    # A repair the writer threw away is reported with its identity, here and in
    # the mapping log -- a bare counter cannot be chased back to a reaction.
    ground_repaired_dropped = [
        d for d in summary['skipped_details']
        if d.get('reason') == 'repaired_ground_only_survivor']
    if summary['repaired_dropped_no_metastable']:
        print("  Repaired reactions dropped (no metastable in the chain): "
              f"{summary['repaired_dropped_no_metastable']}")
        for rec in ground_repaired_dropped:
            print(f"    {rec['nuclide']} {rec['reaction']} (MT={rec['mt']}): "
                  f"synthesized ground {rec['ground_product']} was the only "
                  "product left in the chain")

    # Unified emission summary now that both passes (ground pre + metastable-direct
    # post, run inside add_branching_to_xml) have completed.
    if emit_summary is not None and verbose:
        _print_emission_summary(emit_summary, output_chain_file)

    # Drop the emission pre-pass intermediate now that the writer has read it.
    if emit_mf10_only_reactions and writer_base != base_chain_file:
        try:
            os.remove(writer_base)
        except OSError:
            pass

    if summary['renormalizations']:
        print(f"\nRenormalized: {len(summary['renormalizations'])}")

    if summary['nn_prime_self_loops_pruned']:
        print(f"Pruned (n,n') self-loops: {len(summary['nn_prime_self_loops_pruned'])}")

    if summary['pathway_q_corrections']:
        print(f"Pathway-Q QM/QI-sourced: {len(summary['pathway_q_corrections'])} "
              "reactions differ from the legacy fold ((n,n') rows = the MT=4 "
              "double-subtraction fix; all other rows = ground-slot/metastable "
              "re-sourcing where the file's QM/QI differs from the chain "
              "scalar, adopted only where the file's ground QM corroborates "
              "that scalar); see PATHWAY-Q section of the mapping log")

    if summary['pathway_q_rejected']:
        print(f"WARNING: {len(summary['pathway_q_rejected'])} reaction(s) had a "
              f"file ground QM more than {_q_str(PATHWAY_Q_CHAIN_TOL)} eV from "
              "the chain's scalar Q; their file values were REFUSED and the "
              "chain-anchored legacy values kept "
              f"({summary['q_chain_anchored']} slot(s) incl. ground); see "
              "PATHWAY-Q FILE-QM REJECTED in the mapping log")

    if summary['q_zero_anchor']:
        print(f"Pathway-Q gate skipped: {summary['q_zero_anchor']} reaction(s) "
              "carry a chain Q of exactly 0 (missing-Q placeholder, not an "
              "anchor); their file QM/QI values were kept")

    if summary['q_replicated']:
        print(f"WARNING: {summary['q_replicated']} pathway-Q slot(s) had no "
              "MF=10 Q data and replicated the chain's scalar Q "
              f"({summary['q_from_mf10']} slots came from MF=10 QM/QI"
              + (f", {summary['q_from_elis']} from ELIS arithmetic"
                 if summary['q_from_elis'] else "") + ")")

    if summary['pathway_q_qm_disagreements']:
        print(f"Pathway-Q QM disagreement: "
              f"{len(summary['pathway_q_qm_disagreements'])} section(s) whose "
              "MF=10 subsections carry different QM values; see PATHWAY-Q "
              "INTRA-SECTION QM DISAGREEMENT in the mapping log")

    # Single-target summary (always printed if any exist)
    if summary['single_target_cases']:
        cases = summary['single_target_cases']
        suppressed = summary.get('single_target_suppressed', 0)
        kept = len(cases) - suppressed

        print(f"\n  Single-target isomeric yields: {len(cases)}")
        print(f"    Suppressed (redundant): {suppressed}")
        print(f"    Kept: {kept}")

        # Breakdown by reason
        reason_counts = Counter(c['reason'] for c in cases)
        print(f"\n    By reason:")
        for reason, count in reason_counts.most_common():
            print(f"      {reason}: {count} ({100*count/len(cases):.1f}%)")

    # Step 6: Write logs
    if renormalization_log_file and summary['renormalizations']:
        write_renormalization_log(summary['renormalizations'], renormalization_log_file)

    if isomer_mapping_log_file:
        stats = {
            'base_chain_file': base_chain_file,
            'gendf_dir': endf_gxs_dir,
            'decay_file': decay_file,
            'output_chain_file': output_chain_file,
            'mapping_mode': mapping_mode,
            'orphan_policy': orphan_policy,
            'renormalizations': summary['renormalizations'],
            'elis_rtol': elis_rtol,
            'elis_atol': elis_atol,
            'gendf_nuclides_total': len(lib.available_nuclides()),
            'chain_nuclides_total': len(chain.nuclides),
            'audit_emax': audit_emax,
            'mf10_reject_band_ratio': mf10_reject_band_ratio,
            'audit_clean': audit_clean,
            'audit_offenders': audit_offenders,
            'mf10_rejected': len(rejected_rows),
            'band_reject_exempt': band_reject_exempt,
            'ground_repaired': ground_repaired,
            'ground_absent_skipped': ground_absent_skipped,
            'ground_repaired_dropped': ground_repaired_dropped,
            'pathway_q_corrections': summary['pathway_q_corrections'],
            'pathway_q_qm_disagreements': summary['pathway_q_qm_disagreements'],
            'pathway_q_rejected': summary['pathway_q_rejected'],
            'q_from_mf10': summary['q_from_mf10'],
            'q_from_elis': summary['q_from_elis'],
            'q_replicated': summary['q_replicated'],
            'q_chain_anchored': summary['q_chain_anchored'],
            'q_zero_anchor': summary['q_zero_anchor'],
        }
        write_isomer_mapping_log(summary['elis_mappings'], isomer_mapping_log_file,
                                stats=stats, elis_errors=elis_errors,
                                duplicate_mapping_errors=duplicate_mapping_errors,
                                lfs_order_dropped=lfs_order_dropped,
                                lfs_order_orphan_dk=lfs_order_orphan_dk,
                                single_target_cases=summary.get('single_target_cases', []),
                                lfs_placeholders=lfs_placeholders,
                                audit_rows=audit_rows, rejected_rows=rejected_rows,
                                attribution_records=attrib_records,
                                attribution_counts=attrib_counts,
                                reattribute_mf10_noizap=reattribute_mf10_noizap,
                                hybrid_records=hybrid_records,
                                orphan_levels=orphan_levels,
                                orphan_nuclides_added=summary['orphan_nuclides_added'],
                                reattributions=summary['reattributions'],
                                orphan_policy=orphan_policy,
                                reaction_type_missing=summary['reaction_type_missing'],
                                q55_rows=q55_rows, q55_counts=q55_counts)

    return chain


# =============================================================================
# CLI execution
# =============================================================================

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    # Get library configuration
    config = LIBRARY_CONFIGS[args.library]

    # Verbose output is the default; --quiet is the opt-out
    verbose = not args.quiet

    # Build output filenames with mapping suffix. --output-dir / --log-dir
    # override the pairing's directory so a verification run never writes over
    # a production chain or log.
    suffix = f".{args.map}_mapped"
    out_dir = args.output_dir if args.output_dir is not None else config['output_dir']
    log_dir = args.log_dir if args.log_dir is not None else config['output_dir']
    output_chain = os.path.join(out_dir, f"{config['output_prefix']}{suffix}.xml")
    log_file = os.path.join(log_dir, f"{config['log_prefix']}{suffix}.txt")

    # An explicit -r wins; otherwise the mode decides (0.15 hybrid / 0.50 legacy)
    elis_rtol = resolve_elis_rtol(args.map, args.rtol)

    print("=" * 70)
    print("GENDF Isomeric Branching Chain Patcher v12")
    print("=" * 70)
    print(f"\nLibrary:      {args.library} - {config['description']}")
    print(f"Mapping mode: {args.map}")
    print(f"Orphan policy: {args.orphan_policy}")
    print(f"Tolerances:   rtol={elis_rtol}"
          f"{'' if args.rtol is not None else ' (mode default)'}, "
          f"atol={args.atol}")
    if args.prune_nn_prime_self_loops:
        print("Prune (n,n') self-loops: ENABLED")
    if args.suppress_single_target_yields:
        print("Suppress single-target yields: ENABLED")
    if args.emit_mf10_only_reactions:
        print("Emit MF=10-only reactions: ENABLED")
    if args.reattribute_mf10_noIZAP:
        print("Re-attribute MF=10 IZAP=0 subsections: ENABLED")
    print(f"Output mode:  {args.mode}")
    print(f"Audit emax:   {args.audit_emax:.3e} eV")
    if args.mf10_reject_band_ratio is None:
        print("MF=10 reject: off (audit only)")
    else:
        print(f"MF=10 reject: band ratio - 1 > {args.mf10_reject_band_ratio}")
    print(f"\nInput chain:  {config['base_chain']}")
    print(f"Output chain: {output_chain}")
    print(f"Mapping log:  {log_file}")
    print()

    # Run the patcher
    chain = main(
        endf_gxs_dir=config['endf_gxs_dir'],
        base_chain_file=config['base_chain'],
        output_chain_file=output_chain,
        decay_file=config['decay_file'],
        mapping_mode=args.map,
        orphan_policy=args.orphan_policy,
        elis_rtol=elis_rtol,
        elis_atol=args.atol,
        verbose=verbose,
        isomer_mapping_log_file=log_file,
        prune_nn_prime_self_loops=args.prune_nn_prime_self_loops,
        suppress_single_target_yields=args.suppress_single_target_yields,
        emit_mf10_only_reactions=args.emit_mf10_only_reactions,
        reattribute_mf10_noizap=args.reattribute_mf10_noIZAP,
        mode=args.mode,
        audit_emax=args.audit_emax,
        mf10_reject_band_ratio=args.mf10_reject_band_ratio
    )

    print("\n" + "=" * 70)
    print("Done. Chain saved to:", output_chain)
    print("=" * 70)
