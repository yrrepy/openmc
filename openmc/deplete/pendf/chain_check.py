"""PENDF chain/stamp consistency checks.

Holds the chain/stamp consistency set extracted from
:mod:`openmc.deplete.microxs`: chain resolution and provenance-stamp
verification, MF=10 LFS<->chain-reaction mapping, partials-vs-total consistency
metrics, and the pre-depletion pathway-consistency gate.

Unlike the other PENDF submodules, this one IS imported at ``openmc.deplete``
package-init time (top level of :mod:`openmc.deplete.independent_operator`).
Its module-level import from :mod:`openmc.deplete.microxs` is cycle-safe because
``independent_operator`` already triggers a full ``microxs`` load today and
``microxs`` imports nothing back from ``independent_operator`` or ``pendf.*`` at
module level.

.. versionadded:: 0.15.4
"""

from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
import re
from warnings import warn

import numpy as np

from openmc.checkvalue import PathLike
from openmc.data import REACTION_MT
import openmc
from ..chain import Chain, _get_chain
from ..microxs import _ISOMER_SUFFIX


def _liso_from_gnds(name: str) -> int:
    """Return the isomeric state (LISO) parsed from a GNDS name.

    ``'Am242_m1'`` -> ``1``, ``'Am242'`` -> ``0`` (ground). The suffix is the
    product's isomer ordinal, not the MF=10 LFS level index.
    """
    match = re.search(r'_m(\d+)$', name)
    return int(match.group(1)) if match else 0


# Real TENDL-2017 partials deviate from the MF=3 total by up to ~4e-6
# per group (genuine data property); 1e-6 would warn on nearly every
# isomeric nuclide at full-library scale.
CONSISTENCY_RTOL = 1e-5

# Groups where BOTH the MF=3 total and the summed partials sit below this
# (barns) are evaluator floor placeholders (e.g. the ubiquitous 1e-20 b
# "effective zero" in JEFF-4.0, floored independently per section), not
# physics -- their relative deviation is meaningless. A group is only
# exempt when both sides are dust: a meaningful partial against a dust
# total (or vice versa) is a genuine inconsistency and still warns.
CONSISTENCY_ABS_FLOOR = 1e-15


def _partials_total_max_deviation(total_g, part_sum):
    """Max relative deviation of summed MF=10 partials from the MF=3 total.

    Returns (worst, group_idx) over groups with nonzero total, skipping
    groups where both sides are below ``CONSISTENCY_ABS_FLOOR`` (evaluator
    floor dust). Returns (0.0, -1) when no group qualifies.
    """
    nz = (total_g != 0.0) & (
        (np.abs(total_g) >= CONSISTENCY_ABS_FLOOR)
        | (np.abs(part_sum) >= CONSISTENCY_ABS_FLOOR))
    if not nz.any():
        return 0.0, -1
    dev = np.abs(part_sum[nz] - total_g[nz]) / np.abs(total_g[nz])
    worst = float(dev.max())
    return worst, int(np.nonzero(nz)[0][dev.argmax()])


def _dedupe_base_reactions(reactions: Sequence[str]) -> list[str]:
    """Strip ``_mN`` product qualifiers and dedupe, preserving first-seen order.

    Product-qualified names (e.g. ``(n,gamma)_m1``) are pathway-expansion
    *outputs*, not collapse inputs. Reducing a reaction list to its distinct base
    names keeps ``REACTION_MT[name]`` from raising on a qualified name and lets a
    chain-defaulted list (which carries qualified reaction types) feed the
    collapse unchanged.
    """
    seen: set[str] = set()
    out: list[str] = []
    for name in reactions:
        base = _ISOMER_SUFFIX.sub('', name)
        if base not in seen:
            seen.add(base)
            out.append(base)
    return out


def _default_pendf_reactions(chain: Chain) -> list[str]:
    """Base reaction list for the PENDF collapse defaulted from a chain.

    Strips ``_mN``, dedupes (see :func:`_dedupe_base_reactions`), and drops any
    reaction the collapse cannot map to an MT (no ``REACTION_MT`` entry) -- an
    activation chain carries transmutation channels the pointwise collapse does
    not support -- with a single summary warning. Unlike an explicitly passed
    reaction list, a defaulted one must not crash the build.
    """
    base = _dedupe_base_reactions(chain.reactions)
    known = [r for r in base if r in REACTION_MT]
    dropped = [r for r in base if r not in REACTION_MT]
    if dropped:
        warn('PENDF collapse skipping depletion-chain reaction(s) with no '
             f'REACTION_MT mapping: {", ".join(dropped)}.')
    return known


def _get_pendf_chain(chain_file: PathLike | Chain | None) -> Chain:
    """Resolve the depletion chain required by the PENDF collapse.

    The PENDF collapse always needs a chain: it is the authority for isomeric
    row names (each MF=10 ``LFS`` partial is bound to the chain reaction carrying
    that ``pendf_lfs``). Raises a clear error when no chain can be resolved.
    """
    if chain_file is None and 'chain_file' not in openmc.config:
        raise ValueError(
            'PENDF collapse requires chain_file -- the chain carries the '
            'isomer<->LFS mapping; build one with '
            'tools/add_pendf_isomeric_branching_to_chain.py')
    return _get_chain(chain_file)


def _pendf_library_basename(pendf_library) -> str | None:
    """Best-effort basename of a PENDF library's backing HDF5 file, or ``None``.

    :class:`~openmc.data.PendfLibrary` holds its open ``h5py.File`` handles in
    ``_files``; :class:`~openmc.data.GroupedPendfLibrary` records its ``_path``.
    Neither is part of the duck-typed collapse interface, so a library exposing
    neither (a test fake, or a directory-mode :class:`PendfLibrary` spanning
    several files) yields ``None`` and the basename is dropped from the mismatch
    message.
    """
    path = getattr(pendf_library, '_path', None)
    if path is not None:
        return Path(path).name
    files = getattr(pendf_library, '_files', None)
    if files:
        try:
            names = {Path(f.filename).name for f in files}
        except Exception:
            return None
        if len(names) == 1:
            return next(iter(names))
    return None


def _verify_pendf_chain_stamp(chain, pendf_library) -> None:
    """Warn when a stamped chain's PENDF provenance disagrees with the library.

    The chain patcher (``tools/add_pendf_isomeric_branching_to_chain.py``) stamps
    the exported chain's root element with the identity of the PENDF source it was
    built from: ``pendf_library`` (the tape-derived source identity),
    ``pendf_nuclides`` (the nuclide count), an informational ``pendf_source`` (the
    h5 basename / dir last-two components) and the provenance-only ``decay_source``
    / ``decay_library``. The PENDF collapse is chain-driven -- a stock reaction
    silently takes the MF=3 total -- so a wrong/stale chain paired with a library
    produces
    silently degraded physics that no pathway-set comparison can catch. This makes
    such a pairing self-detecting.

    Behavior:

    * Unstamped chain (no ``pendf_*`` root attrs) -> silent (backward compatible
      with chains built before stamping, and with vanilla chains).
    * Library exposing no identity string at all -- neither a tape-derived
      ``source_identity`` nor a user ``library`` label (e.g. a duck-typed test
      fake) -> silent; there is nothing to verify against.
    * The stamp is a tape-derived identity, so the stamped ``pendf_library`` is
      compared against BOTH the library's ``source_identity`` and its ``library``
      label; a mismatch fires only when it matches NEITHER. The nuclide-count
      trigger is unchanged. Either triggering yields one :class:`UserWarning`
      naming the identities compared. A differing ``pendf_source`` alone (a file
      rename) is never a trigger; the ``decay_*`` provenance attrs are never
      verified.
    """
    root_attrs = getattr(chain, 'root_attrs', None) or {}
    stamped_lib = root_attrs.get('pendf_library')
    stamped_n = root_attrs.get('pendf_nuclides')
    if stamped_lib is None and stamped_n is None:
        return  # unstamped chain -- nothing to verify

    # The stamp is a tape-derived identity; a library belongs to the chain if the
    # stamp matches EITHER its tape-derived source_identity OR its user library
    # label. A library carrying neither (test fakes) cannot be verified against,
    # so the check skips entirely.
    source_identity = getattr(pendf_library, 'source_identity', None)
    lib_name = getattr(pendf_library, 'library', None)
    identities = [x for x in (source_identity, lib_name) if x is not None]
    if not identities:
        return
    lib_nuclides = getattr(pendf_library, 'nuclides', None)
    lib_n = len(lib_nuclides) if lib_nuclides is not None else None

    # Trigger on the library string (matches NEITHER identity) OR the nuclide
    # count (compared as strings so the XML-sourced stamp and the int count
    # agree). The source basename is never a trigger.
    lib_mismatch = stamped_lib is not None and stamped_lib not in identities
    n_mismatch = (stamped_n is not None and lib_n is not None
                  and str(stamped_n) != str(lib_n))
    if not (lib_mismatch or n_mismatch):
        return

    source = root_attrs.get('pendf_source', 'unknown source')
    basename = _pendf_library_basename(pendf_library)
    against = f'{basename} ' if basename else ''
    compared = ' / '.join(repr(x) for x in identities)
    warn(
        f'PENDF provenance mismatch: chain built from {source} (library '
        f'{stamped_lib!r}, {stamped_n} nuclides) but collapsing against '
        f'{against}(identity {compared}, {lib_n} nuclides) -- regenerate the '
        f'chain from this library with '
        f'tools/add_pendf_isomeric_branching_to_chain.py')


def _chain_lfs_reactions(chain: Chain, nuc: str, base_reaction: str) -> dict:
    """Map MF=10 ``LFS`` levels to the chain reactions that consume them.

    Returns ``{pendf_lfs: ReactionTuple}`` for ``nuc``'s chain reactions whose
    base type (``_mN`` stripped) equals ``base_reaction`` and that carry a
    ``pendf_lfs``. A nuclide absent from the chain yields an empty map.

    Raises ``ValueError`` if a *qualified* (metastable) reaction for this base
    carries ``pendf_lfs=None``: such a chain was built without LFS recording and
    cannot bind MF=10 partials, so it must be regenerated with the patcher tool.
    An *unqualified* base reaction without an LFS (e.g. a plain total-fallback
    channel) is simply not bindable and is skipped.
    """
    if nuc not in chain:
        return {}
    by_lfs = {}
    for rx in chain[nuc].reactions:
        if _ISOMER_SUFFIX.sub('', rx.type) != base_reaction:
            continue
        if rx.pendf_lfs is None:
            if _ISOMER_SUFFIX.search(rx.type):
                raise ValueError(
                    f'Depletion chain reaction {nuc} {rx.type!r} carries no '
                    'pendf_lfs, so its MF=10 partials cannot be bound by LFS. '
                    'This chain was built without LFS recording; regenerate it '
                    'with tools/add_pendf_isomeric_branching_to_chain.py.')
            continue
        by_lfs[rx.pendf_lfs] = rx
    return by_lfs


def _check_pathway_consistency(chain: Chain, micro_xs: MicroXS):
    """Fail on chain/MicroXS isomeric-pathway mismatches before depletion.

    :meth:`Chain.form_rxn_matrix` matches reaction rates to chain reactions by
    reaction type, so a product-qualified pathway (e.g. ``(n,gamma)_m1``) that
    exists on only one side is silently dropped or zeroed. For every nuclide
    present in both ``chain`` and ``micro_xs``, this raises ``ValueError``
    listing every offending ``(nuclide, reaction)`` pair when either:

    (a) ``micro_xs`` carries a qualified pathway *with non-zero data* for the
        nuclide whose reaction type the chain nuclide cannot route (its rate
        would be dropped), or
    (b) the chain nuclide carries a qualified pathway whose reaction type is
        entirely *absent from the* ``micro_xs`` *reaction axis* while the
        nuclide's unqualified base row is non-zero (pathway expansion never ran
        for this reaction, so the isomer route silently gets zero rate).

    Rule (b) is axis-level: because pathway expansion stages every partial row
    (even those that group-average to zero), the presence of a qualified name in
    ``micro_xs.reactions`` marks that the collapse resolved pathways for that
    reaction. A per-nuclide zero row under a *present* qualified column is
    therefore legitimate physics (a threshold above the group structure), not a
    mismatch. The residual limitation is that a cross-library chain/MicroXS mix,
    where a nuclide's partials exist in one library but not the other, is not
    detectable per-nuclide once the axis carries the qualified name; the
    axis-level test plus rule (a) is the guarantee. Plain (unqualified) reaction
    differences and nuclides present on only one side are left alone (ordinary
    OpenMC behaviour).
    """
    offenders = []
    for nuc in micro_xs.nuclides:
        if nuc not in chain:
            continue
        chain_rxns = {r.type for r in chain[nuc].reactions}
        n_idx = micro_xs._index_nuc[nuc]
        # Reactions this MicroXS actually carries for this nuclide (non-zero row)
        micro_rxns = {rx for rx in micro_xs.reactions
                      if micro_xs.data[n_idx, micro_xs._index_rx[rx]].any()}
        # (a) MicroXS carries a qualified pathway the chain cannot route -> its
        # rate is dropped when the reaction type is not in the chain.
        for rx in micro_rxns:
            if _ISOMER_SUFFIX.search(rx) and rx not in chain_rxns:
                offenders.append((nuc, rx, 'in MicroXS but not in chain'))
        # (b) Chain carries a qualified pathway whose reaction type is absent
        # from the MicroXS reaction axis while the unqualified base carries data
        # -> pathway expansion never ran here and the isomer route silently gets
        # zero rate. A qualified column that IS in the axis (even if this
        # nuclide's row is zero) means expansion ran, so it is not a mismatch.
        for rx in chain_rxns:
            if (_ISOMER_SUFFIX.search(rx) and rx not in micro_xs.reactions
                    and _ISOMER_SUFFIX.sub('', rx) in micro_rxns):
                offenders.append((nuc, rx, 'in chain but missing from MicroXS'))

    if offenders:
        lines = '\n'.join(f'  {nuc} {rx} ({why})' for nuc, rx, why in offenders)
        raise ValueError(
            'Isomeric pathway mismatch between the depletion chain and MicroXS. '
            'These product-qualified reaction rates would be silently dropped or '
            'zeroed when forming the transmutation matrix (rates are matched by '
            f'reaction type):\n{lines}\n'
            'Regenerate the chain and MicroXS from the same PENDF MF=10 product '
            'mapping so their qualified reactions agree.')
