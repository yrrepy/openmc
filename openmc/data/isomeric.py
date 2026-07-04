"""Isomeric product mapping for MF=10 partial cross sections.

An ENDF MF=10 section stores isomeric production cross sections as a set of
partials indexed by ``LFS`` (final-level flag).  ``LFS`` is a *level index*, not
an isomer ordinal: it may skip values and does not, in general, equal the
product's isomeric state (``LISO``).  For example Am-242m production appears as
``LFS={0, 2}`` (ground + one metastable) yet the metastable is ``LISO=1``.

The physically meaningful discriminator is the excitation energy of the produced
level, ``ELFS = QM - QI``.  This module resolves each partial to a product
isomeric state by matching ``ELFS`` against per-state excitation energies
(``ELIS``) read from a decay library (``'elis'`` mode), with a positional
FISPACT-style fallback (``'lfs_order'`` mode) retained for cross-validation.

The ELIS-matching semantics and tolerance defaults are re-homed, source-neutrally,
from the GENDF depletion fork (``openmc/deplete/gendf.py`` /
``openmc/deplete/decay_elis.py``).

.. versionadded:: 0.15.4
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from warnings import warn

import openmc.checkvalue as cv

from .endf import get_evaluations, py_float_endf

__all__ = [
    'DecayState', 'ELIS_RTOL', 'ELIS_ATOL', 'elis_match', 'lookup_liso',
    'parse_decay_isomeric_levels', 'map_lfs_to_liso',
]

# Default ELIS matching tolerances (from the GENDF reference).  A 50% relative
# tolerance absorbs typical evaluation-to-evaluation differences between the
# transport (PENDF) and decay libraries while rejecting clearly wrong matches;
# no absolute floor is applied by default.
ELIS_RTOL = 0.50   # relative tolerance (dimensionless)
ELIS_ATOL = 0.0    # absolute tolerance (eV)


@dataclass
class DecayState:
    """A nuclear state identified in a decay library (MF=1/MT=451).

    Attributes
    ----------
    z : int
        Atomic number.
    a : int
        Mass number.
    elis : float
        Excitation energy of the state in eV (0.0 for the ground state).
    liso : int
        Isomeric state number (0 = ground, 1 = m1, ...).
    half_life : float or None
        Half-life in seconds, if present (MF=8/MT=457).

    """
    z: int
    a: int
    elis: float
    liso: int
    half_life: float = None


def elis_match(elfs, dk_elis, rtol=ELIS_RTOL, atol=ELIS_ATOL):
    """Return ``True`` if an ELFS value matches a decay-library ELIS.

    The decay-library value is treated as the reference (as in
    :func:`numpy.isclose`): ``abs(elfs - dk_elis) <= atol + rtol*abs(dk_elis)``.

    Parameters
    ----------
    elfs : float
        Excitation energy from the transport file (QM - QI), eV.
    dk_elis : float
        Excitation energy from the decay library (reference), eV.
    rtol, atol : float
        Relative and absolute tolerances.

    """
    return abs(elfs - dk_elis) <= atol + rtol * abs(dk_elis)


def lookup_liso(z, a, target_elis, decay_lookup, rtol=ELIS_RTOL, atol=ELIS_ATOL,
                skip_zero_elis_metastables=True, return_nearest=True):
    """Find the isomeric state whose ELIS matches ``target_elis``.

    Searches the metastable (``LISO > 0``) states of ``(z, a)`` for the one whose
    excitation energy is nearest ``target_elis`` and reports whether it lies
    within tolerance.

    Parameters
    ----------
    z, a : int
        Atomic and mass number of the product nuclide.
    target_elis : float
        Excitation energy to match (from MF=10: QM - QI), eV.
    decay_lookup : dict
        ``{(Z, A): [DecayState, ...]}`` from :func:`parse_decay_isomeric_levels`.
    rtol, atol : float
        Tolerances passed to :func:`elis_match`.
    skip_zero_elis_metastables : bool
        If ``True`` (default) ignore metastable states carrying ``ELIS == 0`` (a
        decay-library data-quality issue, since a metastable must be excited).
    return_nearest : bool
        If ``True`` (default) report the nearest state with ``status='nearest'``
        when it is out of tolerance; otherwise report ``status='no_match'``.

    Returns
    -------
    dict
        A ``'status'`` key with one of ``'matched'``, ``'nearest'``,
        ``'no_match'``, ``'no_decay_data'``, ``'no_metastables'``, or
        ``'zero_elis_only'``, plus status-dependent fields (``'liso'``,
        ``'dk_elis'``, ``'diff_pct'``, ``'skipped_states'``).

    """
    decay_states = decay_lookup.get((z, a), [])
    if not decay_states:
        return {'status': 'no_decay_data'}

    metastables_valid = []       # LISO > 0, ELIS > 0
    metastables_zero_elis = []   # LISO > 0, ELIS == 0 (data-quality issue)
    for state in decay_states:
        if state.liso == 0:
            continue
        if state.elis == 0.0:
            metastables_zero_elis.append((state.liso, state.elis, state.half_life))
        else:
            metastables_valid.append(state)

    if not metastables_valid and metastables_zero_elis:
        if skip_zero_elis_metastables:
            return {'status': 'zero_elis_only',
                    'skipped_states': metastables_zero_elis}
        metastables_valid = [
            DecayState(z=z, a=a, elis=0.0, liso=liso, half_life=hl)
            for liso, _, hl in metastables_zero_elis
        ]

    if not metastables_valid:
        return {'status': 'no_metastables'}

    # Deliberate mixed-metric selection (load-bearing and validated against the
    # LFS 2 -> m1 and LFS 4 -> m2 level-index resolutions):
    #   * SELECT the nearest metastable by ABSOLUTE ELIS difference, then
    #   * ACCEPT it only if within the RELATIVE tolerance (elis_match, below).
    # On an exact absolute tie, ``min`` keeps the first candidate in iteration
    # order -- the lower LISO, since decay states are listed ground -> m1 -> m2.
    # This two-metric behavior is intentional; downstream mappings depend on it,
    # so do not collapse it to a single (all-absolute or all-relative) metric.
    liso, dk_elis = min(
        ((s.liso, s.elis) for s in metastables_valid),
        key=lambda item: abs(target_elis - item[1]),
    )

    if elis_match(target_elis, dk_elis, rtol, atol):
        return {'status': 'matched', 'liso': liso, 'dk_elis': dk_elis}
    if not return_nearest:
        return {'status': 'no_match'}
    diff_pct = (abs(target_elis - dk_elis) / abs(dk_elis) * 100
                if dk_elis != 0 else float('inf'))
    return {'status': 'nearest', 'liso': liso, 'dk_elis': dk_elis,
            'diff_pct': diff_pct}


def _half_life(material):
    """Return the half-life (s) from an MF=8/MT=457 section, or ``None``."""
    section = material.section_data.get((8, 457))
    if not isinstance(section, dict):
        return None
    t12 = section.get('T1/2')
    if t12 is None:
        return None
    # endf returns T1/2 as a (value, uncertainty) pair
    return float(t12[0]) if isinstance(t12, tuple) else float(t12)


def parse_decay_isomeric_levels(decay_path):
    """Build an ELIS lookup table from a decay library.

    Reads ``ELIS``/``LISO`` (MF=1/MT=451) for every material and groups the
    resulting :class:`DecayState` objects by ``(Z, A)``.  ``decay_path`` may be a
    directory of per-nuclide ENDF files (FISPACT style, e.g. ``Am242``,
    ``Am242m``, ``Am242n``) or a single file concatenating several materials.

    Parameters
    ----------
    decay_path : str or path-like
        Directory of decay files, or a single concatenated decay file.

    Returns
    -------
    dict
        ``{(Z, A): [DecayState, ...]}``.

    """
    decay_path = Path(decay_path)
    if not decay_path.exists():
        raise FileNotFoundError(f"Decay library path not found: {decay_path}")

    if decay_path.is_dir():
        paths = [p for p in sorted(decay_path.iterdir()) if p.is_file()]
    else:
        paths = [decay_path]

    lookup = defaultdict(list)
    for path in paths:
        try:
            materials = get_evaluations(path)
        except Exception as exc:  # malformed / non-standard ENDF tape
            # Many FISPACT/EASY-II per-nuclide decay files terminate with a
            # MEND record (MAT=0) but omit the ENDF TEND record (MAT=-1), so
            # the general reader parses a spurious material past end-of-file
            # and raises.  The material itself is well-formed; recover the
            # ELIS lookup fields with a minimal fixed-column MF=1/451 read.
            states = _manual_decay_states(path)
            if not states:
                warn(f"Skipping decay file {path.name}: {exc}")
                continue
            warn(f"manual-parse: {path.name}: get_evaluations failed ({exc}); "
                 f"recovered {len(states)} MF=1/451 state(s) by fixed-column read.")
            for state in states:
                lookup[(state.z, state.a)].append(state)
            continue
        for material in materials:
            info = material.section_data.get((1, 451))
            if not info:
                continue
            za = int(info.get('ZA', 0))
            if za < 1001:
                continue
            z, a = za // 1000, za % 1000
            lookup[(z, a)].append(DecayState(
                z=z, a=a,
                elis=float(info.get('ELIS', 0.0)),
                liso=int(info.get('LISO', 0)),
                half_life=_half_life(material),
            ))
    return dict(lookup)


def _manual_decay_states(path):
    """Extract MF=1/451 states from a decay tape that defeats the ENDF reader.

    Reads only the two records needed for the ELIS lookup -- the HEAD (``ZA``)
    and the second CONT record (``ELIS``/``LIS``/``LISO``) -- of every MF=1/451
    section, using fixed ENDF column positions.  Half-lives (MF=8/457) are not
    recovered; they are optional for the lookup.  Returns a list of
    :class:`DecayState` (empty if nothing usable is found).
    """
    states = []
    section = []           # collected MF=1/451 lines for the current material
    with open(path) as fh:
        for line in fh:
            if len(line) < 75:
                continue
            mf, mt = line[70:72].strip(), line[72:75].strip()
            if mf == '1' and mt == '451':
                section.append(line)
            elif section:            # first non-451 line closes the section
                _append_manual_state(section, states)
                section = []
    if section:
        _append_manual_state(section, states)
    return states


def _append_manual_state(section, states):
    """Parse a collected MF=1/451 header (``section``) into ``states``."""
    if len(section) < 2:
        return
    head, rec2 = section[0], section[1]
    za = int(py_float_endf(head[0:11]))
    if za < 1001:
        return
    elis = py_float_endf(rec2[0:11]) if rec2[0:11].strip() else 0.0
    liso = int(py_float_endf(rec2[33:44])) if rec2[33:44].strip() else 0
    states.append(DecayState(z=za // 1000, a=za % 1000, elis=elis, liso=liso))


def map_lfs_to_liso(partials, decay_lookup, mode='elis', rtol=ELIS_RTOL,
                    atol=ELIS_ATOL, skip_zero_elis_metastables=True, context=''):
    """Map MF=10 final-level indices (LFS) to product isomeric states (LISO).

    Ground partials (``LFS == 0``) always map to ``LISO = 0``.  Metastable
    partials (``LFS > 0``) are resolved by excitation energy:

    * ``mode='elis'`` matches ``ELFS`` against decay-library ``ELIS`` values,
      keeping only the closest partial when several map to the same state.
    * ``mode='lfs_order'`` assigns ``LISO`` by ascending-LFS position (1, 2, ...),
      a FISPACT-like fallback retained for cross-validation.

    Partials that cannot be resolved are omitted from the result and a warning is
    issued (branching is expected to be renormalized downstream); nothing is
    raised.

    Parameters
    ----------
    partials : iterable of dict
        One entry per MF=10 partial with integer keys ``'lfs'`` and ``'izap'``
        and float key ``'elfs'`` (= QM - QI, eV).
    decay_lookup : dict
        ``{(Z, A): [DecayState, ...]}`` from :func:`parse_decay_isomeric_levels`.
    mode : {'elis', 'lfs_order'}
        Mapping strategy.
    rtol, atol : float
        ELIS matching tolerances (``'elis'`` mode; also used for the
        ``'lfs_order'`` cross-check warning).
    skip_zero_elis_metastables : bool
        Passed to :func:`lookup_liso`.
    context : str
        Label used in warnings, e.g. ``"In115 MT=102"``.

    Returns
    -------
    dict
        ``{lfs: liso}`` for every partial that resolves to a product state.

    """
    cv.check_value('mode', mode, ('elis', 'lfs_order'))

    mapping = {}
    metastables = []
    for p in partials:
        if p['lfs'] == 0:
            mapping[0] = 0                       # ground state
        else:
            metastables.append(p)
    if not metastables:
        return mapping

    if mode == 'lfs_order':
        return _map_lfs_order(metastables, decay_lookup, mapping, rtol, atol,
                              skip_zero_elis_metastables, context)
    return _map_elis(metastables, decay_lookup, mapping, rtol, atol,
                     skip_zero_elis_metastables, context)


def _map_elis(metastables, decay_lookup, mapping, rtol, atol,
              skip_zero_elis_metastables, context):
    """ELIS-based mapping of metastable partials; see :func:`map_lfs_to_liso`."""
    # First pass: look up each metastable's LISO by ELFS.
    results = []
    for p in metastables:
        z, a = p['izap'] // 1000, p['izap'] % 1000
        result = lookup_liso(z, a, p['elfs'], decay_lookup, rtol=rtol, atol=atol,
                             skip_zero_elis_metastables=skip_zero_elis_metastables)
        results.append((p, z, a, result))

    # Detect several LFS matching the same LISO; keep the closest in ELFS.
    liso_to_matches = defaultdict(list)
    for idx, (p, _z, _a, result) in enumerate(results):
        if result['status'] == 'matched':
            diff = abs(p['elfs'] - result['dk_elis'])
            liso_to_matches[result['liso']].append((diff, idx, p))
    discarded = set()
    for liso, matches in liso_to_matches.items():
        if len(matches) > 1:
            matches.sort(key=lambda m: m[0])
            keeper = matches[0][2]
            dropped = matches[1:]
            discarded.update(idx for _d, idx, _p in dropped)
            warn(f"DUPLICATE_MAPPING: {context} -> _m{liso}: LFS "
                 f"{', '.join(str(m[2]['lfs']) for m in dropped)} also match "
                 f"this state; keeping closest LFS={keeper['lfs']} and dropping "
                 f"the rest.")

    # Second pass: assign matches; warn+skip everything unresolved.
    for idx, (p, z, a, result) in enumerate(results):
        if idx in discarded:
            continue
        status = result['status']
        if status == 'matched':
            mapping[p['lfs']] = result['liso']
        elif status == 'nearest':
            warn(f"ELIS_TOL_EXCEEDED: {context} LFS={p['lfs']} "
                 f"(ELFS={p['elfs']:.0f} eV): nearest decay state _m"
                 f"{result['liso']} at {result['dk_elis']:.0f} eV "
                 f"({result['diff_pct']:.1f}% off); product not mapped.")
        elif status == 'zero_elis_only':
            warn(f"ZERO_ELIS_METASTABLES: {context} LFS={p['lfs']}: decay "
                 f"library metastable(s) carry ELIS=0; product not mapped.")
        else:  # no_decay_data / no_metastables / no_match
            warn(f"NO_METASTABLE_DECAY_DATA: {context} LFS={p['lfs']} "
                 f"(ELFS={p['elfs']:.0f} eV): {status} for (Z={z}, A={a}); "
                 f"product not mapped.")
    return mapping


def _map_lfs_order(metastables, decay_lookup, mapping, rtol, atol,
                   skip_zero_elis_metastables, context):
    """Positional (FISPACT-like) mapping; see :func:`map_lfs_to_liso`."""
    metastables = sorted(metastables, key=lambda p: p['lfs'])
    z, a = metastables[0]['izap'] // 1000, metastables[0]['izap'] % 1000
    dk_meta_count = sum(1 for s in decay_lookup.get((z, a), []) if s.liso > 0)

    for position, p in enumerate(metastables, start=1):
        if position > dk_meta_count:
            warn(f"LFS_ORDER_DROPPED: {context} LFS={p['lfs']}: decay library "
                 f"has only {dk_meta_count} metastable state(s); product not "
                 f"mapped.")
            continue
        mapping[p['lfs']] = position
        # Cross-check against ELIS matching (diagnostic only).
        result = lookup_liso(z, a, p['elfs'], decay_lookup, rtol=rtol, atol=atol,
                             skip_zero_elis_metastables=skip_zero_elis_metastables)
        if result['status'] == 'matched' and result['liso'] != position:
            warn(f"LFS_ORDER_ELIS_MISMATCH: {context} LFS={p['lfs']} mapped to "
                 f"_m{position} by position, but ELIS matching suggests "
                 f"_m{result['liso']}; consider mapping='elis'.")
    return mapping
