"""PENDF ground-pathway physics.

Holds the ground-pathway machinery extracted from :mod:`openmc.deplete.microxs`:
the silence-fill (:func:`_silence_fill_ground`), ground-by-balance
(:func:`_balance_ground`, :func:`_balance_remainder`) and partial-binding-veto
(:func:`_partial_binding_veto`, :func:`_partial_binding_veto_grouped`)
functions behind :func:`openmc.deplete.microxs._build_xs_table_pendf`.

These functions carry the FROZEN band-gate-adjacent semantics -- the silence
threshold, the significance floor, the spike cap and the sidedness of every gate
-- and are isolated here on purpose so that the freeze surface is a single file;
any semantic change must be flagged to the GENDF side.

This submodule is never imported at ``openmc.deplete.pendf`` package-init time,
so it may import from :mod:`openmc.deplete.microxs` at module level.

.. versionadded:: 0.15.4
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .chain_check import CONSISTENCY_ABS_FLOOR


# Below this Sigma(all MF=10 partials)/total the isomeric branching is a
# bit-identical evaluator placeholder (census: silent <= 3e-5, live >= 0.79)
# while the real cross section lives only in the MF=3 total. On the qualified
# collapse path the ground pathway is then filled by balance
# (total - demanded metastables) inside the ground partial's own energy range;
# see _silence_fill_ground. Hardwired (no per-run knob), mirroring the
# MF=10-always-on precedent.
SILENCE_EPS = 1e-3

# Sigma(all MF=10 partials)/total above this is a corrupt/over-summing grid,
# so the opt-in partial-binding switch refuses to bind such a reaction's LFS=0
# ground (decision B1). Same value as the retired patcher gate's spike cap.
_SPIKE_CAP = 1.5


@dataclass(eq=False)
class _SilenceFill:
    """Result of the in-domain silence-fill of a reaction's ground pathway.

    ``fired`` is True when at least one in-domain group is silent (the ground
    was filled). ``e_dom`` / ``ground_dom`` are the union grid restricted to the
    LFS=0 partial's own tabulated range and the filled ground on it, ready to
    :func:`_group_average` into the tally structure. The remaining fields are
    full-union-grid diagnostics reused by the partial-binding veto (a follow-on
    feature): ``e`` the union of the MF=3 and every MF=10 partial grid, ``total``
    the MF=3 total on ``e``, ``sum_all`` every library partial (all LFS,
    demanded or not) on ``e``, ``silent`` the ``total > floor`` and
    ``sum_all/total < eps`` mask, and ``ground0_range`` the ``(emin, emax)`` of
    the LFS=0 partial (``None`` when the reaction carries no LFS=0 partial).
    """
    fired: bool
    e_dom: np.ndarray
    ground_dom: np.ndarray
    e: np.ndarray
    total: np.ndarray
    sum_all: np.ndarray
    silent: np.ndarray
    ground0_range: tuple[float, float] | None


def _silence_fill_ground(
    pathways_fn,
    pathway_xs_fn,
    nuc: str,
    mt: int,
    energy3: np.ndarray,
    xs3: np.ndarray,
    demanded_lfs: set,
    eps: float = SILENCE_EPS,
    floor: float = CONSISTENCY_ABS_FLOOR,
) -> _SilenceFill:
    """In-domain silence-fill of a reaction's ground (LFS=0) pathway.

    For a qualified (n,gamma)-style reaction whose MF=10 branching is a thermal
    placeholder (every partial ~1e-20 b while the MF=3 total carries the real
    1/v capture), replace the placeholder ground with ``total - Sigma(demanded
    metastables)`` wherever the branching is silent
    (``Sigma(all partials)/total < eps`` with ``total > floor``), restricted to
    the LFS=0 partial's own tabulated range ("in-domain", so the fill only
    REPLACES stored placeholder values and never extends the evaluation past its
    last tabulated point). Elsewhere the ground stays the source-faithful LFS=0
    partial, so a channel whose branching is live wherever the total is does not
    fire and its ground row is bit-identical to the raw LFS=0 partial average.

    The silence *test* sums EVERY library partial (all LFS -- including
    undemanded ELIS-dropped partials and multi-product ``LFS{l}_ZAP{z}`` entries
    enumerated via ``pathways_fn``); the *fill* subtracts only the demanded
    metastables (``LFS > 0`` in ``demanded_lfs``). Using the full partial set
    keeps an undemanded live partial from having its cross section absorbed into
    ground by the subtraction. Positivity is automatic: a filled group is
    silent, so ``Sigma(demanded m) <= Sigma(all) < eps*total`` and the ground
    stays ``>= (1 - eps)*total > 0`` -- no clamp.

    The union-grid / lin-lin / zero-fill-outside-range arithmetic mirrors
    ``claude/ground_by_balance/merit_probe.load_channel``.

    Parameters
    ----------
    pathways_fn : callable
        ``pathways(nuclide, mt) -> [(lfs, izap), ...]`` (pointwise library).
    pathway_xs_fn : callable
        ``pathway_xs(nuclide, mt, lfs, izap) -> (energy, xs)`` (pointwise).
    nuc : str
        Nuclide GNDS name.
    mt : int
        Reaction MT number.
    energy3, xs3 : numpy.ndarray
        MF=3 total energy grid and cross section.
    demanded_lfs : set of int
        Chain-demanded LFS levels; only the ``LFS > 0`` members are subtracted.
    eps, floor : float
        Silence threshold on ``Sigma(all)/total`` and the significance floor on
        ``total``.
    """
    energy3 = np.asarray(energy3, dtype=float)
    xs3 = np.asarray(xs3, dtype=float)

    grids = [energy3]
    partials = []
    for lfs, izap in pathways_fn(nuc, mt):
        pe, pxs = pathway_xs_fn(nuc, mt, lfs, izap)
        pe = np.asarray(pe, dtype=float)
        pxs = np.asarray(pxs, dtype=float)
        partials.append((lfs, pe, pxs))
        grids.append(pe)

    e = np.unique(np.concatenate(grids))
    total = np.interp(e, energy3, xs3)
    sum_all = np.zeros_like(e)
    sum_demanded_meta = np.zeros_like(e)
    ground0 = np.zeros_like(e)
    g0_lo = g0_hi = None
    for lfs, pe, pxs in partials:
        y = np.interp(e, pe, pxs, left=0.0, right=0.0)
        sum_all = sum_all + y
        if lfs == 0:
            ground0 = ground0 + y
            g0_lo = pe[0] if g0_lo is None else min(g0_lo, pe[0])
            g0_hi = pe[-1] if g0_hi is None else max(g0_hi, pe[-1])
        elif lfs in demanded_lfs:
            sum_demanded_meta = sum_demanded_meta + y

    sig = total > floor
    ratio = np.divide(sum_all, total, out=np.zeros_like(total), where=sig)
    silent = sig & (ratio < eps)

    if g0_lo is None:
        in_domain = np.zeros_like(e, dtype=bool)
        ground0_range = None
    else:
        in_domain = (e >= g0_lo) & (e <= g0_hi)
        ground0_range = (float(g0_lo), float(g0_hi))
    mask = silent & in_domain
    fired = bool(mask.any())

    # Restrict the filled ground to the LFS=0 native range so the terminal
    # interval above the ground partial's last tabulated point stays
    # source-faithful (zero contribution, no np.interp zero-fill down-ramp).
    e_dom = e[in_domain]
    ground_dom = np.where(mask[in_domain],
                          total[in_domain] - sum_demanded_meta[in_domain],
                          ground0[in_domain])
    return _SilenceFill(fired=fired, e_dom=e_dom, ground_dom=ground_dom,
                        e=e, total=total, sum_all=sum_all, silent=silent,
                        ground0_range=ground0_range)


def _balance_remainder(total, sum_meta) -> tuple[np.ndarray, int]:
    """Ground remainder ``max(0, total - sum_meta)`` and its clamped-point count.

    Shared by the pointwise (union-grid) and grouped (group-wise) ground-by-
    balance paths so both clamp and count identically. The count is of points
    where the raw remainder was strictly negative -- source data whose
    metastable partials over-sum their MF=3 total there.
    """
    raw = np.asarray(total, dtype=float) - np.asarray(sum_meta, dtype=float)
    return np.maximum(raw, 0.0), int(np.count_nonzero(raw < 0.0))


def _balance_ground(
    pathways_fn,
    pathway_xs_fn,
    nuc: str,
    mt: int,
    energy3: np.ndarray,
    xs3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Pointwise ground-by-balance: ``max(0, total - Sigma(metastable partials))``.

    For a reaction whose MF=10 carries only isomeric (``LFS > 0``) partials while
    an MF=3 total exists, the ground production is not tabulated anywhere but is
    implied: each event yields exactly one final state, so whatever the total
    does not deposit in a metastable level is ground production. Evaluated on the
    union of the MF=3 grid and every partial grid (partials interpolated with
    zero-fill outside their own range, exact because partials are lin-lin at
    ingest), then clamped pointwise at zero.

    **Every** library metastable partial is subtracted, not only the chain-
    demanded ones: yield to an untracked level must not be reattributed to the
    ground channel (GENDF ``d010d2618`` convention, and consistent with the
    collapse's "library extras are ignored silently" demand rule).

    This is NOT :func:`_silence_fill_ground`: that fill needs the LFS=0 partial's
    own energy range as its domain (absent here, by definition) and is gated on
    the branching being *silent*, which a live metastable (In115 ``(n,n')``,
    BR 0.1565) is not. Only the union-grid arithmetic is shared.

    Parameters
    ----------
    pathways_fn : callable
        ``pathways(nuclide, mt) -> [(lfs, izap), ...]`` (pointwise library).
    pathway_xs_fn : callable
        ``pathway_xs(nuclide, mt, lfs, izap) -> (energy, xs)`` (pointwise).
    nuc : str
        Nuclide GNDS name.
    mt : int
        Reaction MT number.
    energy3, xs3 : numpy.ndarray
        MF=3 total energy grid and cross section.

    Returns
    -------
    tuple
        ``(e, ground, n_clamped)``: the union grid clipped to the MF=3 domain,
        the clamped ground remainder on it (ready for :func:`_group_average`),
        and the number of grid points where the raw remainder was negative.
    """
    energy3 = np.asarray(energy3, dtype=float)
    xs3 = np.asarray(xs3, dtype=float)

    grids = [energy3]
    partials = []
    for lfs, izap in pathways_fn(nuc, mt):
        if lfs <= 0:
            continue     # the caller's trigger guarantees no LFS=0 partial
        pe, pxs = pathway_xs_fn(nuc, mt, lfs, izap)
        pe = np.asarray(pe, dtype=float)
        pxs = np.asarray(pxs, dtype=float)
        partials.append((pe, pxs))
        grids.append(pe)

    # In-domain only: ``np.interp`` edge-clamps outside the MF=3 grid, so a
    # metastable partial tabulated past ``energy3[-1]`` (or below its first
    # point) would otherwise fabricate ground production outside the
    # evaluation's own domain. Clip the union grid to the MF=3 range first --
    # the same in-domain discipline :func:`_silence_fill_ground` applies. The
    # MF=3 endpoints are always retained (``energy3`` is itself in ``grids``),
    # so a fully in-domain reaction is unaffected.
    e = np.unique(np.concatenate(grids))
    e = e[(e >= energy3[0]) & (e <= energy3[-1])]
    total = np.interp(e, energy3, xs3)
    sum_meta = np.zeros_like(e)
    for pe, pxs in partials:
        sum_meta = sum_meta + np.interp(e, pe, pxs, left=0.0, right=0.0)

    ground, n_clamped = _balance_remainder(total, sum_meta)
    return e, ground, n_clamped


def _normalize_partial_binding(partial_binding):
    """Normalize ``partial_binding`` to ``False``, ``True``, or a set of pairs.

    Accepts ``False`` (off), ``True`` (bind every candidate that survives the
    veto), or a collection of ``(nuclide, reaction_type)`` string pairs (bind
    only those, e.g. ``{('Np239', '(n,gamma)')}``); ``None`` is treated as
    ``False``. Returns ``False``, ``True``, or a ``set`` of ``(str, str)`` pairs.
    """
    if isinstance(partial_binding, bool):
        return partial_binding
    if partial_binding is None:
        return False
    try:
        return {(str(nuc), str(rx)) for nuc, rx in partial_binding}
    except (TypeError, ValueError):
        raise ValueError(
            'partial_binding must be a bool or a collection of '
            f'(nuclide, reaction_type) pairs; got {partial_binding!r}')


def _partial_binding_veto(fill: _SilenceFill) -> str | None:
    """Veto reason for binding a stock ground to the LFS=0 partial, or ``None``.

    The pointwise/tape analogue of decision B1: refuse to bind when, at or below
    the LFS=0 partial's last tabulated point and where the MF=3 total is
    significant, either the branching is silent anywhere (``'silent'`` -- a
    placeholder/gap region exists, so the stored LFS=0 ground cannot be trusted
    as the full ground row; the Bk247 thermal-placeholder class) or
    ``Sigma(all)/total`` exceeds :data:`_SPIKE_CAP` anywhere (``'spike'`` -- a
    corrupt/over-summing grid). Energies above the LFS=0 partial's last point
    (the universal terminal sliver where the MF=10 partials have ended but the
    MF=3 total still tails off) are NOT examined, matching the in-domain
    silence-fill decision (a bound channel there behaves like a qualified one).
    """
    if fill.ground0_range is None:
        return None
    in_dom = fill.e <= fill.ground0_range[1]
    if bool((fill.silent & in_dom).any()):
        return 'silent'
    sig = fill.total > CONSISTENCY_ABS_FLOOR
    ratio = np.divide(fill.sum_all, fill.total,
                      out=np.zeros_like(fill.total), where=sig)
    if bool((sig & in_dom & (ratio > _SPIKE_CAP)).any()):
        return 'spike'
    return None


def _partial_binding_veto_grouped(sum_all_g: np.ndarray,
                                  total_g: np.ndarray) -> str | None:
    """Group-space veto analogue for the grouped-library partial-binding path.

    Conservative relative to :func:`_partial_binding_veto`: a grouped library
    carries no pointwise grid, so the whole group range is examined and a top
    group lying wholly in the terminal sliver (partials ended, MF=3 total still
    live) may over-veto. Silent group = ``total`` significant and
    ``Sigma(all)/total < SILENCE_EPS``; spike group = ratio ``> _SPIKE_CAP``.
    """
    sig_g = total_g > CONSISTENCY_ABS_FLOOR
    ratio_g = np.divide(sum_all_g, total_g,
                        out=np.zeros_like(total_g), where=sig_g)
    if bool((sig_g & (ratio_g < SILENCE_EPS)).any()):
        return 'silent'
    if bool((sig_g & (ratio_g > _SPIKE_CAP)).any()):
        return 'spike'
    return None
