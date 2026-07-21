#!/usr/bin/env python
"""Pre-bin a pointwise PENDF HDF5 library into a grouped PENDF HDF5 library.

Flat-weights every chain-relevant MF=3 total and MF=10 partial cross section
onto a fixed energy group structure with
:func:`openmc.deplete.microxs._group_average` and stores the results dense
(gzip-compressed) so the depletion collapse can skip runtime binning. The
output schema is ``format='pendf-grouped'`` version 2:

* root attrs: ``format``, ``version``, ``source`` (abs path of the pointwise
  file), ``dtype``, plus the source's ``library``/``temperature``/
  ``source_identity`` attrs; a build baked with the silence-fill (the default)
  also stamps ``ground_fill='silence-fill'`` with the ``silence_eps``/
  ``silence_floor`` constants it used;
* ``/group_edges`` float64[G+1] ascending eV;
* ``/<Nuclide>/`` copies the source nuclide attrs verbatim;
* ``/<Nuclide>/MT<mt>/xs_g`` the MF=3 group cross section (MT attrs copied);
* ``/<Nuclide>/MT<mt>/LFS<l>/xs_g`` each MF=10 partial (LFS attrs copied); a
  ground (``LFS0``) whose thermal placeholder was silence-filled carries a
  ``silence_filled=1`` attr.

Every chain-relevant MT present in the source is written in full -- the total
and *every* LFS partial, including all-zero rows -- so a grouped collapse
reproduces the pointwise ``pathways()`` list (and hence a bit-exact table).

The ground (LFS=0) pathway of a qualified non-lumped MF=10 reaction is
silence-filled at build time (``fill=True``, the default): where the branching
is a thermal placeholder (every partial silent while the MF=3 total carries the
real 1/v capture), the stored ground becomes ``total - Sigma(metastables)`` in
the LFS=0 partial's own range -- the same in-domain fill the pointwise depletion
collapse applies at runtime, baked here where the pointwise source is still in
hand (grouped libraries carry no pointwise data to fill later). ``fill=False``
stores the source-faithful bytes unchanged.
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path
from warnings import warn

import h5py
import numpy as np

from openmc.deplete.chain import REACTIONS
from openmc.deplete.microxs import (
    CONSISTENCY_ABS_FLOOR,
    CONSISTENCY_RTOL,
    SILENCE_EPS,
    _group_average,
    _partials_total_max_deviation,
    _silence_fill_ground,
)
from openmc.mgxs import GROUP_STRUCTURES

# Grouped-schema constants.
GROUPED_FORMAT = 'pendf-grouped'
# Grouped-schema version stamped at the root as the ``version`` attribute:
#   1 -- MF=10 subgroups are always named ``LFS<l>``.
#   2 -- an LFS shared by >=2 product IZAPs is named ``LFS<l>_ZAP<izap>``.
# The build-time silence-fill bake (root ``ground_fill``, per-``LFS0``
# ``silence_filled`` attrs) is additive within version 2: a reader that predates
# it ignores the extra attrs and the LFS0 value it reads is a valid group cross
# section either way, so it warrants no version bump.
GROUPED_VERSION = 2


def chain_relevant_mts() -> set[int]:
    """Return the MT numbers depletion activation cares about.

    The union of every ``REACTIONS`` entry's MTs plus MT=4: ``(n,n')`` joins
    ``REACTIONS`` only on the separate pendf-chain branch, so it is added here
    explicitly so grouped libraries carry it regardless of branch.
    """
    mts = {mt for info in REACTIONS.values() for mt in info.mts}
    mts.add(4)  # (n,n') joins REACTIONS on the separate pendf-chain branch
    return mts


def bin_pendf_library(
    pendf_in,
    out,
    group_edges,
    dtype: str = 'float64',
    mts=None,
    nuclides=None,
    fill: bool = True,
):
    """Bin a pointwise PENDF HDF5 library onto a group structure.

    Parameters
    ----------
    pendf_in : path-like
        Source pointwise PENDF HDF5 file (``/<Nuclide>/MT<mt>/`` layout with
        ``energy``/``xs`` datasets and ``LFS<l>/`` MF=10 subgroups).
    out : path-like
        Destination grouped HDF5 file.
    group_edges : numpy.ndarray
        Ascending energy group boundaries in [eV], length ``n_groups + 1``.
    dtype : {'float64', 'float32'}, optional
        Storage dtype of the group cross sections. Binning is always done in
        float64; float32 only reduces the stored precision. Default 'float64'.
    mts : iterable of int, optional
        Explicit MT set to bin. Defaults to :func:`chain_relevant_mts`.
    nuclides : iterable of str, optional
        Subset of source nuclides to bin. Defaults to all.
    fill : bool, optional
        Bake the in-domain silence-fill of each qualified reaction's ground
        (LFS=0) pathway at build time (default ``True``), matching the always-on
        collapse-time fill the pointwise depletion path applies. For a non-lumped
        MF=10 reaction with a plain ``LFS0`` ground and >=1 plain metastable
        ``LFS<l>`` (``l>0``) partial, the stored ground becomes ``total -
        Sigma(metastable partials)`` wherever every library partial is silent
        (``Sigma(all)/total < SILENCE_EPS`` with ``total`` above
        ``CONSISTENCY_ABS_FLOOR``), restricted to the LFS=0 partial's own
        tabulated range. Lumped ``LFS<l>_ZAP<izap>`` reactions and reactions
        lacking a ground or a metastable are binned verbatim. ``fill=False``
        reproduces the source-faithful bytes exactly (a debugging /
        source-faithful escape hatch).

        The bake uses ALL library LFS as the demanded set, so it is bit-exact
        with the collapse-time fill whenever the depletion chain demands the
        library's full LFS set (the normal case). When a chain demands a proper
        subset (library-extra ELIS-dropped LFS, the Sn122 class), the two differ
        only by the silent undemanded partials' magnitude (< ``SILENCE_EPS`` *
        ``total``, physically ~1e-20 b) -- an unfixable and negligible caveat.

    Returns
    -------
    dict
        Summary statistics: ``rows`` (xs_g datasets written), ``worst_dev``
        (worst Σpartials-vs-total relative deviation), ``n_warnings``,
        ``filled`` (list of ``'<Nuclide> MT=<mt>'`` whose ground was
        silence-filled), and ``wall_time`` in seconds.
    """
    t0 = time.perf_counter()
    edges = np.asarray(group_edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError('group_edges must be a 1-D ascending array of length '
                         '>= 2')
    if dtype not in ('float64', 'float32'):
        raise ValueError(f"dtype must be 'float64' or 'float32', got {dtype!r}")
    want_mts = set(mts) if mts is not None else chain_relevant_mts()

    rows = 0
    worst_dev = 0.0
    n_warnings = 0
    filled: list[str] = []

    with h5py.File(pendf_in, 'r') as src, h5py.File(out, 'w') as dst:
        # Root attrs: schema identity plus provenance carried from the source.
        dst.attrs['format'] = GROUPED_FORMAT
        dst.attrs['version'] = GROUPED_VERSION
        dst.attrs['source'] = str(Path(pendf_in).resolve())
        dst.attrs['dtype'] = dtype
        if fill:
            # Build-mode provenance (additive within version 2): records that the
            # silence-fill was baked, with the microxs constants it keyed on
            # (never re-hardcoded here). Present whenever fill was requested, even
            # if no reaction actually fired.
            dst.attrs['ground_fill'] = 'silence-fill'
            dst.attrs['silence_eps'] = SILENCE_EPS
            dst.attrs['silence_floor'] = CONSISTENCY_ABS_FLOOR
        for key in ('library', 'temperature', 'source_identity'):
            if key in src.attrs:
                dst.attrs[key] = src.attrs[key]
        dst.create_dataset('group_edges', data=edges)

        # Adapters over the open pointwise source: the exact (LFS, IZAP) pairs and
        # (energy, xs) arrays PendfLibrary.pathways/pathway_xs expose to the
        # collapse, so the build-time fill drives _silence_fill_ground with
        # collapse-identical inputs (no reimplementation of the fill math).
        def pathways_fn(nuc, mt):
            g = src[f'{nuc}/MT{mt}']
            return sorted((int(g[k].attrs['LFS']), int(g[k].attrs['IZAP']))
                          for k in g if k.startswith('LFS'))

        def pathway_xs_fn(nuc, mt, lfs, izap=None):
            g = src[f'{nuc}/MT{mt}']
            for k in g:
                if (k.startswith('LFS') and int(g[k].attrs['LFS']) == lfs
                        and (izap is None or int(g[k].attrs['IZAP']) == izap)):
                    return g[k]['energy'][()], g[k]['xs'][()]
            raise KeyError(f'{nuc!r} MT={mt} has no MF=10 partial LFS={lfs}.')

        src_nuclides = list(src.keys()) if nuclides is None else list(nuclides)
        for nuc in src_nuclides:
            src_nuc = src[nuc]
            dst_nuc = dst.create_group(nuc)
            for k, v in src_nuc.attrs.items():
                dst_nuc.attrs[k] = v

            for mt_name in src_nuc:
                if not mt_name.startswith('MT'):
                    continue
                mt = int(mt_name[2:])
                if mt not in want_mts:
                    continue
                src_mt = src_nuc[mt_name]
                dst_mt = dst_nuc.create_group(mt_name)
                for k, v in src_mt.attrs.items():
                    dst_mt.attrs[k] = v

                total_g = _group_average(
                    src_mt['energy'][()], src_mt['xs'][()], edges)
                dst_mt.create_dataset(
                    'xs_g', data=total_g.astype(dtype),
                    compression='gzip', compression_opts=4)
                rows += 1

                # Build-time silence-fill decision (chain-free): a non-lumped
                # MF=10 reaction with a plain LFS0 ground and >=1 plain metastable
                # partial has its ground baked to total - Sigma(metastables)
                # wherever the branching is silent (the thermal-placeholder
                # class), in-domain. Lumped (_ZAP) reactions -- where a level is
                # shared by several products so no single plain LFS0 owns the
                # ground -- are excluded, as are reactions lacking a ground or a
                # metastable. All library LFS are the demanded set: bit-exact with
                # the collapse fill when the chain demands the full set (see the
                # docstring caveat for the Sn122 subset case).
                lfs_names = [n for n in src_mt if n.startswith('LFS')]
                fill_ground = False
                if fill and not any('_ZAP' in n for n in lfs_names):
                    levels = {int(src_mt[n].attrs['LFS']) for n in lfs_names}
                    if 0 in levels and any(l > 0 for l in levels):
                        sf = _silence_fill_ground(
                            pathways_fn, pathway_xs_fn, nuc, mt,
                            src_mt['energy'][()], src_mt['xs'][()], levels)
                        fill_ground = sf.fired
                if fill_ground:
                    filled.append(f'{nuc} MT={mt}')

                # Bin every LFS partial (including all-zero rows) so the grouped
                # pathways() list matches the pointwise one exactly.
                part_sum = np.zeros_like(total_g)
                for lfs_name in src_mt:
                    if not lfs_name.startswith('LFS'):
                        continue
                    src_lfs = src_mt[lfs_name]
                    is_ground = int(src_lfs.attrs['LFS']) == 0
                    if fill_ground and is_ground:
                        # Store the baked ground on the same edges; part_sum below
                        # accumulates this STORED value, so a filled channel is
                        # consistent by construction in the silent region.
                        part_g = _group_average(sf.e_dom, sf.ground_dom, edges)
                    else:
                        part_g = _group_average(
                            src_lfs['energy'][()], src_lfs['xs'][()], edges)
                    part_sum += part_g
                    dst_lfs = dst_mt.create_group(lfs_name)
                    for k, v in src_lfs.attrs.items():
                        dst_lfs.attrs[k] = v
                    if fill_ground and is_ground:
                        dst_lfs.attrs['silence_filled'] = np.int64(1)
                    dst_lfs.create_dataset(
                        'xs_g', data=part_g.astype(dtype),
                        compression='gzip', compression_opts=4)
                    rows += 1

                # Consistency check on the float64 binned values.
                worst, g = _partials_total_max_deviation(total_g, part_sum)
                if (part_sum != 0.0).any():
                    if worst > worst_dev:
                        worst_dev = worst
                    if worst > CONSISTENCY_RTOL:
                        n_warnings += 1
                        warn(f'{nuc} MT={mt}: binned MF=10 partials sum to '
                             f'{part_sum[g]:.6e} b but the MF=3 '
                             f'total is {total_g[g]:.6e} b in group {g} (max '
                             f'relative deviation {worst:.3e} > '
                             f'{CONSISTENCY_RTOL:.0e}).')

            # Carry the URR probability tables through verbatim: they are
            # energy-pointwise (the collapse folds them onto groups at runtime),
            # so a recursive copy preserves them bit-for-bit in the grouped file.
            if 'urr' in src_nuc:
                src.copy(src_nuc['urr'], dst_nuc, name='urr')

    wall = time.perf_counter() - t0
    print(f'Wrote {rows} rows to {out}: worst Sum(partials)-vs-total deviation '
          f'{worst_dev:.3e} ({n_warnings} > {CONSISTENCY_RTOL:.0e}), '
          f'{wall:.1f} s.')
    if fill and filled:
        print(f'Silence-filled the ground of {len(filled)} reaction(s): '
              f'{", ".join(filled)}.')
    return {'rows': rows, 'worst_dev': worst_dev, 'n_warnings': n_warnings,
            'filled': filled, 'wall_time': wall}


def _resolve_edges(edges: Path | None, groups: str | None) -> np.ndarray:
    """Resolve group edges from an ``.npy`` path or a named structure."""
    if (edges is None) == (groups is None):
        raise SystemExit('specify exactly one of --edges or --groups')
    if edges is not None:
        return np.load(edges)
    if groups not in GROUP_STRUCTURES:
        raise SystemExit(
            f'unknown group structure {groups!r}; available: '
            f'{", ".join(sorted(GROUP_STRUCTURES))}')
    return np.asarray(GROUP_STRUCTURES[groups], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description='Bin a pointwise PENDF HDF5 library into a grouped PENDF HDF5 library.')
    parser.add_argument('--pendf-in', type=Path,  required=True,      help='Source pointwise PENDF HDF5 file')
    parser.add_argument('--out',      type=Path,  required=True,      help='Output grouped HDF5 file')
    parser.add_argument('--edges',    type=Path,  default=None,       help='Path to an .npy of ascending group edges in eV (primary path)')
    parser.add_argument('--groups',   type=str,   default=None,       help='Named structure resolved via openmc.mgxs.GROUP_STRUCTURES')
    parser.add_argument('--dtype',    type=str,   default='float64',  help="Storage dtype: 'float64' (default) or 'float32'")
    parser.add_argument('--mts',      type=int,   default=None, nargs='+', help='Explicit MT override list (default: chain-relevant MTs)')
    parser.add_argument('--nuclides', type=str,   default=None, nargs='+', help='Subset of nuclides to bin (default: all)')
    parser.add_argument('--no-fill',  action='store_true',            help='Skip the build-time silence-fill bake (source-faithful escape hatch, for debugging)')
    parser.add_argument('--force',    action='store_true',            help='Overwrite --out if it already exists (default: refuse)')
    parser.add_argument('--log-file', type=Path,  default=None,       help='Write a warning-summary log (counts + top-10 partials-vs-total offenders) here')
    args = parser.parse_args()

    edges = _resolve_edges(args.edges, args.groups)

    # Refuse to silently truncate an existing output unless --force is given.
    if args.out.exists() and not args.force:
        parser.error(
            f'output file {args.out} already exists; pass --force to overwrite')

    # Validate any requested nuclide subset against the source up front, so a
    # typo fails with one clear message -- naming every unknown entry -- before
    # any output file is created.
    if args.nuclides is not None:
        with h5py.File(args.pendf_in, 'r') as src:
            available = [k for k in src if isinstance(src[k], h5py.Group)]
        unknown = [n for n in args.nuclides if n not in available]
        if unknown:
            examples = ', '.join(sorted(available)[:5]) or '(none)'
            parser.error(
                f'--nuclides not found in {args.pendf_in}: '
                f'{", ".join(unknown)} (valid examples: {examples})')

    # A bad output path (missing/unwritable directory, permission denied) should
    # fail with a one-line message here, not an h5py traceback from inside the
    # binning loop.
    try:
        with h5py.File(args.out, 'w'):
            pass
    except OSError as exc:
        parser.error(f'cannot open output file {args.out} for writing: {exc}')

    # Capture binning warnings only when a log is requested; otherwise the hook
    # is never installed so default behaviour is unchanged.
    if args.log_file is not None:
        from pendf_warning_log import WarningCapture, write_warning_log
        capture = WarningCapture()
    else:
        capture = None

    with (capture if capture is not None else nullcontext()):
        bin_pendf_library(
            args.pendf_in, args.out, edges, dtype=args.dtype,
            mts=args.mts, nuclides=args.nuclides, fill=not args.no_fill)

    if capture is not None:
        write_warning_log(args.log_file, capture)


if __name__ == '__main__':
    main()
