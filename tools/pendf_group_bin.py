#!/usr/bin/env python
"""Pre-bin a pointwise PENDF HDF5 library into a grouped PENDF HDF5 library.

Flat-weights every chain-relevant MF=3 total and MF=10 partial cross section
onto a fixed energy group structure with
:func:`openmc.deplete.microxs._group_average` and stores the results dense
(gzip-compressed) so the depletion collapse can skip runtime binning. The
output schema is ``format='pendf-grouped'`` version 1:

* root attrs: ``format``, ``version``, ``source`` (abs path of the pointwise
  file), ``dtype``, plus the source's ``library``/``temperature`` attrs;
* ``/group_edges`` float64[G+1] ascending eV;
* ``/<Nuclide>/`` copies the source nuclide attrs verbatim;
* ``/<Nuclide>/MT<mt>/xs_g`` the MF=3 group cross section (MT attrs copied);
* ``/<Nuclide>/MT<mt>/LFS<l>/xs_g`` each MF=10 partial (LFS attrs copied).

Every chain-relevant MT present in the source is written in full -- the total
and *every* LFS partial, including all-zero rows -- so a grouped collapse
reproduces the pointwise ``pathways()`` list (and hence a bit-exact table).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from warnings import warn

import h5py
import numpy as np

from openmc.deplete.chain import REACTIONS
from openmc.deplete.microxs import _group_average
from openmc.mgxs import GROUP_STRUCTURES

# Grouped-schema constants.
GROUPED_FORMAT = 'pendf-grouped'
GROUPED_VERSION = 1

# Warn when binned partials disagree with the binned total by more than this
# relative amount in any nonzero group -- the same threshold and spirit as the
# runtime check in openmc.deplete.microxs._build_xs_table_pendf.
CONSISTENCY_RTOL = 1e-5


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

    Returns
    -------
    dict
        Summary statistics: ``rows`` (xs_g datasets written), ``worst_dev``
        (worst Σpartials-vs-total relative deviation), ``n_warnings``, and
        ``wall_time`` in seconds.
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

    with h5py.File(pendf_in, 'r') as src, h5py.File(out, 'w') as dst:
        # Root attrs: schema identity plus provenance carried from the source.
        dst.attrs['format'] = GROUPED_FORMAT
        dst.attrs['version'] = GROUPED_VERSION
        dst.attrs['source'] = str(Path(pendf_in).resolve())
        dst.attrs['dtype'] = dtype
        for key in ('library', 'temperature'):
            if key in src.attrs:
                dst.attrs[key] = src.attrs[key]
        dst.create_dataset('group_edges', data=edges)

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

                # Bin every LFS partial (including all-zero rows) so the grouped
                # pathways() list matches the pointwise one exactly.
                part_sum = np.zeros_like(total_g)
                for lfs_name in src_mt:
                    if not lfs_name.startswith('LFS'):
                        continue
                    src_lfs = src_mt[lfs_name]
                    part_g = _group_average(
                        src_lfs['energy'][()], src_lfs['xs'][()], edges)
                    part_sum += part_g
                    dst_lfs = dst_mt.create_group(lfs_name)
                    for k, v in src_lfs.attrs.items():
                        dst_lfs.attrs[k] = v
                    dst_lfs.create_dataset(
                        'xs_g', data=part_g.astype(dtype),
                        compression='gzip', compression_opts=4)
                    rows += 1

                # Consistency check on the float64 binned values.
                nz = total_g != 0.0
                if nz.any() and (part_sum != 0.0).any():
                    dev = np.abs(part_sum[nz] - total_g[nz]) / np.abs(total_g[nz])
                    worst = float(dev.max())
                    if worst > worst_dev:
                        worst_dev = worst
                    if worst > CONSISTENCY_RTOL:
                        n_warnings += 1
                        g = int(np.nonzero(nz)[0][dev.argmax()])
                        warn(f'{nuc} MT={mt}: binned MF=10 partials sum to '
                             f'{part_sum[nz][dev.argmax()]:.6e} b but the MF=3 '
                             f'total is {total_g[g]:.6e} b in group {g} (max '
                             f'relative deviation {worst:.3e} > '
                             f'{CONSISTENCY_RTOL:.0e}).')

    wall = time.perf_counter() - t0
    print(f'Wrote {rows} rows to {out}: worst Sum(partials)-vs-total deviation '
          f'{worst_dev:.3e} ({n_warnings} > {CONSISTENCY_RTOL:.0e}), '
          f'{wall:.1f} s.')
    return {'rows': rows, 'worst_dev': worst_dev,
            'n_warnings': n_warnings, 'wall_time': wall}


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
    args = parser.parse_args()

    edges = _resolve_edges(args.edges, args.groups)
    bin_pendf_library(
        args.pendf_in, args.out, edges, dtype=args.dtype,
        mts=args.mts, nuclides=args.nuclides)


if __name__ == '__main__':
    main()
