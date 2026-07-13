#!/usr/bin/env python
"""Scan PENDF HDF5 libraries for negative cross-section values.

A one-shot, read-only diagnostic that walks every cross-section-valued dataset
in one or more PENDF HDF5 files -- MF=3 reaction totals and MF=10 isomeric
production partials alike, pointwise and grouped variants -- and reports any
dataset carrying a value below zero. A clean library should have none; a
negative cross section signals a corrupt tape, a reconstruction artefact, or a
group-averaging error, and gates a build.

The schema is taken from the library writers, not guessed:

* Pointwise (``openmc/data/pendf.py``): ``/<Nuclide>/MT<mt>/{energy,xs}`` for the
  MF=3 total and ``/<Nuclide>/MT<mt>/LFS<l>[_ZAP<izap>]/{energy,xs}`` for each
  MF=10 partial (the ``LFS<l>`` naming rule is documented at pendf.py:62-64,
  written at pendf.py:404-411; totals at pendf.py:114-116). MF=2 URR probability
  tables live under ``/<Nuclide>/urr/`` (pendf.py:526-532) and are SKIPPED --
  they are not smooth cross sections.
* Grouped (``tools/pendf_group_bin.py``): the same tree with a single ``xs_g``
  group cross section per reaction/partial and no per-reaction energy grid
  (group_bin.py:8-16, 136-155); the shared ascending edges are one root dataset
  ``/group_edges``.

An XS-valued dataset is therefore exactly any dataset whose basename is ``xs``
(pointwise) or ``xs_g`` (grouped); ``energy``, ``table``, ``group_edges`` and the
whole ``urr`` subtree are excluded by name. Datasets are streamed one at a time,
so memory stays bounded regardless of library size. Exit status is 0 when every
scanned dataset is clean and 1 when any negative value is found, so the probe can
gate a build.
"""

from __future__ import annotations

import argparse
import sys
from collections import namedtuple
from pathlib import Path

import h5py
import numpy as np

# Dataset basenames that hold cross-section values: ``xs`` in pointwise
# libraries, ``xs_g`` in grouped ones. Every other leaf (``energy``, ``table``,
# ``group_edges``) is an axis or metadata and is skipped.
XS_NAMES = ('xs', 'xs_g')

# Cap on offender rows echoed to the terminal; the full set always goes to --out.
MAX_STDOUT_ROWS = 100

# One flagged dataset. Energies are in eV; ``gidx_min`` is the group index of the
# most-negative point for grouped files (-1 for pointwise), and the energy fields
# are NaN for a grouped file that carries no ``/group_edges``.
Offender = namedtuple(
    'Offender',
    'file nuclide dataset grid n_neg n_points min_value '
    'gidx_min energy_min energy_lo energy_hi')


def _iter_xs_datasets(group, prefix=''):
    """Yield ``(relpath, dataset)`` for every XS-valued dataset under a nuclide.

    Recurses the reaction tree, pruning any ``urr`` subgroup (MF=2 probability
    tables, out of scope). A dataset qualifies when its basename is ``xs`` or
    ``xs_g``; ``energy`` axes and other leaves are ignored.

    Parameters
    ----------
    group : h5py.Group
        A nuclide group (or a reaction subgroup, during recursion).
    prefix : str
        Path accumulated so far, relative to the nuclide group.

    Yields
    ------
    tuple of (str, h5py.Dataset)
        The dataset path within the nuclide (e.g. ``MT102/xs`` or
        ``MT105/LFS1/xs``) and the dataset itself.
    """
    for key in group:
        item = group[key]
        path = prefix + key
        if isinstance(item, h5py.Group):
            if key == 'urr':
                continue
            yield from _iter_xs_datasets(item, path + '/')
        elif key in XS_NAMES:
            yield path, item


def _scan_dataset(fname, nuclide, path, dset, edges):
    """Return an :class:`Offender` for ``dset`` if it holds negatives, else None.

    The cross-section array is read once. Negatives are located only when
    present, and the sibling ``energy`` grid (pointwise) is read only then, so a
    clean dataset costs a single array read.

    Parameters
    ----------
    fname : str
        Library file name (basename), recorded on the offender.
    nuclide, path : str
        Nuclide name and the dataset's path within it.
    dset : h5py.Dataset
        The ``xs`` or ``xs_g`` dataset.
    edges : numpy.ndarray or None
        The grouped library's ``/group_edges`` (length G+1), or None for a
        pointwise library.
    """
    xs = dset[()]
    if xs.size == 0:
        return None
    neg = np.flatnonzero(xs < 0.0)
    if neg.size == 0:
        return None

    worst = neg[np.argmin(xs[neg])]
    grouped = dset.name.rsplit('/', 1)[-1] == 'xs_g'
    if grouped:
        if edges is not None:
            energy_min = float(edges[worst])
            energy_lo = float(edges[neg.min()])
            energy_hi = float(edges[neg.max() + 1])
        else:
            energy_min = energy_lo = energy_hi = float('nan')
        gidx_min = int(worst)
    else:
        energy = dset.parent['energy'][()]
        energy_min = float(energy[worst])
        energy_lo = float(energy[neg].min())
        energy_hi = float(energy[neg].max())
        gidx_min = -1

    return Offender(
        file=fname, nuclide=nuclide, dataset=path,
        grid='grouped' if grouped else 'pointwise',
        n_neg=int(neg.size), n_points=int(xs.size),
        min_value=float(xs[worst]), gidx_min=gidx_min,
        energy_min=energy_min, energy_lo=energy_lo, energy_hi=energy_hi)


def scan_file(path):
    """Scan one PENDF HDF5 library for negative cross sections.

    Parameters
    ----------
    path : pathlib.Path
        The ``.h5`` library to scan.

    Returns
    -------
    tuple
        ``(n_datasets, n_nuclides, offenders, error)`` -- datasets and nuclides
        scanned, the list of :class:`Offender` records, and an error string (or
        None) if the file could not be opened.
    """
    fname = path.name
    offenders = []
    n_datasets = 0
    try:
        with h5py.File(path, 'r') as f:
            edges = f['group_edges'][()] if 'group_edges' in f else None
            nuclides = [k for k in f if isinstance(f[k], h5py.Group)]
            for nuclide in nuclides:
                for rel, dset in _iter_xs_datasets(f[nuclide]):
                    n_datasets += 1
                    hit = _scan_dataset(fname, nuclide, rel, dset, edges)
                    if hit is not None:
                        offenders.append(hit)
    except OSError as exc:
        return 0, 0, [], str(exc)
    return n_datasets, len(nuclides), offenders, None


def _discover(paths):
    """Expand paths and directories into a de-duplicated list of ``.h5`` files."""
    files, seen = [], set()
    for raw in paths:
        p = Path(raw)
        found = sorted(p.glob('*.h5')) if p.is_dir() else [p]
        for f in found:
            key = f.resolve()
            if key not in seen:
                seen.add(key)
                files.append(f)
    return files


def _fmt_location(off):
    """Human-readable locator of an offender's worst point for the terminal."""
    if off.grid == 'grouped':
        band = ('' if np.isnan(off.energy_min)
                else f' ({off.energy_min:.3e} eV)')
        span = ('' if np.isnan(off.energy_lo)
                else f'  neg in [{off.energy_lo:.3e},{off.energy_hi:.3e}] eV')
        return f'g={off.gidx_min}{band}{span}'
    return (f'E={off.energy_min:.3e} eV  '
            f'neg in [{off.energy_lo:.3e},{off.energy_hi:.3e}] eV')


def _print_offenders(offenders):
    """Print the offender table to stdout, grouped by file."""
    print('\nOFFENDERS')
    print('=' * 9)
    shown = 0
    current = None
    for off in offenders:
        if off.file != current:
            current = off.file
            n = sum(1 for o in offenders if o.file == current)
            print(f'\n{current}  ({n} offender dataset(s))')
        if shown >= MAX_STDOUT_ROWS:
            continue
        print(f'  {off.nuclide:<12} {off.dataset:<20} {off.grid:<10} '
              f'{off.n_neg:>6}/{off.n_points:<7} '
              f'min={off.min_value:.4e} b   {_fmt_location(off)}')
        shown += 1
    if len(offenders) > MAX_STDOUT_ROWS:
        print(f'\n  ... {len(offenders) - MAX_STDOUT_ROWS} more offender row(s) '
              f'not shown (use --out for the full list).')


def _write_tsv(out, offenders):
    """Write all offenders to a tab-separated report."""
    cols = ('file', 'nuclide', 'dataset', 'grid', 'n_neg', 'n_points',
            'min_value', 'gidx_min', 'energy_min_eV', 'energy_neg_lo_eV',
            'energy_neg_hi_eV')
    with open(out, 'w') as fh:
        fh.write('\t'.join(cols) + '\n')
        for o in offenders:
            fh.write('\t'.join(str(v) for v in (
                o.file, o.nuclide, o.dataset, o.grid, o.n_neg, o.n_points,
                repr(o.min_value), o.gidx_min, repr(o.energy_min),
                repr(o.energy_lo), repr(o.energy_hi))) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Scan PENDF HDF5 libraries for negative cross-section values.')
    parser.add_argument('libraries', type=Path, nargs='+',    help='.h5 files and/or directories (globbed for *.h5)')
    parser.add_argument('--out',     type=Path, default=None,  help='Optional TSV report path (default: stdout summary only)')
    parser.add_argument('--quiet',   action='store_true',      help='Suppress the per-file progress line')
    args = parser.parse_args()

    files = _discover(args.libraries)
    if not files:
        parser.error(f'no .h5 files found in: {", ".join(map(str, args.libraries))}')

    offenders, total_datasets, unreadable = [], 0, []
    for i, path in enumerate(files, 1):
        n_datasets, n_nuclides, hits, error = scan_file(path)
        total_datasets += n_datasets
        if error is not None:
            unreadable.append((path.name, error))
            if not args.quiet:
                print(f'[{i}/{len(files)}] {path.name}: ERROR {error}')
            continue
        offenders.extend(hits)
        if not args.quiet:
            verdict = f'{len(hits)} offender(s)' if hits else 'clean'
            print(f'[{i}/{len(files)}] {path.name}: '
                  f'{n_datasets} datasets, {n_nuclides} nuclides -> {verdict}')

    if offenders:
        offenders.sort(key=lambda o: (o.file, o.nuclide, o.dataset))
        _print_offenders(offenders)

    if args.out is not None:
        _write_tsv(args.out, offenders)
        print(f'\nWrote {len(offenders)} offender row(s) to {args.out}')

    n_files = len(files) - len(unreadable)
    print()
    if offenders:
        n_neg = sum(o.n_neg for o in offenders)
        files_hit = len({o.file for o in offenders})
        print(f'FOUND {len(offenders)} offender dataset(s) in {files_hit} file(s) '
              f'({n_neg} negative points) across {total_datasets} datasets '
              f'in {n_files} file(s).')
    else:
        print(f'CLEAN -- no negative values in {total_datasets} datasets '
              f'across {n_files} file(s).')
    if unreadable:
        print(f'WARNING: {len(unreadable)} file(s) could not be read: '
              f'{", ".join(name for name, _ in unreadable)}')

    sys.exit(1 if offenders else 0)


if __name__ == '__main__':
    main()
