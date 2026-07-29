"""MicroXS HDF5 I/O module

Rank-independent global write and per-rank local read of :class:`MicroXS`
data, including the optional memory-mapped ``.microxs.npy`` sidecar. Split out
of :mod:`openmc.deplete.microxs`, which re-exports the two public functions.
"""

from __future__ import annotations
from collections.abc import Iterable, Sequence
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING
import warnings

import h5py
import numpy as np

import openmc
from openmc.checkvalue import check_type, PathLike
from .pool import _distribute

if TYPE_CHECKING:
    from .microxs import MicroXS


def _write_flux_data(f, fluxes, dtype='float64', compression=None,
                     compression_opts=None):
    """Write flux arrays and optional energy bounds to an open HDF5 file."""
    energy_bounds = None
    flux_arrays = []
    for item in fluxes:
        if isinstance(item, tuple) and len(item) == 2:
            flux_arrays.append(np.asarray(item[0], dtype=dtype))
            if energy_bounds is None:
                energy_bounds = np.asarray(item[1], dtype='float64')
        elif getattr(item, 'energy_bounds', None) is not None:
            # Flux ndarray subclass carries its own energy_bounds
            flux_arrays.append(np.asarray(item, dtype=dtype))
            if energy_bounds is None:
                energy_bounds = np.asarray(item.energy_bounds, dtype='float64')
        else:
            flux_arrays.append(np.asarray(item, dtype=dtype))

    f.create_dataset('fluxes', data=np.stack(flux_arrays),
                     compression=compression, compression_opts=compression_opts)
    if energy_bounds is not None:
        f.create_dataset('energy_bounds', data=energy_bounds)


def _open_h5_readonly(filename: PathLike) -> h5py.File:
    """Open an HDF5 file read-only with file locking disabled (NFS-safe).

    Many ranks open the same global file concurrently; default HDF5 locking is
    fragile on NFS. Retry without the kwarg on older h5py/HDF5 that lacks it.
    """
    try:
        return h5py.File(filename, 'r', locking=False)
    except (TypeError, ValueError):
        return h5py.File(filename, 'r')


def _read_global_material_count(filename: PathLike) -> int:
    """Number of materials in a global MicroXS HDF5 file (reads only shape)."""
    with _open_h5_readonly(filename) as f:
        return f['material_ids'].shape[0]


# Sidecar identity token: a full-content hash at read time would page in the
# entire memmap and defeat the point of mmap. The threat model is a stale or
# mismatched sidecar (regeneration/reordering), not bit rot, so a small sample
# of rows is enough to catch it while touching only a handful of pages.
def _sidecar_sample_indices(n_rows: int, cap: int = 64) -> list[int]:
    """Deterministic sorted, deduplicated row-index sample for the sidecar hash."""
    if n_rows <= 0:
        return []
    if n_rows <= cap:
        return list(range(n_rows))
    idx = {0, n_rows // 2, n_rows - 1}
    stride = n_rows / (cap - len(idx))
    idx.update(int(i * stride) for i in range(cap - len(idx)))
    return sorted(idx)


def _sidecar_sample_hash(arr: np.ndarray, indices: Sequence[int]) -> str:
    """SHA-256 hex over the given rows of ``arr``, concatenated in index order."""
    h = hashlib.sha256()
    for i in indices:
        h.update(np.ascontiguousarray(arr[i]).tobytes())
    return h.hexdigest()


def _validate_sidecar(f: h5py.File, h5_path: PathLike, sidecar_path: Path,
                      memmap: np.ndarray) -> None:
    """Verify a memmapped sidecar against the identity token in HDF5 file ``f``."""
    if 'sidecar_format' not in f.attrs:
        warnings.warn(
            f"sidecar unverified: {h5_path} carries no identity token "
            f"(written by older OpenMC), so {sidecar_path} cannot be checked "
            f"for staleness; proceeding.")
        return

    exp_shape = tuple(int(x) for x in f.attrs['sidecar_shape'])
    exp_dtype = f.attrs['sidecar_dtype']
    if isinstance(exp_dtype, bytes):
        exp_dtype = exp_dtype.decode()
    exp_hash = f.attrs['sidecar_sha256']
    if isinstance(exp_hash, bytes):
        exp_hash = exp_hash.decode()

    remedy = ("Regenerate the sidecar, or rewrite with "
              "write_global_microxs_hdf5(..., write_sidecar=True).")
    # Header check reads only the .npy header, not the array data.
    if memmap.shape != exp_shape or str(memmap.dtype) != exp_dtype:
        raise ValueError(
            f"Sidecar {sidecar_path} does not match {h5_path}: shape/dtype "
            f"{memmap.shape}/{memmap.dtype} != expected {exp_shape}/{exp_dtype}. "
            f"{remedy}")
    # Sampled-row hash touches only the sampled pages of the memmap.
    sample = _sidecar_sample_indices(memmap.shape[0])
    if _sidecar_sample_hash(memmap, sample) != exp_hash:
        raise ValueError(
            f"Sidecar {sidecar_path} is stale relative to {h5_path} "
            f"(row-sample hash mismatch). {remedy}")


def write_global_microxs_hdf5(
    micros: Sequence[MicroXS],
    filename: PathLike,
    material_ids: Sequence[str],
    fluxes: Sequence | None = None,
    dtype: str = 'float64',
    compression: bool | tuple = True,
    write_sidecar: bool = False,
) -> None:
    """Write all MicroXS to a single rank-independent HDF5 file.

    Stores cross section data for every burnable material in a stacked 4D
    dataset that can be efficiently subset-read by individual MPI ranks
    using :func:`read_local_microxs_hdf5`.
    This is meant to reduce RAM usage by each individual rank and enable greater MPI scaling.
    
    .. versionadded:: 0.15.4

    Parameters
    ----------
    micros : list of MicroXS
        MicroXS objects, one per burnable material, ordered by
        ``sorted(material_ids, key=int)``.
    filename : path-like
        Output HDF5 file path.
    material_ids : list of str
        Material ID strings in the same order as ``micros``. Must be sorted
        by ``int()`` value.
    fluxes : list, optional
        Flux data for each material. Each element is either a 1D numpy
        array or a ``(flux_array, energy_bounds)`` tuple. MG-flux is needed for isomeric branching
    dtype : str, optional
        NumPy dtype for cross section and flux data. Default ``'float64'``.
        Use ``'float32'`` to halve file size and per-rank RAM.
    compression : bool or tuple, optional
        HDF5 compression. Default ``True`` uses lzf. ``False`` disables
        compression. A tuple ``('gzip', level)`` uses gzip.
    write_sidecar : bool, optional
        Write a ``.microxs.npy`` sidecar file for memory-mapped reading.
        Default ``False``. The sidecar enables ``mmap=True`` in
        :func:`read_local_microxs_hdf5` for zero-copy XS access.

    See Also
    --------
    read_local_microxs_hdf5 : Read local material slices from this file.

    """
    if len(micros) == 0:
        raise ValueError("No MicroXS objects to write.")
    if len(micros) != len(material_ids):
        raise ValueError(
            f"Length of micros ({len(micros)}) != length of "
            f"material_ids ({len(material_ids)})")
    if fluxes is not None and len(fluxes) != len(micros):
        raise ValueError(
            f"Length of fluxes ({len(fluxes)}) != length of "
            f"micros ({len(micros)})")

    int_ids = [int(mid) for mid in material_ids]
    if int_ids != sorted(int_ids):
        raise ValueError("material_ids must be sorted by int() value")

    ref_shape = micros[0].data.shape
    for i, m in enumerate(micros):
        if m.data.shape != ref_shape:
            raise ValueError(
                f"MicroXS[{i}] shape {m.data.shape} != "
                f"MicroXS[0] shape {ref_shape}")

    # Resolve compression settings
    if compression is False:
        comp, comp_opts = None, None
    elif compression is True:
        comp, comp_opts = 'lzf', None
    elif isinstance(compression, tuple):
        comp, comp_opts = compression
    else:
        raise ValueError(
            f"compression must be True, False, or a tuple like "
            f"('gzip', 4), got {compression!r}")

    n_mats = len(micros)
    n_nuc, n_rxn, n_grp = ref_shape
    bytes_per_elem = np.dtype(dtype).itemsize
    target_chunk_bytes = 32 * 1024 * 1024
    row_bytes = n_nuc * n_rxn * n_grp * bytes_per_elem
    chunk_mats = max(1, min(n_mats, 256, target_chunk_bytes // row_bytes))

    with h5py.File(filename, 'w') as f:
        f.attrs['version'] = 1
        f.attrs['n_materials'] = n_mats
        f.attrs['n_nuclides'] = n_nuc
        f.attrs['n_reactions'] = n_rxn
        f.attrs['n_groups'] = n_grp

        stacked = np.stack([m.data for m in micros]).astype(dtype)
        f.create_dataset(
            'xs_data',
            data=stacked,
            chunks=(chunk_mats, n_nuc, n_rxn, n_grp),
            compression=comp,
            compression_opts=comp_opts,
        )

        f.create_dataset(
            'nuclides', data=np.array(micros[0].nuclides, dtype='S'))
        f.create_dataset(
            'reactions', data=np.array(micros[0].reactions, dtype='S'))
        f.create_dataset(
            'material_ids', data=np.array(material_ids, dtype='S'))

        if fluxes is not None:
            _write_flux_data(f, fluxes, dtype=dtype,
                             compression=comp, compression_opts=comp_opts)

        # Stamp an identity token so a mmap read can detect a stale sidecar
        # (see _sidecar_sample_indices for the sampling rationale).
        if write_sidecar:
            sample = _sidecar_sample_indices(stacked.shape[0])
            f.attrs['sidecar_format'] = 1
            f.attrs['sidecar_shape'] = np.asarray(stacked.shape, dtype='int64')
            f.attrs['sidecar_dtype'] = str(stacked.dtype)
            f.attrs['sidecar_sha256'] = _sidecar_sample_hash(stacked, sample)

    if write_sidecar:
        sidecar_path = Path(filename).with_suffix('.microxs.npy')
        np.save(sidecar_path, stacked)


def read_local_microxs_hdf5(
    filename: PathLike,
    local_mat_ids: Sequence[str],
    mmap: bool = False,
) -> tuple[list[MicroXS], list[tuple] | None, np.ndarray | None]:
    """Read local material slices from a global MicroXS HDF5 file.

    Reads only the rows corresponding to ``local_mat_ids`` from the stacked
    dataset, using h5py fancy indexing for efficient I/O.
    This is meant to reduce RAM usage by each individual rank and enable greater MPI scaling.

    .. versionadded:: 0.15.4

    Parameters
    ----------
    filename : path-like
        Path to HDF5 file written by :func:`write_global_microxs_hdf5`.
    local_mat_ids : list of str
        Material IDs for this MPI rank.
    mmap : bool, optional
        Use memory-mapped sidecar file instead of reading into heap.
        Requires a ``.microxs.npy`` sidecar written by
        :func:`write_global_microxs_hdf5` with ``write_sidecar=True``. The
        sidecar is validated against an identity token in the HDF5 file; a
        stale or mismatched sidecar raises ``ValueError``. Default ``False``.

    Returns
    -------
    micros : list of MicroXS
        MicroXS objects in the same order as ``local_mat_ids``.
    flux_with_energy : list of tuple or None
        If flux data exists in the file, a list of
        ``(flux_array, energy_bounds)`` tuples. Returns ``None`` if no
        flux data in file.
    mmap_ref : numpy.ndarray or None
        Reference to the memory-mapped array when ``mmap=True``, to
        prevent garbage collection. ``None`` otherwise.

    See Also
    --------
    write_global_microxs_hdf5 : Write the global file.

    """
    # Imported here: microxs re-exports this module, so a module-level import
    # would cycle.
    from .microxs import MicroXS

    if len(local_mat_ids) == 0:
        return [], None, None

    with _open_h5_readonly(filename) as f:
        version = f.attrs.get('version', None)
        if version != 1:
            raise ValueError(
                f"Unsupported MicroXS HDF5 version: {version}. "
                f"Expected version 1.")

        all_mat_ids = [s.decode() for s in f['material_ids'][:]]
        nuclides = [s.decode() for s in f['nuclides'][:]]
        reactions = [s.decode() for s in f['reactions'][:]]

        mat_id_to_row = {mid: i for i, mid in enumerate(all_mat_ids)}
        local_rows = []
        for mid in local_mat_ids:
            if mid not in mat_id_to_row:
                sample = all_mat_ids[:5]
                raise ValueError(
                    f"Material '{mid}' not found in {filename}. "
                    f"File contains {len(all_mat_ids)} materials: "
                    f"{sample}{'...' if len(all_mat_ids) > 5 else ''}")
            local_rows.append(mat_id_to_row[mid])

        # h5py requires fancy indices to be strictly sorted ascending
        sorted_rows = sorted(local_rows)
        sort_map = {row: pos for pos, row in enumerate(sorted_rows)}
        reorder = [sort_map[r] for r in local_rows]

        # Build MicroXS from mmap sidecar or HDF5 data
        mmap_ref = None
        if mmap:
            sidecar_path = Path(filename).with_suffix('.microxs.npy')
            if not sidecar_path.exists():
                raise FileNotFoundError(
                    f"Sidecar file {sidecar_path} not found. Re-run "
                    f"write_global_microxs_hdf5 with write_sidecar=True.")
            global_xs = np.load(str(sidecar_path), mmap_mode='r')
            _validate_sidecar(f, filename, sidecar_path, global_xs)
            micros = [MicroXS(global_xs[row], nuclides, reactions)
                      for row in local_rows]
            mmap_ref = global_xs
        else:
            xs_block = f['xs_data'][sorted_rows]
            xs_block = xs_block[reorder]
            micros = [MicroXS(xs_block[i], nuclides, reactions)
                      for i in range(len(local_mat_ids))]

        flux_with_energy = None
        if 'fluxes' in f:
            energy_bounds = None
            if 'energy_bounds' in f:
                energy_bounds = f['energy_bounds'][:]

            flux_block = f['fluxes'][sorted_rows]
            flux_block = flux_block[reorder]
            flux_with_energy = [
                (flux_block[i], energy_bounds)
                for i in range(len(local_mat_ids))
            ]

    return micros, flux_with_energy, mmap_ref


def _load_local_microxs(materials, microxs_file, mmap=False):
    """Distribute materials over the MPI ranks and read this rank's slices.

    Returns ``(local_materials, local_fluxes, local_micros, metadata,
    mmap_ref)``, where ``metadata`` is the global depletion metadata gathered
    before the material list is filtered down to the local materials.
    """
    check_type('materials', materials, Iterable, openmc.Material)
    materials_obj = openmc.Materials(materials)

    # Pre-compute global metadata before filtering materials
    all_depletable = sorted(
        [m for m in materials_obj if m.depletable],
        key=lambda m: int(m.id))
    burnable_mats = [str(m.id) for m in all_depletable]
    if not burnable_mats:
        raise RuntimeError(
            "No depletable materials were found in the model.")
    volume = {}
    heavy_metal = 0.0
    for m in all_depletable:
        if m.volume is None:
            name_str = f" Name={m.name}" if m.name else ""
            raise RuntimeError(
                f"Volume not specified for depletable material "
                f"with ID={m.id}{name_str}.")
        volume[str(m.id)] = m.volume
        heavy_metal += m.fissionable_mass
    name_list = [m.name for m in all_depletable]

    local_mats = _distribute(burnable_mats)

    local_micros, local_flux_with_energy, mmap_ref = read_local_microxs_hdf5(
        microxs_file, local_mats, mmap=mmap)

    # Build fluxes list from HDF5 data or default to unit flux.
    # Pass full (flux, energy_bounds) tuples so __init__ populates
    # _flux_with_energy for isomeric branching.
    if local_flux_with_energy is not None:
        local_fluxes = list(local_flux_with_energy)
    else:
        n_groups = local_micros[0].data.shape[2] if local_micros else 1
        local_fluxes = [np.ones(n_groups) for _ in local_mats]

    # Filter materials to local-only before passing to __init__
    local_set = set(local_mats)
    local_materials = openmc.Materials(
        [m for m in materials_obj if str(m.id) in local_set])

    metadata = {
        'heavy_metal': heavy_metal,
        'burnable_mats': burnable_mats,
        'volume': volume,
        'name_list': name_list,
    }

    return local_materials, local_fluxes, local_micros, metadata, mmap_ref
