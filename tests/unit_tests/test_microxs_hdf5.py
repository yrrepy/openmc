"""Unit tests for stacked HDF5 MicroXS write/read infrastructure."""

import h5py
import numpy as np
import pytest

from openmc.deplete import MicroXS
from openmc.deplete.microxs import (
    write_global_microxs_hdf5,
    read_local_microxs_hdf5,
)

# Shared test fixtures
NUCLIDES = ['U235', 'U238', 'Pu239', 'Xe135', 'Gd157']
REACTIONS = ['fission', '(n,gamma)']
N_NUC = len(NUCLIDES)
N_RXN = len(REACTIONS)


def _make_micros(n_mats, seed=42):
    """Create n_mats MicroXS with deterministic but distinct data."""
    rng = np.random.default_rng(seed)
    micros = []
    for _ in range(n_mats):
        data = rng.random((N_NUC, N_RXN, 1))
        micros.append(MicroXS(data, NUCLIDES, REACTIONS))
    return micros


def _mat_ids(n_mats, start=1):
    """Material IDs sorted by int value."""
    return [str(i) for i in range(start, start + n_mats)]


# --- Round-trip tests ---

def test_roundtrip_all(tmp_path):
    """Write 5 MicroXS, read all 5 back."""
    micros = _make_micros(5)
    mat_ids = _mat_ids(5)
    fname = tmp_path / 'microxs.h5'

    write_global_microxs_hdf5(micros, fname, mat_ids)
    result, flux_data = read_local_microxs_hdf5(fname, mat_ids)

    assert len(result) == 5
    assert flux_data is None
    for orig, loaded in zip(micros, result):
        np.testing.assert_array_equal(loaded.data, orig.data)
        assert loaded.nuclides == orig.nuclides
        assert loaded.reactions == orig.reactions


def test_subset_read(tmp_path):
    """Write 10, read only 3 local materials."""
    micros = _make_micros(10)
    mat_ids = _mat_ids(10)
    fname = tmp_path / 'microxs.h5'

    write_global_microxs_hdf5(micros, fname, mat_ids)

    local_ids = ['3', '6', '8']
    result, _ = read_local_microxs_hdf5(fname, local_ids)

    assert len(result) == 3
    for local_id, loaded in zip(local_ids, result):
        orig_idx = mat_ids.index(local_id)
        np.testing.assert_array_equal(loaded.data, micros[orig_idx].data)


def test_ordering_preserved(tmp_path):
    """Read in non-sequential order; verify ordering matches request."""
    micros = _make_micros(6)
    mat_ids = _mat_ids(6)
    fname = tmp_path / 'microxs.h5'

    write_global_microxs_hdf5(micros, fname, mat_ids)

    # Request in reverse-ish order
    local_ids = ['5', '2', '4']
    result, _ = read_local_microxs_hdf5(fname, local_ids)

    assert len(result) == 3
    for local_id, loaded in zip(local_ids, result):
        orig_idx = mat_ids.index(local_id)
        np.testing.assert_array_equal(loaded.data, micros[orig_idx].data)


# --- Flux tests ---

def test_flux_roundtrip(tmp_path):
    """Write with flux tuples (flux, energy_bounds), verify round-trip."""
    micros = _make_micros(3)
    mat_ids = _mat_ids(3)
    energy_bounds = np.array([0.0, 0.625, 1e6, 2e7])
    fluxes = [(np.array([1.0, 2.0, 3.0]), energy_bounds) for _ in range(3)]
    fname = tmp_path / 'microxs.h5'

    write_global_microxs_hdf5(micros, fname, mat_ids, fluxes=fluxes)
    _, flux_data = read_local_microxs_hdf5(fname, mat_ids)

    assert flux_data is not None
    assert len(flux_data) == 3
    for flux_arr, e_bounds in flux_data:
        np.testing.assert_array_equal(flux_arr, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(e_bounds, energy_bounds)


def test_flux_plain_arrays(tmp_path):
    """Write with plain flux arrays (no energy bounds)."""
    micros = _make_micros(3)
    mat_ids = _mat_ids(3)
    fluxes = [np.array([1.0, 2.0]) for _ in range(3)]
    fname = tmp_path / 'microxs.h5'

    write_global_microxs_hdf5(micros, fname, mat_ids, fluxes=fluxes)
    _, flux_data = read_local_microxs_hdf5(fname, mat_ids)

    assert flux_data is not None
    for flux_arr, e_bounds in flux_data:
        np.testing.assert_array_equal(flux_arr, [1.0, 2.0])
        assert e_bounds is None


def test_no_flux(tmp_path):
    """Write without fluxes; read returns None for flux_with_energy."""
    micros = _make_micros(3)
    mat_ids = _mat_ids(3)
    fname = tmp_path / 'microxs.h5'

    write_global_microxs_hdf5(micros, fname, mat_ids)
    _, flux_data = read_local_microxs_hdf5(fname, mat_ids)

    assert flux_data is None


# --- Edge cases ---

def test_empty_local_mat_ids(tmp_path):
    """Rank with no materials gets empty results."""
    micros = _make_micros(3)
    mat_ids = _mat_ids(3)
    fname = tmp_path / 'microxs.h5'

    write_global_microxs_hdf5(micros, fname, mat_ids)
    result, flux_data = read_local_microxs_hdf5(fname, [])

    assert result == []
    assert flux_data is None


def test_missing_material_raises(tmp_path):
    """Requesting a material not in the file raises ValueError."""
    micros = _make_micros(3)
    mat_ids = _mat_ids(3)
    fname = tmp_path / 'microxs.h5'

    write_global_microxs_hdf5(micros, fname, mat_ids)

    with pytest.raises(ValueError, match="999"):
        read_local_microxs_hdf5(fname, ['1', '999'])


# --- Write validation ---

def test_empty_micros_raises():
    with pytest.raises(ValueError, match="No MicroXS"):
        write_global_microxs_hdf5([], 'unused.h5', [])


def test_length_mismatch_raises(tmp_path):
    micros = _make_micros(3)
    with pytest.raises(ValueError, match="Length of micros"):
        write_global_microxs_hdf5(micros, tmp_path / 'x.h5', ['1', '2'])


def test_unsorted_material_ids_raises(tmp_path):
    micros = _make_micros(3)
    with pytest.raises(ValueError, match="sorted"):
        write_global_microxs_hdf5(micros, tmp_path / 'x.h5', ['3', '1', '2'])


def test_inconsistent_shapes_raises(tmp_path):
    m1 = MicroXS(np.zeros((5, 2, 1)), NUCLIDES, REACTIONS)
    m2 = MicroXS(np.zeros((3, 2, 1)), NUCLIDES[:3], REACTIONS)
    with pytest.raises(ValueError, match="shape"):
        write_global_microxs_hdf5([m1, m2], tmp_path / 'x.h5', ['1', '2'])


def test_flux_length_mismatch_raises(tmp_path):
    micros = _make_micros(3)
    fluxes = [np.ones(2) for _ in range(2)]  # 2 != 3
    with pytest.raises(ValueError, match="Length of fluxes"):
        write_global_microxs_hdf5(micros, tmp_path / 'x.h5', _mat_ids(3),
                                  fluxes=fluxes)


# --- HDF5 structure verification ---

def test_hdf5_structure(tmp_path):
    """Verify HDF5 file has expected attributes and datasets."""
    micros = _make_micros(5)
    mat_ids = _mat_ids(5)
    fname = tmp_path / 'microxs.h5'

    write_global_microxs_hdf5(micros, fname, mat_ids)

    with h5py.File(fname, 'r') as f:
        assert f.attrs['version'] == 1
        assert f.attrs['n_materials'] == 5
        assert f.attrs['n_nuclides'] == N_NUC
        assert f.attrs['n_reactions'] == N_RXN
        assert f.attrs['n_groups'] == 1
        assert f['xs_data'].shape == (5, N_NUC, N_RXN, 1)
        assert f['nuclides'].shape == (N_NUC,)
        assert f['reactions'].shape == (N_RXN,)
        assert f['material_ids'].shape == (5,)
        stored_ids = [s.decode() for s in f['material_ids'][:]]
        assert stored_ids == mat_ids


# --- float32 tests ---

def test_float32_roundtrip(tmp_path):
    """Write as float32, read back; verify dtype and values."""
    micros = _make_micros(5)
    mat_ids = _mat_ids(5)
    fname = tmp_path / 'microxs.h5'

    write_global_microxs_hdf5(micros, fname, mat_ids, dtype='float32')
    result, _ = read_local_microxs_hdf5(fname, mat_ids)

    assert result[0].data.dtype == np.float32
    for orig, loaded in zip(micros, result):
        np.testing.assert_allclose(loaded.data, orig.data, rtol=1e-6)


def test_float32_file_structure(tmp_path):
    """Verify HDF5 dataset dtype is float32."""
    micros = _make_micros(3)
    fname = tmp_path / 'microxs.h5'

    write_global_microxs_hdf5(micros, fname, _mat_ids(3), dtype='float32')

    with h5py.File(fname, 'r') as f:
        assert f['xs_data'].dtype == np.float32


def test_dtype_autodetect(tmp_path):
    """Reader auto-detects dtype from HDF5 without explicit parameter."""
    micros = _make_micros(3)
    mat_ids = _mat_ids(3)

    # float32
    f32 = tmp_path / 'f32.h5'
    write_global_microxs_hdf5(micros, f32, mat_ids, dtype='float32')
    result32, _ = read_local_microxs_hdf5(f32, mat_ids)
    assert result32[0].data.dtype == np.float32

    # float64
    f64 = tmp_path / 'f64.h5'
    write_global_microxs_hdf5(micros, f64, mat_ids, dtype='float64')
    result64, _ = read_local_microxs_hdf5(f64, mat_ids)
    assert result64[0].data.dtype == np.float64


# --- Compression tests ---

def test_compression_options(tmp_path):
    """All compression modes produce valid files."""
    # Use sparse data (realistic: most nuclide-reaction pairs are zero)
    # so compression has measurable effect
    n = 100
    rng = np.random.default_rng(42)
    micros = []
    for _ in range(n):
        data = np.zeros((N_NUC, N_RXN, 1))
        data[0, 0, 0] = rng.random()  # only one non-zero entry
        micros.append(MicroXS(data, NUCLIDES, REACTIONS))
    mat_ids = _mat_ids(n)

    for comp, label in [(True, 'lzf'), (False, 'none'), (('gzip', 4), 'gzip')]:
        fname = tmp_path / f'{label}.h5'
        write_global_microxs_hdf5(micros, fname, mat_ids, compression=comp)
        result, _ = read_local_microxs_hdf5(fname, mat_ids)
        for orig, loaded in zip(micros, result):
            np.testing.assert_array_equal(loaded.data, orig.data)

    # Uncompressed should be larger than compressed
    size_none = (tmp_path / 'none.h5').stat().st_size
    size_lzf = (tmp_path / 'lzf.h5').stat().st_size
    assert size_none > size_lzf


def test_bad_compression_raises(tmp_path):
    micros = _make_micros(1)
    with pytest.raises(ValueError, match="compression"):
        write_global_microxs_hdf5(micros, tmp_path / 'x.h5', ['1'],
                                  compression='invalid')
