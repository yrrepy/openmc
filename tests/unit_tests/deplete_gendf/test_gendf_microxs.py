"""MicroXS / Flux unit tests for the GENDF workflow.

Consolidates four legacy modules into one, organized by section:

* Flux return type - the ``Flux`` ndarray subclass returned by
  ``get_microxs_and_flux`` (ndarray mechanics, attribute propagation, pickle
  round-trips, IndependentOperator ingestion of the three flux forms).
* Collapse helper - ``GENDFFluxCollapseHelper`` (gendf-flux reaction-rate mode).
* Nuclide filtering - ``MicroXS.from_multigroup_flux_with_gendf`` filtering
  nuclides to those with GENDF data, matching the CE HDF5 workflow.
* HDF5 round-trip - stacked HDF5 MicroXS write/read infrastructure.
"""

import pickle
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import h5py
import numpy as np
import pytest

from openmc import Material
from openmc.deplete import Flux, IndependentOperator, MicroXS
from openmc.deplete.helpers import GENDFFluxCollapseHelper
from openmc.deplete.microxs import (
    write_global_microxs_hdf5,
    read_local_microxs_hdf5,
)
from openmc.mgxs import GROUP_STRUCTURES

from .gendf_testing import MockGENDFLibrary


# ===========================================================================
# Flux return type
# ===========================================================================
#
# Flux is a backward-compatible numpy.ndarray subclass that also carries the
# energy group boundaries in an ``energy_bounds`` attribute.

# chain_simple.xml / micro_xs_simple.csv live in tests/ (two levels up from
# this file's tests/unit_tests/deplete_gendf/ location).
CHAIN_PATH = Path(__file__).parents[2] / "chain_simple.xml"
ONE_GROUP_XS = Path(__file__).parents[2] / "micro_xs_simple.csv"


# --- Flux ndarray mechanics ---

def test_flux_is_ndarray():
    """Flux behaves like a plain 1D ndarray."""
    eb = np.array([0.0, 1.0, 2.0, 20.0e6])
    f = Flux([1.0, 2.0, 3.0], energy_bounds=eb)
    assert isinstance(f, np.ndarray)
    assert f.sum() == pytest.approx(6.0)
    assert f[1] == pytest.approx(2.0)
    np.testing.assert_array_equal(f * 2, [2.0, 4.0, 6.0])
    np.testing.assert_array_equal(f.energy_bounds, eb)


def test_flux_default_energy_bounds_none():
    """Without energy_bounds the attribute is None."""
    f = Flux([1.0, 2.0])
    assert f.energy_bounds is None


def test_flux_slice_keeps_energy_bounds():
    """__array_finalize__ propagates energy_bounds through views/slices."""
    eb = np.array([0.0, 1.0, 2.0, 3.0])
    f = Flux([1.0, 2.0, 3.0], energy_bounds=eb)
    sl = f[1:]
    assert isinstance(sl, Flux)
    np.testing.assert_array_equal(sl.energy_bounds, eb)
    # Arithmetic results keep the attribute too
    assert (f + 1).energy_bounds is not None


def test_flux_pickle_roundtrip():
    """Pickle preserves both data and energy_bounds (needed for MPI scatter)."""
    eb = np.array([0.0, 1.0, 2.0, 3.0])
    f = Flux([4.0, 5.0, 6.0], energy_bounds=eb)
    g = pickle.loads(pickle.dumps(f))
    assert isinstance(g, Flux)
    np.testing.assert_array_equal(g, f)
    np.testing.assert_array_equal(g.energy_bounds, eb)


def test_flux_pickle_none_energy_bounds():
    """energy_bounds=None survives a pickle round-trip."""
    g = pickle.loads(pickle.dumps(Flux([1.0, 2.0])))
    assert g.energy_bounds is None


# --- IndependentOperator ingestion of the three flux forms ---

def _build_operator(flux_item):
    """Construct a 1-material IndependentOperator with a single flux item."""
    fuel = Material(name="uo2")
    fuel.add_element("U", 1, percent_type="ao", enrichment=4.25)
    fuel.add_element("O", 2)
    fuel.set_density("g/cc", 10.4)
    fuel.depletable = True
    fuel.volume = 1.0
    micro_xs = MicroXS.from_csv(ONE_GROUP_XS)
    return IndependentOperator([fuel], [flux_item], [micro_xs], CHAIN_PATH)


def test_operator_ingests_flux_forms():
    """Flux, (flux, energy_bounds) tuple, and bare-array inputs normalize."""
    arr = np.array([1.0, 2.0, 3.0])
    eb = np.array([0.0, 1.0, 2.0, 20.0e6])

    op_flux = _build_operator(Flux(arr, energy_bounds=eb))
    op_tuple = _build_operator((arr, eb))
    op_bare = _build_operator(arr)

    # Flux and tuple forms both recover the flux array and energy bounds
    for op in (op_flux, op_tuple):
        stored_flux, stored_eb = op._flux_with_energy[0]
        np.testing.assert_array_equal(stored_flux, arr)
        np.testing.assert_array_equal(stored_eb, eb)
        np.testing.assert_array_equal(op._energy_bins, eb)

    # Flux is unwrapped to a plain ndarray on storage
    assert type(op_flux._flux_with_energy[0][0]) is np.ndarray

    # Bare array carries no energy information
    stored_flux, stored_eb = op_bare._flux_with_energy[0]
    np.testing.assert_array_equal(stored_flux, arr)
    assert stored_eb is None
    assert op_bare._energy_bins is None


# ===========================================================================
# Collapse helper (GENDFFluxCollapseHelper)
# ===========================================================================
#
# The helper is exercised without an initialized openmc.lib session by setting
# the state generate_tallies would normally create directly.

ENERGIES = GROUP_STRUCTURES['CCFE-709']
NG = len(ENERGIES) - 1


def _make_helper(gendf, nuclides, scores, mts, fluxes, reactions=()):
    """Build a helper with generate_tallies state injected manually."""
    n_mats = fluxes.shape[0]
    helper = GENDFFluxCollapseHelper(
        len(nuclides), len(scores), gendf, reactions=list(reactions))
    helper._materials = [Mock() for _ in range(n_mats)]
    helper._scores = list(scores)
    helper._mts = list(mts)
    helper._nuclides = list(nuclides)
    helper._flux_tally_means_cache = fluxes.reshape(-1, 1)
    return helper


def test_collapse_matches_hand_calculation():
    """Rates must equal the hand-computed sigma_g . phi_g dot product."""
    rng = np.random.default_rng(42)
    xs_ng = rng.random(NG)
    xs_n2n = rng.random(NG)
    xs_nnp = rng.random(NG)
    gendf = MockGENDFLibrary({
        'Al27': {102: xs_ng, 4: xs_nnp},
        'Fe56': {102: 3 * xs_ng, 16: xs_n2n},
    })
    flux = rng.random((2, NG))

    helper = _make_helper(
        gendf, ['Al27', 'Fe56'], ['(n,gamma)', "(n,n')", '(n,2n)'],
        [102, 4, 16], flux)

    rates = helper.get_material_rates(1, [0, 1], [0, 1, 2])
    expected = np.array([
        [xs_ng @ flux[1], xs_nnp @ flux[1], 0.0],
        [3 * xs_ng @ flux[1], 0.0, xs_n2n @ flux[1]],
    ])
    assert np.allclose(rates, expected)


def test_missing_nuclide_zero_rates_and_single_warning():
    """Nuclides absent from GENDF get zero rates and one warning total."""
    gendf = MockGENDFLibrary({'Al27': {102: np.ones(NG)}})
    flux = np.ones((1, NG))
    helper = _make_helper(gendf, ['Al27', 'Co59'], ['(n,gamma)'], [102], flux)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        rates = helper.get_material_rates(0, [0, 1], [0])
        helper.get_material_rates(0, [0, 1], [0])

    assert rates[0, 0] == pytest.approx(NG)
    assert rates[1, 0] == 0.0
    messages = [str(w.message) for w in record if 'GENDF' in str(w.message)]
    assert len(messages) == 1
    assert 'Co59' in messages[0]


def test_direct_tally_overrides_collapse():
    """Directly tallied scores replace the GENDF collapse values."""
    gendf = MockGENDFLibrary({'U235': {102: np.full(NG, 2.0),
                                       18: np.full(NG, 5.0)}})
    flux = np.ones((1, NG))
    helper = _make_helper(
        gendf, ['U235', 'Co59'], ['(n,gamma)', 'fission'], [102, 18], flux,
        reactions=['fission'])

    helper._rate_tally = Mock()
    helper._rate_tally.nuclides = ['U235', 'Co59']
    helper._rate_tally_means_cache = np.array([[999.0, 777.0]])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rates = helper.get_material_rates(0, [0, 1], [0, 1])

    # (n,gamma) from GENDF collapse; fission from the direct tally, even for
    # the nuclide missing from the GENDF library
    assert rates[0, 0] == pytest.approx(2.0 * NG)
    assert rates[0, 1] == pytest.approx(999.0)
    assert rates[1, 0] == 0.0
    assert rates[1, 1] == pytest.approx(777.0)


def test_isomeric_branching_surface():
    """energies and get_flux_spectrum match DirectWithFluxHelper's surface."""
    gendf = MockGENDFLibrary({'Al27': {102: np.ones(NG)}})
    flux = np.vstack([np.full(NG, 1.5), np.full(NG, 4.0)])
    helper = _make_helper(gendf, ['Al27'], ['(n,gamma)'], [102], flux)

    assert np.array_equal(helper.energies, ENERGIES)
    assert np.allclose(helper.get_flux_spectrum(0), 1.5)
    assert np.allclose(helper.get_flux_spectrum(1), 4.0)


def test_table_rebuild_on_nuclide_growth():
    """Adding nuclides between steps triggers a table rebuild."""
    gendf = MockGENDFLibrary({'Al27': {102: np.full(NG, 2.0)},
                              'Fe56': {102: np.full(NG, 3.0)}})
    flux = np.ones((1, NG))
    helper = _make_helper(gendf, ['Al27'], ['(n,gamma)'], [102], flux)

    rates = helper.get_material_rates(0, [0], [0])
    assert rates[0, 0] == pytest.approx(2.0 * NG)

    helper._nuclides = ['Al27', 'Fe56']
    helper._results_cache = np.empty((2, 1))
    rates = helper.get_material_rates(0, [0, 1], [0])
    assert rates[0, 0] == pytest.approx(2.0 * NG)
    assert rates[1, 0] == pytest.approx(3.0 * NG)


def test_undetected_energy_structure_raises():
    """A library without a detected group structure is rejected."""
    gendf = Mock()
    gendf.energy_structure = None
    with pytest.raises(ValueError, match='energy group structure'):
        GENDFFluxCollapseHelper(1, 1, gendf)


# ===========================================================================
# Nuclide filtering (MicroXS.from_multigroup_flux_with_gendf)
# ===========================================================================
#
# The GENDF workflow filters nuclides at MicroXS creation time, matching the CE
# HDF5 workflow where only nuclides with cross-section data appear.

class MockChain:
    """Mock chain with configurable nuclides."""

    def __init__(self, nuclide_names):
        self.nuclides = [Mock(name=n) for n in nuclide_names]
        for nuc, name in zip(self.nuclides, nuclide_names):
            nuc.name = name
        self.reactions = ['(n,gamma)', 'fission']


def _create_microxs_with_mocks(chain_nuclides, gendf_nuclides, user_nuclides=None):
    """Create MicroXS with mocked chain and GENDF type check."""
    mock_gendf = MockGENDFLibrary(gendf_nuclides)
    mock_chain = MockChain(chain_nuclides)

    flux = np.ones(709)  # CCFE-709

    # Patch chain loading and add MockGENDFLibrary to valid GENDF types
    with patch('openmc.deplete.microxs._get_chain', return_value=mock_chain):
        import openmc.deplete.microxs as microxs_mod
        original_types = microxs_mod._GENDF_TYPES
        try:
            microxs_mod._GENDF_TYPES = (MockGENDFLibrary,) + original_types
            micro_xs = MicroXS.from_multigroup_flux_with_gendf(
                multigroup_flux=flux,
                gendf_library=mock_gendf,
                chain_file='dummy_chain.xml',
                nuclides=user_nuclides
            )
        finally:
            microxs_mod._GENDF_TYPES = original_types

    return micro_xs


def test_microxs_only_contains_gendf_nuclides():
    """MicroXS should only include nuclides present in GENDF library."""
    # Setup: Chain has 5 nuclides, but only 3 have GENDF data
    chain_nuclides = ['U235', 'U238', 'Pu239', 'Am241', 'Cm244']
    gendf_nuclides = {'U235', 'U238', 'Pu239'}  # Am241, Cm244 missing

    micro_xs = _create_microxs_with_mocks(chain_nuclides, gendf_nuclides)

    # Verify: Only nuclides with GENDF data are in MicroXS
    assert set(micro_xs.nuclides) == gendf_nuclides
    assert len(micro_xs.nuclides) == 3

    # Verify: Nuclides without GENDF data are NOT in MicroXS
    assert 'Am241' not in micro_xs.nuclides
    assert 'Cm244' not in micro_xs.nuclides


def test_microxs_excludes_nuclides_without_gendf():
    """Nuclides without GENDF data should not appear in MicroXS."""
    chain_nuclides = ['H1', 'He4', 'Li6', 'Be9', 'B10']
    gendf_nuclides = {'Li6', 'B10'}  # Only 2 of 5 available

    micro_xs = _create_microxs_with_mocks(chain_nuclides, gendf_nuclides)

    # Only Li6 and B10 should be present
    assert len(micro_xs.nuclides) == 2
    assert 'Li6' in micro_xs.nuclides
    assert 'B10' in micro_xs.nuclides

    # H1, He4, Be9 should NOT be present
    for missing in ['H1', 'He4', 'Be9']:
        assert missing not in micro_xs.nuclides


def test_user_provided_nuclides_filtered():
    """User-provided nuclide list should be filtered to GENDF availability."""
    chain_nuclides = ['U235', 'U238', 'Pu239']
    gendf_nuclides = {'U235', 'Pu239'}  # U238 NOT in GENDF

    # User explicitly requests nuclides, including one not in GENDF
    user_nuclides = ['U235', 'U238', 'Pu239']

    micro_xs = _create_microxs_with_mocks(chain_nuclides, gendf_nuclides, user_nuclides)

    # U238 should be filtered out
    assert 'U238' not in micro_xs.nuclides
    assert set(micro_xs.nuclides) == {'U235', 'Pu239'}


def test_empty_result_if_no_gendf_nuclides():
    """If no chain nuclides have GENDF data, MicroXS should be empty."""
    chain_nuclides = ['Og294', 'Ts293', 'Lv292']  # Fictional/rare nuclides
    gendf_nuclides = set()  # Empty GENDF library

    micro_xs = _create_microxs_with_mocks(chain_nuclides, gendf_nuclides)

    # Should have zero nuclides
    assert len(micro_xs.nuclides) == 0


def test_all_nuclides_included_when_all_have_data():
    """When all chain nuclides have GENDF data, all should be included."""
    chain_nuclides = ['U235', 'U238', 'Pu239']
    gendf_nuclides = {'U235', 'U238', 'Pu239'}  # All available

    micro_xs = _create_microxs_with_mocks(chain_nuclides, gendf_nuclides)

    # All nuclides should be present
    assert set(micro_xs.nuclides) == gendf_nuclides
    assert len(micro_xs.nuclides) == 3


def test_nuclide_order_preserved():
    """Nuclide order from chain should be preserved (minus filtered ones)."""
    chain_nuclides = ['A', 'B', 'C', 'D', 'E']
    gendf_nuclides = {'A', 'C', 'E'}  # B, D filtered out

    micro_xs = _create_microxs_with_mocks(chain_nuclides, gendf_nuclides)

    # Order should be A, C, E (chain order, with B, D removed)
    assert micro_xs.nuclides == ['A', 'C', 'E']


def test_array_size_matches_filtered_nuclides():
    """MicroXS data array should only have rows for filtered nuclides."""
    chain_nuclides = ['U235', 'U238', 'Pu239', 'Am241', 'Cm244']
    gendf_nuclides = {'U235', 'Pu239'}  # Only 2 of 5

    micro_xs = _create_microxs_with_mocks(chain_nuclides, gendf_nuclides)

    # Array should have shape (2 nuclides, 2 reactions, 1 group)
    assert micro_xs.data.shape[0] == 2  # Only 2 nuclides
    assert micro_xs.data.shape[0] == len(micro_xs.nuclides)


def test_no_zero_rows_from_missing_nuclides():
    """There should be no all-zero rows from missing nuclides."""
    chain_nuclides = ['U235', 'U238', 'Pu239']
    gendf_nuclides = {'U235', 'Pu239'}  # U238 missing

    micro_xs = _create_microxs_with_mocks(chain_nuclides, gendf_nuclides)

    # Every row should have at least one non-zero value
    # (since mock returns 1.0 for all XS)
    for i, nuc in enumerate(micro_xs.nuclides):
        row_sum = micro_xs.data[i, :, :].sum()
        assert row_sum > 0, f"Row for {nuc} is all zeros"


# ===========================================================================
# HDF5 round-trip (stacked MicroXS write/read infrastructure)
# ===========================================================================

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
    result, flux_data, _ = read_local_microxs_hdf5(fname, mat_ids)

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
    result, _, _ = read_local_microxs_hdf5(fname, local_ids)

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
    result, _, _ = read_local_microxs_hdf5(fname, local_ids)

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
    _, flux_data, _ = read_local_microxs_hdf5(fname, mat_ids)

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
    _, flux_data, _ = read_local_microxs_hdf5(fname, mat_ids)

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
    _, flux_data, _ = read_local_microxs_hdf5(fname, mat_ids)

    assert flux_data is None


# --- Edge cases ---

def test_empty_local_mat_ids(tmp_path):
    """Rank with no materials gets empty results."""
    micros = _make_micros(3)
    mat_ids = _mat_ids(3)
    fname = tmp_path / 'microxs.h5'

    write_global_microxs_hdf5(micros, fname, mat_ids)
    result, flux_data, _ = read_local_microxs_hdf5(fname, [])

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
    result, _, _ = read_local_microxs_hdf5(fname, mat_ids)

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
    result32, _, _ = read_local_microxs_hdf5(f32, mat_ids)
    assert result32[0].data.dtype == np.float32

    # float64
    f64 = tmp_path / 'f64.h5'
    write_global_microxs_hdf5(micros, f64, mat_ids, dtype='float64')
    result64, _, _ = read_local_microxs_hdf5(f64, mat_ids)
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
        result, _, _ = read_local_microxs_hdf5(fname, mat_ids)
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
