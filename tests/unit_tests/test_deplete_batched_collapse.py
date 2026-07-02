"""Unit tests for the batched flux-collapse path in :mod:`openmc.deplete.microxs`.

``_collapse_fluxes`` processes a list/array of multigroup fluxes in chunks,
validating and row-normalizing each chunk and collapsing it against a shared
``_SparseXSTable`` with a single GEMM (``_SparseXSTable.collapse_batch``). These
tests build a random table directly (no nuclear data required) and check that
the batched result reproduces the original per-flux math to ~machine precision,
that chunk boundaries are transparent, and that the zero/invalid/wrong-length
flux guards behave exactly as the per-flux path did.
"""

import warnings

import numpy as np
import pytest

from openmc.deplete import MicroXS
from openmc.deplete.microxs import (
    _SparseXSTable, _collapse_fluxes, _COLLAPSE_CHUNK_SIZE)

# Six valid reaction names (see ``_valid_rxns`` in microxs.py)
REACTIONS = ['(n,gamma)', '(n,2n)', '(n,3n)', '(n,p)', '(n,a)', 'fission']
N_NUC = 50
N_RXN = len(REACTIONS)
N_GROUPS = 200


def _random_table(rng, occupancy=0.6):
    """A random ``_SparseXSTable`` with ~``occupancy`` non-zero (nuc, rxn) rows."""
    nuclides = [f'Nuc{i}' for i in range(N_NUC)]
    rows, nuc_idx, rxn_idx = [], [], []
    for i in range(N_NUC):
        for j in range(N_RXN):
            if rng.random() < occupancy:
                rows.append(rng.random(N_GROUPS))
                nuc_idx.append(i)
                rxn_idx.append(j)
    xs_matrix = np.vstack(rows) if rows else np.empty((0, N_GROUPS))
    return _SparseXSTable(
        nuclides, list(REACTIONS), xs_matrix,
        np.array(nuc_idx, np.int32), np.array(rxn_idx, np.int32))


def _reference_micros(table, fluxes):
    """Reference outputs from the ORIGINAL single-flux math, looped per flux.

    Mirrors the pre-batching ``_collapse_fluxes``: per-flux finite/non-negative
    checks, normalize to sum 1 (zero-sum stays zero), then GEMV + scatter.
    """
    out = []
    for flux in fluxes:
        flux = np.asarray(flux, dtype=float)
        if not np.isfinite(flux).all():
            raise ValueError('Multigroup flux contains non-finite values')
        if (flux < 0).any():
            raise ValueError('Multigroup flux contains negative values')
        flux_sum = flux.sum()
        phi = flux / flux_sum if flux_sum else flux
        arr = np.zeros((len(table.nuclides), len(table.reactions)))
        arr[table.nuc_indices, table.rxn_indices] = table.xs_matrix @ phi
        out.append(arr[:, :, np.newaxis])
    return out


def test_batched_matches_per_flux_reference():
    """3000 random fluxes: batched result == original per-flux math, rtol 1e-12."""
    rng = np.random.default_rng(12345)
    table = _random_table(rng)
    fluxes = [rng.random(N_GROUPS) for _ in range(3000)]

    micros = _collapse_fluxes(table, fluxes)
    reference = _reference_micros(table, fluxes)

    assert len(micros) == len(fluxes)
    for micro, ref in zip(micros, reference):
        assert micro.data.shape == (N_NUC, N_RXN, 1)
        np.testing.assert_allclose(micro.data, ref, rtol=1e-12)


def test_2d_array_input_matches_and_no_mutation():
    """A 2-D ndarray batch collapses correctly and is not mutated in place."""
    rng = np.random.default_rng(7)
    table = _random_table(rng)
    fluxes = rng.random((257, N_GROUPS))
    original = fluxes.copy()

    micros = _collapse_fluxes(table, fluxes)
    reference = _reference_micros(table, fluxes)

    for micro, ref in zip(micros, reference):
        np.testing.assert_allclose(micro.data, ref, rtol=1e-12)
    # The input array must be untouched
    assert np.array_equal(fluxes, original)


@pytest.mark.parametrize('n', [
    1,
    _COLLAPSE_CHUNK_SIZE - 1,
    _COLLAPSE_CHUNK_SIZE,
    _COLLAPSE_CHUNK_SIZE + 1,
    _COLLAPSE_CHUNK_SIZE // 3,   # < chunk
])
def test_chunk_boundary_counts(n):
    """n_fluxes at/around the chunk size returns the right count and values."""
    rng = np.random.default_rng(n)
    table = _random_table(rng)
    fluxes = [rng.random(N_GROUPS) for _ in range(n)]

    micros = _collapse_fluxes(table, fluxes)
    reference = _reference_micros(table, fluxes)

    assert len(micros) == n
    for micro, ref in zip(micros, reference):
        np.testing.assert_allclose(micro.data, ref, rtol=1e-12)


def test_result_independent_of_chunk_size():
    """The output must not depend on how the fluxes are split into chunks."""
    rng = np.random.default_rng(99)
    table = _random_table(rng)
    fluxes = [rng.random(N_GROUPS) for _ in range(300)]

    baseline = [m.data for m in _collapse_fluxes(table, fluxes, chunk_size=1)]
    for chunk_size in (3, 7, 128, 300, _COLLAPSE_CHUNK_SIZE):
        micros = _collapse_fluxes(table, fluxes, chunk_size=chunk_size)
        for micro, ref in zip(micros, baseline):
            np.testing.assert_allclose(micro.data, ref, rtol=1e-12)


def test_zero_flux_is_all_zero_no_nan_no_warning():
    """A zero flux vector collapses to all zeros with no NaN and no warning."""
    rng = np.random.default_rng(3)
    table = _random_table(rng)
    # Mix a zero flux among non-zero ones to exercise the normalization guard
    fluxes = [rng.random(N_GROUPS), np.zeros(N_GROUPS), rng.random(N_GROUPS)]

    with warnings.catch_warnings():
        warnings.simplefilter('error')  # any warning becomes an error
        micros = _collapse_fluxes(table, fluxes)

    zero = micros[1].data
    assert zero.shape == (N_NUC, N_RXN, 1)
    assert np.all(zero == 0.0)
    assert not np.isnan(zero).any()
    # The non-zero neighbours are unaffected
    reference = _reference_micros(table, fluxes)
    np.testing.assert_allclose(micros[0].data, reference[0], rtol=1e-12)
    np.testing.assert_allclose(micros[2].data, reference[2], rtol=1e-12)


@pytest.mark.parametrize('bad_value, keyword', [
    (np.nan, 'non-finite'),
    (np.inf, 'non-finite'),
    (-np.inf, 'non-finite'),
    (-1.0, 'negative'),
])
def test_invalid_flux_raises_with_index(bad_value, keyword):
    """NaN/inf/negative raise ValueError identifying the offending flux index."""
    rng = np.random.default_rng(1)
    table = _random_table(rng)
    fluxes = [rng.random(N_GROUPS) for _ in range(6)]
    bad_index = 4
    fluxes[bad_index] = np.ones(N_GROUPS)
    fluxes[bad_index][10] = bad_value

    with pytest.raises(ValueError) as excinfo:
        _collapse_fluxes(table, fluxes)
    message = str(excinfo.value)
    assert keyword in message
    assert str(bad_index) in message


def test_invalid_flux_index_across_chunk_boundary():
    """The reported index is the global flux index, not the in-chunk offset."""
    rng = np.random.default_rng(2)
    table = _random_table(rng)
    bad_index = _COLLAPSE_CHUNK_SIZE + 5
    fluxes = [rng.random(N_GROUPS) for _ in range(bad_index + 3)]
    fluxes[bad_index] = np.ones(N_GROUPS)
    fluxes[bad_index][0] = np.nan

    with pytest.raises(ValueError, match=f'flux {bad_index} '):
        _collapse_fluxes(table, fluxes)


def test_finite_check_precedes_negative_within_flux():
    """A flux that is both non-finite and negative reports 'non-finite'."""
    rng = np.random.default_rng(4)
    table = _random_table(rng)
    flux = np.ones(N_GROUPS)
    flux[0] = np.nan
    flux[1] = -1.0

    with pytest.raises(ValueError, match='non-finite'):
        _collapse_fluxes(table, [flux])


def test_first_offending_flux_wins_negative_before_nonfinite():
    """When an earlier flux is negative and a later one non-finite, the earlier
    (negative) flux is reported first, matching the original per-flux order."""
    rng = np.random.default_rng(5)
    table = _random_table(rng)
    fluxes = [rng.random(N_GROUPS) for _ in range(5)]
    fluxes[1] = np.ones(N_GROUPS)
    fluxes[1][0] = -1.0            # negative at index 1
    fluxes[3] = np.ones(N_GROUPS)
    fluxes[3][0] = np.inf         # non-finite at index 3

    with pytest.raises(ValueError, match='flux 1 contains negative'):
        _collapse_fluxes(table, fluxes)


@pytest.mark.parametrize('fluxes', [
    [np.ones(N_GROUPS - 1)],                       # single, uniform wrong length
    [np.ones(N_GROUPS), np.ones(N_GROUPS + 3)],    # uniform-but-wrong within chunk
    [np.ones(N_GROUPS), np.ones(N_GROUPS - 1)],    # ragged
])
def test_wrong_length_flux_raises(fluxes):
    """A flux whose length differs from the table's group count raises."""
    rng = np.random.default_rng(6)
    table = _random_table(rng)
    with pytest.raises(ValueError):
        _collapse_fluxes(table, fluxes)


def test_collapse_matches_collapse_batch_single_row():
    """The single-flux collapse equals one row of the batched collapse."""
    rng = np.random.default_rng(8)
    table = _random_table(rng)
    phi = rng.random(N_GROUPS)
    phi = phi / phi.sum()

    single = table.collapse(phi)
    batched = table.collapse_batch(phi[np.newaxis])
    assert single.shape == (N_NUC, N_RXN)
    assert batched.shape == (1, N_NUC, N_RXN)
    np.testing.assert_allclose(single, batched[0], rtol=1e-12)
