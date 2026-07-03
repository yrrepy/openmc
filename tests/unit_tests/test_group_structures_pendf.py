"""Sanity checks for the PENDF/MGB group structures added to
``openmc.mgxs.GROUP_STRUCTURES`` (FOMG-16k, VESTA-43k)."""

import numpy as np
import pytest

import openmc.mgxs

# name -> expected number of energy edges (groups + 1)
PENDF_GROUP_STRUCTURES = {
    'FOMG-16k': 16001,
    'VESTA-43k': 43001,
}

# Landmark/regression values for the two newly added structures. The checksum
# is a float64 sum of every edge; first/last are the boundary edges. Values are
# stored as bit-exact hex float literals (decimal shown in the comment) so the
# regression is insensitive to text round-tripping. Regenerate with, e.g.:
#     e = openmc.mgxs.GROUP_STRUCTURES['FOMG-16k']
#     e[0].hex(), e[-1].hex(), np.float64(e.sum()).hex()
PENDF_REGRESSION = {
    #             n_edges, first_hex,               last_hex,                sum_hex
    'FOMG-16k':  (16001,   '0x1.4f8b588e368f1p-17', '0x1.2b12800000000p+24', '0x1.7bac0f2cc113ap+33'),  # 1e-05, 1.96e7, 12739681881.508411
    'VESTA-43k': (43001,   '0x1.4f8b588e368f1p-17', '0x1.312d000000000p+24', '0x1.2ba4220bcdd67p+34'),  # 1e-05, 2.00e7, 20108576815.216213
}


@pytest.mark.parametrize("name,n_edges", PENDF_GROUP_STRUCTURES.items())
def test_pendf_group_structure(name, n_edges):
    assert name in openmc.mgxs.GROUP_STRUCTURES

    edges = openmc.mgxs.GROUP_STRUCTURES[name]
    assert edges.size == n_edges
    assert np.all(np.diff(edges) > 0.0)
    assert edges[0] >= 0.0


@pytest.mark.parametrize("name", list(openmc.mgxs.GROUP_STRUCTURES))
def test_group_structure_is_valid(name):
    """Every GROUP_STRUCTURES entry must be a well-formed 1-D float64 array of
    strictly ascending, finite, non-negative energy edges."""
    edges = openmc.mgxs.GROUP_STRUCTURES[name]

    assert isinstance(edges, np.ndarray)
    assert edges.ndim == 1
    assert edges.dtype == np.float64
    assert edges.size >= 2
    assert np.all(np.isfinite(edges))
    assert np.all(np.diff(edges) > 0.0)  # strictly ascending
    assert edges[0] >= 0.0


@pytest.mark.parametrize("name", list(PENDF_REGRESSION))
def test_pendf_group_structure_regression(name):
    """Bit-level regression guard for the two new structures: exact length,
    exact first/last edge, and an exact float64 checksum of all edges."""
    n_edges, first_hex, last_hex, sum_hex = PENDF_REGRESSION[name]
    edges = openmc.mgxs.GROUP_STRUCTURES[name]

    assert edges.size == n_edges
    assert edges[0] == np.float64(float.fromhex(first_hex))
    assert edges[-1] == np.float64(float.fromhex(last_hex))
    assert np.float64(edges.sum()) == np.float64(float.fromhex(sum_hex))
