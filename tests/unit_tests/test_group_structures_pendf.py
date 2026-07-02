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


@pytest.mark.parametrize("name,n_edges", PENDF_GROUP_STRUCTURES.items())
def test_pendf_group_structure(name, n_edges):
    assert name in openmc.mgxs.GROUP_STRUCTURES

    edges = openmc.mgxs.GROUP_STRUCTURES[name]
    assert edges.size == n_edges
    assert np.all(np.diff(edges) > 0.0)
    assert edges[0] >= 0.0
