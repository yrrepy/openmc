"""Unit tests for isomeric pathway (MF=10) expansion in the PENDF collapse path.

Covers the ORIGEN-style "Option A" per-product rows built by
``_build_xs_table_pendf`` (ground row keeps the canonical reaction name,
metastable products get an ``_m{n}`` suffix parsed from the product's GNDS
name), the mapping='none' fallback, the Sigma(partials) vs MF=3-total
consistency check, and the ``pathways`` dispatch through
:meth:`MicroXS.from_multigroup_flux`. Pathway rows come exclusively from MF=10
partial cross sections -- never from static branching ratios.
"""
from pathlib import Path

import numpy as np
import pytest

from openmc.deplete.microxs import (
    MicroXS,
    _build_xs_table_pendf,
    _liso_from_gnds,
)

CHAIN_FILE = Path(__file__).parents[1] / "chain_simple.xml"


class _FakePendf:
    """Duck-typed stand-in exposing the frozen §4.2 pathway accessors."""

    def __init__(self, mf3, mf10=None):
        # mf3:  {nuclide: {mt: (energy, xs)}}                      MF=3 totals
        # mf10: {nuclide: {mt: {lfs: (product|None, (energy, xs))}}}  MF=10
        self._mf3 = mf3
        self._mf10 = mf10 or {}

    @property
    def nuclides(self):
        return list(self._mf3)

    def reactions(self, nuclide):
        return list(self._mf3[nuclide])

    def xs(self, nuclide, mt):
        return self._mf3[nuclide][mt]

    def pathways(self, nuclide, mt):
        return list(self._mf10.get(nuclide, {}).get(mt, {}))

    def pathway_xs(self, nuclide, mt, lfs):
        return self._mf10[nuclide][mt][lfs][1]

    def product(self, nuclide, mt, lfs):
        return self._mf10[nuclide][mt][lfs][0]


# Constant cross sections over the full span so every group average equals the
# constant, making hand-computed group cross sections trivial.
_E = np.array([0.0, 2.0e7])


def _const(value):
    return (_E, np.array([value, value]))


# ---------------------------------------------------------------------------
# _liso_from_gnds
# ---------------------------------------------------------------------------

def test_liso_from_gnds():
    assert _liso_from_gnds("Am242") == 0
    assert _liso_from_gnds("Am242_m1") == 1
    assert _liso_from_gnds("Ir192_m2") == 2


# ---------------------------------------------------------------------------
# Mapped MF=10 partials -> per-product rows
# ---------------------------------------------------------------------------

def test_pathway_rows_mapped():
    # LFS is a level index, not an isomer ordinal: LFS {0, 2} map to products
    # Am242 (ground) and Am242_m1. The row suffix must follow the PRODUCT LISO
    # (m1), not the LFS value (2).
    fake = _FakePendf(
        mf3={"Am241": {102: _const(5.0)}},
        mf10={"Am241": {102: {
            0: ("Am242", _const(4.0)),
            2: ("Am242_m1", _const(1.0)),
        }}})
    edges = np.array([0.0, 1.0e7, 2.0e7])

    table = _build_xs_table_pendf(["Am241"], ["(n,gamma)"], edges, fake)

    # Expanded reaction axis: base name (ground) then ascending isomer
    assert table.reactions == ["(n,gamma)", "(n,gamma)_m1"]
    # Two rows, both for nuclide 0; ground staged first -> rxn 0, m1 -> rxn 1
    assert table.nuc_indices.tolist() == [0, 0]
    assert table.rxn_indices.tolist() == [0, 1]
    assert table.xs_matrix.dtype == np.float64
    np.testing.assert_allclose(table.xs_matrix[0], [4.0, 4.0])   # ground
    np.testing.assert_allclose(table.xs_matrix[1], [1.0, 1.0])   # m1

    # A no-pathway build gives the single MF=3 total, and the partials sum to it
    total = _build_xs_table_pendf(
        ["Am241"], ["(n,gamma)"], edges, fake, pathways=False)
    assert total.reactions == ["(n,gamma)"]
    np.testing.assert_allclose(total.xs_matrix[0], [5.0, 5.0])
    np.testing.assert_allclose(
        table.xs_matrix[0] + table.xs_matrix[1], total.xs_matrix[0])


def test_reaction_without_mf10_single_row():
    """A reaction with no MF=10 data yields one canonical (ground) row."""
    fake = _FakePendf(mf3={"Fe56": {102: _const(2.0)}})  # no MF=10 at all
    edges = np.array([0.0, 2.0e7])

    table = _build_xs_table_pendf(["Fe56"], ["(n,gamma)"], edges, fake)

    assert table.reactions == ["(n,gamma)"]
    assert table.rxn_indices.tolist() == [0]
    np.testing.assert_allclose(table.xs_matrix[0], [2.0])


def test_unmapped_partials_fall_back_and_warn():
    """mapping='none' partials cannot be named -> MF=3 total + one warning."""
    fake = _FakePendf(
        mf3={"Am241": {102: _const(5.0)}},
        mf10={"Am241": {102: {
            0: (None, _const(4.0)),
            2: (None, _const(1.0)),
        }}})
    edges = np.array([0.0, 2.0e7])

    with pytest.warns(UserWarning, match="product"):
        table = _build_xs_table_pendf(["Am241"], ["(n,gamma)"], edges, fake)

    # Fell back to the single MF=3-total row (no qualified names)
    assert table.reactions == ["(n,gamma)"]
    assert table.rxn_indices.tolist() == [0]
    np.testing.assert_allclose(table.xs_matrix[0], [5.0])


def test_consistency_check_warns_on_mismatch():
    """Sigma(partials) that disagree with the MF=3 total raise a warning."""
    fake = _FakePendf(
        mf3={"Am241": {102: _const(5.0)}},
        mf10={"Am241": {102: {
            0: ("Am242", _const(4.0)),
            2: ("Am242_m1", _const(0.5)),   # 4.0 + 0.5 = 4.5 != 5.0
        }}})
    edges = np.array([0.0, 2.0e7])

    with pytest.warns(UserWarning, match="partials"):
        table = _build_xs_table_pendf(["Am241"], ["(n,gamma)"], edges, fake)

    # Rows are still emitted from the partials despite the warning
    assert table.reactions == ["(n,gamma)", "(n,gamma)_m1"]
    np.testing.assert_allclose(table.xs_matrix[0], [4.0])
    np.testing.assert_allclose(table.xs_matrix[1], [0.5])


def test_matching_partials_do_not_warn(recwarn):
    """Partials that sum to the total within tolerance emit no warning."""
    fake = _FakePendf(
        mf3={"Am241": {102: _const(5.0)}},
        mf10={"Am241": {102: {
            0: ("Am242", _const(4.0)),
            2: ("Am242_m1", _const(1.0)),
        }}})
    edges = np.array([0.0, 2.0e7])

    _build_xs_table_pendf(["Am241"], ["(n,gamma)"], edges, fake)
    assert len(recwarn) == 0


# ---------------------------------------------------------------------------
# End-to-end through from_multigroup_flux
# ---------------------------------------------------------------------------

def test_from_multigroup_flux_pathways_end_to_end():
    fake = _FakePendf(
        mf3={"Am241": {102: _const(5.0)}},
        mf10={"Am241": {102: {
            0: ("Am242", _const(4.0)),
            2: ("Am242_m1", _const(1.0)),
        }}})
    edges = [0.0, 1.0e7, 2.0e7]
    flux = [1.0, 3.0]  # constant xs is flux-independent

    micro = MicroXS.from_multigroup_flux(
        energies=edges, multigroup_flux=flux, chain_file=CHAIN_FILE,
        nuclides=["Am241"], reactions=["(n,gamma)"], pendf_library=fake)

    assert isinstance(micro, MicroXS)
    assert "(n,gamma)" in micro.reactions
    assert "(n,gamma)_m1" in micro.reactions
    assert micro["Am241", "(n,gamma)"] == pytest.approx([4.0])
    assert micro["Am241", "(n,gamma)_m1"] == pytest.approx([1.0])

    # pathways=False collapses back to the MF=3 total under the canonical name
    micro_no = MicroXS.from_multigroup_flux(
        energies=edges, multigroup_flux=flux, chain_file=CHAIN_FILE,
        nuclides=["Am241"], reactions=["(n,gamma)"], pendf_library=fake,
        pathways=False)
    assert micro_no.reactions == ["(n,gamma)"]
    assert micro_no["Am241", "(n,gamma)"] == pytest.approx([5.0])
