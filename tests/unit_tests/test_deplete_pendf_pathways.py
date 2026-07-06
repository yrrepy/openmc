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

from openmc.deplete.chain import Chain
from openmc.deplete.nuclide import Nuclide
from openmc.deplete.microxs import (
    MicroXS,
    _build_xs_table_pendf,
    _check_pathway_consistency,
    _liso_from_gnds,
)

CHAIN_FILE = Path(__file__).parents[1] / "chain_simple.xml"


class _FakePendf:
    """Duck-typed stand-in exposing the frozen §4.2 pathway accessors."""

    def __init__(self, mf3, mf10=None):
        # mf3:  {nuclide: {mt: (energy, xs)}}                       MF=3 totals
        # mf10: {nuclide: {mt: {lfs or (lfs, izap): (product|None, (energy, xs))}}}
        # An MF=10 key may be a bare LFS (the common non-lumped partial: one
        # product per level, IZAP defaulted) or an explicit (LFS, IZAP) pair (a
        # lumped level shared by several product nuclides, as MT=5 does). Both
        # normalize to (LFS, IZAP) so ``pathways`` yields the frozen pairs.
        self._mf3 = mf3
        self._mf10 = {
            nuc: {mt: {(k if isinstance(k, tuple) else (k, 0)): v
                       for k, v in parts.items()}
                  for mt, parts in by_mt.items()}
            for nuc, by_mt in (mf10 or {}).items()}

    @property
    def nuclides(self):
        return list(self._mf3)

    def reactions(self, nuclide):
        return list(self._mf3[nuclide])

    def xs(self, nuclide, mt):
        return self._mf3[nuclide][mt]

    def pathways(self, nuclide, mt):
        return sorted(self._mf10.get(nuclide, {}).get(mt, {}))

    def pathway_xs(self, nuclide, mt, lfs, izap=None):
        return self._mf10[nuclide][mt][(lfs, izap)][1]

    def product(self, nuclide, mt, lfs, izap=None):
        return self._mf10[nuclide][mt][(lfs, izap)][0]


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


def test_duplicate_product_rows_sum():
    # Two MF=10 levels (LFS 1 and 2) both map to the SAME product isomer
    # Am242_m1 -- a level index is not the observable final state, so two levels
    # feeding one final state is physically legitimate. Their 1-barn partials
    # must SUM into a single qualified row (2 barns), not last-wins overwrite at
    # the collapse fancy-index assignment.
    fake = _FakePendf(
        mf3={"Am241": {102: _const(2.0)}},
        mf10={"Am241": {102: {
            1: ("Am242_m1", _const(1.0)),
            2: ("Am242_m1", _const(1.0)),
        }}})
    edges = np.array([0.0, 1.0e7, 2.0e7])

    table = _build_xs_table_pendf(["Am241"], ["(n,gamma)"], edges, fake)

    # The qualified name appears in the axis exactly once
    assert table.reactions == ["(n,gamma)", "(n,gamma)_m1"]
    assert table.reactions.count("(n,gamma)_m1") == 1
    m1_idx = table.reactions.index("(n,gamma)_m1")
    # Exactly one staged row targets the m1 reaction -> no duplicate to clobber
    assert list(table.rxn_indices).count(m1_idx) == 1
    row = next(table.xs_matrix[i]
               for i, r in enumerate(table.rxn_indices) if r == m1_idx)
    np.testing.assert_allclose(row, [2.0, 2.0])   # SUM of the two partials

    # End to end: the collapsed qualified row equals the summed 2 barns
    micro = MicroXS.from_multigroup_flux(
        energies=edges, multigroup_flux=[1.0, 1.0], chain_file=CHAIN_FILE,
        nuclides=["Am241"], reactions=["(n,gamma)"], pendf_library=fake)
    assert list(micro.reactions).count("(n,gamma)_m1") == 1
    assert micro["Am241", "(n,gamma)_m1"] == pytest.approx([2.0])


def test_duplicate_zero_then_nonzero_row_sums_once():
    # A zero metastable partial staged first (keep_zero) followed by a nonzero
    # duplicate for the same product must end up summed and present exactly once.
    fake = _FakePendf(
        mf3={"Am241": {102: _const(3.0)}},
        mf10={"Am241": {102: {
            1: ("Am242_m1", _const(0.0)),   # zero, staged via keep_zero
            2: ("Am242_m1", _const(3.0)),   # nonzero duplicate -> sums in
        }}})
    edges = np.array([0.0, 1.0e7, 2.0e7])

    table = _build_xs_table_pendf(["Am241"], ["(n,gamma)"], edges, fake)

    assert table.reactions == ["(n,gamma)", "(n,gamma)_m1"]
    m1_idx = table.reactions.index("(n,gamma)_m1")
    assert list(table.rxn_indices).count(m1_idx) == 1
    row = next(table.xs_matrix[i]
               for i, r in enumerate(table.rxn_indices) if r == m1_idx)
    np.testing.assert_allclose(row, [3.0, 3.0])


def test_lumped_multiproduct_channel_raises():
    # A lumped reaction (MT=5 (n,misc)) whose MF=10 partials name two DIFFERENT
    # daughter nuclides at one collapse row is refused: summing unrelated
    # daughters into a single reaction row is never valid. Contrast
    # test_duplicate_product_rows_sum, where two levels of the SAME daughter do
    # sum. Here two products share LFS 0 (the ground row) but differ by IZAP.
    fake = _FakePendf(
        mf3={"Fe56": {5: _const(5.0)}},
        mf10={"Fe56": {5: {
            (0, 26057): ("Fe57", _const(3.0)),   # one daughter
            (0, 25056): ("Mn56", _const(2.0)),   # a DIFFERENT daughter, same LFS
        }}})
    edges = np.array([0.0, 2.0e7])

    with pytest.raises(ValueError, match="lumped"):
        _build_xs_table_pendf(["Fe56"], ["(n,misc)"], edges, fake)


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


# ---------------------------------------------------------------------------
# Deplete-time chain <-> MicroXS pathway-mismatch hard error
# (plan section 1.1: mismatch is a hard error, never a silent fallback)
# ---------------------------------------------------------------------------

def _chain(reactions_by_nuc):
    """Minimal Chain with the given ``{nuc: [(type, target), ...]}`` reactions."""
    chain = Chain()
    for name, rxns in reactions_by_nuc.items():
        nuc = Nuclide(name)
        for rtype, target in rxns:
            nuc.add_reaction(rtype, target, 0.0, 1.0)
        chain.add_nuclide(nuc)
    return chain


def _micro(carried_by_nuc, reactions):
    """MicroXS over ``reactions`` with a non-zero group for carried pairs."""
    nuclides = list(carried_by_nuc)
    data = np.zeros((len(nuclides), len(reactions), 1))
    for i, nuc in enumerate(nuclides):
        for rx in carried_by_nuc[nuc]:
            data[i, reactions.index(rx), 0] = 1.0
    return MicroXS(data, nuclides, reactions)


def test_pathway_mismatch_micro_only_raises():
    """(a) MicroXS carries a qualified row the chain cannot route -> raise."""
    chain = _chain({"In115": [("(n,gamma)", "In116")]})  # no _m1 in chain
    micro = _micro({"In115": ["(n,gamma)", "(n,gamma)_m1"]},
                   ["(n,gamma)", "(n,gamma)_m1"])
    with pytest.raises(ValueError, match="pathway mismatch") as exc:
        _check_pathway_consistency(chain, micro)
    assert "In115 (n,gamma)_m1" in str(exc.value)


def test_pathway_mismatch_chain_only_raises():
    """(b) Chain carries a qualified pathway the MicroXS lacks (base present)."""
    chain = _chain({"In115": [("(n,gamma)", "In116"),
                              ("(n,gamma)_m1", "In116_m1")]})
    micro = _micro({"In115": ["(n,gamma)"]}, ["(n,gamma)"])  # base only
    with pytest.raises(ValueError, match="pathway mismatch") as exc:
        _check_pathway_consistency(chain, micro)
    assert "In115 (n,gamma)_m1" in str(exc.value)


def test_pathway_consistent_qualified_passes():
    """Matching qualified reactions on both sides -> no error."""
    chain = _chain({"In115": [("(n,gamma)", "In116"),
                              ("(n,gamma)_m1", "In116_m1")]})
    micro = _micro({"In115": ["(n,gamma)", "(n,gamma)_m1"]},
                   ["(n,gamma)", "(n,gamma)_m1"])
    _check_pathway_consistency(chain, micro)  # no raise


def test_plain_non_pathway_passes():
    """No qualified names anywhere -> untouched, even with unqualified diffs and
    nuclides present on only one side."""
    chain = _chain({"Fe56": [("(n,gamma)", "Fe57"), ("(n,p)", "Mn56")],
                    "Cs137": [("(n,gamma)", "Cs138")]})  # Cs137 only in chain
    micro = _micro({"Fe56": ["(n,gamma)"],            # (n,p) only in chain: ok
                    "W186": ["(n,gamma)"]},           # W186 only in MicroXS: ok
                   ["(n,gamma)", "(n,p)"])
    _check_pathway_consistency(chain, micro)  # no raise


def test_pathway_chain_qualified_but_micro_has_no_data_passes():
    """Chain has a qualified pathway but MicroXS carries no data for the nuclide
    (base absent) -> a no-data nuclide, not a mismatch."""
    chain = _chain({"In115": [("(n,gamma)", "In116"),
                              ("(n,gamma)_m1", "In116_m1")]})
    micro = _micro({"In115": []}, ["(n,gamma)"])  # In115 present but all-zero
    _check_pathway_consistency(chain, micro)  # no raise


def test_zero_metastable_partial_stages_and_passes():
    """Regression: a metastable MF=10 partial that group-averages to exactly
    zero (its threshold is above the tally groups) must still stage a zero row,
    so the reaction axis carries the qualified name (Part 1) and the chain <->
    MicroXS consistency check does not raise a false positive (Part 2, mode b).

    Physics: at benchmark energies all yield goes to the ground product, so the
    metastable partial averages to 0. The ground partial IS the ground route
    (staged under the base name), not a misrouted MF=3 total.
    """
    fake = _FakePendf(
        mf3={"In115": {102: _const(4.0)}},
        mf10={"In115": {102: {
            0: ("In116", _const(4.0)),      # ground: all yield here
            1: ("In116_m1", _const(0.0)),   # metastable: zero over the groups
        }}})
    edges = [0.0, 1.0e7, 2.0e7]

    # Part 1: the all-zero metastable partial is staged, so the axis carries the
    # qualified name with an all-zero row (ground first -> rxn 0, m1 -> rxn 1).
    table = _build_xs_table_pendf(["In115"], ["(n,gamma)"], edges, fake)
    assert table.reactions == ["(n,gamma)", "(n,gamma)_m1"]
    assert table.nuc_indices.tolist() == [0, 0]
    assert table.rxn_indices.tolist() == [0, 1]
    np.testing.assert_array_equal(table.xs_matrix[0], [4.0, 4.0])   # ground
    np.testing.assert_array_equal(table.xs_matrix[1], [0.0, 0.0])   # m1 all-zero

    # Collapse: the qualified column is present in the MicroXS but all-zero.
    micro = MicroXS.from_multigroup_flux(
        energies=edges, multigroup_flux=[1.0, 1.0], chain_file=CHAIN_FILE,
        nuclides=["In115"], reactions=["(n,gamma)"], pendf_library=fake)
    assert "(n,gamma)_m1" in micro.reactions
    assert micro["In115", "(n,gamma)"] == pytest.approx([4.0])
    assert micro["In115", "(n,gamma)_m1"] == pytest.approx([0.0])

    # Part 2: chain carries the qualified pathway; axis membership marks it
    # resolved, so the check must NOT raise (this was the false positive).
    chain = _chain({"In115": [("(n,gamma)", "In116"),
                              ("(n,gamma)_m1", "In116_m1")]})
    _check_pathway_consistency(chain, micro)  # no raise
