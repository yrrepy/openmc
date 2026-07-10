"""Unit tests for isomeric pathway (MF=10) expansion in the PENDF collapse path.

Covers the ORIGEN-style "Option A" per-product rows built by
``_build_xs_table_pendf`` (ground row keeps the canonical reaction name,
metastable products get an ``_m{n}`` suffix), the **chain-sourced** row naming
(each MF=10 ``LFS`` partial is bound to the depletion-chain reaction carrying
that ``pendf_lfs``), the unbound-partial MF=3-total fallback, the
Sigma(partials) vs MF=3-total consistency check, and the always-on expansion
through :meth:`MicroXS.from_multigroup_flux`. Pathway rows come exclusively from
MF=10 partial cross sections -- never from static branching ratios; the product
*names* come from the chain, never from the library.
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
    _group_average,
    _liso_from_gnds,
)

CHAIN_FILE = Path(__file__).parents[1] / "chain_simple.xml"


class _FakePendf:
    """Duck-typed stand-in exposing the frozen §4.2 raw pathway accessors.

    The library no longer bakes product names; ``product()`` is gone and the
    isomer<->LFS mapping lives on the depletion chain. The MF=10 entries still
    carry a product name here only so :func:`_chain_from_fake` can synthesize the
    matching chain reactions.
    """

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


def _chain_from_fake(fake, names):
    """Build a Chain binding fake MF=10 partials to product-qualified reactions.

    ``names`` maps ``mt -> base reaction name``. For each nuclide/mt the fake
    exposes MF=10 partials, one :class:`ReactionTuple` is added per LFS: the
    ``type`` is the base name for a ground product (no ``_mN``) or ``base_m{n}``
    for a metastable product (``n`` parsed from the product name), the ``target``
    is the product name, and ``pendf_lfs`` the LFS. A partial whose product is
    ``None`` (unmappable) is skipped, so the collapse finds no chain reaction for
    that LFS and falls back to the MF=3 total.
    """
    chain = Chain()
    for nuc in fake.nuclides:
        nuclide = Nuclide(nuc)
        for mt, base in names.items():
            for lfs, izap in fake.pathways(nuc, mt):
                product = fake._mf10[nuc][mt][(lfs, izap)][0]
                if product is None:
                    continue
                liso = _liso_from_gnds(product)
                rtype = base if liso == 0 else f"{base}_m{liso}"
                nuclide.add_reaction(rtype, product, 0.0, 1.0, pendf_lfs=lfs)
        chain.add_nuclide(nuclide)
    return chain


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
# Chain-sourced MF=10 partials -> per-product rows
# ---------------------------------------------------------------------------

def test_pathway_rows_mapped():
    # LFS is a level index, not an isomer ordinal: LFS {0, 2} map to products
    # Am242 (ground) and Am242_m1. The row suffix follows the PRODUCT LISO (m1)
    # as recorded on the chain reaction bound to that LFS, not the LFS value (2).
    fake = _FakePendf(
        mf3={"Am241": {102: _const(5.0)}},
        mf10={"Am241": {102: {
            0: ("Am242", _const(4.0)),
            2: ("Am242_m1", _const(1.0)),
        }}})
    edges = np.array([0.0, 1.0e7, 2.0e7])
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})

    table = _build_xs_table_pendf(["Am241"], ["(n,gamma)"], edges, fake, chain)

    # Expanded reaction axis: base name (ground) then ascending isomer
    assert table.reactions == ["(n,gamma)", "(n,gamma)_m1"]
    # Two rows, both for nuclide 0; ground staged first -> rxn 0, m1 -> rxn 1
    assert table.nuc_indices.tolist() == [0, 0]
    assert table.rxn_indices.tolist() == [0, 1]
    assert table.xs_matrix.dtype == np.float64
    np.testing.assert_allclose(table.xs_matrix[0], [4.0, 4.0])   # ground
    np.testing.assert_allclose(table.xs_matrix[1], [1.0, 1.0])   # m1

    # Conservation: the partial rows sum to the MF=3 total collapsed with the
    # same flat-in-bin kernel the table builder uses.
    total_g = _group_average(*fake.xs("Am241", 102), edges)
    np.testing.assert_allclose(total_g, [5.0, 5.0])
    np.testing.assert_allclose(
        table.xs_matrix[0] + table.xs_matrix[1], total_g)


def test_duplicate_product_rows_sum():
    # Two MF=10 levels (LFS 1 and 2) both map to the SAME product isomer
    # Am242_m1 -- a level index is not the observable final state, so two levels
    # feeding one final state is physically legitimate. Their 1-barn partials
    # must SUM into a single qualified row (2 barns), not last-wins overwrite at
    # the collapse fancy-index assignment. Both LFS bind to the SAME chain
    # reaction (n,gamma)_m1 (same IZAP -> the designed sum case, not a collision).
    fake = _FakePendf(
        mf3={"Am241": {102: _const(2.0)}},
        mf10={"Am241": {102: {
            1: ("Am242_m1", _const(1.0)),
            2: ("Am242_m1", _const(1.0)),
        }}})
    edges = np.array([0.0, 1.0e7, 2.0e7])
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})

    table = _build_xs_table_pendf(["Am241"], ["(n,gamma)"], edges, fake, chain)

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
        energies=edges, multigroup_flux=[1.0, 1.0], chain_file=chain,
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
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})

    table = _build_xs_table_pendf(["Am241"], ["(n,gamma)"], edges, fake, chain)

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
    # sum. Here two products share LFS 0 (the ground row) but differ by IZAP, so
    # they bind to one chain reaction from distinct daughters -> collision.
    fake = _FakePendf(
        mf3={"Fe56": {5: _const(5.0)}},
        mf10={"Fe56": {5: {
            (0, 26057): ("Fe57", _const(3.0)),   # one daughter
            (0, 25056): ("Mn56", _const(2.0)),   # a DIFFERENT daughter, same LFS
        }}})
    edges = np.array([0.0, 2.0e7])
    chain = _chain_from_fake(fake, {5: "(n,misc)"})

    with pytest.raises(ValueError, match="lumped"):
        _build_xs_table_pendf(["Fe56"], ["(n,misc)"], edges, fake, chain)


def test_reaction_without_mf10_single_row():
    """A reaction with no MF=10 data yields one canonical (ground) row."""
    fake = _FakePendf(mf3={"Fe56": {102: _const(2.0)}})  # no MF=10 at all
    edges = np.array([0.0, 2.0e7])

    # No MF=10 partials -> the chain is never consulted (empty chain is fine).
    table = _build_xs_table_pendf(["Fe56"], ["(n,gamma)"], edges, fake, Chain())

    assert table.reactions == ["(n,gamma)"]
    assert table.rxn_indices.tolist() == [0]
    np.testing.assert_allclose(table.xs_matrix[0], [2.0])


def test_unbound_partials_fall_back_and_warn():
    """MF=10 partials with no matching chain reaction -> MF=3 total + one
    summary warning directing the user at the chain patcher tool."""
    fake = _FakePendf(
        mf3={"Am241": {102: _const(5.0)}},
        mf10={"Am241": {102: {
            0: (None, _const(4.0)),
            2: (None, _const(1.0)),
        }}})
    edges = np.array([0.0, 2.0e7])
    # Both products unmappable -> the chain carries no reaction for either LFS.
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})

    with pytest.warns(UserWarning,
                      match="add_pendf_isomeric_branching_to_chain"):
        table = _build_xs_table_pendf(
            ["Am241"], ["(n,gamma)"], edges, fake, chain)

    # Fell back to the single MF=3-total row (no qualified names)
    assert table.reactions == ["(n,gamma)"]
    assert table.rxn_indices.tolist() == [0]
    np.testing.assert_allclose(table.xs_matrix[0], [5.0])


def test_partially_unbound_partials_fall_back():
    """If ANY partial is unbound, the whole reaction falls back to the MF=3
    total (mirrors the historical partially-mapped behavior)."""
    fake = _FakePendf(
        mf3={"Ir193": {102: _const(5.0)}},
        mf10={"Ir193": {102: {
            0: ("Ir194", _const(4.0)),   # ground: bound
            38: (None, _const(1.0)),     # metastable: unbound (ELIS-unmapped)
        }}})
    edges = np.array([0.0, 2.0e7])
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})

    with pytest.warns(UserWarning, match="depletion-chain reaction"):
        table = _build_xs_table_pendf(
            ["Ir193"], ["(n,gamma)"], edges, fake, chain)

    # Single MF=3-total row -- not the bound ground partial alone.
    assert table.reactions == ["(n,gamma)"]
    assert table.rxn_indices.tolist() == [0]
    np.testing.assert_allclose(table.xs_matrix[0], [5.0])


def test_ground_only_unbound_falls_back_silently(recwarn):
    """A ground-only MF=10 reaction (pathway LFS set == {0}) whose ground pathway
    is unbound falls back to the MF=3 total WITHOUT warning: the patcher leaves
    such a reaction stock in the chain by design, and the MF=3 total equals the
    LFS=0 partial, so the fallback warning would be pure noise (e.g. H2 (n,gamma)
    -> H3)."""
    fake = _FakePendf(
        mf3={"H2": {102: _const(5.5e-4)}},
        mf10={"H2": {102: {
            0: (None, _const(5.5e-4)),   # sole ground pathway, unmappable
        }}})
    edges = np.array([0.0, 2.0e7])
    # Product None -> the chain carries no reaction for LFS 0 (stock element).
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})

    table = _build_xs_table_pendf(["H2"], ["(n,gamma)"], edges, fake, chain)

    # Fell back to the single MF=3-total row, with no warning of any kind.
    assert table.reactions == ["(n,gamma)"]
    assert table.rxn_indices.tolist() == [0]
    np.testing.assert_allclose(table.xs_matrix[0], [5.5e-4])
    assert len(recwarn) == 0


def test_unbound_metastable_lfs_still_warns():
    """A reaction whose pathway LFS set contains a nonzero (metastable) LFS keeps
    the loud fallback warning even when the ground pathway is bound -- only the
    ground-only ({LFS=0}) class is silenced."""
    fake = _FakePendf(
        mf3={"Ir193": {102: _const(5.0)}},
        mf10={"Ir193": {102: {
            0: ("Ir194", _const(4.0)),   # ground: bound
            38: (None, _const(1.0)),     # metastable LFS: unbound -> loud warning
        }}})
    edges = np.array([0.0, 2.0e7])
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})

    with pytest.warns(UserWarning, match="depletion-chain reaction"):
        table = _build_xs_table_pendf(
            ["Ir193"], ["(n,gamma)"], edges, fake, chain)

    assert table.reactions == ["(n,gamma)"]
    np.testing.assert_allclose(table.xs_matrix[0], [5.0])


def test_lfs_less_chain_raises():
    """A chain whose QUALIFIED reaction carries pendf_lfs=None cannot bind
    partials -> hard error directing the user at the patcher tool."""
    fake = _FakePendf(
        mf3={"Am241": {102: _const(5.0)}},
        mf10={"Am241": {102: {
            0: ("Am242", _const(4.0)),
            2: ("Am242_m1", _const(1.0)),
        }}})
    edges = np.array([0.0, 2.0e7])

    # Chain has the qualified (n,gamma)_m1 but WITHOUT a pendf_lfs (legacy chain
    # built without LFS recording): partials cannot be bound by LFS.
    chain = Chain()
    nuc = Nuclide("Am241")
    nuc.add_reaction("(n,gamma)", "Am242", 0.0, 1.0, pendf_lfs=0)
    nuc.add_reaction("(n,gamma)_m1", "Am242_m1", 0.0, 1.0)  # pendf_lfs defaults None
    chain.add_nuclide(nuc)

    with pytest.raises(ValueError, match="add_pendf_isomeric_branching_to_chain"):
        _build_xs_table_pendf(["Am241"], ["(n,gamma)"], edges, fake, chain)


def test_consistency_check_warns_on_mismatch():
    """Sigma(partials) that disagree with the MF=3 total raise a warning."""
    fake = _FakePendf(
        mf3={"Am241": {102: _const(5.0)}},
        mf10={"Am241": {102: {
            0: ("Am242", _const(4.0)),
            2: ("Am242_m1", _const(0.5)),   # 4.0 + 0.5 = 4.5 != 5.0
        }}})
    edges = np.array([0.0, 2.0e7])
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})

    with pytest.warns(UserWarning, match="partials"):
        table = _build_xs_table_pendf(
            ["Am241"], ["(n,gamma)"], edges, fake, chain)

    # Rows are still emitted from the partials despite the warning
    assert table.reactions == ["(n,gamma)", "(n,gamma)_m1"]
    np.testing.assert_allclose(table.xs_matrix[0], [4.0])
    np.testing.assert_allclose(table.xs_matrix[1], [0.5])


def test_floor_dust_mismatch_does_not_warn(recwarn):
    """Groups where BOTH the total and the summed partials sit below the
    absolute floor are evaluator placeholder dust (e.g. JEFF-4.0's 1e-20 b
    "effective zero" floored independently per section, giving exact 2:1
    ratios): their relative deviation is meaningless and must not warn."""
    fake = _FakePendf(
        mf3={"Am241": {102: _const(1.0e-20)}},
        mf10={"Am241": {102: {
            0: ("Am242", _const(1.0e-20)),
            2: ("Am242_m1", _const(1.0e-20)),   # sum 2e-20 vs total 1e-20
        }}})
    edges = np.array([0.0, 2.0e7])
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})

    table = _build_xs_table_pendf(["Am241"], ["(n,gamma)"], edges, fake, chain)

    assert table.reactions == ["(n,gamma)", "(n,gamma)_m1"]
    assert len(recwarn) == 0


def test_meaningful_partials_vs_dust_total_still_warns():
    """A meaningful partial against a floor-dust total is a genuine
    inconsistency, not placeholder noise -- the floor exemption requires
    BOTH sides to be dust."""
    fake = _FakePendf(
        mf3={"Am241": {102: _const(1.0e-20)}},
        mf10={"Am241": {102: {
            0: ("Am242", _const(2.0)),          # 2 b vs dust total
            2: ("Am242_m1", _const(0.5)),
        }}})
    edges = np.array([0.0, 2.0e7])
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})

    with pytest.warns(UserWarning, match="partials"):
        _build_xs_table_pendf(["Am241"], ["(n,gamma)"], edges, fake, chain)


def test_matching_partials_do_not_warn(recwarn):
    """Partials that sum to the total within tolerance emit no warning."""
    fake = _FakePendf(
        mf3={"Am241": {102: _const(5.0)}},
        mf10={"Am241": {102: {
            0: ("Am242", _const(4.0)),
            2: ("Am242_m1", _const(1.0)),
        }}})
    edges = np.array([0.0, 2.0e7])
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})

    _build_xs_table_pendf(["Am241"], ["(n,gamma)"], edges, fake, chain)
    assert len(recwarn) == 0


# ---------------------------------------------------------------------------
# Reaction-list sanitation (qualified input == base input)
# ---------------------------------------------------------------------------

def test_default_pendf_reactions_drops_unknown_with_warning():
    """A chain-DEFAULTED reaction list strips _mN, dedupes, and drops channels
    with no REACTION_MT mapping (one summary warning) rather than crashing."""
    from openmc.deplete.microxs import _default_pendf_reactions

    chain = Chain()
    nuc = Nuclide("Fe56")
    nuc.add_reaction("(n,gamma)", "Fe57", 0.0, 1.0)
    nuc.add_reaction("(n,gamma)_m1", "Fe57_m1", 0.0, 1.0)  # qualified -> stripped
    nuc.add_reaction("(n,bogus)", "Xx999", 0.0, 1.0)       # no REACTION_MT entry
    chain.add_nuclide(nuc)

    with pytest.warns(UserWarning, match="REACTION_MT mapping"):
        kept = _default_pendf_reactions(chain)
    assert kept == ["(n,gamma)"]   # deduped base names, unknown dropped


def test_explicit_unknown_reaction_still_raises():
    """An EXPLICITLY passed unknown reaction still raises KeyError (only a
    chain-defaulted list tolerates unmappable names)."""
    fake = _FakePendf(mf3={"Fe56": {102: _const(2.0)}})
    edges = np.array([0.0, 2.0e7])
    with pytest.raises(KeyError):
        _build_xs_table_pendf(["Fe56"], ["(n,bogus)"], edges, fake, Chain())


def test_qualified_reaction_input_matches_base():
    """A qualified reaction in the input list is stripped/deduped, so it builds
    the SAME table as passing the base name (qualified names are outputs)."""
    fake = _FakePendf(
        mf3={"Am241": {102: _const(5.0)}},
        mf10={"Am241": {102: {
            0: ("Am242", _const(4.0)),
            2: ("Am242_m1", _const(1.0)),
        }}})
    edges = np.array([0.0, 2.0e7])
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})

    base = _build_xs_table_pendf(
        ["Am241"], ["(n,gamma)"], edges, fake, chain)
    qualified = _build_xs_table_pendf(
        ["Am241"], ["(n,gamma)_m1", "(n,gamma)"], edges, fake, chain)

    assert base.reactions == qualified.reactions
    np.testing.assert_array_equal(base.xs_matrix, qualified.xs_matrix)
    assert base.nuc_indices.tolist() == qualified.nuc_indices.tolist()
    assert base.rxn_indices.tolist() == qualified.rxn_indices.tolist()


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
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})

    micro = MicroXS.from_multigroup_flux(
        energies=edges, multigroup_flux=flux, chain_file=chain,
        nuclides=["Am241"], reactions=["(n,gamma)"], pendf_library=fake)

    assert isinstance(micro, MicroXS)
    assert "(n,gamma)" in micro.reactions
    assert "(n,gamma)_m1" in micro.reactions
    assert micro["Am241", "(n,gamma)"] == pytest.approx([4.0])
    assert micro["Am241", "(n,gamma)_m1"] == pytest.approx([1.0])

    # Conservation: the expanded rows sum to the MF=3 total collapsed with the
    # same flux weighting (sum_g sigma_g phi_g / sum_g phi_g).
    total_g = _group_average(
        *fake.xs("Am241", 102), np.asarray(edges, dtype=float))
    phi = np.asarray(flux, dtype=float)
    total = float(total_g @ phi / phi.sum())
    assert total == pytest.approx(5.0)
    assert (micro["Am241", "(n,gamma)"] + micro["Am241", "(n,gamma)_m1"]
            == pytest.approx([total]))


def test_from_multigroup_flux_pendf_requires_chain(monkeypatch):
    """The PENDF path without any resolvable chain raises a clear error."""
    import openmc
    monkeypatch.delitem(openmc.config, 'chain_file', raising=False)
    fake = _FakePendf(mf3={"Fe56": {102: _const(2.0)}})
    with pytest.raises(ValueError, match="requires chain_file"):
        MicroXS.from_multigroup_flux(
            energies=[0.0, 2.0e7], multigroup_flux=[1.0],
            nuclides=["Fe56"], reactions=["(n,gamma)"], pendf_library=fake)


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
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})

    # Part 1: the all-zero metastable partial is staged, so the axis carries the
    # qualified name with an all-zero row (ground first -> rxn 0, m1 -> rxn 1).
    table = _build_xs_table_pendf(["In115"], ["(n,gamma)"], edges, fake, chain)
    assert table.reactions == ["(n,gamma)", "(n,gamma)_m1"]
    assert table.nuc_indices.tolist() == [0, 0]
    assert table.rxn_indices.tolist() == [0, 1]
    np.testing.assert_array_equal(table.xs_matrix[0], [4.0, 4.0])   # ground
    np.testing.assert_array_equal(table.xs_matrix[1], [0.0, 0.0])   # m1 all-zero

    # Collapse: the qualified column is present in the MicroXS but all-zero.
    micro = MicroXS.from_multigroup_flux(
        energies=edges, multigroup_flux=[1.0, 1.0], chain_file=chain,
        nuclides=["In115"], reactions=["(n,gamma)"], pendf_library=fake)
    assert "(n,gamma)_m1" in micro.reactions
    assert micro["In115", "(n,gamma)"] == pytest.approx([4.0])
    assert micro["In115", "(n,gamma)_m1"] == pytest.approx([0.0])

    # Part 2: chain carries the qualified pathway; axis membership marks it
    # resolved, so the check must NOT raise (this was the false positive).
    consistency_chain = _chain({"In115": [("(n,gamma)", "In116"),
                                          ("(n,gamma)_m1", "In116_m1")]})
    _check_pathway_consistency(consistency_chain, micro)  # no raise
