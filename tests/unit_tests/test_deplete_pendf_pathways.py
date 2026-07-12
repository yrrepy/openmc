"""Unit tests for isomeric pathway (MF=10) expansion in the PENDF collapse path.

Covers the ORIGEN-style "Option A" per-product rows built by
``_build_xs_table_pendf`` (ground row keeps the canonical reaction name,
metastable products get an ``_m{n}`` suffix), the **chain-sourced** row naming
(each MF=10 ``LFS`` partial is bound to the depletion-chain reaction carrying
that ``pendf_lfs``), the demand-side chain semantics (extra library LFS ignored,
demanded-missing LFS falling back to the MF=3 total, the self-loop ground waiver),
the deplete-time chain<->MicroXS pathway-mismatch hard error, the chain provenance
stamp, the in-domain silence-fill of a placeholder ground, and the always-on
expansion through :meth:`MicroXS.from_multigroup_flux`. Pathway rows come
exclusively from MF=10 partial cross sections -- never from static branching
ratios; the product *names* come from the chain, never from the library.
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
    _silence_fill_ground,
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
# Chain-sourced MF=10 partials -> per-product rows
# ---------------------------------------------------------------------------

def test_pathway_rows_mapped_and_conservation(recwarn):
    """Chain-sourced MF=10 partials expand into per-product rows: the ``_mN`` suffix
    follows the PRODUCT LISO (via ``_liso_from_gnds``) recorded on the chain
    reaction bound to each LFS -- not the LFS value -- the ground keeps the base
    name, the partial rows conserve the MF=3 total, and a reaction with no MF=10
    yields the single canonical row. An exact chain<->library match is silent."""
    # _liso_from_gnds: the row-suffix authority is the product's isomeric state.
    assert _liso_from_gnds("Am242") == 0
    assert _liso_from_gnds("Am242_m1") == 1
    assert _liso_from_gnds("Ir192_m2") == 2

    # LFS is a level index, not an isomer ordinal: LFS {0, 2} map to Am242 (ground)
    # and Am242_m1, so the metastable row is (n,gamma)_m1 (product LISO), not _m2.
    fake = _FakePendf(
        mf3={"Am241": {102: _const(5.0)}},
        mf10={"Am241": {102: {
            0: ("Am242", _const(4.0)),
            2: ("Am242_m1", _const(1.0)),
        }}})
    edges = np.array([0.0, 1.0e7, 2.0e7])
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})

    table = _build_xs_table_pendf(["Am241"], ["(n,gamma)"], edges, fake, chain)

    # Expanded reaction axis: base name (ground) then ascending isomer; ground
    # staged first -> rxn 0, m1 -> rxn 1.
    assert table.reactions == ["(n,gamma)", "(n,gamma)_m1"]
    assert table.nuc_indices.tolist() == [0, 0]
    assert table.rxn_indices.tolist() == [0, 1]
    assert table.xs_matrix.dtype == np.float64
    np.testing.assert_allclose(table.xs_matrix[0], [4.0, 4.0])   # ground
    np.testing.assert_allclose(table.xs_matrix[1], [1.0, 1.0])   # m1

    # Conservation: the partial rows sum to the MF=3 total collapsed with the same
    # flat-in-bin kernel the table builder uses.
    total_g = _group_average(*fake.xs("Am241", 102), edges)
    np.testing.assert_allclose(total_g, [5.0, 5.0])
    np.testing.assert_allclose(table.xs_matrix[0] + table.xs_matrix[1], total_g)

    # A reaction with no MF=10 data at all yields one canonical (ground) row; the
    # chain is never consulted (an empty Chain is fine).
    nomf10 = _FakePendf(mf3={"Fe56": {102: _const(2.0)}})
    table = _build_xs_table_pendf(["Fe56"], ["(n,gamma)"],
                                  np.array([0.0, 2.0e7]), nomf10, Chain())
    assert table.reactions == ["(n,gamma)"]
    assert table.rxn_indices.tolist() == [0]
    np.testing.assert_allclose(table.xs_matrix[0], [2.0])

    # The exact chain<->library LFS match above emits per-product rows silently.
    assert len(recwarn) == 0


def test_duplicate_lfs_same_product_sums():
    """Two MF=10 levels mapping to the SAME product isomer SUM into a single
    qualified row (a level index is not the observable final state) -- present
    exactly once, never a last-wins overwrite, including when a zero partial is
    staged first (keep_zero) ahead of a nonzero duplicate."""
    edges = np.array([0.0, 1.0e7, 2.0e7])

    # Two nonzero levels (LFS 1, 2) both -> Am242_m1 (same IZAP -> designed sum,
    # not a collision): their 1-barn partials sum to a single 2-barn row.
    fake = _FakePendf(
        mf3={"Am241": {102: _const(2.0)}},
        mf10={"Am241": {102: {
            1: ("Am242_m1", _const(1.0)),
            2: ("Am242_m1", _const(1.0)),
        }}})
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})
    table = _build_xs_table_pendf(["Am241"], ["(n,gamma)"], edges, fake, chain)

    assert table.reactions == ["(n,gamma)", "(n,gamma)_m1"]
    assert table.reactions.count("(n,gamma)_m1") == 1        # name appears once
    m1_idx = table.reactions.index("(n,gamma)_m1")
    assert list(table.rxn_indices).count(m1_idx) == 1        # no duplicate to clobber
    row = next(table.xs_matrix[i]
               for i, r in enumerate(table.rxn_indices) if r == m1_idx)
    np.testing.assert_allclose(row, [2.0, 2.0])              # SUM of the two partials

    # End to end: the collapsed qualified row equals the summed 2 barns.
    micro = MicroXS.from_multigroup_flux(
        energies=edges, multigroup_flux=[1.0, 1.0], chain_file=chain,
        nuclides=["Am241"], reactions=["(n,gamma)"], pendf_library=fake)
    assert list(micro.reactions).count("(n,gamma)_m1") == 1
    assert micro["Am241", "(n,gamma)_m1"] == pytest.approx([2.0])

    # A zero metastable partial staged first then a nonzero duplicate for the same
    # product ends up summed and present exactly once.
    fake = _FakePendf(
        mf3={"Am241": {102: _const(3.0)}},
        mf10={"Am241": {102: {
            1: ("Am242_m1", _const(0.0)),   # zero, staged via keep_zero
            2: ("Am242_m1", _const(3.0)),   # nonzero duplicate -> sums in
        }}})
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})
    table = _build_xs_table_pendf(["Am241"], ["(n,gamma)"], edges, fake, chain)

    assert table.reactions == ["(n,gamma)", "(n,gamma)_m1"]
    m1_idx = table.reactions.index("(n,gamma)_m1")
    assert list(table.rxn_indices).count(m1_idx) == 1
    row = next(table.xs_matrix[i]
               for i, r in enumerate(table.rxn_indices) if r == m1_idx)
    np.testing.assert_allclose(row, [3.0, 3.0])


def test_pendf_collapse_hard_errors():
    """Two unrecoverable collapse inputs raise: a lumped multi-daughter channel
    (summing unrelated daughters into one row is never valid) and a qualified chain
    reaction carrying ``pendf_lfs=None`` (partials cannot be bound by LFS)."""
    edges = np.array([0.0, 2.0e7])

    # A lumped MT=5 (n,misc) whose MF=10 partials name two DIFFERENT daughters at
    # one collapse row (same LFS 0, distinct IZAP) is refused. Contrast the
    # same-daughter sum above, where two levels of ONE daughter do sum.
    fake = _FakePendf(
        mf3={"Fe56": {5: _const(5.0)}},
        mf10={"Fe56": {5: {
            (0, 26057): ("Fe57", _const(3.0)),   # one daughter
            (0, 25056): ("Mn56", _const(2.0)),   # a DIFFERENT daughter, same LFS
        }}})
    chain = _chain_from_fake(fake, {5: "(n,misc)"})
    with pytest.raises(ValueError, match="lumped"):
        _build_xs_table_pendf(["Fe56"], ["(n,misc)"], edges, fake, chain)

    # A qualified (n,gamma)_m1 WITHOUT a pendf_lfs (legacy chain built without LFS
    # recording): partials cannot be bound by LFS -> hard error naming the patcher.
    fake = _FakePendf(
        mf3={"Am241": {102: _const(5.0)}},
        mf10={"Am241": {102: {
            0: ("Am242", _const(4.0)),
            2: ("Am242_m1", _const(1.0)),
        }}})
    chain = Chain()
    nuc = Nuclide("Am241")
    nuc.add_reaction("(n,gamma)", "Am242", 0.0, 1.0, pendf_lfs=0)
    nuc.add_reaction("(n,gamma)_m1", "Am242_m1", 0.0, 1.0)  # pendf_lfs defaults None
    chain.add_nuclide(nuc)
    with pytest.raises(ValueError, match="add_pendf_isomeric_branching_to_chain"):
        _build_xs_table_pendf(["Am241"], ["(n,gamma)"], edges, fake, chain)


# ---------------------------------------------------------------------------
# Chain-demand-side pathway binding (the chain is the source of truth)
#
# NOTE: the runtime Sigma(partials)-vs-MF=3-total consistency warning was removed
# from ``_build_xs_table_pendf`` (Change 2); the build-time patcher audit is the
# authoritative diagnosis. These tests exercise the demand-side semantics only.
# ---------------------------------------------------------------------------

def test_qualified_demand_extra_library_lfs_ignored_silently(recwarn):
    """When the chain demands a subset of the library's LFS, the undemanded partials
    are dropped SILENTLY and only the demanded pathway rows are emitted -- shown for
    a single-ground demand with an extra metastable, a two-row demand with an extra
    level, and a non-contiguous subset demand (Sn122 (n,p))."""
    edges = np.array([0.0, 2.0e7])

    # (a) Chain demands ground {0} only; the library carries an extra metastable
    # LFS 38 -> the demanded ground partial (4.0), not the MF=3 total (5.0).
    fake = _FakePendf(
        mf3={"Ir193": {102: _const(5.0)}},
        mf10={"Ir193": {102: {
            0: ("Ir194", _const(4.0)),   # ground: chain binds LFS 0
            38: (None, _const(1.0)),     # metastable LFS: library extra, unmapped
        }}})
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})   # demands {0} only
    table = _build_xs_table_pendf(["Ir193"], ["(n,gamma)"], edges, fake, chain)
    assert table.reactions == ["(n,gamma)"]
    assert table.rxn_indices.tolist() == [0]
    np.testing.assert_allclose(table.xs_matrix[0], [4.0])

    # (b) Chain demands {0, 1}; library {0, 1, 4} -> the extra LFS 4 is dropped.
    fake = _FakePendf(
        mf3={"In115": {102: _const(5.0)}},
        mf10={"In115": {102: {
            0: ("In116", _const(3.0)),
            1: ("In116_m1", _const(1.0)),
            4: (None, _const(1.0)),      # library extra LFS, unmapped in chain
        }}})
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})   # demands {0, 1}
    table = _build_xs_table_pendf(["In115"], ["(n,gamma)"], edges, fake, chain)
    assert table.reactions == ["(n,gamma)", "(n,gamma)_m1"]
    np.testing.assert_allclose(table.xs_matrix[0], [3.0])   # ground partial
    np.testing.assert_allclose(table.xs_matrix[1], [1.0])   # m1 partial

    # (c) Sn122 (n,p) subset demand {0, 5}; library {0, 1, 5} -> the undemanded
    # LFS 1 is ignored and rows come from the LFS 0 and LFS 5 partials.
    fake = _FakePendf(
        mf3={"Sn122": {103: _const(6.0)}},
        mf10={"Sn122": {103: {
            0: ("In122", _const(4.0)),      # demanded ground product
            1: (None, _const(1.0)),         # undemanded extra LFS
            5: ("In122_m1", _const(1.0)),   # demanded metastable product
        }}})
    chain = _chain_from_fake(fake, {103: "(n,p)"})   # demands {0, 5}
    table = _build_xs_table_pendf(["Sn122"], ["(n,p)"], edges, fake, chain)
    assert table.reactions == ["(n,p)", "(n,p)_m1"]
    np.testing.assert_allclose(table.xs_matrix[0], [4.0])   # LFS 0 partial
    np.testing.assert_allclose(table.xs_matrix[1], [1.0])   # LFS 5 partial

    assert len(recwarn) == 0     # every drop above is silent


def test_stock_chain_emits_mf3_total_silently(recwarn):
    """A chain left STOCK for a reaction (no ``pendf_lfs`` pathway) emits the single
    MF=3 total row SILENTLY, regardless of the MF=10 partials the library carries --
    the chain is the demand side. Shown for both products unmappable, a sole
    unmappable ground (H2 (n,gamma) -> H3), and a plain stock reaction whose library
    DOES carry a metastable partial (the big new behavior)."""
    edges = np.array([0.0, 2.0e7])

    # (a) Both products unmappable -> _chain_from_fake builds a fully stock chain.
    fake = _FakePendf(
        mf3={"Am241": {102: _const(5.0)}},
        mf10={"Am241": {102: {
            0: (None, _const(4.0)),
            2: (None, _const(1.0)),
        }}})
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})
    table = _build_xs_table_pendf(["Am241"], ["(n,gamma)"], edges, fake, chain)
    assert table.reactions == ["(n,gamma)"]
    assert table.rxn_indices.tolist() == [0]
    np.testing.assert_allclose(table.xs_matrix[0], [5.0])

    # (b) A sole ground pathway that is unmappable (H2 (n,gamma) -> H3).
    fake = _FakePendf(
        mf3={"H2": {102: _const(5.5e-4)}},
        mf10={"H2": {102: {0: (None, _const(5.5e-4))}}})
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})
    table = _build_xs_table_pendf(["H2"], ["(n,gamma)"], edges, fake, chain)
    assert table.reactions == ["(n,gamma)"]
    assert table.rxn_indices.tolist() == [0]
    np.testing.assert_allclose(table.xs_matrix[0], [5.5e-4])

    # (c) A plain stock (n,gamma) (no pendf_lfs) whose library DOES carry a
    # metastable partial still emits the MF=3 total and stays silent.
    fake = _FakePendf(
        mf3={"In115": {102: _const(5.0)}},
        mf10={"In115": {102: {
            0: ("In116", _const(4.0)),
            1: ("In116_m1", _const(1.0)),   # library carries a metastable pathway
        }}})
    chain = Chain()
    nuc = Nuclide("In115")
    nuc.add_reaction("(n,gamma)", "In116", 0.0, 1.0)   # no pendf_lfs -> stock
    chain.add_nuclide(nuc)
    table = _build_xs_table_pendf(["In115"], ["(n,gamma)"], edges, fake, chain)
    assert table.reactions == ["(n,gamma)"]
    assert table.rxn_indices.tolist() == [0]
    np.testing.assert_allclose(table.xs_matrix[0], [5.0])

    assert len(recwarn) == 0     # a by-design stock reaction never warns


def test_demanded_lfs_missing_falls_back_and_warns():
    """A demanded LFS missing from the library falls back to the MF=3 total and is
    collected into one summary warning naming BOTH LFS sets -- for a missing
    metastable (chain {0,1} vs library {0}), a total absence of MF=10 (library {}),
    and a non-self-loop ground missing (library {1}; the self-loop waiver does NOT
    apply because the ground target != parent). The value is always the MF=3 total,
    identical to the old silent fallback; only the warning is new."""
    edges = np.array([0.0, 2.0e7])

    # (a) Chain demands {0, 1}; the library has only the ground LFS 0.
    fake = _FakePendf(
        mf3={"Ir193": {102: _const(5.0)}},
        mf10={"Ir193": {102: {0: ("Ir194", _const(4.0))}}})
    chain = Chain()
    nuc = Nuclide("Ir193")
    nuc.add_reaction("(n,gamma)", "Ir194", 0.0, 1.0, pendf_lfs=0)
    nuc.add_reaction("(n,gamma)_m1", "Ir194_m1", 0.0, 1.0, pendf_lfs=1)
    chain.add_nuclide(nuc)
    with pytest.warns(UserWarning, match="chain and library disagree") as record:
        table = _build_xs_table_pendf(["Ir193"], ["(n,gamma)"], edges, fake, chain)
    msg = str(record[0].message)
    assert "chain LFS {0, 1}" in msg
    assert "library LFS {0}" in msg
    assert table.reactions == ["(n,gamma)"]
    np.testing.assert_allclose(table.xs_matrix[0], [5.0])

    # (b) Chain demands {0, 1} with NO library MF=10 at all -> library set empty.
    fake = _FakePendf(mf3={"In115": {102: _const(5.0)}})
    chain = Chain()
    nuc = Nuclide("In115")
    nuc.add_reaction("(n,gamma)", "In116", 0.0, 1.0, pendf_lfs=0)
    nuc.add_reaction("(n,gamma)_m1", "In116_m1", 0.0, 1.0, pendf_lfs=1)
    chain.add_nuclide(nuc)
    with pytest.warns(UserWarning, match="does not match") as record:
        table = _build_xs_table_pendf(["In115"], ["(n,gamma)"], edges, fake, chain)
    msg = str(record[0].message)
    assert "chain LFS {0, 1}" in msg
    assert "library LFS {}" in msg
    assert table.reactions == ["(n,gamma)"]
    assert table.rxn_indices.tolist() == [0]
    np.testing.assert_allclose(table.xs_matrix[0], [5.0])

    # (c) Demanded ground missing whose target != parent (NOT a self-loop) -> warns.
    fake = _FakePendf(
        mf3={"In115": {102: _const(5.0)}},
        mf10={"In115": {102: {1: ("In116_m1", _const(1.0))}}})   # ground missing
    chain = Chain()
    nuc = Nuclide("In115")
    nuc.add_reaction("(n,gamma)", "In116", 0.0, 1.0, pendf_lfs=0)
    nuc.add_reaction("(n,gamma)_m1", "In116_m1", 0.0, 1.0, pendf_lfs=1)
    chain.add_nuclide(nuc)
    with pytest.warns(UserWarning, match="does not match") as record:
        table = _build_xs_table_pendf(["In115"], ["(n,gamma)"], edges, fake, chain)
    msg = str(record[0].message)
    assert "chain LFS {0, 1}" in msg
    assert "library LFS {1}" in msg
    assert table.reactions == ["(n,gamma)"]
    np.testing.assert_allclose(table.xs_matrix[0], [5.0])


def test_self_loop_ground_waiver_stages_base_from_total(recwarn):
    """Self-loop ground waiver: In115 (n,n') chain demands {0, 1} with the ground a
    self-loop (target == parent) but the tape carries only the metastable LFS 1
    (JEFF In113/In115 behavior). The base row is staged from the MF=3 total and the
    m1 row from its partial -- no fallback, no warning (restores In115m (n,n')
    production)."""
    fake = _FakePendf(
        mf3={"In115": {4: _const(2.0)}},
        mf10={"In115": {4: {1: ("In115_m1", _const(0.8))}}})   # only the metastable
    edges = np.array([0.0, 2.0e7])
    chain = Chain()
    nuc = Nuclide("In115")
    nuc.add_reaction("(n,n')", "In115", 0.0, 1.0, pendf_lfs=0)         # self-loop
    nuc.add_reaction("(n,n')_m1", "In115_m1", 0.0, 1.0, pendf_lfs=1)
    chain.add_nuclide(nuc)

    table = _build_xs_table_pendf(["In115"], ["(n,n')"], edges, fake, chain)

    assert table.reactions == ["(n,n')", "(n,n')_m1"]
    rows = {r: table.xs_matrix[i] for i, r in enumerate(table.rxn_indices)}
    np.testing.assert_allclose(rows[0], [2.0])   # base row == MF=3 total
    np.testing.assert_allclose(rows[1], [0.8])   # m1 row == LFS 1 partial
    assert len(recwarn) == 0


# ---------------------------------------------------------------------------
# Reaction-list sanitation (qualified input == base input)
# ---------------------------------------------------------------------------

def test_reaction_list_sanitation():
    """Reaction-list sanitation: a chain-DEFAULTED list strips ``_mN``, dedupes, and
    drops channels with no REACTION_MT mapping (one warning); an EXPLICIT unknown
    still raises KeyError; and a qualified input builds the SAME table as the base
    name (qualified names are outputs, not inputs)."""
    from openmc.deplete.microxs import _default_pendf_reactions

    edges = np.array([0.0, 2.0e7])

    # (a) Chain-defaulted list: strip _mN, dedupe, drop unmappable with a warning.
    chain = Chain()
    nuc = Nuclide("Fe56")
    nuc.add_reaction("(n,gamma)", "Fe57", 0.0, 1.0)
    nuc.add_reaction("(n,gamma)_m1", "Fe57_m1", 0.0, 1.0)  # qualified -> stripped
    nuc.add_reaction("(n,bogus)", "Xx999", 0.0, 1.0)       # no REACTION_MT entry
    chain.add_nuclide(nuc)
    with pytest.warns(UserWarning, match="REACTION_MT mapping"):
        kept = _default_pendf_reactions(chain)
    assert kept == ["(n,gamma)"]   # deduped base names, unknown dropped

    # (b) An explicitly passed unknown reaction still raises KeyError.
    fake = _FakePendf(mf3={"Fe56": {102: _const(2.0)}})
    with pytest.raises(KeyError):
        _build_xs_table_pendf(["Fe56"], ["(n,bogus)"], edges, fake, Chain())

    # (c) A qualified reaction in the input is stripped/deduped -> same table as
    # passing the base name.
    fake = _FakePendf(
        mf3={"Am241": {102: _const(5.0)}},
        mf10={"Am241": {102: {
            0: ("Am242", _const(4.0)),
            2: ("Am242_m1", _const(1.0)),
        }}})
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})
    base = _build_xs_table_pendf(["Am241"], ["(n,gamma)"], edges, fake, chain)
    qualified = _build_xs_table_pendf(
        ["Am241"], ["(n,gamma)_m1", "(n,gamma)"], edges, fake, chain)
    assert base.reactions == qualified.reactions
    np.testing.assert_array_equal(base.xs_matrix, qualified.xs_matrix)
    assert base.nuc_indices.tolist() == qualified.nuc_indices.tolist()
    assert base.rxn_indices.tolist() == qualified.rxn_indices.tolist()


# ---------------------------------------------------------------------------
# End-to-end through from_multigroup_flux
# ---------------------------------------------------------------------------

def test_from_multigroup_flux_pathways_end_to_end_and_requires_chain(monkeypatch):
    """The public :meth:`from_multigroup_flux` entry expands MF=10 pathways with
    flux weighting and conserves the MF=3 total; without any resolvable chain the
    PENDF path raises a clear error."""
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

    # Conservation: the expanded rows sum to the MF=3 total collapsed with the same
    # flux weighting (sum_g sigma_g phi_g / sum_g phi_g).
    total_g = _group_average(
        *fake.xs("Am241", 102), np.asarray(edges, dtype=float))
    phi = np.asarray(flux, dtype=float)
    total = float(total_g @ phi / phi.sum())
    assert total == pytest.approx(5.0)
    assert (micro["Am241", "(n,gamma)"] + micro["Am241", "(n,gamma)_m1"]
            == pytest.approx([total]))

    # The PENDF path without any resolvable chain raises.
    import openmc
    monkeypatch.delitem(openmc.config, 'chain_file', raising=False)
    nochain = _FakePendf(mf3={"Fe56": {102: _const(2.0)}})
    with pytest.raises(ValueError, match="requires chain_file"):
        MicroXS.from_multigroup_flux(
            energies=[0.0, 2.0e7], multigroup_flux=[1.0],
            nuclides=["Fe56"], reactions=["(n,gamma)"], pendf_library=nochain)


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


def test_check_pathway_consistency():
    """The deplete-time chain<->MicroXS pathway check (plan 1.1): a qualified row on
    either side that the other cannot route is a hard error naming the offending
    pair, while matching qualified reactions, a plain non-pathway diff (nuclides on
    only one side, unqualified reaction diffs), and a chain-qualified nuclide the
    MicroXS carries no data for all pass untouched."""
    # (a) MicroXS carries a qualified row the chain cannot route -> raise.
    chain = _chain({"In115": [("(n,gamma)", "In116")]})  # no _m1 in chain
    micro = _micro({"In115": ["(n,gamma)", "(n,gamma)_m1"]},
                   ["(n,gamma)", "(n,gamma)_m1"])
    with pytest.raises(ValueError, match="pathway mismatch") as exc:
        _check_pathway_consistency(chain, micro)
    assert "In115 (n,gamma)_m1" in str(exc.value)

    # (b) Chain carries a qualified pathway the MicroXS lacks (base present) -> raise.
    chain = _chain({"In115": [("(n,gamma)", "In116"),
                              ("(n,gamma)_m1", "In116_m1")]})
    micro = _micro({"In115": ["(n,gamma)"]}, ["(n,gamma)"])  # base only
    with pytest.raises(ValueError, match="pathway mismatch") as exc:
        _check_pathway_consistency(chain, micro)
    assert "In115 (n,gamma)_m1" in str(exc.value)

    # (c) Matching qualified reactions on both sides -> no error.
    chain = _chain({"In115": [("(n,gamma)", "In116"),
                              ("(n,gamma)_m1", "In116_m1")]})
    micro = _micro({"In115": ["(n,gamma)", "(n,gamma)_m1"]},
                   ["(n,gamma)", "(n,gamma)_m1"])
    _check_pathway_consistency(chain, micro)  # no raise

    # (d) No qualified names anywhere -> untouched, even with unqualified diffs and
    # nuclides present on only one side.
    chain = _chain({"Fe56": [("(n,gamma)", "Fe57"), ("(n,p)", "Mn56")],
                    "Cs137": [("(n,gamma)", "Cs138")]})  # Cs137 only in chain
    micro = _micro({"Fe56": ["(n,gamma)"],            # (n,p) only in chain: ok
                    "W186": ["(n,gamma)"]},           # W186 only in MicroXS: ok
                   ["(n,gamma)", "(n,p)"])
    _check_pathway_consistency(chain, micro)  # no raise

    # (e) Chain qualified but MicroXS carries no data for the nuclide (base absent)
    # -> a no-data nuclide, not a mismatch.
    chain = _chain({"In115": [("(n,gamma)", "In116"),
                              ("(n,gamma)_m1", "In116_m1")]})
    micro = _micro({"In115": []}, ["(n,gamma)"])  # In115 present but all-zero
    _check_pathway_consistency(chain, micro)  # no raise


def test_zero_metastable_partial_stages_and_passes():
    """Regression: a metastable MF=10 partial that group-averages to exactly zero
    (its threshold is above the tally groups) must still stage a zero row, so the
    reaction axis carries the qualified name (Part 1) and the chain<->MicroXS
    consistency check does not raise a false positive (Part 2).

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


# ---------------------------------------------------------------------------
# Chain provenance stamp verification at collapse (§chain-provenance stamp)
# ---------------------------------------------------------------------------

class _IdentifiedPendf(_FakePendf):
    """A ``_FakePendf`` that also carries a ``library`` identity string.

    ``_FakePendf`` deliberately exposes no ``library`` attr, so the provenance
    stamp check skips it (duck-typed libraries without identity are unverifiable).
    This subclass adds the identity so the mismatch path can be exercised;
    ``nuclides`` (the collapse interface) already supplies the count.
    """

    def __init__(self, mf3, mf10=None, library="TENDL-2017", source_identity=None):
        super().__init__(mf3, mf10)
        self.library = library
        # Tape-derived provenance identity (Change 3e); ``None`` mimics an old h5
        # that carries only the user ``library`` label.
        self.source_identity = source_identity


def _stamped_am241_setup(library="TENDL-2017"):
    """Return (fake, chain, edges, flux) for a 1-nuclide (Am241) collapse.

    The chain binds LFS {0, 2} exactly to the library's partials (no LFS-set
    mismatch, no consistency warning), so any warning captured is the provenance
    stamp's. The chain is stamped by the caller.
    """
    fake = _IdentifiedPendf(
        mf3={"Am241": {102: _const(5.0)}},
        mf10={"Am241": {102: {
            0: ("Am242", _const(4.0)),
            2: ("Am242_m1", _const(1.0)),
        }}}, library=library)
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})
    return fake, chain, [0.0, 1.0e7, 2.0e7], [1.0, 1.0]


def _collapse(fake, chain, edges, flux):
    return MicroXS.from_multigroup_flux(
        energies=edges, multigroup_flux=flux, chain_file=chain,
        nuclides=["Am241"], reactions=["(n,gamma)"], pendf_library=fake)


def _provenance_warnings(record):
    return [w for w in record if "provenance mismatch" in str(w.message)]


def test_stamp_silent_cases(recwarn):
    """Chain provenance-stamp verification is silent whenever the stamp is
    trustworthy or unverifiable: a matching library string+count, a source rename
    only, a duck-typed library without identity (even a wildly wrong stamp), an
    unstamped chain, a stamp matching the tape ``source_identity`` (even when the
    user ``library`` label differs), and a stamp matching the legacy library label
    when no source_identity exists."""
    # (a) A stamp matching the library on both string and count.
    fake, chain, edges, flux = _stamped_am241_setup()
    chain.root_attrs = {'pendf_source': 'tendl2017.h5',
                        'pendf_library': 'TENDL-2017', 'pendf_nuclides': '1'}
    _collapse(fake, chain, edges, flux)

    # (b) A differing pendf_source alone (a file rename) is never a trigger.
    fake, chain, edges, flux = _stamped_am241_setup()
    chain.root_attrs = {'pendf_source': 'renamed-copy.h5',   # only this differs
                        'pendf_library': 'TENDL-2017', 'pendf_nuclides': '1'}
    _collapse(fake, chain, edges, flux)

    # (c) A library with no ``library`` attr (plain _FakePendf) skips the check.
    plain = _FakePendf(
        mf3={"Am241": {102: _const(5.0)}},
        mf10={"Am241": {102: {
            0: ("Am242", _const(4.0)),
            2: ("Am242_m1", _const(1.0)),
        }}})
    plain_chain = _chain_from_fake(plain, {102: "(n,gamma)"})
    plain_chain.root_attrs = {'pendf_source': 'anything.h5',
                              'pendf_library': 'SOME-OTHER-LIB',
                              'pendf_nuclides': '999'}
    _collapse(plain, plain_chain, [0.0, 1.0e7, 2.0e7], [1.0, 1.0])

    # (d) An unstamped chain (the pre-stamp default) never warns.
    fake, chain, edges, flux = _stamped_am241_setup()
    assert chain.root_attrs == {}          # _chain_from_fake leaves it unstamped
    _collapse(fake, chain, edges, flux)

    # (e) The stamp is a tape-derived identity: a library exposing a matching
    # ``source_identity`` verifies even when its user ``library`` label differs.
    fake, chain, edges, flux = _stamped_am241_setup(library="jeff40-user-label")
    fake.source_identity = "JEFF-4.0 Incident Neutron File"
    chain.root_attrs = {'pendf_source': 'jeff40-n/pendf',
                        'pendf_library': 'JEFF-4.0 Incident Neutron File',
                        'pendf_nuclides': '1'}
    _collapse(fake, chain, edges, flux)

    # (f) An old h5 with no source_identity verifies against the stamp when it
    # matches the user ``library`` label (backward compatible).
    fake, chain, edges, flux = _stamped_am241_setup(library="TENDL-2017")
    assert getattr(fake, 'source_identity', None) is None
    chain.root_attrs = {'pendf_source': 'tendl2017.h5',
                        'pendf_library': 'TENDL-2017', 'pendf_nuclides': '1'}
    _collapse(fake, chain, edges, flux)

    assert _provenance_warnings(recwarn) == []


def test_stamp_mismatch_warns():
    """A stamp that names a different library string, a different nuclide count, or
    matches NEITHER the tape ``source_identity`` NOR the library label fires one
    provenance warning naming both sides (and, for the string case, the source
    basename and the patcher tool)."""
    # (a) Different library string (count matches) -> names both + source + tool.
    fake, chain, edges, flux = _stamped_am241_setup(library="TENDL-2017")
    chain.root_attrs = {'pendf_source': 'jeff40.h5',
                        'pendf_library': 'JEFF-4.0',   # != library in use
                        'pendf_nuclides': '1'}          # count matches
    with pytest.warns(UserWarning, match="provenance mismatch") as record:
        _collapse(fake, chain, edges, flux)
    msg = str(_provenance_warnings(record)[0].message)
    assert "JEFF-4.0" in msg                     # stamp's library
    assert "TENDL-2017" in msg                   # library in use
    assert "jeff40.h5" in msg                    # source basename included
    assert "add_pendf_isomeric_branching_to_chain" in msg

    # (b) Different nuclide count even when the string matches.
    fake, chain, edges, flux = _stamped_am241_setup()
    chain.root_attrs = {'pendf_source': 'tendl2017.h5',
                        'pendf_library': 'TENDL-2017',  # matches
                        'pendf_nuclides': '593'}         # != 1 in use
    with pytest.warns(UserWarning, match="provenance mismatch") as record:
        _collapse(fake, chain, edges, flux)
    msg = str(_provenance_warnings(record)[0].message)
    assert "593 nuclides" in msg
    assert "1 nuclides" in msg

    # (c) The stamp matches neither source_identity nor the library label.
    fake, chain, edges, flux = _stamped_am241_setup(library="TENDL-2017")
    fake.source_identity = "TENDL-2017 pointwise"
    chain.root_attrs = {'pendf_source': 'jeff40-n/pendf',
                        'pendf_library': 'JEFF-4.0 Incident Neutron File',
                        'pendf_nuclides': '1'}
    with pytest.warns(UserWarning, match="provenance mismatch") as record:
        _collapse(fake, chain, edges, flux)
    msg = str(_provenance_warnings(record)[0].message)
    assert "JEFF-4.0 Incident Neutron File" in msg          # stamped identity
    assert "TENDL-2017" in msg                              # a compared identity


# ---------------------------------------------------------------------------
# In-domain pointwise silence-fill of the ground pathway
#
# For a qualified (n,gamma)-style reaction whose MF=10 branching is a thermal
# placeholder (every partial ~1e-20 b while the MF=3 total carries the real 1/v
# capture), the ground row is filled by balance (total - demanded metastables)
# wherever the branching is silent, inside the LFS=0 partial's own range. A
# channel whose branching is live wherever the total is does not fire (a provable
# no-op); grouped libraries are detected, not filled.
# ---------------------------------------------------------------------------

_PH = 1.0e-20                     # evaluator "effective zero" placeholder (barn)

_DATA = Path("/home/perry/Projects/OMC_Development/PENDF/data")
_JEFF_V1 = _DATA / "byPerry_v1" / "JEFF40-IST.293K.PENDF.h5"
_ENDFB81 = _DATA / "byPerry_v0" / "ENDFB81-IST.elis_mapped.293K.PENDF.h5"


def _row(table, rxn_name, nuc_idx=0):
    """Return the staged xs vector for ``(nuc_idx, rxn_name)``, or ``None``."""
    ri = table.reactions.index(rxn_name)
    for i in range(len(table.nuc_indices)):
        if table.nuc_indices[i] == nuc_idx and table.rxn_indices[i] == ri:
            return table.xs_matrix[i]
    return None


class _FakeGroupedPendf:
    """Minimal grouped-library stand-in (triggers the grouped collapse path).

    Exposes ``group_edges`` plus ``xs_g`` / ``pathway_xs_g`` returning pre-binned
    group arrays. The ``_mf10`` product slot mirrors :class:`_FakePendf` so
    :func:`_chain_from_fake` can synthesize the matching chain reactions.
    """

    def __init__(self, group_edges, mf3_g, mf10_g):
        # mf3_g:  {nuclide: {mt: [xs_g...]}}
        # mf10_g: {nuclide: {mt: {lfs or (lfs, izap): (product|None, [xs_g...])}}}
        self.group_edges = np.asarray(group_edges, dtype=float)
        self._mf3 = mf3_g
        self._mf10 = {
            nuc: {mt: {(k if isinstance(k, tuple) else (k, 0)): v
                       for k, v in parts.items()}
                  for mt, parts in by_mt.items()}
            for nuc, by_mt in mf10_g.items()}

    @property
    def nuclides(self):
        return list(self._mf3)

    def reactions(self, nuclide):
        return list(self._mf3[nuclide])

    def xs_g(self, nuclide, mt):
        return np.asarray(self._mf3[nuclide][mt], dtype=float)

    def pathways(self, nuclide, mt):
        return sorted(self._mf10.get(nuclide, {}).get(mt, {}))

    def pathway_xs_g(self, nuclide, mt, lfs, izap=None):
        return np.asarray(self._mf10[nuclide][mt][(lfs, izap)][1], dtype=float)


def _qualified_chain_for(lib, nuc, mt, base):
    """Chain demanding a real library's full LFS set for one reaction.

    Ground (LFS 0) -> ``base`` (synthetic non-parent daughter, so no self-loop);
    each LFS>0 -> ``base_m{k}`` (k ascending). Row identities only -- the cross
    sections come from the library.
    """
    chain = Chain()
    nuclide = Nuclide(nuc)
    order = 0
    for lfs, _izap in lib.pathways(nuc, mt):
        if lfs == 0:
            nuclide.add_reaction(base, f"{nuc}_g", 0.0, 1.0, pendf_lfs=0)
        else:
            order += 1
            nuclide.add_reaction(f"{base}_m{order}", f"{nuc}_x{order}",
                                 0.0, 1.0, pendf_lfs=lfs)
    chain.add_nuclide(nuclide)
    return chain


def test_silence_fill_fires_and_recovers_thermal_gap():
    """The in-domain silence-fill FIRES for a placeholder ground: below the
    branching onset the ground is filled to ~total, a straddling group blends the
    filled and source-faithful values, and above onset it stays source-faithful --
    but the fill never extends past the LFS=0 partial's last tabulated point, so a
    terminal interval where only the MF=3 total is live is excluded."""
    # (a) Placeholder below ~1 eV; branching live at/above 10 eV (ground 0.6x total,
    # m 0.4x). Below onset -> ~total; above -> source-faithful split; straddle blends.
    grid = np.array([1e-5, 1e-3, 0.1, 1.0, 10.0, 1e3, 1e6, 2e7])
    total = np.array([100.0, 30.0, 3.0, 1.0, 5.0, 4.0, 2.0, 1.0])
    ground = np.array([_PH, _PH, _PH, _PH, 3.0, 2.4, 1.2, 0.6])
    meta = np.array([_PH, _PH, _PH, _PH, 2.0, 1.6, 0.8, 0.4])
    fake = _FakePendf(
        mf3={"G1": {102: (grid, total)}},
        mf10={"G1": {102: {0: ("H1", (grid, ground)),
                           1: ("H1_m1", (grid, meta))}}})
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})

    fill = _silence_fill_ground(fake.pathways, fake.pathway_xs, "G1", 102,
                                grid, total, {0, 1})
    assert fill.fired is True

    edges = np.array([1e-5, 0.625, 1e6, 2e7])   # thermal / straddle / fast
    table = _build_xs_table_pendf(["G1"], ["(n,gamma)"], edges, fake, chain)
    ground_row = _row(table, "(n,gamma)")
    total_g = _group_average(grid, total, edges)
    srcfaithful = _group_average(grid, ground, edges)
    # thermal group (all below onset): filled to ~total (metastable ~0 there)
    assert ground_row[0] == pytest.approx(total_g[0], rel=1e-9)
    # straddle group blends: above source-faithful but below total
    assert srcfaithful[1] < ground_row[1] < total_g[1]
    # fast group (fully above onset): source-faithful split, well below total
    assert ground_row[2] < total_g[2]
    # the collapse used the fill's union-grid ground exactly
    np.testing.assert_allclose(
        ground_row, _group_average(fill.e_dom, fill.ground_dom, edges))

    # (b) In-domain terminal: the partials END before the MF=3 total. The fill
    # recovers the thermal ground but NEVER the terminal interval above the LFS=0
    # grid's last point, where the MF=3 total is still live.
    pgrid = np.array([1e-5, 1e-3, 0.1, 1.0, 10.0, 1e6, 1e7])       # partials -> 1e7
    tgrid = np.array([1e-5, 1e-3, 0.1, 1.0, 10.0, 1e6, 1e7, 2e7])  # MF3 -> 2e7
    total = np.array([100.0, 30.0, 3.0, 1.0, 5.0, 2.0, 1.0, 1.0])
    ground = np.array([_PH, _PH, _PH, _PH, 3.0, 1.2, 0.6])
    meta = np.array([_PH, _PH, _PH, _PH, 2.0, 0.8, 0.4])
    fake = _FakePendf(
        mf3={"T4": {102: (tgrid, total)}},
        mf10={"T4": {102: {0: ("U4", (pgrid, ground)),
                           1: ("U4_m1", (pgrid, meta))}}})
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})

    fill = _silence_fill_ground(fake.pathways, fake.pathway_xs, "T4", 102,
                                tgrid, total, {0, 1})
    assert fill.fired is True
    assert fill.ground0_range == (1e-5, 1e7)   # never past LFS=0's last point
    assert fill.e_dom.max() == 1e7

    edges = np.array([1e-5, 0.625, 1e6, 2e7])
    table = _build_xs_table_pendf(["T4"], ["(n,gamma)"], edges, fake, chain)
    ground_row = _row(table, "(n,gamma)")
    total_g = _group_average(tgrid, total, edges)
    assert ground_row[-1] < total_g[-1]                          # terminal excluded
    assert ground_row[0] == pytest.approx(total_g[0], rel=1e-9)  # thermal recovered


def test_silence_fill_never_fills_when_source_faithful():
    """The silence-fill is a provable no-op wherever the branching is source-
    faithful: a clean channel (branching live everywhere the total is), a Class-4
    genuine live deficit (Sigma(all)/total ~ 0.8), and an undemanded LFS that is
    live where the demanded partials are silent (its cross section must not be
    absorbed into ground). In every case fill.fired is False and the ground row is
    the raw LFS=0 partial average."""
    # (a) Clean channel: ground 0.7x, m 0.3x -- live everywhere -> never fires.
    grid = np.array([1.0e-5, 1.0, 1.0e3, 1.0e6, 2.0e7])
    total = np.array([10.0, 8.0, 4.0, 2.0, 1.0])
    ground, meta = total * 0.7, total * 0.3
    fake = _FakePendf(
        mf3={"C1": {102: (grid, total)}},
        mf10={"C1": {102: {0: ("D1", (grid, ground)),
                           1: ("D1_m1", (grid, meta))}}})
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})
    edges = np.array([1.0e-5, 0.625, 1.0e3, 2.0e7])
    fill = _silence_fill_ground(fake.pathways, fake.pathway_xs, "C1", 102,
                                grid, total, {0, 1})
    assert fill.fired is False
    table = _build_xs_table_pendf(["C1"], ["(n,gamma)"], edges, fake, chain)
    np.testing.assert_array_equal(_row(table, "(n,gamma)"),
                                  _group_average(grid, ground, edges))

    # (b) Class-4 live deficit: Sigma(all)/total ~ 0.8 (a real missing isomer / MT5
    # lumping), NOT silence -> never fires; the ground stays source-faithful.
    grid = np.array([1e-5, 1.0, 1e3, 1e6, 2e7])
    total = np.array([50.0, 20.0, 5.0, 2.0, 1.0])
    ground, meta = total * 0.5, total * 0.3        # Sigma(all) = 0.8*total
    fake = _FakePendf(
        mf3={"C4": {102: (grid, total)}},
        mf10={"C4": {102: {0: ("D4", (grid, ground)),
                           1: ("D4_m1", (grid, meta))}}})
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})
    edges = np.array([1e-5, 0.625, 2e7])
    fill = _silence_fill_ground(fake.pathways, fake.pathway_xs, "C4", 102,
                                grid, total, {0, 1})
    assert fill.fired is False
    table = _build_xs_table_pendf(["C4"], ["(n,gamma)"], edges, fake, chain)
    np.testing.assert_array_equal(_row(table, "(n,gamma)"),
                                  _group_average(grid, ground, edges))

    # (c) Undemanded live partial: ground + demanded m1 are placeholder at thermal,
    # but an UNDEMANDED partial (lfs 2, product None) is LIVE there. The silence test
    # sums ALL partials -> Sigma(all)/total ~ 1 -> reads "live" -> not filled, so the
    # undemanded cross section is not absorbed into ground.
    grid = np.array([1e-5, 1e-3, 0.1, 1.0, 10.0, 1e3, 2e7])
    total = np.array([100.0, 30.0, 3.0, 1.0, 5.0, 4.0, 1.0])
    ground = np.array([_PH, _PH, _PH, _PH, 3.0, 2.4, 0.6])
    meta = np.array([_PH, _PH, _PH, _PH, 2.0, 1.6, 0.4])
    und = np.array([100.0, 30.0, 3.0, 1.0, _PH, _PH, _PH])   # live where others silent
    fake = _FakePendf(
        mf3={"U5": {102: (grid, total)}},
        mf10={"U5": {102: {0: ("V5", (grid, ground)),
                           1: ("V5_m1", (grid, meta)),
                           2: (None, (grid, und))}}})          # product None -> undemanded
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})
    demanded = {lfs for lfs, _ in fake.pathways("U5", 102)
                if fake._mf10["U5"][102][(lfs, 0)][0] is not None}
    assert demanded == {0, 1}                                 # lfs 2 undemanded
    fill = _silence_fill_ground(fake.pathways, fake.pathway_xs, "U5", 102,
                                grid, total, demanded)
    assert fill.fired is False
    edges = np.array([1e-5, 0.625, 2e7])
    table = _build_xs_table_pendf(["U5"], ["(n,gamma)"], edges, fake, chain)
    total_g = _group_average(grid, total, edges)
    # thermal ground stays the source-faithful placeholder, NOT inflated to total
    assert _row(table, "(n,gamma)")[0] < 1e-6 * total_g[0]


def test_silence_fill_grouped_detection_warns_not_filled():
    """A grouped library carrying a placeholder ground (thermal group: total
    significant, every partial silent) is DETECTED (one warning) but NOT filled --
    the grouped fill is a build-time bake, not a collapse-time operation."""
    edges = np.array([1e-5, 0.625, 2e7])                      # thermal / rest
    fake = _FakeGroupedPendf(
        edges,
        mf3_g={"Gg": {102: [50.0, 1.0]}},
        mf10_g={"Gg": {102: {0: ("Hg", [_PH, 0.6]),
                             1: ("Hg_m1", [_PH, 0.4])}}})
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})
    with pytest.warns(UserWarning, match="unfilled placeholder ground"):
        table = _build_xs_table_pendf(["Gg"], ["(n,gamma)"], edges, fake, chain)
    np.testing.assert_array_equal(_row(table, "(n,gamma)"), np.array([_PH, 0.6]))


@pytest.mark.skipif(not _JEFF_V1.exists(),
                    reason="JEFF-4.0 v1 pointwise PENDF h5 not available")
def test_silence_fill_real_jeff_rh102_thermal_gap():
    # Rh102 (n,gamma) in JEFF-4.0: MF=10 ground/m1 are ~1e-20 b placeholders at
    # thermal while the MF=3 total carries ~35 b of 1/v capture -> the fill must
    # recover it into the ground row below the ~1.08 eV onset.
    import openmc.data as od
    lib = od.PendfLibrary(_JEFF_V1)
    try:
        lfs_set = {lfs for lfs, _ in lib.pathways("Rh102", 102)}
        assert 0 in lfs_set and any(l > 0 for l in lfs_set)   # qualified-able
        chain = _qualified_chain_for(lib, "Rh102", 102, "(n,gamma)")
        edges = np.array([1e-5, 0.625, 2e7])
        table = _build_xs_table_pendf(["Rh102"], ["(n,gamma)"], edges, lib, chain)
        ground_row = _row(table, "(n,gamma)")
        assert ground_row is not None                         # qualified emitted
        total_g = _group_average(*lib.xs("Rh102", 102), edges)
        assert ground_row[0] == pytest.approx(total_g[0], rel=1e-3)  # thermal recovered
        assert ground_row[0] > 1.0                            # not the placeholder
    finally:
        lib.close()


@pytest.mark.skipif(not _ENDFB81.exists(),
                    reason="ENDF/B-8.1 pointwise PENDF h5 not available")
def test_silence_fill_real_endfb81_ir192_noop():
    # Ir192 (n,gamma) in ENDF/B-8.1 partitions live from thermal (ratio 1.0), so
    # the fill is a provable no-op: the ground row equals the raw LFS=0 partial.
    import openmc.data as od
    lib = od.PendfLibrary(_ENDFB81)
    try:
        lfs_set = {lfs for lfs, _ in lib.pathways("Ir192", 102)}
        assert 0 in lfs_set and any(l > 0 for l in lfs_set)   # qualified, has LFS=0
        chain = _qualified_chain_for(lib, "Ir192", 102, "(n,gamma)")
        fill = _silence_fill_ground(lib.pathways, lib.pathway_xs, "Ir192", 102,
                                    *lib.xs("Ir192", 102), lfs_set)
        assert fill.fired is False
        edges = np.array([1e-5, 0.625, 2e7])
        table = _build_xs_table_pendf(["Ir192"], ["(n,gamma)"], edges, lib, chain)
        pe0, px0 = lib.pathway_xs("Ir192", 102, 0, None)
        np.testing.assert_array_equal(_row(table, "(n,gamma)"),
                                      _group_average(pe0, px0, edges))
    finally:
        lib.close()
