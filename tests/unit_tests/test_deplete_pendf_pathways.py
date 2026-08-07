"""Unit tests for isomeric pathway (MF=10) expansion in the PENDF collapse path.

Covers the ORIGEN-style "Option A" per-product rows built by
``_build_xs_table_pendf`` (ground row keeps the canonical reaction name,
metastable products get an ``_m{n}`` suffix), the **chain-sourced** row naming
(each MF=10 ``LFS`` partial is bound to the depletion-chain reaction carrying
that ``pendf_lfs``), the demand-side chain semantics (extra library LFS ignored,
demanded-missing LFS falling back to the MF=3 total, the ground-by-balance serve
when the ground is the ONLY missing demanded LFS), the deplete-time
chain<->MicroXS pathway-mismatch hard error, the chain provenance stamp, the
in-domain silence-fill of a placeholder ground, and the always-on expansion
through :meth:`MicroXS.from_multigroup_flux`. Pathway rows come
exclusively from MF=10 partial cross sections -- never from static branching
ratios; the product *names* come from the chain, never from the library.

The last section takes chains straight out of the patcher's ``--orphan-policy``
triad and checks the collapse binds what the writer wrote: a kept orphan branch
gets its own row, and a reattributed duplicate-target entry sums into its
recipient's row.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

import openmc.deplete
from openmc.deplete.chain import Chain
from openmc.deplete.nuclide import Nuclide
from openmc.deplete.decay_elis import DecayState
from openmc.deplete.microxs import (
    MicroXS,
    _build_xs_table_pendf,
    _group_average,
)
from openmc.deplete.pendf.chain_check import (
    _check_pathway_consistency,
    _liso_from_gnds,
)
from openmc.deplete.pendf.ground import _silence_fill_ground

# The chain patcher lives in the repo's tools/ directory (not an installed
# package); the orphan-policy tests at the end of this module start from its
# output rather than from a hand-written chain.
sys.path.insert(0, str(Path(openmc.deplete.__file__).parents[2] / "tools"))

from add_pendf_isomeric_branching_to_chain import (  # noqa: E402
    MODE_DEFAULT_RTOL, decorate_chain, map_library,
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
    metastable (chain {0,1} vs library {0}), a total absence of MF=10 (library {},
    so ``missing`` is the strict superset {0,1}), and a missing ground PLUS a
    missing metastable (library {2}). Only the EXACT ``missing == {0}`` shape is
    ground-by-balance; a ``missing`` superset of {0} stays on this warn+fallback
    path. The value is always the MF=3 total, identical to the old silent
    fallback; only the warning is new."""
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

    # (c) BOTH the ground and a metastable missing (library {2}): ``missing`` is
    # {0, 1}, a strict superset of {0}, so ground-by-balance does NOT apply and the
    # reaction stays on the warn + full-MF=3-total path.
    fake = _FakePendf(
        mf3={"In115": {102: _const(5.0)}},
        mf10={"In115": {102: {2: ("In116_m2", _const(1.0))}}})   # neither 0 nor 1
    chain = Chain()
    nuc = Nuclide("In115")
    nuc.add_reaction("(n,gamma)", "In116", 0.0, 1.0, pendf_lfs=0)
    nuc.add_reaction("(n,gamma)_m1", "In116_m1", 0.0, 1.0, pendf_lfs=1)
    chain.add_nuclide(nuc)
    with pytest.warns(UserWarning, match="does not match") as record:
        table = _build_xs_table_pendf(["In115"], ["(n,gamma)"], edges, fake, chain)
    msg = str(record[0].message)
    assert "chain LFS {0, 1}" in msg
    assert "library LFS {2}" in msg
    assert "ground-by-balance" not in msg
    assert table.reactions == ["(n,gamma)"]
    np.testing.assert_allclose(table.xs_matrix[0], [5.0])


# ---------------------------------------------------------------------------
# Reaction-list sanitation (qualified input == base input)
# ---------------------------------------------------------------------------

def test_reaction_list_sanitation():
    """Reaction-list sanitation: a chain-DEFAULTED list strips ``_mN``, dedupes, and
    drops channels with no REACTION_MT mapping (one warning); an EXPLICIT unknown
    still raises KeyError; and a qualified input builds the SAME table as the base
    name (qualified names are outputs, not inputs)."""
    from openmc.deplete.pendf.chain_check import _default_pendf_reactions

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


# ---------------------------------------------------------------------------
# Ground-by-balance: a demanded LFS=0 the library tabulates no partial for
# ---------------------------------------------------------------------------
#
# When the ONLY demanded LFS missing from the library is the ground, the ground
# row is served implicitly as ``max(0, MF=3 total - Sigma(ALL library metastable
# partials))`` -- ALL, not just the demanded ones, so yield to an untracked level
# is never reattributed to ground. This replaces BOTH the old self-loop waiver
# (which staged the FULL MF=3 total as the base row) and, for a non-self-loop
# ground, the mismatch fallback (which dropped the metastable rows entirely).
# The mismatch warning no longer fires for this shape; one informational summary
# with the clamp count does.

def _balance_summary(record):
    """The single ground-by-balance summary message captured in ``record``."""
    msgs = [str(w.message) for w in record
            if "ground-by-balance" in str(w.message)]
    assert len(msgs) == 1
    return msgs[0]


def _ground_plus_meta_chain(parent, base, ground_target, metas):
    """Chain demanding LFS=0 (``ground_target``) plus each ``(lfs, product)``."""
    chain = Chain()
    nuclide = Nuclide(parent)
    nuclide.add_reaction(base, ground_target, 0.0, 1.0, pendf_lfs=0)
    for lfs, product in metas:
        nuclide.add_reaction(f"{base}_m{_liso_from_gnds(product)}", product,
                             0.0, 1.0, pendf_lfs=lfs)
    chain.add_nuclide(nuclide)
    return chain


def test_balance_ground_self_loop_in115_style():
    """Self-loop ground (In115 (n,n'), target == parent): the tape carries only the
    metastable LFS 1, so the base row is the BALANCE remainder total - sigma_m1 --
    the true ground-production cross section -- not the full MF=3 total the retired
    waiver staged. The m1 row and its branching ratio are untouched, the rows sum
    to the total exactly, and the only warning is the informational balance summary
    (never the chain<->library mismatch)."""
    fake = _FakePendf(
        mf3={"In115": {4: _const(2.0)}},
        mf10={"In115": {4: {1: ("In115_m1", _const(0.8))}}})   # only the metastable
    edges = np.array([0.0, 2.0e7])
    chain = _ground_plus_meta_chain("In115", "(n,n')", "In115",   # self-loop ground
                                    [(1, "In115_m1")])

    with pytest.warns(UserWarning, match="ground-by-balance") as record:
        table = _build_xs_table_pendf(["In115"], ["(n,n')"], edges, fake, chain)

    assert table.reactions == ["(n,n')", "(n,n')_m1"]
    base, meta = _row(table, "(n,n')"), _row(table, "(n,n')_m1")
    np.testing.assert_allclose(base, [1.2])      # 2.0 - 0.8, NOT the 2.0 total
    np.testing.assert_allclose(meta, [0.8])      # m1 row == LFS 1 partial
    np.testing.assert_allclose(base + meta, [2.0])            # exact conservation
    assert meta[0] / 2.0 == pytest.approx(0.4)                # BR preserved
    assert len(record) == 1                                   # no other warning
    msg = _balance_summary(record)
    assert "In115 (n,n') [clamped 0/2 pts]" in msg
    assert "chain and library disagree" not in msg

    # The balance ground is NOT silence-filled: with a placeholder metastable the
    # branching reads "silent", but there is no LFS=0 partial to supply the fill's
    # domain and the remainder already subtracts the metastables, so the ground is
    # exactly total - sigma_m1 everywhere (filling would double-apply it).
    grid = np.array([1e-5, 1.0, 1e3, 2e7])
    total = np.array([100.0, 1.0, 4.0, 1.0])
    meta_xs = np.array([_PH, _PH, 2.0, 0.4])       # placeholder at thermal
    fake = _FakePendf(
        mf3={"In113": {4: (grid, total)}},
        mf10={"In113": {4: {1: ("In113_m1", (grid, meta_xs))}}})
    chain = _ground_plus_meta_chain("In113", "(n,n')", "In113",
                                    [(1, "In113_m1")])
    edges = np.array([1e-5, 0.625, 2e7])
    with pytest.warns(UserWarning, match="ground-by-balance"):
        table = _build_xs_table_pendf(["In113"], ["(n,n')"], edges, fake, chain)
    np.testing.assert_allclose(
        _row(table, "(n,n')"), _group_average(grid, total - meta_xs, edges))


def test_balance_ground_non_self_loop_keeps_metastables():
    """Non-self-loop ground (target != parent): the metastable rows are KEPT --
    before, a missing demanded ground dropped them and staged the bare MF=3 total --
    and the ground row is the CLAMPED remainder. The summary names the reaction with
    its clamp count (the metastable over-sums the total at one grid point)."""
    grid = np.array([1e-5, 1.0, 1e3, 2e7])
    total = np.array([10.0, 8.0, 4.0, 1.0])
    meta = np.array([2.0, 9.0, 1.0, 0.2])          # over-sums the total at 1.0 eV
    fake = _FakePendf(
        mf3={"Xx100": {102: (grid, total)}},
        mf10={"Xx100": {102: {1: ("Yy101_m1", (grid, meta))}}})
    chain = _ground_plus_meta_chain("Xx100", "(n,gamma)", "Yy101",
                                    [(1, "Yy101_m1")])
    edges = np.array([1e-5, 0.625, 2e7])

    with pytest.warns(UserWarning, match="ground-by-balance") as record:
        table = _build_xs_table_pendf(["Xx100"], ["(n,gamma)"], edges, fake, chain)

    assert table.reactions == ["(n,gamma)", "(n,gamma)_m1"]   # metastable KEPT
    np.testing.assert_allclose(
        _row(table, "(n,gamma)"),
        _group_average(grid, np.maximum(total - meta, 0.0), edges))
    np.testing.assert_allclose(_row(table, "(n,gamma)_m1"),
                               _group_average(grid, meta, edges))
    msg = _balance_summary(record)
    assert "Xx100 (n,gamma) [clamped 1/4 pts]" in msg          # one negative point
    assert "chain and library disagree" not in msg

    # ALL library metastable partials are subtracted, not just the demanded ones:
    # an UNDEMANDED level's yield must not be reattributed to the ground channel.
    # Its row is still dropped (the chain is the demand side).
    grid = np.array([1e-5, 1.0, 2e7])
    total = np.array([10.0, 8.0, 1.0])
    m1 = np.array([2.0, 1.6, 0.2])
    m2 = np.array([3.0, 2.4, 0.3])                 # in the library, not demanded
    fake = _FakePendf(
        mf3={"Xx200": {102: (grid, total)}},
        mf10={"Xx200": {102: {1: ("Yy201_m1", (grid, m1)),
                              2: ("Yy201_m2", (grid, m2))}}})
    chain = _ground_plus_meta_chain("Xx200", "(n,gamma)", "Yy201",
                                    [(1, "Yy201_m1")])
    with pytest.warns(UserWarning, match="ground-by-balance"):
        table = _build_xs_table_pendf(["Xx200"], ["(n,gamma)"], edges, fake, chain)
    assert table.reactions == ["(n,gamma)", "(n,gamma)_m1"]    # m2 not emitted
    np.testing.assert_allclose(_row(table, "(n,gamma)"),
                               _group_average(grid, total - m1 - m2, edges))


def test_balance_ground_grouped_group_wise_remainder():
    """Grouped library: the remainder is taken GROUP-wise (a grouped file carries no
    pointwise data to rebin) and clamped per group; the clamp count is in groups."""
    edges = np.array([1e-5, 0.625, 2e7])
    fake = _FakeGroupedPendf(
        edges,
        mf3_g={"Gb": {102: [10.0, 1.0]}},
        mf10_g={"Gb": {102: {1: ("Hb101_m1", [4.0, 1.5])}}})   # over-sums group 1
    chain = _ground_plus_meta_chain("Gb", "(n,gamma)", "Hb101",
                                    [(1, "Hb101_m1")])

    with pytest.warns(UserWarning, match="ground-by-balance") as record:
        table = _build_xs_table_pendf(["Gb"], ["(n,gamma)"], edges, fake, chain)

    np.testing.assert_allclose(_row(table, "(n,gamma)"), [6.0, 0.0])   # 1-1.5 -> 0
    np.testing.assert_allclose(_row(table, "(n,gamma)_m1"), [4.0, 1.5])
    # The count is in GROUPS here, and the message says so (a pointwise build
    # reports 'pts' -- the union-grid points it actually clamped).
    assert "Gb (n,gamma) [clamped 1/2 groups]" in _balance_summary(record)


def test_metastable_only_fold_leaves_zero_base_row(recwarn):
    """Pin (design section 3.4): a chain fold with NO ground member (metastable-only,
    the MF=10-only Ta181 (n,2na) shape) does NOT trigger balance -- there is no
    ground tuple to serve. The always-emitted unsuffixed base COLUMN reads 0.0
    silently while the metastable row is its partial. Any future change to this
    contract must be deliberate."""
    fake = _FakePendf(
        mf3={"Ta181": {24: _const(3.0)}},
        mf10={"Ta181": {24: {1: ("Lu176_m1", _const(3.0))}}})
    chain = Chain()
    nuc = Nuclide("Ta181")
    nuc.add_reaction("(n,2na)_m1", "Lu176_m1", 0.0, 1.0, pendf_lfs=1)   # no ground
    chain.add_nuclide(nuc)
    edges = np.array([0.0, 2.0e7])

    table = _build_xs_table_pendf(["Ta181"], ["(n,2na)"], edges, fake, chain)
    assert table.reactions == ["(n,2na)", "(n,2na)_m1"]   # base column always there
    assert _row(table, "(n,2na)") is None                 # never staged
    np.testing.assert_allclose(_row(table, "(n,2na)_m1"), [3.0])

    micro = MicroXS.from_multigroup_flux(
        energies=[0.0, 2.0e7], multigroup_flux=[1.0], chain_file=chain,
        nuclides=["Ta181"], reactions=["(n,2na)"], pendf_library=fake)
    assert micro["Ta181", "(n,2na)"] == pytest.approx([0.0])   # zero, silently
    assert micro["Ta181", "(n,2na)_m1"] == pytest.approx([3.0])
    assert len(recwarn) == 0


@pytest.mark.skipif(not _JEFF_V1.exists(),
                    reason="JEFF-4.0 v1 pointwise PENDF h5 not available")
def test_balance_ground_real_jeff_in115():
    """Real data, the live class: JEFF-4.0 In115 (n,n') carries MF=10 LFS 1 only.
    The base row becomes total - sigma_m1 (was the full MF=3 total), the m1 row and
    its 0.1565 branching ratio at 14.1 MeV are unchanged, and base + m1 reproduces
    the MF=3 total exactly in a group with no clamping."""
    import openmc.data as od
    lib = od.PendfLibrary(_JEFF_V1)
    try:
        assert lib.pathways("In115", 4) == [(1, 49115)]   # metastable only
        chain = _ground_plus_meta_chain("In115", "(n,n')", "In115",
                                        [(1, "In115_m1")])
        # A narrow group bracketing 14.1 MeV, well away from the 20 MeV grid
        # discontinuity (a duplicated tape energy the union grid collapses).
        edges = np.array([1e-5, 1.40e7, 1.42e7, 2.0e7])
        with pytest.warns(UserWarning, match="ground-by-balance") as record:
            table = _build_xs_table_pendf(["In115"], ["(n,n')"], edges, lib, chain)
        base, meta = _row(table, "(n,n')"), _row(table, "(n,n')_m1")
        total_g = _group_average(*lib.xs("In115", 4), edges)

        g = 1                                     # the 14.0-14.2 MeV group
        assert base[g] + meta[g] == pytest.approx(total_g[g], rel=1e-9)
        assert base[g] == pytest.approx(total_g[g] - meta[g], rel=1e-9)
        assert base[g] < total_g[g]               # the decision-1 value change
        assert meta[g] / total_g[g] == pytest.approx(0.1565, abs=5e-4)  # BR intact
        assert "In115 (n,n') [clamped 0/" in _balance_summary(record)
    finally:
        lib.close()


# ---------- partial-binding (opt-in collapse switch; commit B) ----------
#
# A chain-STOCK reaction whose library carries an LFS=0 ground partial AND >=1
# other partial can, under ``partial_binding``, bind its ground row to the LFS=0
# partial and drop the unmapped metastables -- past a silence/spike veto (binding
# is reason-blind, so a thermal-placeholder or corrupt-grid LFS=0 must not replace
# the live MF=3 ground). Default off is bit-identical to today. Everything below
# uses only helpers defined ABOVE this marker.

def _pb_summary(record):
    """The single partial-binding summary message captured in ``record``."""
    msgs = [str(w.message) for w in record
            if "partial-binding" in str(w.message)]
    assert len(msgs) == 1
    return msgs[0]


def _stock_candidate_fake(nuc, grid, total, ground, meta):
    """A ``_FakePendf`` + STOCK chain: library has LFS=0 ground + metastable but the
    products are unmapped, so ``_chain_from_fake`` builds a stock chain."""
    fake = _FakePendf(
        mf3={nuc: {102: (grid, total)}},
        mf10={nuc: {102: {0: (None, (grid, ground)),
                          1: (None, (grid, meta))}}})
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})   # products None -> stock
    return fake, chain


def test_partial_binding_noncandidates_keep_mf3_total(recwarn):
    """A non-candidate keeps the MF=3 total and emits no partial-binding diagnostic:
    the toggle OFF (default and explicit False, bit-identical -- the regression
    guard), an LFS=0-only library (nothing to drop) under True, and a plain stock
    reaction with no MF=10 at all under True."""
    import warnings

    edges = np.array([1e-5, 0.625, 2e7])

    # (a) Default off and explicit False build a bit-identical MF=3-total table with
    # nothing emitted at all (the regression guard).
    grid = np.array([1e-5, 1.0, 1e3, 1e6, 2e7])
    total = np.array([10.0, 8.0, 4.0, 2.0, 1.0])
    fake, chain = _stock_candidate_fake("P1", grid, total, total * 0.6,
                                        total * 0.4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                  # any warning -> failure
        default = _build_xs_table_pendf(["P1"], ["(n,gamma)"], edges, fake, chain)
        explicit = _build_xs_table_pendf(["P1"], ["(n,gamma)"], edges, fake, chain,
                                         partial_binding=False)
    total_g = _group_average(grid, total, edges)
    assert default.reactions == ["(n,gamma)"]            # no metastable row
    np.testing.assert_array_equal(_row(default, "(n,gamma)"), total_g)
    assert default.reactions == explicit.reactions
    np.testing.assert_array_equal(default.xs_matrix, explicit.xs_matrix)
    assert list(default.rxn_indices) == list(explicit.rxn_indices)

    # (b) An LFS=0-only library (nothing to drop) is NOT a candidate under True.
    grid = np.array([1e-5, 1.0, 2e7])
    total = np.array([5.0, 4.0, 1.0])
    fake = _FakePendf(
        mf3={"L0": {102: (grid, total)}},
        mf10={"L0": {102: {0: (None, (grid, total))}}})   # LFS=0 only
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})     # stock
    table = _build_xs_table_pendf(["L0"], ["(n,gamma)"], edges, fake, chain,
                                  partial_binding=True)
    np.testing.assert_array_equal(_row(table, "(n,gamma)"),
                                  _group_average(grid, total, edges))
    assert table.reactions == ["(n,gamma)"]

    # (c) A plain stock reaction with no MF=10 at all is never a candidate.
    fake = _FakePendf(mf3={"N0": {102: _const(2.0)}})     # MF=3 only
    table = _build_xs_table_pendf(["N0"], ["(n,gamma)"], np.array([0.0, 2e7]),
                                  fake, Chain(), partial_binding=True)
    np.testing.assert_array_equal(_row(table, "(n,gamma)"), [2.0])

    assert not [w for w in recwarn if "partial-binding" in str(w.message)]


def test_partial_binding_binds_live_candidate():
    """A LIVE stock candidate binds under True: the ground row becomes the group-
    averaged LFS=0 partial (< MF=3 total), the metastable is dropped, and the
    diagnostic names the bound channel -- both directly in ``_build_xs_table_pendf``
    and threaded through the public :meth:`from_multigroup_flux` entry."""
    grid = np.array([1e-5, 1.0, 1e3, 1e6, 2e7])
    total = np.array([10.0, 8.0, 4.0, 2.0, 1.0])
    ground = total * 0.6
    fake, chain = _stock_candidate_fake("P2", grid, total, ground, total * 0.4)
    edges = np.array([1e-5, 0.625, 2e7])

    with pytest.warns(UserWarning, match="partial-binding") as record:
        table = _build_xs_table_pendf(["P2"], ["(n,gamma)"], edges, fake, chain,
                                      partial_binding=True)
    assert table.reactions == ["(n,gamma)"]             # metastable absent
    np.testing.assert_array_equal(_row(table, "(n,gamma)"),
                                  _group_average(grid, ground, edges))
    assert np.all(_row(table, "(n,gamma)") < _group_average(grid, total, edges))
    assert "1 bound to MF=10 ground [P2 (n,gamma)]" in _pb_summary(record)

    # The kwarg threads through the public entry point: off -> MF=3 total; on ->
    # LFS=0 ground (< total), no metastable row.
    grid = np.array([1e-5, 1.0, 2e7])
    total = np.array([10.0, 8.0, 1.0])
    fake, chain = _stock_candidate_fake("P9", grid, total, total * 0.6,
                                        total * 0.4)
    edges = [1e-5, 0.625, 2e7]
    off = MicroXS.from_multigroup_flux(
        energies=edges, multigroup_flux=[1.0, 1.0], chain_file=chain,
        nuclides=["P9"], reactions=["(n,gamma)"], pendf_library=fake)
    with pytest.warns(UserWarning, match="partial-binding"):
        on = MicroXS.from_multigroup_flux(
            energies=edges, multigroup_flux=[1.0, 1.0], chain_file=chain,
            nuclides=["P9"], reactions=["(n,gamma)"], pendf_library=fake,
            partial_binding=True)
    assert "(n,gamma)_m1" not in on.reactions
    assert on["P9", "(n,gamma)"][0] < off["P9", "(n,gamma)"][0]   # ground < total


def test_partial_binding_veto_keeps_mf3():
    """Binding is vetoed and the MF=3 total kept when the LFS=0 ground cannot be
    trusted: a thermal-placeholder ground while the MF=3 total is live (silence
    veto, Bk247 class), and partials that over-sum at an in-domain point (spike
    veto, corrupt-grid class). The diagnostic names each with its veto reason."""
    edges = np.array([1e-5, 0.625, 2e7])

    # (a) Silence veto: LFS=0 (and metastable) are ~1e-20 placeholders < ~100 eV
    # while the MF=3 total is live (8 b).
    grid = np.array([1e-5, 1e-3, 0.1, 100.0, 1e3, 2e7])
    total = np.array([8.0, 2.5, 0.8, 5.0, 4.0, 1.0])
    ground = np.array([_PH, _PH, _PH, 3.0, 2.4, 0.6])   # placeholder < ~100 eV
    meta = np.array([_PH, _PH, _PH, 2.0, 1.6, 0.4])
    fake = _FakePendf(
        mf3={"Bk247": {102: (grid, total)}},
        mf10={"Bk247": {102: {0: (None, (grid, ground)),
                              1: (None, (grid, meta))}}})
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})   # stock
    with pytest.warns(UserWarning, match="partial-binding") as record:
        table = _build_xs_table_pendf(["Bk247"], ["(n,gamma)"], edges, fake,
                                      chain, partial_binding=True)
    np.testing.assert_array_equal(_row(table, "(n,gamma)"),
                                  _group_average(grid, total, edges))
    assert table.reactions == ["(n,gamma)"]
    msg = _pb_summary(record)
    assert "0 bound" in msg
    assert "1 vetoed [Bk247 (n,gamma) (silent)]" in msg

    # (b) Spike veto: partials over-sum (Sigma(all)/total = 3 > 1.5) at one point.
    grid = np.array([1e-5, 1.0, 10.0, 1e3, 2e7])
    total = np.array([10.0, 8.0, 5.0, 4.0, 1.0])
    ground = np.array([6.0, 5.0, 3.0, 2.4, 0.6])
    meta = np.array([4.0, 3.0, 12.0, 1.6, 0.4])          # 3+12=15 vs 5 -> ratio 3
    fake = _FakePendf(
        mf3={"Sp1": {102: (grid, total)}},
        mf10={"Sp1": {102: {0: (None, (grid, ground)),
                            1: (None, (grid, meta))}}})
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})   # stock
    with pytest.warns(UserWarning, match="partial-binding") as record:
        table = _build_xs_table_pendf(["Sp1"], ["(n,gamma)"], edges, fake, chain,
                                      partial_binding=True)
    np.testing.assert_array_equal(_row(table, "(n,gamma)"),
                                  _group_average(grid, total, edges))
    assert "1 vetoed [Sp1 (n,gamma) (spike)]" in _pb_summary(record)


def test_partial_binding_diagnostic_scope_and_counts():
    """The single partial-binding summary scopes to the requested pairs and reports
    per-channel counts with names: a Collection binds only its listed (nuclide, MT)
    and leaves the rest on the MF=3 total (out-of-scope channels unnamed), and a
    full True run reports N bound / V vetoed / K kept-no-usable-LFS=0 with names and
    the veto reason."""
    edges = np.array([1e-5, 0.625, 2e7])

    # (a) A Collection listing only ('A', '(n,gamma)') binds A; B stays MF=3 total.
    grid = np.array([1e-5, 1.0, 2e7])
    tA = np.array([10.0, 8.0, 1.0])
    tB = np.array([6.0, 5.0, 1.0])
    fake = _FakePendf(
        mf3={"A": {102: (grid, tA)}, "B": {102: (grid, tB)}},
        mf10={"A": {102: {0: (None, (grid, tA * 0.6)),
                          1: (None, (grid, tA * 0.4))}},
              "B": {102: {0: (None, (grid, tB * 0.5)),
                          1: (None, (grid, tB * 0.5))}}})
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})    # both stock
    with pytest.warns(UserWarning, match="partial-binding") as record:
        table = _build_xs_table_pendf(
            ["A", "B"], ["(n,gamma)"], edges, fake, chain,
            partial_binding={("A", "(n,gamma)")})
    np.testing.assert_array_equal(_row(table, "(n,gamma)", nuc_idx=0),
                                  _group_average(grid, tA * 0.6, edges))
    np.testing.assert_array_equal(_row(table, "(n,gamma)", nuc_idx=1),
                                  _group_average(grid, tB, edges))     # MF=3 total
    msg = _pb_summary(record)
    assert "1 bound to MF=10 ground [A (n,gamma)]" in msg
    assert "B (n,gamma)" not in msg                       # out of scope, unnamed

    # (b) A full True run: one bound (live), one vetoed (silent thermal placeholder),
    # one kept (LFS=0 group-averages to zero while a metastable is live, so it passes
    # the veto but has no bindable ground). The summary reports N=1 / V=1 / K=1.
    grid = np.array([1e-5, 1.0, 1e3, 2e7])
    tL = np.array([10.0, 8.0, 4.0, 1.0])                  # live candidate
    tS = np.array([8.0, 0.5, 5.0, 1.0])                   # silent thermal
    gS = np.array([_PH, _PH, 3.0, 0.6])
    mS = np.array([_PH, _PH, 2.0, 0.4])
    tK = np.array([6.0, 5.0, 3.0, 1.0])                   # LFS=0 zero, m live
    fake = _FakePendf(
        mf3={"Lv": {102: (grid, tL)}, "Si": {102: (grid, tS)},
             "Kp": {102: (grid, tK)}},
        mf10={"Lv": {102: {0: (None, (grid, tL * 0.6)),
                           1: (None, (grid, tL * 0.4))}},
              "Si": {102: {0: (None, (grid, gS)), 1: (None, (grid, mS))}},
              "Kp": {102: {0: (None, (grid, np.zeros_like(tK))),
                           1: (None, (grid, tK))}}})
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})    # all stock
    with pytest.warns(UserWarning, match="partial-binding") as record:
        _build_xs_table_pendf(["Lv", "Si", "Kp"], ["(n,gamma)"], edges, fake,
                              chain, partial_binding=True)
    msg = _pb_summary(record)
    assert "1 bound to MF=10 ground [Lv (n,gamma)]" in msg
    assert "1 vetoed [Si (n,gamma) (silent)]" in msg
    assert "1 kept MF=3 total, no usable LFS=0 [Kp (n,gamma)]" in msg


def test_partial_binding_grouped_binds_and_vetoes():
    """Grouped-library path: the veto is group-space. Gb (live) binds to its LFS=0
    group vector; Gs (thermal-group placeholder while the total is live) is
    silence-vetoed and keeps the MF=3 total."""
    edges = np.array([1e-5, 0.625, 2e7])
    fake = _FakeGroupedPendf(
        edges,
        mf3_g={"Gb": {102: [10.0, 1.0]}, "Gs": {102: [8.0, 1.0]}},
        mf10_g={"Gb": {102: {0: (None, [6.0, 0.6]), 1: (None, [4.0, 0.4])}},
                "Gs": {102: {0: (None, [_PH, 0.6]), 1: (None, [_PH, 0.4])}}})
    chain = _chain_from_fake(fake, {102: "(n,gamma)"})    # both stock

    with pytest.warns(UserWarning, match="partial-binding") as record:
        table = _build_xs_table_pendf(["Gb", "Gs"], ["(n,gamma)"], edges, fake,
                                      chain, partial_binding=True)
    np.testing.assert_array_equal(_row(table, "(n,gamma)", nuc_idx=0),
                                  [6.0, 0.6])              # bound LFS=0 vector
    np.testing.assert_array_equal(_row(table, "(n,gamma)", nuc_idx=1),
                                  [8.0, 1.0])              # MF=3 total kept
    msg = _pb_summary(record)
    assert "1 bound to MF=10 ground [Gb (n,gamma)]" in msg
    assert "1 vetoed [Gs (n,gamma) (silent)]" in msg


@pytest.mark.skipif(not _JEFF_V1.exists(),
                    reason="JEFF-4.0 v1 pointwise PENDF h5 not available")
def test_partial_binding_real_jeff_np239():
    # Np239 (n,gamma) in JEFF-4.0: LFS=0 is LIVE at thermal (~55 b) while the MF=3
    # total is ~81 b (the metastable carries the rest). A stock chain + True binds
    # the ground to the LFS=0 partial (< MF=3 total) and drops the metastable.
    import openmc.data as od
    lib = od.PendfLibrary(_JEFF_V1)
    try:
        pw = {lfs for lfs, _ in lib.pathways("Np239", 102)}
        assert 0 in pw and any(l > 0 for l in pw)          # candidate-able
        edges = np.array([1e-5, 0.625, 2e7])
        total_g = _group_average(*lib.xs("Np239", 102), edges)
        pe0, px0 = lib.pathway_xs("Np239", 102, 0, None)
        ground0_g = _group_average(pe0, px0, edges)

        # Off: today's behavior -- the MF=3 total (Np239 stock: absent from chain).
        off = _build_xs_table_pendf(["Np239"], ["(n,gamma)"], edges, lib, Chain())
        np.testing.assert_array_equal(_row(off, "(n,gamma)"), total_g)
        assert off.reactions == ["(n,gamma)"]

        # On: binds the live LFS=0 ground, drops the metastable.
        with pytest.warns(UserWarning, match="partial-binding"):
            on = _build_xs_table_pendf(["Np239"], ["(n,gamma)"], edges, lib,
                                       Chain(), partial_binding=True)
        assert on.reactions == ["(n,gamma)"]
        np.testing.assert_array_equal(_row(on, "(n,gamma)"), ground0_g)
        assert _row(on, "(n,gamma)")[0] > 1.0              # live thermal (~55 b)
        assert _row(on, "(n,gamma)")[0] < total_g[0]       # below the MF=3 total
    finally:
        lib.close()


# ---------------------------------------------------------------------------
# Orphan-policy chains meeting the collapse (--orphan-policy add-stable |
# reattribute)
#
# These start at the PATCHER, not at a hand-written chain: the point is the
# seam. The writer emits a kept orphan branch (add-stable) or a duplicate-target
# entry folding the orphan's column onto its recipient (reattribute), and the
# demand loop here has to bind what it wrote. Shape: Au186 (n,2n) -> Au185 with
# LFS=0 and LFS=6, whose 14.1 MeV columns are 1.3312 b and 0.1682 b against an
# MF=3 total of 1.4994 b (verified against the production HDF5 library).
# ---------------------------------------------------------------------------

_AU_GROUND, _AU_ORPHAN = 1.3312, 0.1682          # barns at 14.1 MeV
_AU_TOTAL = _AU_GROUND + _AU_ORPHAN              # == the MF=3 total, 1.4994 b


class _FakePatchSource:
    """Minimal PENDF source adapter for the patcher (not the collapse)."""

    kind, library, mapping = "fake", "synthetic", "elis"
    nuclides = ["Au186"]

    def reactions(self, nuclide):
        return {16: dict(qm=-7928030.0, qi=-7928030.0, partials=[
            dict(lfs=0, izap=79185, qi=-7928030.0, qm=-7928030.0, elfs=0.0),
            dict(lfs=6, izap=79185, qi=-8128030.0, qm=-7928030.0,
                 elfs=200000.0)])}

    def total_xs(self, nuclide, mt):
        raise KeyError("no MF=3 array on this fixture")

    def pathway_xs(self, nuclide, mt, lfs, izap=None):
        raise KeyError("no MF=10 array on this fixture")

    def nuclide_elis(self, nuclide):
        return None

    def close(self):
        pass


def _au186_patched_chain(policy):
    """Run the patcher over the Au186 shape and return the patched chain.

    decay_2020 carries no Au185 metastable, so LFS=6 is a true orphan in every
    mapping mode and the policy alone decides what its branch becomes.
    """
    chain = Chain()
    au186 = Nuclide("Au186")
    au186.add_reaction("(n,2n)", "Au185", -7928030.0, 1.0)
    chain.add_nuclide(au186)
    chain.add_nuclide(Nuclide("Au185"))
    decay = {(79, 185): [DecayState(79, 185, 0.0, 0)],
             (79, 186): [DecayState(79, 186, 0.0, 0)]}
    mode = "elis_lfs_order"
    branching, stats = map_library(_FakePatchSource(), chain, decay, mode,
                                   MODE_DEFAULT_RTOL[mode], 0.0, verbose=False,
                                   orphan_policy=policy)
    decorate_chain(chain, branching, stats, orphan_policy=policy)
    return chain


def _au186_library():
    return _FakePendf(
        mf3={"Au186": {16: _const(_AU_TOTAL)}},
        mf10={"Au186": {16: {0: ("Au185", _const(_AU_GROUND)),
                             6: ("Au185_m1", _const(_AU_ORPHAN))}}})


def test_add_stable_orphan_branch_binds_the_collapse_demand():
    """A chain nuclide the decay library never carried still binds its column.

    ``Au185_m1`` is a minted placeholder name, not a decay-library LISO, but
    the demand loop only needs a chain reaction carrying the LFS -- so the
    orphan's 0.168 b leaves the ground row and gets a row of its own, and the
    two still sum to the MF=3 total.
    """
    edges = np.array([0.0, 2.0e7])
    chain = _au186_patched_chain("add-stable")
    assert "Au185_m1" in chain.nuclide_dict

    table = _build_xs_table_pendf(["Au186"], ["(n,2n)"], edges,
                                  _au186_library(), chain)

    assert table.reactions == ["(n,2n)", "(n,2n)_m1"]
    ground = table.xs_matrix[list(table.rxn_indices).index(0)]
    orphan = table.xs_matrix[list(table.rxn_indices).index(1)]
    np.testing.assert_allclose(ground, [_AU_GROUND])
    np.testing.assert_allclose(orphan, [_AU_ORPHAN])
    np.testing.assert_allclose(ground + orphan, [_AU_TOTAL])


def test_reattribute_duplicate_target_sums_into_one_collapse_row():
    """The fold is arithmetic the collapse already does.

    ``reattribute`` writes two entries with the SAME target and distinct LFS,
    which the demand loop keys separately and ``stage()`` sums into one row --
    so the recipient's row is sigma(LFS=0) + sigma(folded) = 1.3312 + 0.1682 =
    1.4994 b, exactly the MF=3 total. No collapse change was needed for the
    policy, and the row count is unchanged from the stock chain, which is what
    keeps the downstream row/stamp counts stable.
    """
    edges = np.array([0.0, 2.0e7])
    chain = _au186_patched_chain("reattribute")
    assert "Au185_m1" not in chain.nuclide_dict
    assert [(rx.type, rx.target, rx.pendf_lfs)
            for rx in chain["Au186"].reactions] == \
        [("(n,2n)", "Au185", 0), ("(n,2n)", "Au185", 6)]

    lib = _au186_library()
    table = _build_xs_table_pendf(["Au186"], ["(n,2n)"], edges, lib, chain)

    assert table.reactions == ["(n,2n)"]              # one row, not two
    assert list(table.rxn_indices) == [0]             # nothing to clobber
    np.testing.assert_allclose(table.xs_matrix[0], [_AU_TOTAL])
    np.testing.assert_allclose(table.xs_matrix[0],
                               _group_average(*lib.xs("Au186", 16), edges))

    # Row count is identical to the stock (unpatched) chain's single MF=3 row.
    stock = _build_xs_table_pendf(["Au186"], ["(n,2n)"], edges, lib, Chain())
    assert len(table.reactions) == len(stock.reactions) == 1

    # End to end through the public entry point.
    micro = MicroXS.from_multigroup_flux(
        energies=edges, multigroup_flux=[1.0], chain_file=chain,
        nuclides=["Au186"], reactions=["(n,2n)"], pendf_library=lib)
    assert list(micro.reactions) == ["(n,2n)"]
    assert micro["Au186", "(n,2n)"] == pytest.approx([_AU_TOTAL])
