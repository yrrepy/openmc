"""Unit tests for the PENDF isomeric-branching chain patcher mapping core.

These exercise the mapping/decoration logic on small synthetic inputs (no real
PENDF HDF5 or decay library needed): ELIS match happy path, rtol skip,
product-not-in-chain skip, reaction-added-to-chain, ground-only, and the full
decorate -> export -> reload fold round-trip.
"""

import sys
from pathlib import Path

import pytest

import openmc.deplete
from openmc.deplete import Chain, Nuclide
from openmc.deplete.decay_elis import DecayState

# The patcher lives in the repo's tools/ directory (not an installed package).
_TOOLS = Path(openmc.deplete.__file__).parents[2] / "tools"
sys.path.insert(0, str(_TOOLS))

from add_pendf_isomeric_branching_to_chain import (  # noqa: E402
    _classify_metastables, map_library, decorate_chain,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _decay_lookup():
    """(Z, A) -> [DecayState]; In116 has m1/m2, In115 has m1."""
    return {
        (49, 116): [
            DecayState(49, 116, 0.0, 0, half_life=None),
            DecayState(49, 116, 127000.0, 1, half_life=3247.0),
            DecayState(49, 116, 289000.0, 2, half_life=54.0),
        ],
        (49, 115): [
            DecayState(49, 115, 0.0, 0, half_life=None),
            DecayState(49, 115, 336000.0, 1, half_life=1.6e14),
        ],
    }


def _in115_ng_metastables():
    """The two In115 (n,gamma) metastable partials (LFS 1 and 4)."""
    return [
        dict(lfs=1, izap=49116, qi=6657450.0, qm=6784720.0, elfs=127270.0),
        dict(lfs=4, izap=49116, qi=6495060.0, qm=6784720.0, elfs=289660.0),
    ]


class _FakeSource:
    """Minimal PENDF source adapter over a dict of per-nuclide reactions."""

    kind = "fake"
    library = "synthetic"
    mapping = "elis"

    def __init__(self, data):
        self._data = data
        self.nuclides = sorted(data)

    def reactions(self, nuclide):
        return self._data[nuclide]

    def close(self):
        pass


def _chain_with(names, reactions=None):
    """Build a Chain from a list of nuclide names (+ optional base reactions)."""
    chain = Chain()
    for name in names:
        nuc = Nuclide(name)
        if reactions and name in reactions:
            for r_type, target, q in reactions[name]:
                nuc.add_reaction(r_type, target, q, 1.0)
        chain.add_nuclide(nuc)
    return chain


# ---------------------------------------------------------------------------
# _classify_metastables
# ---------------------------------------------------------------------------

def test_elis_match_happy_path():
    chain_names = {"In116", "In116_m1", "In116_m2"}
    recs = _classify_metastables(
        "In115", 102, "(n,gamma)", _in115_ng_metastables(),
        _decay_lookup(), chain_names, "elis", 0.50, 0.0)
    by_lfs = {r["lfs"]: r for r in recs}
    assert by_lfs[1]["bucket"] == "matched"
    assert by_lfs[1]["liso"] == 1
    assert by_lfs[1]["product"] == "In116_m1"
    assert by_lfs[4]["bucket"] == "matched"
    assert by_lfs[4]["liso"] == 2
    assert by_lfs[4]["product"] == "In116_m2"


def test_rtol_skip():
    # ELFS 289660 vs In116_m1 dk 127000: |diff|/127000 = 128% >> 50% and it is
    # the nearest metastable once m2 is removed -> nearest/rtol_exceeded.
    decay = {(49, 116): [
        DecayState(49, 116, 0.0, 0),
        DecayState(49, 116, 127000.0, 1),
    ]}
    recs = _classify_metastables(
        "In115", 102, "(n,gamma)",
        [dict(lfs=4, izap=49116, qi=6495060.0, qm=6784720.0, elfs=289660.0)],
        decay, {"In116", "In116_m1"}, "elis", 0.50, 0.0)
    assert recs[0]["bucket"] == "rtol_exceeded"


def test_product_not_in_chain_skip():
    # Matched to In116_m1 but the product is absent from the chain nuclide set.
    recs = _classify_metastables(
        "In115", 102, "(n,gamma)",
        [dict(lfs=1, izap=49116, qi=6657450.0, qm=6784720.0, elfs=127270.0)],
        _decay_lookup(), {"In116"}, "elis", 0.50, 0.0)
    assert recs[0]["bucket"] == "product_not_in_chain"
    assert recs[0]["product"] == "In116_m1"


def test_no_product_in_decay_library():
    # (Z, A) absent from the decay lookup -> no_dk bucket.
    recs = _classify_metastables(
        "In115", 102, "(n,gamma)",
        [dict(lfs=1, izap=49116, qi=6657450.0, qm=6784720.0, elfs=127270.0)],
        {}, {"In116", "In116_m1"}, "elis", 0.50, 0.0)
    assert recs[0]["bucket"] == "no_dk"


# ---------------------------------------------------------------------------
# map_library + decorate_chain integration
# ---------------------------------------------------------------------------

def test_reaction_added_to_chain():
    # In115 has NO (n,n') in the base chain; the MT=4 metastable partial must
    # add a folded (n,n') group and bump the 'reactions added' counter.
    chain = _chain_with(["In115", "In115_m1", "In116", "In116_m1", "In116_m2"],
                        reactions={"In115": [("(n,gamma)", "In116", 6784720.0)]})
    source = _FakeSource({"In115": {
        4: dict(qm=0.0, qi=-336240.0, partials=[
            dict(lfs=1, izap=49115, qi=-336240.0, qm=0.0, elfs=336240.0)]),
    }})
    branching, stats = map_library(source, chain, _decay_lookup(),
                                   "elis", 0.50, 0.0)
    added = decorate_chain(chain, branching)
    assert added == 1
    rxns = {rx.type: (rx.target, rx.pendf_lfs)
            for rx in chain["In115"].reactions}
    assert rxns["(n,n')"] == ("In115", 0)          # synthesized ground self-loop
    assert rxns["(n,n')_m1"] == ("In115_m1", 1)


def test_ground_only_reaction_left_stock():
    # A reaction whose MF=10 carries only a ground (LFS 0) partial produces no
    # metastable pathway and is counted as ground-only.
    chain = _chain_with(["Fe56", "Fe57"],
                        reactions={"Fe56": [("(n,gamma)", "Fe57", 7.6e6)]})
    source = _FakeSource({"Fe56": {
        102: dict(qm=7.6e6, qi=7.6e6, partials=[
            dict(lfs=0, izap=26057, qi=7.6e6, qm=7.6e6, elfs=0.0)]),
    }})
    branching, stats = map_library(source, chain, _decay_lookup(),
                                   "elis", 0.50, 0.0)
    assert "Fe56" not in branching
    assert stats["ground_only"] >= 1
    decorate_chain(chain, branching)
    rxns = {rx.type for rx in chain["Fe56"].reactions}
    assert rxns == {"(n,gamma)"}                    # untouched, still stock


def test_full_fold_roundtrip(tmp_path):
    # decorate -> export_to_xml -> from_xml: In115 (n,gamma) folds to base +
    # _m1 + _m2 and unfolds back with the expected Q and pendf_lfs values.
    chain = _chain_with(["In115", "In116", "In116_m1", "In116_m2"],
                        reactions={"In115": [("(n,gamma)", "In116", 6784720.0)]})
    source = _FakeSource({"In115": {
        102: dict(qm=6784720.0, qi=6784720.0, partials=(
            [dict(lfs=0, izap=49116, qi=6784720.0, qm=6784720.0, elfs=0.0)]
            + _in115_ng_metastables())),
    }})
    branching, stats = map_library(source, chain, _decay_lookup(),
                                   "elis", 0.50, 0.0)
    decorate_chain(chain, branching)

    out = tmp_path / "chain.xml"
    chain.export_to_xml(out)
    text = out.read_text()
    assert 'targets="In116 In116_m1 In116_m2"' in text
    assert 'pendf_lfs="0 1 4"' in text

    reread = Chain.from_xml(out)
    rxns = {rx.type: (rx.target, rx.Q, rx.pendf_lfs)
            for rx in reread["In115"].reactions}
    assert rxns["(n,gamma)"] == ("In116", 6784720.0, 0)
    assert rxns["(n,gamma)_m1"] == ("In116_m1", 6657450.0, 1)
    assert rxns["(n,gamma)_m2"] == ("In116_m2", 6495060.0, 4)


def test_elis_matched_stat_counts():
    chain = _chain_with(["In115", "In116", "In116_m1", "In116_m2"],
                        reactions={"In115": [("(n,gamma)", "In116", 6784720.0)]})
    source = _FakeSource({"In115": {
        102: dict(qm=6784720.0, qi=6784720.0, partials=_in115_ng_metastables()),
    }})
    _, stats = map_library(source, chain, _decay_lookup(), "elis", 0.50, 0.0)
    assert stats["matched"] == 2
    assert stats["nuclides_with_branching"] == 1
