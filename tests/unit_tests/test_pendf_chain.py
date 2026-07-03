"""Unit tests for building a PENDF-native depletion chain."""

from pathlib import Path

import h5py
import pytest

from openmc.deplete.pendf_chain import chain_from_pendf, _reaction_products

_PENDF_H5 = Path(
    "/home/perry/Projects/OMC_Development/PENDF/data/tendl2017_pendf_293K_elis.h5")
_DECAY_DIR = Path("/home/perry/NukeData/Activation/DecayData/decay_2020")

pytestmark = pytest.mark.skipif(
    not (_PENDF_H5.is_file() and _DECAY_DIR.is_dir()),
    reason="requires the real PENDF HDF5 library and decay_2020 sublibrary")

# Small chain around the Am/In/Fe/W neighborhoods plus decay-demanded targets.
_NUCLIDES = [
    "Am241", "Am242", "Am242_m1",
    "In115", "In115_m1", "In116", "In116_m1", "In116_m2",
    "Fe56", "Fe57",
    "W186", "W187",
    "Np237", "Sn116", "Re187", "Cm242",  # decay daughters
]


@pytest.fixture(scope="module")
def small_chain():
    return chain_from_pendf(_PENDF_H5, _DECAY_DIR, nuclides=_NUCLIDES)


@pytest.fixture(scope="module")
def in115_seed_chain():
    # Seeding ONLY the target: the closure must follow transmutation products
    # (not just decay daughters) so the capture pathways survive. This walks
    # the full activation network upward from In115 (~3k nuclides, tens of s),
    # so it is its own module-scoped fixture kept off the fast path.
    return chain_from_pendf(_PENDF_H5, _DECAY_DIR, nuclides=["In115"])


def _reactions(chain, parent):
    """Map reaction type -> (target, branching_ratio) for a parent nuclide."""
    return {rx.type: (rx.target, rx.branching_ratio)
            for rx in chain[parent].reactions}


def _reaction_q(chain, parent):
    """Map reaction type -> Q value for a parent nuclide."""
    return {rx.type: rx.Q for rx in chain[parent].reactions}


def test_am241_ground_and_metastable(small_chain):
    rxns = _reactions(small_chain, "Am241")
    assert rxns["(n,gamma)"] == ("Am242", 1.0)
    assert rxns["(n,gamma)_m1"] == ("Am242_m1", 1.0)


def test_in115_three_capture_channels(small_chain):
    rxns = _reactions(small_chain, "In115")
    assert rxns["(n,gamma)"] == ("In116", 1.0)
    assert rxns["(n,gamma)_m1"] == ("In116_m1", 1.0)
    assert rxns["(n,gamma)_m2"] == ("In116_m2", 1.0)


def test_in115_inelastic_metastable(small_chain):
    # MT=4 (n,n') must be picked up now that it is in deplete REACTIONS.
    rxns = _reactions(small_chain, "In115")
    assert rxns["(n,n')_m1"] == ("In115_m1", 1.0)


def test_per_lfs_qi(small_chain):
    # Each LFS pathway carries its own QI, not the MT-group QI. For In115 MT4
    # the group QI is -336244 (the LFS1 value) but LFS0 (ground) is 0.
    q = _reaction_q(small_chain, "In115")
    assert q["(n,n')"] == 0.0
    assert q["(n,n')_m1"] == -336244.0
    # (n,gamma): group QI 6784730 (=LFS0), metastable pathways differ.
    assert q["(n,gamma)_m1"] == 6657460.0
    assert q["(n,gamma)_m2"] == 6495070.0


def test_fe56_ground_only(small_chain):
    rxns = _reactions(small_chain, "Fe56")
    assert rxns["(n,gamma)"] == ("Fe57", 1.0)
    assert not any(t.startswith("(n,gamma)_m") for t in rxns)


def test_all_branching_ratios_unity(small_chain):
    for parent in ("Am241", "In115", "Fe56", "W186"):
        for rx in small_chain[parent].reactions:
            assert rx.branching_ratio == 1.0


def test_reduce_keeps_qualified_pathways(small_chain):
    # reduce must not KeyError on product-qualified reaction types, and it must
    # keep the isomeric siblings of any reached nuclide together.
    reduced = small_chain.reduce(["In115", "Fe56"], keep_isomeric_siblings=True)
    names = {n.name for n in reduced.nuclides}
    assert {"In116", "In116_m1", "In116_m2"} <= names
    rxns = _reactions(reduced, "In115")
    assert "(n,gamma)_m1" in rxns
    assert "(n,gamma)_m2" in rxns
    assert "(n,n')_m1" in rxns


def test_reduce_without_siblings_no_crash(small_chain):
    reduced = small_chain.reduce(["In115", "Fe56"], keep_isomeric_siblings=False)
    assert "(n,gamma)_m1" in _reactions(reduced, "In115")


def test_form_rxn_matrix_qualified_types(small_chain):
    # form_rxn_matrix looks up REACTIONS[type].secondaries; product-qualified
    # types must resolve via their base type instead of raising KeyError, and
    # the gain term must route to the isomer target.
    from openmc.deplete.reaction_rates import ReactionRates
    rxn_types = sorted({rx.type for rx in small_chain["In115"].reactions})
    rates = ReactionRates(["1"], ["In115"], rxn_types)
    rates.fill(1.0)
    matrix = small_chain.form_rxn_matrix(rates[0])
    i = small_chain.nuclide_dict["In115"]
    assert matrix[small_chain.nuclide_dict["In116_m1"], i] == 1.0
    assert matrix[small_chain.nuclide_dict["In116_m2"], i] == 1.0
    assert matrix[small_chain.nuclide_dict["In115_m1"], i] == 1.0


def test_reduce_isomeric_sibling_expansion(small_chain):
    # keep_isomeric_siblings pulls in the metastable siblings of a seeded ground
    # state even when they are not otherwise reached.
    with_sib = {n.name for n in small_chain.reduce(
        ["In116"], level=0, keep_isomeric_siblings=True).nuclides}
    without = {n.name for n in small_chain.reduce(
        ["In116"], level=0, keep_isomeric_siblings=False).nuclides}
    assert with_sib == {"In116", "In116_m1", "In116_m2"}
    assert without == {"In116"}


def test_coverage_report_shape(small_chain):
    assert hasattr(small_chain, "coverage")
    for entry in small_chain.coverage:
        assert set(entry) == {"parent", "reaction", "product", "reason"}


def test_xml_roundtrip(tmp_path, small_chain):
    from openmc.deplete import Chain

    path = tmp_path / "pendf_chain.xml"
    small_chain.export_to_xml(path)
    reread = Chain.from_xml(path)

    orig = _reactions(small_chain, "Am241")
    back = _reactions(reread, "Am241")
    assert back["(n,gamma)"] == orig["(n,gamma)"]
    assert back["(n,gamma)_m1"] == orig["(n,gamma)_m1"]


def test_seed_only_target_follows_transmutation(in115_seed_chain):
    # Seeding ONLY In115: the closure must follow reaction products, so the
    # (n,gamma) capture products join the chain and In115 carries all three
    # capture pathways -- not just the (n,n') self-scatter to decay daughters.
    names = {n.name for n in in115_seed_chain.nuclides}
    assert {"In116", "In116_m1", "In116_m2", "In115_m1"} <= names
    rxns = _reactions(in115_seed_chain, "In115")
    assert rxns["(n,gamma)"] == ("In116", 1.0)
    assert rxns["(n,gamma)_m1"] == ("In116_m1", 1.0)
    assert rxns["(n,gamma)_m2"] == ("In116_m2", 1.0)
    assert rxns["(n,n')_m1"] == ("In115_m1", 1.0)
    # No capture pathway should have silently fallen into the coverage report.
    dropped = [c for c in in115_seed_chain.coverage
               if c["parent"] == "In115"
               and c["reason"] == "pathway target not in chain nuclide set"]
    assert dropped == []


def test_reactions_populated_in_memory(small_chain, tmp_path):
    # IndependentOperator reads chain.reactions on the in-memory object; it must
    # be non-empty and match what a from_xml round-trip would produce (ordered).
    from openmc.deplete import Chain

    assert small_chain.reactions
    assert "(n,gamma)" in small_chain.reactions
    assert "(n,gamma)_m1" in small_chain.reactions

    path = tmp_path / "reactions_roundtrip.xml"
    small_chain.export_to_xml(path)
    reread = Chain.from_xml(path)
    assert small_chain.reactions == reread.reactions


def test_reaction_products_skips_out_of_range_z():
    # An exotic multi-particle MT on a low-Z target drives the product below
    # Z=1; _reaction_products must skip it instead of raising a KeyError on the
    # ATOMIC_SYMBOL lookup. (n,3a) on H1 gives Z = 1 + (-6) = -5.
    with h5py.File("mem.h5", "w", driver="core", backing_store=False) as h5:
        grp = h5.create_group("H1")
        grp.create_group("MT999")  # no LFS -> DADZ/ATOMIC_SYMBOL ground path
        products = list(_reaction_products(h5, "H1", {999: "(n,3a)"}))
    assert products == []
