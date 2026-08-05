"""Unit tests for the PENDF isomeric-branching chain patcher mapping core.

These exercise the mapping/decoration logic on small synthetic inputs (no real
PENDF HDF5 or decay library needed): ELIS match happy path, rtol skip,
product-not-in-chain skip, reaction-added-to-chain, ground-only, and the full
decorate -> export -> reload fold round-trip. The later sections cover the
hybrid ``elis_lfs_order`` mapping mode (ELIS first, then a collision-aware
positional pass, then minted orphans) and the ``--orphan-policy`` triad in the
writing layer.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import openmc.deplete
from openmc.deplete import Chain, Nuclide
import openmc.deplete.decay_elis as decay_elis
from openmc.deplete.decay_elis import (
    DecayState, lookup_liso, parse_decay_isomeric_levels,
)

# The patcher lives in the repo's tools/ directory (not an installed package).
_TOOLS = Path(openmc.deplete.__file__).parents[2] / "tools"
sys.path.insert(0, str(_TOOLS))

from add_pendf_isomeric_branching_to_chain import (  # noqa: E402
    _classify_metastables, map_library, decorate_chain, _audit_reaction,
    _self_loop_ground, _classify_elis_lfs_order, _classify_lfs_order,
    _hybrid_census, write_isomer_mapping_log, MAPPING_MODES, MODE_DEFAULT_RTOL,
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
    """Minimal PENDF source adapter over a dict of per-nuclide reactions.

    A reaction dict may carry an ``energy``/``xs`` pair (the MF=3 total) and each
    partial dict may carry its own ``energy``/``xs`` (an MF=10 partial); when
    present these back the pointwise consistency audit. Reactions/partials
    without arrays make ``total_xs``/``pathway_xs`` raise, so the audit becomes a
    no-op for them -- keeping the array-free legacy tests unchanged.

    ``elis`` maps a nuclide to its own MF=1/451 excitation energy [eV]; an
    unlisted nuclide reports ``None`` (the "source has no ELIS" case).
    """

    kind = "fake"
    library = "synthetic"
    mapping = "elis"

    def __init__(self, data, elis=None):
        self._data = data
        self._elis = elis or {}
        self.nuclides = sorted(data)

    def reactions(self, nuclide):
        return self._data[nuclide]

    def total_xs(self, nuclide, mt):
        rx = self._data[nuclide][mt]
        if "energy" in rx and "xs" in rx:
            return np.asarray(rx["energy"], float), np.asarray(rx["xs"], float)
        raise KeyError(f"{nuclide} MT={mt} has no MF=3 total array.")

    def pathway_xs(self, nuclide, mt, lfs, izap=None):
        for p in self._data[nuclide][mt]["partials"]:
            if p["lfs"] == lfs and (izap is None or p["izap"] == izap):
                if "energy" in p and "xs" in p:
                    return (np.asarray(p["energy"], float),
                            np.asarray(p["xs"], float))
                break
        raise KeyError(f"{nuclide} MT={mt} LFS={lfs} has no partial array.")

    def nuclide_elis(self, nuclide):
        return self._elis.get(nuclide)

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


def test_main_stamps_chain_provenance(tmp_path, monkeypatch):
    # main() writes the PENDF source identity onto the exported chain root:
    # pendf_library (source string), pendf_nuclides (count), pendf_source
    # (file basename / dir last-two), plus provenance-only decay_source (basename)
    # and decay_library (tape identity, 'unknown' when unreadable). Uses the module
    # fixtures with open_pendf_source / parse_decay_isomeric_levels monkeypatched so
    # no real data files are needed.
    import add_pendf_isomeric_branching_to_chain as patcher

    base = _chain_with(["In115", "In116", "In116_m1", "In116_m2"],
                       reactions={"In115": [("(n,gamma)", "In116", 6784720.0)]})
    base_xml = tmp_path / "base.xml"
    base.export_to_xml(str(base_xml))

    fake_source = _FakeSource({"In115": {
        102: dict(qm=6784720.0, qi=6784720.0,
                  partials=_in115_ng_metastables())}})
    monkeypatch.setattr(patcher, "open_pendf_source",
                        lambda path, library=None, **kwargs: fake_source)
    monkeypatch.setattr(patcher, "parse_decay_isomeric_levels",
                        lambda decay_file: _decay_lookup())

    pendf_path = tmp_path / "TENDL2017-IST.293K.PENDF.h5"
    out = tmp_path / "chain_out.xml"
    patcher.main(base_chain_file=str(base_xml), pendf_path=str(pendf_path),
                 decay_file="ignored", output_chain_file=str(out),
                 mapping_mode="elis", verbose=False)

    import lxml.etree as ET
    root = ET.parse(str(out)).getroot()
    assert root.get("pendf_library") == "synthetic"     # _FakeSource.library
    assert root.get("pendf_nuclides") == "1"            # one nuclide (In115)
    assert root.get("pendf_source") == "TENDL2017-IST.293K.PENDF.h5"
    assert root.get("decay_source") == "ignored"        # basename of decay_file
    # "ignored" is unreadable as a tape -> tape_identity None -> 'unknown'.
    assert root.get("decay_library") == "unknown"

    # And the stamp round-trips back through Chain.from_xml.
    reread = Chain.from_xml(str(out))
    assert reread.root_attrs == {
        "pendf_library": "synthetic",
        "pendf_nuclides": "1",
        "pendf_source": "TENDL2017-IST.293K.PENDF.h5",
        "decay_source": "ignored",
        "decay_library": "unknown",
    }


def test_elis_matched_stat_counts():
    chain = _chain_with(["In115", "In116", "In116_m1", "In116_m2"],
                        reactions={"In115": [("(n,gamma)", "In116", 6784720.0)]})
    source = _FakeSource({"In115": {
        102: dict(qm=6784720.0, qi=6784720.0, partials=_in115_ng_metastables()),
    }})
    _, stats = map_library(source, chain, _decay_lookup(), "elis", 0.50, 0.0)
    assert stats["matched"] == 2
    assert stats["nuclides_with_branching"] == 1


# ---------------------------------------------------------------------------
# MF=10 consistency audit + threshold-gated rejection + decay-gap log
# ---------------------------------------------------------------------------

def _decay_lookup_with_in114():
    """`_decay_lookup` plus In114 (ground + m1 at ~190 keV)."""
    d = _decay_lookup()
    d[(49, 114)] = [
        DecayState(49, 114, 0.0, 0, half_life=None),
        DecayState(49, 114, 190000.0, 1, half_life=4.3e6),
    ]
    return d


def _in115_ng_offender():
    """In115 (n,gamma): ground+m1+m4 partials whose sum trails the MF=3 total.

    On grid E=[1,2,3] eV the MF=3 total is [10,20,30] b; the partials sum to
    [10,20,27] b, so the worst relative deviation is 3/30 = 0.10 at E=3 eV. The
    v2 lethargy-weighted (int sigma/E dE) integral ratio is 19.5/20 = 0.975
    (all three grid points fall in the epithermal band [0.625 eV, 1e5 eV), so
    ratio_epithermal is the same 0.975 while the other three bands have no
    points -> None).
    """
    grid = [1.0, 2.0, 3.0]
    return dict(qm=6784720.0, qi=6784720.0, energy=grid, xs=[10.0, 20.0, 30.0],
                partials=[
                    dict(lfs=0, izap=49116, qi=6784720.0, qm=6784720.0,
                         elfs=0.0, energy=grid, xs=[6.0, 12.0, 15.0]),
                    dict(lfs=1, izap=49116, qi=6657450.0, qm=6784720.0,
                         elfs=127270.0, energy=grid, xs=[3.0, 6.0, 9.0]),
                    dict(lfs=4, izap=49116, qi=6495060.0, qm=6784720.0,
                         elfs=289660.0, energy=grid, xs=[1.0, 2.0, 3.0])])


def _in113_ng_clean():
    """In113 (n,gamma): ground+m1 partials that sum exactly to the MF=3 total."""
    grid = [1.0, 2.0, 3.0]
    return dict(qm=0.0, qi=0.0, energy=grid, xs=[10.0, 20.0, 30.0],
                partials=[
                    dict(lfs=0, izap=49114, qi=0.0, qm=0.0, elfs=0.0,
                         energy=grid, xs=[7.0, 14.0, 21.0]),
                    dict(lfs=1, izap=49114, qi=-190000.0, qm=0.0,
                         elfs=190000.0, energy=grid, xs=[3.0, 6.0, 9.0])])


def test_audit_worst_dev_and_integral_ratio():
    # Directly exercise the pointwise audit on a constructed mismatch. In v2 the
    # integral_ratio is LETHARGY-weighted (int sigma/E dE), so 19.5/20 = 0.975
    # here, not the v1 unweighted-dE 38.5/40 = 0.9625.
    source = _FakeSource({"In115": {102: _in115_ng_offender()}})
    audit = _audit_reaction(source, "In115", 102,
                            source.reactions("In115")[102]["partials"])
    assert audit is not None
    assert audit["worst_dev"] == pytest.approx(0.10)
    assert audit["energy"] == pytest.approx(3.0)
    assert audit["total"] == pytest.approx(30.0)
    assert audit["sum_partials"] == pytest.approx(27.0)
    assert audit["integral_ratio"] == pytest.approx(0.975)
    # All three grid points sit in the epithermal band [0.625, 1e5) -> the other
    # three bands have no points and read None.
    assert audit["ratio_thermal"] is None
    assert audit["ratio_epithermal"] == pytest.approx(0.975)
    assert audit["ratio_intermediate"] is None
    assert audit["ratio_fast"] is None
    assert audit["notes"] == ""


def test_audit_floor_dust_not_offender():
    # Both the MF=3 total and the summed partials sit below CONSISTENCY_ABS_FLOOR
    # (1e-15 b): floor dust, exempt -> worst_dev 0.0, no energy recorded.
    grid = [1.0, 2.0]
    rxn = dict(qm=0.0, qi=0.0, energy=grid, xs=[1e-20, 1e-20], partials=[
        dict(lfs=0, izap=49116, qi=0.0, qm=0.0, elfs=0.0,
             energy=grid, xs=[1e-20, 1e-20]),
        dict(lfs=1, izap=49116, qi=6657450.0, qm=6784720.0, elfs=127270.0,
             energy=grid, xs=[5e-21, 5e-21])])
    source = _FakeSource({"In115": {102: rxn}})
    audit = _audit_reaction(source, "In115", 102, rxn["partials"])
    assert audit["worst_dev"] == 0.0
    assert audit["energy"] is None

    chain = _chain_with(["In115", "In116", "In116_m1"],
                        reactions={"In115": [("(n,gamma)", "In116", 0.0)]})
    _, stats = map_library(source, chain, _decay_lookup(), "elis", 0.50, 0.0)
    assert stats["audit_offenders"] == 0


def _audit_scene():
    """Chain + source with one offender (In115) and one clean (In113) reaction."""
    chain = _chain_with(
        ["In113", "In114", "In114_m1", "In115", "In116", "In116_m1",
         "In116_m2"],
        reactions={"In115": [("(n,gamma)", "In116", 6784720.0)],
                   "In113": [("(n,gamma)", "In114", 0.0)]})
    source = _FakeSource({
        "In115": {102: _in115_ng_offender()},
        "In113": {102: _in113_ng_clean()},
    })
    return chain, source


def test_audit_offender_logged_when_rejection_off():
    # Default (flag off): the offender is detected + counted but STILL decorated.
    chain, source = _audit_scene()
    branching, stats = map_library(source, chain, _decay_lookup_with_in114(),
                                   "elis", 0.50, 0.0, reject_rtol=None)
    assert stats["audit_offenders"] == 1
    assert stats["audit_clean"] == 1
    assert stats["rejected_count"] == 0
    assert "In115" in branching                      # bad reaction decorated
    assert "In113" in branching

    off = stats["audit_offenders_list"][0]
    assert off["parent"] == "In115"
    assert off["worst_dev"] == pytest.approx(0.10)

    decorate_chain(chain, branching)
    assert "(n,gamma)_m1" in {rx.type for rx in chain["In115"].reactions}


def test_rejection_leaves_offender_stock(tmp_path):
    # Flag on with a threshold below the offender's dev: In115 stays stock while
    # the clean In113 reaction is still decorated.
    chain, source = _audit_scene()
    branching, stats = map_library(source, chain, _decay_lookup_with_in114(),
                                   "elis", 0.50, 0.0, reject_rtol=0.05)
    assert stats["rejected_count"] == 1
    assert stats["audit_offenders"] == 1
    assert stats["rejected"][0]["parent"] == "In115"
    assert stats["rejected"][0]["threshold"] == 0.05
    # (n,gamma) ground product In116 != parent: no self-loop, so the rtol gate
    # is NOT exempted and it alone carries the rejection.
    assert "worst_dev" in stats["rejected"][0]["criterion"]
    assert stats["rtol_reject_exempt"] == 0
    assert "In115" not in branching                  # rejected -> not decorated
    assert "In113" in branching                      # clean -> decorated

    reactions_added = decorate_chain(chain, branching)
    stats["reactions_added"] = reactions_added

    out = tmp_path / "chain.xml"
    chain.export_to_xml(out)
    import xml.etree.ElementTree as ET
    root = ET.parse(out).getroot()

    def nuc(name):
        return next(n for n in root.findall("nuclide")
                    if n.get("name") == name)

    assert nuc("In115").find(".//isomeric_branching") is None
    assert nuc("In113").find(".//isomeric_branching") is not None

    # The rejection is recorded in the log, not the chain XML.
    log = tmp_path / "log.txt"
    source_stats = dict(base_chain="c", pendf="p", decay_file="d",
                        output_chain=str(out), chain_nuclides=len(chain.nuclides))
    from add_pendf_isomeric_branching_to_chain import write_isomer_mapping_log
    write_isomer_mapping_log(log, stats, source_stats, "elis", 0.50, 0.0)
    text = log.read_text()
    assert "MF=10 REJECTED REACTIONS" in text
    assert "MF=10 CONSISTENCY AUDIT" in text
    assert "In115" in text.split("MF=10 REJECTED REACTIONS", 1)[1]


def test_rejected_section_disabled_message(tmp_path):
    # With the flag off, the rejected section shows the audit-only one-liner.
    chain, source = _audit_scene()
    _, stats = map_library(source, chain, _decay_lookup_with_in114(),
                           "elis", 0.50, 0.0, reject_rtol=None)
    stats["reactions_added"] = 0
    log = tmp_path / "log.txt"
    source_stats = dict(base_chain="c", pendf="p", decay_file="d",
                        output_chain="o", chain_nuclides=len(chain.nuclides))
    from add_pendf_isomeric_branching_to_chain import write_isomer_mapping_log
    write_isomer_mapping_log(log, stats, source_stats, "elis", 0.50, 0.0)
    text = log.read_text()
    assert "rejection disabled (audit only)" in text


# ---------------------------------------------------------------------------
# Metric v2: emax cap + lethargy-weighted band ratios + band-ratio rejection
# ---------------------------------------------------------------------------

def test_audit_mismatch_above_emax_invisible():
    # A mismatch that exists ONLY above emax is truncated away: with the default
    # 2e7 cap the summed partials match the total on every surviving point, so
    # the reaction audits clean. Lifting the cap past the offending point makes
    # the same reaction a worst_dev=1.0 offender -- proving the cap is the cause.
    grid = [1.0, 1.0e3, 1.0e6, 1.0e7, 5.0e7]
    total = [10.0, 10.0, 10.0, 10.0, 10.0]
    rxn = dict(qm=0.0, qi=0.0, energy=grid, xs=total, partials=[
        # Ground matches the total up to 1e7, then collapses to 0 at 5e7.
        dict(lfs=0, izap=49116, qi=0.0, qm=0.0, elfs=0.0,
             energy=grid, xs=[10.0, 10.0, 10.0, 10.0, 0.0]),
        # A (zero-valued) metastable so the reaction carries a pathway.
        dict(lfs=1, izap=49116, qi=6657450.0, qm=6784720.0, elfs=127270.0,
             energy=grid, xs=[0.0, 0.0, 0.0, 0.0, 0.0])])
    source = _FakeSource({"In115": {102: rxn}})

    capped = _audit_reaction(source, "In115", 102, rxn["partials"])
    assert capped["worst_dev"] == pytest.approx(0.0)

    uncapped = _audit_reaction(source, "In115", 102, rxn["partials"], emax=1.0e8)
    assert uncapped["worst_dev"] == pytest.approx(1.0)
    assert uncapped["energy"] == pytest.approx(5.0e7)

    # map_library uses the default cap -> the reaction is NOT an offender.
    chain = _chain_with(["In115", "In116", "In116_m1"],
                        reactions={"In115": [("(n,gamma)", "In116", 0.0)]})
    _, stats = map_library(source, chain, _decay_lookup(), "elis", 0.50, 0.0)
    assert stats["audit_offenders"] == 0


def test_audit_band_ratios_thermal_only():
    # partials = 0.5x total in the thermal band [<0.625 eV) and = total
    # elsewhere -> ratio_thermal ~ 0.5; epithermal/intermediate/fast ~ 1.0. The
    # grid seeds >=2 points in every one of the four windows.
    grid = [0.01, 0.1, 1.0, 100.0, 1.0e4, 2.0e5, 5.0e5, 2.0e6, 1.0e7]
    total = [10.0] * 9
    ground = [5.0, 5.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    rxn = dict(qm=0.0, qi=0.0, energy=grid, xs=total, partials=[
        dict(lfs=0, izap=49116, qi=0.0, qm=0.0, elfs=0.0,
             energy=grid, xs=ground)])
    source = _FakeSource({"In115": {102: rxn}})
    audit = _audit_reaction(source, "In115", 102, rxn["partials"])
    assert audit["ratio_thermal"] == pytest.approx(0.5)
    assert audit["ratio_epithermal"] == pytest.approx(1.0)
    assert audit["ratio_intermediate"] == pytest.approx(1.0)
    assert audit["ratio_fast"] == pytest.approx(1.0)
    assert audit["notes"] == ""


def test_audit_four_way_split_intermediate_only():
    # A fixture broken ONLY in the intermediate band [1e5, 1e6): partials are
    # 0.5x total there and = total in the thermal/epithermal/fast bands. The
    # four-way split isolates it -> ratio_intermediate ~ 0.5 while the other
    # three read ~ 1.0. Rejection is ONE-SIDED: an UNDER-summing band (0.5 < 1)
    # never rejects -- deep silence is the collapse silence-fill's job -- so the
    # reaction decorates despite reject_band_ratio=0.3.
    grid = [0.01, 0.1, 1.0, 100.0, 1.0e4, 2.0e5, 5.0e5, 2.0e6, 1.0e7]
    total = [10.0] * 9
    ground = [10.0, 10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 10.0, 10.0]  # 0.5x in [1e5,1e6)
    rxn = dict(qm=6784720.0, qi=6784720.0, energy=grid, xs=total, partials=[
        dict(lfs=0, izap=49116, qi=6784720.0, qm=6784720.0, elfs=0.0,
             energy=grid, xs=ground),
        # metastable maps to In116_m1 (would decorate absent rejection)
        dict(lfs=1, izap=49116, qi=6657450.0, qm=6784720.0, elfs=127270.0,
             energy=grid, xs=[0.0] * 9)])
    source = _FakeSource({"In115": {102: rxn}})

    audit = _audit_reaction(source, "In115", 102, rxn["partials"])
    assert audit["ratio_thermal"] == pytest.approx(1.0)
    assert audit["ratio_epithermal"] == pytest.approx(1.0)
    assert audit["ratio_intermediate"] == pytest.approx(0.5)
    assert audit["ratio_fast"] == pytest.approx(1.0)

    chain = _chain_with(["In115", "In116", "In116_m1"],
                        reactions={"In115": [("(n,gamma)", "In116", 6784720.0)]})
    branching, stats = map_library(source, chain, _decay_lookup(),
                                   "elis", 0.50, 0.0,
                                   reject_rtol=None, reject_band_ratio=0.3)
    assert stats["rejected_count"] == 0              # under-summing -> not rejected
    assert "In115" in branching                      # decorated


def test_reject_band_ratio_leaves_offender_stock(tmp_path):
    # reject_band_ratio set (reject_rtol unset): a reaction OVER-summing only in
    # the thermal band (partials 1.3x total -- MF=10 corruption) is rejected on
    # the one-sided band criterion and left stock in the XML.
    grid = [0.01, 0.1, 1.0, 100.0, 1.0e4, 1.0e5, 1.0e6, 1.0e7]
    total = [10.0] * 8
    ground = [13.0, 13.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]   # 1.3x in thermal
    rxn = dict(qm=0.0, qi=0.0, energy=grid, xs=total, partials=[
        dict(lfs=0, izap=49116, qi=0.0, qm=0.0, elfs=0.0,
             energy=grid, xs=ground),
        # metastable maps to In116_m1 (would decorate absent rejection)
        dict(lfs=1, izap=49116, qi=6657450.0, qm=6784720.0, elfs=127270.0,
             energy=grid, xs=[0.0] * 8)])
    source = _FakeSource({"In115": {102: rxn}})
    chain = _chain_with(["In115", "In116", "In116_m1"],
                        reactions={"In115": [("(n,gamma)", "In116", 0.0)]})

    branching, stats = map_library(source, chain, _decay_lookup(),
                                   "elis", 0.50, 0.0,
                                   reject_rtol=None, reject_band_ratio=0.1)
    assert stats["rejected_count"] == 1
    rej = stats["rejected"][0]
    assert rej["parent"] == "In115"
    assert "band_ratio" in rej["criterion"]
    assert "thermal" in rej["criterion"]
    assert rej["threshold"] is None                  # reject_rtol was unset
    assert "In115" not in branching                  # rejected -> not decorated

    reactions_added = decorate_chain(chain, branching)
    stats["reactions_added"] = reactions_added
    out = tmp_path / "chain.xml"
    chain.export_to_xml(out)
    import xml.etree.ElementTree as ET
    root = ET.parse(out).getroot()
    nuc = next(n for n in root.findall("nuclide") if n.get("name") == "In115")
    assert nuc.find(".//isomeric_branching") is None

    log = tmp_path / "log.txt"
    source_stats = dict(base_chain="c", pendf="p", decay_file="d",
                        output_chain=str(out), chain_nuclides=len(chain.nuclides))
    from add_pendf_isomeric_branching_to_chain import write_isomer_mapping_log
    write_isomer_mapping_log(log, stats, source_stats, "elis", 0.50, 0.0)
    text = log.read_text()
    assert "band_ratio:thermal" in text


def test_band_ratio_none_when_grid_misses_band(tmp_path):
    # A grid that only enters the epithermal band leaves the thermal,
    # intermediate, and fast ratios None ('n/a' in the log) and they can NEVER
    # trigger band rejection.
    grid = [10.0, 100.0, 1000.0]                     # all in the epithermal band
    total = [10.0, 20.0, 30.0]
    rxn = dict(qm=0.0, qi=0.0, energy=grid, xs=total, partials=[
        dict(lfs=0, izap=49116, qi=0.0, qm=0.0, elfs=0.0,
             energy=grid, xs=[7.0, 14.0, 20.9]),      # sum trails total at 1000
        dict(lfs=1, izap=49116, qi=6657450.0, qm=6784720.0, elfs=127270.0,
             energy=grid, xs=[3.0, 6.0, 9.0])])
    source = _FakeSource({"In115": {102: rxn}})
    audit = _audit_reaction(source, "In115", 102, rxn["partials"])
    assert audit["ratio_thermal"] is None
    assert audit["ratio_intermediate"] is None
    assert audit["ratio_fast"] is None
    assert audit["ratio_epithermal"] is not None
    assert audit["worst_dev"] > 1e-5                 # an offender -> logged

    chain = _chain_with(["In115", "In116", "In116_m1"],
                        reactions={"In115": [("(n,gamma)", "In116", 0.0)]})
    branching, stats = map_library(source, chain, _decay_lookup(),
                                   "elis", 0.50, 0.0, reject_band_ratio=0.1)
    assert stats["rejected_count"] == 0              # None bands never trigger
    assert "In115" in branching

    stats["reactions_added"] = 0
    log = tmp_path / "log.txt"
    source_stats = dict(base_chain="c", pendf="p", decay_file="d",
                        output_chain="o", chain_nuclides=len(chain.nuclides))
    from add_pendf_isomeric_branching_to_chain import write_isomer_mapping_log
    write_isomer_mapping_log(log, stats, source_stats, "elis", 0.50, 0.0)
    audit_section = log.read_text().split("MF=10 CONSISTENCY AUDIT", 1)[1]
    assert "n/a" in audit_section


def test_absent_from_decay_section_unique_and_grouped(tmp_path):
    # Y89 (no decay data at all) surfaced by TWO reactions must be listed once;
    # Zn66 (present, ground-only) lands under no_metastables.
    chain = _chain_with(["Co59", "Zn64"])
    source = _FakeSource({
        "Co59": {
            102: dict(qm=0.0, qi=0.0, partials=[
                dict(lfs=1, izap=39089, qi=0.0, qm=100000.0, elfs=100000.0)]),
            16: dict(qm=0.0, qi=0.0, partials=[
                dict(lfs=1, izap=39089, qi=0.0, qm=120000.0, elfs=120000.0)]),
        },
        "Zn64": {
            102: dict(qm=0.0, qi=0.0, partials=[
                dict(lfs=1, izap=30066, qi=0.0, qm=90000.0, elfs=90000.0)]),
        },
    })
    decay = {(30, 66): [DecayState(30, 66, 0.0, 0, half_life=None)]}
    _, stats = map_library(source, chain, decay, "elis", 0.50, 0.0)

    assert stats["absent_unique_count"] == 2
    assert stats["absent_by_status"]["no_decay_data"] == ["Y89"]
    assert stats["absent_by_status"]["no_metastables"] == ["Zn66"]

    stats["reactions_added"] = 0
    log = tmp_path / "log.txt"
    source_stats = dict(base_chain="c", pendf="p", decay_file="d",
                        output_chain="o", chain_nuclides=len(chain.nuclides))
    from add_pendf_isomeric_branching_to_chain import write_isomer_mapping_log
    write_isomer_mapping_log(log, stats, source_stats, "elis", 0.50, 0.0)
    section = log.read_text().split(
        "NUCLIDES ABSENT FROM DECAY LIBRARY", 1)[1]
    assert section.count("Y89") == 1
    assert "Zn66" in section
    assert "no_decay_data" in section
    assert "no_metastables" in section


# ---------------------------------------------------------------------------
# Criterion refinements: band significance floor + self-loop-ground exemption
# ---------------------------------------------------------------------------

def test_band_dust_floor_thermal_none_no_reject():
    # The thermal band [<1 eV) total sits entirely below CONSISTENCY_ABS_FLOOR
    # (1e-15 b) -- evaluator dust -- so ratio_thermal is None and can NEVER
    # trigger band rejection, even at a tiny threshold, despite the ground
    # partial reading 0 there (a spurious ~0 ratio absent the floor). The real
    # epithermal/fast bands stay consistent, so nothing is rejected.
    grid = [0.01, 0.1, 1.0, 100.0, 1.0e4, 1.0e5, 1.0e6, 1.0e7]
    total = [1e-20, 1e-20, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    ground = [0.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]   # 0 in thermal dust
    rxn = dict(qm=6784720.0, qi=6784720.0, energy=grid, xs=total, partials=[
        dict(lfs=0, izap=49116, qi=6784720.0, qm=6784720.0, elfs=0.0,
             energy=grid, xs=ground),
        dict(lfs=1, izap=49116, qi=6657450.0, qm=6784720.0, elfs=127270.0,
             energy=grid, xs=[0.0] * 8)])
    source = _FakeSource({"In115": {102: rxn}})
    audit = _audit_reaction(source, "In115", 102, rxn["partials"])
    assert audit["ratio_thermal"] is None
    assert audit["ratio_epithermal"] == pytest.approx(1.0)
    assert audit["ratio_fast"] == pytest.approx(1.0)

    chain = _chain_with(["In115", "In116", "In116_m1"],
                        reactions={"In115": [("(n,gamma)", "In116", 6784720.0)]})
    branching, stats = map_library(source, chain, _decay_lookup(),
                                   "elis", 0.50, 0.0, reject_band_ratio=1e-6)
    assert stats["rejected_count"] == 0              # dust thermal never fires
    assert "In115" in branching


def test_self_loop_ground_band_reject_exempt():
    # In113 (n,n') is a ground-state self-loop: its LFS=0 partial (izap 49113)
    # produces In113 itself, so the ground pathway is a transmutation-matrix
    # no-op. The fast band OVER-sums (partials 1.5x total -- MF=10 corruption
    # that the one-sided gate would reject), but the ground route is a no-op and
    # only the m1 partial carries real isomer production, so the gate is exempted.
    grid = [0.01, 0.1, 1.0, 100.0, 1.0e4, 1.0e5, 1.0e6, 1.0e7]
    total = [0.0, 0.0, 0.0, 0.0, 0.0, 40.0, 50.0, 60.0]
    ground = [0.0, 0.0, 0.0, 0.0, 0.0, 40.0, 50.0, 60.0]   # self-loop ground = total
    meta = [0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 25.0, 30.0]     # sum_all = 1.5x (fast, over)
    rxn = dict(qm=0.0, qi=0.0, energy=grid, xs=total, partials=[
        dict(lfs=0, izap=49113, qi=0.0, qm=0.0, elfs=0.0,
             energy=grid, xs=ground),
        dict(lfs=1, izap=49113, qi=-391700.0, qm=0.0, elfs=391700.0,
             energy=grid, xs=meta)])
    source = _FakeSource({"In113": {4: rxn}})
    decay = {(49, 113): [
        DecayState(49, 113, 0.0, 0, half_life=None),
        DecayState(49, 113, 391700.0, 1, half_life=6000.0)]}
    chain = _chain_with(["In113", "In113_m1"],
                        reactions={"In113": [("(n,n')", "In113", 0.0)]})
    branching, stats = map_library(source, chain, decay, "elis", 0.50, 0.0,
                                   reject_rtol=None, reject_band_ratio=0.3)
    assert stats["rejected_count"] == 0              # exempted -> not rejected
    assert stats["band_reject_exempt"] == 1
    assert "In113" in branching                      # decorated, not stock

    off = stats["audit_offenders_list"][0]
    assert off["parent"] == "In113"
    assert "self-loop ground" in off["notes"]        # marker for transparency

    decorate_chain(chain, branching)
    assert "(n,n')_m1" in {rx.type for rx in chain["In113"].reactions}


def test_metastable_parent_ground_not_exempt():
    # In115_m1 (n,n') -> In115 (ground) is isomer BURNUP, a real transition, not
    # a self-loop: gnds_name(49,115,0)='In115' != parent 'In115_m1'. An
    # OVER-summing fast band must therefore STILL trigger the one-sided band
    # rejection (no exemption).
    grid = [0.01, 0.1, 1.0, 100.0, 1.0e4, 1.0e5, 1.0e6, 1.0e7]
    total = [0.0, 0.0, 0.0, 0.0, 0.0, 40.0, 50.0, 60.0]
    ground = [0.0, 0.0, 0.0, 0.0, 0.0, 60.0, 75.0, 90.0]     # 1.5x total (fast, over)
    rxn = dict(qm=0.0, qi=0.0, energy=grid, xs=total, partials=[
        dict(lfs=0, izap=49115, qi=0.0, qm=0.0, elfs=0.0,
             energy=grid, xs=ground),
        dict(lfs=1, izap=49115, qi=-336000.0, qm=0.0, elfs=336000.0,
             energy=grid, xs=[0.0] * 8)])
    source = _FakeSource({"In115_m1": {4: rxn}})
    decay = {(49, 115): [
        DecayState(49, 115, 0.0, 0, half_life=None),
        DecayState(49, 115, 336000.0, 1, half_life=1.6e14)]}
    chain = _chain_with(["In115", "In115_m1"],
                        reactions={"In115_m1": [("(n,n')", "In115", 0.0)]})
    branching, stats = map_library(source, chain, decay, "elis", 0.50, 0.0,
                                   reject_rtol=None, reject_band_ratio=0.3)
    assert stats["rejected_count"] == 1              # no exemption -> rejected
    assert stats["band_reject_exempt"] == 0
    assert stats["rejected"][0]["parent"] == "In115_m1"
    assert "band_ratio" in stats["rejected"][0]["criterion"]
    assert "In115_m1" not in branching               # rejected -> not decorated


def test_synthesized_nn_prime_ground_parent_exempt():
    # (n,n') on a GROUND-state parent with NO LFS=0 partial and NO base-chain
    # (n,n'): decorate_chain synthesizes the ground as target == parent, so the
    # audit must agree it is a self-loop and exempt the OVER-summing fast band.
    # (Before this, _self_loop_ground returned False for exactly the reactions
    # the decoration synthesizes -- audit and decoration disagreed.)
    grid = [0.01, 0.1, 1.0, 100.0, 1.0e4, 1.0e5, 1.0e6, 1.0e7]
    total = [0.0, 0.0, 0.0, 0.0, 0.0, 40.0, 50.0, 60.0]
    meta = [0.0, 0.0, 0.0, 0.0, 0.0, 60.0, 75.0, 90.0]     # 1.5x total (fast, over)
    rxn = dict(qm=0.0, qi=0.0, energy=grid, xs=total, partials=[
        dict(lfs=1, izap=49113, qi=-391700.0, qm=0.0, elfs=391700.0,
             energy=grid, xs=meta)])
    source = _FakeSource({"In113": {4: rxn}})
    decay = {(49, 113): [
        DecayState(49, 113, 0.0, 0, half_life=None),
        DecayState(49, 113, 391700.0, 1, half_life=6000.0)]}
    chain = _chain_with(["In113", "In113_m1"])             # no base (n,n')
    branching, stats = map_library(source, chain, decay, "elis", 0.50, 0.0,
                                   reject_rtol=None, reject_band_ratio=0.3)
    assert stats["rejected_count"] == 0                    # exempted
    assert stats["band_reject_exempt"] == 1
    assert "In113" in branching                            # decorated, not stock

    decorate_chain(chain, branching)
    rxns = {rx.type: rx.target for rx in chain["In113"].reactions}
    assert rxns["(n,n')"] == "In113"                       # the synthesized ground
    assert rxns["(n,n')_m1"] == "In113_m1"


def test_synthesized_nn_prime_metastable_parent_not_exempt():
    # Same shape on a METASTABLE parent (In115_m1): its (n,n') ground route is
    # real isomer burnup, never a self-loop, so the over-summing fast band still
    # rejects even though the base chain carries no (n,n') either.
    grid = [0.01, 0.1, 1.0, 100.0, 1.0e4, 1.0e5, 1.0e6, 1.0e7]
    total = [0.0, 0.0, 0.0, 0.0, 0.0, 40.0, 50.0, 60.0]
    meta = [0.0, 0.0, 0.0, 0.0, 0.0, 60.0, 75.0, 90.0]     # 1.5x total (fast, over)
    rxn = dict(qm=0.0, qi=0.0, energy=grid, xs=total, partials=[
        dict(lfs=1, izap=49115, qi=-336000.0, qm=0.0, elfs=336000.0,
             energy=grid, xs=meta)])
    source = _FakeSource({"In115_m1": {4: rxn}})
    decay = {(49, 115): [
        DecayState(49, 115, 0.0, 0, half_life=None),
        DecayState(49, 115, 336000.0, 1, half_life=1.6e14)]}
    chain = _chain_with(["In115", "In115_m1"])             # no base (n,n')
    branching, stats = map_library(source, chain, decay, "elis", 0.50, 0.0,
                                   reject_rtol=None, reject_band_ratio=0.3)
    assert stats["rejected_count"] == 1                    # no exemption
    assert stats["band_reject_exempt"] == 0
    assert stats["rejected"][0]["parent"] == "In115_m1"
    assert "In115_m1" not in branching                     # rejected -> stock


def test_synthesized_nn_prime_metastable_parent_targets_true_ground(tmp_path):
    # In115_m1 (n,n') with NO base-chain (n,n'): the synthesized ground is the
    # super-elastic de-excitation to the TRUE ground (In115) with Q = +ELIS from
    # the source, never the In115_m1 -> In115_m1 self-transition. The audit
    # agrees -- such a reaction is never self-loop-exempt, so it stays
    # rejectable.
    rxn = dict(qm=0.0, qi=336000.0, partials=[
        dict(lfs=1, izap=49115, qi=0.0, qm=0.0, elfs=336000.0)])
    source = _FakeSource({"In115_m1": {4: rxn}}, elis={"In115_m1": 336000.0})
    decay = {(49, 115): [
        DecayState(49, 115, 0.0, 0, half_life=None),
        DecayState(49, 115, 336000.0, 1, half_life=1.6e14)]}
    chain = _chain_with(["In115", "In115_m1"])             # no base (n,n')
    branching, stats = map_library(source, chain, decay, "elis", 0.50, 0.0)
    assert _self_loop_ground("In115_m1", "(n,n')", rxn["partials"],
                             chain) is False              # rejectable, not exempt

    added = decorate_chain(chain, branching, stats)
    assert added == 1
    rxns = {rx.type: (rx.target, rx.Q, rx.pendf_lfs)
            for rx in chain["In115_m1"].reactions}
    assert rxns["(n,n')"] == ("In115", 336000.0, 0)        # m -> g, Q = +ELIS
    assert rxns["(n,n')_m1"][0] == "In115_m1"
    assert stats["mg_ground_synthesized"] == 1
    assert stats["mg_ground_skipped"] == 0

    stats["reactions_added"] = added
    log = tmp_path / "log.txt"
    source_stats = dict(base_chain="c", pendf="p", decay_file="d",
                        output_chain="o", chain_nuclides=len(chain.nuclides))
    from add_pendf_isomeric_branching_to_chain import write_isomer_mapping_log
    write_isomer_mapping_log(log, stats, source_stats, "elis", 0.50, 0.0)
    text = log.read_text()
    assert "Synthesized m->g (n,n') grounds:     1" in text
    assert "Skipped m->g (n,n') (no ELIS/ground):     0" in text
    assert "synthesized m->g (n,n') ground: -> In115, Q=+336000.0 eV" in text


@pytest.mark.parametrize("elis, names, note", [
    (None, ["In115", "In115_m1"], "parent MF=1/451 ELIS absent or 0"),
    ({"In115_m1": 0.0}, ["In115", "In115_m1"], "ELIS absent or 0"),
    ({"In115_m1": 336000.0}, ["In115_m1"], "ground In115 not in chain"),
])
def test_synthesized_nn_prime_metastable_parent_skipped(elis, names, note):
    # No usable ELIS on the parent (absent, or a broken ELIS=0 header), or no
    # true ground in the chain: the m->g (n,n') ground is NOT synthesized (a Q=0
    # self-transition placeholder would be fabricated data) -- the reaction is
    # left stock, counted and noted.
    rxn = dict(qm=0.0, qi=336000.0, partials=[
        dict(lfs=1, izap=49115, qi=0.0, qm=0.0, elfs=336000.0)])
    source = _FakeSource({"In115_m1": {4: rxn}}, elis=elis)
    decay = {(49, 115): [
        DecayState(49, 115, 0.0, 0, half_life=None),
        DecayState(49, 115, 336000.0, 1, half_life=1.6e14)]}
    chain = _chain_with(names)                             # no base (n,n')
    branching, stats = map_library(source, chain, decay, "elis", 0.50, 0.0)
    added = decorate_chain(chain, branching, stats)
    assert added == 0
    assert {rx.type for rx in chain["In115_m1"].reactions} == set()
    assert stats["mg_ground_synthesized"] == 0
    assert stats["mg_ground_skipped"] == 1
    assert note in stats["isomer_mappings"][0]["notes"]


def test_self_loop_ground_rtol_reject_exempt():
    # In113 (n,n') carries NO LFS=0 partial (a stable ground product) and its m1
    # partial only switches on above the metastable threshold, so below it the
    # partials are silent under a live MF=3 total and worst_dev is pinned at
    # exactly 1.0. ANY reject_rtol < 1.0 would therefore reject the whole
    # In113/In115 (n,n') class and destroy its m1 branching -- the self-loop
    # exemption covers the rtol gate too. Band gate off here to isolate it.
    grid = [0.01, 0.1, 1.0, 100.0, 1.0e4, 1.0e5, 1.0e6, 1.0e7]
    total = [10.0] * 8
    meta = [0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0]    # silent -> dev 1.0
    rxn = dict(qm=0.0, qi=0.0, energy=grid, xs=total, partials=[
        dict(lfs=1, izap=49113, qi=-391700.0, qm=0.0, elfs=391700.0,
             energy=grid, xs=meta)])
    source = _FakeSource({"In113": {4: rxn}})
    decay = {(49, 113): [
        DecayState(49, 113, 0.0, 0, half_life=None),
        DecayState(49, 113, 391700.0, 1, half_life=6000.0)]}
    chain = _chain_with(["In113", "In113_m1"],
                        reactions={"In113": [("(n,n')", "In113", 0.0)]})
    assert _audit_reaction(source, "In113", 4,
                           rxn["partials"])["worst_dev"] == pytest.approx(1.0)
    branching, stats = map_library(source, chain, decay, "elis", 0.50, 0.0,
                                   reject_rtol=0.5, reject_band_ratio=None)
    assert stats["rejected_count"] == 0              # exempted -> not rejected
    assert stats["rtol_reject_exempt"] == 1
    assert stats["band_reject_exempt"] == 0          # band gate off
    assert "In113" in branching                      # decorated, not stock

    off = stats["audit_offenders_list"][0]
    assert "self-loop ground: rtol exempt" in off["notes"]

    decorate_chain(chain, branching)
    assert "(n,n')_m1" in {rx.type for rx in chain["In113"].reactions}


# ---------------------------------------------------------------------------
# Metastable-only MF=10 observability (no LFS=0 partial)
# ---------------------------------------------------------------------------

def test_metastable_only_marked_and_counted(tmp_path):
    # MF=10 carrying a metastable partial but NO LFS=0 -- the In113/In115 MT=4
    # class, where a stable ground product means the evaluation legitimately
    # tabulates no ground partial. Observability only: the reaction still
    # decorates, but its audit row is marked and the summary block counts it.
    grid = [1.0, 2.0, 3.0]
    rxn = dict(qm=6784720.0, qi=6784720.0, energy=grid, xs=[10.0, 20.0, 30.0],
               partials=[
                   dict(lfs=1, izap=49116, qi=6657450.0, qm=6784720.0,
                        elfs=127270.0, energy=grid, xs=[3.0, 6.0, 9.0])])
    source = _FakeSource({"In115": {102: rxn}})
    chain = _chain_with(["In115", "In116", "In116_m1"],
                        reactions={"In115": [("(n,gamma)", "In116", 6784720.0)]})
    branching, stats = map_library(source, chain, _decay_lookup(), "elis",
                                   0.50, 0.0, reject_band_ratio=0.3)
    assert stats["metastable_only"] == 1
    assert stats["ground_only"] == 0                 # the opposite class
    assert stats["rejected_count"] == 0              # marker changes no decision
    assert "In115" in branching

    off = stats["audit_offenders_list"][0]           # partials trail the total
    assert "metastable-only MF=10 (no LFS=0)" in off["notes"]

    stats["reactions_added"] = 0
    log = tmp_path / "log.txt"
    source_stats = dict(base_chain="c", pendf="p", decay_file="d",
                        output_chain="o", chain_nuclides=len(chain.nuclides))
    from add_pendf_isomeric_branching_to_chain import write_isomer_mapping_log
    write_isomer_mapping_log(log, stats, source_stats, "elis", 0.50, 0.0)
    text = log.read_text()
    assert "Metastable-only MF=10 reactions:     1" in text
    audit_section = text.split("MF=10 CONSISTENCY AUDIT", 1)[1]
    assert "metastable-only MF=10 (no LFS=0)" in audit_section


def test_band_ratio_rejection_one_sided():
    # One-sided band rejection at threshold 0.3: an UNDER-summing band (ratio ~0,
    # |r-1| = 1) never rejects (deep silence is the collapse silence-fill's job);
    # an OVER-summing band (ratio 1.4) still rejects (partials exceed the total,
    # MF=10 corruption). Both fixtures are broken only in the thermal band.
    grid = [0.01, 0.1, 1.0, 100.0, 1.0e4, 2.0e5, 5.0e5, 2.0e6, 1.0e7]
    total = [10.0] * 9
    chain = _chain_with(["In115", "In116", "In116_m1"],
                        reactions={"In115": [("(n,gamma)", "In116", 0.0)]})

    # Under-summing thermal (ground ~0 while total significant): NOT rejected.
    under = dict(qm=0.0, qi=0.0, energy=grid, xs=total, partials=[
        dict(lfs=0, izap=49116, qi=0.0, qm=0.0, elfs=0.0, energy=grid,
             xs=[0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
        dict(lfs=1, izap=49116, qi=6657450.0, qm=6784720.0, elfs=127270.0,
             energy=grid, xs=[0.0] * 9)])
    src_u = _FakeSource({"In115": {102: under}})
    assert _audit_reaction(src_u, "In115", 102,
                           under["partials"])["ratio_thermal"] == pytest.approx(0.0)
    _, stats_u = map_library(src_u, chain, _decay_lookup(), "elis", 0.50, 0.0,
                             reject_rtol=None, reject_band_ratio=0.3)
    assert stats_u["rejected_count"] == 0            # under-summing spared

    # Over-summing thermal (partials 1.4x total): STILL rejected.
    over = dict(qm=0.0, qi=0.0, energy=grid, xs=total, partials=[
        dict(lfs=0, izap=49116, qi=0.0, qm=0.0, elfs=0.0, energy=grid,
             xs=[14.0, 14.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
        dict(lfs=1, izap=49116, qi=6657450.0, qm=6784720.0, elfs=127270.0,
             energy=grid, xs=[0.0] * 9)])
    src_o = _FakeSource({"In115": {102: over}})
    assert _audit_reaction(src_o, "In115", 102,
                           over["partials"])["ratio_thermal"] == pytest.approx(1.4)
    _, stats_o = map_library(src_o, chain, _decay_lookup(), "elis", 0.50, 0.0,
                             reject_rtol=None, reject_band_ratio=0.3)
    assert stats_o["rejected_count"] == 1            # over-summing rejected
    assert "thermal" in stats_o["rejected"][0]["criterion"]


# ---------------------------------------------------------------------------
# ASC tape source: MF=10 sections with no MF=3 sibling are not decorable
# ---------------------------------------------------------------------------

def _endf_field(value):
    """Format a value as an 11-character ENDF-6 record field."""
    return f"{value:>11.4E}" if isinstance(value, float) else f"{value:>11d}"


def _tab1_lines(c1, c2, l1, l2, pairs):
    """TAB1 record lines for one lin-lin (INT=2) region (NJOY PENDF convention)."""
    lines = [_endf_field(c1) + _endf_field(c2) + _endf_field(l1)
             + _endf_field(l2) + _endf_field(1) + _endf_field(len(pairs)),
             _endf_field(len(pairs)) + _endf_field(2)]      # NBT, INT=lin-lin
    row = ""
    for i, (x, y) in enumerate(pairs):
        row += _endf_field(float(x)) + _endf_field(float(y))
        if (i + 1) % 3 == 0:
            lines.append(row)
            row = ""
    if row:
        lines.append(row)
    return lines


_TAPE_GRID = [(1.0e-5, 4.0), (1.0e6, 4.0)]


def _mf3_text(za, qm, qi):
    head = (_endf_field(float(za)) + _endf_field(114.0) + _endf_field(0)
            + _endf_field(0) + _endf_field(0) + _endf_field(0))
    return "\n".join([head] + _tab1_lines(qm, qi, 0, 0, _TAPE_GRID)) + "\n"


def _mf10_text(za, subs):
    """Synthetic MF=10 section; ``subs`` are ``(QM, QI, IZAP, LFS)`` tuples."""
    lines = [_endf_field(float(za)) + _endf_field(114.0) + _endf_field(0)
             + _endf_field(0) + _endf_field(len(subs)) + _endf_field(0)]
    for qm, qi, izap, lfs in subs:
        lines += _tab1_lines(qm, qi, izap, lfs, _TAPE_GRID)
    return "\n".join(lines) + "\n"


def test_asc_mf10_without_mf3_not_decorable(tmp_path, monkeypatch):
    # A tape whose In115 carries a normal MT=102 (MF=3 + MF=10) alongside a
    # metastable-bearing MT=28 with MF=10 but NO MF=3 -- the JEFF-4.0 W/Ta/Cr
    # class. Every collapse source form drops an MF=3-less MT, so it must not
    # become a chain reaction; it is excluded before any counting and reported
    # instead.
    import types

    import add_pendf_isomeric_branching_to_chain as patcher
    import openmc.data.endf as endf_mod
    import openmc.data.pendf as pendf_mod

    ev = types.SimpleNamespace(
        target=dict(atomic_number=49, mass_number=115, isomeric_state=0,
                    excitation_energy=0.0),
        section={
            (3, 102): _mf3_text(49115, 6784720.0, 6784720.0),
            (10, 102): _mf10_text(49115, [
                (6784720.0, 6784720.0, 49116, 0),
                (6784720.0, 6657450.0, 49116, 1),
                (6784720.0, 6495060.0, 49116, 4)]),
            # (n,np) -> In114/In114_m1, MF=10 only: not a decoration candidate.
            (10, 28): _mf10_text(49115, [(0.0, 0.0, 49114, 0),
                                         (0.0, -190000.0, 49114, 1)]),
        })
    tape = tmp_path / "n-In115.pendf"
    tape.write_text("")
    monkeypatch.setattr(pendf_mod, "_discover_pendf_files",
                        lambda d: [(tape, None)])
    monkeypatch.setattr(endf_mod, "Evaluation", lambda f: ev)
    monkeypatch.setattr(patcher, "tape_identity", lambda p: "synthetic-tape")

    source = patcher._AscSource(tmp_path)
    assert sorted(source.reactions("In115")) == [102]      # MT=28 excluded

    chain = _chain_with(["In115", "In114", "In114_m1", "In116", "In116_m1",
                         "In116_m2"],
                        reactions={"In115": [("(n,gamma)", "In116", 6784720.0)]})
    branching, stats = map_library(source, chain, _decay_lookup_with_in114(),
                                   "elis", 0.50, 0.0)
    added = decorate_chain(chain, branching)
    stats["reactions_added"] = added

    # The MF=10-only MT is neither decorated nor added to the chain...
    assert "(n,np)" not in branching["In115"]
    assert not any(rx.type.startswith("(n,np)") for rx in chain["In115"].reactions)
    # ...while the normal MT=102 still decorates.
    rxns = {rx.type: rx.target for rx in chain["In115"].reactions}
    assert rxns["(n,gamma)_m1"] == "In116_m1"
    assert rxns["(n,gamma)_m2"] == "In116_m2"

    # Counted and described, but never mixed into the candidate stats: only the
    # two MT=102 metastables are LFS found, and the excluded LFS=0 partial of
    # MT=28 does not register as ground-only.
    assert stats["mf10_without_mf3"] == 1
    assert stats["total_lfs"] == 2
    assert stats["ground_only"] == 0
    assert stats["metastable_only"] == 0
    rec = stats["mf10_without_mf3_list"][0]
    assert (rec["parent"], rec["mt"], rec["reaction"]) == ("In115", 28, "(n,np)")
    assert rec["lfs"] == [0, 1] and rec["products"] == ["In114"]
    assert rec["metastable"] is True

    log = tmp_path / "log.txt"
    source_stats = dict(base_chain="c", pendf="p", decay_file="d",
                        output_chain="o", chain_nuclides=len(chain.nuclides))
    from add_pendf_isomeric_branching_to_chain import write_isomer_mapping_log
    write_isomer_mapping_log(log, stats, source_stats, "elis", 0.50, 0.0)
    text = log.read_text()
    assert "MF=10 without MF=3 (not decorable):     1" in text
    section = text.split("MF=10 WITHOUT MF=3 (NOT DECORABLE)", 1)[1]
    assert "In115" in section and "(n,np)" in section and "In114" in section


# ---------------------------------------------------------------------------
# --emit-mf10-only-reactions: symmetric visibility + the three emission shapes
# ---------------------------------------------------------------------------

# In115 as seen by all three source forms: a normal MT=102 (MF=3 + MF=10) plus
# an MF=10-only MT=17 (n,3n) -> In113, the ground-only "Cr (n,3n)" class.
_MF10_ONLY_MT = 17
_MF10_ONLY_Q = -1.2e7
# Q values are chosen to round-trip EXACTLY through the 11-character ENDF-6
# field of the synthetic tape (5 significant digits), so the tape-sourced and
# h5-sourced chains can be compared byte for byte.
_NG_PARTIALS = [(6784700.0, 6784700.0, 49116, 0),
                (6784700.0, 6657400.0, 49116, 1),
                (6784700.0, 6495100.0, 49116, 4)]
_N3N_PARTIALS = [(_MF10_ONLY_Q, _MF10_ONLY_Q, 49113, 0)]
# MF=10-only (n,2n) -> In114 (LFS 0) + In114_m1 (LFS 1, ELFS 190 keV).
_N2N_PARTIALS = [(-9.0e6, -9.0e6, 49114, 0), (-9.0e6, -9.19e6, 49114, 1)]


def _emit_tape_source(tmp_path, monkeypatch, emit=False, with_metastable=False):
    """`_AscSource` over a synthetic In115 tape carrying the MF=10-only MT.

    ``with_metastable`` adds a second MF=10-only MT -- (n,2n) with a ground AND
    a metastable partial -- for the ground+metastable shape.
    """
    import types

    import add_pendf_isomeric_branching_to_chain as patcher
    import openmc.data.endf as endf_mod
    import openmc.data.pendf as pendf_mod

    sections = {
        (3, 102): _mf3_text(49115, 6784700.0, 6784700.0),
        (10, 102): _mf10_text(49115, _NG_PARTIALS),
        (10, _MF10_ONLY_MT): _mf10_text(49115, _N3N_PARTIALS),
    }
    if with_metastable:
        sections[(10, 16)] = _mf10_text(49115, _N2N_PARTIALS)
    ev = types.SimpleNamespace(
        target=dict(atomic_number=49, mass_number=115, isomeric_state=0,
                    excitation_energy=0.0),
        section=sections)
    tape = tmp_path / "n-In115.pendf"
    tape.write_text("")
    monkeypatch.setattr(pendf_mod, "_discover_pendf_files",
                        lambda d: [(tape, None)])
    monkeypatch.setattr(endf_mod, "Evaluation", lambda f: ev)
    monkeypatch.setattr(patcher, "tape_identity", lambda p: "synthetic-tape")
    return patcher._AscSource(tmp_path, emit_mf10_only_reactions=emit)


def _write_emit_h5(path, *, stamped, root_attr):
    """Minimal PendfLibrary-openable h5 with In115 MT=102 (+ optional MT=17).

    ``stamped`` adds the MF=10-only MT=17 group with the builder's
    ``total_source='sum-mf10'`` stamp; ``root_attr`` writes the root
    ``mf10_only_totals`` census (its ABSENCE is what marks a pre-feature file).
    """
    import h5py

    grid = np.array([1.0e-5, 1.0e6])

    def _write(group, qm, qi, subs, total_source=None):
        group.attrs["QM"] = qm
        group.attrs["QI"] = qi
        if total_source is not None:
            group.attrs["total_source"] = np.bytes_(total_source)
        group.create_dataset("energy", data=grid)
        group.create_dataset("xs", data=np.full(2, 4.0))
        for pqm, pqi, izap, lfs in subs:
            sub = group.create_group(f"LFS{lfs}")
            sub.attrs["LFS"] = lfs
            sub.attrs["IZAP"] = izap
            sub.attrs["QM"] = pqm
            sub.attrs["QI"] = pqi
            sub.attrs["ELFS"] = pqm - pqi
            sub.create_dataset("energy", data=grid)
            sub.create_dataset("xs", data=np.full(2, 1.0))

    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = 2
        f.attrs["library"] = np.bytes_("synthetic")
        f.attrs["temperature"] = 293.16
        if root_attr is not None:
            f.attrs["mf10_only_totals"] = int(root_attr)
        nuc = f.create_group("In115")
        nuc.attrs["ELIS"] = 0.0
        _write(nuc.create_group("MT102"), 6784700.0, 6784700.0, _NG_PARTIALS)
        if stamped:
            _write(nuc.create_group(f"MT{_MF10_ONLY_MT}"), _MF10_ONLY_Q,
                   _MF10_ONLY_Q, _N3N_PARTIALS, total_source="sum-mf10")
    return path


def _emit_chain():
    """Chain carrying In115 (n,gamma) plus every product these tests need."""
    return _chain_with(["In115", "In113", "In114", "In114_m1", "In116",
                        "In116_m1", "In116_m2"],
                       reactions={"In115": [("(n,gamma)", "In116", 6784700.0)]})


def _emit_run(source, tmp_path, name, decay=None):
    """map_library + decorate_chain + export; returns (xml bytes, stats)."""
    chain = _emit_chain()
    branching, stats = map_library(source, chain, decay or _decay_lookup(),
                                   "elis", 0.50, 0.0)
    stats["reactions_added"] = decorate_chain(chain, branching, stats)
    out = tmp_path / f"{name}.xml"
    chain.export_to_xml(out)
    return out.read_bytes(), stats, chain


def test_emit_flag_off_identical_on_three_source_forms(tmp_path, monkeypatch):
    # Symmetric visibility contract: with --emit-mf10-only-reactions OFF the
    # MF=10-without-MF=3 class is invisible in EVERY source form, so an old h5
    # (no such group), a NEW h5 (stamped group filtered out and recorded) and an
    # ASC tape (section excluded and recorded) all yield the SAME chain, byte for
    # byte. Rebuilding an h5 with the feature therefore cannot silently change a
    # flag-off chain.
    import add_pendf_isomeric_branching_to_chain as patcher

    old_h5 = _write_emit_h5(tmp_path / "old.h5", stamped=False, root_attr=None)
    new_h5 = _write_emit_h5(tmp_path / "new.h5", stamped=True, root_attr=1)

    old_src = patcher._H5Source(old_h5)
    xml_old, stats_old, _ = _emit_run(old_src, tmp_path, "old")
    new_src = patcher._H5Source(new_h5)
    xml_new, stats_new, chain_new = _emit_run(new_src, tmp_path, "new")
    tape_src = _emit_tape_source(tmp_path, monkeypatch)
    xml_tape, stats_tape, _ = _emit_run(tape_src, tmp_path, "tape")

    assert xml_new == xml_old
    assert xml_tape == xml_old
    assert b"(n,3n)" not in xml_old                  # the class never decorates
    assert not any(rx.type.startswith("(n,3n)")
                   for rx in chain_new["In115"].reactions)

    # The new h5 filters the stamped group and RECORDS it, exactly as the tape
    # adapter records the excluded section; the old h5 carries none.
    assert new_src.serve_mf10_only is False
    assert sorted(new_src.reactions("In115")) == [102]
    assert stats_old["mf10_without_mf3"] == 0
    assert stats_new["mf10_without_mf3"] == 1
    assert stats_tape["mf10_without_mf3"] == 1
    rec_new = stats_new["mf10_without_mf3_list"][0]
    rec_tape = stats_tape["mf10_without_mf3_list"][0]
    for rec in (rec_new, rec_tape):
        assert (rec["parent"], rec["mt"], rec["reaction"]) == \
            ("In115", _MF10_ONLY_MT, "(n,3n)")
        assert rec["lfs"] == [0] and rec["products"] == ["In113"]
        assert rec["metastable"] is False

    # Nothing was examined for emission in any form, and no counter block or
    # emission section reaches the log.
    for stats in (stats_old, stats_new, stats_tape):
        assert stats["mf10_only_enabled"] is False
        assert stats["mf10_only"]["examined"] == 0
    log = tmp_path / "log.txt"
    source_stats = dict(base_chain="c", pendf="p", decay_file="d",
                        output_chain="o", chain_nuclides=7)
    from add_pendf_isomeric_branching_to_chain import write_isomer_mapping_log
    write_isomer_mapping_log(log, stats_new, source_stats, "elis", 0.50, 0.0)
    text = log.read_text()
    assert "MF=10-ONLY EMISSION (--emit-mf10-only-reactions)" not in text
    assert "MF=10-only emitted" not in text


def _emit_decay_lookup():
    """`_decay_lookup_with_in114` plus Ag111 (ground + m1 at 60 keV)."""
    d = _decay_lookup_with_in114()
    d[(47, 111)] = [
        DecayState(47, 111, 0.0, 0, half_life=None),
        DecayState(47, 111, 60000.0, 1, half_life=64.8),
    ]
    return d


def _emit_shapes_source():
    """Synthetic source serving the three MF=10-only shapes on In115.

    MT=17 (n,3n) is ground-only (In113), MT=16 (n,2n) is ground+metastable
    (In114 / In114_m1) and MT=22 (n,na) is metastable-only (Ag111_m1, whose
    ground Ag111 IS in the chain -- proving the metastable-only fold is a
    deliberate choice, not a failed target lookup).
    """
    data = {"In115": {
        17: dict(qm=_MF10_ONLY_Q, qi=_MF10_ONLY_Q, mf3_less=True, partials=[
            dict(lfs=0, izap=49113, qi=_MF10_ONLY_Q, qm=_MF10_ONLY_Q,
                 elfs=0.0)]),
        16: dict(qm=-9.0e6, qi=-9.0e6, mf3_less=True, partials=[
            dict(lfs=0, izap=49114, qi=-9.0e6, qm=-9.0e6, elfs=0.0),
            dict(lfs=1, izap=49114, qi=-9.19e6, qm=-9.0e6, elfs=190000.0)]),
        22: dict(qm=-5.0e6, qi=-5.0e6, mf3_less=True, partials=[
            dict(lfs=1, izap=47111, qi=-5.06e6, qm=-5.0e6, elfs=60000.0)]),
    }}
    source = _FakeSource(data, elis={"In115": 0.0})
    source.emit_mf10_only = True                     # the flag, source side
    return source


def test_emit_flag_on_three_shapes(tmp_path):
    # Flag ON: the ground-only MT becomes a PLAIN reaction (no branching child,
    # Q = the LFS=0 partial's QM), the ground+metastable MT decorates normally
    # with real section-sourced Q values, and the metastable-only MT folds with
    # metastable members only. Counters reconcile.
    chain = _chain_with(["In115", "In113", "In114", "In114_m1", "Ag111",
                         "Ag111_m1"])
    source = _emit_shapes_source()
    branching, stats = map_library(source, chain, _emit_decay_lookup(),
                                   "elis", 0.50, 0.0)
    stats["reactions_added"] = decorate_chain(chain, branching, stats)

    rxns = {rx.type: rx for rx in chain["In115"].reactions}

    # Shape 2 -- ground-only: plain reaction, no pendf_lfs (never folded).
    n3n = rxns["(n,3n)"]
    assert (n3n.target, n3n.Q, n3n.pendf_lfs) == ("In113", _MF10_ONLY_Q, None)

    # Shape 1 -- ground + metastable: branched, both Q values MF=10-sourced.
    assert (rxns["(n,2n)"].target, rxns["(n,2n)"].Q,
            rxns["(n,2n)"].pendf_lfs) == ("In114", -9.0e6, 0)
    assert (rxns["(n,2n)_m1"].target, rxns["(n,2n)_m1"].Q,
            rxns["(n,2n)_m1"].pendf_lfs) == ("In114_m1", -9.19e6, 1)

    # Shape 3 -- metastable-only: no ground member although Ag111 is in chain.
    assert "(n,na)" not in rxns
    assert (rxns["(n,na)_m1"].target, rxns["(n,na)_m1"].pendf_lfs) == \
        ("Ag111_m1", 1)

    # Serialization: the ground-only MT is stock, the others fold.
    out = tmp_path / "chain.xml"
    chain.export_to_xml(out)
    text = out.read_text()
    assert '<reaction type="(n,3n)" Q="-12000000.0" target="In113"/>' in text
    assert 'targets="In114 In114_m1"' in text
    assert 'targets="Ag111_m1"' in text

    book = stats["mf10_only"]
    assert stats["mf10_only_enabled"] is True
    assert (book["examined"], book["emitted_ground_only"],
            book["emitted_branched"]) == (3, 1, 2)
    from add_pendf_isomeric_branching_to_chain import (
        _mf10_only_reconciliation)
    examined, emitted, skipped = _mf10_only_reconciliation(book)
    assert examined == emitted + skipped == 3 and skipped == 0

    # The log gains the greppable emission section with one row per MT, and the
    # summary block gains the counters.
    log = tmp_path / "log.txt"
    source_stats = dict(base_chain="c", pendf="p", decay_file="d",
                        output_chain="o", chain_nuclides=len(chain.nuclides))
    from add_pendf_isomeric_branching_to_chain import write_isomer_mapping_log
    write_isomer_mapping_log(log, stats, source_stats, "elis", 0.50, 0.0)
    text = log.read_text()
    assert "MF=10-only emitted (ground-only):     1" in text
    assert "MF=10-only examined (= emitted+skips):     3   [3 + 0]" in text
    section = text.split("MF=10-ONLY EMISSION", 1)[1]
    assert "EMITTED plain (ground-only)" in section
    assert "EMITTED branched" in section
    assert "ground-only" in section and "g+m" in section and "m-only" in section
    assert "VACUOUS" in section                      # audit-vacuity statement


def test_emit_flag_on_target_not_in_chain_skipped(tmp_path):
    # A ground-only MF=10-only MT whose (dA, dZ) daughter is absent from the
    # chain emits nothing, is counted as a skip, and still reconciles.
    chain = _chain_with(["In115", "In114", "In114_m1", "Ag111", "Ag111_m1"])
    source = _emit_shapes_source()                   # In113 NOT in this chain
    branching, stats = map_library(source, chain, _emit_decay_lookup(),
                                   "elis", 0.50, 0.0)
    stats["reactions_added"] = decorate_chain(chain, branching, stats)

    assert not any(rx.type.startswith("(n,3n)")
                   for rx in chain["In115"].reactions)
    book = stats["mf10_only"]
    assert book["skipped_no_target"] == 1
    assert (book["emitted_ground_only"], book["emitted_branched"]) == (0, 2)
    from add_pendf_isomeric_branching_to_chain import (
        _mf10_only_reconciliation)
    examined, emitted, skipped = _mf10_only_reconciliation(book)
    assert examined == emitted + skipped == 3


def test_emit_flag_on_pre_upgrade_h5_warns_and_emits_nothing(tmp_path, capsys):
    # Flag ON against an h5 that predates MF=10-only totals (no root attr): warn
    # once and behave exactly as flag-off for the class -- even a stamped group
    # (hand-made here) stays filtered, since the missing root attr is the signal
    # that the file cannot serve the class.
    import add_pendf_isomeric_branching_to_chain as patcher

    path = _write_emit_h5(tmp_path / "pre.h5", stamped=True, root_attr=None)
    source = patcher._H5Source(path, emit_mf10_only_reactions=True)
    err = capsys.readouterr().err
    assert "h5 predates MF=10-only totals" in err
    assert "tools/pendf_to_hdf5.py" in err
    assert source.mf10_only_totals is None
    assert source.serve_mf10_only is False
    assert sorted(source.reactions("In115")) == [102]

    xml, stats, chain = _emit_run(source, tmp_path, "pre")
    assert b"(n,3n)" not in xml
    assert stats["mf10_only"]["examined"] == 0       # nothing even examined
    assert stats["mf10_without_mf3"] == 1            # recorded, like flag-off


def test_emit_flag_on_tape_source_serves_and_audits_vacuously(tmp_path,
                                                              monkeypatch):
    # Flag ON against an ASC tape: an MF=10-only MT is served with Q values from
    # the MF=10 section itself (no MF=3 HEAD exists) and a total synthesized as
    # the union-grid sum of its partials -- exactly what the h5 builder stores.
    # The audit is therefore VACUOUS by construction: the summed partials ARE
    # the total, so every band ratio is 1.0 and worst_dev is 0, and the reaction
    # can never be an offender or be band-rejected.
    source = _emit_tape_source(tmp_path, monkeypatch, emit=True,
                               with_metastable=True)
    rxns = source.reactions("In115")
    assert sorted(rxns) == [16, 17, 102]             # both MF=10-only MTs served
    assert rxns[16]["mf3_less"] is True and rxns[102]["mf3_less"] is False
    # Both are the section QM (ELFS = 0 at the ground state); never fabricated.
    assert (rxns[16]["qm"], rxns[16]["qi"]) == (-9.0e6, -9.0e6)

    energy, xs = source.total_xs("In115", 16)
    part = sum(source.pathway_xs("In115", 16, lfs, 49114)[1]
               for lfs in (0, 1))
    assert np.allclose(xs, part) and xs.size == energy.size

    chain = _emit_chain()
    branching, stats = map_library(source, chain, _decay_lookup_with_in114(),
                                   "elis", 0.50, 0.0, reject_band_ratio=0.3)
    stats["reactions_added"] = decorate_chain(chain, branching, stats)

    audit = _audit_reaction(source, "In115", 16, rxns[16]["partials"])
    assert audit["worst_dev"] == pytest.approx(0.0, abs=1e-12)
    assert audit["integral_ratio"] == pytest.approx(1.0)
    assert all(audit[k] is None or audit[k] == pytest.approx(1.0)
               for k in ("ratio_thermal", "ratio_epithermal",
                         "ratio_intermediate", "ratio_fast"))
    assert stats["rejected_count"] == 0               # never rejectable

    types_ = {rx.type: rx for rx in chain["In115"].reactions}
    assert (types_["(n,2n)"].target, types_["(n,2n)"].pendf_lfs) == ("In114", 0)
    assert types_["(n,2n)_m1"].target == "In114_m1"
    assert (types_["(n,3n)"].target, types_["(n,3n)"].pendf_lfs) == \
        ("In113", None)                               # plain, ground-only
    book = stats["mf10_only"]
    assert (book["examined"], book["emitted_ground_only"],
            book["emitted_branched"]) == (2, 1, 1)
    assert stats["mf10_without_mf3"] == 0             # nothing left undecorable


def test_emit_flag_on_h5_source_serves_and_matches_tape(tmp_path, monkeypatch):
    # The h5 SERVE path (flag ON against a stamped, current-format library): the
    # MF=10-only MT is enumerated alongside the ordinary one, carries its
    # section-sourced Q values, and emits the same plain reaction the tape
    # adapter emits -- the two source forms produce byte-identical chains.
    import add_pendf_isomeric_branching_to_chain as patcher

    h5 = _write_emit_h5(tmp_path / "serve.h5", stamped=True, root_attr=1)
    source = patcher._H5Source(h5, emit_mf10_only_reactions=True)
    assert source.serve_mf10_only is True
    assert source.mf10_only_totals == 1

    rxns = source.reactions("In115")
    assert sorted(rxns) == [_MF10_ONLY_MT, 102]      # served, not filtered
    assert rxns[_MF10_ONLY_MT]["mf3_less"] is True
    assert rxns[102]["mf3_less"] is False            # the ordinary MT is normal
    # Q values come from the MF=10 section itself (there is no MF=3 HEAD).
    assert (rxns[_MF10_ONLY_MT]["qm"],
            rxns[_MF10_ONLY_MT]["qi"]) == (_MF10_ONLY_Q, _MF10_ONLY_Q)
    ground = rxns[_MF10_ONLY_MT]["partials"][0]
    assert (ground["lfs"], ground["izap"], ground["qi"]) == \
        (0, 49113, _MF10_ONLY_Q)
    # The stored (builder-synthesized) total is served like any other total.
    energy, xs = source.total_xs("In115", _MF10_ONLY_MT)
    assert xs.size == energy.size == 2
    assert source.pathway_xs("In115", _MF10_ONLY_MT, 0, 49113)[1].size == 2

    xml_h5, stats, chain = _emit_run(source, tmp_path, "serve")
    types_ = {rx.type: rx for rx in chain["In115"].reactions}
    assert (types_["(n,3n)"].target, types_["(n,3n)"].Q,
            types_["(n,3n)"].pendf_lfs) == ("In113", _MF10_ONLY_Q, None)
    book = stats["mf10_only"]
    assert (book["examined"], book["emitted_ground_only"],
            book["emitted_branched"]) == (1, 1, 0)
    assert stats["mf10_without_mf3"] == 0            # nothing left undecorable

    # Parity claim: same content through the ASC-tape adapter -> same chain.
    tape_src = _emit_tape_source(tmp_path, monkeypatch, emit=True)
    xml_tape, stats_tape, _ = _emit_run(tape_src, tmp_path, "serve_tape")
    assert xml_h5 == xml_tape
    assert stats_tape["mf10_only"]["emitted_ground_only"] == 1


def test_emit_name_colliding_mts_reconcile(tmp_path):
    # Two MF=10-only MTs of one parent share a chain reaction name (MT=16 and
    # MT=875 are both '(n,2n)'). Last MT wins the branching entry -- unchanged,
    # pre-existing behavior -- and the DISPLACED entry is terminated in the
    # 'superseded' bucket so the reconciliation identity still holds and no row
    # is left '(pending)'.
    chain = _chain_with(["In115", "In114", "In114_m1"])

    def _n2n(gq):
        """Ground + m1 partials (In114 / In114_m1) with ground Q ``gq``."""
        return [dict(lfs=0, izap=49114, qi=gq, qm=gq, elfs=0.0),
                dict(lfs=1, izap=49114, qi=gq - 1.9e5, qm=gq, elfs=190000.0)]

    source = _FakeSource({"In115": {
        16: dict(qm=-9.0e6, qi=-9.0e6, mf3_less=True, partials=_n2n(-9.0e6)),
        875: dict(qm=-9.1e6, qi=-9.1e6, mf3_less=True, partials=_n2n(-9.1e6)),
    }}, elis={"In115": 0.0})
    source.emit_mf10_only = True

    branching, stats = map_library(source, chain, _decay_lookup_with_in114(),
                                   "elis", 0.50, 0.0)
    stats["reactions_added"] = decorate_chain(chain, branching, stats)

    # Data behavior: exactly one '(n,2n)' fold, taken from the LAST MT (875).
    rxns = {rx.type: rx for rx in chain["In115"].reactions}
    assert sorted(rxns) == ["(n,2n)", "(n,2n)_m1"]
    assert (rxns["(n,2n)"].target, rxns["(n,2n)"].Q,
            rxns["(n,2n)"].pendf_lfs) == ("In114", -9.1e6, 0)
    assert (rxns["(n,2n)_m1"].target, rxns["(n,2n)_m1"].Q,
            rxns["(n,2n)_m1"].pendf_lfs) == ("In114_m1", -9.29e6, 1)

    # Bookkeeping: 2 examined = 1 emitted + 1 skip (the displaced MT=16).
    book = stats["mf10_only"]
    from add_pendf_isomeric_branching_to_chain import (
        _mf10_only_reconciliation)
    examined, emitted, skipped = _mf10_only_reconciliation(book)
    assert (examined, emitted, skipped) == (2, 1, 1)
    assert examined == emitted + skipped
    assert book["emitted_branched"] == 1
    assert book["skipped_superseded"] == 1
    outcomes = {r["mt"]: r["outcome"] for r in book["rows"]}
    assert "(pending)" not in outcomes.values()
    assert "superseded" in outcomes[16] and "MT=875" in outcomes[16]
    assert outcomes[875].startswith("EMITTED branched")

    # ... and the log reconciles, with no MISMATCH marker anywhere.
    log = tmp_path / "log.txt"
    source_stats = dict(base_chain="c", pendf="p", decay_file="d",
                        output_chain="o", chain_nuclides=len(chain.nuclides))
    from add_pendf_isomeric_branching_to_chain import write_isomer_mapping_log
    write_isomer_mapping_log(log, stats, source_stats, "elis", 0.50, 0.0)
    text = log.read_text()
    assert "MISMATCH" not in text
    assert "= 2 = 1 + 1" in text
    assert "superseded by another MT (MT=875)" in text


# ---------------------------------------------------------------------------
# Ground-state invariant on the LFS=0 partial (ELFS = QM - QI = 0)
# ---------------------------------------------------------------------------

# Real MF=10 MT=4 subsection heads, read 2026-08-04 from the production
# libraries in data/byPerry_v1: TENDL2019-IST_decay2020.293K.PENDF.h5 and
# JEFF40-IST.293K.PENDF.h5. TENDL-2019 breaks the ground-state invariant in two
# shapes -- the reaction QI written into an LFS=0 head on a ground target
# (Ag107), a blank QI on an isomer target (Ag107_m1, and U235_m1 at the fleet's
# smallest genuine offset, 76.77 eV) -- while U235 and the whole JEFF-4.0 MT=4
# fleet (Ir191, 146 LFS=0 partials, every one storing ELFS exactly 0.0) are
# clean. A clean head repeats ONE number in both fields, which is why the guard
# tolerance is 1 eV rather than the GENDF twin's 1 keV C4 value.
_MT4_HEADS = {
    "Ag107":    [dict(lfs=0, izap=47107, qm=0.0, qi=-93125.0, elfs=93125.0),
                 dict(lfs=1, izap=47107, qm=0.0, qi=-93125.0, elfs=93125.0)],
    "Ag107_m1": [dict(lfs=0, izap=47107, qm=93124.9, qi=0.0, elfs=93124.9),
                 dict(lfs=1, izap=47107, qm=93124.9, qi=-0.13411,
                      elfs=93125.03411)],
    "U235_m1":  [dict(lfs=0, izap=92235, qm=76.7708, qi=0.0, elfs=76.7708),
                 dict(lfs=1, izap=92235, qm=76.7708, qi=-0.229214,
                      elfs=77.000014)],
    "U235":     [dict(lfs=0, izap=92235, qm=0.0, qi=0.0, elfs=0.0),
                 dict(lfs=1, izap=92235, qm=0.0, qi=-77.0, elfs=77.0)],
    "Ir191":    [dict(lfs=0, izap=77191, qm=0.0, qi=0.0, elfs=0.0),
                 dict(lfs=3, izap=77191, qm=0.0, qi=-171290.0, elfs=171290.0),
                 dict(lfs=72, izap=77191, qm=0.0, qi=-2201000.0,
                      elfs=2201000.0)],
}
_MT4_DEFECTIVE = ("Ag107", "Ag107_m1", "U235_m1")
_QI_COMPLAINT = "LFS=0 partial QI="

# The defect shape used wherever a test needs a reaction that actually REACHES
# the ground-route code: on real data all 1069 offenders sit on MT=4, which
# decorate_chain routes through its (n,n') branches instead, so the Ag107_m1
# head is transplanted onto another MT. Its QM and its tabulated QI are 93 keV
# apart, so the two candidate Q values can never be confused.
_DEFECTIVE_GROUND_QM = 93124.9


def _defective_ground(izap):
    """An LFS=0 partial carrying the real Ag107_m1 MT=4 head on ``izap``."""
    return dict(lfs=0, izap=izap, qm=_DEFECTIVE_GROUND_QM, qi=0.0,
                elfs=_DEFECTIVE_GROUND_QM)


@pytest.mark.parametrize("name", list(_MT4_HEADS))
def test_mf10_only_qm_qi_ground_state_invariant(name, capsys):
    # Both Q values of an MF=3-less reaction are the section QM: a ground-state
    # subsection has ELFS = 0, so QM is the ground route's Q. On the TENDL-2019
    # defectives the tabulated QI is warned about and dropped; on the clean
    # tapes nothing is said because the two fields hold one number.
    from add_pendf_isomeric_branching_to_chain import _mf10_only_qm_qi

    partials = _MT4_HEADS[name]
    qm, qi = _mf10_only_qm_qi(partials, f"{name} MT=4")
    err = capsys.readouterr().err

    assert (qm, qi) == (partials[0]["qm"], partials[0]["qm"])
    assert (_QI_COMPLAINT in err) is (name in _MT4_DEFECTIVE)
    if name in _MT4_DEFECTIVE:
        assert f"{name} MT=4" in err
        assert f"QI={partials[0]['qi']}" in err and f"QM={qm}" in err
        assert "ground-route Q" in err and "ELFS = 0" in err


def test_mf10_only_qm_qi_tolerance_floor(capsys):
    # U235_m1 is 76.77 eV out -- the smallest genuine defect in the fleet, and
    # the reason the gate is 1 eV: a regression to the GENDF C4 1 keV value
    # would pass this section silently.
    from add_pendf_isomeric_branching_to_chain import (_mf10_only_qm_qi,
                                                       _LFS0_QI_QM_TOL_EV)

    assert abs(76.7708 - 0.0) > _LFS0_QI_QM_TOL_EV
    qm, qi = _mf10_only_qm_qi(_MT4_HEADS["U235_m1"], "U235_m1 MT=4")
    assert (qm, qi) == (76.7708, 76.7708)
    assert _QI_COMPLAINT in capsys.readouterr().err


def _ground_only_emit_source(ground):
    """Emit-lane source serving one ground-only MF=10-only MT: In115 (n,3n)."""
    source = _FakeSource({"In115": {
        _MF10_ONLY_MT: dict(qm=ground["qm"], qi=ground["qm"], mf3_less=True,
                            partials=[ground])}})
    source.emit_mf10_only = True
    return source


@pytest.mark.parametrize("ground, expect_warning", [
    (_defective_ground(49113), True),
    (dict(lfs=0, izap=49113, qm=_MF10_ONLY_Q, qi=_MF10_ONLY_Q, elfs=0.0),
     False),
])
def test_emit_mf10_only_ground_writes_section_qm(ground, expect_warning,
                                                 capsys):
    # The plain ground reaction the emit lane appends carries the LFS=0
    # partial's QM. Equally section-sourced as the QI it replaces, and the only
    # one of the two a ground state leaves defined.
    chain = _chain_with(["In115", "In113"])
    source = _ground_only_emit_source(ground)
    branching, stats = map_library(source, chain, _decay_lookup(),
                                   "elis", 0.50, 0.0)
    capsys.readouterr()                              # discard mapping chatter
    stats["reactions_added"] = decorate_chain(chain, branching, stats)
    err = capsys.readouterr().err

    n3n = {rx.type: rx for rx in chain["In115"].reactions}["(n,3n)"]
    assert (n3n.target, n3n.Q, n3n.pendf_lfs) == ("In113", ground["qm"], None)
    assert stats["mf10_only"]["emitted_ground_only"] == 1
    assert (_QI_COMPLAINT in err) is expect_warning
    if expect_warning:
        assert "In115 (n,3n) MT=17" in err


def _synthesized_ground_scene(ground_qi):
    """In115 (n,gamma) ABSENT from the base chain, with one mapped metastable.

    The ground head is the real Ag107_m1 MT=4 defect (QM=93124.9, QI blank); the
    m1 partial keeps In115's own 127270 eV ELFS -- the quantity that decides the
    ELIS match -- re-based onto that QM, so the level still maps to In116_m1.
    """
    qm = _DEFECTIVE_GROUND_QM
    return _FakeSource({"In115": {
        102: dict(qm=qm, qi=qm, partials=[
            dict(lfs=0, izap=49116, qm=qm, qi=ground_qi, elfs=qm - ground_qi),
            dict(lfs=1, izap=49116, qm=qm, qi=qm - 127270.0,
                 elfs=127270.0)])}})


def test_decorate_synthesized_ground_uses_section_qm(capsys):
    # decorate_chain's synthesized-ground branch: a reaction the base chain does
    # not carry gets its ground member's Q from the LFS=0 partial's QM, so the
    # ground route and the metastable levels measured off it share one energy
    # zero. The mapped metastable's own QI is untouched.
    chain = _chain_with(["In115", "In116", "In116_m1", "In116_m2"])
    source = _synthesized_ground_scene(0.0)
    branching, stats = map_library(source, chain, _decay_lookup(),
                                   "elis", 0.50, 0.0)
    capsys.readouterr()
    added = decorate_chain(chain, branching, stats)
    err = capsys.readouterr().err

    rxns = {rx.type: rx for rx in chain["In115"].reactions}
    assert added == 1
    assert (rxns["(n,gamma)"].target, rxns["(n,gamma)"].Q,
            rxns["(n,gamma)"].pendf_lfs) == ("In116", _DEFECTIVE_GROUND_QM, 0)
    assert (rxns["(n,gamma)_m1"].target, rxns["(n,gamma)_m1"].Q) == \
        ("In116_m1", _DEFECTIVE_GROUND_QM - 127270.0)
    assert _QI_COMPLAINT in err
    assert "In115 (n,gamma) MT=102" in err


def test_decorate_synthesized_ground_clean_section_silent(capsys):
    # Same scene with a sound LFS=0 head (QI == QM): identical ground Q, and
    # not a word on stderr -- the guard costs a clean library nothing.
    chain = _chain_with(["In115", "In116", "In116_m1", "In116_m2"])
    source = _synthesized_ground_scene(_DEFECTIVE_GROUND_QM)
    branching, stats = map_library(source, chain, _decay_lookup(),
                                   "elis", 0.50, 0.0)
    capsys.readouterr()
    decorate_chain(chain, branching, stats)

    rxns = {rx.type: rx for rx in chain["In115"].reactions}
    assert rxns["(n,gamma)"].Q == _DEFECTIVE_GROUND_QM
    assert _QI_COMPLAINT not in capsys.readouterr().err


def test_clean_library_scenes_stay_silent(capsys):
    # Regression sweep over the pre-existing clean scenes: the full fold
    # round-trip, the three MF=10-only shapes and the orphan triad's Au186
    # (n,2n) must not gain a single ground-state complaint.
    chain = _chain_with(["In115", "In116", "In116_m1", "In116_m2"],
                        reactions={"In115": [("(n,gamma)", "In116",
                                              6784720.0)]})
    source = _FakeSource({"In115": {
        102: dict(qm=6784720.0, qi=6784720.0, partials=(
            [dict(lfs=0, izap=49116, qi=6784720.0, qm=6784720.0, elfs=0.0)]
            + _in115_ng_metastables()))}})
    branching, stats = map_library(source, chain, _decay_lookup(),
                                   "elis", 0.50, 0.0)
    decorate_chain(chain, branching, stats)

    shapes = _chain_with(["In115", "In113", "In114", "In114_m1", "Ag111",
                          "Ag111_m1"])
    branching, stats = map_library(_emit_shapes_source(), shapes,
                                   _emit_decay_lookup(), "elis", 0.50, 0.0)
    decorate_chain(shapes, branching, stats)

    _orphan_run("add-stable")
    assert _QI_COMPLAINT not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Pathway-Q chain-consistency gate (GENDF parity)
# ---------------------------------------------------------------------------

# JEFF-4.0 Am241(n,gamma), the reference case: the file's ground QM sits 3350 eV
# above the chain's scalar Q, and the chain is the accurate one (9.8 eV from AME
# Sn(Am242) = 5537640.17, vs the file's 3359.8 -- 344x closer).
_AM241_CHAIN_Q = 5537650.0
_AM242M_ELFS = 48063.0          # QM - QI of the mapped LFS=2 level


def _am241_decay():
    return {(95, 242): [
        DecayState(95, 242, 0.0, 0, half_life=None),
        DecayState(95, 242, _AM242M_ELFS, 1, half_life=4907.0),
    ]}


def _am241_scene(file_qm):
    """Chain + source for Am241 (n,gamma) with a chosen file ground QM."""
    chain = _chain_with(
        ["Am241", "Am242", "Am242_m1"],
        reactions={"Am241": [("(n,gamma)", "Am242", _AM241_CHAIN_Q)]})
    source = _FakeSource({"Am241": {
        102: dict(qm=file_qm, qi=file_qm, partials=[
            dict(lfs=0, izap=95242, qi=file_qm, qm=file_qm, elfs=0.0),
            dict(lfs=2, izap=95242, qi=file_qm - _AM242M_ELFS, qm=file_qm,
                 elfs=_AM242M_ELFS)]),
    }})
    return chain, source


@pytest.mark.parametrize("qm_offset, rejected", [(3350.0, True), (0.3, False)])
def test_pathway_q_file_qm_gated_against_chain_scalar(qm_offset, rejected):
    # Outside MT=4 a file QM is adopted only if it corroborates the chain Q.
    # Both offsets are measured magnitudes: 3350 eV is Am241 (n,gamma)'s real
    # divergence (refused), 0.3 eV the benign post-rounding jitter (adopted).
    file_qm = _AM241_CHAIN_Q + qm_offset
    chain, source = _am241_scene(file_qm)
    branching, stats = map_library(source, chain, _am241_decay(),
                                   "elis", 0.50, 0.0)
    decorate_chain(chain, branching, stats)
    rxns = {rx.type: rx for rx in chain["Am241"].reactions}

    # The ground slot is the chain scalar either way -- it never moves.
    assert rxns["(n,gamma)"].Q == _AM241_CHAIN_Q
    assert rxns["(n,gamma)"].pendf_lfs == 0

    if rejected:
        # Metastable reverts to the chain-anchored difference, so both slots
        # share ONE energy zero.
        assert rxns["(n,gamma)_m1"].Q == _AM241_CHAIN_Q - _AM242M_ELFS
        [rec] = stats["pathway_q_rejected"]
        assert (rec["parent"], rec["reaction"], rec["mt"]) == (
            "Am241", "(n,gamma)", 102)
        assert rec["targets"] == ["Am242", "Am242_m1"]
        assert rec["lfs"] == [0, 2]
        assert rec["file_ground_qm"] == file_qm
        assert rec["chain_q"] == _AM241_CHAIN_Q
        assert rec["delta"] == pytest.approx(3350.0)
        # Ledgered, written nowhere.
        assert rec["q_file"] == ["n/a (ground already chain-anchored)",
                                 file_qm - _AM242M_ELFS]
        assert rec["q_kept"] == [_AM241_CHAIN_Q,
                                 _AM241_CHAIN_Q - _AM242M_ELFS]
        # Only the metastable slot moved.
        assert rec["reverted_slots"] == 1
        assert stats["q_chain_anchored"] == 1
    else:
        assert rxns["(n,gamma)_m1"].Q == file_qm - _AM242M_ELFS
        assert stats["pathway_q_rejected"] == []
        assert stats["q_chain_anchored"] == 0


def test_pathway_q_mt4_exempt():
    # MT=4 is exempt, load-bearingly: the chain scalar is QI(MF=3) = -E(level 1)
    # while QM is 0.0, a 336 keV gap that at any other MT would refuse the file.
    # That gap is the quantity mismatch, not evidence against the file.
    chain = _chain_with(["In115", "In115_m1"],
                        reactions={"In115": [("(n,n')", "In115", -336240.0)]})
    source = _FakeSource({"In115": {
        4: dict(qm=0.0, qi=-336240.0, partials=[
            dict(lfs=0, izap=49115, qi=0.0, qm=0.0, elfs=0.0),
            dict(lfs=1, izap=49115, qi=-336244.0, qm=0.0, elfs=336244.0)]),
    }})
    branching, stats = map_library(source, chain, _decay_lookup(),
                                   "elis", 0.50, 0.0)
    decorate_chain(chain, branching, stats)
    rxns = {rx.type: rx for rx in chain["In115"].reactions}
    assert rxns["(n,n')_m1"].Q == -336244.0          # file QI kept
    assert stats["pathway_q_rejected"] == []
    assert stats["q_chain_anchored"] == 0


def test_pathway_q_no_chain_reaction_exempt():
    # A reaction absent from the base chain has no anchor to check against: its
    # ground Q is the LFS=0 partial's QI and the metastables keep their file QI,
    # however far the file sits from anything.
    chain = _chain_with(["In115", "In116", "In116_m1"])
    source = _FakeSource({"In115": {
        102: dict(qm=1.0e7, qi=1.0e7, partials=[
            dict(lfs=0, izap=49116, qi=1.0e7, qm=1.0e7, elfs=0.0),
            dict(lfs=1, izap=49116, qi=1.0e7 - 127270.0, qm=1.0e7,
                 elfs=127270.0)]),
    }})
    branching, stats = map_library(source, chain, _decay_lookup(),
                                   "elis", 0.50, 0.0)
    decorate_chain(chain, branching, stats)
    rxns = {rx.type: rx for rx in chain["In115"].reactions}
    assert rxns["(n,gamma)"].Q == 1.0e7               # synthesized from LFS=0
    assert rxns["(n,gamma)_m1"].Q == 1.0e7 - 127270.0
    assert stats["pathway_q_rejected"] == []
    assert stats["q_chain_anchored"] == 0


@pytest.mark.parametrize("qm_shift, rejected", [(0.0, False), (5000.0, True)])
def test_pathway_q_metastable_only_uses_sibling_qm(qm_shift, rejected):
    # No LFS=0 partial: the probe is a mapped metastable's own QM, reconstructed
    # as QI + ELFS (the classified records carry no 'qm' key).
    chain_q = 6784720.0
    elfs = 127270.0
    qi = chain_q + qm_shift - elfs
    chain = _chain_with(["In115", "In116", "In116_m1"],
                        reactions={"In115": [("(n,gamma)", "In116", chain_q)]})
    source = _FakeSource({"In115": {
        102: dict(qm=chain_q + qm_shift, qi=chain_q, partials=[
            dict(lfs=1, izap=49116, qi=qi, qm=chain_q + qm_shift, elfs=elfs)]),
    }})
    branching, stats = map_library(source, chain, _decay_lookup(),
                                   "elis", 0.50, 0.0)
    decorate_chain(chain, branching, stats)
    rxns = {rx.type: rx for rx in chain["In115"].reactions}
    if rejected:
        [rec] = stats["pathway_q_rejected"]
        assert rec["file_ground_qm"] == pytest.approx(chain_q + qm_shift)
        assert rec["delta"] == pytest.approx(5000.0)
        assert rxns["(n,gamma)_m1"].Q == chain_q - elfs
    else:
        assert stats["pathway_q_rejected"] == []
        assert rxns["(n,gamma)_m1"].Q == qi


def test_pathway_q_reject_logged_and_noted(tmp_path):
    # The refusal is invisible in the written Q values, so the log carries the
    # only trace: a dedicated section plus a per-row note on the mapping table.
    chain, source = _am241_scene(_AM241_CHAIN_Q + 3350.0)
    branching, stats = map_library(source, chain, _am241_decay(),
                                   "elis", 0.50, 0.0)
    stats["reactions_added"] = decorate_chain(chain, branching, stats)

    log = tmp_path / "log.txt"
    source_stats = dict(base_chain="c", pendf="p", decay_file="d",
                        output_chain="o", chain_nuclides=len(chain.nuclides))
    from add_pendf_isomeric_branching_to_chain import write_isomer_mapping_log
    write_isomer_mapping_log(log, stats, source_stats, "elis", 0.50, 0.0)
    text = log.read_text()

    title = "PATHWAY-Q FILE-QM REJECTED (CHAIN-ANCHORED VALUES RETAINED)"
    before, section = text.split(title, 1)
    # Summary counters.
    assert "Pathway-Q file QM rejected (reactions):     1" in text
    assert "Pathway-Q metastable slots chain-anchored:     1" in text
    # Section row: parent, the trigger QM, the anchor, the signed delta.
    assert "Total rejected: 1 reaction(s), 1 metastable slot(s)" in section
    assert "Am241" in section and "5541000.0000" in section
    assert "5537650.0000" in section and "+3350.0000" in section
    assert "n/a (ground already chain-anchored)" in section
    # Per-row marker on the mapping table, BEFORE the section.
    assert "PATHWAY-Q FILE-QM REJECTED" in before


def _n2n_emit_source(file_qm):
    """MF=10-only MT=875 ('(n,2n)') on In115, with a chosen section QM."""
    elfs = 190000.0
    data = {"In115": {
        875: dict(qm=file_qm, qi=file_qm, mf3_less=True, partials=[
            dict(lfs=0, izap=49114, qi=file_qm, qm=file_qm, elfs=0.0),
            dict(lfs=1, izap=49114, qi=file_qm - elfs, qm=file_qm,
                 elfs=elfs)]),
    }}
    source = _FakeSource(data, elis={"In115": 0.0})
    source.emit_mf10_only = True                     # the flag, source side
    return source


def test_pathway_q_emit_lane_bypassed_without_chain_anchor():
    # An MF=10-only MT has no MF=3 total, so a base chain built from MF=3 cannot
    # carry it: nothing anchors the gate and the file Q values stand.
    chain = _chain_with(["In115", "In114", "In114_m1"])
    source = _n2n_emit_source(-9.0e6)
    branching, stats = map_library(source, chain, _decay_lookup_with_in114(),
                                   "elis", 0.50, 0.0)
    decorate_chain(chain, branching, stats)
    rxns = {rx.type: rx for rx in chain["In115"].reactions}
    assert rxns["(n,2n)_m1"].Q == -9.19e6
    assert stats["pathway_q_rejected"] == []


def test_pathway_q_emit_lane_gated_on_name_collision():
    # ... unless a name-colliding MT of the family (MT=16 here) already put the
    # reaction on the chain: that scalar Q IS an anchor, and the gate applies.
    chain_q = -9.0e6
    chain = _chain_with(["In115", "In114", "In114_m1"],
                        reactions={"In115": [("(n,2n)", "In114", chain_q)]})
    source = _n2n_emit_source(chain_q + 7.9e6)
    branching, stats = map_library(source, chain, _decay_lookup_with_in114(),
                                   "elis", 0.50, 0.0)
    decorate_chain(chain, branching, stats)
    rxns = {rx.type: rx for rx in chain["In115"].reactions}
    [rec] = stats["pathway_q_rejected"]
    assert rec["mt"] == 875                          # the surviving level MT
    assert rec["delta"] == pytest.approx(7.9e6)
    assert rxns["(n,2n)"].Q == chain_q
    assert rxns["(n,2n)_m1"].Q == chain_q - 190000.0


def test_pathway_q_clean_section_serializes_unchanged(tmp_path):
    # A corroborated section is untouched: the exported Q list is the file's own
    # values verbatim, with no rounding applied anywhere on the normal path.
    chain = _chain_with(["In115", "In116", "In116_m1", "In116_m2"],
                        reactions={"In115": [("(n,gamma)", "In116",
                                              6784720.0)]})
    source = _FakeSource({"In115": {
        102: dict(qm=6784720.0, qi=6784720.0, partials=(
            [dict(lfs=0, izap=49116, qi=6784720.0, qm=6784720.0, elfs=0.0)]
            + _in115_ng_metastables())),
    }})
    branching, stats = map_library(source, chain, _decay_lookup(),
                                   "elis", 0.50, 0.0)
    decorate_chain(chain, branching, stats)
    assert stats["pathway_q_rejected"] == []

    out = tmp_path / "chain.xml"
    chain.export_to_xml(out)
    assert 'Q="6784720.0 6657450.0 6495060.0"' in out.read_text()


def test_pathway_q_revert_value_rounded(tmp_path):
    # ENDF/B-8.1 (n,alpha) shape: the file QM is 7.9 MeV out, so the revert is a
    # float subtraction that leaves dust (-1758720.2999999998). Reverted values
    # are rounded to 4 dp -- ample for eV-scale level energies -- so no artefact
    # reaches the chain XML.
    chain_q = -1702000.0
    file_qm = 6198000.0                              # wrong by +7.9 MeV
    elfs = file_qm - 6141279.7                       # 56720.299999999814
    chain = _chain_with(["Ir191", "Re188", "Re188_m1"],
                        reactions={"Ir191": [("(n,a)", "Re188", chain_q)]})
    source = _FakeSource({"Ir191": {
        107: dict(qm=file_qm, qi=file_qm, partials=[
            dict(lfs=0, izap=75188, qi=file_qm, qm=file_qm, elfs=0.0),
            dict(lfs=1, izap=75188, qi=6141279.7, qm=file_qm, elfs=elfs)]),
    }})
    decay = {(75, 188): [
        DecayState(75, 188, 0.0, 0, half_life=None),
        DecayState(75, 188, elfs, 1, half_life=1140.0),
    ]}
    branching, stats = map_library(source, chain, decay, "elis", 0.50, 0.0)
    decorate_chain(chain, branching, stats)

    assert len(stats["pathway_q_rejected"]) == 1
    q = {rx.type: rx.Q for rx in chain["Ir191"].reactions}["(n,a)_m1"]
    assert q == pytest.approx(chain_q - elfs)
    assert q == round(q, 4) == -1758720.3            # dust removed

    out = tmp_path / "chain.xml"
    chain.export_to_xml(out)
    assert "-1758720.3" in out.read_text()
    assert "-1758720.2999999998" not in out.read_text()


# ---------------------------------------------------------------------------
# lookup_liso ambiguity contract
#
# The hybrid mapper abstains on an ambiguous ELIS match, so the verdict has to
# travel in the result dict rather than only in the once-per-(Z, A) warning --
# the mapper silences that warning and still needs the answer.
# ---------------------------------------------------------------------------

# Two decay levels at the SAME energy: no ELFS can tell them apart.
_DEGENERATE_DECAY = {(49, 122): [
    DecayState(49, 122, 0.0, 0, half_life=1.5),
    DecayState(49, 122, 200000.0, 1, half_life=10.8),
    DecayState(49, 122, 200000.0, 2, half_life=10.8),
]}


def test_lookup_liso_reports_ambiguity():
    """The ambiguous verdict is computed unconditionally; only the warning is
    gated by ``warn_ambiguity``. A sole candidate reports no second level."""
    decay_elis._WARNED_ELIS_AMBIGUITY.clear()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        quiet = lookup_liso(49, 122, 210000.0, _DEGENERATE_DECAY, rtol=0.15,
                            warn_ambiguity=False)
    assert caught == []
    assert quiet["status"] == "matched" and quiet["liso"] == 1
    assert quiet["ambiguous"] is True
    assert (quiet["second_liso"], quiet["second_dk_elis"]) == (2, 200000.0)

    # The default caller still warns, with the same verdict attached.
    with pytest.warns(UserWarning, match="Ambiguous ELIS match"):
        loud = lookup_liso(49, 122, 210000.0, _DEGENERATE_DECAY, rtol=0.15)
    assert loud["ambiguous"] is True

    # In115 carries a single metastable -> unambiguous, no second level.
    single = lookup_liso(49, 115, 336000.0, _decay_lookup())
    assert single["ambiguous"] is False
    assert (single["second_liso"], single["second_dk_elis"]) == (None, None)


def test_elis_ambiguity_warning_rearmed_per_parse(tmp_path):
    """The warning dedupe is per (Z, A) WITHIN one library load; loading a
    decay library again re-arms it, so a second load in the same process is not
    silently deduped against the first."""
    decay_elis._WARNED_ELIS_AMBIGUITY.clear()
    with pytest.warns(UserWarning, match="Ambiguous ELIS match"):
        lookup_liso(49, 122, 210000.0, _DEGENERATE_DECAY, rtol=0.15)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lookup_liso(49, 122, 205000.0, _DEGENERATE_DECAY, rtol=0.15)
    assert caught == []                              # same (Z, A), still muted

    empty = tmp_path / "decay"
    empty.mkdir()
    assert parse_decay_isomeric_levels(empty) == {}   # a load, however empty
    with pytest.warns(UserWarning, match="Ambiguous ELIS match"):
        lookup_liso(49, 122, 210000.0, _DEGENERATE_DECAY, rtol=0.15)


# ---------------------------------------------------------------------------
# Hybrid mapping mode (elis_lfs_order)
#
# Shapes taken from the July TENDL2019-IST mapping log: In120 (n,n') mixes an
# exact ELIS match with a 471% miss, In127 (n,n') has the higher level matching
# first so the positional pass must skip the LISO it claimed.
# ---------------------------------------------------------------------------

# In120: LFS=1 sits exactly on m1 (70 keV); LFS=2 (400 keV) is 471% off it. The
# decay m2 (In120n) carries a defective 2 keV ELIS with m1's half-life -- zero-
# ish energy, still claimable by the positional pass.
_IN120_DECAY = {(49, 120): [
    DecayState(49, 120, 0.0, 0),
    DecayState(49, 120, 70000.0, 1, half_life=46.2),
    DecayState(49, 120, 2000.0, 2, half_life=46.2),
]}
_IN120_LEVELS = [
    dict(lfs=1, izap=49120, qi=-70000.0, qm=0.0, elfs=70000.0),
    dict(lfs=2, izap=49120, qi=-400000.0, qm=0.0, elfs=400000.0),
]
_IN120_NAMES = {"In120", "In120_m1", "In120_m2"}

# In127: LFS=9 is exactly m2 (1.863 MeV); LFS=1 (408.9 keV) is 155.56% off m1.
_IN127_DECAY = {(49, 127): [
    DecayState(49, 127, 0.0, 0),
    DecayState(49, 127, 160000.0, 1),
    DecayState(49, 127, 1863000.0, 2),
]}
_IN127_LEVELS = [
    dict(lfs=1, izap=49127, qi=-408900.0, qm=0.0, elfs=408900.0),
    dict(lfs=9, izap=49127, qi=-1863000.0, qm=0.0, elfs=1863000.0),
]
_IN127_NAMES = {"In127", "In127_m1", "In127_m2"}


def test_hybrid_mixes_elis_match_and_positional_rescue():
    """In120 (n,n'): one reaction, both phases.

    LFS=1 binds m1 on an exact energy match. LFS=2 is 471% off its nearest
    state, so pure ELIS drops it entirely; the hybrid requeues it and the
    positional pass lands it on the one metastable left -- a decay state whose
    own ELIS is defective (2 keV), which is why the Phase-2 decay pool must not
    be ELIS-filtered.
    """
    recs = {r["lfs"]: r for r in _classify_elis_lfs_order(
        "In120", 4, "(n,n')", _IN120_LEVELS, _IN120_DECAY, _IN120_NAMES,
        0.15, 0.0)}

    exact = recs[1]
    assert (exact["method"], exact["bucket"]) == ("elis", "matched")
    assert (exact["product"], exact["liso"], exact["position"]) == \
        ("In120_m1", 1, 1)

    rescued = recs[2]
    assert (rescued["method"], rescued["fallback_reason"]) == \
        ("lfs_order_fallback", "elis_tol_exceeded")
    assert (rescued["bucket"], rescued["product"], rescued["liso"]) == \
        ("matched", "In120_m2", 2)
    assert rescued["position"] == 2
    assert rescued["large_delta_e"] is True          # positional, flagged
    assert rescued["routed_to_fallback"] is True     # excluded from skip sweeps

    # Contrast: pure ELIS at its own 0.50 tolerance loses the level.
    elis = {r["lfs"]: r for r in _classify_metastables(
        "In120", 4, "(n,n')", _IN120_LEVELS, _IN120_DECAY, _IN120_NAMES,
        "elis", 0.50, 0.0)}
    assert elis[2]["bucket"] == "rtol_exceeded"
    assert elis[2]["diff_pct"] == pytest.approx(471.43, abs=0.01)


def test_hybrid_fallback_skips_the_liso_elis_claimed():
    """In127 (n,n'): the positional pass must not land on a claimed LISO.

    LFS=9 binds m2 by energy, so the only decay state left is m1 -- and m1 is
    also the NEAREST state to LFS=1, which a naive nearest-state pass would
    have taken while m2 was still free. Collision-awareness makes the two
    orders agree here rather than by luck.
    """
    recs = {r["lfs"]: r for r in _classify_elis_lfs_order(
        "In127", 4, "(n,n')", _IN127_LEVELS, _IN127_DECAY, _IN127_NAMES,
        0.15, 0.0)}

    assert (recs[9]["method"], recs[9]["liso"]) == ("elis", 2)
    fb = recs[1]
    assert (fb["method"], fb["fallback_reason"]) == ("lfs_order_fallback",
                                                     "elis_tol_exceeded")
    assert (fb["product"], fb["liso"], fb["position"]) == ("In127_m1", 1, 1)
    assert fb["phase1_claimed_lisos"] == [2]         # m2 was off the table
    assert fb["nearest_liso"] == 1                   # the refused ELIS candidate
    assert fb["diff_pct"] == pytest.approx(155.56, abs=0.01)
    assert fb["large_delta_e"] is True

    # Every level landed in a chain product; nothing was orphaned or dropped.
    assert {r["bucket"] for r in recs.values()} == {"matched"}


def test_hybrid_abstains_when_two_decay_states_are_degenerate():
    """An ambiguous ELIS match is no evidence, so the hybrid re-derives
    positionally; pure ELIS still accepts the arbitrary pick.

    Both decay levels sit at 200 keV, so both file levels "match" m1 and pure
    ELIS keeps only the closer one, discarding the other as a duplicate. The
    hybrid abstains on both and maps them by rank instead -- and it abstains
    QUIETLY (``warn_ambiguity=False``), where the ELIS mapper warns.
    """
    levels = [dict(lfs=1, izap=49122, qi=-210000.0, qm=0.0, elfs=210000.0),
              dict(lfs=5, izap=49122, qi=-205000.0, qm=0.0, elfs=205000.0)]
    names = {"In122", "In122_m1", "In122_m2"}

    decay_elis._WARNED_ELIS_AMBIGUITY.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hybrid = {r["lfs"]: r for r in _classify_elis_lfs_order(
            "In121", 102, "(n,gamma)", levels, _DEGENERATE_DECAY, names,
            0.15, 0.0)}
    assert caught == []                              # abstention is silent

    assert [(r["method"], r["fallback_reason"], r["product"])
            for r in (hybrid[1], hybrid[5])] == \
        [("lfs_order_fallback", "elis_ambiguous", "In122_m1"),
         ("lfs_order_fallback", "elis_ambiguous", "In122_m2")]
    assert (hybrid[1]["second_liso"], hybrid[1]["second_dk_elis"]) == \
        (2, 200000.0)

    # CONTRAST: pure ELIS accepts the coin toss -- LFS=5 (the closer of the two)
    # takes m1 and LFS=1 is discarded as its duplicate, so a whole level is lost.
    decay_elis._WARNED_ELIS_AMBIGUITY.clear()
    with pytest.warns(UserWarning, match="Ambiguous ELIS match"):
        elis = {r["lfs"]: r for r in _classify_metastables(
            "In121", 102, "(n,gamma)", levels, _DEGENERATE_DECAY, names,
            "elis", 0.15, 0.0)}
    assert (elis[5]["bucket"], elis[5]["product"]) == ("matched", "In122_m1")
    assert (elis[1]["bucket"], elis[1]["kept_lfs"]) == ("duplicate", 5)


def test_hybrid_placeholder_lfs_binds_last_and_never_displaces():
    """Placeholder LFS 40/99: no rank, no energy claim, bound by elimination.

    ``QI == 0`` with ``QM > 0`` is the blank-QI disguise -- ELFS "=" QM, an
    energy that is in fact UNKNOWN -- so the placeholder never enters Phase 1
    even though its apparent 5 MeV would have passed for m2. It binds the
    lowest unclaimed LISO afterwards. When real levels have taken every state
    it is report-only. Neither shape may mint an ``_m40`` / ``_m99`` name.
    """
    decay = {(72, 178): [DecayState(72, 178, 0.0, 0),
                         DecayState(72, 178, 1147400.0, 1, half_life=4.0),
                         DecayState(72, 178, 2446000.0, 2, half_life=9.8e8)]}
    names = {"Hf178", "Hf178_m1", "Hf178_m2"}
    placeholder = dict(lfs=99, izap=72178, qi=0.0, qm=5000000.0,
                       elfs=5000000.0)

    bound = {r["lfs"]: r for r in _classify_elis_lfs_order(
        "Hf178", 4, "(n,n')", [placeholder], decay, names, 0.15, 0.0)}[99]
    assert (bound["method"], bound["fallback_reason"]) == ("placeholder_bound",
                                                           "energy_unknown")
    assert (bound["bucket"], bound["product"], bound["liso"]) == \
        ("matched", "Hf178_m1", 1)                   # LOWEST unclaimed, not m2
    assert bound["position"] is None                 # never ranks among levels

    # Real levels outrank it for the remaining states -> report-only.
    crowded = _classify_elis_lfs_order(
        "Hf178", 4, "(n,n')",
        [dict(lfs=1, izap=72178, qi=-1147400.0, qm=0.0, elfs=1147400.0),
         dict(lfs=2, izap=72178, qi=-2446000.0, qm=0.0, elfs=2446000.0),
         dict(lfs=40, izap=72178, qi=0.0, qm=5000000.0, elfs=5000000.0)],
        decay, names, 0.15, 0.0)
    by_lfs = {r["lfs"]: r for r in crowded}
    assert by_lfs[40]["bucket"] == "placeholder_unmapped"
    assert by_lfs[40]["product"] is None
    assert by_lfs[40]["needs_chain_entry"] is False  # never orphan-added
    assert [by_lfs[1]["product"], by_lfs[2]["product"]] == ["Hf178_m1",
                                                            "Hf178_m2"]
    assert not any(str(r.get("product")).endswith(("_m40", "_m99"))
                   for r in crowded)


def test_hybrid_degenerates_to_lfs_order_when_no_elfs_is_usable():
    """A library whose ELFS is identically zero carries no energy evidence at
    all, so every level falls to the positional pass and the hybrid's mapped
    set is exactly ``lfs_order``'s."""
    levels = [dict(lfs=1, izap=49116, qi=0.0, qm=0.0, elfs=0.0),
              dict(lfs=4, izap=49116, qi=0.0, qm=0.0, elfs=0.0)]
    names = {"In116", "In116_m1", "In116_m2"}

    hybrid = _classify_elis_lfs_order("In115", 102, "(n,gamma)", levels,
                                      _decay_lookup(), names, 0.15, 0.0)
    positional = _classify_lfs_order("In115", 102, "(n,gamma)", levels,
                                     _decay_lookup(), names, 0.15, 0.0)

    assert [(r["lfs"], r["product"]) for r in hybrid] == \
        [(r["lfs"], r["product"]) for r in positional] == \
        [(1, "In116_m1"), (4, "In116_m2")]
    assert {r["fallback_reason"] for r in hybrid} == {"elfs_unusable"}
    assert {r["method"] for r in hybrid} == {"lfs_order_fallback"}


def test_unknown_mapping_mode_raises():
    """A mode string registered at one dispatch site and forgotten at another
    must fail loudly rather than fall through to 'elis'."""
    with pytest.raises(ValueError, match="Unknown mapping mode"):
        _classify_metastables("In115", 102, "(n,gamma)",
                              _in115_ng_metastables(), _decay_lookup(),
                              {"In116", "In116_m1"}, "bogus", 0.15, 0.0)


# ---------------------------------------------------------------------------
# Orphan policy (--orphan-policy add-stable | drop | reattribute)
#
# An orphan is a level the mapper could not identify against the decay library.
# The mapper always mints a name and keeps the level; what the branch becomes
# is the writer's decision. Shape: Au186 (n,2n) -> Au185 with LFS=0 and LFS=6
# (200 keV, peak 0.234 b), where decay_2020 carries NO Au185 metastable at all,
# so LFS=6 is a true orphan in every mapping mode.
# ---------------------------------------------------------------------------

_AU185_DECAY = {(79, 185): [DecayState(79, 185, 0.0, 0)],
                (79, 186): [DecayState(79, 186, 0.0, 0)]}


def _au186_n2n(qi_ground=-7928030.0):
    """Au186 (n,2n): ground LFS=0 plus the orphan LFS=6 at ELFS 200 keV."""
    return dict(qm=qi_ground, qi=qi_ground, partials=[
        dict(lfs=0, izap=79185, qi=qi_ground, qm=qi_ground, elfs=0.0),
        dict(lfs=6, izap=79185, qi=qi_ground - 200000.0, qm=qi_ground,
             elfs=200000.0)])


def _orphan_run(policy, data=None, decay=None, names=None, reactions=None,
                mode="elis_lfs_order"):
    """map_library + decorate_chain under one policy; returns (chain, stats)."""
    source = _FakeSource(data if data is not None else
                         {"Au186": {16: _au186_n2n()}})
    chain = _chain_with(names if names is not None else
                        ["Au185", "Au186"],
                        reactions=reactions if reactions is not None else
                        {"Au186": [("(n,2n)", "Au185", -7928030.0)]})
    branching, stats = map_library(
        source, chain, decay if decay is not None else _AU185_DECAY,
        mode, MODE_DEFAULT_RTOL[mode], 0.0, verbose=False,
        orphan_policy=policy)
    n_before = len(chain.nuclides)
    stats["reactions_added"] = decorate_chain(chain, branching, stats,
                                              orphan_policy=policy)
    stats["orphan_nuclides_added_count"] = len(chain.nuclides) - n_before
    return chain, stats


def test_add_stable_mints_one_orphan_shared_by_two_parents(tmp_path):
    """The orphan enters the chain as a stable nuclide, the branch is kept, and
    two parents orphaning the SAME state share one ``<nuclide>`` element.

    ``Au185_m1`` is a placeholder ordinal, not a decay-library LISO: the decay
    library has no Au185 metastable, which is exactly why the state is orphaned.
    """
    chain, stats = _orphan_run(
        "add-stable",
        data={"Au186": {16: _au186_n2n()},
              "Hg185": {103: _au186_n2n(-5000000.0)}},
        names=["Au185", "Au186", "Hg185"],
        reactions={"Au186": [("(n,2n)", "Au185", -7928030.0)],
                   "Hg185": [("(n,p)", "Au185", -5000000.0)]})

    names = [n.name for n in chain.nuclides]
    assert names.count("Au185_m1") == 1               # ONE element, two parents
    assert names.index("Au185_m1") == names.index("Au185") + 1
    assert chain["Au185_m1"].half_life is None        # pure sink, no decay
    assert stats["orphan_products_kept"] == 2
    assert [(s["parent"], s["mt"]) for s in
            stats["orphan_nuclides_added"]["Au185_m1"]] == \
        [("Au186", 16), ("Hg185", 103)]

    members = {(rx.type, rx.target, rx.pendf_lfs)
               for rx in chain["Au186"].reactions}
    assert members == {("(n,2n)", "Au185", 0), ("(n,2n)_m1", "Au185_m1", 6)}

    # The durable contract: the patched chain reloads with the branch intact.
    out = tmp_path / "add_stable.xml"
    chain.export_to_xml(out)
    assert 'targets="Au185 Au185_m1"' in out.read_text()
    reread = Chain.from_xml(out)
    assert "Au185_m1" in reread.nuclide_dict
    assert {(rx.type, rx.target, rx.pendf_lfs)
            for rx in reread["Au186"].reactions} == members
    # validate() is a smoke call only -- its reactions branch carries a live
    # upstream defect, so neither exception type nor text can be asserted on.
    try:
        reread.validate(strict=True)
    except Exception:
        pass


def test_add_stable_materialises_referenced_but_missing_ground(tmp_path):
    """A reaction whose GROUND product the chain never carried is materialised
    too, not just the orphan level -- otherwise the kept branch would point at
    a nuclide that does not exist and the chain would not reload."""
    chain, stats = _orphan_run("add-stable", names=["Au186"], reactions={})

    assert {"Au185", "Au185_m1"} <= set(chain.nuclide_dict)
    assert stats["orphan_grounds_added"] == 1
    assert [s["method"] for s in stats["orphan_nuclides_added"]["Au185"]] == \
        ["missing_ground"]

    out = tmp_path / "ground.xml"
    chain.export_to_xml(out)
    assert {"Au185", "Au185_m1"} <= set(Chain.from_xml(out).nuclide_dict)


# The status quo of the pre-triad patcher, generated by running the fixture
# below through the pinned tool at commit 2edc7cad1 (the last commit before
# this work order). `-m elis --orphan-policy drop` must still reproduce it
# byte for byte: In120's 471%-off LFS=2 vanishes and Au186's undecodable LFS=6
# leaves the whole reaction stock.
_DROP_STATUS_QUO = """\
<depletion_chain>
  <nuclide name="Au185" reactions="0"/>
  <nuclide name="Au186" reactions="1">
    <reaction type="(n,2n)" Q="-7928030.0" target="Au185"/>
  </nuclide>
  <nuclide name="In115" reactions="3">
    <reaction type="(n,gamma)">
      <isomeric_branching targets="In116 In116_m1 In116_m2" pendf_lfs="0 1 4" \
Q="6784720.0 6657450.0 6495060.0"/>
    </reaction>
  </nuclide>
  <nuclide name="In116" reactions="0"/>
  <nuclide name="In116_m1" reactions="0"/>
  <nuclide name="In116_m2" reactions="0"/>
  <nuclide name="In120" reactions="2">
    <reaction type="(n,n')">
      <isomeric_branching targets="In120 In120_m1" pendf_lfs="0 1" \
Q="0.0 -70000.0"/>
    </reaction>
  </nuclide>
  <nuclide name="In120_m1" reactions="0"/>
  <nuclide name="In120_m2" reactions="0"/>
</depletion_chain>
"""


def _regression_scene():
    """One matched reaction, one rtol casualty, one undecodable product."""
    data = {
        "In115": {102: dict(qm=6784720.0, qi=6784720.0, partials=(
            [dict(lfs=0, izap=49116, qi=6784720.0, qm=6784720.0, elfs=0.0)]
            + _in115_ng_metastables()))},
        "In120": {4: dict(qm=0.0, qi=0.0, partials=(
            [dict(lfs=0, izap=49120, qi=0.0, qm=0.0, elfs=0.0)]
            + _IN120_LEVELS))},
        "Au186": {16: _au186_n2n()},
    }
    decay = dict(_decay_lookup())
    decay.update(_IN120_DECAY)
    decay.update(_AU185_DECAY)
    names = ["Au185", "Au186", "In115", "In116", "In116_m1", "In116_m2",
             "In120", "In120_m1", "In120_m2"]
    reactions = {"In115": [("(n,gamma)", "In116", 6784720.0)],
                 "In120": [("(n,n')", "In120", 0.0)],
                 "Au186": [("(n,2n)", "Au185", -7928030.0)]}
    return data, decay, names, reactions


def test_drop_reproduces_the_status_quo_byte_for_byte(tmp_path):
    """``-m elis --orphan-policy drop`` is the regression mode: the triad must
    not have moved a single byte of the chain it used to write."""
    data, decay, names, reactions = _regression_scene()
    chain, stats = _orphan_run("drop", data=data, decay=decay, names=names,
                               reactions=reactions, mode="elis")

    out = tmp_path / "chain.xml"
    chain.export_to_xml(out)
    assert out.read_text() == _DROP_STATUS_QUO

    # Nothing was added and the writer recorded no disposition at all: under
    # `drop` the mapper carries no orphan records, so the policy dispatch is
    # never even reached.
    assert stats["orphan_nuclides_added_count"] == 0
    assert stats["orphan_dispositions"] == []
    assert stats["orphan_policy"] == "drop"


def test_drop_status_quo_holds_in_every_mapping_mode(tmp_path):
    """The regression invariant is a POLICY, not a mode: no mapping mode adds
    a nuclide or records a disposition under ``drop``."""
    data, decay, names, reactions = _regression_scene()
    for mode in MAPPING_MODES:
        chain, stats = _orphan_run("drop", data=data, decay=decay,
                                   names=names, reactions=reactions, mode=mode)
        assert "Au185_m1" not in chain.nuclide_dict, mode
        assert stats["orphan_dispositions"] == [], mode
        assert stats["orphan_policy"] == "drop", mode


def test_reattribute_folds_to_ground_and_round_trips(tmp_path):
    """With no kept sibling below it, the orphan's COLUMN (barns, not a ratio)
    is folded onto the ground: a duplicate-target entry that keeps the orphan's
    own LFS and its own QI, and that survives serialise -> re-parse."""
    chain, stats = _orphan_run("reattribute")

    assert stats["orphan_folds_ground"] == 1
    assert stats["orphan_folds_sibling"] == 0
    assert "Au185_m1" not in chain.nuclide_dict       # nothing minted
    entry, = [d for d in stats["orphan_dispositions"]]
    assert (entry["disposition"], entry["recipient"]) == ("folded_to_ground",
                                                          "Au185")

    members = {(rx.type, rx.target, rx.pendf_lfs, rx.Q)
               for rx in chain["Au186"].reactions}
    assert members == {("(n,2n)", "Au185", 0, -7928030.0),
                       ("(n,2n)", "Au185", 6, -8128030.0)}

    out = tmp_path / "reattr.xml"
    chain.export_to_xml(out)
    reread = Chain.from_xml(out)
    assert {(rx.type, rx.target, rx.pendf_lfs, rx.Q)
            for rx in reread["Au186"].reactions} == members


def test_reattribute_mixed_ground_fold_and_kept_sibling_round_trips(tmp_path):
    """The awkward group: a ground-fold entry (LFS>0 on a ground-named target)
    ALONGSIDE a kept ``_m`` sibling. That mixture is foldable, so it used to
    re-parse as an isomer whose target has no ``_m<n>`` suffix and raise; the
    writer must never emit a chain the parser refuses.
    """
    decay = {(79, 185): [DecayState(79, 185, 0.0, 0),
                         DecayState(79, 185, 300000.0, 1)],
             (79, 186): [DecayState(79, 186, 0.0, 0)]}
    data = {"Au186": {16: dict(qm=-7928030.0, qi=-7928030.0, partials=[
        dict(lfs=0, izap=79185, qi=-7928030.0, qm=-7928030.0, elfs=0.0),
        dict(lfs=2, izap=79185, qi=-8128030.0, qm=-7928030.0, elfs=200000.0),
        dict(lfs=7, izap=79185, qi=-8228030.0, qm=-7928030.0,
             elfs=300000.0)])}}
    chain, stats = _orphan_run("reattribute", data=data, decay=decay,
                               names=["Au185", "Au185_m1", "Au186"])

    members = {(rx.type, rx.target, rx.pendf_lfs)
               for rx in chain["Au186"].reactions}
    # LFS=7 matches the sole decay metastable, LFS=2 has nothing below it but
    # the ground -> a ground fold sitting next to a kept _m1 sibling.
    assert ("(n,2n)", "Au185", 2) in members
    assert any(t.endswith("_m1") for t, _tgt, _l in members)

    out = tmp_path / "mixed.xml"
    chain.export_to_xml(out)
    reread = Chain.from_xml(out)                      # must not raise
    assert {(rx.type, rx.target, rx.pendf_lfs)
            for rx in reread["Au186"].reactions} == members


def test_reattribute_without_a_recipient_drops_loudly():
    """No kept sibling and no usable ground: the share is stranded. It is
    dropped -- but recorded, counted and named, never silently."""
    data = {"Au186": {16: dict(qm=-7928030.0, qi=-7928030.0, partials=[
        dict(lfs=6, izap=79185, qi=-8128030.0, qm=-7928030.0,
             elfs=200000.0)])}}
    chain, stats = _orphan_run("reattribute", data=data, names=["Au186"],
                               reactions={})

    assert stats["orphan_folds_stranded"] == 1
    assert [d["disposition"] for d in stats["orphan_dispositions"]] == \
        ["dropped_no_recipient"]
    assert not chain["Au186"].reactions               # reaction left stock
    assert "Au185_m1" not in chain.nuclide_dict


def test_hybrid_log_totals_identity_and_sections(tmp_path):
    """The counter block's identity anchor: every classified level lands in
    exactly ONE terminal class, stamped [ok] against the raw level count. The
    orphan-disposition section is written whatever the policy."""
    data, decay, names, reactions = _regression_scene()
    chain, stats = _orphan_run("add-stable", data=data, decay=decay,
                               names=names, reactions=reactions)

    log = tmp_path / "log.txt"
    write_isomer_mapping_log(
        log, stats,
        dict(base_chain="fixture", pendf="fixture", decay_file="fixture",
             output_chain="fixture", chain_nuclides=len(chain.nuclides)),
        "elis_lfs_order", MODE_DEFAULT_RTOL["elis_lfs_order"], 0.0)
    text = log.read_text()

    census = _hybrid_census(stats)
    assert census["terminal"] == stats["total_lfs"]
    assert (census["phase1"] + census["phase2"] + census["placeholder_bound"]
            + census["placeholder_unmapped"] + census["orphan_levels"]) == \
        census["terminal"]
    assert f"Total PENDF-LFS found: {census['terminal']:5d}" in text
    assert "terminal classes, one per level (sum)" in text
    assert "[ok]" in text and "[MISMATCH]" not in text

    assert "ORPHAN DISPOSITION\n" in text            # always written
    assert "ORPHAN NUCLIDES ADDED TO CHAIN" in text  # add-stable only
    assert "HYBRID MAPPING" in text
    assert "sentinel" not in text.lower()            # placeholder wording only
