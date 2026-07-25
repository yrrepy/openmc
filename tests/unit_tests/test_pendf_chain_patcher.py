"""Unit tests for the PENDF isomeric-branching chain patcher mapping core.

These exercise the mapping/decoration logic on small synthetic inputs (no real
PENDF HDF5 or decay library needed): ELIS match happy path, rtol skip,
product-not-in-chain skip, reaction-added-to-chain, ground-only, and the full
decorate -> export -> reload fold round-trip.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import openmc.deplete
from openmc.deplete import Chain, Nuclide
from openmc.deplete.decay_elis import DecayState

# The patcher lives in the repo's tools/ directory (not an installed package).
_TOOLS = Path(openmc.deplete.__file__).parents[2] / "tools"
sys.path.insert(0, str(_TOOLS))

from add_pendf_isomeric_branching_to_chain import (  # noqa: E402
    _classify_metastables, map_library, decorate_chain, _audit_reaction,
    _self_loop_ground,
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
                        lambda path, library=None: fake_source)
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
