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
    """

    kind = "fake"
    library = "synthetic"
    mapping = "elis"

    def __init__(self, data):
        self._data = data
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
    (all three grid points fall in the resonance band, so ratio_resonance is
    the same 0.975 while thermal/fast have no points -> None).
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
    # All three grid points sit in the resonance band -> thermal/fast are None.
    assert audit["ratio_thermal"] is None
    assert audit["ratio_resonance"] == pytest.approx(0.975)
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
    # partials = 0.5x total in the thermal band [<1 eV) and = total elsewhere
    # -> ratio_thermal ~ 0.5, ratio_resonance ~ 1.0, ratio_fast ~ 1.0.
    grid = [0.01, 0.1, 1.0, 100.0, 1.0e4, 1.0e5, 1.0e6, 1.0e7]
    total = [10.0] * 8
    ground = [5.0, 5.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    rxn = dict(qm=0.0, qi=0.0, energy=grid, xs=total, partials=[
        dict(lfs=0, izap=49116, qi=0.0, qm=0.0, elfs=0.0,
             energy=grid, xs=ground)])
    source = _FakeSource({"In115": {102: rxn}})
    audit = _audit_reaction(source, "In115", 102, rxn["partials"])
    assert audit["ratio_thermal"] == pytest.approx(0.5)
    assert audit["ratio_resonance"] == pytest.approx(1.0)
    assert audit["ratio_fast"] == pytest.approx(1.0)
    assert audit["notes"] == ""


def test_reject_band_ratio_leaves_offender_stock(tmp_path):
    # reject_band_ratio set (reject_rtol unset): a reaction broken only in the
    # thermal band is rejected on the band criterion and left stock in the XML.
    grid = [0.01, 0.1, 1.0, 100.0, 1.0e4, 1.0e5, 1.0e6, 1.0e7]
    total = [10.0] * 8
    ground = [5.0, 5.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]   # 0.5x in thermal
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
    # A grid that never enters the thermal or fast band leaves those ratios None
    # ('n/a' in the log) and they can NEVER trigger band rejection.
    grid = [10.0, 100.0, 1000.0]                     # all in the resonance band
    total = [10.0, 20.0, 30.0]
    rxn = dict(qm=0.0, qi=0.0, energy=grid, xs=total, partials=[
        dict(lfs=0, izap=49116, qi=0.0, qm=0.0, elfs=0.0,
             energy=grid, xs=[7.0, 14.0, 20.9]),      # sum trails total at 1000
        dict(lfs=1, izap=49116, qi=6657450.0, qm=6784720.0, elfs=127270.0,
             energy=grid, xs=[3.0, 6.0, 9.0])])
    source = _FakeSource({"In115": {102: rxn}})
    audit = _audit_reaction(source, "In115", 102, rxn["partials"])
    assert audit["ratio_thermal"] is None
    assert audit["ratio_fast"] is None
    assert audit["ratio_resonance"] is not None
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
