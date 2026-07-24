"""Unit tests for the patcher's MF=10 pre-passes (emission + IZAP=0 attribution).

``tools/add_gendf_isomeric_branching_to_chain.py`` emits plain ``<reaction>``
elements for EAF-2010 channels stored only in MF=8/10 (no MF=3), which
``Chain.from_endf`` never harvested. Two passes:

* ``emit_mf10_only_prepass`` (pre-pass) emits GROUND-target reactions and defers
  single-metastable-only channels; runs before the branching pipeline.
* ``emit_metastable_direct_postpass`` (post-pass) emits the deferred
  metastable-only channels as plain static-target reactions (e.g.
  In115(n,n')->In115m); runs after decoration so Gate 1 never sees them.

The IZAP=0 attribution pre-pass (``scan_mf10_attribution``, R1-61) runs ahead of
both: it gates anonymous MF=10 subsections on evidence C1-C4 and either recovers
the residual (opt-in ``--reattribute-mf10-noIZAP``) or prunes the reaction's
entire isomeric decoration.

Exercised on a synthetic chain + GENDF stand-in -- no real data files, no
NukeData paths.
"""

import copy
import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

import openmc.deplete
from openmc.deplete.decay_elis import ELIS_ATOL, ELIS_RTOL, DecayState
from openmc.deplete.gendf import (IsomericBranching, _PythonGENDFLibrary,
                                  get_product_name)

# Import the standalone tool module (not an installed package).
_TOOL_PATH = (Path(__file__).parents[3] / "tools"
              / "add_gendf_isomeric_branching_to_chain.py")
_spec = importlib.util.spec_from_file_location("gendf_patcher_tool", _TOOL_PATH)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def _level(lfs, izap, qm, qi):
    return {"LFS": lfs, "IZAP": izap, "QM": qm, "QI": qi, "sigma": None}


class _ToyLib(_PythonGENDFLibrary):
    """GENDF stand-in: in-memory section data behind the REAL library methods.

    Subclassing keeps ``process_library_for_branching`` / ``get_branching_ratios``
    genuine, so extraction is exercised end-to-end without data files.
    """

    def __init__(self, section_data, decay_lookup=None):
        self._sd = section_data
        self.decay_lookup = decay_lookup or {}
        self._mapping_mode = 'elis'
        self._elis_rtol = ELIS_RTOL
        self._elis_atol = ELIS_ATOL
        self._skip_zero_elis_metastables = True

    def available_nuclides(self):
        return sorted(self._sd)

    def available_nuclides_set(self):
        return frozenset(self._sd)

    def _load_material(self, nuclide_name, require_full_parser=False):
        class _Mat:
            pass
        mat = _Mat()
        mat.section_data = self._sd[nuclide_name]
        return mat

    def _load_mf10_data(self, nuclide_name, mt):
        section = self._sd[nuclide_name].get((10, mt))
        return None if section is None else (section, section.get('QM'))


def _toy_chain(tmp_path):
    """Al27 with only (n,p) in MF=3; residual products (incl. Al27_m1) present."""
    chain = openmc.deplete.Chain()
    al = openmc.deplete.Nuclide("Al27")
    al.add_reaction("(n,p)", "Mg27", Q=-1828549.0, branching_ratio=1.0)
    chain.add_nuclide(al)
    for name in ("Mg27", "Na24", "Na24_m1", "Al26", "Al26_m1", "Al27_m1"):
        chain.add_nuclide(openmc.deplete.Nuclide(name))
    path = tmp_path / "base_chain.xml"
    chain.export_to_xml(str(path))
    return chain, path


# Al27 section_data: (n,p) in MF=3; the rest MF=10-only.
def _toy_sections():
    return {"Al27": {
        (3, 103): {"sigma": None},                                        # (n,p): MF=3
        (10, 107): {"levels": [_level(0, 11024, -3131600.0, -3131600.0),
                               _level(1, 11024, -3131600.0, -3605133.0)]},  # (n,a)->Na24 [+meta]
        (10, 16): {"levels": [_level(0, 13026, -13057800.0, -13057800.0),
                              _level(1, 13026, -13057800.0, -13290000.0)]},  # (n,2n)->Al26 [+meta]
        (10, 34): {"levels": [_level(0, 11024, -23710400.0, -23710400.0)]},  # (n,n3He) ground-only
        (10, 37): {"levels": [_level(0, 13024, -41355800.0, -41355800.0),
                              _level(1, 13024, -41355800.0, -41781600.0)]},  # (n,4n)->Al24: absent
        # (n,n') metastable-only single -> Al27_m1 (IN chain): deferred, emitted post.
        (10, 4): {"levels": [_level(1, 13027, 0.0, -843800.0)]},
        # (n,3n) metastable-only single -> Al25_m1 (NOT in chain): deferred, skipped post.
        (10, 17): {"levels": [_level(1, 13025, 0.0, -12000000.0)]},
        # (n,t) metastable-only MULTI (two metastable levels, no ground): not emitted.
        (10, 105): {"levels": [_level(1, 12025, 0.0, -9000000.0),
                               _level(2, 12025, 0.0, -9500000.0)]},
    }}


def test_emit_mf10_only_prepass(tmp_path):
    chain, base = _toy_chain(tmp_path)
    lib = _ToyLib(_toy_sections())
    out = tmp_path / "emitted.xml"

    summary = tool.emit_mf10_only_prepass(lib, chain, str(base), str(out),
                                          verbose=False)

    # Ground emissions: (n,a), (n,2n), (n,n3He, ground-only). (n,4n)->Al24 not in
    # chain -> no-target. (n,t) is multi-metastable -> skipped. (n,n')/(n,3n) are
    # single-metastable-only -> DEFERRED to the post-pass (not terminal here).
    assert summary["emitted"] == 3
    assert summary["emit_ground_only"] == 1
    assert summary["emit_skipped_no_target"] == 1
    assert summary["emit_skipped_multi_metastable"] == 1
    assert summary["emit_skipped_no_name"] == 0
    assert summary["emit_skipped_exists"] == 0
    assert summary["emit_skipped_load_error"] == 0
    assert summary["emitted_metastable_direct"] == 0        # post-pass not run yet
    assert len(summary["metastable_deferred"]) == 2         # MT=4 and MT=17

    # Emitted plain ground reactions carry the LFS=0 subsection QM as the scalar Q.
    root = ET.parse(str(out)).getroot()
    al = next(n for n in root.findall("nuclide") if n.get("name") == "Al27")
    rmap = {r.get("type"): r for r in al.findall("reaction")}
    assert rmap["(n,a)"].get("target") == "Na24"
    assert rmap["(n,a)"].get("Q") == "-3131600.0"
    assert rmap["(n,2n)"].get("target") == "Al26"
    assert rmap["(n,2n)"].get("Q") == "-13057800.0"
    assert "(n,4n)" not in rmap    # Al24 not in chain
    assert "(n,n')" not in rmap    # metastable-only MT=4 deferred, not yet emitted
    assert "(n,3n)" not in rmap    # metastable-only MT=17 deferred

    # The emitted XML round-trips: Gate 1 (Chain object) sees the new reactions.
    reloaded = openmc.deplete.Chain.from_xml(str(out))
    al_types = {r.type for r in reloaded["Al27"].reactions}
    assert {"(n,a)", "(n,2n)", "(n,n3He)", "(n,p)"} <= al_types


def test_emit_metastable_direct_postpass(tmp_path):
    chain, base = _toy_chain(tmp_path)
    lib = _ToyLib(_toy_sections())
    out = tmp_path / "emitted.xml"
    summary = tool.emit_mf10_only_prepass(lib, chain, str(base), str(out),
                                          verbose=False)

    # Post-pass runs on the writer tree (here the pre-pass output stands in).
    root = ET.parse(str(out)).getroot()
    tool.emit_metastable_direct_postpass(root, summary)

    # MT=4 -> Al27_m1 (in chain) emitted directly; MT=17 -> Al25_m1 (absent) skipped.
    assert summary["emitted_metastable_direct"] == 1
    assert summary["emit_skipped_no_target"] == 2          # MT=37 (pre) + MT=17 (post)
    assert summary["metastable_direct_details"] == [
        ("Al27", "(n,n')", "Al27_m1", "-843800.0")]

    al = next(n for n in root.findall("nuclide") if n.get("name") == "Al27")
    rmap = {r.get("type"): r for r in al.findall("reaction")}
    # Plain static-target reaction: target = the metastable product, Q = its QI.
    assert rmap["(n,n')"].get("target") == "Al27_m1"
    assert rmap["(n,n')"].get("Q") == "-843800.0"

    # D5: a direct X->X_m1 (n,n') is NOT an exact self-loop -> prune keeps it.
    assert rmap["(n,n')"].get("target") != "Al27"


def test_emit_metastable_direct_postpass_idempotent(tmp_path):
    chain, base = _toy_chain(tmp_path)
    lib = _ToyLib(_toy_sections())
    out = tmp_path / "emitted.xml"
    summary = tool.emit_mf10_only_prepass(lib, chain, str(base), str(out),
                                          verbose=False)
    root = ET.parse(str(out)).getroot()
    tool.emit_metastable_direct_postpass(root, summary)   # first emission

    # Re-run the post-pass on the ALREADY-patched tree with the same deferred list
    # (fresh counters): the type is now present, so nothing new is emitted.
    second = copy.deepcopy(summary)
    for key in ("emitted_metastable_direct", "emit_skipped_exists",
                "emit_skipped_no_target"):
        second[key] = 0
    second["metastable_direct_details"] = []
    tool.emit_metastable_direct_postpass(root, second)

    assert second["emitted_metastable_direct"] == 0
    assert second["emit_skipped_exists"] == 1              # MT=4 (n,n') already present
    assert second["emit_skipped_no_target"] == 1           # MT=17 -> Al25_m1 absent


# =============================================================================
# Anonymous (IZAP=0) MF=10 re-attribution (R1-61)
# =============================================================================

_AM242M_ELIS = 48600.0     # JEFF-3.3 Am241(n,gamma): QM - QI = 48.6 keV
_AM241_QM = 5537800.0


def _anon_sections(mt=102, elfs_meta=_AM242M_ELIS, meta_izap=0, mf3=True):
    """JEFF-3.3 shape: an Am241 pair whose ground MF=10 level carries IZAP=0.

    ``meta_izap=0`` (default) makes both levels anonymous; a non-zero value
    gives the MIXED section -- anonymous ground, attributed metastable -- whose
    partial decoration is exactly the R1-61 inversion. ``mf3=False`` drops the
    MF=3 section, making the channel MF=10-only (emission-pass territory).
    """
    sections = {(10, mt): {"levels": [_level(0, 0, _AM241_QM, _AM241_QM),
                                      _level(1, meta_izap, _AM241_QM,
                                             _AM241_QM - elfs_meta)]}}
    if mf3:
        sections[3, mt] = {"sigma": None}
    return {"Am241": sections}


def _anon_decay_lookup():
    return {(95, 242): [DecayState(z=95, a=242, elis=0.0, liso=0),
                        DecayState(z=95, a=242, elis=_AM242M_ELIS, liso=1,
                                   half_life=4907.0)]}


def _anon_chain(tmp_path):
    chain = openmc.deplete.Chain()
    am = openmc.deplete.Nuclide("Am241")
    am.add_reaction("(n,gamma)", "Am242", Q=_AM241_QM, branching_ratio=1.0)
    chain.add_nuclide(am)
    for name in ("Am242", "Am242_m1"):
        chain.add_nuclide(openmc.deplete.Nuclide(name))
    path = tmp_path / "anon_chain.xml"
    chain.export_to_xml(str(path))
    return chain, path


def _anon_branching():
    """What the extractor yields once the IZAP=0 levels have been recovered."""
    return {"Am241": {"(n,gamma)": IsomericBranching(
        energies=np.array([1.0, 1.0e7]),
        products=["Am242", "Am242_m1"],
        branching_ratios=np.array([[0.906, 0.55], [0.094, 0.45]]),
        parent_nuclide="Am241", reaction="(n,gamma)", mt=102,
        lfs_mapping={"Am242": 0, "Am242_m1": 1})}}


def _reaction_elem(xml_path, nuclide, rtype):
    root = ET.parse(str(xml_path)).getroot()
    nuc = next(n for n in root.findall("nuclide") if n.get("name") == nuclide)
    return next(r for r in nuc.findall("reaction") if r.get("type") == rtype)


@pytest.mark.parametrize("meta_izap, n_anon", [(0, 2), (95242, 1)])
def test_izap0_reattribution_off_prunes_whole_decoration(tmp_path, capsys,
                                                         meta_izap, n_anon):
    chain, base = _anon_chain(tmp_path)
    lib = _ToyLib(_anon_sections(meta_izap=meta_izap), _anon_decay_lookup())

    records, pruned, counts = tool.scan_mf10_attribution(lib, chain,
                                                         reattribute=False)
    assert [r["status"] for r in records] == ["pruned"]
    assert records[0]["reason"] == "flag_off"
    assert pruned == {("Am241", "(n,gamma)")}
    assert counts["pruned"] == 1
    assert counts["anonymous_levels"] == n_anon
    # Nothing repaired with the flag off.
    assert [lv["IZAP"] for lv in lib._sd["Am241"][(10, 102)]["levels"]] == \
        [0, meta_izap]

    tool._print_attribution_summary(records, counts, False)
    out = capsys.readouterr().out
    assert "WARNING" in out and "PRUNED" in out and "Am241" in out

    # Real extraction: the mixed shape (attributed metastable, anonymous ground)
    # raises "no ground state" unless the pruned set is excluded up front, and
    # the collect-and-raise policy would then kill the run before the prune.
    data = lib.process_library_for_branching(mt_list=[102], chain=chain,
                                             skip_reactions=pruned)
    data = tool.prune_unattributed_decoration(data, pruned)
    assert data == {}
    patched = tmp_path / "patched.xml"
    tool.add_branching_to_xml(str(base), data, str(patched), chain,
                              verbose=False)
    rx = _reaction_elem(patched, "Am241", "(n,gamma)")
    assert rx.find("isomeric_branching") is None
    assert rx.find("isomeric_yields") is None
    assert rx.get("target") == "Am242"     # plain MF=3 route survives


def test_izap0_reattribution_on_recovers_and_writes_flags(tmp_path, capsys):
    chain, base = _anon_chain(tmp_path)
    lib = _ToyLib(_anon_sections(), _anon_decay_lookup())

    records, pruned, counts = tool.scan_mf10_attribution(lib, chain,
                                                         reattribute=True)
    assert [r["status"] for r in records] == ["reattributed"]
    assert pruned == set()
    assert counts["reattributed"] == 1

    # Derived from (dA, dZ) = (+1, 0): Am241 -> Am242, both levels named again
    # (get_product_name returns None while IZAP is 0).
    levels = lib._sd["Am241"][(10, 102)]["levels"]
    assert [lv["IZAP"] for lv in levels] == [95242, 95242]
    assert [get_product_name(lv["IZAP"], lv["LFS"]) for lv in levels] == \
        ["Am242", "Am242_m1"]
    # C4 evidence: QM-QI against the decay-library ELIS, in keV.
    assert "ELIS=48.600 keV" in records[0]["recovered"][1]["evidence"]

    tool._print_attribution_summary(records, counts, True)
    assert "REATTRIBUTED" in capsys.readouterr().out

    data = tool.prune_unattributed_decoration(_anon_branching(), pruned)
    patched = tmp_path / "patched.xml"
    tool.add_branching_to_xml(str(base), data, str(patched), chain,
                              verbose=False)
    iso = _reaction_elem(patched, "Am241", "(n,gamma)").find("isomeric_branching")
    assert iso.get("targets") == "Am242 Am242_m1"
    assert iso.get("gendf_lfs") == "0 1"


@pytest.mark.parametrize("kwargs, condition", [
    ({"elfs_meta": 200000.0}, "C4_q"),        # QM-QI 151 keV off the real ELIS
    ({"mt": 16}, "C3_product"),               # (n,2n) -> Am240: not in decay lib
])
def test_izap0_gate_failure_prunes_all_or_nothing(tmp_path, kwargs, condition):
    chain, _base = _anon_chain(tmp_path)
    lib = _ToyLib(_anon_sections(**kwargs), _anon_decay_lookup())

    records, pruned, counts = tool.scan_mf10_attribution(lib, chain,
                                                         reattribute=True)
    assert records[0]["status"] == "pruned"
    assert records[0]["reason"] == condition
    assert counts["pruned"] == 1 and counts["reattributed"] == 0
    assert pruned == {("Am241", records[0]["reaction"])}
    # All-or-nothing: even the LFS=0 level that clears the gate stays unrepaired,
    # so no partially-attributed decoration can be written.
    mt = records[0]["mt"]
    assert [lv["IZAP"] for lv in lib._sd["Am241"][(10, mt)]["levels"]] == [0, 0]


def _anon_emit_chain(tmp_path):
    """Am241 with NO (n,gamma): the channel under test is MF=10-only."""
    chain = openmc.deplete.Chain()
    for name in ("Am241", "Am242", "Am242_m1"):
        chain.add_nuclide(openmc.deplete.Nuclide(name))
    path = tmp_path / "anon_emit_chain.xml"
    chain.export_to_xml(str(path))
    return chain, path


@pytest.mark.parametrize("reattribute, emitted, skipped",
                         [(False, 0, 1), (True, 1, 0)])
def test_emit_prepass_skips_unrepaired_anonymous(tmp_path, reattribute,
                                                 emitted, skipped):
    """R1-61: an MF=10-only section with an anonymous ground emits nothing.

    Filtering IZAP=0 away leaves a lone attributed metastable, which the pre-pass
    would defer and the post-pass emit as a plain 100%-metastable reaction --
    the inversion, reached by a route ``prune_unattributed_decoration`` cannot
    see. Re-attributed, the ground is named again and the ordinary ground
    emission resumes.
    """
    chain, base = _anon_emit_chain(tmp_path)
    lib = _ToyLib(_anon_sections(meta_izap=95242, mf3=False),
                  _anon_decay_lookup())
    records, _pruned, _counts = tool.scan_mf10_attribution(
        lib, chain, reattribute=reattribute)

    out = tmp_path / "emitted.xml"
    summary = tool.emit_mf10_only_prepass(
        lib, chain, str(base), str(out), verbose=False,
        unrepaired_anonymous=tool.unrepaired_anonymous_mts(records))
    root = ET.parse(str(out)).getroot()
    tool.emit_metastable_direct_postpass(root, summary)

    assert summary["emit_skipped_anonymous"] == skipped
    assert summary["metastable_deferred"] == []
    assert summary["emitted_metastable_direct"] == 0
    assert summary["emitted"] == emitted

    am = next(n for n in root.findall("nuclide") if n.get("name") == "Am241")
    targets = {r.get("target") for r in am.findall("reaction")}
    assert "Am242_m1" not in targets                # never a static 100% metastable
    assert ("Am242" in targets) is bool(emitted)    # repaired: plain ground route


def test_isomeric_flags_take_slot0_lfs_from_mapping(tmp_path):
    """R1-49: a metastable promoted into slot 0 keeps its real LFS."""
    chain = openmc.deplete.Chain()
    ir = openmc.deplete.Nuclide("Ir191")
    ir.add_reaction("(n,gamma)", "Ir192_m1", Q=6198000.0, branching_ratio=1.0)
    chain.add_nuclide(ir)
    for name in ("Ir192_m1", "Ir192_m2"):     # ground Ir192 absent from the chain
        chain.add_nuclide(openmc.deplete.Nuclide(name))
    base = tmp_path / "ir_chain.xml"
    chain.export_to_xml(str(base))

    data = {"Ir191": {"(n,gamma)": IsomericBranching(
        energies=np.array([1.0, 1.0e7]),
        products=["Ir192", "Ir192_m1", "Ir192_m2"],
        branching_ratios=np.array([[0.7, 0.6], [0.2, 0.25], [0.1, 0.15]]),
        parent_nuclide="Ir191", reaction="(n,gamma)", mt=102,
        lfs_mapping={"Ir192": 0, "Ir192_m1": 3, "Ir192_m2": 15})}}

    patched = tmp_path / "ir_patched.xml"
    tool.add_branching_to_xml(str(base), data, str(patched), chain,
                              verbose=False)
    iso = _reaction_elem(patched, "Ir191", "(n,gamma)").find("isomeric_branching")
    assert iso.get("targets") == "Ir192_m1 Ir192_m2"
    assert iso.get("gendf_lfs") == "3 15"     # not "0 15"
