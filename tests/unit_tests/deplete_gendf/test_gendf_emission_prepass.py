"""Unit test for the patcher's MF=10-only emission passes (M1 + M1b).

``tools/add_gendf_isomeric_branching_to_chain.py`` emits plain ``<reaction>``
elements for EAF-2010 channels stored only in MF=8/10 (no MF=3), which
``Chain.from_endf`` never harvested. Two passes:

* ``emit_mf10_only_prepass`` (pre-pass) emits GROUND-target reactions and defers
  single-metastable-only channels; runs before the branching pipeline.
* ``emit_metastable_direct_postpass`` (post-pass) emits the deferred
  metastable-only channels as plain static-target reactions (e.g.
  In115(n,n')->In115m); runs after decoration so Gate 1 never sees them.

Exercised on a synthetic chain + GENDF stand-in -- no real data files, no
NukeData paths.
"""

import copy
import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import openmc.deplete

# Import the standalone tool module (not an installed package).
_TOOL_PATH = (Path(__file__).parents[3] / "tools"
              / "add_gendf_isomeric_branching_to_chain.py")
_spec = importlib.util.spec_from_file_location("gendf_patcher_tool", _TOOL_PATH)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def _level(lfs, izap, qm, qi):
    return {"LFS": lfs, "IZAP": izap, "QM": qm, "QI": qi, "sigma": None}


class _ToyLib:
    """GENDF stand-in exposing only what the emission pre-pass touches."""

    def __init__(self, section_data):
        self._sd = section_data

    def available_nuclides_set(self):
        return frozenset(self._sd)

    def _load_material(self, nuclide_name, require_full_parser=False):
        class _Mat:
            pass
        mat = _Mat()
        mat.section_data = self._sd[nuclide_name]
        return mat


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
