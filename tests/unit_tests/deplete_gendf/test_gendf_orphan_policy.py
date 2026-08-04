"""Orphan-state disposition in the patcher's writing layer (--orphan-policy).

An orphan is a reaction product the mapper could not identify against the decay
library -- an excited state with no decay partner, or a whole product nuclide
the decay library never carried. The mapper always mints a name for it and
emits the level; what the branch then becomes is the writer's decision:

* ``add-stable`` (CLI default) adds ``<nuclide name=… reactions="0"/>`` to the
  output chain and keeps the branch. The state is a pure sink -- mass is
  conserved, its own activity is not modelled.
* ``renorm`` is the historical behaviour: drop the branch and rescale the
  surviving targets pro rata.
* ``reattribute`` folds the share into the kept isomer at the nearest LOWER
  rank (cascade-down), leaving the sums at 1 with nothing to renormalize.

Shape: In127(n,n') from Josey, LA-UR-25-32234 -- the reaction carries levels
#0, #1 and #9 while the decay library knows only the ground and m1, so level #9
is a true orphan (ENSDF: In127m2 decays 100% beta-minus, so folding it into the
ground would fabricate physics). Exercised on a synthetic chain, no data files.
"""

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

import openmc.deplete
from openmc.deplete.gendf import IsomericBranching

# Import the standalone tool module (not an installed package).
_TOOL_PATH = (Path(__file__).parents[3] / "tools"
              / "add_gendf_isomeric_branching_to_chain.py")
_spec = importlib.util.spec_from_file_location("gendf_patcher_tool", _TOOL_PATH)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

ORPHAN = "In127_m2"      # the name the hybrid mapper mints for level #9


def _chain(tmp_path, extra_parent=False):
    """In127(n,n') on a chain holding In127 and In127_m1 -- but not the orphan."""
    chain = openmc.deplete.Chain()
    in127 = openmc.deplete.Nuclide("In127")
    in127.add_reaction("(n,n')", "In127", Q=0.0, branching_ratio=1.0)
    chain.add_nuclide(in127)
    chain.add_nuclide(openmc.deplete.Nuclide("In127_m1"))
    if extra_parent:
        sn127 = openmc.deplete.Nuclide("Sn127")
        sn127.add_reaction("(n,p)", "In127", Q=-1.0e6, branching_ratio=1.0)
        chain.add_nuclide(sn127)
    path = tmp_path / "base_chain.xml"
    chain.export_to_xml(str(path))
    return chain, path


def _branching(parent, reaction, mt):
    """Ground + kept m1 (rank 1) + the orphan level #9 (rank 2)."""
    return IsomericBranching(
        energies=np.array([1.0, 1.0e7]),
        products=["In127", "In127_m1", ORPHAN],
        branching_ratios=np.array([[0.5, 0.4], [0.3, 0.2], [0.2, 0.4]]),
        parent_nuclide=parent, reaction=reaction, mt=mt,
        lfs_mapping={"In127": 0, "In127_m1": 1, ORPHAN: 9},
        elis_mapping={
            "In127_m1": {"method": "elis", "liso": 1, "position": 1,
                         "elis": 247900.0, "qm": 0.0, "qi": -247900.0},
            ORPHAN: {"method": "orphan_added", "liso": None, "position": 2,
                     "elis": 1770000.0, "qm": 0.0, "qi": -1770000.0},
        })


def _iso_elem(xml_path, nuclide, tag):
    root = ET.parse(str(xml_path)).getroot()
    nuc = next(n for n in root.findall("nuclide") if n.get("name") == nuclide)
    return root, nuc.find("reaction").find(tag)


def _ratio_rows(yields_elem):
    """The written ``<branching_ratios>`` block as one row per target."""
    return np.array([[float(v) for v in line.split()]
                     for line in yields_elem.find("branching_ratios")
                     .text.strip().splitlines()])


def test_add_stable_keeps_the_branch_and_the_chain_reloads(tmp_path):
    """The orphan enters the chain as a stable nuclide; nothing renormalizes."""
    chain, base = _chain(tmp_path)
    out = tmp_path / "patched.xml"

    summary = tool.add_branching_to_xml(
        str(base), {"In127": {"(n,n')": _branching("In127", "(n,n')", 4)}},
        str(out), chain, verbose=False, orphan_policy="add-stable")

    assert summary["orphan_products_kept"] == 1
    assert summary["renormalizations"] == []

    root, iso = _iso_elem(out, "In127", "isomeric_branching")
    added = [n for n in root.findall("nuclide") if n.get("name") == ORPHAN]
    assert len(added) == 1
    assert added[0].get("reactions") == "0"
    assert iso.get("targets").split() == ["In127", "In127_m1", ORPHAN]
    assert iso.get("gendf_lfs").split() == ["0", "1", "9"]

    # The durable contract: the patched chain reloads and the added state is
    # there, stable (no decay data). validate() is a smoke call only -- its
    # reactions branch carries a live upstream defect, so neither the exception
    # type nor its text can be asserted on.
    reloaded = openmc.deplete.Chain.from_xml(str(out))
    assert reloaded[ORPHAN].half_life is None
    assert reloaded[ORPHAN].decay_modes == []
    try:
        reloaded.validate(strict=True)
    except Exception:
        pass


def test_renorm_drops_the_orphan_and_rescales_the_survivors(tmp_path):
    """Status quo: the branch goes, the kept targets take its share pro rata."""
    chain, base = _chain(tmp_path)
    out = tmp_path / "patched.xml"

    summary = tool.add_branching_to_xml(
        str(base), {"In127": {"(n,n')": _branching("In127", "(n,n')", 4)}},
        str(out), chain, verbose=False, mode="embedded",
        orphan_policy="renorm")

    assert summary["orphan_nuclides_added"] == {}
    assert [(r["dropped_products"], r["valid_products"])
            for r in summary["renormalizations"]] == \
        [([ORPHAN], ["In127", "In127_m1"])]

    root, yields = _iso_elem(out, "In127", "isomeric_yields")
    assert not any(n.get("name") == ORPHAN for n in root.findall("nuclide"))
    assert yields.find("targets").text.split() == ["In127", "In127_m1"]
    rows = _ratio_rows(yields)
    # The ratios ship at six digits, hence the loose tolerance.
    np.testing.assert_allclose(rows, [[0.5 / 0.8, 0.4 / 0.6],
                                      [0.3 / 0.8, 0.2 / 0.6]], rtol=1e-6)
    np.testing.assert_allclose(rows.sum(axis=0), 1.0, rtol=1e-6)


def test_one_nuclide_element_serves_every_parent_of_the_orphan(tmp_path):
    """Two parents feeding the same unidentified state share one nuclide."""
    chain, base = _chain(tmp_path, extra_parent=True)
    out = tmp_path / "patched.xml"

    summary = tool.add_branching_to_xml(
        str(base), {"In127": {"(n,n')": _branching("In127", "(n,n')", 4)},
                    "Sn127": {"(n,p)": _branching("Sn127", "(n,p)", 103)}},
        str(out), chain, verbose=False, orphan_policy="add-stable")

    root = ET.parse(str(out)).getroot()
    assert [n.get("name") for n in root.findall("nuclide")].count(ORPHAN) == 1
    # One element, one source row per contributing pathway (the ELFS spread
    # between them is what shows whether the parents agree on the state).
    assert [(s["parent"], s["mt"], s["created"])
            for s in summary["orphan_nuclides_added"][ORPHAN]] == \
        [("In127", 4, True), ("Sn127", 103, False)]


def test_reattribute_folds_the_orphan_into_the_isomer_below_it(tmp_path):
    """The share cascades down to the kept sibling at the nearest lower rank."""
    chain, base = _chain(tmp_path)
    out = tmp_path / "patched.xml"

    summary = tool.add_branching_to_xml(
        str(base), {"In127": {"(n,n')": _branching("In127", "(n,n')", 4)}},
        str(out), chain, verbose=False, mode="embedded",
        orphan_policy="reattribute")

    assert summary["renormalizations"] == []
    assert summary["orphan_nuclides_added"] == {}
    rec, = summary["reattributions"]
    assert (rec["orphan"], rec["recipient"]) == (ORPHAN, "In127_m1")
    assert (rec["position"], rec["recipient_position"], rec["rank_distance"]) \
        == (2, 1, 1)

    _, yields = _iso_elem(out, "In127", "isomeric_yields")
    assert yields.find("targets").text.split() == ["In127", "In127_m1"]
    rows = _ratio_rows(yields)
    np.testing.assert_allclose(rows, [[0.5, 0.4], [0.5, 0.6]])  # 0.3+0.2, 0.2+0.4
    np.testing.assert_allclose(rows.sum(axis=0), 1.0)
