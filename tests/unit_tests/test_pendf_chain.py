"""Unit tests for building a PENDF-native depletion chain."""

from pathlib import Path

import pytest

from openmc.deplete.pendf_chain import chain_from_pendf

_PENDF_H5 = Path(
    "/home/perry/Projects/OMC_Development/PENDF/data/tendl2017_pendf_293K_elis.h5")
_DECAY_DIR = Path("/home/perry/NukeData/Activation/DecayData/decay_2020")

pytestmark = pytest.mark.skipif(
    not (_PENDF_H5.is_file() and _DECAY_DIR.is_dir()),
    reason="requires the real PENDF HDF5 library and decay_2020 sublibrary")

# Small chain around the Am/In/Fe/W neighborhoods plus decay-demanded targets.
_NUCLIDES = [
    "Am241", "Am242", "Am242_m1",
    "In115", "In116", "In116_m1", "In116_m2",
    "Fe56", "Fe57",
    "W186", "W187",
    "Np237", "Sn116", "Re187", "Cm242",  # decay daughters
]


@pytest.fixture(scope="module")
def small_chain():
    return chain_from_pendf(_PENDF_H5, _DECAY_DIR, nuclides=_NUCLIDES)


def _reactions(chain, parent):
    """Map reaction type -> (target, branching_ratio) for a parent nuclide."""
    return {rx.type: (rx.target, rx.branching_ratio)
            for rx in chain[parent].reactions}


def test_am241_ground_and_metastable(small_chain):
    rxns = _reactions(small_chain, "Am241")
    assert rxns["(n,gamma)"] == ("Am242", 1.0)
    assert rxns["(n,gamma)_m1"] == ("Am242_m1", 1.0)


def test_in115_three_capture_channels(small_chain):
    rxns = _reactions(small_chain, "In115")
    assert rxns["(n,gamma)"] == ("In116", 1.0)
    assert rxns["(n,gamma)_m1"] == ("In116_m1", 1.0)
    assert rxns["(n,gamma)_m2"] == ("In116_m2", 1.0)


def test_fe56_ground_only(small_chain):
    rxns = _reactions(small_chain, "Fe56")
    assert rxns["(n,gamma)"] == ("Fe57", 1.0)
    assert not any(t.startswith("(n,gamma)_m") for t in rxns)


def test_all_branching_ratios_unity(small_chain):
    for parent in ("Am241", "In115", "Fe56", "W186"):
        for rx in small_chain[parent].reactions:
            assert rx.branching_ratio == 1.0


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
