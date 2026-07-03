"""Tests for openmc.data.pendf against real TENDL-2017 PENDF fixtures."""

import io
from pathlib import Path

import h5py
import numpy as np
import pytest

import openmc.data
from openmc.data.pendf import PendfLibrary
from openmc.data.endf import Evaluation, get_head_record, get_tab1_record

_PENDF_DIR = Path("/home/perry/NukeData/Activation/PENDF/Point_TENDL2017")

_FIXTURES = {
    "Fe56": "n-Fe056.pendf",
    "W186": "n-W186.pendf",
    "Am241": "n-Am241.pendf",
    "In115": "n-In115.pendf",
}

# Decay library used for ELIS-based product mapping. FISPACT-style directory of
# per-nuclide ENDF decay files (ground/m1/m2 as separate materials).
_DECAY_DIR = Path("/home/perry/NukeData/Activation/DecayData/decay_2020")

# Decay files providing product-nuclide level energies for the Am241/In115 bake.
_DECAY_FILES = ["Am242", "Am242m", "Am242n", "In116", "In116m", "In116n"]

pytestmark = pytest.mark.skipif(
    not _PENDF_DIR.is_dir(),
    reason=f"TENDL PENDF data not available at {_PENDF_DIR}",
)


def _mf3(ev, mt):
    """Parse an MF=3 section into (QM, QI, Tabulated1D)."""
    fo = io.StringIO(ev.section[3, mt])
    get_head_record(fo)
    (qm, qi, _l1, _lr), tab = get_tab1_record(fo)
    return qm, qi, tab


def _mf10(ev, mt):
    """Parse an MF=10 section into (NS, [(QM, QI, IZAP, LFS, Tabulated1D), ...])."""
    fo = io.StringIO(ev.section[10, mt])
    _, _, _lis, _liso, ns, _ = get_head_record(fo)
    subs = []
    for _ in range(ns):
        (qm, qi, izap, lfs), tab = get_tab1_record(fo)
        subs.append((qm, qi, izap, lfs, tab))
    return ns, subs


@pytest.fixture(scope="module")
def evaluations():
    return {name: Evaluation(_PENDF_DIR / fn) for name, fn in _FIXTURES.items()}


def test_identity(evaluations):
    # Header-derived GNDS names and isomeric states
    assert evaluations["Fe56"].gnds_name == "Fe56"
    assert evaluations["W186"].gnds_name == "W186"
    assert evaluations["Am241"].gnds_name == "Am241"
    assert evaluations["In115"].gnds_name == "In115"
    for ev in evaluations.values():
        assert ev.target["isomeric_state"] == 0
        assert abs(ev.target["temperature"] - 293.16) < 0.1


def test_am241_mf10(evaluations):
    ev = evaluations["Am241"]
    mf10 = sorted(mt for (mf, mt) in ev.section if mf == 10)
    assert mf10 == [24, 102, 108, 111]

    # MT=102: MF=3 has 16157 points; MF=10 has 2 partials on the same grid
    qm, qi, tab = _mf3(ev, 102)
    assert len(tab.x) == 16157
    ns, subs = _mf10(ev, 102)
    assert ns == 2
    assert [s[3] for s in subs] == [0, 2]          # LFS values (0 skips to 2)
    for pqm, pqi, izap, lfs, ptab in subs:
        assert len(ptab.x) == 16157
        assert izap == 95242
    # ELFS == QM - QI for each partial (LFS=0 is ground, LFS=2 excited)
    assert (subs[0][0] - subs[0][1]) == pytest.approx(0.0)
    assert (subs[1][0] - subs[1][1]) == pytest.approx(48600.0)


def test_in115_mf10(evaluations):
    ev = evaluations["In115"]
    qm, qi, tab = _mf3(ev, 102)
    assert len(tab.x) == 29572
    ns, subs = _mf10(ev, 102)
    assert ns == 3
    assert [s[3] for s in subs] == [0, 1, 4]
    for pqm, pqi, izap, lfs, ptab in subs:
        assert len(ptab.x) == 29572
        assert izap == 49116


def test_mf3_all_lin_lin(evaluations):
    # Every MF=3 TAB1 must be single-region lin-lin (NR=1, INT=2)
    for ev in evaluations.values():
        for (mf, mt) in ev.section:
            if mf != 3:
                continue
            _qm, _qi, tab = _mf3(ev, mt)
            assert len(tab.breakpoints) == 1
            assert int(tab.interpolation[0]) == 2


def test_basic_xs_sanity(evaluations):
    # W186/Fe56 MT lists non-empty; energies non-decreasing; xs finite >= 0.
    # Grids are non-decreasing (not strictly increasing): TENDL repeats the
    # 30 MeV point as an ENDF-6 discontinuity at the TALYS evaluation boundary.
    for name in ("Fe56", "W186"):
        ev = evaluations[name]
        mts = [mt for (mf, mt) in ev.section if mf == 3]
        assert mts
        for mt in mts:
            _qm, _qi, tab = _mf3(ev, mt)
            assert np.all(np.diff(tab.x) >= 0)
            assert np.all(np.isfinite(tab.y))
            assert np.all(tab.y >= 0)


def test_roundtrip(tmp_path, evaluations):
    # Symlink the 4 fixtures into an isolated directory and preprocess
    src = tmp_path / "pendf"
    src.mkdir()
    for fn in _FIXTURES.values():
        (src / fn).symlink_to(_PENDF_DIR / fn)

    out = tmp_path / "tendl.h5"
    lib = PendfLibrary.from_endf_directory(
        src, out, library="TENDL-2017", temperature=293.16)

    # Library-level metadata
    assert sorted(lib.nuclides) == ["Am241", "Fe56", "In115", "W186"]
    assert lib.library == "TENDL-2017"
    assert lib.temperature == pytest.approx(293.16)
    assert lib.mapping == "none"

    # Reopen from disk to exercise the reader path
    reader = PendfLibrary(out)
    try:
        assert sorted(reader.nuclides) == ["Am241", "Fe56", "In115", "W186"]

        # MF=3 arrays exactly equal the direct parse
        _qm, _qi, tab = _mf3(evaluations["Am241"], 102)
        energy, xs = reader.xs("Am241", 102)
        np.testing.assert_array_equal(energy, tab.x)
        np.testing.assert_array_equal(xs, tab.y)
        assert reader._reaction("Am241", 102).attrs["QM"] == pytest.approx(_qm)

        # MF=10 partials present with correct LFS values and ELFS = QM - QI
        assert reader.pathways("Am241", 102) == [0, 2]
        _ns, subs = _mf10(evaluations["Am241"], 102)
        for pqm, pqi, izap, lfs, ptab in subs:
            penergy, pxs = reader.pathway_xs("Am241", 102, lfs)
            np.testing.assert_array_equal(penergy, ptab.x)
            np.testing.assert_array_equal(pxs, ptab.y)
            grp = reader._reaction("Am241", 102)[f"LFS{lfs}"]
            assert grp.attrs["ELFS"] == pytest.approx(pqm - pqi)
            assert grp.attrs["IZAP"] == izap
            # mapping='none' -> no baked product name
            assert reader.product("Am241", 102, lfs) is None

        assert reader.pathways("In115", 102) == [0, 1, 4]

        # Nuclide-level attrs correct
        nuc = reader._nuclide("Am241")
        assert nuc.attrs["ZA"] == 95241
        assert nuc.attrs["LISO"] == 0

        # Extra (non-activation) MTs excluded by default
        assert not (set(reader.reactions("Am241")) &
                    {251, 252, 253, 301, 444})
    finally:
        reader.close()


@pytest.mark.skipif(
    not _DECAY_DIR.is_dir(),
    reason=f"Decay data not available at {_DECAY_DIR}",
)
def test_elis_mapping_bake(tmp_path):
    # Isolate the two parents that have MF=10 isomeric production, and a minimal
    # decay directory holding just their product states.
    src = tmp_path / "pendf"
    src.mkdir()
    for name in ("Am241", "In115"):
        fn = _FIXTURES[name]
        (src / fn).symlink_to(_PENDF_DIR / fn)

    dk = tmp_path / "decay"
    dk.mkdir()
    for fn in _DECAY_FILES:
        (dk / fn).symlink_to(_DECAY_DIR / fn)

    out = tmp_path / "tendl_elis.h5"
    lib = PendfLibrary.from_endf_directory(
        src, out, library="TENDL-2017", temperature=293.16,
        mapping="elis", decay_file=dk)

    # Root-level provenance attrs (frozen schema §4.1)
    assert lib.mapping == "elis"
    with h5py.File(out, "r") as f:
        assert f.attrs["mapping"].decode() == "elis"
        assert f.attrs["decay_file"].decode() == str(dk)
        assert f.attrs["elis_rtol"] == pytest.approx(0.50)
        assert f.attrs["elis_atol"] == pytest.approx(0.0)

    reader = PendfLibrary(out)
    try:
        assert reader.mapping == "elis"

        # Am241(n,gamma): LFS 0 -> ground Am242, LFS 2 -> Am242_m1 (ELFS ~48.6 keV,
        # NOT LISO 2 -- LFS is a level index, LISO comes from ELIS matching).
        assert reader.pathways("Am241", 102) == [0, 2]
        assert reader.product("Am241", 102, 0) == "Am242"
        assert reader.product("Am241", 102, 2) == "Am242_m1"

        # In115(n,gamma): LFS 0/1/4 -> In116 / In116_m1 (~127.3 keV) / In116_m2
        # (~289.7 keV). In116_m2 is present in the decay source, so LFS 4 maps.
        assert reader.pathways("In115", 102) == [0, 1, 4]
        assert reader.product("In115", 102, 0) == "In116"
        assert reader.product("In115", 102, 1) == "In116_m1"
        assert reader.product("In115", 102, 4) == "In116_m2"

        # Baked product name lives on the LFS group; raw MF=10 attrs still present
        grp = reader._reaction("In115", 102)["LFS4"]
        assert grp.attrs["product"].decode() == "In116_m2"
        assert grp.attrs["IZAP"] == 49116
        assert grp.attrs["LFS"] == 4
        assert grp.attrs["ELFS"] == pytest.approx(289660.0)
    finally:
        reader.close()


def _write_garbage(path):
    """Write a non-ENDF file (TENDL-named) that fails to parse as an evaluation."""
    path.write_text("this is not an ENDF tape\n" * 4)


def test_skips_unparseable_file(tmp_path):
    # One good tape plus one garbage tape: the build succeeds, warns about the
    # skipped file, and the library holds only the good nuclide.
    src = tmp_path / "pendf"
    src.mkdir()
    (src / _FIXTURES["Fe56"]).symlink_to(_PENDF_DIR / _FIXTURES["Fe56"])
    _write_garbage(src / "n-Xx999.pendf")

    out = tmp_path / "tendl.h5"
    with pytest.warns(UserWarning, match="skipping"):
        lib = PendfLibrary.from_endf_directory(
            src, out, library="TENDL-2017", temperature=293.16)

    assert lib.nuclides == ["Fe56"]
    assert out.is_file()
    assert not list(tmp_path.glob("*.tmp")) and not list(src.glob("*.tmp"))

    # Output is a valid library: root attrs present and the reader reopens it.
    with h5py.File(out, "r") as f:
        assert f.attrs["format_version"] == 1
        assert f.attrs["library"].decode() == "TENDL-2017"
        assert f.attrs["temperature"] == pytest.approx(293.16)
    reader = PendfLibrary(out)
    try:
        assert reader.nuclides == ["Fe56"]
        assert reader.reactions("Fe56")
    finally:
        reader.close()


def test_all_unparseable_raises(tmp_path):
    # Only garbage: no library is produced. The output file is not created and
    # no stray temporary file is left behind.
    src = tmp_path / "pendf"
    src.mkdir()
    _write_garbage(src / "n-Xx998.pendf")
    _write_garbage(src / "n-Xx999.pendf")

    out = tmp_path / "tendl.h5"
    with pytest.warns(UserWarning, match="skipping"):
        with pytest.raises(ValueError, match="could be converted"):
            PendfLibrary.from_endf_directory(src, out, temperature=293.16)

    assert not out.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_context_manager(tmp_path):
    # PendfLibrary supports the context-manager protocol and closes on exit.
    src = tmp_path / "pendf"
    src.mkdir()
    (src / _FIXTURES["Fe56"]).symlink_to(_PENDF_DIR / _FIXTURES["Fe56"])

    out = tmp_path / "tendl.h5"
    PendfLibrary.from_endf_directory(
        src, out, library="TENDL-2017", temperature=293.16)

    with PendfLibrary(out) as lib:
        assert lib.nuclides == ["Fe56"]
        assert lib._files
    assert lib._files == []
