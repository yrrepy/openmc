"""Tests for openmc.data.pendf against real TENDL-2017 PENDF fixtures."""

import io
import types
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

import openmc.data
from openmc.data import Tabulated1D
from openmc.data.pendf import (PendfLibrary, PendfTapeLibrary,
                               open_pendf_library, _check_lin_lin,
                               _discover_pendf_files, _write_mf10_partials,
                               _identity_from_evaluation, tape_identity,
                               _TENDL_RE)

_CHAIN_SIMPLE = Path(__file__).parents[1] / "chain_simple.xml"
from openmc.data.endf import Evaluation, get_head_record, get_tab1_record

_PENDF_DIR = Path("/home/perry/NukeData/Activation/PENDF/Point_TENDL2017/pendf")

_FIXTURES = {
    "Fe56": "n-Fe056.pendf",
    "W186": "n-W186.pendf",
    "Am241": "n-Am241.pendf",
    "In115": "n-In115.pendf",
}

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


def _endf_field(value):
    """Format a value as an 11-character ENDF-6 record field."""
    return f"{value:>11.4E}" if isinstance(value, float) else f"{value:>11d}"


def _mf10_section_text(subs, za=451230, awr=122.0):
    """Build a synthetic ENDF-6 MF=10 section from partial subsection data.

    Each entry of ``subs`` is ``(QM, QI, IZAP, LFS, [(energy, xs), ...])`` and is
    written as a single lin-lin (INT=2) TAB1 region, matching the NJOY PENDF
    single-region convention.
    """
    lines = [_endf_field(float(za)) + _endf_field(awr) + _endf_field(0)
             + _endf_field(0) + _endf_field(len(subs)) + _endf_field(0)]
    for qm, qi, izap, lfs, pairs in subs:
        lines.append(_endf_field(qm) + _endf_field(qi) + _endf_field(izap)
                     + _endf_field(lfs) + _endf_field(1) + _endf_field(len(pairs)))
        lines.append(_endf_field(len(pairs)) + _endf_field(2))   # NBT, INT=lin-lin
        row = ""
        for i, (x, y) in enumerate(pairs):
            row += _endf_field(float(x)) + _endf_field(float(y))
            if (i + 1) % 3 == 0:
                lines.append(row)
                row = ""
        if row:
            lines.append(row)
    return "\n".join(lines) + "\n"


def _fake_mf10_evaluation(mt, subs, **kwargs):
    """A stand-in evaluation exposing only a synthetic MF=10 section."""
    return types.SimpleNamespace(
        section={(10, mt): _mf10_section_text(subs, **kwargs)})


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


def test_check_lin_lin_multiregion_all_lin_lin_accepted():
    # A multi-region TAB1 in which every region is lin-lin (INT=2) is
    # equivalent to a single lin-lin table and must be accepted (not raise).
    tab = Tabulated1D([0, 5, 10], [1, 2, 3], breakpoints=[2, 3],
                      interpolation=[2, 2])
    _check_lin_lin("Test0", 3, 1, tab)


def test_check_lin_lin_mixed_int_rejected():
    # Any region with INT != 2 (here a lin-log region) is still rejected.
    tab = Tabulated1D([0, 5, 10], [1, 2, 3], breakpoints=[2, 3],
                      interpolation=[2, 1])
    with pytest.raises(ValueError, match="INT=2"):
        _check_lin_lin("Test0", 3, 1, tab)


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
    # De-baked (source-faithful) build writes no mapping root attr.
    assert lib.mapping is None

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
        assert reader.pathways("Am241", 102) == [(0, 95242), (2, 95242)]
        _ns, subs = _mf10(evaluations["Am241"], 102)
        for pqm, pqi, izap, lfs, ptab in subs:
            penergy, pxs = reader.pathway_xs("Am241", 102, lfs)
            np.testing.assert_array_equal(penergy, ptab.x)
            np.testing.assert_array_equal(pxs, ptab.y)
            grp = reader._reaction("Am241", 102)[f"LFS{lfs}"]
            assert grp.attrs["ELFS"] == pytest.approx(pqm - pqi)
            assert grp.attrs["IZAP"] == izap
            # Source-faithful build: no baked product name on the partial.
            assert "product" not in grp.attrs

        assert reader.pathways("In115", 102) == [(0, 49116), (1, 49116), (4, 49116)]

        # Nuclide-level attrs correct
        nuc = reader._nuclide("Am241")
        assert nuc.attrs["ZA"] == 95241
        assert nuc.attrs["LISO"] == 0

        # Extra (non-activation) MTs excluded by default
        assert not (set(reader.reactions("Am241")) &
                    {251, 252, 253, 301, 444})
    finally:
        reader.close()


def test_source_identity_stored_and_read(tmp_path):
    # from_endf_directory stamps the source directory's tape identity; reopening
    # reads it back, and it equals a direct tape_identity() of the source dir.
    src = tmp_path / "pendf"
    src.mkdir()
    for name in ("Fe56", "In115"):
        fn = _FIXTURES[name]
        (src / fn).symlink_to(_PENDF_DIR / fn)

    out = tmp_path / "tendl.h5"
    lib = PendfLibrary.from_endf_directory(
        src, out, library="TENDL-2017", temperature=293.16)
    try:
        expected = tape_identity(src)
        assert expected is not None
        assert lib.source_identity == expected
    finally:
        lib.close()

    reader = PendfLibrary(out)
    try:
        assert reader.source_identity == expected
    finally:
        reader.close()


def test_source_identity_none_on_old_file(tmp_path):
    # A file built without the source_identity attr reads back None.
    out = _mf10_library(tmp_path, "Xx100", _LUMPED_MT, _LUMPED_SUBS)
    with PendfLibrary(out) as lib:
        assert lib.source_identity is None


# ---------------------------------------------------------------------------
# tape_identity
# ---------------------------------------------------------------------------

def _write_tpid_tape(path, tpid_text):
    """Write a minimal tape whose first line is a TPID record (cols 0:66)."""
    path.write_text(f"{tpid_text:<66}   1 0  0    0\n 1.001000+3\n")
    return path


def test_identity_from_evaluation_format():
    ev = types.SimpleNamespace(
        info={"library": ("JEFF", 40, 0), "sublibrary": "Radioactive decay data"})
    assert _identity_from_evaluation(ev) == "JEFF-40 Radioactive decay data"
    # No sublibrary -> bare '<library>-<version>' identity.
    ev2 = types.SimpleNamespace(info={"library": ("ENDF/B", 8, 1)})
    assert _identity_from_evaluation(ev2) == "ENDF/B-8"
    # No library info at all -> None.
    assert _identity_from_evaluation(types.SimpleNamespace(info={})) is None


def test_tape_identity_tpid_text(tmp_path):
    tape = _write_tpid_tape(tmp_path / "tape", "JEFF-4.0 Incident Neutron File")
    assert tape_identity(tape) == "JEFF-4.0 Incident Neutron File"


def test_tape_identity_blank_tpid_falls_back_to_451(tmp_path, monkeypatch):
    tape = _write_tpid_tape(tmp_path / "decay_tape", "")   # blank TPID
    fake_ev = types.SimpleNamespace(
        info={"library": ("JEFF", 40, 0), "sublibrary": "Radioactive decay data"})
    monkeypatch.setattr("openmc.data.pendf.Evaluation", lambda p: fake_ev)
    assert tape_identity(tape) == "JEFF-40 Radioactive decay data"


def test_tape_identity_unreadable_returns_none(tmp_path):
    # A path that does not exist -> None (never raises).
    assert tape_identity(tmp_path / "does_not_exist") is None


def test_tape_identity_directory_samples_first(tmp_path, recwarn):
    d = tmp_path / "tapes"
    d.mkdir()
    for i in range(4):
        _write_tpid_tape(d / f"t{i}", "SAME-LIBRARY Neutron File")
    assert tape_identity(d) == "SAME-LIBRARY Neutron File"
    assert len(recwarn) == 0


def test_tape_identity_directory_disagreement_warns(tmp_path):
    d = tmp_path / "tapes"
    d.mkdir()
    _write_tpid_tape(d / "a", "LIB-A Neutron File")
    _write_tpid_tape(d / "b", "LIB-B Neutron File")
    with pytest.warns(UserWarning, match="different identities"):
        ident = tape_identity(d)
    assert ident == "LIB-A Neutron File"   # first sorted file


def test_raw_mf10_attrs_not_baked(tmp_path):
    # Source-faithful build: the two parents that carry MF=10 isomeric production
    # are converted and the result bakes NO product/mapping provenance. The root
    # has none of the retired mapping/decay_file/elis_* attrs and an LFS subgroup
    # carries only the raw IZAP/LFS/QM/QI/ELFS attrs -- never a product name.
    src = tmp_path / "pendf"
    src.mkdir()
    for name in ("Am241", "In115"):
        fn = _FIXTURES[name]
        (src / fn).symlink_to(_PENDF_DIR / fn)

    out = tmp_path / "tendl_raw.h5"
    lib = PendfLibrary.from_endf_directory(
        src, out, library="TENDL-2017", temperature=293.16)
    try:
        # A de-baked build writes no mapping root attr.
        assert lib.mapping is None
    finally:
        lib.close()

    with h5py.File(out, "r") as f:
        # No baked-product provenance at the root.
        for key in ("mapping", "decay_file", "elis_rtol", "elis_atol"):
            assert key not in f.attrs

        # In115(n,gamma) LFS 4 -> In116_m2 (~289.7 keV). The partial keeps its raw
        # MF=10 attrs (IZAP 49116, LFS 4, ELFS ~289660 eV) but no baked product.
        grp = f["In115"]["MT102"]["LFS4"]
        assert "product" not in grp.attrs
        assert int(grp.attrs["IZAP"]) == 49116
        assert int(grp.attrs["LFS"]) == 4
        assert grp.attrs["ELFS"] == pytest.approx(289660.0)


def test_reader_ignores_baked_attrs_on_old_file(tmp_path):
    # Backward compatibility: an OLD baked file carried per-LFS ``product`` attrs
    # and a legacy ``mapping`` root attr. Synthesize one with ``_mf10_library``
    # (which stamps both via h5py) and prove the reader opens it cleanly and
    # neither requires nor chokes on the baked attrs: pathways()/pathway_xs()
    # still work, the legacy ``mapping`` root attr is read back informationally,
    # and the stamped ``product`` attr survives untouched at the h5py level.
    out = _mf10_library(tmp_path, "Xx100", _LUMPED_MT, _LUMPED_SUBS,
                        products=_LUMPED_PRODUCTS)
    with PendfLibrary(out) as lib:
        # Legacy ``mapping`` root attr is read back (informational only).
        assert lib.mapping == "none"
        assert lib.pathways("Xx100", _LUMPED_MT) == [(0, 47108), (0, 48108)]
        _e, xs = lib.pathway_xs("Xx100", _LUMPED_MT, 0, izap=47108)
        np.testing.assert_array_equal(xs, [2.0, 4.0])

    # The baked product attrs remain on disk (the reader neither strips nor needs
    # them); only reachable at the h5py level now that product() is gone.
    with h5py.File(out, "r") as f:
        g = f["Xx100"]["MT5"]
        stamped = {int(g[k].attrs["IZAP"]): g[k].attrs["product"].decode()
                   for k in g if k.startswith("LFS")}
        assert stamped == {47108: "Ag108", 48108: "Cd108"}


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
        assert f.attrs["format_version"] == 2
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


def test_format_version_written(tmp_path):
    # A freshly written library stamps the current format version (2).
    src = tmp_path / "pendf"
    src.mkdir()
    (src / _FIXTURES["Fe56"]).symlink_to(_PENDF_DIR / _FIXTURES["Fe56"])

    out = tmp_path / "tendl.h5"
    PendfLibrary.from_endf_directory(
        src, out, library="TENDL-2017", temperature=293.16).close()

    with h5py.File(out, "r") as f:
        assert f.attrs["format_version"] == 2


def test_format_version_newer_rejected(tmp_path):
    # A file stamped a newer format than this reader supports is refused, with a
    # message naming the file, the found version, and the supported version.
    src = tmp_path / "pendf"
    src.mkdir()
    (src / _FIXTURES["Fe56"]).symlink_to(_PENDF_DIR / _FIXTURES["Fe56"])

    out = tmp_path / "tendl.h5"
    PendfLibrary.from_endf_directory(
        src, out, library="TENDL-2017", temperature=293.16).close()

    with h5py.File(out, "r+") as f:
        f.attrs["format_version"] = 99
    with pytest.raises(ValueError, match="newer than the supported version"):
        PendfLibrary(out)


def test_format_version_missing_loads(tmp_path):
    # An old file predating the version attr (attr deleted) still loads.
    src = tmp_path / "pendf"
    src.mkdir()
    (src / _FIXTURES["Fe56"]).symlink_to(_PENDF_DIR / _FIXTURES["Fe56"])

    out = tmp_path / "tendl.h5"
    PendfLibrary.from_endf_directory(
        src, out, library="TENDL-2017", temperature=293.16).close()

    with h5py.File(out, "r+") as f:
        del f.attrs["format_version"]
    with PendfLibrary(out) as lib:
        assert lib.nuclides == ["Fe56"]


def test_mf10_shared_lfs_distinct_izap(tmp_path):
    # A lumped reaction (e.g. TENDL MT=5) can carry several MF=10 partials that
    # share an LFS but describe different product nuclides (distinct IZAP). All
    # must be kept, disambiguated by IZAP in the subgroup name, and no duplicate
    # warning may fire.
    mt = 5
    subs = [
        (0.0, 0.0, 451230, 0, [(1.0, 2.0), (3.0, 4.0)]),
        (5.0, -3.0, 461230, 0, [(1.0, 0.5), (3.0, 0.25)]),
    ]
    ev = _fake_mf10_evaluation(mt, subs)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        with h5py.File(tmp_path / "t.h5", "w") as f:
            mtg = f.create_group(f"MT{mt}")
            _write_mf10_partials(mtg, ev, mt, "Xx100", Path("n-Xx100.pendf"))

            # Both partials stored under IZAP-disambiguated names.
            assert sorted(mtg.keys()) == ["LFS0_ZAP451230", "LFS0_ZAP461230"]
            g1 = mtg["LFS0_ZAP451230"]
            g2 = mtg["LFS0_ZAP461230"]
            assert (int(g1.attrs["IZAP"]), int(g1.attrs["LFS"])) == (451230, 0)
            assert (int(g2.attrs["IZAP"]), int(g2.attrs["LFS"])) == (461230, 0)
            # Each subgroup keeps its own cross section (no data lost/overwritten).
            np.testing.assert_array_equal(g1["xs"][()], [2.0, 4.0])
            np.testing.assert_array_equal(g2["xs"][()], [0.5, 0.25])

    assert not records          # legitimate distinct-IZAP data must not warn


def test_mf10_true_izap_lfs_duplicate(tmp_path):
    # A genuine (IZAP, LFS) duplicate: keep the first, warn about the (IZAP, LFS)
    # duplicate, and skip the rest.
    mt = 5
    subs = [
        (0.0, 0.0, 451230, 0, [(1.0, 2.0)]),
        (0.0, 0.0, 451230, 0, [(1.0, 9.0)]),   # same (IZAP, LFS): true duplicate
    ]
    ev = _fake_mf10_evaluation(mt, subs)

    with h5py.File(tmp_path / "t.h5", "w") as f:
        mtg = f.create_group(f"MT{mt}")
        with pytest.warns(UserWarning,
                          match=r"duplicate MF=10 partial \(IZAP=451230, LFS=0\)"):
            _write_mf10_partials(mtg, ev, mt, "Xx100", Path("n-Xx100.pendf"))

        # Only the first partial survives, under the plain LFS name (a duplicate
        # must not make the survivor's LFS look shared).
        assert sorted(mtg.keys()) == ["LFS0"]
        np.testing.assert_array_equal(mtg["LFS0"]["xs"][()], [2.0])


def test_mf10_unique_lfs_schema_compatible(tmp_path):
    # Schema-compatibility guard: a normal reaction whose partials all have
    # unique LFS (single product nuclide) still lands under the bare LFS{lfs}
    # names with no IZAP suffix and no warning.
    mt = 102
    subs = [
        (0.0, 0.0, 95242, 0, [(1.0, 2.0)]),
        (48600.0, 0.0, 95242, 2, [(1.0, 3.0)]),
    ]
    ev = _fake_mf10_evaluation(mt, subs, za=95242)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        with h5py.File(tmp_path / "t.h5", "w") as f:
            mtg = f.create_group(f"MT{mt}")
            _write_mf10_partials(mtg, ev, mt, "Am241", Path("n-Am241.pendf"))

            assert sorted(mtg.keys()) == ["LFS0", "LFS2"]
            assert int(mtg["LFS2"].attrs["LFS"]) == 2
            assert int(mtg["LFS2"].attrs["IZAP"]) == 95242
            assert mtg["LFS2"].attrs["ELFS"] == pytest.approx(48600.0)

    assert not records


# ---------------------------------------------------------------------------
# Read-back tests for the (lfs, izap) partial accessors. A lumped reaction can
# store several MF=10 partials that share an LFS but describe different product
# nuclides; pathways() must enumerate each, and pathway_xs() must be able to
# pick one by IZAP. Fixtures are minimal PendfLibrary-openable files built
# through the real _write_mf10_partials writer.


def _mf10_library(tmp_path, nuclide, mt, subs, mapping="none", products=None):
    """Write a minimal PendfLibrary-openable .h5 with one MF=10 reaction.

    ``subs`` (the ``_fake_mf10_evaluation`` format) is written through the real
    ``_write_mf10_partials`` under a single nuclide/MT group. To synthesize an
    OLD baked file (so the reader's tolerance of retired attrs can be exercised),
    ``mapping`` is written as the legacy ``mapping`` root attr and ``products``,
    when given, maps IZAP -> product name and manually stamps a ``product`` attr
    onto each partial via h5py -- neither is produced by the current writer.
    Returns the file path.
    """
    ev = _fake_mf10_evaluation(mt, subs)
    out = tmp_path / f"{nuclide}.h5"
    with h5py.File(out, "w") as f:
        f.attrs["format_version"] = 2
        f.attrs["library"] = np.bytes_("TENDL-2017")
        f.attrs["temperature"] = 293.16
        f.attrs["mapping"] = np.bytes_(mapping)
        mtg = f.create_group(nuclide).create_group(f"MT{mt}")
        _write_mf10_partials(mtg, ev, mt, nuclide, Path(f"n-{nuclide}.pendf"))
        if products:
            for k in mtg:
                if k.startswith("LFS"):
                    mtg[k].attrs["product"] = np.bytes_(
                        products[int(mtg[k].attrs["IZAP"])])
    return out


# A lumped reaction (MT=5): two partials share LFS=0 but produce Ag108 (IZAP
# 47108) and Cd108 (IZAP 48108), each with its own cross section.
_LUMPED_MT = 5
_LUMPED_SUBS = [
    (0.0, 0.0, 47108, 0, [(1.0, 2.0), (3.0, 4.0)]),
    (5.0, -3.0, 48108, 0, [(1.0, 0.5), (3.0, 0.25)]),
]
_LUMPED_PRODUCTS = {47108: "Ag108", 48108: "Cd108"}


def test_pathways_lumped_repeats_lfs(tmp_path):
    # (a) On a lumped reaction, pathways() returns one (lfs, izap) pair per
    # stored partial -- the shared LFS repeats with distinct IZAP -- sorted.
    out = _mf10_library(tmp_path, "Xx100", _LUMPED_MT, _LUMPED_SUBS)
    with PendfLibrary(out) as lib:
        assert lib.pathways("Xx100", _LUMPED_MT) == [(0, 47108), (0, 48108)]


def test_pathway_lumped_izap_selects_partial(tmp_path):
    # (b) With izap=, pathway_xs() resolves the shared LFS to the one matching
    # partial and returns its cross section.
    out = _mf10_library(tmp_path, "Xx100", _LUMPED_MT, _LUMPED_SUBS,
                        products=_LUMPED_PRODUCTS)
    with PendfLibrary(out) as lib:
        _e, xs = lib.pathway_xs("Xx100", _LUMPED_MT, 0, izap=47108)
        np.testing.assert_array_equal(xs, [2.0, 4.0])
        _e, xs = lib.pathway_xs("Xx100", _LUMPED_MT, 0, izap=48108)
        np.testing.assert_array_equal(xs, [0.5, 0.25])
        # An LFS/IZAP pair that is not stored still raises, naming the izap.
        with pytest.raises(KeyError, match="LFS=0, IZAP=999"):
            lib.pathway_xs("Xx100", _LUMPED_MT, 0, izap=999)


def test_pathway_lumped_ambiguous_without_izap(tmp_path):
    # (c) Without izap=, the shared LFS is ambiguous: raise ValueError naming
    # both candidate IZAPs and pointing the caller at izap=.
    out = _mf10_library(tmp_path, "Xx100", _LUMPED_MT, _LUMPED_SUBS,
                        products=_LUMPED_PRODUCTS)
    with PendfLibrary(out) as lib:
        with pytest.raises(
                ValueError,
                match=r"shared by IZAP values.*47108.*48108.*izap="):
            lib.pathway_xs("Xx100", _LUMPED_MT, 0)


def test_pathway_non_lumped_bare_lfs_unchanged(tmp_path):
    # (d) A non-lumped reaction (each LFS unique): bare-LFS pathways()/
    # pathway_xs() behave exactly as before -- no izap needed -- and a missing
    # LFS raises the same KeyError.
    mt = 102
    subs = [
        (0.0, 0.0, 95242, 0, [(1.0, 2.0)]),
        (48600.0, 0.0, 95242, 2, [(1.0, 3.0)]),
    ]
    out = _mf10_library(tmp_path, "Am241", mt, subs)
    with PendfLibrary(out) as lib:
        assert lib.pathways("Am241", mt) == [(0, 95242), (2, 95242)]
        _e, xs = lib.pathway_xs("Am241", mt, 2)
        np.testing.assert_array_equal(xs, [3.0])
        with pytest.raises(KeyError, match="no MF=10 partial LFS=9"):
            lib.pathway_xs("Am241", mt, 9)


# ---------------------------------------------------------------------------
# Filename-recognition tests for _discover_pendf_files. These exercise only the
# regex/dispatch boundary of discovery: dummy empty files are touched into a
# tmp_path directory and the returned (path, implied-LISO) pairs are checked. No
# data is parsed, so no PENDF fixtures are needed (though the module skipif still
# gates them, matching the rest of the file).


def _discover_liso(tmp_path, names):
    """Touch ``names`` as empty files and map recognized name -> implied LISO."""
    scan = tmp_path / "scan"
    scan.mkdir()
    for name in names:
        (scan / name).touch()
    return {p.name: liso for p, liso in _discover_pendf_files(scan)}


def test_discover_tendl2015_reversed(tmp_path):
    # TENDL-2015 nuclide-first stem ``SymA[m|n]-n.pendf`` (reversed vs 2017).
    got = _discover_liso(tmp_path, ["Ag107-n.pendf", "Ac222m-n.pendf",
                                    "Ac214-n.pendf"])
    assert got == {"Ag107-n.pendf": 0, "Ac222m-n.pendf": 1, "Ac214-n.pendf": 0}


def test_discover_tendl2019_native(tmp_path):
    # TENDL-2019 native ``SymA[m|n]p.asc``; the trailing ``p`` is not an isomer.
    got = _discover_liso(tmp_path, ["Ag107p.asc", "Ac222mp.asc", "Am241p.asc"])
    assert got == {"Ag107p.asc": 0, "Ac222mp.asc": 1, "Am241p.asc": 0}


def test_discover_jeff33(tmp_path):
    # JEFF-3.3 ``Z-Sym-A(g|m|n).jeffNN.pendf`` with any two-digit version token.
    got = _discover_liso(tmp_path, [
        "47-Ag-107g.jeff33.pendf",     # ground (g -> LISO 0)
        "27-Co-58m.jeff33.pendf",      # m -> LISO 1
        "100-Fm-255g.jeff33.pendf",    # 3-digit Z
        "47-Ag-107g.jeff40.pendf",     # any two digits accepted
    ])
    assert got == {
        "47-Ag-107g.jeff33.pendf": 0,
        "27-Co-58m.jeff33.pendf": 1,
        "100-Fm-255g.jeff33.pendf": 0,
        "47-Ag-107g.jeff40.pendf": 0,
    }


def test_discover_jendl5(tmp_path):
    # JENDL-5 ``n_ZZZ-Sym-AAA[m<digit>]_<T>K.dat``; free-form temperature token
    # and ``m<digit>`` isomer index doubling as the implied LISO.
    got = _discover_liso(tmp_path, [
        "n_047-Ag-107_300K.dat",       # no isomer -> LISO 0
        "n_052-Te-123m1_300K.dat",     # m1 -> LISO 1
        "n_065-Tb-156m2_300K.dat",     # m2 -> LISO 2
        "n_047-Ag-107_293.6K.dat",     # non-integer temperature token
    ])
    assert got == {
        "n_047-Ag-107_300K.dat": 0,
        "n_052-Te-123m1_300K.dat": 1,
        "n_065-Tb-156m2_300K.dat": 2,
        "n_047-Ag-107_293.6K.dat": 0,
    }


def test_discover_tendl_infix(tmp_path):
    # Loose ``.tendl20NN``-infixed files for both TENDL stems (projectile-first
    # 2017 and nuclide-first 2015), including a metastable through the infix.
    got = _discover_liso(tmp_path, [
        "n-Ag107.tendl2017.pendf",     # 2017 stem + infix
        "Ag107-n.tendl2015.pendf",     # 2015 stem + infix
        "n-Ag110m.tendl2019.pendf",    # metastable m -> LISO 1
    ])
    assert got == {
        "n-Ag107.tendl2017.pendf": 0,
        "Ag107-n.tendl2015.pendf": 0,
        "n-Ag110m.tendl2019.pendf": 1,
    }


def test_discover_infix_not_half_matched_by_tendl2017(tmp_path):
    # The overlap guard: the frozen projectile-first TENDL-2017 pattern is fully
    # anchored, so a ``.tendl20NN``-infixed name is NOT (half-)matched by it; the
    # dedicated infix pattern must pick it up instead. A plain 2017 name in the
    # same directory is still recognized, proving both coexist.
    assert _TENDL_RE.match("n-Ag107.tendl2017.pendf") is None
    got = _discover_liso(tmp_path, ["n-Ag107.tendl2017.pendf", "n-Fe056.pendf"])
    assert got == {"n-Ag107.tendl2017.pendf": 0, "n-Fe056.pendf": 0}


def test_discover_negatives_and_near_misses(tmp_path):
    # Non-PENDF and near-miss names must be ignored entirely: a FLUKA-style
    # ``.z``, a stray ``.txt``, a ``.pendf.bak`` backup, and a truncated JENDL-5
    # name missing its temperature token.
    got = _discover_liso(tmp_path, [
        "Ag107.z",                     # FLUKA-ish
        "notes.txt",                   # random text file
        "Ag107-n.pendf.bak",           # backup near-miss (trailing .bak)
        "n-Ag107.tendl2017.pendf.bak", # infix backup near-miss
        "n_047-Ag-107.dat",            # JENDL-5 missing _<T>K token
        "ZA000001",                    # ENDF/B free-neutron placeholder (skipped)
    ])
    assert got == {}


# ---------------------------------------------------------------------------
# PendfTapeLibrary -- raw ASC-tape collapse adapter (cross-validation)
#
# The pointwise HDF5 is the production format; this adapter reads the source
# tapes directly so a collapse against a tape directory can be validated against
# a collapse against a pointwise HDF5 built from the same tapes (must be
# bit-identical). The tests below exercise the duck-typed interface, the lazy
# one-nuclide cache, provenance, path dispatch, and the equivalence.
# ---------------------------------------------------------------------------

def _tape_dir(tmp_path, names):
    """Symlink the named fixtures into an isolated tape directory."""
    src = tmp_path / "tapes"
    src.mkdir()
    for name in names:
        fn = _FIXTURES[name]
        (src / fn).symlink_to(_PENDF_DIR / fn)
    return src


def test_tape_adapter_interface(tmp_path, evaluations):
    # nuclides / reactions / xs / pathways / pathway_xs against known content.
    src = _tape_dir(tmp_path, ("Fe56", "In115"))
    with PendfTapeLibrary(src) as lib:
        assert sorted(lib.nuclides) == ["Fe56", "In115"]

        # MF=3 arrays exactly equal the direct parse (float64, no round-trip).
        _qm, _qi, tab = _mf3(evaluations["In115"], 102)
        energy, xs = lib.xs("In115", 102)
        np.testing.assert_array_equal(energy, tab.x)
        np.testing.assert_array_equal(xs, tab.y)

        # MF=10 pathways and partial arrays match the tape (In115(n,gamma) has
        # LFS {0,1,4}, all IZAP 49116).
        assert lib.pathways("In115", 102) == [(0, 49116), (1, 49116), (4, 49116)]
        _ns, subs = _mf10(evaluations["In115"], 102)
        for pqm, pqi, izap, lfs, ptab in subs:
            penergy, pxs = lib.pathway_xs("In115", 102, lfs)
            np.testing.assert_array_equal(penergy, ptab.x)
            np.testing.assert_array_equal(pxs, ptab.y)

        # Extra (non-activation) MTs excluded by default, exactly as the h5 build.
        assert not (set(lib.reactions("In115")) &
                    {251, 252, 253, 301, 444})

        # Reader parity fields for the collapse / stamp check.
        assert lib.mapping is None
        assert lib.is_tape_source is True
        assert lib.has_ptables("In115") is False


def test_tape_adapter_matches_h5_data(tmp_path):
    # Every reaction, pathway, and cross section array is identical to the
    # pointwise HDF5 built from the same tapes -- the raw-data guarantee the
    # bit-identical collapse rests on.
    src = _tape_dir(tmp_path, ("Fe56", "In115"))
    out = tmp_path / "tendl.h5"
    h5lib = PendfLibrary.from_endf_directory(
        src, out, library="TENDL-2017", temperature=293.16)
    tapelib = PendfTapeLibrary(src)
    try:
        assert sorted(tapelib.nuclides) == sorted(h5lib.nuclides)
        for nuc in h5lib.nuclides:
            assert tapelib.reactions(nuc) == h5lib.reactions(nuc)
            for mt in h5lib.reactions(nuc):
                he, hx = h5lib.xs(nuc, mt)
                te, tx = tapelib.xs(nuc, mt)
                np.testing.assert_array_equal(he, te)
                np.testing.assert_array_equal(hx, tx)
                assert tapelib.pathways(nuc, mt) == h5lib.pathways(nuc, mt)
                for lfs, izap in h5lib.pathways(nuc, mt):
                    hpe, hpx = h5lib.pathway_xs(nuc, mt, lfs, izap)
                    tpe, tpx = tapelib.pathway_xs(nuc, mt, lfs, izap)
                    np.testing.assert_array_equal(hpe, tpe)
                    np.testing.assert_array_equal(hpx, tpx)
    finally:
        h5lib.close()


def test_tape_adapter_lazy_one_nuclide_cache(tmp_path, monkeypatch):
    # The constructor scans headers; pointwise arrays are parsed lazily and
    # cached one nuclide at a time. A parse counter on Evaluation (installed
    # after construction) proves the second access to a nuclide does not
    # re-parse, and that touching another nuclide evicts the cache.
    import openmc.data.pendf as pmod

    src = _tape_dir(tmp_path, ("Fe56", "In115"))
    lib = PendfTapeLibrary(src)          # header scan happens here (uncounted)

    calls = {"n": 0}
    real_eval = pmod.Evaluation

    def counting_eval(path, *args, **kwargs):
        calls["n"] += 1
        return real_eval(path, *args, **kwargs)

    monkeypatch.setattr(pmod, "Evaluation", counting_eval)

    lib.xs("In115", 102)                 # first access -> 1 parse
    assert calls["n"] == 1
    lib.pathways("In115", 102)           # same nuclide -> cache hit
    lib.reactions("In115")               # same nuclide -> cache hit
    assert calls["n"] == 1
    lib.xs("Fe56", 102)                  # different nuclide -> evict + 1 parse
    assert calls["n"] == 2
    lib.xs("In115", 102)                 # back to In115 -> re-parse
    assert calls["n"] == 3


def test_tape_adapter_source_identity(tmp_path):
    # Provenance: the adapter's source_identity equals a direct tape_identity()
    # of the directory (the value an h5 built from these tapes also stamps).
    src = _tape_dir(tmp_path, ("Fe56", "In115"))
    with PendfTapeLibrary(src) as lib:
        assert lib.source_identity == tape_identity(src)


def test_tape_adapter_chain_stamp_verifies_silently(tmp_path):
    # A chain stamped from these tapes (pendf_library == tape_identity, matching
    # nuclide count) verifies silently against the adapter -- the stamp compares
    # against the adapter's tape-derived source_identity.
    from openmc.deplete.chain import Chain
    from openmc.deplete.microxs import _verify_pendf_chain_stamp

    src = _tape_dir(tmp_path, ("Fe56", "In115"))
    with PendfTapeLibrary(src) as lib:
        stamp = tape_identity(src)       # computed outside the no-warn block
        chain = Chain()
        chain.root_attrs = {'pendf_library': stamp,
                            'pendf_nuclides': len(lib.nuclides)}
        with warnings.catch_warnings():
            warnings.simplefilter("error")     # any warning fails the test
            _verify_pendf_chain_stamp(chain, lib)


def test_tape_adapter_bogus_and_empty_paths(tmp_path):
    # Helpful errors: a nonexistent path and an empty directory.
    with pytest.raises(FileNotFoundError):
        PendfTapeLibrary(tmp_path / "does_not_exist")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No recognizable PENDF tapes"):
        PendfTapeLibrary(empty)


def test_open_pendf_library_routes_tape_directory(tmp_path):
    # Dispatch: a directory of ASC tapes routes to the tape adapter.
    src = _tape_dir(tmp_path, ("Fe56",))
    lib = open_pendf_library(src)
    try:
        assert isinstance(lib, PendfTapeLibrary)
        assert lib.nuclides == ["Fe56"]
    finally:
        lib.close()


def test_open_pendf_library_routes_pointwise_h5(tmp_path):
    # Dispatch: a pointwise .h5 file routes to PendfLibrary (not the tape adapter).
    src = _tape_dir(tmp_path, ("Fe56",))
    out = tmp_path / "tendl.h5"
    PendfLibrary.from_endf_directory(
        src, out, library="TENDL-2017", temperature=293.16).close()
    lib = open_pendf_library(out)
    try:
        assert isinstance(lib, PendfLibrary)
    finally:
        lib.close()


def test_open_pendf_library_helpful_errors(tmp_path):
    # Dispatch: nonexistent path, empty directory, and a non-HDF5 single file.
    with pytest.raises(FileNotFoundError, match="does not exist"):
        open_pendf_library(tmp_path / "nope")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no .h5 libraries and no"):
        open_pendf_library(empty)
    junk = tmp_path / "junk.h5"
    junk.write_text("not hdf5")
    with pytest.raises(ValueError, match="not an HDF5 PENDF library"):
        open_pendf_library(junk)


def _chain_from_library_pathways(lib, base_names):
    """Chain demanding exactly the library's MF=10 pathways, one distinct product
    per (mt, lfs) so no reaction falls back to the MF=3 total. The SAME chain
    drives both the h5 and the tape collapse, so any expansion difference would
    come from the library data alone.
    """
    from openmc.deplete.chain import Chain
    from openmc.deplete.nuclide import Nuclide
    from openmc.data import gnds_name, ATOMIC_SYMBOL

    chain = Chain()
    for nuc in lib.nuclides:
        nuclide = Nuclide(nuc)
        for mt, base in base_names.items():
            if mt not in lib.reactions(nuc):
                continue
            metas = 0
            for lfs, izap in lib.pathways(nuc, mt):
                z, a = izap // 1000, izap % 1000
                if z not in ATOMIC_SYMBOL:
                    continue
                if lfs == 0:
                    liso = 0
                else:
                    metas += 1
                    liso = metas
                rtype = base if liso == 0 else f"{base}_m{liso}"
                nuclide.add_reaction(rtype, gnds_name(z, a, liso), 0.0, 1.0,
                                     pendf_lfs=lfs)
        chain.add_nuclide(nuclide)
    return chain


def test_tape_vs_h5_collapse_bit_identical(tmp_path):
    # THE cross-validation: collapsing from the ASC tape directory gives a
    # bit-identical MicroXS to collapsing from a pointwise h5 built from the same
    # tapes, driven by the same chain (In115 carries MF=10 metastable pathways).
    from openmc.deplete.microxs import _build_xs_table_pendf

    src = _tape_dir(tmp_path, ("Fe56", "In115"))
    out = tmp_path / "tendl.h5"
    h5lib = PendfLibrary.from_endf_directory(
        src, out, library="TENDL-2017", temperature=293.16)
    tapelib = PendfTapeLibrary(src)

    # Non-lumped activation reactions; In115(n,gamma) expands into ground + m1/m2.
    base_names = {102: "(n,gamma)", 16: "(n,2n)", 17: "(n,3n)"}
    chain = _chain_from_library_pathways(tapelib, base_names)

    edges = np.array([1e-5, 1e-3, 1e-1, 1e1, 1e3, 1e5, 1e6, 5e6, 1e7,
                      1.5e7, 2e7])
    nuclides = sorted(tapelib.nuclides)
    reactions = list(dict.fromkeys(base_names.values()))

    try:
        t_h5 = _build_xs_table_pendf(nuclides, reactions, edges, h5lib, chain)
        t_tp = _build_xs_table_pendf(nuclides, reactions, edges, tapelib, chain)

        assert t_h5.reactions == t_tp.reactions
        assert t_h5.nuc_indices.tolist() == t_tp.nuc_indices.tolist()
        assert t_h5.rxn_indices.tolist() == t_tp.rxn_indices.tolist()
        assert t_h5.xs_matrix.shape == t_tp.xs_matrix.shape
        # EXACTLY equal, not merely close.
        np.testing.assert_array_equal(t_h5.xs_matrix, t_tp.xs_matrix)
        # Isomeric expansion did happen (guards against a trivial all-total pass).
        assert "(n,gamma)_m1" in t_tp.reactions
    finally:
        h5lib.close()


def test_from_multigroup_flux_tape_path_equals_object(tmp_path):
    # Dispatch through the collapse entry point: passing the tape DIRECTORY path
    # as pendf_library routes to the adapter and gives the same MicroXS as
    # passing a constructed adapter object.
    from openmc.deplete.microxs import MicroXS

    src = _tape_dir(tmp_path, ("Fe56",))
    kw = dict(energies=[0.0, 1.0e3, 1.0e5, 1.0e7, 2.0e7],
              multigroup_flux=[1.0, 2.0, 3.0, 4.0], chain_file=_CHAIN_SIMPLE,
              nuclides=["Fe56"], reactions=["(n,gamma)"])
    m_obj = MicroXS.from_multigroup_flux(pendf_library=PendfTapeLibrary(src), **kw)
    m_path = MicroXS.from_multigroup_flux(pendf_library=str(src), **kw)
    np.testing.assert_array_equal(m_obj.data, m_path.data)


def test_from_multigroup_flux_rejects_urr_on_tape_adapter(tmp_path):
    # URR self-shielding needs probability tables the tape adapter does not
    # serve; the combination is rejected loudly, not silently skipped.
    from openmc.deplete.microxs import MicroXS

    src = _tape_dir(tmp_path, ("Fe56",))
    with pytest.raises(ValueError, match="tape adapter"):
        MicroXS.from_multigroup_flux(
            energies=[0.0, 2.0e7], multigroup_flux=[1.0], chain_file=_CHAIN_SIMPLE,
            nuclides=["Fe56"], reactions=["(n,gamma)"],
            pendf_library=PendfTapeLibrary(src),
            urr_material_dilution={"Fe56": 1.0})
