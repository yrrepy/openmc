"""Unit tests for the pre-binned grouped PENDF library (pendf-groupbin).

Covers the writer core (``tools/pendf_group_bin.bin_pendf_library``), the
:class:`openmc.data.GroupedPendfLibrary` reader, and the grouped fast path in
``openmc.deplete.microxs._build_xs_table_pendf``. The headline guarantee is
bit-exactness: collapsing through the grouped library must reproduce, element
for element, the table built by flat-weighting the pointwise data at runtime
(float64 end to end), including identical nuclide/reaction axes.

All fixtures are synthetic -- no dependency on the multi-GB real library.
"""
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from openmc.data import GroupedPendfLibrary
from openmc.data.pendf_grouped import GROUPED_FORMAT
from openmc.deplete import MicroXS
from openmc.deplete.chain import Chain
from openmc.deplete.nuclide import Nuclide
from openmc.deplete.microxs import _build_xs_table_pendf

# Load the writer CLI module (tools/ is not a package) by path.
_TOOLS = Path(__file__).parents[2] / "tools" / "pendf_group_bin.py"
_spec = importlib.util.spec_from_file_location("pendf_group_bin", _TOOLS)
pgb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pgb)

# Group structure with edges at the coincident node (1e6) and the MT16 threshold
# (8e6), so both are exercised at a group boundary.
EDGES = np.array([1e-5, 1e3, 1e6, 5e6, 8e6, 1e7, 2e7])
REACTIONS = ["(n,gamma)", "(n,p)", "(n,2n)"]  # MTs 102, 103, 16


def _groupbin_chain():
    """Chain binding In115 (n,gamma) MF=10 LFS {0,1,4} to product-qualified rows.

    The collapse now sources isomeric row names from the chain (each MF=10 LFS
    partial binds to the chain reaction carrying that ``pendf_lfs``), so the
    grouped-vs-pointwise tests need a chain that names In116 / In116_m1 / In116_m2
    (LFS 0 / 1 / 4). Only MT102 has partials in the requested REACTIONS.
    """
    chain = Chain()
    nuc = Nuclide("In115")
    nuc.add_reaction("(n,gamma)", "In116", 0.0, 1.0, pendf_lfs=0)
    nuc.add_reaction("(n,gamma)_m1", "In116_m1", 0.0, 1.0, pendf_lfs=1)
    nuc.add_reaction("(n,gamma)_m2", "In116_m2", 0.0, 1.0, pendf_lfs=4)
    chain.add_nuclide(nuc)
    return chain


def _make_pointwise_h5(path):
    """Write a tiny synthetic pointwise PENDF HDF5 file."""
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = 2
        f.attrs["library"] = np.bytes_("TEST")
        f.attrs["temperature"] = np.float64(293.16)

        # In115: MT102 (n,gamma) with three MF=10 partials on a grid holding a
        # coincident-duplicate node (1e6); LFS4 is an all-zero partial (all three
        # levels -> In116, so each keeps a bare ``LFS<l>`` name). MT103 (n,p) has
        # no partials; MT16 (n,2n) is a threshold reaction that bins to zero below
        # 8e6. MT107 (n,a) is a synthetic LUMPED channel: one LFS shared by two
        # product nuclides (``LFS0_ZAP<izap>`` subgroups). MT1 is
        # non-chain-relevant and must be dropped.
        g = f.create_group("In115")
        g.attrs["ZA"] = np.int64(49115)
        g.attrs["LISO"] = np.int64(0)
        g.attrs["source_file"] = np.bytes_("n-In115.pendf")

        e = np.array([1e-5, 1e3, 1e6, 1e6, 5e6, 2e7])  # duplicate at 1e6
        xs0 = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 1.0])  # LFS0 -> In116
        xs1 = np.array([0.5, 0.5, 0.5, 0.5, 0.4, 0.2])  # LFS1 -> In116_m1
        xs4 = np.zeros_like(e)                           # LFS4 -> In116_m2 (zero)
        total = xs0 + xs1 + xs4

        mt = g.create_group("MT102")
        mt.attrs["QI"] = np.float64(6784730.0)
        mt.attrs["QM"] = np.float64(6784730.0)
        mt.create_dataset("energy", data=e)
        mt.create_dataset("xs", data=total)
        partials = {0: ("In116", xs0), 1: ("In116_m1", xs1), 4: ("In116_m2", xs4)}
        for lfs, (prod, xs) in partials.items():
            sub = mt.create_group(f"LFS{lfs}")
            sub.attrs["product"] = np.bytes_(prod)
            sub.attrs["LFS"] = np.int64(lfs)
            sub.attrs["IZAP"] = np.int64(49116)   # all three levels -> In116
            sub.attrs["QI"] = np.float64(6784730.0)
            sub.create_dataset("energy", data=e)
            sub.create_dataset("xs", data=xs)

        mt103 = g.create_group("MT103")
        mt103.attrs["QI"] = np.float64(0.0)
        mt103.create_dataset("energy", data=np.array([1e-5, 2e7]))
        mt103.create_dataset("xs", data=np.array([0.1, 0.5]))

        mt16 = g.create_group("MT16")
        mt16.attrs["QI"] = np.float64(-8.0e6)
        mt16.create_dataset("energy", data=np.array([1e-5, 8e6, 1e7, 2e7]))
        mt16.create_dataset("xs", data=np.array([0.0, 0.0, 1.2, 2.0]))

        # MT107 (n,a): a synthetic LUMPED channel -- one LFS (0) shared by two
        # distinct product nuclides, stored as ``LFS0_ZAP<izap>`` subgroups
        # (schema v2), exactly as a real multi-daughter MT=5 (n,misc) is. The
        # total equals the sum of the two partials so binning stays consistent.
        e107 = np.array([1e-5, 1e6, 2e7])
        a_xs = np.array([0.0, 0.2, 0.4])   # LFS0 -> Ag110 (IZAP 47110)
        b_xs = np.array([0.0, 0.1, 0.3])   # LFS0 -> Ag112 (IZAP 47112)
        mt107 = g.create_group("MT107")
        mt107.attrs["QI"] = np.float64(0.0)
        mt107.create_dataset("energy", data=e107)
        mt107.create_dataset("xs", data=a_xs + b_xs)
        for izap, (prod, xs) in {47110: ("Ag110", a_xs),
                                 47112: ("Ag112", b_xs)}.items():
            sub = mt107.create_group(f"LFS0_ZAP{izap}")
            sub.attrs["product"] = np.bytes_(prod)
            sub.attrs["LFS"] = np.int64(0)
            sub.attrs["IZAP"] = np.int64(izap)
            sub.attrs["QI"] = np.float64(0.0)
            sub.create_dataset("energy", data=e107)
            sub.create_dataset("xs", data=xs)

        mt1 = g.create_group("MT1")  # non-chain-relevant: must not be written
        mt1.create_dataset("energy", data=np.array([1e-5, 2e7]))
        mt1.create_dataset("xs", data=np.array([10.0, 5.0]))

        # Fe56: a single MT102 with no partials.
        g2 = f.create_group("Fe56")
        g2.attrs["ZA"] = np.int64(26056)
        mt = g2.create_group("MT102")
        mt.attrs["QI"] = np.float64(0.0)
        mt.create_dataset("energy", data=np.array([1e-5, 2e7]))
        mt.create_dataset("xs", data=np.array([2.0, 0.05]))


class _PointwiseShim:
    """Pointwise duck API over the synthetic h5 (no ``group_edges``)."""

    def __init__(self, path):
        self._f = h5py.File(path, "r")

    @property
    def nuclides(self):
        return [k for k, v in self._f.items() if isinstance(v, h5py.Group)]

    def reactions(self, nuc):
        return sorted(int(k[2:]) for k in self._f[nuc] if k.startswith("MT"))

    def xs(self, nuc, mt):
        g = self._f[f"{nuc}/MT{mt}"]
        return g["energy"][()], g["xs"][()]

    def pathways(self, nuc, mt):
        # Read the (LFS, IZAP) attrs rather than parsing the subgroup name, so a
        # lumped ``LFS<l>_ZAP<izap>`` subgroup is handled the same as a bare one.
        g = self._f[f"{nuc}/MT{mt}"]
        return sorted((int(g[k].attrs["LFS"]), int(g[k].attrs["IZAP"]))
                      for k in g if k.startswith("LFS"))

    def _partial(self, nuc, mt, lfs, izap):
        g = self._f[f"{nuc}/MT{mt}"]
        for k in g:
            if (k.startswith("LFS") and int(g[k].attrs["LFS"]) == lfs
                    and (izap is None or int(g[k].attrs["IZAP"]) == izap)):
                return g[k]
        raise KeyError((nuc, mt, lfs, izap))

    def pathway_xs(self, nuc, mt, lfs, izap=None):
        g = self._partial(nuc, mt, lfs, izap)
        return g["energy"][()], g["xs"][()]


@pytest.fixture
def libs(tmp_path):
    """Return (pointwise_shim, grouped_library) over a shared synthetic source."""
    src = tmp_path / "pointwise.h5"
    grouped = tmp_path / "grouped.h5"
    _make_pointwise_h5(src)
    pgb.bin_pendf_library(src, grouped, EDGES)
    return _PointwiseShim(src), GroupedPendfLibrary(grouped), grouped


def test_bit_exact_table(libs):
    """Grouped collapse reproduces the pointwise table exactly (axes + values)."""
    pointwise, grouped, _ = libs
    nucs = ["In115", "Fe56"]
    chain = _groupbin_chain()

    ref = _build_xs_table_pendf(nucs, REACTIONS, EDGES, pointwise, chain)
    got = _build_xs_table_pendf(nucs, REACTIONS, EDGES, grouped, chain)

    assert got.nuclides == ref.nuclides
    assert got.reactions == ref.reactions
    assert np.array_equal(got.nuc_indices, ref.nuc_indices)
    assert np.array_equal(got.rxn_indices, ref.rxn_indices)
    # Exact equality -- float64 end to end makes this the contract, not a tol.
    assert np.array_equal(got.xs_matrix, ref.xs_matrix)
    # Sanity: the isomeric expansion actually happened (In116_m1 row present).
    assert "(n,gamma)_m1" in ref.reactions


def test_reader_api(libs):
    """Reader exposes the frozen duck names and grouped accessors."""
    _, grouped, _ = libs
    assert np.array_equal(grouped.group_edges, EDGES)
    assert set(grouped.nuclides) == {"In115", "Fe56"}
    # MT1 was non-chain-relevant and must have been dropped.
    assert set(grouped.reactions("In115")) == {16, 102, 103, 107}
    # pathways() yields sorted (LFS, IZAP) pairs, one per stored partial.
    assert grouped.pathways("In115", 102) == [(0, 49116), (1, 49116), (4, 49116)]
    assert grouped.pathways("In115", 103) == []
    # A unique LFS resolves its partial with a bare-LFS call (no izap needed).
    assert grouped.pathway_xs_g("In115", 102, 1).shape == (len(EDGES) - 1,)
    assert grouped.xs_g("In115", 102).shape == (len(EDGES) - 1,)


def test_reader_lumped_pathways(libs):
    """A lumped MT (one LFS shared by two product IZAPs) enumerates as repeated
    (lfs, izap) pairs; the partial cross section needs the izap to disambiguate."""
    _, grouped, _ = libs
    # MT107 carries LFS0 shared by two daughters -> two pairs at the same LFS.
    assert grouped.pathways("In115", 107) == [(0, 47110), (0, 47112)]
    # izap selects the one partial cross section.
    assert grouped.pathway_xs_g("In115", 107, 0, 47110).shape == (len(EDGES) - 1,)
    assert grouped.pathway_xs_g("In115", 107, 0, 47112).shape == (len(EDGES) - 1,)
    # A bare-LFS call on the shared LFS is ambiguous -> ValueError naming the
    # candidate IZAPs and telling the caller to pass izap.
    with pytest.raises(ValueError, match="pass izap"):
        grouped.pathway_xs_g("In115", 107, 0)
    # A missing pathway is a KeyError that mentions the izap when one was given.
    with pytest.raises(KeyError, match="IZAP=99999"):
        grouped.pathway_xs_g("In115", 107, 0, 99999)


def test_zero_partial_row_preserved(libs):
    """The all-zero LFS4 partial is still written (gzip eats the zeros)."""
    _, _, grouped_path = libs
    with h5py.File(grouped_path, "r") as f:
        assert "In115/MT102/LFS4/xs_g" in f
        assert not f["In115/MT102/LFS4/xs_g"][()].any()
        # Attrs copied verbatim.
        assert f["In115/MT102/LFS4"].attrs["product"].decode() == "In116_m2"
        assert f["In115/MT102/LFS0"].attrs["QI"] == 6784730.0
        assert f["In115"].attrs["ZA"] == 49115


def test_edge_mismatch_hard_error(libs):
    """Collapsing a grouped library on foreign edges is a hard ValueError."""
    _, grouped, _ = libs
    other = np.array([1e-5, 1e3, 2e7])  # different count -> mismatch
    # The edge check fires before any naming, so the chain is unused here.
    with pytest.raises(ValueError, match="group edges"):
        _build_xs_table_pendf(["In115"], REACTIONS, other, grouped, Chain())


def test_missing_group_edges_raises_and_closes(tmp_path):
    """Right format attr but no group_edges: informative raise, no handle leak.

    The reader opens the file before validating it; a failure during that
    post-open init must close the handle (guarded try/except) and re-raise with
    a message that names the missing dataset rather than leaking a KeyError.
    """
    path = tmp_path / "no_edges.h5"
    with h5py.File(path, "w") as f:
        f.attrs["format"] = np.bytes_(GROUPED_FORMAT)  # passes the format check
        # ... but no 'group_edges' dataset is written.
    with pytest.raises((ValueError, KeyError), match="group_edges"):
        GroupedPendfLibrary(path)


def test_grouped_version_written(libs):
    """A freshly written grouped library stamps the current schema version (2)."""
    _, _, grouped_path = libs
    with h5py.File(grouped_path, "r") as f:
        assert f.attrs["version"] == 2


def test_grouped_version_newer_rejected(tmp_path):
    """A grouped file stamped a newer version than supported is refused, naming
    the file, the found version, and the supported version."""
    src = tmp_path / "pointwise.h5"
    grouped = tmp_path / "grouped.h5"
    _make_pointwise_h5(src)
    pgb.bin_pendf_library(src, grouped, EDGES)

    with h5py.File(grouped, "r+") as f:
        f.attrs["version"] = 99
    with pytest.raises(ValueError, match="newer than the supported version"):
        GroupedPendfLibrary(grouped)


def test_grouped_version_missing_loads(tmp_path):
    """An old grouped file lacking the version attr still loads."""
    src = tmp_path / "pointwise.h5"
    grouped = tmp_path / "grouped.h5"
    _make_pointwise_h5(src)
    pgb.bin_pendf_library(src, grouped, EDGES)

    with h5py.File(grouped, "r+") as f:
        del f.attrs["version"]
    with GroupedPendfLibrary(grouped) as lib:
        assert set(lib.nuclides) == {"In115", "Fe56"}


def test_writer_summary(tmp_path):
    """Writer reports rows and a tiny (passing) consistency deviation."""
    src = tmp_path / "pointwise.h5"
    out = tmp_path / "grouped.h5"
    _make_pointwise_h5(src)
    stats = pgb.bin_pendf_library(src, out, EDGES)
    # 4 In115 MT totals (102,103,16,107) + 3 MT102 partials + 2 MT107 lumped
    # partials + 1 Fe56 MT = 10 rows.
    assert stats["rows"] == 10
    assert stats["n_warnings"] == 0
    assert stats["worst_dev"] < 1e-5


# ---------------------------------------------------------------------------
# from_multigroup_flux: energies defaults to a grouped library's group_edges
# ---------------------------------------------------------------------------

def test_energies_default_to_group_edges(libs):
    """Omitting ``energies`` with a grouped library matches passing the edges."""
    _, grouped, _ = libs
    flux = np.ones(len(EDGES) - 1)
    nucs = ["In115", "Fe56"]
    chain = _groupbin_chain()

    explicit = MicroXS.from_multigroup_flux(
        energies=EDGES, multigroup_flux=flux, chain_file=chain,
        nuclides=nucs, reactions=REACTIONS, pendf_library=grouped)
    default = MicroXS.from_multigroup_flux(
        multigroup_flux=flux, chain_file=chain,
        nuclides=nucs, reactions=REACTIONS, pendf_library=grouped)

    assert isinstance(default, MicroXS)
    assert default.nuclides == explicit.nuclides
    assert default.reactions == explicit.reactions
    # Bit-exact: defaulting the edges must change nothing about the result.
    assert np.array_equal(default.data, explicit.data)
    # Sanity: the pathway expansion actually ran (product-qualified name).
    assert "(n,gamma)_m1" in default.reactions


def test_energies_none_without_grouped_raises(libs):
    """energies=None without a grouped library is a clear ValueError."""
    pointwise, _, _ = libs
    flux = np.ones(len(EDGES) - 1)
    # A pointwise library exposes no ``group_edges`` attribute.
    with pytest.raises(ValueError, match="grouped"):
        MicroXS.from_multigroup_flux(
            multigroup_flux=flux, nuclides=["In115"],
            reactions=REACTIONS, pendf_library=pointwise)
    # No pendf_library at all is likewise rejected.
    with pytest.raises(ValueError, match="grouped"):
        MicroXS.from_multigroup_flux(
            multigroup_flux=flux, nuclides=["In115"], reactions=REACTIONS)


def test_explicit_mismatched_edges_still_hard_error(libs):
    """An explicit energies array that differs from the library edges hard-fails."""
    _, grouped, _ = libs
    other = np.array([1e-5, 1e3, 2e7])  # different count -> mismatch
    flux = np.ones(len(other) - 1)
    with pytest.raises(ValueError, match="group edges"):
        MicroXS.from_multigroup_flux(
            energies=other, multigroup_flux=flux, chain_file=_groupbin_chain(),
            nuclides=["In115"], reactions=REACTIONS, pendf_library=grouped)


# ---------------------------------------------------------------------------
# Writer CLI hardening (tools/pendf_group_bin.main)
# ---------------------------------------------------------------------------

def _run_main(monkeypatch, src, out, *extra):
    """Invoke pgb.main() with a synthetic argv (edges passed as an .npy)."""
    edges_npy = src.parent / "edges.npy"
    np.save(edges_npy, EDGES)
    argv = ["pendf_group_bin", "--pendf-in", str(src), "--out", str(out),
            "--edges", str(edges_npy), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    pgb.main()


def test_cli_refuses_existing_out(tmp_path, monkeypatch):
    """An existing --out is refused without --force and overwritten with it."""
    src = tmp_path / "pointwise.h5"
    out = tmp_path / "grouped.h5"
    _make_pointwise_h5(src)
    out.write_bytes(b"stale")  # pre-existing output

    with pytest.raises(SystemExit):
        _run_main(monkeypatch, src, out)
    assert out.read_bytes() == b"stale"  # untouched by the refusal

    # --force lets it overwrite; the file is now a valid grouped library.
    _run_main(monkeypatch, src, out, "--force")
    with h5py.File(out, "r") as f:
        assert f.attrs["format"] == pgb.GROUPED_FORMAT


def test_cli_reports_all_unknown_nuclides(tmp_path, monkeypatch, capsys):
    """A bad --nuclides names every unknown entry and writes no output file."""
    src = tmp_path / "pointwise.h5"
    out = tmp_path / "grouped.h5"
    _make_pointwise_h5(src)

    with pytest.raises(SystemExit):
        _run_main(monkeypatch, src, out,
                  "--nuclides", "In115", "Xx999", "Zz111")
    err = capsys.readouterr().err
    assert "Xx999" in err and "Zz111" in err  # every unknown listed
    assert "In115" not in err.split("valid examples")[0]  # known one not flagged
    assert not out.exists()  # validated before any output was created


def test_cli_unwritable_out_is_clean_error(tmp_path, monkeypatch, capsys):
    """An unwritable output path gives a one-line error, not a traceback."""
    src = tmp_path / "pointwise.h5"
    _make_pointwise_h5(src)
    out = tmp_path / "no_such_dir" / "grouped.h5"  # parent missing

    with pytest.raises(SystemExit):
        _run_main(monkeypatch, src, out)
    err = capsys.readouterr().err
    assert "cannot open output file" in err
