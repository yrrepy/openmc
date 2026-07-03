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
from pathlib import Path

import h5py
import numpy as np
import pytest

from openmc.data import GroupedPendfLibrary
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


def _make_pointwise_h5(path):
    """Write a tiny synthetic pointwise PENDF HDF5 file."""
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = 1
        f.attrs["library"] = np.bytes_("TEST")
        f.attrs["temperature"] = np.float64(293.16)

        # In115: MT102 (n,gamma) with three MF=10 partials on a grid holding a
        # coincident-duplicate node (1e6); LFS4 is an all-zero partial. MT103
        # (n,p) has no partials; MT16 (n,2n) is a threshold reaction that bins to
        # zero below 8e6. MT1 is non-chain-relevant and must be dropped.
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
        g = self._f[f"{nuc}/MT{mt}"]
        return sorted(int(k[3:]) for k in g if k.startswith("LFS"))

    def pathway_xs(self, nuc, mt, lfs):
        g = self._f[f"{nuc}/MT{mt}/LFS{lfs}"]
        return g["energy"][()], g["xs"][()]

    def product(self, nuc, mt, lfs):
        p = self._f[f"{nuc}/MT{mt}/LFS{lfs}"].attrs.get("product")
        return p.decode() if isinstance(p, (bytes, np.bytes_)) else p


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

    ref = _build_xs_table_pendf(nucs, REACTIONS, EDGES, pointwise)
    got = _build_xs_table_pendf(nucs, REACTIONS, EDGES, grouped)

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
    assert set(grouped.reactions("In115")) == {16, 102, 103}
    assert grouped.pathways("In115", 102) == [0, 1, 4]
    assert grouped.pathways("In115", 103) == []
    assert grouped.product("In115", 102, 1) == "In116_m1"
    assert grouped.xs_g("In115", 102).shape == (len(EDGES) - 1,)


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
    with pytest.raises(ValueError, match="group edges"):
        _build_xs_table_pendf(["In115"], REACTIONS, other, grouped)


def test_writer_summary(tmp_path):
    """Writer reports rows and a tiny (passing) consistency deviation."""
    src = tmp_path / "pointwise.h5"
    out = tmp_path / "grouped.h5"
    _make_pointwise_h5(src)
    stats = pgb.bin_pendf_library(src, out, EDGES)
    # 3 In115 MTs (102,103,16) + 3 LFS partials + 1 Fe56 MT = 7 rows.
    assert stats["rows"] == 7
    assert stats["n_warnings"] == 0
    assert stats["worst_dev"] < 1e-5
