"""Unit tests for the pointwise PENDF flux-collapse path in openmc.deplete.

Covers the module-level ``_group_average`` flat-in-bin kernel, the
``_build_xs_table_pendf`` sparse-table builder, the ``pendf_library`` dispatch
in :meth:`MicroXS.from_multigroup_flux`, and the re-added group-length guard in
``_SparseXSTable.collapse``.
"""
import io
import os
from pathlib import Path

import numpy as np
import pytest

from openmc.deplete.chain import Chain
from openmc.deplete.microxs import (
    MicroXS,
    _SparseXSTable,
    _group_average,
    _build_xs_table_pendf,
)

CHAIN_FILE = Path(__file__).parents[1] / "chain_simple.xml"
PENDF_DIR = Path(os.environ.get(
    "OPENMC_PENDF_TEST_DATA",
    "/home/perry/NukeData/Activation/PENDF/Point_TENDL2017/pendf"))


class _FakePendf:
    """Minimal duck-typed stand-in for openmc.data.PendfLibrary."""

    def __init__(self, data):
        # data: {nuclide: {mt: (energy, xs)}}
        self._data = data

    @property
    def nuclides(self):
        return list(self._data)

    def reactions(self, nuclide):
        return list(self._data[nuclide])

    def xs(self, nuclide, mt):
        return self._data[nuclide][mt]


def _fake_two_by_two():
    # Constant cross sections tabulated over the full energy span so that every
    # group average equals the constant. (n,gamma)=MT102, fission=MT18.
    e = np.array([0.0, 2.0e7])
    return _FakePendf({
        "Gd157": {102: (e, np.array([3.0, 3.0])),
                  18:  (e, np.array([0.0, 0.0]))},   # all-zero -> skipped
        "U235":  {102: (e, np.array([5.0, 5.0])),
                  18:  (e, np.array([11.0, 11.0]))},
    })


# ---------------------------------------------------------------------------
# _group_average
# ---------------------------------------------------------------------------

def test_group_average_kernel():
    """The flat-in-bin ``_group_average`` kernel across every shape it must
    handle: an exactly-integrated ramp (below / inside / straddling / above the
    tabulated range), a flat plateau, a coincident-energy step, a jump landing on
    a group edge, and a threshold reaction."""
    # Ramp xs = a + b*E is integrated exactly (piecewise-linear).
    a, b = 2.0, 3.0
    energy = np.array([1.0, 1.7, 3.2, 5.0, 6.66, 8.1, 10.0])
    xs = a + b * energy
    # [0.5,1] fully below, [1,2.5] and [2.5,5] fully inside, [5,12] straddles the
    # top tabulated end (10), [12,20] fully outside.
    edges = np.array([0.5, 1.0, 2.5, 5.0, 12.0, 20.0])
    result = _group_average(energy, xs, edges)

    def exact(lo, hi):
        return a * (hi - lo) + 0.5 * b * (hi**2 - lo**2)

    assert result[0] == 0.0                                       # below range
    assert result[1] == pytest.approx(exact(1.0, 2.5) / 1.5, rel=1e-13)
    assert result[2] == pytest.approx(exact(2.5, 5.0) / 2.5, rel=1e-13)
    # Straddling group: only the [5, 10] portion contributes, divided by full dE.
    assert result[3] == pytest.approx(exact(5.0, 10.0) / 7.0, rel=1e-13)
    assert result[4] == 0.0                                       # above range

    # Flat xs collapses to exactly the constant in every covered group.
    energy = np.array([1.0, 4.0, 7.0, 10.0])
    xs = np.full_like(energy, 5.0)
    result = _group_average(energy, xs, np.array([1.0, 3.0, 6.0, 10.0]))
    assert result == pytest.approx([5.0, 5.0, 5.0], rel=1e-14)

    # Coincident-energy jumps keep both sides (C++ for_each_panel semantics):
    # 158 b plateau on [100, 150], step down to a 34 b plateau on [150, 200].
    energy = np.array([100.0, 150.0, 150.0, 200.0])
    xs = np.array([158.0, 158.0, 34.0, 34.0])
    result = _group_average(energy, xs, np.array([100.0, 200.0]))
    assert result[0] == pytest.approx(96.0, rel=1e-13)            # (50*158+50*34)/100

    # The same jump coinciding with a group edge splits cleanly between groups.
    result = _group_average(energy, xs, np.array([100.0, 150.0, 200.0]))
    assert result[0] == pytest.approx(158.0, rel=1e-13)
    assert result[1] == pytest.approx(34.0, rel=1e-13)

    # A threshold reaction gives zero in every group below the threshold.
    energy = np.array([1.0, 5.0, 10.0])
    xs = np.array([0.0, 0.0, 10.0])                               # 0 on [1,5], ramp
    result = _group_average(energy, xs, np.array([1.0, 3.0, 5.0, 10.0]))
    assert result[0] == 0.0                                       # [1,3] below
    assert result[1] == 0.0                                       # [3,5] below
    assert result[2] == pytest.approx(5.0, rel=1e-13)            # integral 25 / dE 5


@pytest.mark.parametrize("energy, xs, edges, match", [
    # A descending tabulated grid is rejected, not silently averaged to zero.
    (np.array([10.0, 5.0, 1.0]), np.array([1.0, 2.0, 3.0]),
     np.array([1.0, 5.0, 10.0]), "non-decreasing"),
    # A NaN cross section is rejected before it can poison the whole table.
    (np.array([1.0, 5.0, 10.0]), np.array([1.0, np.nan, 3.0]),
     np.array([1.0, 10.0]), "NaN"),
])
def test_group_average_rejects_bad_input(energy, xs, edges, match):
    """A descending grid and a NaN cross section are both rejected loudly."""
    with pytest.raises(ValueError, match=match):
        _group_average(energy, xs, edges)


# ---------------------------------------------------------------------------
# from_multigroup_flux(pendf_library=...) and _build_xs_table_pendf
# ---------------------------------------------------------------------------

def test_from_multigroup_flux_pendf_dispatch_and_guards():
    """The ``pendf_library`` dispatch in :meth:`from_multigroup_flux`: the happy
    2x2 collapse (with a skipped all-zero row), the ``temperature=None`` sentinel
    accepted, the guards that reject a non-default temperature and the
    continuous-energy ``cross_sections``/init kwargs, and the
    ``_build_xs_table_pendf`` skip of absent nuclides/MTs."""
    fake = _fake_two_by_two()
    edges = [0.0, 1.0e3, 1.0e5, 1.0e7, 2.0e7]
    flux = [1.0, 2.0, 3.0, 4.0]  # arbitrary; constant xs is flux-independent

    micro = MicroXS.from_multigroup_flux(
        energies=edges, multigroup_flux=flux, chain_file=CHAIN_FILE,
        nuclides=["Gd157", "U235"], reactions=["(n,gamma)", "fission"],
        pendf_library=fake)
    assert isinstance(micro, MicroXS)
    assert micro.nuclides == ["Gd157", "U235"]
    assert micro.reactions == ["(n,gamma)", "fission"]
    assert micro["Gd157", "(n,gamma)"] == pytest.approx([3.0])
    assert micro["Gd157", "fission"] == pytest.approx([0.0])      # skipped zero row
    assert micro["U235", "(n,gamma)"] == pytest.approx([5.0])
    assert micro["U235", "fission"] == pytest.approx([11.0])

    # temperature=None (the sentinel default) is accepted with pendf_library.
    micro = MicroXS.from_multigroup_flux(
        energies=[0.0, 2.0e7], multigroup_flux=[1.0], chain_file=CHAIN_FILE,
        nuclides=["Gd157", "U235"], reactions=["(n,gamma)", "fission"],
        pendf_library=fake, temperature=None)
    assert isinstance(micro, MicroXS)
    assert micro["U235", "(n,gamma)"] == pytest.approx([5.0])

    # Guards: a non-default temperature selects the CE path and is rejected; the
    # continuous-energy cross_sections argument and any openmc.lib.init keyword
    # (captured by **init_kwargs) are rejected too.
    kwargs = dict(
        energies=[0.0, 2.0e7], multigroup_flux=[1.0],
        nuclides=["Gd157"], reactions=["(n,gamma)"], pendf_library=fake)
    with pytest.raises(ValueError, match="temperature"):
        MicroXS.from_multigroup_flux(temperature=500, **kwargs)
    with pytest.raises(ValueError, match="cross_sections"):
        MicroXS.from_multigroup_flux(cross_sections="fake.xml", **kwargs)
    with pytest.raises(ValueError, match="cross_sections|init"):
        MicroXS.from_multigroup_flux(threads=2, **kwargs)

    # _build_xs_table_pendf omits missing nuclides/MTs and all-zero rows. This
    # fake has no MF=10 pathways, so the chain is unused (empty Chain).
    e = np.array([0.0, 2.0e7])
    fake2 = _FakePendf({"U235": {18: (e, np.array([4.0, 4.0]))}})  # no (n,gamma)
    table = _build_xs_table_pendf(
        ["Gd157", "U235"], ["(n,gamma)", "fission"],
        np.array([0.0, 1.0e7, 2.0e7]), fake2, Chain())
    # Only (U235, fission) survives -> one row mapping to (nuc=1, rxn=1).
    assert table.xs_matrix.shape == (1, 2)
    assert table.nuc_indices.tolist() == [1]
    assert table.rxn_indices.tolist() == [1]
    assert table.nuc_indices.dtype == np.int32
    assert table.rxn_indices.dtype == np.int32
    assert table.xs_matrix.dtype == np.float64
    assert table.xs_matrix[0] == pytest.approx([4.0, 4.0])


# ---------------------------------------------------------------------------
# _SparseXSTable.collapse group-length guard
# ---------------------------------------------------------------------------

def test_collapse_group_length_guard():
    table = _SparseXSTable(
        ["Gd157"], ["(n,gamma)"],
        np.array([[1.0, 2.0, 3.0]]),
        np.array([0], np.int32), np.array([0], np.int32))

    # Correct length works
    table.collapse(np.array([0.2, 0.3, 0.5]))

    with pytest.raises(ValueError, match="groups"):
        table.collapse(np.array([0.5, 0.5]))


# ---------------------------------------------------------------------------
# Real-data spot check vs Tabulated1D.integral() (skipped if data absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (PENDF_DIR / "n-Fe056.pendf").exists(),
                    reason="TENDL-2017 PENDF Fe56 tape not available")
def test_group_average_vs_tabulated_integral():
    from openmc.data import Tabulated1D
    from openmc.data.endf import Evaluation, get_head_record, get_tab1_record

    path = PENDF_DIR / "n-Fe056.pendf"
    ev = Evaluation(str(path))
    f = io.StringIO(ev.section[3, 102])
    get_head_record(f)                 # MF=3 HEAD record
    _, tab = get_tab1_record(f)        # (E, xs) TAB1
    energy = np.asarray(tab.x, dtype=float)
    xs = np.asarray(tab.y, dtype=float)

    # ~10 group edges snapped to existing grid points, 1e-5 eV -> 20 MeV, so the
    # cumulative-integral reference is exact at each edge.
    targets = np.array([1e-5, 1e-3, 1e-1, 1e1, 1e3, 1e5, 1e6, 5e6, 1e7,
                        1.5e7, 2e7])
    idx = np.unique(np.clip(np.searchsorted(energy, targets),
                            0, len(energy) - 1))
    edges = energy[idx]

    # Reference: exact cumulative integral on the tabulated (lin-lin) grid.
    # Truncate at the top edge so the reference stays on the same grid points
    # the group average uses. cum[i] = integral from energy[0] to energy[i].
    top = idx[-1]
    cum = Tabulated1D(energy[:top + 1], xs[:top + 1]).integral()

    result = _group_average(energy, xs, edges)
    ref = np.diff(cum[idx]) / np.diff(edges)
    np.testing.assert_allclose(result, ref, rtol=1e-10)

    # Same values must come out of the sparse-table builder via the duck type
    fake = _FakePendf({"Fe56": {102: (energy, xs)}})
    table = _build_xs_table_pendf(["Fe56"], ["(n,gamma)"], edges, fake, Chain())
    assert table.xs_matrix.shape == (1, len(edges) - 1)
    np.testing.assert_allclose(table.xs_matrix[0], result, rtol=1e-12)
