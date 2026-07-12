"""Tests for the Flux ndarray subclass returned by get_microxs_and_flux.

Flux is a backward-compatible numpy.ndarray subclass that also carries the
energy group boundaries in an ``energy_bounds`` attribute. These tests cover
the ndarray mechanics, attribute propagation, pickle round-trips (flux lists
are scattered over MPI via pickle), and IndependentOperator ingestion of the
tuple, Flux, and bare-array flux forms.
"""

from pathlib import Path
import pickle

import numpy as np
import pytest

from openmc import Material
from openmc.deplete import Flux, IndependentOperator, MicroXS

CHAIN_PATH = Path(__file__).parents[1] / "chain_simple.xml"
ONE_GROUP_XS = Path(__file__).parents[1] / "micro_xs_simple.csv"


# --- Flux ndarray mechanics ---

def test_flux_is_ndarray():
    """Flux behaves like a plain 1D ndarray."""
    eb = np.array([0.0, 1.0, 2.0, 20.0e6])
    f = Flux([1.0, 2.0, 3.0], energy_bounds=eb)
    assert isinstance(f, np.ndarray)
    assert f.sum() == pytest.approx(6.0)
    assert f[1] == pytest.approx(2.0)
    np.testing.assert_array_equal(f * 2, [2.0, 4.0, 6.0])
    np.testing.assert_array_equal(f.energy_bounds, eb)


def test_flux_default_energy_bounds_none():
    """Without energy_bounds the attribute is None."""
    f = Flux([1.0, 2.0])
    assert f.energy_bounds is None


def test_flux_slice_keeps_energy_bounds():
    """__array_finalize__ propagates energy_bounds through views/slices."""
    eb = np.array([0.0, 1.0, 2.0, 3.0])
    f = Flux([1.0, 2.0, 3.0], energy_bounds=eb)
    sl = f[1:]
    assert isinstance(sl, Flux)
    np.testing.assert_array_equal(sl.energy_bounds, eb)
    # Arithmetic results keep the attribute too
    assert (f + 1).energy_bounds is not None


def test_flux_pickle_roundtrip():
    """Pickle preserves both data and energy_bounds (needed for MPI scatter)."""
    eb = np.array([0.0, 1.0, 2.0, 3.0])
    f = Flux([4.0, 5.0, 6.0], energy_bounds=eb)
    g = pickle.loads(pickle.dumps(f))
    assert isinstance(g, Flux)
    np.testing.assert_array_equal(g, f)
    np.testing.assert_array_equal(g.energy_bounds, eb)


def test_flux_pickle_none_energy_bounds():
    """energy_bounds=None survives a pickle round-trip."""
    g = pickle.loads(pickle.dumps(Flux([1.0, 2.0])))
    assert g.energy_bounds is None


# --- IndependentOperator ingestion of the three flux forms ---

def _build_operator(flux_item):
    """Construct a 1-material IndependentOperator with a single flux item."""
    fuel = Material(name="uo2")
    fuel.add_element("U", 1, percent_type="ao", enrichment=4.25)
    fuel.add_element("O", 2)
    fuel.set_density("g/cc", 10.4)
    fuel.depletable = True
    fuel.volume = 1.0
    micro_xs = MicroXS.from_csv(ONE_GROUP_XS)
    return IndependentOperator([fuel], [flux_item], [micro_xs], CHAIN_PATH)


def test_operator_ingests_flux_forms():
    """Flux, (flux, energy_bounds) tuple, and bare-array inputs normalize."""
    arr = np.array([1.0, 2.0, 3.0])
    eb = np.array([0.0, 1.0, 2.0, 20.0e6])

    op_flux = _build_operator(Flux(arr, energy_bounds=eb))
    op_tuple = _build_operator((arr, eb))
    op_bare = _build_operator(arr)

    # Flux and tuple forms both recover the flux array and energy bounds
    for op in (op_flux, op_tuple):
        stored_flux, stored_eb = op._flux_with_energy[0]
        np.testing.assert_array_equal(stored_flux, arr)
        np.testing.assert_array_equal(stored_eb, eb)
        np.testing.assert_array_equal(op._energy_bins, eb)

    # Flux is unwrapped to a plain ndarray on storage
    assert type(op_flux._flux_with_energy[0][0]) is np.ndarray

    # Bare array carries no energy information
    stored_flux, stored_eb = op_bare._flux_with_energy[0]
    np.testing.assert_array_equal(stored_flux, arr)
    assert stored_eb is None
    assert op_bare._energy_bins is None
