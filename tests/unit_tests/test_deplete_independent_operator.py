"""Basic unit tests for openmc.deplete.IndependentOperator instantiation

"""

from pathlib import Path

import numpy as np
import pytest

from openmc import Material
from openmc.deplete import IndependentOperator, MicroXS, Chain
from openmc.deplete.microxs import write_global_microxs_hdf5

CHAIN_PATH = Path(__file__).parents[1] / "chain_simple.xml"
ONE_GROUP_XS = Path(__file__).parents[1] / "micro_xs_simple.csv"


def test_operator_init():
    """The test uses a temporary dummy chain. This file will be removed
    at the end of the test, and only contains a depletion_chain node."""
    volume = 1
    nuclides = {'U234': 8.922411359424315e+18,
                'U235': 9.98240191860822e+20,
                'U238': 2.2192386373095893e+22,
                'U236': 4.5724195495061115e+18,
                'O16': 4.639065406771322e+22,
                'O17': 1.7588724018066158e+19}
    flux = 1.0
    micro_xs = MicroXS.from_csv(ONE_GROUP_XS)
    chain = Chain.from_xml(CHAIN_PATH)
    IndependentOperator.from_nuclides(
        volume, nuclides, flux, micro_xs, chain, nuc_units='atom/cm3')

    fuel = Material(name="uo2")
    fuel.add_element("U", 1, percent_type="ao", enrichment=4.25)
    fuel.add_element("O", 2)
    fuel.set_density("g/cc", 10.4)
    fuel.depletable = True
    fuel.volume = 1
    materials = [fuel]
    fluxes = [1.0]
    micros = [micro_xs]
    IndependentOperator(materials, fluxes, micros, CHAIN_PATH)


def test_error_handling():
    micro_xs = MicroXS.from_csv(ONE_GROUP_XS)
    fuel = Material(name="oxygen")
    fuel.add_element("O", 2)
    fuel.set_density("g/cc", 1)
    fuel.depletable = True
    fuel.volume = 1
    materials = [fuel]
    fluxes = [1.0, 2.0]
    micros = [micro_xs]
    with pytest.raises(ValueError, match=r"The length of fluxes \(2\)"):
        IndependentOperator(materials, fluxes, micros, CHAIN_PATH)


# --- Helper for HDF5 integration tests ---

def _make_material(mat_id):
    """Create a depletable material with only chain nuclides."""
    mat = Material(material_id=mat_id)
    mat.add_nuclide('U235', 1e-3)
    mat.add_nuclide('U238', 1e-2)
    mat.set_density('sum')
    mat.depletable = True
    mat.volume = 1.0
    return mat


# --- from_microxs_file integration tests ---

def test_from_microxs_file(tmp_path):
    """from_microxs_file produces a working operator."""
    micro_xs = MicroXS.from_csv(ONE_GROUP_XS)
    n_mats = 3
    materials = [_make_material(i + 1) for i in range(n_mats)]
    mat_ids = sorted([str(m.id) for m in materials], key=int)

    fname = tmp_path / 'microxs.h5'
    write_global_microxs_hdf5(
        [micro_xs] * n_mats, fname, mat_ids)

    op = IndependentOperator.from_microxs_file(
        materials, fname, chain_file=CHAIN_PATH,
        require_isomeric_branching=False)

    assert len(op.cross_sections) == n_mats
    assert len(op.fluxes) == n_mats
    for xs in op.cross_sections:
        np.testing.assert_array_equal(xs.data, micro_xs.data)


def test_from_microxs_file_with_flux(tmp_path):
    """from_microxs_file correctly extracts flux arrays from tuples."""
    micro_xs = MicroXS.from_csv(ONE_GROUP_XS)
    n_mats = 2
    materials = [_make_material(i + 1) for i in range(n_mats)]
    mat_ids = sorted([str(m.id) for m in materials], key=int)

    flux = np.array([1.5, 2.5, 3.5])
    energy_bounds = np.array([0.0, 0.625, 1e6, 2e7])
    fluxes = [(flux, energy_bounds)] * n_mats

    fname = tmp_path / 'microxs.h5'
    write_global_microxs_hdf5(
        [micro_xs] * n_mats, fname, mat_ids, fluxes=fluxes)

    op = IndependentOperator.from_microxs_file(
        materials, fname, chain_file=CHAIN_PATH,
        require_isomeric_branching=False)

    # Fluxes should be plain arrays, not tuples
    for f in op.fluxes:
        assert isinstance(f, np.ndarray)
        np.testing.assert_array_equal(f, flux)


def test_prefiltered_skips_validation(tmp_path):
    """_prefiltered=True allows len(micros) < len(materials)."""
    micro_xs = MicroXS.from_csv(ONE_GROUP_XS)
    materials = [_make_material(i + 1) for i in range(5)]

    # Pass only 2 micros/fluxes for 5 materials -- would fail without
    # _prefiltered=True
    op = IndependentOperator(
        materials,
        [np.ones(1), np.ones(1)],
        [micro_xs, micro_xs],
        chain_file=CHAIN_PATH,
        require_isomeric_branching=False,
        _prefiltered=True,
    )
    assert len(op.cross_sections) == 2
