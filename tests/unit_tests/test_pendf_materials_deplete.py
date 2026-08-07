"""Test that Materials.deplete batches its group cross section table builds by
(energy group structure, temperature).

Split out of ``test_materials.py`` so the upstream-owned file stays free of
branch-added tests.
"""

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import openmc
import openmc.deplete
import openmc.deplete.microxs as microxs_mod
import openmc.lib


def test_materials_deplete_groups_by_energy_temperature(monkeypatch):
    """Materials.deplete groups materials by (energy, temperature): the group
    cross section table is built once per distinct group, the micros keep
    material order, and each equals the ungrouped from_multigroup_flux result.
    """
    chain = Path(__file__).parents[1] / "chain_ni.xml"
    energy = "VITAMIN-J-42"
    n_groups = 42

    rng = np.random.default_rng(0)
    fluxes = [rng.random(n_groups) for _ in range(3)]
    temps = [293.6, 293.6, 600.0]

    def make_mat(nuclide, temperature):
        m = openmc.Material()
        m.add_nuclide(nuclide, 1.0)
        m.set_density("g/cm3", 7.87)
        m.depletable = True
        m.temperature = temperature
        m.volume = 1.0
        return m

    # mat0 and mat1 share (energy, T=293.6); mat2 differs only in temperature
    mats = openmc.Materials([
        make_mat("Ni58", temps[0]),
        make_mat("Ni60", temps[1]),
        make_mat("Ni62", temps[2]),
    ])

    # Spy on the table build and capture the micros the operator receives
    build_spy = mock.MagicMock(wraps=microxs_mod._build_xs_table_ce)
    monkeypatch.setattr(microxs_mod, "_build_xs_table_ce", build_spy)

    op_spy = mock.MagicMock(side_effect=StopIteration)
    monkeypatch.setattr(openmc.deplete, "IndependentOperator", op_spy)

    with pytest.raises(StopIteration):
        mats.deplete(
            multigroup_fluxes=fluxes,
            energy_group_structures=[energy, energy, energy],
            timesteps=[1.0],
            source_rates=1.0,
            chain_file=chain,
        )

    # Two distinct (energy, temperature) groups -> two builds, not three
    assert build_spy.call_count == 2

    micros = op_spy.call_args.kwargs["micros"]
    assert len(micros) == 3
    assert all(m is not None for m in micros)

    # Each per-material micro equals the ungrouped single-flux result
    with openmc.lib.TemporarySession():
        for flux, temperature, micro in zip(fluxes, temps, micros):
            ref = openmc.deplete.MicroXS.from_multigroup_flux(
                energies=energy, multigroup_flux=flux, chain_file=chain,
                temperature=temperature)
            assert micro.data == pytest.approx(ref.data)
