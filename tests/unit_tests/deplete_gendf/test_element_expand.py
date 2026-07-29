"""``Element.expand`` tests for the GENDF and elemental-X0 expansion paths.

Moved verbatim from ``tests/unit_tests/test_element.py`` to keep the
upstream-owned file conflict-free.
"""

import openmc
from pytest import approx, raises

from openmc.data import NATURAL_ABUNDANCE


def _write_c0_cross_sections(tmp_path):
    """Write a cross_sections.xml whose only entry is elemental C0."""
    path = tmp_path / 'cross_sections.xml'
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<cross_sections>\n'
        '  <library materials="C0" path="C0.h5" type="neutron"/>\n'
        '</cross_sections>\n'
    )
    return str(path)


def test_expand_hdf5_elemental_x0(tmp_path):
    """Regression (R1-44): a library with only an elemental X0 entry must
    expand to that X0 nuclide instead of raising."""
    xs = _write_c0_cross_sections(tmp_path)
    result = openmc.Element('C').expand(1.0, 'ao', cross_sections=xs)
    assert len(result) == 1
    nuc, frac, ptype = result[0]
    assert nuc == 'C0'
    assert frac == approx(1.0)
    assert ptype == 'ao'


class _MockGENDFLib:
    """Minimal GENDF stand-in with available_nuclides() for expand()."""

    def __init__(self, nuclides):
        self._nuclides = list(nuclides)

    def available_nuclides(self):
        return self._nuclides


def test_expand_gendf_all_present():
    """GENDF branch: all natural isotopes present -> natural abundances."""
    lib = _MockGENDFLib(['Li6', 'Li7'])
    for nuc, frac, ptype in openmc.Element('Li').expand(
            100.0, 'ao', gendf_library=lib):
        assert frac == approx(NATURAL_ABUNDANCE[nuc] * 100.0)
        assert ptype == 'ao'


def test_expand_gendf_elemental_x0():
    """GENDF branch: only elemental X0 present -> single X0 entry."""
    lib = _MockGENDFLib(['C0'])
    result = openmc.Element('C').expand(1.0, 'ao', gendf_library=lib)
    assert len(result) == 1
    nuc, frac, ptype = result[0]
    assert nuc == 'C0'
    assert frac == approx(1.0)


def test_expand_gendf_partition_absent():
    """GENDF branch: O missing O17/O18 folds their abundance into O16."""
    lib = _MockGENDFLib(['O16'])
    result = openmc.Element('O').expand(100.0, 'ao', gendf_library=lib)
    assert len(result) == 1
    nuc, frac, ptype = result[0]
    assert nuc == 'O16'
    folded = (NATURAL_ABUNDANCE['O16'] + NATURAL_ABUNDANCE['O17']
              + NATURAL_ABUNDANCE['O18'])
    assert frac == approx(100.0 * folded)
    # Total abundance is conserved (all natural O folded into O16)
    assert frac == approx(100.0)


def test_expand_gendf_unplaceable_absent():
    """GENDF branch: an absent isotope with no rule raises ValueError."""
    lib = _MockGENDFLib(['Fe56'])
    with raises(ValueError):
        openmc.Element('Fe').expand(100.0, 'ao', gendf_library=lib)


def test_expand_gendf_no_mutual():
    """GENDF branch: no mutual isotopes, no X0 fallback -> ValueError."""
    lib = _MockGENDFLib(['C12'])
    with raises(ValueError):
        openmc.Element('Fe').expand(100.0, 'ao', gendf_library=lib)
