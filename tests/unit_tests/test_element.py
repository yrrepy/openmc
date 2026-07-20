import openmc
from pytest import approx, raises, warns

from openmc.data import NATURAL_ABUNDANCE, atomic_mass


def test_expand_no_enrichment():
    """ Expand Li in natural compositions"""
    lithium = openmc.Element('Li')

    # Verify the expansion into ATOMIC fraction against natural composition
    for isotope in lithium.expand(100.0, 'ao'):
        assert isotope[1] == approx(NATURAL_ABUNDANCE[isotope[0]] * 100.0)

    # Verify the expansion into WEIGHT fraction against natural composition
    natural = {'Li6': NATURAL_ABUNDANCE['Li6'] * atomic_mass('Li6'),
               'Li7': NATURAL_ABUNDANCE['Li7'] * atomic_mass('Li7')}
    li_am = sum(natural.values())
    for key in natural:
        natural[key] /= li_am

    for isotope in lithium.expand(100.0, 'wo'):
        assert isotope[1] == approx(natural[isotope[0]] * 100.0)


def test_expand_enrichment():
    """ Expand and verify enrichment of Li """
    lithium = openmc.Element('Li')

    # Verify the enrichment by atoms
    ref = {'Li6': 75.0, 'Li7': 25.0}
    for isotope in lithium.expand(100.0, 'ao', 25.0, 'Li7', 'ao'):
        assert isotope[1] == approx(ref[isotope[0]])

    # Verify the enrichment by weight
    for isotope in lithium.expand(100.0, 'wo', 25.0, 'Li7', 'wo'):
        assert isotope[1] == approx(ref[isotope[0]])


def test_expand_no_isotopes():
    """Test that correct warning is raised for elements with no isotopes"""
    with warns(UserWarning, match='No naturally occurring'):
        element = openmc.Element('Tc')
        element.expand(100.0, 'ao')


def test_expand_ta():
    ref = {'Ta181': 100.0}
    element = openmc.Element('Ta')
    for isotope in element.expand(100.0, 'ao'):
        assert isotope[1] == approx(ref[isotope[0]])


def test_expand_exceptions():
    """ Test that correct exceptions are raised for invalid input """

    # 1 Isotope Element
    with raises(ValueError):
        element = openmc.Element('Be')
        element.expand(70.0, 'ao', 4.0, 'Be9')

    # 3 Isotope Element
    with raises(ValueError):
        element = openmc.Element('Cr')
        element.expand(70.0, 'ao', 4.0, 'Cr52')

    # Non-present Enrichment Target
    with raises(ValueError):
        element = openmc.Element('H')
        element.expand(70.0, 'ao', 4.0, 'H4')

    # Enrichment Procedure for Uranium if not Uranium
    with raises(ValueError):
        element = openmc.Element('Li')
        element.expand(70.0, 'ao', 4.0)

    # Missing Enrichment Target
    with raises(ValueError):
        element = openmc.Element('Li')
        element.expand(70.0, 'ao', 4.0, enrichment_type='ao')

    # Invalid Enrichment Type Entry
    with raises(ValueError):
        element = openmc.Element('Li')
        element.expand(70.0, 'ao', 4.0, 'Li7', 'Grand Moff Tarkin')

    # Trying to enrich Uranium
    with raises(ValueError):
        element = openmc.Element('U')
        element.expand(70.0, 'ao', 4.0, 'U235', 'wo')

    # Trying to enrich Uranium with wrong enrichment_target
    with raises(ValueError):
        element = openmc.Element('U')
        element.expand(70.0, 'ao', 4.0, enrichment_type='ao')


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
