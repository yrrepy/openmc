"""Tests for the informational ``pendf_lfs`` field on chain reactions.

Split out of ``test_deplete_chain.py`` so the upstream-owned file stays free of
branch-added tests. The ``_TEST_CHAIN``/``simple_chain`` fixture is copied
verbatim from that file (the original stays there for its own tests).
"""

from pathlib import Path

from openmc.mpi import comm
from openmc.deplete import Chain, nuclide
import pytest

from tests import cdtemp

_TEST_CHAIN = """\
<depletion_chain>
  <nuclide name="H1" reactions="0"/>
  <nuclide name="A" half_life="23652.0" decay_modes="2" decay_energy="0.0" reactions="2">
    <decay type="beta1" target="B" branching_ratio="0.6"/>
    <decay type="beta2" target="C" branching_ratio="0.4"/>
    <reaction type="(n,gamma)" Q="0.0" target="C"/>
    <reaction type="(n,p)" Q="0.0" target="B"/>
  </nuclide>
  <nuclide name="B" half_life="32904.0" decay_modes="1" decay_energy="0.0" reactions="2">
    <decay type="beta" target="A" branching_ratio="1.0"/>
    <reaction type="(n,gamma)" Q="0.0" target="C"/>
    <reaction type="(n,d)" Q="0.0" target="A"/>
  </nuclide>
  <nuclide name="C" reactions="3">
    <reaction type="fission" Q="200000000.0"/>
    <reaction type="(n,gamma)" Q="0.0" target="A" branching_ratio="0.7"/>
    <reaction type="(n,gamma)" Q="0.0" target="B" branching_ratio="0.3"/>
    <neutron_fission_yields>
      <energies>0.0253</energies>
      <fission_yields energy="0.0253">
        <products>A B</products>
        <data>0.0292737 0.002566345</data>
      </fission_yields>
    </neutron_fission_yields>
  </nuclide>
</depletion_chain>
"""


@pytest.fixture(scope='module')
def simple_chain():
    with cdtemp():
        with open('chain_test.xml', 'w') as fh:
            fh.write(_TEST_CHAIN)
        yield Chain.from_xml('chain_test.xml')


def test_pendf_lfs_absent_defaults_none(simple_chain):
    """A chain XML without ``pendf_lfs`` (all existing chains) parses with the
    field defaulted to None."""
    for nuc in simple_chain.nuclides:
        for rx in nuc.reactions:
            assert rx.pendf_lfs is None


def test_pendf_lfs_xml_roundtrip(run_in_tmpdir):
    """The informational ``pendf_lfs`` field survives an XML round-trip. The
    two MF=10-backed pathways refold into a single type-only ``<reaction>``
    with an ``<isomeric_branching>`` child (Phase 1 canonical form); the
    fallback ``(n,p)`` stays a stock element."""
    filename = 'pendf_lfs_{}.xml'.format(comm.rank)

    parent = nuclide.Nuclide("In115")
    # MF=10-backed pathways carry an LFS index; the fallback (n,p) does not.
    parent.add_reaction("(n,gamma)", "In116", 6784730.0, 1.0, 0)
    parent.add_reaction("(n,gamma)_m1", "In116_m1", 6657460.0, 1.0, 1)
    parent.add_reaction("(n,p)", "Cd115", 0.0, 1.0)

    chain = Chain()
    chain.nuclides = [parent]
    chain.export_to_xml(filename)

    xml_text = Path(filename).read_text()
    # The (n,gamma) ground + m1 pathways fold into one type-only element with
    # LFS-ordered parallel lists on the isomeric_branching child.
    assert 'type="(n,gamma)"' in xml_text
    assert 'targets="In116 In116_m1"' in xml_text
    assert 'pendf_lfs="0 1"' in xml_text
    assert 'Q="6784730.0 6657460.0"' in xml_text
    # The fallback reaction has no isomeric pathway -> stock element.
    assert '<reaction type="(n,p)" Q="0.0" target="Cd115"/>' in xml_text

    reread = Chain.from_xml(filename)
    rxns = {rx.type: rx.pendf_lfs for rx in reread["In115"].reactions}
    assert rxns["(n,gamma)"] == 0
    assert rxns["(n,gamma)_m1"] == 1
    assert rxns["(n,p)"] is None
    # Read back as int, not str.
    assert isinstance(rxns["(n,gamma)"], int)
