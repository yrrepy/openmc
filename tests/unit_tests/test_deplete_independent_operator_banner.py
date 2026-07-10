"""Tests for the IndependentOperator PENDF isomeric-branching setup banner.

Mirrors the GENDF operator's ``[openmc.deplete] Depletion is using ...`` banner
(openmc/deplete/abc.py). The banner is printed once at IndependentOperator
initialization under a two-sided guard: the reduced chain must carry a
``pendf_lfs``-qualified ``_m``-suffixed reaction AND the MicroXS must carry at
least one ``_m``-suffixed reaction name. A stock (unbranched) chain, or a folded
chain whose MicroXS resolved zero qualified rows (all fallbacks), stays silent.
"""

from pathlib import Path

import lxml.etree as ET
import numpy as np

from openmc.deplete import IndependentOperator, MicroXS, Chain
from openmc.deplete.nuclide import Nuclide

CHAIN_PATH = Path(__file__).parents[1] / "chain_simple.xml"
ONE_GROUP_XS = Path(__file__).parents[1] / "micro_xs_simple.csv"

_BANNER = "[openmc.deplete] Depletion is using PENDF Isomeric Branching"


def _folded_chain():
    """Chain with a folded In115 ``(n,gamma)`` isomeric-branching group."""
    chain = Chain()
    for xml in (
        '<nuclide name="In115" reactions="1">'
        '<reaction type="(n,gamma)">'
        '<isomeric_branching targets="In116 In116_m1" pendf_lfs="0 1"'
        ' Q="6784720.0 6657450.0"/>'
        '</reaction></nuclide>',
        '<nuclide name="In116" half_life="14.1" decay_modes="1"'
        ' decay_energy="0.0" reactions="0">'
        '<decay type="beta-" target="Sn116" branching_ratio="1.0"/></nuclide>',
        '<nuclide name="In116_m1" half_life="3257.0" decay_modes="1"'
        ' decay_energy="0.0" reactions="0">'
        '<decay type="beta-" target="Sn116" branching_ratio="1.0"/></nuclide>',
        '<nuclide name="Sn116" reactions="0"/>',
    ):
        chain.add_nuclide(Nuclide.from_xml(ET.fromstring(xml)))
    return chain


def _in115_operator(chain, micro_xs):
    return IndependentOperator.from_nuclides(
        1.0, {'In115': 1.0e20}, 1.0, micro_xs, chain,
        nuc_units='atom/cm3', normalization_mode='source-rate')


def test_banner_prints_for_folded_chain_and_qualified_microxs(capsys):
    """Both sides qualified -> the banner prints exactly once."""
    chain = _folded_chain()
    data = np.array([[[1.0], [0.5]]])
    micro = MicroXS(data, ['In115'], ['(n,gamma)', '(n,gamma)_m1'])
    _in115_operator(chain, micro)
    out = capsys.readouterr().out
    assert out.count(_BANNER) == 1


def test_banner_silent_for_stock_chain(capsys):
    """An unbranched (stock) chain never triggers the banner."""
    chain = Chain.from_xml(CHAIN_PATH)
    micro = MicroXS.from_csv(ONE_GROUP_XS)
    IndependentOperator.from_nuclides(
        1.0, {'U235': 1.0e20, 'U238': 1.0e22}, 1.0, micro, chain,
        nuc_units='atom/cm3', normalization_mode='source-rate')
    assert _BANNER not in capsys.readouterr().out


def test_banner_silent_for_folded_chain_without_qualified_microxs(capsys):
    """Folded chain but the MicroXS resolved no ``_m`` rows (all fallbacks):
    chain side qualifies, MicroXS side does not -> silent. The lone unqualified
    fallback row is zeroed so the pathway-consistency check does not fire."""
    chain = _folded_chain()
    data = np.array([[[0.0]]])
    micro = MicroXS(data, ['In115'], ['(n,gamma)'])
    _in115_operator(chain, micro)
    assert _BANNER not in capsys.readouterr().out
