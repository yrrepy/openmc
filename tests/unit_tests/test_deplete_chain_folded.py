"""Tests for folded ``<isomeric_branching>`` chain XML (Phase 1 fold/unfold).

Covers the canonical type-only ``<reaction>`` form whose single
``<isomeric_branching>`` child carries every pathway (ground included) in
index-parallel whitespace-separated lists, its unfold into qualified
:class:`~openmc.deplete.nuclide.ReactionTuple` objects at load, the refold at
export, and back-compat with legacy qualified and GENDF-convention chains.
"""

from pathlib import Path

import lxml.etree as ET
import pytest

from openmc.deplete import Chain
from openmc.deplete.nuclide import Nuclide, ReactionTuple


def _nuclide_from_xml(xml):
    return Nuclide.from_xml(ET.fromstring(xml))


def _rxn_subtree(nuc):
    """Serialize just the ``<reaction>`` children of a nuclide element."""
    elem = nuc.to_xml_element()
    return b"".join(ET.tostring(rx) for rx in elem.findall('reaction')).decode()


# Canonical folded (n,gamma) group: ground In116 (LFS 0) + In116_m1 (LFS 1).
_FOLDED = (
    '<nuclide name="In115" reactions="2">'
    '<reaction type="(n,gamma)">'
    '<isomeric_branching targets="In116 In116_m1" pendf_lfs="0 1"'
    ' Q="6784720.0 6657450.0"/>'
    '</reaction>'
    '</nuclide>'
)

# The same two pathways in the legacy qualified-entry form.
_LEGACY = (
    '<nuclide name="In115" reactions="2">'
    '<reaction type="(n,gamma)" Q="6784720.0" target="In116" pendf_lfs="0"/>'
    '<reaction type="(n,gamma)_m1" Q="6657450.0" target="In116_m1" pendf_lfs="1"/>'
    '</nuclide>'
)


def test_fold_unfold_fold_idempotent():
    """§5.1 fold -> unfold -> fold is byte-stable on the reaction subtree."""
    first = _rxn_subtree(_nuclide_from_xml(_FOLDED))
    second = _rxn_subtree(_nuclide_from_xml(
        '<nuclide name="In115">' + first + '</nuclide>'))
    assert first == second
    # And it is genuinely the folded form, not a stock fallback.
    assert '<isomeric_branching' in first
    assert 'targets="In116 In116_m1"' in first
    assert 'pendf_lfs="0 1"' in first
    # The per-pathway Q list is emitted under the canonical ``Q`` name, never
    # the legacy ``q_values`` alias.
    assert 'Q="6784720.0 6657450.0"' in first
    assert 'q_values' not in first


def test_legacy_and_folded_same_tuples():
    """§5.2 a legacy qualified chain loads to the same tuple set as its fold."""
    legacy = set(_nuclide_from_xml(_LEGACY).reactions)
    folded = set(_nuclide_from_xml(_FOLDED).reactions)
    assert legacy == folded == {
        ReactionTuple("(n,gamma)", "In116", 6784720.0, 1.0, 0),
        ReactionTuple("(n,gamma)_m1", "In116_m1", 6657450.0, 1.0, 1),
    }


def test_gendf_convention_child_loads():
    """§5.3 GENDF child: ``gendf_lfs`` attr, ground in the list, element keeps
    its Q/target; the per-entry Q falls back to the element Q."""
    xml = (
        '<nuclide name="In115" reactions="2">'
        '<reaction type="(n,gamma)" Q="6784720.0" target="In116">'
        '<isomeric_branching targets="In116 In116_m1" gendf_lfs="0 1"/>'
        '</reaction>'
        '</nuclide>'
    )
    assert set(_nuclide_from_xml(xml).reactions) == {
        ReactionTuple("(n,gamma)", "In116", 6784720.0, 1.0, 0),
        ReactionTuple("(n,gamma)_m1", "In116_m1", 6784720.0, 1.0, 1),
    }


def test_metastable_only_child_roundtrips():
    """§5.4 a child with no LFS 0 entry round-trips. The isomer ordinal comes
    from the target suffix, not the LFS value (In116_m2 sits at LFS 4)."""
    xml = (
        '<nuclide name="In115" reactions="2">'
        '<reaction type="(n,gamma)">'
        '<isomeric_branching targets="In116_m1 In116_m2" pendf_lfs="1 4"'
        ' Q="6657450.0 6495070.0"/>'
        '</reaction>'
        '</nuclide>'
    )
    nuc = _nuclide_from_xml(xml)
    assert set(nuc.reactions) == {
        ReactionTuple("(n,gamma)_m1", "In116_m1", 6657450.0, 1.0, 1),
        ReactionTuple("(n,gamma)_m2", "In116_m2", 6495070.0, 1.0, 4),
    }
    # Round-trip preserves the tuple set and stays folded.
    reparsed = _nuclide_from_xml(
        '<nuclide name="In115">' + _rxn_subtree(nuc) + '</nuclide>')
    assert set(reparsed.reactions) == set(nuc.reactions)


def test_missing_pendf_lfs_tolerated():
    """§5.5 no LFS attr -> pendf_lfs None; ground/metastable split falls back
    to the target-name suffix."""
    xml = (
        '<nuclide name="In115" reactions="2">'
        '<reaction type="(n,gamma)">'
        '<isomeric_branching targets="In116 In116_m1"'
        ' Q="6784720.0 6657450.0"/>'
        '</reaction>'
        '</nuclide>'
    )
    assert set(_nuclide_from_xml(xml).reactions) == {
        ReactionTuple("(n,gamma)", "In116", 6784720.0, 1.0, None),
        ReactionTuple("(n,gamma)_m1", "In116_m1", 6657450.0, 1.0, None),
    }


def test_missing_per_pathway_Q_tolerated():
    """§5.5 no per-pathway Q list -> the element Q applies to every entry,
    else 0.0."""
    with_q = (
        '<nuclide name="In115" reactions="2">'
        '<reaction type="(n,gamma)" Q="6784720.0">'
        '<isomeric_branching targets="In116 In116_m1" pendf_lfs="0 1"/>'
        '</reaction>'
        '</nuclide>'
    )
    assert set(_nuclide_from_xml(with_q).reactions) == {
        ReactionTuple("(n,gamma)", "In116", 6784720.0, 1.0, 0),
        ReactionTuple("(n,gamma)_m1", "In116_m1", 6784720.0, 1.0, 1),
    }

    no_q = (
        '<nuclide name="In115" reactions="2">'
        '<reaction type="(n,gamma)">'
        '<isomeric_branching targets="In116 In116_m1" pendf_lfs="0 1"/>'
        '</reaction>'
        '</nuclide>'
    )
    assert set(_nuclide_from_xml(no_q).reactions) == {
        ReactionTuple("(n,gamma)", "In116", 0.0, 1.0, 0),
        ReactionTuple("(n,gamma)_m1", "In116_m1", 0.0, 1.0, 1),
    }


def test_lfs_gt0_without_suffix_raises():
    """An lfs>0 entry whose target lacks an _m suffix is a format error."""
    xml = (
        '<nuclide name="In115" reactions="1">'
        '<reaction type="(n,gamma)">'
        '<isomeric_branching targets="In116 In117" pendf_lfs="0 1"/>'
        '</reaction>'
        '</nuclide>'
    )
    with pytest.raises(ValueError):
        _nuclide_from_xml(xml)


_CHAIN = (
    '<depletion_chain>'
    '<nuclide name="In115" reactions="3">'
    '<reaction type="(n,gamma)">'
    '<isomeric_branching targets="In116 In116_m1 In116_m2" pendf_lfs="0 1 4"'
    ' Q="6784720.0 6657450.0 6495070.0"/>'
    '</reaction>'
    '</nuclide>'
    '<nuclide name="In116" half_life="14.1" decay_modes="1" decay_energy="0.0"'
    ' reactions="0"><decay type="beta-" target="Sn116" branching_ratio="1.0"/>'
    '</nuclide>'
    '<nuclide name="In116_m1" half_life="3257.0" decay_modes="1"'
    ' decay_energy="0.0" reactions="0">'
    '<decay type="beta-" target="Sn116" branching_ratio="1.0"/></nuclide>'
    '<nuclide name="In116_m2" half_life="2.18" decay_modes="1"'
    ' decay_energy="0.0" reactions="0">'
    '<decay type="beta-" target="Sn116" branching_ratio="1.0"/></nuclide>'
    '<nuclide name="Sn116" reactions="0"/>'
    '</depletion_chain>'
)


def test_reduce_unfolded_chain_roundtrips(run_in_tmpdir):
    """§5.6 reduce() on an unfolded chain -> export_to_xml -> from_xml keeps the
    reduced tuple set (through fold on export and unfold on load)."""
    src = Path('chain_folded.xml')
    src.write_text(_CHAIN)
    chain = Chain.from_xml(str(src))

    reduced = chain.reduce(
        ["In115", "In116", "In116_m1", "In116_m2", "Sn116"])
    before = set(reduced["In115"].reactions)
    assert before == {
        ReactionTuple("(n,gamma)", "In116", 6784720.0, 1.0, 0),
        ReactionTuple("(n,gamma)_m1", "In116_m1", 6657450.0, 1.0, 1),
        ReactionTuple("(n,gamma)_m2", "In116_m2", 6495070.0, 1.0, 4),
    }

    out = Path('reduced.xml')
    reduced.export_to_xml(str(out))
    reread = Chain.from_xml(str(out))
    assert set(reread["In115"].reactions) == before


def test_mixed_legacy_group_exports_stock_lossless():
    """A base-type group holding per-level duplicate entries (no LFS)
    alongside the LFS-tagged qualified pair must NOT fold -- complete and
    unambiguous LFS data is a fold precondition. Stock serialization,
    lossless reload (real case: Fe58 (n,p) in Chain_JEFF40-IST)."""
    xml = (
        '<nuclide name="Fe58" reactions="4">'
        '<reaction type="(n,p)" Q="-5464240.0" target="Mn58" pendf_lfs="0"/>'
        '<reaction type="(n,p)_m1" Q="-5536020.0" target="Mn58_m1"'
        ' pendf_lfs="1"/>'
        '<reaction type="(n,p)" Q="-5624240.0" target="Mn58"/>'
        '<reaction type="(n,p)" Q="-5719240.0" target="Mn58"/>'
        '</nuclide>'
    )
    nuc = _nuclide_from_xml(xml)
    sub = _rxn_subtree(nuc)
    assert 'isomeric_branching' not in sub
    reparsed = _nuclide_from_xml('<nuclide name="Fe58">' + sub + '</nuclide>')
    assert reparsed.reactions == nuc.reactions


def test_duplicate_lfs_group_exports_stock():
    """Two members claiming the same LFS is ambiguous -> no fold."""
    xml = (
        '<nuclide name="X1" reactions="2">'
        '<reaction type="(n,2n)" Q="-1.0" target="Y1" pendf_lfs="1"/>'
        '<reaction type="(n,2n)_m1" Q="-2.0" target="Y1_m1" pendf_lfs="1"/>'
        '</nuclide>'
    )
    nuc = _nuclide_from_xml(xml)
    sub = _rxn_subtree(nuc)
    assert 'isomeric_branching' not in sub
    reparsed = _nuclide_from_xml('<nuclide name="X1">' + sub + '</nuclide>')
    assert reparsed.reactions == nuc.reactions


# ---------------------------------------------------------------------------
# Root-attribute (PENDF provenance stamp) round-trip
# ---------------------------------------------------------------------------

def test_root_attrs_survive_roundtrip_and_reduce(run_in_tmpdir):
    """A stamped <depletion_chain> root survives from_xml -> export -> from_xml,
    and rides through reduce() (§ chain-provenance stamp round-trip)."""
    stamped = _CHAIN.replace(
        '<depletion_chain>',
        '<depletion_chain pendf_source="JEFF40-IST.293K.PENDF.h5"'
        ' pendf_library="JEFF-4.0" pendf_nuclides="593">')
    src = Path('chain_stamped.xml')
    src.write_text(stamped)

    chain = Chain.from_xml(str(src))
    assert chain.root_attrs == {
        'pendf_source': 'JEFF40-IST.293K.PENDF.h5',
        'pendf_library': 'JEFF-4.0',
        'pendf_nuclides': '593',
    }

    out = Path('exported.xml')
    chain.export_to_xml(str(out))
    root = ET.parse(str(out)).getroot()
    assert root.get('pendf_library') == 'JEFF-4.0'
    assert root.get('pendf_nuclides') == '593'
    assert root.get('pendf_source') == 'JEFF40-IST.293K.PENDF.h5'
    # Reload preserves the stamp.
    assert Chain.from_xml(str(out)).root_attrs == chain.root_attrs

    # reduce() -> export -> reload keeps the stamp.
    reduced = chain.reduce(
        ["In115", "In116", "In116_m1", "In116_m2", "Sn116"])
    assert reduced.root_attrs == chain.root_attrs
    rout = Path('reduced_stamped.xml')
    reduced.export_to_xml(str(rout))
    assert Chain.from_xml(str(rout)).root_attrs == chain.root_attrs


def test_vanilla_chain_exports_without_root_attrs(run_in_tmpdir):
    """A chain with no stamp emits a bare <depletion_chain> (no spurious attrs,
    output stays byte-identical to the pre-stamp behavior)."""
    src = Path('chain_vanilla.xml')
    src.write_text(_CHAIN)

    chain = Chain.from_xml(str(src))
    assert chain.root_attrs == {}

    out = Path('vanilla_out.xml')
    chain.export_to_xml(str(out))
    assert dict(ET.parse(str(out)).getroot().attrib) == {}
    # A freshly constructed (never-parsed) chain is likewise attribute-free.
    assert Chain().root_attrs == {}


def test_reduce_preserves_pendf_lfs_on_dropped_target():
    """reduce() keeps pendf_lfs when a pathway's target is not retained --
    the PENDF collapse still needs the LFS to bind the MF=10 partial
    (regression: Zn70 (n,2p)_m1 lost its lfs through r2s chain.reduce)."""
    import lxml.etree as _ET
    from openmc.deplete import Chain
    chain = Chain()
    for name, xml in [
        ('In115',
         '<nuclide name="In115" reactions="1">'
         '<reaction type="(n,gamma)">'
         '<isomeric_branching targets="In116 In116_m1" pendf_lfs="0 1"'
         ' Q="6784720.0 6657450.0"/>'
         '</reaction></nuclide>'),
        ('In116', '<nuclide name="In116" half_life="14.1" decay_modes="0" reactions="0"/>'),
        ('In116_m1', '<nuclide name="In116_m1" half_life="3257.0" decay_modes="0" reactions="0"/>'),
    ]:
        chain.add_nuclide(Nuclide.from_xml(_ET.fromstring(xml)))
    reduced = chain.reduce(['In115'], 0)
    rxns = {rx.type: rx for rx in reduced['In115'].reactions}
    assert rxns['(n,gamma)_m1'].target is None
    assert rxns['(n,gamma)_m1'].pendf_lfs == 1
    assert rxns['(n,gamma)'].pendf_lfs == 0
