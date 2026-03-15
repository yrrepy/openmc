"""Unit tests for Chain.reduce() isomeric branching targets compatibility."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

import openmc.deplete


def create_simple_chain_with_isomeric():
    """Create a simple test chain with isomeric branching targets.

    Chain structure:
    - Ag109: parent with (n,gamma) reaction
    - Ag110: ground state product
    - Ag110_m1: metastable state product
    """
    chain = openmc.deplete.Chain()

    parent = openmc.deplete.Nuclide('Ag109')
    parent.add_reaction('(n,gamma)', 'Ag110', Q=6.8e6, branching_ratio=1.0)
    chain.add_nuclide(parent)

    target1 = openmc.deplete.Nuclide('Ag110')
    target1.half_life = 24.6 * 3600
    chain.add_nuclide(target1)

    target2 = openmc.deplete.Nuclide('Ag110_m1')
    target2.half_life = 249.79 * 86400
    chain.add_nuclide(target2)

    chain.isomeric_branching_targets = {
        'Ag109': {
            '(n,gamma)': ['Ag110', 'Ag110_m1']
        }
    }

    return chain


def create_chain_with_multiple_reactions():
    """Create chain with multiple reactions having isomeric branching."""
    chain = openmc.deplete.Chain()

    parent = openmc.deplete.Nuclide('Ag109')
    parent.add_reaction('(n,gamma)', 'Ag110', Q=6.8e6, branching_ratio=1.0)
    parent.add_reaction('(n,2n)', 'Ag108', Q=-9.5e6, branching_ratio=1.0)
    chain.add_nuclide(parent)

    for name, hl in [('Ag110', None), ('Ag110_m1', 249.79 * 86400),
                     ('Ag108', None), ('Ag108_m1', 438 * 365.25 * 86400)]:
        nuc = openmc.deplete.Nuclide(name)
        if hl:
            nuc.half_life = hl
        chain.add_nuclide(nuc)

    chain.isomeric_branching_targets = {
        'Ag109': {
            '(n,gamma)': ['Ag110', 'Ag110_m1'],
            '(n,2n)': ['Ag108', 'Ag108_m1']
        }
    }

    return chain


def create_chain_with_three_targets():
    """Create chain with three isomeric targets."""
    chain = openmc.deplete.Chain()

    parent = openmc.deplete.Nuclide('Cd110')
    parent.add_reaction('(n,gamma)', 'Cd111', Q=5e6, branching_ratio=1.0)
    chain.add_nuclide(parent)

    for name, hl in [('Cd111', None), ('Cd111_m1', 1000), ('Cd111_m2', 10000)]:
        nuc = openmc.deplete.Nuclide(name)
        if hl:
            nuc.half_life = hl
        chain.add_nuclide(nuc)

    chain.isomeric_branching_targets = {
        'Cd110': {
            '(n,gamma)': ['Cd111', 'Cd111_m1', 'Cd111_m2']
        }
    }

    return chain


# ==================== Unit Tests ====================

def test_reduce_preserves_targets_all_retained():
    """Test that target list is preserved when all targets are retained."""
    chain = create_simple_chain_with_isomeric()

    reduced = chain.reduce(['Ag109', 'Ag110', 'Ag110_m1'])

    assert reduced.isomeric_branching_targets is not None
    assert 'Ag109' in reduced.isomeric_branching_targets
    assert '(n,gamma)' in reduced.isomeric_branching_targets['Ag109']
    assert reduced.isomeric_branching_targets['Ag109']['(n,gamma)'] == ['Ag110', 'Ag110_m1']
    assert reduced.reduce_pruned_targets is None


def test_reduce_drops_parent_excluded():
    """Test that targets are dropped when parent nuclide is excluded."""
    chain = create_simple_chain_with_isomeric()

    reduced = chain.reduce(['Ag110', 'Ag110_m1'])

    if reduced.isomeric_branching_targets is not None:
        assert 'Ag109' not in reduced.isomeric_branching_targets
    else:
        assert reduced.isomeric_branching_targets is None


def test_reduce_drops_all_targets_excluded():
    """Test that entry is dropped when all targets are excluded."""
    chain = create_simple_chain_with_isomeric()

    reduced = chain.reduce(['Ag109'], level=0, keep_isomeric_siblings=False)

    assert len(reduced.nuclides) == 1
    assert reduced.nuclides[0].name == 'Ag109'

    if reduced.isomeric_branching_targets is not None and 'Ag109' in reduced.isomeric_branching_targets:
        assert '(n,gamma)' not in reduced.isomeric_branching_targets['Ag109']


def test_reduce_prunes_partial_exclusion():
    """Test that excluded targets are pruned and tracked in reduce_pruned_targets."""
    chain = create_simple_chain_with_isomeric()

    reduced = chain.reduce(['Ag109', 'Ag110'], keep_isomeric_siblings=False)

    assert reduced.isomeric_branching_targets is not None
    assert reduced.isomeric_branching_targets['Ag109']['(n,gamma)'] == ['Ag110']

    # Pruned target should be tracked
    assert reduced.reduce_pruned_targets is not None
    assert reduced.reduce_pruned_targets['Ag109']['(n,gamma)'] == ['Ag110_m1']


def test_reduce_prunes_two_of_three():
    """Test pruning when two of three targets are kept."""
    chain = create_chain_with_three_targets()

    reduced = chain.reduce(['Cd110', 'Cd111', 'Cd111_m2'], keep_isomeric_siblings=False)

    targets = reduced.isomeric_branching_targets['Cd110']['(n,gamma)']
    assert targets == ['Cd111', 'Cd111_m2']

    pruned = reduced.reduce_pruned_targets['Cd110']['(n,gamma)']
    assert pruned == ['Cd111_m1']


def test_reduce_handles_multiple_reactions():
    """Test that multiple reactions are handled independently."""
    chain = create_chain_with_multiple_reactions()

    reduced = chain.reduce(['Ag109', 'Ag110', 'Ag108', 'Ag108_m1'],
                           keep_isomeric_siblings=False)

    # (n,gamma) should have Ag110_m1 pruned
    gamma_targets = reduced.isomeric_branching_targets['Ag109']['(n,gamma)']
    assert gamma_targets == ['Ag110']

    # (n,2n) should be unchanged (all targets retained)
    n2n_targets = reduced.isomeric_branching_targets['Ag109']['(n,2n)']
    assert set(n2n_targets) == {'Ag108', 'Ag108_m1'}


def test_reduce_backward_compatible_no_isomeric():
    """Test that reduce works for chains without isomeric data."""
    chain = openmc.deplete.Chain()

    u235 = openmc.deplete.Nuclide('U235')
    u235.add_reaction('(n,gamma)', 'U236', Q=6.5e6, branching_ratio=1.0)
    chain.add_nuclide(u235)

    u236 = openmc.deplete.Nuclide('U236')
    chain.add_nuclide(u236)

    chain.isomeric_branching_targets = None

    reduced = chain.reduce(['U235', 'U236'])

    assert reduced.isomeric_branching_targets is None
    assert reduced.reduce_pruned_targets is None


def test_reduce_level_zero():
    """Test that level=0 reduction drops targets when siblings excluded."""
    chain = create_simple_chain_with_isomeric()

    reduced = chain.reduce(['Ag109'], level=0, keep_isomeric_siblings=False)

    assert len(reduced.nuclides) == 1
    assert reduced.nuclides[0].name == 'Ag109'

    if reduced.isomeric_branching_targets is not None:
        if 'Ag109' in reduced.isomeric_branching_targets:
            assert '(n,gamma)' not in reduced.isomeric_branching_targets['Ag109']


# ==================== Integration Tests ====================

def test_export_import_roundtrip_with_isomeric():
    """Test that isomeric targets survive export and reload."""
    chain = create_simple_chain_with_isomeric()

    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "test_chain.xml"

        chain.export_to_xml(xml_path)
        reloaded = openmc.deplete.Chain.from_xml(xml_path)

        assert reloaded.isomeric_branching_targets is not None
        assert 'Ag109' in reloaded.isomeric_branching_targets
        assert '(n,gamma)' in reloaded.isomeric_branching_targets['Ag109']
        assert reloaded.isomeric_branching_targets['Ag109']['(n,gamma)'] == ['Ag110', 'Ag110_m1']


def test_export_reduced_chain_roundtrip():
    """Test that reduced chains with pruned targets can be exported and reloaded."""
    chain = create_simple_chain_with_isomeric()

    reduced = chain.reduce(['Ag109', 'Ag110'], keep_isomeric_siblings=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "reduced_chain.xml"

        reduced.export_to_xml(xml_path)
        reloaded = openmc.deplete.Chain.from_xml(xml_path)

        assert reloaded.isomeric_branching_targets is not None
        assert reloaded.isomeric_branching_targets['Ag109']['(n,gamma)'] == ['Ag110']
        # reduce_pruned_targets is transient — not preserved in XML
        assert reloaded.reduce_pruned_targets is None


def test_form_matrix_after_reduce_with_isomeric():
    """Test that form_matrix() works correctly after reduce with isomeric data."""
    from openmc.deplete import ReactionRates

    chain = create_simple_chain_with_isomeric()

    reduced = chain.reduce(['Ag109', 'Ag110'], keep_isomeric_siblings=False)

    nuclides = [n.name for n in reduced.nuclides]
    reactions = list(reduced.reactions)

    rates = ReactionRates(['mat1'], nuclides, reactions)
    rates[:] = 1e-10

    rates_2d = rates[0]

    # Runtime isomeric branching (flux-weighted, computed by helper)
    isomeric_branching = {
        'Ag109': {
            '(n,gamma)': {
                'Ag110': 1.0
            }
        }
    }

    try:
        matrix = reduced.form_matrix(rates_2d, isomeric_branching=isomeric_branching)
        assert matrix is not None
        assert matrix.shape[0] == len(nuclides)
    except KeyError as e:
        pytest.fail(f"form_matrix raised KeyError after reduce: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
