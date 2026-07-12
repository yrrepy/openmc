"""Unit tests for Chain.reduce() isomeric sibling keeping feature."""

import pytest
import tempfile
from pathlib import Path

import openmc.deplete


def create_chain_with_siblings():
    """Create a test chain with isomeric siblings for testing expansion.

    Chain structure designed to test sibling expansion:
    - Ag108: parent that produces Ag109 (ground state) via (n,gamma)
    - Ag109: ground state (directly reachable from Ag108)
    - Ag109_m1: metastable state (NOT directly reachable, sibling of Ag109)
    - Pd108: another parent that has isomeric branching to Ag109/Ag109_m1

    The key: Ag108 only reaches Ag109, not Ag109_m1. With keep_isomeric_siblings=True,
    Ag109_m1 should be included when starting from Ag108.
    """
    chain = openmc.deplete.Chain()

    # First parent - only produces ground state
    ag108 = openmc.deplete.Nuclide('Ag108')
    ag108.add_reaction('(n,gamma)', 'Ag109', Q=5e6, branching_ratio=1.0)
    chain.add_nuclide(ag108)

    # Ground state (directly reachable from Ag108)
    ag109 = openmc.deplete.Nuclide('Ag109')
    ag109.half_life = 100.0
    chain.add_nuclide(ag109)

    # Metastable state (NOT directly reachable from Ag108)
    ag109_m1 = openmc.deplete.Nuclide('Ag109_m1')
    ag109_m1.half_life = 4.9
    ag109_m1.add_decay_mode('IT', 'Ag109', 1.0)  # Isomeric transition to ground
    chain.add_nuclide(ag109_m1)

    # Another parent that has isomeric branching to Ag109 family
    pd108 = openmc.deplete.Nuclide('Pd108')
    pd108.add_reaction('(n,p)', 'Ag109', Q=-2e6, branching_ratio=1.0)
    chain.add_nuclide(pd108)

    # Add isomeric branching data - Pd108 produces both states
    chain.isomeric_branching_targets = {
        'Pd108': {
            '(n,p)': ['Ag109', 'Ag109_m1']
        }
    }

    return chain


def create_complex_sibling_chain():
    """Create a more complex chain with multiple isomeric families."""
    chain = openmc.deplete.Chain()

    # Family 1: Ag109 -> Ag110/Ag110_m1
    ag109 = openmc.deplete.Nuclide('Ag109')
    ag109.add_reaction('(n,gamma)', 'Ag110', Q=6.8e6, branching_ratio=1.0)
    chain.add_nuclide(ag109)

    ag110 = openmc.deplete.Nuclide('Ag110')
    ag110.half_life = 24.6
    ag110.add_reaction('(n,gamma)', 'Ag111', Q=5.5e6, branching_ratio=1.0)
    chain.add_nuclide(ag110)

    ag110_m1 = openmc.deplete.Nuclide('Ag110_m1')
    ag110_m1.half_life = 249.79 * 86400
    ag110_m1.add_reaction('(n,gamma)', 'Ag111_m1', Q=5.5e6, branching_ratio=1.0)
    chain.add_nuclide(ag110_m1)

    # Family 2: Ag111/Ag111_m1 (products of Ag110)
    ag111 = openmc.deplete.Nuclide('Ag111')
    ag111.half_life = 7.45 * 86400
    chain.add_nuclide(ag111)

    ag111_m1 = openmc.deplete.Nuclide('Ag111_m1')
    ag111_m1.half_life = 64.8
    chain.add_nuclide(ag111_m1)

    # Add isomeric branching
    chain.isomeric_branching_targets = {
        'Ag109': {
            '(n,gamma)': ['Ag110', 'Ag110_m1']
        },
        'Ag110': {
            '(n,gamma)': ['Ag111', 'Ag111_m1']
        }
    }

    return chain


# ==================== Tests for Policy = False ====================

def test_reduce_siblings_policy_false():
    """Test that policy=False doesn't expand siblings (original behavior)."""
    chain = create_chain_with_siblings()

    # Reduce with policy=False - should NOT include Ag109_m1
    reduced = chain.reduce(['Ag108'], level=1, keep_isomeric_siblings=False)

    # Should have Ag108 and Ag109 only (not Ag109_m1)
    assert len(reduced.nuclides) == 2
    nuclide_names = {n.name for n in reduced.nuclides}
    assert nuclide_names == {'Ag108', 'Ag109'}
    assert 'Ag109_m1' not in nuclide_names


# ==================== Tests for Policy = True ====================

def test_reduce_siblings_policy_true():
    """Test that policy=True always includes all siblings."""
    chain = create_chain_with_siblings()

    # Reduce with policy=True - should include Ag109_m1
    reduced = chain.reduce(['Ag108'], level=1, keep_isomeric_siblings=True)

    # Should have Ag108, Ag109, and Ag109_m1
    assert len(reduced.nuclides) == 3
    nuclide_names = {n.name for n in reduced.nuclides}
    assert nuclide_names == {'Ag108', 'Ag109', 'Ag109_m1'}


def test_reduce_siblings_true_no_branching():
    """policy=True is a no-op without isomeric branching metadata (vanilla parity)."""
    chain = openmc.deplete.Chain()

    # Create a chain without isomeric branching
    cd111 = openmc.deplete.Nuclide('Cd111')
    cd111.add_reaction('(n,gamma)', 'Cd112', Q=5e6, branching_ratio=1.0)
    chain.add_nuclide(cd111)

    cd112 = openmc.deplete.Nuclide('Cd112')
    chain.add_nuclide(cd112)

    cd112_m1 = openmc.deplete.Nuclide('Cd112_m1')
    cd112_m1.half_life = 1000
    chain.add_nuclide(cd112_m1)

    # No isomeric branching data
    chain.isomeric_branching_targets = None

    # Reduce with policy=True; the flag is gated on isomeric metadata, so an
    # unbranched sibling (Cd112_m1) is NOT pulled in -- matches upstream/False.
    reduced = chain.reduce(['Cd111'], level=1, keep_isomeric_siblings=True)
    nuclide_names = {n.name for n in reduced.nuclides}
    assert nuclide_names == {'Cd111', 'Cd112'}
    assert 'Cd112_m1' not in nuclide_names


# ==================== Tests for Type Validation ====================

def test_reduce_siblings_invalid_type():
    """Test that non-bool values raise TypeError."""
    chain = create_chain_with_siblings()

    with pytest.raises(TypeError, match="keep_isomeric_siblings must be bool"):
        chain.reduce(['Ag108'], level=1, keep_isomeric_siblings='invalid')

    with pytest.raises(TypeError, match="keep_isomeric_siblings must be bool"):
        chain.reduce(['Ag108'], level=1, keep_isomeric_siblings='isomeric_branch_siblings')


# ==================== Tests for Pathway Following ====================

def test_siblings_pathways_followed():
    """Test that pathways from added siblings are properly followed."""
    chain = create_complex_sibling_chain()

    # Start from Ag109, level=2
    # Without siblings: Ag109 -> Ag110 -> Ag111
    # With siblings (policy=True): should also get Ag110_m1 -> Ag111_m1
    reduced = chain.reduce(['Ag109'], level=2, keep_isomeric_siblings=True)

    nuclide_names = {n.name for n in reduced.nuclides}

    # Should include all: Ag109, Ag110, Ag110_m1, Ag111, Ag111_m1
    assert nuclide_names == {'Ag109', 'Ag110', 'Ag110_m1', 'Ag111', 'Ag111_m1'}

    # Verify the pathways exist
    ag110_m1 = reduced['Ag110_m1']
    assert any(r.target == 'Ag111_m1' for r in ag110_m1.reactions)


def test_siblings_with_depth_limit():
    """Test sibling expansion includes isomeric targets beyond depth limit.

    With Phase 8 fix, keep_isomeric_siblings=True includes all isomeric
    branching targets even beyond depth limit to avoid renormalization warnings.
    """
    chain = create_complex_sibling_chain()

    # Level=1: Direct products + their isomeric branching targets
    reduced = chain.reduce(['Ag109'], level=1, keep_isomeric_siblings=True)
    nuclide_names = {n.name for n in reduced.nuclides}

    # Phase 8 behavior: Ag110 has isomeric branching to Ag111/Ag111_m1,
    # so with keep_isomeric_siblings=True, all targets are included
    # even though Ag111 family is beyond level=1
    assert nuclide_names == {'Ag109', 'Ag110', 'Ag110_m1', 'Ag111', 'Ag111_m1'}


# ==================== Tests for Isomeric Data Handling ====================

def test_siblings_isomeric_data_preserved():
    """Test that isomeric branching data is preserved for expanded siblings."""
    chain = create_complex_sibling_chain()

    reduced = chain.reduce(['Ag109'], level=2, keep_isomeric_siblings=True)

    # Check that isomeric branching targets are preserved
    assert reduced.isomeric_branching_targets is not None
    assert 'Ag109' in reduced.isomeric_branching_targets
    assert 'Ag110' in reduced.isomeric_branching_targets

    # Check Ag110 targets (should be unchanged)
    targets = reduced.isomeric_branching_targets['Ag110']['(n,gamma)']
    assert set(targets) == {'Ag111', 'Ag111_m1'}


# ==================== Integration Tests ====================

def test_siblings_export_import_roundtrip():
    """Test that chains with expanded siblings can be exported and reloaded."""
    chain = create_chain_with_siblings()

    reduced = chain.reduce(['Ag108'], level=1, keep_isomeric_siblings=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "sibling_chain.xml"

        # Export
        reduced.export_to_xml(xml_path)

        # Reload
        reloaded = openmc.deplete.Chain.from_xml(xml_path)

        # Verify all nuclides present
        original_names = {n.name for n in reduced.nuclides}
        reloaded_names = {n.name for n in reloaded.nuclides}
        assert original_names == reloaded_names

        # Verify isomeric targets preserved
        if reduced.isomeric_branching_targets:
            assert reloaded.isomeric_branching_targets is not None
            for parent in reduced.isomeric_branching_targets:
                assert parent in reloaded.isomeric_branching_targets


def test_default_keeps_all_siblings():
    """Test that default keep_isomeric_siblings=True keeps all siblings."""
    chain = create_chain_with_siblings()

    # Call reduce without the new parameter (should use default=True)
    reduced = chain.reduce(['Ag108'], level=1)

    # Default is True, so should include all siblings
    nuclide_names = {n.name for n in reduced.nuclides}
    assert nuclide_names == {'Ag108', 'Ag109', 'Ag109_m1'}


def test_siblings_form_matrix():
    """Test that form_matrix works with expanded siblings."""
    chain = create_chain_with_siblings()

    reduced = chain.reduce(['Ag108'], level=1, keep_isomeric_siblings=True)

    # Need to use ReactionRates object, not raw numpy array
    from openmc.deplete import ReactionRates

    # Create mock reaction rates
    n_nuclides = len(reduced.nuclides)
    n_reactions = len(reduced.reactions)

    # Create proper ReactionRates object (3D: materials × nuclides × reactions)
    nuclide_names = [n.name for n in reduced.nuclides]
    reaction_names = list(reduced.reactions)  # Convert to list of strings

    # ReactionRates needs at least one material
    rates = ReactionRates(['mat1'], nuclide_names, reaction_names)
    rates[:] = 1e-10  # Fill with small rates

    # Extract the 2D slice for the single material
    rates_2d = rates[0]  # This preserves index_nuc and index_rx attributes

    # Mock isomeric branching (flux-weighted) - not needed for this chain
    isomeric_branching = None

    # Should work without errors
    matrix = reduced.form_matrix(rates_2d, isomeric_branching=isomeric_branching)
    assert matrix.shape[0] == n_nuclides


if __name__ == "__main__":
    pytest.main([__file__, "-v"])