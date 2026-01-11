"""Unit tests for Chain.reduce() isomeric branching compatibility."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

import openmc.deplete


def create_simple_chain_with_isomeric():
    """Create a simple test chain with isomeric branching data.

    Chain structure:
    - Ag109: parent with (n,gamma) reaction
    - Ag110: ground state product
    - Ag110_m1: metastable state product

    Isomeric branching: Ag109 (n,gamma) -> Ag110 (95%) / Ag110_m1 (5%)
    """
    chain = openmc.deplete.Chain()

    # Parent nuclide
    parent = openmc.deplete.Nuclide('Ag109')
    parent.add_reaction('(n,gamma)', 'Ag110', Q=6.8e6, branching_ratio=1.0)
    chain.add_nuclide(parent)

    # Target nuclides
    target1 = openmc.deplete.Nuclide('Ag110')
    target1.half_life = 24.6 * 3600  # seconds
    chain.add_nuclide(target1)

    target2 = openmc.deplete.Nuclide('Ag110_m1')
    target2.half_life = 249.79 * 86400  # seconds
    chain.add_nuclide(target2)

    # Isomeric branching data
    chain.isomeric_branching = {
        'Ag109': {
            '(n,gamma)': {
                'energies': np.array([1e-5, 1e-4, 1e-3, 1e-2]),
                'targets': ['Ag110', 'Ag110_m1'],
                'branching_ratios': {
                    'Ag110': np.array([0.95, 0.94, 0.93, 0.92]),
                    'Ag110_m1': np.array([0.05, 0.06, 0.07, 0.08])
                }
            }
        }
    }

    return chain


def create_chain_with_multiple_reactions():
    """Create chain with multiple reactions having isomeric branching."""
    chain = openmc.deplete.Chain()

    # Parent
    parent = openmc.deplete.Nuclide('Ag109')
    parent.add_reaction('(n,gamma)', 'Ag110', Q=6.8e6, branching_ratio=1.0)
    parent.add_reaction('(n,2n)', 'Ag108', Q=-9.5e6, branching_ratio=1.0)
    chain.add_nuclide(parent)

    # Targets for (n,gamma)
    target1 = openmc.deplete.Nuclide('Ag110')
    chain.add_nuclide(target1)
    target2 = openmc.deplete.Nuclide('Ag110_m1')
    target2.half_life = 249.79 * 86400
    chain.add_nuclide(target2)

    # Targets for (n,2n)
    target3 = openmc.deplete.Nuclide('Ag108')
    chain.add_nuclide(target3)
    target4 = openmc.deplete.Nuclide('Ag108_m1')
    target4.half_life = 438 * 365.25 * 86400
    chain.add_nuclide(target4)

    # Isomeric branching for both reactions
    chain.isomeric_branching = {
        'Ag109': {
            '(n,gamma)': {
                'energies': np.array([1e-5, 1e-3]),
                'targets': ['Ag110', 'Ag110_m1'],
                'branching_ratios': {
                    'Ag110': np.array([0.95, 0.93]),
                    'Ag110_m1': np.array([0.05, 0.07])
                }
            },
            '(n,2n)': {
                'energies': np.array([1e7, 2e7]),
                'targets': ['Ag108', 'Ag108_m1'],
                'branching_ratios': {
                    'Ag108': np.array([0.70, 0.60]),
                    'Ag108_m1': np.array([0.30, 0.40])
                }
            }
        }
    }

    return chain


def create_chain_with_three_targets():
    """Create chain with three isomeric targets."""
    chain = openmc.deplete.Chain()

    # Parent (using Cd which has real isomeric states)
    parent = openmc.deplete.Nuclide('Cd110')
    parent.add_reaction('(n,gamma)', 'Cd111', Q=5e6, branching_ratio=1.0)
    chain.add_nuclide(parent)

    # Three targets (Cd111 has multiple isomeric levels)
    target1 = openmc.deplete.Nuclide('Cd111')
    chain.add_nuclide(target1)
    target2 = openmc.deplete.Nuclide('Cd111_m1')
    target2.half_life = 1000
    chain.add_nuclide(target2)
    target3 = openmc.deplete.Nuclide('Cd111_m2')
    target3.half_life = 10000
    chain.add_nuclide(target3)

    # Isomeric branching with three targets
    chain.isomeric_branching = {
        'Cd110': {
            '(n,gamma)': {
                'energies': np.array([1e-5, 1e-3]),
                'targets': ['Cd111', 'Cd111_m1', 'Cd111_m2'],
                'branching_ratios': {
                    'Cd111': np.array([0.50, 0.45]),
                    'Cd111_m1': np.array([0.30, 0.35]),
                    'Cd111_m2': np.array([0.20, 0.20])
                }
            }
        }
    }

    return chain


# ==================== Unit Tests ====================

def test_reduce_preserves_isomeric_all_targets_retained():
    """Test that isomeric data is preserved when all targets are retained."""
    chain = create_simple_chain_with_isomeric()

    # Reduce to include all isotopes
    reduced = chain.reduce(['Ag109', 'Ag110', 'Ag110_m1'])

    # Verify isomeric branching is present and unchanged
    assert reduced.isomeric_branching is not None
    assert 'Ag109' in reduced.isomeric_branching
    assert '(n,gamma)' in reduced.isomeric_branching['Ag109']

    iso_data = reduced.isomeric_branching['Ag109']['(n,gamma)']
    assert iso_data['targets'] == ['Ag110', 'Ag110_m1']
    np.testing.assert_array_equal(iso_data['energies'], chain.isomeric_branching['Ag109']['(n,gamma)']['energies'])
    np.testing.assert_array_almost_equal(
        iso_data['branching_ratios']['Ag110'],
        chain.isomeric_branching['Ag109']['(n,gamma)']['branching_ratios']['Ag110']
    )
    np.testing.assert_array_almost_equal(
        iso_data['branching_ratios']['Ag110_m1'],
        chain.isomeric_branching['Ag109']['(n,gamma)']['branching_ratios']['Ag110_m1']
    )


def test_reduce_drops_isomeric_parent_excluded():
    """Test that isomeric data is dropped when parent nuclide is excluded."""
    chain = create_simple_chain_with_isomeric()

    # Reduce to only include targets (no parent)
    reduced = chain.reduce(['Ag110', 'Ag110_m1'])

    # Verify isomeric branching does not include Ag109
    if reduced.isomeric_branching is not None:
        assert 'Ag109' not in reduced.isomeric_branching
    else:
        # Could be None if no other isomeric data
        assert reduced.isomeric_branching is None


def test_reduce_drops_isomeric_all_targets_excluded():
    """Test that isomeric entry is dropped when all targets are excluded."""
    chain = create_simple_chain_with_isomeric()

    # Reduce to only include parent (no targets)
    # Use keep_isomeric_siblings=False to prevent automatic sibling inclusion
    reduced = chain.reduce(['Ag109'], level=0, keep_isomeric_siblings=False)

    # Verify isomeric branching is None or doesn't include this reaction
    if reduced.isomeric_branching is not None and 'Ag109' in reduced.isomeric_branching:
        # The (n,gamma) reaction should not have isomeric branching
        assert '(n,gamma)' not in reduced.isomeric_branching['Ag109']


def test_reduce_renormalizes_partial_exclusion():
    """Test renormalization when one of two targets is excluded."""
    chain = create_simple_chain_with_isomeric()

    # Reduce to exclude metastable state
    # Use False to explicitly exclude siblings and test renormalization
    with pytest.warns(UserWarning, match="Renormalized isomeric branching"):
        reduced = chain.reduce(['Ag109', 'Ag110'], keep_isomeric_siblings=False)

    # Verify isomeric branching is present but renormalized
    assert reduced.isomeric_branching is not None
    assert 'Ag109' in reduced.isomeric_branching
    assert '(n,gamma)' in reduced.isomeric_branching['Ag109']

    iso_data = reduced.isomeric_branching['Ag109']['(n,gamma)']

    # Should only have Ag110 target
    assert iso_data['targets'] == ['Ag110']
    assert 'Ag110_m1' not in iso_data['branching_ratios']

    # Branching ratio should be renormalized to 1.0 at all energies
    np.testing.assert_array_almost_equal(
        iso_data['branching_ratios']['Ag110'],
        np.ones(4),
        decimal=6
    )


def test_reduce_renormalizes_two_of_three():
    """Test renormalization when two of three targets are kept."""
    chain = create_chain_with_three_targets()

    # Reduce to keep Cd110, Cd111, and Cd111_m2 (exclude Cd111_m1)
    # Use False to explicitly control which nuclides are included
    with pytest.warns(UserWarning, match="Renormalized isomeric branching"):
        reduced = chain.reduce(['Cd110', 'Cd111', 'Cd111_m2'], keep_isomeric_siblings=False)

    # Verify renormalization
    iso_data = reduced.isomeric_branching['Cd110']['(n,gamma)']
    assert set(iso_data['targets']) == {'Cd111', 'Cd111_m2'}

    # Original ratios: Cd111: [0.50, 0.45], Cd111_m1: [0.30, 0.35], Cd111_m2: [0.20, 0.20]
    # After excluding Cd111_m1:
    # - Sum of remaining: [0.70, 0.65]
    # - Cd111 renormalized: [0.50/0.70, 0.45/0.65] = [0.714286, 0.692308]
    # - Cd111_m2 renormalized: [0.20/0.70, 0.20/0.65] = [0.285714, 0.307692]

    expected_Cd111 = np.array([0.50/0.70, 0.45/0.65])
    expected_Cd111_m2 = np.array([0.20/0.70, 0.20/0.65])

    np.testing.assert_array_almost_equal(
        iso_data['branching_ratios']['Cd111'],
        expected_Cd111,
        decimal=6
    )
    np.testing.assert_array_almost_equal(
        iso_data['branching_ratios']['Cd111_m2'],
        expected_Cd111_m2,
        decimal=6
    )

    # Verify they sum to 1.0
    total = (iso_data['branching_ratios']['Cd111'] +
             iso_data['branching_ratios']['Cd111_m2'])
    np.testing.assert_array_almost_equal(total, np.ones(2), decimal=6)


def test_reduce_handles_multiple_reactions():
    """Test that multiple reactions are handled independently."""
    chain = create_chain_with_multiple_reactions()

    # Reduce to keep parent and only Ag110 (exclude Ag110_m1)
    # But keep both Ag108 products
    # Use False to explicitly control which nuclides are included
    with pytest.warns(UserWarning):  # Will warn about Ag110_m1 exclusion
        reduced = chain.reduce(['Ag109', 'Ag110', 'Ag108', 'Ag108_m1'], keep_isomeric_siblings=False)

    # (n,gamma) should be renormalized (Ag110_m1 excluded)
    gamma_data = reduced.isomeric_branching['Ag109']['(n,gamma)']
    assert gamma_data['targets'] == ['Ag110']
    np.testing.assert_array_almost_equal(
        gamma_data['branching_ratios']['Ag110'],
        np.ones(2)
    )

    # (n,2n) should be unchanged (all targets retained)
    n2n_data = reduced.isomeric_branching['Ag109']['(n,2n)']
    assert set(n2n_data['targets']) == {'Ag108', 'Ag108_m1'}
    np.testing.assert_array_almost_equal(
        n2n_data['branching_ratios']['Ag108'],
        chain.isomeric_branching['Ag109']['(n,2n)']['branching_ratios']['Ag108']
    )


def test_reduce_backward_compatible_no_isomeric():
    """Test that reduce works for chains without isomeric data."""
    # Create simple chain without isomeric branching
    chain = openmc.deplete.Chain()

    u235 = openmc.deplete.Nuclide('U235')
    u235.add_reaction('(n,gamma)', 'U236', Q=6.5e6, branching_ratio=1.0)
    chain.add_nuclide(u235)

    u236 = openmc.deplete.Nuclide('U236')
    chain.add_nuclide(u236)

    # No isomeric branching
    chain.isomeric_branching = None

    # Reduce should work without errors
    reduced = chain.reduce(['U235', 'U236'])

    # Verify no isomeric branching in reduced chain
    assert reduced.isomeric_branching is None


def test_reduce_level_zero():
    """Test that level=0 reduction drops all isomeric data when siblings excluded."""
    chain = create_simple_chain_with_isomeric()

    # Reduce to level=0 (only initial isotope, no products)
    # Use keep_isomeric_siblings=False to prevent automatic sibling inclusion
    reduced = chain.reduce(['Ag109'], level=0, keep_isomeric_siblings=False)

    # Should have only Ag109
    assert len(reduced.nuclides) == 1
    assert reduced.nuclides[0].name == 'Ag109'

    # Isomeric data should be dropped (no targets in reduced chain)
    if reduced.isomeric_branching is not None:
        if 'Ag109' in reduced.isomeric_branching:
            # Should not have (n,gamma) entry since targets are excluded
            assert '(n,gamma)' not in reduced.isomeric_branching['Ag109']


# ==================== Integration Tests ====================

def test_export_import_roundtrip_with_isomeric():
    """Test that isomeric data survives export and reload."""
    chain = create_simple_chain_with_isomeric()

    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "test_chain.xml"

        # Export
        chain.export_to_xml(xml_path)

        # Reload
        reloaded = openmc.deplete.Chain.from_xml(xml_path)

        # Verify isomeric branching matches
        assert reloaded.isomeric_branching is not None
        assert 'Ag109' in reloaded.isomeric_branching
        assert '(n,gamma)' in reloaded.isomeric_branching['Ag109']

        orig_data = chain.isomeric_branching['Ag109']['(n,gamma)']
        reload_data = reloaded.isomeric_branching['Ag109']['(n,gamma)']

        assert reload_data['targets'] == orig_data['targets']
        np.testing.assert_array_almost_equal(
            reload_data['energies'],
            orig_data['energies']
        )
        for target in orig_data['targets']:
            np.testing.assert_array_almost_equal(
                reload_data['branching_ratios'][target],
                orig_data['branching_ratios'][target],
                decimal=9  # XML precision
            )


def test_export_reduced_chain_roundtrip():
    """Test that reduced chains with renormalized data can be exported and reloaded."""
    chain = create_simple_chain_with_isomeric()

    # Reduce with renormalization
    # Use False to explicitly exclude siblings
    with pytest.warns(UserWarning):
        reduced = chain.reduce(['Ag109', 'Ag110'], keep_isomeric_siblings=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "reduced_chain.xml"

        # Export reduced chain
        reduced.export_to_xml(xml_path)

        # Reload
        reloaded = openmc.deplete.Chain.from_xml(xml_path)

        # Verify renormalized data is preserved
        assert reloaded.isomeric_branching is not None
        reload_data = reloaded.isomeric_branching['Ag109']['(n,gamma)']

        # Should only have Ag110 with ratio 1.0
        assert reload_data['targets'] == ['Ag110']
        np.testing.assert_array_almost_equal(
            reload_data['branching_ratios']['Ag110'],
            np.ones(4),
            decimal=9
        )


def test_form_matrix_after_reduce_with_isomeric():
    """Test that form_matrix() works correctly after reduce with isomeric data."""
    from openmc.deplete import ReactionRates

    chain = create_simple_chain_with_isomeric()

    # Reduce with renormalization
    # Use False to explicitly exclude siblings
    with pytest.warns(UserWarning):
        reduced = chain.reduce(['Ag109', 'Ag110'], keep_isomeric_siblings=False)

    # Create proper ReactionRates object (3D: materials × nuclides × reactions)
    nuclides = [n.name for n in reduced.nuclides]
    reactions = list(reduced.reactions)

    # ReactionRates needs at least one material
    rates = ReactionRates(['mat1'], nuclides, reactions)
    rates[:] = 1e-10  # Fill with small rates

    # Extract the 2D slice for the single material
    rates_2d = rates[0]  # This preserves index_nuc and index_rx attributes

    # Create mock isomeric branching (flux-weighted)
    isomeric_branching = {
        'Ag109': {
            '(n,gamma)': {
                'Ag110': 1.0  # Single value (flux-weighted)
            }
        }
    }

    # Should not raise KeyError
    try:
        matrix = reduced.form_matrix(rates_2d, isomeric_branching=isomeric_branching)
        assert matrix is not None
        assert matrix.shape[0] == len(nuclides)
    except KeyError as e:
        pytest.fail(f"form_matrix raised KeyError after reduce: {e}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
