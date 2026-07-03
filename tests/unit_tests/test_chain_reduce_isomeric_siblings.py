"""Unit tests for the Chain.reduce() isomeric sibling keeping feature."""

import pytest

import openmc.deplete


def create_chain_with_siblings():
    """Create a test chain with isomeric siblings for testing expansion.

    Chain structure designed to test sibling expansion:

    - Ag108: parent that produces Ag109 (ground state) via (n,gamma)
    - Ag109: ground state (directly reachable from Ag108)
    - Ag109_m1: metastable state (NOT directly reachable, sibling of Ag109)

    Ag108 only reaches Ag109, not Ag109_m1. With keep_isomeric_siblings=True,
    Ag109_m1 should be included when starting from Ag108.
    """
    chain = openmc.deplete.Chain()

    # Parent - only produces ground state
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

    return chain


def test_reduce_siblings_policy_false():
    """policy=False does not expand siblings (original behavior)."""
    chain = create_chain_with_siblings()

    reduced = chain.reduce(['Ag108'], level=1, keep_isomeric_siblings=False)

    nuclide_names = {n.name for n in reduced.nuclides}
    assert nuclide_names == {'Ag108', 'Ag109'}
    assert 'Ag109_m1' not in nuclide_names


def test_reduce_siblings_policy_true():
    """policy=True includes the metastable sibling of a reached nuclide."""
    chain = create_chain_with_siblings()

    reduced = chain.reduce(['Ag108'], level=1, keep_isomeric_siblings=True)

    nuclide_names = {n.name for n in reduced.nuclides}
    assert nuclide_names == {'Ag108', 'Ag109', 'Ag109_m1'}


def test_reduce_siblings_true_no_branching():
    """policy=True includes siblings for any chain with metastable states."""
    chain = openmc.deplete.Chain()

    cd111 = openmc.deplete.Nuclide('Cd111')
    cd111.add_reaction('(n,gamma)', 'Cd112', Q=5e6, branching_ratio=1.0)
    chain.add_nuclide(cd111)

    cd112 = openmc.deplete.Nuclide('Cd112')
    chain.add_nuclide(cd112)

    cd112_m1 = openmc.deplete.Nuclide('Cd112_m1')
    cd112_m1.half_life = 1000
    chain.add_nuclide(cd112_m1)

    reduced = chain.reduce(['Cd111'], level=1, keep_isomeric_siblings=True)

    nuclide_names = {n.name for n in reduced.nuclides}
    assert nuclide_names == {'Cd111', 'Cd112', 'Cd112_m1'}


def test_reduce_siblings_invalid_type():
    """Non-bool keep_isomeric_siblings raises TypeError."""
    chain = create_chain_with_siblings()

    with pytest.raises(TypeError, match="keep_isomeric_siblings must be bool"):
        chain.reduce(['Ag108'], level=1, keep_isomeric_siblings='invalid')


def test_siblings_pathways_followed():
    """Reactions of reached nuclides to a kept sibling are preserved."""
    chain = create_complex_sibling_chain()

    # Ag109 -> Ag110 -> Ag111 is followed; Ag110_m1 and Ag111_m1 are added as
    # siblings, and Ag110_m1's (n,gamma) -> Ag111_m1 reaction is retained.
    reduced = chain.reduce(['Ag109'], level=2, keep_isomeric_siblings=True)

    nuclide_names = {n.name for n in reduced.nuclides}
    assert nuclide_names == {'Ag109', 'Ag110', 'Ag110_m1', 'Ag111', 'Ag111_m1'}

    ag110_m1 = reduced['Ag110_m1']
    assert any(r.target == 'Ag111_m1' for r in ag110_m1.reactions)


def test_default_keeps_all_siblings():
    """The default (keep_isomeric_siblings=True) keeps all siblings."""
    chain = create_chain_with_siblings()

    reduced = chain.reduce(['Ag108'], level=1)

    nuclide_names = {n.name for n in reduced.nuclides}
    assert nuclide_names == {'Ag108', 'Ag109', 'Ag109_m1'}
