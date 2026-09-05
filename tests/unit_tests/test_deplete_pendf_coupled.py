"""Unit tests for the coupled PENDF flux-collapse reaction rate path.

Covers :class:`~openmc.deplete.pendf.helpers.PendfFluxCollapseHelper` -- the
raw-flux contraction, the per-chunk collapse cache, the nuclide index and its
warnings, the product-qualified (``_mN``) columns, the direct continuous-energy
overlay, the fission-row guard and the flux sanity check -- and the pure option
resolver :func:`~openmc.deplete.pendf.operators._resolve_pendf_flux_options`.

Everything runs on duck-typed libraries and hand-made flux rows: no data files,
no transport, no ``openmc.lib`` session. The helper is wired the way
``test_pendf_flux_collapse_helper.py`` wires the continuous-energy one, by
setting the attributes ``generate_tallies`` would have set.
"""

import warnings
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

import openmc.deplete.microxs as microxs_mod
import openmc.deplete.pendf.chain_check as chain_check_mod
import openmc.deplete.pendf.collapse as collapse_mod
from openmc.deplete.chain import Chain
from openmc.deplete.microxs import _group_average
from openmc.deplete.nuclide import Nuclide
from openmc.deplete.pendf.helpers import PendfFluxCollapseHelper
from openmc.deplete.pendf.operators import _resolve_pendf_flux_options
from openmc.mgxs import GROUP_STRUCTURES, _canonical_group_structure_name

from tests.unit_tests.test_deplete_pendf_pathways import (
    _FakePendf, _chain_from_fake, _const)
from tests.unit_tests.test_get_pendf_microxs_and_flux import _GroupedDuck


EDGES = np.array([0.0, 1.0e3, 1.0e6, 2.0e7])
N_GROUPS = EDGES.size - 1

# Fe56 capture: a non-constant pointwise shape, so the expected rates have to
# come from the group averaging kernel rather than from a constant.
_FE56_GRID = np.array([0.0, 1.0e3, 1.0e6, 2.0e7])
_FE56_XS = np.array([10.0, 4.0, 2.0, 1.0])


def _basic_fake():
    """Pointwise duck with a fissionable U235 and a non-constant Fe56."""
    return _FakePendf(mf3={
        'U235': {102: _const(5.0), 18: _const(2.0)},
        'Fe56': {102: (_FE56_GRID, _FE56_XS)},
    })


def _wire(library, chain, nuclides, scores, base_reactions, fluxes,
          energies=EDGES, n_nucs=None, reactions_direct=None,
          nuclides_direct=None, materials=None, rate_tally_nuclides=None,
          rate_tally_means=None):
    """Wire a PendfFluxCollapseHelper for hand-made fluxes, without tallies."""
    helper = PendfFluxCollapseHelper(
        n_nucs or len(nuclides), len(scores), library, chain, energies,
        reactions=reactions_direct, nuclides=nuclides_direct)
    flux = np.asarray(fluxes, dtype=float)
    helper._materials = materials or [
        SimpleNamespace(id=i + 1, temperature=293.6)
        for i in range(flux.shape[0])]
    helper._scores = list(scores)
    helper._base_reactions = list(base_reactions)
    # The flux tally mean is flat over (materials, groups); the tally duck lets
    # the cache refill after reset_tally_means() exactly as a real one would.
    helper._flux_tally = SimpleNamespace(mean=flux.ravel())
    helper._flux_tally_means_cache = flux.ravel()
    if reactions_direct:
        helper._rate_tally = SimpleNamespace(
            nuclides=list(rate_tally_nuclides or []))
        helper._rate_tally_means_cache = (
            None if rate_tally_means is None
            else np.asarray(rate_tally_means, dtype=float))
    helper.nuclides = list(nuclides)
    return helper


def _fission_chain(names):
    """Chain whose every nuclide carries a bare fission reaction."""
    chain = Chain()
    for name in names:
        nuclide = Nuclide(name)
        nuclide.add_reaction('fission', None, 200.0e6, 1.0)
        chain.add_nuclide(nuclide)
    return chain


# ---------------------------------------------------------------------------
# 1. Raw-flux contraction
# ---------------------------------------------------------------------------

def test_rates_are_sigma_times_raw_flux():
    """Each rate is ``sum_g sigma_g * phi_g`` over the RAW (un-normalized) flux
    rows, per material: two materials with different fluxes give proportionally
    different rates, a reaction the library lacks stays zero, and a zero-flux
    material returns all zeros."""
    fake = _basic_fake()
    nuclides = ['U235', 'Fe56']
    scores = ['(n,gamma)', 'fission']
    fluxes = [[1.0, 2.0, 3.0], [0.5, 0.0, 4.0], [0.0, 0.0, 0.0]]
    helper = _wire(fake, _chain_from_fake(fake, {}), nuclides, scores,
                   ['(n,gamma)', 'fission'], fluxes)

    fe56_g = _group_average(_FE56_GRID, _FE56_XS, EDGES)

    for mat_index, phi in enumerate(np.asarray(fluxes, dtype=float)):
        rates = helper.get_material_rates(mat_index, [0, 1], [0, 1]).copy()
        expected = np.array([
            [5.0 * phi.sum(), 2.0 * phi.sum()],   # U235 capture, fission
            [fe56_g @ phi, 0.0],                  # Fe56 capture, no MT=18
        ])
        np.testing.assert_allclose(rates, expected, rtol=1e-12)

    # The last material has no flux at all, so every rate is exactly zero
    assert not helper.get_material_rates(2, [0, 1], [0, 1]).any()


# ---------------------------------------------------------------------------
# 2. Chunk cache
# ---------------------------------------------------------------------------

def test_collapse_is_cached_per_chunk(monkeypatch):
    """The engine runs once per chunk of materials per transport cycle: two
    materials of one chunk share a single collapse, ``reset_tally_means`` forces
    a fresh one, and a chunk size of 1 collapses per material."""
    spy = mock.MagicMock(wraps=collapse_mod._collapse_pendf_blocks)
    monkeypatch.setattr(collapse_mod, '_collapse_pendf_blocks', spy)

    fake = _basic_fake()
    helper = _wire(fake, _chain_from_fake(fake, {}), ['U235'], ['(n,gamma)'],
                   ['(n,gamma)'], [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    first = helper.get_material_rates(0, [0], [0]).copy()
    second = helper.get_material_rates(1, [0], [0]).copy()
    assert spy.call_count == 1
    np.testing.assert_allclose(second, 2.0 * first)

    # A new transport cycle drops the cache
    helper.reset_tally_means()
    helper.get_material_rates(0, [0], [0])
    assert spy.call_count == 2

    # One material per chunk means one collapse per material
    monkeypatch.setattr(microxs_mod, '_COLLAPSE_CHUNK_SIZE', 1)
    helper.reset_tally_means()
    helper.get_material_rates(0, [0], [0])
    helper.get_material_rates(1, [0], [0])
    assert spy.call_count == 4


# ---------------------------------------------------------------------------
# 3. Nuclide index growth and the missing-nuclide warning
# ---------------------------------------------------------------------------

def test_nuclide_growth_and_missing_warning():
    """Growing the nuclide list rebuilds the index and clears the cache, so the
    added nuclide gets its own (non-stale) rates; a nuclide the library lacks
    keeps a zero row and warns exactly once per distinct missing set."""
    fake = _basic_fake()
    chain = _chain_from_fake(fake, {})
    helper = _wire(fake, chain, ['U235'], ['(n,gamma)'], ['(n,gamma)'],
                   [[1.0, 1.0, 1.0]], n_nucs=3)

    rates = helper.get_material_rates(0, [0], [0]).copy()
    assert rates[0, 0] == pytest.approx(15.0)

    # Grow the list; Fe56 must pick up its own cross section, not U235's
    helper.nuclides = ['U235', 'Fe56']
    rates = helper.get_material_rates(0, [0, 1], [0]).copy()
    fe56_g = _group_average(_FE56_GRID, _FE56_XS, EDGES)
    assert rates[0, 0] == pytest.approx(15.0)
    assert rates[1, 0] == pytest.approx(fe56_g.sum())

    # A nuclide the library does not carry warns once and keeps a zero row
    with pytest.warns(UserWarning, match='not in PENDF library'):
        helper.nuclides = ['U235', 'Fe56', 'Xe135']
        rates = helper.get_material_rates(0, [0, 1, 2], [0]).copy()
    assert rates[2, 0] == 0.0

    # The same missing set on the next step is silent
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        helper.nuclides = ['U235', 'Fe56', 'Xe135']
        helper.get_material_rates(0, [0, 1, 2], [0])
    assert not [w for w in caught
                if 'not in PENDF library' in str(w.message)]


# ---------------------------------------------------------------------------
# 4. Product-qualified (_mN) columns
# ---------------------------------------------------------------------------

def test_isomeric_columns_match_the_engine():
    """A ``(n,gamma)_m1`` score lands in its own reaction slot carrying the MF=10
    partial, and every column equals what a direct engine call on the same inputs
    returns -- the ground-by-balance semantics belong to the engine, not here."""
    fake = _FakePendf(
        mf3={'Am241': {102: _const(5.0)}},
        mf10={'Am241': {102: {
            0: ('Am242', _const(4.0)),
            2: ('Am242_m1', _const(1.0)),
        }}})
    chain = _chain_from_fake(fake, {102: '(n,gamma)'})
    scores = ['(n,gamma)', '(n,gamma)_m1']
    phi = np.array([[1.0, 2.0, 3.0]])
    helper = _wire(fake, chain, ['Am241'], scores, ['(n,gamma)'], phi)

    rates = helper.get_material_rates(0, [0], [0, 1]).copy()

    reference = collapse_mod._collapse_pendf_blocks(
        ['Am241'], ['(n,gamma)'], EDGES, fake, chain, phi)[0]
    assert list(reference.reactions) == scores
    np.testing.assert_allclose(rates[0], reference.data[0, :, 0], rtol=1e-12)

    # Ground and metastable partials still add up to the MF=3 total
    assert rates[0, 0] == pytest.approx(4.0 * phi.sum())
    assert rates[0, 1] == pytest.approx(1.0 * phi.sum())


# ---------------------------------------------------------------------------
# 5. Direct continuous-energy overlay
# ---------------------------------------------------------------------------

def test_direct_overlay_replaces_only_its_cells():
    """A direct fission tally on U235 replaces exactly that cell; every other
    (nuclide, reaction) keeps its PENDF value."""
    fake = _basic_fake()
    scores = ['(n,gamma)', 'fission']
    helper = _wire(
        fake, _chain_from_fake(fake, {}), ['U235', 'Fe56'], scores,
        ['(n,gamma)', 'fission'], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        reactions_direct=['fission'], nuclides_direct=['U235'],
        rate_tally_nuclides=['U235'], rate_tally_means=[[7.0], [9.0]])

    rates = helper.get_material_rates(0, [0, 1], [0, 1]).copy()
    fe56_g = _group_average(_FE56_GRID, _FE56_XS, EDGES)
    assert rates[0, 1] == pytest.approx(7.0)            # direct U235 fission
    assert rates[0, 0] == pytest.approx(15.0)           # PENDF U235 capture
    assert rates[1, 0] == pytest.approx(fe56_g.sum())   # PENDF Fe56 capture
    assert rates[1, 1] == 0.0                           # Fe56 has no fission

    # The second material takes its own direct value
    rates = helper.get_material_rates(1, [0, 1], [0, 1]).copy()
    assert rates[0, 1] == pytest.approx(9.0)


def test_direct_reactions_are_validated():
    """A product-qualified or unknown direct reaction is rejected at construction:
    a continuous-energy tally can score neither."""
    fake = _basic_fake()
    with pytest.raises(ValueError, match='product-qualified'):
        PendfFluxCollapseHelper(1, 1, fake, Chain(), EDGES,
                                reactions=['(n,gamma)_m1'])
    with pytest.raises(ValueError, match='not a known reaction'):
        PendfFluxCollapseHelper(1, 1, fake, Chain(), EDGES,
                                reactions=['bogus'])


# ---------------------------------------------------------------------------
# 6. Pathway consistency runs once per helper lifetime
# ---------------------------------------------------------------------------

def test_pathway_consistency_checked_once(monkeypatch):
    """The chain/library pathway gate is a property of the pair, so it runs on the
    first collapse only -- not once per material and not again after a reset."""
    spy = mock.MagicMock(wraps=chain_check_mod._check_pathway_consistency)
    monkeypatch.setattr(chain_check_mod, '_check_pathway_consistency', spy)

    fake = _basic_fake()
    helper = _wire(fake, _chain_from_fake(fake, {}), ['U235'], ['(n,gamma)'],
                   ['(n,gamma)'], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    helper.get_material_rates(0, [0], [0])
    helper.get_material_rates(1, [0], [0])
    helper.reset_tally_means()
    helper.get_material_rates(0, [0], [0])
    assert spy.call_count == 1


# ---------------------------------------------------------------------------
# 7. Fission-row guard
# ---------------------------------------------------------------------------

def test_fission_row_guard():
    """With fission collapsed from PENDF, a library carrying no MT=18 at all for
    the chain's fission nuclides is an error (the fission-Q normalization would
    divide by zero); a partial gap warns; direct-tallied fission silences both.

    The index is built from the ``nuclides`` setter, so both fire inside
    ``_wire`` -- before any transport would have been paid for (test 18)."""
    scores = ['(n,gamma)', 'fission']
    no_18 = _FakePendf(mf3={'U235': {102: _const(5.0)}})
    some_18 = _FakePendf(mf3={'U235': {102: _const(5.0)},
                              'Pu239': {102: _const(5.0), 18: _const(2.0)}})

    # None of the fission nuclides has MT=18 -> hard error
    with pytest.raises(ValueError, match='no MT=18 rows for any'):
        _wire(no_18, _fission_chain(['U235']), ['U235'], scores,
              ['(n,gamma)', 'fission'], [[1.0, 1.0, 1.0]])

    # One of two fission nuclides lacks it -> one warning naming it
    with pytest.warns(UserWarning, match=r'1 of 2 fission nuclide.*U235'):
        _wire(some_18, _fission_chain(['U235', 'Pu239']),
              ['U235', 'Pu239'], scores, ['(n,gamma)', 'fission'],
              [[1.0, 1.0, 1.0]])

    # Fission handled by a direct tally -> the guard does not apply
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        helper = _wire(no_18, _fission_chain(['U235']), ['U235'], scores,
                       ['(n,gamma)', 'fission'], [[1.0, 1.0, 1.0]],
                       reactions_direct=['fission'], nuclides_direct=['U235'],
                       rate_tally_nuclides=['U235'], rate_tally_means=[[7.0]])
        helper._ensure_nuclide_index()
    assert not [w for w in caught if 'MT=18' in str(w.message)]


# ---------------------------------------------------------------------------
# 8. Temperature mismatch
# ---------------------------------------------------------------------------

def test_temperature_mismatch_warns_once():
    """A material more than 1 K from the library's preprocessed temperature warns
    once; matching materials never do."""
    class _TempPendf(_FakePendf):
        temperature = 293.6

    fake = _TempPendf(mf3={'U235': {102: _const(5.0)}})
    helper = _wire(fake, Chain(), ['U235'], ['(n,gamma)'], ['(n,gamma)'],
                   [[1.0, 1.0, 1.0]])

    hot = [SimpleNamespace(id=1, temperature=900.0),
           SimpleNamespace(id=2, temperature=293.6)]
    with pytest.warns(UserWarning, match=r'differ from the PENDF library'):
        helper._check_temperature(hot)

    # Warned once for the run, not once per step
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        helper._check_temperature(hot)
    assert not caught

    # Matching temperatures are silent
    helper = _wire(fake, Chain(), ['U235'], ['(n,gamma)'], ['(n,gamma)'],
                   [[1.0, 1.0, 1.0]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        helper._check_temperature([SimpleNamespace(id=1, temperature=293.6)])
    assert not caught


# ---------------------------------------------------------------------------
# 9. Flux sanity
# ---------------------------------------------------------------------------

def test_non_finite_flux_names_the_material():
    """A NaN flux row is reported by material index instead of poisoning every
    rate of that material silently."""
    fake = _basic_fake()
    helper = _wire(fake, _chain_from_fake(fake, {}), ['U235'], ['(n,gamma)'],
                   ['(n,gamma)'], [[1.0, 1.0, 1.0], [np.nan, 1.0, 1.0]])
    with pytest.raises(ValueError, match='material index 1'):
        helper.get_material_rates(0, [0], [0])


# ---------------------------------------------------------------------------
# 10. Option resolution
# ---------------------------------------------------------------------------

def test_resolve_pendf_flux_options():
    """The resolver settles the library, the group structure and the helper
    keyword arguments, and rejects every unsupported combination."""
    pointwise = _basic_fake()
    grouped = _GroupedDuck(EDGES)

    # E1: the mode needs a library
    with pytest.raises(ValueError, match='requires the pendf_library'):
        _resolve_pendf_flux_options(None, 'pendf-flux', {})

    # E2: a library needs the mode
    with pytest.raises(ValueError, match="requires reaction_rate_mode"):
        _resolve_pendf_flux_options(pointwise, 'flux', {'energies': EDGES})

    # Neither given: the other modes are left alone
    assert _resolve_pendf_flux_options(None, 'flux', {'energies': EDGES}) == \
        (None, None, None)

    # E3: a pointwise library has no structure of its own
    with pytest.raises(ValueError, match="requires reaction_rate_opts\\['energies'\\]"):
        _resolve_pendf_flux_options(pointwise, 'pendf-flux', {})

    # E3: a grouped library cannot be rebinned
    with pytest.raises(ValueError, match='cannot be rebinned'):
        _resolve_pendf_flux_options(
            grouped, 'pendf-flux', {'energies': [0.0, 1.0, 2.0e7]})

    # E5: URR self-shielding is not built yet
    for key in ('urr_material_dilution', 'mat_ssf_nuclides'):
        with pytest.raises(NotImplementedError, match='not supported'):
            _resolve_pendf_flux_options(
                pointwise, 'pendf-flux', {'energies': EDGES, key: True})

    # An unrecognized key names the accepted ones
    with pytest.raises(ValueError, match='Accepted keys'):
        _resolve_pendf_flux_options(
            pointwise, 'pendf-flux', {'energies': EDGES, 'temperature': 900.0})

    # Grouped library: its own edges, and the default helper options
    library, energies, helper_opts = _resolve_pendf_flux_options(
        grouped, 'pendf-flux', None)
    assert library is grouped
    np.testing.assert_array_equal(energies, EDGES)
    assert helper_opts == {'reactions': None, 'nuclides': None,
                           'partial_binding': False}

    # Grouped library plus matching edges is accepted
    _, energies, _ = _resolve_pendf_flux_options(
        grouped, 'pendf-flux', {'energies': list(EDGES)})
    np.testing.assert_array_equal(energies, EDGES)

    # Pointwise library plus an explicit edge array
    opts = {'energies': EDGES, 'reactions': ['fission'],
            'nuclides': ['U235', 'Pu239'], 'partial_binding': True}
    library, energies, helper_opts = _resolve_pendf_flux_options(
        pointwise, 'pendf-flux', opts)
    assert library is pointwise
    np.testing.assert_array_equal(energies, EDGES)
    assert helper_opts == {'reactions': ['fission'],
                           'nuclides': ['U235', 'Pu239'],
                           'partial_binding': True}
    # The caller's dict is never modified
    assert set(opts) == {'energies', 'reactions', 'nuclides',
                         'partial_binding'}

    # Pointwise library plus a group structure name
    _, energies, _ = _resolve_pendf_flux_options(
        pointwise, 'pendf-flux', {'energies': 'CASMO-40'})
    np.testing.assert_array_equal(
        energies,
        np.asarray(GROUP_STRUCTURES[_canonical_group_structure_name('CASMO-40')],
                   dtype=float))


# ---------------------------------------------------------------------------
# 15. A direct reaction with isomeric siblings would double count
# ---------------------------------------------------------------------------

def test_direct_reaction_with_isomer_siblings_rejected():
    """When the chain resolves a reaction into isomeric partials, its PENDF
    column is the GROUND partial: a direct continuous-energy tally would put the
    TOTAL there while the sibling keeps its partial, counting the isomer share
    twice. Rejected at construction, and only for nuclides actually in scope."""
    fake = _FakePendf(
        mf3={'Am241': {102: _const(5.0)},
             'U235': {102: _const(5.0), 18: _const(2.0)}},
        mf10={'Am241': {102: {
            0: ('Am242', _const(4.0)),
            2: ('Am242_m1', _const(1.0)),
        }}})
    chain = _chain_from_fake(fake, {102: '(n,gamma)'})

    with pytest.raises(ValueError, match='isomer-resolved siblings'):
        PendfFluxCollapseHelper(2, 2, fake, chain, EDGES,
                                reactions=['(n,gamma)'])

    # U235 carries no isomeric sibling, so restricting the scope to it is safe
    PendfFluxCollapseHelper(2, 2, fake, chain, EDGES,
                            reactions=['(n,gamma)'], nuclides=['U235'])

    # Fission has no siblings anywhere in this chain
    PendfFluxCollapseHelper(2, 2, fake, chain, EDGES, reactions=['fission'])


# ---------------------------------------------------------------------------
# 16. The pathway check does not depend on which material carries flux
# ---------------------------------------------------------------------------

def _isomer_demanding_chain():
    """Chain demanding a ``(n,gamma)_m1`` partial on Am241."""
    chain = Chain()
    nuclide = Nuclide('Am241')
    nuclide.add_reaction('(n,gamma)', 'Am242', 0.0, 1.0)
    nuclide.add_reaction('(n,gamma)_m1', 'Am242_m1', 0.0, 1.0, pendf_lfs=1)
    chain.add_nuclide(nuclide)
    return chain


def test_pathway_check_is_flux_independent(monkeypatch):
    """A chain demanding an isomeric partial the library cannot serve is a
    property of the pair, not of the flux: the mismatch is caught even when
    material 0 -- the one the old check read -- has no flux at all. A chunk with
    no flux anywhere defers the check to the next chunk instead of skipping it."""
    fake = _FakePendf(mf3={'Am241': {102: _const(5.0)}})
    scores = ['(n,gamma)', '(n,gamma)_m1']
    fluxes = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]

    # Both materials in one chunk: the aggregate sees material 1's rows
    helper = _wire(fake, _isomer_demanding_chain(), ['Am241'], scores,
                   ['(n,gamma)'], fluxes)
    with pytest.raises(ValueError, match='pathway mismatch'):
        helper.get_material_rates(0, [0], [0, 1])

    # One material per chunk: the zero-flux chunk defers, the next one raises
    monkeypatch.setattr(microxs_mod, '_COLLAPSE_CHUNK_SIZE', 1)
    helper = _wire(fake, _isomer_demanding_chain(), ['Am241'], scores,
                   ['(n,gamma)'], fluxes)
    assert not helper.get_material_rates(0, [0], [0, 1]).any()
    assert not helper._checked_pathways
    with pytest.raises(ValueError, match='pathway mismatch'):
        helper.get_material_rates(1, [0], [0, 1])


# ---------------------------------------------------------------------------
# 17. Degenerate energy group boundaries
# ---------------------------------------------------------------------------

def test_resolve_rejects_non_ascending_energies():
    """Descending, too-short and repeated edges are rejected at construction
    rather than after the first transport solve, for a pointwise library and for
    the grouped comparison alike."""
    pointwise = _basic_fake()
    for bad in (EDGES[::-1], [1.0], [0.0, 0.0, 2.0e7]):
        with pytest.raises(ValueError, match='strictly ascending'):
            _resolve_pendf_flux_options(
                pointwise, 'pendf-flux', {'energies': bad})

    # The grouped branch resolves through the same function, so the edges are
    # rejected before the 'cannot be rebinned' comparison
    with pytest.raises(ValueError, match='strictly ascending'):
        _resolve_pendf_flux_options(
            _GroupedDuck(EDGES), 'pendf-flux', {'energies': EDGES[::-1]})


# ---------------------------------------------------------------------------
# 18. The nuclide index is built before the transport, not after
# ---------------------------------------------------------------------------

def test_index_guards_fire_at_the_nuclides_assignment():
    """The operator sets ``nuclides`` before every transport solve and after
    ``generate_tallies``, so the fission-row guard and the missing-nuclide
    warning fire at the assignment -- no transport is paid for first."""
    no_18 = _FakePendf(mf3={'U235': {102: _const(5.0)}})
    helper = PendfFluxCollapseHelper(
        1, 2, no_18, _fission_chain(['U235']), EDGES)
    helper._scores = ['(n,gamma)', 'fission']
    with pytest.raises(ValueError, match='no MT=18 rows for any'):
        helper.nuclides = ['U235']

    fake = _basic_fake()
    helper = PendfFluxCollapseHelper(
        1, 1, fake, _chain_from_fake(fake, {}), EDGES)
    helper._scores = ['(n,gamma)']
    with pytest.warns(UserWarning, match='not in PENDF library'):
        helper.nuclides = ['Xe135']


# ---------------------------------------------------------------------------
# 19. Permuted nuclide and reaction indices
# ---------------------------------------------------------------------------

def test_scatter_under_permuted_indices():
    """The operator's ``nuc_index``/``react_index`` are arbitrary permutations
    into a larger matrix, and a nuclide the library lacks is interleaved: every
    collapsed value must land in its own cell and nothing else may be written."""
    fake = _basic_fake()
    nuclides = ['Xe135', 'Fe56', 'U235']       # Xe135 has no PENDF data
    scores = ['fission', '(n,gamma)']
    phi = np.array([[1.0, 2.0, 3.0]])

    helper = PendfFluxCollapseHelper(
        6, 4, fake, _chain_from_fake(fake, {}), EDGES)
    helper._materials = [SimpleNamespace(id=1, temperature=293.6)]
    helper._scores = list(scores)
    helper._base_reactions = ['(n,gamma)', 'fission']
    helper._flux_tally = SimpleNamespace(mean=phi.ravel())
    helper._flux_tally_means_cache = phi.ravel()
    with pytest.warns(UserWarning, match='not in PENDF library'):
        helper.nuclides = list(nuclides)

    rates = helper.get_material_rates(0, [4, 0, 5], [3, 1]).copy()

    fe56_g = _group_average(_FE56_GRID, _FE56_XS, EDGES)
    expected = np.zeros((6, 4))
    expected[0, 1] = fe56_g @ phi[0]            # Fe56 capture
    expected[5, 1] = 5.0 * phi[0].sum()         # U235 capture
    expected[5, 3] = 2.0 * phi[0].sum()         # U235 fission
    np.testing.assert_allclose(rates, expected, rtol=1e-12)

