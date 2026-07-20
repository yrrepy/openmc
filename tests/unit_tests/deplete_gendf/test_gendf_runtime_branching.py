"""Runtime isomeric branching + combined XS/BR acquisition (consolidated).

Merges the operator/pool runtime-branching and combined XS+BR acquisition
tests. Sections:

1. BR computation from GENDF production levels (Python backend, lfs_mapping,
   MT/LFS key encoding)
2. Runtime ``build_runtime_branching`` + warn-once behaviour (+ ELIS ambiguity)
3. Combined XS+BR acquisition loop (``_get_branching_data``, dataclass, MT map,
   C++/Python backend signatures, parser save points)
4. Helper weighting (``IsomericBranchingHelper``)
5. Operator integration (CoupledOperator + IndependentOperator)
6. Pool guard + ``deplete`` signature
7. ``form_matrix``/``form_rxn_matrix`` isomeric branching + ``Chain.reduce`` filtering

Shared mocks come from ``gendf_testing`` (``make_mock_gendf`` for call-assert
tests, ``MockGENDFLibrary`` for the plain operator gate, ``make_mock_chain`` /
``make_isomeric_branching``). The subdir ``conftest`` provides the autouse
warn-dedup reset and the ``ir192_lookup`` fixture.
"""

import inspect
import os
import warnings
from unittest.mock import Mock

import numpy as np
import pytest

import openmc.deplete
from openmc.deplete import CoupledOperator, ReactionRates
from openmc.deplete.chain import Chain
from openmc.deplete.decay_elis import lookup_liso
from openmc.deplete.gendf import (
    IsomericBranching, REACTION_TO_MT, MT_TO_REACTION,
    _PythonGENDFLibrary, build_runtime_branching)
from openmc.deplete.helpers import (
    IsomericBranchingHelper, DirectReactionRateHelper,
    FluxCollapseHelper, GENDFFluxCollapseHelper)
from openmc.deplete.pool import deplete, _accepts_isomeric_branching
from openmc.mgxs import GROUP_STRUCTURES

from .gendf_testing import (
    MockGENDFLibrary, make_mock_gendf, make_isomeric_branching,
    make_mock_chain, bare_coupled_operator)


ENERGIES = GROUP_STRUCTURES['CCFE-709']
NG = len(ENERGIES) - 1
AG109_MASK = (ENERGIES[:-1] >= 1e5) & (ENERGIES[:-1] < 1e7)

# C++ source (one directory deeper than the pre-consolidation location)
_SRC_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')

# 3-group grid; ground XS=[8,4,0], m1 XS=[2,4,0] (runtime build_runtime tests)
ENERGY_BOUNDS_3G = np.array([0.0, 1.0, 2.0, 3.0])
GROUND_M1_LEVELS = [
    (0, 500, np.array([8.0, 4.0, 0.0])),   # ground (LFS=0)
    (1, 501, np.array([2.0, 4.0, 0.0])),   # m1 (LFS=1)
]


def _ir191_branching():
    """Ir191(n,gamma) IsomericBranching over [Ir192, Ir192_m1]."""
    return make_isomeric_branching(
        'Ir191', '(n,gamma)', ['Ir192', 'Ir192_m1'],
        [1e5, 1e6, 5e6], [[.9, .8, .7], [.1, .2, .3]])


def _ag109_branching():
    """Ag109(n,gamma) IsomericBranching over [Ag110, Ag110_m1]."""
    return make_isomeric_branching(
        'Ag109', '(n,gamma)', ['Ag110', 'Ag110_m1'],
        [1e5, 1e6, 5e6, 1e7], [[.9, .8, .7, .6], [.1, .2, .3, .4]])


def _ag109_chain():
    """Mock chain with Ag109(n,gamma) targets and no LFS/embedded data."""
    return make_mock_chain({'Ag109': {'(n,gamma)': ['Ag110', 'Ag110_m1']}})


def _ag109_gendf():
    """Mock GENDF: masked XS (1e5-1e7) + Ag109 branching."""
    return make_mock_gendf(NG, ENERGIES, in_range_mask=AG109_MASK,
                           branching=_ag109_branching())


def _normalize_helper():
    """Bare helper for exercising _normalize_ratios."""
    chain = make_mock_chain({'X': {'(n,gamma)': ['Y', 'Z']}})
    return IsomericBranchingHelper(chain, make_mock_gendf(NG, ENERGIES))


# ============================================================================
# 1. BR computation from GENDF production levels
# ============================================================================

def test_python_backend_patcher_requires_decay_file():
    """Patcher mode (no target_names) with no decay lookup raises."""
    mock_lib = Mock(spec=_PythonGENDFLibrary)
    mock_lib.decay_lookup = None
    with pytest.raises(ValueError, match="Patcher mode requires decay_file"):
        _PythonGENDFLibrary.get_branching_ratios(mock_lib, 'Ir191', 102)


def test_python_backend_runtime_mode():
    """Runtime mode needs no decay file: BR=xs/sum, energies=bounds[:-1]."""
    mock_lib = Mock(spec=_PythonGENDFLibrary)
    mock_lib.n_groups = NG
    mock_lib.energy_bounds = ENERGIES.copy()
    mock_lib._get_production_xs = Mock(return_value=[
        (0, 77192, np.ones(NG) * 5.0),   # ground
        (3, 77192, np.ones(NG) * 1.0),   # m1
    ])
    result = _PythonGENDFLibrary.get_branching_ratios(
        mock_lib, 'Ir191', 102,
        target_names=['Ir192', 'Ir192_m1'], lfs_values=[0, 3])

    assert isinstance(result, IsomericBranching)
    assert result.products == ['Ir192', 'Ir192_m1']
    np.testing.assert_allclose(result.branching_ratios[0], 5.0 / 6.0, rtol=1e-10)
    np.testing.assert_allclose(result.branching_ratios[1], 1.0 / 6.0, rtol=1e-10)
    assert result.energies.shape == (NG,)
    assert result.branching_ratios.shape == (2, NG)
    np.testing.assert_array_equal(result.energies, ENERGIES[:-1])


def test_python_backend_runtime_requires_lfs():
    """Runtime mode without lfs_values raises."""
    mock_lib = Mock(spec=_PythonGENDFLibrary)
    with pytest.raises(ValueError, match="lfs_values required"):
        _PythonGENDFLibrary.get_branching_ratios(
            mock_lib, 'Ir191', 102, target_names=['Ir192'], lfs_values=None)


@pytest.mark.parametrize('names, lfs, expected', [
    (['Ir192', 'Ir192_m1', 'Ir192_m2'], [0, 3, 15], {'Ir192_m1': 3, 'Ir192_m2': 15}),
    (['Ir192'], [0], {}),
    (['X', 'X_m1', 'X_m2'], [0, 1, 3], {'X_m1': 1, 'X_m2': 3}),
    (['Y', 'Y_m1'], [0, 2], {'Y_m1': 2}),
])
def test_lfs_mapping_construction(names, lfs, expected):
    """lfs_mapping keeps only lfs>0 (ground excluded); shared C++/Python rule."""
    mapping = {n: l for n, l in zip(names, lfs) if l > 0}
    assert mapping == expected


def test_mt_lfs_key_no_collision():
    """MT*1000 + LFS is collision-free over valid MT/LFS ranges."""
    seen = set()
    for mt in range(1, 892):
        for lfs in range(0, 51):
            key = mt * 1000 + lfs
            assert key not in seen
            seen.add(key)


@pytest.mark.parametrize('mt, lfs, key', [
    (102, 0, 102000), (102, 3, 102003), (102, 15, 102015),
    (16, 0, 16000), (16, 2, 16002), (891, 50, 891050),
])
def test_lfs_extraction_from_key(mt, lfs, key):
    """LFS recovers from the MT*1000+LFS composite key."""
    assert mt * 1000 + lfs == key
    assert key - mt * 1000 == lfs


# ============================================================================
# 2. Runtime build_runtime_branching + warnings
# ============================================================================

def test_runtime_branching_ground_auto_included():
    """Requesting only m1 still normalizes over ground+m1, not the subset."""
    br = build_runtime_branching(
        GROUND_M1_LEVELS, ['Xx_m1'], [1], ENERGY_BOUNDS_3G, 'Xx0', 102)
    np.testing.assert_allclose(br.branching_ratios[0], [0.2, 0.5, 0.0])


def test_runtime_branching_both_levels_unchanged():
    """Ground+m1 request: BR=xs/total per group, zero group stays zero."""
    br = build_runtime_branching(
        GROUND_M1_LEVELS, ['Xx', 'Xx_m1'], [0, 1], ENERGY_BOUNDS_3G, 'Xx0', 102)
    np.testing.assert_allclose(br.branching_ratios[0], [0.8, 0.5, 0.0])
    np.testing.assert_allclose(br.branching_ratios[1], [0.2, 0.5, 0.0])


def test_runtime_branching_clamps_negative_production():
    """NJOY-noise negative production clamps to 0: no negative ratios, and the
    negative group behaves as zero metastable production."""
    levels = [
        (0, 500, np.array([8.0, 4.0, 5.0])),      # ground
        (1, 501, np.array([2.0, -1e-10, 1.0])),   # m1: tiny negative in group 1
    ]
    br = build_runtime_branching(
        levels, ['Xx', 'Xx_m1'], [0, 1], ENERGY_BOUNDS_3G, 'Xx0', 102)
    assert np.all(br.branching_ratios >= 0.0)
    # Group 1 (clamped m1 -> 0): all yield to ground, none to m1
    np.testing.assert_allclose(br.branching_ratios[0], [0.8, 1.0, 5.0 / 6.0])
    np.testing.assert_allclose(br.branching_ratios[1], [0.2, 0.0, 1.0 / 6.0])


def test_runtime_branching_no_ground_in_file_warns():
    """Metastable-only request with no LFS=0 warns and normalizes the subset."""
    eb = np.array([0.0, 1.0, 2.0])
    levels = [(1, 501, np.array([2.0, 0.0])), (2, 502, np.array([2.0, 0.0]))]
    with pytest.warns(UserWarning, match='no ground-state'):
        br = build_runtime_branching(levels, ['Xx_m1'], [1], eb, 'Zz', 16)
    np.testing.assert_allclose(br.branching_ratios[0], [1.0, 0.0])


def test_runtime_branching_lfs_missing_warns_once():
    """A requested LFS absent from the file -> BR 0 and one warning per (nuc, mt)."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        for _ in range(3):
            br = build_runtime_branching(
                GROUND_M1_LEVELS, ['Xx', 'Xx_m3'], [0, 3],
                ENERGY_BOUNDS_3G, 'Zz', 16)
        msgs = [str(w.message) for w in rec
                if issubclass(w.category, UserWarning)]
    np.testing.assert_allclose(br.branching_ratios[1], [0.0, 0.0, 0.0])
    assert len(msgs) == 1
    assert 'LFS=3' in msgs[0]


def test_runtime_branching_lfs_missing_warns_per_pair():
    """A different (nuclide, mt) produces its own missing-LFS warning."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        build_runtime_branching(GROUND_M1_LEVELS, ['Xx', 'Xx_m3'], [0, 3],
                                ENERGY_BOUNDS_3G, 'Aa', 16)
        build_runtime_branching(GROUND_M1_LEVELS, ['Xx', 'Xx_m3'], [0, 3],
                                ENERGY_BOUNDS_3G, 'Bb', 16)
        msgs = [w for w in rec if issubclass(w.category, UserWarning)]
    assert len(msgs) == 2


def test_build_runtime_branching_three_level():
    """3-level ratios sum to 1.0; lfs_mapping excludes ground; structure is set."""
    energy_bounds = np.linspace(1e-5, 2e7, 6)
    levels = [
        (0, 77192, np.array([5.0, 4.0, 3.0, 2.0, 1.0])),
        (3, 77192, np.array([1.0, 2.0, 3.0, 4.0, 5.0])),
        (15, 77192, np.array([0.5, 1.0, 1.5, 2.0, 2.5])),
    ]
    result = build_runtime_branching(
        levels, ['Ir192', 'Ir192_m1', 'Ir192_m2'], [0, 3, 15],
        energy_bounds, 'Ir191', 102)

    assert result.reaction == '(n,gamma)'
    assert result.mt == 102
    assert result.products == ['Ir192', 'Ir192_m1', 'Ir192_m2']
    assert result.branching_ratios.shape == (3, 5)
    np.testing.assert_allclose(
        result.branching_ratios.sum(axis=0), 1.0, rtol=1e-14)
    assert result.lfs_mapping == {'Ir192_m1': 3, 'Ir192_m2': 15}


def test_build_runtime_branching_empty_levels_none():
    """No production levels -> None."""
    assert build_runtime_branching(
        [], ['Ir192', 'Ir192_m1'], [0, 3],
        np.linspace(1e-5, 2e7, 4), 'Ir191', 102) is None


def test_lookup_liso_ambiguous_warns_once(ir192_lookup):
    """Target ELIS in the m1/m2 overlap band warns once per (Z, A)."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        r1 = lookup_liso(77, 192, 84500.0, ir192_lookup)
        r2 = lookup_liso(77, 192, 84500.0, ir192_lookup)
        msgs = [str(w.message) for w in rec
                if issubclass(w.category, UserWarning)]
    assert r1['status'] == 'matched' and r1['liso'] == 1
    assert r2['status'] == 'matched'
    assert len(msgs) == 1
    assert 'Ambiguous ELIS' in msgs[0]
    assert 'LISO=1' in msgs[0] and 'LISO=2' in msgs[0]


def test_lookup_liso_unambiguous_no_warn(ir192_lookup):
    """A clean m1 match (one candidate within tolerance) does not warn."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        r = lookup_liso(77, 192, 56750.0, ir192_lookup)
        msgs = [w for w in rec if 'Ambiguous' in str(w.message)]
    assert r['status'] == 'matched' and r['liso'] == 1
    assert len(msgs) == 0


# ============================================================================
# 3. Combined XS+BR acquisition loop
# ============================================================================

def test_get_branching_data_embedded_first():
    """Chain-embedded ratios are served directly; GENDF is not consulted."""
    embedded = {
        ('Ir191', '(n,gamma)'): {
            'energies': np.array([1e5, 1e6, 5e6]),
            'targets': ['Ir192', 'Ir192_m1'],
            'branching_ratios': {
                'Ir192': np.array([0.95, 0.85, 0.75]),
                'Ir192_m1': np.array([0.05, 0.15, 0.25]),
            }
        }
    }
    chain = make_mock_chain(
        {'Ir191': {'(n,gamma)': ['Ir192', 'Ir192_m1']}}, embedded=embedded)
    gendf = make_mock_gendf(branching=_ir191_branching())
    helper = IsomericBranchingHelper(chain, gendf)

    data = helper._get_branching_data('Ir191', '(n,gamma)')
    assert isinstance(data, dict)
    np.testing.assert_allclose(data['energies'], [1e5, 1e6, 5e6])
    assert 'Ir192' in data['branching_ratios']
    gendf.get_branching_ratios.assert_not_called()


def test_get_branching_data_gendf_fallback():
    """With LFS but no embedded data, branching comes from GENDF (once)."""
    chain = make_mock_chain(
        {'Ir191': {'(n,gamma)': ['Ir192', 'Ir192_m1']}},
        lfs={'Ir191': {'(n,gamma)': [0, 3]}})
    gendf = make_mock_gendf(branching=_ir191_branching())
    helper = IsomericBranchingHelper(chain, gendf)

    data = helper._get_branching_data('Ir191', '(n,gamma)')
    assert isinstance(data, IsomericBranching)
    gendf.get_branching_ratios.assert_called_once()


def test_get_branching_data_unknown_reaction_none():
    """A reaction absent from REACTION_TO_MT yields None."""
    chain = make_mock_chain({'Ir191': {'(n,gamma)': ['Ir192', 'Ir192_m1']}})
    gendf = make_mock_gendf(branching=_ir191_branching())
    helper = IsomericBranchingHelper(chain, gendf)
    assert helper._get_branching_data('Ir191', '(n,xyzzy)') is None


@pytest.mark.parametrize('exc', [KeyError, ValueError, NotImplementedError])
def test_get_branching_data_catches_all_exceptions(exc):
    """GENDF errors degrade gracefully to None."""
    def _raise(*a, **k):
        raise exc("boom")

    chain = make_mock_chain(
        {'X': {'(n,gamma)': ['Y']}}, lfs={'X': {'(n,gamma)': [0]}})
    gendf = make_mock_gendf(branching=_raise)
    helper = IsomericBranchingHelper(chain, gendf)
    assert helper._get_branching_data('X', '(n,gamma)') is None


def test_get_branching_data_cached():
    """Repeated lookups return the identical cached object; GENDF queried once."""
    chain = make_mock_chain(
        {'Ir191': {'(n,gamma)': ['Ir192', 'Ir192_m1']}},
        lfs={'Ir191': {'(n,gamma)': [0, 3]}})
    gendf = make_mock_gendf(branching=_ir191_branching())
    helper = IsomericBranchingHelper(chain, gendf)

    d1 = helper._get_branching_data('Ir191', '(n,gamma)')
    d2 = helper._get_branching_data('Ir191', '(n,gamma)')
    assert d1 is d2
    assert gendf.get_branching_ratios.call_count == 1


@pytest.mark.parametrize('lfs, exp_targets, exp_lfs', [
    (None, None, None),
    ({'Ir191': {'(n,gamma)': [0, 1]}}, None, None),  # LFS for a different nuclide
    ({'Ag109': {'(n,gamma)': [0, 1]}}, ['Ag110', 'Ag110_m1'], [0, 1]),
])
def test_get_branching_data_lfs_dispatch(lfs, exp_targets, exp_lfs):
    """LFS present for the nuclide -> runtime mode; otherwise patcher mode."""
    chain = make_mock_chain(
        {'Ag109': {'(n,gamma)': ['Ag110', 'Ag110_m1']}}, lfs=lfs)
    gendf = make_mock_gendf(branching=lambda *a, **k: _ag109_branching())
    helper = IsomericBranchingHelper(chain, gendf)

    helper._get_branching_data('Ag109', '(n,gamma)')
    kwargs = gendf.get_branching_ratios.call_args.kwargs
    assert kwargs.get('target_names') == exp_targets
    assert kwargs.get('lfs_values') == exp_lfs


def test_stale_cache_invalidation():
    """Each new helper starts with an empty branching cache."""
    chain = make_mock_chain({'X': {'(n,gamma)': ['Y']}})
    gendf = make_mock_gendf(branching=None)
    h1 = IsomericBranchingHelper(chain, gendf)
    assert len(h1._branching_cache) == 0
    h1._branching_cache[('X', '(n,gamma)')] = 'stale'
    h2 = IsomericBranchingHelper(chain, gendf)
    assert len(h2._branching_cache) == 0


def test_isomeric_branching_to_from_dict():
    """IsomericBranching round-trips through to_dict/from_dict."""
    orig = make_isomeric_branching(
        'Ir191', '(n,gamma)', ['Ir192', 'Ir192_m1'],
        [1e5, 1e6, 5e6, 1e7], [[.9, .8, .7, .6], [.1, .2, .3, .4]])
    restored = IsomericBranching.from_dict(orig.to_dict())

    np.testing.assert_array_equal(restored.energies, orig.energies)
    assert restored.products == orig.products
    np.testing.assert_array_equal(restored.branching_ratios,
                                  orig.branching_ratios)
    assert restored.parent_nuclide == orig.parent_nuclide
    assert restored.reaction == orig.reaction
    assert restored.mt == orig.mt


def test_mt_reaction_roundtrip():
    """MT -> reaction -> MT is the identity for shared entries."""
    for mt, reaction in MT_TO_REACTION.items():
        if reaction in REACTION_TO_MT:
            assert REACTION_TO_MT[reaction] == mt


def test_cpp_get_branching_ratios_runtime_mode():
    """C++ get_branching_ratios exposes target_names and lfs_values params."""
    lib = pytest.importorskip('openmc.lib.gendf')
    sig = inspect.signature(lib.GENDFLibrary.get_branching_ratios)
    assert 'target_names' in sig.parameters
    assert 'lfs_values' in sig.parameters


def test_c_api_get_production_xs_signature():
    """C API _get_production_xs takes nuclide and mt."""
    lib = pytest.importorskip('openmc.lib.gendf')
    params = list(inspect.signature(lib.GENDFLibrary._get_production_xs).parameters)
    assert 'nuclide' in params
    assert 'mt' in params


@pytest.mark.parametrize('field',
                         ['prod_energy_data', 'prod_xs_data', 'prod_izap_data'])
def test_parser_mf10_save_points(field):
    """Parser saves each MF=10 data map at all three transition points."""
    with open(os.path.join(_SRC_DIR, 'gendf_parser.cpp')) as f:
        content = f.read()
    assert content.count(f'result.{field}[key]') == 3


# ============================================================================
# 4. Helper weighting (IsomericBranchingHelper)
# ============================================================================

def test_gendf_library_required():
    """Construction with gendf_library=None raises."""
    chain = _ag109_chain()
    with pytest.raises(ValueError, match="gendf_library is required"):
        IsomericBranchingHelper(chain, None)


def test_negative_flux_raises():
    """Negative flux spectrum raises."""
    chain = _ag109_chain()
    gendf = make_mock_gendf(branching=_ag109_branching())
    helper = IsomericBranchingHelper(chain, gendf)
    with pytest.raises(ValueError, match="negative"):
        helper.weighted_branching_ratios(-np.ones(NG), ENERGIES)


def test_flux_energy_length_mismatch_raises():
    """flux length != len(energy_bins) - 1 raises."""
    chain = _ag109_chain()
    gendf = make_mock_gendf(branching=_ag109_branching())
    helper = IsomericBranchingHelper(chain, gendf)
    with pytest.raises(ValueError, match="groups"):
        helper.weighted_branching_ratios(np.ones(500), ENERGIES)


def test_group_structure_mismatch_raises():
    """GENDF XS groups != flux groups raises."""
    chain = _ag109_chain()
    gendf = make_mock_gendf(NG, ENERGIES, xs=np.ones(500),
                            branching=_ag109_branching())
    helper = IsomericBranchingHelper(chain, gendf)
    with pytest.raises(ValueError, match="group structure mismatch"):
        helper.weighted_branching_ratios(np.ones(NG), ENERGIES)


def test_no_isomeric_targets_returns_empty():
    """Chain with no isomeric targets -> empty result."""
    chain = make_mock_chain(None)
    gendf = make_mock_gendf(NG, ENERGIES)
    helper = IsomericBranchingHelper(chain, gendf)
    assert helper.weighted_branching_ratios(np.ones(NG), ENERGIES) == {}


def test_skip_when_nuclide_not_in_gendf():
    """Nuclide missing from the GENDF library -> skipped, empty result."""
    def _missing(*a, **k):
        raise KeyError("Ag109 not found")

    chain = _ag109_chain()
    gendf = make_mock_gendf(NG, ENERGIES, xs=_missing, branching=None)
    helper = IsomericBranchingHelper(chain, gendf)
    result = helper.weighted_branching_ratios(np.ones(NG), ENERGIES)
    assert result == {} or 'Ag109' not in result


def test_zero_weight_sum_returns_empty():
    """Disjoint sigma and phi (zero total weight) -> empty result."""
    flux = np.zeros(NG)
    flux[:100] = 1.0
    gendf_xs = np.zeros(NG)
    gendf_xs[650:] = 1.0
    chain = _ag109_chain()
    gendf = make_mock_gendf(NG, ENERGIES, xs=gendf_xs,
                            branching=_ag109_branching())
    helper = IsomericBranchingHelper(chain, gendf)
    result = helper.weighted_branching_ratios(flux, ENERGIES)
    assert result == {} or '(n,gamma)' not in result.get('Ag109', {})


def test_activator_no_lfs_graceful():
    """Old-format chain (targets, no LFS) degrades gracefully, not crashes."""
    chain = make_mock_chain({
        'Ag107': {'(n,gamma)': ['Ag108', 'Ag108_m1'],
                  '(n,2n)': ['Ag106', 'Ag106_m1']},
        'Ag109': {'(n,gamma)': ['Ag110', 'Ag110_m1']},
    })

    def br_handler(nuclide, mt, target_names=None, lfs_values=None):
        if target_names is None:
            raise NotImplementedError("C++ patcher mode not supported")
        raise ValueError("should not reach here")

    gendf = make_mock_gendf(NG, ENERGIES, branching=br_handler)
    helper = IsomericBranchingHelper(chain, gendf)
    result = helper.weighted_branching_ratios(np.ones(NG), ENERGIES)
    assert isinstance(result, dict)


def test_weighted_conservation():
    """sigma-phi-weighted ratios sum to 1.0 with energy-varying flux and BR."""
    flux = np.zeros(NG)
    flux[:200] = 1e14
    flux[200:600] = 1e12
    flux[600:] = 1e10
    br_ground = np.zeros(NG)
    br_ground[:400] = 0.9
    br_ground[400:] = 0.5
    br_meta = np.zeros(NG)
    br_meta[:400] = 0.1
    br_meta[400:] = 0.5
    mock_br = make_isomeric_branching(
        'Test', '(n,gamma)', ['GS', 'M1'], ENERGIES[:-1].copy(),
        [br_ground, br_meta])
    chain = make_mock_chain({'Test': {'(n,gamma)': ['GS', 'M1']}},
                            lfs={'Test': {'(n,gamma)': [0, 1]}})
    gendf = make_mock_gendf(NG, ENERGIES, branching=lambda *a, **k: mock_br)
    helper = IsomericBranchingHelper(chain, gendf)

    result = helper.weighted_branching_ratios(flux, ENERGIES)
    if 'Test' in result and '(n,gamma)' in result['Test']:
        total = sum(result['Test']['(n,gamma)'].values())
        assert np.isclose(total, 1.0)


def test_weighted_branching_with_embedded():
    """weighted_branching_ratios works via the chain-embedded dict path."""
    embedded = {
        ('Ir191', '(n,gamma)'): {
            'energies': np.array([1e5, 1e6, 5e6]),
            'targets': ['Ir192', 'Ir192_m1'],
            'branching_ratios': {
                'Ir192': np.array([0.95, 0.85, 0.75]),
                'Ir192_m1': np.array([0.05, 0.15, 0.25]),
            }
        }
    }
    chain = make_mock_chain(
        {'Ir191': {'(n,gamma)': ['Ir192', 'Ir192_m1']}}, embedded=embedded)
    gendf = make_mock_gendf(branching=_ir191_branching())
    helper = IsomericBranchingHelper(chain, gendf)

    result = helper.weighted_branching_ratios(np.ones(NG), ENERGIES)
    assert 'Ir191' in result
    total = sum(result['Ir191']['(n,gamma)'].values())
    assert np.isclose(total, 1.0)


def test_target_filtering_with_reduced_chain():
    """GENDF products are filtered to the chain's target list and renormalized."""
    chain = make_mock_chain({'Ag109': {'(n,gamma)': ['Ag110']}})  # m1 pruned
    gendf = _ag109_gendf()
    helper = IsomericBranchingHelper(chain, gendf)

    result = helper.weighted_branching_ratios(np.ones(NG), ENERGIES)
    if 'Ag109' in result and '(n,gamma)' in result['Ag109']:
        ratios = result['Ag109']['(n,gamma)']
        assert 'Ag110' in ratios
        assert 'Ag110_m1' not in ratios
        assert np.isclose(ratios['Ag110'], 1.0)


def test_energy_validation_cached():
    """Energy validation runs once (flag-based caching)."""
    chain = make_mock_chain({'X': {'(n,gamma)': ['Y']}})
    gendf = make_mock_gendf(NG, ENERGIES)
    helper = IsomericBranchingHelper(chain, gendf)

    assert not helper._energy_validated
    helper.weighted_branching_ratios(np.ones(NG), ENERGIES)
    assert helper._energy_validated
    helper.weighted_branching_ratios(np.ones(NG), ENERGIES)
    assert helper._energy_validated


@pytest.mark.parametrize('iso_indices, below, above', [
    ([0, 0, 0, 0, 0, 1, 1, 2, 2, 2],
     [True] * 4 + [False] * 6, [False] * 10),      # below-range zeroed
    ([0, 0, 1, 1, 2, 2, 2, 2, 2, 2],
     [False] * 10, [False] * 6 + [True] * 4),       # above-range zeroed
])
def test_build_branching_array_no_extrapolation(iso_indices, below, above):
    """_build_branching_array zeroes out-of-range groups, keeps in-range."""
    helper = IsomericBranchingHelper(_ag109_chain(),
                                     make_mock_gendf(branching=_ag109_branching()))
    n_groups = 10
    ratios = np.array([0.9, 0.8, 0.7])
    iso_indices = np.array(iso_indices)
    below = np.array(below)
    above = np.array(above)
    in_range = ~below & ~above

    br = helper._build_branching_array(
        ratios, iso_indices, below, above, in_range, n_groups)
    assert np.all(br[below] == 0.0)
    assert np.all(br[above] == 0.0)
    assert np.any(br[in_range] > 0)


@pytest.mark.parametrize('ratios', [
    {'A': 0.0, 'B': 0.0}, {'A': -0.5, 'B': 0.3}, {}])
def test_normalize_ratios_returns_empty(ratios):
    """Zero/negative/empty totals degrade to {} (not ValueError)."""
    helper = _normalize_helper()
    assert helper._normalize_ratios(ratios, 'Nuc', '(n,gamma)') == {}


def test_normalize_ratios_warns_on_deviation():
    """A >1% deviation from 1.0 warns and renormalizes to 1.0."""
    helper = _normalize_helper()
    with pytest.warns(UserWarning, match="deviates"):
        result = helper._normalize_ratios({'A': 0.5, 'B': 0.3}, 'Nuc', '(n,gamma)')
    assert np.isclose(sum(result.values()), 1.0)


def test_multiple_materials_same_chain():
    """compute_for_materials returns one normalized dict per material."""
    chain = make_mock_chain({'Ag109': {'(n,gamma)': ['Ag110', 'Ag110_m1']}},
                            lfs={'Ag109': {'(n,gamma)': [0, 1]}})
    mock_br = make_isomeric_branching(
        'Ag109', '(n,gamma)', ['Ag110', 'Ag110_m1'], ENERGIES[:-1].copy(),
        [np.full(NG, 0.9), np.full(NG, 0.1)])
    gendf = make_mock_gendf(NG, ENERGIES, branching=lambda *a, **k: mock_br)
    helper = IsomericBranchingHelper(chain, gendf)

    pairs = [(np.ones(NG), ENERGIES), (np.ones(NG) * 2.0, ENERGIES),
             (np.ones(NG) * 0.5, ENERGIES)]
    result = helper.compute_for_materials(pairs)

    assert len(result) == 3
    for mat in result:
        assert 'Ag109' in mat
        assert np.isclose(sum(mat['Ag109']['(n,gamma)'].values()), 1.0)


def test_all_empty_warns_and_returns_none():
    """All materials empty -> warn and return None."""
    chain = make_mock_chain({'X': {'(n,gamma)': ['Y']}},
                            lfs={'X': {'(n,gamma)': [0]}})
    gendf = make_mock_gendf(NG, ENERGIES, branching=None)
    helper = IsomericBranchingHelper(chain, gendf)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        result = helper.compute_for_materials([(np.ones(NG), ENERGIES)])
    assert result is None
    assert any('could not be calculated' in str(w.message) for w in rec)


# ============================================================================
# 5. Operator integration (CoupledOperator + IndependentOperator)
# ============================================================================

def test_flux_collapse_helper_spectrum_surface():
    """FluxCollapseHelper exposes energies and per-material flux spectra."""
    helper = FluxCollapseHelper(2, 2, ENERGIES)
    helper._materials = [Mock(), Mock()]
    flux = np.vstack([np.full(NG, 2.0), np.full(NG, 7.0)])
    helper._flux_tally_means_cache = flux.reshape(-1, 1)

    assert np.array_equal(helper.energies, ENERGIES)
    assert np.allclose(helper.get_flux_spectrum(0), 2.0)
    assert np.allclose(helper.get_flux_spectrum(1), 7.0)


def test_isomeric_enabled_with_matching_energies():
    """flux mode with the GENDF group structure enables isomeric branching."""
    helper = FluxCollapseHelper(1, 1, ENERGIES)
    op = bare_coupled_operator(gendf_library=MockGENDFLibrary(),
                               rate_helper=helper,
                               targets={'Al27': {'(n,gamma)': ['Al28']}})
    op._setup_isomeric_branching()
    assert op._isomeric_helper is not None


def test_isomeric_enabled_with_gendf_flux_helper():
    """gendf-flux mode passes the flux-spectrum gate."""
    helper = GENDFFluxCollapseHelper(1, 1, MockGENDFLibrary())
    op = bare_coupled_operator(gendf_library=MockGENDFLibrary(),
                               rate_helper=helper,
                               targets={'Al27': {'(n,gamma)': ['Al28']}})
    op._setup_isomeric_branching()
    assert op._isomeric_helper is not None


def test_mismatched_energies_raises_with_targets():
    """Wrong flux group structure is a hard error when the chain has data."""
    helper = FluxCollapseHelper(1, 1, np.array([0.0, 1e6, 2e7]))
    op = bare_coupled_operator(gendf_library=MockGENDFLibrary(),
                               rate_helper=helper,
                               targets={'Al27': {'(n,gamma)': ['Al28']}})
    with pytest.raises(ValueError, match='CCFE-709'):
        op._setup_isomeric_branching()


def test_mismatched_energies_disabled_without_targets():
    """Without chain isomeric data, a mismatch just disables the helper."""
    helper = FluxCollapseHelper(1, 1, np.array([0.0, 1e6, 2e7]))
    op = bare_coupled_operator(gendf_library=MockGENDFLibrary(),
                               rate_helper=helper, targets={})
    op._setup_isomeric_branching()
    assert op._isomeric_helper is None


def test_direct_helper_raises_with_targets():
    """A rate helper without a flux spectrum is a hard error when chain has data."""
    helper = DirectReactionRateHelper(1, 1)
    op = bare_coupled_operator(gendf_library=MockGENDFLibrary(),
                               rate_helper=helper,
                               targets={'Al27': {'(n,gamma)': ['Al28']}})
    with pytest.raises(ValueError, match='flux spectrum'):
        op._setup_isomeric_branching()


def test_calculate_isomeric_branching_uses_helper_surface():
    """Flux/energy pairs are built from the rate helper via duck typing."""
    rate_helper = Mock(spec=['energies', 'get_flux_spectrum'])
    rate_helper.energies = ENERGIES
    rate_helper.get_flux_spectrum = lambda i: np.full(NG, float(i + 1))

    op = CoupledOperator.__new__(CoupledOperator)
    op._rate_helper = rate_helper
    op.local_mats = ['1', '2']
    op._mat_index_map = {'1': 0, '2': 1}
    op._isomeric_helper = Mock()
    op._isomeric_helper.compute_for_materials.return_value = {'ok': True}

    result = op._calculate_isomeric_branching()
    assert result == {'ok': True}
    pairs = op._isomeric_helper.compute_for_materials.call_args[0][0]
    assert len(pairs) == 2
    assert np.allclose(pairs[0][0], 1.0)
    assert np.allclose(pairs[1][0], 2.0)
    assert np.array_equal(pairs[0][1], ENERGIES)


def test_coupled_operator_no_targets_silent():
    """Conventional chain (no targets) + no GENDF -> silent, no helper built."""
    op = bare_coupled_operator(gendf_library=None, targets={})
    op._setup_isomeric_branching()
    assert op._isomeric_helper is None
    assert op._isomeric_branching is None
    assert CoupledOperator._calculate_isomeric_branching(op) is None


def test_coupled_operator_raises_without_gendf():
    """Patched chain (targets) + no GENDF -> hard error."""
    op = bare_coupled_operator(gendf_library=None,
                               targets={'Al27': {'(n,gamma)': ['Al28']}})
    with pytest.raises(ValueError, match='no GENDF library'):
        op._setup_isomeric_branching()


def test_hasattr_guard_independent_operator():
    """IndependentOperator raises when the backend lacks get_branching_ratios."""
    from openmc.deplete.independent_operator import IndependentOperator

    mock_op = Mock(spec=IndependentOperator)
    mock_op.chain = make_mock_chain(
        {'Ag109': {'(n,gamma)': ['Ag110', 'Ag110_m1']}})
    mock_op._gendf_library = Mock(spec=[])

    with pytest.raises(ValueError, match='get_branching_ratios'):
        IndependentOperator._setup_isomeric_branching(mock_op)


def test_independent_operator_raises_without_gendf():
    """Patched chain (targets) + no GENDF -> hard error on IndependentOperator."""
    from openmc.deplete.independent_operator import IndependentOperator

    mock_op = Mock(spec=IndependentOperator)
    mock_op.chain = make_mock_chain(
        {'Ag109': {'(n,gamma)': ['Ag110', 'Ag110_m1']}})
    mock_op._gendf_library = None

    with pytest.raises(ValueError, match='no GENDF library'):
        IndependentOperator._setup_isomeric_branching(mock_op)


def test_independent_operator_no_targets_silent():
    """Conventional chain (no targets) + no GENDF -> silent, no helper built."""
    from openmc.deplete.independent_operator import IndependentOperator

    mock_op = Mock(spec=IndependentOperator)
    mock_op.chain = make_mock_chain({})
    mock_op._gendf_library = None

    IndependentOperator._setup_isomeric_branching(mock_op)
    assert mock_op._isomeric_branching is None


# ============================================================================
# 6. Pool guard + deplete signature
# ============================================================================

def test_matrix_funcs_accept_isomeric_branching():
    """All _matrix_funcs integrator kernels accept isomeric_branching."""
    from openmc.deplete import _matrix_funcs

    funcs = [
        _matrix_funcs.celi_f1, _matrix_funcs.celi_f2,
        _matrix_funcs.cf4_f1, _matrix_funcs.cf4_f2,
        _matrix_funcs.cf4_f3, _matrix_funcs.cf4_f4,
        _matrix_funcs.rk4_f1, _matrix_funcs.rk4_f4,
        _matrix_funcs.leqi_f1, _matrix_funcs.leqi_f2,
        _matrix_funcs.leqi_f3, _matrix_funcs.leqi_f4,
    ]
    for fn in funcs:
        assert 'isomeric_branching' in inspect.signature(fn).parameters, \
            f"{fn.__name__} missing isomeric_branching parameter"


def test_form_matrix_accepts_isomeric_branching():
    """Chain.form_matrix exposes the isomeric_branching parameter pool.py needs."""
    assert 'isomeric_branching' in inspect.signature(Chain.form_matrix).parameters


def test_accepts_isomeric_branching_cached():
    """The cached detector recognises the isomeric_branching parameter."""
    def with_iso(chain, rates, fy, isomeric_branching=None):
        pass

    def without_iso(chain, rates, fy):
        pass

    assert _accepts_isomeric_branching(with_iso) is True
    assert _accepts_isomeric_branching(without_iso) is False


def test_deplete_operator_is_keyword_only():
    """operator is keyword-only; matrix_args stays variadic-positional."""
    params = inspect.signature(deplete).parameters
    assert params['operator'].kind is inspect.Parameter.KEYWORD_ONLY
    assert params['matrix_args'].kind is inspect.Parameter.VAR_POSITIONAL
    order = [p for p in params if params[p].kind is
             inspect.Parameter.POSITIONAL_OR_KEYWORD]
    assert order == ['func', 'chain', 'n', 'rates', 'dt', 'current_timestep',
                     'matrix_func', 'transfer_rates', 'external_source_rates']


def test_deplete_positional_operator_goes_to_matrix_args():
    """A 10th positional lands in matrix_args, never in operator (keyword-only)."""
    sig = inspect.signature(deplete)
    bound = sig.bind('f', 'c', 'n', 'r', 'dt', 0, None, None, None, 'EXTRA')
    assert bound.arguments['matrix_args'] == ('EXTRA',)
    assert 'operator' not in bound.arguments


def test_deplete_operator_keyword_flows_through(monkeypatch):
    """operator=... (keyword) feeds isomeric branching into form_matrix."""
    import openmc.deplete.pool as pool_mod
    monkeypatch.setattr(pool_mod, 'USE_MULTIPROCESSING', False)

    class FakeChain:
        fission_yields = [{}]
        seen_iso = 'UNSET'

        def form_matrix(self, rate, fission_yields=None, isomeric_branching=None):
            self.seen_iso = isomeric_branching
            return 'M'

    class FakeOp:
        _isomeric_branching = [{'A': {'r': {'t': 1.0}}}]

    chain = FakeChain()
    deplete(lambda m, n0, t: n0, chain, [np.array([1.0])], [None], 1.0,
            operator=FakeOp())
    assert chain.seen_iso == {'A': {'r': {'t': 1.0}}}


def test_deplete_positional_operator_disables_branching(monkeypatch):
    """Passing the operator positionally (old style) no longer binds operator."""
    import openmc.deplete.pool as pool_mod
    monkeypatch.setattr(pool_mod, 'USE_MULTIPROCESSING', False)

    class FakeChain:
        fission_yields = [{}]
        seen_iso = 'UNSET'

        def form_matrix(self, rate, fission_yields=None, isomeric_branching=None):
            self.seen_iso = isomeric_branching
            return 'M'

    class FakeOp:
        _isomeric_branching = [{'A': {'r': {'t': 1.0}}}]

    chain = FakeChain()
    deplete(lambda m, n0, t: n0, chain, [np.array([1.0])], [None], 1.0,
            0, None, None, None, FakeOp())
    assert chain.seen_iso is None


# ============================================================================
# 7. form_matrix / form_rxn_matrix isomeric branching + Chain.reduce filtering
# ============================================================================

def test_form_matrix_mass_conservation():
    """2-way branching applies iso ratios (over chain br) and conserves mass."""
    chain = openmc.deplete.Chain()
    parent = openmc.deplete.Nuclide('Ir191')
    parent.add_reaction('(n,gamma)', 'Ir192', Q=6e6, branching_ratio=1.0)
    chain.add_nuclide(parent)
    chain.add_nuclide(openmc.deplete.Nuclide('Ir192'))
    chain.add_nuclide(openmc.deplete.Nuclide('Ir192_m1'))

    rates = ReactionRates(['mat1'], ['Ir191', 'Ir192', 'Ir192_m1'], ['(n,gamma)'])
    test_rate = 1.5e-8
    rates[0, 0, 0] = test_rate

    iso_br = {'Ir191': {'(n,gamma)': {'Ir192': 0.65, 'Ir192_m1': 0.35}}}
    dense = chain.form_matrix(rates[0], isomeric_branching=iso_br).toarray()

    i_p = chain.nuclide_dict['Ir191']
    loss = dense[i_p, i_p]
    gain_gs = dense[chain.nuclide_dict['Ir192'], i_p]
    gain_ms = dense[chain.nuclide_dict['Ir192_m1'], i_p]

    assert np.isclose(loss, -test_rate)
    assert np.isclose(gain_gs, test_rate * 0.65)
    assert np.isclose(gain_ms, test_rate * 0.35)
    assert np.isclose(loss + gain_gs + gain_ms, 0.0, atol=1e-20)


def test_form_matrix_self_transmutation_nn_prime():
    """(n,n') self-scatter: metastable gain = rate*br; m1 has decay diagonal."""
    chain = openmc.deplete.Chain()
    in115 = openmc.deplete.Nuclide('In115')
    in115.add_reaction("(n,n')", 'In115', Q=0.0, branching_ratio=0.85)
    chain.add_nuclide(in115)
    in115_m1 = openmc.deplete.Nuclide('In115_m1')
    in115_m1.half_life = 1.61e4
    in115_m1.add_decay_mode('IT', 'In115', 1.0)
    chain.add_nuclide(in115_m1)

    rates = ReactionRates(['mat1'], ['In115', 'In115_m1'], list(chain.reactions))
    rates.set('mat1', 'In115', "(n,n')", 1e-10)

    iso_br = {'In115': {"(n,n')": {'In115': 0.85, 'In115_m1': 0.15}}}
    dense = chain.form_matrix(rates[0], isomeric_branching=iso_br).toarray()

    i_m1 = chain.nuclide_dict['In115_m1']
    assert dense[i_m1, i_m1] < 0  # decay diagonal
    assert np.isclose(dense[i_m1, chain.nuclide_dict['In115']], 1e-10 * 0.15)


def test_form_matrix_nan_br_warns():
    """form_matrix warns on NaN/Inf branching ratios."""
    chain = openmc.deplete.Chain()
    parent = openmc.deplete.Nuclide('Parent')
    parent.add_reaction('(n,gamma)', 'Product', Q=1e6, branching_ratio=1.0)
    chain.add_nuclide(parent)
    chain.add_nuclide(openmc.deplete.Nuclide('Product'))

    rates = ReactionRates(['mat1'], ['Parent', 'Product'], ['(n,gamma)'])
    rates[0, 0, 0] = 1.0
    iso_br = {'Parent': {'(n,gamma)': {'Product': float('nan')}}}
    with pytest.warns(UserWarning, match="Invalid isomeric branching ratio"):
        chain.form_matrix(rates[0], isomeric_branching=iso_br)


def test_form_matrix_missing_target_raises():
    """form_matrix raises KeyError for an isomeric target not in the chain."""
    chain = openmc.deplete.Chain()
    parent = openmc.deplete.Nuclide('Parent')
    parent.add_reaction('(n,gamma)', 'Product', Q=1e6, branching_ratio=1.0)
    chain.add_nuclide(parent)
    chain.add_nuclide(openmc.deplete.Nuclide('Product'))

    rates = ReactionRates(['mat1'], ['Parent', 'Product'], ['(n,gamma)'])
    rates[0, 0, 0] = 1.0
    iso_br = {'Parent': {'(n,gamma)': {'Product': 0.7, 'NonExistent': 0.3}}}
    with pytest.raises(KeyError, match="not in chain"):
        chain.form_matrix(rates[0], isomeric_branching=iso_br)


def test_form_matrix_without_isomeric_uses_chain_br():
    """isomeric_branching=None falls back to the chain's static branching ratio."""
    chain = openmc.deplete.Chain()
    parent = openmc.deplete.Nuclide('Parent')
    parent.add_reaction('(n,gamma)', 'Product', Q=1e6, branching_ratio=0.6)
    chain.add_nuclide(parent)
    chain.add_nuclide(openmc.deplete.Nuclide('Product'))

    rates = ReactionRates(['mat1'], ['Parent', 'Product'], ['(n,gamma)'])
    rates[0, 0, 0] = 2.0
    dense = chain.form_matrix(rates[0], isomeric_branching=None).toarray()

    gain = dense[chain.nuclide_dict['Product'], chain.nuclide_dict['Parent']]
    assert np.isclose(gain, 1.2)  # rate * chain br


def _build_multi_entry_chain():
    """Chain with an official-chain-style branched reaction (two same-type entries)."""
    chain = openmc.deplete.Chain()
    parent = openmc.deplete.Nuclide('Parent')
    parent.add_reaction('(n,gamma)', 'Product', Q=1e6, branching_ratio=0.92)
    parent.add_reaction('(n,gamma)', 'Product_m1', Q=1e6, branching_ratio=0.08)
    chain.add_nuclide(parent)
    chain.add_nuclide(openmc.deplete.Nuclide('Product'))
    chain.add_nuclide(openmc.deplete.Nuclide('Product_m1'))
    return chain


def test_form_rxn_matrix_multi_entry_runtime_mass_conservation():
    """Runtime distribution is applied once per reaction type, conserving mass."""
    chain = _build_multi_entry_chain()
    rates = ReactionRates(['mat1'], ['Parent', 'Product', 'Product_m1'], ['(n,gamma)'])
    rates[0, 0, 0] = 1.0
    iso_br = {'Parent': {'(n,gamma)': {'Product': 0.7, 'Product_m1': 0.3}}}
    dense = chain.form_rxn_matrix(rates[0], isomeric_branching=iso_br).toarray()

    i_p = chain.nuclide_dict['Parent']
    assert np.isclose(dense[chain.nuclide_dict['Product'], i_p], 0.7)
    assert np.isclose(dense[chain.nuclide_dict['Product_m1'], i_p], 0.3)
    assert np.isclose(dense[:, i_p].sum(), 0.0)


def test_form_rxn_matrix_multi_entry_static_br():
    """Without runtime branching each duplicate entry keeps its own static br."""
    chain = _build_multi_entry_chain()
    rates = ReactionRates(['mat1'], ['Parent', 'Product', 'Product_m1'], ['(n,gamma)'])
    rates[0, 0, 0] = 1.0
    dense = chain.form_rxn_matrix(rates[0], isomeric_branching=None).toarray()

    i_p = chain.nuclide_dict['Parent']
    assert np.isclose(dense[chain.nuclide_dict['Product'], i_p], 0.92)
    assert np.isclose(dense[chain.nuclide_dict['Product_m1'], i_p], 0.08)
    assert np.isclose(dense[:, i_p].sum(), 0.0)


def test_reduce_preserves_all_three_attributes():
    """reduce() keeps targets, lfs, and embedded when all nuclides are retained."""
    chain = openmc.deplete.Chain()
    ag109 = openmc.deplete.Nuclide('Ag109')
    ag109.add_reaction('(n,gamma)', 'Ag110', Q=6.8e6, branching_ratio=1.0)
    chain.add_nuclide(ag109)
    ag110 = openmc.deplete.Nuclide('Ag110')
    ag110.half_life = 24.6 * 3600
    chain.add_nuclide(ag110)
    ag110_m1 = openmc.deplete.Nuclide('Ag110_m1')
    ag110_m1.half_life = 249.79 * 86400
    chain.add_nuclide(ag110_m1)

    chain.isomeric_branching_targets = {'Ag109': {'(n,gamma)': ['Ag110', 'Ag110_m1']}}
    chain.isomeric_branching_lfs = {'Ag109': {'(n,gamma)': [0, 3]}}
    chain.isomeric_branching_embedded = {
        ('Ag109', '(n,gamma)'): {
            'energies': np.array([1e5, 1e6]),
            'targets': ['Ag110', 'Ag110_m1'],
            'branching_ratios': {'Ag110': np.array([0.9, 0.8]),
                                 'Ag110_m1': np.array([0.1, 0.2])}}}

    reduced = chain.reduce(['Ag109'])
    assert reduced.isomeric_branching_targets['Ag109']['(n,gamma)'] == \
        ['Ag110', 'Ag110_m1']
    assert reduced.isomeric_branching_lfs['Ag109']['(n,gamma)'] == [0, 3]
    assert ('Ag109', '(n,gamma)') in reduced.isomeric_branching_embedded


def test_reduce_partial_exclusion():
    """Partial exclusion keeps lfs and embedded in sync with pruned targets."""
    chain = openmc.deplete.Chain()
    ag109 = openmc.deplete.Nuclide('Ag109')
    ag109.add_reaction('(n,gamma)', 'Ag110', Q=6.8e6, branching_ratio=1.0)
    chain.add_nuclide(ag109)
    chain.add_nuclide(openmc.deplete.Nuclide('Ag110'))
    ag110_m1 = openmc.deplete.Nuclide('Ag110_m1')
    ag110_m1.half_life = 249.79 * 86400
    chain.add_nuclide(ag110_m1)

    chain.isomeric_branching_targets = {'Ag109': {'(n,gamma)': ['Ag110', 'Ag110_m1']}}
    chain.isomeric_branching_lfs = {'Ag109': {'(n,gamma)': [0, 3]}}
    chain.isomeric_branching_embedded = {
        ('Ag109', '(n,gamma)'): {
            'energies': np.array([1e5, 1e6]),
            'targets': ['Ag110', 'Ag110_m1'],
            'branching_ratios': {'Ag110': np.array([0.9, 0.8]),
                                 'Ag110_m1': np.array([0.1, 0.2])}}}

    reduced = chain.reduce(['Ag109', 'Ag110'], keep_isomeric_siblings=False)
    assert reduced.isomeric_branching_targets['Ag109']['(n,gamma)'] == ['Ag110']
    assert reduced.isomeric_branching_lfs['Ag109']['(n,gamma)'] == [0]

    emb = reduced.isomeric_branching_embedded[('Ag109', '(n,gamma)')]
    assert emb['targets'] == ['Ag110']
    assert 'Ag110' in emb['branching_ratios']
    assert 'Ag110_m1' not in emb['branching_ratios']


def test_chain_xml_roundtrip_lfs():
    """gendf_lfs survives XML export/import."""
    import tempfile
    from pathlib import Path

    chain = openmc.deplete.Chain()
    ag109 = openmc.deplete.Nuclide('Ag109')
    ag109.add_reaction('(n,gamma)', 'Ag110', Q=6.8e6, branching_ratio=1.0)
    chain.add_nuclide(ag109)
    chain.add_nuclide(openmc.deplete.Nuclide('Ag110'))
    ag110_m1 = openmc.deplete.Nuclide('Ag110_m1')
    ag110_m1.half_life = 249.79 * 86400
    chain.add_nuclide(ag110_m1)

    chain.isomeric_branching_targets = {'Ag109': {'(n,gamma)': ['Ag110', 'Ag110_m1']}}
    chain.isomeric_branching_lfs = {'Ag109': {'(n,gamma)': [0, 3]}}

    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "chain.xml"
        chain.export_to_xml(xml_path)
        reloaded = openmc.deplete.Chain.from_xml(xml_path)

    assert reloaded.isomeric_branching_targets['Ag109']['(n,gamma)'] == \
        ['Ag110', 'Ag110_m1']
    assert reloaded.isomeric_branching_lfs['Ag109']['(n,gamma)'] == [0, 3]
