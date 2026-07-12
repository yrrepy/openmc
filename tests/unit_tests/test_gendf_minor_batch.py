"""Unit tests for the GENDF runtime-branching / pool robustness batch.

Covers four review items:
1. Ground-state (LFS=0) auto-inclusion in the runtime BR denominator.
2. Warn-once when the chain requests an LFS absent from the GENDF file.
3. ELIS ambiguity warning when a second decay level also passes tolerance.
4. ``pool.deplete`` ``operator`` made keyword-only with upstream positional
   compatibility.
"""

import inspect
import warnings

import numpy as np
import pytest

import openmc.deplete.decay_elis as de
import openmc.deplete.gendf as g
from openmc.deplete.decay_elis import DecayState, lookup_liso
from openmc.deplete.gendf import build_runtime_branching
from openmc.deplete.pool import deplete, _accepts_isomeric_branching


@pytest.fixture(autouse=True)
def _clear_warn_dedup():
    """Reset module-level warn-once stores so warnings are deterministic."""
    g._WARNED_RUNTIME_BRANCHING.clear()
    de._WARNED_ELIS_AMBIGUITY.clear()
    yield
    g._WARNED_RUNTIME_BRANCHING.clear()
    de._WARNED_ELIS_AMBIGUITY.clear()


# ---------------------------------------------------------------------------
# Item 1: ground-state auto-inclusion in the denominator
# ---------------------------------------------------------------------------

# 3-group grid; ground XS=[8,4,0], m1 XS=[2,4,0]
ENERGY_BOUNDS = np.array([0.0, 1.0, 2.0, 3.0])
GROUND_M1_LEVELS = [
    (0, 500, np.array([8.0, 4.0, 0.0])),   # ground (LFS=0)
    (1, 501, np.array([2.0, 4.0, 0.0])),   # m1 (LFS=1)
]


def test_runtime_branching_ground_auto_included():
    """Requesting only m1 still normalizes over ground+m1, not the subset."""
    br = build_runtime_branching(
        GROUND_M1_LEVELS, ['Xx_m1'], [1], ENERGY_BOUNDS, 'Xx0', 102)
    # BR(m1) = xs_m1 / (xs_0 + xs_m1) = [2/10, 4/8, 0] -- NOT 1.0
    np.testing.assert_allclose(br.branching_ratios[0], [0.2, 0.5, 0.0])


def test_runtime_branching_both_levels_unchanged():
    """Requesting ground+m1 is unaffected by the ground-inclusion fix."""
    br = build_runtime_branching(
        GROUND_M1_LEVELS, ['Xx', 'Xx_m1'], [0, 1], ENERGY_BOUNDS, 'Xx0', 102)
    np.testing.assert_allclose(br.branching_ratios[0], [0.8, 0.5, 0.0])
    np.testing.assert_allclose(br.branching_ratios[1], [0.2, 0.5, 0.0])


def test_runtime_branching_no_ground_in_file_warns():
    """Metastable-only request with no LFS=0 in file warns and normalizes subset."""
    eb = np.array([0.0, 1.0, 2.0])
    levels = [(1, 501, np.array([2.0, 0.0])),
              (2, 502, np.array([2.0, 0.0]))]  # no ground level present
    with pytest.warns(UserWarning, match='no ground-state'):
        br = build_runtime_branching(levels, ['Xx_m1'], [1], eb, 'Zz', 16)
    # Only m1 requested and no ground -> subset normalizes to 1.0 (the pathology)
    np.testing.assert_allclose(br.branching_ratios[0], [1.0, 0.0])


# ---------------------------------------------------------------------------
# Item 2: warn-once when a requested LFS is absent from the GENDF file
# ---------------------------------------------------------------------------

def test_runtime_branching_lfs_missing_warns_once():
    """Chain requests LFS=3 absent from file -> BR 0 and one warning per (nuc, mt)."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        for _ in range(3):
            br = build_runtime_branching(
                GROUND_M1_LEVELS, ['Xx', 'Xx_m3'], [0, 3],
                ENERGY_BOUNDS, 'Zz', 16)
        msgs = [str(w.message) for w in rec
                if issubclass(w.category, UserWarning)]
    # Requested LFS=3 has no production -> BR row is all zeros
    np.testing.assert_allclose(br.branching_ratios[1], [0.0, 0.0, 0.0])
    # Exactly one warning despite three calls
    assert len(msgs) == 1
    assert 'LFS=3' in msgs[0]


def test_runtime_branching_lfs_missing_warns_per_pair():
    """A different (nuclide, mt) produces its own warning."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        build_runtime_branching(GROUND_M1_LEVELS, ['Xx', 'Xx_m3'], [0, 3],
                                ENERGY_BOUNDS, 'Aa', 16)
        build_runtime_branching(GROUND_M1_LEVELS, ['Xx', 'Xx_m3'], [0, 3],
                                ENERGY_BOUNDS, 'Bb', 16)
        msgs = [w for w in rec if issubclass(w.category, UserWarning)]
    assert len(msgs) == 2


# ---------------------------------------------------------------------------
# Item 3: ELIS ambiguity warning
# ---------------------------------------------------------------------------

@pytest.fixture
def ir192_lookup():
    return {(77, 192): [
        DecayState(z=77, a=192, elis=0.0, liso=0),
        DecayState(z=77, a=192, elis=56720.0, liso=1),
        DecayState(z=77, a=192, elis=168140.0, liso=2),
    ]}


def test_lookup_liso_ambiguous_warns_once(ir192_lookup):
    """Target ELIS in the m1/m2 overlap band warns once per (Z, A)."""
    # rtol=0.50: m1 passes for |t-56720|<=28360; m2 for |t-168140|<=84070.
    # Overlap band ~ [84070, 85080]; 84500 lies inside both.
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        r1 = lookup_liso(77, 192, 84500.0, ir192_lookup)
        r2 = lookup_liso(77, 192, 84500.0, ir192_lookup)  # repeat -> no 2nd warn
        msgs = [str(w.message) for w in rec
                if issubclass(w.category, UserWarning)]
    assert r1['status'] == 'matched' and r1['liso'] == 1
    assert r2['status'] == 'matched'
    assert len(msgs) == 1
    assert 'Ambiguous ELIS' in msgs[0]
    assert 'LISO=1' in msgs[0] and 'LISO=2' in msgs[0]


def test_lookup_liso_unambiguous_no_warn(ir192_lookup):
    """A clean m1 match (only one candidate within tolerance) does not warn."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        r = lookup_liso(77, 192, 56750.0, ir192_lookup)
        msgs = [w for w in rec if 'Ambiguous' in str(w.message)]
    assert r['status'] == 'matched' and r['liso'] == 1
    assert len(msgs) == 0


# ---------------------------------------------------------------------------
# Item 4: pool.deplete operator keyword-only + positional compatibility
# ---------------------------------------------------------------------------

def test_deplete_operator_is_keyword_only():
    """operator is keyword-only; matrix_args remains variadic-positional."""
    params = inspect.signature(deplete).parameters
    assert params['operator'].kind is inspect.Parameter.KEYWORD_ONLY
    assert params['matrix_args'].kind is inspect.Parameter.VAR_POSITIONAL
    # Upstream positional layout (params before *matrix_args) is preserved
    order = [p for p in params if params[p].kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,)]
    assert order == ['func', 'chain', 'n', 'rates', 'dt', 'current_timestep',
                     'matrix_func', 'transfer_rates', 'external_source_rates']


def test_deplete_positional_operator_goes_to_matrix_args():
    """A 10th positional lands in matrix_args, never in operator (keyword-only)."""
    sig = inspect.signature(deplete)
    bound = sig.bind('f', 'c', 'n', 'r', 'dt', 0, None, None, None, 'EXTRA')
    assert bound.arguments['matrix_args'] == ('EXTRA',)
    assert 'operator' not in bound.arguments  # stays at its default


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
    # Old-style positional operator now absorbed by *matrix_args and ignored
    deplete(lambda m, n0, t: n0, chain, [np.array([1.0])], [None], 1.0,
            0, None, None, None, FakeOp())
    assert chain.seen_iso is None  # branching disabled without keyword operator


def test_accepts_isomeric_branching_cached():
    """Cached helper detects the isomeric_branching parameter."""
    def with_iso(chain, rates, fy, isomeric_branching=None):
        pass

    def without_iso(chain, rates, fy):
        pass

    assert _accepts_isomeric_branching(with_iso) is True
    assert _accepts_isomeric_branching(without_iso) is False
