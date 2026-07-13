"""Shared test helpers for the GENDF / isomeric-branching unit tests.

Importable objects (test files use ``from .gendf_testing import ...``):

* ``MockGENDFLibrary``           - parameterizable plain-class GENDF stand-in
* ``make_mock_gendf``            - ``unittest.mock.Mock`` GENDF (call-assert tests)
* ``make_isomeric_branching``    - build an ``IsomericBranching`` dataclass
* ``Tab1D`` / ``make_python_gendf_lib`` / ``threshold_level`` - Python-backend stubs
* ``make_isomeric_chain`` / ``make_mock_chain`` - chain factories
* ``endf6_line`` / ``write_synthetic_gendf``    - synthetic ENDF-6/GENDF writers
* ``bare_coupled_operator``      - transport-free CoupledOperator skeleton
* ``gendf_data_root`` / ``gendf_chains_root`` / ``require_existing`` /
  ``first_existing`` - real-data path resolution (see conftest for fixtures)

Real-data env vars are documented in ``conftest.py``.
"""

import os
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

import openmc.deplete
from openmc.deplete import CoupledOperator
from openmc.deplete.chain import Chain
from openmc.deplete.gendf import IsomericBranching, _PythonGENDFLibrary
from openmc.mgxs import GROUP_STRUCTURES

# Default (and physically correct) group structure for the mocks.
CCFE709_BOUNDS = GROUP_STRUCTURES['CCFE-709']
CCFE709_NGROUPS = len(CCFE709_BOUNDS) - 1


# ---------------------------------------------------------------------------
# Mock GENDF libraries
# ---------------------------------------------------------------------------

class MockGENDFLibrary:
    """Parameterizable plain-class GENDF stand-in (superset of the 5 class mocks).

    ``nuclides`` may be an iterable of names (uniform ``xs_value`` for
    ``default_mts``) or a ``{name: {mt: array}}`` dict; ``xs`` is the same dict
    given by keyword. ``branching`` (an ``IsomericBranching``, a callable, or
    ``None``) backs ``get_branching_ratios`` (``None`` -> ``KeyError``).
    """

    def __init__(self, nuclides=None, *, xs=None, n_groups=CCFE709_NGROUPS,
                 energy_bounds=None, energy_structure='CCFE-709',
                 xs_value=1.0, default_mts=(102, 18), branching=None):
        if xs is None and isinstance(nuclides, dict):
            xs, nuclides = nuclides, None
        if xs is not None:
            self._xs = {n: dict(mt_map) for n, mt_map in xs.items()}
            self._available = set(self._xs)
        else:
            self._xs = None
            self._available = set(nuclides or ())
        self.n_groups = n_groups
        self.energy_structure = energy_structure
        self.energy_bounds = (np.asarray(CCFE709_BOUNDS) if energy_bounds is None
                              else np.asarray(energy_bounds))
        self._xs_value = xs_value
        self._default_mts = tuple(default_mts)
        self._branching = branching

    def available_nuclides_set(self):
        """Set of nuclide names with GENDF data."""
        return frozenset(self._available)

    def get_xs(self, nuclide, mt, energies=None):
        """Group cross section for one (nuclide, mt)."""
        if nuclide not in self._available:
            raise KeyError(nuclide)
        if self._xs is not None and mt in self._xs[nuclide]:
            return np.array(self._xs[nuclide][mt], dtype=float)
        return np.full(self.n_groups, self._xs_value)

    def get_all_xs(self, nuclide, mts=None, strict_alignment=True):
        """Dict of {mt: group-xs} for a nuclide, optionally filtered by ``mts``."""
        if nuclide not in self._available:
            raise KeyError(nuclide)
        if self._xs is not None:
            all_xs = {mt: np.array(v, dtype=float)
                      for mt, v in self._xs[nuclide].items()}
        else:
            all_xs = {mt: np.full(self.n_groups, self._xs_value)
                      for mt in self._default_mts}
        if mts is not None:
            keep = set(mts)
            return {mt: v for mt, v in all_xs.items() if mt in keep}
        return all_xs

    def get_branching_ratios(self, nuclide, mt, target_names=None,
                             lfs_values=None):
        """Runtime isomeric branching; ``KeyError`` when unconfigured."""
        br = self._branching
        if br is None:
            raise KeyError(nuclide)
        return br(nuclide, mt, target_names, lfs_values) if callable(br) else br


def make_mock_gendf(n_groups=CCFE709_NGROUPS, energy_bins=None, *,
                    energy_structure='CCFE-709', xs=None, xs_value=1.0,
                    in_range_mask=None, branching=None):
    """``unittest.mock.Mock`` GENDF library for tests that assert on call counts.

    ``get_xs`` returns ``xs`` (array or side-effect callable), else a masked
    array from ``in_range_mask``, else ``xs_value`` everywhere.
    ``get_branching_ratios`` returns ``branching`` (dataclass / callable / None).
    """
    mock = Mock()
    mock.energy_structure = energy_structure
    bounds = CCFE709_BOUNDS if energy_bins is None else energy_bins
    mock.energy_bounds = np.asarray(bounds).copy()
    mock.n_groups = n_groups
    if xs is not None:
        mock.get_xs = Mock(side_effect=xs) if callable(xs) else Mock(return_value=xs)
    elif in_range_mask is not None:
        gendf_xs = np.zeros(n_groups)
        gendf_xs[in_range_mask] = xs_value
        mock.get_xs = Mock(return_value=gendf_xs)
    else:
        mock.get_xs = Mock(return_value=np.full(n_groups, xs_value))
    if callable(branching):
        mock.get_branching_ratios = Mock(side_effect=branching)
    else:
        mock.get_branching_ratios = Mock(return_value=branching)
    return mock


def make_isomeric_branching(parent, reaction, products, energies, ratios,
                            mt=102):
    """Build an ``IsomericBranching`` dataclass from products and a 2-D ratio array."""
    return IsomericBranching(
        energies=np.asarray(energies, dtype=float),
        products=list(products),
        branching_ratios=np.asarray(ratios, dtype=float),
        parent_nuclide=parent,
        reaction=reaction,
        mt=mt,
    )


# ---------------------------------------------------------------------------
# Python-backend (real _PythonGENDFLibrary) stubs for threshold-alignment tests
# ---------------------------------------------------------------------------

class Tab1D:
    """Minimal Tabulated1D stand-in exposing ``.x`` / ``.y`` like the endf package."""

    def __init__(self, x, y):
        self.x = np.asarray(x)
        self.y = np.asarray(y)


def make_python_gendf_lib(mf10_levels, *, energy_bounds=None,
                          energy_structure='CCFE-709'):
    """Bare ``_PythonGENDFLibrary`` with a stubbed MF=10 loader (no data files)."""
    bounds = np.asarray(CCFE709_BOUNDS if energy_bounds is None else energy_bounds)
    lib = _PythonGENDFLibrary.__new__(_PythonGENDFLibrary)
    lib.energy_bounds = bounds
    lib.n_groups = len(bounds) - 1
    lib.energy_structure = energy_structure
    lib._load_mf10_data = lambda nuc, mt: ({'levels': mf10_levels}, None)
    return lib


def threshold_level(lfs, izap, start, value_lo, value_hi, bounds=None):
    """Partial-range MF=10 band spanning groups ``[start, n_groups)``."""
    b = np.asarray(CCFE709_BOUNDS if bounds is None else bounds)
    x = b[start:]
    y = np.linspace(value_lo, value_hi, len(x))
    return {'LFS': lfs, 'IZAP': izap, 'sigma': Tab1D(x, y)}


# ---------------------------------------------------------------------------
# Chain factories
# ---------------------------------------------------------------------------

def make_isomeric_chain(parent, reactions, *, half_lives=None, lfs=None,
                        q=6.8e6):
    """Real ``Chain``: one parent with isomeric branching to sibling targets.

    ``reactions`` maps ``reaction -> [target_names]``; ``q`` is a scalar or a
    ``{reaction: Q}`` dict; ``half_lives`` and ``lfs`` are optional per-target /
    per-reaction overrides.
    """
    chain = openmc.deplete.Chain()
    parent_nuc = openmc.deplete.Nuclide(parent)
    for rxn, targets in reactions.items():
        qval = q[rxn] if isinstance(q, dict) else q
        parent_nuc.add_reaction(rxn, targets[0], Q=qval, branching_ratio=1.0)
    chain.add_nuclide(parent_nuc)
    added = set()
    for targets in reactions.values():
        for name in targets:
            if name in added:
                continue
            added.add(name)
            nuc = openmc.deplete.Nuclide(name)
            if half_lives and name in half_lives:
                nuc.half_life = half_lives[name]
            chain.add_nuclide(nuc)
    chain.isomeric_branching_targets = {
        parent: {r: list(t) for r, t in reactions.items()}}
    chain.isomeric_branching_lfs = {parent: dict(lfs)} if lfs else None
    return chain


def make_mock_chain(targets, *, lfs=None, embedded=None):
    """``Mock(spec=Chain)`` carrying the three ``isomeric_branching_*`` attributes."""
    chain = Mock(spec=Chain)
    chain.isomeric_branching_targets = targets
    chain.isomeric_branching_lfs = lfs
    chain.isomeric_branching_embedded = embedded
    return chain


# ---------------------------------------------------------------------------
# Synthetic ENDF-6 / GENDF writers (C++ parser regression tests)
# ---------------------------------------------------------------------------

def endf6_line(fields, mat, mf, mt):
    """Format one ENDF-6 line: six 11-char fields + MAT(4) MF(2) MT(3)."""
    body = ''.join(f'{f:>11}' for f in fields)
    body = body.ljust(66)
    return f'{body}{mat:>4}{mf:>2}{mt:>3}\n'


def write_synthetic_gendf(path, variant='izap0', mat=None):
    """Write a synthetic ENDF-6/GENDF file for the C++ parser regression tests.

    Variants: ``'izap0'`` (MF=10 IZAP=0 subsection then a real level, NR=1),
    ``'nr_skip'`` (MF=3 and MF=10 TAB1 with NR=2), ``'negative_xs'`` (MF=3 with a
    negative middle-group XS). 3-group grid ``[1, 1e3, 1e6, 1e9]``.
    """
    if variant == 'izap0':
        mat = 1325 if mat is None else mat
        lines = [endf6_line(['1.302700+4', '2.675000+1', 0, 0, 0, 5], mat, 1, 451)]
        lines += [endf6_line(['synthetic', 'test', 'file', '', '', ''], mat, 1, 451)
                  for _ in range(4)]
        lines.append(endf6_line(['0.0', '0.0', 0, 0, 1, 3], mat, 3, 102))
        lines.append(endf6_line([3, 1, '', '', '', ''], mat, 3, 102))
        lines.append(endf6_line(['1.000000+0', '1.100000+0', '1.000000+3',
                                 '2.200000+0', '1.000000+6', '3.300000+0'],
                                mat, 3, 102))
        lines.append(endf6_line(['-1.305820+7', '-1.305820+7', 0, 0, 1, 3],
                                mat, 10, 102))
        lines.append(endf6_line([3, 1, '', '', '', ''], mat, 10, 102))
        lines.append(endf6_line(['1.000000+0', '9.900000+0', '1.000000+3',
                                 '8.800000+0', '1.000000+6', '7.700000+0'],
                                mat, 10, 102))
        lines.append(endf6_line(['-1.305820+7', '-1.328660+7', 13028, 1, 1, 3],
                                mat, 10, 102))
        lines.append(endf6_line([3, 1, '', '', '', ''], mat, 10, 102))
        lines.append(endf6_line(['1.000000+0', '1.500000-1', '1.000000+3',
                                 '2.500000-1', '1.000000+6', '3.500000-1'],
                                mat, 10, 102))
    elif variant == 'nr_skip':
        mat = 1325 if mat is None else mat
        lines = [endf6_line(['1.302700+4', '2.675000+1', 0, 0, 0, 5], mat, 1, 451)]
        lines += [endf6_line(['synthetic', 'test', 'file', '', '', ''], mat, 1, 451)
                  for _ in range(4)]
        lines.append(endf6_line(['0.0', '0.0', 0, 0, 2, 3], mat, 3, 102))
        lines.append(endf6_line([2, 2, 3, 2, '', ''], mat, 3, 102))
        lines.append(endf6_line(['1.000000+0', '1.100000+0', '1.000000+3',
                                 '2.200000+0', '1.000000+6', '3.300000+0'],
                                mat, 3, 102))
        lines.append(endf6_line(['-1.305820+7', '-1.328660+7', 13028, 1, 2, 3],
                                mat, 10, 102))
        lines.append(endf6_line([2, 2, 3, 2, '', ''], mat, 10, 102))
        lines.append(endf6_line(['1.000000+0', '1.500000-1', '1.000000+3',
                                 '2.500000-1', '1.000000+6', '3.500000-1'],
                                mat, 10, 102))
    elif variant == 'negative_xs':
        mat = 2631 if mat is None else mat
        lines = [endf6_line(['2.605600+4', '5.545400+1', 0, 0, 0, 9], mat, 1, 451)]
        lines += [endf6_line(['synthetic', 'test', 'file', '', '', ''], mat, 1, 451)
                  for _ in range(8)]
        lines.append(endf6_line(['0.0', '0.0', 0, 0, 1, 3], mat, 3, 102))
        lines.append(endf6_line([3, 2, '', '', '', ''], mat, 3, 102))
        lines.append(endf6_line(['1.000000+0', '1.100000+0', '1.000000+3',
                                 '-5.000000+0', '1.000000+6', '3.300000+0'],
                                mat, 3, 102))
    else:
        raise ValueError(f"unknown variant {variant!r}")

    with open(path, 'w') as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# Transport-free CoupledOperator skeleton
# ---------------------------------------------------------------------------

def bare_coupled_operator(*, gendf_library=None, rate_helper=None, targets=None,
                          isomeric_branching=None):
    """CoupledOperator skeleton with only the state ``_setup_isomeric_branching`` needs."""
    op = CoupledOperator.__new__(CoupledOperator)
    op._gendf_library = gendf_library
    op._rate_helper = rate_helper
    op.chain = Mock(isomeric_branching_targets=targets or {})
    op._isomeric_helper = None
    op._isomeric_branching = isomeric_branching
    return op


# ---------------------------------------------------------------------------
# Real-data path resolution (fixtures live in conftest.py)
# ---------------------------------------------------------------------------

# Fallback roots preserve the pre-consolidation behaviour on the original
# machine when the env vars are unset. See conftest.py module docstring.
_DEFAULT_DATA_ROOT = '/home/perry/NukeData'
_DEFAULT_CHAINS_ROOT = '/home/perry/Codes/OpenMC/chains'


def gendf_data_root():
    """Root of the real GENDF/decay data (``$OPENMC_GENDF_TEST_DATA``)."""
    return Path(os.environ.get('OPENMC_GENDF_TEST_DATA', _DEFAULT_DATA_ROOT))


def gendf_chains_root():
    """Root of the real depletion-chain XMLs (``$OPENMC_GENDF_TEST_CHAINS``)."""
    return Path(os.environ.get('OPENMC_GENDF_TEST_CHAINS', _DEFAULT_CHAINS_ROOT))


def require_existing(path, reason):
    """Return ``path`` if it exists, else ``pytest.skip(reason)``."""
    path = Path(path)
    if not path.exists():
        pytest.skip(reason)
    return path


def first_existing(paths, reason):
    """Return the first existing path, else ``pytest.skip(reason)``."""
    for p in paths:
        if Path(p).exists():
            return Path(p)
    pytest.skip(reason)
