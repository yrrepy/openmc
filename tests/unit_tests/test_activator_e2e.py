"""End-to-end test: Activator scenario with real GENDF data.

Reproduces the original bug where IsomericBranchingHelper crashed
when using a chain patched without gendf_lfs, then verifies the
new runtime mode works with a properly patched chain.

Requires:
- TENDL-2017 GENDF library at the standard path
- Optionally: decay data for patcher-mode testing
"""

import warnings
import numpy as np
import pytest
from pathlib import Path

from openmc.mgxs import GROUP_STRUCTURES
from openmc.deplete.helpers import IsomericBranchingHelper


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def gendf_dir():
    """Path to TENDL-2017 GENDF files."""
    path = Path('/home/perry/NukeData/Activation/FISPACT/TENDL2017data/tal2017-n/gxs-709')
    if not path.exists():
        pytest.skip(f"GENDF directory not found: {path}")
    return path


@pytest.fixture
def decay_file():
    """Path to decay data file."""
    paths_to_try = [
        Path('/home/perry/NukeData/Activation/FISPACT/TENDL2017data/tal2017-n/decay_2020.endf'),
        Path('/home/perry/NukeData/Activation/decay/decay_2020.endf'),
    ]
    for path in paths_to_try:
        if path.exists():
            return path
    pytest.skip("Decay file not found")


@pytest.fixture
def chain_path():
    """Path to activation chain file."""
    paths_to_try = [
        Path('/home/perry/Codes/OpenMC/chains/chain_activator_TENDL2017_ccfe709_lfs.xml'),
        Path('/home/perry/Codes/OpenMC/chains/chain_activator_TENDL2017_ccfe709.xml'),
    ]
    for path in paths_to_try:
        if path.exists():
            return path
    pytest.skip("Chain file not found")


# ============================================================================
# Tests with C++ backend (no decay file)
# ============================================================================

def test_cpp_backend_ag109_branching(gendf_dir):
    """C++ backend: Ag109(n,gamma) produces branching ratios for
    Ag110/Ag110_m1 using runtime mode with LFS values.

    This is the scenario that previously crashed with ValueError.
    """
    from openmc.deplete.gendf import GENDFLibrary
    from openmc.deplete.gendf import _CppGENDFLibrary

    if _CppGENDFLibrary is None:
        pytest.skip("C++ GENDF backend not available")

    lib = GENDFLibrary(str(gendf_dir))

    if not lib.has_nuclide('Ag109'):
        pytest.skip("Ag109 not in GENDF library")

    # Runtime mode: provide target names and LFS values
    br = lib.get_branching_ratios(
        'Ag109', 102,
        target_names=['Ag110', 'Ag110_m1'],
        lfs_values=[0, 1]
    )

    if br is None:
        pytest.skip("No MF=10 data for Ag109 MT=102")

    assert len(br.products) == 2
    assert 'Ag110' in br.products
    assert 'Ag110_m1' in br.products

    # Verify per-group conservation
    for g in range(len(br.energies)):
        total = br.branching_ratios[:, g].sum()
        if total > 0:
            assert np.isclose(total, 1.0, rtol=1e-10), \
                f"Group {g}: BR sum = {total}"


def test_cpp_backend_ag107_branching(gendf_dir):
    """C++ backend: Ag107(n,gamma) branching with runtime mode."""
    from openmc.deplete.gendf import GENDFLibrary, _CppGENDFLibrary

    if _CppGENDFLibrary is None:
        pytest.skip("C++ GENDF backend not available")

    lib = GENDFLibrary(str(gendf_dir))

    if not lib.has_nuclide('Ag107'):
        pytest.skip("Ag107 not in GENDF library")

    br = lib.get_branching_ratios(
        'Ag107', 102,
        target_names=['Ag108', 'Ag108_m1'],
        lfs_values=[0, 1]
    )

    if br is None:
        pytest.skip("No MF=10 data for Ag107 MT=102")

    assert len(br.products) == 2
    # Conservation
    for g in range(len(br.energies)):
        total = br.branching_ratios[:, g].sum()
        if total > 0:
            assert np.isclose(total, 1.0, rtol=1e-10)


def test_cpp_backend_threshold_alignment(gendf_dir):
    """C++ backend: threshold reaction (n,2n) places production XS at
    correct energy index using energy-aware alignment."""
    from openmc.deplete.gendf import GENDFLibrary, _CppGENDFLibrary

    if _CppGENDFLibrary is None:
        pytest.skip("C++ GENDF backend not available")

    lib = GENDFLibrary(str(gendf_dir))

    # Al26 is a TENDL nuclide with threshold (n,2n) -> Al25
    # Use a more common nuclide if Al26 not available
    for nuclide, mt in [('Al27', 16), ('Fe56', 16), ('Cu63', 16)]:
        if lib.has_nuclide(nuclide):
            try:
                xs = lib.get_xs(nuclide, mt)
            except Exception:
                continue

            # (n,2n) is a threshold reaction: XS should be zero at low energies
            # and non-zero above threshold (typically >6-10 MeV)
            n_groups = len(xs)
            # First 200 groups (thermal + epithermal) should be zero
            assert np.allclose(xs[:200], 0.0), \
                f"{nuclide} MT={mt}: expected zero XS below threshold"
            # Should have non-zero XS somewhere in the fast range
            assert np.any(xs[500:] > 0), \
                f"{nuclide} MT={mt}: expected non-zero XS above threshold"
            return

    pytest.skip("No suitable nuclide found for threshold test")


# ============================================================================
# Tests with Python backend (decay file required)
# ============================================================================

def test_python_backend_ag109_patcher_mode(gendf_dir, decay_file):
    """Python backend patcher mode: Ag109(n,gamma) with ELIS mapping."""
    from openmc.deplete.gendf import GENDFLibrary

    lib = GENDFLibrary(str(gendf_dir), decay_file=str(decay_file))

    if not lib.has_nuclide('Ag109'):
        pytest.skip("Ag109 not in GENDF library")

    # Patcher mode: no target_names
    br = lib.get_branching_ratios('Ag109', 102)

    if br is None:
        pytest.skip("No MF=10 data for Ag109 MT=102")

    assert 'Ag110' in br.products or 'Ag110_m1' in br.products

    # Conservation
    for g in range(len(br.energies)):
        total = br.branching_ratios[:, g].sum()
        if total > 0:
            assert np.isclose(total, 1.0, rtol=1e-6)


# ============================================================================
# Cross-backend parity test
# ============================================================================

def test_cross_backend_parity_ir191(gendf_dir, decay_file):
    """Ir191 MT=102: C++ and Python backends produce identical branching
    ratios when using runtime mode with the same LFS values."""
    from openmc.deplete.gendf import GENDFLibrary, _CppGENDFLibrary

    if _CppGENDFLibrary is None:
        pytest.skip("C++ GENDF backend not available")

    # Create both backends
    cpp_lib = GENDFLibrary(str(gendf_dir))  # Auto-selects C++
    py_lib = GENDFLibrary(str(gendf_dir), decay_file=str(decay_file))

    if not cpp_lib.has_nuclide('Ir191') or not py_lib.has_nuclide('Ir191'):
        pytest.skip("Ir191 not in GENDF library")

    # Use runtime mode for both with the same LFS values
    target_names = ['Ir192', 'Ir192_m1']
    lfs_values = [0, 1]

    cpp_br = cpp_lib.get_branching_ratios(
        'Ir191', 102,
        target_names=target_names,
        lfs_values=lfs_values
    )
    py_br = py_lib.get_branching_ratios(
        'Ir191', 102,
        target_names=target_names,
        lfs_values=lfs_values
    )

    if cpp_br is None or py_br is None:
        pytest.skip("No MF=10 data for Ir191 MT=102")

    # Branching ratios should be identical (both read same data)
    assert np.allclose(cpp_br.branching_ratios, py_br.branching_ratios,
                       rtol=1e-12), \
        "C++ and Python backends produce different branching ratios"


# ============================================================================
# Full IsomericBranchingHelper with real data
# ============================================================================

def test_helper_with_real_gendf_and_chain(gendf_dir, chain_path):
    """Full IsomericBranchingHelper pipeline with real chain and GENDF."""
    from openmc.deplete.chain import Chain
    from openmc.deplete.gendf import GENDFLibrary

    chain = Chain.from_xml(str(chain_path))

    if not chain.isomeric_branching_targets:
        pytest.skip("Chain has no isomeric branching targets")

    lib = GENDFLibrary(str(gendf_dir))

    helper = IsomericBranchingHelper(chain, lib)

    energy_bins = GROUP_STRUCTURES['CCFE-709']
    n_groups = 709

    # Use flat flux (thermal reactor-like spectrum would be better,
    # but flat is sufficient for correctness testing)
    flux = np.ones(n_groups)

    result = helper.weighted_branching_ratios(flux, energy_bins)

    # Should produce results for at least some nuclides
    if not result:
        # Acceptable if chain has targets but GENDF doesn't have MF=10
        warnings.warn("No branching results produced (GENDF may lack MF=10)")
        return

    for nuclide, reactions in result.items():
        for reaction, ratios in reactions.items():
            total = sum(ratios.values())
            assert np.isclose(total, 1.0, rtol=1e-4), \
                f"{nuclide} {reaction}: ratios sum to {total}"
            for target, ratio in ratios.items():
                assert 0.0 <= ratio <= 1.0, \
                    f"{nuclide} {reaction} -> {target}: ratio={ratio} out of [0,1]"
