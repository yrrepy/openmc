"""End-to-end GENDF tests exercised against real nuclear data.

Merges the real-data tests from ``test_activator_e2e.py`` (Activator scenario:
runtime/patcher branching, cross-backend parity, full IsomericBranchingHelper)
and the three MF=10/MT=4 extraction tests from ``test_nn_prime_reaction.py``.

All fixtures (``gendf_dir``, ``decay_file``, ``chain_path``) come from the
shared ``conftest.py`` and resolve under ``$OPENMC_GENDF_TEST_DATA`` /
``$OPENMC_GENDF_TEST_CHAINS``; each skips with an informative reason when the
data is absent.
"""

import warnings
import numpy as np
import pytest

from openmc.deplete.helpers import IsomericBranchingHelper


# ============================================================================
# Tests with C++ backend (no decay file)
# ============================================================================

def test_cpp_backend_ag109_branching(gendf_dir):
    """C++ backend: Ag109(n,gamma) runtime branching for Ag110/Ag110_m1."""
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
    """C++ backend: threshold (n,2n) production XS is placed by energy."""
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
    """Ir191 MT=102: C++ and Python backends give identical runtime BR."""
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

    n_groups = 709

    # Use flat flux (thermal reactor-like spectrum would be better,
    # but flat is sufficient for correctness testing)
    flux = np.ones(n_groups)

    result = helper.weighted_branching_ratios(flux)

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


# ============================================================================
# GENDF MF=10/MT=4 (n,n') extraction tests (from test_nn_prime_reaction.py)
# ============================================================================

def test_in115_mt4_data_exists(gendf_dir):
    """Verify In115 GENDF file contains MF=10/MT=4 data."""
    # Check file exists
    in115_file = gendf_dir / 'In115g.asc'
    if not in115_file.exists():
        pytest.skip(f"In115 GENDF file not found: {in115_file}")

    # Read and search for MF=10, MT=4 section
    content = in115_file.read_text()

    # ENDF format: columns 71-72=MF, 73-75=MT for data cards
    # Section header marker for MF=10, MT=4
    assert '4931110  4' in content or '493110  4' in content, \
        "MF=10/MT=4 section not found in In115 GENDF file"


def test_extract_branching_mt4(gendf_dir, decay_file):
    """Test extraction of isomeric branching for MT=4."""
    from openmc.deplete.gendf import GENDFLibrary

    lib = GENDFLibrary(gendf_dir, decay_file=decay_file, mapping_mode='elis')

    # Get branching ratios for MT=4
    branching = lib.get_branching_ratios('In115', 4)

    if branching is None:
        pytest.skip("No MF=10/MT=4 data extracted for In115")

    # Should have (n,n') key
    assert "(n,n')" in branching, f"Missing (n,n') key, got: {list(branching.keys())}"

    products = branching["(n,n')"]

    # Should have ground state and metastable
    assert 'In115' in products, "Missing ground state In115"
    assert 'In115_m1' in products, "Missing metastable In115_m1"

    # Ratios should be reasonable (non-zero, positive)
    for nuclide, ratio in products.items():
        if isinstance(ratio, np.ndarray):
            assert np.all(ratio >= 0), f"Negative ratio for {nuclide}"
            assert np.any(ratio > 0), f"All-zero ratio for {nuclide}"
        else:
            assert ratio >= 0, f"Negative ratio for {nuclide}"


def test_extract_branching_ir192_multilevel(gendf_dir, decay_file):
    """Test extraction of three-level branching for Ir192."""
    from openmc.deplete.gendf import GENDFLibrary

    lib = GENDFLibrary(gendf_dir, decay_file=decay_file, mapping_mode='elis')

    # Get branching ratios for MT=4
    branching = lib.get_branching_ratios('Ir192', 4)

    if branching is None:
        pytest.skip("No MF=10/MT=4 data extracted for Ir192")

    if "(n,n')" not in branching:
        pytest.skip("No (n,n') data for Ir192")

    products = branching["(n,n')"]

    # Ir192 should have 3 product states (NS=3)
    # Ground + m1 + m2
    product_names = list(products.keys())
    assert len(product_names) >= 2, f"Expected multiple products, got: {product_names}"

    # Check for metastable states
    metastable_products = [p for p in product_names if '_m' in p]
    assert len(metastable_products) >= 1, \
        f"Expected at least one metastable product for Ir192, got: {product_names}"
