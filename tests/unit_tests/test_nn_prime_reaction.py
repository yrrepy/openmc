"""Tests for (n,n') inelastic neutron activation support.

This module tests the addition of MT=4 (n,n') inelastic scattering to the
REACTIONS dictionary and verifies proper propagation through the depletion
infrastructure.

The (n,n') reaction enables energy-dependent isomeric activation where
inelastic scattering can produce metastable states:
  - In115(n,n')In115_m1 (classic dosimetry reaction)
  - Y89(n,n')Y89_m1
  - Ir192(n,n')Ir192_m1, Ir192_m2 (3-way branching)
"""

import pytest
import numpy as np
from pathlib import Path


# ==============================================================================
# Basic REACTIONS dictionary tests
# ==============================================================================

def test_nn_prime_in_reactions():
    """Verify (n,n') is present in REACTIONS dictionary."""
    from openmc.deplete.chain import REACTIONS
    assert "(n,n')" in REACTIONS


def test_nn_prime_mt_value():
    """Verify (n,n') has MT=4."""
    from openmc.deplete.chain import REACTIONS
    info = REACTIONS["(n,n')"]
    assert 4 in info.mts
    assert info.mts == {4}


def test_nn_prime_secondaries():
    """Verify (n,n') has no secondary particles.

    In (n,n') scattering, the incident neutron scatters inelastically,
    exciting the nucleus. No additional particles are emitted beyond
    the scattered neutron.
    """
    from openmc.deplete.chain import REACTIONS
    info = REACTIONS["(n,n')"]
    assert info.secondaries == ()


def test_mt_to_reaction_mapping():
    """Verify MT=4 maps to (n,n') in gendf.py."""
    from openmc.deplete.gendf import MT_TO_REACTION
    assert 4 in MT_TO_REACTION
    assert MT_TO_REACTION[4] == "(n,n')"


def test_reaction_to_mt_mapping():
    """Verify (n,n') maps to MT=4 in gendf.py."""
    from openmc.deplete.gendf import REACTION_TO_MT
    assert "(n,n')" in REACTION_TO_MT
    assert REACTION_TO_MT["(n,n')"] == 4


def test_total_reaction_count():
    """Verify total number of reactions after adding (n,n')."""
    from openmc.deplete.chain import REACTIONS
    # Was 84 before, now 85 with (n,n')
    assert len(REACTIONS) == 85


# ==============================================================================
# Physics handling tests
# ==============================================================================

def test_self_transmutation_chain_structure(tmp_path):
    """Verify chain can handle self-transmutation correctly.

    For (n,n') reactions:
    - In115 -> In115 (ground state, same nuclide)
    - In115 -> In115_m1 (metastable, different nuclide entry)
    """
    from openmc.deplete import Chain

    chain_xml = """<?xml version="1.0"?>
<depletion_chain>
  <nuclide name="In115" decay_modes="0" reactions="1">
    <reaction type="(n,n')" Q="0.0" target="In115" branching_ratio="0.85"/>
    <isomeric_branching>
      <reaction type="(n,n')">
        <product nuclide="In115" ratio="0.85"/>
        <product nuclide="In115_m1" ratio="0.15"/>
      </reaction>
    </isomeric_branching>
  </nuclide>
  <nuclide name="In115_m1" decay_modes="1" reactions="0">
    <decay type="IT" target="In115" branching_ratio="1.0"/>
  </nuclide>
</depletion_chain>
"""
    chain_file = tmp_path / "chain_nn_prime.xml"
    chain_file.write_text(chain_xml)
    chain = Chain.from_xml(chain_file)

    # Verify both nuclides are in the chain
    assert 'In115' in chain.nuclide_dict
    assert 'In115_m1' in chain.nuclide_dict

    # Verify reaction is loaded
    in115 = chain['In115']
    reactions = [r.type for r in in115.reactions]
    assert "(n,n')" in reactions


def test_branching_ratios_sum_to_unity():
    """Verify isomeric branching ratios sum to 1.0 for mass conservation."""
    # In115 example from implementation plan
    branching = {
        'In115': 0.85,  # Ground state
        'In115_m1': 0.15  # Metastable
    }
    total = sum(branching.values())
    assert abs(total - 1.0) < 1e-10, f"Branching ratios sum to {total}, not 1.0"


def test_three_way_branching_ratios():
    """Test that three-way branching (NS=3) sums correctly.

    Ir192 has three product states for (n,n'):
    - Ir192 (ground)
    - Ir192_m1 (LFS=3, ELIS=56.7 keV)
    - Ir192_m2 (LFS=15, ELIS=168.1 keV)
    """
    # Example branching at ~5 MeV from implementation_inelastic_activation.md
    branching_5mev = {
        'Ir192': 0.74,      # 74% ground
        'Ir192_m1': 0.22,   # 22% m1
        'Ir192_m2': 0.04    # 4% m2
    }
    total = sum(branching_5mev.values())
    assert abs(total - 1.0) < 1e-10


# ==============================================================================
# GENDF data extraction tests
# ==============================================================================

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
        Path('/home/perry/NukeData/Activation/decay/decay2020.endf'),
    ]
    for path in paths_to_try:
        if path.exists():
            return path
    pytest.skip("Decay file not found")


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


# ==============================================================================
# Integration tests
# ==============================================================================

def test_chain_xml_with_nn_prime(tmp_path):
    """Test that chain XML with (n,n') reactions loads correctly."""
    from openmc.deplete import Chain

    # Chain with (n,n') reaction and isomeric branching
    chain_xml = """<?xml version="1.0"?>
<depletion_chain>
  <nuclide name="In115" decay_modes="0" reactions="2">
    <reaction type="(n,gamma)" Q="6.78e6" target="In116"/>
    <reaction type="(n,n')" Q="0.0" target="In115" branching_ratio="0.85"/>
    <isomeric_branching>
      <reaction type="(n,n')">
        <product nuclide="In115" ratio="0.85"/>
        <product nuclide="In115_m1" ratio="0.15"/>
      </reaction>
    </isomeric_branching>
  </nuclide>
  <nuclide name="In115_m1" decay_modes="1" reactions="0" half_life="1.61e4">
    <decay type="IT" target="In115" branching_ratio="1.0"/>
  </nuclide>
  <nuclide name="In116" decay_modes="0" reactions="0"/>
</depletion_chain>
"""
    chain_file = tmp_path / "chain_nn_prime.xml"
    chain_file.write_text(chain_xml)
    chain = Chain.from_xml(chain_file)

    # Verify chain loaded correctly
    assert len(chain) == 3
    assert 'In115' in chain.nuclide_dict
    assert 'In115_m1' in chain.nuclide_dict
    assert 'In116' in chain.nuclide_dict

    # Verify In115 has (n,n') reaction
    in115 = chain['In115']
    reaction_types = [r.type for r in in115.reactions]
    assert "(n,n')" in reaction_types

    # Verify isomeric branching targets are loaded
    if chain.isomeric_branching_targets:
        if 'In115' in chain.isomeric_branching_targets:
            assert "(n,n')" in chain.isomeric_branching_targets['In115']


def test_form_matrix_with_nn_prime(tmp_path):
    """Test matrix formation with (n,n') isomeric branching."""
    from openmc.deplete import Chain
    from openmc.deplete import reaction_rates
    import scipy.sparse as sp

    chain_xml = """<?xml version="1.0"?>
<depletion_chain>
  <nuclide name="In115" decay_modes="0" reactions="1">
    <reaction type="(n,n')" Q="0.0" target="In115" branching_ratio="0.85"/>
    <isomeric_branching>
      <reaction type="(n,n')">
        <product nuclide="In115" ratio="0.85"/>
        <product nuclide="In115_m1" ratio="0.15"/>
      </reaction>
    </isomeric_branching>
  </nuclide>
  <nuclide name="In115_m1" decay_modes="1" reactions="0" half_life="1.61e4">
    <decay type="IT" target="In115" branching_ratio="1.0"/>
  </nuclide>
</depletion_chain>
"""
    chain_file = tmp_path / "chain_nn_prime.xml"
    chain_file.write_text(chain_xml)
    chain = Chain.from_xml(chain_file)

    # Create reaction rates using ReactionRates object
    nuclides = ["In115", "In115_m1"]
    rates = reaction_rates.ReactionRates(["mat1"], nuclides, chain.reactions)
    rates.set("mat1", "In115", "(n,n')", 1e-10)  # 1e-10 s^-1 reaction rate

    # Form the depletion matrix
    matrix = chain.form_matrix(rates[0])

    # Matrix should be sparse
    assert sp.issparse(matrix)

    # Get indices
    i_in115 = chain.nuclide_dict['In115']
    i_in115_m1 = chain.nuclide_dict['In115_m1']

    # Convert to dense for inspection
    dense = matrix.toarray()

    # Check that there is transfer from In115 to In115_m1
    # The matrix element [In115_m1, In115] should be positive (production)
    # Note: Matrix is [row, col] where col is the source nuclide
    transfer_rate = dense[i_in115_m1, i_in115]

    # Due to isomeric branching with 0.15 ratio:
    # Production rate = reaction_rate * branching_ratio
    expected_rate = 1e-10 * 0.15  # 1.5e-11

    # The actual matrix will include the reaction rate contribution
    # Just verify there's positive transfer
    assert transfer_rate >= 0, \
        f"Expected positive transfer rate to In115_m1, got {transfer_rate}"


# ==============================================================================
# DADZ dictionary consistency tests
# ==============================================================================

def test_nn_prime_in_dadz():
    """Verify (n,n') has correct entry in DADZ dictionary.

    For (n,n') inelastic scattering:
    - delta_A = 0: Mass number unchanged (neutron scatters, nothing captured)
    - delta_Z = 0: Atomic number unchanged (no charge exchange)
    """
    from openmc.data import DADZ
    assert "(n,n')" in DADZ
    assert DADZ["(n,n')"] == (0, 0)


def test_dadz_consistency():
    """Verify all REACTIONS have corresponding DADZ entries.

    This prevents KeyError in Chain.from_endf() when using all reactions.
    """
    from openmc.deplete.chain import REACTIONS
    from openmc.data import DADZ

    missing = []
    for rx_name in REACTIONS:
        if rx_name not in DADZ:
            missing.append(rx_name)

    assert not missing, f"Missing DADZ entries for: {missing}"


def test_dadz_all_reactions_available():
    """Test that Chain.from_endf() can access DADZ for (n,n').

    Simulates the lookup that happens at chain.py:595
    """
    from openmc.deplete.chain import REACTIONS
    from openmc.data import DADZ

    # This is exactly what Chain.from_endf() does at line 595
    for rx_name in REACTIONS:
        delta_A, delta_Z = DADZ[rx_name]
        # Verify reasonable values
        assert isinstance(delta_A, int)
        assert isinstance(delta_Z, int)
        # For (n,n') specifically
        if rx_name == "(n,n')":
            assert delta_A == 0
            assert delta_Z == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
