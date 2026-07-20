"""Shared fixtures for the GENDF / isomeric-branching unit tests.

Reusable *callables* (mock libraries, chain factories, ENDF-6 writers, the bare
operator, path resolution) live in the importable sibling module
``gendf_testing`` -- test files use ``from .gendf_testing import ...``. This file
holds only pytest *fixtures*.

Real-data fixtures resolve under two env vars (fallbacks preserve behaviour on
the original machine when unset):

    export OPENMC_GENDF_TEST_DATA=/home/perry/NukeData
    export OPENMC_GENDF_TEST_CHAINS=/home/perry/Codes/OpenMC/chains   # optional

Expected layout under ``$OPENMC_GENDF_TEST_DATA``:

    Activation/FISPACT/TENDL2017data/tal2017-n/gxs-709/     (gendf_dir)
    Activation/FISPACT/TENDL2017data/tal2017-n/decay_2020.endf
    Activation/decay/decay_2020.endf | decay2020.endf       (decay_file fallbacks)
    Activation/FISPACT/JEFF33data/decay/                    (jeff33_decay_path)
    Activation/FISPACT/JEFF33data/jeff33-n/gxs-709/         (jeff33_gendf_path)
    Activation/ukdd-12_decay.dat                            (ukdd12_path)

and under ``$OPENMC_GENDF_TEST_CHAINS``:

    chain_activator_TENDL2017_ccfe709_lfs.xml | ..._ccfe709.xml   (chain_path)

Any missing path -> ``pytest.skip`` with a clear reason.
"""

import pytest

from openmc.deplete.decay_elis import DecayState

from .gendf_testing import (
    gendf_chains_root, gendf_data_root, first_existing, require_existing)


# ---------------------------------------------------------------------------
# Real-data fixtures (env-var driven; skip when missing)
# ---------------------------------------------------------------------------

@pytest.fixture
def gendf_dir():
    """TENDL-2017 GENDF (gxs-709) directory."""
    return require_existing(
        gendf_data_root() / 'Activation/FISPACT/TENDL2017data/tal2017-n/gxs-709',
        "GENDF directory not found")


@pytest.fixture
def decay_file():
    """TENDL-2017 decay ENDF file (first of several known locations)."""
    root = gendf_data_root()
    return first_existing([
        root / 'Activation/FISPACT/TENDL2017data/tal2017-n/decay_2020.endf',
        root / 'Activation/decay/decay_2020.endf',
        root / 'Activation/decay/decay2020.endf',
    ], "Decay file not found")


@pytest.fixture
def chain_path():
    """Activation chain XML (LFS variant preferred)."""
    root = gendf_chains_root()
    return first_existing([
        root / 'chain_activator_TENDL2017_ccfe709_lfs.xml',
        root / 'chain_activator_TENDL2017_ccfe709.xml',
    ], "Chain file not found")


@pytest.fixture
def jeff33_decay_path():
    """JEFF-3.3 decay directory."""
    return require_existing(
        gendf_data_root() / 'Activation/FISPACT/JEFF33data/decay/',
        "JEFF33 decay data not available")


@pytest.fixture
def jeff33_gendf_path():
    """JEFF-3.3 GENDF (gxs-709) directory."""
    return require_existing(
        gendf_data_root() / 'Activation/FISPACT/JEFF33data/jeff33-n/gxs-709/',
        "JEFF33 GENDF data not available")


@pytest.fixture
def ukdd12_path():
    """UKDD-12 decay file."""
    return require_existing(
        gendf_data_root() / 'Activation/ukdd-12_decay.dat',
        "UKDD12 decay data not available")


# ---------------------------------------------------------------------------
# Ir-192 DecayState lookup (mock; no real data)
# ---------------------------------------------------------------------------

@pytest.fixture
def ir192_decay_lookup():
    """Mock decay lookup for Ir-192 ground/m1/m2 states."""
    return {(77, 192): [
        DecayState(z=77, a=192, elis=0.0, liso=0),
        DecayState(z=77, a=192, elis=56720.0, liso=1),
        DecayState(z=77, a=192, elis=168140.0, liso=2),
    ]}


@pytest.fixture
def ir192_lookup(ir192_decay_lookup):
    """Alias of ``ir192_decay_lookup`` (name used by test_gendf_minor_batch)."""
    return ir192_decay_lookup


# ---------------------------------------------------------------------------
# Deterministic warnings (subdir-wide autouse)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_gendf_warn_dedup():
    """Reset module-level warn-once stores so warnings are deterministic."""
    import openmc.deplete.decay_elis as de
    import openmc.deplete.gendf as g
    import openmc.deplete.helpers as h
    g._WARNED_RUNTIME_BRANCHING.clear()
    de._WARNED_ELIS_AMBIGUITY.clear()
    h._WARNED_ISOMERIC_NORMALIZE.clear()
    yield
    g._WARNED_RUNTIME_BRANCHING.clear()
    de._WARNED_ELIS_AMBIGUITY.clear()
    h._WARNED_ISOMERIC_NORMALIZE.clear()
