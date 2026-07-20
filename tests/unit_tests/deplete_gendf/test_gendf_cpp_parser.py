"""Regression tests for the C++ GENDF parser (openmc.lib.gendf).

Two independent parser bugs are covered:

* **MF=10 IZAP=0 handling.** A subsection with IZAP=0 (product unspecified — a
  data quality issue seen in e.g. Al27/Am241 in JEFF-3.3) must be *consumed* by
  the parser, not skipped at the head line. The old behavior left the
  subsection's data lines to be re-scanned as potential subsection heads; since
  std::stoi() accepted float prefixes ("7.700000+0" -> 7), data lines were
  misread as new heads, producing garbage production levels and/or swallowing
  the real subsections that followed.

* **TAB1 interpolation-line skip.** ENDF-6 TAB1 records store the interpolation
  table as ``2*NR`` integers (NBT/INT pairs) packed six per line, i.e.
  ``ceil(2*NR/6) = (2*NR + 5)//6`` lines -- *not* ``NR`` lines. For the
  universal GROUPR case ``NR == 1`` both formulas give 1, so real files were
  unaffected; but for ``NR >= 2`` the old C++ parser skipped ``NR`` lines and
  thus swallowed a data line (MF=3 -> no cross-sections stored; MF=10 -> the
  production level dropped). The fix (``interp_table_lines`` in
  ``src/gendf_parser.cpp``) matches the Python fast parser's ``(2*n1 + 5)//6``.

Synthetic GENDF files are built by the shared ``write_synthetic_gendf`` writer.
"""

from pathlib import Path

import numpy as np
import pytest

from .gendf_testing import write_synthetic_gendf

lib_gendf = pytest.importorskip('openmc.lib.gendf')

# 3-group grid; group boundaries 1 eV .. 1 GeV
ENERGY_BOUNDS = np.array([1.0, 1.0e3, 1.0e6, 1.0e9])

# Analytic expectations for the synthetic files below
MF3_XS = [1.1, 2.2, 3.3]           # MT=102 (n,gamma) group XS
MF10_XS = [0.15, 0.25, 0.35]       # MT=102 LFS=1 production XS (-> Al28)
MF10_IZAP = 13028
MF10_LFS = 1


def _python_fast_mf3_xs(filepath, mt):
    """MF=3 group XS from the Python fast parser (independent ground truth)."""
    py = pytest.importorskip('openmc.deplete.gendf')
    # _parse_gendf_mf3_only touches no instance state; call on a bare instance
    cls = py._PythonGENDFLibrary
    section = cls._parse_gendf_mf3_only(cls.__new__(cls), Path(filepath))
    return np.asarray(section[(3, mt)]['sigma'].y)


def test_izap0_subsection_consumed_not_rescanned(tmp_path):
    """MF=10 IZAP=0 subsection is consumed, not rescanned as garbage heads."""
    lib_dir = tmp_path / 'gendf'
    lib_dir.mkdir()
    write_synthetic_gendf(lib_dir / 'Al27g.asc', 'izap0')

    lib = lib_gendf.GENDFLibrary(str(lib_dir), ENERGY_BOUNDS, 'test-3g')
    levels = lib._get_production_xs('Al27', 102)

    # Exactly the real subsection survives: the IZAP=0 level is dropped and
    # its data lines must not be misread as heads (no garbage levels, and the
    # real LFS=1 subsection after it must not be swallowed).
    assert len(levels) == 1
    lfs, izap, xs = levels[0]
    assert lfs == MF10_LFS
    assert izap == MF10_IZAP
    np.testing.assert_allclose(xs, MF10_XS)


def test_mf3_nr2_xs_and_python_parity(tmp_path):
    """MF=3 with NR=2: C++ backend matches Python parser and analytic values."""
    lib_dir = tmp_path / 'gendf'
    lib_dir.mkdir()
    gfile = lib_dir / 'Al27g.asc'
    write_synthetic_gendf(gfile, 'nr_skip')

    lib = lib_gendf.GENDFLibrary(str(lib_dir), ENERGY_BOUNDS, 'test-3g')
    cpp_xs = lib.get_xs('Al27', 102, ENERGY_BOUNDS)

    np.testing.assert_allclose(cpp_xs, MF3_XS)

    # Parity: independent Python fast parser on the same file
    py_xs = _python_fast_mf3_xs(gfile, 102)
    np.testing.assert_allclose(cpp_xs, py_xs)


def test_mf10_nr2_production_level(tmp_path):
    """MF=10 with NR=2: production level survives and has correct XS."""
    lib_dir = tmp_path / 'gendf'
    lib_dir.mkdir()
    write_synthetic_gendf(lib_dir / 'Al27g.asc', 'nr_skip')

    lib = lib_gendf.GENDFLibrary(str(lib_dir), ENERGY_BOUNDS, 'test-3g')
    levels = lib._get_production_xs('Al27', 102)

    assert len(levels) == 1
    lfs, izap, xs = levels[0]
    assert lfs == MF10_LFS
    assert izap == MF10_IZAP
    np.testing.assert_allclose(xs, MF10_XS)


def test_parser_warnings_surface_to_stderr(tmp_path, capfd):
    """P1-4: parser diagnostics reach the user via warning() (stderr)."""
    lib_dir = tmp_path / 'gendf'
    lib_dir.mkdir()
    write_synthetic_gendf(lib_dir / 'Fe56g.asc', 'negative_xs')

    lib = lib_gendf.GENDFLibrary(str(lib_dir), ENERGY_BOUNDS, 'test-3g')
    # First material access triggers load_from_file, which surfaces warnings.
    xs = lib.get_xs('Fe56', 102, ENERGY_BOUNDS)

    # Negative value clamped to zero in the middle group
    np.testing.assert_allclose(xs, [1.1, 0.0, 3.3])

    captured = capfd.readouterr()
    combined = captured.err + captured.out
    assert 'WARNING' in combined
    assert 'Negative XS' in combined


def test_metastable_not_resolved_to_ground_state(tmp_path):
    """R1-1: a metastable request must not silently return ground-state data.

    With only a ground-state file (Al27g.asc -> 'Al27') in the library, a
    request for the metastable 'Al27_m1' (or the legacy 'Al27m' alias) must
    report not-found on BOTH backends -- never fall back to the ground file.
    The removed Python ``_find_gendf_file`` filename-guessing fallback used to
    hand back the ground data for exactly this request. Pins backend parity of
    the exact-name resolution rule.
    """
    from openmc.deplete.gendf import _PythonGENDFLibrary
    from openmc.exceptions import OpenMCError

    lib_dir = tmp_path / 'gendf'
    lib_dir.mkdir()
    write_synthetic_gendf(lib_dir / 'Al27g.asc', 'izap0')  # ground only

    # --- Python backend --------------------------------------------------
    # Instantiate the backend directly: the GENDFLibrary factory auto-selects
    # C++ when the compiled lib is present and no decay_file is given.
    py = _PythonGENDFLibrary(
        lib_dir, validate_energy_grid=False, _energy_structure='CCFE-709')

    # Ground state resolves and carries MF=3 data.
    assert py.has_nuclide('Al27') is True
    assert (3, 102) in py._load_material('Al27').section_data

    # Canonical metastable: not found, and access raises KeyError (no fallback
    # to the ground file).
    assert py.has_nuclide('Al27_m1') is False
    with pytest.raises(KeyError):
        py.get_xs('Al27_m1', 102)

    # Legacy alias spelling (ground name + 'm', no underscore): also rejected
    # (the removed alias acceptance).
    assert py.has_nuclide('Al27m') is False
    with pytest.raises(KeyError):
        py.get_xs('Al27m', 102)

    # --- C++ backend (parity) -------------------------------------------
    # Guarded by the module-level pytest.importorskip('openmc.lib.gendf').
    cpp = lib_gendf.GENDFLibrary(str(lib_dir), ENERGY_BOUNDS, 'test-3g')

    assert cpp.has_nuclide('Al27') is True
    np.testing.assert_allclose(cpp.get_xs('Al27', 102, ENERGY_BOUNDS), MF3_XS)

    assert cpp.has_nuclide('Al27_m1') is False
    with pytest.raises(OpenMCError):
        cpp.get_xs('Al27_m1', 102, ENERGY_BOUNDS)
