"""Regression test for the C++ GENDF parser's TAB1 interpolation-line skip.

ENDF-6 TAB1 records store the interpolation table as ``2*NR`` integers
(NBT/INT pairs) packed six per line, i.e. ``ceil(2*NR/6) = (2*NR + 5)//6``
lines -- *not* ``NR`` lines. For the universal GROUPR case ``NR == 1`` both
formulas give 1, so real files were unaffected; but for ``NR >= 2`` the old
C++ parser skipped ``NR`` lines and thus swallowed a data line:

  * MF=3, NR=2: the reaction's single data line was skipped, so the section
    stored no cross-sections at all (``get_xs`` -> reaction-not-found).
  * MF=10, NR=2: the subsection's data line was skipped, so the production
    level was dropped (``_get_production_xs`` -> empty list).

The fix (``interp_table_lines`` in ``src/gendf_parser.cpp``) matches the
Python fast parser's ``(2*n1 + 5)//6`` arithmetic. This test exercises both
the MF=3 and MF=10 paths with NR=2 through the C++ backend and cross-checks
the MF=3 result against the Python fast parser.
"""

from pathlib import Path

import numpy as np
import pytest

lib_gendf = pytest.importorskip('openmc.lib.gendf')

# 3-group grid; group boundaries 1 eV .. 1 GeV
ENERGY_BOUNDS = np.array([1.0, 1.0e3, 1.0e6, 1.0e9])

# Analytic expectations for the synthetic file below
MF3_XS = [1.1, 2.2, 3.3]           # MT=102 (n,gamma) group XS
MF10_XS = [0.15, 0.25, 0.35]       # MT=102 LFS=1 production XS (-> Al28)
MF10_IZAP = 13028
MF10_LFS = 1


def _line(fields, mat, mf, mt):
    """Format one ENDF-6 line: six 11-char fields + MAT(4) MF(2) MT(3)."""
    body = ''.join(f'{f:>11}' for f in fields)
    body = body.ljust(66)
    return f'{body}{mat:>4}{mf:>2}{mt:>3}\n'


def _write_synthetic_gendf(path):
    """Al27-like GENDF file whose MF=3 and MF=10 TAB1 records use NR=2.

    With NR=2 the interpolation table occupies exactly one line (4 packed
    integers), so a correct parser skips one line; the buggy parser skipped
    two and lost the following data line.
    """
    mat = 1325
    lines = []

    # MF=1, MT=451 header (ZA is a float field on the HEAD record)
    lines.append(_line(['1.302700+4', '2.675000+1', 0, 0, 0, 5], mat, 1, 451))
    for _ in range(4):
        lines.append(_line(['synthetic', 'test', 'file', '', '', ''], mat, 1, 451))

    # MF=3, MT=102: TAB1 head (NR=2, NP=3), 1 interp line (4 ints), 1 data line
    lines.append(_line(['0.0', '0.0', 0, 0, 2, 3], mat, 3, 102))
    lines.append(_line([2, 2, 3, 2, '', ''], mat, 3, 102))
    lines.append(_line(['1.000000+0', '1.100000+0', '1.000000+3',
                        '2.200000+0', '1.000000+6', '3.300000+0'], mat, 3, 102))

    # MF=10, MT=102: one real level (IZAP=13028 Al28, LFS=1), also NR=2
    lines.append(_line(['-1.305820+7', '-1.328660+7', 13028, 1, 2, 3], mat, 10, 102))
    lines.append(_line([2, 2, 3, 2, '', ''], mat, 10, 102))
    lines.append(_line(['1.000000+0', '1.500000-1', '1.000000+3',
                        '2.500000-1', '1.000000+6', '3.500000-1'], mat, 10, 102))

    with open(path, 'w') as f:
        f.writelines(lines)


def _python_fast_mf3_xs(filepath, mt):
    """MF=3 group XS from the Python fast parser (independent ground truth)."""
    py = pytest.importorskip('openmc.deplete.gendf')
    # _parse_gendf_mf3_only touches no instance state; call on a bare instance
    cls = py._PythonGENDFLibrary
    section = cls._parse_gendf_mf3_only(cls.__new__(cls), Path(filepath))
    return np.asarray(section[(3, mt)]['sigma'].y)


def test_mf3_nr2_xs_and_python_parity(tmp_path):
    """MF=3 with NR=2: C++ backend matches Python parser and analytic values.

    Pre-fix the C++ parser skipped NR=2 lines, swallowing the lone data line,
    so MT=102 was never stored and get_xs raised reaction-not-found.
    """
    lib_dir = tmp_path / 'gendf'
    lib_dir.mkdir()
    gfile = lib_dir / 'Al27g.asc'
    _write_synthetic_gendf(gfile)

    lib = lib_gendf.GENDFLibrary(str(lib_dir), ENERGY_BOUNDS, 'test-3g')
    cpp_xs = lib.get_xs('Al27', 102, ENERGY_BOUNDS)

    np.testing.assert_allclose(cpp_xs, MF3_XS)

    # Parity: independent Python fast parser on the same file
    py_xs = _python_fast_mf3_xs(gfile, 102)
    np.testing.assert_allclose(cpp_xs, py_xs)


def test_mf10_nr2_production_level(tmp_path):
    """MF=10 with NR=2: production level survives and has correct XS.

    Pre-fix the subsection's data line was swallowed, dropping the level
    entirely (_get_production_xs returned an empty list).
    """
    lib_dir = tmp_path / 'gendf'
    lib_dir.mkdir()
    _write_synthetic_gendf(lib_dir / 'Al27g.asc')

    lib = lib_gendf.GENDFLibrary(str(lib_dir), ENERGY_BOUNDS, 'test-3g')
    levels = lib._get_production_xs('Al27', 102)

    assert len(levels) == 1
    lfs, izap, xs = levels[0]
    assert lfs == MF10_LFS
    assert izap == MF10_IZAP
    np.testing.assert_allclose(xs, MF10_XS)


def _write_negative_xs_gendf(path):
    """GENDF file with a negative MF=3 cross-section to trigger a clamp warning."""
    mat = 2631
    lines = []
    lines.append(_line(['2.605600+4', '5.545400+1', 0, 0, 0, 9], mat, 1, 451))
    for _ in range(8):  # keep total >= min_file_lines (10)
        lines.append(_line(['synthetic', 'test', 'file', '', '', ''], mat, 1, 451))

    # NR=1, NP=3 with a negative XS in the middle group
    lines.append(_line(['0.0', '0.0', 0, 0, 1, 3], mat, 3, 102))
    lines.append(_line([3, 2, '', '', '', ''], mat, 3, 102))
    lines.append(_line(['1.000000+0', '1.100000+0', '1.000000+3',
                        '-5.000000+0', '1.000000+6', '3.300000+0'], mat, 3, 102))

    with open(path, 'w') as f:
        f.writelines(lines)


def test_parser_warnings_surface_to_stderr(tmp_path, capfd):
    """P1-4: parser diagnostics reach the user via warning() (stderr)."""
    lib_dir = tmp_path / 'gendf'
    lib_dir.mkdir()
    _write_negative_xs_gendf(lib_dir / 'Fe56g.asc')

    lib = lib_gendf.GENDFLibrary(str(lib_dir), ENERGY_BOUNDS, 'test-3g')
    # First material access triggers load_from_file, which surfaces warnings.
    xs = lib.get_xs('Fe56', 102, ENERGY_BOUNDS)

    # Negative value clamped to zero in the middle group
    np.testing.assert_allclose(xs, [1.1, 0.0, 3.3])

    captured = capfd.readouterr()
    combined = captured.err + captured.out
    assert 'WARNING' in combined
    assert 'Negative XS' in combined
