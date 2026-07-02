"""Regression test for the C++ MF=10 parser IZAP=0 handling.

An MF=10 subsection with IZAP=0 (product unspecified — a data quality issue
seen in e.g. Al27/Am241 in JEFF-3.3) must be *consumed* by the parser, not
skipped at the head line. The old behavior left the subsection's data lines
to be re-scanned as potential subsection heads; since std::stoi() accepted
float prefixes ("7.700000+0" -> 7), data lines were misread as new heads,
producing garbage production levels and/or swallowing the real subsections
that followed.
"""

import numpy as np
import pytest

lib_gendf = pytest.importorskip('openmc.lib.gendf')


def _line(fields, mat, mf, mt):
    """Format one ENDF-6 line: six 11-char fields + MAT(4) MF(2) MT(3)."""
    body = ''.join(f'{f:>11}' for f in fields)
    body = body.ljust(66)
    return f'{body}{mat:>4}{mf:>2}{mt:>3}\n'


def _write_synthetic_gendf(path):
    """Al27-like GENDF file: MF=10/MT=102 has an IZAP=0 subsection (LFS=0)
    followed by a real subsection (IZAP=13028, LFS=1). 3 groups."""
    mat = 1325
    lines = []

    # MF=1, MT=451 header (ZA is a float field on the HEAD record)
    lines.append(_line(['1.302700+4', '2.675000+1', 0, 0, 0, 5], mat, 1, 451))
    for _ in range(4):
        lines.append(_line(['synthetic', 'test', 'file', '', '', ''], mat, 1, 451))

    # MF=3, MT=102: TAB1 head (NR=1, NP=3), interp line, 3 (E, xs) pairs
    lines.append(_line(['0.0', '0.0', 0, 0, 1, 3], mat, 3, 102))
    lines.append(_line([3, 1, '', '', '', ''], mat, 3, 102))
    lines.append(_line(['1.000000+0', '1.100000+0', '1.000000+3',
                        '2.200000+0', '1.000000+6', '3.300000+0'], mat, 3, 102))

    # MF=10, MT=102 subsection 1: IZAP=0 (must be consumed, not stored).
    # The data line's 6th field ("7.700000+0") is exactly what the old
    # parser misread as a new head via stoi() -> NP=7.
    lines.append(_line(['-1.305820+7', '-1.305820+7', 0, 0, 1, 3], mat, 10, 102))
    lines.append(_line([3, 1, '', '', '', ''], mat, 10, 102))
    lines.append(_line(['1.000000+0', '9.900000+0', '1.000000+3',
                        '8.800000+0', '1.000000+6', '7.700000+0'], mat, 10, 102))

    # MF=10, MT=102 subsection 2: real level IZAP=13028 (Al28), LFS=1
    lines.append(_line(['-1.305820+7', '-1.328660+7', 13028, 1, 1, 3], mat, 10, 102))
    lines.append(_line([3, 1, '', '', '', ''], mat, 10, 102))
    lines.append(_line(['1.000000+0', '1.500000-1', '1.000000+3',
                        '2.500000-1', '1.000000+6', '3.500000-1'], mat, 10, 102))

    with open(path, 'w') as f:
        f.writelines(lines)


def test_izap0_subsection_consumed_not_rescanned(tmp_path):
    lib_dir = tmp_path / 'gendf'
    lib_dir.mkdir()
    _write_synthetic_gendf(lib_dir / 'Al27g.asc')

    energy_bounds = np.array([1.0, 1.0e3, 1.0e6, 1.0e9])
    lib = lib_gendf.GENDFLibrary(str(lib_dir), energy_bounds, 'test-3g')

    levels = lib._get_production_xs('Al27', 102)

    # Exactly the real subsection survives: the IZAP=0 level is dropped and
    # its data lines must not be misread as heads (no garbage levels, and the
    # real LFS=1 subsection after it must not be swallowed).
    assert len(levels) == 1
    lfs, izap, xs = levels[0]
    assert lfs == 1
    assert izap == 13028
    np.testing.assert_allclose(xs, [0.15, 0.25, 0.35])
