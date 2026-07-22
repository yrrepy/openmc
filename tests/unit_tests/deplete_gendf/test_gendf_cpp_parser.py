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

from .gendf_testing import endf6_line, write_synthetic_gendf

lib_gendf = pytest.importorskip('openmc.lib.gendf')

# 3-group grid; group boundaries 1 eV .. 1 GeV
ENERGY_BOUNDS = np.array([1.0, 1.0e3, 1.0e6, 1.0e9])

# Analytic expectations for the synthetic files below
MF3_XS = [1.1, 2.2, 3.3]           # MT=102 (n,gamma) group XS
MF10_XS = [0.15, 0.25, 0.35]       # MT=102 LFS=1 production XS (-> Al28)
MF10_XS_L0 = [0.11, 0.21, 0.31]    # a distinct LFS=0 production XS
MF10_IZAP = 13028
MF10_LFS = 1

# A valid full-range MF=3 section (NP = n_groups+1 = 4): three group values plus
# the top-boundary dummy 99.0 (dropped on alignment). ENDF-formatted TAB1 pairs.
_FULL_MF3 = [('1.000000+0', '1.100000+0'), ('1.000000+3', '2.200000+0'),
             ('1.000000+6', '3.300000+0'), ('1.000000+9', '9.900000+1')]

# Full-range MF=10 TAB1 points (3 groups + top-boundary dummy) whose aligned XS
# are MF10_XS_L0 and MF10_XS respectively.
_MF10_L0 = [('1.000000+0', '1.100000-1'), ('1.000000+3', '2.100000-1'),
            ('1.000000+6', '3.100000-1'), ('1.000000+9', '9.900000+1')]
_MF10_L1 = [('1.000000+0', '1.500000-1'), ('1.000000+3', '2.500000-1'),
            ('1.000000+6', '3.500000-1'), ('1.000000+9', '9.900000+1')]


def _mf_section(mf, points, *, mat, mt=102, izap=None, lfs=0, nr=1):
    """ENDF-6 lines for one MF=3 or MF=10 TAB1 section (NP = len(points))."""
    np_ = len(points)
    if mf == 3:
        head = endf6_line(['0.0', '0.0', 0, 0, nr, np_], mat, 3, mt)
    else:  # MF=10: IZAP at L1, LFS at L2
        head = endf6_line(['-1.305820+7', '-1.305820+7', izap, lfs, nr, np_],
                          mat, 10, mt)
    lines = [head, endf6_line([np_, 1, '', '', '', ''], mat, mf, mt)]
    flat = [v for pair in points for v in pair]
    for k in range(0, len(flat), 6):
        chunk = list(flat[k:k + 6])
        chunk += [''] * (6 - len(chunk))
        lines.append(endf6_line(chunk, mat, mf, mt))
    return lines


def _write_gendf_mf10(path, mf3, mf10_subs, *, mat=1325):
    """GENDF file: one MF=3 MT=102 section plus a list of MF=10 subsections.

    ``mf10_subs`` is ``[(mt, points, izap, lfs), ...]``; consecutive entries with
    the same MT become multiple LFS/IZAP levels under that MT.
    """
    lines = [endf6_line(['1.302700+4', '2.675000+1', 0, 0, 0, 7], mat, 1, 451)]
    lines += [endf6_line(['synthetic', 'test', 'file', '', '', ''], mat, 1, 451)
              for _ in range(6)]
    lines += _mf_section(3, mf3, mat=mat)
    for mt, pts, izap, lfs in mf10_subs:
        lines += _mf_section(10, pts, mat=mat, mt=mt, izap=izap, lfs=lfs)
    with open(path, 'w') as f:
        f.writelines(lines)


def _write_gendf(path, *, mf3=None, mf10=None, mat=1325):
    """Write a minimal GENDF file with a bespoke MF=3 and/or MF=10 MT=102 section.

    ``mf3`` is a list of ENDF-formatted ``(energy, xs)`` TAB1 pairs; ``mf10`` is
    ``(points, izap, lfs)``. A valid MF=3 section is always required for the file
    to load, so pass ``mf3`` (or rely on the caller providing one).
    """
    # Six text records keep the file above the parser's 10-line floor even for
    # a single tiny threshold section.
    lines = [endf6_line(['1.302700+4', '2.675000+1', 0, 0, 0, 7], mat, 1, 451)]
    lines += [endf6_line(['synthetic', 'test', 'file', '', '', ''], mat, 1, 451)
              for _ in range(6)]
    if mf3 is not None:
        lines += _mf_section(3, mf3, mat=mat)
    if mf10 is not None:
        pts, izap, lfs = mf10
        lines += _mf_section(10, pts, mat=mat, izap=izap, lfs=lfs)
    with open(path, 'w') as f:
        f.writelines(lines)


def _python_fast_mf3_section(filepath, mt):
    """Raw MF=3 (energies, xs) from the Python fast parser (before alignment)."""
    py = pytest.importorskip('openmc.deplete.gendf')
    # _parse_gendf_mf3_only touches no instance state; call on a bare instance
    cls = py._PythonGENDFLibrary
    section = cls._parse_gendf_mf3_only(cls.__new__(cls), Path(filepath))
    sigma = section[(3, mt)]['sigma']
    return np.asarray(sigma.x), np.asarray(sigma.y)


def _python_aligned_xs(energies, xs, bounds):
    """Group-aligned XS from the Python backend's ``_align_to_group_grid``.

    The Python backend is the correctness reference for group placement (R1-62):
    coverage is decided from the energy grid, and the top-boundary dummy dropped.
    """
    py = pytest.importorskip('openmc.deplete.gendf')
    lib = py._PythonGENDFLibrary.__new__(py._PythonGENDFLibrary)
    lib.energy_bounds = np.asarray(bounds, dtype=float)
    lib.n_groups = len(bounds) - 1
    return lib._align_to_group_grid(
        np.asarray(energies, dtype=float), np.asarray(xs, dtype=float),
        'parity', strict_alignment=False)


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
    """MF=3 with NR=2: C++ backend matches Python backend and analytic values."""
    lib_dir = tmp_path / 'gendf'
    lib_dir.mkdir()
    gfile = lib_dir / 'Al27g.asc'
    write_synthetic_gendf(gfile, 'nr_skip')

    lib = lib_gendf.GENDFLibrary(str(lib_dir), ENERGY_BOUNDS, 'test-3g')
    cpp_xs = lib.get_xs('Al27', 102, ENERGY_BOUNDS)

    np.testing.assert_allclose(cpp_xs, MF3_XS)

    # Dual-backend parity on the *aligned* XS: parse the same file with the
    # Python fast parser, then align via the Python backend's reference grid
    # logic (the C++ and Python group placements must agree).
    energies, raw_xs = _python_fast_mf3_section(gfile, 102)
    py_xs = _python_aligned_xs(energies, raw_xs, ENERGY_BOUNDS)
    np.testing.assert_allclose(cpp_xs, py_xs)


def test_get_xs_rejects_bad_n_energy_bounds(tmp_path):
    """C-API get_xs validates n_energy_bounds before building a vector (R1-33)."""
    import ctypes

    from openmc.lib import _dll
    from openmc.exceptions import OpenMCError

    lib_dir = tmp_path / 'gendf'
    lib_dir.mkdir()
    write_synthetic_gendf(lib_dir / 'Al27g.asc', 'nr_skip')
    lib = lib_gendf.GENDFLibrary(str(lib_dir), ENERGY_BOUNDS, 'test-3g')

    # Bypass the wrapper (which always passes len(energy_bounds)) and hit the raw
    # C-API with counts that do not match the library grid (n_groups + 1). A
    # negative/oversized count would otherwise over-read the raw pointer.
    bounds = ENERGY_BOUNDS.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    xs = ctypes.POINTER(ctypes.c_double)()
    ng = ctypes.c_int()
    for bad_n in (-1, 0, len(ENERGY_BOUNDS) + 1):
        with pytest.raises(OpenMCError):
            _dll.openmc_gendf_get_xs(lib._lib_id, b'Al27', 102, bad_n, bounds,
                                     ctypes.pointer(xs), ctypes.pointer(ng))


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


def test_mt5_mf10_skipped_silently(tmp_path, capfd):
    """MF=10 MT=5 (lumped, multi-product LFS=0) is dropped silently; other MTs
    are untouched.  MT=5 is not consumed by any depletion pathway and its many
    LFS=0 products would collide on key 5000, so it is skipped at parse with no
    IZAP=0 / Z=0 warnings."""
    gfile = tmp_path / 'Al27g.asc'
    # MT=5: three LFS=0 products (neutron IZAP=1, residual IZAP=13027, trailing
    # IZAP=0). MT=102: real LFS=0 and LFS=1 levels.
    _write_gendf_mf10(gfile, _FULL_MF3, [
        (5, _MF10_L1, 1, 0), (5, _MF10_L0, 13027, 0), (5, _MF10_L1, 0, 0),
        (102, _MF10_L0, 13027, 0), (102, _MF10_L1, 13028, 1)])

    lib = lib_gendf.GENDFLibrary(str(tmp_path), ENERGY_BOUNDS, 'test-3g')
    # First access triggers the load (and would surface any parser warnings).
    assert lib._get_production_xs('Al27', 5) == []          # MT=5 dropped
    cap = capfd.readouterr()
    assert 'WARNING' not in (cap.err + cap.out)             # silent, incl. MT=5

    levels = lib._get_production_xs('Al27', 102)            # MT=102 intact
    assert [(lfs, izap) for lfs, izap, _ in levels] == [(0, 13027), (1, 13028)]
    np.testing.assert_allclose(levels[0][2], MF10_XS_L0)
    np.testing.assert_allclose(levels[1][2], MF10_XS)


def test_mf10_key_collision_warns(tmp_path, capfd):
    """Two MF=10 subsections of one MT sharing an (LFS) key but differing in IZAP
    warn exactly once; the last subsection wins."""
    gfile = tmp_path / 'Al27g.asc'
    # MT=16, two LFS=0 subsections -> both key 16000; IZAP 13027 then 13028.
    _write_gendf_mf10(gfile, _FULL_MF3, [
        (16, _MF10_L0, 13027, 0), (16, _MF10_L1, 13028, 0)])

    lib = lib_gendf.GENDFLibrary(str(tmp_path), ENERGY_BOUNDS, 'test-3g')
    levels = lib._get_production_xs('Al27', 16)             # triggers load
    cap = capfd.readouterr()
    flat = ' '.join((cap.err + cap.out).split())            # undo line wrapping
    assert flat.count('MF=10 store collision') == 1
    assert 'MT=16 LFS=0: IZAP=13028 overwrites IZAP=13027' in flat

    assert len(levels) == 1                                 # last one wins
    lfs, izap, xs = levels[0]
    assert (lfs, izap) == (0, 13028)
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


# ---------------------------------------------------------------------------
# Group-grid alignment (R1-62 / K20): coverage is decided from the stored
# energy grid, the TAB1 top-boundary dummy is dropped, and oversized sections
# fail loudly instead of yielding a silent all-zero cross section.
# ---------------------------------------------------------------------------

def test_np_equals_ngroups_threshold_parity(tmp_path):
    """R1-62(1): a threshold section with NP == n_groups is placed by energy.

    The section starts one group above the grid bottom, so NP (two real groups
    plus the top-boundary dummy) equals n_groups == 3. The old backend matched
    the ``size == n_groups`` branch and returned the data as full-grid values
    starting at group 0 -- shifting everything down and keeping the dummy. The
    fix keys off the energy grid, so the C++ result must match the Python
    reference group-by-group.
    """
    gfile = tmp_path / 'Fe56g.asc'
    pts = [('1.000000+3', '4.000000+0'),   # group 1
           ('1.000000+6', '5.000000+0'),   # group 2
           ('1.000000+9', '9.900000+1')]   # top-boundary dummy (must vanish)
    _write_gendf(gfile, mf3=pts)

    lib = lib_gendf.GENDFLibrary(str(tmp_path), ENERGY_BOUNDS, 'test-3g')
    cpp_xs = lib.get_xs('Fe56', 102, ENERGY_BOUNDS)

    # Correct placement: group 0 empty, reals in groups 1-2, no dummy anywhere.
    np.testing.assert_allclose(cpp_xs, [0.0, 4.0, 5.0])

    py_xs = _python_aligned_xs([1.0e3, 1.0e6, 1.0e9], [4.0, 5.0, 99.0],
                               ENERGY_BOUNDS)
    np.testing.assert_allclose(cpp_xs, py_xs)


def test_threshold_dummy_not_leaked_above_section_end(tmp_path):
    """R1-62(2): a section ending below the grid top leaves higher groups at 0.

    The band covers group 0 only and ends at the group-1 boundary; the TAB1
    top-boundary dummy must never be copied into group 1. The old threshold
    branch used ``min(size, n_groups - start)`` and leaked the dummy.
    """
    gfile = tmp_path / 'Fe56g.asc'
    pts = [('1.000000+0', '7.000000+0'),   # group 0
           ('1.000000+3', '9.900000+1')]   # section-end dummy (must vanish)
    _write_gendf(gfile, mf3=pts)

    lib = lib_gendf.GENDFLibrary(str(tmp_path), ENERGY_BOUNDS, 'test-3g')
    cpp_xs = lib.get_xs('Fe56', 102, ENERGY_BOUNDS)

    np.testing.assert_allclose(cpp_xs, [7.0, 0.0, 0.0])

    py_xs = _python_aligned_xs([1.0, 1.0e3], [7.0, 99.0], ENERGY_BOUNDS)
    np.testing.assert_allclose(cpp_xs, py_xs)


def test_oversized_section_raises_not_zeroed(tmp_path):
    """K20 / R1-10: a section with more points than a full grid raises on both
    the MF=3 and MF=10 lanes -- never a silent all-zero cross section."""
    from openmc.exceptions import OpenMCError

    # 5 points > n_groups+1 (= 4) on the 3-group grid.
    over = [('1.000000+0', '1.0'), ('1.000000+3', '2.0'), ('1.000000+6', '3.0'),
            ('1.000000+9', '4.0'), ('1.000000+9', '5.0')]

    # MF=3 lane: get_xs must throw (matches its pre-existing size-mismatch throw).
    d3 = tmp_path / 'mf3'
    d3.mkdir()
    _write_gendf(d3 / 'Fe56g.asc', mf3=over)
    lib3 = lib_gendf.GENDFLibrary(str(d3), ENERGY_BOUNDS, 'test-3g')
    with pytest.raises(OpenMCError):
        lib3.get_xs('Fe56', 102, ENERGY_BOUNDS)

    # MF=10 lane: get_production_xs must throw too (K20 previously zeroed it).
    d10 = tmp_path / 'mf10'
    d10.mkdir()
    _write_gendf(d10 / 'Fe56g.asc', mf3=_FULL_MF3, mf10=(over, 13028, 1))
    lib10 = lib_gendf.GENDFLibrary(str(d10), ENERGY_BOUNDS, 'test-3g')
    with pytest.raises(OpenMCError):
        lib10._get_production_xs('Fe56', 102)
