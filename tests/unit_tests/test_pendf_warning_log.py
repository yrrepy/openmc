"""Unit tests for ``tools/pendf_warning_log.py``.

Exercises the warning-capture context manager and the fixed-width summary log:

* both build-warning families (A: MF=10-with-no-MF=3 drops; B: binned
  partials-vs-total mismatches) are parsed out of the emitted message text and
  counted, with anything else falling into the "other" bucket;
* the two family-B top-10 tables sort independently -- one by relative
  deviation, one by absolute deviation -- so a fixture whose rel and abs
  orderings are exact reverses lands in opposite order in the two tables;
* family A is listed in full, sorted;
* every captured warning is *re-emitted* to the previously installed
  ``showwarning`` handler (console behaviour unchanged).

The tools live in ``tools/`` (not a package); the module is loaded by path and
``tools/`` is put on ``sys.path`` so its sibling import resolves the same way it
does when the tools run as scripts.
"""
import importlib.util
import re
import sys
import warnings
from pathlib import Path

import pytest

_TOOLS = Path(__file__).parents[2] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
_spec = importlib.util.spec_from_file_location(
    "pendf_warning_log", _TOOLS / "pendf_warning_log.py")
pwl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pwl)


# ---------------------------------------------------------------------------
# Message builders -- the exact current format strings of the two families, so
# the tests break if either warning's wording drifts from the parser.
# ---------------------------------------------------------------------------
def _family_a_msg(file, nuc, mt):
    return (f"{file}: {nuc} MF=10 MT={mt} has no "
            f"MF=3 section; isomeric partials dropped.")


def _family_b_msg(nuc, mt, partials, total, group, rel, tol=1e-5):
    return (f"{nuc} MT={mt}: binned MF=10 partials sum to "
            f"{partials:.6e} b but the MF=3 "
            f"total is {total:.6e} b in group {group} (max "
            f"relative deviation {rel:.3e} > "
            f"{tol:.0e}).")


def _capture_emit(messages, prev_handler=None):
    """Emit ``messages`` via ``warnings.warn`` inside a fresh WarningCapture.

    ``simplefilter('always')`` defeats the warnings registry so every call --
    including exact duplicates -- reaches the hook. An optional ``prev_handler``
    is installed as the pre-existing ``showwarning`` to probe re-emission.
    """
    cap = pwl.WarningCapture()
    with warnings.catch_warnings():
        warnings.simplefilter('always')
        if prev_handler is not None:
            warnings.showwarning = prev_handler
        with cap:
            for m in messages:
                warnings.warn(m)
    return cap


# Twelve family-B fixtures: rel rises with i while abs = |partials-total| falls
# with i, so "by relative" and "by absolute" are exact reverses of each other.
def _b_fixture_messages():
    msgs = []
    for i in range(12):
        rel = (i + 1) * 1e-3          # 1e-3 .. 1.2e-2, ascending in i
        abs_dev = 12 - i              # 12 .. 1, descending in i
        total = 100.0
        partials = total - abs_dev    # |partials-total| == abs_dev
        msgs.append(_family_b_msg(
            f"NucB{i:02d}", 100 + i, partials, total, group=i, rel=rel))
    return msgs


def _ranked_nuclides(block):
    """Pull the ``NucB##`` names in row order from a rendered table block."""
    return re.findall(r'^\s*\d+\s+(NucB\d+)', block, flags=re.MULTILINE)


# ---------------------------------------------------------------------------
# Classification + counts
# ---------------------------------------------------------------------------
def test_family_classification_and_counts():
    """Each message lands in the right bucket; totals add up."""
    a = [_family_a_msg("n-In115.pendf", "In115", 102),
         _family_a_msg("n-W180.tendl", "W180", 16),
         _family_a_msg("n-Cr50.pendf", "Cr50", 5)]
    b = _b_fixture_messages()                       # 12 entries
    other = ["skipping /x/y.pendf: boom",
             "some unrelated warning",
             "some unrelated warning"]              # duplicate on purpose
    cap = _capture_emit(a + b + other)

    assert len(cap.family_a) == 3
    assert len(cap.family_b) == 12
    assert len(cap.other) == 3
    assert cap.total == 18

    # Family A fields parsed correctly.
    row = next(r for r in cap.family_a if r.nuclide == "In115")
    assert (row.file, row.mt) == ("n-In115.pendf", 102)

    # Family B fields: abs_dev is computed as |partials - total|.
    b0 = next(r for r in cap.family_b if r.nuclide == "NucB00")
    assert b0.mt == 100 and b0.group == 0
    assert b0.total == pytest.approx(100.0)
    assert b0.partials == pytest.approx(88.0)
    assert b0.abs_dev == pytest.approx(12.0)
    assert b0.rel == pytest.approx(1e-3)


def test_top10_orderings_differ(tmp_path):
    """The two family-B tables sort independently and truncate to ten rows."""
    cap = _capture_emit(_b_fixture_messages())
    log = tmp_path / "warn.log"
    pwl.write_warning_log(log, cap)
    text = log.read_text()

    rel_block = text.split("TOP 10 BY RELATIVE DEVIATION")[1].split(
        "TOP 10 BY ABSOLUTE DEVIATION")[0]
    abs_block = text.split("TOP 10 BY ABSOLUTE DEVIATION")[1].split(
        "FAMILY A")[0]

    rel_order = _ranked_nuclides(rel_block)
    abs_order = _ranked_nuclides(abs_block)

    # Ten rows each (12 entries, top-10 truncation).
    assert len(rel_order) == 10
    assert len(abs_order) == 10

    # rel descending -> i = 11,10,...,2 ; abs descending -> i = 0,1,...,9.
    assert rel_order == [f"NucB{i:02d}" for i in range(11, 1, -1)]
    assert abs_order == [f"NucB{i:02d}" for i in range(0, 10)]

    # The two tables are genuinely different orderings.
    assert rel_order != abs_order
    # Lowest-rel entries dropped from the rel table; lowest-abs from the abs one.
    assert "NucB00" not in rel_order and "NucB01" not in rel_order
    assert "NucB10" not in abs_order and "NucB11" not in abs_order

    # The tiny-total caveat note is present under the relative table.
    assert "rel dev of 1.0" in rel_block


def test_family_a_listing_full_and_sorted(tmp_path):
    """Family A is listed in full, sorted by (file, nuclide, MT)."""
    a = [_family_a_msg("n-W180.tendl", "W180", 16),
         _family_a_msg("n-In115.pendf", "In115", 102),
         _family_a_msg("n-In115.pendf", "In115", 103)]
    cap = _capture_emit(a)
    log = tmp_path / "warn.log"
    pwl.write_warning_log(log, cap)
    text = log.read_text()

    a_block = text.split("FAMILY A")[1].split("OTHER USER WARNINGS")[0]
    assert "Total: 3" in a_block
    # Sorted by (file, nuclide, mt): In115/102, In115/103, then W180/16.
    files = re.findall(r'^\s+(\S+)\s+(\S+)\s+(\d+)\s*$', a_block,
                       flags=re.MULTILINE)
    assert files == [("n-In115.pendf", "In115", "102"),
                     ("n-In115.pendf", "In115", "103"),
                     ("n-W180.tendl", "W180", "16")]


def test_other_bucket_counts_unique(tmp_path):
    """The other bucket reports both the total and the unique-message count."""
    cap = _capture_emit(["boom", "boom", "boom", "different"])
    assert len(cap.other) == 4
    log = tmp_path / "warn.log"
    pwl.write_warning_log(log, cap)
    text = log.read_text()
    o_block = text.split("OTHER USER WARNINGS")[1]
    assert "Total: 4 (unique messages: 2)" in o_block
    assert "3x  boom" in o_block


def test_warnings_reemitted_to_previous_handler():
    """Every captured warning is forwarded to the pre-existing showwarning."""
    seen = []

    def prev_handler(message, category, filename, lineno, file=None, line=None):
        seen.append(str(message))

    msgs = ([_family_a_msg("n-In115.pendf", "In115", 102)]
            + _b_fixture_messages()[:3]
            + ["unrelated"])
    cap = _capture_emit(msgs, prev_handler=prev_handler)

    # Console hook saw exactly the emitted set (re-emission), and recording it
    # did not drop or duplicate anything relative to the capture's own total.
    assert seen == msgs
    assert len(seen) == cap.total == len(msgs)


def test_showwarning_restored_on_exit():
    """The hook is uninstalled when the context exits (no global leak)."""
    before = warnings.showwarning
    with pwl.WarningCapture():
        assert warnings.showwarning is not before
    assert warnings.showwarning is before


def test_write_returns_total_and_prints(tmp_path, capsys):
    """write_warning_log returns the count and prints the one-line summary."""
    cap = _capture_emit(_b_fixture_messages())
    log = tmp_path / "warn.log"
    n = pwl.write_warning_log(log, cap)
    assert n == 12
    out = capsys.readouterr().out
    assert f"Warning summary log written to: {log} (12 warnings)" in out
