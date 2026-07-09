"""Capture and summarize the PENDF build warning families into a text log.

Two PENDF build tools spray :class:`UserWarning` during library builds, one per
offending (nuclide, MT) -- enough to bury the console. This module records them
without changing what reaches the console, then renders a fixed-width summary
log (counts block + top-10 tables) in the aesthetic of the isomer-mapping logs.

The two warning families (we own both format strings; the regexes below are
anchored to the current wording):

A) ``openmc/data/pendf.py`` -- an MF=10 reaction with no MF=3 sibling section,
   emitted while ``tools/pendf_to_hdf5.py`` builds a pointwise library::

       "{file}: {nuclide} MF=10 MT={mt} has no MF=3 section; isomeric partials
        dropped."

B) ``tools/pendf_group_bin.py`` -- binned MF=10 partials disagree with the
   MF=3 total in some group, emitted while binning a grouped library::

       "{nuc} MT={mt}: binned MF=10 partials sum to {partials} b but the MF=3
        total is {total} b in group {group} (max relative deviation {rel} >
        {tol})."

:class:`WarningCapture` is a context manager that swaps
:func:`warnings.showwarning` for a hook which *records and re-emits* every
warning -- console output is byte-for-byte unchanged -- classifies each message
into family A, family B, or an "other" bucket, and stashes structured rows.
:func:`write_warning_log` renders the summary and prints one console line.

Both build tools run as loose scripts from ``tools/``, so a plain
``from pendf_warning_log import ...`` sibling import resolves for them.
"""

from __future__ import annotations

import re
import warnings
from collections import Counter, namedtuple

# ---------------------------------------------------------------------------
# Message parsers -- anchored to our own format strings (see module docstring).
# A float rendered by %e/%.Ne (e.g. '1.234560e+00', '3.707e-03', '1e-05').
# ---------------------------------------------------------------------------
_FLOAT = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'

# Family A: openmc/data/pendf.py -- MF=10 reaction with no MF=3 section.
_FAMILY_A_RE = re.compile(
    r'^(?P<file>.+?): (?P<nuclide>\S+) MF=10 MT=(?P<mt>\d+) '
    r'has no MF=3 section; isomeric partials dropped\.$')

# Family B: tools/pendf_group_bin.py -- binned partials vs MF=3 total mismatch.
_FAMILY_B_RE = re.compile(
    r'^(?P<nuclide>\S+) MT=(?P<mt>\d+): binned MF=10 partials sum to '
    r'(?P<partials>' + _FLOAT + r') b but the MF=3 total is '
    r'(?P<total>' + _FLOAT + r') b in group (?P<group>\d+) '
    r'\(max relative deviation (?P<rel>' + _FLOAT + r') > '
    r'(?P<tol>' + _FLOAT + r')\)\.$')

# Structured rows collected per family.
FamilyARow = namedtuple('FamilyARow', ['file', 'nuclide', 'mt'])
FamilyBRow = namedtuple(
    'FamilyBRow', ['nuclide', 'mt', 'group', 'partials', 'total', 'abs_dev',
                   'rel', 'tol'])


class WarningCapture:
    """Context manager: record and re-emit warnings, bucketed by family.

    On entry it replaces :func:`warnings.showwarning` with a hook that appends
    a structured row for every matching warning and then forwards the call to
    the previously installed handler, so the console sees exactly what it would
    have seen without the capture. Non-matching warnings are counted in the
    ``other`` bucket (their message text preserved for a unique-message tally).

    Attributes
    ----------
    family_a : list of FamilyARow
        MF=10-with-no-MF=3 drops (``openmc/data/pendf.py``).
    family_b : list of FamilyBRow
        binned-partials-vs-total mismatches (``tools/pendf_group_bin.py``),
        each carrying the absolute deviation ``|partials - total|`` in barns.
    other : list of str
        message text of every UserWarning matching neither family.
    """

    def __init__(self):
        self.family_a: list[FamilyARow] = []
        self.family_b: list[FamilyBRow] = []
        self.other: list[str] = []
        self._orig_showwarning = None

    @property
    def total(self) -> int:
        """Total number of warnings captured across all three buckets."""
        return len(self.family_a) + len(self.family_b) + len(self.other)

    def record(self, text: str) -> None:
        """Classify one warning message string into its bucket."""
        m = _FAMILY_A_RE.match(text)
        if m is not None:
            self.family_a.append(FamilyARow(
                file=m['file'], nuclide=m['nuclide'], mt=int(m['mt'])))
            return
        m = _FAMILY_B_RE.match(text)
        if m is not None:
            partials = float(m['partials'])
            total = float(m['total'])
            self.family_b.append(FamilyBRow(
                nuclide=m['nuclide'], mt=int(m['mt']), group=int(m['group']),
                partials=partials, total=total,
                abs_dev=abs(partials - total), rel=float(m['rel']),
                tol=float(m['tol'])))
            return
        self.other.append(text)

    def __enter__(self) -> 'WarningCapture':
        self._orig_showwarning = warnings.showwarning
        warnings.showwarning = self._showwarning
        return self

    def __exit__(self, *exc) -> bool:
        warnings.showwarning = self._orig_showwarning
        self._orig_showwarning = None
        return False

    def _showwarning(self, message, category, filename, lineno,
                     file=None, line=None):
        # Record first, then forward to the original handler so the console
        # behaviour is identical to running without the capture installed.
        self.record(str(message))
        self._orig_showwarning(message, category, filename, lineno, file, line)


# ---------------------------------------------------------------------------
# Fixed-width log rendering (isomer-mapping-log aesthetic).
# ---------------------------------------------------------------------------
# Family B table column layout, shared by header and rows.
_B_HEADER = (f"{'Rank':>4}  {'Nuclide':<12}  {'MT':>4}  {'Group':>6}  "
             f"{'partials[b]':>15}  {'MF=3 total[b]':>15}  "
             f"{'|Δ|[b]':>12}  {'rel dev':>10}")


def _fmt_b_row(rank: int, row: FamilyBRow) -> str:
    return (f"{rank:>4}  {row.nuclide:<12}  {row.mt:>4}  {row.group:>6}  "
            f"{row.partials:>15.6e}  {row.total:>15.6e}  "
            f"{row.abs_dev:>12.3e}  {row.rel:>10.3e}")


def _write_b_table(f, title: str, rows: list[FamilyBRow], note=None) -> None:
    """Write one TOP-10 family-B table (rows already sorted, worst first)."""
    f.write(title + "\n")
    f.write("-" * len(_B_HEADER) + "\n")
    f.write(_B_HEADER + "\n")
    f.write("-" * len(_B_HEADER) + "\n")
    if not rows:
        f.write("  (none)\n")
    for rank, row in enumerate(rows[:10], start=1):
        f.write(_fmt_b_row(rank, row) + "\n")
    if note:
        f.write("\n" + note + "\n")


def write_warning_log(log_file, capture: WarningCapture) -> int:
    """Write the fixed-width warning-summary log; print one console line.

    Parameters
    ----------
    log_file : path-like
        Destination text file.
    capture : WarningCapture
        A capture whose context has already exited (rows collected).

    Returns
    -------
    int
        Total number of warnings captured (also echoed to the console).
    """
    a_rows = capture.family_a
    b_rows = capture.family_b
    other = capture.other
    total = capture.total

    with open(log_file, 'w') as f:
        f.write("=" * len(_B_HEADER) + "\n")
        f.write("PENDF BUILD WARNING SUMMARY\n")
        f.write("=" * len(_B_HEADER) + "\n\n")

        # Right-aligned counts block.
        w = 52
        f.write(f"{'Total warnings captured':>{w}}: {total:6d}\n")
        f.write("-" * (w + 8) + "\n")
        f.write(f"{'A) MF=10 with no MF=3 section (dropped)':>{w}}: "
                f"{len(a_rows):6d}\n")
        f.write(f"{'B) binned partials vs MF=3 total mismatch':>{w}}: "
                f"{len(b_rows):6d}\n")
        f.write(f"{'Other UserWarnings':>{w}}: {len(other):6d}\n")

        # -------------------------------------------------------------- Family B
        f.write("\n\n" + "=" * len(_B_HEADER) + "\n")
        f.write("FAMILY B: binned MF=10 partials vs MF=3 total "
                "(pendf_group_bin)\n")
        f.write("=" * len(_B_HEADER) + "\n")
        f.write(f"Total: {len(b_rows)}\n\n")

        by_rel = sorted(b_rows, key=lambda r: r.rel, reverse=True)
        by_abs = sorted(b_rows, key=lambda r: r.abs_dev, reverse=True)
        rel_note = ("Note: a rel dev of 1.0 is usually the partials binning to "
                    "zero in a tiny-total\n      threshold group -- check the "
                    "absolute (|Δ|[b]) column for those rows.")
        _write_b_table(f, "TOP 10 BY RELATIVE DEVIATION", by_rel, note=rel_note)
        f.write("\n")
        _write_b_table(f, "TOP 10 BY ABSOLUTE DEVIATION", by_abs)

        # -------------------------------------------------------------- Family A
        f.write("\n\n" + "=" * len(_B_HEADER) + "\n")
        f.write("FAMILY A: MF=10 reactions with no MF=3 section "
                "(pendf_to_hdf5)\n")
        f.write("=" * len(_B_HEADER) + "\n")
        f.write(f"Total: {len(a_rows)}\n\n")
        a_header = f"  {'File':<40}  {'Nuclide':<12}  {'MT':>5}"
        f.write(a_header + "\n")
        f.write("  " + "-" * (len(a_header) - 2) + "\n")
        if not a_rows:
            f.write("  (none)\n")
        for row in sorted(a_rows, key=lambda r: (r.file, r.nuclide, r.mt)):
            f.write(f"  {row.file:<40}  {row.nuclide:<12}  {row.mt:>5}\n")

        # ---------------------------------------------------------- Other bucket
        uniq = Counter(other)
        f.write("\n\n" + "=" * len(_B_HEADER) + "\n")
        f.write("OTHER USER WARNINGS\n")
        f.write("=" * len(_B_HEADER) + "\n")
        f.write(f"Total: {len(other)} (unique messages: {len(uniq)})\n\n")
        if not other:
            f.write("  (none)\n")
        for msg, count in sorted(uniq.items(), key=lambda kv: (-kv[1], kv[0])):
            f.write(f"  {count:>6d}x  {msg}\n")

    print(f"Warning summary log written to: {log_file} ({total} warnings)")
    return total
