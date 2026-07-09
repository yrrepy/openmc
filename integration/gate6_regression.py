#!/usr/bin/env python
"""Gate 6 - flag-off byte-identity regression (work-order A8) + pathways smoke.

Re-runs the Wave-0 baseline collapse (captured PRE-change,
``claude/urr_gate6_baseline/baseline_collapse.py``) with the post-change code and
asserts the four ``urr_material_dilution=False`` collapse outputs are
byte-identical (matching sha256) to the pre-change baseline. Then re-runs the
existing pathways smoke test. The baseline directory is backed up and restored so
this runner never mutates the canonical baseline.

Run from the clone root::

    python integration/gate6_regression.py
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path('/home/perry/Projects/OMC_Development/PENDF')
BASELINE_DIR = ROOT / 'claude/urr_gate6_baseline'
BASELINE_SCRIPT = BASELINE_DIR / 'baseline_collapse.py'
SMOKE = ROOT / 'integration/smoke_pendf_pathways.py'

# Byte-identity targets captured pre-change (Wave 0).
EXPECTED = {
    'pointwise_data.npy':
        'b9a7374e2ae67f72675715e9fa7341dea63a30e3fb92143cf0775c3581c8f173',
    'pointwise_micro.csv':
        '897264a2180848d343879645c1235d80800b6db35a15cc4497a7e51dd6e2ba34',
    'grouped_data.npy':
        '7539832031785e2fb733ec1f236d9e4babfff1d4b44ea44277309b07854389f4',
    'grouped_micro.csv':
        '238167b4cc6aa6a79039d6003f62091c6dbfd1fe90d1de63747f117ff30bb1c4',
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    # Back up the whole baseline directory so the canonical files are preserved
    # regardless of the outcome.
    backup = Path(tempfile.mkdtemp(prefix='gate6_backup_'))
    for f in BASELINE_DIR.iterdir():
        if f.is_file():
            shutil.copy(f, backup / f.name)

    try:
        print('Re-running the flag-off baseline collapse (post-change code) ...')
        proc = subprocess.run(
            [sys.executable, str(BASELINE_SCRIPT)],
            capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])
            print('GATE 6: FAIL (baseline_collapse.py errored)')
            sys.exit(1)

        results = {}
        for name, exp in EXPECTED.items():
            got = sha256(BASELINE_DIR / name)
            results[name] = (got == exp, got)
    finally:
        # Restore the canonical baseline (identical on PASS; protects it on FAIL).
        for f in backup.iterdir():
            shutil.copy(f, BASELINE_DIR / f.name)
        shutil.rmtree(backup, ignore_errors=True)

    baseline_ok = all(ok for ok, _ in results.values())

    # Pathways smoke test.
    print('Re-running the pathways smoke test ...')
    smoke = subprocess.run([sys.executable, str(SMOKE)],
                           capture_output=True, text=True)
    smoke_ok = smoke.returncode == 0
    smoke_tail = (smoke.stdout.strip().splitlines() or [''])[-1]

    print('=' * 78)
    print('Gate 6 - flag-off byte-identity regression + pathways smoke')
    print('=' * 78)
    print(f'{"output":22s} {"match":>6s}  sha256')
    print('-' * 78)
    for name, (ok, got) in results.items():
        print(f'{name:22s} {"Y" if ok else "N":>6s}  {got}')
    print('-' * 78)
    print(f'flag-off collapse byte-identical to pre-change baseline: '
          f'{"PASS" if baseline_ok else "FAIL"}')
    print(f'pathways smoke test (smoke_pendf_pathways.py):           '
          f'{"PASS" if smoke_ok else "FAIL"}')
    if not smoke_ok:
        print(smoke.stdout[-2000:])
        print(smoke.stderr[-2000:])
    else:
        print(f'    {smoke_tail}')
    all_ok = baseline_ok and smoke_ok
    print('=' * 78)
    print(f'GATE 6: {"PASS" if all_ok else "FAIL"}')
    print('=' * 78)
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
