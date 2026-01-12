# Git Merge Completed: GENDF/Isomeric Branching Feature

**Date:** 2026-01-10
**Directory:** `~/Codes/OpenMC/OpenMC_GENDF_PR_merge/`
**Branch:** `feature/isomeric-branching-gendf`

---

## What This Directory Contains

This directory contains the **completed git 3-way merge** of the GENDF/isomeric branching feature into upstream OpenMC. Unlike other directories that used file copying, this merge properly preserves upstream changes while adding feature code.

### Key Distinction from Other Directories

| Directory | Method | Status |
|-----------|--------|--------|
| `OpenMC_GENDF_PR_merge/` (THIS) | **Git 3-way merge** | ✅ Complete, tests pass |
| `OpenMC_GENDF_PR/` | File copying | Has all commits but may miss upstream changes |
| `OpenMC_GENDF_PR_bad/` | Earlier attempt | Incomplete, rebased commits 0-6 only |

---

## Merge Process Executed

### Phase 1: Created B-git
- Copied cleaned feature files from `OpenMC-0.15.3dev_Isomerics_wGENDF-Final_mt4_Cleaned/`
- Initialized git with base commit `607f6babe` (Add distributed cell densities)
- Created single feature commit `76ecb5a4c` with all changes

### Phase 2: Setup Target Directory
- Checked out `origin/GENDF_IsomericBranching_InelasticAct` (commit `8c8867ea1`)
- Created feature branch `feature/isomeric-branching-gendf`

### Phase 3: 3-Way Merge
- Added B-git as remote `cleaned-git`
- Executed `git merge cleaned-git/feature-all-changes --no-commit`
- Git's 3-way algorithm identified common ancestor and computed deltas

### Phase 4: Conflict Resolution
Four files had merge conflicts, all resolved:

| File | Conflict | Resolution |
|------|----------|------------|
| `chain.py` | Missing `add_redox_term()` | Kept upstream function |
| `microxs.py` | Zero flux handling | Kept cleaner version |
| `pool.py` | Duplicate imports | Removed duplicates |
| `stepresult.py` | Nuclide sorting, reactions check | Kept upstream logic |

### Phase 5: Semantic Commits
Created 11 semantic commits (commit messages borrowed from `OpenMC_GENDF_PR/`, prefixes removed):

### Phase 6: Verification
- Built C++ library with MPI support
- Ran unit tests: **127 passed**, 2 skipped, 2 env var errors

---

## Commit History

```
e8d61dd7e Add chain patcher for isomeric branching data
786cde564 Add comprehensive isomeric branching unit tests
c158ce429 Add GENDF-aware element and material expansion
f35e75586 Integrate isomeric branching into depletion operators
ecb36148c Add IsomericBranchingHelper for flux-weighted branching
9e3e78345 Add MicroXS GENDF integration for IndependentOperator
464cbc8d3 Add C++ GENDF reader for fast cross-section extraction
db940b506 Add GENDF library with dual Python/C++ backends
0fe094455 Add (n,n') inelastic scattering reaction support
2ae973f80 Add isomeric branching data structures and chain reduction
45b9dbf07 Add ELIS-based isomeric target mapping
8c8867ea1 Add --merge-mode-functions=separate to gcovr call in CI (#3716)  <-- upstream base
```

---

## Upstream Changes Preserved

The 3-way merge correctly preserved these upstream changes that occurred after `607f6babe`:

| Change | File | Source |
|--------|------|--------|
| `add_redox_term()` method | `chain.py` | PR #2783 |
| `VERSION_RESULTS = (1, 2)` | `stepresult.py` | Recent upstream |
| SciPy sparse array migration | `chain.py`, `pool.py`, `abc.py` | Commit `9c91bddf0` |
| Reactions existence check | `stepresult.py` | Recent upstream |
| Material constructor updates | `material.py` | PR #3649 |
| SCALE-999 group structure | `mgxs/__init__.py` | Commit `c0f302db6` |

---

## Files Changed (37 total)

### New Files (12)
- `openmc/deplete/decay_elis.py` - ELIS mapping module
- `openmc/deplete/gendf.py` - GENDF library
- `openmc/lib/gendf.py` - Python bindings for C++ GENDF
- `include/openmc/gendf.h` - C++ GENDF header
- `src/gendf.cpp` - C++ GENDF implementation
- `src/gendf_parser.cpp` - GENDF parser for chain patcher
- `tools/add_gendf_isomeric_branching_to_chain.py` - Chain patcher tool
- `tests/unit_tests/test_chain_reduce_isomeric.py`
- `tests/unit_tests/test_chain_reduce_isomeric_siblings.py`
- `tests/unit_tests/test_gendf_elis_mapping.py`
- `tests/unit_tests/test_gendf_nuclide_filtering.py`
- `tests/unit_tests/test_isomeric_branching_weighting.py`
- `tests/unit_tests/test_nn_prime_reaction.py`

### Modified Files (25)
- `CMakeLists.txt`
- `include/openmc/capi.h`
- `openmc/data/data.py`
- `openmc/data/reaction.py`
- `openmc/deplete/__init__.py`
- `openmc/deplete/_matrix_funcs.py`
- `openmc/deplete/abc.py`
- `openmc/deplete/atom_number.py`
- `openmc/deplete/chain.py`
- `openmc/deplete/coupled_operator.py`
- `openmc/deplete/helpers.py`
- `openmc/deplete/independent_operator.py`
- `openmc/deplete/microxs.py`
- `openmc/deplete/openmc_operator.py`
- `openmc/deplete/pool.py`
- `openmc/deplete/stepresult.py`
- `openmc/element.py`
- `openmc/material.py`
- `openmc/mgxs/__init__.py`
- `openmc/mgxs/groups.py`
- `src/reaction.cpp`
- `tests/chain_simple.xml`
- `tests/dummy_operator.py`

---

## Build Configuration Used

```bash
cmake -DCMAKE_INSTALL_PREFIX=../ \
      -DCMAKE_PREFIX_PATH=/home/perry/Codes/MC_Utils/mcpl-2.2.0/lib64/cmake/MCPL/ \
      -DHDF5_PREFER_PARALLEL=on \
      -DOPENMC_USE_MPI=on \
      -Dxtensor_DIR=/home/perry/Codes/xtensor/build \
      -DCMAKE_CXX_FLAGS="-I/home/perry/Codes/xtensor/include -I/home/perry/Codes/xtl/include" \
      ..
```

---

## Test Results

```
tests/unit_tests/test_deplete_chain.py             - 20 passed
tests/unit_tests/test_chain_reduce_isomeric.py     - 11 passed
tests/unit_tests/test_chain_reduce_isomeric_siblings.py - 10 passed
tests/unit_tests/test_isomeric_branching_weighting.py   - 17 passed
tests/unit_tests/test_gendf_nuclide_filtering.py   - 8 passed
tests/unit_tests/test_gendf_elis_mapping.py        - 27 passed
tests/unit_tests/test_nn_prime_reaction.py         - 14 passed, 2 skipped
tests/unit_tests/test_deplete_fission_yields.py    - 13 passed
tests/unit_tests/test_deplete_microxs.py           - 5 passed
tests/unit_tests/test_deplete_cram.py              - 2 passed

Total: 127 passed, 2 skipped, 2 errors (env var only)
```

---

## Next Steps

1. **Push to remote:** `git push -u origin feature/isomeric-branching-gendf`
2. **Create PR** against `develop` or appropriate target branch
3. **Run full CI** to verify all regression tests pass
