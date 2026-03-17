# GENDF Isomeric Branching — Remaining Fixes

Date: 2026-03-20
Branch: uncommitted changes on active branch (helpers.py, independent_operator.py)

---

## CRITICAL

### 1. Shape mismatch in runtime-mode `get_branching_ratios`
**File:** `openmc/deplete/gendf.py:2117-2118`
**Bug:** `energies=self.energy_bounds.copy()` has n_groups+1 values, `branching_ratios=br` has shape [n_products, n_groups]. Causes index-out-of-bounds in `helpers.py:1490` via `_compute_isomeric_indices` clipping to [0, n_groups].
**Fix:** Use `self.energy_bounds[:-1].copy()` for energies (n_groups values matching branching_ratios columns). Patcher mode already uses matched lengths.

### 2. `AttributeError` not caught for C++ backend
**File:** `openmc/deplete/helpers.py:1258-1263`
**Bug:** except catches `KeyError, ValueError, NotImplementedError, OpenMCError` but C++ backend raises `AttributeError` ('GENDFLibrary' object has no attribute 'get_branching_ratios').
**Fix:** Add `AttributeError` to the except tuple at line 1258.

---

## HIGH

### 3. Dead sparse_table code in IsomericBranchingHelper
**File:** `openmc/deplete/helpers.py:1201-1202, 1228-1236, 1628-1635`
**Bug:** `__init__` still accepts `branching_cache`/`sparse_table` params. Sparse table lookup code in `_calculate_weighted` is unreachable (never passed from operator).
**Fix:** Remove `branching_cache`/`sparse_table` params from `__init__`, remove sparse table lookup block in `_calculate_weighted`, remove `_nuc_to_idx`/`_rxn_to_idx` dict setup.

### 4. `isomeric_branching_embedded` lost in `reduce()`
**File:** `openmc/deplete/chain.py:1696-1702`
**Bug:** `reduce()` copies `isomeric_branching_targets` and `_lfs` but not `_embedded`. Legacy chains with `<isomeric_yields>` energy-dependent ratios lose that data silently.
**Fix:** After line 1698, add filtering and copy of `self.isomeric_branching_embedded` to `new_chain`, keeping entries where parent is in `all_isotopes`.

### 5. Factory drops `mapping_mode` for C++ backend
**File:** `openmc/deplete/gendf.py:2447-2453`
**Bug:** `mapping_mode` param accepted but never forwarded to C++ backend. Silently ignored.
**Fix:** Add warning when `mapping_mode != 'elis'` and C++ backend is selected, or document the limitation.

---

## MEDIUM

### 6. No ELIS-without-decay guard
**File:** `openmc/deplete/gendf.py` — `_map_via_elis()` method
**Fix:** Add at top of `get_branching_ratios()`:
```python
if self._mapping_mode == 'elis' and self.decay_lookup is None:
    raise ValueError("ELIS mapping requires decay_file")
```

### 7. Pool.py param counting
**File:** `openmc/deplete/pool.py:153-157`
**Fix:** Replace `if len(params) >= 4:` with `if 'isomeric_branching' in sig.parameters:`

### 8. Stale `_nuclides_set_cache`
**File:** `openmc/deplete/gendf.py` — `_validate_metastable_name()` modifies `_file_index` without invalidating cache.
**Fix:** Add `self._nuclides_set_cache = None` after `_file_index` modification.

### 9. Backend API inconsistencies
**Files:** `openmc/lib/gendf.py` vs `openmc/deplete/gendf.py`
- `nuclide` vs `nuclide_name` param naming
- C++ missing `get_branching_ratios`, `get_production_xs`
**Fix:** Standardize param name to `nuclide_name` on C++ wrapper. Add `get_branching_ratios` with Python MF=10 fallback.

---

## LOW

10. Unused `order_idx` in `_map_via_elis` (gendf.py:1454)
11. Mixed `dict[...]` vs `Dict[...]` type annotations (gendf.py)
12. "May be redundant" comments in pool.py (lines 117, 152, 156)
13. Missing `len(products) == branching_array.shape[0]` assertion in `_build_branching_result`
