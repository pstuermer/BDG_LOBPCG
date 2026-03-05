# Geometry-Aware Planewave Initial Guess — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace 1D-only planewave init with multi-dimensional k-vector seeding supporting elongated, ring, and harmonic trap geometries.

**Architecture:** Add `bdg_geom_hint_t` enum. Rename `BDG_INIT_DEFAULT` → `BDG_INIT_PLANEWAVE` (same value 0). Add static helper `fill_planewave_init` that enumerates k-vectors per geometry hint, sorts by |k|², seeds cos/sin pairs weighted by |ψ₀|. Both `bdg_solve_d` and `bdg_solve_z` call the helper.

**Tech Stack:** C99, qsort, existing `ctx->kx2` arrays

---

### Task 1: Update public API enums

**Files:**
- Modify: `include/bdg/bdg.h:29-34`

**Step 1: Replace init enum and add geometry hint**

Replace the `bdg_init_mode_t` enum (lines 29-34) with:

```c
typedef enum {
  BDG_INIT_PLANEWAVE,    /* Geometry-aware planewave seeding (default, value 0) */
  BDG_INIT_WF_WEIGHTED,  /* |gauss| * |psi0| for both halves (B-positive) */
  BDG_INIT_REUSE,        /* Previous modes + noise; set by bdg_reuse_modes */
  BDG_INIT_CUSTOM        /* User callback; must ensure B-positivity */
} bdg_init_mode_t;

typedef enum {
  BDG_GEOM_AUTO,         /* Lowest |k|^2 across all dimensions */
  BDG_GEOM_ELONGATED,    /* k-vectors along longest L dimension only */
  BDG_GEOM_RING          /* (kx,ky) pairs sorted by kx^2+ky^2, kz=0 */
} bdg_geom_hint_t;
```

Also update the `bdg_set_init_mode` doc comment (line 157-163) to mention that for `BDG_INIT_PLANEWAVE`, `param` carries the geometry hint:

```c
/**
 * Select eigenvector initialization strategy.
 * For BDG_INIT_CUSTOM, fn and param are required.
 * For BDG_INIT_PLANEWAVE, pass fn=NULL, param=(void*)(intptr_t)hint
 *   where hint is a bdg_geom_hint_t. param=NULL defaults to BDG_GEOM_AUTO.
 * Survives bdg_reset — set once, reused across sweeps.
 */
```

**Step 2: Update all BDG_INIT_DEFAULT references**

Replace `BDG_INIT_DEFAULT` with `BDG_INIT_PLANEWAVE` in:
- `examples/3d_jens.c:67` — change to `BDG_INIT_PLANEWAVE` with `(void *)(intptr_t)BDG_GEOM_ELONGATED`
- `examples/3d_dipolar_atom.c:60` — change to `BDG_INIT_PLANEWAVE` with `(void *)(intptr_t)BDG_GEOM_RING`
- `tests/test_io.c:315` — change to `BDG_INIT_PLANEWAVE`
- `src/solver.c:128,258` — handled in Task 3

Note: `bdg_set_init_mode` signature stays the same. For planewave: `bdg_set_init_mode(bdg, BDG_INIT_PLANEWAVE, NULL, (void *)(intptr_t)BDG_GEOM_ELONGATED)`.

**Step 3: Build to verify compilation**

```bash
source /storage/share/intel/ubuntu/setvars.sh
make clean && make lib
```

Expected: may fail on solver.c references to BDG_INIT_DEFAULT — that's fine, fixed in Task 3.

**Step 4: Commit**

```bash
git add include/bdg/bdg.h examples/3d_jens.c examples/3d_dipolar_atom.c tests/test_io.c
git commit -m "rename BDG_INIT_DEFAULT to BDG_INIT_PLANEWAVE, add bdg_geom_hint_t"
```

---

### Task 2: Implement k-vector sorting helper

**Files:**
- Modify: `src/solver.c` — add static helper before `bdg_solve_d`

**Step 1: Add k-vector struct and sorting helper**

Insert before `bdg_solve_d` (line 66):

```c
/* ================================================================
 * K-vector sorting for planewave init
 * ================================================================ */

typedef struct {
  f64 k2;           /* |k|^2 */
  uint64_t idx[3];  /* FFTW indices per dimension */
} kvec_entry_t;

static int kvec_cmp(const void *a, const void *b) {
  const f64 ka = ((const kvec_entry_t *)a)->k2;
  const f64 kb = ((const kvec_entry_t *)b)->k2;
  return (ka > kb) - (ka < kb);
}

/**
 * Enumerate and sort k-vectors by |k|^2. Returns sorted array of n_out entries.
 * Caller must free the returned array with safe_free.
 *
 * @param ctx       matmul context (for kx2, N, dim)
 * @param hint      geometry hint
 * @param n_needed  how many unique k-vectors to return
 * @param n_out     [out] actual number returned (<= n_needed)
 */
static kvec_entry_t *enumerate_kvecs(const matmul_ctx_t *ctx,
                                     bdg_geom_hint_t hint,
                                     uint64_t n_needed,
                                     uint64_t *n_out) {
  const uint64_t dim = ctx->dim;

  /* Determine per-dimension search range */
  uint64_t range[3] = {1, 1, 1};
  switch (hint) {
  case BDG_GEOM_ELONGATED: {
    /* Find longest dimension */
    uint64_t long_d = 0;
    for (uint64_t d = 1; d < dim; d++)
      if (ctx->L[d] > ctx->L[long_d]) long_d = d;
    range[long_d] = (ctx->N[long_d] < 2 * n_needed)
                  ? ctx->N[long_d] : 2 * n_needed;
    break;
  }
  case BDG_GEOM_RING:
    /* Only kx,ky (first two dims), kz=0 */
    for (uint64_t d = 0; d < dim && d < 2; d++)
      range[d] = (ctx->N[d] < 2 * n_needed)
               ? ctx->N[d] : 2 * n_needed;
    break;
  case BDG_GEOM_AUTO:
  default:
    for (uint64_t d = 0; d < dim; d++)
      range[d] = (ctx->N[d] < 2 * n_needed)
               ? ctx->N[d] : 2 * n_needed;
    break;
  }

  /* Count total candidates */
  const uint64_t total = range[0] * range[1] * range[2];

  /* Allocate and fill */
  kvec_entry_t *entries = xcalloc(total, sizeof(kvec_entry_t));
  uint64_t count = 0;
  for (uint64_t iz = 0; iz < range[2]; iz++) {
    const f64 kz2 = (dim > 2) ? ctx->kx2[2][iz] : 0.0;
    for (uint64_t iy = 0; iy < range[1]; iy++) {
      const f64 ky2 = (dim > 1) ? ctx->kx2[1][iy] : 0.0;
      for (uint64_t ix = 0; ix < range[0]; ix++) {
        const f64 kx2 = ctx->kx2[0][ix];
        entries[count].k2 = kx2 + ky2 + kz2;
        entries[count].idx[0] = ix;
        entries[count].idx[1] = iy;
        entries[count].idx[2] = iz;
        count++;
      }
    }
  }

  /* Sort by |k|^2 */
  qsort(entries, count, sizeof(kvec_entry_t), kvec_cmp);

  /* Return at most n_needed */
  *n_out = (count < n_needed) ? count : n_needed;
  return entries;
}
```

**Step 2: Build to verify compilation**

```bash
make clean && make lib
```

**Step 3: Commit**

```bash
git add src/solver.c
git commit -m "add kvec enumeration and sorting helper for planewave init"
```

---

### Task 3: Replace DEFAULT init with planewave seeding (real path)

**Files:**
- Modify: `src/solver.c:128-164` (bdg_solve_d DEFAULT case)

**Step 1: Replace the DEFAULT case in bdg_solve_d**

Replace `case BDG_INIT_DEFAULT:` block (lines 128-164) with:

```c
  case BDG_INIT_PLANEWAVE:
  default: {
    const f64 *wf = (const f64 *)ctx->wf;
    uint32_t seed = 42;

    /* Determine geometry hint from param */
    const bdg_geom_hint_t hint = (NULL != bdg->custom_init_param)
      ? (bdg_geom_hint_t)(intptr_t)bdg->custom_init_param
      : BDG_GEOM_AUTO;

    /* Get sorted k-vectors: need ceil(sizeSub/2) unique vectors */
    const uint64_t n_needed = (sizeSub + 1) / 2;
    uint64_t n_kvecs = 0;
    kvec_entry_t *kvecs = enumerate_kvecs(ctx, hint, n_needed, &n_kvecs);

    /* Precompute per-dimension wavenumbers: freq[d][idx] = 2*pi*f/L
     * FFTW index i maps to frequency i for i <= N/2, else i - N */
    for (uint64_t j = 0; j < sizeSub; j++) {
      const uint64_t kv_idx = (j + 1) / 2;  /* 0, 1, 1, 2, 2, ... */
      const int use_sin = (j % 2 == 1);

      /* Skip sin for k=0 (it's identically zero) */
      if (use_sin && kv_idx < n_kvecs && kvecs[kv_idx].k2 < 1e-30) {
        /* Fill with small noise instead */
        for (uint64_t i = 0; i < size; i++) {
          const f64 wf_w = (NULL != wf) ? fabs(wf[i]) : 1.0;
          const f64 val = 1e-3 * (xrand(&seed) - 0.5) * wf_w;
          alg->S[i + j * n] = val;
          alg->S[i + size + j * n] = val;
        }
        continue;
      }

      if (kv_idx >= n_kvecs) {
        /* Fallback: random for excess columns */
        for (uint64_t i = 0; i < size; i++) {
          const f64 wf_w = (NULL != wf) ? fabs(wf[i]) : 1.0;
          const f64 val = (xrand(&seed) - 0.5) * wf_w;
          alg->S[i + j * n] = val;
          alg->S[i + size + j * n] = val;
        }
        continue;
      }

      /* Get FFTW indices for this k-vector */
      const uint64_t *ki = kvecs[kv_idx].idx;

      /* Compute frequencies per dimension */
      f64 freq[3] = {0.0, 0.0, 0.0};
      for (uint64_t d = 0; d < ctx->dim; d++) {
        const int64_t f = (ki[d] <= ctx->N[d] / 2)
                        ? (int64_t)ki[d]
                        : (int64_t)ki[d] - (int64_t)ctx->N[d];
        freq[d] = 2.0 * M_PI * (f64)f / ctx->L[d];
      }

      /* Strides for multi-dim index decomposition */
      uint64_t stride[3] = {1, 1, 1};
      for (uint64_t d = 1; d < ctx->dim; d++)
        stride[d] = stride[d - 1] * ctx->N[d - 1];

      for (uint64_t i = 0; i < size; i++) {
        /* Compute k·r */
        f64 kr = 0.0;
        for (uint64_t d = 0; d < ctx->dim; d++) {
          const uint64_t i_d = (i / stride[d]) % ctx->N[d];
          const f64 r_d = (f64)i_d * ctx->L[d] / (f64)ctx->N[d];
          kr += freq[d] * r_d;
        }

        const f64 pw = use_sin ? sin(kr) : cos(kr);
        const f64 pert = 1e-4 * (xrand(&seed) - 0.5);
        const f64 wf_w = (NULL != wf) ? fabs(wf[i]) : 1.0;
        const f64 val = (pw + pert) * wf_w;

        alg->S[i + j * n] = val;
        alg->S[i + size + j * n] = val;
      }
    }

    safe_free((void **)&kvecs);
  }
  }
```

Also add `#include <stdint.h>` at the top of solver.c if not already present (for `intptr_t`). Add `#include <stdint.h>` — actually check: `bdg_internal.h` already includes `<stdint.h>`, so `intptr_t` should be available. But `intptr_t` is from `<stdint.h>` which is included.

**Step 2: Build**

```bash
make clean && make lib
```

**Step 3: Commit**

```bash
git add src/solver.c
git commit -m "replace DEFAULT init with geometry-aware planewave seeding (real path)"
```

---

### Task 4: Replace DEFAULT init with planewave seeding (complex path)

**Files:**
- Modify: `src/solver.c:258-287` (bdg_solve_z DEFAULT case)

**Step 1: Replace the DEFAULT case in bdg_solve_z**

Replace `case BDG_INIT_DEFAULT:` block (lines 258-287) with the same logic as Task 3, but using `c64` types:

```c
  case BDG_INIT_PLANEWAVE:
  default: {
    const c64 *wf = (const c64 *)ctx->wf;
    uint32_t seed = 42;

    const bdg_geom_hint_t hint = (NULL != bdg->custom_init_param)
      ? (bdg_geom_hint_t)(intptr_t)bdg->custom_init_param
      : BDG_GEOM_AUTO;

    const uint64_t n_needed = (sizeSub + 1) / 2;
    uint64_t n_kvecs = 0;
    kvec_entry_t *kvecs = enumerate_kvecs(ctx, hint, n_needed, &n_kvecs);

    for (uint64_t j = 0; j < sizeSub; j++) {
      const uint64_t kv_idx = (j + 1) / 2;
      const int use_sin = (j % 2 == 1);

      if (use_sin && kv_idx < n_kvecs && kvecs[kv_idx].k2 < 1e-30) {
        for (uint64_t i = 0; i < size; i++) {
          const f64 wf_w = (NULL != wf) ? cabs(wf[i]) : 1.0;
          const c64 val = 1e-3 * (xrand(&seed) - 0.5) * wf_w + 0.0 * I;
          alg->S[i + j * n] = val;
          alg->S[i + size + j * n] = val;
        }
        continue;
      }

      if (kv_idx >= n_kvecs) {
        for (uint64_t i = 0; i < size; i++) {
          const f64 wf_w = (NULL != wf) ? cabs(wf[i]) : 1.0;
          const c64 val = (xrand(&seed) - 0.5) * wf_w + 0.0 * I;
          alg->S[i + j * n] = val;
          alg->S[i + size + j * n] = val;
        }
        continue;
      }

      const uint64_t *ki = kvecs[kv_idx].idx;

      f64 freq[3] = {0.0, 0.0, 0.0};
      for (uint64_t d = 0; d < ctx->dim; d++) {
        const int64_t f = (ki[d] <= ctx->N[d] / 2)
                        ? (int64_t)ki[d]
                        : (int64_t)ki[d] - (int64_t)ctx->N[d];
        freq[d] = 2.0 * M_PI * (f64)f / ctx->L[d];
      }

      uint64_t stride[3] = {1, 1, 1};
      for (uint64_t d = 1; d < ctx->dim; d++)
        stride[d] = stride[d - 1] * ctx->N[d - 1];

      for (uint64_t i = 0; i < size; i++) {
        f64 kr = 0.0;
        for (uint64_t d = 0; d < ctx->dim; d++) {
          const uint64_t i_d = (i / stride[d]) % ctx->N[d];
          const f64 r_d = (f64)i_d * ctx->L[d] / (f64)ctx->N[d];
          kr += freq[d] * r_d;
        }

        const f64 pw = use_sin ? sin(kr) : cos(kr);
        const f64 pert = 1e-4 * (xrand(&seed) - 0.5);
        const f64 wf_w = (NULL != wf) ? cabs(wf[i]) : 1.0;
        const c64 val = ((pw + pert) * wf_w) + 0.0 * I;

        alg->S[i + j * n] = val;
        alg->S[i + size + j * n] = val;
      }
    }

    safe_free((void **)&kvecs);
  }
  }
```

**Step 2: Build and run tests**

```bash
make clean && make run-tests
```

Expected: all tests pass. The test_io test that checks `BDG_INIT_PLANEWAVE` should work since we updated it in Task 1.

**Step 3: Commit**

```bash
git add src/solver.c
git commit -m "add geometry-aware planewave seeding for complex path"
```

---

### Task 5: Test with 3d_jens

**Step 1: Build examples and run**

```bash
make clean && make examples
./build/ex_3d_jens.ex
```

**Step 2: Verify**

- 3d_jens uses `BDG_GEOM_ELONGATED` — should reproduce ≤29 iterations (same as old DEFAULT for elongated trap)
- If iteration count increases, investigate k-vector ordering

**Step 3: Commit examples if any adjustments needed**

---

### Task 6: Update REUSE fallthrough target

**Files:**
- Modify: `src/solver.c` — both d and z paths

**Step 1: Update REUSE fallthrough**

In both `bdg_solve_d` and `bdg_solve_z`, the `BDG_INIT_REUSE` case falls through to `BDG_INIT_WF_WEIGHTED` and also sets `bdg->init_mode = BDG_INIT_WF_WEIGHTED` (lines 104, 234). This should remain as-is — after consuming the reuse buffer, subsequent solves use WF_WEIGHTED which is reasonable.

Also update `bdg_reuse_modes` doc in `bdg.h:170` to say "reverts to BDG_INIT_WF_WEIGHTED" (already correct).

No code change needed here — just verify the fallthrough is correct.

**Step 2: Run full test suite**

```bash
make clean && make run-tests
```

**Step 3: Final commit if any cleanup needed**

```bash
git add -A && git commit -m "geometry-aware planewave init: final cleanup"
```
