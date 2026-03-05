#include "bdg_internal.h"
#include "lobpcg.h"
#include <stdlib.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @file solver.c
 * @brief LinearOperator wrapping and ilobpcg dispatch.
 *
 * Creates linop wrappers around matmulLrep, matmulSwap, precondLrep,
 * then calls d_ilobpcg or z_ilobpcg.
 *
 * Reference: ~/LREP_post/examples/3d_dipolar_atom.c (setup + solve flow)
 */

/* ================================================================
 * Linop adapters — thin wrappers matching LinearOperator matvec signature
 * ================================================================ */

/* --- Real (d) adapters --- */

static void matvec_lrep_d(const LinearOperator_d_t *op,
                           f64 *restrict x, f64 *restrict y) {
    matmul_ctx_t *ctx = (matmul_ctx_t *)op->ctx->data;
    matmulLrep_d(ctx, x, y);
}

static void matvec_swap_d(const LinearOperator_d_t *op,
                           f64 *restrict x, f64 *restrict y) {
    matmul_ctx_t *ctx = (matmul_ctx_t *)op->ctx->data;
    matmulSwap_d(ctx, x, y);
}

static void matvec_precond_d(const LinearOperator_d_t *op,
                              f64 *restrict x, f64 *restrict y) {
    matmul_ctx_t *ctx = (matmul_ctx_t *)op->ctx->data;
    precondLrep_d(ctx, x, y);
}

/* --- Complex (z) adapters --- */

static void matvec_lrep_z(const LinearOperator_z_t *op,
                           c64 *restrict x, c64 *restrict y) {
    matmul_ctx_t *ctx = (matmul_ctx_t *)op->ctx->data;
    matmulLrep_z(ctx, x, y);
}

static void matvec_swap_z(const LinearOperator_z_t *op,
                           c64 *restrict x, c64 *restrict y) {
    matmul_ctx_t *ctx = (matmul_ctx_t *)op->ctx->data;
    matmulSwap_z(ctx, x, y);
}

static void matvec_precond_z(const LinearOperator_z_t *op,
                              c64 *restrict x, c64 *restrict y) {
    matmul_ctx_t *ctx = (matmul_ctx_t *)op->ctx->data;
    precondLrep_z(ctx, x, y);
}

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

static kvec_entry_t *enumerate_kvecs(const matmul_ctx_t *ctx,
                                     bdg_geom_hint_t hint,
                                     uint64_t n_needed,
                                     uint64_t *n_out) {
  const uint64_t dim = ctx->dim;

  uint64_t range[3] = {1, 1, 1};
  switch (hint) {
  case BDG_GEOM_ELONGATED: {
    uint64_t long_d = 0;
    for (uint64_t d = 1; d < dim; d++)
      if (ctx->L[d] > ctx->L[long_d]) long_d = d;
    range[long_d] = (ctx->N[long_d] < 2 * n_needed)
                  ? ctx->N[long_d] : 2 * n_needed;
    break;
  }
  case BDG_GEOM_RING:
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

  const uint64_t total = range[0] * range[1] * range[2];
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

  qsort(entries, count, sizeof(kvec_entry_t), kvec_cmp);
  *n_out = (count < n_needed) ? count : n_needed;
  return entries;
}

/* ================================================================
 * bdg_solve_d — real path
 * ================================================================ */
int bdg_solve_d(bdg_t *bdg) {
  matmul_ctx_t *ctx = bdg->ctx;
  const uint64_t size = ctx->size;
  const uint64_t n = 2 * size;
  const uint64_t nev = bdg->nev;
  const uint64_t sizeSub = bdg->sizeSub;

  /* Free any previous results (allows re-solve without reset) */
  safe_free((void **)&bdg->eigvals);
  safe_free((void **)&bdg->modes_u);
  safe_free((void **)&bdg->modes_v);

  /* 1. Allocate ilobpcg state */
  d_lobpcg_t *alg = d_ilobpcg_alloc(n, nev, sizeSub);

  /* 2. Create linop context — stack-allocated, must outlive A/B/T linops.
   *    All three share this ctx because they all wrap the same matmul_ctx_t.
   *    cleanup=NULL so linop_destroy won't attempt to free it. */
  linop_ctx_t linop_ctx = { .data = ctx, .data_size = sizeof(matmul_ctx_t) };

  /* 3. Create linear operators */
  LinearOperator_d_t *A = linop_create_d(n, n, matvec_lrep_d, NULL, &linop_ctx);
  LinearOperator_d_t *B = linop_create_d(n, n, matvec_swap_d, NULL, &linop_ctx);
  LinearOperator_d_t *T = linop_create_d(n, n, matvec_precond_d, NULL, &linop_ctx);

  /* 4. Set solver parameters */
  alg->A = A;
  alg->B = B;
  alg->T = T;
  alg->maxIter = bdg->maxIter;
  alg->tol = bdg->tol;

  /* 5. Initialize search vectors */
  switch (bdg->init_mode) {
  case BDG_INIT_REUSE:
    if (NULL != bdg->reuse_buf && bdg->reuse_n == n && bdg->reuse_cols == sizeSub) {
      memcpy(alg->S, bdg->reuse_buf, n * sizeSub * sizeof(f64));
      safe_free((void **)&bdg->reuse_buf);
      bdg->init_mode = BDG_INIT_WF_WEIGHTED;
      break;
    }
    /* fallthrough */
  case BDG_INIT_WF_WEIGHTED: {
    const f64 *wf = (const f64 *)ctx->wf;
    uint32_t seed = time(NULL);
    for (uint64_t j = 0; j < sizeSub; j++) {
      for (uint64_t i = 0; i < size; i++) {
        f64 u1 = (f64) rand_r(&seed)/RAND_MAX;//xrand(&seed);
        f64 u2 = (f64) rand_r(&seed)/RAND_MAX;//xrand(&seed);
        u1 = fmax(u1, 1e-10);
        const f64 val = fabs(sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2))
                      * ((NULL != wf) ? fabs(wf[i]) : 1.0);
        alg->S[i        + j * n] = val;
        alg->S[i + size + j * n] = val;
      }
    }
    break;
  }
  case BDG_INIT_CUSTOM:
    if (NULL != bdg->custom_init_fn)
      bdg->custom_init_fn(bdg, alg->S, n, sizeSub, bdg->custom_init_param);
    break;
  case BDG_INIT_PLANEWAVE:
  default: {
    const f64 *wf = (const f64 *)ctx->wf;
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

      /* sin(0) = 0, fill with noise instead */
      if (use_sin && kv_idx < n_kvecs && kvecs[kv_idx].k2 < 1e-30) {
        for (uint64_t i = 0; i < size; i++) {
          const f64 wf_w = (NULL != wf) ? fabs(wf[i]) : 1.0;
          const f64 val = 1e-3 * (xrand(&seed) - 0.5) * wf_w;
          alg->S[i + j * n] = val;
          alg->S[i + size + j * n] = val;
        }
        continue;
      }

      if (kv_idx >= n_kvecs) {
        for (uint64_t i = 0; i < size; i++) {
          const f64 wf_w = (NULL != wf) ? fabs(wf[i]) : 1.0;
          const f64 val = (xrand(&seed) - 0.5) * wf_w;
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
        const f64 wf_w = (NULL != wf) ? fabs(wf[i]) : 1.0;
        const f64 val = (pw + pert) * wf_w;

        alg->S[i + j * n] = val;
        alg->S[i + size + j * n] = val;
      }
    }

    safe_free((void **)&kvecs);
  }
  }

  /* 6. Solve */
  d_ilobpcg(alg);

  /* 7. Extract results */
  bdg->converged = alg->converged;

  bdg->eigvals = xcalloc(nev, sizeof(f64));
  memcpy(bdg->eigvals, alg->eigVals, nev * sizeof(f64));

  f64 *modes_u = xcalloc(size * nev, sizeof(f64));
  f64 *modes_v = xcalloc(size * nev, sizeof(f64));
  for (uint64_t j = 0; j < nev; j++) {
    memcpy(&modes_u[j * size], &alg->S[j * n], size * sizeof(f64));
    memcpy(&modes_v[j * size], &alg->S[j * n + size], size * sizeof(f64));
  }
  bdg->modes_u = modes_u;
  bdg->modes_v = modes_v;

  /* 8. Cleanup */
  linop_destroy_d(&A);
  linop_destroy_d(&B);
  linop_destroy_d(&T);
  d_lobpcg_free(&alg);

  return 0;
}

/* ================================================================
 * bdg_solve_z — complex path
 * ================================================================ */
int bdg_solve_z(bdg_t *bdg) {
  matmul_ctx_t *ctx = bdg->ctx;
  const uint64_t size = ctx->size;
  const uint64_t n = 2 * size;
  const uint64_t nev = bdg->nev;
  const uint64_t sizeSub = bdg->sizeSub;

  /* Free any previous results (allows re-solve without reset) */
  safe_free((void **)&bdg->eigvals);
  safe_free((void **)&bdg->modes_u);
  safe_free((void **)&bdg->modes_v);

  /* 1. Allocate ilobpcg state */
  z_lobpcg_t *alg = z_ilobpcg_alloc(n, nev, sizeSub);

  /* 2. Create linop context — stack-allocated, must outlive A/B/T linops.
   *    (See bdg_solve_d comment for lifetime rationale.) */
  linop_ctx_t linop_ctx = { .data = ctx, .data_size = sizeof(matmul_ctx_t) };

  /* 3. Create linear operators */
  LinearOperator_z_t *A = linop_create_z(n, n, matvec_lrep_z, NULL, &linop_ctx);
  LinearOperator_z_t *B = linop_create_z(n, n, matvec_swap_z, NULL, &linop_ctx);
  LinearOperator_z_t *T = linop_create_z(n, n, matvec_precond_z, NULL, &linop_ctx);

  /* 4. Set solver parameters */
  alg->A = A;
  alg->B = B;
  alg->T = T;
  alg->maxIter = bdg->maxIter;
  alg->tol = bdg->tol;

  /* 5. Initialize search vectors */
  switch (bdg->init_mode) {
  case BDG_INIT_REUSE:
    if (NULL != bdg->reuse_buf && bdg->reuse_n == n && bdg->reuse_cols == sizeSub) {
      memcpy(alg->S, bdg->reuse_buf, n * sizeSub * sizeof(c64));
      safe_free((void **)&bdg->reuse_buf);
      bdg->init_mode = BDG_INIT_WF_WEIGHTED;
      break;
    }
    /* fallthrough */
  case BDG_INIT_WF_WEIGHTED: {
    const c64 *wf = (const c64 *)ctx->wf;
    uint32_t seed = 42;
    for (uint64_t j = 0; j < sizeSub; j++) {
      for (uint64_t i = 0; i < size; i++) {
        f64 u1 = xrand(&seed);
        f64 u2 = xrand(&seed);
        u1 = fmax(u1, 1e-10);
        const c64 val = (fabs(sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2))
                      * ((NULL != wf) ? cabs(wf[i]) : 1.0)) + 0.0 * I;
        alg->S[i        + j * n] = val;
        alg->S[i + size + j * n] = val;
      }
    }
    break;
  }
  case BDG_INIT_CUSTOM:
    if (NULL != bdg->custom_init_fn)
      bdg->custom_init_fn(bdg, alg->S, n, sizeSub, bdg->custom_init_param);
    break;
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

  /* 6. Solve */
  z_ilobpcg(alg);

  /* 7. Extract results */
  bdg->converged = alg->converged;

  bdg->eigvals = xcalloc(nev, sizeof(f64));
  memcpy(bdg->eigvals, alg->eigVals, nev * sizeof(f64));

  c64 *modes_u = xcalloc(size * nev, sizeof(c64));
  c64 *modes_v = xcalloc(size * nev, sizeof(c64));
  for (uint64_t j = 0; j < nev; j++) {
    memcpy(&modes_u[j * size], &alg->S[j * n], size * sizeof(c64));
    memcpy(&modes_v[j * size], &alg->S[j * n + size], size * sizeof(c64));
  }
  bdg->modes_u = modes_u;
  bdg->modes_v = modes_v;

  /* 8. Cleanup */
  linop_destroy_z(&A);
  linop_destroy_z(&B);
  linop_destroy_z(&T);
  z_lobpcg_free(&alg);

  return 0;
}
