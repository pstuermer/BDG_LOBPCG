#include "bdg_internal.h"
#include "lobpcg.h"

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
 * bdg_solve_d — real path
 * ================================================================ */
int bdg_solve_d(bdg_t *bdg) {
  matmul_ctx_t *ctx = bdg->ctx;
  const uint64_t size = ctx->size;
  const uint64_t n = 2 * size;
  const uint64_t nev = bdg->nev;
  const uint64_t sizeSub = bdg->sizeSub;

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

  /* 5. Initialize search vectors: planewave-seeded, B-positive.
   *
   * For periodic BdG problems the natural eigenmodes are planewaves.
   * Seed column j with cos(k*r)/sin(k*r) for k = 0, 1, 1, 2, 2, ...
   * weighted by |wf| and with small random perturbation.
   * Setting u-part = v-part guarantees x^T*B*x = 2*u^T*u > 0.
   */
  {
    const f64 *wf = (const f64 *)ctx->wf;
    unsigned int seed = 42;

    for (uint64_t j = 0; j < sizeSub; j++) {
      const uint64_t k_idx = (j + 1) / 2;  /* 0, 1, 1, 2, 2, 3, ... */
      const int use_sin = (j % 2 == 1);

      for (uint64_t i = 0; i < size; i++) {
        /* Compute planewave value along first dimension */
        f64 pw;
        if (0 == k_idx) {
          pw = 1.0;
        } else {
          const f64 xpos = (f64)(i % ctx->N[0]) * ctx->L[0] / (f64)ctx->N[0];
          const f64 kval = 2.0 * M_PI * (f64)k_idx / ctx->L[0];
          pw = use_sin ? sin(kval * xpos) : cos(kval * xpos);
        }

        /* Weight by |wf| and add small perturbation */
        const f64 pert = 1e-4 * ((f64)rand_r(&seed) / RAND_MAX - 0.5);
        const f64 wf_weight = (NULL != wf) ? fabs(wf[i]) : 1.0;
        const f64 val = (pw + pert) * wf_weight;

        alg->S[i + j * n] = val;           /* u-part */
        alg->S[i + size + j * n] = val;    /* v-part = u-part → B-positive */
      }
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

  /* 5. Initialize search vectors: planewave-seeded, B-positive.
   *    (See bdg_solve_d comment for rationale.) */
  {
    const c64 *wf = (const c64 *)ctx->wf;
    unsigned int seed = 42;

    for (uint64_t j = 0; j < sizeSub; j++) {
      const uint64_t k_idx = (j + 1) / 2;
      const int use_sin = (j % 2 == 1);

      for (uint64_t i = 0; i < size; i++) {
        f64 pw;
        if (0 == k_idx) {
          pw = 1.0;
        } else {
          const f64 xpos = (f64)(i % ctx->N[0]) * ctx->L[0] / (f64)ctx->N[0];
          const f64 kval = 2.0 * M_PI * (f64)k_idx / ctx->L[0];
          pw = use_sin ? sin(kval * xpos) : cos(kval * xpos);
        }

        const f64 pert = 1e-4 * ((f64)rand_r(&seed) / RAND_MAX - 0.5);
        const f64 wf_weight = (NULL != wf) ? cabs(wf[i]) : 1.0;
        const c64 val = ((pw + pert) * wf_weight) + 0.0 * I;

        alg->S[i + j * n] = val;           /* u-part */
        alg->S[i + size + j * n] = val;    /* v-part = u-part → B-positive */
      }
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
