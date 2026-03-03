#include "bdg_internal.h"
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @file bdg.c
 * @brief Top-level BdG lifecycle: alloc, free, reset, solve dispatch.
 */

/* ----------------------------------------------------------------
 * bdg_alloc
 * ---------------------------------------------------------------- */
bdg_t *bdg_alloc(size_t dim, const size_t *N, const f64 *L, int complex_psi0) {
    bdg_t *bdg = xcalloc(1, sizeof(bdg_t));
    bdg->complex_psi0 = complex_psi0;
    bdg->ctx = matmul_ctx_alloc(dim, N, L);
    bdg->ctx->complex_psi0 = complex_psi0;

    /* Defaults */
    bdg->nev     = 3;
    bdg->sizeSub = 6;
    bdg->maxIter = 500;
    bdg->tol     = 1e-6;

    return bdg;
}

/* ----------------------------------------------------------------
 * bdg_free
 * ---------------------------------------------------------------- */
void bdg_free(bdg_t **bdg) {
    if (!bdg || !*bdg) return;
    bdg_t *b = *bdg;

    /* Results */
    if (b->eigvals)  safe_free((void **)&b->eigvals);
    if (b->modes_u)  safe_free((void **)&b->modes_u);
    if (b->modes_v)  safe_free((void **)&b->modes_v);
    if (b->reuse_buf) safe_free((void **)&b->reuse_buf);

    /* Context */
    matmul_ctx_free(&b->ctx);

    safe_free((void **)bdg);
}

/* ----------------------------------------------------------------
 * bdg_reset — clear physics state, preserve grid/plans/scratch
 * ---------------------------------------------------------------- */
void bdg_reset(bdg_t *bdg) {
    BDG_REQUIRE(bdg, BDG_HAS_SYSTEM, "bdg_reset");

    matmul_ctx_t *ctx = bdg->ctx;

    /* Free physics arrays */
    safe_free((void **)&ctx->wf);
    safe_free((void **)&ctx->longRngInt);
    safe_free((void **)&ctx->precond_sqrtK);
    safe_free((void **)&ctx->precond_sqrtM);

    /* Zero local terms (allocated at alloc time, never freed) */
    memset(ctx->localTermK, 0, ctx->size * sizeof(f64));
    memset(ctx->localTermM, 0, ctx->size * sizeof(f64));

    /* Zero scalars */
    ctx->mu      = 0.0;
    ctx->g_ddi   = 0.0;
    ctx->dipolar = 0;
    ctx->wf_size = 0;

    /* Free results */
    safe_free((void **)&bdg->eigvals);
    safe_free((void **)&bdg->modes_u);
    safe_free((void **)&bdg->modes_v);
    bdg->converged = 0;

    /* Keep only BDG_HAS_SYSTEM */
    bdg->state = BDG_HAS_SYSTEM;
}

/* ----------------------------------------------------------------
 * Setup pass-throughs
 * ---------------------------------------------------------------- */
void bdg_set_system(bdg_t *bdg) {
    matmul_ctx_set_system(bdg->ctx, bdg->complex_psi0);
    bdg->state |= BDG_HAS_SYSTEM;
}

void bdg_set_solver_params(bdg_t *bdg, size_t nev, size_t sizeSub,
                           size_t maxIter, f64 tol) {
    bdg->nev     = nev;
    bdg->sizeSub = sizeSub;
    bdg->maxIter = maxIter;
    bdg->tol     = tol;
}

/* ----------------------------------------------------------------
 * Init strategy
 * ---------------------------------------------------------------- */
void bdg_set_init_mode(bdg_t *bdg, bdg_init_mode_t mode,
                       bdg_init_fn fn, void *param) {
  bdg->init_mode = mode;
  bdg->custom_init_fn = fn;
  bdg->custom_init_param = param;
}

int bdg_reuse_modes(bdg_t *bdg, f64 noise_frac) {
  if (NULL == bdg->modes_u || NULL == bdg->modes_v)
    return -3;

  const uint64_t size = bdg->ctx->size;
  const uint64_t n = 2 * size;
  const uint64_t nev = bdg->nev;
  const uint64_t sizeSub = bdg->sizeSub;
  const size_t elem = bdg->complex_psi0 ? sizeof(c64) : sizeof(f64);

  safe_free((void **)&bdg->reuse_buf);
  bdg->reuse_buf = xcalloc(n * sizeSub, elem);
  bdg->reuse_n = n;
  bdg->reuse_cols = sizeSub;

  uint32_t seed = 12345;

  if (0 == bdg->complex_psi0) {
    f64 *buf = (f64 *)bdg->reuse_buf;
    const f64 *u = (const f64 *)bdg->modes_u;
    const f64 *v = (const f64 *)bdg->modes_v;

    for (uint64_t j = 0; j < nev && j < sizeSub; j++) {
      f64 col_norm = 0.0;
      for (uint64_t i = 0; i < size; i++) {
        col_norm += u[j * size + i] * u[j * size + i];
        col_norm += v[j * size + i] * v[j * size + i];
      }
      col_norm = sqrt(col_norm);
      const f64 sigma = noise_frac * col_norm;

      for (uint64_t i = 0; i < size; i++) {
        f64 u1 = xrand(&seed);
        f64 u2 = xrand(&seed);
        u1 = fmax(u1, 1e-10);
        const f64 z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        const f64 z1 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);
        buf[i        + j * n] = u[j * size + i] + sigma * z0;
        buf[i + size + j * n] = v[j * size + i] + sigma * z1;
      }
    }

    const f64 *wf = (const f64 *)bdg->ctx->wf;
    for (uint64_t j = nev; j < sizeSub; j++) {
      for (uint64_t i = 0; i < size; i++) {
        f64 u1 = xrand(&seed);
        f64 u2 = xrand(&seed);
        u1 = fmax(u1, 1e-10);
        const f64 val = fabs(sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2))
                      * ((NULL != wf) ? fabs(wf[i]) : 1.0);
        buf[i        + j * n] = val;
        buf[i + size + j * n] = val;
      }
    }
  } else {
    c64 *buf = (c64 *)bdg->reuse_buf;
    const c64 *u = (const c64 *)bdg->modes_u;
    const c64 *v = (const c64 *)bdg->modes_v;

    for (uint64_t j = 0; j < nev && j < sizeSub; j++) {
      f64 col_norm = 0.0;
      for (uint64_t i = 0; i < size; i++) {
        col_norm += cabs(u[j * size + i]) * cabs(u[j * size + i]);
        col_norm += cabs(v[j * size + i]) * cabs(v[j * size + i]);
      }
      col_norm = sqrt(col_norm);
      const f64 sigma = noise_frac * col_norm;

      for (uint64_t i = 0; i < size; i++) {
        f64 u1 = xrand(&seed);
        f64 u2 = xrand(&seed);
        u1 = fmax(u1, 1e-10);
        const f64 z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        const f64 z1 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);
        buf[i        + j * n] = u[j * size + i] + sigma * z0;
        buf[i + size + j * n] = v[j * size + i] + sigma * z1;
      }
    }

    const c64 *wf = (const c64 *)bdg->ctx->wf;
    for (uint64_t j = nev; j < sizeSub; j++) {
      for (uint64_t i = 0; i < size; i++) {
        f64 u1 = xrand(&seed);
        f64 u2 = xrand(&seed);
        u1 = fmax(u1, 1e-10);
        const f64 val = fabs(sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2))
                      * ((NULL != wf) ? cabs(wf[i]) : 1.0);
        buf[i        + j * n] = val + 0.0 * I;
        buf[i + size + j * n] = val + 0.0 * I;
      }
    }
  }

  bdg->init_mode = BDG_INIT_REUSE;
  return 0;
}

/* ----------------------------------------------------------------
 * bdg_solve — dispatches to d or z path
 * ---------------------------------------------------------------- */
int bdg_solve(bdg_t *bdg) {
    BDG_REQUIRE(bdg, BDG_HAS_SYSTEM, "bdg_solve");
    BDG_REQUIRE(bdg, BDG_HAS_MU, "bdg_solve");
    if (bdg->complex_psi0)
        return bdg_solve_z(bdg);
    else
        return bdg_solve_d(bdg);
}

/* ----------------------------------------------------------------
 * Result accessors
 * ---------------------------------------------------------------- */
size_t bdg_converged(const bdg_t *bdg) {
    return bdg->converged;
}

const f64 *bdg_eigenvalues(const bdg_t *bdg) {
    return bdg->eigvals;
}

const void *bdg_modes_u(const bdg_t *bdg) {
    return bdg->modes_u;
}

const void *bdg_modes_v(const bdg_t *bdg) {
    return bdg->modes_v;
}
