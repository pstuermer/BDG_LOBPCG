#include "bdg_internal.h"
#include <math.h>
#include <assert.h>
#include <omp.h>

/**
 * @file matmul_ctx.c
 * @brief matmul_ctx_t allocation/free and set_system (k2, kx2, FFTW plans).
 *
 * Port from ~/LREP_post/src/matmul.c:64-209.
 * Removed mode_t wrapper. Added r2c/c2r path for real psi0.
 */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ----------------------------------------------------------------
 * matmul_ctx_alloc
 * ---------------------------------------------------------------- */
matmul_ctx_t *matmul_ctx_alloc(size_t dim, const size_t *N, const f64 *L) {
  matmul_ctx_t *ctx = xcalloc(1, sizeof(matmul_ctx_t));
  ctx->dim = dim;

  ctx->N = xcalloc(dim, sizeof(size_t));
  ctx->L = xcalloc(dim, sizeof(f64));
  ctx->size = 1;
  for (size_t d = 0; d < dim; d++) {
    ctx->N[d] = N[d];
    ctx->L[d] = L[d];
    ctx->size *= N[d];
  }

  /* k_size set later in set_system (depends on complex_psi0) */
  ctx->k_size = 0;

  /* Allocate physics arrays at full real-space size */
  ctx->localTermK = xcalloc(ctx->size, sizeof(f64));
  ctx->localTermM = xcalloc(ctx->size, sizeof(f64));

  return ctx;
}

/* ----------------------------------------------------------------
 * matmul_ctx_free
 * ---------------------------------------------------------------- */
void matmul_ctx_free(matmul_ctx_t **pctx) {
  if (!pctx || !*pctx) return;
  matmul_ctx_t *ctx = *pctx;

  /* FFTW plans */
  if (ctx->fwd_plan) fftw_destroy_plan(ctx->fwd_plan);
  if (ctx->bwd_plan) fftw_destroy_plan(ctx->bwd_plan);

  /* k-space arrays */
  if (ctx->kx2) {
    for (size_t d = 0; d < ctx->dim; d++)
      safe_free((void **)&ctx->kx2[d]);
    safe_free((void **)&ctx->kx2);
  }
  safe_free((void **)&ctx->k2);

  /* Physics */
  safe_free((void **)&ctx->localTermK);
  safe_free((void **)&ctx->localTermM);
  safe_free((void **)&ctx->longRngInt);
  safe_free((void **)&ctx->wf);

  /* Preconditioner aux */
  safe_free((void **)&ctx->precond_sqrtK);
  safe_free((void **)&ctx->precond_sqrtM);

  /* Scratch */
  safe_free((void **)&ctx->f_wrk);
  safe_free((void **)&ctx->c_wrk1);
  safe_free((void **)&ctx->c_wrk2);

  /* Grid arrays */
  safe_free((void **)&ctx->N);
  safe_free((void **)&ctx->L);

  safe_free((void **)pctx);
}

/* ----------------------------------------------------------------
 * matmul_ctx_set_system — k2, kx2, FFTW plans, scratch buffers
 * ---------------------------------------------------------------- */
void matmul_ctx_set_system(matmul_ctx_t *ctx, int complex_psi0) {
  /* Reference: ~/LREP_post/src/matmul.c:92-209 */

  ctx->complex_psi0 = complex_psi0;

  /* --- Step 1: k_size ------------------------------------------------ */
  if (complex_psi0) {
    ctx->k_size = ctx->size;
  } else {
    ctx->k_size = ctx->N[0] / 2 + 1;
    for (size_t d = 1; d < ctx->dim; d++)
      ctx->k_size *= ctx->N[d];
  }

  /* --- Step 2: kx2[d][N[d]] (compact per-dimension) ----------------- */
  ctx->kx2 = xcalloc(ctx->dim, sizeof(f64 *));
  for (size_t d = 0; d < ctx->dim; d++) {
    ctx->kx2[d] = xcalloc(ctx->N[d], sizeof(f64));
    const f64 dk = 2.0 * M_PI / ctx->L[d];
    for (size_t j = 0; j < ctx->N[d]; j++) {
      const f64 kj = (j <= ctx->N[d] / 2)
	? (dk * (f64)j)
	: (dk * (f64)((int)j - (int)ctx->N[d]));
      ctx->kx2[d][j] = kj * kj;
    }
  }

  /* --- Step 3: k2[k_size] (flat, precomputed) ----------------------- */
  const size_t N0_k = complex_psi0 ? ctx->N[0] : (ctx->N[0] / 2 + 1);
  ctx->k2 = xcalloc(ctx->k_size, sizeof(f64));

  if (1 == ctx->dim) {
    for (size_t ix = 0; ix < N0_k; ix++)
      ctx->k2[ix] = ctx->kx2[0][ix];
  } else if (2 == ctx->dim) {
    for (size_t iy = 0; iy < ctx->N[1]; iy++)
      for (size_t ix = 0; ix < N0_k; ix++)
	ctx->k2[iy * N0_k + ix] = ctx->kx2[0][ix] + ctx->kx2[1][iy];
  } else {
    for (size_t iz = 0; iz < ctx->N[2]; iz++)
      for (size_t iy = 0; iy < ctx->N[1]; iy++)
	for (size_t ix = 0; ix < N0_k; ix++)
	  ctx->k2[iz * ctx->N[1] * N0_k + iy * N0_k + ix] =
	    ctx->kx2[0][ix] + ctx->kx2[1][iy] + ctx->kx2[2][iz];
  }

  /* --- Step 4: FFTW thread init + nthreads -------------------------- */
  static int fftw_threads_initialized = 0;
  if (0 == fftw_threads_initialized) {
    fftw_init_threads();
    fftw_threads_initialized = 1;
  }
  fftw_plan_with_nthreads(omp_get_max_threads());

  /* --- Step 5: Scratch buffers (must precede plan creation) ---------- */
  const size_t wrk_size = (ctx->size > ctx->k_size) ? ctx->size : ctx->k_size;
  ctx->f_wrk  = xcalloc(ctx->k_size, sizeof(c64));
  ctx->c_wrk1 = xcalloc(wrk_size, sizeof(c64));
  ctx->c_wrk2 = xcalloc(wrk_size, sizeof(c64));

  /* --- Step 6: FFTW plans ------------------------------------------- */
  int fftw_N[3];
  for (size_t d = 0; d < ctx->dim; d++)
    fftw_N[d] = (int)ctx->N[ctx->dim - 1 - d];

  if (complex_psi0) {
    /* c2c forward/backward */
    ctx->fwd_plan = fftw_plan_dft((int)ctx->dim, fftw_N,
				  (fftw_complex *)ctx->c_wrk1, (fftw_complex *)ctx->f_wrk,
				  FFTW_FORWARD, FFTW_MEASURE);
    ctx->bwd_plan = fftw_plan_dft((int)ctx->dim, fftw_N,
				  (fftw_complex *)ctx->f_wrk, (fftw_complex *)ctx->c_wrk1,
				  FFTW_BACKWARD, FFTW_MEASURE);
  } else {
    /* r2c / c2r */
    ctx->fwd_plan = fftw_plan_dft_r2c((int)ctx->dim, fftw_N,
				      (f64 *)ctx->c_wrk1, (fftw_complex *)ctx->f_wrk,
				      FFTW_MEASURE);
    ctx->bwd_plan = fftw_plan_dft_c2r((int)ctx->dim, fftw_N,
				      (fftw_complex *)ctx->f_wrk, (f64 *)ctx->c_wrk1,
				      FFTW_MEASURE);
  }
  assert(NULL != ctx->fwd_plan);
  assert(NULL != ctx->bwd_plan);
}
