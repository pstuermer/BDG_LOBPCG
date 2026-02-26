#include "bdg_internal.h"
#include <math.h>

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
    /* TODO: Implement
     *
     * 1. Compute k_size:
     *    - complex: k_size = size
     *    - real (r2c): k_size = prod(N[1..dim-1]) * (N[0]/2 + 1)
     *
     * 2. Allocate k2[k_size], kx2[dim][k_size]
     *
     * 3. Fill k-frequencies with reversed FFTW dims: fftw_N[i] = N[dim-1-i]
     *    dk[d] = 2*pi / L[d]
     *    k_j = j < N[d]/2 ? j*dk : (j - N[d])*dk
     *    For r2c: last FFTW dim (= N[0]) index runs 0..N[0]/2
     *
     * 4. Create FFTW plans:
     *    - Real: fftw_plan_dft_r2c_{1d,2d,3d} / fftw_plan_dft_c2r_{1d,2d,3d}
     *    - Complex: fftw_plan_dft_{1d,2d,3d} with FFTW_FORWARD / FFTW_BACKWARD
     *
     * 5. Allocate scratch: f_wrk (c64, k_size), c_wrk1/2
     *
     * Reference: ~/LREP_post/src/matmul.c:92-209
     */
    (void)ctx; (void)complex_psi0;
}
