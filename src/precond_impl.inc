/**
 * @file precond_impl.inc
 * @brief Type-generic product preconditioner for BdG.
 *
 * T_K: y = P_v^{-1/2} * IFFT( 1/(mu + k^2/2) * FFT( P_v^{-1/2} * x ) )
 * where P_v^{-1/2} = precond_sqrtK = 1/sqrt(localTermK + mu)
 *
 * T_Lrep = block [T_K; T_M] on stacked [u; v].
 *
 * DRY note: precondK and precondM are 95% identical (differ only in sqrt array
 * pointer). Could refactor into a shared static precond_apply(ctx, p_sqrt, x, y)
 * helper if maintenance becomes a concern.
 *
 * Reference: ~/LREP_post/src/diagPrecondK.c, diagPrecondM.c, diagPrecond.c
 */

#include "fft_helpers_impl.inc"
#include <omp.h>

void FN(precondK)(matmul_ctx_t *ctx, const CTYPE *x, CTYPE *y) {
    const uint64_t size = ctx->size;
    const uint64_t k_size = ctx->k_size;
    const f64 mu = ctx->mu;
    const f64 *k2 = ctx->k2;
    const f64 *p_sqrt = ctx->precond_sqrtK;
    c64 *fk = ctx->f_wrk;
    CTYPE *buf = (CTYPE *)ctx->c_wrk1;

    /* 1. position-space scale: buf = p_sqrt * x */
#pragma omp parallel for simd schedule(static)
    for (uint64_t i = 0; i < size; i++)
        buf[i] = p_sqrt[i] * x[i];

    /* 2. FFT: c_wrk1 → f_wrk */
    FN(fft_forward_inplace)(ctx);

    /* 3. k-space filter with FFT normalization */
    const f64 inv_size = 1.0 / (f64)size;
#pragma omp parallel for simd schedule(static)
    for (uint64_t i = 0; i < k_size; i++)
        fk[i] *= inv_size / safe_val(mu + 0.5 * k2[i]);

    /* 4. IFFT: f_wrk → y */
    FN(fft_backward)(ctx, y);

    /* 5. position-space scale: y *= p_sqrt */
#pragma omp parallel for simd schedule(static)
    for (uint64_t i = 0; i < size; i++)
        y[i] *= p_sqrt[i];
}

void FN(precondM)(matmul_ctx_t *ctx, const CTYPE *x, CTYPE *y) {
    const uint64_t size = ctx->size;
    const uint64_t k_size = ctx->k_size;
    const f64 mu = ctx->mu;
    const f64 *k2 = ctx->k2;
    const f64 *p_sqrt = ctx->precond_sqrtM;
    c64 *fk = ctx->f_wrk;
    CTYPE *buf = (CTYPE *)ctx->c_wrk1;

    /* 1. position-space scale */
#pragma omp parallel for simd schedule(static)
    for (uint64_t i = 0; i < size; i++)
        buf[i] = p_sqrt[i] * x[i];

    /* 2. FFT */
    FN(fft_forward_inplace)(ctx);

    /* 3. k-space filter */
    const f64 inv_size = 1.0 / (f64)size;
#pragma omp parallel for simd schedule(static)
    for (uint64_t i = 0; i < k_size; i++)
        fk[i] *= inv_size / safe_val(mu + 0.5 * k2[i]);

    /* 4. IFFT */
    FN(fft_backward)(ctx, y);

    /* 5. position-space scale */
#pragma omp parallel for simd schedule(static)
    for (uint64_t i = 0; i < size; i++)
        y[i] *= p_sqrt[i];
}

void FN(precondLrep)(matmul_ctx_t *ctx, const CTYPE *x, CTYPE *y) {
    const uint64_t size = ctx->size;

    FN(precondK)(ctx, x, y);
    FN(precondM)(ctx, &x[size], &y[size]);
}
