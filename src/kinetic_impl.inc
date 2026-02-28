/**
 * @file kinetic_impl.inc
 * @brief Type-generic kinetic energy operator: y = 0.5 * k^2 * x via FFT.
 *
 * Expects macros: CTYPE (f64 or c64), RTYPE (f64), FN(name) (suffix helper),
 * and either CTYPE_IS_REAL or CTYPE_IS_COMPLEX.
 *
 * Real path uses r2c/c2r (k_size < size).
 * Complex path uses c2c (k_size == size).
 *
 * Reference: ~/LREP_post/src/kinetic.c:4-41
 */

#include "fft_helpers_impl.inc"
#include <omp.h>

void FN(kinetic)(matmul_ctx_t *ctx, const CTYPE *x, CTYPE *y) {
    const size_t  k_size = ctx->k_size;
    const f64     norm   = 0.5 / (f64)ctx->size;
    const f64    *k2     = ctx->k2;
    c64          *fk     = ctx->f_wrk;

    FN(fft_forward)(ctx, x);

#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < k_size; i++)
        fk[i] *= k2[i] * norm;

    FN(fft_backward)(ctx, y);
}
