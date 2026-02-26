/**
 * @file matmulK_impl.inc
 * @brief Type-generic K operator: y = kinetic(x) + localTermK * x
 *
 * Reference: ~/LREP_post/src/matmulK.c (drop in-place logic)
 */

#include "bdg_internal.h"

void FN(matmulK)(matmul_ctx_t *ctx, const CTYPE *x, CTYPE *y) {
    /* TODO: Implement
     *
     * 1. kinetic(ctx, x, y)          — y = 0.5*k^2*x via FFT
     * 2. for i: y[i] += localTermK[i] * x[i]  — pointwise multiply-add
     */
    (void)ctx; (void)x; (void)y;
}
