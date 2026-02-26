/**
 * @file matmulM_impl.inc
 * @brief Type-generic M operator: y = kinetic(x) + localTermM * x + 2*dipolar_conv(x)
 *
 * Dipolar convolution is optional (only if ctx->dipolar && ctx->dim == 3).
 *
 * Reference: ~/LREP_post/src/matmulM.c (drop in-place logic)
 */

#include "bdg_internal.h"

void FN(matmulM)(matmul_ctx_t *ctx, const CTYPE *x, CTYPE *y) {
    /* TODO: Implement
     *
     * 1. kinetic(ctx, x, y)
     * 2. for i: y[i] += localTermM[i] * x[i]
     * 3. if (ctx->dipolar && ctx->dim == 3):
     *        dipolar_conv(ctx, x, c_wrk2)   // perturbation convolution
     *        for i: y[i] += 2.0 * c_wrk2[i]
     */
    (void)ctx; (void)x; (void)y;
}
