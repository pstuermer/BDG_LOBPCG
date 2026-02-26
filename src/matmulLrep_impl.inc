/**
 * @file matmulLrep_impl.inc
 * @brief Type-generic Lrep operator: block [K; M] on stacked [u; v].
 *
 * y[0:size]       = K * x[0:size]
 * y[size:2*size]  = M * x[size:2*size]
 *
 * Reference: ~/LREP_post/src/matmulLREP.c
 */

#include "bdg_internal.h"

void FN(matmulLrep)(matmul_ctx_t *ctx, const CTYPE *x, CTYPE *y) {
    /* TODO: Implement
     *
     * matmulK(ctx, x, y);                            // top half
     * matmulM(ctx, x + ctx->size, y + ctx->size);    // bottom half
     */
    (void)ctx; (void)x; (void)y;
}
