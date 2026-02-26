/**
 * @file matmulSwap_impl.inc
 * @brief Type-generic Swap operator: y = [x_lower; x_upper].
 *
 * Two memcpy calls. B operator for the indefinite LOBPCG.
 *
 * Reference: ~/LREP_post/src/matmulSwap.c:5-37
 */

#include "bdg_internal.h"
#include <string.h>

void FN(matmulSwap)(matmul_ctx_t *ctx, const CTYPE *x, CTYPE *y) {
    /* TODO: Implement
     *
     * memcpy(y,             x + ctx->size, ctx->size * sizeof(CTYPE));
     * memcpy(y + ctx->size, x,             ctx->size * sizeof(CTYPE));
     */
    (void)ctx; (void)x; (void)y;
}
