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
  const uint64_t size = ctx->size;

  FN(matmulK)(ctx, x, y);
  FN(matmulM)(ctx, &x[size], &y[size]);
}
