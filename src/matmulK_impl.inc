/**
 * @file matmulK_impl.inc
 * @brief Type-generic K operator: y = kinetic(x) + localTermK * x
 *
 * Reference: ~/LREP_post/src/matmulK.c (drop in-place logic)
 */

#include "bdg_internal.h"

void FN(matmulK)(matmul_ctx_t *ctx, const CTYPE *x, CTYPE *y) {
    const size_t size = ctx->size;
    const f64   *ltK  = ctx->localTermK;

    FN(kinetic)(ctx, x, y);

#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < size; i++)
        y[i] += ltK[i] * x[i];
}
