/**
 * @file matmulM_impl.inc
 * @brief Type-generic M operator: y = kinetic(x) + localTermM * x + 2*dipolar_conv(x)
 *
 * Dipolar convolution is optional (only if ctx->dipolar && ctx->dim == 3).
 *
 * Reference: ~/LREP_post/src/matmulM.c (drop in-place logic)
 */

#include "bdg_internal.h"
#include <omp.h>

void FN(matmulM)(matmul_ctx_t *ctx, const CTYPE *x, CTYPE *y) {
  const uint64_t size = ctx->size;
  const f64 *ltM  = ctx->localTermM;

  FN(kinetic)(ctx, x, y);

#pragma omp parallel for simd schedule(static)
  for (uint64_t i = 0; i < size; i++)
    y[i] += ltM[i] * x[i];

  if (ctx->dipolar && 3 == ctx->dim) {
    CTYPE *dc = (CTYPE *)ctx->c_wrk2;
    FN(dipolar_conv)(ctx, x, dc);

#pragma omp parallel for simd schedule(static)
    for (uint64_t i = 0; i < size; i++)
      y[i] += 2.0 * dc[i];
  }
}
