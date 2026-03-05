/**
 * @file matmulM_impl.inc
 * @brief Type-generic M operator: y = kinetic(x) + localTermM * x + 2*dipolar_conv(x)
 *
 * Dipolar convolution is optional (only if ctx->dipolar && ctx->dim == 3).
 *
 * Reference: ~/LREP_post/src/matmulM.c (drop in-place logic)
 */

#include "bdg_internal.h"
#include "profile.h"
#include <omp.h>

void FN(matmulM)(matmul_ctx_t *ctx, const CTYPE *x, CTYPE *y) {
  const uint64_t size = ctx->size;
  const f64 *ltM  = ctx->localTermM;
  BDG_TIC(t0);

  BDG_TIC(t_kin);
  FN(kinetic)(ctx, x, y);
  BDG_TOC(t_kin, t_matmulM_kin);

  if (ctx->dipolar && 3 == ctx->dim) {
    BDG_TIC(t_dip);
    CTYPE *dc = (CTYPE *)ctx->c_wrk2;
    FN(dipolar_conv)(ctx, x, dc);
    BDG_TOC(t_dip, t_matmulM_dip);

    BDG_TIC(t_elem);
#pragma omp parallel for simd schedule(static)
    for (uint64_t i = 0; i < size; i++)
      y[i] += ltM[i] * x[i] + 2.0 * dc[i];
    BDG_TOC(t_elem, t_matmulM_elem);
  } else {
    BDG_TIC(t_elem);
#pragma omp parallel for simd schedule(static)
    for (uint64_t i = 0; i < size; i++)
      y[i] += ltM[i] * x[i];
    BDG_TOC(t_elem, t_matmulM_elem);
  }

  BDG_TOC(t0, t_matmulM);
  BDG_INC(n_matmulM);
}
