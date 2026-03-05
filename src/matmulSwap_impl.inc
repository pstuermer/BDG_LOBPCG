/**
 * @file matmulSwap_impl.inc
 * @brief Type-generic Swap operator: y = [x_lower; x_upper].
 *
 * Two memcpy calls. B operator for the indefinite LOBPCG.
 *
 * Reference: ~/LREP_post/src/matmulSwap.c:5-37
 */

#include "bdg_internal.h"
#include "profile.h"
#include <omp.h>

void FN(matmulSwap)(matmul_ctx_t *ctx, const CTYPE *x, CTYPE *y) {
  const uint64_t size = ctx->size;
  BDG_TIC(t0);

#pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < size; i++) {
    y[i] = x[size + i];
    y[size + i] = x[i];
  }

  BDG_TOC(t0, t_swap);
  BDG_INC(n_swap);
}
