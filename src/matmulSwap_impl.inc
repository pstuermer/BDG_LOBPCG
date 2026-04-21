/**
 * @file matmulSwap_impl.inc
 * @brief Type-generic Swap operator: y = [x_lower; x_upper].
 *
 * Two copy calls. B operator for the indefinite LOBPCG.
 *
 * Reference: ~/LREP_post/src/matmulSwap.c:5-37
 */

#include "bdg_internal.h"
#include "lobpcg/blas_wrapper.h"
#include "profile.h"

void FN(matmulSwap)(matmul_ctx_t *ctx, const CTYPE *x, CTYPE *y) {
  const uint64_t size = ctx->size;
  BDG_TIC(t0);

  FN(copy)(size, x + size, y);
  FN(copy)(size, x, y + size);

  BDG_TOC(t0, t_swap);
  BDG_INC(n_swap);
}
