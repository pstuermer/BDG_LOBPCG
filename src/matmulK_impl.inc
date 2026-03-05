/**
 * @file matmulK_impl.inc
 * @brief Type-generic K operator: y = kinetic(x) + localTermK * x
 *
 * Reference: ~/LREP_post/src/matmulK.c (drop in-place logic)
 */

#include "bdg_internal.h"
#include "profile.h"

void FN(matmulK)(matmul_ctx_t *ctx, const CTYPE *x, CTYPE *y) {
    const uint64_t size = ctx->size;
    const f64 *ltK  = ctx->localTermK;
    BDG_TIC(t0);

    BDG_TIC(t_kin);
    FN(kinetic)(ctx, x, y);
    BDG_TOC(t_kin, t_matmulK_kin);

    BDG_TIC(t_elem);
#pragma omp parallel for simd schedule(static)
    for (uint64_t i = 0; i < size; i++)
        y[i] += ltK[i] * x[i];
    BDG_TOC(t_elem, t_matmulK_elem);

    BDG_TOC(t0, t_matmulK);
    BDG_INC(n_matmulK);
}
