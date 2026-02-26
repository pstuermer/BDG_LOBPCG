/**
 * @file precond_impl.inc
 * @brief Type-generic product preconditioner for BdG.
 *
 * T_K: y = P_v^{1/2} * IFFT( 1/(mu + k^2/2) * FFT( P_v^{1/2} * x ) )
 * where P_v^{1/2} = precond_sqrtK = 1/sqrt(localTermK + mu)
 *
 * T_Lrep = block [T_K; T_M] on stacked [u; v].
 *
 * Reference: ~/LREP_post/src/diagPrecondK.c, diagPrecondM.c, diagPrecond.c
 */

#include "bdg_internal.h"

void FN(precondK)(matmul_ctx_t *ctx, const CTYPE *x, CTYPE *y) {
    /* TODO: Implement
     *
     * 1. for i: tmp[i] = precond_sqrtK[i] * x[i]    // position-space
     * 2. FFT(tmp) → f_wrk
     * 3. for i: f_wrk[i] *= 1.0 / (ctx->mu + 0.5 * k2[i]) / size  // k-space filter
     * 4. IFFT(f_wrk) → tmp
     * 5. for i: y[i] = precond_sqrtK[i] * tmp[i]    // position-space
     */
    (void)ctx; (void)x; (void)y;
}

void FN(precondM)(matmul_ctx_t *ctx, const CTYPE *x, CTYPE *y) {
    /* TODO: Same as precondK but uses precond_sqrtM */
    (void)ctx; (void)x; (void)y;
}

void FN(precondLrep)(matmul_ctx_t *ctx, const CTYPE *x, CTYPE *y) {
    /* TODO: Implement
     *
     * precondK(ctx, x, y);                           // top half
     * precondM(ctx, x + ctx->size, y + ctx->size);   // bottom half
     */
    (void)ctx; (void)x; (void)y;
}
