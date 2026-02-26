/**
 * @file dipolar_conv_impl.inc
 * @brief Type-generic perturbation-dependent dipolar convolution.
 *
 * Computes: out[i] = g_ddi * wf[i] * IFFT( longRngInt * FFT( conj(wf) * v ) )[i]
 *
 * For real wf, conj(wf) = wf, so the conj is a no-op.
 * For complex wf, conj matters.
 *
 * Reference: ~/LREP_post/src/dipolar.c:129-175
 */

#include "bdg_internal.h"

void FN(dipolar_conv)(matmul_ctx_t *ctx, const CTYPE *v, CTYPE *out) {
    /* TODO: Implement
     *
     * 1. c_wrk1[i] = conj(wf[i]) * v[i]   (conj is identity for f64)
     * 2. FFT(c_wrk1) → f_wrk
     * 3. f_wrk[i] *= longRngInt[i] / size
     * 4. IFFT(f_wrk) → c_wrk1
     * 5. out[i] = g_ddi * wf[i] * c_wrk1[i]
     */
    (void)ctx; (void)v; (void)out;
}
