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

#include "fft_helpers_impl.inc"
#include <omp.h>

#ifdef CTYPE_IS_COMPLEX
#define CONJ(x) conj(x)
#else
#define CONJ(x) (x)
#endif

void FN(dipolar_conv)(matmul_ctx_t *ctx, const CTYPE *v, CTYPE *out) {
    const uint64_t size   = ctx->size;
    const uint64_t k_size = ctx->k_size;
    const f64    norm   = 1.0 / (f64)size;
    const f64    g      = ctx->g_ddi;
    const CTYPE *wf     = (const CTYPE *)ctx->wf;
    const f64   *kernel = ctx->longRngInt;
    CTYPE       *buf    = (CTYPE *)ctx->c_wrk1;
    c64         *fk     = ctx->f_wrk;

    /* 1. c_wrk1 = conj(wf) * v */
#pragma omp parallel for simd schedule(static)
    for (uint64_t i = 0; i < size; i++)
        buf[i] = CONJ(wf[i]) * v[i];

    /* 2. FFT → f_wrk */
    FN(fft_forward_inplace)(ctx);

    /* 3. scale by kernel / size */
#pragma omp parallel for simd schedule(static)
    for (uint64_t i = 0; i < k_size; i++)
        fk[i] *= kernel[i] * norm;

    /* 4. IFFT → c_wrk1 */
    FN(fft_backward_inplace)(ctx);

    /* 5. out = g_ddi * wf * c_wrk1 */
#pragma omp parallel for simd schedule(static)
    for (uint64_t i = 0; i < size; i++)
        out[i] = g * wf[i] * buf[i];
}

#undef CONJ
