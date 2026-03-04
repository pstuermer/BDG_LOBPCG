/**
 * @file dipolar_meanfield_impl.inc
 * @brief Type-generic dipolar mean-field: Φ_dd added to localTermK/M.
 *
 * Computes Φ_dd = IFFT(longRngInt * FFT(|wf|²)) and adds g_ddi * Φ_dd
 * to both localTermK and localTermM.
 */

#include "fft_helpers_impl.inc"
#include <omp.h>

#ifdef CTYPE_IS_COMPLEX
#define CABS(x) cabs(x)
#define CREAL(x) creal(x)
#else
#define CABS(x) fabs(x)
#define CREAL(x) (x)
#endif

void FN(dipolar_add_meanfield)(matmul_ctx_t *ctx) {
    const uint64_t size   = ctx->size;
    const uint64_t k_size = ctx->k_size;
    const f64    norm   = 1.0 / (f64)size;
    const f64    g      = ctx->g_ddi;
    const CTYPE *wf     = (const CTYPE *)ctx->wf;
    const f64   *kernel = ctx->longRngInt;
    CTYPE       *buf    = (CTYPE *)ctx->c_wrk1;
    c64         *fk     = ctx->f_wrk;

    /* 1. density |wf|² → c_wrk1 */
#pragma omp parallel for simd schedule(static)
    for (uint64_t i = 0; i < size; i++) {
        const RTYPE d = CABS(wf[i]);
        buf[i] = (CTYPE)(d * d);
    }

    /* 2. FFT → f_wrk */
    FN(fft_forward_inplace)(ctx);

    /* 3. multiply by kernel / size */
#pragma omp parallel for simd schedule(static)
    for (uint64_t i = 0; i < k_size; i++)
        fk[i] *= kernel[i] * norm;

    /* 4. IFFT → c_wrk1 */
    FN(fft_backward_inplace)(ctx);

    /* 5. accumulate g_ddi * Φ_dd into local terms */
#pragma omp parallel for simd schedule(static)
    for (uint64_t i = 0; i < size; i++) {
        const f64 phi = g * CREAL(buf[i]);
        ctx->localTermK[i] += phi;
        ctx->localTermM[i] += phi;
    }
}

#undef CABS
#undef CREAL
