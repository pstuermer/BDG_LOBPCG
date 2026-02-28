/**
 * @file fft_helpers_impl.inc
 * @brief Type-generic FFT helpers for the BdG operator pipeline.
 *
 * Provides static inline wrappers that hide the r2c/c2r vs c2c branching.
 * Include from any operator .inc file after CTYPE/FN/CTYPE_IS_* are defined.
 *
 * Functions provided:
 *   FN(fft_forward)          — copy x → c_wrk1, forward FFT → f_wrk
 *   FN(fft_forward_inplace)  — forward FFT c_wrk1 → f_wrk (no copy)
 *   FN(fft_backward)         — backward FFT f_wrk → y directly
 *   FN(fft_backward_inplace) — backward FFT f_wrk → c_wrk1
 *
 * Expects macros: CTYPE, FN(name), and CTYPE_IS_REAL or CTYPE_IS_COMPLEX.
 */

#include "bdg_internal.h"
#include <string.h>

/* ----------------------------------------------------------------
 * Forward FFT: c_wrk1 → f_wrk (caller already filled c_wrk1)
 * ---------------------------------------------------------------- */
static inline void FN(fft_forward_inplace)(matmul_ctx_t *ctx) {
#ifdef CTYPE_IS_REAL
    fftw_execute_dft_r2c(ctx->fwd_plan,
                         (f64 *)ctx->c_wrk1, (fftw_complex *)ctx->f_wrk);
#else
    fftw_execute_dft(ctx->fwd_plan,
                     (fftw_complex *)ctx->c_wrk1, (fftw_complex *)ctx->f_wrk);
#endif
}

/* ----------------------------------------------------------------
 * Forward FFT: copy x → c_wrk1, then FFT → f_wrk
 * ---------------------------------------------------------------- */
static inline void FN(fft_forward)(matmul_ctx_t *ctx, const CTYPE *x) {
    memcpy(ctx->c_wrk1, x, ctx->size * sizeof(CTYPE));
    FN(fft_forward_inplace)(ctx);
}

/* ----------------------------------------------------------------
 * Backward FFT: f_wrk → y directly (no intermediate copy)
 * ---------------------------------------------------------------- */
static inline void FN(fft_backward)(matmul_ctx_t *ctx, CTYPE *y) {
#ifdef CTYPE_IS_REAL
    fftw_execute_dft_c2r(ctx->bwd_plan,
                         (fftw_complex *)ctx->f_wrk, (f64 *)y);
#else
    fftw_execute_dft(ctx->bwd_plan,
                     (fftw_complex *)ctx->f_wrk, (fftw_complex *)y);
#endif
}

/* ----------------------------------------------------------------
 * Backward FFT: f_wrk → c_wrk1 (for post-processing in caller)
 * ---------------------------------------------------------------- */
static inline void FN(fft_backward_inplace)(matmul_ctx_t *ctx) {
#ifdef CTYPE_IS_REAL
    fftw_execute_dft_c2r(ctx->bwd_plan,
                         (fftw_complex *)ctx->f_wrk, (f64 *)ctx->c_wrk1);
#else
    fftw_execute_dft(ctx->bwd_plan,
                     (fftw_complex *)ctx->f_wrk, (fftw_complex *)ctx->c_wrk1);
#endif
}
