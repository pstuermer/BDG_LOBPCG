/**
 * @file kinetic_impl.inc
 * @brief Type-generic kinetic energy operator: y = 0.5 * k^2 * x via FFT.
 *
 * Expects macros: CTYPE (f64 or c64), RTYPE (f64), FN(name) (suffix helper),
 * and either CTYPE_IS_REAL or CTYPE_IS_COMPLEX.
 *
 * Real path uses r2c/c2r (k_size < size).
 * Complex path uses c2c (k_size == size).
 *
 * Reference: ~/LREP_post/src/kinetic.c:4-41
 */

#include "bdg_internal.h"
#include <string.h>

void FN(kinetic)(matmul_ctx_t *ctx, const CTYPE *x, CTYPE *y) {
    /* TODO: Implement
     *
     * 1. Copy x into scratch (f_wrk for r2c, or c_wrk1 for c2c)
     * 2. Execute fwd_plan (r2c or c2c)
     * 3. In k-space: f_wrk[i] *= 0.5 * k2[i] / size  (loop over k_size)
     * 4. Execute bwd_plan (c2r or c2c)
     * 5. Copy result into y
     *
     * For r2c:
     *   - Input is f64[size], output f_wrk is c64[k_size]
     *   - fftw_execute_dft_r2c(fwd_plan, (double*)scratch, (fftw_complex*)f_wrk)
     *   - After IFFT: fftw_execute_dft_c2r(bwd_plan, f_wrk, scratch)
     *   - Copy scratch (f64) to y
     *
     * For c2c:
     *   - Input/output are c64[size]
     *   - fftw_execute_dft(fwd_plan, scratch, f_wrk)
     *   - After multiply: fftw_execute_dft(bwd_plan, f_wrk, scratch)
     *   - Copy scratch to y
     */
    (void)ctx; (void)x; (void)y;
}
