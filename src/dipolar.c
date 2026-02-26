#include "bdg_internal.h"
#include <math.h>

/**
 * @file dipolar.c
 * @brief Dipolar kernel construction (3D only) and mean-field potential.
 *
 * Type-independent: kernel is always f64 in k-space.
 * The typed perturbation convolution is in dipolar_conv_{d,z}.c.
 *
 * Reference: ~/LREP_post/src/dipolar.c (lines 3-50 kernel, 103-127 meanfield)
 */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ----------------------------------------------------------------
 * dipolar_set_kernel — build longRngInt in k-space (3D only)
 * ---------------------------------------------------------------- */
void dipolar_set_kernel(matmul_ctx_t *ctx, f64 g_ddi, const f64 *dipole_dir) {
    /* TODO: Implement
     *
     * 3D only. Triple loop over k-grid.
     * longRngInt[i] = (3*cos^2(theta_k) - 1) * f_cutoff(|k|, R_c)
     * where theta_k = angle between k and dipole_dir.
     *
     * Spherical cutoff:
     *   kR = |k| * R_c     (R_c = L[d]/2 typically)
     *   f_cutoff = 1 + 3*cos(kR)/(kR)^2 - 3*sin(kR)/(kR)^3
     *   f_cutoff(k=0) = 0   (removes divergence)
     *
     * ctx->g_ddi = g_ddi
     * ctx->longRngInt = allocated array of k_size f64
     *
     * Reference: ~/LREP_post/src/dipolar.c:3-50
     */
    (void)ctx; (void)g_ddi; (void)dipole_dir;
}

/* ----------------------------------------------------------------
 * dipolar_add_meanfield — Phi_dd added to localTermK/M
 * ---------------------------------------------------------------- */
void dipolar_add_meanfield(matmul_ctx_t *ctx) {
    /* TODO: Implement
     *
     * 1. Compute |wf|^2 into scratch
     * 2. FFT(|wf|^2) → f_wrk
     * 3. f_wrk[i] *= longRngInt[i] / size
     * 4. IFFT(f_wrk) → scratch  (this is Phi_dd)
     * 5. localTermK[i] += g_ddi * Phi_dd[i]
     *    localTermM[i] += g_ddi * Phi_dd[i]
     *
     * Reference: ~/LREP_post/src/dipolar.c:103-127
     */
    (void)ctx;
}
