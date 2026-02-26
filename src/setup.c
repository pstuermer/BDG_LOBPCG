#include "bdg_internal.h"
#include <math.h>

/**
 * @file setup.c
 * @brief Physics setup: trap, wavefunction, interactions, chemical potential.
 *
 * Port from ~/LREP_post/src/matmul.c:211-404.
 * Wavefunction accepts in-memory data (not file I/O).
 */

/* ----------------------------------------------------------------
 * bdg_set_trap — add V_trap(r) to localTermK and localTermM
 * ---------------------------------------------------------------- */
void bdg_set_trap(bdg_t *bdg,
                  f64 (*V_trap)(size_t dim, const f64 *r, void *param),
                  void *param) {
    /* TODO: Implement
     *
     * Loop over grid points, compute x-coordinates on the fly:
     *   x[d] = (i_d - N[d]/2) * L[d] / N[d]   (centered grid)
     * Evaluate V_trap(dim, r, param) and add to both localTermK and localTermM.
     *
     * Use nested loops or linear index with modular arithmetic.
     * Position-space coordinates are transient (not stored).
     *
     * Reference: ~/LREP_post/src/matmul.c:211-275
     */
    (void)bdg; (void)V_trap; (void)param;
}

/* ----------------------------------------------------------------
 * bdg_set_wavefunction — copy wf into ctx (in-memory)
 * ---------------------------------------------------------------- */
void bdg_set_wavefunction(bdg_t *bdg, const void *wf, size_t wf_size) {
    /* TODO: Implement
     *
     * Validate wf_size == ctx->size.
     * Allocate ctx->wf (f64 or c64 depending on complex_psi0).
     * memcpy from wf.
     *
     * Adapted from ~/LREP_post/src/matmul.c:329-368 (was file I/O).
     */
    (void)bdg; (void)wf; (void)wf_size;
}

/* ----------------------------------------------------------------
 * bdg_set_local_interactions — U_intK and U_intM function pointers
 * ---------------------------------------------------------------- */
void bdg_set_local_interactions(bdg_t *bdg,
                                f64 (*U_intK)(void *param, f64 density),
                                f64 (*U_intM)(void *param, f64 density),
                                void *param) {
    /* TODO: Implement
     *
     * For each grid point i:
     *   density = |wf[i]|^2   (real: wf[i]^2, complex: cabs(wf[i])^2)
     *   localTermK[i] += U_intK(param, density)
     *   localTermM[i] += U_intK(param, density) + 2.0 * U_intM(param, density)
     *
     * Requires wf to be set first.
     *
     * Reference: ~/LREP_post/src/matmul.c:279-296
     */
    (void)bdg; (void)U_intK; (void)U_intM; (void)param;
}

/* ----------------------------------------------------------------
 * bdg_set_mu — subtract mu from localTermK/M, store for preconditioner
 * ---------------------------------------------------------------- */
void bdg_set_mu(bdg_t *bdg, f64 mu) {
    /* TODO: Implement
     *
     * ctx->mu = mu;
     * for i: localTermK[i] -= mu;
     * for i: localTermM[i] -= mu;
     *
     * Also compute preconditioner sqrt arrays:
     *   precond_sqrtK[i] = 1.0 / sqrt(localTermK[i] + mu)
     *   precond_sqrtM[i] = 1.0 / sqrt(localTermM[i] + mu)
     * (these are localTermK/M BEFORE mu subtraction, i.e., localTermK[i]+mu)
     *
     * Reference: ~/LREP_post/src/matmul.c:398-404
     */
    (void)bdg; (void)mu;
}

/* ----------------------------------------------------------------
 * bdg_set_dipolar — kernel + mean-field (3D only)
 * ---------------------------------------------------------------- */
void bdg_set_dipolar(bdg_t *bdg, f64 g_ddi, const f64 *dipole_dir) {
    /* TODO: Implement
     *
     * 1. dipolar_set_kernel(ctx, g_ddi, dipole_dir)  — builds longRngInt
     * 2. dipolar_add_meanfield(ctx)  — adds Phi_dd to localTermK/M
     * 3. ctx->dipolar = 1
     *
     * Must be called AFTER set_wavefunction and BEFORE set_mu.
     */
    (void)bdg; (void)g_ddi; (void)dipole_dir;
}
