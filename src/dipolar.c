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
void dipolar_set_kernel(matmul_ctx_t *ctx, f64 g_ddi, const f64 *dipole_dir,
                         f64 cutoff_radius) {
    ctx->g_ddi = g_ddi;

    const uint64_t Nx = ctx->N[0], Ny = ctx->N[1], Nz = ctx->N[2];
    const uint64_t N0k = ctx->complex_psi0 ? Nx : (Nx / 2 + 1);
    const f64 dkx = 2.0 * M_PI / ctx->L[0];
    const f64 dky = 2.0 * M_PI / ctx->L[1];
    const f64 dkz = 2.0 * M_PI / ctx->L[2];
    const f64 dir_sq = dipole_dir[0] * dipole_dir[0]
                     + dipole_dir[1] * dipole_dir[1]
                     + dipole_dir[2] * dipole_dir[2];

    ctx->longRngInt = xcalloc(ctx->k_size, sizeof(f64));

    for (uint64_t iz = 0; iz < Nz; iz++) {
        const int fz = (iz <= Nz / 2) ? (int)iz : (int)iz - (int)Nz;
        const f64 kz = dkz * fz;

        for (uint64_t iy = 0; iy < Ny; iy++) {
            const int fy = (iy <= Ny / 2) ? (int)iy : (int)iy - (int)Ny;
            const f64 ky = dky * fy;

            for (uint64_t ix = 0; ix < N0k; ix++) {
                const int fx = (ctx->complex_psi0 && ix > Nx / 2)
                             ? (int)ix - (int)Nx : (int)ix;
                const f64 kx = dkx * fx;

                const uint64_t idx = iz * Ny * N0k + iy * N0k + ix;
                const f64 ksq = kx * kx + ky * ky + kz * kz;

                if (ksq < 1e-50) {
                    ctx->longRngInt[idx] = 0.0;
                    continue;
                }

                const f64 kdotd = kx * dipole_dir[0]
                                + ky * dipole_dir[1]
                                + kz * dipole_dir[2];
                const f64 cos_sq_theta = (kdotd * kdotd) / (ksq * dir_sq);

                const f64 k_mag = sqrt(ksq);
                const f64 kR  = k_mag * cutoff_radius;
                const f64 kR2 = kR * kR;
                const f64 f_cutoff = 1.0 + 3.0 * cos(kR) / kR2
                                         - 3.0 * sin(kR) / (kR2 * kR);

                ctx->longRngInt[idx] = (3.0 * cos_sq_theta - 1.0) * f_cutoff;
            }
        }
    }
}

/* ----------------------------------------------------------------
 * dipolar_add_meanfield — Phi_dd added to localTermK/M
 * ---------------------------------------------------------------- */
void dipolar_add_meanfield(matmul_ctx_t *ctx) {
    if (ctx->complex_psi0)
        dipolar_add_meanfield_z(ctx);
    else
        dipolar_add_meanfield_d(ctx);
}
