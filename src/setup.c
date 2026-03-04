#include "bdg_internal.h"
#include <math.h>
#include <string.h>

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
                  f64 (*V_trap)(uint64_t dim, const f64 *r, void *param),
                  const void *param) {
    BDG_REQUIRE(bdg, BDG_HAS_SYSTEM, "bdg_set_trap");

    if (NULL == V_trap) {
        BDG_WARN("bdg_set_trap: V_trap is NULL, skipping");
        return;
    }

    matmul_ctx_t *ctx = bdg->ctx;
    f64 *const ltK = ctx->localTermK;
    f64 *const ltM = ctx->localTermM;

    if (1 == ctx->dim) {
        const uint64_t Nx = ctx->N[0];
        const f64    Lx = ctx->L[0];
        for (uint64_t ix = 0; ix < Nx; ix++) {
            const f64 x = ((f64)ix - 0.5 * (f64)Nx) * Lx / (f64)Nx;
            const f64 V = V_trap(ctx->dim, &x, param);
            ltK[ix] += V;
            ltM[ix] += V;
        }
    }

    if (2 == ctx->dim) {
        const uint64_t Nx = ctx->N[0];
        const uint64_t Ny = ctx->N[1];
        const f64    Lx = ctx->L[0];
        const f64    Ly = ctx->L[1];
        for (uint64_t iy = 0; iy < Ny; iy++) {
            const f64 y = ((f64)iy - 0.5 * (f64)Ny) * Ly / (f64)Ny;
            for (uint64_t ix = 0; ix < Nx; ix++) {
                const f64 x = ((f64)ix - 0.5 * (f64)Nx) * Lx / (f64)Nx;
                const f64 r[2] = {x, y};
                const uint64_t idx = iy * Nx + ix;
                const f64 V = V_trap(ctx->dim, r, param);
                ltK[idx] += V;
                ltM[idx] += V;
            }
        }
    }

    if (3 == ctx->dim) {
        const uint64_t Nx = ctx->N[0];
        const uint64_t Ny = ctx->N[1];
        const uint64_t Nz = ctx->N[2];
        const f64    Lx = ctx->L[0];
        const f64    Ly = ctx->L[1];
        const f64    Lz = ctx->L[2];
        for (uint64_t iz = 0; iz < Nz; iz++) {
            const f64 z = ((f64)iz - 0.5 * (f64)Nz) * Lz / (f64)Nz;
            for (uint64_t iy = 0; iy < Ny; iy++) {
                const f64 y = ((f64)iy - 0.5 * (f64)Ny) * Ly / (f64)Ny;
                for (uint64_t ix = 0; ix < Nx; ix++) {
                    const f64 x = ((f64)ix - 0.5 * (f64)Nx) * Lx / (f64)Nx;
                    const f64 r[3] = {x, y, z};
                    const uint64_t idx = iz * Ny * Nx + iy * Nx + ix;
                    const f64 V = V_trap(ctx->dim, r, param);
                    ltK[idx] += V;
                    ltM[idx] += V;
                }
            }
        }
    }

    bdg->state |= BDG_HAS_TRAP;
}

/* ----------------------------------------------------------------
 * bdg_set_wavefunction — copy wf into ctx (in-memory)
 * ---------------------------------------------------------------- */
void bdg_set_wavefunction(bdg_t *bdg, const void *wf, uint64_t wf_size) {
    BDG_REQUIRE(bdg, BDG_HAS_SYSTEM, "bdg_set_wavefunction");

    matmul_ctx_t *ctx = bdg->ctx;

    if (NULL == wf)
        BDG_ERROR("bdg_set_wavefunction: wf is NULL");

    if (wf_size != ctx->size)
        BDG_ERROR("bdg_set_wavefunction: wf_size (%zu) != grid size (%zu)",
                  wf_size, ctx->size);

    /* Free previous wf if re-calling (e.g. after bdg_reset) */
    safe_free((void **)&ctx->wf);

    if (bdg->complex_psi0) {
        ctx->wf = xcalloc(ctx->size, sizeof(c64));
        memcpy(ctx->wf, wf, ctx->size * sizeof(c64));
    } else {
        ctx->wf = xcalloc(ctx->size, sizeof(f64));
        memcpy(ctx->wf, wf, ctx->size * sizeof(f64));
    }
    ctx->wf_size = ctx->size;

    bdg->state |= BDG_HAS_WF;
}

/* ----------------------------------------------------------------
 * bdg_set_local_interactions — U_intK and U_intM function pointers
 * ---------------------------------------------------------------- */
void bdg_set_local_interactions(bdg_t *bdg,
                                f64 (*U_intK)(void *param, f64 density),
                                f64 (*U_intM)(void *param, f64 density),
                                const void *param) {
    BDG_REQUIRE(bdg, BDG_HAS_WF, "bdg_set_local_interactions");
    BDG_FORBID(bdg, BDG_HAS_INTERACTIONS, "bdg_set_local_interactions");

    matmul_ctx_t *ctx = bdg->ctx;
    const uint64_t size = ctx->size;

    for (uint64_t i = 0; i < size; i++) {
        f64 density;
        if (bdg->complex_psi0) {
            const c64 psi = ((const c64 *)ctx->wf)[i];
            density = creal(psi) * creal(psi) + cimag(psi) * cimag(psi);
        } else {
            const f64 psi = ((const f64 *)ctx->wf)[i];
            density = psi * psi;
        }
        const f64 uK = U_intK(param, density);
        const f64 uM = U_intM(param, density);
        ctx->localTermK[i] += uK;
        ctx->localTermM[i] += uK + 2.0 * uM;
    }

    bdg->state |= BDG_HAS_INTERACTIONS;
}

/* ----------------------------------------------------------------
 * bdg_set_mu — subtract mu from localTermK/M, store for preconditioner
 * ---------------------------------------------------------------- */
void bdg_set_mu(bdg_t *bdg, f64 mu) {
    BDG_REQUIRE(bdg, BDG_HAS_SYSTEM, "bdg_set_mu");

    matmul_ctx_t *ctx = bdg->ctx;
    const uint64_t size = ctx->size;

    /* Allocate preconditioner arrays */
    if (NULL == ctx->precond_sqrtK)
        ctx->precond_sqrtK = xcalloc(size, sizeof(f64));
    if (NULL == ctx->precond_sqrtM)
        ctx->precond_sqrtM = xcalloc(size, sizeof(f64));

    /* Compute preconditioner BEFORE subtracting mu
     * (uses localTermK/M which still include mu contribution) */
    for (uint64_t i = 0; i < size; i++) {
        ctx->precond_sqrtK[i] = 1.0 / sqrt(fabs(safe_val(ctx->localTermK[i])));
        ctx->precond_sqrtM[i] = 1.0 / sqrt(fabs(safe_val(ctx->localTermM[i])));
    }

    /* Subtract mu */
    for (uint64_t i = 0; i < size; i++) {
        ctx->localTermK[i] -= mu;
        ctx->localTermM[i] -= mu;
    }

    ctx->mu = mu;
    bdg->state |= BDG_HAS_MU;
}

/* ----------------------------------------------------------------
 * bdg_set_dipolar — kernel + mean-field (3D only)
 * ---------------------------------------------------------------- */
void bdg_set_dipolar(bdg_t *bdg, f64 g_ddi, const f64 *dipole_dir, f64 cutoff_radius) {
    BDG_REQUIRE(bdg, BDG_HAS_WF, "bdg_set_dipolar");
    BDG_FORBID(bdg, BDG_HAS_MU, "bdg_set_dipolar");

    matmul_ctx_t *ctx = bdg->ctx;

    if (3 != ctx->dim)
        BDG_ERROR("bdg_set_dipolar: dipolar interactions require dim == 3 (got %zu)",
                  ctx->dim);

    dipolar_set_kernel(ctx, g_ddi, dipole_dir, cutoff_radius);
    dipolar_add_meanfield(ctx);
    ctx->dipolar = 1;

    bdg->state |= BDG_HAS_DIPOLAR;
}
