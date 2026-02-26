#include "bdg_internal.h"

/**
 * @file bdg.c
 * @brief Top-level BdG lifecycle: alloc, free, solve dispatch.
 */

/* ----------------------------------------------------------------
 * bdg_alloc
 * ---------------------------------------------------------------- */
bdg_t *bdg_alloc(size_t dim, const size_t *N, const f64 *L, int complex_psi0) {
    bdg_t *bdg = xcalloc(1, sizeof(bdg_t));
    bdg->complex_psi0 = complex_psi0;
    bdg->ctx = matmul_ctx_alloc(dim, N, L);
    bdg->ctx->complex_psi0 = complex_psi0;

    /* Defaults */
    bdg->nev     = 3;
    bdg->sizeSub = 6;
    bdg->maxIter = 500;
    bdg->tol     = 1e-6;

    return bdg;
}

/* ----------------------------------------------------------------
 * bdg_free
 * ---------------------------------------------------------------- */
void bdg_free(bdg_t **bdg) {
    if (!bdg || !*bdg) return;
    bdg_t *b = *bdg;

    /* Results */
    if (b->eigvals)  safe_free((void **)&b->eigvals);
    if (b->modes_u)  safe_free((void **)&b->modes_u);
    if (b->modes_v)  safe_free((void **)&b->modes_v);

    /* Context */
    matmul_ctx_free(&b->ctx);

    safe_free((void **)bdg);
}

/* ----------------------------------------------------------------
 * Setup pass-throughs
 * ---------------------------------------------------------------- */
void bdg_set_system(bdg_t *bdg) {
    matmul_ctx_set_system(bdg->ctx, bdg->complex_psi0);
}

void bdg_set_solver_params(bdg_t *bdg, size_t nev, size_t sizeSub,
                           size_t maxIter, f64 tol) {
    bdg->nev     = nev;
    bdg->sizeSub = sizeSub;
    bdg->maxIter = maxIter;
    bdg->tol     = tol;
}

/* ----------------------------------------------------------------
 * bdg_solve — dispatches to d or z path
 * ---------------------------------------------------------------- */
int bdg_solve(bdg_t *bdg) {
    if (bdg->complex_psi0)
        return bdg_solve_z(bdg);
    else
        return bdg_solve_d(bdg);
}

/* ----------------------------------------------------------------
 * Result accessors
 * ---------------------------------------------------------------- */
size_t bdg_converged(const bdg_t *bdg) {
    return bdg->converged;
}

const f64 *bdg_eigenvalues(const bdg_t *bdg) {
    return bdg->eigvals;
}

const void *bdg_modes_u(const bdg_t *bdg) {
    return bdg->modes_u;
}

const void *bdg_modes_v(const bdg_t *bdg) {
    return bdg->modes_v;
}
