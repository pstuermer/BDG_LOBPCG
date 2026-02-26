#include "bdg_internal.h"
#include "lobpcg.h"

/**
 * @file solver.c
 * @brief LinearOperator wrapping and ilobpcg dispatch.
 *
 * Creates linop wrappers around matmulLrep, matmulSwap, precondLrep,
 * then calls d_ilobpcg or z_ilobpcg.
 *
 * Reference: ~/LREP_post/examples/3d_dipolar_atom.c (setup + solve flow)
 */

/* ================================================================
 * Linop adapters — thin wrappers matching LinearOperator matvec signature
 * ================================================================ */

/* --- Real (d) adapters --- */

static void matvec_lrep_d(const LinearOperator_d_t *op,
                           f64 *restrict x, f64 *restrict y) {
    matmul_ctx_t *ctx = (matmul_ctx_t *)op->ctx->data;
    matmulLrep_d(ctx, x, y);
}

static void matvec_swap_d(const LinearOperator_d_t *op,
                           f64 *restrict x, f64 *restrict y) {
    matmul_ctx_t *ctx = (matmul_ctx_t *)op->ctx->data;
    matmulSwap_d(ctx, x, y);
}

static void matvec_precond_d(const LinearOperator_d_t *op,
                              f64 *restrict x, f64 *restrict y) {
    matmul_ctx_t *ctx = (matmul_ctx_t *)op->ctx->data;
    precondLrep_d(ctx, x, y);
}

/* --- Complex (z) adapters --- */

static void matvec_lrep_z(const LinearOperator_z_t *op,
                           c64 *restrict x, c64 *restrict y) {
    matmul_ctx_t *ctx = (matmul_ctx_t *)op->ctx->data;
    matmulLrep_z(ctx, x, y);
}

static void matvec_swap_z(const LinearOperator_z_t *op,
                           c64 *restrict x, c64 *restrict y) {
    matmul_ctx_t *ctx = (matmul_ctx_t *)op->ctx->data;
    matmulSwap_z(ctx, x, y);
}

static void matvec_precond_z(const LinearOperator_z_t *op,
                              c64 *restrict x, c64 *restrict y) {
    matmul_ctx_t *ctx = (matmul_ctx_t *)op->ctx->data;
    precondLrep_z(ctx, x, y);
}

/* ================================================================
 * bdg_solve_d — real path
 * ================================================================ */
int bdg_solve_d(bdg_t *bdg) {
    /* TODO: Implement
     *
     * size_t n = 2 * bdg->ctx->size;
     *
     * 1. d_lobpcg_t *alg = d_ilobpcg_alloc(n, bdg->nev, bdg->sizeSub);
     *
     * 2. Create linop_ctx_t with data = bdg->ctx:
     *    linop_ctx_t linop_ctx = { .data = bdg->ctx, .data_size = sizeof(matmul_ctx_t) };
     *
     * 3. Create LinearOperators:
     *    A = linop_create_d(n, n, matvec_lrep_d,    NULL, &linop_ctx);
     *    B = linop_create_d(n, n, matvec_swap_d,    NULL, &linop_ctx);
     *    T = linop_create_d(n, n, matvec_precond_d, NULL, &linop_ctx);
     *
     * 4. alg->A = A; alg->B = B; alg->T = T;
     *    alg->maxIter = bdg->maxIter; alg->tol = bdg->tol;
     *
     * 5. Initialize: d_fill_random(n * bdg->sizeSub, alg->S);
     *    Make B-positive: set S[i+size] = S[i] for each column.
     *
     * 6. d_ilobpcg(alg);
     *
     * 7. Copy results:
     *    bdg->converged = alg->converged;
     *    bdg->eigvals = xcalloc(bdg->nev, sizeof(f64));
     *    memcpy(bdg->eigvals, alg->eigVals, bdg->nev * sizeof(f64));
     *    Split alg->S columns into modes_u and modes_v.
     *
     * 8. Cleanup: linop_destroy_d(&A); etc.; d_lobpcg_free(&alg);
     *
     * return 0;
     */
    (void)bdg;
    return -1; /* Not implemented */
}

/* ================================================================
 * bdg_solve_z — complex path
 * ================================================================ */
int bdg_solve_z(bdg_t *bdg) {
    /* TODO: Same as bdg_solve_d but with z_ types and c64 arrays. */
    (void)bdg;
    return -1; /* Not implemented */
}
