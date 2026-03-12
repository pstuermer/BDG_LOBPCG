/**
 * @file goldstone_impl.inc
 * @brief Type-generic Goldstone deflation: detection + CG for deflation vectors.
 *
 * Deflation theory: goldstone_modes_derivation.md, Sections 4.3-4.5.
 * V0 = M^{-1} * psi0 (not psi0 itself!) preserves nonzero LREP eigenvalues.
 */

#include "bdg_internal.h"
#include "lobpcg/blas_wrapper.h"
#include <math.h>
#include <stdio.h>

/* ----------------------------------------------------------------
 * deflate_u1: detect K*psi0 = 0, compute V0 = M^{-1}*psi0 via CG
 * ---------------------------------------------------------------- */
int FN(deflate_u1)(bdg_t *bdg, f64 tol) {
  matmul_ctx_t *ctx = bdg->ctx;
  const uint64_t size = ctx->size;
  const CTYPE *wf = (const CTYPE *)ctx->wf;

  /* 1. Check ||K*psi0|| / ||psi0|| < tol */
  CTYPE *Kpsi = xcalloc(size, sizeof(CTYPE));
  FN(matmulK)(ctx, wf, Kpsi);

  const f64 Kpsi_norm = nrm2(size, Kpsi);
  const f64 wf_norm = nrm2(size, wf);
  const f64 residual = Kpsi_norm / wf_norm;
  safe_free((void **)&Kpsi);

  printf("  bdg_deflate_u1: ||K*psi0||/||psi0|| = %.2e (tol=%.2e)\n",
         residual, tol);

  if (residual >= tol) {
    printf("  bdg_deflate_u1: residual above threshold, skipping\n");
    return 0;
  }

  /* 2. CG solve: M * V0 = psi0 */
  CTYPE *V0 = xcalloc(size, sizeof(CTYPE));
  const int cg_ret = FN(cg_solve)(ctx, FN(matmulM), FN(precondM),
                                   wf, V0, size, 1e-6, 500);
  if (0 != cg_ret)
    printf("  bdg_deflate_u1: WARNING — CG did not converge\n");

  /* 3. Store deflation vector */
  const uint64_t old_n = ctx->n_goldK;
  const uint64_t new_n = old_n + 1;

  /* Reallocate (expand) gold_vecK and gold_inv_normK */
  CTYPE *new_vecs = xcalloc(new_n * size, sizeof(CTYPE));
  f64 *new_norms = xcalloc(new_n, sizeof(f64));
  if (NULL != ctx->gold_vecK) {
    copy(old_n * size, (const CTYPE *)ctx->gold_vecK, new_vecs);
    safe_free((void **)&ctx->gold_vecK);
  }
  if (NULL != ctx->gold_inv_normK) {
    d_copy(old_n, ctx->gold_inv_normK, new_norms);
    safe_free((void **)&ctx->gold_inv_normK);
  }

  /* Copy V0 into slot old_n */
  copy(size, V0, &new_vecs[old_n * size]);
  const f64 V0_norm2 = nrm2(size, V0);
  new_norms[old_n] = 1.0 / (V0_norm2 * V0_norm2);

  ctx->gold_vecK = new_vecs;
  ctx->gold_inv_normK = new_norms;
  ctx->n_goldK = new_n;

  safe_free((void **)&V0);

  printf("  bdg_deflate_u1: U(1) mode deflated (V0 norm = %.6e)\n", V0_norm2);
  return 1;
}

/* Stub — implemented in Task 5 */
int FN(deflate_auto)(bdg_t *bdg, uint64_t n_check, f64 tol) {
  (void)bdg; (void)n_check; (void)tol;
  return 0;
}
