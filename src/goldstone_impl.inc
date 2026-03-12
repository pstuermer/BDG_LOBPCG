/**
 * @file goldstone_impl.inc
 * @brief Type-generic Goldstone deflation: detection + CG for deflation vectors.
 *
 * Deflation theory: goldstone_modes_derivation.md, Sections 4.3-4.5.
 * V0 = M^{-1} * psi0 (not psi0 itself!) preserves nonzero LREP eigenvalues.
 */

#include "bdg_internal.h"
#include "lobpcg/blas_wrapper.h"
#include "lobpcg.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>

/* ----------------------------------------------------------------
 * LinearOperator typedef — middle-infix naming, can't use FN()
 * All other LOBPCG calls use _Generic macros or FN(prefix_name).
 * ---------------------------------------------------------------- */
#ifdef CTYPE_IS_REAL
  typedef LinearOperator_d_t FN_linop_t;
#else
  typedef LinearOperator_z_t FN_linop_t;
#endif

/* ----------------------------------------------------------------
 * gold_store_vec: grow a deflation vector array by one entry.
 * Reallocates *vecs (CTYPE[count * size] → CTYPE[(count+1) * size]),
 * copies the new vector, computes 1/||vec||^2, bumps *count.
 * Returns ||vec|| (caller may need it for logging).
 * ---------------------------------------------------------------- */
static f64 FN(gold_store_vec)(void **vecs, f64 **inv_norms, uint64_t *count,
                               const CTYPE *vec, uint64_t size) {
  const uint64_t old_n = *count;
  const uint64_t new_n = old_n + 1;

  CTYPE *new_vecs = xcalloc(new_n * size, sizeof(CTYPE));
  f64 *new_norms = xcalloc(new_n, sizeof(f64));
  if (NULL != *vecs) {
    copy(old_n * size, (const CTYPE *)*vecs, new_vecs);
    safe_free(vecs);
  }
  if (NULL != *inv_norms) {
    d_copy(old_n, *inv_norms, new_norms);
    safe_free((void **)inv_norms);
  }

  copy(size, vec, &new_vecs[old_n * size]);
  const f64 v_norm = nrm2(size, vec);
  new_norms[old_n] = 1.0 / (v_norm * v_norm);

  *vecs = new_vecs;
  *inv_norms = new_norms;
  *count = new_n;
  return v_norm;
}

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
  const f64 V0_norm = FN(gold_store_vec)(&ctx->gold_vecK, &ctx->gold_inv_normK,
                                          &ctx->n_goldK, V0, size);
  safe_free((void **)&V0);

  printf("  bdg_deflate_u1: U(1) mode deflated (V0 norm = %.6e)\n", V0_norm);
  return 1;
}

/* ----------------------------------------------------------------
 * Linop adapters for standalone K and M (size-vector operators)
 * ---------------------------------------------------------------- */

static void FN(matvec_K_linop)(const FN_linop_t *op,
                                CTYPE *restrict x, CTYPE *restrict y) {
  matmul_ctx_t *ctx = (matmul_ctx_t *)op->ctx->data;
  FN(matmulK)(ctx, x, y);
}

static void FN(matvec_M_linop)(const FN_linop_t *op,
                                CTYPE *restrict x, CTYPE *restrict y) {
  matmul_ctx_t *ctx = (matmul_ctx_t *)op->ctx->data;
  FN(matmulM)(ctx, x, y);
}

/* ----------------------------------------------------------------
 * Gram-Schmidt orthonormalize k vectors of length n (column-major)
 * ---------------------------------------------------------------- */
static void FN(gram_schmidt)(CTYPE *V, uint64_t n, uint64_t k, f64 *inv_norms) {
  for (uint64_t j = 0; j < k; j++) {
    CTYPE *vj = V + j * n;
    /* Orthogonalize against previous */
    for (uint64_t i = 0; i < j; i++) {
      const CTYPE *vi = V + i * n;
      const CTYPE proj = dot(n, vi, vj);
      const CTYPE neg_proj = -proj;
      axpy(n, neg_proj, vi, vj);
    }
    /* Normalize */
    const f64 vj_norm = nrm2(n, vj);
    if (vj_norm > 1e-14) {
      scal(n, (CTYPE)(1.0 / vj_norm), vj);
      inv_norms[j] = 1.0;  /* ||vj|| = 1 after normalization */
    } else {
      inv_norms[j] = 0.0;  /* degenerate, skip */
    }
  }
}

/* ----------------------------------------------------------------
 * deflate_auto: U(1) check + LOBPCG on K and M, deflate all nulls
 * ---------------------------------------------------------------- */
int FN(deflate_auto)(bdg_t *bdg, uint64_t n_check, f64 tol) {
  matmul_ctx_t *ctx = bdg->ctx;
  const uint64_t size = ctx->size;

  /* ---- Phase 0: deflate U(1) analytically (always present) ---- */
  int total = FN(deflate_u1)(bdg, tol);

  linop_ctx_t lctx = { .data = ctx, .data_size = sizeof(matmul_ctx_t) };

  /* ---- Phase 1: check K for additional null vectors via LOBPCG ---- */
  {
    const uint64_t sizeSub = 2 * n_check;
    FN(lobpcg_t) *alg = FN(lobpcg_alloc)(size, n_check, sizeSub);

    FN_linop_t *A_K = linop_create(size, size,
                                   FN(matvec_K_linop), NULL, &lctx);
    alg->A = A_K;
    alg->B = NULL;
    alg->T = NULL;
    alg->maxIter = 50;
    alg->tol = tol * 0.1;

    fill_random(size * sizeSub, alg->S);

    lobpcg(alg);

    for (uint64_t j = 0; j < n_check && j < alg->converged; j++) {
      if (alg->eigVals[j] < tol) {
        printf("  deflate_auto: K null eigenvalue[%" PRIu64 "] = %.2e, deflating\n",
               j, alg->eigVals[j]);

        /* CG: M * V0 = eigvec */
        CTYPE *eigvec = &alg->S[j * size];
        CTYPE *V0 = xcalloc(size, sizeof(CTYPE));
        FN(cg_solve)(ctx, FN(matmulM), FN(precondM), eigvec, V0, size, 1e-6, 500);

        FN(gold_store_vec)(&ctx->gold_vecK, &ctx->gold_inv_normK,
                           &ctx->n_goldK, V0, size);
        safe_free((void **)&V0);
        total++;
      }
    }

    linop_destroy(&A_K);
    lobpcg_free(&alg);
  }

  /* Orthonormalize K-deflation vectors if multiple */
  if (ctx->n_goldK > 1) {
    FN(gram_schmidt)((CTYPE *)ctx->gold_vecK, size,
                     ctx->n_goldK, ctx->gold_inv_normK);
  }

  /* ---- Phase 2: check M for translational / exotic null vectors ---- */
  {
    const uint64_t sizeSub = 2 * n_check;
    FN(lobpcg_t) *alg = FN(lobpcg_alloc)(size, n_check, sizeSub);

    FN_linop_t *A_M = linop_create(size, size,
                                   FN(matvec_M_linop), NULL, &lctx);
    alg->A = A_M;
    alg->B = NULL;
    alg->T = NULL;
    alg->maxIter = 50;
    alg->tol = tol * 0.1;

    fill_random(size * sizeSub, alg->S);

    lobpcg(alg);

    for (uint64_t j = 0; j < n_check && j < alg->converged; j++) {
      if (alg->eigVals[j] < tol) {
        printf("  deflate_auto: M null eigenvalue[%" PRIu64 "] = %.2e, deflating\n",
               j, alg->eigVals[j]);

        /* CG: K_bar * W = eigvec (K_bar already includes U(1) shift) */
        CTYPE *eigvec = &alg->S[j * size];
        CTYPE *W = xcalloc(size, sizeof(CTYPE));
        FN(cg_solve)(ctx, FN(matmulK), FN(precondK), eigvec, W, size, 1e-6, 500);

        FN(gold_store_vec)(&ctx->gold_vecM, &ctx->gold_inv_normM,
                           &ctx->n_goldM, W, size);
        safe_free((void **)&W);
        total++;
      }
    }

    linop_destroy(&A_M);
    lobpcg_free(&alg);
  }

  /* Orthonormalize M-deflation vectors if multiple */
  if (ctx->n_goldM > 1) {
    FN(gram_schmidt)((CTYPE *)ctx->gold_vecM, size,
                     ctx->n_goldM, ctx->gold_inv_normM);
  }

  printf("  deflate_auto: %d total mode(s) deflated\n", total);
  return total;
}
