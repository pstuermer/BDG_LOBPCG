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
 * LOBPCG type aliases — explicit ifdefs since LOBPCG types
 * don't follow our CONCAT/FN() pattern
 * ---------------------------------------------------------------- */
#ifdef CTYPE_IS_REAL
  typedef d_lobpcg_t FN_lobpcg_t;
  typedef LinearOperator_d_t FN_linop_t;
  #define FN_lobpcg_alloc d_lobpcg_alloc
  #define FN_lobpcg_run d_lobpcg
  #define FN_lobpcg_free d_lobpcg_free
  #define FN_linop_create linop_create_d
  #define FN_linop_destroy linop_destroy_d
  #define FN_fill_random d_fill_random
#else
  typedef z_lobpcg_t FN_lobpcg_t;
  typedef LinearOperator_z_t FN_linop_t;
  #define FN_lobpcg_alloc z_lobpcg_alloc
  #define FN_lobpcg_run z_lobpcg
  #define FN_lobpcg_free z_lobpcg_free
  #define FN_linop_create linop_create_z
  #define FN_linop_destroy linop_destroy_z
  #define FN_fill_random z_fill_random
#endif

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
 * deflate_auto: run definite LOBPCG on K_bar and M, deflate nulls
 * ---------------------------------------------------------------- */
int FN(deflate_auto)(bdg_t *bdg, uint64_t n_check, f64 tol) {
  matmul_ctx_t *ctx = bdg->ctx;
  const uint64_t size = ctx->size;
  int total = 0;

  linop_ctx_t lctx = { .data = ctx, .data_size = sizeof(matmul_ctx_t) };

  /* ---- Phase 1: check K (post-U1) for exotic null vectors ---- */
  {
    const uint64_t sizeSub = 2 * n_check;
    FN_lobpcg_t *alg = FN_lobpcg_alloc(size, n_check, sizeSub);

    FN_linop_t *A_K = FN_linop_create(size, size,
                                        FN(matvec_K_linop), NULL, &lctx);
    alg->A = A_K;
    alg->B = NULL;
    alg->T = NULL;
    alg->maxIter = 50;
    alg->tol = tol * 0.1;

    /* Random init */
    FN_fill_random(size * sizeSub, alg->S);

    FN_lobpcg_run(alg);

    for (uint64_t j = 0; j < n_check && j < alg->converged; j++) {
      if (alg->eigVals[j] < tol) {
        printf("  deflate_auto: K null eigenvalue[%" PRIu64 "] = %.2e, deflating\n",
               j, alg->eigVals[j]);

        /* CG: M * V0 = eigvec */
        CTYPE *eigvec = &alg->S[j * size];
        CTYPE *V0 = xcalloc(size, sizeof(CTYPE));
        FN(cg_solve)(ctx, FN(matmulM), FN(precondM), eigvec, V0, size, 1e-6, 500);

        /* Store */
        const uint64_t old_n = ctx->n_goldK;
        const uint64_t new_n = old_n + 1;
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
        copy(size, V0, &new_vecs[old_n * size]);
        const f64 V0_n = nrm2(size, V0);
        new_norms[old_n] = 1.0 / (V0_n * V0_n);
        ctx->gold_vecK = new_vecs;
        ctx->gold_inv_normK = new_norms;
        ctx->n_goldK = new_n;

        safe_free((void **)&V0);
        total++;
      }
    }

    FN_linop_destroy(&A_K);
    FN_lobpcg_free(&alg);
  }

  /* Orthonormalize K-deflation vectors if multiple */
  if (ctx->n_goldK > 1) {
    FN(gram_schmidt)((CTYPE *)ctx->gold_vecK, size,
                     ctx->n_goldK, ctx->gold_inv_normK);
  }

  /* ---- Phase 2: check M for translational / exotic null vectors ---- */
  {
    const uint64_t sizeSub = 2 * n_check;
    FN_lobpcg_t *alg = FN_lobpcg_alloc(size, n_check, sizeSub);

    FN_linop_t *A_M = FN_linop_create(size, size,
                                        FN(matvec_M_linop), NULL, &lctx);
    alg->A = A_M;
    alg->B = NULL;
    alg->T = NULL;
    alg->maxIter = 50;
    alg->tol = tol * 0.1;

    FN_fill_random(size * sizeSub, alg->S);

    FN_lobpcg_run(alg);

    for (uint64_t j = 0; j < n_check && j < alg->converged; j++) {
      if (alg->eigVals[j] < tol) {
        printf("  deflate_auto: M null eigenvalue[%" PRIu64 "] = %.2e, deflating\n",
               j, alg->eigVals[j]);

        /* CG: K_bar * W = eigvec (K_bar already includes U(1) shift) */
        CTYPE *eigvec = &alg->S[j * size];
        CTYPE *W = xcalloc(size, sizeof(CTYPE));
        FN(cg_solve)(ctx, FN(matmulK), FN(precondK), eigvec, W, size, 1e-6, 500);

        /* Store */
        const uint64_t old_n = ctx->n_goldM;
        const uint64_t new_n = old_n + 1;
        CTYPE *new_vecs = xcalloc(new_n * size, sizeof(CTYPE));
        f64 *new_norms = xcalloc(new_n, sizeof(f64));
        if (NULL != ctx->gold_vecM) {
          copy(old_n * size, (const CTYPE *)ctx->gold_vecM, new_vecs);
          safe_free((void **)&ctx->gold_vecM);
        }
        if (NULL != ctx->gold_inv_normM) {
          d_copy(old_n, ctx->gold_inv_normM, new_norms);
          safe_free((void **)&ctx->gold_inv_normM);
        }
        copy(size, W, &new_vecs[old_n * size]);
        const f64 W_n = nrm2(size, W);
        new_norms[old_n] = 1.0 / (W_n * W_n);
        ctx->gold_vecM = new_vecs;
        ctx->gold_inv_normM = new_norms;
        ctx->n_goldM = new_n;

        safe_free((void **)&W);
        total++;
      }
    }

    FN_linop_destroy(&A_M);
    FN_lobpcg_free(&alg);
  }

  /* Orthonormalize M-deflation vectors if multiple */
  if (ctx->n_goldM > 1) {
    FN(gram_schmidt)((CTYPE *)ctx->gold_vecM, size,
                     ctx->n_goldM, ctx->gold_inv_normM);
  }

  printf("  deflate_auto: %d additional mode(s) deflated\n", total);
  return total;
}
