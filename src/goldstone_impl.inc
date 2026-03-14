/**
 * @file goldstone_impl.inc
 * @brief Type-generic Goldstone deflation: detection + rank-1 spectral shift.
 *
 * Shift formula: K̲ = K + ξ * |V₀><V₀| / ||V₀||²
 * where V₀ = M⁻¹ψ₀ for U(1) deflation.
 * Analogous shift on M for translational modes.
 */

#include "bdg_internal.h"
#include "lobpcg/blas_wrapper.h"
#include "lobpcg.h"
#include <complex.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#ifdef CTYPE_IS_REAL
  #define REAL_PART(x) (x)
#else
  #define REAL_PART(x) creal(x)
#endif

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
 * gold_precompute_sm: Sherman-Morrison preconditioner correction.
 *
 * For each deflation vector V₀_i, precompute:
 *   w₀_i = T_K(V₀_i)
 *   ξ̃_i  = ξ / ||V₀_i||²   (effective shift with normalization)
 *   σ_i  = ξ̃_i / (1 + ξ̃_i · Re⟨V₀_i, w₀_i⟩)
 * Analogous for M deflation vectors.
 * ---------------------------------------------------------------- */
void FN(gold_precompute_sm)(matmul_ctx_t *ctx) {
  const uint64_t size = ctx->size;
  const f64 xi = ctx->gold_xi;

  /* ---- K deflation ---- */
  if (ctx->n_goldK > 0) {
    const uint64_t nK = ctx->n_goldK;

    /* Free old arrays if called twice */
    safe_free(&ctx->gold_smK);
    safe_free((void **)&ctx->gold_sigmaK);

    CTYPE *smK = xcalloc(nK * size, sizeof(CTYPE));
    f64 *sigK = xcalloc(nK, sizeof(f64));

    for (uint64_t g = 0; g < nK; g++) {
      const CTYPE *v0 = (const CTYPE *)ctx->gold_vecK + g * size;
      CTYPE *w0 = &smK[g * size];

      FN(precondK)(ctx, v0, w0);

      const f64 xi_eff = xi * ctx->gold_inv_normK[g]; /* ξ/||V₀||² */
      const CTYPE d = dot(size, v0, w0);
      const f64 v0w0 = REAL_PART(d);
      sigK[g] = xi_eff / (1.0 + xi_eff * v0w0);
    }

    ctx->gold_smK = smK;
    ctx->gold_sigmaK = sigK;
  }

  /* ---- M deflation ---- */
  if (ctx->n_goldM > 0) {
    const uint64_t nM = ctx->n_goldM;

    /* Free old arrays if called twice */
    safe_free(&ctx->gold_smM);
    safe_free((void **)&ctx->gold_sigmaM);

    CTYPE *smM = xcalloc(nM * size, sizeof(CTYPE));
    f64 *sigM = xcalloc(nM, sizeof(f64));

    for (uint64_t g = 0; g < nM; g++) {
      const CTYPE *w = (const CTYPE *)ctx->gold_vecM + g * size;
      CTYPE *tw = &smM[g * size];

      FN(precondM)(ctx, w, tw);

      const f64 xi_eff = xi * ctx->gold_inv_normM[g]; /* ξ/||W||² */
      const CTYPE d = dot(size, w, tw);
      const f64 wtw = REAL_PART(d);
      sigM[g] = xi_eff / (1.0 + xi_eff * wtw);
    }

    ctx->gold_smM = smM;
    ctx->gold_sigmaM = sigM;
  }
}

/* ----------------------------------------------------------------
 * deflate_u1: detect K*psi0 ≈ 0, compute V0 = M^{-1}*psi0 via CG
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
                                   wf, V0, size, 1e-12, 5000);

  /* Compute and report final CG residual: ||M*V0 - psi0|| / ||psi0|| */
  CTYPE *Mv = xcalloc(size, sizeof(CTYPE));
  FN(matmulM)(ctx, V0, Mv);
  for (uint64_t i = 0; i < size; i++)
    Mv[i] -= wf[i];
  const f64 cg_res = nrm2(size, Mv) / wf_norm;
  safe_free((void **)&Mv);

  printf("  bdg_deflate_u1: CG %s — ||M*V0 - psi0||/||psi0|| = %.2e\n",
         (0 == cg_ret) ? "converged" : "WARNING: did not converge", cg_res);

  /* 3. Store deflation vector */
  const f64 V0_norm = FN(gold_store_vec)(&ctx->gold_vecK, &ctx->gold_inv_normK,
                                          &ctx->n_goldK, V0, size);
  safe_free((void **)&V0);

  printf("  bdg_deflate_u1: U(1) mode deflated (||V0|| = %.6e)\n", V0_norm);
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

static void FN(precond_K_linop)(const FN_linop_t *op,
                                 CTYPE *restrict x, CTYPE *restrict y) {
  matmul_ctx_t *ctx = (matmul_ctx_t *)op->ctx->data;
  FN(precondK)(ctx, x, y);
}

static void FN(precond_M_linop)(const FN_linop_t *op,
                                 CTYPE *restrict x, CTYPE *restrict y) {
  matmul_ctx_t *ctx = (matmul_ctx_t *)op->ctx->data;
  FN(precondM)(ctx, x, y);
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
    FN_linop_t *T_K = linop_create(size, size,
                                   FN(precond_K_linop), NULL, &lctx);
    alg->A = A_K;
    alg->B = NULL;
    alg->T = T_K;
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
        const int cg1 = FN(cg_solve)(ctx, FN(matmulM), FN(precondM),
                                      eigvec, V0, size, 1e-12, 5000);
        printf("  deflate_auto: K CG %s\n",
               (0 == cg1) ? "converged" : "WARNING: did not converge");

        FN(gold_store_vec)(&ctx->gold_vecK, &ctx->gold_inv_normK,
                           &ctx->n_goldK, V0, size);
        safe_free((void **)&V0);
        total++;
      }
    }

    linop_destroy(&A_K);
    linop_destroy(&T_K);
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
    FN_linop_t *T_M = linop_create(size, size,
                                   FN(precond_M_linop), NULL, &lctx);
    alg->A = A_M;
    alg->B = NULL;
    alg->T = T_M;
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
        const int cg2 = FN(cg_solve)(ctx, FN(matmulK), FN(precondK),
                                      eigvec, W, size, 1e-12, 5000);
        printf("  deflate_auto: M CG %s\n",
               (0 == cg2) ? "converged" : "WARNING: did not converge");

        FN(gold_store_vec)(&ctx->gold_vecM, &ctx->gold_inv_normM,
                           &ctx->n_goldM, W, size);
        safe_free((void **)&W);
        total++;
      }
    }

    linop_destroy(&A_M);
    linop_destroy(&T_M);
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
