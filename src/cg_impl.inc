/**
 * @file cg_impl.inc
 * @brief Type-generic preconditioned CG solver for SPD/PSD systems.
 *
 * Solves A*x = b. Preconditioner is optional (NULL = identity).
 * For PSD matrices: start with x=0 and ensure b in range(A).
 * Returns 0 on convergence, -1 on maxiter.
 *
 * Reference: ~/LREP_post/src/precond.c:213-312
 */

#include "bdg_internal.h"
#include "lobpcg/blas_wrapper.h"
#include <math.h>

int FN(cg_solve)(matmul_ctx_t *ctx,
                 void (*A)(matmul_ctx_t *, const CTYPE *, CTYPE *),
                 void (*precond)(matmul_ctx_t *, const CTYPE *, CTYPE *),
                 const CTYPE *b, CTYPE *x, uint64_t n, f64 tol, uint64_t maxiter) {

  CTYPE *r = xcalloc(n, sizeof(CTYPE));
  CTYPE *z = xcalloc(n, sizeof(CTYPE));
  CTYPE *p = xcalloc(n, sizeof(CTYPE));
  CTYPE *q = xcalloc(n, sizeof(CTYPE));

  /* r = b - A*x (x should be zero-initialized for PSD) */
  A(ctx, x, q);
  for (uint64_t i = 0; i < n; i++)
    r[i] = b[i] - q[i];

  const f64 b_norm = nrm2(n, b);
  const f64 threshold = tol * b_norm;

  /* precondition: z = P^{-1} r */
  if (NULL != precond) {
    precond(ctx, r, z);
  } else {
    copy(n, r, z);
  }

  /* p = z */
  copy(n, z, p);

  CTYPE rz_old = dot(n, r, z);

  int status = -1;

  for (uint64_t iter = 0; iter < maxiter; iter++) {
    /* Check convergence */
    const f64 r_norm = nrm2(n, r);
    if (r_norm < threshold) {
      status = 0;
      break;
    }

    /* q = A*p */
    A(ctx, p, q);

    /* alpha = rz_old / (p^H q) */
    const CTYPE pq = dot(n, p, q);

#ifdef CTYPE_IS_REAL
    const f64 pq_abs = fabs(pq);
#else
    const f64 pq_abs = cabs(pq);
#endif

    /* PSD breakdown: p in null space */
    if (pq_abs < 1e-30) {
      status = 0;
      break;
    }

    const CTYPE alpha = rz_old / pq;

    /* x += alpha * p */
    axpy(n, alpha, p, x);

    /* r -= alpha * q */
    const CTYPE neg_alpha = -alpha;
    axpy(n, neg_alpha, q, r);

    /* precondition: z = P^{-1} r */
    if (NULL != precond) {
      precond(ctx, r, z);
    } else {
      copy(n, r, z);
    }

    /* beta = rz_new / rz_old */
    const CTYPE rz_new = dot(n, r, z);
    const CTYPE beta = rz_new / rz_old;
    rz_old = rz_new;

    /* p = z + beta * p */
    scal(n, beta, p);
    axpy(n, (CTYPE)1.0, z, p);
  }

  safe_free((void **)&r);
  safe_free((void **)&z);
  safe_free((void **)&p);
  safe_free((void **)&q);

  return status;
}
