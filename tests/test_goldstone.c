#include "bdg_internal.h"
#include "lobpcg/blas_wrapper.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define TOL 1e-10

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) static void test_##name(void)
#define RUN(name) do { \
    printf("  %-50s ", #name); \
    test_##name(); \
    printf("[PASS]\n"); \
    tests_passed++; \
} while(0)

#define ASSERT(cond) do { \
    if (!(cond)) { \
        printf("[FAIL] line %d: %s\n", __LINE__, #cond); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_CLOSE(a, b, tol) do { \
    const double _a = (a), _b = (b); \
    if (fabs(_a - _b) >= (tol)) { \
        printf("[FAIL] line %d: %.15e vs %.15e (diff %.2e)\n", __LINE__, _a, _b, fabs(_a-_b)); \
        tests_failed++; \
        return; \
    } \
} while(0)

/* Helper: 1D ctx with constant localTermK = V0 - mu */
static matmul_ctx_t *make_1d_ctx(const uint64_t N, const f64 L,
                                  const f64 V0, const f64 mu) {
  const uint64_t Narr[] = {N};
  const f64 Larr[] = {L};
  matmul_ctx_t *ctx = matmul_ctx_alloc(1, Narr, Larr);
  matmul_ctx_set_system(ctx, 0);
  for (uint64_t i = 0; i < ctx->size; i++) {
    ctx->localTermK[i] = V0 - mu;
    ctx->localTermM[i] = V0 - mu;
  }
  return ctx;
}

/* Test 1: K with no deflation vectors behaves identically to before */
TEST(matmulK_no_deflation_unchanged) {
  const uint64_t N = 16;
  matmul_ctx_t *ctx = make_1d_ctx(N, 2.0 * M_PI, 1.0, 0.0);

  /* n_goldK = 0 from xcalloc, so no shift should be applied */
  f64 *x = xcalloc(N, sizeof(f64));
  f64 *y = xcalloc(N, sizeof(f64));
  x[1] = 1.0;

  d_matmulK(ctx, x, y);
  /* y should be kinetic + localTermK*x, same as before */
  /* Just check it doesn't crash and y[1] is nonzero */
  ASSERT(fabs(y[1]) > 0.0);

  safe_free((void **)&x);
  safe_free((void **)&y);
  matmul_ctx_free(&ctx);
}

/* Test 2: K-deflation shifts a known null vector
 *
 * Setup: localTermK = 0 (so K = -0.5*nabla^2). The k=0 mode
 * (constant vector) has eigenvalue 0. Set gold_vecK = const vector,
 * then K_bar * const = xi * const. */
TEST(matmulK_deflation_shifts_null_vector) {
  const uint64_t N = 16;
  matmul_ctx_t *ctx = make_1d_ctx(N, 2.0 * M_PI, 0.0, 0.0);

  /* Deflation vector: constant vector (in null space of Laplacian) */
  const f64 xi = 500.0;
  ctx->gold_xi = xi;
  ctx->n_goldK = 1;
  ctx->gold_vecK = xcalloc(N, sizeof(f64));
  ctx->gold_inv_normK = xcalloc(1, sizeof(f64));

  f64 *v0 = (f64 *)ctx->gold_vecK;
  for (uint64_t i = 0; i < N; i++)
    v0[i] = 1.0;
  const f64 v0_norm2 = d_dot(N, v0, v0);
  ctx->gold_inv_normK[0] = 1.0 / v0_norm2;

  /* Apply K_bar to the constant vector */
  f64 *x = xcalloc(N, sizeof(f64));
  f64 *y = xcalloc(N, sizeof(f64));
  for (uint64_t i = 0; i < N; i++)
    x[i] = 1.0;

  d_matmulK(ctx, x, y);

  /* K*x = 0 (constant is in null space of Laplacian with ltK=0)
   * shift adds xi * (v0^T x / ||v0||^2) * v0 = xi * x
   * so y should be xi * x */
  for (uint64_t i = 0; i < N; i++)
    ASSERT_CLOSE(y[i], xi, 1e-6);

  safe_free((void **)&x);
  safe_free((void **)&y);
  matmul_ctx_free(&ctx);
}

/* Test 3: K-deflation preserves non-null eigenvectors
 *
 * The k=1 mode (sin(x)) has kinetic eigenvalue 0.5*k^2 = 0.5.
 * It is orthogonal to the constant deflection vector, so the
 * shift adds 0. K_bar * sin = K * sin = 0.5 * sin. */
TEST(matmulK_deflation_preserves_orthogonal) {
  const uint64_t N = 32;
  const f64 L = 2.0 * M_PI;
  matmul_ctx_t *ctx = make_1d_ctx(N, L, 0.0, 0.0);

  /* Deflation with constant vector */
  ctx->gold_xi = 1000.0;
  ctx->n_goldK = 1;
  ctx->gold_vecK = xcalloc(N, sizeof(f64));
  ctx->gold_inv_normK = xcalloc(1, sizeof(f64));

  f64 *v0 = (f64 *)ctx->gold_vecK;
  for (uint64_t i = 0; i < N; i++)
    v0[i] = 1.0;
  ctx->gold_inv_normK[0] = 1.0 / d_dot(N, v0, v0);

  /* x = sin(2*pi*x/L), which is orthogonal to constant */
  f64 *x = xcalloc(N, sizeof(f64));
  f64 *y = xcalloc(N, sizeof(f64));
  for (uint64_t i = 0; i < N; i++)
    x[i] = sin(2.0 * M_PI * (f64)i / (f64)N);

  d_matmulK(ctx, x, y);

  /* y should be 0.5 * (2*pi/L)^2 * x = 0.5 * 1^2 * x = 0.5*x */
  for (uint64_t i = 0; i < N; i++)
    ASSERT_CLOSE(y[i], 0.5 * x[i], 1e-6);

  safe_free((void **)&x);
  safe_free((void **)&y);
  matmul_ctx_free(&ctx);
}

/* Test 4: M-deflation shift
 * Same structure as K test but using matmulM. */
TEST(matmulM_deflation_shifts_null_vector) {
  const uint64_t N = 16;
  matmul_ctx_t *ctx = make_1d_ctx(N, 2.0 * M_PI, 0.0, 0.0);

  const f64 xi = 300.0;
  ctx->gold_xi = xi;
  ctx->n_goldM = 1;
  ctx->gold_vecM = xcalloc(N, sizeof(f64));
  ctx->gold_inv_normM = xcalloc(1, sizeof(f64));

  f64 *w = (f64 *)ctx->gold_vecM;
  for (uint64_t i = 0; i < N; i++)
    w[i] = 1.0;
  ctx->gold_inv_normM[0] = 1.0 / d_dot(N, w, w);

  f64 *x = xcalloc(N, sizeof(f64));
  f64 *y = xcalloc(N, sizeof(f64));
  for (uint64_t i = 0; i < N; i++)
    x[i] = 1.0;

  d_matmulM(ctx, x, y);

  for (uint64_t i = 0; i < N; i++)
    ASSERT_CLOSE(y[i], xi, 1e-6);

  safe_free((void **)&x);
  safe_free((void **)&y);
  matmul_ctx_free(&ctx);
}

/* ================================================================
 * Test: CG solves a simple SPD system
 *
 * M = -0.5*nabla^2 + V where V=2.0 (constant). This is SPD.
 * b = constant vector. Solution: x = b / V (since kinetic(const) = 0).
 * ================================================================ */
TEST(cg_solve_simple_spd) {
  const uint64_t N = 32;
  matmul_ctx_t *ctx = make_1d_ctx(N, 2.0 * M_PI, 2.0, 0.0);

  /* b = 1.0 everywhere */
  f64 *b = xcalloc(N, sizeof(f64));
  f64 *x = xcalloc(N, sizeof(f64));
  for (uint64_t i = 0; i < N; i++)
    b[i] = 1.0;

  /* Solve M*x = b where M = kinetic + 2.0*I */
  const int ret = d_cg_solve(ctx, d_matmulM, NULL, b, x, N, 1e-10, 200);
  ASSERT(0 == ret);

  /* The constant vector is in null(kinetic), so M*const = 2.0*const
   * => x = b/2.0 = 0.5 */
  for (uint64_t i = 0; i < N; i++)
    ASSERT_CLOSE(x[i], 0.5, 1e-6);

  safe_free((void **)&b);
  safe_free((void **)&x);
  matmul_ctx_free(&ctx);
}

/* ================================================================
 * Test: CG with preconditioner converges (correctness check)
 * ================================================================ */
TEST(cg_solve_with_precond) {
  const uint64_t N = 32;
  const f64 mu = 1.0;
  const f64 V0 = 3.0;
  matmul_ctx_t *ctx = make_1d_ctx(N, 2.0 * M_PI, V0, mu);

  /* Build preconditioner arrays (normally done by bdg_set_mu) */
  ctx->mu = mu;
  ctx->precond_sqrtK = xcalloc(N, sizeof(f64));
  ctx->precond_sqrtM = xcalloc(N, sizeof(f64));
  for (uint64_t i = 0; i < N; i++) {
    ctx->precond_sqrtK[i] = 1.0 / sqrt(fmax(1e-8, V0));
    ctx->precond_sqrtM[i] = 1.0 / sqrt(fmax(1e-8, V0));
  }

  /* b = sin(2*pi*x/L) */
  f64 *b = xcalloc(N, sizeof(f64));
  f64 *x = xcalloc(N, sizeof(f64));
  for (uint64_t i = 0; i < N; i++)
    b[i] = sin(2.0 * M_PI * (f64)i / (f64)N);

  const int ret = d_cg_solve(ctx, d_matmulM, d_precondM, b, x, N, 1e-10, 200);
  ASSERT(0 == ret);

  /* Verify: M*x should equal b */
  f64 *Mx = xcalloc(N, sizeof(f64));
  d_matmulM(ctx, x, Mx);
  for (uint64_t i = 0; i < N; i++)
    ASSERT_CLOSE(Mx[i], b[i], 1e-6);

  safe_free((void **)&b);
  safe_free((void **)&x);
  safe_free((void **)&Mx);
  matmul_ctx_free(&ctx);
}

/* ================================================================
 * Test: bdg_deflate_u1 on 1D uniform BEC
 *
 * Uniform BEC: psi0 = sqrt(n0), V_trap = 0, contact g.
 * K*psi0 = 0 guaranteed by GPE.
 * After deflation: K_bar*psi0 should have large eigenvalue.
 * ================================================================ */
static f64 U_contact_K(const void *param, f64 density) {
  const f64 g = *(const f64 *)param;
  return g * density;
}

static f64 U_contact_M(const void *param, f64 density) {
  return U_contact_K(param, density);
}

TEST(bdg_deflate_u1_uniform_bec) {
  const uint64_t N = 64;
  const f64 L = 10.0;
  const f64 g = 1.0;
  const f64 n0 = 1.0;
  const f64 psi0_val = sqrt(n0);
  const f64 mu = g * n0;  /* GPE: mu = g * |psi0|^2 for uniform */

  bdg_t *bdg = bdg_alloc(1, &N, &L, 0);
  bdg_set_system(bdg);
  bdg_set_trap(bdg, NULL, NULL);

  /* Uniform wavefunction */
  f64 *wf = xcalloc(N, sizeof(f64));
  for (uint64_t i = 0; i < N; i++)
    wf[i] = psi0_val;
  bdg_set_wavefunction(bdg, wf, N);
  safe_free((void **)&wf);

  bdg_set_local_interactions(bdg, U_contact_K, U_contact_M, &g);
  bdg_set_mu(bdg, mu);

  /* Deflate U(1) */
  const int ndefl = bdg_deflate_u1(bdg, 1e-6);
  ASSERT(1 == ndefl);

  /* Verify: n_goldK = 1 and gold_vecK is non-NULL */
  matmul_ctx_t *ctx = bdg->ctx;
  ASSERT(1 == ctx->n_goldK);
  ASSERT(NULL != ctx->gold_vecK);
  ASSERT(NULL != ctx->gold_inv_normK);

  /* Apply K_bar to psi0 and check it's NOT near zero anymore */
  f64 *x = xcalloc(N, sizeof(f64));
  f64 *y = xcalloc(N, sizeof(f64));
  for (uint64_t i = 0; i < N; i++)
    x[i] = psi0_val;
  d_matmulK(ctx, x, y);
  const f64 y_norm = d_nrm2(N, y);
  const f64 x_norm = d_nrm2(N, x);
  /* With deflation, ||K_bar * psi0|| / ||psi0|| should be >> 1 */
  ASSERT(y_norm / x_norm > 10.0);

  safe_free((void **)&x);
  safe_free((void **)&y);
  bdg_free(&bdg);
}

/* ================================================================
 * Test: bdg_deflate_u1 returns 0 if K*psi0 is not near zero
 * (e.g., psi0 is NOT a GPE ground state)
 * ================================================================ */
TEST(bdg_deflate_u1_not_ground_state) {
  const uint64_t N = 32;
  const f64 L = 10.0;
  const f64 g = 1.0;

  bdg_t *bdg = bdg_alloc(1, &N, &L, 0);
  bdg_set_system(bdg);
  bdg_set_trap(bdg, NULL, NULL);

  /* Garbage wavefunction — NOT a ground state */
  f64 *wf = xcalloc(N, sizeof(f64));
  for (uint64_t i = 0; i < N; i++)
    wf[i] = sin(2.0 * M_PI * (f64)i / (f64)N) + 0.5;
  bdg_set_wavefunction(bdg, wf, N);
  safe_free((void **)&wf);

  bdg_set_local_interactions(bdg, U_contact_K, U_contact_M, &g);
  bdg_set_mu(bdg, 0.5);

  /* Should fail the residual check */
  const int ndefl = bdg_deflate_u1(bdg, 1e-6);
  ASSERT(0 == ndefl);
  ASSERT(0 == bdg->ctx->n_goldK);

  bdg_free(&bdg);
}

/* ================================================================
 * Test: bdg_deflate_auto finds no null vectors for a standard
 * (non-supersolid) 1D BEC where M > 0.
 * ================================================================ */
TEST(bdg_deflate_auto_no_null_vectors) {
  const uint64_t N = 64;
  const f64 L = 10.0;
  const f64 g = 1.0;
  const f64 n0 = 1.0;
  const f64 psi0_val = sqrt(n0);
  const f64 mu = g * n0;

  bdg_t *bdg = bdg_alloc(1, &N, &L, 0);
  bdg_set_system(bdg);
  bdg_set_trap(bdg, NULL, NULL);

  f64 *wf = xcalloc(N, sizeof(f64));
  for (uint64_t i = 0; i < N; i++)
    wf[i] = psi0_val;
  bdg_set_wavefunction(bdg, wf, N);
  safe_free((void **)&wf);

  bdg_set_local_interactions(bdg, U_contact_K, U_contact_M, &g);
  bdg_set_mu(bdg, mu);
  bdg_set_solver_params(bdg, 3, 6, 100, 1e-6);

  /* First deflate U(1) so K_bar > 0 */
  bdg_deflate_u1(bdg, 1e-6);

  /* Auto-detect: M > 0 for standard BEC, so nothing to deflate */
  const int n_auto = bdg_deflate_auto(bdg, 2, 1e-6);
  ASSERT(0 == n_auto);
  ASSERT(0 == bdg->ctx->n_goldM);

  bdg_free(&bdg);
}

/* ================================================================
 * Integration: 1D uniform BEC with deflation
 *
 * Without deflation, the lowest eigenvalue is the (near-zero)
 * Goldstone mode. With deflation, the lowest eigenvalue should be
 * the first physical Bogoliubov mode: omega = k*sqrt(k^2/2 + g*n).
 * For k = 2*pi/L (smallest nonzero k), g=1, n=1, L=20:
 *   k1 = 2*pi/20 = pi/10
 *   omega1 = k1 * sqrt(k1^2/2 + 1) = pi/10 * sqrt(pi^2/200 + 1)
 * ================================================================ */
TEST(integration_1d_uniform_with_deflation) {
  const uint64_t N = 128;
  const f64 L = 20.0;
  const f64 g = 1.0;
  const f64 n0 = 1.0;
  const f64 psi0_val = sqrt(n0);
  const f64 mu = g * n0;

  bdg_t *bdg = bdg_alloc(1, &N, &L, 0);
  bdg_set_system(bdg);
  bdg_set_trap(bdg, NULL, NULL);

  f64 *wf = xcalloc(N, sizeof(f64));
  for (uint64_t i = 0; i < N; i++)
    wf[i] = psi0_val;
  bdg_set_wavefunction(bdg, wf, N);
  safe_free((void **)&wf);

  bdg_set_local_interactions(bdg, U_contact_K, U_contact_M, &g);
  bdg_set_mu(bdg, mu);

  /* Deflate */
  const int ndefl = bdg_deflate_u1(bdg, 1e-6);
  ASSERT(1 == ndefl);

  /* Solve */
  bdg_set_solver_params(bdg, 3, 8, 300, 1e-8);
  const int ret = bdg_solve(bdg);
  ASSERT(0 == ret);

  const f64 *eigs = bdg_eigenvalues(bdg);

  /* Bogoliubov dispersion: omega = k * sqrt(k^2/2 + g*n) */
  const f64 k1 = 2.0 * M_PI / L;
  const f64 omega1 = k1 * sqrt(0.5 * k1 * k1 + g * n0);

  /* First eigenvalue should be the physical mode, not near-zero */
  printf("  eigenvalues: %.6f %.6f %.6f\n", eigs[0], eigs[1], eigs[2]);
  printf("  expected omega1 = %.6f\n", omega1);

  /* The lowest eigenvalue should match omega1 (doubly degenerate: sin/cos) */
  ASSERT_CLOSE(eigs[0], omega1, 1e-3);
  ASSERT_CLOSE(eigs[1], omega1, 1e-3);

  bdg_free(&bdg);
}

int main(void) {
  printf("test_goldstone:\n");
  RUN(matmulK_no_deflation_unchanged);
  RUN(matmulK_deflation_shifts_null_vector);
  RUN(matmulK_deflation_preserves_orthogonal);
  RUN(matmulM_deflation_shifts_null_vector);
  RUN(cg_solve_simple_spd);
  RUN(cg_solve_with_precond);
  RUN(bdg_deflate_u1_uniform_bec);
  RUN(bdg_deflate_u1_not_ground_state);
  RUN(bdg_deflate_auto_no_null_vectors);
  RUN(integration_1d_uniform_with_deflation);
  printf("\n  %d passed, %d failed\n", tests_passed, tests_failed);
  return tests_failed;
}
