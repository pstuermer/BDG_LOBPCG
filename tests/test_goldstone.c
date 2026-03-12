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

int main(void) {
  printf("test_goldstone:\n");
  RUN(matmulK_no_deflation_unchanged);
  RUN(matmulK_deflation_shifts_null_vector);
  RUN(matmulK_deflation_preserves_orthogonal);
  RUN(matmulM_deflation_shifts_null_vector);
  printf("\n  %d passed, %d failed\n", tests_passed, tests_failed);
  return tests_failed;
}
