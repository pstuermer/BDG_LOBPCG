#include "bdg_internal.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>

#define TOL 1e-15

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) static void test_##name(void)
#define RUN(name) do { \
    printf("  %-40s ", #name); \
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
        printf("[FAIL] line %d: %.15e vs %.15e\n", __LINE__, _a, _b); \
        tests_failed++; \
        return; \
    } \
} while(0)

/* ================================================================
 * Helper: minimal 1D ctx (only needs size for swap, no FFT plans)
 * ================================================================ */
static matmul_ctx_t *make_swap_ctx(const uint64_t N) {
    const uint64_t Narr[] = {N};
    const f64 Larr[] = {1.0};
    matmul_ctx_t *ctx = matmul_ctx_alloc(1, Narr, Larr);
    /* Don't call set_system — swap doesn't need FFT plans.
     * But size is set by alloc. */
    return ctx;
}

/* ================================================================
 * Test 1: x = [1,2,3 | 4,5,6], size=3 → y = [4,5,6 | 1,2,3]
 * ================================================================ */
TEST(matmulSwap_d_swaps_halves) {
    const uint64_t N = 3;
    matmul_ctx_t *ctx = make_swap_ctx(N);

    f64 x[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    f64 y[6] = {0};

    matmulSwap_d(ctx, x, y);

    /* Upper half of y should be lower half of x */
    ASSERT_CLOSE(y[0], 4.0, TOL);
    ASSERT_CLOSE(y[1], 5.0, TOL);
    ASSERT_CLOSE(y[2], 6.0, TOL);

    /* Lower half of y should be upper half of x */
    ASSERT_CLOSE(y[3], 1.0, TOL);
    ASSERT_CLOSE(y[4], 2.0, TOL);
    ASSERT_CLOSE(y[5], 3.0, TOL);

    matmul_ctx_free(&ctx);
}

/* ================================================================
 * Test 2: swap(swap(x)) == x (involution)
 * ================================================================ */
TEST(matmulSwap_d_involution) {
    const uint64_t N = 5;
    matmul_ctx_t *ctx = make_swap_ctx(N);

    f64 x[10], y1[10], y2[10];
    for (uint64_t i = 0; i < 2 * N; i++)
        x[i] = (f64)(i + 1) * 0.7;

    matmulSwap_d(ctx, x, y1);
    matmulSwap_d(ctx, y1, y2);

    for (uint64_t i = 0; i < 2 * N; i++)
        ASSERT_CLOSE(y2[i], x[i], TOL);

    matmul_ctx_free(&ctx);
}

/* ================================================================
 * Test 3: complex swap — same logic with c64 data
 * ================================================================ */
TEST(matmulSwap_z_swaps_halves) {
    const uint64_t N = 3;
    matmul_ctx_t *ctx = make_swap_ctx(N);

    c64 x[6] = {1.0+0.1*I, 2.0+0.2*I, 3.0+0.3*I,
                4.0+0.4*I, 5.0+0.5*I, 6.0+0.6*I};
    c64 y[6] = {0};

    matmulSwap_z(ctx, x, y);

    /* Upper half of y = lower half of x */
    ASSERT_CLOSE(creal(y[0]), 4.0, TOL);
    ASSERT_CLOSE(cimag(y[0]), 0.4, TOL);
    ASSERT_CLOSE(creal(y[1]), 5.0, TOL);
    ASSERT_CLOSE(cimag(y[1]), 0.5, TOL);
    ASSERT_CLOSE(creal(y[2]), 6.0, TOL);
    ASSERT_CLOSE(cimag(y[2]), 0.6, TOL);

    /* Lower half of y = upper half of x */
    ASSERT_CLOSE(creal(y[3]), 1.0, TOL);
    ASSERT_CLOSE(cimag(y[3]), 0.1, TOL);
    ASSERT_CLOSE(creal(y[4]), 2.0, TOL);
    ASSERT_CLOSE(cimag(y[4]), 0.2, TOL);
    ASSERT_CLOSE(creal(y[5]), 3.0, TOL);
    ASSERT_CLOSE(cimag(y[5]), 0.3, TOL);

    matmul_ctx_free(&ctx);
}

/* ================================================================ */
int main(void) {
    printf("matmulSwap operator:\n");
    RUN(matmulSwap_d_swaps_halves);
    RUN(matmulSwap_d_involution);
    RUN(matmulSwap_z_swaps_halves);

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");
    return tests_failed > 0 ? 1 : 0;
}
