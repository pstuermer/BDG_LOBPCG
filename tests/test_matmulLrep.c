#include "bdg_internal.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define TOL 1e-10

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
 * Helper: create a 1D ctx with distinct localTermK and localTermM
 * ================================================================ */
static matmul_ctx_t *make_1d_ctx(const uint64_t N, const f64 L,
                                  const int complex_psi0,
                                  const f64 VK, const f64 VM) {
    const uint64_t Narr[] = {N};
    const f64 Larr[] = {L};
    matmul_ctx_t *ctx = matmul_ctx_alloc(1, Narr, Larr);
    matmul_ctx_set_system(ctx, complex_psi0);

    ctx->localTermK = xcalloc(ctx->size, sizeof(f64));
    ctx->localTermM = xcalloc(ctx->size, sizeof(f64));
    for (uint64_t i = 0; i < ctx->size; i++) {
        ctx->localTermK[i] = VK;
        ctx->localTermM[i] = VM;
    }
    ctx->dipolar = 0;
    return ctx;
}

/* ================================================================
 * Test 1: Block structure — Lrep([u;v]) == [K(u); M(v)]
 *
 * Use u = cos(x) (planewave), v = constant, VK != VM.
 * Apply K and M separately as reference, then compare with Lrep.
 * ================================================================ */
TEST(matmulLrep_d_block_structure) {
    const uint64_t N = 64;
    const f64 L = 2.0 * M_PI;
    const f64 VK = 2.0;
    const f64 VM = 3.5;
    matmul_ctx_t *ctx = make_1d_ctx(N, L, 0, VK, VM);
    const uint64_t size = ctx->size;

    /* Build stacked input: u = cos(x), v = 1 */
    f64 *x_stacked = xcalloc(2 * size, sizeof(f64));
    for (uint64_t i = 0; i < size; i++) {
        x_stacked[i] = cos((f64)i * L / (f64)size);
        x_stacked[size + i] = 1.0;
    }

    /* Apply Lrep to stacked vector */
    f64 *y_stacked = xcalloc(2 * size, sizeof(f64));
    matmulLrep_d(ctx, x_stacked, y_stacked);

    /* Reference: apply K and M separately */
    f64 *yK_ref = xcalloc(size, sizeof(f64));
    f64 *yM_ref = xcalloc(size, sizeof(f64));
    matmulK_d(ctx, x_stacked, yK_ref);
    matmulM_d(ctx, &x_stacked[size], yM_ref);

    /* Compare */
    for (uint64_t i = 0; i < size; i++) {
        ASSERT_CLOSE(y_stacked[i], yK_ref[i], TOL);
        ASSERT_CLOSE(y_stacked[size + i], yM_ref[i], TOL);
    }

    safe_free((void **)&x_stacked);
    safe_free((void **)&y_stacked);
    safe_free((void **)&yK_ref);
    safe_free((void **)&yM_ref);
    matmul_ctx_free(&ctx);
}

/* ================================================================
 * Test 2: d and z paths agree on the same real input
 * ================================================================ */
TEST(matmulLrep_d_equals_z) {
    const uint64_t N = 64;
    const f64 L = 2.0 * M_PI;
    const f64 VK = 1.0;
    const f64 VM = 2.5;

    matmul_ctx_t *ctx_d = make_1d_ctx(N, L, 0, VK, VM);
    matmul_ctx_t *ctx_z = make_1d_ctx(N, L, 1, VK, VM);
    const uint64_t size = ctx_d->size;

    /* Real stacked input: u = cos(x), v = sin(2x) */
    f64 *xd = xcalloc(2 * size, sizeof(f64));
    c64 *xz = xcalloc(2 * size, sizeof(c64));
    for (uint64_t i = 0; i < size; i++) {
        const f64 xi = (f64)i * L / (f64)size;
        xd[i] = cos(xi);
        xd[size + i] = sin(2.0 * xi);
        xz[i] = cos(xi) + 0.0 * I;
        xz[size + i] = sin(2.0 * xi) + 0.0 * I;
    }

    f64 *yd = xcalloc(2 * size, sizeof(f64));
    c64 *yz = xcalloc(2 * size, sizeof(c64));

    matmulLrep_d(ctx_d, xd, yd);
    matmulLrep_z(ctx_z, xz, yz);

    for (uint64_t i = 0; i < 2 * size; i++) {
        ASSERT_CLOSE(creal(yz[i]), yd[i], TOL);
        ASSERT_CLOSE(cimag(yz[i]), 0.0, TOL);
    }

    safe_free((void **)&xd);
    safe_free((void **)&xz);
    safe_free((void **)&yd);
    safe_free((void **)&yz);
    matmul_ctx_free(&ctx_d);
    matmul_ctx_free(&ctx_z);
}

/* ================================================================ */
int main(void) {
    printf("matmulLrep operator:\n");
    RUN(matmulLrep_d_block_structure);
    RUN(matmulLrep_d_equals_z);

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");
    return tests_failed > 0 ? 1 : 0;
}
