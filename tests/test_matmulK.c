#include "bdg_internal.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

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
 * Helper: create a 1D matmul_ctx_t with constant localTermK = V0 - mu
 * ================================================================ */
static matmul_ctx_t *make_1d_ctx(const uint64_t N, const f64 Lval,
                                 const int complex_psi0,
                                 const f64 V0, const f64 mu) {
    const uint64_t Narr[] = {N};
    const f64    Larr[] = {Lval};
    matmul_ctx_t *ctx = matmul_ctx_alloc(1, Narr, Larr);
    matmul_ctx_set_system(ctx, complex_psi0);

    const uint64_t size = ctx->size;
    ctx->localTermK = xcalloc(size, sizeof(f64));
    ctx->localTermM = xcalloc(size, sizeof(f64));

    const f64 val = V0 - mu;
    for (uint64_t i = 0; i < size; i++) {
        ctx->localTermK[i] = val;
        ctx->localTermM[i] = val;
    }

    return ctx;
}

/* ================================================================
 * Test 1: 1D cos(x) with V0=2, mu=0
 * K(cos(x)) = (0.5*1 + 2)*cos(x) = 2.5*cos(x)
 * ================================================================ */
TEST(matmulK_d_planewave_1d) {
    const f64 V0 = 2.0;
    const f64 mu = 0.0;
    matmul_ctx_t *ctx = make_1d_ctx(64, 2.0 * M_PI, 0, V0, mu);

    const uint64_t size = ctx->size;
    f64 *x = xcalloc(size, sizeof(f64));
    f64 *y = xcalloc(size, sizeof(f64));

    const f64 L = 2.0 * M_PI;
    for (uint64_t i = 0; i < size; i++) {
        const f64 xj = (f64)i * L / (f64)size;
        x[i] = cos(xj);
    }

    matmulK_d(ctx, x, y);

    for (uint64_t i = 0; i < size; i++) {
        const f64 xj = (f64)i * L / (f64)size;
        ASSERT_CLOSE(y[i], 2.5 * cos(xj), TOL);
    }

    safe_free((void **)&x);
    safe_free((void **)&y);
    matmul_ctx_free(&ctx);
}

/* ================================================================
 * Test 2: Constant input with V0=3, mu=1
 * K(1) = kinetic(1) + (V0 - mu)*1 = 0 + 2 = 2
 * ================================================================ */
TEST(matmulK_d_constant) {
    const f64 V0 = 3.0;
    const f64 mu = 1.0;
    matmul_ctx_t *ctx = make_1d_ctx(32, 2.0 * M_PI, 0, V0, mu);

    const uint64_t size = ctx->size;
    f64 *x = xcalloc(size, sizeof(f64));
    f64 *y = xcalloc(size, sizeof(f64));

    for (uint64_t i = 0; i < size; i++)
        x[i] = 1.0;

    matmulK_d(ctx, x, y);

    for (uint64_t i = 0; i < size; i++)
        ASSERT_CLOSE(y[i], 2.0, TOL);

    safe_free((void **)&x);
    safe_free((void **)&y);
    matmul_ctx_free(&ctx);
}

/* ================================================================
 * Test 3: d and z paths on same real input must match
 * ================================================================ */
TEST(matmulK_dz_consistency) {
    const f64 V0 = 2.0;
    const f64 mu = 0.5;
    const uint64_t N = 64;
    const f64 L = 2.0 * M_PI;

    /* Real path */
    matmul_ctx_t *ctx_d = make_1d_ctx(N, L, 0, V0, mu);
    f64 *xd = xcalloc(N, sizeof(f64));
    f64 *yd = xcalloc(N, sizeof(f64));

    for (uint64_t i = 0; i < N; i++) {
        const f64 xj = (f64)i * L / (f64)N;
        xd[i] = cos(xj);
    }
    matmulK_d(ctx_d, xd, yd);

    /* Complex path */
    matmul_ctx_t *ctx_z = make_1d_ctx(N, L, 1, V0, mu);
    c64 *xz = xcalloc(N, sizeof(c64));
    c64 *yz = xcalloc(N, sizeof(c64));

    for (uint64_t i = 0; i < N; i++) {
        const f64 xj = (f64)i * L / (f64)N;
        xz[i] = cos(xj) + 0.0 * I;
    }
    matmulK_z(ctx_z, xz, yz);

    /* Compare: real parts must match, imag parts ~ 0 */
    for (uint64_t i = 0; i < N; i++) {
        ASSERT_CLOSE(creal(yz[i]), yd[i], TOL);
        ASSERT_CLOSE(cimag(yz[i]), 0.0, TOL);
    }

    safe_free((void **)&xd);
    safe_free((void **)&yd);
    safe_free((void **)&xz);
    safe_free((void **)&yz);
    matmul_ctx_free(&ctx_d);
    matmul_ctx_free(&ctx_z);
}

/* ================================================================
 * Test 4: 3D cos(x+y+z) with V=1, mu=0
 * k=(1,1,1), k^2=3, kinetic = 0.5*3 = 1.5
 * K = (1.5 + 1)*cos(x+y+z) = 2.5*cos(x+y+z)
 * ================================================================ */
TEST(matmulK_d_3d) {
    const uint64_t N[] = {16, 16, 16};
    const f64    L[] = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};
    matmul_ctx_t *ctx = matmul_ctx_alloc(3, N, L);
    matmul_ctx_set_system(ctx, 0);

    const uint64_t size = ctx->size;
    const uint64_t Nx = N[0], Ny = N[1], Nz = N[2];
    const f64 V0 = 1.0;

    ctx->localTermK = xcalloc(size, sizeof(f64));
    ctx->localTermM = xcalloc(size, sizeof(f64));
    for (uint64_t i = 0; i < size; i++) {
        ctx->localTermK[i] = V0;
        ctx->localTermM[i] = V0;
    }

    f64 *x = xcalloc(size, sizeof(f64));
    f64 *y = xcalloc(size, sizeof(f64));

    for (uint64_t iz = 0; iz < Nz; iz++)
        for (uint64_t iy = 0; iy < Ny; iy++)
            for (uint64_t ix = 0; ix < Nx; ix++) {
                const f64 xv = (f64)ix * L[0] / (f64)Nx;
                const f64 yv = (f64)iy * L[1] / (f64)Ny;
                const f64 zv = (f64)iz * L[2] / (f64)Nz;
                x[iz * Ny * Nx + iy * Nx + ix] = cos(xv + yv + zv);
            }

    matmulK_d(ctx, x, y);

    for (uint64_t iz = 0; iz < Nz; iz++)
        for (uint64_t iy = 0; iy < Ny; iy++)
            for (uint64_t ix = 0; ix < Nx; ix++) {
                const f64 xv = (f64)ix * L[0] / (f64)Nx;
                const f64 yv = (f64)iy * L[1] / (f64)Ny;
                const f64 zv = (f64)iz * L[2] / (f64)Nz;
                const uint64_t idx = iz * Ny * Nx + iy * Nx + ix;
                ASSERT_CLOSE(y[idx], 2.5 * cos(xv + yv + zv), TOL);
            }

    safe_free((void **)&x);
    safe_free((void **)&y);
    matmul_ctx_free(&ctx);
}

/* ================================================================ */
int main(void) {
    printf("matmulK operator (real path):\n");
    RUN(matmulK_d_planewave_1d);
    RUN(matmulK_d_constant);

    printf("\nmatmulK operator (d/z consistency):\n");
    RUN(matmulK_dz_consistency);

    printf("\nmatmulK operator (3D):\n");
    RUN(matmulK_d_3d);

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
