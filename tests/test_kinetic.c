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
 * Test 1: Single planewave cos(x), expect 0.5*cos(x)
 * k=1 → 0.5*k²=0.5
 * ================================================================ */
TEST(kinetic_d_planewave_1d) {
    const uint64_t N[] = {64};
    const f64    L[] = {2.0 * M_PI};
    matmul_ctx_t *ctx = matmul_ctx_alloc(1, N, L);
    matmul_ctx_set_system(ctx, 0);

    const uint64_t size = ctx->size;
    f64 *x = xcalloc(size, sizeof(f64));
    f64 *y = xcalloc(size, sizeof(f64));

    for (uint64_t i = 0; i < size; i++) {
        const f64 xj = (f64)i * L[0] / (f64)size;
        x[i] = cos(xj);
    }

    kinetic_d(ctx, x, y);

    for (uint64_t i = 0; i < size; i++) {
        const f64 xj = (f64)i * L[0] / (f64)size;
        ASSERT_CLOSE(y[i], 0.5 * cos(xj), TOL);
    }

    safe_free((void **)&x);
    safe_free((void **)&y);
    matmul_ctx_free(&ctx);
}

/* ================================================================
 * Test 2: Superposition cos(x) + 0.5*cos(3x)
 * → 0.5*cos(x) + 0.5*9*0.5*cos(3x) = 0.5*cos(x) + 2.25*cos(3x)
 * ================================================================ */
TEST(kinetic_d_superposition_1d) {
    const uint64_t N[] = {64};
    const f64    L[] = {2.0 * M_PI};
    matmul_ctx_t *ctx = matmul_ctx_alloc(1, N, L);
    matmul_ctx_set_system(ctx, 0);

    const uint64_t size = ctx->size;
    f64 *x = xcalloc(size, sizeof(f64));
    f64 *y = xcalloc(size, sizeof(f64));

    for (uint64_t i = 0; i < size; i++) {
        const f64 xj = (f64)i * L[0] / (f64)size;
        x[i] = cos(xj) + 0.5 * cos(3.0 * xj);
    }

    kinetic_d(ctx, x, y);

    for (uint64_t i = 0; i < size; i++) {
        const f64 xj = (f64)i * L[0] / (f64)size;
        const f64 expected = 0.5 * cos(xj) + 2.25 * cos(3.0 * xj);
        ASSERT_CLOSE(y[i], expected, TOL);
    }

    safe_free((void **)&x);
    safe_free((void **)&y);
    matmul_ctx_free(&ctx);
}

/* ================================================================
 * Test 3: Complex path on same real input — d and z must match
 * ================================================================ */
TEST(kinetic_z_planewave_1d) {
    const uint64_t N[] = {64};
    const f64    L[] = {2.0 * M_PI};
    const uint64_t size = 64;

    /* Real path */
    matmul_ctx_t *ctx_d = matmul_ctx_alloc(1, N, L);
    matmul_ctx_set_system(ctx_d, 0);
    f64 *xd = xcalloc(size, sizeof(f64));
    f64 *yd = xcalloc(size, sizeof(f64));

    for (uint64_t i = 0; i < size; i++) {
        const f64 xj = (f64)i * L[0] / (f64)size;
        xd[i] = cos(xj);
    }
    kinetic_d(ctx_d, xd, yd);

    /* Complex path */
    matmul_ctx_t *ctx_z = matmul_ctx_alloc(1, N, L);
    matmul_ctx_set_system(ctx_z, 1);
    c64 *xz = xcalloc(size, sizeof(c64));
    c64 *yz = xcalloc(size, sizeof(c64));

    for (uint64_t i = 0; i < size; i++) {
        const f64 xj = (f64)i * L[0] / (f64)size;
        xz[i] = cos(xj) + 0.0 * I;
    }
    kinetic_z(ctx_z, xz, yz);

    /* Compare: real parts must match, imag parts ≈ 0 */
    for (uint64_t i = 0; i < size; i++) {
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
 * Test 4: 3D planewave cos(x+y+z), k=(1,1,1), k²=3 → y=1.5*cos
 * ================================================================ */
TEST(kinetic_d_planewave_3d) {
    const uint64_t N[] = {16, 16, 16};
    const f64    L[] = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};
    matmul_ctx_t *ctx = matmul_ctx_alloc(3, N, L);
    matmul_ctx_set_system(ctx, 0);

    const uint64_t Nx = N[0], Ny = N[1], Nz = N[2];
    const uint64_t size = ctx->size;
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

    kinetic_d(ctx, x, y);

    for (uint64_t iz = 0; iz < Nz; iz++)
        for (uint64_t iy = 0; iy < Ny; iy++)
            for (uint64_t ix = 0; ix < Nx; ix++) {
                const f64 xv = (f64)ix * L[0] / (f64)Nx;
                const f64 yv = (f64)iy * L[1] / (f64)Ny;
                const f64 zv = (f64)iz * L[2] / (f64)Nz;
                const uint64_t idx = iz * Ny * Nx + iy * Nx + ix;
                ASSERT_CLOSE(y[idx], 1.5 * cos(xv + yv + zv), TOL);
            }

    safe_free((void **)&x);
    safe_free((void **)&y);
    matmul_ctx_free(&ctx);
}

/* ================================================================
 * Test 5: Zero mode (constant), k=0 → y ≈ 0
 * ================================================================ */
TEST(kinetic_d_zero_mode) {
    const uint64_t N[] = {32};
    const f64    L[] = {2.0 * M_PI};
    matmul_ctx_t *ctx = matmul_ctx_alloc(1, N, L);
    matmul_ctx_set_system(ctx, 0);

    const uint64_t size = ctx->size;
    f64 *x = xcalloc(size, sizeof(f64));
    f64 *y = xcalloc(size, sizeof(f64));

    for (uint64_t i = 0; i < size; i++)
        x[i] = 1.0;

    kinetic_d(ctx, x, y);

    for (uint64_t i = 0; i < size; i++)
        ASSERT_CLOSE(y[i], 0.0, TOL);

    safe_free((void **)&x);
    safe_free((void **)&y);
    matmul_ctx_free(&ctx);
}

/* ================================================================ */
int main(void) {
    printf("Kinetic operator (real path):\n");
    RUN(kinetic_d_planewave_1d);
    RUN(kinetic_d_superposition_1d);

    printf("\nKinetic operator (complex path):\n");
    RUN(kinetic_z_planewave_1d);

    printf("\nKinetic operator (3D):\n");
    RUN(kinetic_d_planewave_3d);

    printf("\nKinetic operator (edge cases):\n");
    RUN(kinetic_d_zero_mode);

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
