#include "bdg_internal.h"
#include <math.h>
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
 * Test 1: k=0 mode must be zero (divergence removed)
 * ================================================================ */
TEST(kernel_k0_is_zero) {
    const size_t N[] = {8, 8, 8};
    const f64    L[] = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};
    const f64    dipole_dir[] = {0.0, 0.0, 1.0};
    const f64    Rc = L[0] / 2.0;  /* pi */

    matmul_ctx_t *ctx = matmul_ctx_alloc(3, N, L);
    matmul_ctx_set_system(ctx, 0);  /* r2c */

    dipolar_set_kernel(ctx, 1.0, dipole_dir, Rc);

    /* k=0 is at index 0 */
    ASSERT_CLOSE(ctx->longRngInt[0], 0.0, TOL);

    matmul_ctx_free(&ctx);
}

/* ================================================================
 * Test 2: Angular dependence — dipole along z
 *   k along z (iz=1): cos^2(theta)=1, kernel = 2*f_cutoff
 *   k along x (ix=1): cos^2(theta)=0, kernel = -1*f_cutoff
 *
 *   For L=2*pi, N=8: dk = 1
 *   Rc = pi, so kR = 1*pi = pi
 *   f_cutoff = 1 + 3*cos(pi)/pi^2 - 3*sin(pi)/pi^3
 *            = 1 - 3/pi^2 (since sin(pi)~0)
 *            ~ 0.6960...
 *
 *   r2c: N0k = 8/2+1 = 5
 *   k along z: idx = 1*8*5 + 0*5 + 0 = 40
 *   k along x: idx = 0*8*5 + 0*5 + 1 = 1
 * ================================================================ */
TEST(kernel_angular_dependence) {
    const size_t N[] = {8, 8, 8};
    const f64    L[] = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};
    const f64    dipole_dir[] = {0.0, 0.0, 1.0};
    const f64    Rc = M_PI;

    matmul_ctx_t *ctx = matmul_ctx_alloc(3, N, L);
    matmul_ctx_set_system(ctx, 0);  /* r2c */

    dipolar_set_kernel(ctx, 1.0, dipole_dir, Rc);

    const size_t N0k = N[0] / 2 + 1;  /* 5 */

    /* Analytical f_cutoff for kR = pi */
    const f64 kR = M_PI;
    const f64 kR2 = kR * kR;
    const f64 f_cutoff = 1.0 + 3.0 * cos(kR) / kR2
                             - 3.0 * sin(kR) / (kR2 * kR);

    /* k along z: iz=1,iy=0,ix=0 → cos^2(theta)=1 → (3*1-1)=2 */
    const size_t idx_kz = 1 * N[1] * N0k + 0 * N0k + 0;
    ASSERT_CLOSE(ctx->longRngInt[idx_kz], 2.0 * f_cutoff, TOL);

    /* k along x: iz=0,iy=0,ix=1 → cos^2(theta)=0 → (3*0-1)=-1 */
    const size_t idx_kx = 0 * N[1] * N0k + 0 * N0k + 1;
    ASSERT_CLOSE(ctx->longRngInt[idx_kx], -1.0 * f_cutoff, TOL);

    matmul_ctx_free(&ctx);
}

/* ================================================================
 * Test 3: r2c vs c2c — shared k-points must match exactly
 *   For ix <= N[0]/2, the r2c and c2c kernels at the same
 *   (ix,iy,iz) must be identical.
 * ================================================================ */
TEST(kernel_r2c_vs_c2c) {
    const size_t N[] = {8, 8, 8};
    const f64    L[] = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};
    const f64    dipole_dir[] = {0.0, 0.0, 1.0};
    const f64    Rc = M_PI;

    /* r2c path */
    matmul_ctx_t *ctx_r2c = matmul_ctx_alloc(3, N, L);
    matmul_ctx_set_system(ctx_r2c, 0);
    dipolar_set_kernel(ctx_r2c, 1.0, dipole_dir, Rc);

    /* c2c path */
    matmul_ctx_t *ctx_c2c = matmul_ctx_alloc(3, N, L);
    matmul_ctx_set_system(ctx_c2c, 1);
    dipolar_set_kernel(ctx_c2c, 1.0, dipole_dir, Rc);

    const size_t Nx = N[0], Ny = N[1], Nz = N[2];
    const size_t N0k_r2c = Nx / 2 + 1;  /* 5 */
    const size_t N0k_c2c = Nx;           /* 8 */

    for (size_t iz = 0; iz < Nz; iz++) {
        for (size_t iy = 0; iy < Ny; iy++) {
            for (size_t ix = 0; ix <= Nx / 2; ix++) {
                const size_t idx_r2c = iz * Ny * N0k_r2c + iy * N0k_r2c + ix;
                const size_t idx_c2c = iz * Ny * N0k_c2c + iy * N0k_c2c + ix;
                ASSERT_CLOSE(ctx_r2c->longRngInt[idx_r2c],
                             ctx_c2c->longRngInt[idx_c2c], TOL);
            }
        }
    }

    matmul_ctx_free(&ctx_r2c);
    matmul_ctx_free(&ctx_c2c);
}

/* ================================================================ */
int main(void) {
    printf("Dipolar kernel (k=0 edge case):\n");
    RUN(kernel_k0_is_zero);

    printf("\nDipolar kernel (angular dependence):\n");
    RUN(kernel_angular_dependence);

    printf("\nDipolar kernel (r2c vs c2c):\n");
    RUN(kernel_r2c_vs_c2c);

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
