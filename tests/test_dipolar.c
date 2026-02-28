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

/* ================================================================
 * Test 4: Uniform wf=1 → FFT(1) is delta at k=0, kernel[k=0]=0
 *   so Φ_dd = 0 everywhere. localTermK/M remain unchanged.
 * ================================================================ */
TEST(meanfield_uniform_density_d) {
    const size_t N[] = {8, 8, 8};
    const f64    L[] = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};
    matmul_ctx_t *ctx = matmul_ctx_alloc(3, N, L);
    matmul_ctx_set_system(ctx, 0);
    const size_t size = ctx->size;

    ctx->localTermK = xcalloc(size, sizeof(f64));
    ctx->localTermM = xcalloc(size, sizeof(f64));
    ctx->wf = xcalloc(size, sizeof(f64));
    for (size_t i = 0; i < size; i++)
        ((f64 *)ctx->wf)[i] = 1.0;

    const f64 dir[] = {0.0, 0.0, 1.0};
    dipolar_set_kernel(ctx, 1.0, dir, L[0] / 2.0);
    dipolar_add_meanfield(ctx);

    for (size_t i = 0; i < size; i++) {
        ASSERT_CLOSE(ctx->localTermK[i], 0.0, 1e-12);
        ASSERT_CLOSE(ctx->localTermM[i], 0.0, 1e-12);
    }

    matmul_ctx_free(&ctx);
}

/* ================================================================
 * Test 5: d/z consistency — wf = 1 + 0.3*cos(x+y+z), real values
 *   in both d and z contexts. localTermK/M must match.
 * ================================================================ */
TEST(meanfield_dz_consistency) {
    const size_t N[] = {8, 8, 8};
    const f64    L[] = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};

    matmul_ctx_t *ctx_d = matmul_ctx_alloc(3, N, L);
    matmul_ctx_set_system(ctx_d, 0);
    matmul_ctx_t *ctx_z = matmul_ctx_alloc(3, N, L);
    matmul_ctx_set_system(ctx_z, 1);
    const size_t size = ctx_d->size;

    ctx_d->localTermK = xcalloc(size, sizeof(f64));
    ctx_d->localTermM = xcalloc(size, sizeof(f64));
    ctx_d->wf = xcalloc(size, sizeof(f64));
    ctx_z->localTermK = xcalloc(size, sizeof(f64));
    ctx_z->localTermM = xcalloc(size, sizeof(f64));
    ctx_z->wf = xcalloc(size, sizeof(c64));

    for (size_t iz = 0; iz < N[2]; iz++)
        for (size_t iy = 0; iy < N[1]; iy++)
            for (size_t ix = 0; ix < N[0]; ix++) {
                const f64 xv = (f64)ix * L[0] / (f64)N[0];
                const f64 yv = (f64)iy * L[1] / (f64)N[1];
                const f64 zv = (f64)iz * L[2] / (f64)N[2];
                const size_t idx = iz * N[1] * N[0] + iy * N[0] + ix;
                const f64 val = 1.0 + 0.3 * cos(xv + yv + zv);
                ((f64 *)ctx_d->wf)[idx] = val;
                ((c64 *)ctx_z->wf)[idx] = val + 0.0 * I;
            }

    const f64 dir[] = {0.0, 0.0, 1.0};
    const f64 Rc = L[0] / 2.0;
    dipolar_set_kernel(ctx_d, 1.0, dir, Rc);
    dipolar_set_kernel(ctx_z, 1.0, dir, Rc);
    dipolar_add_meanfield(ctx_d);
    dipolar_add_meanfield(ctx_z);

    for (size_t i = 0; i < size; i++) {
        ASSERT_CLOSE(ctx_d->localTermK[i], ctx_z->localTermK[i], 1e-10);
        ASSERT_CLOSE(ctx_d->localTermM[i], ctx_z->localTermM[i], 1e-10);
    }

    matmul_ctx_free(&ctx_d);
    matmul_ctx_free(&ctx_z);
}

/* ================================================================
 * Test 6: dipolar_conv linearity — conv(a*v) = a * conv(v)
 * ================================================================ */
TEST(dipolar_conv_linearity_d) {
    const size_t N[] = {8, 8, 8};
    const f64    L[] = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};
    matmul_ctx_t *ctx = matmul_ctx_alloc(3, N, L);
    matmul_ctx_set_system(ctx, 0);
    const size_t size = ctx->size;

    ctx->wf = xcalloc(size, sizeof(f64));
    for (size_t i = 0; i < size; i++)
        ((f64 *)ctx->wf)[i] = 1.0 + 0.1 * (f64)(i % 7);

    const f64 dir[] = {0.0, 0.0, 1.0};
    dipolar_set_kernel(ctx, 1.5, dir, L[0] / 2.0);

    f64 *v1   = xcalloc(size, sizeof(f64));
    f64 *v2   = xcalloc(size, sizeof(f64));
    f64 *out1 = xcalloc(size, sizeof(f64));
    f64 *out2 = xcalloc(size, sizeof(f64));

    for (size_t i = 0; i < size; i++) {
        v1[i] = cos(2.0 * M_PI * (f64)(i % N[0]) / (f64)N[0]);
        v2[i] = 3.7 * v1[i];
    }

    dipolar_conv_d(ctx, v1, out1);
    dipolar_conv_d(ctx, v2, out2);

    for (size_t i = 0; i < size; i++)
        ASSERT_CLOSE(out2[i], 3.7 * out1[i], 1e-10);

    safe_free((void **)&v1); safe_free((void **)&v2);
    safe_free((void **)&out1); safe_free((void **)&out2);
    matmul_ctx_free(&ctx);
}

/* ================================================================
 * Test 7: d/z consistency — real input gives same result both paths
 * ================================================================ */
TEST(dipolar_conv_dz_consistency) {
    const size_t N[] = {8, 8, 8};
    const f64    L[] = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};

    matmul_ctx_t *ctx_d = matmul_ctx_alloc(3, N, L);
    matmul_ctx_set_system(ctx_d, 0);
    matmul_ctx_t *ctx_z = matmul_ctx_alloc(3, N, L);
    matmul_ctx_set_system(ctx_z, 1);
    const size_t size = ctx_d->size;

    ctx_d->wf = xcalloc(size, sizeof(f64));
    ctx_z->wf = xcalloc(size, sizeof(c64));
    for (size_t i = 0; i < size; i++) {
        const f64 val = 1.0 + 0.2 * (f64)(i % 5);
        ((f64 *)ctx_d->wf)[i] = val;
        ((c64 *)ctx_z->wf)[i] = val + 0.0 * I;
    }

    const f64 dir[] = {0.0, 0.0, 1.0};
    dipolar_set_kernel(ctx_d, 1.0, dir, L[0] / 2.0);
    dipolar_set_kernel(ctx_z, 1.0, dir, L[0] / 2.0);

    f64 *vd = xcalloc(size, sizeof(f64));
    c64 *vz = xcalloc(size, sizeof(c64));
    f64 *od = xcalloc(size, sizeof(f64));
    c64 *oz = xcalloc(size, sizeof(c64));

    for (size_t i = 0; i < size; i++) {
        vd[i] = sin(2.0 * M_PI * (f64)(i % N[0]) / (f64)N[0]);
        vz[i] = vd[i] + 0.0 * I;
    }

    dipolar_conv_d(ctx_d, vd, od);
    dipolar_conv_z(ctx_z, vz, oz);

    for (size_t i = 0; i < size; i++) {
        ASSERT_CLOSE(creal(oz[i]), od[i], 1e-10);
        ASSERT_CLOSE(cimag(oz[i]), 0.0, 1e-10);
    }

    safe_free((void **)&vd); safe_free((void **)&vz);
    safe_free((void **)&od); safe_free((void **)&oz);
    matmul_ctx_free(&ctx_d); matmul_ctx_free(&ctx_z);
}

/* ================================================================ */
int main(void) {
    printf("Dipolar kernel (k=0 edge case):\n");
    RUN(kernel_k0_is_zero);

    printf("\nDipolar kernel (angular dependence):\n");
    RUN(kernel_angular_dependence);

    printf("\nDipolar kernel (r2c vs c2c):\n");
    RUN(kernel_r2c_vs_c2c);

    printf("\ndipolar_add_meanfield:\n");
    RUN(meanfield_uniform_density_d);
    RUN(meanfield_dz_consistency);

    printf("\ndipolar_conv:\n");
    RUN(dipolar_conv_linearity_d);
    RUN(dipolar_conv_dz_consistency);

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
