#include "bdg_internal.h"
#include <complex.h>
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
 * Helper: create a 1D matmul_ctx_t with constant localTermK=VK, localTermM=VM
 * ================================================================ */
static matmul_ctx_t *make_1d_ctx(const uint64_t N, const f64 L,
                                  const int complex_psi0,
                                  const f64 VK, const f64 VM) {
    const uint64_t Narr[] = {N};
    const f64    Larr[] = {L};
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
 * Test 1: M without dipolar on cos(x), VM=2
 * M(cos(x)) = (0.5*1 + 2)*cos(x) = 2.5*cos(x)
 * ================================================================ */
TEST(matmulM_d_planewave_no_dipolar) {
    matmul_ctx_t *ctx = make_1d_ctx(64, 2.0 * M_PI, 0, 0.0, 2.0);
    const uint64_t size = ctx->size;
    f64 *x = xcalloc(size, sizeof(f64));
    f64 *y = xcalloc(size, sizeof(f64));

    for (uint64_t i = 0; i < size; i++)
        x[i] = cos((f64)i * 2.0 * M_PI / (f64)size);

    matmulM_d(ctx, x, y);

    for (uint64_t i = 0; i < size; i++) {
        const f64 expected = 2.5 * cos((f64)i * 2.0 * M_PI / (f64)size);
        ASSERT_CLOSE(y[i], expected, TOL);
    }

    safe_free((void **)&x);
    safe_free((void **)&y);
    matmul_ctx_free(&ctx);
}

/* ================================================================
 * Test 2: M matches K when localTermK == localTermM and no dipolar
 * ================================================================ */
TEST(matmulM_matches_K_no_dipolar) {
    matmul_ctx_t *ctx_K = make_1d_ctx(64, 2.0 * M_PI, 0, 1.5, 1.5);
    matmul_ctx_t *ctx_M = make_1d_ctx(64, 2.0 * M_PI, 0, 1.5, 1.5);
    const uint64_t size = ctx_K->size;

    f64 *x  = xcalloc(size, sizeof(f64));
    f64 *yK = xcalloc(size, sizeof(f64));
    f64 *yM = xcalloc(size, sizeof(f64));

    for (uint64_t i = 0; i < size; i++)
        x[i] = cos((f64)i * 2.0 * M_PI / (f64)size)
             + 0.5 * cos(3.0 * (f64)i * 2.0 * M_PI / (f64)size);

    matmulK_d(ctx_K, x, yK);
    matmulM_d(ctx_M, x, yM);

    for (uint64_t i = 0; i < size; i++)
        ASSERT_CLOSE(yM[i], yK[i], TOL);

    safe_free((void **)&x);
    safe_free((void **)&yK);
    safe_free((void **)&yM);
    matmul_ctx_free(&ctx_K);
    matmul_ctx_free(&ctx_M);
}

/* ================================================================
 * Test 3: M d vs z consistency (no dipolar)
 * ================================================================ */
TEST(matmulM_dz_consistency) {
    const uint64_t N = 64;
    const f64 L = 2.0 * M_PI;

    matmul_ctx_t *ctx_d = make_1d_ctx(N, L, 0, 0.0, 2.0);
    matmul_ctx_t *ctx_z = make_1d_ctx(N, L, 1, 0.0, 2.0);

    f64 *xd = xcalloc(N, sizeof(f64));
    f64 *yd = xcalloc(N, sizeof(f64));
    c64 *xz = xcalloc(N, sizeof(c64));
    c64 *yz = xcalloc(N, sizeof(c64));

    for (uint64_t i = 0; i < N; i++) {
        xd[i] = cos((f64)i * 2.0 * M_PI / (f64)N);
        xz[i] = xd[i] + 0.0 * I;
    }

    matmulM_d(ctx_d, xd, yd);
    matmulM_z(ctx_z, xz, yz);

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
 * Test 4: M with dipolar in 3D — verify dipolar term is nonzero
 * Apply to cos(z) with uniform wf=1, check M differs from K.
 * cos(z) has k=(0,0,1); dipole along z gives cos^2(theta)=1,
 * kernel = 2*f_cutoff != 0, so the dipolar convolution is nonzero.
 * ================================================================ */
TEST(matmulM_d_3d_with_dipolar) {
    const uint64_t N[] = {8, 8, 8};
    const f64    L[] = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};
    matmul_ctx_t *ctx = matmul_ctx_alloc(3, N, L);
    matmul_ctx_set_system(ctx, 0);
    const uint64_t size = ctx->size;

    /* Constant local terms, same for K and M */
    ctx->localTermK = xcalloc(size, sizeof(f64));
    ctx->localTermM = xcalloc(size, sizeof(f64));
    for (uint64_t i = 0; i < size; i++) {
        ctx->localTermK[i] = 1.0;
        ctx->localTermM[i] = 1.0;
    }

    /* Uniform wf */
    ctx->wf = xcalloc(size, sizeof(f64));
    for (uint64_t i = 0; i < size; i++)
        ((f64 *)ctx->wf)[i] = 1.0;

    /* Set dipolar kernel */
    const f64 dir[] = {0.0, 0.0, 1.0};
    dipolar_set_kernel(ctx, 1.0, dir, L[0] / 2.0);
    ctx->dipolar = 1;

    /* Input: cos(z) — has k=(0,0,1) along the dipole axis */
    f64 *x  = xcalloc(size, sizeof(f64));
    f64 *yK = xcalloc(size, sizeof(f64));
    f64 *yM = xcalloc(size, sizeof(f64));

    for (uint64_t iz = 0; iz < N[2]; iz++)
        for (uint64_t iy = 0; iy < N[1]; iy++)
            for (uint64_t ix = 0; ix < N[0]; ix++) {
                const f64 zv = (f64)iz * L[2] / (f64)N[2];
                x[iz * N[1] * N[0] + iy * N[0] + ix] = cos(zv);
            }

    matmulK_d(ctx, x, yK);
    matmulM_d(ctx, x, yM);

    /* K and M should differ because dipolar term is nonzero for cos(z) */
    f64 max_diff = 0.0;
    for (uint64_t i = 0; i < size; i++) {
        const f64 diff = fabs(yM[i] - yK[i]);
        if (diff > max_diff) max_diff = diff;
    }
    ASSERT(max_diff > 1e-6);

    safe_free((void **)&x);
    safe_free((void **)&yK);
    safe_free((void **)&yM);
    matmul_ctx_free(&ctx);
}

/* ================================================================
 * Test 5: M with dipolar d vs z consistency in 3D
 * ================================================================ */
TEST(matmulM_dz_3d_dipolar_consistency) {
    const uint64_t N[] = {8, 8, 8};
    const f64    L[] = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};

    matmul_ctx_t *ctx_d = matmul_ctx_alloc(3, N, L);
    matmul_ctx_set_system(ctx_d, 0);
    matmul_ctx_t *ctx_z = matmul_ctx_alloc(3, N, L);
    matmul_ctx_set_system(ctx_z, 1);
    const uint64_t size = ctx_d->size;

    ctx_d->localTermK = xcalloc(size, sizeof(f64));
    ctx_d->localTermM = xcalloc(size, sizeof(f64));
    ctx_d->wf = xcalloc(size, sizeof(f64));
    ctx_z->localTermK = xcalloc(size, sizeof(f64));
    ctx_z->localTermM = xcalloc(size, sizeof(f64));
    ctx_z->wf = xcalloc(size, sizeof(c64));

    for (uint64_t i = 0; i < size; i++) {
        ctx_d->localTermK[i] = 1.0;
        ctx_d->localTermM[i] = 1.5;
        ctx_z->localTermK[i] = 1.0;
        ctx_z->localTermM[i] = 1.5;
        const f64 wval = 1.0 + 0.1 * (f64)(i % 7);
        ((f64 *)ctx_d->wf)[i] = wval;
        ((c64 *)ctx_z->wf)[i] = wval + 0.0 * I;
    }

    const f64 dir[] = {0.0, 0.0, 1.0};
    dipolar_set_kernel(ctx_d, 1.0, dir, ctx_d->L[0] / 2.0);
    dipolar_set_kernel(ctx_z, 1.0, dir, ctx_z->L[0] / 2.0);
    ctx_d->dipolar = 1;
    ctx_z->dipolar = 1;

    f64 *xd = xcalloc(size, sizeof(f64));
    c64 *xz = xcalloc(size, sizeof(c64));
    f64 *yd = xcalloc(size, sizeof(f64));
    c64 *yz = xcalloc(size, sizeof(c64));

    for (uint64_t i = 0; i < size; i++) {
        xd[i] = sin(2.0 * M_PI * (f64)(i % N[0]) / (f64)N[0]);
        xz[i] = xd[i] + 0.0 * I;
    }

    matmulM_d(ctx_d, xd, yd);
    matmulM_z(ctx_z, xz, yz);

    for (uint64_t i = 0; i < size; i++) {
        ASSERT_CLOSE(creal(yz[i]), yd[i], 1e-10);
        ASSERT_CLOSE(cimag(yz[i]), 0.0, 1e-10);
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
    printf("matmulM (no dipolar):\n");
    RUN(matmulM_d_planewave_no_dipolar);
    RUN(matmulM_matches_K_no_dipolar);
    RUN(matmulM_dz_consistency);

    printf("\nmatmulM (3D dipolar):\n");
    RUN(matmulM_d_3d_with_dipolar);
    RUN(matmulM_dz_3d_dipolar_consistency);

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");
    return tests_failed > 0 ? 1 : 0;
}
