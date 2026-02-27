#include "bdg_internal.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define TOL 1e-12

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

#define ASSERT_NEAR(a, b, tol) ASSERT(fabs((a) - (b)) < (tol))

/* prints actual vs expected on failure — better for debugging k-space indexing */
#define ASSERT_CLOSE(a, b, tol) do { \
    const double _a = (a), _b = (b); \
    if (fabs(_a - _b) >= (tol)) { \
        printf("[FAIL] line %d: %.15e vs %.15e\n", __LINE__, _a, _b); \
        tests_failed++; \
        return; \
    } \
} while(0)

/* ================================================================
 * test_k_size: verify k_size for r2c and c2c
 * ================================================================ */
TEST(k_size) {
    /* 1D: N=32 */
    {
        const size_t N[] = {32};
        const f64    L[] = {1.0};
        matmul_ctx_t *ctx = matmul_ctx_alloc(1, N, L);

        matmul_ctx_set_system(ctx, 0);
        ASSERT(ctx->k_size == 17);
        matmul_ctx_free(&ctx);

        ctx = matmul_ctx_alloc(1, N, L);
        matmul_ctx_set_system(ctx, 1);
        ASSERT(ctx->k_size == 32);
        matmul_ctx_free(&ctx);
    }
    /* 2D: N={16,8} */
    {
        const size_t N[] = {16, 8};
        const f64    L[] = {1.0, 1.0};
        matmul_ctx_t *ctx = matmul_ctx_alloc(2, N, L);

        matmul_ctx_set_system(ctx, 0);
        ASSERT(ctx->k_size == 9 * 8);
        matmul_ctx_free(&ctx);

        ctx = matmul_ctx_alloc(2, N, L);
        matmul_ctx_set_system(ctx, 1);
        ASSERT(ctx->k_size == 128);
        matmul_ctx_free(&ctx);
    }
    /* 3D: N={8,6,4} */
    {
        const size_t N[] = {8, 6, 4};
        const f64    L[] = {1.0, 1.0, 1.0};
        matmul_ctx_t *ctx = matmul_ctx_alloc(3, N, L);

        matmul_ctx_set_system(ctx, 0);
        ASSERT(ctx->k_size == 5 * 6 * 4);
        matmul_ctx_free(&ctx);

        ctx = matmul_ctx_alloc(3, N, L);
        matmul_ctx_set_system(ctx, 1);
        ASSERT(ctx->k_size == 192);
        matmul_ctx_free(&ctx);
    }
}

/* ================================================================
 * test_kx2: per-dimension k² arrays
 * ================================================================ */
TEST(kx2) {
    const size_t N[] = {8, 6};
    const f64    L[] = {2.0 * M_PI, 2.0 * M_PI};
    matmul_ctx_t *ctx = matmul_ctx_alloc(2, N, L);
    matmul_ctx_set_system(ctx, 0);

    /* kx2[0]: 8 entries, dk=1 */
    const f64 expect_kx0[] = {0, 1, 4, 9, 16, 9, 4, 1};
    for (size_t j = 0; j < 8; j++)
        ASSERT_CLOSE(ctx->kx2[0][j], expect_kx0[j], TOL);

    /* kx2[1]: 6 entries, dk=1 */
    const f64 expect_kx1[] = {0, 1, 4, 9, 4, 1};
    for (size_t j = 0; j < 6; j++)
        ASSERT_CLOSE(ctx->kx2[1][j], expect_kx1[j], TOL);

    matmul_ctx_free(&ctx);
}

/* ================================================================
 * test_k2_1d: k2 values with L=2*pi so dk=1
 * ================================================================ */
TEST(k2_1d) {
    const size_t N_val = 32;
    const size_t N[] = {N_val};
    const f64 L[] = {2.0 * M_PI};

    /* r2c path: k_size = 17, indices 0..16 */
    {
        matmul_ctx_t *ctx = matmul_ctx_alloc(1, N, L);
        matmul_ctx_set_system(ctx, 0);

        ASSERT(ctx->k_size == 17);
        for (size_t j = 0; j <= 16; j++) {
            const f64 expected = (f64)(j * j);
            ASSERT_CLOSE(ctx->k2[j], expected, TOL);
        }
        matmul_ctx_free(&ctx);
    }
    /* c2c path: k_size = 32 */
    {
        matmul_ctx_t *ctx = matmul_ctx_alloc(1, N, L);
        matmul_ctx_set_system(ctx, 1);

        ASSERT(ctx->k_size == 32);
        for (size_t j = 0; j < 32; j++) {
            const int kj = (j <= 16) ? (int)j : (int)j - 32;
            const f64 expected = (f64)(kj * kj);
            ASSERT_CLOSE(ctx->k2[j], expected, TOL);
        }
        matmul_ctx_free(&ctx);
    }
}

/* ================================================================
 * test_k2_2d: 2D r2c k2 — previously untested code path
 * ================================================================ */
TEST(k2_2d) {
    const size_t N[] = {8, 6};
    const f64    L[] = {2.0 * M_PI, 2.0 * M_PI};
    matmul_ctx_t *ctx = matmul_ctx_alloc(2, N, L);
    matmul_ctx_set_system(ctx, 0);

    const size_t N0_k = 5;
    ASSERT(ctx->k_size == 30);

    /* k2[iy * N0_k + ix] = kx2[0][ix] + kx2[1][iy] */
    ASSERT_CLOSE(ctx->k2[0], 0.0, TOL);                   /* ix=0, iy=0 */
    ASSERT_CLOSE(ctx->k2[1], 1.0, TOL);                   /* ix=1, iy=0 */
    ASSERT_CLOSE(ctx->k2[1 * N0_k], 1.0, TOL);            /* ix=0, iy=1 */
    ASSERT_CLOSE(ctx->k2[1 * N0_k + 1], 2.0, TOL);        /* ix=1, iy=1 */
    ASSERT_CLOSE(ctx->k2[4 * N0_k], 4.0, TOL);            /* ix=0, iy=4 → ky=-2 */

    matmul_ctx_free(&ctx);
}

/* ================================================================
 * test_k2_2d_c2c: 2D c2c k2 — different stride from r2c
 * ================================================================ */
TEST(k2_2d_c2c) {
    const size_t N[] = {8, 6};
    const f64    L[] = {2.0 * M_PI, 2.0 * M_PI};
    matmul_ctx_t *ctx = matmul_ctx_alloc(2, N, L);
    matmul_ctx_set_system(ctx, 1);

    const size_t N0_k = 8;
    ASSERT(ctx->k_size == 48);

    /* k2[iy * N0_k + ix] = kx2[0][ix] + kx2[1][iy] */
    ASSERT_CLOSE(ctx->k2[0], 0.0, TOL);                   /* ix=0, iy=0 */
    ASSERT_CLOSE(ctx->k2[1], 1.0, TOL);                   /* ix=1, iy=0 */
    ASSERT_CLOSE(ctx->k2[1 * N0_k], 1.0, TOL);            /* ix=0, iy=1 */
    ASSERT_CLOSE(ctx->k2[1 * N0_k + 1], 2.0, TOL);        /* ix=1, iy=1 */

    matmul_ctx_free(&ctx);
}

/* ================================================================
 * test_k2_3d: spot-check known entries for 3D k2
 * ================================================================ */
TEST(k2_3d) {
    const size_t N[] = {8, 8, 8};
    const f64 L[] = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};

    /* r2c: N0_k = 5, k_size = 5*8*8 = 320 */
    {
        matmul_ctx_t *ctx = matmul_ctx_alloc(3, N, L);
        matmul_ctx_set_system(ctx, 0);

        const size_t N0_k = 5;
        ASSERT(ctx->k_size == 320);

        ASSERT_CLOSE(ctx->k2[0], 0.0, TOL);

        /* ix=1, iy=0, iz=0 */
        ASSERT_CLOSE(ctx->k2[1], 1.0, TOL);

        /* ix=0, iy=1, iz=0 */
        ASSERT_CLOSE(ctx->k2[1 * N0_k], 1.0, TOL);

        /* ix=0, iy=0, iz=1 */
        ASSERT_CLOSE(ctx->k2[1 * 8 * N0_k], 1.0, TOL);

        /* ix=1, iy=1, iz=1 → k2 = 3 */
        const size_t idx_111 = 1 * 8 * N0_k + 1 * N0_k + 1;
        ASSERT_CLOSE(ctx->k2[idx_111], 3.0, TOL);

        /* ix=2, iy=3, iz=5 → kx=2, ky=3, kz=5-8=-3 → k2=4+9+9=22 */
        const size_t idx_235 = 5 * 8 * N0_k + 3 * N0_k + 2;
        ASSERT_CLOSE(ctx->k2[idx_235], 22.0, TOL);

        matmul_ctx_free(&ctx);
    }
}

/* ================================================================
 * test_fftw_roundtrip_c2c_2d: FFT → IFFT recovers input
 * ================================================================ */
TEST(fftw_roundtrip_c2c_2d) {
    const size_t N[] = {16, 8};
    const f64 L[] = {1.0, 1.0};
    matmul_ctx_t *ctx = matmul_ctx_alloc(2, N, L);
    matmul_ctx_set_system(ctx, 1);

    const size_t size = ctx->size;
    c64 *in  = (c64 *)ctx->c_wrk1;
    c64 *wrk = (c64 *)ctx->f_wrk;

    for (size_t i = 0; i < size; i++)
        in[i] = (f64)i + I * (f64)(size - i);

    c64 *saved = xcalloc(size, sizeof(c64));
    memcpy(saved, in, size * sizeof(c64));

    fftw_execute_dft(ctx->fwd_plan,
                     (fftw_complex *)in, (fftw_complex *)wrk);
    fftw_execute_dft(ctx->bwd_plan,
                     (fftw_complex *)wrk, (fftw_complex *)in);

    int ok = 1;
    for (size_t i = 0; i < size; i++) {
        const f64 re_err = fabs(creal(in[i]) / (f64)size - creal(saved[i]));
        const f64 im_err = fabs(cimag(in[i]) / (f64)size - cimag(saved[i]));
        if (re_err > 1e-10 || im_err > 1e-10) { ok = 0; break; }
    }
    ASSERT(ok);

    safe_free((void **)&saved);
    matmul_ctx_free(&ctx);
}

/* ================================================================
 * test_fftw_roundtrip_r2c_1d: r2c → c2r recovers input
 * ================================================================ */
TEST(fftw_roundtrip_r2c_1d) {
    const size_t N[] = {32};
    const f64 L[] = {1.0};
    matmul_ctx_t *ctx = matmul_ctx_alloc(1, N, L);
    matmul_ctx_set_system(ctx, 0);

    const size_t size = ctx->size;
    f64 *in = (f64 *)ctx->c_wrk1;
    fftw_complex *wrk = (fftw_complex *)ctx->f_wrk;

    for (size_t i = 0; i < size; i++)
        in[i] = sin(2.0 * M_PI * (f64)i / (f64)size) + 0.5;

    f64 *saved = xcalloc(size, sizeof(f64));
    memcpy(saved, in, size * sizeof(f64));

    fftw_execute_dft_r2c(ctx->fwd_plan, in, wrk);
    fftw_execute_dft_c2r(ctx->bwd_plan, wrk, in);

    int ok = 1;
    for (size_t i = 0; i < size; i++) {
        if (fabs(in[i] / (f64)size - saved[i]) > 1e-10) { ok = 0; break; }
    }
    ASSERT(ok);

    safe_free((void **)&saved);
    matmul_ctx_free(&ctx);
}

/* ================================================================
 * test_fftw_roundtrip_r2c_3d: 3D r2c → c2r recovers input
 * ================================================================ */
TEST(fftw_roundtrip_r2c_3d) {
    const size_t N[] = {8, 6, 4};
    const f64 L[] = {1.0, 2.0, 3.0};
    matmul_ctx_t *ctx = matmul_ctx_alloc(3, N, L);
    matmul_ctx_set_system(ctx, 0);

    const size_t size = ctx->size;
    f64 *in = (f64 *)ctx->c_wrk1;
    fftw_complex *wrk = (fftw_complex *)ctx->f_wrk;

    for (size_t i = 0; i < size; i++)
        in[i] = (f64)(i % 7) - 3.0;

    f64 *saved = xcalloc(size, sizeof(f64));
    memcpy(saved, in, size * sizeof(f64));

    fftw_execute_dft_r2c(ctx->fwd_plan, in, wrk);
    fftw_execute_dft_c2r(ctx->bwd_plan, wrk, in);

    int ok = 1;
    for (size_t i = 0; i < size; i++) {
        if (fabs(in[i] / (f64)size - saved[i]) > 1e-10) { ok = 0; break; }
    }
    ASSERT(ok);

    safe_free((void **)&saved);
    matmul_ctx_free(&ctx);
}

/* ================================================================ */
int main(void) {
    printf("k-space geometry:\n");
    RUN(k_size);
    RUN(kx2);

    printf("\nk2 values:\n");
    RUN(k2_1d);
    RUN(k2_2d);
    RUN(k2_2d_c2c);
    RUN(k2_3d);

    printf("\nFFTW roundtrip:\n");
    RUN(fftw_roundtrip_c2c_2d);
    RUN(fftw_roundtrip_r2c_1d);
    RUN(fftw_roundtrip_r2c_3d);

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
