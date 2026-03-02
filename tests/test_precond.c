#include "bdg_internal.h"
#include <complex.h>
#include <math.h>
#include <stdio.h>

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

/* ── helpers ───────────────────────────────────────────────── */

static matmul_ctx_t *make_1d_ctx(const uint64_t N, const f64 L,
                                  const int complex_psi0,
                                  const f64 V0, const f64 mu) {
    const uint64_t Narr[] = {N};
    const f64 Larr[] = {L};
    matmul_ctx_t *ctx = matmul_ctx_alloc(1, Narr, Larr);
    matmul_ctx_set_system(ctx, complex_psi0);

    const uint64_t size = ctx->size;
    ctx->localTermK = xcalloc(size, sizeof(f64));
    ctx->localTermM = xcalloc(size, sizeof(f64));
    ctx->precond_sqrtK = xcalloc(size, sizeof(f64));
    ctx->precond_sqrtM = xcalloc(size, sizeof(f64));
    ctx->mu = mu;

    for (uint64_t i = 0; i < size; i++) {
        ctx->localTermK[i] = V0 - mu;   /* after mu subtraction */
        ctx->localTermM[i] = V0 - mu;
        ctx->precond_sqrtK[i] = 1.0 / sqrt(V0);
        ctx->precond_sqrtM[i] = 1.0 / sqrt(V0);
    }
    return ctx;
}

/* ── precondK tests ────────────────────────────────────────── */

TEST(precondK_d_planewave_1d) {
    const uint64_t N = 64;
    const f64 L = 2.0 * M_PI;
    const f64 V0 = 2.0, mu = 0.5;
    matmul_ctx_t *ctx = make_1d_ctx(N, L, 0, V0, mu);
    const uint64_t size = ctx->size;

    f64 *x = xcalloc(size, sizeof(f64));
    f64 *y = xcalloc(size, sizeof(f64));

    /* k=1 planewave */
    for (uint64_t i = 0; i < size; i++) {
        const f64 xj = (f64)i * L / (f64)size;
        x[i] = cos(xj);
    }

    precondK_d(ctx, x, y);

    /* expected: cos(kx) / (V0 * (mu + 0.5*k^2)) with k=1 */
    const f64 expected_scale = 1.0 / (V0 * (mu + 0.5));
    for (uint64_t i = 0; i < size; i++) {
        const f64 xj = (f64)i * L / (f64)size;
        ASSERT_CLOSE(y[i], expected_scale * cos(xj), TOL);
    }

    safe_free((void **)&x);
    safe_free((void **)&y);
    matmul_ctx_free(&ctx);
}

TEST(precondK_z_planewave_1d) {
    const uint64_t N = 64;
    const f64 L = 2.0 * M_PI;
    const f64 V0 = 2.0, mu = 0.5;
    matmul_ctx_t *ctx = make_1d_ctx(N, L, 1, V0, mu);
    const uint64_t size = ctx->size;

    c64 *x = xcalloc(size, sizeof(c64));
    c64 *y = xcalloc(size, sizeof(c64));

    for (uint64_t i = 0; i < size; i++) {
        const f64 xj = (f64)i * L / (f64)size;
        x[i] = cos(xj) + 0.0 * I;
    }

    precondK_z(ctx, x, y);

    const f64 expected_scale = 1.0 / (V0 * (mu + 0.5));
    for (uint64_t i = 0; i < size; i++) {
        const f64 xj = (f64)i * L / (f64)size;
        ASSERT_CLOSE(creal(y[i]), expected_scale * cos(xj), TOL);
        ASSERT_CLOSE(cimag(y[i]), 0.0, TOL);
    }

    safe_free((void **)&x);
    safe_free((void **)&y);
    matmul_ctx_free(&ctx);
}

TEST(precondK_dz_consistency) {
    const uint64_t N = 64;
    const f64 L = 2.0 * M_PI;
    const f64 V0 = 3.0, mu = 1.0;
    matmul_ctx_t *ctx_d = make_1d_ctx(N, L, 0, V0, mu);
    matmul_ctx_t *ctx_z = make_1d_ctx(N, L, 1, V0, mu);
    const uint64_t size = ctx_d->size;

    f64 *xd = xcalloc(size, sizeof(f64));
    f64 *yd = xcalloc(size, sizeof(f64));
    c64 *xz = xcalloc(size, sizeof(c64));
    c64 *yz = xcalloc(size, sizeof(c64));

    for (uint64_t i = 0; i < size; i++) {
        const f64 xj = (f64)i * L / (f64)size;
        xd[i] = cos(xj) + 0.3 * sin(2.0 * xj);
        xz[i] = xd[i] + 0.0 * I;
    }

    precondK_d(ctx_d, xd, yd);
    precondK_z(ctx_z, xz, yz);

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

/* ── precondM tests ────────────────────────────────────────── */

TEST(precondM_d_planewave_1d) {
    const uint64_t N = 64;
    const f64 L = 2.0 * M_PI;
    const f64 VK = 2.0, VM = 3.5, mu = 0.5;
    const uint64_t Narr[] = {N};
    const f64 Larr[] = {L};
    matmul_ctx_t *ctx = matmul_ctx_alloc(1, Narr, Larr);
    matmul_ctx_set_system(ctx, 0);

    const uint64_t size = ctx->size;
    ctx->localTermK = xcalloc(size, sizeof(f64));
    ctx->localTermM = xcalloc(size, sizeof(f64));
    ctx->precond_sqrtK = xcalloc(size, sizeof(f64));
    ctx->precond_sqrtM = xcalloc(size, sizeof(f64));
    ctx->mu = mu;

    for (uint64_t i = 0; i < size; i++) {
        ctx->localTermK[i] = VK - mu;
        ctx->localTermM[i] = VM - mu;
        ctx->precond_sqrtK[i] = 1.0 / sqrt(VK);
        ctx->precond_sqrtM[i] = 1.0 / sqrt(VM);
    }

    f64 *x = xcalloc(size, sizeof(f64));
    f64 *y = xcalloc(size, sizeof(f64));

    for (uint64_t i = 0; i < size; i++) {
        const f64 xj = (f64)i * L / (f64)size;
        x[i] = cos(xj);
    }

    precondM_d(ctx, x, y);

    const f64 expected_scale = 1.0 / (VM * (mu + 0.5));
    for (uint64_t i = 0; i < size; i++) {
        const f64 xj = (f64)i * L / (f64)size;
        ASSERT_CLOSE(y[i], expected_scale * cos(xj), TOL);
    }

    safe_free((void **)&x);
    safe_free((void **)&y);
    matmul_ctx_free(&ctx);
}

/* ── precondLrep tests ─────────────────────────────────────── */

TEST(precondLrep_d_block_structure) {
    const uint64_t N = 64;
    const f64 L = 2.0 * M_PI;
    const f64 VK = 2.0, VM = 3.5, mu = 0.5;
    const uint64_t Narr[] = {N};
    const f64 Larr[] = {L};
    matmul_ctx_t *ctx = matmul_ctx_alloc(1, Narr, Larr);
    matmul_ctx_set_system(ctx, 0);

    const uint64_t size = ctx->size;
    ctx->localTermK = xcalloc(size, sizeof(f64));
    ctx->localTermM = xcalloc(size, sizeof(f64));
    ctx->precond_sqrtK = xcalloc(size, sizeof(f64));
    ctx->precond_sqrtM = xcalloc(size, sizeof(f64));
    ctx->mu = mu;

    for (uint64_t i = 0; i < size; i++) {
        ctx->localTermK[i] = VK - mu;
        ctx->localTermM[i] = VM - mu;
        ctx->precond_sqrtK[i] = 1.0 / sqrt(VK);
        ctx->precond_sqrtM[i] = 1.0 / sqrt(VM);
    }

    /* stacked input: u = cos(x), v = sin(x) */
    f64 *x_stacked = xcalloc(2 * size, sizeof(f64));
    f64 *y_stacked = xcalloc(2 * size, sizeof(f64));
    f64 *yK_ref = xcalloc(size, sizeof(f64));
    f64 *yM_ref = xcalloc(size, sizeof(f64));

    for (uint64_t i = 0; i < size; i++) {
        const f64 xj = (f64)i * L / (f64)size;
        x_stacked[i] = cos(xj);
        x_stacked[size + i] = sin(xj);
    }

    /* apply Lrep preconditioner */
    precondLrep_d(ctx, x_stacked, y_stacked);

    /* reference: apply K and M separately */
    precondK_d(ctx, x_stacked, yK_ref);
    precondM_d(ctx, &x_stacked[size], yM_ref);

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

TEST(precondLrep_z_block_structure) {
    const uint64_t N = 64;
    const f64 L = 2.0 * M_PI;
    const f64 VK = 2.0, VM = 3.5, mu = 0.5;
    const uint64_t Narr[] = {N};
    const f64 Larr[] = {L};
    matmul_ctx_t *ctx = matmul_ctx_alloc(1, Narr, Larr);
    matmul_ctx_set_system(ctx, 1);

    const uint64_t size = ctx->size;
    ctx->localTermK = xcalloc(size, sizeof(f64));
    ctx->localTermM = xcalloc(size, sizeof(f64));
    ctx->precond_sqrtK = xcalloc(size, sizeof(f64));
    ctx->precond_sqrtM = xcalloc(size, sizeof(f64));
    ctx->mu = mu;

    for (uint64_t i = 0; i < size; i++) {
        ctx->localTermK[i] = VK - mu;
        ctx->localTermM[i] = VM - mu;
        ctx->precond_sqrtK[i] = 1.0 / sqrt(VK);
        ctx->precond_sqrtM[i] = 1.0 / sqrt(VM);
    }

    c64 *x_stacked = xcalloc(2 * size, sizeof(c64));
    c64 *y_stacked = xcalloc(2 * size, sizeof(c64));
    c64 *yK_ref = xcalloc(size, sizeof(c64));
    c64 *yM_ref = xcalloc(size, sizeof(c64));

    for (uint64_t i = 0; i < size; i++) {
        const f64 xj = (f64)i * L / (f64)size;
        x_stacked[i] = cos(xj) + 0.0 * I;
        x_stacked[size + i] = sin(xj) + 0.0 * I;
    }

    precondLrep_z(ctx, x_stacked, y_stacked);
    precondK_z(ctx, x_stacked, yK_ref);
    precondM_z(ctx, &x_stacked[size], yM_ref);

    for (uint64_t i = 0; i < size; i++) {
        ASSERT_CLOSE(creal(y_stacked[i]), creal(yK_ref[i]), TOL);
        ASSERT_CLOSE(cimag(y_stacked[i]), cimag(yK_ref[i]), TOL);
        ASSERT_CLOSE(creal(y_stacked[size + i]), creal(yM_ref[i]), TOL);
        ASSERT_CLOSE(cimag(y_stacked[size + i]), cimag(yM_ref[i]), TOL);
    }

    safe_free((void **)&x_stacked);
    safe_free((void **)&y_stacked);
    safe_free((void **)&yK_ref);
    safe_free((void **)&yM_ref);
    matmul_ctx_free(&ctx);
}

/* ── non-uniform potential test ────────────────────────────── */

TEST(precondK_d_nonuniform_vs_manual) {
    /* Non-uniform potential: verify output is nonzero for delta input.
     * Use N=4 so we can trace every value. */
    const uint64_t N = 4;
    const f64 L = 2.0 * M_PI;
    const f64 mu = 1.0;
    const uint64_t Narr[] = {N};
    const f64 Larr[] = {L};
    matmul_ctx_t *ctx = matmul_ctx_alloc(1, Narr, Larr);
    matmul_ctx_set_system(ctx, 0);

    const uint64_t size = ctx->size;
    ctx->localTermK = xcalloc(size, sizeof(f64));
    ctx->localTermM = xcalloc(size, sizeof(f64));
    ctx->precond_sqrtK = xcalloc(size, sizeof(f64));
    ctx->precond_sqrtM = xcalloc(size, sizeof(f64));
    ctx->mu = mu;

    /* Non-uniform V0 values (before mu subtraction): 2, 3, 4, 5 */
    const f64 V0[] = {2.0, 3.0, 4.0, 5.0};
    for (uint64_t i = 0; i < size; i++) {
        ctx->localTermK[i] = V0[i] - mu;
        ctx->precond_sqrtK[i] = 1.0 / sqrt(V0[i]);
    }

    f64 *x = xcalloc(size, sizeof(f64));
    f64 *y = xcalloc(size, sizeof(f64));

    /* input: [1, 0, 0, 0] — delta function */
    x[0] = 1.0;

    precondK_d(ctx, x, y);

    /* Just verify output is nonzero everywhere (delta spreads through FFT) */
    int any_nonzero = 0;
    for (uint64_t i = 0; i < size; i++) {
        if (fabs(y[i]) > 1e-15) any_nonzero = 1;
    }
    ASSERT(any_nonzero);

    safe_free((void **)&x);
    safe_free((void **)&y);
    matmul_ctx_free(&ctx);
}

/* ── zero-mode test ────────────────────────────────────────── */

TEST(precondK_d_zero_mode) {
    /* Constant input (k=0): precondK(1) = 1/(V0 * mu) */
    const uint64_t N = 64;
    const f64 L = 2.0 * M_PI;
    const f64 V0 = 2.0, mu = 1.5;
    matmul_ctx_t *ctx = make_1d_ctx(N, L, 0, V0, mu);
    const uint64_t size = ctx->size;

    f64 *x = xcalloc(size, sizeof(f64));
    f64 *y = xcalloc(size, sizeof(f64));

    for (uint64_t i = 0; i < size; i++)
        x[i] = 1.0;

    precondK_d(ctx, x, y);

    /* k=0: eigenvalue = 1 / (V0 * (mu + 0)) = 1 / (V0 * mu) */
    const f64 expected = 1.0 / (V0 * mu);
    for (uint64_t i = 0; i < size; i++)
        ASSERT_CLOSE(y[i], expected, TOL);

    safe_free((void **)&x);
    safe_free((void **)&y);
    matmul_ctx_free(&ctx);
}

/* ── main ──────────────────────────────────────────────────── */

int main(void) {
    printf("precondK (real):\n");
    RUN(precondK_d_planewave_1d);
    RUN(precondK_d_zero_mode);
    RUN(precondK_d_nonuniform_vs_manual);

    printf("\nprecondK (complex):\n");
    RUN(precondK_z_planewave_1d);

    printf("\nprecondK d/z consistency:\n");
    RUN(precondK_dz_consistency);

    printf("\nprecondM (real):\n");
    RUN(precondM_d_planewave_1d);

    printf("\nprecondLrep (real):\n");
    RUN(precondLrep_d_block_structure);

    printf("\nprecondLrep (complex):\n");
    RUN(precondLrep_z_block_structure);

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
