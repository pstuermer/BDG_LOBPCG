#include "bdg_internal.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

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

#define ASSERT_CLOSE(a, b, tol) do { \
    const double _a = (a), _b = (b); \
    if (fabs(_a - _b) >= (tol)) { \
        printf("[FAIL] line %d: %.15e vs %.15e\n", __LINE__, _a, _b); \
        tests_failed++; \
        return; \
    } \
} while(0)

/* ================================================================
 * Existing k-space / FFTW tests
 * ================================================================ */

TEST(k_size) {
    {
        const uint64_t N[] = {32};
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
    {
        const uint64_t N[] = {16, 8};
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
    {
        const uint64_t N[] = {8, 6, 4};
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

TEST(kx2) {
    const uint64_t N[] = {8, 6};
    const f64    L[] = {2.0 * M_PI, 2.0 * M_PI};
    matmul_ctx_t *ctx = matmul_ctx_alloc(2, N, L);
    matmul_ctx_set_system(ctx, 0);

    const f64 expect_kx0[] = {0, 1, 4, 9, 16, 9, 4, 1};
    for (uint64_t j = 0; j < 8; j++)
        ASSERT_CLOSE(ctx->kx2[0][j], expect_kx0[j], TOL);

    const f64 expect_kx1[] = {0, 1, 4, 9, 4, 1};
    for (uint64_t j = 0; j < 6; j++)
        ASSERT_CLOSE(ctx->kx2[1][j], expect_kx1[j], TOL);

    matmul_ctx_free(&ctx);
}

TEST(k2_1d) {
    const uint64_t N_val = 32;
    const uint64_t N[] = {N_val};
    const f64 L[] = {2.0 * M_PI};
    {
        matmul_ctx_t *ctx = matmul_ctx_alloc(1, N, L);
        matmul_ctx_set_system(ctx, 0);
        ASSERT(ctx->k_size == 17);
        for (uint64_t j = 0; j <= 16; j++) {
            const f64 expected = (f64)(j * j);
            ASSERT_CLOSE(ctx->k2[j], expected, TOL);
        }
        matmul_ctx_free(&ctx);
    }
    {
        matmul_ctx_t *ctx = matmul_ctx_alloc(1, N, L);
        matmul_ctx_set_system(ctx, 1);
        ASSERT(ctx->k_size == 32);
        for (uint64_t j = 0; j < 32; j++) {
            const int kj = (j <= 16) ? (int)j : (int)j - 32;
            const f64 expected = (f64)(kj * kj);
            ASSERT_CLOSE(ctx->k2[j], expected, TOL);
        }
        matmul_ctx_free(&ctx);
    }
}

TEST(k2_2d) {
    const uint64_t N[] = {8, 6};
    const f64    L[] = {2.0 * M_PI, 2.0 * M_PI};
    matmul_ctx_t *ctx = matmul_ctx_alloc(2, N, L);
    matmul_ctx_set_system(ctx, 0);

    const uint64_t N0_k = 5;
    ASSERT(ctx->k_size == 30);
    ASSERT_CLOSE(ctx->k2[0], 0.0, TOL);
    ASSERT_CLOSE(ctx->k2[1], 1.0, TOL);
    ASSERT_CLOSE(ctx->k2[1 * N0_k], 1.0, TOL);
    ASSERT_CLOSE(ctx->k2[1 * N0_k + 1], 2.0, TOL);
    ASSERT_CLOSE(ctx->k2[4 * N0_k], 4.0, TOL);

    matmul_ctx_free(&ctx);
}

TEST(k2_2d_c2c) {
    const uint64_t N[] = {8, 6};
    const f64    L[] = {2.0 * M_PI, 2.0 * M_PI};
    matmul_ctx_t *ctx = matmul_ctx_alloc(2, N, L);
    matmul_ctx_set_system(ctx, 1);

    const uint64_t N0_k = 8;
    ASSERT(ctx->k_size == 48);
    ASSERT_CLOSE(ctx->k2[0], 0.0, TOL);
    ASSERT_CLOSE(ctx->k2[1], 1.0, TOL);
    ASSERT_CLOSE(ctx->k2[1 * N0_k], 1.0, TOL);
    ASSERT_CLOSE(ctx->k2[1 * N0_k + 1], 2.0, TOL);

    matmul_ctx_free(&ctx);
}

TEST(k2_3d) {
    const uint64_t N[] = {8, 8, 8};
    const f64 L[] = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};
    {
        matmul_ctx_t *ctx = matmul_ctx_alloc(3, N, L);
        matmul_ctx_set_system(ctx, 0);

        const uint64_t N0_k = 5;
        ASSERT(ctx->k_size == 320);
        ASSERT_CLOSE(ctx->k2[0], 0.0, TOL);
        ASSERT_CLOSE(ctx->k2[1], 1.0, TOL);
        ASSERT_CLOSE(ctx->k2[1 * N0_k], 1.0, TOL);
        ASSERT_CLOSE(ctx->k2[1 * 8 * N0_k], 1.0, TOL);

        const uint64_t idx_111 = 1 * 8 * N0_k + 1 * N0_k + 1;
        ASSERT_CLOSE(ctx->k2[idx_111], 3.0, TOL);

        const uint64_t idx_235 = 5 * 8 * N0_k + 3 * N0_k + 2;
        ASSERT_CLOSE(ctx->k2[idx_235], 22.0, TOL);

        matmul_ctx_free(&ctx);
    }
}

TEST(fftw_roundtrip_c2c_2d) {
    const uint64_t N[] = {16, 8};
    const f64 L[] = {1.0, 1.0};
    matmul_ctx_t *ctx = matmul_ctx_alloc(2, N, L);
    matmul_ctx_set_system(ctx, 1);

    const uint64_t size = ctx->size;
    c64 *in  = (c64 *)ctx->c_wrk1;
    c64 *wrk = (c64 *)ctx->f_wrk;

    for (uint64_t i = 0; i < size; i++)
        in[i] = (f64)i + I * (f64)(size - i);

    c64 *saved = xcalloc(size, sizeof(c64));
    memcpy(saved, in, size * sizeof(c64));

    fftw_execute_dft(ctx->fwd_plan,
                     (fftw_complex *)in, (fftw_complex *)wrk);
    fftw_execute_dft(ctx->bwd_plan,
                     (fftw_complex *)wrk, (fftw_complex *)in);

    int ok = 1;
    for (uint64_t i = 0; i < size; i++) {
        const f64 re_err = fabs(creal(in[i]) / (f64)size - creal(saved[i]));
        const f64 im_err = fabs(cimag(in[i]) / (f64)size - cimag(saved[i]));
        if (re_err > 1e-10 || im_err > 1e-10) { ok = 0; break; }
    }
    ASSERT(ok);

    safe_free((void **)&saved);
    matmul_ctx_free(&ctx);
}

TEST(fftw_roundtrip_r2c_1d) {
    const uint64_t N[] = {32};
    const f64 L[] = {1.0};
    matmul_ctx_t *ctx = matmul_ctx_alloc(1, N, L);
    matmul_ctx_set_system(ctx, 0);

    const uint64_t size = ctx->size;
    f64 *in = (f64 *)ctx->c_wrk1;
    fftw_complex *wrk = (fftw_complex *)ctx->f_wrk;

    for (uint64_t i = 0; i < size; i++)
        in[i] = sin(2.0 * M_PI * (f64)i / (f64)size) + 0.5;

    f64 *saved = xcalloc(size, sizeof(f64));
    memcpy(saved, in, size * sizeof(f64));

    fftw_execute_dft_r2c(ctx->fwd_plan, in, wrk);
    fftw_execute_dft_c2r(ctx->bwd_plan, wrk, in);

    int ok = 1;
    for (uint64_t i = 0; i < size; i++) {
        if (fabs(in[i] / (f64)size - saved[i]) > 1e-10) { ok = 0; break; }
    }
    ASSERT(ok);

    safe_free((void **)&saved);
    matmul_ctx_free(&ctx);
}

TEST(fftw_roundtrip_r2c_3d) {
    const uint64_t N[] = {8, 6, 4};
    const f64 L[] = {1.0, 2.0, 3.0};
    matmul_ctx_t *ctx = matmul_ctx_alloc(3, N, L);
    matmul_ctx_set_system(ctx, 0);

    const uint64_t size = ctx->size;
    f64 *in = (f64 *)ctx->c_wrk1;
    fftw_complex *wrk = (fftw_complex *)ctx->f_wrk;

    for (uint64_t i = 0; i < size; i++)
        in[i] = (f64)(i % 7) - 3.0;

    f64 *saved = xcalloc(size, sizeof(f64));
    memcpy(saved, in, size * sizeof(f64));

    fftw_execute_dft_r2c(ctx->fwd_plan, in, wrk);
    fftw_execute_dft_c2r(ctx->bwd_plan, wrk, in);

    int ok = 1;
    for (uint64_t i = 0; i < size; i++) {
        if (fabs(in[i] / (f64)size - saved[i]) > 1e-10) { ok = 0; break; }
    }
    ASSERT(ok);

    safe_free((void **)&saved);
    matmul_ctx_free(&ctx);
}

/* ================================================================
 * Setup tests — trap, wavefunction, interactions, mu
 * ================================================================ */

/* --- Helper: harmonic trap V(r) = 0.5 * omega^2 * r^2 --- */
static f64 harmonic_trap(uint64_t dim, const f64 *r, void *param) {
    const f64 omega = *(const f64 *)param;
    f64 r2 = 0.0;
    for (uint64_t d = 0; d < dim; d++)
        r2 += r[d] * r[d];
    return 0.5 * omega * omega * r2;
}

/* ----------------------------------------------------------------
 * test_trap_harmonic_1d: verify localTermK/M at each grid point
 * ---------------------------------------------------------------- */
TEST(trap_harmonic_1d) {
    const uint64_t N[] = {16};
    const f64    L[] = {8.0};
    f64 omega = 1.0;

    bdg_t *bdg = bdg_alloc(1, N, L, 0);
    bdg_set_system(bdg);
    bdg_set_trap(bdg, harmonic_trap, &omega);

    const matmul_ctx_t *ctx = bdg->ctx;
    for (uint64_t ix = 0; ix < 16; ix++) {
        const f64 x = ((f64)ix - 8.0) * 8.0 / 16.0;
        const f64 expected = 0.5 * x * x;
        ASSERT_CLOSE(ctx->localTermK[ix], expected, TOL);
        ASSERT_CLOSE(ctx->localTermM[ix], expected, TOL);
    }

    /* Verify state flag */
    ASSERT(bdg->state & BDG_HAS_TRAP);

    bdg_free(&bdg);
}

/* ----------------------------------------------------------------
 * test_trap_harmonic_2d: spot-check center and corners
 * ---------------------------------------------------------------- */
TEST(trap_harmonic_2d) {
    const uint64_t N[] = {8, 6};
    const f64    L[] = {4.0, 3.0};
    f64 omega = 2.0;

    bdg_t *bdg = bdg_alloc(2, N, L, 0);
    bdg_set_system(bdg);
    bdg_set_trap(bdg, harmonic_trap, &omega);

    const matmul_ctx_t *ctx = bdg->ctx;
    const uint64_t Nx = 8;

    /* Check all grid points */
    for (uint64_t iy = 0; iy < 6; iy++) {
        const f64 y = ((f64)iy - 3.0) * 3.0 / 6.0;
        for (uint64_t ix = 0; ix < 8; ix++) {
            const f64 x = ((f64)ix - 4.0) * 4.0 / 8.0;
            const f64 expected = 0.5 * 4.0 * (x * x + y * y);
            const uint64_t idx = iy * Nx + ix;
            ASSERT_CLOSE(ctx->localTermK[idx], expected, 1e-10);
            ASSERT_CLOSE(ctx->localTermM[idx], expected, 1e-10);
        }
    }

    bdg_free(&bdg);
}

/* ----------------------------------------------------------------
 * test_wavefunction_real: copy + verify f64 array
 * ---------------------------------------------------------------- */
TEST(wavefunction_real) {
    const uint64_t N[] = {8};
    const f64    L[] = {1.0};

    bdg_t *bdg = bdg_alloc(1, N, L, 0);
    bdg_set_system(bdg);

    f64 wf[8];
    for (uint64_t i = 0; i < 8; i++)
        wf[i] = sin(2.0 * M_PI * (f64)i / 8.0);

    bdg_set_wavefunction(bdg, wf, 8);

    const f64 *stored = (const f64 *)bdg->ctx->wf;
    for (uint64_t i = 0; i < 8; i++)
        ASSERT_CLOSE(stored[i], wf[i], TOL);

    ASSERT(bdg->state & BDG_HAS_WF);
    ASSERT(bdg->ctx->wf_size == 8);

    bdg_free(&bdg);
}

/* ----------------------------------------------------------------
 * test_wavefunction_complex: copy + verify c64 array
 * ---------------------------------------------------------------- */
TEST(wavefunction_complex) {
    const uint64_t N[] = {8};
    const f64    L[] = {1.0};

    bdg_t *bdg = bdg_alloc(1, N, L, 1);
    bdg_set_system(bdg);

    c64 wf[8];
    for (uint64_t i = 0; i < 8; i++)
        wf[i] = (f64)i + I * (f64)(8 - i);

    bdg_set_wavefunction(bdg, wf, 8);

    const c64 *stored = (const c64 *)bdg->ctx->wf;
    for (uint64_t i = 0; i < 8; i++) {
        ASSERT_CLOSE(creal(stored[i]), creal(wf[i]), TOL);
        ASSERT_CLOSE(cimag(stored[i]), cimag(wf[i]), TOL);
    }

    ASSERT(bdg->state & BDG_HAS_WF);

    bdg_free(&bdg);
}

/* ----------------------------------------------------------------
 * test_local_interactions_contact:
 * Uniform wf, contact U_intK = U_intM = g*n.
 * K gets g*n, M gets g*n + 2*g*n = 3*g*n.
 * ---------------------------------------------------------------- */
static f64 contact_int(void *param, f64 density) {
    const f64 g = *(const f64 *)param;
    return g * density;
}

TEST(local_interactions_contact) {
    const uint64_t N[] = {16};
    const f64    L[] = {1.0};
    const f64 g = 2.5;
    const f64 psi_val = 0.7;
    const f64 n = psi_val * psi_val;

    bdg_t *bdg = bdg_alloc(1, N, L, 0);
    bdg_set_system(bdg);

    /* Uniform wavefunction */
    f64 wf[16];
    for (uint64_t i = 0; i < 16; i++)
        wf[i] = psi_val;
    bdg_set_wavefunction(bdg, wf, 16);

    f64 g_param = g;
    bdg_set_local_interactions(bdg, contact_int, contact_int, &g_param);

    const matmul_ctx_t *ctx = bdg->ctx;
    for (uint64_t i = 0; i < 16; i++) {
        ASSERT_CLOSE(ctx->localTermK[i], g * n, 1e-14);
        ASSERT_CLOSE(ctx->localTermM[i], 3.0 * g * n, 1e-14);
    }

    ASSERT(bdg->state & BDG_HAS_INTERACTIONS);

    bdg_free(&bdg);
}

/* ----------------------------------------------------------------
 * test_set_mu: full pipeline alloc → system → trap → wf →
 *              interactions → mu. Verify subtraction + preconditioner.
 * ---------------------------------------------------------------- */
TEST(set_mu) {
    const uint64_t N[] = {8};
    const f64    L[] = {4.0};
    const f64 g = 1.0;
    const f64 psi_val = 1.0;
    const f64 mu = 3.0;
    f64 omega = 1.0;

    bdg_t *bdg = bdg_alloc(1, N, L, 0);
    bdg_set_system(bdg);

    /* Trap */
    bdg_set_trap(bdg, harmonic_trap, &omega);

    /* Uniform wavefunction */
    f64 wf[8];
    for (uint64_t i = 0; i < 8; i++)
        wf[i] = psi_val;
    bdg_set_wavefunction(bdg, wf, 8);

    /* Contact interactions */
    f64 g_param = g;
    bdg_set_local_interactions(bdg, contact_int, contact_int, &g_param);

    /* Snapshot localTermK/M before mu subtraction (for preconditioner check) */
    f64 ltK_before[8], ltM_before[8];
    memcpy(ltK_before, bdg->ctx->localTermK, 8 * sizeof(f64));
    memcpy(ltM_before, bdg->ctx->localTermM, 8 * sizeof(f64));

    bdg_set_mu(bdg, mu);

    const matmul_ctx_t *ctx = bdg->ctx;

    for (uint64_t i = 0; i < 8; i++) {
        /* Check mu was subtracted */
        ASSERT_CLOSE(ctx->localTermK[i], ltK_before[i] - mu, 1e-14);
        ASSERT_CLOSE(ctx->localTermM[i], ltM_before[i] - mu, 1e-14);

        /* Check preconditioner: 1/sqrt(|safe_val(ltK_before)|) */
        const f64 valK = fabs(ltK_before[i]) < 1e-8 ? 1e-8 : ltK_before[i];
        const f64 valM = fabs(ltM_before[i]) < 1e-8 ? 1e-8 : ltM_before[i];
        const f64 expK = 1.0 / sqrt(fabs(valK));
        const f64 expM = 1.0 / sqrt(fabs(valM));
        ASSERT_CLOSE(ctx->precond_sqrtK[i], expK, 1e-14);
        ASSERT_CLOSE(ctx->precond_sqrtM[i], expM, 1e-14);
    }

    ASSERT_CLOSE(ctx->mu, mu, TOL);
    ASSERT(bdg->state & BDG_HAS_MU);

    bdg_free(&bdg);
}

/* ----------------------------------------------------------------
 * test_trap_additive: calling set_trap twice adds both potentials
 * ---------------------------------------------------------------- */
TEST(trap_additive) {
    const uint64_t N[] = {8};
    const f64    L[] = {4.0};
    f64 omega1 = 1.0;
    f64 omega2 = 2.0;

    bdg_t *bdg = bdg_alloc(1, N, L, 0);
    bdg_set_system(bdg);
    bdg_set_trap(bdg, harmonic_trap, &omega1);
    bdg_set_trap(bdg, harmonic_trap, &omega2);

    const matmul_ctx_t *ctx = bdg->ctx;
    for (uint64_t ix = 0; ix < 8; ix++) {
        const f64 x = ((f64)ix - 4.0) * 4.0 / 8.0;
        const f64 expected = 0.5 * (1.0 + 4.0) * x * x;  /* omega1^2 + omega2^2 */
        ASSERT_CLOSE(ctx->localTermK[ix], expected, 1e-10);
    }

    bdg_free(&bdg);
}

/* ----------------------------------------------------------------
 * test_state_validation: fork-based tests for out-of-order calls.
 * Only tests paths that DON'T call bdg_set_system (FFTW+fork=deadlock).
 * ---------------------------------------------------------------- */

/* Run fn() in a child; return 1 if child exits with EXIT_FAILURE */
static int expect_exit_failure(void (*fn)(void)) {
    fflush(stdout);
    fflush(stderr);
    const pid_t pid = fork();
    if (0 == pid) {
        /* Child: redirect stderr to /dev/null to suppress error output */
        FILE *devnull = freopen("/dev/null", "w", stderr);
        (void)devnull;
        fn();
        _exit(0);  /* If we reach here, the check didn't fire */
    }
    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status) && EXIT_FAILURE == WEXITSTATUS(status))
        return 1;
    return 0;
}

/* Try bdg_set_trap without bdg_set_system → should fail */
static void call_trap_without_system(void) {
    const uint64_t N[] = {8};
    const f64    L[] = {1.0};
    f64 omega = 1.0;
    bdg_t *bdg = bdg_alloc(1, N, L, 0);
    /* Skip bdg_set_system */
    bdg_set_trap(bdg, harmonic_trap, &omega);
    bdg_free(&bdg);
}

/* Try bdg_set_wavefunction without bdg_set_system → should fail */
static void call_wf_without_system(void) {
    const uint64_t N[] = {8};
    const f64    L[] = {1.0};
    f64 wf[8] = {0};
    bdg_t *bdg = bdg_alloc(1, N, L, 0);
    /* Skip bdg_set_system */
    bdg_set_wavefunction(bdg, wf, 8);
    bdg_free(&bdg);
}

/* Try bdg_set_local_interactions without wavefunction → should fail.
 * NOTE: We CAN call bdg_set_system here because the child process
 * has a fresh address space. The FFTW deadlock risk is from fork()
 * AFTER fftw_init_threads in the PARENT. Since we fork first and
 * then call set_system in the child, this is safe. */
static void call_interactions_without_wf(void) {
    const uint64_t N[] = {4};
    const f64    L[] = {1.0};
    f64 g = 1.0;
    bdg_t *bdg = bdg_alloc(1, N, L, 0);
    bdg_set_system(bdg);
    /* Skip bdg_set_wavefunction */
    bdg_set_local_interactions(bdg, contact_int, contact_int, &g);
    bdg_free(&bdg);
}

TEST(state_validation) {
    ASSERT(expect_exit_failure(call_trap_without_system));
    ASSERT(expect_exit_failure(call_wf_without_system));
    ASSERT(expect_exit_failure(call_interactions_without_wf));
}

/* ================================================================
 * Reset tests
 * ================================================================ */

/* ----------------------------------------------------------------
 * test_reset_and_rerun: full pipeline → reset → verify clean →
 * re-run with different params → verify new values
 * ---------------------------------------------------------------- */
TEST(reset_and_rerun) {
    const uint64_t N[] = {8};
    const f64    L[] = {4.0};

    bdg_t *bdg = bdg_alloc(1, N, L, 0);
    bdg_set_system(bdg);

    /* === First run: omega=1, g=1, psi=1, mu=2 === */
    {
        f64 omega = 1.0;
        bdg_set_trap(bdg, harmonic_trap, &omega);

        f64 wf[8];
        for (uint64_t i = 0; i < 8; i++) wf[i] = 1.0;
        bdg_set_wavefunction(bdg, wf, 8);

        f64 g = 1.0;
        bdg_set_local_interactions(bdg, contact_int, contact_int, &g);
        bdg_set_mu(bdg, 2.0);

        /* Confirm pipeline completed */
        ASSERT(bdg->state & BDG_HAS_TRAP);
        ASSERT(bdg->state & BDG_HAS_WF);
        ASSERT(bdg->state & BDG_HAS_INTERACTIONS);
        ASSERT(bdg->state & BDG_HAS_MU);
    }

    /* === Reset === */
    bdg_reset(bdg);

    /* Verify cleared state */
    ASSERT(bdg->state == BDG_HAS_SYSTEM);
    ASSERT(NULL == bdg->ctx->wf);
    ASSERT(NULL == bdg->ctx->longRngInt);
    ASSERT(NULL == bdg->ctx->precond_sqrtK);
    ASSERT(NULL == bdg->ctx->precond_sqrtM);
    ASSERT(0 == bdg->ctx->wf_size);
    ASSERT_CLOSE(bdg->ctx->mu, 0.0, TOL);
    ASSERT_CLOSE(bdg->ctx->g_ddi, 0.0, TOL);
    ASSERT(0 == bdg->ctx->dipolar);
    ASSERT(0 == bdg->converged);
    ASSERT(NULL == bdg->eigvals);
    ASSERT(NULL == bdg->modes_u);
    ASSERT(NULL == bdg->modes_v);

    /* Verify localTermK/M zeroed */
    for (uint64_t i = 0; i < 8; i++) {
        ASSERT_CLOSE(bdg->ctx->localTermK[i], 0.0, TOL);
        ASSERT_CLOSE(bdg->ctx->localTermM[i], 0.0, TOL);
    }

    /* Verify grid/k-space survived */
    ASSERT(bdg->ctx->k2 != NULL);
    ASSERT(bdg->ctx->size == 8);

    /* === Second run: omega=3, g=5, psi=0.5, mu=1 === */
    {
        f64 omega2 = 3.0;
        bdg_set_trap(bdg, harmonic_trap, &omega2);

        f64 wf2[8];
        for (uint64_t i = 0; i < 8; i++) wf2[i] = 0.5;
        bdg_set_wavefunction(bdg, wf2, 8);

        /* BDG_FORBID should NOT fire — interactions flag was cleared */
        f64 g2 = 5.0;
        bdg_set_local_interactions(bdg, contact_int, contact_int, &g2);
        bdg_set_mu(bdg, 1.0);
    }

    /* Verify second-run values are correct and different from first */
    const matmul_ctx_t *ctx = bdg->ctx;
    const f64 n2 = 0.25;  /* 0.5^2 */
    const f64 g2 = 5.0;
    const f64 mu2 = 1.0;
    const f64 omega2 = 3.0;

    for (uint64_t ix = 0; ix < 8; ix++) {
        const f64 x = ((f64)ix - 4.0) * 4.0 / 8.0;
        const f64 V = 0.5 * omega2 * omega2 * x * x;
        /* K = V + g*n - mu;  M = V + 3*g*n - mu */
        const f64 expK = V + g2 * n2 - mu2;
        const f64 expM = V + 3.0 * g2 * n2 - mu2;
        ASSERT_CLOSE(ctx->localTermK[ix], expK, 1e-12);
        ASSERT_CLOSE(ctx->localTermM[ix], expM, 1e-12);
    }

    ASSERT_CLOSE(ctx->mu, mu2, TOL);
    ASSERT(bdg->state & BDG_HAS_MU);

    /* Verify wf was set correctly */
    const f64 *wf_stored = (const f64 *)ctx->wf;
    for (uint64_t i = 0; i < 8; i++)
        ASSERT_CLOSE(wf_stored[i], 0.5, TOL);

    bdg_free(&bdg);
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

    printf("\nSetup functions:\n");
    RUN(trap_harmonic_1d);
    RUN(trap_harmonic_2d);
    RUN(wavefunction_real);
    RUN(wavefunction_complex);
    RUN(local_interactions_contact);
    RUN(set_mu);
    RUN(trap_additive);

    printf("\nReset:\n");
    RUN(reset_and_rerun);

    printf("\nState validation (fork-based):\n");
    RUN(state_validation);

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
