#include "bdg/bdg.h"
#include "lobpcg/types.h"
#include <complex.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) static void test_##name(void)

#define RUN(name) do { \
    printf("  %-50s ", #name); \
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

#define ASSERT_NEAR(a, b, tol) do { \
    const double _a = (a), _b = (b); \
    if (fabs(_a - _b) >= (tol)) { \
        printf("[FAIL] line %d: %.15e vs %.15e (diff %.3e)\n", \
               __LINE__, _a, _b, fabs(_a - _b)); \
        tests_failed++; \
        return; \
    } \
} while(0)

/* ── interaction callbacks ────────────────────────────────── */

static const f64 g_contact = 1.0;

static f64 U_intK_cb(void *param, f64 density) {
    (void)param;
    return g_contact * density;
}

static f64 U_intM_cb(void *param, f64 density) {
    (void)param;
    return g_contact * density;
}

/* ── Test 1: real-valued uniform BEC Bogoliubov spectrum ── */

TEST(bdg_1d_uniform_d_bogoliubov_spectrum) {
    const uint64_t N = 64;
    const f64 L = 2.0 * M_PI;
    const f64 n0 = 1.0;
    const f64 mu = g_contact * n0;  /* = 1.0 */
    const uint64_t nev = 4;

    /* Analytical Bogoliubov spectrum:
     *   ek = k^2 / 2
     *   E(k) = sqrt( ek * (ek + 2*g*n0) )
     * k=0:    E(0) = 0 (Goldstein mode)
     * k=+/-1: E(1) = sqrt(0.5 * 2.5) = sqrt(1.25) ~ 1.118
     * k=+/-2: E(2) = sqrt(2.0 * 4.0) = sqrt(8) ~ 2.828
     */
    const f64 ek1 = 0.5;
    const f64 ek2 = 2.0;
    const f64 E1 = sqrt(ek1 * (ek1 + 2.0 * g_contact * n0));
    const f64 E2 = sqrt(ek2 * (ek2 + 2.0 * g_contact * n0));

    printf("\n    analytical: E(k=0)=0, E(k=1)=%.10f, E(k=2)=%.10f\n", E1, E2);

    /* Setup */
    bdg_t *bdg = bdg_alloc(1, &N, &L, 0);
    bdg_set_system(bdg);
    bdg_set_solver_params(bdg, nev, 10, 200, 1e-4);

    f64 *wf = calloc(N, sizeof(f64));
    ASSERT(NULL != wf);
    const f64 sqrt_n0 = sqrt(n0);
    for (uint64_t i = 0; i < N; i++)
        wf[i] = sqrt_n0;

    bdg_set_wavefunction(bdg, wf, N);
    free(wf);

    bdg_set_local_interactions(bdg, U_intK_cb, U_intM_cb, NULL);
    bdg_set_mu(bdg, mu);

    /* Solve */
    const int ret = bdg_solve(bdg);
    ASSERT(0 == ret);

    const uint64_t nconv = bdg_converged(bdg);
    printf("    converged: %" PRIu64 " / %" PRIu64 "\n", nconv, nev);
    ASSERT(nconv >= nev);

    const f64 *evals = bdg_eigenvalues(bdg);
    ASSERT(NULL != evals);

    printf("    eigenvalues:");
    for (uint64_t j = 0; j < nev; j++)
        printf(" %.10f", evals[j]);
    printf("\n");

    /* Count how many eigenvalues match E1 and E2 within 1% relative tol */
    const f64 rel_tol = 0.01;
    int count_E1 = 0;
    int count_E2 = 0;
    for (uint64_t j = 0; j < nev; j++) {
        if (fabs(evals[j] - E1) < rel_tol * E1)
            count_E1++;
        if (fabs(evals[j] - E2) < rel_tol * E2)
            count_E2++;
    }

    printf("    E1 matches: %d (expect 2), E2 matches: %d (expect >= 1)\n",
           count_E1, count_E2);

    ASSERT(2 == count_E1);
    ASSERT(count_E2 >= 1);

    /* Verify E1 values precisely */
    for (uint64_t j = 0; j < nev; j++) {
        if (fabs(evals[j] - E1) < rel_tol * E1)
            ASSERT_NEAR(evals[j], E1, rel_tol * E1);
    }

    bdg_free(&bdg);
}

/* ── Test 2: complex path matches real path ───────────────── */

TEST(bdg_1d_uniform_z_matches_d) {
    const uint64_t N = 64;
    const f64 L = 2.0 * M_PI;
    const f64 n0 = 1.0;
    const f64 mu = g_contact * n0;
    const uint64_t nev = 4;

    const f64 ek1 = 0.5;
    const f64 E1 = sqrt(ek1 * (ek1 + 2.0 * g_contact * n0));
    const f64 rel_tol = 0.01;

    /* --- Solve real path --- */
    bdg_t *bdg_d = bdg_alloc(1, &N, &L, 0);
    bdg_set_system(bdg_d);
    bdg_set_solver_params(bdg_d, nev, 10, 200, 1e-4);

    f64 *wf_d = calloc(N, sizeof(f64));
    ASSERT(NULL != wf_d);
    const f64 sqrt_n0 = sqrt(n0);
    for (uint64_t i = 0; i < N; i++)
        wf_d[i] = sqrt_n0;
    bdg_set_wavefunction(bdg_d, wf_d, N);
    free(wf_d);

    bdg_set_local_interactions(bdg_d, U_intK_cb, U_intM_cb, NULL);
    bdg_set_mu(bdg_d, mu);

    const int ret_d = bdg_solve(bdg_d);
    ASSERT(0 == ret_d);
    ASSERT(bdg_converged(bdg_d) >= nev);

    const f64 *evals_d = bdg_eigenvalues(bdg_d);
    ASSERT(NULL != evals_d);

    /* --- Solve complex path --- */
    bdg_t *bdg_z = bdg_alloc(1, &N, &L, 1);
    bdg_set_system(bdg_z);
    bdg_set_solver_params(bdg_z, nev, 10, 200, 1e-4);

    c64 *wf_z = calloc(N, sizeof(c64));
    ASSERT(NULL != wf_z);
    for (uint64_t i = 0; i < N; i++)
        wf_z[i] = sqrt_n0 + 0.0 * I;
    bdg_set_wavefunction(bdg_z, wf_z, N);
    free(wf_z);

    bdg_set_local_interactions(bdg_z, U_intK_cb, U_intM_cb, NULL);
    bdg_set_mu(bdg_z, mu);

    const int ret_z = bdg_solve(bdg_z);
    ASSERT(0 == ret_z);
    ASSERT(bdg_converged(bdg_z) >= nev);

    const f64 *evals_z = bdg_eigenvalues(bdg_z);
    ASSERT(NULL != evals_z);

    printf("\n    real path:    ");
    for (uint64_t j = 0; j < nev; j++)
        printf(" %.10f", evals_d[j]);
    printf("\n    complex path: ");
    for (uint64_t j = 0; j < nev; j++)
        printf(" %.10f", evals_z[j]);
    printf("\n");

    /* Both paths must find E1 at least twice */
    int d_E1 = 0, z_E1 = 0;
    for (uint64_t j = 0; j < nev; j++) {
        if (fabs(evals_d[j] - E1) < rel_tol * E1) d_E1++;
        if (fabs(evals_z[j] - E1) < rel_tol * E1) z_E1++;
    }
    ASSERT(d_E1 >= 2);
    ASSERT(z_E1 >= 2);

    /* Find the first E1-like eigenvalue in each path and compare */
    const f64 abs_tol = 1e-4;
    int d_idx = -1, z_idx = -1;
    for (uint64_t j = 0; j < nev; j++) {
        if (-1 == d_idx && fabs(evals_d[j] - E1) < rel_tol * E1) d_idx = (int)j;
        if (-1 == z_idx && fabs(evals_z[j] - E1) < rel_tol * E1) z_idx = (int)j;
    }
    ASSERT(d_idx >= 0);
    ASSERT(z_idx >= 0);
    ASSERT_NEAR(evals_d[d_idx], evals_z[z_idx], abs_tol);

    bdg_free(&bdg_d);
    bdg_free(&bdg_z);
}

/* ── main ──────────────────────────────────────────────────── */

int main(void) {
    printf("1D uniform BEC Bogoliubov spectrum (real):\n");
    RUN(bdg_1d_uniform_d_bogoliubov_spectrum);

    printf("\n1D uniform BEC: complex matches real:\n");
    RUN(bdg_1d_uniform_z_matches_d);

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
