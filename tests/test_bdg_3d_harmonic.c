#include "bdg/bdg.h"
#include "lobpcg/types.h"
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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

/* ── trap callback ────────────────────────────────────────── */

static f64 V_harmonic(uint64_t dim, const f64 *r, void *param) {
  (void)param;
  f64 rsq = 0.0;
  for (uint64_t d = 0; d < dim; d++)
    rsq += r[d] * r[d];
  return 0.5 * rsq;
}

/* ── Test: 3D isotropic harmonic oscillator BdG spectrum ── */

TEST(bdg_3d_harmonic_excitations) {
  /*
   * Non-interacting 3D isotropic harmonic oscillator, omega = 1.
   *
   * Ground state energy: E0 = 0.5*(wx + wy + wz) = 1.5
   * Chemical potential:  mu = E0 = 1.5
   *
   * BdG excitation energies = E_n - mu.
   *   - Goldstone-like mode at ~0 (from ground state)
   *   - First excited: omega = 1.0 (3-fold degenerate: nx=1, ny=1, nz=1)
   */
  const uint64_t N[3] = {64, 64, 64};
  const f64 L[3] = {10.0, 10.0, 10.0};
  const f64 mu = 1.5;
  const uint64_t nev = 10;
  const uint64_t sizeSub = 20;
  const uint64_t maxIter = 500;
  const f64 tol = 1e-4;

  /* Setup: no wavefunction, no interactions, no dipolar */
  bdg_t *bdg = bdg_alloc(3, N, L, 0);
  bdg_set_system(bdg);
  bdg_set_trap(bdg, V_harmonic, NULL);
  bdg_set_mu(bdg, mu);
  bdg_set_solver_params(bdg, nev, sizeSub, maxIter, tol);

  /* Solve */
  const int ret = bdg_solve(bdg);
  printf("\n    bdg_solve returned: %d\n", ret);
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

  /* Goldstone mode: eigenvalue[0] ~ 0 */
  printf("    evals[0] = %.10f (expect ~0)\n", evals[0]);
  ASSERT_NEAR(evals[0], 0.0, 0.1);

  /* First excited: eigenvalue[1..3] ~ 1.0 (3-fold degenerate) */
  const f64 omega_expected = 1.0;
  const f64 rel_tol = 0.05;
  for (uint64_t j = 1; j < nev; j++) {
    printf("    evals[%lu] = %.10f (expect ~%.1f, tol=%.0f%%)\n",
           (unsigned long)j, evals[j], omega_expected, rel_tol * 100.0);
    ASSERT_NEAR(evals[j], omega_expected, rel_tol * omega_expected);
  }

  bdg_free(&bdg);
}

/* ── main ──────────────────────────────────────────────────── */

int main(void) {
  printf("3D harmonic oscillator BdG excitations:\n");
  RUN(bdg_3d_harmonic_excitations);

  printf("\n========================================\n");
  printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
  printf("========================================\n");

  return tests_failed > 0 ? 1 : 0;
}
