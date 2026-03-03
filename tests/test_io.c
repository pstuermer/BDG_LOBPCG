#include "bdg_internal.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <complex.h>
#include <stdint.h>

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
 * Task 4: bdg_load_wavefunction tests
 * ================================================================ */

TEST(load_wf_real) {
  const char *fname = "/tmp/test_load_wf_real.dat";
  FILE *f = fopen(fname, "w");
  ASSERT(NULL != f);
  fprintf(f, "1.0 0.0\n");
  fprintf(f, "2.0 0.0\n");
  fprintf(f, "3.0 0.0\n");
  fprintf(f, "4.0 0.0\n");
  fclose(f);

  const size_t N[] = {4};
  const f64 L[] = {1.0};
  bdg_t *bdg = bdg_alloc(1, N, L, 0);
  bdg_set_system(bdg);

  const int rc = bdg_load_wavefunction(bdg, fname);
  ASSERT(0 == rc);
  ASSERT(bdg->state & BDG_HAS_WF);

  const f64 *wf = (const f64 *)bdg->ctx->wf;
  ASSERT_CLOSE(wf[0], 1.0, TOL);
  ASSERT_CLOSE(wf[1], 2.0, TOL);
  ASSERT_CLOSE(wf[2], 3.0, TOL);
  ASSERT_CLOSE(wf[3], 4.0, TOL);

  bdg_free(&bdg);
  unlink(fname);
}

TEST(load_wf_complex) {
  const char *fname = "/tmp/test_load_wf_complex.dat";
  FILE *f = fopen(fname, "w");
  ASSERT(NULL != f);
  fprintf(f, "1.0 0.5\n");
  fprintf(f, "2.0 -0.5\n");
  fclose(f);

  const size_t N[] = {2};
  const f64 L[] = {1.0};
  bdg_t *bdg = bdg_alloc(1, N, L, 1);
  bdg_set_system(bdg);

  const int rc = bdg_load_wavefunction(bdg, fname);
  ASSERT(0 == rc);
  ASSERT(bdg->state & BDG_HAS_WF);

  const c64 *wf = (const c64 *)bdg->ctx->wf;
  ASSERT_CLOSE(creal(wf[0]), 1.0, TOL);
  ASSERT_CLOSE(cimag(wf[0]), 0.5, TOL);
  ASSERT_CLOSE(creal(wf[1]), 2.0, TOL);
  ASSERT_CLOSE(cimag(wf[1]), -0.5, TOL);

  bdg_free(&bdg);
  unlink(fname);
}

TEST(load_wf_file_not_found) {
  const size_t N[] = {4};
  const f64 L[] = {1.0};
  bdg_t *bdg = bdg_alloc(1, N, L, 0);
  bdg_set_system(bdg);

  const int rc = bdg_load_wavefunction(bdg, "/tmp/no_such_file_12345.dat");
  ASSERT(-1 == rc);

  bdg_free(&bdg);
}

/* ================================================================
 * Task 5: bdg_write_eigenvalues tests
 * ================================================================ */

TEST(write_eigenvalues) {
  const char *fname = "/tmp/test_write_eigvals.dat";
  unlink(fname);

  const size_t N[] = {4};
  const f64 L[] = {1.0};
  bdg_t *bdg = bdg_alloc(1, N, L, 0);
  bdg_set_system(bdg);
  bdg_set_solver_params(bdg, 3, 6, 100, 1e-6);

  /* Manually inject fake eigenvalues */
  bdg->eigvals = xcalloc(3, sizeof(f64));
  bdg->eigvals[0] = 1.0;
  bdg->eigvals[1] = 2.0;
  bdg->eigvals[2] = 3.0;

  /* Write twice (append mode) */
  int rc = bdg_write_eigenvalues(bdg, fname);
  ASSERT(0 == rc);
  rc = bdg_write_eigenvalues(bdg, fname);
  ASSERT(0 == rc);

  /* Read back and verify */
  FILE *f = fopen(fname, "r");
  ASSERT(NULL != f);

  for (int line = 0; line < 2; line++) {
    f64 v0, v1, v2;
    const int n = fscanf(f, "%lf %lf %lf", &v0, &v1, &v2);
    ASSERT(3 == n);
    ASSERT_CLOSE(v0, 1.0, 1e-6);
    ASSERT_CLOSE(v1, 2.0, 1e-6);
    ASSERT_CLOSE(v2, 3.0, 1e-6);
  }
  fclose(f);

  /* Clean up manually-set eigvals before bdg_free */
  safe_free((void **)&bdg->eigvals);
  bdg_free(&bdg);
  unlink(fname);
}

/* ================================================================
 * Task 6: bdg_write_mode_u/v tests
 * ================================================================ */

TEST(write_mode_1d_real) {
  const char *fname_base = "/tmp/test_mode_1d";
  const char *fname_u = "/tmp/test_mode_1d_u_0.dat";

  const size_t N[] = {4};
  const f64 L[] = {1.0};
  bdg_t *bdg = bdg_alloc(1, N, L, 0);
  bdg_set_system(bdg);
  bdg_set_solver_params(bdg, 1, 2, 100, 1e-6);

  /* Manually inject fake mode */
  bdg->modes_u = xcalloc(4, sizeof(f64));
  f64 *u = (f64 *)bdg->modes_u;
  u[0] = 1.0; u[1] = 2.0; u[2] = 3.0; u[3] = 4.0;

  const int rc = bdg_write_mode_u(bdg, 0, fname_base);
  ASSERT(0 == rc);

  /* Read back and verify 4 lines of "real imag" pairs */
  FILE *f = fopen(fname_u, "r");
  ASSERT(NULL != f);

  const f64 expected[] = {1.0, 2.0, 3.0, 4.0};
  for (uint64_t i = 0; i < 4; i++) {
    f64 re, im;
    const int n = fscanf(f, "%lf %lf", &re, &im);
    ASSERT(2 == n);
    ASSERT_CLOSE(re, expected[i], 1e-6);
    ASSERT_CLOSE(im, 0.0, 1e-6);
  }
  fclose(f);

  safe_free((void **)&bdg->modes_u);
  bdg_free(&bdg);
  unlink(fname_u);
}

TEST(write_mode_2d_real) {
  const char *fname_base = "/tmp/test_mode_2d";
  const char *fname_u = "/tmp/test_mode_2d_u_0.dat";

  const size_t N[] = {3, 2};
  const f64 L[] = {1.0, 1.0};
  bdg_t *bdg = bdg_alloc(2, N, L, 0);
  bdg_set_system(bdg);
  bdg_set_solver_params(bdg, 1, 2, 100, 1e-6);

  /* Manually inject fake mode: 3*2 = 6 values */
  bdg->modes_u = xcalloc(6, sizeof(f64));
  f64 *u = (f64 *)bdg->modes_u;
  for (uint64_t i = 0; i < 6; i++)
    u[i] = (f64)(i + 1);

  const int rc = bdg_write_mode_u(bdg, 0, fname_base);
  ASSERT(0 == rc);

  /* Read back as raw text, verify blank line between y-rows */
  FILE *f = fopen(fname_u, "r");
  ASSERT(NULL != f);

  char buf[1024];
  /* Line 1: y=0 row: "1 0  2 0  3 0" (tab-separated pairs) */
  ASSERT(NULL != fgets(buf, sizeof(buf), f));
  {
    f64 r0, i0, r1, i1, r2, i2;
    const int n = sscanf(buf, "%lf %lf %lf %lf %lf %lf", &r0, &i0, &r1, &i1, &r2, &i2);
    ASSERT(6 == n);
    ASSERT_CLOSE(r0, 1.0, 1e-6);
    ASSERT_CLOSE(r1, 2.0, 1e-6);
    ASSERT_CLOSE(r2, 3.0, 1e-6);
  }

  /* Blank line between y-rows */
  ASSERT(NULL != fgets(buf, sizeof(buf), f));
  ASSERT(0 == strcmp(buf, "\n"));

  /* Line 3: y=1 row: "4 0  5 0  6 0" */
  ASSERT(NULL != fgets(buf, sizeof(buf), f));
  {
    f64 r0, i0, r1, i1, r2, i2;
    const int n = sscanf(buf, "%lf %lf %lf %lf %lf %lf", &r0, &i0, &r1, &i1, &r2, &i2);
    ASSERT(6 == n);
    ASSERT_CLOSE(r0, 4.0, 1e-6);
    ASSERT_CLOSE(r1, 5.0, 1e-6);
    ASSERT_CLOSE(r2, 6.0, 1e-6);
  }

  fclose(f);

  safe_free((void **)&bdg->modes_u);
  bdg_free(&bdg);
  unlink(fname_u);
}

TEST(write_mode_out_of_range) {
  const size_t N[] = {4};
  const f64 L[] = {1.0};
  bdg_t *bdg = bdg_alloc(1, N, L, 0);
  bdg_set_system(bdg);
  bdg_set_solver_params(bdg, 2, 4, 100, 1e-6);

  /* Inject fake mode for nev=2 */
  bdg->modes_u = xcalloc(8, sizeof(f64));

  /* mode_idx=5 is out of range (nev=2) */
  const int rc = bdg_write_mode_u(bdg, 5, "/tmp/test_oor");
  ASSERT(-2 == rc);

  safe_free((void **)&bdg->modes_u);
  bdg_free(&bdg);
}

/* ================================================================ */
int main(void) {
  printf("Load wavefunction:\n");
  RUN(load_wf_real);
  RUN(load_wf_complex);
  RUN(load_wf_file_not_found);

  printf("\nWrite eigenvalues:\n");
  RUN(write_eigenvalues);

  printf("\nWrite modes:\n");
  RUN(write_mode_1d_real);
  RUN(write_mode_2d_real);
  RUN(write_mode_out_of_range);

  printf("\n========================================\n");
  printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
  printf("========================================\n");

  return tests_failed > 0 ? 1 : 0;
}
