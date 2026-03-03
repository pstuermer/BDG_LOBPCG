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

TEST(load_wf_fmt) {
  /* Create temp file with known name */
  const char *dir = "/tmp";
  const int idx = 7;
  char expected_name[256];
  snprintf(expected_name, sizeof(expected_name), "%s/test_wf_%03d.dat", dir, idx);

  FILE *f = fopen(expected_name, "w");
  fprintf(f, "1.5 0.0\n2.5 0.0\n");
  fclose(f);

  const size_t N[] = {2};
  const f64 L[] = {1.0};
  bdg_t *bdg = bdg_alloc(1, N, L, 0);
  bdg_set_system(bdg);

  int rc = bdg_load_wavefunction_fmt(bdg, "%s/test_wf_%03d.dat", dir, idx);
  ASSERT(0 == rc);

  const f64 *wf = (const f64 *)bdg->ctx->wf;
  ASSERT(NULL != wf);
  ASSERT_NEAR(wf[0], 1.5, 1e-12);
  ASSERT_NEAR(wf[1], 2.5, 1e-12);

  bdg_free(&bdg);
  unlink(expected_name);
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

/* ================================================================
 * Task 7: Init strategy unit tests
 * ================================================================ */

TEST(set_init_mode) {
  const size_t N[] = {4};
  const f64 L[] = {1.0};
  bdg_t *bdg = bdg_alloc(1, N, L, 0);

  bdg_set_init_mode(bdg, BDG_INIT_WF_WEIGHTED, NULL, NULL);
  ASSERT(BDG_INIT_WF_WEIGHTED == bdg->init_mode);

  bdg_set_init_mode(bdg, BDG_INIT_DEFAULT, NULL, NULL);
  ASSERT(BDG_INIT_DEFAULT == bdg->init_mode);

  bdg_free(&bdg);
}

TEST(reuse_modes_no_results) {
  const size_t N[] = {4};
  const f64 L[] = {1.0};
  bdg_t *bdg = bdg_alloc(1, N, L, 0);

  const int rc = bdg_reuse_modes(bdg, 0.01);
  ASSERT(-3 == rc);

  bdg_free(&bdg);
}

TEST(reuse_modes_packs_buffer) {
  const size_t N[] = {4};
  const f64 L[] = {1.0};
  bdg_t *bdg = bdg_alloc(1, N, L, 0);
  bdg_set_system(bdg);
  bdg_set_solver_params(bdg, 2, 4, 100, 1e-6);

  const uint64_t size = 4;
  bdg->modes_u = xcalloc(size * 2, sizeof(f64));
  bdg->modes_v = xcalloc(size * 2, sizeof(f64));
  bdg->converged = 2;
  f64 *u = (f64 *)bdg->modes_u;
  f64 *v = (f64 *)bdg->modes_v;
  for (uint64_t i = 0; i < size * 2; i++) {
    u[i] = 1.0;
    v[i] = 1.0;
  }

  f64 wf[] = {1.0, 1.0, 1.0, 1.0};
  bdg_set_wavefunction(bdg, wf, size);

  const int rc = bdg_reuse_modes(bdg, 0.01);
  ASSERT(0 == rc);
  ASSERT(BDG_INIT_REUSE == bdg->init_mode);
  ASSERT(NULL != bdg->reuse_buf);
  ASSERT(8 == bdg->reuse_n);
  ASSERT(4 == bdg->reuse_cols);

  const f64 *buf = (const f64 *)bdg->reuse_buf;
  f64 sum = 0.0;
  for (uint64_t i = 0; i < 8 * 4; i++) sum += fabs(buf[i]);
  ASSERT(sum > 0.0);

  safe_free((void **)&bdg->modes_u);
  safe_free((void **)&bdg->modes_v);
  bdg_free(&bdg);
}

TEST(init_survives_reset) {
  const size_t N[] = {4};
  const f64 L[] = {1.0};
  bdg_t *bdg = bdg_alloc(1, N, L, 0);
  bdg_set_system(bdg);

  bdg_set_init_mode(bdg, BDG_INIT_WF_WEIGHTED, NULL, NULL);
  bdg_reset(bdg);
  ASSERT(BDG_INIT_WF_WEIGHTED == bdg->init_mode);

  bdg_free(&bdg);
}

/* ================================================================
 * Task 8: Sweep integration test
 * ================================================================ */

static f64 sweep_U_K(void *p, f64 n) { return (*(f64 *)p) * n; }
static f64 sweep_U_M(void *p, f64 n) { return (*(f64 *)p) * n; }

TEST(sweep_reuse_1d) {
  const size_t N[] = {32};
  const f64 L[] = {10.0};
  const uint64_t size = 32;
  const f64 g = 1.0;
  const f64 mu1 = g;

  bdg_t *bdg = bdg_alloc(1, N, L, 0);
  bdg_set_system(bdg);
  bdg_set_solver_params(bdg, 2, 5, 200, 1e-5);

  f64 wf[32];
  for (uint64_t i = 0; i < size; i++) wf[i] = 1.0;

  /* First solve */
  bdg_set_wavefunction(bdg, wf, size);
  f64 gval = g;
  bdg_set_local_interactions(bdg, sweep_U_K, sweep_U_M, &gval);
  bdg_set_mu(bdg, mu1);
  int rc = bdg_solve(bdg);
  ASSERT(0 == rc);

  /* Write first eigenvalues */
  const char *eigfile = "/tmp/test_sweep_eigvals.dat";
  unlink(eigfile);
  rc = bdg_write_eigenvalues(bdg, eigfile);
  ASSERT(0 == rc);

  /* Reuse modes for next solve */
  rc = bdg_reuse_modes(bdg, 0.01);
  ASSERT(0 == rc);

  /* Reset and re-setup */
  bdg_reset(bdg);
  bdg_set_wavefunction(bdg, wf, size);
  bdg_set_local_interactions(bdg, sweep_U_K, sweep_U_M, &gval);
  bdg_set_mu(bdg, mu1 * 1.01);
  rc = bdg_solve(bdg);
  ASSERT(0 == rc);

  /* Write second eigenvalues */
  rc = bdg_write_eigenvalues(bdg, eigfile);
  ASSERT(0 == rc);

  /* Verify eigenvalue file has two lines */
  FILE *f = fopen(eigfile, "r");
  ASSERT(NULL != f);
  int line_count = 0;
  char line[1024];
  while (NULL != fgets(line, sizeof(line), f)) line_count++;
  ASSERT(2 == line_count);
  fclose(f);

  bdg_free(&bdg);
  unlink(eigfile);
}

/* ================================================================ */
int main(void) {
  printf("Load wavefunction:\n");
  RUN(load_wf_real);
  RUN(load_wf_complex);
  RUN(load_wf_file_not_found);
  RUN(load_wf_fmt);

  printf("\nWrite eigenvalues:\n");
  RUN(write_eigenvalues);

  printf("\nWrite modes:\n");
  RUN(write_mode_1d_real);
  RUN(write_mode_2d_real);
  RUN(write_mode_out_of_range);

  printf("\nInit strategy:\n");
  RUN(set_init_mode);
  RUN(reuse_modes_no_results);
  RUN(reuse_modes_packs_buffer);
  RUN(init_survives_reset);

  printf("\nSweep integration:\n");
  RUN(sweep_reuse_1d);

  printf("\n========================================\n");
  printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
  printf("========================================\n");

  return tests_failed > 0 ? 1 : 0;
}
