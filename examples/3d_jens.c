#include "bdg/bdg.h"
#include "bdg/constants.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static f64 U_intK(void *param, const f64 density) {
  const f64 *p = param;
  const f64 apsi = sqrt(density);
  return p[0] * density + p[1] * apsi * density;
}

static f64 U_intM(void *param, const f64 density) {
  const f64 *p = param;
  const f64 apsi = sqrt(density);
  return p[0] * density + 1.5 * p[1] * apsi * density;
}

static f64 V_trap(uint64_t dim, const f64 *r, void *param) {
  (void)dim; (void)param;
  const f64 omega_unit = 2.0 * M_PI * 164 * MASS_UNIT * 1.0e-12 / HBAR;
  const f64 omega_trap = 30.0 * omega_unit;
  const f64 lambday = 90.0 / 30.0;
  const f64 lambdaz = 110.0 / 30.0;
  return 0.5 * omega_trap * omega_trap * (r[0]*r[0]
    + lambday*lambday * r[1]*r[1]
    + lambdaz*lambdaz * r[2]*r[2]);
}

int main(void) {
  const uint64_t dim = 3;
  const uint64_t N[3] = {256, 128, 64};
  const f64 L[3] = {32.0, 32.0, 16.0};

  const f64 a_unit = BOHR_RADIUS * 1.0e6;
  const f64 epsdd = 130.8 / 97.5;
  const f64 add = 130.8 * a_unit;
  const f64 as = add / epsdd;

  f64 param[2];
  param[0] = 4.0 * M_PI * as;
  param[1] = (128.0 * sqrt(M_PI) / 3.0) * as * as * sqrt(as) * (1.0 + 1.5 * epsdd * epsdd);

  const f64 g_ddi = 4.0 * M_PI * add;
  const f64 dir_ddi[3] = {0.0, 1.0, 0.0};
  const f64 cutoff_R = 0.5 * L[0];

  bdg_t *bdg = bdg_alloc(dim, N, L, 0);
  bdg_set_system(bdg);
  bdg_set_trap(bdg, V_trap, NULL);
  bdg_load_wavefunction(bdg, "examples/3d_jens_wf.dat");
  bdg_set_local_interactions(bdg, U_intK, U_intM, param);
  bdg_set_dipolar(bdg, g_ddi, dir_ddi, cutoff_R);
  // needs to be set last
  bdg_set_mu(bdg, 11.976798753);

  const uint64_t nev = 8;
  const uint64_t sizeSub = 12;
  const uint64_t maxIter = 300;
  const f64 tol = 1.0e-4;
  bdg_set_solver_params(bdg, nev, sizeSub, maxIter, tol);
  bdg_set_init_mode(bdg, BDG_INIT_DEFAULT, NULL, NULL);

  const f64 start = omp_get_wtime();
  const int ret = bdg_solve(bdg);
  const f64 elapsed = omp_get_wtime() - start;

  printf("bdg_solve returned %d\n", ret);
  if (0 == ret) {
    const uint64_t nconv = bdg_converged(bdg);
    const f64 *evals = bdg_eigenvalues(bdg);
    printf("converged: %zu / %zu\n", nconv, nev);
    for (uint64_t j = 0; j < nev; j++)
      printf("  evals[%2lu] = %.10f\n", (unsigned long)j, evals[j]);
    printf("elapsed: %.2f s\n", elapsed);
  }

  bdg_free(&bdg);
  return ret;
}
