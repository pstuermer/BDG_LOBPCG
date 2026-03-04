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
  const f64 omega_unit = 2.0 * M_PI * 162 * MASS_UNIT * 1.0e-12 / HBAR;
  const f64 omega_trap = 155.0 * omega_unit;
  const f64 lambda_trap = 250.0 / 155.0;
  const f64 rho0 = 5.0;
  const f64 rho = sqrt(r[0]*r[0] + r[1]*r[1]);
  return 0.5 * omega_trap * omega_trap * ((rho - rho0) * (rho - rho0)
    + lambda_trap * lambda_trap * r[2] * r[2]);
}

int main(void) {
  const uint64_t N[3] = {256, 256, 128};
  const f64 L[3] = {45.0, 45.0, 45.0};

  const f64 a_unit = BOHR_RADIUS * 1.0e6;
  const f64 epsdd = 130.0 / 87.7;
  const f64 add = 130.0 * a_unit;
  const f64 as = add / epsdd;

  f64 param[2];
  param[0] = 4.0 * M_PI * as;
  param[1] = (128.0 * sqrt(M_PI) / 3.0) * as * as * sqrt(as) * (1.0 + 1.5 * epsdd * epsdd);

  const f64 g_ddi = 4.0 * M_PI * add;
  const f64 dir_ddi[3] = {0.0, 0.0, 1.0};
  const f64 cutoff_R = 0.5 * L[0];

  bdg_t *bdg = bdg_alloc(3, N, L, 0);
  bdg_set_system(bdg);
  bdg_set_trap(bdg, V_trap, NULL);
  bdg_load_wavefunction(bdg, "examples/3d_dipolar_wf.dat");
  bdg_set_local_interactions(bdg, U_intK, U_intM, param);
  bdg_set_dipolar(bdg, g_ddi, dir_ddi, cutoff_R);
  bdg_set_mu(bdg, 110.14092367);
  bdg_set_solver_params(bdg, 20, 30, 1000, 1.0e-5);
  bdg_set_init_mode(bdg, BDG_INIT_DEFAULT, NULL, NULL);

  const f64 start = omp_get_wtime();
  const int ret = bdg_solve(bdg);
  const f64 elapsed = omp_get_wtime() - start;

  printf("bdg_solve returned %d\n", ret);
  if (0 == ret) {
    const uint64_t nconv = bdg_converged(bdg);
    const f64 *evals = bdg_eigenvalues(bdg);
    printf("converged: %zu / 20\n", nconv);
    for (uint64_t j = 0; j < 20; j++)
      printf("  evals[%2lu] = %.10f\n", (unsigned long)j, evals[j]);
    printf("elapsed: %.2f s\n", elapsed);
  }

  bdg_free(&bdg);
  return ret;
}
