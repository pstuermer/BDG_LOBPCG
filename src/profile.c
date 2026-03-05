/**
 * @file profile.c
 * @brief Global timing state and summary printer for BDG_PROFILE.
 */

#ifdef BDG_PROFILE

#include "profile.h"
#include <stdio.h>
#include <string.h>

bdg_profile_t g_bdg_profile;

void bdg_profile_reset(void) {
  memset(&g_bdg_profile, 0, sizeof(g_bdg_profile));
}

static void print_row(const char *name, double t, uint64_t n) {
  if (0 == n) return;
  fprintf(stderr, "  %-22s %10.3f ms  %8lu calls  %8.3f ms/call\n",
          name, t * 1e3, (unsigned long)n, (t / (double)n) * 1e3);
}

static void print_sub(const char *name, double t, uint64_t n) {
  if (0 == n) return;
  fprintf(stderr, "    %-20s %10.3f ms  %19s  %8.3f ms/call\n",
          name, t * 1e3, "", (t / (double)n) * 1e3);
}

void bdg_profile_print(void) {
  const bdg_profile_t *p = &g_bdg_profile;
  const double total = p->t_matmulK + p->t_matmulM + p->t_swap
                      + p->t_precondK + p->t_precondM;

  fprintf(stderr, "\n=== BdG Profile ===\n");
  fprintf(stderr, "  %-22s %10.3f ms\n", "Total measured", total * 1e3);
  fprintf(stderr, "  ─────────────────────────────────────────────────────────────────\n");

  /* matmulK */
  print_row("matmulK", p->t_matmulK, p->n_matmulK);
  print_sub("kinetic", p->t_matmulK_kin, p->n_matmulK);
  print_sub("elem", p->t_matmulK_elem, p->n_matmulK);

  /* matmulM */
  print_row("matmulM", p->t_matmulM, p->n_matmulM);
  print_sub("kinetic", p->t_matmulM_kin, p->n_matmulM);
  print_sub("elem", p->t_matmulM_elem, p->n_matmulM);
  print_sub("dipolar_conv", p->t_matmulM_dip, p->n_matmulM);

  /* swap */
  print_row("matmulSwap", p->t_swap, p->n_swap);

  /* precondK */
  print_row("precondK", p->t_precondK, p->n_precondK);
  print_sub("scale1", p->t_precondK_scale1, p->n_precondK);
  print_sub("fft", p->t_precondK_fft, p->n_precondK);
  print_sub("kfilt", p->t_precondK_kfilt, p->n_precondK);
  print_sub("ifft", p->t_precondK_ifft, p->n_precondK);
  print_sub("scale2", p->t_precondK_scale2, p->n_precondK);

  /* precondM */
  print_row("precondM", p->t_precondM, p->n_precondM);
  print_sub("scale1", p->t_precondM_scale1, p->n_precondM);
  print_sub("fft", p->t_precondM_fft, p->n_precondM);
  print_sub("kfilt", p->t_precondM_kfilt, p->n_precondM);
  print_sub("ifft", p->t_precondM_ifft, p->n_precondM);
  print_sub("scale2", p->t_precondM_scale2, p->n_precondM);

  fprintf(stderr, "  ─────────────────────────────────────────────────────────────────\n\n");
}

#endif /* BDG_PROFILE */
