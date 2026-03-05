/**
 * @file profile.h
 * @brief Operator-level timing instrumentation for BdG, activated by -DBDG_PROFILE.
 */

#ifndef BDG_PROFILE_H
#define BDG_PROFILE_H

#include <stdint.h>

#ifdef BDG_PROFILE

#include <omp.h>

typedef struct {
  /* matmulK */
  double t_matmulK;
  double t_matmulK_kin;
  double t_matmulK_elem;
  uint64_t n_matmulK;

  /* matmulM */
  double t_matmulM;
  double t_matmulM_kin;
  double t_matmulM_elem;
  double t_matmulM_dip;
  uint64_t n_matmulM;

  /* matmulSwap */
  double t_swap;
  uint64_t n_swap;

  /* precondK */
  double t_precondK;
  double t_precondK_scale1;
  double t_precondK_fft;
  double t_precondK_kfilt;
  double t_precondK_ifft;
  double t_precondK_scale2;
  uint64_t n_precondK;

  /* precondM */
  double t_precondM;
  double t_precondM_scale1;
  double t_precondM_fft;
  double t_precondM_kfilt;
  double t_precondM_ifft;
  double t_precondM_scale2;
  uint64_t n_precondM;
} bdg_profile_t;

extern bdg_profile_t g_bdg_profile;

#define BDG_TIC(var) double var = omp_get_wtime()
#define BDG_TOC(var, field) g_bdg_profile.field += omp_get_wtime() - (var)
#define BDG_INC(field) g_bdg_profile.field++

void bdg_profile_print(void);
void bdg_profile_reset(void);

#else /* BDG_PROFILE not defined */

#define BDG_TIC(var)        (void)0
#define BDG_TOC(var, field) (void)0
#define BDG_INC(field)      (void)0

static inline void bdg_profile_print(void) {}
static inline void bdg_profile_reset(void) {}

#endif /* BDG_PROFILE */
#endif /* BDG_PROFILE_H */
