#include "bdg_internal.h"
#include <stdio.h>
#include <stdarg.h>
#include <complex.h>
#include <stdint.h>
#include <string.h>

/**
 * @file io.c
 * @brief File I/O for the BdG library.
 */

int bdg_load_wavefunction(bdg_t *bdg, const char *filename) {
  BDG_REQUIRE(bdg, BDG_HAS_SYSTEM, "bdg_load_wavefunction");
  const uint64_t size = bdg->ctx->size;

  FILE *file = fopen(filename, "r");
  if (NULL == file) return -1;

  if (0 == bdg->complex_psi0) {
    f64 *wf = xcalloc(size, sizeof(f64));
    for (uint64_t i = 0; i < size; i++) {
      f64 re, im;
      if (2 != fscanf(file, "%lf %lf", &re, &im)) {
        safe_free((void **)&wf);
        fclose(file);
        return -2;
      }
      wf[i] = re;
    }
    fclose(file);
    bdg_set_wavefunction(bdg, wf, size);
    safe_free((void **)&wf);
  } else {
    c64 *wf = xcalloc(size, sizeof(c64));
    for (uint64_t i = 0; i < size; i++) {
      f64 re, im;
      if (2 != fscanf(file, "%lf %lf", &re, &im)) {
        safe_free((void **)&wf);
        fclose(file);
        return -2;
      }
      wf[i] = re + im * I;
    }
    fclose(file);
    bdg_set_wavefunction(bdg, wf, size);
    safe_free((void **)&wf);
  }
  return 0;
}

int bdg_write_eigenvalues(const bdg_t *bdg, const char *filename) {
  if (NULL == bdg->eigvals) return -3;

  FILE *file = fopen(filename, "a");
  if (NULL == file) return -1;

  const uint64_t nev = bdg->nev;
  for (uint64_t i = 0; i < nev; i++) {
    fprintf(file, "%.8e", bdg->eigvals[i]);
    if (i + 1 < nev) fprintf(file, "\t");
  }
  fprintf(file, "\n");
  fclose(file);
  return 0;
}

static int write_mode_impl(const bdg_t *bdg, uint64_t mode_idx,
                           const void *modes, const char *basename,
                           const char *part_name) {
  if (NULL == modes) return -3;
  if (mode_idx >= bdg->nev) return -2;

  char fname[512];
  snprintf(fname, sizeof(fname), "%s_%s_%lu.dat", basename, part_name,
           (unsigned long)mode_idx);

  FILE *file = fopen(fname, "w");
  if (NULL == file) return -1;

  const uint64_t size = bdg->ctx->size;
  const uint64_t dim = bdg->ctx->dim;
  const uint64_t *N = bdg->ctx->N;

  if (0 == bdg->complex_psi0) {
    const f64 *mode = (const f64 *)modes + mode_idx * size;
    if (1 == dim) {
      for (uint64_t i = 0; i < N[0]; i++)
        fprintf(file, "%.8e\t%.8e\n", mode[i], 0.0);
    } else if (2 == dim) {
      for (uint64_t j = 0; j < N[1]; j++) {
        for (uint64_t i = 0; i < N[0]; i++) {
          const uint64_t idx = j * N[0] + i;
          fprintf(file, "%.8e\t%.8e", mode[idx], 0.0);
          if (i + 1 < N[0]) fprintf(file, "\t");
        }
        fprintf(file, "\n");
        if (j + 1 < N[1]) fprintf(file, "\n");
      }
    } else {
      for (uint64_t k = 0; k < N[2]; k++) {
        for (uint64_t j = 0; j < N[1]; j++) {
          for (uint64_t i = 0; i < N[0]; i++) {
            const uint64_t idx = k * N[1] * N[0] + j * N[0] + i;
            fprintf(file, "%.8e\t%.8e", mode[idx], 0.0);
            if (i + 1 < N[0]) fprintf(file, "\t");
          }
          fprintf(file, "\n");
        }
        if (k + 1 < N[2]) fprintf(file, "\n");
      }
    }
  } else {
    const c64 *mode = (const c64 *)modes + mode_idx * size;
    if (1 == dim) {
      for (uint64_t i = 0; i < N[0]; i++)
        fprintf(file, "%.8e\t%.8e\n", creal(mode[i]), cimag(mode[i]));
    } else if (2 == dim) {
      for (uint64_t j = 0; j < N[1]; j++) {
        for (uint64_t i = 0; i < N[0]; i++) {
          const uint64_t idx = j * N[0] + i;
          fprintf(file, "%.8e\t%.8e", creal(mode[idx]), cimag(mode[idx]));
          if (i + 1 < N[0]) fprintf(file, "\t");
        }
        fprintf(file, "\n");
        if (j + 1 < N[1]) fprintf(file, "\n");
      }
    } else {
      for (uint64_t k = 0; k < N[2]; k++) {
        for (uint64_t j = 0; j < N[1]; j++) {
          for (uint64_t i = 0; i < N[0]; i++) {
            const uint64_t idx = k * N[1] * N[0] + j * N[0] + i;
            fprintf(file, "%.8e\t%.8e", creal(mode[idx]), cimag(mode[idx]));
            if (i + 1 < N[0]) fprintf(file, "\t");
          }
          fprintf(file, "\n");
        }
        if (k + 1 < N[2]) fprintf(file, "\n");
      }
    }
  }

  fclose(file);
  return 0;
}

int bdg_write_mode_u(const bdg_t *bdg, uint64_t mode_idx,
                     const char *basename) {
  return write_mode_impl(bdg, mode_idx, bdg->modes_u, basename, "u");
}

int bdg_write_mode_v(const bdg_t *bdg, uint64_t mode_idx,
                     const char *basename) {
  return write_mode_impl(bdg, mode_idx, bdg->modes_v, basename, "v");
}

int bdg_load_wavefunction_fmt(bdg_t *bdg, const char *fmt, ...) {
  char fname[512];
  va_list args;
  va_start(args, fmt);
  const int n = vsnprintf(fname, sizeof(fname), fmt, args);
  va_end(args);
  if (n < 0 || (size_t)n >= sizeof(fname)) return -1;
  return bdg_load_wavefunction(bdg, fname);
}
