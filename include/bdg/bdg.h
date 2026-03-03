#ifndef BDG_H
#define BDG_H

#include "lobpcg/types.h"
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

/**
 * @file bdg.h
 * @brief Public API for the BdG dipolar library.
 *
 * Solves the Bogoliubov-de Gennes eigenvalue problem for a dipolar
 * superfluid using the indefinite LOBPCG (iLOBPCG) eigensolver.
 *
 * K/M formalism: A = [K, 0; 0, M], B = [0, I; I, 0] (swap).
 * Real psi0 uses d_ilobpcg (f64), complex psi0 uses z_ilobpcg (c64).
 */

/* ================================================================
 * Opaque handle
 * ================================================================ */

typedef struct bdg_t bdg_t;

/* ================================================================
 * Init strategy for eigenvector seeding
 * ================================================================ */

typedef enum {
  BDG_INIT_DEFAULT,      /* Planewave-seeded, B-positive (current behavior) */
  BDG_INIT_WF_WEIGHTED,  /* |gauss| * |psi0| for both halves (B-positive) */
  BDG_INIT_REUSE,        /* Previous modes + noise; set by bdg_reuse_modes */
  BDG_INIT_CUSTOM        /* User callback; must ensure B-positivity */
} bdg_init_mode_t;

/**
 * Custom init callback type.
 * @param bdg      The BdG handle (read-only for grid info)
 * @param X        Eigenvector block to fill: ctype[n * sizeSub], column-major.
 *                 ctype is f64 if complex_psi0=0, c64 if complex_psi0=1.
 *                 n = 2*size. Each column j: X[0..size-1] = u-part, X[size..n-1] = v-part.
 *                 For B-positivity: set u-part = v-part per column.
 * @param n        Row dimension (2 * grid_size)
 * @param sizeSub  Number of columns (search space width)
 * @param param    User data from bdg_set_init_mode
 */
typedef void (*bdg_init_fn)(const bdg_t *bdg, void *X,
                            uint64_t n, uint64_t sizeSub, void *param);

/* ================================================================
 * Lifecycle
 * ================================================================ */

/**
 * Allocate a BdG problem.
 * @param dim        Spatial dimension (1, 2, or 3)
 * @param N          Grid points per dimension (array of length dim)
 * @param L          Box lengths per dimension (array of length dim)
 * @param complex_psi0  0 = real wavefunction (d path), 1 = complex (z path)
 */
bdg_t *bdg_alloc(size_t dim, const size_t *N, const f64 *L, int complex_psi0);

/**
 * Free all resources. Sets *bdg = NULL.
 */
void bdg_free(bdg_t **bdg);

/**
 * Reset physics state for parameter sweeps.
 *
 * Clears ALL physics state (trap, wavefunction, interactions, dipolar,
 * chemical potential, solver results) while preserving grid geometry,
 * k-space arrays, FFTW plans, scratch buffers, and solver parameters.
 *
 * After reset, state == BDG_HAS_SYSTEM — resume from bdg_set_trap().
 *
 * WARNING: Pointers from bdg_eigenvalues()/bdg_modes_u()/bdg_modes_v()
 * become invalid after reset.
 */
void bdg_reset(bdg_t *bdg);

/* ================================================================
 * Setup — call in order
 * ================================================================ */

/**
 * Build k-space arrays (k2, kx2) and create FFTW plans.
 * Must be called before any operator application.
 */
void bdg_set_system(bdg_t *bdg);

/**
 * Set the trapping potential. Adds V_trap(r) to localTermK and localTermM.
 * @param V_trap  Function V(dim, r[dim], param) returning potential at position r
 * @param param   User data passed to V_trap
 */
void bdg_set_trap(bdg_t *bdg,
                  f64 (*V_trap)(size_t dim, const f64 *r, void *param),
                  void *param);

/**
 * Set the condensate wavefunction (in-memory copy).
 * @param wf       Pointer to wavefunction data (f64* or c64* matching complex_psi0)
 * @param wf_size  Number of grid points (must equal product of N)
 */
void bdg_set_wavefunction(bdg_t *bdg, const void *wf, size_t wf_size);

/**
 * Set local interaction terms. Two function pointers for BdG K/M asymmetry.
 *
 * localTermK[i] += U_intK(param, |wf[i]|^2)
 * localTermM[i] += U_intK(param, |wf[i]|^2) + 2 * U_intM(param, |wf[i]|^2)
 *
 * For contact-only: U_intK = U_intM = g * density.
 * For LHY: U_intK and U_intM have different beyond-mean-field coefficients.
 */
void bdg_set_local_interactions(bdg_t *bdg,
                                f64 (*U_intK)(void *param, f64 density),
                                f64 (*U_intM)(void *param, f64 density),
                                void *param);

/**
 * Subtract chemical potential from both localTermK and localTermM.
 * Also stores mu for the preconditioner.
 */
void bdg_set_mu(bdg_t *bdg, f64 mu);

/**
 * Set dipolar interaction (3D only).
 * Computes the k-space kernel and adds the mean-field dipolar potential
 * to localTermK and localTermM.
 *
 * @param g_ddi          Dipolar coupling strength
 * @param dipole_dir     Unit vector for dipole orientation (length 3)
 * @param cutoff_radius  Spherical cutoff radius for the dipolar kernel
 */
void bdg_set_dipolar(bdg_t *bdg, f64 g_ddi, const f64 *dipole_dir, f64 cutoff_radius);

/* ================================================================
 * Solver
 * ================================================================ */

/**
 * Set solver parameters.
 * @param nev      Number of eigenvalues to compute
 * @param sizeSub  Subspace size (>= nev, determines search space width)
 * @param maxIter  Maximum iterations
 * @param tol      Convergence tolerance on residual norms
 */
void bdg_set_solver_params(bdg_t *bdg, size_t nev, size_t sizeSub,
                           size_t maxIter, f64 tol);

/* ================================================================
 * Init strategy
 * ================================================================ */

/**
 * Select eigenvector initialization strategy.
 * For BDG_INIT_CUSTOM, fn and param are required; ignored otherwise.
 * Survives bdg_reset — set once, reused across sweeps.
 */
void bdg_set_init_mode(bdg_t *bdg, bdg_init_mode_t mode,
                       bdg_init_fn fn, void *param);

/**
 * Copy current modes into internal buffer with Gaussian noise for reuse.
 * noise_frac: per-column noise stddev = noise_frac * ||column||.
 * Automatically sets init_mode to BDG_INIT_REUSE.
 * Must be called BEFORE bdg_reset (while results exist).
 * After next bdg_solve, init_mode reverts to BDG_INIT_WF_WEIGHTED.
 * @return 0 on success, -3 if no results exist.
 */
int bdg_reuse_modes(bdg_t *bdg, f64 noise_frac);

/**
 * Solve the BdG eigenvalue problem.
 * Dispatches to d_ilobpcg or z_ilobpcg based on complex_psi0.
 * @return 0 on success, nonzero on failure
 */
int bdg_solve(bdg_t *bdg);

/* ================================================================
 * Results — valid after bdg_solve returns 0
 * ================================================================ */

/** Number of converged eigenvalues. */
size_t bdg_converged(const bdg_t *bdg);

/** Array of nev eigenvalues (always real, sorted). */
const f64 *bdg_eigenvalues(const bdg_t *bdg);

/** Eigenmodes u-part: size*nev column-major. Cast to f64* or c64*. */
const void *bdg_modes_u(const bdg_t *bdg);

/** Eigenmodes v-part: size*nev column-major. Cast to f64* or c64*. */
const void *bdg_modes_v(const bdg_t *bdg);

/* ================================================================
 * File I/O
 * ================================================================ */

/**
 * Load wavefunction from text file. Format: one "real imag" pair per line.
 * Reads ctx->size pairs, calls bdg_set_wavefunction internally.
 * For real psi0, stores only the real part; for complex, stores both.
 * @return 0 success, -1 file error, -2 format/size mismatch.
 */
int bdg_load_wavefunction(bdg_t *bdg, const char *filename);

/**
 * Load wavefunction from a printf-formatted filename.
 * Example: bdg_load_wavefunction_fmt(bdg, "data/wf_%03d.dat", step);
 * @return same as bdg_load_wavefunction.
 */
int bdg_load_wavefunction_fmt(bdg_t *bdg, const char *fmt, ...);

/**
 * Append eigenvalues to text file. One row per call: nev tab-separated
 * values in %.8e format, terminated by newline. Creates file if absent.
 * @return 0 success, -1 file error, -3 no results.
 */
int bdg_write_eigenvalues(const bdg_t *bdg, const char *filename);

/**
 * Write u-part of mode mode_idx to basename_u_<idx>.dat.
 * Dimension-aware formatting: 1D one pair per line; 2D blank line
 * between y-rows; 3D blank line between z-slices.
 * Always writes "real imag" pairs (imag=0 for real psi0).
 * @return 0 success, -1 file error, -2 mode_idx out of range, -3 no results.
 */
int bdg_write_mode_u(const bdg_t *bdg, uint64_t mode_idx,
                     const char *basename);

/** Write v-part of mode mode_idx. Same format as bdg_write_mode_u. */
int bdg_write_mode_v(const bdg_t *bdg, uint64_t mode_idx,
                     const char *basename);

#endif /* BDG_H */
