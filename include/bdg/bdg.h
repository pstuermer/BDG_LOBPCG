#ifndef BDG_H
#define BDG_H

#include "lobpcg/types.h"
#include <stddef.h>

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
 * @param g_ddi       Dipolar coupling strength
 * @param dipole_dir  Unit vector for dipole orientation (length 3)
 */
void bdg_set_dipolar(bdg_t *bdg, f64 g_ddi, const f64 *dipole_dir);

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

#endif /* BDG_H */
