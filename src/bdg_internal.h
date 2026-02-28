#ifndef BDG_INTERNAL_H
#define BDG_INTERNAL_H

#include "bdg/bdg.h"
#include "lobpcg/types.h"
#include "bdg/memory.h"
#include "lobpcg/linop.h"
#include "bdg_log.h"
#include <fftw3.h>
#include <math.h>
#include <stdint.h>

/**
 * @file bdg_internal.h
 * @brief Private definitions for the BdG library.
 *
 * Full struct definitions and internal function prototypes.
 * NOT installed — only included by src/ .c files.
 */

/* ================================================================
 * Setup state tracking — bitmask flags
 * ================================================================ */

enum {
    BDG_HAS_SYSTEM       = 1u << 0,
    BDG_HAS_TRAP         = 1u << 1,
    BDG_HAS_WF           = 1u << 2,
    BDG_HAS_INTERACTIONS = 1u << 3,
    BDG_HAS_DIPOLAR      = 1u << 4,
    BDG_HAS_MU           = 1u << 5
};

/** Exit if prerequisite flag is NOT set. */
#define BDG_REQUIRE(bdg, flag, caller) do { \
    if (0 == ((bdg)->state & (flag))) \
        BDG_ERROR("%s: prerequisite %s not satisfied", caller, #flag); \
} while (0)

/** Exit if flag IS already set (prevents double-call). */
#define BDG_FORBID(bdg, flag, caller) do { \
    if (0 != ((bdg)->state & (flag))) \
        BDG_ERROR("%s: %s already set (double-call forbidden)", caller, #flag); \
} while (0)

/** Clamp near-zero values to threshold (for preconditioner sqrt). */
static inline f64 safe_val(f64 x) {
    const f64 thresh = 1e-8;
    return (fabs(x) < thresh) ? thresh : x;
}

/* ================================================================
 * matmul_ctx_t — holds all operator data
 * ================================================================ */

typedef struct {
    /* Grid geometry */
    size_t dim;         /* 1, 2, or 3 */
    size_t *N;          /* Grid points per dimension [dim] */
    f64    *L;          /* Box lengths per dimension [dim] */
    size_t  size;       /* Product of N (total real-space grid points) */
    size_t  k_size;     /* k-space size: reduced for r2c, == size for c2c */

    /* k-space arrays (persistent, read every matvec) */
    f64   *k2;          /* |k|^2, length k_size */
    f64  **kx2;         /* Per-dimension k_d^2 arrays, kx2[d][N[d]] */

    /* Physics (persistent, set during setup) */
    f64   *localTermK;  /* V_trap + U_intK - mu, length size */
    f64   *localTermM;  /* V_trap + U_intK + 2*U_intM + g_ddi*Phi_dd - mu, length size */
    f64   *longRngInt;  /* Dipolar kernel in k-space, length k_size (or NULL) */
    f64    g_ddi;       /* Dipolar coupling strength */
    f64    mu;          /* Chemical potential (stored for preconditioner) */
    unsigned int dipolar; /* 1 if dipolar interaction is active */

    /* Wavefunction — void* because f64 (real) or c64 (complex) */
    void  *wf;
    size_t wf_size;     /* == size */

    /* FFTW plans (no mode_t wrapper) */
    fftw_plan fwd_plan; /* r2c (real path) or c2c forward (complex path) */
    fftw_plan bwd_plan; /* c2r (real path) or c2c backward (complex path) */

    /* Scratch buffers (mutated every matvec — NOT thread-safe)
     * Shared between A, B, T operators (LOBPCG calls them sequentially).
     * f_wrk:  c64[k_size] — FFT workspace
     * c_wrk1: sized for max(size, k_size) * sizeof(c64)
     * c_wrk2: sized for max(size, k_size) * sizeof(c64)
     */
    c64  *f_wrk;        /* c64[k_size] — FFT workspace */
    void *c_wrk1;
    void *c_wrk2;

    /* Preconditioner auxiliary arrays (persistent, set after set_mu) */
    f64 *precond_sqrtK; /* 1/sqrt(localTermK + mu), length size (or NULL) */
    f64 *precond_sqrtM; /* 1/sqrt(localTermM + mu), length size (or NULL) */

    /* Flag: is wavefunction complex? (mirrors bdg_t.complex_psi0) */
    int complex_psi0;
} matmul_ctx_t;

/* ================================================================
 * bdg_t — full definition (opaque in public header)
 * ================================================================ */

struct bdg_t {
    matmul_ctx_t *ctx;
    uint32_t state;     /* Bitmask of BDG_HAS_* flags (zeroed by xcalloc) */

    /* Solver parameters */
    size_t nev;
    size_t sizeSub;
    size_t maxIter;
    f64    tol;
    int    complex_psi0; /* 0 = real (d_ilobpcg), 1 = complex (z_ilobpcg) */

    /* Results (populated by bdg_solve, owned by bdg_t) */
    f64   *eigvals;     /* nev eigenvalues (always real) */
    void  *modes_u;     /* f64* or c64*, size*nev, column-major */
    void  *modes_v;     /* f64* or c64*, size*nev, column-major */
    size_t converged;
};

/* ================================================================
 * matmul_ctx allocation
 * ================================================================ */

matmul_ctx_t *matmul_ctx_alloc(size_t dim, const size_t *N, const f64 *L);
void          matmul_ctx_free(matmul_ctx_t **ctx);
void          matmul_ctx_set_system(matmul_ctx_t *ctx, int complex_psi0);

/* ================================================================
 * Kinetic operator — d (r2c/c2r) and z (c2c)
 * ================================================================ */

void kinetic_d(matmul_ctx_t *ctx, const f64 *x, f64 *y);
void kinetic_z(matmul_ctx_t *ctx, const c64 *x, c64 *y);

/* ================================================================
 * K operator: kinetic + localTermK
 * ================================================================ */

void matmulK_d(matmul_ctx_t *ctx, const f64 *x, f64 *y);
void matmulK_z(matmul_ctx_t *ctx, const c64 *x, c64 *y);

/* ================================================================
 * M operator: kinetic + localTermM + 2*dipolar_conv (optional)
 * ================================================================ */

void matmulM_d(matmul_ctx_t *ctx, const f64 *x, f64 *y);
void matmulM_z(matmul_ctx_t *ctx, const c64 *x, c64 *y);

/* ================================================================
 * Lrep operator: block [K; M] on stacked [u; v]
 * ================================================================ */

void matmulLrep_d(matmul_ctx_t *ctx, const f64 *x, f64 *y);
void matmulLrep_z(matmul_ctx_t *ctx, const c64 *x, c64 *y);

/* ================================================================
 * Swap operator: y = [x_lower; x_upper]
 * ================================================================ */

void matmulSwap_d(matmul_ctx_t *ctx, const f64 *x, f64 *y);
void matmulSwap_z(matmul_ctx_t *ctx, const c64 *x, c64 *y);

/* ================================================================
 * Product preconditioner: T = [T_K; T_M]
 * ================================================================ */

void precondK_d(matmul_ctx_t *ctx, const f64 *x, f64 *y);
void precondK_z(matmul_ctx_t *ctx, const c64 *x, c64 *y);
void precondM_d(matmul_ctx_t *ctx, const f64 *x, f64 *y);
void precondM_z(matmul_ctx_t *ctx, const c64 *x, c64 *y);
void precondLrep_d(matmul_ctx_t *ctx, const f64 *x, f64 *y);
void precondLrep_z(matmul_ctx_t *ctx, const c64 *x, c64 *y);

/* ================================================================
 * Dipolar (3D only)
 * ================================================================ */

/** Build k-space kernel longRngInt (3D only). */
void dipolar_set_kernel(matmul_ctx_t *ctx, f64 g_ddi, const f64 *dipole_dir, f64 cutoff_radius);

/** Add mean-field dipolar potential to localTermK and localTermM. */
void dipolar_add_meanfield(matmul_ctx_t *ctx);

/** Perturbation-dependent dipolar convolution: conj(wf)*v → FFT → *kernel → IFFT → g_ddi*wf*result */
void dipolar_conv_d(matmul_ctx_t *ctx, const f64 *v, f64 *out);
void dipolar_conv_z(matmul_ctx_t *ctx, const c64 *v, c64 *out);

/* ================================================================
 * Solver internals
 * ================================================================ */

int bdg_solve_d(bdg_t *bdg);
int bdg_solve_z(bdg_t *bdg);

#endif /* BDG_INTERNAL_H */
