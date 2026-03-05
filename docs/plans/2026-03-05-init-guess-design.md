# Geometry-Aware Planewave Initial Guess — Design

## Goal

Replace the 1D-only planewave initial guess with multi-dimensional k-vector seeding that works for elongated traps, ring traps, and radially symmetric harmonic traps.

## Motivation

Current `BDG_INIT_DEFAULT` seeds planewaves along dimension 0 only. Works for elongated traps (29 iterations on 3d_jens) but poor for ring/torus geometries where modes are angular momentum modes needing (kx,ky) pairs.

## API

```c
typedef enum {
  BDG_INIT_PLANEWAVE,     // geometry-aware planewave seeding (new default, value 0)
  BDG_INIT_WF_WEIGHTED,   // random Gaussian × |ψ₀|
  BDG_INIT_REUSE,         // previous eigenvectors
  BDG_INIT_CUSTOM          // user callback
} bdg_init_mode_t;

typedef enum {
  BDG_GEOM_AUTO,          // lowest |k|² in all dimensions
  BDG_GEOM_ELONGATED,     // k-vectors along longest L dimension only
  BDG_GEOM_RING           // (kx,ky) pairs, kz=0
} bdg_geom_hint_t;
```

- `BDG_INIT_PLANEWAVE` replaces `BDG_INIT_DEFAULT` at value 0 — backward compatible (xcalloc zeros).
- Geometry hint passed via `param` pointer: `bdg_set_init_mode(bdg, BDG_INIT_PLANEWAVE, (void *)(intptr_t)BDG_GEOM_RING, NULL)`.
- `param == NULL` defaults to `BDG_GEOM_AUTO`.

## K-Vector Selection per Geometry Hint

**BDG_GEOM_AUTO:** Enumerate k-vectors (i,j,k) across all dimensions, compute |k|² = kx2[0][i] + kx2[1][j] + kx2[2][k], sort ascending, take lowest ceil(sizeSub/2) unique vectors.

**BDG_GEOM_ELONGATED:** Only vary k-index along dimension with largest L[d]. Other dimensions fixed at k=0. Reproduces current DEFAULT behavior but auto-selects the long axis.

**BDG_GEOM_RING:** Enumerate (kx, ky) pairs sorted by kx2[0][i] + kx2[1][j], with kz=0. Represents angular momentum modes via Cartesian planewave decomposition.

## Seeding

- Column 2j: cos(k·r) × |ψ₀(r)|
- Column 2j+1: sin(k·r) × |ψ₀(r)| (skip for k=0)
- u-part = v-part (B-positive)
- Small random perturbation (1e-4) to break degeneracies

## K-Vector Enumeration Detail

- Enumerate up to min(N[d], 2*sizeSub) indices per dimension
- Use ctx->kx2[d][i] directly for |k|² (already stores squared wavenumber per FFTW index)
- Allocate small temp array of (|k|², indices) tuples, qsort, take lowest, free
- Position: r[d] = (i_d / N[d]) × L[d], with i_d = (flat_index / stride_d) % N[d]
- Dot product: k·r = Σ_d (2π × freq[d] / L[d]) × r[d]
- 1D/2D handled naturally (missing dimensions have k=0)

## Bogoliubov u/v Ratio

Deferred. Currently u=v. Can add once µ is computed internally.

## Testing

- Unit test: verify k-vectors sorted by |k|² for AUTO
- Regression: 3d_jens with ELONGATED hint should give ≤29 iterations
- Functional: run with each hint, verify convergence
