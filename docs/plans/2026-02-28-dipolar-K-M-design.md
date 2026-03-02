# Dipolar, K, and M Operators — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement dipolar convolution (kernel, meanfield, perturbation conv), matmulK, and matmulM operators.

**Architecture:** Bottom-up: implement matmulK first (trivial, depends only on kinetic), then dipolar (kernel, meanfield template, conv template), then matmulM (depends on both kinetic and dipolar). Each operator follows the established .inc template pattern with CABS/CONJ/CREAL macros for type-generic dispatch. Tests use plane-wave eigenvalue verification.

**Tech Stack:** C11, FFTW3 (via MKL), OpenMP, X-macro type-generic pattern

---

### Task 1: Implement matmulK

**Files:**
- Modify: `src/matmulK_impl.inc:10-17`

**Step 1: Write failing test**

Create `tests/test_matmulK.c`:

```c
#include "bdg_internal.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define TOL 1e-10

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

#define ASSERT_CLOSE(a, b, tol) do { \
    const double _a = (a), _b = (b); \
    if (fabs(_a - _b) >= (tol)) { \
        printf("[FAIL] line %d: %.15e vs %.15e\n", __LINE__, _a, _b); \
        tests_failed++; \
        return; \
    } \
} while(0)

/* ================================================================
 * Helper: set up a 1D ctx with constant potential V0 and mu
 * localTermK = V0 - mu at every point
 * ================================================================ */
static matmul_ctx_t *make_1d_ctx(const size_t N, const f64 L,
                                  int complex_psi0, f64 V0, f64 mu) {
    const size_t Narr[] = {N};
    const f64    Larr[] = {L};
    matmul_ctx_t *ctx = matmul_ctx_alloc(1, Narr, Larr);
    matmul_ctx_set_system(ctx, complex_psi0);

    ctx->localTermK = xcalloc(ctx->size, sizeof(f64));
    ctx->localTermM = xcalloc(ctx->size, sizeof(f64));
    for (size_t i = 0; i < ctx->size; i++) {
        ctx->localTermK[i] = V0 - mu;
        ctx->localTermM[i] = V0 - mu;
    }
    return ctx;
}

/* ================================================================
 * Test 1: K on cos(x) with V0=2, mu=0
 * K(cos(x)) = (0.5*1 + 2)*cos(x) = 2.5*cos(x)
 * ================================================================ */
TEST(matmulK_d_planewave_1d) {
    matmul_ctx_t *ctx = make_1d_ctx(64, 2.0 * M_PI, 0, 2.0, 0.0);
    const size_t size = ctx->size;
    f64 *x = xcalloc(size, sizeof(f64));
    f64 *y = xcalloc(size, sizeof(f64));

    for (size_t i = 0; i < size; i++)
        x[i] = cos((f64)i * 2.0 * M_PI / (f64)size);

    matmulK_d(ctx, x, y);

    for (size_t i = 0; i < size; i++) {
        const f64 expected = 2.5 * cos((f64)i * 2.0 * M_PI / (f64)size);
        ASSERT_CLOSE(y[i], expected, TOL);
    }

    safe_free((void **)&x);
    safe_free((void **)&y);
    matmul_ctx_free(&ctx);
}

/* ================================================================
 * Test 2: K on constant (k=0 mode) with V0=3, mu=1
 * K(1) = (0 + 3 - 1)*1 = 2
 * ================================================================ */
TEST(matmulK_d_constant) {
    matmul_ctx_t *ctx = make_1d_ctx(32, 2.0 * M_PI, 0, 3.0, 1.0);
    const size_t size = ctx->size;
    f64 *x = xcalloc(size, sizeof(f64));
    f64 *y = xcalloc(size, sizeof(f64));

    for (size_t i = 0; i < size; i++)
        x[i] = 1.0;

    matmulK_d(ctx, x, y);

    for (size_t i = 0; i < size; i++)
        ASSERT_CLOSE(y[i], 2.0, TOL);

    safe_free((void **)&x);
    safe_free((void **)&y);
    matmul_ctx_free(&ctx);
}

/* ================================================================
 * Test 3: d vs z consistency
 * ================================================================ */
TEST(matmulK_dz_consistency) {
    const size_t N = 64;
    const f64 L = 2.0 * M_PI;

    matmul_ctx_t *ctx_d = make_1d_ctx(N, L, 0, 2.0, 0.5);
    matmul_ctx_t *ctx_z = make_1d_ctx(N, L, 1, 2.0, 0.5);

    f64 *xd = xcalloc(N, sizeof(f64));
    f64 *yd = xcalloc(N, sizeof(f64));
    c64 *xz = xcalloc(N, sizeof(c64));
    c64 *yz = xcalloc(N, sizeof(c64));

    for (size_t i = 0; i < N; i++) {
        xd[i] = cos((f64)i * 2.0 * M_PI / (f64)N);
        xz[i] = xd[i] + 0.0 * I;
    }

    matmulK_d(ctx_d, xd, yd);
    matmulK_z(ctx_z, xz, yz);

    for (size_t i = 0; i < N; i++) {
        ASSERT_CLOSE(creal(yz[i]), yd[i], TOL);
        ASSERT_CLOSE(cimag(yz[i]), 0.0, TOL);
    }

    safe_free((void **)&xd); safe_free((void **)&yd);
    safe_free((void **)&xz); safe_free((void **)&yz);
    matmul_ctx_free(&ctx_d); matmul_ctx_free(&ctx_z);
}

/* ================================================================
 * Test 4: 3D planewave cos(x+y+z), V0=1, mu=0
 * k=(1,1,1), k²=3, K = 0.5*3 + 1 = 2.5
 * ================================================================ */
TEST(matmulK_d_3d) {
    const size_t N[] = {16, 16, 16};
    const f64    L[] = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};
    matmul_ctx_t *ctx = matmul_ctx_alloc(3, N, L);
    matmul_ctx_set_system(ctx, 0);
    const size_t size = ctx->size;

    ctx->localTermK = xcalloc(size, sizeof(f64));
    ctx->localTermM = xcalloc(size, sizeof(f64));
    for (size_t i = 0; i < size; i++)
        ctx->localTermK[i] = 1.0;

    f64 *x = xcalloc(size, sizeof(f64));
    f64 *y = xcalloc(size, sizeof(f64));

    for (size_t iz = 0; iz < N[2]; iz++)
        for (size_t iy = 0; iy < N[1]; iy++)
            for (size_t ix = 0; ix < N[0]; ix++) {
                const f64 xv = (f64)ix * L[0] / (f64)N[0];
                const f64 yv = (f64)iy * L[1] / (f64)N[1];
                const f64 zv = (f64)iz * L[2] / (f64)N[2];
                x[iz * N[1] * N[0] + iy * N[0] + ix] = cos(xv + yv + zv);
            }

    matmulK_d(ctx, x, y);

    for (size_t iz = 0; iz < N[2]; iz++)
        for (size_t iy = 0; iy < N[1]; iy++)
            for (size_t ix = 0; ix < N[0]; ix++) {
                const f64 xv = (f64)ix * L[0] / (f64)N[0];
                const f64 yv = (f64)iy * L[1] / (f64)N[1];
                const f64 zv = (f64)iz * L[2] / (f64)N[2];
                const size_t idx = iz * N[1] * N[0] + iy * N[0] + ix;
                ASSERT_CLOSE(y[idx], 2.5 * cos(xv + yv + zv), TOL);
            }

    safe_free((void **)&x); safe_free((void **)&y);
    matmul_ctx_free(&ctx);
}

int main(void) {
    printf("matmulK (real):\n");
    RUN(matmulK_d_planewave_1d);
    RUN(matmulK_d_constant);

    printf("\nmatmulK (d/z consistency):\n");
    RUN(matmulK_dz_consistency);

    printf("\nmatmulK (3D):\n");
    RUN(matmulK_d_3d);

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");
    return tests_failed > 0 ? 1 : 0;
}
```

**Step 2: Run test to verify it fails**

```bash
source /storage/share/intel/ubuntu/setvars.sh
make build/test_matmulK.ex && ./build/test_matmulK.ex
```
Expected: FAIL (matmulK_d/z are stubs producing zeros)

**Step 3: Implement matmulK_impl.inc**

Replace the body of `src/matmulK_impl.inc:10-17` with:

```c
void FN(matmulK)(matmul_ctx_t *ctx, const CTYPE *x, CTYPE *y) {
    const size_t size    = ctx->size;
    const f64   *ltK     = ctx->localTermK;

    FN(kinetic)(ctx, x, y);

#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < size; i++)
        y[i] += ltK[i] * x[i];
}
```

**Step 4: Run test to verify it passes**

```bash
make build/test_matmulK.ex && ./build/test_matmulK.ex
```
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/matmulK_impl.inc tests/test_matmulK.c
git commit -m "Implement matmulK operator with tests"
```

---

### Task 2: Implement dipolar_set_kernel

**Files:**
- Modify: `src/dipolar.c:21-39`
- Modify: `src/bdg_internal.h:183` (add cutoff_radius param)
- Modify: `include/bdg/bdg.h:110` (add cutoff_radius param)
- Modify: `src/setup.c:181,191` (thread cutoff_radius)

**Step 1: Write failing test**

Create `tests/test_dipolar.c`:

```c
#include "bdg_internal.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define TOL 1e-10

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

#define ASSERT_CLOSE(a, b, tol) do { \
    const double _a = (a), _b = (b); \
    if (fabs(_a - _b) >= (tol)) { \
        printf("[FAIL] line %d: %.15e vs %.15e\n", __LINE__, _a, _b); \
        tests_failed++; \
        return; \
    } \
} while(0)

/* ================================================================
 * Test 1: k=0 mode gives longRngInt = 0
 * ================================================================ */
TEST(kernel_k0_is_zero) {
    const size_t N[] = {8, 8, 8};
    const f64    L[] = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};
    matmul_ctx_t *ctx = matmul_ctx_alloc(3, N, L);
    matmul_ctx_set_system(ctx, 0);

    const f64 dir[] = {0.0, 0.0, 1.0};
    dipolar_set_kernel(ctx, 1.0, dir, L[0] / 2.0);

    /* k=0 is at index (iz=0, iy=0, ix=0) = flat index 0 */
    ASSERT_CLOSE(ctx->longRngInt[0], 0.0, TOL);

    matmul_ctx_free(&ctx);
}

/* ================================================================
 * Test 2: For dipole along z, k along z → cos²θ = 1, kernel = 2*f_cutoff
 * For dipole along z, k along x → cos²θ = 0, kernel = -1*f_cutoff
 * ================================================================ */
TEST(kernel_angular_dependence) {
    const size_t N[] = {8, 8, 8};
    const f64    L[] = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};
    matmul_ctx_t *ctx = matmul_ctx_alloc(3, N, L);
    matmul_ctx_set_system(ctx, 0);

    const f64 dir[] = {0.0, 0.0, 1.0};
    const f64 Rc = L[0] / 2.0;
    dipolar_set_kernel(ctx, 1.0, dir, Rc);

    const size_t N0k = N[0] / 2 + 1;

    /* k along z: (iz=1, iy=0, ix=0) → cos²θ = 1, (3*1-1)=2 */
    const size_t idx_kz = 1 * N[1] * N0k + 0 * N0k + 0;
    const f64 kz = 2.0 * M_PI / L[2];
    const f64 kR_z = kz * Rc;
    const f64 f_z = 1.0 + 3.0 * cos(kR_z) / (kR_z * kR_z)
                        - 3.0 * sin(kR_z) / (kR_z * kR_z * kR_z);
    ASSERT_CLOSE(ctx->longRngInt[idx_kz], 2.0 * f_z, 1e-12);

    /* k along x: (iz=0, iy=0, ix=1) → cos²θ = 0, (3*0-1)=-1 */
    const size_t idx_kx = 0 * N[1] * N0k + 0 * N0k + 1;
    const f64 kx = 2.0 * M_PI / L[0];
    const f64 kR_x = kx * Rc;
    const f64 f_x = 1.0 + 3.0 * cos(kR_x) / (kR_x * kR_x)
                        - 3.0 * sin(kR_x) / (kR_x * kR_x * kR_x);
    ASSERT_CLOSE(ctx->longRngInt[idx_kx], -1.0 * f_x, 1e-12);

    matmul_ctx_free(&ctx);
}

/* ================================================================
 * Test 3: c2c kernel at same k-point matches r2c
 * ================================================================ */
TEST(kernel_r2c_vs_c2c) {
    const size_t N[] = {8, 8, 8};
    const f64    L[] = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};

    matmul_ctx_t *ctx_d = matmul_ctx_alloc(3, N, L);
    matmul_ctx_set_system(ctx_d, 0);
    matmul_ctx_t *ctx_z = matmul_ctx_alloc(3, N, L);
    matmul_ctx_set_system(ctx_z, 1);

    const f64 dir[] = {0.0, 0.0, 1.0};
    const f64 Rc = L[0] / 2.0;
    dipolar_set_kernel(ctx_d, 1.0, dir, Rc);
    dipolar_set_kernel(ctx_z, 1.0, dir, Rc);

    /* Compare values at shared k-points (ix in [0, N[0]/2]) */
    const size_t N0k_d = N[0] / 2 + 1;
    const size_t N0k_z = N[0];

    for (size_t iz = 0; iz < N[2]; iz++)
        for (size_t iy = 0; iy < N[1]; iy++)
            for (size_t ix = 0; ix < N0k_d; ix++) {
                const size_t idx_d = iz * N[1] * N0k_d + iy * N0k_d + ix;
                const size_t idx_z = iz * N[1] * N0k_z + iy * N0k_z + ix;
                ASSERT_CLOSE(ctx_d->longRngInt[idx_d],
                             ctx_z->longRngInt[idx_z], 1e-14);
            }

    matmul_ctx_free(&ctx_d);
    matmul_ctx_free(&ctx_z);
}

int main(void) {
    printf("dipolar_set_kernel:\n");
    RUN(kernel_k0_is_zero);
    RUN(kernel_angular_dependence);
    RUN(kernel_r2c_vs_c2c);

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");
    return tests_failed > 0 ? 1 : 0;
}
```

**Step 2: Update API signatures**

In `include/bdg/bdg.h:107-110`, change to:
```c
 * @param g_ddi          Dipolar coupling strength
 * @param dipole_dir     Unit vector for dipole orientation (length 3)
 * @param cutoff_radius  Spherical cutoff radius for regularization
 */
void bdg_set_dipolar(bdg_t *bdg, f64 g_ddi, const f64 *dipole_dir, f64 cutoff_radius);
```

In `src/bdg_internal.h:183`, change to:
```c
void dipolar_set_kernel(matmul_ctx_t *ctx, f64 g_ddi, const f64 *dipole_dir, f64 cutoff_radius);
```

In `src/setup.c:181,191`, change signature and call:
```c
void bdg_set_dipolar(bdg_t *bdg, f64 g_ddi, const f64 *dipole_dir, f64 cutoff_radius) {
    ...
    dipolar_set_kernel(ctx, g_ddi, dipole_dir, cutoff_radius);
    ...
}
```

**Step 3: Run test to verify it fails**

```bash
make build/test_dipolar.ex && ./build/test_dipolar.ex
```
Expected: FAIL (kernel is all zeros from stub)

**Step 4: Implement dipolar_set_kernel**

Replace `src/dipolar.c:21-39` body with:

```c
void dipolar_set_kernel(matmul_ctx_t *ctx, f64 g_ddi, const f64 *dipole_dir,
                         f64 cutoff_radius) {
    ctx->g_ddi = g_ddi;

    const size_t Nx = ctx->N[0], Ny = ctx->N[1], Nz = ctx->N[2];
    const size_t N0k = ctx->complex_psi0 ? Nx : (Nx / 2 + 1);
    const f64 dkx = 2.0 * M_PI / ctx->L[0];
    const f64 dky = 2.0 * M_PI / ctx->L[1];
    const f64 dkz = 2.0 * M_PI / ctx->L[2];
    const f64 dir_sq = dipole_dir[0] * dipole_dir[0]
                     + dipole_dir[1] * dipole_dir[1]
                     + dipole_dir[2] * dipole_dir[2];

    ctx->longRngInt = xcalloc(ctx->k_size, sizeof(f64));

    for (size_t iz = 0; iz < Nz; iz++) {
        const int fz = (iz <= Nz / 2) ? (int)iz : (int)iz - (int)Nz;
        const f64 kz = dkz * fz;

        for (size_t iy = 0; iy < Ny; iy++) {
            const int fy = (iy <= Ny / 2) ? (int)iy : (int)iy - (int)Ny;
            const f64 ky = dky * fy;

            for (size_t ix = 0; ix < N0k; ix++) {
                const int fx = (ctx->complex_psi0 && ix > Nx / 2)
                             ? (int)ix - (int)Nx : (int)ix;
                const f64 kx = dkx * fx;

                const size_t idx = iz * Ny * N0k + iy * N0k + ix;
                const f64 ksq = kx * kx + ky * ky + kz * kz;

                if (ksq < 1e-50) {
                    ctx->longRngInt[idx] = 0.0;
                    continue;
                }

                const f64 kdotd = kx * dipole_dir[0]
                                + ky * dipole_dir[1]
                                + kz * dipole_dir[2];
                const f64 cos_sq_theta = (kdotd * kdotd) / (ksq * dir_sq);

                const f64 k_mag = sqrt(ksq);
                const f64 kR  = k_mag * cutoff_radius;
                const f64 kR2 = kR * kR;
                const f64 f_cutoff = 1.0 + 3.0 * cos(kR) / kR2
                                         - 3.0 * sin(kR) / (kR2 * kR);

                ctx->longRngInt[idx] = (3.0 * cos_sq_theta - 1.0) * f_cutoff;
            }
        }
    }
}
```

**Step 5: Run test to verify it passes**

```bash
make build/test_dipolar.ex && ./build/test_dipolar.ex
```
Expected: All 3 tests PASS

**Step 6: Commit**

```bash
git add src/dipolar.c src/bdg_internal.h include/bdg/bdg.h src/setup.c tests/test_dipolar.c
git commit -m "Implement dipolar_set_kernel with cutoff_radius parameter"
```

---

### Task 3: Implement dipolar_add_meanfield (template)

**Files:**
- Create: `src/dipolar_meanfield_impl.inc`
- Create: `src/dipolar_meanfield_d.c`
- Create: `src/dipolar_meanfield_z.c`
- Modify: `src/dipolar.c:44-57` (add dispatcher)
- Modify: `src/bdg_internal.h` (add meanfield_d/z declarations)

**Step 1: Add tests to `tests/test_dipolar.c`**

Append these tests before `main()`:

```c
/* ================================================================
 * Test 4: Meanfield on uniform density with real wf
 * Uniform ρ = 1 → FFT(ρ) is a delta at k=0 → kernel[k=0]=0
 * So Φ_dd = 0 everywhere, localTermK/M unchanged
 * ================================================================ */
TEST(meanfield_uniform_density_d) {
    const size_t N[] = {8, 8, 8};
    const f64    L[] = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};
    matmul_ctx_t *ctx = matmul_ctx_alloc(3, N, L);
    matmul_ctx_set_system(ctx, 0);
    const size_t size = ctx->size;

    ctx->localTermK = xcalloc(size, sizeof(f64));
    ctx->localTermM = xcalloc(size, sizeof(f64));
    ctx->wf = xcalloc(size, sizeof(f64));
    for (size_t i = 0; i < size; i++)
        ((f64 *)ctx->wf)[i] = 1.0;

    const f64 dir[] = {0.0, 0.0, 1.0};
    dipolar_set_kernel(ctx, 1.0, dir, L[0] / 2.0);
    dipolar_add_meanfield(ctx);

    /* kernel[k=0]=0 means no contribution */
    for (size_t i = 0; i < size; i++) {
        ASSERT_CLOSE(ctx->localTermK[i], 0.0, 1e-12);
        ASSERT_CLOSE(ctx->localTermM[i], 0.0, 1e-12);
    }

    matmul_ctx_free(&ctx);
}

/* ================================================================
 * Test 5: Meanfield d vs z consistency
 * Same real wf, same kernel → same result
 * ================================================================ */
TEST(meanfield_dz_consistency) {
    const size_t N[] = {8, 8, 8};
    const f64    L[] = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};

    matmul_ctx_t *ctx_d = matmul_ctx_alloc(3, N, L);
    matmul_ctx_set_system(ctx_d, 0);
    matmul_ctx_t *ctx_z = matmul_ctx_alloc(3, N, L);
    matmul_ctx_set_system(ctx_z, 1);
    const size_t size = ctx_d->size;

    ctx_d->localTermK = xcalloc(size, sizeof(f64));
    ctx_d->localTermM = xcalloc(size, sizeof(f64));
    ctx_d->wf = xcalloc(size, sizeof(f64));
    ctx_z->localTermK = xcalloc(size, sizeof(f64));
    ctx_z->localTermM = xcalloc(size, sizeof(f64));
    ctx_z->wf = xcalloc(size, sizeof(c64));

    /* Non-uniform wf: cos(x+y+z) */
    for (size_t iz = 0; iz < N[2]; iz++)
        for (size_t iy = 0; iy < N[1]; iy++)
            for (size_t ix = 0; ix < N[0]; ix++) {
                const f64 xv = (f64)ix * L[0] / (f64)N[0];
                const f64 yv = (f64)iy * L[1] / (f64)N[1];
                const f64 zv = (f64)iz * L[2] / (f64)N[2];
                const size_t idx = iz * N[1] * N[0] + iy * N[0] + ix;
                const f64 val = 1.0 + 0.3 * cos(xv + yv + zv);
                ((f64 *)ctx_d->wf)[idx] = val;
                ((c64 *)ctx_z->wf)[idx] = val + 0.0 * I;
            }

    const f64 dir[] = {0.0, 0.0, 1.0};
    const f64 Rc = L[0] / 2.0;
    dipolar_set_kernel(ctx_d, 1.0, dir, Rc);
    dipolar_set_kernel(ctx_z, 1.0, dir, Rc);
    dipolar_add_meanfield(ctx_d);
    dipolar_add_meanfield(ctx_z);

    for (size_t i = 0; i < size; i++) {
        ASSERT_CLOSE(ctx_d->localTermK[i], ctx_z->localTermK[i], 1e-10);
        ASSERT_CLOSE(ctx_d->localTermM[i], ctx_z->localTermM[i], 1e-10);
    }

    matmul_ctx_free(&ctx_d);
    matmul_ctx_free(&ctx_z);
}
```

And add to `main()`:
```c
    printf("\ndipolar_add_meanfield:\n");
    RUN(meanfield_uniform_density_d);
    RUN(meanfield_dz_consistency);
```

**Step 2: Create template files**

Create `src/dipolar_meanfield_impl.inc`:

```c
/**
 * @file dipolar_meanfield_impl.inc
 * @brief Type-generic dipolar mean-field: Φ_dd added to localTermK/M.
 *
 * Computes Φ_dd = IFFT(longRngInt * FFT(|wf|²)) and adds g_ddi * Φ_dd
 * to both localTermK and localTermM.
 */

#include "fft_helpers_impl.inc"

#ifdef CTYPE_IS_COMPLEX
#define CABS(x) cabs(x)
#define CREAL(x) creal(x)
#else
#define CABS(x) fabs(x)
#define CREAL(x) (x)
#endif

void FN(dipolar_add_meanfield)(matmul_ctx_t *ctx) {
    const size_t size   = ctx->size;
    const size_t k_size = ctx->k_size;
    const f64    norm   = 1.0 / (f64)size;
    const f64    g      = ctx->g_ddi;
    const CTYPE *wf     = (const CTYPE *)ctx->wf;
    const f64   *kernel = ctx->longRngInt;
    CTYPE       *buf    = (CTYPE *)ctx->c_wrk1;
    c64         *fk     = ctx->f_wrk;

    /* 1. density |wf|² → c_wrk1 */
#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < size; i++) {
        const RTYPE d = CABS(wf[i]);
        buf[i] = (CTYPE)(d * d);
    }

    /* 2. FFT → f_wrk */
    FN(fft_forward_inplace)(ctx);

    /* 3. multiply by kernel / size */
#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < k_size; i++)
        fk[i] *= kernel[i] * norm;

    /* 4. IFFT → c_wrk1 */
    FN(fft_backward_inplace)(ctx);

    /* 5. accumulate g_ddi * Φ_dd into local terms */
#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < size; i++) {
        const f64 phi = g * CREAL(buf[i]);
        ctx->localTermK[i] += phi;
        ctx->localTermM[i] += phi;
    }
}

#undef CABS
#undef CREAL
```

Create `src/dipolar_meanfield_d.c`:

```c
#define CTYPE f64
#define RTYPE f64
#define CTYPE_IS_REAL
#define FN(name) name##_d

#include "dipolar_meanfield_impl.inc"
```

Create `src/dipolar_meanfield_z.c`:

```c
#define CTYPE c64
#define RTYPE f64
#define CTYPE_IS_COMPLEX
#define FN(name) name##_z

#include "dipolar_meanfield_impl.inc"
```

**Step 3: Add declarations and dispatcher**

In `src/bdg_internal.h`, after the existing `dipolar_add_meanfield` declaration (line 186), add:

```c
void dipolar_add_meanfield_d(matmul_ctx_t *ctx);
void dipolar_add_meanfield_z(matmul_ctx_t *ctx);
```

In `src/dipolar.c`, replace the `dipolar_add_meanfield` stub (lines 44-57) with a dispatcher:

```c
void dipolar_add_meanfield(matmul_ctx_t *ctx) {
    if (ctx->complex_psi0)
        dipolar_add_meanfield_z(ctx);
    else
        dipolar_add_meanfield_d(ctx);
}
```

**Step 4: Run test to verify it passes**

```bash
make build/test_dipolar.ex && ./build/test_dipolar.ex
```
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/dipolar_meanfield_impl.inc src/dipolar_meanfield_d.c \
        src/dipolar_meanfield_z.c src/dipolar.c src/bdg_internal.h \
        tests/test_dipolar.c
git commit -m "Implement dipolar_add_meanfield as type-generic template"
```

---

### Task 4: Implement dipolar_conv

**Files:**
- Modify: `src/dipolar_conv_impl.inc:15-25`

**Step 1: Add tests to `tests/test_dipolar.c`**

Append before `main()`:

```c
/* ================================================================
 * Test 6: dipolar_conv linearity — conv(a*v) = a * conv(v)
 * ================================================================ */
TEST(dipolar_conv_linearity_d) {
    const size_t N[] = {8, 8, 8};
    const f64    L[] = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};
    matmul_ctx_t *ctx = matmul_ctx_alloc(3, N, L);
    matmul_ctx_set_system(ctx, 0);
    const size_t size = ctx->size;

    ctx->wf = xcalloc(size, sizeof(f64));
    for (size_t i = 0; i < size; i++)
        ((f64 *)ctx->wf)[i] = 1.0 + 0.1 * (f64)(i % 7);

    const f64 dir[] = {0.0, 0.0, 1.0};
    dipolar_set_kernel(ctx, 1.5, dir, L[0] / 2.0);

    f64 *v1  = xcalloc(size, sizeof(f64));
    f64 *v2  = xcalloc(size, sizeof(f64));
    f64 *out1 = xcalloc(size, sizeof(f64));
    f64 *out2 = xcalloc(size, sizeof(f64));

    for (size_t i = 0; i < size; i++) {
        v1[i] = cos(2.0 * M_PI * (f64)(i % N[0]) / (f64)N[0]);
        v2[i] = 3.7 * v1[i];
    }

    dipolar_conv_d(ctx, v1, out1);
    dipolar_conv_d(ctx, v2, out2);

    for (size_t i = 0; i < size; i++)
        ASSERT_CLOSE(out2[i], 3.7 * out1[i], 1e-10);

    safe_free((void **)&v1); safe_free((void **)&v2);
    safe_free((void **)&out1); safe_free((void **)&out2);
    matmul_ctx_free(&ctx);
}

/* ================================================================
 * Test 7: dipolar_conv d vs z consistency
 * ================================================================ */
TEST(dipolar_conv_dz_consistency) {
    const size_t N[] = {8, 8, 8};
    const f64    L[] = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};

    matmul_ctx_t *ctx_d = matmul_ctx_alloc(3, N, L);
    matmul_ctx_set_system(ctx_d, 0);
    matmul_ctx_t *ctx_z = matmul_ctx_alloc(3, N, L);
    matmul_ctx_set_system(ctx_z, 1);
    const size_t size = ctx_d->size;

    ctx_d->wf = xcalloc(size, sizeof(f64));
    ctx_z->wf = xcalloc(size, sizeof(c64));
    for (size_t i = 0; i < size; i++) {
        const f64 val = 1.0 + 0.2 * (f64)(i % 5);
        ((f64 *)ctx_d->wf)[i] = val;
        ((c64 *)ctx_z->wf)[i] = val + 0.0 * I;
    }

    const f64 dir[] = {0.0, 0.0, 1.0};
    dipolar_set_kernel(ctx_d, 1.0, dir, L[0] / 2.0);
    dipolar_set_kernel(ctx_z, 1.0, dir, L[0] / 2.0);

    f64 *vd = xcalloc(size, sizeof(f64));
    c64 *vz = xcalloc(size, sizeof(c64));
    f64 *od = xcalloc(size, sizeof(f64));
    c64 *oz = xcalloc(size, sizeof(c64));

    for (size_t i = 0; i < size; i++) {
        vd[i] = sin(2.0 * M_PI * (f64)(i % N[0]) / (f64)N[0]);
        vz[i] = vd[i] + 0.0 * I;
    }

    dipolar_conv_d(ctx_d, vd, od);
    dipolar_conv_z(ctx_z, vz, oz);

    for (size_t i = 0; i < size; i++) {
        ASSERT_CLOSE(creal(oz[i]), od[i], 1e-10);
        ASSERT_CLOSE(cimag(oz[i]), 0.0, 1e-10);
    }

    safe_free((void **)&vd); safe_free((void **)&vz);
    safe_free((void **)&od); safe_free((void **)&oz);
    matmul_ctx_free(&ctx_d); matmul_ctx_free(&ctx_z);
}
```

Add to `main()`:
```c
    printf("\ndipolar_conv:\n");
    RUN(dipolar_conv_linearity_d);
    RUN(dipolar_conv_dz_consistency);
```

**Step 2: Run test to verify it fails**

```bash
make build/test_dipolar.ex && ./build/test_dipolar.ex
```
Expected: kernel tests pass, conv tests FAIL

**Step 3: Implement dipolar_conv_impl.inc**

Replace `src/dipolar_conv_impl.inc` body (lines 15-25):

```c
#include "fft_helpers_impl.inc"
#include <omp.h>

#ifdef CTYPE_IS_COMPLEX
#define CONJ(x) conj(x)
#else
#define CONJ(x) (x)
#endif

void FN(dipolar_conv)(matmul_ctx_t *ctx, const CTYPE *v, CTYPE *out) {
    const size_t size   = ctx->size;
    const size_t k_size = ctx->k_size;
    const f64    norm   = 1.0 / (f64)size;
    const f64    g      = ctx->g_ddi;
    const CTYPE *wf     = (const CTYPE *)ctx->wf;
    const f64   *kernel = ctx->longRngInt;
    CTYPE       *buf    = (CTYPE *)ctx->c_wrk1;
    c64         *fk     = ctx->f_wrk;

    /* 1. c_wrk1 = conj(wf) * v */
#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < size; i++)
        buf[i] = CONJ(wf[i]) * v[i];

    /* 2. FFT → f_wrk */
    FN(fft_forward_inplace)(ctx);

    /* 3. scale by kernel / size */
#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < k_size; i++)
        fk[i] *= kernel[i] * norm;

    /* 4. IFFT → c_wrk1 */
    FN(fft_backward_inplace)(ctx);

    /* 5. out = g_ddi * wf * c_wrk1 */
#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < size; i++)
        out[i] = g * wf[i] * buf[i];
}

#undef CONJ
```

**Step 4: Run test to verify it passes**

```bash
make build/test_dipolar.ex && ./build/test_dipolar.ex
```
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add src/dipolar_conv_impl.inc tests/test_dipolar.c
git commit -m "Implement dipolar_conv perturbation convolution"
```

---

### Task 5: Implement matmulM and final integration

**Files:**
- Modify: `src/matmulM_impl.inc:12-22`

**Step 1: Write failing test**

Create `tests/test_matmulM.c`:

```c
#include "bdg_internal.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define TOL 1e-10

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

#define ASSERT_CLOSE(a, b, tol) do { \
    const double _a = (a), _b = (b); \
    if (fabs(_a - _b) >= (tol)) { \
        printf("[FAIL] line %d: %.15e vs %.15e\n", __LINE__, _a, _b); \
        tests_failed++; \
        return; \
    } \
} while(0)

static matmul_ctx_t *make_1d_ctx(const size_t N, const f64 L,
                                  int complex_psi0, f64 VK, f64 VM) {
    const size_t Narr[] = {N};
    const f64    Larr[] = {L};
    matmul_ctx_t *ctx = matmul_ctx_alloc(1, Narr, Larr);
    matmul_ctx_set_system(ctx, complex_psi0);

    ctx->localTermK = xcalloc(ctx->size, sizeof(f64));
    ctx->localTermM = xcalloc(ctx->size, sizeof(f64));
    for (size_t i = 0; i < ctx->size; i++) {
        ctx->localTermK[i] = VK;
        ctx->localTermM[i] = VM;
    }
    ctx->dipolar = 0;
    return ctx;
}

/* ================================================================
 * Test 1: M without dipolar on cos(x), VM=2
 * M(cos(x)) = (0.5*1 + 2)*cos(x) = 2.5*cos(x)
 * ================================================================ */
TEST(matmulM_d_planewave_no_dipolar) {
    matmul_ctx_t *ctx = make_1d_ctx(64, 2.0 * M_PI, 0, 0.0, 2.0);
    const size_t size = ctx->size;
    f64 *x = xcalloc(size, sizeof(f64));
    f64 *y = xcalloc(size, sizeof(f64));

    for (size_t i = 0; i < size; i++)
        x[i] = cos((f64)i * 2.0 * M_PI / (f64)size);

    matmulM_d(ctx, x, y);

    for (size_t i = 0; i < size; i++) {
        const f64 expected = 2.5 * cos((f64)i * 2.0 * M_PI / (f64)size);
        ASSERT_CLOSE(y[i], expected, TOL);
    }

    safe_free((void **)&x); safe_free((void **)&y);
    matmul_ctx_free(&ctx);
}

/* ================================================================
 * Test 2: M matches K when localTermK == localTermM and no dipolar
 * ================================================================ */
TEST(matmulM_matches_K_no_dipolar) {
    matmul_ctx_t *ctx_K = make_1d_ctx(64, 2.0 * M_PI, 0, 1.5, 1.5);
    matmul_ctx_t *ctx_M = make_1d_ctx(64, 2.0 * M_PI, 0, 1.5, 1.5);
    const size_t size = ctx_K->size;

    f64 *x = xcalloc(size, sizeof(f64));
    f64 *yK = xcalloc(size, sizeof(f64));
    f64 *yM = xcalloc(size, sizeof(f64));

    for (size_t i = 0; i < size; i++)
        x[i] = cos((f64)i * 2.0 * M_PI / (f64)size)
             + 0.5 * cos(3.0 * (f64)i * 2.0 * M_PI / (f64)size);

    matmulK_d(ctx_K, x, yK);
    matmulM_d(ctx_M, x, yM);

    for (size_t i = 0; i < size; i++)
        ASSERT_CLOSE(yM[i], yK[i], TOL);

    safe_free((void **)&x); safe_free((void **)&yK); safe_free((void **)&yM);
    matmul_ctx_free(&ctx_K); matmul_ctx_free(&ctx_M);
}

/* ================================================================
 * Test 3: M d vs z consistency (no dipolar)
 * ================================================================ */
TEST(matmulM_dz_consistency) {
    const size_t N = 64;
    const f64 L = 2.0 * M_PI;

    matmul_ctx_t *ctx_d = make_1d_ctx(N, L, 0, 0.0, 2.0);
    matmul_ctx_t *ctx_z = make_1d_ctx(N, L, 1, 0.0, 2.0);

    f64 *xd = xcalloc(N, sizeof(f64));
    f64 *yd = xcalloc(N, sizeof(f64));
    c64 *xz = xcalloc(N, sizeof(c64));
    c64 *yz = xcalloc(N, sizeof(c64));

    for (size_t i = 0; i < N; i++) {
        xd[i] = cos((f64)i * 2.0 * M_PI / (f64)N);
        xz[i] = xd[i] + 0.0 * I;
    }

    matmulM_d(ctx_d, xd, yd);
    matmulM_z(ctx_z, xz, yz);

    for (size_t i = 0; i < N; i++) {
        ASSERT_CLOSE(creal(yz[i]), yd[i], TOL);
        ASSERT_CLOSE(cimag(yz[i]), 0.0, TOL);
    }

    safe_free((void **)&xd); safe_free((void **)&yd);
    safe_free((void **)&xz); safe_free((void **)&yz);
    matmul_ctx_free(&ctx_d); matmul_ctx_free(&ctx_z);
}

/* ================================================================
 * Test 4: M with dipolar in 3D — verify dipolar term is nonzero
 * Apply to cos(x+y+z) with uniform wf=1, check M differs from K
 * ================================================================ */
TEST(matmulM_d_3d_with_dipolar) {
    const size_t N[] = {8, 8, 8};
    const f64    L[] = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};
    matmul_ctx_t *ctx = matmul_ctx_alloc(3, N, L);
    matmul_ctx_set_system(ctx, 0);
    const size_t size = ctx->size;

    /* Constant local terms, same for K and M */
    ctx->localTermK = xcalloc(size, sizeof(f64));
    ctx->localTermM = xcalloc(size, sizeof(f64));
    for (size_t i = 0; i < size; i++) {
        ctx->localTermK[i] = 1.0;
        ctx->localTermM[i] = 1.0;
    }

    /* Uniform wf */
    ctx->wf = xcalloc(size, sizeof(f64));
    for (size_t i = 0; i < size; i++)
        ((f64 *)ctx->wf)[i] = 1.0;

    /* Set dipolar kernel */
    const f64 dir[] = {0.0, 0.0, 1.0};
    dipolar_set_kernel(ctx, 1.0, dir, L[0] / 2.0);
    ctx->dipolar = 1;

    /* Input: cos(x+y+z) */
    f64 *x = xcalloc(size, sizeof(f64));
    f64 *yK = xcalloc(size, sizeof(f64));
    f64 *yM = xcalloc(size, sizeof(f64));

    for (size_t iz = 0; iz < N[2]; iz++)
        for (size_t iy = 0; iy < N[1]; iy++)
            for (size_t ix = 0; ix < N[0]; ix++) {
                const f64 xv = (f64)ix * L[0] / (f64)N[0];
                const f64 yv = (f64)iy * L[1] / (f64)N[1];
                const f64 zv = (f64)iz * L[2] / (f64)N[2];
                x[iz * N[1] * N[0] + iy * N[0] + ix] = cos(xv + yv + zv);
            }

    matmulK_d(ctx, x, yK);
    matmulM_d(ctx, x, yM);

    /* K and M should differ because dipolar term is nonzero for cos(x+y+z) */
    f64 max_diff = 0.0;
    for (size_t i = 0; i < size; i++) {
        const f64 diff = fabs(yM[i] - yK[i]);
        if (diff > max_diff) max_diff = diff;
    }
    ASSERT(max_diff > 1e-6);

    safe_free((void **)&x); safe_free((void **)&yK); safe_free((void **)&yM);
    matmul_ctx_free(&ctx);
}

/* ================================================================
 * Test 5: M with dipolar d vs z consistency in 3D
 * ================================================================ */
TEST(matmulM_dz_3d_dipolar_consistency) {
    const size_t N[] = {8, 8, 8};
    const f64    L[] = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};

    matmul_ctx_t *ctx_d = matmul_ctx_alloc(3, N, L);
    matmul_ctx_set_system(ctx_d, 0);
    matmul_ctx_t *ctx_z = matmul_ctx_alloc(3, N, L);
    matmul_ctx_set_system(ctx_z, 1);
    const size_t size = ctx_d->size;

    ctx_d->localTermK = xcalloc(size, sizeof(f64));
    ctx_d->localTermM = xcalloc(size, sizeof(f64));
    ctx_d->wf = xcalloc(size, sizeof(f64));
    ctx_z->localTermK = xcalloc(size, sizeof(f64));
    ctx_z->localTermM = xcalloc(size, sizeof(f64));
    ctx_z->wf = xcalloc(size, sizeof(c64));

    for (size_t i = 0; i < size; i++) {
        ctx_d->localTermK[i] = 1.0;
        ctx_d->localTermM[i] = 1.5;
        ctx_z->localTermK[i] = 1.0;
        ctx_z->localTermM[i] = 1.5;
        const f64 wval = 1.0 + 0.1 * (f64)(i % 7);
        ((f64 *)ctx_d->wf)[i] = wval;
        ((c64 *)ctx_z->wf)[i] = wval + 0.0 * I;
    }

    const f64 dir[] = {0.0, 0.0, 1.0};
    dipolar_set_kernel(ctx_d, 1.0, dir, ctx_d->L[0] / 2.0);
    dipolar_set_kernel(ctx_z, 1.0, dir, ctx_z->L[0] / 2.0);
    ctx_d->dipolar = 1;
    ctx_z->dipolar = 1;

    f64 *xd = xcalloc(size, sizeof(f64));
    c64 *xz = xcalloc(size, sizeof(c64));
    f64 *yd = xcalloc(size, sizeof(f64));
    c64 *yz = xcalloc(size, sizeof(c64));

    for (size_t i = 0; i < size; i++) {
        xd[i] = sin(2.0 * M_PI * (f64)(i % N[0]) / (f64)N[0]);
        xz[i] = xd[i] + 0.0 * I;
    }

    matmulM_d(ctx_d, xd, yd);
    matmulM_z(ctx_z, xz, yz);

    for (size_t i = 0; i < size; i++) {
        ASSERT_CLOSE(creal(yz[i]), yd[i], 1e-10);
        ASSERT_CLOSE(cimag(yz[i]), 0.0, 1e-10);
    }

    safe_free((void **)&xd); safe_free((void **)&xz);
    safe_free((void **)&yd); safe_free((void **)&yz);
    matmul_ctx_free(&ctx_d); matmul_ctx_free(&ctx_z);
}

int main(void) {
    printf("matmulM (no dipolar):\n");
    RUN(matmulM_d_planewave_no_dipolar);
    RUN(matmulM_matches_K_no_dipolar);
    RUN(matmulM_dz_consistency);

    printf("\nmatmulM (3D dipolar):\n");
    RUN(matmulM_d_3d_with_dipolar);
    RUN(matmulM_dz_3d_dipolar_consistency);

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");
    return tests_failed > 0 ? 1 : 0;
}
```

**Step 2: Run test to verify it fails**

```bash
make build/test_matmulM.ex && ./build/test_matmulM.ex
```
Expected: FAIL (matmulM stubs produce zeros)

**Step 3: Implement matmulM_impl.inc**

Replace `src/matmulM_impl.inc:12-22` body:

```c
void FN(matmulM)(matmul_ctx_t *ctx, const CTYPE *x, CTYPE *y) {
    const size_t size = ctx->size;
    const f64   *ltM  = ctx->localTermM;

    FN(kinetic)(ctx, x, y);

#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < size; i++)
        y[i] += ltM[i] * x[i];

    if (ctx->dipolar && 3 == ctx->dim) {
        CTYPE *dc = (CTYPE *)ctx->c_wrk2;
        FN(dipolar_conv)(ctx, x, dc);

#pragma omp parallel for simd schedule(static)
        for (size_t i = 0; i < size; i++)
            y[i] += 2.0 * dc[i];
    }
}
```

**Step 4: Run test to verify it passes**

```bash
make build/test_matmulM.ex && ./build/test_matmulM.ex
```
Expected: All 5 tests PASS

**Step 5: Run all tests**

```bash
make run-tests
```
Expected: ALL tests pass (test_setup, test_kinetic, test_matmulK, test_dipolar, test_matmulM)

**Step 6: Commit**

```bash
git add src/matmulM_impl.inc tests/test_matmulM.c
git commit -m "Implement matmulM operator with dipolar support"
```

---

### Task 6: Update TODO.md

**Step 1: Update TODO.md with completed items**

Mark Phase 3 operators (matmulK, matmulM, dipolar) as done.

**Step 2: Commit**

```bash
git add TODO.md
git commit -m "Update TODO: mark dipolar, K, M operators complete"
```
