# Kinetic Operator Design

## Files
- `src/kinetic_impl.inc` — type-generic implementation
- `src/kinetic_d.c`, `src/kinetic_z.c` — instantiation (unchanged)
- `tests/test_kinetic.c` — planewave tests

## Algorithm

`kinetic(ctx, x, y)` computes `y = -1/2 nabla^2 x` via FFT:

1. Copy x -> c_wrk1 (f64[size] for real, c64[size] for complex)
2. Forward FFT: c_wrk1 -> f_wrk (r2c or c2c)
3. k-space multiply: f_wrk[i] *= 0.5 * k2[i] / size, i in [0, k_size)
4. Backward FFT: f_wrk -> c_wrk1 (c2r or c2c)
5. Copy c_wrk1 -> y

## Approach: new-array execute variants

Use `fftw_execute_dft_r2c` / `fftw_execute_dft_c2r` (real path) and `fftw_execute_dft` (complex path) with explicit buffer arguments. Same plan-bound buffers (c_wrk1, f_wrk), so alignment is guaranteed.

### Branching (#ifdef CTYPE_IS_REAL / CTYPE_IS_COMPLEX)

Steps 1,2,4,5 branch on CTYPE. Step 3 (k-space multiply) is always c64 on f_wrk — no branching.

| Step | Real (r2c/c2r) | Complex (c2c) |
|------|----------------|---------------|
| 1 | memcpy size*sizeof(f64) | memcpy size*sizeof(c64) |
| 2 | fftw_execute_dft_r2c | fftw_execute_dft |
| 4 | fftw_execute_dft_c2r | fftw_execute_dft |
| 5 | memcpy size*sizeof(f64) | memcpy size*sizeof(c64) |

### Note: c2r destroys f_wrk
FFTW c2r destroys the input array. Not a problem since f_wrk is not needed after step 4.

## Tests

All use L=2*pi so k0=1 gives clean wavenumbers. Tolerance ~1e-10.

| Test | Dim | Input | Expected |
|------|-----|-------|----------|
| kinetic_d_zero_mode | 1D N=64 | x=1.0 | y=0 |
| kinetic_d_planewave_1d | 1D N=64 | cos(k0*xj) | 0.5*k0^2*cos(k0*xj) |
| kinetic_z_planewave_1d | 1D N=64 | cos(k0*xj)+0i | match kinetic_d |
| kinetic_d_planewave_3d | 3D 16^3 | cos(kx+ky+kz) | 0.5*3*cos(...) |
| kinetic_d_superposition | 1D N=64 | cos(x)+0.5*cos(2x) | 0.5*cos(x)+0.5*0.5*4*cos(2x) |
