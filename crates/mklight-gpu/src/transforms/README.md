# Resident transforms

`Request` runs through `ResidentOp::Transforms`; results inherit the resident
plan's `RunInfo`, identity, seed and stats. The wave adapters
(`mklight-wave/src/gpu`) build their transform, convolution and correlation
chains from these operations, and the engine routes the wave analyses and
image simulation to them (`docs/WAVE.md` §9, `docs/GPU.md` §9.2).

Grids are contiguous complex f32, row-major `[rows, columns]` or `[length]`,
with an optional element offset. Inputs and outputs are distinct buffers.
FFT bins are unshifted; sign `-1` is forward, `+1` inverse, both
unnormalised before `scale` (a round trip uses `1/product(dimensions)`).

## Schedule

`recipe.rs` is the backend-neutral schedule; `cpu.rs` is its CPU-f32
specification and Metal encodes the same steps.

- **FFT**: Stockham radices 8/4/2 then 3/5/7. A row of ≤4096 points is
  transformed in one threadgroup pass in shared memory (several short rows
  per group); longer rows use global Stockham stages. 2-D = rows, tiled
  transpose, rows, transpose. Twiddles are computed in f64 on the host and
  stored as f32 (`recipe::twiddles`).
- **Other lengths** (Bluestein) and **ChirpZ** use exact chirp convolution.
  A padded row of ≤4096 points is one fused kernel: pre-chirp, FFT, kernel
  spectrum, inverse FFT, post-chirp. Chirp factors are f64 host tables
  (`recipe::chirp_factors`), each phase product exact in f64 and reduced
  modulo a turn. Separable 2-D = chirp rows, transpose, chirp rows, transpose.
- **AffineMft** (symmetric skew map) uses a 2-D chirp convolution whose
  quadratic phases are formed in double-float on the device.
- `Convolve` is **circular** on the common grid (zero-pad for linear).
  `Phase` multiplies by `exp(2πi·(a0x²+a1xy+a2y²+a3x+a4y+a5))`, formed in
  double-float.

Padding is internal and never changes the requested output lattice.

## Physical-grid mapping

```
output[l,k] = scale * sum input[y,x] * exp(sign * 2πi *
    (x*(a0*k + a1*l + a2) + y*(a3*k + a4*l + a5)))
```

For source frequencies `ν = ν0 + dν*(x,y)` (cycles/mm), image positions
`X = X0 + dX*(k,l)` (mm) and a symmetric chief-frame map `B`, use
`[a0,a1,a3,a4] = dν*dX*[B00,B01,B10,B11]` and `[a2,a5] = dν*B*X0`. Compose a
`Phase` for the source-origin factor `exp(sign*2πi*ν0·B*(X0+dX*(k,l)))`:
linear coefficients `sign*dX*Bᵀ*ν0`, constant `sign*ν0·B*X0`. Nonsymmetric
maps are rejected rather than replaced by an interpolated FFT.

## Correlation

`CorrelationStencil`: bilinear complex autocorrelation of a sampled
angular-spectrum grid; the six map coefficients convert f32
`[frequency_count,2]` sensor frequencies to sample shifts. Optional coverage
weights the amplitude; normalisation integrates `coverage*|amplitude|²` and
restores the fractional-cell zero-lag deficit.

`OpticalCorrelation`: wave's contract — separate f32 amplitude and coverage
and double-float unwrapped OPD (waves), mapped through the unit sphere with
a right-handed chief frame, angular scale `sin_max` and `λ(mm)/n'`,
including the sensor Jacobian. Output is the normalised complex OTF. A cone
crossing the sensor horizon is unsupported. `PrepareOpticalStencil` writes
geometry-only f32 `[frequency_count, n*n, 5]` (≤256 MiB) for reuse across
focus planes with the **same** frequencies/frame/scale/grid.

Frequencies are processed in tiles (≤4M samples·frequencies per command)
with the existing double-float reduction tree.

## Resources

Twiddles, chirp tables and chirp spectra live in the bounded operation
cache (≤64 MiB entries; larger spectra are recomputed as plan scratch).
Recipe steps are encoded serially into command buffers of ≤32M complex
elements of work; scratch is reused within the plan and returned to the
pool afterwards. No intermediate host readback. Limits: side ≤8192,
working set ≤1 GiB, polynomial phases ≤1e9 turns — resource limits, not
qualification of every grid or design.
