# P2 resident transforms

`Request` runs through `ResidentOp::Transforms`; final results inherit the
resident plan's `RunInfo`, identity, seed and stats. No engine route is enabled
here. P6 owns wave adapters and P11 owns optical qualification/fallback policy.

Grids are contiguous complex f32, row-major `[rows, columns]` or `[length]`.
Plan views may have a nonzero element offset. All operations use distinct
input/output buffers. FFT bins are unshifted and the sign is explicit:
`-1` forward, `+1` inverse, both unnormalised before the supplied `scale`.
A round trip uses inverse scale `1/product(dimensions)`.

FFT uses fixed Stockham stages with radices 2/3/5/7. Other lengths use exact
Bluestein convolution. ChirpZ uses two row convolutions and tiled transposes;
a symmetric skew AffineMft uses a two-dimensional chirp convolution. Padding
is internal and never changes the requested output lattice. `Convolve` is
**circular** on the supplied common grid: a linear convolution requires the
caller to supply zero-padded inputs. `Multiply` optionally conjugates its
second input. `Transpose` swaps the axes. `Phase` forms its polynomial in
double-float turns and reduces before trigonometry.

## Physical-grid mapping

The MFT contract is

```
output[l,k] = scale * sum input[y,x] * exp(sign * 2πi *
    (x*(a0*k + a1*l + a2) + y*(a3*k + a4*l + a5)))
```

For source frequencies `ν = ν0 + dν*(x,y)` in cycles/mm, image positions
`X = X0 + dX*(k,l)` in mm and symmetric chief-frame map `B`, use
`[a0,a1,a3,a4] = dν*dX*[B00,B01,B10,B11]` and
`[a2,a5] = dν*B*X0`. Compose a `Phase` operation for the source-origin
factor `exp(sign*2πi*ν0·B*(X0+dX*(k,l)))`; its quadratic coefficients are
zero, its linear coefficients are `sign*dX*Bᵀ*ν0`, and its constant is
`sign*ν0·B*X0`. Unequal axis spacings are allowed when the resulting matrix
is diagonal or symmetric. Nonsymmetric maps report unsupported input rather
than substituting an interpolated FFT. Callers keep physical units and grid
origins in `ResidentPlan.settings`.

## Correlation

`CorrelationStencil` is the bilinear **complex** autocorrelation of an
already sampled angular-spectrum grid. Its six double-float map coefficients
convert f32 `[frequency_count,2]` sensor frequencies into sample shifts.
Coverage, when supplied, multiplies the complex amplitude. Normalisation
integrates `coverage*|amplitude|²`; the fractional-cell zero-lag deficit is
restored. No phase unwrapping is inferred from a complex number.

`OpticalCorrelation` preserves wave's other contract: separate f32 amplitude,
f32 coverage and double-float **unwrapped OPD in waves**. It maps the unit
sphere into the sensor using a right-handed chief-ray frame, angular pupil
scale and `λ(mm)/n'`. Shifted amplitude is interpolated linearly and OPD by
coverage weights. The integration and zero-lag correction include the
sensor Jacobian. Its output is a normalised complex OTF; image-reference
translation can be composed separately. A cone crossing the sensor horizon
is explicitly unsupported. The producer must supply valid optical samples,
with zero coverage for unused cells, as the wave pupil boundary does.

`PrepareOpticalStencil` produces geometry-only f32
`[frequency_count, n*n, 5]`: shifted lattice x/y, absolute shifted qz,
source gamma and absolute source qz. It reads only the frequencies and host
geometry descriptor. Bind the resulting view in `OpticalCorrelation.stencil`
for every focus plane, including across plans through a final buffer lease.
It must be paired with the **same** frequencies/frame/scale/grid; changing
only amplitude, coverage or OPD is safe. A stencil is limited to 256 MiB;
tile the frequency list for larger cases. Without a supplied stencil the
same geometry is evaluated per frequency, using only three scalar work planes.

## Resources and qualification

Twiddles, chirp spectra and schedules use the existing bounded operation
cache with full resident identity and content keys. Mutable scratch is local
to a job and recycled after its last consumer. Commands process at most
262144 complex outputs (transpose: 1024 16×16 tiles), or 65536 optical samples.
Correlation uses the existing deterministic double-float reduction tree.
No intermediate host readback occurs. A detailed `transforms` stats entry
supplements P0's coarse `domain` boundary.

The boundary limits a side to 8192 and the estimated transform working set
to 1 GiB, with polynomial phase products within 1e9 turns. These are resource
and arithmetic limits, **not qualification of every grid or optical design**.
Sampled errors must retain GPU §9.4's safety margin. Tests cover both Metal
library origins, f64 DFT/MFT, large phase, round-trip/Parseval, correlation,
concurrency, cancellation, offsets, portable layouts and pool reuse.
The standalone `mklight-metal/src/transforms/benchmark.rs` compiles the
unchanged wave FFT source for matched CPU baselines without adding solver
dependencies to either backend crate. Its temporary manifest and raw evidence
locations are recorded in the task's `P2-evidence.md`.
