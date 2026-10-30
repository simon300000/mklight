# MKLight GPU Infrastructure

This document describes how solvers reach a GPU: the backend interface, the
packed data layout every backend shares, the f32 numerics strategy that keeps
GPU results inside the CONVENTIONS §7 error budget, and how results are
validated against the CPU f64 reference. Read `docs/CONVENTIONS.md` first.

| crate | role |
|---|---|
| `mklight-gpu` | `ComputeBackend` trait and job types, the packed layout (`include/mklight_gpu_types.h` + Rust mirror + packer), the double-float / utility-math / reduction / **trace-kernel specifications** in Rust, the CPU-f32 emulation backend, cross-validation against the f64 reference and precision qualification (`xval`) |
| `mklight-metal` | Metal implementation: MSL kernels (`shaders/`), offline `.metallib` build with runtime-source fallback, device/queue/pipeline/buffer wrappers |

Vulkan (AMD/Intel) and CUDA/OptiX (NVIDIA) backends will be further crates
implementing the same trait over the same layout.

## 1. The backend interface

`mklight_gpu::ComputeBackend` is the only GPU interface a solver sees. It is
object-safe (`Arc<dyn ComputeBackend>`) and small:

```rust
trait ComputeBackend: Send + Sync {
    fn info(&self) -> &DeviceInfo;                 // name, memory, unified memory, RT support, SIMD width
    fn precision(&self) -> &PrecisionDescriptor;   // numerics contract + measured probe + error budget
    fn supports(&self, kind: JobKind) -> bool;
    fn upload_systems(&self, systems: &[PackedSystem]) -> MkResult<SystemHandle>;
    fn release_systems(&self, handle: SystemHandle) -> MkResult<()>;
    fn submit(&self, job: Job, ctx: &JobContext) -> MkResult<Pending<JobResult>>;
    // provided: upload_system, run (= submit + wait)
}
```

* **Systems** are uploaded once per snapshot (and wavelength chunk) and are
  immutable on the device, like the `Snapshot`s they come from. One upload can
  hold many systems (optimiser candidates, tolerance samples, zoom
  configurations); rays select one by index. A `SystemHandle` is only valid on
  the backend instance that issued it, and its fields are private: backends
  use it only as a lookup key and check every index against the stored set.
  Uploading runs `PackedSystem::validate` on every record (enum codes, flags,
  `num_asphere`, finiteness, `n > 0`, `α ≥ 0`), so kernels can trust it.
* **Jobs** (`Job`) are batched, backend-neutral descriptions of work over host
  structure-of-arrays data: `PhiloxStream`, `PhiloxBlocks`, `DfOps`,
  `Reduce`, `SagProbe`, and `Trace` (the sequential / path ray-trace
  kernel, §4). Each has a typed `JobOutput`.
  `Job::validate` performs the backend-independent input checks.
* **Asynchrony.** `submit` returns a `Pending<JobResult>` immediately. It can
  be awaited (`Future`), waited on (`wait`, `wait_timeout`) or polled. A
  worker that dies without a result resolves it with an error, never a hang,
  and a worker panic becomes `MkError::Backend` naming the job.
* **Cancellation and progress.** Every submission has its own cancel token.
  The worker stops when either that token or the caller's `JobContext` token
  is cancelled. `Pending::cancel`, or dropping an unfinished `Pending` (a
  superseded request), cancels only that submission. Sibling jobs under the
  same context keep running, and cancelling the context stops all of them.
  Backends split every job into command buffers (launches) of `chunk_items`
  work items (values, for reductions), check cancellation between them, and
  report progress through the `JobContext`. Progress reaches 1 on success,
  also for empty jobs. A cancelled job returns `Err(Cancelled)`, never a
  partial result.
* **Stats.** `JobStats` reports device time (GPU timestamps), wall time and
  the number of command buffers.
* **Reproducibility.** `DeviceInfo::backend_id()` ("metal:Apple M2 Pro") and
  the `PrecisionDescriptor` (serialisable; it includes the measured numerics
  probe and `kernel_build`, i.e. library origin, MSL version and compiler
  flags) go into `RunInfo` together with the seed. Philox streams and reductions are deterministic (see §3), so a
  `(snapshot, settings, backend, seed)` tuple reproduces bit-identical output.
* **Never silently degraded.** A backend that cannot run a job reports
  `supports(kind) == false` and returns `MkError::Unsupported`; the engine then
  runs the CPU reference. The Metal backend also refuses double-float work if
  its start-up numerics probe fails (§3.3).

## 2. The packed system layout

`crates/mklight-gpu/include/mklight_gpu_types.h` is **the single definition**
of every struct and constant that crosses the host/GPU boundary. The Metal
shaders `#include` it; CUDA can include it directly; GLSL (std430) can include
it as well, since it uses only 4-byte scalars, fixed arrays and `#define`s. The
Rust mirror (`layout.rs`) is checked against it three ways:

1. compile-time `size_of` asserts against `MKL_PACKED_*_SIZE`;
2. `tests/c_layout.rs` compiles a probe with the system C compiler
   (`-Wpadded -Werror`) and compares every `offsetof` with `offset_of!`, and
   checks that the header is valid C++;
3. the Metal kernel `mkl_layout_probe` reports `sizeof`/`alignof`/offsets as
   the Metal compiler lays them out, compared in `mklight-metal`'s tests.

The integer `#define MKL_*` constants are parsed by `mklight-gpu/build.rs`
into `mklight_gpu::consts`, so no constant is written twice.

**Layout rules:** only 4-byte scalars and fixed arrays (no `float3`: it is 16
bytes in MSL and 16-byte aligned in GLSL), every struct a multiple of 16
bytes with explicit `reserved` padding, no implicit padding (the Rust mirror
derives `bytemuck::Pod`). Any change bumps `MKL_LAYOUT_VERSION` (3: the
surface record gained ideal-lens vergences, double-float residuals of the
curvature, conic, indices and transforms, and coating codes; the trace
kernel's parameter block, path-step words, ray planes and status codes were
added; now 4: residuals of the ideal-lens power and vergences, the aperture
sizes and the asphere coefficients, and the trace flag
`MKL_TRACE_NEWTON_CAP_NOT_CONVERGED`; now **5**: the coefficient-pool
indices in the former final padding words, extended shape codes, and
`MKL_RAY_EVANESCENT`); the header carries a magic word and
the version, and backends validate both on upload.

**Representability.** The packer rounds each model value once to f32 and
rejects any whose f32 is infinite, **subnormal**, or zero while the model
value is not (for example a curvature of 1e-46). Kernels may compare packed
parameters with exactly `0.0f`, and GPUs flush subnormals (§3.3), so either
case would silently change the system. The only exceptions are rotation
entries and `*_lo` residuals below `f32::MIN_POSITIVE`, which are written as
zeros because their effect is far below one ulp of what they modify.
Everything the trace kernel uses in double-float carries an f32 residual
(`hi + lo` equals the f64 value to ~2⁻⁴⁸): curvature, conic, asphere
coefficients, indices, the transforms, the ideal-lens power and vergences,
and the aperture sizes.

### 2.1 `MklPackedHeader` (176 bytes, one per system)

| field | meaning |
|---|---|
| `magic`, `layout_version` | `'MKLP'`, `MKL_LAYOUT_VERSION` |
| `num_surfaces` | N, including object (0) and image (N−1) |
| `num_wavelengths` | 1 … `MKL_MAX_WAVELENGTHS` (16) |
| `surface_offset` | index of this system's surface 0 in the shared surface array |
| `stop` | aperture-stop index |
| `primary_wavelength` | packed index of the primary wavelength or `MKL_INDEX_NONE` |
| `flags` | `MKL_SYS_OBJECT_AT_INFINITY`, `MKL_SYS_HAS_TILTS` |
| `wavelength_um[16]` | µm |
| `wavelength_source[16]` | index into the model's wavelength list |

Systems with more than 16 wavelengths are packed in chunks
(`PackedSystem::pack_chunks`); the packer refuses rather than truncates.

### 2.2 `MklPackedSurface` (624 bytes)

| field | meaning / units |
|---|---|
| `shape` | `MKL_SHAPE_{PLANE, CONIC, EVEN_ASPHERE, PARAXIAL, COORD_BREAK, ODD_ASPHERE, Q_CON, Q_BFS, DIFFRACTIVE, TOROID, BICONIC, XY, ZERNIKE}` |
| `interaction` | `MKL_INTERACT_{REFRACT, REFLECT, DUMMY}` |
| `aperture`, `aperture_params[4]` (+`_lo`) | `MKL_APERTURE_*`; mm. A fixed semi-diameter becomes `CIRCULAR` with `MKL_SURF_APERTURE_FROM_SD`; an explicit aperture wins |
| `flags` | `HAS_TO_NEXT`, `AXIAL`, `IS_STOP`, `COATED`, `APERTURE_FROM_SD`, `OBJECT`, `IMAGE` |
| `curvature` (+`_lo`), `conic` (+`_lo`) | 1/mm, dimensionless |
| `paraxial_power` (+`_lo`) | 1/f (1/mm) for ideal thin lenses |
| `vergence[16]` (+`_lo`) | ideal lens: paraxial object vergence 1/a (1/mm) per packed wavelength (`mklight_trace::paraxial::ideal_lens_object_vergence`), the conjugate at which it is perfect in phase |
| `asphere_inv_norm2`, `num_asphere`, `asphere[10]` (+`_lo`) | normalised even-asphere terms A4 … A22 (below) |
| `source_index` | surface index in the model |
| `coating`, `coating_r`, `coating_t` | `MKL_COATING_{BARE, IDEAL, TABLE}`; an ideal coating's R and T |
| `to_next_rot[9]` (+`_lo`), `to_next_origin[3]` (+`_lo`), `to_next_shift[3]` (+`_lo`) | transform into the next local frame (below) |
| `n_after[16]` (+`_lo`), `alpha_after[16]` | index after the surface, internal absorption (1/mm), per packed wavelength |

| `coefficient_offset`, `coefficient_count` | first pair and number of pairs in the shared coefficient pool (0 count for legacy shapes) |

**Aspheres.** The sag is `conic(r²) + ρ⁴ Σⱼ asphere[j] ρ²ʲ` with
`ρ² = r²·asphere_inv_norm2`. The packer stores `asphere[j] = A₂ⱼ₊₄ R²ʲ⁺⁴`
with `R = 2ᵉ` chosen near the surface's semi-diameter. Because `R` is a power
of two, the normalisation is exact (it changes only exponents), yet it keeps
A22-scale coefficients (~1e-35) out of the f32 subnormal range and keeps
Horner intermediates O(sag). Terms beyond A22, or coefficient sets that no
exponent can make representable, are rejected.

**Extended shapes (v5).** The surface record stays **624 bytes**; bytes
616 and 620 now index a separate coefficient buffer. An entry is two
4-byte scalars `(hi, lo)`, addressed in pairs, without vector alignment.
`PackedSystemSet` concatenates pools and rebases each surface offset.
Upload validates ranges, coefficient capacities, tags and all pool values
before either kernel reads them. The C and Metal layout probes check the
new offsets as well as the unchanged record size.

The ten-pair prefix is defined by `MKL_COEFF_*` in the header: radius,
x curvature/conic, two coefficient counts, base-shape/Zernike-ordering
code, exact-conic flag, trace order, design order and design wavelength.
It is followed by coefficients A then B (x then y for axis series). A DOE
inserts sixteen double-float `m λ_mm/(2π)` values before its phase
coefficients; the base uses the existing conic/even-asphere fields.
Precomputing these wavelength factors preserves the model wavelength and
signed i32 order without rounding them to a single f32. Design order and
wavelength remain metadata and never rescale the traced phase.

Capacities are odd 20, Q 20, XY 65 (degree 10), Zernike 37 Fringe or 45
Standard, phase 10, and 10 per toroid/biconic axis. Odd and axis power
series are normalized with an exact power-of-two radius chosen near the
aperture, as for even aspheres. Q, XY, Zernike and phase retain their model
normalization radius as hi + lo. QBfs uses the Forbes triangular conversion
once in the packer, storing auxiliary-basis coefficients as hi + lo; the
kernel evaluates the differentiated auxiliary Clenshaw recurrence directly.
No conversion to monomials is involved.

**Transforms** (CONVENTIONS §7.1). `to_next_origin` is the next vertex in this
surface's local frame and `to_next_rot` maps this frame's axes to the next
one's: `p_next = rot · (p − origin)`, `d_next = rot · d`. Subtracting before
rotating keeps the operand small near the next vertex (the `rot·p + t` form
cancels). `to_next_origin_lo` holds the f32 rounding residual of the origin,
so a 1000 mm air gap, whose f32 rounding alone is ~6e-5 mm (above the 2e-5 mm
position budget), can be applied in double-float. Relation to the core:
`origin = ResolvedSurface::to_next.origin`, `rot = to_next.rot` transposed.
The trace kernel applies forward transforms as `rot·p − shift`, with
`shift = rot·origin` precomputed in f64 (both terms exact to ~2⁻⁴⁸ for a
point near this surface's vertex), and backward ones as `rotᵀ·p + origin`.
For an object at infinity surface 0 stores the map from the entrance
(global) frame into surface 1, so a trace's starting frame is always
"surface 0's frame".

**Exact-comparison sentinels.** Kernels may test packed parameters for exactly
`0.0f` (the packer writes an exact zero whenever the model value is zero). No
other float is compared for equality (CONVENTIONS §7.6).

### 2.3 Ray batches

`RayBatch` / `TraceResult` are SoA arrays (§4.2): position and direction in
the local frame of the start surface (f32, with optional double-float
residual planes), optical path as double-float (`opl_hi`, `opl_lo`), the
packed wavelength index, an optional system index, and a per-ray flux
**weight** (empty means 1). The kernel reads `MKL_TRACE_IN_FLOATS` float and
`MKL_TRACE_IN_UINTS` uint planes and writes `MKL_TRACE_OUT_*` planes
(`PreparedTrace::input_planes`, `TraceResult::from_planes`).
`TraceRequest::radiometry` (`Off`, or `Unpolarized` by default) sets how
absorption and the interface `T`/`R` act on the weight. The enum is
`non_exhaustive`: a polarised mode (Jones/Stokes state per ray, as extra SoA
fields) will be an additive variant. `RayBatch`, `TraceRequest` and
`TraceResult` are `non_exhaustive` too.

## 3. Numerics strategy

Apple GPUs have no FP64 (and FP64 on consumer NVIDIA/AMD parts is slow), so
every GPU kernel computes in f32 with the CONVENTIONS §7 techniques:

1. **Local frames.** Rays are carried in the current surface's frame and
   moved with relative transforms only; coordinates stay O(aperture).
2. **Cancellation-free closed forms** (stable conic sag and intersection,
   `1 − x² = (1 − x)(1 + x)`, fma-nested dot products, scaled `hypot`).
3. **Double-float** (`float2` = hi + lo, ≈ 48 significand bits) wherever the
   budget needs more than f32: optical path, OPD against the chief ray, long
   axial translations, and accumulations.
4. **Deterministic compensated reductions.**

### 3.1 Double-float arithmetic

Specified in `mklight_gpu::df` and implemented in
`shaders/mklight_common.h` operation for operation (Joldes, Muller & Popescu,
ACM TOMS 2017):

| operation | algorithm | bound (u = 2⁻²⁴) |
|---|---|---|
| `two_sum`, `fast_two_sum`, `two_prod` (fma) | error-free transforms | exact |
| df + f32 | DWPlusFP | 2u² |
| df + df | AccurateDWPlusDW | 3u² |
| df × f32 | DWTimesFP3 | 2u² |
| df × df | DWTimesDW3 | 4u² |
| df ÷ df | DWDivDW2 | 15u² |
| √df | Karp/Newton correction of `sqrt(hi)` | ≈ 4u² |

Division and square root start from one hardware `/` or `sqrt` and correct it
with exact residuals, so their accuracy does not depend on those being
correctly rounded; bitwise CPU/GPU agreement for them does.

### 3.2 Fast-math is off, and contraction is off

Error-free transforms only work with IEEE round-to-nearest `+ − ×`, a true
fused `fma`, **no reassociation** (fast-math turns `(s − a) − b` into `0`) and
**no implicit contraction** (`a·b + c` silently becoming `fma(a, b, c)` changes
results bit-wise against the CPU emulation). The Metal build therefore uses
`-fno-fast-math -fmetal-math-mode=safe -fmetal-math-fp32-functions=precise
-ffp-contract=off`, and `mklight_common.h` adds `#pragma METAL fp
math_mode(safe)` and `#pragma METAL fp contract(off)` for MSL ≥ 3.2. All
intended fusion is written explicitly with `fma()`.

The runtime compiler has no contraction option: even with
`MTLMathMode.safe` it fuses `a·b + c` at MSL 3.0 (the probe below caught this).
The runtime-source path therefore compiles as MSL 3.2 where available so the
pragma applies, and falls back to 3.0 otherwise, where the probe disables
double-float work.

### 3.3 Numerics probe

At start-up the Metal backend runs `mkl_numerics_probe` on 4096 input triples
(including `c = −fl(a·b)`, where fused and unfused results differ, and
operands with subnormal products) and compares each output bit for bit with
IEEE binary32 on the CPU. The result (`NumericsProbe`) is part of the
`PrecisionDescriptor`:

| property | needed for | Apple M2 Pro |
|---|---|---|
| IEEE add/mul | error-free transforms | yes |
| true fma | `two_prod` | yes |
| no contraction | bitwise CPU/GPU agreement | yes (offline and MSL 3.2 runtime) |
| no reassociation | error-free transforms | yes |
| correctly rounded `/`, `sqrt` | bitwise df ÷, √, sag | yes |
| subnormals preserved | bitwise emulation in the matching environment | **no, flushed to zero** |

`DfOps` and `Reduce` (and later every double-float kernel) are only offered
when `error_free_transforms()` holds.

**Flush-to-zero.** Measured on the M2 Pro: arithmetic (`+ − × fma / sqrt`)
reads subnormal operands as zeros of the same sign and writes subnormal
results as signed zeros (`fma` rounds the exact value once, then flushes).
Comparisons also read subnormal operands as zero. `abs`, negation,
`max`/`min`, selects, loads and stores only move bits. Every f32
specification in `mklight-gpu` (`df`, `fmath`, `reduce`) is generic over an
`FpEnv`: `Ieee`, or `Ftz` with exactly these rules. `CpuF32Backend::matching(&probe)`
picks the device's environment, so the emulation stays bit-identical when
inputs, intermediates or results are subnormal (tested on the device with
operands from 2⁻¹⁵⁰ to 2⁻⁶⁰). Under flush-to-zero the error-free transforms
are exact as long as no error term falls below 2⁻¹²⁶, so double-float values
keep their full accuracy for `|hi| ≥ 2⁻⁷⁸`, far below any optical quantity.

### 3.4 Reductions

`mklight_gpu::reduce` specifies a two-pass tree reduction whose order depends
only on the data size: 256 threads per group, `G = clamp(⌈n/4096⌉, 1, 256)`
groups, grid-stride accumulation per thread (plain f32, Neumaier f32, or
double-float), an in-group binary tree combined in double-float, and a final
single-group tree over the partials. No atomics, so the result is
reproducible run to run and bit-identical to the CPU emulation. Several
channels (Σw, Σwx, Σwy, Σwx², …) reduce in one dispatch. Single-level
Neumaier in f32 loses accuracy over long systematic runs (its compensation
term is itself f32), so it is only used for the short per-thread runs; the
double-float mode is the default for statistics.

For cancellation granularity, the Metal backend runs pass 1 in command
buffers of about `chunk_items` values. A buffer covers several whole channels
when they fit, otherwise a range of groups of one channel (`first_group` in
`MklReduceParams`). Pass 2 then runs over channel ranges. Each group's
partial depends only on its own index, so the split changes no bit (tested
against the one-shot specification).

### 3.5 Random numbers

Philox4x32-10 on the GPU is bit-identical to `mklight_core::rng` (raw words
for any counter/key, and the `(seed, stream)` stream layout of `Philox`). The
canonical f32 uniform is `mklight_core::rng::u32_to_unit_f32` (re-exported as
`fmath::u32_to_unit_f32`): the f64 reference `(k + ½)·2⁻²⁴` (top 24 bits)
rounded toward zero to f32. It equals the reference for `k < 2²³` and is half
an f32 ulp lower otherwise. (`(k + ½)·2⁻²⁴` needs 25 bits for `k ≥ 2²³`;
rounding it to nearest could produce 1.0.) Every operation in the mapping is
exact, so it is identical on every device and on the CPU. CONVENTIONS §6
states the contract: raw words and f32 uniforms are identical across CPU and
GPU. Consumers that need the f64 uniform apply `u32_to_unit_f64` to the raw
words on the host. `Philox::seek(i)` positions the core generator at any
draw (tested against sequential draws).

## 4. The trace kernel

`Job::Trace` runs the sequential ray trace of `mklight-trace` on the
backend. The kernel is specified in Rust (`mklight_gpu::trace_kernel`,
generic over the f32 environment) and implemented in MSL
(`shaders/trace.metal`) operation for operation alike, so the CPU-f32
emulation reproduces the GPU bit for bit (§5).

### 4.1 Algorithm: f32 decisions, double-float geometry

Per surface, the kernel performs the reference's steps (transfer,
intersection, clipping, interaction) with one split:

* **Decisions in f32**, mirroring the reference operation for operation:
  the closest-approach conic intersection with Kahan's discriminant, stable
  roots and vertex-sheet selection; the even-asphere Newton iteration from
  the conic seed with the reference's Miss / NoConvergence / OutsideDomain
  classification; the TIR test of the stable vector Snell law
  (CONVENTIONS §7.3). These decide the status and give a first `t`.
* **Geometry in double-float** (CONVENTIONS §7.4): the ray point and
  direction between surfaces, the transfers (axial fast path `p − (0, 0, t)`
  or `rot·p − shift`, both hi + lo), the closest approach `t₀ = −P·D`,
  `Q = P + t₀D`, one Newton refinement of every hit on the exact line
  (`c(x² + y² + κz²) − 2z`, or `z − sag(r²)`), the unnormalised normal `G = (−cx, −cy, 1 − κcz)` or
  `(−x·g, −y·g, w)`, and the interactions: refraction `D' = μD + bG` with
  `μ = n₁/n₂` from double-float indices and reflection `D' = D + bG`, where
  `b` comes from the f32 law and is then corrected along `G` until
  `|D'| = 1` (twice for refraction). The tangential component — Snell's law
  — is exact by construction. The asphere refinement evaluates the whole
  sag in double-float (the polynomial by Horner from the hi + lo
  coefficients). The **ideal lens** runs entirely in double-float: the
  tangents `t = k/|k_z|`, `t'⊥ = r·t⊥ − X·φ` with the packed hi + lo power,
  the output direction, and the phase `Δ` from both cancellation-free
  `sphere_term` forms with the hi + lo vergence. **Clipping** compares the
  double-float point with the hi + lo aperture sizes (`r² − a²`, `|x| − a`,
  `u² + v² − 1` in double-float). Optical path `Σ n·t` (and `Δ`) accumulates
  in double-float; bulk absorption `Σ α·t` in f32, `exp` from `fmath::exp`
  (a specified f32 exponential; platform `exp` differs between devices).

Why not plain f32: one f32 ulp of a direction (6e-8) moves an image point by
3e-5 mm over a 500 mm throw (the Newtonian sample), a refracted direction
whose norm is two ulps off lengthens a 100 mm segment by 1.6e-5 mm (0.03
waves), refining a hit from the *rounded* closest-approach point shifts the
ray sideways by half an ulp of |Q| and tilts the normal by `c` times that,
and near the critical angle refraction amplifies any normal error by
`1/cos θ'`. The ideal-lens phase `Δ` is O(|X| sin θ) (13 mm for a ray 50 mm
off axis at 15°): evaluated in f32 it was off by ~3e-6 mm per lens, and
four F/2 ideal lenses at 15° exceeded the OPD budget 4× (2e-2 waves; now
1.3e-3, the fixture "ideal relay" below). Half an f32 ulp of a 300 mm
aperture radius is 1.5e-5 mm, more than the 1e-5 mm survival band, and the
f32 polynomial of a deep asphere (sag 55 mm, grazing) put hits 3e-5 mm
off. Each of these was measured and is removed by the corresponding
double-float step above; all f32 parameters the geometry depends on carry
their residual in the layout (§2).

f32 tolerances, the only intended differences from the reference: the
vertex-sheet test uses `1 − κcz ≥ −2⁻²⁰`; asphere Newton converges when
`|g| ≤ tol_abs + tol_rel·(|t| + |z| + |conic sag| + |poly|)` (default
`1e-9 mm + 2⁻²⁰·…`, the f32 resolution of the residual) within
`NewtonSettings::max_iterations`, followed by the double-float refinement.
The default cap is the reference's, 40 steps (the reference's tolerance is
1e-12 mm), so a search is classified by the same rule at the same point:
most rays converge in 2–6 steps and only rays the conic seed serves badly
pay for more (an asphere whose polynomial cancels the conic needs up to
~38). With a cap below the reference's the host sets
`MKL_TRACE_NEWTON_CAP_NOT_CONVERGED`: a search the cap stops is then
`NOT_CONVERGED`, never `MISS`, since the reference would iterate on.

**Extended sag paths.** `surface_math.rs` and `surface_math.h` carry the
same first-order Cartesian jets in f32 and double-float. Horner evaluates
odd/axis/XY/phase polynomials, differentiated Clenshaw evaluates Q, and
Jacobi plus complex Cartesian powers evaluates Zernike without polar
singularities. Toroid and biconic formulas keep their cylinder limits
without dividing by curvature. The odd cone has the reference's axial
normal at the vertex. An exact-conic departure retains the closed-form
intersection and checks its finite normalization domain afterward.

General sag intersections use the reference's y-curvature/conic seed,
positive/negative crossing bracket, Newton step and twenty-step residual
backtracking, with the same Miss/NoConvergence/OutsideDomain decisions.
The f32 convergence tolerance is `abs + rel*(|t|+|z|+|sag|)`; two Newton
refinements on the double-float line and a fresh double-float gradient
produce the carried point and normal. Normalization-domain comparisons in
the refinement use hi + lo and the reference's radial margin. A DOE first
intersects its base using the legacy path, then checks the phase domain.

DOE interaction projects the phase gradient onto the actual base tangent
plane. The selected order adds `m λ ∇surfaceΦ/(2π)` to tangential momentum
and `m λ Φ/(2π)` to double-float OPL before writing the surface record.
Transmission preserves the incident normal hemisphere, reflection reverses
it and uses the incident index. A negative outgoing normal-momentum square
returns `EVANESCENT`, distinct from TIR. Zero order takes ordinary
refraction/reflection; dummy and image surfaces have no phase interaction.
The f32 propagating-order test and double-float direction/phase calculations
have matching CPU and MSL operations, including FTZ.

**Reported state.** The output point and direction are the double-float
ones rounded to f32. The rounding residual of the point along the
direction is moved into the reported optical path (`OPL − n·(δ·D)`), so
`(point, direction, OPL)` describe one ray consistently and an OPD formed
on the host from them does not see the rounding (compare OPLs at the same
point: `OPL_ref + n·(p_gpu − p_ref)·d`).

### 4.2 Inputs, outputs and statuses

Rays are given in the start surface's local frame (`TraceRequest::new`:
sequential from `start_surface` to `end_surface`). The host generates and
aims them with `mklight-trace` and passes them in double-float
(`RayBatch::push_f64`: hi + lo planes, so the f64 ray is not rounded) or in
plain f32 (residual planes empty; the direction is renormalised in
double-float on entry). Outputs (SoA, `TraceResult`): the final point and
direction in the last surface's frame, OPL (hi, lo), weight, bulk
transmittance `exp(−Σαt)`, status word, last surface, steps completed, and
with `RecordMode::AllSurfaces` one record (x, y, z, l, m, n, opl_hi,
opl_lo, weight) and status per step slot (slot 0 the input; a failed ray
repeats its final state).

| `MKL_RAY_*` | `mklight_trace::TraceStatus` |
|---|---|
| `OK` | `Ok` |
| `MISS` | `Miss` |
| `VIGNETTED` | `Clipped` |
| `TIR` | `TotalInternalReflection` |
| `NOT_CONVERGED` | `Failed { NoConvergence }` |
| `OUTSIDE_DOMAIN` | `Failed { OutsideDomain }` |
| `EVANESCENT` | `Failed { EvanescentOrder }` |
| `DEGENERATE` | `Failed { Degenerate }` |
| `NONFINITE` | `Failed { NonFinite }` (at the start surface for non-finite input) |

The surface in the status word is the reference's `surface`.
(`AimNotConverged` and `FieldUnreachable` arise in ray generation, on the
host.)

**Radiometry** (`TraceRequest::radiometry`, unpolarised by default): the
weight is multiplied by bulk transmittance and by the unpolarised core
Fresnel `T` (refraction) or `R` (reflection) at bare refracting surfaces,
`R = 1`, `T = 0` at bare mirrors, the constant `R`/`T` of ideal coatings;
ideal lenses are lossless unless coated. A request whose program touches a
thin-film or tabulated coating with radiometry on is refused with
`MkError::Unsupported` (geometry-only traces run).

### 4.3 Path programs (ghost paths)

A trace runs a **path program**: a list of `PathStep { surface, mode,
backward }` (`MKL_STEP_*` words, at most `MKL_MAX_STEPS`). Each step moves
the ray from the current surface `c` to `c + 1` (forward, through `c`'s
`to_next`) or `c − 1` (backward, through the inverse of `(c − 1)`'s
`to_next`) and interacts there:

| mode | interaction |
|---|---|
| `Sequential` | the surface's own interaction, exactly as `mklight_trace::trace_ray` (forward only; the image never interacts) |
| `Transmit` | refraction into the medium on the far side; dummy surfaces and the image pass; an ideal lens transmits as a lens (packed vergence forward, zero vergence backward). Not at mirrors |
| `Reflect` | specular reflection (any surface but a coordinate break). At a mirror it is the mirror's own reflection (a mirror-type ideal lens with its power); a refracting ideal lens reflects as a plane |

The medium of the gap between surfaces `j` and `j + 1` is `n_after[j]`,
`alpha_after[j]` whatever the travel direction (gap 0 has no absorption for
an object at infinity). The host validates a program against every system
the rays use (`PathStep::validate_program`): every step moves by one
surface in range, sequential steps go forward, there is no reflection at a
coordinate break and no transmission through a mirror, and the direction
of travel through the surface sequence follows each step's physics:

* a transmission (and any step at a coordinate break) keeps it;
* a reflection at a refracting, dummy or image surface (a ghost
  reflection) reverses it;
* a reflection at a **mirror** keeps it, forward or backward. In the
  sequential model both gaps next to a mirror lie on its reflecting side
  (the medium behind it does not exist), so light reflected there goes on
  to the next surface in the direction it was travelling through the
  sequence: forward in a sequential trace, backward for a ghost returning
  through a catadioptric system.

A sequential trace is `PathStep::sequential(s, e)`; a ghost reflecting at
`j` then at `i < j` is

```text
(1..j−1, Transmit) (j, Reflect) (j−1 … i+1, Transmit, backward)
(i, Reflect, backward) (i+1 … image, Transmit)
```

with every mirror on the way passed by `Reflect` instead of `Transmit`.
The Newtonian's sensor ghost, for example, is the sequential path to the
image, `(5, Reflect)`, then `(4, Transmit, backward) (3, Reflect, backward)
(2, Transmit, backward) (1, Reflect, backward)`: back from the sensor by
the fold flat's and the primary's reflections and out of the telescope
(`testutil::newtonian_sensor_ghost`).

`TraceRequest::path(handle, rays, start, steps)` builds such a request;
records then hold one slot per step. Tested with a plate's two-reflection
ghost (straight and tilted/decentred, forward and backward steps) against an
independent f64 path computation (`testutil::f64_path`): points ≤ 1e-6 mm,
directions ≤ 1e-7, OPL ≤ 1e-9 mm, Fresnel weight `T·R·R·T` ≤ 1e-5
relative; `T + R = 1` at a bare interface from one transmit and one
reflect step; the Newtonian's sensor ghost against `f64_path` and against
its known answer (the paraboloid images the axial point perfectly, so the
ghost leaves the primary antiparallel to the incoming beam at the mirrored
pupil point: ≤ 2e-5 mm, ≤ 2e-6); a `Reflect` step at an ideal mirror
identical to the sequential step; and valid and invalid catadioptric
programs.

## 5. Validation

Three implementations are compared, each answering a different question:

```
 mklight-trace (CPU f64 reference) ── budget ──▶ CPU-f32 emulation ── bitwise ──▶ GPU backend
        "is the algorithm right?"        "is f32 precise enough?"       "is the backend/compiler right?"
```

* The **CPU-f32 emulation** (`CpuF32Backend`) runs the same f32 algorithms in
  the same order on the CPU. Differences against the f64 reference are
  precision error and must fit the CONVENTIONS §7 budget
  (`ErrorBudget::CONVENTIONS`); differences against a GPU must be zero (or,
  for `/` and `sqrt` on a device without correctly rounded ones, within the
  stated bound). The trace kernel needs neither: on the M2 Pro (correctly
  rounded `/` and `sqrt`) it is bit-identical.
* Tests on this machine's GPU (skipped without a Metal device):
  Philox known answers and 65 536 random counter/key blocks, a 2¹⁸-block
  stream against the stateful core generator, 64-bit counter carry; two_sum
  and two_prod exact in f64 on the GPU for 65 536 random pairs (fast-math
  would zero the error terms); every df operation bit-identical to the
  emulation and inside its f64 error bound; 10⁶-value sums with the
  double-float GPU result at ≤ 1e-14 relative to the f64 sum while plain f32
  is not; ill-conditioned multi-channel sums; packed sag on the GPU against
  the emulation and the core's f64 sag; layout against the Metal compiler;
  the runtime-compiled library bit-identical to the precompiled one;
  subnormal-range operands and sums bit-identical to the flush-to-zero
  emulation; hypot special values (NaN never masked); chunked reductions
  bit-identical to the specification; chunking, progress (including empty
  jobs), per-submission cancellation, forged handles, tampered records and
  unsupported jobs.
* **Trace kernel.** The GPU's outputs and per-surface records are
  bit-identical to the emulation (in the device's flush-to-zero
  environment) for every sample and fixture below, all fields and
  wavelengths, with radiometry, and under command-buffer chunking; and for
  every two-reflection ghost program of every sample (clipped and not),
  the Newtonian's sensor ghost, plain-f32 inputs, invalid inputs (NaN,
  infinite, zero direction; NaNs need only be NaN on both sides, as IEEE
  leaves their sign and payload open) and the strong-asphere sweeps at
  Newton caps 8 and 40. Against the reference: the cross-validation suite
  (§5.1), ghost path programs (§4.3), the double-float ideal lens against
  `mklight_trace::interact::ideal_lens` (phase ≤ 1e-11 mm, direction
  ≤ 1e-13 over wide fields, fast lenses and both `sphere_term` forms),
  status classification of every failure class (miss, asphere
  non-crossing, non-convergence, outside the sag domain, TIR, vignetting,
  non-finite input), strong aspheres (below), aperture edges of every
  shape at 300 mm (rays 2e-6 mm inside and outside survive as in the
  reference), per-surface records, sequential radiometry against core
  Fresnel along the reference's records (bare and ideal coatings,
  absorbing glass), request validation, cancellation, progress and
  determinism.
* **Strong aspheres** (`testutil::strong_aspheres`, polynomials fighting
  the conic; 4941 rays at ±60° each): every ray the reference traces in up
  to 20 Newton steps (some need 36) has the reference's status and
  position. Longer searches wander between the roots of a line that
  crosses the surface several times, and f32 and f64 may settle on
  different ones (< 1 % of these rays, none in any designed system); for
  rays that never converge the failure class (`MISS` against
  `NOT_CONVERGED`) depends on whether a chaotic walk happened to bracket a
  root, and can differ while both fail. With a cap of 8 no reference hit
  becomes a `MISS`.

### 5.1 Cross-validation suite

`xval::cross_validate` generates rays with the reference tracer (field and
pupil mapping, aiming), passes them in double-float, traces them on the
backend and with `mklight_trace::trace_ray`, and compares every ray that
both complete: position and direction on every surface (Euclidean
distances, from the per-surface records; coordinate breaks, which record
the previous hit in their own frame up to hundreds of mm from their
vertex, are skipped) and at the image, OPD in
waves — for the backend formed on the CPU from its returned OPLs with the
reference sphere of `mklight_trace::opd` (centre and OPL from the backend's
own chief ray, exit pupil from the paraxial data) — and bulk transmittance.
Both traces must end with the same status word; a difference is excused
only for rays whose f64 trace passes within `budget.aperture_edge_mm`
(1e-5 mm) of an aperture edge or within `tir_band` (1e-5) of the TIR
condition `1 − μ² sin² i = 0` at a refracting surface; such rays are counted
("band"), and none occurred.

The suite (`cross_validation_suite_gpu` in `mklight-metal`,
`cross_validation_suite_cpu_f32` in `mklight-gpu`) covers every sample of
`mklight_core::samples`, each also with its automatic semi-diameters frozen
at 0.93× (vignetting every off-axis pupil edge), and fixtures for what the
samples do not reach (`mklight_gpu::testutil`): a tilted/decentred triplet
with a tilted image plane behind coordinate breaks, two ideal lenses at
finite conjugates, four F/2 ideal lenses at 15° ("ideal relay"), every aperture shape (annular obstruction, rectangle,
ellipse, fixed semi-diameter), a block that totally reflects most of its
beam, a fast molded-asphere singlet (conic + A4…A12, ±25°), a 1000 mm-track
f/10 achromat, and a triplet in an absorbing glass. Every field and
wavelength, a 64 × 64 pupil grid, unpolarised radiometry. Measured on an
Apple M2 Pro (the emulation gives the same numbers, bit for bit):

| system | rays | compared | position (mm) | direction | OPD (waves) | band |
|---|---:|---:|---:|---:|---:|---:|
| singlet | 27864 | 27864 | 5.3e-7 | 3.0e-8 | 7.3e-5 | 0 |
| singlet (vignetted) | 27864 | 25846 | 5.3e-7 | 3.0e-8 | 7.1e-5 | 0 |
| achromat | 27864 | 27864 | 5.2e-7 | 3.0e-8 | 1.5e-5 | 0 |
| achromat (vignetted) | 27864 | 24870 | 5.1e-7 | 3.0e-8 | 1.4e-5 | 0 |
| cooke_triplet | 27864 | 27864 | 9.5e-7 | 4.1e-8 | 7.7e-5 | 0 |
| cooke_triplet (vignetted) | 27864 | 25776 | 9.5e-7 | 4.1e-8 | 7.7e-5 | 0 |
| double_gauss | 27864 | 27864 | 9.5e-7 | 4.2e-8 | 1.8e-4 | 0 |
| double_gauss (vignetted) | 27864 | 26650 | 9.5e-7 | 4.1e-8 | 1.8e-4 | 0 |
| newtonian | 27864 | 27864 | 2.5e-6 | 4.2e-8 | 4.8e-6 | 0 |
| newtonian (vignetted) | 27864 | 27726 | 2.5e-6 | 4.2e-8 | 4.8e-6 | 0 |
| petzval_asphere | 27864 | 27864 | 4.8e-7 | 3.1e-8 | 2.8e-5 | 0 |
| petzval_asphere (vignetted) | 27864 | 25644 | 4.8e-7 | 3.1e-8 | 2.8e-5 | 0 |
| relay | 27864 | 27864 | 4.8e-7 | 3.0e-8 | 3.4e-5 | 0 |
| relay (vignetted) | 27864 | 26322 | 3.3e-7 | 3.0e-8 | 3.1e-5 | 0 |
| tilted triplet | 27864 | 27864 | 9.8e-7 | 4.1e-8 | 7.8e-5 | 0 |
| ideal lenses | 27864 | 27864 | 3.2e-7 | 3.0e-8 | 5.6e-9 | 0 |
| ideal relay | 27864 | 27864 | 3.7e-6 | 3.3e-8 | 1.3e-3 | 0 |
| apertures | 27864 | 20184 | 4.9e-7 | 3.0e-8 | 7.1e-5 | 0 |
| TIR block | 18576 | 10034 | 5.8e-7 | 4.1e-8 | 6.5e-4 | 0 |
| asphere singlet | 27864 | 27864 | 1.3e-7 | 4.2e-8 | 7.5e-5 | 0 |
| f1000 achromat | 27864 | 27864 | 2.6e-6 | 3.0e-8 | 2.6e-5 | 0 |
| absorbing triplet | 27864 | 27864 | 9.5e-7 | 4.2e-8 | 3.4e-4 | 0 |

Budget (CONVENTIONS §7): position 2e-5 mm, direction 2e-6, OPD 5e-3 waves.
Positions and directions are at the f32 rounding of the *reported* values
(half an ulp of the coordinate, e.g. 4.8e-7 mm at 8 mm, 1.9e-6 mm at the
Newtonian's 50 mm primary; half an ulp of a direction cosine near 1 is
3e-8). OPD is 4× or more inside the budget. What remains of it is mostly
the same rounding seen by the reference sphere: the reported image point
is off by up to half an ulp across the ray, which changes its OPD by about
`n·δp⊥·(d − d_chief)` (and the chief ray's own rounding moves the sphere's
centre). It grows with image height times numerical aperture: the ideal
relay (27 mm, NA 0.25, ideal lenses now exact in phase: the NA 0.05
"ideal lenses" fixture shows 6e-9 waves) reaches 1.3e-3 waves, the
near-critical refraction of the TIR block 6.5e-4. Survival agreed for
every ray, vignetted ones included. Bulk transmittance agrees to ≤ 4e-8.

### 5.2 Precision qualification and CPU fallback

`xval::qualify(backend, tracer, settings, ctx)` runs a small
cross-validation for one system and returns
`MkResult<Qualification { backend, kernel_build, passed, budget, margin,
budget_fraction, report, reason }>`. The default sample
(`XvalSettings::qualification()`) is, per field and wavelength, 61
hexapolar pupil points (rim ring included) plus 65-point tangential and
sagittal fans: 1719 rays and ~3 ms for the Double Gauss on an M2 Pro.

**Margin.** A sample can miss the worst rays: against the 64 × 64 grid of
§5.1 the sample's maxima were up to 1.8× lower (the TIR block, whose
largest errors sit in a narrow near-critical pupil zone: the hexapolar
pattern alone saw 1/50 of them, hence the fans). The sampled position,
direction and OPD maxima must therefore stay within `margin` (default
**1/3**) of the CONVENTIONS budget. `budget_fraction` reports the largest
measured error as a fraction of the budget (0.05 for the Double Gauss,
0.28 for the ideal relay). Tested with an OPD budget the full grid exceeds
but the sample meets: without the margin the system would qualify, with it
it is rejected.

A backend without the trace kernel, a system the packer cannot represent
in f32 (for example asphere terms beyond A22), a thin-film coating with
radiometry, a backend error or a measurement outside the margin all give
`passed = false` with the reason. Systems with more than 16 wavelengths
are validated chunk by chunk, as the packer splits them. The only error is
`MkError::Cancelled`: a cancelled qualification is no verdict, and the
caller stops instead of falling back. Callers that get a failure run the
CPU reference and **say so**: `Qualification::record(&mut RunInfo)` writes

* on success: `backend = "metal:Apple M2 Pro"`, `precision = F32Gpu`;
* on failure: `backend = "cpu-f64"`, `precision = F64Reference`, and a
  warning "GPU backend metal:… not used (reason); computed with the CPU f64
  reference";

and in both cases the whole qualification (measured errors included) under
`settings["backend_qualification"]`. A qualification holds for one system
(snapshot) on one backend; callers cache it with the packed system.

### 5.3 Throughput

`benchmark_double_gauss_1e6` (ignored by default; `cargo test --release
-p mklight-metal --test trace -- --ignored --nocapture`): 992 244 rays of
the Double Gauss (3 fields, 12 surfaces each), GPU wall time including
writing the input planes into the shared buffer, the kernel and unpacking
the outputs, against `mklight_trace::trace_ray` over the same rays on all
CPU cores (rayon, 12 threads). Apple M2 Pro, best of three, varying ±15 %
with machine load:

| | time | ray·surfaces/s |
|---|---:|---:|
| GPU kernel (device time) | 13.2 ms | 9.0e8 |
| GPU incl. upload/download (wall) | 28 ms | 4.2e8 |
| CPU f64 reference, rayon | 71 ms | 1.7e8 |

The kernel runs ~5× faster than the reference on 12 CPU cores; end to end,
with host-side packing and unpacking of ~110 MB of planes, ~2.5×. (The
double-float aperture tests of layout v4 cost ~5 % of device time.) Chained
device-resident jobs (trace → reduce without readback) are the next
optimisation (§8).

## 6. Metal backend notes

* **Build.** `build.rs` compiles `shaders/*.metal` to `.air` and links one
  `mklight.metallib`, embedded with `include_bytes!`. It finds the compiler
  via `MKLIGHT_METAL_TOOLCHAIN_BIN`, then `xcrun -sdk macosx metal`, then the
  system-mounted Metal toolchain component (which `xcrun` does not always
  resolve). The include-inlined source is always embedded as a fallback;
  `MKLIGHT_METAL_SKIP_OFFLINE=1` forces the fallback for testing.
  `LibraryChoice::{Auto, Precompiled, RuntimeSource}` selects at run time.
  If the strict flag set fails, the build retries without `-Wall -Werror`, and
  then without the math-mode flags. Every set keeps `-fno-fast-math
  -ffp-contract=off`, which `build.rs` asserts. With `Auto`, a precompiled
  library that fails the numerics probe is replaced by the runtime-compiled
  source when that one passes. The flags or options actually used are in
  `PrecisionDescriptor::kernel_build`.
* **Device and pipelines.** `DeviceSelector::{Default, NameContains,
  RegistryId}`. Pipelines are cached by kernel name + function-constant values
  (`mkl_fc_df_op` at index 0, `mkl_fc_reduce_mode` at index 1).
* **Memory and soundness.** `SharedBuffer<T: Pod>` wraps
  `MTLStorageModeShared` buffers (unified memory on Apple Silicon:
  uploads/readbacks are memcpys). Encoding and dispatch (`MetalContext::run`,
  `Encoder`) are crate-private and `unsafe`, because a kernel indexes by its
  parameters, which the type system cannot relate to buffer lengths. Each
  call site states its bounds argument. A buffer the kernel writes is bound
  with `set_buffer_mut(&mut SharedBuffer)`, and the borrow lasts until the
  command buffer has completed, so no host slice of it can exist meanwhile
  on any thread. The public safe surface is `ComputeBackend` plus device,
  library, buffer and pipeline accessors.
* **Trace kernel.** `mkl_trace` runs one thread per ray. Systems, the path
  program and the parameter block are in the `constant` address space (all
  threads of a launch usually read the same surface record); ray planes are
  bound at each launch's first ray, so chunking changes no bit (tested).
  Input planes are written straight into a shared buffer and the output
  buffers skip the zeroing pass (the kernel writes every element).
  `PreparedTrace::new` validates the program, every ray's system and
  wavelength and the Newton settings against the stored set before
  submission, which is what the kernel's unchecked indexing relies on.
  Plane and record element indices are formed in 64-bit arithmetic
  (`mkl_plane`): a recorded batch has `slots · n · 9` floats, more than
  2³² for, e.g., 13 slots and 37 M rays (17 GB, which an M2 Pro allocates).
* **Execution.** Each submission runs on a worker thread: per chunk it checks
  cancellation, encodes, commits, waits, checks `MTLCommandBufferStatus.error`
  (reported as `MkError::Backend` with the Metal error text), and adds
  `GPUEndTime − GPUStartTime` to the stats. Everything runs inside an
  autorelease pool.
* **Ray tracing hardware.** `DeviceInfo::ray_tracing_api` reports
  acceleration-structure support; `hardware_ray_tracing` is set for Apple9+
  GPUs (M3 and later). These are reserved for the non-sequential solver's
  spatial searches; sequential tracing uses analytic per-surface kernels.

## 7. Adding a backend (Vulkan, CUDA/OptiX)

1. Create `crates/mklight-<api>`, implement `ComputeBackend`, and return
   `Unsupported` for any job kind not yet ported.
2. Include `mklight_gpu_types.h` in the kernels (CUDA directly; GLSL via
   `#include` with `GL_GOOGLE_include_directive`, or HLSL/Slang) and port the
   primitives of `mklight_common.h` verbatim, with explicit `fma`, no
   fast-math, and contraction disabled (`-fmad=false` for NVCC; SPIR-V
   `NoContraction` decorations or `precise` in GLSL/HLSL).
3. Add a layout probe kernel and a numerics probe; reuse the Metal tests'
   structure: every kernel compared bit for bit with `CpuF32Backend` and
   against the f64 references.
4. Keep the reduction geometry (`MKL_REDUCE_*`) so sums stay bit-identical
   across backends; use subgroup operations only where they preserve the
   specified order.
5. Report `DeviceInfo` (`unified_memory = false` on discrete GPUs: uploads
   become staging copies, which the interface already hides) and a
   `PrecisionDescriptor`.

## 8. Current limitations

* Thin-film and tabulated coatings are not on the GPU: a trace with
  radiometry through such a surface is `Unsupported` (geometry-only traces
  run). Coating tables (R/T against wavelength and angle) are the planned
  extension, a new `MKL_COATING_*` code with a table offset.
* Ray generation (field/pupil mapping, real ray aiming) runs on the CPU in
  f64; the GPU traces the generated rays. Polarised (Jones/Stokes)
  radiometry is not ported.
* Reported points and directions are f32. Their rounding is inside the
  budget but shows in OPDs formed from them at large image height × NA
  (§5.1); a consumer needing more could be given residual output planes.
* Asphere searches that wander far from the conic seed (> 20 Newton steps)
  can settle on another root than the reference, and the failure class of
  never-converging searches can differ (§5: strong aspheres). Designed
  systems do not reach this.
* Systems are limited by the packer: 16 wavelengths per packed chunk,
  asphere terms up to A22, model values representable as normal f32.
* Job inputs and outputs are host arrays; device-resident buffers chained
  between jobs (e.g. trace → reduce without a readback) are a later
  optimisation that does not change the interface's semantics.
* Command buffers within a job run one after another (commit, wait); with
  the default 2²⁰ items per buffer the gap is negligible next to kernel time.
