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
`MKL_RAY_EVANESCENT`; now **6**: an optional sag-overlay suffix and
`MKL_SURF_HAS_SAG_OVERLAY`); the header carries a magic word and
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
| `aperture`, `aperture_params[4]` (+`_lo`) | `MKL_APERTURE_*`; mm, including the CONVENTIONS §2 edge tolerance. A fixed semi-diameter becomes `CIRCULAR` with `MKL_SURF_APERTURE_FROM_SD`; an explicit aperture wins |
| `flags` | `HAS_TO_NEXT`, `AXIAL`, `IS_STOP`, `COATED`, `APERTURE_FROM_SD`, `OBJECT`, `IMAGE`, `HAS_SAG_OVERLAY` |
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
Upload validates what the kernels index or branch on without checks: the
range, integral counts within the shape's capacity that add up to
`coefficient_count`, the DOE base and Zernike ordering codes, a positive
radius, and every pool value finite and normal. The C and Metal layout
probes check the new offsets as well as the unchanged record size.

The ten-pair prefix is defined by `MKL_COEFF_*` in the header: radius,
x curvature/conic, two coefficient counts, base-shape/Zernike-ordering
code, exact-conic flag, trace order, design order and design wavelength.
It is followed by coefficients A then B (x then y for axis series). A DOE
inserts sixteen double-float `m λ_mm/(2π)` values before its phase
coefficients; the base uses the existing conic/even-asphere fields.
Precomputing these wavelength factors preserves the model wavelength and
signed i32 order without rounding them to a single f32. Design order and
wavelength remain metadata and never rescale the traced phase. Trailing
zero coefficients are dropped (they would only cost per-ray work), so a
QBfs without terms is recognised by its zero count.

Capacities are odd 20, Q 20, XY 65 (degree 10), Zernike 37 Fringe or 45
Standard, phase 10, and 10 per toroid/biconic axis. Odd and axis power
series are normalized with an exact power-of-two radius chosen near the
aperture, as for even aspheres. Q, XY, Zernike and phase retain their model
normalization radius as hi + lo. QBfs uses the Forbes triangular conversion
once in the packer, storing auxiliary-basis coefficients as hi + lo; the
kernel evaluates the differentiated auxiliary Clenshaw recurrence directly.
No conversion to monomials is involved.

**Manufacturing overlays (v6).** `HAS_SAG_OVERLAY` keeps the base's shape
code and pool entries (a DOE's phase included) and appends five pairs
`inv_radius, cos, sin, s2, s4` (`MKL_OVERLAY_*`). With U = x·inv_radius,
V = y·inv_radius, t = U² + V², the departure is
`cos·(U−V)(U+V) + sin·2UV + t·(s2 + t·s4)`, its gradient written out on
scalars. The host sums nested overlays, scales fringes to mm and rotates the
astigmatism (sin/cos of 2θ) in f64; the radius is a power of two near the
aperture, so U and V are exact. A zero departure adds nothing and keeps the
base's exact intersection. The departure has no domain of its own. Systems
with overlays run `mkl_trace_overlays`, the same kernel compiled with the
overlay code in its own translation unit (`trace_overlays.metal`): in one
unit, the overlay code slowed every surface of `mkl_trace` by 3–11 %.

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
single-group tree over the partials. Tree strides below 32 run as SIMD
shuffles, which is the same order. No atomics, so the result is
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
normal at the vertex. A nonzero cone radius whose square underflows uses
a scaled hypot and unit radial gradient; it must not select the vertex
normal (tested at radii ±1e-30 mm). An exact-conic departure retains the closed-form
intersection and checks its finite normalization domain afterward.

General sag intersections use the reference's y-curvature/conic seed,
positive/negative crossing bracket, Newton step and twenty-step residual
backtracking, with the same Miss/NoConvergence/OutsideDomain decisions.
The f32 convergence tolerance is `abs + rel*(|t|+|z|+|sag|)`; two Newton
refinements on the double-float line and a fresh double-float gradient
produce the carried point and normal. Normalization-domain comparisons in
the refinement use hi + lo and the reference's radial margin. A DOE first
intersects its base using the legacy path, then checks the phase domain.
An overlaid surface (a legacy or DOE base included) takes the general sag
path; a DOE checks the phase domain at the refined hit.

DOE interaction projects the phase gradient onto the actual base tangent
plane. The selected order adds `m λ ∇surfaceΦ/(2π)` to tangential momentum
and `m λ Φ/(2π)` to double-float OPL before writing the surface record.
Transmission preserves the incident normal hemisphere, reflection reverses
it and uses the incident index. A negative outgoing normal-momentum square
returns `EVANESCENT`, distinct from TIR. Zero order takes ordinary
refraction/reflection; dummy and image surfaces have no phase interaction.
Unlike TIR, whose f32 decision also supplies the refraction coefficient,
the whole diffraction (normal, phase gradient, tangential momentum, the
propagating-order test and the phase) is one double-float computation.

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

### 5.1.1 Extended-surface qualification (layout v5)

Measured on Apple M2 Pro with the strict offline compiler flags, 2026-10-31.
Each sample and its 0.93× aperture variant has 27,864 rays (64 × 64 pupil,
all fields and wavelengths). GPU and matching FTZ emulation are bit-identical
in every output plane and every surface record. Maxima below include both
aperture variants; no survival mismatch or tolerance-band exception occurred.
All nine samples also pass the default one-third-budget qualification.

| sample | position (mm) | direction | OPD (waves) |
|---|---:|---:|---:|
| hybrid_doe | 8.2e-8 | 3.0e-8 | 1.4e-6 |
| anamorphic | 1.6e-7 | 3.0e-8 | 9.3e-7 |
| biconic | 8.4e-8 | 3.0e-8 | 1.9e-6 |
| toroid | 8.3e-8 | 3.0e-8 | 1.9e-6 |
| q_con | 8.4e-8 | 3.0e-8 | 1.9e-6 |
| q_bfs | 8.3e-8 | 3.0e-8 | 2.0e-6 |
| odd_asphere | 8.4e-8 | 3.0e-8 | 1.5e-6 |
| xy_freeform | 8.4e-8 | 3.0e-8 | 2.4e-6 |
| zernike_sag | 8.3e-8 | 3.0e-8 | 2.4e-6 |

`surface_math::tests::full_capacity_sags_and_gradients_match_f64` checks
all coefficient capacities, both Zernike orderings, negative curvature and
both cylinder orientations in IEEE and FTZ environments against the core's
independent f64 jets (absolute sag/gradient error < 2e-10). Device tests
also upload these systems together, exercising rebased pool offsets and
positive-y sag probes. DOE tests cover orders −2, 0, 1, 3 and 10,000,
transmission, reflection, dummy pass-through, curved bases and evanescence;
forced reflection at dummy/image DOEs also preserves specular direction and
geometric OPL without adding phase;
OPL is compared at the same reported point, within 1e-8 mm. The hybrid's
OPD test includes the phase jump. Design order 2 and design wavelength
0.55 µm are deliberately different from the selected order/wavelengths.

**Steep random rays.** `extended_steep_rays_classify_and_match_f64` uses
Philox seed `0x4d4b4c5355524635`, stream 0, 8,192 rays per full-capacity
fixture: x/y uniform in [−14,14] mm, polar angle in [−80°,80°], random
azimuth, no aperture clipping. The same set, plus DOE order/interaction
variants, is bit-identical between Metal and FTZ emulation. Against f64:

| fixture | equal statuses / 8192 | agreement | common hits |
|---|---:|---:|---:|
| odd, 20 terms | 8113 | 99.036% | 7238 |
| QCon, 20 terms | 8192 | 100% | 3341 |
| QBfs, 20 terms | 8192 | 100% | 3330 |
| toroid, 10 y terms | 8183 | 99.890% | 7349 |
| biconic, 10 terms per axis | 8169 | 99.719% | 5926 |
| XY, 65 terms | 8192 | 100% | 3325 |
| DOE, 10 phase terms on even-asphere base | 8166 | 99.683% | 3336 |
| Fringe, 37 terms | 8192 | 100% | 3327 |
| Standard, 45 terms | 8192 | 100% | 3327 |
| x cylinder | 8191 | 99.988% | 6133 |
| y cylinder | 8192 | 100% | 8151 |

Every common hit meets the full budget: maximum position 1.86e-6 mm,
direction 5.08e-8, same-point OPL 3.90e-12 mm, including DOE phase.
The test requires >99% status agreement for each fixture and no common-hit
geometry/OPL outlier; it does not hide failures in an aperture/TIR band.

Residual classification differences, as `(f64 → f32, count)`:

* Odd: OK→Miss 44, OK→NoConvergence 9, Miss→OK 26.
* Toroid: OK→Miss 4, Miss→OK 5.
* Biconic: OK→Miss 9, Miss→OK 8, OutsideDomain→OK 5,
  OutsideDomain→Miss 1.
* DOE base (the even-asphere path, §5 strong aspheres): Miss→NoConvergence
  12, Miss→OutsideDomain 1, NoConvergence→Miss 10,
  NoConvergence→OutsideDomain 1, OutsideDomain→NoConvergence 2. Both
  implementations fail all 26.
* x cylinder: OutsideDomain→Miss 1; both fail.

These are rays whose **reference classification is itself
ill-conditioned**, not a divergence of the seed or domain logic. The f32
and f64 conic seeds (from the same start) agree on every differing odd,
toroid, biconic and cylinder ray. For all but six of them the seed
misses, so the search starts at the closest approach and Newton walks
towards an extremum of the residual `g(t)` along a line that does not
cross the surface nearby; there `g′` falls to ~10⁻⁴ and the step `g/g′`
amplifies a relative perturbation of the iterate by ~10⁸, so where the
walk lands (a far root, a backtracking failure, the domain edge) is
decided by the last bits. Measured:

| fixture | f64 status changes under ±5e-7 mm / ±5e-8 input perturbations | f32 differences | of which such rays | f64 Newton from the f32-rounded start gives the f32 status |
|---|---:|---:|---:|---:|
| odd | 276 / 8192 | 79 | 63 | 33 |
| toroid | 24 / 8192 | 9 | 8 | 5 |
| biconic | 100 / 8192 | 23 | 15 | 12 |
| DOE base | 80 / 8192 | 26 | 23 | — |
| QCon, QBfs, XY, Zernike, cylinders | 0 / 8192 | 0 or 1 | — | — |

The perturbations (each input coordinate and direction cosine moved by
about the f32 rounding of the ray, twelve variants per ray) flip the f64
reference's own status for most differing rays; under ±1e-12 mm they
flip none of them, so the reference is deterministic but ill-conditioned
there. Running the reference algorithm in f64 from the f32-rounded
closest-approach point and direction already reproduces the f32 status
for 50 of the 111 odd, toroid and biconic differences. The rest are the
same walk decided by the f32 rounding of `g′` itself: the x-cylinder ray
stalls at an extremum (`g = −2.87 mm`) where the f32 `g′` rounds to
exactly zero (a `Miss`) while the f64 `g′ = −8e-8` steps out of the
domain (`OutsideDomain`). The f32-only hits are genuine roots (residual
≤ 2.7e-5 mm at the rounded output point, mostly gradient-amplified
coordinate rounding). No f32 search can reproduce these decisions;
qualification remains mandatory for steep systems, and the sample passes
above do not claim exhaustive qualification of every ray.

### 5.1.2 Manufacturing overlays (layout v6)

`extended_shapes` adds rotated astigmatism plus spherical overlays on plane,
even-asphere, QCon, XY, nested and DOE bases, and the pupil-grid set a
tilted, decentred Cooke triplet with overlays on every lens surface. Metal is
bit-identical to the FTZ emulation; against f64 the common hits stay within
the budget above (Apple M2 Pro: position 1.9e-6 mm, direction 4.2e-8).
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

### 5.4 Extended-shape throughput

Ignored test: `source ~/.cargo/env && cargo test --release -p mklight-metal
--test trace benchmark_extended_shapes -- --ignored --nocapture`.
Apple M2 Pro, strict offline flags, 262,144 axial input rays over an
8 mm radius disk, one refracting surface, radiometry off, no records, all
rays successful. One warm-up round, then the best of ten rounds that
interleave the shapes, so GPU clock ramps and machine load affect them
alike (device rates run to run within ~3 %). Wall includes ray-plane upload and result
unpacking, excludes system upload and request cloning. Columns are
millions of **rays/s** (one surface). Each fixture uses its full
coefficient capacity. The DOE base has two even-asphere terms, so its
relative rate is not a standalone phase cost.

| shape | device M rays/s | wall M rays/s | device / even10 |
|---|---:|---:|---:|
| even, 10 terms | 246 | 57 | 1.000 |
| odd, 20 terms | 104 | 44 | 0.425 |
| QCon, 20 terms | 37 | 25 | 0.150 |
| QBfs, 20 terms | 77 | 38 | 0.313 |
| toroid, 10 y terms | 99 | 43 | 0.403 |
| biconic, 10 terms per axis | 94 | 42 | 0.384 |
| XY, 65 terms | 34 | 23 | 0.138 |
| DOE, 10 phase terms | 220 | 57 | 0.894 |
| Fringe, 37 terms | 10.3 | 8.8 | 0.042 |
| Standard, 45 terms | 7.0 | 6.3 | 0.028 |
| x cylinder | 122 | 46 | 0.495 |
| y cylinder | 121 | 46 | 0.492 |

Sag overlays (`mkl_trace_overlays`, 2026-10-31, same machine; `mkl_trace`
rates unchanged): on a plane 140, even asphere 128, QCon 34, XY 32, nested
on a sphere 140, DOE 111 M rays/s.

The high-order Zernike path evaluates a differentiated Jacobi recurrence
and Cartesian angular powers for each populated term at every Newton
sample/refinement. It is the largest cost here; these measurements use all
terms, while the sample designs have sparse, shorter coefficient lists.

## 6. Metal backend notes

* **Build.** `build.rs` compiles `shaders/*.metal` to `.air` and links one
  `mklight.metallib`, embedded with `include_bytes!`. It finds the compiler
  via `MKLIGHT_METAL_TOOLCHAIN_BIN`, then `xcrun -sdk macosx metal`, then the
  system-mounted Metal toolchain component (which `xcrun` does not always
  resolve). Shaders are compiled offline: an offline compile error fails the
  build, and so does a release build without the toolchain (install it with
  `xcodebuild -downloadComponent MetalToolchain`); only a debug build on a
  machine without it falls back. The include-inlined source is always embedded
  for that fallback; `MKLIGHT_METAL_SKIP_OFFLINE=1` forces it for testing.
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
  never-converging searches can differ (§5: strong aspheres). Steep
  extended-surface searches that walk across an extremum of the residual
  have a classification that f32-sized input changes already flip in the
  f64 reference (§5.1.1); none of the sample grids shows this.
* Systems are limited by the packer: 16 wavelengths per packed chunk,
  even-asphere terms up to A22, the extended capacities in §2.2, and model
  values representable as normal f32 after coefficient normalization.
* Job inputs and outputs are host arrays; device-resident buffers chained
  between jobs (e.g. trace → reduce without a readback) are a later
  optimisation that does not change the interface's semantics.
* Command buffers within a job run one after another (commit, wait); with
  the default 2²⁰ items per buffer the gap is negligible next to kernel time.

## 9. GPU-first architecture

**Accepted direction; implementation plan, not implemented capability.**
MKLight is high-performance software: every GPU-efficient computation runs
on qualified Metal when a device is available. CPU f64 remains the accuracy
reference, fallback and small-work control path; there is no CPU speed-work
programme. Vulkan is deferred. Solvers use the backend-neutral
`ComputeBackend`, typed buffers and explicitly specified kernels, with Metal
encoding confined to `mklight-metal`. This section governs new work; §§1–8
describe the existing implementation and its measured limitations.

### 9.1 Inventory and placement

Source baseline: `mklight-engine/src/{gpu.rs,analysis/gpu.rs}`, the analysis
registry, `mklight-optim/src/{deriv.rs,tolerance.rs,merit/gpu_eval.rs}` and
the solver modules named below. Layout v5 and the existing double-float
trace, reduction and Philox specifications are reused.

Today the five ray-grid analyses and the optimiser evaluator accelerate
**tracing only**. Host f64 generates/aims the rays and reads/reduces the
results. `AUTO_MIN_GPU_RAYS` is exactly 65,536. FD evaluations already batch
their candidate traces; uncompensated tolerance trials do too. Glass
substitution's local refinements already share `GpuTraceEvaluator`
(`analysis/glass.rs`), although `GPU_ANALYSES` omits that registry name.
Jacobian assembly, candidate setup and the outer searches remain on the
host. Wave optics, field maps, layout, ghost irradiance, stray transport and
polarisation have no complete Metal path. These distinctions matter when
measuring the baseline: batching is being extended, not introduced anew.

Sizes below describe workloads to qualify/benchmark, not new input limits.
Let S = surfaces (qualification domain ≤30, track ≤1000 mm), F = fields,
W = wavelengths (packed in chunks of ≤16), P = pupil rays, B = systems,
N = pupil side, M = image side, V = variables, O = operands. Larger domains
need separate qualification; unqualified work retains reported fallback.
Precision tags map to §9.4:
G = geometry (§7 position/direction/survival), OPL = double-float paths and
OPD (§7), R = compensated statistics (§7 RMS/MTF), E = energy/radiometry,
T = transforms, D = derivative/conditioning gates. E/T/D add qualification
tests; they do not relax §7.

| Computation / source | Work size | Placement and reason | Precision |
|---|---|---|---|
| Sequential and path traces, all supported sags, DOE phase/order, refraction/reflection, clipping (`trace`, `trace_kernel`) | FWP, typically 10²–10⁶ rays × S; optional S records/ray | GPU: independent rays with shared surface data | G + OPL; local frames, double-float geometry/phase |
| Field/pupil mapping, real-image-height field solve, chief/pupil Newton aiming (`trace/tracer`, `pupil`) | FW chiefs + FWP solves, up to configured iteration cap | GPU: many independent small Newton solves; cached chief seeds | G; double-float residual/Jacobian/update; preserve requested aim tolerance |
| Automatic semi-diameters, surface profiles/meshes, layout rays (`trace/semidia`, `analysis/layout`) | FW × rim samples × S; 10²–10⁶ vertices | GPU for traces, sag samples and extrema; host for small topology/labels | G; double-float world-to-local differences, f32 display coordinates relative to origin |
| Spot, ray grid/fans, OPD fans, footprint/overlays, longitudinal aberration, lateral colour (`analysis/{spot,fans,illumination,seidel}`) | FWP or 10²–10⁴ fan points; footprint records P×S | GPU trace and reductions; host assembles result metadata/small hull topology | G + OPL + R; chief-relative two-pass moments |
| Geometric EE/ensquared energy, geometric MTF, best focus (`analysis/energy`) | P order statistics; P×frequencies; P×focus trials | GPU stable sort/scan, complex sums and focus merits; scalar Brent decisions may stay host | R; fixed trees, stable radius/index tie order |
| RI curve/map, CRA, pupil aberration, grid distortion, field curvature (`analysis/{illumination,field_maps,curvature}`) | 5×5 default maps; 12×48 RI nodes/field, about 15 traces/node; dense 33²–129² field sweeps | GPU: fields, quadrature and differential pencils are parallel; include nested image-point aiming | G + D + E; double-float differences, sag Hessians and Coddington matrices |
| First order, Seidel, chromatic focal shift, model/surface operands, solves/pickups (`trace/{paraxial,solves}`, `analysis/{system_data,seidel}`) | O(SW) for one design; BSW in searches | Host for a single small report/dependency walk; GPU across candidate/wavelength batches and dense sag extrema | f64 control/reference; double-float batched recurrences, G/D end-to-end |
| Pupil lattice, rim, OPD, resampling, phase/amplitude, wavefront statistics and Zernike fit (`wave/{traced,interp,pupil,zernike}`) | default 65² trace lattice; N=64–1024 automatic, 16 subpoints/cell; ≈37 fit columns | GPU trace, boundary probes, triangulated-grid evaluation and tall-skinny QR; host may chain a small rim topology | OPL + G + R + D; no normal equations for ill-conditioned fits |
| FFT PSF, chirp/affine MFT, OTF/MTF direct/FFT, field/through-focus MTF (`wave/{fft,psf,mtf}`) | N² pupils; M≤2048 automatic; F×W×focus/frequency batches | GPU: regular transforms, stencils and spectral sums, with resident pupils/stencils | T + OPL; phase formed/reduced in double-float, complex f32 arrays |
| Huygens PSF/MTF (`wave/psf`) | image points × covered pupil samples; existing cap 2×10¹⁰ terms, M≤1024 | GPU tiled direct sums; bounded tiles preserve cancellation | OPL + T + E; rationalised distance differences, compensated complex sums |
| Diffraction EE, line spread, Strehl, detector response (`wave/{energy,detector}`) | M² pixels, radii/frequencies and Bessel kernels | GPU shared radial-bin/stencil reductions; isolated scalar analytic reference curves may stay host | R + T + E; normalisation, Parseval and known answers |
| Image simulation (`wave/image_simulation`) | field-grid PSFs; height×width×C samples, C∈{1,3}; haloed convolution tiles | GPU pattern generation, inverse distortion, resampling, RI, FFT convolution, pixel integration and noise; file decode/encode host | G + T + E; preserve linear light, low-y row zero, halo flux |
| Coating TMM/tabulated response, absorption, transmission, PRT/Jones/Stokes pupil maps (`core/materials/coating`, `trace/{radiometry,polarization}`) | FWP×S; angle/wavelength/stack tables; 4 complex Jones entries/ray | GPU for rays and table batches; host for a tiny isolated lookup and reference table checks | E + G + OPL for phase; complex f32, double-float where conditioning needs it |
| Ghost enumeration/paraxial ranking (`ghost/{path,paraxial}`) | two-bounce O(S²); filtered higher orders ≤8 reflections | Small branch-and-bound enumeration stays host; large path/wavelength ranking batches GPU | exact integer signatures/order; double-float first-order terms |
| Ghost irradiance/focus (`ghost/{sequence,irradiance}`) | path×FW×pupil², refinement ≤4 levels; measured 0.3–4 M distinct probes on Double Gauss | GPU path probes, stable dedup/compaction and segmented maps; host controls refinement levels | G + E; separate s/p products, virtual-segment decisions and exact coverage |
| Stray sources, BVH queries/intersections, BSDF/TIS, branch transport, tallies/errors (`stray/*`) | 20 k engine-default rays, benchmark 400 k; tens–hundreds of primitives, split path trees | GPU wavefront queues and deterministic tallies; small scene topology build host; GPU batched bounds/build when large | G + E + exact Philox; double-float close hits/ledger/moments |
| Merit and constraints, FD Jacobians, LM/trust-region, global populations (`optim/*`) | B×O residuals, B=V or 2V for FD; e.g. V=8–100, O=10²–10⁴ | GPU candidate preparation, solver operands, residuals, differences, J products; batched/large QR solves GPU. Small serial trust-region/acceptance decisions stay host | underlying G/OPL/R/T/E plus D; residuals/differences retained as hi+lo |
| Tolerance sensitivity/Monte Carlo and compensators (`optim/tolerance`) | 2×tolerances; 10²–10⁵ trials; nested compensator FD batches | GPU perturbations, trial evaluation, batched compensator work and statistics; host decisions per active trial | D + exact trial RNG; fixed trial order and per-trial outcomes |
| Glass substitution (`optim/{glass,discrete,mixed}`, engine glass job) | catalogue filters, 10²–10⁴ choices; combinations × local optimisation | Host names/filters/search control; GPU numeric catalogue sweeps and all candidate/refinement batches | D; final f64 reference scores/ranking retained |
| Material dispersion/thermal offsets, spectra, sensor tables, glass_info/glass_map (`core/materials`, `analysis/{spectrum,glass_map}`) | few SW values or 10³–10⁶ catalogue/λ/candidate values | Host small immutable resolution; GPU large numeric sweeps, reused tables and candidate updates | double-float indices/dispersion; require resulting G/OPL and E gates |
| RNG stream, random pupils, optimiser/tolerance/image/stray draws | 1–10⁷ counter blocks | GPU where consumed; tiny diagnostic stream may stay host | exact u32 and specified f32 conversion, CONVENTIONS §6 |
| Validation, catalogue/project parsing, snapshots/hashes, schemas, result packing, session/jobs/FFI | small irregular graphs/strings, bulk final copies | Host: authoritative boundary validation, I/O and control, no efficient optical GPU work | exact model/metadata; f64 reference values preserved |

Memory planning is part of dispatch: v5 surfaces cost `176 + 624S` bytes
plus coefficient pools per wavelength chunk; a resident ray with hi+lo
position/direction/OPL already needs 56 bytes before weights/status. One
million rays is therefore tens of MiB; all-surface records cost about
`40P(S+1)` bytes even before extra residuals. A complex f32 2048² plane is
32 MiB; chirp padding needs length ≥N+M−1 and multiple scratch planes.
Batch/chunk by the device working-set limit, retaining logical indices and
reduction trees. Do not materialise every candidate×ray×surface or every
Huygens pair. Explicit larger grids are tiled or reported unsupported, not
silently resized.

### 9.2 Shared primitives and interfaces

These are extensions of the existing trait, not a second backend API.
P0 freezes the following typed contracts before the parallel wave. Names
below are proposed Rust types; the C layout umbrella remains the authority.

| Contract | Inputs → outputs | Owner |
|---|---|---|
| `ResidentPlan`, `BufferHandle`, `BufferView`, `PlanStats` | validated typed operations/views → pending resident outputs, explicit final readback and per-stage stats | P0 |
| `GenerateAim`, `TraceResident`, `ProjectPupil` | system set + logical field/λ/pupil indices → hi+lo rays/status; trace → hi+lo hits/OPL; reference descriptor → OPD, pupil coordinates | P1 |
| `TransformPlan`, `CorrelationStencil` | complex grids, dimensions/strides, sign/scale, affine map → complex grids or frequency values | P2 |
| `SegmentReduce`, `StableCompact`, `StableSort`, `Scan`, `QrFit` | indexed typed resident channels → moments, ordered records, segment outputs or fit factors | P3 |
| `SystemBatchPlan`, `CandidateMap` | immutable topology/material tables + ordered parameter vectors/configurations → resident system set, derived first-order data, candidate status/index map | P4 |

#### Resident interface gate (P0)

The compiling API is `mklight_gpu::resident` and the ten package modules
in the table below. Engine dispatch is unchanged. Only the compatibility
`TraceUpload` and `Sum` operations execute in P0; all domain handlers
return `Unsupported` until their owners qualify them. The solver crates
(analysis, wave, ghost, stray, optim) depend on `mklight-gpu` and declare an
empty public `gpu` adapter module for their package to fill in.

`ComputeBackend` adds object-safe methods:

```rust
fn fill_buffer(&self, desc: BufferDesc,
    fill: &mut dyn FnMut(BufferFill<'_>) -> MkResult<()>) -> MkResult<BufferHandle>;
fn import_buffer(&self, desc: BufferDesc, data: HostAllocation) -> MkResult<BufferHandle>;
fn read_view(&self, view: BufferView) -> MkResult<ReadView>;
// Copying conveniences for existing owned arrays:
fn upload_buffer(&self, desc: BufferDesc, data: BufferData) -> MkResult<BufferHandle>;
fn readback(&self, view: BufferView, ctx: &JobContext) -> MkResult<Readback>;
fn submit_plan(&self, plan: ResidentPlan, ctx: &JobContext) -> MkResult<Pending<PlanResult>>;
fn upload_systems_cached(&self, key: &SystemCacheKey, systems: &[PackedSystem])
    -> MkResult<SystemHandle>;
fn resident_cache_stats(&self) -> ResidentCacheStats;
```

**Zero-copy host access (unified memory).** Producers write device storage in
place and consumers read it in place; the borrow checker enforces the rules.

- `fill_buffer` takes a pooled `MTLStorageModeShared` allocation (contents
  unspecified) and passes it to the callback, which must write every element
  through `BufferFill::as_mut_slice::<T>()` (`f32`, `u32`, `Df32`, `[f32; 2]`,
  checked against the descriptor). The slice cannot escape the callback, and
  nothing else can see the allocation until it is published afterwards. An
  `Err` from the callback returns the allocation to the pool. Long fills check
  their own job context.
- `import_buffer` publishes a `HostAllocation` (zeroed, 16 KiB-aligned,
  16 KiB-multiple pages) without copying, via
  `newBufferWithBytesNoCopy`. Metal's deallocator block owns the allocation
  and frees it when the buffer's last reference goes; imports never enter the
  pool. A `Vec` cannot be imported: its alignment and capacity do not
  qualify.
- `read_view` returns a `ReadView` of a contiguous view of a published
  buffer. Published buffers are immutable and complete: kernels write only
  plan-owned scratch through `&mut` bindings, and a plan publishes only after
  its last command buffer has completed. The guard pins the allocation, so the
  pool cannot reuse it, and it survives dropping the handle, the result or the
  backend. `as_slice::<T>()` and `words()` borrow the guard.
- `readback` copies once into owned `BufferData` (a memcpy when contiguous)
  and also handles strided views. `upload_buffer` copies an owned array.

`PlanStats.copied_bytes` counts bulk `SharedBuffer::from_slice` copies made
while a plan executes. It is zero for the trace/sum, aiming/OPD and transform
paths, cold tables included. Small cached path programs and `setBytes`
parameter blocks are written directly and are not counted.
`ResidentCacheStats.copied_bytes` accumulates the bytes copied by
`upload_buffer` and `readback`. CPU-f32 implements the same API on
`HostAllocation` storage, so typed views, `Df32` alignment included, behave
identically there. A discrete device would stage at fill publication and at
view acquisition behind the same API. None is implemented, and
`DeviceInfo.unified_memory` stays authoritative.

`ResidentPlan::new(snapshot)` holds ordered `operations`, typed `buffers`,
final `outputs`, `settings`, `seed` and `PlanIdentity` (library, derived
snapshot, configuration and spectrum identities, requested backend and optional
qualification report). `import` adds an
immutable lease view; `scratch` declares a plan-local destination. `BufferId`
is an index into this plan, never a global resource ID. Submission validates
shape/extent/type, ownership, dependencies and single writers. Imported
buffers cannot be destinations. Uninitialised reads, duplicate writers,
foreign handles and released system handles fail before encoding. Final
IDs must be produced buffers; repeated final IDs share a lease. A lease
cannot become stale while held, including by a pending submission; pooled
allocations receive a new generation when republished. A cancelled worker
waits for its last committed command before releasing any resources.

`BufferDesc { element, shape }` is contiguous row-major. Element types are
`F32`, `U32` (4 bytes), `DoubleFloat {hi,lo}` and `Complex32 {re,im}` (8 bytes).
`BufferView { buffer, offset, desc, stride }` and plan-local `PlanView`
measure offsets/strides in **typed elements**, with the stride applied to
the flattened row-major index. P0 plan imports and sums require stride 1;
`readback` accepts a strided view; `read_view` requires stride 1. Empty buffers are legal;
a sum requires positive `[channels,count]`. Scratch destinations come from
an exact-size pool and start with unspecified contents: an operation writes
every element it publishes. Rust request structs contain host metadata/IDs
and are never memcpy'd to a shader. The C header umbrella includes one empty
`include/mklight_<module>_types.h` per package for its numeric records; its
`MKL_*` defines join the generated Rust constants. Packed system layout v6
and all existing trace/reduction layouts and constants remain unchanged.

`TraceUpload::new(request, output)` shares its immutable host request between
plan clones and adapts the existing host ray batch without altering the
trace algorithm. The four
outputs are f32 `[10,rays]`, u32 `[3,rays]`, f32
`[record_slots,rays,9]`, and u32 `[record_slots,rays]`, using the existing
`MKL_TRACE_*` plane order. No-record buffers have zero slots.
`Sum { input: PlanView, mode: ReduceMode, output: BufferId }` reads a
contiguous f32 `[channels,count]` view and writes `DoubleFloat [channels]`
using the existing two-pass fixed tree. It sums every supplied value;
masking/compaction belongs to P3. The proof chain selects trace planes by
view offset, with no intermediate host array or upload. The legacy adapter
still uploads its initial host rays on each call; P1 owns resident generation.

| Package module | Frozen request variants / output records |
|---|---|
| `aiming` | `GenerateAim`, `TraceResident`, `ProjectPupil`; `ResidentRays`, `ResidentHits`, `PupilReference` |
| `transforms` | `TransformPlan`, `CorrelationStencil`; explicit dimensions, sign, scale and double-float affine coefficients |
| `collectives` | `SegmentReduce`, `StableCompact`, `StableSort`, `Scan`, `QrFit`; explicit IDs/counts, factors and rank/condition destinations |
| `systems` | `SystemBatchPlan`, `CandidateMap`; parameter/dependency/material views and numeric system/first-order sidecars |
| `geometry`, `radiometry` | numeric geometry settings/samples and coating-stack/hit records, with status destinations |
| `wave_ops`, `ghost_ops`, `stray_ops`, `optim_ops` | numeric solver records and settings views; ghost/stray share `ghost_ops::Coverage` |

Exact field names and types are in each `src/<module>/mod.rs`; these are
owned by the named package after P0. `SystemSet` accepts an uploaded handle
or plan-local header/surface/coefficient sidecars, so P4 output can feed P1
without host materialisation. `ResidentRays` uses the 15 existing input
float planes (including position/direction residuals), two index planes and
separate two-word logical IDs. `ResidentHits` adds status and optional
record/residual sidecars; P1 must retain hi+lo geometry internally.
The central routers invoke the frozen `validate(request, plan)`,
`accesses(request) -> Access { reads, writes, systems }`, and
`execute(request, execution, control)` hooks. Referenced immutable systems
are pinned at submission, and both execution contexts provide a system
lookup and the operation-cache hooks `cache_key(operation)`, `cached<T>` and
`cache<T>` for package handlers. CPU execution exposes word
slices; Metal execution exposes typed view bindings, exclusive output
leases, pooled temporaries and the existing command encoder. New owners
change these handlers without editing crate roots or central routers.

The pool holds at most 256 MiB of idle storage per backend; immutable
system caching holds at most 64 entries and 64 MiB. A separate immutable
operation cache (64 entries / 64 MiB) serves aim seeds, transform plans,
twiddles, pupils and stencils; `cache_key(operation)` combines the plan's
snapshot/library/configuration/spectrum, layout, build and settings
identity with the operation name and plan version. P0 uses it for path-program buffers. The pipeline cache
holds at most 128 specialisations; live jobs retain evicted objects. Active leases/handles
pin resources outside idle-cache limits. System cache identity includes
snapshot/library/configuration/spectrum, layout, build and operation
settings, plus a digest of the actual packed content. Cache hits issue
independently releasable handles. P0 caps resident trace commands at 262,144
rays; sums use the backend chunk and partition reduction groups without
changing the tree. The idle pool reuses the most recently returned buffer
of the exact size and evicts the oldest. Limits
are resource/cancellation controls, not engine backend selection thresholds.

**Resident command fusion.** The Metal executor owns a worker-local deferred
command buffer. Trace tiles and fixed-tree sum passes share its serial compute
encoder, with buffer memory barriers after dispatches. Existing tile boundaries
remain cancellation checkpoints; accumulated dispatch work (2²⁰ threads) or
10 ms of encoding triggers a commit and completion wait. These are work/encoding
bounds, not a promise of 10 ms device time for an arbitrary kernel. A genuine
host read or publication flushes the buffer. Recycled scratch stays pinned
until completion; abandoned, uncommitted encoding is discarded on cancellation
or error. Internal shared-storage guards reject host slices while a pending
GPU write exists. No pending storage becomes a public `ReadView`.
`PlanStats` counts actual completed command buffers; deferred dispatches have
zero commands at encode time and their timestamps/counts appear in the
`fused completion` stage. Arithmetic and reduction trees are unchanged.

`PlanResult` contains final leases, `PlanStats` and `RunInfo`. Stats include
per-stage wall/device/max-command times, commands, allocations/reuse,
uploaded/readback, copied and parameter-block bytes, peak live plan storage and fallback/polish counts.
The run records the actual backend/precision/build/probe, version, identity,
operation manifest, seed and stage stats. Domain-specific
settings belong in `ResidentPlan.settings`. Explicit `Readback` reports its
own copied bytes and wall time. Worker wall time excludes caller-side
submission/validation; benchmark end-to-end wall time includes them and final
readback.
System transfers/cache retention are reported separately by
`resident_cache_stats`; peak plan storage excludes cached systems, idle
pool storage, immutable operation caches and host metadata. P0 returns errors for unsupported/resource
failures and cancellation; fallback policy remains with the engine adapter.

**Residency.** A backend owns its queue, pipelines, system cache and buffer
pool for its lifetime. Views carry type, shape, stride, owner and generation;
validate at upload/plan submission, not in every kernel. Leases pin buffers
until all dependent command buffers finish, including cancelled jobs;
mutable scratch is never shared between live jobs. Bounded caches use the
derived snapshot, library, configuration/spectrum, layout and kernel build
as keys; optical caches also include all sampling/radiometry settings.
Warm calls reuse allocations, packed systems, aim seeds, transform plans,
twiddles/chirps, pupil grids and stencils. Upload only changed candidate
parameters; read back requested final buffers/scalars, never intermediate
trace arrays just to upload them again. Keep current host `Job` calls as
adapters. Solvers hold no Metal objects. A small ordered operation list
suffices; this is not a general graph compiler.

**Aiming and OPD.** Generate all current field kinds, finite/infinite and
telecentric launches, vignetting and pupil patterns on device. Cache
per-system/field/λ chief aims and Jacobians. Newton–Broyden uses double-float
stop hits, finite differences, 2×2 solves and updates, the reference's
continuation schedule and configured cap/tolerance (normally 1e-10 relative
to stop radius). A rounded f32 stop hit cannot drive that solve. Compact
only unresolved rays for optional CPU f64 polish; scatter them back at their
original indices and record counts/reasons. A failed aim remains a failed
aim; never silently accept an inaccurate ray or polish the whole grid by
default.

Keep hi+lo point/direction/OPL through pupil construction. Use the chief's
unclipped trace and the signed exit-pupil distance for the reference sphere;
support centroid references via the shared reduction, both sphere sheets,
telecentric limit, finite conjugates and negative propagation. Evaluate
`OPL_chief − OPL_ray` and the rationalised sphere correction of WAVE §3 in
double-float before conversion to waves. Reference and ray states must use
the same point convention: if exporting rounded geometry, apply the §12
OPL correction once; internal unrounded geometry needs no such correction.
Lattice/rim probes retain stable IDs; `ProjectPupil` accepts a reference
descriptor so the wave consumer can build its centroid without another
trace. P1 exposes shared trace/normal hooks; consumers do not fork
`trace.metal`.

**Multiple systems.** Reuse `upload_systems` and ray `system_index`, extending
them to resident generated rays and parameter batches. A host-validated
parameter plan maps stable surface IDs/ParamPaths to numeric slots, with
bounds, configurations, pickup/solve dependencies and material choices.
Across B candidates, GPU kernels apply parameters, recompute indices,
frames, first-order/solve data and aperture/aim seeds; never reuse nominal
values invalidated by a perturbation. Discrete topology/unsupported solves
use the reference preparation path with a recorded reason. Batch by
compatible topology, path program, clipping and last surface, preserving
candidate/configuration/field/λ/pupil order. Source snapshots remain
immutable; transient candidate IDs refer to the base hash plus parameter
vectors, and published candidate systems receive their normal content hash.

*Measured (P4, quiet Apple M2 Pro after zero-copy, warm end-to-end medians, best of two
runs, 16 samples plus rigid, 20-wavelength thermal and material-choice fixtures).*
Resident preparation including final readback costs about 0.4–1.7 ms up to
B=64, almost all command submission and per-candidate latency; parallel CPU
f64 reference preparation costs 0.04–0.07 ms at B=1, 0.22–0.35 ms at B=16
and 0.52–0.92 ms at B=64, so it was faster on every workload up to B=64
(Metal 0.53–0.78×). Metal was faster on 17 of 19 at B=128 (0.7–1.8 ms
against 0.9–1.6 ms; the rigid and 20-wavelength thermal fixtures trail
at 0.98× and 0.90×), on all at B=256 (0.5–2.0 ms against 1.7–3.1 ms) and
on all at B=1024 (1.2–3.5 ms against 6.1–12 ms). Batches of fewer than ≈100
candidates (≈150 for the rigid and thermal fixtures) are therefore
CPU-preferred; the earlier 128-candidate break-even moves down to ≈100. This is a cost
exception only; an explicit Metal request still runs on Metal.

**Transforms.** Implement portable complex f32 1-D/2-D FFT kernels (radices
2/3/5/7 and Bluestein for other lengths), tiled transpose, spectral products,
chirp-z/separable MFT and symmetric affine MFT. Specify signs, normalisation,
padding and bin order exactly as `wave/fft.rs`. Form large quadratic chirp
and defocus phases in double-float, reduce modulo a turn before f32 sin/cos.
Direct autocorrelation keeps the angular-spectrum sensor mapping, coverage
and interpolation of `wave/mtf.rs`; it is not interchangeable with an
interpolated FFT curve. Cache stencils across focus planes. Huygens uses
the same phase/sum primitives with distance tiles. No approximate transform
replaces the existing finite sum or discards sampling warnings.

**Transforms, measured (P2).** Quiet Apple M2 Pro after zero-copy, release,
precompiled library, warm medians of 15 runs, best of two; CPU is the f64
`wave/fft.rs` reference (rustfft + rayon), Metal end-to-end includes
submission and a copying final readback.

| Case | CPU f64 | Metal end-to-end | Metal device |
|---|---:|---:|---:|
| FFT 64² / 128² | 0.21 / 0.26 ms | 0.35 / 0.32 ms | 0.06 / 0.04 ms |
| FFT 256² / 512² | 0.40 / 0.78 ms | 0.36 / 0.82 ms | 0.08 / 0.15 ms |
| FFT 1024² / 2048² | 2.8 / 10.8 ms | 1.4 / 4.2 ms | 0.5 / 2.7 ms |
| MFT 256→512 / 1024→2048 | 1.5 / 23 ms | 1.4 / 9.3 ms | 0.4 / 7.7 ms |
| Skew MFT 256→512 / 1024→2048 | 5.5 / 86 ms | 2.1 / 28 ms | 1.0 / 25 ms |
| Optical correlation 256² / 1024², 64 frequencies | 11 / 174 ms | 3.2 / 41 ms | 2.3 / 35 ms |

**CPU-preferred:** standalone host-fed FFTs up to 128² (0.6–0.8×), where the
fixed ≈0.3 ms Metal command latency plus readback exceeds the whole CPU
transform; 256²–512² (1.1×, 0.95×) and the 256→512 MFT (1.1×) are at parity
end-to-end; 1024² and larger are Metal-preferred (2.0× and up). Inside resident chains only
the device time applies. Accuracy against f64: FFT relative L2 ≤1.6e-7 on all
144 default sample pupils, ≤1.5e-5 for dense 1024→2048 MFTs; optical MTF
≤1.9e-6.

**Collectives and RNG.** Extend §3.4's fixed tree to segments keyed by
candidate/field/λ/path/pixel. Chunk boundaries partition work, not the
mathematical reduction order. Use hi+lo totals, two-pass chief-relative
moments, stable radix sort/scan for EE and detector bins; never float
atomics. Zernike/large least-squares work uses deterministic QR/TSQR with
rank/condition reporting, not normal equations that square conditioning.
Reuse Philox4x32-10 and the core stream namespaces. Logical item IDs, not
thread/queue positions, address draws; preserve the reference draw order
for tolerance trials and a stray source ray's depth-first split tree.
Moving to a different RNG addressing scheme would require an explicit
version and reference implementation, not an unnoticed backend difference.

**Collectives, measured (P3).** Segment terms keep §3.4's tree; segments
of at most 4096 records run one SIMD group each (eight emulated tree threads
per lane, no partials) when there are at least 512 segments or all are at
most 64 long, otherwise 256-thread groups plus a SIMD-group pass 2. Both
paths, all scans, the radix sort (8-bit digits, ID digits first) and the
pivoted QR are bit-identical to the matching CPU-f32 specification, so
chunking, command boundaries, workers and concurrency change no bit. QR is
equilibrated pivoted MGS with two projections per step plus one classical
reorthogonalisation pass of each pivot column; Q stays orthogonal to
rounding on ill-conditioned Jacobians (columns are not capped). Quiet Apple M2 Pro
after zero-copy, release, precompiled library, warm medians of 10, best of two;
inputs resident, outputs read through `read_view` (no copies); CPU is the
serial f64 host reference.

| Case | 1e3 | 1e4 | 1e5 | 1e6 |
|---|---:|---:|---:|---:|
| Spot moments, 9 segments: CPU / Metal ms | 0.009 / 0.31 | 0.092 / 0.28 | 0.95 / 0.29 | 9.4 / 1.4 |
| Summary, 17-record segments | 0.001 / 0.25 | 0.011 / 0.27 | 0.12 / 0.46 | 1.2 / 2.4 |
| Stable sort, f32 keys + IDs | 0.015 / 0.51 | 0.23 / 0.47 | 2.8 / 0.87 | 33 / 3.6 |
| Inclusive f32 scan to double-float | 0.001 / 0.28 | 0.009 / 0.23 | 0.088 / 0.33 | 0.91 / 0.66 |
| Compaction of [n,2] double-float | 0.001 / 0.28 | 0.010 / 0.26 | 0.10 / 0.36 | 1.1 / 1.1 |

QR (rows×cols×batch), CPU f64 Householder / Metal ms: 200×20×1 0.05 / 0.93;
200×20×256 13 / 7.8; 2000×60×1 4.2 / 5.1; 4225×37×1 3.7 / 3.3;
20000×120×1 175 / 48 (46 device); 65536×37×1 66 / 20. Coefficient error
≤6.2e-15. Small Metal timings vary up to 1.35× between runs.

**CPU-preferred (standalone, host-fed):** spot moments below ≈3e4 records,
sorts below ≈2e4, scans below ≈5e5, compactions up to 1e6 (parity there),
summaries of very short segments at every measured size (0.49× at 1e6;
extrema terms dominate), and single QR fits up to about 2000×60 (parity at
4225×37). A plan costs ≈0.25 ms to submit and complete, which dominates
small work; inside a resident chain only device time applies. Batched and tall
fits are Metal-preferred. These are cost exceptions only.

**Aiming and chains, measured (P0/P1).** Quiet Apple M2 Pro after zero-copy,
warm end-to-end medians, best of two; CPU is the f64 rayon reference (aim +
trace + OPD for P1, trace + compensated sums for P0), Metal includes final
readback. **CPU-preferred (standalone, host-fed):** chief-only and
marginal-only aiming (≤36 rays, 0.10–0.21×), 64² pupil grids (3,096 rays:
0.63–0.67× with real aiming, 0.41–0.52× with aim off), and the trace→reduce
chain below ≈3e4 rays on systems of three or more elements (0.25–0.83× at
1e4; ≈1.6e5 for a singlet). Interpolated crossovers are ≈6e3 rays with real
aiming (256² = 51,040 rays 2.3–2.9×, 1024² 3.9–4.7×) and ≈2e4 with aim off
(256² 1.2–1.3×, 1024² 2.2–2.5×); 1e6 trace→reduce is 1.4–2.4×. These are
cost exceptions only.

**Radiometry and non-sequential reuse.** One device coating service provides
both powers and complex flux-normalised amplitudes, keyed by wavelength,
incidence side/angle and material/stack. Batch TMM/table construction on GPU;
qualify interpolation including grazing/critical angles against f64 and
refine or use direct evaluation where needed. Preserve each solver's model:
scalar primary/stray products of averages, ghost separate s/p products,
and polarised PRT are different. Ghost transport adds virtual-segment,
sensor and DOE-visit rules to shared geometry. Stray uses portable analytic
intersections and a compute BVH on every Metal device; hardware RT is an
optional qualified spatial-search implementation, not a requirement or a
replacement of curved optics by triangles. Stable nearest-hit tie rules,
shared apertures and exact coverage signatures preserve the solver partition.

**Stray transport, measured (P10).** Busy shared M2 Pro, native analysis
jobs, median of three warm runs. **CPU-preferred:** every measured workload
except the Q-bfs sample at 20k rays (Metal 66 ms, CPU 79 ms). Metal/CPU is
1.6–5.1× on the samples at 20k and 1M rays, 3.3–3.5× on split-heavy plates
and 3.0–14.5× on the 36-primitive barrel (400k rays: 1.33/0.24 s on axis,
0.70/0.05 s at 20°). Scheduling, plan sizing and estimator collection were
optimised (3–8× faster than first delivery); the remaining limit is the
double-float interaction megakernel at about 3–5 M events/s versus
12–20 M events/s on the CPU, while the same BVH casts first hits at about
37 M rays/s. A split closest-hit/interaction wavefront is the next lever.

**Optimisation batches, measured (P8).** Busy shared M2 Pro, provisional
warm medians of ten runs; resident timings include candidate preparation and
operand assembly, not new-candidate qualification. Cooke default spot merit,
CPU f64 / resident Metal: B=1 0.17 / 3.5 ms, B=16 0.66 / 4.5 ms, B=256
6.9 / 15.9 ms, B=1024 27 / 51 ms (14 commands, 60 MiB peak). Every
changing-candidate workload adds a full f64 reference and a one-command
geometry qualification per new candidate: 32 compensated Cooke trials 7.8 /
127 ms, three catalogue refinements 1.7 / 24 ms. **CPU-preferred:** every
measured optical merit/tolerance/glass workload; residual targets, FD stencil
differences, `Jᵀr` and trial summaries over host-owned f64 values at every
size (the fill alone matches the host arithmetic); and LM tall QR below
`m·n²` = 2²⁴ (400×100: 29 ms host against 72 ms; 4000×100: 320 against
154 ms resident, which the solver then uses). Per-command latency dominates
B≤16 (9 commands); batched geometry qualification and a qualified parameter
envelope, not kernel speed, are the remaining levers.

### 9.3 `auto`, determinism and jobs

Change the engine's initial default from `CpuReference` to `Auto` when
integration qualifies the paths; explicit CPU remains available. `auto`
selects Metal for every supported, qualified operation chain. Remove the
fixed 65,536-ray and old candidate-count crossovers after measuring the
resident chains. CPU cost exceptions require reproducible end-to-end
measurements for the operation/shape/device/build/residency state, with a
clear win beyond measurement noise. Measure cold and warm separately;
include qualification, setup, upload, dispatch, reductions and final result
packing. Cache that decision with its evidence/version; do not time-race
backends on every request. Unknown cost defaults to qualified Metal.
Small serial control/I/O is inherently host work (§9.1), not an excuse to
keep a large solver CPU-only. The goal of residency/fusion is to shrink
the measured CPU-exception set; no CPU optimisation package is planned.

An explicit Metal request bypasses cost exceptions, never accuracy gates.
Preserve existing unavailable-backend/`allow_backend_fallback` behaviour.
Unsupported work, failed qualification, resource or device failure produces
a reported reference fallback; cancellation never triggers fallback. A
partial solver fallback names stages/candidates, counts polished rays and
records why. `RunInfo` and the backend report retain requested/used backend,
precision, seed, base/derived hashes, device, build/probe, qualification,
effective algorithm/plan versions, cost decision and stage timings. Do not
label a GPU computation f64 just because its public buffer widens to f64.

For fixed inputs, seed, device, kernel build and plan, outputs are bitwise
repeatable; scheduling, worker count, chunk size and unrelated jobs cannot
change them. Cross-device/build agreement for transforms/transcendentals is
within the numerical gates, not a promise of bit identity. Version and record
plan choices that affect operation order. Stable compaction preserves IDs;
adaptive wave/rim/refinement and candidate decisions use fixed tie rules.
Time-limited or manually stopped searches are reproducible only for the
recorded completed prefix; elapsed-time stopping is not deterministic.

Every long API takes `JobContext`. Use bounded command buffers/tiles with
cancellation checks before encoding and after completion. Target ≤50 ms
between checks on qualification hardware, measure worst cases including
Huygens and split trees, and reduce tile size if needed. In-flight Metal
commands finish before leases recycle; cancelled jobs discard outputs.
Monotone progress belongs to the outer job: allocate stage subranges,
aggregate completed logical tiles, reserve a tail for adaptive work, and
emit 1 only after successful final assembly. Optimiser **Stop** still returns
its best completed state; **Cancel** discards the result. Neither changes
the existing solver semantics to a partial success by accident.

### 9.4 Accuracy and qualification

Qualification is per operation chain, system/domain and settings, not just
the presence of a trace kernel. Cache keys additionally include radiometry,
field/λ domain, shape/size family, precision/plan version and tolerances.
Keep the §5.2 one-third safety margin for sampled optical error maxima;
full-grid regression must meet the whole budget. Cancellation leaves no
qualification verdict. A baseline-system pass does not qualify arbitrary
perturbations: candidate plans need a tested parameter envelope and guard
checks, sampled f64 sentinels within batches, and reference qualification
when leaving that envelope. Without an established envelope, qualify the
candidate itself. Recheck accepted/final designs and f64-score reported
optimisation/glass finalists. Record unqualified candidate fallbacks.

| Quantity / kernel | Acceptance against independent CPU f64 |
|---|---|
| G: geometry and statuses | §7: positions ≤2e-5 mm on each surface, directions ≤2e-6; identical survival except the 1e-5 mm aperture band. Count band cases and all failure classes; never compare only common survivors. Existing TIR-band diagnostics remain explicit, not a blanket status exemption. |
| OPL: aiming/OPD/phase | Honour the requested aiming residual; OPD ≤5e-3 waves. Internal OPL/sphere target ≈1e-7 mm on 100+ mm paths (≈1.8e-4 waves at 0.55 µm); this is tighter than the external λ/200 budget (2.75e-6 mm there). Double-float must cover geometry and reference construction as well as the path sum. |
| R: centroids/RMS/MTF | Position budget for centroids; differences of two positions ≤4e-5 mm. RMS ≤max(0.005×reference,2e-5 mm); MTF ≤2e-3 absolute, including direct/FFT/through-focus and detector multiplication. |
| T: FFT/chirp/affine transforms | Proposed gates: relative L2 complex error ≤1e-4 and max absolute error ≤1e-4×reference L1 input norm (with declared transform scaling); inverse round-trip/Parseval ≤1e-4 relative. Small direct-DFT, skew-MFT and non-power-of-two known answers supplement large f64 comparisons. Final optical MTF still meets §7. |
| E: PSF/EE/LSF/image | Proposed gates on matched sampling: integrated energy/flux ≤1e-3 relative, normalised PSF/image L1 error ≤2e-3, EE/ensquared fractions and Strehl ≤2e-3 absolute, LSF integral ≤1e-3. Centroids use G; r50/r80/r90 compare the energy curve and ≤one radial sample interval, tightening with sampling. Include Airy, annulus, tilt, defocus, Parseval and convolution conservation. |
| E: coating, polarisation and ghosts | Retain GHOST §6 weight limit 1e-5 relative; proposed zero-safe comparison `abs(error) ≤1e-5×abs(reference)+1e-10×incident_weight`. Per-interface R+T+A residual ≤1e-6, integrated ledger ≤1e-5 of emitted power. Jones/PRT relative norm ≤1e-4 (absolute 1e-7 of incident field near zero), D/intensity ≤1e-4 absolute, retardance ≤1e-3 rad where conditioned; otherwise reported f64 fallback, preserving undefined-retarder rules. Ghost map L1/flux ≤1e-3 of incident-normalised reference flux; count pixel-edge migrations and also test fixed-hit binning separately. |
| E: stray Monte Carlo | Identical raw counters and controlled-path tests first; same-sample f64 transport uses the same uniform mapping. Ledger ≤1e-5 of emitted power. Ensemble class flux/maps/moments agree within 5 combined standard errors plus deterministic quadrature error measured by refinement; test rare signatures and split-child correlations, not only total flux. Sampling error cannot excuse a biased ledger or wrong path ownership. |
| D: RI/pupil maps/curvature | Proposed gates: RI ≤1e-3 absolute, CRA ≤2e-4 degrees, pupil magnification ≤1e-3 relative or 1e-5 absolute; curvature focus ≤1e-3×(1+abs(reference_mm)) mm, cross-checked by close rays. Double-float pencil/Jacobian arithmetic; singular/undefined values match the reference. |
| D: material/first-order batches | Proposed index error ≤1e-10 absolute (1e-7 mm over a 1000 mm path); first-order lengths ≤1e-6 relative or 2e-5 mm absolute. Check thermal/dispersion offsets and solved constraints against f64, and resulting rays/OPD against §7; ill-conditioned afocal/solve cases preserve undefined values or fall back. |
| D: fits, FD and optimisation | Proposed reconstructed Zernike OPD/RMS ≤5e-3 waves; coefficients and rank checked against f64 QR with condition-aware residual tests. Scaled FD column error ≤1e-3 vs f64 Richardson (column norm floor 1e-8 in scaled units), plus analytic thick-lens derivative and step sweeps. Keep evaluator-specific FD steps until remeasured. f64 final merit within 1% or 1e-8 absolute of device merit and same feasibility classification outside the recorded constraint uncertainty band; ambiguity gets a reference decision. |

New E/T/D numbers are release gates to demonstrate, not measured claims.
Compare identical settings/sampling/methods: the CPU's direct-versus-FFT MTF
discrepancy is not GPU error. Keep wave sampling, truncation, conditioning
and root-search warnings; increasing precision does not fix undersampling.
For insufficient margin use double-float, a better-conditioned formulation
or the reported reference fallback, never loosen the budget to pass.

**Emulation policy.** Require matching-environment bit identity for integer
RNG, layouts/status/IDs, existing trace/df/fmath, new aiming/system-update
algebra, fixed collectives/sorts and specified coating/transport algebra.
Preserve the existing probe-qualified division/sqrt exception. FFT,
chirp/MFT, autocorrelation, QR and library transcendental-heavy BSDF/complex
operations need tolerance-based CPU-f32 and independent f64 checks; they
need not reproduce a different CPU FFT's bits. Their index/mask/branch
contracts remain exact and repeated Metal runs must match. Test post-
transcendental branches against the reference, particularly critical angles,
Newton convergence and roulette thresholds. Hardware RT is qualified for
hit accuracy and tie rules against the compute caster, not vendor traversal
bits. A CPU-f32 match alone is never accuracy qualification.

### 9.5 Delivery and measurement

The task briefs and ownership ledger live in the task's `scratchpad/gpu/`
directory (`PLAN.md`). P0 is a serial interface gate; the **first parallel
wave is P1–P4**. Shared registration/layout files have one owner; subsequent
packages use separately registered module/header/shader files. Final engine
integration is deliberately last, so incomplete kernels never advertise
production capability.

| Package | Goal | Dependencies | Difficulty |
|---|---|---|---|
| P0 | Resident backend, buffer lifetimes, frozen interfaces and module seams | — | difficult |
| P1 | Device generation, aiming, reusable trace state and reference-sphere OPD | P0 | hardest |
| P2 | FFT, chirp/affine MFT and autocorrelation primitives | P0 | hardest |
| P3 | Deterministic segmented reductions, sort/scan and QR | P0 | difficult |
| P4 | Resident multi-system parameter/material/first-order preparation | P0 | hardest |
| P5 | Geometric analyses, field maps, sizing and layout | P1, P3, P4 | difficult |
| P7 | Shared coating/radiometry and polarisation | P1, P3, P4 | hardest |
| P6 | Complete wave/image-simulation pipelines | P1, P2, P3, P5, P7 | hardest |
| P9 | Ghost path probes, refinement and irradiance | P1, P3, P7 | difficult |
| P10 | Stray transport, BVH, deterministic tallies | P1, P3, P7 | hardest |
| P8 | Full merit/FD, optimisation, tolerance and glass batches | P4, P5, P6, P7, P9, P10 | hardest |
| P11 | Engine routes, qualification, auto policy and release evidence | P5, P6, P7, P8, P9, P10 | difficult |

Every package supplies f64/appropriate emulation tests, actual-Metal tests,
determinism/cancellation/progress checks and before/after measurements on
`mklight_core::samples::all()` (16 current samples). Emulation-only or skipped
device tests cannot qualify a release. Run `source ~/.cargo/env` before
Cargo, `cargo test --workspace`, and preserve the reference tests. Benchmark
the unchanged CPU reference, old Metal path where present and new Metal
path on identical inputs; retain raw settings, seeds, hardware/build,
fallback counts, peak memory, cold/warm wall and device timings, and
median/p95 after warm-up. Use the existing Metal trace, wave interactive,
ghost/stray bench, optimiser gpu_bench and engine glass benchmarks as
anchors. Include tiny/default/dense loads and end-to-end jobs; kernel-only
speedups cannot justify `auto`. No production code is changed by this plan.
