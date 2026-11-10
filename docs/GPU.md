# MKLight GPU Infrastructure

This document describes how solvers reach a GPU: the backend interface, the
packed data layout every backend shares, the f32 numerics strategy that keeps
GPU results inside the CONVENTIONS §7 error budget, and how results are
validated against the CPU f64 reference. Read `docs/CONVENTIONS.md` first.

| crate | role |
|---|---|
| `mklight-gpu` | `ComputeBackend` (production) and `QualificationBackend` (offline qualification) traits and job types, the packed layout (`include/mklight_gpu_types.h` + Rust mirror + packer), the double-float / utility-math / reduction / **trace-kernel specifications** in Rust, the CPU-f32 emulation backend, cross-validation against the f64 reference and precision qualification (`xval`) |
| `mklight-metal` | Metal implementation: MSL kernels (`shaders/`), offline `.metallib` build with runtime-source fallback, device/queue/pipeline/buffer wrappers |

Vulkan (AMD/Intel) and CUDA/OptiX (NVIDIA) backends will be further crates
implementing the same trait over the same layout.

## 1. The backend interface

`mklight_gpu::ComputeBackend` is the production interface: the only GPU
interface the solvers and the engine see. It is object-safe
(`Arc<dyn ComputeBackend>`), and every method but `prepare_kernels` is
required:

```rust
trait ComputeBackend: Send + Sync {
    fn info(&self) -> &DeviceInfo;                 // name, memory, unified memory, RT support, SIMD width
    fn precision(&self) -> &PrecisionDescriptor;   // numerics contract + measured probe + error budget
    fn supports(&self, kind: JobKind) -> bool;     // the engine opens a device only with JobKind::Trace

    // Systems.
    fn upload_systems(&self, systems: &[PackedSystem]) -> MkResult<SystemHandle>;
    fn upload_systems_cached(&self, key: &SystemCacheKey, systems: &[PackedSystem])
        -> MkResult<SystemHandle>;
    fn release_systems(&self, handle: SystemHandle) -> MkResult<()>;

    // Resident buffers and plans (§9.2).
    fn fill_buffer(&self, desc: BufferDesc,
        fill: &mut dyn FnMut(BufferFill<'_>) -> MkResult<()>) -> MkResult<BufferHandle>;
    fn read_view(&self, view: BufferView) -> MkResult<ReadView>;
    fn submit_plan(&self, plan: ResidentPlan, ctx: &JobContext) -> MkResult<Pending<PlanResult>>;
    fn resident_cache_stats(&self) -> ResidentCacheStats;

    // Kernel preparation (§6). Provided: nothing to prepare.
    fn prepare_kernels(&self, cache_dir: Option<&Path>,
        progress: &(dyn Fn(usize, usize) + Sync), cancel: &AtomicBool)
        -> MkResult<KernelPreparation>;
}
```

`mklight_gpu::QualificationBackend` extends it with the host-array job API
and the single-buffer primitives. `CpuF32Backend` and `MetalBackend`
implement both traits:

```rust
trait QualificationBackend: ComputeBackend {
    // Host-array jobs (this section, §4).
    fn submit(&self, job: Job, ctx: &JobContext) -> MkResult<Pending<JobResult>>;

    // Single buffers.
    fn upload_buffer(&self, desc: BufferDesc, data: BufferData) -> MkResult<BufferHandle>;
    fn import_buffer(&self, desc: BufferDesc, data: HostAllocation) -> MkResult<BufferHandle>;
    fn readback(&self, view: BufferView, ctx: &JobContext) -> MkResult<Readback>;

    // Provided conveniences.
    fn upload_system(&self, system: &PackedSystem) -> MkResult<SystemHandle>;
    fn run(&self, job: Job, ctx: &JobContext) -> MkResult<JobResult>;   // submit + wait
}
```

The qualification interface is an offline test utility. `xval` (§5.1, §5.2)
and the test suites compare a backend with the CPU f64 reference and with
the CPU-f32 emulation through it. No solver or engine code calls it, and
nothing falls back to it at run time: a production request runs as resident
plans or on the CPU reference (§9.2).

* **Systems** are uploaded once per snapshot (and wavelength chunk) and are
  immutable on the device, like the `Snapshot`s they come from. One upload can
  hold many systems (optimiser candidates, tolerance samples, zoom
  configurations); rays select one by index. A `SystemHandle` is only valid on
  the backend instance that issued it, and its fields are private: backends
  use it only as a lookup key and check every index against the stored set.
  Uploading runs `PackedSystem::validate` on every record (enum codes, flags,
  `num_asphere`, finiteness, `n > 0`, `α ≥ 0`), so kernels can trust it.
* **Plans** (`ResidentPlan`, §9.2) are ordered operations over buffers that
  stay on the device: `fill_buffer` writes an input in place, `submit_plan`
  runs the operations, and `read_view` borrows an output in place.
* **Jobs** (`Job`, qualification interface) are batched, backend-neutral
  descriptions of work over host structure-of-arrays data: `PhiloxStream`,
  `PhiloxBlocks`, `DfOps`, `Reduce`, `SagProbe`, and `Trace` (the
  sequential / path ray-trace kernel, §4). Each has a typed `JobOutput`.
  `Job::validate` performs the backend-independent input checks.
* **Asynchrony.** `submit_plan` and `submit` return a `Pending` immediately.
  It can be waited on (`wait`) or polled (`is_ready`). A
  worker that dies without a result resolves it with an error, never a hang,
  and a worker panic becomes `MkError::Backend` naming the job.
* **Cancellation and progress.** Every submission has its own cancel token.
  The worker stops when either that token or the caller's `JobContext` token
  is cancelled. `Pending::cancel`, or dropping an unfinished `Pending` (a
  superseded request), cancels only that submission. Sibling jobs under the
  same context keep running, and cancelling the context stops all of them.
  Backends split every plan and job into command buffers (launches; of
  `chunk_items` work items for a job, values for reductions), check
  cancellation between them, and report progress through the `JobContext`.
  Progress reaches 1 on success, also for empty jobs. A cancelled submission
  returns `Err(Cancelled)`, never a partial result.
* **Stats.** `PlanStats` (§9.2) and `JobStats` report device time (GPU
  timestamps), wall time and the number of command buffers.
* **Reproducibility.** `DeviceInfo::backend_id()` ("metal:Apple M2 Pro") and
  the `PrecisionDescriptor` (serialisable; it includes the measured numerics
  probe and `kernel_build`, i.e. library origin, MSL version and compiler
  flags) go into `RunInfo` together with the seed. Philox streams and reductions are deterministic (see §3), so a
  `(snapshot, settings, backend, seed)` tuple reproduces bit-identical output.
* **Never silently degraded.** A backend that cannot run a job kind reports
  `supports(kind) == false` and returns `MkError::Unsupported` for it. The
  engine opens a device only when it supports `JobKind::Trace` and otherwise
  runs the CPU reference with that reason. The Metal backend also refuses
  double-float work if its start-up numerics probe fails (§3.3).

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
(`PackOptions::wavelengths`); the packer refuses rather than truncates.

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

### 3.6 One arithmetic

The kernels have a single build: f32, with double-float wherever the budget
needs more (§3, §3.1) and compensated reductions (§3.4). The CPU-f32
emulation specifies exactly that arithmetic, and
`PrecisionDescriptor::{opl_double_float, compensated_reductions}` are true
on both backends.

A pure-f32 build of the same kernels (every double-float operation plain
f32, plain sums) was measured and rejected: it exceeds the OPD budget, fails
real aims at the default tolerance and does not hold the position budget on
every design, and an f32 predictor phase ahead of the double-float aim solve
does not pay. The measurements are recorded in
[history/gpu-work-packages.md](history/gpu-work-packages.md).

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
  `Q = P + t₀D`, the refinement of every hit on the exact line (one Newton
  step on `c(x² + y² + κz²) − 2z`; two on `z − sag(r²)` for an even
  asphere), the unnormalised normal `G = (−cx, −cy, 1 − κcz)` or
  `(−x·s, −y·s, w)`, and the interactions: refraction `D' = μD + bG` with
  `μ = n₁/n₂` from double-float indices and reflection `D' = D + bG`, where
  `b` comes from the f32 law and is then corrected along `G` until
  `|D'| = 1` (twice for refraction). The tangential component — Snell's law
  — is exact by construction. The even-asphere refinement evaluates the
  whole sag *and its slope* in double-float (`P` and `dP/d(r²)` by Horner
  from the hi + lo coefficients, `mkl_asphere_poly_dp_df` /
  `trace_kernel::asphere_poly_dp_df`); the normal takes the double-float
  slope `s = 2w·dP/d(r²) + c` of the last step, so the f32 Newton search
  only seeds the hit. This holds for every even asphere, with no flag and
  no selection key. An f32 slope at the unrefined point put image hits on
  strong aspheres up to ~7e-6 mm off and made device aiming fail on them;
  with the double-float slope the kernel's CPU-f32 specification matches
  the f64 trace to ~1e-12 mm (`crates/mklight-gpu/tests/trace_slope.rs`).
  The stray caster and the ghost paths use the same slope. The
  **ideal lens** runs entirely in double-float: the
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
The status differences that remain are rays whose reference classification
is itself ill-conditioned: the search starts at the closest approach of a
line that does not cross the surface nearby, and f32-sized changes of the
input flip the f64 status too (analysis in
[history/gpu-work-packages.md](history/gpu-work-packages.md)). No f32 search
can reproduce these decisions, so the sample passes above do not claim
exhaustive qualification of every ray of a steep system.

### 5.1.2 Manufacturing overlays (layout v6)

`extended_shapes` adds rotated astigmatism plus spherical overlays on plane,
even-asphere, QCon, XY, nested and DOE bases, and the pupil-grid set a
tilted, decentred Cooke triplet with overlays on every lens surface. Metal is
bit-identical to the FTZ emulation; against f64 the common hits stay within
the budget above (Apple M2 Pro: position 1.9e-6 mm, direction 4.2e-8).

### 5.2 Precision qualification (test utility)

`xval` is an offline test utility: it takes a `&dyn QualificationBackend`
(§1) and traces through `Job::Trace`, which no production code calls.
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
`MkError::Cancelled`: a cancelled qualification is no verdict.
`Qualification::record(&mut RunInfo)` writes

* on success: `backend = "metal:Apple M2 Pro"`, `precision = F32Gpu`;
* on failure: `backend = "cpu-f64"`, `precision = F64Reference`, and a
  warning "GPU backend metal:… not used (reason); computed with the CPU f64
  reference";

and in both cases the whole qualification (measured errors included) under
`settings["backend_qualification"]`. A qualification holds for one system
(snapshot) on one backend. Only tests call `qualify` and `record`
(`mklight-gpu` and `mklight-metal` `tests/trace.rs`, the engine's
`tests/p11_routes.rs`): the engine and the candidate solvers run no
qualification and no qualification fallback at run time (§9.2, "Runtime
publication policy").

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
| CPU f64 scalar `trace_ray`, rayon | 71 ms | 1.7e8 |

The kernel runs ~5× faster than the **scalar** reference on 12 CPU cores;
end to end, with host-side packing and unpacking of ~110 MB of planes,
~2.5×. (The double-float aperture tests of layout v4 cost ~5 % of device
time.) The CPU row is the scalar oracle. The engine's CPU batch paths use
the f64 packet tracer instead (CONVENTIONS §7, "CPU packets"), measured
single-threaded at 2.0–2.9× the scalar trace on conic and plane systems
(Apple M2 Pro, indicative); this benchmark has not been re-run against it,
so the CPU–GPU ratio for packet traces is smaller than shown. Device-resident
chains (trace → reduce without readback) are the resident plans of §9.2.

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

* **Build.** `build.rs` compiles `shaders/*.metal` to `.air` (all files
  concurrently) and links one `mklight.metallib`, embedded
  with `include_bytes!`. It finds the compiler
  via `MKLIGHT_METAL_TOOLCHAIN_BIN`, then `xcrun -sdk macosx metal`, then the
  system-mounted Metal toolchain component (which `xcrun` does not always
  resolve). Shaders are compiled offline: an offline compile error fails the
  build, and so does a release build without the toolchain (install it with
  `xcodebuild -downloadComponent MetalToolchain`); only a debug build on a
  machine without it falls back. The include-inlined source is always embedded
  for that fallback; `MKLIGHT_METAL_SKIP_OFFLINE=1` forces it for testing.
  `LibraryChoice::{Auto, Precompiled, RuntimeSource}` selects at run time.
  The offline compile uses one strict flag set (`-Wall -Werror`, with
  `-fno-fast-math -ffp-contract=off`). With `Auto`, a precompiled
  library that fails the numerics probe is replaced by the runtime-compiled
  source when that one passes. The flags or options actually used are in
  `PrecisionDescriptor::kernel_build`.
* **Device and pipelines.** The system default device. Pipelines are cached
  by kernel name + function-constant values
  (`mkl_fc_df_op` at index 0, `mkl_fc_reduce_mode` at index 1); the cache
  holds at most 512 of them (`PIPELINE_CACHE`).
* **Kernel preparation.** `prepare_kernels` creates every pipeline the
  backend uses (all kernels and the common specialisations of the
  specialised ones) ahead of their first use, on up to six threads, and
  reports `(done, total)` after each; setting `cancel` stops it early
  without an error. With a `cache_dir` the pipelines go through a Metal
  binary archive in that directory
  (`pipelines-<gpu>-<hash>.metalarchive`, the hash covering the system
  version and the shader library, so a new build, a system update or another
  GPU never reads an archive made for something else): pipelines the archive
  holds are created without compiling (`KernelPreparation::from_cache`), the
  others are compiled and added, and the archive is saved before the call
  returns, also when a pipeline failed. Archives of other builds in the
  directory are removed once unused for 30 days. A cache that cannot be
  attached or saved is reported as `KernelPreparation::warning`, not as an
  error: the kernels are ready, and a later process compiles them again. The
  engine runs the preparation on a background thread (`Engine::prepare_gpu`,
  state in `Engine::gpu_preparation`; `mk_engine_prepare_gpu` in the C ABI,
  FFI.md), which the app starts at launch (APP.md).
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
* **Execution** (`Job` submissions; resident plans commit without waiting,
  §9.2 "Asynchronous execution and admission"). Each submission runs on a
  worker thread: per chunk it checks
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
* `Job::Trace` traces rays the host generated (field/pupil mapping and
  real ray aiming in f64). Resident plans generate paraxial launches on the
  device; real-aimed pupil rays are aimed on the host by default and on the
  device only where a division asks for it (§9.2, "Aiming and OPD").
  Polarised (Jones/Stokes) radiometry is not ported to `Job::Trace`.
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
* `Job` inputs and outputs are host arrays, and the command buffers of a
  `Job` run one after another (commit, wait); with the default 2¹⁹ items per
  buffer the gap is negligible next to kernel time. Device-resident chains
  without intermediate readbacks, and asynchronous command buffers with an
  in-flight window and admission budget, are the resident plans of §9.2.
* The CPU/GPU divisions of the analyses use fixed shares and rules measured
  on one machine (an M2 Pro: the split-aiming device share 0.35, the stray
  share rule, the ghost CPU share of 20 %). GPU-heavier divisions remain
  selectable through the route options of §9.3, but nothing measures or
  chooses them per machine yet.

## 9. Resident GPU architecture and routing

**GPU for throughput, CPU for latency** (DECISIONS.md, "Compute placement").
Metal runs large batched work where it is faster end to end; the CPU f64
path, traced in SIMD packets (CONVENTIONS §7), serves small, interactive and
solver work, and stays the accuracy reference and reported fallback. `auto`
chooses per request with the calibrated cost model of §9.2; qualification
per operation and domain decides what may run natively at all. Vulkan is
deferred. Solvers use the backend-neutral
`ComputeBackend`, typed buffers and explicitly specified kernels, with Metal
encoding confined to `mklight-metal`. §§1–8 describe the backend interface,
the trace kernel and their validation; this section describes the resident
plans built on them, the division of each analysis between host and device,
and routing.

### 9.1 Inventory and placement

The native adapters are the `gpu` modules of the solver crates named below;
the engine routes to them from `mklight-engine/src/analysis/` (`gpu.rs`
lists the analyses with a native path, `GPU_ANALYSES`).

Sizes below describe workloads to qualify/benchmark, not new input limits.
Let S = surfaces (qualification domain ≤30, track ≤1000 mm), F = fields,
W = wavelengths (packed in chunks of ≤16), P = pupil rays, B = systems,
N = pupil side, M = image side, V = variables, O = operands. Larger domains
need separate qualification; unqualified work retains reported fallback.
Precision tags map to §9.4:
G = geometry (CONVENTIONS §7 position/direction/survival), OPL =
double-float paths and OPD (CONVENTIONS §7), R = compensated statistics
(CONVENTIONS §7 RMS/MTF), E = energy/radiometry, T = transforms, D =
derivative/conditioning gates. E/T/D add qualification tests; they do not
relax CONVENTIONS §7.
The placement column is each computation's current division between host
and device: the defaults, measured on an M2 Pro.
The route options of §9.3 select GPU-heavier divisions where they exist.

| Computation / source | Work size | Placement and reason | Precision |
|---|---|---|---|
| Sequential and path traces, all supported sags, DOE phase/order, refraction/reflection, clipping (`trace`, `trace_kernel`) | FWP, typically 10²–10⁶ rays × S; optional S records/ray | GPU: independent rays with shared surface data | G + OPL; local frames, double-float geometry/phase |
| Field/pupil mapping, real-image-height field solve, chief/pupil Newton aiming (`trace/tracer`, `pupil`) | FW chiefs + FWP solves, up to configured iteration cap | Field source and chief/Jacobian rows: host f64 (reference tracer, serial per field). Pupil aiming: host f64 from per-row seed tables by default (geometric, transmission, wave and candidate adapters), with a fixed device share of dense spot-diagram and geometric-MTF generations; GPU double-float Newton solves seeded by those rows for `aiming=device`/`table`, the solvers' `solver_aiming=device`, `device-chief` maps, ghost chief aims and the device pupil tracer (§9.2) | G; double-float residual/Jacobian/update; honour the run's aim-stop policy (CONVENTIONS §7) |
| Automatic semi-diameters, surface profiles/meshes, layout rays (`trace/semidia`, `analysis/layout`) | FW × rim samples × S; 10²–10⁶ vertices | GPU for traces, sag samples and extrema; host for small topology/labels | G; double-float world-to-local differences, f32 display coordinates relative to origin |
| Spot, ray grid/fans, OPD fans, footprint/overlays, longitudinal aberration, lateral colour (`analysis/{spot,fans,illumination,seidel}`) | FWP or 10²–10⁴ fan points; footprint records P×S | Host aims (seed tables); GPU traces and reduces per field group of ≥ 2¹⁵ rays while the host aims the next group; host assembles result metadata/small hull topology | G + OPL + R; chief-relative two-pass moments |
| Geometric EE/ensquared energy, geometric MTF, best focus (`analysis/energy`) | P order statistics; P×frequencies; P×focus trials | Encircled and ensquared energy: host reference (f64 order statistics of the ray distances); geometric MTF and best focus: GPU complex sums and focus merits; scalar Brent decisions may stay host | R; fixed trees, stable radius/index tie order |
| RI curve/map, CRA, pupil aberration, grid distortion, field curvature (`analysis/{illumination,field_maps,curvature}`) | 5×5 default maps; 12×48 RI nodes/field, about 15 traces/node; dense 33²–129² field sweeps | RI: host prepares fields and solves the pupil nodes' image points (reference Newton), GPU traces each block's differential pencils while the host solves the next; chief-ray maps: host f64 chiefs, GPU rays (`fieldmaps=device-chief`: device chiefs too); field curvature: host chiefs and close-ray pencils, GPU Coddington traces | G + D + E; double-float differences, sag Hessians and Coddington matrices |
| First order, Seidel, chromatic focal shift, model/surface operands, solves/pickups (`trace/{paraxial,solves}`, `analysis/{system_data,seidel}`) | O(SW) for one design; BSW in searches | Host for a single small report/dependency walk; GPU across candidate/wavelength batches and dense sag extrema | f64 control/reference; double-float batched recurrences, G/D end-to-end |
| Pupil lattice, rim, OPD, resampling, phase/amplitude, wavefront statistics and Zernike fit (`wave/{traced,interp,pupil,zernike}`) | default 65² trace lattice; N=64–1024 automatic, 16 subpoints/cell; ≈37 fit columns | Analyses: host traces, resamples and fits the lattices with the reference code and makes every sampling decision; the wavefront map's statistics are the reference's too (f64 reductions of the pupil the host holds). Device pupil tracer (GPU lattice, rim probes, triangulated-grid evaluation) for the solvers' MTF operands | OPL + G + R + D; no normal equations for ill-conditioned fits |
| FFT PSF, chirp/affine MFT, OTF/MTF direct/FFT, field/through-focus MTF (`wave/{fft,psf,mtf}`) | N² pupils; M≤2048 automatic; F×W×focus/frequency batches | GPU: regular transforms, stencils and spectral sums on host-resampled pupils written in place, overlapping the host's preparation of later fields | T + OPL; phase formed/reduced in double-float, complex f32 arrays |
| Huygens PSF/MTF (`wave/psf`) | image points × covered pupil samples; existing cap 2×10¹⁰ terms, M≤1024 | GPU tiled direct sums; bounded tiles preserve cancellation | OPL + T + E; rationalised distance differences, compensated complex sums |
| Diffraction EE, line spread, Strehl, detector response (`wave/{energy,detector}`) | M² pixels, radii/frequencies and Bessel kernels | GPU shared radial-bin/stencil reductions; isolated scalar analytic reference curves may stay host | R + T + E; normalisation, Parseval and known answers |
| Image simulation (`wave/image_simulation`) | field-grid PSFs; height×width×C samples, C∈{1,3}; haloed convolution tiles | Host-traced pupils and RI rows; GPU PSF transforms, pattern generation, inverse distortion, resampling, RI multiplication, FFT convolution, pixel integration and noise; file decode/encode host | G + T + E; preserve linear light, low-y row zero, halo flux |
| Coating TMM/tabulated response, absorption, transmission, PRT/Jones/Stokes pupil maps (`core/materials/coating`, `trace/{radiometry,polarization}`) | FWP×S; angle/wavelength/stack tables; 4 complex Jones entries/ray | Transmission: host aims every ray, GPU traces and evaluates the coating transport (`radiometry`) per block of rows while the host aims the next; polarisation pupil map: host reference; host for a tiny isolated lookup and reference table checks | E + G + OPL for phase; complex f32, double-float where conditioning needs it |
| Ghost enumeration/paraxial ranking (`ghost/{path,paraxial}`) | two-bounce O(S²); filtered higher orders ≤8 reflections | Small branch-and-bound enumeration stays host; large path/wavelength ranking batches GPU | exact integer signatures/order; double-float first-order terms |
| Ghost irradiance/focus (`ghost/{sequence,irradiance}`) | path×FW×pupil², refinement ≤4 levels; measured 0.3–4 M distinct probes on Double Gauss | Four path groups, pipelined: GPU traces one group's probe tiles while the host completes, deduplicates and splits another and gathers finished leaves; GPU segmented maps; the CPU reference traces a fixed 20 % of the path groups beside the device run | G + E; separate s/p products, virtual-segment decisions and exact coverage |
| Stray sources, BVH queries/intersections, BSDF/TIS, branch transport, tallies/errors (`stray/*`) | 20 k engine-default rays, benchmark 400 k; tens–hundreds of primitives, split path trees | GPU wavefront transport of the first 64 events of the device's sources (a fixed share of the source chunks beside the CPU reference trace of the others; `stray_share=1`: every source), writing detections, ledgers and continuation stacks into shared memory; the CPU reads them in place, continues longer path trees and forms the estimators; small scene topology build host; GPU batched bounds/build when large | G + E + exact Philox; double-float close hits/ledger/moments |
| Merit and constraints, FD Jacobians, LM/trust-region, global populations (`optim/*`) | B×O residuals, B=V or 2V for FD; e.g. V=8–100, O=10²–10⁴ | Host candidate preparation (f64 tracers, chief rays, model/paraxial/first-order operands) and pupil-ray aiming by default; GPU traces each batch's pupil rays in one plan and forms chief-relative coordinates, moments, OPD and fits; host completes unresolved rays, forms operand values and differences; tall LM QR on the GPU from m·n ≥ 65,536. Trust-region/acceptance decisions stay host | underlying G/OPL/R/T/E plus D |
| Tolerance sensitivity/Monte Carlo and compensators (`optim/tolerance`) | 2×tolerances; 10²–10⁵ trials; nested compensator FD batches | Trial and sensitivity batches through the candidate division above (pipelined sub-batches from 64 candidates); host perturbations, compensator decisions per active trial and statistics | D + exact trial RNG; fixed trial order and per-trial outcomes |
| Glass substitution (`optim/{glass,discrete,mixed}`, engine glass job) | catalogue filters, 10²–10⁴ choices; combinations × local optimisation | Host names/filters/search control; candidate/refinement batches through the candidate division above | D; final f64 reference scores/ranking retained |
| Material dispersion/thermal offsets, spectra, sensor tables, glass_info/glass_map (`core/materials`, `analysis/{spectrum,glass_map}`) | few SW values or 10³–10⁶ catalogue/λ/candidate values | Host small immutable resolution; GPU large numeric sweeps, reused tables and candidate updates | double-float indices/dispersion; require resulting G/OPL and E gates |
| RNG stream, random pupils, optimiser/tolerance/image/stray draws | 1–10⁷ counter blocks | GPU where consumed; tiny diagnostic stream may stay host | exact u32 and specified f32 conversion, CONVENTIONS §6 |
| Validation, catalogue/project parsing, snapshots/hashes, schemas, result packing, session/jobs/FFI | small irregular graphs/strings, bulk final copies | Host: authoritative boundary validation, I/O and control, no efficient optical GPU work | exact model/metadata; f64 reference values preserved |

Memory planning is part of dispatch: packed surfaces cost `176 + 624S` bytes
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

The resident interface extends `ComputeBackend` (§1); it is not a second
backend API. Its typed contracts, by module of `mklight_gpu` (the C headers
in `include/` are the authority for the numeric records):

| Contract | Inputs → outputs | Module |
|---|---|---|
| `ResidentPlan`, `BufferHandle`, `BufferView`, `PlanStats` | validated typed operations/views → pending resident outputs, explicit final readback and per-stage stats | `resident` |
| `GenerateAim`, `TraceResident`, `ProjectPupil` | uploaded systems + logical field/λ/pupil indices → hi+lo rays/status; trace → hi+lo hits/OPL; reference descriptor → OPD, pupil coordinates | `aiming` |
| `TransformPlan`, `CorrelationStencil` | complex grids, dimensions/strides, sign/scale, affine map → complex grids or frequency values | `transforms` |
| `SegmentReduce`, `StableCompact`, `StableSort`, `Scan`, `QrFit` | indexed typed resident channels → moments, ordered records, segment outputs or fit factors | `collectives` |

#### Resident interface

The API is `mklight_gpu::resident` and the nine operation modules in the
table below. The solver crates (analysis, wave, ghost, stray, optim) depend
on `mklight-gpu` and hold their adapters in a public `gpu` module. The
buffer and plan methods of `ComputeBackend` are listed in §1;
`upload_buffer`, `readback` (the copying conveniences for owned arrays) and
`import_buffer` belong to `QualificationBackend` and are used by tests only.

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
the flattened row-major index. Plan imports and sums require stride 1;
`readback` accepts a strided view; `read_view` requires stride 1. Empty buffers are legal;
a sum requires positive `[channels,count]`. Scratch destinations come from
an exact-size pool and start with unspecified contents: an operation writes
every element it publishes. Rust request structs contain host metadata/IDs
and are never memcpy'd to a shader. The C header umbrella
(`mklight_gpu_types.h`) includes one `include/mklight_<module>_types.h` per
operation module for its numeric records; its `MKL_*` defines join the
generated Rust constants.

`TraceUpload::new(request, output)` shares its immutable host request between
plan clones and adapts the existing host ray batch without altering the
trace algorithm. The four
outputs are f32 `[10,rays]`, u32 `[3,rays]`, f32
`[record_slots,rays,9]`, and u32 `[record_slots,rays]`, using the existing
`MKL_TRACE_*` plane order. No-record buffers have zero slots.
`Sum { input: PlanView, mode: ReduceMode, output: BufferId }` reads a
contiguous f32 `[channels,count]` view and writes `DoubleFloat [channels]`
using the existing two-pass fixed tree. It sums every supplied value;
masking and compaction are `collectives` operations. A sum selects trace
planes by view offset, with no intermediate host array or upload.

| Operation module | Request variants / output records |
|---|---|
| `aiming` | `GenerateAim`, `TraceResident`, `ProjectPupil`; `ResidentRays`, `ResidentHits`, `PupilReference` |
| `transforms` | `TransformPlan`, `CorrelationStencil`; explicit dimensions, sign, scale and double-float affine coefficients |
| `collectives` | `SegmentReduce`, `StableCompact`, `StableSort`, `Scan`, `QrFit`; explicit IDs/counts, factors and rank/condition destinations |
| `geometry`, `radiometry` | numeric geometry settings/samples and coating-stack/hit records, with status destinations |
| `wave_ops`, `ghost_ops`, `stray_ops`, `optim_ops` | numeric solver records and settings views |

Exact field names and types are in each `src/<module>/mod.rs`. Requests
read uploaded packed systems
by their `SystemHandle`. `ResidentRays` uses the 15 existing input
float planes (including position/direction residuals), two index planes and
separate two-word logical IDs. `ResidentHits` adds status and optional
record/residual sidecars; the aiming kernels keep hi+lo geometry internally.
The central routers invoke each module's `validate(request, plan)`,
`accesses(request) -> Access { reads, writes, systems }`, and
`execute(request, execution, control)` hooks. Referenced immutable systems
are pinned at submission, and both execution contexts provide a system
lookup and the operation-cache hooks `cache_key(operation)`, `cached<T>` and
`cache<T>` for package handlers. CPU execution exposes word
slices; Metal execution exposes typed view bindings, exclusive output
leases, pooled temporaries and the existing command encoder. A module
changes its handlers without editing crate roots or central routers.

The pool holds at most 256 MiB of idle storage per backend. Immutable
caches are split into categories with independent byte and entry budgets
(`CacheCategory::budget`), so one never evicts another: prepared (uploaded)
systems 64 MiB / 64 entries, chief sources 8 MiB / 1024, seed tables
64 MiB / 256, traced pupils 128 MiB / 64, transform plans (recipes,
twiddles, chirps, immutable spectra) 64 MiB / 256 and small metadata such
as path programs 4 MiB / 512. Within an operation category, entries are
retained by measured rebuild cost per byte with recency aging
(GreedyDual-Size: priority = category clock + cost/bytes, refreshed on a
hit; the lowest priority is evicted and advances the clock), so an
expensive seed table outlives cheap recent entries. Build cost is host
time, or for device builds the wall time until the building command
completed (an upper bound when fused with later work).
`cache_key(category, operation)` combines the plan's
snapshot/library/configuration/spectrum, layout, build and settings
identity with the operation name and plan version;
`content_key(category, operation)` keeps only layout and build, for
operations whose name carries digests of every input. Entries publish only
after the work building them completed: a cancelled or failed plan
publishes nothing (the CPU specification defers publication to plan
completion too). `PlanStats.cache` holds each plan's hits, misses, inserts,
rejections, evictions, evicted bytes and build seconds (spent, saved,
evicted) per category; `resident_cache_stats().categories` accumulates them
with occupancy and budgets. The pipeline cache
holds at most 512 specialisations (`PIPELINE_CACHE`); live jobs retain
evicted objects. Active leases/handles
pin resources outside idle-cache limits. System cache identity includes
snapshot/library/configuration/spectrum, layout, build and operation
settings, plus a digest of the actual packed content. Cache hits issue
independently releasable handles. Resident trace commands are capped at
262,144 rays (`MAX_TRACE_CHUNK`); sums use the backend chunk and partition reduction groups without
changing the tree. The idle pool reuses the most recently returned buffer
of the exact size and evicts the oldest. Limits
are resource/cancellation controls, not engine backend selection thresholds.

**Resident command fusion.** The Metal executor owns a worker-local deferred
command buffer. Domain operations, trace tiles and fixed-tree sum passes share its serial compute
encoder, with buffer memory barriers after dispatches. Existing tile boundaries
remain cancellation checkpoints; accumulated dispatch work (2¹⁹ threads, charging at least 2¹⁵ per dispatch) or
10 ms of encoding triggers a commit (the host waits only where the
in-flight window below requires it). These are work/encoding
bounds, not a promise of 10 ms device time for an arbitrary kernel. A genuine
host read or publication flushes the buffer. Reporting-only fallback counters
and pupil-normalisation checks are deferred to completion. Cache entries built
by GPU work remain local to the executor until a flush; same-plan consumers can
reuse them immediately. Time-calibrated tiles (Newton aiming,
coating/polarization, ghost probes) still target 10–15 ms of
measured device time: they calibrate from a submission the work bound
completed, or complete the batch as a timing checkpoint only when another tile
follows and it would exceed the operation's unmeasured size (aiming: the first
of several tiles, and every tile of large batches). Single-tile operations and
small chief/table/pupil chains stay fused. Tiny expensive instruction
sequences are bounded by the dispatch charge as well as thread work. Recycled
scratch stays pinned until completion; abandoned, uncommitted encoding is
discarded on cancellation or error. Internal shared-storage guards reject host
slices, and synchronous (non-batched) command bindings, while a conflicting
pending device access exists, since such a command would run before the
uncommitted batch. No pending storage becomes a public `ReadView`.
`ResidentPlan.command_work_budget` selects that work bound (default 2¹⁹);
zero completes each encoder call separately for scheduling cross-checks. The
CPU-f32 executor accepts the same plans and preserves its specified arithmetic.
The selected budget is recorded in `RunInfo`.

Dependent chains compose by appending operations and retaining `BufferId` /
`PlanView` references until one final submission. The wave probe builder
uses this pattern across its complete rim schedule: each `Lattice::Rim`
consumes the previous round's bracket state and trace status directly on
device, appends aiming and trace operations, and feeds the next round
without host inspection; all unresolved-aim and final-bracket checks run on
completed leases. Candidate ray plans (`aiming` trace, `geometry`
coordinates and OPD, `collectives` moments and fits) and the per-field
transform chains share command buffers within their plans. Host-computed sphere/scale/grid metadata still
separates plans where the next operation genuinely depends on it; there is no
public pending-buffer view or implicit asynchronous import.

**Asynchronous execution and admission.** A worker commits its command
buffers without waiting for them; at most two committed ones stay in flight
(`MKLIGHT_METAL_IN_FLIGHT`, 0 = complete each before the next), and only one
per worker once three are in flight process-wide, so side-by-side plans queue
about as deep as the synchronous executor did. The host
waits only where it needs device results (a pending count, a timing
checkpoint, a final or intermediate readback), when the window is full, and
when an abandoned batch drops: every committed command keeps the access guards
of its bindings until it has completed, so no host slice, pool reuse or cache
publication observes storage it may still access, and a cancelled plan waits
for at most the bounded commands already in flight. Unified memory is used
as is: counts and results are read in place from shared-storage leases, never
copied or formatted. The Metal backend admits at most four resident plans at
once and at most 1 GiB of their declared storage (a larger plan runs alone;
`MKLIGHT_METAL_ADMIT_PLANS`, `MKLIGHT_METAL_ADMIT_MIB`); a plan whose consumer
was cancelled while it waited never reaches the device. Consumers may
therefore run independent plan chains side by side: the GPU wave adapter
queues each field the host has prepared (host-traced and resampled pupils)
for up to four device workers,
which run its transforms while the host prepares later fields, and merges
results, statistics and warnings in item order. `PlanStats` records host waits and
their seconds, admission seconds, command buffers over 50 ms of device time
(`long_commands`) and speculation misses; `MKLIGHT_METAL_TRACE_PLANS=1`
prints one line per plan. Device time sums command-buffer spans, so
overlapping plans count shared device time more than once.

`PlanStats` counts actual completed command buffers; deferred dispatches have
zero commands at encode time and their timestamps/counts appear in the
`fused completion` stage. Arithmetic and reduction trees are unchanged.

Scratch allocations begin at their producer and retire after their last
consumer; settled retired leases return to the pool after work-bound
completions, without waiting for a later explicit flush. Rim rounds of the
wave probe builder retain only intermediate aim checks and the final probe
batch. Command counts and timestamps are recorded at completion, not at the
encoding site. Routing between the backends is the cost model ("Automatic
routing").

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
pool storage, immutable operation caches and host metadata. A backend
returns errors for unsupported work, resource failures and cancellation;
the fallback policy is the engine's (§9.3).

**Residency.** A backend owns its queue, pipelines, system cache and buffer
pool for its lifetime. Views carry type, shape, stride, owner and generation;
validate at upload/plan submission, not in every kernel. Leases pin buffers
until all dependent command buffers finish, including cancelled jobs;
mutable scratch is never shared between live jobs. Plan-scoped caches use
the derived snapshot, library, configuration/spectrum, layout and kernel
build as keys, and optical caches also include all sampling/radiometry
settings; content-addressed entries (aiming; transform recipes, twiddle
and chirp tables and chirp spectra, which depend only on their parameters)
use digests of every input instead, so they survive edits and are shared
by plans of other analyses. The engine's analysis graph
(`docs/ARCHITECTURE.md`) hands every plan of a snapshot the same reference
tracer, so host chief rows reuse its prepared fields across analyses.
Warm calls reuse allocations, packed systems, aim seeds, transform plans,
twiddles/chirps, pupil grids and stencils. Upload only changed candidate
parameters; read back requested final buffers/scalars, never intermediate
trace arrays just to upload them again. Keep current host `Job` calls as
adapters. Solvers hold no Metal objects. A small ordered operation list
suffices; this is not a general graph compiler.

**Aiming and OPD.** Field kinds, finite/infinite and telecentric launches,
vignetting and pupil patterns are generated on the device; pupil rays that
need real aiming are aimed where the analysis's division puts them.

*Host aiming* is the default. The geometric adapters (`gpu::aim` in
`mklight-analysis`, route option `aiming`, default `host-table`) aim every
ray with the reference tracer in f64 and write the start rays straight into
the shared ray planes the device traces (four `fill_buffer` buffers filled
in place by the rayon workers; no sample or field records). Generations of
at least 2,048 rays per field row (`HOST_TABLE_MIN_PER_ROW`) first solve a
coarse f64 seed table per row and start each ray's Newton solve from its
interpolated aim and stop Jacobian (`SequentialTracer::generate_many_from`;
the reference schedule takes over where it does not converge); smaller
generations are aimed bitwise as the CPU reference aims them. System
transmission, the wave analyses' pupil lattices and, except for large
aspheric batches, the solvers' candidate rays are aimed on the host as well
(§9.1).

*Split generations.* In the spot diagram and the geometric MTF, a
generation with a seed table and at least 1.5 M rays × surfaces
(`SPLIT_MIN_WORK`: rays times the image surface index) is aimed by host and
device together. It is cut into tiles of whole 128-ray chunks (a sixteenth
of the generation, 2¹³–2¹⁷ rays); a fixed, evenly interleaved share of the
tiles (`SPLIT_DEVICE_SHARE = 0.35`; the route option `aiming_share` sets
another, 0 aims everything on the host) goes to the device's double-float
Newton solve, seeded from the same table. Of its tiles the device takes only the
chunks whose rays all start inside the seed table's solved nodes: the rays
the reference cannot aim lie at unsolved nodes, and on the device they
would run to the end of the schedule before being aimed again on the host.
The device's chunks form two plans per generation (`DEVICE_PLANS`), both
submitted from a scoped thread of their own outside the rayon pool (it
mostly waits for the GPU); the rayon workers aim the host's chunks and then
complete each device plan's `MKL_AIM_FAILED` rays in f64 as it lands. The
assignment depends only on the generation and the share, never on which
side is faster, so the output is bitwise reproducible
(`crates/mklight-analysis/tests/aim_designs.rs` runs it on one and four
rayon threads and with other shares). Measured on a quiet M2 Pro (8 + 4 CPU
cores, 19 GPU cores; shuffled columns against the same build aiming on the
host alone): dense spot diagrams and geometric MTFs of four complex camera
lenses (4–13 M rays × surfaces per field group) took 0.94 of the host-only
time (0.84–1.08 per case), 0.96 between 1.5 and 4 M, and lost up to 18 %
below 1.5 M; shares 0.5 and 0.65 gave the same mean with worst cases of 1.26
and 1.32. One device plan per tile ran 4–5× slower per ray than the twelve
host threads and up to 2.5× slower than host-only aiming, which is why the
device gets few large plans of table-covered chunks. The footprint and the
ray grid gained nothing from a split (1.00–1.01, 0.90–1.17) and do not split.

*Plans.* Field-group plans are the geometric adapters' only reduction: the
spot diagram, geometric MTF, footprint and ray grid split their fields into
groups of at least 2¹⁵ rays (`gpu::plans`), each a plan submitted as soon as
its rays are aimed, so the host aims group k + 1 while the device traces and
reduces group k (ANALYSIS.md §6.1). Against one plan per stage the groups
took 4–7 % off (geometric means 0.93–0.96 by CPU-time class); with the split
generations, dense spot diagrams and geometric MTFs took 0.81 of the time
before both (0.66–0.99, 14 cases on four complex camera lenses), their
largest relative deviation from the CPU reference unchanged (≤ 1e-7).

*Device aiming* (the double-float solve described next) remains for
`aiming=device` and `aiming=table` (on the M2 Pro 1.63× and 1.15× the
default's time on large geometric cases, as accurate; kept as routes for
GPUs with more cores per CPU core), the solvers' large aspheric batches,
`fieldmaps=device-chief`, the chief aims of the ghost device paths and the
wave `PupilTracer` (the solvers' MTF operands). Chief rows come from the
host: `ChiefSeed::host` converts the reference tracer's f64
`PreparedField` (the real-image-height source once per field, the chief aim
point and its analytic stop Jacobian once per field and wavelength,
with the same 1e-12 relative image-height residual the device solve used)
into the device row. Every device-aiming path passes these rows as
`GenerateAim.seeds`, reusing the tracer's prepared system fields
and solving other field points once each. The device chief kernel
(`mkl_aim_chiefs`, one thread per field row, a long nested serial solve)
remains only for requests without rows (`fieldmaps=device-chief`). Chief rows and seed tables are
reused across plans by content (`aiming::aim_identity`): the digest of the
immutable upload (`PackedSystemSet::digest`, shared by every lease and by
re-uploads of equal content), the exact imported field rows, the host rows
or the device solve, tolerance and iteration cap. Handles, external cache
keys, snapshots and plan settings never identify them, so analyses and
leases with equal inputs share them and any geometry, catalogue or
environment change misses; rows produced inside a plan stay plan-local. `PlanStats.aiming` records each plan's generations
(rays, failures, iteration sum/maximum and a histogram, reduced on the
device by `mkl_aim_stats`), `PlanStats.host` the host preparation (rows,
reused/solved fields, seconds) and `PlanStats.memory` the device allocation
and process RSS high-water mark at completion. Newton–Broyden uses double-float
stop hits, finite differences, 2×2 solves and updates, the reference's
continuation schedule and configured cap/tolerance (the request's aim-stop
policy, CONVENTIONS §7: the requested tolerance, normally 1e-10 relative to
the stop radius, or the tightest conditioned stop of the request's fields,
`SequentialTracer::device_aim_tolerance`; one value per generation, part of
`aim_identity`). A rounded f32 stop hit cannot drive that solve. Unresolved
rays keep their index, logical ID and failure report; the adapters complete
them on the host in f64 where they do (host completion above) and record
counts/reasons. (The CPU reference takes its
fresh Jacobians from the analytic differential tracer, CONVENTIONS §7; the
device solve keeps its finite-difference stencil and is seeded with the
host rows' analytic chief Jacobians.) A failed aim remains a failed
aim; never silently accept an inaccurate ray or polish the whole grid by
default.

**Bounded aiming commands.** A pupil aim's schedule (table solve, chief
solve, four continuation stages, each up to the cap with a 16-step line
search and Jacobian refreshes) can cost thousands of stop traces, and one
such lane serialises its SIMD group, so bounding the *ray count* of a
command never bounded its duration (single 4096-ray commands reached 0.5–0.9 s
and tripped the driver's interactivity watchdog). Device aiming is therefore
resumable: `mkl_generate_aim` and `mkl_aim_table` run at most a budget of
iteration units per lane (a phase entry, or one Newton iteration; 4 per
command by default, `MetalBackend::set_aim_budget` for qualification), park
unfinished rays in a per-tile `MklAimResume` state buffer
(`MKL_AIM_STATE_WORDS`) and append their indices to a pending list; the
executor encodes continuation rounds (`mkl_aim_indirect` sizes
`mkl_aim_continue`/`mkl_aim_table_continue` from the pending count on the
device) in groups (2, 2, 4, 8, …), one command buffer per round: a round of
a few hard lanes runs as long as the slowest lane's budget, so fused rounds
add up past the bound (a 4096-lane candidate tile with two rounds fused into
its first command ran 77–115 ms). Once a tile of a plan has needed more than its
first group, the next group is committed before the host reads the previous
group's count, so the device does not idle on that readback; after the last
needed group the speculative one finds empty lists and dispatches no thread. A resumed solve performs the specification's operations in
the specification's order, so results are bitwise those of the monolithic
CPU-f32 kernel (`aim_resume.rs` forces one unit per command). Time-calibrated
tiles still bound the throughput part. `AimOptions::speculative_rounds`
lets a generation that is expected to converge within them (the wave pupil
tracer's table- and chief-seeded aims: one round, fused with the first
command; the candidate evaluator's capped device aims: the rounds that cover
the cap) skip the readback altogether: the executor verifies at the plan's
next completion, before any check, cache publication or host read, that no
ray was left pending, and the backend otherwise runs the plan again without
speculation (`PlanStats::speculation_misses`), so results never depend on it.
`AimOptions::schedule_cap` bounds a ray's Newton iterations over its whole
schedule (every solve runs with the iterations left under it, and an
unconverged solve that reaches it ends the schedule; 0, the default, keeps
the per-solve cap as the only bound). The candidate evaluator aims on the
host by default; with device aiming (`solver_aiming=device`) it sets 8
(`SCHEDULE_CAP`) and aims every ray the device leaves unresolved
again on the host in f64 with the candidate's tracer, so no candidate is
reported whole to the reference. Without a cap an unaimable ray of a bad
trial step (a line-search candidate whose marginal rays miss or vignette)
runs the whole schedule at low occupancy: 300–400 ms of device time per
such candidate against 1–5 ms for a healthy one on a 29-surface F/1.4
design (Apple M2 Pro, indicative). Healthy candidate rays converge in at
most 4–6 iterations; a ray the full schedule converges within the cap is
bitwise unchanged (`aim_resume.rs`). Speculative Jacobians (OPTIMIZATION.md)
are used only with device aiming. The geometric adapters' table-seeded
device corrections (`aiming=table`, the device share of split generations)
stop after 12 iterations (`TABLE_SCHEDULE_CAP`, ANALYSIS.md §6.1); every
other analysis aim keeps the full schedule.
`MKLIGHT_METAL_TRACE_AIMS=1` prints every generation's convergence
statistics (rays, failures, iteration sum/maximum, histogram) to stderr.

**Excluded samples and seed tables.** `AimOptions::cull_outside` excludes
samples outside the unit pupil disk (the lattice guard nodes a survival mask
discards anyway, which were the costliest aims: unreachable targets running
the whole schedule) and `MKL_AIM_SKIP` in a sample's field word excludes
that sample (rim probes of edges already within tolerance); both carry
`MKL_AIM_OUTSIDE`, are never aimed or traced, and are not counted as aims,
failures or unresolved rays. `AimOptions::table` supplies a host f64 seed
table (`aiming::host_tables`: the reference tracer solves each node by its
own chief-prediction/continuation schedule) of `table_side` nodes per axis,
used for every sample of the request whatever its size; without one the
device builds and caches a table of that side for large generations. The
wave pupil tracer hands 25×25 host tables to every generation of a batch
(chief probe, lattice, saddle and rim probes alike), so a sample's
converged ray does not depend on the batch it was aimed in. The
tables are solved (one batched `generate_many` per row, bitwise the scalar
`generate` per node) while the batch's chief plan runs on the device.

**Operation modules.** What each device service computes, and its
conventions, is described with its consumer:

* pupil construction, the reference sphere and OPD: WAVE.md §3 and §9
  ("Pupils and automatic sampling"); the point convention of exported
  geometry: CONVENTIONS §12;
* transforms (FFT, chirp-z and affine MFT, convolution, correlation):
  `crates/mklight-gpu/src/transforms/README.md` and WAVE.md §9;
* collectives (segmented fixed-tree reductions, stable compaction, sort and
  scan, QR fits) and Philox addressing by logical item: §3.4, §3.5 and
  `mklight_gpu::collectives`;
* coatings, radiometry and polarisation: POLARIZATION.md §9;
* ghost probes and maps: GHOST.md §6; stray transport and the compute
  caster: STRAY.md, "Resident GPU transport";
* candidate systems: prepared on the host in f64, one packed system per
  candidate ("Candidate solvers" below).

**Host/device divisions of the analyses.** Each native analysis has one
default division of its work between the host's f64 reference code (on all
cores) and the device, measured on an M2 Pro; the route options of §9.3 keep
GPU-heavier alternatives where a stronger GPU may win.

* *Geometric analyses*: host aiming with seed tables, split generations for
  dense spot diagrams and geometric MTFs, field-group plans ("Aiming and
  OPD" above; ANALYSIS.md §6.1).
* *Field maps* (ANALYSIS.md §6.2): relative illumination has one pipelined
  division (the host prepares the fields as the reference does and solves
  the pupil nodes' image points in blocks of at least 2048 nodes; the
  device traces each block's differential pencils, `mkl_geometry_ri_rays`,
  while the host solves the next); system transmission has one too (the
  host aims each block of rows, at least 4096 rays, and the device traces
  it and evaluates the coating transport while the host aims the next;
  POLARIZATION.md §9). The chief-ray maps (`chief_ray_angle`,
  `pupil_aberration`, `grid_distortion`) solve their chiefs on the host and
  trace on the device (`fieldmaps=host-chief`, the default) or solve them
  on the device too (`device-chief`: a fixed 30–100 ms of serial device
  solves on the M2 Pro, a tie on 41² grid distortion);
  `field_curvature_distortion` always solves its chiefs on the host.
* *Wave optics* (WAVE.md §9): one division. The host traces the pupil
  lattices with the reference tracer, resamples and refines them with the
  reference's code and makes every sampling decision (pupil grids, image
  windows, automatic frequencies, focus ranges); the device runs the
  transforms (PSFs, OTFs, focus sweeps, energy, line spreads, convolution),
  and the device transforms early fields while the host prepares later
  ones. `wavefront_map` has no device part: its statistics and Zernike fit
  are reductions of the f64 pupil the host already holds.
* *Stray light* (STRAY.md, "Divisions between the CPU and the GPU"): the
  device runs the wavefront transport of the first 64 events of its
  sources and writes detections, ledgers and continuation stacks into
  shared memory; the CPU reads them in place, continues longer path trees
  and forms the estimators. By default the device gets a fixed share of
  the source chunks (`gpu::hybrid::default_share`: 0.2, and 0 for scenes
  with iteratively intersected sags or fewer than 32 768 device sources)
  while the CPU reference traces the others; `stray_share=1` gives it every
  source. A run whose share is zero is the CPU reference run, which `auto`
  selects from the request alone ("Automatic routing" below).
* *Ghost irradiance* (GHOST.md): one division. The paths form four groups
  (`gpu::PIPELINE_GROUPS`), pipelined: the device traces one group's probe
  tiles while the host completes, deduplicates and splits another; the CPU
  reference traces a fixed 20 % of the path groups (`gpu::CPU_SHARE`)
  beside the device run.
* *Candidate solvers*: host candidates, aiming by batch (the host, the
  device for large aspheric batches), one device ray plan per batch ("Candidate solvers" below; OPTIMIZATION.md, "Host/device division
  of candidate batches").

#### Runtime publication policy

Edits do not trigger a CPU run. Native Metal results publish immediately as
`gpu:metal` with their actual f32/double-float precision; there is no sampled
per-snapshot geometry qualification, full-chain reference comparison or
per-candidate f64 check at run time. Accuracy rests on the offline f64
cross-validation suites (perturbed systems and known answers included) and on
the ignored `metal_accuracy::default_sample_accuracy` harness
(`mklight-bench`), which compares every GPU-capable analysis on all samples
at default settings with the CPU reference under the per-quantity contracts
of §9.4. The startup numerics
probe, boundary packing checks, device status/finite guards and the kernels'
conservation sums remain as runtime guards; none repeats the CPU solve.

**Analyses without a GPU path.** `gpu::GPU_ANALYSES`
(`crates/mklight-engine/src/analysis/gpu.rs`) is the one catalogue of the
native routes; every other analysis is computed by the CPU reference on any
backend request, like first-order data or the Seidel sums. Three of them
belong to families that otherwise have native routes:

| Analysis | Why the reference computes it |
|---|---|
| `encircled_energy` | its curve is a step function of the ray distances: a ray within the device's position budget of a radius sample falls on either side of it, so the energy bins need the f64 positions |
| `polarization_pupil_map` | its fast and diattenuation axes are undefined where retardance or diattenuation vanishes, and f32 rounding turns them by 1–2 rad there |
| `wavefront_map` | the host traces and resamples every wave pupil in f64 (WAVE.md §9); the map's statistics, Zernike fit and Seidel terms are reductions of those arrays, which a device could only read back at lower precision |

**Residual gaps on Metal.** Native results publish unchecked, so known
differences are listed here. Under the per-quantity contracts of §9.4,
`metal_accuracy` (every GPU-capable analysis on the 16 samples at default
settings, release, real Metal) has no native gate failure; the cases that
still fail on a complex lens are the `auto` accuracy exclusions below. The
mechanisms the contracts rely on:

* `ghost_irradiance` `rays_traced`: clear apertures are sized by the edge
  field's rim rays, so lattice points can lie exactly on an aperture edge;
  clipping tests within 2⁻²⁸ of an edge are decided by the f64 reference.
* `stray_light`: at `n.z = −0` (coaxial cylinder normals: lens edges,
  barrels) both backends take the same branch of the tangent basis, so
  scatter frames agree; analog transport is bit-close (1e-16) and weighted
  transport differs only through rare Russian-roulette decisions, which the
  statistical contract classes cover.
* `footprint` hulls and pupil rims are compared by Hausdorff distance, not
  by vertex count.
* `field_curvature_distortion`: the close-ray cross-check column is the
  host f64 differential pencil on both backends; the native Coddington
  column is within budget.
* Relay `ghost_paths`: R3·R1 and R7·R5 have the same relative irradiance and
  the backends round it differently in the last bits, so the ranking can
  swap the two (every value agrees to 1e-12); the harness matches ghost
  paths by label.

**Candidate solvers.** `merit_evaluation`, `optimize`, `tolerance` and
`glass_substitution` divide each batch of candidates between host and device
(`mklight_optim::gpu::ResidentEvaluator`; OPTIMIZATION.md, "Host/device
division of candidate batches"). The host prepares every candidate in f64 in
parallel (tracers with their real-image-height and chief solves, operand
plans, the chief rays themselves, model, paraxial and first-order operands)
and aims the pupil rays where the batch's rule puts them (below). One device
plan per batch traces the pupil rays and forms chief-relative
coordinates, moments, OPD and piston/tilt fits; batches of at least 64
candidates (`PIPELINE_FROM`) run as three pipelined sub-batches, so the host
prepares sub-batch k + 1 while the device works on k. The host reads the
device rays in place, completes the rays the device left unresolved in
f64 and forms every operand value; the search loops (LM, trust region, Monte Carlo, glass
search) and their decisions stay on the host, and LM's QR moves to the
device for tall Jacobians (QR policy below).

The evaluator decides per batch where the pupil rays are aimed
(`gpu::DEVICE_AIM_FROM`). A batch whose estimated device rays times the
system's surfaces without a closed-form intersection (aspheres, polynomial
and freeform sags) reach 2·10⁶ is aimed on the device with host completion
(capped at 8 iterations, "Bounded aiming commands" above). Every other
batch is aimed on the host with the candidate's tracer (`generate_many`, the
reference's aims bit for bit), and the device only traces and reduces it.
Device aiming pays only where host aiming is expensive per ray and the
batch amortises the device's plan: dense Monte Carlo on aspheric lenses
(0.79–0.84× host aiming's time; 1000 trials on a 35-surface ultra-wide zoom
took 0.69× the CPU's time). It never pays on spherical or paraxially aimed
lenses, LM Jacobians or single candidates. The route option `solver_aiming`
pins either side for every batch (`device`, `host`; `by-batch` is this
rule), and LM and dogleg speculate only with device aiming pinned: a trial
point that follows an accepted step is evaluated in one batch with its
Jacobian's first-pass probes. Forward differences of device operands take
`η = 1e-6` (ray positions) and `1e-5` (wavefront operands), central
differences the reference's steps: device values are double-float traces of
f64 aims (about 1e-12 relative noise), and a step as large as `1e-3` lets a
probe of a 35-surface ultra-wide zoom cross a vignetting edge. Unsupported preparation/features and
actual resource/device failures receive a counted reference completion, and
a run in which device aiming leaves more than half of a candidate's rays to
the host continues on the reference (counted as a fallback). The backend block
reports `precision: f32_gpu`, the tag of every native Metal result (f32
decisions with double-float geometry, CONVENTIONS §7), not plain f32 values;
`backend.aiming` says whether the host or the device aimed. The
final report is scored once in f64: optimisation scores the start and the selected
design; tolerance scores each completed case with its returned compensators
(no second compensator solve) and derives the sensitivity/Monte Carlo figures
from those scores; glass substitution scores and ranks the kept finalists.
`merit_evaluation` publishes the evaluator's values. Cancellation never starts
a reference replacement or report. `auto` routes them by the cost model like
every other analysis ("Automatic routing" below; `glass_substitution` is an
accuracy exclusion). An explicit Metal request still runs them natively.

**QR policy.** Resident LM solves use the device tall QR when the Jacobian
has `m·n ≥ 65,536` elements (and `m ≥ n`), the host Householder QR below.
The forced probe measures actual tall compression, retained Q, readout and
the small f64 completion; host QR won up to 400 × 100 (40,000), the device
from 4000 × 32 (128,000) and 1000 × 100 (100,000, near parity).

| Rows × columns | m·n | CPU ms median / p95 | Resident ms median / p95 |
|---|---:|---:|---:|
| 32 × 8 | 256 | 0.003 / 0.003 | 1.166 / 1.208 |
| 128 × 32 | 4,096 | 0.130 / 0.145 | 4.199 / 4.637 |
| 400 × 100 | 40,000 | 4.014 / 4.394 | 10.255 / 10.603 |
| 1000 × 100 | 100,000 | 8.654 / 8.689 | 9.854 / 10.332 |
| 2000 × 60 | 120,000 | 9.311 / 9.462 | 6.410 / 6.608 |
| 4000 × 32 | 128,000 | 6.827 / 7.507 | 6.129 / 7.524 |
| 4225 × 37 | 156,325 | 8.130 / 8.257 | 5.656 / 5.751 |
| 1600 × 100 | 160,000 | 19.035 / 19.510 | 11.559 / 11.789 |
| 2000 × 100 | 200,000 | 24.669 / 25.067 | 12.433 / 20.034 |
| 4000 × 100 | 400,000 | 50.199 / 51.128 | 16.766 / 23.121 |
| 65536 × 37 | 2,424,832 | 267.163 / 286.784 | 46.402 / 49.793 |

`m·n²` does not separate these shapes (400 × 100 and 4000 × 32 have the same
work); `m·n` does. A rank-deficient reference factor remains a named,
counted numerical fallback; an underdetermined system is solved on the host.

#### Automatic routing

`auto` routes each engine job with a calibrated stage-level cost model
(`crates/mklight-engine/src/analysis/cost/`; `routing::auto_route` calls
`CostModel::decide`, model version 5). In order: a result the analysis graph
already holds is returned (below); then

- *Division rule* (`routing::reference_by_division`): a run whose native
  division gives the device no part of the work is the CPU reference, and
  is settled from the request alone, before any device opens and without a
  prediction (`decision: "cpu_reference"`, `rule: "no device share"`).
  `stray_light` has such a division: the transport's device share of the
  source chunks (`mklight_stray::gpu::hybrid::planned_share`, the rule of
  `default_share` evaluated on the system's surface shapes: the route
  option `stray_share` when set, else 0.2 from 163,840 rays on when every
  sag is a plane or a conic, and 0 otherwise) is known from the system, the
  ray count and the route options. A native stray run that still gets no
  device source (a share smaller than one unit of chunks; an explicit
  Metal request below the threshold) is the same reference run and reports
  the same reason after the fact, without a failed attempt.
- *Accuracy exclusions* (`cost::EXCLUSIONS`): analyses whose native output fails
  its f64 contract on the current evidence run on the CPU reference on every
  device, at every size (table below).
  Explicit Metal requests still run natively.
- *Known failures*: a request that already failed deterministically on the
  device (below) runs on the reference (`rule: "known failure"`).
- Every other GPU analysis is predicted on both backends as a sum of stage terms
  over its work profile (`cost::profile`), computed from the normalised settings
  and the system without tracing:

  | Stage | Feature | Notes |
  |---|---|---|
  | host f64 preparation | field–wavelength rows × surfaces | sources, chiefs, Jacobians; shared by both backends through the graph `tracer` node, so 0 when it is already built |
  | aiming and tracing | rays × (surfaces + aim traces × stop surfaces) | aim traces = 4.43 per aimed ray, the reference solver's mean 3.43 Newton iterations and the ray's own trace (`cost::AIM_TRACES`; paraxial aiming: 0) |
  | transform / reduction | samples × log₂ samples (FFT), samples × frequencies (direct sums) | the automatic wave pupil grid is the one an earlier call decided for the design family, else 2·lattice |
  | output bytes | typed buffers | spot points, ray grids, PSF images; zero-copy readback |
  | device seed table | rows × 37² nodes × aim traces × stop surfaces | Metal on an edited snapshot only: aimed geometric generations of ≥ 2·37² rays per row build a device table, content-cached while the snapshot is resident |
  | fixed | 1 | each analysis' own measured constant on each backend (plan floor, packing); there is no universal GPU overhead |
  | device setup | device open (cold provider) + first native use of the analysis in the process | measured by the calibration; charged in full to the first call of an analysis and design family, then spread over up to 4 calls. The first-use part is pipeline creation: it is not charged once `Engine::prepare_gpu` has created the kernels (`state.kernels_prepared` in the decision) |
  | fallback risk | transient native failure rate × (native + CPU time) | deterministic failures in the device's failure cache route to the CPU outright |

  The device-seed-table term describes the device-aiming routes
  (`aiming=device`/`table`); host aiming is the default (above).
  Four non-negative coefficient sets per analysis — CPU and Metal, each on a
  repeated and on a freshly edited snapshot (`cost::Mode`) — are fitted by
  load-weighted relative least squares from complete calls. A set is fitted
  with the features the prediction applies it to (`Mode::features`, called
  by both): the seed-table term is zero for the CPU sets and for Metal on a
  resident snapshot, so only the edited-Metal set has a table coefficient.
  CPU coefficients are measured
  with the graph off (cold intermediates); the CPU drops the preparation term
  when the graph holds the snapshot's tracer, and the ray term of wave
  analyses when it already built the request's pupil lattices. Each backend
  uses its repeated set when this snapshot ran before (natively, for Metal:
  resident content, including the seed table), the edit set otherwise.
- *Cache state*: an `auto` request whose result the analysis graph already
  holds is answered before routing (no device, no route), whichever backend
  computed it; its result key is the request content, not the route. Native
  `auto` results without failed attempts are stored like CPU results.
- *Decision*: Metal only when the predicted CPU time is at least `RATIO` = 1.3×
  the predicted Metal time *and* the saving is at least `MIN_GAIN_MS` = 5 ms.
  The ratio keeps the choice robust to the model's error (median calibration
  error per analysis in the shipped table: median 0.34 in ln of
  measured/predicted, 0.61 at the 90th percentile; `stray_light` fits worst,
  1.2, its device share following the scene rule); the absolute floor keeps
  sub-perceptual gains from paying a device round trip and contention with
  other GPU work.
- *Online refinement* (on by default; `MKLIGHT_COST_REFINEMENT=0` or
  `CostModel::with_table(.., false)` turns it off): every completed call
  updates (a) the CPU's host-contention factor (what other processes cost
  it now: the load-average prior √(load per core / 4), at least 1, times what
  observations learned beyond it (1/3…3×), in all 1/3…30×; it follows the
  load even while the CPU is not chosen; Metal has none, since the device is
  not shared with the host's threads) and (b) a residual per analysis,
  backend, design family and workload decade (1/10…10×), as exponentially
  weighted means (α = 0.5) of ln(measured/predicted); decided pupil grids
  are remembered. All state is bounded (LRU), recorded in each decision and
  reported by `Engine::cost_model_stats()`. A decision depends only on the
  state when it is made.
- *Repeated requests*: an identical `auto` request (snapshot and library
  content, analysis, normalised settings) takes the route its first decision
  chose, marked `"repeated":
  true` in the decision (LRU of 4,096 routes; `repeated_routes` in the
  stats). Refinement, the load prior and warm caches move the prediction
  between calls, and the two backends agree only within the accuracy
  contracts, so a fresh decision could change the result of the same
  request: a seeded Monte Carlo or optimisation repeats bit-identically in
  one engine. An edit or other settings decide afresh. A repeated route on
  a device that has since failed on the request deterministically is
  decided again (the reference).
- Devices without a calibration table (other GPU families, the CPU-f32
  emulation), and analyses the table does not cover, use qualified Metal
  (`decision: "qualified_metal"`).
- The table is one device class's (`calibration.device_class`: the shipped
  one was measured on an Apple M2 Pro) and predicts every Apple GPU, with
  the CPU's ray, transform and output stages scaled by the thread count. A
  decision made on an opened device of another name says so in
  `calibration.note` ("calibrated on Apple M2 Pro; this device is …"); the
  route is the table's either way.
- The model predicts each analysis's default division (§9.1). A route
  option of the run (§9.3) applies to its native run, `auto`'s included,
  but does not enter the prediction (`stray_share` enters the division rule
  above); `auto` never selects one.

**Calibration status.** The shipped table
(`crates/mklight-engine/src/analysis/cost/apple-m2-pro.json`, M2 Pro, model
version 5) is calibrated (2026-11-10) for the current divisions of the
native analyses: host aiming with seed tables,
field-group plans and split generations (geometric), one pipelined division
for RI and transmission, host-traced pupils with device transforms (wave),
`cosplit` stray transport, four pipelined ghost path groups with a CPU share,
and the host/device candidate division with its per-batch aiming rule
(solvers). It is fitted from a full run on the nine samples and the complex
lens at load 8–22 (partly the run's own CPU calls), with the four designs
measured above load 13 re-measured at 5–13. Device open is the warm 113 ms;
the first open after a shader change, which compiles the library, measured
222 ms. Two properties of this table differ from what the harness measures
and fits (below), and the next calibration removes them:

* Its CPU and resident-Metal sets were fitted with the seed-table column.
  The prediction never applies that column to them, so the table
  coefficients of those sets (non-zero for `spot_diagram`, `ray_grid`,
  `best_focus`, `geometric_mtf`, `footprint`, `system_transmission` and
  `stray_light`) are unused, and the time the fit attributed to them is
  missing from the prediction of aimed generations large enough to build a
  table: the CPU is predicted too fast there.
* Its `stray_light` rows were measured at 20,000 and 100,000 rays, where
  the device gets no source: both backends ran the CPU reference, so the
  Metal sets describe the CPU and predict a device share only by
  extrapolation. `auto` routes such runs by the division rule, not by these
  rows.

The row of `glass_substitution` is measured and never read (accuracy
exclusion); a unit test (`builtin_table_matches_this_model`) requires a row
for every analysis of `GPU_ANALYSES` and for no other.

**Calibration.** `calibrate` in
`crates/mklight-bench/tests/cost_calibration.rs` (release, real Metal)
measures every analysis with a GPU path at a ladder of two or three
workloads on nine sample designs (refracting, reflecting, finite,
anamorphic, freeform, diffractive) and an optional local project
(`MKLIGHT_BENCH_PROJECT=<file>`, the design `local`; nothing of it is
embedded) — the CPU and Metal, each repeated on one snapshot and after an
edit, interleaved so host load hits all four alike — and records the load,
device time and the decided pupil grid of every rung. `stray_light` is
measured where the device takes sources: 200,000, 400,000 and 800,000 rays,
on the designs whose sags it intersects in closed form (the other runs are
the CPU reference by rule and are never predicted). The fit keeps a rung's
least loaded record, weights records by 1/max(1, load per core / 2)², uses
only calls that ran on the backend they were asked for, gives each mode the
features of `Mode::features`, and writes the table, versioned by format,
model version, device class and measurement record. To recalibrate (on a
quiet machine, without other GPU work):

    MKLIGHT_BENCH_PROJECT=<local project file> \
    MKLIGHT_COST_CALIBRATION_OUT=$PWD/crates/mklight-engine/src/analysis/cost/apple-m2-pro.json \
    cargo test --release -p mklight-bench --test cost_calibration -- --ignored --exact calibrate --nocapture --test-threads=1

then `cargo test -p mklight-engine --lib cost` (the table against the
registry) and `validate` (same command with `--exact validate`), and update
"Calibration status" and "Measured routing" from their output. Without
`MKLIGHT_BENCH_PROJECT` the table is fitted from the nine samples alone.

`MKLIGHT_COST_MODES` and `MKLIGHT_COST_DESIGNS` restrict a run (for example
to re-measure designs that ran on a busy host), `MKLIGHT_COST_FILTER` its
analyses (a partial table, which `_OUT` refuses); `MKLIGHT_COST_REFIT=1` fits
one or more record files (`MKLIGHT_COST_RECORDS`, comma-separated) without
measuring. `MKLIGHT_COST_CALIBRATION=<file>` loads another table at engine
start (format and model version must match). `validate` in the same file compares `auto`
against both backends on fresh edits and warm repeats.

*Accuracy exclusions* are qualified with the per-quantity contracts of §9.4:
`metal_accuracy` on the 16 samples (release, real Metal: 431 rows, 0 native
gate failures) and a complex 29-surface F/1.4 lens (`project_bench` accuracy
records, Metal DF vs CPU f64, warm and edited). Analyses not in the table
pass both.

| Analysis | Status | Evidence |
|---|---|---|
| `glass_substitution` | excluded | complex lens, warm and edited: the native search returns another finalist glass (finalist merit 3× outside its gate); the CPU is about as fast (900 against 846 ms) |
| `ghost_irradiance` | routed by cost | 16/16; complex lens warm and edited: discrete counts exact (6,092 compared), positions ≤ 3e-7 of their gate |
| `spot_diagram` | routed by cost | 16/16; complex lens at 33.5M rays, warm and edited: `rays_arrived` exact, positions ≤ 3.2e-4 of their gate; Metal 7.1 s against 12.4 s |
| `psf`, `mtf`, `line_spread`, `diffraction_encircled_energy`, `through_focus_mtf`, `mtf_vs_field` | routed by cost | 16/16 within contract; complex lens within contract (MTF frequency class ≤ 0.016 of its gate, dense grids included) |
| `chief_ray_angle`, `pupil_aberration`, `relative_illumination_map`, `grid_distortion` | routed by cost | 16/16; complex lens without mask mismatches (host f64 chief rows) |
| `field_curvature_distortion` | routed by cost | 16/16; complex lens 2.8e-7 (host f64 differential pencil) |
| `stray_light` | routed by cost | the Metal route is `cosplit` (STRAY.md, "Divisions between the CPU and the GPU"): on nine lenses, the complex lens included, with six independent seeds per side, class totals and stray flux lie within 2.3 Monte Carlo standard errors of the reference and the irradiance blocks match the CPU control's statistics; with the reference's seeds the device reproduces its paths to rounding. Fit error 1.2 in ln, the table's worst; `validate` regret geometric mean 1.11, worst +5 ms |
| `footprint` | routed by cost | 16/16 under the outline contract (newtonian included); complex lens outline 0.07 of its gate |

The other candidate solvers (`optimize`, `tolerance`, `merit_evaluation`)
have no separate rule: the model predicts them like every other analysis.

*Measured routing* (indicative: Apple M2 Pro, `validate` with the shipped
table: the nine samples and the complex 29-surface F/1.4 lens, every
calibrated analysis at its first two workloads, after an edit and as a warm
repeat, CPU, Metal and `auto` interleaved, median of three repetitions, load
5–13; the numbers move with load and with every engine change, so the
shipped table is recalibrated rather than these figures kept current). Over
1,177 case-phases `auto` took 78.1 s, the faster backend of each case
78.7 s, the CPU alone 94.0 s and Metal alone 141.6 s; the median regret
(`auto` against the faster backend) was 1.07, the geometric mean 1.13.
`auto` chose Metal for most wave, map and image-simulation jobs on the larger
designs after an edit and for the complex lens's ghost maps after an edit
(Metal 10.3 s against 14.9 s on the CPU at 48 pupil samples, 35.8 s against
38.9 s at 96), and kept the CPU for small geometric
jobs, the solvers and glass substitution (accuracy exclusion). The
stray-light cases of the run (20,000 and 100,000 rays) lie below the
device's source threshold, where every backend runs the CPU reference and
`auto` selects it by the division rule. Five
case-phases missed by more than 1.3× and 20 ms, all warm repeats: four image
simulations (+30 to +185 ms: a repeat keeps its first call's route, Metal,
while the CPU would have answered from its cached intermediates) and one
`optimize` on the complex lens (+301 ms). A repeated identical `auto` request
is answered from the graph on either backend.

**Deciding before the device opens.** Providers describe their analyses,
device family and whether the device is open without opening it
(`BackendProvider::hint`; every Metal device on Apple silicon is the Apple
GPU). Routes the model settles as the reference are decided before any
device initialisation, so a cold call that ends on the CPU pays no Metal
setup; the decision carries `"before_device": true`. A cold device's setup is
part of the native prediction. Each decision is recorded in the backend
report (`cost_decision`, RESULTS §5). An analysis the provider has no path
for (first-order data, Seidel, glass data, solves, validation, encircled
energy, the polarisation pupil map, the wavefront map, ...) is settled from
the same hint, without opening the device: `auto` runs the reference with
the reason (no cost decision), and an explicit GPU request fails with
`BackendUnavailable`, or with `allow_backend_fallback` runs on the reference
with a warning.

**Failed and skipped native attempts.** `RunInfo.failed_attempts` lists every
backend attempt that did not produce the result: `failed` (the native run
failed after `elapsed_ms` and the reference completed the job), `skipped`
(the same request already failed deterministically on this device) and
`reference_completion` (the native adapter itself completed the whole
result in f64). Each device remembers
deterministic failures (model, support and numerical errors, not
device/driver, I/O or cancellation errors) by request identity — snapshot
hash, glass library hash, analysis and normalised settings — in a bounded
LRU (256 requests per device); an edit changes the identity.

### 9.3 `auto`, determinism and jobs

The engine initially uses `Auto`; explicit CPU remains available. Native
geometric, wave/image, radiometry, ghost/stray and candidate adapters share one
operation catalogue, including glass substitution. A native adapter is a route,
not a blanket precision certificate: offline suites establish accuracy, and an
analysis whose device result cannot meet its contract has no GPU path (§9.2).

`auto` applies the cost model of §9.2: accuracy exclusions run on the
reference on any device; predictions are scoped to the calibrated device
class. A device without a calibration table uses qualified Metal; requests
do not time-race backends. Routes that end on
the reference are settled before the device opens. Cold first calls and warm reuse are measured separately. A CPU choice by cost is deliberate and reported without a
warning. Small serial host control, I/O and final result packing remain
host work. Kernel/primitive-only measurements do not set complete-job
choices.

An explicit Metal request bypasses the cost model, including the accuracy
exclusions and the division rule (a stray run that gives the device no
source is then reported as the CPU reference it is, with a warning). For an
analysis without a GPU path it fails, or with `allow_backend_fallback` runs
on the reference with a warning.
Preserve existing unavailable-backend/`allow_backend_fallback` behaviour.
Unsupported work, resource or device failure produce
a reported reference fallback; cancellation never triggers fallback. A
partial solver fallback names stages/candidates, counts polished rays and
records why. `RunInfo` and the backend report retain requested/used backend,
precision, seed, base/derived hashes, device, build/probe,
effective algorithm/plan versions, cost decision and stage timings. Do not
label a GPU computation f64 just because its public buffer widens to f64.

For fixed inputs, seed, device, kernel build and plan, outputs are bitwise
repeatable; scheduling, worker count, chunk size and unrelated jobs cannot
change them. The host/device divisions keep this: they assign work by fixed
rules of the request (split-aiming tiles by generation and share, stray
source chunks by the share, ghost path groups by step order, candidate rays
under device aiming by the device iterations of earlier batches), never by
which side finishes first. Cross-device/build agreement for transforms/transcendentals is
within the numerical gates, not a promise of bit identity. Version and record
plan choices that affect operation order. Stable compaction preserves IDs;
adaptive wave/rim/refinement and candidate decisions use fixed tie rules.
Time-limited or manually stopped searches are reproducible only for the
recorded completed prefix; elapsed-time stopping is not deterministic.

**Route options.** Six options select another division than the default,
mostly a GPU-heavier one (`mklight_engine::RouteOptions`). The engine
configuration holds the defaults of requests that carry none
(`EngineConfig::route`, the `route` object of the C ABI's and Python's
engine configuration); a request carries its own in `AnalysisRequest::route`
(`"route"` in the C ABI's request JSON, `route=` in Python,
`--route KEY=VALUE` in the CLI), which replace the engine's as a whole: a
key the request leaves out takes its default below. An unknown key or name,
or a share outside [0, 1], fails the request when it is submitted. The
options are part of the key of a stored result, and reach the solver that
divides the work as typed parameters (`mklight_analysis::gpu::Route`,
`mklight_stray::gpu::TransportOptions`, the evaluator's `with_host_aim`):
two jobs of one process can run different divisions.

| Key | Values (default first) | Moves to the GPU |
|---|---|---|
| `aiming` | `host-table`, `table`, `device` | every pupil aim of the geometric adapters (`table`: device Newton corrections from host seed tables; `device`: device chief-seeded solve) |
| `aiming_share` | `null` (0.35), number in [0, 1] | under `host-table`, the device's share of split generations (0 aims on the host alone) |
| `fieldmaps` | `host-chief`, `device-chief` | the chief solves of `chief_ray_angle`, `pupil_aberration` and `grid_distortion` |
| `stray_share` | `null` (the scene rule), number in [0, 1] | the device's share of the stray source chunks (1: the first 64 events of every stray source) |
| `stray_sag` | `f32-scan`, `f32-grazing` | — (hands oblique and grazing extended-sag rays back to the CPU, which cuts device time on aspheric lenses) |
| `solver_aiming` | `by-batch`, `host`, `device` | every solver batch's pupil aims (`device`; `host` pins every batch to the host; `optimize`, `tolerance`, `glass_substitution`, `merit_evaluation`) |

Each option was measured on the M2 Pro against its default, is as accurate
(the same contracts, or Monte Carlo agreement for stray light) and is kept
because a GPU with more cores per CPU core (Max, Ultra) may win with it.
The run evidence names the division that ran (`run_info.settings`:
`aiming.design` and `aiming.device_share`, `fieldmaps_variant`, the stray
`gpu_share`, `sag_mode` and `resident_options`; the solvers'
`backend.aiming`). The options are the hook for a future feature that lets
users measure the routes on their machine and choose them; until then
`auto` and explicit requests use the defaults unless an option is set. The
device paths that other solvers reach inside a job (the distortion grid of
`image_simulation`, the MTF operands of the solvers) run the default
division.

Every long API takes `JobContext`. Use bounded command buffers/tiles with
cancellation checks before encoding and after completion. Target ≤50 ms
between checks on qualification hardware, measure worst cases including
Huygens and split trees, and reduce tile size if needed. In-flight Metal
commands finish before leases recycle; cancelled jobs discard outputs.
Monotone progress belongs to the outer job: allocate stage subranges,
aggregate completed logical tiles, reserve a tail for adaptive work, and
emit 1 only after successful final assembly. Optimiser **Stop**
(`Engine::stop_job`, the job's stop token) still returns its best completed
state; **Cancel** discards the result. Neither changes
the existing solver semantics to a partial success by accident.

### 9.4 Accuracy and qualification

Accuracy qualification is offline, per operation chain and tested settings/domain.
No snapshot edit or candidate evaluation runs a reference comparison. Retain
all geometry, complete-grid, CPU-f32 and f64 regression tests, with the §5.2
one-third sampled safety margin as test evidence rather than a runtime gate.
Device status masks, finite outputs and kernel conservation summaries remain
cheap runtime guards. Unresolved aiming and unsupported features keep explicit
reference fallbacks; cancellation yields no replacement result.

The following budgets remain offline acceptance criteria. Final reported
optimisation best designs and glass finalists are f64-scored once after the
device search; tolerance reports device scores.

| Quantity / kernel | Acceptance against independent CPU f64 |
|---|---|
| G: geometry and statuses | §7: positions ≤2e-5 mm on each surface, directions ≤2e-6; identical survival except the 1e-5 mm aperture band. Count band cases and all failure classes; never compare only common survivors. Existing TIR-band diagnostics remain explicit, not a blanket status exemption. |
| OPL: aiming/OPD/phase | Honour the run's aim-stop policy (CONVENTIONS §7: the requested residual, or the conditioned stop derived from the propagated error, never looser than its allocation); OPD ≤5e-3 waves. Internal OPL/sphere target ≈1e-7 mm on 100+ mm paths (≈1.8e-4 waves at 0.55 µm); this is tighter than the external λ/200 budget (2.75e-6 mm there). Double-float must cover geometry and reference construction as well as the path sum. |
| R: centroids/RMS/MTF | Position budget for centroids; differences of two positions ≤4e-5 mm. RMS ≤max(0.005×reference,2e-5 mm); MTF ≤2e-3 absolute, including direct/FFT/through-focus and detector multiplication. |
| T: FFT/chirp/affine transforms | Proposed gates: relative L2 complex error ≤1e-4 and max absolute error ≤1e-4×reference L1 input norm (with declared transform scaling); inverse round-trip/Parseval ≤1e-4 relative. Small direct-DFT, skew-MFT and non-power-of-two known answers supplement large f64 comparisons. Final optical MTF still meets §7. |
| E: PSF/EE/LSF/image | Proposed gates on matched sampling: integrated energy/flux ≤1e-3 relative, normalised PSF/image L1 error ≤2e-3, EE/ensquared fractions and Strehl ≤2e-3 absolute, LSF integral ≤1e-3. Centroids use G; r50/r80/r90 compare the energy curve and ≤one radial sample interval, tightening with sampling. Include Airy, annulus, tilt, defocus, Parseval and convolution conservation. |
| E: coating, polarisation and ghosts | Retain GHOST §6 weight limit 1e-5 relative; proposed zero-safe comparison `abs(error) ≤1e-5×abs(reference)+1e-10×incident_weight`. Per-interface R+T+A residual ≤1e-6, integrated ledger ≤1e-5 of emitted power. Jones/PRT relative norm ≤1e-4 (absolute 1e-7 of incident field near zero), D/intensity ≤1e-4 absolute, retardance ≤1e-3 rad where conditioned; otherwise reported f64 fallback, preserving undefined-retarder rules. Ghost map L1/flux ≤1e-3 of incident-normalised reference flux; count pixel-edge migrations and also test fixed-hit binning separately. |
| E: stray Monte Carlo | Identical raw counters and controlled-path tests first; same-sample f64 transport uses the same uniform mapping. Ledger ≤1e-5 of emitted power. Ensemble class flux/maps/moments agree within 5 combined standard errors plus deterministic quadrature error measured by refinement; test rare signatures and split-child correlations, not only total flux. Sampling error cannot excuse a biased ledger or wrong path ownership. |
| D: RI/pupil maps/curvature | Proposed gates: RI ≤1e-3 absolute, CRA ≤2e-4 degrees, pupil magnification ≤1e-3 relative or 1e-5 absolute; curvature focus ≤1e-3×(1+abs(reference_mm)) mm, cross-checked by close rays. Double-float pencil/Jacobian arithmetic; singular/undefined values match the reference. |
| D: material/first-order batches | Proposed index error ≤1e-10 absolute (1e-7 mm over a 1000 mm path); first-order lengths ≤1e-6 relative or 2e-5 mm absolute. Check thermal/dispersion offsets and solved constraints against f64, and resulting rays/OPD against §7; ill-conditioned afocal/solve cases preserve undefined values or fall back. |
| D: fits, FD and optimisation | Proposed reconstructed Zernike OPD/RMS ≤5e-3 waves; coefficients and rank checked against f64 QR with condition-aware residual tests. Scaled FD column error ≤1e-3 vs f64 Richardson (column norm floor 1e-8 in scaled units), plus analytic thick-lens derivative and step sweeps. Evaluator-specific FD steps: forward 1e-6 for device ray operands and 1e-5 for wavefront operands, central the reference's (OPTIMIZATION.md, "Step size"). f64 final merit within 1% or 1e-8 absolute of device merit and same feasibility classification outside the recorded constraint uncertainty band; ambiguity gets a reference decision. |

New E/T/D numbers are release gates to demonstrate, not measured claims.
Compare identical settings/sampling/methods: the CPU's direct-versus-FFT MTF
discrepancy is not GPU error. Keep wave sampling, truncation, conditioning
and root-search warnings; increasing precision does not fix undersampling.
For insufficient margin use double-float, a better-conditioned formulation
or the reported reference fallback, never loosen the budget to pass.

**Contracts.** The offline comparators (`metal_accuracy`, `aim_stop_qualify`,
the `project_bench` accuracy columns; `crates/mklight-bench/src/
comparison.rs`) judge every result field by its **quantity class**, never by
a tolerance guessed from its name. One table (`FIELDS`) maps every numeric
field and buffer of every compared analysis to a class; a field it does not
map fails loudly, and a test checks the table against the declared result
schemas. Deterministic classes gate at one third of their budget; the
statistical classes gate at their stated confidence.

| class | budget | source | for |
|---|---|---|---|
| position | 2e-5 mm | §7 position | hit points, centroids, chief/image positions, foci, lengths |
| position difference | 4e-5 mm | R | geometric spot radii (max distance between positions) |
| direction | 2e-6 | §7 direction | direction cosines, NA, `sin_max` |
| frequency, F-number | 2e-6 relative | §7 direction: `f = n sinθ/λ`, so δf/f = δsinθ/(n sinθ) ≥ 2e-6 | cutoffs, frequency axes, Nyquist, working F/# |
| waves | 5e-3 waves | §7 OPD | OPD, Zernike/Seidel terms, wavefront RMS/PV, OPD fans |
| modulation | 2e-3 | §7 MTF, R, E | MTF, Strehl, encircled/ensquared fractions |
| RMS | max(0.5 %, 2e-5 mm) | §7 RMS | RMS spot and moment radii |
| quantile | min(2e-5 mm, one radial sample) | E | r50/r80/r90 |
| angle, orientation | 2e-4° | D | chief-ray angles; Seidel axes modulo their period, undefined below 1e-3 waves |
| magnification | 1e-3 relative or 1e-5 | D | pupil magnification and its variation |
| illumination | 1e-3 | D | relative illumination |
| relative 1e-3 | 1e-3 relative | D, E | projected solid angle, integrated PSF energy, diffraction peak |
| radiometric | 1e-5 relative + 1e-10 | E, GHOST §6 | fluxes, transmissions, ghost peaks |
| ledger | 1e-5 of emitted + 1e-10 | E stray | stray energy report |
| polarimetric, retardance | 1e-4; 1e-3 rad | E | diattenuation, intensity; retardance; fast axis only where the retarder is defined |
| Jones | 1e-4‖J‖ + 1e-7/√2 | E | Jones matrices |
| image | L1 2e-3, energy 1e-3 | E | PSF, LSF, irradiance and simulated images |
| outline, loop | Hausdorff ≤ 2e-5 mm (2e-5 in pupil units) | §7 position | footprint hulls and pupil rim loops, never by vertex count |
| exact | equal | survival | counts, indices, statuses, echoed inputs |
| count ratio | 1e-12 | survival | vignetted/unvignetted fractions: one ray moves them ≥ 1e-9 |
| statistical | 5·√(a + b) | E stray | Monte-Carlo counts (events, hits) under roulette |
| Monte Carlo | 5 combined standard errors | E stray | class fluxes and moments, with their own standard errors; ranked signatures matched by identity, a tail signature only within 5σ of the other list's cut |
| merit | 1 % or 1e-8 | D | solver merits (violations generic; feasibility exact) |
| generic | 2e-5 | (unchanged) | dimensionless geometry without a derived budget: distortion %, area and vignetting fractions, normalised pupil coordinates, operand values |
| diagnostic | 1e-3 relative | — | fit condition numbers (not an optical output) |
| sampling | recorded | — | pupil/image grid sizes and spacings a backend chooses |

Every class is at least as strict as the §7 budget of its physical quantity
(`mklight-bench/tests/contract.rs`). Data on a sampling axis or grid are
compared at the
**same physical coordinates**: index by index where both results sampled the
same coordinates (within the axis class), otherwise the denser result is
resampled onto the sparser one's coordinates (linear on 1-D axes such as
frequencies, focus planes, radii, field and pupil coordinates; bilinear on
the regular pupil-map and PSF image grids), so a backend's different grid
is a recorded sampling choice and only a physical difference fails.

**Tiers.** Every result carries `run_info.tier` and `run_info.aim_stop`
(`AnalysisRequest::{tier, aim_stop}`; the CLI's `--tier`/`--aim-stop`; the
FFI request JSON). The budgets above are the **design** tier, the default;
the engine's default aim stop is `conditioned` (CONVENTIONS §7), qualified
under the contracts above on the CPU and Metal, and an explicit aiming
tolerance in the system is always honoured exactly.
The **preview** tier is opt-in and never silent: it exists for responsive
exploration and is not evidence for a design decision. Its contract against
the design tier of the same request is

| quantity | preview budget |
|---|---|
| ray/centroid position at the image | ≤ min(0.1 pixel, 0.05 λN) |
| MTF (same physical frequencies) | ≤ 0.01 absolute |
| OPD RMS (wave previews) | ≤ 0.02 waves (≈1.6 % Strehl) |

achieved by cheaper sampling and aiming, both recorded in
`run_info.settings`: analyses declare preview defaults (`x-preview` in the
settings schema: half the wave lattice rays per axis, 33 instead of 65, for
the MTF, MTF-vs-field and wavefront-map analyses (the PSF-family image
windows grow with the geometric blur of a coarser lattice, so those keep
the design lattice); half the dense square-pattern density, 16 instead of
32; 11 field or focus samples instead of 21, 6 instead of 11 for
`mtf_vs_field`) that replace the
design defaults only for settings the request leaves out, and the
conditioned aim stop uses the preview allocation (CONVENTIONS §7). Requested
tolerances, survival-mask rules and the design budgets are untouched by the
tier; a preview result is compared to the design result with the gates
above, not to the CPU f64 reference with the design gates.

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


## 10. Environment variables

Every variable the library code reads, and the families the test harnesses
read. All are optional; the defaults are the documented behaviour.

**Runtime** (read by library code in a running process):

| Variable | Read in | Effect |
|---|---|---|
| `MKLIGHT_DISABLE_GPU=1` | `mklight-engine/src/gpu.rs` | no GPU provider opens its device; requests run on the CPU reference with that reason |
| `MKLIGHT_COST_CALIBRATION=<file>` | `mklight-engine/src/analysis/cost/model.rs` | loads another calibration table at engine start (§9.2, "Calibration") |
| `MKLIGHT_COST_REFINEMENT=0` | `mklight-engine/src/analysis/cost/mod.rs` | turns the cost model's online refinement off |
| `MKLIGHT_METAL_IN_FLIGHT=<n>` | `mklight-metal/src/context.rs` | committed command buffers a worker keeps in flight (default 2, at most 8; 0 completes each before the next) |
| `MKLIGHT_METAL_ADMIT_PLANS=<n>`, `MKLIGHT_METAL_ADMIT_MIB=<n>` | `mklight-metal/src/admission.rs` | resident plans admitted at once (default 4) and their declared storage (default 1024 MiB) |

**Diagnostics** (read by library code; they print, and change no result):

| Variable | Read in | Effect |
|---|---|---|
| `MKLIGHT_METAL_TRACE_PLANS=1` | `mklight-metal/src/backend.rs` | prints one line per resident plan to stderr |
| `MKLIGHT_METAL_TRACE_AIMS=1` | `mklight-metal/src/aiming/mod.rs` | prints every aim generation's convergence statistics to stderr |

**Build** (read by `mklight-metal/build.rs`): `MKLIGHT_METAL_TOOLCHAIN_BIN`
(directory of the Metal compiler) and `MKLIGHT_METAL_SKIP_OFFLINE=1` (skip
the offline shader compile, so the library is compiled from source at run
time), §6.

**Harnesses** (read only by tests and benchmarks; the files define their
meaning). The engine's measurement and qualification harnesses are the crate
`mklight-bench`; its `README.md` lists each one's command and variables.

| Family | Read in |
|---|---|
| `MKLIGHT_BENCH_*` | `mklight-bench/tests/project_bench.rs`, `cpu_trace_bench.rs` (`MKLIGHT_BENCH_PROJECT` also in `cost_calibration.rs`) |
| `MKLIGHT_COST_{CALIBRATION_OUT, DESIGNS, MODES, FILTER, REPS, RECORDS, REFIT, DEVICE_OPEN_MS}` | `mklight-bench/tests/cost_calibration.rs` |
| `MKLIGHT_QUALIFY_*` | `mklight-bench/tests/aim_stop_qualify.rs` |
| `MKLIGHT_IDENTITY_*` | `mklight-bench/tests/buffer_identity.rs` |
| `MKLIGHT_STUDY_*` | `mklight-bench/tests/aim_precision_study.rs` |
| `MKLIGHT_SEED_PROJECT` | `mklight-bench/tests/seeded_jacobian.rs` |
| `MKLIGHT_P5_{CASE, SAMPLE}` | `mklight-analysis/tests/p5_bench.rs` |
| `MKLIGHT_P10_BENCH_{FILTER, ITERS}` | `mklight-stray/tests/p10_support/transport_bench.rs` |
| `MKLIGHT_UPDATE_GOLDEN`, `MKLIGHT_UPDATE_RESULTS_DOC` | `mklight-analysis/tests/golden.rs`, `mklight-engine/tests/results_doc.rs` (rewrite the stored golden results, the generated block of RESULTS.md) |
| `MKLIGHT_RENDER_DIR` | `apps/macos/Tests/MKLightFFITests/` (render tests) |
| unprefixed selectors: `RESIDENT_BENCH_*`, `P11_{SAMPLE, ANALYSES, RUNTIME_SOURCE}`, `P1_BENCH_*`, `P6_*`, `P7_*`, `P9_SAMPLE`, `FUSION_BENCH`, `SPLIT_SAMPLE` | `mklight-optim/tests/resident_bench.rs`, `mklight-engine/tests/p11_routes.rs`, `mklight-metal/tests/{p1_bench, p7_bench, p0_zerocopy}.rs`, `mklight-wave/tests/{p6_native, p6_bench, split_designs}.rs`, `mklight-gpu/tests/p7_support.rs`, `mklight-ghost/tests/p9_bench.rs` |
