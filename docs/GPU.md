# MKLight GPU Infrastructure

This document describes how solvers reach a GPU: the backend interface, the
packed data layout every backend shares, the f32 numerics strategy that keeps
GPU results inside the CONVENTIONS §7 error budget, and how results are
validated against the CPU f64 reference. Read `docs/CONVENTIONS.md` first.

| crate | role |
|---|---|
| `mklight-gpu` | `ComputeBackend` trait and job types, the packed layout (`include/mklight_gpu_types.h` + Rust mirror + packer), the double-float / utility-math / reduction **specifications** in Rust, and the CPU-f32 emulation backend |
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
  the backend instance that issued it.
* **Jobs** (`Job`) are batched, backend-neutral descriptions of work over host
  structure-of-arrays data: `PhiloxStream`, `PhiloxBlocks`, `DfOps`,
  `Reduce`, `SagProbe`, and `Trace` (the sequential ray-trace kernel, added
  when it is ported from `mklight-trace`). Each has a typed `JobOutput`.
  `Job::validate` performs the backend-independent input checks.
* **Asynchrony.** `submit` returns a `Pending<JobResult>` immediately. It can
  be awaited (`Future`), waited on (`wait`, `wait_timeout`) or polled, and
  cancelled (`Pending::cancel` or the `JobContext`'s token). A worker that dies
  without a result resolves it with an error, never a hang.
* **Cancellation and progress.** Backends split large jobs into command
  buffers (launches) of `chunk_items` work items, check the cancel token
  between them and report progress through the `JobContext`. A cancelled job
  returns `Err(Cancelled)`, never a partial result.
* **Stats.** `JobStats` reports device time (GPU timestamps), wall time and
  the number of command buffers.
* **Reproducibility.** `DeviceInfo::backend_id()` ("metal:Apple M2 Pro") and
  the `PrecisionDescriptor` (serialisable) go into `RunInfo` together with the
  seed. Philox streams and reductions are deterministic (see §3), so a
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
derives `bytemuck::Pod`). Any change bumps `MKL_LAYOUT_VERSION`; the header
carries a magic word and the version, and backends validate both on upload.

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

### 2.2 `MklPackedSurface` (304 bytes)

| field | meaning / units |
|---|---|
| `shape` | `MKL_SHAPE_{PLANE, CONIC, EVEN_ASPHERE, PARAXIAL, COORD_BREAK}` |
| `interaction` | `MKL_INTERACT_{REFRACT, REFLECT, DUMMY}` |
| `aperture`, `aperture_params[4]` | `MKL_APERTURE_*`; mm. A fixed semi-diameter becomes `CIRCULAR` with `MKL_SURF_APERTURE_FROM_SD`; an explicit aperture wins |
| `flags` | `HAS_TO_NEXT`, `AXIAL`, `IS_STOP`, `COATED`, `APERTURE_FROM_SD`, `OBJECT`, `IMAGE` |
| `curvature`, `conic` | 1/mm, dimensionless |
| `paraxial_power` | 1/f (1/mm) for ideal thin lenses |
| `asphere_inv_norm2`, `num_asphere`, `asphere[10]` | normalised even-asphere terms A4 … A22 (below) |
| `source_index` | surface index in the model |
| `to_next_rot[9]`, `to_next_origin[3]`, `to_next_origin_lo[3]` | transform into the next local frame (below) |
| `n_after[16]`, `alpha_after[16]` | index after the surface, internal absorption (1/mm), per packed wavelength |

**Aspheres.** The sag is `conic(r²) + ρ⁴ Σⱼ asphere[j] ρ²ʲ` with
`ρ² = r²·asphere_inv_norm2`. The packer stores `asphere[j] = A₂ⱼ₊₄ R²ʲ⁺⁴`
with `R = 2ᵉ` chosen near the surface's semi-diameter. Because `R` is a power
of two, the normalisation is exact (it changes only exponents), yet it keeps
A22-scale coefficients (~1e-35) out of the f32 subnormal range and keeps
Horner intermediates O(sag). Terms beyond A22, or coefficient sets that no
exponent can make representable, are rejected.

**Transforms** (CONVENTIONS §7.1). `to_next_origin` is the next vertex in this
surface's local frame and `to_next_rot` maps this frame's axes to the next
one's: `p_next = rot · (p − origin)`, `d_next = rot · d`. Subtracting before
rotating keeps the operand small near the next vertex (the `rot·p + t` form
cancels). `to_next_origin_lo` holds the f32 rounding residual of the origin,
so a 1000 mm air gap, whose f32 rounding alone is ~6e-5 mm (above the 2e-5 mm
position budget), can be applied in double-float. Relation to the core:
`origin = ResolvedSurface::to_next.origin`, `rot = to_next.rot` transposed.

**Exact-comparison sentinels.** Kernels may test packed parameters for exactly
`0.0f` (the packer writes an exact zero whenever the model value is zero). No
other float is compared for equality (CONVENTIONS §7.6).

### 2.3 Ray batches

`RayBatch` / `TraceResult` are SoA arrays: position and direction cosines in
the local frame of the start surface, optical path as double-float
(`opl_hi`, `opl_lo`), packed wavelength index, optional system index, and a
status word `code | surface << 8` (`MKL_RAY_OK, MISS, TIR, VIGNETTED,
NONFINITE, NOT_CONVERGED`). Ray generation (field/pupil mapping, ray aiming)
stays in the f64 reference code; rays are rounded to f32 once, on the start
surface's local frame.

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
| subnormals preserved | – (values are far from 1e-38) | **no, flushed to zero** |

`DfOps` and `Reduce` (and later every double-float kernel) are only offered
when `error_free_transforms()` holds.

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

### 3.5 Random numbers

Philox4x32-10 on the GPU is bit-identical to `mklight_core::rng` (raw words
for any counter/key, and the `(seed, stream)` stream layout of `Philox`). The
f32 uniform is `fmath::u32_to_unit_f32`: the core's `(k + ½)·2⁻²⁴` (top 24
bits) rounded toward zero to f32, which is exact for `k < 2²³` and half an f32
ulp lower otherwise. (`(k + ½)·2⁻²⁴` needs 25 bits for `k ≥ 2²³`; rounding it to
nearest could produce 1.0.) Every operation in the mapping is exact, so it is
identical on every device. Consumers that need the f64 uniform apply
`u32_to_unit_f64` to the raw words on the host.

## 4. Validation

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
  stated bound).
* Tests now in place, on this machine's GPU (skipped without a Metal device):
  Philox known answers and 65 536 random counter/key blocks, a 2¹⁸-block
  stream against the stateful core generator, 64-bit counter carry; two_sum
  and two_prod exact in f64 on the GPU for 65 536 random pairs (fast-math
  would zero the error terms); every df operation bit-identical to the
  emulation and inside its f64 error bound; 10⁶-value sums with the
  double-float GPU result at ≤ 1e-14 relative to the f64 sum while plain f32
  is not; ill-conditioned multi-channel sums; packed sag on the GPU against
  the emulation and the core's f64 sag; layout against the Metal compiler;
  the runtime-compiled library bit-identical to the precompiled one;
  chunking, progress, cancellation, handle checks and unsupported jobs.
* When the trace kernel lands, the cross-validation suite traces a fixed set
  of designs (singlets, a double Gauss, a Cooke triplet, a tilted/decentred
  system, a long-track telephoto, an asphere-heavy phone lens) through all
  three implementations and enforces the budget per quantity: positions,
  direction cosines, OPD, RMS spot, MTF and identical vignetting except
  within 1e-5 mm of an aperture edge. A design that fails the budget makes
  the backend return `Unsupported` for it.

## 5. Metal backend notes

* **Build.** `build.rs` compiles `shaders/*.metal` to `.air` and links one
  `mklight.metallib`, embedded with `include_bytes!`. It finds the compiler
  via `MKLIGHT_METAL_TOOLCHAIN_BIN`, then `xcrun -sdk macosx metal`, then the
  system-mounted Metal toolchain component (which `xcrun` does not always
  resolve). The include-inlined source is always embedded as a fallback;
  `MKLIGHT_METAL_SKIP_OFFLINE=1` forces the fallback for testing.
  `LibraryChoice::{Auto, Precompiled, RuntimeSource}` selects at run time.
* **Device and pipelines.** `DeviceSelector::{Default, NameContains,
  RegistryId}`. Pipelines are cached by kernel name + function-constant values
  (`mkl_fc_df_op` at index 0, `mkl_fc_reduce_mode` at index 1).
* **Memory.** `SharedBuffer<T: Pod>` wraps `MTLStorageModeShared` buffers
  (unified memory on Apple Silicon: uploads/readbacks are memcpys). The GPU
  only touches a buffer inside `MetalContext::run`, which borrows it until the
  command buffer has completed, so host and device never race.
* **Execution.** Each submission runs on a worker thread: per chunk it checks
  cancellation, encodes, commits, waits, checks `MTLCommandBufferStatus.error`
  (reported as `MkError::Backend` with the Metal error text), and adds
  `GPUEndTime − GPUStartTime` to the stats. Everything runs inside an
  autorelease pool.
* **Ray tracing hardware.** `DeviceInfo::ray_tracing_api` reports
  acceleration-structure support; `hardware_ray_tracing` is set for Apple9+
  GPUs (M3 and later). These are reserved for the non-sequential solver's
  spatial searches; sequential tracing uses analytic per-surface kernels.

## 6. Adding a backend (Vulkan, CUDA/OptiX)

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

## 7. Current limitations

* The sequential trace kernel and the GPU coating (thin-film) tables are not
  ported yet: `Job::Trace` is `Unsupported` on every backend, and
  `MKL_SURF_COATED` only marks coated surfaces.
* Job inputs and outputs are host arrays; device-resident buffers chained
  between jobs (e.g. trace → reduce without a readback) are a later
  optimisation that does not change the interface's semantics.
* Command buffers within a job run one after another (commit, wait); with
  the default 2²⁰ items per buffer the gap is negligible next to kernel time.
