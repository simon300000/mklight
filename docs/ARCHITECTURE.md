# MKLight Architecture

MKLight is a GPU-accelerated optical design and optimization package for
sequential imaging systems (camera lenses, relays, telescopes), with ghost,
stray-light and wave-optics solvers that share one physical model.

The guiding rule: **unify physics, data and correctness — not UI frameworks or
GPU implementation details.**

```
 ┌──────────────────────────┐   ┌──────────────────────┐   ┌──────────────┐
 │ macOS app (Swift/SwiftUI │   │ Windows/Linux apps   │   │ Python / CLI │
 │ AppKit, Metal viewport)  │   │ (framework TBD)      │   │  (headless)  │
 └────────────┬─────────────┘   └──────────┬───────────┘   └──────┬───────┘
              │  C ABI (mklight-ffi, JSON + binary buffers)        │ pyo3 / Rust API
              └───────────────┬────────────┴───────────────────────┘
                     ┌────────▼─────────┐
                     │  mklight-engine  │  sessions, projects, job scheduler,
                     │                  │  analysis dispatch, backend choice
                     └────────┬─────────┘
   ┌───────────────┬──────────┼────────────┬──────────────┬──────────────┐
   │ mklight-      │ mklight- │ mklight-   │ mklight-     │ mklight-     │
   │ analysis      │ wave     │ ghost      │ stray        │ optim        │
   │ (geometric)   │ (PSF/MTF)│ (explicit  │ (non-seq.    │ (merit, LM,  │
   │               │          │  ghosts)   │  Monte Carlo)│  global, DFO)│
   └───────┬───────┴────┬─────┴─────┬──────┴──────┬───────┴──────┬───────┘
           └────────────┴─────┬─────┴─────────────┴──────────────┘
                     ┌────────▼─────────┐        ┌────────────────────────┐
                     │  mklight-trace   │        │  mklight-gpu           │
                     │  CPU f64         │◄──────►│  backend trait, packed │
                     │  REFERENCE       │ cross- │  f32 layout, CPU-f32   │
                     │  sequential +    │ valid- │  emulation backend     │
                     │  paraxial        │ ation  └──────────┬─────────────┘
                     └────────┬─────────┘          ┌───────▼────────┐
                     ┌────────▼─────────┐          │ mklight-metal  │ (macOS)
                     │  mklight-core    │          │ (vulkan/cuda   │  later)
                     │  model, units,   │          └────────────────┘
                     │  materials,      │
                     │  coatings, jobs  │
                     └──────────────────┘
```

## Crates (`crates/`)

| crate | role |
|---|---|
| `mklight-core` | Units, math, the **single physical model** (`OpticalSystem`: surfaces, materials, coatings, apertures, fields, wavelengths, mechanics, sensor), material library and dispersion, thin-film coatings, parameter addressing (`ParamPath`), immutable `Snapshot`s with content hashes, `ResolvedSystem` (model + materials → numbers a tracer needs), the job framework (cancellation, progress, run metadata), deterministic RNG. No solvers. |
| `mklight-trace` | The **CPU double-precision reference** sequential tracer: surface intersection, refraction/reflection, apertures/vignetting, ray aiming, pupil/field normalisation, paraxial (y-nu) trace and first-order properties, auto semi-diameters. Every GPU result is validated against this crate. |
| `mklight-analysis` | Geometric analyses on top of `trace`: spot diagrams, ray/OPD fans, field curvature & distortion, Seidel sums, chief-ray data, relative illumination, longitudinal/lateral colour. |
| `mklight-wave` | Wave optics: OPD maps on the exit-pupil reference sphere, Zernike fitting, FFT PSF, Huygens-checked PSF, MTF (FFT and pupil autocorrelation), Strehl, encircled energy. |
| `mklight-ghost` | Explicit ghost-path tracing: enumerates even-order specular reflection sequences inside the sequential train, weighted by coating reflectance; ghost irradiance and focus maps. |
| `mklight-stray` | General non-sequential stray-light transport: Monte-Carlo with BSDFs on optical and mechanical geometry; path classification with the ghost solver to prevent double counting. |
| `mklight-optim` | Merit functions (weighted residuals, extensible operand registry), variables and bounds, explicit constraints, LM/trust-region least squares, global/derivative-free/discrete methods, batched evaluation, finite-difference derivative validation, tolerancing sensitivity and Monte Carlo. |
| `mklight-gpu` | Narrow, backend-agnostic compute interface (`ComputeBackend`), the packed f32 system layout shared bit-for-bit by all GPU backends, and a CPU-f32 emulation backend used to separate *algorithm* error from *precision* error. |
| `mklight-metal` | Metal backend: MSL kernels (precompiled `.metallib`, runtime-source fallback), command scheduling, shared-memory buffers. |
| `mklight-engine` | Facade used by every front end: `Session` (project + undo-free immutable snapshots), the async job scheduler, analysis dispatch, backend selection, project file I/O. |
| `mklight-ffi` | C ABI for native apps. Structured data crosses as versioned JSON; bulk data as typed binary buffers. |
| `mklight-py` | Python bindings (pyo3/maturin). |
| `mklight-cli` | Headless command line (`mklight analyze`, `mklight optimize`, `mklight validate`). |

## Apps (`apps/`)

* `apps/macos` — Swift package: `MKLightKit` (Swift wrapper over the C ABI,
  `async` jobs, `Sendable` snapshots) and `MKLight` (SwiftUI/AppKit app with a
  Metal-backed viewport). Built by `scripts/build-macos.sh` into `MKLight.app`.

## State model

* **Simulation state is independent of UI state.** The engine owns
  `OpticalSystem` values; the UI holds a document that edits them and keeps its
  own view state (selection, panes, camera) separately and never inside the
  model.
* Every edit produces a new immutable `Snapshot` (`Arc<OpticalSystem>` + BLAKE3
  content hash of its canonical JSON). Analyses run against a snapshot, never
  against live mutable state, so the UI can keep editing while jobs run.
* A job = `(snapshot, settings, backend, precision, seed)`. Every result carries
  a `RunInfo` with the engine version, snapshot hash, settings JSON, backend,
  seed and timings, which makes it reproducible and lets the UI tell when a
  result is stale.
* Jobs run on a worker pool, report progress (fraction + stage text), and check
  a cancellation token at bounded intervals. A newer request for the same
  analysis view supersedes and cancels the older one.

## Solvers and double counting

All four solvers use the same surfaces, materials, coatings, sign conventions
and radiometric units. Light reaching the detector is partitioned into disjoint
**path classes**:

| class | definition | owned by |
|---|---|---|
| `Primary` | the designed sequential path, transmitted at every optical surface | sequential / wave |
| `Ghost(k)` | purely specular paths through the optical surfaces with ≥2 reflections | ghost |
| `Scatter` | any path with at least one scattering event | stray |
| `Mechanical` | any path touching non-optical geometry without scattering (specular reflection off barrels etc.) | stray |

The stray-light solver tags each Monte-Carlo path and, when run alongside the
ghost solver, drops the purely specular optical-surface classes. The ghost
solver uses the same coating reflectance `R(λ, θ, pol)` that the primary path
uses for its transmittance `T`, so `T + R + A = 1` holds at every interface
(tested).

## Precision on the GPU

Apple GPUs have no FP64. GPU kernels are FP32, designed so that their error
stays inside an explicit budget (see `docs/CONVENTIONS.md` §Precision):

* rays are carried in each surface's local frame, never in large global coordinates;
* intersections use cancellation-free closed forms, then Newton refinement for aspheres;
* optical path length is accumulated in double-float (`float2` two-sum)
  arithmetic, *relative* to the chief ray;
* reductions use compensated (Kahan/Neumaier) or pairwise summation.

The CPU f64 tracer is the reference. CI-style tests compare reference ↔ CPU-f32
emulation ↔ Metal on a fixed set of designs, with per-quantity tolerances.

## Platform backends

`mklight-gpu::ComputeBackend` is the only interface a solver sees. Metal is
implemented now; Vulkan (AMD/Intel) and CUDA/OptiX (NVIDIA) slot in as further
crates that implement the same trait over the same packed layout. Hardware ray
tracing (Metal acceleration structures / OptiX / Vulkan RT) is reserved for
spatial searches in the non-sequential solver.
