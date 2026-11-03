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
| `mklight-trace` polarisation & radiometry | Polarisation ray tracing (3×3 PRT matrices from complex coating amplitudes, Jones pupils, diattenuation/retardance, vector pupils for wave optics) and primary-path radiometry (per-ray throughput, system transmission vs field/wavelength, polarisation pupil maps). See `docs/POLARIZATION.md`. |
| `mklight-analysis` | Geometric analyses on top of `trace`: spot diagrams, ray/OPD fans, field curvature & distortion, Seidel sums, chief-ray data, relative illumination, longitudinal/lateral colour. |
| `mklight-wave` | Wave optics: OPD maps on the exit-pupil reference sphere, Zernike fitting, FFT PSF, Huygens-checked PSF, MTF (FFT and pupil autocorrelation), Strehl, encircled energy. |
| `mklight-ghost` | Explicit ghost-path tracing: enumerates even-order specular reflection sequences inside the sequential train, weighted by coating reflectance; ghost irradiance and focus maps. |
| `mklight-stray` | General non-sequential stray-light transport: Monte-Carlo with BSDFs on optical and mechanical geometry; path classification with the ghost solver to prevent double counting. |
| `mklight-optim` | Merit functions (weighted residuals, extensible operand registry), variables and bounds, explicit constraints, LM/trust-region least squares, global/derivative-free/discrete methods, batched evaluation, finite-difference derivative validation, tolerancing sensitivity and Monte Carlo. |
| `mklight-gpu` | Narrow, backend-agnostic compute interface (`ComputeBackend`), the packed f32 system layout shared bit-for-bit by all GPU backends, and a CPU-f32 emulation backend used to separate *algorithm* error from *precision* error. |
| `mklight-metal` | Metal backend: MSL kernels (precompiled `.metallib`, runtime-source fallback), command scheduling, shared-memory buffers. |
| `mklight-engine` | Facade used by every front end: `Session` (project + undo-free immutable snapshots), the async job scheduler, analysis dispatch, backend selection, project file I/O. |
| `mklight-ffi` | C ABI for native apps. Structured data crosses as versioned JSON; bulk data as typed binary buffers. |
| `mklight-py` | Python bindings (pyo3/maturin, `import mklight`) over the engine facade, plus low-level reference tracing; see `docs/PYTHON.md`. |
| `mklight-cli` | Headless command line `mklight` over the engine facade: projects (`new`, `validate`, `show`, `migrate`, `set`), catalogs, `analyses`/`run` over the analysis registry, reference `trace` and `paraxial`; JSON or tables, documented exit codes (`docs/CLI.md`). |

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

All solvers use the same surfaces, materials, coatings, sign conventions
and radiometric units. Light reaching the detector is partitioned into five
disjoint **path classes**. The stray solver assigns them from a path's
history, first match wins in the order Scatter, Mechanical, Sneak (outside
an aperture), Ghost, Primary, Sneak (any other):

| class | definition | owned by |
|---|---|---|
| `Scatter` | at least one scattering (BSDF) event, on any surface | stray |
| `Mechanical` | no scatter, but a specular interaction with non-optical geometry (lens edges and flats, mounts, stop plates, barrels, baffles) | stray |
| `Sneak` | purely specular on optical surfaces, but crossing or reflecting off one outside its clipping aperture, or out of the sequential order with fewer than two ghost reflections (e.g. a single reflection off a tilted window); the `Irregular` signatures of CONVENTIONS §10 | stray |
| `Ghost(k)` | purely specular, inside every clipping aperture, `k ≥ 2` reflections at refracting interfaces (or the sensor), in the sequential order the ghost reflections imply | ghost solver, for the paths its run covered; otherwise stray |
| `Primary` | the designed path: the designed interaction (transmission, or reflection at a mirror) at every optical surface in order, inside every clipping aperture, not obstructed by other geometry | sequential / wave |

The precise rule, with path signatures, is CONVENTIONS §10; the class
tables of `docs/STRAY.md` and `docs/GHOST.md` §2 implement it. When the
stray solver runs alongside the ghost solver it drops the `Ghost` paths
the ghost run covered (`exclude_ghost_class` with the run's
`ghost_coverage`) and keeps the other ghost orders, so

```text
detected = Primary (sequential) + Ghost (ghost solver) + Scatter + Mechanical + Sneak (stray)
```

with nothing counted twice (tested on a coated singlet and Cooke triplet,
`mklight-stray/tests/partition.rs`). This needs both solvers to clip at
identical apertures: the stray scene takes its clear apertures from
`mklight_ghost::clip_apertures`, the ghost solver's own definition, and
`StraySystem::check_ghost_coverage` (or `ghost_coverage` in the stray
analysis settings) refuses a ghost run that clipped differently; the
comparison is exact, so a ghost model built for a stray scene with
non-default sizing settings takes the scene's semi-diameters
(`GhostModel::with_semi_diameters`). Ghost orders the ghost run did not
trace (beyond its `max_reflections`, or filtered out) stay in the stray
total, as CONVENTIONS §10 assigns them (tested with a run to 2
reflections on a Cooke triplet with 30 % reflectors); without the coverage
record the stray solver can only drop the whole class, and those orders
are counted by neither solver. A purely sequential `Primary` overcounts
light that non-sequential geometry obstructs (use the obstruction-checked
reference, `docs/STRAY.md`). The ghost solver uses the same coating reflectance
`R(λ, θ, pol)` that the primary path uses for its transmittance `T`, so
`T + R + A = 1` holds at every interface (tested).

## Bulk results

JSON carries only an analysis' metadata; bulk arrays (ray points, maps,
images, curves) travel as typed `Buffer`s and never pass through JSON on the
way. Solvers fill the final arrays once: device outputs are read in place
from shared (unified-memory) storage and widened straight into the result
layout, and the engine moves solver vectors into buffers instead of copying
them. Spot statistics without `include_points` allocate no point output; the
CPU reference traces one field at a time, so only that field's rays are
alive. Buffer values follow the JSON conventions of the documented result
(canonical NaN; infinities only where the schema allows them), so the typed
route is byte-identical to the former JSON route. The C ABI hands each
buffer's allocation to the caller without a copy.

## Analysis graph

An edit is usually followed by several graphs redrawing the same snapshot
(layout, spot, fans, MTF, PSF, field curvature, …). Instead of each analysis
preparing fields and tracing pupils on its own, the engine keeps a shared,
content-addressed dependency graph under the analysis requests
(`mklight-engine` `graph`, `mklight_core::memo`):

| node | content key | shared by |
|---|---|---|
| resolved system (material resolution) | snapshot, library generation | session cache |
| `tracer`, `field_tracer` | snapshot, library (field selection) | geometric, map and radiometry analyses: first order, prepared fields with chief rays and Jacobians, host seeds of the Metal path |
| `configured` | snapshot, library, configuration | wave analyses: configuration, solves, first order, prepared fields |
| `pupil_lattice` | configured snapshot, tracer settings, prepared field, wavelength, lattice settings, library | CPU f64 PSF, MTF, LSF, encircled energy, through-focus, wavefront, MTF vs field: rays, rim, reference sphere, masks, automatic pupil grid |
| `resampled_pupil` | lattice, grid, supersampling | every consumer of the same grid (decided grid and window refinements) |
| `result` | snapshot, library, analysis, normalised settings, backend selection | repeated requests (refresh, undo); CPU-reference results only |

Keys digest every input a node reads plus an algorithm version, so a hit is
bit-identical to a rebuild and an edit misses exactly the nodes that depend
on what changed; nothing stale can be returned. Methods are different nodes
(direct and FFT MTF never substitute for each other); sampling decisions such
as the automatic pupil grid are computed once per lattice from host f64 data.
The graph has one byte budget (`EngineConfig::analysis_graph_bytes`, 0
disables it) with GreedyDual-Size retention by rebuild cost per byte.
Concurrent requests claim missing nodes and wait for nodes another job is
building (outside the rayon pool), so a queued refresh traces each pupil once;
failed or cancelled builds publish nothing. Metal keeps its own resident
caches (aiming rows, seed tables, transform plans); transform recipes, tables
and chirp spectra are content-addressed there and survive edits.

## Structural validation

`OpticalSystem::validation_issues` lists every structural problem of a
model with a stable code and the JSON path of the value
(CONVENTIONS §13); `validate` (and so every `Snapshot`) rejects the first.
Model JSON (and settings objects such as `SemiDiameterSettings` and
`AgfOptions`) rejects unknown keys, so a misspelt key is an error rather
than a silently ignored setting. Deliberately permissive: the project
envelope keeps unknown top-level keys (data added by a newer application at
the same schema version survives a round trip), `RunInfo` and other result
documents (output, read by front ends of other versions), and glass-catalog
JSON (`Glass`, `GlassCatalog`: data that other tools may annotate).

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
