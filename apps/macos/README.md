# MKLight for macOS

The native macOS front end: a SwiftUI/AppKit document app with a Metal
layout viewport, built around the headless Rust engine. The app never
computes optics. Every number it shows comes from an `EngineClient`.

```
../../scripts/build-macos.sh --engine-only   # Rust engine → Frameworks/libmklight_ffi.a
swift build                                  # debug build (links the engine when present)
swift test                                   # model, boundary, FFI integration and UI-logic tests
../../scripts/build-macos.sh                 # release build → build/macos/MKLight.app
```

The app runs on the Rust engine through `FFIEngine`. Toolchain and OS
requirements are in docs/APP.md "Platform"; the build integration, the debug
launch options, the binding, its thread safety and the tests are in
docs/APP.md "Building", "`FFIEngine`" and "Tests".

## Package layout

| target | contents |
|---|---|
| `MKLightModel` | Codable, `Sendable`, `Hashable` value types that mirror the engine's `OpticalSystem` JSON exactly: tagged enums, snake_case keys, explicit `null`s, `"inf"`/`"-inf"` thicknesses, curvature rather than radius, and every surface shape of `docs/SURFACES.md` with its `shape_coefficient`/`shape_param` parameter paths and term numbering (`CoefficientSet`, `ShapeParameter`, `XYTerm`, `ZernikeTerm`), checked against `extended_shapes.json`. It also includes structural validation (the engine's `validate` checks, verified case by case against engine verdicts in `validation_cases.json`; values JSON cannot carry are errors; what the mirror leaves to the engine is the validation of each configured system and the per-kind rules of solves), reference-preserving edits (surface insert/delete/stop, field and wavelength removal that renumbers index-addressed overrides), radius↔curvature display helpers, the `.mklight` project envelope (`ProjectFile`) and sample systems. It has no UI and no optics. |
| `MKLightEngine` | The engine boundary. It holds the `EngineClient` protocol, `ModelSnapshot`/`SnapshotHash`, analysis requests by name (`EngineAnalysisRequest`), the analysis descriptors with their settings schemas, results as JSON with buffers (`AnalysisOutput`, `ResultBuffer`), `RunInfo`, the `AnalysisJob` handle, `GPUPreparation`, the viewport's `LayoutScene`, the prescription import types and `MockEngine`. |
| `MKLightUI` | The document app. `Document/` (`ProjectDocument` file adapter, `DesignDocument` design state with undo), `Editor/` (`LensEdit`, the lens-data edits), `ViewModels/` (`Workspace` snapshots and panes, `AnalysisPane`, `WindowState` and the saved `WorkspaceLayout`, `PanelArrangement`, `LensTable`, the data-driven `SurfaceTypes` and the display sags of their previews (`ShapeSag`), design-job sessions, result readers, `GPUPreparationMonitor`; platform-neutral), `Views/` (window, toolbar status, sidebar, lens table, inspectors with shape previews, analysis panes and dedicated plots in Swift Charts, design-job inspector panels, welcome window with layout thumbnails), `Viewport/` (Metal renderer, engine layout reader, NS/UIViewRepresentable wrappers), `Support/` (palette, parsers, platform adapters). |
| `CMKLight` | System-library module over `crates/mklight-ffi/include/mklight.h`; present only when `Frameworks/libmklight_ffi.a` exists. |
| `MKLightFFI` | `FFIEngine`: `EngineClient` over the C ABI (sessions, catalogs, jobs, zero-copy buffers). Compiles without the library and then reports it unavailable. |
| `MKLight` | The executable. It uses `FFIEngine` (debug builds without the library fall back to `MockEngine`) and runs `MKLightScene`. |

## Engine boundary

`EngineClient` (`Sources/MKLightEngine/EngineClient.swift`) is the protocol
every engine implements:

| member | what it does |
|---|---|
| `info` | `EngineInfo`: name, version and `isMock` |
| `openSnapshot(_:)` | validates a system and freezes it into a hashed `ModelSnapshot` |
| `analysisDescriptors()` | every analysis with its settings schema and availability |
| `submit(_:on:)` | starts any engine analysis by name (`EngineAnalysisRequest`), with JSON settings |
| `materialNames()`, `catalogs()`, `loadCatalog(at:)` | glass names for completion, the glass catalogs, loading a user catalog |
| `makeDocumentEngine()` | the same engine with a session of its own for one document |
| `applySolves(settings:to:)` | solves applied by the engine, returning the report and the new snapshot |
| `importPrescription(_:format:)`, `importPrescription(at:)` | `.zmx` and `.seq` import |
| `prepareGPU(cacheDirectory:)`, `gpuPreparation()` | the GPU kernel preparation at launch and its state (`GPUPreparation`) |

The requirements after `materialNames()` have default implementations in an
extension, for engines without those capabilities: they fail with
`EngineError.unsupported`, list no catalogs, return the engine itself as
the document engine, apply solves from the `solves` analysis' report, or
have no GPU to prepare. An analysis is described by data alone: what it
takes is its settings schema, and what it returns is the engine's result
JSON (docs/RESULTS.md).

* **Snapshots.** `ModelSnapshot` is an immutable system plus its
  engine-computed content hash and the hash of the glass library it was
  taken with. Analyses run only against snapshots, never against the live
  document.
* **Jobs.** `AnalysisJob` exposes `progress: AsyncStream<JobProgress>`
  (a monotone fraction plus a stage label), `result() async throws`,
  `cancel()` and, for jobs that support it, `stop()` (`canStop`: end early
  and keep the best result so far). Cancelling the task that awaits a job
  does not cancel the job. Only `cancel()` does. Implementations build jobs
  with `AnalysisJob.run` and report through `JobReporter`.
* **Requests.** Every analysis and design job is an
  `EngineAnalysisRequest`: the engine's analysis name, the settings its
  schema accepts as JSON (a seed is the analysis' own `seed` setting) and an
  `ExecutionSettings` with the backend preference. `AnalysisDescriptor`
  carries the schema (`SettingsField`s, from which the app builds its
  settings forms) and reduces settings to what the schema declares
  (`acceptedSettings`). `FFIEngine` fails a job whose settings the engine
  refuses at once with `EngineError.invalidSettings`.
* **Results.** An `AnalysisOutput` is the engine's result JSON plus its
  binary `ResultBuffer`s, the `RunInfo`, the glass library hash and the
  backend report (the envelope, docs/RESULTS.md §1). The views read the JSON and the buffers through tolerant accessors
  (`JSONValue.nf`, `nfArray`, `[path:]`, `buffer(named:)`): a missing key or
  another type reads as `nil`, and the pane shows the result generically.
* **Reproducibility.** `AnalysisOutput.isCurrent(for:request:)` drives the "stale" state: the
  snapshot and glass library must match, and so must the request (the one
  that produced the output, or else every requested setting must have the
  requested value in the engine's recorded settings). Settings compare as
  exact JSON values (`JSONValue` keeps integers as `Int64` / `UInt64`, so
  64-bit seeds are never rounded through `Double`).
* **Layout scene.** `LayoutScene` contains polylines for surface profiles,
  rims, element edges, the stop, the image and the axis, plus rays. Each
  polyline is tagged with its surface, field and wavelength and with
  whether it lies in the drawn section. The scene also has matched
  front/back element profiles for glass fills, the surfaces' semi-diameters
  (fixed or automatic) and the optional surface mesh. `EngineLayout` reads
  it from the engine's `layout` buffers; the viewport only projects and
  draws it.
* **Non-finite results.** serde_json writes non-finite `f64` as `null`; the
  engine writes infinities as `"inf"`/`"-inf"`. Result scalars that can
  legitimately be non-finite (EFL of an afocal system, RMS radius with no
  surviving rays, MTF samples) are read with `JSONValue.nf`, which takes a
  number, `null` (NaN) or those strings; Codable mirrors of engine JSON
  (`GlassInfo`) use the `EngineDouble` wrapper, which also round-trips
  them.

### MockEngine (synthetic data)

The mock runs a few analyses **by name in the engine's result format**
(`validate`, `first_order`, `layout`, `spot_diagram`, `ray_fan`, `opd_fan`,
`field_curvature_distortion`, `solves`, `glass_info`, `merit_evaluation`,
and synthetic `optimize` and `tolerance` jobs that report progress and
support Stop), so the views and view-model tests work without the Rust
engine. It lists `seidel`, `mtf`, `psf` and `wavefront_map` as unavailable,
so the app can show that state.

`MockEngine` implements the core of the contract: snapshots that refuse an
invalid model, hashing (SHA-256 of sorted-key JSON, where the Rust engine
uses BLAKE3), jobs with progress and cancellation, and seeded random
sampling. Glass catalogs, applying solves, prescription import and GPU
preparation take the protocol's default implementations. Its numbers come
from `MockOptics`, a deliberately small stand-in. It uses a Cauchy index
fitted to n_d/V_d and a rotationally symmetric real-ray trace that ignores
tilts, decenters, ray aiming and configurations; its settings and results
are internal types that `MockByName.swift` writes out as the engine's JSON
and buffers. Extended surface shapes are traced as their
rotational y-meridian base (toroid and biconic: the y profile; Q, XY and
Zernike: the base conic without the departure; diffractive: the base, plus the
first phase term's paraxial power). The app labels its output: an orange
warning mark in the toolbar's status capsule (its popover says "Mock engine:
results are synthetic"), the engine line of the welcome window ("Mock
engine — synthetic data"), and every result's `RunInfo` with
`backend == "mock"` and a "MOCK ENGINE" warning. Its paraxial and ray code is
checked against known answers (thick-lens equations, a paraboloid's perfect
focus, third-order spherical aberration of a mirror), so the synthetic data is
at least self-consistent.

### The Rust engine

`FFIEngine` (module `MKLightFFI`) is the production `EngineClient`; see
`docs/APP.md`, "`FFIEngine`".

## State model

The design (`DesignDocument`), the engine snapshots and analysis panes
(`Workspace`) and the view state (`WindowState`) are separate; see
docs/APP.md, "State model".

## Files

`.mklight` files use the engine's project format (docs/FFI.md, "Project
file"). The system is decoded strictly, like the engine's model: an unknown
field in it is an error rather than silently dropped. The exported UTI is
`com.mklight.project`, which conforms to `public.json` and is declared in the
bundle's Info.plist by the build script.

## Lens data editor

`LensTableView` is a SwiftUI `Table` (docs/APP.md, "Window structure"). Numbers follow
`LensValueParser`: `inf` or `∞` for a plane radius and an infinite object
distance, `auto` for an automatic semi-diameter (shown dimmed with the
engine's computed value). The material column accepts glass names (completed
from the session's catalogs), `AIR`, `MIRROR`, `DUMMY`, `nd/vd`, `n=…` and
`CATALOG:NAME`.

## Metal viewport

`ViewportView` wraps an `MTKView` in an `NSViewRepresentable` on macOS and
a `UIViewRepresentable` on iPadOS (`ViewportBridge` is shared; only input
handling differs). The scene is the engine's `layout` result
(`EngineLayout` reads its buffers). It draws on demand only, with 4× MSAA.
Polylines become instanced screen-space quads with analytic anti-aliasing
and round caps. Lens elements get translucent glass fills in the section
view. The modes (y-z and x-z sections, 3-D orbit), display options and
footprints are described in docs/APP.md, "Window structure". Controls are drag
(pan or orbit), scroll or pinch to zoom about the cursor, and double-click
(or ⌘0) to fit. Rays are coloured by field or by wavelength.

Precision: positions are transformed in double precision before rounding to
float, and the view matrix is built in double precision. A float32 vertex at
distance `D` from the upload origin is off by about `D·2⁻²⁴` mm, i.e.
`D·2⁻²⁴·zoom` points on screen. While that stays below 0.02 pt, geometry is
uploaded relative to the scene centre and camera moves only change the matrix.
Beyond it (deep zoom away from the centre), `SceneGeometry.frame` switches to
a camera-relative, view-aligned frame and clips segments and fills in double
precision to a window just larger than the view, rebuilding on camera moves.
Every uploaded coordinate is then within about one viewport of the origin,
so lines stay sub-pixel exact up to the maximum zoom (tested at 10⁶ pt/mm).
The shaders guard divisions by the segment length and by `w`.

Shaders: `Viewport.metal` ships as a package resource and is compiled at run
time with `MTLCompileOptions.mathMode = .safe`. SwiftPM compiles `.metal`
sources only when the optional Metal Toolchain component is installed
(`xcodebuild -downloadComponent MetalToolchain`), and on this machine it is
not installed. When it is installed, `build-macos.sh` precompiles
`MKLightViewport.metallib` into the app bundle, and the app prefers that
library.

## Open contract items

* The engine validates every configured system (each override applied) and
  the per-kind rules of solves; the Swift mirror's `structuralIssues()`
  checks only that an override's path resolves and its value is one the
  setter takes. With `FFIEngine` the engine's verdict decides; `MockEngine`
  accepts such a model.
