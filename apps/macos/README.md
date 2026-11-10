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

The app runs on the Rust engine through `FFIEngine`; see `docs/APP.md` for
the build integration, the binding, its thread safety and the tests.

Requires Xcode 27.2 or later (Swift 6.4, Swift 6 language mode, strict
concurrency) and macOS 27 or later: the window is designed for Liquid Glass
and uses its SwiftUI APIs without availability checks (docs/APP.md,
"Platform" and "Design brief").

## Package layout

| target | contents |
|---|---|
| `MKLightModel` | Codable, `Sendable`, `Hashable` value types that mirror the engine's `OpticalSystem` JSON exactly. That covers tagged enums (`"type"`, `"mode"`, `"param"`), snake_case keys, explicit `null`s, `"inf"`/`"-inf"` thicknesses and curvature rather than radius, and every surface field (including the DOE order-efficiency model `diffraction_efficiency`: ideal, kinoform, multilevel N). Every surface shape of `docs/SURFACES.md` is mirrored (odd asphere, Q-con/Q-bfs, diffractive on a plane/conic/even-asphere base, toroid, biconic with its cylinder preset, XY polynomial, Zernike sag, and the fringe irregularity `sag_overlay` on a geometric base), with the `shape_coefficient`/`shape_param` parameter paths, their limits and term numbering (`CoefficientSet`, `ShapeParameter`, `XYTerm`, `ZernikeTerm`), checked against `extended_shapes.json`. It also includes structural validation (the engine's `validate` checks, including `ParamPath::get`'s shape, material, semi-diameter and aperture rules for configuration overrides, verified case by case against engine verdicts in `validation_cases.json`; values JSON cannot carry are errors, and mechanical parts on missing surfaces, which the engine rejects, are only warnings here), reference-preserving edits (surface insert/delete/stop, field and wavelength removal that renumbers index-addressed overrides), radius↔curvature display helpers, the `.mklight` project envelope (`ProjectFile`) and sample systems. It has no UI and no optics. |
| `MKLightEngine` | The engine boundary. It holds the `EngineClient` protocol, `ModelSnapshot`/`SnapshotHash`, analysis requests by name and typed requests with their settings, result types and buffers, `RunInfo`, the `AnalysisJob` handle, `GPUPreparation`, the prescription import types and `MockEngine`. |
| `MKLightUI` | The document app. `Document/` (`ProjectDocument` file adapter, `DesignDocument` design state with undo), `Editor/` (`LensEdit`, the lens-data edits), `ViewModels/` (`Workspace` snapshots and panes, `AnalysisPane`, `WindowState` and the saved `WorkspaceLayout`, `PanelArrangement`, `LensTable`, the data-driven `SurfaceTypes` and the display sags of their previews (`ShapeSag`), design-job sessions, result readers, `GPUPreparationMonitor`; platform-neutral), `Views/` (window, toolbar status, sidebar, lens table, inspectors with shape previews, analysis panes and dedicated plots in Swift Charts, design-job inspector panels, welcome window with layout thumbnails), `Viewport/` (Metal renderer, engine layout reader, NS/UIViewRepresentable wrappers), `Support/` (palette, parsers, platform adapters). See docs/APP.md for the design. |
| `CMKLight` | System-library module over `crates/mklight-ffi/include/mklight.h`; present only when `Frameworks/libmklight_ffi.a` exists. |
| `MKLightFFI` | `FFIEngine`: `EngineClient` over the C ABI (sessions, projects, catalogs, jobs, zero-copy buffers). Compiles without the library and then reports it unavailable. |
| `MKLight` | The executable. It uses `FFIEngine` (debug builds without the library fall back to `MockEngine`) and runs `MKLightScene`. |

## Engine boundary

`EngineClient` (`Sources/MKLightEngine/EngineClient.swift`) is the protocol
every engine implements:

| member | what it does |
|---|---|
| `info` | `EngineInfo`: name, version and `isMock` |
| `openSnapshot(_:)` | validates a system and freezes it into a hashed `ModelSnapshot` |
| `validate(_:)` | every `ModelIssue`, without throwing |
| `submit(_: EngineAnalysisRequest, on:)` | starts any engine analysis by name, with JSON settings |
| `submit(_: AnalysisRequest, on:)` | starts one of the typed analyses (`AnalysisKind`) |
| `analysisDescriptors()` | every analysis with its settings schema and availability |
| `materialNames()`, `catalogs()`, `loadCatalog(at:)` | glass names for completion, the glass catalogs, loading a user catalog |
| `makeDocumentEngine()` | the same engine with a session of its own for one document |
| `applyParamEdits(_:to:)`, `applySolves(settings:to:)` | parameter edits and solves applied by the engine, returning the new snapshot |
| `importPrescription(_:format:)`, `importPrescription(at:)` | `.zmx` and `.seq` import |
| `prepareGPU(cacheDirectory:)`, `gpuPreparation()` | the GPU kernel preparation at launch and its state (`GPUPreparation`) |

The requirements after `materialNames()` have default implementations in an
extension, for engines without those capabilities: they fail with
`EngineError.unsupported`, list no catalogs, return the engine itself as
the document engine, or have no GPU to prepare.

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
* **Requests by name.** The app submits every analysis and design job as an
  `EngineAnalysisRequest`: the engine's analysis name, the settings its
  schema accepts as JSON (a seed is the analysis' own `seed` setting) and an
  `ExecutionSettings` with the backend preference. `FFIEngine` fails a job
  whose settings the engine refuses at once with
  `EngineError.invalidSettings`.
* **Typed requests.** `AnalysisRequest` covers the layout scene, spot
  diagram, ray fan, OPD fan, field curvature/distortion, MTF, PSF and
  first-order data with one Codable settings struct each
  (`SpotDiagramSettings`, `MTFSettings`, …) and its `execution` block
  (`ExecutionSettings`: backend preference and the seed of random pupil
  sampling). `AnalysisRequest.problems(for:)` checks sample counts,
  frequencies and field/wavelength/configuration indices against the
  snapshot, and engines fail such requests at once with
  `EngineError.invalidSettings`. `MockEngine` and the tests use this layer;
  the app does not.
* **Reproducibility.** Every `AnalysisOutput` carries a `RunInfo` that
  mirrors the engine's. It records the engine version, snapshot hash,
  settings JSON, the backend and precision actually used, the seed, the
  accuracy tier, timing, warnings and failed backend attempts.
  `AnalysisOutput.isCurrent(for:request:)` drives the "stale" state: the
  snapshot and glass library must match, and so must the request (for a
  request by name, every requested setting must have the requested value in
  the engine's recorded settings). Settings compare as exact JSON values
  (`JSONValue` keeps integers as `Int64` / `UInt64`, so 64-bit seeds are
  never rounded through `Double`).
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
  surviving rays, MTF samples, points) decode through `EngineDouble`, which
  accepts a number, `null` (NaN) or those strings and round-trips them.
* **Results.** A result by name is a `GenericResult`: the engine's result
  JSON plus its binary `ResultBuffer`s, which point into the engine's memory
  without a copy. The views read both through tolerant accessors.

### MockEngine (synthetic data)

The mock also runs a few analyses **by name in the engine's result format**
(`validate`, `first_order`, `layout`, `spot_diagram`, `ray_fan`, `opd_fan`,
`field_curvature_distortion`, `solves`, `glass_info`, `merit_evaluation`,
and synthetic `optimize` and `tolerance` jobs that report progress and
support Stop), so the views and view-model tests work without the Rust
engine.

`MockEngine` implements the core of the contract: validation, hashing
(SHA-256 of sorted-key JSON, where the Rust engine uses BLAKE3), simulated
stages with progress and cancellation, and seeded random sampling. Glass
catalogs, parameter edits, prescription import and GPU preparation take the
protocol's default implementations. Its numbers come from
`MockOptics`, a deliberately small stand-in. It uses a Cauchy index fitted
to n_d/V_d, a rotationally symmetric real-ray trace that ignores tilts,
decenters, ray aiming and configurations, and diffraction-limit × Gaussian-blur
approximations for MTF and PSF. Extended surface shapes are traced as their
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
`docs/APP.md`. Requests by name (`EngineAnalysisRequest`) go to the engine
as they are and come back as `GenericResult`s. Typed requests run as the
engine analysis `AnalysisKind.engineName` with the settings its schema
accepts (`AnalysisRequest.engineSettings(for:)`); their results decode into
the typed mirrors when the JSON matches and are shown generically
otherwise.

## State model

The design (`DesignDocument`), the engine snapshots and analysis panes
(`Workspace`) and the view state (`WindowState`) are separate; see
docs/APP.md, "State model". Undo and redo restore whole design values;
analyses only ever run against immutable engine snapshots; panes go stale
(dimmed, with an Update button) when the design, the glass library or their
settings change.

## Files

`.mklight` files use the engine's project format (docs/FFI.md):
`{"format": "mklight-project", "schema_version": 1, "metadata", "system",
"catalogs", "views", "merit", "variables", "tolerances"}`. The app edits the
system and the catalog list and keeps every other section, including
unknown keys, verbatim. It also writes `merit` (the merit editor's
default-generator settings or custom operands as an engine
`MeritFunctionSpec`, plus the EFL hold), `variables` (`VariableSpec`s
with their bounds) and `tolerances` (the tolerance set as an engine
`ToleranceSpec`), all undoable design changes, and `views` (the panes and
the panel arrangement), which is not an undoable change. The system is
decoded strictly, like the engine's
model: an unknown field in it is an error rather than silently dropped. Like
the engine, the app reads only the current `schema_version` and refuses a
file without one or at another version. The exported
UTI is `com.mklight.project`, which conforms to `public.json` and is declared
in the bundle's Info.plist by the build script.

## Lens data editor

A SwiftUI `Table` with the columns every surface has: surface (OBJ, 1, …,
STO with the aperture symbol, IMA), type, comment, radius, thickness,
material, semi-diameter and conic. Type-specific parameters are in the
surface inspector, built from `SurfaceTypes` descriptors.

* Return or a double click edits the selected row; in a cell, Tab / ⇧Tab
  move across, ↑ / ↓ and Return move between rows (committing), Esc
  reverts. Rejected input reverts and the reason appears in a glass toast.
* Numbers follow `LensValueParser`: `inf` or `∞` for a plane radius and an
  infinite object distance, `auto` for an automatic semi-diameter (shown
  dimmed with the engine's computed value). The material column accepts
  glass names (completed from the session's catalogs), `AIR`, `MIRROR`,
  `DUMMY`, `nd/vd`, `n=…` and `CATALOG:NAME`.
* Inline markers: `V` for a variable (a small bar under it when it has
  bounds, a dot when it is pinned by equal bounds), a letter for the solve
  controlling the value; validation issues badge the surface number.
* ⇧⌘I / ⌥⌘I insert, ⌘⌫ deletes, ⇧⌘T makes the stop, ⌃⌘V toggles the
  focused value's variable; the same commands are in the header buttons,
  the context menu and the Lens menu.

## Metal viewport

`ViewportView` wraps an `MTKView` in an `NSViewRepresentable` on macOS and
a `UIViewRepresentable` on iPadOS (`ViewportBridge` is shared; only input
handling differs). The scene is the engine's `layout` result
(`EngineLayout` reads its buffers). It draws on
demand only, with 4× MSAA. Polylines become instanced screen-space quads with
analytic anti-aliasing and round caps. Lens elements get translucent glass
fills in the section view. The engine draws every surface type. There are
three modes:

* the y-z section (2-D);
* the x-z section (2-D, offered for designs without rotational symmetry;
  the layout's `section` setting follows the mode, ⌥⌘2);
* an orthographic 3-D orbit view, which adds sagittal profiles, rims and
  skew rays, and optionally the engine's triangulated surface mesh
  (`mesh_rings`, flat shaded) and the beam footprint of every field on
  every surface (the `footprint` analysis' global outlines, filled
  translucently in the field colours).

DOE surfaces carry a fine violet hatch (layout flag bit 2). Controls are
drag (pan or orbit), scroll or pinch to zoom about the cursor, and
double-click (or ⌘0) to fit; Fit Lens frames the elements only, and the
first fit of a slim, long scene does so by itself. Rays are coloured by
field or by wavelength. A previewed candidate design (optimisation, glass
substitution or a tolerance build) is labelled under the controls.

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

* The typed result mirrors (`SpotDiagramResult`, `FanResult`, `MTFResult`,
  …) have no round-trip tests against engine fixtures (the model fixtures
  come from `dump_fixtures`); a typed result whose JSON does not match is
  shown generically.
* The engine's `OpticalSystem::validate` rejects mechanical parts attached
  to missing surfaces (`missing_surface`), while the Swift mirror's
  `structuralIssues()` reports them only as warnings. With `FFIEngine` the
  engine's verdict decides; `MockEngine` accepts such a model.
