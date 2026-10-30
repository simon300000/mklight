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

Requires Xcode 16 or later (Swift 6 language mode, strict concurrency) and
macOS 15 or later.

## Package layout

| target | contents |
|---|---|
| `MKLightModel` | Codable, `Sendable`, `Hashable` value types that mirror the engine's `OpticalSystem` JSON exactly. That covers tagged enums (`"type"`, `"mode"`, `"param"`), snake_case keys, explicit `null`s, `"inf"`/`"-inf"` thicknesses and curvature rather than radius. It also includes structural validation (the engine's `validate` checks, including `ParamPath::get`'s shape, material, semi-diameter and aperture rules for configuration overrides, verified case by case against engine verdicts in `validation_cases.json`; plus errors for values JSON cannot carry and warnings for mechanics on missing surfaces), reference-preserving edits (surface insert/delete/stop, field and wavelength removal that renumbers index-addressed overrides), radius↔curvature display helpers, the `.mklight` project envelope (`ProjectFile`) and sample systems. It has no UI and no optics. |
| `MKLightEngine` | The engine boundary. It holds the `EngineClient` protocol, `ModelSnapshot`/`SnapshotHash`, analysis requests and settings, result types, `RunInfo`, the `AnalysisJob` handle and `MockEngine`. |
| `MKLightUI` | The document app: `ProjectDocument`, `WindowViewState`, `AnalysisCoordinator`, the lens data editor (`NSTableView`), the inspector, the analysis plots (Swift Charts), the PSF heat map and the Metal viewport. |
| `CMKLight` | System-library module over `crates/mklight-ffi/include/mklight.h`; present only when `Frameworks/libmklight_ffi.a` exists. |
| `MKLightFFI` | `FFIEngine`: `EngineClient` over the C ABI (sessions, projects, catalogs, jobs, zero-copy buffers). Compiles without the library and then reports it unavailable. |
| `MKLight` | The executable. It uses `FFIEngine` (debug builds without the library fall back to `MockEngine`) and runs `MKLightScene`. |

## Engine boundary

```swift
public protocol EngineClient: Sendable {
    var info: EngineInfo { get }                                     // name, version, backends, isMock
    func openSnapshot(_ system: OpticalSystem) async throws -> ModelSnapshot   // validate + hash
    func validate(_ system: OpticalSystem) async -> [ModelIssue]
    func submit(_ request: AnalysisRequest, on snapshot: ModelSnapshot) -> AnalysisJob
    func materialNames() async -> [String]
}
```

* **Snapshots.** `ModelSnapshot` is an immutable system plus its
  engine-computed content hash. Analyses run only against snapshots, never
  against the live document.
* **Jobs.** `AnalysisJob` exposes `progress: AsyncStream<JobProgress>`
  (a monotone fraction plus a stage label), `result() async throws`, and
  `cancel()`. Cancelling the task that awaits a job does not cancel the job.
  Only `cancel()` does. Implementations build jobs with `AnalysisJob.run`
  and report through `JobReporter`.
* **Reproducibility.** Each settings struct (`SpotDiagramSettings`,
  `MTFSettings`, …) is a Codable value with an `execution` block
  (`ExecutionSettings`: backend preference, requested precision and the
  Philox seed), so a request carries a job's whole identity
  `(snapshot, settings, backend, precision, seed)`. Every `AnalysisOutput`
  carries a `RunInfo` that mirrors the engine's. It records the engine
  version, snapshot hash, settings JSON (including the execution block), the
  backend and precision actually used, the seed, timing and warnings.
  `RunInfo.isCurrent(for:request:)` drives the "stale" badges; it compares
  settings as exact JSON values (`JSONValue` keeps integers as `Int64` /
  `UInt64`, so 64-bit seeds are never rounded through `Double`).
* **Settings validation.** `AnalysisRequest.problems(for:)` checks sample
  counts, frequencies and field/wavelength/configuration indices against the
  snapshot. Engines fail such requests at once with
  `EngineError.invalidSettings`.
* **Analyses.** The boundary covers the layout scene, spot diagram, ray fan,
  OPD fan, field curvature/distortion, MTF, PSF and first-order data.
  `LayoutScene` contains polylines for surface profiles, rims, element edges,
  the stop, the image and the axis, plus rays. Each polyline is tagged with
  its surface, field and wavelength and with whether it lies in the y-z
  section. The scene also has matched front/back element profiles for glass
  fills and the automatic semi-diameters. The viewport only projects and
  draws it.
* **Non-finite results.** serde_json writes non-finite `f64` as `null`; the
  engine writes infinities as `"inf"`/`"-inf"`. Result scalars that can
  legitimately be non-finite (EFL of an afocal system, RMS radius with no
  surviving rays, MTF samples, points) decode through `EngineDouble`, which
  accepts a number, `null` (NaN) or those strings and round-trips them.
* **Wire format.** Every request and result type is Codable with the snake_case
  JSON the Rust side will emit. Requests are `{"analysis", "settings"}`, results
  are `{"analysis", "data"}` and outputs are `{"run_info", "result"}`. An FFI
  client therefore only has to move JSON across the C ABI. Large arrays
  (spot points, PSF pixels) can switch to binary buffers later without
  changing UI code.

### MockEngine (synthetic data)

`MockEngine` implements the full contract: validation, hashing (SHA-256 of
sorted-key JSON, where the Rust engine uses BLAKE3), simulated stages with
progress and cancellation, and seeded random sampling. Its numbers come from
`MockOptics`, a deliberately small stand-in. It uses a Cauchy index fitted
to n_d/V_d, a rotationally symmetric real-ray trace that ignores tilts,
decenters, ray aiming and configurations, and diffraction-limit × Gaussian-blur
approximations for MTF and PSF. The app labels its output everywhere: the
orange "MOCK ENGINE" badge in the viewport, the status bar, and
`RunInfo.backend == "mock"` with a warning. Its paraxial and ray code is
checked against known answers (thick-lens equations, a paraboloid's perfect
focus, third-order spherical aberration of a mirror), so the synthetic data is
at least self-consistent.

### The Rust engine

`FFIEngine` (module `MKLightFFI`) is the production `EngineClient`; see
`docs/APP.md`. Typed requests run as the engine analysis
`AnalysisKind.engineName`; results the typed mirrors cannot decode, and
analyses requested by name (`EngineAnalysisRequest`, the Engine pane), are
shown generically (`GenericResult`: JSON plus `ResultBuffer`s).

## State model

* **Simulation state.** `ProjectDocument` (a `ReferenceFileDocument`) holds
  only the `OpticalSystem` value. Each edit goes through
  `edit(_:undoManager:_:)`, which replaces the value and registers the
  previous value for undo, so undo and redo restore whole value snapshots.
  An edit that invalidates configuration overrides or mechanical parts (a
  conic override on a surface made plane, a semi-diameter override on a
  surface switched to automatic) removes them in the same undoable step.
* **View state.** `WindowViewState` is per window. It holds the selection (by
  `SurfaceID`, so it survives inserts and deletes), the visible panels, the
  inspector tab, the viewport mode and camera, ray colouring, and every
  analysis's settings. None of it is written into the model.
* **Analysis orchestration.** `AnalysisCoordinator` is per window. It
  debounces model changes (150 ms), asks the engine for a snapshot, and then
  re-runs only the visible analyses. A newer request cancels the older job
  for the same view. Old results stay on screen, dimmed with a STALE badge,
  until the new ones arrive. While the model fails validation, no analyses
  run and the errors are listed in the status bar.

## Files

`.mklight` files use the engine's project format (docs/FFI.md):
`{"format": "mklight-project", "schema_version": 1, "metadata", "system",
"catalogs", "views", "merit", "variables", "tolerances"}`. The app edits the
system and the catalog list and keeps every other section, including
unknown keys, verbatim. Bare `OpticalSystem` JSON and the app's early
`{"format", "version": 1, "system"}` files open too (as the engine's
version-0 migration does); newer schema versions are refused. The exported
UTI is `com.mklight.project`, which conforms to `public.json` and is declared
in the bundle's Info.plist by the build script.

## Lens data editor

The editor is an `NSTableView` with these columns: surface, type, comment,
radius, thickness, material, semi-diameter, conic and coating.

* Use ↑/↓ to select a row and ←/→ or Tab to move the focused cell.
* Return starts editing the focused cell. Typing a character also starts
  editing, with that character as the first input. While editing, Return
  commits and moves down, Tab commits and moves across, and Esc cancels.
* Backspace/Delete clears the focused cell into an edit (Esc restores it);
  `+`, `-` and digits start an edit like any other character. Structural
  edits need explicit commands: ⌘⌫ deletes the selected surface and Insert
  (or ⌘⇧I / ⌘⌥I) inserts one. The Lens menu and the context menu have the
  same commands, plus Make Stop.
* Numeric validation follows `LensValueParser`: `inf` or `∞` means a plane
  radius and an infinite object distance, and semi-diameter accepts `auto`.
  The material column accepts glass names, `AIR`, `MIRROR`, `DUMMY`,
  `nd/vd`, `n=…` and `CATALOG:NAME`. Rejected input beeps and the message
  appears in the status bar.

## Metal viewport

`ViewportView` wraps an `MTKView` in an `NSViewRepresentable`. It draws on
demand only, with 4× MSAA. Polylines become instanced screen-space quads with
analytic anti-aliasing and round caps. Lens elements get translucent glass
fills in the section view. There are two modes:

* the y-z section (2-D);
* an orthographic 3-D orbit view, which adds sagittal profiles, rims and
  skew rays.

Controls are drag (pan or orbit), scroll or pinch to zoom about the cursor,
and double-click (or ⌘0) to fit. Rays are coloured by field or by
wavelength.

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

* The typed analysis result schemas are defined only here for now. When
  `mklight-engine` defines its result structs, it should dump fixtures for
  them (as `dump_fixtures` does for the model), and the Swift types should be
  round-trip tested against those; until then mismatching results are shown
  generically.
* The engine's `OpticalSystem::validate` accepts two things the Swift check
  reports: mechanical parts attached to missing surfaces (a warning here), and
  values with no JSON form, such as a NaN curvature (an error here; such a
  model cannot cross the JSON boundary anyway). The engine should reject both.
