# MKLight for macOS

The native macOS front end: a SwiftUI/AppKit document app with a Metal
layout viewport, built around the headless Rust engine. The app never
computes optics. Every number it shows comes from an `EngineClient`.

```
swift build                     # debug build
swift test                      # model, engine-boundary and UI-logic tests
../../scripts/build-macos.sh    # release build → build/macos/MKLight.app
```

Requires Xcode 16 or later (Swift 6 language mode, strict concurrency) and
macOS 15 or later.

## Package layout

| target | contents |
|---|---|
| `MKLightModel` | Codable, `Sendable`, `Hashable` value types that mirror the engine's `OpticalSystem` JSON exactly. That covers tagged enums (`"type"`, `"mode"`, `"param"`), snake_case keys, explicit `null`s, `"inf"`/`"-inf"` thicknesses and curvature rather than radius. It also includes structural validation (the same checks as the engine's `validate`), surface insert/delete/stop edits, radius↔curvature display helpers, the `.mklight` project envelope (`ProjectFile`) and sample systems. It has no UI and no optics. |
| `MKLightEngine` | The engine boundary. It holds the `EngineClient` protocol, `ModelSnapshot`/`SnapshotHash`, analysis requests and settings, result types, `RunInfo`, the `AnalysisJob` handle and `MockEngine`. |
| `MKLightUI` | The document app: `ProjectDocument`, `WindowViewState`, `AnalysisCoordinator`, the lens data editor (`NSTableView`), the inspector, the analysis plots (Swift Charts), the PSF heat map and the Metal viewport. |
| `MKLight` | The executable. It picks the engine (`MockEngine` for now) and runs `MKLightScene`. |

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
  `MTFSettings`, …) is a Codable value and carries its seed where one
  applies. Every `AnalysisOutput` carries a `RunInfo` that mirrors the
  engine's. It records the engine version, snapshot hash, settings JSON,
  backend, precision, seed, timing and warnings.
  `RunInfo.isCurrent(for:request:)` drives the "stale" badges.
* **Analyses.** The boundary covers the layout scene, spot diagram, ray fan,
  OPD fan, field curvature/distortion, MTF, PSF and first-order data.
  `LayoutScene` contains polylines for surface profiles, rims, element edges,
  the stop, the image and the axis, plus rays. Each polyline is tagged with
  its surface, field and wavelength and with whether it lies in the y-z
  section. The scene also has matched front/back element profiles for glass
  fills and the automatic semi-diameters. The viewport only projects and
  draws it.
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

### Plugging in the Rust engine

1. `crates/mklight-ffi` exports a C ABI (cbindgen header) that takes JSON
   strings and returns JSON plus binary buffers. It also exports job handles
   with progress polling and cancellation.
2. Add a `MKLightFFI` target with a module map over the header and an
   `FFIEngine: EngineClient`. Bridge each job's progress into the
   `JobReporter` and forward `cancel()` to the engine's cancel token.
3. Build and link `libmklight_ffi.a` at the hook in `scripts/build-macos.sh`.
4. Switch `Sources/MKLight/main.swift` from `MockEngine()` to the FFI engine.
   No UI code changes.

## State model

* **Simulation state.** `ProjectDocument` (a `ReferenceFileDocument`) holds
  only the `OpticalSystem` value. Each edit goes through
  `edit(_:undoManager:_:)`, which replaces the value and registers the
  previous value for undo, so undo and redo restore whole value snapshots.
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

`.mklight` files are pretty-printed JSON of the form
`{"format": "mklight-project", "version": 1, "system": {…}}`. Bare
`OpticalSystem` JSON (engine fixtures, CLI output) opens too. The exported
UTI is `com.mklight.project`, which conforms to `public.json` and is declared
in the bundle's Info.plist by the build script.

## Lens data editor

The editor is an `NSTableView` with these columns: surface, type, comment,
radius, thickness, material, semi-diameter, conic and coating.

* Use ↑/↓ to select a row and ←/→ or Tab to move the focused cell.
* Return starts editing the focused cell. Typing a character also starts
  editing, with that character as the first input. While editing, Return
  commits and moves down, Tab commits and moves across, and Esc cancels.
* Delete removes the selected surface. Insert or `+` inserts a surface.
  The Lens menu and the context menu have the same commands, plus Make Stop.
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

Precision: positions are made relative to the scene centre in double
precision before rounding to float, and the view matrix is built in double
precision. The shaders guard divisions by the segment length and by `w`.

Shaders: `Viewport.metal` ships as a package resource and is compiled at run
time with `MTLCompileOptions.mathMode = .safe`. SwiftPM compiles `.metal`
sources only when the optional Metal Toolchain component is installed
(`xcodebuild -downloadComponent MetalToolchain`), and on this machine it is
not installed. When it is installed, `build-macos.sh` precompiles
`MKLightViewport.metallib` into the app bundle, and the app prefers that
library.
