# MKLight for macOS on the Rust engine

The native app (`apps/macos`, SwiftPM) runs on the Rust engine through its C
ABI (`crates/mklight-ffi`, docs/FFI.md). This page covers the design of the
user interface, the window structure, building, the Swift binding
(`FFIEngine`), the state model, the tests and what remains for iPadOS.
Package layout and the Metal viewport are described in
`apps/macos/README.md`; screenshots are in `apps/macos/docs/screenshots/`.

## Design principles

* **The lens data is the heart.** The table is always visible at the top of
  the window and shows only what every surface has: surface (with the stop
  marked), type, comment, radius, thickness, material, semi-diameter and
  conic. Everything type-specific lives in the inspector.
* **Progressive disclosure.** Surface types are described as data
  (`SurfaceTypes`): sections of parameters addressed by their key in the
  model's JSON, coefficient lists with add and remove, a variable toggle per
  parameter and a sag preview. The inspector shows only the selected type's
  parameters. A new type needs its model case and one descriptor, no view.
  Placement (decenters, tilts) appears when it is used, first for coordinate
  breaks. Analysis settings show the simple ones in a compact bar and the
  rest behind "More".
* **The engine is the authority.** The app never computes optics. Analyses
  are run by name and drawn from the engine's result JSON and buffers
  (docs/RESULTS.md). The list of analyses, their settings and their
  availability are read at run time, so a new engine analysis appears in the
  sidebar and runs with a generic, schema-driven view and no code change.
  Unknown results are shown as data, never as an error.
* **Reproducible and honest.** Every result carries the settings and seed
  the engine recorded, the backend that actually computed it (CPU f64
  reference or Metal f32, with the fallback reason) and its warnings, one
  click away in the pane footer. A result that no longer matches the design,
  the glass library or its settings is dimmed and says "Design changed",
  with an Update button.
* **Calm and native.** System materials, semantic colours, SF Symbols, the
  system font with monospaced digits for numbers, a 2–24 pt spacing scale
  (`Space`), no custom chrome. The accent colour marks only selection, the
  focused pane and variables. Plot colours come from one validated
  categorical palette (`Palette`); wavelengths keep their physical meaning
  (F blue, d yellow, C red) everywhere, fields take the categorical slots in
  order; maps use one hue for magnitude and blue ↔ red for signed values.
  Light and dark appearance are both designed.
* **No modal alerts for routine errors.** Rejected input is reverted and
  explained in the status bar; failing analyses explain themselves in their
  pane; validation issues are listed in a popover and badged in the table.

## Window structure

```
┌ toolbar: Optimize · Tolerance · Solves | Issues | Add Analysis · Layout · Inspector ┐
├ sidebar ──────┬ Lens Data ─────────────────────────────────────┬ inspector ────────┤
│ Analyses      │ Surf Type Comment Radius Thickness Material …  │ Surface | System  │
│  (searchable, ├────────────────────────┬───────────────────────┤ type-driven form  │
│  by category) │ Layout (Metal 2D / 3D) │ analysis panes …      │ or aperture,      │
│  Issues       │                        │ header · settings ·   │ fields, λ, …      │
│               │                        │ plot · footer         │                   │
├───────────────┴ status: compute backend · snapshot · message · EFL F/# TT image ───┤
```

* **`NavigationSplitView`**: the sidebar lists every engine analysis by
  category (`.searchable`); a click shows it in the focused pane, ⌥-click
  or the context menu opens a new pane. Open analyses carry a dot,
  unavailable ones are dimmed with the engine's reason as help.
* **Lens data** (`LensTableView`, SwiftUI `Table`): text cells edit in place
  with full keyboard navigation (Return / double click edits the row; Tab
  and ⇧Tab move across, ↑/↓ and Return move between rows committing, Esc
  reverts). Numbers accept `∞`/`inf`, and the material column completes glass
  names from the session's catalogs. Small inline markers show variables
  (`V`) and solves (`M`, `C`, `F`, `P`, …); validation issues badge their
  surface. Insert, delete, make stop and toggle variable are in the header,
  the context menu and the Lens menu.
* **Layout pane**: the engine's `layout` scene in the Metal viewport, section
  or 3-D orbit, rays by field or wavelength, the selected surface in the
  accent colour, scale bar, fit.
* **Analysis panes** (`AnalysisPaneView`): side by side with resizable
  widths. Each has a header (analysis menu, progress and Cancel, automatic
  update toggle, Run, settings toggle, more), a schema-driven settings bar
  (fields, wavelengths and surfaces as menus with their values), the plot
  and a footer with key numbers and the backend actually used. Dedicated
  views: first-order data, spot diagram (all fields, common scale, Airy
  disk, RMS/GEO), ray and OPD fans, field curvature and distortion, grid
  distortion, Seidel bars per surface with the sum and wavefront terms,
  longitudinal aberration, chromatic focal shift, lateral colour, relative
  illumination, footprint, encircled energy (geometric and diffraction),
  geometric and diffraction MTF (diffraction limit, Nyquist), MTF vs field,
  through-focus MTF, wavefront map with the Zernike table, PSF with Strehl
  (log scale), line spread, ghost paths, ghost irradiance and stray-light
  maps per class, polarisation pupil maps, system transmission, best focus,
  validation. Everything else: the generic view (scalars, record tables,
  curves, buffer maps, a JSON outline).
* **Inspector** (`.inspector`): Surface (type picker, data-driven parameter
  form, spacing and medium, aperture and clipping shape, placement, coating,
  solves) or System (aperture, fields, wavelengths with presets,
  environment, ray aiming, configurations, glass catalogs).
* **Design jobs** are sheets. *Optimize*: merit (RMS spot or wavefront,
  pupil sampling, hold EFL), variables (the design's `V` marks or the
  engine's automatic set), algorithm, iterations and seed; live iteration,
  merit and a log-scale merit history; **Stop** keeps the best design so far
  (ABI 1.2 `mk_job_stop`), Cancel discards; the result is a before/after
  comparison of merit and every variable, applied as one undoable step.
  *Tolerance*: ranked sensitivities with bars, Monte-Carlo statistics and a
  histogram. *Apply Solves*: a dry run report (now, solved, residual), then
  one undoable commit through `mk_session_apply_solves`.
* **Issues**: the toolbar shows the validation state; the popover lists
  severity, code and message; a click selects the surface (or highlights
  the field in the system inspector).
* **Welcome window**: new design, open, and the engine's sample designs
  (also File ▸ New from Sample, ⇧⌘1 to show it again).
* **Layout persistence**: the panes (analysis, settings, automatic update,
  widths), split positions, viewport mode and ray colouring are saved in the
  project's `views` section (`[{key, analysis, settings, ui}]`), written with
  the next save, not as an undoable design change.

Keyboard: ⇧⌘I / ⌥⌘I insert before / after, ⌘⌫ delete, ⇧⌘T make stop,
⌃⌘V toggle variable, ⌃⌘O optimize, ⌃⌘T tolerance, ⌃⌘L apply solves, ⌃⌘E issues,
⌘R run analyses, ⌘. cancel, ⌘2 / ⌘3 section / 3-D, ⌘0 fit, ⌥⌘L layout,
⌃⌘I inspector, ⌃⌘S sidebar.

## Building

```sh
scripts/build-macos.sh --engine-only   # engine library → apps/macos/Frameworks/libmklight_ffi.a
cd apps/macos && swift build && swift test

scripts/build-macos.sh                 # release app → build/macos/MKLight.app
scripts/build-macos.sh --debug         # debug app (launch options below)
scripts/build-macos.sh --universal     # engine for arm64 + x86_64 (rustup target x86_64-apple-darwin)
scripts/build-macos.sh --no-engine --engine-only   # remove the library again
```

* The script builds `mklight-ffi` as a **release static library** for
  `aarch64-apple-darwin` (plus `x86_64-apple-darwin` and `lipo` with
  `--universal`) and copies it to `apps/macos/Frameworks/` (git-ignored).
* `Package.swift` checks for that file. When it exists, the system-library
  module `CMKLight` (`Sources/CMKLight/module.modulemap`, which includes
  `crates/mklight-ffi/include/mklight.h` and links `mklight_ffi`) becomes a
  dependency of `MKLightFFI`, and `FFIEngine` is live. Without it the
  package still builds: `FFIEngine.isLinked` is false, `FFIEngine()` throws
  `EngineError.libraryUnavailable`, and the integration tests are skipped
  with the message "libmklight_ffi.a is not linked: run
  scripts/build-macos.sh --engine-only, then swift test".
* SwiftPM caches the evaluated manifest by its contents, so it does not
  notice the library appearing or disappearing. The script therefore runs
  `swift package purge-cache` whenever the library's presence changes. If
  you add or delete the file by hand, run that command yourself.
* Engine choice (`Sources/MKLight/main.swift`): `FFIEngine` always. A debug
  build without the library falls back to `MockEngine` (an orange "Mock
  engine" label in the status bar and the welcome window). A release build
  without it shows an alert and quits.
* Debug builds accept launch options for screenshots and UI checks:
  `-MKLightNoWelcome YES`, `-MKLightSidebar hidden`, `-MKLightSelectSurface
  <i>`, `-MKLightInspector system`, `-MKLightViewport 3D`, `-MKLightSheet
  optimize|tolerance|solves`, `-MKLightAutoRun lm|de|tolerance`,
  `-MKLightShowIssues YES`. Pane layouts come from the document's `views`.

## `FFIEngine`

`FFIEngine` (module `MKLightFFI`) implements `EngineClient` over the C ABI.
Every C function is wrapped once in `CABI.swift`, which converts strings,
frees what the library allocated and turns a failed `MKStatus` into an
`EngineError` with the engine's message (read from `mk_last_error()` on the
same thread).

| concern | how |
|---|---|
| lifetimes | `FFIEngineCore` owns `MKEngine*`, `FFISessionHandle` owns `MKSession*`; both free their handle in `deinit`. A session and every running job keep the core alive. |
| sessions | `FFIEngine` = core + one session. `makeDocumentEngine()` gives each document window its own session (user catalogs, library hash). `newSession(system:)`, `openProject(at:)`, `save(to:)`, `project()`, `sessionInfo()`, `setProjectSection` cover projects. |
| snapshots | `openSnapshot` sets the session's system (`mk_session_set_system_json`) and returns the engine's BLAKE3 hash and the library hash. An invalid model throws `.invalidModel` with the engine's message first, then the mirror's structural errors. |
| validation | `validate` runs the engine's `validate` analysis (issues with severity, code and surface); a system the engine refuses gives its message plus the mirror's checks. The app runs `validate` as a pane of its own on every snapshot. |
| edits | `applyParamEdits` applies `ParamPath` edits atomically and returns the new snapshot (the edited system read back from the engine). |
| solves | `applySolves(settings:to:)` calls `mk_session_apply_solves` and reads the solved system back; the report is the `solves` result. Engines without it fall back to the `solves` dry run plus `OpticalSystem.setValue`. |
| catalogs | `catalogs()`, `loadCatalog(at:)`, `glasses(in:)`, `materialNames()` (completion). |
| analyses | `analysisDescriptors()` lists every analysis with its settings schema. The app submits requests by name (`EngineAnalysisRequest`): the analysis, the settings the schema accepts and the backend preference. The older typed requests (`AnalysisRequest`) still work for tools and tests. |
| jobs | `submit` returns at once; the job's task does the rest off the caller's thread. Submission is one locked step on a global dispatch queue: set the snapshot's system if the session holds another, then `mk_session_submit_analysis_json`, registering the job's event stream before any event can be delivered. Events from the engine's `mklight-events` thread feed the job's `progress` stream; a 250 ms status poll is a safety net. The terminal event takes the result at once. `cancel()` calls `mk_job_cancel`; a job cancelled before it was submitted never reaches the engine and throws `EngineError.cancelled`. |
| stop | For `optimize` and `tolerance`, `AnalysisJob.canStop` is true and `stop()` calls `mk_job_stop` (ABI 1.2): the job ends normally with its best result so far (`result.stopped: true`). A stop requested before the engine job exists is sent right after submission. |
| results | The envelope's `run_info` decodes into `RunInfo`, `library_hash` into `AnalysisOutput.libraryHash`, `backend` into `AnalysisOutput.backend` (`BackendReport`: requested, used, precision, device, fallback reason). The result is `.generic(GenericResult)`: the JSON tree plus `ResultBuffer`s, read by the views through tolerant accessors (`JSONValue.nf`, `nfArray`, `[path:]`, `AnalysisOutput.buffer(named:)`). |
| buffers | `ResultBuffer` points into the engine's memory (no copy) and keeps the buffer array alive; `mk_buffer_free` runs when the last buffer of a result goes. Typed access checks the dtype (`withUnsafeBufferPointer(of:)`); `buffer[[i, j]]` indexes row-major. |
| errors | `EngineError.from(code:message:)`: cancelled, invalid model, unsupported and failed jobs get their own cases; every other status is `.engine(EngineErrorCode, message)`. Submission errors (unknown analysis, invalid settings) fail the job immediately. |

**Thread safety.** `FFIEngine` is a `Sendable` value; use it from any task.
The C engine and sessions are thread-safe. The `EngineClient` methods never
block their caller: C calls that parse, validate, wait for the session lock
or run jobs go to a global dispatch queue, never the main thread. Only
`mk_session_new` and `mk_session_destroy` run on the caller's thread, when a
window's session is created or released. Progress reaches the UI through
`AsyncStream`s that the `@MainActor` `Workspace` consumes. The engine's
callback never runs on the main thread and holds no engine lock.

## State model

* **Simulation state** (`DesignDocument`, `@Observable`, Foundation and
  Observation only): the `OpticalSystem` value and the project's other
  sections (catalogs, variables, views, metadata, unknown keys). Every edit
  replaces the system with a new value and registers the previous one with
  the undo manager, so undo and redo restore whole design states; edits that
  invalidate configuration overrides, solves, mechanics or variables drop
  them in the same step. Optimisation and solve results replace the system
  as one undoable step (`replace(system:actionName:undoManager:)`).
  Variables are a project section (`[{param, lower, upper}]`, the engine's
  `VariableSpec`) and their changes are undoable too.
* **File** (`ProjectDocument`, `ReferenceFileDocument`): reads and writes the
  engine's project format through `ProjectFile`; SwiftUI's document
  machinery provides autosave in place, versions and the dirty state (it
  follows the undo manager).
* **Engine snapshots** (`Workspace`, per window, `@Observable`,
  platform-neutral): each design value is debounced (150 ms) and frozen
  into an immutable engine snapshot (system hash + glass library hash).
  Analyses run asynchronously against the snapshot they were submitted
  with; a newer request cancels the older job of the same pane; jobs end
  with the window. Results are **stale** as soon as the snapshot, the
  library or the request (settings, backend, seed) differs; panes with
  automatic update re-run, the others wait for Update. Long analyses (stray
  light, ghost irradiance, through-focus and MTF vs field) start manual.
  The project's glass catalogs are loaded before the first snapshot;
  undoing Load Glass Catalog rebuilds the session without it.
* **Window state** (`WindowState`): selection (by `SurfaceID`), inspector
  tab, split fractions, viewport mode and camera, ray colouring, transient
  messages, open sheet. It never enters the design; its layout part is
  saved in `views` (`WorkspaceLayout`).
* **Design jobs** (`OptimizeSession`, `ToleranceSession`, `SolvesSession`):
  view models that submit the engine's `optimize`, `tolerance` and `solves`
  analyses, follow their progress, and apply results to the document.

## Tests

`swift test` in `apps/macos` (163 tests in 23 suites):

* `MKLightModelTests`: round trips against JSON written by the Rust side
  (every model variant, solve kind, samples, the project format), strict
  decoding.
* `MKLightEngineTests`: schemas, request translation, tolerant decoding,
  buffers, error mapping, staleness; the mock's analyses by name.
* `MKLightFFITests` (need the library): BLAKE3 hashes, samples, EFL from
  engine data, non-blocking submission, progress, reproducible seeds,
  cancellation, error codes, projects, parameter edits, catalogs,
  concurrent submissions, handles; and `FFIDesignTests`: **every dedicated
  result view reads the real engine's result** for 25 analyses (geometric,
  wave optics, ghosts, radiometry, polarisation), the layout buffers become
  a scene with rays per field, **Stop** ends a DE optimisation with
  `stopped: true` and its design applies and undoes, and **Apply Solves**
  commits the engine's solved back focus.
* `MKLightUITests`: the workspace (panes run, go stale, automatic vs
  manual update, settings re-run, unavailable and unknown analyses, an
  invalid design blocks analyses, newer requests cancel older ones, the
  layout round-trips through the project's `views`, layout kind codes,
  catalogs before the first snapshot, session rebuild, jobs end with the
  window); editing (table rows, typed values incl. `∞` and materials,
  rejected input, insert / delete / stop with undo and redo, variables with
  undo, solve markers, the data-driven surface parameters and coefficient
  lists, parameter paths); design jobs (merit progress parsing, Optimize
  applied as one undoable step, Stop keeps and Cancel discards, tolerance
  ranking and Monte Carlo, Apply Solves dry run then undoable commit);
  parsers, viewport math and shaders.

## iPadOS readiness

Ready:

* `MKLightModel`, `MKLightEngine` and the view models (`DesignDocument`,
  `Workspace`, `AnalysisPane`, `WindowState`, `WorkspaceLayout`,
  `LensEditor`, `SurfaceTypes`, the design-job sessions, result readers)
  use Foundation and Observation only.
* The window is `NavigationSplitView` + `.inspector` + toolbar items with
  placements + sheets and popovers, which collapse on compact width.
* The split layout (`SplitLayout`, `PaneRow`) is pure SwiftUI with drag
  gestures; only the resize pointer is macOS-specific (`resizePointer`).
* The Metal viewport has one bridge (`ViewportBridge`, `ViewportInput`) and
  two representables: `NSViewRepresentable` with mouse, trackpad and scroll
  input, and `UIViewRepresentable` with pan, two-finger pan, pinch and
  double tap.
* Platform specifics are isolated: dynamic colours (`PlatformColor`), the
  pasteboard (`Clipboard`), the option-click in the sidebar, glass-name
  completion (`materialSuggestions`), file import uses `.fileImporter`.

Remaining for an iPadOS target:

* Add the target (an Xcode project or an iOS platform in `Package.swift`)
  and build the engine as an iOS static library (`aarch64-apple-ios`, plus
  the simulator triple) with the same C module.
* SwiftUI `Table` shows only its first column in compact width: give the
  lens table a compact row layout (surface, type, radius, thickness,
  material, with the rest in the inspector).
* Material completion: `textInputSuggestions` is macOS-only here; use a
  suggestions popover or the inspector's list on iPadOS.
* Keyboard navigation of the table relies on hardware keyboards; add an
  edit mode with swipe actions for insert and delete.
* Hover-only affordances (help tags) need long-press equivalents; the
  welcome window should become the document browser's launch scene
  (`DocumentGroupLaunchScene`).
* Check the Metal viewport's gesture mapping on device and the shader
  compile path (the package ships `Viewport.metal` as source).

## Known gaps

* The lens table's keyboard navigation is implemented with SwiftUI focus
  and key handlers and checked by hand in the app, not by UI tests.
* Merit functions are the engine's default merit generator (criterion,
  pupil sampling, hold EFL); explicit operand editing, bounds on variables
  and tolerance sets are not in the UI yet (the engine accepts them).
* Configurations are listed but not editable; multi-configuration analyses
  use the `configuration` setting where the schema has one.
* Engine 0.1 writes the `layout` kind codes in its enum order while
  `kind_names` lists another order; the reader detects which by where the
  rays are (`EngineLayout.kindTable`). The engine should make them agree.
* Saving goes through the document machinery (atomic write by AppKit), not
  `mk_session_save`, so `relative_path` of catalogs is not recomputed on
  save. Layout changes alone do not mark the document dirty; they are saved
  with the next design change.
* The model mirror is strict: when the engine's model gains a field or a
  surface type, the mirror must learn it (then add a `SurfaceTypes`
  descriptor for a new type).
