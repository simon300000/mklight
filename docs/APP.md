# MKLight for macOS on the Rust engine

The native app (`apps/macos`, SwiftPM) runs on the Rust engine through its C
ABI (`crates/mklight-ffi`, docs/FFI.md). This page covers building, the
Swift binding (`FFIEngine`), the state model and the tests. Package layout,
the lens editor and the Metal viewport are described in
`apps/macos/README.md`.

## Building

```sh
scripts/build-macos.sh --engine-only   # engine library → apps/macos/Frameworks/libmklight_ffi.a
cd apps/macos && swift build && swift test

scripts/build-macos.sh                 # release app → build/macos/MKLight.app
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
* From a clean checkout: run the script (any mode that builds the engine)
  first, then `swift build` / `swift test` work as usual.
* Engine choice (`Sources/MKLight/main.swift`): `FFIEngine` always. A debug
  build without the library falls back to `MockEngine` (orange "Mock
  engine" labels everywhere). A release build without it shows an alert
  and quits.

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
| validation | `validate` runs the engine's `validate` analysis (issues with severity, code and surface); a system the engine refuses gives its message plus the mirror's checks. |
| edits | `applyParamEdits` applies `ParamPath` edits atomically and returns the new snapshot (the edited system read back from the engine). |
| catalogs | `catalogs()`, `loadCatalog(at:)`, `glasses(in:)`. |
| analyses | `analysisDescriptors()` lists every analysis with its settings schema. Typed requests (`AnalysisRequest`) run as `AnalysisKind.engineName(in: descriptors)`: the kind's raw value (`first_order`, `spot_diagram`, …), or for first-order data the `paraxial` placeholder of engines that predate `first_order`. They carry the settings the schema accepts; unavailable analyses fail with `.unsupported` and the engine's reason. Requests by name (`EngineAnalysisRequest`) run any analysis. |
| jobs | `submit` returns at once; the job's task does the rest off the caller's thread. Submission is one locked step on a global dispatch queue: set the snapshot's system if the session holds another, then `mk_session_submit_analysis_json`, registering the job's event stream before any event can be delivered. Events from the engine's `mklight-events` thread feed the job's `progress` stream; a 250 ms status poll is a safety net. The terminal event takes the result at once. `cancel()` calls `mk_job_cancel` with the id the submission recorded; a job cancelled before it was submitted never reaches the engine. A cancelled job throws `EngineError.cancelled`. |
| results | The envelope's `run_info` decodes into `RunInfo`, `library_hash` into `AnalysisOutput.libraryHash`. The result decodes into the typed mirror when it matches; otherwise (and for requests by name) it is `.generic(GenericResult)`: the JSON tree plus `ResultBuffer`s. First-order data is read from the numbers of any JSON layout. |
| buffers | `ResultBuffer` points into the engine's memory (no copy) and keeps the buffer array alive; `mk_buffer_free` runs when the last buffer of a result goes. Typed access checks the dtype (`withUnsafeBufferPointer(of:)`); `buffer[[i, j]]` indexes row-major. |
| errors | `EngineError.from(code:message:)`: cancelled, invalid model, unsupported and failed jobs get their own cases; every other status is `.engine(EngineErrorCode, message)`. `error.code` gives the `MKStatus` back. Submission errors (unknown analysis, invalid settings) fail the job immediately. |

**Thread safety.** `FFIEngine` is a `Sendable` value; use it from any task.
The C engine and sessions are thread-safe. The `EngineClient` methods never
block their caller: C calls that parse, validate, wait for the session lock
or run jobs go to a global dispatch queue, never the main thread. Only
`mk_session_new` and `mk_session_destroy` run on the caller's thread, when a
window's session is created or released. The synchronous project helpers
(`openProject`, `save`, `project`, `system`, `setSystem`, `sessionInfo`, …)
block and are meant for background tasks, tools and tests. Progress reaches the UI
through `AsyncStream`s that the `@MainActor` `AnalysisCoordinator` consumes.
The engine's callback never runs on the main thread and holds no engine
lock, so it cannot deadlock with the UI.

**Tolerance.** JSON readers ignore unknown keys (descriptors, results, run
info; an unknown precision becomes a run-info warning). Unknown analyses and
results that the typed mirror cannot decode are shown by the Engine pane's
generic view (scalars, record tables, an outline, buffer ranges and plots),
never as errors. The settings schema drives a generic settings form.

## State model

* The **document** (`ProjectDocument`) holds the `OpticalSystem` value and
  the other project sections (catalog references, views, merit, metadata,
  unknown keys), written back unchanged in the engine's project format
  (`schema_version`). Every edit replaces the value and registers the old
  one for undo.
* Each window's `AnalysisCoordinator` freezes the debounced system into an
  **immutable engine snapshot** (hash of the system plus hash of the glass
  library) and runs the visible analyses against it, asynchronously, with
  progress and cancellation. Results carry `RunInfo` (backend, precision,
  seed, settings) and are **stale** as soon as the snapshot, the library or
  the request (including backend, precision and seed) differs. A newer
  request cancels the older job.
* The session's user catalogs follow the project's catalog list. On open
  they are loaded **before** the first snapshot (model edits wait), so no
  analysis runs against a library that lacks them. Undoing "Load Glass
  Catalog" rebuilds the session without that catalog, because the engine
  cannot unload one; redo loads it again (`syncCatalogs`).
* A window's jobs end with it: closing the window calls `cancelAll()`, and a
  coordinator that is released cancels its jobs. Engine jobs otherwise
  outlive their session and would occupy the shared workers.
* View state (`WindowViewState`: selection, panes, camera, analysis
  settings, Engine pane settings) never enters the model.

## App features on the engine

* File ▸ New from Sample: the engine's built-in samples
  (`Sources/MKLightModel/Resources/samples.json`, written by
  `cargo run -p mklight-core --example dump_fixtures`).
* Lens ▸ Load Glass Catalog…: loads an AGF (or `GlassCatalog` JSON) file
  into the document's session and records it in the project (undoable).
  On open, the project's catalogs are loaded before any analysis runs
  (relative path first, then absolute); missing ones are shown as warnings
  in the status bar.
* The Engine pane runs any engine analysis by name with a schema-generated
  settings form. Windows start on it when the engine cannot run the
  selected typed analysis.
* Execution menu: backend, precision and (for multi-configuration systems)
  the configuration of every analysis of the window.
* Inspector ▸ Solves: marginal/chief ray height and edge thickness on a
  thickness; marginal/chief ray angle (the paraxial slope, labelled
  "paraxial angle u", CONVENTIONS §8) and F/# on a curvature; pickups and
  other solves on the surface are listed. The engine applies solves.

## Tests

`swift test` in `apps/macos`:

* `MKLightModelTests`: round trips against JSON written by the Rust side
  (`dump_fixtures` in mklight-core, `dump_solve_fixtures` in mklight-trace):
  every model variant, every solve kind, a solved system with
  configurations, solve reports (including `configuration`), samples, and
  the project format. Decoding is strict like the engine's
  `deny_unknown_fields`: an extra key in any of the fixtures' objects is
  rejected.
* `MKLightEngineTests`: schemas, request translation, tolerant decoding,
  buffers, error mapping, staleness (no library needed).
* `MKLightFFITests` (need the library): BLAKE3 hashes equal the engine
  fixtures; every sample opens, validates and is summarised; the achromat's
  paraxial EFL from engine data is 100.0 mm with the image at the paraxial
  focus, and equals the engine's own first-order EFL whenever that analysis
  is available; `submit` does not wait for a busy session lock; progress is monotone and seeded results reproduce; cancellation;
  error codes; project round trips through engine and Swift mirror; atomic
  parameter edits; catalogs change the library hash; concurrent submissions
  keep their snapshots; handles are freed.
* `MKLightUITests`: coordinator behaviour with an engine that has sessions,
  analyses by name, unavailable analyses and catalogs; catalogs load before
  the first snapshot, removing one rebuilds the session, and released
  coordinators cancel their jobs.

## Known gaps

* The engine currently offers `validate`, `system_summary`, `glass_info` and
  `rng_stream`; layout, spot, fans, MTF, PSF and first-order data are listed as
  unavailable, so the viewport and typed tabs show the engine's reason. When
  they land, results that match the typed mirrors are drawn by the existing
  views; others appear in the generic view.
* The C ABI has no call that applies solves; the app stores and edits them,
  and the editor does not yet mark solve-controlled cells.
* Saving goes through the document machinery (atomic write by AppKit), not
  `mk_session_save`, so `relative_path` of catalogs is not recomputed on
  save.
* The model mirror is strict: when the engine's model gains a field, the
  mirror must learn it (regenerate the fixtures; the round-trip tests fail
  until it does). Until then the app reports such a system as unreadable
  instead of dropping the field.
