# MKLight for Windows

The native Windows app (`apps/windows`) is a WinUI 3 app on .NET 11 that runs
on the Rust engine through its C ABI (`crates/mklight-ffi`, docs/FFI.md), like
the macOS app (docs/APP.md). It is built in rounds toward parity with the Mac
app; `apps/windows/PARITY.md` tracks every feature of the Swift app with its
status. This page covers the build, the architecture, the engine binding and
how the Mac app's design brief was translated to Fluent.

## Platform

* Windows 10 1809 (build 17763) or later, x64 or ARM64; designed and checked
  on Windows 11.
* .NET SDK 11 (`apps/windows/global.json`), C# latest.
* Windows App SDK 2.7 (WinUI 3), **unpackaged and self-contained**: the app
  carries the Windows App SDK and the .NET runtime, so nothing has to be
  installed to run it. Win2D 1.4 draws the plots, the section view and the
  inspector's previews; Direct3D 11 through Vortice 3.8 draws the 3-D layout view.
* The engine: `mklight_ffi.dll`, the `cdylib` of `crates/mklight-ffi` (the
  crate also builds its `staticlib`, which the macOS app links, and its
  `rlib`).

## Building

```powershell
scripts/build-windows.ps1                        # engine DLL + release app -> build/windows/MKLight/MKLight.exe
scripts/build-windows.ps1 -EngineOnly            # engine DLL only -> apps/windows/native/x64/mklight_ffi.dll
scripts/build-windows.ps1 -Configuration Debug   # engine + debug build of the solution
scripts/build-windows.ps1 -Test                  # ... and dotnet test
scripts/build-windows.ps1 -NoEngine              # skip cargo (the DLL is already in place)
```

* The script builds `mklight-ffi` in release (`cargo build --release -p
  mklight-ffi`, or `--target aarch64-pc-windows-msvc` with `-Platform ARM64`)
  and copies the DLL to `apps/windows/native/<x64|ARM64>/` (git-ignored, one folder per platform). Every project
  that needs it (the app, the tests) copies it next to its output when it is
  there (`MKLightNativeDir` in `apps/windows/Directory.Build.props`).
* Release: `dotnet publish` of `MKLight.App` into `build/windows/MKLight`,
  self-contained. Debug: `dotnet build MKLight.slnx -p:Platform=x64`.
* Tests: from `apps/windows`, `dotnet test --project
  tests/MKLight.Tests/MKLight.Tests.csproj` (xUnit v3 on Microsoft Testing
  Platform). Without the DLL the integration tests are skipped with
  "mklight_ffi.dll not found: run scripts/build-windows.ps1 -EngineOnly, then
  dotnet test".
* UI tests (opt-in, skipped otherwise): build the solution, then from
  `apps/windows` set `MKLIGHT_UI_TESTS=1` and run `dotnet test --project
  tests/MKLight.UITests/MKLight.UITests.csproj` (`MKLIGHT_APP` overrides the exe;
  failures leave a capture and the automation tree in `ui-artifacts/`); see its
  `README.md`.
* Without the DLL the app opens a window that says the engine is missing and
  how to build it; there is no mock engine.
* `scripts/build-macos.sh` is unchanged: it builds the same crate and copies
  only `libmklight_ffi.a`; the added `cdylib` type also leaves a
  `libmklight_ffi.dylib` in `target/<triple>/release`, which nothing links
  (the Swift package links from `apps/macos/Frameworks`, where only the
  static library is copied).

### Launch options

For self-checks and screenshots without clicking (the Mac app's debug launch
options):

| option | effect |
|---|---|
| `file.mklight` / `.zmx` / `.seq` | open that design |
| `--sample <id>` | a new design from a built-in sample (`double_gauss`, `cooke_triplet`, …), no welcome window |
| `--analysis <name>[,<name>…]` | open these analysis panes instead of the defaults (Spot Diagram, MTF) |
| `--settings <json>` | settings merged into every pane opened with `--analysis` |
| `--show-settings` | open every pane's settings form |
| `--theme light\|dark` | override the saved appearance for this run |
| `--select <i>` | select surface *i* |
| `--edit <i>.<radius\|thickness\|conic>=<value>` | apply a lens edit once the panes are up (repeatable), one undoable step each |
| `--undo` | then undo once |
| `--save-as <file>` | then save the design there |
| `--size <w>x<h>` | window size in effective pixels, even beyond the display |
| `--screenshot <file.png>` | once every pane has its result, render the window (as composed: Mica, acrylic, Win2D) to a PNG and quit |
| `--delay <s>` | wait longer before the screenshot |
| `--welcome` | show the welcome window even with a file or `--sample` |
| `--sidebar hidden\|shown` | hide or show the sidebar |
| `--show-issues` | open the status capsule's issues list |
| `--toast <text>` | show a toast in the design window |
| `--support-dir <path>` | the support folder for this run: glass library, autosaves (`MKLIGHT_SUPPORT_DIRECTORY`) |
| `--fit lens\|all` | fit the layout to the lens or to everything |
| `--curvature-bounds-unit radius\|curvature` | how variables bound curvatures, for this run |
| `--recover-autosaves` | offer to recover autosaved designs in a scripted run too |
| `--gpu-emulation` | register the engine's CPU-f32 emulation of the GPU kernels as `gpu:cpu_f32` (`register_emulation`; also `MKLIGHT_GPU_EMULATION=1`), so the GPU paths can be tried without a GPU; `auto` never picks it |
| `--do <action>[=<arg>]` | run a registered launch action in the design window once its panes have their results, before the screenshot (repeatable, in order); see below |

Launch actions are registered by the features (`AppHooks.LaunchActions`), so a
screenshot can show any state without clicking; an action that opens another
window sets `AppHooks.ScreenshotTarget` so that window is captured:

* menus and documents: `menu=<Menu>[/<Item>]` (opens that menu bar menu or submenu),
  `revert`, `duplicate`, `rename=<name>`, `replace-sample=<id>`, `autosave`,
  `recover-autosave`, `import-catalog=<file>`;
* inspector: `inspector-surface[=i]`, `inspector-system`, `inspector-scroll=<px>`,
  `inspector-set=<row>:<value>`;
* panels: `panels-demo`, `panels-detach=<panel>`, `panels-drag=<panel>@fx,fy[@window]`,
  `panels-drop`, `panels-hide=<panel>`, `panels-show=<panel>`, `panels-reset`,
  `panels-capture=<window>`;
* layout: `layout-3d`, `layout-xz`, `layout-yz`, `layout-mesh`, `layout-footprints`,
  `layout-stretch=<n>`, `layout-yaw=<deg>`, `layout-pitch=<deg>`,
  `layout-coloring=field|wavelength`, `layout-capture`;
* settings and run details: `spectrum`, `spectrum-csv`, `settings-scroll`,
  `run-details`, `remedy`, `popup`;
* results: `result-view`, `image-simulation-start`, `dump-results`;
* design jobs: `job=<id>` (`optimize`, `glass_substitution`, `tolerance`, `solves`; opens the page),
  `variables=c1,c3,t11` (make those values variables), `optimize-run[=algorithm]`,
  `optimize-running=<s>`, `optimize-apply`, `merit-custom`, `add-solve`, `solves-run`,
  `glass-run[=rank]`, `glass-apply`, `tolerance-run[=trials]`, `tolerance-preview[=k|sN]`;
* glass catalogs: `glass-catalogs` and `glass-catalogs-search`, `-scope`, `-filter`,
  `-select`, `-reload`, `-new-glass`, `-edit`, `-add-to-design`;
* results: also `result-view=design|candidate|pick` (the Design | Candidate switch),
  `image-simulation-import=<file>`, `-fov=<deg>`, `-center=hx,hy`, `-size=<px>`,
  `-zoom=<n>`, and `settings-open-multiselect=<setting>`;
* design jobs: also `tolerance-bin[=k[:t]]`, `tolerance-pivot[=last|s<i>|clash]`,
  `tolerance-export=<file.csv>`, `job-figures`, `job-scroll`;
* panes: `pane-preview`, `pane-more`, `pane-state=<state>` (a placeholder or phase),
  `pane-motion=<state>`;
* lens table: `lens-columns=<widths>`, `lens-fit`, `lens-more`, `lens-type-menu=<row>`,
  `lens-menu=<row>`, `lens-inspect=<row>`, `lens-set=<cell>:<text>`, `lens-automation=<cell>`;
* layout: also `layout-more`, `layout-fit-lens`, `layout-placeholder`, `layout-unavailable`;
* appearance: `reduce-motion` (Motion as if the system setting were on);
* compute and GPU: `simulate-gpu-preparation[=pct|ready|cache-warning]`,
  `simulate-backends[=file]` (a backend list as the engine returns it),
  `compute-backend=<name>`, `status-details` (opens the capsule's flyout),
  `simulate-gpu-results`, `run-details=<pane index>`.
* timing: `bench-panes=N[:backend]` runs each open pane, one at a time, once untimed
  and then N times, each after a unique tiny edit of the design (the first curved
  surface's curvature changed by one to two parts per billion, so the engine reuses
  nothing), on `cpu_reference`, `auto`, `gpu`, `cuda` or `gpu:<name>` (default: the
  window's backend). Other panes wait meanwhile; the design and backend are put back
  and the document is never edited. One JSON line per run goes to the file
  `bench-out=<file>` names (default `bench-<time>.jsonl` beside `mklight.log`):
  `pane`, `analysis`, `run`, `backend_requested`, `backend_used`, `engine_ms`
  (run_info), `device` (per GPU execution `wall_ms`, `device_ms` and its stages as
  `[name, device_ms, wall_ms]`), `app_view_ms` (submit to the pane's view rebuilt) and
  `app_frame_ms` (to the next frame after that), or `error`.

Without a file or `--sample` the welcome window opens (`--screenshot` then
captures it). The log is `%LOCALAPPDATA%\MKLight\mklight.log`.

## Architecture

```
apps/windows/
  MKLight.slnx, Directory.Build.props, global.json, .editorconfig
  src/MKLight.Model/    net11.0   the model JSON mirror (no UI, no optics)
  src/MKLight.Engine/   net11.0   IEngineClient + FFIEngine over mklight_ffi.dll
  src/MKLight.App/      net11.0-windows  the WinUI 3 app
  tests/MKLight.Tests/  xUnit v3: Model and Engine, FFI integration on the real DLL
```

The layering mirrors the Swift package (`MKLightModel`, `MKLightEngine` +
`MKLightFFI`, `MKLightUI`), so later rounds port view models file by file.

* **`MKLight.Model`** — immutable records mirroring the engine's
  `OpticalSystem` JSON exactly (tagged unions with custom converters,
  snake_case, explicit nulls, `"inf"` thicknesses, curvature not radius, every
  surface shape with its parameter paths and term numbering), `ParamPath`
  get/set, solves, materials, edits (insert, delete, stop, shape kind) that
  return a new system or null when refused, structural validation, the
  `.mklight` project file (unknown sections kept verbatim), the samples and
  the one number format (`NumberDisplay`). Records and value collections
  (`ValueList`, `ValueMap`) compare by value, so the app keeps whole design
  states for undo. Tested against the engine-written fixtures in
  `apps/macos/Tests/MKLightModelTests/Fixtures`, read in place.
* **`MKLight.Engine`** — the engine boundary. Systems cross it as JSON text,
  so it does not depend on the model; the app glues the two
  (`ModelJson.Encode`). See "`FFIEngine`".
* **`MKLight.App`**:
  * `ViewModels/` — `DesignDocument` (the system and project sections; every
    edit replaces the system and pushes the previous value, so undo and redo
    restore whole states; dirty tracking by revision), `Workspace` (debounced
    snapshots, panes, staleness, validation issues), `AnalysisPaneModel`,
    `DesignSession` (document + workspace + selection; the lens table's
    `ILensDataHost`), the inspector's models (`SurfaceTypes`, `ShapeSag`,
    `InspectorModel`), the panel tree (`PanelArrangement`, `WorkspaceLayout`)
    and the glass library and browser (`GlassLibrary*`, `GlassBrowser*`).
  * `Windows/` — `DesignWindow`, `PanelWindow` (a separate panel window),
    `GlassCatalogsWindow`, `WelcomeWindow`, `EngineMissingWindow`.
  * `Views/` — the status capsule, the lens data table (`Editor/`), the
    layout section (`Layout/`: `EngineLayout` projects the engine's scene,
    `SectionRenderer` draws it for the pane and the welcome thumbnails), the
    3-D view (`Viewport3D/`), the inspector (`Inspector/`: pages, form,
    shape previews), the panel host (`Panels/`), the glass catalog views
    (`Glass/`), the design jobs' registration (`Jobs/`), the analysis pane
    chrome (`Analysis/`) with its result views (`Results/`), settings form
    (`Settings/`) and run details (`RunDetails/`).
  * `Plotting/PlotKit.cs` — Win2D plot primitives: theme-aware `PlotStyle`,
    text with tabular figures, nice ticks in the one number format, line
    plots with solid and dashed series.
  * `Support/` — `Palette` (the Mac app's validated colours), `AnalysisCatalog`
    (titles, glyphs, categories), `Fmt`, `ThemeLookup`, `AppSettings` and
    recent files (`%LOCALAPPDATA%\MKLight`), launch options, window capture,
    and `AppHooks`.

### Integration seams

Features plug in from their own files (a `[ModuleInitializer]` registers them),
so none edits `App.xaml.cs`, `LaunchOptions.cs` or `DesignWindow` to be added:

* **`InspectorPages`**: the inspector's page registry (id, title or glyph,
  order, factory per design window). Surface and System are registered by
  `Views/Inspector`, the design jobs register theirs (`Views/Jobs`); pages are built once per
  window and kept while hidden. `DesignSession.ShowInspectorPage` selects one.
* **`AppHooks`**: `LaunchActions` (`--do`), `Commands` (File menu items with
  shortcuts), `WindowCreated` (per design window), `SessionStarting` (before a
  session's project catalogs and first snapshot; the glass library syncs here)
  and `ScreenshotTarget`.
* **`DesignDocument`**: `EditSection(key, value, name)` and `EditProject` change
  project sections as undoable named steps (every section except `views`, which
  `SetSection` writes without an undo step).
* **`Workspace`**: `Preview(candidate, title)` and `Baseline` show an unapplied
  candidate on every pane and the layout (strip, badge, End Preview in the status
  capsule) until the next design edit or `Preview(null)`; `SubmitDesignJob` runs a
  job on the design's own snapshot, never on a preview; `LoadProjectCatalogsAsync`
  loads a project's catalogs into its session and `CatalogsDidChangeAsync`
  refreshes after any catalog change.
* **`DesignSession`**: `LayoutScene` (the engine's layout, shared by the section
  and the 3-D view), `ShowInspectorPage`, `SetJobActivity` (a running job's text in
  the capsule and a dot on its inspector tab). A window disposes the engine
  session it owns, never the shared engine.
* **Native DLL**: `apps/windows/native/<x64|ARM64>/` (`MKLightNativeDir` in
  `Directory.Build.props`), so the two platforms cannot be mixed up.

### Round 2 features

* **Inspector** (`Views/Inspector`, `ViewModels/SurfaceTypes.cs`). The Surface
  page is built from a descriptor per surface type (all 13 plus the cylinder
  preset): type menu with summary, shape preview, parameter and coefficient
  sections, spacing and medium with the melt offset, aperture, placement,
  coating and solves; every edit is one undoable named step. The System page
  has name and notes, aperture and field kinds (converted through `convert_spec`),
  fields, wavelengths with presets, environment, ray aiming, configurations
  (read-only) and the glass catalogs. A tab bar with glyphs for the design jobs
  and a running dot sits above the pages.
* **Panels** (`Views/Panels`, `ViewModels/PanelArrangement.cs`). The panes, lens
  data and layout are a tree of splits and tab groups with draggable dividers.
  Dragging a panel header shows drop zones (tab, split, window edge); dropping
  outside the windows opens a separate window (`PanelWindow`) on the same design
  session. Context menus move, hide and close; Reset Panel Layout is in the File
  menu. The arrangement and window frames are saved in `views` and restored on open.
* **3-D layout** (`Views/Viewport3D`, `Views/Layout`). A Direct3D 11 view (Vortice)
  in a `SwapChainPanel`, fed by the same `LayoutScene` as the Win2D section view.
  The layout has a Section / 3D and Y–Z / X–Z selector, a Display menu (ray
  colouring, rays per field, wavelengths, mesh, footprints, fit), vertical
  stretch, and a label while a candidate is previewed.
* **Result views** (`Views/Analysis/Results`). Dedicated views for first order,
  Seidel, PSF, wavefront, the field maps (pupil aberration, relative illumination
  with iso-lines, chief ray angle with the sensor limit), glass map and data,
  image simulation (side by side, split, blink, exposure), grid distortion,
  footprint, ghost paths and irradiance, stray light, polarization pupil map and
  system transmission, besides the earlier plots; an unreadable result falls back
  to the generic view.
* **Settings and run details** (`Views/Analysis/Settings`, `RunDetails`). The
  spectrum control (presets, blackbody, explicit list, sensor CSV, chart), pupil
  sampling, automatic MTF frequencies, the sensor limit and row tables, image
  simulation source and optics; run details with the route line and tier,
  grouped warnings and one-click remedies.
* **Glass catalogs** (`Windows/GlassCatalogs*`, `Views/Glass`,
  `ViewModels/GlassLibrary*`, `GlassBrowser*`). An app-wide library synced into
  every session, and a window with the catalog sidebar, info bar, searchable
  sortable table with filters, detail with the index curve, reload report, user
  glass editor, user catalogs and AGF export; Use for Surface; adding and removing
  catalogs is undoable and a saved project records the library catalogs it uses.
  The vendor files are never written.
* **Design jobs** (`Views/Jobs`, `ViewModels/Jobs`). The four job pages in the
  inspector share one frame with action-bar states (setup, running, review, done,
  failed); a running job shows in the status capsule (a click reopens its page)
  and ends if the window closes. Apply is one undoable step that drops variables
  that no longer exist. Optimize: Default or Custom merit (operand editor,
  Convert to Custom, live evaluation), variables and bounds, algorithm, running
  statistics with a sparkline, and a result previewed on every pane with spot
  before / after and the variables before and after. Glass Substitution: glasses
  that may change, candidates, search, ranked results, preview and apply.
  Tolerance: rows, generator, compensators, criterion, sensitivities, Monte-Carlo
  histogram and previews of builds and trials. Apply Solves: dry-run list and apply.
  The lens table's variable toggle (context menu, Ctrl+Shift+V) and its "N variables"
  button feed the Optimize page. Documents open `.zmx` / `.seq` prescriptions
  off the UI thread.

### Round 3 features

* **Result views and image simulation** (`Views/Analysis/Results`, `Plotting/Sparkline.cs`).
  A PSF opens on a log scale when its Strehl ratio is below 0.1 and crops to the
  core holding 95 % of the energy; First Order labels wrap and carry tooltips; a
  validation view lists the issues. While a candidate is previewed the spot and
  line charts have a Design | Candidate switch and key numbers that say which design
  they belong to. Image simulation takes an imported image (picker or drop, kept in
  a saved view by path only, at most 512 px per side as engine buffers) with zoom,
  pan, FOV, patch centre and resolution. Integer-list settings (fields,
  wavelengths, surfaces) are check-mark menus; plots have accessibility summaries.
* **Design-job leftovers** (`Views/Jobs`). Optimize and Glass Substitution results
  show spot and layout thumbnails before and after on one scale, and a candidate's
  glass map; the tolerance table's groups collapse, element rows have pivot and
  range, the Monte-Carlo bins are clickable, compensators show range bars, results
  export to CSV and a preview has a caption.
* **Menus and documents** (`Windows/DesignWindow.Menu.cs`, `DesignWindow.Documents.cs`).
  A menu bar (File, Edit, View, Lens, Design, Analysis, Help) sits in the title bar;
  items are enabled by context and per-window commands go through
  `AppHooks.WindowCommand` (`Menu` null means File). Documents can be reverted,
  duplicated, renamed and replaced with a sample, and files open by drag and drop.
  Autosave is the app's own writer, the one Save uses (round 4 kept it after ABI 1.6
  added `mk_session_set_project_json`: the engine would write the session's catalog
  list, not the document's catalog references); autosaved designs are offered for
  recovery at launch. A lens file's import report opens as a sheet (groups, Copy Report, Load
  Glass Catalog). File associations and versions are not done.
* **Compute backends and GPU readiness** (`Views/StatusCapsule*`, `Windows/DesignWindow.Status.cs`).
  The capsule's flyout picks the compute backend from the engine's list (one entry per
  GPU, by name), kept as `ui.compute_backend`; `Workspace.Execution` carries the choice
  and the GPU name. The capsule shows "Preparing GPU n %" and "GPU ready" while the
  kernels are prepared, and its flyout lists the first order with the anamorphic
  x/y table, symmetry, DOE powers and the library hash. IR glasses work in the glass
  views.
* **Panes, panels, layout, lens table and accessibility.** Pane chrome has a More
  menu with Duplicate Pane, folds its buttons into it on a narrow pane, pulses while
  updating, shows placeholders for every phase and a warning capsule when a rerun
  fails over a kept result; the layout and lens table have More menus, the layout
  a placeholder, a "Viewport Unavailable" state and pinch zoom; lens table
  columns resize (`ui.lens_columns`) and cells expose automation peers. Reduce
  Motion (`Support/Motion.cs`) follows Windows' animation setting and makes moves,
  fades and pulses instant. High contrast switches the brushes to system colours
  (`Theme/Theme.xaml`, `ThemeLookup`). Controls carry automation names and ids
  (`Layout.More`, cell ids, pane ids) so scripts and UI Automation find them.
* **UI Automation tests** (`tests/MKLight.UITests`, FlaUI.UIA3 on xUnit v3, opt-in).
  Each test launches the built app with launch options (`--sample double_gauss
  --size 1600x1000 --theme light`) and a scratch `MKLIGHT_SUPPORT_DIRECTORY`;
  settings and recent files follow that folder too (`AppSettings.SupportDirectory`),
  so a test never touches the user's. Input rule: UI Automation patterns plus
  window-targeted messages only, never global input (no `SendInput`, no real mouse
  or keyboard, no screen capture). Key finding: WinUI 3 ignores window-targeted
  mouse and pointer messages (it reads the system pointer pipeline), so the
  posted-gesture tests skip and name their UI Automation counterparts. 15 tests
  pass: lens selection, editing a cell (EFL follows), undo and redo, row menus,
  sidebar and Add Analysis, Run on a stale pane, the settings form, the File menu,
  moving a pane to a new window, the 3-D switch, inspector tabs. Not covered: 3-D
  orbit, wheel zoom, drop-to-split (checked in-process only). `MKLight.Tests` and
  `MKLight.App.Tests` run without a window.

### Round 4 (ABI 1.6, CUDA)

* The app needs ABI 1.6. The glass map plots only glasses with an n_d and V_d;
  the ones the engine leaves out (`omitted`, infrared glasses) are listed beside
  the plot under their reason, the design's own first ("S10 · not on the map"),
  and the key numbers count them. The glass details index curve spans the glass's
  own range, so the partial-range note is gone.
* `--gpu-emulation` (or `MKLIGHT_GPU_EMULATION=1`) creates the engine with
  `register_emulation`: the picker lists "GPU emulation on the CPU · f32", and a
  pane run on it is badged "GPU · f32 · cpu_f32". The GPU kernel cache follows the
  support folder, so a `--support-dir` run never touches the user's cache.
* Checked on an RTX 4090 with the release engine (CUDA kernels built in): the
  picker lists "CUDA GPU (NVIDIA GeForce RTX 4090) · f32"; the spot diagram on it
  is badged "GPU · f32 · cuda"; under Automatic the run details read "auto: the
  CPU reference is predicted faster for this workload" for the default spot
  diagram; the capsule's kernel row reads "Ready (45, loaded from cache)". The
  kernels ship inside the DLL, so preparation finishes before the first pane is
  drawn and "Preparing GPU n %" does not appear on this machine, even with an
  empty cache (`--do simulate-gpu-preparation` still shows it).
* Spot diagram captions use one unit per grid (µm unless a radius reaches 1 mm). In a
  pane too narrow for every focus plane at a legible cell (58 px) the grid folds to
  its base plane and its note says so; below 420 px the Common | Per field toggle
  goes under the legend, so the wavelength legend keeps its width.
* Ray Grid (`ray_grid`, raw rays the Mac app leaves out of its analysis list and the
  sidebar here too, reachable with `--analysis ray_grid`) has its own view
  (`RayGridView`): one density image per field instead of overplotted points, each
  bin's colour mixing the wavelengths by count and its opacity following the log of
  the count, centred on the field's centroid, with "arrived / launched".
* An invalid `--settings` (not a JSON object, e.g. quotes a shell stripped) is a
  parse error: the app logs it and exits with code 2. A failure while a design
  window starts is logged, and a scripted run exits with code 3 instead of waiting.

## `FFIEngine`

`FFIEngine` implements `IEngineClient` over the C ABI with the semantics of
the Swift binding (docs/APP.md "`FFIEngine`"):

| concern | how |
|---|---|
| binding | every C function is a source-generated `[LibraryImport("mklight_ffi")]`; returned strings are read as UTF-8 and freed with `mk_string_free`; a failed `MKStatus` becomes an `EngineException` with `mk_last_error()` read on the same thread. A test parses `mklight.h` and checks status codes, dtypes, job states, the ABI major and every bound function name. |
| lifetimes | `EngineHandle` and `SessionHandle` are `SafeHandle`s; a session holds a reference on its engine, so the engine is destroyed after its last session; a forgotten handle is released by its finalizer. |
| sessions | `MakeDocumentEngine()` gives every design its own session (user catalogs, library hash); the welcome thumbnails share one more. |
| snapshots | `OpenSnapshotAsync(systemJson)` sets the session's system and returns the engine's BLAKE3 hash and the library hash; a refused system throws `InvalidModel` with the engine's message, and the app adds the model mirror's structural problems, which name their surface. |
| jobs | `Submit` returns at once. Setting the snapshot's system and submitting is one locked step off the caller's thread; the job's channel is registered before the engine can call back. The engine's callback (an `[UnmanagedCallersOnly]` function on its `mklight-events` thread, user data a `GCHandle` to the event hub) only writes to channels; a 250 ms status poll is the safety net; the terminal event takes the result at once. `Cancel()` calls `mk_job_cancel`; a job cancelled before submission never reaches the engine; `Stop()` (stoppable analyses) calls `mk_job_stop`, sent right after submission when asked earlier. Cancelling the wait does not cancel the job. |
| results | `AnalysisOutput`: the result JSON, `RunInfo`, the library hash, the `BackendReport` (with `cost_decision` as sent) and the buffers; `IsCurrent(snapshot, request)` drives staleness. |
| buffers | `ResultBuffer` points into the engine's memory (no copy): typed spans with a dtype check, shape, row-major indexers. All buffers of a result share one owner; `mk_buffer_free` runs once, when the last buffer is released or collected. |
| threading | long calls (set system, submit, take result, validate, solves, catalogs) run on the thread pool under a per-session lock; cancel, stop and GPU preparation run on the caller's thread. The app's `Workspace` lives on the UI thread; `await` returns to it. |
| GPU kernels | the app calls `PrepareGpu(%LOCALAPPDATA%\MKLight\GPU Kernels)` at launch; the status capsule's details show the state. |

## Look: the design brief in Fluent

The Mac app's brief (docs/APP.md "Design brief", "Design principles") is
about *where* material goes and *what* colour means, so it translates
directly; only the materials change.

* **Glass → Mica and acrylic, only on chrome.** The window's backdrop is Mica;
  the sidebar, the title bar and the inspector column sit on it. Floating
  controls (the layout's title chip, its Section/3D and display clusters, the
  stale card, toasts) are capsules on in-app acrylic. Content is never glass:
  the lens table, the layout and the plots share one opaque surface
  (`ContentSurfaceBrush`, the Mac palette's plot surface `#FCFCFB` / `#1A1A19`)
  with a hairline border and rounded top corners, divided by hairlines.
* **Accent only with meaning**: selection, the focused pane (a thin accent
  line on top), active toggles (automatic update), the GPU badge and the one
  prominent action of a view (New Design on the welcome window, Update on a
  stale pane). Everything else is neutral.
* **Type**: Segoe UI Variable (Text, and Display for large titles) at 13 px
  body; **every number uses tabular figures** (`Typography.NumeralAlignment`
  in the `NumberText` styles, the `tnum` feature in Win2D text). One number
  format everywhere: decimal point, no grouping, whatever the region (the
  development machine runs a Chinese locale).
* **Shapes**: capsules for floating controls and the status capsule, circles
  for icon buttons (`CircleButton`), 8 px rounded cards. Segoe Fluent Icons
  stand in for SF Symbols, chosen per analysis (`AnalysisCatalog`).
* **Spacing**: the Mac's scale (2, 4, 6, 8, 12, 16, 24) as `Space*` resources.
* **Colour in plots**: the validated categorical `Palette` (light and dark
  steps of eight hues), fields by slot, wavelengths by their spectral slot,
  series also differing by dash; light grids and few ticks.
* **No status bar**: activity, EFL, F/#, total track and the issues button live
  in the status capsule, centred in the title bar; engine, snapshot, backend
  and GPU kernels are in its flyout; transient messages are a toast.
* **High contrast and Reduce Motion**: with a Windows high-contrast theme the
  brushes come from the system colours (`Theme/Theme.xaml`, `ThemeLookup`); with
  "Animation effects" off, `Support/Motion.cs` replaces animations (panel moves,
  fades, the updating pulse) by instant changes.
* **Light and dark** are both designed; the theme is System, Light or Dark per
  user (File menu), applied live to every window. The app's own theme
  (`Application.RequestedTheme`) can only be set at launch, from the saved
  choice or `--theme`. A window root given another theme before it is loaded
  keeps the app's theme for styles outside the app's dictionaries (a
  control's own resources, dictionaries loaded at run time), so every window
  takes its theme from `Support/ThemeSwitch` once its root is loaded, and
  switches are live; System follows the system's light or dark. Brushes picked in code go through `ThemeLookup`, so a
  window's own theme wins (separate panel windows and the glass catalog window
  follow the same choice).

## Window structure

```
┌ ◉ File Edit View Lens Design Analysis Help │ name  [ ⟳ EFL · F/# · TT · ✓ ]  ↶ ↷ │ ⊕▾ ◨  ─ □ ✕ ┐
├ sidebar (Mica) ─┬ Lens Data ───────────────────────────────┬ inspector (Mica) ┤
│ Analyses ⌕      │ Surf Type Comment Radius Thickness …      │ Surface System ⚙ │
│ Design          ├──────────────┬──────────────┬────────────┤ ▸ Type, preview  │
│  Lens Data      │ Layout ◉     │ Spot Diagram │ MTF        │ ▸ Parameters     │
│  Layout, Issues │ Section|3D ⤢ │ ⟳ ⚡ ⚙ …      │ ⟳ ⚡ ⚙ …    │ ▸ Aperture …     │
│ Geometric …     │ 20 mm        │ keys  [CPU]  │ keys [CPU] │ (inspector page) │
└─────────────────┴──────────────┴──────────────┴────────────┴──────────────────┘
```

* **Menu bar** (title bar, left): File, Edit, View, Lens, Design, Analysis, Help;
  the same commands as the shortcuts, enabled by context (`DesignWindow.Menu.cs`).
* **Sidebar** (`NavigationView`): Design (Lens Data, Layout, Issues with a
  count badge), then every engine analysis by category in the Mac app's
  order, searchable (title, name and description), unavailable analyses
  dimmed with the engine's reason as tooltip, open analyses marked with a dot
  (accent while running). A click shows the analysis in the focused pane;
  Shift+click or the context menu opens a new pane.
* **Content**: lens data across the top, then the layout and the analysis
  panes side by side in the standard arrangement; the user changes it by
  dragging a panel by its header and dropping it as a tab, a split or a window
  edge, or moving it to a separate window.
* **Inspector** (right column, Mica): a tab bar (Surface and System as words,
  the design jobs as glyphs with a dot while running) over the selected page,
  a form of sections (type and preview, parameters, spacing and medium,
  aperture, placement, coating, solves; or the System page's sections); the
  column is hidden and shown from the View menu (Ctrl+Alt+P).
* **Analysis panes**: title with the analysis menu (also the drag handle), a
  state line (the settings that differ from the defaults, or the stage while
  running), Run / Cancel, automatic update, Settings and a More menu (update
  automatically, show settings, duplicate pane, defaults, copy result JSON, close;
  the buttons fold into it on a narrow pane); Lens Data and Layout have More menus
  too; the result as content; key numbers and the backend badge (CPU · f64
  neutral, GPU · f32 in the accent, a warning mark for warnings or a
  fallback) in the footer, which opens the run details. A stale result dims
  under a "Design changed" card with Update; automatic panes re-run.
* **Keyboard**: Ctrl+Z / Ctrl+Y (Ctrl+Shift+Z), Ctrl+S, Ctrl+Shift+S, Ctrl+O,
  Ctrl+N (welcome), F5 run all, Ctrl+Esc cancel all, Ctrl+Alt+P inspector,
  Ctrl+0 fit the layout, plus the lens table's own keys. Features add menu
  commands and shortcuts through `AppHooks.Commands` (Glass Catalogs…, Add Glass
  Catalog to Design…, Reset Panel Layout in View).
