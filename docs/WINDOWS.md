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
| `--do <action>[=<arg>]` | run a registered launch action in the design window once its panes have their results, before the screenshot (repeatable, in order); see below |

Launch actions are registered by the features (`AppHooks.LaunchActions`), so a
screenshot can show any state without clicking; an action that opens another
window sets `AppHooks.ScreenshotTarget` so that window is captured:

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
* glass catalogs: `glass-catalogs` and `glass-catalogs-search`, `-scope`, `-filter`,
  `-select`, `-reload`, `-new-glass`, `-edit`, `-add-to-design`.

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
  `Views/Inspector`, the design jobs register theirs; pages are built once per
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
* **Light and dark** are both designed; the theme is System, Light or Dark per
  user (File menu), applied live to every window. The app's own theme
  (`Application.RequestedTheme`) is set at launch from the saved choice or
  `--theme`: WinUI resolves the theme resources of styles applied in code
  (the lens table's cells) at the application level, so they would otherwise
  follow the system's theme. A switch to the other appearance during a run
  therefore completes on the next launch (a toast says so): the app theme is
  not switched live. Brushes picked in code go through `ThemeLookup`, so a
  window's own theme wins (separate panel windows and the glass catalog window
  follow the same choice).

## Window structure

```
┌ ☰ ◉ name  subtitle      [ ⟳ EFL · F/# · TT · ✓ ]        ↶ ↷ │ ⊕▾ 🗎 ◨  ─ □ ✕ ┐
├ sidebar (Mica) ─┬ Lens Data ───────────────────────────────┬ inspector (Mica) ┤
│ Analyses ⌕      │ Surf Type Comment Radius Thickness …      │ Surface System ⚙ │
│ Design          ├──────────────┬──────────────┬────────────┤ ▸ Type, preview  │
│  Lens Data      │ Layout ◉     │ Spot Diagram │ MTF        │ ▸ Parameters     │
│  Layout, Issues │ Section|3D ⤢ │ ⟳ ⚡ ⚙ …      │ ⟳ ⚡ ⚙ …    │ ▸ Aperture …     │
│ Geometric …     │ 20 mm        │ keys  [CPU]  │ keys [CPU] │ (inspector page) │
└─────────────────┴──────────────┴──────────────┴────────────┴──────────────────┘
```

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
  aperture, placement, coating, solves; or the System page's sections).
* **Analysis panes**: title with the analysis menu, a state line (the
  settings that differ from the defaults, or the stage while running), Run /
  Cancel, automatic update, Settings and More (defaults, copy result JSON,
  close); the result as content; key numbers and the backend badge (CPU · f64
  neutral, GPU · f32 in the accent, a warning mark for warnings or a
  fallback) in the footer, which opens the run details. A stale result dims
  under a "Design changed" card with Update; automatic panes re-run.
* **Keyboard**: Ctrl+Z / Ctrl+Y (Ctrl+Shift+Z), Ctrl+S, Ctrl+Shift+S, Ctrl+O,
  Ctrl+N (welcome), F5 run all, Ctrl+Esc cancel all, Ctrl+Alt+P inspector,
  Ctrl+0 fit the layout, plus the lens table's own keys. Features add File menu
  commands and shortcuts through `AppHooks.Commands` (Glass Catalogs…, Add Glass
  Catalog to Design…, Reset Panel Layout).
