# Windows parity with the macOS app

Purpose: every feature of the macOS SwiftUI app (`apps/macos`, the reference)
with its status in the native Windows app (`apps/windows`, WinUI 3, design in
`docs/WINDOWS.md`). The list comes from an inventory of the Swift sources; each
row is one inventory item (a few trivially duplicate items are merged). The
statuses describe the end of round 3 and are updated as rounds land.

Legend: ✅ done, 🟡 partial, ⬜ not started. The "Windows notes" column says
where the feature lives or what is missing. "Binding ready" means the engine
binding (`MKLight.Engine`, ABI 1.6) already supports it and only the UI is
missing.

## 1. Window and chrome

| Feature | Status | Windows notes |
|---|---|---|
| Document app: one design window per `.mklight`, default size, minimum size | 🟡 | One `DesignWindow` per design; Mac default/minimum sizes not checked |
| Window title is the document, design name as subtitle | ✅ | Custom title bar (`TitleBar`): file name + system name, dirty marker |
| Three-column layout: sidebar, panel area, full-height inspector with its own toolbar part | ✅ | Sidebar, panel area, full-height inspector column with its own page tab bar |
| Column widths and fixed-bound column layout | ⬜ | |
| Toolbar centre: status capsule | ✅ | Centred in the title bar |
| Toolbar trailing: Add Analysis menu (+) by category | ✅ | Title bar drop-down: categories → analyses, adds a pane after the focused one |
| Toolbar trailing: inspector toggle | ✅ | Button and Ctrl+Alt+P (Ctrl+Alt+I inserts a surface after) |
| Inspector toolbar tab bar (Surface, System, Optimize, Glass Substitution, Tolerance, Apply Solves, running dot) | ✅ | `Views/Inspector/InspectorView.cs`: Surface and System as words, job pages (Optimize, Glass Substitution, Tolerance, Apply Solves) as glyphs, dot on a running job |
| Capsule: GPU preparation badge (Preparing n%, GPU ready) | ✅ | Badge in the capsule (`StatusCapsule.SetGpu`): Preparing n % with a spinner, GPU ready for 3 s; tooltip explains the wait |
| Capsule: one button per running job | ✅ | Running job shown in the capsule (`DesignSession.SetJobActivity`); a click reopens its page |
| Capsule: "Computing" spinner when analyses busy | ✅ | Activity spinner / percent |
| Capsule: key numbers EFL, F/#, TT; stale dim; error / name fallback; anamorphic EFL y and squeeze | 🟡 | EFL · F/# · TT from `first_order`; anamorphic EFL y and squeeze (`StatusCapsuleModel`); stale dimming unchecked; no mock engine |
| Capsule: issues button (checkmark / severity icon and count) | ✅ | Button with count |
| Status popover: First Order table | ✅ | Details flyout lists first-order entries with the "(y)" suffix for anamorphic designs |
| Status popover: anamorphic x/y table, symmetry, DOE power, notes | ✅ | x/y table, symmetry, DOE power and notes in the details flyout (`StatusCapsuleModel`) |
| Status popover: Engine section (version, snapshot, library hash, GPU kernel state) | ✅ | Engine version, snapshot, library hash, GPU kernel state with the cache note |
| Status popover: compute backend picker | ✅ | Picker in the details flyout (`StatusCapsule.SetBackends`) listing the engine's backends, one per GPU by name; choice kept as `ui.compute_backend`; `Workspace.Execution` carries the GPU name |
| Issues popover: list with severity, message, code, chips; click selects surface; empty state | ✅ | Issues flyout; click selects the surface (field highlight ⬜) |
| Transient toast for rejected edits and notices | ✅ | Toast messages |
| No modal alerts for routine errors (revert + toast) | 🟡 | Toast exists; per-field revert rule belongs to lens table / settings form |
| Sidebar "Design": Lens Data, Layout, Issues badge; dot when open; Show/Hide context menu | ✅ | Rows with count badge, dot when open; hide and show from the View menu |
| Sidebar: all engine analyses by category in the Mac order | ✅ | `NavigationView`, `AnalysisCatalog` |
| Sidebar: search (title, name, description), empty-search state | ✅ | |
| Sidebar selection follows focus; click shows in focused pane; new pane; open dot | ✅ | Click, Shift+click or context "Open in New Pane"; dot (accent while running) |
| Unavailable analyses dimmed, engine reason as help | ✅ | Tooltip with reason |
| Short/long titles, icons, categories for ~40 known analyses; unknown get title-case | ✅ | `AnalysisCatalog` with Segoe Fluent Icons |
| Drag and drop of `.mklight/.json/.zmx/.seq` onto a window | ✅ | Drop onto a design window opens the file (`Windows/DesignWindow.Documents.cs`) |
| Per-window focused actions published to menus | ✅ | Menu bar commands act on the focused window (`AppHooks.WindowCommand`, `Menu` null = File) and are disabled by context |
| One number format everywhere (decimal point, no grouping, comma accepted, no "-0.0000") | ✅ | `NumberDisplay`; settings form commit rule |
| Unit display convention | 🟡 | Units in the settings form; other places as they are built |
| Tooltips on controls; accessibility labels and values | 🟡 | Tooltips, automation names and ids on built controls, plot summaries (area 19); the rest as controls are added |

## 2. Menus, commands and keyboard shortcuts

| Feature | Status | Windows notes |
|---|---|---|
| Undo/Redo with action names | ✅ | Buttons with names; Ctrl+Z, Ctrl+Y, Ctrl+Shift+Z; whole-state undo |
| File: New from Sample submenu | ✅ | File > New from Sample, grouped like the welcome window (`Windows/DesignWindow.Menu.cs`) |
| File: Show Welcome Window | ✅ | Ctrl+N |
| File: New/Open/Save/Save As/Open Recent; Revert, Duplicate, Rename, autosave, versions | 🟡 | File menu: New, Open, Open Recent, Save, Save As, Revert to Saved, Duplicate, Rename, autosave (area 18); versions ⬜ |
| View: Sidebar and Inspector commands | ✅ | View menu: Sidebar, Inspector (Ctrl+Alt+P) |
| View: Show/Hide Lens Data, Show/Hide Layout | ✅ | View menu: Show / Hide Lens Data and Layout |
| View > Panel: Move to New/Main Window, Hide/Close, Reset Panel Layout | ✅ | Panel context menus and tab close buttons; Reset Panel Layout in the View menu |
| View: Section, X-Z Section, 3D View, Fit Layout, Ray Colors | ✅ | Section / 3D and Y–Z / X–Z selector and Display menu in the layout; Fit Ctrl+0 |
| Lens menu: Insert Before/After, Delete, Make Aperture Stop, Toggle Variable | ✅ | Lens menu with the table's keys (Ctrl+Shift+I / Ctrl+Alt+I, Ctrl+Delete, Ctrl+Shift+T, Ctrl+Shift+V) |
| Lens menu: Replace with Sample | ✅ | Lens > Replace with Sample, one undoable step |
| Lens menu: Glass Catalogs…, Add Glass Catalog to Design… | ✅ | File menu commands (`Windows/GlassCatalogsRegistration.cs`) |
| Design menu: Optimize, Glass Substitution, Tolerance, Apply Solves, Show Issues | ✅ | Design menu: Optimize, Glass Substitution, Tolerance, Apply Solves (open the inspector pages), Show Issues |
| Analysis menu: New Analysis Pane, Run Analyses, Cancel Running, Compute On | ✅ | Analysis menu: New Analysis Pane, Run Analyses (F5), Cancel Running (Ctrl+Esc), Compute On |
| Job panels: Return is primary action, Escape cancels | ✅ | Return is the primary action, Escape cancels (`Views/Jobs/JobControls.cs`); Ctrl+Enter too |
| Menu items disabled by context | ✅ | Menu bar in the title bar (File, Edit, View, Lens, Design, Analysis, Help); items enabled by context |

## 3. Panels and arrangement

| Feature | Status | Windows notes |
|---|---|---|
| Panels as a tree of H/V splits with tab groups | ✅ | `ViewModels/PanelArrangement.cs`, `Views/Panels/PanelTreeView.cs` |
| Default arrangement: lens data top; layout and panes side by side | ✅ | Hairline dividers |
| Header title is the drag handle | ✅ | `Views/Panels/PanelDrag.cs` |
| Panel context menu (Move to New/Main Window, Hide, Close) | ✅ | Move to New / Main Window, Hide, Close |
| Panel "More" menu in every header | ✅ | Analysis panes (with Duplicate Pane), Lens Data and Layout |
| Tab strip for groups of panels | ✅ | |
| Drop zones: tab, split, edge | ✅ | |
| Drop outside windows opens a separate window; Escape cancels | ✅ | |
| Drops judged against MKLight windows front to back | ✅ | |
| Same-axis merge, draggable dividers, double-click equal shares | ✅ | Splits merge on normalising; dividers drag, double-click equalises |
| Separate windows share the design session | ✅ | `Windows/PanelWindow.cs` |
| Separate window: own tree, toolbar, subtitle | ✅ | |
| Separate window uses document undo, forwards document commands | ✅ | |
| Closing a separate window; Move to Main Window restores places | ✅ | |
| Separate windows persist with frames, moved on-screen | ✅ | Frames saved in `views` |
| Closing a panel: analysis removed; Lens Data / Layout hidden and restorable | ✅ | Analysis pane removed; lens data and layout hidden, shown again from the sidebar |
| Reset Panel Layout | ✅ | File menu command |
| Focused pane: accent hairline, tap focuses, drives sidebar selection | ✅ | |
| Persistence in project `views` (panes, arrangement, viewport, ray colouring, stretch) | ✅ | Panes, settings and arrangement with window frames, the layout's settings, mode, ray colouring and stretch in `views` (`PanelPersistence`, `WorkspaceLayout`); the 3-D camera is not saved (nor on the Mac) |
| Views without arrangement open in the standard one | ✅ | Open in the standard arrangement |

## 4. Lens data table

Checked after the merge of the round-1 table (`Views/Editor`, `ViewModels/LensTable*.cs`).

| Feature | Status | Windows notes |
|---|---|---|
| Header: title, caption "N surfaces · stop on S · mm · V variables", Insert / Delete / More buttons | ✅ | Title, caption "11 surfaces · stop on 6 · mm", "N variables" button (opens Optimize), insert (+), delete (−) and More buttons |
| Columns Surf, Type, Comment, Radius, Thickness, Material, Semi-Diameter, Conic; pinned header; resizable; horizontal scroll | 🟡 | All eight columns, pinned header, vertical scroll; resizable columns saved as `ui.lens_columns` (`ViewModels/LensTableColumns.cs`); no horizontal scroll |
| Rows OBJ, numbered, STO marker, IMA; object/image partly read-only | ✅ | OBJ, numbers, STO with an accent marker, IMA; object/image read-only where the Mac is |
| Alternating rows; selection emphasis only when table focused | ✅ | Stripes; accent tint and pill while the table has focus, neutral otherwise |
| Selection on mouse-down; click edits; double-click row edits radius | 🟡 | Click selects, second click or double click edits; not exercised with a real mouse in round 1 |
| Keyboard: Up/Down, Return, Tab/Shift+Tab, Esc | ✅ | ↑/↓, Home/End, Enter/F2, Tab/Shift+Tab, Esc (checked in-process through the key handlers) |
| Keyboard navigation skips non-editable cells, anamorphic x continues | ✅ | Tab moves y → x on anamorphic radii |
| In-place cell editing: compact at rest, exact on focus, commit on Return/focus loss, rejected reverts with toast | ✅ | One shared editor over the cell; one text-field rule (LensTableFieldEdit), toast on rejection |
| Number parsing: ∞/inf, decimal comma, per-column validation | ✅ | ∞/inf, decimal comma, typographic minus, per-column validation (LensTableValues) |
| Material column: glass-name completion from session catalogs, AIR, MIRROR, model glass | ✅ | Completion from the session's glass names, AIR, MIRROR, model glass |
| Retyping same glass keeps melt offset | ✅ | LensTableEdits keeps the offset of the same glass |
| Melt-offset suffix in the material cell | ✅ | Subtle note after the glass name |
| Type column menu grouped by family, cylinder presets | ✅ | Menu with group headers and Fluent glyphs, current type checked, cylinder presets checked |
| Type title "Sphere" for k=0, plane etc. | ✅ | |
| Anamorphic radius/conic two sections ("y"/"x"), Tab y then x | ✅ | Both sections ("y ∞   x −50"), each editable |
| Cylinder: flat section dimmed, biconic shows both conics | ✅ | Flat section dimmed; biconic shows both conics |
| Automatic semi-diameters dimmed; editing fixes it | ✅ | Engine's resolved value from the layout, dimmed; typing a value fixes it |
| Variable marker V badge (bounded bar, pinned dot, orange when solved) | ✅ | V badges from the project's variables (bounded, pinned); toggle by context menu and Ctrl+Shift+V |
| Solve marker letters M C F P D O E | ✅ | Letters before numbers so digits stay aligned |
| Validation issue badge on the surface label | ✅ | Warning/error glyph in the Surf cell, messages as tooltip |
| Insert before/after, delete (disabled rules), make stop | ✅ | Buttons, context menu, Ctrl+Shift+I / Ctrl+Alt+I, Ctrl+Delete, Ctrl+Shift+T |
| Context menu: Insert, Delete, Make Stop, Variables submenu, Show in Inspector | ✅ | Insert, Delete, Make Stop, Make / Remove Variable, Variables submenu, Show in Inspector |
| Variables menu and toggle refuse a solve-controlled parameter | ✅ | `DesignVariables.Refusal`, shown as a toast |
| Every edit an undoable named step; invalidated solves/overrides dropped in same step | ✅ | DesignDocument.Edit with the Mac action names; whole-state undo |
| Setting type keeps carry-over parameters | ✅ | Model SetShapeKind |
| Table accessibility labels and Edit action | ✅ | Automation names and ids on rows, cells, type button and badges; cell peers with a value pattern (`Views/Editor/LensTableAutomation.cs`) |

## 5. Inspector: Surface page

Built in round 2 (`Views/Inspector/`, `ViewModels/SurfaceTypes.cs`, `ShapeSag.cs`, `InspectorModel.cs`): one data-driven form per surface type descriptor.

| Feature | Status | Windows notes |
|---|---|---|
| Empty state "No Surface Selected" | ✅ | `Views/Inspector/SurfacePage.cs` |
| Title (Object / Image / Surface n) with Stop badge | ✅ | Stop badge |
| Type menu grouped with symbols, type summary | ✅ | `Views/Inspector/SurfacePage.cs` |
| Shape preview per type | ✅ | `Views/Inspector/SurfacePage.cs` |
| Comment field | ✅ | `Views/Inspector/SurfacePage.cs` |
| Data-driven parameter sections with units, V toggle, S badge | ✅ | `ViewModels/SurfaceTypes.cs` descriptors |
| Number fields: formatted at rest, exact on focus, ∞, rejected text reverts | ✅ | `InspectorFields.cs`, one number format |
| Radius displayed as radius of curvature | ✅ | `Views/Inspector/SurfacePage.cs` |
| Coefficient lists with meaning labels, add/remove, Zero Terms | ✅ | `Views/Inspector/SurfacePage.cs` |
| Zernike ordering; XY polynomial degree | ✅ | `Views/Inspector/SurfacePage.cs` |
| Spacing and Medium: thickness, interaction, material, model glass | ✅ | `Views/Inspector/SurfacePage.cs` |
| Melt offset rows | ✅ | Δn_d / ΔV_d rows; no melt glass readout |
| Aperture section: semi-diameter auto/fixed, clipping | ✅ | `Views/Inspector/SurfacePage.cs` |
| Placement section: decenter, tilt, order, return after surface | ✅ | `Views/Inspector/SurfacePage.cs` |
| Coating picker | ✅ | `Views/Inspector/SurfacePage.cs` |
| Solves section: list, add, remove | ✅ | `Views/Inspector/SurfacePage.cs` |
| Diffractive: base, phase coefficients, orders, wavelength | ✅ | `Views/Inspector/SurfacePage.cs` |
| Diffraction efficiency section | ✅ | `Views/Inspector/SurfacePage.cs` |
| Biconic: Make Cylinder | ✅ | `Views/Inspector/SurfacePage.cs` |
| Inspector stays built while hidden | ✅ | Pages built once, kept while hidden |

## 6. Inspector: System page

| Feature | Status | Windows notes |
|---|---|---|
| Name and Notes fields | ✅ | `Views/Inspector/SystemPage.cs` |
| Aperture: type picker + value, conversion via `convert_spec` | ✅ | Conversion through `convert_spec` |
| First Order section for anamorphic designs | ✅ | `Views/Inspector/SystemPage.cs` |
| Fields: type picker, grid X/Y/Weight, add/remove | ✅ | `Views/Inspector/SystemPage.cs` |
| Wavelengths: primary, λ, weight, add/remove, presets | ✅ | `Views/Inspector/SystemPage.cs` |
| Environment: temperature, pressure | ✅ | `Views/Inspector/SystemPage.cs` |
| Ray Aiming toggle | ✅ | `Views/Inspector/SystemPage.cs` |
| Configurations read-only list | ✅ | Read-only list |
| Glass Catalogs section (origins, counts, context menu, warnings, buttons) | ✅ | List with origins and counts, warnings, Glass Catalogs… button |
| Every system edit undoable with named action | ✅ | `DesignDocument.Edit` with named actions |

## 7. Surface types

All 13 types plus the cylinder preset are descriptors in `ViewModels/SurfaceTypes.cs`, shown by the Surface page (sections and previews); the lens table's type menu is in section 4.

| Feature | Status | Windows notes |
|---|---|---|
| Plane | ✅ | Descriptor, form section, round-trip tests |
| Sphere / Conic | ✅ | Descriptor, form section, round-trip tests |
| Even Asphere | ✅ | Descriptor, form section, round-trip tests |
| Odd Asphere | ✅ | Descriptor, form section, round-trip tests |
| Q-Type (Conic) | ✅ | Descriptor, form section, round-trip tests |
| Q-Type (Best-Fit Sphere) | ✅ | Descriptor, form section, round-trip tests |
| XY Polynomial | ✅ | Descriptor, form section, round-trip tests |
| Zernike Sag | ✅ | Descriptor, form section, round-trip tests |
| Toroid | ✅ | Descriptor, form section, round-trip tests |
| Biconic | ✅ | Descriptor, form section, round-trip tests |
| Cylinder preset | ✅ | Descriptor, form section, round-trip tests |
| Diffractive | ✅ | Descriptor, form section, round-trip tests |
| Ideal Lens | ✅ | Descriptor, form section, round-trip tests |
| Coordinate Break | ✅ | Descriptor, form section, round-trip tests |
| Previews: rotational and toroid/biconic profiles | ✅ | `Views/Inspector/ShapePreview.cs` |
| Previews: XY and Zernike profiles and departure map | ✅ | `Views/Inspector/ShapePreview.cs` |
| Previews: DOE phase profile and figures | ✅ | `Views/Inspector/ShapePreview.cs` |
| Changing type drops configuration overrides that no longer resolve | ✅ | Model `SetShapeKind` |
| Variable paths per descriptor | ✅ | V toggle on each parameter row |

## 8. Layout viewport

| Feature | Status | Windows notes |
|---|---|---|
| Engine `layout` scene edge to edge; floating controls | ✅ | Win2D section from the engine's layout result (not Metal) |
| Title chip (drag handle) + More menu | ✅ | Title chip is the drag handle with the panel context menu; More menu (`Layout.More`) |
| Mode selector Section / 3D, Y–Z / X–Z | ✅ | Section / 3D and Y–Z / X–Z (`Views/Layout/LayoutPanel.xaml`, `Views/Viewport3D/Logic/LayoutModes.cs`) |
| Display menu: ray colours, rays per field, wavelengths, fit, stretch, mesh, footprints | ✅ | Ray colouring, rays per field, wavelengths, mesh, footprints, fit all / lens |
| Fit button, Ctrl+0, double-click fits | ✅ | |
| Vertical stretch x2…x50, auto stretch | ✅ | Stretch chip and menu, automatic stretch |
| Stretch menu; scale bar with "vertical xN" | ✅ | Scale bar only |
| Scale bar of a round length, accessibility description | ✅ | Round-length bar; accessibility name with length and stretch |
| Footprints (3D) | ✅ | `Views/Viewport3D/Logic/FootprintOverlay.cs` |
| Surface mesh in 3D | ✅ | Layout setting `mesh_rings` |
| Selected surface highlighted in accent | ✅ | |
| DOE violet hatch, stops and image plane distinct, light/dark ink | ✅ | Glass fills, profiles, stop, image plane, axis, DOE violet dash |
| Camera: pan, zoom about cursor, orbit, double-click fit | 🟡 | Drag pans (orbits in 3-D), wheel zooms about the pointer, double-click fits, pinch zooms; orbit, wheel zoom, pinch and drops checked in-process only (real mouse gestures cannot be posted to WinUI); UI Automation tests cover the rest |
| Section views of toleranced builds (markNearSection) | ✅ | Near-section marking in `MKLight.Engine/LayoutScene.cs` (`MarkNearSection`) |
| "Design changed" card when stale | ✅ | Click updates |
| Preview label "Previewing … · not applied" | ✅ | Accent chip "Previewing … · not applied" on the layout |
| Placeholder, "Viewport Unavailable", accessibility description | ✅ | Placeholder, "Viewport Unavailable" and the scene's accessibility description |
| Layout as a pane of `layout` with settings (rays_per_field, wavelengths, mesh_rings), re-run on change | ✅ | `rays_per_field`, `wavelengths`, `mesh_rings` from the Display menu, re-run on change |
| Reduce Motion replaces animations | ✅ | `Support/Motion.cs`: Windows' animation setting; moves, fades and pulses become instant |
| Click selects surface | ✅ | Click in the layout selects the surface |

## 9. Analysis panes (generic chrome)

| Feature | Status | Windows notes |
|---|---|---|
| Header: glyph + title (drag handle), chevron analysis menu by category | ✅ | Glyph, title (drag handle) and category menu |
| State line: settings differing from defaults, or the stage | ✅ | |
| Glass cluster: Run / Cancel, auto-update, Settings toggle, More | ✅ | |
| More menu: Update Automatically, Show Settings, Duplicate Pane, Copy Result as JSON, Close Pane | ✅ | Update Automatically, Show Settings, Duplicate Pane, restore defaults, copy result JSON, close pane |
| Narrow-pane behaviour: folding cluster, short titles | ✅ | The cluster folds into More and the title shortens (`AnalysisPaneChrome.Fold`) |
| Tabbed pane | ⬜ | |
| Settings panel: Defaults, Done, open state saved | ✅ | Defaults and Done; open state saved as `ui.settings_visible` |
| Footer key numbers (subscripts, spoken form) | ✅ | Key numbers in the footer; VoiceOver forms ⬜ |
| Backend badge (CPU · f64 / GPU · f32, duration, warning, fallback) | ✅ | Opens run details |
| Run details: backend, requested, reason, precision, seed, snapshot, engine, time, failed attempts, Route, Tier | ✅ | Route line (`cost_decision`) and tier added (`Views/Analysis/RunDetails`); unit-tested, not checked on a real GPU fallback |
| Warnings grouped, numbers masked, fields/wavelengths listed, raw messages | ✅ | `RunDetails/Model/RunWarnings.cs` |
| One-click remedies (auto pupil grid, image grid, PSF window) | ✅ | Three remedies (auto pupil grid, image grid, PSF window) change the pane's settings |
| Stale result: dimmed, "Design changed" card with Update | ✅ | |
| Updating: progress line, pulse | ✅ | Progress bar and pulse (still under Reduce Motion) |
| Placeholders: running, Not Available, Design Has Errors, Failed, Not Ready, Cancelled | ✅ | `PanePlaceholderKind`: running, unavailable, design has errors, failed, not ready, cancelled |
| Failed rerun over existing result: warning capsule | ✅ | Warning capsule over the kept result (`AnalysisPaneChrome`) |
| Automatic vs manual update; long analyses start manual; newer cancels older | ✅ | |
| Debounce from edit to snapshot; stale on snapshot/library/request difference | ✅ | `Workspace` debounced snapshots, `IsCurrent` |
| Manual retry asks for a fresh run | ✅ | Run and retry ask for a fresh run (`Rerun(fresh: true)`) |
| Unknown engine analyses appear and run with generic view | ✅ | |

## 10. Analysis settings form

| Feature | Status | Windows notes |
|---|---|---|
| Two-column form, small controls, spectrum control last | ✅ | Spectrum control last |
| Controls from schema: number with units/bounds, toggle, enum, text | ✅ | One number format commit rule |
| Field / wavelength / surface menus and index settings | ✅ | Single index |
| Integer-list settings as check-mark menus | ✅ | Check-mark menus for fields, wavelengths and surfaces (`Settings/SettingsForm.MultiSelect.cs`) |
| Two-number arrays as pair fields | ✅ | |
| Number lists; row-array popover tables | ✅ | Number lists and row-array tables (`Settings/Model/TableSetting.cs`) |
| Object settings as grouped rows with on/off | ✅ | |
| Tagged unions as variant picker + rows | ✅ | |
| Spectrum: kind menu | ✅ | `Views/Analysis/Settings/SpectrumControl.cs`: presets |
| Spectrum: blackbody, samples, spacing, range | ✅ | |
| Spectrum: explicit list editor | ✅ | |
| Spectrum: CSV import | ✅ | Sensor CSV (`Settings/Model/SpectrumCsv.cs`) |
| Spectrum: preview chart | ✅ | `SpectrumPreview.cs` |
| Spectrum summary in pane state line | ✅ | `SettingsSummary.cs` |
| Pupil sampling control (Grid with Auto, Rays) | ✅ | `SettingsForm.Special.cs` |
| MTF frequency list with automatic state | ✅ | Automatic state shown as "Auto (n)" |
| Spot through-focus control | ✅ | |
| Image simulation: source menu, optics, FOV/centre/resolution | ✅ | Source pattern, optics, imported image, FOV, patch centre and resolution |
| Sensor CRA limit table | ✅ | Table in a popup |
| Settings with no sensible control hidden, values kept | ✅ | `ControlKind.Hidden`, values kept |
| Auto settings show "Auto (n)" with last value | ✅ | |
| Configuration selector when design has configurations | ✅ | Shown only when the design has configurations |
| Spot settings: pattern-dependent controls | ✅ | Seed shown only for patterns that use it |

## 11. Dedicated result views

| Feature | Status | Windows notes |
|---|---|---|
| Generic view: record tables, buffer previews, JSON outline | ✅ | Scalars, tables, curves, heat maps, raw JSON |
| First Order view (table, anamorphic card, DOE card) | ✅ | Table, anamorphic card, DOE card (`TableViews.cs`) |
| First Order labels wrap, with a tooltip per entry | ✅ | `TableViews.cs`, `FirstOrderHelp` |
| Validation view (issue list, "No Issues") | ✅ | `Results/ValidationView.cs`: issue list, "No Issues" |
| Spot Diagram (field × focus grid, per-plane RMS, Airy, common/per-field scale) | ✅ | |
| Ray Fan / OPD Fan | ✅ | |
| Field Curvature & Distortion | ✅ | |
| Grid Distortion | ✅ | Generic view only |
| Seidel (bars, Seidel/Chromatic switch) | ✅ | Seidel / Chromatic bars (`SeidelView.cs`) |
| Longitudinal Aberration | ✅ | |
| Chromatic Focal Shift | ✅ | |
| Lateral Color | ✅ | |
| Relative Illumination (curve) | ✅ | |
| Footprint | ✅ | Generic view only |
| Encircled Energy (geometric and diffraction) | ✅ | |
| Geometric MTF / MTF | ✅ | |
| Best Focus | ✅ | Curve |
| MTF vs Field; Through-Focus MTF | ✅ | |
| Wavefront Map (map picker, Zernike table) | ✅ | Map and Zernike table (`WaveViews.cs`) |
| PSF (item picker, log scale, Airy) | ✅ | Field picker, linear / log, core / full; see the two PSF rows below |
| PSF opens on a log scale when the Strehl ratio is below 0.1 | ✅ | `Readers/WaveReaders.cs` (`PrefersLog`); the core view crops to 95 % of the energy |
| Line Spread | ✅ | |
| Ghost Paths | ✅ | Generic view only |
| Ghost Irradiance | ✅ | Sensor map view |
| Stray Light | ✅ | Sensor map view |
| Polarization Pupil Map | ✅ | Generic view only |
| System Transmission | ✅ | Generic view only |
| Pupil Aberration, RI Map, Chief Ray Angle: shared field-map reader | ✅ | `FieldMapViews.cs` |
| Heat-map scales (diverging, viridis, 0–1, hatched nulls) | ✅ | `HeatScale` in `HeatMap.cs` |
| RI map iso-lines, key numbers | ✅ | Generic view only |
| CRA vs image height with sensor limit | ✅ | Sensor limit from the run's settings |
| Glass Map | ✅ | `GlassMapView.cs` |
| Glass Data | ✅ | `GlassDataView` |
| Image Simulation: compare modes, exposure, zoom, pan, drop image | ✅ | Side by side, split, blink, exposure, zoom, pan, drop an image (`ImageSimulationView*.cs`) |
| Image simulation controls: FOV, patch centre, resolution, picker | ✅ | FOV, patch centre, resolution and image picker in the pane (`ImageSimulationView.Controls.cs`) |
| Image simulation footer and field-mapping table | ✅ | Pixel, patch, PSF window and Field Mapping flyout |
| Image simulation starting settings, source drawing, imported images | ✅ | Starting settings, built-in source patterns and imported images (picker and drop; 512 px per side; a saved view keeps only the path) |
| Shared chart components (line chart, legend, colour bar, sparkline, key/value grid) | ✅ | `PlotKit`, `Legend`, heat-map colour bar, `Plotting/Sparkline.cs` |
| Design \| Candidate switch on spot and line charts while a candidate is previewed, key numbers say which design | ✅ | `Results/CompareSwitch.cs`, `SpotDiagramView.cs`, `LineChartView.cs` |
| Sparkline | ✅ | `Plotting/Sparkline.cs`, used by the job running views |
| Palette: categorical slots, spectral colours, dashes, ramps | ✅ | `Palette` (the Mac's validated colours) |

## 12. Design jobs: common

Built in round 2 (`Views/Jobs`, `ViewModels/Jobs`) on the engine binding's job submit, stop (best so far), cancel and `ApplySolvesAsync`.

| Feature | Status | Windows notes |
|---|---|---|
| Job panel structure: header, body, action bar | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Action bar states (setup, running, review, done, failed) | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Failure page and "Applied to the design" notice | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Done returns job to its starting page | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Jobs keep running when the panel closes; capsule reopens | ✅ | Capsule activity, a click reopens the page |
| Jobs end when the main window closes | ✅ | `DesignJobs.CancelAll` on window close |
| Apply is one undoable step | ✅ | Variables that vanished are dropped in the same step |

## 13. Optimize

| Feature | Status | Windows notes |
|---|---|---|
| Merit Function box (Default/Custom, live readout, Convert to Custom, hold focal length) | ✅ | Default / Custom, live readout, Convert to Custom, hold focal length |
| Default merit: criterion, reference, MTF options, rings/arms, fields/wavelengths, limits, squeeze | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Custom merit: operand list and searchable picker | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Operand editor | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Operand list actions (add, delete, duplicate, move, reorder, multi-select) | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Live merit evaluation (200 ms) | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Merit edits undoable, saved in `merit` section | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Variables group ("Vary" choice) | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Variables and bounds list | ✅ | Radius windows, pin and problems (`VariablesBoundsList.cs`) |
| Algorithm group (method, iterations, seed) | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Running view (iteration, merit, sparkline) | ✅ | Iteration, merit, log sparkline |
| Result: before/after, preview in the window, variable list | ✅ | Preview titled with the merit, spot and layout thumbnails before / after on one scale, variables before → after (`Views/Jobs/DesignFigures.cs`, `LayoutComparison.cs`) |
| "Fix the design's errors first" and blocked start | ✅ | `Views/Jobs`, `ViewModels/Jobs` |

## 14. Glass Substitution

| Feature | Status | Windows notes |
|---|---|---|
| Glasses That May Change (checkbox per gap) | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Per-slot override button | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Candidates for every lens (catalogs, status filter, rule, limits) | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Search (Quick/Thorough, max combinations, time, iterations, keep N) | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Merit/variables group, Edit Merit and Variables… | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Running view, Stop keeps ranking, Cancel discards | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Ranked results list | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Candidate detail (glasses, shape changes, layouts, glass map) | ✅ | Glasses, shape changes, merit, layouts and a glass map (`Views/Jobs/CandidateGlassMap.cs`) |
| Preview selected row; Apply as one undoable step | ✅ | `Views/Jobs`, `ViewModels/Jobs` |

## 15. Tolerance

| Feature | Status | Windows notes |
|---|---|---|
| "Show the last results" shortcut | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Tolerances table grouped by element | ✅ | Grouped by element; groups collapse |
| Row kinds | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Row editing (limits, surface menu, unit conversion, problems) | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Row context menu (unit, distribution, symmetric, duplicate, delete) | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Add menu (surface and element sections) | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Default Tolerances… generator | ✅ | Generator flyout |
| Test fringes wavelength field | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Element tolerances (move together, pivot, range) | ✅ | Move together, pivot and range |
| Compensators | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Criterion section | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Analysis section (mode, trials, seed, percentiles) | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Running view (stage, elapsed, time left; Stop/Cancel) | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Results (nominal, RSS, percentiles, yield, Export menu) | ✅ | Nominal, RSS, levels, yield and an Export menu (CSV) |
| Sensitivity table | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Monte Carlo histogram | 🟡 | Histogram with clickable bins and worst trials; cumulative curve and level labels ⬜ |
| Compensator results | ✅ | Compensator values with range bars |
| Preview card and layout caption | ✅ | Preview of a build or trial with its caption |
| Tolerance spec saved in `tolerances` section | ✅ | `Views/Jobs`, `ViewModels/Jobs` |

## 16. Apply Solves

| Feature | Status | Windows notes |
|---|---|---|
| Empty state "No solves" | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Dry run list (target, kind, previous → value, residual) | ✅ | `Views/Jobs`, `ViewModels/Jobs` |
| Caption and Apply to Design as one undoable commit | ✅ | `Views/Jobs`, `ViewModels/Jobs` |

## 17. Glass catalogs

Built in round 2 on the engine binding (ABI 1.5: remove / restore / reload catalog,
add library catalog, glass overrides, glass details, `mk_glass_from_spec`,
`mk_catalog_to_agf`, library sync, glass rows).

| Feature | Status | Windows notes |
|---|---|---|
| Separate Glass Catalogs window, follows key design window or own session | ✅ | `GlassCatalogsWindow`, follows the key design window |
| Sidebar of catalogs with counts and context menus (remove, new glass, rename, export AGF, trash, reload) | ✅ | `Windows/GlassCatalogs*`, `Views/Glass/**`, `ViewModels/GlassLibrary*.cs`, `GlassBrowser*.cs` |
| Info bar for selected catalog | ✅ | `Windows/GlassCatalogs*`, `Views/Glass/**`, `ViewModels/GlassLibrary*.cs`, `GlassBrowser*.cs` |
| Add menu (new glass, new user catalog, add catalog/folder to library or design) | ✅ | `Windows/GlassCatalogs*`, `Views/Glass/**`, `ViewModels/GlassLibrary*.cs`, `GlassBrowser*.cs` |
| Table: sortable columns, search, count footer | ✅ | `Windows/GlassCatalogs*`, `Views/Glass/**`, `ViewModels/GlassLibrary*.cs`, `GlassBrowser*.cs` |
| Filter menu (hide obsolete, preferred only, usable in substitution) | ✅ | `Windows/GlassCatalogs*`, `Views/Glass/**`, `ViewModels/GlassLibrary*.cs`, `GlassBrowser*.cs` |
| Marks: user-set status, excluded from substitution | ✅ | `Windows/GlassCatalogs*`, `Views/Glass/**`, `ViewModels/GlassLibrary*.cs`, `GlassBrowser*.cs` |
| Row context menu (use for surface, copy, status, exclude, reset, duplicate) | ✅ | Use for Surface, copy, status, exclude, reset, duplicate |
| Detail column (properties, index curve, formula, resistances) | ✅ | `Windows/GlassCatalogs*`, `Views/Glass/**`, `ViewModels/GlassLibrary*.cs`, `GlassBrowser*.cs` |
| Reload sheet listing added/removed/changed glasses | ✅ | `Windows/GlassCatalogs*`, `Views/Glass/**`, `ViewModels/GlassLibrary*.cs`, `GlassBrowser*.cs` |
| User glass editor sheet (model or AGF formula, live n_d V_d) | ✅ | `Windows/GlassCatalogs*`, `Views/Glass/**`, `ViewModels/GlassLibrary*.cs`, `GlassBrowser*.cs` |
| User catalogs: create, rename, export AGF, trash | ✅ | `Windows/GlassCatalogs*`, `Views/Glass/**`, `ViewModels/GlassLibrary*.cs`, `GlassBrowser*.cs` |
| App-wide library persisted, loaded into every design, re-synced | ✅ | `GlassLibrary` synced into every session (`AppHooks.SessionStarting`) |
| Statuses and exclusions as app-wide overrides | ✅ | `Windows/GlassCatalogs*`, `Views/Glass/**`, `ViewModels/GlassLibrary*.cs`, `GlassBrowser*.cs` |
| Portability: project records only used library catalogs | ✅ | Saved project records the library catalogs in use |
| Add Glass Catalog to Design… picker; undoable load/removal | ✅ | Add and remove catalogs are undoable steps |
| Vendor catalog files never written | ✅ | Catalog files are read only; user catalogs and AGF export write their own files |

## 18. Documents, welcome and import

| Feature | Status | Windows notes |
|---|---|---|
| `.mklight` open/save; `.zmx` and `.seq` imported, never written | ✅ | Atomic write; Save As asks for `.mklight` |
| Open via File > Open, Welcome, drag and drop, file associations | 🟡 | File > Open, welcome, drag and drop; file associations ⬜ |
| Welcome window (New Design prominent, Open, recent files, drop hint) | ✅ | New Design, Open, recent files, drop hint |
| Sample cards with engine-computed layout thumbnails; Surface Types group | ✅ | |
| Mock engine mark in the welcome window | ⬜ | Not ported by design: no mock engine |
| Import report sheet (groups, Copy Report, Load Glass Catalog) | ✅ | `Windows/ImportReport*.cs`: groups, Copy Report, Load Glass Catalog |
| Autosave, versions; dirty state follows undo | 🟡 | Autosave by the app's own writer, recovery at launch (`Windows/AutosaveRecovery.cs`); dirty follows undo; versions ⬜ |
| Save stamps metadata and library-catalog references | ✅ | Metadata stamp and library-catalog references (`DesignDocument`) |
| Project sections preserved (system, catalogs, variables, views, merit, tolerances, metadata, unknown keys) | ✅ | Unknown sections kept verbatim |
| Undo/redo naming for all edits across windows | ✅ | Whole-state undo; one window |
| Replace with Sample (undoable) | ✅ | Lens > Replace with Sample |

## 19. Appearance and accessibility

| Feature | Status | Windows notes |
|---|---|---|
| Light and dark both designed; semantic colours; dynamic palette | ✅ | System / Light / Dark in the File menu, switched live in every window (`Support/ThemeSwitch`); `--do theme=…` |
| Liquid Glass chrome | ✅ | Translated to Mica / acrylic on chrome only |
| Reduce Motion | ✅ | `Support/Motion.cs` |
| Reduce Transparency / Increase Contrast | 🟡 | High-contrast dictionary uses system colours (`Theme.xaml`, `ThemeLookup`), not screenshotted; Reduce Transparency ⬜ |
| Tabular figures everywhere | ✅ | |
| Spacing scale | ✅ | `Space*` resources |
| Accessibility labels/values on plots, drawings, markers, cells, panes | ✅ | Automation names and ids on panes, cells and markers; plot summaries; the layout's scene description |
| Physical wavelength colours; colour never sole carrier | ✅ | Palette plus dashes |
| Help tags with shortcut hints | 🟡 | On built controls |

## 20. Engine integration

| Feature | Status | Windows notes |
|---|---|---|
| FFIEngine always; missing library handling | ✅ | `FFIEngine` over `mklight_ffi.dll`; missing DLL shows an explanatory window; mock engine not ported by design |
| GPU kernel preparation at launch with cache directory | ✅ | `%LOCALAPPDATA%\MKLight\GPU Kernels`; App Nap n/a |
| "GPU kernels" row texts (Preparing, Ready, No GPU, cache unavailable) | ✅ | State in the details flyout and the capsule badge |
| Backend preference per design (Automatic / CPU / GPU) | ✅ | Picker in the status details flyout |
| Descriptors, schemas, availability read once at engine creation | ✅ | |
| Per-design session; snapshot and library hash; validate on every snapshot | ✅ | |
| Glass library and project catalogs synced before first snapshot | ✅ | `AppHooks.SessionStarting`, `Workspace.LoadProjectCatalogsAsync` |
| Preview mechanism for candidate designs with "not applied" labels | ✅ | `Workspace.Preview` / `Baseline`; strip on every pane and the layout, badge and End Preview in the capsule; spot and line charts have a Design | Candidate switch |
| `convert_spec`, `glass_info` cache, `merit_evaluation`, `merit_spec`, `apply_solves`, `preview_case` | ✅ | `convert_spec` (System page), `glass_info`, `merit_evaluation`, `merit_spec`, `apply_solves`, `preview_case` (jobs) |
| Job streaming progress, Stop vs Cancel | ✅ | Used by analyses and the design jobs (stop keeps the best so far, cancel discards) |
| Engine is the authority: no optics in the UI | ✅ | Display formulas only where the Mac has them |

## 21. Debug launch options

Windows has its own set (docs/WINDOWS.md "Launch options"): `--sample`, `--analysis`, `--settings`, `--show-settings`, `--theme`, `--select`, `--edit`, `--undo`, `--save-as`, `--size`, `--screenshot`, `--delay`, `--welcome`, `--sidebar`, `--show-issues`, `--toast`, `--support-dir`, `--fit`, `--curvature-bounds-unit` and `--recover-autosaves`, plus a file argument, and `--do <action>[=arg]` for actions features register (inspector, panels, layout, settings, run details, results, glass catalogs).

| Mac option | Status | Windows equivalent |
|---|---|---|
| `-MKLightNoWelcome YES` | ✅ | A file argument or `--sample` skips the welcome window; `--welcome` shows it |
| `-MKLightSidebar hidden` | ✅ | `--sidebar hidden|shown` |
| `-MKLightSelectSurface <i>` | ✅ | `--select <i>` |
| `-MKLightInspector system` | ✅ | `--do inspector-system` / `inspector-surface` |
| `-MKLightViewport 3D` | ✅ | `--do layout-3d` |
| `-MKLightSheet` (optimize, tolerance, solves, glass) | ✅ | `--do job=<optimize|glass_substitution|tolerance|solves>` |
| `-MKLightAutoRun`, `-AutoPick`, `-AutoStop`, `-AutoApply` | 🟡 | `optimize-run`, `optimize-running=<s>`, `optimize-apply`, `glass-run`, `glass-apply`, `solves-run`; no auto-stop |
| `-MKLightToleranceTrials`, `-MKLightTolerancePreview` | ✅ | `--do tolerance-run[=trials]`, `tolerance-preview=<k|sN>` |
| `-MKLightShowIssues YES` | ✅ | `--show-issues` |
| `-MKLightToast <text>` | ✅ | `--toast <text>` |
| `-MKLightAppearance light` or `dark` | ✅ | `--theme light` or `dark` |
| `-MKLightFootprints`, `-MKLightFit lens`, `-MKLightYaw` | ✅ | `--do layout-footprints`, `layout-yaw`, `layout-pitch`, `--fit lens|all` |
| `-MKLightGlassCatalogs`, `-GlassScope`, `-GlassSearch` | ✅ | `--do glass-catalogs`, `-search`, `-scope` |
| `-MKLightSupportDirectory <path>` | ✅ | `--support-dir <path>` |
| `-MKLightRunDetails <analysis>` | ✅ | `--do run-details` |
| `-MKLightAutoEdit`, `-MKLightAutoConvert` (merit editor) | 🟡 | `--do merit-custom`, `add-solve`, `variables=`; `--edit` for lens edits |
| `-MKLightCurvatureBoundsUnit curvature` | ✅ | `--curvature-bounds-unit radius|curvature` |
| Test render hook `MKLIGHT_RENDER_DIR` | 🟡 | `--screenshot <file.png>` renders the whole window (as composed), plus `--size`, `--delay` |

## Summary

| Area | Done | Partial | Not started | Total |
|---|---|---|---|---|
| 1. Window and chrome | 25 | 5 | 1 | 31 |
| 2. Menus, commands and keyboard shortcuts | 14 | 1 | 0 | 15 |
| 3. Panels and arrangement | 20 | 0 | 0 | 20 |
| 4. Lens data table | 24 | 2 | 0 | 26 |
| 5. Inspector: Surface page | 20 | 0 | 0 | 20 |
| 6. Inspector: System page | 10 | 0 | 0 | 10 |
| 7. Surface types | 19 | 0 | 0 | 19 |
| 8. Layout viewport | 19 | 1 | 0 | 20 |
| 9. Analysis panes (generic chrome) | 19 | 0 | 1 | 20 |
| 10. Analysis settings form | 23 | 0 | 0 | 23 |
| 11. Dedicated result views | 41 | 0 | 0 | 41 |
| 12. Design jobs: common | 7 | 0 | 0 | 7 |
| 13. Optimize | 13 | 0 | 0 | 13 |
| 14. Glass Substitution | 9 | 0 | 0 | 9 |
| 15. Tolerance | 18 | 1 | 0 | 19 |
| 16. Apply Solves | 3 | 0 | 0 | 3 |
| 17. Glass catalogs | 17 | 0 | 0 | 17 |
| 18. Documents, welcome and import | 8 | 2 | 1 | 11 |
| 19. Appearance and accessibility | 7 | 2 | 0 | 9 |
| 20. Engine integration | 11 | 0 | 0 | 11 |
| 21. Debug launch options | 15 | 3 | 0 | 18 |
| **Total** | **342** | **17** | **3** | **362** |

## Known issues

Not features of the Mac app, so not counted above.

* **Interactions are verified through UI Automation patterns** by the opt-in
  `MKLight.UITests` (15 tests, `MKLIGHT_UI_TESTS=1`). Real mouse gestures cannot
  be posted to WinUI windows, so 3-D orbit, wheel zoom and drop-to-split are
  checked in-process only; panel drags, pinch and divider resizing likewise.
* **Versions and file associations** are not done (areas 2, 18).
* **High contrast** uses the system colours but was not screenshotted; Reduce
  Transparency is not handled.
* **Left out of round 3**: the Monte-Carlo histogram's cumulative curve and level
  labels; the lens table's horizontal scroll; a tabbed pane; stale dimming of the
  capsule's key numbers; hover on the candidate glass map.

## Needs engine changes

None open. The four items found in round 3 are in the engine since ABI 1.6 and
adopted in round 4:

* The CPU-f32 emulation backend registers as `gpu:cpu_f32` with
  `register_emulation` (`--gpu-emulation`).
* `glass_map` leaves infrared glasses out and lists them in `omitted`, which the
  glass map shows beside the plot with the reason.
* The glass details index curve spans the glass's own range.
* `mk_session_set_project_json` can carry every project section into
  `mk_session_autosave`. The app keeps its own autosave writer: Save uses the same
  writer, and the engine would write the session's catalog list rather than the
  document's catalog references (with their relative paths).

## Design notes for later rounds

* **Previewing an unapplied candidate (optimizer, glass substitution,
  tolerance trial).** On the Mac, users found it confusing in the spot diagram
  which design was shown while a candidate was previewed. On Windows it must be
  unmistakable, everywhere the preview reaches:
  * Every pane header and the layout carry a labelled badge and tint, for
    example an accent-tinted capsule "Previewing Candidate 3 · not applied" (or
    "Tolerance trial 12 · not applied"), not just a caption under the controls.
    The label is also in the pane's accessibility name.
  * The spot diagram (and other plots where it makes sense) shows baseline
    versus candidate: a "Design | Candidate" segmented switch, and optionally
    the baseline as faint ghost points or a second column on the same scale.
    The key numbers state which design they belong to.
  * The status capsule and the window title also mark an active preview, and
    one "End Preview" button is always visible; applying or ending the preview
    removes every mark at once.
* Known gaps in the Swift app, which do not require parity (not counted
  above): configurations are listed but not editable; pickups have no editor
  (only the solve marker); tolerancing has no element tilt z, no irregularity
  orientation, and radius perturbations in ΔR/% scale linearly from the limit;
  layout-only changes do not mark the document dirty; library entries store
  plain paths (no bookmarks) and renaming a user catalog does not rewrite
  designs; an imported image for image simulation is limited to 512 px per side
  as JSON buffers.
