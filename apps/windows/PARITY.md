# Windows parity with the macOS app

Purpose: every feature of the macOS SwiftUI app (`apps/macos`, the reference)
with its status in the native Windows app (`apps/windows`, WinUI 3, design in
`docs/WINDOWS.md`). The list comes from an inventory of the Swift sources; each
row is one inventory item (a few trivially duplicate items are merged). The
statuses describe the end of round 2 (design jobs pending: areas 12–16 and the
lens-table variable rows are not updated yet) and are updated as rounds land.

Legend: ✅ done, 🟡 partial, ⬜ not started. The "Windows notes" column says
where the feature lives or what is missing. "Binding ready" means the engine
binding (`MKLight.Engine`, ABI 1.5) already supports it and only the UI is
missing. The lens data table was built by a separate agent in round 1: its rows
are marked 🟡 "round 1 (lens table agent): verify" until their status is checked
after the merge.

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
| Inspector toolbar tab bar (Surface, System, Optimize, Glass Substitution, Tolerance, Apply Solves, running dot) | ✅ | `Views/Inspector/InspectorView.cs`: Surface and System as words, job pages as glyphs, dot on a running job; job pages are placeholders (areas 12–16) |
| Capsule: GPU preparation badge (Preparing n%, GPU ready) | 🟡 | State shown only in the details flyout, no badge |
| Capsule: one button per running job | 🟡 | Running jobs show as activity text in the capsule (`DesignSession.SetJobActivity`); no per-job buttons |
| Capsule: "Computing" spinner when analyses busy | ✅ | Activity spinner / percent |
| Capsule: key numbers EFL, F/#, TT; stale dim; error / name fallback; anamorphic EFL y and squeeze | 🟡 | EFL · F/# · TT from `first_order`; anamorphic variants and stale dimming ⬜; no mock engine |
| Capsule: issues button (checkmark / severity icon and count) | ✅ | Button with count |
| Status popover: First Order table | 🟡 | Details flyout lists first-order entries; "(y)" suffix for anamorphic ⬜ |
| Status popover: anamorphic x/y table, symmetry, DOE power, notes | ⬜ | |
| Status popover: Engine section (version, snapshot, library hash, GPU kernel state) | 🟡 | Engine version, snapshot, GPU kernel state shown; cache-unavailable line and library hash unchecked |
| Status popover: compute backend picker | ✅ | Picker in the details flyout (`StatusCapsule.SetBackends`) |
| Issues popover: list with severity, message, code, chips; click selects surface; empty state | ✅ | Issues flyout; click selects the surface (field highlight ⬜) |
| Transient toast for rejected edits and notices | ✅ | Toast messages |
| No modal alerts for routine errors (revert + toast) | 🟡 | Toast exists; per-field revert rule belongs to lens table / settings form |
| Sidebar "Design": Lens Data, Layout, Issues badge; dot when open; Show/Hide context menu | 🟡 | Rows with count badge; hide from sidebar ⬜ |
| Sidebar: all engine analyses by category in the Mac order | ✅ | `NavigationView`, `AnalysisCatalog` |
| Sidebar: search (title, name, description), empty-search state | ✅ | |
| Sidebar selection follows focus; click shows in focused pane; new pane; open dot | ✅ | Click, Shift+click or context "Open in New Pane"; dot (accent while running) |
| Unavailable analyses dimmed, engine reason as help | ✅ | Tooltip with reason |
| Short/long titles, icons, categories for ~40 known analyses; unknown get title-case | ✅ | `AnalysisCatalog` with Segoe Fluent Icons |
| Drag and drop of `.mklight/.json/.zmx/.seq` onto a window | ⬜ | |
| Per-window focused actions published to menus | 🟡 | No menu bar by design; File flyout plus shortcuts |
| One number format everywhere (decimal point, no grouping, comma accepted, no "-0.0000") | ✅ | `NumberDisplay`; settings form commit rule |
| Unit display convention | 🟡 | Units in the settings form; other places as they are built |
| Tooltips on controls; accessibility labels and values | 🟡 | Tooltips on built controls; accessibility labels for plots, cells, markers ⬜ |

## 2. Menus, commands and keyboard shortcuts

| Feature | Status | Windows notes |
|---|---|---|
| Undo/Redo with action names | ✅ | Buttons with names; Ctrl+Z, Ctrl+Y, Ctrl+Shift+Z; whole-state undo |
| File: New from Sample submenu | 🟡 | "New from sample" opens the welcome window (sample cards) |
| File: Show Welcome Window | ✅ | Ctrl+N |
| File: New/Open/Save/Save As/Open Recent; Revert, Duplicate, Rename, autosave, versions | 🟡 | File flyout: New, Open, Save, Save As; Open Recent entry exists, list in the welcome window; Revert/Duplicate/Rename/autosave ⬜ |
| View: Sidebar and Inspector commands | 🟡 | Inspector Ctrl+Alt+P; sidebar via the pane button |
| View: Show/Hide Lens Data, Show/Hide Layout | 🟡 | A sidebar row shows a hidden panel (`PanelHost.ShowPanel`); hide by the panel's context menu or tab close; no menu commands |
| View > Panel: Move to New/Main Window, Hide/Close, Reset Panel Layout | ✅ | Panel context menus and tab close buttons; Reset Panel Layout in the File menu |
| View: Section, X-Z Section, 3D View, Fit Layout, Ray Colors | ✅ | Section / 3D and Y–Z / X–Z selector and Display menu in the layout; Fit Ctrl+0 |
| Lens menu: Insert Before/After, Delete, Make Aperture Stop, Toggle Variable | 🟡 | Round 1 (lens table agent): verify; no menu bar, keys on the table |
| Lens menu: Replace with Sample | ⬜ | |
| Lens menu: Glass Catalogs…, Add Glass Catalog to Design… | ✅ | File menu commands (`Windows/GlassCatalogsRegistration.cs`) |
| Design menu: Optimize, Glass Substitution, Tolerance, Apply Solves, Show Issues | ⬜ | Show Issues: the sidebar Issues row and capsule button exist |
| Analysis menu: New Analysis Pane, Run Analyses, Cancel Running, Compute On | 🟡 | F5 run all, Ctrl+Esc cancel all, sidebar opens panes; Compute On is the backend picker in the status details |
| Job panels: Return is primary action, Escape cancels | ⬜ | |
| Menu items disabled by context | ⬜ | No menu bar |

## 3. Panels and arrangement

| Feature | Status | Windows notes |
|---|---|---|
| Panels as a tree of H/V splits with tab groups | ✅ | `ViewModels/PanelArrangement.cs`, `Views/Panels/PanelTreeView.cs` |
| Default arrangement: lens data top; layout and panes side by side | ✅ | Hairline dividers |
| Header title is the drag handle | ✅ | `Views/Panels/PanelDrag.cs` |
| Panel context menu (Move to New/Main Window, Hide, Close) | ✅ | Move to New / Main Window, Hide, Close |
| Panel "More" menu in every header | 🟡 | Analysis panes only |
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
| Persistence in project `views` (panes, arrangement, viewport, ray colouring, stretch) | 🟡 | Panes, settings and arrangement with window frames in `views` (`PanelPersistence`, `WorkspaceLayout`); layout viewport mode, camera, stretch not saved |
| Views without arrangement open in the standard one | ✅ | Open in the standard arrangement |

## 4. Lens data table

Checked after the merge of the round-1 table (`Views/Editor`, `ViewModels/LensTable*.cs`).

| Feature | Status | Windows notes |
|---|---|---|
| Header: title, caption "N surfaces · stop on S · mm · V variables", Insert / Delete / More buttons | ✅ | Title, caption "11 surfaces · stop on 6 · mm", insert (+) and delete (−) circle buttons; no More menu |
| Columns Surf, Type, Comment, Radius, Thickness, Material, Semi-Diameter, Conic; pinned header; resizable; horizontal scroll | 🟡 | All eight columns, pinned header, vertical scroll; not resizable, no horizontal scroll |
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
| Type column menu grouped by family, cylinder presets | 🟡 | Flat menu with group headers and Fluent glyphs, current type checked; cylinder presets unchecked |
| Type title "Sphere" for k=0, plane etc. | ✅ | |
| Anamorphic radius/conic two sections ("y"/"x"), Tab y then x | ✅ | Both sections ("y ∞   x −50"), each editable |
| Cylinder: flat section dimmed, biconic shows both conics | ✅ | Flat section dimmed; biconic shows both conics |
| Automatic semi-diameters dimmed; editing fixes it | ✅ | Engine's resolved value from the layout, dimmed; typing a value fixes it |
| Variable marker V badge (bounded bar, pinned dot, orange when solved) | 🟡 | V badges from the project's variables (bounded, pinned); no toggle (needs an undoable section edit) |
| Solve marker letters M C F P D O E | ✅ | Letters before numbers so digits stay aligned |
| Validation issue badge on the surface label | ✅ | Warning/error glyph in the Surf cell, messages as tooltip |
| Insert before/after, delete (disabled rules), make stop | ✅ | Buttons, context menu, Ctrl+Shift+I / Ctrl+Alt+I, Ctrl+Delete, Ctrl+Shift+T |
| Context menu: Insert, Delete, Make Stop, Variables submenu, Show in Inspector | 🟡 | Insert, Delete, Make Stop; no Variables submenu or Show in Inspector |
| Variables menu and toggle refuse a solve-controlled parameter | ⬜ | Not in round 1 scope |
| Every edit an undoable named step; invalidated solves/overrides dropped in same step | ✅ | DesignDocument.Edit with the Mac action names; whole-state undo |
| Setting type keeps carry-over parameters | ✅ | Model SetShapeKind |
| Table accessibility labels and Edit action | 🟡 | Automation names on rows, type button and badges; no Edit action pattern |

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
| Title chip (drag handle) + More menu | 🟡 | Title chip is the panel's drag handle with the panel context menu; no More menu |
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
| Camera: pan, zoom about cursor, orbit, double-click fit | 🟡 | Drag pans (orbits in 3-D), wheel zooms about the pointer, double-click fits; touch pinch ⬜ |
| Section views of toleranced builds (markNearSection) | ⬜ | |
| "Design changed" card when stale | ✅ | Click updates |
| Preview label "Previewing … · not applied" | ✅ | Accent chip "Previewing … · not applied" on the layout |
| Placeholder, "Viewport Unavailable", accessibility description | 🟡 | Accessibility description of the scene; placeholder / "Viewport Unavailable" ⬜ |
| Layout as a pane of `layout` with settings (rays_per_field, wavelengths, mesh_rings), re-run on change | ✅ | `rays_per_field`, `wavelengths`, `mesh_rings` from the Display menu, re-run on change |
| Reduce Motion replaces animations | ⬜ | |
| Click selects surface | ✅ | Click in the layout selects the surface |

## 9. Analysis panes (generic chrome)

| Feature | Status | Windows notes |
|---|---|---|
| Header: glyph + title (drag handle), chevron analysis menu by category | 🟡 | Glyph, title and menu; drag handle ⬜ |
| State line: settings differing from defaults, or the stage | ✅ | |
| Glass cluster: Run / Cancel, auto-update, Settings toggle, More | ✅ | |
| More menu: Update Automatically, Show Settings, Duplicate Pane, Copy Result as JSON, Close Pane | 🟡 | Restore defaults, copy result JSON, close pane; Duplicate Pane ⬜ |
| Narrow-pane behaviour: folding cluster, short titles | 🟡 | Short titles on narrow panes; folding unchecked |
| Tabbed pane | ⬜ | |
| Settings panel: Defaults, Done, open state saved | 🟡 | Settings toggle and restore defaults; open state not saved |
| Footer key numbers (subscripts, spoken form) | ✅ | Key numbers in the footer; VoiceOver forms ⬜ |
| Backend badge (CPU · f64 / GPU · f32, duration, warning, fallback) | ✅ | Opens run details |
| Run details: backend, requested, reason, precision, seed, snapshot, engine, time, failed attempts, Route, Tier | ✅ | Route line (`cost_decision`) and tier added (`Views/Analysis/RunDetails`); unit-tested, not checked on a real GPU fallback |
| Warnings grouped, numbers masked, fields/wavelengths listed, raw messages | ✅ | `RunDetails/Model/RunWarnings.cs` |
| One-click remedies (auto pupil grid, image grid, PSF window) | ✅ | Three remedies (auto pupil grid, image grid, PSF window) change the pane's settings |
| Stale result: dimmed, "Design changed" card with Update | ✅ | |
| Updating: progress line, pulse | 🟡 | Progress bar; pulse ⬜ |
| Placeholders: running, Not Available, Design Has Errors, Failed, Not Ready, Cancelled | 🟡 | Unavailable and failed messages; others unchecked |
| Failed rerun over existing result: warning capsule | ⬜ | |
| Automatic vs manual update; long analyses start manual; newer cancels older | ✅ | |
| Debounce from edit to snapshot; stale on snapshot/library/request difference | ✅ | `Workspace` debounced snapshots, `IsCurrent` |
| Manual retry asks for a fresh run | 🟡 | Run exists; fresh-run request unchecked |
| Unknown engine analyses appear and run with generic view | ✅ | |

## 10. Analysis settings form

| Feature | Status | Windows notes |
|---|---|---|
| Two-column form, small controls, spectrum control last | ✅ | Spectrum control last |
| Controls from schema: number with units/bounds, toggle, enum, text | ✅ | One number format commit rule |
| Field / wavelength / surface menus and index settings | ✅ | Single index |
| Integer-list settings as check-mark menus | 🟡 | Integer lists work; multi-select menus ⬜ |
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
| Image simulation: source menu, optics, FOV/centre/resolution | 🟡 | Source pattern and optics; FOV, patch centre, resolution ⬜ |
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
| Validation view (issue list, "No Issues") | 🟡 | Generic view; issues also in the Issues row and flyout |
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
| PSF (item picker, log scale, Airy) | ✅ | Field picker, linear / log, core / full |
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
| Image Simulation: compare modes, exposure, zoom, pan, drop image | 🟡 | Side by side, split, blink and exposure; zoom, pan and dropping an image ⬜ |
| Image simulation controls: FOV, patch centre, resolution, picker | ⬜ | |
| Image simulation footer and field-mapping table | ✅ | Pixel, patch, PSF window and Field Mapping flyout |
| Image simulation starting settings, source drawing, imported images | 🟡 | Starting settings (`--do image-simulation-start`) and built-in source patterns; imported images ⬜ |
| Shared chart components (line chart, legend, colour bar, sparkline, key/value grid) | 🟡 | `PlotKit`, `Legend`, heat-map colour bar; sparkline ⬜ |
| Palette: categorical slots, spectral colours, dashes, ramps | ✅ | `Palette` (the Mac's validated colours) |

## 12. Design jobs: common

UI not started. The engine binding supports job submit, stop (best so far) and cancel, and `ApplySolvesAsync`.

| Feature | Status | Windows notes |
|---|---|---|
| Job panel structure: header, body, action bar | ⬜ | Binding ready |
| Action bar states (setup, running, review, done, failed) | ⬜ | |
| Failure page and "Applied to the design" notice | ⬜ | |
| Done returns job to its starting page | ⬜ | |
| Jobs keep running when the panel closes; capsule reopens | ⬜ | |
| Jobs end when the main window closes | ⬜ | |
| Apply is one undoable step | ⬜ | Whole-state undo exists |

## 13. Optimize

| Feature | Status | Windows notes |
|---|---|---|
| Merit Function box (Default/Custom, live readout, Convert to Custom, hold focal length) | ⬜ | |
| Default merit: criterion, reference, MTF options, rings/arms, fields/wavelengths, limits, squeeze | ⬜ | |
| Custom merit: operand list and searchable picker | ⬜ | |
| Operand editor | ⬜ | |
| Operand list actions (add, delete, duplicate, move, reorder, multi-select) | ⬜ | |
| Live merit evaluation (200 ms) | ⬜ | |
| Merit edits undoable, saved in `merit` section | ⬜ | Model keeps the section verbatim |
| Variables group ("Vary" choice) | ⬜ | |
| Variables and bounds list | ⬜ | |
| Algorithm group (method, iterations, seed) | ⬜ | |
| Running view (iteration, merit, sparkline) | ⬜ | |
| Result: before/after, preview in the window, variable list | ⬜ | See design notes (preview) |
| "Fix the design's errors first" and blocked start | ⬜ | |

## 14. Glass Substitution

| Feature | Status | Windows notes |
|---|---|---|
| Glasses That May Change (checkbox per gap) | ⬜ | |
| Per-slot override button | ⬜ | |
| Candidates for every lens (catalogs, status filter, rule, limits) | ⬜ | Binding ready for glass rows/catalogs |
| Search (Quick/Thorough, max combinations, time, iterations, keep N) | ⬜ | |
| Merit/variables group, Edit Merit and Variables… | ⬜ | |
| Running view, Stop keeps ranking, Cancel discards | ⬜ | Binding has stop/cancel |
| Ranked results list | ⬜ | |
| Candidate detail (glasses, shape changes, layouts, glass map) | ⬜ | |
| Preview selected row; Apply as one undoable step | ⬜ | See design notes (preview) |

## 15. Tolerance

| Feature | Status | Windows notes |
|---|---|---|
| "Show the last results" shortcut | ⬜ | |
| Tolerances table grouped by element | ⬜ | |
| Row kinds | ⬜ | |
| Row editing (limits, surface menu, unit conversion, problems) | ⬜ | |
| Row context menu (unit, distribution, symmetric, duplicate, delete) | ⬜ | |
| Add menu (surface and element sections) | ⬜ | |
| Default Tolerances… generator | ⬜ | |
| Test fringes wavelength field | ⬜ | |
| Element tolerances (move together, pivot, range) | ⬜ | |
| Compensators | ⬜ | |
| Criterion section | ⬜ | |
| Analysis section (mode, trials, seed, percentiles) | ⬜ | |
| Running view (stage, elapsed, time left; Stop/Cancel) | ⬜ | Binding has stop/cancel |
| Results (nominal, RSS, percentiles, yield, Export menu) | ⬜ | |
| Sensitivity table | ⬜ | |
| Monte Carlo histogram | ⬜ | |
| Compensator results | ⬜ | |
| Preview card and layout caption | ⬜ | See design notes (preview) |
| Tolerance spec saved in `tolerances` section | ⬜ | Model keeps the section verbatim |

## 16. Apply Solves

| Feature | Status | Windows notes |
|---|---|---|
| Empty state "No solves" | ⬜ | |
| Dry run list (target, kind, previous → value, residual) | ⬜ | `ApplySolvesAsync` ready |
| Caption and Apply to Design as one undoable commit | ⬜ | |

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
| Open via File > Open, Welcome, drag and drop, file associations | 🟡 | File > Open and welcome; drag and drop and associations ⬜ |
| Welcome window (New Design prominent, Open, recent files, drop hint) | ✅ | Drop hint ⬜ |
| Sample cards with engine-computed layout thumbnails; Surface Types group | ✅ | |
| Mock engine mark in the welcome window | ⬜ | Not ported by design: no mock engine |
| Import report sheet (groups, Copy Report, Load Glass Catalog) | ⬜ | A toast notes warnings |
| Autosave, versions; dirty state follows undo | 🟡 | Dirty marker in the title by revision; autosave and versions ⬜ |
| Save stamps metadata and library-catalog references | ✅ | Metadata stamp and library-catalog references (`DesignDocument`) |
| Project sections preserved (system, catalogs, variables, views, merit, tolerances, metadata, unknown keys) | ✅ | Unknown sections kept verbatim |
| Undo/redo naming for all edits across windows | ✅ | Whole-state undo; one window |
| Replace with Sample (undoable) | ⬜ | |

## 19. Appearance and accessibility

| Feature | Status | Windows notes |
|---|---|---|
| Light and dark both designed; semantic colours; dynamic palette | ✅ | System / Light / Dark in the File menu; the app theme is fixed at launch (see Known issues) |
| Liquid Glass chrome | ✅ | Translated to Mica / acrylic on chrome only |
| Reduce Motion | ⬜ | |
| Reduce Transparency / Increase Contrast | 🟡 | High-contrast theme dictionary exists, unchecked |
| Tabular figures everywhere | ✅ | |
| Spacing scale | ✅ | `Space*` resources |
| Accessibility labels/values on plots, drawings, markers, cells, panes | ⬜ | |
| Physical wavelength colours; colour never sole carrier | ✅ | Palette plus dashes |
| Help tags with shortcut hints | 🟡 | On built controls |

## 20. Engine integration

| Feature | Status | Windows notes |
|---|---|---|
| FFIEngine always; missing library handling | ✅ | `FFIEngine` over `mklight_ffi.dll`; missing DLL shows an explanatory window; mock engine not ported by design |
| GPU kernel preparation at launch with cache directory | ✅ | `%LOCALAPPDATA%\MKLight\GPU Kernels`; App Nap n/a |
| "GPU kernels" row texts (Preparing, Ready, No GPU, cache unavailable) | 🟡 | State shown in the details flyout |
| Backend preference per design (Automatic / CPU / GPU) | ✅ | Picker in the status details flyout |
| Descriptors, schemas, availability read once at engine creation | ✅ | |
| Per-design session; snapshot and library hash; validate on every snapshot | ✅ | |
| Glass library and project catalogs synced before first snapshot | ✅ | `AppHooks.SessionStarting`, `Workspace.LoadProjectCatalogsAsync` |
| Preview mechanism for candidate designs with "not applied" labels | 🟡 | `Workspace.Preview` / `Baseline`; strip on every pane and the layout, badge and End Preview in the capsule; spot Design | Candidate switch ⬜ |
| `convert_spec`, `glass_info` cache, `merit_evaluation`, `merit_spec`, `apply_solves`, `preview_case` | 🟡 | `convert_spec` (System page) and `glass_info` (glass browser) in use; `merit_*`, `apply_solves`, `preview_case` ⬜ |
| Job streaming progress, Stop vs Cancel | 🟡 | Binding done and used by analyses; design-job UI ⬜ |
| Engine is the authority: no optics in the UI | ✅ | Display formulas only where the Mac has them |

## 21. Debug launch options

Windows has its own set (docs/WINDOWS.md "Launch options"): `--sample`, `--analysis`, `--settings`, `--show-settings`, `--theme`, `--select`, `--edit`, `--undo`, `--save-as`, `--size`, `--screenshot`, `--delay` and `--welcome`, plus a file argument, and `--do <action>[=arg]` for actions features register (inspector, panels, layout, settings, run details, results, glass catalogs).

| Mac option | Status | Windows equivalent |
|---|---|---|
| `-MKLightNoWelcome YES` | ✅ | A file argument or `--sample` skips the welcome window; `--welcome` shows it |
| `-MKLightSidebar hidden` | ⬜ | |
| `-MKLightSelectSurface <i>` | ✅ | `--select <i>` |
| `-MKLightInspector system` | ✅ | `--do inspector-system` / `inspector-surface` |
| `-MKLightViewport 3D` | ✅ | `--do layout-3d` |
| `-MKLightSheet` (optimize, tolerance, solves, glass) | ⬜ | |
| `-MKLightAutoRun`, `-AutoPick`, `-AutoStop`, `-AutoApply` | ⬜ | |
| `-MKLightToleranceTrials`, `-MKLightTolerancePreview` | ⬜ | |
| `-MKLightShowIssues YES` | ⬜ | |
| `-MKLightToast <text>` | ⬜ | |
| `-MKLightAppearance light` or `dark` | ✅ | `--theme light` or `dark` |
| `-MKLightFootprints`, `-MKLightFit lens`, `-MKLightYaw` | 🟡 | `--do layout-footprints`, `layout-yaw`, `layout-pitch`; fit lens ⬜ |
| `-MKLightGlassCatalogs`, `-GlassScope`, `-GlassSearch` | ✅ | `--do glass-catalogs`, `-search`, `-scope` |
| `-MKLightSupportDirectory <path>` | ⬜ | |
| `-MKLightRunDetails <analysis>` | ✅ | `--do run-details` |
| `-MKLightAutoEdit`, `-MKLightAutoConvert` (merit editor) | ⬜ | `--edit` exists for lens edits only |
| `-MKLightCurvatureBoundsUnit curvature` | ⬜ | |
| Test render hook `MKLIGHT_RENDER_DIR` | 🟡 | `--screenshot <file.png>` renders the whole window (as composed), plus `--size`, `--delay` |

## Summary

| Area | Done | Partial | Not started | Total |
|---|---|---|---|---|
| 1. Window and chrome | 17 | 11 | 3 | 31 |
| 2. Menus, commands and keyboard shortcuts | 5 | 6 | 4 | 15 |
| 3. Panels and arrangement | 18 | 2 | 0 | 20 |
| 4. Lens data table | 19 | 6 | 1 | 26 |
| 5. Inspector: Surface page | 20 | 0 | 0 | 20 |
| 6. Inspector: System page | 10 | 0 | 0 | 10 |
| 7. Surface types | 19 | 0 | 0 | 19 |
| 8. Layout viewport | 15 | 3 | 2 | 20 |
| 9. Analysis panes (generic chrome) | 11 | 7 | 2 | 20 |
| 10. Analysis settings form | 21 | 2 | 0 | 23 |
| 11. Dedicated result views | 32 | 4 | 1 | 37 |
| 12. Design jobs: common | 0 | 0 | 7 | 7 |
| 13. Optimize | 0 | 0 | 13 | 13 |
| 14. Glass Substitution | 0 | 0 | 9 | 9 |
| 15. Tolerance | 0 | 0 | 19 | 19 |
| 16. Apply Solves | 0 | 0 | 3 | 3 |
| 17. Glass catalogs | 17 | 0 | 0 | 17 |
| 18. Documents, welcome and import | 6 | 2 | 3 | 11 |
| 19. Appearance and accessibility | 5 | 2 | 2 | 9 |
| 20. Engine integration | 7 | 4 | 0 | 11 |
| 21. Debug launch options | 7 | 2 | 9 | 18 |
| **Total** | **229** | **51** | **78** | **358** |

## Known issues

Not features of the Mac app, so not counted above.

* **Layout viewport state is not saved**: mode, camera and vertical stretch are not
  written to `views`.
* **Theme**: the app theme (`Application.RequestedTheme`) is fixed at launch; a switch
  during a run completes on the next launch.
* **Image simulation** has no image-file import, no zoom/pan and no FOV controls.
* **Drag and drop and 3-D mouse input** were verified in-process only (simulated
  drags, scripted camera), not with a real mouse.

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
* The engine binding already provides what the design jobs need (submit, stop,
  cancel, `ApplySolvesAsync`); the rounds that follow are UI and view-model work.
  The preview seam (`Workspace.Preview`, `SubmitDesignJob`, `DesignSession.SetJobActivity`,
  `ShowInspectorPage`, the inspector page registry) is in place for them.
* The lens data table statuses (section 4) must be re-checked by the
  coordinator after the lens table agent's merge.
* Known gaps in the Swift app, which do not require parity (not counted
  above): configurations are listed but not editable; pickups have no editor
  (only the solve marker); tolerancing has no element tilt z, no irregularity
  orientation, and radius perturbations in ΔR/% scale linearly from the limit;
  layout-only changes do not mark the document dirty; library entries store
  plain paths (no bookmarks) and renaming a user catalog does not rewrite
  designs; an imported image for image simulation is limited to 512 px per side
  as JSON buffers.
