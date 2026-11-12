# MKLight for macOS on the Rust engine

The native app (`apps/macos`, SwiftPM) runs on the Rust engine through its C
ABI (`crates/mklight-ffi`, docs/FFI.md). This page covers the design brief and
the design of the user interface, the window structure, building, the Swift
binding (`FFIEngine`), the state model, the tests and what remains for
iPadOS. Package layout and the Metal viewport are described in
`apps/macos/README.md`; screenshots are in `apps/macos/docs/screenshots/`.

## Platform

The package targets **macOS 27** (`swift-tools-version: 6.4`,
`platforms: [.macOS(.v27)]`); the later iPad target will be **iPadOS 27**. On
27 every Liquid Glass and SwiftUI API the app uses (glass effects and
containers, glass button styles, `ToolbarSpacer`, `safeAreaBar`, scroll edge
effects) is available without `#available` checks, and building with the
27 SDK always gives the new design: the `UIDesignRequiresCompatibility`
opt-out is ignored from 27 on.
Build with Xcode 27.2 / Swift 6.4.2 (Swift 6 language mode, strict
concurrency).

## Design brief: Liquid Glass

What we took from Apple's current guidance is below, with how MKLight
applies it. The sources, cited by their short names, are listed at the end of
this page.

**Principles, and what MKLight does with them**

1. *Glass is the navigation and control layer that floats above content;
   never put it on content* (Materials, Meet Liquid Glass: a table "keeps it
   in the content layer"). The lens table, plots, PSF and wavefront images
   and the Metal viewport are plain content: system backgrounds, crisp lines,
   no glass. Glass is on the toolbar, the sidebar and inspector (system), the
   controls floating over the viewport, the pane header clusters, the stale
   card, the job action bar and transient messages.
2. *Use custom glass sparingly; avoid glass on glass; group nearby glass in a
   `GlassEffectContainer`* (Applying Liquid Glass to custom views, WWDC25 323).
   Each cluster (viewport controls, a pane's buttons, the lens table's insert
   and delete, the job actions) is one container; popovers and menus that
   open from glass use the system's own material.
3. *One prominent action per view, tinted; hierarchy by style, not size*
   (Toolbars, Buttons, Color). `.glassProminent` or an accent-tinted glass
   capsule only for the single primary action of a flow: Optimize → Apply to
   Design, Find Glasses → Apply to Design (Glass Substitution), Run
   (Tolerance), Apply to Design (Solves), New Design (welcome), Update on a
   stale pane. A disabled action dims and loses its tint. Everything else is regular glass or plain.
4. *Morph glass when controls change state* (`glassEffectID`). The job
   action bar morphs Optimize → [progress, Cancel, Stop] → [Discard, Apply to
   Design] → [Run Again, Done]; the viewport's Section / 3D selection slides
   between its options.
5. *Group toolbar items by function, at most three groups, spacers between
   them; the primary action and inspector toggle trailing; every item also a
   menu command* (Toolbars, WWDC25 323). Add Analysis and the inspector
   toggle trailing, left of the inspector; the inspector's page tabs in
   its own part of the toolbar; the design's status in
   the centre (`.principal`). The lens data and the layout are sidebar
   entries; the design jobs are pages of the inspector. All have menu
   commands and shortcuts.
6. *Avoid critical information at the bottom of a Mac window* (Layout). There
   is no status bar: activity, EFL, F/#, total track and the issue
   count live in the toolbar's status capsule; engine, backend and snapshot
   are in its popover; transient messages float as a glass toast.
7. *Scroll edge effects instead of bar backgrounds; hard for pinned table
   headers, soft elsewhere; one per view* (Scroll views, WWDC25 356). The lens
   table uses `.hard` at the top; the inspector's form scrolls under the
   toolbar (its page tabs) with the system's soft edge; job panels have soft
   edges under their header and their action bar.
8. *Content extends under floating chrome* (Sidebars, Layout,
   `backgroundExtensionEffect`). The viewport draws edge to edge under its
   glass controls. We do not apply `backgroundExtensionEffect` to the Metal
   view: the effect mirrors its view's content, and copies of a live
   `MTKView` are neither cheap nor reliable; the sidebar's glass already
   floats over the window's background.
9. *Inspectors show the selection and update with it; less modality*
   (Panels, Designing for macOS). Design jobs are inspector panels, not modal
   sheets: the design, layout and analyses stay usable while a job
   runs, and a job keeps running when its panel closes (the status capsule
   shows it and reopens it).
10. *Concentric, capsule and fixed shapes; dense inspectors use small
    controls* (WWDC25 356). Capsules for floating controls, circles for icon
    buttons, a rounded card for the stale message; system form controls in
    the inspector.
11. *Color sparingly and with meaning; supply light, dark and increased
    contrast; don't rely on color alone* (Color, Charts). The accent marks
    selection, the focused pane, variables, the GPU badge and the primary
    action. Plots keep the validated `Palette`; wavelengths keep their
    physical colours; series also differ by dash.
12. *Typography: system font, 13 pt body on macOS, tabular figures for
    numbers* (Typography; macOS has no Dynamic Type). Every number uses
    `monospacedDigit()`; text styles, not point sizes, so the iPad build gets
    Dynamic Type.
13. *Charts: the data is the most prominent thing; light grids and labels;
    don't hide critical information behind interaction; label for VoiceOver*
    (Charts). Plots are content in the panes with few, light ticks; key
    numbers sit in the pane footer; every plot and custom drawing has an
    accessibility label and value. `Chart3D` is used nowhere yet: the guidance
    keeps 3D for shapes whose form matters more than values (a candidate for
    a future freeform sag view).
14. *Accessibility adapts glass automatically; check custom views* (Meet
    Liquid Glass, Accessibility). Reduce Transparency and Increase Contrast
    are handled by the system glass; custom views use system materials and
    semantic colours. Reduce Motion turns off the morphing animations, the
    stale blur and the updating pulse, and replaces slides by fades.

## Design principles

* **The lens data is the heart.** The table is always visible at the top of
  the window and shows only what every surface has: surface (with the stop
  marked), type, comment, radius, thickness, material, semi-diameter and
  conic. Everything type-specific lives in the inspector. Anamorphic
  surfaces (toroid, biconic, cylinder) show both sections in the radius
  cell, `y ∞   x −50`, each edited in place (Tab moves y → x); a
  cylinder's flat section is dimmed and only its powered section's conic
  shows; a biconic shows both conics.
* **One number format.** Numbers read the same everywhere, whatever the
  system region: a decimal point and no digit grouping (`n=1.5`,
  `12345.678`), as in prescriptions, glass catalogues and the engine's
  files. Every window runs with `NumberDisplay.locale` (en_US_POSIX), so
  chart axes, number fields and `formatted()` follow the rule; the
  `String(format:)` paths are locale-free. Typed numbers also accept a
  decimal comma (`1,5`). Signed columns never show `−0.0000`.
* **One text-field rule.** Every field that edits a value commits by the
  same rule (`FieldEdit`, shown by `CommittingField`): compact text at rest
  and the exact value with the focus, only edited text is committed (on
  Return and when the focus leaves), a value equal to the current one is
  not, and text typed for one object goes to that object.
* **Units at hand.** Every bounded or edited quantity names its unit: the
  lens table's header (mm), the bounds list's unit column (`mm`, `1/mm`,
  `mm⁻³` for A₄, `°`, `µm`, `rad`; dimensionless shows `–`), the colour
  bars, the axis titles.
* **Progressive disclosure.** Surface types are described as data
  (`SurfaceTypes`): sections of parameters addressed by their key path in the
  model's JSON (`curvature`, `phase.norm_radius`), coefficient lists with add
  and remove, each term labelled with its meaning (`A₄ r⁴`, `β₃ r³`,
  `a₂ u⁴Q₂ᶜᵒⁿ`, `C(x²y)`, `Z₇ coma x`), a variable toggle per continuous
  value, and a preview kind. The inspector shows only the selected type's
  parameters. A new type needs its model case and one descriptor, no view.
  Placement (decenters, tilts) appears when it is used, first for coordinate
  breaks. Analysis settings live in a popover, with a summary of what
  differs from the defaults under the pane title.
* **The engine is the authority.** The app never computes optics. Analyses
  are run by name and drawn from the engine's result JSON and buffers
  (docs/RESULTS.md). The list of analyses, their settings and their
  availability are read at run time, so a new engine analysis appears in the
  sidebar and runs with a generic, schema-driven view and no code change.
  Unknown results are shown as data, never as an error. The inspector's
  shape previews evaluate the documented sag formulas (display geometry,
  `ShapeSag`); a DOE's paraxial power comes from A₁ by docs/SURFACES.md §3.
* **Reproducible and honest.** Every result carries the settings and seed
  the engine recorded, the backend that actually computed it (CPU f64
  reference or GPU f32, with the fallback reason, the route of an
  automatic request and any backend attempt that failed) and its warnings,
  one click away from the pane's backend badge. A result that no longer matches
  the design, the glass library or its settings is dimmed and blurred under
  a glass "Design changed" card with Update. A previewed candidate design
  is labelled as not applied.
* **Calm and native.** Liquid Glass for controls (above), system materials
  and semantic colours elsewhere, SF Symbols, the system font with tabular
  digits, a 2–24 pt spacing scale (`Space`). Light and dark appearance are
  both designed.
* **No modal alerts for routine errors.** Rejected input is reverted and
  explained in a glass toast; failing analyses explain themselves in their
  pane; validation issues are listed in a popover from the status capsule and
  badged in the table.

## Window structure

```
┌ toolbar: ◧ title │ ⟳ EFL · F/# · TT · ⓘ  + ◨ ┬ ◐ ☰ │ ✦ ▦ ± f(x)        ┐
├ sidebar (glass) ─┬ Lens Data (hard scroll edge) ──────────────────────┤ inspector (full   │
│ Lens Data, Layout│ Surf Type Comment Radius Thickness Material …      │ height)           │
│  Issues          ├─────────────────────────┬──────────────────────────┤ type menu, preview│
│ Analyses         │ Layout (Metal, edge to  │ analysis panes …         │ data-driven form  │
│  (searchable)    │ edge) [Layout] [2D|3D]◉⤢│ title ⌄   ⟳ ⚡ ⚙ …        │ — or a job panel: │
│                  │                         │ plot (content)           │ Optimize, Tol.,   │
│                  │ 20 mm                   │ key numbers  [CPU · f64] │ Solves + actions  │
└──────────────────┴─────────────────────────┴──────────────────────────┴───────────────────┘
```

* **Panel headers**: the lens data and the analysis panes share one
  header style (icon and title as the drag handle, a caption state line,
  small circular glass buttons ending in More); the layout keeps its
  controls floating over the scene, with the same icon and More menu
  (`PanelHeaderTitle`, `PanelMoreMenu`).
* **`NavigationSplitView`**: the sidebar's Design section has the lens
  data and the layout (selecting one shows and focuses that panel, a dot
  marks it open, its context menu closes it) and the issues; below, it
  lists every engine analysis by category (`.searchable` in the sidebar); a click shows it in the focused
  pane, ⌥-click or the context menu opens a new pane. Open analyses carry a
  dot, unavailable ones are dimmed with the engine's reason as help.
* **Toolbar**: the inspector is a full-height column with its own part of
  the toolbar (the `.inspector` modifier sits on the `NavigationSplitView`,
  not its detail column: only then does SwiftUI add the tracking separator
  for it). That part holds the inspector's page tabs (`InspectorTabBar`:
  Surface, System and the four design jobs), declared on the inspector's
  content; while the inspector is hidden the item is empty. Add Analysis
  and the inspector toggle are the content's `.primaryAction` items, left
  of the inspector, sliding with its edge. They are not declared on the
  inspector: SwiftUI would then move them between the two parts of the
  toolbar as the inspector opens and closes, and they blink. Nothing in
  the toggle reads `showInspector` (its help text is fixed), so a toggle
  does not redraw it. Only the toggle (and ⌃⌘I) shows or hides the
  inspector. In the centre,
  the **status capsule** (`DesignStatus`): the GPU kernel preparation at
  launch while it lasts ("Preparing GPU 42%", see "GPU kernel preparation"
  under `FFIEngine`), a running job ("Optimizing · 479", click to show it)
  or analysis activity, EFL (EFL y for a design that is not rotationally
  symmetric), F/# and total track, and the issues button. Its popover has
  the first-order table, the anamorphic data, the engine, the snapshot, the
  state of the GPU kernels (with a second line when their cache could not
  be used) and the compute backend picker.
* **Panels and windows** (`PanelArrangement`, `PanelViews`, `PanelDrag`,
  `DetachedWindows`): the lens data, the layout and every analysis pane are
  panels. The content area is a tree of horizontal and vertical splits whose
  leaves are tab groups; the default is the layout above (lens data across
  the top, the layout and the panes side by side under it). Key decisions:
  * *The header is the handle.* A panel alone in its group shows no extra
    chrome: its title (the lens data's and analysis panes' titles, the
    layout's title chip) is what you drag, with a grab pointer and a context
    menu (Move to New Window, Move to Main Window, Close / Hide). A group of
    several shows a compact tab strip (the front tab on glass, close on
    hover); tabs switch on click and drag like titles. The analysis menu is
    the chevron beside the title.
  * *Drop zones like a browser or an IDE.* While a panel is dragged, the
    place it would land is outlined in the accent colour: a group's header
    or middle adds it as a tab (between the tabs under the pointer), its
    outer 30 % towards an edge splits that side, a window's outer edge adds
    a full-height column or full-width row. Dropping outside the design's
    windows opens a **separate window** there, previewed by an outline of
    the window while dragging; Escape cancels. Splits of the same axis
    merge, so a row stays one row; dividers drag (double click: equal
    shares). Drops are judged against MKLight's own windows, front to back.
  * *Separate windows are views of the same design* (`DesignSession`):
    selection, edits, analysis updates, the preview of a candidate and the
    design jobs are shared, so everything stays live. A separate window has
    its own tree (tabs and splits, and panels can be dragged on from it to
    yet another window), a toolbar with Add Analysis and Move to Main
    Window, and the design's name as subtitle. It uses the document's undo
    manager and passes document commands (Save, Revert, Duplicate…) to the
    document; Undo and Redo name the action from any of the design's
    windows. Closing it closes its panels (like a browser window); Move to
    Main Window puts them back where their kind belongs (an analysis beside
    the others, the lens data on top, the layout on the left). The
    separate windows close with the main window and open again with the
    project, at their saved frames (moved on screen if a display is gone).
  * *Closing.* Closing an analysis pane removes it; the lens data and the
    layout are hidden and come back from the toolbar, the View menu or the
    empty window's buttons, at their default places. View ▸ Panel ▸ Reset
    Panel Layout brings every panel back into the standard arrangement.
* **Lens data** (`LensTableView`, SwiftUI `Table`): text cells edit in place
  with full keyboard navigation (Return / double click edits the row; Tab
  and ⇧Tab move across, ↑/↓ and Return move between rows committing, Esc
  reverts). Numbers accept `∞`/`inf`, and the material column completes glass
  names from the session's catalogs. The type column is a compact menu with
  SF Symbols, grouped Spherical, Aspheric, Freeform, Anamorphic, Diffractive,
  Special (cylinder presets under Anamorphic). Small inline markers show
  variables (`V`, with a small bar under it when bounded and a dot when
  pinned) and solves (`M`, `C`, `F`, `P`, …); validation issues badge
  their surface. Insert and delete are glass buttons in the header, and in
  the context menu and the Lens menu. A catalogue glass with a melt offset
  (docs/MATERIALS.md) carries it as a small suffix, `N-SK16 +0.0012/−0.8`
  (Δn_d / ΔV_d; the column widens to fit, a `V` marks an offset variable).
  Retyping the same glass keeps its offset; another glass starts without.
* **Glass offsets in the inspector**: a catalogue glass shows "Add Melt
  Offset…"; then Δn_d and ΔV_d (`MeltOffsetRows`), validated against the
  engine's ±0.05 and ±10 as typed, each a possible variable (bounded to
  those limits) and solve target, one undoable edit each, with the melt
  glass's n_d and V_d against the catalogue's (`glass_info`, cached per
  glass in `Workspace.glassNumbers`) and Remove Melt Offset. The glass map
  marks the offset glass at its shifted position (`kind: "offset"`, its own
  id), dotted from its catalogue point, with a "Melt offset" legend entry.
* **Layout pane**: the engine's `layout` scene in the Metal viewport, edge to
  edge, with glass controls floating over it (title chip, Section / 3D
  selector, display options, fit), a scale bar, and a glass "Design changed"
  button when stale. The engine draws every surface type. For designs
  without rotational symmetry the selector offers Y–Z, X–Z and 3D; the
  layout's `section` setting follows it and the glass fills pair the
  profiles along the section's own axis. DOE surfaces
  carry a fine violet hatch (flag bit 2). In 3D the Display menu adds the
  engine's surface mesh (`mesh_rings`, flat shaded, the selected surface in
  the accent) and a Footprints button runs `footprint` and draws every
  surface's beam outline per field (`global_outlines`), filled
  translucently. Fit Lens frames the elements; the first fit of a long,
  slim scene does so by itself. **Vertical stretch**: the section views
  can be drawn ×2 … ×50 taller than long (Display menu); a first fit that
  would leave the section a sliver (under an eighth of the view's height)
  turns it on at the smallest step that fills a third. A stretched view
  says so in a tinted "Stretch ×N" menu among the controls and on the
  scale bar, which adds an upright bar of a round length as drawn and
  "vertical ×N"; the 3-D view always keeps true proportions. The stretch
  is saved in the project's `views` (`ui.stretch` of the layout entry);
  a saved one, even ×1, is kept instead of the automatic choice. A previewed candidate design shows a glass
  "Previewing … · not applied" label under the controls.
* **Analysis panes** (`AnalysisPaneView`): panels of their own, side by
  side by default. The header has the title (the drag handle), the
  analysis menu (chevron) and a one-line state or settings
  summary (the values that differ from a new pane's, "±0.193 mm · 3
  planes"), and a glass cluster: Run (Cancel while running), automatic
  update (bolt), Settings (a toggle) and More. The settings stay closed
  by default so the plot has the pane; the toggle opens them as a form
  between the header and the plot (`SettingsPanel`: a grouped card in the
  content layer that scrolls when long, with Defaults and Done), and
  whether a pane's settings are open is saved with the layout
  (`ui.settings_visible`). The plot is content. The footer has
  the key numbers and a backend badge (CPU · f64 neutral, GPU · f32 in the
  accent, an orange mark for warnings, a fallback from a backend that was
  asked for or tried among them, with their count when there are several
  kinds) that opens the run details.
  * *Settings from the schema* (`SettingControls`, `SettingsBar`): every
    control is chosen from the engine's settings schema alone, so no
    setting is ever typed as JSON. Scalars are number fields (with units
    and bounds), toggles, pickers and text fields; `{"type": …}` field and
    wavelength selections and index settings are menus of the system's
    fields, wavelengths and surfaces; two-number arrays are pairs
    (`[min, max]` ranges and `[x, y]` points that may be null have an
    automatic state); integer lists are check-mark menus of the fields,
    wavelengths or surfaces (empty: all); number lists are comma-separated
    fields; arrays of same-shaped rows (`[{"hx", "hy"}]`, `[[r, θ]]`) are
    small tables in a popover. An object with known properties (the
    schema's `properties`, or the keys of its default) is a group of rows
    under its label, with an on/off switch when null is allowed (the MTF
    detector); a tagged union, which the engine publishes as `oneOf`
    alternatives whose tag is a `const` (`schema::variants`: the
    transmission pupil sampling and model, the polarisation state, the
    encircled-energy centre, the stray-light source, the ghost clipping),
    is a picker of the variants followed by the chosen variant's own rows.
    The spectrum, the wave pupil sampling, the MTF frequencies, the image
    source and optics, the sensor CRA limit and the spot diagram's
    through-focus keep dedicated controls. Settings with no sensible
    control (availability overrides, explicit ghost path lists) are left
    out of the form and keep their value. `SettingsSchemaTests` and
    `FFISettingsSchemaTests` check the mapping, the latter over every
    analysis the engine lists.
  * *Narrow panes* (checked at about 240 pt): the header folds Update and
    Settings into More first, then sets the title a type step smaller,
    then abbreviates it (`RI Map`, `CRA`, `Pupil Aberr.`; the full title
    stays in the help tag and the VoiceOver label); the state line
    truncates and never decides the layout. Legends are one flow that
    wraps (wavelengths, line styles, the Airy disk together), spot
    statistics move to two lines, fan plots put the field above the
    tangential and sagittal plots, chart axes take about one label per
    70 pt and keep their end labels inside, the CRA legend moves under
    its picker, the Seidel Petzval radius under its picker and the
    wavefront coefficients wrap; wide segmented controls (polarisation
    quantity, image comparison) become pop-up menus (`SegmentedOrMenu`),
    so no pane overflows into its neighbour.
  * *Run details* (`RunDetails`, `RunRoute`): the backend that computed
    the result, what was requested, the engine's reason when the run was on
    the CPU, precision, seed, snapshot, engine version and time. Rows that
    carry nothing are left out. "Route" is the engine's
    `backend.cost_decision` of an automatic request in one line
    (docs/RESULTS.md §5): the backend chosen and how ("GPU by predicted
    time (GPU predicted 312 ms, CPU 557 ms)", "CPU: the GPU's predicted
    gain is too small (…)", "CPU: this analysis is not accurate enough on
    the GPU", "CPU: fixed for this analysis, without a prediction", "CPU:
    this request already failed on the GPU", "GPU: no timing data for this
    device, so no prediction"), with "; kept from the first identical
    request" for a repeated one; an explicit request and a decision the
    app does not know have no such row. "Tier" appears for a run that was
    not design tier. Backend attempts that did not compute the result
    (`run_info.failed_attempts`) are listed under "Attempt Not Used", one
    row each in the warnings' style with the engine's reason below: "GPU
    failed after 12 ms; completed on the CPU", "GPU not tried: the same
    input already failed on it", "GPU request run on the CPU", "GPU path
    completed the result on the CPU (1.8 s)".
  * *Warnings* (`RunWarnings`): the engine's run warnings are grouped by
    message (numbers masked), with the fields and wavelengths they
    concern numbered from 1 as everywhere in the app (`Field 1, Field 3`).
    They are readable without a click (the badge's help tag lists them),
    in VoiceOver (the badge's label reads them) and in the run details
    popover, which also lists the engine's raw messages and offers a
    one-click remedy where the analysis has the setting: an explicit coarse
    pupil grid, image grid or image-simulation PSF window goes back to the
    engine's automatic choice ("Sample the Pupil Automatically (grid 32 →
    auto)", "Size the Image Grid Automatically", "Size the PSF Window
    Automatically"). An automatic setting that still warns has reached the
    engine's cap, so no button is offered.
  * *Automatic settings* (`AutoSettings`): the engine chooses the pupil
    grid, the PSF / line-spread / encircled-energy image grids, the PSF
    spacing, the MTF frequency limit, the fixed frequencies of MTF vs field
    and through-focus MTF, the through-focus range and the image
    simulation's PSF window and spacing from the system when they are 0,
    null or empty (docs/WAVE.md). The settings form shows such a
    setting as "Auto (n)" with the value the last run used, read from its
    result (`Auto (282–371)` for PSF windows that differ by field, `Auto
    (2, 5, 10)` for frequencies, `Auto (±0.437)` for a focus range), never
    as a bare 0; clearing a field, typing 0 or "auto" makes it automatic
    again. The pupil object has its own compact control (grid and rays), and
    the image simulation's `optics` one (PSF window, spacing, pupil grid,
    rays, PSF grid). Values the engine does not report (the pupil grid of
    MTF and PSF, the encircled-energy grid) read "Auto". Stale: dimmed, desaturated, slightly blurred, with a glass card
  and a prominent Update; updating: a progress line and a gentle pulse. The
  dedicated views are: first order with the anamorphic note, spot,
  fans, field curvature, Seidel with S_I … S_V and C_L, C_T set with real
  subscripts in its legend and key numbers (`Subscripted`), MTF, PSF, wavefront, ghosts, stray light,
  polarisation, …); everything else uses the generic view.
* **Field maps, glass map and image simulation** have dedicated views.
  `pupil_aberration`, `relative_illumination_map` and `chief_ray_angle`
  share one reader (`FieldMapData`). A grid run is a heat map over the
  normalised field, interpolated between the sample dots, with a column
  picker that shows each column's definition (beside it, or below in
  narrow panes); an x or y sweep is a curve. **Heat-map scales**
  (`HeatMapScale`, one rule for every map of a quantity): signed data
  keeps the diverging blue–red scale centred on 0 and symmetric about it;
  other data takes the perceptual sequential ramp (viridis from its 0.1
  step: lightness rises monotonically, readable in light and dark and
  with colour-vision deficiencies) over the data's own range, so a 0.85…1
  illumination map shows its fall-off; fractions offer "0 to 1" to
  compare designs; a flat map gets a narrow band around its value. The
  polarisation pupil maps and generic grids follow it; PSF and irradiance
  images keep the one-hue magnitude ramp whose zero recedes into the
  plot. Cells the engine reports as null are no data, never 0: the
  relative illumination map's cells outside the field (hypot(Hx, Hy) > 1,
  image points included) leave a fine hatch showing through, the map is
  interpolated from the finite corners and clipped at the field's edge,
  and its iso-lines are traced on that clipped grid, so they reach the
  edge and never enter masked cells. Relative illumination adds iso-lines (marching squares, about
  five round levels inside the data's range, white on a dark halo,
  labels kept apart) and the illumination against image radius. The chief ray angle is plotted against image height with the
  sensor limit; the margin between them is shaded and turns red where the
  limit is exceeded, and asymmetric systems open on a map. The glass map
  (`GlassMapChart`, reused by Glass Substitution) draws the catalogue in the
  n_d–V_d or P_g,F–V_d plane with V_d decreasing to the right; status is
  shown by colour and, for obsolete glasses, by a hollow mark; the design's
  glasses are ringed and labelled with their surfaces; hover shows a glass's
  data and a click inspects it. The image simulation shows the simulated
  sensor image beside its source, side by side, split or blinking, with
  exposure, zoom and pan. The simulated patch is a square of the
  normalised field around a centre (`optics.field_center`) chosen in the
  view: on axis, at a design field point (normalised by the largest radial
  field), or anywhere on a small field diagram that a click or drag moves
  (snapping to the axis and the field points, warning when part of the
  patch leaves the field). The footer names the patch ("At Field 2 (0.00,
  +0.71)", with its image point in the help) and the PSF window the run
  used ("PSF 505 × 0.39 µm"). New panes start on a visible patch
  (`ImageSimulationSettings.starting`): 10 % of the field on axis at 256 px,
  with the engine's automatic PSF window and pupil (the engine's own view,
  the inscribed square at 64 px, has pixels larger than the spot). The
  view's controls set the field of view (full field = the square inscribed
  in the field disk, ½, ¼, 10 %, 5 %, 2 %, each with its width and pixel
  size on the sensor), the centre and the resolution (128 … 1024 px). Image
  simulation starts manual (seconds per run). The view's first control is
  the source menu: the built-in patterns or "Choose Image…"; an image file
  can also be dropped on the images. Built-in patterns go to the engine as
  `{kind: pattern}`, which evaluates them analytically at every pixel (so
  the Siemens star's spokes stay sharp and only the optics blur its
  centre); the source shown beside the result is drawn by the app with
  the engine's definition, area-sampled 4 × 4 per pixel. A user image (any
  format ImageIO reads) is cropped to the field, resampled in linear light
  and sent as a buffer, and the project keeps only its path. The images
  are clipped to their area for hit testing as well as drawing, so a
  zoomed image never takes the clicks of the controls around it. The footer has the pixel size, the largest distortion and
  the lowest relative illumination of the field mapping, which a popover
  lists node by node. The settings form has compact controls for two-number
  settings (grid, extent, size; a field point such as grid distortion's
  `field_center` reads "Hx … Hy …" and accepts signed values), a table for the sensor CRA limit and a
  source menu.
* **Spot diagram** (`SpotGrid`, `SpotGridView`): a field × focus grid,
  fields as rows and focus planes as columns, so the blur before, on and
  after focus is visible at once. A new pane samples three planes over the
  engine's automatic range (`through_focus: {min: null, max: null, steps:
  3}`; Δ = max(8 λ N², 4 N · RMS), the depth of focus or the geometric
  blur); the settings form has the through-focus control (off, or ±Δ with
  "Auto (0.193)" and the number of planes, 1 to 15). The engine returns
  the points once, at the base defocus, with a `point_slope` buffer, and
  the view places every point on every plane as `points + (d − defocus) ·
  slope` in its initialiser, never in `body`. Column headers name the
  plane, row headers the field, each cell its RMS about the reference
  from the engine's per-plane statistics. The box scale is common to the
  grid or per field (a small segmented control), stated under the grid.
  Markers are sized by density and cell size (`SpotMarker.style`: about
  1 to 2.5 pt across, more translucent in dense patterns) and drawn as one
  path per wavelength. All columns stay side by side; rows scroll, cells
  are 48 to 420 pt, and the pane scrolls sideways only below 48 pt.
* **Run-local spectrum.** Every analysis whose schema has `spectrum` shows
  a spectrum control (`SpectrumControl`) in its settings popover, with a
  compact "Spectrum: …" button in the pinned bar: the design's wavelengths
  (the setting is left out), Photopic V(λ), Scotopic V′(λ), a blackbody
  (temperature in K, quick picks 5500 K and 3200 K), explicit wavelengths
  (an editable list seeded from the design), or a sensor response imported
  from CSV (two columns, λ and response; optional header; comma, semicolon
  or tab; nm detected by magnitude or header; unreadable rows listed by
  line). Presets, blackbody and sensor take the engine's sampling (samples,
  range, equal-energy or uniform spacing). A small chart shows the weights
  the run will use in spectral colours with the primary marked, resolved by
  the engine (`system_summary` with the spectrum) or, until it answers, a
  local preview labelled approximate. The pane summary reads e.g.
  "Spectrum photopic". A spectrum pane is current while its
  `run_info.snapshot_hash` matches like any other (the engine reports the
  run-local system separately as `derived_snapshot_hash`).
* **Inspector** (`.inspector`): its pages are a row of symbols in its part
  of the toolbar (`InspectorTabBar`, as in Xcode): Surface and System, then
  the design jobs (Optimize, Glass Substitution, Tolerance, Apply Solves),
  a running job's tab marked with a dot. `WindowState.page` is the
  selection; a job's Done returns to the surface or system page. The menu
  commands and the status capsule open a hidden inspector on a job's page.
  The inspector stays built while hidden (building its form on every open
  stalled the window for ~120 ms) and does not follow the selection then;
  both window columns are rooted in a `ColumnLayout`, which answers the
  split view's size queries from fixed bounds, so opening or closing a
  panel does not squeeze the inspector. The slide itself still runs below
  60 fps with analysis panes open (AppKit lays out their platform views on
  every frame; freezing the panels' width during the slide was measured
  and made no difference). Surface: title with a stop badge, the type
  menu, the type's summary and preview, then the data-driven sections,
  spacing and medium, aperture, placement, coating, solves. A diffractive
  surface adds Diffraction Efficiency: the order-efficiency model (ideal,
  kinoform, multilevel with its number of levels), the trace order's
  efficiency at the design wavelength and, estimated with the blaze
  m₀λ₀/λ, at the system wavelengths as bars. Previews:
  rotational shapes a sag profile; toroid and biconic x and y profiles; XY
  and Zernike x and y profiles plus a false-colour departure map (diverging
  blue–red, PV); DOE the phase profile in waves, its paraxial power at the
  primary wavelength and trace order (and at the design wavelength and
  order), focal length, smallest zone period and zone count.
  System: name and notes, aperture, fields, wavelengths (with presets),
  environment, ray aiming, configurations and glass catalogs. Switching the
  field type or the aperture type converts the numbers so the system stays
  the same (the engine's `convert_spec`), as one undoable step: each field
  keeps its chief ray (a real image height is the real chief ray's, at the
  primary wavelength), weights and vignetting stay; the aperture keeps the
  paraxial marginal ray, and "Float by stop size" fixes an automatic stop
  semi-diameter at the beam's current size (a fixed one stays). Where no
  equivalent exists (an object height or NA with the object at infinity)
  the numbers stay and validation says why. Configuration overrides of
  field values are not converted.
* **Design jobs** are inspector panels (`JobPanel`) with a header, a
  scrolling body and a glass action bar whose buttons morph between states.
  *Optimize*:
  * **Merit function** (`MeritFunctionSection`), Default or Custom, with the
    merit at the current design read out live. *Default* edits the engine's
    generator: criterion RMS spot (centroid or chief, per-wavelength or
    white-light centroid), RMS wavefront, or MTF (frequencies, target,
    diffraction or geometric, weakest field); rings and arms; fields and
    wavelengths as checklists; glass and air thickness limits; the
    anamorphic squeeze target. *Custom* is an operand list: a searchable
    picker grouped by category (first order, geometry, real rays, image
    quality, ghosts and stray light, manufacturing, composition); each row
    shows the operand, its parameters, comparison and target, weight, its
    current value and its share of the merit as a bar; the selected
    operand's parameters, field, wavelength, comparison (=, ≤, ≥), target,
    weight, constraint role and enable are edited below the list (the
    row's summary leads with its field and wavelength); add after the
    selection, delete (⌫), duplicate, drag or ⌥⌘↑/↓ to reorder, with
    `combine` references following their operands; local problems and the
    engine's operand errors show inline. *Convert to Custom* takes the
    expanded merit the engine evaluated (`merit_spec`: the generated
    operands, with the spec's failure policy kept in the saved spec); the
    EFL hold stays with the "Hold the focal length" switch, which adds it
    again, so the custom merit equals the default's. Every change is
    undoable and saved in the project's `merit` section.
  * **Live evaluation** (`OptimizeSession.scheduleEvaluation`): the
    engine's `merit_evaluation` with exactly `optimize`'s settings (merit,
    variables, `hold_efl`, algorithm), on the design's snapshot, 200 ms
    after the last edit; a newer edit cancels the pending request, an edit
    undone before its answer drops it, and the readout dims from the edit
    until the answer. The merit, every operand's value, its contribution
    (weight × residual²) and its share come from the engine's report, a
    failed operand with its error; the readout's help names the evaluator
    and a degraded or failed status. On the Double Gauss an MTF or
    RMS-spot evaluation takes 40–110 ms (CPU f64) on an idle machine.
  * **Solves and variables.** `optimize` and `merit_evaluation` apply the
    solves first and refuse a solve-controlled parameter as a variable. A
    solved value's `V` toggle (inspector) is struck through, its help says
    which solve sets it and how to vary it; the table's Variables menu and
    ⌃⌘V refuse it with that message. A variable left on a parameter that
    later got a solve keeps an orange `V` in the table, is flagged "Set by a
    solve: not varied" in the bounds list (the solve in its help) and is
    left out of the run; a run whose variables are all pinned or solved is
    blocked with an explanation. `curvature` and `shape_param
    curvature_y` name one parameter of a toroid or biconic
    (`OpticalSystem.canonical`), so variables and solves match by either.
  * **Variables and bounds** (`VariablesBoundsList`): every design variable
    with its current value, inline min and max (empty is unbounded) in the
    unit of the row's unit column, pin (equal bounds: kept but held, and
    left out of the run), remove; out-of-bounds values, automatic
    semi-diameters, solved parameters and vanished parameters are flagged.
    **Radii are bounded as a radius window in mm** by default (`RadiusBounds`; a
    "Curvatures as Radius mm | Curvature 1/mm" switch shows the engine's
    curvature bounds instead, and each row's help gives the bounds in the
    other unit). The window `min … max` rises from min to max on the
    projective line of radii: 40 … 60 is c ∈ [1/60, 1/40] and −60 … −40 is
    c ∈ [−1/40, −1/60]; min above max wraps through flat (R = ∞), so
    100 … −100 is |R| ≥ 100, the curvature interval [−0.01, 0.01] through
    0 (the caption says "through ∞"); an empty side is open (50 …
    empty is R ≥ 50, c ∈ [0, 0.02]); 0 is the limit R → 0, leaving the
    curvature unbounded on that side; ∞ … ∞ holds the surface flat. A
    window containing R = 0 (−50 … 50) is no curvature interval and is
    refused with an explanation. New variables get
    sensible bounds: air gaps and the back focus ≥ 0, glass thickness ≥ 1
    mm, model n_d 1.4…2.1 and V_d 15…100, melt offsets Δn_d ±0.05 and ΔV_d
    ±10, semi-diameters and normalisation
    radii > 0.
  * Algorithm, iterations and seed; live iteration, search merit,
    improvement and a log-scale merit sparkline; **Stop** keeps the best
    design so far, Cancel discards. The result is previewed in the window
    (layout and analyses, with a before/after layout comparison) and lists
    every variable before → after; Apply to Design is one undoable step.
  *Glass Substitution* (`GlassSubstitutionPanel`, the engine's
  `glass_substitution`): the glass gaps as checkboxes with their current
  glass (none ticked snaps every model glass); candidates for every lens or
  per lens in plain words (catalogues, status, the n nearest or an
  n_d/V_d window, always include, never use, and transmission, cost,
  climate and stain limits); Quick (tabu) or Thorough (global) with the
  number of combinations, a time limit, shape iterations per combination
  and how many to keep; the merit and variables are the Optimize panel's.
  Live: combinations so far, the best merit, the best glasses, a sparkline.
  Stop keeps the ranking, Cancel discards. The result is a ranked list
  (merit, improvement, glasses, re-optimised variables); selecting a row
  previews that combination's system in the layout and every analysis and
  shows a before/after card (glasses per surface with n_d / V_d, merit,
  shape changes, the two layouts) and the glass map with the candidate's
  glasses ringed. Apply to Design is one undoable step. A ticked glass with
  a melt offset shows it in its row, and a note says that a replacement is
  installed without it, even the same glass (the offset belonged to the
  original melt); offset and model-glass variables of ticked glasses are
  left out of the re-optimisation, which the engine would refuse. The
  before/after card shows the offset on the glass it leaves.
  *Tolerance* (`TolerancePanel`, the engine's `tolerance`):
  * **Tolerance set** (`ToleranceSettings`, saved in the project's
    `tolerances` section as the engine's `ToleranceSpec`, `schema_version`,
    `operands`, `compensators`, `criterion`, `monte_carlo` (with
    `percentiles` and `record_trials`), as last expanded, plus the app's
    rows and the test wavelength under `app`; a spec written elsewhere is
    read back as rows, element motions and irregularity included).
    *Default Tolerances…* generates a set from a class of manufacture,
    Commercial / Precision / High precision, with every value editable (0
    leaves a kind out), in three groups: surfaces (power in test fringes
    over the clear aperture, Δc = Nλ/h²; astigmatic and spherical
    irregularity in test fringes; thickness; decentre and tilt of each
    surface alone), elements (decentre and tilt of each element as a rigid
    body, a cemented group together) and glasses (index Δn_d and Abbe
    number in % of V_d of every glass), and the trial distribution. It
    covers every lens surface, every gap inside the lens (glass, air, the
    stop's; the back focus is the compensator), every element and every
    glass. **Test fringes at 632.8 nm** is one setting for the whole set,
    under the table's toolbar: power and irregularity are counted at it,
    and it is sent as `irregularity_wavelength_um`. The table is grouped by
    element (`L1 · N-SK16`, `Air after L1`, `Stop S6`, collapsible): kind
    (short titles; the full label is the tooltip), surface (a menu; `L2`
    for an element), one ± field or − and + fields, and the unit (fringes,
    ΔR mm or % of R for radii; ′ or ° for tilts; % or ΔV for Abbe),
    changing which converts the limits. Add has a Surface section (by kind
    and surface) and an Element section (by kind and element);
    duplicate (⌘D) and delete (⌫); the context menu sets the unit, the
    distribution (uniform, normal with the limits at ±2σ or ±3σ, end
    points) and symmetric / asymmetric limits. Problems show under their
    row in red and block the run. Every change is undoable and saved; rows
    keep their identity across undo (aligned by content).
  * **Element tolerances** (`element_placement`): the element's first row
    says which surfaces move ("S3 S4 S5 move together") and its pivot, a
    menu: first vertex (the default), last vertex or any surface. The
    surface menu picks another element, or the first and last surface of
    a custom range. Every element row starting on one surface shares the
    range and pivot (the engine requires it): an edit to one moves the
    group, a row added there joins it, and rows that disagree (a file
    written elsewhere) are flagged, as is a range that clashes with an
    element placement of the design itself.
  * **Surface decentres and tilts** move one surface: the engine returns
    to the axis after a surface without a placement of its own, so the
    design is sent as it is. **Index and Abbe** address a model glass's
    n_d and V_d, a catalogue glass's melt offset (`material_dn_d`,
    `material_dv_d`, added to the saved offset; a limit that would take the
    offset past the engine's bounds is flagged). Abbe in % needs the
    glass's V_d: the panel asks the engine (`glass_info`) once per glass.
  * **Compensators**: the back focus (±0.5 mm) by default; any air space,
    design variable, or a surface's decentre, tilt, thickness or curvature
    from a menu, each with a ± range around nominal (empty is free).
  * **Criterion**: RMS spot radius (centroid or chief), RMS wavefront, MTF
    at frequencies (diffraction or geometric, weighted or weakest field;
    the shortfall from 1, so it never saturates) with the merit editor's
    rings, arms, fields and wavelengths controls, or the Optimize panel's
    merit; the engine's criterion (a mean of squares) is shown as a lens
    designer reads it: √ in µm, waves, or 1 − √ as an MTF. An optional pass
    limit gives the yield.
  * **Analysis**: sensitivity, Monte Carlo or both, trials and seed, the
    levels (percentiles, "50, 90, 97.7" by default, any list in 0…100) and
    how many trials keep their perturbations and compensators (200 by
    default, at most every trial and 10 000). After a run, "Last run: 20 ms
    per trial, so 200 trials take about 4 s" estimates the next. While it
    runs: progress, the stage ("Monte Carlo 8048 / 20000", the trials
    counted from the progress), elapsed time, the time per trial (measured
    in the Monte Carlo; before it, the last run's or the sensitivity's time
    per evaluation) and the time left; Stop keeps the sensitivities and
    trials completed, Cancel discards.
  * **Results**: the engine's nominal (with a Monte Carlo alone too) and
    RSS prediction, the engine's percentile levels and the yield; the
    sensitivity table by share of the RSS with a bar, the worst offenders
    (the largest shares making up half of it) tinted, a click selecting
    the surface in the lens table, and an eye that previews that build
    (the tolerance at its worse limit). The Monte-Carlo histogram with the
    cumulative share of builds (right axis), the nominal and the levels
    marked: a click on a bar lists its trials (else the worst trials), each
    with its value and recorded compensator move; a click on a trial
    previews it, and the previewed trial is marked under the histogram.
    Then the time per trial, the compensators' nominal value, the range the
    sensitivity evaluations used against the allowed range (a bar, a
    warning at the limit) and their Monte-Carlo mean and σ. Export copies a
    text summary or saves the sensitivity table or the trials (with the
    recorded compensators and perturbations) as CSV.
  * **Previews** of a trial or a sensitivity case are rebuilt by the
    engine (`preview_case` with the run's own settings and system), so the
    build shown is exactly the one evaluated: read-only in the layout and
    every analysis ("Previewing Tolerance: trial 298, compensated · not
    applied"). A card above the sensitivities shows the build's criterion,
    its compensators (value and move from nominal) and its perturbations
    in the rows' units, largest share of their limit first, each with a
    bar from −limit to +limit; End Preview returns to the design. Labels
    are the app's own, by row position, never the engine's.
  * The layout of a toleranced build has no line exactly in the section
    plane, so the engine flags none in section; the reader then takes the
    profiles, stops, edges and rays within 2 % of the scene's height of the
    plane (`EngineLayout.markNearSection`), so the section view still
    draws the build.
  *Apply Solves*: a dry run (now →
  solved, residual), then one undoable commit.
* **Lens prescription files**: `.zmx` and `.seq` files open as designs
  (File ▸ Open…, the Welcome window, drag and drop onto any MKLight window,
  Finder's Open With). `ProjectDocument` reads them through
  `EngineClient.importPrescription` (the engine's importer, `IMPORT.md`);
  they are declared as imported types with the Viewer role and are never
  written, so saving asks for a new `.mklight` file. When the import left
  something out, the window opens with the *import report* sheet
  (`ImportReportSheet`): the warnings grouped as Problems, Glasses,
  Surfaces, Not Imported and Assumed or Approximated, each with its surface
  and line, Copy Report, and, when glasses were not found, Load Glass
  Catalog…, which loads a catalog and imports the file again (undoable).
  A clean import shows a short note instead.
* **Welcome window**: New Design (prominent glass) and Open (glass), and the
  engine's sample designs as cards with thumbnails of their layouts,
  computed by the engine on a session of their own (`SampleThumbnails`);
  samples with extended surface types are listed under "Surface Types".
  Slim, long designs are framed on their lens.
* **Layout persistence**: the panes (analysis, settings, automatic update,
  pinned settings), the panel arrangement of every window, viewport mode
  and ray colouring are saved in the project's `views` section (`[{key,
  analysis, settings, ui}]`), written with the next save, not as an
  undoable design change. The arrangement is the layout entry's
  `ui.arrangement`: `{"main": node, "windows": [{"frame": [x, y, w, h],
  "root": node}]}` with nodes `{"tabs": [keys], "selected": key}` or
  `{"axis": "h"|"v", "children": [nodes], "fractions": [shares]}`; keys are
  `lens_data`, `layout` or a pane key, frames in screen points. Views
  without an arrangement open in the standard one.

Keyboard: ⇧⌘I / ⌥⌘I insert before / after, ⌘⌫ delete, ⇧⌘T make stop,
⌃⌘V toggle variable, ⌃⌘O optimize, ⌃⌘G glass substitution, ⌃⌘T tolerance,
⌃⌘L apply solves, ⌃⌘E issues, ⌘R run analyses, ⌘. cancel, ⌘2 / ⌥⌘2 / ⌘3
y–z section / x–z section / 3-D, ⌘0 fit, ⌥⌘L layout, ⌥⌘E lens data,
⌃⌘I inspector, ⌃⌘S sidebar. View ▸ Panel has Move to New Window, Move to
Main Window, Close / Hide and Reset Panel Layout for the focused panel. In a job panel Return is the primary action and
Esc closes the panel.

## Surface types

Every engine shape (docs/SURFACES.md) is mirrored in `MKLightModel`
(`SurfaceShape`, strict and round-tripping, tested against the engine's
`extended_shapes.json`, `samples.json`, `param_paths.json` and
`validation_cases.json`) and has a `SurfaceTypes` descriptor:

| group | type | parameters (inspector) | variable paths |
|---|---|---|---|
| Spherical | Plane, Sphere / Conic | radius, conic | `curvature`, `conic` |
| Aspheric | Even asphere | base conic; A₄ r⁴ … A₂₂ (10) | `asphere` |
| | Odd asphere | base conic; β₁ r … β₂₀ r²⁰ | `shape_coefficient odd` |
| | Q-type (conic) | base conic, norm radius; a₀ … a₁₉ u⁴Qₘᶜᵒⁿ | `shape_coefficient q`, `shape_param norm_radius` |
| | Q-type (best-fit sphere) | BFS radius, norm radius; a₀ … a₁₉ Qₘᵇᶠˢ | idem |
| Freeform | XY polynomial | base conic, norm radius; C(xᵐyⁿ), degree 1–10 (65), any term from a menu, zero terms hidden | `shape_coefficient xy` |
| | Zernike sag | base conic, norm radius, ordering (Fringe 37 / Standard 45); Zⱼ with names | `shape_coefficient zernike` |
| Anamorphic | Toroid | y profile (R_y, k_y, B₄ y⁴ …), rotation R_x | `curvature` (the y section), `shape_param curvature_x, conic_y`, `shape_coefficient y_asphere` |
| | Biconic | x and y sections (R, k, A x⁴ … / B y⁴ …) | `curvature` (y), `shape_param curvature_x, conic_x/y`, `shape_coefficient x_asphere/y_asphere` |
| | Cylinder (x or y) | a biconic preset ("Make Cylinder") | as biconic |
| Diffractive | Diffractive | base (plane, conic, even asphere) and its parameters; phase norm radius, A₁ ρ² … A₁₀ ρ²⁰ (rad); design order, design λ, trace order | `curvature`/`conic`/`asphere` address the base; `shape_coefficient phase`, `shape_param norm_radius`; orders are never variables |
| Special | Ideal lens, Coordinate break | focal length; placement | `paraxial_focal_length`, decenters and tilts |

Changing the type keeps what carries over (curvature, conic, norm radius;
a DOE wraps the old shape as its base) and drops configuration overrides
that no longer resolve, as one undoable step.

**First order of anamorphic designs.** The First-Order pane, the status
capsule, its popover and the System inspector read the engine's
`paraxial_xy` and `symmetry`. A design that is not rotationally symmetric
shows EFL and BFL in x and y, the signed squeeze EFL_y/EFL_x (the engine's
definition as help), the symmetry class (meridional or none), DOE powers,
the engine's warnings, and the other wavelengths in a disclosure. The
capsule shows EFL y and the squeeze. The surface kinds decide only when an
engine does not report the symmetry. Automatic semi-diameters of every
surface type come from the engine's layout, in the table and the previews.

## Building

```sh
scripts/build-macos.sh --engine-only   # engine library → apps/macos/Frameworks/libmklight_ffi.a
cd apps/macos && swift build && swift test

scripts/build-macos.sh                 # release app → build/macos/MKLight.app
scripts/build-macos.sh --debug         # debug app (launch options below)
scripts/build-macos.sh --universal     # engine for arm64 + x86_64 (rustup target x86_64-apple-darwin)
scripts/build-macos.sh --no-engine --engine-only   # remove the library again
```

* The script builds `mklight-ffi` as a **release static library** (deployment
  target macOS 27, like the app bundle's `LSMinimumSystemVersion`) for
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
  engine" mark in the status capsule and the welcome window). A release build
  without it shows an alert and quits.
* Debug builds accept launch options for screenshots and UI checks:
  `-MKLightNoWelcome YES`, `-MKLightSidebar hidden`, `-MKLightSelectSurface
  <i>`, `-MKLightInspector system`, `-MKLightViewport 3D`, `-MKLightSheet
  optimize|tolerance|solves` (opens the job panel), `-MKLightAutoRun
  <algorithm>|tolerance` (an Optimize algorithm: `lm`, `dogleg`, `de`,
  `cma_es`, `nelder_mead`, `hammer`), `-MKLightShowIssues YES`, `-MKLightToast <text>`,
  `-MKLightAppearance light|dark` (overrides the system appearance),
  `-MKLightSheet glass`, `-MKLightAutoRun glass` (and `-MKLightAutoPick n`),
  `-MKLightAutoStop s` and `-MKLightAutoApply s` (press Stop / Apply after
  s seconds), `-MKLightToleranceTrials n` and `-MKLightTolerancePreview k`
  (with `-MKLightAutoRun tolerance`: the trials, and the k-th largest
  sensitivity previewed once the run ends; `trial` previews the worst
  trial), `-MKLightFootprints YES`, `-MKLightFit lens`, `-MKLightYaw
  <radians>` (the 3-D view's yaw), `-MKLightRunDetails <analysis>` (opens
  that pane's run details), `-MKLightAutoEdit s` (edits the merit s seconds
  after the Optimize panel opens and restores it a second later) and
  `-MKLightAutoConvert s` (presses Convert to Custom), and
  `-MKLightCurvatureBoundsUnit curvature` (the bounds list's unit, an
  `@AppStorage` preference). Pane layouts come from the document's
  `views`.

## `FFIEngine`

`FFIEngine` (module `MKLightFFI`) implements `EngineClient` over the C ABI.
Every C function the app uses is wrapped once in `CABI.swift`, which
converts strings, frees what the library allocated and turns a failed
`MKStatus` into an `EngineError` with the engine's message (read from
`mk_last_error()` on the same thread). The app has no binding for the
engine's project calls (`mk_session_open`, `mk_session_save`, …): it reads
and writes `.mklight` files through `ProjectFile`, and `FFIEngineTests`
checks the two against each other.

| concern | how |
|---|---|
| lifetimes | `FFIEngineCore` owns `MKEngine*`, `FFISessionHandle` owns `MKSession*`; both free their handle in `deinit`. A session and every running job keep the core alive. |
| sessions | `FFIEngine` = core + one session. `makeDocumentEngine()` (`newSession()`) gives each open design its own session (user catalogs, library hash). |
| snapshots | `openSnapshot` sets the session's system (`mk_session_set_system_json`) and returns the engine's BLAKE3 hash and the library hash. An invalid model throws `.invalidModel` with the engine's message first, then the mirror's structural errors. |
| validation | The app runs the engine's `validate` analysis (issues with severity, code and surface) as a pane of its own on every snapshot. A system the engine refuses has no snapshot: its message and the mirror's checks are the error of `openSnapshot`. |
| solves | `applySolves(settings:to:)` calls `mk_session_apply_solves` and reads the solved system back; the report is the `solves` result. Engines without it fall back to the `solves` dry run plus `OpticalSystem.setValue`. |
| catalogs | `catalogs()`, `loadCatalog(at:)`, `glasses(in:)`, `materialNames()` (completion). |
| analyses | `analysisDescriptors()` lists every analysis with its settings schema (read once, when the engine is created). The app submits every analysis pane and design job by name (`EngineAnalysisRequest`, built by `Workspace.request(for:)` and `Workspace.submitJob`): the analysis, the settings the schema accepts and the backend preference. That is the only way to run an analysis; `MockEngine` answers the same requests in the engine's result format. |
| jobs | `submit` returns at once; the job's task does the rest off the caller's thread. Submission is one locked step on a global dispatch queue: set the snapshot's system if the session holds another, then `mk_session_submit_analysis_json`, registering the job's event stream before any event can be delivered. Events from the engine's `mklight-events` thread feed the job's `progress` stream; a 250 ms status poll is a safety net. The terminal event takes the result at once. `cancel()` calls `mk_job_cancel`; a job cancelled before it was submitted never reaches the engine and throws `EngineError.cancelled`. |
| stop | For the analyses whose descriptor is `stoppable` (the engine's listing), `AnalysisJob.canStop` is true and `stop()` calls `mk_job_stop` (ABI 1.2): the job ends normally with its best result so far (`result.stopped: true`). A stop requested before the engine job exists is sent right after submission. |
| results | The envelope's `run_info` decodes into `RunInfo`, `library_hash` into `AnalysisOutput.libraryHash`, `backend` into `AnalysisOutput.backend` (`BackendReport`: requested, used, precision, device, fallback reason, and the `cost_decision` object as sent). The envelope's `result` is `AnalysisOutput.json`, with the `ResultBuffer`s in `AnalysisOutput.buffers`; the views read both through tolerant accessors (`JSONValue.nf`, `nfArray`, `[path:]`, `AnalysisOutput.buffer(named:)`). |
| buffers | `ResultBuffer` points into the engine's memory (no copy) and keeps the buffer array alive; `mk_buffer_free` runs when the last buffer of a result goes. Typed access checks the dtype (`withUnsafeBufferPointer(of:)`); `buffer[[i, j]]` indexes row-major. |
| errors | `EngineError.from(code:message:)`: cancelled, invalid model and unsupported get their own cases; every other status is `.engine(EngineErrorCode, message)`, and a failed job arrives with the status of its own error (`.failed` is a job that panicked). Submission errors (unknown analysis, invalid settings) fail the job immediately. |

**Thread safety.** `FFIEngine` is a `Sendable` value; use it from any task.
The C engine and sessions are thread-safe. The asynchronous `EngineClient`
methods never block their caller: their C calls (parsing, validation, waits
for the session lock, setting the system and submitting, taking a result)
run on a global dispatch queue (`offMain`), never the main thread. These C
calls run on the caller's thread:

* creating and destroying handles: `mk_engine_create` (with
  `mk_engine_set_job_callback` and `mk_engine_list_analyses`) when the app
  starts, `mk_session_new` when a design's `Workspace` is created or its
  session rebuilt and for the welcome window's thumbnails,
  `mk_session_destroy` and `mk_engine_destroy` on the thread that drops the
  last reference;
* `mk_job_cancel` and `mk_job_stop`, from `AnalysisJob.cancel()` and
  `stop()`;
* `mk_engine_prepare_gpu`, which returns at once, and
  `mk_engine_gpu_preparation_json`, which `GPUPreparationMonitor` polls on
  the main actor;
* `importPrescription(_:format:)` (`mk_import_prescription`), which blocks
  its caller: `ProjectDocument` calls it while it reads a `.zmx` or `.seq`
  file.

Progress reaches the UI through `AsyncStream`s that the `@MainActor`
`Workspace` consumes. The engine's callback runs on its `mklight-events`
thread, never on the thread that submitted or cancelled the job, and with
no engine or session lock held.

**GPU kernel preparation.** Metal compiles the GPU's compute kernels when
they are first created (docs/FFI.md, "GPU kernel preparation"). The app
therefore prepares the kernels when it starts: `MKLightScene` calls
`GPUPreparationMonitor.start(engine:)` once per app run, which calls
`EngineClient.prepareGPU(cacheDirectory:)` with
`~/Library/Caches/<bundle identifier>/GPU Kernels` (`com.mklight.MKLight`
in the app bundle) and keeps App Nap off (`ProcessInfo.beginActivity`)
while it follows the preparation.

* *Engine.* `mk_engine_prepare_gpu` (ABI 1.4, `Engine::prepare_gpu`) runs on
  its own `mklight-gpu-prepare` thread; a second call does nothing.
  `mk_engine_gpu_preparation_json` reports the state (`GpuPreparation`): the
  states of docs/FFI.md plus `idle` before the first call; a panic in the
  preparation also ends as `failed`. When the engine is destroyed, a running
  preparation stops at its next kernel.
* *Swift.* `FFIEngine.prepareGPU` records a refusal of
  `mk_engine_prepare_gpu`, and `gpuPreparation()` then reads `.failed`, as
  it does when the state cannot be read. `GPUPreparation` mirrors the
  states, `ready` with the engine's optional `warning`. Engines without a
  GPU (`MockEngine`) keep the protocol's defaults: nothing to prepare,
  `.idle`.
* *Window.* `GPUPreparationMonitor` (`@MainActor`, `@Observable`, one per
  app) reads the state every 200 ms while it is `preparing`, for at most
  600 s (`followLimit`; a longer preparation keeps running in the engine
  but is no longer shown). A preparation still running after 400 ms
  appears at the leading end of the status capsule (`GPUPreparationBadge`)
  as a small spinner and "Preparing GPU n%" (kernels done of total), with a
  help tag; one that loads from the cache at once never shows. When a shown
  preparation ends ready, the capsule says "GPU ready" for 3 s. The status
  popover's Engine section has a "GPU kernels" row: "Preparing (d of t)",
  "Ready (n, loaded from cache)", "Ready (n, prepared in s s)", "No GPU:
  reason" or "Compiled on first use (message)". Kernels that are ready
  with the engine's `warning` (`GPUPreparationMonitor.note`) have a second
  line under the row: "Kernel cache unavailable: the next launch compiles
  again (message)".

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
  `VariableSpec`) and their changes, bounds and pins are undoable too; so
  is the merit function (`merit`: `{mode, default_merit, spec, hold_efl}`,
  `spec` an engine `MeritFunctionSpec`, keeping keys such as
  `on_failure` that the app does not edit).
* **File** (`ProjectDocument`, `ReferenceFileDocument`): reads and writes the
  engine's project format through `ProjectFile`; SwiftUI's document
  machinery provides autosave in place, versions and the dirty state (it
  follows the undo manager).
* **Engine snapshots** (`Workspace`, one per open design and shared by its
  windows, `@Observable`, platform-neutral): each design value is debounced
  (150 ms) and frozen into an immutable engine snapshot (system hash +
  glass library hash). Analyses run asynchronously against the snapshot
  they were submitted with; a newer request cancels the older job of the
  same pane; jobs end when the design's main window closes. Results are
  **stale** as soon as the snapshot, the library or the request (settings,
  backend, seed) differs; panes with automatic update re-run, the others
  wait for Update. Long analyses (stray light, ghost irradiance,
  through-focus MTF, MTF vs field and image simulation) start manual.
  The project's glass catalogs are loaded before the first snapshot;
  undoing Load Glass Catalog rebuilds the session without it. A **preview**
  (`Workspace.preview`) snapshots a candidate design (an optimisation or
  glass substitution result) instead of the document's, so the layout and
  every pane show it; the outputs of the design just before are kept as a
  baseline for comparisons; any design change ends the preview, and design
  jobs always run on the design's own snapshot (`designSnapshot`).
* **Window state** (`WindowState`): selection (by `SurfaceID`), inspector
  tab, viewport mode and camera, ray colouring, transient messages, the
  open job panel. It never enters the design; its layout part is saved in
  `views` (`WorkspaceLayout`).
* **Design session** (`DesignSession`, one per open design, owned by the
  main window): the document, the workspace, the window state and the
  panel arrangement (`PanelArrangement`: a tree of `PanelSplit`s and
  `PanelGroup`s per window, platform-neutral, unit-tested), and the
  separate windows. Every window of the design draws from it, so they stay
  in step; a pane added or closed anywhere is placed or removed by
  `reconcile()`. The arrangement's nodes are observable objects, so
  dragging a divider updates only that split.
* **Design jobs** (`OptimizeSession` with the merit editor's working copy
  and live evaluation, `GlassSubstitutionSession`, `ToleranceSession`,
  `SolvesSession`): view models that submit the engine's `optimize`,
  `glass_substitution`, `tolerance` and `solves` analyses, follow their
  progress, preview and apply results to the document. The
  window state owns one of each, so a job outlives its panel; closing the
  design's main window cancels the optimisation, glass substitution and
  tolerancing jobs and every analysis.

## Tests

`swift test` in `apps/macos` runs the package's four test targets
(`Tests/`, one directory each). `MKLightModelTests` and `MKLightEngineTests`
need nothing else; `MKLightUITests` runs the view models on `MockEngine`
and small test engines; `MKLightFFITests` runs against the real engine and
is skipped while the library is not linked (see "Building").

* `MKLightModelTests`: round trips against JSON written by the Rust side
  (every model variant including the extended shapes, every `ParamPath`
  form, solve kinds, the 16 samples, the project format, with the fixtures
  in `Fixtures/`), strict decoding, `ParamPath` resolution and values checked
  case by case against the engine's verdicts, shape-kind conversions, XY and
  Zernike term numbering.
* `MKLightEngineTests`: schemas and the settings they accept, buffers, error
  mapping, staleness, non-finite numbers; jobs on the mock (progress,
  cancellation, seeds) and its analyses by name; the mock's optics against
  known answers; every state of `mk_engine_gpu_preparation_json` in the
  engine's exact JSON; the envelope's `cost_decision` kept in
  `BackendReport`.
* `MKLightUITests` (view models on `MockEngine` and small test engines):
  * the workspace (panes run, go stale, automatic vs manual update,
    settings re-run, unavailable and unknown analyses, newer requests cancel
    older ones, the layout round-trips through `views`, catalogs before the
    first snapshot, session rebuild) and the panel arrangement (tabs,
    splits, separate windows, showing and hiding, saved with the project);
  * editing, surface types (every descriptor's keys and variable paths
    against the model for all 13 types, the display sags against the closed
    forms of SURFACES.md), the text-field rule and the lens table;
  * design jobs: Optimize applied as one undoable step, Stop keeps and
    Cancel discards, the merit editor, live evaluation, solved parameters
    and variable bounds, glass substitution, tolerancing (presets, the
    generator, element groups and pivots, percentiles and records, previews,
    export) and Apply Solves;
  * glass offsets, spectrum settings and CSV import, the settings form from
    the schema, the field-map, glass-map and image-simulation readers, the
    run details' route line on the engine's JSON of each decision kind, the
    spot diagram's field × focus grid, the import report, automatic settings
    read from results ("Auto (n)"), number formatting, heat-map scales and
    the GPU preparation's window status.
* `MKLightFFITests` (need the library), on the real engine: hashes, samples,
  non-blocking submission, progress, seeds, cancellation, error codes,
  project files between the engine and `ProjectFile`, catalogs, concurrent
  submissions; **every dedicated result view reads the real engine's result**
  for 25 analyses; Stop and Apply Solves; the default merit, its conversion
  to custom operands and solved variables; the glass substitution flow on the
  achromat; tolerancing on the Cooke (the whole default set, ranked
  sensitivities, Monte Carlo, trial previews, Stop); run-local spectra and
  the x/y first order; the settings form's mapping over every analysis the
  engine lists; `.zmx` / `.seq` import.
* `RenderSnapshotTests`, `ResultViewRenders` and `ResultViewRendersDirect`
  render panes, panels and the run details of recorded runs to PNGs (light
  and dark) for a visual check, only when `MKLIGHT_RENDER_DIR` names an
  output directory.

## iPadOS readiness

Ready:

* `MKLightModel`, `MKLightEngine` and the view models (`DesignDocument`,
  `Workspace`, `AnalysisPane`, `WindowState`, `WorkspaceLayout`,
  `LensEditor`, `SurfaceTypes`, `PanelArrangement`, the design-job
  sessions, result readers) import neither SwiftUI nor AppKit or UIKit.
* The window structure (`NavigationSplitView`, `.inspector`, toolbar
  items, popovers) and the Liquid Glass APIs used (`glassEffect`,
  `GlassEffectContainer`, `glassEffectID`, `.glass` / `.glassProminent`,
  `safeAreaBar`, `scrollEdgeEffectStyle`) are the same on iPadOS 27;
  `tabViewBottomAccessory` (iOS only) is not needed while the app has no tab
  view.
* The panel tree (`PanelArrangement`) is platform-neutral, and its views
  (`PanelViews`: splits with draggable dividers, tab groups, panel headers
  as drag handles) are SwiftUI with drag gestures. macOS-only are the
  drop-zone tracking while a panel is dragged (`PanelDrag`, AppKit events),
  the separate windows (`DetachedWindows`, `NSWindow`) and the resize and
  grab pointers (`resizePointer`).
* The Metal viewport has one bridge (`ViewportBridge`, `ViewportInput`) and
  two representables: `NSViewRepresentable` with mouse, trackpad and scroll
  input, and `UIViewRepresentable` with pan, two-finger pan, pinch and
  double tap.
* Platform specifics are isolated: dynamic colours (`PlatformColor`), the
  pasteboard (`Clipboard`), the option-click in the sidebar, glass-name
  completion (`materialSuggestions`), the resize pointer, file import uses
  `.fileImporter`. The welcome thumbnails, shape previews and sag maps are
  SwiftUI `Canvas` and Swift Charts.

Remaining for an iPadOS target:

* Add the target (an Xcode project or `.iOS(.v27)` in `Package.swift`)
  and build the engine as an iOS static library (`aarch64-apple-ios`, plus
  the simulator triple) with the same C module.
* The toolbar's centre status capsule needs a compact form (or a move to
  the inspector) on narrow iPad windows; the job panels should present as a
  sheet when the inspector cannot show.
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
* Tolerancing: element tilt z and irregularity orientation (the engine's
  astigmatism is x² − y², fixed) are not offered; a trial's perturbation of
  a radius in ΔR or % is shown scaled linearly from its limit. The trial
  list only shows compensator moves for recorded trials (any trial can be
  previewed). Pickups have no editor anywhere in the app, so melt offsets
  get variables and solve markers but no pickup UI.
* Configurations are listed but not editable; multi-configuration analyses
  use the `configuration` setting where the schema has one.
* Saving goes through the document machinery (atomic write by AppKit), not
  `mk_session_save`, so `relative_path` of catalogs is not recomputed on
  save. Layout changes alone do not mark the document dirty; they are saved
  with the next design change.
* The model mirror is strict: when the engine's model gains a field or a
  surface type, the mirror must learn it (then add a `SurfaceTypes`
  descriptor for a new type).
* Reduce Transparency and Increase Contrast were not checked in screenshots
  (they change a system accessibility setting); see principle 14.
* Engine gaps the UI works around:
  * `optimize` and `merit_evaluation` still refuse a variable on an
    automatic semi-diameter ("semi-diameter is automatic"), although
    `mk_session_get_param` reads its resolved value; the bounds list
    flags such a variable ("set a value in the lens data first"). A trace
    order of 0.5 is refused (the app rejects non-integer orders before the
    engine sees them).
  * Imported images go to `image_simulation` as JSON buffers (at most
    512 px a side); the engine's `path` source reads only PNG.
  * `diffraction_encircled_energy` does not report the image grid it chose,
    and `mtf`, `psf` and `line_spread` not the pupil grid; those settings
    read "Auto" without a value.
  * On the sample designs the automatic MTF limit `1/(2 r_rms)` is low for
    camera lenses: 19 cycles/mm on the Double Gauss and the Cooke, so MTF
    vs field and through-focus MTF use 2, 5 and 10 cycles/mm. The curves
    fill the plot, but a designer of a 50 mm F/2 lens usually looks at 10 to
    40 cycles/mm; the fields accept explicit frequencies.
  * `tolerance`: with irregularity in the set (the default) the job runs
    on the CPU reference (sag overlays are not packed for the GPU), which
    the backend report says. With the MTF criterion and a back-focus
    compensator a trial costs about 1 s on the Cooke (40 trials in under a
    minute, the time per trial shown). Engine labels use surface ids; the
    app labels operands itself by row.
  * `layout` of a slightly decentred or tilted system (a toleranced build)
    flags no polyline as in the section, so the reader keeps those within
    2 % of the scene's height of the plane (`markNearSection`); the engine
    could flag meridional lines with a tolerance itself.
  * `glass_map` reports no glass status unless AGF metadata is passed, so
    catalogue glasses show as "Catalogue glass".
  * LM with the MTF weakest-field merit and the EFL hold can end slightly
    infeasible (violation 2.7e-6 after 100 iterations on the Double Gauss),
    so the feasibility-first rule returns the start although the merit
    improved by a quarter.

## Appendix: sources of the design brief

The developer.apple.com pages, read through their JSON form:

* HIG: [Materials](https://developer.apple.com/design/human-interface-guidelines/materials),
  [Layout](https://developer.apple.com/design/human-interface-guidelines/layout),
  [Toolbars](https://developer.apple.com/design/human-interface-guidelines/toolbars),
  [Sidebars](https://developer.apple.com/design/human-interface-guidelines/sidebars),
  [Split views](https://developer.apple.com/design/human-interface-guidelines/split-views),
  [Panels](https://developer.apple.com/design/human-interface-guidelines/panels) (inspectors),
  [Scroll views](https://developer.apple.com/design/human-interface-guidelines/scroll-views),
  [Tab views](https://developer.apple.com/design/human-interface-guidelines/tab-views),
  [Color](https://developer.apple.com/design/human-interface-guidelines/color),
  [Typography](https://developer.apple.com/design/human-interface-guidelines/typography),
  [SF Symbols](https://developer.apple.com/design/human-interface-guidelines/sf-symbols),
  [Icons](https://developer.apple.com/design/human-interface-guidelines/icons),
  [Charts](https://developer.apple.com/design/human-interface-guidelines/charts),
  [Buttons](https://developer.apple.com/design/human-interface-guidelines/buttons),
  [Accessibility](https://developer.apple.com/design/human-interface-guidelines/accessibility),
  [Designing for macOS](https://developer.apple.com/design/human-interface-guidelines/designing-for-macos),
  [Designing for iPadOS](https://developer.apple.com/design/human-interface-guidelines/designing-for-ipados).
* Developer documentation: [Adopting Liquid Glass](https://developer.apple.com/documentation/technologyoverviews/adopting-liquid-glass),
  [Applying Liquid Glass to custom views](https://developer.apple.com/documentation/swiftui/applying-liquid-glass-to-custom-views),
  SwiftUI [`glassEffect(_:in:)`](https://developer.apple.com/documentation/swiftui/view/glasseffect(_:in:)),
  [`GlassEffectContainer`](https://developer.apple.com/documentation/swiftui/glasseffectcontainer),
  [`glassEffectID(_:in:)`](https://developer.apple.com/documentation/swiftui/view/glasseffectid(_:in:)),
  [`GlassButtonStyle`](https://developer.apple.com/documentation/swiftui/glassbuttonstyle),
  [`backgroundExtensionEffect()`](https://developer.apple.com/documentation/swiftui/view/backgroundextensioneffect()),
  [`scrollEdgeEffectStyle(_:for:)`](https://developer.apple.com/documentation/swiftui/view/scrolledgeeffectstyle(_:for:)),
  [`ToolbarSpacer`](https://developer.apple.com/documentation/swiftui/toolbarspacer),
  [`inspector(isPresented:content:)`](https://developer.apple.com/documentation/swiftui/view/inspector(ispresented:content:)),
  [`tabViewBottomAccessory`](https://developer.apple.com/documentation/swiftui/view/tabviewbottomaccessory(content:)),
  [Swift Charts `Chart3D`](https://developer.apple.com/documentation/charts/chart3d),
  [Observation](https://developer.apple.com/documentation/observation),
  [What's new in SwiftUI](https://developer.apple.com/documentation/updates/swiftui).
* WWDC: [Meet Liquid Glass (WWDC25 219)](https://developer.apple.com/videos/play/wwdc2025/219/),
  [Get to know the new design system (WWDC25 356)](https://developer.apple.com/videos/play/wwdc2025/356/),
  [Build a SwiftUI app with the new design (WWDC25 323)](https://developer.apple.com/videos/play/wwdc2025/323/),
  [What's new in SwiftUI (WWDC25 256)](https://developer.apple.com/videos/play/wwdc2025/256/),
  [Bring Swift Charts to the third dimension (WWDC25 313)](https://developer.apple.com/videos/play/wwdc2025/313/),
  [What's new in SwiftUI (WWDC26 269)](https://developer.apple.com/videos/play/wwdc2026/269/),
  [Modernize your AppKit app (WWDC26 289)](https://developer.apple.com/videos/play/wwdc2026/289/),
  [Principles of great design (WWDC26 250)](https://developer.apple.com/videos/play/wwdc2026/250/).
