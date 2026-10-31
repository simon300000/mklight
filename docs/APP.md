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
effects, `ConcentricRectangle`, `Observations`) is available without
`#available` checks, and building with the 27 SDK always gives the new
design: the `UIDesignRequiresCompatibility` opt-out is ignored from 27 on.
Build with Xcode 27.2 / Swift 6.4.2 (Swift 6 language mode, strict
concurrency).

## Design brief: Liquid Glass

What we took from Apple's current guidance, and how MKLight applies it.

**Sources read** (the developer.apple.com pages through their JSON form):

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
   menu command* (Toolbars, WWDC25 323). Design jobs | Add Analysis, Layout |
   Inspector, separated by `ToolbarSpacer(.fixed)`; the design's status in
   the centre (`.principal`). All have menu commands and shortcuts.
6. *Avoid critical information at the bottom of a Mac window* (Layout). The
   old status bar is gone: activity, EFL, F/#, total track and the issue
   count live in the toolbar's status capsule; engine, backend and snapshot
   are in its popover; transient messages float as a glass toast.
7. *Scroll edge effects instead of bar backgrounds; hard for pinned table
   headers, soft elsewhere; one per view* (Scroll views, WWDC25 356). The lens
   table uses `.hard` at the top; the inspector's Surface / System switch is
   a `safeAreaBar` over the form with a `.soft` edge; job panels have soft
   edges under their header and their action bar.
8. *Content extends under floating chrome* (Sidebars, Layout,
   `backgroundExtensionEffect`). The viewport draws edge to edge under its
   glass controls. We do not apply `backgroundExtensionEffect` to the Metal
   view: the effect mirrors its view's content, and copies of a live
   `MTKView` are neither cheap nor reliable; the sidebar's glass already
   floats over the window's background.
9. *Inspectors show the selection and update with it; less modality*
   (Panels, Designing for macOS). Design jobs moved from modal sheets into
   the inspector: the design, layout and analyses stay usable while a job
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
  conic. Everything type-specific lives in the inspector. A toroid's or
  biconic's radius and conic are its y section's, marked `y`.
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
  reference or GPU f32, with the fallback reason) and its warnings, one
  click away from the pane's backend badge. A result that no longer matches
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
┌ toolbar: ◧ title │ ⟳ EFL · F/# · TT · ⓘ │ Optimize Glass Tolerance Solves │ + Layout │ ◨ ┐
├ sidebar (glass) ─┬ Lens Data (hard scroll edge) ──────────────────────┬ inspector ─────────┤
│ Analyses         │ Surf Type Comment Radius Thickness Material …      │ [Surface|System]  │
│  (searchable,    ├─────────────────────────┬──────────────────────────┤ type menu, preview│
│  by category)    │ Layout (Metal, edge to  │ analysis panes …         │ data-driven form  │
│  Issues          │ edge) [Layout] [2D|3D]◉⤢│ title ⌄   ⟳ ⚡ ⚙ …        │ — or a job panel: │
│                  │                         │ plot (content)           │ Optimize, Tol.,   │
│                  │ 20 mm                   │ key numbers  [CPU · f64] │ Solves + actions  │
└──────────────────┴─────────────────────────┴──────────────────────────┴───────────────────┘
```

* **`NavigationSplitView`**: the sidebar lists every engine analysis by
  category (`.searchable` in the sidebar); a click shows it in the focused
  pane, ⌥-click or the context menu opens a new pane. Open analyses carry a
  dot, unavailable ones are dimmed with the engine's reason as help.
* **Toolbar**: three groups separated by `ToolbarSpacer`: the design jobs
  (Optimize, Glass Substitution, Tolerance, Apply Solves, each opening its
  inspector panel);
  Add Analysis and the layout toggle; the inspector toggle. In the centre,
  the **status capsule** (`DesignStatus`): a running job ("Optimizing ·
  479", click to show it) or analysis activity, EFL (EFL y for a design that
  is not rotationally symmetric), F/# and total track, and the issues button.
  Its popover has the first-order table, the anamorphic data, the engine,
  the snapshot and the compute backend picker.
* **Lens data** (`LensTableView`, SwiftUI `Table`): text cells edit in place
  with full keyboard navigation (Return / double click edits the row; Tab
  and ⇧Tab move across, ↑/↓ and Return move between rows committing, Esc
  reverts). Numbers accept `∞`/`inf`, and the material column completes glass
  names from the session's catalogs. The type column is a compact menu with
  SF Symbols, grouped Spherical, Aspheric, Freeform, Anamorphic, Diffractive,
  Special (cylinder presets under Anamorphic). Small inline markers show
  variables (`V`) and solves (`M`, `C`, `F`, `P`, …); validation issues badge
  their surface. Insert and delete are glass buttons in the header, and in
  the context menu and the Lens menu.
* **Layout pane**: the engine's `layout` scene in the Metal viewport, edge to
  edge, with glass controls floating over it (title chip, Section / 3D
  selector, display options, fit), a scale bar, and a glass "Design changed"
  button when stale. The engine draws every surface type (the model
  schematic is gone). For designs without rotational symmetry the selector
  offers Y–Z, X–Z and 3D; the layout's `section` setting follows it and the
  glass fills pair the profiles along the section's own axis. DOE surfaces
  carry a fine violet hatch (flag bit 2). In 3D the Display menu adds the
  engine's surface mesh (`mesh_rings`, flat shaded, the selected surface in
  the accent) and a Footprints button runs `footprint` and draws every
  surface's beam outline per field (`global_outlines`), filled
  translucently. Fit Lens frames the elements; the first fit of a long,
  slim scene does so by itself. A previewed candidate design shows a glass
  "Previewing … · not applied" label under the controls.
* **Analysis panes** (`AnalysisPaneView`): side by side with resizable
  widths. The header has the analysis menu and a one-line state or settings
  summary, and a glass cluster: Run (Cancel while running), automatic update
  (bolt), Settings (a popover with the schema-driven form; "Pin in pane"
  keeps it as a compact strip) and More. The plot is content. The footer has
  the key numbers and a backend badge (CPU · f64 neutral, GPU · f32 in the
  accent, an orange mark for warnings or a fallback) that opens the run
  details. Stale: dimmed, desaturated, slightly blurred, with a glass card
  and a prominent Update; updating: a progress line and a gentle pulse. The
  dedicated views are as before (first order with the anamorphic note, spot,
  fans, field curvature, Seidel, MTF, PSF, wavefront, ghosts, stray light,
  polarisation, …); everything else uses the generic view.
* **Field maps, glass map and image simulation** have dedicated views.
  `pupil_aberration`, `relative_illumination_map` and `chief_ray_angle`
  share one reader (`FieldMapData`). A grid run is a heat map over the
  normalised field, interpolated between the sample dots, with a column
  picker that shows each column's definition (beside it, or below in
  narrow panes) and the diverging scale for signed quantities; an x or y
  sweep is a curve. Relative illumination adds iso-lines (marching squares,
  0.5…0.95) on a fixed 0…1 scale and the illumination against image
  radius. The chief ray angle is plotted against image height with the
  sensor limit; the margin between them is shaded and turns red where the
  limit is exceeded, and asymmetric systems open on a map. The glass map
  (`GlassMapChart`, reused by Glass Substitution) draws the catalogue in the
  n_d–V_d or P_g,F–V_d plane with V_d decreasing to the right; status is
  shown by colour and, for obsolete glasses, by a hollow mark; the design's
  glasses are ringed and labelled with their surfaces; hover shows a glass's
  data and a click inspects it. The image simulation shows the simulated
  sensor image beside its source, side by side, split or blinking, with
  exposure, zoom and pan; built-in patterns are drawn as the engine defines
  them; a user image (any format ImageIO reads) is cropped to the field,
  resampled in linear light and sent as a buffer, and the project keeps
  only its path. The footer has the pixel size, the largest distortion and
  the lowest relative illumination of the field mapping, which a popover
  lists node by node. The settings bar has compact controls for two-number
  settings (grid, extent, size), a table for the sensor CRA limit and a
  source menu.
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
  "Spectrum photopic". A spectrum run records the derived system's hash in
  `run_info`, so a pane remembers the snapshot it submitted to (otherwise
  it would look stale forever).
* **Inspector** (`.inspector`): Surface or System, switched by a segmented
  control in a `safeAreaBar`. Surface: title with a stop badge, the type
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
    weight, constraint role and enable are edited below the list; add after
    the selection, delete (⌫), duplicate, drag or ⌥⌘↑/↓ to reorder, with
    `combine` references following their operands; local problems and the
    engine's operand errors show inline. *Convert to Custom* starts from
    the operands the engine generates for the default merit. Every change
    is undoable and saved in the project's `merit` section.
  * **Live evaluation.** The engine has no evaluate-only call, so the panel
    runs `optimize` with one LM iteration (debounced, cancelled by newer
    edits, on the design's snapshot) and reads its `start` merit and
    operand `start` values; shares are weight × residual² over their sum.
  * **Variables and bounds** (`VariablesBoundsList`): every design variable
    with its current value (radius and curvature for curvatures), inline
    min and max (empty is unbounded), pin (equal bounds: kept but held, and
    left out of the run), remove; out-of-bounds values, automatic
    semi-diameters and vanished parameters are flagged. New variables get
    sensible bounds: air gaps and the back focus ≥ 0, glass thickness ≥ 1
    mm, model n_d 1.4…2.1 and V_d 15…100, semi-diameters and normalisation
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
  glasses ringed. Apply to Design is one undoable step. *Tolerance*: ranked sensitivities with bars,
  Monte-Carlo statistics and a histogram. *Apply Solves*: a dry run (now →
  solved, residual), then one undoable commit.
* **Welcome window**: New Design (prominent glass) and Open (glass), and the
  engine's sample designs as cards with thumbnails of their layouts,
  computed by the engine on a session of their own (`SampleThumbnails`);
  samples with extended surface types are listed under "Surface Types".
  Slim, long designs are framed on their lens.
* **Layout persistence**: the panes (analysis, settings, automatic update,
  pinned settings, widths), split positions, viewport mode and ray colouring
  are saved in the project's `views` section (`[{key, analysis, settings,
  ui}]`), written with the next save, not as an undoable design change.

Keyboard: ⇧⌘I / ⌥⌘I insert before / after, ⌘⌫ delete, ⇧⌘T make stop,
⌃⌘V toggle variable, ⌃⌘O optimize, ⌃⌘G glass substitution, ⌃⌘T tolerance,
⌃⌘L apply solves, ⌃⌘E issues, ⌘R run analyses, ⌘. cancel, ⌘2 / ⌥⌘2 / ⌘3
y–z section / x–z section / 3-D, ⌘0 fit, ⌥⌘L layout,
⌃⌘I inspector, ⌃⌘S sidebar. In a job panel Return is the primary action and
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
| Anamorphic | Toroid | y profile (R_y, k_y, B₄ y⁴ …), rotation R_x | `shape_param curvature_x/y, conic_y`, `shape_coefficient y_asphere` |
| | Biconic | x and y sections (R, k, A x⁴ … / B y⁴ …) | `shape_param …`, `shape_coefficient x_asphere/y_asphere` |
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
  lm|de|tolerance`, `-MKLightShowIssues YES`, `-MKLightToast <text>`,
  `-MKLightAppearance light|dark` (overrides the system appearance),
  `-MKLightSheet glass`, `-MKLightAutoRun glass` (and `-MKLightAutoPick n`),
  `-MKLightAutoStop s` and `-MKLightAutoApply s` (press Stop / Apply after
  s seconds), `-MKLightFootprints YES`, `-MKLightFit lens`. Pane layouts
  come from the document's `views`.

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
  `VariableSpec`) and their changes, bounds and pins are undoable too; so
  is the merit function (`merit`: `{mode, default_merit, spec, hold_efl}`,
  `spec` an engine `MeritFunctionSpec`).
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
  undoing Load Glass Catalog rebuilds the session without it. A **preview**
  (`Workspace.preview`) snapshots a candidate design (an optimisation or
  glass substitution result) instead of the document's, so the layout and
  every pane show it; the outputs of the design just before are kept as a
  baseline for comparisons; any design change ends the preview, and design
  jobs always run on the design's own snapshot (`designSnapshot`).
* **Window state** (`WindowState`): selection (by `SurfaceID`), inspector
  tab, split fractions, viewport mode and camera, ray colouring, transient
  messages, the open job panel. It never enters the design; its layout part
  is saved in `views` (`WorkspaceLayout`).
* **Design jobs** (`OptimizeSession` with the merit editor's working copy
  and live evaluation, `GlassSubstitutionSession`, `ToleranceSession`,
  `SolvesSession`): view models that submit the engine's `optimize`,
  `glass_substitution`, `tolerance` and `solves` analyses, follow their
  progress, preview and apply results to the document. The
  window state owns one of each, so a job outlives its panel; closing the
  window cancels them.

## Tests

`swift test` in `apps/macos` (227 tests in 41 suites: model 69, engine 33,
FFI 35, UI 90):

* `MKLightModelTests`: round trips against JSON written by the Rust side
  (every model variant including the extended shapes in
  `extended_shapes.json`, every `ParamPath` form, solve kinds, the 16
  samples, the project format), strict decoding, `ParamPath` resolution and
  values checked case by case against the engine's verdicts, shape-kind
  conversions, XY and Zernike term numbering.
* `MKLightEngineTests`: schemas, request translation, tolerant decoding,
  buffers, error mapping, staleness; the mock's analyses by name.
* `MKLightFFITests` (need the library): BLAKE3 hashes, samples, EFL from
  engine data, non-blocking submission, progress, reproducible seeds,
  cancellation, error codes, projects, parameter edits, catalogs,
  concurrent submissions, handles; and `FFIDesignTests`: **every dedicated
  result view reads the real engine's result** for 25 analyses (geometric,
  wave optics, ghosts, radiometry, polarisation), the engine validates the
  extended shapes and its layout and Seidel refuse them with a clear
  error, the layout buffers become
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
  lists, parameter paths); surface types (every descriptor's keys and
  variable paths against the model for all 13 types, term labels, the type
  menu, the display sags against the closed forms of SURFACES.md — Q
  polynomials, reductions to the base conic, XY and Zernike terms — DOE
  power and zone period, the departure map, anamorphic table rows, the
  anamorphic first-order reader); design jobs (merit progress parsing, Optimize
  applied as one undoable step, Stop keeps and Cancel discards, tolerance
  ranking and Monte Carlo, Apply Solves dry run then undoable commit);
  parsers, viewport math and shaders; the merit editor (add after the
  selection, reorder, duplicate and delete with `combine` references kept,
  the operand JSON, contributions from the start values, undo and the saved
  section) and variable bounds (defaults, edit, pin, held when pinned, table
  marks, undo); glass substitution settings, progress and results; spectrum
  settings encoding and CSV import; the field-map, glass-map and
  image-simulation readers.
* FFI additions: the extended shapes run through the layout (x–z section,
  surface mesh, DOE flag) and Seidel; footprint overlays; the default merit
  evaluates and converts to custom operands with the same merit; custom
  operands report values; spectrum panes stay current; the glass
  substitution flow on the achromat (select, run, preview, apply, undo,
  redo) and Stop keeping the ranking; the new analyses' views read the
  engine's results; spectra and x/y first order on the real engine.

## iPadOS readiness

Ready:

* `MKLightModel`, `MKLightEngine` and the view models (`DesignDocument`,
  `Workspace`, `AnalysisPane`, `WindowState`, `WorkspaceLayout`,
  `LensEditor`, `SurfaceTypes`, the design-job sessions, result readers)
  use Foundation and Observation only.
* The window is `NavigationSplitView` + `.inspector` + toolbar items with
  placements and `ToolbarSpacer` + popovers; design jobs are inspector
  panels, which become a sheet on compact width. The Liquid Glass APIs used
  (`glassEffect`, `GlassEffectContainer`, `glassEffectID`, `.glass` /
  `.glassProminent`, `safeAreaBar`, `scrollEdgeEffectStyle`) are the same on
  iPadOS 27; `tabViewBottomAccessory` (iOS only) is not needed while the app
  has no tab view.
* The split layout (`SplitLayout`, `PaneRow`) is pure SwiftUI with drag
  gestures; only the resize pointer is macOS-specific (`resizePointer`).
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
* Tolerance sets and compensators are not editable yet (the engine accepts
  them); the Tolerance panel uses the engine's basic set.
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
* Reduce Transparency and Increase Contrast were not checked in screenshots:
  switching them changes a system accessibility setting, which this
  verification did not do. The app relies on the system glass adapting to
  them and uses no custom translucency of its own; Reduce Motion is honoured
  in code (`accessibilityReduceMotion`).
* Engine gaps the UI works around:
  * No evaluate-only merit call: the live merit runs `optimize` with one LM
    iteration (n + 1 merit evaluations) and reads its start columns.
  * A run with a `spectrum` records the run-local system's hash in
    `run_info.snapshot_hash`, which breaks the "current while the hash
    matches" rule of RESULTS §1; panes remember the snapshot they submitted.
  * `layout` and `ParamPath` still treat a few cases strictly: the
    `curvature` path refuses toroids and biconics (the app addresses their
    sections with `shape_param curvature_x/y`), reading an automatic
    semi-diameter through a path fails (the app shows the layout's value and
    flags such a variable), and a trace order of 0.5 is refused (the app
    rejects non-integer orders before the engine sees them).
  * `image_simulation`'s text chart and PNG-path sources come out
    vertically mirrored (their top row lands at the low-y edge); imported
    images go as buffers, which are correct.
  * `glass_map` reports no glass status unless AGF metadata is passed, so
    catalogue glasses show as "Catalogue glass".
  * LM with the MTF weakest-field merit and the EFL hold can end slightly
    infeasible (violation 2.7e-6 after 100 iterations on the Double Gauss),
    so the feasibility-first rule returns the start although the merit
    improved by a quarter.
