# Screenshots

The Liquid Glass design (docs/APP.md, "Design brief") on macOS 27, taken from
a debug build on the real engine (`scripts/build-macos.sh --debug`) with
`screencapture -l`, the engine's sample designs and pane layouts saved in the
documents' `views` section, and `-MKLightAppearance light|dark`; downscaled to
1600 px. Appearance in the file name. Reduce Transparency and Increase
Contrast were not captured (switching them changes a system setting).

| file | shows |
|---|---|
| `main-light.png`, `main-dark.png` | Double Gauss: glass sidebar, toolbar in three groups around the status capsule (EFL, F/#, TT, issues), lens data with type menus, layout viewport with floating glass controls, spot diagram and ray fan panes with glass control clusters and backend badges, surface inspector with the sag preview |
| `welcome-light.png`, `welcome-dark.png` | welcome window: New Design (prominent glass), Open, sample cards with layout thumbnails rendered by the engine; extended-surface samples with schematics |
| `layout-3d-dark.png` | the 3-D layout with the selected surface highlighted |
| `surface-diffractive-light.png` | hybrid refractive-diffractive singlet: DOE inspector with base picker, phase profile, paraxial power from A₁, smallest zone period, orders; the viewport's labelled schematic (the engine's layout does not draw extended shapes yet) |
| `surface-freeform-dark.png` | XY polynomial: x/y profiles, false-colour departure map, terms labelled C(x²), C(x²y)…, each with a variable toggle |
| `surface-zernike-light.png` | Zernike sag: Fringe ordering, Z₅ astigmatism 0°, Z₁₀ trefoil x, Add Term menu, departure map |
| `anamorphic-light.png` | cylindrical adapter: cylinder preset of the biconic, x and y sections, `y` hints in the table, the First-Order pane's anamorphic note (EFL x / squeeze not reported by the engine yet) |
| `toroid-dark.png` | toroid: y profile with B₄ y⁴, rotation radius, x/y sag profiles; the line focus in the spot diagram |
| `optimize-running-dark.png` | Optimize as an inspector panel while running: iteration, merit, improvement, log merit sparkline; the glass action bar [progress · Cancel · Stop]; the status capsule shows the running job |
| `optimize-result-light.png` | the result: before → after for every variable; the action bar morphed to Discard and the prominent Apply to Design |
| `tolerance-dark.png` | Tolerance panel: ranked sensitivities with bars, Monte-Carlo histogram and statistics |
| `solves-light.png` | Apply Solves panel: the dry run (now → solved, residual) before the commit |
| `issues-dark.png` | an unknown glass: the issues popover from the status capsule, the badge in the table, panes and viewport explaining why they cannot run |
| `wave-optics-light.png`, `wave-optics-dark.png` | wavefront map with the Zernike table, PSF, MTF; system inspector |
| `geometric-light.png` | Seidel bars, field curvature and distortion, lateral colour in narrow panes (titles truncate, controls stay) |
