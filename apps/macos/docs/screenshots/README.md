# Screenshots

The Liquid Glass design (docs/APP.md, "Design brief") on macOS 27, taken from
a debug build on the real engine (`scripts/build-macos.sh --debug`) with
`screencapture -l`, the engine's sample designs saved as projects whose
`views`, `merit` and `variables` sections set up the panes, and the debug
launch options (`-MKLightAppearance light|dark`, `-MKLightSheet`,
`-MKLightAutoRun`, `-MKLightAutoStop`, `-MKLightAutoApply`,
`-MKLightAutoPick`, `-MKLightFootprints`, `-MKLightFit lens`,
`-MKLightRunDetails`, `-MKLightToleranceTrials`, `-MKLightTolerancePreview`,
docs/APP.md "Building"); downscaled to 1600 px. The `tolerance-panel-*`
shots are the whole panel content, taller than the window, rendered from
the views on the real engine by the opt-in `RenderSnapshotTests`
(`MKLIGHT_RENDER_DIR=… swift test --filter RenderSnapshotTests/tolerancePanel`,
340 pt wide at scale 2). The tolerancing round (system-aware analysis
defaults and the tolerance UI) took the shots marked *round 6*. The
review round (narrow panes, bounds, sections, scales, image simulation,
warnings) retook the shots marked *new* or *updated*; the two merit shots
are from the `merit_evaluation` round (`-MKLightAutoConvert` pressed
Convert to Custom); the other job-panel shots (Optimize running and result,
Glass Substitution) and the welcome window are from the round before. Appearance in the file name. Reduce
Transparency and Increase Contrast were not captured (switching them changes
a system setting).

| file | shows |
|---|---|
| `main-light.png`, `main-dark.png` | *updated* Double Gauss: glass sidebar (analyses in title case), toolbar with the four design jobs, lens data, layout, spot diagram (statistics on two lines in the narrow pane), MTF (its warning badge) and the glass map with the design's glasses ringed and labelled with their surfaces; the dark one with the surface inspector |
| `merit-default-light.png` | *updated* Optimize on the Double Gauss: the default merit set to MTF (20 and 40 cycles/mm, target 0.6, weakest field) with the live merit from `merit_evaluation` (0.351515), Convert to Custom; variables with bounds in their units (radius windows in mm, one through ∞, thicknesses in mm) and a variable left on the back focus that a marginal-ray-height solve sets, flagged "Set by a solve: not varied"; the Seidel legend with S_I … S_V subscripts |
| `merit-custom-dark.png` | *updated* the same merit converted to custom: the 29 generated operands (rows lead with their field: F1, F2, F3 …), their values and shares, the `combine` minimum carrying the merit, and the same merit, 0.351515 |
| `optimize-running-light.png` | an LM run in progress: iteration, search merit, sparkline, the glass action bar with Cancel and Stop |
| `optimize-result-light.png` | the stopped run's result previewed in the window ("Previewing Optimized design (stopped) · not applied"), the before/after layouts, every variable before → after, Discard and the prominent Apply to Design |
| `glass-substitution-dark.png` | Glass Substitution on the achromat: 96 combinations ranked by merit with their improvement; candidate 2 previewed in the window and the panes (spot and glass map follow it), the before/after card and the panel's glass map with the candidate's glasses ringed |
| `glass-applied-light.png` | the best combination (SF5 · N-BAK1) applied as one undoable step; the table shows the new glasses and re-optimised radii |
| `anamorphic-xz-light.png` | *updated* the cylindrical adapter: both sections in the lens data (`y ∞  x −50`, the conic of the powered x section only, `n=1.5`); the X–Z section with the automatic vertical stretch ×20 ("Stretch ×20" in the controls, the scale bar's upright 2 mm bar and "vertical ×20"); the First-Order pane's x/y table (EFL x 49.76, EFL y 99.52, squeeze 2.0000×, meridional), the status capsule's EFL y and squeeze, and the System inspector's first order |
| `freeform-mesh-light.png` | *updated* XY freeform corrector in 3D: the engine's surface mesh, flat shaded, the selected freeform in the accent, framed on the lens; its inspector with the departure map |
| `footprints-dark.png` | *updated* Double Gauss in 3D with the beam footprint of every field on every surface (the rays step back) |
| `image-simulation-light.png` | *round 6* off-axis image simulation of the Siemens star on the Double Gauss: a 10 % patch centred at Field 2 ("At Field 2 (0.00, +0.71)" in the menu and the footer), 256 px (9.49 × 9.79 µm pixels), the engine's automatic PSF window ("PSF 505 × 0.39 µm") and pupil, the blur visible; beside it the relative illumination map with the cells outside the field hatched as no data, the map and its iso-lines reaching the field's edge |
| `mtf-auto-light.png` | *round 6* Double Gauss: MTF, MTF vs field and through-focus MTF with their settings pinned: "Auto (19.3)" maximum frequency, "Auto (2, 5, 10)" frequencies, "Auto (±0.206)" focus range, the pupil grid "Auto"; the curves fill their plots and the through-focus axis is the engine's range |
| `mtf-auto-dark.png` | *round 6* the same on the Cooke triplet (f50 F/4) in dark appearance: limit 19.4 cycles/mm, focus range ±0.437 mm |
| `spectrum-dark.png` | *updated* MTF with a photopic spectrum and the spot diagram with a 5500 K blackbody, from the pinned settings bars' spectrum buttons |
| `field-maps-light.png` | *updated* Cooke triplet: relative illumination map on the perceptual ramp over its own range (0.852 … 1, "Data range \| 0 to 1"), iso-lines every 0.02 with their labels apart, and the curve against image radius; chief ray angle against image height with the sensor limit and the shaded margin turning red where it is exceeded; the pupil aberration map (diverging, centred on 0) |
| `narrow-panes-light.png` | *new* four panes of about 240 pt: Spot Diagram (title kept, controls folded into More, statistics on two lines), Ray Fan (field above its tangential and sagittal plots, −1 0 1 on both axes), MTF and Field Curv. & Dist. (abbreviated title, one legend flow) |
| `narrow-maps-dark.png` | *new* the field maps at about 240 pt in dark appearance: RI Map, Chief Ray Angle (legend under its picker), Pupil Aberration, Rel. Illumination |
| `run-details-light.png` | *round 6* an MTF pane with an explicit 32 pupil grid: its run details group the engine's warning (Field 1, Field 2, Field 3) and offer "Sample the Pupil Automatically (grid 32 → auto)"; the PSF pane beside it with an explicit 64 grid and its pinned settings ("Spacing Auto (0.387–0.393 µm)") |
| `tolerance-editor-light.png` | *round 6* the Tolerance panel on the Double Gauss: the precision class's 60 tolerances grouped by element (L1 · N-SK16, Air after L1, L2 · N-SK16, F2 …), each with its kind, surface, ± limit and unit (fringes, mm, ′), Default Tolerances…, Add, the note on catalogue glasses, Run |
| `tolerance-results-dark.png` | *round 6* the Cooke triplet's result (35 tolerances, back-focus compensator, 500 trials): nominal 19.9 µm, RSS prediction 23.4 µm, 50 / 90 / 97.7 % of builds, the sensitivities by share of the RSS with the two worst offenders tinted, and the worst build (Decentre y S3 at +0.025 mm, compensated) previewed in the layout and the analyses, "not applied" |
| `tolerance-panel-results-light.png` | *round 6* the Double Gauss's whole result: summary with the yield (86.8 % at ≤ 20.5 µm), the twelve largest sensitivities, the Monte-Carlo histogram with the cumulative share, the nominal and the three levels, and the back focus: re-optimised nominal, the range the sensitivity used against ±0.5 mm, its Monte-Carlo mean and σ |
| `tolerance-panel-editor-dark.png` | *round 6* the Cooke triplet's whole editor in dark appearance: the table, the back-focus compensator, the criterion (RMS spot, rings, arms, fields, wavelengths, pass limit) and the analysis (both, 200 trials, seed) |
| `tolerance-panel-mtf-light.png` | *round 6* a Monte-Carlo-only run on the Cooke judged by the MTF at 20 cycles/mm (40 trials): the MTF levels of the builds, the histogram with the cumulative share falling to the right (higher is better), the compensator's spread |
| `diffractive-light.png` | *updated* hybrid refractive-diffractive singlet: the DOE marked by a violet hatch in the layout (framed on the lens), the First-Order pane's DOE power, the diffractive inspector |
| `welcome-light.png`, `welcome-dark.png` | welcome window: sample cards with the engine's layouts for every surface type, slim designs framed on their lens |
