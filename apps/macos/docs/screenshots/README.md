# Screenshots

The Liquid Glass design (docs/APP.md, "Design brief") on macOS 27, taken from
a debug build on the real engine (`scripts/build-macos.sh --debug`) with
`screencapture -l`, the engine's sample designs saved as projects whose
`views`, `merit` and `variables` sections set up the panes, and the debug
launch options (`-MKLightAppearance light|dark`, `-MKLightSheet`,
`-MKLightAutoRun`, `-MKLightAutoStop`, `-MKLightAutoApply`,
`-MKLightAutoPick`, `-MKLightFootprints`, `-MKLightFit lens`,
`-MKLightRunDetails`, docs/APP.md "Building"); downscaled to 1600 px. The
review round (narrow panes, bounds, sections, scales, image simulation,
warnings) retook the shots marked *new* or *updated*; the job-panel shots
(Optimize running and result, Glass Substitution, the custom merit) and the
welcome window are from the round before. Appearance in the file name. Reduce
Transparency and Increase Contrast were not captured (switching them changes
a system setting).

| file | shows |
|---|---|
| `main-light.png`, `main-dark.png` | *updated* Double Gauss: glass sidebar (analyses in title case), toolbar with the four design jobs, lens data, layout, spot diagram (statistics on two lines in the narrow pane), MTF (its warning badge) and the glass map with the design's glasses ringed and labelled with their surfaces; the dark one with the surface inspector |
| `merit-default-light.png` | *updated* Optimize: the default merit generator set to MTF (20 and 40 cycles/mm, target 0.6, weakest field), the live merit, Convert to Custom; variables with bounds in their units: radii as radius windows in mm (23.81 … 33.33; 16.67 … −100 through ∞; −25 … −16.67), the A₄ coefficient in mm⁻³, thicknesses in mm, the Radius mm / Curvature 1/mm switch; bounded marks in the table |
| `merit-custom-dark.png` | the custom merit: operands with parameters, comparison, target and weight, their current values and shares of the merit as bars, constraints badged |
| `optimize-running-light.png` | an LM run in progress: iteration, search merit, sparkline, the glass action bar with Cancel and Stop |
| `optimize-result-light.png` | the stopped run's result previewed in the window ("Previewing Optimized design (stopped) · not applied"), the before/after layouts, every variable before → after, Discard and the prominent Apply to Design |
| `glass-substitution-dark.png` | Glass Substitution on the achromat: 96 combinations ranked by merit with their improvement; candidate 2 previewed in the window and the panes (spot and glass map follow it), the before/after card and the panel's glass map with the candidate's glasses ringed |
| `glass-applied-light.png` | the best combination (SF5 · N-BAK1) applied as one undoable step; the table shows the new glasses and re-optimised radii |
| `anamorphic-xz-light.png` | *updated* the cylindrical adapter: both sections in the lens data (`y ∞  x −50`, the conic of the powered x section only, `n=1.5`); the X–Z section with the automatic vertical stretch ×20 ("Stretch ×20" in the controls, the scale bar's upright 2 mm bar and "vertical ×20"); the First-Order pane's x/y table (EFL x 49.76, EFL y 99.52, squeeze 2.0000×, meridional), the status capsule's EFL y and squeeze, and the System inspector's first order |
| `freeform-mesh-light.png` | *updated* XY freeform corrector in 3D: the engine's surface mesh, flat shaded, the selected freeform in the accent, framed on the lens; its inspector with the departure map |
| `footprints-dark.png` | *updated* Double Gauss in 3D with the beam footprint of every field on every surface (the rays step back) |
| `image-simulation-light.png` | *updated* image simulation of the Siemens star with the new starting settings: the centre 10 % of the field at 256 px (9.49 µm pixels), the blur of the Double Gauss visible; the antialiased (area-sampled) source beside it, the View and resolution menus, exposure and zoom, no warnings |
| `spectrum-dark.png` | *updated* MTF with a photopic spectrum and the spot diagram with a 5500 K blackbody, from the pinned settings bars' spectrum buttons |
| `field-maps-light.png` | *updated* Cooke triplet: relative illumination map on the perceptual ramp over its own range (0.852 … 1, "Data range \| 0 to 1"), iso-lines every 0.02 with their labels apart, and the curve against image radius; chief ray angle against image height with the sensor limit and the shaded margin turning red where it is exceeded; the pupil aberration map (diverging, centred on 0) |
| `narrow-panes-light.png` | *new* four panes of about 240 pt: Spot Diagram (title kept, controls folded into More, statistics on two lines), Ray Fan (field above its tangential and sagittal plots, −1 0 1 on both axes), MTF and Field Curv. & Dist. (abbreviated title, one legend flow) |
| `narrow-maps-dark.png` | *new* the field maps at about 240 pt in dark appearance: RI Map, Chief Ray Angle (legend under its picker), Pupil Aberration, Rel. Illumination |
| `run-details-light.png` | *new* the MTF's run details: the engine's warnings grouped (one message, Field 1, Field 3) with the one-click "Sample the Pupil Finer (grid 64 → 128)" and the raw messages in a disclosure |
| `diffractive-light.png` | *updated* hybrid refractive-diffractive singlet: the DOE marked by a violet hatch in the layout (framed on the lens), the First-Order pane's DOE power, the diffractive inspector |
| `welcome-light.png`, `welcome-dark.png` | welcome window: sample cards with the engine's layouts for every surface type, slim designs framed on their lens |
