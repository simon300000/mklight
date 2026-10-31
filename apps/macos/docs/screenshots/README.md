# Screenshots

The Liquid Glass design (docs/APP.md, "Design brief") on macOS 27, taken from
a debug build on the real engine (`scripts/build-macos.sh --debug`) with
`screencapture -l`, the engine's sample designs saved as projects whose
`views`, `merit` and `variables` sections set up the panes, and the debug
launch options (`-MKLightAppearance light|dark`, `-MKLightSheet`,
`-MKLightAutoRun`, `-MKLightAutoStop`, `-MKLightAutoApply`,
`-MKLightAutoPick`, `-MKLightFootprints`, `-MKLightFit lens`, docs/APP.md
"Building"); downscaled to 1600 px. Appearance in the file name. Reduce
Transparency and Increase Contrast were not captured (switching them changes
a system setting).

| file | shows |
|---|---|
| `main-light.png`, `main-dark.png` | Double Gauss: glass sidebar (analyses in title case), toolbar with the four design jobs, lens data, layout, spot diagram, MTF and the glass map with the design's glasses ringed and labelled with their surfaces; the dark one with the surface inspector |
| `merit-default-light.png` | Optimize: the default merit generator set to MTF (20 and 40 cycles/mm, target 0.6, weakest field), the live merit, Convert to Custom, the EFL hold; variables with min/max bounds (curvature bounds, an asphere coefficient, the back focus ≥ 0) and their bounded marks in the table |
| `merit-custom-dark.png` | the custom merit: operands with parameters, comparison, target and weight, their current values and shares of the merit as bars, constraints badged |
| `optimize-running-light.png` | an LM run in progress: iteration, search merit, sparkline, the glass action bar with Cancel and Stop |
| `optimize-result-light.png` | the stopped run's result previewed in the window ("Previewing Optimized design (stopped) · not applied"), the before/after layouts, every variable before → after, Discard and the prominent Apply to Design |
| `glass-substitution-dark.png` | Glass Substitution on the achromat: 96 combinations ranked by merit with their improvement; candidate 2 previewed in the window and the panes (spot and glass map follow it), the before/after card and the panel's glass map with the candidate's glasses ringed |
| `glass-applied-light.png` | the best combination (SF5 · N-BAK1) applied as one undoable step; the table shows the new glasses and re-optimised radii |
| `anamorphic-xz-light.png` | the cylindrical adapter in the X–Z section, the First-Order pane's x/y table (EFL x 49.76, EFL y 99.52, squeeze 2.0000×, meridional), the status capsule's EFL y and squeeze, and the System inspector's first order |
| `freeform-mesh-light.png` | XY freeform corrector in 3D: the engine's surface mesh, flat shaded, the selected freeform in the accent, framed on the lens; its inspector with the departure map |
| `footprints-dark.png` | Double Gauss in 3D with the beam footprint of every field on every surface (the rays step back) |
| `image-simulation-light.png` | image simulation of the Siemens star, source and simulated image side by side, exposure and zoom, pixel size, distortion and relative illumination in the footer |
| `spectrum-dark.png` | MTF with a photopic spectrum and the spot diagram with a 5500 K blackbody, from the pinned settings bars' spectrum buttons |
| `field-maps-light.png` | Cooke triplet: relative illumination map with iso-lines and the curve against image radius, chief ray angle against image height with the sensor limit and the shaded margin turning red where it is exceeded, and the pupil aberration map |
| `diffractive-light.png` | hybrid refractive-diffractive singlet: the DOE marked by a violet hatch in the layout (framed on the lens), the First-Order pane's DOE power, the diffractive inspector |
| `welcome-light.png`, `welcome-dark.png` | welcome window: sample cards with the engine's layouts for every surface type, slim designs framed on their lens |
