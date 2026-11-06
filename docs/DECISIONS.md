# Design Decisions and Open Questions

This file records choices the original brief left open: what was decided,
why, and which questions are still open. The **Status** column shows what
exists in the code today.

## Decided

| topic | decision | status |
|---|---|---|
| Units and signs | mm, µm, degrees in files. Curvature is stored, not radius. Indices are always positive, and the direction vector carries the sign. Indices are relative to air. All of this is in `CONVENTIONS.md`. | in force |
| FP64 on Apple GPUs | Apple GPUs have **no double precision**. Metal kernels run in f32 with double-float (f32 hi + lo pairs) wherever the budget needs more: transfers, hit refinement, OPL/OPD, aiming residuals and updates. They use local frames, cancellation-free intersections and compensated reductions. There is an explicit error budget (CONVENTIONS §7, per-quantity contracts in GPU §9.4), and the CPU f64 tracer is the reference. Accuracy is qualified offline; analyses that fail it stay on the CPU, and a native run that fails falls back and says so (GPU §9.2). | in force |
| Mixed precision (2026-11-04) | Double-float stays the only production Metal arithmetic; the pure-f32 build is a measurement tool. Reason: measured, an f32 predictor phase in the aim solve cannot pay (its stop-residual floor lies above the aim stop, so double-float iterations follow anyway, and the aim kernel is control-flow bound), and chief-relative f32 offsets are too large for sag differences and OPL (GPU §3.6). | in force |
| Compute placement (2026-11-04; replaces "GPU-first computation", 2026-10-31) | **GPU for throughput, CPU for latency.** Metal serves large batched work (dense ray grids, wave pupils and transforms, large candidate batches) when it is faster end to end; the CPU f64 path serves small, interactive and solver work and is the accuracy reference. CPU speed work is in scope: f64 SoA/SIMD packet tracing and batched aiming are shipped and bitwise equal to the scalar tracer, which stays the independent oracle (CONVENTIONS §7). `auto` routes every request by a calibrated stage-level cost model with accuracy exclusions (GPU §9.2), not by a static GPU preference or ray-count thresholds. Reason: end-to-end measurements showed the CPU faster for most small and edited jobs and for the solvers, and a static GPU preference was several times slower than the faster backend on such jobs. Keep `ComputeBackend` and portable kernel specifications; Vulkan is deferred. | in force |
| Aim stop and tiers (2026-11-04) | The engine's default aim stop is `conditioned`: the stop residual is derived from the propagated error and never looser than its allocation of the §7 budgets; the solver library default stays `requested`, and an explicit aiming tolerance is always honoured (CONVENTIONS §7). A cheaper **preview** tier exists, opt-in and labelled in every result (GPU §9.4). Reason: the conditioned stop passed the per-quantity contracts with identical survival masks on CPU and Metal while saving Newton iterations; design budgets, requested tolerances and survival rules are not relaxed. | in force |
| Fast math | Metal shaders compile **without fast math**, because fast math breaks the error-free transforms behind double-float arithmetic. A test checks this on the device. | in force |
| Random numbers | Philox4x32-10 counter RNG, identical on CPU and GPU, keyed by `(seed, stream, index)`. It is pinned by known-answer tests. | in force |
| State model | Immutable, content-hashed `Snapshot`s. Every result carries `RunInfo` (engine version, snapshot hash, settings, backend, precision, seed). The UI keeps a separate view state. | in force |
| Glass data | A small built-in catalog of published Sellmeier coefficients, each checked against its catalog nd/vd. Vendor catalogs are **not shipped**: users load AGF-format files, which avoids redistribution licensing problems. | in force |
| Engine ↔ app boundary | A C ABI. Structured data crosses as versioned JSON and bulk arrays as binary buffers. The UI never computes optics: layout geometry, sag profiles and ray paths all come from the engine. | planned |
| Multi-configuration | Zoom and focus positions are `Configuration` parameter overrides on one base system. | model in place |
| Coatings | A thin-film TMM that obeys `R + T + A = 1` per polarisation. Primary transmission, ghosts and stray light all use the same coating code. | in force |

## Points the brief did not mention (worth deciding)

1. **Polarisation ray tracing.** Coatings are computed per s/p polarisation,
   but rays carry incoherent intensity. Full Jones/PRT tracing (retardance,
   diattenuation, polarisation-dependent PSF) is needed for high-NA or
   heavily coated systems.
2. **Tolerancing.** "Manufacturing robustness" implies sensitivity analysis,
   Monte-Carlo tolerancing with compensators (back focus, air spaces), and an
   ISO 10110 tolerance set. The model has `ParamPath` for this, but the
   operand set and the UI still need to be specified.
3. **Solves and pickups.** Marginal-ray-height and angle solves (for example
   the image distance), F/# solves, and pickups between parameters. Lens
   designers expect these.
4. **Interchange formats.** *Partly decided (2026-11-07):* MKLight
   **imports** `.zmx` and `.seq` lens prescription files (`IMPORT.md`):
   surfaces and their common types, stops, apertures, glasses resolved
   against the loaded catalogs (unknown ones become model glasses from the
   file's `n_d`/`V_d`, or stay references by name), fields, wavelengths,
   aperture and object distance. Anything that does not map (solves,
   pickups, macros, multi-configuration, non-sequential data) is dropped
   with a structured warning instead of failing the import; the app shows
   the warnings in a sheet. Available in the engine, C ABI 1.3, CLI
   (`mklight import`), Python and the app. Still open: exporting either
   format, other prescription formats, ISO 10110 drawings, and STEP/IGES
   mechanical geometry.
5. **Thermal and environmental analysis.** Glass dn/dT is modelled. Thermal
   expansion of spacers and mounts, and athermalisation, are not.
6. **Freeforms and more surface types.** Q-type (Forbes) aspheres, XY and
   Zernike polynomials, gratings and GRIN are common in modern camera lenses.
   `SurfaceShape` is extensible, but every new type must be implemented in the
   CPU reference **and** in every GPU backend.
7. **Detector and image simulation.** Pixel-integrated MTF, relative
   illumination, colour image simulation with a sensor spectral response, and
   chief-ray-angle checks against microlens limits.
8. **Scripting and macros.** Python bindings cover headless use. Is an
   in-app console or macro language also needed?
9. **Document features.** Undo and redo are in the app. Still open: autosave,
   versioned project-file migration (`schema_version`), and comparing two
   designs.
10. **Distribution.** Code signing, notarisation, the App Sandbox (security-scoped
    access to user glass catalogs), the minimum macOS version (currently 15),
    and the licence model.
11. **CI and validation data.** GitHub-hosted macOS runners have only
    paravirtualised GPUs, so Metal cross-validation needs a self-hosted Apple
    Silicon runner. Golden reference results for the sample designs should be
    frozen and checked on every change.
12. **Performance targets.** Concrete targets per analysis would guide
    optimisation work: for example spot diagrams in < 16 ms for interactive
    editing, or optimiser evaluations per second.
13. **Units other than mm** in the UI (inches, µm), and display precision
    settings.
14. **Accessibility and localisation** of the app.
