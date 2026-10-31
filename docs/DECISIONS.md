# Design Decisions and Open Questions

This file records choices the original brief left open: what was decided,
why, and which questions are still open. The **Status** column shows what
exists in the code today.

## Decided

| topic | decision | status |
|---|---|---|
| Units and signs | mm, µm, degrees in files. Curvature is stored, not radius. Indices are always positive, and the direction vector carries the sign. Indices are relative to air. All of this is in `CONVENTIONS.md`. | in force |
| FP64 on Apple GPUs | Apple GPUs have **no double precision**. Metal kernels run in f32, using local frames, cancellation-free intersections, double-float OPL accumulation and compensated reductions. There is an explicit error budget, and the CPU f64 tracer is the reference. A backend that cannot meet the budget for a design must fall back to the CPU and say so. | in force |
| GPU-first computation (2026-10-31) | GPU-efficient work runs on qualified Metal, including generation/aiming, reductions, wave optics and candidate batches. CPU f64 is the accuracy reference and reported fallback; no CPU speed work is planned. `auto` prefers Metal, with CPU cost exceptions supported by end-to-end measurements as residency removes overhead. Keep `ComputeBackend` and portable kernel specifications; Vulkan is deferred. See GPU §9 for precision gates and work packages. | accepted; staged implementation planned |
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
4. **Interchange formats.** Should MKLight import or export other programs'
   lens files, ISO 10110 drawings, or STEP/IGES mechanical geometry? Each one
   needs a format decision. None is implemented yet.
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
