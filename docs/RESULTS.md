# MKLight Analysis Results

Every analysis the engine runs (`mk_engine_list_analyses`,
`Engine::analyses()`, `mklight analyses`, `mklight.Engine.analyses()`)
publishes two JSON Schemas: its **settings** (with defaults) and its
**result** (with the typed buffers that travel next to the JSON). Front
ends decode results from them; this document explains the shared rules and
lists every analysis. Section 8 is generated from the registry's schemas and
checked by `crates/mklight-engine/tests/results_doc.rs`, so it always
matches the code. Units and signs follow `docs/CONVENTIONS.md`.

## 1. The envelope

`mk_job_take_result`, `mk_session_run_analysis_json`, `mklight run` and the
Python `Result.envelope()` return:

```json
{ "run_info": { "engine_version": "0.1.0", "snapshot_hash": "<64 hex>",
                "analysis": "spot_diagram", "settings": { …complete settings… },
                "backend": "gpu:metal", "precision": "f32_gpu", "seed": null,
                "started_unix_ms": 1761840000000, "duration_ms": 4.2,
                "warnings": [] },
  "library_hash": "<64 hex>",
  "backend": { "requested": "auto", "used": "gpu:metal", "precision": "f32_gpu",
               "device": "Apple M2 Pro", "fallback_reason": null,
               "qualification": { "passed": true, "budget_fraction": 0.05, … } },
  "result_version": 1,
  "result": { … the analysis' result (section 8) … },
  "buffers": [ {"index": 0, "name": "points", "dtype": "f64", "shape": [1143, 2]} ] }
```

* `run_info.settings` holds every effective setting (defaults filled in);
  resubmitting it reproduces the run. `run_info.seed` is set for stochastic
  runs (random or dithered pupil patterns, random pupil samples, stray
  light).
* `run_info.backend` is the backend that **computed** the result
  (`cpu_reference`, `gpu:metal`, …), not the one requested; see section 5.
* A result is current while `run_info.snapshot_hash` and `library_hash`
  equal the session's (`docs/FFI.md`).
* The solver crates' own `RunInfo` is not repeated inside `result`; its
  warnings are merged into `run_info.warnings`.

## 2. Versioning and settings

* `result_version` (also in the analysis description) is per analysis. It
  is bumped when a field changes meaning, unit or shape, or disappears.
  Adding a field or a buffer does **not** bump it: readers must ignore
  unknown keys and buffers. The envelope itself grows only with the ABI
  minor version.
* Settings may be partial; missing keys take the schema defaults. Unknown
  top-level keys are rejected (`MK_ERR_INVALID_ARGUMENT`); values are
  checked against the schema's `type`, `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum` and `enum`
  before the job is queued. Nested objects (field selections, source specs,
  …) are validated when the analysis parses them.
* Settings schemas of analyses implemented in a solver crate are derived
  from that crate's settings `Default`, so the engine's defaults are the
  crate's. Where the engine deliberately differs it says so in the setting's
  description and below: `ghost_irradiance.pupil_samples` defaults to 48
  (crate: 128) and `stray_light.rays` to 20 000 (crate: 100 000), so that the
  default runs stay interactive.

## 3. Numbers, units, frames

* Lengths mm, wavelengths µm, angles degrees (radians where a field says
  `rad`), OPD and wavefront in waves at the ray's own wavelength, frequencies
  cycles/mm, flux in W (stray light) or relative to the entrance-pupil input
  (ghosts), irradiance per mm².
* **Type labels** in section 8: `nf` is a number that may be non-finite or
  undefined (CONVENTIONS §11): a JSON number, `null` (NaN: undefined, e.g. a
  vignetted field or a failed ray), or one of the strings `"inf"`, `"-inf"`,
  `"nan"`. `T?` is `T` or `null`. `[T; n]` is a fixed-length array (e.g.
  `[nf; 2]` = `[x, y]`). `array<T>` has any length. `object` without listed
  fields is a model type documented in `mklight-core` / the solver crate.
* **Image positions** (spots, fans, ray grids, footprints) are in the local
  frame of the surface concerned (the image surface unless stated): for an
  untilted image this is the global frame shifted to the image vertex.
  `layout` is in global coordinates.
* `field_value` is the field as specified (`[x, y]`, degrees for angles, mm
  for heights); `normalized_field` divides by the maximum radial field.

## 4. Buffers

Bulk numeric data travels as typed binary buffers, not JSON: ray clouds
(spot points, ray grids, footprints), grids (grid distortion), maps (ghost
and stray irradiance), per-sample pupil data (polarisation) and scene
geometry (layout). Curves sampled along one axis (fans, field curves,
energy and MTF curves, through-focus) stay in the JSON.

* `dtype` is one of `f64, f32, i32, u32, i64, u64, u8` (C codes
  `MK_DTYPE_*`); data are row-major with the declared axes (section 8
  names them: `[F, W, P, 2]` = field, wavelength, pupil sample, x/y).
* Floating-point buffers use **NaN** for "no value" (a ray that did not
  arrive).
* Variable-length groups share one buffer and are addressed from the JSON by
  `[offset, count]` pairs (`spot_diagram.fields[].point_range`,
  `footprint.sets[].points`) or by an offsets buffer (`layout.offsets`).
* Each analysis' result schema lists its buffers under `x-buffers`
  (`name`, `dtype`, `shape` axis names, `unit`, `description`); a buffer
  name ending in `*` would stand for a family. Optional buffers may be
  absent (`ghost_irradiance.path_maps` without `keep_path_maps`).

## 5. Backends and GPU paths

Requests name a backend (`{"kind": "cpu_reference"}`,
`{"kind": "gpu", "name": "metal"}`) or `{"kind": "auto"}`; the engine
default (initially the CPU reference, `mk_engine_set_default_backend`)
applies when they name none (`backend.requested` = `default:<id>`).

* `auto` uses the first available GPU backend whose capabilities list the
  analysis (those with `gpu_path: true`), else the CPU reference;
  `backend.fallback_reason` then says why (no GPU path, no device). This is
  not a warning. `auto` also keeps ray grids smaller than
  `AUTO_MIN_GPU_RAYS` (65 536 rays: fields × wavelengths × pupil points) on
  the CPU reference, where the GPU's fixed costs outweigh the trace it
  saves (reason "below the GPU threshold", no warning). An explicitly named
  GPU backend is used for any grid size.
* An explicit GPU backend that is unavailable or has no path for the
  analysis is an error, or with `allow_backend_fallback` the CPU reference
  plus a `run_info` warning.
* Before a GPU computes anything for a snapshot, its precision is
  **qualified** for that system (docs/GPU.md §5.2: a small cross-validation
  whose errors must stay within a third of the CONVENTIONS §7 budget;
  cached per snapshot and glass library). A failed qualification, a system
  the packer cannot represent, or a device error makes the CPU f64
  reference compute the result: `run_info.backend` is then
  `cpu_reference`, `backend.fallback_reason` holds the reason, a
  `run_info` warning names the rejected backend (CONVENTIONS §12), and
  `backend.qualification` holds the measured evidence. Cancelling is never
  a fallback.
* GPU paths today (`gpu_path: true`): `spot_diagram`, `encircled_energy`,
  `geometric_mtf`, `best_focus`, `ray_grid`. The host generates and aims the
  rays with the f64 reference, the GPU traces them (f32 with double-float
  geometry), and the reduction is the same code as on the CPU, so results
  agree within the CONVENTIONS §7 budget (tested on every sample system, on
  the Metal device when present and on the CPU-f32 emulation everywhere:
  hit positions and directions, spot statistics, MTF, encircled energy and
  best focus). GPU hit points are f32 values widened to f64 in the buffers.
* **Speed.** Only the trace runs on the device; the host generates (and,
  with real ray aiming, aims) every ray with the f64 reference, in parallel
  blocks of 65 536 rays with cancellation and progress between blocks, and
  the reductions are the CPU's. Measured on Apple-silicon Metal with
  double_gauss (best of 9, warm qualification cache): `ray_grid` at
  98 829 rays 7.4 ms against 10.9 ms on the CPU, at 1 085 409 rays 67 ms
  against 105 ms; at 11 349 rays the CPU is faster (1.3 ms against 2.7 ms),
  hence the `auto` threshold. `spot_diagram` gains less (345 ms against
  375 ms at 1 085 409 rays): its time goes mostly to the shared reduction
  and result assembly. Moving ray aiming and the reductions to the device
  is future work.
* `MKLIGHT_DISABLE_GPU=1` makes every GPU backend unavailable.
* `auto` has the same meaning in Rust, FFI, CLI and Python. Python `None`
  (or an omitted backend) uses the engine default.
* `optimize` and `tolerance` also use the selected provider and its cached
  qualification. Automatic merit evaluation uses the candidate-batch
  crossover in `docs/OPTIMIZATION.md`; an explicitly requested GPU bypasses
  that threshold. The envelope and `RunInfo` name the search backend; the
  optimiser's start/final operand tables are recomputed on the CPU f64
  reference. Per-candidate CPU fallbacks are counted in `result.backend.gpu_stats`.

## 6. Cross-solver conventions

* **Ghost and stray partition** (CONVENTIONS §10, docs/STRAY.md). Run
  `ghost_paths` / `ghost_irradiance` and `stray_light` with
  `ghost_partition` set to the same ghost-model options (`sensor_reflectance`,
  `clip`, `reject_virtual`, `enumeration`, `paths`). The stray run then
  clips the optical surfaces at the ghost run's apertures (checked with
  `GhostCoverage::same_apertures`; a mismatch is an error) and leaves the
  ghost paths **in the ghost run's coverage** out of its total, so
  `detected = Primary (sequential) + Ghost (ghost run) + stray_flux`
  path by path. Ghost-class paths the ghost run does not cover (a subset of
  explicit `paths`, orders above `max_reflections`, paths pruned by
  `min_reflectance_product` or `max_higher_order_paths`, irregular
  reflection sequences) belong to the stray solver (CONVENTIONS §10): they
  are added to `stray_flux` as `partition.uncovered_ghost_flux`, with a
  `run_info` warning. The stray tally splits by class, not by signature, so
  the maps hold only `partition.mapped_stray_flux` (scatter, mechanical,
  sneak; exact standard error), not the handed-back ghosts; and the
  handed-back flux's standard error is the bound `Σ SEₖ` over its
  signatures, so `stray_std_error` is then an upper bound.
  `result.partition.apertures` equals the ghost result's
  `coverage.apertures`; `partition.ghost_class_flux_mc` (the whole class)
  and `partition.covered_ghost_flux_mc` (the covered part, to compare with
  the ghost solver) are Monte-Carlo estimates. The partition
  needs `clip: {"mode": "clear_apertures", …}` in the ghost options (the
  stray scene always clips at clear apertures).
* **Validation codes** (`validate`, `mk_session_validate_json`): structural
  problems are rejected before a snapshot/session exists, with the core codes
  of CONVENTIONS §13. The CLI's `validate` on such a file reports all core
  issues with their JSON paths (`structural_only: true`). For valid snapshots,
  material/coating resolution, material ranges, aperture definition, thickness
  signs and zero weights are reported by the analysis. Engine-only tracer
  checks are `paraxial` (setup error), `chief_ray` and `marginal_ray`
  (reachability warnings, including a chief field that could not be prepared).
* **Solves** change the model, so applying them is a session operation
  (`Session::apply_solves`, `mk_session_apply_solves`: a new snapshot plus
  the `solves` report); the `solves` analysis is its dry run.

## 7. Signs and meanings by analysis

* `spot_diagram`: buffer `points` holds `(x, y)` relative to each field's
  `reference` (the primary chief ray or the polychromatic centroid,
  `settings.reference`); `chief`, `centroid` and `reference` are absolute
  image-local positions; RMS values are two-pass, compensated, about the
  stated centre. Defocus moves along the image's local `+z`.
* `ray_fan`: `εy = y_ray − y_chief` on the tangential fan (`P_y`) and
  `εx = x_ray − x_chief` on the sagittal fan (`P_x`), mm; `cross` is the
  other component. `opd_fan`: `OPD = OPL_chief − OPL_ray` in waves against
  the exit-pupil reference sphere: positive at the pupil edge for a
  wavefront converging more than the reference (CONVENTIONS §4).
* `field_curvature_distortion`: focus shifts along the image's local `z`
  from the image surface (positive: behind it); distortion
  `100 (y_real − y_ref)/y_ref` against the paraxial chief height,
  `f·tanθ` and `f·θ`.
* `seidel`: positive singlet `S_I > 0`; `W040 > 0` has the OPD sign above;
  Petzval radius with the signed image-space index (docs/ANALYSIS.md §2.3).
* `relative_illumination`: `Ω_p(H)/Ω_p(0)` (1 on axis).
* `encircled_energy`: fractions of the energy that arrived; radii about the
  `reference`. `geometric_mtf`: 1 at zero frequency; tangential = along the
  image's `y`. `best_focus.defocus`: along the image's local `+z`.
* `layout`: global frame; `vertices` (f32) are relative to `origin` for
  rendering precision, `vertices_f64` absolute.
* `ghost_irradiance`: maps `[ny, nx]`, row 0 at `y = −half_size[1]`, `x`
  fastest; values per mm² relative to the entrance-pupil input flux;
  `flux_ratio`/`peak_ratio` against the primary image.
* `stray_light`: flux in W, irradiance in W/mm², detector local frame,
  maps `[ny, nx]` with row 0 at `y = −half_y`; every flux has a Monte-Carlo
  standard error.
* `system_transmission`: dimensionless throughput in [0, 1].
* `polarization_pupil_map`: Jones matrices in the double-pole bases
  (CONVENTIONS §9), `[row][col][re, im]`; retardance in radians.
* `solves`: `residual` in the condition's units (mm, slope, F/#, or
  dimensionless).

* `optimize`: merit is Σ weight × residual² (default spot merit in mm²,
  wavefront merit in waves²); custom operands can mix units through their
  weights. Equal residuals are value − target; inequalities keep only the
  violating side. Constraint violation is normalised and non-negative.
  `variables[].change` is value − start in the parameter's model units
  (curvature 1/mm, thickness/decenter mm, tilt degrees). `improvement` is
  1 − final/start merit, so positive means improvement. History and the
  `merit_history` buffer use the search evaluator; the final operand table
  uses the CPU f64 reference and chooses the better of start and answer,
  feasibility first. `stopped: true` is a retained early result from Stop;
  Cancel/supersession discards it. The app applies `result.system` with
  `mk_session_set_system_json` (`Session::set_system`) as an undoable document
  edit, creating a snapshot with `result.system_hash`. Check the input
  snapshot and library hashes before applying to a document edited meanwhile.
* `tolerance`: `minus`/`plus` are positive perturbation magnitudes in each
  parameter's model units; evaluate nominal − minus and nominal + plus.
  Criterion values have the criterion merit's units; `change_minus` and
  `change_plus` are trial − nominal (positive is degradation), `worst` is
  the largest absolute change and `rss_change` is their root sum of squares.
  Compensators report parameter values in model units. Yield is the fraction
  with criterion ≤ threshold; a failed Monte-Carlo trial has criterion `inf`.
  Stop retains completed sensitivity operands and Monte-Carlo trials with
  `stopped: true`. Tolerancing does not replace the session's model.

### Wave optics

`wavefront_map`, `psf`, `mtf`, `diffraction_encircled_energy`,
`line_spread`, `through_focus_mtf` and `mtf_vs_field` run the CPU f64 wave
solver on the selected, solved `configuration` (default 0). Their RunInfo
records the session snapshot hash, configuration, settings and solve/sampling
warnings. These analyses have no GPU path.

* **Wavefront.** OPD is `OPL_chief − OPL_ray` at the exit-pupil reference
  sphere, in waves at the reported wavelength (µm); the chief ray has zero
  OPD. Pupil coordinates are cell centres over `[-1, 1]`, `u` along image
  x and `v` along image y. Maps are `[v, u]`, u fastest, with NaN outside
  the clear pupil. Zernike coefficients are in waves, Fringe order Z1
  onward by default; `standard` selects Noll order and RMS normalisation.
  Coefficients fit the lattice rays with amplitude-squared weights, rather
  than the resampled map. RMS removes piston; tilt-removed statistics also
  remove best-fit tilt.
* **PSF.** The default `strehl` normalisation sets the aberration-free peak
  of the same pupil to 1; the peak intensity is the Strehl ratio.
  `unit_energy` instead gives intensity in mm⁻² with whole-plane integral
  1. `total_energy` is the whole-plane integral, so a finite window can
  contain less. PSF buffers are `[image_y, image_x]`, x fastest. Spacing,
  grid centres, peak and centroid coordinates are mm relative to the
  common image-local `reference_mm`. Polychromatic PSFs use normalised
  system wavelength weights and include lateral colour.
* **Energy and line spread.** `diffraction_encircled_energy` gives fractions
  of total PSF energy inside circles of `radius_mm`, and squares of that
  half-width, plus an aberration-free encircled reference and r50/r80/r90
  in mm. An unreachable radius is undefined. `line_spread.x` integrates
  the unit-energy PSF over y, and `.y` over x, in mm⁻¹.
* **MTF.** Frequencies and cutoffs are image-space cycles/mm; optical MTF
  is normalised to 1 at zero frequency. Tangential is along the local
  field direction (image y on axis), sagittal perpendicular (image x on
  axis). `mtf_vs_field` also uses the sweep direction at its axial sample.
  Through-focus shifts are mm, positive along image-space propagation;
  the reference point shifts along the chief ray by δz/cosθ.
* **Detector option.** All three MTF analyses accept `detector`: pixel
  aperture sinc factors and Gaussian diffusion multiply the optical MTF.
  `mtf` retains separate optical, detector and system curves. Its
  `nyquist = [1/(2 pitch_x), 1/(2 pitch_y)]` and per-field directional
  Nyquist values are markers in cycles/mm, not clipping thresholds. The
  directional marker is `min(1/(2 pitch_x |e_x|), 1/(2 pitch_y |e_y|))`.
  Field/focus sweeps return the product when a detector is supplied.
* **Buffers.** Wave arrays, including sampled curve axes and Zernike
  coefficients, travel as f64 buffers. Their JSON fields contain the
  corresponding buffer names. Numeric suffixes identify the JSON group
  position (not the system field index), with a second suffix for nested
  curves or rim loops. Shapes and units are listed below. Unusable field
  samples in MTF-versus-field buffers are NaN; unusable per-field pupils
  are skipped with a warning.
* **Sampling warnings.** `pupil_undersampled` (phase steps over half a
  wave), `field_wraps` (image window wider than the alias-free field), and
  `psf_truncated` (less than 95 % energy in the window) are retained in
  RunInfo as the solver's descriptive warning text. Increase pupil
  sampling or adjust the image window as described in `docs/WAVE.md`.

## 8. Analysis reference

Generated from the registry (one section per analysis, by category):
settings with defaults, result fields (nested objects and arrays of objects
flattened as `a.b` and `a[].b`), and buffers.

<!-- BEGIN GENERATED: analysis reference (tests/results_doc.rs) -->
### `solves` — Solves (preview)

Category `model`, result version 1.

Evaluates the system's solves (marginal/chief ray height and angle, F/#, pickups, …) without changing the session: solved values, residuals and methods. `apply_solves` on the session stores them as a new snapshot.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `closed_form` | boolean | `true` |  | Use closed-form updates (false: bracketed root finding only, a cross-check). |
| `configuration` | integer\|null | `null` |  | ≥ 0; Solve this configuration (dry run only); null = the base system. |
| `max_iterations` | integer | `200` |  | 1 … 1000000; Upper bound on evaluations per solve and sweep. |
| `max_sweeps` | integer | `50` |  | 1 … 100000; Upper bound on sweeps over all solves. |
| `tolerance` | number | `1e-12` |  | 0.0 … 1.0; Relative tolerance on each condition and on the change of every solved parameter between sweeps, in (0, 1). |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `configuration` | integer? |  |  |
| `max_change` | number |  | Largest relative parameter change in the last sweep. |
| `order` | array<integer> |  | Evaluation order (indices into solves). |
| `outcomes` | array<object> |  | One per solve, in declaration order. |
| `outcomes[].evaluations` | integer |  | ≥ 0 |
| `outcomes[].index` | integer |  | ≥ 0; Index into OpticalSystem::solves. |
| `outcomes[].kind` | string |  | Solve kind. |
| `outcomes[].method` | string |  | one of `"unchanged"`, `"closed_form"`, `"fixed_point"`, `"root_finding"` |
| `outcomes[].previous` | number? |  |  |
| `outcomes[].residual` | number |  | measured − goal of the condition in the solved system (mm, slope, F/# or c·D − 1). |
| `outcomes[].target` | object |  | The ParamPath the solve sets. |
| `outcomes[].value` | number |  | Solved value (units of the target parameter). |
| `solved_params` | array<object> |  | Parameters controlled by solves (exclude them from optimisation). |
| `sweeps` | integer |  | ≥ 0 |
| `warnings` | array<string> |  |  |

### `system_summary` — System summary

Category `model`, result version 1.

Surface table with resolved refractive indices at every system wavelength and the global frame (vertex position and orientation) of every surface. Buffers: `n_after` [surfaces × wavelengths], `vertex_global` [surfaces × 3] (mm).

Settings: none.

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `aperture` | object |  | The model's ApertureSpec. |
| `environment` | object |  | Temperature (°C) and pressure (atm). |
| `fields` | object |  | The model's FieldSet. |
| `first_to_image_distance` | number | mm | Distance from surface 1 to the image vertex. |
| `n_object` | array<number> |  | Object-space index per wavelength. |
| `name` | string |  |  |
| `num_configurations` | integer |  | ≥ 0 |
| `num_surfaces` | integer |  | ≥ 0 |
| `object_at_infinity` | boolean |  |  |
| `primary_wavelength` | integer |  | ≥ 0 |
| `ray_aiming` | object |  |  |
| `stop` | integer |  | ≥ 0 |
| `surfaces` | array<object> |  |  |
| `surfaces[].alpha_after_per_mm` | array<number> |  | Absorption coefficient per system wavelength. |
| `surfaces[].aperture` | any |  | Resolved aperture shape or null. |
| `surfaces[].coating` | string? |  |  |
| `surfaces[].conic` | number |  |  |
| `surfaces[].curvature` | number | 1/mm |  |
| `surfaces[].frame` | object? |  |  |
| `surfaces[].frame.origin` | [number; 3] |  | Global vertex. |
| `surfaces[].frame.rot` | any |  | 3×3 rotation, rows; columns are the local axes in global coordinates. |
| `surfaces[].id` | integer |  | ≥ 0; Stable surface id. |
| `surfaces[].index` | integer |  | ≥ 0; Surface index. |
| `surfaces[].interaction` | string |  | refract or reflect. |
| `surfaces[].is_stop` | boolean |  |  |
| `surfaces[].label` | string |  |  |
| `surfaces[].material` | string? |  |  |
| `surfaces[].material_ref` | any |  | The model's MaterialRef. |
| `surfaces[].n_after` | array<number> |  | Index after the surface at every system wavelength (relative to air). |
| `surfaces[].nd_after` | nf |  |  |
| `surfaces[].radius` | nf | mm | 1/curvature; "inf" for a plane. |
| `surfaces[].semi_diameter` | any |  | The model's SemiDiameter. |
| `surfaces[].shape` | object |  | The model's SurfaceShape. |
| `surfaces[].thickness` | nf | mm | To the next vertex ("inf" for an object at infinity). |
| `surfaces[].to_next_is_axial` | boolean |  |  |
| `surfaces[].type` | string |  | one of `"plane"`, `"sphere"`, `"conic"`, `"even_asphere"`, `"paraxial"`, `"coordinate_break"` |
| `surfaces[].vd_after` | nf? |  |  |
| `warnings` | array<string> |  |  |
| `wavelength_weights` | array<number> |  |  |
| `wavelengths_um` | array<number> |  |  |

Buffers:

| name | dtype | shape | unit | notes |
|---|---|---|---|---|
| `n_after` | f64 | [surfaces, wavelengths] |  | Index after each surface. |
| `vertex_global` | f64 | [surfaces, 3] | mm | Global vertex (NaN for an object at infinity). |

### `validate` — Validate system

Category `model`, result version 1.

Checks that the system is complete and consistent: every number is finite, materials and coatings resolve, wavelengths lie inside glass data ranges, the aperture is defined and positive, weights are not negative, mechanical parts reference existing surfaces, thickness signs follow mirrors, configurations apply, the reference tracer can set the system up, and the chief and marginal rays reach the image.

Settings: none.

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `errors` | integer |  | ≥ 0 |
| `issues` | array<object> |  |  |
| `issues[].code` | string |  | Issue code (docs/RESULTS.md). |
| `issues[].message` | string |  |  |
| `issues[].severity` | string |  | one of `"error"`, `"warning"`, `"info"` |
| `issues[].surface` | integer? |  |  |
| `ok` | boolean |  | No error-severity issue. |
| `warnings` | integer |  | ≥ 0 |

### `glass_info` — Glass data

Category `materials`, result version 1.

Catalog data of one glass and its index relative to air at chosen wavelengths and environment. Buffer `dispersion_curve` [samples × 2] holds (λ µm, n) over the glass's valid range.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `catalog` | string\|null | `null` |  | Catalog to search; null searches all in load order. |
| `curve_samples` | integer | `0` |  | 0 … 100000 |
| `name` | string | **required** |  |  |
| `pressure_atm` | number\|null | `null` | atm | ≥ 0.0 |
| `temperature_c` | number\|null | `null` | °C | null uses the system environment. |
| `wavelengths_um` | array\|null | `null` | µm | null uses the system wavelengths. |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `catalog` | string |  | Catalog the glass was found in. |
| `catalog_nd` | number |  | Catalog n_d. |
| `catalog_vd` | number |  | Catalog V_d. |
| `comment` | any |  |  |
| `cte` | any |  | Thermal expansion (1/K) or null. |
| `density` | any |  | g/cm³ or null. |
| `dispersion` | object |  | Dispersion formula and coefficients. |
| `environment` | object |  | Temperature and pressure used. |
| `has_thermal_data` | boolean |  |  |
| `indices` | array<object> |  |  |
| `indices[].absorption_per_mm` | number | 1/mm |  |
| `indices[].in_range` | boolean |  |  |
| `indices[].n` | nf |  | Index relative to air at the environment. |
| `indices[].n_catalog` | nf |  | At catalog conditions. |
| `indices[].wavelength_um` | number | µm |  |
| `max_um` | number | µm |  |
| `min_um` | number | µm |  |
| `name` | string |  |  |
| `nd` | number |  | Computed at the d line (catalog conditions). |
| `ne` | number |  |  |
| `pgf` | nf |  | Partial dispersion P_g,F. |
| `relative_cost` | any |  |  |
| `vd` | nf |  |  |

Buffers:

| name | dtype | shape | unit | notes |
|---|---|---|---|---|
| `dispersion_curve` | f64 | [samples, 2] |  | (λ µm, n) over the glass's range. |

### `best_focus` — Best focus

Category `geometric`, result version 1, **GPU path**.

Defocus minimising the field-weighted mean RMS spot radius (Brent), with per-field RMS at best and current focus. GPU path.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `fields` | object | `{"type":"all"}` |  | Field selection: {"type": "all"}, {"type": "index", "index": i} or {"type": "list", "indices": [i, …]} (system field indices). |
| `max_evaluations` | integer | `200` |  | 3 … 100000; Merit evaluations allowed. |
| `pattern` | string | `"hexapolar"` |  | one of `"hexapolar"`, `"square"`, `"random"`, `"dithered"`; Pupil pattern: hexapolar (1 + 3r(r+1) rays), square (pitch 1/rings, clipped to the disk), random (1 + 3r(r+1) uniform points, Philox stream 0), dithered (one jittered point per square cell, stream 1). |
| `range` | array\|null | `null` | mm | Search range [min, max] of the defocus; null = automatic. |
| `reference` | string | `"centroid"` |  | one of `"chief"`, `"centroid"`; Reference point: the primary-wavelength chief ray or the polychromatic centroid. |
| `rings` | integer | `8` |  | 1 … 512; Density of the pupil pattern (rings, or grid half-count). |
| `seed` | integer | `1` |  | ≥ 0; Philox seed of the random and dithered patterns. |
| `through_focus_samples` | integer | `0` |  | 0 … 100000; Samples of the through-focus merit curve (0: none). |
| `tolerance` | number | `1e-8` | mm | ≥ 0.0; Absolute tolerance on the defocus. |
| `wavelengths` | object | `{"type":"all"}` |  | Wavelength selection: {"type": "all"}, {"type": "primary"}, {"type": "index", "index": i} or {"type": "list", "indices": [i, …]} (system wavelength indices). |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `converged` | boolean |  |  |
| `defocus` | number | mm | Best defocus from the image surface (image local +z). |
| `evaluations` | integer |  | ≥ 0 |
| `field_rms` | array<nf> | mm | Per-field RMS radius at the best focus. |
| `field_rms_at_zero` | array<nf> | mm | Per-field RMS radius at zero defocus. |
| `fields` | array<integer> |  | Fields used. |
| `mean_square_defocus` | number | mm | Closed-form minimiser of the mean square RMS (cross-check). |
| `merit` | number | mm | Field-weighted mean RMS radius at the best focus. |
| `merit_at_zero` | number | mm | The merit at the current image surface. |
| `range` | [number; 2] |  | Search range. |
| `through_focus_defocus` | array<number> |  | Empty unless requested. |
| `through_focus_merit` | array<number> |  |  |

### `chromatic_focal_shift` — Chromatic focal shift

Category `geometric`, result version 1.

Paraxial focus versus wavelength over a sweep, relative to the primary wavelength's focus.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `max_um` | number\|null | `null` | µm | ≥ 0.1; Sweep end; null = the longest system wavelength. |
| `min_um` | number\|null | `null` | µm | ≥ 0.1; Sweep start; null = the shortest system wavelength. |
| `samples` | integer | `41` |  | 2 … 100000; Wavelength samples of the sweep. |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `focal_shift` | array<nf> | mm | Focus relative to the primary wavelength's. |
| `paraxial_focus` | array<nf> | mm | Paraxial focus from the image surface. |
| `primary_wavelength_um` | number | µm |  |
| `range` | nf | mm | max − min of `focal_shift`. |
| `wavelengths_um` | array<number> |  | Sweep wavelengths. |

### `encircled_energy` — Encircled energy

Category `geometric`, result version 1, **GPU path**.

Geometric encircled and ensquared energy versus radius per field, with r50/r80/r90. GPU path.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `defocus` | number | `0.0` | mm | Defocus from the image surface. |
| `fields` | object | `{"type":"all"}` |  | Field selection: {"type": "all"}, {"type": "index", "index": i} or {"type": "list", "indices": [i, …]} (system field indices). |
| `max_radius` | number\|null | `null` | mm | ≥ 0.0; Largest radius; null = the largest GEO radius over the fields. |
| `pattern` | string | `"square"` |  | one of `"hexapolar"`, `"square"`, `"random"`, `"dithered"`; Pupil pattern: hexapolar (1 + 3r(r+1) rays), square (pitch 1/rings, clipped to the disk), random (1 + 3r(r+1) uniform points, Philox stream 0), dithered (one jittered point per square cell, stream 1). |
| `reference` | string | `"centroid"` |  | one of `"chief"`, `"centroid"`; Reference point: the primary-wavelength chief ray or the polychromatic centroid. |
| `rings` | integer | `32` |  | 1 … 512; Density of the pupil pattern (rings, or grid half-count). |
| `samples` | integer | `101` |  | 2 … 100001; Radius samples from 0 to max_radius. |
| `seed` | integer | `1` |  | ≥ 0; Philox seed of the random and dithered patterns. |
| `wavelengths` | object | `{"type":"all"}` |  | Wavelength selection: {"type": "all"}, {"type": "primary"}, {"type": "index", "index": i} or {"type": "list", "indices": [i, …]} (system wavelength indices). |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `fields` | array<object> |  |  |
| `fields[].encircled` | array<number> |  | Fraction of the arrived energy inside radius[i]. |
| `fields[].ensquared` | array<number> |  | Fraction inside the square of half-width radius[i]. |
| `fields[].field` | integer |  | ≥ 0 |
| `fields[].field_value` | [nf; 2] |  | Field value as specified (degrees or mm, x and y). |
| `fields[].half_width80` | nf | mm | Half-width of the square enclosing 80 %. |
| `fields[].r50` | nf | mm |  |
| `fields[].r80` | nf | mm |  |
| `fields[].r90` | nf | mm |  |
| `fields[].rays_arrived` | integer |  | ≥ 0 |
| `radius` | array<number> |  | Radii / half-widths shared by every field. |
| `wavelengths` | array<integer> |  | Selected system wavelength indices. |

### `field_curvature_distortion` — Field curvature and distortion

Category `geometric`, result version 1.

Tangential and sagittal focus shifts along the real chief ray (generalised Coddington, cross-checked by close rays) and distortion (paraxial, f·tanθ, f·θ) versus field.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `close_ray_step` | number | `0.001` |  | 1e-9 … 0.5; Normalised pupil step of the close-ray cross-check. |
| `field_samples` | integer | `21` |  | 2 … 10000; Field samples from the axis to the maximum field along +y. |
| `wavelengths` | object | `{"type":"all"}` |  | Wavelength selection: {"type": "all"}, {"type": "primary"}, {"type": "index", "index": i} or {"type": "list", "indices": [i, …]} (system wavelength indices). |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `curves` | array<object> |  | One per selected wavelength. |
| `curves[].distortion_percent` | array<nf> | % | 100 (y_real − y_paraxial)/y_paraxial. |
| `curves[].f_tan_theta_distortion_percent` | array<nf> | % | Against f·tanθ. |
| `curves[].f_theta_distortion_percent` | array<nf> | % | Against f·θ. |
| `curves[].paraxial_height` | array<nf> | mm | Paraxial chief-ray image height. |
| `curves[].real_height` | array<nf> | mm | Real chief-ray image height. |
| `curves[].sagittal` | array<nf> | mm | Sagittal focus shift (Coddington). |
| `curves[].sagittal_close_rays` | array<nf> | mm | Close-ray cross-check of `sagittal`. |
| `curves[].tangential` | array<nf> | mm | Tangential focus shift from the image surface along z (Coddington). |
| `curves[].tangential_close_rays` | array<nf> | mm | Close-ray cross-check of `tangential`. |
| `curves[].wavelength` | integer |  | ≥ 0; System wavelength index. |
| `curves[].wavelength_um` | number | µm |  |
| `field_unit` | string |  | `°` for angles, `mm` for heights. |
| `field_values` | array<nf> |  | Field samples in `field_unit`. |
| `normalized_field` | array<nf> |  | Field samples normalised to the maximum field. |
| `wavelengths_um` | array<number> |  | All system wavelengths (µm), indexed by system wavelength index. |

### `first_order` — First-order properties

Category `geometric`, result version 1.

Paraxial first-order data (focal lengths, pupils, magnifications, F-numbers, …) as a keyed report table per selected wavelength.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `wavelength` | object | `{"type":"primary"}` |  | Wavelength selection: {"type": "all"}, {"type": "primary"}, {"type": "index", "index": i} or {"type": "list", "indices": [i, …]} (system wavelength indices). |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `entries` | array<object> |  | The report at the first selected wavelength. |
| `entries[].key` | string |  | Stable key (e.g. `efl`, `bfl`, `epd`). |
| `entries[].label` | string |  | Display label. |
| `entries[].unit` | string |  | Unit of `value` (`mm`, `°`, or empty). |
| `entries[].value` | nf |  |  |
| `system` | object |  | System summary: name, surfaces, stop, wavelengths_um, primary_wavelength, field_kind, field_unit, max_field, object_at_infinity, ray_aiming. |
| `tables` | array<object> |  | One report per selected wavelength. |
| `tables[].entries` | array<object> |  |  |
| `tables[].entries[].key` | string |  | Stable key (e.g. `efl`, `bfl`, `epd`). |
| `tables[].entries[].label` | string |  | Display label. |
| `tables[].entries[].unit` | string |  | Unit of `value` (`mm`, `°`, or empty). |
| `tables[].entries[].value` | nf |  |  |
| `tables[].wavelength` | integer |  | ≥ 0; System wavelength index. |
| `tables[].wavelength_um` | number | µm |  |
| `wavelength_um` | number | µm | Wavelength of `entries`. |

### `footprint` — Surface footprint

Category `geometric`, result version 1.

Hit points of a pupil pattern on one surface (local frame), with the points the surface's aperture clips. Points in typed buffers.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `fields` | object | `{"type":"all"}` |  | Field selection: {"type": "all"}, {"type": "index", "index": i} or {"type": "list", "indices": [i, …]} (system field indices). |
| `pattern` | string | `"square"` |  | one of `"hexapolar"`, `"square"`, `"random"`, `"dithered"`; Pupil pattern: hexapolar (1 + 3r(r+1) rays), square (pitch 1/rings, clipped to the disk), random (1 + 3r(r+1) uniform points, Philox stream 0), dithered (one jittered point per square cell, stream 1). |
| `rings` | integer | `8` |  | 1 … 512; Density of the pupil pattern (rings, or grid half-count). |
| `seed` | integer | `1` |  | ≥ 0; Philox seed of the random and dithered patterns. |
| `surface` | integer | `1` |  | 1 … 100000; Surface index (1 … image). |
| `wavelengths` | object | `{"type":"all"}` |  | Wavelength selection: {"type": "all"}, {"type": "primary"}, {"type": "index", "index": i} or {"type": "list", "indices": [i, …]} (system wavelength indices). |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `aperture` | any |  | The surface's aperture shape (model JSON) or null. |
| `extent` | array<nf> | mm | [x_min, x_max, y_min, y_max] over all passing hits. |
| `max_radius` | nf | mm |  |
| `semi_diameter` | nf | mm | Clear semi-diameter (fixed, stop or traced). |
| `sets` | array<object> |  | One per (field, wavelength). |
| `sets[].clipped` | [integer; 2] |  | [offset, count] in `clipped`. |
| `sets[].field` | integer |  | ≥ 0 |
| `sets[].launched` | integer |  | ≥ 0 |
| `sets[].lost_before` | integer |  | ≥ 0; Rays lost before reaching the surface. |
| `sets[].points` | [integer; 2] |  | [offset, count] in `points`. |
| `sets[].wavelength` | integer |  | ≥ 0 |
| `surface` | integer |  | ≥ 0 |

Buffers:

| name | dtype | shape | unit | notes |
|---|---|---|---|---|
| `points` | f64 | [N, 2] | mm | Hit points (local frame) of rays that pass. |
| `clipped` | f64 | [M, 2] | mm | Hit points of rays the aperture clips. |

### `geometric_mtf` — Geometric MTF

Category `geometric`, result version 1, **GPU path**.

Geometric (ray-based) MTF, tangential and sagittal, per field. No diffraction: compare with the wave-optics MTF. GPU path.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `defocus` | number | `0.0` | mm | Defocus from the image surface. |
| `fields` | object | `{"type":"all"}` |  | Field selection: {"type": "all"}, {"type": "index", "index": i} or {"type": "list", "indices": [i, …]} (system field indices). |
| `frequency_samples` | integer | `41` |  | 2 … 100000; Frequency samples from 0 to max_frequency. |
| `max_frequency` | number\|null | `null` | cycles/mm | ≥ 0.0; Highest frequency; null = the diffraction cutoff 1/(λN_w) at the primary wavelength. |
| `pattern` | string | `"square"` |  | one of `"hexapolar"`, `"square"`, `"random"`, `"dithered"`; Pupil pattern: hexapolar (1 + 3r(r+1) rays), square (pitch 1/rings, clipped to the disk), random (1 + 3r(r+1) uniform points, Philox stream 0), dithered (one jittered point per square cell, stream 1). |
| `rings` | integer | `32` |  | 1 … 512; Density of the pupil pattern (rings, or grid half-count). |
| `seed` | integer | `1` |  | ≥ 0; Philox seed of the random and dithered patterns. |
| `wavelengths` | object | `{"type":"all"}` |  | Wavelength selection: {"type": "all"}, {"type": "primary"}, {"type": "index", "index": i} or {"type": "list", "indices": [i, …]} (system wavelength indices). |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `curves` | array<object> |  |  |
| `curves[].field` | integer |  | ≥ 0; System field index. |
| `curves[].modulation` | array<nf> |  | Modulation per frequency (1 at 0). |
| `curves[].orientation` | string |  | one of `"tangential"`, `"sagittal"` |
| `cutoff_frequency` | nf | cycles/mm | Incoherent diffraction cutoff 1/(λN_w), for reference. |
| `diffraction_limit` | any |  | Always null for the geometric MTF. |
| `field_values` | array<[nf; 2]> |  | Field value of each computed field. |
| `frequencies` | array<number> |  |  |

### `grid_distortion` — Grid distortion

Category `geometric`, result version 1.

Chief rays of a square grid of normalised field points against their paraxial image points. Grid in typed buffers.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `grid_size` | integer | `11` |  | 2 … 1001; Grid points per side. |
| `half_width` | number | `0.7071067811865476` |  | 0.0 … 1.0; Half-width of the normalised field square. |
| `wavelength` | integer\|null | `null` |  | ≥ 0; System wavelength index; null = the primary wavelength. |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `grid_size` | integer |  | ≥ 0; Points per side n. |
| `max_abs_distortion_percent` | nf | % |  |
| `wavelength` | integer |  | ≥ 0; System wavelength index. |

Buffers:

| name | dtype | shape | unit | notes |
|---|---|---|---|---|
| `field` | f64 | [n*n, 2] |  | Normalised field (Hx, Hy), x fastest. |
| `paraxial` | f64 | [n*n, 2] | mm | Paraxial image point. |
| `real` | f64 | [n*n, 2] | mm | Real chief-ray image point (NaN if it failed). |
| `distortion_percent` | f64 | [n*n] | % | Radial distortion of each point. |

### `lateral_colour` — Lateral colour

Category `geometric`, result version 1.

Real chief-ray image heights per wavelength versus field, relative to the primary wavelength, and the short−long difference.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `field_samples` | integer | `21` |  | 2 … 10000; Field samples from the axis to the maximum field along +y. |
| `long_wavelength` | integer\|null | `null` |  | ≥ 0; null = the longest system wavelength. |
| `short_wavelength` | integer\|null | `null` |  | ≥ 0; null = the shortest system wavelength. |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `airy_radius` | nf | mm |  |
| `curves` | array<object> |  |  |
| `curves[].offset` | array<nf> | mm | Chief-ray height minus the primary wavelength's. |
| `curves[].wavelength` | integer |  | ≥ 0 |
| `curves[].wavelength_um` | number | µm |  |
| `field_unit` | string |  |  |
| `field_values` | array<nf> |  | Field samples in `field_unit`. |
| `lateral_colour` | array<nf> | mm | Short minus long chief-ray height. |
| `long_wavelength` | integer |  | ≥ 0 |
| `normalized_field` | array<nf> |  |  |
| `short_wavelength` | integer |  | ≥ 0 |

### `layout` — Layout

Category `geometric`, result version 1.

The viewport scene in global coordinates: surface profiles, rings and rims, element outlines, stop, image, axis and ray paths as polylines in typed buffers.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `edge_margin` | number | `0.5` | mm | ≥ 0.0; Mechanical margin beyond the clear semi-diameter where a surface has no mechanical semi-diameter. |
| `fields` | object | `{"type":"all"}` |  | Field selection: {"type": "all"}, {"type": "index", "index": i} or {"type": "list", "indices": [i, …]} (system field indices). |
| `include_3d` | boolean | `true` |  | Also produce out-of-section geometry (sagittal profiles, rings, rims, sagittal fans). |
| `lead_in` | number\|null | `null` | mm | ≥ 0.0; Ray length drawn before the first surface for an object at infinity; null = a quarter of the vertex track. |
| `profile_samples` | integer | `41` |  | 2 … 100000; Points per surface profile. |
| `rays_per_field` | integer | `7` |  | 1 … 10000; Rays per fan, evenly spaced across the pupil (1: the chief ray). |
| `ring_samples` | integer | `64` |  | 3 … 100000; Points per ring. |
| `rings` | integer | `3` |  | 1 … 1000; Concentric rings per surface in 3-D (the last one is the rim). |
| `wavelengths` | object | `{"type":"primary"}` |  | Wavelength selection: {"type": "all"}, {"type": "primary"}, {"type": "index", "index": i} or {"type": "list", "indices": [i, …]} (system wavelength indices). |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `bounds` | object |  |  |
| `bounds.max` | [nf; 3] |  |  |
| `bounds.min` | [nf; 3] |  |  |
| `elements` | array<object> |  | Lens elements (consecutive refracting surfaces with glass between them). |
| `elements[].first_surface` | integer |  | ≥ 0 |
| `elements[].last_surface` | integer |  | ≥ 0 |
| `elements[].mechanical_semi_diameter` | nf | mm |  |
| `elements[].outline_polyline` | integer |  | ≥ 0; Index of the element's closed outline in the polyline buffers (kind 7). |
| `field_count` | integer |  | ≥ 0; Number of system fields. |
| `kind_names` | array<string> |  | Name of each kind code: 0 surface_profile, 1 ring, 2 rim, 3 ray, 4 stop, 5 image, 6 axis, 7 element_outline. |
| `origin` | [number; 3] |  | Global point subtracted from the f32 `vertices` (the bounds centre). |
| `polyline_count` | integer |  | ≥ 0; Polylines P (rays, profiles, … then element outlines). |
| `surfaces` | array<object> |  | Per drawn surface. |
| `surfaces[].axis` | [nf; 3] |  | Global direction of the local +z axis. |
| `surfaces[].id` | integer |  | ≥ 0; Stable surface id. |
| `surfaces[].index` | integer |  | ≥ 0; Surface index. |
| `surfaces[].is_mirror` | boolean |  |  |
| `surfaces[].is_stop` | boolean |  |  |
| `surfaces[].label` | string |  |  |
| `surfaces[].mechanical_semi_diameter` | nf | mm |  |
| `surfaces[].semi_diameter` | nf | mm | Clear semi-diameter drawn. |
| `surfaces[].semi_diameter_is_auto` | boolean |  |  |
| `surfaces[].vertex` | [nf; 3] |  | Global vertex position (mm). |
| `vertex_count` | integer |  | ≥ 0; Vertices V. |
| `wavelengths_um` | array<number> |  | All system wavelengths (µm), indexed by system wavelength index. |

Buffers:

| name | dtype | shape | unit | notes |
|---|---|---|---|---|
| `vertices` | f32 | [V, 3] | mm | x, y, z relative to `origin` (global frame). |
| `vertices_f64` | f64 | [V, 3] | mm | Absolute global x, y, z. |
| `offsets` | u32 | [P+1] |  | Polyline i is vertices offsets[i] .. offsets[i+1]. |
| `kinds` | u8 | [P] |  | Kind code (see `kind_names`). |
| `surfaces` | i32 | [P] |  | Surface index, −1 for none. |
| `fields` | i32 | [P] |  | Field index of a ray, −1 for none. |
| `wavelengths` | i32 | [P] |  | Wavelength index of a ray, −1 for none. |
| `flags` | u8 | [P] |  | Bit 0: in the meridional section (x = 0); bit 1: vignetted ray. |

### `longitudinal_aberration` — Longitudinal aberration

Category `geometric`, result version 1.

On-axis zonal rays: where each crosses the axis, relative to the primary paraxial focus, versus pupil zone, per wavelength.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `pupil_samples` | integer | `21` |  | 2 … 10000; Pupil zones from 0 to 1 (P_y). |
| `wavelengths` | object | `{"type":"all"}` |  | Wavelength selection: {"type": "all"}, {"type": "primary"}, {"type": "index", "index": i} or {"type": "list", "indices": [i, …]} (system wavelength indices). |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `curves` | array<object> |  |  |
| `curves[].aberration` | array<nf> | mm | Crossing relative to the primary paraxial focus. |
| `curves[].focus` | array<nf> | mm | Axis crossing of each zonal ray, from the image surface. |
| `curves[].paraxial_focus` | nf | mm |  |
| `curves[].wavelength` | integer |  | ≥ 0 |
| `curves[].wavelength_um` | number | µm |  |
| `primary_paraxial_focus` | nf | mm | Primary paraxial focus from the image surface. |
| `pupil` | array<number> |  | Pupil zones P_y from 0 to 1. |

### `opd_fan` — OPD fans

Category `geometric`, result version 1.

Optical path difference (waves) against the exit-pupil reference sphere versus normalised pupil coordinate, tangential and sagittal.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `fields` | object | `{"type":"all"}` |  | Field selection: {"type": "all"}, {"type": "index", "index": i} or {"type": "list", "indices": [i, …]} (system field indices). |
| `reference` | string | `"primary_chief"` |  | one of `"primary_chief"`, `"same_wavelength"`; Chief ray the aberrations are measured from: the primary wavelength's, or each wavelength's own. |
| `samples` | integer | `41` |  | 2 … 100001; Pupil samples across each fan, from −1 to 1. |
| `wavelengths` | object | `{"type":"all"}` |  | Wavelength selection: {"type": "all"}, {"type": "primary"}, {"type": "index", "index": i} or {"type": "list", "indices": [i, …]} (system wavelength indices). |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `fields` | array<object> |  |  |
| `fields[].field` | integer |  | ≥ 0; System field index. |
| `fields[].field_value` | [nf; 2] |  | Field value as specified (degrees or mm, x and y). |
| `fields[].sagittal` | array<object> |  | One curve per selected wavelength. |
| `fields[].sagittal[].pupil` | array<number> |  | Normalised pupil coordinate P_y (tangential) or P_x (sagittal), −1 … 1. |
| `fields[].sagittal[].values` | array<nf> | waves | OPD in waves at the curve's wavelength |
| `fields[].sagittal[].wavelength` | integer |  | ≥ 0; System wavelength index. |
| `fields[].tangential` | array<object> |  | One curve per selected wavelength. |
| `fields[].tangential[].pupil` | array<number> |  | Normalised pupil coordinate P_y (tangential) or P_x (sagittal), −1 … 1. |
| `fields[].tangential[].values` | array<nf> | waves | OPD in waves at the curve's wavelength |
| `fields[].tangential[].wavelength` | integer |  | ≥ 0; System wavelength index. |
| `quantity` | string |  | one of `"transverse_aberration"`, `"opd"` |
| `wavelengths_um` | array<number> |  | All system wavelengths (µm), indexed by system wavelength index. |

### `ray_fan` — Ray aberration fans

Category `geometric`, result version 1.

Transverse ray aberration εy (tangential fan, P_y) and εx (sagittal fan, P_x) on the image surface versus normalised pupil coordinate, per field and wavelength (mm).

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `fields` | object | `{"type":"all"}` |  | Field selection: {"type": "all"}, {"type": "index", "index": i} or {"type": "list", "indices": [i, …]} (system field indices). |
| `reference` | string | `"primary_chief"` |  | one of `"primary_chief"`, `"same_wavelength"`; Chief ray the aberrations are measured from: the primary wavelength's, or each wavelength's own. |
| `samples` | integer | `41` |  | 2 … 100001; Pupil samples across each fan, from −1 to 1. |
| `wavelengths` | object | `{"type":"all"}` |  | Wavelength selection: {"type": "all"}, {"type": "primary"}, {"type": "index", "index": i} or {"type": "list", "indices": [i, …]} (system wavelength indices). |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `fields` | array<object> |  |  |
| `fields[].field` | integer |  | ≥ 0; System field index. |
| `fields[].field_value` | [nf; 2] |  | Field value as specified (degrees or mm, x and y). |
| `fields[].sagittal` | array<object> |  | One curve per selected wavelength. |
| `fields[].sagittal[].cross` (optional) | array<nf> | mm | The other transverse component (εx on the tangential fan, εy on the sagittal fan). |
| `fields[].sagittal[].pupil` | array<number> |  | Normalised pupil coordinate P_y (tangential) or P_x (sagittal), −1 … 1. |
| `fields[].sagittal[].values` | array<nf> | mm | εy (tangential) / εx (sagittal) on the image surface, mm |
| `fields[].sagittal[].wavelength` | integer |  | ≥ 0; System wavelength index. |
| `fields[].tangential` | array<object> |  | One curve per selected wavelength. |
| `fields[].tangential[].cross` (optional) | array<nf> | mm | The other transverse component (εx on the tangential fan, εy on the sagittal fan). |
| `fields[].tangential[].pupil` | array<number> |  | Normalised pupil coordinate P_y (tangential) or P_x (sagittal), −1 … 1. |
| `fields[].tangential[].values` | array<nf> | mm | εy (tangential) / εx (sagittal) on the image surface, mm |
| `fields[].tangential[].wavelength` | integer |  | ≥ 0; System wavelength index. |
| `quantity` | string |  | one of `"transverse_aberration"`, `"opd"` |
| `wavelengths_um` | array<number> |  | All system wavelengths (µm), indexed by system wavelength index. |

### `ray_grid` — Ray grid

Category `geometric`, result version 1, **GPU path**.

Image-surface hits (local frame) and directions of a fields × wavelengths × pupil ray grid, plus the chief rays, as typed buffers (NaN for rays that did not arrive). GPU path.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `fields` | object | `{"type":"all"}` |  | Field selection: {"type": "all"}, {"type": "index", "index": i} or {"type": "list", "indices": [i, …]} (system field indices). |
| `pattern` | string | `"hexapolar"` |  | one of `"hexapolar"`, `"square"`, `"random"`, `"dithered"`; Pupil pattern: hexapolar (1 + 3r(r+1) rays), square (pitch 1/rings, clipped to the disk), random (1 + 3r(r+1) uniform points, Philox stream 0), dithered (one jittered point per square cell, stream 1). |
| `rings` | integer | `6` |  | 1 … 512; Density of the pupil pattern (rings, or grid half-count). |
| `seed` | integer | `1` |  | ≥ 0; Philox seed of the random and dithered patterns. |
| `wavelengths` | object | `{"type":"all"}` |  | Wavelength selection: {"type": "all"}, {"type": "primary"}, {"type": "index", "index": i} or {"type": "list", "indices": [i, …]} (system wavelength indices). |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `arrived` | integer |  | ≥ 0; Rays that reached the image surface. |
| `fields` | array<integer> |  | Selected system fields (axis F). |
| `launched` | integer |  | ≥ 0; F × W × P. |
| `pupil_count` | integer |  | ≥ 0; Pupil samples P. |
| `wavelengths` | array<integer> |  | Selected system wavelength indices. |
| `wavelengths_um` | array<number> |  | Wavelengths of axis W. |

Buffers:

| name | dtype | shape | unit | notes |
|---|---|---|---|---|
| `pupil` | f64 | [P, 2] |  | Normalised pupil coordinates (Px, Py). |
| `points` | f64 | [F, W, P, 2] | mm | Hit (x, y) on the image surface, local frame; NaN where the ray did not arrive. |
| `directions` | f64 | [F, W, P, 3] |  | Unit direction (image local frame); NaN where the ray did not arrive. |
| `chief` | f64 | [F, W, 2] | mm | Unclipped chief-ray hit (x, y); NaN if it failed. |

### `relative_illumination` — Relative illumination

Category `geometric`, result version 1.

Image irradiance of a uniform Lambertian object versus field, from the projected solid angle of the real exit pupil seen from each image point (real ray aiming), with the vignetted fraction.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `azimuth_samples` | integer | `48` |  | 3 … 10000; Trapezoidal nodes in the pupil azimuth. |
| `field_samples` | integer | `11` |  | 2 … 10000; Field samples from the axis to the maximum field along +y. |
| `radial_samples` | integer | `12` |  | 1 … 1000; Gauss–Legendre nodes in the pupil radius. |
| `wavelength` | integer\|null | `null` |  | ≥ 0; System wavelength index; null = the primary wavelength. |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `chief_ray_angle_deg` | array<nf> | ° | Chief-ray angle of incidence on the image surface. |
| `cos4_object` | array<nf> |  | cos⁴ of the object-space chief angle, for comparison. |
| `field_unit` | string |  |  |
| `field_values` | array<nf> |  | Field samples in `field_unit`. |
| `normalized_field` | array<nf> |  |  |
| `projected_solid_angle` | array<nf> | sr | Projected solid angle of the real exit pupil seen from the image point. |
| `relative_illumination` | array<nf> |  | Ω_p(H)/Ω_p(0). |
| `vignetting_fraction` | array<nf> |  | Clipped share of the stop area. |
| `wavelength` | integer |  | ≥ 0 |

### `seidel` — Seidel aberration sums

Category `geometric`, result version 1.

Third-order (Seidel) coefficients S_I…S_V and chromatic C_L, C_T per surface and summed, Petzval sum and radius, wavefront coefficients.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `long_wavelength` | integer\|null | `null` |  | ≥ 0; Long wavelength of δn; null = the longest system wavelength. |
| `short_wavelength` | integer\|null | `null` |  | ≥ 0; Short wavelength of δn; null = the shortest system wavelength. |
| `wavelength` | integer\|null | `null` |  | ≥ 0; System wavelength index; null = the primary wavelength. |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `image_marginal_slope` | nf |  | Image-space reduced marginal slope ω'. |
| `lagrange_invariant` | nf | mm |  |
| `long_wavelength` | integer |  | ≥ 0; Long wavelength of δn. |
| `petzval_radius` | nf | mm | −1/(n'_k P), signed image-space reduced index. |
| `petzval_sum` | nf | 1/mm | Σ φ/(n n'). |
| `short_wavelength` | integer |  | ≥ 0; Short wavelength of δn. |
| `sums` | object |  |  |
| `sums.cl` | nf | mm | Axial colour C_L. |
| `sums.ct` | nf | mm | Lateral colour C_T. |
| `sums.s1` | nf | mm | Spherical S_I. |
| `sums.s2` | nf | mm | Coma S_II. |
| `sums.s3` | nf | mm | Astigmatism S_III. |
| `sums.s4` | nf | mm | Petzval S_IV. |
| `sums.s5` | nf | mm | Distortion S_V. |
| `surfaces` | array<object> |  | Per-surface contributions (surfaces that contribute). |
| `surfaces[].cl` | nf | mm | Axial colour C_L. |
| `surfaces[].ct` | nf | mm | Lateral colour C_T. |
| `surfaces[].s1` | nf | mm | Spherical S_I. |
| `surfaces[].s2` | nf | mm | Coma S_II. |
| `surfaces[].s3` | nf | mm | Astigmatism S_III. |
| `surfaces[].s4` | nf | mm | Petzval S_IV. |
| `surfaces[].s5` | nf | mm | Distortion S_V. |
| `surfaces[].surface` | integer |  | ≥ 0; Surface index. |
| `wavefront_mm` | object |  |  |
| `wavefront_mm.w020` | nf | mm |  |
| `wavefront_mm.w040` | nf | mm |  |
| `wavefront_mm.w111` | nf | mm |  |
| `wavefront_mm.w131` | nf | mm |  |
| `wavefront_mm.w220` | nf | mm |  |
| `wavefront_mm.w220p` | nf | mm |  |
| `wavefront_mm.w222` | nf | mm |  |
| `wavefront_mm.w311` | nf | mm |  |
| `wavefront_waves` | object |  |  |
| `wavefront_waves.w020` | nf | waves |  |
| `wavefront_waves.w040` | nf | waves |  |
| `wavefront_waves.w111` | nf | waves |  |
| `wavefront_waves.w131` | nf | waves |  |
| `wavefront_waves.w220` | nf | waves |  |
| `wavefront_waves.w220p` | nf | waves |  |
| `wavefront_waves.w222` | nf | waves |  |
| `wavefront_waves.w311` | nf | waves |  |
| `wavelength` | integer |  | ≥ 0; System wavelength index of the paraxial rays. |
| `wavelength_um` | number | µm |  |

### `spot_diagram` — Spot diagram

Category `geometric`, result version 1, **GPU path**.

Ray intercepts on the image surface per field and wavelength: centroid, RMS and GEO radii about centroid and chief ray, polychromatic statistics, vignetted fraction, Airy radius, optional through-focus. Ray points in typed buffers. GPU path.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `defocus` | number | `0.0` | mm | Defocus from the image surface along its local +z. |
| `fields` | object | `{"type":"all"}` |  | Field selection: {"type": "all"}, {"type": "index", "index": i} or {"type": "list", "indices": [i, …]} (system field indices). |
| `include_points` | boolean | `true` |  | Return the individual ray points (buffers `points`, `point_field`, `point_wavelength`). |
| `pattern` | string | `"hexapolar"` |  | one of `"hexapolar"`, `"square"`, `"random"`, `"dithered"`; Pupil pattern: hexapolar (1 + 3r(r+1) rays), square (pitch 1/rings, clipped to the disk), random (1 + 3r(r+1) uniform points, Philox stream 0), dithered (one jittered point per square cell, stream 1). |
| `reference` | string | `"chief"` |  | one of `"chief"`, `"centroid"`; Reference point: the primary-wavelength chief ray or the polychromatic centroid. |
| `rings` | integer | `6` |  | 1 … 512; Density of the pupil pattern (rings, or grid half-count). |
| `seed` | integer | `1` |  | ≥ 0; Philox seed of the random and dithered patterns. |
| `through_focus` | object\|null | `null` |  | {"min": mm, "max": mm, "steps": n} or null. |
| `wavelengths` | object | `{"type":"all"}` |  | Wavelength selection: {"type": "all"}, {"type": "primary"}, {"type": "index", "index": i} or {"type": "list", "indices": [i, …]} (system wavelength indices). |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `airy_radius` | nf | mm | 1.22 λ N_w at the primary wavelength; "inf" when afocal. |
| `defocus` | number | mm |  |
| `fields` | array<object> |  | One entry per selected field. |
| `fields[].centroid` | [nf; 2] |  | Polychromatic centroid (image local mm). |
| `fields[].chief` | [nf; 2] |  | Primary-wavelength chief ray (image local mm). |
| `fields[].field` | integer |  | ≥ 0; System field index. |
| `fields[].field_value` | [nf; 2] |  | Field value as specified (degrees or mm, x and y). |
| `fields[].geo_radius` | nf | mm | Largest distance from the reference. |
| `fields[].per_wavelength` | array<object> |  |  |
| `fields[].per_wavelength[].chief` | [nf; 2] |  | This wavelength's chief ray on the image surface (local mm). |
| `fields[].per_wavelength[].stats` | object |  |  |
| `fields[].per_wavelength[].stats.centroid` | [nf; 2] |  | Centroid relative to the chief ray (mm). |
| `fields[].per_wavelength[].stats.geo_radius_centroid` | nf | mm |  |
| `fields[].per_wavelength[].stats.geo_radius_chief` | nf | mm |  |
| `fields[].per_wavelength[].stats.rays_arrived` | integer |  | ≥ 0 |
| `fields[].per_wavelength[].stats.rays_launched` | integer |  | ≥ 0 |
| `fields[].per_wavelength[].stats.rms_radius_centroid` | nf | mm |  |
| `fields[].per_wavelength[].stats.rms_radius_chief` | nf | mm |  |
| `fields[].per_wavelength[].stats.rms_x` | nf | mm | RMS of x about the centroid. |
| `fields[].per_wavelength[].stats.rms_y` | nf | mm | RMS of y about the centroid. |
| `fields[].per_wavelength[].stats.vignetted_fraction` | nf |  | Weighted fraction of launched rays that did not arrive. |
| `fields[].per_wavelength[].wavelength` | integer |  | ≥ 0; System wavelength index. |
| `fields[].per_wavelength[].wavelength_um` | number | µm |  |
| `fields[].per_wavelength[].weight` | number |  | Spectral weight. |
| `fields[].point_range` | [integer; 2] |  | [offset, count] of this field's rows in the point buffers. |
| `fields[].polychromatic` | object |  |  |
| `fields[].polychromatic.centroid` | [nf; 2] |  | Centroid relative to the chief ray (mm). |
| `fields[].polychromatic.geo_radius_centroid` | nf | mm |  |
| `fields[].polychromatic.geo_radius_chief` | nf | mm |  |
| `fields[].polychromatic.rays_arrived` | integer |  | ≥ 0 |
| `fields[].polychromatic.rays_launched` | integer |  | ≥ 0 |
| `fields[].polychromatic.rms_radius_centroid` | nf | mm |  |
| `fields[].polychromatic.rms_radius_chief` | nf | mm |  |
| `fields[].polychromatic.rms_x` | nf | mm | RMS of x about the centroid. |
| `fields[].polychromatic.rms_y` | nf | mm | RMS of y about the centroid. |
| `fields[].polychromatic.vignetted_fraction` | nf |  | Weighted fraction of launched rays that did not arrive. |
| `fields[].reference` | [nf; 2] |  | Reference point of the plotted points (image local mm). |
| `fields[].rms_radius` | nf | mm | Polychromatic RMS radius about the reference. |
| `fields[].through_focus` (optional) | array<object> |  | Present when through_focus was requested. |
| `fields[].through_focus[].defocus` | number | mm |  |
| `fields[].through_focus[].rms_radius_centroid` | nf | mm |  |
| `fields[].through_focus[].rms_radius_chief` | nf | mm |  |
| `wavelengths` | array<integer> |  | Selected system wavelength indices. |
| `wavelengths_um` | array<number> |  | All system wavelengths (µm), indexed by system wavelength index. |

Buffers:

| name | dtype | shape | unit | notes |
|---|---|---|---|---|
| `points` | f64 | [N, 2] | mm | Ray points (x, y) relative to their field's `reference`, image local frame; empty unless include_points. |
| `point_field` | u32 | [N] |  | Position of each point's field in `fields`. |
| `point_wavelength` | u32 | [N] |  | System wavelength index of each point. |

### `system_transmission` — System transmission

Category `radiometry`, result version 1.

Pupil-averaged primary-path transmission (coatings, Fresnel losses, bulk absorption, vignetting) per field and wavelength, and the spectrally weighted polychromatic transmission per field; unpolarised or by polarisation ray tracing.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `clip` | boolean | `true` |  | Apply apertures (vignetted rays transmit nothing). |
| `fields` | array | `[]` |  | Normalised field points [{"hx", "hy"}]; empty = the system fields (with their vignetting factors). |
| `model` | object | `{"type":"unpolarized"}` |  | {"type": "unpolarized"} (product of s/p-averaged interface factors) or {"type": "polarized", "state": …} (polarisation ray tracing). |
| `pupil` | object | `{"n":32,"type":"grid"}` |  | Pupil sampling: {"type": "grid", "n": n} (n×n grid inside the unit disk, n = 1 or ≥ 3) or {"type": "random", "count": n, "seed": s}. |
| `wavelengths` | array<integer> | `[]` |  | System wavelength indices; empty = all. |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `fields` | array<object> |  |  |
| `points` | array<object> |  | Field-major: entry f·W + w. |
| `points[].failed` | integer |  | ≥ 0; Samples excluded after a numerical failure. |
| `points[].field` | integer |  | ≥ 0; Index into `fields`. |
| `points[].mean_unvignetted` | number |  | Mean throughput of the rays that reach the image. |
| `points[].samples` | integer |  | ≥ 0; Usable samples. |
| `points[].transmission` | number |  | Mean throughput over all usable samples (blocked rays count 0). |
| `points[].unvignetted_fraction` | number |  |  |
| `points[].wavelength` | integer |  | ≥ 0; System wavelength index. |
| `points[].wavelength_um` | number | µm |  |
| `polychromatic` | array<number> |  | Per field: spectral-weight average of transmission. |
| `wavelengths` | array<integer> |  |  |

### `polarization_pupil_map` — Polarisation pupil map

Category `polarization`, result version 1.

Polarisation ray tracing over the pupil of one field and wavelength: amplitude Jones matrix (double-pole bases), transmitted intensity for an input state, diattenuation and retardance per sample, with pupil statistics.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `clip` | boolean | `true` |  | Apply apertures. |
| `field` | integer | `0` |  | ≥ 0; System field index (with its vignetting factors). |
| `grid` | integer | `33` |  | 1 … 2049; Pupil grid n (n×n inside the unit disk; 1 or ≥ 3). |
| `point` | array\|null | `null` |  | Normalised field point [hx, hy] used instead of `field`. |
| `state` | object | `{"type":"unpolarized"}` |  | Input polarisation in the object-space double-pole basis: {"type": "unpolarized"}, {"type": "jones", "e": …} or {"type": "stokes", "s": {"s0", "s1", "s2", "s3"}} (CONVENTIONS §9). |
| `wavelength` | integer\|null | `null` |  | ≥ 0; System wavelength index; null = the primary. |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `field` | object |  | The normalised field point {hx, hy, vignetting}. |
| `grid` | integer |  | ≥ 0 |
| `max_diattenuation` | number |  |  |
| `max_retardance` | number | rad | Over samples with a defined retarder. |
| `mean_intensity` | number |  | Pupil-mean intensity over the samples reaching the image. |
| `rms_retardance` | number | rad |  |
| `sample_count` | integer |  | ≥ 0; Samples N (grid points inside the unit disk). |
| `undefined_retarder` | integer |  | ≥ 0; Samples reaching the image with a singular Jones matrix (excluded from the retardance statistics). |
| `wavelength` | integer |  | ≥ 0; System wavelength index. |
| `wavelength_um` | number | µm |  |

Buffers:

| name | dtype | shape | unit | notes |
|---|---|---|---|---|
| `pupil` | f64 | [N, 2] |  | Normalised pupil coordinates (Px, Py). |
| `reached` | u8 | [N] |  | 1 when the ray reached the image (else every value is zero). |
| `jones` | f64 | [N, 2, 2, 2] |  | Amplitude Jones matrix √τ·J, [row][col][re, im], double-pole bases. |
| `intensity` | f64 | [N] |  | Transmitted power fraction for `state` (normalised by S₀). |
| `diattenuation` | f64 | [N] |  | (T_max − T_min)/(T_max + T_min). |
| `diattenuation_axis` | f64 | [N, 3] |  | Unit Stokes vector of the best-transmitted state. |
| `retardance` | f64 | [N] | rad | Proper retardance δ ∈ [0, π] (0 when undefined). |
| `fast_axis` | f64 | [N, 3] |  | Unit Stokes vector of the fast eigenpolarisation. |
| `retarder_defined` | u8 | [N] |  |  |
| `retardance_jones` | f64 | [N] | rad | Retardance of the Jones matrix itself (includes the geometric rotation and mirror parity). |
| `t_max` | f64 | [N] |  |  |
| `t_min` | f64 | [N] |  |  |

### `ghost_irradiance` — Ghost irradiance

Category `ghost`, result version 1.

Real-ray ghost irradiance on the sensor: per-path flux and peak ratios against the primary image, per-field statistics, the total ghost map and the primary map (typed buffers).

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `clip` | object | `{"margin_mm":0.0,"mode":"clear_apertures"}` |  | Clipping apertures: {"mode": "clear_apertures", "margin_mm": m} (clear semi-diameters + margin; the stop at its size), {"mode": "fixed_apertures"} or {"mode": "none"}. |
| `enumeration` | object | `{"exclude_surfaces":[],"max_higher_order_paths":2000,"max_reflections":2,"min_reflectance_product":0.0,"surface_range":null}` |  | Path enumeration: {"max_reflections": 2\|4\|6\|8, "max_higher_order_paths": n, "min_reflectance_product": r, "surface_range": [first, last] \| null, "exclude_surfaces": [i, …]}. |
| `fields` | array<integer> | `[]` |  | System field indices; empty = all fields (unless points are given). |
| `keep_path_maps` | boolean | `false` |  | Also return every path's map (buffer path_maps). |
| `map_bins` | array<integer> | `[64,64]` |  | Map bins [nx, ny]. |
| `map_half_size` | array\|null | `null` | mm | Map half-size [x, y]; null = the sensor rectangle, else the image clear aperture. |
| `overfill` | number | `1.5` |  | ≥ 1.0; Sampled pupil radius in entrance-pupil radii. |
| `paths` | array\|null | `null` |  | Explicit ghost paths (each an ordered list of reflecting surface indices) instead of the enumeration. |
| `points` | array | `[]` |  | Extra normalised field points [{"hx", "hy"}], weight 1 each, analysed after the system fields. |
| `pupil_samples` | integer | `48` |  | 2 … 4096; Grid cells across the overfilled pupil square. |
| `reject_virtual` | boolean | `true` |  | Treat an intersection behind the ray as a lost ray. |
| `sensor_reflectance` | number\|null | `null` |  | 0.0 … 1.0; Unpolarised, angle-independent reflectance of the sensor; null: the sensor does not reflect. |
| `wavelengths` | array<integer> | `[]` |  | System wavelength indices; empty = all. |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `coverage` | object |  |  |
| `coverage.apertures` | array<object> |  | Clipping of every surface in this run. |
| `coverage.interfaces` | array<string> |  | Interface kind per surface: object, passive, refractive, ideal_lens, mirror, image. |
| `coverage.paths` | array<array<integer>> |  | The exact set of ghost paths this run counted. |
| `ghosts` | array<object> |  | Ranked by peak_ratio, strongest first. |
| `ghosts[].fields` | array<object> |  |  |
| `ghosts[].fields[].centroid` | [nf; 2] |  | mm |
| `ghosts[].fields[].field` | object |  | {"type": "system", "index": i} or {"type": "point", "point": {hx, hy}} |
| `ghosts[].fields[].flux` | number |  | Detected flux (on or off the map), relative to the entrance-pupil input. |
| `ghosts[].fields[].flux_on_map` | number |  |  |
| `ghosts[].fields[].peak` | number | 1/mm² | Peak irradiance of this field's map. |
| `ghosts[].fields[].rays_detected` | integer |  | ≥ 0 |
| `ghosts[].fields[].rays_traced` | integer |  | ≥ 0 |
| `ghosts[].fields[].rms_radius` | nf | mm |  |
| `ghosts[].flux` | number |  | Flux on the map, all fields. |
| `ghosts[].flux_ratio` | number |  | Flux on the map / primary flux on the map. |
| `ghosts[].label` | string |  |  |
| `ghosts[].map_index` | integer? |  |  |
| `ghosts[].path` | array<integer> |  | Reflecting surfaces (empty: the primary). |
| `ghosts[].peak_ratio` | number |  | Max over fields of peak / primary peak. |
| `map` | object |  |  |
| `map.bins` | [integer; 2] |  | [nx, ny] |
| `map.half_size` | [number; 2] |  | [x, y]; the maps span −half_size … +half_size. |
| `primary` | object |  |  |
| `primary.fields` | array<object> |  |  |
| `primary.fields[].centroid` | [nf; 2] |  | mm |
| `primary.fields[].field` | object |  | {"type": "system", "index": i} or {"type": "point", "point": {hx, hy}} |
| `primary.fields[].flux` | number |  | Detected flux (on or off the map), relative to the entrance-pupil input. |
| `primary.fields[].flux_on_map` | number |  |  |
| `primary.fields[].peak` | number | 1/mm² | Peak irradiance of this field's map. |
| `primary.fields[].rays_detected` | integer |  | ≥ 0 |
| `primary.fields[].rays_traced` | integer |  | ≥ 0 |
| `primary.fields[].rms_radius` | nf | mm |  |
| `primary.flux` | number |  | Flux on the map, all fields. |
| `primary.flux_ratio` | number |  | Flux on the map / primary flux on the map. |
| `primary.label` | string |  |  |
| `primary.map_index` | integer? |  |  |
| `primary.path` | array<integer> |  | Reflecting surfaces (empty: the primary). |
| `primary.peak_ratio` | number |  | Max over fields of peak / primary peak. |
| `total_flux_ratio` | number |  | Total ghost flux / primary flux. |
| `total_peak_ratio` | number |  | Total ghost peak / primary peak. |
| `truncated` | boolean |  |  |

Buffers:

| name | dtype | shape | unit | notes |
|---|---|---|---|---|
| `primary_map` | f64 | [ny, nx] | 1/mm² | Primary irradiance (flux per mm², relative to the entrance-pupil input); row 0 at y = −half_y, x fastest. |
| `total_ghost_map` | f64 | [ny, nx] | 1/mm² | Sum of every ghost's irradiance. |
| `path_maps` | f64 | [G, ny, nx] | 1/mm² | Per-ghost maps in `ghosts` order (keep_path_maps only). |

### `ghost_paths` — Ghost paths

Category `ghost`, result version 1.

First-order ghost ranking: every enumerated two-reflection (or higher) ghost with its reflectance product, transmittance, focus from the sensor, blur diameter, image height and estimated irradiance relative to the signal.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `clip` | object | `{"margin_mm":0.0,"mode":"clear_apertures"}` |  | Clipping apertures: {"mode": "clear_apertures", "margin_mm": m} (clear semi-diameters + margin; the stop at its size), {"mode": "fixed_apertures"} or {"mode": "none"}. |
| `enumeration` | object | `{"exclude_surfaces":[],"max_higher_order_paths":2000,"max_reflections":2,"min_reflectance_product":0.0,"surface_range":null}` |  | Path enumeration: {"max_reflections": 2\|4\|6\|8, "max_higher_order_paths": n, "min_reflectance_product": r, "surface_range": [first, last] \| null, "exclude_surfaces": [i, …]}. |
| `near_focus_factor` | number | `10.0` |  | ≥ 0.0; Ghosts whose spot is at most this many signal diameters are flagged near_focus. |
| `paths` | array\|null | `null` |  | Explicit ghost paths (each an ordered list of reflecting surface indices) instead of the enumeration. |
| `reject_virtual` | boolean | `true` |  | Treat an intersection behind the ray as a lost ray. |
| `sensor_reflectance` | number\|null | `null` |  | 0.0 … 1.0; Unpolarised, angle-independent reflectance of the sensor; null: the sensor does not reflect. |
| `signal_diameter_mm` | number\|null | `null` | mm | ≥ 0.0; Signal spot diameter; null = automatic. |
| `wavelength` | integer\|null | `null` |  | ≥ 0; System wavelength index; null = the primary. |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `coverage` | object |  |  |
| `coverage.apertures` | array<object> |  | Clipping of every surface in this run. |
| `coverage.interfaces` | array<string> |  | Interface kind per surface: object, passive, refractive, ideal_lens, mirror, image. |
| `coverage.paths` | array<array<integer>> |  | The exact set of ghost paths this run counted. |
| `ghosts` | array<object> |  | Ranked by relative_irradiance, strongest first. |
| `ghosts[].focus_from_sensor` | nf | mm | Signed distance of the ghost focus from the sensor (image local z). |
| `ghosts[].image_height` | nf | mm | Ghost centre height on the sensor at the maximum field. |
| `ghosts[].label` | string |  | e.g. `R11·R9`. |
| `ghosts[].near_focus` | boolean |  |  |
| `ghosts[].path` | array<integer> |  | Reflecting surfaces, in order. |
| `ghosts[].reflectance_product` | number |  | Π R, normal incidence, unpolarised. |
| `ghosts[].relative_irradiance` | nf |  | Estimated ghost irradiance relative to the signal. |
| `ghosts[].spot_diameter` | nf | mm | Ghost blur diameter on the sensor (full aperture, on axis). |
| `ghosts[].transmittance` | number |  | Π T of the passes times axial bulk transmittance. |
| `ghosts[].vignetting` | nf |  | Paraxial vignetting estimate in (0, 1]. |
| `primary_focus_from_sensor` | nf | mm |  |
| `primary_image_height` | nf | mm |  |
| `primary_transmittance` | number |  | Normal incidence, axial absorption. |
| `signal_diameter` | nf | mm |  |
| `truncated` | boolean |  | Higher-order paths were dropped by the budget. |
| `wavelength` | integer |  | ≥ 0; System wavelength index. |

### `stray_light` — Stray light

Category `stray`, result version 2.

Non-sequential Monte-Carlo stray light: detector irradiance per path class (primary, ghost, scatter, mechanical, sneak) with standard errors, the stray total, the strongest path signatures and an energy balance. With `ghost_partition` the ghost paths the explicit ghost run covers are left to it, with identical apertures; uncovered ghost paths stay in the stray total.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `classes` | integer | `30` |  | 0 … 31; Classes in the stray total and maps, a bit mask: 1 primary, 2 ghost, 4 scatter, 8 mechanical, 16 sneak (default 30: all but primary). |
| `detector_margin` | number | `0.05` |  | ≥ 0.0; Relative detector margin over the traced image height (automatic image surface without a sensor). |
| `detector_pixels` | array<integer> | `[128,128]` |  | Detector map [nx, ny]. |
| `exclude_ghost_class` | boolean | `false` |  | Remove the ghost class from the total (set automatically by ghost_partition). |
| `ghost_partition` | object\|null | `null` |  | Partition against the explicit ghost solver: null (stray alone), or the ghost run's model options {sensor_reflectance, clip, reject_virtual, enumeration, paths} (the same as its ghost_paths / ghost_irradiance settings). |
| `rays` | integer | `20000` |  | 1 … 1000000000; Source rays. |
| `scene` | object | `{"clear_semi_diameters":[],"default_edge_finish":{"tis":0.05,"type":"lambertian"},"default_edge_margin":1.0,"detector_pixels":[128,128],"include_mechanics":true,"mechanical_specular_reflectance":0.0,"polished_outer_zone":false}` |  | Scene construction (mklight_stray::SceneOptions); clear_semi_diameters is replaced under ghost_partition. |
| `seed` | integer | `1` |  | ≥ 0; Philox seed; ray i uses stream (seed, i). |
| `semi_diameters` | object | `{"azimuths":64,"clip":true}` |  | Sizing traces of the automatic clear semi-diameters. |
| `source` | object | `{"index":0,"margin":0.1,"strength":1.0,"type":"system_field"}` |  | Source: {"type": "system_field", "index", "margin", "strength"}, {"type": "field", "hx", "hy", "margin", "strength"}, {"type": "out_of_field", "angle_x_deg", "angle_y_deg", "margin", "irradiance"} or {"type": "custom", "spec", "reference_flux"}. |
| `top_signatures` | integer | `20` |  | ≥ 0; Number of path signatures reported. |
| `transport` | object | `{"chunk_rays":1024,"max_events":512,"mode":{"mode":"weighted","rare_floor":0.02},"roulette_threshold":0.001,"scatter_reflection":true,"scatter_transmission":true,"split_ceiling":2.0}` |  | Monte-Carlo transport controls (mode, max_events, roulette_threshold, split_ceiling, scatter_reflection, scatter_transmission, chunk_rays). |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `classes` | integer |  | ≥ 0; Class mask of the stray total and maps. |
| `clear_semi_diameters` | array<number?> |  | Clear semi-diameters the scene used per surface (null: the surface's own). |
| `detector` | object |  | Detector: surface, nx, ny, half_x, half_y (mm), accept. |
| `emitted` | number | W | Emitted power. |
| `energy` | object |  | Energy report: emitted, detected, absorbed, escaped, balance_residual, … (docs/STRAY.md). |
| `map_classes` | array<string> |  | Class of each row of class_irradiance, in path-class order. |
| `partition` | object? |  |  |
| `partition.apertures` | array<object> |  | The shared clipping per surface. |
| `partition.covered_ghost_flux_mc` | number | W | Monte-Carlo flux of the ghost signatures the ghost run covers (its counterpart is the ghost solver's total). |
| `partition.covered_ghost_std_error` | number | W | Exact per-source-ray Monte-Carlo standard error. |
| `partition.excluded_class` | string |  | Always `ghost`. |
| `partition.ghost_class_flux_mc` | number | W | Monte-Carlo estimate of the excluded ghost class (comparison only). |
| `partition.ghost_class_std_error` | number | W |  |
| `partition.ghost_paths` | integer |  | ≥ 0; Paths in the ghost run's coverage. |
| `partition.same_apertures` | boolean |  | GhostCoverage::same_apertures held (always true: a mismatch is an error). |
| `partition.uncovered_ghost_flux` | number | W | Ghost-class flux outside the ghost run's coverage, included in stray_flux and the maps (CONVENTIONS §10). |
| `partition.uncovered_ghost_signatures` | integer |  | ≥ 0; Number of handed-back ghost signatures. |
| `partition.uncovered_ghost_std_error` | number | W | Exact per-source-ray Monte-Carlo standard error. |
| `primary_obstruction` | nf? |  |  |
| `source` | object |  | Resolved source beam: spec, reference_flux, description. |
| `stray_flux` | number | W | Stray-owned flux, including uncovered ghost paths. |
| `stray_fraction` | number? |  |  |
| `stray_std_error` | number | W | Exact per-source-ray Monte-Carlo standard error, including uncovered ghosts. |
| `top_signatures` | array<object> |  | Strongest (class, signature) pairs among the included classes. |
| `totals` | array<object> |  | Every class, included or not. |
| `totals[].class` | string |  | one of `"primary"`, `"ghost"`, `"scatter"`, `"mechanical"`, `"sneak"` |
| `totals[].flux` | number | W |  |
| `totals[].fraction` | number? |  |  |
| `totals[].hits` | integer |  | ≥ 0; Detector hits. |
| `totals[].included` | boolean |  | Whole class selected; uncovered ghosts may be included separately. |
| `totals[].moments` | object |  | Image moments: flux, centroid, centroid_std_error, rms_radius, rms_radius_std_error (detector local mm). |
| `totals[].std_error` | number | W | Monte-Carlo standard error. |
| `warnings` | array<string> |  |  |

Buffers:

| name | dtype | shape | unit | notes |
|---|---|---|---|---|
| `class_irradiance` | f64 | [C, ny, nx] | W/mm² | Irradiance per owned class (rows of map_classes); under ghost_partition the Ghost row contains only uncovered paths. Row 0 at y = −half_y, x fastest. |
| `class_irradiance_std_error` | f64 | [C, ny, nx] | W/mm² | Standard error per pixel. |
| `total_irradiance` | f64 | [ny, nx] | W/mm² | Stray total. |
| `total_irradiance_std_error` | f64 | [ny, nx] | W/mm² |  |

### `diffraction_encircled_energy` — Diffraction encircled energy

Category `wave`, result version 1.

Encircled and ensquared energy of the diffraction PSF, with the diffraction-limited reference.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `center` | object | `{"type":"centroid"}` |  | Tagged energy centre: {type: reference}, {type: centroid}, {type: peak}, or {type: point, x, y} in mm. |
| `configuration` | integer | `0` |  | ≥ 0; System configuration index; 0 also selects the base system when there are no configurations. |
| `fields` | object | `{"type":"all"}` |  | System field selection: all, index or list (tagged by type). |
| `grid_size` | integer | `256` |  | ≥ 1; Image samples per axis; Huygens is limited to 1024. |
| `max_radius_mm` | number\|null | `null` | mm | > 0.0; Positive maximum radius; null uses the PSF window. |
| `method` | string | `"fft"` |  | one of `"fft"`, `"huygens"` |
| `padding` | number | `4.0` |  | > 0.0; Positive factor: image spacing = smallest wavelength × working F/# / padding. |
| `pupil` | object | `{"amplitude":"transmission","grid":64,"rays":65,"reference":"chief","rim":true,"rim_tolerance":1e-10,"supersample":4}` |  |  |
| `samples` | integer | `51` |  | ≥ 2 |
| `wavelengths` | object | `{"type":"all"}` |  | System wavelength selection: all, primary, index or list (tagged by type). |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `fields` | array<object> |  | Usable selected fields. |
| `fields[].center_mm` | [nf; 2] |  | Energy centre relative to the common reference, mm. |
| `fields[].diffraction_limit` | string | 1 | Buffer name; aberration-free encircled energy. |
| `fields[].encircled` | string | 1 | Buffer name; energy inside a circle of radius radius_mm. |
| `fields[].ensquared` | string | 1 | Buffer name; energy inside a square of half-width radius_mm. |
| `fields[].field` | integer |  | ≥ 0; System field index. |
| `fields[].field_value` | [nf; 2] |  | Field [x, y] in the system's units (degrees or mm). |
| `fields[].r50_mm` | nf | mm | Radius enclosing 50 % of total energy. |
| `fields[].r80_mm` | nf | mm | Radius enclosing 80 % of total energy. |
| `fields[].r90_mm` | nf | mm | Radius enclosing 90 % of total energy. |
| `radius_mm` | string | mm | Buffer name; radii / square half-widths. |

Buffers:

| name | dtype | shape | unit | notes |
|---|---|---|---|---|
| `radius_mm` | f64 | [radii] | mm | Circle radii / square half-widths. |
| `encircled_*` | f64 | [radii] | 1 | Encircled fraction of total PSF energy. |
| `ensquared_*` | f64 | [radii] | 1 | Ensquared fraction of total PSF energy. |
| `diffraction_limit_*` | f64 | [radii] | 1 | Aberration-free encircled energy. |

### `line_spread` — Line spread function

Category `wave`, result version 1.

Diffraction line-spread functions along the image axes.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `configuration` | integer | `0` |  | ≥ 0; System configuration index; 0 also selects the base system when there are no configurations. |
| `fields` | object | `{"type":"all"}` |  | System field selection: all, index or list (tagged by type). |
| `grid_size` | integer | `256` |  | ≥ 1; Image samples per axis; Huygens is limited to 1024. |
| `method` | string | `"fft"` |  | one of `"fft"`, `"huygens"` |
| `padding` | number | `4.0` |  | > 0.0; Positive factor: image spacing = smallest wavelength × working F/# / padding. |
| `pupil` | object | `{"amplitude":"transmission","grid":64,"rays":65,"reference":"chief","rim":true,"rim_tolerance":1e-10,"supersample":4}` |  |  |
| `wavelengths` | object | `{"type":"all"}` |  | System wavelength selection: all, primary, index or list (tagged by type). |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `fields` | array<object> |  | Usable selected fields. |
| `fields[].field` | integer |  | ≥ 0; System field index. |
| `fields[].field_value` | [nf; 2] |  | Field [x, y] in the system's units (degrees or mm). |
| `fields[].positions_mm` | string | mm | Buffer name; positions relative to the common reference. |
| `fields[].x` | string | mm^-1 | Buffer name; integral of the unit-energy PSF over y. |
| `fields[].y` | string | mm^-1 | Buffer name; integral of the unit-energy PSF over x. |

Buffers:

| name | dtype | shape | unit | notes |
|---|---|---|---|---|
| `positions_mm_*` | f64 | [positions] | mm | Coordinates relative to the common reference. |
| `x_*` | f64 | [positions] | mm^-1 | Integral of PSF over y. |
| `y_*` | f64 | [positions] | mm^-1 | Integral of PSF over x. |

### `mtf` — MTF

Category `wave`, result version 1.

Diffraction MTF (FFT, pupil autocorrelation or from the Huygens PSF) vs frequency, tangential and sagittal, with the diffraction limit and an optional detector (pixel) MTF.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `configuration` | integer | `0` |  | ≥ 0; System configuration index; 0 also selects the base system when there are no configurations. |
| `detector` | object\|null | `null` |  | Optional detector pixel-aperture and diffusion MTF; null gives optical MTF only. |
| `fields` | object | `{"type":"all"}` |  | System field selection: all, index or list (tagged by type). |
| `frequency_samples` | integer | `41` |  | ≥ 2 |
| `huygens` | object | `{"grid_size":128,"padding":4.0}` |  |  |
| `max_frequency` | number\|null | `null` | cycles/mm | > 0.0; Positive maximum image-space frequency; null uses the largest optical cutoff. |
| `method` | string | `"direct"` |  | one of `"direct"`, `"fft"`, `"huygens"` |
| `pupil` | object | `{"amplitude":"transmission","grid":64,"rays":65,"reference":"chief","rim":true,"rim_tolerance":1e-10,"supersample":4}` |  |  |
| `wavelengths` | object | `{"type":"all"}` |  | System wavelength selection: all, primary, index or list (tagged by type). |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `fields` | array<object> |  | Usable selected fields. |
| `fields[].cutoff` | number | cycles/mm | Largest image-space cutoff over the selected wavelengths. |
| `fields[].detector` | object? |  |  |
| `fields[].detector.nyquist_sagittal` | nf | cycles/mm | Detector Nyquist marker along the sagittal direction. |
| `fields[].detector.nyquist_tangential` | nf | cycles/mm | Detector Nyquist marker along the tangential direction. |
| `fields[].detector.sagittal` | string | 1 | Buffer name; detector sagittal MTF. |
| `fields[].detector.tangential` | string | 1 | Buffer name; detector tangential MTF. |
| `fields[].diffraction_limit` | string | 1 | Buffer name; aberration-free MTF of the same pupil. |
| `fields[].field` | integer |  | ≥ 0; System field index. |
| `fields[].field_value` | [nf; 2] |  | Field [x, y] in the system's units (degrees or mm). |
| `fields[].sagittal` | string | 1 | Buffer name; optical sagittal MTF. |
| `fields[].system_sagittal` | string? |  |  |
| `fields[].system_tangential` | string? |  |  |
| `fields[].tangential` | string | 1 | Buffer name; optical tangential MTF. |
| `fields[].tangential_direction` | [nf; 2] |  | Tangential unit vector in image x/y; sagittal is perpendicular. |
| `frequencies` | string | cycles/mm | Buffer name; image-space spatial frequencies. |
| `nyquist` | [nf; 2]? |  |  |

Buffers:

| name | dtype | shape | unit | notes |
|---|---|---|---|---|
| `tangential_*` | f64 | [frequency_samples] | 1 | MTF, normalised to one at zero frequency; NaN for unusable field samples. |
| `sagittal_*` | f64 | [frequency_samples] | 1 | MTF, normalised to one at zero frequency; NaN for unusable field samples. |
| `diffraction_limit_*` | f64 | [frequency_samples] | 1 | MTF along the frequencies axis; detector/system buffers only when a detector is supplied. |
| `detector_tangential_*` | f64 | [frequency_samples] | 1 | MTF along the frequencies axis; detector/system buffers only when a detector is supplied. |
| `detector_sagittal_*` | f64 | [frequency_samples] | 1 | MTF along the frequencies axis; detector/system buffers only when a detector is supplied. |
| `system_tangential_*` | f64 | [frequency_samples] | 1 | MTF along the frequencies axis; detector/system buffers only when a detector is supplied. |
| `system_sagittal_*` | f64 | [frequency_samples] | 1 | MTF along the frequencies axis; detector/system buffers only when a detector is supplied. |
| `frequencies` | f64 | [frequency_samples] | cycles/mm | Image-space spatial frequencies. |

### `mtf_vs_field` — MTF vs field

Category `wave`, result version 1.

Diffraction MTF at fixed frequencies across the field.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `configuration` | integer | `0` |  | ≥ 0; System configuration index; 0 also selects the base system when there are no configurations. |
| `detector` | object\|null | `null` |  | Optional detector pixel-aperture and diffusion MTF; null gives optical MTF only. |
| `direction` | [number; 2] | `[0.0,1.0]` |  | Nonzero direction in normalised field coordinates. |
| `field_samples` | integer | `11` |  | 1 … 1001 |
| `frequencies` | array<number> | `[]` | cycles/mm | Image-space frequencies; empty selects cutoff × (1/8, 1/4, 1/2). |
| `huygens` | object | `{"grid_size":128,"padding":4.0}` |  |  |
| `method` | string | `"direct"` |  | one of `"direct"`, `"fft"`, `"huygens"` |
| `pupil` | object | `{"amplitude":"transmission","grid":64,"rays":65,"reference":"chief","rim":true,"rim_tolerance":1e-10,"supersample":4}` |  |  |
| `wavelengths` | object | `{"type":"all"}` |  | System wavelength selection: all, primary, index or list (tagged by type). |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `curves` | array<object> |  | One curve per frequency; includes the detector when supplied. |
| `curves[].frequency` | number | cycles/mm | Image-space frequency. |
| `curves[].sagittal` | string | 1 | Buffer name; MTF perpendicular to the local field direction. |
| `curves[].tangential` | string | 1 | Buffer name; MTF along the local field direction. |
| `field_values` | string | degrees or mm | Buffer name; specified [x, y] of each field sample. |
| `frequencies` | string | cycles/mm | Buffer name; image-space frequencies. |
| `normalized_field` | string | 1 | Buffer name; normalised radial field from axis to edge. |

Buffers:

| name | dtype | shape | unit | notes |
|---|---|---|---|---|
| `tangential_*` | f64 | [field_samples] | 1 | MTF, normalised to one at zero frequency; NaN for unusable field samples. |
| `sagittal_*` | f64 | [field_samples] | 1 | MTF, normalised to one at zero frequency; NaN for unusable field samples. |
| `normalized_field` | f64 | [field_samples] | 1 | Normalised radial field. |
| `field_values` | f64 | [field_samples, 2] | degrees or mm | Specified field [x, y]. |
| `frequencies` | f64 | [frequencies] | cycles/mm | Image-space frequencies. |

### `psf` — Point spread function

Category `wave`, result version 1.

Diffraction PSF by FFT or Huygens integration, monochromatic or polychromatic, with the Strehl ratio.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `configuration` | integer | `0` |  | ≥ 0; System configuration index; 0 also selects the base system when there are no configurations. |
| `fields` | object | `{"type":"all"}` |  | System field selection: all, index or list (tagged by type). |
| `grid_size` | integer | `128` |  | ≥ 1; Image samples per axis; Huygens is limited to 1024. |
| `method` | string | `"fft"` |  | one of `"fft"`, `"huygens"` |
| `normalization` | string | `"strehl"` |  | one of `"strehl"`, `"unit_energy"`; strehl: ideal peak = 1; unit_energy: integral = 1, in mm^-2. |
| `padding` | number | `4.0` |  | > 0.0; Positive factor: image spacing = smallest wavelength × working F/# / padding. |
| `polychromatic` | boolean | `true` |  |  |
| `pupil` | object | `{"amplitude":"transmission","grid":64,"rays":65,"reference":"chief","rim":true,"rim_tolerance":1e-10,"supersample":4}` |  |  |
| `spacing_mm` | number\|null | `null` | mm | > 0.0; Positive explicit spacing; null uses padding. |
| `wavelengths` | object | `{"type":"all"}` |  | System wavelength selection: all, primary, index or list (tagged by type). |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `psfs` | array<object> |  | One PSF per usable field (and wavelength when monochromatic). |
| `psfs[].centroid_mm` | [nf; 2] |  | Centroid relative to reference_mm, mm. |
| `psfs[].field` | integer |  | ≥ 0; System field index. |
| `psfs[].field_value` | [nf; 2] |  | Field [x, y] in the system's units (degrees or mm). |
| `psfs[].peak_mm` | [nf; 2] |  | Peak relative to reference_mm, mm. |
| `psfs[].psf` | object |  |  |
| `psfs[].psf.center_mm` | [nf; 2] |  | Grid centre relative to reference_mm, mm; coordinate i is center + (i - floor(size/2)) × spacing. |
| `psfs[].psf.data` | string | 1 or mm^-2 | Buffer name; Strehl-normalised intensity (ideal peak 1), or unit-energy density. |
| `psfs[].psf.diffraction_peak` | number | 1 or mm^-2 | Aberration-free peak in the selected normalisation. |
| `psfs[].psf.normalization` | string |  | one of `"strehl"`, `"unit_energy"` |
| `psfs[].psf.size` | integer |  | ≥ 0; Image samples per axis. |
| `psfs[].psf.spacing_mm` | number | mm | Image-plane sample spacing. |
| `psfs[].psf.total_energy` | number | mm^2 or 1 | Integral over the infinite image plane in the selected normalisation. |
| `psfs[].psf.warnings` | array<any> |  | Also carried as text in RunInfo warnings. |
| `psfs[].psf.wavelength_um` | number? |  |  |
| `psfs[].reference_mm` | [nf; 2] |  | Common image-local reference point, mm. |
| `psfs[].strehl` | number | 1 | Peak / aberration-free peak for the same pupil and spectral weights. |
| `psfs[].wavelength` | integer? |  |  |

Buffers:

| name | dtype | shape | unit | notes |
|---|---|---|---|---|
| `data_*` | f64 | [image_y, image_x] | 1 or mm^-2 | Strehl-normalised intensity (ideal peak 1), or unit-energy density; x fastest. |

### `through_focus_mtf` — Through-focus MTF

Category `wave`, result version 1.

Diffraction MTF at fixed frequencies vs image-plane defocus.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `configuration` | integer | `0` |  | ≥ 0; System configuration index; 0 also selects the base system when there are no configurations. |
| `detector` | object\|null | `null` |  | Optional detector pixel-aperture and diffusion MTF; null gives optical MTF only. |
| `fields` | object | `{"type":"all"}` |  | System field selection: all, index or list (tagged by type). |
| `focus_range_mm` | number\|null | `null` | mm | > 0.0; Positive half-range; null uses 4 wavelength × F/# squared. |
| `focus_samples` | integer | `21` |  | 1 … 1001 |
| `frequencies` | array<number> | `[]` | cycles/mm | Image-space frequencies; empty selects cutoff × (1/8, 1/4, 1/2). |
| `huygens` | object | `{"grid_size":128,"padding":4.0}` |  |  |
| `method` | string | `"direct"` |  | one of `"direct"`, `"fft"`, `"huygens"` |
| `pupil` | object | `{"amplitude":"transmission","grid":64,"rays":65,"reference":"chief","rim":true,"rim_tolerance":1e-10,"supersample":4}` |  |  |
| `wavelengths` | object | `{"type":"all"}` |  | System wavelength selection: all, primary, index or list (tagged by type). |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `fields` | array<object> |  | Usable selected fields. |
| `fields[].curves` | array<object> |  | One curve per frequency; includes the detector when supplied. |
| `fields[].curves[].frequency` | number | cycles/mm | Image-space frequency. |
| `fields[].curves[].sagittal` | string | 1 | Buffer name; MTF perpendicular to the local field direction. |
| `fields[].curves[].tangential` | string | 1 | Buffer name; MTF along the local field direction. |
| `fields[].field` | integer |  | ≥ 0; System field index. |
| `fields[].field_value` | [nf; 2] |  | Field [x, y] in the system's units (degrees or mm). |
| `focus_mm` | string | mm | Buffer name; image-plane shift, positive along propagation. |
| `frequencies` | string | cycles/mm | Buffer name; image-space frequencies. |

Buffers:

| name | dtype | shape | unit | notes |
|---|---|---|---|---|
| `tangential_*` | f64 | [focus_samples] | 1 | MTF, normalised to one at zero frequency; NaN for unusable field samples. |
| `sagittal_*` | f64 | [focus_samples] | 1 | MTF, normalised to one at zero frequency; NaN for unusable field samples. |
| `focus_mm` | f64 | [focus_samples] | mm | Image-plane shift; positive along propagation. |
| `frequencies` | f64 | [frequencies] | cycles/mm | Image-space frequencies. |

### `wavefront_map` — Wavefront map

Category `wave`, result version 1.

OPD on the exit-pupil reference sphere from the reference tracer, with RMS/PV, Strehl estimate and Zernike fit.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `configuration` | integer | `0` |  | ≥ 0; System configuration index; 0 also selects the base system when there are no configurations. |
| `fields` | object | `{"type":"all"}` |  | System field selection: all, index or list (tagged by type). |
| `pupil` | object | `{"amplitude":"transmission","grid":64,"rays":65,"reference":"chief","rim":true,"rim_tolerance":1e-10,"supersample":4}` |  |  |
| `wavelengths` | object | `{"type":"primary"}` |  | System wavelength selection: all, primary, index or list (tagged by type). |
| `zernike_ordering` | string | `"fringe"` |  | one of `"fringe"`, `"standard"`; Fringe Z1 onward by default; standard uses Noll normalisation. |
| `zernike_terms` | integer | `37` |  | ≥ 1; Number of fitted terms; Fringe supports at most 37. |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `maps` | array<object> |  | One map per usable field and wavelength. |
| `maps[].amplitude` | string | 1 | Buffer name; relative field amplitude, zero outside the pupil. |
| `maps[].coordinates` | string | 1 | Buffer name; cell centres over [-1, 1], shared by u and v. |
| `maps[].exit_pupil` | object |  |  |
| `maps[].exit_pupil.chief_direction` | [number; 3] |  | Chief direction, image-local. |
| `maps[].exit_pupil.chief_point_mm` | [nf; 3] |  | Chief image point, image-local mm. |
| `maps[].exit_pupil.clear_area` | number | 1 | Clear area in normalised pupil coordinates. |
| `maps[].exit_pupil.effective_f_number` | number | 1 | Working F/# including the clear-area scale. |
| `maps[].exit_pupil.exit_pupil_point_mm` | [nf; 3]? |  |  |
| `maps[].exit_pupil.image_index` | number | 1 | Image-space refractive index. |
| `maps[].exit_pupil.lattice_rays` | integer |  | ≥ 0 |
| `maps[].exit_pupil.numerical_aperture` | number | 1 | Image index × sin_max. |
| `maps[].exit_pupil.paraxial_position_mm` | nf | mm | Paraxial exit-pupil position from the image. |
| `maps[].exit_pupil.paraxial_radius_mm` | nf | mm | Paraxial exit-pupil radius. |
| `maps[].exit_pupil.propagation_sign` | number | 1 | Sign of propagation along image-local z. |
| `maps[].exit_pupil.reference` | string |  | one of `"chief"`, `"centroid"` |
| `maps[].exit_pupil.reference_point_mm` | [nf; 3] |  | Reference-sphere centre, image-local mm. |
| `maps[].exit_pupil.reference_radius_mm` | nf | mm | Reference-sphere radius; infinity for telecentric beams. |
| `maps[].exit_pupil.rim_points` | integer |  | ≥ 0 |
| `maps[].exit_pupil.sin_max` | number | 1 | Pupil angular scale. |
| `maps[].exit_pupil.surviving_rays` | integer |  | ≥ 0 |
| `maps[].exit_pupil.telecentric` | boolean |  | Exit pupil at infinity. |
| `maps[].exit_pupil.transmitted_fraction` | number | 1 | Surviving entrance-pupil area fraction. |
| `maps[].exit_pupil.wavelength_um` | number | µm |  |
| `maps[].field` | integer |  | ≥ 0; System field index. |
| `maps[].field_value` | [nf; 2] |  | Field [x, y] in the system's units (degrees or mm). |
| `maps[].grid` | integer |  | ≥ 0; Samples per pupil axis. |
| `maps[].opd_waves` | string | waves | Buffer name; OPL_chief minus OPL_ray at the reference sphere; NaN outside the pupil. |
| `maps[].pv_waves` | number | waves | Peak-to-valley OPD. |
| `maps[].pv_waves_tilt_removed` | number | waves | Peak-to-valley with best-fit tilt removed. |
| `maps[].rim` | array<string> |  | Clear-aperture loops; empty when rim tracing is disabled. |
| `maps[].rms_waves` | number | waves | Weighted RMS with piston removed. |
| `maps[].rms_waves_tilt_removed` | number | waves | Weighted RMS with piston and best-fit tilt removed. |
| `maps[].seidel` | object |  |  |
| `maps[].seidel.astigmatism` | number | waves |  |
| `maps[].seidel.astigmatism_angle_deg` | number | degrees | From +x toward +y. |
| `maps[].seidel.coma` | number | waves |  |
| `maps[].seidel.coma_angle_deg` | number | degrees | From +x toward +y. |
| `maps[].seidel.focus` | number | waves |  |
| `maps[].seidel.piston` | number | waves |  |
| `maps[].seidel.spherical` | number | waves | Pupil-edge Seidel-equivalent term. |
| `maps[].seidel.tilt` | number | waves |  |
| `maps[].seidel.tilt_angle_deg` | number | degrees | From +x toward +y. |
| `maps[].strehl_at_reference` | number | 1 | Sampled-pupil Strehl at the reference point. |
| `maps[].strehl_marechal` | number | 1 | exp(-(2π RMS_tilt_removed)^2). |
| `maps[].wavelength` | integer |  | ≥ 0; System wavelength index. |
| `maps[].wavelength_um` | number | µm |  |
| `maps[].zernike` | object |  |  |
| `maps[].zernike.coefficients` | string | waves | Buffer name; Z1 onward, fitted to lattice rays with amplitude-squared weights. |
| `maps[].zernike.ordering` | string |  | one of `"fringe"`, `"standard"`; Fringe by default. |
| `maps[].zernike_condition` | number | 1 | Fit condition estimate. |
| `maps[].zernike_residual_rms` | number | waves | Weighted fit residual RMS. |

Buffers:

| name | dtype | shape | unit | notes |
|---|---|---|---|---|
| `coordinates_*` | f64 | [grid] | 1 | Cell-centred normalised pupil coordinates. |
| `opd_waves_*` | f64 | [grid_v, grid_u] | waves | OPL_chief minus OPL_ray; NaN outside the clear pupil. u along image x, v along image y. |
| `amplitude_*` | f64 | [grid_v, grid_u] | 1 | Relative field amplitude; zero outside the clear pupil. |
| `coefficients_*` | f64 | [zernike_terms] | waves | Z1 onward in the reported ordering; Fringe by default. |
| `rim_*` | f64 | [points, 2] | 1 | Clear-aperture loop vertices [u, v]. |

### `optimize` — Optimize

Category `optimization`, result version 1, **GPU path**.

Optimises the variables against a merit function (explicit, or the default RMS spot / wavefront merit with thickness boundaries) with explicit constraints, by Levenberg–Marquardt, dogleg, differential evolution, CMA-ES, Nelder–Mead or basin hopping. Returns the optimised system, the merit history and the operand table.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `algorithm` | string | `"lm"` |  | one of `"lm"`, `"dogleg"`, `"de"`, `"cma_es"`, `"nelder_mead"`, `"hammer"` |
| `backend` | string | `"auto"` |  | one of `"auto"`, `"cpu"`, `"gpu"`, `"gpu_emulation"`; auto: the engine's GPU device when the run is past the measured GPU/CPU crossover (rays per candidate and batch size, see backend.workload), else the CPU reference. |
| `constraints` | array | `[]` |  | Operands appended as explicit constraints (role forced to constraint). |
| `default_merit` | object | `{"auto_symmetry":true,"configs":[],"criterion":{"reference":"centroid","scope":"monochromatic","type":"rms_spot"},"fields":null,"sampling":{"arm_offset_deg":0.0,"arms":6,"obscuration":0.0,"rings":3,"symmetry":"none"},"thickness":{"air_max_centre":null,"air_min_centre":0.1,"air_min_edge":0.1,"as_constraints":true,"auto_semi_diameters":false,"glass_max_centre":30.0,"glass_min_centre":1.0,"glass_min_edge":1.0,"weight":1.0},"wavelengths":null,"weight":1.0}` |  | Default-merit generator settings, laid over the defaults (criterion, pupil sampling, fields, wavelengths, thickness boundaries). |
| `fd_relative_step` | number\|null | `null` |  | > 0; null chooses per operand and backend (wavefront 1e-6/1e-4 on the CPU, GPU f32 steps; docs/OPTIMIZATION.md). |
| `fd_scheme` | string | `"forward"` |  | one of `"forward"`, `"central"` |
| `feasibility_tolerance` | number | `1e-6` |  | ≥ 0; Total normalised constraint violation regarded as feasible (best-so-far ranking, `final.feasible`). |
| `gpu_batch` | integer | `512` |  | ≥ 1; Candidates per GPU dispatch. |
| `hold_efl` | boolean | `true` |  | Hold the EFL at its start value unless the merit already has an efl operand: a constraint for lm, dogleg and hammer, a penalty residual for de, cma_es and nelder_mead. |
| `max_iterations` | integer\|null | `null` |  | ≥ 1; Iterations (lm, dogleg, nelder_mead), generations (de, cma_es) or rounds (hammer); null: 200, 200, 2000, 100, 200, 20. |
| `merit` | object\|null | `null` |  | A merit-function spec; null generates the default merit from `default_merit` for the variables. |
| `operands` | array | `[]` |  | Operands appended to the merit (e.g. an EFL target). |
| `options` | object | `{}` |  | Settings of the algorithm, laid over its defaults. |
| `seed` | integer | `0` |  | ≥ 0 |
| `stop_key` | string\|null | `null` |  | Key for `request_stop`: stop and keep the best point. One running job per key. |
| `variables` | array | `[]` |  | Variable specs: {"param": ParamPath, "lower", "upper", "scale", "fd_step", "label"}. Empty: the curvature of every curved optical surface and the back focus. |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `algorithm` | string |  |  |
| `backend` | object |  |  |
| `backend.evaluated_on` | string |  | Search evaluator backend; envelope uses engine backend ids. |
| `backend.final_qualification` (optional) | object |  | GPU qualification at the optimised system. |
| `backend.gpu_stats` (optional) | object |  |  |
| `backend.gpu_stats.cpu_candidates` | integer |  | ≥ 0 |
| `backend.gpu_stats.device_seconds` | number | s |  |
| `backend.gpu_stats.dispatches` | integer |  | ≥ 0 |
| `backend.gpu_stats.gpu_candidates` | integer |  | ≥ 0 |
| `backend.gpu_stats.rays` | integer |  | ≥ 0 |
| `backend.precision` | string |  | one of `"f64_reference"`, `"f32_gpu"`; Search precision. |
| `backend.qualification` (optional) | object |  | Cached GPU precision qualification for the start snapshot. |
| `backend.reported_on` | string |  | CPU reference for optimize; search evaluator for tolerance. |
| `backend.requested` | string |  | one of `"auto"`, `"cpu"`, `"gpu"`, `"gpu_emulation"`; Merit evaluator preference. |
| `backend.workload` | object |  |  |
| `backend.workload.batch` | integer |  | ≥ 0 |
| `backend.workload.gpu_faster` | boolean |  |  |
| `backend.workload.rays_per_candidate` | integer? |  |  |
| `evaluations` | integer |  | ≥ 0 |
| `failed_variables` | array<integer> |  |  |
| `final` | object |  |  |
| `final.feasible` | boolean |  | Violation within feasibility_tolerance. |
| `final.merit` | nf |  |  |
| `final.status` | object |  | EvalStatus. |
| `final.violation` | nf |  |  |
| `history` | array<object> |  | One record per reported iteration. |
| `history[].best_merit` | nf |  |  |
| `history[].evaluations` | integer |  | ≥ 0 |
| `history[].iteration` | integer |  | ≥ 0 |
| `history[].merit` | nf |  |  |
| `history[].stage` | string |  |  |
| `history[].violation` | nf |  |  |
| `improved` | boolean |  | Better than the start, feasibility first. |
| `improvement` | nf |  | 1 - final/start merit; null for zero start. |
| `iterations` | integer |  | ≥ 0 |
| `merit_spec` | object |  | Effective MeritFunctionSpec, including generated operands. |
| `multipliers` | array<nf>? |  |  |
| `operands` | array<object> |  | Start/final values on the CPU f64 reference; operand units in OPTIMIZATION.md. |
| `operands[].comparison` | string |  | one of `"equal"`, `"less_than"`, `"greater_than"` |
| `operands[].config` | integer |  | ≥ 0 |
| `operands[].contribution` | nf | merit | weight * residual squared; null for constraints/disabled operands. |
| `operands[].contribution_fraction` | nf |  |  |
| `operands[].enabled` | boolean |  |  |
| `operands[].error` | string? |  |  |
| `operands[].field` | object |  | FieldSelector. |
| `operands[].index` | integer |  | ≥ 0 |
| `operands[].kind` | string |  | Operand name. |
| `operands[].label` | string |  |  |
| `operands[].residual` | nf | operand | Signed value - target for equal; one-sided violation otherwise. |
| `operands[].role` | string |  | one of `"residual"`, `"constraint"` |
| `operands[].start` | nf |  |  |
| `operands[].target` | nf |  |  |
| `operands[].value` | nf |  |  |
| `operands[].wavelength` | object |  | WavelengthSelector. |
| `operands[].weight` | nf |  |  |
| `search` | object |  |  |
| `search.merit` | nf |  |  |
| `search.reference_merit` | nf |  |  |
| `search.reference_violation` | nf |  |  |
| `search.violation` | nf |  |  |
| `start` | object |  |  |
| `start.merit` | nf |  |  |
| `start.status` | object |  | EvalStatus. |
| `start.violation` | nf |  |  |
| `status` | string |  | one of `"converged"`, `"finished"`, `"stopped"`, `"cancelled"` |
| `stopped` | boolean |  | Stop requested; best-so-far result retained. |
| `system` | object |  | Optimised OpticalSystem. Apply with Session::set_system / mk_session_set_system_json. |
| `system_hash` | string |  | Hash of the optimised model, not the input snapshot. |
| `termination` | string |  | Optimiser termination reason. |
| `variables` | array<object> |  | Values in each parameter's model units. |
| `variables[].at_bound` | boolean |  |  |
| `variables[].change` | nf | parameter | value - start. |
| `variables[].label` | string |  |  |
| `variables[].lower` | nf |  |  |
| `variables[].param` | object |  | ParamPath; stable surface id. |
| `variables[].start` | nf |  |  |
| `variables[].upper` | nf |  |  |
| `variables[].value` | nf |  |  |

Buffers:

| name | dtype | shape | unit | notes |
|---|---|---|---|---|
| `merit_history` | f64 | [records, 4] | mixed | Columns: iteration, evaluations, merit, best merit (search evaluator). |

### `tolerance` — Tolerance analysis

Category `optimization`, result version 1, **GPU path**.

Sensitivity of a criterion merit (default: the RMS spot merit) to each tolerance at its limits, ranked worst first with the RSS prediction, and a reproducible Monte-Carlo analysis, with compensators re-optimised for every trial.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `backend` | string | `"auto"` |  | one of `"auto"`, `"cpu"`, `"gpu"`, `"gpu_emulation"` |
| `batch` | integer | `16` |  | ≥ 1 |
| `compensator_iterations` | integer | `20` |  | ≥ 1 |
| `compensators` | array | `[]` |  | {"param": ParamPath, "range", "scale"}: re-optimised for every trial. |
| `criterion` | object\|null | `null` |  | A merit-function spec; null generates the default merit for tolerancing from `default_merit`. |
| `default_merit` | object | `{"auto_symmetry":true,"configs":[],"criterion":{"reference":"centroid","scope":"monochromatic","type":"rms_spot"},"fields":null,"sampling":{"arm_offset_deg":0.0,"arms":6,"obscuration":0.0,"rings":3,"symmetry":"none"},"thickness":null,"wavelengths":null,"weight":1.0}` |  |  |
| `gpu_batch` | integer | `512` |  | ≥ 1 |
| `mode` | string | `"both"` |  | one of `"sensitivity"`, `"monte_carlo"`, `"both"` |
| `seed` | integer | `0` |  | ≥ 0 |
| `stop_key` | string\|null | `null` |  |  |
| `threshold` | number\|null | `null` |  | Criterion value for the yield (criterion ≤ threshold). |
| `tolerances` | array | `[]` |  | Tolerance operands: {"param": ParamPath, "minus", "plus", "distribution", "label"}. Empty: ±0.05 mm on every thickness but the back focus and ±0.5 % on every curvature. |
| `trials` | integer | `100` |  | ≥ 1 |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `backend` | object |  |  |
| `backend.evaluated_on` | string |  | Search evaluator backend; envelope uses engine backend ids. |
| `backend.final_qualification` (optional) | object |  | GPU qualification at the optimised system. |
| `backend.gpu_stats` (optional) | object |  |  |
| `backend.gpu_stats.cpu_candidates` | integer |  | ≥ 0 |
| `backend.gpu_stats.device_seconds` | number | s |  |
| `backend.gpu_stats.dispatches` | integer |  | ≥ 0 |
| `backend.gpu_stats.gpu_candidates` | integer |  | ≥ 0 |
| `backend.gpu_stats.rays` | integer |  | ≥ 0 |
| `backend.precision` | string |  | one of `"f64_reference"`, `"f32_gpu"`; Search precision. |
| `backend.qualification` (optional) | object |  | Cached GPU precision qualification for the start snapshot. |
| `backend.reported_on` | string |  | CPU reference for optimize; search evaluator for tolerance. |
| `backend.requested` | string |  | one of `"auto"`, `"cpu"`, `"gpu"`, `"gpu_emulation"`; Merit evaluator preference. |
| `backend.workload` | object |  |  |
| `backend.workload.batch` | integer |  | ≥ 0 |
| `backend.workload.gpu_faster` | boolean |  |  |
| `backend.workload.rays_per_candidate` | integer? |  |  |
| `compensators` | array<object> |  | Model parameter units. |
| `criterion_spec` | object |  | Effective MeritFunctionSpec. |
| `monte_carlo` | object? |  |  |
| `monte_carlo.cancelled` | boolean |  | Statistics of completed trials after interruption. |
| `monte_carlo.compensator_mean` | array<nf> |  |  |
| `monte_carlo.compensator_std` | array<nf> |  |  |
| `monte_carlo.criteria` | array<nf> |  | Criterion per trial; inf for failed trials. |
| `monte_carlo.failures` | integer |  | ≥ 0 |
| `monte_carlo.max` | nf |  |  |
| `monte_carlo.mean` | nf |  |  |
| `monte_carlo.min` | nf |  |  |
| `monte_carlo.percentiles` | array<object> |  |  |
| `monte_carlo.percentiles[].percentile` | number | % |  |
| `monte_carlo.percentiles[].value` | nf |  |  |
| `monte_carlo.seed` | integer |  | ≥ 0; Philox seed. |
| `monte_carlo.std_dev` | nf |  |  |
| `monte_carlo.threshold` | nf |  |  |
| `monte_carlo.trials` | integer |  | ≥ 0; Completed trials. |
| `monte_carlo.yield_fraction` | nf |  | Fraction at or below threshold; null without threshold. |
| `sensitivity` | object? |  |  |
| `sensitivity.cancelled` | boolean |  | Partial sensitivity after interruption. |
| `sensitivity.evaluations` | integer |  | ≥ 0 |
| `sensitivity.nominal` | object |  |  |
| `sensitivity.nominal.compensators` | array<nf> |  | Parameter values. |
| `sensitivity.nominal.criterion` | nf |  |  |
| `sensitivity.nominal.ok` | boolean |  |  |
| `sensitivity.operands` | array<object> |  | Completed operands in declaration order. |
| `sensitivity.operands[].change_minus` | nf | criterion | criterion_minus - nominal. |
| `sensitivity.operands[].change_plus` | nf | criterion | criterion_plus - nominal. |
| `sensitivity.operands[].compensators_minus` | array<nf> |  |  |
| `sensitivity.operands[].compensators_plus` | array<nf> |  |  |
| `sensitivity.operands[].criterion_minus` | nf |  |  |
| `sensitivity.operands[].criterion_plus` | nf |  |  |
| `sensitivity.operands[].index` | integer |  | ≥ 0; Tolerance index. |
| `sensitivity.operands[].label` | string |  |  |
| `sensitivity.operands[].minus` | nf |  |  |
| `sensitivity.operands[].ok` | boolean |  | Both limits evaluated successfully. |
| `sensitivity.operands[].param` | object |  | ParamPath. |
| `sensitivity.operands[].plus` | nf |  |  |
| `sensitivity.operands[].worst` | nf | criterion | Largest absolute criterion change at either limit. |
| `sensitivity.predicted_criterion` | nf | criterion | Nominal + rss_change. |
| `sensitivity.ranking` | array<integer> |  | Worst first. |
| `sensitivity.rss_change` | nf | criterion | Root sum of squared worst changes. |
| `status` | string |  | one of `"complete"`, `"stopped"`, `"cancelled"` |
| `stopped` | boolean |  | Stop requested; completed sensitivity operands/trials retained. |

Buffers:

| name | dtype | shape | unit | notes |
|---|---|---|---|---|
| `monte_carlo_criteria` | f64 | [trials] | criterion | Per completed Monte-Carlo trial; inf for failures. Absent without Monte Carlo. |

### `rng_stream` — Random stream check

Category `diagnostic`, result version 1.

Draws `count` Philox4x32-10 uniforms for (seed, stream) and reports their mean and variance (expected 1/2 and 1/12). Buffer `uniform` [count]. `delay_ms` sleeps after each chunk, to test progress and cancellation.

Settings:

| key | type | default | unit | notes |
|---|---|---|---|---|
| `chunk` | integer | `65536` |  | ≥ 1 |
| `count` | integer | `100000` |  | 1 … 50000000 |
| `delay_ms` | number | `0.0` |  | 0.0 … 1000.0 |
| `seed` | integer | `0` |  | ≥ 0 |
| `stream` | integer | `0` |  | ≥ 0 |

Result (`result` in the envelope):

| field | type | unit | notes |
|---|---|---|---|
| `count` | integer |  | ≥ 0 |
| `first` | array<number> |  | The first (up to) four draws. |
| `max` | number |  |  |
| `mean` | number |  |  |
| `min` | number |  |  |
| `variance` | number |  |  |

Buffers:

| name | dtype | shape | unit | notes |
|---|---|---|---|---|
| `uniform` | f64 | [count] |  | The draws, in order. |

<!-- END GENERATED -->
