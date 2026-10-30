# Optimisation and Tolerancing (`mklight-optim`)

`mklight-optim` holds the optimisation numerics (batched least-squares
problems, Levenberg–Marquardt / dogleg trust regions with bounds and
augmented-Lagrangian constraints, differential evolution, Nelder–Mead,
CMA-ES, basin hopping, tabu search), the serialisable merit-function model
and tolerancing. This document covers how a merit function is evaluated on a
real lens: the tracer-backed operand evaluator, the operand reference, the
default merit function, derivatives, determinism, tolerancing and measured
throughput.

## Pieces

```
MeritFunctionSpec ──┐                      ┌── TraceEvaluator (paraxial, real ray,
VariableSpec[] ─────┼─► OpticalProblem ──► │     wavefront, model + auto edges)
OperandEvaluator ───┘     (Problem)        ├── ModelEvaluator (model only)
                              │            └── MTF / ghost / stray evaluators of
                              ▼                other crates, via CompositeEvaluator
      least_squares, hammer, differential_evolution, cma_es, …,
      sensitivity / monte_carlo (OpticalToleranceEvaluator)
```

* `OpticalProblem` turns a spec, variables (`ParamPath`s with bounds) and an
  evaluator into a batched `Problem`. Residual operands become weighted
  residuals, constraint operands become explicit constraints.
* `TraceEvaluator` (`merit::trace_eval`) computes every paraxial, real-ray and
  wavefront operand with the CPU double-precision reference tracer
  (`mklight-trace`), plus all model-only operands (so one evaluator covers a
  complete optical merit function).
* MTF, ghost, stray-light and plug-in (`Custom`) operands are **not**
  computed here. Their crates provide `OperandEvaluator`s, combined with
  `CompositeEvaluator::new().with(TraceEvaluator::…).with(…)`; each operand
  goes to the first evaluator that supports it.

## The tracer-backed evaluator

For each candidate system (a parameter vector applied to the base system,
then the operand's configuration):

1. `Snapshot::new` (validation + content hash) → `ResolvedSystem::new` against
   the evaluator's shared `Arc<MaterialLibrary>` → `SequentialTracer` (paraxial
   data at every wavelength, pupils, prepared system fields, aim caches).
2. **Plan.** The operand list is scanned for what must be traced: arbitrary
   normalised field points, the chief ray and OPD reference sphere of every
   needed (field, wavelength), and every pupil sampling (ray group or aggregate
   sampling) at a (field, wavelength).
3. **Trace** each planned item once into a per-candidate cache. Within one
   candidate this runs in parallel when the batch is smaller than the rayon
   pool and the candidate traces at least 256 sampling rays (below that the
   synchronisation costs more than it saves).
4. **Read** each operand from the cache. Group references (spot centroids,
   piston/tilt) are computed on first use and shared by every ray operand of
   the group, and single pupil points that are group nodes reuse the group's
   trace (bit-exact match of `(px, py)`).

Batches are evaluated in parallel over candidates. Every value is a pure
function of the candidate and the operand, and results are assembled in a
fixed order, so values are bit-identical whatever the thread count (tested).

Build failures map onto the operand errors that the spec's `FailurePolicy`
handles: an invalid system (validation, unknown glass) gives `Invalid`, and a
paraxial failure gives `RayFailure`. In both cases model operands are still
evaluated. A failed ray gives `RayFailure`. Aggregates (RMS spot, RMS
wavefront, relative illumination) leave failed rays out and fail only when no
ray survives. `OpticalProblem` then applies the policy: a penalty residual
(candidate `Degraded`) or a failed candidate. A failed *constraint* operand
always fails the candidate.

`TraceEvalSettings` controls the tracer settings (ray-aiming override,
intersection settings), whether fixed apertures clip rays (`clip`, default
on), the azimuth count of automatic semi-diameters (16), and the pupil
integration of relative illumination (12 radial samples × 32 azimuths, see
below).

**Run info.** `TraceEvaluator::run(start, analysis, settings, seed, || driver)`
runs an optimisation or tolerancing driver (`least_squares`, `hammer`,
`sensitivity`, `monte_carlo`, …) and returns `RunResult { result, info }`. The
`RunInfo` holds the snapshot hash of the start system, the run's settings, the
seed, the backend `cpu-f64-reference`, f64 precision, the tracer's warnings
and the wall time of the driver. `run_info` builds the same record from a
`RunTimer` for callers that cannot wrap the driver in a closure.

## Conventions

* Units follow CONVENTIONS §1: mm, degrees, OPD in waves at the ray's
  wavelength. Real-ray positions and directions are in the surface's local
  frame unless `global`.
* **Field and wavelength selectors.** `FieldSelector::Index` names a system
  field (with its vignetting factors). `Normalized { hx, hy }` is an arbitrary
  field point without vignetting factors. `All` means every field with positive
  weight. `WavelengthSelector::Primary` is the system's primary wavelength,
  and `All` means every wavelength with positive weight.
* **Averages.** First-order operands at `All` wavelengths are the
  weight-averaged value. `RmsSpot` and `WavefrontRms` average the *squared*
  RMS over fields (and, for the wavefront, over wavelengths) with their weights
  and report the square root. `RmsSpot` at `All` wavelengths is the
  polychromatic spot: all wavelengths' rays, weighted `w_node · w_λ`, about one
  reference.
* **Spot references.** `Chief` uses the chief ray of the field at the
  **primary** wavelength, so lateral colour stays in the merit. `Centroid`
  follows the `RayGroup` contract.
* **Wavefront references.** `Chief` uses the CONVENTIONS §4 reference sphere
  at the operand's wavelength (for `WavefrontRms`, only the piston is
  removed). `Centroid` removes the least-squares piston and tilt of the
  group.
* **RayGroup contract.** Every node of the group's sampling is traced at the
  operand's field and configuration. A `Monochromatic` group uses the
  operand's wavelength. A `Polychromatic` group uses every positive-weight
  wavelength, with weights `w_node · w_λ`. The reference comes from
  `reference::spot_centroid` / `reference::piston_tilt`, honouring
  `PupilSymmetry::Bilateral`, where the half pupil stands for the mirrored full
  pupil, so the x centroid and x tilt are zero by symmetry. Failed rays are
  left out. The reference therefore does not depend on which operands are in
  the spec. The tests check it against the mirrored full pupil to 1e-12.
* **Half pupils on asymmetric candidates.** The half pupil is exact only for
  a candidate that is mirror-symmetric about the meridional plane (no x
  decenters, no tilts about y or z, fields on the y axis) at a field point on
  the y axis. The evaluator checks every candidate. For any other candidate it
  also traces the mirrored nodes `(−px, py)`, each node keeping half its
  weight, and computes references and aggregates over that full set. A
  per-ray operand at half-pupil node `p` then reports the pair value
  `s·√((v(p)² + v(p̄)²)/2)`, where `s` is the sign of the mirror-even part
  (`v(p) − v(p̄)` for x aberrations, `v(p) + v(p̄)` otherwise). This keeps
  `Σ w r²` equal to the full-pupil merit and reduces to `v(p)` for a
  symmetric candidate, at twice the ray cost. Tests on the Cooke triplet with
  x decenter, y and z tilts and off-axis field points check it against the
  mirrored full pupil to 1e-10. Before this check, a half-pupil tolerancing
  criterion overstated x-decenter and y-tilt sensitivities 18–39×.

## Operand reference

`C` = capability, which tells the engine which evaluator to use: `M` model,
`P` paraxial, `R` real ray, `W` wavefront, `T` tolerance, `X` other
evaluator. Selectors: `f` uses the field (`f1` = one field required), `λ`
uses the wavelength (`λ1` = one wavelength required).

| name | C | meaning | units | sel. | conventions |
|---|---|---|---|---|---|
| `efl` | P | effective focal length | mm | λ | `−1/ω'` of the y–nu trace (unit-index system power) |
| `bfl` | P | back focal length, last surface to paraxial focus | mm | λ | signed along the local z |
| `f_number` | P/R | `paraxial_image_space`: `f'/EPD`; `paraxial_working`: `1/(2|ω'_m|)`; `real_working`: `1/(2 n' sin θ')` from the on-axis real marginal ray `(0, 1)` and the chief ray | – | λ | real: angle between the marginal and chief image-space directions |
| `magnification` | P | paraxial lateral magnification | – | λ | 0 for an object at infinity |
| `entrance_pupil_diameter` | P | | mm | λ | |
| `entrance_pupil_position` | P | entrance pupil from the first surface, object space | mm | λ | `Invalid` when telecentric (at infinity) |
| `exit_pupil_position` | P | exit pupil from the image surface | mm | λ | `Invalid` when telecentric |
| `paraxial_ray` | P | marginal or chief paraxial ray `height` (mm), `slope` `u` (–) or `angle` `atan u` (deg) after `surface` (default image) | mm / – / deg | λ | rays scaled by the aperture spec / maximum field |
| `petzval_sum` | P | `Σ c (N' − N)/(N N')` over the surfaces (reduced indices, so mirrors count) | mm⁻¹ | λ | Petzval surface curvature `−N'_image · P`; ideal (paraxial) lenses add 0 (they image plane to plane) |
| `axial_colour` | P | chromatic focal shift: paraxial image distance at `short` minus at `long` | mm | – | negative when blue focuses closer (uncorrected positive lens) |
| `total_track` | M | Σ thicknesses from `from` to `to` | mm | – | signed model thicknesses |
| `centre_thickness` | M | physical gap from `surface` to `to` on axis | mm | – | positive along propagation (also after mirrors) |
| `edge_thickness` | M | physical gap at `height`; by default at the rim of the larger clear aperture, each surface's sag at its **own** semi-diameter (the smaller surface continues flat) | mm | – | automatic semi-diameters from the tracer (below) |
| `min_thickness`, `max_thickness` | M | min / max over the optical gaps of a medium | mm | – | the back-focus gap is excluded unless `to` is given |
| `param_value` | M | any `ParamPath` value | param | – | |
| `real_ray` | R | `x y z` (mm), `l m n` (direction cosines after the surface), `angle_x/angle_y` = `atan2(l or m, |n|)` (deg), `incidence_angle` (deg), `optical_path` (mm from the start of the trace) at `surface` (default image) for pupil `(px, py)` | mm / – / deg | f1 λ1 | local frame unless `global`; clipped per settings |
| `ray_aberration` | R | transverse aberration `ε = ray − reference` along `axis` on the image | mm | f1 λ1 | `chief`: primary-λ chief; `centroid`: group centroid |
| `wavefront_aberration` | W | OPD at `(px, py)` (CONVENTIONS §4 sign) | waves | f1 λ1 | `centroid`: minus the group's piston + tilt |
| `chief_ray_angle` | R | chief-ray angle of incidence on `surface` (default image) | deg | f1 λ1 | unsigned; image: CRA, 0 = telecentric |
| `rms_spot` | R | RMS spot radius, or one component, over `sampling` | mm | f λ | about the centroid or the primary-λ chief |
| `spot_centroid` | R | centroid minus the primary-λ chief ray (full default sampling) | mm | f1 λ | |
| `wavefront_rms` | W | RMS OPD over `sampling` | waves | f λ | piston + tilt removed (`centroid`) or piston only (`chief`) |
| `distortion` | R | `100 (h_real − h_ref)/h_ref` of the real chief ray at λ along the field direction; `f_tan_theta`: `h_ref` = paraxial chief height (`h'/tan θ · tan θ`, or `h'/h · h` for finite objects), `f_theta`: `h'/tan θ · θ` | % | f1 λ1 | 0 on axis; `f_theta` needs an object at infinity |
| `lateral_colour` | R | chief-ray image height at `short` minus at `long`, along the field direction | mm | f1 | `+y` on axis |
| `relative_illumination` | R | image irradiance at the chief ray relative to on axis, Lambertian uniform-radiance object | – | f1 λ | energy conservation, vignetting included (below); geometric (no coating or bulk losses); the on-axis reference has no vignetting factors |
| `tolerance_sensitivity` | T | `√((c(+δ) − c₀)² + (c(−δ) − c₀)²)/√2` for the criterion operand | criterion | as criterion | supported when the criterion is |
| `mtf` | X | MTF at a frequency | – | f λ | MTF evaluator (not here) |
| `ghost_focus`, `ghost_irradiance` | X | ghost focus distance / irradiance | mm / rel. | | ghost evaluator (not here) |
| `stray_light_power` | X | stray-light power in a path class | rel. | | stray-light evaluator (not here) |
| `combine` | – | arithmetic on earlier operands | | | resolved by `OpticalProblem` |
| `custom` | X | plug-in kind registered in the `OperandRegistry` | | | plug-in evaluator |

**Telecentricity** is `chief_ray_angle` at the image (or `real_ray` `angle_y`
of the chief ray, which keeps the sign), and the exit-pupil position.
**Paraxial distortion** comes from the same `distortion` operand: the real
chief ray against the paraxial chief ray.

### Relative illumination

The irradiance at the image of a field point, for an object of uniform
radiance, follows from energy conservation:

```
E ∝ G · |dμ/dh| / |dA'/dh|
```

* `G` is the étendue of the transmitted beam per unit object measure. For an
  object at infinity it is the beam's cross-section perpendicular to its
  direction (mm²). For a finite object it is the projected solid angle of the
  transmitted cone about the object normal, which is the area of its
  direction cosines `(L, M)`. Both are `∬ |∂Q/∂(px, py)| dpx dpy` over the
  part of the unit pupil whose rays reach the image. `Q` is the generated
  ray's position across the beam, or its direction cosines, and the formula
  holds with any ray aiming.
* `dμ/dh` is the solid angle (at infinity) or object area per unit
  normalised field area. `dA'/dh` is the image area per unit normalised field
  area at the chief ray. Both are central differences in `(hx, hy)` with step
  1e-4.

This holds for any lens: real, distorting or vignetting, aplanatic or not.
The previous ratio of image-space solid angles was right only for aplanatic
lenses and gave 1.005 for the unvignetted singlet at 5°, where the correct
value is 0.985.

**Pupil integration.** Along each azimuth, the transmitted radial intervals
are found from 12 equally spaced samples, and every pass/block transition is
refined by bisection to 1e-7. Each interval is integrated by 6-point
Gauss–Legendre. Across azimuths there are two cases:

* The surfaces bounding the intervals are the same on all 32 azimuths. The
  integrand is then smooth and periodic, so the trapezoidal sum is spectrally
  accurate.
* Otherwise, the azimuths where the bounding surface changes are located by
  bisection to 1e-6 rad. These are the corners of a vignetted pupil, or the
  tangents of a pupil region that excludes the centre. Each smooth arc
  between them is integrated by Gauss–Legendre in a cosine-mapped variable,
  which also handles the square-root behaviour at a tangent.

Tests (`tests/trace_regressions.rs`):

| case | result |
|---|---|
| real singlet, stop at the front, against `cos θ · y'_θ(0)² sin θ / (y' y'_θ)` from real chief heights | < 1e-7 |
| ideal lens: cos⁴θ | < 1e-9 |
| ideal lens behind a 9 mm aperture 20 mm in front of the 10 mm pupil, `cos⁴θ · A∩(d tan θ)/A∩(0)` at 10°, 20°, 35° (RI down to 0.078; at 35° the pupil centre is blocked) | < 2e-6 |
| finite object, ideal lens, view factor of the pupil disc | < 1e-8 |
| vignetted Cooke triplet (surfaces 1 and 6 stopped down 15 %): 6 × 16 vs 12 × 32 vs 48 × 256 | agree to 1e-8 (the old dropped-cell grid moved 10 % between 12 × 24 and 96 × 192) |

**Cost.** On the Cooke triplet, one operand (field and axis) takes 0.45 ms
unvignetted and 2.5 ms vignetted.

**Limits.** The integration can miss transmitted or blocked features
narrower than the radial sample spacing. A corner formed by a single
surface's own non-circular aperture (for example a rectangle) is not located,
so near it the azimuthal rule falls back to O(h²).

## Default merit function

`DefaultMeritSpec` emits one `ray_aberration` per pupil node and axis (or one
`wavefront_aberration` per node) for every (configuration, field, wavelength).
The weights are chosen so that `Σ w r²` is the weighted mean of the squared
RMS spot (or wavefront) radius. A test checks this identity against the
aggregate operands to 1e-12. A meridionally symmetric system automatically
uses the half pupil, unless a variable (`generate_for`) can break the
symmetry. For a tolerancing criterion, `generate_for_tolerancing(sys,
&tolerance_spec)` counts the toleranced parameters and compensators as
variables. Every ray operand carries its `RayGroup`, also with a `chief`
reference, so the evaluator knows which half pupil it belongs to. An
asymmetric candidate is still evaluated exactly (see Conventions).

Thickness boundaries go on every physical gap except the back focus.
`ThicknessBoundaries::with_auto_semi_diameters()` (field
`auto_semi_diameters`) also emits edge boundaries on gaps whose surfaces have
automatic semi-diameters. For such operands the `TraceEvaluator` computes the
clear semi-diameters of the candidate with `SequentialTracer::semi_diameters`
(all fields and wavelengths, chief plus pupil rim, fixed apertures clipping),
freezes them (`fix_semi_diameters`, margin 0) and applies the model formula.
Each surface's sag is taken at its own clear semi-diameter, and the smaller
surface continues flat out to the larger rim. Before this change, the edge of
the Double Gauss's cemented L3 was measured at the larger L2 aperture. That
height lies outside the steep interface's sag domain, so the operand was
`Invalid` and every candidate failed. The edges now evaluate. Two outer edges
(L1 and L6) are negative at the sample's own clear apertures, a genuine
violation that the optimiser removes.
The edge height therefore moves with the design during optimisation. The
semi-diameters are computed once per candidate, only when such an operand is
present and some surface is `Auto`. `ModelEvaluator` alone reports these
operands as `Invalid`.

## Derivatives

The optimisers use batched finite differences: one batch of `n` (forward) or
`2n` (central) candidates per Jacobian, evaluated in parallel. The tests
validate them on real optical operands with `deriv::check_jacobian`, which
compares the optimiser's Jacobian (exposed through the analytic hook) with
Richardson-extrapolated central differences. The checks cover ray
aberrations, OPD, EFL, distortion and CRA on the Cooke triplet, over
curvatures, an air space and the back focus. An analytic thick-lens
`∂EFL/∂c` passes the same check, and a perturbed Jacobian is rejected.

**Step size for wavefront operands.** Path lengths of ~100 mm carry an
absolute rounding noise of ~1e-14 mm, which is ~2e-11 waves. The default
steps (`√ε` forward, `∛ε` central, relative to `max(|x|, scale)`) are tuned to
relative noise and are too small for OPD residuals. They give ~1e-4 relative
Jacobian errors. Use `JacobianSettings::relative_step ≈ 1e-6` (forward) or
`1e-4` (central) for wavefront merit functions. Spot (ray-position) operands
are fine with the defaults.

Analytic derivatives are not implemented. The `Problem::jacobian` hook is
where they belong: paraxial operands through differentiated y–nu traces, and
ray operands through ray-transfer matrices.

## Tolerancing

The criterion merit should come from
`DefaultMeritSpec::generate_for_tolerancing`, which selects the full pupil
when a tolerance can break the meridional symmetry. A half-pupil criterion is
still evaluated exactly on asymmetric trials, at twice the ray cost. A test
checks that x-decenter and y-tilt sensitivities equal those of the mirrored
full pupil to 1e-9. Wrap the drivers in `TraceEvaluator::run` to get their
`RunInfo`.

`OpticalToleranceEvaluator` with a `TraceEvaluator` perturbs the nominal
system by the `ToleranceSpec`, re-optimises the compensators (for example the
back focus, within `±range` of nominal) by LM on the criterion merit, and
reports the resulting merit. `sensitivity` evaluates each tolerance at its
limits in one batch and ranks the operands worst first, with the RSS
prediction. `monte_carlo` draws trial `i` from Philox stream `TOLERANCE | i`,
so statistics are bit-reproducible for a seed and any trial can be replayed on
its own. Both are tested on the Cooke triplet with a back-focus compensator.

## End-to-end tests (`tests/trace_optim.rs`)

| test | system | result |
|---|---|---|
| singlet best form | f = 100, F/10, N-BK7, start equi-convex; vars c₁, c₂, back focus; EFL = 100 constraint; on-axis d-light RMS spot | shape factor `q = 0.730`, thin-lens best form `2(n²−1)/(n+2) = 0.740`; merit ÷150; EFL within 1e-6 mm |
| doublet achromatisation | N-BK7/F2, start with R₂ = −60, R₃ = −300 (focal shift > 0.5 mm) | F–C focal shift 2e-10 mm (test: < 1e-6), EFL = 100 ± 1e-6, R₂ → −40.2 mm, R₃ → −375 mm (the sample achromat, bent for minimum spherical aberration, has −42.5 / −410) |
| Cooke triplet | default spot merit (3 fields × 3 λ), 6 curvatures + back focus, EFL constraint, glass/air centre and auto-SD edge bounds | LM (40 iterations) lowers the merit by 35 %; seeded basin hopping is bit-reproducible and does not increase it; all constraints hold |
| tolerancing | Cooke triplet, 9 tolerances, back-focus compensator | ranked sensitivities; the compensator removes > 50 % of an air-space error; Monte Carlo identical for a seed, different for another |
| cancellation | Cooke triplet | LM and basin hopping return `Cancelled` with the best-so-far point (feasibility-ranked) |

## Throughput

Benchmark: `cargo test -p mklight-optim --release --test trace_bench --
--ignored --nocapture`. The system is the Double Gauss f50 F/2 sample (13
surfaces, 3 fields × 3 wavelengths), with 8 curvatures and the back focus as
variables, on an Apple M2 Pro (12 rayon threads) using the CPU f64 reference
tracer. "Rays" counts the pupil samples plus one chief ray per (field, λ).

| merit (half pupil) | residuals | single candidate | batch of 256 | FD Jacobian (9 vars) |
|---|---|---|---|---|
| RMS spot, 3 rings × 6 arms | 324 | 4.6 k evals/s (0.22 ms) | **24 k evals/s** (4.1 M rays/s) | 15 k evals/s |
| RMS spot, 6 rings × 12 arms | 1296 | 2.0 k evals/s (0.51 ms) | 8.6 k evals/s (5.6 M rays/s) | 6.1 k evals/s |
| wavefront, 3 rings × 6 arms | 162 | 5.4 k evals/s (0.19 ms) | 32 k evals/s (5.5 M rays/s) | 17 k evals/s |

These figures come from an otherwise idle machine. The benchmark runs for only
about 0.3 s, so other load shows directly. With other jobs running on the same
machine, the 3 × 6 spot batch measured 16–20 k evals/s.
Symmetric candidates pay nothing for the asymmetry check of half pupils. An
asymmetric candidate traces its half-pupil groups twice.

Each candidate pays ~17 µs for the snapshot hash and ~5 µs for resolving and
setting up the tracer. A reference ray through the Double Gauss costs ~1 µs.
At 3 × 6 sampling the rays dominate. A single candidate is traced
sequentially, because its 171 rays are too few to parallelise, which is why
a one-candidate LM trial step runs at ~5 k evals/s while batches (FD
Jacobians, DE/CMA-ES populations, basin-hopping screens, tolerance trials)
run at the full rate. `ParametricBatch`/`evaluate_parametric` remains the
entry point for a GPU backend that keeps the base system packed and uploads
only the parameter vectors.

## Known gaps

* No analytic Jacobians (see Derivatives).
* Relative illumination is geometric and ignores coating and bulk
  transmission. For its integration limits, see Relative illumination.
* `spot_centroid` uses the default full-pupil sampling (3 rings × 6 arms).
* There is no cross-call cache: a candidate evaluated twice is traced twice.
