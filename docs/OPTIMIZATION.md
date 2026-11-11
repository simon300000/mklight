# Optimisation and Tolerancing (`mklight-optim`)

`mklight-optim` holds the optimisation numerics (batched least-squares
problems, Levenberg–Marquardt / dogleg trust regions with bounds and
augmented-Lagrangian constraints, differential evolution, Nelder–Mead,
CMA-ES, basin hopping, tabu search), the serialisable merit-function model
and tolerancing. This document covers how a merit function is evaluated on a
real lens: the tracer-backed operand evaluator, the GPU-batched evaluator,
the operand reference, the default merit function, derivatives, determinism,
tolerancing, measured throughput, and the engine's `optimize`,
`merit_evaluation`, `glass_substitution` and `tolerance` jobs.

## Pieces

```
MeritFunctionSpec ──┐                      ┌── TraceEvaluator (paraxial, real ray,
VariableSpec[] ─────┼─► OpticalProblem ──► │     wavefront, model + auto edges)
OperandEvaluator ───┘     (Problem)        ├── ResidentEvaluator (ray operands of a
                              │            │     batch traced on a mklight-gpu backend)
                              │            ├── ModelEvaluator (model only)
                              │            └── MTF / ghost / stray evaluators of
                              ▼                other crates, via CompositeEvaluator
      least_squares, hammer, differential_evolution, cma_es, …,
      sensitivity / monte_carlo (OpticalToleranceEvaluator)
                              │
                              ▼
      engine analyses `optimize`, `tolerance` (mklight-engine analysis::optim)
```

* `OpticalProblem` turns a spec, variables (`ParamPath`s with bounds) and an
  evaluator into a batched `Problem`. Residual operands become weighted
  residuals, constraint operands become explicit constraints.
* `TraceEvaluator` (`merit::trace_eval`) computes every paraxial, real-ray and
  wavefront operand with the CPU double-precision reference tracer
  (`mklight-trace`), plus all model-only operands (so one evaluator covers a
  complete optical merit function).
* `ResidentEvaluator` (`gpu`) traces the ray-based operands of a batch of
  candidates on a `mklight-gpu` backend and evaluates everything else with
  its `BuiltinEvaluator` reference (see GPU-batched evaluation).
* `BuiltinEvaluator` combines `TraceEvaluator`, `MtfEvaluator` and
  `SolverEvaluator`. Engine optimise and tolerance jobs use this full set,
  also after GPU ray evaluation. `Custom` operands still need a plug-in.
  `CompositeEvaluator` routes each operand to its first supporting evaluator.
* `WithSolves` (`merit::solved`) wraps any of them and applies the solves of
  every candidate system first (Tolerancing, "Solves").

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
   candidate this runs in parallel when the candidate traces at least 256
   sampling rays (below that the synchronisation costs more than it saves):
   field points, then chief rays per (field, wavelength), then the pupil
   samplings. A candidate evaluated alone or in a batch smaller than the
   pool (a trial point, the anchor of a finite difference) traces its
   samplings in runs of 8 nodes (one trace packet) rather than whole
   groups, because the unaimable rim rays of the outer fields (~140 µs each
   against ~6 µs for a healthy ray) would otherwise leave the pool waiting
   on a few groups; in a batch at least as large as the pool, where the
   candidates fill it, whole groups aim and trace together in full packets
   (the runs cost ~15 % more CPU there). The automatic semi-diameters that
   edge-thickness and surface operands need are traced alongside
   (`rayon::join`). Every ray is traced exactly as in its whole group
   (`trace_many` is bitwise equal to tracing one by one) and put back in
   node order, so the values are bit-identical to the serial ones (tested).
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
| `efl_x`, `efl_y` | P | axis-specific effective focal length from `ParaxialXY` | mm | λ | signed unit-index referenced focal lengths; `efl_y` equals legacy `efl` |
| `bfl_x`, `bfl_y` | P | axis-specific back focal length from `ParaxialXY` | mm | λ | signed axial distance from the last optical surface; `bfl_y` equals legacy `bfl` |
| `anamorphic_squeeze` | P | `EFL_y/EFL_x` from `ParaxialXY` | – | λ | signed; at `all` wavelengths averages the ratios, not the two focal lengths |
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
| `doe_min_zone_period` | M | `2π / max |∇surface Φ|` on `surface` | mm | – | positive order-1 physical groove period; infinite for constant phase |
| `doe_max_phase_slope` | M | `max |∇surface Φ|` on `surface` | rad/mm | – | nonnegative metric tangent gradient of the unwrapped phase lag |
| `doe_total_phase` | M | `max Φ − min Φ` on `surface` | rad | – | nonnegative peak-to-valley unwrapped order-1 phase lag |
| `q_max_departure` | M | maximum absolute QCon/QBfs sag departure from an area least-squares sphere | mm | – | nonnegative; sphere curvature and axial piston are fitted over the clear aperture |
| `q_max_departure_slope` | M | maximum gradient magnitude of that departure | mm/mm | – | nonnegative; uses the same fitted sphere, whose piston has zero slope |
| `freeform_max_departure` | M | maximum absolute XY/Zernike sag departure from its base conic | mm | – | nonnegative; includes piston and tilt when present |
| `tolerance_sensitivity` | T | `√((c(+δ) − c₀)² + (c(−δ) − c₀)²)/√2` for the criterion operand | criterion | as criterion | supported when the criterion is |
| `mtf` | MTF | MTF at a frequency | – | f λ | fixed-grid diffraction FFT or geometric analysis |
| `ghost_focus`, `ghost_irradiance` | Ghost | ghost focus distance / irradiance | mm / rel. | λ / f λ | explicit solver (below) |
| `stray_light_power` | StrayLight | MC stray power fraction | rel. | f λ | fixed seed / common random numbers; unsuitable for LM |
| `combine` | – | arithmetic on earlier operands | | | resolved by `OpticalProblem` |
| `custom` | X | plug-in kind registered in the `OperandRegistry` | | | plug-in evaluator |

**Telecentricity** is `chief_ray_angle` at the image (or `real_ray` `angle_y`
of the chief ray, which keeps the sign), and the exit-pupil position.
**Paraxial distortion** comes from the same `distortion` operand: the real
chief ray against the paraxial chief ray.

### Image quality, ghosts and stray light

`mtf` has `frequency` in cycles/mm, `orientation` (`tangential`, `sagittal`,
`average`) and `method` (`diffraction`, the default, or `geometric`). Use
`comparison: "greater_than"` for an MTF lower bound. The outer field and
wavelength selectors apply; `all` fields gives their weighted average.
`all` wavelengths sums **complex OTFs** with normalised spectral weights,
including lateral colour, before taking the modulus. Optional `wavelengths`
inside the kind selects a spectral subset (takes precedence over the outer
selector). `defocus_mm` defaults to zero; positive shifts the image plane
along propagation. `detector` takes the wave solver's `DetectorSettings`
(pixel pitch, fill factor/aperture, diffusion), multiplying each directional
optical MTF by its detector MTF before an `average`.

Diffraction uses the wave solver's FFT MTF with fixed `grid=64`, `rays=65`,
`supersample=4`, chief-ray reference and transmission amplitudes. Adaptive
rim tracing is disabled (`rim=false`). Entrance/exit grid counts and
supersampling never change between candidates. Each candidate caches its
traced pupils by field/wavelength and FFT OTFs by field/wavelength/defocus;
frequency, direction and spectral operands share those computations. Caches
are discarded after that candidate. The geometric route uses the geometric
analysis' fixed square sampling (`rings=32`) and defocus projection.

Recommended relative finite-difference steps are **1e-6 forward / 1e-5
central**, selected automatically by the evaluator, with
`h = eta * max(abs(parameter), variable scale)`. The Richardson check uses
`eta=3e-5`; the Cooke curvature/focus test requires relative agreement of
2e-3 forward / 2e-4 central. Fixed sampling removes adaptive sampling noise,
but clipping, resampling-cell crossings and FFT interpolation knots remain
piecewise smooth: check derivatives at the intended design and refine the
fixed grid for final analysis. These are sampled MTF targets, not a
convergence guarantee for arbitrarily aberrated pupils.

`ghost_focus` accepts either `pair: {first, second}` (stable surface IDs;
reflection at `second` then `first`) or `path` (the explicit solver's
`GhostPath` signature, surface **indices**, including optional DOE order
overrides). Without either, it ranks ghosts by paraxial relative irradiance
and selects the strongest `worst` N (default 1). `metric: "distance"`
(default) returns the **minimum absolute axial focus distance** from the
image across those paths, in mm; larger is better. Wavelength selection
averages those distances with spectral weights. `metric: "peak"` instead
ranks at the primary wavelength and returns the largest selected path's
real-ray peak/primary peak ratio over the selected spectrum (smaller is
better). The paraxial focus ranking is field independent; peak mode uses
the solver's system fields.

`ghost_irradiance` accepts the same `pair` or `path`, or sums enumerated
paths when neither is supplied. `max_reflections` (default 2) sets the
highest enumerated even order, up to the solver's limit of 8. Optional
`orders` filters enumerated reflection orders (empty keeps all up to that
limit); explicit signatures select their own order. `metric` is `peak`
(total map peak/primary peak), `power` (integrated detector ghost/primary
power), or `mean` (mean ghost/mean primary irradiance over the same detector,
equal to the power ratio). A ratio that is undefined (the primary is zero on
the detector) fails the operand. It uses the explicit ghost solver, with fixed
32×32 initial pupil cells and 64×64 detector bins; the solver refines clipped
cells. Detector dimensions come from the sensor or image clear aperture.
Peak ratios therefore depend on the bin size. `SolverEvaluator::ghost`
allows a Rust caller to choose finer settings, which must accompany the
run settings. Ghost objectives have path/ranking/bin/clipping changes;
derivative-free search is recommended.

`stray_light_power` uses the MC solver's fraction of the unobstructed
primary reference power, with `class` = `ghost`, `scatter`, `mechanical`,
`sneak` or `all_non_primary`. This is a standalone stray run: when combining
it with an explicit ghost operand, select the disjoint stray classes to
avoid counting ghosts twice. `rays` and `seed` are fixed settings, with the
**same seed for every candidate**, including finite-difference neighbours
(common random numbers, Philox). The snapshot hash and evaluation order
never modify that seed. Outer field/spectral selectors apply; field `all`
averages the per-field fractions with field weights. The merit uses one
detector bin (only integrated power matters), with the solver's default
scene and transport controls. `SolverEvaluator::stray` exposes these to Rust
callers. This stochastic operand is **unsuitable for LM**: CRN improves
comparisons but does not remove event/clipping discontinuities or MC error.
Use derivative-free/global methods and verify with more rays afterwards.
Solver errors follow the merit's failure policy; cancellation propagates.
Engine results carry the effective merit/settings and `RunInfo` as usual.

### Extended-surface operands

All extended sag and phase types in `SURFACES.md` are accepted by the CPU
trace evaluator. Axis first order uses `ParaxialXY`: placements are ignored,
and mixed xy power gives diagonal projections rather than principal focal
lengths. Higher-order terms have only their local quadratic meaning.

Manufacturing operands select a stable surface id. They sample its **clear
aperture**, including both rims of an annulus and rectangular/elliptical
boundaries: 64 equally spaced azimuths (plus explicit rectangular corner
spokes for extrema) and 65 radial nodes uniformly spaced in r² on each spoke (endpoints included). Explicit apertures override the
semi-diameter, as in tracing. Automatic semi-diameters are resolved once per
candidate by `TraceEvaluator`; `ModelEvaluator` needs a fixed semi-diameter
or an explicit aperture. Extrema are sampled estimates, so features narrower
than this grid can be missed. A domain violation at any node is `Invalid`.

DOE metrics use the physical order-1 structure, independent of selected
trace order, design order and design wavelength. The slope is the metric
surface gradient, not the planar gradient: on a radial base with slope z_r,
`|∇surface Φ| = |Φ_r| / sqrt(1 + z_r²)`. The period therefore measures
surface distance between successive 2π zones. A negative focusing phase
still has positive slope magnitude, period and peak-to-valley phase.

The Q sphere minimises area-weighted squared sag error, with curvature and
axial piston free. The radial trapezoidal weights include the aperture's
spoke areas; the additional corner spokes have zero fit weight. Curvature
is fitted by a bounded 72-step golden-section search inside the real sphere
domain. Both departure operands use that sphere, including for QCon with a
nonspherical base; QBfs's stored sphere is a basis
parameter and is not assumed to be the fitted sphere.

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

`DefaultCriterion::Mtf` emits polychromatic tangential and sagittal lower
bounds at every requested frequency and selected field/configuration. For
example:

```json
{"criterion":{"type":"mtf","frequencies":[10,30],"target":0.5,
              "method":"diffraction","worst_field":true},"thickness":null}
```

The default generator's `fields` and `wavelengths` select fields and the
polychromatic spectral subset, using their model weights. Without
`worst_field`, weights normalise over fields, configurations, frequencies
and the two directions. With `worst_field`, each field/direction MTF is a
zero-weight value operand and one `combine:min` over **every field and both
directions** adds the lower-bound residual of each frequency/configuration,
so least squares maximises the weakest value (minimax). Field weights
select the participating positive-weight fields; they do not scale the
minimum. The minimum has a kink where the weakest value changes.

A separate minimum per direction would let least squares trade one
direction's weakest field against the other's when the target is out of
reach: on the Cooke triplet (30 cycles/mm, target 0.8, back focus variable)
it shortens the back focus by 0.12 mm and drops the weakest MTF from about
0.17 to 0.05. The joint minimum reaches 0.173 from the nominal focus or from
0.1 mm of defocus; the field-weighted bounds settle at 0.061. The optimisation
test checks both. These Cooke values are physical: on axis the
polychromatic MTF at 30 cycles/mm is 0.69, but the 14° field's astigmatism
holds it near 0.17 (T 0.169, S 0.177; geometric 0.184/0.174; a 128 grid
gives 0.172/0.179). Far from focus (0.3 mm, about 4 waves at F/4) every
field oscillates through zeros and the minimum is multimodal, so LM from
there stops in a local maximum near 0.017.

Symmetry uses the core system classification, including shape coefficients,
placements and apertures. A system that is not meridionally symmetric
(class `none`, x fields, or the field grid below) never uses a half pupil,
even if the requested sampling explicitly says `bilateral`; a symmetric one
keeps a requested `bilateral`. XY and Zernike coefficient
variables conservatively disable the shortcut, including coefficients that
start at zero. This also applies to tolerance parameters and compensators.

For systems with a toroid, biconic, XY or Zernike surface, the default
fields are a Cartesian **3×3 grid** with normalised x and y coordinates
`{-1/√2, 0, +1/√2}`. Its corners lie on the unit field disk; it includes
both signs on both axes and all four quadrants, with equal field weights.
The grid is used even when the current coefficients happen to be
rotational, so the system stays sampled appropriately as optimisation or
tolerancing changes them. Other systems, including legacy meridional ones
(folded or tilted in the y–z plane), keep their own fields.
The maximum radial system field sets the physical scale as usual. An
explicit `fields` index list keeps the caller's field points and weights.
Grid merits use the full pupil because their fields have nonzero x.

`squeeze_target: <number>` adds an `anamorphic_squeeze` equality constraint
at the primary wavelength in every selected configuration (default: absent).
For example, `{"thickness": null, "squeeze_target": 2.0}` generates the
full-pupil grid merit and holds signed EFL_y/EFL_x = 2. The engine's optional
EFL hold still holds EFL_y; an explicit `efl_y` operand replaces that hold.
Automatic engine variables/tolerances address both x and y curvatures of
biconics and toroids rather than only the legacy y curvature.

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

**One evaluator per derivative.** A Jacobian batch carries its base point
(`CandidateBatch::anchor`). When the problem's evaluator says its values
depend on the batch (`Problem::anchored_differences`: real-aiming CPU
reference, resident GPU evaluator), the base point is re-evaluated in the
probes' batch and the probes are differenced against that row, not against
the `f0` of an earlier batch. The resident evaluator also keeps each operand
on one evaluator within a batch: when the reference completed an operand for
some candidates, it evaluates it for all of them. A finite difference thus
never divides a device/f64 or solve-path difference by the step.

**Seeded aiming (CPU reference).** With real aiming, the trace evaluator
traces the base point once as an anchor that records its field sources,
chief aims and pupil-ray aims with their stop Jacobians (analytic, one trace)
(`mklight_trace::AimSeeds`); every candidate of the batch, the re-evaluated
base point included, starts its solves there, takes at least one Newton step
and must resolve its change to 1e-6 of itself (never looser than the aiming
tolerance), else it solves unseeded. Measured on a 29-surface benchmark lens
(49 variables, real image heights, real aiming 1e-10): column errors against
a central-difference reference stay at the unseeded level (median 1.5e-6,
max 7e-4), while seeding without the re-evaluated base point raised them to
2.4e-2 (the unseeded `f0` carries aim errors up to the aiming tolerance that
cancel only between solves taken from the same start). `seeded_jacobian`
(ignored test of `mklight-bench`) prints the comparison. Serial CPU work falls 10–20 % (3-iteration LM, glass substitution); with 12 workers the gain is smaller, since the anchor precedes its batch. Prefix re-tracing (reusing rays before
the first changed surface) does not apply to such lenses: a real-image-height
field or image-space F-number moves every ray when any surface changes.

`ShapeCoefficient` and continuous `ShapeParam` paths work through the same
variable and tolerance machinery as legacy paths, including missing trailing
coefficients (read as zero, extended on write). For `ShapeCoefficient`,
`VariableSpec::to_variable` uses a surface-aware scale (`default_scale_for`;
legacy paths keep `default_scale`). For odd and axis power series and
normalised XY or phase terms of degree d, its floor is
`amplitude / (d · r^d)`, where r is the clear radius in the coefficient's
coordinates (divided by the normalisation radius for XY and phase). The
amplitude is 1e-3 mm of sag or 0.1 rad of phase. For automatic sizes, the normalisation radius, entrance-pupil radius, or
legacy 10 mm fallback supplies the radius. Q and Zernike bases contain lower
powers too: their floor is 1e-3 mm divided by the maximum of unit-basis sag
and clear-radius times unit-basis gradient, sampled with the core recurrence
at 65 radii × 32 azimuths. Thus polynomial order, basis normalisation and
normalisation radius all enter the scale. The larger of this floor and
`0.1 |coefficient|` is used; user `scale` and `fd_step` overrides still win.

`tests/surfaces.rs` checks production central differences against
`check_jacobian`'s Richardson stencil on all nine extended fixtures, with
x/y real rays and multiple coefficient orders (including Q order 19 at a
clear radius smaller than the normalisation radius), normalisation radii,
axis curvatures and conics. The per-entry tolerance is 1e-4 relative with a
0.1 derivative magnitude floor, allowing 1e-5 absolute error in very weak
cross-axis derivatives.

**Step size: chosen per operand and evaluator.** The default steps (`√ε`
forward, `∛ε` central, relative to `max(|x|, scale)`) are tuned to relative
rounding noise. Values with absolute noise need larger ones. When
`JacobianSettings::relative_step` is `None` (the default), the problem picks
the relative step `η` (`Problem::fd_relative_step`): an `OpticalProblem`
asks each enabled operand's evaluator (`OperandEvaluator::fd_relative_step`)
and takes the largest answer, because a step large enough for the noisiest
operand costs the others only `O(h)` truncation, far below their noise floor
at these sizes. An explicit `relative_step` still wins.

* **f64 wavefront operands** (`TraceEvaluator`): path lengths of ~100 mm
  carry an absolute rounding noise of ~1e-14 mm (~2e-11 waves), so the
  defaults gave ~3e-5 relative Jacobian column errors on the Cooke triplet.
  `η = 1e-6` (forward) / `1e-4` (central) is chosen automatically. Spot
  (ray-position) operands keep the defaults.
* **Device operands of the resident evaluator** (`ResidentEvaluator`,
  double-float traces of f64 aims, ~1e-12 relative noise): forward
  `η = 1e-6` for ray positions and `1e-5` for wavefront operands; central
  differences take the reference's steps. Largest relative column error
  against f64 central differences on the samples (default spot merit,
  every curvature; Metal and emulation, device and host aiming alike):
  3e-8–1.6e-6 at `1e-6`, against 4e-6–1.5e-3 with the f32-era `1e-3` it
  used before and 1.9e-6–4.5e-5 with `√ε`; wavefront merit 5e-7–2.2e-5 at
  `1e-5`, 4e-6–1.1e-4 at `1e-6`. The f32-era step also broke optimisation
  of vignetted lenses: on a 35-surface ultra-wide zoom a probe `1e-3` away
  crossed the vignetting edge of a ray that fails at the base point, so one
  Jacobian column held the failure penalty divided by the step (7e5 against
  2e3); the trial steps followed that column and were rejected, the trust
  region shrank below `xtol · ‖D x‖` (inflated by the column's Moré scale)
  and LM stopped after one iteration (`small_step`) on every device
  variant, while the reference improved the merit by 30 % in 65
  iterations. The reference fails the same way when given
  `fd_relative_step: 1e-3`.

Measured largest relative column error `‖J_j − R_j‖/‖R_j‖` on the Cooke
triplet (curvatures 1, 3, 6, an air space and the back focus; all fields and
wavelengths; `R` = f64 central differences):

| operands, evaluator | forward, `√ε` | forward, automatic | central, automatic |
|---|---|---|---|
| spot, f64 reference | 4e-7 | 4e-7 (`√ε`) | 2e-9 (`∛ε`) |
| wavefront, f64 reference | 2.7e-5 | 8e-7 (`1e-6`) | 8e-9 (`1e-4`) |

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
reports the resulting merit. `sensitivity` evaluates the nominal and each
tolerance at its limits in batches of `SENSITIVITY_BATCH` (32) points and
ranks the operands worst first, with the RSS prediction. Cancelled after the
nominal, it returns the operands whose both limits were evaluated
(`SensitivityReport::cancelled`). The compensator solves of the trials share
the job's cancellation but not its progress bar. `monte_carlo` draws trial `i` from Philox stream `TOLERANCE | i`,
so statistics are bit-reproducible for a seed and any trial can be replayed on
its own. The CPU reference and a GPU evaluator agree only within the accuracy
contract (the compensator solves settle within it), so a backend `auto` run
repeats bit-identically in the same engine because an identical request keeps
its first route (GPU.md §9.2); across engines or hosts `auto` may route
differently, and `backend: "cpu"` is bit-reproducible across sessions. Both are tested on the Cooke triplet with a back-focus compensator.
Without compensators every trial is a single evaluation, so the evaluator
receives a whole batch of perturbed systems in one call (one device batch
with the resident evaluator). With compensators, each trial runs its own small LM
in parallel. MTF is not unimodal in focus, so LM from the start point can
settle on a side lobe: with an MTF criterion, each bounded compensator is
first screened on 33 evenly spaced values across its range (plus the start
point), and LM starts from the best feasible one. This costs one batched
evaluation of 34 systems per compensator per trial (Double Gauss, one
focus compensator: about 15 → 29 ms per trial single-threaded); other criteria
skip it. The Monte Carlo report includes the compensated `nominal`, the
requested `percentiles` (linear interpolation between order statistics of
the successful trials; default 50, 90, 97.7), `records` of the first
`record_trials` trials (perturbations and compensators, so any trial can be
rebuilt with `ToleranceSpec::trial_case_system`) and `seconds_per_trial`, the
wall time over the completed trials. The engine job `tolerance` wraps both
drivers (see Engine jobs).

**Solves.** The drivers perturb parameters and evaluate the system as it
then stands. To keep a model's solves (`docs/SOLVES.md`) in effect, give
`OpticalToleranceEvaluator` its operand evaluator wrapped in
`merit::WithSolves`: it applies the solves of every candidate system before
the inner evaluator sees it, so the perturbed system of a case and each
candidate of its compensator solve are re-solved, as the engine job does. A
candidate whose solves cannot be applied has no operand values (`Invalid`,
with the solver's reason); a system without solves passes through unchanged.

### Irregularity and rigid elements

`ParamPath::Irregularity` (`{"param":"irregularity","surface":3,"mode":"astigmatism"}`
or `"spherical"`) is in **test fringes**: one fringe is λ_test/2 sag
peak-to-valley, i.e. one wave in a reflected (double-pass) test, whether the
surface refracts or reflects in the system. Setting it wraps the surface's
geometric shape (a diffractive surface's base) in a `sag_overlay` with the
surface's clear radius `a` (aperture or fixed semi-diameter) and the primary
wavelength as λ_test. The `tolerance` job first freezes the nominal traced
clear radius of each irregular surface, so automatic semi-diameters work
too; `irregularity_wavelength_um` overrides λ_test. With X = x/a, Y = y/a,
s = X² + Y² and θ = `astigmatism_angle_deg` in radians, the added sag in mm is

```
Δz = λ_test_um · 1e-3 · [f_astig ((X−Y)(X+Y) cos(2θ) + 2XY sin(2θ))/4
                         + 2 f_spherical (s² − s)].
```

the fringe astigmatism and primary spherical terms scaled to a PV of one
fringe on the disk, without piston (the vertex stays at z = 0). It extends
smoothly beyond the disk. In a reflected test the piston-removed RMS is
`sqrt(f_astig²/24 + 4 f_spherical²/45)` waves over a full circular pupil.
The criterion traces the real sag and normals in the actual media, on the
CPU or the GPU (GPU.md, manufacturing overlays).

θ (`astigmatism_angle_deg`, default 0, from +x toward +y) is a shape
parameter (`{"param":"shape_param","surface":id,"parameter":"astigmatism_angle_deg"}`).
Sensitivity and the nominal keep it. Monte Carlo with
`random_irregularity_orientation` (default true) replaces it, per trial and
per astigmatism-toleranced surface, with an independent uniform angle in
[0, 180)°; PV and RMS are unchanged. The angles come from Philox stream
`TOLERANCE_ORIENTATION | trial` of the seed, one draw per surface in
surface-id order, so they are reproducible and independent of the amplitude
draws, batching and threads. `ToleranceSpec::trial_case_system` rebuilds a
trial (seed, trial index, perturbations, compensators), as the engine's trial
preview does; `case_system` rebuilds sensitivity cases.

`ParamPath::ElementPlacement`
(`{"param":"element_placement","surface":3,"last":5,"pivot":5,"component":"tilt_y"}`)
moves the inclusive surface range `surface..=last` rigidly: `decenter_x`,
`decenter_y` (mm), `tilt_x`, `tilt_y`, `tilt_z` (degrees), in the first
vertex's axes, about the `pivot` vertex (default: the first). The model keeps
it as `element_placement` on the first surface; surfaces after the range do
not move (an entry/return pair). All components on one first surface share
the range and pivot. A single-surface decentre or tilt set through a
`ParamPath` on a surface without a placement now returns after that surface
(`restore_after`), so it no longer moves the rest of the system.

## End-to-end tests (`tests/trace_optim.rs`)

| test | system | result |
|---|---|---|
| singlet best form | f = 100, F/10, N-BK7, start equi-convex; vars c₁, c₂, back focus; EFL = 100 constraint; on-axis d-light RMS spot | shape factor `q = 0.730`, thin-lens best form `2(n²−1)/(n+2) = 0.740`; merit ÷150; EFL within 1e-6 mm |
| doublet achromatisation | N-BK7/F2, start with R₂ = −60, R₃ = −300 (focal shift > 0.5 mm) | F–C focal shift 2e-10 mm (test: < 1e-6), EFL = 100 ± 1e-6, R₂ → −40.2 mm, R₃ → −375 mm (the sample achromat, bent for minimum spherical aberration, has −42.5 / −410) |
| Cooke triplet | default spot merit (3 fields × 3 λ), 6 curvatures + back focus, EFL constraint, glass/air centre and auto-SD edge bounds | LM (40 iterations) lowers the merit by 35 %; seeded basin hopping is bit-reproducible and does not increase it; all constraints hold |
| MTF focus tolerancing (`tests/tolerance.rs`) | Double Gauss with a 0.3 mm focus offset, focus compensator ±0.8 mm, 18 trials, seed 71 | every trial stays on the main MTF peak (without screening, trial 6 settles on a side lobe: 0.84 vs 0.49); the first trial is at least as good as a 129-point focus scan |
| tolerancing | Cooke triplet, 9 tolerances, back-focus compensator | ranked sensitivities; the compensator removes > 50 % of an air-space error; Monte Carlo identical for a seed, different for another |
| cancellation | Cooke triplet | LM and basin hopping return `Cancelled` with the best-so-far point (feasibility-ranked) |

### Extended surfaces (`tests/surfaces.rs`)

| case | verification |
|---|---|
| odd, QCon and QBfs singlets | coefficient plus back focus ends below 0.2× the refocus-only optimum (which is 0.40, 0.085, 0.090× the start merit); repeated runs give identical variables and merit |
| hybrid refractive-diffractive singlet | starting with A₁ = 0, optimise A₁ to F–C focal shift <1e-6 mm; recover the independently constructed sample coefficient |
| anamorphic cylindrical adapter | over curvature_x and the singlet curvature, signed squeeze 1.7 within 1e-6 while the EFL_y constraint holds to 1e-8 mm |
| XY corrector | quadratic/cubic coefficients plus focus end below 0.2× the refocus-only optimum of the full-pupil x/y field-grid merit |
| Q coefficient and toroid curvature | finite, nonzero ranked sensitivities; seeded Monte Carlo repeats exactly |
| manufacturing | analytic plane/curved DOE phase gradients, phase range, annular aperture, pure-sphere zero Q departure, small quartic's fitted sphere departure/slope, rectangular XY departure |
| axis operands | EFL_x/y and BFL_x/y checked against independent thick-lens powers |
| default merit | anamorphic/freeform systems get the 3×3 grid and the full pupil (also with a requested half pupil); an XY coefficient variable disables the half pupil; legacy meridional systems such as the Newtonian keep their fields and half pupil |

## Throughput

Benchmark: `cargo test -p mklight-optim --release --test tests trace_bench:: --
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
run at the full rate. These are single timed runs.

## GPU-batched evaluation

`gpu::ResidentEvaluator` evaluates the ray-based operands of a batch of
candidates on any `mklight-gpu` backend (`ComputeBackend`: the Metal GPU, or
the CPU emulation, which produces the same bits). Every other operand goes
to its `BuiltinEvaluator` reference.

**Device operands** (`merit::gpu_eval::gpu_supports`): `ray_aberration`,
`rms_spot`, `spot_centroid`, `real_ray` (every quantity except the incidence
angle), `distortion`, `lateral_colour`, `wavefront_aberration` and
`wavefront_rms`; relative illumination has a device route of its own.
Paraxial, model, chief-ray-angle and real-F/# operands stay on the host.

How a batch is split between host and device, the aiming rule, pipelining
and accuracy are in "Resident evaluator" and "Host/device division of
candidate batches" below; routing between the CPU reference and
the device is the cost model of `GPU.md` §9.2. The engine reports the
estimated device rays per candidate as `backend.workload.rays_per_candidate`
(`gpu::device_rays_per_candidate`).

## Engine jobs: `optimize`, `merit_evaluation` and `tolerance`

`mklight-engine` registers these as analyses (`analysis::optim`, category
`optimization`). They run on the scheduler like any analysis, with
`AnalysisRequest { analysis: "optimize", settings, .. }`, settings
normalised against their schema, and a `RunInfo` whose `seed` is
`settings.seed`. The reference of their settings and result fields is
generated from the schemas (`docs/RESULTS.md` §8); the tables and examples
below add the meaning. Numbers follow CONVENTIONS §11: NaN is `null`, ±∞ is
`"inf"` / `"-inf"`. `ParamPath`, `VariableSpec`, `Operand`,
`MeritFunctionSpec`, `DefaultMeritSpec`, `ToleranceOperand` and
`Compensator` use their serde forms (for example
`{"param": {"param": "curvature", "surface": 3}, "lower": 0.0}`).

**Progress and interruption.** Progress is reported once per optimiser
iteration (generation, round) with the stage text
`"<algorithm> <iteration>: merit <best merit>"`, and once per Monte-Carlo
batch. Cancelling the job context stops the search of any of these jobs
between evaluations and returns `Cancelled`: no result, no final f64
scoring, whether the job runs on the scheduler or in a synchronous
`Session::run_analysis`. The scheduler drops cancelled
results, because a superseded result is stale.
Front ends use `Engine::stop_job`, `mk_job_stop` (ABI 1.2), Python
`Job.stop()`, or the CLI's first Ctrl-C to keep the result. Stop has its own
job token: even a queued request is retained until the job starts, and the
job ends normally (`Done`) with `status: "stopped"` and `stopped: true`.
A second CLI Ctrl-C cancels. Cancel/supersession still discards the result.
A stopped sensitivity keeps completed
operands; stopped Monte Carlo keeps completed trials. `status` is the same
function of how the search ended in every job (`job_status`): `stopped`
after a Stop, `converged` when the optimiser met a convergence test, else
`finished` (a budget or time limit, or a search without a convergence
test that ran to its end: `tolerance`, `glass_substitution`).

**Never worse than the start.** The job evaluates the start and the
optimiser's answer on the CPU f64 reference and returns the better of the
two, feasibility first, then merit (`ScalarEval::better_than` with
`feasibility_tolerance`). When the optimiser found nothing better,
`"improved": false`, the result is the start design and a warning says so.
The global methods never evaluate the start itself, and a GPU-driven search
ranks on the device's values, so this check is needed for them.

**Reported numbers.** The `start` and `final` merits, the operand table and
the constraint check are always computed on the CPU f64 reference, whichever
evaluator drove the search. `backend` says what evaluated the search
(`evaluated_on`, `precision`) and actual device/fallback accounting. Optimize
and glass reports use the CPU f64 reference at the end; tolerance and merit
evaluation return the active evaluator's values. Requested/used backend,
precision, device/build and cost decisions are in the envelope. Result
versions for these four jobs are 2.
Unsupported preparation/features, unresolved aiming and resource/device errors
remain real, counted fallbacks. Cancellation never starts reference replacement.
With device aiming, the device aims a ray for at most 8 Newton iterations
(`AimOptions::schedule_cap`, GPU.md "Bounded aiming commands") and the host
completes the rays it leaves unresolved in f64 (Host/device division below):
an unaimable ray of a bad trial step costs the device a bounded number of
iterations instead of the reference schedule, while healthy candidate rays
converge well within the cap.

**One model.** The four jobs (`optimize`, `merit_evaluation`, `tolerance`,
`glass_substitution`) work on the same system and build their merit the
same way (`SolvedStart`, `PreparedMerit` in `analysis::optim`): the solves
(`docs/SOLVES.md`) are applied to the snapshot's system, and the merit
(explicit or generated, fixed boundaries relaxed, extra operands and
constraints, the EFL hold) is built on that solved start. On a design with
solves they therefore report the same start merit (tested). No job changes a
solve-controlled parameter itself: such a parameter is never a variable, a
tolerance or a compensator. Automatic variables and tolerances skip it, an
explicit one is rejected (`'<key>'[i]: a solve controls this parameter`),
and a material variable on a glass whose model or offset parameters a solve
controls is rejected. Where a job derives other systems from the start:

* `optimize`: the search does not re-solve. The solve-controlled parameters
  keep the values of the solved start in every candidate and in the result.
* `glass_substitution`: every glass combination is the start of a
  re-optimisation of its own, so the solves are applied to the substituted
  design (built from the solved input) before it; the re-optimisation, like
  `optimize`'s search, does not re-solve.
* `tolerance`: the solves stay in effect in every system the job
  evaluates (`merit::WithSolves`): the perturbed system of each sensitivity
  limit and Monte Carlo trial is re-solved, and so is each candidate of its
  compensator solve. A solve then acts like a compensator that is always
  applied: an F/# solve restores the F/# of every trial, a marginal-height
  solve on the back focus refocuses it. The previewed case system is the
  re-solved one, so applying its solves changes nothing and its merit is
  the reported criterion (tested).

**Settings are checked through.** Beyond the top-level normalisation, each
job validates its settings recursively against its schema: an unknown key
or a wrong type inside `default_merit` (its `sampling`, `thickness` and the
`criterion` variant selected by `type`) is an `InvalidSettings` error, as
is an unknown key of `options` (below). The serde forms listed above
(`VariableSpec`, `Operand`, `ToleranceOperand`, `Compensator`,
`MeritFunctionSpec`) are parsed as their types define.

Both jobs also run with no settings at all: `optimize` then varies every
curvature and the back focus against the default merit (thickness boundaries
included) and holds the focal length; `tolerance` analyses a basic tolerance
set without compensators. With the generated default merit, a thickness
boundary that the start violates and that no variable changes (moving each
variable by its scale leaves its value unchanged) is relaxed to its start
value, with a warning and the label `"<kind> (relaxed to the start)"`. The
Cooke triplet sample's 0.99997 mm flint is an example: no search can meet the
default 1 mm glass minimum on it.

**Constraints by method.** LM, dogleg and basin hopping (its local LM) hold
constraints with the augmented Lagrangian. The job caps every inner solve at
an infeasible point at 10 iterations (`AugLagSettings::max_inner_iterations`,
`options.constraints.max_inner_iterations`, `null` for no cap), so the
multipliers are updated and a run that ends on its iteration budget ends close
to feasible. Without the cap, the first round of the Petzval and relay
samples used all 200 iterations at `μ₀` and ended with EFL errors of 6e-5
and 1.3e-4 mm. DE, CMA-ES and Nelder–Mead rank trial points by Deb's
feasibility rule, and random trial points never meet an equality exactly,
so for them the EFL hold is a penalty residual instead (label
`hold EFL (penalty)`). Its weight makes a relative EFL error of 1e-4 cost as
much as the start merit. That is about 5000 times the merit gained by
scaling the lens, so the EFL stays within about 1e-8 relative of what
scaling alone would push it to. Explicit equality constraints with these
methods get a warning. Their populations also start closer to the design
than the library defaults: DE `init_spread` 0.03 and CMA-ES `sigma0` 0.05
variable scales (10 % of a curvature by default), instead of 1.0 and 0.3,
which sample candidates so much worse that the search stays at the start.
In a one-off measurement on seven of the samples with the default settings,
every method (`lm`, `dogleg`, `de`, `cma_es`, `nelder_mead`, `hammer`)
improved every sample and ended feasible; the tests below cover the singlet
and the Cooke triplet.

**End-point feasibility restoration.** A nonlinear equality such as the
EFL hold can end an LM or dogleg run (and basin hopping's local solves) just
outside the feasibility tolerance, after which the feasible start would win
the ranking despite a much lower merit. Such a run therefore ends with up to
three Gauss–Newton corrections of the constraints alone; the tolerance is not
relaxed. With `r` the normalised equality errors and inequality violations
(zero for satisfied inequalities), `J` their Jacobian and
`S = diag(variable.scale)`, a correction solves
`min_z ‖A z + b‖² + sqrt(eps) ‖z‖²` with `A = J S / ‖J S‖F`, `b = r / ‖J S‖F`
by Householder QR: the ridge selects the minimum-norm step for dependent or
underdetermined constraints, and the normalisation makes it unit-free.
Failed derivative columns, and variables at a bound that the step would push
outward, are frozen. The step is capped at `‖z‖ ≤ 1e-3` scales, projected
onto the bounds and halved (at most eight times) until the total normalised
violation strictly decreases. The pass stops within `feasibility_tolerance`
(the ranking's) or without progress, so a feasible end point is never
touched. Its evaluations count against
`max_evaluations`, accepted corrections are recorded with stage
`feasibility restoration`, cancellation is honoured, and the termination
reason and iteration count are unchanged.

**Applying the result.** `result.system` is the optimised `OpticalSystem` in
model JSON. The app applies it with `Session::set_system`, which creates a new
undoable snapshot whose hash equals `result.system_hash` (tested).

### `merit_evaluation`

The merit of the start of an `optimize` request, evaluated once: the same
settings object, the same merit construction (explicit `merit` or
`default_merit`, extra `operands` and `constraints`, automatic variables,
fixed-boundary relaxation, the EFL hold that `algorithm` selects, solves
applied first) and the same operand report, without a search or derivatives.
A derivative-free EFL penalty is weighted by the start merit on the CPU
reference, as in `optimize`, whichever evaluator computed the values, so
both jobs report the same `merit_spec` on any backend (tested). On the
CPU every number equals `optimize`'s `start` report (tested). `backend`
and fallback follow `optimize` with a workload of
one candidate, so `auto` normally stays on the CPU; an explicit GPU traces
the ray operands there, and `backend.reported_on` names it.

The result holds `merit`, `violation` (total normalised), `status`
(`EvalStatus`), `operands`, `merit_spec` and `backend`, and no buffers.
`merit_spec` is the expanded `MeritFunctionSpec` (generated operands and the
EFL hold included), usable as a custom `merit`. `operands` rows are
`optimize`'s, with `start` and `value` both the snapshot value. Failed
operands have a null `value` and an `error`; the totals follow the spec's
failure policy.

### `optimize` settings

| key | type, default | meaning |
|---|---|---|
| `variables` | array, `[]` | `VariableSpec`s (`param`, optional `lower`, `upper`, `scale`, `fd_step`, `label`); empty: the curvature of every curved, optically active surface and the back focus (the thickness before the image) |
| `hold_efl` | boolean, `true` | hold the EFL at its start value, unless the merit already has an enabled `efl` operand: a constraint (label `hold EFL`) for `lm`, `dogleg`, `hammer`, a penalty residual (label `hold EFL (penalty)`) for `de`, `cma_es`, `nelder_mead` (Constraints by method) |
| `merit` | object or null, `null` | a `MeritFunctionSpec`; `null` generates the default merit from `default_merit` for the variables |
| `default_merit` | object, the full `DefaultMeritSpec` default | generator settings, laid key by key over the defaults (`{"thickness": null}` drops the thickness boundaries; `{"thickness": {"glass_min_centre": 0.9}}` changes one) |
| `operands` | array, `[]` | operands appended to the merit (e.g. an EFL target residual) |
| `constraints` | array, `[]` | operands appended as explicit constraints (role forced to `constraint`) |
| `algorithm` | `lm` \| `dogleg` \| `de` \| `cma_es` \| `nelder_mead` \| `hammer`, `lm` | the optimiser |
| `options` | object, `{}` | the algorithm's settings (`LsqSettings`, `DeSettings`, `CmaEsSettings`, `NelderMeadSettings`, `HammerSettings`), laid over the job's defaults: the library defaults except `constraints.max_inner_iterations` 10 (LM, dogleg, hammer's `local`), `init_spread` 0.03 (DE) and `sigma0` 0.05 (CMA-ES), with `max_iterations`, `fd_scheme`, `fd_relative_step` and `seed` in their fields. One rule for every algorithm: a key of `options` wins over the job setting that fills the same field (`options.max_iterations` over `max_iterations`, `options.jacobian` or hammer's `options.local.jacobian` over `fd_scheme` and `fd_relative_step`). A key must be a field of the selected algorithm's settings, through nested objects (`jacobian`, `constraints`, `local`); an unknown one is rejected with the valid keys. `method` (LM, dogleg) and `seed` (DE, CMA-ES, hammer) belong to the job's `algorithm` and `seed` and are rejected in `options` |
| `max_iterations` | integer or null, `null` | iterations (`lm`, `dogleg`, `nelder_mead`), generations (`de`, `cma_es`) or rounds (`hammer`); `null`: 200, 200, 2000, 100, 200, 20 |
| `fd_scheme` | `forward` \| `central`, `forward` | finite differences of LM / dogleg (and hammer's local runs) |
| `fd_relative_step` | number > 0 or null, `null` | `null`: chosen per operand and evaluator (Derivatives); 0 or a negative step is rejected |
| `seed` | integer, `0` | seed of `de`, `cma_es`, `hammer` |
| `backend` | `auto` \| `cpu` \| `gpu` \| `gpu_emulation`, `auto` | merit evaluation. `gpu` uses the GPU device the engine selected for the analysis (`AnalysisInput::device` holding the provider's `GpuDevice`), or the CPU reference with a warning when there is none. `auto` uses that device when the engine handed one over (under an engine `auto` request the engine's cost model decides whether it does, `GPU.md` §9.2). `backend.workload` reports the device rays per candidate estimated by `gpu::device_rays_per_candidate` (an upper bound of the resident plan's probes) for the merit's configuration-0 operands on the start system and the batch: the FD Jacobian (n, or 2n central) for `lm` / `dogleg`, the population for `de` (default max(10n, 15)) and `cma_es` (default 4 + ⌊3 ln n⌋), 1 for `nelder_mead`, `candidates_per_round` for `hammer`. A GPU provider lists `optimize` and `tolerance` in its capabilities so that the engine hands them its device. `gpu_emulation` requires the selected `gpu:cpu_f32` provider (validation, tests) |
| `feasibility_tolerance` | number, `1e-6` | total normalised constraint violation regarded as feasible: best-so-far ranking and `final.feasible` |

### `optimize` result

```jsonc
{
  "status": "converged" | "finished" | "stopped",
  "stopped": false,
  "termination": "small_merit_change" | "small_step" | "small_gradient" | "target_reached"
               | "converged" | "max_iterations" | "max_evaluations" | "cancelled"
               | "evaluation_failure" | "infeasible" | "stalled" | "derivative_failure",
  "algorithm": "lm",
  "iterations": 40, "evaluations": 361,
  "improved": true,                                  // false: the result is the start
  "start": {"merit": 3.97e-4, "violation": 0.0, "status": {"status": "ok"}},
  "final": {"merit": 2.6e-4, "violation": 2e-9, "status": {"status": "ok"}, "feasible": true},
  "search": {"merit": 2.6e-4, "violation": 2e-9,     // the optimiser's answer, as seen by the
             "reference_merit": 2.6e-4,              // search evaluator, and on the reference
             "reference_violation": 2e-9},
  "improvement": 0.35,                               // 1 − final/start (null if start is 0)
  "multipliers": [/* per constraint */] | null,
  "failed_variables": [],
  "variables": [{"label": "...", "param": {/* ParamPath */}, "start": 0.04, "value": 0.045,
                 "change": 0.005, "lower": null, "upper": null, "at_bound": false}],
  "operands": [{"index": 0, "kind": "ray_aberration", "label": "", "enabled": true,
                "role": "residual" | "constraint",
                "comparison": "equal" | "less_than" | "greater_than",
                "target": 0.0, "weight": 0.0123, "config": 0,
                "field": {"field": "index", "index": 1}, "wavelength": {"wavelength": "primary"},
                "start": 0.0071, "value": 0.0042,     // null where the operand failed
                "residual": 0.0042, "contribution": 2.2e-7, "contribution_fraction": 8.4e-4,
                "error": null | "RayFailure(\"…\")"}],
  "history": [{"iteration": 0, "evaluations": 9, "stage": "", "merit": 3.9e-4,
               "best_merit": 3.9e-4, "violation": 0.0}],
  "backend": {"requested": "gpu", "evaluated_on": "metal:Apple M2 Pro" | "cpu-f64-reference",
              "precision": "f32_gpu" | "f64_reference", "reported_on": "cpu-f64-reference",
              "workload": {"rays_per_candidate": 38, "batch": 7},
              // The next three only for runs handed a GPU device:
              "aiming": "by-batch" | "device" | "host",  // where the pupil rays are aimed
                                                     // (Host/device division)
              "gpu_stats": {"gpu_candidates": 0, "cpu_candidates": 0, "rays": 0,
                            "dispatches": 0, "device_seconds": 0.0},
              "resident_evaluation": {/* device, host-stage and fallback counters */}},
  "merit_spec": {/* the MeritFunctionSpec used, generated operands included */},
  "system": {/* OpticalSystem, model JSON */},
  "system_hash": "…64 hex digits…"
}
```

`status` is `converged` when the optimiser met a convergence test, and
`finished` when it stopped on a budget or a failure (see `termination`).
Buffer 0 is `merit_history` (f64, shape `[iterations, 4]`: iteration,
evaluations, merit, best merit). The search merit (`search`) equals `final`
on the CPU reference, and differs from it by the device's rounding on a
GPU (double-float traces, Derivatives). The
history holds one row per accepted iteration (LM, dogleg), generation (DE,
CMA-ES), iteration (Nelder–Mead) or round (hammer). `best_merit` is the
feasibility-ranked best, so it can rise when a feasible point replaces a
better infeasible one. The optimiser's answer is its result on normal
termination; when stopped or cancelled it is the better of that result and
the best point seen. `final` is the better of the answer and the start
(Never worse than the start).

### `tolerance` settings

| key | type, default | meaning |
|---|---|---|
| `tolerances` | array, `[]` | `ToleranceOperand`s (`param`, `minus`, `plus`, `distribution`, `label`); empty: a basic set, ±0.05 mm on every glass and air thickness except the back focus and ±0.5 % on every curvature of a curved optically active surface, without the solve-controlled ones. A parameter the system does not have, or that a solve controls (here and in `compensators`), is rejected |
| `compensators` | array, `[]` | `Compensator`s (`param`, `range` around nominal, `scale`), re-optimised by LM for every trial (screened first for an MTF criterion) |
| `criterion` | object or null, `null` | the criterion `MeritFunctionSpec`; `null` generates it with `DefaultMeritSpec::generate_for_tolerancing` from `default_merit` |
| `default_merit` | object, the default merit without thickness boundaries | generator settings, laid over that default |
| `mode` | `sensitivity` \| `monte_carlo` \| `both`, `both` | what to run |
| `trials`, `seed`, `batch` | `100`, `0`, `16` | Monte Carlo: trial `i` draws from Philox stream `TOLERANCE \| i` of `seed`; `batch` trials per evaluation batch, at least 128 (below) |
| `threshold` | number or null, `null` | criterion value for the yield |
| `compensator_iterations` | integer, `20` | LM iterations per trial |
| `percentiles` | array, `[50, 90, 97.7]` | Monte Carlo percentiles, each in [0, 100] |
| `record_trials` | integer, `0` (≤ 10000) | keep `records` of the first this many trials |
| `preview_case` | object or null, `null` | evaluate only `{"type":"trial","trial":k}` or `{"type":"sensitivity","operand":k,"positive":true}` (zero-based) and return it as `preview` instead of `sensitivity`/`monte_carlo` |
| `irregularity_wavelength_um` | positive number or null, `null` | fringe test wavelength; null keeps an existing overlay's, else the primary wavelength |
| `random_irregularity_orientation` | boolean, `true` | Monte Carlo: a uniform astigmatism axis in [0, 180)° per toleranced surface and trial, reproducible from the seed; false keeps the model's angle |
| `backend` | as for `optimize` | |

Without compensators, every trial is a single evaluation, and a batch of trials
goes to the evaluator as one batch (one GPU dispatch). The job hands the
evaluator at least 128 trials (and sensitivity points) per batch on either
backend: values do not depend on the batching, and with 16 trials per batch the CPU's 12 threads
idled at the end of every batch (29-surface F/1.4 lens: 155 → 120 ms for
the default run, 2.4 → 1.9 s for 1000 trials at 6 × 12 sampling, identical
results). Progress and a stop request act between batches.

### `tolerance` result

```jsonc
{
  "status": "finished" | "stopped",
  "stopped": false,
  "sensitivity": {                                  // null when not run, or stopped before the nominal
    "nominal": {"criterion": 3.97e-4, "ok": true, "compensators": [42.25]},
    "operands": [{"index": 0, "label": "...", "param": {/* ParamPath */},
                  "minus": 0.05, "plus": 0.05,
                  "criterion_minus": 4.1e-4, "criterion_plus": 4.0e-4,
                  "change_minus": 1.3e-5, "change_plus": 3e-6, "worst": 1.3e-5, "ok": true,
                  "compensators_minus": [42.31], "compensators_plus": [42.19]}],
    "ranking": [2, 0, 5, 1, 4, 3],                   // operand indices, worst first
    "rss_change": 2.1e-5,
    "predicted_criterion": 4.18e-4,                  // nominal + rss_change
    "evaluations": 13,
    "cancelled": false                               // true: the operands completed before a stop
  },
  "monte_carlo": {                                  // null when not run
    "trials": 100, "failures": 0, "seed": 0,
    "nominal": {"criterion": 3.97e-4, "ok": true, "compensators": [42.25]},
    "seconds_per_trial": 0.02,                       // wall time over completed trials
    "records": [{"trial": 0, "criterion": 4.1e-4, "ok": true,  // first record_trials trials
                 "compensators": [42.31], "perturbations": [0.01, -0.02]}],
    "mean": 4.2e-4, "std_dev": 1.1e-5, "min": 4.0e-4, "max": 4.6e-4,
    "percentiles": [{"percentile": 50.0, "value": 4.2e-4}, …90, 97.7],
    "yield_fraction": 0.93 | null, "threshold": 4.5e-4 | null,
    "compensator_mean": [42.25], "compensator_std": [0.04],
    "cancelled": false,                              // true: statistics of the completed trials
    "criteria": [/* per trial, "inf" for failed trials */]
  },
  "preview": null,                                  // with preview_case: {system, system_hash,
                                                    //   criterion, compensators, perturbations}
  "compensators": [/* the Compensators */],
  "criterion_spec": {/* MeritFunctionSpec */},
  "backend": {/* as for optimize */}
}
```

Buffer 0 (when Monte Carlo ran) is `monte_carlo_criteria` (f64, one per trial).

### Tests (`mklight-engine`, `analysis::optim::tests`)

| test | result |
|---|---|
| schemas | both registered, available and runnable with their defaults; unknown algorithm, malformed options or `ParamPath` → `InvalidSettings` |
| defaults | singlet: variables c₁, c₂ and the back focus, EFL held at its start value (replaced by an explicit `efl` operand); tolerances: the lens thickness and both curvatures |
| singlet through the scheduler | equi-convex start, EFL = 100 constraint: converged, merit ÷5 or more, EFL within 1e-6, the result applies as a session snapshot with the reported hash |
| Cooke triplet | default merit with thickness boundaries and EFL constraint, LM 40 iterations: merit −20 % or more, every constraint holds to 1e-6, two runs give identical JSON; seeded DE: identical for a seed, different for another, improves |
| cancel and stop | cancelled after two LM iterations: `Cancelled` without reference scoring; `Engine::stop_job` through the scheduler: `Done`, `stopped`, improved; a cancelled `tolerance` run and trial preview: `Cancelled`; `job_status` and the `status` values of the three result schemas |
| one solved model | a triplet with an F/# solve and a paraxial-focus solve through the four jobs: the same start merit, that of the solved system; automatic variables and tolerances skip the solved parameters, explicit ones (variables, tolerances, compensators) are rejected; every tolerance case (sensitivity limit, trial, compensated trial) is re-solved, its previewed system meets its solves and reproduces the criterion; every glass combination starts from the re-solved substituted design |
| nested settings | a misspelt key in `default_merit`, its `sampling`, `thickness` or `criterion` is rejected by all four jobs; unknown `options` keys are rejected with the algorithm's valid keys, `method` and `seed` with the setting to use; every field of an algorithm's settings is accepted and wins over the job's setting |
| tolerance | 6 tolerances, back-focus compensator: ranked sensitivities, compensator moves, Monte Carlo identical for a seed, different for another |
| GPU backends | `gpu_emulation` everywhere and `gpu` (Metal) when present: GPU precision, no runtime qualification, reported on the reference, merit decreases; `auto` with a handed-over device uses it at both merit sizes |
| global methods | `cma_es`, `de`, `nelder_mead` with the defaults on the singlet and Cooke triplet: improve, feasible, EFL within 1e-4 relative by the penalty; a DE run that cannot improve returns the start (`improved: false`, same snapshot hash) with a warning |
| snapshot merit | Cooke triplet and Double Gauss, default RMS spot, worst-field MTF and a custom merit, and the DE penalty hold: equal to the optimize start report; expanded defaults round-trip as custom; CPU-f32 GPU counters record one candidate; solves applied first; per-operand errors |
| feasibility restoration | nonlinear equality leaves a 2.7e-6 defect after one improving LM/dogleg step: repaired below the ranking tolerance, bounds and hard evaluation budgets respected, cancellation retains the best point; 20 LM iterations on the Double Gauss with the worst-field MTF merit (release, ignored) end 3e-3 off the EFL hold and are restored instead of returning the start |
| defaults feasible | `{}` on the Cooke triplet (the 1 mm glass minimum relaxed to 0.99997, warned), the Petzval and the relay: feasible, improved, EFL held to 1e-6 mm |
| bad input | `fd_relative_step` 0 or negative, tolerances or compensators on a missing surface → `InvalidSettings` |
| stopped sensitivity | a compensated sensitivity of 90 tolerances stopped through the job's stop token after its first batch of 128 points: `stopped`, the completed operands kept, `cancelled: true` |

## Glass substitution: `glass_substitution`

Replaces the glasses of chosen lenses with real catalogue glasses. For every
glass combination it tries, the job re-optimises the continuous variables
by LM, then ranks the combinations. Use it to replace model or unavailable
glasses with real ones, and to find better glass pairs. The analysis
registry and the existing Stop API expose it to every client; no new FFI
call is needed.

The only setting a run needs is which glasses may change. A material
variable names the **stable id of the surface in front of the glass**; a
cemented doublet is two variables (its front and cemented surfaces):

```json
{"material_variables": [{"surface": 1}, {"surface": 2}]}
```

With no material variables at all, every model glass is snapped (below).
Every other setting has a default:

| setting | default | meaning |
|---|---|---|
| `material_variables` | `[]` | `[{surface, candidates?}]`; empty: every model glass |
| `algorithm` | `"tabu"` | `tabu` searches glass combinations; `hammer` hops over glasses and shape together |
| `max_combinations` | 100 | LM re-optimisations (tabu: distinct combinations; hammer: also the same glasses from another shape) |
| `max_seconds` | `null` | search wall-time limit; keeps the results so far, `termination: "time_limit"` |
| `local_iterations` | 100 | LM iterations per re-optimisation |
| `top_n` | 10 | combinations returned |

`variables`, `merit`, `default_merit`, `operands`, `constraints`,
`hold_efl`, `fd_scheme`, `fd_relative_step`, `seed`, `backend` and
`feasibility_tolerance` are those of `optimize`, and the merit is built by
the same preparation on the solved start (Engine jobs, "One model"). Empty
continuous variables select the curvatures of the curved surfaces and the
back focus; explicit variables may be any `ParamPath` that no solve
controls. The re-optimisation is
`optimize`'s LM: the same merit, augmented-Lagrangian constraints, FD steps
and evaluator. The solves of the design are applied to every substituted
combination before its re-optimisation. `hold_efl` holds the input EFL as a constraint unless the
merit has an EFL operand.

### Candidates and model-glass snapping

Each `candidates` object is optional; its fields are:

| field | default | meaning |
|---|---|---|
| `catalogs` | `[]` | loaded catalogues to draw from; empty: every loaded catalogue except `BUILTIN`, or `BUILTIN` when it is the only one |
| `statuses` | `["standard", "preferred", "melt"]` | allowed AGF status codes: 0 standard, 1 preferred, 2 obsolete, 3 special, 4 melt |
| `window` | `null` | half-widths `{nd, vd}` around the current glass |
| `nearest` | 10 | the K glasses nearest the current one, after the other filters; `null`: no limit |
| `include`, `exclude` | `[]` | names or `catalog:name`, case-insensitive. Includes are kept beyond `window` and `nearest`; excludes always win |
| `min_transmission` | `null` | internal transmission at the shortest system wavelength through the element's centre thickness |
| `max_relative_cost` | `null` | AGF relative cost (`OD` field 1) |
| `max_climate_class`, `max_stain_class` | `null` | AGF climate (CR) and stain (FR) resistance classes (`OD` fields 2, 3) |

The AGF reader keeps status, the exclude-substitution flag and the CR/FR
classes in `Glass.substitution`, so loaded and saved catalogues keep their
filters (see `MATERIALS.md`). Glasses flagged exclude-substitution are never
used, and status and quality filters also apply to includes. A glass
without transmission, cost or resistance data passes that filter. A
candidate must cover every system wavelength. Unknown catalogue or include
names, and an empty candidate set, are errors.

`min_transmission` uses the catalogue's internal transmittance,
`τ = exp(−α·t)` (no interface losses, axial path only). A glass with data
also gets the constraint `thickness ≤ −ln(min_transmission)/α` in every
configuration during its re-optimisation, so a thickness variable cannot
break the filter.

**Distance.** Glasses are compared at `(n_d, V_d/100, ΔP_g,F)`, computed
from each glass's dispersion formula at catalogue conditions (not the
rounded catalogue n_d, V_d), with
`ΔP_g,F = (n_g − n_F)/(n_F − n_C) − (0.6438 − 0.001682 V_d)`:

```
d² = (Δn_d)² + (ΔV_d / 100)² + 100 · (Δ ΔP_g,F)²
```

A change of 0.01 in index, 1 in Abbe number or 0.001 in partial-dispersion
anomaly counts the same: for typical glasses (n_d − 1 ≈ 0.5, V_d ≈ 50, a
crown/flint P_g,F difference ≈ 0.04) each changes an element's power, its
colour or the pair's secondary spectrum by roughly 2–3 %. Model glasses use their stored
`(nd, vd, dpgf)`. Ties sort by catalogue, then name. Search moves use the
same metric: each glass can move to its 4 nearest candidates, plus 4 sampled
two-glass moves per tabu iteration or hammer round.

**Snapping.** With `material_variables` empty, every optical gap with a model
glass becomes a variable with the default candidates: its 10 nearest real
glasses. The nearest one is the search's starting point. To choose K, list
the surfaces with `candidates.nearest`. A substituted surface cannot also
have `model_nd`, `model_vd` or `model_dpgf` variables.

### Search, ranking and interruption

**Tabu** (`discrete::tabu_search`) starts from each variable's nearest
candidate and evaluates each iteration's moves as one parallel batch of LM
re-optimisations (`LocallyRefined`). Every combination starts from the
**input shape**, so its result does not depend on the order the search
reached it; tabu caches it by combination. It uses tabu's defaults (tenure 5,
stall window 20, five seeded restarts of two random changes).

**Hammer** (`mixed::mixed_hammer`) is basin hopping over glasses and shape:
each round proposes 8 moves, each a Gaussian shape perturbation (0.5
variable scales, adapted towards 50 % acceptance) plus, with probability
0.5, a neighbourhood glass swap. All 8 are re-optimised in parallel, and the
best is accepted when better, or by the relative-temperature (0.05)
Metropolis rule of `hammer`. There is no pre-refinement screen, which could
discard a glass pair that is only good once re-bent. It stops on
`max_combinations` or 20 rounds without a new best, and keeps the best shape
for each combination.

Each re-optimisation returns the best point it saw under the feasibility
rule, not just its last iterate. A short or interrupted solve that ends
infeasible therefore never ranks below its own feasible start (for the
input glasses, the input design).

The kept combinations are re-scored on the CPU f64 reference and ranked:
feasible first, then by merit (infeasible ones by violation, then merit),
ties in combination order. `results` holds up to `top_n` distinct
combinations, each with merit, violation, its glasses (catalogue, name,
n_d, V_d, ΔP_g,F), the continuous variables against the input, and the
complete optimised `system` with its hash, so a front end can preview and
apply any entry as one undoable edit. `starting_merit`, `candidate_sets`,
`statistics`, a `history` row per re-optimisation, the effective
`merit_spec` and the `backend` record come with it.

With fixed settings and seed, and no time limit or Stop, the result is
identical for any number of Rayon workers, on the CPU reference and on a
device with either aiming (tested with device aiming on the emulation). Progress reports the number of
re-optimisations, the best merit and the best combination's glasses. Stop
(`Engine::stop_job`) ends the search and returns the ranking
so far, including re-optimisations cut short (`local_termination:
"cancelled"`); Cancel discards the job. `max_seconds` covers the search only,
not setup or the final ranking.

### Cost and benchmark

Re-optimisation dominates. For C combinations, n continuous variables and
I LM iterations, forward differences cost about `C · (1 + I · (n + 1))`
merit evaluations, plus rejected steps and constraint updates (central
differences: 2n). One search iteration's combinations run in parallel. All re-optimisations share the resident GPU evaluator's device context and counters, each with routing state of its own (`ResidentEvaluator::for_solve`), with each FD Jacobian
in one batch. No candidate/reference checks run during the
search. `auto` runs the search on the CPU reference: `glass_substitution` is an
accuracy exclusion (`GPU.md` §9.2, "Automatic routing"). The final ranking
adds one CPU f64 merit evaluation per kept combination.

The ignored Cooke-triplet benchmark has three glass variables (K = 3), a
12-combination budget, seven continuous variables and 8 LM iterations per
combination, with the d-line axis/edge default merit:

```sh
source ~/.cargo/env
cargo test -p mklight-engine --release --lib benchmark_cooke_three_glasses -- --ignored --nocapture
```

On 2026-10-31 (Apple M2 Pro, CPU f64, 12 Rayon workers, a busy host) it ran
the 12 re-optimisations, 825 merit evaluations, in **24 ms** at best (about
500 combinations/s; up to several times slower under load), including setup
and the final ranking. Eight LM iterations do not reach feasibility on this
lens, so the input design ranks first: the benchmark measures throughput
only.

Tests cover the cemented-doublet known answer (crown n_d 1.52, V_d 60 and
flint n_d 1.62, V_d 30 from a bad pair, with the F–C focal shift checked by
an independent paraxial trace), model snapping, AGF status and quality
filters, the transmission limit under thickness optimisation, determinism
across worker counts, sorted unique results, Stop and Cancel, the time
limit, the defaults, settings validation, and GPU re-optimisation on CPU-f32
emulation and, when available, Metal. They use only the built-in glasses
and synthetic catalogues.

## Known gaps

* No analytic Jacobians (see Derivatives).
* Manufacturing extrema use the documented finite aperture grid, without
  certified bounds between nodes. Coupled freeform first order still needs a
  4×4 model; the x/y operands expose the reference diagonal sweeps.
* Relative illumination is geometric and ignores coating and bulk
  transmission. For its integration limits, see Relative illumination.
* `spot_centroid` uses the default full-pupil sampling (3 rings × 6 arms).
* There is no cross-call cache: a candidate evaluated twice is traced twice.
  (Tabu glass substitution caches whole re-optimisations by combination.)
* GPU-batched evaluation keeps the per-candidate host work on the CPU:
  snapshot, resolution and paraxial setup, ray generation (and aiming),
  packing, and operand reads. At the default merit's sampling this work costs
  as much as the CPU trace it replaces (see Throughput). The division
  (Host/device division of candidate batches) overlaps the host share with
  the device but keeps it: tracer construction, automatic semi-diameters and
  the f64 failure path of unaimable rays bound its gain.
* The GPU evaluator evaluates chief-ray angles, incidence angles and real
  working F/# on the CPU, and of relative illumination only the radial
  visibility scans run on the device (Resident evaluator).
* Accuracy evidence is offline. Unchecked Metal output can retain the known
  residual gaps listed in GPU §9.2.

## Resident evaluator

`gpu::ResidentEvaluator` evaluates candidate batches on a device; the
engine's candidate jobs use it for a run handed a GPU device, and their
routing is in GPU.md §9.2. It is constructed from a `GpuContext` (the base
snapshot and the backend) and the configured `BuiltinEvaluator` reference,
and `ResidentEvaluator::run` wraps a driver to retain the actual backend's
`RunInfo` and the cumulative plan, copy, allocation and fallback counters.
How a batch is divided between host and device is the next section.

* **Device stages.** One resident plan per batch (or pipelined sub-batch):
  `aiming` operations generate or take the rays and trace them in hi + lo,
  `geometry` forms chief-relative coordinates and reference-sphere OPD, and
  `collectives` supplies weighted two-pass spot moments and batched weighted
  piston/tilt QR fits. Field and spectral combinations and the operand
  values are formed on the host.
* **Other operand families.** Fixed-sampling MTF operands
  (`merit::mtf_eval`) and ghost and stray power operands
  (`merit::solver_eval`) go through the wave, ghost and stray adapters with
  the same `GpuContext`. Relative illumination runs its radial visibility
  scans on the device in a candidate batch and completes the adaptive
  boundary search and quadrature on the host (`ri_visibility` counts the
  lookups). Model scalars and paraxial operands are host evaluations.
* **One evaluator per solve.** An evaluator serves one solve: a sequence
  of batches evaluated one after another. What it learns from its batches
  (the routing below) belongs to that solve. Clones are the same solve;
  `ResidentEvaluator::for_solve` gives the evaluator of another one, with
  the same device context, reference and counters and routing state of its
  own. Solves that run in parallel each get one: a glass combination's
  re-optimisation (the engine's problem builder), and each parallel local
  solve of basin hopping, which asks its problem for an instance
  (`Problem::for_solve`, `OperandEvaluator::for_solve`). Their values then
  do not depend on thread timing or on the number of workers (tested).
* **Independent solves.** `gpu::scheduler::run` rendezvous-batches
  independent serial solvers: each active worker supplies its next optical
  evaluation, and one coordinator combines matching operand lists in stable
  worker order; finished workers leave subsequent rounds. Compensated
  tolerance trials use it. The joined batches are one sequence in stable
  order, so the workers share the routing state of the evaluator behind
  them; a joined batch is evaluated near the anchor of its first worker
  that has one.
* **Tall QR.** A `Problem` that returns the context from `gpu_context()`
  gets the resident tall QR of LM from `m·n ≥ 2¹⁶` Jacobian elements
  (GPU.md §9.2, "QR policy"); a rank-deficient factor is a counted reference
  factorisation, and neither path forms normal equations.
* **Publication.** Device values publish directly: no candidate, merit,
  finite-difference or constraint value is re-evaluated on the reference to
  check it. Final reports are scored as "Reported numbers" above describes.

Tests: `resident_algebra`, `resident_candidates`, `resident_solvers`,
`scheduler`, `resident_tolerance` and `host_device_division`
(`crates/mklight-optim/tests/`); the ignored `resident_bench` and
`resident_workloads` harnesses time the evaluator against the f64 reference
(`RESIDENT_BENCH_MERIT=wave` selects the wavefront merits). The adapter's
earlier design and its measurements are recorded in
[history/optimization-resident-adapter.md](history/optimization-resident-adapter.md).

## Host/device division of candidate batches

The engine's candidate jobs (`merit_evaluation`, `optimize`, `tolerance`,
`glass_substitution`) divide a run handed a GPU device between the CPU and
the GPU as described here (`gpu::ResidentEvaluator`); CPU-reference runs
are unaffected.

**The division.** Per batch:

1. *Host, parallel over candidates (f64):* tracers (snapshot, resolution,
   fields with their real-image-height and chief solves), operand plans,
   chief seed rows, the chief rays themselves (aimed by the host tracer and
   handed to the device as rays: on the 35-surface wide-angle lens device
   aims from the host seeds failed for 13 of 30 field/wavelength chiefs),
   packed systems, and one resident plan: chief traces, pupil-ray aiming
   from the host chief seeds, trace, chief-relative coordinates and moment
   reductions (OPD and piston/tilt fits for wavefront merits). The device's
   aiming schedule is capped at 8 Newton iterations per ray and its continuation rounds are committed
   without reading the pending count back (`AimOptions::speculative_rounds`;
   one round without real aiming).
2. *Host, while the device runs:* the rays routed to the host (below) and
   every operand that needs no ray (model operands with their automatic
   semi-diameters, paraxial and first-order operands), from the same
   tracers; no separate first-order plan.
3. *Host, after the plan:* the device rays are read in place; every ray the
   device left unresolved is aimed and traced again with the candidate's
   f64 tracer, so an aim fails only when the reference's fails (a ray the
   device aimed and then traced to an aperture stays clipped); a group with host rays forms its aggregates from rays on
   the host; then all operand values, with `Candidate::value_with`.

A chief ray the device fails to trace is replaced by the host's, with every
ray of its field and wavelength traced on the host (they depend on it
through the device's chief-relative coordinates and OPD); no candidate is
reported whole to the reference. In a finite-difference batch (anchored:
probes and the re-evaluated base point) a ray completed on the host for one
candidate is completed on the host for every candidate, so a derivative
never differences a device ray against an f64 one. Pupil rays that were
unresolved or slow (more than 6 device iterations) in an eighth of a
device-aimed batch's candidates are routed to the host before the plan in
the following device-aimed batches of the same solve, the same for every
candidate of a batch, for the rest of that solve (One evaluator per solve,
above): the device then runs only lanes that converge in a few
iterations (2–3 ms of device time per plan on the 29-surface lens), and the
host traces the hard rays while the device works. A solve's first
device-aimed batch of four or more candidates is preceded by three of its
candidates traced alone, to learn those rays (one perturbed Monte Carlo
trial learned 18 of the 34 rays the batches route on the 35-surface lens,
three learn all): the Metal executor runs a large
generation in tiles one after another, each as long as its slowest lane,
so a cold 91-candidate sensitivity batch spent 370 ms of device time in
67 command buffers on the 29-surface lens. When a device-aimed batch leaves
more than half of a candidate's rays to the host, counting the rays of that
batch routed before its plan and those it adds to the routing against the
rays of that batch (on the 35-surface ultra-wide lens the device
aimed 166 of 570 within the cap), the evaluator hands the rest of the solve
to the reference (`evaluate_batch_near`, counted as a fallback).

**Pipelined batches** (`PIPELINE_FROM`). A batch of at least 64 candidates is split into
three sub-batches (at least 8 candidates each), and the host preparation of
sub-batch k + 1 and the finishing of k − 1 overlap the device's work on k:
Monte Carlo and sensitivity batches gain (1000 trials at 6 × 12 sampling on
the 29- and 35-surface lenses: 1.92 → 1.72 s and 1.98 → 1.75 s), a
23-candidate Jacobian loses (Double Gauss `optimize` 792 vs 582 ms), so
Jacobians stay one plan.

**Aiming** (the route option `solver_aiming`, GPU.md §9.3). *Host aiming:*
every pupil ray is aimed on the host by the candidate's f64 tracer (`generate_many`: the reference's
aims, bit for bit) and the device only traces it from the host's starting
ray (`TraceResident`) and forms the chief-relative coordinates, moments,
OPD and fits. A ray the host cannot aim fails as on the reference (it
enters the plan with an aim-outside report and is not completed again);
there is no device aiming, no probe candidate and no routing. In a
finite-difference batch the base point (the anchor the batch is given,
whichever candidate comes first) is first evaluated on the host and
its recorded solves seed every candidate's tracer, as in the reference's
`evaluate_batch_near`, so the probes' aims are the reference's seeded
aims. A host-aimed batch of a solve that has learned rays under device
aiming routes none of them. *Device aiming:* the device aims with host completion as above.

*The rule* (`solver_aiming` = `by-batch`, `gpu::DEVICE_AIM_FROM`) picks per
batch: the device aims a batch whose estimated device rays times the system's
surfaces without a closed-form intersection (aspheres, polynomial and
freeform sags; none under paraxial aiming, which solves nothing) reach
2e6, the host aims every other. Host aiming costs
what the reference spends; device aiming saves host work only where a ray
is expensive on the host and the batch amortises the device's plan and
its cold first batch. Measured (shuffled, quiet machine, Monte Carlo
batches of 128 unless noted; time relative to host aiming):

| lens | batch work | device aiming |
|---|---:|---:|
| 50 spherical surfaces, 1000 trials at 6 × 12 | 0 | 1.04 |
| Double Gauss (paraxial aiming), 1000 trials at 6 × 12 | 0 | 1.00 |
| 29 surfaces, 2 aspheres: default tolerance batches | 4e5 | 1.19 |
| 29 surfaces, 2 aspheres: 256 trials at 5 × 10 | 1.5e6 | 1.49 |
| 29 surfaces, 2 aspheres: 1000 trials at 6 × 12 | 2.2e6 | 0.84 |
| 35 surfaces, 8 aspheres: LM Jacobian (33 candidates) | 6e5 | 1.2 |
| 35 surfaces, 8 aspheres: 256 trials, default sampling | 2.2e6 | 0.90 |
| 35 surfaces, 8 aspheres: 1000 trials at 6 × 12 | 8.8e6 | 0.79 |
| 39 surfaces, 7 aspheres: 1000 trials at 6 × 12 | 7.7e6 | 0.82 |

Batch size alone does not separate these: the spherical lens loses with
device aiming at every size, the aspheric zoom wins at default
tolerance sizes. LM Jacobians, trial points, glass refinements and merit
evaluations stay far below the threshold. The route option `solver_aiming`
set to `device` pins device aiming for every batch, set to `host` host
aiming; `by-batch` (the default) is the rule above. Reported as
`backend.aiming` (`by-batch`, `host` or `device`) and `solvers_aiming`; the batches the
device aimed are counted in `resident_evaluation.device_aimed_batches`.

**Speculative Jacobians** (`LsqSettings::speculative_jacobian`). A trial
point of LM or dogleg that follows an accepted step is evaluated in one
batch with the first-pass finite-difference probes of the Jacobian at it
(`deriv::fd_first_batch`); when the step is accepted, its Jacobian needs no
further batch, so an iteration costs one batch instead of two. A rejected
step wastes the probes, so trial points after a rejection are evaluated
alone: on the 29-surface lens, whose merit (failed-ray penalties) rejects
about ten trial points per Jacobian, speculating on every trial point
evaluated 9× as many candidates (10 177 against 1110) for the same 34
iterations. Every evaluation performed counts against `max_evaluations`. On an
evaluator whose values do not depend on the batch the iterates are
bit-identical to the plain run (tested on the Cooke triplet).

Speculation applies with device aiming pinned only. With host aiming every probe
is a host evaluation, as on the CPU reference (which does not speculate),
and the probes of a rejected trial point sit at a design whose rim rays
fail and run the whole aiming schedule: on the 35-surface zoom, 8 LM
iterations with every trial rejected took 606–623 ms and 544–577
evaluations with speculation and 466 ms and 387 evaluations without
(CPU reference 368 ms, 387 evaluations); on a full optimisation of that
lens speculation added 64 % of evaluations per iteration. The Double Gauss
optimisation is unaffected (335–444 against 366–389 ms), its glass
substitution 6 % slower without it.

**Accuracy.** Device rays are double-float traces of f64 aims;
host-completed rays and rayless operands are the reference's f64 values
(bit-identical for model and paraxial operands, tested). Forward
differences take the device steps of "Step size" above.
`tests/host_device_division.rs` checks the division against the reference on
every sample with a default spot and wavefront merit plus thickness
boundaries, first-order, aggregate, single and inner real-ray operands
(emulation and Metal; device and host aiming), host
completion, routing and handover to the reference at device caps of 1–8
iterations (the real-aimed Cooke triplet at cap 4 routes rays without
handing over), routing state per solve (`for_solve` starts afresh after a
handover, clones stay handed over), the handover judged per batch (a dense
merit's learned rays do not hand a sparse merit's batches over), host-aimed
batches without routing, anchor seeding independent of the candidate order,
equal values of a pipelined batch and its halves,
finite-difference column errors (1.8e-7 against f64 central differences on
the real-aimed Cooke triplet, also with host completion), LM on the
evaluator, and speculative LM. `crates/mklight-bench/tests/solver_aiming.rs`
runs both aiming options of all four jobs against the reference under the
solver-outcome contract (emulation; Metal ignored), and a basin-hopping
`optimize` and a glass substitution with device aiming on 1, 2 and 8 rayon
workers: identical results, counters included. A truncated solve
(3 LM iterations, 2 local glass iterations) may miss a merit or violation
gate when it deviates no more than twice as much as the reference itself
moves under the two differences a device run brings: a perturbation of
the values (the reference on a copy with one curvature changed by 1e-9)
and the device's forward step (the reference with `fd_relative_step:
1e-6`). Such trajectories react to tiny perturbations: the nudged
reference differs by up to 2 % in the Double Gauss's merit after 3
iterations and by up to 52 % in the glass finalists' merits (anamorphic
sample; 1.5–3.5 % on the achromat, Double Gauss and relay), against the
merit gate of 0.33 %; and the reference with the device's step reproduces
the device's glass finalists to about 1 % (violations of infeasible
intermediate finalists 4e-5–1e-4 apart, against a gate of 6.7e-6).
