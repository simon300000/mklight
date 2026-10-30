# Optimisation and Tolerancing (`mklight-optim`)

`mklight-optim` holds the optimisation numerics (batched least-squares
problems, Levenberg–Marquardt / dogleg trust regions with bounds and
augmented-Lagrangian constraints, differential evolution, Nelder–Mead,
CMA-ES, basin hopping, tabu search), the serialisable merit-function model
and tolerancing. This document covers how a merit function is evaluated on a
real lens: the tracer-backed operand evaluator, the GPU-batched evaluator,
the operand reference, the default merit function, derivatives, determinism,
tolerancing, measured throughput, and the engine's `optimize`,
`glass_substitution` and `tolerance` jobs.

## Pieces

```
MeritFunctionSpec ──┐                      ┌── TraceEvaluator (paraxial, real ray,
VariableSpec[] ─────┼─► OpticalProblem ──► │     wavefront, model + auto edges)
OperandEvaluator ───┘     (Problem)        ├── GpuTraceEvaluator (ray operands of a
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
* `GpuTraceEvaluator` (`merit::gpu_eval`) traces the ray-based operands of
  many candidates on a GPU in one dispatch and reads them with the same code;
  `GpuTraceEvaluator::composite` puts it in front of a `TraceEvaluator` for
  everything else (see GPU-batched evaluation).
* `BuiltinEvaluator` combines `TraceEvaluator`, `MtfEvaluator` and
  `SolverEvaluator`. Engine optimise and tolerance jobs use this full set,
  also after GPU ray evaluation. `Custom` operands still need a plug-in.
  `CompositeEvaluator` routes each operand to its first supporting evaluator.

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
equal to the power ratio). It uses the explicit ghost solver, with fixed
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
it shortens the back focus by 0.12 mm and drops the weakest MTF from 0.169 to
0.050. The joint minimum reaches 0.174 from the nominal focus or from 0.1 mm
of defocus; the field-weighted bounds settle at 0.063. The optimisation
test checks both. These Cooke values are physical: on axis the
polychromatic MTF at 30 cycles/mm is 0.69, but the 14° field's astigmatism
holds it near 0.17 (T 0.169, S 0.180; geometric 0.184/0.174; a 128 grid
gives 0.173/0.178). Far from focus (0.3 mm, about 4 waves at F/4) every
field oscillates through zeros and the minimum is multimodal, so LM from
there stops in a local maximum near 0.016.

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
* **f32 GPU operands** (`GpuTraceEvaluator`, see GPU-batched evaluation):
  `η = 1e-3` / `3e-3` for ray positions, `1e-4` / `1e-3` for wavefront
  operands.

Measured largest relative column error `‖J_j − R_j‖/‖R_j‖` on the Cooke
triplet (curvatures 1, 3, 6, an air space and the back focus; all fields and
wavelengths; `R` = f64 central differences, `tests/gpu_eval.rs`):

| operands, evaluator | forward, `√ε` | forward, automatic | central, automatic |
|---|---|---|---|
| spot, f64 reference | 4e-7 | 4e-7 (`√ε`) | 2e-9 (`∛ε`) |
| wavefront, f64 reference | 2.7e-5 | 8e-7 (`1e-6`) | 8e-9 (`1e-4`) |
| spot, GPU f32 | 7 | 4.1e-4 (`1e-3`) | 6.1e-5 (`3e-3`) |
| wavefront, GPU f32 | 0.11 | 8.7e-5 (`1e-4`) | 4.7e-6 (`1e-3`) |

The GPU's best steps sit where the f32 noise (`~δ/h`) meets the truncation
error: spot columns measured 3.8e-2, 3.9e-3, 4.1e-4, 5.3e-4 at forward
`η = 1e-5, 1e-4, 1e-3, 3e-3`, and 1.7e-4, 6.1e-5, 2.1e-5 at central
`1e-3, 3e-3, 1e-2`. Errors of 1e-4 do not slow LM down measurably: 8 GPU LM
iterations on the triplet reach the merit of the same run on the reference
(within 5 %, tested).

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
its own. Both are tested on the Cooke triplet with a back-focus compensator.
Without compensators every trial is a single evaluation, so the evaluator
receives a whole batch of perturbed systems in one call (one GPU dispatch
with `GpuTraceEvaluator`). With compensators, each trial runs its own small LM
in parallel. The engine job `tolerance` wraps both drivers (see Engine jobs).

## End-to-end tests (`tests/trace_optim.rs`)

| test | system | result |
|---|---|---|
| singlet best form | f = 100, F/10, N-BK7, start equi-convex; vars c₁, c₂, back focus; EFL = 100 constraint; on-axis d-light RMS spot | shape factor `q = 0.730`, thin-lens best form `2(n²−1)/(n+2) = 0.740`; merit ÷150; EFL within 1e-6 mm |
| doublet achromatisation | N-BK7/F2, start with R₂ = −60, R₃ = −300 (focal shift > 0.5 mm) | F–C focal shift 2e-10 mm (test: < 1e-6), EFL = 100 ± 1e-6, R₂ → −40.2 mm, R₃ → −375 mm (the sample achromat, bent for minimum spherical aberration, has −42.5 / −410) |
| Cooke triplet | default spot merit (3 fields × 3 λ), 6 curvatures + back focus, EFL constraint, glass/air centre and auto-SD edge bounds | LM (40 iterations) lowers the merit by 35 %; seeded basin hopping is bit-reproducible and does not increase it; all constraints hold |
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
run at the full rate. These are single timed runs. The GPU benchmark
(GPU-batched evaluation) reports the best of repeated runs, so its CPU column
is up to 1.5× higher for the same evaluator. The GPU evaluator packs every
candidate (10 µs) instead of patching parameter vectors into a packed base:
the packer's representability checks and double-float residuals depend on
the values.

## GPU-batched evaluation

`GpuTraceEvaluator` (`merit::gpu_eval`) evaluates the ray-based operands of
a batch of candidates on any `mklight-gpu` backend (`ComputeBackend`: the
Metal GPU, or the CPU-f32 emulation, which produces the same bits). It is
built for a system, `GpuTraceEvaluator::new(backend, cpu, system,
settings, ctx)`, and `composite()` puts it in front of the CPU
`TraceEvaluator`, which computes every operand the GPU does not.

**Accelerated operands** (`gpu_supports`): `ray_aberration`, `rms_spot`,
`spot_centroid`, `real_ray` (every quantity except the incidence angle),
`distortion`, `lateral_colour`, and, with `GpuEvalSettings::wavefront`
(default on), `wavefront_aberration` and `wavefront_rms`. Paraxial, model,
relative-illumination, chief-ray-angle and real-F/# operands stay on the
CPU.

**Per batch** (sub-batches of `GpuEvalSettings::batch`, default 512
candidates):

1. *Host, in parallel over candidates (f64).* Each candidate gets its
   `SequentialTracer`, exactly as in `TraceEvaluator`. Its operands are
   planned with the same code: chief rays, pupil samplings (half pupils
   mirrored on asymmetric candidates), single rays without a group, and real
   rays to a surface. Every planned ray is generated with the reference
   field/pupil mapping and ray aiming, and the candidate is packed
   (`PackedSystem`, all wavelengths, asphere radii from the qualified start).
2. *Device.* The packed candidates are uploaded as one system set. Their
   rays, each selecting its system by index, are traced in one request per
   (clipping, last surface): unclipped chief rays, clipped pupil rays to the
   image, and real rays to other surfaces. Rays enter in double-float
   (`RayBatch::push_f64`), with radiometry off.
3. *Host.* The candidate caches are filled from the outputs. The operands
   are then read by `TraceEvaluator`'s own code (`Candidate::value`). OPD is
   formed on the host from the GPU's double-float optical path, against the
   reference sphere of the GPU's own chief ray (CONVENTIONS §12).

The sub-batches are pipelined: while the GPU traces one, the host prepares
the next and reads back the previous one.

**Determinism.** A ray's GPU result depends only on the ray and its system:
chunking changes no bit (`docs/GPU.md` §6), and host work runs in a fixed
order. A candidate therefore gets the same bits alone, in any batch, and
under any sub-batch split (tested), and repeated runs are bit-identical.

**Precision and fallback** (CONVENTIONS §7, §12). The constructor qualifies
the backend for the system with `xval::qualify`: the qualification sample,
a third of the budget, the evaluator's clipping and radiometry off (thin-film
coatings do not prevent it). If the backend fails, the evaluator is inactive
and the CPU reference computes everything. Candidates the packer cannot
represent (more than 16 wavelengths, asphere terms beyond A22, values outside
the normal f32 range, for example a conic of 1e-300) also go to the CPU
reference, and so does a sub-batch whose upload or dispatch fails. All of
these are counted in `GpuEvalStats::cpu_candidates`. Two kinds of real-ray
values are always traced on the CPU, because the qualification does not
sample them: rays to a passive surface (coordinate break, dummy), whose
reported point can lie hundreds of mm from the frame origin (a fold's
coordinate break exceeded the position budget on a perturbed Newtonian),
and points farther than `REAL_RAY_F32_LIMIT_MM` (100 mm) from their
surface's origin, where rounding the point to f32 alone uses about a third
of the budget. A cancelled or failed evaluation releases the system sets of
its unfinished sub-batches on the backend.
`GpuTraceEvaluator::record(&mut RunInfo)` writes the qualification decision
(`Qualification::record`) and the counters (`settings["gpu_evaluation"]`).
`requalify(system)` checks a later design, for example the end of an
optimisation, which may have moved far from the qualified start.

**Agreement with the f64 reference** (`tests/gpu_eval.rs`). The default spot
and wavefront merits plus every other accelerated kind make 521 operands.
They are evaluated on each of the seven samples (the Newtonian's mirrors
and fold, the finite-conjugate relay included), an x-decentered copy
(mirrored half pupils), a bent copy, two tilted and decentred copies and six
strongly perturbed candidates (curvatures ±25 %, thicknesses ±30 %), with the
emulation backend and on the M2 Pro (the two are bit-identical). The worst
value over all of them is 0.14 of its budget (the Newtonian's distortion).
The largest |GPU − CPU| on the four refracting samples' unperturbed
candidates:

| system | ray aberration (mm) | real ray (mm, cosines) | RMS spot (mm) | OPD (waves) | distortion (%) | worst, fraction of budget |
|---|---:|---:|---:|---:|---:|---:|
| singlet | 5.7e-7 | 4.3e-7 | 8.0e-8 | 5.3e-5 | 4.9e-6 | 0.022 |
| Cooke triplet | 1.1e-6 | 1.0e-6 | 7.4e-8 | 1.1e-4 | 4.8e-6 | 0.042 |
| Double Gauss | 5.5e-7 | 5.3e-7 | 1.1e-7 | 9.0e-5 | 2.5e-6 | 0.020 |
| Petzval asphere | 2.7e-7 | 1.7e-7 | 3.5e-8 | 5.3e-7 | 2.3e-11 | 0.013 |

The budget per kind is twice the position budget (4e-5 mm) for differences
of two points, 2e-6 for direction cosines, 0.5 % or 2e-5 mm for RMS spots,
and 5e-3 waves for OPD.

**Optimisers on GPU evaluations.** Finite-difference Jacobian columns are one
batch (n or 2n candidates), and DE / CMA-ES populations and basin-hopping
screens are one batch per generation. The f32 noise needs the larger FD steps
of Derivatives. On the Cooke triplet, 8 LM iterations on the emulated GPU
went from 3.97e-4 to 2.67e-4, the same as on the reference. Seeded DE and
CMA-ES runs were bit-reproducible, and every result re-evaluated on the f64
reference to within 1e-3 of its GPU merit.

**Throughput**: `cargo test -p mklight-optim --release --test gpu_bench --
--ignored --nocapture --test-threads=1`. The system is the Double Gauss f50
F/2 sample (default merit, 8 curvatures and the back focus as variables)
on an Apple M2 Pro (12 rayon threads, Metal). Each figure is the best
evaluations/s over ~0.3 s at a batch size:

| batch | spot 3 × 6 (171 rays/cand.): CPU / GPU evals/s | wavefront 3 × 6 | spot 6 × 12 (657 rays) | spot 12 × 24 (2601 rays) |
|---:|---|---|---|---|
| 1 | 5 703 / 1 956 (0.34) | 5 954 / 2 048 (0.34) | 3 191 / 1 542 (0.48) | 1 098 / 806 (0.73) |
| 4 | 15 509 / 4 878 (0.31) | 17 824 / 5 561 (0.31) | 7 271 / 3 507 (0.48) | 2 179 / 1 766 (0.81) |
| 16 | 27 760 / 11 910 (0.43) | 32 673 / 13 162 (0.40) | 8 936 / 6 952 (0.78) | 2 453 / 2 800 (**1.14**) |
| 64 | 33 257 / 22 183 (0.67) | 38 576 / 25 383 (0.66) | 10 359 / 9 954 (0.96) | 2 732 / 3 390 (**1.24**) |
| 256 | 36 464 / 27 942 (0.77) | 42 581 / 30 634 (0.72) | 10 626 / 11 224 (**1.06**) | 2 861 / 3 484 (**1.22**) |
| 1024 | 35 750 / 30 540 (0.85) | 42 778 / 36 617 (0.86) | 10 716 / 12 718 (**1.19**) | 2 767 / 4 218 (**1.52**) |
| 4096 | 35 410 / 32 435 (0.92) | 41 710 / 37 329 (0.89) | 10 458 / 13 529 (**1.29**) | 2 680 / 4 029 (**1.50**) |

(GPU/CPU ratio in parentheses; measured with the machine otherwise idle,
±10 % run to run. Sub-batches of 128 or 2048 instead of 512 changed the
large-batch figures by less than that.)

**Crossover.** It depends on the rays per candidate more than on the batch
size:

* Default merit (3 rings × 6 arms, half pupil, 171 rays per candidate): there
  is **no crossover**. The GPU path reaches 0.9× the CPU at 1024–4096
  candidates and is 3× slower for single LM trial points.
* 6 × 12 (657 rays): crossover at about **256** candidates, 1.3× at 4096.
* 12 × 24 (2601 rays): crossover at about **16** candidates, 1.5× from 1024 on.

The CPU f64 reference is efficient: one ray through the Double Gauss costs
~0.7 µs, and 12 threads trace 5–6 M rays/s. The GPU traces the same rays in
~5 µs of device time per default-merit candidate at large batches (14 µs
averaged over all batch sizes, launch overhead included), so the device is
not the limit. The limit is the per-candidate host work, which both paths
pay or which the GPU path adds. Measured serially per Double Gauss
candidate:

| work | µs |
|---|---:|
| `Snapshot` (content hash) + `ResolvedSystem` + `SequentialTracer` | 27 (17 of them the hash) |
| CPU trace of the 162 pupil rays (CPU path only) | 119 |
| packing (`PackedSystem::pack` + validation; GPU only) | 10 |
| generating the 171 rays (GPU only) | 3.5 |

The operands are read the same way on both paths (`value_with`, a few µs).
With the GPU path's extra allocation, the serial upload / concatenation /
unpacking between pipelined sub-batches, and the 4 efficiency cores, the
GPU path ends up about even with the CPU at 171 rays. It pulls ahead in
proportion to the rays a candidate traces. To speed it up further, move ray
generation and the operand reductions (centroids, RMS) onto the device, and
skip the snapshot hash for transient candidates (`mklight-core`).

**Use.** `gpu_beats_cpu(rays_per_candidate, batch)` encodes the crossover,
conservatively between the measured points: at least 2600 rays per
candidate and 16 candidates per batch, or 650 rays and 256 candidates.
`gpu_rays_per_candidate` counts the rays the evaluator would trace for a
system and operand list (its own plan). The engine jobs' `backend: "auto"`
uses the engine's GPU device only when that rule says the GPU is faster for
the run (see Engine jobs), so the default merit's LM, DE and CMA-ES runs stay
on the CPU. The gain comes from dense samplings (wavefront merits for
MTF-limited designs, many fields, many configurations) in batched work: FD
Jacobians with many variables, DE / CMA-ES populations, uncompensated
tolerancing trials. Small single-point LM steps are faster on the CPU.

## Engine jobs: `optimize` and `tolerance`

`mklight-engine` registers both as analyses (`analysis::optim`, category
`optimization`). They run on the scheduler like any analysis, with
`AnalysisRequest { analysis: "optimize", settings, .. }`, settings
normalised against their schema, and a `RunInfo` whose `seed` is
`settings.seed`. The shapes below are meant to be copied into
`docs/RESULTS.md`. Numbers follow CONVENTIONS §11: NaN is `null`, ±∞ is
`"inf"` / `"-inf"`. `ParamPath`, `VariableSpec`, `Operand`,
`MeritFunctionSpec`, `DefaultMeritSpec`, `ToleranceOperand` and
`Compensator` use their serde forms (for example
`{"param": {"param": "curvature", "surface": 3}, "lower": 0.0}`).

**Progress and interruption.** Progress is reported once per optimiser
iteration (generation, round) with the stage text
`"<algorithm> <iteration>: merit <best merit>"`, and once per Monte-Carlo
batch. Cancelling the job context stops the optimiser between evaluations,
and `Analysis::run` returns the best point so far, ranked by feasibility,
with `"status": "cancelled"`. The scheduler, however, ends every cancelled
job `Cancelled` and drops its result, because a superseded result is stale.
Front ends use `Engine::stop_job`, `mk_job_stop` (ABI 1.2), Python
`Job.stop()`, or the CLI's first Ctrl-C to keep the result. Stop has its own
job token: even a queued request is retained until the job starts, and the
job ends normally (`Done`) with `status: "stopped"` and `stopped: true`.
A second CLI Ctrl-C cancels. Cancel/supersession still discards the result.
The legacy `stop_key` / `analysis::optim::request_stop(key)` remains supported;
one running job holds each key. A stopped sensitivity keeps completed
operands; stopped Monte Carlo keeps completed trials.

**Never worse than the start.** The job evaluates the start and the
optimiser's answer on the CPU f64 reference and returns the better of the
two, feasibility first, then merit (`ScalarEval::better_than` with
`feasibility_tolerance`). When the optimiser found nothing better,
`"improved": false`, the result is the start design and a warning says so.
The global methods never evaluate the start itself, and a GPU-driven search
ranks on f32 values, so this check is needed for them.

**Reported numbers.** The `start` and `final` merits, the operand table and
the constraint check are always computed on the CPU f64 reference, whichever
evaluator drove the search. `backend` says what evaluated the search
(`evaluated_on`, `precision`), with the GPU qualification at the start and
at the final design. `RunInfo.warnings` names the search evaluator or the
reason a GPU was not used. Tolerance statistics use the search evaluator;
`reported_on` names it there. `RunInfo.backend` and
the envelope's `backend` record the backend that evaluated the search after
fallback, using the same ids as other analyses (`gpu:metal`, `cpu_reference`).
The device comes from the engine's lazily opened `GpuProvider`; its
`GpuDevice::qualify` caches the start and final qualifications per snapshot
and library. `GpuTraceEvaluator::with_qualification` reuses that verdict.
An explicit GPU request bypasses the automatic workload crossover.

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
Measured on the seven samples with the default settings, every method
(`lm`, `dogleg`, `de`, `cma_es`, `nelder_mead`, `hammer`) improved every
sample and ended feasible.

**Applying the result.** `result.system` is the optimised `OpticalSystem` in
model JSON. The app applies it with `Session::set_system`, which creates a new
undoable snapshot whose hash equals `result.system_hash` (tested).

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
| `options` | object, `{}` | the algorithm's settings (`LsqSettings`, `DeSettings`, `CmaEsSettings`, `NelderMeadSettings`, `HammerSettings`), laid over the job's defaults: the library defaults except `constraints.max_inner_iterations` 10 (LM, dogleg, hammer's `local`), `init_spread` 0.03 (DE) and `sigma0` 0.05 (CMA-ES) |
| `max_iterations` | integer or null, `null` | iterations (`lm`, `dogleg`, `nelder_mead`), generations (`de`, `cma_es`) or rounds (`hammer`); `null`: 200, 200, 2000, 100, 200, 20 |
| `fd_scheme` | `forward` \| `central`, `forward` | finite differences of LM / dogleg (and hammer's local runs) |
| `fd_relative_step` | number > 0 or null, `null` | `null`: chosen per operand and evaluator (Derivatives); 0 or a negative step is rejected |
| `seed` | integer, `0` | seed of `de`, `cma_es`, `hammer` |
| `backend` | `auto` \| `cpu` \| `gpu` \| `gpu_emulation`, `auto` | merit evaluation. `gpu` uses the GPU device the engine selected for the analysis (`AnalysisInput::device` holding the provider's `GpuDevice`), or the CPU reference with a warning when there is none. `auto` uses that device only when `gpu_beats_cpu(rays per candidate, batch)` holds for the run (GPU-batched evaluation, Use): rays counted by `gpu_rays_per_candidate` for the merit's configuration-0 operands on the start system, batch = the FD Jacobian (n, or 2n central) for `lm` / `dogleg`, the population for `de` (default max(10n, 15)) and `cma_es` (default 4 + ⌊3 ln n⌋), 1 for `nelder_mead`, `candidates_per_round` for `hammer`. The decision is in `backend.workload`. A GPU provider lists `optimize` and `tolerance` in its capabilities so that the engine hands them its device. `gpu_emulation` requires the selected `gpu:cpu_f32` provider (validation, tests) |
| `gpu_batch` | integer, `512` | candidates per GPU dispatch (`GpuEvalSettings::batch`) |
| `feasibility_tolerance` | number, `1e-6` | total normalised constraint violation regarded as feasible: best-so-far ranking and `final.feasible` |
| `stop_key` | string or null, `null` | key for `request_stop`; a key held by a running job is rejected |

### `optimize` result

```jsonc
{
  "status": "converged" | "finished" | "stopped" | "cancelled",
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
              "workload": {"rays_per_candidate": 38, "batch": 7, "gpu_faster": false},
              "qualification": {/* xval::Qualification at the start; GPU only */},
              "final_qualification": {/* at the final design; GPU only */},
              "gpu_stats": {"gpu_candidates": 0, "cpu_candidates": 0, "rays": 0,
                            "dispatches": 0, "device_seconds": 0.0}},
  "merit_spec": {/* the MeritFunctionSpec used, generated operands included */},
  "system": {/* OpticalSystem, model JSON */},
  "system_hash": "…64 hex digits…"
}
```

`status` is `converged` when the optimiser met a convergence test, and
`finished` when it stopped on a budget or a failure (see `termination`).
Buffer 0 is `merit_history` (f64, shape `[iterations, 4]`: iteration,
evaluations, merit, best merit). The search merit (`search`) equals `final`
on the CPU reference, and differs from it by the f32 noise on a GPU. The
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
| `tolerances` | array, `[]` | `ToleranceOperand`s (`param`, `minus`, `plus`, `distribution`, `label`); empty: a basic set, ±0.05 mm on every glass and air thickness except the back focus and ±0.5 % on every curvature of a curved optically active surface. A parameter the system does not have (here and in `compensators`) is rejected |
| `compensators` | array, `[]` | `Compensator`s (`param`, `range` around nominal, `scale`), re-optimised by LM for every trial |
| `criterion` | object or null, `null` | the criterion `MeritFunctionSpec`; `null` generates it with `DefaultMeritSpec::generate_for_tolerancing` from `default_merit` |
| `default_merit` | object, the default merit without thickness boundaries | generator settings, laid over that default |
| `mode` | `sensitivity` \| `monte_carlo` \| `both`, `both` | what to run |
| `trials`, `seed`, `batch` | `100`, `0`, `16` | Monte Carlo: trial `i` draws from Philox stream `TOLERANCE \| i` of `seed` |
| `threshold` | number or null, `null` | criterion value for the yield |
| `compensator_iterations` | integer, `20` | LM iterations per trial |
| `backend`, `gpu_batch`, `stop_key` | as for `optimize` | |

Without compensators, every trial is a single evaluation, and a batch of trials
goes to the evaluator as one batch (one GPU dispatch).

### `tolerance` result

```jsonc
{
  "status": "complete" | "stopped" | "cancelled",
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
    "mean": 4.2e-4, "std_dev": 1.1e-5, "min": 4.0e-4, "max": 4.6e-4,
    "percentiles": [{"percentile": 50.0, "value": 4.2e-4}, …80, 90, 95, 98],
    "yield_fraction": 0.93 | null, "threshold": 4.5e-4 | null,
    "compensator_mean": [42.25], "compensator_std": [0.04],
    "cancelled": false,                              // true: statistics of the completed trials
    "criteria": [/* per trial, "inf" for failed trials */]
  },
  "compensators": [/* the Compensators */],
  "criterion_spec": {/* MeritFunctionSpec */},
  "backend": {/* as for optimize; no final_qualification */}
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
| cancel and stop | cancelled after two LM iterations: `cancelled`, best-so-far below the start; `stop_key` through the scheduler: `Done`, `stopped`, improved; the key is unregistered afterwards |
| tolerance | 6 tolerances, back-focus compensator: ranked sensitivities, compensator moves, Monte Carlo identical for a seed, different for another |
| GPU backends | `gpu_emulation` everywhere and `gpu` (Metal) when present: GPU precision, both qualifications pass, reported on the reference, merit decreases; `auto` with a device stays on the CPU for LM on a two-field, one-wavelength default merit (38 rays, batch 7) and uses the GPU for a 12 × 24 merit with DE populations of 16 |
| global methods | `cma_es`, `de`, `nelder_mead` with the defaults on the singlet and Cooke triplet: improve, feasible, EFL within 1e-4 relative by the penalty; a DE run that cannot improve returns the start (`improved: false`, same snapshot hash) with a warning |
| defaults feasible | `{}` on the Cooke triplet (the 1 mm glass minimum relaxed to 0.99997, warned), the Petzval and the relay: feasible, improved, EFL held to 1e-6 mm |
| bad input | `fd_relative_step` 0 or negative, tolerances or compensators on a missing surface, a `stop_key` in use → `InvalidSettings` |
| stopped sensitivity | a compensated sensitivity of 30 tolerances stopped after its first batch: `stopped`, 15 operands kept, `cancelled: true` |

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
`hold_efl`, `fd_scheme`, `fd_relative_step`, `seed`, `backend`, `gpu_batch`,
`feasibility_tolerance` and `stop_key` are those of `optimize`. Empty
continuous variables select the curvatures of the curved surfaces and the
back focus; explicit variables may be any `ParamPath`. The re-optimisation is
`optimize`'s LM: the same merit, augmented-Lagrangian constraints, FD steps
and evaluator. `hold_efl` holds the input EFL as a constraint unless the
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
identical for any number of Rayon workers. Progress reports the number of
re-optimisations, the best merit and the best combination's glasses. Stop
(`Engine::stop_job` or `stop_key`) ends the search and returns the ranking
so far, including re-optimisations cut short (`local_termination:
"cancelled"`); Cancel discards the job. `max_seconds` covers the search only,
not setup, backend qualification or the final ranking.

### Cost and benchmark

Re-optimisation dominates. For C combinations, n continuous variables and
I LM iterations, forward differences cost about `C · (1 + I · (n + 1))`
merit evaluations, plus rejected steps and constraint updates (central
differences: 2n). One search iteration's combinations run in parallel. When
the backend qualifies, all re-optimisations share the qualified GPU
evaluator, each FD Jacobian batched and split at `gpu_batch`; `auto` decides
on the Jacobian's workload, as in `optimize`. Qualification is at the input
and at the best result; the ranking is on the CPU reference. The final
ranking adds one CPU merit evaluation per kept combination.

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
  as much as the CPU trace it replaces (see Throughput). Device-side ray
  generation and reductions would remove it.
* The GPU evaluator traces chief-ray angles, incidence angles, real working
  F/# and relative illumination on the CPU.
* The engine, C ABI, Python and CLI expose Stop; the native app still needs
  to wire its Stop button to `mk_job_stop`.
* The GPU evaluator is qualified at the start and, for the report, at the
  final design; candidates in between are not requalified. The agreement
  tests cover strongly perturbed candidates of every sample (worst 0.14 of
  the budget), and real rays on passive surfaces or far from their surface's
  origin, the one case found over budget, are traced on the CPU.
