# MKLight Solves and Pickups

A **solve** makes one numeric parameter of an `OpticalSystem` follow from
other parameters or from a paraxial ray condition: "put the image at the
paraxial focus", "make the lens F/4", "this radius is minus that one". The
data model lives in `crates/mklight-core/src/model/solve.rs`; the solver is
`crates/mklight-trace/src/solves.rs`. Read `docs/CONVENTIONS.md` first.

## 1. Data model

```rust
pub struct OpticalSystem { /* … */ pub solves: Vec<Solve> }   // #[serde(default)], omitted when empty
pub struct Solve { pub target: ParamPath, pub kind: SolveKind }
```

* Each solve controls exactly one `ParamPath`, its **target**. A parameter
  can be the target of at most one solve.
* The stored model keeps whatever value the target had last; that value is
  only the starting point of the solve. `apply_solves` returns a **new**
  system in which every target is set, and the engine builds the `Snapshot`
  of that solved system. Snapshots are never modified.
* `OpticalSystem::solved_params()` (also `mklight_trace::solved_params`)
  lists the targets, sorted and deduplicated. The optimiser must exclude them
  from its variables, and the UI should show them as solve-controlled.
* `SolveKind` is internally tagged by `"type"` (snake case), like the rest of
  the model; `ParamPath` is tagged by `"param"`.

## 2. Paraxial conventions

Ray solves (marginal/chief height and angle, F/#) use the reference tracer's
first-order model (`mklight_trace::paraxial`, `FirstOrder`) at the **primary
wavelength**:

* The trace runs on the **unfolded, rotationally symmetric equivalent**:
  tilts, decenters and coordinate breaks are ignored (the report then carries
  a warning), thicknesses keep their signs, and mirrors flip the sign of the
  reduced index `N`.
* The **marginal ray** starts at the axial object point (parallel to the axis
  for an object at infinity) and is scaled by the aperture specification so
  that it passes the edge of the entrance pupil at `+y`. The **chief ray**
  passes through the centre of the stop and is scaled to the **maximum
  radial field** (CONVENTIONS §3), with the sign of a positive field. A
  system whose fields are all on axis has a zero chief ray, so chief-ray
  solves cannot be satisfied there (except trivially).
* Heights `y` are in mm on the **vertex plane** of a surface. The paraxial
  angle is the **slope** `u = dy/dz` (dimensionless, *not* degrees) after a
  surface, with `z` the local axis. After an odd number of mirrors light
  travels toward `−z`, so a beam converging toward the axis there has
  `u > 0`: a concave mirror at `y = 10` that focuses 200 mm in front of
  itself has `u = (0 − 10)/(−200) = +0.05`.
* The **image-space F/#** is `f'/EPD` with the rear focal length
  `f' = |N'|·|EFL|` (CONVENTIONS / `FirstOrder::image_f_number`).

## 3. Solve kinds

`j` below is the index of the target's surface. "Closed form" is the update
that solves the condition in one step when the ray scaling does not depend
on the target (§5).

| kind | target | condition | closed form |
|---|---|---|---|
| `marginal_ray_height { height }` | `thickness` of surface `j`, `1 ≤ j ≤ N−2` | marginal `y_{j+1} = height` | `t_j = (height − y_j)/u_j` |
| `chief_ray_height { height }` | same | chief `y_{j+1} = height` | same, chief ray |
| `marginal_ray_angle { slope }` | `curvature` or `paraxial_focal_length` of a powered surface `j` | marginal `u_j = slope` | `φ_j = (ω_{j−1} − N_j·slope)/y_j` |
| `chief_ray_angle { slope }` | same | chief `u_j = slope` | same, chief ray |
| `f_number { value }` | same | image-space F/# `= value` | `φ_j` from the affine dependence of `ω'_A` on `φ_j` |
| `pickup { source, scale = 1, offset = 0 }` | any parameter | `target = scale·source + offset` | itself |
| `position { from, to, distance }` | a thickness `t_j` with `from ≤ j < to` | `Σ_{i=from}^{to−1} t_i = distance` | `t_j = distance − Σ_{i≠j} t_i` |
| `center_of_curvature { surface }` | `curvature` of surface `j` | centre of curvature on the vertex of `surface` | `c_j = 1/d` |
| `edge_thickness { value, radius = None }` | `thickness` of surface `j`, `1 ≤ j ≤ N−2` | `t_j + sag_{j+1}(r) − sag_j(r) = value` | itself |

Details:

* **Marginal/chief height.** `height = 0` on the last thickness (surface
  `N−2`) puts the image surface at the paraxial focus: the solved thickness
  equals `FirstOrder::image_distance` (= BFL for an object at infinity). A
  thickness after which the ray is parallel to the axis (`u_j = 0`) cannot
  change the height and the solve fails with that reason. Chief height 0
  puts surface `j+1` on a pupil image (e.g. a dummy surface on the exit
  pupil).
* **Angles and power.** With `ω = N·u` the reduced slope, refraction is
  `ω_j = ω_{j−1} − y_j φ_j`. For a conic or asphere `φ_j = c_j (N_j − N_{j−1})`
  (so a mirror has `N_j = −N_{j−1}`), for an ideal lens `φ_j = |N_j|/f`. A
  surface where the ray has zero height cannot bend it (error). Note that
  with an image-space or working F/# aperture the image-space marginal
  slope is fixed by the aperture, so a marginal-angle solve on the last
  surface cannot change it; such a solve fails with "no sign change".
* **F/#.** With `A` the unit axial ray (`y₁ = 1`, `u₀ = 0`),
  `EFL = −1/ω'_A`, and `ω'_A` is affine in `φ_j` with slope
  `−y_{A,j}·∂ω'/∂ω_j` (a tangent y–nu trace from `j`). The solve keeps the
  sign of the EFL and sets `|EFL| = value·EPD/|N'|`. It may target any
  powered surface; the usual choice is the last powered one. It is rejected
  when the aperture is itself an F/# (`image_space_f_number`,
  `paraxial_working_f_number`), which already fixes it. When the entrance
  pupil depends on the target (a stop-size aperture or a finite object with
  the target in front of the stop), the update is iterated.
* **Pickup.** Any `ParamPath` from any other, including global ones
  (aperture value, fields, wavelengths, environment). The source may itself
  be solve-controlled; pickups of pickups are evaluated in dependency order.
  An `asphere` target beyond the stored coefficients extends the list with
  zeros. Explicit dependency cycles (`a ← b ← a`) are rejected by
  validation.
* **Position.** `from = surface 1, to = image` keeps the **total track**
  (`FirstOrder::total_track`) constant; any other `from` holds the distance
  from a reference vertex. Thicknesses are signed (CONVENTIONS §2), so after
  a mirror the distance is negative. `from` may be the object only for a
  finite object.
* **Centre of curvature.** `d` is the signed axial position of the vertex of
  `surface` relative to vertex `j`: `Σ_{i=j}^{r−1} t_i` when `r > j`, and
  `−Σ_{i=r}^{j−1} t_i` when `r < j`. Because thicknesses are signed and frames
  do not flip at mirrors, this is the position along the local `z` axis of
  `j` (on the unfolded axis when there are tilts). `c_j = 1/d`; the residual
  reported is `c_j·d − 1`. Example: a mirror 150 mm behind a finite object,
  solved to have its centre of curvature on the object, gets `R = −150` and
  images the object onto itself at `m = −1`.
* **Edge thickness.** The signed axial distance between surfaces `j` and
  `j+1` at radius `r`, from the full sags (conic and asphere terms,
  `SurfaceShape::sag`). With `radius = None`, `r` is the larger of the two
  surfaces' radii, each taken as the first that exists of: mechanical
  semi-diameter (`edge.mechanical_semi_diameter`), explicit aperture
  (bounding radius), fixed semi-diameter, automatic semi-diameter (traced by
  `SequentialTracer::semi_diameters` with default settings over all fields
  and wavelengths). Automatic semi-diameters depend on the thickness being
  solved, so that case is iterated to a self-consistent value. A radius
  outside a conic's domain is an error.
* **Planes and shapes.** A plane has curvature 0. A curvature solve on a
  `plane` is allowed and **turns it into a sphere** (`conic` with `conic = 0`),
  as `ParamPath::set` does. Ideal lenses (`paraxial`) have no curvature: use
  a `paraxial_focal_length` target. Curvature targets on coordinate breaks,
  on `dummy` surfaces and on the object and image surfaces are rejected.

## 4. Validation (`OpticalSystem::validate`)

`validate` (and so `Snapshot::new`) rejects:

* non-finite numbers in a solve; `f_number.value ≤ 0`; `edge_thickness.radius ≤ 0`;
* a target or source that does not exist or cannot be read (unknown surface,
  `conic` of a plane, `model_nd` of a catalog glass, …); an automatic
  semi-diameter can be a target (it becomes fixed) but can only be a source
  if a solve sets it;
* two solves on the same parameter, and a pickup of itself;
* a target of the wrong type for the kind (table above), e.g. a height solve
  on a curvature, a ray solve on the object or image thickness, an F/# solve
  on a thickness;
* power solves on surfaces without power (object, image, dummy, coordinate
  break) and F/# solves with an F/#-specified aperture;
* `position` with `from` not before `to`, the target outside `[from, to)`,
  or `from` the object at infinity; `center_of_curvature` on the surface's
  own vertex or involving an object at infinity; `edge_thickness` next to a
  coordinate break;
* **cycles of explicit dependencies** (below), with the solves involved
  named in the error.

## 5. Solver (`mklight_trace::solves`)

```rust
apply_solves(&sys, &lib) -> MkResult<(OpticalSystem, SolveReport)>
apply_solves_with(&sys, &lib, &SolveSettings, &JobContext)
apply_solves_configuration(&sys, &lib, k, &settings, &ctx)   // overrides of k, then solves
apply_solves_all(&sys, &lib, &settings, &ctx) -> Vec<(OpticalSystem, SolveReport)>
solved_params(&sys) -> Vec<ParamPath>
```

**Order.** A solve *explicitly* reads parameters: a pickup its source, a
position solve the other thicknesses in its range, a centre-of-curvature
solve the thicknesses between the two vertices, an edge-thickness solve the
shape parameters and semi-diameters of its two surfaces. Paraxial ray solves
*implicitly* read every parameter upstream of their target.
`OpticalSystem::solve_order` is a topological order of the explicit
dependencies in which, among the solves that are ready, the most upstream
target goes first (global parameters first; on one surface the shape before
the thickness before the semi-diameter; ties by declaration order). Explicit
cycles are errors; implicit ones (e.g. an upstream curvature that picks up
the focus thickness) are legal and are resolved by the sweeps below.

**One solve.** The condition `measured(x) = goal` is evaluated on a
candidate system (`Snapshot` → `ResolvedSystem` → `FirstOrder` at the
primary wavelength, plus traced semi-diameters only when needed):

1. *Closed form / fixed point.* The update of the table is applied and the
   condition re-measured. It is exact in one step when the ray scaling does
   not depend on the target; otherwise (F/#- or stop-sized apertures,
   paraxial-image-height fields, automatic semi-diameters, pickups of the
   target that feed back) it is repeated while the residual shrinks by at
   least 10 % per step.
2. *Root finding.* If the closed form is unavailable (zero height or slope),
   non-finite, or the iteration stalls, the residual is solved by
   **Brent's method** (inverse quadratic interpolation and secant steps
   safeguarded by bisection) in a smooth variable (`t`, `c`, or `1/f` for
   ideal lenses). The bracket expands geometrically in both directions from
   the current value (first step 1 % of the value, at least 0.01 mm for a
   thickness and 1e-5/mm for a curvature) and the nearest sign change is
   used. No bracket, an invalid system inside the bracket, or a "root" on a
   discontinuity (the residual check fails at the end) is an error that
   names the solve, its target and why the closed form was not used.
3. Every trial value is applied together with the solves that explicitly
   read the target (transitively), so a solve and its pickups are one 1-D
   problem (e.g. an F/# solve on `c₁` with `c₂ = −c₁`).

**Sweeps.** All solves are run in order; sweeps repeat until no target
changes by more than `tolerance` (relative, absolute below a per-parameter
floor: 1 mm for lengths, 1e-6/mm for curvatures). The report gives the
number of sweeps and the last change; failure to converge within
`max_sweeps` is an error.

**Convergence criteria** (`SolveSettings`, default `tolerance = 1e-12`,
`max_sweeps = 50`, `max_iterations = 200`, `closed_form = true`): a
condition holds when `|measured − goal| ≤ tolerance · scale`, with scale the
largest height of the ray (heights), the largest slope (angles), the goal (pickups), the F/#
itself, the sum of absolute thicknesses (position), 1 (centre of curvature),
or `max(|value|, |t|, 1 mm)` (edge thickness). `closed_form = false` uses
root finding alone, a cross-check of the closed forms in the tests.

**Report** (`SolveReport`): per solve (declaration order) the previous and
new value, final residual measured on the returned system, method
(`unchanged`, `closed_form`, `fixed_point`, `root_finding`) and evaluation
count; the evaluation order, sweeps, last change, `solved_params`, warnings
and a `RunInfo` (input snapshot hash, settings, `cpu-f64`).

**Jobs.** Deterministic (fixed order, no randomness, no threads apart from
the deterministic semi-diameter trace). Cancellation is checked before each
solve; progress is monotone and reaches 1 on success.

**Configurations.** Solves are applied per configuration **after** its
overrides (`apply_solves_configuration`). An override of a solve-controlled
parameter is overwritten by the solve and reported as a warning. The
returned system is the solved configuration: its base values are those of
configuration `k`, and its `configurations` list is kept unchanged.

## 6. JSON

Each solve is an object with a `target` (`ParamPath`) and a `kind`
(`SolveKind`). Surfaces are referenced by `SurfaceId`. These are the forms
the Swift model mirror should decode (tested in
`crates/mklight-trace/tests/solves.rs`):

```json
"solves": [
  {"target": {"param": "thickness", "surface": 2},
   "kind": {"type": "marginal_ray_height", "height": 0.0}},
  {"target": {"param": "curvature", "surface": 2},
   "kind": {"type": "marginal_ray_angle", "slope": -0.1}},
  {"target": {"param": "thickness", "surface": 1},
   "kind": {"type": "chief_ray_height", "height": 1.0}},
  {"target": {"param": "curvature", "surface": 1},
   "kind": {"type": "chief_ray_angle", "slope": 0.0}},
  {"target": {"param": "conic", "surface": 1},
   "kind": {"type": "pickup", "source": {"param": "conic", "surface": 2},
            "scale": 2.0, "offset": -1.0}},
  {"target": {"param": "thickness", "surface": 0},
   "kind": {"type": "position", "from": 1, "to": 3, "distance": 100.0}},
  {"target": {"param": "decenter_x", "surface": 1},
   "kind": {"type": "center_of_curvature", "surface": 3}},
  {"target": {"param": "thickness", "surface": 1},
   "kind": {"type": "edge_thickness", "value": 2.0, "radius": 8.0}},
  {"target": {"param": "curvature", "surface": 2},
   "kind": {"type": "f_number", "value": 4.0}}
]
```

(The list shows the syntax of every kind; it is not one valid system.)
Defaults: `pickup.scale = 1`, `pickup.offset = 0`, `edge_thickness.radius`
absent or `null` means "from the semi-diameters". `solves` is omitted from
the JSON when empty, so systems without solves serialise and hash as
before.

## 7. Tests

`crates/mklight-trace/tests/solves.rs`: marginal height 0 on the last
thickness equals `FirstOrder::image_distance`/BFL for every sample
(including the folded Newtonian and the finite relay) and under every
aperture specification at finite and infinite conjugates; F/# solves reach
the requested F/# to 1e-12 relative (lens, ideal lens, mirror, finite
object, pupil depending on the target), closed form against root finding;
marginal/chief angle and chief height solves against known first-order
answers (telecentricity, exit-pupil position); symmetric lens by a −1
curvature pickup, coupled with an F/# solve; chained pickups with offsets
and their order; cycle detection; edge thickness against sags from the
circle equation (explicit radius, fixed and automatic semi-diameters);
total track kept by a position solve; centre-of-curvature solves including
a mirror imaging its centre onto itself; per-configuration solving and the
override warning; serde round trips and the JSON above; validation errors;
clear failures for impossible conditions; determinism, cancellation and
progress; idempotence (solving a solved system changes nothing).
