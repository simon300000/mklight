# MKLight Geometric Analyses (`mklight-analysis`)

This crate turns the CPU f64 reference tracer (`mklight-trace`) into the
geometric image-quality analyses a front end shows: spot diagrams, fans,
field curvature and distortion, Seidel sums, relative illumination,
encircled energy, geometric MTF, best focus, the layout scene and the
first-order report. Read `docs/CONVENTIONS.md` first; the module docs in
`crates/mklight-analysis/src/*.rs` hold the exact formulas.

## 1. Shape of an analysis

* **Pure functions.** `analysis(&SequentialTracer, &Settings) -> MkResult<Result>`.
  The tracer owns the `ResolvedSystem` (and so the `Snapshot`), its paraxial
  data and its prepared fields; `mklight_analysis::tracer_for(system)` builds
  one against the built-in material library.
* **Settings** are serde structs with `Default` and `#[serde(default)]`, so a
  partial JSON object is valid and unknown keys are ignored.
* **Results** are serde structs carrying `run_info: RunInfo` (engine version,
  snapshot hash, analysis name, settings JSON, backend `"cpu-f64"`, precision
  `f64_reference`, the seed when a random pupil pattern was used, and the
  tracer's plus the analysis' warnings).
* **Jobs.** Every analysis that traces more than a handful of rays has a
  `*_job(…, &JobContext)` variant: it checks cancellation between blocks
  (returning `Err(Cancelled)`, never a partial result) and reports monotone
  progress that ends at 1. The plain function runs the job with a fresh
  context.
* **Determinism.** Batches use `mklight_trace::trace_grid_job` (fixed output
  order); all sums are compensated (Neumaier) and run in a fixed order;
  RMS values are two-pass (`Σw|q − μ|²`, never `E[q²] − μ²`) on points taken
  relative to the chief ray, so the large common image height is removed
  before any sum. Random pupils use Philox (CONVENTIONS §6).
* **Selections.** `FieldSelection` (`{"type":"all"}`, `{"type":"index","index":i}`,
  `{"type":"list","indices":[…]}`) and `WavelengthSelection` (adds
  `{"type":"primary"}`) resolve to ascending, de-duplicated index lists;
  invalid indices are errors. `all` also adds missing axis fields outside
  rotational symmetry: +x and ±y, and -x when x reflection is unavailable.
  These fields have analysis-local indices after the original system fields,
  unit weight and no pupil compression. Their coordinates are reported;
  the physical snapshot and its hash are unchanged. Explicit index/list
  selections use only the requested fields.
* **Non-finite numbers** (`util::nf`): `±∞` are written as `"inf"`/`"-inf"`
  (the core `serde_util` convention) and NaN, which means *undefined* (a
  vignetted field, a failed ray), as `null`. Reading accepts numbers, `null`,
  `"inf"`, `"-inf"` and `"nan"`. See CONVENTIONS §11.

## 2. Analyses

| function | module | method in one line |
|---|---|---|
| `spot_diagram` | `spot` | pupil pattern traced per field × wavelength; centroid, RMS/GEO about centroid and chief, weighted polychromatic stats, vignetted fraction, Airy radius `1.22 λ N_w`, through-focus by transfer along the rays |
| `ray_fan`, `opd_fan` | `fans` | tangential `(0, P_y)` and sagittal `(P_x, 0)` fans; `εy`/`εx` about the primary (or same-wavelength) chief; OPD from the exit-pupil reference sphere of `mklight-trace` |
| `field_curvature` | `curvature` | generalised Coddington (2×2 pencil matrices) along the real chief ray, labelled by the image frame, cross-checked by the differential pencil of the real trace (analytic tangent, `mklight_trace::diff`); distortion vs the paraxial chief height, `f·tan θ`, `f·θ` |
| `grid_distortion` | `curvature` | chief rays of a square field grid vs their paraxial image points |
| `seidel` | `seidel` | per-surface `S_I…S_V`, `C_L`, `C_T` from the paraxial marginal and chief rays (aspheric terms included), Petzval sum and radius, `W040…W311`, third-order transverse aberration |
| `longitudinal_aberration` | `seidel` | on-axis zonal rays: crossing with the chief ray vs the primary paraxial focus |
| `chromatic_focal_shift` | `seidel` | paraxial focus of a wavelength sweep (copy of the system with the sweep as its wavelengths) |
| `lateral_colour` | `seidel` | real chief heights per wavelength vs field |
| `relative_illumination` | `illumination` | projected solid angle of the real exit pupil seen from the chief-ray image point, by real-ray aiming of the rays through that point (§2.4) |
| `footprint` | `illumination` | hit points on any surface, clipped points flagged |
| `encircled_energy` | `energy` | weighted distance order statistics (circle and square) |
| `geometric_mtf` | `energy` | `|Σ w e^{−2πiν q·e}|/Σw` along the tangential / sagittal direction |
| `best_focus` | `energy` | Brent minimisation of the field-weighted mean polychromatic RMS radius |
| `layout` | `layout` | global polylines (profiles, rings, element outlines, stop, image, axis, rays) and compact buffers |
| `system_data` | `system_data` | first-order table with keys, labels and units |

### 2.1 Spots, defocus and best focus

Image positions are in the **image surface's local frame**. Each arrived ray
stores its offset `q` from the primary chief ray and its slope difference
`Δt = (d_x, d_y)/d_z − t_chief`, so the point at defocus `δ` is `q + δ Δt`:
exact for a plane image surface, first order for a curved one. On a curved
image surface `best_focus` always warns, and `spot_diagram` (defocus or
through-focus), `encircled_energy` and `geometric_mtf` warn when a defocus
is set. The RMS radius is then the norm of an affine function of `δ`, so the
best-focus merit (mean of per-field RMS radii) is convex and its minimiser
lies between the per-field minimisers; that interval is Brent's default
bracket. The closed-form minimiser of the mean *square* merit is reported as
a cross-check, and the merit itself is evaluated from the rays (two-pass)
because the quadratic form cancels near a sharp focus. With the chief-ray
reference, fields whose primary chief ray failed are skipped with a warning.
A minimum at the edge of the search range (given or automatic) and an
unconverged search (`max_evaluations` reached) are reported as warnings.

Pupil patterns (`util::PupilPattern`, density `rings`): `hexapolar`
(`1 + 3r(r+1)` points), `square` (pitch `1/r`, clipped to the disk),
`random` (`1 + 3r(r+1)` uniform points, `ρ = √u₁`, Philox stream 0) and
`dithered` (one jittered point per square cell, stream 1).

### 2.2 Field curvature

The pencil around the real chief ray is carried as a ray family
`x_⊥ = X a`, `δk_⊥ = T a` (2×2 matrices) in a basis transverse to the ray.
Its wavefront curvature is `C = −T X⁻¹`, and `(X, T)` stays finite both for
collimated pencils and at a focus. The steps are:

* transfer: `X ← X + dT`;
* surfaces: second-order phase matching on the surface,
  `n' P'ᵀC'P' = n PᵀCP + (n' k'·N − n k·N) K`, where `K` is the second
  fundamental form of the actual sag graph: in an orthonormal tangent basis
  E, `K_ij = N_z E_i,xyᵀ H_sag E_j,xy`. Thus all Cartesian mixed derivatives
  are retained, including toroids, biconics and freeforms. `P = UᵀE`
  projects the tangent basis. For a DOE add `-q H_phase,surface` to the
  curvature-matching right hand side (`q=mλ/(2π)`, λ in mm). The covariant
  phase Hessian is `E_xyᵀ H_Φ E_xy + (∇Φ·N)K`; its connection term matters
  on a curved base. In ray form the matching numerator is
  `n Pᵀ T + (q H_phase,surface − gK) δs`;
* ideal lenses: the matrix form of `τ' = (n/n')τ − s/f`.

This makes no principal-plane assumption, so it holds for skew chief rays,
tilts about any axis and compound tilts. In a principal plane it reduces to
Coddington's equations. The foci are the eigenvalues of the focal-distance
matrix `F = −X T⁻¹`. A focus is labelled *tangential* when its collapsing
direction is closer to the image frame's local `y` (CONVENTIONS §2), and
*sagittal* otherwise. The labelling therefore follows the image frame, not
each surface's plane of incidence. On a mirror tilted about `y`, the
mirror's in-plane focus `(R/2) cos I` is the sagittal one.

The cone vertex with a nonzero r¹ coefficient has no Hessian: its differential
focus is undefined (`null`), and the cross-check there falls back to close
rays; nearby real rays remain usable.

`field_axes` selects `auto`, `x`, `y` or `both` for field curvature,
lateral colour and relative illumination. Auto keeps the y sweep for
rotational systems and covers both axes otherwise. Positive-y-only
sampling requires rotational symmetry; meridional symmetry permits only
the positive-x shortcut. `field_coordinates` gives normalised `[Hx,Hy]`
for every scalar sample, and scalar heights/distortions use that sample's
axis. Focal-line labels continue to follow image x/y. Grid distortion
uses the separate x and y first-order image scales, with the tracer's
warnings for mixed xy power. `grid_distortion.field_center` shifts the grid
in normalised coordinates (default `[0,0]`), also used by image simulation
for off-axis patches. Relative-illumination maps retain their rectangular
buffer shape but mark cells beyond the design's normalised radial domain
(`hypot(Hx,Hy) > 1`) as no-data: NaN in buffers, null in JSON, including
their image coordinates. Those cells are not traced or warned about;
interior and boundary cells keep the same projected-solid-angle integral.

The cross-check (`*_close_rays` columns) is the **differential pencil of
the real trace**: the chief ray's tangent with respect to its aim
parameters, carried analytically through every surface by
`mklight_trace::diff` (implicit-function derivative of each intersection,
derivative of the normal from the conic gradient or the sag Hessian,
differentiated Snell/reflection/diffraction/ideal-lens laws). Projected on
the transverse basis of the chief direction it gives the pencil matrices
`X`, `T`, and the foci follow as above (`F = −X T⁻¹` is invariant under the
parametrisation). It is the `h → 0` limit of the former close-ray pairs at
pupil offsets `(0, ±h)`, `(±h, 0)`, without their `O(h²)` truncation and
their cancellation (the two methods now agree to about 10⁻¹² mm on every
sample, against 10⁻⁶ mm for close rays with `h = 10⁻³`; on Metal the
double-float close rays lost up to 10⁻⁴ mm). Where the tangent cannot be
formed (grazing incidence `|N·d| < 10⁻⁹`, near total internal reflection,
an undefined Hessian) the close rays with `close_ray_step` are traced
instead and a warning counts the samples. `field_curvature` warns where the
two foci differ by more than `10⁻³(1 + |Δz|)` mm. The reported curve is
the axial (`z`) focus shift from the image surface.

### 2.3 Seidel conventions

Signed reduced indices make mirrors regular (`n' = −n`). With these signs a
positive singlet has `S_I > 0`, `W040 > 0` matches the OPD sign of
CONVENTIONS §4, and the third-order transverse aberration is
`εy = (1/(2ω'))[S_I ρ³cos φ + S_II H ρ²(2 + cos 2φ) + (3S_III + S_IV)H²ρ cos φ + S_V H³]`
with `ω'` the image-space reduced marginal slope (negative when converging);
colour: `Δz(short − long) = −n'C_L/ω'²`, `Δy = C_T H/ω'`. The Petzval
radius is `R_p = −1/(n'_k P)` with the **signed** image-space reduced index.
After an odd number of mirrors the light travels towards local `−z`, and a
concentric mirror has `R_p = −R/2`, matching its focal sphere. `S_V` is evaluated
in the division-free form `−Ā³yΔ(1/n²) + Āȳc(Āy + H)Δ(1/n)`. Ideal lenses
are aberration-free and contribute nothing (with a warning).

Extended rotational shapes use the **exact** fourth-order sag expansion
`z=c_eff r²/2 + a4 r⁴ + …`, with `G=a4-c_eff³/8` in the formulas above:

* Odd asphere: `c_eff=c+2β₂`, `a4=(1+k)c³/8+β₄`. Nonzero r¹ and r³
  terms are omitted with a warning: neither has an analytic fourth-order
  vertex expansion, and their real-ray aberrations are not described by
  these sums. With β₁ nonzero even first order is only formal.
* QCon: `a4=(1+k)c³/8 + Σ a_m (-1)^m binomial(m+4,4)/R⁴`.
* QBfs: if `S0=Σa_m Q_m(0)`, `S1=Σa_m Q'_m(0)`, then
  `c_eff=c+2S0/R²`, `a4=c³/8+(S1-S0)/R⁴+c²S0/(2R²)`.
  Both values come from the defining triangular recurrence, without finite
  differences or a fitted polynomial. Rotational XY/Zernike departures
  likewise use their analytic radial r⁴ coefficients.

A DOE uses the standard thin phase plate / infinite-index lens limit
([Buralli and Morris, *Design of diffractive singlets for monochromatic
imaging*](https://doi.org/10.1364/AO.30.002151)). Put `q2=mλA1/(2πR²)`,
`q4=mλA2/(2πR⁴)`, `p=-2q2`. Two coincident interfaces with middle index
N, `c1=c`, `c2=c-p/N`, `a4_2=a4_1+q4/N` reproduce the base plus phase
as `N→∞`. The implementation evaluates this limit algebraically (no huge
index and no cancellation of huge terms). Its aspheric combination is
`(n'-n)G-q4-3c²p/8`, and the second interface invariants are `A-py` and
`Ā-pȳ`. The plate has **zero Petzval contribution**, despite finite power.
With `d=(λ_short-λ_long)/λ`, its colour terms, including the base, are
`C_L=y[(A-py)δn'/n'-Aδn/n+pyd]` and
`C_T=y[(Ā-pȳ)δn'/n'-Āδn/n+pȳd]`. Colour sums are first order in spectral
changes; exact F/C correction in a finite-thickness hybrid need not make
this linearised colour identically zero.

For non-rotational geometry, placement or apertures, Seidel returns
`status="not_applicable"`, a reason suggesting full real-ray analyses,
an empty per-surface array and undefined (`null`) coefficients. This is a
successful result with RunInfo, not an unsupported-shape error.

Longitudinal aberration traces signed pupil zones on both axes by default
outside rotational symmetry (`axes` can select x/y/both); each curve names
its axis and subtracts that axis's primary focus. Chromatic focal shift
adds `paraxial_focus_x` and `focal_shift_x`; legacy arrays remain y.

### 2.4 Relative illumination: method

For a uniform Lambertian object the image irradiance is `E = L (n'/n)² Ω_p`,
with `Ω_p` the projected solid angle of the **real exit pupil seen from the
image point**: the area, in the direction-cosine plane `(L, M)` of the
image surface's tangent plane, of the directions in which light arrives at
the chief-ray image point `X₀`.

With aberrations, the rays through `X₀` come from neighbouring object
points, not from the field point itself. They are found by **real ray
aiming**: a real-aiming copy of the tracer is used when the system aims
paraxially. Normalised pupil coordinates `P` then label stop points for
every object point `q` (direction tangents, or object-surface `x, y`). For
each quadrature node the object point with `X(q, P) = X₀` is solved by
chord Newton. The integrand is the implicit Jacobian

`d(L,M)/dP|_X = ∂(L,M)/∂P − ∂(L,M)/∂q (∂X/∂q)⁻¹ ∂X/∂P`

(central differences of unclipped rays), times the survival `s` of the
clipped ray. Polar Gauss–Legendre × trapezoidal quadrature integrates an
unvignetted beam to about 10⁻⁹. A vignetting edge limits the accuracy to
`O(1/N)`. `RI(H) = Ω_p(H)/Ω_p(0)`, and the vignetting fraction is the
clipped share of the stop area. Cost: about 15 traced rays per node and
field.

Two cross-checks pin the method:

* **Object-side integral.** Integrating `δ(X − X₀)` on the object side,
  `∫ A_⊥ ω / |det ∂X/∂q| dP`, agrees with it to 10⁻¹⁰ (étendue
  conservation).
* **Chief-ray étendue form.** `E ∝ A_⊥ dΩ_obj/dA_img` agrees only to first
  order in the aberrations: within 10⁻⁴ on the singlet, but 0.5–0.8 % at
  full field of the triplet and double Gauss.

An earlier version measured the beam of one object point. That gives the
solid angle at the beam's own astigmatic foci and grows with field
curvature (RI > 1 at 5° on the singlet). It is not the image irradiance.

### 2.5 Layout

All geometry is global (`ResolvedSurface::frame`). Semi-diameters come from
`SequentialTracer::semi_diameters` (stop: `max(paraxial, traced)`); a zero
value falls back to 5 % of the largest one. Elements are consecutive
refracting surfaces with glass between them (cemented doublets share a
surface); the outline uses `EdgeSpec::mechanical_semi_diameter` or the clear
radius plus `edge_margin` (0.5 mm; the model defines no default margin),
with edge flats at the sag of each clear edge. Planes between equal media
(pure aperture stops) and dummies get no profile.

`section` selects `yz` (default) or `xz`; profiles and element outlines
use the true signed Cartesian sag in the corresponding local plane.
`in_section` marks geometry drawn in the selected section (profiles, stop
strokes, element edges and outlines, axis, the section's ray fan) whose
points all lie within a slab about global `x=0` or `y=0`, respectively, of
half-width 1 % of the largest mechanical diameter of the optics (object and
image excluded). Tolerance-scale decentres and tilts thus stay in the
section, independently of lead-in and track; rings, rims, the other
section's profiles and sagittal fans never are.
Rings and mechanical edge flats evaluate sag at each azimuth, rather than
rotating a meridian. Clear radii stay inside the sag domain of both axes
(toroids, biconics) and inside the normalisation radius. DOE surfaces carry `is_diffractive`, also bit 2 of
polyline flags. `mesh_rings>0` with `include_3d=true` requests a coarse
polar triangle mesh (`ring_samples` angular subdivisions); invalid-domain
triangles are omitted. Engine buffers `surface_mesh` (f32 [T,3,3], relative
to origin) and `mesh_surfaces` (u32 [T]) let the app draw the actual 3-D
freeform graph. The legacy MKLL blob continues to contain polylines only.
`LayoutScene::to_buffers` gives structure-of-arrays
records (vertices as f32 relative to the scene centre plus absolute f64,
offsets, kind codes 0–6 as `PolylineKind`, 7 for element outlines, −1 for
"no index") and `LayoutBuffers::to_le_bytes` one little-endian blob
(`"MKLL"`, version 1; layout in its doc comment).

The first-order report and engine system summary include `paraxial_xy`:
EFL_x/EFL_y and BFL_x/BFL_y at every system wavelength, signed anamorphic
squeeze **EFL_y/EFL_x**, its definition, symmetry and per-surface DOE
reduced power `-mλA1/(πR²)`. The legacy first-order entries remain the y
sweep. Tilts/decentres are unfolded; mixed xy power is explicitly warned
as a diagonal projection, not principal focal lengths (SURFACES §8).
Piston, vertex tilt and an odd cone retain the tracer's nominal-vertex
warnings. The plate's power is excluded from the report's Petzval sum.

The symmetry audit covers spots, fans, footprints, energy/MTF/best focus,
layout rays, grid distortion and all scalar field sweeps: pupils are
sampled over the full disk/fan. Only the rotational longitudinal fan uses
a half pupil. Tangential/sagittal fan labels denote image y/x; the
geometric MTF's tangential direction is the field's radial direction (y on
axis). Neither assumes the field lies on y. Scalar Airy/F-number reference
scales retain the legacy y first-order convention.

### 2.6 Reductions from ray batches (backend seam)

The statistics of `spot`, `energy` (encircled energy, geometric MTF, best
focus) are public as reductions of a `spot::RayBatch`. A batch holds a list
of system wavelengths and one `FieldRays` per field: arrived rays relative
to the primary chief ray with their slopes, plus per-wavelength chiefs and
launched/arrived counts. `FieldRays::from_image_rays` builds one from plain
image hits (`ImageRay { wavelength, hit: Option<ImageHit { point, dir }> }`
in the image local frame), so a GPU backend or an optimiser operand that
already has image hits uses exactly the same compensated two-pass statistics
and conventions:

* `spot_from_rays`, `encircled_energy_from_rays`, `geometric_mtf_from_rays`,
  `best_focus_from_rays` take `(&SequentialTracer, &Settings, &RayBatch,
  &BackendInfo, warnings)`.
* The tracer supplies metadata only (snapshot, fields, paraxial data); no
  ray is traced. `BackendInfo { name, precision }` goes into `RunInfo`.
* The batch's fields and wavelengths replace the settings' selections.
  `RayBatch::validate` rejects inconsistent counts or indices.
* The CPU analyses are `trace_field_rays` (public) followed by the same
  reduction, and a test checks that both paths give identical results.
* `util::Moments` (weighted two-pass point statistics) is public too.

## 3. Validation

| test file | what it pins |
|---|---|
| `tests/curvature.rs` | Coddington = differential pencil (≤10⁻⁹ rel.) and = close rays (≤10⁻⁵ rel.) on every sample, field and wavelength, with the close-ray error plateau against the pencil; concentric mirror focal sphere `Δz = −(R/2)(1 − cos θ)` to 10⁻⁹; ideal lens flat and distortion-free, `f·θ` distortion `100(tan θ/θ − 1)`; mirrors tilted about `y` and about `x`: foci `(R/2)cos I` and `R/(2 cos I)` labelled by the image frame (10⁻⁹); compound-tilted mirror: both methods find the two focal lines and label them alike |
| `tests/seidel.rs` | spherical mirror closed forms (stop at mirror / at centre, paraboloid `S_I = 0`, `W040 = y⁴/4R³`, Petzval sum `−2/R`, radius `−R/2`); Petzval radius vs the Coddington and close-ray foci of a concentric mirror and vs the third-order foci `(3Δz_s − Δz_t)/2` of a singlet; two separated lenses `P = ΣK_i/n_i`; thin plano-convex lens vs thin-lens `S_I…S_V` (coma with its sign, independent `H`); `S_IV = H²P` and the `S_V` form on every sample; Seidel vs real rays (spherical, coma sign and size, astigmatism, sagittal coma, distortion) on the Cooke triplet and singlet at reduced aperture; `W040` vs OPD; exact mirror LSA `(R/2)(sec I − 1)`; thin-lens colour `−f/V`, `C_L`, `C_T` vs real colour |
| `tests/illumination.rs` | RI of an ideal lens = exact disk view factor (10⁻⁸) and ≈ `cos⁴θ`; real singlet with the stop at the lens = `cos⁴θ` × distortion Jacobian (10⁻³); exact object-side integral on the singlet, triplet and double Gauss (10⁻⁸) and the chief-ray étendue form (first order); RI < 1 and falling on the achromat and finite relay; vignetting by a field aperture = disk-overlap area; footprints |
| `tests/spot_energy.rs` | defocused ideal lens: RMS/GEO exactly the scaled pupil statistics, EE vs exact grid counts and `(r/R)²`, geometric MTF vs `2J₁(x)/x`; best focus recovers a known defocus and equals the retraced spot; paraboloid on-axis spot ≈ 0; polychromatic weighting; vignetted fraction; fans vs direct traces and symmetry; seeds; cancellation and progress; JSON round trips with NaN; curved-image defocus warnings; batch reductions of an externally traced batch equal the traced analyses (and record the backend); best focus skips chief-less fields for the chief reference and warns at range edges and on non-convergence |
| `tests/layout.rs` | global profile end points, elements of cemented lenses, folded Newtonian in global coordinates, vignetted ray end points, buffer round trip |
| `tests/system_data.rs` | report vs paraxial data, afocal `"inf"` serialisation |
| `tests/extended.rs` | all extended Hessians vs close rays at skew signed fields; curved DOE phase tensor; QCon/QBfs vs independent even-asphere equivalents; hybrid DOE Seidel vs real rays and colour; Cartesian layout sections/rims/meshes and the x domain of steep toroids/biconics; freeform/anamorphic spots, fans, distortion, signed longitudinal fans and first order |
| `tests/golden.rs` | frozen first-order data, RMS spots per field and wavelength, Seidel sums and field curvature / distortion of every sample (relative 10⁻⁹) |

**Golden files** live in `crates/mklight-analysis/tests/golden/<sample>.json`.
After an intended change, regenerate and review the diff:

```text
MKLIGHT_UPDATE_GOLDEN=1 cargo test -p mklight-analysis --test golden
```

**Benchmark** (ignored by default):
`cargo test --release -p mklight-analysis --test bench -- --ignored --nocapture`
traces a 98 829-ray polychromatic spot diagram of the Double-Gauss; about
8 Mrays/s on a 12-thread Apple-silicon CPU.

## 4. Compatibility with the macOS app's current shapes

The app (`apps/macos/Sources/MKLightEngine/AnalysisRequest.swift`,
`AnalysisResults.swift`) currently runs on a mock. The Rust shapes are
authoritative; these are the differences to align:

* **Envelope.** Rust results embed `run_info`; the app's `AnalysisOutput`
  keeps `run_info` next to `{analysis, data}`. The engine should move it out
  (analysis names already match: `layout`, `spot_diagram`, `ray_fan`,
  `opd_fan`, `field_curvature_distortion`, `first_order`).
* **Settings.** `configuration` and `execution` (backend, precision, seed)
  belong to the engine, not to analysis settings; unknown keys are ignored,
  so the app's JSON decodes. Random pupil sampling takes its seed from the
  spot settings' own `seed` (the engine should copy `execution.seed` there).
  `FieldSelection` gains a `list` variant, which the app cannot decode yet.
* **Spot diagram.** Pattern `dithered` is new; `random` uses `1 + 3r(r+1)`
  points (the app's comment says `rings²`). `reference`, `centroid` and
  `chief` are in the image surface's *local* frame (the app says global; equal
  for untilted images). Added: `chief`, `polychromatic`, `per_wavelength`,
  `through_focus`, `wavelengths`, `defocus`; `rms_radius`/`geo_radius` are
  `null` when undefined.
* **Fans.** Added `cross` (the other transverse component) on ray fans and
  the `reference` setting.
* **Field curvature.** Same `field_values`, `field_unit`, `curves[{wavelength,
  tangential, sagittal, distortion_percent}]`, `wavelengths_um`; added
  `normalized_field`, close-ray columns, heights and `f·tan θ`/`f·θ`
  distortion. Values can be `"inf"` (collimated image space), which the app's
  `[Double?]` does not decode yet. `close_ray_step` is a new setting.
* **MTF.** `geometric_mtf` returns the app's `MTFResult` shape (`frequencies`,
  `curves[{field, orientation, modulation}]`, `diffraction_limit` = `null`,
  `cutoff_frequency`, `field_values`) for the *geometric* MTF; the app's
  `pupil_sampling` corresponds to `rings` here. The diffraction MTF and PSF
  belong to `mklight-wave`.
* **First order.** Same `entries[{key, label, value, unit}]` and
  `wavelength_um`; added `tables` (one per selected wavelength) and `system`.
* **Layout.** Same polyline kinds and fields, `elements[{first_surface,
  last_surface, front, back}]`, `surfaces[{index, id, vertex, semi_diameter,
  semi_diameter_is_auto}]`, `bounds`, `wavelengths_um`, `field_count`; added
  element `outline` and `mechanical_semi_diameter`, surface `label`, `axis`,
  `mechanical_semi_diameter`, `is_mirror`, `is_stop`; mirrors are
  `surface_profile` polylines flagged through `is_mirror`. New settings:
  `rings`, `ring_samples`, `edge_margin`, `lead_in`.
* **New analyses without an app counterpart yet:** grid distortion, Seidel,
  longitudinal aberration, chromatic focal shift, lateral colour, relative
  illumination, footprint, encircled energy, best focus.

## 4. Run-local spectra (engine)

Every wavelength-dependent geometric, wave, radiometry, ghost and stray
analysis accepts optional `spectrum`, as do optimisation and wavelength-based
model/material reports through the same dispatcher (`run_analysis`). The engine applies it **once**, before
material resolution or backend qualification, to a derived immutable snapshot.
The session design is unchanged. `RunInfo.snapshot_hash` identifies the
submitted snapshot; the envelope's `derived_snapshot_hash` identifies the
run-local system (null without a spectrum). `RunInfo.settings.spectrum`
records the request. Wavelength
indices in the result and in selections refer to the replacement list.
Absent or null preserves the design wavelengths and existing defaults.

Examples:

```json
{"spectrum":{"kind":"explicit","wavelengths":[{"um":0.55,"weight":1.0}]}}
```

```json
{"spectrum":{"kind":"photopic","sampling":{"samples":9,"range_um":[0.38,0.78],"spacing":"equal_weight"}}}
```

Kinds are `explicit`, `photopic`, `scotopic`, `blackbody` (positive
`temperature_k`), and `sensor` (`table:[[um,response],…]`, strictly increasing
wavelengths and nonnegative responses). Presets and sensor responses accept
`sampling` with defaults shown above. Ranges must lie inside a response
(table ranges: photopic 0.360–0.830 µm, scotopic 0.380–0.780 µm).
The blackbody uses energy radiance per unit wavelength, proportional to
`λ^-5 / expm1(hc/(λkT))`, evaluated in log space and tabulated at 4097
points over the requested range. It is not photon-count weighting.

The primary is the sample nearest the energy-weighted mean wavelength;
exact ties take the first sample. Explicit ordering and weights are preserved.
For a continuous response, weights are **bin integrals**, in response·µm:

* `uniform`: N nodes including range endpoints (N=1: midpoint), with
  exact integrals over their clipped midpoint/Voronoi bins.
* `equal_weight`: N equal-energy bins of the piecewise-linear response,
  each represented by its energy centroid. Both total energy and its first
  wavelength moment are preserved (up to floating-point rounding).

The in-code 1 nm tables are CIE 018:2019 Tables 1 and 2:
[CIE 1924 photopic V(λ), DOI 10.25039/CIE.DS.dktna2s3](https://doi.org/10.25039/CIE.DS.dktna2s3)
and [CIE 1951 scotopic V′(λ), DOI 10.25039/CIE.DS.gr6w4b5g](https://doi.org/10.25039/CIE.DS.gr6w4b5g).
All 471 photopic (360–830 nm) and 401 scotopic (380–780 nm) rows match
the CIE CSVs (MD5 `f389958555461a7d9a7562145e8ca9c0` and
`3e45714a429d02e5d1f2a752226d7698`); V′ is 1.000 at 506–508 nm. Tests pin
published samples and the metadata column sums, 106.8569171011719 and
97.0713165740; the integrals are those sums less half of each endpoint
value (1 nm trapezoids), and sampled weights sum to them. Ordinary
material resolution emits the same formula-range warnings as a design using those wavelengths; extrapolation is not hidden.
A wavelength-sweep analysis still evaluates its requested sweep, using the
replacement spectrum to set its primary reference and default spectral range.

## 5. Signed field maps and glass maps

`pupil_aberration`, `relative_illumination_map`, and `chief_ray_angle` use
`FieldMapSettings`: a full signed rectangular grid (`grid:[nx,ny]`,
`extent:[Hx,Hy]`) or `mode:"x"` / `"y"` sweeps. There is no rotational
symmetry shortcut. A one-sample axis is zero. Default wavelength is the
run spectrum's primary; `wavelength` selects a different index. Results
carry normalised field coordinates, real chief image x/y in mm, and a
`values[ny,nx,columns]` f64 buffer. Column metadata specifies names, units
and definitions. Undefined values use NaN in buffers.

* **Pupil aberration.** Force paraxial launch (ray aiming off), then trace
  the chief through the real surfaces to the stop. `stop_error_x/y` are
  its actual local stop coordinates relative to the stop vertex, in mm.
  With real aiming this error would be zero by construction; it must not
  be measured from an aimed chief. Central pupil differences at ±1e-4
  give `J = d(stop x,y)/d(entrance x,y)` in physical mm/mm.
  `magnification_x/y` are the norms of J's columns;
  `variation_x/y = magnification(H)/magnification(0) - 1`.
  These describe the entrance-pupil-to-stop map, not exit-pupil size.
  An entrance pupil at infinity has no finite physical magnification.
* **Relative illumination map.** The projected-solid-angle integral of
  §2.4, by the same routine and with the same default quadrature
  (12 radial × 48 azimuthal nodes), at every x/y field, so on any field it
  equals the radial curve. Reports RI, projected solid angle (sr), and the
  clipped fraction of stop area. Only this map takes `radial_samples` and
  `azimuth_samples`.
* **Chief ray angle.** The real chief ray through the stop centre (real
  aiming, as in §2.4, whatever the design's aiming), so it equals the
  radial analysis's `chief_ray_angle_deg`. The total incidence is against
  the actual image sag normal (recorded by the tracer); signed x/y angles are
  `atan2(d_x/y, abs(d_z))` relative to the image vertex frame.
  `sensor_limit:[[image_radius_mm,max_angle_deg],…]` is linearly
  interpolated with constant endpoint extrapolation. Margin = limit
  minus incidence, negative when the sensor limit is exceeded (NaN without
  a limit). Only this map takes `sensor_limit`. Both image x/y heights are
  provided; grid mode supports asymmetric systems.

`glass_map` reports every loaded catalogue's glasses in library order,
then each model glass used in the design. Coordinates are catalogue n_d,
V_d, and `(n_g-n_F)/(n_F-n_C)` from the dispersion formula; model values
are computed at reference conditions. `surfaces` marks each surface's
following medium, resolving unqualified catalogue names in library order.
The aligned `coordinates[glasses,3]` buffer supports either map projection.
Catalogue name, glass kind, availability and substitution exclusion are
metadata. The material library retains AGF status/exclusion, relative cost,
climate/stain resistance and internal transmission samples; `glass_map`
reports them by default (unspecified catalogue status is standard, model
status/exclusion are null). Callers can override availability with `metadata:[{catalog,name,status,exclude_substitution},…]`; these
settings are recorded in RunInfo. No availability is inferred from a name.

The existing `footprint` result keeps its selected-surface local x/y
buffers. Added `overlays` name every surface/field/wavelength and offset /
count ranges in `global_points[N,3]` and `global_outlines[M,3]` (f64 mm).
Every hit includes traced sag and the surface's global placement. Clouds
include rays passing that surface even if lost downstream. Convex outlines
are counterclockwise, unclosed, and retain traced sag at each hull vertex;
they enclose the cloud and do not describe nonconvex clipping boundaries.
Hull vertices within 1e-9 of their neighbours' chord (a pupil grid row,
up to rounding) are removed, so both backends give the same outline.

Known-answer tests cover stop-plane ideal-lens pupil maps, CRA and sensor
margin from geometric image height, 2-D disk view factors, the map against
the radial curve and its chief-ray angle, catalogue/model spectral-line
coordinates and markers, and global footprints against the local ones on
the surface sag. The engine's `tests/analyses.rs` runs all new entries
through strict result and buffer schema validation.

## 6. Resident geometric adapters (P5)

`mklight_analysis::gpu` exposes backend-neutral job adapters; the engine's
native geometric routes call them, and capability and `auto` routing are in
GPU.md §9.2–9.3. The reference functions above remain available.
Every adapter takes a `JobContext` and returns the existing result type with
its actual backend and precision in `RunInfo`. `Geometry` pins an immutable
system upload for a sequence of resident plans. Its `spot_diagram_job` method
reuses uploaded systems, chief seeds and aim tables across repeated jobs;
the free function includes that preparation in each call.

| Adapter | Resident computation |
|---|---|
| `spot_diagram_job`, `ray_grid_job`, `ray_fan_job`, `opd_fan_job` | Aiming by the adapter's design (§6.1), P1 tracing; chief-relative double-float positions/slopes and reference-sphere OPD; P3 two-pass spot moments; optional requested points |
| `encircled_energy_job` | Double-float radii/square half-widths; P3 stable distance/ID sort and inclusive weighted scan; device binary-search curve/quantile queries with a bin-uncertainty bound |
| `geometric_mtf_job` | Double-float turn reduction, weighted complex characteristic functions and P3 fixed reductions; frequency tiles bound scratch storage |
| `best_focus_job` | Trace once; one two-pass reduction of centred position/slope products gives each field's exact defocus quadratic, which brackets and drives the host Brent search and the through-focus curve; the optimum is re-reduced two-pass, where the quadratic form could cancel |
| `field_map_job`, `relative_illumination_job`, `illumination_rows` | Aiming-off pupil probes; real-aimed CRA against the image sag normal; the nested image-point RI solve and implicit direction Jacobian of §2.4, followed by P3 quadrature reductions |
| `field_curvature_job`, `grid_distortion_job` | Resident chief/source tables and double-float Coddington pencils; the cross-check column is the host f64 differential pencil (device close rays differenced in double-float failed the f64 budget; `run_info.settings.host_stages`); host packs scalar first-order labels and distortion columns |
| `longitudinal_aberration_job`, `lateral_colour_job` | Zonal chief crossings and chief-relative spectral offsets; small paraxial reference labels remain host work |
| `semi_diameters_job`, `footprint_job`, `footprint_overlay_job`, `layout_job` | P1 all-surface records, device radii/extrema and global projection, sag/gradient samples and meshes; host assembles topology, hulls, labels and final result arrays |

`semi_diameters_job` returns `(SemiDiameters, RunInfo)`, because the reference
sizing record has no metadata field. `GridResult` contains selected fields,
wavelengths, pupil coordinates and optional image hits in stable grid order.
Footprint clouds keep passing records even when a ray later fails. Layout
also retains terminal clipped/TIR points; invalid launches produce no ray
polyline. CPU and resident layout providers share the same host topology,
mechanical edge flats, domain filtering, surface flags and section slab test.
Sag/vertex evaluation is tiled; global positions are computed in double-float
relative to a display origin before final public f64 packing. The existing
`LayoutScene::to_buffers` still supplies origin-relative f32 display buffers.
Dense footprint overlays tile at at most 2²⁰ surface records, preserving
field/wavelength/surface order while retaining requested public clouds. Extra
plan statistics remain in RunInfo; final output storage scales with the
requested cloud size. Through-focus spots evaluate the same per-field defocus
quadratic (chief-relative RMS adds the moving centroid), so one reduction
serves every plane without focus × ray scratch. Chief rays are traced only
for adapters that reference them.

The private `mklight_gpu::geometry` operation records are version 1, with
numeric layouts in `include/mklight_geometry_types.h`. They consume P1
`ResidentHits` and accept uploaded or P4 resident systems. `Coordinates`,
`FocusTerms`, `Opd`, `Coddington`, `CloseFocus`, `Illumination`,
`SurfaceSamples` and `Records` publish typed buffers for P6/P8 consumers;
`Geometry::{coordinates,moments,coddington,image_source,illumination}` append
these chains without an intermediate optical readback. `FieldData` kinds are
0 (image x/y and incidence/x/y angles), 1 (stop x/y and pupil magnifications),
and 2 (image x/y and launch-source x/y). Source coordinates are tangents at
infinity and object x/y at finite conjugates. `FieldData::record_slots` is
the logical record extent, including for a subview of a larger allocation.

Coddington differentiates the shared Cartesian sag/phase gradient recurrence
with a dual scalar, retaining mixed Hessians and the curved-base DOE
connection term. The odd-cone vertex remains undefined. RI uses chord Newton
with refresh for both the stop aim and the source-to-image solve (the
restructured designs of §6.1; the original `Illumination` request remains
as the `legacy` design). Failed
image-point nodes retain a status and are counted in the result warnings;
P3 compacts unresolved RI node indices, and the adapter evaluates only those
nodes with the existing f64 image-point solver. `ri_polish` reports attempted,
resolved and remaining failed nodes plus CPU time; successful device nodes
are never retraced. `IlluminationBuffers` exposes provisional resident
integrals, field data and the sparse unresolved list/count, so P6/P8 must
handle that list before consuming a final map. EE also bounds the energy that can cross a
requested bin within the geometry error budget. When that bound exceeds
0.002, the complete reference EE result preserves its exact bins and automatic
radius, records the attempted resident chain and counts the reference
fallback; its actual result backend/precision are CPU f64. No requested bin
is widened to hide classification error. The automatic outer radius is the
largest arrived distance, so that bin holds every arrived ray by construction
(as in the reference) and is not classified against a recomputed distance. A caller with an independently
qualified, tighter full-grid position bound can pass it to
`encircled_energy_with_uncertainty_job`; that bound must include its safety
margin and is recorded in RunInfo. Singular distortion denominators use a
small, explicitly counted f64 reference decision to preserve signed
infinities. A cancellation discards the result and never starts a CPU
replacement. First-order, chromatic paraxial reports and an isolated Seidel
report remain the existing small host computations; this adapter does not
supply a batched Seidel reduction.

### 6.1 Aiming and the division between host and device

Every generation of a `Geometry` (grid rays, chief rays, layout fans, the
sizing and field-map grids built on `append_grid`) is aimed before its plan
is built, by one design read once per adapter from the work-split key
`aiming` (`mklight_core::split`; `gpu::aim`):

| `aiming` | Where pupil rays are aimed |
|---|---|
| `host-table` (default) | The reference tracer aims every ray in f64. Generations with at least 2,048 rays per field row first solve a coarse seed table per row (about one node per four rays, 5–37 nodes per axis over the vignetted pupil square ±1.125, as the device table); each ray's Newton solve starts from the table's Catmull–Rom interpolated aim and stop Jacobian (`SequentialTracer::generate_many_from`) and falls back to the reference schedule if it does not converge. Smaller generations are aimed as by `host`. Paraxial launches (aiming off) need no solve; the device generates them inside the plan. |
| `host` | The reference tracer aims every ray (`generate_many`: the rays of a chunk advance together, chunks in parallel on rayon), bitwise as the CPU reference does. |
| `device` | The double-float device solve of GPU.md §9.2 (host chief rows, device seed table for large generations), in a plan of its own, followed by host completion. |
| `table` | A host f64 seed table and device Newton corrections from its interpolated seeds, at most 12 iterations per ray, followed by host completion. |
| `split` | The host and the device aim one generation at the same time from one host seed table: the device takes tiles from the front (a sixteenth of the generation, 2¹³–2¹⁷ rays, two in flight), the host's rayon workers take 128-ray chunks from the back until the claims meet, then complete the device tiles' unconverged aims as each tile finishes. Generations under 2¹⁵ rays, and paraxial launches, are handled as by `host-table`. |

Host aims are written straight into the shared ray planes the trace reads
(four `fill_buffer` buffers filled in place by the rayon workers, the start
rays rounded to double-float once); no sample or field record is built.

**Host completion.** A device aim cannot always certify the requested stop
tolerance: the double-float trace evaluates the even-asphere slope in f32
(GPU.md §4.1), so on wide-angle aspheric designs the evaluated stop hit
jitters by 1e-9 to 1e-8 of the stop radius, above the 1e-10 the reference
reaches. On a 35-surface ultra-wide a third of the device aims, chief rays
among them, ended unconverged, and the device spot diagram's vignetted
fraction differed from the reference by up to 0.77. Every device aim
reported `MKL_AIM_FAILED` is therefore aimed again in f64 by the reference
tracer; a ray it aims replaces the device's (the position and report planes
are copied once, with the host rays in place), a ray it cannot aim stays
failed, exactly as on the CPU, and the hard field statuses of the host rows
are kept. Each plan's `RunInfo.settings.aiming` records the design, the host
and device rays, iterations and seconds, the completed and resolved rays and
the seed-table nodes (`MKLIGHT_TRACE_AIM_EVIDENCE=1` prints it to stderr).

The image trace and the reductions run on the device as before in the
default division. The key `geometric` selects how the geometric analyses'
remaining stages divide (`gpu::division`):

| `geometric` | Division |
|---|---|
| `fused` (default) | One plan per stage: the device traces the aimed grid and reduces it. Geometric MTF submits its characteristic-function tiles without waiting for each (at most four in flight). |
| `pipelined` | The fields are split into groups of at least 2¹⁵ rays, each group a plan submitted as soon as its rays are aimed, so the host aims group k + 1 while the device traces and reduces group k (spot diagram, geometric MTF with its tiles, footprint, ray grid); smaller generations form one group. |
| `host-reduce` | The device traces the grid and the chief rays; the host reads the hits in place into a `RayBatch` (§2.6) and runs the reference reduction (spot diagram, geometric MTF, best focus). |

Encircled energy, the fans and the chromatic and layout adapters keep the
fused division. The study keys are measurement scaffolding: the designs that
lose the comparison are to be removed with their keys.

High-level `Geometry` metadata belongs to its nominal immutable snapshot.
Changed candidates must use P4's updated first-order/field metadata through
the numeric requests, rather than retaining nominal launch descriptors.

Ray, field, quadrature and vertex inputs are written with `fill_buffer`; final buffers are borrowed with
`read_view`. No staging ray/optical-intermediate upload/readback array is introduced.
Immutable packed-system registration retains the existing validated upload
API; its cold copied bytes are reported separately from the zero-copy resident
plan counters. Result packing
widens requested final values into the public result arrays. `RunInfo.settings`
retains the resident execution evidence (build/probe, operation manifest,
plan and stage statistics, seed), including additional plans for energy,
focus, layout and RI tiles. `geometry_version` records the numeric recipe
version inside that execution evidence. `adapter_fallbacks` records singular
distortion reference decisions. These plan timings cover backend execution;
end-to-end benchmark wall time also includes preparation and result packing.

P5's owned harnesses are `tests/p5_geometry.rs` and the ignored
`tests/p5_bench.rs` in `mklight-analysis`. They compare complete public arrays
with f64 on all 16 samples; cover both shader origins, matched FTZ bits,
analytic ideal-lens/disk answers, P4→P1→P5 chaining, concurrent jobs,
cancellation and resource release. The benchmark compares unchanged f64,
legacy trace-only Metal plus host statistics, and the resident chain at
matched settings. Numerical maxima, hardware, raw runs, cold preparation,
median/p95, copy/storage counters and any qualification gaps are retained
in the task's `scratchpad/gpu/P5-evidence.md`. Engine routing is the
calibrated cost model of GPU.md §9.2, not a fixed ray-count assumption.

On the measured M2 Pro, 113- and 9,845-ray spots and the default spots on
all 16 samples favor f64. At the capped 823,473-ray square-pupil setting,
resident spot medians were 23–34 ms versus 37–89 ms f64 across the six dense
representatives. Complete Cooke global footprints at that density measured
434 ms resident, 477 ms legacy Metal and 831 ms f64, with 173 MiB peak
resident-plan scratch (public output and host hull storage excluded).
Default interactive layouts and Cooke RI at 5²/33² fields still favor f64;
RI medians were 42/1819 ms resident versus 23/1441 ms f64. Each owned series
uses one warm run plus ten measured runs. These measurements do not install
a ray-count crossover or qualify changed candidates. The external evidence retains p95, raw settings/load snapshots,
explicit copy/fallback counts and the remaining acceptance gaps.

### 6.1 CPU/GPU divisions of the field maps (study key `fieldmaps`)

The image-point illumination, the chief-ray maps and the system
transmission keep several divisions of their work between the CPU and the
GPU side by side, selected per run by `mklight_core::split` (key
`fieldmaps`; `MKLIGHT_SPLIT_FIELDMAPS` or the benchmark's variant columns).
The design that ran is named in `run_info.settings` (`fieldmaps_variant`,
`default:<design>` without a selection). Every design returns the reference
result type and is tested against the f64 reference on all built-in
samples (`tests/fieldmaps_split.rs`; the Metal half is ignored by default).

**Restructured image-point illumination.** The host prepares every field
exactly as the reference does (`illumination::image_point`: the real-aimed
chief, `X₀`, the image basis, chief ray angle, `cos⁴`, `q₀` and the chord
inverse of `∂X/∂q`), in parallel over fields; CRA and `cos⁴` are therefore
the reference's own values. The device rows (`MklRiField`) carry these with
the chief aim and the inverse stop Jacobian that seed every aim. Two
kernels (`mklight_gpu::geometry::ri`, bitwise equal to their CPU-f32
specification) run a resumable state machine per thread whose every step is
one trace from a single call site, so the threads of a SIMD group trace
together whatever stage they are in:

* `mkl_geometry_ri_rays`, one thread per aimed ray `(q, P)`: chord-Newton
  stop aim from a seed or the chief prediction (finite-difference refresh,
  backtracking, continuation in the pupil radius), then the image trace;
* `mkl_geometry_ri_nodes`, one thread per pupil node: the image-point
  Newton (each ray seeded with the previous converged aim and stop chord),
  then either the survival ray and the eight differentiation rays (full
  mode) or nine pencil tasks for `mkl_geometry_ri_rays` (Newton-only mode).

With the f32 even-asphere slope (`trace_slope=f32`, the trace before
stream A's fix) the double-float trace of aspheric systems resolved image
points only to about 1e-7 mm; with the default double-float slope it
follows the f64 trace to about 1e-12 mm, and on the wide-angle aspheric
samples the device now leaves 0–23 of 6336 nodes to the reference instead
of 1700–5000. The device still accepts a stop aim that stalls within the
reference's own aim tolerance clamped to [1e-10, 1e-9] of the stop radius,
and an image point that stalls within 1000 image-point tolerances; a chord step that shrinks
a residual less than 4× refreshes its Jacobian; a ray or node that does not
finish within its trace budget (48 traces, 16 for a pencil ray seeded with
a converged aim; 192 per node) fails on the device. Failed device nodes are
completed by the reference (`illumination::reference_nodes`) and counted
(`ri_stages`, `ri_polish`), so a design never returns fewer nodes than the
reference. The host forms the pencils' central differences and the
per-field Kahan sums in f64, in the reference's node order and weights.

| design | division |
|---|---|
| `legacy` | the original `Illumination` request: one thread per node with nested aims and per-node field work |
| `nodes` | one node kernel dispatch, everything on the device; host sums |
| `split` | node kernel (Newton only) writes the pencils, a flat ray pass traces them (9 rays per node, one thread each) in the same plan; host determinants and sums |
| `hybrid` | the CPU solves the image points (the reference's Newton, bitwise, all cores), the flat ray pass traces the pencils; a pencil the device leaves undecided is finished by the rest of the reference node from the CPU's solve (`illumination::finish_nodes`) |
| `pipelined` (default) | the hybrid in field blocks (about six, at least 2048 nodes each): the GPU traces block k while the CPU solves block k+1; once a block's pencils are mostly undecided, the CPU finishes the blocks not yet submitted |
| `shared` | the CPU (reference nodes, about 2048 nodes' worth of fields at a time from the back) and the GPU (`split`, field blocks from the front) drain one queue; which fields each takes depends on timing, so results are not bitwise repeatable (measurement only, GPU.md §9.3) |

The reference (`illumination::illumination_rows`, shared by the curve, the
map and image simulation) runs its fields in parallel blocks of about
16384 nodes and the nodes of each field in parallel too; every field sums
its own nodes in node order, so the rows are bitwise those of a serial run
(cancellation is checked per field). The hybrids' CPU stages run the same
way: a block's fields and their nodes in parallel, and the CPU finishing of
undecided pencils across fields. All designs report their progress from the
host side with stage `geometry` once device work has started (the
`pipelined` and `hybrid` pencil plans share the job's cancellation but not
its progress, since their CPU solves and plans overlap).

**CPU share thread budget** (study key `fieldmaps_threads`). The CPU work
inside the hybrids (the `hybrid`/`pipelined` solves and finishing, the
`shared` CPU fields; for the transmission the `cpu-aim`/`pipelined` host
aims and reference completion and the `shared` CPU rows) uses the global
rayon pool by default; `fieldmaps_threads=N` gives it a pool of `N`
threads of its own (`mklight_gpu::cpu_share`), leaving the other cores to
the threads that feed the GPU. The budget is recorded as `cpu_threads`
(`ri_stages` or the transmission's settings) and never changes a result.

**Chief-ray maps.** `chief_ray_angle`, `pupil_aberration`,
`grid_distortion` and `field_curvature_distortion` need one aimed chief per
field point: `host-chief` (the default) solves them in f64 on all cores and
traces on the device; `device-chief` solves them with the aiming kernels'
own chief solve (no host seeds; a map with a failed device chief is
completed by the reference); `cpu` runs the reference path; `sized` runs
`cpu` below 64 field points (field-wavelength rows for the field curvature)
and `host-chief` above (a size rule for `auto`, measured as a design, since
an explicit GPU request must run natively). The reference itself solves and traces the fields of these maps in
parallel (blocks of 256 fields; grid distortion in blocks of about 1024
points; the field curvature all its (wavelength, field) rows at once), with
unchanged results.

System transmission and the polarisation pupil map are described in
POLARIZATION.md §9.
