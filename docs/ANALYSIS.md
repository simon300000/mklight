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
| `field_curvature` | `curvature` | generalised Coddington (2×2 pencil matrices) along the real chief ray, labelled by the image frame, cross-checked by close rays; distortion vs the paraxial chief height, `f·tan θ`, `f·θ` |
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
focus is undefined (`null`); nearby real rays remain usable. An r³ departure
has a Hessian but only O(h) convergence of the vertex close-ray check.

`field_axes` selects `auto`, `x`, `y` or `both` for field curvature,
lateral colour and relative illumination. Auto keeps the y sweep for
rotational systems and covers both axes otherwise. Positive-y-only
sampling requires rotational symmetry; meridional symmetry permits only
the positive-x shortcut. `field_coordinates` gives normalised `[Hx,Hy]`
for every scalar sample, and scalar heights/distortions use that sample's
axis. Focal-line labels continue to follow image x/y. Grid distortion
uses the separate x and y first-order image scales, with the tracer's
warnings for mixed xy power.

The close-ray cross-check traces pairs at pupil offsets `(0, ±h)` and
`(±h, 0)` and finds where the pencil `X + sT` of the pair differences
degenerates (`det = 0`, the eigenvalues of `−X T⁻¹`), labelled by the same
rule. The two agree to about 10⁻⁶ mm on every sample and on tilted mirrors.
`field_curvature` warns where they differ by more than `10⁻³(1 + |Δz|)` mm.
The reported curve is the axial (`z`) focus shift from the image surface.

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
`in_section` means every point has global `x=0` or `y=0`, respectively.
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
| `tests/curvature.rs` | Coddington = close rays (≤10⁻⁵ mm rel.) on every sample, field and wavelength; concentric mirror focal sphere `Δz = −(R/2)(1 − cos θ)` to 10⁻⁹; ideal lens flat and distortion-free, `f·θ` distortion `100(tan θ/θ − 1)`; mirrors tilted about `y` and about `x`: foci `(R/2)cos I` and `R/(2 cos I)` labelled by the image frame (10⁻⁹); compound-tilted mirror: both methods find the two focal lines and label them alike |
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
