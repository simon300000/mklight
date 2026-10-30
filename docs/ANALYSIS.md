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
  invalid indices are errors.
* **Non-finite numbers** (`util::nf`): `±∞` are written as `"inf"`/`"-inf"`
  (the core `serde_util` convention) and NaN, which means *undefined* (a
  vignetted field, a failed ray), as `null`. Reading accepts numbers, `null`,
  `"inf"`, `"-inf"` and `"nan"`. See CONVENTIONS §8.

## 2. Analyses

| function | module | method in one line |
|---|---|---|
| `spot_diagram` | `spot` | pupil pattern traced per field × wavelength; centroid, RMS/GEO about centroid and chief, weighted polychromatic stats, vignetted fraction, Airy radius `1.22 λ N_w`, through-focus by transfer along the rays |
| `ray_fan`, `opd_fan` | `fans` | tangential `(0, P_y)` and sagittal `(P_x, 0)` fans; `εy`/`εx` about the primary (or same-wavelength) chief; OPD from the exit-pupil reference sphere of `mklight-trace` |
| `field_curvature` | `curvature` | Coddington vergence equations along the real chief ray, cross-checked by close rays; distortion vs the paraxial chief height, `f·tan θ`, `f·θ` |
| `grid_distortion` | `curvature` | chief rays of a square field grid vs their paraxial image points |
| `seidel` | `seidel` | per-surface `S_I…S_V`, `C_L`, `C_T` from the paraxial marginal and chief rays (aspheric terms included), Petzval sum and radius, `W040…W311`, third-order transverse aberration |
| `longitudinal_aberration` | `seidel` | on-axis zonal rays: crossing with the chief ray vs the primary paraxial focus |
| `chromatic_focal_shift` | `seidel` | paraxial focus of a wavelength sweep (copy of the system with the sweep as its wavelengths) |
| `lateral_colour` | `seidel` | real chief heights per wavelength vs field |
| `relative_illumination` | `illumination` | projected solid angle of the beam at the image point, with real ray aiming (§2.4) |
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
exact for a plane image surface, first order for a curved one (`best_focus`
warns). The RMS radius is then the norm of an affine function of `δ`, so the
best-focus merit (mean of per-field RMS radii) is convex and its minimiser
lies between the per-field minimisers; that interval is Brent's default
bracket. The closed-form minimiser of the mean *square* merit is reported as
a cross-check, and the merit itself is evaluated from the rays (two-pass)
because the quadratic form cancels near a sharp focus.

Pupil patterns (`util::PupilPattern`, density `rings`): `hexapolar`
(`1 + 3r(r+1)` points), `square` (pitch `1/r`, clipped to the disk),
`random` (`1 + 3r(r+1)` uniform points, `ρ = √u₁`, Philox stream 0) and
`dithered` (one jittered point per square cell, stream 1).

### 2.2 Field curvature

Thin tangential and sagittal pencils are followed along the real chief ray
with vergences `V = n/t` (`t` the distance to the focus along the ray):
transfer `V ← V/(1 − dV/n)`, refraction
`n' cos²I'/t' = n cos²I/t + (n' cos I' − n cos I) κ_t` and
`n'/s' = n/s + (n' cos I' − n cos I) κ_s`, reflection
`V_t' = V_t − 2nκ_t/cos I`, `V_s' = V_s − 2nκ_s cos I`, and the ideal lens
`V' = cos U'(V/cos U + n'/f)`. `κ_t`, `κ_s` are the surface's normal
curvatures in and across the plane of incidence, from the principal
curvatures of the surface of revolution (`κ_m = f''/(1+f'²)^{3/2}`,
`κ_s = f'/(r√(1+f'²))`, evaluated without dividing by `r`). This is exact
whenever the plane of incidence is a principal plane (meridional chief rays
of centred and x-folded systems); twist terms for skew chief rays are
neglected. Each value is cross-checked by close rays at pupil offsets `±h`
(central differences, `O(h²)`); the two agree to about 10⁻⁶ mm on every
sample. The reported curve is the axial (`z`) focus shift from the image
surface.

### 2.3 Seidel conventions

Signed reduced indices make mirrors regular (`n' = −n`). With these signs a
positive singlet has `S_I > 0`, `W040 > 0` matches the OPD sign of
CONVENTIONS §4, and the third-order transverse aberration is
`εy = (1/(2ω'))[S_I ρ³cos φ + S_II H ρ²(2 + cos 2φ) + (3S_III + S_IV)H²ρ cos φ + S_V H³]`
with `ω'` the image-space reduced marginal slope (negative when converging);
colour: `Δz(short − long) = −n'C_L/ω'²`, `Δy = C_T H/ω'`. `S_V` is evaluated
in the division-free form `−Ā³yΔ(1/n²) + Āȳc(Āy + H)Δ(1/n)`. Ideal lenses
are aberration-free and contribute nothing (with a warning).

### 2.4 Relative illumination: method

For a uniform Lambertian object the image irradiance is
`E = L (n'/n)² Ω_p` with `Ω_p` the projected solid angle of the converging
beam, i.e. its area in the direction-cosine plane `(L, M)` of the image
surface. The beam is generated with **real ray aiming** (a real-aiming copy
of the tracer when the system uses paraxial aiming), so the normalised pupil
maps linearly onto the real stop and every field's beam fills exactly the
stop. Then `Ω_p = ∫∫_{disk} s(P) |∂(L,M)/∂(P_x,P_y)| dP`, with `s` the
survival of the clipped ray (fixed apertures vignette) and the Jacobian from
central differences of unclipped rays. Polar Gauss–Legendre × trapezoidal
quadrature integrates an unvignetted beam to about 10⁻⁹ (checked against the
exact disk view factor of an ideal lens); a vignetting edge limits it to
`O(1/N)`. `RI(H) = Ω_p(H)/Ω_p(0)`; the vignetting fraction is the clipped
share of the stop area.

### 2.5 Layout

All geometry is global (`ResolvedSurface::frame`). Semi-diameters come from
`SequentialTracer::semi_diameters` (stop: `max(paraxial, traced)`); a zero
value falls back to 5 % of the largest one. Elements are consecutive
refracting surfaces with glass between them (cemented doublets share a
surface); the outline uses `EdgeSpec::mechanical_semi_diameter` or the clear
radius plus `edge_margin` (0.5 mm; the model defines no default margin),
with edge flats at the sag of each clear edge. Planes between equal media
(pure aperture stops) and dummies get no profile. `in_section` means every
point has global `x = 0`. `LayoutScene::to_buffers` gives structure-of-arrays
records (vertices as f32 relative to the scene centre plus absolute f64,
offsets, kind codes 0–6 as `PolylineKind`, 7 for element outlines, −1 for
"no index") and `LayoutBuffers::to_le_bytes` one little-endian blob
(`"MKLL"`, version 1; layout in its doc comment).

## 3. Validation

| test file | what it pins |
|---|---|
| `tests/curvature.rs` | Coddington = close rays (≤10⁻⁵ mm rel.) on every sample, field and wavelength; concentric mirror focal sphere `Δz = −(R/2)(1 − cos θ)` to 10⁻⁹; ideal lens flat and distortion-free, `f·θ` distortion `100(tan θ/θ − 1)` |
| `tests/seidel.rs` | spherical mirror closed forms (stop at mirror / at centre, paraboloid `S_I = 0`, `W040 = y⁴/4R³`, Petzval radius `R/2`); thin plano-convex lens vs thin-lens `S_I…S_V`; Petzval sum = `Σφ/(nn')` on every sample; Seidel vs real rays (spherical, coma sign and size, astigmatism, sagittal coma, distortion) on the Cooke triplet and singlet at reduced aperture; `W040` vs OPD; exact mirror LSA `(R/2)(sec I − 1)`; thin-lens colour `−f/V`, `C_L`, `C_T` vs real colour |
| `tests/illumination.rs` | RI of an ideal lens = exact disk view factor (10⁻⁸) and ≈ `cos⁴θ`; vignetting by a field aperture = disk-overlap area; footprints |
| `tests/spot_energy.rs` | defocused ideal lens: RMS/GEO exactly the scaled pupil statistics, EE vs exact grid counts and `(r/R)²`, geometric MTF vs `2J₁(x)/x`; best focus recovers a known defocus and equals the retraced spot; paraboloid on-axis spot ≈ 0; polychromatic weighting; vignetted fraction; fans vs direct traces and symmetry; seeds; cancellation and progress; JSON round trips with NaN |
| `tests/layout.rs` | global profile end points, elements of cemented lenses, folded Newtonian in global coordinates, vignetted ray end points, buffer round trip |
| `tests/system_data.rs` | report vs paraxial data, afocal `"inf"` serialisation |
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
