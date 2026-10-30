# Wave optics (`mklight-wave`)

Diffraction image quality from the sampled exit-pupil function: OPD maps,
Zernike fits, PSFs (FFT, exact matrix transform, Huygens), OTF/MTF (FFT,
pupil autocorrelation, from a PSF), encircled/ensquared energy, line spread,
Strehl ratios, detector (pixel) MTF, and the link that builds pupils from
the CPU f64 reference tracer (`mklight-trace`). Read `docs/CONVENTIONS.md`
first (§4 OPD sign, §13 pupil coordinates); the module docs in
`crates/mklight-wave/src/*.rs` hold the exact formulas.

## 1. Modules

| module | contents |
|---|---|
| `pupil` | `PupilGeometry` (exit-pupil radius `a`, reference radius `R`, `n'`, chief-ray frame; `R = ∞` for telecentric beams), `PupilFunction` (`n × n` cell-centred grid of amplitude, coverage, OPD), `PupilMask`, statistics, refocusing |
| `traced` | the tracer link: `trace_pupil(tracer, field, wavelength, &PupilSettings, ctx)` → `TracedPupil`; `tracer_for_configuration` |
| `interp` | resampling traced rays onto the grid (triangulated lattice with a rim mask, or a Zernike fit of scattered rays) |
| `zernike` | Fringe (Z1–Z37) and Noll-normalised Standard polynomials, weighted least-squares fits, Seidel-equivalent terms |
| `psf` | FFT / matrix-transform PSF on the image surface, Huygens–Fresnel PSF on any plane, polychromatic PSF |
| `mtf` | OTF by FFT (with lattice interpolation), direct pupil autocorrelation, MTF from a sampled PSF, through-focus, analytic circular/annular MTF |
| `energy` | encircled/ensquared energy through the band-limited interpolant, line-spread functions, Strehl estimates |
| `detector` | pixel-aperture and diffusion MTF, Nyquist frequencies |
| `analysis` | engine-facing analyses with serde settings/results and `RunInfo`, plus the `ANALYSES` registry table |

## 2. Pupil conventions

* **Grid.** `n × n` cell-centred samples over `[−1, 1]²`; the unit disk is
  the pupil scale. Index `j·n + i`, `i` along `u` (image `x`, sagittal),
  `j` along `v` (image `y`, tangential).
* **Coordinates are positions on the reference sphere.** The point
  `R·(α ê_u + β ê_v − γ ê_w)` (relative to the sphere's centre `P`) has
  `(u, v) = (α, β)/sinθ'max`. An ideal ray from it travels along
  `−α ê_u − β ê_v + γ ê_w`, so a ray *direction* `k̂` corresponds to
  `−(k̂·ê_u, k̂·ê_v)/sinθ'max` (`PupilGeometry::pupil_coords`): a ray from the
  top of the pupil travels downwards. This is the orientation for which the
  PSF kernel `exp(+i2πν·x)` and the Huygens sum agree and a positive tilt
  `W = t·v` moves the image toward `−y` (`ε = −(λ/NA)∂W/∂v`). (Before the
  tracer link existed, `pupil_coords` returned `+k̂·ê/sinθ'max`, the
  point-reflected position; that was a sign error, fixed here.)
* **Chief-ray frame** `(ê_u, ê_v, ê_w)`: the image surface axes carried by
  the smallest rotation onto the chief ray. Off-axis results are reported on
  the image surface via the projection `B` (`pupil` module docs).
* **Telecentric beams.** `PupilGeometry::telecentric(sinθ'max, n')` has
  `a = R = ∞` (serialised as `"inf"`) and stores `sinθ'max`. FFT, MTF and
  energy routes depend only on `sinθ'max`; refocusing is
  `W = (n'/λ)·δz·(1 − γ)`, and `huygens_psf` becomes an exact plane-wave sum
  `Σ (cA/γ)·e^{i2πW}·e^{−ik'k̂·x}` (valid on tilted and defocused planes).

## 3. The tracer link (`traced`)

`trace_pupil` builds the pupil of one (field, wavelength) of a
`SequentialTracer`. A tracer is one configuration;
`tracer_for_configuration(system, library, k)` applies configuration `k`
and its solves (`mklight_trace::apply_solves_configuration`).

1. **Reference.** The unclipped chief ray gives `I`, `d_c` and its optical
   path. The paraxial exit pupil at that wavelength fixes
   `E = I + s_E·d_c` (as `mklight_trace::opd`). The sphere is centred on
   `P` (`ReferencePoint::Chief`: `P = I`; `Centroid`: the flux-weighted
   centroid of the traced rays on the image surface) and passes through
   `E`: `R = |E − P|`. An image-space telecentric system (paraxial exit pupil
   at infinity) is computed in the limit `R → ∞`.
2. **Lattice.** `rays × rays` entrance-pupil points `P_x, P_y ∈ [−1, 1]`
   (odd counts put the chief ray on a node) plus a guard ring outside the
   disk, traced with clipping (vignetting factors and ray aiming from the
   tracer). Each surviving ray is placed at its sphere position and carries
   its OPD and amplitude.
3. **Rim.** Marching squares on the survival of the lattice nodes gives
   closed boundary loops; every crossed lattice edge is refined by bisection
   to `rim_tolerance` (1e-10), the disk edge analytically when its point
   survives; saddle cells are resolved by tracing their centre. The
   surviving end of each bracket is placed on the pupil, and the loops
   become the `PupilMask` (even–odd rule: obscurations are holes). The same
   loops in entrance-pupil coordinates give `transmitted_fraction`.
4. **Resampling** onto the `grid × grid` pupil (`interp::grid_from_structured`
   with that mask; edge cells extrapolate the nearest triangle).

**OPD** (CONVENTIONS §4 sign), for a ray reaching the image at `Q` with
direction `d`, `w = Q − P`, `b = d·w`, `p² = |w × d|²`, `S = √(R² − p²)`,
`σ = sgn(s_E)`:
`OPD = (OPL_c − OPL_r) + n'·[b_r − b_c + σ·(p_r²/(R + S_r) − p_c²/(R + S_c))]`.
With `P = I` it is the formula of `mklight_trace::opd` (tested to 1e-9
waves on every lattice ray of four sample systems). As `R → ∞` it becomes
`(OPL_c − OPL_r) + n'(b_r − b_c)`, the phase of each ray's plane wave at
`P`, which is what the Debye integral needs. **Note:** for telecentric
systems `mklight_trace::opd` instead refers the OPD to the plane through
`I` normal to the chief ray; the two differ by `O(δz·θ⁴)` for a defocused
image (and agree in focus). The wave link uses the `R → ∞` limit of §4.

**Pupil scale.** `sinθ'max` is the largest `|(α, β)|` of the traced rays and
rim points, so the grid always holds the whole beam. It is a scale, not a
claim about the F/#: `ExitPupilInfo::effective_f_number` uses the clear
area (`1/(2n' sinθ'max √(area/π))`).

**Mirrors.** When the light reaches the image toward local `−z`, all
vectors are mirrored to `(x, y, −z)` before the frame is built. Distances
are unchanged, so OPD, PSF and MTF are too, and the transverse axes stay the
image surface's `x`, `y` for refracting and reflecting systems alike
(`ExitPupilInfo::propagation_sign = −1`).

**Amplitudes** (`AmplitudeModel`), normalised to the chief ray:

| model | `A` | notes |
|---|---|---|
| `uniform` | 1 | classical |
| `transmission` (default) | `√T` | `T` from `mklight_trace::radiometry` (unpolarised model: coatings, Fresnel, bulk absorption) |
| `radiometric` | `√(T·m·γ/J)` | adds the ray density `J = |∂(u,v)/∂P|` and the source measure `m` (plane-wavefront area for an object at infinity, `|∂(L,M)/∂P|/cosθ` for an isotropic point source), by lattice differences |

**Image surface.** PSFs live on the plane through `P` normal to the image
surface's vertex `z` axis (a warning flags curved image surfaces).

## 4. Detector MTF (`detector`)

`DetectorSettings { pitch_mm: [p_x, p_y], fill_factor, aperture_mm,
diffusion_sigma_mm }`:
`MTF_det(ν) = |sinc(w_x ν_x) sinc(w_y ν_y)|·exp(−2π²σ²|ν|²)`, aperture widths
`w = p·√FF` (or explicit), so a fully filled pixel has its first zero at
`1/p`. Nyquist along a unit direction `e` is
`min(1/(2p_x|e_x|), 1/(2p_y|e_y|))`. The MTF analyses report the detector
curve, the product `system_* = optical × detector`, and the Nyquist
frequencies. The phase-dependent sampling MTF is not included.

## 5. Analyses (`analysis`)

Shape as in `mklight-analysis` (`docs/ANALYSIS.md` §1): `analysis(tracer,
&settings)` and `analysis_job(tracer, &settings, ctx)` (cancellation between
pupils, blocks of rays and image rows; monotone progress ending at 1);
settings are `#[serde(default)]` structs; results carry `run_info`
(backend `cpu-f64`, precision `f64_reference`, tracer and analysis
warnings, including the wave sampling warnings). Selections use
`mklight_analysis::util::{FieldSelection, WavelengthSelection}` (same
JSON). Every analysis takes `pupil: PupilSettings` (`grid` 64, `rays` 65,
`supersample` 4, `reference`, `amplitude`, `rim`, `rim_tolerance`).

| name | function | result |
|---|---|---|
| `wavefront_map` | `wavefront_map` | per (field, wavelength): exit-pupil info, OPD grid (`null` outside), amplitude, RMS/PV (with and without tilt), Strehl at the reference and Maréchal, Zernike set (fit to the lattice rays, weights `A²`), residual, condition, Seidel terms, rim loops |
| `psf` | `psf_analysis` | FFT (exact matrix transform on the requested grid) or Huygens on the image surface; polychromatic or per wavelength; Strehl, centroid, peak; `reference_mm` is the field's common reference point |
| `mtf` | `mtf_analysis` | tangential/sagittal MTF vs frequency by `fft`, `direct` (autocorrelation) or `huygens` (DFT of the Huygens PSF); diffraction limit; detector |
| `mtf_vs_field` | `mtf_vs_field` | MTF at fixed frequencies over `field_samples` normalised fields along `direction` |
| `through_focus_mtf` | `through_focus_mtf_analysis` | MTF at fixed frequencies vs image-plane shift `δz` (reference shift `δz/cosθ_c` along the chief ray) |
| `diffraction_encircled_energy` | `diffraction_encircled_energy` | encircled and ensquared energy of the PSF, diffraction-limited reference, `r50/r80/r90` |
| `line_spread` | `line_spread` | `∫PSF dy` and `∫PSF dx` profiles |

**Registry.** `mklight_wave::analysis::ANALYSES` lists
`WaveAnalysisEntry { name, title, description, default_settings, run }` with
`run: fn(&SequentialTracer, &serde_json::Value, &JobContext) ->
MkResult<serde_json::Value>`; a null or partial settings object takes the
defaults. An engine registers all of them with one loop (category `wave`).

**Polychromatic weighting.** Spectral weights are the system's wavelength
weights over the selected wavelengths. PSFs are normalised to unit energy
per wavelength before weighting, OTFs likewise; each wavelength's pupil is
placed at its own reference point relative to the field's common reference
(primary-wavelength chief point, or the weighted mean of the centroids),
so lateral colour appears in the PSF and as OTF phase.

**Sampling.** The default PSF spacing is `λ_min F#/padding` (`padding` 4).
A PSF grid wider than the alias-free field `n·λF#` wraps (warning
`field_wraps`); use `grid·spacing ≤ n·λF#` (e.g. 128 samples at padding 2
for `n = 64`).

## 6. Validation

| test file | what it pins |
|---|---|
| `tests/analytic.rs`, `tests/regressions.rs` | the pupil-function layer against closed forms (Airy pattern and EE, circular/annular/square MTF, defocus Strehl, Parseval, FFT vs Huygens, tilt direction, polychromatic sums, sampling warnings, determinism, cancellation, resampling) |
| `tests/traced.rs` | lattice OPD = `mklight_trace` OPD (1e-9) on triplet, Newtonian, finite relay and Petzval asphere; diffraction-limited ideal lens and paraboloid (Strehl, analytic MTF to 2e-3, Airy zero at 1.2197 λN), achromat at F/10 and singlet at F/20 (Strehl > 0.99, MTF within 0.01, scale = paraxial working F/# to 1 %); defocus: every ray = exact sphere-to-sphere closed form (1e-8 waves), Zernike focus = `δz NA²/(2λ)`, RMS = `W020/√12`; spherical: `6·Z9` = Seidel `W040` (3 %), fit = traced marginal OPD; Cooke triplet off axis: FFT vs Huygens (0.03) vs direct (0.02) MTF, tangential ≠ sagittal; 10-wave defocus: wave MTF → geometric MTF (0.05); vignetted triplet: rim area = 801² ray-grid transmission (2e-3), coverage = mask area, rim beats the hull; telecentric: `R = ∞`, OPD = `n δz (1 − cosθ)` (1e-8), refocus, FFT = plane-wave Huygens (1e-9), chief ray parallel off axis; mirrors: single paraboloid (`propagation_sign = −1`, defocus closed form), Newtonian, and the PSF centroid = ray centroid off axis (paraboloid, Newtonian, triplet; 3 %); radiometric amplitude of an ideal lens `A² = 1/cos³θ` (2e-3) and flat for a 1:1 finite relay; detector sinc zeros at `1/p`, fill factor, diffusion, product and Nyquist; registry JSON runs, round trips, determinism, cancellation and progress; polychromatic = weighted sum of placed monochromatic PSFs; configurations and through-focus symmetry; Airy EE, `r50`, `r80` and LSF normalisation; MTF vs field = MTF at the system fields |

## 7. Open points

* `mklight_trace::opd` uses a plane at the image for telecentric systems
  (§3 note); aligning it with the `R → ∞` limit would make the geometric OPD
  fans and the wave maps agree for defocused telecentric images.
* A curved image surface uses the vertex-frame plane through the reference
  point; a tangent-plane frame would need the surface normal at `P`.
* The rim polygon is inscribed between bisected boundary points (area
  deficit `≈ 2π²/(3N²)` for `N` vertices, 1.6e-4 at `rays = 65`).
* `mtf_vs_field` sweeps normalised field points without the system
  fields' vignetting factors (the traced rim still models real
  vignetting).
* Polarisation (vector pupils from `mklight_trace::vector_pupil`) and GPU
  pupil tracing are not wired in; the link is CPU f64.
