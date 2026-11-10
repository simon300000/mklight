# Wave optics (`mklight-wave`)

Diffraction image quality from the sampled exit-pupil function: OPD maps,
Zernike fits, PSFs (FFT, exact matrix transform, Huygens), OTF/MTF (FFT,
pupil autocorrelation, from a PSF), encircled/ensquared energy, line spread,
Strehl ratios, detector (pixel) MTF, and the link that builds pupils from
the CPU f64 reference tracer (`mklight-trace`). Read `docs/CONVENTIONS.md`
first (§4 OPD sign, and the section "Wave-optics pupils" for pupil
coordinates); the module docs in
`crates/mklight-wave/src/*.rs` hold the exact formulas.

## 1. Modules

| module | contents |
|---|---|
| `pupil` | `PupilGeometry` (exit-pupil radius `a`, reference radius `R`, `n'`, chief-ray frame; `R = ∞` for telecentric beams), `PupilFunction` (`n × n` cell-centred grid of amplitude, coverage, OPD), `PupilMask`, statistics, refocusing |
| `traced` | the tracer link: `trace_pupil(tracer, field, wavelength, &PupilSettings, ctx)` → `TracedPupil` (= `trace_lattice` → `PupilLattice`, then `PupilLattice::resample` and `assemble`); `tracer_for_configuration` |
| `shared` | traced lattices and resampled pupils shared between analyses through a `mklight_core::memo::Memo` (the engine's analysis graph), keyed by content |
| `interp` | resampling traced rays onto the grid (triangulated lattice with a rim mask, or a Zernike fit of scattered rays) |
| `zernike` | Fringe (Z1–Z37) and Noll-normalised Standard polynomials, weighted least-squares fits, Seidel-equivalent terms |
| `psf` | FFT / matrix-transform PSF on the image surface, Huygens–Fresnel PSF on any plane, polychromatic PSF |
| `mtf` | OTF by FFT (with lattice interpolation), direct pupil autocorrelation, MTF from a sampled PSF, through-focus, analytic circular/annular MTF |
| `energy` | encircled/ensquared energy through the band-limited interpolant, line-spread functions, Strehl estimates |
| `detector` | pixel-aperture and diffusion MTF, Nyquist frequencies |
| `analysis` | engine-facing analyses with serde settings/results and `RunInfo`, plus the `ANALYSES` registry table |
| `image_simulation` | space-varying incoherent image simulation (§8) |
| `sampling` | system-derived sampling defaults |
| `fft` | 2-D FFTs on square grids and the exact matrix Fourier transform |
| `warnings` | sampling diagnostics attached to results |
| `linalg`, `special` | least-squares QR; Bessel functions, the Airy pattern, Gauss–Legendre nodes |
| `gpu` | the resident scalar-wave adapters (§9) |

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
  the smallest rotation onto the chief ray. FFT PSFs are reported on the
  image surface via the projection `B` (`pupil` module docs); the MTF uses
  the angular spectrum on the surface (§5).
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
   On the CPU, lattice rays are aimed and traced in packets, and the
   crossings are bisected in lockstep (one probe per crossing per round,
   probes traced together, survival only; the kept ray's throughput is
   computed once). Both are bitwise identical to tracing and bisecting
   one ray at a time (CONVENTIONS §7, "CPU packets").
4. **Resampling** onto the `grid × grid` pupil (`interp::grid_from_structured`
   with that mask; edge cells extrapolate the nearest triangle).

**OPD** (CONVENTIONS §4 sign), for a ray reaching the image at `Q` with
direction `d`, `w = Q − P`, `b = d·w`, `p² = |w × d|²`, `S = √(R² − p²)`,
`σ = sgn(s_E)`:
`OPD = (OPL_c − OPL_r) + n'·[b_r − b_c + σ·(p_r²/(R + S_r) − p_c²/(R + S_c))]`.
With `P = I` it is the formula of `mklight_trace::opd` (tested to 1e-9
waves on every lattice ray of four sample systems). As `R → ∞` it becomes
`(OPL_c − OPL_r) + n'(b_r − b_c)`, the phase of each ray's plane wave at
`P`, which is what the Debye integral needs. `mklight_trace::opd` takes the
same limit, so wave maps and geometric OPD fans agree for telecentric
systems too (a defocus `δz` beyond focus gives `+n'δz(1 − cosθ)`).
(`mklight_trace::opd` used to refer telecentric OPDs to the plane through
`I` normal to the chief ray, `n'(w·d_c)/(d·d_c)`; that is not the
`R → ∞` limit and reported defocus with the opposite sign,
`n'δz(1 − 1/cosθ)`, differing at `O(δz·θ²)`. It was fixed together with
this link; `tests/telecentric_opd.rs` in `mklight-trace` and the wave
telecentric test pin it.)

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

The default `transmission` model leaves out the ray density (pupil
distortion) and obliquity: rays are resampled onto a uniform `(u, v)` grid,
so where the exit pupil is distorted the PSF's energy distribution, its
centroid and its encircled energy are biased. On the singlet sample at
full field the PSF centroid falls about a third short of the ray centroid
with `transmission` and within a few per cent with `radiometric`
(`default_amplitude_model_biases_the_psf_centroid`). Use `radiometric`
when PSF energy placement matters (centroids, encircled energy of
systems with pupil aberration); wavefront maps, Zernike fits and Strehl
ratios of well-corrected systems are hardly affected.

**Image surface.** PSFs live on the plane through `P` normal to the image
surface's vertex `z` axis (a warning flags curved image surfaces).

**Shared pupils.** `trace_pupil` is two stages: `trace_lattice` (lattice and
rim rays, reference sphere, placed samples, masks and the automatic grid
decision; independent of `grid` and `supersample`) and the resampling onto
an `n × n` grid. Every `*_shared` analysis job (`psf_analysis_shared`,
`mtf_analysis_shared`, …) takes an optional memo and requests all its
(field, wavelength) lattices and grids from it in one batch (`shared`): the
lattice key digests the tracer's configured snapshot and settings, the
prepared field bit for bit, the wavelength and the lattice settings; a grid
key adds `n` and the supersampling. PSF, MTF, LSF, encircled energy,
through-focus MTF, wavefront maps and MTF vs field of one configuration
therefore trace each pupil once, window refinements resample a shared
lattice, and results are bit-identical to the unshared jobs (`*_job`).

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
JSON). Every analysis takes `pupil: PupilSettings` (`grid` 0 = automatic, `rays` 65,
`supersample` 4, `reference`, `amplitude`, `rim`, `rim_tolerance`).

**Unusable fields.** A field whose pupil cannot be built (chief ray failed,
fully vignetted, too few rays for the Zernike fit) is left out of the
per-field results with a `RunInfo` warning `field k skipped: …`, as the
geometric analyses skip fields whose chief ray fails; `mtf_vs_field`
reports `NaN` (JSON `null`) at such samples. An analysis fails only when no
selected field is usable.

**Configurations.** A tracer is one configuration, and its `RunInfo`
names the snapshot it was built from (the solved, configured system).
`configured_tracer(system, library, k)` returns a `ConfiguredTracer`
(tracer, `k`, the base snapshot hash, the solve warnings);
`WaveAnalysisEntry::run_configured` (or `ConfiguredTracer::stamp` on a
typed result) writes the session snapshot's hash, `"configuration": k`
in the settings and the solve warnings into the `RunInfo`. Engines that
analyse configurations should use it.

**Limits.** Huygens image grids are limited to `MAX_HUYGENS_GRID` = 1024
per side in every analysis (`huygens.grid_size` of the MTF analyses, and
`grid_size` of PSF, encircled energy and line spread with `method:
huygens`), and one Huygens PSF to `MAX_HUYGENS_WORK` = 2·10¹⁰ (image
points × covered pupil samples, summed over wavelengths); both are
`InvalidParam` errors naming the limit. A PSF holding less than
`MIN_ENERGY_IN_GRID` = 95 % of its energy inside the grid carries the
warning `psf_truncated`: centroids, encircled energies and Huygens MTFs
from it are biased; enlarge the grid or its spacing.

| name | function | result |
|---|---|---|
| `wavefront_map` | `wavefront_map` | per (field, wavelength): exit-pupil info, OPD grid (`null` outside), amplitude, RMS/PV (with and without tilt), Strehl at the reference and Maréchal, Zernike set (fit to the lattice rays, weights `A²`), residual, condition, Seidel terms, rim loops |
| `psf` | `psf_analysis` | FFT (exact matrix transform on the requested grid) or Huygens on the image surface; polychromatic or per wavelength; Strehl, centroid, peak; `reference_mm` is the field's common reference point |
| `mtf` | `mtf_analysis` | tangential/sagittal MTF vs frequency by `direct` (autocorrelation, default), `fft` or `huygens` (DFT of the Huygens PSF); diffraction limit; detector |
| `mtf_vs_field` | `mtf_vs_field` | MTF at fixed frequencies over `field_samples` normalised fields along `direction` (the tangential direction at every sample, the axial one included) |
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

**Automatic sampling and ranges.** `pupil.grid: 0` (the default) selects
sampling from the traced exit pupil, independently for each field and wavelength.
A positive grid is always used as supplied, as are explicit ray counts, image
grid sizes, spacings, frequency lists/limits and focus ranges. Zero image
`grid_size` (or image simulation `psf_size`) selects the automatic window.
These sentinel settings remain in RunInfo; the deterministic rule below plus
the snapshot and engine version reproduce the chosen grids. PSF and wavefront
results report their actual grids and image spacing.

* Trace the `rays × rays` entrance-pupil lattice and rim first (65 rays per
  diameter by default). In its mapped exit-pupil triangles, find
  `G = max(|∂W/∂u|, |∂W/∂v|)` in waves per normalised pupil unit.
  Choose the next power of two at least `max(64, 2G/0.35)`. The phase
  target 0.35 waves per sample leaves margin below the diagnostic's
  0.5-wave limit.
* Estimate geometric ray offsets from `ε_normal = −2 λN ∇W`, project them
  onto the image surface with `B⁻¹`, and include the wavelengths' reference
  offsets. The image half-width is the largest Cartesian offset plus
  `16 λN`, maximised over the selected wavelengths. This holds the
  aberrated flare and diffraction wings, rather than only an Airy core.
  Spacing is `min(λN)/padding`, with default padding 2.5 (safely finer
  than intensity Nyquist). Round the required image side up to whole pixels
  (at least 64; Huygens at most 1024). Encircled energy uses the largest
  field's window for all fields. Increase an automatic pupil grid further
  if needed to keep the window within every wavelength's alias-free field
  `n λN`.
* `mtf` runs to the frequency where an estimate of the polychromatic
  contrast falls to 0.1 along both image axes: the geometric
  characteristic function of the phase-slope offsets above (at most 64
  samples per pupil diameter, spectrally weighted, so lateral colour
  counts) times the diffraction-limited MTF. Unlike the RMS blur, this
  keeps the contrast of a sharp core with a little flare. Scan from
  `min(40 cycles/mm, cutoff/10)` in steps of 1.25 up to at most
  `0.8 cutoff` (the largest spectral cutoff), round up to two significant
  digits and take the smallest value over the fields. `mtf_vs_field`
  (over its sweep) and `through_focus_mtf` (over its fields) take the
  `1, 2, 5 × 10^k` value at or below a quarter of that limit and its 2×
  and 4× multiples, e.g. 10/20/40 cycles/mm for the Double Gauss.
* The default through-focus half-range is `max(8 λN², 4 N r_rms)` over
  all selected pupils (λ in mm). Refine automatic pupil grids against
  the phase steps at both focus endpoints before computing the curves.

**Cost caps.** Automatic pupil grids stop at 1024 per side (1,048,576
cells, 16 subpoints per cell at the default supersampling); automatic
image grids stop at 2048 per side (2047, odd, in image simulation). The
trace lattice remains the requested `rays` count. Polychromatic PSFs (and
image simulation) evaluate the exact matrix transform by chirp
convolutions of 7-smooth length `≥ n + M − 1` on large grids: separable
surface maps by two 1-D convolutions, skew maps by one 2-D convolution
(the chief-frame projection `B` is symmetric). Huygens retains its
separate work/grid limits above.
These are per pupil/PSF caps, not a wall-time guarantee for arbitrary numbers
of fields or wavelengths. Cancellation and progress also cover refinements.

Fields, wavelengths and refinements run in parallel; reductions keep the
selection order, so results are deterministic. Through-focus prepares the
autocorrelation stencils of each frequency once and only adds the exact
defocus OPD per plane. Encircled energy sums Fourier bins of equal `|ν|`
first and shares each Bessel kernel across the fields and diffraction
references.

Results report the actual pupil grids as `pupil_sampling` lists of
`(wavelength, grid)` after every refinement: per field for `psf`, `mtf`,
`through_focus_mtf`, `line_spread` and `diffraction_encircled_energy`, per
sweep sample for `mtf_vs_field` (empty for skipped samples) and per node in
image simulation's `field_mapping`. Encircled energy also reports
`grid_size` and `spacing_mm`.

Diagnostics are retained at the cap and for explicit coarse settings:
`pupil_undersampled` above half a wave per pupil cell, `field_wraps` when
`grid × spacing > n λN`, `image_undersampled` above intensity Nyquist, and
`psf_truncated` below 95% energy. Automatic defaults pass those checks on
all built-in samples, including the anamorphic, freeform and DOE designs;
this is a sampling check, not a claim that those designs are corrected.

**Anamorphic coverage.** The pupil mask is the actual traced boundary on
the reference sphere, not an assumed circle. The sparse-ray diagnostic
compares each extrapolation gap with the longer local lattice edge (the
median triangle edge), rather than `sqrt(2 × triangle area)`. The latter
is a geometric mean of unequal pitches and falsely flagged the ordinary
rim of a squeezed pupil. A genuinely missing region still warns (tested
with rays absent over half of a masked aperture).

**MTF methods.** `direct` (default) evaluates the pupil autocorrelation at
each requested frequency. `fft` is exact on its lattice `Δν = cutoff/n`
and interpolated between lattice points (`Otf`). For strongly aberrated
pupils that interpolation is poor within the first few `Δν`: singlet at
full field, `n = 64` (`Δν` = 5.4 cycles/mm), gives 0.092 at 5 cycles/mm
where the converged value is 0.037 (`direct`: 0.037). Elsewhere the two
agree to about 0.02 (Cooke triplet off axis). `huygens` needs the whole
PSF inside its window (see `psf_truncated` above).

**Off-axis MTF.** On the image surface the field is the angular spectrum of
the directions `q = α·ê_u + β·ê_v − γ·ê_w` (`γ = √(1 − α² − β²)`) with
kernel `exp(ik'(q_x x + q_y y))`. With the Jacobian
`|∂(q_x, q_y)/∂(α, β)| = |q_z|/γ` the spectrum is `c·A·exp(i2πW)/|q_z|`,
and the OTF is its autocorrelation under the shift `λν/n'` in
`(q_x, q_y)`. The chief-normal projection `B` (§2) amounts to replacing
`−γ·ê_w` by `−ê_w`; that drops the longitudinal phase across an off-axis
blur (`|x|·sinθ_c·NA²/(2λ)` waves) and does not shrink with sampling.
`direct` maps each shifted direction back to the pupil lattice and
interpolates the pupil there. `fft` samples the spectrum at the directions
whose transverse part is `α·ê_u + β·ê_v − γ_rim·ê_w`, a linearisation of
the lattice through the aperture rim, so its transform is still the PSF at
`B·x` on the normal-plane lattice `Δν = cutoff/n` (read at `B^{−T}·ν`).
Only the remainder `(γ_rim − γ)·ê_w`, a fraction of a cell that vanishes
on the rim (no blurred edge), is interpolated. On axis both are unchanged, bit for bit.
On the double Gauss (F/2) the change reaches 0.015 at the 14° field, where
the result agrees with the transformed Huygens PSF to 0.01 (the projection
was off by up to 0.02). Pupils crossing the sensor horizon (`q_z ≥ 0`) are
rejected. The FFT **PSF** still uses `B`; use the Huygens PSF when the
longitudinal phase matters.

## 6. Validation

| test file | what it pins |
|---|---|
| `tests/analytic.rs`, `tests/regressions.rs` | the pupil-function layer against closed forms (Airy pattern and EE, circular/annular/square MTF, defocus Strehl, Parseval, FFT vs Huygens, tilt direction, polychromatic sums, sampling warnings, determinism, cancellation, resampling) |
| `tests/sensor_mtf.rs` | off-axis MTF on the image surface: Gaussian angular spectra with quadratic phase at 55° (two azimuths) against the closed-form autocorrelation (direct and FFT, 1.5e-3); a sensor translation cancels against the image offset in the complex OTF; a defocused F/2.1 ideal lens at 55° against a spherical-wave sum over the lens plane (2e-3 at 40 and 60 cycles/mm) |
| `tests/traced.rs` | lattice OPD = `mklight_trace` OPD (1e-9) on triplet, Newtonian, finite relay and Petzval asphere; diffraction-limited ideal lens and paraboloid (Strehl, analytic MTF to 2e-3, Airy zero at 1.2197 λN), achromat at F/10 and singlet at F/20 (Strehl > 0.99, MTF within 0.01, scale = paraxial working F/# to 1 %); defocus: every ray = exact sphere-to-sphere closed form (1e-8 waves), Zernike focus = `δz NA²/(2λ)`, RMS = `W020/√12`; spherical: `6·Z9` = Seidel `W040` (3 %), fit = traced marginal OPD; Cooke triplet off axis: FFT vs Huygens (0.03) vs direct (0.02) MTF, tangential ≠ sagittal; 10-wave defocus: wave MTF → geometric MTF (0.05); vignetted triplet: rim area = 801² ray-grid transmission (2e-3), coverage = mask area, rim beats the hull; telecentric: `R = ∞`, OPD = `n δz (1 − cosθ)` (1e-8), refocus, FFT = plane-wave Huygens (1e-9), chief ray parallel off axis; mirrors: single paraboloid (`propagation_sign = −1`, defocus closed form), Newtonian, and the PSF centroid = ray centroid off axis (paraboloid, Newtonian, triplet; 3 %); radiometric amplitude of an ideal lens `A² = 1/cos³θ` (2e-3) and flat for a 1:1 finite relay; detector sinc zeros at `1/p`, fill factor, diffusion, product and Nyquist; registry JSON runs, round trips, determinism, cancellation (before and during a run, from another thread) and progress (sampled: monotone, with intermediate values); polychromatic = weighted sum of placed monochromatic PSFs; configurations and through-focus symmetry; Airy EE, `r50`, `r80` and LSF normalisation; MTF vs field = MTF at the system fields; `R → ∞` lattice OPD = `mklight_trace` OPD on and off axis; exit pupil beyond the image (`σ = +1`): every ray = far-sheet closed form (1e-8), positive defocus; annular stop: two rim loops, transmission `1 − ε²` (1e-3), mapped clear area (2e-3), analytic annular MTF (3e-3); PSF centroid = ray centroid for `−x` and diagonal fields (triplet, paraboloid); `transmission` amplitude biases the singlet's centroid by > 15 %, `radiometric` < 10 %; Huygens grid and cost limits (`InvalidParam`, no tracing); `psf_truncated` on a defocused PSF but not an Airy one; unusable fields skipped with warnings (`null` in MTF vs field, error when no field is usable); the axial MTF-vs-field sample follows the sweep direction (non-square pixels); configured runs record the base hash, configuration and solve warnings; default MTF right near zero frequency where the FFT interpolation is not |

Unit tests compare the chirp transforms with direct sums (including a skew
projection), the grouped energy kernels with the full spectral integrals
and the cached through-focus curves with independent autocorrelations
(2e-12). `tests/interactive.rs` holds an ignored timing run of every default
wave analysis and image simulation on built-in samples:

```sh
cargo test --release -p mklight-wave --test interactive -- --ignored --nocapture
```

## 7. Open points

* A curved image surface uses the vertex-frame plane through the reference
  point; a tangent-plane frame would need the surface normal at `P`.
* The rim polygon is inscribed between bisected boundary points (area
  deficit `≈ 2π²/(3N²)` for `N` vertices, 1.6e-4 at `rays = 65`).
* `mtf_vs_field` sweeps normalised field points without the system
  fields' vignetting factors (the traced rim still models real
  vignetting).
* Vector diffraction is not implemented.
* The automatic grid resolves the phase interpolated from the traced lattice;
  it does not adaptively discover features smaller than the `rays` spacing.
  Increase `pupil.rays` for sharply localised wavefront structure.
* The default amplitude model is `transmission` (§3); `radiometric` is
  more faithful for PSF energy placement but not yet the default.

## 8. Image simulation

`image_simulation` (`image_simulation` module, registered by the engine)
accepts built-in `grid`, `siemens_star`, `slanted_edge`, and `text_chart`
patterns (grid and slanted-edge pixels are analytically area-averaged;
star and text use deterministic 16×16 midpoint supersampling), `source:{kind:"buffer",width,height,channels:1|3,data:[…]}`
(row-major interleaved, finite nonnegative linear-light samples), or
`source:{kind:"path",path:"…"}` (an 8- or 16-bit grey or RGB PNG, alpha
ignored, decoded from sRGB; other formats come as a buffer).
Output dimensions are `size:[width,height]`; `monochrome` selects one
channel, otherwise three. RGB source channels share the run's polychromatic
energy PSF; there is no implicit colour matching or separate channel response.
Monochrome conversion of RGB uses linear-light Rec. 709 luminance weights.
The shared engine `spectrum` setting is documented in ANALYSIS §4.

`optics` controls the square Cartesian PSF grid, normalised field half-width
`field_extent` and centre `field_center: [hx, hy]` (default `[0,0]`), traced
pupil settings, odd `psf_size` (0 = automatic), optional `psf_spacing_mm` and
pupil quadrature for illumination. The default half-width `1/√2` inscribes
the square in the design's radial field. To simulate an off-axis patch, set
its centre and a smaller half-width; its pupils, distortion and illumination
are all evaluated at those absolute normalised fields, with illumination
still referenced to the optical axis. At each field node:

1. Trace the first-order chief image (separate x/y first-order scales); the
   real image point is the PSF reference (`pupil.reference`: the primary
   chief ray by default), where the PSF centre belongs.
2. Build pupils at all wavelengths, including lateral-colour offsets relative
   to the primary chief, using radiometric amplitude by default.
3. Form the unit-energy polychromatic PSF on a common physical grid. Default
   spacing is the smallest λF/# across **all** grid nodes divided by 2.5;
   its automatic window follows the largest blur-plus-diffraction extent
   across those nodes, with automatic pupils refined to hold it. PSF warnings
   propagate; a sampled grid containing less than 95% energy warns
   `psf_truncated`. Discrete kernels are normalised to conserve image flux.
4. Integrate each wavelength's projected exit-pupil solid angle, including
   geometric vignetting, with the relative-illumination routine of
   ANALYSIS §2.4 (default 12 × 48 pupil nodes). Illumination is the weighted sum divided by its
   weighted on-axis value (coating spectral throughput is not an extra factor).

The source occupies the rectangular **first-order image plane** bounded by
that grid. Output pixel edges span the same rectangle in the real image
frame. The result provides `pixel_size_mm`, `origin_mm`, and all field nodes'
normalised fields, ideal/real chief positions and RI. Coordinates increase
with image x/y; row zero is the low-y edge (CONVENTIONS §14). All source
types use that order. PNG rows and text glyphs are converted from top-down
order at input, so upright display reads output rows from high to low y. Every output pixel inverts the
bilinearly interpolated chief mapping by Newton iteration, then resamples
the ideal source with bilinear interpolation and applies RI. Unmapped pixels
are black. Singular or folded field mappings need finer grids or a different
imaging model; multiple images are not resolved.

Each warped pixel spreads with the PSF bilinearly interpolated at its own
field: `out = Σ_k K_k ∗ (w_k · I)`, with `w_k` the pixel's bilinear weight
of grid node `k` and `K_k` that node's PSF, energy-conservingly deposited
onto the output lattice. Each tile convolves the weighted planes of the
nodes its haloed sources use (zero-padded FFT, halo = kernel radius, so
nothing wraps); tiles run in parallel and assemble in fixed order. The
result does not depend on `tile_size`, which only partitions the work.
Between nodes the PSF, distortion and RI are bilinear in the field: use a
finer `field_grid` where they vary quickly (with the default 3 nodes per
axis the Cooke triplet's RI at half field, midway between nodes, is 2.5%
low).
The engine retains the last immutable PSF grid, keyed by derived snapshot,
material library hash and every `optics` setting, across source/noise changes.
The direct Rust API can reuse an `Arc<PsfGrid>` with identical optical inputs.

The output lattice supplies detector sampling. Optional
`detector_aperture_mm:[x,y]` integrates an active rectangular pixel aperture
with 4×4 midpoint quadrature (limited to four output pixels per axis).
Optional `full_well_electrons` adds Poisson shot noise; `read_noise_sigma`
adds Gaussian read noise. Both use Philox indexed by output sample and
`seed`, recorded in RunInfo, so scheduling does not change the image.
Noise is not clipped: linear output can exceed one or become negative.
This models scalar incoherent imaging, without Bayer mosaics, demosaicing,
charge saturation or coherent scene propagation. A path input must remain
available to reproduce a run; the image file is not embedded in the result.

`image[height,width,channels]` is an f64 engine buffer. Tests cross-check
FFT halo convolution against a direct spatial sum, an ideal lens against
the Airy pattern and its blurred impulse image (including flux), distorted
grid nodes against real chief traces, tile-size independence, the PSF
centre at its reference point, PNG sRGB decoding, and shot-noise
mean/variance. `tests/defaults.rs` runs all seven wave analyses, image simulation
and the relative-illumination map with untouched defaults on every built-in
sample and rejects accuracy warnings. Additional closed-form tests pin
quadratic-phase sampling/blur, ideal-cone frequency/depth defaults and explicit
overrides.

## 9. Resident scalar-wave service

`mklight_wave::gpu` composes the `aiming` (aiming, tracing, projection),
`transforms`, `collectives` (reductions, QR), `geometry` (field maps,
illumination) and `radiometry` (scalar throughput) operations of
`mklight-gpu` through the single `ComputeBackend` resident API. The engine's native wave routes call
it without changing the reference registry; capability and `auto` routing
are in GPU.md §9.2–9.3. Solver calls report the actual backend and f32
precision.

### Host/device division

The native jobs (`gpu::analysis`, `gpu::image_simulation::build_psf_grid`)
divide each analysis between the host (the CPU f64 reference code, in
parallel) and the device (`gpu::split`): the host traces the pupil
lattices with the reference tracer (`trace_lattice`), resamples and refines
the pupils and makes every sampling decision that reads them; the device
runs the transforms (PSFs, OTFs, focus sweeps, energy, line spreads,
wavefront statistics, convolution).

Host lattices are the reference's own `PupilLattice`s (shared with the CPU
analyses through the engine memo, `shared`), and their wavefront maps use
the reference's lattice Zernike fit (identical Zernike sets and Seidel
terms, so explicit Metal requests run them; `auto` keeps `wavefront_map`
on the reference until the cost model is calibrated for it). Host
resampling yields the reference's `PupilFunction`s, written into shared
allocations in place, so the native jobs make the reference's own sampling
decisions (pupil grids and refinements, image windows, automatic
frequencies and focus ranges) and report the reference's grids.

Every field's host preparation starts at once on the thread pool, and each
prepared field is queued for up to four device
workers, so the device transforms early fields while the host still
prepares later ones. The analyses whose decisions are per field (PSF, line
spread, MTF with an explicit frequency limit) trace field by field; the
others trace all pupils first because their automatic frequencies, windows
and focus ranges span all fields. Image simulation computes the
relative-illumination rows beside the device PSFs.

The device pupil tracer below (`PupilTracer`) serves the solvers' MTF
operands.

The CPU reference runs fields, wavelengths, focus planes, image-grid nodes
and Zernike fits side by side on the thread pool (each transform parallel
inside) and merges every sum, warning and error in the serial order, so its
results are bit-identical for any thread count.

### Pupils and automatic sampling

`PupilTracer::new(backend, tracer, ctx)` prepares reusable packed geometry and
the `radiometry` material tables. `trace(field, wavelength, settings, ctx)`
generates the requested entrance lattice and guard ring on the device, traces
with hi/lo records, and uses the `radiometry` scalar product of unpolarised
interface factors and bulk loss. Wavelength chunks always include the primary
wavelength for real-image-height aiming; the radiometry operations receive
the original wavelength indices. A wavelength-only DOE
factor cancels in chief-normalised amplitudes and centroid weights; zero
selected-order throughput remains unusable.

The host chains the small boundary topology. Saddle-centre probes run only
where marching squares needs them. All field × wavelength pupils of a job share
the chief, lattice, saddle and rim commands (up to 2¹⁶ lattice rays per batch);
each pupil's rays are then split into its own channels for placement and
resampling, bit-identical to tracing it alone. Lattice guard nodes outside
the unit disk are never aimed (`AimOptions::cull_outside`): the survival mask
discards them whatever their trace, so the topology and rim edges are
unchanged while the costliest aims (unreachable targets running the whole
continuation schedule) disappear. Rim probes evaluate `RIM_LEVELS` bisection
levels per round (one: plain bisection, which aims the fewest rays — rounds
are dispatches of one plan, not host round trips), retaining the same binary
decisions and the run's aim tolerance (CONVENTIONS §7, "Aim stop"); edges already within tolerance emit
skip-flagged probes that are neither aimed nor traced. The host f64 seed
tables (25×25 per field–wavelength row, `aiming::host_tables`) seed every
generation of a batch, so the chief probe, lattice, saddle and rim probes
aim from the same seeds whatever their batch size; a traced lattice is
therefore identified exactly by `gpu::lattice_key` (`ResidentTracedPupil::
key`), the shared graph's `pupil_lattice` key in a Metal double-float family,
for analyses that share aimed pupils. The
analytic disk root uses its cancellation-free form. Each rim edge keeps the
ray of its surviving bracket end from the round that traced it (first the
inside lattice node; `Op::Rim` names the probe, `Op::Keep` gathers its
channels), exactly as the reference keeps the bracketing ray. The final point
is never traced again: re-aimed from other seeds, a ray converged within the
aiming tolerance of an aperture edge may land on its other side and clip,
dropping a sector of a vignetted off-axis rim (regression test
`tests/p6_offaxis.rs`). A failed aim whose stop residual is far
above the tolerance (> 1e-6 of the stop radius) or whose stop is unreachable
is a ray that cannot be aimed through the stop at that field; the reference
treats it as vignetted (`Tracing::ray` is `None`) and so does the survival
mask (on a fast lens 0.7–1.0 of the field, the lower pupil edge fails with
residuals of 5e-3–1e-1 on both backends, identically). Tolerance-level
non-convergence, rim brackets, sphere intersections and radiometry rows that
require the reference explicitly return `Unsupported`, rather than turning into missing rays
or silently zero amplitudes.

The chief/centroid reference uses the unclipped chief and weighted moments
(`collectives`). Small reference-frame metadata remains host control. Sphere
projection (`ProjectPupil`) and
resident amplitude/Jacobian work preserve both sheets, telecentric limits,
negative propagation and finite conjugates. A device-traced
`ResidentTracedPupil` retains its resident lattice nodes, mask and
`ExitPupilInfo`; a host-traced one (the native analyses) keeps the reference
lattice and pupil instead (`host`), uploads only the resampled pupil, and
refines on the host without tracing again.

`from_structured` and `ResidentPupil::from_pupil` remain explicit input boundaries
for existing host data. `from_resident_lattice` consumes already resident nodes.
Structured triangles, even-odd masks (including holes), supersampling, unwrapped
OPD interpolation, nearest-triangle extrapolation, coverage counts, clear area,
phase-step diagnostics and automatic-grid slopes run on device. The automatic
0.35-wave rule and native-analysis grid/work caps are unchanged. The resident
resampler bounds its supersample raster at 8192² subpoints; larger requests fail
explicitly without reducing sampling. Extrapolation
beyond 128 subpoints reports an unresolved diagnostic and requests the reference
resampler. The aperture is never replaced by a disk or convex hull. Polygon edge
crossings
are shared per supersample row, preserving the original half-open even-odd
arithmetic; this optional cache is bounded at 256 MiB. Exact left/right
column neighbours replace repeated perimeter searches for missing raster points:
minimum Chebyshev distance and the lowest row-major source tie are preserved.

### Native results and composition

`gpu::analysis` exposes backend-taking job variants of all seven native wave
analyses: wavefront, PSF, MTF, field MTF, through-focus MTF, diffraction energy
and line spread. Results retain their existing types, selected sampling,
references, warnings and `RunInfo`. `resident` metadata includes the device,
probe/build, plan version and aggregated stage/copy/fallback statistics.
Configured callers still apply `ConfiguredTracer::stamp` to retain the base
snapshot and configuration identity.

* Wavefront statistics use Debye weights, double-float reductions and a
  three-column `collectives` QR tilt fit (device basis evaluation, pivoted and
  reorthogonalised QR; a rank failure is an error). The Zernike fit and
  Seidel terms are the reference's own, from the host lattice (§9).
* `append_psf`, `append_huygens`, `append_correlation` and `append_fft_otf`
  compose into caller plans. FFT/chirp/affine transforms
  preserve the reference's negligible-skew rule before satisfying the exact
  diagonal/symmetric contracts of `transforms`. Huygens sums use rationalised distances and
  fixed 256-sample double-float blocks, one thread per pixel and block, added
  in block order (as are direct-MTF correlations, image queries and radial
  bins); no sample-by-pixel matrix is allocated.
* Direct MTF keeps the angular-spectrum mapping and coverage-weighted phase
  interpolation. FFT MTF builds the sensor spectrum and uses the same shifted
  lattice, centroid phase removal, cubic aberration interpolation and bilinear
  diffraction reference as the f64 method. Huygens MTF uses the sampled-PSF DFT.
  A covered pupil crossing the sensor horizon is rejected.
* Direct-correlation stencils (`[frequencies, n², 5]` f32, on-axis MTF and
  every focus sweep) are tiled in frequency so each stays within 256 MiB
  (`ResidentPupil::stencil_frequencies`); tile outputs are concatenated by
  `Op::Concat`, and every frequency's sum is evaluated and reduced exactly as
  in one stencil, so results do not depend on the tiling. An explicit pupil
  grid whose stencil cannot hold one frequency (4096²) is rejected by the
  direct MTF, field MTF and through-focus jobs before any pupil is traced.
* Focus sweeps share correlation stencils and cache exact defocus phases up to
  256 MiB; larger sweeps run in plane chunks that each cache their planes'
  phases (the cached phases are the values the on-the-fly path evaluates, five
  per correlation sample and plane), and only a single plane beyond the limit
  evaluates them on the fly. Per-plane GPU
  reductions retain phase-step warnings. FFT/Huygens use bounded plane batches.
  Automatic pupils are refined from the phase steps at both focus endpoints.
  Polychromatic fields/OTFs accumulate on device before taking magnitudes.
  Clearing OPD preserves lateral-colour offsets for PSF/energy references; the
  MTF diffraction reference deliberately omits the placement phase, as in f64.
* `ResidentPsf` keeps images available to subsequent MTF, energy, line-spread
  and image-simulation stages. Energy uses the band-limited spectral integrals,
  not pixel bins. Stable integer-radius sorting/compaction and segmented sums
  group equal Fourier radii; batched images share each Bessel kernel. Image
  moments and line-spread reductions also remain resident.

Published inputs are immutable shared allocations. Producers use `fill_buffer`,
consumers use `read_view`, and final owned results widen directly from those
views. `ResidentPupil::opd` is a `BufferView`, so focused planes share their batch
allocation without copying. `WaveResult` retains `PlanStats` and actual-backend
`RunInfo`; callers that compose lower-level operations (the optimiser, the
engine) must preserve the snapshot and library identity. Cancellation discards outputs and never
runs a replacement reference computation.

### Image simulation

`gpu::image_simulation::{build_psf_grid,image_simulation_job}` implements the
same scalar, linear-light model as §8. `ResidentPsfGrid` is reusable across
source/noise changes and checks snapshot, material-library and optical-settings
identity. The geometric adapter supplies distortion; the spectrally weighted RI of the grid
nodes is host f64 preparation (the CPU grid's own `illumination_rows`, recorded
as `illumination` in the grid's RunInfo settings), because the device
projected-solid-angle kernel solves every pupil node's object and aim problems
inside one command (≈1.4 s per wavelength against ≈0.2 s on the host for all
five on a 29-surface lens).

Patterns, inverse bilinear distortion, source resampling, RI multiplication,
node-weighted convolution, detector aperture integration and noise run on GPU.
PNG decoding and final result assembly stay on the host. Pixel deposition is
separable and compensated, preserving the reference's 4×4 aperture quadrature.
Convolution is zero padded with the complete kernel halo. A whole-image tile is
used when its estimated transform storage fits a quarter of the device working
set; otherwise the requested tile partition is used. This changes only work
partitioning, not sampling. RunInfo records `image_plan_version: 2` and
`effective_tile_size`. Kernel-cache storage is bounded at 64 MiB, and optical
grids exceeding half the recommended working set explicitly request fallback.

Shot/read noise uses the shared Philox raw stream `(seed, output_sample, draw)`
and the specified open-interval f32 uniform conversion. The f64 reference uses
its documented f64 conversion; compare moments and raw addressing, not an
unqualified bitwise claim for transcendental/rejection arithmetic. Fixed-device
replay is deterministic. Poisson rejection uses a cancellation-free double-float
log probability; means above 1e12 electrons or nonfinite output explicitly request
fallback. Subnormal full-well parameters are rejected at the request boundary
because FTZ would otherwise turn shot noise off. Matching-emulation branch tests and moment tests cover the qualified
range. Low-y row zero, RGB/monochrome conversion, unclipped
noise output and discrete convolution flux are preserved.

### Tests

`tests/p6_wave.rs`, `tests/p6_native.rs`, `tests/p6_boundaries.rs` and
`tests/p6_offaxis.rs` cover the resident primitives and the complete native
results against matched f64 methods, including automatic sampling, on both
shader origins; the ignored `all_gpu_traced_pupils` suite checks every
built-in field/wavelength pupil traced on the device, and
`tests/split_designs.rs` the host/device division. `tests/p6_bench.rs`
(ignored) times complete native jobs against the CPU f64 reference. Engine
routing uses the calibrated cost model of GPU.md §9.2, not these timings;
earlier measurements are in
[history/wave-ghost-stray-measurements.md](history/wave-ghost-stray-measurements.md).

