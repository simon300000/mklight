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

**Sampling.** The default PSF spacing is `λ_min F#/padding` (`padding` 4).
A PSF grid wider than the alias-free field `n·λF#` wraps (warning
`field_wraps`); use `grid·spacing ≤ n·λF#` (e.g. 128 samples at padding 2
for `n = 64`).

The pupil grid must resolve the wavefront: where it steps by more than
half a wave between samples (warning `pupil_undersampled`), the flare
falls outside the alias-free field and wraps to the opposite side. The
results are then not merely noisy: on the singlet sample at full field
(0.67 waves per sample at `n = 128`) the PSF centroid comes out with the
wrong sign, and it converges at `n = 256`. Raise `pupil.grid` until the
warning disappears.

**MTF methods.** `direct` (default) evaluates the pupil autocorrelation at
each requested frequency. `fft` is exact on its lattice `Δν = cutoff/n`
and interpolated between lattice points (`Otf`). For strongly aberrated
pupils that interpolation is poor within the first few `Δν`: singlet at
full field, `n = 64` (`Δν` = 5.4 cycles/mm), gives 0.088 at 5 cycles/mm
where the converged value is 0.041 (`direct`: 0.038). Elsewhere the two
agree to about 0.02 (Cooke triplet off axis). `huygens` needs the whole
PSF inside its window (see `psf_truncated` above).

## 6. Validation

| test file | what it pins |
|---|---|
| `tests/analytic.rs`, `tests/regressions.rs` | the pupil-function layer against closed forms (Airy pattern and EE, circular/annular/square MTF, defocus Strehl, Parseval, FFT vs Huygens, tilt direction, polychromatic sums, sampling warnings, determinism, cancellation, resampling) |
| `tests/traced.rs` | lattice OPD = `mklight_trace` OPD (1e-9) on triplet, Newtonian, finite relay and Petzval asphere; diffraction-limited ideal lens and paraboloid (Strehl, analytic MTF to 2e-3, Airy zero at 1.2197 λN), achromat at F/10 and singlet at F/20 (Strehl > 0.99, MTF within 0.01, scale = paraxial working F/# to 1 %); defocus: every ray = exact sphere-to-sphere closed form (1e-8 waves), Zernike focus = `δz NA²/(2λ)`, RMS = `W020/√12`; spherical: `6·Z9` = Seidel `W040` (3 %), fit = traced marginal OPD; Cooke triplet off axis: FFT vs Huygens (0.03) vs direct (0.02) MTF, tangential ≠ sagittal; 10-wave defocus: wave MTF → geometric MTF (0.05); vignetted triplet: rim area = 801² ray-grid transmission (2e-3), coverage = mask area, rim beats the hull; telecentric: `R = ∞`, OPD = `n δz (1 − cosθ)` (1e-8), refocus, FFT = plane-wave Huygens (1e-9), chief ray parallel off axis; mirrors: single paraboloid (`propagation_sign = −1`, defocus closed form), Newtonian, and the PSF centroid = ray centroid off axis (paraboloid, Newtonian, triplet; 3 %); radiometric amplitude of an ideal lens `A² = 1/cos³θ` (2e-3) and flat for a 1:1 finite relay; detector sinc zeros at `1/p`, fill factor, diffusion, product and Nyquist; registry JSON runs, round trips, determinism, cancellation (before and during a run, from another thread) and progress (sampled: monotone, with intermediate values); polychromatic = weighted sum of placed monochromatic PSFs; configurations and through-focus symmetry; Airy EE, `r50`, `r80` and LSF normalisation; MTF vs field = MTF at the system fields; `R → ∞` lattice OPD = `mklight_trace` OPD on and off axis; exit pupil beyond the image (`σ = +1`): every ray = far-sheet closed form (1e-8), positive defocus; annular stop: two rim loops, transmission `1 − ε²` (1e-3), mapped clear area (2e-3), analytic annular MTF (3e-3); PSF centroid = ray centroid for `−x` and diagonal fields (triplet, paraboloid); `transmission` amplitude biases the singlet's centroid by > 15 %, `radiometric` < 10 %; Huygens grid and cost limits (`InvalidParam`, no tracing); `psf_truncated` on a defocused PSF but not an Airy one; unusable fields skipped with warnings (`null` in MTF vs field, error when no field is usable); the axial MTF-vs-field sample follows the sweep direction (non-square pixels); configured runs record the base hash, configuration and solve warnings; default MTF right near zero frequency where the FFT interpolation is not |

## 7. Open points

* A curved image surface uses the vertex-frame plane through the reference
  point; a tangent-plane frame would need the surface normal at `P`.
* The rim polygon is inscribed between bisected boundary points (area
  deficit `≈ 2π²/(3N²)` for `N` vertices, 1.6e-4 at `rays = 65`).
* `mtf_vs_field` sweeps normalised field points without the system
  fields' vignetting factors (the traced rim still models real
  vignetting).
* Polarisation (vector pupils from `mklight_trace::vector_pupil`) and GPU
  pupil tracing are not wired in; the link is CPU f64.
* Sampling is not adaptive: the pupil grid (`pupil.grid`) and the PSF or
  Huygens window are fixed settings, and the analyses only warn
  (`pupil_undersampled`, `field_wraps`, `psf_truncated`). Choosing the
  pupil grid from the largest phase step and the window from the geometric
  spot extent would remove the need to tune them.
* The default amplitude model is `transmission` (§3); `radiometric` is
  more faithful for PSF energy placement but not yet the default.

## 8. Image simulation

`image_simulation` (`image_simulation` module, registered by the engine)
accepts built-in `grid`, `siemens_star`, `slanted_edge`, and `text_chart`
patterns, `source:{kind:"buffer",width,height,channels:1|3,data:[…]}`
(row-major interleaved, finite nonnegative linear-light samples), or
`source:{kind:"path",path:"…"}` (an 8- or 16-bit grey or RGB PNG, alpha
ignored, decoded from sRGB; other formats come as a buffer).
Output dimensions are `size:[width,height]`; `monochrome` selects one
channel, otherwise three. RGB source channels share the run's polychromatic
energy PSF; there is no implicit colour matching or separate channel response.
Monochrome conversion of RGB uses linear-light Rec. 709 luminance weights.
The shared engine `spectrum` setting is documented in ANALYSIS §4.

`optics` controls the square Cartesian PSF grid, normalised field extent,
traced pupil settings, odd `psf_size`, optional `psf_spacing_mm` and pupil
quadrature for illumination. At each field node:

1. Trace the first-order chief image (separate x/y first-order scales); the
   real image point is the PSF reference (`pupil.reference`: the primary
   chief ray by default), where the PSF centre belongs.
2. Build pupils at all wavelengths, including lateral-colour offsets relative
   to the primary chief, using radiometric amplitude by default.
3. Form the unit-energy polychromatic PSF on a common physical grid. Default
   spacing is half the smallest λF/# at the first grid node. PSF warnings
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
mean/variance.
