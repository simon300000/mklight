# Polarisation ray tracing and radiometry

This note covers the complex amplitude coefficients of coatings
(`mklight_core::materials::coating`), polarisation ray tracing
(`mklight_trace::polarization`) and radiometry along the sequential primary
path (`mklight_trace::radiometry`). The phase and basis conventions are
CONVENTIONS §9. Tests: `crates/mklight-trace/tests/polarization.rs` and the
`amplitude_tests` module of `coating.rs`.

## 1. Complex amplitude coefficients

`ResolvedCoating::amplitudes(λ, n_in, n_out, cos_i, mirror)` is the amplitude
counterpart of `ResolvedCoating::rt` with the same arguments. It returns
`InterfaceAmplitudes { rs, rp, ts, tp, flux_s, flux_p }`, and
`amplitudes(..).power()` equals `rt(..)` to rounding (tested to 1e-12 for
random angles, indices and stacks, absorbing and metal ones included, and
down to grazing incidence: where `n₀ cos θ₀` rounds to zero, about
`cos θ₀ < 1e-8`, `tmm_amplitudes` returns the grazing limit
`r_s = r_p = −1`, `t = 0` instead of the `0/0` of the p admittance
`N₀²/(N₀ cos θ₀)`, matching `fresnel_amplitudes` at `cos θ₀ = 0` and the
`R = 1` of `tmm`).

| coating | amplitudes |
|---|---|
| bare refracting | `fresnel_amplitudes`: closed-form Fresnel. The numerators use `n₁²c₁² − n₂²c₂² = (n₁ − n₂)(n₁ + n₂)` (and the p analogue), so cemented interfaces keep full relative accuracy. Beyond the critical angle the unit-modulus TIR phases of CONVENTIONS §9, with an evanescent `t` and `flux = 0`. |
| bare mirror | perfect conductor, `r_s = −1`, `r_p = +1` |
| `ThinFilm` | `tmm_amplitudes`: Macleod characteristic matrices (`N = n − ik`, tilted admittances `η_s = N cos θ`, `η_p = N / cos θ`). Tangential-field coefficients `ρ`, `τ` map to field coefficients as `r_s = ρ_s`, `r_p = −ρ_p`, `t_s = τ_s`, `t_p = τ_p cos θ₀ / cos θ_sub`. Same branch choices and degenerate cases as `tmm`. |
| `Ideal` | no phase data, so non-polarising: `r_s = −√R`, `r_p = +√R` (a scaled perfect conductor, zero retardance) and `t̃_s = t̃_p = +√T` |
| `Tabulated` | the bare-Fresnel phases, moduli rescaled to the tabulated powers; mirrors are scaled perfect conductors |

`flux` converts `|t|²` to power: `T = |t|²·flux`, with
`flux = n₂ cos θ₂ / (n₁ cos θ₁)` for non-absorbing media (the general
expressions are in the `InterfaceAmplitudes` docs). `ts_flux()` and
`tp_flux()` give the flux-normalised `t̃ = t √flux`, which is what the
polarisation tracer uses.

Checks: Brewster `r_p = 0` and `t_p = n₁/n₂`; TIR phases against
`tan(φ_s/2) = √(sin²θ − n²)/cos θ` and `tan(φ_p/2) = √(sin²θ − n²)/(n² cos θ)`
(including the 45° phase difference of a Fresnel rhomb); metal mirror
`r_s`, `r_p` against the closed-form complex Fresnel equations, and the
reflection retardance against the good-conductor expansion; normal-incidence
isotropy `r_p = −r_s`, `t_p = t_s` for any stack; reciprocity of `t̃`
(forward stack vs reversed stack at the Snell angle) and the Stokes relations
`|r|² + |t̃|² = 1`, `r' = −r* t̃/t̃*` for lossless stacks.

The polariser case of the task ("an ideal coating with `r_p = 0`") cannot be
expressed: `CoatingDef::Ideal` has one `R` and one `T` for both
polarisations, and adding a polarising variant would change a shared enum
that every solver matches on. The Malus-type checks use an ideal polariser
built directly as an interface PRT matrix (`t̃_s = 0`, `t̃_p = 1`) and a real
Brewster-angle plate instead (§6).

## 2. PRT matrices

For each interacting surface of a ray recorded by `trace_ray`, in global
coordinates (directions are rotated by the exact global rotation of each
surface frame, so no large coordinates are involved):

```
P_i = a_s · s' sᵀ + a_p · p' pᵀ + k' kᵀ
s = (k × n̂)⊥k / |…|,  p = k × s,  s' = s re-orthogonalised against k',  p' = k' × s'
(a_s, a_p) = (t̃_s, t̃_p) refracting   |   (r_s, r_p) reflecting
```

`P_i k = k'`, `|P_i s|² = T_s`, `|P_i p|² = T_p`, and `P_i s ⟂ P_i p`. The
cumulative matrix is `P = P_N ⋯ P_1`. Dummy surfaces, coordinate breaks and
the image surface are the identity. An ideal (`Paraxial`) lens is lossless
and non-polarising: the minimal rotation `k → k'`. A reflecting ideal lens
applies its coating's mirror amplitudes on the lens plane, then that rotation.

**Normal incidence.** `k × n̂` vanishes. Below `|k × n̂| = 1e-12`, `s_vector`
falls back to `k × e`, with `e` the coordinate axis least aligned with `k`
(deterministic). The choice does not matter. At normal incidence every
coating has `a_p = a_s` (transmission), or `r_p = −r_s` with `p' = −p`
(reflection). So `P_i = a (I − kkᵀ) + k'kᵀ` for any `s ⟂ k`. Near normal
incidence the anisotropic part (`a_s − a_p` or `r_s + r_p`) is `O(θ²)`, so
either an arbitrary `s` (fallback) or an `s` with rounding error `ε/θ`
(the cross product of nearly parallel vectors) changes `P_i` by at most
`O(θ²)` or `O(θ ε)`. `P_i` is continuous through normal incidence. The tests
check basis independence at normal incidence for transmission, an absorbing
multilayer and a metal mirror, and `‖P(θ) − P(0)‖ = O(θ)` from `θ = 1e-2`
down through the fallback threshold. They also check continuity of the
traced Jones pupil through the pupil centre.

**Geometric transformation.** `Q` is the same product for the ideal system:
`a_s = a_p = 1` at refracting surfaces and a perfect conductor at mirrors.
It is real orthogonal (determinant `(−1)^mirrors`). The **proper Jones
matrix** `[x y]ᵀ Qᵀ P [x y]` in the input basis removes the geometric
rotation of the s/p planes and the mirror parity. It is the identity for
non-polarising systems (perfect mirrors, ideal lenses). For one metal mirror
it is `diag(−r_s, r_p)` in `(s, p)`. Its diattenuation and retardance do not
depend on the pupil basis (a real rotation of the basis is a similarity
transform).

## 3. Jones pupils and the double-pole basis

`PolarizedRay::jones()` is `J_ij = out_i · P in_j` (flux-normalised, bulk
absorption excluded; `jones_amplitude()` includes `√τ`). The bases are
double-pole bases (CONVENTIONS §9):

* **input**: the starting frame's `(x̂, ŷ)` carried by the minimal rotation
  from `+ẑ` to the object-space ray direction. For an object at infinity all
  rays of a field share it.
* **output**: the image frame's `(x̂, ŷ)` carried from `+ẑ` to the
  image-space direction. When the light travels toward the image frame's `−z`
  (an odd number of mirrors without coordinate breaks), the pole is `−ẑ` and
  the reference axes are `(x̂, −ŷ)`, so that the triad stays right-handed.

With `(k₁, k₂, k₃)` the ray direction on the reference triad (`k₃ ≥ 0`):
`x̂' = (1 − k₁²/(1+k₃), −k₁k₂/(1+k₃), −k₁)`,
`ŷ' = (−k₁k₂/(1+k₃), 1 − k₂²/(1+k₃), −k₂)`. Nothing cancels, the basis is
smooth over the whole hemisphere, and it follows the fixed `x̂`, `ŷ`.
Rotating `k` by `φ` about the pole gives `B(R_z k) = R_z B(k) R(−φ)`. A
rotationally symmetric system therefore has
`J(r, φ) = R(φ) J(r, 0) R(−φ)` (tested to 1e-13), and an uncoated lens
transmits the radial (p) state best.

## 4. Diattenuation and retardance

`PolarizationProperties::from_jones`:

* `J†J = (t/2) I + m·σ` (Pauli matrices ordered to match `S₁, S₂, S₃`).
  Then `T_max = t/2 + |m|`, `T_min = |det J|²/T_max` (no cancellation),
  `D = 2|m|/t`, and the diattenuation axis is `m/|m|` (the Stokes vector of
  the best-transmitted state).
* Polar decomposition `J = U H`. For 2×2 matrices,
  `H = (J†J + |det J| I)/√(t + 2|det J|)` and `U = J adj(H)/|det J|`. Write
  `U = e^{iα}(cos(δ/2) I − i sin(δ/2) a·σ)`, with `e^{2iα} = det U` and the
  sign chosen so that `cos(δ/2) ≥ 0`. Then the retardance is
  `δ = 2 atan2(|sin(δ/2) a|, cos(δ/2)) ∈ [0, π]`, which stays accurate for
  small `δ`. The fast axis is `−a`.
* **Accuracy near a polariser.** The retarder of a nearly singular `J` is
  ill-conditioned in itself, not only through this algorithm: a relative
  perturbation `ε` of `J` changes `δ` by up to about `ε·√(T_max/T_min)`.
  With rounding alone (`ε ≈ 1e-16`) that is 1e-11 rad at `T_min/T_max = 1e-10`
  and 1e-4 rad at `1e-24` (tested against
  `R(θ) diag(1, ε e^{iδ}) R(−θ)`). Below `T_min/T_max = 1e-24`
  (`RETARDER_MIN_CONTRAST`) the retarder is undefined:
  `retarder_defined = false`, `δ = 0` and a zero fast axis as placeholders.
  The pupil map excludes such samples from its retardance statistics,
  counts them in `undefined_retarder` and adds a `RunInfo` warning.

`PolarizedRay::properties()` uses the proper Jones matrix. The pupil map
also reports the retardance of the pupil-basis Jones matrix
(`retardance_jones`), which includes the geometric rotation and the mirror
parity.

## 5. Radiometry

Per-ray throughput is `T = T_interfaces · exp(−Σ αL)`. The internal
transmittance comes from the reference trace (Beer–Lambert over the real
path in every medium). `RadiometryModel` selects `T_interfaces`:

* `Unpolarized`: `Π ½(X_s + X_p)`, with `X = T` at refracting surfaces,
  `R` at mirrors and 1 at ideal lenses. This is the scalar model of the GPU
  `Radiometry::Unpolarized` mode. It re-depolarises the light after every
  interface. It needs only `ResolvedCoating::rt`, so
  `SequentialTracer::throughput` evaluates it straight from the surface
  records (`unpolarized_path_product`, `RayThroughput::unpolarized`), with
  no amplitude, PRT or basis computation. It shares the per-surface
  classification and power call with `interface_steps`, and equals the
  PRT-derived value bit for bit (tested).
* `Polarized { state }`: PRT, `|J e|²`, or `½ S₀ tr(J†J) + m·S` for a
  Stokes state, divided by `S₀`. `PolarizationState::Unpolarized` gives the
  exact unpolarised throughput `½ tr(J†J)` (the average over any two
  orthogonal states, tested). `PolarizedRay::intensity(state)` is the same
  fraction of the input power `S₀` (times `τ_internal`).

The jobs validate the input state (`PolarizationState::validate`): it must be
finite, with `S₀ > 0` and degree of polarisation `≤ 1` (to 1e-12), or they
return `MkError::InvalidModel`.

For a meridional ray of a coaxial lens, the exact value is
`½(Π T_s + Π T_p)`. For two surfaces it differs from the incoherent model by
exactly `¼(T_s1 − T_p1)(T_s2 − T_p2)` (tested): second order in the
diattenuations, about 1e-9 for an uncoated F/10 singlet.

**Energy budget.** `PolarizedRay::energy_budget(state)` propagates the 3×3
coherency matrix `Φ → τ_seg P_i Φ P_i†`. At each interface, `path` (the power
continuing) is the trace of the propagated `Φ`, so it comes from the PRT
amplitudes. The incident s and p powers `sᵀΦs` and `pᵀΦp` are reported, and
the shared power coefficients split them into `diverted` (specular
reflection at a refracting surface, the ghost solver's share,
`R_s w_s + R_p w_p`) and `absorbed` (coating absorption, plus transmission
through a mirror, `A_s w_s + A_p w_p` with `A = 1 − R − T`). None of the
terms is defined as a remainder. The residual of
`input = transmitted + Σ diverted + Σ absorbed + bulk` is therefore zero
only if the field stays transverse, the PRT amplitudes reproduce `rt`, and
the bulk factors chain correctly.

**System transmission.** `system_transmission(tracer, settings, ctx)`
averages per-ray throughput over pupil samples for each (field, wavelength),
with equal weight per sample (`PupilSamples::Grid { n }` inside the unit
disk, `n = 1` or `n ≥ 3`, or `Random { count, seed }`: Philox stream `(seed, 0)`, `r = √u₁`,
`φ = 2πu₂`, the same points for every field and wavelength). A sampling
with no points (`Grid` with `n = 0` or `n = 2`, whose four nodes lie outside
the disk, or `count = 0`) is rejected with `MkError::InvalidModel`. Otherwise
it would report zero transmission, which looks like an opaque system. Blocked rays
(clipped, missed, TIR) count 0. Numerically failed rays are excluded and
listed in `RunInfo::warnings`. Each point reports `transmission`,
`mean_unvignetted` and `unvignetted_fraction`, and each field gets a
spectral-weight average. Projection and pupil-distortion effects (relative
illumination) are left to the geometric analyses. Results are deterministic:
fixed sample order and compensated sums.

**Polarisation pupil map.** `polarization_pupil_map` returns, on an `n × n`
pupil grid of one field and wavelength: the amplitude Jones matrix, the
intensity for an input state, the unpolarised transmittance, the
diattenuation and proper retardance with their axes, and the pupil-basis
retardance. It also gives the pupil mean intensity, maximum diattenuation,
and maximum and RMS retardance.

Both jobs take a `JobContext` (cancellation, monotone progress) and return a
`RunInfo` (backend `cpu-f64`, `F64Reference`, seed for random sampling).

## 6. Tests (energy and known answers)

| test | checks |
|---|---|
| `energy_is_conserved_along_real_rays` | coated Cooke triplet (lossless and absorbing thin films, ideal and tabulated coatings), Double-Gauss, an absorbing-glass singlet and the aluminised Newtonian, at all fields, wavelengths and hexapolar pupil points: per-interface `T + R + A = 1` per polarisation, step powers equal `ResolvedCoating::rt` and the amplitudes (NaN-propagating comparisons), lossless films have `A = 0`. For unpolarised, elliptical and partially polarised input, at every surface: `w_s + w_p` equals the incident power (transverse field), the PRT path power plus the `rt`-based diverted and absorbed powers equals the incident power to 1e-12, and the incident power is the previous path power times the bulk factor. The whole budget closes to 1e-12 |
| `lossless_plate_closes_with_all_reflected_orders` | tilted AR-coated plate: the primary path is `T₁T₂` per polarisation; the front stack seen from the glass side (reversed layers) is reciprocal, `T₁' = T₁` and `R₁' = R₁`; an explicit order-by-order sum of the multiply-reflected orders equals the closed form `T₁T₂/(1 − R₂R₁')` and gives `T_tot + R_tot = 1` |
| `uncoated_singlet_normal_incidence_is_one_minus_r_squared` | `(1 − R)²·τ` for every model and state; isotropic Jones matrix; pupil average of an F/10 singlet within 2e-4 below `(1 − R)²` |
| `quarter_wave_ar_reduces_losses_as_predicted` | `R = ((n_s − n₁²)/(n_s + n₁²))²` per face, budget `R`, `R(1 − R)`; predicted loss ratio |
| `metal_fold_mirror_retardance` | 45° aluminium fold: proper retardance `|arg(r_p/(−r_s))|`, diattenuation and fast axis from the closed-form complex Fresnel equations; a bare mirror is non-polarising |
| `brewster_plate_malus_type_law`, `malus_law_with_ideal_polarizer_interface` | `I(θ) = T_p cos²θ + T_s sin²θ` with `T_p = 1` at Brewster's angle; `cos²θ` for an ideal polariser |
| `jones_pupil_is_rotationally_covariant` | `J(r, φ) = R(φ)J(r, 0)R(−φ)`, radial diattenuation axis, continuity through the pupil centre |
| `normal_incidence_basis_is_irrelevant_and_continuous` | §2 |
| `jones_properties_of_known_elements` | random homogeneous elements, quarter-wave plate, rotator, small retardance, perfect polariser |
| `near_polarizer_retardance_accuracy_and_undefined_retarder` | the §4 error bound for `T_min/T_max` down to 1e-22; the undefined-retarder flag beyond it |
| `vignetted_vector_pupil_samples_carry_no_light` | clipped rays give zero Jones matrices, fields and direction, consistent with the pupil map |
| `intensity_is_normalized_and_states_are_validated` | `intensity` is independent of `S₀` and equals `RayThroughput`; unphysical states are rejected |
| `empty_pupil_samplings_are_rejected`, `unpolarized_fast_path_equals_prt_path` | §5 |
| `amplitude_tests` (core) | also grazing incidence `cos θ ∈ [0, 1e-7]` for every coating kind: finite amplitudes whose powers equal `rt` |

## 7. Interface for polarised wave optics

The wave-optics crate builds a vector (Jones) pupil from this API, with no
changes here:

```rust
use mklight_trace::{vector_pupil, VectorPupilSample, JonesVector};

// One sample per pupil point, in input order; cancellable.
let samples: Vec<VectorPupilSample> =
    vector_pupil(&tracer, &field, wl, &pupil_points, &opts, &ctx)?;
for s in &samples {
    if !s.status.is_ok() { continue; }      // obscured/vignetted: all zero
    let e_img = s.field(&JonesVector::linear(0.0)); // image-local complex 3-vector
    let jones = s.jones;                     // 2×2, double-pole bases, includes √τ
    let k = s.dir;                           // image-local unit direction
}
```

* `field_x` and `field_y` are the image-local 3-D field vectors for unit
  input along the object-space double-pole `x̂` and `ŷ`. They include coating
  and Fresnel amplitudes and phases (flux-normalised, so `|E|²` is the power
  per ray) and `√τ_internal`, and they are transverse (`k·E = 0`). For
  Richards–Wolf style vector diffraction, the pupil function of input state
  `e` is `E(P) = s.field(&e) · exp(+i2π W(P))`, with `W` the OPD in waves
  from `opd`/`opd_grid` at the same pupil points. The `+` sign follows from
  CONVENTIONS §9: the field phase is `exp(−i2π OPL/λ)` and
  `OPD = OPL_chief − OPL_ray`. Each Cartesian component is propagated like a
  scalar pupil, and the intensity PSF is `Σ_j |E_j|²`.
* For an unpolarised source, average the PSFs of two orthogonal inputs (for
  example `x̂` and `ŷ`). For partially polarised light, mix the coherency
  terms with the source Stokes vector.
* `jones` (2×2) is enough for low-NA scalar-per-component work and for Jones
  pupil displays. It maps object-space double-pole components to image-space
  double-pole components.
* A sample whose ray did not reach the image (clipped, missed, TIR, failed)
  carries no light: `jones`, `field_x` and `field_y` are zero and `dir` is
  the zero vector. Skipping these samples and adding them as zeros give the
  same result.
* Uniform sampling of `pupil_points` is the caller's choice. Ray aiming and
  vignetting follow the tracer's settings, exactly as for OPD.

## 8. Limitations

* Multiple-beam interference between ghost orders is not part of the
  primary path. The plate closure sums orders incoherently, which is what
  the ghost solver does.
* Media are isotropic. There is no birefringence, stress birefringence or
  optical activity.
* `Ideal` and `Tabulated` coatings carry no measured phase (§1), and ideal
  lenses are lossless and non-polarising.
* A bare `Reflect` surface is a perfect conductor (`r_s = −1`, `r_p = +1`)
  at every angle. Prism faces modelled as mirrors (TIR faces, roof prisms,
  Fresnel rhombs) therefore get **no TIR retardance**, although physical
  total internal reflection adds the s/p phases of CONVENTIONS §9, which
  `fresnel_amplitudes` already computes. The mirror surface does not know the
  medium outside the glass: its `n_after` equals `n_before`. Roof-prism phase
  errors, a common reason for a polarisation analysis, are therefore not
  modelled yet. A later additive option (a coating kind or surface flag that
  names the outer medium) could apply `fresnel_amplitudes(n_glass, n_outer, cos θ)`
  at such faces. Meanwhile, a `ThinFilm` mirror coating carries an
  angle-dependent phase.
* The legacy GPU trace mode is scalar `Unpolarized`. The resident P7
  polarisation service is described in §9; engine routing is a separate step.

## 9. Resident GPU service (P7)

`mklight_gpu::radiometry` implements backend-neutral resident requests with
matching CPU-f32 code and strict Metal kernels. Coating and transport
results are bit-identical between the two in a matching floating-point
environment, except the retardance angles, which leave double-float through
an f32 `atan2`. This is a service for the
GPU solver packages; engine capability and `auto` routing are in GPU.md
§9.2. No reference tracer or solver weighting rule is changed.

### Coatings and immutable material state

`CoatingSpec`, `CoatingLayer`, `CoatingQuery` and `CoatingValue` have the
layouts in `include/mklight_radiometry_types.h`. Use `record_desc::<T>` for
U32 record buffers. The immutable domains contain resolved wavelength,
incident/exit index, incidence side, coating definition and layer indices;
`CoatingSpec::resolve` preserves the reference stack reversal rule. Small
material resolution and spectral table interpolation use the existing host
material service. All angle-dependent amplitudes, powers and thin-film
matrix products run on the device. `PreparedCoatings` retains shared-memory
leases for reuse; constructing changed domains produces new leases. Its
`identity` is a digest of the actual spec and layer tables, computed once at
construction and recorded as the plan's library identity, so changed
material or wavelength data cannot silently reuse an old identity.

The service evaluates angles directly, with **no angle interpolation**.
Complex arithmetic, Brewster numerators, Snell discriminants, matrix
products and phase reduction use double-float. Sin/cos and exponential
polynomials operate on reduced arguments. The s and p matrices share the
same layer propagation phase. `amplitudes` stores `rs,rp,ts_flux,tp_flux`
as real/imaginary double-float pairs; `powers` stores `Rs,Rp,Ts,Tp,As,Ap`.
Film absorption is independently computed from Poynting-flux differences
between layer faces. Power entering an opaque mirror substrate is included
in absorption. It is not defined as the conservation residual.

A raw `CoatingRequest` reports `MKL_RAD_REFERENCE`, with a counted fallback,
for nonzero thin-film grazing cosines below 1e-4, a Snell discriminant
within 1e-8 of critical incidence, unrepresentable results or layer phases
outside the evaluated range (`|Im δ| > 30` or `|Re δ| > 1e6`). These are
precision envelopes, not changed optical thresholds. Exact grazing retains
the reference limit. `PreparedCoatings::evaluate` completes flagged queries
with the original f64 coating, records the count in `PlanStats` and adds a
`RunInfo` warning. Cancellation does not execute this replacement path.
Stacks beyond 32 layers return unsupported for the caller's reference route.

### Primary transport, maps and reductions

`PreparedRadiometry::append` consumes P1 `ResidentHits` with paired hi/lo
records, primary slots `0..image` in order and radiometry off. Metadata is
surface-major/wavelength-minor; the public `PolarizationRequest` can also
consume plan-local `SurfaceSpec` records from a candidate producer. Its
material state must match the P1 records. The supplied immutable preparation
helper prepares one resolved system; it does not rebuild P4 candidate
material tables on the host after each trace.

Directions/normals use P1's records. Segment lengths use relative local
frame transfers and double-float differences, then Beer–Lambert absorption.
Transport retains the double-pole bases, ideal-lens rotations, mirror
parity and geometric-rotation removal of §§2–4. `PolarizationValue` contains:

* pupil and proper Jones matrices, including bulk attenuation;
* the two image-local transverse vector-pupil columns and output direction;
* scalar product of s/p averages, average of separate s/p products, selected
  Stokes-state intensity and exact unpolarised intensity;
* separately accumulated diverted, coating-absorbed and bulk-absorbed power;
* diattenuation/axis, proper and pupil-basis retardance, fast axis and
  defined-retarder flags.

The separate-s/p product is a diagnostic for the ghost approximation; it is
not substituted for PRT on non-coplanar paths. As with `vector_pupil` and
`PolarizedRay`, these raw pupil channels exclude DOE efficiency. The
`system_transmission` adapter multiplies by the reference selected-order
factor, as the CPU transmission job does. A wave or pupil-map adapter must
apply its existing DOE factor once.

`PupilChannels` optionally publishes `Complex32 [4,N]` Jones or `[6,N]`
vector channels for resident wave consumers, plus an explicit status mask.
Blocked or reference-required channels are zero; consumers must handle the
reference mask before publishing a result. `append_summary` extracts
channel-major double-float data and uses P3's fixed segmented tree. The ten
sum/count/min/max channels are scalar intensity, selected-state intensity,
exact unpolarised intensity, diattenuation, proper retardance squared,
usable count, surviving count, defined-retarder count, reference-required
count and numerical-failure count. Physical clipping contributes zero;
numerically failed rays are excluded from the transmission denominator.

`pupil_plan` builds P1 generation → recorded trace → P7 transport → P3
summary with no intermediate optical readback. `pupil` is its cancellable
host result adapter; `system_transmission` and `polarization_pupil_map`
expose the existing result types for all requested fields/wavelengths.
Reference-required pupil rows and unresolved aims are evaluated with the f64
tracer and counted in metadata. Only when such rows change is the final
summary rebuilt in f64, in original input order, with a warning. The
physical undefined-retarder contrast remains 1e-24; the device requests f64
below a 1e-18 conditioning envelope, including the final mask decision.
Jones matrices are scaled before polar decomposition to prevent determinant
underflow for nearly opaque pupils. An exactly opaque matrix is completed by
the reference mask decision without division by zero. A P1 failure retains
its original status alongside the zero pupil.

Immutable coating and surface tables are produced directly inside
`fill_buffer`; outputs use `read_view`, with no bulk staging copies.
`ResultSlice` keeps completed coating/pupil rows and statuses in place; only
reference completion creates a mutable working copy (reported as
`reference_read_copy_bytes`). Borrowed host coating-query arrays require one
input copy, validated while copying and recorded as `input_copy_bytes`;
direct producers can use `fill_buffer` and the raw request. The existing P0
system-cache API copies cold packed geometry once and reuses it on warm
jobs; benchmarks report this separately from resident-plan copies. Scalar
transmission has a dedicated transport kernel that skips PRT and
retardance, and reads only the P3 summary when no reference completion is
needed. Numeric outputs are final result assembly, not f64-labelled GPU
arithmetic. Per-ray operation order is independent of tile/command sizes;
Metal uses adaptive commands aiming at 10 ms (at most `MAX_TILE` rows) and
checks cancellation before and after each one. Only integer fallback
counters return to the host between resident stages. All jobs carry
`RunInfo` with plan/build, material, sampling, backend and fallback
metadata.

`system_transmission` divides its (field, wavelength) rows between the CPU
and the GPU at the aim: the CPU aims every ray (the reference's f64 packet
aiming, all cores) and writes the launch rays straight into shared ray
planes; the GPU traces them and evaluates the P7 transport and the per-ray
summary channels; the host sums each row in f64 from the shared buffer and
completes the rays the device could not decide with the reference. The
rows run in blocks of at least 4096 rays (about four): the GPU traces block
k while the CPU aims block k+1. Unpolarised rows use the scalar transport
kernel; the sums, denominators and failure counts are the reference's.
`pupil` traces its reference-completed rays in parallel before merging them
in pupil order (unchanged results). `polarization_pupil_map` stays on the
reference by static capability (undefined polarisation axes).

Tests are `tests/p7_radiometry.rs` in both GPU crates; host Metal tests require
a device and exercise both precompiled and runtime-source libraries. The
ignored `p7_bench` harness records cold setup and at least ten warm samples
for all built-ins, dense grids and 0–32-layer stacks. These paths do not by
themselves enable production engine routing or qualify arbitrary P4 candidate
parameter envelopes.
