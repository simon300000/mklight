# MKLight Physical and Numerical Conventions

Every crate, kernel, binding and UI follows these conventions. When code and
this document disagree, the code is wrong. Fix the code, or change this
document on purpose in the same change.

## 1. Units

| quantity | unit |
|---|---|
| length (radii, thicknesses, sags, apertures, ray coordinates) | millimetre (mm) |
| wavelength | micrometre (µm) |
| OPD / wavefront | waves at the ray's own wavelength (reported), mm internally |
| angles in files, UI and settings | degrees |
| angles inside math code | radians |
| spatial frequency | cycles/mm |
| temperature / pressure | °C / atm |
| flux / irradiance | W / W·mm⁻² (relative units allowed when only ratios matter) |

## 2. Coordinates and signs

* Right-handed coordinates. The optical axis is **z**. Light leaves the object
  travelling toward **+z**. The meridional (tangential) plane is **y–z** and the
  sagittal direction is **x**.
* Surfaces are indexed `0..N-1`. Surface `0` is the **object**, `N-1` is the
  **image**. `system.stop` names the aperture-stop surface.
* Each surface has a **local frame**. The vertex is at the origin and the sag is
  `z = f(x, y)`. The surface is defined by the frame plus its shape.
* **Apertures** clip rays at the local `(x, y)` of the hit: a fixed
  semi-diameter (a circle) or an explicit circular, annular, rectangular or
  elliptical aperture, which wins over the semi-diameter. The edge passes,
  to a relative tolerance of 10⁻¹⁰ (`APERTURE_EDGE_TOLERANCE`, the default
  real-aiming accuracy): outer sizes are grown by that factor and an
  annulus's inner radius shrunk by it, so a ray aimed at the rim of the stop
  is not clipped by its own aiming residual. Every CPU and GPU path tests
  these same sizes (`ApertureShape::passes`; the GPU packer packs them).
  The same fixed outline also decides which of two crossings of a deep
  surface a steep ray meets (SURFACES.md §8, conic root choice), clipped or
  not; an automatic semi-diameter does not.
* **Curvature** `c = 1/R` is stored instead of radius (a plane is `c = 0`).
  `R > 0` means the centre of curvature lies at **+z** from the vertex.
* **Thickness** `t_i` is the signed distance along the local z axis, after
  surface `i`'s post-transform, from vertex `i` to vertex `i+1`. After an odd
  number of mirrors, light travels toward −z and physical gaps are negative.
* `t_0 = +∞` means the object is at infinity. In JSON, `±∞` is written as the
  strings `"inf"` and `"-inf"`.
* **Refractive indices are always positive.** Propagation direction lives only
  in the unit direction vector. (The paraxial module internally uses signed
  "reduced" indices, `n·sign(dz)`.)
* Indices are **relative to air** at the system temperature and pressure (the
  same convention glass catalogs use at their 20 °C / 1 atm reference). Vacuum
  then has index `1/n_air(λ,T,P)`.
* **Tilts and decenters** (`Placement`): decentres `(dx, dy)` in mm, tilts
  `tilt_x`, `tilt_y`, `tilt_z` in degrees by the right-hand rule, each about
  the axis of the frame as moved so far. `order = DecenterThenTilt` (the
  default; `.zmx` coordinate-break order flag 0): decentre, then tilt about x, then y,
  then z — frame rotation `R = R_x R_y R_z`, origin `(dx, dy, 0)`.
  `order = TiltThenDecenter` (order flag ≠ 0): the reverse sequence — tilt
  about z, then y, then x, then decentre along the tilted axes —
  `R = R_z R_y R_x`, origin `R (dx, dy, 0)`. A placement followed by the same
  placement negated in the other order is exactly the identity (the usual
  return coordinate break). A `CoordinateBreak`
  surface only changes the frame. Any surface's placement can be `Restore`d after
  the surface. Rotations act on the frame, so all following surfaces move too.
* **Rigid ranges.** `element_placement` on a surface moves the inclusive
  range up to its `last` surface rigidly (decentres mm, tilts degrees), in
  that surface's axes, about its first vertex or a `pivot` vertex. Surfaces
  after the range keep their frames. A decentre or tilt `ParamPath` on a
  surface without a placement creates one with `restore_after`.
* **Field angle** `θy > 0` means the chief ray's direction has a positive y
  component in object space. Field heights are measured on the object (or
  image) surface, in mm along +y/+x.

## 3. Normalised coordinates

* Field `(Hx, Hy)`: the field point divided by the **maximum radial field**
  value. `(0, 1)` is the edge of the field in +y.
* Pupil `(Px, Py)`: coordinates on the unit disk of the **entrance pupil**. With
  ray aiming off, they map linearly onto the paraxial entrance pupil. With real
  ray aiming, they map linearly onto the stop surface's clear aperture, and the
  aiming solve finds the object-space ray that lands there.
* The **stop radius** of that clear aperture (`PupilGeometry::r_stop`) is the
  stop's own size when the aperture floats by the stop size. For every other
  aperture specification it depends on the system's ray aiming: with aiming
  off it is the paraxial marginal-ray height at the stop, and with real aiming
  the stop height of the **real** on-axis marginal ray: the mean over the four
  rays through `(±1, 0)` and `(0, ±1)` of the paraxial entrance pupil, at the
  primary wavelength, unclipped. The on-axis beam is therefore the one the
  aperture specifies with aiming on and off. If one of those rays does not
  reach the stop, the paraxial height is used and the tracer warns. The stop
  radius is the system's: an analysis that aims real rays through a system
  whose aiming is off (relative illumination) keeps the paraxial height.
  `FirstOrder::stop_semi_diameter` is always the paraxial height.
* The **chief ray** is `(Px, Py) = (0, 0)`. The **marginal ray** is
  `(0, 1)` at the on-axis field.
* Vignetting factors `(vdx, vdy, vcx, vcy)` use the usual decenter/compression
  form: `Py' = vdy + (1 − vcy)·Py`, and likewise for x.

## 4. Aberration definitions

* **Transverse ray aberration**: `εx = x_ray − x_chief` and
  `εy = y_ray − y_chief` on the image surface, in mm.
* **OPD**: `OPD(P) = OPL_chief − OPL_ray`, where each OPL runs from the object
  (or a plane wavefront normal to the chief ray for objects at infinity) to the
  **reference sphere**. That sphere is centred on the chief ray's intersection
  with the image surface and passes through the chief ray's exit-pupil point.
  Reported in waves at the ray's wavelength, with the chief ray's OPD = 0 by
  construction. With this sign, a wavefront that converges *more* than the
  reference sphere has positive OPD at the pupil edge.
* **Zernike**: Fringe (University of Arizona) ordering, unit-disk normalised,
  coefficients in waves. A Noll-normalised standard ordering is also available.
* **MTF**: normalised to 1 at zero frequency. Tangential means frequency along
  the meridional (y) direction of the image surface. Sagittal means along x.
  Off-axis fields use the local field orientation.

## 5. Interactions, polarisation and radiometry

* Every surface interaction splits flux into transmitted `T`, reflected `R` and
  absorbed `A`, with `T + R + A = 1` for the same `(λ, θ, polarisation)` and
  coating. Bare interfaces use the Fresnel equations. Coated ones use the
  thin-film characteristic matrix (TMM).
* s and p are defined relative to the local plane of incidence. "Unpolarised"
  means the incoherent average of s and p.
* Absorption inside glass uses the catalog's internal transmittance, scaled
  with Beer–Lambert to the actual path length.
* Each solver owns disjoint path classes (`docs/ARCHITECTURE.md`), so no flux is
  counted twice.

## 6. Randomness and reproducibility

* All stochastic methods (Monte-Carlo stray light, random pupil sampling,
  tolerancing, global optimisers) use the counter-based **Philox4x32-10**
  generator, keyed by `(seed, stream, index)`. Draw `index` is lane
  `index % 4` of the block with counter `(index / 4, stream)` and key `seed`.
  CPU and GPU produce identical streams of **raw 32-bit words**, and identical
  **f32 uniforms** through `rng::u32_to_unit_f32`. The f64 uniform
  `rng::u32_to_unit_f64` is the CPU reference mapping. It differs from the f32
  one by half an f32 ulp in the upper half of the range, so a cross-validation
  that must sample identical points uses the same mapping on both sides
  (the f32 one, or the f64 one applied on the host to the GPU's raw words).
* Settings structs are serialisable. Each result carries a `RunInfo` with the
  engine version, snapshot hash, settings, backend, precision and seed.
* The same seed gives bit-identical results on the same backend. Backend
  `auto` keeps an identical request on its first route within one engine
  (`docs/GPU.md` §9.2), so a repeat is bit-identical there too; between
  engines or hosts `auto` may choose the other backend, and the results then
  agree within the accuracy contracts. An explicit backend is bit-reproducible
  across sessions on the same device and build.

## 7. Precision and error budget

The **CPU f64 tracer (`mklight-trace`) is the reference**. GPU backends compute
in f32, with double-float (f32 hi + lo pair) arithmetic where the budget
needs more (`docs/GPU.md` §3–4), using the techniques of §7.1–7.6.

### 7.1 Local coordinates

Trace in surface-local coordinates. Translate by the (exactly representable)
per-surface thickness, never by global positions.

### 7.2 Cancellation-free intersections

Intersect with cancellation-free closed forms. For a conic, solve
`t = B / (A·… )` in the "stable root" form with `B = c·(…)`, choosing the
root that avoids subtracting nearly equal numbers. Asphere departures are
then found by Newton iteration to a tolerance on the sag residual (the
tolerance and the iteration cap are in GPU §4.1).

### 7.3 Stable vector Snell law

Use the stable vector Snell law
`k' = μ·k + (√(1 − μ²(1 − (k·n)²)) − μ·(k·n))·n`, computed with
`fma` and `1 − x² = (1 − x)(1 + x)` where that helps.

### 7.4 Double-float optical path

Accumulate optical path length in **double-float** (`float2` hi/lo via
two-sum / fast-two-sum). OPD is formed as a difference against the chief
ray in double-float before rounding.

### 7.5 Compensated reductions

Sum reductions (spot centroids, RMS, energy) with Kahan/Neumaier
compensation, or as a pairwise tree.

### 7.6 Factored forms and exact comparisons

Prefer `x·x − y·y` in factored form, `hypot`, and `fma`. Never test floats
for exact equality, except against the sentinel values listed in the kernel
header.

### 7.7 Error budget, CPU packets and aiming

Budget for GPU against the f64 reference, for systems up to 1000 mm of total
track and 30 surfaces (the cross-validation tests enforce these limits and the
designs they cover):

| quantity | tolerance |
|---|---|
| ray position on any surface | ≤ 2·10⁻⁵ mm |
| direction cosines | ≤ 2·10⁻⁶ |
| OPD | ≤ 5·10⁻³ waves (≤ λ/200) |
| RMS spot radius | ≤ 0.5 % relative or 2·10⁻⁵ mm |
| MTF | ≤ 2·10⁻³ absolute |
| ray survival (vignetting) | identical except rays within 1·10⁻⁵ mm of an aperture edge |

A backend must never return silently degraded numbers. Accuracy is
qualified offline, per analysis, under the per-quantity contracts of GPU
§9.4 (each class at least as strict as this table). An analysis whose
native output fails them has no GPU path or is an `auto` accuracy
exclusion (GPU §9.2) and runs on the CPU reference, and a native run that
fails is completed by the reference and recorded in
`run_info.failed_attempts` (RESULTS §5).

**CPU packets.** Batches on the CPU (`trace_grid`, `opd_grid`, wave pupils,
rim bisection, merit groups, semi-diameters) aim and trace rays in packets
of 8 (`trace_rays`, `SequentialTracer::generate_many` / `trace_many`): the
direct aims of many rays advance together through a resumable copy of the
Newton–Broyden solver, and planes and conics with axial transfers run
through a structure-of-arrays lane kernel that repeats the scalar
operations (same `fma`, division and square-root order) with selects in
place of branches. Any lane that would leave the regular path (miss,
tangent root, clipping, TIR, non-finite values, other shapes) is redone by
the scalar per-surface step. The packet results are therefore **bitwise
identical** to `trace_ray` / `generate` / `trace` one ray at a time, which
stay the independent oracle (`crates/mklight-trace/tests/packets.rs`).

*What is vectorised* (measured 2026-11-13, `mklight-bench` example
`packet_bench`; the numbers below are for one thread). The lane kernel is plain
Rust that LLVM vectorises on its own. On aarch64, `trace_packets` has about 700 NEON `f64x2`
instructions (`fmul.2d`, `fmla.2d`, `fdiv.2d`, `fsqrt.2d`, …). Apple's
`objdump` prints them as `fmul.2d v0, v1, v2`, so grep for `\.2d`. Turning
the vectoriser off (`-C no-vectorize-slp -C no-vectorize-loops`) makes
`trace_rays` 1.28× slower and `trace_grid` / `opd_grid` about 1.2–1.25×
slower on the Cooke triplet and the double Gauss, and about 1.1× slower on
a 35 mm asphere lens (its aspheres take the scalar step). An explicit
whole-packet SIMD type (element-wise `[f64; 8]` operations, `mul_add` kept
as a per-lane `fma` so the results stayed bitwise identical) was tried and
not kept. It was 13–17 % slower on aarch64 because eight-lane values
spill, and 1.9× slower when walked in register-width chunks. On x86_64 and
wasm32 it vectorised no better than the plain loops. On x86_64, LLVM
leaves the kernel scalar even with AVX2 + FMA. Without `+fma` (the default
x86_64 target) every `mul_add` is a call to `fma` (113 calls in
`trace_packets`). On wasm32, with or without `simd128`, every `mul_add` is
a call to a software `fma` (117 calls), and the rest of the kernel stays
scalar. simd128 has no fused `f64x2` multiply–add. `relaxed_madd` may or
may not fuse, so it would break bitwise identity. Native packed f64 on
x86_64 would need intrinsics (for example the `wide` crate) and an
x86-64-v3 build. `wide`'s `mul_add` is unfused without `fma`, so it is
not bitwise identical on baseline x86_64 or wasm32. That is a
per-architecture path, worth it only if x86 throughput matters.
Building x86_64 releases with `-C target-cpu=x86-64-v3` (inline FMA) is
the cheaper first step.

**Aiming Jacobians.** The real-aim solve (`SequentialTracer`, 2-D Newton with
Broyden updates, `AIM_POLICY = JacobianPolicy::AnalyticRefresh`) takes every *fresh* Jacobian `∂(stop hit)/∂(aim point)`
from the analytic differential tracer (`mklight_trace::diff`): the tangent
of the starting ray is carried through the stop trace along its records,
so a fresh Jacobian costs one recorded trace instead of a four-trace
central-difference stencil (chief-ray aims, continuation stages, and the
refreshes after singular, slow or rejected steps). Where the tangent is
undefined (grazing incidence, near total internal reflection, an undefined
sag Hessian) the solver falls back to the central differences and
`AimStats` counts it. Between fresh Jacobians the solve uses Broyden
updates, so a seeded direct aim that needs no refresh traces exactly as
with finite-difference Jacobians; otherwise results differ from them only
within the aiming tolerance. (Analytic Jacobians at every iterate, `AnalyticNewton`, need
fewer iterations but cost more per iteration and are not used.)
The same tangent gives the real-image-height field solve its Jacobian
through the aim solve (implicit function theorem) and the differential
pencil of the field-curvature cross-check (ANALYSIS §2.2). The device
kernels use finite-difference stencils (GPU.md §9.2, "Aiming and OPD").

**Aim stop.** A real-aim solve is converged when its stop residual
`|stop_hit − target|` is at most the stop tolerance. Two policies exist
(`TracerSettings::aim_stop`, engine `AnalysisRequest::aim_stop`, recorded as
`run_info.aim_stop`):

* `requested`: the tolerance is the system's `ray_aiming.tolerance` (a
  fraction of the stop radius; the model default 1e-10 is about 2 pm on a
  20 mm stop) for every ray. This is the library default
  (`TracerSettings`, so the solver evaluators) and the strict reproduction
  mode.
* `conditioned` (the **engine default**, `EngineConfig::aim_stop`, for
  requests that name no policy): the tolerance is derived per field and
  wavelength from the
  **propagated error** of a stop residual. For the chief ray and four marginal
  probe rays (the chief Jacobian's prediction of the pupil edge, unaimed) the
  analytic tangent with respect to the aim point is carried through an
  unclipped image trace; with `J_stop` the stop-hit Jacobian and `T_k` the
  position or direction tangent at surface `k`, `‖T_k J_stop⁻¹‖₂` is the
  error at surface `k` per unit stop residual, and the OPD sensitivity comes
  from the tangent at the image through the field's reference sphere
  (`OpdReference::opd_tangent`, the eikonal equation on the sphere
  intersection). The derived stop is the largest residual whose propagated
  error stays within the **allocation** of every budget above, divided by a
  safety margin of 2 on the sampled sensitivities, then clamped to
  `[requested tolerance, cap]`: it is never tighter than the request (an
  ill-conditioned field keeps the request) and never looser than the cap.
  Chief rays are always aimed at the requested tolerance.

  | tier | allocation per budget | cap |
  |---|---|---|
  | design | 1/10 of 1/3 of the table above: position 6.7·10⁻⁷ mm, direction 6.7·10⁻⁸, OPD 1.7·10⁻⁴ waves | 10⁻⁶ of the stop radius |
  | preview | 1/10 of the preview budgets (GPU §9.4): position 2·10⁻⁵ mm, direction 2·10⁻⁶, OPD 2·10⁻³ waves | 10⁻⁵ of the stop radius |

  On a well-conditioned design the derived stop lies well above the
  model default; a field with a surface near
  grazing incidence derives a much tighter one. The device aiming kernel
  takes one tolerance per request: the tightest derived stop of the
  request's fields (`SequentialTracer::device_aim_tolerance`).

  **Rays at the edge of a clipping stop.** Where the stop surface itself
  clips (a fixed semi-diameter or an explicit aperture, as when the aperture
  floats by the stop size), a ray whose stop target lies within the derived
  stop of that edge is aimed at the requested tolerance, like a chief ray.
  The edge tolerance of apertures (§2) covers that residual, so a pupil
  point on the unit circle is never clipped by its own aim. Device aiming,
  with its one tolerance per request, uses the requested tolerance for such
  a system.

  **An explicit tolerance is honoured exactly.** Any `ray_aiming.tolerance`
  other than the model default (`RayAiming::DEFAULT_TOLERANCE`, 1e-10) is an
  explicit request (`RayAiming::explicit_tolerance`): the tracer derives no
  stop for it under either policy, and the engine runs and labels such a
  system `requested` whatever the request or the engine default says. The
  model default itself is what saved projects state when nobody chose a
  tolerance (the serialiser writes it, and the app's model always writes
  it), so it stays derivable; a run that needs exactly 1e-10 for every ray
  asks for `aim_stop: requested`. (A model that omits an unset tolerance
  would also tell an explicit 1e-10 from the default; the app's strict
  decoder and project schema version are unchanged, so that is left to a
  schema step together with the app.)

  The conditioned stop is qualified under the per-quantity contracts of
  GPU §9.4 ("Contracts"): the 16 samples with real aiming, perturbed and
  adversarial variants and a 29-surface F/1.4 design, every GPU-capable
  analysis on the CPU and on Metal, with identical survival masks and every
  gate met (`aim_stop_qualify` in `mklight-bench`; the decision is in
  DECISIONS.md).

Neither policy changes the budgets above, the requested-tolerance semantics
of `requested`, nor the survival rule: a ray's survived/clipped status may
only differ for rays within the aim allocation (6.7·10⁻⁷ mm) of an aperture
edge, well inside the 10⁻⁵ mm band.

**Device aim schedule cap.** The device aim solve runs the reference's
schedule (seed-table solve, chief prediction, four continuation stages, each
up to the iteration cap) in double-float; a consumer may bound a ray's
Newton iterations over the whole schedule (`AimOptions::schedule_cap`,
GPU.md "Bounded aiming commands"). Two consumers do: the candidate
evaluator's device aims stop after 8 iterations (`SCHEDULE_CAP`), so an
unaimable ray of a bad trial step need not run the schedule to its end, and
the geometric adapters' device corrections from a host seed table
(`aiming=table` and the device share of split generations) after 12
(`TABLE_SCHEDULE_CAP`). Both complete the rays the device leaves unresolved
on the host in f64. Every other device aim keeps the full schedule. The cap
never changes a ray the full
schedule converges within it (bitwise, `aim_resume.rs`), and the
arithmetic of every solve is unchanged: GPU precision is not the lever for
aiming cost (an f32 predictor phase cannot pay: DECISIONS.md, "Mixed
precision", and GPU.md §3.6; the measurements are in
`docs/history/gpu-numerics.md`).

## 8. Paraxial slopes in solves

Ray-angle solves (`marginal_ray_angle`, `chief_ray_angle`, `docs/SOLVES.md`)
take the paraxial **slope** `u = dy/dz` after a surface: dimensionless, not
an angle in degrees, and not `tan` of a real-ray angle. The JSON field is
`slope`. UI and FFI must label it as a slope (or "paraxial angle u") and pass
it through unconverted; §1's "angles in degrees" does not apply to it.

## 9. Complex amplitudes and polarisation

Coatings return complex amplitude coefficients next to the power ones
(`mklight_core::materials::coating`), and polarisation ray tracing
(`mklight_trace::polarization`) builds on them. Details: `docs/POLARIZATION.md`.

* **Time dependence** `exp(i(ωt − k·r))`, with complex indices `N = n − ik`
  (Macleod). A wave delayed by an extra optical path `L` picks up the factor
  `exp(−i2πL/λ)`, so a phase *lag* is a negative phase. Evanescent and
  absorbed waves decay because `Im(N cos θ) ≤ 0`.
* **s/p basis.** At an interface with unit normal `n̂` (either orientation)
  and incident direction `k`: `s = (k × n̂)/|k × n̂|`, `p = k × s` before and
  `p' = k' × s` after the interface (reflected or transmitted `k'`). Both
  `(s, p, k)` and `(s, p', k')` are right-handed. At normal incidence `s` is
  any unit vector normal to `k` (results do not depend on the choice,
  because then `r_p = −r_s` and `t_p = t_s`).
* With this basis the bare Fresnel coefficients are
  `r_s = (n₁c₁ − n₂c₂)/(n₁c₁ + n₂c₂)`, `r_p = (n₂c₁ − n₁c₂)/(n₂c₁ + n₁c₂)`,
  `t_s = 2n₁c₁/(n₁c₁ + n₂c₂)`, `t_p = 2n₁c₁/(n₂c₁ + n₁c₂)`. So
  `r_p = −r_s` at normal incidence, `r_p = 0` at Brewster's angle, and a
  **perfect conductor** (`E_tangential = 0`) has `r_s = −1`, `r_p = +1`. The
  perfect conductor is the phase of every phase-free reflector (bare mirrors,
  `Ideal` coatings, tabulated mirrors): it is non-retarding. A `Reflect`
  surface therefore never gets a total-internal-reflection phase, even when
  it models a prism face (TIR face, roof, rhomb): the surface has no outer
  medium. Prism faces whose retardance matters need a coating that carries
  the phase (e.g. a `ThinFilm` mirror) until such an option exists.
* At grazing incidence (`cos θ₁ = 0`, or so small that `n₁ cos θ₁` rounds
  to zero) every coefficient takes the limit `r_s = r_p = −1`, `t = 0`.
* Beyond the critical angle `|r_s| = |r_p| = 1` with **positive** phases
  `φ_s = 2 atan(g/(n₁c₁))`, `φ_p = 2 atan(n₁g/(n₂²c₁))`,
  `g = √(n₁²sin²θ₁ − n₂²)`.
* Field transmission `t` is the ratio of full field amplitudes. The power
  transmittance is `T = |t|²·flux`, with `flux = n₂c₂/(n₁c₁)` for
  non-absorbing media. Polarisation ray tracing uses the **flux-normalised**
  `t̃ = t·√flux` (same phase), so `|E|²` is the power carried by the ray.
  Reflection needs no factor (`R = |r|²`).
* Coatings without phase data: `Ideal` coatings are non-polarising
  (`r_s = −√R`, `r_p = +√R`, `t̃_s = t̃_p = +√T`). `Tabulated` coatings keep
  the bare-Fresnel phases with moduli rescaled to the tabulated powers.
* **Jones vectors** `(E_x, E_y)` are given in a right-handed transverse basis
  `(x̂, ŷ, k)`. **Stokes** parameters:
  `S₀ = |E_x|² + |E_y|²`, `S₁ = |E_x|² − |E_y|²`, `S₂ = 2 Re(E_x* E_y)`,
  `S₃ = 2 Im(E_x* E_y)`. `S₃ > 0` (e.g. `E_y = +i E_x`) is a field that
  turns from `+x̂` toward `−ŷ`: clockwise for an observer looking toward the
  source.
* **Diattenuation** `D = (T_max − T_min)/(T_max + T_min)` from the singular
  values of the Jones (or PRT) matrix. **Retardance** `δ ∈ [0, π]` is the
  eigenphase difference of the unitary factor of the polar decomposition
  `J = U·H`. Its **fast axis** is the Stokes vector of the eigenpolarisation
  with the larger phase (the leading one).
* **Pupil basis.** Jones pupils use the *double-pole* basis: the local
  `(x̂, ŷ)` of a ray with direction `k` is the frame's `(x̂, ±ŷ)` carried by
  the minimal rotation that takes the pole `±ẑ` to `k`. The pole is the
  starting frame's `+ẑ` in object space, and in image space the image
  frame's `+ẑ`, or `−ẑ` (with `−ŷ`) when the light travels toward `−z`.

## 10. Path classes and signatures

* A path's **signature** is the ordered list of physical events
  `(surface, Transmit | Reflect)` at the non-passive surfaces, from leaving
  the object to the end of the path. A designed mirror's bounce is a
  `Reflect`; detection is a final `Transmit` at the image surface. Passive
  surfaces (dummies, coordinate breaks, bare surfaces with the same index on
  both sides) never appear.
* Its **ghost reflections** are the `Reflect` events at non-mirror surfaces.
  A detected, purely specular path is `Primary` (none) or `Ghost(k)`
  (`k` of them) when its signature equals the sequential expansion of its
  ghost reflections; otherwise it is `Irregular`. Any scatter event makes it
  `Scatter`, any non-optical geometry `Mechanical`.
* Owners: `Primary` → sequential/wave solvers; `Ghost(k)` → the ghost solver
  if the path is in that run's coverage, else the stray solver;
  `Scatter`, `Mechanical`, `Irregular` → the stray solver. The stray
  solver reports `Irregular` paths, and specular paths outside a clipping
  aperture, as its class `Sneak` (five classes in all, ARCHITECTURE
  "Solvers and double counting"). The reference
  implementation is `mklight_ghost::signature` (`PathSignature`, `classify`,
  `GhostCoverage::owner`); details in `docs/GHOST.md` §2.
* Ownership assumes both solvers clip at identical apertures: the stray
  solver must clip the optical surfaces exactly as the ghost run did
  (`GhostCoverage::apertures`, checked with `same_apertures`), otherwise a
  ray clipped by one solver and not the other is counted twice or not at
  all. Both build them with `mklight_ghost::clip_apertures`.

### 10.1 Diffractive orders

A DOE physical event also carries its signed diffraction order. The designed
sequential interaction uses `trace_order` (which need not equal the blaze
`design_order`); every other order is a **diffraction ghost** if the optical
walk is sequentially consistent and inside its clipping apertures. It may
have zero extra reflections (`Ghost(0)`), or also carry reflection ghosts.
Scatter, Mechanical and aperture/sequence violations still take precedence.

`GhostPath` preserves reflection-array JSON for existing paths. Order-changing
paths use `{reflections:[…], diffraction:[{surface,visit,order},…]}`, where
visit is the zero-based encounter with that surface, including its reflected
visits. Overrides are unique and sorted by (surface,visit); omitted visits
use the selected order. `Step.order` / `PathEvent.order` are overrides;
combined with the immutable snapshot they specify the complete ordered
(surface, interaction, order) signature. Stray signatures publish every DOE
visit, including selected orders, with its physical interaction.

Coverage matches both reflection sequence and exact order overrides. Covered
specular diffraction ghosts belong to the explicit ghost run, uncovered ones
to stray, while selected-order Primary belongs to sequential/wave. A ghost
run with no diffraction orders covers only selected-order reflection paths.
Thus a covered reflection sequence never excludes its other DOE orders.
As with reflection-signature overflow, combined runs reject coverage with
more DOE visits than the compact stray signature can record.

## 11. Non-finite numbers in results

Result JSON follows §2 for infinities (`"inf"` / `"-inf"`) and writes NaN,
which means *undefined* (a vignetted field, a failed ray, a quantity without
a value), as `null`. Readers accept numbers, `null`, `"inf"`, `"-inf"` and
`"nan"`. `mklight_analysis::util::nf` implements this for `f64`, vectors and
fixed arrays.

## 12. GPU traces and path programs

* A GPU trace reports f32 points and directions with a double-float optical
  path that refers to the **reported** point: the rounding of the point
  along the ray is moved into the OPL. Compare OPLs of two traces at the
  same point (`OPL_ref + n·(p_gpu − p_ref)·d`), and form OPD from the
  reported values as `mklight_trace::opd` does.
* In a path program (ghost paths) the medium of the gap between surfaces
  `j` and `j + 1` is the medium after surface `j`, in both travel
  directions; a backward step uses the inverse of surface `j`'s
  `to_next` transform. Sequential steps reproduce `mklight_trace`.
* Direction of travel through the surface sequence in a path program: a
  transmission keeps it, a ghost reflection at a refracting surface
  reverses it, and a reflection at a mirror keeps it (forward or
  backward), because both gaps next to a sequential mirror lie on its
  reflecting side. Transmission through a mirror does not exist.
* Precision qualification is offline evidence, not a run-time gate: no
  run compares a GPU result with the reference. `xval::qualify` passes a
  system on a backend only with the sampled errors within a third of the
  §7 budget (`docs/GPU.md` §5.2), and the offline accuracy harnesses judge
  complete analyses by the contracts of GPU §9.4. Which analyses run
  natively, and how a failing native run falls back and is reported, is
  §7 above and GPU §9.2–9.3; `RunInfo.backend` always names the backend
  that computed the result (§14).

## 13. Structural validation codes

`OpticalSystem::validation_issues` reports each problem as
`[code] path: message`, with `path` the JSON path of the value
(`surfaces[3].shape.curvature`, `coatings["AR"].layers[0].extinction`).
Numbers must be finite everywhere except the object thickness (and an
override or pickup of it) and the image thickness. Codes:
`non_finite`, `too_few_surfaces`, `stop_out_of_range`,
`invalid_aperture_value` (≤ 0), `invalid_wavelengths`,
`invalid_wavelength` (λ ≤ 0), `negative_weight` (field or wavelength),
`no_fields`, `object_height_at_infinity`, `object_na_at_infinity`,
`field_angle_too_large`, `invalid_environment`, `invalid_ray_aiming`,
`duplicate_surface_id`, `infinite_thickness`, `invalid_focal_length`,
`invalid_material` (also a model glass with `n_d ≤ 1`), `unknown_coating`,
`invalid_semi_diameter`,
`invalid_aperture`, `invalid_mechanical_semi_diameter`, `invalid_scatter`
(also a TIS outside `[0, 1]`), `invalid_vignetting` (a pupil compression
`vcx`/`vcy` outside `[0, 1)`),
`coordinate_break_at_end`, `image_mirror`, `invalid_coating` (also an
empty reflectance table, or one whose wavelengths do not increase),
`missing_surface` (mechanics, overrides, solves and pickups),
`invalid_mechanics`, `invalid_sensor` (also a spectral response whose
wavelengths do not increase), `invalid_configuration`,
`invalid_solve`, `invalid_sag_overlay_base` (non-geometric overlay base),
`invalid_element_range` (an unordered or non-interior rigid
range). A configuration override is checked by the same rules on
the configured system: an issue it causes is reported at
`configurations[k].overrides[j]` with the underlying code
(`invalid_aperture_value` for an F/# override of 0). Unit variants of
tagged enums (`{"mode": "auto"}`, `{"type": "plane"}`, …) take no other
keys, like every other object of the model. Codes are stable; messages
may change. The macOS app's model mirrors these rules
(`OpticalSystem.structuralIssues()`); `validation_cases.json`, written by
`dump_fixtures`, holds the engine's verdicts both are tested against.

## 14. Analysis results

* Every engine analysis publishes a settings schema (with defaults) and a
  versioned result schema whose `x-buffers` declare its typed buffers;
  `docs/RESULTS.md` documents them and is generated from the schemas.
* `RunInfo.backend` names the backend that computed a result (after any
  fallback, §7), not the one requested; the envelope's `backend` object
  records the request and the reason for a fallback.
* Ray clouds, grids, maps and scene geometry travel as row-major buffers
  with NaN for "no value" (a ray that did not arrive); curves along one
  axis stay in the JSON (§11 rules for non-finite numbers).
* Image simulation rasters use row zero at low y, rows increasing with image
  y and columns with x, with channels interleaved. Every source follows this
  order: top-to-bottom file rows and text glyphs are converted on input.
  Upright display traverses rows from high to low y.

## 15. Wave-optics pupils

* Normalised exit-pupil coordinates `(u, v)` are **positions on the
  reference sphere** relative to its centre, divided by `R·sinθ'max`, with
  `u` along image `x` and `v` along image `y`. A ray converging to the
  centre with direction `k̂` sits at `−(k̂·ê_u, k̂·ê_v)/sinθ'max`.
* An image-space telecentric system (exit pupil at infinity) takes the
  `R → ∞` limit of the §4 OPD:
  `OPD = (OPL_c − OPL_r) + n'(d_r·(Q_r − P) − d_c·(Q_c − P))` for image
  points `Q`, directions `d` and sphere centre `P`.
* When light reaches the image toward local `−z`, wave optics works in the
  image frame mirrored in `z`, so maps, PSFs and MTF directions are in the
  image surface's `x`, `y`. Details: `docs/WAVE.md`.

## 16. Extended surfaces and diffractive phase

* `docs/SURFACES.md` defines the additional surface bases, coefficient
  ordering, domains, vertex exceptions and structural limits. Sag and its
  first two derivatives live in core; tracing uses the same definitions.
* Normalisation radii and sag coefficients are in mm; physical power-series
  coefficients carry the corresponding inverse powers of mm. Zernike sag
  amplitudes are mm, distinct from Zernike wavefront amplitudes in waves.
* DOE Φ is an order-1 phase **lag**, `exp(−imΦ)` in §9's time convention.
  A fixed structure adds `mλΦ/(2π)` to OPL and `mλ∇ₛΦ/(2π)` to tangential
  index-weighted direction. λ is converted from µm to mm. Selected order
  controls tracing; design wavelength/order do not rescale the structure.
* The existing first-order report remains the y sweep. `ParaxialXY` adds x,
  with signed squeeze `EFL_y/EFL_x`; mixed xy power requires a coupled model
  and is explicitly warned about. An odd r term has an axial normal by
  convention at its vertex but no Hessian there.
* Additional §13 codes: `invalid_norm_radius`, `coefficient_count`,
  `invalid_design_wavelength`, `invalid_diffraction_order`,
  `invalid_diffractive_base`. The corresponding limits are in SURFACES §7.
