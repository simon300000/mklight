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
* **Tilts and decenters** (`Placement`): decenter `(dx, dy)` in mm, then
  rotations `tilt_x`, `tilt_y` and `tilt_z` in degrees, applied in that order
  by the right-hand rule, unless `order = TiltThenDecenter`. A `CoordinateBreak`
  surface only changes the frame. Any surface's placement can be `Restore`d after
  the surface. Rotations act on the frame, so all following surfaces move too.
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

## 7. Precision and error budget

The **CPU f64 tracer (`mklight-trace`) is the reference**. GPU backends compute
in f32, using these techniques:

1. Trace in surface-local coordinates. Translate by the (exactly representable)
   per-surface thickness, never by global positions.
2. Intersect with cancellation-free closed forms. For a conic, solve
   `t = B / (A·… )` in the "stable root" form with `B = c·(…)`, choosing the
   root that avoids subtracting nearly equal numbers. Then take ≤ 3 Newton steps
   for asphere departures, with the error bounded by a tolerance on the sag
   residual.
3. Use the stable vector Snell law
   `k' = μ·k + (√(1 − μ²(1 − (k·n)²)) − μ·(k·n))·n`, computed with
   `fma` and `1 − x² = (1 − x)(1 + x)` where that helps.
4. Accumulate optical path length in **double-float** (`float2` hi/lo via
   two-sum / fast-two-sum). OPD is formed as a difference against the chief
   ray in double-float before rounding.
5. Sum reductions (spot centroids, RMS, energy) with Kahan/Neumaier
   compensation, or as a pairwise tree.
6. Prefer `x·x − y·y` in factored form, `hypot`, and `fma`. Never test floats
   for exact equality, except against the sentinel values listed in the kernel
   header.

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

If a backend can't meet the budget for a design, it must say so and fall back
to the CPU reference. It must never return silently degraded numbers.

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
* A backend that fails precision qualification for a system is not used
  for it: the result is computed by the CPU f64 reference and its `RunInfo`
  names the rejected backend and the reason (`docs/GPU.md` §5.2). A
  qualification passes only with the sampled errors within a third of the
  §7 budget; a cancelled qualification is an error, not a failure.

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
`invalid_material`, `unknown_coating`, `invalid_semi_diameter`,
`invalid_aperture`, `invalid_mechanical_semi_diameter`, `invalid_scatter`
(also a TIS outside `[0, 1]`), `invalid_vignetting` (a pupil compression
`vcx`/`vcy` outside `[0, 1)`),
`coordinate_break_at_end`, `image_mirror`, `invalid_coating`,
`missing_surface` (mechanics, overrides, solves and pickups),
`invalid_mechanics`, `invalid_sensor`, `invalid_configuration`,
`invalid_solve`. A configuration override is checked by the same rules on
the configured system: an issue it causes is reported at
`configurations[k].overrides[j]` with the underlying code
(`invalid_aperture_value` for an F/# override of 0). Unit variants of
tagged enums (`{"mode": "auto"}`, `{"type": "plane"}`, …) take no other
keys, like every other object of the model. Codes are stable; messages
may change.
