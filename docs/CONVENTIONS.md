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

## 8. Path classes and signatures

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
  `Scatter`, `Mechanical`, `Irregular` → the stray solver. The reference
  implementation is `mklight_ghost::signature` (`PathSignature`, `classify`,
  `GhostCoverage::owner`); details in `docs/GHOST.md` §2.
