# MKLight Ghost Solver (`mklight-ghost`)

Ghosts are the deterministic, purely specular paths that reach the image
after an even number of *extra* reflections at refracting interfaces
(including cemented ones) or at the sensor. This crate enumerates them,
traces them with real rays and first-order optics, and reports ghost foci,
spot sizes, irradiance maps and ghost-to-signal ratios. It owns path class
`Ghost(k)` of the solver partition (ARCHITECTURE "Solvers and double
counting"; CONVENTIONS §10). Read `docs/CONVENTIONS.md` first: units, signs,
frames and radiometry are the same as in the reference tracer.

| module | contents |
|---|---|
| `path` | `GhostPath` (ordered extra reflections), expansion into `Step`s, `enumerate` with filters and budget |
| `signature` | `InterfaceKind`, `PathSignature`, `classify`, `GhostCoverage`: the partition rule shared with the stray solver |
| `model` | `GhostModel`: reference tracer + interface kinds, clipping apertures, inverse frame transforms, sensor reflectance |
| `sequence` | the general sequence tracer (`GhostModel::trace_steps` / `trace_from`) |
| `paraxial` | y–nu along unfolded ghost sequences, `paraxial_ghosts` ranking, `real_ghost_focus` |
| `irradiance` | `ghost_irradiance`: real-ray maps and ratios (rayon, deterministic, cancellable) |

## 1. Paths

A `GhostPath` is the list of surfaces where light is reflected **in
addition to** the designed sequential interactions, in the order they
happen. Light leaves the object forward (increasing surface index); each
reflection reverses the direction of travel along the surface order, so the
list alternates `r₀ > r₁ < r₂ > …` (`r₀ ≥ 1`, at most the image index). Between
reflections every surface is met with its designed interaction (refraction
in either direction, a designed mirror's reflection, nothing for dummies and
coordinate breaks). An even number of reflections ends at the image; an odd
number ends by leaving through surface 1 into object space (not detected,
used in the energy bookkeeping). The empty path is the primary path.

The classic two-reflection ghost `[i, j]`, `j < i`, expands into
`1 … i` forward (reflect at `i`), `i−1 … j` backward (reflect at `j`),
`j+1 … N−1` forward.

**Reflectors** are the `Refractive` interfaces (below) and, when
`GhostSettings::sensor_reflectance` is set, the image surface. Designed
mirrors, ideal lenses, dummies, coordinate breaks and bare same-index planes
(a stop in air) cannot carry a ghost reflection.

**Enumeration** (`enumerate`) lists every alternating sequence of 2, 4, 6 or
8 reflections over the reflectors that passes the filters:
`surface_range`, `exclude_surfaces` and `min_reflectance_product` (the
unpolarised normal-incidence reflectance product at the primary wavelength,
from the shared coating code, with the correct incidence side for each
reflection). Products never increase as reflections are added, so the search
prunes on them. All 2-reflection paths that pass are kept. Higher orders
compete for `max_higher_order_paths` slots in a bounded heap ordered by
(product, path); once it is full, a branch below the weakest kept product is
pruned, and the kept set is exactly the top of that total order (independent
of search order; tested against an unbudgeted run). The output is sorted by
order, then lexicographically, and comes with its `GhostCoverage`.

## 2. Partition rule (shared with the stray solver)

`interface_kinds` classifies each surface: `Object`, `Passive` (dummies,
coordinate breaks, and bare refracting surfaces with the same index on both
sides at every system wavelength), `Refractive`, `IdealLens`, `Mirror`,
`Image`.

A `PathSignature` is the ordered list of physical events
`(surface, Transmit | Reflect)` at the non-passive surfaces, from leaving
the object to the end of the path. Events are *physical*: a designed
mirror's bounce is `Reflect`, detection is a final `Transmit` at the image.
Passive surfaces never appear, so a non-sequential tracer that does not know
about them produces the same signature. The **ghost reflections** of a
signature are its `Reflect` events at non-mirror surfaces; given those, the
sequential model fixes the rest, and `PathSignature::for_path` gives the
*canonical* signature of a `GhostPath`.

`classify(signature, kinds, scattered, mechanical)`:

1. any scatter event → `Scatter`;
2. else any interaction with non-optical geometry → `Mechanical`;
3. else not ending in `(image, Transmit)` → `Undetected`;
4. else, with `g` = the ghost reflections, if the signature equals the
   canonical expansion of `GhostPath(g)`: `Primary` (`g` empty) or
   `Ghost { reflections: |g| }`;
5. else → `Irregular` (specular, but not sequentially consistent, e.g. it
   skipped an element).

`GhostCoverage::owner` assigns the counting solver: `Primary` → sequential;
`Ghost` → ghost **iff the path is in the coverage of the ghost run** (the
exact set it traced, after order limit, filters and budget), else stray;
`Scatter`, `Mechanical`, `Irregular` → stray; `Undetected` → nobody. Every
detected path has exactly one owner, and a ghost the ghost run left out is
still counted, by the stray solver. The stray solver records signatures of
its Monte-Carlo paths (at the same surfaces, with the same kinds) and calls
`owner` with the coverage of the ghost run it is combined with.

Ownership by signature is only disjoint and complete if both solvers clip
a ray at the same places. If the stray solver's apertures were wider than
the ghost run's, a ray the ghost run clipped would survive there with a
covered signature and be dropped by both; if narrower, a ray the ghost run
counted would also be counted by stray as `Mechanical` after hitting a
mount. `GhostCoverage` therefore records the per-surface clipping of the
ghost run (`apertures`, from `GhostModel::coverage`), and the stray solver
must clip the optical surfaces identically (`GhostCoverage::same_apertures`).
Both build them with `model::clip_apertures`; the stray side checks a
coverage record with `StraySystem::check_ghost_coverage` (`docs/STRAY.md`).
The radii are compared exactly, so both must come from one sizing:
`GhostModel::new` sizes with the default `SemiDiameterSettings`,
`GhostModel::sized` with given settings (cancellable), and
`GhostModel::with_semi_diameters` takes a stray scene's
(`StraySystem::semi_diameters`). Ghost paths outside the coverage stay
with the stray solver (`StraySettings::covered_ghosts`).

## 3. The sequence tracer

`GhostModel::trace_steps(start, wl, steps, record)` traces one ray along an
arbitrary `Step` list (`trace_from` starts in any surface's frame). It
reuses `mklight_trace::surface::intersect`, `interact::{refract, reflect,
ideal_lens}` and `trace::transfer_into`; no intersection or Snell math is
duplicated.

**Frames.** A ray is always in the local frame of the surface it last met.
Forward (`i−1 → i`) uses the reference transfer: with
`to_next(i−1) = (o, R)`, `p_i = Rᵀ(p_{i−1} − o)`, `d_i = Rᵀd_{i−1}`, or the
exact `−t` translation on axial gaps. Backward (`i+1 → i`) uses the inverse
of `to_next(i)` in the same subtract-then-rotate form,
`p_i = R(p_{i+1} − q)`, `d_i = R d_{i+1}`, with `q = −Rᵀo` (vertex `i` in
frame `i+1`) precomputed per surface, or the exact `+t_i` translation. The
composition backward∘forward is the identity to rounding (tested on tilted
and decentered elements), and reversed rays retrace forward ones surface by
surface (tested on the Double Gauss, including a ghost's backward segment).

**Media.** Arriving forward at `i`, the incident medium is the one before
`i` and the far medium the one after it; arriving backward they swap. A pass
refracts into the far medium; any reflection (designed mirror or ghost)
stays in the incident medium. The gap between `i` and `i+1` is in the medium
after `i` whichever way it is crossed, and absorbs with `alpha_after[i]` ×
geometric length. An ideal lens crossed backward has the image-space focal
length `f·n_before/n_after`, the exact inverse of the forward crossing.

**Clipping** (`ClipMode`). Ghost light leaves the designed footprint, so
apertures matter. `ClearApertures { margin_mm }` (default) clips any surface
at its fixed aperture if it has one, the stop at
`max(paraxial stop radius, traced height)`, and every other `Auto` surface
that is physically present (not `Passive`, not the image) at its traced
clear semi-diameter plus the margin. Dummies, coordinate breaks and bare
same-index planes that are neither the stop nor have an explicit aperture
never clip: they are not there physically, and inserting one leaves every
ghost result (real and paraxial) unchanged (tested). `FixedApertures` clips
at the fixed apertures plus an `Auto` stop at the radius above (the
reference trace aims every ray inside the stop; without this an overfilled
grid would let light around the stop and the primary flux exceed the
entrance-pupil flux). `None` disables clipping. A sensor reflection only
happens inside the `Sensor` rectangle when the model has one (tested).

**Virtual segments.** With `reject_virtual` (default) the signed distance
travelled since the last physical interaction (any non-passive surface) is
summed over passive surfaces; a negative sum at a physical surface loses the
ray (`Virtual`). Passive planes may sit "behind" the ray as in the
sequential model. For an object at infinity nothing is counted before the
first physical interaction (light comes from `z = −∞`; the start plane and
any passive planes before the first element, e.g. a stop plane at the
vertex of a concave first surface, are reference planes). Tested: a
concave first surface behind a stop plane and a stop plane listed after a
lens but placed inside it are accepted and match the reference tracer;
overlapping elements are rejected.

The rule applies to the primary path too (it is the signal reference of
the maps), so a design ray that crosses a physically impossible element
(negative edge thickness, overlapping elements) is lost here although the
reference tracer passes it. `ghost_irradiance` counts such primary rays and
reports them, with their share of the signal, as a `RunInfo` warning. (The
Double Gauss sample without its vignetting factors triggers it at full
field: its last element has a negative edge thickness at the unvignetted
14° marginal ray height.)

**Radiometry.** At every interaction the shared coating code
`ResolvedCoating::rt(λ, n_in, n_far, cos θ, mirror)` (bare Fresnel, thin film,
ideal, tabulated) is evaluated once at the actual angle of incidence. A pass
multiplies the s and p channel weights by `T_s, T_p`, a ghost reflection by
`R_s, R_p`, a designed mirror by its mirror `R_s, R_p`, a sensor reflection
by the sensor reflectance. The channels are averaged once at the end:
`w = ½(w_s + w_p)·exp(−Σαℓ)`. This is exact for meridional rays of
rotationally symmetric systems and neglects the rotation of the plane of
incidence between surfaces for skew rays (the polarisation module is the
place for a full treatment). Because the primary pass and the ghost
reflection at an interface come from one `rt` call at the same incidence,
`T + R + A = 1` holds at every interface (tested, bare and absorbing
coatings), and the sum over all orders of detected plus returned plus
absorbed flux is 1 (tested with absorbing glass and coatings to 1e-14).
With a sensor reflectance `R_s` the sensor is an interface too: a sensor
reflection multiplies by `R_s`, detection by the absorbed fraction `1 − R_s`,
so the detected flux of all orders plus the flux returned to object space
is 1 (tested for `R_s = 0, 0.3, 1`; a perfectly reflecting sensor detects
nothing). Ratios to the primary are unaffected, since every detected path
carries the same final factor.

## 4. Paraxial ghost analysis

`paraxial_ghosts` ranks every path in microseconds. The y–nu trace of the
reference paraxial model runs along the **unfolded** step list
(`paraxial::trace_steps`): the signed reduced index `N = σn` starts at
`+n_object` and flips at **every** reflection; pass at a conic
`φ = c(N' − N)` (the same surface power in both directions), ideal lens
`φ = n_after/f`; reflection `N' = −N`, `φ = −2cN` (the sensor included,
with its own curvature, as the real tracer reflects about the curved normal;
tested against a vergence calculation with a defocused curved sensor);
transfer `y' = y + Δz·ω/N` with `Δz = t_i` forward and `−t_{i−1}` backward.
The empty path reproduces the reference first-order rays to 1e-12 (tested
for every sample and wavelength).

Starting from the reference marginal and chief rays at surface 1, at the
image surface:

* `focus_from_sensor = −y_m/u_m`: signed distance of the paraxial ghost
  image from the sensor along the image's local z (negative: in front);
* `spot_diameter = 2|y_m|`; `image_height = y_c` (maximum field);
* `vignetting V`: the smallest `(r_clip/|y_m|)²` over path surfaces whose
  clip radius the ghost marginal ray overfills;
* `relative_irradiance ≈ (ΠR)(ΠT_g/ΠT_p)·(D_s/max(D_g, D_s))²·V`, with
  normal-incidence unpolarised coating values and axial bulk absorption, and
  `D_s` the signal spot diameter (largest of the primary paraxial blur at the
  sensor, the Airy diameter `2.44λN_w` and the pixel pitch, or set);
* `near_focus`: `D_g ≤ near_focus_factor·D_s` — ghosts that focus near the
  sensor, the dangerous ones.

`real_ghost_focus` traces the on-axis meridional ray at pupil height `ρ`
along a path and returns its axis crossing relative to the sensor; it
converges quadratically to the paraxial value (tested on a singlet against a
hand-derived vergence calculation of the inner reflections as mirrors in
glass, and on a plate behind an ideal lens, where the ghost focus lies `2d/n`
in front of the primary).

## 5. Real-ray irradiance

The fields are the selected system fields (`fields`, with their weights and
vignetting factors; empty = all, unless points are given) followed by
arbitrary normalised `FieldPoint`s (`points`, weight 1 each), so a bright
source outside the field of view (the sun) is analysed without editing the
model, which would change the snapshot. Per-field results carry a
`FieldRef` (`System { index }` or `Point { point }`).

`ghost_irradiance` traces, for each selected field and wavelength, a square
grid of `pupil_samples²` cell centres over a disc of `overfill` entrance-pupil
radii around the chief ray's aim point (ghost light may enter outside the
pupil and still pass the stop after a reflection; the apertures decide).
Starting rays come from the reference tracer (`SequentialTracer::start_ray`,
including real ray aiming) and are shared by all paths. Each ray carries
`w_f w_λ h²/π` (normalised field and wavelength weights, wavelength weights
including the sensor's spectral response, `h` the cell size in pupil
radii), so flux is relative to the flux entering the paraxial entrance
pupil.

The grid is a base quadrature, not a binary area estimate at every cell
centre. For each path, compare a cell's centre and four corners by survival
and whether the detector hit is on the map. If they disagree, subdivide
into four equal-area cells, recursively to four levels (1/16 of the base
cell width). The circular source boundary is refined too, including cells
whose centres are outside the source but whose area intersects it. Each
leaf receives its own area weight. `rays_traced`/`rays_detected` count leaf
samples, excluding corner probes; they no longer imply equal ray weights.
Refinement order is fixed, and cancellation is also checked per source-grid
row. The finite refinement is not a rigorous quadrature error bound: raise
`pupil_samples` to resolve features smaller than a base cell and to check
convergence. Pixel interiors still use point binning.

Detected rays are binned into `map_bins` in the image's local x, y, as flux
per mm², over an explicit `map_half_size` if given, else the model `Sensor`
rectangle, else the square of the image clear semi-diameter. The primary
image is computed by the same code from the empty path. Per path and field:
flux (on and off the map), peak, centroid, RMS radius; per path:
`flux_ratio` (ghost/primary flux on the map) and `peak_ratio` (maximum over
fields of ghost peak/primary peak; meaningful for bins about a pixel and many
rays per bin); both are 1 for the primary itself. Paths are ranked by `peak_ratio`; the total ghost map is the
sum of path maps.

Rays are traced with rayon in blocks of 16384; results are collected in the
fixed order (field, wavelength, pupil) and binned sequentially with
compensated sums, so maps are bit-identical between runs and thread counts
(tested with rayon pools of 1 and 5 threads).
The `JobContext` is checked before every block; progress is monotone and
ends at 1. Results carry `RunInfo` (backend `cpu-f64`, settings of the model
and of the analysis, no seed: nothing is random).

### High-reflectance discrepancy: pupil quadrature, not transport

A singlet with ideal 30 % reflectors, a 24 × 24 mm detector, one wavelength
(0.55 µm), and ghost path `[2, 1]` exposed aliasing of the detector boundary
by the old cell-centre grid. Relative to entrance-pupil flux, on axis:

| base grid | old centre-only flux | boundary-refined flux |
|---|---:|---:|
| 120² | 2.63592417e-3 | 2.57960188e-3 |
| 240² | 2.63592417e-3 | 2.57579215e-3 |
| 480² | 2.59473785e-3 | 2.57778067e-3 |
| 960² | 2.57043793e-3 | 2.57764714e-3 |

The unchanged 120²/240² answer was false convergence. A sensor accepts only
a small part of this broad ghost, so whole-cell acceptance inflated its
pupil area. The same effect explains the small negative z scores at 5 %;
it scales with the ideal coating weight `(1 − R)² R²`.

Targeted isolation: replay exactly the stray solver's Philox launch rays in
the sequence tracer, with identical apertures, one wavelength, no scatter,
no sensor reflection, and enough splitting to follow every order-2 branch.
At 200,000 rays, seeds 7–9 and both singlet fields, the two order-2 fluxes
agree within 4e-14 relative. The coefficients are angle independent and
identical for s/p; both sums are incoherent. Thus neither angle/polarisation,
sensor absorption, roulette, clipping geometry nor source normalisation
causes this discrepancy. Compare **order 2**, not all ghost orders: at 30 %,
higher orders contribute measurable flux.

`mklight-stray/tests/partition.rs` retains the paired flux and exact per-ray
standard-error checks, requires the 120²/480² refined fluxes to agree within
0.2 %, and requires |z| < 3 for each seed and the combined mean. The normal
suite uses 60,000 rays × three seeds × two fields; the ignored large test
uses 2,000,000 rays × four seeds × two fields. The large run gave
z from −0.73 to +1.54, with combined-mean z = +1.12 (on axis) and
+0.76 (off axis):

```sh
cargo test --release -p mklight-stray --test partition high_reflectance_ghost_flux_large_run -- --ignored --nocapture
```

## 6. GPU port: per-ray kernel interface

The work is independent rays indexed `(path, field, wavelength, pupil)`;
starting rays depend only on `(field, wavelength, pupil)`. The host schedules the boundary-refinement probes and leaf rays per path
(§5); only base starting rays are shared across paths. The kernel the
ghost solver needs from `mklight-gpu` (a `Job::GhostTrace`, added additively
by the GPU crate) mirrors `GhostModel::trace_steps` one-to-one:

* **Uploaded once per snapshot** (in addition to the packed system):
  per surface the inverse transfer `(q, R)` of `to_next` and the axial flag
  (both directions then use the exact `±t` translation on axial gaps), the
  ghost clip (`none | surface aperture | radius`), the interface kind, and
  per (surface, wavelength, incidence side) a coating table of
  `R_s, R_p, T_s, T_p` against `cos θ` (the TMM stays on the host;
  bare Fresnel can be evaluated in-kernel); sensor reflectance (reflection
  `× R_s`, detection `× (1 − R_s)`) and rectangle.
* **Paths** as their reflection lists (≤ 8 × u16 plus a count). The kernel
  expands the steps itself from the alternation rule (`path::GhostPath::steps`),
  so no per-step arrays are needed.
* **Start rays** as the existing `RayBatch` SoA in the starting frame, with
  the wavelength index and flux weight; a launch covers
  `paths × start rays` with `ray = gid % n_start`, `path = gid / n_start`.
* **Per-ray output**: status word (code incl. `VIRTUAL`, step index),
  image-local `(x, y)` and the unpolarised weight `½(w_s + w_p)·τ`
  (12 bytes + status). Binning stays on the host in fixed order (or a
  deterministic segmented reduction), never float atomics, so maps remain
  reproducible.
* Precision: positions in local frames, backward transfers in the
  subtract-then-rotate form, no OPL needed for irradiance; the result is
  validated against `trace_steps` with the CONVENTIONS §7 budget on hit
  points and a 1e-5 relative budget on weights.

## 7. Limitations

* Incoherent sums: no interference between ghost orders or with the
  primary (no etalon fringes); diffraction of ghost beams is ignored.
* s/p are carried in the local plane of incidence of each surface (exact
  for meridional rays, approximate for skew rays).
* The sensor reflectance is unpolarised and angle independent.
* The paraxial estimate uses the unfolded, rotationally symmetric
  equivalent and the on-axis beam for the spot size and vignetting.
* Orders above 8 are not enumerated (they belong to the stray solver via
  the coverage rule).
* The primary signal of the maps excludes design rays that cross a
  physically impossible element (reported as a warning, §3).
