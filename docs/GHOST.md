# MKLight Ghost Solver (`mklight-ghost`)

Ghosts are the deterministic, purely specular paths that reach the image
after an even number of *extra* reflections at refracting interfaces
(including cemented ones) or at the sensor. This crate enumerates them,
traces them with real rays and first-order optics, and reports ghost foci,
spot sizes, irradiance maps and ghost-to-signal ratios. It owns path class
`Ghost(k)` of the solver partition (ARCHITECTURE "Solvers and double
counting"; CONVENTIONS §8). Read `docs/CONVENTIONS.md` first: units, signs,
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
apertures matter. `ClearApertures { margin_mm }` (default) clips every
surface `1 … N−2` at its fixed aperture if it has one, the stop at
`max(paraxial stop radius, traced height)` and every other `Auto` surface at
its traced clear semi-diameter plus the margin. `FixedApertures` clips like
the reference trace; `None` disables clipping. A sensor reflection only
happens inside the `Sensor` rectangle when the model has one.

**Virtual segments.** With `reject_virtual` (default) the signed distance
travelled since the last physical interaction (any non-passive surface) is
summed over passive surfaces; a negative sum at a physical surface loses the
ray (`Virtual`). Passive planes may sit "behind" the ray as in the
sequential model; the first segment of an object at infinity starts on an
arbitrary reference plane and is not counted.

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
Detection counts the flux incident on the sensor (the sensor's own
reflectance is a separate ghost source, not a loss of signal).

## 4. Paraxial ghost analysis

`paraxial_ghosts` ranks every path in microseconds. The y–nu trace of the
reference paraxial model runs along the **unfolded** step list
(`paraxial::trace_steps`): the signed reduced index `N = σn` starts at
`+n_object` and flips at **every** reflection; pass at a conic
`φ = c(N' − N)` (the same surface power in both directions), ideal lens
`φ = n_after/f`; reflection `N' = −N`, `φ = −2cN` (flat sensor `φ = 0`);
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

Detected rays are binned into `map_bins` over the sensor (model `Sensor`
rectangle, else the square of the image clear semi-diameter, else
`map_half_size`), in the image's local x, y, as flux per mm². The primary
image is computed by the same code from the empty path. Per path and field:
flux (on and off the map), peak, centroid, RMS radius; per path:
`flux_ratio` (ghost/primary flux on the map) and `peak_ratio` (maximum over
fields of ghost peak/primary peak; meaningful for bins about a pixel and many
rays per bin). Paths are ranked by `peak_ratio`; the total ghost map is the
sum of path maps.

Rays are traced with rayon in blocks of 16384; results are collected in the
fixed order (field, wavelength, pupil) and binned sequentially with
compensated sums, so maps are bit-identical between runs and thread counts.
The `JobContext` is checked before every block; progress is monotone and
ends at 1. Results carry `RunInfo` (backend `cpu-f64`, settings of the model
and of the analysis, no seed: nothing is random).

## 6. GPU port: per-ray kernel interface

The work is independent rays indexed `(path, field, wavelength, pupil)`;
starting rays depend only on `(field, wavelength, pupil)`. The kernel the
ghost solver needs from `mklight-gpu` (a `Job::GhostTrace`, added additively
by the GPU crate) mirrors `GhostModel::trace_steps` one-to-one:

* **Uploaded once per snapshot** (in addition to the packed system):
  per surface the inverse transfer `(q, R)` of `to_next` and the axial flag
  (both directions then use the exact `±t` translation on axial gaps), the
  ghost clip (`none | surface aperture | radius`), the interface kind, and
  per (surface, wavelength, incidence side) a coating table of
  `R_s, R_p, T_s, T_p` against `cos θ` (the TMM stays on the host;
  bare Fresnel can be evaluated in-kernel); sensor reflectance and
  rectangle.
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
