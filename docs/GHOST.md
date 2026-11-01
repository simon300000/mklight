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

### 2.1 Diffraction ghosts

A purely specular DOE path in a non-selected order is `Ghost(k)`, with k
extra reflections (possibly zero), when it follows the same unfolded
sequential walk. `trace_order` defines Primary, not `design_order`.
Reflection paths and diffraction ghosts share one `GhostPath`: existing
reflection-array JSON remains valid; an extended path is
`{"reflections":[],"diffraction":[{"surface":2,"visit":0,"order":0}]}`.
Visits count encounters with that surface; overrides name non-selected orders
and must be unique and sorted by (surface,visit). `check_path` validates them
at the solver boundary. The resulting ordered surface/interaction/order
signature distinguishes multiple encounters with the same DOE.

Optional `EnumerationSettings::diffraction_orders` expands the primary and
every enumerated reflection path into the Cartesian product of selected and
requested orders at all DOE visits. `max_diffraction_paths` (default 10000)
bounds this expansion; exceeding it is an error, so no incomplete coverage
is claimed. Explicit `paths` can instead select exact combinations. Empty
`diffraction_orders` preserves the reflection-only run. Orders outside the
requested window stay with stray; coverage is never inferred from an order
range or just from a reflection sequence.

Real rays use the reference `diffract` grating equation and phase OPL jump,
and multiply each outgoing coating channel by its scalar η from SURFACES
§3.1. Reflected and transmitted visits use the same manufactured relief
with the appropriate index contrast. Paraxial ghost ranking includes DOE
power for each chosen order and its efficiency. Implicit sag geometry uses
the reference intersection for every extended surface type.

The hybrid achromat partition regression uses a four-level DOE and requested
orders −1, 0, 2, 3, alongside the selected order 1. It checks covered and
uncovered diffraction flux, primary, and combined totals against MC standard
errors, with absorption/escape accounted for by the MC energy ledger.

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
including real ray aiming). Each ray carries
`w_f w_λ h²/π` (normalised field and wavelength weights, wavelength weights
including the sensor's spectral response, `h` the cell size in pupil
radii), so flux is relative to the flux entering the paraxial entrance
pupil.

The grid is a base quadrature, not a binary area estimate at every cell
centre. For each path, a cell whose centre and four corners disagree on
survival or on whether the detector hit is on the map is split into four
equal-area cells, recursively to four levels (1/16 of the base cell width).
The circular source boundary is refined too, including cells whose centres
are outside the source but whose area intersects it. `rays_traced` /
`rays_detected` count leaf samples (not corner probes), whose weights are
their own cell areas. The refinement is not a rigorous error bound: raise
`pupil_samples` to resolve features smaller than a base cell and to check
convergence. Pixel interiors still use point binning.

Centres and corners lie on an integer lattice of 32 units per base cell;
`irradiance::pupil` is the one place a lattice point becomes a pupil
coordinate, so a point has one representation. Per path, a hash map from
(field, wavelength, lattice point) to the traced result makes every point
traced once: refinement runs level by level, tracing the level's new centres
and then its cells' new corners in parallel batches. Neighbouring cells share
corners, children reuse their parent's centre and corners, and a repeated
wavelength reuses its probes. The leaves are then read from the map depth
first, exactly as an uncached recursion would produce them (tested bit for
bit, together with the traced set equalling the distinct lattice points).

Detected rays are binned into `map_bins` in the image's local x, y, as flux
per mm², over an explicit `map_half_size` if given, else the model `Sensor`
rectangle, else the square of the image clear semi-diameter. The primary
image is computed by the same code from the empty path. Per path and field:
flux (on and off the map), peak, centroid, RMS radius; per path:
`flux_ratio` (ghost/primary flux on the map) and `peak_ratio` (maximum over
fields of ghost peak/primary peak; meaningful for bins about a pixel and many
rays per bin); both are 1 for the primary itself. Paths are ranked by `peak_ratio`; the total ghost map is the
sum of path maps.

Probes are traced with rayon in blocks of up to 16384; leaves are collected
in the fixed order (field, wavelength, base cell, depth first) and binned
sequentially with compensated sums, so maps are bit-identical between runs
and thread counts (tested with rayon pools of 1 and 5 threads).
The `JobContext` is checked before every block; progress is monotone and
ends at 1. Results carry `RunInfo` (backend `cpu-f64`, settings of the model
and of the analysis, no seed: nothing is random).

### Probe cache performance

Double Gauss, 45 two-reflection ghosts plus the primary, fields 0 and 2,
wavelength 1, overfill 1.5, 64² map, path maps kept; macOS arm64, release,
five rayon threads; median of seven runs after a warm-up
(`tests/bench.rs`):

| pupil grid | probes before | probes cached | before (ms) | cached (ms) |
|---|---:|---:|---:|---:|
| 20² | 578,242 | 305,537 | 192 | 123 |
| 48² | 1,908,374 | 946,211 | 439 | 273 |
| 128² | 8,909,468 | 4,070,403 | 1552 | 914 |

```sh
RAYON_NUM_THREADS=5 cargo test --release -p mklight-ghost --test bench -- --ignored --nocapture
```

Lattice coordinates round differently from the former recursive
`p ± half` arithmetic. Against it, on the Double Gauss (20², 48², 128²;
21² at overfill 1.3), Cooke triplet, Petzval and finite relay, totals,
per-path and per-field flux, peaks, ratios, centroids, RMS radii and every
map bin (relative to the map's peak) agree within 1e-12 relative; dyadic
grids are bit-identical. The exception is a probe exactly on the sampled
rim: at overfill 1 such points lie on the entrance-pupil edge, where an
ulp decides whether a ray passes. Double Gauss at 20², overfill 1, changed
by up to 1e-6 relative in on-map flux, peaks and map bins, and by up to 2e-4
in a weak field's off-map flux and RMS radius.

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

## 6. Resident GPU jobs (P9)

`mklight_ghost::gpu` exposes explicit backend-neutral `ghost_irradiance`,
`paraxial_ghosts`, `paraxial_batch` and `real_ghost_focus` jobs. They take a
`ComputeBackend` and `JobContext`, and preserve the reference functions and
result shapes. Engine routes, automatic selection and chain qualification
remain with P11; these adapters do not advertise an engine capability.

`GhostTransport` prepares immutable, checked expanded step programs, the
exact `GhostModel` clips and P7 coating domains for each wavelength and
incidence side. It accepts explicit hi/lo start probes for same-sample
transport and combined conservation checks. `ProbeResult` borrows completed
shared memory through a pinned view. Raw probes retain their status, step
count, hi/lo final position/direction, separate s/p weights and bulk
transmittance; conditioning failures retain a reference-required status.
The numeric records and version are in `mklight_ghost_ops_types.h`.

Irradiance starts use P1 device chief aiming and source construction. The
sampled square is an entrance-pupil disc about that chief's aim point, as
in §5; individual overfilled ghost rays are not re-aimed at the stop.
Lattice coordinates stay integer keys alongside their resident samples.
The host deduplicates and stably sorts new keys and schedules refinement
breadth first across paths, fields and wavelengths. Centres and corners
share cached probes; the original depth-first leaf order is reconstructed
from the same split rule. The source boundary, virtual losses, on/off-map
regions, repeated wavelengths and leaf counts keep their reference meaning.

Geometry reuses P1 local-frame intersection, transfer and interaction hooks,
with double-float geometry. Ghost transport separately implements physical
virtual-segment rejection, inverse backward ideal-lens power, sensor
reflection/detection and the expanded DOE visit/order metadata. It batches
all incidences through P7's direct coating service, then multiplies the
s and p channels separately and applies signed Beer–Lambert absorption.
There is no independently interpolated coating table or second path
enumerator. Probes flagged by P7 are completed with the reference and counted.

Completed probes remain resident. Gathered leaves carry their stable IDs,
field/path map segment and quadrature area. One batched map chain computes
pixel keys, uses P3 stable sorting and scan to find occupied segments,
reduces those segments with P3's fixed compensated tree, and scatters into
dense maps. Empty bins never require a large reduction scratch plane.
Only the occupied-segment count and maximum length cross a host control
boundary; there are no floating-point atomics. Host code assembles the
small statistics, ratios, coverage and peak ranking in reference order.
Inputs are filled directly in shared buffers and outputs are read in place.

Point binning is discontinuous at pixel edges. Hits within the 2e-5 mm
geometry uncertainty band of a pixel edge receive a counted f64 reference
probe and retain that reference bin decision, including f64 rounding at an
exact edge. This matters for a perfect image on a bin boundary: passing the
geometry budget alone does not qualify its map. `RunInfo.settings.gpu`
records boundary probes, observed pixel migrations, coating/reference
completion counts. Reference values are filled into sparse shared records;
unaffected resident probes are reused without copying their batch.
Cancellation checks precede every reference-completion block; cancellation
never starts a replacement CPU job.

Probe tiles bound coating/query storage to about two million step rows,
with at most 65536 probes per plan. Geometry, coatings, gathers, scans and
reductions use bounded command buffers. Progress reserves ranges for
preparation, refinement, maps and final assembly, and reaches one only
after a successful result. `RunInfo` retains backend/device/build/probe,
precision, material identity, operation version, settings, stage timings,
command counts, memory and fallback counters. Widening public numbers to
f64 does not relabel device arithmetic as f64.

`paraxial_batch` evaluates marginal/chief recurrences across path/wavelength
batches in double-float. Small path enumeration, constitutive metadata,
normal-incidence axial throughput and final ranking remain host work.
`real_ghost_focus` shares P1's fully aimed meridional launch across paths,
then runs the same ghost sequence geometry without unused coating work.

Explicit jobs currently accept uploaded immutable systems with up to 16
wavelengths and default reference intersection settings. Other wavelength
shapes, custom intersection settings and resident candidate systems return
`Unsupported` for the caller's reported reference route. Qualification for
new designs, candidate envelopes and automatic cost decisions belongs to
P11. Tests cover all built-in samples, matched CPU-f32/Metal algebra, both
shader-library origins, physical fixtures, fixed-hit map conservation,
pixel-edge reference handling, concurrent determinism and cancellation.
Provisional timings and raw qualification evidence live in the package's
`P9-evidence.md` handoff, outside the repository.

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
