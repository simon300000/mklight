# Stray light (`mklight-stray`)

Non-sequential Monte-Carlo transport of light through the optical and
mechanical geometry of a system, with every detected path assigned to
exactly one *path class*, so that the sequential, explicit-ghost and
stray-light solvers can be added without counting any light twice.

Units follow CONVENTIONS §1 (mm, µm, W, W/mm²). Rays carry unpolarised
(s/p-averaged) flux.

## Modules

| module | contents |
|---|---|
| `bsdf` | Lambertian, Harvey–Shack ABg and Gaussian-microfacet scatter: evaluation, exact importance sampling, TIS tables, energy normalisation |
| `geom` | bounded sag patches, cylinders, placed primitives, the `RayCaster` interface and a CPU BVH |
| `scene` | the scene built from a `ResolvedSystem`: interfaces bounded by their clear apertures, mount zones, lens edges and flats, stop plates, barrels, baffles, detector |
| `source` | collimated (disk normal to the beam, or a disk in any plane such as the pupil plane), point and Lambertian-disk sources |
| `path` | path histories, the five classes, path signatures |
| `tally` | per-class maps, totals and image moments with Monte-Carlo standard errors, the exact energy ledger, signature histogram |
| `tracer` | the thread-count-independent path tracer (rare-branch forcing, splitting, Russian roulette, weight windows) |
| `system` | connection to the sequential tracer: automatic semi-diameters, field and out-of-field sources |
| `reference` | the deterministic sequential expectation of the `Primary` class, with designed paths checked for obstruction by the scene |
| `analysis` | the engine entry point (`analyze`, `analyze_system`) |

## Connection to the sequential tracer

`StraySystem::new(resolved, &SystemOptions)` (or `StraySystem::build` with
a `JobContext`: cancellation inside the sizing traces, after the scene
build and per grid row of the obstruction check) builds a
`mklight_trace::SequentialTracer`, computes the automatic clear
semi-diameters with `SequentialTracer::semi_diameters_with` (the largest
traced ray height over all system fields, with their vignetting factors,
wavelengths and the pupil rim), and builds the scene with them
(`SceneOptions::clear_semi_diameters`). The radii of the stop and of the
optical surfaces are those of `mklight_ghost::clip_apertures` (clear
apertures, no margin), the ghost solver's own definition, so the two
solvers clip identically (see "Partition"):

| surface | clear semi-diameter used by the scene |
|---|---|
| fixed semi-diameter or explicit aperture | its own |
| `Auto` | traced maximum |
| `Auto` aperture stop | `max(paraxial stop radius, traced maximum) × (1 + 10⁻¹²)` (the freeze rule of `fix_semi_diameters`, so the stop plate never vignettes the design's own fields, with the ghost solver's guard factor) |
| `Auto` image surface without a sensor | traced maximum `× (1 + detector_margin)` (default 5 %) |
| value supplied in `SceneOptions::clear_semi_diameters` | the supplied value |

The non-clear ring out to the mechanical semi-diameter is an opaque mount
zone, the stop is an opaque plate (or a mount zone plus a ring on a
refracting surface), and the image surface is the detector.

`StraySystem::clipped_system()` is the resolved system with every automatic
semi-diameter frozen to the scene's clear aperture: the *clipped sequential
model*. Its unvignetted, transmitted rays are the paths the Monte Carlo
classes as `Primary`, **except those that non-sequential geometry
obstructs** (next section).

With paraxial ray aiming (`RayAiming::Off`) the off-axis beams cross the
stop slightly outside its paraxial radius, so the frozen stop is larger
than the paraxial one (sample singlet: 10.075 vs 10 mm) and on axis the
scene passes a slightly wider beam than the paraxial entrance pupil (+1.5 %
flux, +1.8 % RMS spot for the singlet). The scene is the physical system
here; with real aiming the two agree.

### Obstruction of the designed beam

The sequential model only sees its own surfaces. Geometry that is not on
the designed path can still block it: the back of a Newtonian's fold flat
and its mount in the incoming beam, a secondary mirror and spider, a
retaining ring or baffle reaching into an oblique beam, or the second
surface of an element whose surfaces cross inside their clear apertures.
`reference::primary_reference` follows every designed path segment by
segment (source → first interface → … → detector, through the sequential
hit points) with the scene's ray caster, and a segment whose first hit is
not the next designed surface is *obstructed*. The reference reports

* `flux` (and the moments): unobstructed designed paths, the expectation of
  the Monte-Carlo `Primary` class;
* `sequential_flux`: every designed path, what a purely sequential solver
  reports; `obstruction() = 1 − flux / sequential_flux`.

`StraySystem::new` evaluates the obstruction of a pupil-filling beam for
every system field (`StraySystem::obstruction`, 48 × 48 grid) and warns
above 0.1 %; the analysis reports `primary_obstruction` for field sources
and warns likewise. Measured: Newtonian 11.8 % (fold flat back and mount),
the Double Gauss *without its vignetting factors* at 14° 5.0 % (crossing
surfaces of L1 and L6), the test fixture "Double Gauss in a barrel" at 10°
3.9 % (the front retaining ring clips the oblique beam); 0 for the samples
(the Double Gauss sample included).

The scene reports two surfaces of one element that **intersect inside their
clear apertures** (negative edge thickness at the smaller clear radius).
Rays near such a rim meet the second surface first, which the sequential
model ignores, so the `Primary` class departs from the sequential answer
there. The Double Gauss sample without its vignetting factors is such a
case: the unvignetted 14° beam sizes clear apertures of L1 (18.6 mm) and L6
(13.9 mm) beyond the radii where the element surfaces cross. The sample
itself vignettes its 10° and 14° fields (`vcy` = 0.15, 0.30) and is
physical over its clear apertures (tested).

### Sources

All margins are relative overfills of a radius, so the stop and the mounts,
never the source, limit the beam.

| source | geometry | strength | reference flux |
|---|---|---|---|
| field point, object at infinity (`StraySystem::field_source`) | collimated along the field direction, uniform over a disk of radius `r_EP (1 + margin)` on the paraxial entrance-pupil plane `z = z_EP`, centred on the chief ray's aim point | irradiance `E` normal to the beam; power `E π r² \|d_z\|` | `E π r_EP² \|d_z\|` |
| field point, finite object | point source at the object point, uniform in solid angle in a cone about the line to the chief aim point with `tan α = (1 + margin) tan α_EP` (`α_EP`: largest angle to the pupil rim; telecentric: chief direction and marginal slope) | intensity `I` (W/sr) | `2π I (1 − cos α_EP)` |
| out of field (`out_of_field_source`), e.g. the sun | collimated from `(θx, θy)`, uniform on a disk normal to the beam that covers the projection of the whole scene along the beam: centred on the centre of the bounding rectangle of all projected primitive bounding-box corners, radius `(1 + margin)` × their largest distance from it | irradiance `E` | `E π r_front² cos θ` (front mechanical aperture) |

A disk on the pupil plane is uniform in flux even for an oblique beam,
because its projection on the beam cross-section has the constant Jacobian
`|d·n|`. Every ray of the out-of-field beam that meets any geometry starts
inside its disk, including light entering a lens hood or telescope tube far
in front of the first vertex (`tests/system.rs`: a singlet behind a 40 mm
hood at 30° detects the same flux as a three times wider beam, z = +1.4,
while a disk sized from the front lens catches only 5 % of it). The price
is efficiency for long systems at large angles, where most of the disk
misses the scene (a missed ray costs one BVH query).

## Partition of detected light (no double counting)

Every Monte-Carlo path that reaches the detector gets one class from its
history (`path::History::classify`, first match wins):

| class | rule in the stray solver | owner |
|---|---|---|
| `Scatter` | at least one BSDF scattering event, on any surface | stray |
| `Mechanical` | no scatter, but a specular reflection off non-optical geometry (edges, flats, mounts, stop plates, barrels, baffles) | stray |
| `Sneak` | specular only, but crossed or reflected off an optical surface outside its sequential clipping aperture | stray |
| `Ghost` | specular only, inside every clipping aperture, an even number ≥ 2 of *stray reflections* in the sequential order they imply | explicit ghost solver for covered paths, otherwise stray |
| `Primary` | specular only, every interface met in designed order with its designed interaction, 0 stray reflections | sequential tracer |
| `Sneak` | every other specular path (out of sequence at any reflection order, e.g. one reflection off a tilted window) | stray |

A *stray reflection* is a specular reflection, including total internal
reflection, at a refracting (or ideal-lens) interface. A mirror's
reflection is its designed interaction and does not count; its back side
absorbs. A surface that the sequential model visits twice (double-pass
elements) is one primitive with two passes, and the designed return pass
stays `Primary`.

`History` checks the actual optical-surface walk as it happens, reversing
its expected order at each extra reflection. A skipped interface, an odd
number of reflections, or detection before the walk reaches the image is
`Sneak`, even if its reflection list matches a covered ghost. The list alone
cannot prove coverage. For a folded train with two indices on one physical
interface, a backward ghost leg selects the reverse of its expected pass;
this keeps signatures canonical without changing the designed primary path.
Tests cover a skipped interface with an otherwise covered reflection list
and the exact even/odd reflection sums in a double-pass plate.

Ownership, precisely:

* **Sequential tracer (`Primary`)**: the designed path, transmitted (or, at
  mirrors, reflected) at every optical surface in sequence, inside every
  clear aperture, **and not obstructed by other geometry**. Its flux is
  `τ · Π T_i` with the unpolarised coating transmittance
  `T_i = ResolvedCoating::rt(...)` (`R` at mirrors) and the internal
  transmittance `τ`. A purely sequential solver cannot see obstruction and
  overcounts by the obstructed flux (the obstructed light is absorbed, or
  scattered into the stray classes, by the Monte Carlo). When
  `StraySystem::obstruction` / `primary_obstruction` is not negligible, use
  the obstruction-checked `primary_reference(...).flux` (or the Monte-Carlo
  `Primary`) as the `Primary` term; scaling a sequential result by
  `1 − obstruction` is exact for the flux of the same source, not for maps.
* **Explicit ghost solver (`Ghost`)**: deterministic, purely specular paths
  through the optical interfaces with at least two reflections at
  refracting interfaces, all inside the clear (clipping) apertures, never
  touching mechanical geometry. Each ghost is identified by its ordered
  `(surface, transmit | reflect)` signature over the interfaces it meets.
* **Stray solver (`Scatter`, `Mechanical`, `Sneak`)**: everything else that
  reaches the detector. It may also report `Ghost` and `Primary`, for
  comparison only.

When both solvers run, set `StrayAnalysisSettings::exclude_ghost_class`
and pass the ghost run's `GhostCoverage` as `ghost_coverage` (or use
`ClassMask::STRAY_WITH_GHOST` as `StraySettings::owned_classes` with the
coverage's paths as `StraySettings::covered_ghosts`): the stray total and
maps then hold `Scatter + Mechanical + Sneak` plus the `Ghost` paths the
ghost run did not trace (`EnergyReport::detected_uncovered_ghost`: orders
above its `max_reflections`, paths it filtered out; a path whose signature
overflowed is never covered, so a coverage with more reflections than a
signature keeps is refused):

```text
detected = Primary (sequential) + Ghost (ghost solver) + stray total
```

The ownership decision is made at each detection, before accumulating a
source ray's split children. `ChunkTally` adds uncovered ghosts to the owned
pixel, flux and image-moment slots. Covered and uncovered ghost fluxes have
separate per-source-ray second moments (`detected_covered_ghost_std_error`,
`detected_uncovered_ghost_std_error`); their errors and the owned error retain
all split-path covariance. No sum of signature errors is used.

The analysis's `Ghost` map row contains **only uncovered ghosts** when
coverage is supplied with exclusion. All map rows sum to `total_irradiance`,
which integrates to `stray_flux`. The diagnostic `totals` still contains the
whole Ghost class; its `included` flag means the whole class is selected,
not that no subset can be owned. `classes` likewise describes whole-class
selection. Strongest signatures follow per-path ownership.
The engine adapter uses this same API; `stray_light` result version 2 removes
`mapped_stray_flux`, `mapped_stray_std_error` and `ambiguous_ghost_flux`.
A coverage longer than the stored signature is rejected at the boundary,
rather than claiming ambiguous ownership.

The identity holds path by path with the obstruction-checked `Primary`. An
explicit ghost solver that ignores non-sequential geometry overcounts
obstructed ghost paths in the same way.

This needs both solvers to clip at the same apertures (CONVENTIONS §10):
the scene's clear apertures come from `mklight_ghost::clip_apertures`, and
`StraySystem::ghost_apertures` reports the scene's clipping in the ghost
solver's terms. `StraySystem::check_ghost_coverage(&coverage)` (or
`ghost_coverage` in the analysis settings) refuses a ghost run whose
`GhostCoverage` has different interface kinds or apertures, e.g. a
`ClearApertures` margin or other semi-diameter settings. The comparison
is bit-exact (a ray at the rim must be clipped by both or neither), so
two independent sizings must agree to the last bit; with non-default
`semi_diameters`, build the ghost model from the scene's sizing,
`GhostModel::with_semi_diameters(tracer, settings,
system.semi_diameters.clone())` (`GhostModel::sized` sizes with given
settings, cancellably).
`tests/partition.rs` checks the sum on a singlet and a Cooke triplet
coated with an ideal 5 % reflector (orders 2 and 4 in the ghost run,
on-axis and off-axis fields): the stray `Ghost` class matches the ghost
solver within |z| < 1.2, the remainder of the detected flux equals ghost
solver + stray total within |z| < 1.2, and counting the `Ghost` class in
both solvers is off by z < −30. A ghost run to 2 reflections on the Cooke
triplet with 30 % reflectors leaves a quarter of the ghost light to higher
orders: with its coverage the stray total keeps them and the partition
closes (|z| < 1), without it (the whole class dropped) it misses them by
z ≈ 33. The exclusion changes only the tallies, not the random streams, so
a run with and without it differs by exactly the covered `Ghost` flux
(tested).

### Path signatures

`path::Signature` stores the ordered *non-designed* events of a path (the
first six): `R5` stray reflection at surface 5, `TIR5`, `Sr5` / `St5`
scatter into the reflected / transmitted side of surface 5, `C5` outside
surface 5's clipping aperture, `M[label]` / `Ms[label]` specular / scatter
off a non-optical primitive (named by its scene label, e.g. `barrel 0`),
`…` after six events, `designed` for none. Designed transmissions between
the events are implied, so a ghost's signature is the list of its
reflecting surfaces, and `Signature::expand_ghost(sequence, is_mirror)`
turns it into the ghost solver's full ordered `(surface, op)` signature
(for example `R2 R1` on a plate → `T1 R2 R1 T2`). `StrayResult::signatures`
lists flux, standard error and hits per (class, signature), largest first;
the analysis reports the top `top_signatures` of the included classes.

## Estimators and errors

Source ray `i` uses Philox stream `(seed, i)` for its whole path tree;
chunk tallies are folded in chunk order with compensated sums, so results
are bit-identical for any thread count. Each detected quantity is a sum of
independent per-ray contributions `xᵢ`, with standard error
`SE = √(N/(N−1) (Σxᵢ² − (Σxᵢ)²/N))`. This is reported for every class, the
owned (stray) total, every pixel of every class map and of the owned sum
map, every path signature, and the net Monte-Carlo weight adjustment.

**Image moments** per class (and for the owned sum): from the sums
`S = (F, Σwx, Σwy, Σw(x²+y²))` over detector hits (local detector frame)
and the 4 × 4 matrix `Q` of per-ray cross products, the centroid
`(Sx/F, Sy/F)` and RMS radius about it get delta-method errors
`SE(g)² = N/(N−1) (∇gᵀQ∇g − (∇gᵀS)²/N)`. A test checks them (and the flux
error) against the scatter of 16 independent runs (ratio 1.05–1.11).

**Energy ledger**: every emitted unit ends detected (one class), absorbed
(by sink), escaped, escaped from a solid (a leak; warned about), truncated
or lost to a sampling failure, and

```text
emitted + mc_adjustment = detected + absorbed + escaped + escaped_from_solid + truncated + failed
```

holds to rounding. `mc_adjustment` (net Russian-roulette and reweighting
change) has expectation zero; `EnergyReport::mc_adjustment_std_error` is its
error bar, so *Σ classes = emitted − absorbed − escaped* is tested within
the Monte-Carlo error.

`tests/partition.rs` checks class/coverage conservation, map integrals
and the high-reflectance ghost cross-check. The tally's three-source-ray known-answer test checks covered,
uncovered and owned pixel/flux errors with correlated split children.

## Cross-validation (tests)

`tests/sequential.rs`, all scatter disabled (black, non-scattering edges,
no mechanics, no surface BSDFs), 60 000 rays, default (weighted) transport,
a distinct seed per case, `|z| < 3` required.
The reference is `reference::primary_reference`: an `n × n` grid over the
source's own sampling disk (exact cell weights, so only the clip and
obstruction boundaries inside the beam add quadrature error; checked to be
< 0.2 SE between `n = 250` and `360`), traced through the clipped
sequential model at every wavelength and checked for obstruction.
z = (MC − reference)/SE_MC:

| system, field | transmitted fraction MC / ref (z) | centroid y z | RMS radius MC / ref (z) |
|---|---|---|---|
| Double Gauss, 0° | 0.5336 / 0.5335 (+0.04) | −0.83 | 21.00 / 21.03 µm (−0.49) |
| Double Gauss, 10° (vignetted by its apertures) | 0.4579 / 0.4569 (+0.47) | −1.65 | 15.26 / 15.26 µm (+0.10) |
| Cooke triplet, 0° | 0.5857 / 0.5855 (+0.07) | −0.77 | 6.69 / 6.67 µm (+1.20) |
| Cooke triplet, 20° | 0.6051 / 0.6042 (+0.42) | −0.35 | 26.12 / 25.91 µm (+1.46) |
| Cooke triplet, Hy = −0.55 | 0.4946 / 0.4956 (−0.48) | −0.37 | 17.32 / 17.30 µm (+0.32) |
| Cooke, lossy AR coating + 0.02/mm glass absorption, 14° | 0.6742 / 0.6718 (+1.51) | +1.14 | 25.18 / 25.14 µm (+0.39) |
| 1:1 relay (point source), 0 mm | 0.6821 / 0.6822 (−0.05) | −1.04 | 3.15 / 3.14 µm (+0.52) |
| 1:1 relay, 5 mm | 0.6770 / 0.6769 (+0.06) | −1.01 | 45.37 / 45.34 µm (+0.34) |
| Newtonian (obstructed 11.8 %), 0.25° | 0.7280 / 0.7293 (−0.76) | −1.55 | 4.36 / 4.39 µm (−2.05) |
| Newtonian (obstructed 11.8 %), 0.35° | 0.7306 / 0.7295 (+0.60) | +0.59 | 6.17 / 6.18 µm (−0.46) |

Fractions are of the emitted (overfilled) power. For the refracting
systems nothing is obstructed and the reference equals the purely
sequential model exactly (asserted). Against the purely sequential flux the
Newtonian gives z = −54 and the unvignetted Double Gauss at 14° z = −9.7
(obstruction checked: −0.68).

The reference shares the source sampling with the Monte Carlo, so two
checks are independent of it:

* **Margin**: the reference flux in W at fixed irradiance agrees within
  `2·10⁻³` between margins 0.1 and 0.5 for the singlet, achromat, Cooke
  triplet, Double Gauss and Petzval at two fields each (observed ≤ 8·10⁻⁴);
  without a margin the pupil-plane disk underfills the stop and loses
  0.3–8.6 %.
* **Spot**: with real ray aiming, the reference's geometric (equal-weight)
  centroid and RMS radius match the sequential tracer's own spot over a
  121 × 121 pupil grid (`SequentialTracer::trace`) within 2 % of the RMS:
  centroids within 0.01 RMS, RMS radii within 0.3 % except the Cooke
  triplet (0.8 % on axis, 1.5 % at 20°: pupil aberration, since the beam is
  uniform on the entrance-pupil plane and the tracer's grid uniform on the
  stop).

Ghost class on an uncoated plate (n = 1.5, normal incidence,
`rare_floor = 0.25`): all orders 0.1187 vs `E π r² T²R²/(1−R²)` = 0.1160 W
(z = +0.70), second order vs `T²R²` (z = +0.69), 3.3 % relative SE. With
every branch split the second order matches `T²R²` to 1e-10
(`tests/window.rs`).

## Analysis entry point

`analysis::analyze(Arc<ResolvedSystem>, &StrayAnalysisSettings, &JobContext)`
(or `analyze_system` on a built `StraySystem`).

Settings (JSON, every field optional, unknown keys rejected, also inside
`semi_diameters`): `source` (`field {hx, hy, margin,
strength}`, `system_field {index, margin, strength}`, `out_of_field
{angle_x_deg, angle_y_deg, margin, irradiance}`, `custom {spec,
reference_flux}`), `rays`, `seed`, `classes` (bit mask, default all but
`Primary`), `exclude_ghost_class`, `detector_pixels`, `top_signatures`,
`ghost_coverage` (the coverage record of the ghost run combined with this
one; the analysis fails unless the scene clips at its apertures, and with
`exclude_ghost_class` the ghost paths outside it stay in the total),
`scene` (`SceneOptions`), `semi_diameters`, `detector_margin`, `transport`
(`mode`, `max_events`, `roulette_threshold`, `split_ceiling`,
`scatter_reflection`, `scatter_transmission`, `chunk_rays`). Validation:
the detector map is limited to `MAX_DETECTOR_PIXELS` = 2²⁰ pixels (about
0.5 kB of tallies and results per pixel), source strengths and powers must
be finite, a custom reference flux finite and ≥ 0.

`analyze_system` keeps the scene and detector its `StraySystem` was built
with; when the settings' scene, semi-diameter or detector options differ
from `StraySystem::options` the result carries a warning, and `RunInfo`
records the options actually used (`system_options`).

Result: the resolved source with its reference flux; one irradiance map
with per-pixel standard errors per included class; the total stray map and
its errors; stray flux, error and fraction of the reference flux (`None`
for a zero reference flux); for field sources the obstructed fraction of
the designed flux (`primary_obstruction`); per-class totals (flux, error,
fraction, hits, image moments) for all five classes;
the top path signatures; the energy report; the clear semi-diameters used;
`RunInfo` (snapshot hash, settings, resolved source, scene fingerprint,
seed, effective system options) and warnings. Progress (monotone): system
build 0–10 %, the source's obstruction check to 12 %, transport to 100 %;
cancellation between build stages, per grid row of the obstruction checks
and every 64 rays.

## Performance

`tests/bench.rs` (ignored; `cargo test --release -p mklight-stray --test
bench -- --ignored --nocapture`): Double Gauss (±10°) in a black barrel with
a front retaining ring, 36 primitives, 400 000 rays, 12 threads (Apple
Silicon): on-axis field 1.4·10⁶ rays/s (1.5·10⁷ events/s), sun at 20°
2.9·10⁶ rays/s (1.8·10⁷ events/s).

## Limitations

* Rays carry s/p-averaged flux and each interface uses the unpolarised
  `R`, `T`; the sequential reference uses the same product of averages.
  Polarisation-resolved transport is not modelled.
* The sequential reference supports collimated and point sources (not
  Lambertian disks).
* Lens edges follow the first surface's axis when an element is tilted or
  decentred (warned).
* GPU (hardware ray tracing) casters are not implemented yet; the CPU BVH
  is the reference `RayCaster`.

## Extended sag geometry and DOE transport

`geom::SagPatch` intersects odd asphere, QCon, QBfs, toroid, biconic,
XY-polynomial, Zernike sag and DOE bases through the core Cartesian
`surface_math::evaluate`. The implicit residual is `F=z−sag(x,y)` and its
ray derivative is `d_z−sag_x d_x−sag_y d_y`. The existing finite-cell scan
brackets sign changes and searches derivative sign changes for grazing root
pairs; safeguarded Newton/bisection refines the nearest valid positive root.
Normals are the core `normal`, toward local +z. A patch never extends past
the shape's real domain (the norm radius of Q, XY, Zernike and DOE phases,
the conic domains of both biconic axes, and for a toroid the y conic and
the x circle `1/|c_x|`); in the clip's thin pad beyond the patch radius the
residual continues the rim radially, so a root just inside the rim is
still bracketed. Cartesian sag/gradient sampling extends the radial z
bounds for freeforms. These sampled bounds and the finite scan are
numerical bounds, not interval proofs for arbitrarily oscillatory
prescriptions. Closed-form plane/conic intersections retain their
stable roots, including vertical conic rims.

A DOE is a physical interface even when its two indices match. After choosing
the coating's R/T/scatter channel, transport draws an integer order from the
shared scalar efficiency spectrum (SURFACES §3.1), then evaluates the same
reference `diffract` equation in the primitive's local frame. A scattered
channel samples its BSDF about that order's deflected direction. The power
of a selected order is encoded by its sampling probability, with no second
η multiplier. Coating absorption remains `1−R−T`; an evanescent order ends
in the separately reported `diffraction` sink, preserving the per-ray ledger.

DOE histories retain `(surface,visit,reflected,order,selected)` in encounter
order. A non-selected order changes a regular optical path from Primary to
Ghost, including zero-reflection ghosts; it does not by itself mean scatter.
Scatter, Mechanical and Sneak take precedence as before. When combined with
an explicit ghost run, `covered` compares both reflections and non-selected
order overrides. Selected-order reflection coverage cannot exclude an
uncovered diffraction path. Compact-signature overflow stays uncovered;
coverage with more than 32 DOE visits is rejected at transport input to
avoid claiming an exclusion the signature cannot express. The 32 visits
are counted over all DOE surfaces, separately from the six non-designed
events, and are stored in 8 bytes each (the visit number is recounted on
read) so path histories stay allocation-free `Copy` values on splitting.
The limit affects coverage and signature detail only: longer paths still
trace, overflow, and stay owned by stray. Thus

```
MC detected = sequential Primary + explicit covered Ghost
              + stray (Scatter + Mechanical + Sneak + uncovered Ghost)
```

within MC and deterministic pupil-quadrature error. The hybrid DOE test in
`tests/partition.rs` checks this identity and nonzero uncovered-order flux.
`tests/sequential.rs` compares Primary transmission, spot centroid and RMS
with the obstruction-checked sequential reference on all extended samples,
reporting z scores and checking quadrature convergence.

## Resident GPU transport (P10)

`gpu::{analyze, analyze_system, analyze_with_signatures}` return the native
analysis result, including aperture/coverage checks and the obstruction-checked
primary reference. The reference diagnostic remains CPU f64 and is distinct
from the transport backend.

`gpu::run` executes source generation, non-sequential transport and detector
estimators through `ComputeBackend`, returning the existing `StrayResult`.
`gpu::TransportScene` retains immutable scene/material leases across plans;
`append` exposes raw device detections and per-source ledgers for composed
jobs. `prepare_geometry`, `generate` and `first_hits` remain available for
geometry diagnostics. Engine capability and `auto` routing are in GPU.md
§9.2 (`stray_light` is routed by the cost model).

The four source measures preserve wavelength-first Philox draws and the
reference uniform conversion, including its lower half ulp. Every source
keeps its own 64-bit logical ID and 64-bit transport draw cursor. Splitting
applies branches in reference order, then traverses children depth first.
Other sources, command boundaries and stack growth never consume its random
stream. Rare-branch forcing, analog/weighted/split modes, roulette and the
64-copy weight window retain the reference bookkeeping.

The portable compute caster uses analytic planes, conics and cylinders,
local-frame double-float roots/normals and stable original-primitive-ID ties.
Extended sags reuse P1's packed bases and gradients. Their nearest-hit search
clips to a conservative box/cylinder, scans signed residuals and derivative
extrema, and refines brackets with safeguarded Newton/bisection; this also
resolves close root pairs near tangency. Coefficient bounds enclose the
whole sag domain, including asymmetric surfaces; sampled extrema do not
stand in for a bound. Small scenes retain the median host topology. At 512
primitives and above, device kernels transform bounds and build a balanced
preorder tree by levels. Traversal is stackless on every Metal device,
including devices without hardware ray tracing. Optics are never triangulated.

Interactions call the P7 coating implementation directly (the shader includes
P7's source and the CPU-f32 specification uses P7's module, so there is one
coating algorithm). Angle-dependent TMM
and powers run on device, with scalar products of unpolarised averages.
Beer--Lambert attenuation, all absorption/leak/escape/truncation/failure
sinks, backside rules, ideal lenses, repeated physical-interface passes,
TIS interpolation and Lambertian/Gaussian/ABg rejection sampling follow the
reference. TIS incidence and proposal log/exponential formulas retain
double-float residuals, including grazing angles and uniform endpoints.
DOE order draws retain the full ideal/kinoform/multilevel spectrum,
the curved-surface grating equation and evanescent sink. Histories preserve
all five classes, six non-designed events and 32 DOE visits. Coverage matches
exact reflection and non-selected order overrides on device; overflow stays
uncovered. Shared reference boundary checks reject unrepresentable coverage.

Immutable mode/material/shape features specialize the transport shader so
unused split, film, extended-sag and scatter state does not burden other
scenes. This changes no physical decisions. The work-split study key
`stray_kernel` selects how: `specialized` (default; every feature a
function constant, up to 108 pipelines), `partial` (split mode, thin films
and extended sags constants, 8 pipelines; analog/weighted, DOE presence and
order kind, scatter presence and BSDF kind uniform branches) or `uber` (one
pipeline, every feature a branch on the run's feature word). The kernel folds
the constant and the run's bits into one mask, so the designs share the
source and produce identical records. Every design compiles to the same
register-limited pipeline (384 threads per threadgroup on an M2 Pro).

Each source has a resident continuation stack, ledger, detection journal and
draw cursor. Up to 16,384 persistent lanes claim fresh sources and paused
sources from device queues (integer atomics only), so SIMD groups stay full
while long split trees finish; claim order never changes a source's records.
Several rounds share one command buffer. The lane event budget and rounds per
command adapt towards 15 ms of device time, and cancellation is checked
between commands. A plan holds as many 256-source numerical blocks as fit a
160 MiB scratch budget (`TransportOptions::source_tile` caps it), and the next
plan's transport overlaps host key enumeration of the current one.
A full stack pauses before the next event consumes any draws, grows into a
larger device allocation and resumes through a device spill copy of the paused
sources. Stack storage is capped at 256 MiB per allocation. `TransportOptions`
also controls initial stack capacity, detection-journal capacity and the
maximum lane event budget. Journal rows past a source's count are unspecified.
Exhausted journal/stack capacity or P7's reference-required precision domain
returns `Unsupported`; non-ideal DOE blazes beyond 2^20 also use reference
sampling so every rounded Cauchy proposal remains an exactly represented
integer. Ideal selected orders retain all i32 bits in the grating equation.
Unrepresentable numeric BSDF/DOE parameters and estimator-product power
scales also use the reported reference route.
The high-level adapter completes a labelled CPU-f64
replacement with a fallback count and reason. Cancellation returns
`Cancelled` and never starts replacement work. No partial GPU ledger is
published as a completed result.

Tallies use P3 source products and fixed source-block reductions. Moment
channels combine each source's depth-first children in double-float before
the one-record P3 source-product formula is fused into channel generation;
scalar sink/class estimators keep only (sum, sum²), and one plan normally
forms every estimator of a transport plan.
Sparse pixel and signature groups gather the
original device weights into P3 segments, sum each source's children, then
form squares and reduce source blocks. Owned, uncovered and covered ghost
quantities have separate source estimators. Integer pixel/signature/source
keys are enumerated on the host (in parallel 256-source blocks) for sparse
topology and labels; no optical weights or positions are reduced there. Final compact block summaries are
folded in canonical source order using the existing compensated result
assembler, which also derives scalar centroids, RMS radii and standard
errors. This keeps public maps, signature ordering and error definitions
compatible with the reference. There are no floating-point atomics.

Inputs are produced with `fill_buffer`; output and integer-key metadata are
borrowed with `read_view`. Device stack spill is an explicit device copy,
not an upload/readback of path state. RunInfo records the actual backend,
F32Gpu precision, seed, scene fingerprint, numeric recipe version, kernel
build/probe, stage timings, command/allocation/storage/copy statistics and
any reference replacement. Widening final values to f64 does not change the
reported computation precision. Preparation and transport use progress
subranges, and successful public assembly emits the final 1.

Owned harnesses cover all 16 sample geometries and full transports, both
shader origins, f64 and matching CPU-f32 comparisons, analytic plate and
enclosure energy, BSDF integrals, split-child covariance, exact coverage and
DOE overflow, concurrent/chunked jobs, cancellation/resource reuse and large
device scene builds. The actual-Metal harnesses in `tests/p10_support` are
macOS test targets of this crate; the paired benchmarks there are ignored by
default. Provisional busy-host timings do not establish an engine routing
threshold.

### Divisions between the CPU and the GPU

`gpu::Design` (work-split study key `stray`, `gpu::run_design`,
`gpu::analyze_design`) selects how a run divides the transport:

| design | device | CPU |
|---|---|---|
| `gpu-tally` | transport and detector estimators (above) | integer keys, final assembly |
| `cpu-tally` | transport of every source to completion | tally of the device records in place; continuation of sources whose stack or detection journal is full; retrace of sources that need reference arithmetic |
| `gpu-head` | the first `TransportOptions::head_events` (64) events of every source | as `cpu-tally`, plus the continuation of longer path trees |
| `wavefront` | as `gpu-head`, as a wavefront: per bounce, every live source advances by one event in three kernels (pop, closest hit, interaction) over device-compacted lists with indirect dispatch | as `gpu-head` |
| `cosplit` (default) | `wavefront` on a fixed share of the source chunks | the reference trace of the other chunks at the same time, and the `wavefront` work |

In every design but `gpu-tally` the device writes its detection journal,
ledgers and per-source stacks (handoff mode, `ops::Transport::stack_output`)
into shared memory and the CPU reads them in place. A source whose stack or
journal is full, or that reaches the head budget, stops on the device before
its next event with a consistent stack, ledger and draw cursor; the CPU tracer
continues it from there (`Philox::seek` to the device cursor), so the paths
and draws are those of the reference. Its estimator terms combine the device
ledger and the CPU continuation as one source. Chunks of `chunk_rays` sources
are tallied in parallel, each in source order, and folded in chunk order as in
the reference; a chunk that spans two plans continues in the next. Two plans
are in flight: the device transports plan `k + 1` while the CPU finishes plan
`k`. A capacity limit is no longer a reason for a CPU fallback.

Which part of a source runs where depends only on the source and on fixed
options, never on timing, so a seed gives bit-identical results. The device
share of `cosplit` (`TransportOptions::gpu_share`, else the study key
`stray_share`) is a fixed rule of the scene when not set: 0.2, and 0 with iteratively intersected sags or when
that share is below 32 768 sources (`gpu::hybrid::default_share`); device chunks come in units of up to one
plan, spread evenly. With a share of 0 `cosplit` is the reference bit for bit,
with a share of 1 it is `gpu-head`. The persistent and wavefront kernels
produce identical records. Options that move the split (head budget, stack and
journal capacity) change results only by the rounding of the part that moved
(f64 instead of double-float).

The hybrid analysis adapter builds the scene, starts the device transport and
then runs the obstruction checks of the system fields and the source's primary
reference on the CPU beside it (`StraySystem::build_scene`); the result,
warnings and progress are those of the sequential order.

The CPU share of `cosplit` runs in the global rayon pool, or in a pool of
its own size (study key `stray_threads`), beside the CPU completion of the
device plans; the records do not depend on it.

**Extended-sag intersection on the device** (`ops::Transport::sag_mode`,
study key `stray_sag`): `scan` is the double-float cell scan above;
`f32-scan` (default) evaluates the cells' end points in plain f32 first and skips a cell
only when the residual and its slope are clearly of one sign at both ends
(about a hundred times the f32 rounding), so every root is decided by the
same double-float cells and the records are those of `scan` (tested on the
aspheric, Q, odd-asphere and Zernike samples); `cpu-grazing` hands a ray whose
clipped interval needs more than 32 cells (oblique and grazing rays) back to
the CPU before the event (`Ledger::status` 6: the state goes back on the
stack, no draw consumed), and the CPU continues the source; `f32-grazing`
combines both.

**Even-asphere evaluation on the device** (bits 2-3 of
`ops::Transport::sag_mode`, study key `stray_asphere`): `jet` (default)
evaluates every extended sag by the jets of `surface_math.h`. On an even
asphere whose sag is the plain radial form (the stray packing's zero overlay)
the radial forms evaluate `S(u) = c·u/(1 + w) + P(u)`, `u = x² + y²`, with
the gradient `2S'(u)·(x, y)`: `radial-scan` only in the f32 pre-scan (the
records of `jet`); `radial` also in double-float, with `P` and `P'` from the
trace's double-float slope (`mkl_asphere_poly_dp_df` in `trace_shared.h`);
`radial-newton` as `radial`, with each bracketed root located in f32 and
polished by two double-float Newton steps as the trace refines an asphere
hit, and a sign change that is clear in f32 solved without the cell's
double-float end points. `radial` and `radial-newton` agree with `jet` to
double-float rounding, not bit for bit (no CPU specification). The trace's
`trace_slope` key packs `MKL_SURF_DF_SLOPE` into the stray surfaces too, but
the stray kernels never read it: their sag slope is double-float in every
mode. Measured on a quiet M2 Pro, wavefront transport: `radial-newton` takes
26 % off the device time of a 29-surface lens with two even aspheres (100k
rays), 46 % off a 37-surface zoom with 7 aspheres and 55 % off a 35-surface
ultra-wide (10k rays); the 12-thread f64 reference is still 3–5 times faster
per event there, so the default share on such lenses stays 0.

**CPU stages.** The CPU path overlaps the obstruction checks of the system
fields and the source's primary reference with the transport (all three are
parallel, none fills the cores alone), checks the system fields in parallel,
and traces chunks in batches of 16 per thread; results are unchanged.

Measured on a heavily loaded M2 Pro (provisional): the device's double-float
transport kernel is register-limited; the same kernel compiled as plain f32
runs about 13 times faster. The closest-hit search dominates scenes with
iteratively intersected sags: two even aspheres in a 29-surface lens make the
cast kernel 85 % of the wavefront's device time, and one grazing ray's sag
scan takes up to about 3 ms on the device. Device transport reaches about
4 M events/s on the Double Gauss and 1 M events/s on that lens, against about
16 M and 10 M events/s for the 12-thread reference, so only `cosplit` with a
small device share is faster than the reference (the Double Gauss at 400 000
rays: 413 against 633 ms with a share of 0.25).

The provisional Apple M2 Pro native-analysis benchmark (33 paired workloads,
median of three warm runs, busy shared host) still prefers the CPU f64 route
for 32 of them: Metal/CPU median wall time is 1.6–5.1× on the samples at 20k
and 1M rays, 3.3–3.5× on the split-heavy plate and 3.0–14.5× on the
36-primitive barrel (1.33/0.24 s at 400k rays on axis, 0.70/0.05 s at 20
degrees). Only
the Q-bfs sample at 20k rays is faster on Metal (66/79 ms). Command waits stay
below 46 ms. The limit is the transport kernel itself: about 3–5 M events/s,
against 12–20 M events/s for the multi-core reference. The same scenes cast
first hits at about 37 M rays/s, so the double-float interaction code, not
traversal or scheduling, dominates. These jobs include the primary-reference
and public-result work, use default stray ownership, and report zero
reference replacements and host-copied buffer bytes.
