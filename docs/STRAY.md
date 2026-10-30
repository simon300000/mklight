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
| `Ghost` | specular only, inside every clipping aperture, ≥ 2 *stray reflections* | explicit ghost solver (stray: comparison only) |
| `Primary` | specular only, every interface met in designed order with its designed interaction, 0 stray reflections | sequential tracer |
| `Sneak` | every other specular path (out of sequence with < 2 stray reflections, e.g. one reflection off a tilted window) | stray |

A *stray reflection* is a specular reflection, including total internal
reflection, at a refracting (or ideal-lens) interface. A mirror's
reflection is its designed interaction and does not count; its back side
absorbs. A surface that the sequential model visits twice (double-pass
elements) is one primitive with two passes, and the designed return pass
stays `Primary`.

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

`tests/partition.rs` checks class/coverage conservation, map integrals,
seed and thread-count reproducibility, and the high-reflectance ghost
cross-check. The tally's three-source-ray known-answer test checks covered,
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
