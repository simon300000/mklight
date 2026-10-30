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
| `reference` | the deterministic sequential expectation of the `Primary` class |
| `analysis` | the engine entry point (`analyze`, `analyze_system`) |

## Connection to the sequential tracer

`StraySystem::new(resolved, &SystemOptions)` builds a
`mklight_trace::SequentialTracer`, computes the automatic clear
semi-diameters with `SequentialTracer::semi_diameters` (the largest traced
ray height over all system fields, wavelengths and the pupil rim), and
builds the scene with them (`SceneOptions::clear_semi_diameters`):

| surface | clear semi-diameter used by the scene |
|---|---|
| fixed semi-diameter or explicit aperture | its own |
| `Auto` | traced maximum |
| `Auto` aperture stop | `max(paraxial stop radius, traced maximum)` (the freeze rule of `fix_semi_diameters`, so the stop plate never vignettes the design's own fields) |
| `Auto` image surface without a sensor | traced maximum `× (1 + detector_margin)` (default 5 %) |
| value supplied in `SceneOptions::clear_semi_diameters` | the supplied value |

The non-clear ring out to the mechanical semi-diameter is an opaque mount
zone, the stop is an opaque plate (or a mount zone plus a ring on a
refracting surface), and the image surface is the detector.

`StraySystem::clipped_system()` is the resolved system with every automatic
semi-diameter frozen to the scene's clear aperture: the *clipped sequential
model*. Its unvignetted, transmitted rays are exactly the paths the Monte
Carlo classes as `Primary`.

The scene reports two surfaces of one element that **intersect inside their
clear apertures** (negative edge thickness at the smaller clear radius).
Rays near such a rim meet the second surface first, which the sequential
model ignores, so the `Primary` class departs from the sequential answer
there. The sample Double Gauss at its full ±14° field is such a case: its
automatic clear apertures of L1 (18.6 mm) and L6 (13.9 mm) exceed the radii
where the element surfaces cross. At ±10° it is physical.

### Sources

All margins are relative overfills of a radius, so the stop and the mounts,
never the source, limit the beam.

| source | geometry | strength | reference flux |
|---|---|---|---|
| field point, object at infinity (`StraySystem::field_source`) | collimated along the field direction, uniform over a disk of radius `r_EP (1 + margin)` on the paraxial entrance-pupil plane `z = z_EP`, centred on the chief ray's aim point | irradiance `E` normal to the beam; power `E π r² \|d_z\|` | `E π r_EP² \|d_z\|` |
| field point, finite object | point source at the object point, uniform in solid angle in a cone about the line to the chief aim point with `tan α = (1 + margin) tan α_EP` (`α_EP`: largest angle to the pupil rim; telecentric: chief direction and marginal slope) | intensity `I` (W/sr) | `2π I (1 − cos α_EP)` |
| out of field (`out_of_field_source`), e.g. the sun | collimated from `(θx, θy)`, uniform on a disk normal to the beam centred on the first vertex, radius `(1 + margin)` × the largest mechanical radius (surfaces, stop plates, barrels, baffles) | irradiance `E` | `E π r_front² cos θ` (front mechanical aperture) |

A disk on the pupil plane is uniform in flux even for an oblique beam,
because its projection on the beam cross-section has the constant Jacobian
`|d·n|`. Any ray that can enter the lens through its front opening crosses
the out-of-field disk.

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
  clear aperture. Its flux is `τ · Π T_i` with the unpolarised coating
  transmittance `T_i = ResolvedCoating::rt(...)` (`R` at mirrors) and the
  internal transmittance `τ`.
* **Explicit ghost solver (`Ghost`)**: deterministic, purely specular paths
  through the optical interfaces with at least two reflections at
  refracting interfaces, all inside the clear (clipping) apertures, never
  touching mechanical geometry. Each ghost is identified by its ordered
  `(surface, transmit | reflect)` signature over the interfaces it meets.
* **Stray solver (`Scatter`, `Mechanical`, `Sneak`)**: everything else that
  reaches the detector. It may also report `Ghost` and `Primary`, for
  comparison only.

When both solvers run, set `StrayAnalysisSettings::exclude_ghost_class`
(or use `ClassMask::STRAY_WITH_GHOST` as `StraySettings::owned_classes`):
the stray total and maps then hold `Scatter + Mechanical + Sneak` only, and

```text
detected = Primary (sequential) + Ghost (ghost solver) + stray total
```

holds path by path. The exclusion changes only the tallies, not the random
streams, so a run with and without it differs by exactly the `Ghost` flux
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

## Cross-validation (tests)

`tests/sequential.rs`, all scatter disabled (black, non-scattering edges,
no mechanics, no surface BSDFs), 60 000 rays, default (weighted) transport.
The reference is `reference::primary_reference`: an `n × n` grid over the
source's own sampling disk (exact cell weights, so only the clip boundary
inside the beam adds quadrature error; checked to be < 0.2 SE between
`n = 250` and `360`), traced through the clipped sequential model at every
wavelength. z = (MC − reference)/SE_MC:

| system, field | transmitted fraction MC / ref (z) | centroid y z | RMS radius MC / ref (z) |
|---|---|---|---|
| Double Gauss, 0° | 0.5675 / 0.5669 (+0.26) | −0.18 | 21.69 / 21.69 µm (−0.13) |
| Double Gauss, 10° | 0.5620 / 0.5601 (+0.88) | −1.27 | 18.35 / 18.39 µm (−0.45) |
| Cooke triplet, 0° | 0.5857 / 0.5855 (+0.07) | −0.77 | 6.69 / 6.67 µm (+1.20) |
| Cooke triplet, 20° | 0.6051 / 0.6042 (+0.42) | −0.35 | 26.12 / 25.91 µm (+1.46) |
| Cooke triplet, Hy = −0.55 | 0.4946 / 0.4956 (−0.48) | −0.37 | 17.32 / 17.30 µm (+0.32) |
| Cooke, lossy AR coating + 0.02/mm glass absorption, 14° | 0.6742 / 0.6718 (+1.51) | +1.14 | 25.18 / 25.14 µm (+0.39) |
| 1:1 relay (point source), 0 mm | 0.6821 / 0.6822 (−0.05) | −1.04 | 3.15 / 3.14 µm (+0.52) |
| 1:1 relay, 5 mm | 0.6770 / 0.6769 (+0.06) | −1.01 | 45.37 / 45.34 µm (+0.34) |

Fractions are of the emitted (overfilled) power. Ghost class on an
uncoated plate (n = 1.5, normal incidence, `rare_floor = 0.25`): all orders
0.1187 vs `E π r² T²R²/(1−R²)` = 0.1160 W (z = +0.70), second order vs
`T²R²` (z = +0.69), 3.3 % relative SE. With every branch split the second
order matches `T²R²` to 1e-10 (`tests/window.rs`).

## Analysis entry point

`analysis::analyze(Arc<ResolvedSystem>, &StrayAnalysisSettings, &JobContext)`
(or `analyze_system` on a built `StraySystem`).

Settings (JSON, every field optional): `source` (`field {hx, hy, margin,
strength}`, `system_field {index, margin, strength}`, `out_of_field
{angle_x_deg, angle_y_deg, margin, irradiance}`, `custom {spec,
reference_flux}`), `rays`, `seed`, `classes` (bit mask, default all but
`Primary`), `exclude_ghost_class`, `detector_pixels`, `top_signatures`,
`scene` (`SceneOptions`), `semi_diameters`, `detector_margin`, `transport`
(`mode`, `max_events`, `roulette_threshold`, `split_ceiling`,
`scatter_reflection`, `scatter_transmission`, `chunk_rays`).

Result: the resolved source with its reference flux; one irradiance map
with per-pixel standard errors per included class; the total stray map and
its errors; stray flux, error and fraction of the reference flux; per-class
totals (flux, error, fraction, hits, image moments) for all five classes;
the top path signatures; the energy report; the clear semi-diameters used;
`RunInfo` (snapshot hash, settings, resolved source, scene fingerprint,
seed) and warnings. Progress: sizing 0–10 %, transport 10–100 %, monotone;
cancellation every 64 rays.

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
