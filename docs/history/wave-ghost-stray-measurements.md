# Wave, ghost and stray adapter measurements and investigations

Moved verbatim from `docs/WAVE.md`, `docs/GHOST.md` and `docs/STRAY.md`. A
record of finished work; it does not describe the present system (see
[README.md](README.md)).

## From WAVE.md §9, "Host/device division": the alternatives the work-split study measured

The work-split study measured the alternatives on the M2 Pro: host
transforms beside the device's and a size rule for them tied this
division; device resampling of host lattices, host aiming with device
tracing and the fully device-traced pipeline were 1.1×, 2.0× and 3.2×
slower on large cases. The device pupil tracer below (`PupilTracer`)
remains for the solvers' MTF operands.

## From WAVE.md §9, "Qualification and timing"

### Qualification and timing

Host-required suites in `tests/p6_wave.rs`, `tests/p6_native.rs` and
`tests/p6_boundaries.rs` cover the
resident primitives and complete native results against matched f64 methods,
including automatic sampling. The ignored `all_gpu_traced_pupils` suite checks
all 144 built-in field/wavelength pupils, with actual P1/P7 work. The image unit
test checks forced halo tiles and whole-image convolution against independent
impulse deposition, including edge flux loss. Both shader origins and the
numerics probe are exercised. No skipped test qualifies Metal.

`tests/p6_bench.rs` measures complete native PSFs on all built-ins, explicit
64/256/1024 pupils through a 2048 image, dense representative designs,
11/101/1001-side traced lattices, complete/cached-grid image simulation, and
matched complete wavefront/MTF/field/focus/energy/line-spread methods.
It alternates unchanged CPU f64 and GPU runs, separates workload-cold/setup and
warm-up, and records ten measured runs, median/p95, stages, copies, allocations,
tracked peak plan bytes and fallback counts. The older prepared-pupil benchmark
remains a distinct residency case. Timings during concurrent package work are
provisional; engine routing uses the calibrated cost model of GPU.md §9.2,
not these timings. Detailed current
acceptance status, raw logs, numerical maxima and outstanding qualifications
are maintained in the task's `P6-evidence.md`.

On the qualified M2 Pro measurements (provisional), complete native jobs are
bound by command round trips (about 0.2 ms each), not arithmetic: after
batching, a default nine-pupil PSF job issues about 215 commands (it issued
about 790), most of them per-pupil placement, resampling and PSF steps. GPU now wins the
default singlet PSF, the 1024/2048 dense PSF and the million-ray lattice;
other small/default jobs, the Huygens/FFT focus cases and 64×64 or default
image simulation remain CPU-preferred because the f64 reference traces its
pupils in parallel in 10–50 ms. Dense 256/1024 transforms with an already
resident pupil favor GPU, a different cost state from tracing and resampling a
complete job; the cost model's stage terms keep that distinction (GPU.md
§9.2). These measurements do not change engine dispatch.

## From GHOST.md §5, "Probe cache performance" and "High-reflectance discrepancy: pupil quadrature, not transport"

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

## From GHOST.md §6, the plan count before and after the in-kernel coating evaluation

On a large test lens this replaced about 2,500 plans
and 12,000 command buffers (coating records of every step row written and
re-read) by 60 plans and about 1,000 command buffers, with bit-identical
results.

## From STRAY.md, "Divisions between the CPU and the GPU": the benchmark of the earlier all-device transport

With the earlier all-device transport, the provisional Apple M2 Pro
native-analysis benchmark (33 paired workloads,
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
