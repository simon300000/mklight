# GPU work packages P0–P11

Moved verbatim from `docs/GPU.md`. A record of finished work; it does not
describe the present system (see [README.md](README.md)).

## From GPU.md §9.5, "Delivery and measurement" (the work-package plan)

### 9.5 Delivery and measurement

The task briefs and ownership ledger live in the task's `scratchpad/gpu/`
directory (`PLAN.md`). P0 is a serial interface gate; the **first parallel
wave is P1–P4**. Shared registration/layout files have one owner; subsequent
packages use separately registered module/header/shader files. Final engine
integration is deliberately last, so incomplete kernels never advertise
production capability.

| Package | Goal | Dependencies | Difficulty |
|---|---|---|---|
| P0 | Resident backend, buffer lifetimes, frozen interfaces and module seams | — | difficult |
| P1 | Device generation, aiming, reusable trace state and reference-sphere OPD | P0 | hardest |
| P2 | FFT, chirp/affine MFT and autocorrelation primitives | P0 | hardest |
| P3 | Deterministic segmented reductions, sort/scan and QR | P0 | difficult |
| P4 | Resident multi-system parameter/material/first-order preparation (removed 2026-11 with the resident solver design, its only user) | P0 | hardest |
| P5 | Geometric analyses, field maps, sizing and layout | P1, P3, P4 | difficult |
| P7 | Shared coating/radiometry and polarisation | P1, P3, P4 | hardest |
| P6 | Complete wave/image-simulation pipelines | P1, P2, P3, P5, P7 | hardest |
| P9 | Ghost path probes, refinement and irradiance | P1, P3, P7 | difficult |
| P10 | Stray transport, BVH, deterministic tallies | P1, P3, P7 | hardest |
| P8 | Full merit/FD, optimisation, tolerance and glass batches | P4, P5, P6, P7, P9, P10 | hardest |
| P11 | Engine routes, qualification, auto policy and release evidence | P5, P6, P7, P8, P9, P10 | difficult |

Every package supplies f64/appropriate emulation tests, actual-Metal tests,
determinism/cancellation/progress checks and before/after measurements on
`mklight_core::samples::all()` (16 current samples). Emulation-only or skipped
device tests cannot qualify a release. Run `source ~/.cargo/env` before
Cargo, `cargo test --workspace`, and preserve the reference tests. Benchmark
the unchanged CPU reference, old Metal path where present and new Metal
path on identical inputs; retain raw settings, seeds, hardware/build,
fallback counts, peak memory, cold/warm wall and device timings, and
median/p95 after warm-up. Use the existing Metal trace, wave interactive,
ghost/stray bench, optimiser resident_bench and engine glass benchmarks as
anchors. Include tiny/default/dense loads and end-to-end jobs; kernel-only
speedups cannot justify `auto`. P11 routes only qualified operation/domain
combinations and records the measured cost policy.

## From GPU.md §9.2, "Consolidated P11 measurements" (part before the QR policy, which stays in GPU.md)

#### Consolidated P11 measurements

The measurements and implementation descriptions in this subsection are the
preceding P11 baseline; the runtime policy is defined above and in §9.3–9.4.

Final release measurements: Apple M2 Pro (12 CPU / 19 GPU cores), macOS build `26A434`, 12 CPU workers; strict precompiled Metal `-fno-fast-math -fmetal-math-mode=safe -fmetal-math-fp32-functions=precise -ffp-contract=off`. The shader source digest and complete compiler/device/probe records accompany every native run.

The host passed a load/process idle gate before measurement; CPU/Metal calls alternate, with first calls and warm-up separated from ten measured pairs. Medians use both middle samples; p95 is nearest rank (the maximum of ten). The raw session manifest records executable hashes, source commit, environment, load, process inventory and sampled RSS. Earlier wave/fusion numbers and superseded P11 source epochs do not set this policy. Inherited literal `provisional` labels in some package harness headers are superseded by the enclosing final session manifest.

Representative matched calls follow; times are **median / p95 in ms**. The transform row has resident inputs; only complete engine job observations below set engine `auto`. All raw old-Metal comparisons, preparation/transfers, readback/packing, device times, copies, allocations, peaks and fallback counts remain in the package rows.

| Package and workload | CPU f64 | New Metal |
|---|---:|---:|
| P0 trace→reduce, Cooke 1M | 49.119 / 54.192 | 24.932 / 26.834 |
| P1 real aim, Double Gauss 821,904 | 218.084 / 225.569 | 56.700 / 58.409 |
| P2 skew MFT 1024→2048, resident input | 83.779 / 92.297 | 28.702 / 31.436 |
| P3 trace→moments, Cooke 1M | 59.180 / 67.776 | 19.640 / 19.886 |
| P5 Cooke footprint, 823,473 rays | 842.594 / 847.701 | 428.991 / 442.493 |
| P6 PSF lattice 1001, pupil 1024, output 2048 | 443.304 / 453.256 | 230.371 / 242.783 |
| P7 Double Gauss PRT, 1024 pupil | 343.401 / 362.102 | 233.677 / 239.567 |
| P8 complete LM, 4000×100 | 303.818 / 314.618 | 135.959 / 140.026 |
| P9 Double Gauss ghost maps, 20² | 119.509 / 127.044 | 93.359 / 97.297 |
| P9 Double Gauss ghost maps, 48² | 212.283 / 225.021 | 219.859 / 226.878 |
| P9 Double Gauss ghost maps, 128² | 590.736 / 634.644 | 758.109 / 819.120 |

These rows measure the P11 package paths. The analyses' CPU/GPU divisions
have changed since (§9.1, §9.2: host aiming and field groups, host-traced
wave pupils, grouped ghost paths with a CPU share); the candidate batches
(P8) and the stray transport (P10) now run the divisions of "Candidate
solvers" above and of STRAY.md. These package rows, and the P5–P9 analysis
rows, are kept as the P11 record; the current routing figures are under
"Automatic routing" ("Measured routing").

Where a predecessor exists, the same-input old/new complete calls are retained below. The old merit adapter had weaker candidate checking; this is a cost comparison, not a qualification of that predecessor.

| Matched predecessor | Old Metal ms median / p95 | Resident ms median / p95 |
|---|---:|---:|
| P0 Cooke 1M | 31.229 / 32.372 | 24.932 / 26.834 |
| P3 Cooke 1M moments | 42.634 / 45.511 | 19.640 / 19.886 |
| P5 Cooke dense footprint | 482.950 / 492.125 | 428.991 / 442.493 |

The P11 cold-start observations predate the runtime publication policy:
every cold Metal job then ran an f64 chain qualification, so they do not
describe current cold calls and are not repeated here. Cold device setup is
now a measured term of the routing prediction (above).

| Resident call | Peak live plan MiB | Reported warm copied bytes |
|---|---:|---:|
| Cooke footprint 823,473 | 172.501 | 0 |
| PSF lattice 1001 / 1024 / 2048 | 1358.524 | 0 |
| Double Gauss ghosts 128² | 278.099 | 0 |

These are resident-plan/storage counters, not a zero-copy claim for qualification or public owned results. The engine candidate rows report the separate real validation/owned-output copy totals, candidate IDs/counts and reference time; cold immutable system/operation caches and explicit readback bytes remain separate.

## From GPU.md §9.2, "Consolidated P11 measurements" (part after the QR policy)

**Qualification and limits.** Both precompiled and runtime-source libraries run the complete requested-result matrix. Default EE bin ambiguity, ill-conditioned on-axis Seidel/PRT axes and exact discrete/mask disagreements remain reported f64 domains. Those outputs are not counted as Metal results. No candidate envelope is assumed: changed systems receive all-surface geometry and complete operand f64 checks, and finalists are reference-scored. These checks and counted owned validation readbacks remain expensive; primary resident fill/read-view paths avoid staging copies, but the whole validation chain is not claimed zero-copy.

Qualified default maxima: `max_direction` = 4.21012e-08, `max_opd_waves` = 8.01771e-05, `max_position_mm` = 1.89671e-06, `maximum_image_energy_relative` = 4.80666e-07, `maximum_image_l1_relative` = 4.90827e-07. Undefined masks and discrete outputs agree exactly; known answers, conservation, CPU-f32 emulation, f64 cross-checks, cancellation and concurrent repeats are separate tests.

The stray device transport is the wavefront with source-local depth-first Philox state (per bounce, pop, closest-hit and interaction kernels over device-compacted lists), the only device transport: `cosplit` by default, `wavefront` as an option ("Host/device divisions of the analyses" above). The device writes detections, ledgers and continuation stacks into shared memory and the CPU forms the estimators; no tallies run on the device. Complete transport, not first-hit throughput, is what the routing cost model measures. Bounded P1 aiming calibration remains a performance limit; no wider precision or parameter envelope is claimed.

Peak sampled process RSS over the final tasks: 4.401 GiB. This is a one-second sample, not a process high-water guarantee. Per-plan live peaks exclude immutable caches, idle pools and host metadata; those counters remain separate. Owned final/validation copies and explicit readback are reported separately from in-place resident views; overlapping context/backend counters must not be summed.

The binding scratchpad `P11-evidence.md` and final `P11-quiet/` directory contain commands, complete before/after rows, source/executable identities, numerical evidence, cold/warm distributions, copies/peaks, command maxima and the scope/gap ledger. Routing does not use these numbers; it uses the calibration table of "Automatic routing".

Final reported warm command maxima: p7-radiometry.log: 38.548 ms, p7-dense-pupils.log: 43.250 ms, p9-ghosts.log: 16.540 ms. Dense film/PRT work calibrates above 16K items with a 32K tile cap; large full-record resident traces complete 128K-ray tiles. The cheap bare-coating path retains its larger fusion allowance. These controls change submission boundaries, not per-item arithmetic or RNG addressing.

P9 has one 16.540 ms warm outlier above the 10–15 ms aim; its other largest samples are about 13 ms. All remain below the 50 ms cancellation target.

## From GPU.md §9.2, the package briefs after "Aiming and OPD": pupil construction, "Multiple systems", "Transforms", "Collectives and RNG", "Radiometry and non-sequential reuse"

Keep hi+lo point/direction/OPL through pupil construction. Use the chief's
unclipped trace and the signed exit-pupil distance for the reference sphere;
support centroid references via the shared reduction, both sphere sheets,
telecentric limit, finite conjugates and negative propagation. Evaluate
`OPL_chief − OPL_ray` and the rationalised sphere correction of WAVE §3 in
double-float before conversion to waves. Reference and ray states must use
the same point convention: if exporting rounded geometry, apply the
CONVENTIONS §12 OPL correction once; internal unrounded geometry needs no such correction.
Lattice/rim probes retain stable IDs; `ProjectPupil` accepts a reference
descriptor so the wave consumer can build its centroid without another
trace. P1 exposes shared trace/normal hooks; consumers do not fork
`trace.metal`.

**Multiple systems.** Reuse `upload_systems` and ray `system_index`, extending
them to resident generated rays and parameter batches. A host-validated
parameter plan maps stable surface IDs/ParamPaths to numeric slots, with
bounds, configurations, pickup/solve dependencies and material choices.
Across B candidates, GPU kernels apply parameters, recompute indices,
frames, first-order/solve data and aperture/aim seeds; never reuse nominal
values invalidated by a perturbation. Discrete topology/unsupported solves
use the reference preparation path with a recorded reason. Batch by
compatible topology, path program, clipping and last surface, preserving
candidate/configuration/field/λ/pupil order. Source snapshots remain
immutable; transient candidate IDs refer to the base hash plus parameter
vectors, and published candidate systems receive their normal content hash.

**Transforms.** Implement portable complex f32 1-D/2-D FFT kernels (radices
2/3/5/7 and Bluestein for other lengths), tiled transpose, spectral products,
chirp-z/separable MFT and symmetric affine MFT. Specify signs, normalisation,
padding and bin order exactly as `wave/fft.rs`. Form large quadratic chirp
and defocus phases in double-float, reduce modulo a turn before f32 sin/cos.
Direct autocorrelation keeps the angular-spectrum sensor mapping, coverage
and interpolation of `wave/mtf.rs`; it is not interchangeable with an
interpolated FFT curve. Cache stencils across focus planes. Huygens uses
the same phase/sum primitives with distance tiles. No approximate transform
replaces the existing finite sum or discards sampling warnings.

**Collectives and RNG.** Extend §3.4's fixed tree to segments keyed by
candidate/field/λ/path/pixel. Chunk boundaries partition work, not the
mathematical reduction order. Use hi+lo totals, two-pass chief-relative
moments, stable radix sort/scan for EE and detector bins; never float
atomics. Zernike/large least-squares work uses deterministic QR/TSQR with
rank/condition reporting, not normal equations that square conditioning.
Reuse Philox4x32-10 and the core stream namespaces. Logical item IDs, not
thread/queue positions, address draws; preserve the reference draw order
for tolerance trials and a stray source ray's depth-first split tree.
Moving to a different RNG addressing scheme would require an explicit
version and reference implementation, not an unnoticed backend difference.

**Radiometry and non-sequential reuse.** One device coating service provides
both powers and complex flux-normalised amplitudes, keyed by wavelength,
incidence side/angle and material/stack. Batch TMM/table construction on GPU;
qualify interpolation including grazing/critical angles against f64 and
refine or use direct evaluation where needed. Preserve each solver's model:
scalar primary/stray products of averages, ghost separate s/p products,
and polarised PRT are different. Ghost transport adds virtual-segment,
sensor and DOE-visit rules to shared geometry. Stray uses portable analytic
intersections and a compute BVH on every Metal device; hardware RT is an
optional qualified spatial-search implementation, not a requirement or a
replacement of curved optics by triangles. Stable nearest-hit tie rules,
shared apertures and exact coverage signatures preserve the solver partition.

## From GPU.md §3.6, "Pure f32 variant: measured cost and accuracy" and "Selective mixed precision (studied, not adopted)"

### 3.6 Pure f32 variant: measured cost and accuracy

An opt-in measurement build answers what double-float costs and buys. With
`MKLIGHT_PURE_F32` (mklight_common.h) every double-float operation becomes
plain f32 with a zero lo part and compensated sums become plain sums; the
packed ABI, algorithms and strict flags (§3.2) are unchanged. `build.rs`
compiles it as a second embedded metallib (both variants' shaders compile
concurrently). `MetalOptions { precision: MetalPrecision::PureF32, .. }`, or
`MetalOptions::from_env()` with `MKLIGHT_METAL_PRECISION=f32`, selects it; the
precision descriptor (`opl_double_float`, `compensated_reductions` false) and
`kernel_build` (`pure_f32 …`) name the variant. `MetalOptions::default()`, and
so the engine's Metal provider (explicit or `auto`), is always double-float. CPU-f32 emulation stays the
double-float specification.

Measured 2026-11-02 on an Apple M2 Pro under normal host load (provisional;
median of ten alternating runs after a warm-up), Double Gauss trace, three
fields, 12 surfaces:

| Rays | DF device | f32 device | DF wall | f32 wall | CPU f64 wall |
|---|---:|---:|---:|---:|---:|
| 10⁴ | 0.39 ms | 0.13 ms | 1.08 ms | 0.62 ms | 0.94 ms |
| 10⁶ | 29.5 ms | 7.4 ms | 45.6 ms | 23.7 ms | 76.3 ms |

At 10⁶ rays f32 is **4.0× faster on the device and 1.9× end to end**; on the
16 sample designs' 64×64 all-surface trace grids the device ratio is 1.8–4.2×.
Real aiming gets no reliable speed-up (0.09–1.8× DF), because f32 changes
its convergence.

Accuracy against independent CPU f64 (§5.1 harness, all 16 designs, every
field and wavelength, 64×64 pupil, all-surface records; maximum error and,
in parentheses, the fraction of the full §9.4 budget):

| Quantity | Double-float | Pure f32 |
|---|---:|---:|
| Position, mm | 2.5e-6 (0.12×) | 7.2e-5 (3.6×) |
| Direction cosine | 4.2e-8 (0.02×) | 6.2e-7 (0.31×) |
| OPD, waves | 7.7e-5 (0.015×) | 0.21 (43×) |
| Per-surface OPL, mm | 3.3e-11 | 8.0e-5 |

Ray statuses agree in both. f32 exceeds the OPD budget on every design (6–43×)
and the position budget on the Newtonian. Real aiming at the default
tolerance 1e-10 fails 450 of 4,176 f32 aims (Cooke triplet, Double Gauss,
relay; none with DF), and f32 aims reported as converged miss the stop by up
to 2.4e-7 of its radius. The engine never runs it: its Metal provider always
opens the double-float build (`MetalBackend::new`), and
only test harnesses select the f32 build (`MetalOptions`). Forced through
the raw adapters, f32 PSFs are off by 8% L1 (41× budget), optical MTF by 0.016 (8×) and image
simulation by 80% L1.

**Conclusion.** Double-float stays the default and the only qualified mode.
Pure f32 is not qualified for OPD, wavefront and wave analyses or real
aiming, and does not hold the position budget on every design. Reproduce with the ignored tests
`cargo test --release -p mklight-metal --test f32_study -- --ignored --nocapture`
(trace, OPL, aiming, speed) and `-p mklight-engine --test f32_study` (engine
analyses), or `MKLIGHT_METAL_PRECISION=f32` with the trace benchmark (§5.3).

**Selective mixed precision (studied, not adopted).** The question whether an
f32 *predictor* phase of the aim solve (f32 traces and Jacobians until the
residual nears the f32 floor, then double-float residuals and corrections)
could cut the double-float work was measured with the ignored harness
`cargo test --release -p mklight-engine --test aim_precision_study -- --ignored --nocapture`
(`MKLIGHT_STUDY_PROJECT` points it at a local project; otherwise three
samples): 65×65 lattices of every field and wavelength aimed from the host
chief row with and without the wave path's 25×25 host seed table, with both
metallibs at stop tolerances from 1e-10 to 1e-4, every start ray retraced to
the stop in f64. On a 29-surface F/1.4 design (30 field rows, 96,270 rays,
stop at surface 15; Apple M2 Pro, indicative):

| arithmetic | tolerance | iterations / ray | aiming device ms | true residual (f64 retrace) |
|---|---:|---:|---:|---:|
| double-float | 1e-10 | 1.26 | 22.7 | 1e-10 |
| double-float | 1e-6 | 0.75 | 20.5 | 1e-6 |
| double-float | 1e-4 | 0 | 10.9 | 3.2e-5 (the seed's error) |
| pure f32 | 1e-6 | 0.75 | 11.2 | 1.6e-6 |
| pure f32 | 1e-4 | 0 | 6.0 | 3.2e-5 |
| pure f32 | 1e-10 | 13.4 (cap) | 1,584 | 0.74; 88 % of the rays fail |

Below its floor the pure-f32 build stops early: a solve asked for a stop
residual below `MKL_AIM_F32_FLOOR` (1e-6 of the stop radius) ends unconverged
once it reaches that floor, and the ray's schedule ends with it (later phases
would only reach the floor again). The ray stays a failed aim, reported as
before; tolerances at or above the floor (`f32_relaxed`, 1e-6) are unchanged.
On the lens this takes dense spot (23.6M rays) from 97.8 s to 2.0 s and dense
geometric MTF from 101 s to 3.5 s; the results remain accuracy failures.

One double-float stop trace in the aim kernel costs 113 ns per ray (7.5 ns
per ray-surface; the bulk resident trace: 5 ns), one more double-float
Newton iteration 98 ns; the same in f32 costs 62 ns and 72 ns: the aim kernel
is bound by its control flow and state, not by the arithmetic, so f32 buys
1.4× per iteration there against 2–4× in the bulk trace. The f32 floor on
this design is a stop residual of 1.6e-6 of the stop radius (3e-5 mm), 2.5
decades above the conditioned stop (6–10e-9) and four above the requested
one, so a mixed solve always needs the double-float entry evaluation plus at
least one double-float iteration after the f32 phase: 11.2 + 10.9 + ≈ 3 ms
against 22.7 ms for double-float alone. It loses; even a free f32 phase would
save at most one double-float iteration (≈ 9 ms of a 385 ms MTF). At the
engine level four decades of stop tolerance (conditioned, 1e-8, 1e-6) move
the MTF device time by less than 10 % (385 / 352 / 375 ms) and the optimize
device time not at all (663 / 632 / 642 ms). Chief-relative f32 geometry does
not help either: the f32 rounding of an offset from the chief ray is ε·|δp|,
and |δp| is 10–20 mm for a marginal ray at the front group of an F/1.4 lens;
the sag difference at the intersection and the OPL need double-float
regardless (CONVENTIONS §7). The trace kernel's existing split (§4.1: f32
decisions and predictors, double-float transfers, hit refinement, direction
renormalisation and OPL) is already the selective scheme; the double-float
work left per surface is the part that error propagation requires.

## From GPU.md §5.1.1, "Extended-surface qualification": residual classification differences of the steep random rays

Residual classification differences, as `(f64 → f32, count)`:

* Odd: OK→Miss 44, OK→NoConvergence 9, Miss→OK 26.
* Toroid: OK→Miss 4, Miss→OK 5.
* Biconic: OK→Miss 9, Miss→OK 8, OutsideDomain→OK 5,
  OutsideDomain→Miss 1.
* DOE base (the even-asphere path, §5 strong aspheres): Miss→NoConvergence
  12, Miss→OutsideDomain 1, NoConvergence→Miss 10,
  NoConvergence→OutsideDomain 1, OutsideDomain→NoConvergence 2. Both
  implementations fail all 26.
* x cylinder: OutsideDomain→Miss 1; both fail.

These are rays whose **reference classification is itself
ill-conditioned**, not a divergence of the seed or domain logic. The f32
and f64 conic seeds (from the same start) agree on every differing odd,
toroid, biconic and cylinder ray. For all but six of them the seed
misses, so the search starts at the closest approach and Newton walks
towards an extremum of the residual `g(t)` along a line that does not
cross the surface nearby; there `g′` falls to ~10⁻⁴ and the step `g/g′`
amplifies a relative perturbation of the iterate by ~10⁸, so where the
walk lands (a far root, a backtracking failure, the domain edge) is
decided by the last bits. Measured:

| fixture | f64 status changes under ±5e-7 mm / ±5e-8 input perturbations | f32 differences | of which such rays | f64 Newton from the f32-rounded start gives the f32 status |
|---|---:|---:|---:|---:|
| odd | 276 / 8192 | 79 | 63 | 33 |
| toroid | 24 / 8192 | 9 | 8 | 5 |
| biconic | 100 / 8192 | 23 | 15 | 12 |
| DOE base | 80 / 8192 | 26 | 23 | — |
| QCon, QBfs, XY, Zernike, cylinders | 0 / 8192 | 0 or 1 | — | — |

The perturbations (each input coordinate and direction cosine moved by
about the f32 rounding of the ray, twelve variants per ray) flip the f64
reference's own status for most differing rays; under ±1e-12 mm they
flip none of them, so the reference is deterministic but ill-conditioned
there. Running the reference algorithm in f64 from the f32-rounded
closest-approach point and direction already reproduces the f32 status
for 50 of the 111 odd, toroid and biconic differences. The rest are the
same walk decided by the f32 rounding of `g′` itself: the x-cylinder ray
stalls at an extremum (`g = −2.87 mm`) where the f32 `g′` rounds to
exactly zero (a `Miss`) while the f64 `g′ = −8e-8` steps out of the
domain (`OutsideDomain`). The f32-only hits are genuine roots (residual
≤ 2.7e-5 mm at the rounded output point, mostly gradient-amplified
coordinate rounding). No f32 search can reproduce these decisions;
qualification remains mandatory for steep systems, and the sample passes
above do not claim exhaustive qualification of every ray.

## From GPU.md §9.1 and §9.2, the planning baseline and the status of the P0 interface gate (sentences taken out of the running text)

§9.1, "Inventory and placement":

Source baseline: `mklight-engine/src/{gpu.rs,analysis/gpu.rs}`, the analysis
registry, `mklight-optim/src/{deriv.rs,tolerance.rs,merit/gpu_eval.rs}` and
the solver modules named below. Layout v5 and the existing double-float
trace, reduction and Philox specifications are reused.

At the planning baseline, the five ray-grid analyses and the optimiser evaluator accelerated
**tracing only**. Host f64 generated/aimed the rays and read/reduced the
results. P11 retires that engine analysis route and its fixed crossovers. FD evaluations already batch
their candidate traces; uncompensated tolerance trials do too. Glass
substitution's local refinements already share `GpuTraceEvaluator`
(`analysis/glass.rs`), although `GPU_ANALYSES` omits that registry name.
Jacobian assembly, candidate setup and the outer searches remain on the
host. Wave optics, field maps, layout, ghost irradiance, stray transport and
polarisation have no complete Metal path. These distinctions matter when
measuring the baseline: batching is being extended, not introduced anew.

§9.2, "Shared primitives and interfaces" and "Resident interface gate (P0)":

These are extensions of the existing trait, not a second backend API.
P0 freezes the following typed contracts before the parallel wave. Names
below are proposed Rust types; the C layout umbrella remains the authority.

The compiling API is `mklight_gpu::resident` and the nine package modules
in the table below. Engine dispatch is unchanged. Only the compatibility
`TraceUpload` and `Sum` operations execute in P0; all domain handlers
return `Unsupported` until their owners qualify them. The solver crates
(analysis, wave, ghost, stray, optim) depend on `mklight-gpu` and declare an
empty public `gpu` adapter module for their package to fill in.

The C header umbrella includes one empty
`include/mklight_<module>_types.h` per package for its numeric records; its
`MKL_*` defines join the generated Rust constants. Packed system layout v6
and all existing trace/reduction layouts and constants remain unchanged.

It sums every supplied value;
masking/compaction belongs to P3. The proof chain selects trace planes by
view offset, with no intermediate host array or upload. The legacy adapter
still uploads its initial host rays on each call; P1 owns resident generation.

Exact field names and types are in each `src/<module>/mod.rs`; these are
owned by the named package after P0.

P0 returns errors for unsupported/resource
failures and cancellation; fallback policy remains with the engine adapter.

§9.2, "Host/device divisions of the analyses", wave optics:

Host transforms beside the device's tied
this division, and device resampling, host aiming with device tracing
and a fully device-traced pipeline were 1.1×, 2.0× and 3.2× slower on
large cases.

§9.3, "`auto`, determinism and jobs":

The legacy host-generated Metal analysis bridge and its ray and
candidate-count heuristics are retired.
