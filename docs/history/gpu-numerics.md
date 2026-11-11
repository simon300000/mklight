# GPU numerics: the measurements behind two standing decisions

Records of finished measurements, kept because the reference documents
cite them as the reason for a present design. Section references, names and
numbers are those of the time of writing. The pure-f32 kernel build and its
harnesses (`f32_study`, `MKLIGHT_PURE_F32`, `MKLIGHT_METAL_PRECISION`) are
no longer in the tree, so the first record cannot be reproduced as written.

## Pure f32 kernels: measured cost and accuracy (GPU.md §3.6)

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

## Steep random rays on extended surfaces: residual classification differences (GPU.md §5.1.1)

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
