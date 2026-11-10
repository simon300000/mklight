# Geometric-analysis adapter measurements

Moved verbatim from `docs/ANALYSIS.md`. A record of finished work; it does
not describe the present system (see [README.md](README.md)).

## From ANALYSIS.md §6.1, "Host completion": the f32 asphere slope

A device aim cannot always certify the requested stop
tolerance. When the double-float trace still took the even-asphere slope in
f32, the evaluated stop hit on wide-angle aspheric designs jittered by 1e-9
to 1e-8 of the stop radius, above the 1e-10 the reference reaches: on a
35-surface ultra-wide a third of the device aims, chief rays among them,
ended unconverged, and the device spot diagram's vignetted fraction differed
from the reference by up to 0.77. With the slope in double-float (the hit
refined and its normal formed from `asphere_poly_dp_df`) the device aims
there converge except for the rays the reference cannot aim either. Every
device aim reported `MKL_AIM_FAILED` is still aimed again in f64 by the
reference tracer [...]

## From ANALYSIS.md §6.1, the P5 harnesses and their measurements

P5's owned harnesses are `tests/p5_geometry.rs` and the ignored
`tests/p5_bench.rs` in `mklight-analysis`. They compare complete public arrays
with f64 on all 16 samples; cover both shader origins, matched FTZ bits,
analytic ideal-lens/disk answers, P1→P5 chaining, concurrent jobs,
cancellation and resource release. The benchmark compares unchanged f64,
legacy trace-only Metal plus host statistics, and the resident chain at
matched settings. Numerical maxima, hardware, raw runs, cold preparation,
median/p95, copy/storage counters and any qualification gaps are retained
in the task's `scratchpad/gpu/P5-evidence.md`. Engine routing is the
calibrated cost model of GPU.md §9.2, not a fixed ray-count assumption.

On the measured M2 Pro, 113- and 9,845-ray spots and the default spots on
all 16 samples favor f64. At the capped 823,473-ray square-pupil setting,
resident spot medians were 23–34 ms versus 37–89 ms f64 across the six dense
representatives. Complete Cooke global footprints at that density measured
434 ms resident, 477 ms legacy Metal and 831 ms f64, with 173 MiB peak
resident-plan scratch (public output and host hull storage excluded).
Default interactive layouts and Cooke RI at 5²/33² fields still favor f64;
RI medians were 42/1819 ms resident versus 23/1441 ms f64. Each owned series
uses one warm run plus ten measured runs. These measurements do not install
a ray-count crossover or qualify changed candidates. The external evidence retains p95, raw settings/load snapshots,
explicit copy/fallback counts and the remaining acceptance gaps.
