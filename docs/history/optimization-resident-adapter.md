# The resident optimisation adapter (work package P8) and the solver work split

Moved verbatim from `docs/OPTIMIZATION.md`. A record of finished work; it
does not describe the present system (see [README.md](README.md)).

## From OPTIMIZATION.md, section "Resident optimisation adapter (P8)"

## Resident optimisation adapter (P8)

This section describes the adapter as P8 built it. Since the work-split
study the engine's candidate jobs run its device stages within the hybrid
division of "Host/device division of candidate batches" below; the P4
candidate template and reports of whole candidates to the reference
described here were removed.

`gpu::ResidentEvaluator` evaluates candidate batches on a device; the
engine's native candidate jobs use it, and their routing is in GPU.md §9.2. Construct it with a `GpuContext` containing
the base snapshot and backend, and the configured `BuiltinEvaluator` reference.
Use it in `OpticalProblem`, or wrap a driver with `ResidentEvaluator::run` to
retain actual-backend `RunInfo`, library identity, native solver
metadata and cumulative plan/copy/allocation/fallback counters. Any `Problem`
that returns the context from `gpu_context()` gets the resident tall QR.

The parametric path caches P4's immutable compiled template. It writes candidate
rows directly into shared buffers, prepares their systems on device, and uses
P1 generation/aiming and hi/lo tracing. Every candidate keeps its own current
field and first-order metadata. P5 forms reference-sphere OPD and chief-relative
coordinates. P3 supplies weighted two-pass spot moments and batched weighted
piston/tilt QR fits, including the full mirrored pupil on asymmetric candidates.
The final small field/spectral combinations and operand result packing remain
host work. A polychromatic piston/tilt reference spanning several wavelength
groups retains the existing host fit; monochromatic fits and wavefront RMS are
resident. Unresolved aims and preparation failures select the affected
candidate's reference values, without replacing neighbouring successful rows.
The current raw-ray batch domain has common topology and at most 16 wavelengths.
Pupil metadata and samples are filled directly.
Candidates prepare their probes in parallel and merge in candidate order, so
the plan is identical to a serial pass. A plan holds up to 1024 candidates
within an estimated-ray budget (786,432 for spot-type merits, 196,608 when OPD
and fit channels are needed), keeping measured peaks at 60 and 127 MiB.
Aggregate-only spot queries do not construct host ray arrays or run OPD kernels.
Aiming-table eligibility is decided per candidate, preserving the same policy
when a full asymmetric pupil is batched with a symmetric half pupil.

The evaluator routes fixed-sampling diffraction and geometric MTF through P6
and P5, preserving complex spectral sums, detector response and focus sign.
Within a candidate, every frequency of a focused pupil shares its FFT; all
field/wavelength probes share P6's trace batches. Ghost and stray operands use the
P9/P10 adapters and retain their native reference-completion reports. Their
current public adapters prepare uploaded systems per candidate; they do not
consume the P4 candidate sidecar. Native solver commands across different
candidates are not fused into one resident plan. RI uses its existing chief-ray
étendue definition: the initial radial visibility scans run through P1 in a
candidate batch, then the reference adaptive boundary search and quadrature
consume those decisions. This host completion is counted; `ri_visibility`
records consumed and blocked device lookups. There is no duplicate f64 scalar
comparison before publication; visibility and derivative agreement remain tested offline. The measured forward/central RI steps
are 1e-6/1e-5. P5's image-point irradiance has a different definition and is not
substituted for this merit.
Model scalars, unsupported paraxial quantities, and unimplemented
operand adapters retain reported host evaluation.

### Offline accuracy and final reports

Resident candidates publish device values directly: no candidate, combined-merit,
finite-difference or constraint value is re-evaluated on the reference to check
it. Device status masks, non-finite values, unsupported preparation/features
and unresolved aiming determine the actual fallbacks, counted separately from
model-only host work.

Final optimize reports score the start and selected best design in f64. Glass
reports score and rank kept finalists in f64. Tolerance reports the active
evaluator's case scores and statistics: rescoring every Monte Carlo trial in f64
would repeat the whole run on the CPU. Merit evaluation returns actual device operands. Cancellation
never starts final reference scoring. Existing f64 cross-validation, perturbed
candidate, known-answer, conservation and derivative tests remain mandatory.

### Derivatives, linear algebra and independent solves

`optim_ops` defines zero-copy double-float residual comparisons, stable
finite-difference combinations, candidate perturbations, matrix products,
weighted fit rows/errors and damped least-squares rows. Request shapes and
indices are checked at the resident boundary. Integer stencil errors produce
status values. The CPU-f32 specification follows the same arithmetic order as
Metal; products and command tiles never change the summation order.

`fd_jacobian` retains the existing forward/central/second-order one-sided
stencils, bound handling, pinned columns, failed-probe retry and evaluation
budgets; evaluator-specific FD steps are unchanged. Residual and Jacobian
arrays cross the public `Problem` interface as owned f64 arrays, and every
published operand value is widened to host f64 after device evaluation, so residual
targets/hinges, stencil differences and `Jᵀr` are formed on the host: a device
pass over host-owned values costs one fill of the same size plus a command and
measured slower at every size (FD differences 0.35–4.7 ms against well under
1 ms on the host up to 10000×100). The `Residuals`/`Difference` requests
remain for resident chains whose inputs already live on the device.

**P11 policy update:** the `m·n²` threshold and timings in the next
paragraph are historical. The resident tall QR now starts at `m·n ≥ 2¹⁶`
Jacobian elements, measured on the complete compression (`GPU.md` §9.2).
Engine `auto` routes candidate jobs by the cost model (`GPU.md` §9.2). The
following P8/P11 timings are historical and include per-candidate f64 checks.

The local solver hands its tall QR to P3 once the free-column work
`m·n²` reaches 2²⁴; Thin Q then stays resident for the second-order
correction's projections and only the small unpermuted factor is reduced on
the host for the existing serial trust-region secular equation. Below that
the host Householder QR is faster (complete LM solves, provisional M2 Pro:
400×100 host 29 ms against 72 ms resident; 4000×32 parity at 46 ms;
4000×100 host 320 ms against 154 ms). Rank-deficient factorisations receive a
counted reference factorisation. The `LmRows` request builds
`[sqrt(w) J; sqrt(lambda) D]` and its RHS for a batched P3 QR. Neither path
forms normal equations.

`gpu::scheduler::run` rendezvous-batches independent serial solvers: each active
worker supplies its next optical evaluation, then one coordinator combines
matching operand lists in stable worker order. Finished workers leave subsequent
rounds. It uses no time window or search pruning. Compensated tolerance trials
use this scheduler, preserving the 33-point MTF focus screen plus the current
start. Ordinary uncompensated parameter trials retain P4 preparation; random
irregularity orientations that change extra model fields use the counted
complete-system preparation route. Catalogue filtering,
trial RNG and population/search decisions retain the reference implementation.
The low-level perturbation and batched QR operations are available to composed
callers; trial RNG and independent workers' small local linear solves remain
host stages. Trial summaries stay on the host: they must equal the f64
reference summary, which costs 0.04 ms for 1000 trials and 0.9 ms for 100000,
while a device summary (P3 moments plus stable sort) measured 1.9 and 4.9 ms.

### Evidence and cost handoff

Host-required tests are `resident_algebra`, `resident_candidates`,
`resident_solvers`, `scheduler` and `resident_tolerance`. They include both shader origins and numerics
probes, exact emulation algebra, all 16 samples with perturbed spot/wavefront
merits, asymmetric half pupils, an analytic thick-lens derivative, one-sided
retry, LM solves on both sides of the tall-QR crossover, native mixed operands, cancellation, trial
replay, matched-method RI visibility with clipped pupils, and independent refinements. A fallback-only result cannot satisfy the
device-work assertions. Existing CPU reference tests remain in force.

The ignored `resident_bench` harness alternates unchanged f64 and the
resident path at B=1/16/256/1024, with cold, warm-up and ten measured runs.
Historical runs included candidate qualification and recorded transfers, device/wall time,
command counts, allocation reuse, peak plan storage and fallback proportions.
These measurements do not set engine routing (the cost model of `GPU.md`
§9.2 does). Exact numerical maxima, provisional median/p95 results, raw records
and unmeasured acceptance cases are in the task's `P8-evidence.md`.

The additional ignored `resident_workloads` harness measures complete synthetic LM
solves on both sides of the tall-QR crossover, 32 Cooke tolerance trials with/without
compensation, three built-in catalogue refinements, and hybrid RI at B=1/16.
These include qualification and f64 final scoring, and currently remain
CPU-preferred. `RESIDENT_BENCH_MERIT=wave` runs the default and dense wave
counterparts of `resident_bench`; its default is spot. Timing evidence remains
provisional.

Provisional M2 Pro measurements (2026-11-01), ten measured runs after warm-up,
show why small candidate jobs favour the CPU. For an exact-input warm
Cooke default spot evaluation at B=1, CPU / legacy Metal (an f32 evaluator, since removed) / resident medians were
0.21 / 1.14 / 5.1 ms (about 3.5 ms on a quieter run); at B=1024, 28 / 32 /
54 ms after parallel preparation and 1024-candidate spot plans (92 ms
before). These warm rows exclude new candidate qualification. Complete
changing-candidate workloads include it: 32 compensated Cooke trials took
7.8 ms CPU versus 127 ms resident, and three catalogue refinements took 1.7
versus 24 ms; per-candidate geometry qualification commands dominate both.
Plan peaks are 60 MiB (spot) and 127 MiB (dense wavefront). Cold costs, p95, copies, fallback proportions and raw records are in
`P8-evidence.md`; these numbers do not set a universal backend threshold.

## From OPTIMIZATION.md, section "Host/device division of candidate batches": the outcome of the work-split study and "Why the fully resident adapter was slow"

This is the outcome of the work-split study `solvers`. Its
alternatives were removed with the study key: the fully resident P1/P4
adapter (master's design: 3.3× the time of this division, up to 7× on
large cases), joint glass refinements through the rendezvous scheduler
(1.7×: lock-step rounds wait for the slowest solve), the CPU reference
with this division's driver (CPU-heavier), plain host/device batches
without speculation and always-pipelined batches (ties within noise). Of
the P8 adapter above, the division keeps the device stages (P1 aiming,
P3 moments and fits, P5 OPD); the P4 candidate template, the separate
first-order plan and candidates reported whole to the reference are gone.
The CPU thread layouts of the study `solver_threads` (two lanes, dedicated
pools) never beat one lane and were removed too.

**Why the fully resident adapter was slow.** On real-aimed lenses (the 29- and
35-surface benchmark lenses, real aiming at 1e-10) a few pupil rays of
every candidate cannot be aimed (vignetted rim rays at the outer fields:
12 of 570 on the 29-surface lens). The resident adapter reported a candidate
with any unresolved ray whole to the reference, so every candidate was
evaluated twice: on the device, where the unaimable rays ran their whole
capped schedule (32 iterations with line searches, 20–90 ms of device time
per plan, the slowest lane setting the command's duration) behind
continuation rounds that read the pending count back, and then completely
on the CPU. A 22-variable optimisation took 30–50 times as long as on the
CPU. Without real aiming (the Double Gauss) the cost was the fixed latency
of a plan per evaluation (3–4 ms: P4 system preparation, aiming rounds,
waits) and a second plan for the first-order operands, against 0.2 ms of
CPU work per candidate.
