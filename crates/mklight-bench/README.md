# mklight-bench

Measurement and qualification of the engine. Nothing here is linked into a
product; the engine does not depend on this crate.

| | |
|---|---|
| `src/` | what the harnesses share: the result comparator and its contract tables (`comparison`, GPU.md §9.4), the allowance of the solver-outcome checks (`solver_spread`), the solver jobs' variables and settings (`solver`), the unique tiny edits of a session (`edit`), load average and CPU time (`host`), and `require_metal` |
| `tests/` | two regression suites that need the comparator, which run with `cargo test`, and eight harnesses, which are `#[ignore]` and run by hand |
| `examples/` | seven command-line measurement programs |

This file is the one list of the commands and of the environment variables
each harness reads. The variables the engine and the Metal backend read in
any process are in GPU.md §10.

## Regression suites

    cargo test -p mklight-bench

| Test file | Checks |
|---|---|
| `contract` | the comparator: every declared result field has a quantity class, the classes are at least as strict as the CONVENTIONS §7 budgets, and the class rules give known answers |
| `solver_aiming` | the solver jobs on the CPU-f32 emulation with each `solver_aiming` route option against the CPU reference; reads no variable |

`solver_aiming` has one ignored test, the same check on Metal:

    cargo test -p mklight-bench --test tests solver_aiming:: -- --ignored --test-threads=1

## Harnesses

All are ignored tests, run in release builds. The ones that record Metal's
device time take `--test-threads=1` (it is one counter per process); the
ones marked *Metal* need a real Metal device.

### `metal_accuracy` (Metal)

Every GPU-capable analysis and the solver outcomes on the 16 samples at
default settings, Metal against the CPU f64 reference under the contract.
Reads no variable.

    cargo test --release -p mklight-bench --test tests metal_accuracy:: -- --ignored --nocapture --test-threads=1

### `cost_calibration` (Metal)

`calibrate` measures and fits the table of the `auto` cost model (GPU.md
§9.2); `validate` compares `auto` with both backends.

    MKLIGHT_BENCH_PROJECT=<local project file> \
    MKLIGHT_COST_CALIBRATION_OUT=$PWD/crates/mklight-engine/src/analysis/cost/apple-m2-pro.json \
    cargo test --release -p mklight-bench --test tests -- --ignored --exact cost_calibration::calibrate --nocapture --test-threads=1

    cargo test --release -p mklight-bench --test tests -- --ignored --exact cost_calibration::validate --nocapture --test-threads=1

| Variable | Meaning |
|---|---|
| `MKLIGHT_COST_CALIBRATION_OUT` | file the fitted table is written to (default: the target tmp dir); refused for a partial table |
| `MKLIGHT_BENCH_PROJECT` | a local project, measured as the design `local` |
| `MKLIGHT_COST_DESIGNS` | sample names and `local`, comma-separated (default nine samples and `local`); `all` for the 16 samples and `local` |
| `MKLIGHT_COST_FILTER` | analyses, comma-separated (default every analysis with a GPU path); fits a partial table |
| `MKLIGHT_COST_MODES` | of `cpu`, `cpu_edit`, `metal`, `metal_edit` (default all); fits a partial table when measuring |
| `MKLIGHT_COST_REPS` | timed repetitions per rung (default 2; `validate`: 3) |
| `MKLIGHT_COST_RECORDS` | file of the measured records, JSON lines (default: the target tmp dir); with `_REFIT`, one or more files to read, comma-separated |
| `MKLIGHT_COST_REFIT=1` | fits the records without measuring |
| `MKLIGHT_COST_DEVICE_OPEN_MS` | with `_REFIT`: the device-open time to store (default 40) |

### `project_bench` (Metal)

Warm, edited and cold-process timings of every analysis on a local project
on the CPU reference, Metal and `auto`, with the accuracy of the first
results under the contract; the edit loop of the app's graphs, the same
through the worker queue, and cancellation latency.

    MKLIGHT_BENCH_PROJECT=/path/project.mklight \
    cargo test --release -p mklight-bench --test tests project_bench:: -- --ignored --nocapture --test-threads=1

| Variable | Meaning |
|---|---|
| `MKLIGHT_BENCH_PROJECT` | the project (required) |
| `MKLIGHT_BENCH_AUTOSD_PROJECT` | a second project, measured as the variant `autosd` (geometric cases only) |
| `MKLIGHT_BENCH_FILTER` | case names, comma-separated; `edit_loop` and `edit_loop_queued` name those two measurements; `cancel_latency`, alone, runs the cancellation latency, which no other value does |
| `MKLIGHT_BENCH_RUNS` | timed rounds per block (default and minimum 10); cancels per case of `cancel_latency` |
| `MKLIGHT_BENCH_COLD_RUNS` | fresh-process rounds per case (default and minimum 3) |
| `MKLIGHT_BENCH_MIN_RUNS` | lowers the two minimums, for quick provisional passes |
| `MKLIGHT_BENCH_SKIP_COLD=1` | no fresh-process rounds |
| `MKLIGHT_BENCH_MODES` | of `cpu`, `df` (Metal), `auto`; `cpu` always runs, as the accuracy reference |
| `MKLIGHT_BENCH_PHASES` | of `warm`, `edit` |
| `MKLIGHT_BENCH_EDIT_GRAPHS` | analyses of the edit loop, comma-separated (default six) |
| `MKLIGHT_BENCH_CANCEL_CASES` | cases of `cancel_latency` (default `mtf,psf,glass_substitution,optimize,tolerance`) |
| `MKLIGHT_BENCH_QUIET_SECONDS` | longest wait for a quiet host before a block (default 30) |
| `MKLIGHT_BENCH_ALLOW_BUSY=1` | measure next to another build; every gate record is then provisional |
| `MKLIGHT_BENCH_GRAPH_BYTES` | the engine's analysis graph budget; 0 disables it |
| `MKLIGHT_BENCH_AIM_STOP` | `conditioned`, or the requested stop (CONVENTIONS §7) |
| `MKLIGHT_BENCH_TIER` | `preview`, or the design tier |
| `MKLIGHT_BENCH_CHILD`, `_CASE`, `_ANALYSIS`, `_SETTINGS`, `_MODE` | set by the harness for its fresh-process child; not for the command line |

### `cpu_trace_bench`

Throughput of the CPU f64 tracer's stages on a local project.

    MKLIGHT_BENCH_PROJECT=/path/project.mklight \
    cargo test --release -p mklight-bench --test tests cpu_trace_bench:: -- --ignored --nocapture

| Variable | Meaning |
|---|---|
| `MKLIGHT_BENCH_PROJECT` | the project (required) |
| `MKLIGHT_BENCH_RINGS` | hexapolar pupil rings (default 24) |
| `MKLIGHT_BENCH_REPEATS` | repetitions; the best time is reported (default 3) |
| `MKLIGHT_BENCH_CHUNK` | rays per `generate_many` / `trace_many` call (default 256) |
| `MKLIGHT_BENCH_AIM_STOP` | `conditioned` or `conditioned:preview`, or the requested stop |

### `aim_stop_qualify`

The conditioned aim stop and the preview tier against the CPU reference at
the requested tolerance, on the samples with real ray aiming and on
perturbed variants (CONVENTIONS §7).

    cargo test --release -p mklight-bench --test tests aim_stop_qualify:: -- --ignored --nocapture

| Variable | Meaning |
|---|---|
| `MKLIGHT_QUALIFY_METAL=1` | also the requested and the conditioned stop on Metal (*Metal*) |
| `MKLIGHT_QUALIFY_PREVIEW=1` | also records the preview tier's errors (not gated) |
| `MKLIGHT_QUALIFY_FILTER` | systems, comma-separated |
| `MKLIGHT_QUALIFY_ANALYSES` | analyses, comma-separated |
| `MKLIGHT_QUALIFY_PROJECT` | a local project, added under the name `project` |

### `buffer_identity`

Hashes of every result's JSON and buffers with wall and device time, to
compare two builds line by line, and each non-CPU mode against the CPU
result under the contract.

    cargo test --release -p mklight-bench --test tests buffer_identity:: -- --ignored --nocapture --test-threads=1

| Variable | Meaning |
|---|---|
| `MKLIGHT_IDENTITY_PROJECT` | a local project, added to the samples |
| `MKLIGHT_IDENTITY_SAMPLES=0` | skips the sample designs |
| `MKLIGHT_IDENTITY_MODES` | of `cpu`, `metal`, `auto` (default `cpu,metal`) |
| `MKLIGHT_IDENTITY_CASES` | JSON `[{name, analysis, settings}]` replacing the default cases |
| `MKLIGHT_IDENTITY_RUNS` | timed calls per case (default 1) |
| `MKLIGHT_IDENTITY_GRAPH_BYTES` | the engine's analysis graph budget; 0 disables it |

### `aim_precision_study` (Metal)

Cost and convergence of device aiming against the stop tolerance.

    cargo test --release -p mklight-bench --test tests aim_precision_study:: -- --ignored --nocapture

| Variable | Meaning |
|---|---|
| `MKLIGHT_STUDY_PROJECT` | a local project instead of three samples |
| `MKLIGHT_STUDY_TOLERANCES` | stop tolerances, comma-separated (default `1e-10,1e-8,1e-6,1e-5,1e-4`) |
| `MKLIGHT_STUDY_ROWS` | `all` or the number of field rows (default `all`) |
| `MKLIGHT_STUDY_REPEATS` | repetitions; the least device time is reported (default 3) |

### `seeded_jacobian`

Errors and times of the default merit's finite-difference Jacobians with
seeded and unseeded aims; prints a table and asserts nothing.

    cargo test --release -p mklight-bench --test tests seeded_jacobian:: -- --ignored --nocapture

| Variable | Meaning |
|---|---|
| `MKLIGHT_SEED_PROJECT` | a local project, added to the samples |

## Programs

They take arguments and read no variable; each file's header describes the
arguments and the output.

| Program | Measures |
|---|---|
| `backend_bench` | a suite of cases (JSON file) on the CPU reference, Metal, `auto` and route-option variants, each call after an edit |
| `fieldmaps_profile` | one analysis on one backend: wall and CPU time per call, with the backend's execution evidence |
| `solver_profile` | one solver job on one backend, with the resident plan stages |
| `stray_ghost_profile` | one analysis (stray light, ghosts) on one backend, with the stage timings of its run |
| `ghost_stages` | `ghost_irradiance` below the engine, the CPU reference and Metal alternated |
| `stray_stages` | `stray_light` below the engine: system build, primary reference and the CPU and Metal transports |
| `packet_bench` | the CPU packet path (`trace_rays`) inside `trace_grid`, `opd_grid` and a traced wave pupil, Cooke triplet and double Gauss (and an optional imported lens), on one thread and on all |

    cargo run --release -p mklight-bench --example backend_bench -- SUITE.json \
        [--filter TEXT] [--backends cpu,metal,auto] [--reps N] [--out FILE.jsonl] \
        [--label TEXT] [--no-compare] [--budget SECONDS] [--shuffle] [--idle MS]

    cargo run --release -p mklight-bench --example fieldmaps_profile -- LENS.mklight \
        ANALYSIS '{"settings": "json"}' [--backend cpu|metal] [--route KEY=VALUE]... \
        [--reps N] [--evidence] [--serial N]

    cargo run --release -p mklight-bench --example solver_profile -- LENS.mklight ANALYSIS \
        [--backend cpu|metal|auto] [--route KEY=VALUE]... [--settings JSON] [--reps N] [--json] \
        [--out FILE]

    cargo run --release -p mklight-bench --example stray_ghost_profile -- LENS.mklight \
        ANALYSIS BACKEND [SETTINGS_JSON] [REPS] [KEY=VALUE ...]

    cargo run --release -p mklight-bench --example ghost_stages -- LENS.mklight \
        [REPS] [COLUMNS] [SETTINGS_JSON]

    cargo run --release -p mklight-bench --example stray_stages -- LENS.mklight \
        [RAYS] [REPS] [SETTINGS_JSON] [COLUMNS] [KEY=VALUE ...]

    cargo run --release -p mklight-bench --example packet_bench -- \
        [RINGS] [REPEATS] [LENS.zmx|LENS.seq]
