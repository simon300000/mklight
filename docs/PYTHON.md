# MKLight Python bindings (`mklight-py`)

`import mklight` gives headless access to the engine: open and edit
designs, run registered analyses on the worker pool, and trace rays with the
CPU f64 reference tracer. The bindings are a pyo3 extension
(`crates/mklight-py`) over the engine facade `mklight-engine`, so Python, the
C ABI (`docs/FFI.md`) and the CLI share **one code path** for sessions,
snapshots, parameter edits, catalogs, analyses, jobs and backend selection.
Only the low-level tracing functions bypass the analysis registry; they call
`mklight-trace` directly on the session's resolved system.

Units and signs are those of `docs/CONVENTIONS.md`: mm, µm, degrees in
settings, normalised field `(Hx, Hy)` and pupil `(Px, Py)`.

## Building and installing

Requirements: a Rust toolchain (`source ~/.cargo/env`) and CPython ≥ 3.11.
The extension uses the stable ABI (`abi3-py311`), so one wheel serves every
CPython from 3.11 on (3.11 is the first version whose limited API includes
the buffer protocol, which the zero-copy arrays need).

Development install into a virtualenv inside the crate (ignored by git):

```sh
cd crates/mklight-py
python3 -m venv .venv
.venv/bin/pip install maturin pytest numpy
source .venv/bin/activate
maturin develop --release        # builds mklight/_mklight.abi3.so in place (editable)
python -m pytest                 # the Python test suite (tests/)
```

Wheels:

```sh
cd crates/mklight-py
maturin build --release -o ../../target/wheels
pip install ../../target/wheels/mklight-*-cp311-abi3-*.whl
```

`pyproject.toml` configures maturin: the pure-Python package lives in
`python/mklight` (re-exports, the default-engine helpers, `_mklight.pyi`
type stubs, `py.typed`) and the Rust extension is built as
`mklight._mklight`. maturin sets `PYO3_BUILD_EXTENSION_MODULE`, so the
extension does not link `libpython`. A plain `cargo build --workspace` /
`cargo test --workspace` builds the crate as an ordinary library linked
against the `python3` found on `PATH` (set `PYO3_PYTHON` to choose another),
so the Rust unit tests run without maturin. No Cargo feature is needed for
either build.

numpy is optional. With numpy installed, bulk arrays are numpy arrays;
without it (or after `mklight.set_numpy_enabled(False)`), nested lists.

## Quick tour

```python
import mklight

engine = mklight.Engine()                    # or mklight.new()/open() on a default engine
s = engine.new(sample="achromat")            # mklight.samples() lists the samples
mklight.paraxial(s)["efl"]                   # 100.0009...

sysd = s.system                              # the OpticalSystem JSON model as a dict
sid = sysd["surfaces"][3]["id"]
s.set_param({"param": "thickness", "surface": sid}, 94.5)   # returns the snapshot hash
s.save("achromat.mklight")

r = s.run("system_summary")                  # GIL released while waiting
r.data["surfaces"][1]["radius"], r.run_info["snapshot_hash"]
r.arrays["n_after"]                          # numpy float64 (surfaces, wavelengths)

job = s.submit("rng_stream", {"count": 10**6}, seed=7)
job.progress, job.state                      # 0.42, "running"
job.result(timeout=10).arrays["uniform"]

g = mklight.trace_grid(s, pupil=64)          # dict of (F, W, P, ...) numpy arrays
ray = mklight.trace(s, field=2, wavelength=None, px=0.0, py=1.0)
```

## API

Type stubs with docstrings: `crates/mklight-py/python/mklight/_mklight.pyi`.

### Module

| name | |
|---|---|
| `Engine(config=None, **kwargs)` | Engine configuration keys as in `docs/FFI.md` (`worker_threads`, `max_retained_jobs`, `resolved_cache_capacity`, `progress_interval_ms`, `app_name`, `app_version`, `default_backend`). Unknown keys raise `InvalidArgumentError`; `default_backend` takes the `backend` forms of `run`; `max_retained_jobs` defaults to unlimited (see *Job*). |
| `default_engine()`, `new(sample=, system=)`, `open(path)` | Convenience wrappers over a lazily created process-wide engine. |
| `samples()`, `sample(name)` | Built-in sample identifiers; a sample's system dict. |
| `trace`, `trace_grid`, `paraxial` | Low-level reference tracing (below). |
| `TRACE_STATUS` | Names of `trace_grid`'s status codes. |
| `set_numpy_enabled(bool)`, `numpy_enabled()` | numpy output switch. |
| `__version__`, `ENGINE_VERSION` | The engine version recorded in every `RunInfo`. |

### `Engine`

`new(sample=None, system=None)` → `Session`; `open(path)` → `Session`
(older project schemas are migrated; see `Session.info["migration"]`);
`analyses()` (name, title, category, `available`, settings schema,
`default_settings`); `backends()`; `jobs()` (status dicts); `cancel_all()`;
`config`; `default_backend`.

### `Session`

| member | |
|---|---|
| `system` | The `OpticalSystem` as a dict (fresh copy per access; `±∞` are the strings `"inf"`/`"-inf"`). Assigning calls `set_system`. |
| `set_system(dict)` | Validate and replace; `InvalidModelError` leaves the session unchanged. Returns the snapshot hash. |
| `set_param(path, value)` | One `ParamPath` edit, e.g. `{"param": "curvature", "surface": 2}` (`surface` is the stable id). `value` may be `"inf"` / `float("inf")`. |
| `set_params(edits)` | `[{"path": …, "value": …}, …]`, atomic. |
| `snapshot_hash`, `library_hash` | 64-hex content hashes. A result is current while both match. |
| `id`, `path` (`pathlib.Path` or None), `dirty`, `info` | Session state. |
| `save(path=None)` | Atomic save (to the session's path when omitted). |
| `load_catalog(path)`, `catalogs()`, `glasses(catalog=None)` | AGF (or `.json`) glass catalogs. |
| `import_prescription(path)` | Replace the system with a `.zmx` or `.seq` lens file, glasses resolved against the session's catalogs (load them first). Returns `{format, encoding, system, warnings}`; content that does not map is a warning (`docs/IMPORT.md`). |
| `set_project_section(section, value)` | `views`, `merit`, `variables`, `tolerances`, `metadata`. |
| `run(analysis, settings=None, seed=None, backend=None, progress=None, allow_backend_fallback=False, timeout=None)` | Run and wait → `Result`. |
| `submit(analysis, settings=None, seed=None, backend=None, allow_backend_fallback=False, view_key=None)` | Queue → `Job`. |

* `settings` is a partial dict; missing keys take the schema defaults and
  unknown keys raise `InvalidSettingsError`. The complete normalised settings
  are in `result.run_info["settings"]`.
* `seed` sets the analysis' `seed` setting. Deterministic analyses (no
  `seed` in their schema) reject it, as does a conflicting
  `settings["seed"]`. Equal seeds give bit-identical results (Philox,
  `docs/CONVENTIONS.md` §6); the seed is recorded in `run_info["seed"]`.
  It must be an integer in `0..2**64` (otherwise `InvalidSettingsError`).
* `backend`: `None` (engine default, initially `auto`, a preference), `"auto"` (engine automatic
  selection, consistent with the CLI), `"cpu"`,
  `"metal"` / `"gpu:metal"`, or `{"kind": "gpu", "name": "metal"}`.
  Semantics as in `docs/FFI.md` (*Backends*).
* Requests run at the design tier with the engine's aim-stop policy
  (`EngineConfig::aim_stop`, `conditioned` unless the engine configuration
  sets `aim_stop`); there is no per-request `tier`/`aim_stop` argument.
  `run_info["tier"]` and `run_info["aim_stop"]` record what ran
  (`docs/RESULTS.md` §1).
* `view_key`: a newer job with the same key in the same session supersedes
  (cancels) the older one; use one key per live view.
* `progress(fraction, stage)` is called on the calling thread, with
  monotone fractions, only when something changed, and with `1.0` at the
  end of a successful run.

### `Job`

`id`, `analysis`, `state` (`queued|running|done|failed|cancelled`, or
`evicted`, below), `progress` (monotone: 1.0 once a result was retrieved,
the last value seen for a failed or cancelled job), `stage`, `status()`
(the C ABI status dict, `None` after the result was retrieved), `done()`,
`stop()`, `cancel()`, `wait(timeout=None)` → bool, `result(timeout=None,
progress=None)` → `Result`. Timeouts are seconds; `None` or `inf` wait
forever; NaN or negative values raise `InvalidArgumentError`.

`stop()` requests an early `optimize`/`tolerance` result: `result()` returns
normally, the job ends `done`, and `result.data["stopped"]` is true. Queued
jobs still start; a finished job is unchanged. Other active jobs raise
`UnsupportedError`. `cancel()` (also after Stop) discards the result.

**Result retention.** The engine keeps a finished job's outcome until it is
collected. Its `max_retained_jobs` bound (256 in the C ABI, for front ends
that may never collect) would silently discard results a script still holds
a `Job` for, e.g. a sweep that submits 300 jobs before collecting any, so
Python engines default to **unlimited** retention: every job belongs to a
`Job` object (or a blocking `run`), and a `Job` that is dropped releases its
job. With an explicit `Engine(max_retained_jobs=n)` the oldest uncollected
outcomes beyond `n` are discarded; such a job reports `state == "evicted"`,
`done()` is true and `result()` raises `NotFoundError` naming the bound.

`result()` raises `CancelledError` (cancelled or superseded),
`JobTimeoutError` (the job keeps running), or the analysis' own error
class. The result can be fetched repeatedly. **Dropping a `Job` whose
result was never retrieved cancels it** and frees its slot in the engine.
For `asyncio`, use `await asyncio.to_thread(job.result)`.

### `Result`

| member | |
|---|---|
| `data` | The analysis result dict (also `result["key"]`). |
| `run_info` | Engine version, snapshot hash, analysis, normalised settings, backend, precision, seed, start time, duration, warnings. |
| `arrays` | `{name: ndarray}` (or nested lists). |
| `buffers()` | The underlying `mklight.Array` objects. |
| `analysis`, `snapshot_hash`, `library_hash`, `seed`, `warnings` | Shortcuts. |
| `envelope()` | The C ABI JSON envelope (`run_info`, `library_hash`, `backend`, `result_version`, `result`, buffer descriptors). |

### Arrays and zero copy

Engine buffers are wrapped in `mklight.Array`, which exports the Python
buffer protocol (format, shape and C-order strides; a Fortran-contiguous
request on an array that is not also Fortran-contiguous raises
`BufferError`, as the protocol requires). `numpy.asarray(array)`
therefore wraps the engine's memory **without copying**; the resulting
ndarrays are read-only (`arr.copy()` for a writable copy) and keep the data
alive. `Array` also has `name`, `dtype` (numpy name), `shape`, `ndim`,
`size`, `tolist()` and `flat(i)`.

### Low-level tracing

```python
mklight.trace(session, field=None, wavelength=None, px=0.0, py=0.0, records=True, clip=True)
mklight.trace_grid(session, fields=None, wavelengths=None, pupil=None, clip=True, progress=None)
mklight.paraxial(session, wavelength=None)
```

* `field`: a system field index, or a normalised `(hx, hy)` tuple (ints or
  floats: `(0, 1)` is the full +y field point, not fields 0 and 1).
  `wavelength`: a system wavelength index (`None`: primary). Negative or
  out-of-range indices raise `InvalidArgumentError`.
* `trace` returns `status` (`ok`, `miss`, `clipped`,
  `total_internal_reflection`, `failed`), `reason` (for failures), `ok`,
  `surface` (where the ray stopped; the image on success), `point`, `dir`
  (that surface's local frame), `opl` (mm, from the start of the trace),
  `transmittance` (bulk absorption only), `wavelength_index`,
  `wavelength_um` and, with `records=True`, per-surface arrays
  `point`, `dir`, `normal`, `global_point`, `global_dir` `(N, 3)` and
  `cos_incidence`, `opl`, `transmittance` `(N,)`. Record 0 is the starting
  ray (entrance frame for an object at infinity), record `i` surface `i`.
* `trace_grid` traces fields × wavelengths × pupil points in parallel.
  `fields`: `None` (all), an index, one `(hx, hy)` tuple (read exactly as
  `trace` reads it), or a **list** of indices and `(hx, hy)` pairs. A list
  is always a list of fields: `[0, 1]` is system fields 0 and 1,
  `(0, 1)` the point hx = 0, hy = 1, `[(0, 1)]` that point as a list;
  `wavelengths`:
  `None` (all), an index or a list; `pupil`: an int `n` (`n × n` grid
  clipped to the unit disk; default 16), a pattern dict
  (`{"type": "hexapolar", "rings": 6}`, `{"type": "fan_y", "n": 21}`, … as
  `mklight_trace::PupilSampling`) or `(P, 2)` points. Result arrays, with
  leading axes `(F, W, P)` in the order given:

  | key | shape | dtype |
  |---|---|---|
  | `point`, `dir` | `(F, W, P, 3)` | float64, image local frame on success |
  | `opl`, `transmittance` | `(F, W, P)` | float64 |
  | `status` | `(F, W, P)` | uint8, index into `TRACE_STATUS` |
  | `surface` | `(F, W, P)` | int32, stopping surface of failed rays, −1 for `ok` |
  | `pupil` | `(P, 2)` | float64 |
  | `field` | `(F, 2)` | float64, normalised `(hx, hy)` |
  | `wavelength_index`, `wavelength_um` | `(W,)` | int64 / float64 |

  The values are identical (bit for bit) to single `trace` calls; ray
  failures are reported in `status`, never raised.
* `paraxial` returns the `FirstOrder` data of `mklight-trace` at one
  wavelength (`efl`, `bfl`, `ffl`, `image_distance`, pupil positions and
  diameters, F-numbers, NAs, magnification, Lagrange invariant, total
  track, paraxial image height, `marginal`/`chief` ray tables) plus
  `wavelength_index`, `primary_wavelength` and `warnings`. Infinite values
  (telecentric pupils) are `float("inf")`.

The tracer for the current (snapshot, library) pair is built once and
cached in the session; any edit or catalog load invalidates it.

## Errors

```text
MKLightError(Exception)
├── InvalidArgumentError(MKLightError, ValueError)
│   └── InvalidSettingsError
├── ParseError(MKLightError, ValueError)
├── InvalidModelError(MKLightError, ValueError)
├── NotFoundError(MKLightError, LookupError)
├── FileError(MKLightError, OSError)
├── UnsupportedError(MKLightError, NotImplementedError)
├── CancelledError(MKLightError)
├── JobTimeoutError(MKLightError, TimeoutError)
├── JobFailedError(MKLightError)
├── ComputeError(MKLightError)
└── InternalError(MKLightError)
```

Engine errors are classified with the same table as the C ABI status codes
(`docs/FFI.md`): e.g. an unknown analysis or glass is `NotFoundError`, a
placeholder analysis or unavailable backend `UnsupportedError`, a trace or
numerical failure `ComputeError`.

## Threading, the GIL and Ctrl-C

* Every computation and every wait runs with the GIL released, so other
  Python threads keep running (tested). `Engine`, `Session`, `Job` and
  `Result` are thread-safe; calls on one session are serialised by its lock,
  which is never waited for while holding the GIL.
* Analyses run on the engine's worker pool against the snapshot current at
  submission, so editing a session never races with its running jobs.
* Waiting happens in 20 ms slices. Between slices the calling thread
  checks for signals and calls the progress callback. **Ctrl-C
  (`KeyboardInterrupt`) cancels the running job** (in `Session.run`,
  `Job.result` and `Job.wait`) or the running `trace_grid`, which then stops
  at its next cancellation check (bounded intervals), and the exception
  propagates. An exception raised by a progress callback cancels the same
  way.

## Tests

* Rust (`cargo test -p mklight-py`): error classification against the C ABI
  table, JSON conversion of infinities, buffer strides, argument parsing,
  `trace_grid` layout cross-checked ray by ray against
  `SequentialTracer::trace`, cancellation, seed merging, engine
  configuration validation, timeout → deadline mapping, Fortran
  contiguity, and job vs. synchronous analysis runs giving identical
  results.
* Python (`python -m pytest` in `crates/mklight-py` after `maturin
  develop`): every sample opens, validates and survives save/open with the
  same snapshot hash; edits, invalid models, catalogs; paraxial EFL of the
  achromat (100 mm) and F/5; `trace_grid` shapes, dtypes and bitwise
  agreement with single traces; known answers (on-axis focus, chief-ray
  height ≈ f·tan θ, chief-ray vertex crossings against `system_summary`);
  every runnable registered analysis (placeholders must raise
  `UnsupportedError`); monotone progress; cancellation, timeouts,
  supersession; seeded determinism; Ctrl-C cancellation of a running job and
  of a trace; zero-copy read-only arrays and the list fallback; GIL release
  during `trace_grid` and `run`; 300 uncollected jobs all keep their
  results (and an explicit retention bound reports `evicted`); NaN /
  negative / huge timeouts; progress monotone after a cancelled result;
  `trace` and `trace_grid` reading `(0, 1)` alike; Fortran-contiguous
  buffer requests refused; unknown engine keys and out-of-range seeds and
  indices raising `MKLightError` subclasses; afocal paraxial infinities as
  floats.

Automatic selection uses the same static capability policy and calibrated
cost model as the engine (RESULTS §5, GPU §9.2). Devices without a
calibration prefer qualified Metal; explicit Metal bypasses the model. The result
envelope retains `backend.execution` and `backend.cost_decision`, including
partial reference completion and actual precision. Public f64 buffer storage
does not make GPU arithmetic f64.
