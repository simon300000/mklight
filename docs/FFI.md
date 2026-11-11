# MKLight C ABI (`mklight-ffi`)

Native front ends (the macOS app first) drive the engine through a small C
ABI. Structured data crosses as UTF-8 JSON; bulk numeric results cross as
typed binary buffers. The header is `crates/mklight-ffi/include/mklight.h`
(hand-written, checked against the Rust constants by a unit test), and
`include/module.modulemap` exposes it to Swift as the module `CMKLight`.
`crates/mklight-ffi/tests/c/smoke.c` is compiled, linked against the static
library and run by `cargo test -p mklight-ffi`. It calls every function
except the two prescription imports and the two GPU-preparation functions,
which the Rust tests `tests/abi_1_3.rs` and `tests/gpu_preparation.rs` call.

## Building and linking

```sh
cargo build -p mklight-ffi --release   # → target/release/libmklight_ffi.a
```

Link `libmklight_ffi.a` plus the native libraries it needs
(`cargo rustc -p mklight-ffi --lib --crate-type staticlib -- --print
native-static-libs` lists them): on macOS the library contains the Metal
backend, so link `-framework Metal -framework Foundation -framework
CoreFoundation -lobjc -liconv` (`-lSystem` is implicit); on Linux
`-lpthread -ldl -lm`.

Swift: add `crates/mklight-ffi/include` to the import search paths (it holds
the module map), make the static library findable (`-L<dir>`), then
`import CMKLight`. The module map autolinks `mklight_ffi` and the Metal,
Foundation and CoreFoundation frameworks plus `objc` and `iconv`, so no
further linker flags are needed.

## Versioning

* `mk_abi_version()` returns `(major << 16) | minor`; the header has
  `MK_ABI_VERSION_MAJOR/MINOR` (currently 1.4). A front end must check that
  the runtime major equals the compiled major. Minor releases only **add**
  functions and JSON fields, so JSON readers must ignore unknown keys.
  1.1 added `mk_engine_analysis_info`, `mk_engine_set_default_backend`,
  `mk_session_run_analysis_json`,
  `mk_session_validate_json`, `mk_session_apply_solves`,
  `mk_session_autosave`, `mk_session_get_param`, and the analysis fields
  `result_schema`, `result_version`, `gpu_path` and envelope fields
  `backend`, `result_version`.
* 1.2 adds `mk_job_stop`: Stop retains the best-so-far optimisation or
  tolerancing result, with `result.stopped: true`.
* 1.3 adds `mk_import_prescription` and `mk_session_import_prescription`
  (lens prescription files, *Lens prescription import* below).
* 1.4 adds `mk_engine_prepare_gpu` and `mk_engine_gpu_preparation_json`
  (*GPU kernel preparation* below).
* Each analysis' result shape has its own `result_version`
  (`docs/RESULTS.md` §2).
* `mk_engine_version()` is the engine (crate) version recorded in every
  `RunInfo` and project file.
* Project files carry `schema_version` (currently 1). A file at any other
  version is refused (see *Project file*).

## Status codes and errors

Every fallible function returns an `MKStatus`. On failure, `mk_last_error()`
returns a human-readable message for **the calling thread**; it is `""` after
a successful call and stays valid until the next `mk_*` call on that thread.
It is never NULL.

| code | name | meaning |
|---|---|---|
| 0 | `MK_OK` | success |
| 1 | `MK_ERR_INVALID_ARGUMENT` | NULL where not allowed, invalid UTF-8, invalid settings or parameter edit, JSON of the wrong shape (an unknown key included) |
| 2 | `MK_ERR_PARSE` | text that is not JSON; a malformed project file or catalog |
| 3 | `MK_ERR_INVALID_MODEL` | the optical system fails validation |
| 4 | `MK_ERR_NOT_FOUND` | unknown job, analysis, glass or catalog |
| 5 | `MK_ERR_IO` | file-system error |
| 6 | `MK_ERR_UNSUPPORTED` | analysis or backend not available in this build |
| 7 | `MK_ERR_CANCELLED` | the job was cancelled or superseded |
| 8 | `MK_ERR_TIMEOUT` | `mk_job_wait` timed out |
| 9 | `MK_ERR_NOT_READY` | the job has not finished |
| 10 | `MK_ERR_JOB_FAILED` | the job ended with an internal error (a panic) |
| 11 | `MK_ERR_COMPUTE` | trace, numerical or backend failure |
| 12 | `MK_ERR_PANIC` | an internal panic was caught at the boundary |
| 13 | `MK_ERR_INTERNAL` | other internal error |

No panic ever unwinds into C: every entry point catches it and returns
`MK_ERR_PANIC`. The library stays usable afterwards.

**JSON input.** One rule for every JSON argument: text that is not JSON is
`MK_ERR_PARSE`; JSON of the wrong shape (not an object where one is
expected, a missing key, a value of the wrong type, a key the object does
not have) is `MK_ERR_INVALID_ARGUMENT`, and the message names the key. No
unknown key is ignored: a misspelt configuration or request key fails the
call instead of running with a default.

**One classification.** The engine gives every error a kind
(`mklight_engine::ErrorKind`), and each front end maps kinds: this table for
the C ABI, the exception classes of `docs/PYTHON.md`, the exit codes of
`docs/CLI.md`. A job that fails reports the same status as the synchronous
call that fails the same way (`mk_job_take_result`, below).

## Ownership

| what | allocated by | released with |
|---|---|---|
| `MKEngine*` | `mk_engine_create` | `mk_engine_destroy` |
| `MKSession*` | `mk_session_new` / `mk_session_open` | `mk_session_destroy` |
| `char*` returned through `char**` | the library | `mk_string_free` |
| `MKBuffer*` array + everything it points to | `mk_job_take_result`, `mk_session_run_analysis_json` | `mk_buffer_free(array, count)`, once |
| `mk_last_error()`, `mk_engine_version()` | the library | never (borrowed) |
| `stage` passed to the job callback | the library | never; valid only during the call |
| all input strings | the caller | the caller (copied when needed) |

* Out-parameters are set to NULL / 0 on failure.
* Optional ("nullable") parameters are marked in the header; all others must
  be non-NULL.
* Sessions and the engine can be destroyed in any order. The worker pool
  stops when the engine handle and every session are destroyed.
* Destroying a session does not cancel its jobs; their results remain
  retrievable through the engine (or are dropped by retention, below).

## Threading

* `MKEngine` is fully thread-safe.
* `MKSession` is thread-safe; calls on one session are serialised by an
  internal lock. Different sessions do not block each other.
* A handle must not be destroyed while another thread is using it.
* Analyses run on the engine's worker pool (`worker_threads`, default
  `min(cores, 4)`), each against the immutable snapshot that was current at
  submission, so editing never races with running jobs.

### Job callback

`mk_engine_set_job_callback(engine, cb, user_data)` installs one callback per
engine:

```c
void cb(void *user_data, uint64_t job_id, MKJobState state, double progress, const char *stage);
```

* It is invoked on **one dedicated engine thread** (`mklight-events`), one
  event at a time. It never runs on the thread that submitted, cancelled or
  released the job, and no engine or session lock is held during the call.
  Workers and the progress poller only queue events, so a slow callback
  delays later events but never stalls a job. Marshal to the main thread
  yourself (for example with `DispatchQueue.main.async`).
* Per job, events arrive in lifecycle order whichever threads submit,
  cancel or run it: `QUEUED`, then either `CANCELLED` (cancelled or
  superseded while queued) or `RUNNING` (repeated for progress updates)
  followed by exactly one of `DONE` | `FAILED` | `CANCELLED`. Progress
  updates are polled every `progress_interval_ms` and sent only when the
  fraction or stage changed. Progress is monotone.
* The callback may call any `mk_*` function, including session functions,
  `mk_job_wait` (even on the job it is being told about) and
  `mk_engine_set_job_callback`. The one exception is `mk_engine_destroy` on
  the engine whose callback is running.
* Events raised while no callback is installed are not kept.
* When `mk_engine_set_job_callback` returns (including when clearing it with
  `cb == NULL`), no other thread is still inside the previous callback, so
  its `user_data` can be released. `mk_engine_destroy` clears the callback
  the same way.
* `mk_job_wait` may return slightly before the terminal event has been
  delivered.

## Jobs

```
submit ──► QUEUED ──► RUNNING ──► DONE
              │           ├─────► FAILED
              └───────────┴─────► CANCELLED   (cancel, supersede, shutdown)
```

* `mk_session_submit_analysis(session, name, settings_json, view_key, &job)`
  validates the analysis name, availability, settings and backend **before**
  queueing, so those errors are returned directly (not as failed jobs). Job
  ids are non-zero and unique per engine.
* **Supersede:** a job submitted with a `view_key` cancels the previous job
  with the same key *in the same session* (its status reports
  `superseded_by`). Use one key per analysis window.
* `mk_job_cancel` sets the job's cancellation token. A queued job is
  cancelled immediately; a running job stops at its next check (analyses
  check at bounded intervals). A job whose token was set ends `CANCELLED`
  even if it finished its work.
* `mk_job_stop(engine, job)` requests cooperative Stop for the analyses
  listed with `stoppable: true` (`optimize`, `tolerance` and
  `glass_substitution`). A queued job still starts and returns its initial/best result;
  a running job returns the best completed work as `DONE` with
  `result.stopped: true`. Other active jobs return `MK_ERR_UNSUPPORTED`;
  finished jobs are unchanged. Cancel, supersession and shutdown still
  discard the result, including after a Stop request.
* `mk_job_wait(engine, job, timeout_ms, &state)`: `timeout_ms < 0` waits
  forever. Returns `MK_OK` with a terminal state or `MK_ERR_TIMEOUT` with
  the current state.
* `mk_job_take_result` returns the result once and forgets the job (see the
  header for the per-state status codes). A job that failed returns the
  status of its own error, the one `mk_session_run_analysis_json` returns
  for the same request (`MK_ERR_NOT_FOUND` for an unknown glass,
  `MK_ERR_COMPUTE` for a trace or numerical failure, …), with the message
  in `mk_last_error()`; `MK_ERR_JOB_FAILED` means the job panicked. The
  failed job is forgotten as well, so read its state (`MK_JOB_FAILED`)
  first where `MK_ERR_NOT_FOUND` must be told from an unknown job.
  `mk_job_release` forgets a job without taking the result (cancelling it
  if needed).
* Retention: at most `max_retained_jobs` finished jobs are kept; older ones
  are dropped with their results (`MK_ERR_NOT_FOUND` afterwards). Take or
  release results promptly.

## Buffers

```c
typedef struct MKBuffer {
    const char *name;   const void *data;   size_t len;
    MKDType dtype;      uint32_t ndim;      const size_t *shape;
    const void *owner;  /* private */
} MKBuffer;
```

`data` holds `len` elements (not bytes) of `dtype`, aligned for that type,
row-major with `shape[0..ndim]` (product = `len`). dtypes: `F64=1, F32=2,
I32=3, U32=4, I64=5, U64=6, U8=7`. The result JSON lists the buffers
(`buffers: [{index, name, dtype, shape}]`) in the same order as the array,
so they can be found by name. The whole array is freed with one
`mk_buffer_free(array, count)` call.

**Shared, read-only data.** `data` is the array the engine computed: no
copy is made on the way out, for a new result or for one the engine answers
from its analysis graph. The engine may keep the same array for later
identical requests, so two takes of the same result can return the same
`data` address, and the memory must never be written. Each buffer holds a
reference to the array in `owner` (private: do not read, change or copy it
into another struct that is freed separately). `data` therefore stays valid
until the buffer array is passed to `mk_buffer_free`, also after the
engine and the session are destroyed and after other arrays that share the
data are freed; the memory is released when its last holder lets go. Free
each returned array exactly once, unmodified.

## JSON reference

Units and signs follow `docs/CONVENTIONS.md` (mm, µm, degrees; curvature
stored, not radius). JSON cannot hold infinities: `±∞` are the strings
`"inf"` / `"-inf"` (object distance, radius of a plane). NaN is `null`.

### Engine configuration (`mk_engine_create`)

All keys optional:

```json
{ "worker_threads": 0, "max_retained_jobs": 256, "resolved_cache_capacity": 8,
  "analysis_graph_bytes": 268435456, "progress_interval_ms": 50,
  "app_name": "MKLight", "app_version": "1.0",
  "default_backend": {"kind": "auto"}, "aim_stop": "conditioned",
  "route": { "aiming": "host-table", "aiming_share": null,
             "fieldmaps": "host-chief", "stray_share": null,
             "stray_sag": "f32-scan", "solver_aiming": "by-batch" } }
```

The initial `default_backend` is `auto`; explicit CPU remains available.
`worker_threads: 0` picks `min(cores, 4)`. `analysis_graph_bytes` is the
analysis graph's byte budget (default 256 MiB, 0 disables it;
ARCHITECTURE "Analysis graph"). `aim_stop` is the aim-stop policy of
requests that name none (`conditioned` by default, or `requested`;
CONVENTIONS §7). `route` holds the route options of requests that carry
none (GPU §9.3: the keys, their values and what each moves to the GPU;
every key optional, an unknown key or name fails `mk_engine_create`). A key
the configuration does not have fails it too (`MK_ERR_INVALID_ARGUMENT`).
`app_name`/`app_version` are stamped into project metadata on save.

### Analyses (`mk_engine_list_analyses`)

```json
[{ "name": "glass_info", "title": "Glass data", "category": "materials",
   "description": "…", "available": true, "unavailable_reason": null,
   "settings_schema": { "type": "object", "properties": { … }, "required": ["name"] },
   "default_settings": null,
   "result_schema": { "type": "object", "properties": { … }, "required": [ … ],
                      "x-buffers": [ {"name": "dispersion_curve", "dtype": "f64",
                                      "shape": ["samples", "2"], "unit": "", "description": "…"} ] },
   "result_version": 1, "gpu_path": false, "stoppable": false }]
```

`mk_engine_analysis_info(engine, name, &json)` returns one such object.
`stoppable` is true for the analyses whose jobs accept `mk_job_stop`.

`settings_schema` is a JSON-Schema subset: `type` (name or list of
`number|integer|string|boolean|array|object|null`), `default`, `minimum`,
`maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `enum`, `items`, `required`, plus descriptive `title`,
`description` and `unit`. `result_schema` uses the same subset plus
`anyOf`, `minItems`/`maxItems` and `additionalProperties`, and lists the
result's typed buffers under `x-buffers` (`docs/RESULTS.md` §4). Submitted settings may be partial; missing keys
take their defaults, unknown keys are rejected. The complete, normalised
settings are recorded in `run_info.settings`. `default_settings` is `null`
when some setting is required.

### Backends (`mk_engine_list_backends`)

```json
[{ "backend": {"kind": "cpu_reference"}, "id": "cpu_reference",
   "title": "CPU reference (f64)", "available": true, "reason": null,
   "precision": "f64_reference",
   "capabilities": { "fp64": true, "reference": true, "analyses": [],
                     "device_name": null, "unified_memory": null } }]
```

A backend is requested as `{"kind": "cpu_reference"}`,
`{"kind": "gpu", "name": "metal"}` or `{"kind": "auto"}`. On macOS the
Metal backend (`gpu:metal`) is listed; it is opened on first use and is
`available` when the machine has a Metal device that passes the numerics
probe (`reason` says why not; `MKLIGHT_DISABLE_GPU=1` disables every GPU
backend). Its `capabilities.analyses` lists the analyses with a GPU path
(`gpu_path: true`). Requesting an unavailable backend fails with
`MK_ERR_UNSUPPORTED` unless the request sets `allow_backend_fallback`, in
which case the job runs on the CPU reference and `run_info.warnings` says
so.

Selection is per analysis: a GPU backend runs only the analyses listed in
its `capabilities.analyses` (the CPU reference runs all). Requesting it
explicitly for any other analysis is handled like an unavailable backend
(error, or fallback with a warning). `auto` chooses per request with the
calibrated cost model of GPU §9.2; a CPU choice carries no warning, and the
result's `backend.cost_decision` and `backend.fallback_reason` say why. When the request names no
backend, the engine default (`default_backend` in the configuration, or
`mk_engine_set_default_backend`) applies as a preference: analyses the
default backend does not implement run on the CPU reference, with a
`run_info.warnings` note for an explicit GPU default.

What a selected GPU publishes, how fallbacks and cancellation are reported
and what `backend.execution` and `backend.cost_decision` hold are in
RESULTS §5 and GPU §9.2.

### Analysis request (`mk_session_submit_analysis_json`)

```json
{ "analysis": "rng_stream", "settings": {"seed": 1}, "view_key": "win-3",
  "backend": {"kind": "gpu", "name": "metal"}, "allow_backend_fallback": true,
  "tier": "design", "aim_stop": null,
  "route": { "fieldmaps": "device-chief", "stray_share": 1 } }
```

`tier` (`design`, the default, or `preview`) and `aim_stop` (`requested`,
`conditioned`, or null for the engine default, `conditioned`; an explicit
aiming tolerance in the system is always honoured as `requested`) are the accuracy policies of
docs/RESULTS.md; both are echoed in `run_info`. `route` holds the route
options of this request (the object of the engine configuration's `route`;
GPU §9.3): null or absent takes the engine configuration's, and a key the
object leaves out takes its default, not the configuration's value. An
unknown key or name, or a share outside [0, 1], fails the submission
(`MK_ERR_INVALID_ARGUMENT`), as does any key the request does not have.

`mk_session_run_analysis_json(session, request, &json, &buffers, &count)`
runs the same request in the calling thread (blocking, no job, no
callback; `view_key` is ignored) and returns the result like
`mk_job_take_result`; errors of the run are returned directly. The
session lock is held only while the request is checked and the snapshot
captured (as for submit), so other calls on the session, from other
threads, proceed during the run; edits made meanwhile do not affect it.
The run cannot be cancelled and has no progress: use it for quick, small
analyses (validation, first-order data); interactive windows and anything
long (stray light, large grids) should submit jobs.

### Job status (`mk_job_status_json`)

```json
{ "id": 7, "state": "running", "progress": 0.42, "stage": "drawing",
  "label": "rng_stream @ 3f2a9c01b7de", "view_key": "session-1/win-3",
  "error": null, "superseded_by": null, "elapsed_ms": 812.5 }
```

`state` is `queued|running|done|failed|cancelled` (numeric `MKJobState`
through `mk_job_status`).

### Result envelope (`mk_job_take_result`)

The envelope (`run_info`, `library_hash`, `derived_snapshot_hash`, `backend`,
`result_version`, `result`, `buffers`) is described in `docs/RESULTS.md` §1
and §5. When the request names no backend, `backend.requested` carries the
`default:` prefix (for example `default:auto`).

A result's inputs are identified by the pair (`run_info.snapshot_hash`,
`library_hash`): the system and the glass data it was computed with.
`library_hash` is the BLAKE3 hash of every catalog in the session's
material library (name and canonical JSON, in load order), so loading,
replacing or reloading an edited catalog changes it even though the system
(and its snapshot hash) did not change. A displayed result is **current**
only while both equal `mk_session_current_snapshot_hash` and
`mk_session_current_library_hash`; otherwise it is stale. (`library_hash`
sits beside `run_info` until `RunInfo` in mklight-core carries it.)

### Session info (`mk_session_info_json`)

```json
{ "session_id": 1, "path": "/…/lens.mklight", "dirty": false,
  "snapshot_hash": "<64 hex>", "library_hash": "<64 hex>", "name": "Untitled",
  "open_warnings": ["glass catalog 'MYCAT' could not be loaded (…)"],
  "catalogs": [ {"name": "BUILTIN", "source": "builtin", "path": null, "glass_count": 13} ] }
```

### Optical system and parameter edits

`mk_session_get_system_json` / `mk_session_set_system_json` use the
`OpticalSystem` serialisation of `mklight-core` (see `crates/mklight-core/src/model`).
Replacing the system validates it; an invalid system is rejected and the
session keeps its previous snapshot. Materials are resolved lazily, so an
unknown glass is accepted by `set_system` and reported by analyses
(`validate` lists it; analyses that need indices fail with it).

`mk_session_apply_param_edit` takes one edit or an array, applied
atomically (all or none):

```json
[{ "path": {"param": "thickness", "surface": 3}, "value": 12.5 },
 { "path": {"param": "asphere", "surface": 4, "index": 0}, "value": 1.2e-6 },
 { "path": {"param": "field_y", "index": 2}, "value": 20.0 },
 { "path": {"param": "aperture_value"}, "value": 25.0 }]
```

`surface` is the stable `SurfaceId`, not the index. Parameters
(`mklight_core::param::ParamPath`):

* per surface (+`surface`): `curvature`, `thickness` (accepts `"inf"`),
  `conic`, `asphere` (+`index`: 0 → A4), `paraxial_focal_length`,
  `model_nd`, `model_vd`, `model_dpgf`, `material_dn_d`/`material_dv_d`
  (the Δn_d and ΔV_d offsets of a catalogue glass), `decenter_x`,
  `decenter_y`, `tilt_x`, `tilt_y`, `tilt_z`, `semi_diameter`;
* `shape_coefficient` (+`surface`, `set`: `odd`, `q`, `phase`,
  `x_asphere`, `y_asphere`, `xy` or `zernike`, and a zero-based `index`)
  and `shape_param` (+`surface`, `parameter`: `curvature_x`, `conic_x`,
  `curvature_y`, `conic_y`, `norm_radius`, `astigmatism_angle_deg`,
  `design_order`, `design_wavelength_um` or `trace_order`) for the surface
  types of SURFACES.md;
* `irregularity` (+`surface`, `mode`: `astigmatism` or `spherical`; test
  fringes) and `element_placement` (+`surface`, `last`, optional `pivot`,
  `component`: `decenter_x`, `decenter_y`, `tilt_x`, `tilt_y` or `tilt_z`;
  the rigid motion of the surfaces `surface..=last`);
* system: `aperture_value`, `field_x`/`field_y`/`field_weight` (+`index`),
  `wavelength_um`/`wavelength_weight` (+`index`), `temperature_c`,
  `pressure_atm`.

Both edit functions return the new snapshot hash (unchanged when the
values were already set).

`mk_session_get_param(session, path_json, &json)` reads one parameter:
`{"value": number | "inf" | "-inf" | null}`. Automatic semi-diameters
return the resolved layout radius (same aperture solve and domain clipping);
writes still set a fixed radius. A path the system does not have is
`MK_ERR_INVALID_ARGUMENT`.

### Validation, solves, autosave

* `mk_session_validate_json(session, &json)`: the `validate` report
  `{ok, errors, warnings, issues: [{severity, code, surface, message}]}` of
  the current snapshot. Structural problems (non-finite values, aperture
  values ≤ 0, negative weights, missing references, …) are already rejected
  by core at session/snapshot creation (CONVENTIONS §13). The analysis checks
  material resolution and tracer setup and warns about chief/marginal rays
  that do not reach the image (`docs/RESULTS.md` §6).
* `mk_session_apply_solves(session, settings, &report, &hash)`: solves the
  system (`docs/SOLVES.md`) and commits it as a new snapshot (marks the
  session dirty); `report` is the `solves` result. The `solves` analysis is
  the dry run (and can solve one configuration). The solve runs without the
  session lock (blocking the calling thread, not cancellable; solves are
  fast); if the session changed meanwhile (an edit, undo, a catalog load),
  nothing is committed and the call returns `MK_ERR_CANCELLED`
  (superseded): solve again. `mk_session_validate_json` likewise runs
  outside the lock.
* `mk_session_autosave(session, path, &written)`: writes the project to
  `path` or `<project file>.autosave` without touching the session (path,
  dirty flag). Recover by opening the autosave file.

`mk_session_set_project_section_json(session, section, json)` replaces
`views` (array of `{key, analysis, settings, ui}`), `merit`, `variables`,
`tolerances` (the front end's tolerance setup: stored, never interpreted by
the engine) or `metadata`.

### Glass catalogs

* `mk_session_load_catalog(session, path, &json)` reads an AGF-format text
  file, or a `GlassCatalog` JSON file when the extension is `.json`, adds it
  to the session's material library (replacing a catalog of the same name)
  and records it in the project. AGF catalogs are named after the file stem.
  The name `BUILTIN` is reserved. Result:
  `{"name", "source": "user", "path", "glass_count"}`.
* `mk_session_list_catalogs` → array of the same objects (built-in first).
* `mk_session_list_glasses(session, catalog|NULL)` →
  `[{"catalog", "name", "nd", "vd", "min_um", "max_um"}]`.

Vendor catalogs are never embedded in project files, only referenced (by
absolute path and by path relative to the project file). A catalog that
cannot be found on open is reported in `open_warnings`; the reference is
kept.

### Lens prescription import

* `mk_import_prescription(bytes, len, format, &json)` imports `.zmx`
  (`format` `"zmx"`) or `.seq` (`"seq"`) text in any common encoding
  without a session; glasses resolve against the built-in catalog. An
  unknown format is `MK_ERR_INVALID_ARGUMENT`, text without surfaces
  `MK_ERR_PARSE`. Result: `{"format", "encoding", "system", "warnings":
  [{"kind", "line", "surface", "message"}]}`, where `system` is an
  `OpticalSystem`.
* `mk_session_import_prescription(session, path, &json|NULL)` imports a
  file (format by extension) with the session's catalogs and replaces the
  session's system; the result adds `"snapshot_hash"`. On failure the
  session is unchanged.

Content that does not map is a warning, never an error; `IMPORT.md` lists
the mapping and the warning kinds.

## Analyses

Every analysis, its settings (with defaults) and its result (fields, units,
signs, buffer names, dtypes and shapes) is documented in
**`docs/RESULTS.md`**, whose reference section (§8) is generated from the
same schemas `mk_engine_list_analyses` returns and lists every registered
analysis. An analysis listed with `available: false` gives its
`unavailable_reason`; submitting it returns `MK_ERR_UNSUPPORTED`.

## Project file (`.mklight`)

Pretty-printed UTF-8 JSON, written atomically (temporary file in the same
directory, `fsync` — `F_FULLFSYNC` on Apple platforms — then rename), so a
crash never leaves a half-written file. Saving through a symbolic link
replaces the link's target and keeps the link; the existing file's
permissions are kept; a read-only file is refused (`MK_ERR_IO`).

```json
{ "format": "mklight-project", "schema_version": 1,
  "metadata": { "created": "2026-10-30T12:00:00Z", "modified": "…",
                "app_name": "MKLight", "app_version": "1.0",
                "engine_version": "0.1.0", "author": "" },
  "system": { …OpticalSystem… },
  "catalogs": [ {"name": "MYCAT", "path": "/abs/mycat.agf",
                 "relative_path": "glass/mycat.agf", "format": "agf"} ],
  "views": [ {"key": "win-1", "analysis": "system_summary", "settings": {}, "ui": {}} ],
  "merit": null, "variables": null, "tolerances": null }
```

Catalog references: on open, `relative_path` (relative to the project's
directory) is tried first, then `path`. The catalog's `path` is updated to
where it was actually found, and saving recomputes `relative_path` from it,
so a project folder can be moved (or copied to another machine) any number
of times. When the catalog file is not found at all, the previous
`relative_path` is kept on save. A catalog outside the project folder gets
`relative_path: null`.

Unknown top-level keys are preserved on a round trip. Only the current
`schema_version` is read: a file without one or at any other version is
refused with `MK_ERR_PARSE`.

## Swift usage sketch

```swift
import CMKLight

var engine: OpaquePointer?
guard mk_engine_create(nil, &engine) == MK_OK else { fatalError(String(cString: mk_last_error())) }
var session: OpaquePointer?
mk_session_new(engine, &session)

var job: UInt64 = 0
mk_session_submit_analysis(session, "system_summary", nil, "summary-pane", &job)
var state: Int32 = 0
mk_job_wait(engine, job, -1, &state)

var json: UnsafeMutablePointer<CChar>?
var buffers: UnsafeMutablePointer<MKBuffer>?
var count = 0
if mk_job_take_result(engine, job, &json, &buffers, &count) == MK_OK {
    let text = String(cString: json!)
    mk_string_free(json)
    // read buffers![i].data as UnsafeRawPointer, then:
    mk_buffer_free(buffers, count)
}
mk_session_destroy(session)
mk_engine_destroy(engine)
```

## GPU kernel preparation

Metal compiles each compute kernel the first time a process creates its
pipeline. All of MKLight's kernels take about 160 CPU-seconds when nothing is
cached (a new install or update, a macOS update, or an evicted system shader
cache): about 30 s on an M2 Pro, where Metal's compiler service runs about
six compiles at once. The first GPU analysis would wait for its kernels. A front end calls `mk_engine_prepare_gpu(engine,
cache_dir)` at launch: a background thread opens the GPU and creates every
kernel and the common specialisations of the specialised ones (205
pipelines, six at a time). With `cache_dir`, the compiled pipelines are kept in a Metal
binary archive there (one file per GPU and shader library, about 50 MB;
archives of earlier builds are removed), and later launches load them in
milliseconds instead of compiling. Analyses may run meanwhile: a kernel the
preparation has not reached yet compiles for the job that needs it.

`mk_engine_gpu_preparation_json` reports the state, for a status display:
`{"state": "preparing", "device", "done", "total"}`, then
`{"state": "ready", "device", "kernels", "from_cache", "seconds"}` (plus a
`warning` when the saved kernels could not be read or written), or
`"unavailable"` with a `reason` (no GPU; analyses use the CPU reference) or
`"failed"` with a `message` (kernels compile on first use instead).

