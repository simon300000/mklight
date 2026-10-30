# MKLight C ABI (`mklight-ffi`)

Native front ends (the macOS app first) drive the engine through a small C
ABI. Structured data crosses as UTF-8 JSON; bulk numeric results cross as
typed binary buffers. The header is `crates/mklight-ffi/include/mklight.h`
(hand-written, checked against the Rust constants by a unit test), and
`include/module.modulemap` exposes it to Swift as the module `CMKLight`.
`crates/mklight-ffi/tests/c/smoke.c` exercises every function and is
compiled, linked against the static library and run by `cargo test -p mklight-ffi`.

## Building and linking

```sh
cargo build -p mklight-ffi --release   # → target/release/libmklight_ffi.a
```

Link `libmklight_ffi.a` plus the native libraries the Rust standard library
needs (`rustc --print native-static-libs`): nothing extra on macOS (`-lSystem`
is implicit), `-lpthread -ldl -lm` on Linux.

Swift: add `crates/mklight-ffi/include` to the import search paths (it holds
the module map), link the static library, then `import CMKLight`.

## Versioning

* `mk_abi_version()` returns `(major << 16) | minor`; the header has
  `MK_ABI_VERSION_MAJOR/MINOR`. A front end must check that the runtime major
  equals the compiled major. Minor releases only **add** functions and JSON
  fields, so JSON readers must ignore unknown keys.
* `mk_engine_version()` is the engine (crate) version recorded in every
  `RunInfo` and project file.
* Project files carry `schema_version` (currently 1). Older files are
  migrated on open; newer ones are refused (see *Project file*).

## Status codes and errors

Every fallible function returns an `MKStatus`. On failure, `mk_last_error()`
returns a human-readable message for **the calling thread**; it is `""` after
a successful call and stays valid until the next `mk_*` call on that thread.
It is never NULL.

| code | name | meaning |
|---|---|---|
| 0 | `MK_OK` | success |
| 1 | `MK_ERR_INVALID_ARGUMENT` | NULL where not allowed, invalid UTF-8, invalid settings or parameter edit |
| 2 | `MK_ERR_PARSE` | malformed JSON, project file or catalog |
| 3 | `MK_ERR_INVALID_MODEL` | the optical system fails validation |
| 4 | `MK_ERR_NOT_FOUND` | unknown job, analysis, glass or catalog |
| 5 | `MK_ERR_IO` | file-system error |
| 6 | `MK_ERR_UNSUPPORTED` | analysis or backend not available in this build |
| 7 | `MK_ERR_CANCELLED` | the job was cancelled or superseded |
| 8 | `MK_ERR_TIMEOUT` | `mk_job_wait` timed out |
| 9 | `MK_ERR_NOT_READY` | the job has not finished |
| 10 | `MK_ERR_JOB_FAILED` | the job ended with an error |
| 11 | `MK_ERR_COMPUTE` | trace, numerical or backend failure |
| 12 | `MK_ERR_PANIC` | an internal panic was caught at the boundary |
| 13 | `MK_ERR_INTERNAL` | other internal error |

No panic ever unwinds into C: every entry point catches it and returns
`MK_ERR_PANIC`. The library stays usable afterwards.

## Ownership

| what | allocated by | released with |
|---|---|---|
| `MKEngine*` | `mk_engine_create` | `mk_engine_destroy` |
| `MKSession*` | `mk_session_new` / `mk_session_open` | `mk_session_destroy` |
| `char*` returned through `char**` | the library | `mk_string_free` |
| `MKBuffer*` array + everything it points to | `mk_job_take_result` | `mk_buffer_free(array, count)` |
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

* It is invoked on **engine threads** (workers, the progress poller), or on
  the thread that calls `mk_session_submit_analysis*` / `mk_job_cancel` /
  `mk_job_release` (for the `QUEUED` event and for jobs cancelled while
  queued). Marshal to the main thread yourself (for example with
  `DispatchQueue.main.async`).
* Per job, events arrive in lifecycle order and never concurrently:
  `QUEUED` → `RUNNING` (+ progress updates) → `DONE` | `FAILED` | `CANCELLED`.
  Progress updates are polled every `progress_interval_ms` and sent only
  when the fraction or stage changed. Progress is monotone.
* The callback must be thread-safe and return quickly. It may call other
  `mk_*` functions, but not `mk_engine_set_job_callback` or
  `mk_engine_destroy`.
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
* `mk_job_wait(engine, job, timeout_ms, &state)`: `timeout_ms < 0` waits
  forever. Returns `MK_OK` with a terminal state or `MK_ERR_TIMEOUT` with
  the current state.
* `mk_job_take_result` returns the result once and forgets the job (see the
  header for the per-state status codes). `mk_job_release` forgets a job
  without taking the result (cancelling it if needed).
* Retention: at most `max_retained_jobs` finished jobs are kept; older ones
  are dropped with their results (`MK_ERR_NOT_FOUND` afterwards). Take or
  release results promptly.

## Buffers

```c
typedef struct MKBuffer {
    const char *name;   const void *data;   size_t len;
    MKDType dtype;      uint32_t ndim;      const size_t *shape;
} MKBuffer;
```

`data` holds `len` elements (not bytes) of `dtype`, aligned for that type,
row-major with `shape[0..ndim]` (product = `len`). dtypes: `F64=1, F32=2,
I32=3, U32=4, I64=5, U64=6, U8=7`. The result JSON lists the buffers
(`buffers: [{index, name, dtype, shape}]`) in the same order as the array,
so they can be found by name. The whole array is freed with one
`mk_buffer_free(array, count)` call.

## JSON reference

Units and signs follow `docs/CONVENTIONS.md` (mm, µm, degrees; curvature
stored, not radius). JSON cannot hold infinities: `±∞` are the strings
`"inf"` / `"-inf"` (object distance, radius of a plane). NaN is `null`.

### Engine configuration (`mk_engine_create`)

All keys optional:

```json
{ "worker_threads": 0, "max_retained_jobs": 256, "resolved_cache_capacity": 8,
  "progress_interval_ms": 50, "app_name": "MKLight", "app_version": "1.0",
  "default_backend": {"kind": "cpu_reference"} }
```

`worker_threads: 0` picks `min(cores, 4)`. `app_name`/`app_version` are
stamped into project metadata on save.

### Analyses (`mk_engine_list_analyses`)

```json
[{ "name": "glass_info", "title": "Glass data", "category": "materials",
   "description": "…", "available": true, "unavailable_reason": null,
   "settings_schema": { "type": "object", "properties": { … }, "required": ["name"] },
   "default_settings": null }]
```

`settings_schema` is a JSON-Schema subset: `type` (name or list of
`number|integer|string|boolean|array|object|null`), `default`, `minimum`,
`maximum`, `enum`, `items`, `required`, plus descriptive `title`,
`description` and `unit`. Submitted settings may be partial; missing keys
take their defaults, unknown keys are rejected. The complete, normalised
settings are recorded in `run_info.settings`. `default_settings` is `null`
when some setting is required.

### Backends (`mk_engine_list_backends`)

```json
[{ "backend": {"kind": "cpu_reference"}, "id": "cpu_reference",
   "title": "CPU reference (f64)", "available": true, "reason": null,
   "precision": "f64_reference",
   "capabilities": { "fp64": true, "reference": true, "analyses": [],
                     "device_name": null, "unified_memory": null,
                     "max_rays_per_dispatch": null } }]
```

A backend is requested as `{"kind": "cpu_reference"}` or
`{"kind": "gpu", "name": "metal"}`. On macOS a Metal entry is listed with
`available: false` until the Metal backend is wired in. Requesting an
unavailable backend fails with `MK_ERR_UNSUPPORTED` unless the request sets
`allow_backend_fallback`, in which case the job runs on the CPU reference
and `run_info.warnings` says so.

### Analysis request (`mk_session_submit_analysis_json`)

```json
{ "analysis": "rng_stream", "settings": {"seed": 1}, "view_key": "win-3",
  "backend": {"kind": "gpu", "name": "metal"}, "allow_backend_fallback": true }
```

### Job status (`mk_job_status_json`)

```json
{ "id": 7, "state": "running", "progress": 0.42, "stage": "drawing",
  "label": "rng_stream @ 3f2a9c01b7de", "view_key": "session-1/win-3",
  "error": null, "superseded_by": null, "elapsed_ms": 812.5 }
```

`state` is `queued|running|done|failed|cancelled` (numeric `MKJobState`
through `mk_job_status`).

### Result envelope (`mk_job_take_result`)

```json
{ "run_info": { "engine_version": "0.1.0", "snapshot_hash": "<64 hex>",
                "analysis": "system_summary", "settings": {…},
                "backend": "cpu_reference", "precision": "f64_reference",
                "seed": null, "started_unix_ms": 1761840000000,
                "duration_ms": 0.31, "warnings": [] },
  "result": { … analysis specific … },
  "buffers": [ {"index": 0, "name": "n_after", "dtype": "f64", "shape": [4, 3]} ] }
```

Compare `run_info.snapshot_hash` with `mk_session_current_snapshot_hash` to
tell whether a displayed result is stale.

### Session info (`mk_session_info_json`)

```json
{ "session_id": 1, "path": "/…/lens.mklight", "dirty": false,
  "snapshot_hash": "<64 hex>", "name": "Untitled",
  "open_warnings": ["glass catalog 'MYCAT' could not be loaded (…)"],
  "migration": {"from_version": 0, "to_version": 1, "notes": ["…"]},
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

`surface` is the stable `SurfaceId`, not the index. Parameters:
`curvature`, `thickness` (accepts `"inf"`), `conic`, `asphere` (+`index`:
0 → A4), `paraxial_focal_length`, `model_nd`, `model_vd`, `model_dpgf`,
`decenter_x`, `decenter_y`, `tilt_x`, `tilt_y`, `tilt_z`, `semi_diameter`,
`aperture_value`, `field_x`/`field_y`/`field_weight` (+`index`),
`wavelength_um`/`wavelength_weight` (+`index`), `temperature_c`,
`pressure_atm`. Both edit functions return the new snapshot hash
(unchanged when the values were already set).

`mk_session_set_project_section_json(session, section, json)` replaces
`views` (array of `{key, analysis, settings, ui}`), `merit`, `variables`,
`tolerances` (opaque until the optimiser defines them) or `metadata`.

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

## Built-in analyses

| name | settings | result | buffers |
|---|---|---|---|
| `validate` | none | `{ok, errors, warnings, issues: [{severity, code, surface, message}]}`; codes: `structure`, `material`, `coating`, `resolve`, `material_range`, `stop_size_undefined`, `stop_on_coordinate_break`, `thickness_sign`, `zero_weights`, `configuration`, `unused_coating` | — |
| `system_summary` | none | surface table (`index, id, label, type, is_stop, curvature, radius, conic, shape, thickness, material, material_ref, interaction, semi_diameter, aperture, coating, n_after[w], nd_after, vd_after, alpha_after_per_mm, frame{origin, rot}, to_next_is_axial`), system wavelengths, aperture, fields, `n_object`, `first_to_image_distance`, warnings | `n_after` f64 [surfaces × wavelengths]; `vertex_global` f64 [surfaces × 3] (NaN for an object at infinity) |
| `glass_info` | `name` (required), `catalog`, `wavelengths_um` (default: system), `temperature_c`, `pressure_atm` (default: system environment), `curve_samples` | catalog data, computed `nd`, `ne`, `vd`, `pgf`, dispersion formula, and `indices: [{wavelength_um, n, n_catalog, absorption_per_mm, in_range}]` (n relative to air at the chosen environment) | `dispersion_curve` f64 [samples × 2] = (λ µm, n) |
| `rng_stream` (diagnostic) | `seed`, `stream`, `count`, `chunk`, `delay_ms` | `{count, mean, variance, min, max, first}` of a Philox4x32-10 uniform stream | `uniform` f64 [count] |

Names reserved for analyses of later phases are listed with
`available: false`: `paraxial`, `layout`, `spot_diagram`, `ray_fan`,
`opd_fan`, `field_curvature_distortion`, `seidel`, `wavefront_map`, `psf`,
`mtf`, `ghost_paths`, `stray_light`. Submitting one returns
`MK_ERR_UNSUPPORTED`.

## Project file (`.mklight`)

Pretty-printed UTF-8 JSON, written atomically (temporary file in the same
directory, `fsync`, rename), so a crash never leaves a half-written file.

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

Unknown top-level keys are preserved on a round trip. Migrations run one
version step at a time (`MIGRATIONS[k]`: version k → k+1). Version 0 denotes
unversioned drafts: a bare `OpticalSystem` or `{ "system": … }`, which are
wrapped into a v1 project (with a note in `migration.notes`). A file with a
newer `schema_version` than the engine supports is refused with
`MK_ERR_PARSE`.

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

## Integration points for later phases

* Solver crates register analyses with `Engine::register_analysis` (or in
  `analysis::builtin::register_all`), replacing the placeholders of the same
  name. They receive the snapshot, the cached `ResolvedSystem`, the selected
  backend and a `JobContext`.
* GPU backends implement `backend::BackendProvider` and are registered with
  `Engine::register_backend`; analyses read `AnalysisInput::backend`.
* Typed merit/variables/tolerances replace the opaque project sections
  through a schema migration (v1 → v2).
