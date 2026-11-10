/*
 * mklight.h — C ABI of the MKLight optical engine.
 *
 * Hand-written; kept in sync with crates/mklight-ffi/src/lib.rs by the
 * `header_matches_rust_constants` test. The full contract (ownership,
 * threading, JSON schemas) is in docs/FFI.md.
 *
 * Conventions
 *  - Every fallible function returns an MKStatus. MK_OK (0) means success.
 *    On failure, mk_last_error() returns a message for the calling thread.
 *  - Strings passed in are NUL-terminated UTF-8. Parameters documented as
 *    "nullable" accept NULL; all others must not be NULL.
 *  - Strings returned through `char **` are owned by the caller and must be
 *    released with mk_string_free(). On failure they are set to NULL.
 *  - Buffer arrays returned by mk_job_take_result() and
 *    mk_session_run_analysis_json() are released with mk_buffer_free().
 *    Their data is shared with the engine, never copied, and read-only.
 *  - Structured data is JSON. Infinite lengths are the strings "inf"/"-inf".
 *  - JSON passed in: text that is not JSON is MK_ERR_PARSE; JSON of the
 *    wrong shape (a missing key, a value of the wrong type, a key the object
 *    does not have) is MK_ERR_INVALID_ARGUMENT, and the message names the
 *    key. No unknown key is ignored.
 *  - No function lets a panic or exception escape: internal errors are
 *    reported as MK_ERR_PANIC / MK_ERR_INTERNAL.
 */
#ifndef MKLIGHT_H
#define MKLIGHT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Versioning --------------------------------------------------------- */

#define MK_ABI_VERSION_MAJOR 1
#define MK_ABI_VERSION_MINOR 4
/* (major << 16) | minor. A different major version is incompatible; a newer
 * minor version only adds functions and JSON fields. */
#define MK_ABI_VERSION ((MK_ABI_VERSION_MAJOR << 16) | MK_ABI_VERSION_MINOR)

/* Runtime ABI version of the linked library. Check that
 * (mk_abi_version() >> 16) == MK_ABI_VERSION_MAJOR. */
uint32_t mk_abi_version(void);

/* Engine version string, statically allocated (do not free). */
const char *mk_engine_version(void);

/* ---- Status codes ------------------------------------------------------- */

typedef int32_t MKStatus;

#define MK_OK 0
#define MK_ERR_INVALID_ARGUMENT 1 /* NULL/invalid argument, bad UTF-8, bad settings,
                                     JSON of the wrong shape, unknown key */
#define MK_ERR_PARSE 2            /* text that is not JSON; malformed project or
                                     catalog file */
#define MK_ERR_INVALID_MODEL 3    /* optical system fails validation */
#define MK_ERR_NOT_FOUND 4        /* unknown job, analysis, glass, catalog */
#define MK_ERR_IO 5               /* file system error */
#define MK_ERR_UNSUPPORTED 6      /* analysis or backend not available */
#define MK_ERR_CANCELLED 7        /* job was cancelled */
#define MK_ERR_TIMEOUT 8          /* mk_job_wait timed out */
#define MK_ERR_NOT_READY 9        /* job has not finished */
#define MK_ERR_JOB_FAILED 10      /* job ended with an internal error (a panic) */
#define MK_ERR_COMPUTE 11         /* trace / numerical / backend failure */
#define MK_ERR_PANIC 12           /* internal error caught at the boundary */
#define MK_ERR_INTERNAL 13

/* Message of the last failed mk_* call on this thread ("" after a successful
 * call). Never NULL. Valid until the next mk_* call on the same thread. */
const char *mk_last_error(void);

/* Free a string returned by this library. NULL is ignored. */
void mk_string_free(char *s);

/* ---- Buffers ------------------------------------------------------------ */

typedef int32_t MKDType;
#define MK_DTYPE_F64 1
#define MK_DTYPE_F32 2
#define MK_DTYPE_I32 3
#define MK_DTYPE_U32 4
#define MK_DTYPE_I64 5
#define MK_DTYPE_U64 6
#define MK_DTYPE_U8 7

/* A named, row-major typed array. All pointers are owned by the enclosing
 * array and stay valid until mk_buffer_free().
 *
 * `data` is the array the engine computed, not a copy: the engine may keep
 * the same memory for later requests, and another take of the same result
 * may return the same `data`. It is read-only. Each buffer array holds its
 * own reference (`owner`), so its pointers stay valid until it is freed,
 * whatever happens to the engine or to other arrays. */
typedef struct MKBuffer {
    const char *name;    /* NUL-terminated UTF-8 */
    const void *data;    /* `len` elements of `dtype`, aligned; non-NULL */
    size_t len;          /* element count (not bytes) */
    MKDType dtype;
    uint32_t ndim;
    const size_t *shape; /* `ndim` extents whose product is `len` */
    const void *owner;   /* private: keeps `data` alive; do not read or change */
} MKBuffer;

/* Free an array of `count` buffers from mk_job_take_result() or
 * mk_session_run_analysis_json(), exactly once. NULL is ignored. `count`
 * must be the count returned with the array, and the array must be as it
 * was returned. */
void mk_buffer_free(MKBuffer *buffers, size_t count);

/* ---- Handles ------------------------------------------------------------ */

typedef struct MKEngine MKEngine;   /* thread-safe */
typedef struct MKSession MKSession; /* thread-safe; calls are serialised */

/* ---- Engine ------------------------------------------------------------- */

/* config_json (nullable): {"worker_threads": 0, "max_retained_jobs": 256,
 * "resolved_cache_capacity": 8, "analysis_graph_bytes": 268435456,
 * "progress_interval_ms": 50, "app_name": "", "app_version": "",
 * "default_backend": {"kind": "auto"}, "aim_stop": "conditioned",
 * "route": {}}; every key is optional and these are the defaults.
 * "aim_stop" ("conditioned" or "requested") is the aim-stop policy of
 * requests that name none. "route" holds the route options of requests
 * that carry none: {"aiming": "host-table" | "table" | "device",
 * "aiming_share": null | 0…1, "fieldmaps": "host-chief" | "device-chief",
 * "stray_share": null | 0…1, "stray_sag": "f32-scan" | "f32-grazing",
 * "solver_aiming": "by-batch" | "host" | "device"}, every key optional
 * with the first value as its default (docs/GPU.md §9.3); an unknown key or
 * name fails here, a share outside 0…1 fails the requests that use it. */
MKStatus mk_engine_create(const char *config_json, MKEngine **out_engine);

/* Clears the job callback (waiting for calls in progress), then releases
 * the handle. Worker threads stop when all sessions are destroyed too. */
void mk_engine_destroy(MKEngine *engine);

/* JSON array of analysis descriptions (name, title, category, description,
 * available, unavailable_reason, settings_schema, default_settings,
 * result_schema, result_version, gpu_path, stoppable). "stoppable": the
 * analysis' jobs accept mk_job_stop. The result schemas, and every
 * analysis' result fields and buffers, are documented in docs/RESULTS.md. */
MKStatus mk_engine_list_analyses(const MKEngine *engine, char **out_json);

/* JSON array of compute backends (backend, id, title, available, reason,
 * precision, capabilities). On macOS "gpu:metal" is listed; it is available
 * when the machine has a Metal device that passes the numerics probe. */
MKStatus mk_engine_list_backends(const MKEngine *engine, char **out_json);

/* ---- Jobs: states and event callback ------------------------------------ */

typedef int32_t MKJobState;
#define MK_JOB_QUEUED 0
#define MK_JOB_RUNNING 1
#define MK_JOB_DONE 2
#define MK_JOB_FAILED 3
#define MK_JOB_CANCELLED 4

/* Job event callback. Called on every state change and, while a job runs,
 * when its progress or stage changed (polled every progress_interval_ms).
 * THREADING: invoked on one dedicated engine thread ("mklight-events"), one
 * event at a time, never on the thread that submitted, cancelled or
 * released the job, and with no engine or session lock held. Per job the
 * events arrive in lifecycle order: QUEUED, then CANCELLED, or RUNNING
 * (repeated for progress) followed by one terminal state. Events are queued,
 * so a slow callback delays later events but never blocks jobs. The
 * callback may call any mk_* function (including mk_job_wait and
 * mk_engine_set_job_callback) except mk_engine_destroy on this engine.
 * `stage` is valid only during the call. */
typedef void (*MKJobCallback)(void *user_data, uint64_t job_id, MKJobState state,
                              double progress, const char *stage);

/* Install (cb != NULL) or clear (cb == NULL) the callback. When this
 * returns, no other thread is still inside the previous callback, so its
 * user_data may be released. */
MKStatus mk_engine_set_job_callback(MKEngine *engine, MKJobCallback cb, void *user_data);

/* ---- Sessions ----------------------------------------------------------- */

/* New untitled project. */
MKStatus mk_session_new(const MKEngine *engine, MKSession **out_session);

/* Open a .mklight project (only the current schema version). Missing user
 * catalogs do not fail the open: see "open_warnings" in mk_session_info_json. */
MKStatus mk_session_open(const MKEngine *engine, const char *path, MKSession **out_session);

/* Jobs already submitted keep running; their results stay available through
 * the engine. */
void mk_session_destroy(MKSession *session);

/* Atomic save (temporary file + rename). path (nullable): NULL saves to the
 * path the session was opened from or last saved to. */
MKStatus mk_session_save(MKSession *session, const char *path);

/* {"session_id", "path", "dirty", "snapshot_hash", "library_hash", "name",
 *  "open_warnings", "catalogs"} */
MKStatus mk_session_info_json(const MKSession *session, char **out_json);

/* The whole project document (same schema as the .mklight file). */
MKStatus mk_session_get_project_json(const MKSession *session, char **out_json);

/* The current OpticalSystem. */
MKStatus mk_session_get_system_json(const MKSession *session, char **out_json);

/* Replace the OpticalSystem. A system that fails validation is rejected
 * (MK_ERR_INVALID_MODEL) and leaves the session unchanged. out_hash
 * (nullable) receives the new snapshot hash. */
MKStatus mk_session_set_system_json(MKSession *session, const char *system_json,
                                    char **out_hash);

/* Apply {"path": ParamPath, "value": number|"inf"} or an array of them,
 * atomically. out_hash (nullable) receives the new snapshot hash. */
MKStatus mk_session_apply_param_edit(MKSession *session, const char *edits_json,
                                     char **out_hash);

/* Replace a project section: "views", "merit", "variables", "tolerances" or
 * "metadata". */
MKStatus mk_session_set_project_section_json(MKSession *session, const char *section,
                                             const char *json);

/* 64 lowercase hex characters (BLAKE3 of the canonical system JSON). */
MKStatus mk_session_current_snapshot_hash(const MKSession *session, char **out_hash);

/* 64 lowercase hex characters (BLAKE3 of the glass library's catalogs in
 * load order). Changes when a catalog is added, replaced or reloaded with
 * different data. A result is current while BOTH its run_info.snapshot_hash
 * and its library_hash equal the session's. */
MKStatus mk_session_current_library_hash(const MKSession *session, char **out_hash);

/* ---- Glass catalogs ----------------------------------------------------- */

/* Load a user glass catalog: AGF text, or GlassCatalog JSON when the file
 * ends in .json. Recorded in the project. out_json (nullable) receives
 * {"name", "source", "path", "glass_count"}. */
MKStatus mk_session_load_catalog(MKSession *session, const char *path, char **out_json);

/* JSON array of {"name", "source": "builtin"|"user", "path", "glass_count"}. */
MKStatus mk_session_list_catalogs(const MKSession *session, char **out_json);

/* JSON array of {"catalog", "name", "nd", "vd", "min_um", "max_um"}.
 * catalog (nullable): NULL lists every catalog. */
MKStatus mk_session_list_glasses(const MKSession *session, const char *catalog,
                                 char **out_json);

/* ---- Analyses and jobs -------------------------------------------------- */

/* Queue `analysis` on the current snapshot. settings_json (nullable) is a
 * partial settings object (defaults fill the rest). view_key (nullable): a
 * newer job with the same key in this session cancels the older one.
 * Unknown analyses, unavailable analyses and invalid settings fail here,
 * not as failed jobs. *out_job receives a non-zero job id. */
MKStatus mk_session_submit_analysis(const MKSession *session, const char *analysis,
                                    const char *settings_json, const char *view_key,
                                    uint64_t *out_job);

/* Same, from {"analysis", "settings", "view_key", "backend",
 * "allow_backend_fallback", "tier", "aim_stop", "route"}; every key but
 * "analysis" is optional. "backend" is {"kind": "cpu_reference"},
 * {"kind": "gpu", "name": "metal"} or {"kind": "auto"} (chosen per request
 * by the engine's cost model: a GPU when it implements the analysis and is
 * predicted to be faster, else the CPU reference); without it the engine's
 * default backend applies. The result's "backend" object says what was used
 * and why. "allow_backend_fallback": run on the CPU reference, with a
 * run_info warning, when the requested backend is unavailable, instead of
 * failing. "tier": "design" (default) or "preview". "aim_stop": "requested"
 * or "conditioned"; null or absent takes the engine configuration's.
 * "route": route options of this request (the object of the engine
 * configuration's "route"; keys left out take their defaults, not the
 * configuration's values); null or absent takes the engine
 * configuration's. Invalid route options fail here. */
MKStatus mk_session_submit_analysis_json(const MKSession *session, const char *request_json,
                                         uint64_t *out_job);

/* out_state / out_progress (nullable): MKJobState and progress in [0, 1]. */
MKStatus mk_job_status(const MKEngine *engine, uint64_t job, MKJobState *out_state,
                       double *out_progress);

/* Current stage text of the job. */
MKStatus mk_job_stage(const MKEngine *engine, uint64_t job, char **out_stage);

/* {"id", "state", "progress", "stage", "label", "view_key", "error",
 *  "superseded_by", "elapsed_ms"} */
MKStatus mk_job_status_json(const MKEngine *engine, uint64_t job, char **out_json);

/* Stop a job of an analysis listed with "stoppable": true
 * (mk_engine_list_analyses), keeping the best-so-far result as DONE with
 * result.stopped = true. Queued jobs start and return their initial result.
 * MK_ERR_UNSUPPORTED for other active jobs. No-op for finished jobs.
 * Cancellation/supersession still discards the result, even after stop. */
MKStatus mk_job_stop(const MKEngine *engine, uint64_t job);

/* Request cancellation. A queued job is cancelled at once; a running job
 * stops at its next cancellation check. No-op for finished jobs. */
MKStatus mk_job_cancel(const MKEngine *engine, uint64_t job);

/* Block until the job finishes. timeout_ms < 0 waits forever. Returns MK_OK
 * with the terminal state, or MK_ERR_TIMEOUT with the current state.
 * out_state is nullable. */
MKStatus mk_job_wait(const MKEngine *engine, uint64_t job, int64_t timeout_ms,
                     MKJobState *out_state);

/* Take the result of a finished job and forget the job.
 *   MK_OK:             *out_json = {"run_info", "library_hash", "backend",
 *                      "result_version", "result",
 *                      "buffers": [{"index", "name", "dtype", "shape"}]};
 *                      *out_buffers and
 *                      *out_count = typed buffers (NULL and 0 when none).
 *   MK_ERR_CANCELLED:  the job was cancelled or superseded.
 *   MK_ERR_NOT_READY:  still queued/running; the job is kept.
 *   MK_ERR_NOT_FOUND:  unknown or already taken.
 * A job that FAILED returns the status of its own error, the one
 * mk_session_run_analysis_json returns for the same request (for example
 * MK_ERR_NOT_FOUND for an unknown glass, MK_ERR_COMPUTE for a trace or
 * numerical failure, MK_ERR_INVALID_MODEL, MK_ERR_UNSUPPORTED), with its
 * message in mk_last_error; MK_ERR_JOB_FAILED when the job panicked. The
 * failed job is forgotten too. mk_job_status tells a failed job
 * (MK_JOB_FAILED) from an unknown one before the result is taken. */
MKStatus mk_job_take_result(const MKEngine *engine, uint64_t job, char **out_json,
                            MKBuffer **out_buffers, size_t *out_count);

/* Cancel if needed and forget a job without taking its result. */
MKStatus mk_job_release(const MKEngine *engine, uint64_t job);

/* ---- ABI 1.1 ------------------------------------------------------------ */

/* One analysis description (an element of mk_engine_list_analyses). */
MKStatus mk_engine_analysis_info(const MKEngine *engine, const char *name, char **out_json);

/* Default backend for requests that name none: {"kind": "auto"} (the
 * initial default unless the configuration names another),
 * {"kind": "cpu_reference"} or {"kind": "gpu", "name": ...}.
 * An unavailable explicit backend is refused (MK_ERR_UNSUPPORTED). */
MKStatus mk_engine_set_default_backend(const MKEngine *engine, const char *backend_json);

/* Run an analysis in the calling thread (blocking; no job, no callback, not
 * cancellable) on the current snapshot. request_json as for
 * mk_session_submit_analysis_json (view_key ignored). Outputs as for
 * mk_job_take_result; errors of the run are returned directly. The session
 * is locked only while the request is checked, not during the run. For
 * quick analyses; submit long ones as jobs. */
MKStatus mk_session_run_analysis_json(const MKSession *session, const char *request_json,
                                      char **out_json, MKBuffer **out_buffers,
                                      size_t *out_count);

/* The validate report {"ok", "errors", "warnings", "issues"} of the current
 * snapshot (non-finite numbers, aperture value, weights, mechanics, materials,
 * coatings, tracer set-up, chief and marginal rays; docs/RESULTS.md).
 * Blocking; runs without holding the session lock. */
MKStatus mk_session_validate_json(const MKSession *session, char **out_json);

/* Apply the system's solves and commit the solved system as a new snapshot.
 * settings_json (nullable): solve settings (see the "solves" analysis).
 * out_report (nullable): the solve report; out_hash (nullable): the new
 * snapshot hash (unchanged when every solve already held). On failure the
 * session is unchanged. Blocking and not cancellable, but the solve runs
 * without holding the session lock; if the session changed meanwhile nothing
 * is committed and MK_ERR_CANCELLED (superseded) is returned. */
MKStatus mk_session_apply_solves(MKSession *session, const char *settings_json,
                                 char **out_report, char **out_hash);

/* Write the current project to path, or to "<project file>.autosave" when
 * path is NULL, without changing the session (its path and dirty flag stay).
 * An untitled session needs a path. out_path (nullable): the path written.
 * Recover by opening the autosave file with mk_session_open. */
MKStatus mk_session_autosave(const MKSession *session, const char *path, char **out_path);

/* Current value of a parameter: {"value": number | "inf" | "-inf" | null}
 * (an automatic semi-diameter reads its resolved layout radius). path_json
 * is a ParamPath as in mk_session_apply_param_edit. */
MKStatus mk_session_get_param(const MKSession *session, const char *path_json,
                              char **out_json);

/* ---- Lens prescription import (ABI 1.3) --------------------------------- */

/* Import .zmx or .seq text (format "zmx" or "seq"; UTF-8, UTF-16 LE/BE with
 * or without a byte-order mark, or Latin-1) without a session. Glasses
 * resolve against the built-in catalog. out_json receives {"format",
 * "encoding", "system", "warnings": [{"kind", "line", "surface",
 * "message"}]}. Content that does not map is a warning, never an error;
 * text without surfaces is MK_ERR_PARSE. */
MKStatus mk_import_prescription(const uint8_t *bytes, size_t len, const char *format,
                                char **out_json);

/* Replace the session's system with one imported from a .zmx or .seq file
 * (format by extension), resolving glasses against the session's catalogs.
 * out_json (nullable) receives the mk_import_prescription object plus
 * "snapshot_hash". On failure the session is unchanged. */
MKStatus mk_session_import_prescription(MKSession *session, const char *path,
                                        char **out_json);

/* ---- GPU kernel preparation (ABI 1.4) ----------------------------------- */

/* Start creating the GPU's compute kernels in the background, so the first
 * GPU analysis does not wait for shader compilation. Opens the first
 * registered GPU. cache_dir (nullable): directory where the compiled kernels
 * are saved, so later launches load them instead of compiling. Returns at
 * once; a second call is a no-op. */
MKStatus mk_engine_prepare_gpu(const MKEngine *engine, const char *cache_dir);

/* out_json receives the preparation's state, by "state":
 *   "idle"; "preparing" {device, done, total};
 *   "ready" {device, kernels, from_cache, seconds, and "warning" when the
 *   saved kernels could not be read or written};
 *   "unavailable" {reason}; "failed" {device, message} (kernels then
 *   compile on first use). */
MKStatus mk_engine_gpu_preparation_json(const MKEngine *engine, char **out_json);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MKLIGHT_H */
