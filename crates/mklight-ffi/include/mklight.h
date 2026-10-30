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
 *  - Buffer arrays returned by mk_job_take_result() are released with
 *    mk_buffer_free().
 *  - Structured data is JSON. Infinite lengths are the strings "inf"/"-inf".
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
#define MK_ABI_VERSION_MINOR 2
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
#define MK_ERR_INVALID_ARGUMENT 1 /* NULL/invalid argument, bad UTF-8, bad settings */
#define MK_ERR_PARSE 2            /* malformed JSON, project or catalog file */
#define MK_ERR_INVALID_MODEL 3    /* optical system fails validation */
#define MK_ERR_NOT_FOUND 4        /* unknown job, analysis, glass, catalog */
#define MK_ERR_IO 5               /* file system error */
#define MK_ERR_UNSUPPORTED 6      /* analysis or backend not available */
#define MK_ERR_CANCELLED 7        /* job was cancelled */
#define MK_ERR_TIMEOUT 8          /* mk_job_wait timed out */
#define MK_ERR_NOT_READY 9        /* job has not finished */
#define MK_ERR_JOB_FAILED 10      /* job ended with an error */
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
 * array and stay valid until mk_buffer_free(). */
typedef struct MKBuffer {
    const char *name;    /* NUL-terminated UTF-8 */
    const void *data;    /* `len` elements of `dtype`, aligned; non-NULL */
    size_t len;          /* element count (not bytes) */
    MKDType dtype;
    uint32_t ndim;
    const size_t *shape; /* `ndim` extents whose product is `len` */
} MKBuffer;

/* Free an array of `count` buffers from mk_job_take_result(). NULL is
 * ignored. `count` must be the count returned with the array. */
void mk_buffer_free(MKBuffer *buffers, size_t count);

/* ---- Handles ------------------------------------------------------------ */

typedef struct MKEngine MKEngine;   /* thread-safe */
typedef struct MKSession MKSession; /* thread-safe; calls are serialised */

/* ---- Engine ------------------------------------------------------------- */

/* config_json (nullable): {"worker_threads": 0, "max_retained_jobs": 256,
 * "resolved_cache_capacity": 8, "progress_interval_ms": 50, "app_name": "",
 * "app_version": "", "default_backend": {"kind": "cpu_reference"}}; every
 * key is optional. */
MKStatus mk_engine_create(const char *config_json, MKEngine **out_engine);

/* Clears the job callback (waiting for calls in progress), then releases
 * the handle. Worker threads stop when all sessions are destroyed too. */
void mk_engine_destroy(MKEngine *engine);

/* JSON array of analysis descriptions (name, title, category, description,
 * available, unavailable_reason, settings_schema, default_settings,
 * result_schema, result_version, gpu_path). The result schemas, and every
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

/* Open a .mklight project (older schema versions are migrated). Missing user
 * catalogs do not fail the open: see "open_warnings" in mk_session_info_json. */
MKStatus mk_session_open(const MKEngine *engine, const char *path, MKSession **out_session);

/* Jobs already submitted keep running; their results stay available through
 * the engine. */
void mk_session_destroy(MKSession *session);

/* Atomic save (temporary file + rename). path (nullable): NULL saves to the
 * path the session was opened from or last saved to. */
MKStatus mk_session_save(MKSession *session, const char *path);

/* {"session_id", "path", "dirty", "snapshot_hash", "library_hash", "name",
 *  "open_warnings", "migration", "catalogs"} */
MKStatus mk_session_info_json(const MKSession *session, char **out_json);

/* The whole project document (same schema as the .mklight file). */
MKStatus mk_session_get_project_json(const MKSession *session, char **out_json);

/* The current OpticalSystem. */
MKStatus mk_session_get_system_json(const MKSession *session, char **out_json);

/* Replace the OpticalSystem. Invalid systems are rejected (MK_ERR_PARSE /
 * MK_ERR_INVALID_MODEL) and leave the session unchanged. out_hash
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
 * "allow_backend_fallback"}. "backend" is {"kind": "cpu_reference"},
 * {"kind": "gpu", "name": "metal"} or {"kind": "auto"} (a GPU when one
 * implements the analysis and qualifies for the system, else the CPU
 * reference); the result's "backend" object says what was used and why. */
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

/* Stop optimize/tolerance, keeping the best-so-far result as DONE with
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
 *   MK_ERR_JOB_FAILED: the job failed (message in mk_last_error).
 *   MK_ERR_CANCELLED:  the job was cancelled or superseded.
 *   MK_ERR_NOT_READY:  still queued/running; the job is kept.
 *   MK_ERR_NOT_FOUND:  unknown or already taken. */
MKStatus mk_job_take_result(const MKEngine *engine, uint64_t job, char **out_json,
                            MKBuffer **out_buffers, size_t *out_count);

/* Cancel if needed and forget a job without taking its result. */
MKStatus mk_job_release(const MKEngine *engine, uint64_t job);

/* ---- ABI 1.1 ------------------------------------------------------------ */

/* One analysis description (an element of mk_engine_list_analyses). */
MKStatus mk_engine_analysis_info(const MKEngine *engine, const char *name, char **out_json);

/* Default backend for requests that name none: {"kind": "cpu_reference"}
 * (the initial default), {"kind": "auto"} or {"kind": "gpu", "name": ...}.
 * An unavailable explicit backend is refused (MK_ERR_UNSUPPORTED). */
MKStatus mk_engine_set_default_backend(const MKEngine *engine, const char *backend_json);

/* Migrate a project file in place to the current schema version. backup != 0
 * keeps the original as "<file>.v<N>.bak"; dry_run != 0 writes nothing.
 * out_json (nullable): {"path", "from_version", "to_version", "notes",
 * "migrated", "backup", "dry_run"}. */
MKStatus mk_engine_migrate_project(const MKEngine *engine, const char *path, int32_t backup,
                                   int32_t dry_run, char **out_json);

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
 * (null: no value, e.g. an automatic semi-diameter). path_json is a
 * ParamPath as in mk_session_apply_param_edit. */
MKStatus mk_session_get_param(const MKSession *session, const char *path_json,
                              char **out_json);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MKLIGHT_H */
