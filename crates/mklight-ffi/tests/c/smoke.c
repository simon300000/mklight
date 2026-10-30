/*
 * End-to-end smoke test of the MKLight C ABI, linked against the static
 * library. Built and run by tests/c_smoke.rs (or by hand:
 *   clang -std=c11 -Iinclude tests/c/smoke.c target/debug/libmklight_ffi.a -o smoke
 *   ./smoke /tmp/somedir
 * ). Exits 0 on success; prints the failing check otherwise.
 */
#include "mklight.h"

#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int g_checks = 0;

#define CHECK(cond)                                                                \
    do {                                                                           \
        g_checks++;                                                                \
        if (!(cond)) {                                                             \
            fprintf(stderr, "smoke.c:%d: CHECK failed: %s\n  last error: %s\n",    \
                    __LINE__, #cond, mk_last_error());                             \
            exit(1);                                                               \
        }                                                                          \
    } while (0)

#define EXPECT(call, status)                                                       \
    do {                                                                           \
        g_checks++;                                                                \
        MKStatus st_ = (call);                                                     \
        if (st_ != (status)) {                                                     \
            fprintf(stderr, "smoke.c:%d: %s returned %d, expected %d\n"            \
                            "  last error: %s\n",                                  \
                    __LINE__, #call, (int)st_, (int)(status), mk_last_error());    \
            exit(1);                                                               \
        }                                                                          \
    } while (0)

#define OK(call) EXPECT(call, MK_OK)

static void sleep_ms(long ms) {
    struct timespec ts = {ms / 1000, (ms % 1000) * 1000000L};
    nanosleep(&ts, NULL);
}

/* ---- callback bookkeeping (called on engine threads) ---- */
static atomic_int g_events = 0;
static atomic_int g_progress_events = 0;
static atomic_int g_terminal_events = 0;

static void on_job_event(void *user, uint64_t job, MKJobState state, double progress,
                         const char *stage) {
    (void)job;
    atomic_fetch_add((atomic_int *)user, 1);
    atomic_fetch_add(&g_events, 1);
    if (state == MK_JOB_RUNNING && progress > 0.0 && stage != NULL && stage[0] != '\0')
        atomic_fetch_add(&g_progress_events, 1);
    if (state == MK_JOB_DONE || state == MK_JOB_FAILED || state == MK_JOB_CANCELLED)
        atomic_fetch_add(&g_terminal_events, 1);
}

/* Submit, wait, take; returns the JSON (caller frees) and buffers. */
static char *run_job(MKEngine *eng, MKSession *s, const char *analysis, const char *settings,
                     MKBuffer **bufs, size_t *n) {
    uint64_t job = 0;
    OK(mk_session_submit_analysis(s, analysis, settings, NULL, &job));
    CHECK(job != 0);
    MKJobState state = -1;
    OK(mk_job_wait(eng, job, 30000, &state));
    CHECK(state == MK_JOB_DONE);
    char *json = NULL;
    OK(mk_job_take_result(eng, job, &json, bufs, n));
    CHECK(json != NULL);
    return json;
}

/* Replace the first occurrence of `from` by `to` (new malloc'd string). */
static char *replace(const char *s, const char *from, const char *to) {
    const char *p = strstr(s, from);
    if (!p) return NULL;
    size_t a = (size_t)(p - s), lf = strlen(from), lt = strlen(to), ls = strlen(s);
    char *out = malloc(ls - lf + lt + 1);
    memcpy(out, s, a);
    memcpy(out + a, to, lt);
    memcpy(out + a + lt, p + lf, ls - a - lf + 1);
    return out;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: smoke <scratch-dir>\n");
        return 2;
    }
    const char *dir = argv[1];
    char path[4096];

    /* ---- versioning ---- */
    CHECK(mk_abi_version() == MK_ABI_VERSION);
    CHECK((mk_abi_version() >> 16) == MK_ABI_VERSION_MAJOR);
    CHECK(strlen(mk_engine_version()) > 0);
    CHECK(mk_last_error() != NULL);

    /* ---- engine creation and its error paths ---- */
    MKEngine *eng = (MKEngine *)1;
    EXPECT(mk_engine_create("{not json", &eng), MK_ERR_PARSE);
    CHECK(eng == NULL);
    CHECK(strlen(mk_last_error()) > 0);
    EXPECT(mk_engine_create(NULL, NULL), MK_ERR_INVALID_ARGUMENT);
    EXPECT(mk_engine_create("{\"worker_threads\": \"many\"}", &eng), MK_ERR_PARSE);
    OK(mk_engine_create("{\"worker_threads\": 2, \"progress_interval_ms\": 5, "
                        "\"app_name\": \"smoke\", \"app_version\": \"1.0\"}",
                        &eng));
    CHECK(eng != NULL);
    CHECK(strcmp(mk_last_error(), "") == 0);

    atomic_int user_count = 0;
    OK(mk_engine_set_job_callback(eng, on_job_event, &user_count));

    char *json = NULL;
    OK(mk_engine_list_analyses(eng, &json));
    CHECK(strstr(json, "\"validate\"") && strstr(json, "\"system_summary\"") &&
          strstr(json, "\"glass_info\"") && strstr(json, "\"settings_schema\""));
    mk_string_free(json);
    OK(mk_engine_list_backends(eng, &json));
    CHECK(strstr(json, "\"cpu_reference\""));
    mk_string_free(json);

    /* ---- session and edits ---- */
    MKSession *s = NULL;
    OK(mk_session_new(eng, &s));
    char *h0 = NULL;
    OK(mk_session_current_snapshot_hash(s, &h0));
    CHECK(strlen(h0) == 64);

    char *sys = NULL;
    OK(mk_session_get_system_json(s, &sys));
    CHECK(strstr(sys, "\"surfaces\""));
    EXPECT(mk_session_set_system_json(s, "{]", NULL), MK_ERR_PARSE);
    char *bad = replace(sys, "\"stop\":1", "\"stop\":0");
    CHECK(bad != NULL);
    char *hbad = (char *)1;
    EXPECT(mk_session_set_system_json(s, bad, &hbad), MK_ERR_INVALID_MODEL);
    CHECK(hbad == NULL);
    free(bad);

    char *h1 = NULL;
    OK(mk_session_apply_param_edit(
        s, "{\"path\": {\"param\": \"thickness\", \"surface\": 1}, \"value\": 12.5}", &h1));
    CHECK(strlen(h1) == 64 && strcmp(h0, h1) != 0);
    /* Atomic batch: the second edit fails, so nothing changes. */
    EXPECT(mk_session_apply_param_edit(
               s,
               "[{\"path\": {\"param\": \"thickness\", \"surface\": 1}, \"value\": 3},"
               " {\"path\": {\"param\": \"thickness\", \"surface\": 999}, \"value\": 1}]",
               NULL),
           MK_ERR_INVALID_ARGUMENT);
    EXPECT(mk_session_apply_param_edit(s, "42", NULL), MK_ERR_INVALID_ARGUMENT);
    char *h2 = NULL;
    OK(mk_session_current_snapshot_hash(s, &h2));
    CHECK(strcmp(h1, h2) == 0);
    mk_string_free(h2);

    OK(mk_session_set_project_section_json(s, "merit", "{\"operands\": []}"));
    EXPECT(mk_session_set_project_section_json(s, "system", "{}"), MK_ERR_INVALID_ARGUMENT);
    OK(mk_session_get_project_json(s, &json));
    CHECK(strstr(json, "\"schema_version\":1") && strstr(json, "\"operands\""));
    mk_string_free(json);

    /* ---- system_summary with buffers ---- */
    MKBuffer *bufs = NULL;
    size_t nb = 0;
    json = run_job(eng, s, "system_summary", NULL, &bufs, &nb);
    CHECK(strstr(json, "\"run_info\"") && strstr(json, "\"snapshot_hash\""));
    CHECK(strstr(json, h1) != NULL); /* stamped with the snapshot it ran on */
    CHECK(nb == 2 && bufs != NULL);
    CHECK(strcmp(bufs[0].name, "n_after") == 0);
    CHECK(bufs[0].dtype == MK_DTYPE_F64 && bufs[0].ndim == 2);
    CHECK(bufs[0].shape[0] * bufs[0].shape[1] == bufs[0].len);
    CHECK(bufs[0].shape[0] == 3 && bufs[0].shape[1] == 3);
    {
        const double *n = (const double *)bufs[0].data;
        for (size_t i = 0; i < bufs[0].len; i++) CHECK(n[i] == 1.0); /* all air */
        const double *v = (const double *)bufs[1].data;               /* vertex_global */
        CHECK(fabs(v[2 * 3 + 2] - 12.5) < 1e-12); /* image vertex z = edited thickness */
    }
    mk_buffer_free(bufs, nb);
    mk_string_free(json);

    /* ---- glass catalogs ---- */
    OK(mk_session_list_catalogs(s, &json));
    CHECK(strstr(json, "\"BUILTIN\""));
    mk_string_free(json);
    OK(mk_session_list_glasses(s, "builtin", &json));
    CHECK(strstr(json, "\"N-BK7\""));
    mk_string_free(json);
    EXPECT(mk_session_list_glasses(s, "NOPE", &json), MK_ERR_NOT_FOUND);
    CHECK(json == NULL);

    snprintf(path, sizeof path, "%s/mycat.json", dir);
    {
        FILE *f = fopen(path, "w");
        CHECK(f != NULL);
        fputs("{\"name\": \"MYCAT\", \"glasses\": {\"FAKE1\": {\"name\": \"FAKE1\","
              " \"dispersion\": {\"formula\": \"cauchy\", \"a\": 1.6, \"b\": 0.0, \"c\": 0.0},"
              " \"nd\": 1.6, \"vd\": 1e9, \"min_um\": 0.3, \"max_um\": 2.0}}}",
              f);
        fclose(f);
    }
    OK(mk_session_load_catalog(s, path, &json));
    CHECK(strstr(json, "\"MYCAT\"") && strstr(json, "\"glass_count\":1"));
    mk_string_free(json);
    OK(mk_session_list_glasses(s, "mycat", &json));
    CHECK(strstr(json, "\"FAKE1\""));
    mk_string_free(json);
    snprintf(path, sizeof path, "%s/does-not-exist.agf", dir);
    EXPECT(mk_session_load_catalog(s, path, NULL), MK_ERR_IO);

    json = run_job(eng, s, "glass_info",
                   "{\"name\": \"N-BK7\", \"wavelengths_um\": [0.5875618], \"curve_samples\": 5}",
                   &bufs, &nb);
    CHECK(strstr(json, "\"catalog\":\"BUILTIN\""));
    CHECK(nb == 1 && bufs[0].len == 10 && bufs[0].ndim == 2 && bufs[0].shape[1] == 2);
    {
        const double *c = (const double *)bufs[0].data;
        CHECK(fabs(c[0] - 0.3) < 1e-12 && c[1] > 1.5 && c[1] < 1.6);
    }
    mk_buffer_free(bufs, nb);
    mk_string_free(json);

    /* ---- submission errors are reported synchronously ---- */
    uint64_t job = 0;
    EXPECT(mk_session_submit_analysis(s, "no_such_analysis", NULL, NULL, &job), MK_ERR_NOT_FOUND);
    EXPECT(mk_session_submit_analysis(s, "mtf", NULL, NULL, &job), MK_ERR_UNSUPPORTED);
    EXPECT(mk_session_submit_analysis(s, "rng_stream", "{\"count\": 0}", NULL, &job),
           MK_ERR_INVALID_ARGUMENT);
    EXPECT(mk_session_submit_analysis(s, "rng_stream", "{\"bogus\": 1}", NULL, &job),
           MK_ERR_INVALID_ARGUMENT);
    EXPECT(mk_session_submit_analysis(s, "rng_stream", "{oops", NULL, &job), MK_ERR_PARSE);
    EXPECT(mk_session_submit_analysis(s, NULL, NULL, NULL, &job), MK_ERR_INVALID_ARGUMENT);
    CHECK(job == 0);
    EXPECT(mk_session_submit_analysis_json(
               s, "{\"analysis\": \"validate\", \"backend\": {\"kind\": \"gpu\", \"name\": \"metal\"}}",
               &job),
           MK_ERR_UNSUPPORTED);
    OK(mk_session_submit_analysis_json(
        s,
        "{\"analysis\": \"validate\", \"backend\": {\"kind\": \"gpu\", \"name\": \"metal\"},"
        " \"allow_backend_fallback\": true}",
        &job));
    MKJobState state = -1;
    OK(mk_job_wait(eng, job, 30000, &state));
    CHECK(state == MK_JOB_DONE);
    OK(mk_job_take_result(eng, job, &json, &bufs, &nb));
    CHECK(strstr(json, "\"backend\":\"cpu_reference\"") && strstr(json, "gpu:metal"));
    CHECK(strstr(json, "\"ok\":true"));
    CHECK(bufs == NULL && nb == 0);
    mk_string_free(json);
    /* Taken jobs are forgotten. */
    EXPECT(mk_job_take_result(eng, job, &json, &bufs, &nb), MK_ERR_NOT_FOUND);
    EXPECT(mk_job_status(eng, job, &state, NULL), MK_ERR_NOT_FOUND);
    EXPECT(mk_job_cancel(eng, 987654321), MK_ERR_NOT_FOUND);

    /* ---- a long job: progress, stage, timeout, not-ready, cancel ---- */
    OK(mk_session_submit_analysis(
        s, "rng_stream", "{\"count\": 5000000, \"chunk\": 1000, \"delay_ms\": 1}", NULL, &job));
    EXPECT(mk_job_wait(eng, job, 20, &state), MK_ERR_TIMEOUT);
    CHECK(state == MK_JOB_QUEUED || state == MK_JOB_RUNNING);
    double progress = 0.0;
    for (int i = 0; i < 3000 && progress <= 0.0; i++) {
        OK(mk_job_status(eng, job, &state, &progress));
        sleep_ms(1);
    }
    CHECK(state == MK_JOB_RUNNING && progress > 0.0 && progress < 1.0);
    char *stage = NULL;
    OK(mk_job_stage(eng, job, &stage));
    CHECK(strcmp(stage, "drawing") == 0);
    mk_string_free(stage);
    EXPECT(mk_job_take_result(eng, job, &json, &bufs, &nb), MK_ERR_NOT_READY);
    CHECK(json == NULL && bufs == NULL && nb == 0);
    OK(mk_job_cancel(eng, job));
    OK(mk_job_wait(eng, job, 30000, &state));
    CHECK(state == MK_JOB_CANCELLED);
    EXPECT(mk_job_take_result(eng, job, &json, &bufs, &nb), MK_ERR_CANCELLED);

    /* ---- supersede by view key ---- */
    uint64_t slow = 0, fast = 0;
    OK(mk_session_submit_analysis(s, "rng_stream",
                                  "{\"count\": 5000000, \"chunk\": 1000, \"delay_ms\": 1}",
                                  "spot-window", &slow));
    OK(mk_session_submit_analysis(s, "rng_stream", "{\"count\": 16, \"seed\": 3}", "spot-window",
                                  &fast));
    OK(mk_job_wait(eng, slow, 30000, &state));
    CHECK(state == MK_JOB_CANCELLED);
    OK(mk_job_status_json(eng, slow, &json));
    {
        char expect[64];
        snprintf(expect, sizeof expect, "\"superseded_by\":%llu", (unsigned long long)fast);
        CHECK(strstr(json, expect) != NULL);
    }
    mk_string_free(json);
    OK(mk_job_release(eng, slow));
    OK(mk_job_wait(eng, fast, 30000, &state));
    CHECK(state == MK_JOB_DONE);
    OK(mk_job_take_result(eng, fast, &json, &bufs, &nb));
    CHECK(nb == 1 && bufs[0].dtype == MK_DTYPE_F64 && bufs[0].len == 16);
    CHECK(strstr(json, "\"seed\":3"));
    {
        const double *u = (const double *)bufs[0].data;
        for (size_t i = 0; i < 16; i++) CHECK(u[i] > 0.0 && u[i] < 1.0);
    }
    mk_buffer_free(bufs, nb);
    mk_string_free(json);

    /* ---- save, reopen ---- */
    snprintf(path, sizeof path, "%s/smoke.mklight", dir);
    EXPECT(mk_session_save(s, NULL), MK_ERR_INVALID_ARGUMENT); /* never saved: no path */
    OK(mk_session_save(s, path));
    OK(mk_session_info_json(s, &json));
    CHECK(strstr(json, "\"dirty\":false"));
    mk_string_free(json);
    MKSession *s2 = NULL;
    OK(mk_session_open(eng, path, &s2));
    char *h3 = NULL;
    OK(mk_session_current_snapshot_hash(s2, &h3));
    CHECK(strcmp(h1, h3) == 0);
    mk_string_free(h3);
    OK(mk_session_get_project_json(s2, &json));
    CHECK(strstr(json, "\"app_name\":\"smoke\"") && strstr(json, "\"MYCAT\""));
    mk_string_free(json);
    OK(mk_session_info_json(s2, &json));
    CHECK(strstr(json, "\"open_warnings\":[]"));
    mk_string_free(json);
    snprintf(path, sizeof path, "%s/missing.mklight", dir);
    MKSession *s3 = (MKSession *)1;
    EXPECT(mk_session_open(eng, path, &s3), MK_ERR_IO);
    CHECK(s3 == NULL);

    /* ---- a job that fails (unknown glass is caught when resolving) ---- */
    char *bad_glass = replace(sys, "\"material\":{\"type\":\"air\"}",
                              "\"material\":{\"type\":\"catalog\",\"name\":\"NOPE-42\"}");
    CHECK(bad_glass != NULL);
    OK(mk_session_set_system_json(s2, bad_glass, NULL));
    free(bad_glass);
    OK(mk_session_submit_analysis(s2, "system_summary", NULL, NULL, &job));
    OK(mk_job_wait(eng, job, 30000, &state));
    CHECK(state == MK_JOB_FAILED);
    OK(mk_job_status_json(eng, job, &json));
    CHECK(strstr(json, "NOPE-42"));
    mk_string_free(json);
    EXPECT(mk_job_take_result(eng, job, &json, &bufs, &nb), MK_ERR_JOB_FAILED);
    CHECK(strstr(mk_last_error(), "NOPE-42") != NULL);
    /* validate reports it as an issue instead of failing. */
    json = run_job(eng, s2, "validate", NULL, &bufs, &nb);
    CHECK(strstr(json, "\"ok\":false") && strstr(json, "\"code\":\"material\""));
    mk_string_free(json);

    /* ---- callback: lifecycle and progress were delivered ---- */
    OK(mk_engine_set_job_callback(eng, NULL, NULL));
    int events_after_clear = atomic_load(&g_events);
    CHECK(atomic_load(&user_count) == events_after_clear);
    CHECK(atomic_load(&g_progress_events) >= 1);
    /* 8 jobs finished; the last terminal event may still be on its way when
     * mk_job_wait returns, so allow one to be missing. */
    CHECK(atomic_load(&g_terminal_events) >= 7);
    OK(mk_session_submit_analysis(s, "validate", NULL, NULL, &job));
    OK(mk_job_wait(eng, job, 30000, &state));
    OK(mk_job_release(eng, job));
    CHECK(atomic_load(&g_events) == events_after_clear); /* cleared: no more calls */

    /* ---- teardown (sessions may outlive the engine handle) ---- */
    mk_string_free(sys);
    mk_string_free(h0);
    mk_string_free(h1);
    mk_session_destroy(s);
    mk_engine_destroy(eng);
    mk_session_destroy(s2);
    mk_string_free(NULL);
    mk_buffer_free(NULL, 0);

    printf("smoke: %d checks passed\n", g_checks);
    return 0;
}
