using System.Text.Json.Nodes;
using System.Threading.Channels;

namespace MKLight.Engine;

/// <summary>
/// The Rust engine over its C ABI (<c>crates/mklight-ffi</c>, docs/FFI.md, docs/APP.md "FFIEngine").
/// <para>
/// An <see cref="FFIEngine"/> is one engine handle plus one <b>session</b> (the current
/// system, user glass catalogs, library hash). <see cref="MakeDocumentEngine"/> gives each
/// document its own session on the shared engine.
/// </para>
/// <para>
/// Lifetimes: <see cref="Dispose"/> destroys this engine's session (<c>mk_session_destroy</c>);
/// a forgotten one is destroyed by its finalizer. The engine handle
/// (<c>mk_engine_destroy</c>) is freed once no <see cref="FFIEngine"/>, session or running job
/// refers to it. Jobs already submitted keep running after their session is gone.
/// </para>
/// <para>
/// Threading: use it from any thread. The C engine and sessions are thread-safe. The
/// asynchronous members never block their caller: their C calls (including waits for the
/// session lock) run on the thread pool. Job events arrive on the engine's
/// <c>mklight-events</c> thread and are forwarded to each job's progress sequence;
/// consumers continue on their own context. A job runs against the snapshot it was
/// submitted with: the session's system is set to that snapshot and the job submitted in
/// one locked step. On the caller's thread run: construction and
/// <see cref="NewSession"/>/<see cref="MakeDocumentEngine"/>, <see cref="AnalysisJob.Cancel"/>
/// and <see cref="AnalysisJob.Stop"/>, <see cref="PrepareGpu"/>, <see cref="GetGpuPreparation"/>,
/// <see cref="GlassFromSpec"/>, <see cref="Agf"/> and <see cref="ImportPrescription"/>.
/// </para>
/// <para>
/// When the library is missing (or its ABI major version differs), the constructor throws
/// <see cref="EngineException"/> of kind <c>LibraryUnavailable</c>; <see cref="IsAvailable"/>
/// tells beforehand without throwing.
/// </para>
/// </summary>
public sealed class FFIEngine : IEngineClient, IDisposable
{
    readonly FFIEngineCore core;
    readonly SessionHandle session;
    readonly Lock sessionLock = new();

    /// <summary>Whether <c>mklight_ffi.dll</c> loads and its ABI major version matches this app's.</summary>
    public static bool IsAvailable => CAbi.LoadFailure is null;

    /// <summary>Why the engine library is unavailable, or <c>null</c> when it is available.</summary>
    public static string? UnavailableReason => CAbi.LoadFailure;

    /// <summary>
    /// Creates an engine with one session. <paramref name="configuration"/>: engine keys
    /// (docs/FFI.md, <c>mk_engine_create</c>), e.g. <c>worker_threads</c>,
    /// <c>max_retained_jobs</c>, <c>progress_interval_ms</c>.
    /// </summary>
    public FFIEngine(JsonObject? configuration = null)
    {
        var config = (JsonObject?)configuration?.DeepClone() ?? [];
        config["app_name"] ??= "MKLight";
        core = new FFIEngineCore(config.ToJsonString());
        session = CAbi.SessionNew(core.Handle);
    }

    FFIEngine(FFIEngineCore core, SessionHandle session)
    {
        this.core = core;
        this.session = session;
    }

    internal FFIEngineCore Core => core;
    internal SessionHandle Session => session;

    public EngineInfo Info => new("MKLight engine", core.Version, IsMock: false);

    /// <summary>A new session on the same engine.</summary>
    public FFIEngine NewSession() => new(core, CAbi.SessionNew(core.Handle));

    public IEngineClient MakeDocumentEngine()
    {
        try
        {
            return NewSession();
        }
        catch (EngineException)
        {
            return this;
        }
    }

    /// <summary>Destroys this engine's session. Jobs already submitted keep running.</summary>
    public void Dispose() => session.Dispose();

    /// <summary>Runs <paramref name="body"/> holding this session's lock, which makes a sequence of calls atomic.</summary>
    internal T Locked<T>(Func<SessionHandle, T> body)
    {
        lock (sessionLock) return body(session);
    }

    /// <summary>Runs a blocking C call off the caller's thread.</summary>
    static Task<T> OffThread<T>(Func<T> body) => Task.Run(body);

    Task<T> InSession<T>(Func<SessionHandle, T> body) => OffThread(() => Locked(body));

    /// <summary>Sets the session's system to the snapshot's unless it holds it already (call while locked).</summary>
    static void Hold(SessionHandle s, ModelSnapshot snapshot)
    {
        var current = CAbi.String("mk_session_current_snapshot_hash", (out nint p) => Native.mk_session_current_snapshot_hash(s, out p));
        if (current != snapshot.Hash.Hex) SetSystem(s, snapshot.SystemJson);
    }

    static string SetSystem(SessionHandle s, string json) =>
        CAbi.String("mk_session_set_system_json", (out nint p) => Native.mk_session_set_system_json(s, json, out p));

    static string LibraryHash(SessionHandle s) =>
        CAbi.String("mk_session_current_library_hash", (out nint p) => Native.mk_session_current_library_hash(s, out p));

    // GPU kernel preparation

    /// <summary>Starts preparing the GPU kernels; a refusal is reported by <see cref="GetGpuPreparation"/> as failed.</summary>
    public void PrepareGpu(string? cacheDirectory)
    {
        try
        {
            CAbi.Check(Native.mk_engine_prepare_gpu(core.Handle, cacheDirectory), "mk_engine_prepare_gpu");
            core.GpuPreparationError = null;
        }
        catch (EngineException e)
        {
            core.GpuPreparationError = e.Detail;
        }
    }

    /// <summary>The engine's state; <c>Failed</c> when <see cref="PrepareGpu"/> was refused or the state cannot be read.</summary>
    public GpuPreparation GetGpuPreparation()
    {
        if (core.GpuPreparationError is { } refusal) return new GpuPreparation.Failed("", refusal);
        try
        {
            return GpuPreparation.FromJson(
                CAbi.String("mk_engine_gpu_preparation_json", (out nint p) => Native.mk_engine_gpu_preparation_json(core.Handle, out p)));
        }
        catch (EngineException e)
        {
            return new GpuPreparation.Failed("", e.Detail);
        }
    }

    /// <summary>Records a refusal of the preparation (tests).</summary>
    internal void SetGpuPreparationError(string? message) => core.GpuPreparationError = message;

    /// <summary>The engine's compute backends (<c>mk_engine_list_backends</c>, read when the engine was created).</summary>
    public JsonNode? Backends() => Json.Parse(core.BackendsJson, "backend list");

    // Snapshots, validation, solves, parameters

    /// <summary>
    /// Sets the session's system (<c>mk_session_set_system_json</c>) and returns the engine's
    /// BLAKE3 hash and the library hash. A system the engine refuses (or cannot read as one)
    /// throws <c>InvalidModel</c> with the engine's message.
    /// </summary>
    public async Task<ModelSnapshot> OpenSnapshotAsync(string systemJson)
    {
        try
        {
            var (hash, library) = await InSession(s => (SetSystem(s, systemJson), LibraryHash(s))).ConfigureAwait(false);
            return new ModelSnapshot(systemJson, new SnapshotHash(hash), library);
        }
        catch (EngineException e) when (e.Code is EngineErrorCode.InvalidModel or EngineErrorCode.Parse or EngineErrorCode.InvalidArgument)
        {
            throw EngineException.InvalidModel([e.Code == EngineErrorCode.InvalidModel ? e.Detail : e.Message]);
        }
    }

    /// <summary>The engine's validate report (<c>mk_session_validate_json</c>) of the snapshot's system.</summary>
    public async Task<JsonNode?> ValidateAsync(ModelSnapshot snapshot)
    {
        var text = await InSession(s =>
        {
            Hold(s, snapshot);
            return CAbi.String("mk_session_validate_json", (out nint p) => Native.mk_session_validate_json(s, out p));
        }).ConfigureAwait(false);
        return Json.Parse(text, "validate report");
    }

    /// <summary>Solves the snapshot's system in the engine (<c>mk_session_apply_solves</c>) and reads the solved system back.</summary>
    public async Task<SolveApplication> ApplySolvesAsync(IReadOnlyDictionary<string, JsonNode?> settings, ModelSnapshot snapshot)
    {
        var settingsJson = Json.ObjectOf(settings).ToJsonString();
        var (report, hash, system, library) = await InSession(s =>
        {
            Hold(s, snapshot);
            var (r, h) = CAbi.ApplySolves(s, settingsJson);
            var json = CAbi.String("mk_session_get_system_json", (out nint p) => Native.mk_session_get_system_json(s, out p));
            return (r, h, json, LibraryHash(s));
        }).ConfigureAwait(false);
        return new SolveApplication(Json.Parse(report, "solve report"), new ModelSnapshot(system, new SnapshotHash(hash), library));
    }

    /// <summary>The current value of a parameter (<c>mk_session_get_param</c>; an automatic semi-diameter reads its resolved radius).</summary>
    public async Task<double> GetParamAsync(ModelSnapshot snapshot, string paramPathJson)
    {
        var text = await InSession(s =>
        {
            Hold(s, snapshot);
            return CAbi.String("mk_session_get_param", (out nint p) => Native.mk_session_get_param(s, paramPathJson, out p));
        }).ConfigureAwait(false);
        return Json.Parse(text, "parameter value").NfAt("value") ?? double.NaN;
    }

    /// <summary>
    /// Writes the session's project to <paramref name="path"/> (<c>mk_session_autosave</c>; an
    /// untitled session needs one) without changing the session. Returns the path written.
    /// </summary>
    public Task<string> AutosaveAsync(string? path) =>
        InSession(s => CAbi.String("mk_session_autosave", (out nint p) => Native.mk_session_autosave(s, path, out p)));

    // Catalogs

    /// <summary>Glasses of one catalog, or of every catalog.</summary>
    public async Task<IReadOnlyList<GlassInfo>> GlassesAsync(string? catalog = null)
    {
        var text = await ListGlassesAsync(catalog).ConfigureAwait(false);
        try
        {
            return (Json.Parse(text, "glass list").ArrayOrNull() ?? []).Select(GlassInfo.FromJson).ToArray();
        }
        catch (FormatException e)
        {
            throw EngineException.Engine(EngineErrorCode.Parse, $"unexpected engine JSON for a glass: {e.Message}");
        }
    }

    Task<string> ListGlassesAsync(string? catalog) =>
        InSession(s => CAbi.String("mk_session_list_glasses", (out nint p) => Native.mk_session_list_glasses(s, catalog, out p)));

    public async Task<IReadOnlyList<string>> MaterialNamesAsync()
    {
        try
        {
            return (await GlassesAsync().ConfigureAwait(false)).Select(g => g.Name).Distinct().Order(StringComparer.Ordinal).ToArray();
        }
        catch (EngineException)
        {
            return [];
        }
    }

    public async Task<IReadOnlyList<CatalogInfo>> CatalogsAsync()
    {
        try
        {
            var text = await InSession(s => CAbi.String("mk_session_list_catalogs", (out nint p) => Native.mk_session_list_catalogs(s, out p)))
                .ConfigureAwait(false);
            return (Json.Parse(text).ArrayOrNull() ?? []).Select(CatalogInfo.FromJson).ToArray();
        }
        catch (EngineException)
        {
            return [];
        }
    }

    public async Task<CatalogInfo> LoadCatalogAsync(string path)
    {
        var text = await InSession(s => CAbi.String("mk_session_load_catalog", (out nint p) => Native.mk_session_load_catalog(s, path, out p)))
            .ConfigureAwait(false);
        return CatalogInfo.FromJson(Json.Parse(text, "catalog"));
    }

    public Task RemoveCatalogAsync(string name) =>
        InSession(s => CAbi.String("mk_session_remove_catalog", (out nint p) => Native.mk_session_remove_catalog(s, name, out p)));

    public async Task<CatalogChanges> ReloadCatalogAsync(string name)
    {
        var text = await InSession(s => CAbi.String("mk_session_reload_catalog", (out nint p) => Native.mk_session_reload_catalog(s, name, out p)))
            .ConfigureAwait(false);
        return CatalogChanges.FromJson(Json.Parse(text, "catalog reload"))
            ?? throw EngineException.Engine(EngineErrorCode.Parse, "unexpected engine JSON for a catalog reload");
    }

    public async Task<LibrarySync> SetLibraryCatalogsAsync(IReadOnlyList<string> paths)
    {
        var pathsJson = new JsonArray(paths.Select(p => (JsonNode?)p).ToArray()).ToJsonString();
        try
        {
            var text = await InSession(s =>
                CAbi.String("mk_session_set_library_catalogs", (out nint p) => Native.mk_session_set_library_catalogs(s, pathsJson, out p)))
                .ConfigureAwait(false);
            var v = Json.Parse(text, "library sync");
            var errors = (v.At("errors").ArrayOrNull() ?? [])
                .Select(e => e.At("path").StringOrNull() is { } p && e.At("message").StringOrNull() is { } m ? new LibrarySync.Failure(p, m) : null)
                .OfType<LibrarySync.Failure>().ToArray();
            return new LibrarySync(
                (v.At("catalogs").ArrayOrNull() ?? []).Select(CatalogInfo.FromJson).ToArray(), errors, v.At("changed").BoolOrNull() ?? false);
        }
        catch (EngineException e)
        {
            return new LibrarySync([], [new LibrarySync.Failure("", e.Detail)], false);
        }
    }

    public Task SetGlassOverridesAsync(IReadOnlyList<GlassOverride> overrides)
    {
        var json = new JsonArray(overrides.Select(o => (JsonNode?)o.ToJson()).ToArray()).ToJsonString();
        return InSession(s =>
        {
            CAbi.Check(Native.mk_session_set_glass_overrides(s, json), "mk_session_set_glass_overrides");
            return true;
        });
    }

    public async Task<IReadOnlyList<GlassRow>> GlassRowsAsync(string? catalog)
    {
        var text = await ListGlassesAsync(catalog).ConfigureAwait(false);
        return (Json.Parse(text, "glass list").ArrayOrNull() ?? []).Select(GlassRow.FromJson).OfType<GlassRow>().ToArray();
    }

    public async Task<GlassDetails> GlassDetailsAsync(string? catalog, string name)
    {
        var text = await InSession(s =>
            CAbi.String("mk_session_glass_details", (out nint p) => Native.mk_session_glass_details(s, catalog, name, out p)))
            .ConfigureAwait(false);
        return GlassDetails.FromJson(Json.Parse(text, "glass details"))
            ?? throw EngineException.Engine(EngineErrorCode.Parse, "unexpected engine JSON for glass details");
    }

    public async Task<string> LibraryCatalogsInUseJsonAsync()
    {
        try
        {
            return await InSession(s =>
                CAbi.String("mk_session_library_catalogs_in_use", (out nint p) => Native.mk_session_library_catalogs_in_use(s, out p)))
                .ConfigureAwait(false);
        }
        catch (EngineException)
        {
            return "[]";
        }
    }

    public JsonNode? GlassFromSpec(JsonNode spec)
    {
        var json = spec.ToJsonString();
        return Json.Parse(CAbi.String("mk_glass_from_spec", (out nint p) => Native.mk_glass_from_spec(json, out p)), "glass");
    }

    public AgfExport Agf(JsonNode catalog)
    {
        var json = catalog.ToJsonString();
        var v = Json.Parse(CAbi.String("mk_catalog_to_agf", (out nint p) => Native.mk_catalog_to_agf(json, out p)), "AGF export");
        var warnings = (v.At("warnings").ArrayOrNull() ?? [])
            .Select(w => string.Join(": ", new[] { w.At("glass").StringOrNull(), w.At("message").StringOrNull() }.OfType<string>()))
            .ToArray();
        return new AgfExport(v.At("text").StringOrNull() ?? "", warnings);
    }

    // Lens prescription files

    public PrescriptionImport ImportPrescription(ReadOnlySpan<byte> data, PrescriptionFormat format) =>
        PrescriptionImport.FromJson(CAbi.ImportPrescription(data, format.WireName()));

    public async Task<PrescriptionImport> ImportPrescriptionAsync(string path)
    {
        var text = await InSession(s =>
            CAbi.String("mk_session_import_prescription", (out nint p) => Native.mk_session_import_prescription(s, path, out p)))
            .ConfigureAwait(false);
        return PrescriptionImport.FromJson(text);
    }

    // Analyses

    public Task<IReadOnlyList<AnalysisDescriptor>> AnalysisDescriptorsAsync() => Task.FromResult(core.Descriptors);

    /// <summary>
    /// Runs the analysis in the engine. The settings go as given: what the request leaves
    /// out (a seed included) takes the engine's default, the one its schema declares.
    /// <para>
    /// Returns at once. The job's task sets the session to <paramref name="snapshot"/> and
    /// submits in one locked step off the caller's thread, registering the job's event
    /// channel before any event can be delivered; engine events feed the job's progress; a
    /// 250 ms status poll is a safety net; the terminal event takes the result at once (its
    /// buffers wrapped without copying). <see cref="AnalysisJob.Cancel"/> calls
    /// <c>mk_job_cancel</c>; a job cancelled before it reached the engine is never submitted.
    /// <see cref="AnalysisJob.Stop"/> calls <c>mk_job_stop</c> for stoppable analyses (sent
    /// right after submission if requested earlier). Submission errors (unknown analysis,
    /// invalid settings, unavailable backend) fail the job at once; a job that fails while
    /// running throws the error of its own status code (an unknown glass is <c>NotFound</c>,
    /// a numerical failure <c>Compute</c>).
    /// </para>
    /// </summary>
    public AnalysisJob Submit(EngineAnalysisRequest request, ModelSnapshot snapshot)
    {
        var core = this.core;
        var requestJson = WireJson(request, core.GpuName);
        var submission = new Submission();
        Func<bool>? onStop = core.Stoppable.Contains(request.Analysis)
            ? () =>
            {
                if (submission.RequestStop() is not { } id) return true;
                return Try(() => CAbi.JobStop(core.Handle, id));
            }
            : null;
        return AnalysisJob.Run(request, snapshot.Hash, async (reporter, token) =>
        {
            ulong id;
            ChannelReader<JobEvent> events;
            try
            {
                (id, events) = await InSession(s =>
                {
                    if (submission.IsCancelled) throw EngineException.Cancelled();
                    Hold(s, snapshot);
                    return core.Hub.Submit(() => CAbi.Submit(s, requestJson));
                }).ConfigureAwait(false);
            }
            catch (EngineException e) when (e.Kind == EngineErrorKind.Engine && e.Code == EngineErrorCode.InvalidArgument)
            {
                throw EngineException.InvalidSettings([e.Detail]);
            }
            catch (ObjectDisposedException)
            {
                // The document engine was disposed before the job reached it.
                throw EngineException.Cancelled();
            }
            // Cancel() raced with the submission: cancel the engine job.
            if (!submission.Record(id)) Try(() => CAbi.JobCancel(core.Handle, id));
            // Stop() came before the engine job existed.
            if (submission.StopRequested) Try(() => CAbi.JobStop(core.Handle, id));
            using (var poll = new CancellationTokenSource())
            {
                _ = PollAsync(core, id, poll.Token);
                try
                {
                    await foreach (var e in events.ReadAllAsync(token).ConfigureAwait(false))
                    {
                        switch (e.State)
                        {
                            case JobState.Queued: reporter.Report(0, "Queued"); break;
                            case JobState.Running: reporter.Report(e.Progress, e.Stage.Length == 0 ? "Running" : e.Stage); break;
                        }
                    }
                }
                catch (OperationCanceledException) when (token.IsCancellationRequested)
                {
                }
                finally
                {
                    poll.Cancel();
                }
            }
            if (token.IsCancellationRequested)
            {
                core.Hub.Remove(id);
                Try(() => CAbi.JobRelease(core.Handle, id));
                throw EngineException.Cancelled();
            }
            var (text, buffers) = await OffThread(() => CAbi.JobTakeResult(core.Handle, id)).ConfigureAwait(false);
            try
            {
                return ReadEnvelope(text, buffers, request);
            }
            catch
            {
                foreach (var b in buffers) b.Dispose();
                throw;
            }
        },
        onCancel: () =>
        {
            if (submission.Cancel() is { } id) Try(() => CAbi.JobCancel(core.Handle, id));
        },
        onStop);
    }

    static AnalysisOutput ReadEnvelope(string text, ResultBuffer[] buffers, EngineAnalysisRequest request)
    {
        if (Json.Parse(text, "result envelope") is not JsonObject envelope || envelope["run_info"] is not { } runInfo)
        {
            throw EngineException.Engine(EngineErrorCode.Parse, "result envelope without run_info");
        }
        envelope.TryGetPropertyValue("result", out var result);
        envelope.Remove("result");
        Json.Materialize(result);
        return new AnalysisOutput(
            RunInfo.FromJson(runInfo), result, buffers, envelope["library_hash"].StringOrNull(), request,
            BackendReport.FromJson(envelope["backend"]));
    }

    /// <summary>Safety net: should an event be missed, poll the job's state every 250 ms and deliver a terminal one.</summary>
    static async Task PollAsync(FFIEngineCore core, ulong id, CancellationToken token)
    {
        try
        {
            while (true)
            {
                await Task.Delay(250, token).ConfigureAwait(false);
                try
                {
                    var (state, fraction) = CAbi.JobStatus(core.Handle, id);
                    if (state.IsTerminal())
                    {
                        core.Hub.Deliver(id, new JobEvent(state, fraction, ""));
                        return;
                    }
                }
                catch (EngineException)
                {
                }
            }
        }
        catch (OperationCanceledException)
        {
        }
        catch (ObjectDisposedException)
        {
        }
    }

    static bool Try(Action call)
    {
        try
        {
            call();
            return true;
        }
        catch (Exception e) when (e is EngineException or ObjectDisposedException)
        {
            return false;
        }
    }

    /// <summary>
    /// The request as the C ABI takes it: <c>{"analysis", "settings", "view_key", "backend",
    /// "allow_backend_fallback", "fresh"}</c> (<c>view_key</c> and <c>fresh</c> only when set). A GPU preference
    /// names <paramref name="gpuName"/> (the engine's first GPU backend) and allows the
    /// fallback to the CPU reference; without a GPU backend it runs as automatic.
    /// </summary>
    internal static string WireJson(EngineAnalysisRequest request, string? gpuName)
    {
        var o = new JsonObject { ["analysis"] = request.Analysis, ["settings"] = request.SettingsObject() };
        if (request.Fresh) o["fresh"] = true;
        if (request.ViewKey is { } key) o["view_key"] = key;
        switch (request.Execution.Backend)
        {
            case BackendPreference.CpuReference:
                o["backend"] = new JsonObject { ["kind"] = "cpu_reference" };
                break;
            case BackendPreference.Gpu when gpuName is not null:
                o["backend"] = new JsonObject { ["kind"] = "gpu", ["name"] = gpuName };
                o["allow_backend_fallback"] = true;
                break;
        }
        return o.ToJsonString();
    }
}
