// Tests await jobs and their progress without the test's token on purpose: the job's own
// cancellation is what they check.
#pragma warning disable xUnit1051
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using MKLight.Engine;
using static MKLight.Tests.Engine.EngineTestSupport;

namespace MKLight.Tests.Engine;

/// <summary>
/// Integration tests against the real Rust engine (<c>mklight_ffi.dll</c>). They are skipped,
/// with the build hint, when the library is not there.
/// </summary>
[Collection("FFI engine")]
public class FFIEngineTests
{
    static EngineAnalysisRequest Request(string analysis, params (string Key, JsonNode? Value)[] settings) =>
        new(analysis, Settings(settings));

    static EngineAnalysisRequest Request(string analysis, (string Key, JsonNode? Value) setting) => Request(analysis, [setting]);

    static Task<AnalysisOutput> Run(IEngineClient e, ModelSnapshot snap, string analysis, (string Key, JsonNode? Value) setting) =>
        Run(e, snap, analysis, [setting]);

    static async Task<AnalysisOutput> Run(IEngineClient e, ModelSnapshot snap, string analysis, params (string Key, JsonNode? Value)[] settings) =>
        await e.Submit(Request(analysis, settings), snap).ResultAsync();

    static async Task<EngineException> Failure(Func<Task> body) => await Assert.ThrowsAsync<EngineException>(body);

    [Fact]
    public void LibraryLoadsWithTheRightAbi()
    {
        RequireEngine();
        Assert.Null(FFIEngine.UnavailableReason);
        var version = Native.mk_abi_version();
        Assert.Equal(Native.AbiVersionMajor, version >> 16);
        Assert.True((version & 0xFFFF) >= 5, "ABI 1.5 or newer");
        var e = SharedEngine();
        Assert.False(e.Info.IsMock);
        Assert.NotEmpty(e.Info.Version);
        Assert.Contains((SharedEngine().Backends().ArrayOrNull() ?? []), b => b.At("id").StringOrNull() == "cpu_reference");
    }

    [Fact]
    public async Task AnalysisListHasSchemas()
    {
        var descriptors = await SharedEngine().AnalysisDescriptorsAsync();
        var names = descriptors.Select(d => d.Name).ToHashSet();
        foreach (var n in new[] { "validate", "system_summary", "glass_info", "rng_stream", "first_order", "spot_diagram", "ray_fan", "mtf", "layout" })
        {
            Assert.Contains(n, names);
        }
        var glass = descriptors.Single(d => d.Name == "glass_info");
        Assert.True(glass.SettingsFields[0].Name == "name" && glass.SettingsFields[0].Required);
        Assert.Null(glass.DefaultSettings);
        Assert.Contains(descriptors, d => d.GpuPath);
        Assert.Equal(["diagnostic", "geometric", "materials", "model", "optimization", "wave"],
            descriptors.Select(d => d.Category).Intersect(["diagnostic", "geometric", "materials", "model", "optimization", "wave"]).Order());
        // Every analysis has an object schema; group and variant settings have fields to show,
        // and every declared default is a value its field accepts.
        foreach (var d in descriptors)
        {
            Assert.True(d.SettingsSchema["type"].StringOrNull() == "object", $"{d.Name}: {d.SettingsSchema.ToJsonString()}");
            foreach (var f in d.SettingsFields)
            {
                if (f.Variants.Count > 0) Assert.All(f.Variants, v => Assert.NotNull(v.Tag));
                if (f.HasDefault && f.Types.Count > 0) Assert.True(f.Accepts(f.DefaultValue), $"{d.Name}.{f.Name} default {f.DefaultValue?.ToJsonString()}");
            }
            if (d.DefaultSettings is JsonObject defaults)
            {
                Assert.Equal(defaults.Count, d.AcceptedSettings(defaults.ToDictionary(kv => kv.Key, kv => kv.Value)).Count);
            }
        }
    }

    /// <summary>The binding reproduces the engine's BLAKE3 content hashes of the fixtures.</summary>
    [Fact]
    public async Task SnapshotHashesMatchEngineFixtures()
    {
        var e = SharedEngine().NewSession();
        var hashes = JsonNode.Parse(File.ReadAllText(Fixture("snapshot_hashes.json")))!.AsObject();
        Assert.True(hashes.Count >= 3);
        foreach (var (file, hex) in hashes)
        {
            var snap = await e.OpenSnapshotAsync(File.ReadAllText(Fixture(file)));
            Assert.Equal(hex!.GetValue<string>(), snap.Hash.Hex);
            Assert.Equal(64, snap.LibraryHash?.Length);
        }
    }

    public static TheoryData<string> Samples() => new(SampleIds);

    [Theory]
    [MemberData(nameof(Samples))]
    public async Task EverySampleOpensValidatesAndSummarises(string id)
    {
        using var e = SharedEngine().NewSession();
        var system = Sample(id);
        var snap = await e.OpenSnapshotAsync(system.ToJsonString());
        using var validation = await Run(e, snap, "validate");
        Assert.True(validation.Json.At("ok").BoolOrNull() == true && validation.Json.At("errors").IntOrNull() == 0, $"{id}: {validation.Json}");
        var report = await e.ValidateAsync(snap);
        Assert.True(report.At("ok").BoolOrNull() == true);
        using var output = await Run(e, snap, "system_summary");
        Assert.Equal(snap.Hash, output.RunInfo.SnapshotHash);
        Assert.Equal("system_summary", output.RunInfo.Analysis);
        Assert.Equal(snap.LibraryHash, output.LibraryHash);
        var n = output.Buffer("n_after")!;
        Assert.Equal([system["surfaces"]!.AsArray().Count, system["wavelengths"]!["wavelengths"]!.AsArray().Count], n.Shape);
        Assert.All(n.ToDoubleArray(), x => Assert.InRange(x, 0.9, 2.1));
    }

    [Fact]
    public async Task HashesAreStableAndSensitive()
    {
        var e = SharedEngine();
        var a = await e.OpenSnapshotAsync(SampleJson("cooke_triplet"));
        var b = await e.OpenSnapshotAsync(SampleJson("cooke_triplet"));
        Assert.Equal(a, b);
        var edited = Sample("cooke_triplet");
        edited["surfaces"]![2]!["thickness"] = edited["surfaces"]![2]!["thickness"]!.GetValue<double>() + 1e-9;
        var c = await e.OpenSnapshotAsync(edited.ToJsonString());
        Assert.NotEqual(a.Hash, c.Hash);
        Assert.Equal(12, a.Hash.Short.Length);
    }

    /// <summary>The analyses the app draws, with their buffers read without copying.</summary>
    [Fact]
    public async Task CoreAnalysesReturnTheirBuffers()
    {
        var e = SharedEngine();
        var snap = await e.OpenSnapshotAsync(SampleJson("cooke_triplet"));

        using (var spot = await Run(e, snap, "spot_diagram", ("rings", 4)))
        {
            Assert.NotEmpty(spot.Json.At("fields").ArrayOrNull()!);
            var points = spot.Buffer("points")!;
            Assert.True(points.DType == BufferDType.F64 && points.Shape.Count == 2 && points.Shape[1] == 2 && points.Count > 0);
            var xy = points.AsSpan<double>();
            Assert.Equal(points.Count, xy.Length);
            var fields = spot.Buffer("point_field")!;
            Assert.Equal(BufferDType.U32, fields.DType);
            Assert.Equal(points.Rows, fields.Count);
            Assert.True(fields.TryGetSpan<uint>(out var f) && f.Length == fields.Count);
            Assert.False(fields.TryGetSpan<int>(out _), "typed access checks the dtype");
        }

        using (var fan = await Run(e, snap, "ray_fan", ("samples", 21)))
        {
            Assert.Equal(21, fan.Json.At("fields", 0, "tangential", 0, "values").NfArray()?.Length);
            Assert.Empty(fan.Buffers);
        }

        using (var grid = await Run(e, snap, "ray_grid", ("rings", 3)))
        {
            var points = grid.Buffer("points")!;
            Assert.Equal(4, points.Shape.Count);
            Assert.Equal(2, points.Shape[3]);
            var pupil = grid.Buffer("pupil")!;
            Assert.Equal(points.Shape[2], pupil.Rows);
            Assert.Equal(points[0, 0, 0, 0], points.ToDoubleArray()[0]);
        }

        using (var mtf = await Run(e, snap, "mtf"))
        {
            var tangential = mtf.Buffers.Where(b => b.Name.StartsWith("tangential_", StringComparison.Ordinal)).ToArray();
            Assert.NotEmpty(tangential);
            Assert.All(tangential, b => Assert.Equal(1.0, b[0], 6));
        }

        using (var layout = await Run(e, snap, "layout"))
        {
            var scene = LayoutScene.FromOutput(layout)!;
            Assert.Contains(scene.Polylines, p => p.Kind == LayoutScene.PolylineKind.Ray && p.InSection);
            Assert.Contains(scene.Polylines, p => p.Kind == LayoutScene.PolylineKind.SurfaceProfile);
            Assert.Equal(3, scene.Elements.Count);
            Assert.InRange(scene.Surfaces.Count, Sample("cooke_triplet")["surfaces"]!.AsArray().Count - 1, Sample("cooke_triplet")["surfaces"]!.AsArray().Count);
            Assert.True(scene.Bounds.Min.Z < 0 && scene.Bounds.Max.Z > 60);
            Assert.All(scene.Elements, el => Assert.Equal(el.Front.Count, el.Back.Count));
            Assert.NotEmpty(LayoutScene.AutoSemiDiameters(scene));
        }

        using (var fo = await Run(e, snap, "first_order"))
        {
            var efl = fo.Json.At("entries").ArrayOrNull()!.First(x => x.At("key").StringOrNull() == "efl");
            Assert.InRange(efl.NfAt("value")!.Value, 49.9, 50.1);
        }
    }

    /// <summary>The x–z section and the 3-D surface mesh, as the viewport asks for them.</summary>
    [Theory]
    [InlineData("q_con")]
    [InlineData("hybrid_doe")]
    [InlineData("anamorphic")]
    public async Task LayoutSectionsAndMeshesRead(string id)
    {
        var e = SharedEngine();
        var snap = await e.OpenSnapshotAsync(SampleJson(id));
        using var xz = await Run(e, snap, "layout", ("section", "xz"), ("mesh_rings", 6));
        var scene = LayoutScene.FromOutput(xz)!;
        Assert.Equal("xz", scene.Section);
        Assert.NotEmpty(scene.Mesh);
        Assert.All(scene.Mesh, t => Assert.True(t.Surface >= 1 && double.IsFinite(t.A.Z)));
        Assert.NotEmpty(scene.Elements);
        if (id == "hybrid_doe") Assert.Contains(scene.Polylines, p => p.Doe);
    }

    /// <summary>
    /// Submitting never blocks the caller: while another thread holds the session lock, as a
    /// catalog load does, Submit returns at once and the job runs once the lock is free,
    /// against its own snapshot. A job cancelled while waiting is never run.
    /// </summary>
    [Fact]
    public async Task SubmitDoesNotBlockTheCaller()
    {
        using var e = SharedEngine().NewSession();
        var achromat = await e.OpenSnapshotAsync(SampleJson("achromat"));
        var singlet = await e.OpenSnapshotAsync(SampleJson("singlet")); // the session holds the singlet now
        var held = new TaskCompletionSource();
        var holder = Task.Run(() => e.Locked(_ =>
        {
            held.SetResult();
            Thread.Sleep(400);
            return 0;
        }));
        await held.Task;
        var clock = Stopwatch.StartNew();
        var job = e.Submit(Request("system_summary"), achromat);
        var doomed = e.Submit(Request("system_summary"), singlet);
        var elapsed = clock.Elapsed;
        doomed.Cancel();
        Assert.True(elapsed < TimeSpan.FromMilliseconds(100), $"submit waited {elapsed} for the session lock");
        using var output = await job.ResultAsync();
        Assert.Equal(achromat.Hash, output.RunInfo.SnapshotHash);
        Assert.Equal(Sample("achromat")["surfaces"]!.AsArray().Count, output.Json.At("surfaces").ArrayOrNull()!.Count);
        Assert.Equal(EngineErrorKind.Cancelled, (await Failure(() => doomed.ResultAsync())).Kind);
        Assert.True(clock.Elapsed >= TimeSpan.FromMilliseconds(350), "the job waited for the lock");
        await holder;
    }

    [Fact]
    public async Task AnalysisReportsProgressAndIsReproducible()
    {
        var e = SharedEngine();
        var snap = await e.OpenSnapshotAsync(SampleJson("singlet"));
        async Task<(AnalysisOutput, List<JobProgress>)> RunSeed(long seed)
        {
            var job = e.Submit(Request("rng_stream", ("count", 4000), ("chunk", 500), ("delay_ms", 5), ("seed", seed)), snap);
            var updates = new List<JobProgress>();
            await foreach (var p in job.ProgressAsync()) updates.Add(p);
            return (await job.ResultAsync(), updates);
        }
        var (a, updates) = await RunSeed(7);
        Assert.True(updates.Count >= 2);
        Assert.True(updates.Zip(updates.Skip(1)).All(p => p.First.Fraction <= p.Second.Fraction), "monotone");
        Assert.Contains(updates, u => u.Fraction > 0 && u.Fraction < 1);
        Assert.Equal(1, updates[^1].Fraction);
        Assert.Equal(7UL, a.RunInfo.Seed);
        Assert.True(Json.DeepEquals(a.RunInfo.Settings.At("seed"), 7));
        var u = a.Buffer("uniform")!;
        Assert.True(u.Count == 4000 && u.DType == BufferDType.F64);
        double mean = 0;
        foreach (var x in u.AsSpan<double>()) mean += x;
        Assert.InRange(mean / u.Count, 0.47, 0.53);
        var (b, _) = await RunSeed(7);
        var (c, _) = await RunSeed(8);
        Assert.Equal(u, b.Buffer("uniform"));
        Assert.NotEqual(u, c.Buffer("uniform"));
        Assert.True(a.IsCurrent(snap, a.EngineRequest!));
        Assert.False(a.IsCurrent(snap, c.EngineRequest!), "another seed is another request");
    }

    /// <summary>A seed the request leaves out is the engine's default (the one the schema shows); a seed it sets is the one used.</summary>
    [Fact]
    public async Task SeedComesFromTheSettingsOnly()
    {
        var e = SharedEngine();
        var snap = await e.OpenSnapshotAsync(SampleJson("singlet"));
        var descriptor = (await e.AnalysisDescriptorsAsync()).Single(d => d.Name == "rng_stream");
        var shown = descriptor.SettingsFields.Single(f => f.Name == "seed").DefaultValue;
        using var byDefault = await Run(e, snap, "rng_stream", ("count", 16));
        Assert.True(Json.DeepEquals(shown, byDefault.RunInfo.Settings.At("seed")));
        Assert.Equal(shown.UInt64OrNull(), byDefault.RunInfo.Seed);
        using var explicitSeed = await Run(e, snap, "rng_stream", ("count", 16), ("seed", 5));
        Assert.Equal(5UL, explicitSeed.RunInfo.Seed);
        Assert.NotEqual(byDefault.Buffer("uniform"), explicitSeed.Buffer("uniform"));
    }

    [Fact]
    public async Task CancellingAJobStopsIt()
    {
        var e = SharedEngine();
        var snap = await e.OpenSnapshotAsync(SampleJson("singlet"));
        var job = e.Submit(Request("rng_stream", ("count", 1_000_000), ("chunk", 1000), ("delay_ms", 20)), snap);
        var clock = Stopwatch.StartNew();
        bool sawRunning = false;
        await foreach (var p in job.ProgressAsync())
        {
            if (p.Stage == "Queued") continue;
            sawRunning = true;
            job.Cancel();
            break;
        }
        Assert.True(sawRunning);
        Assert.Equal(EngineErrorKind.Cancelled, (await Failure(() => job.ResultAsync())).Kind);
        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(5), "the job stopped long before its 20 s of work");
        using var ok = await Run(e, snap, "validate");
        Assert.Equal("validate", ok.RunInfo.Analysis);
    }

    /// <summary>A job cancelled before it reached the engine never runs.</summary>
    [Fact]
    public async Task AJobCancelledBeforeSubmissionNeverRuns()
    {
        using var e = SharedEngine().NewSession();
        var snap = await e.OpenSnapshotAsync(SampleJson("singlet"));
        var job = e.Locked(_ =>
        {
            var j = e.Submit(Request("system_summary"), snap);
            j.Cancel();
            return j;
        });
        Assert.Equal(EngineErrorKind.Cancelled, (await Failure(() => job.ResultAsync())).Kind);
    }

    /// <summary>A newer job with the same view key supersedes (cancels) the older one.</summary>
    [Fact]
    public async Task ViewKeysSupersedeOlderJobs()
    {
        using var e = SharedEngine().NewSession();
        var snap = await e.OpenSnapshotAsync(SampleJson("singlet"));
        var slow = Request("rng_stream", ("count", 1_000_000), ("chunk", 1000), ("delay_ms", 20)).WithViewKey("pane-1");
        var older = e.Submit(slow, snap);
        await foreach (var p in older.ProgressAsync())
        {
            if (p.Stage != "Queued") break;
        }
        var newer = e.Submit(Request("rng_stream", ("count", 64)).WithViewKey("pane-1"), snap);
        Assert.Equal(EngineErrorKind.Cancelled, (await Failure(() => older.ResultAsync())).Kind);
        using var output = await newer.ResultAsync();
        Assert.Equal(64, output.Buffer("uniform")!.Count);
    }

    /// <summary>Which jobs can be stopped is what the engine's analysis list says, and a stopped job keeps its best result.</summary>
    [Fact]
    public async Task StopSupportComesFromTheAnalysisList()
    {
        var e = SharedEngine();
        var stoppable = (await e.AnalysisDescriptorsAsync()).Where(d => d.Stoppable).Select(d => d.Name).ToHashSet();
        Assert.Equal(["glass_substitution", "optimize", "tolerance"], stoppable.Order());
        var snap = await e.OpenSnapshotAsync(SampleJson("singlet"));
        foreach (var name in new[] { "optimize", "tolerance", "glass_substitution", "rng_stream", "validate" })
        {
            var job = e.Submit(Request(name), snap);
            Assert.Equal(stoppable.Contains(name), job.CanStop);
            job.Cancel();
            try { (await job.ResultAsync()).Dispose(); } catch (EngineException) { }
        }
    }

    [Fact]
    public async Task StoppingKeepsTheBestResultSoFar()
    {
        using var e = SharedEngine().NewSession();
        var snap = await e.OpenSnapshotAsync(SampleJson("cooke_triplet"));
        // Stop requested before the engine job exists: sent right after submission.
        var job = e.Submit(Request("tolerance", ("trials", 100_000)), snap);
        Assert.True(job.Stop());
        using var output = await job.ResultAsync().WaitAsync(TimeSpan.FromSeconds(60));
        Assert.True(output.Json.At("stopped").BoolOrNull() == true);
    }

    [Fact]
    public async Task ErrorsMapToCodes()
    {
        using var e = SharedEngine().NewSession();
        var broken = Sample("singlet");
        broken["stop"] = 0;
        var invalid = await Failure(() => e.OpenSnapshotAsync(broken.ToJsonString()));
        Assert.Equal(EngineErrorKind.InvalidModel, invalid.Kind);
        Assert.Contains("stop", invalid.Issues[0]);
        Assert.Equal(EngineErrorKind.InvalidModel, (await Failure(() => e.OpenSnapshotAsync("not json"))).Kind);
        Assert.Equal(EngineErrorKind.InvalidModel, (await Failure(() => e.OpenSnapshotAsync("""{"name":1}"""))).Kind);
        var snap = await e.OpenSnapshotAsync(SampleJson("singlet"));
        async Task<EngineException> Fails(EngineAnalysisRequest r) => await Failure(() => e.Submit(r, snap).ResultAsync());
        Assert.Equal(EngineErrorCode.NotFound, (await Fails(Request("no_such_analysis"))).Code);
        var bogus = await Fails(Request("rng_stream", ("bogus", 1)));
        Assert.True(bogus.Kind == EngineErrorKind.InvalidSettings && bogus.Code == EngineErrorCode.InvalidArgument, bogus.Message);
        Assert.Equal(EngineErrorCode.InvalidArgument, (await Fails(Request("glass_info"))).Code);
        Assert.Equal(EngineErrorCode.NotFound, (await Fails(Request("glass_info", ("name", "NO-SUCH-GLASS")))).Code);
        // Statuses of the session calls keep their codes too.
        Assert.Equal(EngineErrorCode.Io, (await Failure(() => e.LoadCatalogAsync(@"Z:\nonexistent\x.agf"))).Code);
        var junk = Path.Combine(Path.GetTempPath(), $"mklight-junk-{Guid.NewGuid()}.json");
        await File.WriteAllTextAsync(junk, "not json");
        try
        {
            Assert.Equal(EngineErrorCode.Parse, (await Failure(() => e.LoadCatalogAsync(junk))).Code);
        }
        finally
        {
            File.Delete(junk);
        }
    }

    /// <summary>A job that fails while running throws the error of its own status code and message, not a generic "job failed".</summary>
    [Fact]
    public async Task FailedJobsKeepTheirErrorCode()
    {
        using var e = SharedEngine().NewSession();
        var system = Sample("singlet");
        system["surfaces"]![1]!["material"] = new JsonObject { ["type"] = "catalog", ["catalog"] = null, ["name"] = "NOPE-42" };
        var snap = await e.OpenSnapshotAsync(system.ToJsonString());
        var error = await Failure(() => e.Submit(Request("system_summary"), snap).ResultAsync());
        Assert.True(error.Kind == EngineErrorKind.Engine && error.Code == EngineErrorCode.NotFound, error.ToString());
        Assert.Contains("NOPE-42", error.Message);
    }

    [Fact]
    public async Task CatalogsChangeTheLibraryHash()
    {
        using var e = SharedEngine().NewSession();
        var dir = Directory.CreateTempSubdirectory("mklight-cat-");
        try
        {
            var agf = Path.Combine(dir.FullName, "MYCAT.agf");
            await File.WriteAllTextAsync(agf, """
                CC Test catalog
                NM MY-BK7 2 517642 1.5168 64.17 0 1 0
                CD 1.03961212E+000 6.00069867E-003 2.31792344E-001 2.00179144E-002 1.01046945E+000 1.03560653E+002 0 0 0 0
                LD 3.0E-001 2.5E+000

                """);
            var system = Sample("singlet");
            system["surfaces"]![1]!["material"] = new JsonObject { ["type"] = "catalog", ["catalog"] = "MYCAT", ["name"] = "MY-BK7" };
            async Task<long?> ValidationErrors(ModelSnapshot s)
            {
                using var output = await Run(e, s, "validate");
                return output.Json.At("errors").IntOrNull();
            }
            var before = await e.OpenSnapshotAsync(system.ToJsonString());
            Assert.True(await ValidationErrors(before) > 0, "unknown glass before loading");
            var info = await e.LoadCatalogAsync(agf);
            Assert.True(info.Name == "MYCAT" && info.GlassCount == 1 && info.Source == "user");
            Assert.Equal(["BUILTIN", "MYCAT"], (await e.CatalogsAsync()).Select(c => c.Name));
            var after = await e.OpenSnapshotAsync(system.ToJsonString());
            Assert.Equal(before.Hash, after.Hash);
            Assert.NotEqual(before.LibraryHash, after.LibraryHash);
            Assert.Equal(0, await ValidationErrors(after));
            var glasses = await e.GlassesAsync("MYCAT");
            Assert.Equal(["MY-BK7"], glasses.Select(g => g.Name));
            Assert.Equal(1.5168, glasses[0].Nd, 3);
            using var output = await Run(e, after, "system_summary");
            Assert.True(output.IsCurrent(after, Request("system_summary")));
            Assert.False(output.IsCurrent(before, Request("system_summary")), "stale against the old library");
            await Failure(() => e.LoadCatalogAsync(Path.Combine(dir.FullName, "missing.agf")));
            var rows = await e.GlassRowsAsync("MYCAT");
            Assert.Equal("MY-BK7", rows.Single().Name);
            var details = await e.GlassDetailsAsync("MYCAT", "MY-BK7");
            Assert.Equal("MY-BK7", details.Row.Name);
            Assert.NotEmpty(details.Coefficients);
            var reload = await e.ReloadCatalogAsync("MYCAT");
            Assert.True(reload.Catalog == "MYCAT" && reload.IsEmpty);
            await e.RemoveCatalogAsync("MYCAT");
            Assert.Equal(["BUILTIN"], (await e.CatalogsAsync()).Select(c => c.Name));
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }

    [Fact]
    public async Task BuiltinGlassesAndUserGlasses()
    {
        var e = SharedEngine();
        var names = await e.MaterialNamesAsync();
        Assert.Contains("N-BK7", names);
        Assert.Equal(names.Order(StringComparer.Ordinal), names);
        var rows = await e.GlassRowsAsync(null);
        Assert.Contains(rows, r => r.Name == "N-BK7" && Math.Abs(r.Nd - 1.5168) < 1e-3);
        var details = await e.GlassDetailsAsync(null, "N-BK7");
        Assert.True(details.Lines.Count > 0 && details.IndexCurve.Count > 0);
        Assert.Equal(EngineErrorCode.NotFound, (await Failure(() => e.GlassDetailsAsync(null, "NO-SUCH-GLASS"))).Code);
        var agf = e.Agf(new JsonObject { ["name"] = "EMPTY", ["glasses"] = new JsonObject() });
        Assert.Empty(agf.Warnings);
        var sync = await e.NewSession().SetLibraryCatalogsAsync([]);
        Assert.Empty(sync.Errors);
        Assert.Equal("[]", await e.LibraryCatalogsInUseJsonAsync());
    }

    /// <summary>Jobs submitted concurrently on different snapshots each run on their own snapshot (set + submit is one locked step).</summary>
    [Fact]
    public async Task ConcurrentSubmissionsKeepTheirSnapshots()
    {
        using var e = SharedEngine().NewSession();
        var systems = Enumerable.Range(0, 8).Select(k =>
        {
            var s = Sample("singlet");
            s["surfaces"]![2]!["thickness"] = 90.0 + k;
            return s.ToJsonString();
        }).ToArray();
        var snaps = await Task.WhenAll(systems.Select(s => Task.Run(() => e.OpenSnapshotAsync(s))));
        await Task.WhenAll(snaps.Select((snap, k) => Task.Run(async () =>
        {
            using var output = await e.Submit(Request("system_summary"), snap).ResultAsync();
            Assert.Equal(snap.Hash, output.RunInfo.SnapshotHash);
            Assert.Equal(90.0 + k, output.Json.At("surfaces", 2, "thickness").DoubleValue());
        })));
    }

    [Fact]
    public async Task SolvesApplyAndReadTheSolvedSystemBack()
    {
        using var e = SharedEngine().NewSession();
        var snap = await e.OpenSnapshotAsync(File.ReadAllText(Fixture("solved_triplet.json")));
        var solved = await e.ApplySolvesAsync(new Dictionary<string, JsonNode?>(), snap);
        Assert.NotEmpty(solved.Report.At("outcomes").ArrayOrNull()!);
        Assert.Equal(64, solved.Snapshot.Hash.Hex.Length);
        Assert.Equal(snap.LibraryHash, solved.Snapshot.LibraryHash);
        var reopened = await e.NewSession().OpenSnapshotAsync(solved.Snapshot.SystemJson);
        Assert.Equal(solved.Snapshot.Hash, reopened.Hash);
        var thickness = await e.GetParamAsync(solved.Snapshot, """{"param":"thickness","surface":2}""");
        Assert.Equal(JsonNode.Parse(solved.Snapshot.SystemJson).At("surfaces", 2, "thickness").DoubleValue(), thickness);
        Assert.Equal(double.PositiveInfinity, await e.GetParamAsync(solved.Snapshot, """{"param":"thickness","surface":0}"""));
    }

    [Fact]
    public async Task ImportsALensPrescription()
    {
        var e = SharedEngine();
        var path = RepoPath("crates", "mklight-core", "tests", "fixtures", "prescriptions", "cooke_triplet.zmx");
        var data = await File.ReadAllBytesAsync(path);
        var r = e.ImportPrescription(data, PrescriptionFormat.Zmx);
        Assert.True(r.Format == PrescriptionFormat.Zmx && r.Encoding == "utf16_le");
        var system = JsonNode.Parse(r.SystemJson)!;
        Assert.Equal("Cooke triplet f50 F/4", system.At("name").StringOrNull());
        Assert.Equal(8, system.At("surfaces").ArrayOrNull()!.Count);
        Assert.Equal(4, system.At("stop").IntOrNull());
        using var doc = e.NewSession();
        await doc.OpenSnapshotAsync(r.SystemJson);
        Assert.Equal(EngineErrorCode.Parse, Assert.Throws<EngineException>(() => e.ImportPrescription("hello"u8, PrescriptionFormat.Seq)).Code);
        var intoSession = await doc.ImportPrescriptionAsync(path);
        Assert.NotNull(intoSession.SnapshotHash);
        Assert.Equal("Cooke triplet f50 F/4", JsonNode.Parse(intoSession.SystemJson).At("name").StringOrNull());
    }

    [Fact]
    public async Task AutosaveWritesTheProject()
    {
        using var e = SharedEngine().NewSession();
        await e.OpenSnapshotAsync(SampleJson("singlet"));
        var path = Path.Combine(Path.GetTempPath(), $"mklight-autosave-{Guid.NewGuid()}.mklight");
        try
        {
            Assert.Equal(path, await e.AutosaveAsync(path));
            Assert.Contains("Singlet f100 F/5", await File.ReadAllTextAsync(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    /// <summary>A preparation the engine refused to start reads as failed, with the engine's message; a started one leaves the idle state.</summary>
    [Fact]
    public async Task GpuPreparationStates()
    {
        RequireEngine();
        using var e = new FFIEngine();
        Assert.Equal(new GpuPreparation.Idle(), e.GetGpuPreparation());
        e.SetGpuPreparationError("cache_dir is not valid UTF-8");
        Assert.Equal(new GpuPreparation.Failed("", "cache_dir is not valid UTF-8"), e.GetGpuPreparation());
        e.SetGpuPreparationError(null);
        e.PrepareGpu(Path.Combine(Path.GetTempPath(), "mklight-gpu-kernels-test"));
        var clock = Stopwatch.StartNew();
        var state = e.GetGpuPreparation();
        while (state.IsPreparing && clock.Elapsed < TimeSpan.FromSeconds(60))
        {
            await Task.Delay(50);
            state = e.GetGpuPreparation();
        }
        Assert.IsNotType<GpuPreparation.Idle>(state);
        Assert.False(state.IsPreparing);
    }

    /// <summary>The native buffer array is freed once, when the last buffer of a result is released, or by the GC when the app forgets.</summary>
    [Fact]
    public async Task BuffersFreeTheirArrayWithTheLastOne()
    {
        var e = SharedEngine();
        var snap = await e.OpenSnapshotAsync(SampleJson("cooke_triplet"));
        var output = await Run(e, snap, "spot_diagram");
        Assert.True(output.Buffers.Count >= 2);
        var owner = (BufferArray)output.Buffers[0].Owner!;
        Assert.All(output.Buffers, b => Assert.Same(owner, b.Owner));
        output.Buffers[0].Dispose();
        Assert.False(owner.IsFreed);
        Assert.Equal(output.Buffers[1].Count, output.Buffers[1].ToDoubleArray().Length);
        output.Dispose();
        output.Dispose();
        Assert.True(owner.IsFreed);
        // Forgotten buffers: the owner is collected (and frees the array) once none is reachable.
        var weak = await ForgottenOwner(e, snap);
        for (int i = 0; i < 20 && weak.IsAlive; i++)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            await Task.Delay(10);
        }
        Assert.False(weak.IsAlive);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    static async Task<WeakReference> ForgottenOwner(FFIEngine e, ModelSnapshot snap)
    {
        var output = await Run(e, snap, "spot_diagram");
        return new WeakReference(output.Buffers[0].Owner, trackResurrection: true);
    }

    /// <summary>Engine and session handles are released when the last reference goes; Dispose destroys a session at once.</summary>
    [Fact]
    public async Task HandlesAreFreed()
    {
        RequireEngine();
        var (engine, session) = await UseAndDrop();
        for (int i = 0; i < 50 && (engine.IsAlive || session.IsAlive); i++)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            await Task.Delay(20);
        }
        Assert.False(session.IsAlive);
        Assert.False(engine.IsAlive);
        var doc = SharedEngine().NewSession();
        doc.Dispose();
        Assert.True(doc.Session.IsClosed);
        Assert.False(SharedEngine().Core.Handle.IsClosed, "the shared engine outlives a document session");
        Assert.Equal(EngineErrorKind.Cancelled, (await Failure(() =>
            doc.Submit(Request("validate"), new ModelSnapshot("{}", new SnapshotHash("x"))).ResultAsync())).Kind);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    static async Task<(WeakReference Engine, WeakReference Session)> UseAndDrop()
    {
        var e = new FFIEngine();
        var doc = (FFIEngine)e.MakeDocumentEngine();
        var snap = await doc.OpenSnapshotAsync(SampleJson("singlet"));
        using (await doc.Submit(Request("validate"), snap).ResultAsync()) { }
        return (new WeakReference(e.Core.Handle, trackResurrection: true), new WeakReference(doc.Session, trackResurrection: true));
    }
}

[CollectionDefinition("FFI engine")]
public class FFIEngineCollection;

/// <summary>Without the library, construction fails with a clear error instead of crashing; with it, it succeeds.</summary>
public class FFIAvailabilityTests
{
    [Fact]
    public void ConstructionReflectsAvailability()
    {
        if (FFIEngine.IsAvailable)
        {
            using var e = new FFIEngine();
            Assert.NotEmpty(e.Info.Version);
        }
        else
        {
            var error = Assert.Throws<EngineException>(() => new FFIEngine());
            Assert.Equal(EngineErrorKind.LibraryUnavailable, error.Kind);
            Assert.Contains("build-windows.ps1", error.Message);
            Assert.NotNull(FFIEngine.UnavailableReason);
        }
    }
}
