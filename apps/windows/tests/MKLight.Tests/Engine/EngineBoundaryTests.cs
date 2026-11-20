// Tests await jobs and their progress without the test's token on purpose: the job's own
// cancellation is what they check.
#pragma warning disable xUnit1051
using System.Text.Json.Nodes;
using MKLight.Engine;

namespace MKLight.Tests.Engine;

/// <summary>
/// The engine boundary pieces an FFI client relies on, tested without the engine library:
/// schema handling, the settings a schema accepts, buffers, error mapping and staleness.
/// </summary>
public class EngineBoundaryTests
{
    /// <summary>A descriptor in the exact JSON form of <c>mk_engine_list_analyses</c>, plus a key a newer engine might add.</summary>
    const string DescriptorJson = """
        {"name":"spot_diagram","title":"Spot diagram","category":"geometric","description":"d",
         "available":true,"unavailable_reason":null,"future_field":{"x":1},"gpu_path":true,
         "settings_schema":{"type":"object","required":["rings"],"properties":{
            "rings":{"type":"integer","minimum":1,"default":6,"title":"Rings"},
            "seed":{"type":"integer","default":1},
            "fields":{"type":["integer","null"]},
            "wavelengths":{"type":"integer"},
            "pattern":{"type":"string","enum":["hexapolar","square"]},
            "max_frequency":{"type":"number","unit":"cycles/mm"}}},
         "default_settings":null}
        """;

    static AnalysisDescriptor Descriptor() => AnalysisDescriptor.FromJson(JsonNode.Parse(DescriptorJson));

    [Fact]
    public void DescriptorAndSchemaDecode()
    {
        var d = Descriptor();
        Assert.True(d.Available);
        Assert.Null(d.UnavailableReason);
        Assert.Null(d.DefaultSettings);
        Assert.True(d.GpuPath);
        Assert.Equal("geometric", d.Category);
        var fields = d.SettingsFields;
        Assert.Equal("rings", fields[0].Name);
        Assert.True(fields[0].Required, "required first");
        Assert.Equal(["fields", "max_frequency", "pattern", "seed", "wavelengths"], fields.Skip(1).Select(f => f.Name));
        Assert.Equal("cycles/mm", fields.Single(f => f.Name == "max_frequency").Unit);
        Assert.Equal(["integer", "null"], fields.Single(f => f.Name == "fields").Types);
        Assert.True(fields.Single(f => f.Name == "fields").Nullable);
        var rings = fields.Single(f => f.Name == "rings");
        Assert.Equal(1, rings.Minimum);
        Assert.Equal("Rings", rings.Title);
        Assert.True(rings.Accepts(3) && rings.Accepts(3.0) && !rings.Accepts(3.5));
        Assert.False(rings.Accepts("3"));
        Assert.True(Json.DeepEquals(rings.DefaultValue, 6));
        var pattern = fields.Single(f => f.Name == "pattern");
        Assert.True(pattern.Accepts("square") && !pattern.Accepts("random"));
        // Minimal descriptors (older engines) decode with defaults.
        var minimal = AnalysisDescriptor.FromJson(JsonNode.Parse("""{"name":"x"}"""));
        Assert.True(minimal.Title == "x" && minimal.Available && minimal.SettingsFields.Count == 0 && !minimal.GpuPath);
        // Stop support is what the engine lists; absent means none.
        Assert.False(d.Stoppable || minimal.Stoppable);
        var solver = AnalysisDescriptor.FromJson(JsonNode.Parse("""{"name":"optimize","stoppable":true}"""));
        Assert.True(solver.Stoppable);
        Assert.True(AnalysisDescriptor.FromJson(solver.ToJson()).Stoppable);
        Assert.Throws<EngineException>(() => AnalysisDescriptor.FromJson(JsonNode.Parse("""{"title":"no name"}""")));
    }

    /// <summary>Tagged unions (<c>oneOf</c> with <c>const</c> tags) and object settings known only from their default.</summary>
    [Fact]
    public void VariantsAndInferredGroups()
    {
        var schema = JsonNode.Parse("""
            {"type":"object","properties":{
              "pupil":{"oneOf":[
                {"title":"Grid","properties":{"type":{"const":"grid"},"n":{"type":"integer","default":8}}},
                {"title":"Rings","properties":{"type":{"const":"rings"},"rings":{"type":"integer","default":3}}},
                {"title":"no tag","properties":{"x":{"type":"number"}}}]},
              "detector":{"type":["object","null"],"default":{"pitch_um":2.5,"fill":1,"name":null}}}}
            """)!.AsObject();
        var d = new AnalysisDescriptor("x", "X", "test", settingsSchema: schema);
        var pupil = d.SettingsFields.Single(f => f.Name == "pupil");
        Assert.Equal(["Grid", "Rings"], pupil.Variants.Select(v => v.Title));
        Assert.Equal(("type", "grid"), (pupil.Variants[0].Tag!.Value.Key, pupil.Variants[0].Tag!.Value.Value!.GetValue<string>()));
        Assert.True(Json.DeepEquals(pupil.Variants[1].DefaultObject(), JsonNode.Parse("""{"type":"rings","rings":3}""")));
        Assert.Equal("Rings", pupil.VariantOf(JsonNode.Parse("""{"type":"rings","rings":5}"""))?.Title);
        Assert.Null(pupil.VariantOf(JsonNode.Parse("""{"type":"hex"}""")));
        Assert.Empty(pupil.Properties);
        var detector = d.SettingsFields.Single(f => f.Name == "detector");
        Assert.Equal(["fill", "name", "pitch_um"], detector.Properties.Select(p => p.Name));
        Assert.Equal(["integer"], detector.Properties[0].Types);
        Assert.Equal(["null"], detector.Properties[1].Types);
        Assert.Equal(["number"], detector.Properties[2].Types);
        Assert.True(detector.Properties[2].Accepts(0.5), "a number default keeps the number type");
    }

    /// <summary>Settings are reduced to what the schema declares: unknown keys, values of another type and values outside an enum are dropped.</summary>
    [Fact]
    public void AcceptedSettingsFollowTheSchema()
    {
        var output = Descriptor().AcceptedSettings(EngineTestSupport.Settings(
            ("rings", 9), ("seed", 42), ("fields", 2), ("wavelengths", "all"), ("pattern", "random"), ("reference", "chief"),
            ("max_frequency", 62.5)));
        Assert.Equal(["fields", "max_frequency", "rings", "seed"], output.Keys.Order());
        Assert.True(Json.DeepEquals(output["rings"], 9) && Json.DeepEquals(output["max_frequency"], 62.5));
        var nullable = Descriptor().AcceptedSettings(EngineTestSupport.Settings(("fields", null), ("seed", ulong.MaxValue)));
        Assert.Equal(2, nullable.Count);
        Assert.Null(nullable["fields"]);
        Assert.Equal(ulong.MaxValue, nullable["seed"].UInt64OrNull());
    }

    [Fact]
    public void BuffersIndexRowMajorAndCompareByContent()
    {
        var b = ResultBuffer.FromDoubles("n_after", [1, 2, 3, 4, 5, 6], [2, 3]);
        Assert.Equal(4, b[1, 0]);
        Assert.Equal(3, b[0, 2]);
        Assert.Equal(6, b[5]);
        Assert.Equal(6, b[[1, 2]]);
        Assert.Equal([1.0, 2, 3, 4, 5, 6], b.AsSpan<double>().ToArray());
        Assert.False(b.TryGetSpan<long>(out _), "dtype mismatch");
        Assert.Throws<InvalidOperationException>(() => b.AsSpan<float>().Length);
        Assert.Throws<IndexOutOfRangeException>(() => b[2, 0]);
        Assert.Throws<ArgumentException>(() => b[[1]]);
        Assert.Equal(b, ResultBuffer.FromDoubles("n_after", [1, 2, 3, 4, 5, 6], [2, 3]));
        Assert.NotEqual(b, ResultBuffer.FromDoubles("n_after", [1, 2, 3, 4, 5, 7], [2, 3]));
        Assert.Equal("n_after: f64[2×3]", b.ToString());
        Assert.Equal([4.0, 5, 6], b.Row(1));
        Assert.Equal(2, b.Rows);
        // Every dtype converts, element by element and as a whole.
        var cases = new (ResultBuffer Buffer, double[] Expected)[]
        {
            (ResultBuffer.Create<double>("b", [1.5, -2.25]), [1.5, -2.25]),
            (ResultBuffer.Create<float>("b", [1.5f, -2.25f]), [1.5, -2.25]),
            (ResultBuffer.Create<int>("b", [-7, 8]), [-7, 8]),
            (ResultBuffer.Create<uint>("b", [7, uint.MaxValue]), [7, 4_294_967_295]),
            (ResultBuffer.Create<long>("b", [-7, 1L << 40]), [-7, 1_099_511_627_776]),
            (ResultBuffer.Create<ulong>("b", [7, 1UL << 40]), [7, 1_099_511_627_776]),
            (ResultBuffer.Create<byte>("b", [0, 255]), [0, 255]),
        };
        foreach (var (buffer, expected) in cases)
        {
            Assert.Equal(expected, buffer.ToDoubleArray());
            Assert.Equal(expected, expected.Select((_, i) => buffer[i]));
            Assert.Equal(expected.Length * buffer.DType.ByteWidth(), buffer.Bytes.Length);
        }
        Assert.Empty(ResultBuffer.FromDoubles("empty", []).ToDoubleArray());
        Assert.Equal(BufferDType.F32, BufferDTypeExtensions.FromName("f32"));
        Assert.Equal(1, BufferDType.U8.ByteWidth());
        Assert.Throws<ArgumentException>(() => ResultBuffer.FromDoubles("bad", [1, 2, 3], [2, 2]));
        // A disposed buffer refuses access.
        b.Dispose();
        b.Dispose();
        Assert.Throws<ObjectDisposedException>(() => b[0]);
    }

    /// <summary><c>IntOrNull</c> reads whole numbers a long holds, and nothing else.</summary>
    [Fact]
    public void IntReadsOnlyWholeNumbersInRange()
    {
        Assert.Equal(-3, JsonValue.Create(-3.0).IntOrNull());
        Assert.Equal(7, JsonValue.Create(7).IntOrNull());
        Assert.Equal(3, JsonNode.Parse("3.0").IntOrNull());
        Assert.Null(JsonValue.Create(2.5).IntOrNull());
        foreach (var bad in new[] { 1e30, -1e30, 9.3e18 }) Assert.Null(JsonValue.Create(bad).IntOrNull());
        Assert.Null(JsonValue.Create(ulong.MaxValue).IntOrNull());
        Assert.Null(JsonValue.Create("3").IntOrNull());
        Assert.Null(JsonNode.Parse("[1, 1e30]").IntArray());
        Assert.Equal([1L, 2], JsonNode.Parse("[1, 2.0]").IntArray()!);
    }

    [Fact]
    public void JsonEqualityComparesNumbersByValue()
    {
        Assert.True(Json.DeepEquals(JsonNode.Parse("6"), JsonNode.Parse("6.0")));
        Assert.True(Json.DeepEquals(JsonNode.Parse("100"), JsonNode.Parse("1e2")));
        Assert.True(Json.DeepEquals(JsonValue.Create(6), JsonNode.Parse("6")));
        Assert.True(Json.DeepEquals(JsonValue.Create(0.1), JsonNode.Parse("0.1")));
        // Seeds above 2^53 stay distinct although they collide as doubles.
        Assert.False(Json.DeepEquals(JsonValue.Create(1L << 60), JsonValue.Create((1L << 60) + 1)));
        Assert.True(Json.DeepEquals(JsonNode.Parse("12345678901234567891"), JsonValue.Create(12_345_678_901_234_567_891UL)));
        Assert.True(Json.DeepEquals(JsonNode.Parse("""{"a":1,"b":[true,null]}"""), JsonNode.Parse("""{"b":[true,null],"a":1.0}""")));
        Assert.False(Json.DeepEquals(JsonNode.Parse("""{"a":1}"""), JsonNode.Parse("""{"a":1,"b":null}""")));
        Assert.False(Json.DeepEquals(JsonNode.Parse("\"1\""), JsonNode.Parse("1")));
        Assert.Equal("integer", Json.TypeName(JsonNode.Parse("3")));
        Assert.Equal("number", Json.TypeName(JsonNode.Parse("3.0")));
        Assert.Equal("number", Json.TypeName(JsonValue.Create(3.0)));
        Assert.Equal("integer", Json.TypeName(JsonValue.Create(3L)));
    }

    [Fact]
    public void ErrorCodesMapToCases()
    {
        Assert.Equal(EngineErrorKind.Cancelled, EngineException.From(EngineErrorCode.Cancelled, "x").Kind);
        var model = EngineException.From(EngineErrorCode.InvalidModel, "bad stop");
        Assert.Equal(EngineErrorKind.InvalidModel, model.Kind);
        Assert.Equal(["bad stop"], model.Issues);
        Assert.Equal("Invalid model: bad stop", model.Message);
        var gpu = EngineException.From(EngineErrorCode.Unsupported, "no gpu");
        Assert.True(gpu.Kind == EngineErrorKind.Unsupported && gpu.Detail == "no gpu");
        var failed = EngineException.From(EngineErrorCode.JobFailed, "boom");
        Assert.True(failed.Kind == EngineErrorKind.Failed && failed.Message == "boom");
        var io = EngineException.From(EngineErrorCode.Io, "disk");
        Assert.True(io.Kind == EngineErrorKind.Engine && io.Code == EngineErrorCode.Io && io.Detail == "disk");
        // What a failed job reports: the code of its own error.
        var glass = EngineException.From(EngineErrorCode.NotFound, "unknown material 'X'");
        Assert.True(glass.Kind == EngineErrorKind.Engine && glass.Code == EngineErrorCode.NotFound);
        Assert.Equal("diverged", EngineException.From(EngineErrorCode.Compute, "diverged").Message);
        Assert.Equal("Engine error: I/O error", EngineException.Engine(EngineErrorCode.Io, "").Message);
        foreach (var code in Enum.GetValues<EngineErrorCode>())
        {
            Assert.Equal(code, EngineException.From(code, "m").Code);
        }
        var settings = EngineException.InvalidSettings(["rings: expected integer"]);
        Assert.True(settings.Kind == EngineErrorKind.InvalidSettings && settings.Code == EngineErrorCode.InvalidArgument);
        Assert.Null(EngineException.LibraryUnavailable("x").Code);
    }

    [Fact]
    public void RunInfoToleratesUnknownPrecision()
    {
        var info = RunInfo.FromJson(JsonNode.Parse("""
            {"engine_version":"9","snapshot_hash":"ab","analysis":"x","settings":{},"backend":"vulkan",
             "precision":"f16_tensor","seed":null,"started_unix_ms":1,"duration_ms":2.5}
            """));
        Assert.Equal(Precision.F32Gpu, info.Precision);
        Assert.Equal(["unknown precision 'f16_tensor'"], info.Warnings);
        Assert.Null(info.Seed);
        Assert.Throws<EngineException>(() => RunInfo.FromJson(JsonNode.Parse("""{"engine_version":"9"}""")));
    }

    /// <summary>The envelope's <c>backend</c> object keeps the engine's <c>cost_decision</c> as sent; a missing or null one is none.</summary>
    [Fact]
    public void BackendReportCarriesTheCostDecision()
    {
        var report = BackendReport.FromJson(JsonNode.Parse("""
            {"requested":"default:auto","used":"gpu:metal","precision":"f32_gpu","device":"Apple M2 Pro","fallback_reason":null,
             "execution":[],"cost_decision":{"decision":"qualified_metal","rule":"cost model","predicted_ms":{"cpu":556.6,"metal":312.3}}}
            """))!;
        Assert.True(report.Requested == "default:auto" && report.Used == "gpu:metal" && report.FallbackReason is null);
        Assert.Equal("qualified_metal", report.CostDecision.At("decision").StringOrNull());
        Assert.Equal(312.3, report.CostDecision.At("predicted_ms", "metal").Nf());
        Assert.Equal("Metal (Apple M2 Pro) · f32", report.Summary);
        var explicitCpu = BackendReport.FromJson(JsonNode.Parse("""{"requested":"cpu_reference","cost_decision":null}"""))!;
        Assert.True(explicitCpu.Requested == "cpu_reference" && explicitCpu.CostDecision is null);
        Assert.Null(BackendReport.FromJson(JsonNode.Parse("""{"requested":"auto"}"""))!.CostDecision);
        Assert.Null(BackendReport.FromJson(null));
        Assert.Equal("CPU reference · f64", new BackendReport("auto", "cpu_reference", "f64_reference").Summary);
    }

    static RunInfo Info(SnapshotHash hash, JsonNode settings, string analysis = "mtf") => new()
    {
        EngineVersion = "1", SnapshotHash = hash, Analysis = analysis, Settings = settings, Backend = "cpu_reference",
        Precision = Precision.F64Reference,
    };

    /// <summary>Staleness compares the request that produced a result and both hashes (system and glass library).</summary>
    [Fact]
    public void OutputStalenessUsesRequestAndLibraryHash()
    {
        var snap = new ModelSnapshot("{}", new SnapshotHash("aa"), "L1");
        var request = new EngineAnalysisRequest("mtf", EngineTestSupport.Settings(("max_frequency", 100)));
        var info = Info(snap.Hash, new JsonObject { ["normalised"] = true });
        var output = new AnalysisOutput(info, null, libraryHash: "L1", engineRequest: request);
        Assert.True(output.IsCurrent(snap, request), "engine-normalised settings do not matter");
        Assert.False(output.IsCurrent(snap, new EngineAnalysisRequest("mtf", EngineTestSupport.Settings(("max_frequency", 50)))));
        Assert.False(output.IsCurrent(snap, new EngineAnalysisRequest("psf", request.Settings)));
        Assert.False(output.IsCurrent(null, request), "no snapshot");
        var newLibrary = snap with { LibraryHash = "L2" };
        Assert.NotEqual(newLibrary, snap);
        Assert.False(output.IsCurrent(newLibrary, request), "a catalog change makes it stale");
        Assert.False(output.IsCurrent(snap with { Hash = new SnapshotHash("bb") }, request), "an edit makes it stale");
        Assert.Equal(snap, snap with { SystemJson = "{\"other\":1}" });
        // Without its request: the engine's recorded settings must contain the requested ones.
        var recorded = new EngineAnalysisRequest("mtf", EngineTestSupport.Settings(("normalised", true)));
        var bare = output.WithRequest(null);
        Assert.True(bare.IsCurrent(snap, recorded));
        Assert.False(bare.IsCurrent(snap, new EngineAnalysisRequest("mtf", EngineTestSupport.Settings(("normalised", false)))));
        Assert.False(bare.IsCurrent(newLibrary, recorded));
        Assert.False(bare.IsCurrent(snap, new EngineAnalysisRequest("mtf", EngineTestSupport.Settings(("missing", null)))), "a missing key is not null");
    }

    /// <summary>The backend preference is part of a job's identity; <c>fresh</c> is not.</summary>
    [Fact]
    public void RequestEqualityAndWireJson()
    {
        var plain = new EngineAnalysisRequest("psf", EngineTestSupport.Settings(("grid_size", 32)));
        var wire = (EngineAnalysisRequest r, string? gpu) => JsonNode.Parse(FFIEngine.WireJson(r, gpu))!.AsObject();
        Assert.False(wire(plain, null).ContainsKey("fresh"));
        var fresh = plain.Freshened();
        Assert.True(fresh.Fresh && fresh == plain && fresh.GetHashCode() == plain.GetHashCode());
        Assert.True(wire(fresh, null)["fresh"]!.GetValue<bool>() == true);
        Assert.Equal("psf", wire(fresh, null)["analysis"]!.GetValue<string>());
        Assert.Equal(32, wire(plain, null)["settings"].At("grid_size").IntOrNull());
        var gpu = plain.With(new ExecutionSettings(BackendPreference.Gpu));
        Assert.NotEqual(plain, gpu);
        Assert.False(wire(gpu, null).ContainsKey("backend"), "no GPU backend: automatic");
        Assert.Equal("""{"kind":"gpu","name":"vulkan"}""", wire(gpu, "vulkan")["backend"]!.ToJsonString());
        Assert.True(wire(gpu, "vulkan")["allow_backend_fallback"]!.GetValue<bool>() == true);
        var cpu = plain.With(new ExecutionSettings(BackendPreference.CpuReference));
        Assert.Equal("""{"kind":"cpu_reference"}""", wire(cpu, "vulkan")["backend"]!.ToJsonString());
        Assert.Equal(plain, new EngineAnalysisRequest("psf", EngineTestSupport.Settings(("grid_size", 32.0))));
        Assert.NotEqual(plain, plain.With("grid_size", 64));
        // Settings are copied: editing the caller's node does not change the request.
        var node = new JsonObject { ["a"] = 1 };
        var r = new EngineAnalysisRequest("x", EngineTestSupport.Settings(("nested", node)));
        node["a"] = 2;
        Assert.Equal(1, r.Settings["nested"].At("a").IntOrNull());
    }

    [Fact]
    public void CatalogAndGlassJsonRead()
    {
        var catalog = CatalogInfo.FromJson(JsonNode.Parse("""{"name":"MYCAT","source":"user","path":"/x.agf","glass_count":3}"""));
        Assert.True(catalog.InProject && !catalog.Library && !catalog.IsBuiltin, "older engines: every user catalog is in the project");
        var status = GlassStatus.FromJson(JsonNode.Parse("""{"other":7}"""));
        Assert.Equal(7, status!.Value.Code);
        Assert.Equal("Status 7", status.Value.Title);
        Assert.Equal("Preferred", GlassStatus.FromJson(JsonValue.Create("preferred"))!.Value.Title);
        var o = new GlassOverride("Schott", "N-BK7", GlassStatus.Obsolete);
        Assert.Equal("""{"catalog":"Schott","name":"N-BK7","status":"obsolete"}""", o.ToJson().ToJsonString());
        Assert.Equal(o, GlassOverride.FromJson(o.ToJson()));
        Assert.True(o.Matches("SCHOTT", "n-bk7"));
        var row = GlassRow.FromJson(JsonNode.Parse("""
            {"catalog":"S","name":"G","nd":1.5,"vd":60,"pgf":null,"min_um":0.3,"max_um":"inf","status":"melt","tce":7.1}
            """))!;
        Assert.True(row.MaxUm == double.PositiveInfinity && row.UnrestrictedRange && row.Pgf is null && row.Tce == 7.1);
        Assert.Equal(GlassStatus.Melt, row.Status);
        var changes = CatalogChanges.FromJson(JsonNode.Parse("""
            {"catalog":"S","added":["A","B"],"removed":[],"changed":[{"name":"G","nd_before":1.5,"nd_after":1.51,"vd_before":60,"vd_after":59,"fields":["nd"]}],"unchanged":4}
            """))!;
        Assert.Equal("2 added, 1 changed", changes.Summary);
        Assert.Equal(4, changes.Unchanged);
    }
}
