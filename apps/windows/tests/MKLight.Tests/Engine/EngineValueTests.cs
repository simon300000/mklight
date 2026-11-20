// Tests await jobs and their progress without the test's token on purpose: the job's own
// cancellation is what they check.
#pragma warning disable xUnit1051
using System.Text.Json.Nodes;
using MKLight.Engine;

namespace MKLight.Tests.Engine;

public class RunInfoTests
{
    static JsonObject Fixture() => JsonNode.Parse(File.ReadAllText(EngineTestSupport.Fixture("run_info.json")))!.AsObject();

    [Fact]
    public void DecodesEngineRunInfo()
    {
        var json = Fixture();
        var info = RunInfo.FromJson(json);
        Assert.Equal(Precision.F32Gpu, info.Precision);
        Assert.Equal(42UL, info.Seed);
        Assert.Equal(64, info.SnapshotHash.Hex.Length);
        Assert.Equal(12, info.SnapshotHash.Short.Length);
        Assert.True(Json.DeepEquals(info.Settings.At("rings"), 6) && Json.DeepEquals(info.Settings.At("rings"), 6.0));
        Assert.True(info.Tier == Tier.Design && info.AimStop == AimStopKind.Requested && info.FailedAttempts.Count == 0);
        Assert.True(Json.DeepEquals(json, info.ToJson()), info.ToJson().ToJsonString());
    }

    /// <summary>
    /// The tier, the aim stop and the failed backend attempts round-trip; a result without them
    /// reads as the engine's defaults, and a tier this app does not know reads as not design
    /// grade, with a warning.
    /// </summary>
    [Fact]
    public void DecodesTierAimStopAndFailedAttempts()
    {
        var o = Fixture();
        o["tier"] = "preview";
        o["aim_stop"] = "conditioned";
        o["failed_attempts"] = JsonNode.Parse("""[{"backend":"gpu:metal","outcome":"failed","reason":"device lost","elapsed_ms":3.5}]""");
        var info = RunInfo.FromJson(o);
        Assert.True(info.Tier == Tier.Preview && info.AimStop == AimStopKind.Conditioned);
        Assert.Equal([new BackendAttempt("gpu:metal", "failed", "device lost", 3.5)], info.FailedAttempts);
        Assert.True(Json.DeepEquals(o, info.ToJson()));
        o["tier"] = "draft";
        var unknown = RunInfo.FromJson(o);
        Assert.True(unknown.Tier == Tier.Preview && unknown.Warnings.Contains("unknown tier 'draft'"));
        o.Remove("tier");
        o.Remove("aim_stop");
        o.Remove("failed_attempts");
        var bare = RunInfo.FromJson(o);
        Assert.True(bare.Tier == Tier.Design && bare.AimStop == AimStopKind.Requested && bare.FailedAttempts.Count == 0);
    }

    /// <summary>Seeds are full 64-bit values: the run info keeps every digit, also above <c>long.MaxValue</c>.</summary>
    [Fact]
    public void SeedsAbove2To53AreExact()
    {
        var o = Fixture();
        o["seed"] = 12_345_678_901_234_567_891UL;
        o["settings"]!["seed"] = 12_345_678_901_234_567_891UL;
        var text = RunInfo.FromJson(o).ToJson().ToJsonString();
        Assert.Contains("\"seed\":12345678901234567891", text);
        var decoded = RunInfo.FromJson(JsonNode.Parse(text));
        Assert.Equal(12_345_678_901_234_567_891UL, decoded.Seed);
        Assert.Equal(12_345_678_901_234_567_891UL, decoded.Settings.At("seed").UInt64OrNull());
        var request = new EngineAnalysisRequest("spot_diagram", EngineTestSupport.Settings(("seed", 12_345_678_901_234_567_891UL)));
        Assert.True(request.Matches(decoded));
        Assert.False(new EngineAnalysisRequest("spot_diagram", EngineTestSupport.Settings(("seed", 12_345_678_901_234_567_890UL))).Matches(decoded));
    }
}

/// <summary><c>mk_engine_gpu_preparation_json</c>'s states, in the engine's exact JSON.</summary>
public class GpuPreparationTests
{
    [Fact]
    public void DecodesEveryState()
    {
        Assert.Equal(new GpuPreparation.Idle(), GpuPreparation.FromJson("""{"state":"idle"}"""));
        var preparing = GpuPreparation.FromJson("""{"state":"preparing","device":"RTX 4090","done":41,"total":205}""");
        Assert.Equal(new GpuPreparation.Preparing("RTX 4090", 41, 205), preparing);
        Assert.True(preparing.IsPreparing);
        Assert.Equal(0.2, preparing.Fraction, 12);
        Assert.Equal(
            new GpuPreparation.Ready("RTX 4090", 205, 205, 0.005),
            GpuPreparation.FromJson("""{"state":"ready","device":"RTX 4090","kernels":205,"from_cache":205,"seconds":0.005}"""));
        Assert.Equal(new GpuPreparation.Unavailable("no device"), GpuPreparation.FromJson("""{"state":"unavailable","reason":"no device"}"""));
        Assert.Equal(
            new GpuPreparation.Failed("RTX 4090", "x"), GpuPreparation.FromJson("""{"state":"failed","device":"RTX 4090","message":"x"}"""));
        Assert.False(new GpuPreparation.Idle().IsPreparing);
        Assert.Equal(new GpuPreparation.Idle(), GpuPreparation.FromJson("""{"state":"something_new"}"""));
        Assert.Throws<EngineException>(() => GpuPreparation.FromJson("""{"state":"ready","device":"d"}"""));
        Assert.Throws<EngineException>(() => GpuPreparation.FromJson("not json"));
    }

    /// <summary><c>ready</c> carries the engine's <c>warning</c> when the saved kernels could not be read or written; without the key there is none.</summary>
    [Fact]
    public void ReadyCarriesTheKernelCacheWarning()
    {
        var warned = GpuPreparation.FromJson(
            """{"state":"ready","device":"RTX 4090","kernels":205,"from_cache":0,"seconds":31.2,"warning":"kernel archive: permission denied"}""");
        Assert.Equal(new GpuPreparation.Ready("RTX 4090", 205, 0, 31.2, "kernel archive: permission denied"), warned);
        var plain = GpuPreparation.FromJson("""{"state":"ready","device":"RTX 4090","kernels":205,"from_cache":205,"seconds":0.005}""");
        Assert.Equal(new GpuPreparation.Ready("RTX 4090", 205, 205, 0.005, null), plain);
        Assert.NotEqual(plain, warned);
    }

    [Fact]
    public void TotalNotYetKnownIsZeroProgress()
    {
        var s = GpuPreparation.FromJson("""{"state":"preparing","device":"","done":0,"total":0}""");
        Assert.True(s.IsPreparing);
        Assert.Equal(0, s.Fraction);
    }
}

/// <summary>The engine writes non-finite <c>f64</c> as <c>null</c> and infinite lengths as <c>"inf"</c>. Readers must take both.</summary>
public class NonFiniteResultTests
{
    const string GlassJson = """{"catalog":"X","name":"G","nd":1.5,"vd":null,"min_um":"-inf","max_um":"inf"}""";

    [Fact]
    public void DecodesNullAndInfinityStrings()
    {
        var g = GlassInfo.FromJson(JsonNode.Parse(GlassJson));
        Assert.True(g.Nd == 1.5 && double.IsNaN(g.Vd) && g.MinUm == double.NegativeInfinity && g.MaxUm == double.PositiveInfinity);
        var back = GlassInfo.FromJson(g.ToJson());
        Assert.Equal(g, back);
        Assert.True(Json.DeepEquals(g.ToJson(), JsonNode.Parse(GlassJson)));
        Assert.Throws<FormatException>(() => GlassInfo.FromJson(JsonNode.Parse(GlassJson.Replace("\"inf\"", "\"wide\""))));
        // The tolerant accessors read the same forms from a result's JSON.
        var tree = JsonNode.Parse(GlassJson);
        Assert.True(double.IsNaN(tree.NfAt("vd")!.Value));
        Assert.Equal(double.NegativeInfinity, tree.NfAt("min_um"));
        Assert.Null(tree.NfAt("name"));
        Assert.Null(tree.NfAt("missing"));
        Assert.Equal(double.PositiveInfinity, JsonValue.Create("+inf").Nf());
        Assert.True(double.IsNaN(JsonValue.Create("NaN").Nf()!.Value));
        Assert.Equal([1.0, double.PositiveInfinity, double.NaN], JsonNode.Parse("""[1, "inf", null]""").NfArray()!);
        Assert.Null(JsonNode.Parse("""[1, "x"]""").NfArray());
        Assert.Equal(-1e300, JsonValue.Create("-1e300").DoubleValue());
        Assert.Equal(2.5, JsonNode.Parse("""{"a":[{"b":2.5}]}""").At("a", 0, "b").DoubleValue());
        Assert.Null(JsonNode.Parse("""{"a":[]}""").At("a", 0, "b"));
        // Points compare the same way, so a scene with an undefined point still equals itself.
        Assert.Equal(new Point3(0, double.NaN, 1), new Point3(0, double.NaN, 1));
        Assert.NotEqual(new Point3(0, double.NaN, 1), new Point3(0, 2, 1));
    }
}

public class PrescriptionImportDecodingTests
{
    [Fact]
    public void DecodesTheEngineResult()
    {
        var r = PrescriptionImport.FromJson("""
            {"format": "seq", "encoding": "utf8", "system": {"name": "x", "surfaces": []},
             "warnings": [
               {"kind": "unknown_glass", "line": 4, "surface": 1, "message": "glass X …"},
               {"kind": "something_new", "line": null, "surface": null, "message": "?"}
             ], "snapshot_hash": "abc"}
            """);
        Assert.True(r.Format == PrescriptionFormat.Seq && r.Encoding == "utf8");
        Assert.Equal([ImportWarningKind.UnknownGlass, ImportWarningKind.Other], r.Warnings.Select(w => w.Kind));
        Assert.True(r.Warnings[0].Line == 4 && r.Warnings[0].Surface == 1 && r.Warnings[1].Line is null);
        Assert.Equal("x", JsonNode.Parse(r.SystemJson).At("name").StringOrNull());
        Assert.Equal(new SnapshotHash("abc"), r.SnapshotHash);
        Assert.Equal(PrescriptionFormat.Zmx, PrescriptionFormats.FromExtension("ZMX"));
        Assert.Equal(PrescriptionFormat.Seq, PrescriptionFormats.FromExtension(".seq"));
        Assert.Null(PrescriptionFormats.FromExtension("txt"));
    }
}

/// <summary>The defaults an engine without a capability answers with.</summary>
public class EngineClientDefaultsTests
{
    sealed class Bare : IEngineClient
    {
        public EngineInfo Info => new("bare", "0", IsMock: true);
        public Task<ModelSnapshot> OpenSnapshotAsync(string systemJson) => Task.FromResult(new ModelSnapshot(systemJson, new("h")));
        public Task<IReadOnlyList<AnalysisDescriptor>> AnalysisDescriptorsAsync() => Task.FromResult<IReadOnlyList<AnalysisDescriptor>>([]);
        public AnalysisJob Submit(EngineAnalysisRequest request, ModelSnapshot snapshot) =>
            AnalysisJob.Run(request, snapshot.Hash, (_, _) => throw EngineException.Unsupported(request.Analysis));
        public Task<IReadOnlyList<string>> MaterialNamesAsync() => Task.FromResult<IReadOnlyList<string>>([]);
    }

    [Fact]
    public async Task EnginesWithoutCapabilitiesAnswerEmptyOrUnsupported()
    {
        IEngineClient e = new Bare();
        Assert.Same(e, e.MakeDocumentEngine());
        Assert.Empty(await e.CatalogsAsync());
        Assert.Equal(new GpuPreparation.Idle(), e.GetGpuPreparation());
        var snap = await e.OpenSnapshotAsync("{}");
        var unsupported = await Assert.ThrowsAsync<EngineException>(() => e.LoadCatalogAsync("x.agf"));
        Assert.Equal(EngineErrorKind.Unsupported, unsupported.Kind);
        Assert.Throws<EngineException>(() => e.ImportPrescription([], PrescriptionFormat.Zmx));
        await Assert.ThrowsAsync<EngineException>(() => e.ImportPrescriptionAsync("lens.txt"));
        await Assert.ThrowsAsync<EngineException>(() => e.ApplySolvesAsync(new Dictionary<string, JsonNode?>(), snap));
        var validate = await Assert.ThrowsAsync<EngineException>(() => e.ValidateAsync(snap));
        Assert.Equal(EngineErrorKind.Unsupported, validate.Kind);
    }
}
