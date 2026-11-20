using System.Text.Json.Nodes;
using MKLight.Engine;
using static MKLight.Tests.Engine.EngineTestSupport;

namespace MKLight.Tests.Engine;

/// <summary>
/// The compute backends: the engine's list (docs/FFI.md "Backends"), the picker's choices, a
/// request naming one GPU, and the envelope's backend report (docs/RESULTS.md §5).
/// </summary>
[Collection("FFI engine")]
public class ComputeBackendTests
{
    /// <summary>The list as <c>mk_engine_list_backends</c> documents it, with an available CUDA and an unavailable Vulkan backend.</summary>
    const string ListJson = """
        [{"backend": {"kind": "cpu_reference"}, "id": "cpu_reference", "title": "CPU reference (f64)", "available": true, "reason": null,
          "precision": "f64_reference", "capabilities": {"fp64": true, "reference": true, "analyses": [], "device_name": null, "unified_memory": null}},
         {"backend": {"kind": "gpu", "name": "cuda"}, "id": "gpu:cuda", "title": "CUDA", "available": true, "reason": null,
          "precision": "f32_gpu", "capabilities": {"fp64": false, "reference": false, "analyses": ["spot_diagram", "ray_fan"],
          "device_name": "RTX 4090", "unified_memory": false}},
         {"backend": {"kind": "gpu", "name": "vulkan"}, "id": "gpu:vulkan", "title": "Vulkan (f32)", "available": false,
          "reason": "no device passed the numerics probe", "precision": "f32_gpu", "capabilities": {"analyses": []}}]
        """;

    [Fact]
    public void ReadsTheEnginesList()
    {
        var list = EngineBackend.ListFromJson(ListJson);
        Assert.Equal(["cpu_reference", "gpu:cuda", "gpu:vulkan"], list.Select(b => b.Id));
        Assert.False(list[0].IsGpu);
        Assert.Equal("cuda", list[1].GpuName);
        Assert.Equal("RTX 4090", list[1].DeviceName);
        Assert.Equal(["spot_diagram", "ray_fan"], list[1].Analyses!);
        Assert.False(list[2].Available);
        Assert.Equal("CPU reference", list[0].DisplayTitle);
        Assert.Equal("CUDA (RTX 4090)", list[1].DisplayTitle);
        Assert.Equal("Vulkan", list[2].DisplayTitle);
        // An entry with only the backend object still has an id; one without either is skipped.
        var bare = EngineBackend.ListFromJson("""[{"backend": {"kind": "gpu", "name": "metal"}}, {"title": "x"}]""");
        Assert.Equal("gpu:metal", Assert.Single(bare).Id);
        Assert.Equal("Metal", bare[0].DisplayTitle);
    }

    [Fact]
    public void RegisterEmulationListsTheCpuF32BackendWithAShortTitle()
    {
        using var engine = new FFIEngine(new JsonObject { ["register_emulation"] = true });
        var emulation = Assert.Single(engine.BackendList(), b => b.Id == "gpu:cpu_f32");
        Assert.Equal("GPU emulation on the CPU", emulation.DisplayTitle);
        using var plain = new FFIEngine();
        Assert.DoesNotContain(plain.BackendList(), b => b.Id == "gpu:cpu_f32");
    }

    [Fact]
    public void PickerChoices()
    {
        var options = ComputeBackendOptions.From(EngineBackend.ListFromJson(ListJson));
        Assert.Equal(["Automatic", "CPU reference · f64", "CUDA (RTX 4090) · f32", "Vulkan · f32"], options.Select(o => o.Title));
        Assert.Equal([true, true, true, false], options.Select(o => o.Enabled));
        Assert.Equal("no device passed the numerics probe", options[3].Reason);
        Assert.Equal(new ExecutionSettings(BackendPreference.Gpu, "cuda"), options[2].Execution);
        Assert.Equal(0, ComputeBackendOptions.IndexOf(options, default));
        Assert.Equal(1, ComputeBackendOptions.IndexOf(options, new(BackendPreference.CpuReference)));
        Assert.Equal(3, ComputeBackendOptions.IndexOf(options, new(BackendPreference.Gpu, "vulkan")));
        // A GPU preference without a name is the first GPU.
        Assert.Equal(2, ComputeBackendOptions.IndexOf(options, new(BackendPreference.Gpu)));

        // Only the CPU: a GPU asked for (a design from another computer) is shown, disabled.
        var cpuOnly = EngineBackend.ListFromJson("""[{"backend": {"kind": "cpu_reference"}, "id": "cpu_reference", "title": "CPU reference (f64)", "available": true, "precision": "f64_reference"}]""");
        Assert.Equal(["Automatic", "CPU reference · f64"], ComputeBackendOptions.From(cpuOnly).Select(o => o.Title));
        var asked = ComputeBackendOptions.From(cpuOnly, new(BackendPreference.Gpu, "cuda"));
        Assert.Equal("CUDA · f32", asked[2].Title);
        Assert.False(asked[2].Enabled);
        Assert.Equal(2, ComputeBackendOptions.IndexOf(asked, new(BackendPreference.Gpu, "cuda")));
    }

    [Fact]
    public void ExecutionSettingsNameAGpuAndKeepEquality()
    {
        Assert.Equal(new ExecutionSettings(), default);
        Assert.Equal(new ExecutionSettings(BackendPreference.Auto, "cuda"), new ExecutionSettings(BackendPreference.Auto));
        Assert.Null(new ExecutionSettings(BackendPreference.CpuReference, "cuda").GpuName);
        Assert.NotEqual(new ExecutionSettings(BackendPreference.Gpu, "cuda"), new ExecutionSettings(BackendPreference.Gpu));
        Assert.NotEqual(new ExecutionSettings(BackendPreference.Gpu, "cuda"), new ExecutionSettings(BackendPreference.Gpu, "vulkan"));
        Assert.Equal(new ExecutionSettings(BackendPreference.Gpu, ""), new ExecutionSettings(BackendPreference.Gpu));
        foreach (var e in new ExecutionSettings[] { default, new(BackendPreference.CpuReference), new(BackendPreference.Gpu), new(BackendPreference.Gpu, "cuda") })
            Assert.Equal(e, ExecutionSettings.FromWireName(e.WireName));
        Assert.Equal(["auto", "cpu_reference", "gpu", "gpu:cuda"],
            new ExecutionSettings[] { default, new(BackendPreference.CpuReference), new(BackendPreference.Gpu), new(BackendPreference.Gpu, "cuda") }.Select(e => e.WireName));
        Assert.Equal(default, ExecutionSettings.FromWireName("gpu:"));
        Assert.Equal(default, ExecutionSettings.FromWireName(null));

        // Part of a request's identity: another GPU makes a result stale.
        var plain = new EngineAnalysisRequest("spot_diagram");
        Assert.NotEqual(plain.With(new ExecutionSettings(BackendPreference.Gpu, "cuda")), plain.With(new ExecutionSettings(BackendPreference.Gpu)));
        Assert.Equal(plain.With(new ExecutionSettings(BackendPreference.Gpu, "cuda")), plain.With(new ExecutionSettings(BackendPreference.Gpu, "cuda")));
    }

    [Fact]
    public void RequestJsonNamesTheChosenGpu()
    {
        JsonObject Wire(ExecutionSettings e, string? gpu) =>
            JsonNode.Parse(FFIEngine.WireJson(new EngineAnalysisRequest("spot_diagram", execution: e), gpu))!.AsObject();
        Assert.Equal("""{"kind":"gpu","name":"cuda"}""", Wire(new(BackendPreference.Gpu, "cuda"), null)["backend"]!.ToJsonString());
        Assert.Equal("""{"kind":"gpu","name":"cuda"}""", Wire(new(BackendPreference.Gpu, "cuda"), "metal")["backend"]!.ToJsonString());
        Assert.True(Wire(new(BackendPreference.Gpu, "cuda"), null)["allow_backend_fallback"]!.GetValue<bool>());
        Assert.Equal("""{"kind":"gpu","name":"metal"}""", Wire(new(BackendPreference.Gpu), "metal")["backend"]!.ToJsonString());
        Assert.False(Wire(new(BackendPreference.Gpu), null).ContainsKey("backend"));
        Assert.False(Wire(default, "cuda").ContainsKey("backend"));
    }

    [Fact]
    public void BackendReportAsResultsDocumentsIt()
    {
        var report = BackendReport.FromJson(JsonNode.Parse("""
            {"requested": "auto", "used": "gpu:cuda", "precision": "f32_gpu", "device": "NVIDIA GeForce RTX 4090",
             "fallback_reason": null, "execution": [],
             "cost_decision": {"decision": "qualified_cuda", "rule": "cost model", "predicted_ms": {"cpu": 38.2, "cuda": 6.1}}}
            """))!;
        Assert.Equal("CUDA (NVIDIA GeForce RTX 4090) · f32", report.Summary);
        Assert.Equal("cuda", report.UsedGpuName);
        Assert.Equal("qualified_cuda", report.CostDecision!["decision"]!.GetValue<string>());
        Assert.Equal("Metal (Apple M2 Pro) · f32", new BackendReport("auto", "gpu:metal", "f32_gpu", "Apple M2 Pro").Summary);
        Assert.Equal("WebGPU · f32", new BackendReport("auto", "gpu:webgpu", "f32_gpu").Summary);
        Assert.Equal("CPU reference · f64", new BackendReport("gpu:cuda", "cpu_reference", "f64_reference", null, "native spot_diagram failed after 12 ms").Summary);
        Assert.Null(new BackendReport("auto", "cpu_reference", "f64_reference").UsedGpuName);
    }

    [Fact]
    public void TheRealEnginesList()
    {
        RequireEngine();
        var list = SharedEngine().BackendList();
        var cpu = Assert.Single(list, b => b.Id == "cpu_reference");
        Assert.True(cpu.Available);
        Assert.Equal("f64_reference", cpu.Precision);
        var options = ComputeBackendOptions.From(list);
        Assert.Equal("Automatic", options[0].Title);
        Assert.Equal("CPU reference · f64", options[1].Title);
        Assert.All(options.Skip(2), o => Assert.Equal(BackendPreference.Gpu, o.Execution.Backend));
    }
}
