using System.Text.Json.Nodes;
using MKLight.App.Views.Analysis.RunDetails;

namespace MKLight.App.Tests.Status;

/// <summary>Run details of GPU runs: every <c>cost_decision</c> the engine sends, the request and the badge (docs/RESULTS.md §5).</summary>
public class RunRouteGpuTests
{
    static string? Line(string requested, string decision) => RunRoute.Line(requested, JsonNode.Parse(decision));

    [Fact]
    public void AnyGpuDeviceAndTheBrowserGpusRules()
    {
        Assert.Equal("GPU by predicted time (GPU predicted 6 ms, CPU 38 ms)",
            Line("auto", """{"decision": "qualified_cuda", "rule": "cost model", "version": 5, "before_device": false, "predicted_ms": {"cpu": 38.2, "cuda": 6.1}, "advantage": 6.26, "required": {"ratio": 1.5, "gain_ms": 5}}"""));
        Assert.Equal("GPU by measured timings (GPU predicted 3 ms, CPU 9 ms)",
            Line("auto", """{"decision": "qualified_webgpu", "rule": "webgpu calibration", "predicted_ms": {"cpu": 9, "webgpu": 3}}"""));
        Assert.Equal("CPU: the GPU's measured gain is too small (GPU predicted 8 ms, CPU 9 ms)",
            Line("auto", """{"decision": "cpu_reference", "rule": "webgpu calibration", "predicted_ms": {"cpu": 9, "webgpu": 8}}"""));
        Assert.Equal("GPU: a run above the measured break-even",
            Line("auto", """{"decision": "qualified_webgpu", "rule": "webgpu break-even", "rays": 90000, "min_rays": 40000}"""));
        Assert.Equal("CPU: a run below the measured break-even",
            Line("auto", """{"decision": "cpu_reference", "rule": "webgpu break-even", "rays": 900, "min_rays": 40000}"""));
        Assert.Equal("CPU: no measured GPU gain for this analysis",
            Line("auto", """{"decision": "cpu_reference", "rule": "webgpu break-even", "rays": 900, "min_rays": null}"""));
        Assert.Equal("GPU, without a prediction", Line("auto", """{"decision": "qualified_cuda", "reason": "no cost decision"}"""));
        // An explicit request bypasses the cost rules: no route line.
        Assert.Null(Line("gpu:cuda", """{"decision": "explicit_backend", "reason": "cost rules bypassed"}"""));
    }

    [Fact]
    public void RequestedInWords()
    {
        Assert.Equal("Automatic", RunRoute.Requested("auto"));
        Assert.Equal("Automatic (the engine's default)", RunRoute.Requested("default:auto"));
        Assert.Equal("CPU reference", RunRoute.Requested("cpu_reference"));
        Assert.Equal("GPU (cuda)", RunRoute.Requested("gpu:cuda"));
        Assert.Equal("something_new", RunRoute.Requested("something_new"));
    }

    [Fact]
    public void BadgeAndFallback()
    {
        Assert.Equal("GPU · f32 · cuda", RunRoute.Badge("gpu:cuda", "f32_gpu"));
        Assert.Equal("CPU · f64", RunRoute.Badge("cpu_reference", "f64_reference"));
        Assert.Equal("GPU · f32", RunRoute.Badge("gpu", ""));
        Assert.Equal("GPU failed after 12 ms; completed on the CPU", RunRoute.AttemptTitle("gpu:cuda", "failed", 12.3));
        Assert.Equal("GPU not tried: the same input already failed on it", RunRoute.AttemptTitle("gpu:cuda", "skipped", 0));
    }
}
