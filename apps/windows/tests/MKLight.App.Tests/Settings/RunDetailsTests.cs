using System.Text.Json.Nodes;
using MKLight.App.Views.Analysis.Results.Readers;
using MKLight.App.Views.Analysis.RunDetails;

namespace MKLight.App.Tests.Settings;

/// <summary>Route wording from <c>backend.cost_decision</c> and attempts from <c>run_info.failed_attempts</c> (docs/RESULTS.md §5).</summary>
public class RunRouteTests
{
    static string? Line(string requested, string decision) => RunRoute.Line(requested, JsonNode.Parse(decision));

    [Fact]
    public void CostModelDecisions()
    {
        Assert.Equal("GPU by predicted time (GPU predicted 12 ms, CPU 31 ms)",
            Line("auto", """{"decision": "qualified_metal", "rule": "cost model", "predicted_ms": {"cpu": 31.2, "metal": 12.4}, "version": 3}"""));
        Assert.Equal("CPU: the GPU's predicted gain is too small (GPU predicted 20 ms, CPU 24 ms)",
            Line("default:auto", """{"decision": "cpu_reference", "rule": "cost model", "predicted_ms": {"cpu": 24, "metal": 20}}"""));
        Assert.Equal("CPU by predicted time (GPU predicted 1.5 s, CPU <1 ms)",
            Line("auto", """{"decision": "cpu_reference", "rule": "cost model", "predicted_ms": {"cpu": 0.2, "metal": 1500}}"""));
        // Any device names its own key.
        Assert.Equal("GPU by predicted time (GPU predicted 3 ms, CPU 9 ms)",
            Line("auto", """{"decision": "qualified_webgpu", "rule": "cost model", "predicted_ms": {"cpu": 9, "webgpu": 3}}"""));
        Assert.Equal("CPU by predicted time", Line("auto", """{"decision": "cpu_reference", "rule": "cost model"}"""));
    }

    [Fact]
    public void RuleDecisions()
    {
        Assert.Equal("CPU: this analysis is not accurate enough on the GPU", Line("auto", """{"decision": "cpu_reference", "rule": "accuracy exclusion", "evidence": "x"}"""));
        Assert.Equal("CPU: the GPU takes no part of a run of this size and shape", Line("auto", """{"decision": "cpu_reference", "rule": "no device share", "reason": "x"}"""));
        Assert.Equal("CPU: this request already failed on the GPU; kept from the first identical request",
            Line("auto", """{"decision": "cpu_reference", "rule": "known failure", "repeated": true}"""));
        Assert.Equal("GPU: no timing data for this device, so no prediction",
            Line("auto", """{"decision": "qualified_metal", "reason": "no cost calibration for this device", "version": 3}"""));
        Assert.Equal("GPU: no timing data for this analysis, so no prediction",
            Line("auto", """{"decision": "qualified_metal", "reason": "no calibration for this analysis"}"""));
        Assert.Equal("GPU, without a prediction", Line("auto", """{"decision": "qualified_metal", "reason": "no cost decision"}"""));
    }

    [Fact]
    public void NoLineForExplicitRequestsOrUnknownDecisions()
    {
        Assert.Null(Line("cpu_reference", """{"decision": "explicit_backend", "reason": "cost rules bypassed"}"""));
        Assert.Null(Line("auto", """{"decision": "explicit_backend"}"""));
        Assert.Null(Line("auto", """{"decision": "cpu_reference", "rule": "a rule from the future"}"""));
        Assert.Null(RunRoute.Line("auto", null));
    }

    [Fact]
    public void AttemptsAndTier()
    {
        Assert.Equal("GPU failed after 12 ms; completed on the CPU", RunRoute.AttemptTitle("gpu:metal", "failed", 12.3));
        Assert.Equal("GPU failed; completed on the CPU", RunRoute.AttemptTitle("gpu:vulkan", "failed", 0));
        Assert.Equal("GPU not tried: the same input already failed on it", RunRoute.AttemptTitle("gpu:metal", "skipped", 0));
        Assert.Equal("GPU path completed the result on the CPU (1.2 s)", RunRoute.AttemptTitle("gpu:metal", "reference_completion", 1234));
        Assert.Equal("CPU reference not used (timed out)", RunRoute.AttemptTitle("cpu_reference", "timed_out", 0));
        Assert.Null(RunRoute.Tier(false));
        Assert.Equal("Preview: cheaper sampling, not for final decisions", RunRoute.Tier(true));
    }
}

/// <summary>Run warnings as the wave solvers word them, grouped for reading.</summary>
public class RunWarningsTests
{
    const string Pupil = "wavefront changes by up to {0} waves between pupil samples (limit 0.5); increase the pupil sampling";

    [Fact]
    public void GroupsByMessageWithNumbersMaskedAndScopesFromOne()
    {
        var g = RunWarnings.Group([
            "field 0, wavelength 1: " + string.Format(Pupil, "1.16"),
            "field 2, wavelength 0: " + string.Format(Pupil, "0.72"),
            "field 0, wavelength 1: " + string.Format(Pupil, "0.9"),
            "only 5.0 % of the PSF energy lies inside the image grid; the PSF is truncated (enlarge the grid or its spacing)",
            "field (1, -1): no rays arrived",
            "surface 3: total internal reflection",
        ]);
        Assert.Equal(4, g.Count);
        Assert.Equal("Wavefront changes by up to 1.16 waves between pupil samples (limit 0.5); increase the pupil sampling", g[0].Message);
        Assert.Equal(["Field 1 · λ2", "Field 3 · λ1"], g[0].Scopes);
        Assert.Equal(WarningRemedy.PupilSampling, g[0].Remedy);
        Assert.Empty(g[1].Scopes);
        Assert.Equal(WarningRemedy.ImageGrid, g[1].Remedy);
        Assert.Equal(["field (1, −1)"], g[2].Scopes);
        Assert.Equal("Surface 3: Total internal reflection", g[3].Line);
        Assert.Equal(WarningRemedy.None, g[3].Remedy);
    }

    [Fact]
    public void UnscopedHeadsStayInTheMessage()
    {
        Assert.Equal((null, "psf_truncated: field [-1.0, -1.0], 3 %"), RunWarnings.Split("psf_truncated: field [-1.0, -1.0], 3 %"));
        Assert.Equal((null, "field two: x"), RunWarnings.Split("field two: x"));
        Assert.Equal("Field 2: A (and 1 more)", RunWarnings.Summary(["field 1: a", "b"]));
        Assert.Equal("", RunWarnings.Summary([]));
    }

    [Fact]
    public void RemediesSendCoarseExplicitSettingsBackToAutomatic()
    {
        var psf = new JsonObject { ["pupil"] = JsonNode.Parse("""{"grid": 32, "rays": 65}"""), ["grid_size"] = 64 };
        var r = RunRemedy.For(WarningRemedy.PupilSampling, k => psf[k])!;
        Assert.Equal("Sample the Pupil Automatically (grid 32 → auto)", r.Title);
        Assert.Equal("pupil", r.Key);
        Assert.True(Json.Same(JsonNode.Parse("""{"grid": 0, "rays": 65}"""), r.Value));
        var g = RunRemedy.For(WarningRemedy.ImageGrid, k => psf[k])!;
        Assert.Equal(("Size the Image Grid Automatically (64 → auto)", "grid_size"), (g.Title, g.Key));
        Assert.Equal(0, Json.Int(g.Value));

        // Already automatic: the engine's cap was reached, nothing to offer.
        var auto = new JsonObject { ["pupil"] = JsonNode.Parse("""{"grid": 0}"""), ["grid_size"] = 0 };
        Assert.Null(RunRemedy.For(WarningRemedy.PupilSampling, k => auto[k]));
        Assert.Null(RunRemedy.For(WarningRemedy.ImageGrid, k => auto[k]));
        Assert.Null(RunRemedy.For(WarningRemedy.None, k => psf[k]));

        // Image simulation keeps its pupil and PSF window in `optics`.
        var sim = new JsonObject { ["optics"] = JsonNode.Parse("""{"psf_size": 129, "field_grid": 3, "pupil": {"grid": 48, "rays": 65}}""") };
        var p = RunRemedy.For(WarningRemedy.PupilSampling, k => sim[k])!;
        Assert.Equal("optics", p.Key);
        Assert.Equal(0, Json.Int(p.Value["pupil"]!["grid"]));
        Assert.Equal(129, Json.Int(p.Value["psf_size"]));
        var w = RunRemedy.For(WarningRemedy.ImageGrid, k => sim[k])!;
        Assert.Equal("Size the PSF Window Automatically (129 → auto)", w.Title);
        Assert.Equal(0, Json.Int(w.Value["psf_size"]));
        Assert.Equal(48, Json.Int(w.Value["pupil"]!["grid"]));
    }
}
