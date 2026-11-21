using System.Globalization;
using System.Text.Json.Nodes;
using MKLight.App.Support;
using MKLight.App.Views.Analysis.Results.Readers;

namespace MKLight.App.Views.Analysis.RunDetails;

/// <summary>
/// What the engine reports about where a run was computed, in words for the
/// run details: how an <c>auto</c> request was routed (the envelope's
/// <c>backend.cost_decision</c>, docs/RESULTS.md §5), the tier of a run that
/// was not design tier, and the backend attempts that did not compute the
/// result (<c>run_info.failed_attempts</c>).
/// </summary>
public static class RunRoute
{
    /// <summary>
    /// The route of an <c>auto</c> request in one line: the decision and, for
    /// a decision of the cost model, both predicted times. Null for an
    /// explicit request, a run without a decision and a decision this app does not know.
    /// </summary>
    public static string? Line(string requested, JsonNode? decision)
    {
        var asked = requested.StartsWith("default:", StringComparison.Ordinal) ? requested["default:".Length..] : requested;
        if (asked != "auto" || decision is not JsonObject d || Json.Str(d["decision"]) is not string chosen) return null;
        // The device decision is "qualified_<device>" (metal, webgpu, …).
        var gpu = chosen.StartsWith("qualified_", StringComparison.Ordinal);
        string line;
        switch (gpu ? "gpu" : chosen, Json.Str(d["rule"]))
        {
            case ("gpu", "cost model"):
                line = Loc.F("Analysis_Route_GpuPredicted", Predicted(d));
                break;
            case ("cpu_reference", "cost model"):
                // The GPU is chosen only for a clear predicted advantage.
                line = Loc.F(Times(d) is (double g, double c) && g < c ? "Analysis_Route_CpuPredictedGainSmall" : "Analysis_Route_CpuPredicted", Predicted(d));
                break;
            case ("cpu_reference", "accuracy exclusion"): line = Loc.S("Analysis_Route_CpuAccuracy"); break;
            case ("cpu_reference", "no device share"): line = Loc.S("Analysis_Route_CpuNoShare"); break;
            case ("cpu_reference", "known failure"): line = Loc.S("Analysis_Route_CpuKnownFailure"); break;
            // The browser GPU's own rules: its measured timings, or the break-even of its prior.
            case ("gpu", "webgpu calibration"): line = Loc.F("Analysis_Route_GpuMeasured", Predicted(d)); break;
            case ("cpu_reference", "webgpu calibration"): line = Loc.F("Analysis_Route_CpuMeasuredGainSmall", Predicted(d)); break;
            case ("gpu", "webgpu break-even"): line = Loc.S("Analysis_Route_GpuAboveBreakEven"); break;
            case ("cpu_reference", "webgpu break-even"):
                line = Loc.S(Json.Number(d["min_rays"]) is null ? "Analysis_Route_CpuNoMeasuredGain" : "Analysis_Route_CpuBelowBreakEven");
                break;
            case ("gpu", null):
                line = Loc.S(Json.Str(d["reason"]) switch
                {
                    "no cost calibration for this device" => "Analysis_Route_GpuNoDeviceTimings",
                    "no calibration for this analysis" => "Analysis_Route_GpuNoAnalysisTimings",
                    _ => "Analysis_Route_GpuNoPrediction",
                });
                break;
            default:
                return null;
        }
        return Json.Bool(d["repeated"]) == true ? Loc.F("Analysis_Route_Repeated", line) : line;
    }

    /// <summary>The tier of a run that was not design tier; null for design tier.</summary>
    public static string? Tier(bool preview) => preview ? Loc.S("Analysis_Route_PreviewTier") : null;

    /// <summary>A backend attempt that did not compute the result: what happened ("GPU failed after 12 ms; completed on the CPU").</summary>
    public static string AttemptTitle(string backend, string outcome, double elapsedMs)
    {
        var name = BackendName(backend);
        var took = elapsedMs > 0 ? Duration(elapsedMs) : null;
        return outcome switch
        {
            "failed" => took is null ? Loc.F("Analysis_Route_AttemptFailed", name) : Loc.F("Analysis_Route_AttemptFailedAfter", name, took),
            "skipped" => Loc.F("Analysis_Route_AttemptSkipped", name),
            "reference_completion" => took is null ? Loc.F("Analysis_Route_AttemptCompleted", name) : Loc.F("Analysis_Route_AttemptCompletedTook", name, took),
            _ => Loc.F("Analysis_Route_AttemptNotUsed", name, outcome.Replace('_', ' ')),
        };
    }

    /// <summary><c>gpu:metal</c> → <c>GPU</c>; other ids as the engine names them.</summary>
    public static string BackendName(string id) =>
        id.StartsWith("gpu:", StringComparison.Ordinal) ? "GPU" : id == "cpu_reference" ? Loc.S("Analysis_Route_CpuReference") : id;

    /// <summary>
    /// What was requested, in words: "Automatic", "CPU reference", "GPU (cuda)"; the engine default
    /// (<c>default:…</c>) says so ("Automatic (the engine's default)").
    /// </summary>
    public static string Requested(string requested)
    {
        var isDefault = requested.StartsWith("default:", StringComparison.Ordinal);
        var asked = isDefault ? requested["default:".Length..] : requested;
        var name = asked switch
        {
            "auto" => Loc.S("Analysis_Route_Automatic"),
            "cpu_reference" => Loc.S("Analysis_Route_CpuReference"),
            var s when s.StartsWith("gpu:", StringComparison.Ordinal) => Loc.F("Analysis_Route_Gpu", s[4..]),
            var s => s,
        };
        return isDefault ? Loc.F("Analysis_Route_EngineDefault", name) : name;
    }

    /// <summary>
    /// The backend badge's label: "CPU · f64", or a GPU with its backend's name ("GPU · f32 · cuda").
    /// </summary>
    public static string Badge(string used, string precision)
    {
        var gpu = used.StartsWith("gpu", StringComparison.Ordinal);
        var p = precision switch { "f32_gpu" => "f32", "f64_reference" => "f64", { Length: > 0 } x => x, _ => gpu ? "f32" : "f64" };
        return used.StartsWith("gpu:", StringComparison.Ordinal) && used.Length > 4 ? $"GPU · {p} · {used[4..]}" : $"{(gpu ? "GPU" : "CPU")} · {p}";
    }

    /// <summary>Both predicted times (GPU, CPU) in ms, when the decision has them (the device key is any but <c>cpu</c>).</summary>
    static (double Gpu, double Cpu)? Times(JsonObject d)
    {
        if (d["predicted_ms"] is not JsonObject p || Json.Number(p["cpu"]) is not double cpu) return null;
        var device = p.Where(kv => kv.Key != "cpu").Select(kv => Json.Number(kv.Value)).FirstOrDefault(v => v is not null);
        if (device is not double gpu || !double.IsFinite(gpu) || !double.IsFinite(cpu) || gpu < 0 || cpu < 0) return null;
        return (gpu, cpu);
    }

    /// <summary>" (GPU predicted 12 ms, CPU 31 ms)", or nothing without both times.</summary>
    static string Predicted(JsonObject d) => Times(d) is (double g, double c) ? Loc.F("Analysis_Route_Predicted", Duration(g), Duration(c)) : "";

    /// <summary>"12 ms", "1.2 s", with a time that rounds to nothing as "&lt;1 ms".</summary>
    public static string Duration(double ms) =>
        ms < 0.5 ? "<1 ms" : ms >= 1000 ? (ms / 1000).ToString("0.0", CultureInfo.InvariantCulture) + " s" : ms.ToString("0", CultureInfo.InvariantCulture) + " ms";
}
