using System.Text.Json.Nodes;

namespace MKLight.Engine;

/// <summary>
/// The engine's preparation of the GPU's compute kernels at launch
/// (<c>mk_engine_gpu_preparation_json</c>, docs/FFI.md "GPU kernel preparation"):
/// compiling every kernel takes seconds on a new install or update; prepared ahead, no
/// analysis waits for it. One of <see cref="Idle"/>, <see cref="Preparing"/>,
/// <see cref="Ready"/>, <see cref="Unavailable"/>, <see cref="Failed"/>.
/// </summary>
public abstract record GpuPreparation
{
    GpuPreparation() { }

    /// <summary>Not started (or an engine without a GPU path).</summary>
    public sealed record Idle : GpuPreparation;

    /// <summary>Creating kernels: <paramref name="Done"/> of <paramref name="Total"/>.</summary>
    public sealed record Preparing(string Device, int Done, int Total) : GpuPreparation;

    /// <summary>
    /// Every kernel is ready; <paramref name="FromCache"/> were loaded instead of compiled.
    /// <paramref name="Warning"/>: the saved kernels could not be read or written, so the next
    /// launch compiles again.
    /// </summary>
    public sealed record Ready(string Device, int Kernels, int FromCache, double Seconds, string? Warning = null) : GpuPreparation;

    /// <summary>No GPU: analyses use the CPU reference.</summary>
    public sealed record Unavailable(string Reason) : GpuPreparation;

    /// <summary>Preparation failed; kernels compile when first used.</summary>
    public sealed record Failed(string Device, string Message) : GpuPreparation;

    /// <summary>Still running.</summary>
    public bool IsPreparing => this is Preparing;

    /// <summary>Fraction done while preparing (0 before the total is known).</summary>
    public double Fraction => this is Preparing { Total: > 0 } p ? (double)p.Done / p.Total : 0;

    /// <summary>Reads the engine's state JSON; an unknown state is <see cref="Idle"/>, a missing key a parse error.</summary>
    public static GpuPreparation FromJson(string text)
    {
        var json = Json.Parse(text, "GPU preparation state");
        string device = json.At("device").StringOrNull() ?? "";
        string Str(string key) => json.At(key).StringOrNull() ?? throw Missing(key);
        int Int(string key) => (int)(json.At(key).IntOrNull() ?? throw Missing(key));
        return json.At("state").StringOrNull() switch
        {
            "preparing" => new Preparing(device, Int("done"), Int("total")),
            "ready" => new Ready(device, Int("kernels"), Int("from_cache"),
                json.At("seconds").DoubleValue() ?? throw Missing("seconds"), json.At("warning").StringOrNull()),
            "unavailable" => new Unavailable(Str("reason")),
            "failed" => new Failed(device, Str("message")),
            null => throw Missing("state"),
            _ => new Idle(),
        };
    }

    static EngineException Missing(string key) =>
        EngineException.Engine(EngineErrorCode.Parse, $"unexpected engine JSON for the GPU preparation: no '{key}'");
}
