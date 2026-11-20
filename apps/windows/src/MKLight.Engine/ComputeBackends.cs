using System.Text.Json.Nodes;

namespace MKLight.Engine;

/// <summary>
/// One compute backend the engine lists (<c>mk_engine_list_backends</c>, docs/FFI.md
/// "Backends"): <c>{"backend", "id", "title", "available", "reason", "precision",
/// "capabilities": {"fp64", "reference", "analyses", "device_name", "unified_memory"}}</c>.
/// Unknown keys are ignored, missing ones take defaults.
/// </summary>
/// <param name="Id"><c>cpu_reference</c>, <c>gpu:metal</c>, <c>gpu:cuda</c>, …</param>
/// <param name="Precision"><c>f64_reference</c> or <c>f32_gpu</c>.</param>
/// <param name="Analyses">The analyses with a path on this backend (empty for the CPU reference, which runs all).</param>
public sealed record EngineBackend(
    string Id, string Title, bool Available, string? Reason, string Precision,
    string? DeviceName = null, IReadOnlyList<string>? Analyses = null)
{
    /// <summary>Whether this is a GPU backend (<c>gpu:&lt;name&gt;</c>).</summary>
    public bool IsGpu => Id.StartsWith("gpu:", StringComparison.Ordinal);

    /// <summary>The GPU backend's name (<c>cuda</c> of <c>gpu:cuda</c>); <c>null</c> for others.</summary>
    public string? GpuName => IsGpu ? Id[4..] : null;

    public static EngineBackend? FromJson(JsonNode? json)
    {
        var backend = json.At("backend");
        var id = json.At("id").StringOrNull()
            ?? (backend.At("kind").StringOrNull() is { } kind
                ? kind == "gpu" && backend.At("name").StringOrNull() is { } n ? "gpu:" + n : kind
                : null);
        if (id is null) return null;
        var caps = json.At("capabilities");
        return new EngineBackend(
            id, json.At("title").StringOrNull() ?? id, json.At("available").BoolOrNull() ?? true,
            json.At("reason").StringOrNull(), json.At("precision").StringOrNull() ?? "",
            caps.At("device_name").StringOrNull(),
            (caps.At("analyses").ArrayOrNull() ?? []).Select(a => a.StringOrNull()).OfType<string>().ToArray());
    }

    /// <summary>The engine's list; entries without an id are skipped.</summary>
    public static IReadOnlyList<EngineBackend> ListFromJson(string text) =>
        (Json.Parse(text, "backend list").ArrayOrNull() ?? []).Select(FromJson).OfType<EngineBackend>().ToArray();

    /// <summary>"f32" / "f64", or the engine's name of another precision.</summary>
    public static string PrecisionShort(string precision) => precision switch
    {
        "f64_reference" => "f64",
        "f32_gpu" => "f32",
        var p => p,
    };

    /// <summary>A GPU backend's name for people: <c>cuda</c> → CUDA, <c>metal</c> → Metal, …</summary>
    public static string GpuDisplayName(string name) => name switch
    {
        "cuda" => "CUDA",
        "metal" => "Metal",
        "webgpu" => "WebGPU",
        "vulkan" => "Vulkan",
        "cpu_f32" => "CPU f32 emulation",
        "" => "GPU",
        var s => char.ToUpperInvariant(s[0]) + s[1..],
    };

    /// <summary>
    /// The backend's name with its device, without the precision: the engine's title less a
    /// trailing precision note ("CPU reference (f64)" → "CPU reference"), with the device in
    /// parentheses when the title does not name it ("CUDA (RTX 4090)").
    /// </summary>
    public string DisplayTitle
    {
        get
        {
            // The emulation's engine title ("Emulation on the CPU (f32), not a real GPU") does not fit a picker.
            if (GpuName == "cpu_f32") return "GPU emulation on the CPU";
            var title = Title.Trim();
            foreach (var note in new[] { "(f64)", "(f32)", "(" + PrecisionShort(Precision) + ")" })
                if (note.Length > 2 && title.EndsWith(note, StringComparison.OrdinalIgnoreCase)) title = title[..^note.Length].TrimEnd();
            if (title.Length == 0 || title == Id) title = GpuName is { } g ? GpuDisplayName(g) : Id == "cpu_reference" ? "CPU reference" : Id;
            return DeviceName is { Length: > 0 } d && !title.Contains(d, StringComparison.OrdinalIgnoreCase) ? $"{title} ({d})" : title;
        }
    }
}

/// <summary>One choice of the compute backend picker: its title, what it requests, and whether it can be chosen.</summary>
/// <param name="Title">"Automatic", "CPU reference · f64", "CUDA (RTX 4090) · f32".</param>
/// <param name="Reason">Why it cannot be chosen (the engine's <c>reason</c>).</param>
/// <param name="Help">What it does, for a tooltip.</param>
public sealed record ComputeBackendOption(string Title, ExecutionSettings Execution, bool Enabled, string? Reason, string Help);

/// <summary>The compute backend picker's choices, built from the engine's backend list.</summary>
public static class ComputeBackendOptions
{
    /// <summary>
    /// Automatic, the CPU reference, then every GPU backend by its title, the unavailable ones
    /// disabled with their reason. A GPU choice names its backend. When <paramref name="current"/>
    /// asks for a GPU the engine does not list, a disabled choice for it is added so the picker
    /// still shows what is asked for.
    /// </summary>
    public static IReadOnlyList<ComputeBackendOption> From(IReadOnlyList<EngineBackend> backends, ExecutionSettings current = default)
    {
        var cpu = backends.FirstOrDefault(b => b.Id == "cpu_reference");
        var gpus = backends.Where(b => b.IsGpu).ToList();
        var list = new List<ComputeBackendOption>
        {
            new("Automatic", new(BackendPreference.Auto), true, null,
                gpus.Any(g => g.Available)
                    ? "The engine chooses per analysis: the GPU where it has a path and it pays off, else the CPU reference."
                    : "The engine chooses per analysis; without a GPU every analysis runs on the CPU reference."),
            new($"CPU reference · {EngineBackend.PrecisionShort(cpu?.Precision ?? "f64_reference")}", new(BackendPreference.CpuReference),
                cpu?.Available ?? true, cpu?.Available == false ? cpu.Reason : null,
                "Every analysis on the CPU in double precision, the reference."),
        };
        foreach (var g in gpus)
        {
            var precision = EngineBackend.PrecisionShort(g.Precision);
            list.Add(new(precision.Length == 0 ? g.DisplayTitle : $"{g.DisplayTitle} · {precision}",
                new(BackendPreference.Gpu, g.GpuName), g.Available, g.Available ? null : g.Reason ?? "Not available",
                "Analyses with a GPU path run on this GPU in single precision within the error budget; the others, and a failed GPU run, on the CPU reference."));
        }
        if (current.Backend == BackendPreference.Gpu && IndexOf(list, current) < 0)
        {
            var name = current.GpuName is { } n ? EngineBackend.GpuDisplayName(n) : "GPU";
            list.Add(new(name + " · f32", current, false, gpus.Count == 0 ? "This engine lists no GPU backend" : "This engine does not list it",
                "Asked for by the design; analyses run on the CPU reference."));
        }
        return list;
    }

    /// <summary>
    /// The choice for <paramref name="execution"/>: the same request, or for a GPU preference
    /// that names no GPU the first GPU choice. -1 when none matches.
    /// </summary>
    public static int IndexOf(IReadOnlyList<ComputeBackendOption> options, ExecutionSettings execution)
    {
        for (var i = 0; i < options.Count; i++)
            if (options[i].Execution == execution) return i;
        if (execution is { Backend: BackendPreference.Gpu, GpuName: null })
            for (var i = 0; i < options.Count; i++)
                if (options[i].Execution.Backend == BackendPreference.Gpu) return i;
        return -1;
    }
}
