using System.Globalization;
using System.Text.Json.Nodes;

namespace MKLight.Engine;

/// <summary>Floating-point model used for a computation (engine <c>Precision</c>).</summary>
public enum Precision
{
    /// <summary>CPU double precision, the reference (<c>f64_reference</c>).</summary>
    F64Reference,
    /// <summary>GPU (or emulated) single precision within the documented error budget (<c>f32_gpu</c>).</summary>
    F32Gpu,
}

/// <summary>Accuracy tier of a run (engine <c>Tier</c>, CONVENTIONS §7).</summary>
public enum Tier
{
    /// <summary>Design-grade budgets and sampling.</summary>
    Design,
    /// <summary>Cheaper sampling and aiming within the preview budgets; not for final decisions.</summary>
    Preview,
}

/// <summary>How the real-aim solves of a run stopped (engine <c>AimStopKind</c>).</summary>
public enum AimStopKind
{
    /// <summary>At the system's requested aiming tolerance.</summary>
    Requested,
    /// <summary>At the residual derived per field and wavelength from the propagated error.</summary>
    Conditioned,
}

/// <summary>One backend attempt that did not produce the published result (engine <c>BackendAttempt</c>).</summary>
/// <param name="Backend">Backend id (<c>gpu:metal</c>).</param>
/// <param name="Outcome"><c>failed</c>, <c>skipped</c> or <c>reference_completion</c>.</param>
/// <param name="ElapsedMs">Wall time spent in the attempt; 0 when it never started.</param>
public sealed record BackendAttempt(string Backend, string Outcome, string Reason, double ElapsedMs);

/// <summary>
/// Metadata attached to every result (mirror of the engine's <c>RunInfo</c>): what was
/// computed, from which snapshot, with which settings, backend and seed. It makes results
/// reproducible and tells the UI when a result is stale.
/// </summary>
public sealed class RunInfo
{
    public required string EngineVersion { get; init; }
    public required SnapshotHash SnapshotHash { get; init; }
    public required string Analysis { get; init; }
    /// <summary>The complete settings the engine ran with.</summary>
    public required JsonNode? Settings { get; init; }
    public required string Backend { get; init; }
    public required Precision Precision { get; init; }
    public ulong? Seed { get; init; }
    public ulong StartedUnixMs { get; init; }
    public double DurationMs { get; init; }
    public Tier Tier { get; init; } = Tier.Design;
    public AimStopKind AimStop { get; init; } = AimStopKind.Requested;
    public IReadOnlyList<string> Warnings { get; init; } = [];
    /// <summary>Backend attempts that did not compute this result, in order; their time is part of <see cref="DurationMs"/>.</summary>
    public IReadOnlyList<BackendAttempt> FailedAttempts { get; init; } = [];

    /// <summary>
    /// Reads the engine's <c>run_info</c>. A precision, tier or aim stop this app does not
    /// know (a newer engine) reads as the non-default one, with a warning, rather than
    /// failing the result. Missing required keys throw an <see cref="EngineException"/> (<c>Parse</c>).
    /// </summary>
    public static RunInfo FromJson(JsonNode? json)
    {
        string Required(string key) => json.At(key).StringOrNull()
            ?? throw EngineException.Engine(EngineErrorCode.Parse, $"unexpected engine JSON for run_info: no '{key}'");
        var warnings = (json.At("warnings").ArrayOrNull() ?? []).Select(w => w.StringOrNull()).OfType<string>().ToList();
        var precisionName = Required("precision");
        var precision = precisionName switch { "f64_reference" => Precision.F64Reference, "f32_gpu" => Precision.F32Gpu, _ => (Precision?)null };
        var tierName = json.At("tier").StringOrNull() ?? "design";
        var tier = tierName switch { "design" => Tier.Design, "preview" => Tier.Preview, _ => (Tier?)null };
        var aimName = json.At("aim_stop").StringOrNull() ?? "requested";
        var aim = aimName switch { "requested" => AimStopKind.Requested, "conditioned" => AimStopKind.Conditioned, _ => (AimStopKind?)null };
        if (precision is null) warnings.Add($"unknown precision '{precisionName}'");
        if (tier is null) warnings.Add($"unknown tier '{tierName}'");
        if (aim is null) warnings.Add($"unknown aim stop '{aimName}'");
        return new RunInfo
        {
            EngineVersion = Required("engine_version"),
            SnapshotHash = new SnapshotHash(Required("snapshot_hash")),
            Analysis = Required("analysis"),
            Settings = Json.Clone(json.At("settings")),
            Backend = Required("backend"),
            Precision = precision ?? Precision.F32Gpu,
            Seed = json.At("seed").UInt64OrNull(),
            StartedUnixMs = json.At("started_unix_ms").UInt64OrNull() ?? 0,
            DurationMs = json.At("duration_ms").DoubleValue() ?? 0,
            Tier = tier ?? Tier.Preview,
            AimStop = aim ?? AimStopKind.Conditioned,
            Warnings = warnings,
            FailedAttempts = (json.At("failed_attempts").ArrayOrNull() ?? []).Select(a => new BackendAttempt(
                a.At("backend").StringOrNull() ?? "", a.At("outcome").StringOrNull() ?? "", a.At("reason").StringOrNull() ?? "",
                a.At("elapsed_ms").DoubleValue() ?? 0)).ToArray(),
        };
    }

    /// <summary>The engine's JSON form (<c>failed_attempts</c> only when there are some).</summary>
    public JsonObject ToJson()
    {
        var o = new JsonObject
        {
            ["engine_version"] = EngineVersion,
            ["snapshot_hash"] = SnapshotHash.Hex,
            ["analysis"] = Analysis,
            ["settings"] = Json.Clone(Settings),
            ["backend"] = Backend,
            ["precision"] = Precision == Precision.F64Reference ? "f64_reference" : "f32_gpu",
            ["seed"] = Seed is { } s ? JsonValue.Create(s) : null,
            ["started_unix_ms"] = StartedUnixMs,
            ["duration_ms"] = DurationMs,
            ["tier"] = Tier == Tier.Design ? "design" : "preview",
            ["aim_stop"] = AimStop == AimStopKind.Requested ? "requested" : "conditioned",
            ["warnings"] = new JsonArray(Warnings.Select(w => (JsonNode?)w).ToArray()),
        };
        if (FailedAttempts.Count > 0)
        {
            o["failed_attempts"] = new JsonArray(FailedAttempts.Select(a => (JsonNode?)new JsonObject
            {
                ["backend"] = a.Backend, ["outcome"] = a.Outcome, ["reason"] = a.Reason, ["elapsed_ms"] = a.ElapsedMs,
            }).ToArray());
        }
        return o;
    }
}

/// <summary>
/// How the engine computed a result (the envelope's <c>backend</c> object,
/// docs/RESULTS.md §5): what was requested, what was used, why another backend was used,
/// if one was, and how the backend was selected.
/// </summary>
public sealed record BackendReport(
    string Requested, string Used, string Precision, string? Device = null, string? FallbackReason = null,
    JsonNode? CostDecision = null)
{
    /// <summary>Reads the envelope's <c>backend</c> object; <c>null</c> when it is missing. <c>cost_decision</c> is kept as sent.</summary>
    public static BackendReport? FromJson(JsonNode? json) => json is JsonObject o
        ? new BackendReport(
            o["requested"].StringOrNull() ?? "", o["used"].StringOrNull() ?? "", o["precision"].StringOrNull() ?? "",
            o["device"].StringOrNull(), o["fallback_reason"].StringOrNull(), Json.Clone(o["cost_decision"]))
        : null;

    /// <summary>The GPU backend that computed the result (<c>cuda</c> of <c>gpu:cuda</c>); <c>null</c> for the CPU.</summary>
    public string? UsedGpuName => Used.StartsWith("gpu:", StringComparison.Ordinal) ? Used[4..] : null;

    /// <summary>e.g. "CPU reference · f64" or "CUDA (RTX 4090) · f32".</summary>
    public string Summary
    {
        get
        {
            var name = Used switch
            {
                "cpu_reference" => "CPU reference",
                "mock" => "Mock",
                _ when UsedGpuName is { } g => EngineBackend.GpuDisplayName(g),
                var s => s,
            };
            var device = Device is { Length: > 0 } d ? $" ({d})" : "";
            var precision = EngineBackend.PrecisionShort(Precision);
            return precision.Length == 0 ? name + device : $"{name}{device} · {precision}";
        }
    }
}
