using System.Text.Json.Nodes;

namespace MKLight.Engine;

/// <summary>Lens prescription formats the engine imports (docs/IMPORT.md).</summary>
public enum PrescriptionFormat
{
    /// <summary><c>.zmx</c> keyword records.</summary>
    Zmx,
    /// <summary><c>.seq</c> command scripts.</summary>
    Seq,
}

public static class PrescriptionFormats
{
    /// <summary>The format of a file extension (case-insensitive, with or without the dot).</summary>
    public static PrescriptionFormat? FromExtension(string extension) => extension.TrimStart('.').ToLowerInvariant() switch
    {
        "zmx" => PrescriptionFormat.Zmx,
        "seq" => PrescriptionFormat.Seq,
        _ => null,
    };

    /// <summary>The engine's name and file extension: <c>zmx</c>, <c>seq</c>.</summary>
    public static string WireName(this PrescriptionFormat f) => f == PrescriptionFormat.Zmx ? "zmx" : "seq";
}

/// <summary>Kinds of <see cref="ImportWarning"/>.</summary>
public enum ImportWarningKind
{
    UnsupportedRecord,
    UnsupportedSurface,
    UnknownGlass,
    SolveDropped,
    NonSequential,
    Approximated,
    Malformed,
    Defaulted,
    Validation,
    /// <summary>A kind this app does not know yet (a newer engine).</summary>
    Other,
}

/// <summary>Something the importer dropped or changed: the engine's <c>ImportWarning</c>.</summary>
/// <param name="Line">1-based line in the file.</param>
/// <param name="Surface">Index of the surface in the imported system.</param>
public sealed record ImportWarning(ImportWarningKind Kind, int? Line, int? Surface, string Message)
{
    public static ImportWarning FromJson(JsonNode? json) => new(
        json.At("kind").StringOrNull() switch
        {
            "unsupported_record" => ImportWarningKind.UnsupportedRecord,
            "unsupported_surface" => ImportWarningKind.UnsupportedSurface,
            "unknown_glass" => ImportWarningKind.UnknownGlass,
            "solve_dropped" => ImportWarningKind.SolveDropped,
            "non_sequential" => ImportWarningKind.NonSequential,
            "approximated" => ImportWarningKind.Approximated,
            "malformed" => ImportWarningKind.Malformed,
            "defaulted" => ImportWarningKind.Defaulted,
            "validation" => ImportWarningKind.Validation,
            _ => ImportWarningKind.Other,
        },
        (int?)json.At("line").IntOrNull(), (int?)json.At("surface").IntOrNull(), json.At("message").StringOrNull() ?? "");
}

/// <summary>
/// An imported prescription: the system (as the engine's OpticalSystem JSON, for the model
/// to decode) and everything that did not map.
/// </summary>
/// <param name="Encoding">Text encoding the file was read in (<c>utf8</c>, <c>utf16_le</c>, …).</param>
/// <param name="SnapshotHash">The session's new snapshot hash, for an import into a session.</param>
public sealed record PrescriptionImport(
    PrescriptionFormat Format, string Encoding, string SystemJson, IReadOnlyList<ImportWarning> Warnings,
    SnapshotHash? SnapshotHash = null)
{
    public static PrescriptionImport FromJson(string text)
    {
        var json = Json.Parse(text, "prescription import");
        var system = json.At("system") as JsonObject
            ?? throw EngineException.Engine(EngineErrorCode.Parse, "unexpected engine JSON for an import: no system");
        return new PrescriptionImport(
            PrescriptionFormats.FromExtension(json.At("format").StringOrNull() ?? "") ?? PrescriptionFormat.Zmx,
            json.At("encoding").StringOrNull() ?? "",
            system.ToJsonString(),
            (json.At("warnings").ArrayOrNull() ?? []).Select(ImportWarning.FromJson).ToArray(),
            json.At("snapshot_hash").StringOrNull() is { } h ? new SnapshotHash(h) : null);
    }
}
