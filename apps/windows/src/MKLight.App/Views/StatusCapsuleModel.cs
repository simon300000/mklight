using System.Globalization;
using System.Text.Json.Nodes;
using MKLight.App.Views.Analysis.Results.Readers;
using MKLight.Engine;

namespace MKLight.App.Views;

/// <summary>One line of the status details flyout.</summary>
/// <param name="Kind">How it is drawn.</param>
/// <param name="Label">The section title, the row's label, or the note's text.</param>
/// <param name="Value">The row's value (the x value of an x / y row).</param>
/// <param name="Value2">The y value of an x / y row.</param>
/// <param name="Help">A tooltip.</param>
public sealed record StatusDetail(StatusDetail.Kinds Kind, string Label, string? Value = null, string? Value2 = null, string? Help = null)
{
    public enum Kinds
    {
        /// <summary>A section title.</summary>
        Header,
        /// <summary>Label and value.</summary>
        Row,
        /// <summary>The column titles of the x / y table.</summary>
        XyHeader,
        /// <summary>Label, x and y values.</summary>
        Xy,
        /// <summary>A secondary line of text.</summary>
        Note,
        /// <summary>A warning line.</summary>
        Warning,
    }

    public static StatusDetail Header(string title) => new(Kinds.Header, title);
    public static StatusDetail Row(string label, string value, string? help = null) => new(Kinds.Row, label, value, Help: help);
    public static StatusDetail Note(string text) => new(Kinds.Note, text);
}

/// <summary>
/// What the status capsule shows of a design (the Mac app's DesignStatus): the key numbers,
/// the details' first order (with the x / y table, symmetry, DOE powers and notes of a design
/// that is not rotationally symmetric) and the engine section, and the compute backend kept
/// in the project's <c>views</c>. No UI types.
/// </summary>
public static class StatusCapsuleModel
{
    /// <summary>
    /// EFL, F/# and TT; for an anamorphic design the y section's EFL ("EFL y") and its squeeze,
    /// or "EFL y" from the keyed report when the engine sent no x / y data.
    /// </summary>
    public static IReadOnlyList<(string Label, string Value)> KeyNumbers(FirstOrderData? fo, bool anamorphic)
    {
        var list = new List<(string, string)>();
        if (fo is null) return list;
        if (anamorphic && fo.Primary is { } p)
        {
            list.Add(("EFL y", Fmt.Value(p.EflY, "mm")));
            list.Add(("Squeeze", Squeeze(p.Squeeze, 2)));
        }
        else if (Entry(fo, "efl") is { } e)
        {
            list.Add((anamorphic ? "EFL y" : "EFL", e.Formatted));
        }
        if (Entry(fo, "working_f_number", "image_f_number") is { } f) list.Add(("F/#", f.Formatted));
        if (Entry(fo, "total_track") is { } t) list.Add(("TT", t.Formatted));
        return list;
    }

    /// <summary>The rows and the engine's <c>first_order</c> keys they show (the first one present).</summary>
    static readonly (string Label, string[] Keys)[] Keys =
    [
        ("Effective focal length", ["efl"]), ("Back focal length", ["bfl"]), ("Working F/#", ["working_f_number", "image_f_number"]),
        ("Entrance pupil diameter", ["entrance_pupil_diameter"]), ("Total track", ["total_track"]),
        ("Image height", ["paraxial_image_height"]),
    ];

    /// <summary>The rows that are the y section's for an anamorphic design.</summary>
    static readonly HashSet<string> YSection = ["Effective focal length", "Back focal length", "Working F/#", "Image height"];

    /// <summary>The details' First Order section.</summary>
    public static IReadOnlyList<StatusDetail> FirstOrder(FirstOrderData? fo, bool anamorphic)
    {
        var rows = new List<StatusDetail> { StatusDetail.Header("First Order") };
        if (fo is null)
        {
            rows.Add(StatusDetail.Note("No first-order data yet."));
            return rows;
        }
        foreach (var (label, keys) in Keys)
            if (Entry(fo, keys) is { } e)
                rows.Add(StatusDetail.Row(anamorphic && YSection.Contains(label) ? label + " (y)" : label, e.Formatted));
        if (!anamorphic) return rows;
        if (fo.Primary is not { } p)
        {
            rows.Add(StatusDetail.Note("This design is not rotationally symmetric. The engine's first-order data covers the y section."));
            return rows;
        }
        rows.Add(new(StatusDetail.Kinds.XyHeader, "", "x (mm)", "y (mm)"));
        rows.Add(new(StatusDetail.Kinds.Xy, "EFL", Fmt.Short(p.EflX), Fmt.Short(p.EflY), "Effective focal length"));
        rows.Add(new(StatusDetail.Kinds.Xy, "BFL", Fmt.Short(p.BflX), Fmt.Short(p.BflY), "Back focal length"));
        rows.Add(StatusDetail.Row("Squeeze", Squeeze(p.Squeeze, 4), "Squeeze: " + (fo.SqueezeDefinition ?? "signed EFL y / EFL x")));
        if (fo.Symmetry is not null) rows.Add(StatusDetail.Row("Symmetry", fo.SymmetryTitle));
        foreach (var d in p.Doe)
            rows.Add(StatusDetail.Row("DOE power, surface " + d.Surface.ToString(CultureInfo.InvariantCulture), Fmt.Value(d.Power, "1/mm"),
                "Phase power of the trace order (base refractive power excluded)"));
        if (p.Doe.Count > 0 && double.IsFinite(p.WavelengthUm))
            rows.Add(StatusDetail.Note($"At λ = {p.WavelengthUm.ToString("0.0000", CultureInfo.InvariantCulture)} µm (primary)"));
        foreach (var w in fo.Warnings) rows.Add(new(StatusDetail.Kinds.Warning, w));
        return rows;
    }

    /// <summary>The details' Engine section: version, snapshot, glass library and GPU kernels (with the cache note).</summary>
    public static IReadOnlyList<StatusDetail> Engine(string engine, ModelSnapshot? snapshot, GpuPreparation? gpu)
    {
        var rows = new List<StatusDetail> { StatusDetail.Header("Engine"), StatusDetail.Row("Engine", engine) };
        if (snapshot is not null)
        {
            rows.Add(StatusDetail.Row("Snapshot", snapshot.Hash.Short, snapshot.Hash.Hex));
            if (snapshot.LibraryHash is { Length: > 0 } lib)
                rows.Add(StatusDetail.Row("Glass library", lib.Length > 12 ? lib[..12] : lib, "Hash of the glass catalogs the design's results are computed with: " + lib));
        }
        if (gpu is not null && ViewModels.GpuPreparationMonitor.Summary(gpu) is { } summary)
        {
            rows.Add(StatusDetail.Row("GPU kernels", summary));
            if (ViewModels.GpuPreparationMonitor.Note(gpu) is { } note) rows.Add(StatusDetail.Note(note));
        }
        return rows;
    }

    /// <summary>"1.50×", signed; not finite as the one number format has it.</summary>
    public static string Squeeze(double v, int digits) => double.IsFinite(v) ? Fmt.Fixed(v, digits) + "×" : Fmt.Short(v);

    static FirstOrderData.Entry? Entry(FirstOrderData fo, params string[] keys) =>
        keys.Select(k => fo.Entries.FirstOrDefault(e => e.Key == k)).FirstOrDefault(e => e is not null);

    // The compute backend in the project's views

    /// <summary>The key of the layout entry's <c>ui</c> that keeps the design's compute backend.</summary>
    public const string BackendKey = "compute_backend";

    /// <summary>The compute backend a project's <c>views</c> keep (automatic when none).</summary>
    public static ExecutionSettings ReadBackend(JsonNode? views) =>
        ExecutionSettings.FromWireName(LayoutEntry(views)?["ui"]?[BackendKey] is JsonValue v && v.TryGetValue(out string? s) ? s : null);

    /// <summary>
    /// <paramref name="views"/> with the compute backend kept in the layout entry's <c>ui</c>
    /// (other keys untouched; automatic removes the key). Null when the views have no layout
    /// entry yet (the panes are saved first, the backend with the next change).
    /// </summary>
    public static JsonArray? WithBackend(JsonNode? views, ExecutionSettings execution)
    {
        if (views?.DeepClone() is not JsonArray copy || LayoutEntry(copy) is not { } layout) return null;
        if (layout["ui"] is not JsonObject ui) layout["ui"] = ui = [];
        if (execution.Backend == BackendPreference.Auto) ui.Remove(BackendKey);
        else ui[BackendKey] = execution.WireName;
        return copy;
    }

    static JsonObject? LayoutEntry(JsonNode? views) =>
        (views as JsonArray ?? []).OfType<JsonObject>().FirstOrDefault(o => o["key"] is JsonValue k && k.TryGetValue(out string? s) && s == "layout");
}
