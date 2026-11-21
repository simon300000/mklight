using System.Text.Json.Nodes;
using MKLight.App.Support;
using MKLight.App.Views.Analysis.Results.Readers;
using MKLight.Engine;
using MKLight.Model;

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
            list.Add((Loc.S("Shell_Status_Squeeze"), Squeeze(p.Squeeze, 2)));
        }
        else if (Entry(fo, "efl") is { } e)
        {
            list.Add((anamorphic ? "EFL y" : "EFL", e.Formatted));
        }
        if (Entry(fo, "working_f_number", "image_f_number") is { } f) list.Add(("F/#", f.Formatted));
        if (Entry(fo, "total_track") is { } t) list.Add(("TT", t.Formatted));
        return list;
    }

    /// <summary>The rows (their label's resource key) and the engine's <c>first_order</c> keys they show (the first one present).</summary>
    static readonly (string Label, string[] Keys)[] Keys =
    [
        ("Shell_Status_Efl", ["efl"]), ("Shell_Status_Bfl", ["bfl"]), ("Shell_Status_WorkingFNumber", ["working_f_number", "image_f_number"]),
        ("Shell_Status_Epd", ["entrance_pupil_diameter"]), ("Shell_Status_TotalTrack", ["total_track"]),
        ("Shell_Status_ImageHeight", ["paraxial_image_height"]),
    ];

    /// <summary>The rows that are the y section's for an anamorphic design.</summary>
    static readonly HashSet<string> YSection = ["Shell_Status_Efl", "Shell_Status_Bfl", "Shell_Status_WorkingFNumber", "Shell_Status_ImageHeight"];

    /// <summary>The details' First Order section.</summary>
    public static IReadOnlyList<StatusDetail> FirstOrder(FirstOrderData? fo, bool anamorphic)
    {
        var rows = new List<StatusDetail> { StatusDetail.Header(Loc.S("Shell_Status_FirstOrder")) };
        if (fo is null)
        {
            rows.Add(StatusDetail.Note(Loc.S("Shell_Status_NoFirstOrder")));
            return rows;
        }
        foreach (var (label, keys) in Keys)
            if (Entry(fo, keys) is { } e)
                rows.Add(StatusDetail.Row(anamorphic && YSection.Contains(label) ? Loc.F("Shell_Status_YSection", Loc.S(label)) : Loc.S(label), e.Formatted));
        if (!anamorphic) return rows;
        if (fo.Primary is not { } p)
        {
            rows.Add(StatusDetail.Note(Loc.S("Shell_Status_NotRotational")));
            return rows;
        }
        rows.Add(new(StatusDetail.Kinds.XyHeader, "", "x (mm)", "y (mm)"));
        rows.Add(new(StatusDetail.Kinds.Xy, "EFL", Fmt.Short(p.EflX), Fmt.Short(p.EflY), Loc.S("Shell_Status_Efl")));
        rows.Add(new(StatusDetail.Kinds.Xy, "BFL", Fmt.Short(p.BflX), Fmt.Short(p.BflY), Loc.S("Shell_Status_Bfl")));
        rows.Add(StatusDetail.Row(Loc.S("Shell_Status_Squeeze"), Squeeze(p.Squeeze, 4),
            Loc.F("Shell_Status_SqueezeHelp", fo.SqueezeDefinition is { } def ? EngineText.T(def) : Loc.S("Shell_Status_SqueezeDefault"))));
        if (fo.Symmetry is not null) rows.Add(StatusDetail.Row(Loc.S("Shell_Status_Symmetry"), fo.SymmetryTitle));
        foreach (var d in p.Doe)
            rows.Add(StatusDetail.Row(Loc.F("Shell_Status_DoePower", d.Surface), Fmt.Value(d.Power, "1/mm"), Loc.S("Shell_Status_DoePowerHelp")));
        if (p.Doe.Count > 0 && double.IsFinite(p.WavelengthUm))
            rows.Add(StatusDetail.Note(Loc.F("Shell_Status_AtWavelength", p.WavelengthUm)));
        foreach (var w in fo.Warnings) rows.Add(new(StatusDetail.Kinds.Warning, EngineText.T(w)));
        return rows;
    }

    /// <summary>The details' Engine section: version, snapshot, glass library and GPU kernels (with the cache note).</summary>
    public static IReadOnlyList<StatusDetail> Engine(string engine, ModelSnapshot? snapshot, GpuPreparation? gpu)
    {
        var rows = new List<StatusDetail> { StatusDetail.Header(Loc.S("Shell_Status_EngineHeader")), StatusDetail.Row(Loc.S("Shell_Status_Engine"), engine) };
        if (snapshot is not null)
        {
            rows.Add(StatusDetail.Row(Loc.S("Shell_Status_Snapshot"), snapshot.Hash.Short, snapshot.Hash.Hex));
            if (snapshot.LibraryHash is { Length: > 0 } lib)
                rows.Add(StatusDetail.Row(Loc.S("Shell_Status_GlassLibrary"), lib.Length > 12 ? lib[..12] : lib, Loc.F("Shell_Status_GlassLibraryHelp", lib)));
        }
        if (gpu is not null && ViewModels.GpuPreparationMonitor.Summary(gpu) is { } summary)
        {
            rows.Add(StatusDetail.Row(Loc.S("Shell_Status_GpuKernels"), summary));
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
