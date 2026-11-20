using System.Text.Json.Nodes;

namespace MKLight.Engine;

// The engine's glass catalog management (docs/FFI.md, "Glass catalog management"):
// catalogs, table rows, glass details, reload reports, availability overrides and user
// glasses. Readers are tolerant: a missing optional value is null.

/// <summary>A glass catalog of a session (<c>mk_session_list_catalogs</c>).</summary>
/// <param name="Source"><c>builtin</c> or <c>user</c>.</param>
/// <param name="Library">Loaded from the app-wide library.</param>
/// <param name="InProject">Recorded in the project (it may be in the library too).</param>
public sealed record CatalogInfo(string Name, string Source, string? Path, int GlassCount, bool Library = false, bool InProject = false)
{
    public bool IsBuiltin => Source == "builtin";

    public static CatalogInfo FromJson(JsonNode? json)
    {
        var name = json.At("name").StringOrNull() ?? throw Bad("name");
        var source = json.At("source").StringOrNull() ?? throw Bad("source");
        return new CatalogInfo(
            name, source, json.At("path").StringOrNull(), (int)(json.At("glass_count").IntOrNull() ?? throw Bad("glass_count")),
            json.At("library").BoolOrNull() ?? false,
            // Engines before the library recorded every user catalog in the project.
            json.At("in_project").BoolOrNull() ?? source == "user");
    }

    static EngineException Bad(string key) =>
        EngineException.Engine(EngineErrorCode.Parse, $"unexpected engine JSON for a catalog: no '{key}'");
}

/// <summary>One glass of a catalog (<c>mk_session_list_glasses</c>), with the engine's non-finite numbers (NaN, ±∞).</summary>
public sealed record GlassInfo(string Catalog, string Name, double Nd, double Vd, double MinUm, double MaxUm)
{
    /// <summary>Reads a row; a number that is neither a number, null, "inf", "-inf" nor "nan" throws <see cref="FormatException"/>.</summary>
    public static GlassInfo FromJson(JsonNode? json) => new(
        json.At("catalog").StringOrNull() ?? throw new FormatException("glass without catalog"),
        json.At("name").StringOrNull() ?? throw new FormatException("glass without name"),
        JsonRead.EngineDouble(json.At("nd")), JsonRead.EngineDouble(json.At("vd")),
        JsonRead.EngineDouble(json.At("min_um")), JsonRead.EngineDouble(json.At("max_um")));

    public JsonObject ToJson() => new()
    {
        ["catalog"] = Catalog,
        ["name"] = Name,
        ["nd"] = JsonRead.EngineDoubleNode(Nd),
        ["vd"] = JsonRead.EngineDoubleNode(Vd),
        ["min_um"] = JsonRead.EngineDoubleNode(MinUm),
        ["max_um"] = JsonRead.EngineDoubleNode(MaxUm),
    };
}

/// <summary>A catalog glass's availability (AGF status code).</summary>
public readonly record struct GlassStatus(int Code)
{
    public static readonly GlassStatus Standard = new(0);
    public static readonly GlassStatus Preferred = new(1);
    public static readonly GlassStatus Obsolete = new(2);
    public static readonly GlassStatus Special = new(3);
    public static readonly GlassStatus Melt = new(4);

    /// <summary>The statuses a user chooses from.</summary>
    public static readonly IReadOnlyList<GlassStatus> Choices = [Preferred, Standard, Special, Melt, Obsolete];

    static readonly string[] Names = ["standard", "preferred", "obsolete", "special", "melt"];

    public string Title => Code is >= 0 and < 5 ? char.ToUpperInvariant(Names[Code][0]) + Names[Code][1..] : $"Status {Code}";

    /// <summary>The engine's JSON form (<c>"preferred"</c>, or <c>{"other": 7}</c>).</summary>
    public JsonNode ToJson() => Code is >= 0 and < 5 ? JsonValue.Create(Names[Code]) : new JsonObject { ["other"] = Code };

    public static GlassStatus? FromJson(JsonNode? json)
    {
        if (json.StringOrNull() is { } s)
        {
            var i = Array.IndexOf(Names, s);
            return i >= 0 ? new GlassStatus(i) : null;
        }
        return json.At("other").IntOrNull() is { } c ? new GlassStatus((int)c) : null;
    }

    public override string ToString() => Title;
}

/// <summary>The user's availability setting for one catalog glass. <c>null</c> keeps the catalog's value.</summary>
public sealed record GlassOverride(string Catalog, string Name, GlassStatus? Status = null, bool? ExcludeSubstitution = null)
{
    public bool IsEmpty => Status is null && ExcludeSubstitution is null;

    /// <summary>Whether this override is for <paramref name="catalog"/>'s glass <paramref name="name"/> (case-insensitive, as the engine matches).</summary>
    public bool Matches(string catalog, string name) =>
        string.Equals(Catalog, catalog, StringComparison.OrdinalIgnoreCase) && string.Equals(Name, name, StringComparison.OrdinalIgnoreCase);

    public JsonObject ToJson()
    {
        var o = new JsonObject { ["catalog"] = Catalog, ["name"] = Name };
        if (Status is { } s) o["status"] = s.ToJson();
        if (ExcludeSubstitution is { } e) o["exclude_substitution"] = e;
        return o;
    }

    public static GlassOverride FromJson(JsonNode? json) => new(
        json.At("catalog").StringOrNull() ?? "", json.At("name").StringOrNull() ?? "",
        GlassStatus.FromJson(json.At("status")), json.At("exclude_substitution").BoolOrNull());
}

static class Finite
{
    public static double? Of(JsonNode? v) => v.DoubleValue() is { } d && double.IsFinite(d) ? d : null;
}

/// <summary>One row of a glass table (<c>mk_session_list_glasses</c>).</summary>
public sealed record GlassRow
{
    public required string Catalog { get; init; }
    public required string Name { get; init; }
    public double Nd { get; init; }
    public double Vd { get; init; }
    public double? Pgf { get; init; }
    public double? Dpgf { get; init; }
    public double MinUm { get; init; }
    public double MaxUm { get; init; }
    /// <summary>In effect, after the user's overrides.</summary>
    public GlassStatus Status { get; init; }
    public bool ExcludeSubstitution { get; init; }
    /// <summary>The user's overrides changed status or exclusion.</summary>
    public bool Overridden { get; init; }
    public double? Density { get; init; }
    public double? RelativeCost { get; init; }
    /// <summary>10⁻⁶/K, −30…+70 °C.</summary>
    public double? Tce { get; init; }
    public double? ClimateResistance { get; init; }
    public double? StainResistance { get; init; }
    public double? TransmissionMinUm { get; init; }
    public double? TransmissionMaxUm { get; init; }
    public double? Lambda80Um { get; init; }

    public string Id => Catalog + ":" + Name;

    /// <summary>Whether the valid range is unrestricted (no <c>LD</c> record).</summary>
    public bool UnrestrictedRange => MaxUm > 1e100;

    /// <summary><c>null</c> without catalog and name.</summary>
    public static GlassRow? FromJson(JsonNode? v)
    {
        if (v.At("catalog").StringOrNull() is not { } catalog || v.At("name").StringOrNull() is not { } name) return null;
        return new GlassRow
        {
            Catalog = catalog,
            Name = name,
            Nd = v.At("nd").DoubleValue() ?? double.NaN,
            Vd = v.At("vd").DoubleValue() ?? double.NaN,
            Pgf = Finite.Of(v.At("pgf")),
            Dpgf = Finite.Of(v.At("dpgf")),
            MinUm = v.At("min_um").DoubleValue() ?? 0,
            MaxUm = v.At("max_um").DoubleValue() ?? double.PositiveInfinity,
            Status = GlassStatus.FromJson(v.At("status")) ?? GlassStatus.Standard,
            ExcludeSubstitution = v.At("exclude_substitution").BoolOrNull() ?? false,
            Overridden = v.At("overridden").BoolOrNull() ?? false,
            Density = Finite.Of(v.At("density")),
            RelativeCost = Finite.Of(v.At("relative_cost")),
            Tce = Finite.Of(v.At("tce")),
            ClimateResistance = Finite.Of(v.At("climate_resistance")),
            StainResistance = Finite.Of(v.At("stain_resistance")),
            TransmissionMinUm = Finite.Of(v.At("transmission_min_um")),
            TransmissionMaxUm = Finite.Of(v.At("transmission_max_um")),
            Lambda80Um = Finite.Of(v.At("lambda80_um")),
        };
    }
}

/// <summary>Everything about one glass (<c>mk_session_glass_details</c>).</summary>
public sealed class GlassDetails
{
    public sealed record LineIndex(string Line, double Um, double N);
    public sealed record Model(double Nd, double Vd, double Dpgf);
    public sealed record Transmittance(double Um, double Tau, double ThicknessMm);

    public required GlassRow Row { get; init; }
    public GlassStatus CatalogStatus { get; init; }
    public bool CatalogExcludeSubstitution { get; init; }
    public int? Formula { get; init; }
    public string FormulaName { get; init; } = "";
    public IReadOnlyList<double> Coefficients { get; init; } = [];
    public double? ComputedNd { get; init; }
    public double? ComputedVd { get; init; }
    public Model? AbbeModel { get; init; }
    public double? CatalogDpgf { get; init; }
    public double? Tce100To300 { get; init; }
    public string? MilCode { get; init; }
    public int? MeltFrequency { get; init; }
    public double? AcidResistance { get; init; }
    public double? AlkaliResistance { get; init; }
    public double? PhosphateResistance { get; init; }
    public bool HasThermalData { get; init; }
    public IReadOnlyList<Transmittance> TransmittanceData { get; init; } = [];
    public string Comment { get; init; } = "";
    public IReadOnlyList<LineIndex> Lines { get; init; } = [];
    /// <summary>(λ µm, n) across the glass's range.</summary>
    public IReadOnlyList<(double Um, double N)> IndexCurve { get; init; } = [];
    /// <summary>The glass in the engine's catalog JSON form (before overrides).</summary>
    public required JsonObject Glass { get; init; }

    /// <summary><c>null</c> without a row or the glass object.</summary>
    public static GlassDetails? FromJson(JsonNode? v)
    {
        if (GlassRow.FromJson(v) is not { } row || v.At("glass") is not JsonObject glass) return null;
        var model = v.At("model");
        return new GlassDetails
        {
            Row = row,
            CatalogStatus = GlassStatus.FromJson(v.At("catalog_status")) ?? row.Status,
            CatalogExcludeSubstitution = v.At("catalog_exclude_substitution").BoolOrNull() ?? row.ExcludeSubstitution,
            Formula = (int?)v.At("formula").IntOrNull(),
            FormulaName = v.At("formula_name").StringOrNull() ?? "",
            Coefficients = (v.At("coefficients").ArrayOrNull() ?? []).Select(c => c.DoubleValue()).OfType<double>().ToArray(),
            ComputedNd = Finite.Of(v.At("computed_nd")),
            ComputedVd = Finite.Of(v.At("computed_vd")),
            AbbeModel = Finite.Of(model.At("nd")) is { } nd && Finite.Of(model.At("vd")) is { } vd
                ? new Model(nd, vd, Finite.Of(model.At("dpgf")) ?? 0) : null,
            CatalogDpgf = Finite.Of(v.At("catalog_dpgf")),
            Tce100To300 = Finite.Of(v.At("tce_100_300")),
            MilCode = v.At("mil_code").StringOrNull(),
            MeltFrequency = (int?)v.At("melt_frequency").IntOrNull(),
            AcidResistance = Finite.Of(v.At("acid_resistance")),
            AlkaliResistance = Finite.Of(v.At("alkali_resistance")),
            PhosphateResistance = Finite.Of(v.At("phosphate_resistance")),
            HasThermalData = v.At("thermal") is JsonObject,
            TransmittanceData = (v.At("transmittance").ArrayOrNull() ?? []).Select(t =>
                Finite.Of(t.At("um")) is { } um && Finite.Of(t.At("tau")) is { } tau && Finite.Of(t.At("thickness_mm")) is { } th
                    ? new Transmittance(um, tau, th) : null).OfType<Transmittance>().ToArray(),
            Comment = v.At("comment").StringOrNull() ?? "",
            Lines = (v.At("lines").ArrayOrNull() ?? []).Select(l =>
                l.At("line").StringOrNull() is { } name && Finite.Of(l.At("um")) is { } um && Finite.Of(l.At("n")) is { } n
                    ? new LineIndex(name, um, n) : null).OfType<LineIndex>().ToArray(),
            IndexCurve = (v.At("index_curve").ArrayOrNull() ?? [])
                .Where(p => p is JsonArray { Count: 2 } a && a[0].DoubleValue() is not null && a[1].DoubleValue() is not null)
                .Select(p => (p![0].DoubleValue()!.Value, p[1].DoubleValue()!.Value)).ToArray(),
            Glass = (JsonObject)glass.DeepClone(),
        };
    }
}

/// <summary>What reloading a catalog changed (<c>mk_session_reload_catalog</c>).</summary>
public sealed record CatalogChanges(
    string Catalog, IReadOnlyList<string> Added, IReadOnlyList<string> Removed, IReadOnlyList<CatalogChanges.Change> Changed, int Unchanged)
{
    public sealed record Change(string Name, double NdBefore, double NdAfter, double VdBefore, double VdAfter, IReadOnlyList<string> Fields);

    /// <summary><c>null</c> without a catalog name.</summary>
    public static CatalogChanges? FromJson(JsonNode? v)
    {
        if (v.At("catalog").StringOrNull() is not { } catalog) return null;
        static string[] Strings(JsonNode? a) => (a.ArrayOrNull() ?? []).Select(s => s.StringOrNull()).OfType<string>().ToArray();
        return new CatalogChanges(
            catalog, Strings(v.At("added")), Strings(v.At("removed")),
            (v.At("changed").ArrayOrNull() ?? []).Select(c => c.At("name").StringOrNull() is { } name
                ? new Change(name, c.At("nd_before").DoubleValue() ?? double.NaN, c.At("nd_after").DoubleValue() ?? double.NaN,
                    c.At("vd_before").DoubleValue() ?? double.NaN, c.At("vd_after").DoubleValue() ?? double.NaN, Strings(c.At("fields")))
                : null).OfType<Change>().ToArray(),
            (int)(v.At("unchanged").IntOrNull() ?? 0));
    }

    public bool IsEmpty => Added.Count == 0 && Removed.Count == 0 && Changed.Count == 0;

    /// <summary>One line, e.g. "2 added, 1 removed, 3 changed".</summary>
    public string Summary
    {
        get
        {
            if (IsEmpty) return "No changes";
            var parts = new List<string>();
            if (Added.Count > 0) parts.Add($"{Added.Count} added");
            if (Removed.Count > 0) parts.Add($"{Removed.Count} removed");
            if (Changed.Count > 0) parts.Add($"{Changed.Count} changed");
            return string.Join(", ", parts);
        }
    }
}

/// <summary>The outcome of <see cref="IEngineClient.SetLibraryCatalogsAsync"/>.</summary>
/// <param name="Changed">Whether the session's glass library changed.</param>
public sealed record LibrarySync(IReadOnlyList<CatalogInfo> Catalogs, IReadOnlyList<LibrarySync.Failure> Errors, bool Changed)
{
    public sealed record Failure(string Path, string Message);

    public static readonly LibrarySync Empty = new([], [], false);
}

/// <summary>A catalog written as AGF text (<c>mk_catalog_to_agf</c>).</summary>
/// <param name="Warnings">Glasses the writer could not express or had to rename.</param>
public sealed record AgfExport(string Text, IReadOnlyList<string> Warnings);
