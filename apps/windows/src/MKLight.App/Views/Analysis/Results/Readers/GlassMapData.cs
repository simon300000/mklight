using System.Text.Json.Nodes;
using MKLight.App.Support;

namespace MKLight.App.Views.Analysis.Results.Readers;

/// <summary>A glass's retained catalogue status (AGF), or unknown.</summary>
public enum GlassStatus { Preferred, Standard, Special, Melt, Obsolete, Unknown }

public static class GlassStatuses
{
    public static (GlassStatus Status, int? Other) Read(JsonNode? n)
    {
        if (Json.Str(n) is { } s)
            return (s.ToLowerInvariant() switch
            {
                "preferred" => GlassStatus.Preferred,
                "standard" => GlassStatus.Standard,
                "obsolete" => GlassStatus.Obsolete,
                "special" => GlassStatus.Special,
                "melt" => GlassStatus.Melt,
                _ => GlassStatus.Unknown,
            }, null);
        return (GlassStatus.Unknown, Json.Int(Json.At(n, "other")));
    }

    public static string Label(this GlassStatus s) => s switch
    {
        GlassStatus.Preferred => Loc.S("Results_GlassMap_StatusPreferred"),
        GlassStatus.Standard => Loc.S("Results_GlassMap_StatusStandard"),
        GlassStatus.Special => Loc.S("Results_GlassMap_StatusSpecial"),
        GlassStatus.Melt => Loc.S("Results_GlassMap_StatusMelt"),
        GlassStatus.Obsolete => Loc.S("Results_GlassMap_StatusObsolete"),
        _ => Loc.S("Results_GlassMap_StatusUnknown"),
    };

    /// <summary>The categorical palette slot (green, blue, violet, yellow, red); -1 neutral grey.</summary>
    public static int Slot(this GlassStatus s) => s switch
    {
        GlassStatus.Preferred => 5,
        GlassStatus.Standard => 0,
        GlassStatus.Special => 6,
        GlassStatus.Melt => 3,
        GlassStatus.Obsolete => 7,
        _ => -1,
    };

    /// <summary>Drawn hollow, so status does not rely on colour alone.</summary>
    public static bool Hollow(this GlassStatus s) => s == GlassStatus.Obsolete;
}

/// <summary>One glass of a glass map.</summary>
public sealed record GlassPoint(string Id, string? Catalog, string Name, string Kind, double Nd, double Vd, double Pgf,
    GlassStatus Status, int? OtherStatus, IReadOnlyList<int> Surfaces, bool? ExcludeSubstitution, double Dnd = 0, double Dvd = 0)
{
    /// <summary>Whether the design uses the glass (offset and model glasses always are).</summary>
    public bool InDesign => Kind != "catalog" || Surfaces.Count > 0;

    /// <summary>Whether the glass has a place on the map: finite n_d and V_d (a guard; the engine omits the others).</summary>
    public bool HasNd => double.IsFinite(Nd) && double.IsFinite(Vd) && Nd > 1 && Vd > 0;

    /// <summary>"S1, S3".</summary>
    public string SurfaceLabel => string.Join(", ", Surfaces.Select(s => "S" + s.ToString(System.Globalization.CultureInfo.InvariantCulture)));

    public string DisplayName => Kind switch
    {
        "model" => Loc.F("Results_GlassMap_ModelName", Name),
        "offset" => $"{Name} {OffsetText}",
        _ => Name,
    };

    public string OffsetText => $"Δn {Fmt.Signed(Dnd, 4).Trim()} ΔV {Fmt.Signed(Dvd, 2).Trim()}";

    public string StatusLabel => OtherStatus is int n ? Loc.F("Results_GlassMap_StatusOther", n) : Status.Label();

    /// <summary>The id of the catalogue point an offset glass is shifted from.</summary>
    public string NominalId => $"{Catalog}/{Name}";

    /// <summary>The deviation of P_g,F from the normal line through K7 and F2.</summary>
    public double DeltaPgf => Pgf - NormalLine(Vd);

    public static double NormalLine(double vd) => 0.6438 - 0.001682 * vd;
}

/// <summary>
/// A catalogue glass the engine left off the map (<c>omitted[]</c>): its range does not cover the d, F and C
/// lines (an infrared glass), so it has no n_d or V_d. <see cref="Surfaces"/>: the design surfaces using it.
/// </summary>
public sealed record OmittedGlass(string? Catalog, string Name, string Reason, IReadOnlyList<int> Surfaces)
{
    public bool InDesign => Surfaces.Count > 0;

    public string SurfaceLabel => string.Join(", ", Surfaces.Select(s => "S" + s.ToString(System.Globalization.CultureInfo.InvariantCulture)));
}

/// <summary>The <c>glass_map</c> result.</summary>
public sealed class GlassMapData
{
    public required IReadOnlyList<GlassPoint> Glasses { get; init; }

    /// <summary>Glasses without map coordinates, with the engine's reason: the design's first, then by catalogue order.</summary>
    public IReadOnlyList<OmittedGlass> Omitted { get; init; } = [];

    public static GlassMapData? Read(ResultData d)
    {
        if (Json.Arr(Json.At(d.Json, "glasses")) is not { } list) return null;
        var coords = Json.BufferRef(d, Json.At(d.Json, "coordinates")) ?? d.Buffer("coordinates");
        var fromBuffer = coords is { Shape: [var n, 3] } && n == list.Count;
        var glasses = new List<GlassPoint>();
        for (var k = 0; k < list.Count; k++)
        {
            var g = list[k];
            var kind = Json.Str(Json.At(g, "kind")) is "offset" or "model" ? Json.Str(Json.At(g, "kind"))! : "catalog";
            var name = Json.Str(Json.At(g, "name")) ?? "?";
            var catalog = Json.Str(Json.At(g, "catalog"));
            var surfaces = Json.IntArray(Json.At(g, "surfaces")) ?? [];
            double Coord(int c, string key) => fromBuffer ? coords![k * 3 + c] : Json.Nf(Json.At(g, key));
            var (status, other) = GlassStatuses.Read(Json.At(g, "status"));
            var first = surfaces.Length > 0 ? surfaces[0] : -1;
            var id = kind switch
            {
                "model" => $"model:{name}:{first}",
                "offset" => $"offset:{catalog}/{name}:{first}",
                _ => $"{catalog}/{name}",
            };
            var point = new GlassPoint(id, catalog, name, kind, Coord(0, "nd"), Coord(1, "vd"), Coord(2, "pgf"), status, other, surfaces,
                Json.Bool(Json.At(g, "exclude_substitution")),
                kind == "offset" ? Json.Number(Json.At(g, "offset", "dn_d")) ?? 0 : 0,
                kind == "offset" ? Json.Number(Json.At(g, "offset", "dv_d")) ?? 0 : 0);
            if (point.HasNd) glasses.Add(point);
        }
        var omitted = (Json.Arr(Json.At(d.Json, "omitted")) ?? [])
            .Select(o => new OmittedGlass(Json.Str(Json.At(o, "catalog")), Json.Str(Json.At(o, "name")) ?? "?",
                Json.Str(Json.At(o, "reason")) ?? "No n_d or V_d", Json.IntArray(Json.At(o, "surfaces")) ?? []))
            .OrderByDescending(o => o.InDesign).ToList();
        return new GlassMapData { Glasses = glasses, Omitted = omitted };
    }

    public IEnumerable<string> Catalogs => Glasses.Select(g => g.Catalog).Concat(Omitted.Select(o => o.Catalog)).OfType<string>().Distinct();

    /// <summary>The statuses present among the catalogue glasses, in legend order.</summary>
    public IReadOnlyList<GlassStatus> Statuses =>
        Enum.GetValues<GlassStatus>().Where(s => Glasses.Any(g => g.Kind == "catalog" && g.Status == s)).ToList();
}
