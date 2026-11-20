using System.Globalization;
using System.Text.Json.Nodes;
using MKLight.Engine;
using MKLight.Model;

namespace MKLight.App.ViewModels.Jobs;

// Glass substitution (the Mac app's GlassSubstitution): the engine's `glass_substitution` settings, its progress and
// its ranked result (docs/OPTIMIZATION.md "Glass substitution").

public enum CandidateRule { Nearest, Window }

/// <summary>Which catalogue glasses may replace a lens's glass (the engine's <c>CandidateSet</c>), in plain terms.</summary>
public sealed record GlassCandidates
{
    public static readonly string[] AllStatuses = ["preferred", "standard", "melt", "special", "obsolete"];
    static readonly string[] DefaultStatuses = ["standard", "preferred", "melt"];

    /// <summary>Catalogues to draw from; empty: every loaded catalogue.</summary>
    public IReadOnlyList<string> Catalogs { get; init; } = [];
    public IReadOnlySet<string> Statuses { get; init; } = new HashSet<string>(DefaultStatuses);
    public CandidateRule Rule { get; init; } = CandidateRule.Nearest;
    /// <summary>The K glasses nearest the current one.</summary>
    public int Nearest { get; init; } = 10;
    public double WindowNd { get; init; } = 0.05;
    public double WindowVd { get; init; } = 5;
    /// <summary>Names or <c>catalog:name</c>.</summary>
    public IReadOnlyList<string> Include { get; init; } = [];
    public IReadOnlyList<string> Exclude { get; init; } = [];
    public double? MinTransmission { get; init; }
    public double? MaxRelativeCost { get; init; }
    public double? MaxClimateClass { get; init; }
    public double? MaxStainClass { get; init; }

    /// <summary>The engine's <c>candidates</c> object.</summary>
    public JsonObject ToJson()
    {
        static JsonArray Strings(IEnumerable<string> s) => new([.. s.Select(x => (JsonNode)x)]);
        var o = new JsonObject
        {
            ["catalogs"] = Strings(Catalogs),
            ["statuses"] = Strings(AllStatuses.Where(Statuses.Contains)),
            ["include"] = Strings(Include),
            ["exclude"] = Strings(Exclude),
        };
        if (Rule == CandidateRule.Nearest)
        {
            o["nearest"] = Nearest;
            o["window"] = null;
        }
        else
        {
            o["nearest"] = null;
            o["window"] = new JsonObject { ["nd"] = WindowNd, ["vd"] = WindowVd };
        }
        void Put(string k, double? v) { if (v is { } d) o[k] = d; }
        Put("min_transmission", MinTransmission);
        Put("max_relative_cost", MaxRelativeCost);
        Put("max_climate_class", MaxClimateClass);
        Put("max_stain_class", MaxStainClass);
        return o;
    }

    /// <summary>A short description: <c>10 nearest · all catalogs</c>.</summary>
    public string Summary
    {
        get
        {
            var parts = new List<string>
            {
                Rule == CandidateRule.Nearest ? $"{Nearest} nearest" : $"n_d ±{MeritFormat.Short(WindowNd)}, V_d ±{MeritFormat.Short(WindowVd)}",
                Catalogs.Count == 0 ? "all catalogs" : string.Join(", ", Catalogs),
            };
            if (!Statuses.SetEquals(DefaultStatuses)) parts.Add(string.Join(", ", AllStatuses.Where(Statuses.Contains)));
            if (Include.Count > 0) parts.Add($"+{Include.Count}");
            if (Exclude.Count > 0) parts.Add($"−{Exclude.Count}");
            if (MinTransmission is not null || MaxRelativeCost is not null || MaxClimateClass is not null || MaxStainClass is not null) parts.Add("limits");
            return string.Join(" · ", parts);
        }
    }

    /// <summary>Splits <c>N-BK7, SCHOTT:F2</c> into names.</summary>
    public static IReadOnlyList<string> Names(string text) =>
        text.Split([',', ';', '\n'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToList();
}

/// <summary>A glass gap of the design that substitution may change: the surface in front of the glass and the glass now there.</summary>
public sealed record GlassSlot(SurfaceID Id, int Index, string Material, bool IsModel, MaterialOffset Offset)
{
    /// <summary>Every glass gap of a design (refracting surfaces followed by glass).</summary>
    public static IReadOnlyList<GlassSlot> All(OpticalSystem system)
    {
        var list = new List<GlassSlot>();
        for (var i = 1; i < system.ImageIndex; i++)
        {
            var s = system.Surfaces[i];
            if (s.IsCoordinateBreak || s.Interaction != Interaction.Refract || s.Material.IsAir) continue;
            var offset = s.Material is MaterialRef.CatalogGlass c ? c.Offset : MaterialOffset.Zero;
            list.Add(new GlassSlot(s.Id, i, s.Material.DisplayName, s.Material is MaterialRef.ModelGlass, offset));
        }
        return list;
    }

    /// <summary>The glass with its melt offset, as the comparison shows the glass it leaves.</summary>
    public string Label => Offset.IsZero ? Material : Material + " " + LensFormat.Offset(Offset);
}

/// <summary>A glass in a candidate combination.</summary>
public sealed record SubstitutedGlass(SurfaceID Surface, string Catalog, string Name, double Nd, double Vd)
{
    public static SubstitutedGlass FromJson(JsonNode? g) => new(
        new SurfaceID(g?["surface"].UInt64OrNull() ?? 0), g?["catalog"].StringOrNull() ?? "", g?["name"].StringOrNull() ?? "?",
        g?["nd"].Nf() ?? double.NaN, g?["vd"].Nf() ?? double.NaN);
}

/// <summary>One ranked glass combination with its re-optimised design.</summary>
public sealed record GlassCombination(int Rank, double Merit, double Violation, bool Feasible, IReadOnlyList<SubstitutedGlass> Glasses,
    IReadOnlyList<OptimizeResult.Variable> Changes, OpticalSystem? System, bool CutShort)
{
    public static GlassCombination FromJson(JsonNode? r, int fallbackRank)
    {
        OpticalSystem? system = null;
        try { if (r?["system"] is { } s) system = OpticalSystem.FromJson(s); }
        catch (Exception e) when (e is global::System.Text.Json.JsonException or InvalidOperationException or FormatException or ArgumentException) { }
        var changes = (r?["variables"] as JsonArray ?? []).Select(v =>
        {
            ParamPath? p = null;
            try { if (v?["param"] is { } pj) p = ParamPath.FromJson(pj); }
            catch (Exception e) when (e is global::System.Text.Json.JsonException or InvalidOperationException or FormatException) { }
            return new OptimizeResult.Variable("", p, v?["start"].Nf() ?? double.NaN, v?["value"].Nf() ?? double.NaN, false);
        }).ToList();
        return new GlassCombination((int)(r?["rank"].IntOrNull() ?? fallbackRank), r?["merit"].Nf() ?? double.NaN, r?["violation"].Nf() ?? double.NaN,
            r?["feasible"].BoolOrNull() ?? true, (r?["glasses"] as JsonArray ?? []).Select(SubstitutedGlass.FromJson).ToList(), changes, system,
            r?["local_termination"].StringOrNull() == "cancelled");
    }

    /// <summary>Variables that moved noticeably.</summary>
    public IReadOnlyList<OptimizeResult.Variable> Moved => Changes.Where(c =>
        double.IsFinite(c.Value) && double.IsFinite(c.Start) && Math.Abs(c.Value - c.Start) > 1e-9 * Math.Max(Math.Abs(c.Start), 1)).ToList();

    public string ChangeSummary
    {
        get
        {
            var n = Moved.Count;
            var s = n == 0 ? "shape unchanged" : $"{n} variable{(n == 1 ? "" : "s")} re-optimized";
            return CutShort ? s + " · cut short" : s;
        }
    }
}

/// <summary>The <c>glass_substitution</c> result.</summary>
public sealed record GlassSubstitutionResult(string Status, bool Stopped, string Termination, double StartingMerit, int Combinations, int Failed,
    IReadOnlyList<GlassCombination> Ranked, IReadOnlyList<double> History)
{
    public static GlassSubstitutionResult? FromJson(JsonNode? json)
    {
        if (json is not JsonObject o || o["results"] is not JsonArray list) return null;
        var history = (o["history"] as JsonArray ?? []).Select(h => h?["best_merit"].Nf() ?? double.NaN).Where(double.IsFinite).ToList();
        return new GlassSubstitutionResult(o["status"].StringOrNull() ?? "", o["stopped"].BoolOrNull() ?? false, o["termination"].StringOrNull() ?? "",
            o["starting_merit"].Nf() ?? double.NaN, (int)(o["statistics"]?["combinations"].IntOrNull() ?? 0), (int)(o["statistics"]?["failed"].IntOrNull() ?? 0),
            list.Select((r, k) => GlassCombination.FromJson(r, k + 1)).ToList(), RunningMinimum(history));
    }

    public static IReadOnlyList<double> RunningMinimum(IEnumerable<double> v)
    {
        var output = new List<double>();
        foreach (var x in v) output.Add(output.Count == 0 ? x : Math.Min(x, output[^1]));
        return output;
    }

    public string StatusText(bool applied)
    {
        var s = $"{Combinations} combination{(Combinations == 1 ? "" : "s")} re-optimized";
        if (Failed > 0) s += $", {Failed} failed";
        s += ". ";
        if (Stopped) s += "Stopped early: the ranking so far. ";
        if (Termination == "time_limit") s += "The time limit ended the search. ";
        if (!applied) s += "Select a combination to preview it in the window.";
        return s.TrimEnd();
    }
}

/// <summary>Live progress of a running substitution, from the stage <c>glass_substitution: combinations N; best merit M; glasses [...]</c>.</summary>
public sealed record GlassProgressInfo(int Combinations, double BestMerit, IReadOnlyList<string> Glasses)
{
    public static GlassProgressInfo? Parse(string stage)
    {
        const string Prefix = "glass_substitution:";
        if (!stage.StartsWith(Prefix, StringComparison.Ordinal)) return null;
        var body = stage[Prefix.Length..];
        var combinations = 0;
        var best = double.NaN;
        IReadOnlyList<string> glasses = [];
        var c = body.IndexOf("combinations ", StringComparison.Ordinal);
        if (c >= 0) int.TryParse(new string(body[(c + 13)..].TakeWhile(char.IsDigit).ToArray()), NumberStyles.Integer, CultureInfo.InvariantCulture, out combinations);
        var b = body.IndexOf("best merit ", StringComparison.Ordinal);
        if (b >= 0 && double.TryParse(new string(body[(b + 11)..].TakeWhile(ch => !char.IsWhiteSpace(ch) && ch != ';').ToArray()),
            NumberStyles.Float, CultureInfo.InvariantCulture, out var m)) best = m;
        var g = body.IndexOf("glasses ", StringComparison.Ordinal);
        if (g >= 0)
        {
            try { glasses = (JsonNode.Parse(body[(g + 8)..]) as JsonArray ?? []).Select(x => x?["name"].StringOrNull() ?? "?").ToList(); }
            catch (global::System.Text.Json.JsonException) { }
        }
        return new GlassProgressInfo(combinations, best, glasses);
    }
}

public enum GlassSearch { Tabu, Hammer }

/// <summary>The settings of a substitution run (the page's working state, not saved).</summary>
public sealed class GlassSubstitutionSettings
{
    /// <summary>Surfaces whose glass may change. Empty: the engine snaps every model glass to its nearest real glasses.</summary>
    public HashSet<SurfaceID> Selected { get; } = [];
    public GlassCandidates Candidates { get; set; } = new();
    /// <summary>Per-lens candidates that replace <see cref="Candidates"/>.</summary>
    public Dictionary<SurfaceID, GlassCandidates> Overrides { get; } = [];
    public GlassSearch Algorithm { get; set; } = GlassSearch.Tabu;
    public int MaxCombinations { get; set; } = 100;
    public double? MaxSeconds { get; set; }
    public int LocalIterations { get; set; } = 100;
    public int TopN { get; set; } = 10;
    /// <summary>Re-optimize the design's variables (else every curvature and the back focus).</summary>
    public bool UseDesignVariables { get; set; } = true;

    IReadOnlyList<SurfaceID> _configuredFor = [];

    public static string Title(GlassSearch a) => a == GlassSearch.Tabu ? "Quick (tabu search)" : "Thorough (global search)";
    public static string Detail(GlassSearch a) => a == GlassSearch.Tabu
        ? "Tries glass combinations, each re-optimized from the current shape."
        : "Hops over glasses and shape together to escape local minima; slower.";

    /// <summary>Selects every glass of a design the first time (or when its glass gaps changed).</summary>
    public void Configure(OpticalSystem system)
    {
        var slots = GlassSlot.All(system).Select(s => s.Id).ToList();
        if (slots.SequenceEqual(_configuredFor)) return;
        var first = _configuredFor.Count == 0;
        _configuredFor = slots;
        var keep = first || Selected.Count == 0 ? slots.ToHashSet() : Selected.Intersect(slots).ToHashSet();
        Selected.Clear();
        Selected.UnionWith(keep);
        foreach (var id in Overrides.Keys.Where(k => !slots.Contains(k)).ToList()) Overrides.Remove(id);
    }

    /// <summary>Material parameters substitution may not vary: the offsets and model numbers of the glasses it replaces.</summary>
    public bool IsSubstitutedMaterialParam(ParamPath p) => p switch
    {
        ParamPath.MaterialDnD x => Selected.Contains(x.Surface),
        ParamPath.MaterialDvD x => Selected.Contains(x.Surface),
        ParamPath.ModelNd x => Selected.Contains(x.Surface),
        ParamPath.ModelVd x => Selected.Contains(x.Surface),
        ParamPath.ModelDpgf x => Selected.Contains(x.Surface),
        _ => false,
    };

    /// <summary>The ticked glasses that carry a melt offset, and what happens to it.</summary>
    public string? OffsetNote(OpticalSystem system)
    {
        var slots = GlassSlot.All(system).Where(s => Selected.Contains(s.Id) && !s.Offset.IsZero).ToList();
        if (slots.Count == 0) return null;
        var names = slots.Select(s => $"S{s.Index} {s.Label}").ToList();
        var list = names.Count == 1 ? names[0] : string.Join(", ", names.Take(names.Count - 1)) + " and " + names[^1];
        var one = slots.Count == 1;
        return $"{list} {(one ? "carries a melt offset" : "carry melt offsets")}. A replacement glass is installed without {(one ? "it" : "them")}, even the same glass: the offset belonged to the original melt.";
    }

    /// <summary>The <c>glass_substitution</c> settings.</summary>
    public Dictionary<string, JsonNode?> Settings(OpticalSystem system, MeritSettings merit, IReadOnlyList<DesignVariable> variables)
    {
        var s = new Dictionary<string, JsonNode?>
        {
            ["algorithm"] = Algorithm == GlassSearch.Tabu ? "tabu" : "hammer",
            ["max_combinations"] = MaxCombinations, ["local_iterations"] = LocalIterations, ["top_n"] = TopN,
        };
        if (MaxSeconds is { } t) s["max_seconds"] = t;
        foreach (var (k, v) in merit.JobSettings()) s[k] = v;
        var free = variables.Where(v => !IsSubstitutedMaterialParam(v.Param)).ToList();
        if (OptimizeVariables.Settings(free, UseDesignVariables, system) is { } vars) s["variables"] = vars;
        var order = GlassSlot.All(system).Select(x => x.Id).Where(Selected.Contains);
        s["material_variables"] = new JsonArray([.. order.Select(id => (JsonNode)new JsonObject
        {
            ["surface"] = id.Value, ["candidates"] = (Overrides.GetValueOrDefault(id) ?? Candidates).ToJson(),
        })]);
        return s;
    }
}

/// <summary>The <c>variables</c> setting of the optimising jobs.</summary>
public static class OptimizeVariables
{
    /// <summary>
    /// The design's variables without the pinned ones (equal bounds hold them) and those a solve controls (the
    /// engine applies solves first and refuses them), or null for the engine's automatic set.
    /// </summary>
    public static JsonArray? Settings(IReadOnlyList<DesignVariable> variables, bool useDesign, OpticalSystem? system)
    {
        var free = variables.Where(v => !v.IsPinned && !(system?.IsSolved(v.Param) ?? false)).ToList();
        if (!useDesign || free.Count == 0) return null;
        return new JsonArray([.. free.Select(v => (JsonNode)v.ToJson())]);
    }

    /// <summary>Why the run cannot start with these variables, if it cannot.</summary>
    public static string? Problem(IReadOnlyList<DesignVariable> variables, bool useDesign, OpticalSystem? system)
    {
        if (!useDesign || variables.Count == 0) return null;
        bool Solved(DesignVariable v) => system?.IsSolved(v.Param) ?? false;
        if (variables.All(v => v.IsPinned || Solved(v)))
            return variables.Any(Solved)
                ? "Every variable is pinned or controlled by a solve. Add a free variable, or vary the automatic set."
                : "Every variable is pinned. Unpin one, or vary the automatic set.";
        return null;
    }
}
