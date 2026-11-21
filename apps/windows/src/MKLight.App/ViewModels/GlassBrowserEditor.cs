using System.Text.Json.Nodes;
using MKLight.App.Support;
using MKLight.Engine;

namespace MKLight.App.ViewModels;

/// <summary>
/// The editor of one user glass (the Mac app's GlassEditorModel): a model glass (n_d, V_d,
/// ΔP_g,F) or an AGF dispersion formula with its coefficients, plus range, physical data and
/// status. The engine makes the glass (<see cref="IEngineClient.GlassFromSpec"/>) and computes
/// its n_d and V_d. Free of WinUI types.
/// </summary>
public sealed class GlassEditorModel
{
    public enum Kind { Model, Formula }

    GlassEditorModel(GlassLibrary library, string catalog, string name, string? originalName, JsonObject? baseGlass)
    {
        Library = library;
        Catalog = catalog;
        Name = name;
        OriginalName = originalName;
        Base = baseGlass;
    }

    public GlassLibrary Library { get; }
    public string Catalog { get; set; }
    public string Name { get; set; }
    /// <summary>The glass's name before editing (null for a new glass).</summary>
    public string? OriginalName { get; }
    public Kind Dispersion { get; set; }
    public double Nd { get; set; } = 1.5168;
    public double Vd { get; set; } = 64.17;
    public double Dpgf { get; set; }

    int _formula = 2;
    public int Formula
    {
        get => _formula;
        set { _formula = value; Coefficients = Resized(Coefficients, CoefficientNames(value).Count); }
    }

    public double[] Coefficients { get; set; } = new double[6];
    public double? MinUm { get; set; }
    public double? MaxUm { get; set; }
    public double? Density { get; set; }
    public double? Tce { get; set; }
    public double? RelativeCost { get; set; }
    public GlassStatus Status { get; set; } = GlassStatus.Standard;
    public bool ExcludeSubstitution { get; set; }
    public string Comment { get; set; } = "";
    /// <summary>The glass edited or duplicated: its other data (thermal, transmittance, AGF records) are kept.</summary>
    public JsonObject? Base { get; }

    /// <summary>The engine's answer for the current values: the glass, or why not.</summary>
    public JsonObject? Preview { get; private set; }
    public string? Problem { get; private set; }

    public string Title => OriginalName is null ? Loc.S("Glass_Editor_NewTitle") : Loc.F("Glass_Editor_EditTitle", OriginalName);

    public static GlassEditorModel New(string catalog, GlassLibrary library) =>
        new(library, catalog, library.FreeGlassName("NEWGLASS", catalog), null, null);

    /// <summary>A copy of a catalog glass in a user catalog (the catalog's data, not the user's status settings).</summary>
    public static GlassEditorModel Duplicate(GlassDetails d, string catalog, GlassLibrary library)
    {
        var e = new GlassEditorModel(library, catalog, library.FreeGlassName(d.Row.Name, catalog), null, (JsonObject)d.Glass.DeepClone());
        e.Load(d);
        return e;
    }

    /// <summary>An existing user glass.</summary>
    public static GlassEditorModel Edit(GlassDetails d, GlassLibrary library)
    {
        var e = new GlassEditorModel(library, d.Row.Catalog, d.Row.Name, d.Row.Name, (JsonObject)d.Glass.DeepClone());
        e.Load(d);
        return e;
    }

    void Load(GlassDetails d)
    {
        if (d.AbbeModel is { } m)
        {
            Dispersion = Kind.Model;
            (Nd, Vd, Dpgf) = (m.Nd, m.Vd, m.Dpgf);
        }
        else if (d.Formula is int f)
        {
            Dispersion = Kind.Formula;
            _formula = f;
            Coefficients = Resized(d.Coefficients, CoefficientNames(f).Count);
            (Nd, Vd, Dpgf) = (d.Row.Nd ?? 1.5, d.Row.Vd ?? 50, d.Row.Dpgf ?? 0);
        }
        else
        {
            // Tabulated or other data without a formula: a model glass at its n_d, V_d.
            Dispersion = Kind.Model;
            // The catalog's listed values unless they are an IR glass's placeholders (n_d 1, V_d 0).
            (Nd, Vd, Dpgf) = (d.Row.Nd ?? (d.Row.CatalogNd is > 1 and var cn ? cn : 1.5),
                d.Row.Vd ?? (d.Row.CatalogVd is > 0 and var cv ? cv : 50), d.Row.Dpgf ?? 0);
        }
        MinUm = d.Row.MinUm;
        MaxUm = d.Row.UnrestrictedRange ? null : d.Row.MaxUm;
        Density = d.Row.Density;
        Tce = d.Row.Tce;
        RelativeCost = d.Row.RelativeCost;
        Status = d.CatalogStatus;
        ExcludeSubstitution = d.CatalogExcludeSubstitution;
        Comment = d.Comment;
    }

    /// <summary>The names of formula <paramref name="f"/>'s <c>CD</c> coefficients, in order.</summary>
    public static IReadOnlyList<string> CoefficientNames(int f)
    {
        static string[] Sellmeier(int n) => Enumerable.Range(1, n).SelectMany(i => new[] { $"K{i}", $"L{i}" }).ToArray();
        static string[] Series(string p, int from, int to) => Enumerable.Range(from, to - from + 1).Select(i => p + i).ToArray();
        return f switch
        {
            1 => Series("A", 0, 5),
            2 => Sellmeier(3),
            3 => ["A", "B", "C", "D", "E", "F"],
            4 => ["A", "B1", "λ1", "B2", "λ2"],
            5 => ["n0", "A", "B"],
            6 => Sellmeier(4),
            7 or 8 => ["A", "B", "C", "D"],
            9 => ["A", "B", "C", "D", "E"],
            10 or 12 => Series("a", 0, 7),
            11 => Sellmeier(5),
            13 => Series("a", 0, 8),
            _ => [],
        };
    }

    public static readonly IReadOnlyList<(int Formula, string Name)> FormulaNames =
    [
        (1, "Schott"), (2, "Sellmeier 1"), (3, "Herzberger"), (4, "Sellmeier 2"), (5, "Conrady"), (6, "Sellmeier 3"),
        (7, "Handbook of Optics 1"), (8, "Handbook of Optics 2"), (9, "Sellmeier 4"), (10, "Extended"), (11, "Sellmeier 5"),
        (12, "Extended 2"), (13, "Extended 3"),
    ];

    public static double[] Resized(IReadOnlyList<double> a, int n) => Enumerable.Range(0, n).Select(i => i < a.Count ? a[i] : 0).ToArray();

    /// <summary>The engine's <c>GlassSpec</c> for the current values.</summary>
    public JsonObject Spec()
    {
        var o = new JsonObject
        {
            ["name"] = Name.Trim(),
            ["status"] = Status.ToJson(),
            ["exclude_substitution"] = ExcludeSubstitution,
            ["comment"] = Comment,
        };
        if (Dispersion == Kind.Model) o["model"] = new JsonObject { ["nd"] = Nd, ["vd"] = Vd, ["dpgf"] = Dpgf };
        else
        {
            o["formula"] = Formula;
            o["coefficients"] = new JsonArray(Coefficients.Select(c => (JsonNode)c).ToArray());
        }
        if (MinUm is { } min) o["min_um"] = min;
        if (MaxUm is { } max) o["max_um"] = max;
        if (Density is { } density) o["density"] = density;
        if (Tce is { } tce) o["tce"] = tce;
        if (RelativeCost is { } cost) o["relative_cost"] = cost;
        if (Base is { } b) o["base"] = b.DeepClone();
        return o;
    }

    /// <summary>Asks the engine for the glass; sets <see cref="Preview"/> or <see cref="Problem"/>.</summary>
    public JsonObject? Check(IEngineClient engine)
    {
        if (GlassLibrary.NameProblem(Name, glass: true) is { } why)
        {
            (Preview, Problem) = (null, why);
            return null;
        }
        if (Dispersion == Kind.Model && !(double.IsFinite(Nd) && double.IsFinite(Vd) && double.IsFinite(Dpgf))
            || Dispersion == Kind.Formula && !Coefficients.All(double.IsFinite))
        {
            (Preview, Problem) = (null, Loc.S("Glass_Editor_EnterNumbers"));
            return null;
        }
        try
        {
            var g = engine.GlassFromSpec(Spec()) as JsonObject ?? throw EngineException.Engine(EngineErrorCode.Parse, Loc.S("Glass_Editor_NoGlassMade"));
            (Preview, Problem) = (g, null);
            return g;
        }
        catch (EngineException e)
        {
            (Preview, Problem) = (null, e.Detail);
            return null;
        }
    }

    /// <summary>
    /// n_d and V_d of the previewed glass. Each is null when its lines are outside the glass's
    /// range (an IR glass: the engine then gives the vendors' placeholders n_d 1, V_d 0).
    /// </summary>
    public (double? Nd, double? Vd)? PreviewNumbers
    {
        get
        {
            if (Preview is not { } p || p["nd"].DoubleValue() is not { } nd || p["vd"].DoubleValue() is not { } vd) return null;
            var min = p["min_um"].DoubleValue() ?? 0;
            var max = p["max_um"].DoubleValue() ?? double.PositiveInfinity;
            bool Covers(double lo, double hi) => min <= lo && max >= hi;
            return (Covers(LineD, LineD) && nd > 1 ? nd : null, Covers(LineF, LineC) && vd > 0 ? vd : null);
        }
    }

    /// <summary>The editor's answer line: "nd 1.51680    Vd 64.17", or why there is none.</summary>
    public string PreviewText => PreviewNumbers switch
    {
        null => "",
        ({ } nd, { } vd) => $"nd {GlassFormat.Fixed(nd, 5)}    Vd {GlassFormat.Fixed(vd, 2)}",
        ({ } nd, null) => Loc.F("Glass_Editor_PreviewNoVd", GlassFormat.Fixed(nd, 5)),
        _ => Loc.S("Glass_Editor_PreviewNone"),
    };

    /// <summary>The d, F and C lines (µm).</summary>
    const double LineD = 0.5875618, LineF = 0.4861327, LineC = 0.6562725;

    /// <summary>Makes the glass and stores it in the user catalog. Returns whether it saved.</summary>
    public bool Save(IEngineClient engine)
    {
        if (Check(engine) is not { } g) return false;
        try
        {
            Library.PutGlass(g, Catalog, OriginalName);
            return true;
        }
        catch (Exception e) when (e is GlassLibrary.LibraryException or IOException or UnauthorizedAccessException)
        {
            Problem = e.Message;
            return false;
        }
    }
}
