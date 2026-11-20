using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using MKLight.Model;

namespace MKLight.App.ViewModels;

/// <summary>
/// The design's variables as the inspector's <c>V</c> toggles see them: the project's <c>variables</c> section
/// (<c>[{param, lower, upper}]</c>, the engine's <c>VariableSpec</c>). Pure: the page writes the result with
/// <c>Document.EditSection("variables", …)</c> (the Mac app's <c>setVariable</c>).
/// </summary>
public static class InspectorVariables
{
    /// <summary>The smallest glass centre thickness a new glass-thickness variable gets.</summary>
    public const double GlassMinimum = 1.0;

    static IEnumerable<(ParamPath? Param, JsonNode? Item)> Entries(JsonTree section)
    {
        if (section.Node is not JsonArray list) yield break;
        foreach (var item in list)
        {
            ParamPath? p = null;
            try { if (item is JsonObject o) p = ParamPath.FromJson(o["param"]); }
            catch (Exception e) when (e is JsonException or InvalidOperationException or FormatException) { }
            yield return (p, item);
        }
    }

    public static bool IsVariable(JsonTree section, OpticalSystem system, ParamPath path)
    {
        var p = system.Canonical(path);
        return Entries(section).Any(e => e.Param is { } q && system.Canonical(q) == p);
    }

    /// <summary>Why a parameter cannot become a variable: a solve sets it (the optimizer applies solves first).</summary>
    public static string? Refusal(OpticalSystem system, ParamPath path) =>
        system.SolveFor(path) is { } s ? $"Set by a solve ({s.Kind.Type.Title()}), so the optimizer cannot vary it. Remove the solve to make it a variable." : null;

    /// <summary>The section with <paramref name="path"/> made a variable (with default bounds) or fixed; null when nothing changes.</summary>
    public static JsonTree? Set(JsonTree section, OpticalSystem system, ParamPath path, bool on)
    {
        var p = system.Canonical(path);
        var entries = Entries(section).ToList();
        var present = entries.Any(e => e.Param is { } q && system.Canonical(q) == p);
        if (on == present) return null;
        var list = new JsonArray();
        if (on)
        {
            if (Refusal(system, path) is not null) return null;
            foreach (var e in entries) list.Add(e.Item?.DeepClone());
            var (lower, upper) = DefaultBounds(path, system);
            var o = new JsonObject { ["param"] = path.ToJson() };
            if (lower is { } l) o["lower"] = l;
            if (upper is { } u) o["upper"] = u;
            list.Add(o);
        }
        else
        {
            foreach (var e in entries.Where(e => !(e.Param is { } q && system.Canonical(q) == p))) list.Add(e.Item?.DeepClone());
        }
        return new JsonTree(list);
    }

    /// <summary>
    /// Sensible bounds for a new variable: air gaps stay non-negative, glass keeps a minimum centre thickness,
    /// model glasses stay in the range of real glasses, semi-diameters and normalisation radii stay positive.
    /// </summary>
    public static (double? Lower, double? Upper) DefaultBounds(ParamPath param, OpticalSystem system)
    {
        switch (param)
        {
            case ParamPath.Thickness t:
                if (system.SurfaceIndexOf(t.Surface) is not { } i || i <= 0 || i >= system.ImageIndex) return (null, null);
                var s = system.Surfaces[i];
                if (s.IsCoordinateBreak) return (null, null);
                if (i == system.ImageIndex - 1) return (0, null);
                return s.Material.IsAir ? (0, null) : (GlassMinimum, null);
            case ParamPath.SemiDiameter: return (0, null);
            case ParamPath.ModelNd: return (1.4, 2.1);
            case ParamPath.ModelVd: return (15, 100);
            case ParamPath.MaterialDnD: return (-MaterialOffset.MaxDnD, MaterialOffset.MaxDnD);
            case ParamPath.MaterialDvD: return (-MaterialOffset.MaxDvD, MaterialOffset.MaxDvD);
            case ParamPath.ShapeParam { Parameter: ShapeParameter.NormRadius }: return (1e-3, null);
            default: return (null, null);
        }
    }
}

/// <summary>The clipping choices of the aperture section.</summary>
public enum ClipShape { None, Circular, Annular, Rectangular, Elliptical }

/// <summary>The inspector's surface and system edits as pure model functions (the Mac app's inspector actions).</summary>
public static class InspectorEdits
{
    /// <summary>The page title: Object, Image or Surface n.</summary>
    public static string SurfaceTitle(OpticalSystem s, int index) =>
        index == 0 ? "Object" : index == s.ImageIndex ? "Image" : "Surface " + index.ToString(CultureInfo.InvariantCulture);

    /// <summary>Edits the surface with <paramref name="id"/>; null when it is gone or the edit refuses.</summary>
    public static OpticalSystem? Surface(OpticalSystem s, SurfaceID id, Func<Surface, Surface?> edit)
    {
        if (s.SurfaceIndexOf(id) is not { } i || edit(s.Surfaces[i]) is not { } changed) return null;
        return s.WithSurface(i, _ => changed);
    }

    /// <summary>
    /// The surface with its placement changed; an identity placement that is the default (decenter, then tilt;
    /// no return) is removed, except on a coordinate break, whose placement is its content.
    /// </summary>
    public static Surface WithPlacement(Surface s, Func<Placement, Placement> change)
    {
        var p = change(s.Placement ?? new Placement());
        var plain = p.IsIdentity && !s.IsCoordinateBreak && p.Order == TransformOrder.DecenterThenTilt && !p.RestoreAfter;
        return s with { Placement = plain ? null : p };
    }

    // ---- Aperture ----

    public static ClipShape Clip(ApertureShape? a) => a switch
    {
        ApertureShape.Circular => ClipShape.Circular,
        ApertureShape.Annular => ClipShape.Annular,
        ApertureShape.Rectangular => ClipShape.Rectangular,
        ApertureShape.Elliptical => ClipShape.Elliptical,
        _ => ClipShape.None,
    };

    /// <summary>A new clipping aperture sized by the semi-diameter (fixed, else automatic, else 5 mm).</summary>
    public static ApertureShape? MakeClip(ClipShape shape, double? fixedSemiDiameter, double? autoSemiDiameter)
    {
        var r = fixedSemiDiameter ?? autoSemiDiameter ?? 5;
        return shape switch
        {
            ClipShape.Circular => new ApertureShape.Circular(r),
            ClipShape.Annular => new ApertureShape.Annular(r / 3, r),
            ClipShape.Rectangular => new ApertureShape.Rectangular(r, r),
            ClipShape.Elliptical => new ApertureShape.Elliptical(r, r),
            _ => null,
        };
    }

    /// <summary>Automatic or fixed semi-diameter; a new fixed one starts at the current or engine-computed value.</summary>
    public static SemiDiameter SemiDiameterMode(Surface s, bool auto, double? autoValue) =>
        auto ? new SemiDiameter.Auto() : new SemiDiameter.Fixed(s.SemiDiameter.FixedValue ?? autoValue ?? 1);

    // ---- Material ----

    /// <summary>A catalogue glass's melt offset with one component changed; null when the material is not one.</summary>
    public static Surface? WithOffset(Surface s, Func<MaterialOffset, MaterialOffset> change) =>
        s.Material is MaterialRef.CatalogGlass g ? s with { Material = g with { Offset = change(g.Offset) } } : null;

    public static Parsed<double> OffsetDnD(string text) => Bounded(text, MaterialOffset.MaxDnD);
    public static Parsed<double> OffsetDvD(string text) => Bounded(text, MaterialOffset.MaxDvD);

    static Parsed<double> Bounded(string text, double max)
    {
        var p = LensValueParser.Finite(text);
        if (!p.Ok) return p;
        return Math.Abs(p.Value) <= max ? p : Parsed<double>.Fail("Value must be within ±" + NumberDisplay.Significant(max));
    }

    /// <summary>A positive, finite number (an F/#, a pressure, a wavelength).</summary>
    public static Parsed<double> Positive(string text)
    {
        var p = LensValueParser.Finite(text);
        return !p.Ok || p.Value > 0 ? p : Parsed<double>.Fail(LensValueParser.MustBePositive);
    }

    // ---- Diffractive ----

    /// <summary>The DOE's base switched to a plane, a conic or an even asphere, keeping c, k and the terms where they fit.</summary>
    public static SurfaceShape? WithDiffractiveBase(SurfaceShape shape, ShapeKind kind)
    {
        if (shape is not DiffractiveShape d) return null;
        var b = d.Base;
        SurfaceShape nb = kind switch
        {
            ShapeKind.Conic => new ConicShape(b.Curvature, b.ConicConstant),
            ShapeKind.EvenAsphere => new EvenAsphereShape(b.Curvature, b.ConicConstant, b.AsphereCoefficients),
            _ => new PlaneShape(),
        };
        return d with { Base = nb };
    }

    /// <summary>The trace order's efficiency at λ (µm), the relief blazed for the design order at λ₀ (non-dispersive estimate).</summary>
    public static double Efficiency(DiffractionEfficiency model, DiffractivePhase phase, double um)
    {
        if (!(um > 0)) return double.NaN;
        var blaze = phase.DesignOrder * phase.DesignWavelengthUm / um;
        return model.Efficiency(phase.TraceOrder, phase.TraceOrder, blaze);
    }

    // ---- Solves ----

    /// <summary>The solves the Add Solve menu offers for a surface (thickness solves; curvature ones when it has a curvature).</summary>
    public static IReadOnlyList<(string Title, ParamPath Target, SolveKind Kind)> SolvePresets(Surface s)
    {
        var list = new List<(string, ParamPath, SolveKind)>
        {
            ("Marginal ray height on thickness", new ParamPath.Thickness(s.Id), new SolveKind.MarginalRayHeight(0)),
            ("Chief ray height on thickness", new ParamPath.Thickness(s.Id), new SolveKind.ChiefRayHeight(0)),
        };
        if (s.Shape.HasCurvature)
        {
            list.Add(("Marginal ray angle on curvature", new ParamPath.Curvature(s.Id), new SolveKind.MarginalRayAngle(0)));
            list.Add(("F-number on curvature", new ParamPath.Curvature(s.Id), new SolveKind.FNumber(4)));
        }
        return list;
    }

    /// <summary>The system with a solve on <paramref name="target"/> (replacing one there).</summary>
    public static OpticalSystem AddSolve(OpticalSystem s, ParamPath target, SolveKind kind) =>
        s with { Solves = s.Solves.RemoveAll(x => x.Target == target).Add(new Solve(target, kind)) };

    /// <summary>What a solve's target is, for its row.</summary>
    public static string TargetLabel(ParamPath p) => p switch
    {
        ParamPath.Thickness => "Thickness",
        ParamPath.Curvature => "Curvature",
        ParamPath.Conic => "Conic",
        ParamPath.ParaxialFocalLength => "Focal length",
        ParamPath.SemiDiameter => "Semi-diameter",
        ParamPath.ShapeParam sp => sp.Parameter.Title(),
        _ => "Parameter",
    };

    // ---- System ----

    /// <summary>Wavelength presets: name, (µm, weight) pairs, primary.</summary>
    public static readonly IReadOnlyList<(string Name, (double Um, double Weight)[] Lines, int Primary)> WavelengthPresets =
    [
        ("F, d, C (visible)", [(0.4861327, 1), (0.5875618, 1), (0.6562725, 1)], 1),
        ("F′, e, C′ (visible)", [(0.4799914, 1), (0.5460740, 1), (0.6438469, 1)], 1),
        ("g, F, e, d, C", [(0.4358343, 1), (0.4861327, 1), (0.5460740, 1), (0.5875618, 1), (0.6562725, 1)], 3),
        ("d only", [(0.5875618, 1)], 0),
        ("e only", [(0.5460740, 1)], 0),
        ("Photopic (5)", [(0.47, 0.091), (0.51, 0.503), (0.555, 1), (0.61, 0.503), (0.65, 0.107)], 2),
        ("Near infrared", [(0.75, 1), (0.85, 1), (0.95, 1)], 1),
    ];

    public static OpticalSystem ApplyPreset(OpticalSystem s, int preset)
    {
        var p = WavelengthPresets[preset];
        return s with { Wavelengths = new WavelengthSet { Wavelengths = p.Lines.Select(l => new Wavelength(l.Um, l.Weight)).ToValueList(), Primary = p.Primary } };
    }

    /// <summary>A new field after the last: same x, y 25 % further (5 from an axial field).</summary>
    public static OpticalSystem AddField(OpticalSystem s)
    {
        var last = s.Fields.Fields.Count > 0 ? s.Fields.Fields[^1] : new Field();
        var f = new Field { X = last.X, Y = last.Y * 1.25 + (last.Y == 0 ? 5 : 0) };
        return s with { Fields = s.Fields with { Fields = s.Fields.Fields.Add(f) } };
    }

    /// <summary>A new wavelength 0.05 µm after the last.</summary>
    public static OpticalSystem AddWavelength(OpticalSystem s)
    {
        var w = s.Wavelengths.Wavelengths;
        return s with { Wavelengths = s.Wavelengths with { Wavelengths = w.Add(new Wavelength((w.Count > 0 ? w[^1].Um : 0.55) + 0.05)) } };
    }

    public static OpticalSystem WithField(OpticalSystem s, int k, Func<Field, Field> edit) =>
        s.Fields.Fields.IsValidIndex(k) ? s with { Fields = s.Fields with { Fields = s.Fields.Fields.With(k, edit) } } : s;

    public static OpticalSystem WithWavelength(OpticalSystem s, int k, Func<Wavelength, Wavelength> edit) =>
        s.Wavelengths.Wavelengths.IsValidIndex(k) ? s with { Wavelengths = s.Wavelengths with { Wavelengths = s.Wavelengths.Wavelengths.With(k, edit) } } : s;

    /// <summary>The settings of the engine's <c>convert_spec</c> for a field-kind or aperture-type switch.</summary>
    public static Dictionary<string, JsonNode?> ConvertSpecSettings(FieldKind? fieldKind, ApertureKind? aperture) => new()
    {
        ["field_kind"] = fieldKind is { } f ? JsonValue.Create(Wire.Name(f)) : null,
        ["aperture"] = aperture is { } a ? JsonValue.Create(Wire.Name(a)) : null,
    };

    /// <summary>Where a glass catalog comes from, for its row.</summary>
    public static string CatalogOrigin(bool builtin, bool library, bool inProject) => (builtin, library, inProject) switch
    {
        (true, _, _) => "Built into MKLight",
        (_, true, true) => "Library, recorded in the design",
        (_, true, false) => "Library",
        _ => "This design",
    };

    /// <summary>Whether any surface lacks rotational symmetry (toroid, biconic, XY or Zernike freeform).</summary>
    public static bool HasNonRotationalSurfaces(OpticalSystem s) =>
        s.Surfaces.Any(x => x.Shape.Kind is ShapeKind.Toroid or ShapeKind.Biconic or ShapeKind.XyPolynomial or ShapeKind.ZernikeSag);
}

/// <summary>Number text of the inspector: the app's one format (invariant, true minus, never −0).</summary>
public static class InspectorFormat
{
    /// <summary>Up to 6 significant digits (the Mac app's <c>Fmt.short</c>).</summary>
    public static string Short(double v) => LensFormat.Plain(v);

    public static string WithUnit(double v, string unit) =>
        string.IsNullOrEmpty(unit) || !double.IsFinite(v) ? Short(v) : unit == "°" ? Short(v) + "°" : Short(v) + " " + unit;

    /// <summary>A length in µm below 0.1 mm: <c>25.92 µm</c>, <c>0.1234 mm</c>.</summary>
    public static string Length(double mm)
    {
        if (!double.IsFinite(mm)) return Short(mm);
        return Math.Abs(mm) < 0.1 ? Fixed(mm * 1000, 2) + " µm" : Fixed(mm, 4) + " mm";
    }

    public static string Fixed(double v, int decimals)
    {
        if (!double.IsFinite(v)) return Short(v);
        var s = v.ToString("F" + decimals, CultureInfo.InvariantCulture);
        if (s.StartsWith('-')) s = s.Trim('-', '0', '.').Length == 0 ? s[1..] : "−" + s[1..];
        return s;
    }

    public static string Percent(double v, int decimals = 1) => double.IsFinite(v) ? Fixed(v * 100, decimals) + " %" : "—";

    /// <summary>The text a number field edits: every digit (∞ for infinity).</summary>
    public static string Exact(double v) =>
        double.IsInfinity(v) ? (v > 0 ? "∞" : "−∞") : v == 0 ? "0" : v.ToString("R", CultureInfo.InvariantCulture).Replace("E", "e");
}

/// <summary>
/// The x / y first order of a design that is not rotationally symmetric: the engine's <c>paraxial_xy</c> in the
/// <c>first_order</c> result, at the primary wavelength (the Mac app's <c>ParaxialXYData</c>).
/// </summary>
public sealed record FirstOrderXY(string? Symmetry, double EflX, double EflY, double BflX, double BflY, double Squeeze, string? SqueezeDefinition)
{
    static double Num(JsonNode? n) => n is JsonValue v && v.TryGetValue<double>(out var d) ? d : double.NaN;
    static string? Str(JsonNode? n) => n is JsonValue v && v.TryGetValue<string>(out var s) ? s : null;
    static int? Int(JsonNode? n) => n is JsonValue v && v.TryGetValue<double>(out var d) && d == Math.Round(d) ? (int)d : null;

    /// <summary>The primary wavelength's row (else the first), or null when the result has no <c>paraxial_xy</c>.</summary>
    public static FirstOrderXY? Read(JsonNode? json)
    {
        if (json?["paraxial_xy"] is not JsonObject xy) return null;
        var rows = (xy["wavelengths"] as JsonArray ?? []).OfType<JsonObject>().ToList();
        var primary = Int(xy["primary_wavelength"]);
        var row = rows.Where((r, k) => primary is { } w && (Int(r["wavelength"]) ?? k) == w).FirstOrDefault() ?? rows.FirstOrDefault();
        if (row is null || !double.IsFinite(Num(row["efl_y"]))) return null;
        var eflX = Num(row["efl_x"]);
        var eflY = Num(row["efl_y"]);
        var squeeze = row["squeeze"] is JsonValue ? Num(row["squeeze"]) : eflX != 0 ? eflY / eflX : double.NaN;
        return new FirstOrderXY(Str(xy["symmetry"]) ?? Str(json["symmetry"]), eflX, eflY, Num(row["bfl_x"]), Num(row["bfl_y"]), squeeze,
            Str(xy["squeeze_definition"]));
    }

    /// <summary>The engine's symmetry class when it reports one; otherwise the surface kinds decide.</summary>
    public static bool IsAnamorphic(JsonNode? json, OpticalSystem system)
    {
        var symmetry = Str(json?["paraxial_xy"]?["symmetry"]) ?? Str(json?["symmetry"]);
        if (symmetry is "rotational" or "meridional" or "none") return symmetry != "rotational";
        return InspectorEdits.HasNonRotationalSurfaces(system);
    }

    public static string SymmetryTitle(string s) => s switch { "rotational" => "Rotational", "meridional" => "Meridional plane", _ => "None" };

    /// <summary><c>1.5000×</c>, signed.</summary>
    public static string SqueezeText(double v) => double.IsFinite(v) ? InspectorFormat.Fixed(v, 4) + "×" : InspectorFormat.Short(v);
}
