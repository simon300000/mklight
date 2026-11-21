using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using MKLight.App.Support;
using MKLight.Model;

namespace MKLight.App.ViewModels;

/// <summary>How a stored shape parameter is shown and entered.</summary>
public enum ParamDisplay
{
    /// <summary>The stored value as is.</summary>
    Value,
    /// <summary>A stored curvature shown and entered as a radius (∞ for 0).</summary>
    RadiusOfCurvature,
    /// <summary>A signed integer (diffraction orders).</summary>
    Integer,
}

/// <summary>
/// One scalar parameter of a surface type, addressed by its key path in the shape's JSON object (the
/// engine's model form), e.g. <c>curvature</c> or <c>phase.norm_radius</c>. <see cref="Path"/> makes it a variable.
/// </summary>
public sealed record ShapeParam(string Key, string Title, string Unit, ParamDisplay Display = ParamDisplay.Value, string Help = "",
    Func<SurfaceID, ParamPath>? Path = null);

/// <summary>A parameter with a fixed set of values (the Zernike ordering).</summary>
public sealed record ShapeChoice(string Key, string Title, IReadOnlyList<(string Value, string Title)> Options, string Help = "");

/// <summary>
/// A list of coefficients (polynomial terms) with add and remove. <c>Term</c> is the symbol of term i on a
/// shape (<c>A₄</c>, <c>β₃</c>, <c>C(x²y)</c>, <c>Z₇</c>), <c>Meaning</c> what it multiplies (<c>r⁴</c>, <c>coma x</c>).
/// <c>Sparse</c> lists (XY, Zernike) add any term from a menu and hide zero terms by default.
/// </summary>
public sealed record CoefficientList(
    string Key,
    Func<int, SurfaceShape, string> Term,
    Func<int, SurfaceShape, string> Meaning,
    Func<int, string> Unit,
    Func<SurfaceID, int, ParamPath>? Path,
    Func<SurfaceShape, int> MaxCount,
    bool Sparse = false);

/// <summary>A titled group of a type's parameters, choices and coefficient list; <c>Key</c> is its stable English name (row ids).</summary>
public sealed record ParamSection(string Key, string Title)
{
    public IReadOnlyList<ShapeParam> Params { get; init; } = [];
    public IReadOnlyList<ShapeChoice> Choices { get; init; } = [];
    public CoefficientList? Coefficients { get; init; }
    public string Footer { get; init; } = "";
    /// <summary>Whether the section applies to this shape (a DOE's base conic only on a curved base).</summary>
    public Func<SurfaceShape, bool> Applies { get; init; } = _ => true;
}

/// <summary>How the inspector previews a type.</summary>
public enum ShapePreviewKind { Radial, Anamorphic, Freeform, Diffractive, IdealLens, CoordinateBreak }

/// <summary>
/// A surface type described as data (the Mac app's <c>SurfaceTypeDescriptor</c>): its sections of parameters
/// and its preview. The inspector builds its form from this, so a new surface type needs a model case and one
/// descriptor here, not a new view. Sags for the previews come from <see cref="ShapeSag"/>.
/// </summary>
public sealed record SurfaceTypeDescriptor(string Id, ShapeKind Kind, string Title, SurfaceTypeGroup Group, string Summary,
    IReadOnlyList<ParamSection> Sections)
{
    /// <summary>Placement (decenters, tilts) is the type's main content.</summary>
    public bool PlacementFirst { get; init; }
    public ShapePreviewKind Preview { get; init; } = ShapePreviewKind.Radial;
    /// <summary>The surface's diffraction order efficiency model applies (a property of the surface, not of its shape).</summary>
    public bool OrderEfficiency { get; init; }
}

/// <summary>The surface types as data, and access to a shape's values by key path (the Mac app's <c>SurfaceTypes</c>).</summary>
public static class SurfaceTypes
{
    // ---- Building blocks ----

    static ShapeParam Radius(string key = "curvature", string? title = null, Func<SurfaceID, ParamPath>? path = null) =>
        new(key, title ?? Loc.S("Lens_Param_Radius"), "mm", ParamDisplay.RadiusOfCurvature, Loc.S("Lens_Help_Radius"),
            path ?? (id => new ParamPath.Curvature(id)));

    static ShapeParam Conic(string key = "conic") =>
        new(key, Loc.S("Lens_Param_Conic"), "", Help: Loc.S("Lens_Help_Conic"), Path: id => new ParamPath.Conic(id));

    static ShapeParam NormRadius(string key = "norm_radius") =>
        new(key, Loc.S("Lens_Param_NormRadius"), "mm",
            Help: Loc.S("Lens_Help_NormRadius"),
            Path: id => new ParamPath.ShapeParam(id, ShapeParameter.NormRadius));

    static Func<SurfaceID, int, ParamPath> ShapeCoefficient(CoefficientSet set) => (id, k) => new ParamPath.ShapeCoefficient(id, set, k);
    static Func<SurfaceShape, int> Limit(CoefficientSet set) => shape => set.MaxCount(shape) ?? 0;

    /// <summary>Unicode subscript digits, with a sign.</summary>
    public static string Sub(int n) => Script(n, "₀₁₂₃₄₅₆₇₈₉", '₋');
    /// <summary>Unicode superscript digits, with a sign.</summary>
    public static string Super(int n) => Script(n, "⁰¹²³⁴⁵⁶⁷⁸⁹", '⁻');

    static string Script(int n, string digits, char minus) =>
        (n < 0 ? minus.ToString() : "") + string.Concat(Math.Abs(n).ToString(CultureInfo.InvariantCulture).Select(c => digits[c - '0']));

    public static string MmPower(int p) => p == 0 ? "" : p == 1 ? "mm" : "mm" + Super(p);

    /// <summary>Even axis polynomial of a toroid or biconic: <c>A₄ x⁴</c>, <c>B₆ y⁶</c>, …</summary>
    static CoefficientList AxisList(string axis, CoefficientSet set, string letter) =>
        new(Wire.Name(set), (k, _) => letter + Sub(2 * k + 4), (k, _) => axis + Super(2 * k + 4), k => MmPower(-(2 * k + 3)),
            ShapeCoefficient(set), Limit(set));

    static readonly CoefficientList EvenTerms = new("coefficients", (k, _) => "A" + Sub(2 * k + 4), (k, _) => "r" + Super(2 * k + 4),
        k => MmPower(-(2 * k + 3)), (id, k) => new ParamPath.Asphere(id, k), _ => 10);

    /// <summary>A name for Zernike term (n, m).</summary>
    public static string ZernikeName(ZernikeTerm t)
    {
        var m = Math.Abs(t.M);
        var axis = t.M > 0 ? " x" : t.M < 0 ? " y" : "";
        string? name = (t.N, m) switch
        {
            (0, 0) => Loc.S("Lens_Zernike_Piston"),
            (1, 1) => Loc.S("Lens_Zernike_Tilt"),
            (2, 0) => Loc.S("Lens_Zernike_Defocus"),
            (2, 2) => Loc.S("Lens_Zernike_Astigmatism"),
            (3, 1) => Loc.S("Lens_Zernike_Coma"),
            (3, 3) => Loc.S("Lens_Zernike_Trefoil"),
            (4, 0) => Loc.S("Lens_Zernike_Spherical"),
            (4, 2) => Loc.S("Lens_Zernike_Astigmatism2"),
            (4, 4) => Loc.S("Lens_Zernike_Tetrafoil"),
            (5, 1) => Loc.S("Lens_Zernike_Coma2"),
            (5, 3) => Loc.S("Lens_Zernike_Trefoil2"),
            (5, 5) => Loc.S("Lens_Zernike_Pentafoil"),
            (6, 0) => Loc.S("Lens_Zernike_Spherical2"),
            (8, 0) => Loc.S("Lens_Zernike_Spherical3"),
            _ => null,
        };
        if (name is not null) return name + (m == 0 ? "" : t.N is 1 or 3 or 5 ? axis : t.M > 0 ? " 0°" : " 45°");
        return FormattableString.Invariant($"n{t.N} m{t.M}");
    }

    // ---- Descriptors ----

    public static readonly IReadOnlyList<SurfaceTypeDescriptor> All =
    [
        new("plane", ShapeKind.Plane, Loc.S("Lens_Type_Plane"), SurfaceTypeGroup.Spherical, Loc.S("Lens_Summary_Plane"), []),
        new("conic", ShapeKind.Conic, Loc.S("Lens_Type_Conic"), SurfaceTypeGroup.Spherical, Loc.S("Lens_Summary_Conic"),
            [new ParamSection("Shape", Loc.S("Lens_Section_Shape")) { Params = [Radius(), Conic()] }]),
        new("even_asphere", ShapeKind.EvenAsphere, Loc.S("Lens_Type_EvenAsphere"), SurfaceTypeGroup.Aspheric, Loc.S("Lens_Summary_EvenAsphere"),
        [
            new ParamSection("Base Conic", Loc.S("Lens_Section_BaseConic")) { Params = [Radius(), Conic()] },
            new ParamSection("Polynomial", Loc.S("Lens_Section_Polynomial")) { Coefficients = EvenTerms },
        ]),
        new("odd_asphere", ShapeKind.OddAsphere, Loc.S("Lens_Type_OddAsphere"), SurfaceTypeGroup.Aspheric,
            Loc.S("Lens_Summary_OddAsphere"),
        [
            new ParamSection("Base Conic", Loc.S("Lens_Section_BaseConic")) { Params = [Radius(), Conic()] },
            new ParamSection("Polynomial", Loc.S("Lens_Section_Polynomial"))
            {
                Coefficients = new("coefficients", (k, _) => "β" + Sub(k + 1), (k, _) => k == 0 ? "r" : "r" + Super(k + 1), k => MmPower(-k),
                    ShapeCoefficient(CoefficientSet.Odd), Limit(CoefficientSet.Odd)),
                Footer = Loc.S("Lens_Footer_OddAsphere"),
            },
        ]),
        new("q_con", ShapeKind.QCon, Loc.S("Lens_Type_QCon"), SurfaceTypeGroup.Aspheric,
            Loc.S("Lens_Summary_QCon"),
        [
            new ParamSection("Base Conic", Loc.S("Lens_Section_BaseConic")) { Params = [Radius(), Conic(), NormRadius()] },
            new ParamSection("Q Polynomial", Loc.S("Lens_Section_QPolynomial"))
            {
                Coefficients = new("coefficients", (k, _) => "a" + Sub(k), (k, _) => "u⁴Q" + Sub(k) + "ᶜᵒⁿ", _ => "mm",
                    ShapeCoefficient(CoefficientSet.Q), Limit(CoefficientSet.Q)),
            },
        ]),
        new("q_bfs", ShapeKind.QBfs, Loc.S("Lens_Type_QBfs"), SurfaceTypeGroup.Aspheric,
            Loc.S("Lens_Summary_QBfs"),
        [
            new ParamSection("Best-Fit Sphere", Loc.S("Lens_Section_BestFitSphere")) { Params = [Radius(), NormRadius()] },
            new ParamSection("Q Polynomial", Loc.S("Lens_Section_QPolynomial"))
            {
                Coefficients = new("coefficients", (k, _) => "a" + Sub(k), (k, _) => "Q" + Sub(k) + "ᵇᶠˢ", _ => "mm",
                    ShapeCoefficient(CoefficientSet.Q), Limit(CoefficientSet.Q)),
                Footer = Loc.S("Lens_Footer_QBfs"),
            },
        ]),
        new("xy_polynomial", ShapeKind.XyPolynomial, Loc.S("Lens_Type_XyPolynomial"), SurfaceTypeGroup.Freeform,
            Loc.S("Lens_Summary_XyPolynomial"),
        [
            new ParamSection("Base Conic", Loc.S("Lens_Section_BaseConic")) { Params = [Radius(), Conic(), NormRadius()] },
            new ParamSection("XY Terms", Loc.S("Lens_Section_XyTerms"))
            {
                Coefficients = new("coefficients",
                    (k, _) => XYTerm.FromIndex(k) is { } t ? $"C({t.Monomial})" : "C" + k.ToString(CultureInfo.InvariantCulture),
                    (k, _) => XYTerm.FromIndex(k) is { } t ? Loc.F("Lens_Term_Degree", t.Degree) : "",
                    _ => "mm", ShapeCoefficient(CoefficientSet.Xy), Limit(CoefficientSet.Xy), Sparse: true),
            },
        ]) { Preview = ShapePreviewKind.Freeform },
        new("zernike_sag", ShapeKind.ZernikeSag, Loc.S("Lens_Type_ZernikeSag"), SurfaceTypeGroup.Freeform,
            Loc.S("Lens_Summary_ZernikeSag"),
        [
            new ParamSection("Base Conic", Loc.S("Lens_Section_BaseConic"))
            {
                Params = [Radius(), Conic(), NormRadius()],
                Choices =
                [
                    new("ordering", Loc.S("Lens_Param_Ordering"), [("fringe", Loc.S("Lens_Ordering_Fringe")), ("standard", Loc.S("Lens_Ordering_Standard"))],
                        Loc.S("Lens_Help_Ordering")),
                ],
            },
            new ParamSection("Zernike Terms", Loc.S("Lens_Section_ZernikeTerms"))
            {
                Coefficients = new("coefficients", (k, _) => "Z" + Sub(k + 1),
                    (k, shape) => shape is ZernikeSagShape z && ZernikeTerm.FromJ(k + 1, z.Ordering) is { } t ? ZernikeName(t) : "",
                    _ => "mm", ShapeCoefficient(CoefficientSet.Zernike), Limit(CoefficientSet.Zernike), Sparse: true),
            },
        ]) { Preview = ShapePreviewKind.Freeform },
        new("toroid", ShapeKind.Toroid, Loc.S("Lens_Type_Toroid"), SurfaceTypeGroup.Anamorphic,
            Loc.S("Lens_Summary_Toroid"),
        [
            new ParamSection("Y Profile", Loc.S("Lens_Section_YProfile"))
            {
                Params =
                [
                    Radius("curvature_y", Loc.S("Lens_Param_RadiusY")),
                    new("conic_y", Loc.S("Lens_Param_ConicY"), "", Help: Loc.S("Lens_Help_ConicYProfile"), Path: id => new ParamPath.ShapeParam(id, ShapeParameter.ConicY)),
                ],
                Coefficients = AxisList("y", CoefficientSet.YAsphere, "B"),
            },
            new ParamSection("Rotation", Loc.S("Lens_Section_Rotation"))
            {
                Params = [Radius("curvature_x", Loc.S("Lens_Param_RadiusX"), id => new ParamPath.ShapeParam(id, ShapeParameter.CurvatureX))],
                Footer = Loc.S("Lens_Footer_Toroid"),
            },
        ]) { Preview = ShapePreviewKind.Anamorphic },
        new("biconic", ShapeKind.Biconic, Loc.S("Lens_Type_Biconic"), SurfaceTypeGroup.Anamorphic,
            Loc.S("Lens_Summary_Biconic"),
        [
            new ParamSection("X Section", Loc.S("Lens_Section_XSection"))
            {
                Params =
                [
                    Radius("curvature_x", Loc.S("Lens_Param_RadiusX"), id => new ParamPath.ShapeParam(id, ShapeParameter.CurvatureX)),
                    new("conic_x", Loc.S("Lens_Param_ConicX"), "", Help: Loc.S("Lens_Help_ConicX"), Path: id => new ParamPath.ShapeParam(id, ShapeParameter.ConicX)),
                ],
                Coefficients = AxisList("x", CoefficientSet.XAsphere, "A"),
            },
            new ParamSection("Y Section", Loc.S("Lens_Section_YSection"))
            {
                Params =
                [
                    Radius("curvature_y", Loc.S("Lens_Param_RadiusY")),
                    new("conic_y", Loc.S("Lens_Param_ConicY"), "", Help: Loc.S("Lens_Help_ConicY"), Path: id => new ParamPath.ShapeParam(id, ShapeParameter.ConicY)),
                ],
                Coefficients = AxisList("y", CoefficientSet.YAsphere, "B"),
            },
        ]) { Preview = ShapePreviewKind.Anamorphic },
        new("diffractive", ShapeKind.Diffractive, Loc.S("Lens_Type_Diffractive"), SurfaceTypeGroup.Diffractive,
            Loc.S("Lens_Summary_Diffractive"),
        [
            new ParamSection("Base Shape", Loc.S("Lens_Section_BaseShape"))
            {
                Params = [Radius("base.curvature"), Conic("base.conic")],
                Applies = s => s is DiffractiveShape d && d.Base.Kind != ShapeKind.Plane,
            },
            new ParamSection("Base Polynomial", Loc.S("Lens_Section_BasePolynomial"))
            {
                Coefficients = EvenTerms with { Key = "base.coefficients" },
                Applies = s => s is DiffractiveShape d && d.Base.Kind == ShapeKind.EvenAsphere,
            },
            new ParamSection("Phase", Loc.S("Lens_Section_Phase"))
            {
                Params = [NormRadius("phase.norm_radius")],
                Coefficients = new("phase.coefficients", (k, _) => "A" + Sub(k + 1), (k, _) => "ρ" + Super(2 * (k + 1)), _ => "rad",
                    ShapeCoefficient(CoefficientSet.Phase), Limit(CoefficientSet.Phase)),
                Footer = Loc.S("Lens_Footer_Phase"),
            },
            new ParamSection("Orders", Loc.S("Lens_Section_Orders"))
            {
                Params =
                [
                    new("phase.design_order", Loc.S("Lens_Param_DesignOrder"), "", ParamDisplay.Integer,
                        Loc.S("Lens_Help_DesignOrder")),
                    new("phase.design_wavelength_um", Loc.S("Lens_Param_DesignWavelength"), "µm", Help: Loc.S("Lens_Help_DesignWavelength")),
                    new("phase.trace_order", Loc.S("Lens_Param_TraceOrder"), "", ParamDisplay.Integer, Loc.S("Lens_Help_TraceOrder")),
                ],
                Footer = Loc.S("Lens_Footer_Orders"),
            },
        ]) { Preview = ShapePreviewKind.Diffractive, OrderEfficiency = true },
        new("paraxial", ShapeKind.Paraxial, Loc.S("Lens_Type_IdealLens"), SurfaceTypeGroup.Special, Loc.S("Lens_Summary_IdealLens"),
        [
            new ParamSection("Ideal Lens", Loc.S("Lens_Section_IdealLens"))
            {
                Params = [new("focal_length", Loc.S("Lens_Param_FocalLength"), "mm", Help: Loc.S("Lens_Help_FocalLength"), Path: id => new ParamPath.ParaxialFocalLength(id))],
            },
        ]) { Preview = ShapePreviewKind.IdealLens },
        new("coordinate_break", ShapeKind.CoordinateBreak, Loc.S("Lens_Type_CoordinateBreak"), SurfaceTypeGroup.Special,
            Loc.S("Lens_Summary_CoordinateBreak"), [])
        { PlacementFirst = true, Preview = ShapePreviewKind.CoordinateBreak },
    ];

    public static SurfaceTypeDescriptor Descriptor(SurfaceShape shape) => Descriptor(shape.Kind);
    public static SurfaceTypeDescriptor Descriptor(ShapeKind kind) => All.FirstOrDefault(d => d.Kind == kind) ?? All[0];

    /// <summary>The type menu's title of a choice (a cylinder shows as its preset).</summary>
    public static string Title(SurfaceTypeChoice choice) => SurfaceTypeCatalog.EntryFor(choice).Title;

    // ---- JSON access ----

    /// <summary>The JSON value at a dotted key path (<c>phase.norm_radius</c>), or null.</summary>
    public static JsonNode? Value(SurfaceShape shape, string key)
    {
        JsonNode? j = shape.ToJson();
        foreach (var k in key.Split('.'))
        {
            if (j is not JsonObject o || !o.TryGetPropertyValue(k, out j)) return null;
        }
        return j;
    }

    /// <summary>
    /// The shape with the value at a dotted key path replaced, or null when the model does not accept the
    /// result (unknown key, wrong type).
    /// </summary>
    public static SurfaceShape? WithValue(SurfaceShape shape, string key, JsonNode? value)
    {
        var root = shape.ToJson();
        var parts = key.Split('.');
        var o = root as JsonObject;
        for (var i = 0; i < parts.Length - 1; i++)
        {
            if (o is null || !o.TryGetPropertyValue(parts[i], out var inner)) return null;
            o = inner as JsonObject;
        }
        if (o is null || !o.ContainsKey(parts[^1])) return null;
        o[parts[^1]] = value;
        try { return SurfaceShape.FromJson(root); }
        catch (Exception e) when (e is JsonException or InvalidOperationException or FormatException) { return null; }
    }

    static double? Number(JsonNode? n) => n is not JsonValue v ? null
        : v.TryGetValue<double>(out var d) ? d : v.TryGetValue<int>(out var i) ? i : v.TryGetValue<long>(out var l) ? l : null;

    /// <summary>Display value of a parameter (a radius for a curvature), or null when the shape has none.</summary>
    public static double? DisplayValue(ShapeParam p, SurfaceShape shape)
    {
        if (Number(Value(shape, p.Key)) is not { } v) return null;
        return p.Display == ParamDisplay.RadiusOfCurvature ? RadiusDisplay.RadiusFromCurvature(v) : v;
    }

    public static string MustBeWhole => Loc.S("Lens_Error_MustBeWhole");

    /// <summary>Parses entered text for a parameter into the number it stores (a curvature for a radius).</summary>
    public static Parsed<double> StoredValue(ShapeParam p, string text) => p.Display switch
    {
        ParamDisplay.RadiusOfCurvature => LensValueParser.CurvatureFromRadius(text),
        ParamDisplay.Integer => LensValueParser.Finite(text) is { Ok: true } f
            ? Math.Round(f.Value) == f.Value && Math.Abs(f.Value) <= int.MaxValue ? f : Parsed<double>.Fail(MustBeWhole)
            : LensValueParser.Finite(text),
        _ => LensValueParser.Finite(text),
    };

    /// <summary>A display value as the stored one (a radius → curvature).</summary>
    public static double Stored(ShapeParam p, double display) =>
        p.Display == ParamDisplay.RadiusOfCurvature ? RadiusDisplay.CurvatureFromRadius(display) : display;

    /// <summary>The shape with a parameter set to its stored value, or null when refused.</summary>
    public static SurfaceShape? WithParam(SurfaceShape shape, ShapeParam p, double stored) =>
        WithValue(shape, p.Key, p.Display == ParamDisplay.Integer ? JsonValue.Create((int)stored) : JsonValue.Create(stored));

    /// <summary>Coefficients of a list on a shape (empty when it has none).</summary>
    public static IReadOnlyList<double> Coefficients(CoefficientList list, SurfaceShape shape) =>
        Value(shape, list.Key) is JsonArray a ? a.Select(n => Number(n) ?? 0).ToList() : [];

    /// <summary>Whether the list belongs to the shape.</summary>
    public static bool HasList(CoefficientList list, SurfaceShape shape) => Value(shape, list.Key) is JsonArray;

    public static SurfaceShape? WithCoefficients(SurfaceShape shape, CoefficientList list, IEnumerable<double> values) =>
        WithValue(shape, list.Key, new JsonArray(values.Select(v => (JsonNode?)JsonValue.Create(v)).ToArray()));

    // ---- Coefficient list edits (the Mac app's CoefficientRows) ----

    /// <summary>The list with term k set.</summary>
    public static IReadOnlyList<double> SetTerm(IReadOnlyList<double> values, int k, double v)
    {
        var a = values.ToList();
        while (a.Count <= k) a.Add(0);
        a[k] = v;
        return a;
    }

    /// <summary>The list with term k added (zero, extending the list).</summary>
    public static IReadOnlyList<double> AddTerm(IReadOnlyList<double> values, int k)
    {
        var a = values.ToList();
        while (a.Count <= k) a.Add(0);
        return a;
    }

    /// <summary>
    /// The list with term k removed: the last term goes (and in a sparse list the zero terms before it that
    /// were not just added, <paramref name="revealed"/>); any other term becomes zero.
    /// </summary>
    public static IReadOnlyList<double> RemoveTerm(IReadOnlyList<double> values, int k, bool sparse, IReadOnlySet<int>? revealed = null)
    {
        var a = values.ToList();
        if (k == a.Count - 1)
        {
            a.RemoveAt(a.Count - 1);
            if (sparse)
                while (a.Count > 0 && a[^1] == 0 && !(revealed?.Contains(a.Count - 1) ?? false)) a.RemoveAt(a.Count - 1);
        }
        else if (k >= 0 && k < a.Count)
        {
            a[k] = 0;
        }
        return a;
    }

    /// <summary>Indices of the terms a list shows: all, or in a sparse list the nonzero, revealed and last ones.</summary>
    public static IReadOnlyList<int> VisibleTerms(IReadOnlyList<double> values, bool sparse, bool showAll, IReadOnlySet<int> revealed) =>
        Enumerable.Range(0, values.Count).Where(k => !sparse || showAll || values[k] != 0 || revealed.Contains(k) || k == values.Count - 1).ToList();

    // ---- Previews ----

    /// <summary>The half-width a preview covers: the clear semi-diameter, else the normalisation radius, else 10 mm.</summary>
    public static double PreviewRadius(SurfaceShape shape, double? semiDiameter)
    {
        if (semiDiameter is { } h && h > 0 && double.IsFinite(h)) return h;
        if (shape.NormRadius is { } r && r > 0) return r;
        return 10;
    }

    /// <summary>Sag profile points (t, z) from −h to h along x or y, or null.</summary>
    public static IReadOnlyList<(double T, double Z)>? Profile(SurfaceShape shape, double h, CylinderAxis axis = CylinderAxis.Y, int samples = 64)
    {
        if (!(h > 0) || !double.IsFinite(h)) return null;
        var list = new List<(double, double)>();
        for (var k = 0; k <= samples; k++)
        {
            var t = -h + 2 * h * k / samples;
            var z = axis == CylinderAxis.Y ? ShapeSag.Sag(shape, 0, t) : ShapeSag.Sag(shape, t, 0);
            if (z is { } v && double.IsFinite(v)) list.Add((t, v));
        }
        return list.Count > 2 ? list : null;
    }

    /// <summary>An n × n grid of the departure from the base conic over the disk of radius h (row-major, top row first); NaN outside.</summary>
    public static double[] DepartureMap(SurfaceShape shape, double h, int n = 40)
    {
        var cells = new double[n * n];
        for (var k = 0; k < n * n; k++)
        {
            int i = k % n, j = k / n;
            var x = -h + 2 * h * (i + 0.5) / n;
            var y = h - 2 * h * (j + 0.5) / n;
            cells[k] = x * x + y * y <= h * h && ShapeSag.Departure(shape, x, y) is { } d ? d : double.NaN;
        }
        return cells;
    }
}
