using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
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

/// <summary>A titled group of a type's parameters, choices and coefficient list.</summary>
public sealed record ParamSection(string Title)
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

    static ShapeParam Radius(string key = "curvature", string title = "Radius", Func<SurfaceID, ParamPath>? path = null) =>
        new(key, title, "mm", ParamDisplay.RadiusOfCurvature, "Radius of curvature; ∞ for a plane. Stored as curvature c = 1/R.",
            path ?? (id => new ParamPath.Curvature(id)));

    static ShapeParam Conic(string key = "conic") =>
        new(key, "Conic", "", Help: "Conic constant k: 0 sphere, −1 paraboloid, < −1 hyperboloid.", Path: id => new ParamPath.Conic(id));

    static ShapeParam NormRadius(string key = "norm_radius") =>
        new(key, "Norm radius", "mm",
            Help: "Normalization radius r_norm: the polynomial is defined for r ≤ r_norm. Usually the clear semi-diameter.",
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
            (0, 0) => "piston",
            (1, 1) => "tilt",
            (2, 0) => "defocus",
            (2, 2) => "astigmatism",
            (3, 1) => "coma",
            (3, 3) => "trefoil",
            (4, 0) => "spherical",
            (4, 2) => "2nd astigmatism",
            (4, 4) => "tetrafoil",
            (5, 1) => "2nd coma",
            (5, 3) => "2nd trefoil",
            (5, 5) => "pentafoil",
            (6, 0) => "2nd spherical",
            (8, 0) => "3rd spherical",
            _ => null,
        };
        if (name is not null) return name + (m == 0 ? "" : t.N is 1 or 3 or 5 ? axis : t.M > 0 ? " 0°" : " 45°");
        return FormattableString.Invariant($"n{t.N} m{t.M}");
    }

    // ---- Descriptors ----

    public static readonly IReadOnlyList<SurfaceTypeDescriptor> All =
    [
        new("plane", ShapeKind.Plane, "Plane", SurfaceTypeGroup.Spherical, "A flat surface.", []),
        new("conic", ShapeKind.Conic, "Sphere / Conic", SurfaceTypeGroup.Spherical, "Sphere or conic of revolution.",
            [new ParamSection("Shape") { Params = [Radius(), Conic()] }]),
        new("even_asphere", ShapeKind.EvenAsphere, "Even Asphere", SurfaceTypeGroup.Aspheric, "Conic plus even polynomial terms A₄r⁴ + A₆r⁶ + …",
        [
            new ParamSection("Base Conic") { Params = [Radius(), Conic()] },
            new ParamSection("Polynomial") { Coefficients = EvenTerms },
        ]),
        new("odd_asphere", ShapeKind.OddAsphere, "Odd Asphere", SurfaceTypeGroup.Aspheric,
            "Conic plus all powers of r: β₁r + β₂r² + β₃r³ + … A nonzero β₁ is a cone at the vertex.",
        [
            new ParamSection("Base Conic") { Params = [Radius(), Conic()] },
            new ParamSection("Polynomial")
            {
                Coefficients = new("coefficients", (k, _) => "β" + Sub(k + 1), (k, _) => k == 0 ? "r" : "r" + Super(k + 1), k => MmPower(-k),
                    ShapeCoefficient(CoefficientSet.Odd), Limit(CoefficientSet.Odd)),
                Footer = "The paraxial curvature is c + 2β₂; β₁ has no first-order meaning.",
            },
        ]),
        new("q_con", ShapeKind.QCon, "Q-Type (Conic)", SurfaceTypeGroup.Aspheric,
            "Forbes conic-based asphere: conic plus u⁴ Σ aₘ Qₘᶜᵒⁿ(u²), u = r/r_norm. Keeps the vertex curvature.",
        [
            new ParamSection("Base Conic") { Params = [Radius(), Conic(), NormRadius()] },
            new ParamSection("Q Polynomial")
            {
                Coefficients = new("coefficients", (k, _) => "a" + Sub(k), (k, _) => "u⁴Q" + Sub(k) + "ᶜᵒⁿ", _ => "mm",
                    ShapeCoefficient(CoefficientSet.Q), Limit(CoefficientSet.Q)),
            },
        ]),
        new("q_bfs", ShapeKind.QBfs, "Q-Type (Best-Fit Sphere)", SurfaceTypeGroup.Aspheric,
            "Forbes slope-orthogonal departure from a best-fit sphere: u²(1−u²) Σ aₘ Qₘᵇᶠˢ(u²) / √(1−c²r²).",
        [
            new ParamSection("Best-Fit Sphere") { Params = [Radius(), NormRadius()] },
            new ParamSection("Q Polynomial")
            {
                Coefficients = new("coefficients", (k, _) => "a" + Sub(k), (k, _) => "Q" + Sub(k) + "ᵇᶠˢ", _ => "mm",
                    ShapeCoefficient(CoefficientSet.Q), Limit(CoefficientSet.Q)),
                Footer = "Vertex curvature c_bfs + 2Σaₘ Qₘ(0)/r_norm².",
            },
        ]),
        new("xy_polynomial", ShapeKind.XyPolynomial, "XY Polynomial", SurfaceTypeGroup.Freeform,
            "Freeform: conic plus Σ C(xᵐyⁿ) Xᵐ Yⁿ with X = x/r_norm, Y = y/r_norm; degrees 1 to 10.",
        [
            new ParamSection("Base Conic") { Params = [Radius(), Conic(), NormRadius()] },
            new ParamSection("XY Terms")
            {
                Coefficients = new("coefficients",
                    (k, _) => XYTerm.FromIndex(k) is { } t ? $"C({t.Monomial})" : "C" + k.ToString(CultureInfo.InvariantCulture),
                    (k, _) => XYTerm.FromIndex(k) is { } t ? "degree " + t.Degree.ToString(CultureInfo.InvariantCulture) : "",
                    _ => "mm", ShapeCoefficient(CoefficientSet.Xy), Limit(CoefficientSet.Xy), Sparse: true),
            },
        ]) { Preview = ShapePreviewKind.Freeform },
        new("zernike_sag", ShapeKind.ZernikeSag, "Zernike Sag", SurfaceTypeGroup.Freeform,
            "Freeform: conic plus Σ cⱼ Zⱼ(ρ, θ), ρ = r/r_norm, θ from +x. Sag coefficients in mm, piston and tilt kept.",
        [
            new ParamSection("Base Conic")
            {
                Params = [Radius(), Conic(), NormRadius()],
                Choices =
                [
                    new("ordering", "Ordering", [("fringe", "Fringe (37 terms)"), ("standard", "Standard / Noll (45 terms)")],
                        "Fringe terms are unnormalized; Standard (Noll) terms are normalized."),
                ],
            },
            new ParamSection("Zernike Terms")
            {
                Coefficients = new("coefficients", (k, _) => "Z" + Sub(k + 1),
                    (k, shape) => shape is ZernikeSagShape z && ZernikeTerm.FromJ(k + 1, z.Ordering) is { } t ? ZernikeName(t) : "",
                    _ => "mm", ShapeCoefficient(CoefficientSet.Zernike), Limit(CoefficientSet.Zernike), Sparse: true),
            },
        ]) { Preview = ShapePreviewKind.Freeform },
        new("toroid", ShapeKind.Toroid, "Toroid", SurfaceTypeGroup.Anamorphic,
            "The y profile (conic plus even terms) rotated about an axis parallel to y at the distance R_x.",
        [
            new ParamSection("Y Profile")
            {
                Params =
                [
                    Radius("curvature_y", "Radius y"),
                    new("conic_y", "Conic y", "", Help: "Conic constant of the y profile.", Path: id => new ParamPath.ShapeParam(id, ShapeParameter.ConicY)),
                ],
                Coefficients = AxisList("y", CoefficientSet.YAsphere, "B"),
            },
            new ParamSection("Rotation")
            {
                Params = [Radius("curvature_x", "Radius x", id => new ParamPath.ShapeParam(id, ShapeParameter.CurvatureX))],
                Footer = "Radius x is the distance from the vertex to the axis of rotation; ∞ makes a y cylinder.",
            },
        ]) { Preview = ShapePreviewKind.Anamorphic },
        new("biconic", ShapeKind.Biconic, "Biconic", SurfaceTypeGroup.Anamorphic,
            "Independent conics in x and y with even axis polynomials. A cylinder is a biconic with one plane axis.",
        [
            new ParamSection("X Section")
            {
                Params =
                [
                    Radius("curvature_x", "Radius x", id => new ParamPath.ShapeParam(id, ShapeParameter.CurvatureX)),
                    new("conic_x", "Conic x", "", Help: "Conic constant in x.", Path: id => new ParamPath.ShapeParam(id, ShapeParameter.ConicX)),
                ],
                Coefficients = AxisList("x", CoefficientSet.XAsphere, "A"),
            },
            new ParamSection("Y Section")
            {
                Params =
                [
                    Radius("curvature_y", "Radius y"),
                    new("conic_y", "Conic y", "", Help: "Conic constant in y.", Path: id => new ParamPath.ShapeParam(id, ShapeParameter.ConicY)),
                ],
                Coefficients = AxisList("y", CoefficientSet.YAsphere, "B"),
            },
        ]) { Preview = ShapePreviewKind.Anamorphic },
        new("diffractive", ShapeKind.Diffractive, "Diffractive", SurfaceTypeGroup.Diffractive,
            "A fixed-period phase Φ = Σ Aᵢ ρ²ⁱ (order +1, radians) on a plane, conic or even-asphere base.",
        [
            new ParamSection("Base Shape")
            {
                Params = [Radius("base.curvature"), Conic("base.conic")],
                Applies = s => s is DiffractiveShape d && d.Base.Kind != ShapeKind.Plane,
            },
            new ParamSection("Base Polynomial")
            {
                Coefficients = EvenTerms with { Key = "base.coefficients" },
                Applies = s => s is DiffractiveShape d && d.Base.Kind == ShapeKind.EvenAsphere,
            },
            new ParamSection("Phase")
            {
                Params = [NormRadius("phase.norm_radius")],
                Coefficients = new("phase.coefficients", (k, _) => "A" + Sub(k + 1), (k, _) => "ρ" + Super(2 * (k + 1)), _ => "rad",
                    ShapeCoefficient(CoefficientSet.Phase), Limit(CoefficientSet.Phase)),
                Footer = "A focusing phase has negative A₁. Coefficients describe order +1 and stay fixed when the wavelength or order changes.",
            },
            new ParamSection("Orders")
            {
                Params =
                [
                    new("phase.design_order", "Design order", "", ParamDisplay.Integer,
                        "The order the structure is designed for (nonzero). Metadata: it does not rescale the phase."),
                    new("phase.design_wavelength_um", "Design λ", "µm", Help: "Design wavelength λ₀. Metadata: it does not rescale the phase."),
                    new("phase.trace_order", "Trace order", "", ParamDisplay.Integer, "The order rays follow; 0 is ordinary refraction or reflection."),
                ],
                Footer = "Orders are discrete and never optimization variables.",
            },
        ]) { Preview = ShapePreviewKind.Diffractive, OrderEfficiency = true },
        new("paraxial", ShapeKind.Paraxial, "Ideal Lens", SurfaceTypeGroup.Special, "A thin, aberration-free lens of the given focal length.",
        [
            new ParamSection("Ideal Lens")
            {
                Params = [new("focal_length", "Focal length", "mm", Help: "Focal length of the ideal lens.", Path: id => new ParamPath.ParaxialFocalLength(id))],
            },
        ]) { Preview = ShapePreviewKind.IdealLens },
        new("coordinate_break", ShapeKind.CoordinateBreak, "Coordinate Break", SurfaceTypeGroup.Special,
            "Changes the coordinate frame only: decenters, then tilts (or the reverse).", [])
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

    public const string MustBeWhole = "Value must be a whole number";

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
