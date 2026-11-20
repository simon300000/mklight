using System.Globalization;
using System.Text.Json.Nodes;

namespace MKLight.Model;

/// <summary>
/// Coefficient families of the extended surface shapes, addressed by
/// <c>ParamPath.ShapeCoefficient</c> with zero-based indices (SURFACES §7). Which
/// family belongs to which shape, and its limit, mirror the engine's <c>coefficient_slot!</c>.
/// </summary>
public enum CoefficientSet
{
    /// <summary>Odd asphere β1, β2, … (index 0 is β1).</summary>
    [Wire("odd")] Odd,

    /// <summary>Forbes Q a₀, a₁, … of <c>QCon</c> and <c>QBfs</c>.</summary>
    [Wire("q")] Q,

    /// <summary>Diffractive phase A₁, A₂, … (index 0 is A₁).</summary>
    [Wire("phase")] Phase,

    /// <summary>Biconic x polynomial from x⁴.</summary>
    [Wire("x_asphere")] XAsphere,

    /// <summary>Toroid or biconic y polynomial from y⁴.</summary>
    [Wire("y_asphere")] YAsphere,

    /// <summary>XY polynomial terms in <see cref="XYTerm"/> order.</summary>
    [Wire("xy")] Xy,

    /// <summary>Zernike sag terms; index 0 is Z1.</summary>
    [Wire("zernike")] Zernike,
}

public static class CoefficientLimits
{
    /// <summary>Even asphere A4 … A42, addressed by <c>ParamPath.Asphere</c>.</summary>
    public const int MaxEvenTerms = 20;
    public const int MaxOddTerms = 20;
    public const int MaxQTerms = 20;
    public const int MaxPhaseTerms = 10;
    public const int MaxAxisTerms = 10;

    /// <summary>Degrees 1 through 10.</summary>
    public const int MaxXYTerms = 65;

    public static readonly IReadOnlyList<CoefficientSet> All = Enum.GetValues<CoefficientSet>();

    /// <summary>
    /// Number of coefficients <paramref name="shape"/> allows in this set, or <c>null</c> when
    /// the set does not belong to the shape.
    /// </summary>
    public static int? MaxCount(this CoefficientSet set, SurfaceShape shape)
    {
        if (shape is SagOverlayShape o) return set.MaxCount(o.Base);
        return (shape, set) switch
        {
            (OddAsphereShape, CoefficientSet.Odd) => MaxOddTerms,
            (QConShape, CoefficientSet.Q) or (QBfsShape, CoefficientSet.Q) => MaxQTerms,
            (DiffractiveShape, CoefficientSet.Phase) => MaxPhaseTerms,
            (BiconicShape, CoefficientSet.XAsphere) or (BiconicShape, CoefficientSet.YAsphere) or (ToroidShape, CoefficientSet.YAsphere) =>
                MaxAxisTerms,
            (XyPolynomialShape, CoefficientSet.Xy) => MaxXYTerms,
            (ZernikeSagShape z, CoefficientSet.Zernike) => z.Ordering.MaxTerms(),
            _ => null,
        };
    }

    /// <summary>Short label of coefficient <paramref name="index"/>, e.g. <c>β3</c>, <c>Q2</c>, <c>A1</c>, <c>A4x</c>, <c>A6y</c>, <c>C(x²y)</c>, <c>Z7</c>.</summary>
    public static string TermLabel(this CoefficientSet set, int index) => set switch
    {
        CoefficientSet.Odd => $"β{index + 1}",
        CoefficientSet.Q => $"Q{index}",
        CoefficientSet.Phase => $"A{index + 1}",
        CoefficientSet.XAsphere => $"A{2 * index + 4}x",
        CoefficientSet.YAsphere => $"A{2 * index + 4}y",
        CoefficientSet.Xy => XYTerm.FromIndex(index) is { } t ? $"C({t.Monomial})" : $"C{index}",
        _ => $"Z{index + 1}",
    };
}

/// <summary>
/// Scalar parameters of the extended shapes, addressed by
/// <c>ParamPath.ShapeParam</c> (mirror of the engine's <c>parameter_slot!</c>).
/// </summary>
public enum ShapeParameter
{
    [Wire("curvature_x")] CurvatureX,
    [Wire("conic_x")] ConicX,
    [Wire("curvature_y")] CurvatureY,
    [Wire("conic_y")] ConicY,
    [Wire("norm_radius")] NormRadius,

    /// <summary>A sag overlay's astigmatism axis (degrees from +x toward +y).</summary>
    [Wire("astigmatism_angle_deg")] AstigmatismAngleDeg,
    [Wire("design_order")] DesignOrder,
    [Wire("design_wavelength_um")] DesignWavelengthUm,
    [Wire("trace_order")] TraceOrder,
}

public static class ShapeParameterExtensions
{
    public static readonly IReadOnlyList<ShapeParameter> All = Enum.GetValues<ShapeParameter>();

    /// <summary>Diffraction orders: integral <c>i32</c> values, never continuous variables.</summary>
    public static bool IsOrder(this ShapeParameter p) => p is ShapeParameter.DesignOrder or ShapeParameter.TraceOrder;

    public static string Title(this ShapeParameter p) => p switch
    {
        ShapeParameter.CurvatureX => "R_x",
        ShapeParameter.ConicX => "k_x",
        ShapeParameter.CurvatureY => "R_y",
        ShapeParameter.ConicY => "k_y",
        ShapeParameter.NormRadius => "Norm radius",
        ShapeParameter.AstigmatismAngleDeg => "Astigmatism axis",
        ShapeParameter.DesignOrder => "Design order",
        ShapeParameter.DesignWavelengthUm => "Design λ",
        _ => "Trace order",
    };
}

/// <summary>
/// A term <c>X^m Y^n</c> of an XY polynomial (SURFACES §5). Total degree
/// <c>d = m + n</c> rises from 1 to 10 and, within a degree, <c>m</c> falls from <c>d</c>
/// to 0: zero-based index <c>j = d(d+1)/2 − 1 + n</c>.
/// </summary>
public readonly record struct XYTerm(int M, int N)
{
    /// <summary>The term at zero-based coefficient <paramref name="index"/>, or <c>null</c> outside 0..&lt;65.</summary>
    public static XYTerm? FromIndex(int index)
    {
        if (index < 0 || index >= CoefficientLimits.MaxXYTerms) return null;
        var d = 1;
        while ((d + 1) * (d + 2) / 2 - 1 <= index) d++;
        var n = index - (d * (d + 1) / 2 - 1);
        return new XYTerm(d - n, n);
    }

    public int Degree => M + N;

    /// <summary>Zero-based coefficient index.</summary>
    public int Index => Degree * (Degree + 1) / 2 - 1 + N;

    /// <summary>E.g. <c>x</c>, <c>y²</c>, <c>x²y</c>.</summary>
    public string Monomial
    {
        get
        {
            static string Power(string v, int p) => p == 0 ? "" : p == 1 ? v : v + Superscript(p);
            return Power("x", M) + Power("y", N);
        }
    }

    internal static string Superscript(int n)
    {
        const string digits = "⁰¹²³⁴⁵⁶⁷⁸⁹";
        return string.Concat(n.ToString(CultureInfo.InvariantCulture).Select(c => digits[c - '0']));
    }
}

/// <summary>
/// A Zernike term <c>(n, m)</c> (SURFACES §6): <c>R_n^|m|(ρ)</c> times <c>cos(mθ)</c> for
/// m &gt; 0, <c>sin(|m|θ)</c> for m &lt; 0. Term numbers <c>j</c> start at 1 (piston).
/// </summary>
public readonly record struct ZernikeTerm(int N, int M)
{
    /// <summary>Term <paramref name="j"/> (1-based) in <paramref name="ordering"/>, or <c>null</c> outside 1…<c>MaxTerms</c>.</summary>
    public static ZernikeTerm? FromJ(int j, ZernikeOrdering ordering)
    {
        if (j < 1 || j > ordering.MaxTerms()) return null;
        if (ordering == ZernikeOrdering.Fringe)
        {
            if (j == 37) return new ZernikeTerm(12, 0);
            var k = (int)Math.Sqrt(j - 1);
            var o = j - 1 - k * k;
            var am = k - o / 2;
            return new ZernikeTerm(2 * k - am, o % 2 == 0 ? am : -am);
        }
        var nn = 0;
        while ((nn + 1) * (nn + 2) / 2 < j) nn++;
        var q = j - nn * (nn + 1) / 2 - 1;
        var abs = nn % 2 == 0 ? 2 * ((q + 1) / 2) : 2 * (q / 2) + 1;
        return new ZernikeTerm(nn, abs == 0 ? 0 : (j % 2 == 0 ? abs : -abs));
    }
}

/// <summary>
/// Term ordering and normalisation of a <see cref="ZernikeSagShape"/> (SURFACES §6).
/// <c>"noll"</c> decodes as <c>Standard</c>, which always encodes as <c>"standard"</c>.
/// </summary>
public enum ZernikeOrdering
{
    /// <summary>Unnormalised Fringe order, Z1–Z37.</summary>
    [Wire("fringe")] Fringe,

    /// <summary>Noll-normalised standard order, Z1–Z45 (n ≤ 8).</summary>
    [Wire("standard")] Standard,
}

public static class ZernikeOrderingExtensions
{
    public static readonly IReadOnlyList<ZernikeOrdering> All = Enum.GetValues<ZernikeOrdering>();

    public static string Title(this ZernikeOrdering o) => o == ZernikeOrdering.Fringe ? "Fringe" : "Standard (Noll)";

    /// <summary>Number of supported terms (the engine's <c>ZernikeOrdering::max_terms</c>).</summary>
    public static int MaxTerms(this ZernikeOrdering o) => o == ZernikeOrdering.Fringe ? 37 : 45;

    internal static ZernikeOrdering FromJson(JsonNode? node) => J.S(node) switch
    {
        "fringe" => ZernikeOrdering.Fringe,
        "standard" or "noll" => ZernikeOrdering.Standard,
        var s => throw J.Fail($"unknown Zernike ordering '{s}'"),
    };
}

// MARK: - Coefficient and parameter access

public abstract partial record SurfaceShape
{
    /// <summary>The coefficient sets this shape has (empty for legacy shapes).</summary>
    public IReadOnlyList<CoefficientSet> CoefficientSets => AllCoefficientSets.Where(s => s.MaxCount(this) is not null).ToArray();

    private static readonly CoefficientSet[] AllCoefficientSets = Enum.GetValues<CoefficientSet>();

    /// <summary>The scalar shape parameters this shape has.</summary>
    public IReadOnlyList<ShapeParameter> ShapeParameters => AllShapeParameters.Where(p => ShapeParameterValue(p) is not null).ToArray();

    private static readonly ShapeParameter[] AllShapeParameters = Enum.GetValues<ShapeParameter>();

    /// <summary>
    /// The stored coefficients of <paramref name="set"/> (possibly shorter than its limit;
    /// missing trailing terms are zero), or <c>null</c> when the set does not belong to the shape.
    /// </summary>
    public ValueList<double>? CoefficientsIn(CoefficientSet set) => (this, set) switch
    {
        (SagOverlayShape o, _) => o.Base.CoefficientsIn(set),
        (OddAsphereShape s, CoefficientSet.Odd) => s.Coefficients,
        (QConShape s, CoefficientSet.Q) => s.Coefficients,
        (QBfsShape s, CoefficientSet.Q) => s.Coefficients,
        (BiconicShape s, CoefficientSet.XAsphere) => s.XAsphere,
        (BiconicShape s, CoefficientSet.YAsphere) => s.YAsphere,
        (ToroidShape s, CoefficientSet.YAsphere) => s.YAsphere,
        (XyPolynomialShape s, CoefficientSet.Xy) => s.Coefficients,
        (ZernikeSagShape s, CoefficientSet.Zernike) => s.Coefficients,
        (DiffractiveShape s, CoefficientSet.Phase) => s.Phase.Coefficients,
        _ => null,
    };

    /// <summary>
    /// The shape with the coefficients of <paramref name="set"/> replaced, or <c>null</c> when the
    /// set does not belong to the shape (the count is not checked here; structural validation reports it).
    /// </summary>
    public SurfaceShape? WithCoefficients(ValueList<double> a, CoefficientSet set) => (this, set) switch
    {
        (SagOverlayShape o, _) => o.Base.WithCoefficients(a, set) is { } b ? o with { Base = b } : null,
        (OddAsphereShape s, CoefficientSet.Odd) => s with { Coefficients = a },
        (QConShape s, CoefficientSet.Q) => s with { Coefficients = a },
        (QBfsShape s, CoefficientSet.Q) => s with { Coefficients = a },
        (DiffractiveShape s, CoefficientSet.Phase) => s with { Phase = s.Phase with { Coefficients = a } },
        (BiconicShape s, CoefficientSet.XAsphere) => s with { XAsphere = a },
        (BiconicShape s, CoefficientSet.YAsphere) => s with { YAsphere = a },
        (ToroidShape s, CoefficientSet.YAsphere) => s with { YAsphere = a },
        (XyPolynomialShape s, CoefficientSet.Xy) => s with { Coefficients = a },
        (ZernikeSagShape s, CoefficientSet.Zernike) => s with { Coefficients = a },
        _ => null,
    };

    /// <summary>
    /// The value of a scalar shape parameter (orders as <c>double</c>), or <c>null</c>
    /// when the parameter does not belong to the shape.
    /// </summary>
    public double? ShapeParameterValue(ShapeParameter p)
    {
        // The angle belongs to the overlay of the sag shape (as in the engine).
        if (p == ShapeParameter.AstigmatismAngleDeg) return SagShape is SagOverlayShape a ? a.AstigmatismAngleDeg : null;
        if (this is SagOverlayShape o) return o.Base.ShapeParameterValue(p);
        return (this, p) switch
        {
            (ToroidShape s, ShapeParameter.CurvatureX) => s.CurvatureX,
            (BiconicShape s, ShapeParameter.CurvatureX) => s.CurvatureX,
            (ToroidShape s, ShapeParameter.CurvatureY) => s.CurvatureY,
            (BiconicShape s, ShapeParameter.CurvatureY) => s.CurvatureY,
            (BiconicShape s, ShapeParameter.ConicX) => s.ConicX,
            (ToroidShape s, ShapeParameter.ConicY) => s.ConicY,
            (BiconicShape s, ShapeParameter.ConicY) => s.ConicY,
            (QConShape s, ShapeParameter.NormRadius) => s.NormRadiusMm,
            (QBfsShape s, ShapeParameter.NormRadius) => s.NormRadiusMm,
            (XyPolynomialShape s, ShapeParameter.NormRadius) => s.NormRadiusMm,
            (ZernikeSagShape s, ShapeParameter.NormRadius) => s.NormRadiusMm,
            (DiffractiveShape s, ShapeParameter.NormRadius) => s.Phase.NormRadius,
            (DiffractiveShape s, ShapeParameter.DesignWavelengthUm) => s.Phase.DesignWavelengthUm,
            (DiffractiveShape s, ShapeParameter.DesignOrder) => s.Phase.DesignOrder,
            (DiffractiveShape s, ShapeParameter.TraceOrder) => s.Phase.TraceOrder,
            _ => null,
        };
    }

    /// <summary>
    /// Why <paramref name="v"/> cannot be stored in parameter <paramref name="p"/> (an order must be
    /// an integer <c>i32</c>, as in the engine's setter), or <c>null</c>.
    /// </summary>
    internal static string? OrderRejection(ShapeParameter p, double v)
    {
        if (!p.IsOrder()) return null;
        // The engine's check: `v.fract() != 0 || |v| > i32::MAX`.
        return Math.Truncate(v) != v || Math.Abs(v) > int.MaxValue ? "[invalid_diffraction_order] expected an i32" : null;
    }

    /// <summary>
    /// The shape with a scalar shape parameter set, or <c>null</c> when it does not belong to
    /// the shape, or for an order that is not an integral <c>i32</c>. Range rules (positive radius,
    /// nonzero design order) are structural validation, as in the engine.
    /// </summary>
    public SurfaceShape? WithShapeParameter(ShapeParameter p, double v)
    {
        if (ShapeParameterValue(p) is null || OrderRejection(p, v) is not null) return null;
        if (p == ShapeParameter.AstigmatismAngleDeg)
        {
            return this switch
            {
                DiffractiveShape d => d with { Base = d.Base.WithShapeParameter(p, v)! },
                SagOverlayShape o => o with { AstigmatismAngleDeg = v },
                _ => this,
            };
        }
        if (this is SagOverlayShape ov) return ov.Base.WithShapeParameter(p, v) is { } b ? ov with { Base = b } : null;
        return this switch
        {
            ToroidShape s => p switch
            {
                ShapeParameter.CurvatureX => s with { CurvatureX = v },
                ShapeParameter.CurvatureY => s with { CurvatureY = v },
                _ => s with { ConicY = v },
            },
            BiconicShape s => p switch
            {
                ShapeParameter.CurvatureX => s with { CurvatureX = v },
                ShapeParameter.CurvatureY => s with { CurvatureY = v },
                ShapeParameter.ConicX => s with { ConicX = v },
                _ => s with { ConicY = v },
            },
            QConShape s => s with { NormRadiusMm = v },
            QBfsShape s => s with { NormRadiusMm = v },
            XyPolynomialShape s => s with { NormRadiusMm = v },
            ZernikeSagShape s => s with { NormRadiusMm = v },
            DiffractiveShape s => s with
            {
                Phase = p switch
                {
                    ShapeParameter.NormRadius => s.Phase with { NormRadius = v },
                    ShapeParameter.DesignWavelengthUm => s.Phase with { DesignWavelengthUm = v },
                    ShapeParameter.DesignOrder => s.Phase with { DesignOrder = (int)v },
                    _ => s.Phase with { TraceOrder = (int)v },
                },
            },
            _ => null,
        };
    }
}
