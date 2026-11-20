using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using MKLight.Engine;
using MKLight.Model;

namespace MKLight.App.ViewModels.Jobs;

// The tolerance set of a design (the Mac app's ToleranceModel; docs/OPTIMIZATION.md, `tolerance`): rows in the
// units a lens designer thinks in (fringes of power and irregularity, ΔR, arc minutes, Δn, % of V_d), expanded to
// the engine's `ToleranceOperand`s on parameter paths; compensators; the criterion; and the run's mode, trials,
// seed, percentiles and recorded trials. Saved in the project's `tolerances` section.
//
// Swift structs mutated in place are classes here with Clone(); equality of rows, compensators and settings is by
// their saved JSON (SameAs), session ids aside.

// ---- Kinds and units ----

/// <summary>What a tolerance row perturbs.</summary>
public enum ToleranceKind
{
    /// <summary>Surface power: the curvature, in fringes, ΔR or % of R.</summary>
    [Wire("radius")] Radius,
    /// <summary>Surface irregularity in test fringes: astigmatic (x² − y²).</summary>
    [Wire("irregularity_astigmatism")] IrregularityAstigmatism,
    /// <summary>Surface irregularity in test fringes: spherical (the engine's sag overlay).</summary>
    [Wire("irregularity_spherical")] IrregularitySpherical,
    /// <summary>The axial gap after a surface (glass or air).</summary>
    [Wire("thickness")] Thickness,
    /// <summary>Surface decentre in x / y, the rest of the system staying in place.</summary>
    [Wire("decenter_x")] DecenterX,
    [Wire("decenter_y")] DecenterY,
    /// <summary>Surface tilt about x / y (a wedge), the rest staying in place.</summary>
    [Wire("tilt_x")] TiltX,
    [Wire("tilt_y")] TiltY,
    /// <summary>A whole element (a surface range, a cemented group together) moved rigidly about a pivot vertex.</summary>
    [Wire("element_decenter_x")] ElementDecenterX,
    [Wire("element_decenter_y")] ElementDecenterY,
    [Wire("element_tilt_x")] ElementTiltX,
    [Wire("element_tilt_y")] ElementTiltY,
    /// <summary>The refractive index n_d of the glass after a surface (a catalogue glass's Δn_d is added to its melt offset).</summary>
    [Wire("index")] Index,
    /// <summary>The Abbe number V_d of the glass after a surface (model n_d / V_d, catalogue ΔV_d).</summary>
    [Wire("abbe")] Abbe,
    /// <summary>Any other parameter, in its own unit.</summary>
    [Wire("other")] Other,
}

public static class ToleranceKinds
{
    public static readonly IReadOnlyList<ToleranceKind> All = Enum.GetValues<ToleranceKind>();

    /// <summary>Kinds offered when adding a row on a surface (other parameters come from a list; element kinds from the elements).</summary>
    public static readonly IReadOnlyList<ToleranceKind> Addable =
    [
        ToleranceKind.Radius, ToleranceKind.IrregularityAstigmatism, ToleranceKind.IrregularitySpherical, ToleranceKind.Thickness,
        ToleranceKind.DecenterX, ToleranceKind.DecenterY, ToleranceKind.TiltX, ToleranceKind.TiltY, ToleranceKind.Index, ToleranceKind.Abbe,
    ];

    /// <summary>Kinds offered when adding a row on an element.</summary>
    public static readonly IReadOnlyList<ToleranceKind> AddableElement =
        [ToleranceKind.ElementDecenterX, ToleranceKind.ElementDecenterY, ToleranceKind.ElementTiltX, ToleranceKind.ElementTiltY];

    static readonly IReadOnlyList<ToleranceUnit> Fr = [ToleranceUnit.Fringes], Mm = [ToleranceUnit.Mm], Tilt = [ToleranceUnit.Arcmin, ToleranceUnit.Degrees];

    public static string Wire(this ToleranceKind k) => Model.Wire.Name(k);

    public static ToleranceKind? Parse(string? wire) => wire is null ? null : Model.Wire.Parse<ToleranceKind>(wire);

    public static string Title(this ToleranceKind k) => k switch
    {
        ToleranceKind.Radius => "Radius",
        ToleranceKind.IrregularityAstigmatism => "Astig. irregularity",
        ToleranceKind.IrregularitySpherical => "Sph. irregularity",
        ToleranceKind.Thickness => "Thickness",
        ToleranceKind.DecenterX => "Surface decenter x",
        ToleranceKind.DecenterY => "Surface decenter y",
        ToleranceKind.TiltX => "Surface tilt x",
        ToleranceKind.TiltY => "Surface tilt y",
        ToleranceKind.ElementDecenterX => "Element decenter x",
        ToleranceKind.ElementDecenterY => "Element decenter y",
        ToleranceKind.ElementTiltX => "Element tilt x",
        ToleranceKind.ElementTiltY => "Element tilt y",
        ToleranceKind.Index => "Index",
        ToleranceKind.Abbe => "Abbe",
        _ => "Parameter",
    };

    /// <summary>The title in the tolerance table, where the surface column tells a surface (<c>S3</c>) from an element (<c>L2</c>).</summary>
    public static string ShortTitle(this ToleranceKind k) => k switch
    {
        ToleranceKind.IrregularityAstigmatism => "Irreg. astig.",
        ToleranceKind.IrregularitySpherical => "Irreg. sph.",
        ToleranceKind.DecenterX or ToleranceKind.ElementDecenterX => "Decenter x",
        ToleranceKind.DecenterY or ToleranceKind.ElementDecenterY => "Decenter y",
        ToleranceKind.TiltX or ToleranceKind.ElementTiltX => "Tilt x",
        ToleranceKind.TiltY or ToleranceKind.ElementTiltY => "Tilt y",
        _ => k.Title(),
    };

    /// <summary>An element's rigid motion (a surface range about a pivot).</summary>
    public static bool IsElement(this ToleranceKind k) => k.Component() is not null;

    /// <summary>The rigid-motion component of an element kind.</summary>
    public static PlacementComponent? Component(this ToleranceKind k) => k switch
    {
        ToleranceKind.ElementDecenterX => PlacementComponent.DecenterX,
        ToleranceKind.ElementDecenterY => PlacementComponent.DecenterY,
        ToleranceKind.ElementTiltX => PlacementComponent.TiltX,
        ToleranceKind.ElementTiltY => PlacementComponent.TiltY,
        _ => null,
    };

    /// <summary>The element kind of a rigid-motion component (tilt z has none).</summary>
    public static ToleranceKind? Element(PlacementComponent c) => c switch
    {
        PlacementComponent.DecenterX => ToleranceKind.ElementDecenterX,
        PlacementComponent.DecenterY => ToleranceKind.ElementDecenterY,
        PlacementComponent.TiltX => ToleranceKind.ElementTiltX,
        PlacementComponent.TiltY => ToleranceKind.ElementTiltY,
        _ => null,
    };

    /// <summary>The units a row of this kind can be edited in, the first the default.</summary>
    public static IReadOnlyList<ToleranceUnit> Units(this ToleranceKind k) => k switch
    {
        ToleranceKind.Radius => [ToleranceUnit.Fringes, ToleranceUnit.RadiusMM, ToleranceUnit.Percent],
        ToleranceKind.IrregularityAstigmatism or ToleranceKind.IrregularitySpherical => Fr,
        ToleranceKind.Thickness or ToleranceKind.DecenterX or ToleranceKind.DecenterY or ToleranceKind.ElementDecenterX or ToleranceKind.ElementDecenterY => Mm,
        ToleranceKind.TiltX or ToleranceKind.TiltY or ToleranceKind.ElementTiltX or ToleranceKind.ElementTiltY => Tilt,
        ToleranceKind.Index => [ToleranceUnit.Index],
        ToleranceKind.Abbe => [ToleranceUnit.Percent, ToleranceUnit.Abbe],
        _ => [ToleranceUnit.Native],
    };

    public static ToleranceUnit DefaultUnit(this ToleranceKind k) => k.Units()[0];
}

/// <summary>The unit a row's limits are written in.</summary>
public enum ToleranceUnit
{
    /// <summary>Test fringes at the test wavelength: of power over the clear aperture, or of irregularity.</summary>
    [Wire("fringes")] Fringes,
    /// <summary>ΔR in mm.</summary>
    [Wire("radius_mm")] RadiusMM,
    /// <summary>Percent (of R, or of V_d).</summary>
    [Wire("percent")] Percent,
    [Wire("mm")] Mm,
    [Wire("arcmin")] Arcmin,
    [Wire("degrees")] Degrees,
    /// <summary>Δn (absolute).</summary>
    [Wire("index")] Index,
    /// <summary>ΔV_d (absolute).</summary>
    [Wire("abbe")] Abbe,
    /// <summary>The parameter's own unit.</summary>
    [Wire("native")] Native,
}

public static class ToleranceUnits
{
    public static string Wire(this ToleranceUnit u) => Model.Wire.Name(u);

    public static ToleranceUnit? Parse(string? wire) => wire is null ? null : Model.Wire.Parse<ToleranceUnit>(wire);

    public static string Label(this ToleranceUnit u) => u switch
    {
        ToleranceUnit.Fringes => "fr",
        ToleranceUnit.RadiusMM => "mm",
        ToleranceUnit.Percent => "%",
        ToleranceUnit.Mm => "mm",
        ToleranceUnit.Arcmin => "′",
        ToleranceUnit.Degrees => "°",
        ToleranceUnit.Index => "Δn",
        ToleranceUnit.Abbe => "ΔV",
        _ => "",
    };

    public static string Title(this ToleranceUnit u) => u switch
    {
        ToleranceUnit.Fringes => "Test fringes",
        ToleranceUnit.RadiusMM => "ΔR (mm)",
        ToleranceUnit.Percent => "Percent",
        ToleranceUnit.Mm => "mm",
        ToleranceUnit.Arcmin => "Arc minutes",
        ToleranceUnit.Degrees => "Degrees",
        ToleranceUnit.Index => "Δn",
        ToleranceUnit.Abbe => "ΔV_d",
        _ => "Parameter units",
    };

    /// <summary>The unit after a number: <c>" fr"</c>, <c>"′"</c> (no space), <c>""</c>.</summary>
    internal static string Suffix(this ToleranceUnit u) => u.Label() is { Length: > 0 } l ? (u == ToleranceUnit.Arcmin ? l : " " + l) : "";
}

public enum ToleranceDistributionKind { Uniform, Normal, EndPoint, Parabolic }

/// <summary>How a trial draws a perturbation within [−minus, +plus] (the engine's <c>Distribution</c>).</summary>
public readonly record struct ToleranceDistribution(ToleranceDistributionKind Kind, double Sigmas = 0)
{
    public static ToleranceDistribution Uniform => new(ToleranceDistributionKind.Uniform);

    /// <summary>Normal, truncated at the limits, which lie at ±<paramref name="sigmas"/> σ.</summary>
    public static ToleranceDistribution Normal(double sigmas) => new(ToleranceDistributionKind.Normal, sigmas);

    /// <summary>Exactly −minus or +plus (worst-case style).</summary>
    public static ToleranceDistribution EndPoint => new(ToleranceDistributionKind.EndPoint);

    /// <summary>Density ∝ δ², most builds near the limits (a part adjusted only until it is within tolerance).</summary>
    public static ToleranceDistribution Parabolic => new(ToleranceDistributionKind.Parabolic);

    public static readonly IReadOnlyList<ToleranceDistribution> Choices = [Uniform, Normal(2), Normal(3), Parabolic, EndPoint];

    public JsonObject ToJson() => Kind switch
    {
        ToleranceDistributionKind.Normal => new JsonObject { ["type"] = "truncated_normal", ["sigmas"] = Sigmas },
        ToleranceDistributionKind.EndPoint => new JsonObject { ["type"] = "end_point" },
        ToleranceDistributionKind.Parabolic => new JsonObject { ["type"] = "parabolic" },
        _ => new JsonObject { ["type"] = "uniform" },
    };

    public static ToleranceDistribution FromJson(JsonNode? json) => json.At("type").StringOrNull() switch
    {
        "truncated_normal" => Normal(json.At("sigmas").Nf() is { } k && !double.IsNaN(k) ? k : 2),
        "end_point" => EndPoint,
        "parabolic" => Parabolic,
        _ => Uniform,
    };

    public string Title => Kind switch
    {
        ToleranceDistributionKind.Normal => $"Normal (limits at ±{MeritFormat.Short(Sigmas)} σ)",
        ToleranceDistributionKind.EndPoint => "End points",
        ToleranceDistributionKind.Parabolic => "Parabolic (mostly near the limits)",
        _ => "Uniform",
    };
}

/// <summary>Why a row cannot run.</summary>
public sealed record ToleranceIssue(string Text);

/// <summary>The engine's limits of a row in parameter units, or why the row cannot be expanded.</summary>
public readonly record struct ToleranceLimits(double Minus, double Plus, ToleranceIssue? Issue)
{
    public bool Ok => Issue is null;
    internal static ToleranceLimits Of(double minus, double plus) => new(minus, plus, null);
    internal static ToleranceLimits Fail(string why) => new(double.NaN, double.NaN, new ToleranceIssue(why));
}

// ---- Rows ----

/// <summary>
/// What a row needs to become an engine operand: the system, the clear semi-diameters (fringes of power are counted
/// over them), the test wavelength and the Abbe numbers of catalogue glasses (for % of V_d).
/// </summary>
public sealed record ToleranceContext(OpticalSystem System)
{
    static readonly IReadOnlyDictionary<SurfaceID, double> None = new Dictionary<SurfaceID, double>();

    /// <summary>Clear semi-diameters (mm), fixed or as the engine's layout resolved them.</summary>
    public IReadOnlyDictionary<SurfaceID, double> SemiDiameters { get; init; } = None;

    /// <summary>Interferometer wavelength for fringes (µm): He–Ne by default.</summary>
    public double TestWavelengthUm { get; init; } = ToleranceSettings.DefaultTestWavelengthUm;

    /// <summary>V_d of the catalogue glass after a surface, its offset included (the engine's <c>glass_info</c>).</summary>
    public IReadOnlyDictionary<SurfaceID, double> AbbeNumbers { get; init; } = None;

    /// <summary>The clear semi-diameter of a surface: its fixed value, else the layout's.</summary>
    public double? SemiDiameter(SurfaceID id)
    {
        if (System.SurfaceIndexOf(id) is { } i && System.Surfaces[i].SemiDiameter.FixedValue is { } v) return v;
        return SemiDiameters.TryGetValue(id, out var d) ? d : null;
    }

    /// <summary>V_d of the glass after a surface: a model glass's own, a catalogue glass's from the engine.</summary>
    public double? AbbeNumber(SurfaceID id)
    {
        if (System.SurfaceIndexOf(id) is not { } i) return null;
        if (System.Surfaces[i].Material is MaterialRef.ModelGlass m) return m.Vd;
        return AbbeNumbers.TryGetValue(id, out var d) ? d : null;
    }
}

/// <summary>
/// One tolerance: a kind on a surface (or an element's surface range, or any parameter), its − and + limits in the
/// row's unit, and the distribution of Monte-Carlo trials.
/// </summary>
public sealed class ToleranceRow
{
    /// <summary>Session identity (not saved).</summary>
    public Guid Id { get; set; } = Guid.NewGuid();
    public ToleranceKind Kind { get; set; }
    /// <summary>The surface, or an element's first surface.</summary>
    public SurfaceID Surface { get; set; }
    /// <summary>An element's last surface (inclusive; <c>null</c>: <see cref="Surface"/> alone).</summary>
    public SurfaceID? Last { get; set; }
    /// <summary>The vertex an element tilts about (<c>null</c>: its first surface).</summary>
    public SurfaceID? Pivot { get; set; }
    /// <summary>The parameter of an <see cref="ToleranceKind.Other"/> row.</summary>
    public ParamPath? Param { get; set; }
    /// <summary>Magnitudes of the limits (≥ 0), in <see cref="Unit"/>.</summary>
    public double Minus { get; set; }
    public double Plus { get; set; }
    public ToleranceUnit Unit { get; set; }
    public ToleranceDistribution Distribution { get; set; } = ToleranceDistribution.Uniform;

    public ToleranceRow(
        ToleranceKind kind, SurfaceID surface, double minus, double? plus = null, ToleranceUnit? unit = null, ParamPath? param = null,
        SurfaceID? last = null, SurfaceID? pivot = null, ToleranceDistribution? distribution = null)
    {
        Kind = kind;
        Surface = surface;
        Minus = minus;
        Plus = plus ?? minus;
        Unit = unit ?? kind.DefaultUnit();
        Param = param;
        Last = kind.IsElement() ? last : null;
        Pivot = kind.IsElement() ? pivot : null;
        Distribution = distribution ?? ToleranceDistribution.Uniform;
    }

    /// <summary>A copy with the same session id (give it a new <see cref="Id"/> to make a duplicate).</summary>
    public ToleranceRow Clone() => (ToleranceRow)MemberwiseClone();

    /// <summary>Equal saved content (session ids aside).</summary>
    public bool SameAs(ToleranceRow other) => Json.DeepEquals(ToJson(), other.ToJson());

    /// <summary>
    /// The curvature change (1/mm) per test fringe of power, as the engine converts fringes
    /// (<c>curvature_per_power_fringe</c>): N fringes are N·λ/2 of departure from the nominal sphere along the normal at
    /// the rim of the clear aperture, Δc = N·λ(1 + cos θ)/(2h²) with sin θ = c·h (N·λ/h² on a flat, ΔR ≈ 4NλR²/D² on a
    /// shallow surface). <c>null</c> when the aperture is not inside the sphere.
    /// </summary>
    public static double? CurvaturePerFringe(double curvature, double semiDiameter, double wavelengthUm)
    {
        var h = semiDiameter;
        var ch2 = curvature * h * (curvature * h);
        if (!(h > 0 && double.IsFinite(h) && ch2 < 1 && wavelengthUm > 0 && double.IsFinite(wavelengthUm))) return null;
        return wavelengthUm * 1e-3 * (1 + Math.Sqrt(1 - ch2)) / (2 * h * h);
    }

    /// <summary>Whether <paramref name="m"/> is a glass (a medium with n_d and V_d).</summary>
    internal static bool HasGlass(MaterialRef m) => m is MaterialRef.ModelGlass or MaterialRef.CatalogGlass;

    /// <summary>
    /// The engine parameter the row perturbs. Index and Abbe address a model glass's n_d and V_d, a catalogue glass's
    /// offset (added to the saved melt offset).
    /// </summary>
    public ParamPath Path(OpticalSystem system)
    {
        var catalogue = system.SurfaceIndexOf(Surface) is { } i && system.Surfaces[i].Material is MaterialRef.CatalogGlass;
        return Kind switch
        {
            ToleranceKind.Radius => new ParamPath.Curvature(Surface),
            ToleranceKind.Thickness => new ParamPath.Thickness(Surface),
            ToleranceKind.DecenterX => new ParamPath.DecenterX(Surface),
            ToleranceKind.DecenterY => new ParamPath.DecenterY(Surface),
            ToleranceKind.TiltX => new ParamPath.TiltX(Surface),
            ToleranceKind.TiltY => new ParamPath.TiltY(Surface),
            ToleranceKind.ElementDecenterX or ToleranceKind.ElementDecenterY or ToleranceKind.ElementTiltX or ToleranceKind.ElementTiltY =>
                new ParamPath.ElementPlacement(Surface, Last ?? Surface, Pivot, Kind.Component() ?? PlacementComponent.DecenterX),
            ToleranceKind.IrregularityAstigmatism => new ParamPath.Irregularity(Surface, IrregularityMode.Astigmatism),
            ToleranceKind.IrregularitySpherical => new ParamPath.Irregularity(Surface, IrregularityMode.Spherical),
            ToleranceKind.Index => catalogue ? new ParamPath.MaterialDnD(Surface) : new ParamPath.ModelNd(Surface),
            ToleranceKind.Abbe => catalogue ? new ParamPath.MaterialDvD(Surface) : new ParamPath.ModelVd(Surface),
            _ => Param ?? new ParamPath.Curvature(Surface),
        };
    }

    /// <summary>Surface indices that move together (an element's range, inclusive).</summary>
    public IReadOnlyList<int> MovingSurfaces(OpticalSystem system)
    {
        if (system.SurfaceIndexOf(Surface) is not { } i) return [];
        if (!Kind.IsElement() || Last is not { } id || system.SurfaceIndexOf(id) is not { } l || l < i) return [i];
        return [.. Enumerable.Range(i, l - i + 1)];
    }

    /// <summary>The pivot's surface index (the first surface by default).</summary>
    public int? PivotIndex(OpticalSystem system) => system.SurfaceIndexOf(Pivot ?? Surface);

    /// <summary><c>S3</c>, <c>S3–4</c> (a thickness between two surfaces, or an element).</summary>
    public string SurfaceLabel(OpticalSystem system)
    {
        if (system.SurfaceIndexOf(Surface) is not { } i) return "S?";
        if (Kind == ToleranceKind.Thickness) return $"S{i}–{i + 1}";
        if (Kind.IsElement())
        {
            var l = Last is { } id && system.SurfaceIndexOf(id) is { } li ? li : i;
            return l == i ? $"S{i}" : $"S{i}–{l}";
        }
        return $"S{i}";
    }

    /// <summary>The element (<c>L2</c>) whose surfaces this row moves, if it is one.</summary>
    public string? ElementTitle(OpticalSystem system)
    {
        var moving = MovingSurfaces(system);
        if (!Kind.IsElement() || moving.Count == 0) return null;
        return LensElements.Groups(system).FirstOrDefault(g => g.Surfaces[0] == moving[0] && g.Surfaces[^1] == moving[^1])?.Title;
    }

    /// <summary>The engine's limits <c>(minus, plus)</c> in parameter units, or why the row cannot be expanded.</summary>
    public ToleranceLimits EngineLimits(ToleranceContext c)
    {
        if (!(double.IsFinite(Minus) && double.IsFinite(Plus) && Minus >= 0 && Plus >= 0)) return ToleranceLimits.Fail("Limits must be zero or positive numbers.");
        if (!(Minus > 0 || Plus > 0)) return ToleranceLimits.Fail("Both limits are zero.");
        var s = c.System;
        if (s.SurfaceIndexOf(Surface) is not { } i) return ToleranceLimits.Fail("The surface no longer exists.");
        var sf = s.Surfaces[i];
        switch (Kind)
        {
            case ToleranceKind.Radius:
                if (s.ValueAt(new ParamPath.Curvature(Surface)) is not { } curv) return ToleranceLimits.Fail($"Surface {i} has no radius.");
                switch (Unit)
                {
                    case ToleranceUnit.Fringes:
                        if (c.SemiDiameter(Surface) is not { } h || !(h > 0))
                            return ToleranceLimits.Fail($"Fringes need the clear aperture of surface {i}; run the layout or set a semi-diameter.");
                        if (CurvaturePerFringe(curv, h, c.TestWavelengthUm) is not { } k)
                            return ToleranceLimits.Fail($"The clear aperture of surface {i} is wider than its sphere.");
                        return ToleranceLimits.Of(Minus * k, Plus * k);
                    case ToleranceUnit.RadiusMM or ToleranceUnit.Percent:
                        if (curv == 0) return ToleranceLimits.Fail($"Surface {i} is flat: give its power in fringes.");
                        var r = 1 / curv;
                        var dm = Unit == ToleranceUnit.Percent ? Math.Abs(r) * Minus / 100 : Minus;
                        var dp = Unit == ToleranceUnit.Percent ? Math.Abs(r) * Plus / 100 : Plus;
                        // R in [R − ΔR−, R + ΔR+]: c in [1/(R + ΔR+), 1/(R − ΔR−)].
                        double lo = r + dp, hi = r - dm;
                        if (lo == 0 || hi == 0 || (lo > 0) != (r > 0) || (hi > 0) != (r > 0))
                            return ToleranceLimits.Fail($"ΔR must stay below |R| = {MeritFormat.Short(Math.Abs(r))} mm.");
                        return ToleranceLimits.Of(curv - 1 / lo, 1 / hi - curv);
                    default:
                        return ToleranceLimits.Fail("Radius tolerances are in fringes, mm or %.");
                }
            case ToleranceKind.Thickness:
                return i < s.ImageIndex ? ToleranceLimits.Of(Minus, Plus) : ToleranceLimits.Fail("The image surface has no thickness.");
            case ToleranceKind.DecenterX or ToleranceKind.DecenterY:
                return i > 0 && i < s.ImageIndex ? ToleranceLimits.Of(Minus, Plus) : ToleranceLimits.Fail("Decenter a surface between the object and the image.");
            case ToleranceKind.TiltX or ToleranceKind.TiltY or ToleranceKind.ElementTiltX or ToleranceKind.ElementTiltY
                or ToleranceKind.ElementDecenterX or ToleranceKind.ElementDecenterY:
                if (Kind.IsElement() && ElementProblem(s, i) is { } why) return ToleranceLimits.Fail(why);
                if (!Kind.IsElement() && (i == 0 || i >= s.ImageIndex)) return ToleranceLimits.Fail("Tilt a surface between the object and the image.");
                var tk = Unit == ToleranceUnit.Arcmin ? 1.0 / 60 : 1;
                return ToleranceLimits.Of(Minus * tk, Plus * tk);
            case ToleranceKind.IrregularityAstigmatism or ToleranceKind.IrregularitySpherical:
                if (!(i > 0 && i < s.ImageIndex && !sf.IsCoordinateBreak && sf.Shape.Kind != ShapeKind.Paraxial))
                    return ToleranceLimits.Fail($"Surface {i} has no form to make irregular.");
                return ToleranceLimits.Of(Minus, Plus);
            case ToleranceKind.Index or ToleranceKind.Abbe:
                if (!HasGlass(sf.Material))
                {
                    var name = sf.Material.IsAir ? "air" : sf.Material.DisplayName;
                    return ToleranceLimits.Fail($"Surface {i} is followed by {name}, not a glass with n_d and V_d.");
                }
                var (lm, lp) = (Minus, Plus);
                if (Kind == ToleranceKind.Abbe && Unit == ToleranceUnit.Percent)
                {
                    if (c.AbbeNumber(Surface) is not { } vd || !(vd > 0))
                        return ToleranceLimits.Fail(
                            $"The Abbe number of {sf.Material.DisplayName} is not known yet: give the limit as ΔV_d, or wait for the glass data.");
                    (lm, lp) = (vd * Minus / 100, vd * Plus / 100);
                }
                if (sf.Material is MaterialRef.CatalogGlass { Offset: var o })
                {
                    // A tolerance adds to the saved melt offset, which stays within the engine's bounds.
                    var (b, cap, what) = Kind == ToleranceKind.Index ? (o.DnD, MaterialOffset.MaxDnD, "Δn_d") : (o.DvD, MaterialOffset.MaxDvD, "ΔV_d");
                    if (Math.Abs(b - lm) > cap + 1e-12 || Math.Abs(b + lp) > cap + 1e-12)
                        return ToleranceLimits.Fail($"With the melt offset ({MeritFormat.Short(b)}), {what} would pass ±{MeritFormat.Short(cap)}, the engine's limit.");
                }
                return ToleranceLimits.Of(lm, lp);
            default:
                if (Param is not { } p || !p.Resolves(s) || s.ValueAt(p) is null) return ToleranceLimits.Fail("The parameter no longer exists.");
                return ToleranceLimits.Of(Minus, Plus);
        }
    }

    /// <summary>Why an element row's range or pivot cannot be moved, if it cannot.</summary>
    string? ElementProblem(OpticalSystem s, int i)
    {
        if (s.SurfaceIndexOf(Last ?? Surface) is not { } l) return "The element's last surface no longer exists.";
        if (!(i > 0 && l >= i && l < s.ImageIndex)) return "An element moves surfaces between the object and the image, first to last.";
        if (Pivot is { } p && s.SurfaceIndexOf(p) is null) return "The pivot surface no longer exists.";
        if (s.Surfaces[i].ElementPlacement is { } e && (e.Last != (Last ?? Surface) || e.Pivot != Pivot))
        {
            var el = s.SurfaceIndexOf(e.Last) is { } a ? a.ToString(CultureInfo.InvariantCulture) : "?";
            var ep = s.SurfaceIndexOf(e.Pivot ?? Surface) is { } b ? b.ToString(CultureInfo.InvariantCulture) : "?";
            return $"The design already moves S{i}–{el} as an element about S{ep}: use that range and pivot.";
        }
        return null;
    }

    /// <summary>The engine's <c>ToleranceOperand</c>, or <c>null</c> when the row cannot be expanded (see <see cref="EngineLimits"/>).</summary>
    public JsonObject? Operand(ToleranceContext c)
    {
        var l = EngineLimits(c);
        if (!l.Ok) return null;
        return new JsonObject
        {
            ["param"] = Path(c.System).ToJson(), ["minus"] = l.Minus, ["plus"] = l.Plus, ["distribution"] = Distribution.ToJson(),
            ["label"] = Label(c.System),
        };
    }

    /// <summary><c>Radius S3</c>, <c>Thickness S3–4</c>, <c>Element tilt x L2</c>, <c>Index S5</c>.</summary>
    public string Label(OpticalSystem system)
    {
        if (Kind == ToleranceKind.Other) return Path(system).Label(system);
        if (Kind.IsElement() && ElementTitle(system) is { } e) return $"{Kind.Title()} {e}";
        return $"{Kind.Title()} {SurfaceLabel(system)}";
    }

    /// <summary>The limits as text: <c>±3 fr</c>, <c>−0.02 / +0.05 mm</c>.</summary>
    public string LimitsText => Minus == Plus
        ? "±" + MeritFormat.Short(Minus) + Unit.Suffix()
        : "−" + MeritFormat.Short(Minus) + " / +" + MeritFormat.Short(Plus) + Unit.Suffix();

    /// <summary>One limit with its unit: <c>+0.8′</c>, <c>−0.02 mm</c>.</summary>
    public string LimitText(bool plus) => (plus ? "+" : "−") + MeritFormat.Short(plus ? Plus : Minus) + Unit.Suffix();

    // ---- Saved form ----

    public JsonObject ToJson()
    {
        var o = new JsonObject
        {
            ["kind"] = Kind.Wire(), ["surface"] = Surface.ToJson(), ["minus"] = Minus, ["plus"] = Plus, ["unit"] = Unit.Wire(),
            ["distribution"] = Distribution.ToJson(),
        };
        if (Param is { } p) o["param"] = p.ToJson();
        if (Last is { } l) o["last"] = l.ToJson();
        if (Pivot is { } v) o["pivot"] = v.ToJson();
        return o;
    }

    internal static ParamPath? TryPath(JsonNode? json)
    {
        if (json is null) return null;
        try { return ParamPath.FromJson(json); }
        catch (Exception e) when (e is JsonException or InvalidOperationException or FormatException) { return null; }
    }

    static SurfaceID? SurfaceIdOf(JsonNode? n) => n.IntOrNull() is { } v and >= 0 ? new SurfaceID((ulong)v) : null;

    public static ToleranceRow? FromJson(JsonNode? json)
    {
        if (json is not JsonObject || ToleranceKinds.Parse(json.At("kind").StringOrNull()) is not { } k || SurfaceIdOf(json.At("surface")) is not { } surface) return null;
        var unit = ToleranceUnits.Parse(json.At("unit").StringOrNull()) is { } u && k.Units().Contains(u) ? u : k.DefaultUnit();
        var row = new ToleranceRow(
            k, surface, json.At("minus").Nf() ?? 0, json.At("plus").Nf() ?? 0, unit, TryPath(json.At("param")),
            SurfaceIdOf(json.At("last")), SurfaceIdOf(json.At("pivot")), ToleranceDistribution.FromJson(json.At("distribution")));
        return k == ToleranceKind.Other && row.Param is null ? null : row;
    }

    /// <summary>
    /// A row from an engine operand written elsewhere (Python, CLI): kinds the app knows in their natural unit
    /// (curvature as ΔR), others as parameters in their own unit.
    /// </summary>
    public static ToleranceRow? FromOperand(JsonNode? o, OpticalSystem system)
    {
        if (TryPath(o.At("param")) is not { } p || p.SurfaceId is not { } id) return null;
        double m = o.At("minus").Nf() ?? 0, pl = o.At("plus").Nf() ?? 0;
        var d = ToleranceDistribution.FromJson(o.At("distribution"));
        ToleranceRow Other() => new(ToleranceKind.Other, id, m, pl, param: p, distribution: d);
        switch (p)
        {
            case ParamPath.Curvature when o.At("unit").StringOrNull() == "fringes":
                return new ToleranceRow(ToleranceKind.Radius, id, m, pl, ToleranceUnit.Fringes, distribution: d);
            case ParamPath.Curvature:
                if (system.ValueAt(p) is not { } c || c == 0) return Other();
                // c in [c − m, c + p] → R in [1/(c + p), 1/(c − m)].
                var r = 1 / c;
                double dm = r - 1 / (c + pl), dp = 1 / (c - m) - r;
                return new ToleranceRow(ToleranceKind.Radius, id, Math.Abs(dm), Math.Abs(dp), ToleranceUnit.RadiusMM, distribution: d);
            case ParamPath.Thickness: return new ToleranceRow(ToleranceKind.Thickness, id, m, pl, distribution: d);
            case ParamPath.DecenterX: return new ToleranceRow(ToleranceKind.DecenterX, id, m, pl, distribution: d);
            case ParamPath.DecenterY: return new ToleranceRow(ToleranceKind.DecenterY, id, m, pl, distribution: d);
            case ParamPath.TiltX: return new ToleranceRow(ToleranceKind.TiltX, id, m * 60, pl * 60, distribution: d);
            case ParamPath.TiltY: return new ToleranceRow(ToleranceKind.TiltY, id, m * 60, pl * 60, distribution: d);
            case ParamPath.ModelNd or ParamPath.MaterialDnD: return new ToleranceRow(ToleranceKind.Index, id, m, pl, distribution: d);
            case ParamPath.ModelVd or ParamPath.MaterialDvD: return new ToleranceRow(ToleranceKind.Abbe, id, m, pl, ToleranceUnit.Abbe, distribution: d);
            case ParamPath.Irregularity ir:
                return new ToleranceRow(
                    ir.Mode == IrregularityMode.Astigmatism ? ToleranceKind.IrregularityAstigmatism : ToleranceKind.IrregularitySpherical, id, m, pl,
                    distribution: d);
            case ParamPath.ElementPlacement e:
                if (ToleranceKinds.Element(e.Component) is not { } k) return Other();
                var tilt = k is ToleranceKind.ElementTiltX or ToleranceKind.ElementTiltY;
                return new ToleranceRow(k, id, tilt ? m * 60 : m, tilt ? pl * 60 : pl, last: e.Last, pivot: e.Pivot, distribution: d);
            default: return Other();
        }
    }

    // ---- Unit conversion (the Mac app's ToleranceReport extension) ----

    /// <summary>
    /// The row with its limits written in <paramref name="unit"/> (the same engine limits): fringes ↔ ΔR ↔ % of R for
    /// radii, arc minutes ↔ degrees, % of V_d ↔ ΔV_d. <c>null</c> when the conversion needs what the design lacks (a
    /// radius of a flat surface, a clear aperture). A unit the kind does not have gives a copy of the row.
    /// </summary>
    public ToleranceRow? ConvertedTo(ToleranceUnit unit, ToleranceContext c)
    {
        var output = Clone();
        if (unit == Unit || !Kind.Units().Contains(unit)) return output;
        output.Unit = unit;
        switch (Kind)
        {
            case ToleranceKind.TiltX or ToleranceKind.TiltY or ToleranceKind.ElementTiltX or ToleranceKind.ElementTiltY:
                var tk = unit == ToleranceUnit.Arcmin ? 60.0 : 1.0 / 60;
                output.Minus = Tidy(Minus * tk);
                output.Plus = Tidy(Plus * tk);
                return output;
            case ToleranceKind.Abbe:
                if (c.AbbeNumber(Surface) is not { } vd || !(vd > 0)) return null;
                var ak = unit == ToleranceUnit.Percent ? 100 / vd : vd / 100;
                output.Minus = Tidy(Minus * ak);
                output.Plus = Tidy(Plus * ak);
                return output;
            case ToleranceKind.Radius:
                var limits = EngineLimits(c);
                if (!limits.Ok || c.System.ValueAt(new ParamPath.Curvature(Surface)) is not { } curv) return null;
                double m = limits.Minus, p = limits.Plus;
                switch (unit)
                {
                    case ToleranceUnit.Fringes:
                        if (c.SemiDiameter(Surface) is not { } h || CurvaturePerFringe(curv, h, c.TestWavelengthUm) is not { } k) return null;
                        output.Minus = m / k;
                        output.Plus = p / k;
                        break;
                    case ToleranceUnit.RadiusMM or ToleranceUnit.Percent:
                        if (curv == 0 || curv + p == 0 || curv - m == 0) return null;
                        var r = 1 / curv;
                        // c in [c − m, c + p]: R in [1/(c + p), 1/(c − m)] (same sign as R).
                        double r1 = 1 / (curv + p), r2 = 1 / (curv - m);
                        if ((r1 > 0) != (r > 0) || (r2 > 0) != (r > 0)) return null;
                        double dm = r - r1, dp = r2 - r;
                        var rk = unit == ToleranceUnit.Percent ? 100 / Math.Abs(r) : 1;
                        output.Minus = Math.Max(dm, 0) * rk;
                        output.Plus = Math.Max(dp, 0) * rk;
                        break;
                    default: return null;
                }
                output.Minus = Tidy(output.Minus);
                output.Plus = Tidy(output.Plus);
                return output;
            default:
                return output;
        }
    }

    /// <summary>Rounds off conversion noise (9 significant digits).</summary>
    internal static double Tidy(double v)
    {
        if (v == 0 || !double.IsFinite(v)) return v;
        var q = Math.Pow(10, Math.Floor(Math.Log10(Math.Abs(v))) - 8);
        return Math.Round(v / q, MidpointRounding.AwayFromZero) * q;
    }
}

// ---- Default tolerances ----

/// <summary>The class of manufacture a <see cref="ToleranceGenerator"/>'s values came from.</summary>
public enum ToleranceGeneratorPreset
{
    [Wire("commercial")] Commercial,
    [Wire("precision")] Precision,
    [Wire("high_precision")] HighPrecision,
}

public static class ToleranceGeneratorPresets
{
    public static readonly IReadOnlyList<ToleranceGeneratorPreset> All = Enum.GetValues<ToleranceGeneratorPreset>();

    public static string Title(this ToleranceGeneratorPreset p) => p switch
    {
        ToleranceGeneratorPreset.Commercial => "Commercial",
        ToleranceGeneratorPreset.Precision => "Precision",
        _ => "High precision",
    };
}

/// <summary>
/// The "Default tolerances" generator: a preset class of manufacture and its values per kind, expanded over the
/// design's elements. A value of 0 leaves its kind out.
/// </summary>
public sealed class ToleranceGenerator
{
    /// <summary>The preset the values came from (<c>null</c>: edited).</summary>
    public ToleranceGeneratorPreset? Preset { get; set; } = ToleranceGeneratorPreset.Precision;
    /// <summary>Power, fringes at the test wavelength.</summary>
    public double RadiusFringes { get; set; } = 3.0;
    public double ThicknessMM { get; set; } = 0.05;
    /// <summary>Surface decentre (one surface, the rest staying in place).</summary>
    public double DecentreMM { get; set; } = 0.025;
    /// <summary>Surface tilt, a wedge.</summary>
    public double TiltArcmin { get; set; } = 1.0;
    /// <summary>Element decentre: the whole element (a cemented group together).</summary>
    public double ElementDecentreMM { get; set; } = 0.03;
    /// <summary>Element tilt about its first vertex.</summary>
    public double ElementTiltArcmin { get; set; } = 1.0;
    public double Index { get; set; } = 0.0005;
    public double AbbePercent { get; set; } = 0.5;
    /// <summary>Surface irregularity, test fringes: astigmatic and spherical, each.</summary>
    public double IrregularityFringes { get; set; } = 0.5;
    public ToleranceDistribution Distribution { get; set; } = ToleranceDistribution.Uniform;

    public ToleranceGenerator() { }

    public ToleranceGenerator(ToleranceGeneratorPreset p) => Apply(p);

    public ToleranceGenerator Clone() => (ToleranceGenerator)MemberwiseClone();

    /// <summary>The values of a preset (typical shop classes; one surface's figures).</summary>
    public void Apply(ToleranceGeneratorPreset p)
    {
        Preset = p;
        switch (p)
        {
            case ToleranceGeneratorPreset.Commercial:
                (RadiusFringes, ThicknessMM, DecentreMM, TiltArcmin, Index, AbbePercent, IrregularityFringes) = (5, 0.2, 0.1, 3, 0.001, 1, 2);
                (ElementDecentreMM, ElementTiltArcmin) = (0.1, 3);
                break;
            case ToleranceGeneratorPreset.Precision:
                (RadiusFringes, ThicknessMM, DecentreMM, TiltArcmin, Index, AbbePercent, IrregularityFringes) = (3, 0.05, 0.025, 1, 0.0005, 0.5, 0.5);
                (ElementDecentreMM, ElementTiltArcmin) = (0.03, 1);
                break;
            default:
                (RadiusFringes, ThicknessMM, DecentreMM, TiltArcmin, Index, AbbePercent, IrregularityFringes) = (1, 0.025, 0.01, 0.5, 0.0002, 0.2, 0.25);
                (ElementDecentreMM, ElementTiltArcmin) = (0.015, 0.5);
                break;
        }
    }

    /// <summary>
    /// The tolerances of <paramref name="system"/>: power and irregularity of every refracting lens surface, every glass
    /// thickness and air space between them (not the back focus), decentre and tilt of every lens surface and of every
    /// element (a cemented group moving as one) in x and y, and the index and Abbe number of every glass.
    /// </summary>
    public List<ToleranceRow> Rows(OpticalSystem system)
    {
        var output = new List<ToleranceRow>();
        var lens = LensElements.LensSurfaces(system);
        if (lens.Count == 0) return output;
        int first = lens[0], last = lens[^1];
        var elements = LensElements.Groups(system);
        void Add(ToleranceKind k, SurfaceID id, double v, SurfaceID? lastId = null)
        {
            if (v > 0) output.Add(new ToleranceRow(k, id, v, last: lastId, distribution: Distribution));
        }
        for (var i = first; i <= last; i++)
        {
            var s = system.Surfaces[i];
            if (s.IsCoordinateBreak) continue;
            var id = s.Id;
            if (lens.Contains(i) && system.ValueAt(new ParamPath.Curvature(id)) is not null)
            {
                Add(ToleranceKind.Radius, id, RadiusFringes);
                if (s.Interaction != Interaction.Dummy)
                {
                    Add(ToleranceKind.IrregularityAstigmatism, id, IrregularityFringes);
                    Add(ToleranceKind.IrregularitySpherical, id, IrregularityFringes);
                }
            }
            // Every gap inside the lens, glass or air (the back focus is the compensator).
            if (i < last) Add(ToleranceKind.Thickness, id, ThicknessMM);
            if (lens.Contains(i))
            {
                foreach (var k in new[] { ToleranceKind.DecenterX, ToleranceKind.DecenterY }) Add(k, id, DecentreMM);
                foreach (var k in new[] { ToleranceKind.TiltX, ToleranceKind.TiltY }) Add(k, id, TiltArcmin);
            }
            if (i < last && ToleranceRow.HasGlass(system.Surfaces[i].Material))
            {
                Add(ToleranceKind.Index, id, Index);
                Add(ToleranceKind.Abbe, id, AbbePercent);
            }
            // The element's own motion after its last surface's rows.
            if (elements.FirstOrDefault(e => e.Surfaces[^1] == i) is { } el && el.Surfaces[0] is var f && f < i)
            {
                var start = system.Surfaces[f].Id;
                foreach (var k in new[] { ToleranceKind.ElementDecenterX, ToleranceKind.ElementDecenterY }) Add(k, start, ElementDecentreMM, id);
                foreach (var k in new[] { ToleranceKind.ElementTiltX, ToleranceKind.ElementTiltY }) Add(k, start, ElementTiltArcmin, id);
            }
        }
        return output;
    }

    public JsonObject ToJson() => new()
    {
        ["preset"] = Preset is { } p ? Wire.Name(p) : null, ["radius_fringes"] = RadiusFringes, ["thickness_mm"] = ThicknessMM,
        ["decentre_mm"] = DecentreMM, ["tilt_arcmin"] = TiltArcmin, ["element_decentre_mm"] = ElementDecentreMM,
        ["element_tilt_arcmin"] = ElementTiltArcmin, ["index"] = Index, ["abbe_percent"] = AbbePercent,
        ["irregularity_fringes"] = IrregularityFringes, ["distribution"] = Distribution.ToJson(),
    };

    public static ToleranceGenerator FromJson(JsonNode? j)
    {
        var g = new ToleranceGenerator();
        if (j is not JsonObject) return g;
        double N(string key, double def) => j.At(key).Nf() ?? def;
        g.Preset = j.At("preset").StringOrNull() is { } name ? Wire.Parse<ToleranceGeneratorPreset>(name) : null;
        g.RadiusFringes = N("radius_fringes", g.RadiusFringes);
        g.ThicknessMM = N("thickness_mm", g.ThicknessMM);
        g.DecentreMM = N("decentre_mm", g.DecentreMM);
        g.TiltArcmin = N("tilt_arcmin", g.TiltArcmin);
        g.ElementDecentreMM = N("element_decentre_mm", g.ElementDecentreMM);
        g.ElementTiltArcmin = N("element_tilt_arcmin", g.ElementTiltArcmin);
        g.Index = N("index", g.Index);
        g.AbbePercent = N("abbe_percent", g.AbbePercent);
        g.IrregularityFringes = N("irregularity_fringes", g.IrregularityFringes);
        g.Distribution = ToleranceDistribution.FromJson(j.At("distribution"));
        return g;
    }
}

// ---- Elements ----

/// <summary>
/// The design's lens elements, for grouping tolerances: runs of surfaces joined by glass (a cemented group is one
/// element), numbered L1, L2, …
/// </summary>
public static class LensElements
{
    /// <param name="Title"><c>L1</c>, or <c>S7</c> for a surface outside any element.</param>
    /// <param name="Surfaces">Surface indices of the group, in order.</param>
    /// <param name="Glasses">Glass names inside the element.</param>
    public sealed record Group(string Title, IReadOnlyList<int> Surfaces, IReadOnlyList<string> Glasses);

    /// <summary>Whether the medium after surface <paramref name="i"/> is a solid (not air or vacuum).</summary>
    static bool IsSolid(OpticalSystem s, int i) => s.Surfaces[i].Material is not (MaterialRef.Air or MaterialRef.Vacuum);

    /// <summary>Surface indices that bound or lie inside a lens element (refracting, with glass on at least one side).</summary>
    public static IReadOnlyList<int> LensSurfaces(OpticalSystem s)
    {
        if (s.ImageIndex <= 1) return [];
        return [.. Enumerable.Range(1, s.ImageIndex - 1).Where(i => !s.Surfaces[i].IsCoordinateBreak && (IsSolid(s, i) || IsSolid(s, i - 1)))];
    }

    /// <summary>The elements, then any other surface a tolerance names.</summary>
    public static IReadOnlyList<Group> Groups(OpticalSystem s)
    {
        var output = new List<Group>();
        var current = new List<int>();
        foreach (var i in LensSurfaces(s))
        {
            current.Add(i);
            if (!IsSolid(s, i))
            {
                output.Add(new Group($"L{output.Count + 1}", current, [.. current.SkipLast(1).Select(k => s.Surfaces[k].Material.DisplayName)]));
                current = [];
            }
        }
        if (current.Count > 0) output.Add(new Group($"L{output.Count + 1}", current, [.. current.Select(k => s.Surfaces[k].Material.DisplayName)]));
        return output;
    }

    /// <summary>The group a row belongs to: its element (an air space goes with the element before it), else the surface itself.</summary>
    public static string GroupTitle(ToleranceRow row, OpticalSystem s, IReadOnlyList<Group> groups)
    {
        if (s.SurfaceIndexOf(row.Surface) is not { } i) return "Other";
        if (groups.FirstOrDefault(g => g.Surfaces.Contains(i)) is { } g)
        {
            // The thickness after an element's last surface is the air space after it.
            if (row.Kind == ToleranceKind.Thickness && g.Surfaces[^1] == i) return $"Air after {g.Title}";
            return g.Title;
        }
        return i == s.Stop ? $"Stop S{i}" : $"S{i}";
    }
}

// ---- Compensators ----

/// <summary>A parameter re-optimised for every trial (the engine's <c>Compensator</c>), within ±<see cref="Range"/> of its nominal value (<c>null</c>: unbounded).</summary>
public sealed class ToleranceCompensator(ParamPath param, double? range = null)
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public ParamPath Param { get; set; } = param;
    public double? Range { get; set; } = range;

    public ToleranceCompensator Clone() => (ToleranceCompensator)MemberwiseClone();

    /// <summary>Equal saved content (session ids aside).</summary>
    public bool SameAs(ToleranceCompensator other) => Json.DeepEquals(ToJson(), other.ToJson());

    /// <summary>The back focus: the thickness before the image surface.</summary>
    public static ToleranceCompensator? BackFocus(OpticalSystem s, double range = 0.5) =>
        s.ImageIndex >= 1 ? new ToleranceCompensator(new ParamPath.Thickness(s.Surfaces[s.ImageIndex - 1].Id), range) : null;

    public static bool IsBackFocus(ParamPath p, OpticalSystem s) =>
        p is ParamPath.Thickness t && s.ImageIndex >= 1 && s.Surfaces[s.ImageIndex - 1].Id == t.Surface;

    /// <summary>"Back focus" or the parameter's label.</summary>
    public string Title(OpticalSystem s) => IsBackFocus(Param, s) ? "Back focus" : Param.Label(s);

    public JsonObject ToJson() => new() { ["param"] = Param.ToJson(), ["range"] = Range is { } r ? r : null };

    public static ToleranceCompensator? FromJson(JsonNode? json)
    {
        if (ToleranceRow.TryPath(json.At("param")) is not { } p) return null;
        return new ToleranceCompensator(p, json.At("range").Nf() is { } r && double.IsFinite(r) ? r : null);
    }

    /// <summary>
    /// Parameters a compensator can take: the back focus, every air space, the design's variables and the decentres,
    /// tilts, thicknesses and curvatures of every surface, in that order, each once.
    /// </summary>
    public static IReadOnlyList<(string Group, IReadOnlyList<ParamPath> Params)> Candidates(OpticalSystem s, IEnumerable<ParamPath> variables)
    {
        var seen = new HashSet<ParamPath>();
        IReadOnlyList<ParamPath> Fresh(IEnumerable<ParamPath> ps) => [.. ps.Where(p => seen.Add(s.Canonical(p)))];
        var bf = BackFocus(s) is { } b ? [b.Param] : Array.Empty<ParamPath>();
        var air = Enumerable.Range(1, Math.Max(Math.Max(s.ImageIndex - 1, 1) - 1, 0)).Where(i => s.Surfaces[i].Material.IsAir)
            .Select(i => (ParamPath)new ParamPath.Thickness(s.Surfaces[i].Id));
        var output = new List<(string, IReadOnlyList<ParamPath>)> { ("Back focus", Fresh(bf)), ("Air spaces", Fresh(air)), ("Design variables", Fresh(variables)) };
        var other = new List<ParamPath>();
        for (var i = 1; i < Math.Max(s.ImageIndex, 1); i++)
        {
            var id = s.Surfaces[i].Id;
            other.AddRange([new ParamPath.DecenterX(id), new ParamPath.DecenterY(id), new ParamPath.TiltX(id), new ParamPath.TiltY(id), new ParamPath.Thickness(id)]);
            if (s.Surfaces[i].Shape.HasCurvature) other.Add(new ParamPath.Curvature(id));
        }
        output.Add(("Surfaces", Fresh(other.Where(p => s.ValueAt(p) is not null))));
        return [.. output.Where(g => g.Item2.Count > 0)];
    }
}

// ---- Criterion ----

/// <summary>What a trial is judged by.</summary>
public enum ToleranceCriterionKind
{
    [Wire("rms_spot")] RmsSpot,
    [Wire("wavefront")] Wavefront,
    [Wire("mtf")] Mtf,
    [Wire("design_merit")] DesignMerit,
}

public static class ToleranceCriterionKinds
{
    public static readonly IReadOnlyList<ToleranceCriterionKind> All = Enum.GetValues<ToleranceCriterionKind>();

    public static string Title(this ToleranceCriterionKind k) => k switch
    {
        ToleranceCriterionKind.RmsSpot => "RMS spot radius",
        ToleranceCriterionKind.Wavefront => "RMS wavefront",
        ToleranceCriterionKind.Mtf => "MTF",
        _ => "Optimizer's merit",
    };
}

/// <summary>
/// What a trial is judged by: the RMS spot, the RMS wavefront or the MTF (the engine's default merit for
/// tolerancing, without thickness boundaries), or the Optimize page's merit function.
/// </summary>
public sealed class ToleranceCriterion
{
    public ToleranceCriterionKind Kind { get; set; } = ToleranceCriterionKind.RmsSpot;

    /// <summary>Spot and wavefront reference, MTF frequencies and method, pupil sampling, fields and wavelengths (the merit editor's generator).</summary>
    public DefaultMeritSettings Defaults { get; set; } = new() { Thickness = false, MtfFrequencies = [20] };

    /// <summary>The pass limit for the yield, in the criterion's display unit (µm, waves, MTF or merit); <c>null</c>: no yield.</summary>
    public double? Threshold { get; set; }

    public ToleranceCriterion Clone() => (ToleranceCriterion)MemberwiseClone();

    /// <summary>The <c>criterion</c> / <c>default_merit</c> settings of the job.</summary>
    public Dictionary<string, JsonNode?> JobSettings(MeritSettings merit)
    {
        if (Kind == ToleranceCriterionKind.DesignMerit)
        {
            return merit.Mode == MeritMode.Custom
                ? new() { ["criterion"] = merit.Spec() }
                : new() { ["default_merit"] = (merit.Defaults with { Thickness = false }).ToJson() };
        }
        var d = Defaults with
        {
            Thickness = false, SqueezeTarget = null,
            Criterion = Kind == ToleranceCriterionKind.Wavefront ? MeritCriterion.Wavefront : Kind == ToleranceCriterionKind.Mtf ? MeritCriterion.Mtf : MeritCriterion.RmsSpot,
        };
        // The MTF shortfall from 1, so the criterion never saturates at 0.
        if (Kind == ToleranceCriterionKind.Mtf) d = d with { MtfTarget = 1 };
        return new() { ["default_merit"] = d.ToJson() };
    }

    /// <summary>How criterion values (merits) read for this kind.</summary>
    public CriterionDisplay Display(MeritSettings merit)
    {
        switch (Kind)
        {
            case ToleranceCriterionKind.RmsSpot: return CriterionDisplay.Spot;
            case ToleranceCriterionKind.Wavefront: return CriterionDisplay.Wavefront;
            case ToleranceCriterionKind.Mtf: return CriterionDisplay.Mtf;
            default:
                if (merit.Mode != MeritMode.Default) return CriterionDisplay.Merit;
                return merit.Defaults.Criterion switch
                {
                    MeritCriterion.RmsSpot => CriterionDisplay.Spot,
                    MeritCriterion.Wavefront => CriterionDisplay.Wavefront,
                    _ => merit.Defaults.MtfTarget >= 1 ? CriterionDisplay.Mtf : CriterionDisplay.Merit,
                };
        }
    }

    public JsonObject ToJson() => new() { ["kind"] = Wire.Name(Kind), ["default_merit"] = Defaults.ToJson(), ["threshold"] = Threshold is { } t ? t : null };

    public static ToleranceCriterion FromJson(JsonNode? j)
    {
        var c = new ToleranceCriterion();
        if (j is not JsonObject) return c;
        c.Kind = j.At("kind").StringOrNull() is { } k ? Wire.Parse<ToleranceCriterionKind>(k) ?? ToleranceCriterionKind.RmsSpot : ToleranceCriterionKind.RmsSpot;
        if (j.At("default_merit") is { } d) c.Defaults = DefaultMeritSettings.FromJson(d);
        c.Threshold = j.At("threshold").Nf() is { } t && double.IsFinite(t) ? t : null;
        return c;
    }
}

/// <summary>
/// Criterion values as a lens designer reads them. The engine's criterion is a merit, the weighted mean of squares:
/// its square root is the RMS spot radius (mm) or wavefront (waves); for the MTF, the root mean square shortfall from
/// 1, so 1 − √c is the mean MTF over fields, directions and frequencies. Lower merits are better in every case.
/// </summary>
public enum CriterionDisplay { Spot, Wavefront, Mtf, Merit }

public static class CriterionDisplays
{
    static readonly CultureInfo C = CultureInfo.InvariantCulture;

    public static string Title(this CriterionDisplay d) => d switch
    {
        CriterionDisplay.Spot => "RMS spot",
        CriterionDisplay.Wavefront => "RMS wavefront",
        CriterionDisplay.Mtf => "MTF",
        _ => "Merit",
    };

    public static string Unit(this CriterionDisplay d) => d switch
    {
        CriterionDisplay.Spot => "µm",
        CriterionDisplay.Wavefront => "waves",
        _ => "",
    };

    /// <summary>Higher shown values are better (MTF).</summary>
    public static bool HigherIsBetter(this CriterionDisplay d) => d == CriterionDisplay.Mtf;

    /// <summary>The shown value of a criterion (merit) <paramref name="c"/>.</summary>
    public static double Value(this CriterionDisplay d, double c)
    {
        if (!double.IsFinite(c)) return double.NaN;
        return d switch
        {
            CriterionDisplay.Spot => Math.Sqrt(c) * 1000,
            CriterionDisplay.Wavefront => Math.Sqrt(c),
            CriterionDisplay.Mtf => 1 - Math.Sqrt(c),
            _ => c,
        };
    }

    /// <summary>The criterion (merit) of a shown value: the yield threshold.</summary>
    public static double Criterion(this CriterionDisplay d, double v) => d switch
    {
        CriterionDisplay.Spot => v / 1000 * (v / 1000),
        CriterionDisplay.Wavefront => v * v,
        CriterionDisplay.Mtf => (1 - v) * (1 - v),
        _ => v,
    };

    public static string Format(this CriterionDisplay d, double v)
    {
        if (!double.IsFinite(v)) return "—";
        var s = d switch
        {
            CriterionDisplay.Spot => v.ToString(v < 10 ? "F2" : "F1", C),
            CriterionDisplay.Wavefront or CriterionDisplay.Mtf => v.ToString("F3", C),
            _ => MeritFormat.Short(v),
        };
        return s.StartsWith('-') ? (s.All(ch => ch is '-' or '0' or '.') ? s[1..] : "−" + s[1..]) : s;
    }

    public static string FormatWithUnit(this CriterionDisplay d, double v) => d.Unit().Length == 0 ? d.Format(v) : d.Format(v) + " " + d.Unit();

    /// <summary>A change in the shown value, signed.</summary>
    public static string FormatChange(this CriterionDisplay d, double change)
    {
        if (!double.IsFinite(change)) return "—";
        var s = d.Format(Math.Abs(change));
        if (s.All(ch => ch is '0' or '.')) return " " + s;
        return (change < 0 ? "−" : "+") + s;
    }
}

// ---- The saved set ----

/// <summary>How a run analyses.</summary>
public enum ToleranceMode
{
    [Wire("sensitivity")] Sensitivity,
    [Wire("monte_carlo")] MonteCarlo,
    [Wire("both")] Both,
}

public static class ToleranceModes
{
    public static readonly IReadOnlyList<ToleranceMode> All = Enum.GetValues<ToleranceMode>();

    public static string Title(this ToleranceMode m) => m switch
    {
        ToleranceMode.Sensitivity => "Sensitivity",
        ToleranceMode.MonteCarlo => "Monte Carlo",
        _ => "Both",
    };
}

/// <summary>
/// The design's tolerancing: rows, compensators, the generator's values, the criterion and the analysis to run. The
/// project's <c>tolerances</c> section holds the engine's <c>ToleranceSpec</c> (<c>schema_version</c>, <c>operands</c>,
/// <c>compensators</c>, <c>criterion</c>, <c>monte_carlo</c>) as last expanded, plus the app's rows under <c>app</c>.
/// A working copy the editor mutates; compare with <see cref="SameAs"/>.
/// </summary>
public sealed class ToleranceSettings
{
    /// <summary>He–Ne, the usual interferometer.</summary>
    public const double DefaultTestWavelengthUm = 0.6328;

    /// <summary>The engine's default levels: the median, 90 % and 97.7 % (2σ one-sided).</summary>
    public static readonly IReadOnlyList<double> DefaultPercentiles = [50, 90, 97.7];

    /// <summary>The engine keeps at most this many trial records.</summary>
    public const int MaxRecordTrials = 10_000;

    public List<ToleranceRow> Rows { get; set; } = [];
    public List<ToleranceCompensator> Compensators { get; set; } = [];
    public ToleranceGenerator Generator { get; set; } = new();
    public ToleranceCriterion Criterion { get; set; } = new();
    public ToleranceMode Mode { get; set; } = ToleranceMode.Both;
    public int Trials { get; set; } = 200;
    public long Seed { get; set; }
    /// <summary>The interferometer wavelength (µm) fringes are counted at: power and irregularity (the engine's <c>irregularity_wavelength_um</c>).</summary>
    public double TestWavelengthUm { get; set; } = DefaultTestWavelengthUm;
    /// <summary>Monte-Carlo levels the engine reports, in percent of the builds.</summary>
    public IReadOnlyList<double> Percentiles { get; set; } = DefaultPercentiles;
    /// <summary>Trials whose perturbations and compensators are kept for the list.</summary>
    public int RecordTrials { get; set; } = 200;

    /// <summary>Whether the set has no tolerances and no compensators.</summary>
    public bool IsEmpty => Rows.Count == 0 && Compensators.Count == 0;

    public ToleranceSettings Clone() => new()
    {
        Rows = [.. Rows.Select(r => r.Clone())], Compensators = [.. Compensators.Select(c => c.Clone())], Generator = Generator.Clone(),
        Criterion = Criterion.Clone(), Mode = Mode, Trials = Trials, Seed = Seed, TestWavelengthUm = TestWavelengthUm, Percentiles = [.. Percentiles],
        RecordTrials = RecordTrials,
    };

    /// <summary>The default set of a design: the precision preset and the back focus as compensator.</summary>
    public static ToleranceSettings Defaults(OpticalSystem s)
    {
        var t = new ToleranceSettings();
        t.Rows = t.Generator.Rows(s);
        t.Compensators = ToleranceCompensator.BackFocus(s) is { } b ? [b] : [];
        return t;
    }

    /// <summary>Reads the saved <c>tolerances</c> section (a spec written elsewhere has its operands turned into rows).</summary>
    public static ToleranceSettings FromSection(JsonNode? section, OpticalSystem system)
    {
        var t = new ToleranceSettings();
        if (section is not JsonObject s) return t;
        var app = s.At("app");
        t.Rows = app.At("rows") is JsonArray rows
            ? [.. rows.Select(ToleranceRow.FromJson).OfType<ToleranceRow>()]
            : [.. (s.At("operands") as JsonArray ?? []).Select(o => ToleranceRow.FromOperand(o, system)).OfType<ToleranceRow>()];
        t.Compensators = [.. (s.At("compensators") as JsonArray ?? []).Select(ToleranceCompensator.FromJson).OfType<ToleranceCompensator>()];
        t.Generator = ToleranceGenerator.FromJson(app.At("generator"));
        t.Criterion = ToleranceCriterion.FromJson(app.At("criterion"));
        t.Mode = app.At("mode").StringOrNull() is { } m ? Wire.Parse<ToleranceMode>(m) ?? ToleranceMode.Both : ToleranceMode.Both;
        t.Trials = s.At("monte_carlo", "trials").IntOrNull() is { } n ? (int)Math.Clamp(n, int.MinValue, int.MaxValue) : t.Trials;
        t.Seed = s.At("monte_carlo", "seed").IntOrNull() ?? t.Seed;
        if (s.At("monte_carlo", "percentiles").NfArray() is { } p && ValidPercentiles(p) is { } v) t.Percentiles = v;
        if (s.At("monte_carlo", "record_trials").IntOrNull() is { } rt) t.RecordTrials = (int)Math.Min(Math.Max(rt, 0), MaxRecordTrials);
        if (app.At("test_wavelength_um").Nf() is { } w && w > 0 && double.IsFinite(w)) t.TestWavelengthUm = w;
        return t;
    }

    /// <summary>Percent levels in [0, 100], sorted and without repeats; <c>null</c> when none is left.</summary>
    public static double[]? ValidPercentiles(IEnumerable<double> v)
    {
        var ok = v.Where(x => double.IsFinite(x) && x >= 0 && x <= 100).Distinct().Order().ToArray();
        return ok.Length == 0 ? null : ok;
    }

    /// <summary><paramref name="c"/> with this set's test wavelength.</summary>
    public ToleranceContext Context(ToleranceContext c) => c with { TestWavelengthUm = TestWavelengthUm };

    /// <summary>Records the engine keeps: at most every trial, at most its cap.</summary>
    public int EffectiveRecordTrials => Math.Min(Math.Min(Math.Max(RecordTrials, 0), Trials), MaxRecordTrials);

    /// <summary>The section: the engine's spec as expanded in <paramref name="context"/> (none: no operands), and the rows.</summary>
    public JsonObject Section(ToleranceContext? context, MeritSettings? merit = null)
    {
        merit ??= new MeritSettings();
        var ops = new JsonArray();
        if (context is not null)
        {
            var c = Context(context);
            foreach (var r in Rows) if (r.Operand(c) is { } o) ops.Add(o);
        }
        var threshold = Criterion.Threshold is { } t ? Criterion.Display(merit).Criterion(t) : (double?)null;
        return new JsonObject
        {
            ["schema_version"] = 1,
            ["operands"] = ops,
            ["compensators"] = new JsonArray([.. Compensators.Select(c => (JsonNode)c.ToJson())]),
            ["criterion"] = Criterion.JobSettings(merit).GetValueOrDefault("criterion"),
            ["monte_carlo"] = new JsonObject
            {
                ["trials"] = Trials, ["seed"] = Seed, ["batch"] = 16, ["threshold"] = threshold,
                ["percentiles"] = new JsonArray([.. Percentiles.Select(p => (JsonNode)p)]), ["record_trials"] = EffectiveRecordTrials,
            },
            ["app"] = new JsonObject
            {
                ["rows"] = new JsonArray([.. Rows.Select(r => (JsonNode)r.ToJson())]), ["generator"] = Generator.ToJson(), ["criterion"] = Criterion.ToJson(),
                ["mode"] = Wire.Name(Mode), ["test_wavelength_um"] = TestWavelengthUm,
            },
        };
    }

    /// <summary>Equal saved content (rows' session ids aside).</summary>
    public bool SameAs(ToleranceSettings other) => Json.DeepEquals(Section(null), other.Section(null));

    /// <summary>
    /// A copy of <paramref name="next"/> whose rows and compensators keep their session ids from <paramref name="old"/>
    /// where they match by content, so a selection survives an undo.
    /// </summary>
    public static ToleranceSettings KeepingIds(ToleranceSettings next, ToleranceSettings old)
    {
        var output = next.Clone();
        // Align the old and new rows (longest common subsequence by content), so duplicates keep their own ids.
        var o = old.Rows;
        var n = next.Rows;
        var lcs = new int[o.Count + 1, n.Count + 1];
        for (var i = o.Count - 1; i >= 0; i--)
            for (var j = n.Count - 1; j >= 0; j--)
                lcs[i, j] = o[i].SameAs(n[j]) ? lcs[i + 1, j + 1] + 1 : Math.Max(lcs[i + 1, j], lcs[i, j + 1]);
        HashSet<int> used = [], matched = [];
        int a = 0, b = 0;
        while (a < o.Count && b < n.Count)
        {
            if (o[a].SameAs(n[b]) && lcs[a, b] == lcs[a + 1, b + 1] + 1)
            {
                output.Rows[b].Id = o[a].Id;
                used.Add(a);
                matched.Add(b);
                a++;
                b++;
            }
            else if (lcs[a, b + 1] >= lcs[a + 1, b]) b++;
            else a++;
        }
        // A row whose limits changed keeps the id of the row of the same kind and surface it replaces.
        for (var k = 0; k < output.Rows.Count; k++)
        {
            if (matched.Contains(k)) continue;
            var m = Enumerable.Range(0, o.Count).FirstOrDefault(i => !used.Contains(i) && o[i].Kind == n[k].Kind && o[i].Surface == n[k].Surface, -1);
            if (m < 0) continue;
            output.Rows[k].Id = o[m].Id;
            used.Add(m);
        }
        var pool = old.Compensators.ToList();
        foreach (var c in output.Compensators)
        {
            var j = pool.FindIndex(p => p.SameAs(c));
            if (j < 0) continue;
            c.Id = pool[j].Id;
            pool.RemoveAt(j);
        }
        return output;
    }

    // ---- Editing ----

    /// <summary>Adds rows after row <paramref name="after"/> (or at the end); returns their ids.</summary>
    public List<Guid> Add(IReadOnlyList<ToleranceRow> add, Guid? after = null)
    {
        var at = after is { } id && Rows.FindIndex(r => r.Id == id) is var k and >= 0 ? k + 1 : Rows.Count;
        Rows.InsertRange(at, add);
        // An element row joins the range and pivot of the rows already moving from its first surface.
        var newIds = add.Select(r => r.Id).ToHashSet();
        foreach (var r in add.Where(r => r.Kind.IsElement()))
        {
            if (Rows.FirstOrDefault(x => !newIds.Contains(x.Id) && x.Kind.IsElement() && x.Surface == r.Surface) is { } other)
            {
                r.Last = other.Last;
                r.Pivot = other.Pivot;
            }
        }
        return [.. add.Select(r => r.Id)];
    }

    public void Delete(IReadOnlySet<Guid> ids) => Rows.RemoveAll(r => ids.Contains(r.Id));

    /// <summary>Copies of the rows, each after its original; returns their ids.</summary>
    public List<Guid> Duplicate(IReadOnlySet<Guid> ids)
    {
        var output = new List<Guid>();
        for (var i = Rows.Count - 1; i >= 0; i--)
        {
            if (!ids.Contains(Rows[i].Id)) continue;
            var copy = Rows[i].Clone();
            copy.Id = Guid.NewGuid();
            Rows.Insert(i + 1, copy);
            output.Add(copy.Id);
        }
        return output;
    }

    /// <summary>Replaces the rows by the generator's set for <paramref name="system"/>.</summary>
    public void Generate(OpticalSystem system) => Rows = Generator.Rows(system);

    /// <summary>
    /// Moves element row <paramref name="id"/> to the range <c>first…last</c> about <paramref name="pivot"/> (<c>null</c>:
    /// the first vertex). Every element row starting on <paramref name="first"/> takes the same range and pivot, as the
    /// engine requires: they move one rigid group.
    /// </summary>
    public void SetElement(Guid id, SurfaceID first, SurfaceID last, SurfaceID? pivot)
    {
        var k = Rows.FindIndex(r => r.Id == id);
        if (k < 0 || !Rows[k].Kind.IsElement()) return;
        Rows[k].Surface = first;
        Rows[k].Last = last;
        Rows[k].Pivot = pivot == first ? null : pivot;
        foreach (var r in Rows.Where(r => r.Kind.IsElement() && r.Surface == first))
        {
            r.Last = Rows[k].Last;
            r.Pivot = Rows[k].Pivot;
        }
    }

    /// <summary>The rows that cannot run, with why.</summary>
    public Dictionary<Guid, ToleranceIssue> Issues(ToleranceContext context)
    {
        var c = Context(context);
        var output = new Dictionary<Guid, ToleranceIssue>();
        foreach (var r in Rows) if (r.EngineLimits(c).Issue is { } e) output[r.Id] = e;
        // Element rows that start on one surface share their range and pivot.
        foreach (var group in Rows.Where(r => r.Kind.IsElement() && !output.ContainsKey(r.Id)).GroupBy(r => r.Surface))
        {
            var first = group.Key;
            if (group.Select(r => (r.Last ?? first, r.Pivot ?? first)).Distinct().Count() <= 1) continue;
            var at = c.System.SurfaceIndexOf(first) is { } i ? $"S{i}" : "one surface";
            foreach (var r in group)
                output[r.Id] = new ToleranceIssue($"Element tolerances starting at {at} move one group: give them the same last surface and pivot.");
        }
        return output;
    }

    // ---- The run ----

    /// <summary>
    /// The job settings in <paramref name="c"/>, with the ids of the rows sent, in the engine's operand order (rows that
    /// cannot run are left out).
    /// </summary>
    public (Dictionary<string, JsonNode?> Settings, List<Guid> RowIds) JobSettings(ToleranceContext c, MeritSettings merit)
    {
        c = Context(c);
        var ops = new JsonArray();
        var ids = new List<Guid>();
        var blocked = Issues(c);
        foreach (var r in Rows)
        {
            if (blocked.ContainsKey(r.Id) || r.Operand(c) is not { } o) continue;
            ops.Add(o);
            ids.Add(r.Id);
        }
        var s = new Dictionary<string, JsonNode?>
        {
            ["tolerances"] = ops, ["compensators"] = new JsonArray([.. Compensators.Select(x => (JsonNode)x.ToJson())]), ["mode"] = Wire.Name(Mode),
            ["trials"] = Trials, ["seed"] = Seed, ["percentiles"] = new JsonArray([.. Percentiles.Select(p => (JsonNode)p)]),
            ["record_trials"] = Mode == ToleranceMode.Sensitivity ? 0 : EffectiveRecordTrials, ["irregularity_wavelength_um"] = TestWavelengthUm,
        };
        foreach (var (k, v) in Criterion.JobSettings(merit)) s[k] = v;
        if (Criterion.Threshold is { } t) s["threshold"] = Criterion.Display(merit).Criterion(t);
        return (s, ids);
    }
}
