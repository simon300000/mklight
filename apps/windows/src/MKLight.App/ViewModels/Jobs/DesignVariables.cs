using System.Globalization;
using System.Text.Json.Nodes;
using MKLight.Model;

namespace MKLight.App.ViewModels.Jobs;

/// <summary>
/// A variable of the design: a parameter the optimiser may change, with optional bounds (the engine's
/// <c>VariableSpec</c>, stored in the project's <c>variables</c> section as <c>[{param, lower, upper}]</c>).
/// </summary>
public sealed record DesignVariable(ParamPath Param, double? Lower = null, double? Upper = null)
{
    /// <summary>Held at one value (equal bounds): kept as a variable but not varied.</summary>
    public bool IsPinned => Lower is not null && Lower == Upper;
    public bool IsBounded => Lower is not null || Upper is not null;

    /// <summary>Whether <paramref name="value"/> violates the bounds.</summary>
    public bool OutOfBounds(double value) => (Lower is { } lo && value < lo - 1e-12) || (Upper is { } hi && value > hi + 1e-12);

    public JsonObject ToJson()
    {
        var o = new JsonObject { ["param"] = Param.ToJson() };
        if (Lower is { } lo) o["lower"] = lo;
        if (Upper is { } hi) o["upper"] = hi;
        return o;
    }

    /// <summary>The smallest glass centre thickness a new glass-thickness variable gets.</summary>
    public const double GlassMinimum = 1.0;

    /// <summary>
    /// Sensible bounds for a new variable: air gaps and the back focus stay non-negative, glass keeps a
    /// minimum centre thickness, model glasses stay in the range of real glasses, melt offsets within the
    /// engine's validation, semi-diameters and normalisation radii positive. Curvatures, conics and
    /// coefficients are left free.
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

/// <summary>
/// The design's variables (the project's <c>variables</c> section) and their undoable edits (the Mac app's
/// <c>DesignDocument</c> variable methods). Variables match by the canonical parameter
/// (<see cref="OpticalSystem.Canonical"/>): <c>curvature</c> and <c>shape_param curvature_y</c> name one
/// parameter of a toroid or biconic.
/// </summary>
public static class DesignVariables
{
    /// <summary>The variables of a <c>variables</c> section, in order; entries that do not read are skipped.</summary>
    public static IReadOnlyList<DesignVariable> Read(JsonTree section)
    {
        var list = new List<DesignVariable>();
        if (section.Node is not JsonArray array) return list;
        foreach (var item in array)
        {
            if (item is not JsonObject o) continue;
            try
            {
                list.Add(new DesignVariable(ParamPath.FromJson(o["param"]), Bound(o["lower"]), Bound(o["upper"])));
            }
            catch (Exception e) when (e is System.Text.Json.JsonException or InvalidOperationException or FormatException or ArgumentException)
            {
                // A variable the model does not read: the engine reports it.
            }
        }
        return list;
    }

    static double? Bound(JsonNode? n) => n is JsonValue v && v.TryGetValue<double>(out var d) && double.IsFinite(d) ? d : null;

    public static JsonTree Encode(IEnumerable<DesignVariable> list) => new(new JsonArray([.. list.Select(v => (JsonNode)v.ToJson())]));

    public static IReadOnlyList<DesignVariable> Of(DesignDocument document) => Read(document.Section("variables"));

    /// <summary>The variable of <paramref name="path"/>, if it is one.</summary>
    public static DesignVariable? Find(DesignDocument document, ParamPath path)
    {
        var p = document.System.Canonical(path);
        return Of(document).FirstOrDefault(v => document.System.Canonical(v.Param) == p);
    }

    public static bool IsVariable(DesignDocument document, ParamPath path) => Find(document, path) is not null;

    /// <summary>
    /// Why <paramref name="path"/> cannot become a variable: a solve sets it, and <c>optimize</c> applies solves
    /// first and refuses solved parameters as variables.
    /// </summary>
    public static string? Refusal(OpticalSystem system, ParamPath path) =>
        system.SolveFor(path) is { } s
            ? $"Set by a solve ({s.Kind.Type.Title()}), so the optimizer cannot vary it. Remove the solve to make it a variable."
            : null;

    /// <summary>
    /// Makes <paramref name="path"/> a variable (with <see cref="DesignVariable.DefaultBounds"/>) or a fixed
    /// parameter, as one undoable step. Returns the refusal when a solve sets it (nothing changes then).
    /// </summary>
    public static string? Set(DesignDocument document, ParamPath path, bool on)
    {
        var system = document.System;
        var list = Of(document).ToList();
        var p = system.Canonical(path);
        if (on)
        {
            if (Refusal(system, path) is { } why) return why;
            if (list.Any(v => system.Canonical(v.Param) == p)) return null;
            var (lo, hi) = DesignVariable.DefaultBounds(path, system);
            list.Add(new DesignVariable(path, lo, hi));
        }
        else
        {
            list.RemoveAll(v => system.Canonical(v.Param) == p);
        }
        document.EditSection("variables", Encode(list), on ? "Make Variable" : "Remove Variable");
        return null;
    }

    /// <summary>Toggles the variable of <paramref name="path"/>; returns the refusal, if any.</summary>
    public static string? Toggle(DesignDocument document, ParamPath path) => Set(document, path, !IsVariable(document, path));

    /// <summary>Sets a variable's bounds (null: unbounded on that side) as one undoable step.</summary>
    public static bool SetBounds(DesignDocument document, ParamPath path, double? lower, double? upper)
    {
        var list = Of(document).ToList();
        var i = list.FindIndex(v => v.Param == path);
        if (i < 0) return false;
        var old = list[i];
        if (old.Lower == lower && old.Upper == upper) return false;
        var action = lower is not null && lower == upper ? "Pin Variable" : old.IsPinned ? "Unpin Variable" : "Change Bounds";
        list[i] = old with { Lower = lower, Upper = upper };
        return document.EditSection("variables", Encode(list), action);
    }

    /// <summary>Pins a variable at its value (equal bounds), or unpins it back to its default bounds.</summary>
    public static bool TogglePin(DesignDocument document, DesignVariable variable)
    {
        if (variable.IsPinned)
        {
            var (lo, hi) = DesignVariable.DefaultBounds(variable.Param, document.System);
            return SetBounds(document, variable.Param, lo, hi);
        }
        return document.System.ValueAt(variable.Param) is { } v && SetBounds(document, variable.Param, v, v);
    }

    /// <summary>
    /// Replaces the whole system (an optimisation, solve or substitution result) as one undoable step; variables of
    /// parameters the new system no longer has go with it, in the same step.
    /// </summary>
    public static bool ReplaceSystem(DesignDocument document, OpticalSystem system, string actionName) =>
        document.EditProject(actionName, p =>
        {
            var section = p.Sections.TryGetValue("variables", out var v) ? v : JsonTree.Null;
            var list = Read(section);
            var kept = Resolving(list, system);
            var sections = kept.Count == list.Count ? p.Sections : p.Sections.Set("variables", Encode(kept));
            return p with { System = system, Sections = sections };
        });

    /// <summary>The variables whose parameters still exist in <paramref name="system"/> (a replaced design keeps only those).</summary>
    public static IReadOnlyList<DesignVariable> Resolving(IEnumerable<DesignVariable> list, OpticalSystem system) =>
        list.Where(v => v.Param.Resolves(system)).ToList();

    /// <summary>
    /// Why a variable cannot be optimised as it is (shown under its row), or null: a solve sets it, an automatic
    /// semi-diameter, a vanished parameter, min above max, a value outside the bounds.
    /// </summary>
    public static string? Problem(DesignVariable v, OpticalSystem system, bool asRadius)
    {
        if (system.IsSolved(v.Param)) return "Set by a solve: not varied";
        if (v.Param is ParamPath.SemiDiameter sd && system.SurfaceIndexOf(sd.Surface) is { } i && system.Surfaces[i].SemiDiameter is SemiDiameter.Auto)
            return "The semi-diameter is automatic: set a value in the lens data first.";
        if (system.ValueAt(v.Param) is not { } value) return "This parameter does not exist on the surface any more.";
        if (v.Lower is { } lo && v.Upper is { } hi && lo > hi) return "The minimum is above the maximum.";
        if (v.OutOfBounds(value))
        {
            var now = asRadius && v.Param.IsCurvature
                ? "R " + BoundsFormat.Short(RadiusDisplay.RadiusFromCurvature(value)) + " mm"
                : BoundsFormat.Short(value) + (v.Param.Unit(system) is { Length: > 0 } u ? " " + u : "");
            return $"Now {now}: outside its bounds";
        }
        return null;
    }
}

/// <summary>How curvature variables show their value and bounds: as a radius in mm (default) or as c = 1/R.</summary>
public enum CurvatureBoundsUnit { Radius, Curvature }

/// <summary>
/// Radius ↔ curvature bounds (the Mac app's <c>RadiusBounds</c>). The engine bounds the curvature c = 1/R; the
/// app lets the designer type a radius window instead, read on the projective line of radii: it rises from min
/// to max and wraps through ±∞ (a flat surface) when min &gt; max, so its curvatures are <c>[1/max, 1/min]</c>.
/// An empty side is open, 0 is the limit R → 0 (no curvature bound on that side), ∞ … ∞ holds the surface flat;
/// a window that contains R = 0 is no interval of curvatures and is refused.
/// </summary>
public static class RadiusBounds
{
    public const string ContainsZero =
        "This radius window passes through R = 0, which is not a range of curvatures. For |R| ≥ r, enter min r and max −r (through ∞).";

    public const string Help =
        "Bounds as a radius window in mm; the optimizer bounds the curvature c = 1/R. Empty is open on that side. Min above max wraps through flat (R = ∞): min 100, max −100 is |R| ≥ 100. 0 leaves the curvature unbounded on that side.";

    /// <summary>Curvature bounds for a typed radius window (null = an empty field); the error when it contains R = 0.</summary>
    public static (double? Lower, double? Upper, string? Error) Curvature(double? minRadius, double? maxRadius)
    {
        if (minRadius is null && maxRadius is null) return (null, null, null);
        var lo = minRadius ?? double.NegativeInfinity;
        var hi = maxRadius ?? double.PositiveInfinity;
        double? upper = lo == 0 ? null : Clean(1 / lo);
        double? lower = hi == 0 ? null : Clean(1 / hi);
        if (lower is { } l && upper is { } u && l > u) return (null, null, ContainsZero);
        return (lower, upper, null);
    }

    /// <summary>The radius window (null = open on that side) that shows curvature bounds; the inverse of <see cref="Curvature"/>.</summary>
    public static (double? Min, double? Max) Radius(double? lower, double? upper)
    {
        if (lower is null && upper is null) return (null, null);
        if (lower == 0 && upper == 0) return (double.PositiveInfinity, double.PositiveInfinity);
        var minR = upper is { } u ? (u == 0 ? double.NegativeInfinity : 1 / u) : 0;
        var maxR = lower is { } l ? (l == 0 ? double.PositiveInfinity : 1 / l) : 0;
        return (double.IsNegativeInfinity(minR) ? null : minR, double.IsPositiveInfinity(maxR) ? null : maxR);
    }

    /// <summary>Whether the window wraps through a flat surface (min &gt; max, both set).</summary>
    public static bool PassesThroughFlat(double? minRadius, double? maxRadius) =>
        minRadius is { } a && maxRadius is { } b && double.IsFinite(a) && double.IsFinite(b) && a > b && a > 0 && b < 0;

    static double Clean(double v) => v == 0 ? 0 : v;
}

/// <summary>Formatting of variable values and bounds (invariant culture, never −0).</summary>
public static class BoundsFormat
{
    static readonly CultureInfo C = CultureInfo.InvariantCulture;

    /// <summary>Compact: 4 significant digits, ∞ for infinities.</summary>
    public static string Compact(double v)
    {
        if (double.IsInfinity(v)) return v > 0 ? "∞" : "−∞";
        if (double.IsNaN(v)) return "—";
        return Minus(NoNegativeZero(v.ToString("G4", C)));
    }

    /// <summary>A short value for captions: up to 6 significant digits.</summary>
    public static string Short(double v)
    {
        if (double.IsInfinity(v)) return v > 0 ? "∞" : "−∞";
        if (double.IsNaN(v)) return "—";
        var a = Math.Abs(v);
        var s = a != 0 && (a < 1e-4 || a >= 1e6) ? v.ToString("0.###e+0", C) : v.ToString("G6", C);
        return Minus(NoNegativeZero(s));
    }

    /// <summary>The exact value for a field with the focus (round-trips).</summary>
    public static string Exact(double v) => double.IsInfinity(v) ? (v > 0 ? "∞" : "-∞") : NumberDisplay.Exact(v);

    /// <summary>A one-line summary of curvature bounds in the other unit: <c>c 0.0167 … 0.025 1/mm</c> or <c>R 40 … 60 mm</c>.</summary>
    public static string? Counterpart(double? lower, double? upper, CurvatureBoundsUnit shownAs)
    {
        if (lower is null && upper is null) return null;
        if (shownAs == CurvatureBoundsUnit.Radius)
            return "c " + (lower is { } l ? Compact(l) : "−∞") + " … " + (upper is { } u ? Compact(u) : "∞") + " 1/mm";
        var (min, max) = RadiusBounds.Radius(lower, upper);
        var wraps = RadiusBounds.PassesThroughFlat(min, max);
        return "R " + (min is { } a ? Compact(a) : "−∞") + (wraps ? " … ∞ … " : " … ") + (max is { } b ? Compact(b) : "∞") + " mm";
    }

    static string NoNegativeZero(string s) => s.StartsWith('-') && s.Skip(1).All(ch => ch is '0' or '.') ? s[1..] : s;
    static string Minus(string s) => s.StartsWith('-') ? "−" + s[1..] : s;
}
