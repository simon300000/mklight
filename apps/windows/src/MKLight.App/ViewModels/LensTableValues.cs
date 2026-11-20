using System.Globalization;
using MKLight.Model;

namespace MKLight.App.ViewModels;

/// <summary>A parsed value, or why the text is not one (shown in a toast).</summary>
public readonly record struct Parsed<T>(bool Ok, T Value, string? Error)
{
    public static Parsed<T> Success(T value) => new(true, value, null);
    public static Parsed<T> Fail(string why) => new(false, default!, why);
    public Parsed<U> Map<U>(Func<T, U> f) => Ok ? Parsed<U>.Success(f(Value)) : Parsed<U>.Fail(Error!);
}

/// <summary>A material edit: the new medium (<c>null</c> keeps the current one) and the surface interaction.</summary>
public sealed record MaterialEdit(MaterialRef? Material, Interaction Interaction);

/// <summary>
/// Parses and validates text typed into the lens table (the Mac app's
/// <c>LensValueParser</c>). Pure functions, invariant culture: a decimal point
/// or a decimal comma, <c>∞</c>/<c>inf</c>, the typographic minus.
/// </summary>
public static class LensValueParser
{
    public static string NotANumber(string text) => $"“{text.Trim()}” is not a number";
    public const string MustBePositive = "Value must be greater than zero";
    public const string MustBeFinite = "Value must be finite";
    public static string InvalidMaterial(string text) =>
        $"“{text.Trim()}” is not a material (glass name, AIR, MIRROR, nd/vd, n=…, CATALOG:NAME)";

    /// <summary>A number (<see cref="NumberDisplay.Parse"/>): a decimal comma, <c>∞</c>, a Unicode minus; NaN refused.</summary>
    public static double? Number(string text) => NumberDisplay.Parse(text);

    /// <summary>Length units accepted after a number, with their size in mm.</summary>
    static readonly (string Unit, double Mm)[] LengthUnits =
        [("mm", 1), ("cm", 10), ("µm", 1e-3), ("um", 1e-3), ("nm", 1e-6), ("in", 25.4), ("\"", 25.4), ("m", 1000)];

    /// <summary>A length in mm: a number, optionally with a unit (<c>mm</c>, <c>cm</c>, <c>m</c>, <c>µm</c>, <c>nm</c>, <c>in</c>, <c>"</c>).</summary>
    public static double? Length(string text)
    {
        var t = text.Trim();
        if (Number(t) is { } v) return v;
        foreach (var (unit, mm) in LengthUnits)
        {
            if (!t.EndsWith(unit, StringComparison.OrdinalIgnoreCase)) continue;
            if (Number(t[..^unit.Length]) is { } n) return n * mm;
        }
        return null;
    }

    /// <summary>Radius text → curvature. <c>∞</c>, <c>inf</c> and <c>0</c> mean a plane (c = 0).</summary>
    public static Parsed<double> CurvatureFromRadius(string text)
    {
        if (Length(text) is not { } r) return Parsed<double>.Fail(NotANumber(text));
        var c = RadiusDisplay.CurvatureFromRadius(r);
        return double.IsFinite(c) ? Parsed<double>.Success(c) : Parsed<double>.Fail(MustBeFinite);
    }

    /// <summary>A thickness; only the object surface may be infinite.</summary>
    public static Parsed<double> Thickness(string text, bool allowInfinite)
    {
        if (Length(text) is not { } t) return Parsed<double>.Fail(NotANumber(text));
        if (double.IsInfinity(t) && !allowInfinite) return Parsed<double>.Fail(MustBeFinite);
        return Parsed<double>.Success(t);
    }

    public static Parsed<double> Finite(string text)
    {
        if (Number(text) is not { } v) return Parsed<double>.Fail(NotANumber(text));
        return double.IsFinite(v) ? Parsed<double>.Success(v) : Parsed<double>.Fail(MustBeFinite);
    }

    /// <summary>A positive, finite length in mm (units accepted).</summary>
    public static Parsed<double> PositiveLength(string text)
    {
        if (Length(text) is not { } v) return Parsed<double>.Fail(NotANumber(text));
        if (!double.IsFinite(v)) return Parsed<double>.Fail(MustBeFinite);
        return v > 0 ? Parsed<double>.Success(v) : Parsed<double>.Fail(MustBePositive);
    }

    /// <summary><c>auto</c> (or empty) for an automatic semi-diameter, else a fixed value &gt; 0.</summary>
    public static Parsed<SemiDiameter> SemiDiameter(string text)
    {
        var t = text.Trim().ToLowerInvariant();
        if (t.Length == 0 || t == "auto" || t == "a" || t.StartsWith("auto ", StringComparison.Ordinal))
            return Parsed<SemiDiameter>.Success(new SemiDiameter.Auto());
        return PositiveLength(text).Map(v => (SemiDiameter)new SemiDiameter.Fixed(v));
    }

    /// <summary>
    /// <c>AIR</c>, <c>VACUUM</c>, <c>MIRROR</c>, <c>DUMMY</c>, <c>nd/vd</c> (model glass), <c>n=1.5</c> or a
    /// bare number (constant index), <c>CATALOG:NAME</c>, or a glass name.
    /// </summary>
    public static Parsed<MaterialEdit> Material(string text)
    {
        var t = text.Trim();
        var u = t.ToUpperInvariant();
        switch (u)
        {
            case "" or "AIR": return Parsed<MaterialEdit>.Success(new(new MaterialRef.Air(), Interaction.Refract));
            case "VACUUM": return Parsed<MaterialEdit>.Success(new(new MaterialRef.Vacuum(), Interaction.Refract));
            case "MIRROR": return Parsed<MaterialEdit>.Success(new(null, Interaction.Reflect));
            case "DUMMY": return Parsed<MaterialEdit>.Success(new(null, Interaction.Dummy));
        }
        var fail = Parsed<MaterialEdit>.Fail(InvalidMaterial(t));
        var parts = t.Split('/');
        if (parts.Length == 2)
        {
            if (Number(parts[0]) is not { } nd || Number(parts[1]) is not { } vd || !(nd >= 1) || !double.IsFinite(nd) || !(vd > 0) || !double.IsFinite(vd))
                return fail;
            return Parsed<MaterialEdit>.Success(new(new MaterialRef.ModelGlass(nd, vd, 0), Interaction.Refract));
        }
        var constant = u.StartsWith("N=", StringComparison.Ordinal) ? u[2..] : u.StartsWith('=') ? u[1..] : u;
        if (constant.Length > 0 && (char.IsAsciiDigit(constant[0]) || constant[0] == '.') && Number(constant) is { } n)
        {
            if (!(n > 0) || !double.IsFinite(n)) return fail;
            return Parsed<MaterialEdit>.Success(new(new MaterialRef.ConstantIndex(n), Interaction.Refract));
        }
        var colon = t.IndexOf(':');
        if (colon >= 0)
        {
            var catalog = t[..colon].Trim();
            var name = t[(colon + 1)..].Trim();
            if (catalog.Length == 0 || name.Length == 0) return fail;
            return Parsed<MaterialEdit>.Success(new(
                new MaterialRef.CatalogGlass { Catalog = catalog.ToUpperInvariant(), Name = name.ToUpperInvariant() }, Interaction.Refract));
        }
        if (!t.All(ch => char.IsLetterOrDigit(ch) || "-_.+".Contains(ch))) return fail;
        return Parsed<MaterialEdit>.Success(new(MaterialRef.Glass(u), Interaction.Refract));
    }

    /// <summary>
    /// <paramref name="current"/> replaced by <paramref name="next"/>: retyping the same catalogue glass keeps
    /// its melt offset; another glass starts without.
    /// </summary>
    public static MaterialRef Replace(MaterialRef current, MaterialRef next)
    {
        if (current is not MaterialRef.CatalogGlass { Offset.IsZero: false } old || next is not MaterialRef.CatalogGlass { Offset.IsZero: true } typed)
            return next;
        if (!string.Equals(old.Name, typed.Name, StringComparison.OrdinalIgnoreCase)) return next;
        if (typed.Catalog is not null && old.Catalog is not null && !string.Equals(typed.Catalog, old.Catalog, StringComparison.OrdinalIgnoreCase)) return next;
        return typed with { Catalog = typed.Catalog ?? old.Catalog, Offset = old.Offset };
    }
}

/// <summary>Number formats of the lens table: invariant, the typographic minus, never <c>−0.0000</c>.</summary>
public static class LensFormat
{
    static readonly CultureInfo Inv = CultureInfo.InvariantCulture;
    public const char Minus = '−';

    /// <summary>Fixed decimals (radius, thickness 4; semi-diameter 3), scientific outside 1e-4 … 1e7, <c>∞</c>.</summary>
    public static string Length(double v, int decimals = 4)
    {
        if (double.IsNaN(v)) return "—";
        if (double.IsInfinity(v)) return v > 0 ? "∞" : Minus + "∞";
        var a = Math.Abs(v);
        if (a != 0 && (a < 1e-4 || a >= 1e7)) return Signed(v.ToString("0.0000e+00", Inv));
        return Signed(v.ToString("F" + decimals, Inv));
    }

    /// <summary>Up to 6 significant digits without trailing zeros (the conic, and the two sections that share an anamorphic cell).</summary>
    public static string Plain(double v)
    {
        if (double.IsNaN(v)) return "—";
        if (double.IsInfinity(v)) return v > 0 ? "∞" : Minus + "∞";
        if (v == 0) return "0";
        var a = Math.Abs(v);
        if (a < 1e-3 || a >= 1e6) return Signed(v.ToString("0.0000e+00", Inv));
        return Signed(NumberDisplay.Significant(v, 6));
    }

    /// <summary>A signed number with a true minus and no trailing zeros; <c>0</c> for zero (melt offsets: <c>+0.0012/−0.8</c>).</summary>
    public static string SignedTrimmed(double v, int decimals)
    {
        if (v == 0) return "0";
        var s = Math.Abs(v).ToString("F" + decimals, Inv);
        if (s.Contains('.')) s = s.TrimEnd('0').TrimEnd('.');
        return (v < 0 ? Minus : '+') + s;
    }

    /// <summary>The melt offset after a catalogue glass, <c>+0.0012/−0.8</c>.</summary>
    public static string Offset(MaterialOffset o) => o.IsZero ? "" : SignedTrimmed(o.DnD, 4) + "/" + SignedTrimmed(o.DvD, 2);

    /// <summary>The typographic minus, and no sign on a value that rounds to zero.</summary>
    static string Signed(string s)
    {
        if (!s.StartsWith('-')) return s;
        var mantissa = s.Split('e')[0];
        return mantissa.All(c => c is '-' or '0' or '.') ? s[1..] : Minus + s[1..];
    }
}
