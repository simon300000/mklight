using System.Globalization;
using MKLight.Engine;

namespace MKLight.App.ViewModels;

/// <summary>
/// The glass table's and detail column's numbers in the app's one format: invariant culture,
/// a decimal point, no grouping, a typographic minus, never "−0"; "—" for a missing value.
/// </summary>
public static class GlassFormat
{
    public const string None = "—";

    public static string Fixed(double? v, int digits)
    {
        if (v is not double x || !double.IsFinite(x)) return None;
        var s = Math.Abs(x).ToString("F" + digits, CultureInfo.InvariantCulture);
        return x < 0 && s.Any(c => c is >= '1' and <= '9') ? "−" + s : s;
    }

    /// <summary>Always signed (ΔP_g,F): "+0.0012", "−0.0040".</summary>
    public static string Signed(double? v, int digits = 4)
    {
        if (v is not double x || !double.IsFinite(x)) return None;
        var s = Math.Abs(x).ToString("F" + digits, CultureInfo.InvariantCulture);
        return (x < 0 && s.Any(c => c is >= '1' and <= '9') ? "−" : "+") + s;
    }

    /// <summary>Up to <paramref name="digits"/> significant digits (coefficients), with a typographic minus.</summary>
    public static string Significant(double v, int digits = 10) =>
        double.IsFinite(v) ? MKLight.Model.NumberDisplay.Significant(v, digits).Replace('-', '−').Replace("e−", "e-") : None;

    /// <summary>An editable number: shortest round-trip text, a decimal point, no grouping.</summary>
    public static string Editable(double? v) => v is double x && double.IsFinite(x) ? x.ToString("R", CultureInfo.InvariantCulture) : "";

    public static string Nd(GlassRow r) => Fixed(r.Nd, 5);
    public static string Vd(GlassRow r) => Fixed(r.Vd, 2);
    public static string Dpgf(GlassRow r) => Signed(r.Dpgf);
    public static string Density(GlassRow r) => Fixed(r.Density, 2);
    public static string Cost(GlassRow r) => Fixed(r.RelativeCost, 1);
    public static string Tce(GlassRow r) => Fixed(r.Tce, 1);
    public static string Lambda80(GlassRow r) => r.Lambda80Um is { } um ? Fixed(um * 1000, 0) : None;
    public static string Range(GlassRow r) => r.UnrestrictedRange ? "any" : $"{Fixed(r.MinUm, 2)}–{Fixed(r.MaxUm, 2)}";

    /// <summary>The tooltip of an IR glass's n_d / V_d cell: the values its catalog lists.</summary>
    public static string? IrTip(GlassRow r)
    {
        if (r.Nd is not null && r.Vd is not null) return null;
        var listed = r.CatalogNd is { } nd && nd > 0
            ? $"The catalog lists nd {Fixed(nd, 5)}" + (r.CatalogVd is { } vd && vd > 0 ? $", Vd {Fixed(vd, 2)}" : "") + "."
            : "The catalog lists no nd, Vd.";
        return $"The d line (587.6 nm) is outside this glass's range. {listed}";
    }

    /// <summary>One line of a reload report: what changed for a glass.</summary>
    public static string Describe(CatalogChanges.Change c)
    {
        var parts = new List<string>();
        if (!c.NdBefore.Equals(c.NdAfter)) parts.Add($"nd {Fixed(c.NdBefore, 5)} → {Fixed(c.NdAfter, 5)}");
        if (!c.VdBefore.Equals(c.VdAfter)) parts.Add($"Vd {Fixed(c.VdBefore, 3)} → {Fixed(c.VdAfter, 3)}");
        var other = c.Fields.Where(f => f is not ("nd" or "vd")).ToList();
        if (other.Count > 0) parts.Add(string.Join(", ", other));
        return string.Join("; ", parts);
    }

    /// <summary>"Built into MKLight", "Library · loaded into every design", …</summary>
    public static string Origin(CatalogInfo c, bool user)
    {
        if (c.IsBuiltin) return "Built into MKLight";
        if (user) return "Your glasses · saved in the app's library";
        return (c.Library, c.InProject) switch
        {
            (true, true) => "Library · recorded in this design",
            (true, false) => "Library · loaded into every design",
            _ => "This design only",
        };
    }
}
