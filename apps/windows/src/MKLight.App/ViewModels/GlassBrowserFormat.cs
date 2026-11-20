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

    /// <summary>
    /// The tooltip of an IR glass's n_d / V_d cell: why there is no value, and the values its
    /// catalog lists (vendors list IR glasses as n_d 1, V_d 0). Null when both are computed.
    /// </summary>
    public static string? IrTip(GlassRow r)
    {
        if (r.Nd is not null && r.Vd is not null) return null;
        var why = r.Nd is null
            ? "The d line (587.6 nm) is outside this glass's range."
            : "The F (486.1 nm) or C (656.3 nm) line is outside this glass's range.";
        return CatalogListed(r) is { } listed ? $"{why}\nCatalog lists {listed}." : why;
    }

    /// <summary>The n_d, V_d the catalog file lists, shortest form ("nd 1, Vd 0"); null when it lists neither.</summary>
    public static string? CatalogListed(GlassRow r)
    {
        var parts = new List<string>();
        if (r.CatalogNd is { } nd && nd > 0) parts.Add("nd " + Short(nd));
        if (r.CatalogVd is { } vd && (vd > 0 || parts.Count > 0)) parts.Add("Vd " + Short(vd));
        return parts.Count > 0 ? string.Join(", ", parts) : null;
    }

    /// <summary>Up to six significant digits without trailing zeros: "1", "2.4", "64.17".</summary>
    public static string Short(double v) => double.IsFinite(v) ? Significant(v, 6) : None;

    /// <summary>
    /// A caption for the index curve when it does not span the glass's range (the engine
    /// samples it within 0.25–2.5 µm, which an IR glass mostly lies beyond); null when it does.
    /// </summary>
    public static string? CurveNote(GlassDetails d)
    {
        var r = d.Row;
        if (r.UnrestrictedRange) return null;
        var range = $"{Fixed(r.MinUm, 2)}–{Fixed(r.MaxUm, 2)} µm";
        if (d.IndexCurve.Count < 2) return $"No index curve: it is drawn within 0.25–2.5 µm, outside this glass's range ({range}).";
        var (lo, hi) = (d.IndexCurve[0].Um, d.IndexCurve[^1].Um);
        const double eps = 1e-6;
        return lo > r.MinUm + eps || hi < r.MaxUm - eps ? $"Drawn for {Fixed(lo, 2)}–{Fixed(hi, 2)} µm of its {range} range." : null;
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
