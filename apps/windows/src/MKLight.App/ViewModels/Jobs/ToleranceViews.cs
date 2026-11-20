using System.Globalization;
using MKLight.Model;

namespace MKLight.App.ViewModels.Jobs;

// The pure parts of the Mac app's tolerance views: the histogram's bars and their trials, a compensator's range used
// against the range allowed, the tolerances table's groups, and an element row's moving surfaces and pivot.

/// <summary>The Monte-Carlo histogram's bars (the Mac app's <c>MonteCarloChart.bins</c>) over the shown values.</summary>
public static class ToleranceHistogram
{
    /// <summary>A bar: its centre, the trials in it and the shown values it spans (inclusive).</summary>
    public readonly record struct Bin(double Center, int Count, double Low, double High)
    {
        public bool Contains(double v) => v >= Low && v <= High;
    }

    /// <summary>10, 16 or 24 bars by the number of trials.</summary>
    public static int BarCount(int trials) => trials >= 200 ? 24 : trials >= 60 ? 16 : 10;

    /// <summary>The bars over the result's finite shown values; one bar when every trial has the same value.</summary>
    public static IReadOnlyList<Bin> Bins(ToleranceResult r)
    {
        var shown = r.Criteria.Select(x => r.Display.Value(x)).Where(double.IsFinite).ToList();
        if (shown.Count == 0) return [];
        var lo = shown.Min();
        var hi = shown.Max();
        if (!(hi > lo)) return [new Bin(lo, shown.Count, lo, lo)];
        var n = BarCount(shown.Count);
        var w = (hi - lo) / n;
        var counts = new int[n];
        foreach (var x in shown) counts[Math.Min((int)((x - lo) / w), n - 1)]++;
        // Each bar spans its own edges; the last one ends on the largest value exactly.
        return [.. Enumerable.Range(0, n).Select(k => new Bin(lo + (k + 0.5) * w, counts[k], lo + k * w, k == n - 1 ? hi : lo + (k + 1) * w))];
    }

    /// <summary>The bar that counts shown value <paramref name="v"/> (a value on a shared edge counts in the bar above it).</summary>
    public static int? IndexOf(IReadOnlyList<Bin> bins, double v)
    {
        if (bins.Count == 0 || !double.IsFinite(v) || v < bins[0].Low || v > bins[^1].High) return null;
        if (bins.Count == 1) return 0;
        var w = (bins[^1].High - bins[0].Low) / bins.Count;
        return Math.Clamp((int)((v - bins[0].Low) / w), 0, bins.Count - 1);
    }

    /// <summary>The bar holding trial <paramref name="trial"/>, if it completed.</summary>
    public static int? BinOfTrial(ToleranceResult r, IReadOnlyList<Bin> bins, int trial) =>
        trial >= 0 && trial < r.TrialCriteria.Count && double.IsFinite(r.Display.Value(r.TrialCriteria[trial])) ? IndexOf(bins, r.Display.Value(r.TrialCriteria[trial])) : null;

    /// <summary>The trials a bar counts, worst first.</summary>
    public static List<int> Trials(ToleranceResult r, IReadOnlyList<Bin> bins, int bin) =>
        bin < 0 || bin >= bins.Count ? [] : [.. r.TrialsWorstFirst.Where(k => BinOfTrial(r, bins, k) == bin)];

    /// <summary><c>12 trials at 19.4–19.8 µm</c>.</summary>
    public static string Title(CriterionDisplay d, Bin b, int count) =>
        $"{count.ToString(CultureInfo.InvariantCulture)} trial{(count == 1 ? "" : "s")} at {d.Format(b.Low)}–{d.Format(b.High)}{(d.Unit().Length == 0 ? "" : " " + d.Unit())}";
}

/// <summary>The range a compensator used in the sensitivity evaluations against the range allowed (the Mac app's <c>CompensatorStats</c>).</summary>
/// <param name="Low">Lowest move from nominal (≤ 0).</param>
/// <param name="High">Highest move from nominal (≥ 0).</param>
/// <param name="Range">The ± range allowed, if bounded.</param>
public readonly record struct ToleranceCompensatorUsage(double Low, double High, double? Range)
{
    /// <summary>Compensator <paramref name="j"/> of the result; <c>null</c> without sensitivity evaluations.</summary>
    public static ToleranceCompensatorUsage? Of(ToleranceResult r, int j) =>
        r.CompensatorRange(j) is { } u && j < r.Compensators.Count ? new ToleranceCompensatorUsage(u.Low, u.High, r.Compensators[j].Range) : null;

    /// <summary>The used span as fractions of the allowed range, −1 … 1 (centred on nominal); <c>null</c> when unbounded.</summary>
    public (double From, double To)? Fractions => Range is > 0 and { } g ? (Math.Max(Low / g, -1), Math.Min(High / g, 1)) : null;

    /// <summary>The largest move reached the limit (99 % of the range): some builds need more adjustment than allowed.</summary>
    public bool AtLimit => Range is > 0 and { } g && Math.Max(-Low, High) >= g * 0.99;

    /// <summary>A trial's compensator move from nominal with three decimals: <c>+0.170</c>, <c>−0.042</c> (never <c>−0.000</c>).</summary>
    public static string Move(double change) => Signed(change, 3);

    /// <summary>A move with four decimals, <c>−0.1260</c>, <c>+0.0010</c> (never <c>−0.0000</c>).</summary>
    public static string Signed4(double change) => Signed(change, 4);

    static string Signed(double v, int decimals)
    {
        if (!double.IsFinite(v)) return "—";
        var text = Math.Abs(v).ToString("F" + decimals.ToString(CultureInfo.InvariantCulture), CultureInfo.InvariantCulture);
        return (v < 0 && text.Any(ch => ch is >= '1' and <= '9') ? "−" : "+") + text;
    }
}

/// <summary>The tolerances table's groups (the Mac app's <c>ToleranceTableSection.groups</c>).</summary>
/// <param name="Key"><c>L1</c>, <c>Air after L1</c>, <c>Stop S6</c>: also what collapses.</param>
/// <param name="Title">The key with the element's glasses, <c>L1 · N-SK16</c>.</param>
public sealed record ToleranceTableGroup(string Key, string Title, IReadOnlyList<ToleranceRow> Rows)
{
    /// <summary>Rows by element in surface order (an air space after its element), rows in their order within a group.</summary>
    public static List<ToleranceTableGroup> Of(IReadOnlyList<ToleranceRow> rows, OpticalSystem system)
    {
        var lens = LensElements.Groups(system);
        var order = new List<string>();
        var byKey = new Dictionary<string, List<ToleranceRow>>();
        var sortKey = new Dictionary<string, (int, int)>();
        foreach (var r in rows)
        {
            var t = LensElements.GroupTitle(r, system, lens);
            if (!byKey.TryGetValue(t, out var list)) { order.Add(t); byKey[t] = list = []; }
            list.Add(r);
            var k = (system.SurfaceIndexOf(r.Surface) ?? int.MaxValue, t.StartsWith("Air", StringComparison.Ordinal) ? 1 : 0);
            sortKey[t] = sortKey.TryGetValue(t, out var old) && old.CompareTo(k) <= 0 ? old : k;
        }
        return [.. order.OrderBy(t => sortKey[t]).Select(t =>
        {
            var glasses = lens.FirstOrDefault(g => g.Title == t) is { Glasses.Count: > 0 } g ? " · " + string.Join(", ", g.Glasses) : "";
            return new ToleranceTableGroup(t, t + glasses, byKey[t]);
        })];
    }

    /// <summary>The first element row of each first surface: rows starting on one surface share range and pivot, so one line shows them.</summary>
    public static HashSet<Guid> LeadingElementRows(IEnumerable<ToleranceRow> rows)
    {
        var seen = new HashSet<SurfaceID>();
        return [.. rows.Where(r => r.Kind.IsElement() && seen.Add(r.Surface)).Select(r => r.Id)];
    }
}

/// <summary>Where an element's pivot sits relative to its range, kept when the range moves.</summary>
public enum TolerancePivotPlace { FirstVertex, LastVertex, Surface }

/// <summary>An element row's range and pivot as the table shows and edits them (the Mac app's <c>ToleranceRowView</c> element line).</summary>
public readonly record struct TolerancePivotChoice(TolerancePivotPlace Place, int Surface = 0)
{
    public static readonly TolerancePivotChoice First = new(TolerancePivotPlace.FirstVertex);
    public static readonly TolerancePivotChoice Last = new(TolerancePivotPlace.LastVertex);

    /// <summary>The row's pivot relative to its moving surfaces.</summary>
    public static TolerancePivotChoice Of(ToleranceRow row, OpticalSystem system)
    {
        var moving = row.MovingSurfaces(system);
        if (moving.Count == 0 || row.PivotIndex(system) is not { } p || p == moving[0]) return First;
        return p == moving[^1] ? Last : new TolerancePivotChoice(TolerancePivotPlace.Surface, p);
    }

    /// <summary>The pivot's surface index for the range <c>first…last</c>.</summary>
    public int Index(int first, int last) => Place switch
    {
        TolerancePivotPlace.LastVertex => last,
        TolerancePivotPlace.Surface => Surface,
        _ => first,
    };

    /// <summary>
    /// Moves element row <paramref name="id"/> to surfaces <paramref name="first"/>…<paramref name="last"/> about this
    /// pivot (every element row starting on the same surface follows). False when an index is out of the system.
    /// </summary>
    public bool Apply(ToleranceSettings s, Guid id, OpticalSystem system, int first, int last)
    {
        var n = system.Surfaces.Count;
        var p = Index(first, last);
        if (first < 0 || first >= n || last < first || last >= n || p < 0 || p >= n) return false;
        s.SetElement(id, system.Surfaces[first].Id, system.Surfaces[last].Id, Place == TolerancePivotPlace.FirstVertex ? null : system.Surfaces[p].Id);
        return true;
    }

    /// <summary><c>S3, first vertex</c>, <c>S5, last vertex</c>, <c>S6</c>.</summary>
    public static string Text(int? pivot, IReadOnlyList<int> moving)
    {
        if (pivot is not { } p) return "?";
        var s = "S" + p.ToString(CultureInfo.InvariantCulture);
        if (moving.Count > 0 && p == moving[0]) return s + ", first vertex";
        if (moving.Count > 0 && p == moving[^1]) return s + ", last vertex";
        return s;
    }

    /// <summary><c>S3 S4 S5 move together</c>, <c>S3 moves</c>.</summary>
    public static string MovingText(IReadOnlyList<int> moving) =>
        string.Join(" ", moving.Select(i => "S" + i.ToString(CultureInfo.InvariantCulture))) + (moving.Count > 1 ? " move together" : " moves");
}

/// <summary>Names of the exported files.</summary>
public static class ToleranceExport
{
    /// <summary><c>Double Gauss sensitivity.csv</c>, <c>Double Gauss trials.csv</c>; characters a file name cannot hold become <c>-</c>.</summary>
    public static string FileName(string systemName, string what)
    {
        var name = systemName.Trim().Length == 0 ? "Tolerance" : systemName.Trim();
        var bad = Path.GetInvalidFileNameChars();
        return new string([.. name.Select(ch => bad.Contains(ch) ? '-' : ch)]) + " " + what + ".csv";
    }
}
