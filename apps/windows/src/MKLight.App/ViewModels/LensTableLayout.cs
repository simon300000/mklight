namespace MKLight.App.ViewModels;

/// <summary>A column of the lens grid: its width limits (device-independent pixels).</summary>
public readonly record struct LensGridColumn(LensColumn Column, double Min, double Ideal, double Max = double.PositiveInfinity);

/// <summary>
/// Column widths of the lens grid (the Mac app's <c>LensGridLayout</c>): every row and the header compute
/// the same widths from the available width, starting from the ideal widths and sharing the difference
/// among the columns that can still grow (or shrink).
/// </summary>
public static class LensGridLayout
{
    /// <summary>The columns: the surface label, the type, then the shared parameters; the material widens for melt offsets.</summary>
    public static LensGridColumn[] Columns(bool hasOffsets) =>
    [
        new(LensColumn.Surface, 56, 60, 76),
        new(LensColumn.Type, 104, 124, 160),
        new(LensColumn.Comment, 72, 124),
        new(LensColumn.Radius, 84, 116, 220),
        new(LensColumn.Thickness, 84, 104, 160),
        new(LensColumn.Material, hasOffsets ? 150 : 84, hasOffsets ? 168 : 108, 240),
        new(LensColumn.SemiDiameter, 92, 108, 160),
        new(LensColumn.Conic, 64, 88, 180),
    ];

    public static double MinimumWidth(IReadOnlyList<LensGridColumn> columns) => columns.Sum(c => c.Min);

    public static double[] Widths(IReadOnlyList<LensGridColumn> columns, double total)
    {
        var w = columns.Select(c => c.Ideal).ToArray();
        var diff = total - w.Sum();
        for (var pass = 0; pass <= columns.Count && Math.Abs(diff) > 0.5; pass++)
        {
            var open = Enumerable.Range(0, columns.Count).Where(i => diff > 0 ? w[i] < columns[i].Max : w[i] > columns[i].Min).ToArray();
            if (open.Length == 0) break;
            var share = diff / open.Length;
            foreach (var i in open)
            {
                var n = Math.Clamp(w[i] + share, columns[i].Min, columns[i].Max);
                diff -= n - w[i];
                w[i] = n;
            }
        }
        return w;
    }
}
