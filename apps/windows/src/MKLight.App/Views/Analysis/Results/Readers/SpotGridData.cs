using System.Globalization;
using System.Text.Json.Nodes;

namespace MKLight.App.Views.Analysis.Results.Readers;

/// <summary>The box scale of the spot grid.</summary>
public enum SpotScale
{
    /// <summary>One half-width for every cell.</summary>
    Grid,
    /// <summary>One half-width per field (row).</summary>
    Field,
}

/// <summary>
/// The spot diagram result as a field × focus grid (docs/APP.md "Spot
/// diagram"): one row per field, one column per focus plane. The engine
/// returns the points once, at the base defocus, with a <c>point_slope</c>
/// buffer; every cell's points are placed here, once, as
/// <c>points + (d − defocus) · slope</c>.
/// </summary>
public sealed class SpotGridData
{
    /// <summary>The points of one wavelength in one cell, relative to the cell's reference (mm), x and y interleaved.</summary>
    public sealed record Group(int Wavelength, double[] Xy)
    {
        public int Count => Xy.Length / 2;
    }

    /// <summary>
    /// A dense cell's points of one wavelength as a coverage mask: <see cref="MaskSize"/>² bytes over ±<see cref="Cell.Extent"/>
    /// (row 0 at +y), 1 where a point falls. Drawn as one layer like the markers, it looks the same at dense counts.
    /// </summary>
    public sealed record Coverage(int Wavelength, byte[] Mask);

    /// <summary>
    /// One (field, plane) cell: its points by wavelength (<see cref="Groups"/>), or for a dense grid (more than
    /// <see cref="DensePoints"/> points per field) their <see cref="Masks"/> instead.
    /// </summary>
    public sealed record Cell(int Column, double Defocus, IReadOnlyList<Group> Groups, int PointCount, double Rms, double Geo, double Extent,
        IReadOnlyList<Coverage>? Masks = null);

    /// <summary>Points per field above which cells keep coverage masks instead of the points (fine pupil grids).</summary>
    public const int DensePoints = 20_000;

    /// <summary>The side of a dense cell's coverage mask: about one device pixel per bin at the largest cell.</summary>
    public const int MaskSize = 512;

    /// <summary>One field.</summary>
    public sealed record Row(int Index, string Title, string Value, IReadOnlyList<Cell> Cells)
    {
        public string Label => Value.Length == 0 ? Title : Title + " · " + Value;
    }

    /// <summary>Columns drawn at most; more sampled planes are thinned evenly.</summary>
    public const int MaxColumns = 9;

    /// <summary>Defocus of each column (mm, absolute from the image surface).</summary>
    public IReadOnlyList<double> Planes { get; }
    /// <summary>Planes the engine sampled (before thinning).</summary>
    public int SampledPlanes { get; }
    /// <summary>The column at the base defocus, if one is.</summary>
    public int? BaseColumn { get; }
    public IReadOnlyList<Row> Rows { get; }

    /// <summary>
    /// The grid's one length unit: µm unless a cell's RMS or GEO radius reaches 1 mm, so the captions of one grid
    /// never mix "25.92 µm" with "0.2007 mm".
    /// </summary>
    public bool Micrometres => _micrometres ??= Rows.SelectMany(r => r.Cells).SelectMany(c => new[] { c.Rms, c.Geo })
        .Where(double.IsFinite).Select(Math.Abs).DefaultIfEmpty(0).Max() < 1;
    bool? _micrometres;

    /// <summary>A length of this grid in its unit (<see cref="Micrometres"/>).</summary>
    public string Length(double mm) => Fmt.Length(mm, Micrometres);
    public double Airy { get; }
    public IReadOnlyList<int> Wavelengths { get; }
    public IReadOnlyList<double> WavelengthsUm { get; }
    /// <summary>"chief" or "centroid".</summary>
    public string Reference { get; }

    SpotGridData(IReadOnlyList<double> planes, int sampled, int? baseColumn, IReadOnlyList<Row> rows, double airy,
        IReadOnlyList<int> wavelengths, IReadOnlyList<double> um, string reference)
    {
        Planes = planes; SampledPlanes = sampled; BaseColumn = baseColumn; Rows = rows; Airy = airy;
        Wavelengths = wavelengths; WavelengthsUm = um; Reference = reference;
    }

    /// <summary>
    /// The base plane's column alone (for a pane too narrow for every focus plane), or this grid when it has
    /// one column or no base plane.
    /// </summary>
    public SpotGridData BaseOnly() => _baseOnly ??= MakeBaseOnly();
    SpotGridData? _baseOnly;

    SpotGridData MakeBaseOnly()
    {
        if (Planes.Count <= 1 || BaseColumn is not int b) return this;
        var rows = Rows.Select(r => r with { Cells = r.Cells.Where(c => c.Column == b).Select(c => c with { Column = 0 }).ToList() }).ToList();
        return new SpotGridData([Planes[b]], SampledPlanes, 0, rows, Airy, Wavelengths, WavelengthsUm, Reference);
    }

    /// <summary>Reads a spot diagram result; null when <c>fields</c> is missing or the buffers are malformed.</summary>
    public static SpotGridData? Read(ResultData d)
    {
        var j = d.Json;
        if (Json.Arr(Json.At(j, "fields")) is not { Count: > 0 } list) return null;
        var pts = d.Buffer("points");
        var pw = d.Buffer("point_wavelength");
        var slope = d.Buffer("point_slope");
        if (pts is not null && pts.Count % 2 != 0) return null;
        var um = LineReaders.WavelengthsUm(d);
        var wavelengths = Json.IntArray(Json.At(j, "wavelengths")) ?? Enumerable.Range(0, um.Length).ToArray();
        var airy = Json.Nf(Json.At(j, "airy_radius"));
        var baseDefocus = Json.Number(Json.At(j, "defocus")) ?? 0;
        var reference = ReferenceOf(list);
        var rmsKey = reference == "centroid" ? "rms_radius_centroid" : "rms_radius_chief";

        // Planes from the first field's through-focus list; the base plane
        // alone without one (or without slopes to move the points).
        var sampled = (Json.Arr(Json.At(list[0], "through_focus")) ?? []).Select(e => Json.Number(Json.At(e, "defocus")))
            .Where(v => v is not null).Select(v => v!.Value).ToArray();
        var hasFocus = slope is not null && sampled.Length > 0;
        var all = hasFocus ? sampled : [baseDefocus];
        var picked = Thin(all.Length, MaxColumns, Nearest(all, baseDefocus));
        var planes = picked.Select(i => all[i]).ToArray();
        int? baseIndex = null;
        for (var c = 0; c < planes.Length; c++)
            if (Math.Abs(planes[c] - baseDefocus) <= 1e-12 * Math.Max(1, Math.Abs(baseDefocus))) { baseIndex = c; break; }

        // The buffers as doubles in pooled arrays (reused between runs: no fresh pages for about 80 MB at 250 rings).
        var pool = System.Buffers.ArrayPool<double>.Shared;
        double[] Rent(IResultArray? a)
        {
            if (a is null) return [];
            var r = pool.Rent(a.Count);
            a.CopyTo(r);
            return r;
        }
        var xy = Rent(pts);
        var sl = Rent(slope);
        var wl = Rent(pw);
        int nXy = pts?.Count ?? 0, nSl = slope?.Count ?? 0, nWl = pw?.Count ?? 0;
        try
        {
            var nPoints = nXy / 2;
            var rows = new Row[list.Count];
            // A dense grid builds its fields (and their planes) in parallel: the result JSON is materialised, so read-only access is safe.
            void FieldRow(int k)
            {
                var f = list[k];
                var index = Json.Int(Json.At(f, "field")) ?? k;
                var entries = Json.Arr(Json.At(f, "through_focus"));
                int lo = 0, hi = 0;
                if (pts is not null && Json.IntArray(Json.At(f, "point_range")) is { Length: 2 } r)
                {
                    lo = Math.Max(r[0], 0);
                    hi = (int)Math.Min((long)r[0] + r[1], nPoints);
                }
                // The field's wavelength groups, in order of first appearance, and each point's group (same on every plane).
                var groupOf = new int[Math.Max(hi - lo, 0)];
                var order = new List<int>();
                var sizes = new List<int>();
                for (var i = lo; i < hi; i++)
                {
                    var w = i < nWl && double.IsFinite(wl[i]) ? (int)wl[i] : 0;
                    var g = order.IndexOf(w);
                    if (g < 0) { g = order.Count; order.Add(w); sizes.Add(0); }
                    groupOf[i - lo] = g;
                    sizes[g]++;
                }
                var count = Math.Max(hi - lo, 0);
                var dense = count > DensePoints;
                // Each plane's points (or masks) and extent; the planes of a dense field in parallel.
                var built = new (IReadOnlyList<Group> Groups, IReadOnlyList<Coverage>? Masks, double Extent)[planes.Length];
                void Plane(int c)
                {
                    var delta = planes[c] - baseDefocus;
                    var moves = delta != 0 && nSl > 0;
                    double X(int i) => moves && i * 2 + 1 < nSl ? xy[i * 2] + delta * sl[i * 2] : xy[i * 2];
                    double Y(int i) => moves && i * 2 + 1 < nSl ? xy[i * 2 + 1] + delta * sl[i * 2 + 1] : xy[i * 2 + 1];
                    var extent2 = 0.0; // squared: one square root per cell
                    for (var i = lo; i < hi; i++)
                    {
                        double x = X(i), y = Y(i), r2 = x * x + y * y;
                        if (r2 > extent2 && double.IsFinite(r2)) extent2 = r2;
                    }
                    var extent = Math.Sqrt(extent2);
                    if (dense)
                    {
                        var masks = order.Select(_ => new byte[MaskSize * MaskSize]).ToArray();
                        var scale = MaskSize / (2 * Math.Max(extent, 1e-12) * 1.0001);
                        for (var i = lo; i < hi; i++)
                        {
                            double x = X(i), y = Y(i);
                            if (!double.IsFinite(x) || !double.IsFinite(y)) continue;
                            var col = (int)(x * scale + MaskSize / 2.0);
                            var row = (int)(MaskSize / 2.0 - y * scale);
                            if ((uint)col < MaskSize && (uint)row < MaskSize) masks[groupOf[i - lo]][row * MaskSize + col] = 1;
                        }
                        built[c] = ([], order.Select((w, g) => new Coverage(w, masks[g])).ToList(), extent);
                        return;
                    }
                    var arrays = sizes.Select(n => GC.AllocateUninitializedArray<double>(2 * n)).ToArray(); // every slot is written below
                    var fill = new int[arrays.Length];
                    for (var i = lo; i < hi; i++)
                    {
                        var g = groupOf[i - lo];
                        var at = fill[g];
                        arrays[g][at] = X(i);
                        arrays[g][at + 1] = Y(i);
                        fill[g] = at + 2;
                    }
                    built[c] = (order.Select((w, g) => new Group(w, arrays[g])).ToList(), null, extent);
                }
                if (dense) Parallel.For(0, planes.Length, Plane);
                else for (var c = 0; c < planes.Length; c++) Plane(c);

                var cells = new List<Cell>();
                for (var c = 0; c < planes.Length; c++)
                {
                    var (groups, masks, extent) = built[c];
                    // Statistics: the plane's through-focus entry, else (base plane
                    // without through-focus) the field's own values.
                    var entry = hasFocus && entries is not null && picked[c] < entries.Count ? entries[picked[c]] : null;
                    var isBase = c == baseIndex;
                    var rms = Json.Number(Json.At(entry, rmsKey)) ?? (isBase ? Json.Number(Json.At(f, "rms_radius")) : null) ?? double.NaN;
                    var engineGeo = Json.Nf(Json.At(f, "geo_radius"));
                    var geo = isBase && double.IsFinite(engineGeo) ? engineGeo : count > 0 ? extent : double.NaN;
                    cells.Add(new Cell(c, planes[c], groups, count, rms, geo, count > 0 ? extent : double.IsFinite(geo) ? geo : 0, masks));
                }
                var value = index < d.FieldLabels.Count ? d.FieldLabels[index] : Fmt.Field(Json.NfArray(Json.At(f, "field_value")));
                rows[k] = new Row(index, $"Field {index + 1}", value, cells);
            }
            if (nPoints > DensePoints) Parallel.For(0, list.Count, FieldRow);
            else for (var k = 0; k < list.Count; k++) FieldRow(k);
            return new SpotGridData(planes, all.Length, baseIndex, rows, airy, wavelengths, um, reference);
        }
        finally
        {
            foreach (var a in new[] { xy, sl, wl }) if (a.Length > 0) pool.Return(a);
        }

    }

    /// <summary>
    /// The reference of the plotted points: "centroid" when each field's
    /// <c>reference</c> is its centroid and not its chief ray (on axis both
    /// coincide, which reads as the default, the chief ray).
    /// </summary>
    static string ReferenceOf(JsonArray fields)
    {
        foreach (var f in fields)
        {
            var r = Json.NfArray(Json.At(f, "reference"));
            var chief = Json.NfArray(Json.At(f, "chief"));
            var cen = Json.NfArray(Json.At(f, "centroid"));
            if (r is not { Length: 2 } || chief is not { Length: 2 } || cen is not { Length: 2 }) continue;
            bool Near(double[] a) => Math.Abs(a[0] - r[0]) + Math.Abs(a[1] - r[1]) <= 1e-9 * (1 + Math.Abs(r[0]) + Math.Abs(r[1]));
            bool atChief = Near(chief), atCentroid = Near(cen);
            if (atChief != atCentroid) return atCentroid ? "centroid" : "chief";
        }
        return "chief";
    }

    /// <summary>Box half-width of each row (mm): the largest cell extent (or the Airy radius) × 1.15, over the grid or per row.</summary>
    public static double[] HalfWidths(IReadOnlyList<IReadOnlyList<double>> extents, double airy, SpotScale scale)
    {
        var a = double.IsFinite(airy) ? airy : 0;
        double Hw(IEnumerable<double> e) => Math.Max(Math.Max(e.Where(double.IsFinite).DefaultIfEmpty(0).Max(), a), 1e-6) * 1.15;
        if (scale == SpotScale.Grid)
        {
            var w = Hw(extents.SelectMany(e => e));
            return extents.Select(_ => w).ToArray();
        }
        return extents.Select(Hw).ToArray();
    }

    public double[] HalfWidths(SpotScale scale) =>
        HalfWidths(Rows.Select(r => (IReadOnlyList<double>)r.Cells.Select(c => c.Extent).ToList()).ToList(), Airy, scale);

    /// <summary>
    /// The half-widths shared with <paramref name="other"/> (the design's grid beside a previewed
    /// candidate's): the larger of the two per row, so switching between them keeps the boxes and
    /// shows which spot is smaller. Its own when the rows differ.
    /// </summary>
    public double[] HalfWidths(SpotScale scale, SpotGridData? other)
    {
        var own = HalfWidths(scale);
        if (other is null || other.Rows.Count != Rows.Count) return own;
        return own.Zip(other.HalfWidths(scale), Math.Max).ToArray();
    }

    /// <summary>Indices of <paramref name="count"/> items thinned evenly to at most <paramref name="limit"/>, keeping the first, the last and <paramref name="keeping"/>.</summary>
    public static int[] Thin(int count, int limit, int? keeping)
    {
        if (count <= limit || limit < 2) return Enumerable.Range(0, count).ToArray();
        var set = new SortedSet<int>(Enumerable.Range(0, limit).Select(k => (int)Math.Round((double)k * (count - 1) / (limit - 1), MidpointRounding.AwayFromZero)));
        if (keeping is int keep && keep >= 0 && keep < count && !set.Contains(keep))
        {
            // Swap the nearest interior pick for the kept index.
            var interior = set.Where(i => i != 0 && i != count - 1).ToList();
            if (interior.Count > 0) set.Remove(interior.MinBy(i => Math.Abs(i - keep)));
            set.Add(keep);
        }
        return [.. set];
    }

    static int? Nearest(double[] values, double x) =>
        values.Length == 0 ? null : Enumerable.Range(0, values.Length).MinBy(i => Math.Abs(values[i] - x));

    /// <summary>A plane's column label: "0", "+0.08 mm", "−0.0125 mm", "+5 µm".</summary>
    public static string PlaneLabel(double d)
    {
        if (!double.IsFinite(d)) return Fmt.Short(d);
        var a = Math.Abs(d);
        if (a < 1e-9) return "0";
        var sign = d < 0 ? "−" : "+";
        return a < 1e-3
            ? sign + (a * 1000).ToString("G3", CultureInfo.InvariantCulture) + " µm"
            : sign + a.ToString("G3", CultureInfo.InvariantCulture) + " mm";
    }

    /// <summary>
    /// Marker radius (DIP) and opacity of a cell of side <paramref name="cell"/>
    /// holding <paramref name="points"/> points: radius 0.3 · cell / √n within
    /// 0.5…1.25; the opacity falls from 0.85 (≤ 200 points) to 0.5 (≥ 1500),
    /// so dense spots keep their structure.
    /// </summary>
    public static (double Radius, double Opacity) MarkerStyle(int points, double cell)
    {
        var n = (double)Math.Max(points, 1);
        var radius = Math.Clamp(0.3 * cell / Math.Sqrt(n), 0.5, 1.25);
        var t = Math.Clamp((Math.Log(n) - Math.Log(200)) / (Math.Log(1500) - Math.Log(200)), 0, 1);
        return (radius, 0.85 - 0.35 * t);
    }

    /// <summary>The scale note under the grid.</summary>
    public string ScaleNote(SpotScale scale, SpotGridData? other = null)
    {
        var s = scale == SpotScale.Grid || Rows.Count < 2
            ? $"Box half-width {Fmt.Length(HalfWidths(SpotScale.Grid, other).FirstOrDefault(double.NaN))}, common to all cells"
              + (other is null ? "" : " (design and candidate)")
            : "Box half-width per field, shown by each row";
        s += Reference == "centroid" ? " · about the centroid" : " · about the chief ray";
        if (Planes.Count < SampledPlanes) s += $" · {Planes.Count} of {SampledPlanes} planes shown";
        return s;
    }
}

/// <summary>Cell size of the spot grid: all columns side by side, rows scrolling.</summary>
public static class SpotGridLayout
{
    public const double MinCell = 48, ComfortableCell = 120, MaxCell = 420, Spacing = 6, ColumnHeader = 18, Caption = 16;

    public readonly record struct Metrics(double Cell, double RowHeader, bool ScrollsHorizontally);

    public static double RowHeaderWidth(double width) => width < 420 ? 60 : 84;

    /// <summary>
    /// The narrowest cell whose captions ("42.43 µm") stay legible: below it a grid of several focus planes
    /// folds to its base plane rather than cut or scroll its columns.
    /// </summary>
    public const double FoldCell = 58;

    /// <summary>Whether <paramref name="columns"/> focus planes are too many for <paramref name="width"/>.</summary>
    public static bool Folds(int columns, double width) =>
        columns > 1 && (width - RowHeaderWidth(width) - columns * Spacing) / columns < FoldCell;

    /// <summary>
    /// The cell side for <paramref name="columns"/> × <paramref name="rows"/> cells: as wide as the
    /// width allows, capped so the grid fits the height when that leaves cells of at least
    /// <see cref="ComfortableCell"/>, never below <see cref="MinCell"/> nor above <see cref="MaxCell"/>.
    /// </summary>
    public static Metrics Compute(int columns, int rows, double width, double height)
    {
        double cols = Math.Max(columns, 1), rws = Math.Max(rows, 1);
        var header = RowHeaderWidth(width);
        var byWidth = (width - header - cols * Spacing) / cols;
        var byHeight = (height - ColumnHeader - Spacing - rws * (Caption + 2 + Spacing)) / rws;
        var cell = Math.Min(Math.Min(byWidth, Math.Max(byHeight, ComfortableCell)), MaxCell);
        var scrolls = cell < MinCell;
        cell = Math.Floor(Math.Max(cell, MinCell));
        return new Metrics(cell, header, scrolls);
    }
}
