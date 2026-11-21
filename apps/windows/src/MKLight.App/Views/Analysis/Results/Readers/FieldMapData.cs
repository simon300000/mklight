using System.Text.Json.Nodes;
using MKLight.Model;

namespace MKLight.App.Views.Analysis.Results.Readers;

/// <summary>
/// The shared result of the signed Cartesian field maps (<c>pupil_aberration</c>,
/// <c>relative_illumination_map</c>, <c>chief_ray_angle</c>): <c>field_coordinates
/// [ny, nx, 2]</c>, <c>image_mm [ny, nx, 2]</c> and <c>values [ny, nx, columns]</c>,
/// the columns described in the JSON. Sample k = j·nx + i, row 0 the lowest Hy.
/// </summary>
public sealed class FieldMapData
{
    public sealed record Column(string Name, string Unit, string Definition)
    {
        /// <summary>"Stop error x" (engine text, in the UI language).</summary>
        public string Title => Name.Length == 0 ? "" : EngineText.T(char.ToUpperInvariant(Name[0]) + Name[1..].Replace('_', ' '));
        public string UnitLabel => Unit switch { "" or "1" => "", "deg" => "°", _ => Unit };
        public string TitleWithUnit => UnitLabel.Length == 0 ? Title : $"{Title} ({UnitLabel})";

        /// <summary>The value format of the column: degrees, lengths, plain numbers.</summary>
        public string Format(double v) => Unit switch
        {
            _ when !double.IsFinite(v) => "—",
            "deg" => Fmt.Fixed(v, 2) + "°",
            "mm" => Fmt.Length(v),
            "1" or "" => Fmt.G4(v),
            _ => Fmt.Value(v, Unit),
        };

        public bool IsFraction => Name is "relative_illumination" or "vignetting_fraction";
    }

    public enum Layouts { Map, SweepX, SweepY }

    public required int Nx { get; init; }
    public required int Ny { get; init; }
    public required IReadOnlyList<Column> Columns { get; init; }
    /// <summary>Normalised (Hx, Hy) per sample.</summary>
    public required (double X, double Y)[] Field { get; init; }
    /// <summary>Chief-ray image point (mm) per sample; NaN for unusable fields.</summary>
    public required (double X, double Y)[] Image { get; init; }
    /// <summary><c>Values[k * Columns.Count + c]</c>.</summary>
    public required double[] Values { get; init; }

    public static FieldMapData? Read(ResultData d)
    {
        var j = d.Json;
        var cols = Json.Arr(Json.At(j, "columns"));
        var f = BufferOf(d, "field_coordinates");
        var v = BufferOf(d, "values");
        if (cols is null || cols.Count == 0 || f is null || v is null || f.Shape.Count != 3 || f.Shape[2] != 2 || v.Shape.Count != 3) return null;
        int ny = f.Shape[0], nx = f.Shape[1], nc = v.Shape[2];
        if (nx <= 0 || ny <= 0 || v.Shape[0] != ny || v.Shape[1] != nx || nc != cols.Count) return null;
        var n = nx * ny;
        var img = BufferOf(d, "image_mm");
        var hasImage = img is not null && img.Count == 2 * n;
        return new FieldMapData
        {
            Nx = nx,
            Ny = ny,
            Columns = cols.Select(c => new Column(Json.Str(Json.At(c, "name")) ?? "", Json.Str(Json.At(c, "unit")) ?? "", Json.Str(Json.At(c, "definition")) ?? "")).ToList(),
            Field = Enumerable.Range(0, n).Select(k => (f[2 * k], f[2 * k + 1])).ToArray(),
            Image = Enumerable.Range(0, n).Select(k => hasImage ? (img![2 * k], img[2 * k + 1]) : (double.NaN, double.NaN)).ToArray(),
            Values = Json.Doubles(v),
        };
    }

    /// <summary>The buffer the result's key names, else the buffer of that name.</summary>
    static IResultArray? BufferOf(ResultData d, string key) => Json.BufferRef(d, Json.At(d.Json, key)) ?? d.Buffer(key);

    public Layouts Layout => Nx > 1 && Ny > 1 ? Layouts.Map : Ny == 1 && Nx > 1 ? Layouts.SweepX : Layouts.SweepY;
    public int Count => Nx * Ny;
    public int? ColumnIndex(string name)
    {
        for (var c = 0; c < Columns.Count; c++) if (Columns[c].Name == name) return c;
        return null;
    }

    public double Value(int sample, int column) => Values[sample * Columns.Count + column];
    public double[] ColumnValues(int c) => Enumerable.Range(0, Count).Select(k => Value(k, c)).ToArray();

    public (double Lo, double Hi) FieldRangeX => Span(Field.Select(p => p.X));
    public (double Lo, double Hi) FieldRangeY => Span(Field.Select(p => p.Y));

    /// <summary>
    /// Whether the samples outside the normalised field carry no data (the engine
    /// reports them as null), and there are some: they are drawn as no data, not 0.
    /// </summary>
    public bool OutsideFieldIsNoData
    {
        get
        {
            var outside = Enumerable.Range(0, Count).Where(k => MapGrid.Excluded(Field[k].X, Field[k].Y)).ToList();
            return outside.Count > 0 && outside.All(k => Enumerable.Range(0, Columns.Count).All(c => !double.IsFinite(Value(k, c))));
        }
    }

    /// <summary>A column as a node grid over the normalised field (map layout).</summary>
    public MapGrid Grid(int c)
    {
        var (x0, x1) = FieldRangeX;
        var (y0, y1) = FieldRangeY;
        return new MapGrid(Nx, Ny, ColumnValues(c)) { X0 = x0, X1 = x1, Y0 = y0, Y1 = y1, UnitDiskNoData = OutsideFieldIsNoData };
    }

    /// <summary>A column along the swept field axis: (H, value) in sample order.</summary>
    public (double H, double V)[] Sweep(int c) =>
        Enumerable.Range(0, Count).Select(k => (Layout == Layouts.SweepX ? Field[k].X : Field[k].Y, Value(k, c))).ToArray();

    public double[] ImageRadius => Image.Select(p => Math.Sqrt(p.X * p.X + p.Y * p.Y)).ToArray();

    /// <summary>(image radius, value) at every usable sample, by radius.</summary>
    public (double R, double V)[] Radial(int c)
    {
        var r = ImageRadius;
        return Enumerable.Range(0, Count).Select(k => (r[k], Value(k, c)))
            .Where(p => double.IsFinite(p.Item1) && double.IsFinite(p.Item2)).OrderBy(p => p.Item1).ToArray();
    }

    /// <summary>The finite range of a column; null when no value is finite.</summary>
    public (double Lo, double Hi)? Range(int c)
    {
        double lo = double.PositiveInfinity, hi = double.NegativeInfinity;
        for (var k = 0; k < Count; k++)
        {
            var v = Value(k, c);
            if (!double.IsFinite(v)) continue;
            lo = Math.Min(lo, v);
            hi = Math.Max(hi, v);
        }
        return double.IsFinite(lo) ? (lo, hi) : null;
    }

    /// <summary>Whether a column is best on a signed (diverging) scale: both signs, or a difference that is zero on axis.</summary>
    public bool IsSigned(int c)
    {
        if (Range(c) is not var (lo, hi)) return false;
        if (lo < 0 && hi > 0) return true;
        var name = Columns[c].Name;
        return lo < 0 || name.Contains("error") || name.Contains("variation") || name.StartsWith("angle_", StringComparison.Ordinal) || name == "margin";
    }

    /// <summary>Whether a column depends on the field radius only (within tolerance of its range).</summary>
    public bool IsRadial(int c, double tolerance = 1e-3)
    {
        if (Range(c) is not var (lo, hi)) return true;
        var scale = Math.Max(Math.Max(hi - lo, Math.Abs(hi)), 1e-12);
        var (fx0, fx1) = FieldRangeX;
        var (fy0, fy1) = FieldRangeY;
        var rx = Math.Max(Math.Max(Math.Abs(fx0), Math.Abs(fx1)), 1e-12);
        var ry = Math.Max(Math.Max(Math.Abs(fy0), Math.Abs(fy1)), 1e-12);
        if (Math.Abs(rx - ry) >= 1e-9) return false;
        var byRadius = new Dictionary<long, double>();
        for (var k = 0; k < Count; k++)
        {
            var v = Value(k, c);
            if (!double.IsFinite(v)) continue;
            var key = (long)(Math.Sqrt(Field[k].X * Field[k].X + Field[k].Y * Field[k].Y) * 1e6);
            if (byRadius.TryGetValue(key, out var w) && Math.Abs(w - v) > tolerance * scale) return false;
            byRadius[key] = v;
        }
        return true;
    }

    public int? ArgMax(int c)
    {
        int? best = null;
        for (var k = 0; k < Count; k++)
            if (double.IsFinite(Value(k, c)) && (best is null || Value(k, c) > Value(best.Value, c))) best = k;
        return best;
    }

    /// <summary>The first column that varies (a y sweep has no x stop error).</summary>
    public int FirstVaryingColumn() =>
        Enumerable.Range(0, Columns.Count).FirstOrDefault(c => Range(c) is var (lo, hi) && hi - lo > 1e-12);

    public static (double Lo, double Hi) Span(IEnumerable<double> v)
    {
        var f = v.Where(double.IsFinite).ToList();
        if (f.Count == 0) return (-1, 1);
        double lo = f.Min(), hi = f.Max();
        return lo < hi ? (lo, hi) : (lo - 1, hi + 1);
    }

    // ---- Chief ray angle ----

    /// <summary><c>[[radius mm, max deg], …]</c> from the run's <c>sensor_limit</c>.</summary>
    public static (double R, double Deg)[] SensorLimit(JsonNode? v) =>
        (Json.Arr(v) ?? []).Select(row => Json.NfArray(row) is [var a, var b] && double.IsFinite(a) && double.IsFinite(b) ? (a, b) : (double.NaN, double.NaN))
            .Where(p => double.IsFinite(p.Item1)).ToArray();

    /// <summary>
    /// The sensor limit as the run applied it: the run's table when known, else
    /// the <c>limit</c> column against the image radius (the engine's interpolation
    /// at the samples).
    /// </summary>
    public (double R, double Deg)[] LimitTable(JsonNode? settings)
    {
        var t = SensorLimit(Json.At(settings, "sensor_limit"));
        if (t.Length > 0 || settings is not null && Json.Has(settings, "sensor_limit")) return t;
        if (ColumnIndex("limit") is not int c) return [];
        var o = new List<(double, double)>();
        foreach (var (r, v) in Radial(c))
            if (o.Count == 0 || r - o[^1].Item1 > 1e-9) o.Add((r, v));
        return o.ToArray();
    }

    /// <summary>The limit at a radius: linear, held at the end points (as the engine).</summary>
    public static double Limit((double R, double Deg)[] table, double r)
    {
        if (table.Length == 0) return double.NaN;
        if (r <= table[0].R) return table[0].Deg;
        if (r >= table[^1].R) return table[^1].Deg;
        for (var k = 1; k < table.Length; k++)
        {
            if (r > table[k].R) continue;
            var (r0, a0) = table[k - 1];
            var (r1, a1) = table[k];
            return r1 > r0 ? a0 + (a1 - a0) * (r - r0) / (r1 - r0) : a1;
        }
        return table[^1].Deg;
    }

    /// <summary>A band between the CRA curve and the limit: a margin, or a violation.</summary>
    public sealed record Band(bool Violated, List<(double X, double Lo, double Hi)> Points);

    /// <summary>Area segments between the curve and the limit, split where they cross.</summary>
    public static List<Band> Bands((double X, double Y)[] curve, (double R, double Deg)[] limit)
    {
        var bands = new List<Band>();
        if (curve.Length < 2 || limit.Length == 0) return bands;
        double lo = curve[0].X, hi = curve[^1].X;
        var xs = new SortedSet<double>(curve.Select(p => p.X));
        foreach (var (r, _) in limit)
            foreach (var x in new[] { r, -r })
                if (x > lo && x < hi) xs.Add(x);
        double Cra(double x)
        {
            for (var k = 1; k < curve.Length; k++)
            {
                if (x > curve[k].X) continue;
                var (x0, y0) = curve[k - 1];
                var (x1, y1) = curve[k];
                return x1 > x0 ? y0 + (y1 - y0) * (x - x0) / (x1 - x0) : y1;
            }
            return curve[^1].Y;
        }
        var pts = xs.Select(x => { var c = Cra(x); var l = Limit(limit, Math.Abs(x)); return (X: x, M: l - c, C: c, L: l); }).ToList();
        for (var k = 1; k < pts.Count; k++)
        {
            var (a, b) = (pts[k - 1], pts[k]);
            if ((a.M < 0) != (b.M < 0) && a.M != 0 && b.M != 0)
            {
                var t = a.M / (a.M - b.M);
                var x = a.X + (b.X - a.X) * t;
                var c = Cra(x);
                pts.Insert(k, (x, 0, c, c));
                k++;
            }
        }
        for (var k = 1; k < pts.Count; k++)
        {
            var (a, b) = (pts[k - 1], pts[k]);
            var violated = (a.M + b.M) / 2 < 0;
            var ib = (b.X, Math.Min(b.C, b.L), Math.Max(b.C, b.L));
            if (bands.Count > 0 && bands[^1].Violated == violated) bands[^1].Points.Add(ib);
            else bands.Add(new Band(violated, [(a.X, Math.Min(a.C, a.L), Math.Max(a.C, a.L)), ib]));
        }
        return bands.Where(b => b.Points.Count >= 2).ToList();
    }
}
