namespace MKLight.App.Views.Analysis.Results.Readers;

/// <summary>
/// A row-major grid of values with its extent in plot units: row 0 at the
/// bottom (<c>Y0</c>), x fastest, samples spanning X0…X1 × Y0…Y1. Maps of
/// cells (PSF, irradiance) and of nodes (field maps) both use it.
/// </summary>
public sealed class MapGrid
{
    public int Nx { get; }
    public int Ny { get; }
    public double[] Values { get; }
    public double X0 { get; init; } = -1;
    public double X1 { get; init; } = 1;
    public double Y0 { get; init; } = -1;
    public double Y1 { get; init; } = 1;
    /// <summary>
    /// Outside the unit disk (hypot(Hx, Hy) &gt; 1) the map has no data by
    /// definition: upsampling interpolates from the finite corners only and
    /// leaves the outside empty, so the map reaches the field's edge exactly.
    /// </summary>
    public bool UnitDiskNoData { get; init; }

    public MapGrid(int nx, int ny, double[] values)
    {
        Nx = nx;
        Ny = ny;
        Values = values;
    }

    /// <summary>A <c>[ny, nx]</c> buffer, or slice <paramref name="k"/> of a <c>[k, ny, nx]</c> one.</summary>
    public static MapGrid? From(IResultArray? b, int k = 0, double x0 = -1, double x1 = 1, double y0 = -1, double y1 = 1)
    {
        if (b is null || b.Shape.Count < 2) return null;
        int ny = b.Shape[^2], nx = b.Shape[^1];
        if (nx <= 0 || ny <= 0) return null;
        long start = (long)k * nx * ny;
        if (start < 0 || start + (long)nx * ny > b.Count) return null;
        var v = new double[nx * ny];
        for (var i = 0; i < v.Length; i++) v[i] = b[(int)start + i];
        return new MapGrid(nx, ny, v) { X0 = x0, X1 = x1, Y0 = y0, Y1 = y1 };
    }

    public static bool Excluded(double x, double y) => Math.Sqrt(x * x + y * y) > 1 + 1e-9;

    /// <summary>The finite range; null when no value is finite.</summary>
    public (double Lo, double Hi)? Range()
    {
        double lo = double.PositiveInfinity, hi = double.NegativeInfinity;
        foreach (var v in Values)
        {
            if (!double.IsFinite(v)) continue;
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        return double.IsFinite(lo) ? (lo, hi) : null;
    }

    public bool HasMissing => UnitDiskNoData || Values.Any(v => !double.IsFinite(v));

    /// <summary>The position of node k (<c>j * Nx + i</c>), samples at the nodes.</summary>
    public (double X, double Y) Node(int k)
    {
        int i = k % Nx, j = k / Nx;
        var dx = Nx > 1 ? (X1 - X0) / (Nx - 1) : 0;
        var dy = Ny > 1 ? (Y1 - Y0) / (Ny - 1) : 0;
        return (X0 + i * dx, Y0 + j * dy);
    }

    /// <summary>
    /// Bilinear upsampling of a node grid to <paramref name="n"/> columns (for
    /// display, so the colours and the linear iso-lines agree). NaN where a
    /// corner is, except on a unit-disk map, which interpolates from its finite
    /// corners and leaves only the outside of the field empty.
    /// </summary>
    public MapGrid Upsampled(int n)
    {
        if (Nx < 2 || Ny < 2 || n <= Math.Max(Nx, Ny)) return this;
        int w = n, h = Math.Max(2, (int)Math.Round((double)n * (Ny - 1) / Math.Max(Nx - 1, 1)));
        var o = new double[w * h];
        for (var row = 0; row < h; row++)
        {
            var y = (double)row / (h - 1) * (Ny - 1);
            int j = Math.Min((int)y, Ny - 2);
            var b = y - j;
            for (var col = 0; col < w; col++)
            {
                var x = (double)col / (w - 1) * (Nx - 1);
                int i = Math.Min((int)x, Nx - 2);
                var a = x - i;
                double v00 = Values[j * Nx + i], v10 = Values[j * Nx + i + 1], v01 = Values[(j + 1) * Nx + i], v11 = Values[(j + 1) * Nx + i + 1];
                if (UnitDiskNoData)
                {
                    var px = X0 + (double)col / (w - 1) * (X1 - X0);
                    var py = Y0 + (double)row / (h - 1) * (Y1 - Y0);
                    if (Excluded(px, py)) { o[row * w + col] = double.NaN; continue; }
                    double sum = 0, ws = 0;
                    Span<double> wt = [(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b];
                    Span<double> vs = [v00, v10, v01, v11];
                    for (var q = 0; q < 4; q++)
                        if (double.IsFinite(vs[q])) { sum += wt[q] * vs[q]; ws += wt[q]; }
                    o[row * w + col] = ws > 1e-12 ? sum / ws : double.NaN;
                    continue;
                }
                o[row * w + col] = (1 - b) * ((1 - a) * v00 + a * v10) + b * ((1 - a) * v01 + a * v11);
            }
        }
        return new MapGrid(w, h, o) { X0 = X0, X1 = X1, Y0 = Y0, Y1 = Y1, UnitDiskNoData = UnitDiskNoData };
    }

    /// <summary>The node nearest a point in plot units.</summary>
    public int Nearest(double x, double y)
    {
        var best = 0;
        var bd = double.PositiveInfinity;
        for (var k = 0; k < Nx * Ny; k++)
        {
            var (px, py) = Node(k);
            var d = (px - x) * (px - x) + (py - y) * (py - y);
            if (d < bd) { bd = d; best = k; }
        }
        return best;
    }
}

public enum MapScaleKind
{
    /// <summary>Magnitude: one hue, zero recedes into the plot surface (PSF, irradiance).</summary>
    Sequential,
    /// <summary>Signed values, symmetric about zero.</summary>
    Diverging,
    /// <summary>A quantity over its range: the perceptual multi-hue ramp, every value visible (field and pupil maps).</summary>
    Perceptual,
}

/// <summary>How a heat map's colours span its values.</summary>
public enum HeatRangeMode
{
    /// <summary>The data's own range: small variations stay visible.</summary>
    Data,
    /// <summary>0 to 1, for fractions compared across designs.</summary>
    Unit,
}

/// <summary>
/// The colour scale of a map: its kind and value range (and, for a
/// magnitude, whether it is logarithmic). One rule for every map of a
/// quantity (docs/APP.md "Heat-map scales"): signed data keeps a diverging
/// scale centred on 0 and symmetric about it; other data takes the
/// perceptual ramp over its own range, or 0…1 for fractions when asked; a
/// flat map gets a narrow band around its value.
/// </summary>
public readonly record struct MapScale(MapScaleKind Kind, double Min, double Max, bool Log = false)
{
    /// <summary>The smallest value a logarithmic scale tells apart: five decades below the largest.</summary>
    public static double LogFloor(double max) => Math.Max(max * 1e-5, 1e-300);

    public static MapScale ForQuantity(double lo, double hi, bool signed, HeatRangeMode mode = HeatRangeMode.Data, bool isFraction = false)
    {
        if (!double.IsFinite(lo) || !double.IsFinite(hi)) return new(signed ? MapScaleKind.Diverging : MapScaleKind.Perceptual, signed ? -1 : 0, 1);
        if (signed)
        {
            var m = Math.Max(Math.Max(Math.Abs(lo), Math.Abs(hi)), 1e-300);
            return new(MapScaleKind.Diverging, -m, m);
        }
        if (mode == HeatRangeMode.Unit && isFraction) return new(MapScaleKind.Perceptual, 0, Math.Max(1, hi));
        var tiny = Math.Max(Math.Max(Math.Abs(lo), Math.Abs(hi)), 1e-300) * 1e-9;
        if (hi - lo <= tiny)
        {
            var d = Math.Max(Math.Abs(lo) * 1e-3, 1e-12);
            return new(MapScaleKind.Perceptual, lo - d, hi + d);
        }
        return new(MapScaleKind.Perceptual, lo, hi);
    }

    /// <summary>A magnitude from 0 (or the smallest value, if negative) to the largest.</summary>
    public static MapScale Magnitude(double lo, double hi, bool log)
    {
        if (!double.IsFinite(hi) || hi <= 0) hi = 1;
        var min = double.IsFinite(lo) ? Math.Min(lo, 0) : 0;
        return new(MapScaleKind.Sequential, log ? LogFloor(hi) : min, hi, log);
    }

    /// <summary>
    /// The value's place on the ramp: 0…1 (diverging: 0.5 is zero); NaN stays NaN.
    /// </summary>
    public double Position(double v)
    {
        if (!double.IsFinite(v)) return double.NaN;
        if (Kind == MapScaleKind.Diverging) return Math.Clamp(0.5 + v / Math.Max(Math.Abs(Max), 1e-300) / 2, 0, 1);
        if (Log)
        {
            var lo = Math.Log10(Min);
            return Math.Clamp((Math.Log10(Math.Max(v, Min)) - lo) / Math.Max(Math.Log10(Max) - lo, 1e-12), 0, 1);
        }
        return Math.Clamp((v - Min) / Math.Max(Max - Min, 1e-300), 0, 1);
    }

    /// <summary>The value at place t (0…1) on the ramp: the inverse of <see cref="Position"/>.</summary>
    public double ValueAt(double t)
    {
        if (Log) return Math.Pow(10, Math.Log10(Min) + t * (Math.Log10(Max) - Math.Log10(Min)));
        return Min + t * (Max - Min);
    }

    /// <summary>Round values for the colour bar's ticks: about <paramref name="count"/>, decades on a log scale.</summary>
    public IReadOnlyList<double> Ticks(int count = 4)
    {
        var list = new List<double>();
        if (Log)
        {
            for (var e = Math.Ceiling(Math.Log10(Min) - 1e-9); e <= Math.Log10(Max) + 1e-9; e++) list.Add(Math.Pow(10, e));
            while (list.Count > count + 1) list = list.Where((_, i) => i % 2 == (list.Count - 1) % 2).ToList();
            return list;
        }
        var step = Contours.Nice((Max - Min) / Math.Max(count, 1));
        if (!(step > 0) || !double.IsFinite(step)) return [Min, Max];
        for (var k = Math.Ceiling(Min / step - 1e-9); k * step <= Max + step * 1e-9 && list.Count < 20; k++)
            list.Add(Math.Abs(k) < 1e-12 ? 0 : Math.Round(k * step / step) * step);
        return list;
    }

    /// <summary>A bar label: "%.3g", scientific for very small or large values.</summary>
    public static string Label(double v)
    {
        if (!double.IsFinite(v)) return Fmt.Short(v);
        var a = Math.Abs(v);
        if (a != 0 && (a < 1e-2 || a >= 1e4))
        {
            var e = (int)Math.Floor(Math.Log10(a));
            var m = v / Math.Pow(10, e);
            if (Math.Abs(Math.Abs(m) - 1) < 1e-9) return (v < 0 ? "−" : "") + "10" + Superscript(e);
            return Fmt.G3(m).Replace('-', '−') + "×10" + Superscript(e);
        }
        return Fmt.G3(v).Replace('-', '−');
    }

    static string Superscript(int e) => string.Concat(e.ToString(System.Globalization.CultureInfo.InvariantCulture).Select(c => c switch
    {
        '-' => '⁻', '0' => '⁰', '1' => '¹', '2' => '²', '3' => '³', '4' => '⁴', '5' => '⁵', '6' => '⁶', '7' => '⁷', '8' => '⁸', _ => '⁹',
    }));
}

/// <summary>
/// Iso-lines of a sampled grid by marching squares, with linear
/// interpolation along the cell edges and the cell mean to resolve saddles.
/// </summary>
public static class Contours
{
    public readonly record struct Segment(double Ax, double Ay, double Bx, double By);

    /// <summary>A "nice" step (1, 2, 5 × 10ⁿ) near x.</summary>
    public static double Nice(double x)
    {
        if (!(x > 0) || !double.IsFinite(x)) return 1;
        var e = Math.Pow(10, Math.Floor(Math.Log10(x)));
        var m = x / e;
        return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * e;
    }

    /// <summary>The iso-line of a map as drawn: a map with no data outside the field is traced on its interpolated grid, so lines reach the edge.</summary>
    public static List<Segment> MapSegments(MapGrid grid, double level) =>
        grid.UnitDiskNoData ? Segments(grid.Upsampled(Math.Max(4 * Math.Max(grid.Nx, grid.Ny), 48)), level) : Segments(grid, level);

    /// <summary>The segments of the <paramref name="level"/> iso-line, in plot units. Cells with a non-finite corner are skipped.</summary>
    public static List<Segment> Segments(MapGrid g, double level)
    {
        var o = new List<Segment>();
        int nx = g.Nx, ny = g.Ny;
        if (nx < 2 || ny < 2 || !double.IsFinite(level)) return o;
        var dx = (g.X1 - g.X0) / (nx - 1);
        var dy = (g.Y1 - g.Y0) / (ny - 1);
        (double, double) P(double i, double j) => (g.X0 + i * dx, g.Y0 + j * dy);
        double T(double a, double b) => a == b ? 0.5 : Math.Clamp((level - a) / (b - a), 0, 1);
        void Add((double X, double Y) a, (double X, double Y) b) => o.Add(new Segment(a.X, a.Y, b.X, b.Y));
        for (var j = 0; j < ny - 1; j++)
        {
            for (var i = 0; i < nx - 1; i++)
            {
                double v0 = g.Values[j * nx + i], v1 = g.Values[j * nx + i + 1], v2 = g.Values[(j + 1) * nx + i + 1], v3 = g.Values[(j + 1) * nx + i];
                if (!double.IsFinite(v0) || !double.IsFinite(v1) || !double.IsFinite(v2) || !double.IsFinite(v3)) continue;
                var code = (v0 >= level ? 1 : 0) | (v1 >= level ? 2 : 0) | (v2 >= level ? 4 : 0) | (v3 >= level ? 8 : 0);
                if (code is 0 or 15) continue;
                var bottom = P(i + T(v0, v1), j);
                var right = P(i + 1, j + T(v1, v2));
                var top = P(i + T(v3, v2), j + 1);
                var left = P(i, j + T(v0, v3));
                switch (code)
                {
                    case 1 or 14: Add(left, bottom); break;
                    case 2 or 13: Add(bottom, right); break;
                    case 3 or 12: Add(left, right); break;
                    case 4 or 11: Add(right, top); break;
                    case 6 or 9: Add(bottom, top); break;
                    case 7 or 8: Add(left, top); break;
                    case 5 or 10:
                        var centreHigh = (v0 + v1 + v2 + v3) / 4 >= level;
                        if ((code == 5) == centreHigh) { Add(left, top); Add(bottom, right); }
                        else { Add(left, bottom); Add(right, top); }
                        break;
                }
            }
        }
        return o;
    }

    /// <summary>Iso-illumination levels: round values strictly inside the range, about five (0.88, 0.90, … for 0.87 to 1).</summary>
    public static List<double> IlluminationLevels(double lo, double hi)
    {
        var o = new List<double>();
        var span = hi - lo;
        if (!(span > 1e-9) || !double.IsFinite(span)) return o;
        var step = Nice(span / 5);
        for (var k = Math.Floor(lo / step) + 1; k * step < hi - step * 1e-6; k++)
        {
            var v = Math.Round(k * step * 1e9) / 1e9;
            if (v > lo + step * 1e-6) o.Add(v);
        }
        return o;
    }
}
