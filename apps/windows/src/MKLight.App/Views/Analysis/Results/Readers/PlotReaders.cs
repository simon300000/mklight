using System.Globalization;
using System.Text.Json.Nodes;

namespace MKLight.App.Views.Analysis.Results.Readers;

/// <summary><c>grid_distortion</c>: the paraxial and real chief-ray grids (n × n, x fastest).</summary>
public sealed class GridDistortionData
{
    public required int N { get; init; }
    public required (double X, double Y)[] Paraxial { get; init; }
    public required (double X, double Y)[] Real { get; init; }
    public double MaxDistortion { get; init; } = double.NaN;

    public static GridDistortionData? Read(ResultData d)
    {
        if (Json.Int(Json.At(d.Json, "grid_size")) is not int n || n < 2) return null;
        var p = Plot.Pairs(d.Buffer("paraxial"));
        var r = Plot.Pairs(d.Buffer("real"));
        if (p is null || r is null || p.Length != n * n || r.Length != n * n) return null;
        return new GridDistortionData { N = n, Paraxial = p, Real = r, MaxDistortion = Json.Nf(Json.At(d.Json, "max_abs_distortion_percent")) };
    }

    /// <summary>The grid's lines: rows then columns, each a polyline broken at missing points.</summary>
    public IEnumerable<(double X, double Y)[]> Lines((double X, double Y)[] pts)
    {
        for (var line = 0; line < N; line++)
        {
            yield return Enumerable.Range(0, N).Select(k => pts[line * N + k]).ToArray();
            yield return Enumerable.Range(0, N).Select(k => pts[k * N + line]).ToArray();
        }
    }
}

/// <summary><c>footprint</c>: the hit points on one surface by wavelength, the clipped points and the clear aperture.</summary>
public sealed class FootprintData
{
    public required (double X, double Y, int Wavelength)[] Points { get; init; }
    public required (double X, double Y)[] Clipped { get; init; }
    public double SemiDiameter { get; init; } = double.NaN;
    public int Surface { get; init; }
    public IReadOnlyList<int> Wavelengths { get; init; } = [];

    public static FootprintData? Read(ResultData d)
    {
        var j = d.Json;
        if (Json.Arr(Json.At(j, "sets")) is not { } sets || Plot.Pairs(d.Buffer("points")) is not { } p) return null;
        var pts = new List<(double, double, int)>();
        foreach (var s in sets)
        {
            if (Json.IntArray(Json.At(s, "points")) is not [var off, var count]) continue;
            var w = Json.Int(Json.At(s, "wavelength")) ?? 0;
            for (var i = Math.Max(off, 0); i < Math.Min((long)off + count, p.Length); i++) pts.Add((p[i].X, p[i].Y, w));
        }
        var c = d.Buffer("clipped");
        var clipped = c is null ? [] : Plot.Pairs(c);
        if (clipped is null) return null;
        return new FootprintData
        {
            Points = pts.ToArray(),
            Clipped = clipped,
            SemiDiameter = Json.Nf(Json.At(j, "semi_diameter")),
            Surface = Json.Int(Json.At(j, "surface")) ?? 0,
            Wavelengths = pts.Select(q => q.Item3).Distinct().Order().ToList(),
        };
    }
}

/// <summary><c>ghost_paths</c>: the ranked first-order ghosts.</summary>
public sealed class GhostPathsData
{
    public sealed record Ghost(string Label, int[] Path, double Irradiance, double Focus, double Spot, double Reflectance, bool NearFocus);

    public required IReadOnlyList<Ghost> Ghosts { get; init; }
    public bool Truncated { get; init; }

    public static GhostPathsData? Read(ResultData d)
    {
        if (Json.Arr(Json.At(d.Json, "ghosts")) is not { } list) return null;
        return new GhostPathsData
        {
            Ghosts = list.Select(g =>
            {
                var path = Json.IntArray(Json.At(g, "path")) ?? [];
                var label = Json.Str(Json.At(g, "label"));
                return new Ghost(string.IsNullOrEmpty(label) ? string.Join("·", path) : label, path,
                    Json.Nf(Json.At(g, "relative_irradiance")), Json.Nf(Json.At(g, "focus_from_sensor")), Json.Nf(Json.At(g, "spot_diameter")),
                    Json.Nf(Json.At(g, "reflectance_product")), Json.Bool(Json.At(g, "near_focus")) ?? false);
            }).ToList(),
            Truncated = Json.Bool(Json.At(d.Json, "truncated")) ?? false,
        };
    }

    /// <summary>"2.8e-07".</summary>
    public static string Sci(double v) => double.IsFinite(v) ? v.ToString("0.0e+00", CultureInfo.InvariantCulture).Replace("e-0", "e-").Replace("e+0", "e+") : "—";
}

/// <summary>Sensor maps (ghost irradiance, stray light): several maps over the sensor in mm, with a summary table.</summary>
public sealed class SensorMapsData
{
    public required IReadOnlyList<(string Title, MapGrid Grid)> Maps { get; init; }
    public required string Unit { get; init; }
    public IReadOnlyList<(string Label, string Value)> Rows { get; init; } = [];

    /// <summary>A number with a value: null for null, non-finite or anything else.</summary>
    static double? Defined(JsonNode? n) => Json.Number(n) is double v && double.IsFinite(v) ? v : null;
    static string D(double? v, string unit = "") => Fmt.Value(v ?? double.NaN, unit);

    public static SensorMapsData? GhostIrradiance(ResultData d)
    {
        var j = d.Json;
        var half = Json.NfArray(Json.At(j, "map", "half_size")) ?? [1, 1];
        if (half is not [var hx, var hy] || !(hx >= 0) || !(hy >= 0)) return null;
        var maps = new List<(string, MapGrid)>();
        if (MapGrid.From(d.Buffer("total_ghost_map"), 0, -hx, hx, -hy, hy) is { } g) maps.Add(("All ghosts", g));
        if (MapGrid.From(d.Buffer("primary_map"), 0, -hx, hx, -hy, hy) is { } p) maps.Add(("Primary image", p));
        if (maps.Count == 0) return null;
        var rows = new List<(string, string)>
        {
            ("Ghost / primary flux", D(Defined(Json.At(j, "total_flux_ratio")))),
            ("Ghost / primary peak", D(Defined(Json.At(j, "total_peak_ratio")))),
        };
        rows.AddRange((Json.Arr(Json.At(j, "ghosts")) ?? []).Take(6).Select(g => (Json.Str(Json.At(g, "label")) ?? "", D(Defined(Json.At(g, "peak_ratio"))))));
        return new SensorMapsData { Maps = maps, Unit = "mm⁻²", Rows = rows };
    }

    public static SensorMapsData? StrayLight(ResultData d)
    {
        var j = d.Json;
        var hx = Json.Number(Json.At(j, "detector", "half_x")) ?? 1;
        var hy = Json.Number(Json.At(j, "detector", "half_y")) ?? 1;
        if (!(hx >= 0) || !(hy >= 0)) return null;
        var maps = new List<(string, MapGrid)>();
        if (MapGrid.From(d.Buffer("total_irradiance"), 0, -hx, hx, -hy, hy) is { } t) maps.Add(("Stray total", t));
        var classes = (Json.Arr(Json.At(j, "map_classes")) ?? []).Select(Json.Str).ToList();
        for (var k = 0; k < classes.Count; k++)
            if (classes[k] is { } c && MapGrid.From(d.Buffer("class_irradiance"), k, -hx, hx, -hy, hy) is { } g) maps.Add((GenericData.Title(c), g));
        if (maps.Count == 0) return null;
        var rows = new List<(string, string)>
        {
            ("Stray flux", D(Defined(Json.At(j, "stray_flux")), "W")),
            ("± standard error", D(Defined(Json.At(j, "stray_std_error")), "W")),
        };
        if (Defined(Json.At(j, "stray_fraction")) is double f) rows.Add(("Stray fraction", Fmt.Short(f)));
        rows.AddRange((Json.Arr(Json.At(j, "totals")) ?? []).Select(t => (GenericData.Title(Json.Str(Json.At(t, "class")) ?? ""), D(Defined(Json.At(t, "flux")), "W"))));
        return new SensorMapsData { Maps = maps, Unit = "W/mm²", Rows = rows };
    }
}

/// <summary><c>polarization_pupil_map</c>: per-sample quantities over the pupil, and the pupil statistics.</summary>
public sealed class PolarizationData
{
    public static readonly string[] Quantities = ["Intensity", "Diattenuation", "Retardance"];

    public required (double X, double Y)[] Pupil { get; init; }
    public required bool[] Reached { get; init; }
    public required IReadOnlyDictionary<string, double[]> Values { get; init; }
    public IReadOnlyList<(string Label, string Value)> Stats { get; init; } = [];
    public int Grid { get; init; }

    public static PolarizationData? Read(ResultData d)
    {
        if (Plot.Pairs(d.Buffer("pupil")) is not { } p || d.Buffer("intensity") is not { } i) return null;
        var reached = d.Buffer("reached") is { } r ? Enumerable.Range(0, r.Count).Select(k => r[k] != 0).ToArray() : Enumerable.Repeat(true, p.Length).ToArray();
        if (reached.Length != p.Length) return null;
        var j = d.Json;
        return new PolarizationData
        {
            Pupil = p,
            Reached = reached,
            Values = new Dictionary<string, double[]>
            {
                ["Intensity"] = Json.Doubles(i),
                ["Diattenuation"] = d.Buffer("diattenuation") is { } a ? Json.Doubles(a) : [],
                ["Retardance"] = d.Buffer("retardance") is { } t ? Json.Doubles(t) : [],
            },
            Grid = Json.Int(Json.At(j, "grid")) ?? 0,
            Stats =
            [
                ("Mean intensity", Fmt.Short(Json.Nf(Json.At(j, "mean_intensity")))),
                ("Max diattenuation", Fmt.Short(Json.Nf(Json.At(j, "max_diattenuation")))),
                ("Max retardance", Fmt.Value(Json.Nf(Json.At(j, "max_retardance")), "rad")),
                ("RMS retardance", Fmt.Value(Json.Nf(Json.At(j, "rms_retardance")), "rad")),
            ],
        };
    }

    /// <summary>The quantity as a grid over the pupil square (cells of the engine's n × n grid; NaN where no ray reached).</summary>
    public MapGrid? Map(string quantity)
    {
        if (!Values.TryGetValue(quantity, out var v) || v.Length != Pupil.Length || Pupil.Length == 0) return null;
        var n = Grid > 0 ? Grid : (int)Math.Round(Math.Sqrt(Pupil.Length * 4 / Math.PI));
        if (n < 1) return null;
        var cells = Enumerable.Repeat(double.NaN, n * n).ToArray();
        for (var k = 0; k < Pupil.Length; k++)
        {
            if (!Reached[k] || !double.IsFinite(v[k])) continue;
            var i = (int)Math.Round((Pupil[k].X + 1) / 2 * (n - 1));
            var jj = (int)Math.Round((Pupil[k].Y + 1) / 2 * (n - 1));
            if (i >= 0 && i < n && jj >= 0 && jj < n) cells[jj * n + i] = v[k];
        }
        return new MapGrid(n, n, cells) { X0 = -1, X1 = 1, Y0 = -1, Y1 = 1 };
    }
}

/// <summary><c>system_transmission</c>: transmission per field and wavelength, and per field polychromatic.</summary>
public sealed class TransmissionData
{
    public sealed record Point(int Field, double Um, double T);

    public required IReadOnlyList<Point> Points { get; init; }
    public IReadOnlyList<double> Polychromatic { get; init; } = [];
    public IReadOnlyList<string> FieldLabels { get; init; } = [];

    public static TransmissionData? Read(ResultData d)
    {
        var j = d.Json;
        if (Json.Arr(Json.At(j, "points")) is not { Count: > 0 } list) return null;
        return new TransmissionData
        {
            Points = list.Select(p => new Point(Json.Int(Json.At(p, "field")) ?? 0, Json.Nf(Json.At(p, "wavelength_um")), Json.Nf(Json.At(p, "transmission")))).ToList(),
            Polychromatic = Json.NfArray(Json.At(j, "polychromatic")) ?? [],
            FieldLabels = (Json.Arr(Json.At(j, "fields")) ?? []).Select(f =>
                "H " + ImageSimulationData.Coordinates(Json.Number(Json.At(f, "hx")) ?? 0, Json.Number(Json.At(f, "hy")) ?? 0)).ToList(),
        };
    }

    public IReadOnlyList<int> Fields => Points.Select(p => p.Field).Distinct().Order().ToList();
    public bool Spectral => Points.Select(p => p.Um).Distinct().Count() > 1;
}

static class Plot
{
    /// <summary>The rows of an [n, 2] buffer; null for any other shape (an empty buffer has no rows).</summary>
    public static (double X, double Y)[]? Pairs(IResultArray? b)
    {
        if (b is null) return null;
        if (b.Count == 0) return [];
        if (b.Shape.Count != 2 || b.Shape[1] != 2 || b.Count != b.Shape[0] * 2) return null;
        return Enumerable.Range(0, b.Shape[0]).Select(k => (b[2 * k], b[2 * k + 1])).ToArray();
    }
}
