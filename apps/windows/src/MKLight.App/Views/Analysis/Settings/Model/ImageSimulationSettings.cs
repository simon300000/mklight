using System.Globalization;
using System.Text.Json.Nodes;
using MKLight.App.Support;
using MKLight.App.Views.Analysis.Results.Readers;

namespace MKLight.App.Views.Analysis.Settings;

/// <summary>The image simulation's built-in source patterns (<c>{"kind": "pattern", "pattern": id}</c>).</summary>
public static class ImageSourcePatterns
{
    /// <summary>The patterns by id, with their titles in the UI language.</summary>
    public static IReadOnlyList<(string Id, string Title)> All =>
    [
        ("grid", Loc.S("Analysis_ImageSim_PatternGrid")), ("siemens_star", Loc.S("Analysis_ImageSim_PatternSiemensStar")),
        ("slanted_edge", Loc.S("Analysis_ImageSim_PatternSlantedEdge")), ("text_chart", Loc.S("Analysis_ImageSim_PatternTextChart")),
    ];

    public static string Title(string id) => All.FirstOrDefault(p => p.Id == id).Title ?? id;

    public static JsonObject Source(string id) => new() { ["kind"] = "pattern", ["pattern"] = id };
}

/// <summary>
/// The image simulation's optics sampling (<c>optics</c>): one key changed,
/// or one key of its <c>pupil</c>; a reset key takes the default back. Other keys are kept.
/// </summary>
public static class ImageOptics
{
    /// <param name="current">The setting's value (its own, else the default).</param>
    /// <param name="defaults">The schema's default object.</param>
    public static JsonObject With(JsonNode? current, JsonNode? defaults, string key, Edit e)
    {
        var o = current?.DeepClone() as JsonObject ?? new JsonObject();
        o[key] = e.Remove ? defaults?[key]?.DeepClone() : e.Value?.DeepClone();
        return o;
    }

    public static JsonObject WithPupil(JsonNode? current, JsonNode? defaults, string key, Edit e)
    {
        var pupil = current?["pupil"]?.DeepClone() as JsonObject ?? defaults?["pupil"]?.DeepClone() as JsonObject ?? new JsonObject();
        pupil[key] = e.Remove ? defaults?["pupil"]?[key]?.DeepClone() : e.Value?.DeepClone();
        return With(current, defaults, "pupil", new Edit(pupil));
    }
}

/// <summary>
/// The simulated patch of an <c>image_simulation</c> pane (a square of the normalised field
/// around a chosen centre: on axis, a design field point, or a point picked on a field diagram)
/// and its resolution, read from and written into the pane's settings. Other settings are kept.
/// </summary>
public static class ImageSimulationSettings
{
    /// <summary>The inscribed square's half-width (1/√2 of the normalised field), the engine's default.</summary>
    public static readonly double FullExtent = Math.Sqrt(0.5);
    /// <summary>Field extents offered: half-width of the square, in units of the normalised field.</summary>
    public static readonly IReadOnlyList<double> Extents = [FullExtent, 0.5, 0.25, 0.1, 0.05, 0.02];
    /// <summary>Output sizes offered (pixels per side).</summary>
    public static readonly IReadOnlyList<int> Sizes = [128, 256, 512, 1024];

    public static double FieldExtent(JsonNode? settings) => Json.Number(Json.At(settings, "optics", "field_extent")) ?? FullExtent;

    /// <summary>Output pixels per side (the width; the engine's default 64).</summary>
    public static int Size(JsonNode? settings) =>
        Json.NfArray(Json.At(settings, "size")) is [var w, ..] && double.IsFinite(w) ? (int)Math.Clamp(w, 0, 1 << 20) : 64;

    public static (double X, double Y) FieldCenter(JsonNode? settings) =>
        Json.NfArray(Json.At(settings, "optics", "field_center")) is [var x, var y] && double.IsFinite(x) && double.IsFinite(y) ? (x, y) : (0, 0);

    static JsonObject WithOptics(JsonObject settings, string key, JsonNode value)
    {
        var s = (JsonObject)settings.DeepClone();
        var optics = s["optics"]?.DeepClone() as JsonObject ?? new JsonObject();
        optics[key] = value;
        s["optics"] = optics;
        return s;
    }

    public static JsonObject WithFieldExtent(JsonObject settings, double extent) => WithOptics(settings, "field_extent", extent);

    /// <summary><c>optics.field_center = [hx, hy]</c>, rounded to 1e-4 so picked points read cleanly.</summary>
    public static JsonObject WithFieldCenter(JsonObject settings, (double X, double Y) c) =>
        WithOptics(settings, "field_center", new JsonArray(Round(c.X), Round(c.Y)));

    static double Round(double v)
    {
        var x = Math.Round(v * 1e4) / 1e4;
        return x == 0 ? 0 : x;
    }

    /// <summary><c>size = [n, n]</c>.</summary>
    public static JsonObject WithSize(JsonObject settings, int n)
    {
        var s = (JsonObject)settings.DeepClone();
        s["size"] = new JsonArray(n, n);
        return s;
    }

    /// <summary>"full field", "10 %", "2 %".</summary>
    public static string ExtentTitle(double e)
    {
        if (e >= 1) return Loc.S("Analysis_ImageSim_FullSquare");
        if (Math.Abs(e - FullExtent) < 1e-6) return Loc.S("Analysis_ImageSim_FullField");
        return Fmt.G4(e * 100) + " %";
    }

    /// <summary>
    /// A field-of-view choice with what it means on the sensor: "10 % · 2.40 mm wide, pixel 9.38 µm"
    /// (<paramref name="mmPerExtent"/>: image-plane half-width per unit of extent, from the last run).
    /// </summary>
    public static string ExtentLabel(double e, double? mmPerExtent, int size)
    {
        var t = ExtentTitle(e);
        if (mmPerExtent is double k && size > 0)
        {
            var width = 2 * k * e;
            t = Loc.F("Analysis_ImageSim_ExtentLabel", t, Fmt.Length(width), Fmt.Length(width / size));
        }
        return t;
    }

    /// <summary>Image-plane half-width (mm) per unit of field extent, from a run: its pixel pitch, width in pixels and field extent.</summary>
    public static double? MmPerExtent(double pixelX, int width, double? runExtent)
    {
        var w = pixelX * width;
        return double.IsFinite(w) && w > 0 && runExtent is double e && e > 0 ? w / 2 / e : null;
    }

    /// <summary>A choice of patch centre: the axis or a design field point, in normalised field coordinates.</summary>
    public sealed record FieldPoint(string Title, double X, double Y);

    /// <summary>The axis and the design's field points (normalised by the largest radial field), each listed once.</summary>
    public static IReadOnlyList<FieldPoint> FieldPoints(IEnumerable<(double X, double Y)>? fields)
    {
        var out_ = new List<FieldPoint> { new(Loc.S("Analysis_ImageSim_OnAxisTitle"), 0, 0) };
        var f = fields?.ToList() ?? [];
        var r = f.Count == 0 ? 0 : f.Max(p => Math.Sqrt(p.X * p.X + p.Y * p.Y));
        if (!(r > 0)) return out_;
        for (var k = 0; k < f.Count; k++)
        {
            double hx = f[k].X / r, hy = f[k].Y / r;
            if (out_.Any(p => Math.Abs(p.X - hx) < 1e-9 && Math.Abs(p.Y - hy) < 1e-9)) continue;
            out_.Add(new FieldPoint(Loc.F("Analysis_ImageSim_FieldPoint", k + 1, ImageSimulationData.Coordinates(hx, hy)), hx, hy));
        }
        return out_;
    }

    /// <summary>"on axis", "Field 3 (0.00, +1.00)" or "(+0.30, −0.45)".</summary>
    public static string CenterTitle((double X, double Y) c, IReadOnlyList<FieldPoint> points)
    {
        if (c.X == 0 && c.Y == 0) return Loc.S("Analysis_ImageSim_OnAxis");
        var p = points.FirstOrDefault(p => Math.Abs(p.X - c.X) < 1e-4 && Math.Abs(p.Y - c.Y) < 1e-4);
        return p?.Title ?? ImageSimulationData.Coordinates(c.X, c.Y);
    }

    /// <summary>The nearest of <paramref name="points"/> (and the axis) within <paramref name="within"/>, else <paramref name="p"/>.</summary>
    public static (double X, double Y) Snap((double X, double Y) p, IEnumerable<FieldPoint> points, double within)
    {
        (double X, double Y) best = p;
        var bestD = double.PositiveInfinity;
        foreach (var q in points.Select(q => (X: q.X, Y: q.Y)).Prepend((X: 0.0, Y: 0.0)))
        {
            var d = Math.Sqrt((q.X - p.X) * (q.X - p.X) + (q.Y - p.Y) * (q.Y - p.Y));
            if (d < bestD) { bestD = d; best = q; }
        }
        return bestD < within ? best : p;
    }

    /// <summary>Whether a corner of the patch lies outside the unit field disk.</summary>
    public static bool OutsideField((double X, double Y) c, double extent)
    {
        double fx = Math.Abs(c.X) + extent, fy = Math.Abs(c.Y) + extent;
        return Math.Sqrt(fx * fx + fy * fy) > 1 + 1e-9;
    }
}
