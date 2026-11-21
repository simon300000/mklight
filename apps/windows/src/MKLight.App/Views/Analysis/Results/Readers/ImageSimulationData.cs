using System.Text.Json.Nodes;
using MKLight.App.Support;

namespace MKLight.App.Views.Analysis.Results.Readers;

/// <summary>The sRGB transfer function (IEC 61966-2-1).</summary>
public static class Srgb
{
    public static double Encode(double linear)
    {
        if (!double.IsFinite(linear)) return 0;
        if (linear >= 1) return 1;
        var x = Math.Max(linear, 0);
        return x <= 0.0031308 ? 12.92 * x : Math.Min(1.055 * Math.Pow(x, 1 / 2.4) - 0.055, 1);
    }

    public static double Decode(double code)
    {
        var x = Math.Clamp(code, 0, 1);
        return x <= 0.04045 ? x / 12.92 : Math.Pow((x + 0.055) / 1.055, 2.4);
    }

    /// <summary>The 8-bit code value of a linear sample (clamped, rounded).</summary>
    public static byte Byte(double linear) => (byte)Math.Round(Encode(linear) * 255);
}

/// <summary>A linear-light raster, row 0 at the low-y edge (the engine's convention), 1 or 3 interleaved channels.</summary>
public sealed record LinearImage(int Width, int Height, int Channels, double[] Data)
{
    public bool IsValid => Width > 0 && Height > 0 && Channels is 1 or 3 && Data.Length == Width * Height * Channels;

    /// <summary>BGRA8 sRGB bytes, row 0 at the top (flipped), after scaling by <paramref name="gain"/> (2^exposure).</summary>
    public byte[] Bgra8(double gain = 1)
    {
        var px = new byte[Width * Height * 4];
        for (var y = 0; y < Height; y++)
        {
            var row = Height - 1 - y;
            for (var x = 0; x < Width; x++)
            {
                int s = (y * Width + x) * Channels, o = (row * Width + x) * 4;
                var r = Data[s] * gain;
                var g = Channels == 3 ? Data[s + 1] * gain : r;
                var b = Channels == 3 ? Data[s + 2] * gain : r;
                px[o] = Srgb.Byte(b);
                px[o + 1] = Srgb.Byte(g);
                px[o + 2] = Srgb.Byte(r);
                px[o + 3] = 255;
            }
        }
        return px;
    }

    /// <summary>The fraction of samples above 1 after <paramref name="gain"/> (clipped highlights).</summary>
    public double ClippedFraction(double gain = 1) => Data.Length == 0 ? 0 : (double)Data.Count(v => v * gain > 1) / Data.Length;

    /// <summary>A <c>{kind: buffer}</c> source from settings.</summary>
    public static LinearImage? FromSource(JsonNode? s)
    {
        if (Json.Str(Json.At(s, "kind")) != "buffer" || Json.Int(Json.At(s, "width")) is not int w || Json.Int(Json.At(s, "height")) is not int h
            || Json.Int(Json.At(s, "channels")) is not int c || Json.Arr(Json.At(s, "data")) is not { } d) return null;
        var img = new LinearImage(w, h, c, d.Select(v => Json.Number(v) ?? 0).ToArray());
        return img.IsValid ? img : null;
    }

    public static readonly (string Id, string Title)[] Patterns =
        [("grid", "Grid"), ("siemens_star", "Siemens star"), ("slanted_edge", "Slanted edge"), ("text_chart", "Text chart")];

    /// <summary>
    /// One of the engine's built-in test patterns at <paramref name="w"/> × <paramref name="h"/>
    /// with the engine's definition (<c>pattern_value</c> in the wave crate's image simulation).
    /// With <paramref name="n"/> &gt; 1 each pixel is the mean of n × n samples: the pattern's
    /// area average, as the engine evaluates it over each pixel.
    /// </summary>
    public static LinearImage? Pattern(string name, int w, int h, int n = 1)
    {
        if (w <= 0 || h <= 0 || n < 1 || !Patterns.Any(p => p.Id == name)) return null;
        int[][] font = [[17, 27, 21, 17, 17, 17, 17], [17, 18, 20, 24, 20, 18, 17], [16, 16, 16, 16, 16, 16, 31]];
        double Value(double px, double py)
        {
            double u = px / w, v = py / h;
            switch (name)
            {
                case "grid":
                    var fu = u * 8 % 1;
                    var fv = v * 8 % 1;
                    return fu < 0.1 || fv < 0.1 ? 0 : 1;
                case "siemens_star":
                    return Math.Cos(Math.Atan2(v - 0.5, u - 0.5) * 24) > 0 ? 1 : 0;
                case "slanted_edge":
                    return u > 0.5 + 0.1 * (v - 0.5) ? 1 : 0;
                default:
                    int x = Math.Min((int)px, w - 1), y = h - 1 - Math.Min((int)py, h - 1);
                    var band = Math.Min(y * 3 / h, 2);
                    var scale = new[] { 4, 2, 1 }[band];
                    int gx = x / scale, gy = y / scale;
                    var glyph = gx / 6 % 3;
                    return gy % 9 < 7 && gx % 6 < 5 && (font[glyph][gy % 9] & (1 << (4 - gx % 6))) != 0 ? 0 : 1;
            }
        }
        var data = new double[w * h];
        double inv = 1.0 / n, norm = 1.0 / (n * n);
        for (var y = 0; y < h; y++)
            for (var x = 0; x < w; x++)
            {
                double sum = 0;
                for (var j = 0; j < n; j++)
                    for (var i = 0; i < n; i++)
                        sum += Value(x + (i + 0.5) * inv, y + (j + 0.5) * inv);
                data[y * w + x] = sum * norm;
            }
        return new LinearImage(w, h, 1, data);
    }
}

/// <summary>The <c>image_simulation</c> result and the source it was run with.</summary>
public sealed class ImageSimulationData
{
    public sealed record Node(double Hx, double Hy, double IdealX, double IdealY, double RealX, double RealY, double RelativeIllumination)
    {
        /// <summary>Distortion of the node in percent of its ideal radius.</summary>
        public double DistortionPercent
        {
            get
            {
                var ri = Math.Sqrt(IdealX * IdealX + IdealY * IdealY);
                if (ri <= 1e-12) return double.NaN;
                return (Math.Sqrt(RealX * RealX + RealY * RealY) - ri) / ri * 100;
            }
        }
    }

    public required LinearImage Image { get; init; }
    public double PixelX { get; init; } = double.NaN;
    public double PixelY { get; init; } = double.NaN;
    public IReadOnlyList<Node> Nodes { get; init; } = [];
    public int? PsfSize { get; init; }
    public double PsfSpacing { get; init; } = double.NaN;
    /// <summary>The run's <c>source</c> setting (null when unknown).</summary>
    public JsonNode? Source { get; init; }

    public static ImageSimulationData? Read(ResultData d, JsonNode? settings = null)
    {
        var j = d.Json;
        var b = Json.BufferRef(d, Json.At(j, "image")) ?? d.Buffer("image");
        if (b is null || b.Shape.Count != 3) return null;
        var img = new LinearImage(b.Shape[1], b.Shape[0], b.Shape[2], Json.Doubles(b));
        if (!img.IsValid) return null;
        var px = Json.NfArray(Json.At(j, "pixel_size_mm"));
        return new ImageSimulationData
        {
            Image = img,
            PixelX = px is [var x, _] ? x : double.NaN,
            PixelY = px is [_, var y] ? y : double.NaN,
            Nodes = ReadNodes(j),
            PsfSize = Json.Int(Json.At(j, "psf_size")),
            PsfSpacing = Json.Nf(Json.At(j, "psf_spacing_mm")),
            Source = Json.At(settings, "source"),
        };
    }

    public static List<Node> ReadNodes(JsonNode? j) =>
        (Json.Arr(Json.At(j, "field_mapping")) ?? []).Select(n =>
            Json.NfArray(Json.At(n, "field")) is [var hx, var hy] && Json.NfArray(Json.At(n, "ideal_mm")) is [var ix, var iy]
            && Json.NfArray(Json.At(n, "real_mm")) is [var rx, var ry]
                ? new Node(hx, hy, ix, iy, rx, ry, Json.Nf(Json.At(n, "relative_illumination")))
                : null).OfType<Node>().ToList();

    /// <summary>The largest node distortion (signed, percent, by magnitude).</summary>
    public static double? MaxDistortionPercent(IEnumerable<Node> nodes) =>
        nodes.Select(n => n.DistortionPercent).Where(double.IsFinite).Select(v => (double?)v).MaxBy(v => Math.Abs(v!.Value));

    public static double? MinRelativeIllumination(IEnumerable<Node> nodes) =>
        nodes.Select(n => n.RelativeIllumination).Where(double.IsFinite).Select(v => (double?)v).Min();

    /// <summary>The run's patch centre: the middle of its field mapping nodes.</summary>
    public (double X, double Y)? FieldCenter =>
        Nodes.Count == 0 ? null : ((Nodes.Min(n => n.Hx) + Nodes.Max(n => n.Hx)) / 2, (Nodes.Min(n => n.Hy) + Nodes.Max(n => n.Hy)) / 2);

    /// <summary>The run's field extent: half the span of its nodes.</summary>
    public double? FieldExtent => Nodes.Count < 2 || Nodes.Max(n => n.Hx) <= Nodes.Min(n => n.Hx) ? null : (Nodes.Max(n => n.Hx) - Nodes.Min(n => n.Hx)) / 2;

    public string SourceKind => Json.Str(Json.At(Source, "kind")) ?? (Source is null ? "unknown" : "pattern");

    /// <summary>
    /// The source at the output size: a pattern drawn with the engine's definition and
    /// area-averaged as the engine does, a buffer the run's own; null for a file or unknown source.
    /// </summary>
    public LinearImage? SourceImage() => SourceKind switch
    {
        "buffer" => LinearImage.FromSource(Source),
        "pattern" => LinearImage.Pattern(Json.Str(Json.At(Source, "pattern")) ?? "grid", Image.Width, Image.Height, 4),
        _ => null,
    };

    /// <summary>"Siemens star pattern".</summary>
    static string PatternTitle(string id) => id switch
    {
        "grid" => Loc.S("Results_ImageSim_PatternGrid"),
        "siemens_star" => Loc.S("Results_ImageSim_PatternSiemensStar"),
        "slanted_edge" => Loc.S("Results_ImageSim_PatternSlantedEdge"),
        "text_chart" => Loc.S("Results_ImageSim_PatternTextChart"),
        _ => Loc.S("Results_ImageSim_Pattern"),
    };

    public string SourceTitle => SourceKind switch
    {
        "buffer" => Loc.S("Results_ImageSim_ImportedImage"),
        "path" => Path.GetFileName(Json.Str(Json.At(Source, "path")) ?? Loc.S("Results_ImageSim_Image")),
        "pattern" => PatternTitle(Json.Str(Json.At(Source, "pattern")) ?? "grid"),
        _ => Loc.S("Results_ImageSim_Source"),
    };

    /// <summary>"(0.00, +0.71)": a normalised field point.</summary>
    public static string Coordinates(double x, double y)
    {
        static string F(double v) => Fmt.Signed(v, 2).Trim().Replace('-', '−');
        return "(" + F(x) + ", " + F(y) + ")";
    }

    /// <summary>Width / height of the image in mm (pixels may be rectangular).</summary>
    public double Aspect
    {
        get
        {
            var w = Image.Width * (PixelX > 0 ? PixelX : 1);
            var h = Image.Height * (PixelY > 0 ? PixelY : 1);
            return w / Math.Max(h, 1e-12);
        }
    }
}
