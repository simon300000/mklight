using System.Text.Json.Nodes;
using MKLight.App.Views.Analysis.Results.Readers;

namespace MKLight.App.Views.Analysis.Results;

/// <summary>
/// A user image as the image simulation's source (no WinUI, unit-tested). The decoded picture
/// (8-bit sRGB, BGRA, top row first) is cropped to fill the field's aspect, resampled in linear
/// light (area average) to the output size, at most <see cref="MaxSide"/> px a side, and sent as
/// a <c>{kind: buffer}</c> source with row 0 at the low-y edge, the engine's convention for
/// buffers and its output. The pane keeps the file's path in <see cref="FileKey"/> (an app-only
/// setting the engine never sees), so the samples can be derived again for another output size
/// or when a project opens.
/// </summary>
public static class ImageSourceImport
{
    /// <summary>The app-only setting holding the imported file's path.</summary>
    public const string FileKey = "source_file";
    /// <summary>The app-only setting recording the field aspect the samples were cropped to.</summary>
    public const string AspectKey = "source_field_aspect";
    public const int MaxSide = 512;
    /// <summary>Images are decoded at most this large a side before the linear-light resample.</summary>
    public const int MaxDecodeSide = 2048;

    static readonly string[] Extensions = [".png", ".jpg", ".jpeg", ".jpe", ".bmp", ".gif", ".tif", ".tiff", ".jxr", ".wdp", ".heic", ".heif", ".webp", ".ico"];

    /// <summary>The image file types the picker offers and a drop accepts.</summary>
    public static IReadOnlyList<string> FileTypes => Extensions;

    public static bool IsImageFile(string path) => Extensions.Contains(Path.GetExtension(path).ToLowerInvariant());

    /// <summary>
    /// The size the source is resampled to: the run's <c>size</c> (default 64 × 64), scaled down
    /// to at most <see cref="MaxSide"/> on its longer side.
    /// </summary>
    public static (int Width, int Height) TargetSize(JsonNode? size)
    {
        var a = Json.NfArray(size) ?? [];
        int w = 64, h = 64;
        if (a.Length == 2 && double.IsFinite(a[0]) && double.IsFinite(a[1]))
        {
            w = Math.Max((int)Math.Clamp(a[0], 0, 1 << 20), 2);
            h = Math.Max((int)Math.Clamp(a[1], 0, 1 << 20), 2);
        }
        var s = Math.Min(1, (double)MaxSide / Math.Max(w, h));
        return (Math.Max((int)Math.Round(w * s, MidpointRounding.AwayFromZero), 2), Math.Max((int)Math.Round(h * s, MidpointRounding.AwayFromZero), 2));
    }

    /// <summary>The field's aspect (width / height in mm) of an <c>image_simulation</c> result.</summary>
    public static double? FieldAspect(JsonNode? result)
    {
        if (Json.NfArray(Json.At(result, "size")) is not [var sw, var sh] || Json.NfArray(Json.At(result, "pixel_size_mm")) is not [var px, var py]) return null;
        var a = sw * px / (sh * py);
        return double.IsFinite(a) && a > 0 ? a : null;
    }

    /// <summary>
    /// The centred part of an <paramref name="iw"/> × <paramref name="ih"/> image with the
    /// aspect <paramref name="aspect"/> (width / height): the longer side is cropped.
    /// </summary>
    public static (double X, double Y, double Width, double Height) Crop(int iw, int ih, double aspect)
    {
        var cw = Math.Min(iw, ih * aspect);
        var ch = cw / aspect;
        return ((iw - cw) / 2, (ih - ch) / 2, cw, ch);
    }

    /// <summary>
    /// The image cropped to <paramref name="aspect"/> (width / height in mm; default the pixel
    /// grid's) and area-averaged in linear light to <paramref name="width"/> × <paramref name="height"/>,
    /// row 0 at the bottom; one channel for grey images, else three (R, G, B interleaved).
    /// </summary>
    /// <param name="bgra">8-bit sRGB pixels, B G R A, top row first, <paramref name="iw"/> × <paramref name="ih"/>.</param>
    public static LinearImage? Linear(ReadOnlySpan<byte> bgra, int iw, int ih, bool grey, int width, int height, double? aspect = null)
    {
        if (iw <= 0 || ih <= 0 || width <= 0 || height <= 0 || bgra.Length < iw * ih * 4) return null;
        var a = aspect is double v && double.IsFinite(v) && v > 0 ? v : (double)width / height;
        var (cx, cy, cw, ch) = Crop(iw, ih, a);
        var c = grey ? 1 : 3;

        // sRGB code values to linear light.
        var lut = new double[256];
        for (var i = 0; i < 256; i++) lut[i] = Srgb.Decode(i / 255.0);
        // B G R A: the channels' byte offsets in R, G, B order.
        int[] channel = grey ? [2] : [2, 1, 0];

        // Area average, separable: each output pixel covers a box of the crop. First along
        // each source row (in linear light), then down the columns.
        var wx = BoxWeights(cx, cw, width, iw);
        var wy = BoxWeights(cy, ch, height, ih);
        var rows = new double[ih * width * c];
        for (var y = 0; y < ih; y++)
            for (var x = 0; x < width; x++)
                foreach (var (i, w) in wx[x])
                    for (var k = 0; k < c; k++) rows[(y * width + x) * c + k] += w * lut[bgra[(y * iw + i) * 4 + channel[k]]];
        var data = new double[width * height * c];
        for (var y = 0; y < height; y++)
        {
            // Source rows run top down; the engine's row 0 is the bottom.
            var o = (height - 1 - y) * width * c;
            foreach (var (j, w) in wy[y])
                for (var x = 0; x < width * c; x++) data[o + x] += w * rows[j * width * c + x];
        }
        return new LinearImage(width, height, c, data);
    }

    /// <summary>For each of <paramref name="n"/> output pixels over [start, start + span): the source pixels it covers and their normalised weights.</summary>
    static List<(int Index, double Weight)>[] BoxWeights(double start, double span, int n, int count)
    {
        var step = span / n;
        var out_ = new List<(int, double)>[n];
        for (var o = 0; o < n; o++)
        {
            double lo = start + o * step, hi = lo + step;
            var list = new List<(int, double)>();
            var sum = 0.0;
            for (var i = Math.Max((int)Math.Floor(lo), 0); i < Math.Min((int)Math.Ceiling(hi), count); i++)
            {
                var w = Math.Min(hi, i + 1) - Math.Max(lo, i);
                if (w <= 0) continue;
                list.Add((i, w));
                sum += w;
            }
            if (sum <= 0) list.Add((Math.Clamp((int)lo, 0, count - 1), sum = 1));
            for (var k = 0; k < list.Count; k++) list[k] = (list[k].Item1, list[k].Item2 / sum);
            out_[o] = list;
        }
        return out_;
    }

    /// <summary>The image source as the engine's settings expect it.</summary>
    public static JsonObject SourceJson(LinearImage img)
    {
        var data = new JsonArray();
        foreach (var v in img.Data) data.Add(v);
        return new JsonObject
        {
            ["kind"] = "buffer", ["width"] = img.Width, ["height"] = img.Height, ["channels"] = img.Channels, ["data"] = data,
        };
    }

    /// <summary>The settings with an imported file's samples as the source (other settings kept).</summary>
    public static JsonObject WithImage(JsonObject settings, string path, LinearImage img, double? aspect)
    {
        var s = (JsonObject)settings.DeepClone();
        s[FileKey] = path;
        s["source"] = SourceJson(img);
        if (aspect is double a) s[AspectKey] = a; else s.Remove(AspectKey);
        return s;
    }

    /// <summary>The settings with a built-in pattern as the source (an imported file forgotten).</summary>
    public static JsonObject WithPattern(JsonObject settings, string pattern)
    {
        var s = (JsonObject)settings.DeepClone();
        s.Remove(FileKey);
        s.Remove(AspectKey);
        s["source"] = new JsonObject { ["kind"] = "pattern", ["pattern"] = pattern };
        return s;
    }

    /// <summary>
    /// The settings to save in a project: an imported image's samples are dropped (its path
    /// stays), so the file is decoded again when the project opens.
    /// </summary>
    public static JsonObject Persistable(JsonObject settings)
    {
        if (!settings.ContainsKey(FileKey) || Json.Str(Json.At(settings, "source", "kind")) != "buffer") return settings;
        var s = (JsonObject)settings.DeepClone();
        s.Remove("source");
        s.Remove(AspectKey);
        return s;
    }

    /// <summary>
    /// Whether an imported file's samples must be derived again: missing, of another output size,
    /// or cropped to another field aspect than <paramref name="aspect"/>.
    /// </summary>
    public static bool NeedsRefresh(JsonObject settings, double? aspect)
    {
        if (Json.Str(Json.At(settings, FileKey)) is null) return false;
        var (w, h) = TargetSize(Json.At(settings, "size"));
        var src = Json.At(settings, "source");
        if (Json.Str(Json.At(src, "kind")) != "buffer" || Json.Int(Json.At(src, "width")) != w || Json.Int(Json.At(src, "height")) != h) return true;
        var was = Json.Number(Json.At(settings, AspectKey));
        return aspect is double a ? was is not double b || Math.Abs(a - b) > 1e-9 * a : was is not null;
    }

    /// <summary>The imported file's name, or null when the source is not an imported file.</summary>
    public static string? FileName(JsonNode? settings) =>
        Json.Str(Json.At(settings, FileKey)) is string p && Json.Str(Json.At(settings, "source", "kind")) == "buffer" ? Path.GetFileName(p) : null;
}
