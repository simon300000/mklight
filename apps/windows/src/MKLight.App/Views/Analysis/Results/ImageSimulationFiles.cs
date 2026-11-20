using System.Text.Json.Nodes;
using Microsoft.UI.Xaml;
using Microsoft.Windows.Storage.Pickers;
using MKLight.App.ViewModels;
using MKLight.App.Views.Analysis.Results.Readers;
using MKLight.App.Views.Analysis.Settings;
using Windows.Graphics.Imaging;
using Windows.Storage;

namespace MKLight.App.Views.Analysis.Results;

/// <summary>
/// Image files as the image simulation's source: choosing one (the file picker), decoding it
/// (any format Windows Imaging reads, EXIF orientation applied, colour-managed to sRGB, at most
/// <see cref="ImageSourceImport.MaxDecodeSide"/> px) and setting a pane's <c>source</c> to its
/// linear-light samples (<see cref="ImageSourceImport"/>).
/// </summary>
public static class ImageSimulationFiles
{
    /// <summary>A decoded picture: 8-bit sRGB BGRA, top row first.</summary>
    public sealed record Decoded(byte[] Bgra, int Width, int Height, bool Grey);

    /// <summary>Each file's decoded picture, with its modification time and size: a file that changed is decoded again.</summary>
    static readonly Dictionary<string, (DateTime Modified, long Size, Decoded Image)> cache = [];

    public static async Task<Decoded?> DecodeAsync(string path)
    {
        try
        {
            var info = new FileInfo(path);
            if (!info.Exists) return null;
            lock (cache)
                if (cache.TryGetValue(info.FullName, out var c) && c.Modified == info.LastWriteTimeUtc && c.Size == info.Length) return c.Image;
            var file = await StorageFile.GetFileFromPathAsync(info.FullName);
            using var stream = await file.OpenReadAsync();
            var dec = await BitmapDecoder.CreateAsync(stream);
            var s = Math.Min(1, (double)ImageSourceImport.MaxDecodeSide / Math.Max(dec.PixelWidth, dec.PixelHeight));
            uint sw = (uint)Math.Max(1, Math.Round(dec.PixelWidth * s)), sh = (uint)Math.Max(1, Math.Round(dec.PixelHeight * s));
            var t = new BitmapTransform { ScaledWidth = sw, ScaledHeight = sh, InterpolationMode = BitmapInterpolationMode.Fant };
            var pd = await dec.GetPixelDataAsync(BitmapPixelFormat.Bgra8, BitmapAlphaMode.Ignore, t, ExifOrientationMode.RespectExifOrientation, ColorManagementMode.ColorManageToSRgb);
            var bytes = pd.DetachPixelData();
            // A quarter turn (EXIF) swaps the sides.
            var turned = dec.OrientedPixelWidth != dec.PixelWidth;
            int w = (int)(turned ? sh : sw), h = (int)(turned ? sw : sh);
            if (bytes.Length != w * h * 4) return null;
            var grey = dec.BitmapPixelFormat is BitmapPixelFormat.Gray8 or BitmapPixelFormat.Gray16;
            var d = new Decoded(bytes, w, h, grey);
            lock (cache)
            {
                // The last two files only: a decoded picture can be 16 MB.
                if (!cache.ContainsKey(info.FullName) && cache.Count >= 2) cache.Remove(cache.Keys.First());
                cache[info.FullName] = (info.LastWriteTimeUtc, info.Length, d);
            }
            return d;
        }
        catch (Exception)
        {
            return null; // not an image Windows can read
        }
    }

    /// <summary>The settings with <paramref name="image"/> (from <paramref name="path"/>) as the source at the settings' output size.</summary>
    static JsonObject? WithImage(JsonObject settings, string path, Decoded image, double? aspect)
    {
        var (w, h) = ImageSourceImport.TargetSize(Json.At(settings, "size"));
        return ImageSourceImport.Linear(image.Bgra, image.Width, image.Height, image.Grey, w, h, aspect) is { } lin
            ? ImageSourceImport.WithImage(settings, path, lin, aspect)
            : null;
    }

    static double? Aspect(AnalysisPaneModel pane) =>
        pane.Analysis == "image_simulation" ? ImageSourceImport.FieldAspect(pane.Output?.Json) : null;

    /// <summary>Uses an image file as the pane's source; false when it cannot be read.</summary>
    public static async Task<bool> UseAsync(AnalysisPaneModel pane, string path)
    {
        path = Path.GetFullPath(path);
        if (await DecodeAsync(path) is not { } image) return false;
        var aspect = Aspect(pane);
        var lin = await Task.Run(() => WithImage(pane.Settings, path, image, aspect));
        if (lin is null) return false;
        pane.SetSettings(lin);
        return true;
    }

    /// <summary>
    /// <paramref name="settings"/> with the imported file's samples derived again when they must be
    /// (another output size or field aspect; after a project opened): null when nothing changes,
    /// the settings with the default pattern when the file cannot be read.
    /// </summary>
    public static async Task<(JsonObject Settings, bool Missing)?> RefreshedAsync(JsonObject settings, double? aspect)
    {
        if (!ImageSourceImport.NeedsRefresh(settings, aspect) || Json.Str(Json.At(settings, ImageSourceImport.FileKey)) is not string path) return null;
        if (await DecodeAsync(path) is { } image && await Task.Run(() => WithImage(settings, path, image, aspect)) is { } s) return (s, false);
        if (Json.Str(Json.At(settings, "source", "kind")) != "buffer") return null;
        var gone = (JsonObject)settings.DeepClone();
        gone.Remove("source");
        gone.Remove(ImageSourceImport.AspectKey);
        return (gone, true);
    }

    /// <summary>Re-derives an imported file's samples when the pane needs them again; false when the file cannot be read.</summary>
    public static async Task<bool> RefreshAsync(AnalysisPaneModel pane)
    {
        if (await RefreshedAsync(pane.Settings, Aspect(pane)) is not { } r) return true;
        pane.SetSettings(r.Settings);
        return !r.Missing;
    }

    /// <summary>The pane's settings at another output size, an imported file resampled to it in the same step.</summary>
    public static async Task<JsonObject> WithSizeAsync(AnalysisPaneModel pane, int n)
    {
        var s = ImageSimulationSettings.WithSize(pane.Settings, n);
        if (await RefreshedAsync(s, Aspect(pane)) is not { } r) return ImageSimulationSettings.WithSize(pane.Settings, n);
        // Settings edited during the decode stay: only the resampled source comes from it.
        var now = ImageSimulationSettings.WithSize(pane.Settings, n);
        if (r.Settings["source"] is { } source) now["source"] = source.DeepClone();
        return now;
    }

    /// <summary>Asks for an image file in the window that shows <paramref name="element"/>.</summary>
    public static async Task<string?> PickAsync(UIElement element)
    {
        if (SettingsServices.Window(element) is not { } host) return null;
        var picker = new FileOpenPicker(host.WindowId) { SuggestedStartLocation = PickerLocationId.PicturesLibrary };
        foreach (var ext in ImageSourceImport.FileTypes) picker.FileTypeFilter.Add(ext);
        var file = await picker.PickSingleFileAsync();
        return file?.Path;
    }
}
