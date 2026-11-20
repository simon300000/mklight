using Microsoft.Graphics.Canvas;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using MKLight.App.Plotting;
using MKLight.App.Support;
using MKLight.App.Views.Analysis.Results.Readers;
using Windows.Foundation;
using Windows.UI;

namespace MKLight.App.Views.Analysis.Results;

/// <summary>
/// <c>ray_grid</c>: each field's image-surface hits as a density image, not overplotted points (a fine grid
/// launches hundreds of thousands of rays). Each bin's colour mixes the wavelengths by their counts; its
/// opacity follows the log of the count, so sparse halos stay visible beside a dense core. One cell per field,
/// centred on the hits' centroid, each on its own scale (±half-width under it).
/// </summary>
public sealed partial class RayGridView : UserControl
{
    const double MinCell = 140, MaxCell = 360, Gap = 10, TitleHeight = 18, CaptionHeight = 16;

    readonly RayGridData _d;
    readonly ScrollViewer _scroll;
    readonly PlotCanvas _canvas;
    (int Columns, double Cell) _layout = (1, MinCell);
    readonly Dictionary<int, (CanvasBitmap Bitmap, CanvasDevice Device, bool Dark, int Bins)> _bitmaps = [];

    public RayGridView(RayGridData d)
    {
        _d = d;
        IsTabStop = false;
        _canvas = new PlotCanvas(Draw) { VerticalAlignment = VerticalAlignment.Top };
        _scroll = new ScrollViewer { Content = _canvas, HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled, VerticalScrollBarVisibility = ScrollBarVisibility.Auto };
        _scroll.SizeChanged += (_, e) => Fit(e.NewSize.Width);

        var entries = d.WavelengthsUm.Select(um => new LegendEntry(Fmt.Wavelength(um), SeriesColor.Wavelength(um), LegendMark.Dot)).ToList();
        var header = new Grid { ColumnSpacing = 8 };
        header.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        header.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        header.Children.Add(new Legend(entries) { VerticalAlignment = VerticalAlignment.Center });
        var count = Kit.Text(Kit.Inv($"{d.Arrived:N0} / {d.Launched:N0} arrived"), "NumberText", 11);
        count.VerticalAlignment = VerticalAlignment.Center;
        count.Opacity = 0.75;
        ToolTipService.SetToolTip(count, Kit.Inv($"{d.PupilCount:N0} pupil rays per field and wavelength"));
        Grid.SetColumn(count, 1);
        header.Children.Add(count);

        var note = Kit.Caption("Ray density on the image surface, centred on each field's centroid; the more opaque, the more rays.", wrap: true);
        var root = new Grid { Padding = new Thickness(8, 4, 8, 6), RowSpacing = 6 };
        root.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        root.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) });
        root.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        root.Children.Add(header);
        Grid.SetRow(_scroll, 1);
        root.Children.Add(_scroll);
        Grid.SetRow(note, 2);
        root.Children.Add(note);
        Content = root;

        Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(_canvas,
            Kit.Inv($"Ray grid density, {d.Fields.Count} fields, {d.Arrived:N0} of {d.Launched:N0} rays arrived"));
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetHelpText(_canvas,
            string.Join(", ", d.Fields.Select(f => $"{f.Title} ±{d.Length(f.HalfWidth)}")));
        Unloaded += (_, _) => { foreach (var b in _bitmaps.Values) b.Bitmap.Dispose(); _bitmaps.Clear(); };
    }

    void Fit(double width)
    {
        var columns = Math.Clamp((int)((width + Gap) / (MinCell + Gap)), 1, Math.Max(_d.Fields.Count, 1));
        var cell = Math.Min(Math.Floor((width - (columns - 1) * Gap) / columns), MaxCell);
        _layout = (columns, Math.Max(cell, 60));
        var rows = (_d.Fields.Count + columns - 1) / columns;
        _canvas.Height = rows * (TitleHeight + _layout.Cell + CaptionHeight) + (rows - 1) * Gap;
        _canvas.Invalidate();
    }

    /// <summary>The field's density image at <paramref name="bins"/>², rebuilt for a new device, theme or size.</summary>
    CanvasBitmap Bitmap(CanvasDrawingSession ds, int f, int bins, bool dark)
    {
        if (_bitmaps.TryGetValue(f, out var c) && c.Device == ds.Device && c.Dark == dark && c.Bins == bins) return c.Bitmap;
        if (_bitmaps.Remove(f, out var old)) old.Bitmap.Dispose();
        var counts = RayGridData.Bin(_d.Fields[f], bins);
        var colors = new Color[bins * bins];
        var wl = _d.WavelengthsUm.Select(um => Palette.Wavelength(um, dark)).ToArray();
        var max = 1;
        for (var i = 0; i < colors.Length; i++)
        {
            var total = 0;
            for (var w = 0; w < counts.Length; w++) total += counts[w][i];
            max = Math.Max(max, total);
        }
        var logMax = Math.Log(1 + max);
        for (var i = 0; i < colors.Length; i++)
        {
            double r = 0, g = 0, b = 0;
            var total = 0;
            for (var w = 0; w < counts.Length; w++)
            {
                var n = counts[w][i];
                if (n == 0 || w >= wl.Length) continue;
                r += n * wl[w].R; g += n * wl[w].G; b += n * wl[w].B;
                total += n;
            }
            if (total == 0) continue;
            // A floor so a single ray shows (higher on a light background, where pale colours fade).
            var floor = dark ? 0.22 : 0.38;
            var alpha = floor + (1 - floor) * Math.Log(1 + total) / logMax;
            colors[i] = Color.FromArgb((byte)Math.Round(255 * alpha), (byte)(r / total), (byte)(g / total), (byte)(b / total));
        }
        var bitmap = CanvasBitmap.CreateFromColors(ds, colors, bins, bins);
        _bitmaps[f] = (bitmap, ds.Device, dark, bins);
        return bitmap;
    }

    void Draw(CanvasDrawingSession ds, Size size, PlotStyle style)
    {
        var (columns, cell) = _layout;
        for (var f = 0; f < _d.Fields.Count; f++)
        {
            var field = _d.Fields[f];
            double x = f % columns * (cell + Gap), y = f / columns * (TitleHeight + cell + CaptionHeight + Gap);
            PlotText.Draw(ds, field.Title, x + cell / 2, y + TitleHeight / 2, style.Title, 11, bold: true);
            var box = new Rect(x, y + TitleHeight, cell, cell);
            ds.DrawRectangle(box, style.Grid, 1);
            // Cross hairs through the centroid.
            ds.DrawLine((float)(x + cell / 2), (float)box.Top, (float)(x + cell / 2), (float)box.Bottom, style.Grid.WithAlpha(0.5), 1);
            ds.DrawLine((float)x, (float)(box.Top + cell / 2), (float)(x + cell), (float)(box.Top + cell / 2), style.Grid.WithAlpha(0.5), 1);
            if (field.Arrived == 0)
            {
                PlotText.Draw(ds, "No rays arrived", x + cell / 2, box.Top + cell / 2, style.Label, 10);
                continue;
            }
            // About two device pixels per bin, so single rays show and dense cores do not saturate one pixel.
            var bins = Math.Clamp((int)(cell * ds.Dpi / 96 / 2), 32, 320);
            ds.DrawImage(Bitmap(ds, f, bins, style.Dark), box, new Rect(0, 0, bins, bins), 1, CanvasImageInterpolation.NearestNeighbor);
            PlotText.Draw(ds, "±" + _d.Length(field.HalfWidth * 1.04), x + cell / 2, box.Bottom + CaptionHeight / 2 + 1, style.Label, 10);
        }
    }
}
