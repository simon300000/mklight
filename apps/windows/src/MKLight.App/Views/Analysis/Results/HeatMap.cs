using Microsoft.Graphics.Canvas;
using MKLight.App.Plotting;
using MKLight.App.Support;
using MKLight.App.Views.Analysis.Results.Readers;
using Windows.Foundation;
using Windows.UI;

namespace MKLight.App.Views.Analysis.Results;

/// <summary>
/// A 2-D grid drawn as a map with a colour bar (<see cref="HeatScale"/>:
/// diverging blue–red about 0 for signed data, the perceptual ramp
/// otherwise). Row 0 is drawn at the bottom (low y, as the engine's maps
/// and images). Cells without a value are left transparent.
/// </summary>
public sealed class HeatMap
{
    readonly IResultArray _data;
    readonly int _rows, _cols;
    readonly HeatScale _scale;
    CanvasBitmap? _bitmap;
    (CanvasDevice Device, bool Dark)? _key;

    public HeatMap(IResultArray data)
    {
        _data = data;
        _rows = data.Shape[0];
        _cols = data.Count / Math.Max(_rows, 1);
        var (lo, hi, _) = GenericData.Range(data);
        _scale = HeatScale.For(lo, hi);
    }

    Color ColorOf(double v, bool dark)
    {
        var t = _scale.Position(v);
        if (double.IsNaN(t)) return Color.FromArgb(0, 0, 0, 0);
        return _scale.Diverging ? Palette.Diverging(t, dark) : Palette.Perceptual(t);
    }

    CanvasBitmap Bitmap(CanvasDrawingSession ds, bool dark)
    {
        if (_bitmap is not null && _key is { } k && k.Device == ds.Device && k.Dark == dark) return _bitmap;
        _bitmap?.Dispose();
        var colors = new Color[_rows * _cols];
        for (var r = 0; r < _rows; r++)
        {
            var dst = (_rows - 1 - r) * _cols; // row 0 at the bottom
            for (var c = 0; c < _cols; c++) colors[dst + c] = ColorOf(_data[r * _cols + c], dark);
        }
        _bitmap = CanvasBitmap.CreateFromColors(ds, colors, _cols, _rows);
        _key = (ds.Device, dark);
        return _bitmap;
    }

    /// <summary>Draws the map (aspect kept) and its colour bar into <paramref name="bounds"/>.</summary>
    public void Draw(CanvasDrawingSession ds, Rect bounds, PlotStyle style)
    {
        if (_rows < 1 || _cols < 1) return;
        const double barW = 10, barGap = 10, labelW = 56;
        var avail = new Size(Math.Max(bounds.Width - barW - barGap - labelW, 10), Math.Max(bounds.Height, 10));
        var aspect = (double)_cols / _rows;
        var w = Math.Min(avail.Width, avail.Height * aspect);
        var h = w / aspect;
        var map = new Rect(bounds.X, bounds.Y + (bounds.Height - h) / 2, w, h);
        var interp = _cols >= w && _rows >= h ? CanvasImageInterpolation.Linear : CanvasImageInterpolation.NearestNeighbor;
        ds.DrawImage(Bitmap(ds, style.Dark), map, new Rect(0, 0, _cols, _rows), 1, interp);
        ds.DrawRectangle(map, style.Grid, 1);

        // Colour bar: the ramp from the scale's bottom to its top, with its end values.
        var bar = new Rect(map.Right + barGap, map.Y, barW, map.Height);
        const int steps = 64;
        for (var i = 0; i < steps; i++)
        {
            var t = (double)i / (steps - 1);
            var v = _scale.Min + t * (_scale.Max - _scale.Min);
            var y0 = bar.Bottom - (i + 1) * bar.Height / steps;
            ds.FillRectangle(new Rect(bar.X, y0, bar.Width, bar.Height / steps + 0.5), ColorOf(v, style.Dark));
        }
        ds.DrawRectangle(bar, style.Grid, 1);
        PlotText.Draw(ds, Fmt.Short(_scale.Max), bar.Right + 4, bar.Y + 5, style.Label, anchor: TextAnchor.Start);
        PlotText.Draw(ds, Fmt.Short(_scale.Min), bar.Right + 4, bar.Bottom - 5, style.Label, anchor: TextAnchor.Start);
        if (_scale.Diverging) PlotText.Draw(ds, "0", bar.Right + 4, bar.Y + bar.Height / 2, style.Label, anchor: TextAnchor.Start);
    }
}
