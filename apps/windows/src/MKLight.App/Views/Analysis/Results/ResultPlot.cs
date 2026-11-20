using System.Numerics;
using Microsoft.Graphics.Canvas;
using Microsoft.Graphics.Canvas.Geometry;
using MKLight.App.Plotting;
using MKLight.App.Support;
using MKLight.App.Views.Analysis.Results.Readers;
using Windows.Foundation;
using Windows.UI;

namespace MKLight.App.Views.Analysis.Results;

/// <summary>Plot text with <c>_</c> subscripts (<c>V_d</c>, <c>P_g,F</c>, <c>S_III</c>) drawn as real subscripts.</summary>
static class SubText
{
    public static Size Measure(ICanvasResourceCreator rc, string label, float size = PlotStyle.LabelSize, bool bold = false)
    {
        double w = 0, h = 0;
        foreach (var (t, sub) in Kit.Pieces(label))
        {
            var m = PlotText.Measure(rc, t, sub ? size * 0.72f : size, bold);
            w += m.Width;
            h = Math.Max(h, m.Height);
        }
        return new Size(w, h);
    }

    /// <summary>Draw anchored horizontally at x, vertically centred on y (rotated about that point).</summary>
    public static void Draw(CanvasDrawingSession ds, string label, double x, double y, Color color, float size = PlotStyle.LabelSize,
        TextAnchor anchor = TextAnchor.Middle, bool bold = false, double rotation = 0)
    {
        if (string.IsNullOrEmpty(label)) return;
        if (!label.Contains('_')) { PlotText.Draw(ds, label, x, y, color, size, anchor, bold, rotation); return; }
        var w = Measure(ds, label, size, bold).Width;
        var old = ds.Transform;
        ds.Transform = Matrix3x2.CreateRotation((float)rotation) * Matrix3x2.CreateTranslation((float)x, (float)y) * old;
        var cx = anchor switch { TextAnchor.Start => 0, TextAnchor.Middle => -w / 2, _ => -w };
        foreach (var (t, sub) in Kit.Pieces(label))
        {
            var s = sub ? size * 0.72f : size;
            var m = PlotText.Measure(ds, t, s, bold);
            PlotText.Draw(ds, t, cx, sub ? size * 0.32 : 0, color, s, TextAnchor.Start, bold);
            cx += m.Width;
        }
        ds.Transform = old;
    }
}

/// <summary>
/// A cartesian frame: light grid (or tick marks only, for maps), few ticks, axis titles with
/// subscripts, an optional reversed x axis (V_d decreasing to the right). Returns the data
/// rectangle; <see cref="X"/> and <see cref="Y"/> map data to it.
/// </summary>
sealed class Frame
{
    public double XMin, XMax = 1, YMin, YMax = 1;
    public bool ReverseX;
    public string? XTitle, YTitle;
    /// <summary>Grid lines across the plot (charts); false draws short tick marks only (maps).</summary>
    public bool Grid = true;
    public bool ZeroLine;
    public double XTickSpacing = 70, YTickSpacing = 40;
    /// <summary>The fewest ticks per axis (maps: 3, so a small map still reads).</summary>
    public int MinTicks = 2;
    public IReadOnlyList<double>? XTicks;
    public Func<double, double, string> XLabel = Ticks.Label, YLabel = Ticks.Label;
    public Rect Plot { get; private set; }

    public float X(double v) => (float)(ReverseX
        ? Plot.Right - (v - XMin) / (XMax - XMin) * Plot.Width
        : Plot.X + (v - XMin) / (XMax - XMin) * Plot.Width);
    public float Y(double v) => (float)(Plot.Bottom - (v - YMin) / (YMax - YMin) * Plot.Height);
    public double DataX(double px) => ReverseX ? XMin + (Plot.Right - px) / Plot.Width * (XMax - XMin) : XMin + (px - Plot.X) / Plot.Width * (XMax - XMin);
    public double DataY(double py) => YMin + (Plot.Bottom - py) / Plot.Height * (YMax - YMin);

    /// <summary>The left margin the y labels and title need.</summary>
    public double LeftMargin(ICanvasResourceCreator rc, double height)
    {
        var (ticks, step) = YTicks(height);
        var w = ticks.Select(v => PlotText.Measure(rc, YLabel(v, step)).Width).DefaultIfEmpty(0).Max();
        return w + 7 + (YTitle is null ? 0 : 16);
    }

    public const double BottomMargin = 18;
    public double Bottom => BottomMargin + (XTitle is null ? 0 : 16);

    (List<double> Ticks, double Step) YTicks(double height)
    {
        var t = Ticks.Values(YMin, YMax, Math.Max(MinTicks, (int)(height / YTickSpacing)));
        return (t, t.Count > 1 ? t[1] - t[0] : 1);
    }

    /// <summary>Draws into <paramref name="plot"/> (the data rectangle, margins outside it).</summary>
    public Rect Draw(CanvasDrawingSession ds, Rect plot, PlotStyle style)
    {
        Plot = plot;
        if (plot.Width < 10 || plot.Height < 10 || !(XMax > XMin) || !(YMax > YMin)) return Rect.Empty;
        var xs = XTicks?.Where(v => v >= XMin - 1e-12 && v <= XMax + 1e-12).ToList() ?? Ticks.Values(XMin, XMax, Math.Max(MinTicks, (int)(plot.Width / XTickSpacing)));
        var xStep = xs.Count > 1 ? Math.Abs(xs[1] - xs[0]) : 1;
        var (ys, yStep) = YTicks(plot.Height);
        // Labels never overlap: one that would touch its neighbour is left out (its tick stays).
        var taken = new List<(double L, double R)>();
        foreach (var v in xs.OrderBy(v => Math.Abs(v)))
        {
            var x = X(v);
            if (Grid) ds.DrawLine(x, (float)plot.Y, x, (float)plot.Bottom, style.Grid, 1);
            else ds.DrawLine(x, (float)plot.Bottom, x, (float)plot.Bottom + 4, style.Axis, 0.8f);
            var label = XLabel(v, xStep);
            var half = PlotText.Measure(ds, label).Width / 2;
            var cx = Math.Clamp(x, plot.X - 12 + half, plot.Right + 12 - half);
            if (taken.Any(t => cx - half - 4 < t.R && cx + half + 4 > t.L)) continue;
            taken.Add((cx - half, cx + half));
            PlotText.Draw(ds, label, cx, plot.Bottom + 10, style.Label);
        }
        foreach (var v in ys)
        {
            var y = Y(v);
            if (Grid) ds.DrawLine((float)plot.X, y, (float)plot.Right, y, style.Grid, 1);
            else ds.DrawLine((float)plot.X - 4, y, (float)plot.X, y, style.Axis, 0.8f);
            PlotText.Draw(ds, YLabel(v, yStep), plot.X - 6, Math.Clamp(y, plot.Y + 5, plot.Bottom - 5), style.Label, anchor: TextAnchor.End);
        }
        if (ZeroLine && YMin < 0 && YMax > 0) ds.DrawLine((float)plot.X, Y(0), (float)plot.Right, Y(0), style.Axis, 1);
        if (Grid)
        {
            ds.DrawLine((float)plot.X, (float)plot.Bottom, (float)plot.Right, (float)plot.Bottom, style.Axis, 1);
            ds.DrawLine((float)plot.X, (float)plot.Y, (float)plot.X, (float)plot.Bottom, style.Axis, 1);
        }
        if (XTitle is not null) SubText.Draw(ds, XTitle, plot.X + plot.Width / 2, plot.Bottom + 27, style.Label, 11);
        if (YTitle is not null)
        {
            var left = plot.X - LeftMargin(ds, plot.Height) + 7;
            SubText.Draw(ds, YTitle, left, plot.Y + plot.Height / 2, style.Label, 11, rotation: -Math.PI / 2);
        }
        return plot;
    }

    /// <summary>The data rectangle inside <paramref name="bounds"/> with room for labels and titles.</summary>
    public Rect Inside(ICanvasResourceCreator rc, Rect bounds, double top = 6, double right = 10)
    {
        var left = LeftMargin(rc, Math.Max(bounds.Height - Bottom - top, 10));
        return new Rect(bounds.X + left, bounds.Y + top, Math.Max(bounds.Width - left - right, 0), Math.Max(bounds.Height - Bottom - top, 0));
    }
}

/// <summary>
/// A map drawn as an image with the colour of <see cref="MapScale"/>: cell maps (PSF,
/// irradiance) fill their extent, node maps (field maps, interpolated) put pixel centres on
/// the nodes. Cells without a value stay transparent, so the hatch behind shows them as no
/// data, never as 0. The bitmap is made once per device, theme and scale.
/// </summary>
sealed class MapImage(MapGrid grid, bool cells)
{
    CanvasBitmap? _bitmap;
    (CanvasDevice Device, bool Dark, MapScale Scale)? _key;

    public MapGrid Grid => grid;

    /// <summary>The extent of the image in data units.</summary>
    public (double X0, double X1, double Y0, double Y1) Extent
    {
        get
        {
            if (!cells) return (grid.X0, grid.X1, grid.Y0, grid.Y1);
            var hx = grid.Nx > 1 ? (grid.X1 - grid.X0) / (grid.Nx - 1) / 2 : 0.5;
            var hy = grid.Ny > 1 ? (grid.Y1 - grid.Y0) / (grid.Ny - 1) / 2 : 0.5;
            return (grid.X0 - hx, grid.X1 + hx, grid.Y0 - hy, grid.Y1 + hy);
        }
    }

    public static Color ColorOf(double v, MapScale s, bool dark)
    {
        var t = s.Position(v);
        if (double.IsNaN(t)) return Color.FromArgb(0, 0, 0, 0);
        return s.Kind switch
        {
            MapScaleKind.Diverging => Palette.Diverging(2 * t - 1, dark),
            MapScaleKind.Sequential => Palette.Sequential(t, dark),
            _ => Palette.Perceptual(t),
        };
    }

    CanvasBitmap Bitmap(CanvasDrawingSession ds, MapScale scale, bool dark)
    {
        if (_bitmap is not null && _key is { } k && k.Device == ds.Device && k.Dark == dark && k.Scale == scale) return _bitmap;
        _bitmap?.Dispose();
        var colors = new Color[grid.Nx * grid.Ny];
        for (var r = 0; r < grid.Ny; r++)
        {
            var dst = (grid.Ny - 1 - r) * grid.Nx; // row 0 at the bottom
            for (var c = 0; c < grid.Nx; c++) colors[dst + c] = ColorOf(grid.Values[r * grid.Nx + c], scale, dark);
        }
        _bitmap = CanvasBitmap.CreateFromColors(ds, colors, grid.Nx, grid.Ny);
        _key = (ds.Device, dark, scale);
        return _bitmap;
    }

    /// <summary>Draws the map over <paramref name="frame"/>'s data rectangle (clipped to it).</summary>
    public void Draw(CanvasDrawingSession ds, Frame frame, MapScale scale, bool dark)
    {
        var (x0, x1, y0, y1) = Extent;
        var left = Math.Min(frame.X(x0), frame.X(x1));
        var right = Math.Max(frame.X(x0), frame.X(x1));
        var top = frame.Y(y1);
        var bottom = frame.Y(y0);
        var dest = new Rect(left, top, right - left, bottom - top);
        if (!cells)
        {
            // Pixel centres on the nodes: extend by half a pixel each side.
            var px = dest.Width / Math.Max(grid.Nx - 1, 1) / 2;
            var py = dest.Height / Math.Max(grid.Ny - 1, 1) / 2;
            dest = new Rect(dest.X - px, dest.Y - py, dest.Width + 2 * px, dest.Height + 2 * py);
        }
        var interp = grid.Nx > 96 ? CanvasImageInterpolation.Linear : CanvasImageInterpolation.NearestNeighbor;
        using (ds.CreateLayer(1, frame.Plot))
            ds.DrawImage(Bitmap(ds, scale, dark), dest, new Rect(0, 0, grid.Nx, grid.Ny), 1, interp);
    }
}

/// <summary>The pieces drawn around and over maps: colour bar, no-data hatch, iso-lines, sample dots, hover card.</summary>
static class MapParts
{
    public const double BarWidth = 10, BarGap = 10;

    /// <summary>The width the bar and its labels take.</summary>
    public static double BarSpace(ICanvasResourceCreator rc, MapScale s, string unit)
    {
        var labels = BarTicks(s, 300).Select(MapScale.Label).Append(MapScale.Label(s.Kind == MapScaleKind.Diverging ? -s.Max : s.Min)).Append(unit);
        return BarGap + BarWidth + 5 + labels.Select(l => PlotText.Measure(rc, l, 10).Width).DefaultIfEmpty(0).Max() + 2;
    }

    static IEnumerable<double> BarTicks(MapScale s, double height)
    {
        var n = Math.Clamp((int)(height / 45), 2, 5);
        return s.Ticks(n);
    }

    /// <summary>A vertical colour bar with its end values, a few round ticks between, and the unit above.</summary>
    public static void Bar(CanvasDrawingSession ds, Rect bar, MapScale s, string unit, PlotStyle style)
    {
        const int steps = 96;
        for (var i = 0; i < steps; i++)
        {
            var t = (i + 0.5) / steps;
            var y0 = bar.Bottom - (i + 1) * bar.Height / steps;
            ds.FillRectangle(new Rect(bar.X, y0, bar.Width, bar.Height / steps + 0.6), MapImage.ColorOf(s.ValueAt(t), s, style.Dark));
        }
        ds.DrawRoundedRectangle(bar, 2, 2, style.Grid, 1);
        var lx = bar.Right + 5;
        var hi = s.Max;
        var lo = s.Kind == MapScaleKind.Diverging ? -s.Max : s.Min;
        PlotText.Draw(ds, MapScale.Label(hi), lx, bar.Y + 4, style.Label, 10, TextAnchor.Start);
        PlotText.Draw(ds, MapScale.Label(lo), lx, bar.Bottom - 4, style.Label, 10, TextAnchor.Start);
        foreach (var v in BarTicks(s, bar.Height))
        {
            var y = bar.Bottom - s.Position(v) * bar.Height;
            if (y < bar.Y + 16 || y > bar.Bottom - 16) continue;
            ds.DrawLine((float)bar.X, (float)y, (float)bar.X + 3, (float)y, style.Surface, 1);
            ds.DrawLine((float)bar.Right - 3, (float)y, (float)bar.Right, (float)y, style.Surface, 1);
            PlotText.Draw(ds, MapScale.Label(v), lx, y, style.Label, 10, TextAnchor.Start);
        }
        if (!string.IsNullOrEmpty(unit)) PlotText.Draw(ds, unit, bar.X, bar.Y - 10, style.Label, 10, TextAnchor.Start);
    }

    /// <summary>Fine diagonal hatching: shows through where a map has no data.</summary>
    public static void Hatch(CanvasDrawingSession ds, Rect r, PlotStyle style)
    {
        var c = style.Label.WithAlpha(style.Dark ? 0.28 : 0.30);
        using var layer = ds.CreateLayer(1, r);
        for (var x = r.X - r.Height; x < r.Right; x += 6)
            ds.DrawLine((float)x, (float)r.Bottom, (float)(x + r.Height), (float)r.Y, c, 0.6f);
    }

    /// <summary>Dots at the sample nodes (faint where a node has no value).</summary>
    public static void Nodes(CanvasDrawingSession ds, Frame f, MapGrid g, PlotStyle style)
    {
        var n = g.Nx * g.Ny;
        if (n > 441) return;
        var r = n > 121 ? 1.1f : 1.6f;
        var ink = style.Dark ? Color.FromArgb(255, 255, 255, 255) : Color.FromArgb(255, 0, 0, 0);
        for (var k = 0; k < n; k++)
        {
            var (x, y) = g.Node(k);
            ds.FillCircle(f.X(x), f.Y(y), r, ink.WithAlpha(double.IsFinite(g.Values[k]) ? 0.35 : 0.15));
        }
    }

    /// <summary>Iso-lines (a white line on a dark halo, readable over every colour), each level labelled once.</summary>
    public static void Contours(CanvasDrawingSession ds, Frame f, IReadOnlyList<(double Level, List<Readers.Contours.Segment> Segments)> levels, Func<double, string> format, PlotStyle style)
    {
        using var layer = ds.CreateLayer(1, f.Plot);
        var round = new CanvasStrokeStyle { StartCap = CanvasCapStyle.Round, EndCap = CanvasCapStyle.Round };
        foreach (var (_, segs) in levels)
            foreach (var s in segs) ds.DrawLine(f.X(s.Ax), f.Y(s.Ay), f.X(s.Bx), f.Y(s.By), Color.FromArgb(0x73, 0, 0, 0), 2.2f, round);
        foreach (var (_, segs) in levels)
            foreach (var s in segs) ds.DrawLine(f.X(s.Ax), f.Y(s.Ay), f.X(s.Bx), f.Y(s.By), Color.FromArgb(0xE6, 255, 255, 255), 0.9f, round);
        var placed = new List<Rect>();
        foreach (var (level, segs) in levels)
        {
            if (segs.Count == 0) continue;
            var text = format(level);
            var size = PlotText.Measure(ds, text, 10);
            foreach (var s in segs.OrderByDescending(s => s.Ax + s.Ay + s.Bx + s.By))
            {
                var mx = (f.X(s.Ax) + f.X(s.Bx)) / 2;
                var my = (f.Y(s.Ay) + f.Y(s.By)) / 2;
                var box = new Rect(mx - size.Width / 2 - 3, my - size.Height / 2 - 1, size.Width + 6, size.Height + 2);
                if (box.X < f.Plot.X || box.Right > f.Plot.Right || box.Y < f.Plot.Y || box.Bottom > f.Plot.Bottom) continue;
                if (placed.Any(p => Intersects(Inflate(p, 2, 1), box))) continue;
                placed.Add(box);
                ds.FillRoundedRectangle(box, 3, 3, style.Dark ? Color.FromArgb(0x99, 0, 0, 0) : Color.FromArgb(0xC0, 255, 255, 255));
                PlotText.Draw(ds, text, mx, my, style.Title, 10);
                break;
            }
        }
    }

    /// <summary>A small card with lines of text near a point, kept inside <paramref name="area"/>.</summary>
    public static void Card(CanvasDrawingSession ds, double px, double py, Rect area, IReadOnlyList<(string Text, bool Strong)> lines, PlotStyle style)
    {
        if (lines.Count == 0) return;
        var sizes = lines.Select(l => SubText.Measure(ds, l.Text, l.Strong ? 11 : 10, l.Strong)).ToList();
        var w = sizes.Max(s => s.Width) + 14;
        var h = sizes.Sum(s => s.Height + 1) + 8;
        var x = px + 12 + w > area.Right ? px - 12 - w : px + 12;
        var y = Math.Clamp(py - h - 6, area.Y, Math.Max(area.Y, area.Bottom - h));
        x = Math.Clamp(x, area.X, Math.Max(area.X, area.Right - w));
        var box = new Rect(x, y, w, h);
        ds.FillRoundedRectangle(box, 5, 5, style.Surface.WithAlpha(0.96));
        ds.DrawRoundedRectangle(box, 5, 5, style.Dark ? Color.FromArgb(0x33, 255, 255, 255) : Color.FromArgb(0x26, 0, 0, 0), 1);
        var ty = y + 4;
        for (var i = 0; i < lines.Count; i++)
        {
            SubText.Draw(ds, lines[i].Text, x + 7, ty + sizes[i].Height / 2, lines[i].Strong ? style.Title : style.Label, lines[i].Strong ? 11 : 10, TextAnchor.Start, lines[i].Strong);
            ty += sizes[i].Height + 1;
        }
    }

    public static Rect Inflate(Rect r, double dx, double dy) => new(r.X - dx, r.Y - dy, r.Width + 2 * dx, r.Height + 2 * dy);
    public static bool Intersects(Rect a, Rect b) => a.X < b.Right && b.X < a.Right && a.Y < b.Bottom && b.Y < a.Bottom;
    public static bool Contains(Rect r, Point p) => p.X >= r.X && p.X <= r.Right && p.Y >= r.Y && p.Y <= r.Bottom;
}

/// <summary>
/// A map with its axes and colour bar, as large as fits with its aspect kept: the map is the
/// content, so no grid over it; ticks and labels around it, the bar on the right.
/// </summary>
sealed class MapPlot
{
    public required MapImage Image { get; init; }
    /// <summary>The sampled grid behind an interpolated image (nodes, no-data hatch); the image's own grid when null.</summary>
    public MapGrid? Samples { get; init; }
    public MapScale Scale { get; set; }
    public string Unit { get; set; } = "";
    public string XTitle { get; init; } = "Hx";
    public string YTitle { get; init; } = "Hy";
    public bool ShowNodes { get; init; }
    /// <summary>Hatch behind cells without a value (no data); off where the outside is simply not part of the map (a pupil).</summary>
    public bool HatchMissing { get; init; } = true;
    public IReadOnlyList<(double Level, List<Contours.Segment> Segments)> Contours { get; set; } = [];
    public Func<double, string> Format { get; set; } = Fmt.Short;
    /// <summary>Extra drawing over the map (rims, scale bars).</summary>
    public Action<CanvasDrawingSession, Frame, PlotStyle>? Overlay { get; init; }
    public Frame Frame { get; } = new() { Grid = false, XTickSpacing = 46, YTickSpacing = 34, MinTicks = 3 };
    /// <summary>A hovered point in view coordinates (null: none).</summary>
    public Point? Hover { get; set; }
    public Func<Frame, Point, IReadOnlyList<(string, bool)>?>? HoverText { get; init; }

    public void Draw(CanvasDrawingSession ds, Rect bounds, PlotStyle style)
    {
        var (x0, x1, y0, y1) = Image.Extent;
        var f = Frame;
        (f.XMin, f.XMax, f.YMin, f.YMax) = (x0, x1, y0, y1);
        f.XTitle = XTitle;
        f.YTitle = YTitle;
        var barSpace = MapParts.BarSpace(ds, Scale, Unit);
        var top = string.IsNullOrEmpty(Unit) ? 8 : 18;
        var availH = bounds.Height - f.Bottom - top;
        var left = f.LeftMargin(ds, Math.Max(availH, 10));
        var availW = bounds.Width - left - barSpace - 4;
        if (availW < 30 || availH < 30) return;
        var aspect = (x1 - x0) / Math.Max(y1 - y0, 1e-12);
        var w = Math.Min(availW, availH * aspect);
        var h = w / aspect;
        var plot = new Rect(bounds.X + left, bounds.Y + top, w, h); // top-aligned: controls above, the map right under them
        if (HatchMissing && (Samples ?? Image.Grid).HasMissing) MapParts.Hatch(ds, plot, style);
        f.Draw(ds, plot, style);
        Image.Draw(ds, f, Scale, style.Dark);
        ds.DrawRectangle(plot, style.Grid, 1);
        if (Contours.Count > 0) MapParts.Contours(ds, f, Contours, Format, style);
        if (ShowNodes) MapParts.Nodes(ds, f, Samples ?? Image.Grid, style);
        Overlay?.Invoke(ds, f, style);
        var barH = Math.Min(h, 240);
        MapParts.Bar(ds, new Rect(plot.Right + MapParts.BarGap, plot.Y + (h - barH) / 2, MapParts.BarWidth, barH), Scale, Unit, style);
        if (Hover is { } p && HoverText?.Invoke(f, p) is { } lines) MapParts.Card(ds, p.X, p.Y, bounds, lines, style);
    }

    /// <summary>The hover card of a node map: the nearest sample's value and position.</summary>
    public static IReadOnlyList<(string, bool)>? NodeHover(Frame f, Point p, MapGrid g, Func<double, string> format)
    {
        if (!MapParts.Contains(MapParts.Inflate(f.Plot, 6, 6), p)) return null;
        var k = g.Nearest(f.DataX(p.X), f.DataY(p.Y));
        var (x, y) = g.Node(k);
        var v = g.Values[k];
        var text = double.IsFinite(v) ? format(v) : g.UnitDiskNoData && MapGrid.Excluded(x, y) ? "Outside the field" : "No value";
        return [(text, true), ($"Hx {Fmt.G3(x)}  Hy {Fmt.G3(y)}".Replace('-', '−'), false)];
    }
}
