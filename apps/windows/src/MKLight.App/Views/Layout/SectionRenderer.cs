using System.Numerics;
using Microsoft.Graphics.Canvas;
using Microsoft.Graphics.Canvas.Geometry;
using Microsoft.UI.Xaml;
using MKLight.App.Support;
using Windows.UI;

namespace MKLight.App.Views.Layout;

/// <summary>
/// Draws a <see cref="SectionScene"/> with Win2D (the Mac viewport's section
/// style): glass fills, rays by field or wavelength (vignetted rays faint),
/// then outlines, the selected surface last in the accent; DOE surfaces
/// carry a violet dash. Shared by the layout pane and the welcome thumbnails.
/// </summary>
public static class SectionRenderer
{
    static readonly CanvasStrokeStyle Round = new() { LineJoin = CanvasLineJoin.Round, StartCap = CanvasCapStyle.Round, EndCap = CanvasCapStyle.Round };
    static readonly CanvasStrokeStyle AxisDash = new() { CustomDashStyle = [8, 5] };
    static readonly CanvasStrokeStyle DoeDash = new() { CustomDashStyle = [2, 2] };

    public sealed record Style(bool Dark, RayColoring Coloring = RayColoring.Field, int? Selected = null, bool Stale = false, float LineScale = 1);

    public static void Draw(CanvasDrawingSession ds, SectionScene scene, Func<Vector2, Vector2> toScreen, Style style)
    {
        var dark = style.Dark;
        var k = style.LineScale;
        Color ink = dark ? Color.FromArgb(255, 224, 227, 232) : Color.FromArgb(255, 33, 36, 41);
        Color faint = dark ? Color.FromArgb(204, 140, 143, 153) : Color.FromArgb(204, 140, 143, 148);
        var accent = (Color)Application.Current.Resources[dark ? "SystemAccentColorLight2" : "SystemAccentColor"];
        var glass = dark ? Color.FromArgb(56, 89, 153, 242) : Color.FromArgb(64, 115, 173, 242);
        var doeInk = Palette.Slot(6, dark).WithAlpha(0.85);
        var dim = style.Stale ? 0.35 : 1.0;

        foreach (var e in scene.Elements)
        {
            if (e.Front.Length < 2 || e.Front.Length != e.Back.Length) continue;
            using var pb = new CanvasPathBuilder(ds);
            pb.BeginFigure(toScreen(e.Front[0]));
            for (var i = 1; i < e.Front.Length; i++) pb.AddLine(toScreen(e.Front[i]));
            for (var i = e.Back.Length - 1; i >= 0; i--) pb.AddLine(toScreen(e.Back[i]));
            pb.EndFigure(CanvasFigureLoop.Closed);
            using var g = CanvasGeometry.CreatePath(pb);
            ds.FillGeometry(g, glass.WithAlpha(glass.A / 255.0 * (style.Stale ? 0.6 : 1)));
        }

        foreach (var l in scene.Lines.Where(l => l.Kind == SectionKind.Ray))
        {
            var c = style.Coloring == RayColoring.Field
                ? Palette.Field(l.Field ?? 0, dark)
                : Palette.Wavelength(l.Wavelength is int w && w < scene.WavelengthsUm.Count ? scene.WavelengthsUm[w] : 0.55, dark);
            Line(ds, l.Points, toScreen, c.WithAlpha((l.Vignetted ? 0.35 : 0.9) * dim), 1.1f * k, Round);
        }
        bool IsSelected(SectionLine l) => l.Surface is not null && l.Surface == style.Selected && l.Kind != SectionKind.ElementEdge;
        foreach (var l in scene.Lines.Where(l => l.Kind != SectionKind.Ray).OrderBy(l => IsSelected(l) ? 1 : 0))
        {
            var sel = IsSelected(l);
            switch (l.Kind)
            {
                case SectionKind.Axis: Line(ds, l.Points, toScreen, faint, 0.8f * k, AxisDash); break;
                case SectionKind.ElementEdge: Line(ds, l.Points, toScreen, ink, 1.4f * k, Round); break;
                case SectionKind.SurfaceProfile:
                    Line(ds, l.Points, toScreen, sel ? accent : ink, (sel ? 3.2f : 1.4f) * k, Round);
                    if (l.Doe) Line(ds, l.Points, toScreen, doeInk, (sel ? 4.2f : 3.4f) * k, DoeDash);
                    break;
                case SectionKind.SurfaceRim: Line(ds, l.Points, toScreen, sel ? accent : faint, (sel ? 1.8f : 1f) * k, Round); break;
                case SectionKind.Stop: Line(ds, l.Points, toScreen, sel ? accent : ink, (sel ? 2.8f : 2.2f) * k, Round); break;
                case SectionKind.ImagePlane: Line(ds, l.Points, toScreen, sel ? accent : ink, (sel ? 2.4f : 1.8f) * k, Round); break;
            }
        }
    }

    static void Line(CanvasDrawingSession ds, Vector2[] pts, Func<Vector2, Vector2> toScreen, Color color, float width, CanvasStrokeStyle stroke)
    {
        if (pts.Length < 2) return;
        using var pb = new CanvasPathBuilder(ds);
        var open = false;
        foreach (var p in pts)
        {
            if (!float.IsFinite(p.X) || !float.IsFinite(p.Y)) { if (open) pb.EndFigure(CanvasFigureLoop.Open); open = false; continue; }
            var s = toScreen(p);
            if (!open) { pb.BeginFigure(s); open = true; } else pb.AddLine(s);
        }
        if (open) pb.EndFigure(CanvasFigureLoop.Open);
        using var g = CanvasGeometry.CreatePath(pb);
        ds.DrawGeometry(g, color, width, stroke);
    }

    /// <summary>
    /// The scale and centre that fit a scene into a viewport (mm → px, y up). A long,
    /// slim scene (an object far away) is framed on its lens; the frame stays
    /// symmetric about the axis.
    /// </summary>
    public static (float Scale, Vector2 Center) Fit(SectionScene scene, float width, float height, float marginX, float marginTop, float marginBottom)
    {
        var (min, max) = scene.Bounds();
        var (lmin, lmax) = scene.Bounds(lensOnly: true);
        var span = max - min;
        if (span.X > 12 * Math.Max(span.Y, 1e-6f) && (lmax.X - lmin.X) * 5 < span.X)
            min = new Vector2(lmin.X - (lmax.X - lmin.X) * 0.5f, min.Y);
        min.Y = Math.Min(min.Y, -max.Y); max.Y = Math.Max(max.Y, -min.Y);
        var size = max - min;
        var sx = (width - 2 * marginX) / Math.Max(size.X, 1e-6f);
        var sy = (height - marginTop - marginBottom) / Math.Max(size.Y, 1e-6f);
        var scale = Math.Max(1e-4f, Math.Min(sx, sy));
        var center = (min + max) / 2;
        center.Y += (marginTop - marginBottom) / 2 / scale;
        return (scale, center);
    }
}
