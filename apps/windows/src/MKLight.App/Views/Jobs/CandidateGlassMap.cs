using Microsoft.Graphics.Canvas;
using Microsoft.Graphics.Canvas.Geometry;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using MKLight.App.Plotting;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.App.ViewModels.Jobs;
using MKLight.App.Views.Analysis;
using MKLight.App.Views.Analysis.Results;
using MKLight.App.Views.Analysis.Results.Readers;
using MKLight.Model;
using Windows.Foundation;
using Windows.UI;
using Frame = MKLight.App.Views.Analysis.Results.Frame;

namespace MKLight.App.Views.Jobs;

/// <summary>
/// The glass map under a glass substitution candidate (the Mac app's panel map): the catalogue glasses of the
/// design's <c>glass_map</c> in the n_d – V_d or P_g,F – V_d plane, the candidate's glasses ringed in the accent and
/// labelled with their surfaces, and each design glass they replace hollow, joined to its replacement by a dashed line.
/// </summary>
public sealed partial class CandidateGlassMap : UserControl
{
    static readonly CanvasStrokeStyle Dash = new() { CustomDashStyle = [3, 2] };
    readonly IReadOnlyList<GlassMark> _marks;
    readonly IReadOnlyList<(int Surface, string To)> _moves;
    readonly Func<bool> _pgf;
    readonly PlotCanvas _canvas;
    readonly ProgressRing _ring = new() { Width = 18, Height = 18, IsActive = true, HorizontalAlignment = HorizontalAlignment.Center, VerticalAlignment = VerticalAlignment.Center };
    readonly Workspace _ws;
    readonly OpticalSystem _design;
    DesignFigures.Lease? _lease;
    GlassMapData? _data;

    /// <param name="design">The design the candidate came from (its glasses are the ones replaced).</param>
    /// <param name="pgf">Whether the P_g,F plane is shown; <paramref name="setPgf"/> keeps the choice across rebuilds.</param>
    public CandidateGlassMap(Workspace ws, OpticalSystem design, GlassCombination candidate, Func<bool> pgf, Action<bool> setPgf)
    {
        _ws = ws;
        _design = design;
        _pgf = pgf;
        _marks = CandidateGlassMarks.Marks(candidate.Glasses, design.SurfaceIndexOf);
        _moves = _marks.SelectMany(m => m.Surfaces.Select(s => (s, m.Id))).OrderBy(x => x.s).ToList();
        IsTabStop = false;
        _canvas = new PlotCanvas(Draw) { Height = 230 };
        var plane = new Choice("Plane", ["n_d", "P_g,F"], pgf() ? 1 : 0, menuBelow: 0);
        ToolTipService.SetToolTip(plane, "Refractive index nd or partial dispersion Pg,F against Abbe number Vd");
        plane.Changed += k => { setPgf(k == 1); _canvas.Invalidate(); };
        plane.HorizontalAlignment = HorizontalAlignment.Right;
        var head = new Grid();
        head.Children.Add(JobUI.Text("Glass Map", "PanelTitleText"));
        head.Children.Add(plane);
        var plot = new Grid();
        plot.Children.Add(_canvas);
        plot.Children.Add(_ring);
        Content = JobUI.Column(6, head, plot,
            JobUI.Hint("Ringed: the candidate's glasses, labeled with their surfaces. Hollow: the design's glasses they replace.", 11.5));
        AutomationProperties.SetName(_canvas, "Glass map, the candidate's glasses: " + string.Join("; ", _marks.Select(m => $"{m.Name} {m.Label}")));
        Request();
        Loaded += (_, _) => Request();
        Unloaded += (_, _) => { _lease?.Dispose(); _lease = null; };
    }

    void Request()
    {
        if (_lease is not null) return;
        _lease = DesignFigures.Of(_ws).Request(_design, "glass_map",
            (o, s) => GlassMapData.Read(ResultHosting.Data("glass_map", o, s)),
            d =>
            {
                _data = d;
                _ring.IsActive = false;
                _ring.Visibility = Visibility.Collapsed;
                _canvas.Invalidate();
            });
    }

    static Color StatusColor(GlassStatus s, bool dark) => s.Slot() < 0 ? (dark ? Palette.Rgb(0x9a9a95) : Palette.Rgb(0x8a8a85)) : Palette.Slot(s.Slot(), dark);

    /// <summary>A candidate glass on the map: its catalogue point, else the catalogue glass of that name.</summary>
    GlassPoint? Point(GlassMark m) =>
        _data?.Glasses.FirstOrDefault(g => g.Id == m.Id)
        ?? _data?.Glasses.FirstOrDefault(g => g.Kind == "catalog" && g.Name == m.Name);

    void Draw(CanvasDrawingSession ds, Size size, PlotStyle style)
    {
        if (_data is not { } d) return;
        var pgf = _pgf();
        double Y(GlassPoint g) => pgf ? g.Pgf : g.Nd;
        var shown = d.Glasses.Where(g => double.IsFinite(g.Vd) && double.IsFinite(Y(g))).ToList();
        if (shown.Count == 0) return;
        double vlo = shown.Min(g => g.Vd), vhi = shown.Max(g => g.Vd);
        double ylo = shown.Min(Y), yhi = shown.Max(Y);
        var vpad = Math.Max((vhi - vlo) * 0.05, 2);
        var ypad = Math.Max((yhi - ylo) * 0.06, pgf ? 0.005 : 0.02);
        var f = new Frame
        {
            XMin = vlo - vpad, XMax = vhi + vpad, YMin = ylo - ypad, YMax = yhi + ypad, ReverseX = true, XTickSpacing = 46, YTickSpacing = 40,
            MinTicks = 3, XTitle = "Abbe number V_d", YTitle = pgf ? "Partial dispersion P_g,F" : "Refractive index n_d",
        };
        var r = f.Draw(ds, f.Inside(ds, new Rect(0, 0, size.Width, size.Height), top: 8, right: 8), style);
        if (r.IsEmpty) return;
        var accent = Kit.Accent(this);
        var marked = _marks.Select(m => (Mark: m, Point: Point(m))).Where(x => x.Point is { } p && double.IsFinite(Y(p))).ToList();
        var markedIds = marked.Select(x => x.Point!.Id).ToHashSet();
        var design = shown.Where(g => g.InDesign).Select(g => (g.Id, g.Kind, g.Surfaces)).ToList();
        var originals = _moves
            .Select(m => (From: CandidateGlassMarks.Original(design, m.Surface), m.To))
            .Select(m => (From: shown.FirstOrDefault(g => g.Id == m.From), To: marked.FirstOrDefault(x => x.Mark.Id == m.To).Point))
            .Where(m => m.From is not null && m.To is not null && m.From.Id != m.To.Id)
            .ToList();
        var originalIds = originals.Select(m => m.From!.Id).ToHashSet();
        using (ds.CreateLayer(1, MapParts.Inflate(r, 4, 4)))
        {
            if (pgf)
                ds.DrawLine(f.X(f.XMin), f.Y(GlassPoint.NormalLine(f.XMin)), f.X(f.XMax), f.Y(GlassPoint.NormalLine(f.XMax)), style.Label.WithAlpha(0.6), 1, Dash);
            foreach (var g in shown.Where(g => !markedIds.Contains(g.Id) && !originalIds.Contains(g.Id)))
            {
                var c = StatusColor(g.Status, style.Dark);
                if (g.Status.Hollow()) ds.DrawCircle(f.X(g.Vd), f.Y(Y(g)), 2.4f, c, 1);
                else ds.FillCircle(f.X(g.Vd), f.Y(Y(g)), 2.4f, c.WithAlpha(0.8));
            }
            // The move from each design glass to its replacement.
            foreach (var (from, to) in originals)
                ds.DrawLine(f.X(from!.Vd), f.Y(Y(from)), f.X(to!.Vd), f.Y(Y(to)), accent.WithAlpha(0.55), 1.2f, Dash);
            var placed = new List<Rect>();
            foreach (var g in originals.Select(m => m.From!).DistinctBy(g => g.Id))
            {
                float x = f.X(g.Vd), y = f.Y(Y(g));
                ds.FillCircle(x, y, 4.5f, style.Surface);
                ds.DrawCircle(x, y, 4.5f, style.Title, 1.3f);
                placed.Add(new Rect(x - 5, y - 5, 10, 10));
            }
            foreach (var (_, g) in marked)
            {
                float x = f.X(g!.Vd), y = f.Y(Y(g));
                ds.FillCircle(x, y, 5, StatusColor(g.Status, style.Dark));
                ds.DrawCircle(x, y, 7, accent, 2);
                placed.Add(new Rect(x - 8, y - 8, 16, 16));
            }
            foreach (var (m, g) in marked)
            {
                float x = f.X(g!.Vd), y = f.Y(Y(g));
                var label = m.Name + "  " + m.Label;
                var sz = PlotText.Measure(ds, label, 10, true);
                Rect[] boxes =
                [
                    new(x + 10, y - sz.Height / 2, sz.Width + 6, sz.Height), new(x - 16 - sz.Width, y - sz.Height / 2, sz.Width + 6, sz.Height),
                    new(x - sz.Width / 2 - 3, y - 11 - sz.Height, sz.Width + 6, sz.Height), new(x - sz.Width / 2 - 3, y + 11, sz.Width + 6, sz.Height),
                    new(x + 6, y - 8 - sz.Height, sz.Width + 6, sz.Height), new(x + 6, y + 8, sz.Width + 6, sz.Height),
                    new(x - 12 - sz.Width, y - 8 - sz.Height, sz.Width + 6, sz.Height), new(x - 12 - sz.Width, y + 8, sz.Width + 6, sz.Height),
                ];
                // The first free place inside the plot, else the one inside that covers the least of the marks and labels.
                var box = boxes.Where(b => Inside(r, b)).DefaultIfEmpty(boxes[0])
                    .OrderBy(b => placed.Sum(p => Overlap(p, b))).First();
                placed.Add(box);
                ds.FillRoundedRectangle(box, 3, 3, style.Dark ? Color.FromArgb(0x8C, 0, 0, 0) : Color.FromArgb(0xCC, 255, 255, 255));
                PlotText.Draw(ds, label, box.X + 3, box.Y + box.Height / 2, style.Title, 10, TextAnchor.Start, bold: true);
            }
        }
    }

    static double Overlap(Rect a, Rect b) =>
        Math.Max(0, Math.Min(a.Right, b.Right) - Math.Max(a.X, b.X)) * Math.Max(0, Math.Min(a.Bottom, b.Bottom) - Math.Max(a.Y, b.Y));

    static bool Inside(Rect outer, Rect r) => r.X >= outer.X && r.Right <= outer.Right && r.Y >= outer.Y && r.Bottom <= outer.Bottom;
}
