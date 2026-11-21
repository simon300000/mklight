using Microsoft.Graphics.Canvas;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using MKLight.App.Plotting;
using MKLight.App.Support;
using MKLight.App.Views.Analysis.Results.Readers;
using MKLight.Model;
using Windows.Foundation;

namespace MKLight.App.Views.Analysis.Results;

/// <summary>
/// <c>seidel</c>: grouped bars per surface and their sum (Σ) for S_I…S_V, or C_L and C_T,
/// with real subscripts in the legend; the Petzval radius and the wavefront coefficients.
/// A design without rotational symmetry says why it has no coefficients.
/// </summary>
public sealed partial class SeidelView : UserControl, IViewOptions
{
    readonly Dictionary<string, Action> _opts = [];
    public void Apply(string option) { if (_opts.TryGetValue(option, out var a)) a(); }
    readonly SeidelData _d;
    readonly PlotCanvas _canvas;
    readonly FlowPanel _legend = new() { Spacing = 14, LineSpacing = 3 };
    bool _chromatic;
    Point? _hover;

    public SeidelView(SeidelData d)
    {
        _d = d;
        IsTabStop = false;
        _canvas = new PlotCanvas(Draw);
        if (d.NotApplicable is not null)
        {
            Content = Message(Loc.S("Results_Seidel_NotApplicable"), EngineText.T(d.NotApplicable));
            return;
        }
        var choice = new Choice(Loc.S("Results_Seidel_Coefficients"), [Loc.S("Results_Seidel_ChoiceSeidel"), Loc.S("Results_Seidel_ChoiceChromatic")]);
        choice.Changed += k => { _chromatic = k == 1; BuildLegend(); _canvas.Invalidate(); };
        _opts["chromatic"] = () => choice.Select(1);
        var top = new FlowPanel { Spacing = 16, LineSpacing = 4 };
        top.Children.Add(choice);
        if (double.IsFinite(d.PetzvalRadius))
        {
            var p = Kit.Caption(Loc.F("Results_Seidel_PetzvalRadius", Fmt.Value(d.PetzvalRadius, "mm")));
            p.VerticalAlignment = VerticalAlignment.Center;
            p.Margin = new Thickness(0, 4, 0, 0);
            top.Children.Add(p);
        }
        var root = new Grid { Padding = new Thickness(10, 6, 10, 8), RowSpacing = 6 };
        for (var i = 0; i < 4; i++) root.RowDefinitions.Add(new RowDefinition { Height = i == 2 ? new GridLength(1, GridUnitType.Star) : GridLength.Auto });
        root.Children.Add(top);
        Grid.SetRow(_legend, 1);
        root.Children.Add(_legend);
        Grid.SetRow(_canvas, 2);
        root.Children.Add(_canvas);
        if (d.Wavefront.Count > 0)
        {
            var wf = new FlowPanel { Spacing = 14, LineSpacing = 3 };
            wf.Children.Add(Kit.Text(Loc.S("Results_Seidel_WavefrontWaves"), "CaptionText"));
            foreach (var (k, v) in d.Wavefront)
            {
                var item = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 4 };
                item.Children.Add(Kit.Sub(k, 11));
                item.Children.Add(Kit.Text(Fmt.Signed(v, 4), "NumberText", 11));
                AutomationProperties.SetName(item, Loc.F("Results_Seidel_WavefrontItemName", Kit.Spoken(k), Fmt.Signed(v, 4)));
                wf.Children.Add(item);
            }
            Grid.SetRow(wf, 3);
            root.Children.Add(wf);
        }
        Content = root;
        BuildLegend();
        ActualThemeChanged += (_, _) => BuildLegend();
        AutomationProperties.SetName(_canvas, Loc.S("Results_Seidel_ChartName"));
        _canvas.PointerMoved += (_, e) => { _hover = e.GetCurrentPoint(_canvas).Position; _canvas.Invalidate(); };
        _canvas.PointerExited += (_, _) => { _hover = null; _canvas.Invalidate(); };
    }

    (string Key, string Label)[] Terms => _chromatic ? SeidelData.Chromatic : SeidelData.Terms;
    int SlotOffset => _chromatic ? 5 : 0;

    void BuildLegend()
    {
        _legend.Children.Clear();
        var dark = ActualTheme == ElementTheme.Dark;
        for (var k = 0; k < Terms.Length; k++)
            _legend.Children.Add(Kit.LegendItem(Kit.Dot(Palette.Field(k + SlotOffset, dark)), Kit.Sub(Terms[k].Label, 11)));
    }

    internal static FrameworkElement Message(string title, string text)
    {
        var p = new StackPanel { Spacing = 6, VerticalAlignment = VerticalAlignment.Center, HorizontalAlignment = HorizontalAlignment.Center, MaxWidth = 320, Padding = new Thickness(16) };
        var t = Kit.Text(title, "PanelTitleText", 14);
        t.HorizontalAlignment = HorizontalAlignment.Center;
        p.Children.Add(t);
        var b = Kit.Caption(text, wrap: true);
        b.TextAlignment = TextAlignment.Center;
        b.HorizontalAlignment = HorizontalAlignment.Center;
        p.Children.Add(b);
        return p;
    }

    void Draw(CanvasDrawingSession ds, Size size, PlotStyle style)
    {
        var terms = Terms;
        var n = _d.Surfaces.Count;
        if (n == 0) return;
        var all = terms.SelectMany(t => _d.Values[t.Key]).Where(double.IsFinite);
        var f = new Frame { XTitle = Loc.S("Results_Seidel_AxisSurface"), YTitle = Loc.S("Results_Seidel_AxisCoefficient"), XTicks = [], ZeroLine = true, YTickSpacing = 45 };
        (f.XMin, f.XMax) = (-0.5, n - 0.5);
        (f.YMin, f.YMax) = Ticks.Range(all, 0.06, includeZero: true);
        var r = f.Draw(ds, f.Inside(ds, new Rect(0, 0, size.Width, size.Height)), style);
        if (r.IsEmpty) return;
        var group = r.Width / n;
        var bw = group * 0.8 / terms.Length;
        // Σ is set apart by a light divider.
        var sx = f.X(n - 1.5);
        ds.DrawLine(sx, (float)r.Y, sx, (float)r.Bottom, style.Grid, 1);
        var every = Math.Max(1, (int)Math.Ceiling(22 / group));
        for (var k = 0; k < n; k++)
        {
            var cx = f.X(k);
            if (k % every == 0 || k == n - 1) PlotText.Draw(ds, _d.Surfaces[k], cx, r.Bottom + 10, k == n - 1 ? style.Title : style.Label, bold: k == n - 1);
            for (var t = 0; t < terms.Length; t++)
            {
                var v = _d.Values[terms[t].Key][k];
                if (!double.IsFinite(v)) continue;
                var x = cx - group * 0.4 + t * bw;
                float y0 = f.Y(0), y1 = f.Y(v);
                var rect = new Rect(x + Math.Min(0.5, bw * 0.1), Math.Min(y0, y1), Math.Max(bw - Math.Min(1, bw * 0.2), 0.8), Math.Max(Math.Abs(y1 - y0), 0.8));
                ds.FillRoundedRectangle(rect, 1.5f, 1.5f, Palette.Field(t + SlotOffset, style.Dark));
            }
        }
        if (_hover is { } p && MapParts.Contains(r, p))
        {
            var k = Math.Clamp((int)Math.Floor((p.X - r.X) / group), 0, n - 1);
            ds.FillRectangle(new Rect(r.X + k * group, r.Y, group, r.Height), style.Title.WithAlpha(0.05));
            var lines = new List<(string, bool)> { (k == n - 1 ? Loc.S("Results_Seidel_Sum") : Loc.F("Results_Seidel_SurfaceNumber", _d.Surfaces[k]), true) };
            lines.AddRange(terms.Select(t => (t.Label.Split(' ')[0] + "  " + Fmt.Short(_d.Values[t.Key][k]), false)));
            MapParts.Card(ds, p.X, p.Y, new Rect(0, 0, size.Width, size.Height), lines, style);
        }
    }
}
