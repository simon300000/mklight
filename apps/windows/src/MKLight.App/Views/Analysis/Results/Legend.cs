using Microsoft.UI.Text;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Shapes;
using MKLight.App.Plotting;
using MKLight.App.Support;
using MKLight.App.Views.Analysis.Results.Readers;
using Windows.Foundation;
using Windows.UI;

namespace MKLight.App.Views.Analysis.Results;

/// <summary>Series colours: fields take the categorical slots, wavelengths their spectral slot; reference curves a quiet ink.</summary>
public static class ChartColors
{
    public static Color Resolve(SeriesColor c, bool dark) => c.Role switch
    {
        ColorRole.Field => Palette.Field(c.Index, dark),
        ColorRole.Wavelength => Palette.Wavelength(c.Um, dark),
        _ => dark ? Color.FromArgb(0x80, 255, 255, 255) : Color.FromArgb(0x73, 0, 0, 0),
    };
}

/// <summary>Lays children out in rows, wrapping when the width runs out (legends: one flow that wraps).</summary>
public sealed partial class FlowPanel : Panel
{
    public double Spacing { get; set; } = 12;
    public double LineSpacing { get; set; } = 3;

    protected override Size MeasureOverride(Size available)
    {
        double x = 0, y = 0, line = 0, maxX = 0;
        foreach (var c in Children)
        {
            c.Measure(new Size(available.Width, double.PositiveInfinity));
            var s = c.DesiredSize;
            if (x > 0 && x + s.Width > available.Width) { y += line + LineSpacing; x = 0; line = 0; }
            x += s.Width + Spacing;
            maxX = Math.Max(maxX, x - Spacing);
            line = Math.Max(line, s.Height);
        }
        return new Size(double.IsFinite(available.Width) ? Math.Min(maxX, available.Width) : maxX, y + line);
    }

    protected override Size ArrangeOverride(Size final)
    {
        double x = 0, y = 0, line = 0;
        foreach (var c in Children)
        {
            var s = c.DesiredSize;
            if (x > 0 && x + s.Width > final.Width) { y += line + LineSpacing; x = 0; line = 0; }
            c.Arrange(new Rect(x, y, Math.Min(s.Width, final.Width), s.Height));
            x += s.Width + Spacing;
            line = Math.Max(line, s.Height);
        }
        return final;
    }
}

/// <summary>A chart legend: swatches (line, dashed line, dot, dashed circle) and labels, in one wrapping flow.</summary>
public sealed partial class Legend : UserControl
{
    readonly IReadOnlyList<LegendEntry> _entries;
    readonly FlowPanel _panel = new();

    public Legend(IReadOnlyList<LegendEntry> entries)
    {
        _entries = entries;
        IsTabStop = false;
        Content = _panel;
        Build();
        ActualThemeChanged += (_, _) => Build();
        Visibility = entries.Count == 0 ? Visibility.Collapsed : Visibility.Visible;
    }

    void Build()
    {
        _panel.Children.Clear();
        var dark = ActualTheme == ElementTheme.Dark;
        var label = new SolidColorBrush(PlotStyle.For(ActualTheme).Label);
        foreach (var e in _entries)
        {
            var row = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 5 };
            if (e.Mark != LegendMark.Text) row.Children.Add(Swatch(e.Mark, new SolidColorBrush(ChartColors.Resolve(e.Color, dark))));
            row.Children.Add(Caption(e.Label, label));
            _panel.Children.Add(row);
        }
    }

    public static TextBlock Caption(string text, Brush foreground, double size = 11) => new TextBlock
    {
        Text = text,
        FontSize = size,
        Foreground = foreground,
        VerticalAlignment = VerticalAlignment.Center,
        TextTrimming = TextTrimming.CharacterEllipsis,
    }.Tabular();

    static FrameworkElement Swatch(LegendMark mark, Brush brush)
    {
        var c = new Canvas { Width = 16, Height = 10, VerticalAlignment = VerticalAlignment.Center };
        switch (mark)
        {
            case LegendMark.Dot:
                var dot = new Ellipse { Width = 7, Height = 7, Fill = brush };
                Canvas.SetLeft(dot, 4.5); Canvas.SetTop(dot, 1.5);
                c.Children.Add(dot);
                break;
            case LegendMark.DashedCircle:
                var ring = new Ellipse { Width = 9, Height = 9, Stroke = brush, StrokeThickness = 1, StrokeDashArray = Dash(2, 1.5) };
                Canvas.SetLeft(ring, 3.5); Canvas.SetTop(ring, 0.5);
                c.Children.Add(ring);
                break;
            default:
                var line = new Line { X1 = 0.5, Y1 = 5, X2 = 15.5, Y2 = 5, Stroke = brush, StrokeThickness = 1.8, StrokeStartLineCap = PenLineCap.Round, StrokeEndLineCap = PenLineCap.Round };
                if (mark == LegendMark.Dashed) line.StrokeDashArray = Dash(2.2, 1.6);
                c.Children.Add(line);
                break;
        }
        return c;
    }

    static DoubleCollection Dash(double on, double off)
    {
        var d = new DoubleCollection();
        d.Add(on);
        d.Add(off);
        return d;
    }
}

static class TextExtensions
{
    /// <summary>Tabular figures: every number in the app uses them.</summary>
    public static TextBlock Tabular(this TextBlock t)
    {
        Microsoft.UI.Xaml.Documents.Typography.SetNumeralAlignment(t, FontNumeralAlignment.Tabular);
        return t;
    }
}
