using System.Globalization;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Shapes;
using MKLight.App.Plotting;
using MKLight.App.Support;
using Windows.Foundation;

namespace MKLight.App.Views.Analysis.Settings;

/// <summary>
/// The weights a run uses against wavelength: one bar per wavelength in its
/// spectral colour (tallest = 1) over the continuous response it samples
/// (a light area), the primary marked, and a caption saying where the
/// weights come from. Content, not chrome: plain and crisp.
/// </summary>
public sealed partial class SpectrumPreview : UserControl
{
    public enum Source
    {
        /// <summary>The design's own wavelengths.</summary>
        Design,
        /// <summary>Explicit wavelengths: exactly what the run uses.</summary>
        Exact,
        /// <summary>Resolved by the engine for this spectrum.</summary>
        Engine,
        /// <summary>Computed locally from approximations.</summary>
        Approximate,
    }

    const double ChartHeight = 64, AxisHeight = 16;

    readonly IReadOnlyList<SpectrumSample> _samples;
    readonly IReadOnlyList<SpectrumSample>? _curve;
    readonly int? _primary;
    readonly Canvas _canvas = new() { Height = ChartHeight + AxisHeight };

    /// <param name="primary">The primary's index; by default the engine's rule for spectra (nearest the energy-weighted mean).</param>
    public SpectrumPreview(IReadOnlyList<SpectrumSample> samples, IReadOnlyList<SpectrumSample>? curve, Source source, int? primary = null)
    {
        _samples = samples;
        _curve = curve;
        _primary = primary ?? SpectrumSetting.PrimaryIndex(samples);
        IsTabStop = false;
        MinWidth = 220;
        MaxWidth = 420;
        HorizontalAlignment = HorizontalAlignment.Stretch;

        var caption = new TextBlock { Text = Caption(source), Style = (Style)Application.Current.Resources["CaptionText"] };
        var panel = new StackPanel { Spacing = 2 };
        panel.Children.Add(_canvas);
        panel.Children.Add(caption);
        Content = panel;
        _canvas.SizeChanged += (_, _) => Draw();
        ActualThemeChanged += (_, _) => Draw();
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(this, Loc.F("Analysis_Spectrum_PreviewName", Title(source).ToLowerInvariant()));
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetHelpText(this, Spoken());
    }

    static string Title(Source s) => Loc.S(s switch
    {
        Source.Design => "Analysis_Spectrum_SourceDesign",
        Source.Exact => "Analysis_Spectrum_SourceExplicit",
        Source.Engine => "Analysis_Spectrum_SourceEngine",
        _ => "Analysis_Spectrum_SourceApproximate",
    });

    string Caption(Source s)
    {
        var t = Loc.F("Analysis_Spectrum_Caption", Title(s), _samples.Count);
        if (_primary is int p) t = Loc.F("Analysis_Spectrum_CaptionPrimary", t, _samples[p].Um.ToString("0.0000", CultureInfo.InvariantCulture));
        return t;
    }

    string Spoken()
    {
        if (_samples.Count == 0) return "";
        double lo = _samples.Min(w => w.Um), hi = _samples.Max(w => w.Um);
        var s = Loc.F("Analysis_Spectrum_Spoken", _samples.Count, lo.ToString("0.000", CultureInfo.InvariantCulture), hi.ToString("0.000", CultureInfo.InvariantCulture));
        if (_primary is int p) s = Loc.F("Analysis_Spectrum_SpokenPrimary", s, _samples[p].Um.ToString("0.0000", CultureInfo.InvariantCulture));
        return s;
    }

    (double Lo, double Hi) Domain()
    {
        var xs = _samples.Select(w => w.Um).Concat(_curve?.Select(w => w.Um) ?? []).ToList();
        if (xs.Count == 0) return (0.4, 0.7);
        double lo = xs.Min(), hi = xs.Max();
        var pad = Math.Max((hi - lo) * 0.08, 0.012);
        return (lo - pad, hi + pad);
    }

    void Draw()
    {
        _canvas.Children.Clear();
        var width = _canvas.ActualWidth;
        if (width <= 0) return;
        var dark = ActualTheme == ElementTheme.Dark;
        var (lo, hi) = Domain();
        double X(double um) => (um - lo) / (hi - lo) * width;
        double Y(double v) => ChartHeight - Math.Clamp(v, 0, 1.08) / 1.08 * ChartHeight;
        var secondary = ThemeLookup.Brush(this, "TextFillColorTertiaryBrush");
        var grid = ThemeLookup.Brush(this, "DividerBrush");

        // Axis: gridlines and labels at round wavelengths, a baseline.
        foreach (var t in Ticks.Values(lo, hi, 4))
        {
            if (t < lo || t > hi) continue;
            var x = Math.Round(X(t)) + 0.5;
            _canvas.Children.Add(new Line { X1 = x, X2 = x, Y1 = 0, Y2 = ChartHeight, Stroke = grid, StrokeThickness = 1 });
            var label = new TextBlock { Text = t.ToString("0.00", CultureInfo.InvariantCulture), FontSize = 10, Foreground = secondary };
            Microsoft.UI.Xaml.Documents.Typography.SetNumeralAlignment(label, FontNumeralAlignment.Tabular);
            label.Measure(new Size(double.PositiveInfinity, double.PositiveInfinity));
            Canvas.SetLeft(label, Math.Clamp(x - label.DesiredSize.Width / 2, 0, Math.Max(0, width - label.DesiredSize.Width)));
            Canvas.SetTop(label, ChartHeight + 2);
            _canvas.Children.Add(label);
        }
        _canvas.Children.Add(new Line { X1 = 0, X2 = width, Y1 = ChartHeight + 0.5, Y2 = ChartHeight + 0.5, Stroke = grid, StrokeThickness = 1 });

        // The continuous response, as a light area.
        if (_curve is { Count: > 1 } curve)
        {
            var area = new Polygon { Fill = new SolidColorBrush(ThemeColor("TextFillColorPrimary", dark).WithAlpha(0.07)) };
            area.Points.Add(new Point(X(curve[0].Um), ChartHeight));
            foreach (var p in curve) area.Points.Add(new Point(X(p.Um), Y(p.Weight)));
            area.Points.Add(new Point(X(curve[^1].Um), ChartHeight));
            _canvas.Children.Add(area);
        }

        // One bar per wavelength in its colour; the primary gets a dot.
        var peak = Math.Max(_samples.Select(w => w.Weight).DefaultIfEmpty(1).Max(), double.Epsilon);
        var thickness = _samples.Count > 40 ? 1.5 : 3;
        for (var k = 0; k < _samples.Count; k++)
        {
            var w = _samples[k];
            var brush = new SolidColorBrush(Palette.Wavelength(w.Um, dark));
            var x = X(w.Um);
            var top = Y(w.Weight / peak);
            _canvas.Children.Add(new Line { X1 = x, X2 = x, Y1 = ChartHeight, Y2 = top, Stroke = brush, StrokeThickness = thickness });
            if (k == _primary)
            {
                var dot = new Ellipse { Width = 6, Height = 6, Fill = brush };
                Canvas.SetLeft(dot, x - 3);
                Canvas.SetTop(dot, top - 3);
                _canvas.Children.Add(dot);
            }
        }
    }

    static global::Windows.UI.Color ThemeColor(string name, bool dark) =>
        ThemeLookup.Resource(dark ? ElementTheme.Dark : ElementTheme.Light, name + "Brush") is SolidColorBrush b ? b.Color
            : dark ? Microsoft.UI.Colors.White : Microsoft.UI.Colors.Black;
}
