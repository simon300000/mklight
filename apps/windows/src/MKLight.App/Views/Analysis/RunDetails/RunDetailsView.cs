using System.Globalization;
using System.Text.Json.Nodes;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.Engine;

namespace MKLight.App.Views.Analysis.RunDetails;

/// <summary>
/// The run details a pane's backend badge opens (the Mac app's RunDetails):
/// the backend that computed the result, what was requested, how an
/// automatic request was routed and why a run was on the CPU, the precision,
/// tier, seed, snapshot, engine and time, the attempts not used, and the
/// engine's warnings grouped by message (fields and wavelengths numbered
/// from 1), with a one-click remedy where the analysis has the setting for
/// it; a remedy changes the pane's settings through <see cref="AnalysisPaneModel.SetSettings"/>.
/// </summary>
public static class RunDetailsView
{
    const double LabelWidth = 104;

    /// <summary>Fill <paramref name="panel"/> (the badge's flyout) for <paramref name="output"/> of <paramref name="pane"/>.</summary>
    public static void Build(StackPanel panel, AnalysisOutput output, AnalysisPaneModel pane, DesignSession session)
    {
        panel.Children.Clear();
        panel.Spacing = 4;
        var info = output.RunInfo;
        var b = output.Backend;
        var rows = new Grid { ColumnSpacing = 12, RowSpacing = 3 };
        rows.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(LabelWidth) });
        rows.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        void Row(string label, string? value)
        {
            if (string.IsNullOrEmpty(value)) return;
            var r = rows.RowDefinitions.Count;
            rows.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            var l = Text(label, "SecondaryText");
            l.VerticalAlignment = VerticalAlignment.Top;
            var v = Text(value, "NumberText");
            v.VerticalAlignment = VerticalAlignment.Top;
            v.TextWrapping = TextWrapping.Wrap;
            v.TextTrimming = TextTrimming.None;
            v.IsTextSelectionEnabled = true;
            Grid.SetRow(l, r);
            Grid.SetRow(v, r);
            Grid.SetColumn(v, 1);
            rows.Children.Add(l);
            rows.Children.Add(v);
        }

        panel.Children.Add(Heading("Run Information", top: 0));
        Row("Computed on", b?.Summary ?? info.Backend);
        Row("Requested", b?.Requested);
        if (b is not null) Row("Route", RunRoute.Line(b.Requested, b.CostDecision));
        Row("Ran on the CPU", b?.FallbackReason);
        Row("Precision", info.Precision == MKLight.Engine.Precision.F64Reference ? "f64 reference" : "f32 (GPU, qualified)");
        Row("Tier", RunRoute.Tier(info.Tier == Tier.Preview));
        Row("Seed", info.Seed?.ToString(CultureInfo.InvariantCulture));
        Row("Snapshot", info.SnapshotHash.ToString() is { Length: > 12 } h ? h[..12] + "…" : info.SnapshotHash.ToString());
        Row("Engine", info.EngineVersion);
        Row("Time", Time(output));
        panel.Children.Add(rows);

        if (info.FailedAttempts.Count > 0)
        {
            panel.Children.Add(Divider());
            var n = info.FailedAttempts.Count;
            panel.Children.Add(Heading(n == 1 ? "Attempt Not Used" : $"{n.ToString(CultureInfo.InvariantCulture)} Attempts Not Used"));
            foreach (var a in info.FailedAttempts)
                panel.Children.Add(Item(RunRoute.AttemptTitle(a.Backend, a.Outcome, a.ElapsedMs), a.Reason, null));
        }

        if (info.Warnings.Count > 0)
        {
            panel.Children.Add(Divider());
            var groups = RunWarnings.Group(info.Warnings);
            panel.Children.Add(Heading(groups.Count == 1 ? "Warning" : $"{groups.Count.ToString(CultureInfo.InvariantCulture)} Warnings"));
            var defaults = session.Workspace.Descriptor(pane.Analysis)?.ToJson()["default_settings"] as JsonObject;
            JsonNode? Value(string key) => pane.Settings[key] ?? defaults?[key];
            foreach (var g in groups)
            {
                var fix = RunRemedy.For(g.Remedy, Value);
                panel.Children.Add(Item(g.Message, g.Scopes.Count > 0 ? string.Join(", ", g.Scopes) : null, fix is null ? null : RemedyButton(fix, pane)));
            }
            if (info.Warnings.Count > groups.Count)
            {
                var all = new StackPanel { Spacing = 2 };
                foreach (var w in info.Warnings)
                {
                    var t = Text(w, "CaptionText");
                    t.FontFamily = new FontFamily("Cascadia Mono, Consolas");
                    t.TextWrapping = TextWrapping.Wrap;
                    t.TextTrimming = TextTrimming.None;
                    t.IsTextSelectionEnabled = true;
                    all.Children.Add(t);
                }
                panel.Children.Add(new Expander
                {
                    Header = Text($"All {info.Warnings.Count.ToString(CultureInfo.InvariantCulture)} messages from the engine", "CaptionText"),
                    Content = all, HorizontalAlignment = HorizontalAlignment.Stretch, HorizontalContentAlignment = HorizontalAlignment.Left,
                    MinHeight = 0, Padding = new Thickness(12, 6, 8, 6), Margin = new Thickness(0, 4, 0, 0),
                });
            }
        }
    }

    static TextBlock Text(string text, string style) => new() { Text = text, Style = (Style)Application.Current.Resources[style] };

    static TextBlock Heading(string text, double top = 2)
    {
        var t = Text(text, "PanelTitleText");
        t.Margin = new Thickness(0, top, 0, 2);
        return t;
    }

    /// <summary>A hairline in the flyout's theme (resolved once shown).</summary>
    static Border Divider()
    {
        var line = new Border { Height = 1, Margin = new Thickness(0, 6, 0, 4) };
        Themed(line, () => line.Background = ThemeLookup.Brush(line, "DividerBrush"));
        return line;
    }

    static void Themed(FrameworkElement e, Action paint)
    {
        e.Loaded += (_, _) => paint();
        e.ActualThemeChanged += (_, _) => paint();
    }

    /// <summary>A warning mark and its text, a secondary line under it, and an optional action.</summary>
    static UIElement Item(string title, string? detail, UIElement? action)
    {
        var g = new Grid { ColumnSpacing = 8, RowSpacing = 2 };
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(14) });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        var glyph = new FontIcon { Glyph = "\uE7BA", FontSize = 12, VerticalAlignment = VerticalAlignment.Top, Margin = new Thickness(0, 3, 0, 0) };
        Themed(glyph, () => glyph.Foreground = ThemeLookup.Brush(glyph, "WarningBrush"));
        g.Children.Add(glyph);
        var lines = new StackPanel { Spacing = 2 };
        Grid.SetColumn(lines, 1);
        var t = Text(title, "NumberText");
        t.TextWrapping = TextWrapping.Wrap;
        t.TextTrimming = TextTrimming.None;
        t.IsTextSelectionEnabled = true;
        lines.Children.Add(t);
        if (!string.IsNullOrEmpty(detail))
        {
            var d = Text(detail, "CaptionText");
            d.TextWrapping = TextWrapping.Wrap;
            d.TextTrimming = TextTrimming.None;
            d.IsTextSelectionEnabled = true;
            lines.Children.Add(d);
        }
        if (action is not null) lines.Children.Add(action);
        g.Children.Add(lines);
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(g, string.IsNullOrEmpty(detail) ? title : $"{detail}: {title}");
        return g;
    }

    static Button RemedyButton(RunRemedy fix, AnalysisPaneModel pane)
    {
        var b = new Button
        {
            Content = fix.Title, FontSize = 12, MinHeight = 26, Padding = new Thickness(10, 2, 10, 3), CornerRadius = new CornerRadius(13),
            Margin = new Thickness(0, 4, 0, 2), HorizontalAlignment = HorizontalAlignment.Left,
        };
        Microsoft.UI.Xaml.Documents.Typography.SetNumeralAlignment(b, FontNumeralAlignment.Tabular);
        ToolTipService.SetToolTip(b, fix.Help);
        b.Click += (_, _) =>
        {
            var settings = (JsonObject)pane.Settings.DeepClone();
            settings[fix.Key] = fix.Value.DeepClone();
            pane.SetSettings(settings);
            b.IsEnabled = false;
        };
        return b;
    }

    public static string Precision(AnalysisOutput output) => output.Backend?.Precision switch
    {
        "f32_gpu" => "f32", "f64_reference" => "f64", { Length: > 0 } p => p,
        _ => output.RunInfo.Precision.ToString().ToLowerInvariant(),
    };

    public static string Time(AnalysisOutput output)
    {
        var ms = output.RunInfo.DurationMs;
        return ms >= 1000 ? (ms / 1000).ToString("0.0", CultureInfo.InvariantCulture) + " s" : ms.ToString("0", CultureInfo.InvariantCulture) + " ms";
    }

    /// <summary>
    /// The warnings the badge marks and lists in its tooltip: the engine's run warnings grouped by message,
    /// attempts not used, and a fallback from a backend that was asked for (an automatic request on a
    /// machine without a GPU is none).
    /// </summary>
    public static IReadOnlyList<string> Warnings(AnalysisOutput output)
    {
        var b = output.Backend;
        var fallback = b?.FallbackReason is not null && b.Requested is not ("auto" or "default:auto" or "" or "cpu_reference" or "default:cpu_reference") ? "Ran on the CPU: " + b.FallbackReason : null;
        return
        [
            .. RunWarnings.Group(output.RunInfo.Warnings).Select(g => g.Line),
            .. output.RunInfo.FailedAttempts.Select(a => RunRoute.AttemptTitle(a.Backend, a.Outcome, a.ElapsedMs)),
            .. fallback is null ? [] : new[] { fallback },
        ];
    }
}
