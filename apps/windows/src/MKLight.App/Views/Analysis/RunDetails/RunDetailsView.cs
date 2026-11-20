using System.Globalization;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using MKLight.App.ViewModels;
using MKLight.Engine;

namespace MKLight.App.Views.Analysis.RunDetails;

/// <summary>
/// The run details a pane's backend badge opens (the Mac app's RunDetails):
/// the backend that computed the result, what was requested and why, the
/// precision, seed, snapshot, engine and time, attempts not used and the
/// engine's warnings. Remedies (an explicit coarse grid back to automatic)
/// change the pane's settings through <see cref="AnalysisPaneModel.SetSettings"/>.
/// </summary>
public static class RunDetailsView
{
    /// <summary>Fill <paramref name="panel"/> (the badge's flyout) for <paramref name="output"/> of <paramref name="pane"/>.</summary>
    public static void Build(StackPanel panel, AnalysisOutput output, AnalysisPaneModel pane, DesignSession session)
    {
        panel.Children.Clear();
        var b = output.Backend;
        var precision = Precision(output);
        void Row(string label, string? value)
        {
            if (string.IsNullOrEmpty(value)) return;
            var g = new Grid { ColumnSpacing = 12 };
            g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(96) });
            g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            g.Children.Add(new TextBlock { Text = label, Style = (Style)Application.Current.Resources["SecondaryText"] });
            var v = new TextBlock { Text = value, Style = (Style)Application.Current.Resources["NumberText"], TextWrapping = TextWrapping.Wrap, TextTrimming = TextTrimming.None, IsTextSelectionEnabled = true };
            Grid.SetColumn(v, 1);
            g.Children.Add(v);
            panel.Children.Add(g);
        }
        panel.Children.Add(new TextBlock { Text = "Run Details", Style = (Style)Application.Current.Resources["PanelTitleText"], Margin = new Thickness(0, 0, 0, 4) });
        Row("Computed on", b?.Summary ?? output.RunInfo.Backend);
        Row("Requested", b?.Requested);
        Row("Reason", b?.FallbackReason);
        Row("Precision", precision);
        Row("Seed", output.RunInfo.Seed?.ToString(CultureInfo.InvariantCulture));
        Row("Snapshot", output.RunInfo.SnapshotHash.ToString() is { Length: > 12 } h ? h[..12] + "…" : output.RunInfo.SnapshotHash.ToString());
        Row("Engine", output.RunInfo.EngineVersion);
        Row("Time", Time(output));
        foreach (var a in output.RunInfo.FailedAttempts)
            Row("Not used", $"{a.Backend}: {a.Outcome} after {a.ElapsedMs.ToString("0", CultureInfo.InvariantCulture)} ms — {a.Reason}");
        foreach (var w in output.RunInfo.Warnings) Row("Warning", w);
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
    /// The warnings the badge marks and lists in its tooltip: the engine's run warnings, attempts not used,
    /// and a fallback from a backend that was asked for (an automatic request on a machine without a GPU is none).
    /// </summary>
    public static IReadOnlyList<string> Warnings(AnalysisOutput output)
    {
        var b = output.Backend;
        var fallback = b?.FallbackReason is not null && b.Requested is not ("auto" or "default:auto" or "" or "cpu_reference" or "default:cpu_reference") ? b.FallbackReason : null;
        return [.. output.RunInfo.Warnings, .. output.RunInfo.FailedAttempts.Select(a => $"{a.Backend}: {a.Outcome} — {a.Reason}"), .. fallback is null ? [] : new[] { fallback }];
    }
}
