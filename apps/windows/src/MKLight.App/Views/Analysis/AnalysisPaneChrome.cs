using MKLight.App.Support;
using MKLight.Model;

namespace MKLight.App.Views.Analysis;

/// <summary>What an analysis pane without a result shows instead (the Mac app's placeholders).</summary>
public enum PanePlaceholderKind { None, Running, Unavailable, DesignHasErrors, Failed, NotReady, Cancelled, OnRequest, Waiting }

/// <summary>A placeholder: glyph, title, description and the one action (null: none).</summary>
public sealed record PanePlaceholder(PanePlaceholderKind Kind, string Glyph, string Title, string Text, string? Action);

/// <summary>The facts about a pane the chrome decides from (no UI or engine types, so it is unit-tested).</summary>
public readonly record struct PaneFacts
{
    public bool HasOutput { get; init; }
    public bool Running { get; init; }
    public bool Failed { get; init; }
    public bool Unavailable { get; init; }
    public bool Cancelled { get; init; }
    public bool AutoUpdate { get; init; }
    /// <summary>The engine's message for a failed or unavailable run.</summary>
    public string? Message { get; init; }
    /// <summary>The running stage ("Tracing rays").</summary>
    public string? Stage { get; init; }
    /// <summary>The design has issues of severity error.</summary>
    public bool DesignHasErrors { get; init; }
    /// <summary>The first reason the design cannot be snapshotted (null: it can).</summary>
    public string? ModelError { get; init; }
}

/// <summary>
/// The analysis pane's chrome decisions: which placeholder a pane without a
/// result shows, when the header's cluster folds into More, when a rerun
/// failed over a result that stays, when the result pulses, and the
/// automation ids UI tests find the pane's parts by.
/// </summary>
public static class AnalysisPaneChrome
{
    /// <summary>Below this pane width the cluster always folds into Run and More.</summary>
    public const double FoldWidth = 240;

    public const string GlyphNotAvailable = "\uE733";
    public const string GlyphErrors = "\uEA39";
    public const string GlyphFailed = "\uE7BA";
    public const string GlyphCancelled = "\uE71A";

    /// <summary>The placeholder of a pane (<see cref="PanePlaceholderKind.None"/> while it has a result).</summary>
    public static PanePlaceholder Placeholder(PaneFacts f, string analysisGlyph = "")
    {
        // The engine's messages and stages are engine text; the rest is the app's.
        if (f.Unavailable)
            return new(PanePlaceholderKind.Unavailable, GlyphNotAvailable, Loc.S("Analysis_Placeholder_NotAvailable"),
                string.IsNullOrEmpty(f.Message) ? Loc.S("Analysis_Placeholder_NotAvailableText") : EngineText.T(f.Message), null);
        if (f.HasOutput) return new(PanePlaceholderKind.None, "", "", "", null);
        if (f.Running)
            return new(PanePlaceholderKind.Running, "", "", string.IsNullOrEmpty(f.Stage) ? Loc.S("Analysis_Placeholder_Computing") : EngineText.T(f.Stage), Loc.S("Common_Cancel"));
        if (f.Failed)
            return f.DesignHasErrors
                ? new(PanePlaceholderKind.DesignHasErrors, GlyphErrors, Loc.S("Analysis_Placeholder_DesignHasErrors"), EngineText.T(f.Message ?? ""), Loc.S("Analysis_Placeholder_ShowIssues"))
                : new(PanePlaceholderKind.Failed, GlyphFailed, Loc.S("Analysis_Placeholder_Failed"),
                    string.IsNullOrEmpty(f.Message) ? Loc.S("Analysis_Placeholder_FailedText") : EngineText.T(f.Message), Loc.S("Analysis_Placeholder_TryAgain"));
        if (!string.IsNullOrEmpty(f.ModelError))
            return new(PanePlaceholderKind.NotReady, GlyphErrors, Loc.S("Analysis_Placeholder_NotReady"), EngineText.T(f.ModelError), Loc.S("Analysis_Placeholder_ShowIssues"));
        if (f.Cancelled) return new(PanePlaceholderKind.Cancelled, GlyphCancelled, Loc.S("Analysis_Placeholder_Cancelled"), "", Loc.S("Analysis_Placeholder_Run"));
        return f.AutoUpdate
            ? new(PanePlaceholderKind.Waiting, "", "", "", null)
            : new(PanePlaceholderKind.OnRequest, analysisGlyph, Loc.S("Analysis_Placeholder_NotRunYet"), Loc.S("Analysis_Placeholder_OnRequest"), Loc.S("Analysis_Placeholder_Run"));
    }

    /// <summary>A rerun failed while the pane has a result: the result stays, a capsule says the update failed.</summary>
    public static bool RerunFailed(PaneFacts f) => f.HasOutput && f.Failed && !f.Running && !f.Unavailable;

    /// <summary>The result breathes while it updates (a rerun over an existing result), unless motion is reduced.</summary>
    public static bool Pulses(PaneFacts f, bool motion) => motion && f.Running && f.HasOutput;

    /// <summary>
    /// How the header fits a pane <paramref name="width"/> wide (the Mac's order): the cluster folds into
    /// Run and More first, then the title takes its short form. <paramref name="title"/> and
    /// <paramref name="shortTitle"/> are the title texts' widths, <paramref name="cluster"/> and
    /// <paramref name="compactCluster"/> the cluster's, <paramref name="chrome"/> everything else in the row
    /// (padding, glyph, chevron, spacing). Below <see cref="FoldWidth"/> the cluster always folds.
    /// </summary>
    public static (bool Compact, bool ShortTitle) Fold(double width, double title, double shortTitle, double cluster, double compactCluster, double chrome)
    {
        if (width <= 0) return (false, false);
        var mustFold = width < FoldWidth;
        if (!mustFold && chrome + title + cluster <= width) return (false, false);
        if (chrome + title + compactCluster <= width) return (true, false);
        return (true, shortTitle < title);
    }

    /// <summary>A stable UI-automation id of a pane's part: <c>Pane.{analysis}.{part}</c>.</summary>
    public static string AutomationId(string analysis, string part) => $"Pane.{analysis}.{part}";

    /// <summary>The preview banner's label.</summary>
    public static string PreviewLabel(string candidate) => Loc.F("Analysis_Pane_PreviewLabel", candidate);

    /// <summary>The pane's accessibility name: the analysis, and the preview while one is shown.</summary>
    public static string AccessibleName(string title, string? preview) =>
        preview is null ? Loc.F("Analysis_Pane_AccessibleName", title) : Loc.F("Analysis_Pane_AccessibleNamePreview", title, PreviewLabel(preview));
}
