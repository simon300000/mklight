using System.Globalization;
using System.Text.Json.Nodes;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Support;
using MKLight.App.Views.Analysis.Results;
using MKLight.App.Views.Analysis.RunDetails;
using MKLight.App.Views.Analysis.Settings;
using MKLight.App.ViewModels;
using MKLight.Engine;

namespace MKLight.App.Views.Analysis;

/// <summary>
/// An analysis pane (the Mac app's AnalysisPaneView): title with the
/// analysis menu (and the handle the pane is dragged by), a state line,
/// Run / automatic update / Settings / More, the preview banner while a
/// candidate is previewed, the result as content, key numbers and the
/// backend badge in the footer, the "Design changed" card with Update when
/// the result is stale, placeholders when there is no result, and a capsule
/// when a rerun failed over a result that stays.
/// </summary>
public sealed partial class AnalysisPaneView : UserControl
{
    readonly DesignSession _session;
    readonly AnalysisPaneModel _pane;
    AnalysisOutput? _shown;
    AnalysisOutput? _retired;
    ResultData? _data;
    ElementTheme _shownTheme;
    PanePlaceholder? _placeholder;

    public AnalysisPaneView(DesignSession session, AnalysisPaneModel pane)
    {
        _session = session;
        _pane = pane;
        InitializeComponent();
        _fade = ResultHost.OpacityTransition;
        ActualThemeChanged += (_, _) => Update();
        // Follow the pane while in the tree; the window re-parents views when panes change.
        Loaded += (_, _) =>
        {
            pane.Changed -= Update; pane.Changed += Update;
            session.Workspace.PreviewChanged -= Update; session.Workspace.PreviewChanged += Update;
            Motion.Changed -= OnMotionChanged; Motion.Changed += OnMotionChanged;
            Update();
        };
        Unloaded += (_, _) =>
        {
            pane.Changed -= Update;
            session.Workspace.PreviewChanged -= Update;
            Motion.Changed -= OnMotionChanged;
            SetPulse(false);
        };
        SizeChanged += (_, e) => { if (e.NewSize.Width != e.PreviousSize.Width) UpdateFold(); };
        PointerPressed += (_, _) => Focused?.Invoke(this);
        BuildAnalysisMenu();
    }

    public AnalysisPaneModel Pane => _pane;
    public event Action<AnalysisPaneView>? CloseRequested;
    public event Action<AnalysisPaneView>? Focused;

    public void SetFocused(bool focused) => FocusBar.Opacity = focused ? 1 : 0;

    void BuildAnalysisMenu()
    {
        AnalysisMenu.Items.Clear();
        foreach (var group in _session.Workspace.PaneAnalyses.GroupBy(d => d.Category).OrderBy(g => AnalysisCatalog.CategoryRank(g.Key)).ThenBy(g => g.Key))
        {
            var sub = new MenuFlyoutSubItem { Text = AnalysisCatalog.CategoryTitle(group.Key) };
            foreach (var d in group.OrderBy(d => AnalysisCatalog.Title(d.Name, d.Title)))
            {
                var item = new MenuFlyoutItem { Text = AnalysisCatalog.Title(d.Name, d.Title), IsEnabled = d.Available, Icon = new FontIcon { Glyph = AnalysisCatalog.Glyph(d.Name) } };
                var name = d.Name;
                item.Click += (_, _) => _pane.Show(name, _session.Workspace.StartingSettings(name));
                sub.Items.Add(item);
            }
            AnalysisMenu.Items.Add(sub);
        }
    }

    void Update()
    {
        var d = _session.Workspace.Descriptor(_pane.Analysis);
        var title = AnalysisCatalog.Title(_pane.Analysis, d?.Title);
        ToolTipService.SetToolTip(TitleButton, title + (string.IsNullOrEmpty(d?.Description) ? "" : ": " + d.Description)
            + "\nDrag to move the pane, or out of the window to open it in a window of its own.");
        TitleGlyph.Glyph = AnalysisCatalog.Glyph(_pane.Analysis);
        if (AnalysisMenu.Items.Count == 0) BuildAnalysisMenu();
        AutoButton.IsChecked = _pane.AutoUpdate;
        ToolTipService.SetToolTip(AutoButton, _pane.AutoUpdate
            ? "Updates when the design changes. Click to update on request only."
            : "Updates on request only. Click to update automatically.");
        SettingsButton.IsChecked = _pane.ShowsSettings;
        ToolTipService.SetToolTip(SettingsButton, _pane.ShowsSettings ? "Hide the settings" : "Show the settings of this analysis");

        var facts = Facts();
        var stale = _pane.IsStale;
        var running = facts.Running;
        Progress.Visibility = running ? Visibility.Visible : Visibility.Collapsed;
        if (running && Fraction > 0) { Progress.IsIndeterminate = false; Progress.Value = Fraction * 100; }
        else Progress.IsIndeterminate = true;
        RunButton.Content = running ? "\uE711" : "\uE72C";
        ToolTipService.SetToolTip(RunButton, running ? "Cancel this analysis" : "Run again (F5)");
        AutomationProperties.SetName(RunButton, running ? "Cancel" : "Run");

        // Content: the result view, rebuilt only when the output or the theme changes.
        var output = _pane.Output;
        if (output is not null && (!ReferenceEquals(output, _shown) || _shownTheme != ActualTheme))
        {
            // The view of the previous output goes now: free its buffers (unless a preview keeps it as baseline).
            if (_shown is { } old && !ReferenceEquals(old, output) && !ReferenceEquals(old, _pane.Output) && !_session.Workspace.IsRetained(old)) _retired = old;
            _shown = output;
            _shownTheme = ActualTheme;
            try
            {
                _data = ResultHosting.Data(_pane.Analysis, output, _session.Workspace.PreviewSystem ?? _session.Document.System);
                var ws = _session.Workspace;
                if (ws.IsPreviewing && ws.Baseline.TryGetValue(_pane.Id, out var designOutput) && !ReferenceEquals(designOutput, output))
                    _data = new ResultData
                    {
                        Analysis = _data.Analysis, Json = _data.Json, Buffers = _data.Buffers, WavelengthsUm = _data.WavelengthsUm,
                        FieldLabels = _data.FieldLabels, Settings = _data.Settings,
                        Baseline = ResultHosting.Data(_pane.Analysis, designOutput, _session.Document.System),
                    };
                var (view, summary) = ResultViews.Create(_data, ActualTheme, _pane.SpotScale);
                if (view is SpotDiagramView spot) spot.BoxScaleChanged += scale => _pane.SpotScale = scale;
                ResultHost.Content = view;
                _retired?.Dispose();
                _retired = null;
                // The form's "Auto (n)" values come from the latest result.
                if (_pane.ShowsSettings) SettingsContent.Content = null;
                KeyNumbers.Text = summary.KeyNumbers;
                ToolTipService.SetToolTip(KeyNumbers, string.IsNullOrEmpty(summary.KeyNumbers) ? null : summary.KeyNumbers); // narrow panes cut it
                _drawError = null;
            }
            catch (Exception e)
            {
                ResultHost.Content = null;
                KeyNumbers.Text = "";
                _drawError = e.Message;
            }
            UpdateBackendBadge(output);
        }
        if (output is null) { ResultHost.Content = null; _shown = null; _drawError = null; KeyNumbers.Text = ""; BackendBadge.Visibility = Visibility.Collapsed; }
        StateText.Text = StateLine(d, facts, stale); // after the result is read: automatic values come from it

        // Placeholders for panes without a result (or a result that could not be drawn).
        var placeholder = AnalysisPaneChrome.Placeholder(facts, AnalysisCatalog.Glyph(_pane.Analysis));
        if (placeholder.Kind == PanePlaceholderKind.None && _drawError is { } why)
            placeholder = new(PanePlaceholderKind.Failed, AnalysisPaneChrome.GlyphFailed, "Result Not Drawn", $"This result could not be drawn: {why}", null);
        ShowPlaceholder(placeholder);
        if (placeholder.Kind == PanePlaceholderKind.Unavailable) ResultHost.Content = null;
        PulseHost.Visibility = Footer.Visibility = placeholder.Kind == PanePlaceholderKind.None ? Visibility.Visible : Visibility.Collapsed;

        // Stale: dim the result, show the card (the pulse says it is updating while it re-runs).
        var showStale = stale && output is not null && !facts.Unavailable;
        ResultHost.OpacityTransition = Motion.Enabled ? _fade : null;
        ResultHost.Opacity = showStale ? (running ? 0.55 : 0.3) : 1;
        StaleCard.Visibility = showStale && !running ? Visibility.Visible : Visibility.Collapsed;
        var blocked = _session.Workspace.ModelErrors.Count > 0;
        StaleTitle.Text = blocked ? "The design has errors" : "Design changed";
        StaleReason.Text = blocked ? "Fix them to update this result." : "This result is from an earlier design.";
        StaleUpdate.Visibility = blocked ? Visibility.Collapsed : Visibility.Visible;
        SetPulse(AnalysisPaneChrome.Pulses(facts, Motion.Enabled));

        // A rerun failed over a result: the result stays, the capsule says why.
        var rerunFailed = AnalysisPaneChrome.RerunFailed(facts);
        FailedCapsule.Visibility = rerunFailed ? Visibility.Visible : Visibility.Collapsed;
        var failure = string.IsNullOrEmpty(facts.Message) ? "The analysis failed." : facts.Message;
        FailedText.Text = "Update failed · " + failure;
        FailedMessage.Text = failure;
        ToolTipService.SetToolTip(FailedCapsule, rerunFailed ? "The update failed: " + failure + "\nThe result shown is from the last run that succeeded. Click for details." : null);
        AutomationProperties.SetName(FailedCapsule, "Update failed: " + failure);

        // Preview: the result is of a candidate, not of the design.
        var previewing = _session.Workspace.PreviewTitle;
        PreviewBanner.Visibility = previewing is null ? Visibility.Collapsed : Visibility.Visible;
        PreviewTitleRun.Text = previewing is null ? "" : $"Previewing {previewing}";
        AutomationProperties.SetName(PreviewBanner, previewing is null ? "" : AnalysisPaneChrome.PreviewLabel(previewing));
        ToolTipService.SetToolTip(PreviewBanner, previewing is null ? null
            : $"This pane shows {previewing}, a candidate that is not applied to the design. Apply it or end the preview in the design job.");

        // Settings form.
        SettingsHost.Visibility = _pane.ShowsSettings ? Visibility.Visible : Visibility.Collapsed;
        if (_pane.ShowsSettings && SettingsContent.Content is null) BuildSettings();

        UpdateAutomation(title, previewing);
        UpdateFold();
    }

    string? _drawError;

    /// <summary>The running fraction (0: not known; a forced state has none).</summary>
    double Fraction => ForcedFacts is null ? _pane.Progress.Fraction : 0;

    string StateLine(AnalysisDescriptor? d, PaneFacts facts, bool stale)
    {
        if (facts.Running) return facts.Stage is { Length: > 0 } s ? s + (Fraction > 0 ? " · " + (Fraction * 100).ToString("0", CultureInfo.InvariantCulture) + "%" : "") : "Computing…";
        var summary = d is null ? "" : SettingsSummary.Describe(d.ToJson(), _pane.Settings, _data);
        if (stale && _pane.Output is not null) return string.IsNullOrEmpty(summary) ? "Out of date" : "Out of date · " + summary;
        if (!string.IsNullOrEmpty(summary)) return summary;
        return _pane.AutoUpdate ? "Default settings" : "Default settings · manual";
    }

    void ShowPlaceholder(PanePlaceholder p)
    {
        _placeholder = p;
        if (p.Kind == PanePlaceholderKind.None) { MessagePanel.Visibility = Visibility.Collapsed; MessageRing.IsActive = false; return; }
        MessagePanel.Visibility = Visibility.Visible;
        MessageGlyph.Glyph = p.Glyph;
        MessageGlyph.Visibility = p.Glyph.Length > 0 ? Visibility.Visible : Visibility.Collapsed;
        MessageGlyph.Foreground = ThemeLookup.Brush(this, p.Kind switch
        {
            PanePlaceholderKind.Failed => "WarningBrush",
            PanePlaceholderKind.DesignHasErrors or PanePlaceholderKind.NotReady => "ErrorBrush",
            _ => "TextFillColorTertiaryBrush",
        });
        var running = p.Kind == PanePlaceholderKind.Running;
        MessageProgress.Visibility = running ? Visibility.Visible : Visibility.Collapsed;
        if (running && Fraction > 0) { MessageProgress.IsIndeterminate = false; MessageProgress.Value = Fraction * 100; }
        else MessageProgress.IsIndeterminate = true;
        var waiting = p.Kind == PanePlaceholderKind.Waiting;
        MessageRing.Visibility = waiting ? Visibility.Visible : Visibility.Collapsed;
        MessageRing.IsActive = waiting;
        MessageTitle.Text = p.Title;
        MessageTitle.Visibility = p.Title.Length > 0 ? Visibility.Visible : Visibility.Collapsed;
        MessageText.Text = p.Text;
        MessageText.Visibility = p.Text.Length > 0 ? Visibility.Visible : Visibility.Collapsed;
        MessageText.FontSize = running ? 11 : 12.5;
        MessageAction.Visibility = p.Action is null ? Visibility.Collapsed : Visibility.Visible;
        MessageAction.Content = p.Action;
        AutomationProperties.SetName(MessagePanel, string.Join(". ", new[] { p.Title, p.Text }.Where(s => s.Length > 0)) is { Length: > 0 } n ? n : "Waiting for the design");
    }

    void UpdateBackendBadge(AnalysisOutput output)
    {
        var gpu = output.Backend?.Used.StartsWith("gpu", StringComparison.Ordinal) == true;
        BackendText.Text = RunDetailsView.Badge(output) + "  " + RunDetailsView.Time(output);
        BackendGlyph.Glyph = gpu ? "\uE945" : "\uE950";
        var brush = ThemeLookup.Brush(this, gpu ? "AccentTextFillColorPrimaryBrush" : "TextFillColorSecondaryBrush");
        BackendText.Foreground = brush;
        BackendGlyph.Foreground = brush;
        var warnings = RunDetailsView.Warnings(output);
        BackendWarning.Visibility = warnings.Count > 0 ? Visibility.Visible : Visibility.Collapsed;
        BackendBadge.Visibility = Visibility.Visible;
        ToolTipService.SetToolTip(BackendBadge, warnings.Count > 0 ? string.Join(Environment.NewLine, warnings) : "Run details");
        RunDetailsView.Build(RunDetails, output, _pane, _session);
    }

    void BuildSettings()
    {
        var d = _session.Workspace.Descriptor(_pane.Analysis);
        if (d is null) return;
        var sys = _session.Document.System;
        var n = sys.Surfaces.Count;
        var ctx = new SettingsContext
        {
            Analysis = _pane.Analysis,
            FieldLabels = ResultHosting.FieldLabels(sys),
            WavelengthsUm = sys.Wavelengths.Wavelengths.Select(w => w.Um).ToList(),
            PrimaryWavelength = sys.Wavelengths.Primary,
            SurfaceLabels = Enumerable.Range(0, n).Select(i => i == 0 ? "Object" : i == n - 1 ? "Image"
                : i == sys.Stop ? $"{i} (stop)" : i.ToString(CultureInfo.InvariantCulture)).ToList(),
            HasConfigurations = sys.Configurations.Count > 0,
            LastResult = _data,
        };
        SettingsContent.Content = SettingsForm.Create(d.ToJson(), (JsonObject)_pane.Settings.DeepClone(), ctx, s => _pane.SetSettings(s));
    }

    // ---- Actions ----

    /// <summary>Run asks for a fresh run (not a cached result); while running it cancels.</summary>
    void OnRun(object sender, RoutedEventArgs e)
    {
        if (_pane.IsRunning) _pane.Cancel();
        else _pane.Rerun(fresh: true);
    }

    /// <summary>The stale card's Update: the result for the design as it is now.</summary>
    void OnUpdate(object sender, RoutedEventArgs e) => _pane.Rerun();

    void OnRetry(object sender, RoutedEventArgs e)
    {
        FailedCapsule.Flyout?.Hide();
        _pane.Rerun(fresh: true);
    }

    void OnMessageAction(object sender, RoutedEventArgs e)
    {
        switch (_placeholder?.Kind)
        {
            case PanePlaceholderKind.Running: _pane.Cancel(); break;
            case PanePlaceholderKind.DesignHasErrors or PanePlaceholderKind.NotReady: ShowIssues(); break;
            default: _pane.Rerun(fresh: true); break;
        }
    }

    /// <summary>Opens the status capsule's issues (the window has no command for it yet).</summary>
    void ShowIssues()
    {
        var window = App.Windows.OfType<Windows.DesignWindow>().FirstOrDefault(w => w.Session == _session);
        if (window is not null && Descendants<StatusCapsule>(window.RootElement).FirstOrDefault() is { } status) status.ShowIssues();
    }

    void OnAutoUpdate(object sender, RoutedEventArgs e) => SetAutoUpdate(AutoButton.IsChecked == true);

    void SetAutoUpdate(bool on)
    {
        _pane.AutoUpdate = on;
        _pane.RaiseChanged();
        if (_pane.AutoUpdate && _pane.IsStale) _pane.Rerun();
        Update();
    }

    void OnSettingsToggle(object sender, RoutedEventArgs e) => SetShowsSettings(SettingsButton.IsChecked == true);

    void SetShowsSettings(bool on)
    {
        _pane.ShowsSettings = on;
        _pane.RaiseChanged();
        if (!_pane.ShowsSettings) SettingsContent.Content = null;
        Update();
    }

    void OnSettingsDone(object sender, RoutedEventArgs e) => SetShowsSettings(false);

    void OnDefaults(object sender, RoutedEventArgs e)
    {
        _pane.SetSettings(_session.Workspace.StartingSettings(_pane.Analysis));
        SettingsContent.Content = null;
        Update();
    }

    void CopyJson()
    {
        if (_pane.Output?.Json is not { } json) return;
        var p = new global::Windows.ApplicationModel.DataTransfer.DataPackage();
        p.SetText(MKLight.Model.ModelJson.ToJsonString(json, pretty: true));
        global::Windows.ApplicationModel.DataTransfer.Clipboard.SetContent(p);
        _session.ShowToast("Copied the result as JSON.");
    }

    /// <summary>A second pane with this one's analysis and settings, beside it.</summary>
    void Duplicate() => _session.Workspace.AddPane(_pane.Analysis, (JsonObject)_pane.Settings.DeepClone(), after: _pane);

    void Close() => CloseRequested?.Invoke(this);

    internal static IEnumerable<T> Descendants<T>(DependencyObject root) where T : DependencyObject
    {
        var n = VisualTreeHelper.GetChildrenCount(root);
        for (var i = 0; i < n; i++)
        {
            var child = VisualTreeHelper.GetChild(root, i);
            if (child is T t) yield return t;
            foreach (var d in Descendants<T>(child)) yield return d;
        }
    }
}
