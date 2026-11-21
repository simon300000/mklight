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
using MKLight.Model;

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
        // The cluster's names and the More tooltip (the XAML's English is the fallback).
        var more = Loc.S("Analysis_Pane_More");
        ToolTipService.SetToolTip(MoreButton, more);
        AutomationProperties.SetName(MoreButton, more);
        AutomationProperties.SetName(AutoButton, Loc.S("Analysis_More_AutoUpdate"));
        AutomationProperties.SetName(SettingsButton, Loc.S("Analysis_Pane_Settings"));
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
        ToolTipService.SetToolTip(TitleButton, (string.IsNullOrEmpty(d?.Description) ? title : Loc.F("Analysis_Pane_TitleTip", title, EngineText.T(d.Description)))
            + "\n" + Loc.S("Analysis_Pane_TitleTipDrag"));
        TitleGlyph.Glyph = AnalysisCatalog.Glyph(_pane.Analysis);
        if (AnalysisMenu.Items.Count == 0) BuildAnalysisMenu();
        AutoButton.IsChecked = _pane.AutoUpdate;
        ToolTipService.SetToolTip(AutoButton, Loc.S(_pane.AutoUpdate ? "Analysis_Pane_AutoOnTip" : "Analysis_Pane_AutoOffTip"));
        SettingsButton.IsChecked = _pane.ShowsSettings;
        ToolTipService.SetToolTip(SettingsButton, Loc.S(_pane.ShowsSettings ? "Analysis_Pane_HideSettingsTip" : "Analysis_Pane_ShowSettingsTip"));

        var facts = Facts();
        var stale = _pane.IsStale;
        var running = facts.Running;
        Progress.Visibility = running ? Visibility.Visible : Visibility.Collapsed;
        if (running && Fraction > 0) { Progress.IsIndeterminate = false; Progress.Value = Fraction * 100; }
        else Progress.IsIndeterminate = true;
        RunButton.Content = running ? "\uE711" : "\uE72C";
        ToolTipService.SetToolTip(RunButton, Loc.S(running ? "Analysis_Pane_CancelTip" : "Analysis_Pane_RunTip"));
        AutomationProperties.SetName(RunButton, Loc.S(running ? "Common_Cancel" : "Analysis_Placeholder_Run"));

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
            placeholder = new(PanePlaceholderKind.Failed, AnalysisPaneChrome.GlyphFailed, Loc.S("Analysis_Placeholder_NotDrawn"), Loc.F("Analysis_Placeholder_NotDrawnText", why), null);
        ShowPlaceholder(placeholder);
        if (placeholder.Kind == PanePlaceholderKind.Unavailable) ResultHost.Content = null;
        PulseHost.Visibility = Footer.Visibility = placeholder.Kind == PanePlaceholderKind.None ? Visibility.Visible : Visibility.Collapsed;

        // Stale: dim the result, show the card (the pulse says it is updating while it re-runs).
        var showStale = stale && output is not null && !facts.Unavailable;
        ResultHost.OpacityTransition = Motion.Enabled ? _fade : null;
        ResultHost.Opacity = showStale ? (running ? 0.55 : 0.3) : 1;
        StaleCard.Visibility = showStale && !running ? Visibility.Visible : Visibility.Collapsed;
        var blocked = _session.Workspace.ModelErrors.Count > 0;
        StaleTitle.Text = Loc.S(blocked ? "Analysis_Pane_StaleErrors" : "Analysis_Pane_StaleChanged");
        StaleReason.Text = Loc.S(blocked ? "Analysis_Pane_StaleErrorsReason" : "Analysis_Pane_StaleChangedReason");
        StaleUpdate.Visibility = blocked ? Visibility.Collapsed : Visibility.Visible;
        SetPulse(AnalysisPaneChrome.Pulses(facts, Motion.Enabled));

        // A rerun failed over a result: the result stays, the capsule says why.
        var rerunFailed = AnalysisPaneChrome.RerunFailed(facts);
        FailedCapsule.Visibility = rerunFailed ? Visibility.Visible : Visibility.Collapsed;
        var failure = string.IsNullOrEmpty(facts.Message) ? Loc.S("Analysis_Placeholder_FailedText") : EngineText.T(facts.Message);
        FailedText.Text = Loc.F("Analysis_Pane_FailedCapsule", failure);
        FailedMessage.Text = failure;
        ToolTipService.SetToolTip(FailedCapsule, rerunFailed ? Loc.F("Analysis_Pane_FailedTip", failure) + "\n" + Loc.S("Analysis_Pane_FailedTipNote") : null);
        AutomationProperties.SetName(FailedCapsule, Loc.F("Analysis_Pane_FailedName", failure));

        // Preview: the result is of a candidate, not of the design.
        var previewing = _session.Workspace.PreviewTitle;
        PreviewBanner.Visibility = previewing is null ? Visibility.Collapsed : Visibility.Visible;
        PreviewTitleRun.Text = previewing is null ? "" : Loc.F("Analysis_Pane_PreviewTitle", previewing);
        PreviewNoteRun.Text = Loc.S("Analysis_Pane_PreviewNote");
        AutomationProperties.SetName(PreviewBanner, previewing is null ? "" : AnalysisPaneChrome.PreviewLabel(previewing));
        ToolTipService.SetToolTip(PreviewBanner, previewing is null ? null : Loc.F("Analysis_Pane_PreviewTip", previewing));

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
        if (facts.Running)
            return facts.Stage is { Length: > 0 } s
                ? Fraction > 0 ? Loc.F("Analysis_Pane_StageProgress", EngineText.T(s), (Fraction * 100).ToString("0", CultureInfo.InvariantCulture)) : EngineText.T(s)
                : Loc.S("Analysis_Placeholder_Computing");
        var summary = d is null ? "" : SettingsSummary.Describe(d.ToJson(), _pane.Settings, _data);
        if (stale && _pane.Output is not null) return string.IsNullOrEmpty(summary) ? Loc.S("Analysis_Pane_OutOfDate") : Loc.F("Analysis_Pane_OutOfDateSummary", summary);
        if (!string.IsNullOrEmpty(summary)) return summary;
        return Loc.S(_pane.AutoUpdate ? "Analysis_Pane_DefaultSettings" : "Analysis_Pane_DefaultSettingsManual");
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
        AutomationProperties.SetName(MessagePanel, p.Title.Length > 0 && p.Text.Length > 0 ? Loc.F("Analysis_Placeholder_Name", p.Title, p.Text)
            : p.Title.Length > 0 ? p.Title : p.Text.Length > 0 ? p.Text : Loc.S("Analysis_Placeholder_Waiting"));
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
        ToolTipService.SetToolTip(BackendBadge, warnings.Count > 0 ? string.Join(Environment.NewLine, warnings) : Loc.S("Analysis_Pane_RunDetailsTip"));
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
            SurfaceLabels = Enumerable.Range(0, n).Select(i => i == 0 ? Loc.S("Analysis_Settings_SurfaceObject") : i == n - 1 ? Loc.S("Analysis_Settings_SurfaceImage")
                : i == sys.Stop ? Loc.F("Analysis_Settings_SurfaceStop", i) : i.ToString(CultureInfo.InvariantCulture)).ToList(),
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
        _session.ShowToast(Loc.S("Analysis_Pane_CopiedJson"));
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
