using System.Globalization;
using System.Text.Json.Nodes;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Support;
using MKLight.App.Views.Analysis.Results;
using MKLight.App.Views.Analysis.RunDetails;
using MKLight.App.Views.Analysis.Results.Readers;
using MKLight.App.Views.Analysis.Settings;
using MKLight.App.ViewModels;
using MKLight.Engine;
using MKLight.Model;
using Windows.ApplicationModel.DataTransfer;

namespace MKLight.App.Views.Analysis;

/// <summary>
/// An analysis pane (the Mac app's AnalysisPaneView): title with the
/// analysis menu, a state line, Run / automatic update / Settings / More,
/// the result as content, key numbers and the backend badge in the footer,
/// and the "Design changed" card with Update when the result is stale.
/// </summary>
public sealed partial class AnalysisPaneView : UserControl
{
    readonly DesignSession _session;
    readonly AnalysisPaneModel _pane;
    AnalysisOutput? _shown;
    AnalysisOutput? _retired;
    ResultData? _data;
    ElementTheme _shownTheme;

    public AnalysisPaneView(DesignSession session, AnalysisPaneModel pane)
    {
        _session = session;
        _pane = pane;
        InitializeComponent();
        ActualThemeChanged += (_, _) => Update();
        // Follow the pane while in the tree; the window re-parents views when panes change.
        Loaded += (_, _) => { pane.Changed -= Update; pane.Changed += Update; session.Workspace.PreviewChanged -= Update; session.Workspace.PreviewChanged += Update; Update(); };
        Unloaded += (_, _) => { pane.Changed -= Update; session.Workspace.PreviewChanged -= Update; };
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
        TitleText.Text = ActualWidth is > 0 and < 260 ? AnalysisCatalog.ShortTitle(_pane.Analysis) : title;
        ToolTipService.SetToolTip(TitleButton, title + (d is null ? "" : "\n" + d.Description));
        TitleGlyph.Glyph = AnalysisCatalog.Glyph(_pane.Analysis);
        if (AnalysisMenu.Items.Count == 0) BuildAnalysisMenu();
        AutoButton.IsChecked = _pane.AutoUpdate;
        SettingsButton.IsChecked = _pane.ShowsSettings;

        var stale = _pane.IsStale;
        var running = _pane.IsRunning;
        Progress.Visibility = running ? Visibility.Visible : Visibility.Collapsed;
        if (running && _pane.Progress.Fraction > 0) { Progress.IsIndeterminate = false; Progress.Value = _pane.Progress.Fraction * 100; }
        else Progress.IsIndeterminate = true;
        RunButton.Content = running ? "\uE711" : "\uE72C";
        ToolTipService.SetToolTip(RunButton, running ? "Cancel" : "Run (F5)");

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
            }
            catch (Exception e)
            {
                ResultHost.Content = null;
                KeyNumbers.Text = "";
                ShowMessage("\uE9CE", $"This result could not be drawn: {e.Message}", null);
            }
            UpdateBackendBadge(output);
        }
        if (output is null) { ResultHost.Content = null; _shown = null; KeyNumbers.Text = ""; BackendBadge.Visibility = Visibility.Collapsed; }
        StateText.Text = StateLine(d, running); // after the result is read: automatic values come from it

        // Messages for panes without a result.
        switch (_pane.Phase)
        {
            case PanePhase.Failed when output is null: ShowMessage("\uE783", _pane.Message ?? "The analysis failed.", "Try Again"); break;
            case PanePhase.Unavailable: ShowMessage("\uE7BA", _pane.Message ?? "Not available.", null); break;
            case PanePhase.Idle or PanePhase.Cancelled when output is null && !running:
                ShowMessage(AnalysisCatalog.Glyph(_pane.Analysis), _pane.AutoUpdate ? "Waiting for the design." : "This analysis runs on request.", _pane.AutoUpdate ? null : "Run");
                break;
            default:
                if (output is null && running) ShowMessage(AnalysisCatalog.Glyph(_pane.Analysis), _pane.Progress.Stage is { Length: > 0 } s ? s : "Computing…", null);
                else MessagePanel.Visibility = Visibility.Collapsed;
                break;
        }
        if (_pane.Phase == PanePhase.Unavailable) ResultHost.Content = null;

        // Stale: dim the result, show the card (or a quiet "Updating" line while an automatic pane re-runs).
        var showStale = stale && output is not null && _pane.Phase != PanePhase.Unavailable;
        ResultHost.Opacity = showStale ? (running ? 0.55 : 0.3) : 1;
        StaleCard.Visibility = showStale && !running ? Visibility.Visible : Visibility.Collapsed;
        StaleReason.Text = _session.Workspace.ModelErrors.Count > 0 ? "The design has errors; fix them to update." : "This result is from an earlier design.";

        // Preview: the result is of a candidate, not of the design.
        var previewing = _session.Workspace.PreviewTitle;
        PreviewStrip.Visibility = previewing is null ? Visibility.Collapsed : Visibility.Visible;
        PreviewText.Text = previewing is null ? "" : $"Previewing {previewing} · not applied";

        // Settings form.
        SettingsHost.Visibility = _pane.ShowsSettings ? Visibility.Visible : Visibility.Collapsed;
        if (_pane.ShowsSettings && SettingsContent.Content is null) BuildSettings();
    }

    string StateLine(AnalysisDescriptor? d, bool running)
    {
        if (running) return _pane.Progress.Stage is { Length: > 0 } s ? s + " · " + (_pane.Progress.Fraction * 100).ToString("0", CultureInfo.InvariantCulture) + "%" : "Computing";
        if (_pane.Phase == PanePhase.Failed && _pane.Output is not null) return "Last run failed: " + _pane.Message;
        if (d is not null)
        {
            var s = SettingsSummary.Describe(d.ToJson(), _pane.Settings, _data);
            if (!string.IsNullOrEmpty(s)) return s;
        }
        return _pane.AutoUpdate ? "Default settings" : "Default settings · manual";
    }

    void ShowMessage(string glyph, string text, string? action)
    {
        MessagePanel.Visibility = Visibility.Visible;
        MessageGlyph.Glyph = glyph;
        MessageText.Text = text;
        MessageAction.Visibility = action is null ? Visibility.Collapsed : Visibility.Visible;
        MessageAction.Content = action;
    }

    void UpdateBackendBadge(AnalysisOutput output)
    {
        var gpu = output.Backend?.Used.StartsWith("gpu", StringComparison.Ordinal) == true;
        BackendText.Text = $"{(gpu ? "GPU" : "CPU")} · {RunDetailsView.Precision(output)}  {RunDetailsView.Time(output)}";
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

    void OnRun(object sender, RoutedEventArgs e)
    {
        if (_pane.IsRunning) _pane.Cancel();
        else _pane.Rerun(fresh: _pane.Phase == PanePhase.Failed);
    }

    void OnAutoUpdate(object sender, RoutedEventArgs e)
    {
        _pane.AutoUpdate = AutoButton.IsChecked == true;
        _pane.RaiseChanged();
        if (_pane.AutoUpdate && _pane.IsStale) _pane.Rerun();
        Update();
    }

    void OnSettingsToggle(object sender, RoutedEventArgs e)
    {
        _pane.ShowsSettings = SettingsButton.IsChecked == true;
        _pane.RaiseChanged();
        if (!_pane.ShowsSettings) SettingsContent.Content = null;
        Update();
    }

    void OnSettingsDone(object sender, RoutedEventArgs e)
    {
        _pane.ShowsSettings = false;
        _pane.RaiseChanged();
        SettingsContent.Content = null;
        Update();
    }

    void OnDefaults(object sender, RoutedEventArgs e)
    {
        _pane.SetSettings(_session.Workspace.StartingSettings(_pane.Analysis));
        SettingsContent.Content = null;
        Update();
    }

    void OnCopyJson(object sender, RoutedEventArgs e)
    {
        if (_pane.Output?.Json is not { } json) return;
        var p = new DataPackage();
        p.SetText(ModelJson.ToJsonString(json, pretty: true));
        Clipboard.SetContent(p);
        _session.ShowToast("Copied the result as JSON.");
    }

    void OnClose(object sender, RoutedEventArgs e) => CloseRequested?.Invoke(this);
}
