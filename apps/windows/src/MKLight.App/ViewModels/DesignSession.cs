using System.Text.Json.Nodes;
using MKLight.App.Views;
using MKLight.App.Views.Layout;
using MKLight.Engine;

namespace MKLight.App.ViewModels;

/// <summary>
/// One open design (the Mac app's DesignSession): the document, its engine
/// workspace and the window state (selection, toasts). The document's
/// changes feed the workspace; the workspace's results feed the views.
/// </summary>
public sealed class DesignSession : ILensDataHost
{
    public DesignSession(DesignDocument document, IEngineClient engine)
    {
        Document = document;
        Workspace = new Workspace(engine.MakeDocumentEngine());
        Document.Changed += () => Workspace.Update(Document.System);
        Workspace.StateChanged += () => EngineDataChanged?.Invoke();
        Workspace.Layout.Changed += OnLayoutChanged;
    }

    public DesignDocument Document { get; }
    public Workspace Workspace { get; }

    public async Task StartAsync(IEnumerable<string>? analyses = null, JsonObject? extraSettings = null)
    {
        Workspace.Update(Document.System, immediately: true);
        await Workspace.LoadEngineMetadataAsync();
        foreach (var a in analyses ?? DefaultPanes)
        {
            var settings = Workspace.StartingSettings(a);
            if (extraSettings is not null)
                foreach (var kv in extraSettings) settings[kv.Key] = kv.Value?.DeepClone();
            Workspace.AddPane(a, settings);
        }
        Workspace.FocusedPaneId = Workspace.Panes.FirstOrDefault()?.Id;
    }

    /// <summary>The panes a new window opens with.</summary>
    public static readonly string[] DefaultPanes = ["spot_diagram", "mtf"];

    // ---- Selection ----

    int? _selected;
    public int? SelectedSurface
    {
        get => _selected;
        set { if (_selected == value) return; _selected = value; SelectionChanged?.Invoke(); }
    }
    public event Action? SelectionChanged;

    // ---- Engine data for the lens table ----

    public IReadOnlyDictionary<int, IReadOnlyList<IssueItem>> IssuesBySurface => Workspace.IssuesBySurface;
    public IReadOnlyDictionary<int, double> AutoSemiDiameters { get; private set; } = new Dictionary<int, double>();
    public IReadOnlyList<string> MaterialNames => Workspace.MaterialNames;
    public event Action? EngineDataChanged;

    /// <summary>The layout's section, read once per layout output.</summary>
    public SectionScene? Section { get; private set; }
    public string? LayoutMessage { get; private set; }
    AnalysisOutput? _layoutRead;

    void OnLayoutChanged()
    {
        var output = Workspace.Layout.Output;
        if (ReferenceEquals(output, _layoutRead)) return;
        _layoutRead = output;
        var scene = LayoutScene.FromOutput(output);
        Section = scene is null ? null : EngineLayout.Section(scene);
        AutoSemiDiameters = scene?.Surfaces.Where(s => double.IsFinite(s.SemiDiameter)).GroupBy(s => s.Index)
            .ToDictionary(g => g.Key, g => g.First().SemiDiameter) ?? new Dictionary<int, double>();
        EngineDataChanged?.Invoke();
    }

    // ---- Toasts ----

    public event Action<string>? Toast;
    public void ShowToast(string message) => Toast?.Invoke(message);
}
