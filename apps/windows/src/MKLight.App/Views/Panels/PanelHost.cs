using System.Runtime.CompilerServices;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.App.Views.Analysis;
using MKLight.App.Views.Layout;
using MKLight.App.Windows;

namespace MKLight.App.Views.Panels;

/// <summary>
/// The design window's content area and the design's separate panel
/// windows (the Mac app's PanelViews, PanelDrag and DetachedWindows): the
/// lens data, the layout and the analysis panes are panels in a
/// <see cref="PanelArrangement"/>, a tree of splits and tab groups per
/// window. Panels move by dragging their title or tab (drop zones like an
/// IDE; outside the windows a separate window opens) or from their context
/// menu; the arrangement is saved in the project's <c>views</c>.
/// </summary>
public sealed class PanelHost : UserControl
{
    readonly DesignSession _session;
    readonly List<AnalysisPaneView> _paneViews = [];
    readonly PanelTreeView _main;
    readonly Dictionary<Guid, PanelWindow> _windows = [];
    /// <summary>The group view each shown panel is in.</summary>
    readonly Dictionary<PanelId, PanelGroupView> _hostOf = [];
    DesignWindow? _window;
    bool _placedOnce, _reconcilePending, _tearingDown, _ready;
    string? _lastFocused;
    DispatcherTimer? _saveTimer;

    public PanelHost(DesignSession session)
    {
        _session = session;
        Hosts.AddOrUpdate(session, this);
        LensData = new LensDataPanel { Host = session };
        Layout = new LayoutPanel(session);
        Arrangement = PanelArrangement.Standard([]);
        Drag = new PanelDragController(this);
        _main = new PanelTreeView(this, Arrangement.Main);
        Content = _main;
        PanelPersistence.Register(session, this);
        Track(LensData);
        Track(Layout);
        PanelHandles.AttachToPanel(this, LensData, PanelId.LensData);
        PanelHandles.AttachToPanel(this, Layout, PanelId.Layout);
        Arrangement.Changed += Refresh;
        session.Workspace.PanesChanged += OnPanesChanged;
        session.Workspace.StateChanged += FollowFocus;
        OnPanesChanged();
        Refresh();
    }

    static readonly ConditionalWeakTable<DesignSession, PanelHost> Hosts = new();

    /// <summary>
    /// The panel host of a design's session: a panel view reaches the panel commands (Move to New
    /// Window, Move to Main Window, Hide) through it, e.g. for its header's More menu.
    /// </summary>
    public static PanelHost? Of(DesignSession session) => Hosts.TryGetValue(session, out var h) ? h : null;

    /// <summary>
    /// Appends a panel's own commands to a menu, as its title's context menu has them: Move to Main
    /// Window (in a separate window), Move to New Window (unless alone in its window), then Hide
    /// (the lens data, the layout) or Close Pane (an analysis). Call it from the menu's Opening.
    /// </summary>
    public void AddPanelMenuItems(IList<MenuFlyoutItemBase> items, PanelId panel) => PanelMenus.AddPanelItems(items, this, panel);

    /// <summary>Whether the panel is in a separate window.</summary>
    public bool IsDetached(PanelId panel) => Arrangement.IsDetached(panel);

    public DesignSession Session => _session;
    public LensDataPanel LensData { get; }
    public LayoutPanel Layout { get; }
    public IReadOnlyList<AnalysisPaneView> PaneViews => _paneViews;
    /// <summary>Where the panels are, in the main window and the separate windows.</summary>
    public PanelArrangement Arrangement { get; }
    internal PanelDragController Drag { get; }
    /// <summary>The design window this host is the content of.</summary>
    public DesignWindow? Window => _window;
    /// <summary>The panel the panel commands act on (the last one clicked).</summary>
    public PanelId? FocusedPanel { get; private set; }

    /// <summary>Raised when the user focuses a pane (the sidebar follows).</summary>
    public event Action? FocusChanged;

    // ---- Views of panels ----

    internal FrameworkElement? ViewOf(PanelId panel) =>
        panel == PanelId.LensData ? LensData : panel == PanelId.Layout ? Layout : _paneViews.FirstOrDefault(v => v.Pane.Id == panel.Key);

    readonly HashSet<FrameworkElement> _live = [];

    /// <summary>Follows whether a panel view is loaded (its Unloaded comes later than its removal).</summary>
    void Track(FrameworkElement view)
    {
        view.Loaded += (_, _) => _live.Add(view);
        view.Unloaded += (_, _) => _live.Remove(view);
    }

    /// <summary>Whether the view is loaded (or was, and has not had its Unloaded yet).</summary>
    internal bool IsLive(FrameworkElement view) => _live.Contains(view);

    internal AnalysisPaneModel? Pane(PanelId panel) => panel.PaneId is { } id ? _session.Workspace.Panes.FirstOrDefault(p => p.Id == id) : null;

    internal string Title(PanelId panel) =>
        panel == PanelId.LensData ? "Lens Data" : panel == PanelId.Layout ? "Layout"
        : Pane(panel) is { } p ? AnalysisCatalog.Title(p.Analysis, _session.Workspace.Descriptor(p.Analysis)?.Title) : "Analysis";

    internal string Glyph(PanelId panel) =>
        panel == PanelId.LensData ? AnalysisCatalog.LensDataGlyph : panel == PanelId.Layout ? AnalysisCatalog.LayoutGlyph
        : Pane(panel) is { } p ? AnalysisCatalog.Glyph(p.Analysis) : "\uE9D2";

    // ---- Windows ----

    /// <summary>The main window is built: separate windows saved with the project open beside it and close with it.</summary>
    internal void Attach(DesignWindow window)
    {
        _window = window;
        _main.HostWindow = window;
        window.Closed += (_, _) => TearDown();
        _saveTimer = new DispatcherTimer { Interval = TimeSpan.FromSeconds(1) };
        // Pane toggles (automatic update, settings shown) raise no event: saved within a second.
        _saveTimer.Tick += (_, _) => Save();
        _saveTimer.Start();
        SyncWindows();
    }

    /// <summary>The panel trees on screen: the main window's, then the separate windows'.</summary>
    internal IEnumerable<PanelTreeView> Trees => _windows.Values.Select(w => w.Tree).Prepend(_main);

    internal PanelTreeView? TreeOf(XamlRoot? root) => Trees.FirstOrDefault(t => t.XamlRoot == root && root is not null);

    internal Microsoft.UI.Xaml.Window? WindowOf(PanelWindowModel model) =>
        model == Arrangement.Main ? _window : _windows.GetValueOrDefault(model.Id);

    /// <summary>Opens a window for each separate panel window and closes the ones whose panels have left.</summary>
    void SyncWindows()
    {
        var live = Arrangement.Detached.Select(w => w.Id).ToHashSet();
        foreach (var (id, w) in _windows.ToList())
        {
            if (live.Contains(id)) continue;
            _windows.Remove(id);
            w.CloseQuietly();
        }
        if (_window is null || _tearingDown) return;
        foreach (var model in Arrangement.Detached.Where(m => !_windows.ContainsKey(m.Id)))
        {
            var w = new PanelWindow(this, model);
            _windows[model.Id] = w;
            w.Tree.Refresh();
            w.Activate();
        }
        foreach (var w in _windows.Values) w.UpdateTitle();
    }

    /// <summary>The main window closes: the separate windows close with it (they stay in the saved arrangement).</summary>
    void TearDown()
    {
        if (_tearingDown) return;
        Save();
        _tearingDown = true;
        _saveTimer?.Stop();
        Drag.Cancel();
        foreach (var w in _windows.Values.ToList()) w.CloseQuietly();
        _windows.Clear();
    }

    /// <summary>The user closed a separate window: its panels close with it.</summary>
    internal void WindowClosed(PanelWindowModel model)
    {
        if (_tearingDown || !_windows.Remove(model.Id)) return;
        var panels = Arrangement.Discard(model);
        foreach (var p in panels)
            if (Pane(p) is { } pane) _session.Workspace.RemovePane(pane);
        if (FocusedPanel is { } f && panels.Contains(f)) FocusedPanel = null;
    }

    internal void FrameChanged() => SaveSoon();

    // ---- Showing, moving, closing ----

    /// <summary>Show and focus a panel: <c>lens_data</c>, <c>layout</c> or an analysis pane's id (its tab comes to the front, its window forward).</summary>
    public void ShowPanel(string kind)
    {
        var panel = new PanelId(kind);
        var wasShown = Arrangement.Contains(panel);
        if (!wasShown && panel.PaneId is null) Arrangement.Show(panel);
        Reveal(panel);
        Focus(panel);
        if (panel == PanelId.LensData) LensData.Focus(FocusState.Programmatic);
        else if (panel == PanelId.Layout && wasShown) Layout.Fit();
    }

    /// <summary>Brings a panel's tab to the front and its window forward.</summary>
    void Reveal(PanelId panel)
    {
        Arrangement.Select(panel);
        if (Arrangement.Location(panel) is { } l && l.Window != Arrangement.Main) _windows.GetValueOrDefault(l.Window.Id)?.Activate();
    }

    internal void Focus(PanelId panel)
    {
        FocusedPanel = panel;
        if (panel.PaneId is { } id && _session.Workspace.FocusedPaneId != id)
        {
            _session.Workspace.FocusedPaneId = id;
            _lastFocused = id;
            UpdateFocus();
            FocusChanged?.Invoke();
        }
    }

    /// <summary>Moves a panel (a drop, or a menu command). The only panel of a separate window dropped outside moves that window.</summary>
    internal void Move(PanelId panel, PanelDropTarget target, global::Windows.Graphics.RectInt32? frame = null)
    {
        if (target is PanelDropTarget.NewWindow && Arrangement.Location(panel) is { } l && l.Window != Arrangement.Main && l.Window.Panels.SequenceEqual([panel]))
        {
            if (frame is { } f && _windows.GetValueOrDefault(l.Window.Id) is { } w) { w.AppWindow.Move(new(f.X, f.Y)); w.Activate(); }
            return;
        }
        if (!Arrangement.CanMove(panel, target)) return;
        Arrangement.Move(panel, target, frame is { } px ? PanelWindow.ToFrame(px, _window) : null);
        Arrangement.Select(panel);
        Focus(panel);
        SyncWindows();
        if (Arrangement.Location(panel) is { } now) WindowOf(now.Window)?.Activate();
    }

    public void MoveToNewWindow(PanelId panel) => Move(panel, new PanelDropTarget.NewWindow(), Drag.DefaultFrame(panel));

    public void MoveToMainWindow(PanelId panel)
    {
        if (!Arrangement.IsDetached(panel)) return;
        Arrangement.ReturnToMain(panel);
        Focus(panel);
        SyncWindows();
        _window?.Activate();
    }

    /// <summary>Moves a separate window's panels back into the main window.</summary>
    public void Dock(PanelWindowModel model)
    {
        Arrangement.Dock(model);
        SyncWindows();
        _window?.Activate();
    }

    /// <summary>Closes a panel: an analysis pane is removed, the lens data and the layout are hidden until shown again.</summary>
    public void Close(PanelId panel)
    {
        if (Pane(panel) is { } pane) _session.Workspace.RemovePane(pane);
        Arrangement.Remove(panel);
        if (FocusedPanel == panel) FocusedPanel = null;
        SyncWindows();
    }

    /// <summary>Every panel back in the main window, in the standard arrangement.</summary>
    public void ResetLayout()
    {
        Arrangement.Reset(PanelArrangement.Standard([.. _session.Workspace.Panes.Select(p => p.Id)]));
        SyncWindows();
        _window?.Activate();
    }

    /// <summary>Adds an analysis pane to a given window (a separate window's Add Analysis menu).</summary>
    internal void AddPane(string analysis, Guid windowId)
    {
        var ws = _session.Workspace;
        var pane = ws.AddPane(analysis);
        Reconcile();
        // Reconcile placed it beside its neighbour; a separate window's pane goes into that window.
        if (Arrangement.Window(windowId) is { } w && w != Arrangement.Main)
        {
            Arrangement.Remove(PanelId.Analysis(pane.Id));
            Arrangement.Place(PanelId.Analysis(pane.Id), windowId);
            SyncWindows();
        }
    }

    // ---- Panes and the arrangement ----

    void OnPanesChanged()
    {
        var ws = _session.Workspace;
        _paneViews.RemoveAll(v => !ws.Panes.Contains(v.Pane));
        foreach (var pane in ws.Panes.Where(p => _paneViews.All(v => v.Pane != p)))
        {
            var view = new AnalysisPaneView(_session, pane);
            view.CloseRequested += v => Close(PanelId.Analysis(v.Pane.Id));
            view.Focused += v => Focus(PanelId.Analysis(v.Pane.Id));
            PanelHandles.AttachToPanel(this, view, PanelId.Analysis(pane.Id));
            Track(view);
            _paneViews.Add(view);
        }
        if (ws.Panes.Count > 0) _ready = true;
        // Panes added in one go (a window's first panes) are placed together.
        if (!_reconcilePending)
        {
            _reconcilePending = true;
            DispatcherQueue.TryEnqueue(Reconcile);
        }
        foreach (var t in Trees) t.UpdateTitles();
        foreach (var w in _windows.Values) w.UpdateTitle();
    }

    /// <summary>Brings the arrangement in line with the workspace's panes: closed panes leave it; new ones go beside the pane before them.</summary>
    internal void Reconcile()
    {
        _reconcilePending = false;
        var ws = _session.Workspace;
        var ids = ws.Panes.Select(p => p.Id).ToHashSet();
        var removed = Arrangement.AllPanels.Where(p => p.PaneId is { } id && !ids.Contains(id)).ToList();
        foreach (var p in removed) Arrangement.Remove(p);
        var missing = ws.Panes.Where(p => !Arrangement.Contains(PanelId.Analysis(p.Id))).ToList();
        if (missing.Count > 0 && !_placedOnce && !Arrangement.AllPanels.Any(p => p.PaneId is not null) && Arrangement.Detached.Count == 0)
        {
            // A window's first panes: the standard arrangement.
            Arrangement.Reset(PanelArrangement.Standard([.. ws.Panes.Select(p => p.Id)],
                Arrangement.Contains(PanelId.LensData), Arrangement.Contains(PanelId.Layout)));
        }
        else
        {
            foreach (var pane in missing)
            {
                var i = ws.Panes.IndexOf(pane);
                var before = ws.Panes.Take(i).Select(p => PanelId.Analysis(p.Id)).LastOrDefault(Arrangement.Contains);
                Arrangement.Place(PanelId.Analysis(pane.Id), before.Key is null ? null : before);
            }
        }
        if (missing.Count > 0) _placedOnce = true;
        if (missing.Count > 0 || removed.Count > 0) SyncWindows();
        FollowFocus();
        UpdateFocus();
    }

    /// <summary>The panes and the arrangement read back from the project (the session's start).</summary>
    internal void ApplySaved(WorkspaceLayout layout)
    {
        _ready = true;
        _placedOnce = true;
        _reconcilePending = false;
        OnPanesChanged();
        _reconcilePending = false;
        Arrangement.Replace(layout.MakeArrangement());
        // The focus goes to a pane in front, so the saved tabs stay as they were.
        var front = Arrangement.Windows.SelectMany(w => w.Groups).Select(g => g.Current?.PaneId).FirstOrDefault(id => id is not null);
        if (front is not null) _session.Workspace.FocusedPaneId = front;
        _lastFocused = _session.Workspace.FocusedPaneId;
        Reconcile();
        SyncWindows();
    }

    /// <summary>The focused pane (the sidebar opened an analysis) comes to the front of its tab group.</summary>
    void FollowFocus()
    {
        var id = _session.Workspace.FocusedPaneId;
        if (id == _lastFocused) return;
        _lastFocused = id;
        if (id is not null && Arrangement.Contains(PanelId.Analysis(id))) Reveal(PanelId.Analysis(id));
        UpdateFocus();
    }

    public void UpdateFocus()
    {
        var ws = _session.Workspace;
        foreach (var v in _paneViews) v.SetFocused(ws.Panes.Count > 1 && v.Pane.Id == ws.FocusedPane?.Id);
    }

    /// <summary>
    /// Every tree follows the arrangement. Views that move are taken out of their old place first (a view
    /// has one parent); the others stay where they are.
    /// </summary>
    internal void Refresh()
    {
        var rebuild = Trees.Where(t => t.NeedsRebuild).ToHashSet();
        foreach (var (panel, gv) in _hostOf.ToList())
        {
            var keep = !rebuild.Contains(gv.Tree) && Arrangement.Location(panel) is { } l && l.Group == gv.Group && gv.Group.Current == panel;
            if (keep) continue;
            gv.Release(panel);
            _hostOf.Remove(panel);
        }
        foreach (var t in Trees) t.Refresh();
        SaveSoon();
    }

    internal void Hosted(PanelId panel, PanelGroupView gv) => _hostOf[panel] = gv;

    /// <summary>A panel view's place on screen right now (for a separate window's starting frame).</summary>
    internal PanelGroupView? GroupViewOf(PanelId panel) => _hostOf.GetValueOrDefault(panel);

    // ---- Saving ----

    /// <summary>Writes the views now (nothing before they were read back).</summary>
    internal void SaveNow() => Save();

    internal void SaveSoon()
    {
        if (_saveTimer is null) return;
        _saveTimer.Stop();
        _saveTimer.Interval = TimeSpan.FromMilliseconds(250);
        _saveTimer.Start();
    }

    void Save()
    {
        if (_saveTimer is { } t) t.Interval = TimeSpan.FromSeconds(1);
        // Nothing is written before the saved views were read back (or the first panes placed).
        if (!_ready || _tearingDown) return;
        foreach (var w in _windows.Values) w.RecordFrame();
        PanelPersistence.Write(_session, Arrangement);
    }

    /// <summary>The views were read back (the session's start): saving may begin.</summary>
    internal void MarkReady() => _ready = true;
}
