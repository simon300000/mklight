using System.Globalization;
using System.Numerics;
using System.Text.Json.Nodes;
using Microsoft.Graphics.Canvas.UI.Xaml;
using Microsoft.UI.Input;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.App.Views.Viewport3D;
using MKLight.Engine;
using Windows.System;
using Windows.UI;

namespace MKLight.App.Views.Layout;

/// <summary>
/// The layout pane (the Mac app's LayoutPaneView): the engine's layout scene edge to edge, as the
/// Y–Z or X–Z section (Win2D) or in 3-D (Direct3D 11), with glass controls floating over it: the
/// view selector, a Display menu (ray colours, sampling, fit, vertical stretch, surface mesh, beam
/// footprints), a footprints toggle in 3-D and fit; a scale bar (with the stretch), the ray colour
/// legend, the stale card and the preview label. One camera serves every view: drag pans (orbits
/// in 3-D; right or middle drag, or Shift, pans), the wheel zooms about the pointer, double-click
/// fits, a click selects a surface. The selected surface is drawn in the accent.
/// </summary>
public sealed partial class LayoutPanel : UserControl
{
    DesignSession? _session;
    LayoutScene? _scene;
    SectionScene? _section;
    bool _stale;
    int? _selected;
    RayColoring _coloring = RayColoring.Field;
    ViewportMode _mode = ViewportMode.SectionYZ;
    ViewportCamera _camera = new();
    bool _anamorphic;
    // A fit waits for the scene of the shown view (after a mode change, the engine redraws the section).
    bool _fitPending;
    bool _fitLens;
    // The fit after a mode change frames the lens of a slim scene, as a first fit does.
    bool _modeFit;

    // Beam footprints (3-D): the footprint analysis on the layout's snapshot.
    bool _showFootprints;
    FootprintOverlay? _footprints;
    AnalysisJob? _footprintJob;
    SnapshotHash? _footprintHash;
    TaskCompletionSource? _footprintsReady;

    // Pointer gestures.
    Vector2? _dragFrom, _pressAt;
    bool _dragPans;

    // Fit leaves the bands under the floating controls and the scale bar clear.
    const double TopInset = 48, BottomInset = 34;

    public LayoutPanel()
    {
        InitializeComponent();
        ActualThemeChanged += (_, _) => { UpdateStyle(); Canvas.Invalidate(); };
        SizeChanged += (_, e) =>
        {
            if (!_camera.IsFitted || e.PreviousSize.Width == 0) Fit(initial: !_camera.IsFitted);
            else Redraw();
        };
        Unloaded += (_, _) => { Canvas.RemoveFromVisualTree(); _footprintJob?.Cancel(); };
        View3D.ErrorChanged += () => Message.Text = View3D.Error ?? (_scene is null ? Message.Text : "");
        StaleCard.Tapped += (_, _) => _session?.Workspace.Layout.Rerun();
        BuildModeSegments();
        UpdateStyle();
        Controls.SizeChanged += (_, _) => UpdateTitleChip();
        SizeChanged += (_, _) => UpdateTitleChip();
    }

    double _titleWidth = 110;

    /// <summary>The title chip steps aside when the controls need its room.</summary>
    void UpdateTitleChip()
    {
        if (TitleChip.Visibility == Visibility.Visible && TitleChip.ActualWidth > 0) _titleWidth = TitleChip.ActualWidth;
        var fits = ActualWidth - 20 - 8 >= Controls.ActualWidth + _titleWidth;
        TitleChip.Visibility = fits ? Visibility.Visible : Visibility.Collapsed;
    }

    /// <summary>The layout of a design window: follows the session's layout scene, selection and preview.</summary>
    public LayoutPanel(DesignSession session) : this()
    {
        _session = session;
        var layout = session.Workspace.Layout;
        _mode = layout.Settings["section"].StringOrNull() == "xz" ? ViewportMode.SectionXZ : ViewportMode.SectionYZ;
        void Update()
        {
            SetScene(session.LayoutScene, layout.Phase == PanePhase.Failed ? layout.Message : session.Workspace.ModelErrors.FirstOrDefault());
            SetStale(layout.IsStale, layout.IsRunning);
            var previewing = session.Workspace.PreviewTitle;
            PreviewChip.Visibility = previewing is null ? Visibility.Collapsed : Visibility.Visible;
            PreviewText.Text = previewing is null ? "" : $"Previewing {previewing} · not applied";
            AutomationProperties.SetName(PreviewChip, previewing is null ? "" : $"Previewing {previewing}. The design has not changed.");
            UpdateAnamorphic();
        }
        SurfaceClicked += i => session.SelectedSurface = i;
        session.SelectionChanged += () => SetSelectedSurface(session.SelectedSurface);
        session.LayoutSceneChanged += () => { Update(); RefreshFootprints(); };
        layout.Changed += Update;
        session.Workspace.PreviewChanged += Update;
        session.Workspace.Summary.Changed += UpdateAnamorphic;
        SetSelectedSurface(session.SelectedSurface);
        Update();
    }

    /// <summary>Raised when the user clicks a surface in the layout.</summary>
    public event Action<int>? SurfaceClicked;

    public RayColoring Coloring
    {
        get => _coloring;
        set { _coloring = value; UpdateLegend(); UpdateStyle(); }
    }

    /// <summary>The shown view; the layout's <c>section</c> setting follows it.</summary>
    public ViewportMode Mode
    {
        get => _mode;
        set
        {
            if (_mode == value) return;
            _mode = value;
            if (_session?.Workspace.Layout is { } layout)
            {
                var wanted = LayoutModes.SectionSetting(value);
                if (layout.Settings["section"].StringOrNull() != wanted) SetLayoutSetting("section", wanted);
            }
            _fitPending = true;
            BuildModeSegments();
            ShowView();
            TryPendingFit();
            if (_showFootprints) RefreshFootprints();
        }
    }

    /// <summary>The camera of every view (zoom, centre, orbit angles, stretch).</summary>
    public ViewportCamera Camera
    {
        get => _camera;
        set { _camera = value; Redraw(); }
    }

    /// <summary>The section views' vertical stretch (1 = true proportions); setting it refits.</summary>
    public double Stretch
    {
        get => _camera.Stretch;
        set
        {
            _camera.Stretch = Math.Clamp(value, 1, 50);
            _camera.StretchChosen = true;
            Fit();
        }
    }

    public bool ShowFootprints
    {
        get => _showFootprints;
        set
        {
            if (_showFootprints == value) return;
            _showFootprints = value;
            FootprintsButton.IsChecked = value;
            if (value) RefreshFootprints();
            else
            {
                _footprintJob?.Cancel();
                _footprintJob = null;
                _footprints = null;
                _footprintHash = null;
                UpdateStyle();
            }
            UpdateFootprintsTip();
        }
    }

    /// <summary>The surface mesh in 3-D (the layout's <c>mesh_rings</c>).</summary>
    public bool ShowMesh
    {
        get => (_session?.Workspace.Layout.Settings["mesh_rings"].IntOrNull() ?? 0) > 0;
        set => SetLayoutSetting("mesh_rings", value ? JsonValue.Create(16) : null);
    }

    /// <summary>Completes when the footprints of the current design are drawn (or failed).</summary>
    public Task FootprintsReady => _footprintsReady?.Task ?? Task.CompletedTask;

    public void SetScene(LayoutScene? scene, string? message = null)
    {
        _scene = scene;
        _section = scene is null ? null : EngineLayout.Section(scene);
        Message.Text = scene is null ? message ?? "" : View3D.Error ?? "";
        if (!_camera.IsFitted) Fit(initial: true);
        else TryPendingFit();
        UpdateLegend();
        UpdateStyle();
        AutomationProperties.SetName(Surface, LayoutModes.AccessibilityText(scene, _mode, _session?.Document.System.Surfaces.Count ?? 0));
    }

    /// <summary>Kept for callers that draw a section alone (no 3-D).</summary>
    public void SetScene(SectionScene? scene, string? message = null)
    {
        _section = scene;
        Message.Text = scene is null ? message ?? "" : "";
        if (!_camera.IsFitted) Fit(initial: true);
        UpdateLegend();
        Canvas.Invalidate();
    }

    public void SetStale(bool stale, bool updating)
    {
        _stale = stale;
        StaleCard.Visibility = stale ? Visibility.Visible : Visibility.Collapsed;
        StaleRing.IsActive = updating;
        StaleText.Text = updating ? "Updating" : "Design changed";
        ToolTipService.SetToolTip(StaleCard, updating ? null : "The layout is out of date. Click to update it.");
        TitleRing.IsActive = updating;
        TitleRing.Visibility = updating ? Visibility.Visible : Visibility.Collapsed;
        UpdateStyle();
    }

    public void SetSelectedSurface(int? index) { _selected = index; UpdateStyle(); }

    void UpdateAnamorphic()
    {
        var a = _session is { } s && LayoutModes.IsAnamorphic(s.Workspace.Summary.Output?.Json, s.Document.System);
        if (a == _anamorphic) return;
        _anamorphic = a;
        if (!a && _mode == ViewportMode.SectionXZ) Mode = ViewportMode.SectionYZ;
        BuildModeSegments();
    }

    void UpdateLegend()
    {
        if (_scene is null && _section is null) { ColoringLegend.Text = ""; return; }
        var fields = _scene?.FieldCount ?? _section!.FieldCount;
        var wavelengths = _scene?.WavelengthsUm ?? _section!.WavelengthsUm;
        ColoringLegend.Text = _coloring == RayColoring.Field
            ? $"Rays by field · {fields}"
            : "Rays by wavelength · " + string.Join("  ", wavelengths.Select(w => w.ToString("0.####", CultureInfo.InvariantCulture)));
    }

    // ---- Views ----

    void ShowView()
    {
        var orbit = _mode == ViewportMode.Orbit3D;
        Canvas.Visibility = orbit ? Visibility.Collapsed : Visibility.Visible;
        View3D.Visibility = orbit ? Visibility.Visible : Visibility.Collapsed;
        if (!orbit) Frame3D.Visibility = Visibility.Collapsed;
        FootprintsButton.Visibility = orbit ? Visibility.Visible : Visibility.Collapsed;
        UpdateStretchChip();
        UpdateStyle();
    }

    /// <summary>The 3-D view's drawing style from the window's theme and the panel's state.</summary>
    void UpdateStyle()
    {
        var dark = ActualTheme == ElementTheme.Dark;
        var accent = (Color)Application.Current.Resources[dark ? "SystemAccentColorLight2" : "SystemAccentColor"];
        static Vector3 Rgb(Color c) => new(c.R / 255f, c.G / 255f, c.B / 255f);
        var style = new ViewportStyle(
            ViewportMode.Orbit3D, dark,
            FieldRgb: i => Rgb(Palette.Field(i, dark)),
            WavelengthRgb: um => Rgb(Palette.Wavelength(um, dark)),
            DoeRgb: Rgb(Palette.Slot(6, dark)),
            Accent: Rgb(accent),
            Coloring: _coloring, SelectedSurface: _selected, Stale: _stale, Scale: View3D.PixelScale,
            Footprints: _showFootprints ? _footprints : null);
        if (ThemeLookup.Brush(this, "ContentSurfaceBrush") is SolidColorBrush bg)
            View3D.ClearColor = new Vortice.Mathematics.Color4(bg.Color.R / 255f, bg.Color.G / 255f, bg.Color.B / 255f, 1);
        View3D.SetScene(_scene, style);
        Canvas.Invalidate();
    }

    void Redraw()
    {
        if (_mode == ViewportMode.Orbit3D) { View3D.Mode = _mode; View3D.Camera = _camera; View3D.Invalidate(); }
        else Canvas.Invalidate();
        UpdateScaleBar();
    }

    // ---- Fit ----

    public void Fit() => Fit(initial: false);

    /// <summary>
    /// Fits the scene, or the lens alone (Fit Lens). The first fit of a slim, long scene (a slow
    /// lens far from its image) frames the lens, and a first fit of a section that would be a thin
    /// sliver turns on a vertical stretch (unless one was chosen).
    /// </summary>
    void Fit(bool initial)
    {
        var w = ActualWidth;
        var h = ActualHeight;
        if (w <= 1 || h <= 1) return;
        if (_scene is null)
        {
            if (_section is not null) FitSectionOnly(w, h);
            return;
        }
        _fitPending = false;
        var lensOnly = _fitLens || ((initial || _modeFit) && SceneGeometry.IsSlim(_scene, _mode));
        _modeFit = false;
        var points = SceneGeometry.FitPoints(_scene, _mode, lensOnly);
        var c = ViewportMath.Fit(_camera, _mode, points, w, h, TopInset, BottomInset);
        if (initial && _mode != ViewportMode.Orbit3D && c.Stretch == 1 && !c.StretchChosen)
        {
            var height = ViewportMath.Extent(c, _mode, points).Height * c.PointsPerMM;
            var s = ViewportCamera.AutomaticStretch(height, h);
            if (s != 1)
            {
                c.Stretch = s;
                c = ViewportMath.Fit(c, _mode, points, w, h, TopInset, BottomInset);
            }
        }
        _camera = c;
        UpdateStretchChip();
        Redraw();
    }

    /// <summary>A section without its 3-D scene (the parameterless panel): frame its bounds.</summary>
    void FitSectionOnly(double w, double h)
    {
        var (min, max) = _section!.Bounds();
        var pts = new List<V3> { new(0, min.Y, min.X), new(0, max.Y, max.X) };
        _camera = ViewportMath.Fit(_camera, ViewportMode.SectionYZ, pts, w, h, TopInset, BottomInset);
        Redraw();
    }

    void TryPendingFit()
    {
        if (_fitPending && _scene is not null && LayoutModes.SceneMatches(_mode, _scene.Section)) { _modeFit = true; Fit(initial: false); }
        else Redraw();
    }

    // ---- Section drawing (Win2D) ----

    /// <summary>A section point (z, up) on screen: the camera's view, stretched vertically.</summary>
    Vector2 ToScreen(Vector2 p)
    {
        var r = ViewportMath.Rotation(_camera, _mode);
        var c = r * _camera.Center; // the camera centre in view axes: (z, up, depth)
        var vs = _camera.VerticalScale(_mode);
        return new Vector2(
            (float)((p.X - c.X) * _camera.PointsPerMM + ActualWidth / 2),
            (float)(ActualHeight / 2 - (p.Y - c.Y) * _camera.PointsPerMM * vs));
    }

    void OnDraw(CanvasControl sender, CanvasDrawEventArgs args)
    {
        UpdateScaleBar();
        if (_section is null || _mode == ViewportMode.Orbit3D) return;
        SectionRenderer.Draw(args.DrawingSession, _section, ToScreen,
            new SectionRenderer.Style(ActualTheme == ElementTheme.Dark, _coloring, _selected, _stale));
    }

    /// <summary>A round length (1, 2, 5 × 10ⁿ mm) about 90 px long; with a stretch, an upright bar of a round length as drawn and "vertical ×N".</summary>
    void UpdateScaleBar()
    {
        var ppm = _camera.PointsPerMM;
        if ((_scene is null && _section is null) || !(ppm > 0)) { ScaleBar.Visibility = Visibility.Collapsed; return; }
        ScaleBar.Visibility = Visibility.Visible;
        var len = ViewportMath.NiceLength(90 / ppm);
        ScaleLine.Width = Math.Max(len * ppm, 1);
        ScaleText.Text = LayoutModes.LengthLabel(len);
        var vs = _camera.VerticalScale(_mode);
        var stretched = vs != 1;
        StretchLabel.Text = stretched ? $"   vertical ×{vs.ToString("0", CultureInfo.InvariantCulture)}" : "";
        UpBar.Visibility = stretched ? Visibility.Visible : Visibility.Collapsed;
        if (stretched)
        {
            var up = ViewportMath.NiceLength(60 / (ppm * vs));
            UpLine.Height = Math.Max(up * ppm * vs, 1);
            UpText.Text = LayoutModes.LengthLabel(up);
        }
        AutomationProperties.SetName(ScaleBar, $"Scale bar {LayoutModes.LengthLabel(len)}" + (stretched ? $", vertical stretch {vs:0} times" : ""));
    }

    // ---- Controls ----

    void BuildModeSegments()
    {
        var modes = LayoutModes.Available(_anamorphic, _mode);
        ModeSegments.Children.Clear();
        foreach (var m in modes)
        {
            var b = new Button
            {
                Content = LayoutModes.Title(m, modes.Length),
                Style = (Style)Resources[m == _mode ? "ModeSegmentSelected" : "ModeSegment"],
            };
            AutomationProperties.SetName(b, m switch { ViewportMode.Orbit3D => "3D view", ViewportMode.SectionXZ => "X–Z section", _ => modes.Length == 3 ? "Y–Z section" : "Section" });
            var mode = m;
            b.Click += (_, _) => Mode = mode;
            ModeSegments.Children.Add(b);
        }
        ToolTipService.SetToolTip(ModeSegments, modes.Length == 3 ? "Y–Z section, X–Z section or 3D view" : "Meridional section or 3D view");
    }

    void UpdateStretchChip()
    {
        var show = _mode != ViewportMode.Orbit3D && _camera.Stretch != 1;
        StretchChip.Visibility = show ? Visibility.Visible : Visibility.Collapsed;
        if (show)
        {
            var n = _camera.Stretch.ToString("0", CultureInfo.InvariantCulture);
            StretchText.Text = $"Stretch ×{n}";
            ToolTipService.SetToolTip(StretchButton, $"The section is drawn {n} times taller than true proportions. Click to change or turn off.");
        }
        UpdateScaleBar();
    }

    void UpdateFootprintsTip() => ToolTipService.SetToolTip(FootprintsButton,
        _showFootprints ? "Hide the beam footprints on the surfaces" : "Show the beam footprint of every field on every surface");

    void OnStretchMenuOpening(object? sender, object e) => FillStretchItems(StretchMenu.Items, enabled: true);

    void FillStretchItems(IList<MenuFlyoutItemBase> items, bool enabled)
    {
        items.Clear();
        foreach (var s in ViewportCamera.Stretches)
        {
            var item = new RadioMenuFlyoutItem
            {
                Text = s == 1 ? "Off (true proportions)" : $"×{s.ToString("0", CultureInfo.InvariantCulture)}",
                GroupName = "stretch", IsChecked = _camera.Stretch == s, IsEnabled = enabled,
            };
            var value = s;
            item.Click += (_, _) => Stretch = value;
            items.Add(item);
        }
    }

    /// <summary>The Display menu, built from the current state each time it opens.</summary>
    void OnDisplayMenuOpening(object? sender, object e)
    {
        var items = DisplayMenu.Items;
        items.Clear();
        RadioMenuFlyoutItem ColoringItem(string text, RayColoring c)
        {
            var item = new RadioMenuFlyoutItem { Text = text, GroupName = "coloring", IsChecked = _coloring == c };
            item.Click += (_, _) => Coloring = c;
            return item;
        }
        items.Add(ColoringItem("Rays by Field", RayColoring.Field));
        items.Add(ColoringItem("Rays by Wavelength", RayColoring.Wavelength));
        items.Add(new MenuFlyoutSeparator());

        var settings = _session?.Workspace.Layout.Settings;
        var rays = new MenuFlyoutSubItem { Text = "Rays per Field", IsEnabled = _session is not null };
        var current = settings?["rays_per_field"].IntOrNull();
        foreach (var n in new[] { 3, 5, 7, 11, 15 })
        {
            var item = new RadioMenuFlyoutItem { Text = n.ToString(CultureInfo.InvariantCulture), GroupName = "rays", IsChecked = current == n };
            item.Click += (_, _) => SetLayoutSetting("rays_per_field", JsonValue.Create(n));
            rays.Items.Add(item);
        }
        items.Add(rays);
        var wl = new MenuFlyoutSubItem { Text = "Wavelengths", IsEnabled = _session is not null };
        var type = settings?["wavelengths"].At("type").StringOrNull();
        foreach (var (text, value) in new[] { ("Primary", "primary"), ("All", "all") })
        {
            var item = new RadioMenuFlyoutItem { Text = text, GroupName = "wavelengths", IsChecked = type == value };
            item.Click += (_, _) => SetLayoutSetting("wavelengths", new JsonObject { ["type"] = value });
            wl.Items.Add(item);
        }
        items.Add(wl);
        items.Add(new MenuFlyoutSeparator());

        var fitAll = new MenuFlyoutItem { Text = "Fit All", KeyboardAcceleratorTextOverride = "Ctrl+0" };
        fitAll.Click += (_, _) => { _fitLens = false; Fit(); };
        var fitLens = new MenuFlyoutItem { Text = "Fit Lens" };
        fitLens.Click += (_, _) => { _fitLens = true; Fit(); _fitLens = false; };
        items.Add(fitAll);
        items.Add(fitLens);
        items.Add(new MenuFlyoutSeparator());

        var stretch = new MenuFlyoutSubItem { Text = "Vertical Stretch", IsEnabled = _mode != ViewportMode.Orbit3D };
        FillStretchItems(stretch.Items, enabled: _mode != ViewportMode.Orbit3D);
        items.Add(stretch);
        items.Add(new MenuFlyoutSeparator());

        var mesh = new ToggleMenuFlyoutItem { Text = "Surface Mesh in 3D", IsChecked = ShowMesh, IsEnabled = _session is not null };
        mesh.Click += (_, _) => ShowMesh = mesh.IsChecked;
        var footprints = new ToggleMenuFlyoutItem { Text = "Beam Footprints in 3D", IsChecked = _showFootprints, IsEnabled = _session is not null };
        footprints.Click += (_, _) => ShowFootprints = footprints.IsChecked;
        items.Add(mesh);
        items.Add(footprints);
    }

    void OnFootprintsClick(object sender, RoutedEventArgs e) => ShowFootprints = FootprintsButton.IsChecked == true;
    void OnFit(object sender, RoutedEventArgs e) { _fitLens = false; Fit(); }
    void OnDoubleTapped(object sender, DoubleTappedRoutedEventArgs e) => Fit();

    /// <summary>Changes one of the layout pane's settings (null removes it); the layout re-runs.</summary>
    void SetLayoutSetting(string key, JsonNode? value)
    {
        if (_session?.Workspace.Layout is not { } layout) return;
        var settings = (JsonObject)layout.Settings.DeepClone();
        if (value is null) settings.Remove(key); else settings[key] = value;
        layout.SetSettings(settings);
    }

    // ---- Footprints ----

    /// <summary>Runs the footprint analysis on the layout's snapshot (once per snapshot) while footprints are shown in 3-D.</summary>
    void RefreshFootprints()
    {
        if (!_showFootprints || _session is not { } session) return;
        var ws = session.Workspace;
        if (ws.Snapshot is not { } snapshot || ws.ModelErrors.Count > 0) return;
        if (_footprintHash == snapshot.Hash && (_footprints is not null || _footprintJob is not null)) return;
        _footprintJob?.Cancel();
        _footprintHash = snapshot.Hash;
        var job = ws.Engine.Submit(new EngineAnalysisRequest("footprint", FootprintOverlay.Settings()), snapshot);
        _footprintJob = job;
        var ready = _footprintsReady is { Task.IsCompleted: false } pending ? pending : new TaskCompletionSource();
        _footprintsReady = ready;
        _ = ObserveFootprints(job, ready);
    }

    async Task ObserveFootprints(AnalysisJob job, TaskCompletionSource ready)
    {
        FootprintOverlay? overlay = null;
        string? error = null;
        try
        {
            using var output = await job.ResultAsync();
            overlay = FootprintOverlay.FromOutput(output);
            if (overlay is null) error = "The footprint result has no beam outlines.";
        }
        catch (EngineException e) when (e.Kind == EngineErrorKind.Cancelled) { return; }
        catch (Exception e) { error = e.Message; }
        if (!ReferenceEquals(job, _footprintJob)) return;
        _footprintJob = null;
        _footprints = overlay;
        if (error is not null) _session?.ShowToast("Beam footprints: " + error);
        UpdateStyle();
        ready.TrySetResult();
    }

    // ---- Pointer input ----

    Vector2 Centred(Vector2 p) => new((float)(p.X - ActualWidth / 2), (float)(ActualHeight / 2 - p.Y));

    void OnPointerPressed(object sender, PointerRoutedEventArgs e)
    {
        var p = e.GetCurrentPoint(Surface);
        var props = p.Properties;
        if (!props.IsLeftButtonPressed && !props.IsMiddleButtonPressed && !props.IsRightButtonPressed) return;
        _dragFrom = p.Position.ToVector2();
        _pressAt = _dragFrom;
        Frame3D.Visibility = Visibility.Collapsed;
        var shift = (e.KeyModifiers & VirtualKeyModifiers.Shift) != 0;
        _dragPans = _mode != ViewportMode.Orbit3D || !props.IsLeftButtonPressed || shift;
        Surface.CapturePointer(e.Pointer);
        ProtectedCursor = InputSystemCursor.Create(_dragPans ? InputSystemCursorShape.SizeAll : InputSystemCursorShape.Cross);
        e.Handled = true;
    }

    void OnPointerMoved(object sender, PointerRoutedEventArgs e)
    {
        if (_dragFrom is not Vector2 from) return;
        var at = e.GetCurrentPoint(Surface).Position.ToVector2();
        var d = at - from;
        _dragFrom = at;
        _camera = _dragPans ? ViewportMath.Pan(_camera, _mode, d.X, -d.Y) : ViewportMath.Orbit(_camera, d.X, -d.Y);
        Redraw();
    }

    void OnPointerReleased(object sender, PointerRoutedEventArgs e)
    {
        var at = e.GetCurrentPoint(Surface).Position.ToVector2();
        if (_pressAt is Vector2 p0 && Vector2.Distance(p0, at) < 3) HitTest(at);
        EndDrag(e.Pointer);
    }

    void OnPointerCanceled(object sender, PointerRoutedEventArgs e) => EndDrag(e.Pointer);

    void EndDrag(Pointer pointer)
    {
        _dragFrom = null;
        _pressAt = null;
        Surface.ReleasePointerCapture(pointer);
        ProtectedCursor = null;
    }

    void OnPointerWheel(object sender, PointerRoutedEventArgs e)
    {
        var pt = e.GetCurrentPoint(Surface);
        Frame3D.Visibility = Visibility.Collapsed;
        var c = Centred(pt.Position.ToVector2());
        _camera = ViewportMath.Zoom(_camera, _mode, Math.Pow(1.0015, pt.Properties.MouseWheelDelta), c.X, c.Y);
        Redraw();
        e.Handled = true;
    }

    /// <summary>A click within 5 px of a surface (profile, rim, stop or image plane) selects it.</summary>
    void HitTest(Vector2 at)
    {
        int? best = null;
        var bestD = 5.0;
        if (_scene is { } scene)
        {
            var orbit = _mode == ViewportMode.Orbit3D;
            foreach (var l in scene.Polylines)
            {
                if (l.Surface is null || !(orbit || l.InSection)) continue;
                if (l.Kind is not (LayoutScene.PolylineKind.SurfaceProfile or LayoutScene.PolylineKind.Stop or LayoutScene.PolylineKind.ImagePlane
                    or LayoutScene.PolylineKind.SurfaceRim)) continue;
                if (!orbit && l.Kind == LayoutScene.PolylineKind.SurfaceRim) continue;
                var prev = ViewportMath.ToScreen(_camera, _mode, V3.Of(l.Points.Count > 0 ? l.Points[0] : default), ActualWidth, ActualHeight);
                for (var k = 1; k < l.Points.Count; k++)
                {
                    var cur = ViewportMath.ToScreen(_camera, _mode, V3.Of(l.Points[k]), ActualWidth, ActualHeight);
                    var d = DistanceToSegment(at, prev, cur);
                    prev = cur;
                    if (d < bestD) { bestD = d; best = l.Surface; }
                }
            }
        }
        if (best is int s) SurfaceClicked?.Invoke(s);
    }

    static double DistanceToSegment(Vector2 p, (double X, double Y) a, (double X, double Y) b)
    {
        double abx = b.X - a.X, aby = b.Y - a.Y, apx = p.X - a.X, apy = p.Y - a.Y;
        var len2 = abx * abx + aby * aby;
        var t = len2 == 0 ? 0 : Math.Clamp((apx * abx + apy * aby) / len2, 0, 1);
        double dx = apx - t * abx, dy = apy - t * aby;
        var d = Math.Sqrt(dx * dx + dy * dy);
        return double.IsFinite(d) ? d : double.PositiveInfinity;
    }

    // ---- Captures ----

    /// <summary>
    /// Shows a frame of the 3-D view rendered offscreen in an image over the swap chain, so a
    /// window capture (PrintWindow) includes it; any camera change removes it.
    /// </summary>
    public void FreezeFrameForCapture()
    {
        if (_mode != ViewportMode.Orbit3D) return;
        var scale = XamlRoot?.RasterizationScale ?? 1;
        if (View3D.RenderOffscreen(scale) is not var (pixels, w, h)) return;
        var bitmap = new Microsoft.UI.Xaml.Media.Imaging.WriteableBitmap(w, h);
        using (var stream = System.Runtime.InteropServices.WindowsRuntime.WindowsRuntimeBufferExtensions.AsStream(bitmap.PixelBuffer))
            stream.Write(pixels, 0, pixels.Length);
        bitmap.Invalidate();
        Frame3D.Source = bitmap;
        Frame3D.Visibility = Visibility.Visible;
    }
}
