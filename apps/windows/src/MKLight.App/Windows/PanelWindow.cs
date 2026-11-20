using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Markup;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.App.Views.Panels;
using Windows.System;
using RectInt32 = Windows.Graphics.RectInt32;

namespace MKLight.App.Windows;

/// <summary>
/// A window that panels were moved out to (the Mac app's detached panel
/// window): a view of the same design session, so selection, edits, analysis
/// updates and the preview of a candidate stay live. Its own panel tree
/// (tabs, splits, drag and drop), a title bar with Add Analysis and Move to
/// Main Window, the design's name as subtitle; Undo, Redo and Save act on
/// the design. Closing it closes its panels; it closes quietly with the main
/// window and opens again with the project.
/// </summary>
public sealed partial class PanelWindow : Window
{
    readonly PanelHost _host;
    readonly Grid _root;
    readonly TitleBar _bar;
    bool _quiet;

    internal PanelWindow(PanelHost host, PanelWindowModel model)
    {
        _host = host;
        Model = model;
        SystemBackdrop = new Microsoft.UI.Xaml.Media.MicaBackdrop();
        _root = (Grid)XamlReader.Load("""
            <Grid xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
                  xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml">
                <Grid.RowDefinitions>
                    <RowDefinition Height="Auto" />
                    <RowDefinition Height="*" />
                </Grid.RowDefinitions>
                <TitleBar x:Name="Bar" Height="48" IsBackButtonVisible="False" IsPaneToggleButtonVisible="False">
                    <TitleBar.IconSource>
                        <ImageIconSource ImageSource="ms-appx:///Assets/AppIcon.png" />
                    </TitleBar.IconSource>
                    <TitleBar.RightHeader>
                        <StackPanel Orientation="Horizontal" Spacing="2" Margin="0,0,4,0" VerticalAlignment="Center">
                            <DropDownButton x:Name="AddButton" Height="30" CornerRadius="15" Padding="9,0,7,0"
                                            Background="Transparent" BorderThickness="0"
                                            ToolTipService.ToolTip="Add an analysis pane to this window">
                                <FontIcon Glyph="&#xECCD;" FontSize="14" />
                            </DropDownButton>
                            <Button x:Name="DockButton" Style="{StaticResource CircleButton}" Content="&#xE73F;"
                                    ToolTipService.ToolTip="Move to Main Window: these panels go back into the design's window" />
                        </StackPanel>
                    </TitleBar.RightHeader>
                </TitleBar>
                <Border x:Name="Card" Grid.Row="1" Background="{ThemeResource ContentSurfaceBrush}"
                        BorderBrush="{ThemeResource ContentStrokeBrush}" BorderThickness="0,1,0,0" />
            </Grid>
            """);
        Content = _root;
        _bar = (TitleBar)_root.FindName("Bar");
        ExtendsContentIntoTitleBar = true;
        SetTitleBar(_bar);
        AppWindow.TitleBar.PreferredHeightOption = TitleBarHeightOption.Tall;
        AppWindow.SetIcon(Path.Combine(AppContext.BaseDirectory, "Assets", "MKLight.ico"));

        Tree = new PanelTreeView(host, model) { HostWindow = this };
        ((Border)_root.FindName("Card")).Child = Tree;
        var add = new MenuFlyout { Placement = Microsoft.UI.Xaml.Controls.Primitives.FlyoutPlacementMode.BottomEdgeAlignedRight };
        PanelMenus.FillAnalyses(add, host.Session.Workspace, a => host.AddPane(a, model.Id));
        ((DropDownButton)_root.FindName("AddButton")).Flyout = add;
        ((Button)_root.FindName("DockButton")).Click += (_, _) => host.Dock(model);

        ThemeSwitch.Attach(this, _root);
        BuildAccelerators();
        PlaceOnScreen();
        if (AppWindow.Presenter is OverlappedPresenter p)
        {
            var scale = Scale(host.Window);
            p.PreferredMinimumWidth = (int)(360 * scale);
            p.PreferredMinimumHeight = (int)(260 * scale);
        }
        host.Session.Document.Changed += UpdateTitle;
        _root.SizeChanged += (_, _) => UpdateTitle();
        AppWindow.Changed += (_, a) => { if (a.DidPositionChange || a.DidSizeChange) host.FrameChanged(); };
        Closed += (_, _) =>
        {
            host.Session.Document.Changed -= UpdateTitle;
            if (!_quiet) host.WindowClosed(Model);
        };
        UpdateTitle();
    }

    internal PanelWindowModel Model { get; }
    internal PanelTreeView Tree { get; }

    /// <summary>Closes the window without closing its panels (they moved, or the main window is closing).</summary>
    internal void CloseQuietly()
    {
        _quiet = true;
        Close();
    }


    /// <summary>The panels in front, e.g. "Spot Diagram, Ray Fan"; the design's name below.</summary>
    internal void UpdateTitle()
    {
        var shown = Model.Groups.Select(g => g.Current).OfType<PanelId>().Select(_host.Title).ToList();
        var title = shown.Count == 0 ? "Panels" : shown.Count > 3 ? string.Join(", ", shown.Take(3)) + $" +{shown.Count - 3}" : string.Join(", ", shown);
        var doc = _host.Session.Document;
        _bar.Title = title;
        // A narrow window keeps the title and the buttons clear of the caption buttons.
        _bar.Subtitle = _root.ActualWidth is > 0 and < 560 ? "" : string.IsNullOrEmpty(doc.System.Name) ? doc.DisplayName : doc.System.Name;
        Title = $"{title} — {doc.DisplayName}";
    }

    void BuildAccelerators()
    {
        void Add(VirtualKey key, VirtualKeyModifiers mods, Action action)
        {
            var a = new KeyboardAccelerator { Key = key, Modifiers = mods };
            a.Invoked += (_, e) => { action(); e.Handled = true; };
            _root.KeyboardAccelerators.Add(a);
        }
        _root.KeyboardAcceleratorPlacementMode = KeyboardAcceleratorPlacementMode.Hidden;
        var doc = _host.Session.Document;
        // Edits made here are the design's: the document's undo, the design window's save.
        Add(VirtualKey.Z, VirtualKeyModifiers.Control, () => { if (!IsTextEditing()) doc.Undo(); });
        Add(VirtualKey.Y, VirtualKeyModifiers.Control, () => { if (!IsTextEditing()) doc.Redo(); });
        Add(VirtualKey.Z, VirtualKeyModifiers.Control | VirtualKeyModifiers.Shift, () => { if (!IsTextEditing()) doc.Redo(); });
        Add(VirtualKey.S, VirtualKeyModifiers.Control, () => _ = _host.Window?.SaveAsync(false));
        Add(VirtualKey.S, VirtualKeyModifiers.Control | VirtualKeyModifiers.Shift, () => _ = _host.Window?.SaveAsync(true));
        Add(VirtualKey.F5, VirtualKeyModifiers.None, () => _host.Session.Workspace.RerunAll());
        Add(VirtualKey.Escape, VirtualKeyModifiers.Control, () => _host.Session.Workspace.CancelAll());
    }

    bool IsTextEditing() => FocusManager.GetFocusedElement(_root.XamlRoot) is TextBox or AutoSuggestBox or NumberBox;

    // ---- Frame ----

    /// <summary>At the saved frame if it is still on a display, else over the main window.</summary>
    void PlaceOnScreen()
    {
        var scale = Scale(_host.Window);
        RectInt32? saved = Model.Frame is { } f ? new RectInt32((int)Math.Round(f.X * scale), (int)Math.Round(f.Y * scale), (int)Math.Round(f.Width * scale), (int)Math.Round(f.Height * scale)) : null;
        if (saved is not { } r || !IsOnADisplay(r))
        {
            var main = _host.Window?.AppWindow;
            var cx = main is null ? 800 : main.Position.X + main.Size.Width / 2;
            var cy = main is null ? 500 : main.Position.Y + main.Size.Height / 2;
            r = new RectInt32(cx - (int)(280 * scale), cy - (int)(220 * scale), (int)(560 * scale), (int)(440 * scale));
        }
        AppWindow.MoveAndResize(r);
        RecordFrame();
    }

    static bool IsOnADisplay(RectInt32 f)
    {
        // The projected list does not enumerate (CsWinRT): by index.
        var displays = DisplayArea.FindAll();
        for (var i = 0; i < displays.Count; i++)
        {
            var a = displays[i].WorkArea;
            var w = Math.Min(a.X + a.Width, f.X + f.Width) - Math.Max(a.X, f.X);
            var h = Math.Min(a.Y + a.Height, f.Y + f.Height) - Math.Max(a.Y, f.Y);
            if (w > 80 && h > 80) return true;
        }
        return false;
    }

    internal void RecordFrame()
    {
        if (!AppWindow.IsVisible && Model.Frame is not null) return;
        Model.Frame = ToFrame(new RectInt32(AppWindow.Position.X, AppWindow.Position.Y, AppWindow.Size.Width, AppWindow.Size.Height), _host.Window);
    }

    /// <summary>A frame in screen pixels as saved: effective pixels at the main window's scale.</summary>
    internal static PanelFrame ToFrame(RectInt32 px, Window? main)
    {
        var s = Scale(main);
        return new PanelFrame(px.X / s, px.Y / s, px.Width / s, px.Height / s);
    }

    static double Scale(Window? w) => w is null ? 1 : GetDpiForWindow(WinRT.Interop.WindowNative.GetWindowHandle(w)) / 96.0;

    [System.Runtime.InteropServices.LibraryImport("user32.dll")]
    private static partial uint GetDpiForWindow(nint hwnd);
}
