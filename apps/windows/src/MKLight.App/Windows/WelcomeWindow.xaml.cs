using System.Globalization;
using Microsoft.Graphics.Canvas.UI.Xaml;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.App.Views.Layout;
using MKLight.Engine;
using MKLight.Model;

namespace MKLight.App.Windows;

/// <summary>
/// The welcome window: New Design (the one prominent action), Open, recent
/// files, and the engine's sample designs as cards with layout thumbnails
/// computed by the engine on a session of their own. Samples with extended
/// surface types are listed under "Surface Types".
/// </summary>
public sealed partial class WelcomeWindow : Window
{
    public WelcomeWindow()
    {
        InitializeComponent();
        Title = "Welcome to MKLight";
        ExtendsContentIntoTitleBar = true;
        SetTitleBar(DragRegion);
        AppWindow.SetIcon(Path.Combine(AppContext.BaseDirectory, "Assets", "MKLight.ico"));
        Root.RequestedTheme = AppSettings.Theme;
        var dark = Root.ActualTheme == ElementTheme.Dark || (AppSettings.Theme == ElementTheme.Default && Application.Current.RequestedTheme == ApplicationTheme.Dark);
        AppWindow.TitleBar.ButtonForegroundColor = dark ? Microsoft.UI.Colors.White : Microsoft.UI.Colors.Black;
        EngineText.Text = App.Engine is { } e ? $"{e.Info.Name} {e.Info.Version}" : "Engine unavailable";
        BuildRecent();
        BuildGallery();
    }

    public void ShowError(string message)
    {
        RecentPanel.Children.Insert(0, new InfoBar { IsOpen = true, Severity = InfoBarSeverity.Error, Message = message, IsClosable = true, Margin = new Thickness(0, 0, 0, 12) });
    }

    void BuildRecent()
    {
        var recent = RecentFiles.List;
        if (recent.Count == 0) { RecentPanel.Visibility = Visibility.Collapsed; return; }
        foreach (var path in recent.Take(6))
        {
            var content = new StackPanel { Spacing = 0 };
            content.Children.Add(new TextBlock { Text = Path.GetFileNameWithoutExtension(path), Style = (Style)Application.Current.Resources["BodyText"] });
            content.Children.Add(new TextBlock { Text = Path.GetDirectoryName(path), Style = (Style)Application.Current.Resources["CaptionText"] });
            var b = new Button
            {
                Content = content, HorizontalAlignment = HorizontalAlignment.Stretch, HorizontalContentAlignment = HorizontalAlignment.Left,
                Background = new SolidColorBrush(Microsoft.UI.Colors.Transparent), BorderThickness = new Thickness(0), Padding = new Thickness(8, 5, 8, 6),
            };
            ToolTipService.SetToolTip(b, path);
            b.Click += (_, _) => _ = App.OpenFileAsync(path);
            RecentPanel.Children.Add(b);
        }
    }

    void BuildGallery()
    {
        var all = SampleSystems.Bundled.ToList();
        var extended = all.Where(s => !s.System.Surfaces.All(f => f.Shape.IsLegacy)).ToList();
        Section("Sample Designs", all.Except(extended).ToList());
        if (extended.Count > 0) Section("Surface Types", extended);
    }

    void Section(string title, IReadOnlyList<SampleEntry> samples)
    {
        Gallery.Children.Add(new TextBlock { Text = title, FontFamily = (FontFamily)Application.Current.Resources["UIDisplayFont"], FontSize = 20, FontWeight = Microsoft.UI.Text.FontWeights.SemiBold, Margin = new Thickness(2, Gallery.Children.Count == 0 ? 0 : 18, 0, 2) });
        var grid = new VariableSizedWrapGrid { Orientation = Orientation.Horizontal, ItemWidth = 214, ItemHeight = 214 };
        foreach (var s in samples) grid.Children.Add(Card(s));
        Gallery.Children.Add(grid);
    }

    UIElement Card(SampleEntry sample)
    {
        var thumb = new SampleThumbnail(sample);
        var thumbBorder = new Border
        {
            Height = 120, CornerRadius = new CornerRadius(10), BorderThickness = new Thickness(1),
            BorderBrush = ThemeLookup.Brush(Root, "CardStrokeColorDefaultBrush"),
            Background = ThemeLookup.Brush(Root, "ContentSurfaceBrush"), Child = thumb,
        };
        var text = new StackPanel { Spacing = 1, Margin = new Thickness(4, 8, 4, 0) };
        text.Children.Add(new TextBlock { Text = sample.System.Name, Style = (Style)Application.Current.Resources["PanelTitleText"], FontSize = 14 });
        text.Children.Add(new TextBlock { Text = Detail(sample.System), Style = (Style)Application.Current.Resources["CaptionText"], FontSize = 12, TextWrapping = TextWrapping.Wrap, MaxLines = 2, TextTrimming = TextTrimming.CharacterEllipsis });
        var content = new StackPanel { Children = { thumbBorder, text } };
        var button = new Button
        {
            Content = content, Width = 202, Height = 204, Padding = new Thickness(0), VerticalContentAlignment = VerticalAlignment.Top,
            HorizontalContentAlignment = HorizontalAlignment.Stretch, Background = new SolidColorBrush(Microsoft.UI.Colors.Transparent), BorderThickness = new Thickness(0),
        };
        ToolTipService.SetToolTip(button, $"New design from “{sample.System.Name}”");
        button.Click += (_, _) => App.OpenDocument(new DesignDocument(new ProjectFile(sample.System)));
        return button;
    }

    static string Detail(OpticalSystem s)
    {
        var n = s.Surfaces.Count - 2;
        var kinds = s.Surfaces.Select(f => f.Shape).Where(sh => !sh.IsLegacy).Select(sh => CultureInfo.InvariantCulture.TextInfo.ToTitleCase(sh.TypeName.Replace('_', ' '))).Distinct().Order().ToList();
        var notes = kinds.Count > 0 ? string.Join(", ", kinds) : s.Notes;
        return $"{n} surfaces" + (string.IsNullOrEmpty(notes) ? "" : " · " + notes);
    }

    void OnNew(object sender, RoutedEventArgs e) => App.OpenDocument(new DesignDocument(new ProjectFile(SampleSystems.Get("singlet") ?? SampleSystems.Singlet())));
    void OnOpen(object sender, RoutedEventArgs e) => _ = App.OpenWithPickerAsync(this);
}

/// <summary>A sample's layout drawn small: the engine's layout on a session of its own.</summary>
sealed class SampleThumbnail : Grid
{
    static IEngineClient? _engine;
    readonly CanvasControl _canvas = new();
    SectionScene? _scene;

    public SampleThumbnail(SampleEntry sample)
    {
        Children.Add(_canvas);
        _canvas.Draw += (s, a) =>
        {
            if (_scene is null) return;
            var (scale, center) = SectionRenderer.Fit(_scene, (float)s.ActualWidth, (float)s.ActualHeight, 12, 10, 10);
            SectionRenderer.Draw(a.DrawingSession, _scene, p => new((p.X - center.X) * scale + (float)s.ActualWidth / 2, -(p.Y - center.Y) * scale + (float)s.ActualHeight / 2),
                new SectionRenderer.Style(ActualTheme == ElementTheme.Dark, LineScale: 0.8f));
        };
        ActualThemeChanged += (_, _) => _canvas.Invalidate();
        Unloaded += (_, _) => _canvas.RemoveFromVisualTree();
        _ = LoadAsync(sample);
    }

    async Task LoadAsync(SampleEntry sample)
    {
        if (App.Engine is null) return;
        try
        {
            _engine ??= App.Engine.MakeDocumentEngine();
            var snap = await _engine.OpenSnapshotAsync(ModelJson.Encode(sample.System));
            var output = await _engine.Submit(new EngineAnalysisRequest("layout"), snap).ResultAsync();
            if (LayoutScene.FromOutput(output) is { } scene) _scene = EngineLayout.Section(scene);
            _canvas.Invalidate();
        }
        catch (Exception e) { App.Log($"thumbnail {sample.Id}: {e.Message}"); }
    }
}
