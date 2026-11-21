using System.Text.Json.Nodes;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.App.Views.Analysis.Results.Readers;
using MKLight.App.Views.Analysis.Settings;
using Windows.UI;

namespace MKLight.App.Views.Analysis.Results;

/// <summary>
/// The image simulation's pane controls: the source (a built-in pattern or an image file), the
/// simulated patch (its field of view, with the width and pixel size it means on the sensor, and
/// its centre: the axis, a design field point, or a point picked on a field diagram) and the
/// resolution. Each changes the pane's settings, which runs it again.
/// </summary>
public sealed partial class ImageSimulationView
{
    DropDownButton? _centerButton;
    Flyout? _picker;

    /// <summary>The axis and the design's field points (of the previewed candidate while there is one).</summary>
    IReadOnlyList<ImageSimulationSettings.FieldPoint> FieldPoints()
    {
        var ws = _pane?.Workspace;
        var system = ws?.PreviewSystem ?? ws?.DesignSystem;
        return ImageSimulationSettings.FieldPoints(system?.Fields.Fields.Select(f => (f.X, f.Y)));
    }

    void AddPaneControls(AnalysisPaneModel pane)
    {
        var s = pane.Settings;
        var i = 0;
        _top.Children.Insert(i++, SourceMenu(pane));
        _top.Children.Insert(i++, ExtentMenu(pane, s));
        _top.Children.Insert(i++, CenterMenu(pane, s));
        _top.Children.Insert(i, SizeMenu(pane, s));
        _opts["pick"] = () => { if (_centerButton is not null) _picker?.ShowAt(_centerButton); };
    }

    static DropDownButton MenuButton(string title, string name, string value, string help, FlyoutBase flyout)
    {
        var b = new DropDownButton
        {
            Content = Kit.Text(title, "BodyText", 12), Flyout = flyout, Padding = new Thickness(8, 2, 6, 3), MinHeight = 26, MinWidth = 0,
            Background = new SolidColorBrush(Color.FromArgb(0, 0, 0, 0)), BorderThickness = new Thickness(0),
        };
        ToolTipService.SetToolTip(b, help);
        AutomationProperties.SetName(b, name);
        AutomationProperties.SetHelpText(b, value);
        return b;
    }

    static RadioMenuFlyoutItem Radio(string text, string group, bool on, Action click)
    {
        var item = new RadioMenuFlyoutItem { Text = text, GroupName = group, IsChecked = on };
        item.Click += (_, _) => click();
        return item;
    }

    /// <summary>The source: a built-in pattern, the imported file, or Choose Image….</summary>
    FrameworkElement SourceMenu(AnalysisPaneModel pane)
    {
        var s = pane.Settings;
        var file = ImageSourceImport.FileName(s);
        var src = Json.At(s, "source");
        var kind = Json.Str(Json.At(src, "kind")) ?? "pattern";
        var pattern = Json.Str(Json.At(src, "pattern")) ?? "grid";
        var title = file ?? kind switch
        {
            "buffer" => Loc.S("Results_ImageSim_ImageBuffer"),
            "path" => Path.GetFileName(Json.Str(Json.At(src, "path")) ?? ""),
            _ => ImageSourcePatterns.Title(pattern),
        };
        var menu = new MenuFlyout { Placement = FlyoutPlacementMode.BottomEdgeAlignedLeft };
        foreach (var (id, t) in ImageSourcePatterns.All)
            menu.Items.Add(Radio(t, "source", file is null && kind == "pattern" && pattern == id, () => pane.SetSettings(ImageSourceImport.WithPattern(pane.Settings, id))));
        if (file is not null) menu.Items.Add(Radio(file, "source", true, () => { }));
        menu.Items.Add(new MenuFlyoutSeparator());
        var choose = new MenuFlyoutItem { Text = Loc.S("Results_ImageSim_ChooseImage"), Icon = new FontIcon { Glyph = "\uEB9F" } };
        choose.Click += async (_, _) =>
        {
            if (await ImageSimulationFiles.PickAsync(this) is { } path) await UseFileAsync(pane, path);
        };
        menu.Items.Add(choose);
        return MenuButton(Loc.F("Results_ImageSim_SourceButton", title), Loc.S("Results_ImageSim_Source"), title, Loc.S("Results_ImageSim_SourceToolTip"), menu);
    }

    /// <summary>The field of view: a square around the patch centre, as a fraction of the field.</summary>
    FrameworkElement ExtentMenu(AnalysisPaneModel pane, JsonObject s)
    {
        var extent = ImageSimulationSettings.FieldExtent(s);
        var size = ImageSimulationSettings.Size(s);
        var k = ImageSimulationSettings.MmPerExtent(_d.PixelX, _d.Image.Width, _d.FieldExtent);
        var menu = new MenuFlyout { Placement = FlyoutPlacementMode.BottomEdgeAlignedLeft };
        foreach (var e in ImageSimulationSettings.Extents)
            menu.Items.Add(Radio(ImageSimulationSettings.ExtentLabel(e, k, size), "extent", Math.Abs(e - extent) < 1e-9,
                () => pane.SetSettings(ImageSimulationSettings.WithFieldExtent(pane.Settings, e))));
        var title = ImageSimulationSettings.ExtentTitle(extent);
        return MenuButton(Loc.F("Results_ImageSim_ViewButton", title), Loc.S("Results_ImageSim_FieldOfView"), ImageSimulationSettings.ExtentLabel(extent, k, size),
            Loc.S("Results_ImageSim_FieldOfViewToolTip"), menu);
    }

    /// <summary>The patch centre: the axis, a design field point, or a point picked on the field diagram.</summary>
    FrameworkElement CenterMenu(AnalysisPaneModel pane, JsonObject s)
    {
        var center = ImageSimulationSettings.FieldCenter(s);
        var extent = ImageSimulationSettings.FieldExtent(s);
        var points = FieldPoints();
        var menu = new MenuFlyout { Placement = FlyoutPlacementMode.BottomEdgeAlignedLeft };
        void Set((double X, double Y) c) => pane.SetSettings(ImageSimulationSettings.WithFieldCenter(pane.Settings, c));
        var known = false;
        foreach (var p in points)
        {
            var on = Math.Abs(p.X - center.X) < 1e-4 && Math.Abs(p.Y - center.Y) < 1e-4;
            known |= on;
            menu.Items.Add(Radio(p.Title, "center", on, () => Set((p.X, p.Y))));
        }
        if (!known) menu.Items.Add(Radio(ImageSimulationSettings.CenterTitle(center, points), "center", true, () => { }));
        menu.Items.Add(new MenuFlyoutSeparator());
        var pick = new MenuFlyoutItem { Text = Loc.S("Results_ImageSim_PickOnField"), Icon = new FontIcon { Glyph = "\uE81D" } };
        menu.Items.Add(pick);
        var title = ImageSimulationSettings.CenterTitle(center, points);
        var b = MenuButton(center.X == 0 && center.Y == 0 ? Loc.S("Results_ImageSim_AtAxis") : Loc.F("Results_ImageSim_At", title), Loc.S("Results_ImageSim_PatchCenter"), title, Loc.S("Results_ImageSim_PatchCenterToolTip"), menu);
        _centerButton = b;
        _picker = new Flyout
        {
            Content = new ImageSimulationFieldPicker(center, extent, points, Set),
            Placement = FlyoutPlacementMode.BottomEdgeAlignedLeft,
        };
        pick.Click += (_, _) => _picker.ShowAt(b);
        return b;
    }

    /// <summary>The resolution: output pixels per side (an imported image is resampled to it in the same step).</summary>
    FrameworkElement SizeMenu(AnalysisPaneModel pane, JsonObject s)
    {
        var size = ImageSimulationSettings.Size(s);
        var menu = new MenuFlyout { Placement = FlyoutPlacementMode.BottomEdgeAlignedLeft };
        foreach (var n in ImageSimulationSettings.Sizes)
            menu.Items.Add(Radio(Loc.F("Results_ImageSim_SizeItem", n), "size", n == size, async () => pane.SetSettings(await ImageSimulationFiles.WithSizeAsync(pane, n))));
        return MenuButton(Loc.F("Results_ImageSim_SizeButton", size), Loc.S("Results_ImageSim_Resolution"), Loc.F("Results_ImageSim_Pixels", size), Loc.S("Results_ImageSim_ResolutionToolTip"), menu);
    }
}
