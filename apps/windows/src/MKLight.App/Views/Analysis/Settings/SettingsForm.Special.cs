using System.Text.Json.Nodes;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using MKLight.App.Views.Analysis.Results.Readers;
using static MKLight.App.Views.Analysis.Settings.FormControls;

namespace MKLight.App.Views.Analysis.Settings;

// The form's dedicated controls: the pupil sampling, row tables and the
// sensor limit in popups, the image simulation's source and optics.
public sealed partial class SettingsForm
{
    // MARK: Pupil sampling

    /// <summary>The wave analyses' pupil sampling in one row: the grid (automatic by default, "Auto (128)") and the traced rays.</summary>
    UIElement Pupil(Slot slot)
    {
        var row = Row();
        if (slot.Field.Properties.FirstOrDefault(p => p.Name == "grid") is { } grid) row.Children.Add(Number(slot.Child(grid)));
        if (slot.Field.Properties.FirstOrDefault(p => p.Name == "rays") is { } rays)
        {
            var label = Label("Rays", Help(rays));
            label.Margin = new Thickness(6, 0, 0, 0);
            row.Children.Add(label);
            row.Children.Add(Number(slot.Child(rays)));
        }
        return row;
    }

    // MARK: Popups

    readonly Dictionary<string, Button> _popups = [];

    /// <summary>The popup button of a top-level setting (launch actions open it for screenshots).</summary>
    public Button? PopupFor(string setting) => _popups.GetValueOrDefault(setting);

    /// <summary>A borderless button opening a popup editor; its title follows the value.</summary>
    static Button PopupButton(string title, string help, Func<Action<string>, UIElement> editor)
    {
        var b = new Button { Content = title, FontSize = 12, MinHeight = 28, Padding = new Thickness(8, 3, 8, 4), MinWidth = 0 };
        ToolTipService.SetToolTip(b, help);
        var flyout = new Flyout { Placement = FlyoutPlacementMode.BottomEdgeAlignedLeft };
        flyout.Opening += (_, _) => flyout.Content = editor(t => b.Content = t);
        b.Flyout = flyout;
        return b;
    }

    static TextBlock Heading(string text) => new() { Text = text, Style = (Style)Application.Current.Resources["PanelTitleText"] };

    static TextBlock Hint(string text)
    {
        var t = Label(text, style: "CaptionText");
        t.TextWrapping = TextWrapping.Wrap;
        t.TextTrimming = TextTrimming.None;
        return t;
    }

    static Button IconButton(string glyph, string help, Action click)
    {
        var b = new Button
        {
            Content = new FontIcon { Glyph = glyph, FontSize = 11 }, Style = (Style)Application.Current.Resources["CircleButton"], Width = 24, Height = 24,
        };
        ToolTipService.SetToolTip(b, help);
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(b, help);
        b.Click += (_, _) => click();
        return b;
    }

    static Button TextButton(string glyph, string text, Action click)
    {
        var content = Row(new FontIcon { Glyph = glyph, FontSize = 11 }, new TextBlock { Text = text, FontSize = 12 });
        var b = new Button { Content = content, FontSize = 12, MinHeight = 26, MinWidth = 0, Padding = new Thickness(8, 2, 10, 3), CornerRadius = new CornerRadius(13) };
        b.Click += (_, _) => click();
        return b;
    }

    /// <summary>A grid of fixed-width columns with a caption header, rows added by the caller.</summary>
    static Grid TableGrid(IEnumerable<string> headers, double width)
    {
        var g = new Grid { ColumnSpacing = 6, RowSpacing = 4 };
        var k = 0;
        g.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        foreach (var h in headers)
        {
            g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(width) });
            var t = Label(h, style: "CaptionText");
            t.HorizontalAlignment = HorizontalAlignment.Right;
            Grid.SetColumn(t, k++);
            g.Children.Add(t);
        }
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(26) });
        return g;
    }

    static void Put(Grid g, UIElement e, int row, int col)
    {
        while (g.RowDefinitions.Count <= row) g.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        Grid.SetRow((FrameworkElement)e, row);
        Grid.SetColumn((FrameworkElement)e, col);
        g.Children.Add(e);
    }

    // MARK: Row tables ([{"hx", "hy"}], [[r, θ]])

    /// <summary>Rows of the same scalar columns, edited as a small table in a popup; the row count is the button.</summary>
    UIElement TableButton(Slot slot, IReadOnlyList<SettingsField> columns, bool rowsAreArrays)
    {
        var f = slot.Field;
        return _popups[slot.Path] = PopupButton(TableSetting.ButtonTitle(f, slot.Current), Help(f), retitle =>
        {
            var rows = TableSetting.Rows(slot.Current, columns, rowsAreArrays);
            var panel = new StackPanel { Spacing = 8, Width = Math.Clamp(80 * columns.Count + 70, 240, 520) };
            panel.Children.Add(Heading(SettingEntry.Label(f)));
            if (f.Description.Length > 0) panel.Children.Add(Hint(f.Description));
            var host = new ContentControl { HorizontalContentAlignment = HorizontalAlignment.Left, IsTabStop = false };
            panel.Children.Add(host);
            void Commit()
            {
                var value = TableSetting.Value(rows, columns, rowsAreArrays);
                slot.Write(new Edit(value), false);
                retitle(TableSetting.ButtonTitle(f, value));
            }
            void Fill()
            {
                var g = TableGrid(columns.Select(c => c.Unit.Length > 0 ? $"{SettingEntry.Label(c)} ({c.Unit})" : SettingEntry.Label(c)), 74);
                for (var r = 0; r < rows.Count; r++)
                {
                    var cells = rows[r];
                    for (var k = 0; k < columns.Count; k++)
                    {
                        var col = columns[k];
                        var i = k;
                        var box = new CommitBox(exact => TableSetting.Text(cells[i], exact), text => TableSetting.Parse(text, col),
                            e => { cells[i] = e.Value?.DeepClone(); Commit(); }, 74, number: !col.BaseTypes.SetEquals(["string"]));
                        Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(box, SettingEntry.Label(col));
                        Put(g, box, r + 1, k);
                    }
                    Put(g, IconButton("\uE738", "Delete this row", () => { rows.Remove(cells); Commit(); DispatcherQueue.TryEnqueue(Fill); }), r + 1, columns.Count);
                }
                var footer = Row(TextButton("\uE710", "Add Row", () =>
                {
                    rows.Add(rows.Count > 0 ? rows[^1].Select(c => c?.DeepClone()).ToArray() : columns.Select(c => c.Default?.DeepClone()).ToArray());
                    Commit();
                    DispatcherQueue.TryEnqueue(Fill);
                }));
                if (rows.Count > 0) footer.Children.Add(TextButton("\uE894", "Clear", () => { rows.Clear(); Commit(); DispatcherQueue.TryEnqueue(Fill); }));
                var p = new StackPanel { Spacing = 8 };
                if (rows.Count > 0) p.Children.Add(g);
                p.Children.Add(footer);
                host.Content = p;
            }
            Fill();
            return panel;
        });
    }

    // MARK: Sensor limit (chief-ray angle)

    /// <summary>The chief-ray sensor limit, <c>[[image radius mm, max incidence °], …]</c>, as a small table in a popup.</summary>
    UIElement SensorLimitButton(Slot slot)
    {
        return _popups[slot.Path] = PopupButton(SensorLimit.Title(slot.Current), Help(slot.Field), retitle =>
        {
            var rows = SensorLimit.Rows(slot.Current);
            var panel = new StackPanel { Spacing = 8, Width = 300 };
            panel.Children.Add(Heading("Sensor CRA Limit"));
            panel.Children.Add(Hint("Maximum chief-ray incidence against image radius; linear between points, held beyond the ends."));
            var host = new ContentControl { HorizontalContentAlignment = HorizontalAlignment.Left, IsTabStop = false };
            panel.Children.Add(host);
            void Commit(bool refill)
            {
                var value = SensorLimit.Value(rows);
                rows = SensorLimit.Rows(value);
                slot.Write(value is null ? Edit.Reset : new Edit(value), false);
                retitle(SensorLimit.Title(value));
                if (refill) DispatcherQueue.TryEnqueue(Fill);
            }
            CommitBox Box(Func<double> get, Action<double> set, double minimum, string name)
            {
                var box = new CommitBox(exact => exact ? NumberText.Exact(get()) : NumberText.Short(get()),
                    text => NumberText.Parse(text) is double d && double.IsFinite(d) && d >= minimum ? new Edit(d) : null,
                    e => { set(Json.Number(e.Value) ?? get()); Commit(true); }, 96);
                Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(box, name);
                return box;
            }
            void Fill()
            {
                var g = TableGrid(["Radius (mm)", "Max angle (°)"], 96);
                for (var r = 0; r < rows.Count; r++)
                {
                    var i = r;
                    Put(g, Box(() => rows[i].Radius, v => rows[i] = (v, rows[i].Angle), 0, "Radius"), r + 1, 0);
                    Put(g, Box(() => rows[i].Angle, v => rows[i] = (rows[i].Radius, v), double.NegativeInfinity, "Maximum angle"), r + 1, 1);
                    Put(g, IconButton("\uE738", "Delete this point", () => { rows.RemoveAt(i); Commit(true); }), r + 1, 2);
                }
                var footer = Row(TextButton("\uE710", "Add Point", () => { rows.Add(SensorLimit.Next(rows)); Commit(true); }));
                if (rows.Count > 0) footer.Children.Add(TextButton("\uE894", "No Limit", () => { rows.Clear(); Commit(true); }));
                var p = new StackPanel { Spacing = 8 };
                if (rows.Count > 0) p.Children.Add(g);
                p.Children.Add(footer);
                host.Content = p;
            }
            Fill();
            return panel;
        });
    }

    // MARK: Image simulation

    /// <summary>The image simulation's source: a built-in pattern (an image from a file or buffer keeps its name).</summary>
    UIElement ImageSource(Slot slot)
    {
        var v = slot.Current;
        var title = Json.Str(Json.At(v, "kind")) switch
        {
            "buffer" => Results.ImageSourceImport.FileName(_settings) ?? "Image buffer",
            "path" => Path.GetFileName(Json.Str(Json.At(v, "path")) ?? ""),
            _ => ImageSourcePatterns.Title(Json.Str(Json.At(v, "pattern")) ?? "grid"),
        };
        var menu = new MenuFlyout();
        foreach (var (id, t) in ImageSourcePatterns.All)
        {
            var item = new ToggleMenuFlyoutItem { Text = t, IsChecked = Json.Str(Json.At(v, "kind")) is null or "pattern" && (Json.Str(Json.At(v, "pattern")) ?? "grid") == id };
            // A pattern forgets an imported file (its path and field aspect go with the samples).
            item.Click += (_, _) => Replace(Results.ImageSourceImport.WithPattern(_settings, id));
            menu.Items.Add(item);
        }
        menu.Items.Add(new MenuFlyoutSeparator());
        var choose = new MenuFlyoutItem { Text = "Choose Image…" };
        choose.Click += async (_, _) =>
        {
            if (PaneOf(this) is not { } pane || await Results.ImageSimulationFiles.PickAsync(this) is not { } path) return;
            if (await Results.ImageSimulationFiles.UseAsync(pane, path))
            {
                _settings = (JsonObject)pane.Settings.DeepClone();
                QueueRebuild();
            }
        };
        menu.Items.Add(choose);
        var control = Menu(title, menu, Help(slot.Field));
        // An imported file's samples are derived again when the pane needs them (after a project opened).
        if (control is FrameworkElement fe)
            fe.Loaded += async (_, _) =>
            {
                if (PaneOf(fe) is { } pane && Results.ImageSourceImport.NeedsRefresh(pane.Settings, Results.ImageSourceImport.FieldAspect(pane.Output?.Json)))
                {
                    await Results.ImageSimulationFiles.RefreshAsync(pane);
                    _settings = (JsonObject)pane.Settings.DeepClone();
                    QueueRebuild();
                }
            };
        return control;
    }

    /// <summary>Replaces the whole settings (a change that touches more than one key) and rebuilds the form.</summary>
    void Replace(JsonObject settings)
    {
        _settings = (JsonObject)settings.DeepClone();
        _onChange(Settings);
        QueueRebuild();
    }

    /// <summary>The pane whose settings the form edits, found through the visual tree (null before it is shown).</summary>
    static ViewModels.AnalysisPaneModel? PaneOf(DependencyObject? e)
    {
        while (e is not null)
        {
            if (e is AnalysisPaneView pane) return pane.Pane;
            e = Microsoft.UI.Xaml.Media.VisualTreeHelper.GetParent(e);
        }
        return null;
    }

    static SettingsField Synthetic(string name, string type, JsonNode? defaultValue, string help, string unit = "")
    {
        var schema = new JsonObject { ["type"] = type, ["minimum"] = 0, ["description"] = help, ["default"] = defaultValue?.DeepClone() };
        if (unit.Length > 0) schema["unit"] = unit;
        return new SettingsField(name, schema);
    }

    /// <summary>
    /// The image simulation's optics sampling: the PSF window and spacing, the
    /// pupil grid and rays (automatic by default, shown with the values the run
    /// used) and the field grid of PSFs. Other keys are kept as they are.
    /// </summary>
    UIElement ImageOptics(Slot slot)
    {
        var props = slot.Field.Properties;
        SettingsField Prop(string name, string type, string help, string unit = "") =>
            props.FirstOrDefault(p => p.Name == name) ?? Synthetic(name, type, Json.At(slot.Field.Default, name), help, unit);
        var pupilDefault = Json.At(slot.Field.Default, "pupil");
        var pupil = slot.Child(Synthetic("pupil", "object", pupilDefault, ""));
        var grid = Synthetic("grid", "integer", Json.At(pupilDefault, "grid") ?? 0, "Pupil grid samples per side; 0 chooses from the wavefront slope (64 to 1024).");
        var rays = Synthetic("rays", "integer", Json.At(pupilDefault, "rays") ?? 65, "Rays traced per pupil diameter.");

        UIElement Labelled(string label, UIElement control, string help)
        {
            var l = Label(label, help);
            l.MinWidth = 70;
            l.TextAlignment = TextAlignment.Right;
            return Row(l, control);
        }
        var p = new StackPanel { Spacing = 6 };
        p.Children.Add(Row(
            Labelled("PSF window", Number(slot.Child(Prop("psf_size", "integer", "PSF window samples per side (odd); 0 sizes it from the blur and diffraction."))), "PSF window samples per side"),
            Labelled("Spacing", Number(slot.Child(Prop("psf_spacing_mm", "number", "PSF sample spacing; empty uses the smallest λ·F/# over the patch divided by 2.5.", "mm"))), "PSF sample spacing")));
        p.Children.Add(Row(
            Labelled("Pupil", Number(pupil.Child(grid)), grid.Description),
            Labelled("Rays", Number(pupil.Child(rays)), rays.Description)));
        p.Children.Add(Labelled("PSF grid", Number(slot.Child(Prop("field_grid", "integer", "PSFs per side of the patch; the image is mapped between them."))), "PSFs per side of the patch"));
        return p;
    }

}
