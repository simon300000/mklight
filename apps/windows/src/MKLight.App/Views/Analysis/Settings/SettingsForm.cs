using System.Globalization;
using System.Text.Json.Nodes;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using MKLight.App.Views.Analysis.Results.Readers;
using static MKLight.App.Views.Analysis.Settings.FormControls;

namespace MKLight.App.Views.Analysis.Settings;

/// <summary>
/// The settings of a pane, generated from the analysis' settings schema
/// alone (the Mac app's SettingsBar): a two-column form, labels on the
/// left, controls on the right. Objects are groups of their properties,
/// tagged unions a variant picker with the chosen variant's rows, index
/// settings menus of the system's fields, wavelengths and surfaces. Unset
/// settings show the engine default; settings without a control keep their value.
/// </summary>
public sealed partial class SettingsForm : UserControl
{
    readonly List<SettingsField> _fields;
    readonly SettingsContext _ctx;
    readonly string _analysis;
    readonly Action<JsonObject> _onChange;
    readonly StackPanel _root = new() { Spacing = 8 };
    JsonObject _settings;
    bool _rebuildQueued;

    /// <summary>
    /// The form for <paramref name="descriptor"/> (an element of
    /// <c>mk_engine_list_analyses</c>) editing <paramref name="current"/>;
    /// <paramref name="onChange"/> receives the complete new settings after
    /// every change (never for a value equal to the current one).
    /// </summary>
    public static UIElement Create(JsonObject descriptor, JsonObject current, SettingsContext ctx, Action<JsonObject> onChange) =>
        new SettingsForm(descriptor, current, ctx, onChange);

    SettingsForm(JsonObject descriptor, JsonObject current, SettingsContext ctx, Action<JsonObject> onChange)
    {
        _fields = SettingsField.Of(descriptor);
        _ctx = ctx;
        _analysis = ctx.Analysis.Length > 0 ? ctx.Analysis : Json.Str(descriptor["name"]) ?? "";
        _onChange = onChange;
        _settings = (JsonObject)current.DeepClone();
        IsTabStop = false;
        Content = _root;
        Build();
    }

    /// <summary>The settings as the form holds them.</summary>
    public JsonObject Settings => (JsonObject)_settings.DeepClone();

    // MARK: Values

    /// <summary>Where a control reads and writes its value: a top-level setting, or a key of an enclosing object.</summary>
    sealed class Slot
    {
        public required string Path;
        public required SettingsField Field;
        public required Func<(bool Has, JsonNode? Value)> Read;
        public required Action<Edit, bool> Write;

        /// <summary>The value shown: the stored one, else the schema default.</summary>
        public JsonNode? Current => Read() is (true, var v) ? v : Field.Default;

        /// <summary>The slot of a property inside this object. Resetting a child puts its default back (a partial object is never sent).</summary>
        public Slot Child(SettingsField f) => new()
        {
            Path = Path + "." + f.Name,
            Field = f,
            Read = () => Current is JsonObject o && o.TryGetPropertyValue(f.Name, out var v) ? (true, v) : (false, null),
            Write = (e, rebuild) =>
            {
                var o = Current is JsonObject cur ? (JsonObject)cur.DeepClone() : new JsonObject();
                if (e.Remove) { if (f.HasDefault) o[f.Name] = Clone(f.Default); else o.Remove(f.Name); }
                else o[f.Name] = Clone(e.Value);
                Write(new Edit(o), rebuild);
            },
        };
    }

    Slot TopSlot(SettingsField f) => new()
    {
        Path = f.Name,
        Field = f,
        Read = () => _settings.TryGetPropertyValue(f.Name, out var v) ? (true, v) : (false, null),
        Write = (e, rebuild) =>
        {
            var next = (JsonObject)_settings.DeepClone();
            if (e.Remove) next.Remove(f.Name); else next[f.Name] = Clone(e.Value);
            if (Json.Same(next, _settings)) return;
            _settings = next;
            _onChange((JsonObject)_settings.DeepClone());
            if (rebuild) QueueRebuild();
        },
    };

    /// <summary>Rebuilds after a change that may change the rows (outside the control's own event).</summary>
    void QueueRebuild()
    {
        if (_rebuildQueued) return;
        _rebuildQueued = true;
        DispatcherQueue.TryEnqueue(() => { _rebuildQueued = false; Build(); });
    }

    // MARK: Layout

    void Build()
    {
        _root.Children.Clear();
        _popups.Clear();
        _multiSelects.Clear();
        var pattern = _fields.Any(f => f.Name == "pattern") ? Json.Str(_settings["pattern"] ?? _fields.First(f => f.Name == "pattern").Default) : null;
        var rows = _fields.Select(f => (Field: f, Control: SettingControls.For(f, _analysis, _ctx.HasConfigurations, pattern)))
            .Where(r => r.Control.Kind != ControlKind.Hidden).ToList();
        if (rows.Count == 0)
        {
            _root.Children.Add(Label("This analysis has no settings to change."));
            return;
        }
        // The spectrum, the largest control, comes last on its own.
        _root.Children.Add(Rows(rows.Where(r => r.Control.Kind != ControlKind.Spectrum).Select(r => (TopSlot(r.Field), r.Control))));
        foreach (var r in rows.Where(r => r.Control.Kind == ControlKind.Spectrum))
        {
            _root.Children.Add(Divider());
            var slot = TopSlot(r.Field);
            var system = _ctx.WavelengthsUm.Select(um => new SpectrumSample(um, 1)).ToList();
            _root.Children.Add(new SpectrumControl(slot.Current, system, _ctx.PrimaryWavelength,
                v => slot.Write(v is null ? Edit.Reset : new Edit(v), false)));
        }
    }

    /// <summary>A hairline in the window's theme (resolved once the line is shown).</summary>
    static Border Divider()
    {
        var line = new Border { Height = 1, Margin = new Thickness(0, 2, 0, 2) };
        void Paint() => line.Background = MKLight.App.Support.ThemeLookup.Brush(line, "DividerBrush");
        line.Loaded += (_, _) => Paint();
        line.ActualThemeChanged += (_, _) => Paint();
        return line;
    }

    /// <summary>A two-column grid: right-aligned labels, controls on the left of their column.</summary>
    Grid Rows(IEnumerable<(Slot Slot, SettingControl Control)> rows)
    {
        var g = new Grid { ColumnSpacing = 10, RowSpacing = 6 };
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        var r = 0;
        foreach (var (slot, control) in rows)
        {
            var f = slot.Field;
            var label = Label(SettingEntry.Label(f), string.IsNullOrEmpty(f.Description) ? f.Title : f.Description);
            label.HorizontalAlignment = HorizontalAlignment.Right;
            label.VerticalAlignment = VerticalAlignment.Top;
            label.Margin = new Thickness(0, 5, 0, 0);
            if (Control(slot, control) is not FrameworkElement element) continue;
            element.HorizontalAlignment = HorizontalAlignment.Left;
            Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(element, SettingEntry.Label(f));
            Microsoft.UI.Xaml.Automation.AutomationProperties.SetAutomationId(element, $"Settings.{_analysis}.{slot.Path}");
            g.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            Grid.SetRow(label, r);
            Grid.SetRow(element, r);
            Grid.SetColumn(element, 1);
            g.Children.Add(label);
            g.Children.Add(element);
            r++;
        }
        return g;
    }

    UIElement? Control(Slot slot, SettingControl c) => c.Kind switch
    {
        ControlKind.Toggle => Toggle(slot),
        ControlKind.Number => Number(slot),
        ControlKind.Text => TextField(slot),
        ControlKind.Picker => Picker(slot),
        ControlKind.FieldSelection => MultiSelect(slot, IndexDomain.Fields, SelectionEncoding.Tagged, allowsPrimary: false),
        ControlKind.WavelengthSelection => MultiSelect(slot, IndexDomain.Wavelengths, SelectionEncoding.Tagged, allowsPrimary: true),
        ControlKind.FieldIndex => IndexMenu(slot, _ctx.FieldOptions, slot.Field.Nullable ? "Auto" : null),
        ControlKind.WavelengthIndex => IndexMenu(slot, _ctx.WavelengthOptions, slot.Field.Name == "wavelength" ? "Primary" : "Auto"),
        ControlKind.SurfaceIndex => IndexMenu(slot, _ctx.SurfaceOptions, slot.Field.Nullable ? "Auto" : null),
        ControlKind.IndexList => MultiSelect(slot, c.Domain, SelectionEncoding.List, allowsPrimary: false),
        ControlKind.NumberList => NumberList(slot, c.Integer),
        ControlKind.Pair => Pair(slot, c.Integer, c.Nullable),
        ControlKind.Group => Group(slot, c.Nullable),
        ControlKind.Variants => Variants(slot, c.Nullable),
        ControlKind.Pupil => Pupil(slot),
        ControlKind.ThroughFocus => ThroughFocus(slot),
        ControlKind.Table => TableButton(slot, c.Columns ?? [], c.RowsAreArrays),
        ControlKind.SensorLimit => SensorLimitButton(slot),
        ControlKind.ImageSource => ImageSource(slot),
        ControlKind.ImageOptics => ImageOptics(slot),
        _ => null,
    };

    static string Help(SettingsField f) => string.IsNullOrEmpty(f.Description) ? f.Title : f.Description;

    // MARK: Scalars

    UIElement Toggle(Slot slot)
    {
        var box = new CheckBox { IsChecked = Json.Bool(slot.Current) ?? false, MinWidth = 0, Padding = new Thickness(0), MinHeight = 28 };
        ToolTipService.SetToolTip(box, Help(slot.Field));
        box.Click += (_, _) => slot.Write(new Edit(box.IsChecked == true), false);
        return box;
    }

    UIElement Number(Slot slot)
    {
        var f = slot.Field;
        var auto = AutoSettings.Sentinel(slot.Path, _analysis);
        var placeholder = auto is not null ? AutoSettings.Display(AutoSettings.Used(slot.Path, _analysis, _ctx.LastResult)) : f.Nullable ? "auto" : "";
        var box = new CommitBox(
            exact => SettingEntry.NumberDisplay(slot.Current, f.IsInteger, auto, exact),
            text => SettingEntry.ParseNumber(text, f, auto),
            e => slot.Write(e, false),
            auto is null ? 72 : Math.Max(72, placeholder.Length * 7 + 20)) { PlaceholderText = placeholder };
        var help = Help(f);
        if (auto is not null)
            help += AutoSettings.IsAuto(slot.Current, auto.Value)
                ? "\nAutomatic: the engine chooses from the system."
                : auto == AutoSentinel.Zero ? "\nClear the field (or type 0) to let the engine choose." : "\nClear the field to let the engine choose.";
        ToolTipService.SetToolTip(box, help);
        return f.Unit.Length > 0 ? Row(box, Label(f.Unit)) : box;
    }

    UIElement TextField(Slot slot)
    {
        var f = slot.Field;
        var box = new CommitBox(_ => Json.Str(slot.Current) ?? "", text => SettingEntry.ParseText(text, f), e => slot.Write(e, false), 150, number: false)
        {
            PlaceholderText = f.Nullable ? "any" : "",
        };
        ToolTipService.SetToolTip(box, Help(f));
        return box;
    }

    UIElement Picker(Slot slot)
    {
        var f = slot.Field;
        var combo = new ComboBox { FontSize = 12, MinHeight = 28, MinWidth = 0, Padding = new Thickness(8, 3, 6, 4) };
        foreach (var v in f.Enum) combo.Items.Add(SettingEntry.Title(v));
        var current = slot.Current;
        for (var i = 0; i < f.Enum.Count; i++) if (Json.Same(f.Enum[i], current)) combo.SelectedIndex = i;
        combo.SelectionChanged += (_, _) =>
        {
            if (combo.SelectedIndex >= 0) slot.Write(new Edit(Clone(f.Enum[combo.SelectedIndex])), slot.Field.Name == "pattern");
        };
        ToolTipService.SetToolTip(combo, Help(f));
        return combo;
    }

    // MARK: Menus of the system's fields, wavelengths and surfaces

    UIElement IndexMenu(Slot slot, IReadOnlyList<(int Index, string Title)> options, string? nullTitle)
    {
        var menu = new MenuFlyout();
        if (nullTitle is not null) menu.Items.Add(Item(nullTitle, () => slot.Write(new Edit(null), true)));
        foreach (var (k, title) in options) menu.Items.Add(Item(title, () => slot.Write(new Edit(k), true)));
        return Menu(SettingEntry.IndexTitle(slot.Current, options, nullTitle), menu, Help(slot.Field));
    }

    // MARK: Lists and pairs

    UIElement NumberList(Slot slot, bool integer)
    {
        var f = slot.Field;
        var auto = AutoSettings.Sentinel(slot.Path, _analysis);
        var placeholder = auto is not null ? AutoSettings.Display(AutoSettings.Used(slot.Path, _analysis, _ctx.LastResult)) : f.Nullable ? "auto" : "none";
        var box = new CommitBox(
            exact => SettingEntry.ListDisplay(slot.Current, integer, exact),
            text => SettingEntry.ParseList(text, f, integer, auto),
            e => slot.Write(e, false),
            Math.Max(110, placeholder.Length * 7 + 20)) { PlaceholderText = placeholder, TextAlignment = TextAlignment.Left };
        ToolTipService.SetToolTip(box, Help(f) + "\nComma separated" + (auto is not null ? "; empty lets the engine choose." : "."));
        var unit = f.Unit.Length > 0 ? f.Unit : auto is not null ? "cycles/mm" : "";
        return unit.Length > 0 ? Row(box, Label(unit)) : box;
    }

    /// <summary>
    /// Two numbers in two boxes; the pair is one value, committed from the
    /// typed number and, for the other box, its stored one. Half a pair (one
    /// empty box of an automatic pair) is not a value and waits for the other.
    /// </summary>
    UIElement Pair(Slot slot, bool integer, bool nullable)
    {
        var f = slot.Field;
        var bounds = SettingEntry.PairBounds(f);
        var (before, between) = SettingEntry.PairWords(f);
        double? Stored(int k) => Json.Arr(slot.Current) is { Count: 2 } a ? Json.Number(a[k]) : null;
        string Text(int k, bool exact) => Stored(k) is double v ? integer ? NumberText.Integer(v) : exact ? NumberText.Exact(v) : NumberText.Short(v) : "";
        var boxes = new CommitBox[2];
        for (var k = 0; k < 2; k++)
        {
            var i = k;
            boxes[k] = new CommitBox(
                exact => Text(i, exact),
                text =>
                {
                    double? x = null;
                    if (text.Trim().Length > 0)
                    {
                        x = SettingEntry.ParsePairPart(text, integer, bounds);
                        if (x is null) return null;
                    }
                    var other = boxes[1 - i];
                    var o = other.Text.Trim().Length == 0 ? null : SettingEntry.ParsePairPart(other.Text, integer, bounds) ?? Stored(1 - i);
                    return i == 0 ? SettingEntry.Pair(x, o, integer, nullable) : SettingEntry.Pair(o, x, integer, nullable);
                },
                e => slot.Write(e, false),
                nullable ? 60 : integer ? 52 : 64) { PlaceholderText = nullable ? "auto" : "" };
            ToolTipService.SetToolTip(boxes[k], Help(f));
        }
        var row = Row();
        if (before is not null) row.Children.Add(Label(before));
        row.Children.Add(boxes[0]);
        row.Children.Add(Label(between));
        row.Children.Add(boxes[1]);
        if (f.Unit.Length > 0) row.Children.Add(Label(f.Unit));
        return row;
    }

    // MARK: Objects and tagged unions

    /// <summary>The rows of an object's properties (constants, the tags, are not shown).</summary>
    UIElement Nested(Slot slot, IEnumerable<SettingsField> properties)
    {
        var rows = properties.Where(p => !p.HasConst)
            .Select(p => (Slot: slot.Child(p), Control: SettingControls.For(p, _analysis, nested: true)))
            .Where(r => r.Control.Kind != ControlKind.Hidden);
        return Rows(rows);
    }

    /// <summary>An object's properties as rows; an optional object gets an on/off switch first.</summary>
    UIElement Group(Slot slot, bool nullable)
    {
        var f = slot.Field;
        var on = !nullable || slot.Current is not null;
        var p = new StackPanel { Spacing = 4 };
        if (nullable)
        {
            var box = new CheckBox { IsChecked = on, Content = on ? "On" : "Off", MinWidth = 0, MinHeight = 28, FontSize = 12 };
            ToolTipService.SetToolTip(box, Help(f));
            box.Click += (_, _) => slot.Write(new Edit(box.IsChecked == true ? SettingControls.EnabledValue(f) : null), true);
            p.Children.Add(box);
        }
        if (on) p.Children.Add(Nested(slot, f.Properties));
        return p;
    }

    /// <summary>A tagged union: a picker of the variants ("Off" when null is allowed), then the chosen variant's rows.</summary>
    UIElement Variants(Slot slot, bool nullable)
    {
        var f = slot.Field;
        var current = f.VariantOf(slot.Current);
        var combo = new ComboBox { FontSize = 12, MinHeight = 28, MinWidth = 0, Padding = new Thickness(8, 3, 6, 4) };
        if (nullable) combo.Items.Add("Off");
        foreach (var v in f.Variants) combo.Items.Add(v.Title);
        var offset = nullable ? 1 : 0;
        combo.SelectedIndex = current is null ? (nullable && slot.Current is null ? 0 : -1) : f.Variants.ToList().IndexOf(current) + offset;
        combo.SelectionChanged += (_, _) =>
        {
            var i = combo.SelectedIndex - offset;
            if (combo.SelectedIndex < 0) return;
            slot.Write(new Edit(i < 0 ? null : SettingControls.ChooseVariant(f.Variants[i], slot.Current)), true);
        };
        ToolTipService.SetToolTip(combo, Help(f));
        var p = new StackPanel { Spacing = 4 };
        p.Children.Add(combo);
        if (current is not null && current.Properties.Any(x => !x.HasConst)) p.Children.Add(Nested(slot, current.Properties));
        return p;
    }

    // MARK: Through focus (spot diagram)

    static readonly SettingsField Bound = new("bound", new JsonObject
    {
        ["type"] = new JsonArray("number", "null"),
        ["description"] = "Half-range of the planes; empty lets the engine choose from the depth of focus and the geometric blur.",
    });

    /// <summary>Off (the image surface only), or planes from −Δ to +Δ with Δ automatic or typed, and the number of planes (1 to 15).</summary>
    UIElement ThroughFocus(Slot slot)
    {
        // The current bounds and steps, read when a control commits (a typed bound does not rebuild the row).
        double? Min() => Json.Number(Json.At(slot.Current, "min"));
        double? Max() => Json.Number(Json.At(slot.Current, "max"));
        int Steps() => Json.Int(Json.At(slot.Current, "steps")) ?? 3;
        void Write(double? min, double? max, int steps, bool rebuild) => slot.Write(new Edit(SettingEntry.ThroughFocus(min, max, steps)), rebuild);

        var on = slot.Current is not null;
        double? min0 = Min(), max0 = Max();
        var steps = Steps();
        var symmetric = (min0 is null && max0 is null) || (min0 is double a && max0 is double b && Math.Abs(a + b) <= 1e-12 * Math.Max(1, Math.Abs(b)));
        var p = new StackPanel { Spacing = 4 };
        var box = new CheckBox { IsChecked = on, Content = on ? "Planes before and after focus" : "Image surface only", MinWidth = 0, MinHeight = 28, FontSize = 12 };
        ToolTipService.SetToolTip(box, Help(slot.Field));
        box.Click += (_, _) => slot.Write(new Edit(box.IsChecked == true ? SettingEntry.ThroughFocus(null, null, steps) : null), true);
        p.Children.Add(box);
        if (!on) return p;

        CommitBox BoundBox(Func<double?> get, Action<double?> set, string placeholder) => new(
            exact => SettingEntry.NumberDisplay(get() is double d ? d : null, false, AutoSentinel.Null, exact),
            text => SettingEntry.ParseNumber(text, Bound, AutoSentinel.Null),
            e => set(Json.Number(e.Value)),
            Math.Max(72, placeholder.Length * 7 + 20)) { PlaceholderText = placeholder };

        var row = Row();
        if (symmetric)
        {
            var used = AutoSettings.Used("through_focus", _analysis, _ctx.LastResult)?.Replace("±", "");
            var half = BoundBox(Max, d => Write(d is double x ? -Math.Abs(x) : null, d is double y ? Math.Abs(y) : null, Steps(), false), AutoSettings.Display(used));
            ToolTipService.SetToolTip(half, Bound.Description);
            row.Children.Add(Label("±"));
            row.Children.Add(half);
        }
        else
        {
            row.Children.Add(BoundBox(Min, d => Write(d, Max(), Steps(), false), "auto"));
            row.Children.Add(Label("to"));
            row.Children.Add(BoundBox(Max, d => Write(Min(), d, Steps(), false), "auto"));
        }
        row.Children.Add(Label("mm"));
        var count = Label(steps == 1 ? "1 plane" : steps.ToString(CultureInfo.InvariantCulture) + " planes", style: "NumberText");
        count.Margin = new Thickness(6, 0, 0, 0);
        row.Children.Add(count);
        row.Children.Add(StepButton("\uE949", "Fewer planes", steps > 1, () => Write(Min(), Max(), Math.Max(Steps() - 1, 1), true)));
        row.Children.Add(StepButton("\uE948", "More planes", steps < 15, () => Write(Min(), Max(), Math.Min(Steps() + 1, 15), true)));
        p.Children.Add(row);
        return p;
    }

    static Button StepButton(string glyph, string help, bool enabled, Action click)
    {
        var b = new Button
        {
            Content = new FontIcon { Glyph = glyph, FontSize = 10 },
            Width = 26, Height = 26, Padding = new Thickness(0), MinWidth = 0, MinHeight = 0, CornerRadius = new CornerRadius(13), IsEnabled = enabled,
        };
        ToolTipService.SetToolTip(b, help);
        b.Click += (_, _) => click();
        return b;
    }
}
