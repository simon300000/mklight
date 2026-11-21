using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Shapes;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.Engine;
using MKLight.Model;
using Windows.UI;

namespace MKLight.App.Views.Inspector;

/// <summary>
/// The inspector's System page (the Mac app's <c>SystemInspector</c>): name and notes, aperture, the x / y first
/// order of an anamorphic design, fields, wavelengths (with presets), environment, ray aiming, configurations and
/// the glass catalogs. Every edit is one undoable step; switching the aperture type or the field kind converts the
/// values with the engine's <c>convert_spec</c> so the design stays equivalent.
/// </summary>
sealed partial class SystemPage : UserControl
{
    readonly DesignSession _session;
    readonly FormHost _form;
    bool _pending, _dirty = true, _showVignetting;
    IReadOnlyList<CatalogInfo> _catalogs = [];

    public SystemPage(DesignSession session)
    {
        _session = session;
        _form = new FormHost(new FormContext(session.ShowToast));
        Content = _form;
        session.Document.Changed += Schedule;
        session.EngineDataChanged += Schedule;
        session.Workspace.CatalogsChanged += () => _ = LoadCatalogsAsync();
        RegisterPropertyChangedCallback(VisibilityProperty, (_, _) => { if (Visibility == Visibility.Visible && _dirty) Schedule(); });
        Loaded += (_, _) => { Schedule(); _ = LoadCatalogsAsync(); };
    }

    public FormHost Form => _form;
    DesignDocument Doc => _session.Document;
    Workspace Ws => _session.Workspace;

    async Task LoadCatalogsAsync()
    {
        try { _catalogs = await Ws.Engine.CatalogsAsync(); }
        catch (EngineException) { _catalogs = []; }
        Schedule();
    }

    void Schedule()
    {
        _dirty = true;
        if (_pending || Visibility != Visibility.Visible) return;
        _pending = true;
        DispatcherQueue.TryEnqueue(() =>
        {
            _pending = false;
            Refresh();
        });
    }

    public void Refresh()
    {
        _dirty = false;
        _form.Show(Sections(Doc.System));
    }

    /// <summary>One undoable edit of the system; a refused one shows the value again.</summary>
    void Edit(string name, Func<OpticalSystem, OpticalSystem?> change)
    {
        if (!Doc.Edit(name, change)) Schedule();
    }

    VariableSpec Variable(OpticalSystem sys, ParamPath path) =>
        new(InspectorVariables.IsVariable(Doc.Section("variables"), sys, path), SolveTitle(sys, path), on =>
        {
            if (InspectorVariables.Set(Doc.Section("variables"), Doc.System, path, on) is { } section)
                Doc.EditSection("variables", section, on ? Loc.S("Lens_Undo_MakeVariable") : Loc.S("Lens_Undo_RemoveVariable"));
        });

    static string? SolveTitle(OpticalSystem sys, ParamPath path) => sys.SolveFor(path) is { } s ? LensText.Solve(s.Kind.Type) : null;

    // ---- Conversions (convert_spec) ----

    /// <summary>
    /// The engine's conversion of the system's fields or aperture to another kind, run on the design's snapshot
    /// (or a snapshot of exactly this system); null where the engine cannot convert (the numbers then stay).
    /// </summary>
    async Task<SpecConversion?> ConvertSpecAsync(OpticalSystem system, FieldKind? fieldKind, ApertureKind? aperture)
    {
        try
        {
            var json = ModelJson.Encode(system);
            var snapshot = Ws.DesignSnapshot is { } s && s.SystemJson == json ? s : await Ws.Engine.OpenSnapshotAsync(json);
            var request = new EngineAnalysisRequest("convert_spec", InspectorEdits.ConvertSpecSettings(fieldKind, aperture));
            using var output = await Ws.Engine.Submit(request, snapshot).ResultAsync();
            return SpecConversion.FromJson(output.Json);
        }
        catch (Exception e) when (e is EngineException or ObjectDisposedException)
        {
            return null;
        }
    }

    async void SetApertureKind(ApertureKind k)
    {
        var before = Doc.System;
        var c = await ConvertSpecAsync(before, null, k);
        Edit(Loc.S("Lens_Undo_ChangeApertureType"), s => s.SetApertureKind(k, s == before ? c : null));
    }

    async void SetFieldKind(FieldKind k)
    {
        var before = Doc.System;
        var c = await ConvertSpecAsync(before, k, null);
        Edit(Loc.S("Lens_Undo_ChangeFieldType"), s => s.SetFieldKind(k, s == before ? c : null));
    }

    // ---- The form ----

    IReadOnlyList<FormSection> Sections(OpticalSystem sys)
    {
        var list = new List<FormSection>
        {
            new("names", null,
            [
                new TextRow("name", Loc.S("Lens_System_Name"), sys.Name, v => Edit(Loc.S("Lens_Undo_Rename"), s => s with { Name = v })),
                new TextRow("notes", Loc.S("Lens_System_Notes"), sys.Notes, v => Edit(Loc.S("Lens_Undo_ChangeNotes"), s => s with { Notes = v })) { Prompt = Loc.S("Lens_Field_None") },
            ]),
        };

        var aperture = new List<FormRow>
        {
            new ChoiceRow("aperture.kind", Loc.S("Lens_System_Type"), ApertureKindExtensions.All.Select(k => (Wire.Name(k), LensText.Aperture(k))).ToList(), Wire.Name(sys.Aperture.Kind),
                v => { if (Wire.Parse<ApertureKind>(v) is { } k) SetApertureKind(k); }),
        };
        if (sys.Aperture.Value is { } av)
        {
            aperture.Add(new NumberRow("aperture.value", Loc.S("Lens_Field_Value"), av, v => Edit(Loc.S("Lens_Undo_ChangeAperture"), s => s with { Aperture = ApertureSpec.Of(s.Aperture.Kind, v) }))
            {
                Unit = sys.Aperture.Kind.Unit(), Parse = InspectorEdits.Positive,
                Variable = Variable(sys, new ParamPath.ApertureValue()), Solve = SolveTitle(sys, new ParamPath.ApertureValue()),
            });
        }
        list.Add(new FormSection("aperture", Loc.S("Lens_Section_Aperture"), aperture));

        if (FirstOrder(sys) is { } fo) list.Add(fo);
        list.AddRange(Fields(sys));
        list.Add(Wavelengths(sys));

        list.Add(new FormSection("environment", Loc.S("Lens_System_Environment"),
        [
            new NumberRow("env.temperature", Loc.S("Lens_System_Temperature"), sys.Environment.TemperatureC,
                v => Edit(Loc.S("Lens_Undo_ChangeTemperature"), s => s with { Environment = s.Environment with { TemperatureC = v } })) { Unit = "°C" },
            new NumberRow("env.pressure", Loc.S("Lens_System_Pressure"), sys.Environment.PressureAtm,
                v => Edit(Loc.S("Lens_Undo_ChangePressure"), s => s with { Environment = s.Environment with { PressureAtm = v } })) { Unit = "atm", Parse = InspectorEdits.Positive },
        ]));

        list.Add(new FormSection("aiming", Loc.S("Lens_System_RayAiming"),
        [
            new ToggleRow("aiming.real", Loc.S("Lens_System_RealRayAiming"), sys.RayAiming is RayAiming.Real,
                on => Edit(Loc.S("Lens_Undo_ChangeRayAiming"), s => s with { RayAiming = on ? new RayAiming.Real() : new RayAiming.Off() })),
        ])
        {
            Footer = Loc.S("Lens_System_RayAimingFooter"),
        });

        if (sys.Configurations.Count > 0)
        {
            list.Add(new FormSection("configurations", Loc.S("Lens_System_Configurations"), sys.Configurations.Select((c, k) =>
                (FormRow)new InfoRow($"config.{k}", c.Name.Length == 0 ? Loc.F("Lens_System_ConfigurationN", k + 1) : c.Name,
                    Loc.F(c.Overrides.Count == 1 ? "Lens_System_Overrides_One" : "Lens_System_Overrides_Many", c.Overrides.Count))).ToList()));
        }

        var catalogs = _catalogs.Select(c => (FormRow)new InfoRow("catalog." + c.Name, c.IsBuiltin ? Loc.S("Lens_System_CatalogBuiltin") : c.Name, Loc.F("Lens_System_Glasses", c.GlassCount))
        {
            Caption = InspectorEdits.CatalogOrigin(c.IsBuiltin, c.Library, c.InProject),
            Help = c.Path ?? "",
        }).ToList();
        catalogs.AddRange(Ws.CatalogWarnings.Select((w, k) => (FormRow)new NoteRow($"catalog.warning.{k}", w) { Warning = true }));
        catalogs.Add(new ButtonRow("catalog.manage", Loc.S("Lens_System_ManageCatalogs"), () =>
            Windows.GlassCatalogsWindow.Show(App.Windows.OfType<Windows.DesignWindow>().FirstOrDefault(w => w.Session == _session))));
        if (catalogs.Count > 0)
        {
            list.Add(new FormSection("catalogs", Loc.S("Lens_System_Catalogs"), catalogs)
            {
                Footer = Loc.S("Lens_System_CatalogsFooter"),
            });
        }
        return list;
    }

    /// <summary>The x / y first order of a design that is not rotationally symmetric.</summary>
    FormSection? FirstOrder(OpticalSystem sys)
    {
        var json = Ws.Summary.Output?.Json;
        if (!FirstOrderXY.IsAnamorphic(json, sys)) return null;
        var rows = new List<FormRow>();
        if (FirstOrderXY.Read(json) is { } p)
        {
            rows.Add(new InfoRow("fo.efl", Loc.S("Lens_FirstOrder_Efl"), InspectorFormat.Short(p.EflX) + " / " + InspectorFormat.Short(p.EflY) + " mm") { Secondary = false });
            rows.Add(new InfoRow("fo.bfl", Loc.S("Lens_FirstOrder_Bfl"), InspectorFormat.Short(p.BflX) + " / " + InspectorFormat.Short(p.BflY) + " mm") { Secondary = false });
            rows.Add(new InfoRow("fo.squeeze", Loc.S("Lens_FirstOrder_Squeeze"), FirstOrderXY.SqueezeText(p.Squeeze))
            { Secondary = false, Help = Loc.F("Lens_FirstOrder_SqueezeHelp", p.SqueezeDefinition ?? Loc.S("Lens_FirstOrder_SqueezeDefault")) });
            if (p.Symmetry is { } s) rows.Add(new InfoRow("fo.symmetry", Loc.S("Lens_FirstOrder_Symmetry"), FirstOrderXY.SymmetryTitle(s)) { Secondary = false });
        }
        else
        {
            rows.Add(new NoteRow("fo.note", Loc.S("Lens_FirstOrder_Note")));
        }
        return new FormSection("firstorder", Loc.S("Lens_FirstOrder_Title"), rows);
    }

    IEnumerable<FormSection> Fields(OpticalSystem sys)
    {
        var kind = sys.Fields.Kind;
        var unit = kind.Unit();
        var fields = sys.Fields.Fields;
        var lines = fields.Select((f, k) => new GridLine(
            new GridLead(k, (dark) => Palette.Field(k, dark), false, null, Loc.F("Lens_System_FieldN", k + 1)),
            [
                new GridCell(f.X, null, v => Edit(Loc.S("Lens_Undo_ChangeField"), s => InspectorEdits.WithField(s, k, x => x with { X = v }))),
                new GridCell(f.Y, null, v => Edit(Loc.S("Lens_Undo_ChangeField"), s => InspectorEdits.WithField(s, k, x => x with { Y = v }))),
                new GridCell(f.Weight, null, v => Edit(Loc.S("Lens_Undo_ChangeFieldWeight"), s => InspectorEdits.WithField(s, k, x => x with { Weight = v }))),
            ],
            fields.Count > 1 ? () => Edit(Loc.S("Lens_System_RemoveField"), s => s.RemoveField(k)) : null, Loc.S("Lens_System_RemoveField"))).ToList();
        var rows = new List<FormRow>
        {
            new ChoiceRow("fields.kind", Loc.S("Lens_System_Type"), FieldKindExtensions.All.Select(k => (Wire.Name(k), LensText.Field(k))).ToList(), Wire.Name(kind),
                v => { if (Wire.Parse<FieldKind>(v) is { } k) SetFieldKind(k); }),
            new CustomRow("fields.grid", () => new NumberGrid(), e => ((NumberGrid)e).Show([$"X ({unit})", $"Y ({unit})", Loc.S("Lens_System_Weight")], lines, _session.ShowToast)),
            new ButtonRow("fields.add", Loc.S("Lens_System_AddField"), () => Edit(Loc.S("Lens_System_AddField"), InspectorEdits.AddField)) { Glyph = "\uE710" },
        };
        // Vignetting factors (CONVENTIONS §3), shown when used or asked for.
        var used = fields.Any(f => f.Vignetting != new Vignetting());
        if (!used && !_showVignetting)
            rows.Add(new ButtonRow("vignetting.show", Loc.S("Lens_System_ShowVignetting"), () => { _showVignetting = true; Refresh(); }));
        yield return new FormSection("fields", Loc.S("Lens_System_Fields"), rows);
        if (used || _showVignetting)
        {
            var vlines = fields.Select((f, k) => new GridLine(
                new GridLead(k, (dark) => Palette.Field(k, dark), false, null, Loc.F("Lens_System_FieldN", k + 1)),
                [
                    new GridCell(f.Vignetting.Vdx, null, v => Edit(Loc.S("Lens_Undo_ChangeVignetting"), s => InspectorEdits.WithField(s, k, x => x with { Vignetting = x.Vignetting with { Vdx = v } }))),
                    new GridCell(f.Vignetting.Vdy, null, v => Edit(Loc.S("Lens_Undo_ChangeVignetting"), s => InspectorEdits.WithField(s, k, x => x with { Vignetting = x.Vignetting with { Vdy = v } }))),
                    new GridCell(f.Vignetting.Vcx, null, v => Edit(Loc.S("Lens_Undo_ChangeVignetting"), s => InspectorEdits.WithField(s, k, x => x with { Vignetting = x.Vignetting with { Vcx = v } }))),
                    new GridCell(f.Vignetting.Vcy, null, v => Edit(Loc.S("Lens_Undo_ChangeVignetting"), s => InspectorEdits.WithField(s, k, x => x with { Vignetting = x.Vignetting with { Vcy = v } }))),
                ], null, "")).ToList();
            var vrows = new List<FormRow>
            {
                new CustomRow("vignetting.grid", () => new NumberGrid(), e => ((NumberGrid)e).Show(["VDX", "VDY", "VCX", "VCY"], vlines, _session.ShowToast)),
            };
            if (!used) vrows.Add(new ButtonRow("vignetting.hide", Loc.S("Lens_System_HideVignetting"), () => { _showVignetting = false; Refresh(); }));
            yield return new FormSection("vignetting", Loc.S("Lens_System_Vignetting"), vrows) { Footer = Loc.S("Lens_System_VignettingFooter") };
        }
    }

    FormSection Wavelengths(OpticalSystem sys)
    {
        var ws = sys.Wavelengths;
        var lines = ws.Wavelengths.Select((w, k) => new GridLine(
            new GridLead(k, dark => Palette.Wavelength(w.Um, dark), ws.Primary == k,
                ws.Primary == k ? null : () => Edit(Loc.S("Lens_Undo_ChangePrimaryWavelength"), s => s with { Wavelengths = s.Wavelengths with { Primary = k } }),
                ws.Primary == k ? Loc.S("Lens_System_PrimaryWavelength") : Loc.S("Lens_System_MakePrimary")),
            [
                new GridCell(w.Um, InspectorEdits.Positive, v => Edit(Loc.S("Lens_Undo_ChangeWavelength"), s => InspectorEdits.WithWavelength(s, k, x => x with { Um = v }))),
                new GridCell(w.Weight, null, v => Edit(Loc.S("Lens_Undo_ChangeWavelengthWeight"), s => InspectorEdits.WithWavelength(s, k, x => x with { Weight = v }))),
            ],
            ws.Wavelengths.Count > 1 ? () => Edit(Loc.S("Lens_System_RemoveWavelength"), s => s.RemoveWavelength(k)) : null, Loc.S("Lens_System_RemoveWavelength"))).ToList();
        var presets = InspectorEdits.WavelengthPresets.Select((p, k) => (p.Name, (Action)(() => Edit(Loc.S("Lens_Undo_ApplyWavelengthPreset"), s => InspectorEdits.ApplyPreset(s, k))))).ToList();
        return new FormSection("wavelengths", Loc.S("Lens_System_Wavelengths"),
        [
            new CustomRow("wavelengths.grid", () => new NumberGrid(), e => ((NumberGrid)e).Show(["λ (µm)", Loc.S("Lens_System_Weight")], lines, _session.ShowToast)),
            new ButtonRow("wavelengths.add", Loc.S("Lens_System_AddWavelength"), () => Edit(Loc.S("Lens_System_AddWavelength"), InspectorEdits.AddWavelength))
            {
                Glyph = "\uE710",
                Right = new MenuRow("wavelengths.presets", Loc.S("Lens_System_Presets"), presets) { Row = false },
            },
        ]);
    }
}

/// <summary>The leading cell of a grid line: a colour (a field's, a wavelength's), filled when selected (the primary).</summary>
sealed record GridLead(int Index, Func<bool, Color> Color, bool Selected, Action? Click, string Name);
sealed record GridCell(double Value, Func<string, Parsed<double>>? Parse, Action<double> Commit);
sealed record GridLine(GridLead Lead, IReadOnlyList<GridCell> Cells, Action? Remove, string RemoveName);

/// <summary>
/// A compact grid of numbers (fields, wavelengths, vignetting): a coloured lead, one number field per column,
/// and a remove button. Fields follow the one text-field rule; the grid is rebuilt only when its shape changes.
/// </summary>
sealed partial class NumberGrid : Grid
{
    readonly List<NumberEntry[]> _boxes = [];
    readonly List<TextBlock> _headers = [];
    readonly List<(Ellipse Dot, TextBlock Text, Button Hit, Button Remove)> _leads = [];
    IReadOnlyList<GridLine> _lines = [];
    (int, int) _shape = (-1, -1);

    public NumberGrid()
    {
        Padding = new Thickness(8, 6, 6, 6);
        ColumnSpacing = 4;
        RowSpacing = 2;
        ActualThemeChanged += (_, _) => Paint();
    }

    public void Show(IReadOnlyList<string> headers, IReadOnlyList<GridLine> lines, Action<string> reject)
    {
        _lines = lines;
        var shape = (headers.Count, lines.Count);
        if (shape != _shape) Build(headers, lines.Count);
        _shape = shape;
        for (var c = 0; c < headers.Count; c++) _headers[c].Text = headers[c];
        for (var r = 0; r < lines.Count; r++)
        {
            var line = lines[r];
            for (var c = 0; c < headers.Count; c++)
            {
                var cell = line.Cells[c];
                _boxes[r][c].Set(cell.Value, null, cell.Parse, cell.Commit, reject);
                NumberRow.AutomationName(_boxes[r][c].Box, Loc.F("Lens_System_GridCellName", line.Lead.Name, headers[c]));
            }
            var (_, text, hit, remove) = _leads[r];
            text.Text = (line.Lead.Index + 1).ToString(System.Globalization.CultureInfo.InvariantCulture);
            ToolTipService.SetToolTip(hit, line.Lead.Name);
            NumberRow.AutomationName(hit, line.Lead.Name);
            hit.IsHitTestVisible = line.Lead.Click is not null;
            remove.Visibility = line.RemoveName.Length == 0 ? Visibility.Collapsed : Visibility.Visible;
            remove.IsEnabled = line.Remove is not null;
            ToolTipService.SetToolTip(remove, line.RemoveName);
            NumberRow.AutomationName(remove, line.RemoveName);
        }
        Paint();
    }

    void Build(IReadOnlyList<string> headers, int count)
    {
        Children.Clear();
        ColumnDefinitions.Clear();
        RowDefinitions.Clear();
        _boxes.Clear();
        _leads.Clear();
        _headers.Clear();
        ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(34) });
        foreach (var _ in headers) ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        for (var r = 0; r <= count; r++) RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        for (var c = 0; c < headers.Count; c++)
        {
            var h = new TextBlock { Text = headers[c], Style = InspectorStyles.Get("InspectorCaption"), HorizontalAlignment = HorizontalAlignment.Right, Margin = new Thickness(0, 0, 7, 2) };
            SetColumn(h, c + 1);
            Children.Add(h);
            _headers.Add(h);
        }
        for (var r = 0; r < count; r++)
        {
            var row = r;
            var dot = new Ellipse { Width = 9, Height = 9, StrokeThickness = 1.5, VerticalAlignment = VerticalAlignment.Center };
            var text = new TextBlock { Style = InspectorStyles.Get("InspectorSecondary"), FontSize = 12 };
            var lead = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 5 };
            lead.Children.Add(dot);
            lead.Children.Add(text);
            var hit = new Button { Content = lead, Background = new SolidColorBrush(Microsoft.UI.Colors.Transparent), BorderThickness = new Thickness(0), Padding = new Thickness(2, 2, 2, 2), MinWidth = 0, VerticalAlignment = VerticalAlignment.Center };
            hit.Click += (_, _) => { if (row < _lines.Count) _lines[row].Lead.Click?.Invoke(); };
            SetRow(hit, r + 1);
            Children.Add(hit);
            var boxes = new NumberEntry[headers.Count];
            for (var c = 0; c < headers.Count; c++)
            {
                var b = new NumberEntry(InspectorStyles.Get("InspectorField"));
                b.Box.HorizontalAlignment = HorizontalAlignment.Stretch;
                SetRow(b.Box, r + 1);
                SetColumn(b.Box, c + 1);
                Children.Add(b.Box);
                boxes[c] = b;
            }
            _boxes.Add(boxes);
            var remove = new Button { Style = InspectorStyles.Get("InspectorIconButton"), Content = "\uE738" };
            remove.Click += (_, _) => { if (row < _lines.Count) _lines[row].Remove?.Invoke(); };
            SetRow(remove, r + 1);
            SetColumn(remove, headers.Count + 1);
            Children.Add(remove);
            _leads.Add((dot, text, hit, remove));
        }
    }

    void Paint()
    {
        var dark = ActualTheme == ElementTheme.Dark;
        for (var r = 0; r < _leads.Count && r < _lines.Count; r++)
        {
            var color = new SolidColorBrush(_lines[r].Lead.Color(dark));
            var dot = _leads[r].Dot;
            dot.Stroke = color;
            // A wavelength list marks its primary by a filled dot (a ring for the others); a field list fills them all.
            var hollow = _lines.Any(l => l.Lead.Selected) && !_lines[r].Lead.Selected;
            dot.Fill = hollow ? null : color;
        }
    }
}
