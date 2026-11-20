using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.Model;

namespace MKLight.App.Views.Inspector;

/// <summary>
/// The inspector's Surface page (the Mac app's <c>SurfaceInspector</c>): the selected surface, built from its
/// type's descriptor (<see cref="SurfaceTypes"/>): title with the stop badge, the type menu and summary, a preview
/// of the shape, the type's parameter sections, spacing and medium, aperture, placement, coating and solves.
/// Every edit is one undoable step named as on the Mac; every continuous value can be made a variable.
/// The form is updated in place for another surface (its fields follow their values).
/// </summary>
sealed partial class SurfacePage : UserControl
{
    readonly DesignSession _session;
    readonly FormHost _form;
    bool _pending, _dirty = true;

    // Per-surface view state (reset for another surface).
    SurfaceID? _stateFor;
    readonly Dictionary<string, HashSet<int>> _revealed = [];
    readonly HashSet<string> _showAll = [];
    bool _placementExpanded, _meltExpanded;

    public SurfacePage(DesignSession session)
    {
        _session = session;
        _form = new FormHost(new FormContext(session.ShowToast));
        Content = _form;
        session.Document.Changed += Schedule;
        session.SelectionChanged += Schedule;
        session.EngineDataChanged += Schedule;
        RegisterPropertyChangedCallback(VisibilityProperty, (_, _) => { if (Visibility == Visibility.Visible && _dirty) Schedule(); });
        Loaded += (_, _) => Schedule();
    }

    public FormHost Form => _form;

    DesignDocument Doc => _session.Document;

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

    /// <summary>Builds the form for the current state (or updates it in place).</summary>
    public void Refresh()
    {
        _dirty = false;
        var sys = Doc.System;
        if (_session.SelectedSurface is not int index || !sys.Surfaces.IsValidIndex(index))
        {
            _stateFor = null;
            _form.ShowEmpty(Empty());
            return;
        }
        var surface = sys.Surfaces[index];
        if (_stateFor != surface.Id)
        {
            _stateFor = surface.Id;
            _revealed.Clear();
            _showAll.Clear();
            _placementExpanded = _meltExpanded = false;
        }
        _form.Show(Sections(sys, index, surface));
    }

    static FrameworkElement Empty()
    {
        var p = new StackPanel { Spacing = 6, HorizontalAlignment = HorizontalAlignment.Center, Margin = new Thickness(16, 120, 16, 0) };
        p.Children.Add(new FontIcon { Glyph = "\uE8B0", FontSize = 28, Margin = new Thickness(0, 0, 0, 6), Opacity = 0.6 });
        p.Children.Add(new TextBlock { Text = "No Surface Selected", Style = InspectorStyles.Get("InspectorLabel"), FontWeight = Microsoft.UI.Text.FontWeights.SemiBold, HorizontalAlignment = HorizontalAlignment.Center });
        p.Children.Add(new TextBlock { Text = "Select a row in the lens data to edit its surface.", Style = InspectorStyles.Get("InspectorCaption"), TextAlignment = TextAlignment.Center, HorizontalAlignment = HorizontalAlignment.Center });
        return p;
    }

    // ---- Edits ----

    void Edit(string name, SurfaceID id, Func<Surface, Surface?> change) => Doc.Edit(name, s => InspectorEdits.Surface(s, id, change));

    void SetShape(string name, SurfaceID id, Func<SurfaceShape, SurfaceShape?> change) =>
        Edit(name, id, sf => change(sf.Shape) is { } shape ? sf with { Shape = shape } : null);

    void SetType(SurfaceID id, SurfaceTypeChoice choice)
    {
        var edit = new LensEdit.SetType(id, choice);
        Doc.Edit(edit.ActionName, edit.Apply);
    }

    VariableSpec? Variable(OpticalSystem sys, ParamPath? path)
    {
        if (path is null) return null;
        var on = InspectorVariables.IsVariable(Doc.Section("variables"), sys, path);
        return new VariableSpec(on, sys.SolveFor(path)?.Kind.Type.Title(), v =>
        {
            if (InspectorVariables.Set(Doc.Section("variables"), Doc.System, path, v) is { } section)
                Doc.EditSection("variables", section, v ? "Make Variable" : "Remove Variable");
        });
    }

    static string? SolveTitle(OpticalSystem sys, ParamPath? path) => path is null ? null : sys.SolveFor(path)?.Kind.Type.Title();

    // ---- The form ----

    IReadOnlyList<FormSection> Sections(OpticalSystem sys, int index, Surface surface)
    {
        var id = surface.Id;
        var type = SurfaceTypes.Descriptor(surface.Shape);
        var isEnd = index == 0 || index == sys.ImageIndex;
        var autoSd = _session.AutoSemiDiameters.TryGetValue(index, out var a) ? a : (double?)null;
        var semiDiameter = surface.SemiDiameter.FixedValue ?? autoSd;
        var primaryUm = sys.Wavelengths.PrimaryUm ?? 0.5875618;
        var list = new List<FormSection>();

        // Title, type, summary, preview and comment.
        var head = new List<FormRow>
        {
            new CustomRow("title", () => new SurfaceTitle(choice => SetType(_stateFor!.Value, choice)),
                e => ((SurfaceTitle)e).Update(InspectorEdits.SurfaceTitle(sys, index), index == sys.Stop && !isEnd, isEnd ? null : surface.Shape, type.Summary)),
        };
        if (!isEnd && type.Preview is not (ShapePreviewKind.IdealLens or ShapePreviewKind.CoordinateBreak))
            head.Add(new CustomRow("preview", () => new ShapePreview(), e => ((ShapePreview)e).Update(surface.Shape, semiDiameter, primaryUm)));
        head.Add(new TextRow("comment", "Comment", surface.Label, v => Edit("Change Comment", id, s => s with { Label = v })) { Prompt = "none", Owner = id });
        list.Add(new FormSection("head", null, head));

        if (type.PlacementFirst) list.Add(Placement(sys, surface));

        if (surface.Shape is DiffractiveShape doe)
        {
            list.Add(new FormSection("doe-base", "Base Shape",
            [
                new ChoiceRow("doe-base.kind", "Base", [("plane", "Plane"), ("conic", "Sphere / Conic"), ("even_asphere", "Even Asphere")],
                    Wire.Name(doe.Base.Kind is ShapeKind.Conic or ShapeKind.EvenAsphere ? doe.Base.Kind : ShapeKind.Plane),
                    v => SetShape("Change Base Shape", id, s => InspectorEdits.WithDiffractiveBase(s, Wire.Parse<ShapeKind>(v) ?? ShapeKind.Plane))),
            ]));
        }

        foreach (var section in type.Sections)
        {
            if (!section.Applies(surface.Shape)) continue;
            var rows = new List<FormRow>();
            var key = "type." + section.Title;
            foreach (var c in section.Choices)
            {
                var current = SurfaceTypes.Value(surface.Shape, c.Key)?.GetValue<string>() ?? c.Options[0].Value;
                rows.Add(new ChoiceRow(key + "." + c.Key, c.Title, c.Options.Select(o => (o.Value, o.Title)).ToList(), current,
                    v => SetShape("Change " + c.Title, id, s => SurfaceTypes.WithValue(s, c.Key, System.Text.Json.Nodes.JsonValue.Create(v)))) { Help = c.Help });
            }
            foreach (var p in section.Params)
            {
                if (SurfaceTypes.DisplayValue(p, surface.Shape) is not { } value) continue;
                var path = p.Path?.Invoke(id);
                rows.Add(new NumberRow(key + "." + p.Key, p.Title, value,
                    v => SetShape("Change " + p.Title, id, s => SurfaceTypes.WithParam(s, p, v)))
                {
                    Unit = p.Unit, Help = p.Help, Owner = id,
                    Parse = text => SurfaceTypes.StoredValue(p, text),
                    Variable = Variable(sys, path), Solve = SolveTitle(sys, path),
                });
            }
            if (section.Coefficients is { } cl && SurfaceTypes.HasList(cl, surface.Shape)) rows.AddRange(Coefficients(sys, surface, key, cl));
            if (rows.Count == 0) continue;
            list.Add(new FormSection(key, section.Title, rows) { Footer = section.Footer.Length > 0 ? section.Footer : null });
        }

        if (type.OrderEfficiency && surface.Shape is DiffractiveShape d) list.Add(Efficiency(sys, surface, d.Phase));

        if (surface.Shape.Kind == ShapeKind.Biconic)
        {
            list.Add(new FormSection("cylinder", null,
            [
                new MenuRow("cylinder.make", "Make Cylinder",
                [
                    ("Power in x (plane in y)", () => SetType(id, new(ShapeKind.Biconic, CylinderAxis.X))),
                    ("Power in y (plane in x)", () => SetType(id, new(ShapeKind.Biconic, CylinderAxis.Y))),
                ]),
            ]) { Footer = "A cylinder is a biconic with one section plane." });
        }

        if (index != sys.ImageIndex) list.Add(Medium(sys, index, surface));
        if (!surface.IsCoordinateBreak) list.Add(Aperture(sys, surface, autoSd));
        if (!type.PlacementFirst && !isEnd) list.Add(Placement(sys, surface));
        if (!isEnd && !surface.IsCoordinateBreak)
        {
            var options = new List<(string, string)> { ("", "Uncoated") };
            options.AddRange(sys.Coatings.Keys.OrderBy(k => k, StringComparer.Ordinal).Select(k => (k, k)));
            list.Add(new FormSection("coating", "Coating",
            [
                new ChoiceRow("coating.name", "Coating", options, surface.Coating ?? "",
                    v => Edit("Change Coating", id, s => s with { Coating = v.Length == 0 ? null : v })),
            ]));
        }
        if (Solves(sys, index, surface) is { } solves) list.Add(solves);
        return list;
    }

    IEnumerable<FormRow> Coefficients(OpticalSystem sys, Surface surface, string section, CoefficientList cl)
    {
        var id = surface.Id;
        var shape = surface.Shape;
        var key = section + "." + cl.Key;
        var values = SurfaceTypes.Coefficients(cl, shape);
        var limit = cl.MaxCount(shape);
        var revealed = _revealed.TryGetValue(key, out var r) ? r : _revealed[key] = [];
        var showAll = _showAll.Contains(key);

        void Write(IReadOnlyList<double> a, string name) => SetShape(name, id, s => SurfaceTypes.WithCoefficients(s, cl, a));

        foreach (var k in SurfaceTypes.VisibleTerms(values, cl.Sparse, showAll, revealed))
        {
            var term = cl.Term(k, shape);
            var removable = k == values.Count - 1 || cl.Sparse;
            var path = cl.Path?.Invoke(id, k);
            var kk = k;
            yield return new NumberRow($"{key}.{k}", term, values[k], v => Write(SurfaceTypes.SetTerm(values, kk, v), "Change " + term))
            {
                Detail = cl.Meaning(k, shape), Unit = cl.Unit(k), Owner = id,
                Variable = Variable(sys, path), Solve = SolveTitle(sys, path),
                Trailing = new TrailingButton("\uE738", k == values.Count - 1 ? "Remove this term" : "Set this term to zero", () =>
                {
                    revealed.Remove(kk);
                    Write(SurfaceTypes.RemoveTerm(values, kk, cl.Sparse, revealed), "Remove " + term);
                }, removable),
            };
        }

        void Add(int k)
        {
            revealed.Add(k);
            Write(SurfaceTypes.AddTerm(values, k), "Add " + cl.Term(k, shape));
        }

        if (values.Count < limit || (cl.Sparse && values.Contains(0)))
        {
            if (cl.Sparse)
            {
                var items = Enumerable.Range(0, limit)
                    .Where(k => k >= values.Count || (values[k] == 0 && !revealed.Contains(k)))
                    .Select(k => (cl.Term(k, shape) + "   " + cl.Meaning(k, shape), (Action)(() => Add(k)))).ToList();
                yield return new MenuRow(key + ".add", "Add Term", items) { Glyph = "\uE710" };
            }
            else
            {
                var n = values.Count;
                yield return new ButtonRow(key + ".add", $"Add {cl.Term(n, shape)}  {cl.Meaning(n, shape)}", () => Add(n)) { Glyph = "\uE710" };
            }
        }
        if (cl.Sparse && values.Take(Math.Max(0, values.Count - 1)).Contains(0))
        {
            yield return new ButtonRow(key + ".zero", showAll ? "Hide Zero Terms" : "Show Zero Terms", () =>
            {
                if (!_showAll.Remove(key)) _showAll.Add(key);
                Refresh();
            }) { Help = "Show the terms that are zero" };
        }
        if (values.Count >= limit && limit > 0)
            yield return new NoteRow(key + ".limit", $"{limit} terms, the most this type allows.");
    }

    FormSection Efficiency(OpticalSystem sys, Surface surface, DiffractivePhase phase)
    {
        var id = surface.Id;
        var model = surface.DiffractionEfficiency;
        void Set(DiffractionEfficiency m) => Edit("Change Diffraction Efficiency", id, s => s with { DiffractionEfficiency = m });
        var kind = model switch { DiffractionEfficiency.Kinoform => "kinoform", DiffractionEfficiency.Multilevel => "multilevel", _ => "ideal" };
        var levels = model is DiffractionEfficiency.Multilevel ml ? ml.Levels : 8;
        var rows = new List<FormRow>
        {
            new ChoiceRow("eff.model", "Model", [("ideal", "Ideal"), ("kinoform", "Kinoform"), ("multilevel", "Multilevel")], kind,
                v => Set(v switch { "kinoform" => new DiffractionEfficiency.Kinoform(), "multilevel" => new DiffractionEfficiency.Multilevel(levels), _ => new DiffractionEfficiency.Ideal() }))
            {
                Help = "Ideal: every photon in the trace order. Kinoform: a continuous blazed relief. Multilevel: a staircase of N etch levels.",
            },
        };
        if (model is DiffractionEfficiency.Multilevel m)
        {
            rows.Add(new NumberRow("eff.levels", "Levels", m.Levels, v => Set(new DiffractionEfficiency.Multilevel((int)Math.Clamp(v, 1, 256))))
            {
                Owner = id, Help = "Number of etch levels of the staircase (2 binary, 4, 8, 16 …).",
                Parse = Levels,
            });
        }
        rows.Add(new InfoRow("eff.design", "Trace order at λ₀", InspectorFormat.Percent(InspectorEdits.Efficiency(model, phase, phase.DesignWavelengthUm)))
        {
            Secondary = false,
            Help = $"Efficiency of the trace order ({phase.TraceOrder}) at the design wavelength {InspectorFormat.Short(phase.DesignWavelengthUm)} µm.",
        });
        var wavelengths = sys.Wavelengths.Wavelengths.Select(w => w.Um).ToList();
        if (model is not DiffractionEfficiency.Ideal && wavelengths.Count > 0)
            rows.Add(new CustomRow("eff.bars", () => new EfficiencyBars(), e => ((EfficiencyBars)e).Update(wavelengths.Select(um => (um, InspectorEdits.Efficiency(model, phase, um))).ToList())));
        return new FormSection("eff", "Diffraction Efficiency", rows)
        {
            Footer = model is DiffractionEfficiency.Ideal ? null
                : "System wavelengths estimated with blaze m₀λ₀/λ, ignoring the relief material's dispersion. The engine traces the exact model.",
        };
    }

    static readonly ShapeParam Whole = new("", "", "", ParamDisplay.Integer);

    static Parsed<double> Levels(string text)
    {
        var p = SurfaceTypes.StoredValue(Whole, text);
        return p.Ok && p.Value is < 1 or > 256 ? Parsed<double>.Fail("Levels must be 1 … 256") : p;
    }

    FormSection Medium(OpticalSystem sys, int index, Surface surface)
    {
        var id = surface.Id;
        var rows = new List<FormRow>
        {
            new NumberRow("medium.thickness", "Thickness", surface.Thickness, v => Edit("Change Thickness", id, s => s with { Thickness = v }))
            {
                Unit = "mm", Help = "Distance to the next vertex.", Owner = id,
                Parse = t => LensValueParser.Thickness(t, allowInfinite: index == 0),
                Variable = Variable(sys, new ParamPath.Thickness(id)), Solve = SolveTitle(sys, new ParamPath.Thickness(id)),
            },
        };
        if (!surface.IsCoordinateBreak && index != 0)
        {
            rows.Add(new ChoiceRow("medium.interaction", "Interaction", [("refract", "Refract"), ("reflect", "Mirror"), ("dummy", "Dummy")],
                Wire.Name(surface.Interaction), v => Edit("Change Interaction", id, s => s with { Interaction = Wire.Parse<Interaction>(v) ?? Interaction.Refract })));
        }
        if (!surface.IsCoordinateBreak && surface.Interaction == Interaction.Refract)
        {
            rows.Add(new TextRow("medium.material", "Material", surface.Material.IsAir ? "AIR" : surface.Material.DisplayName, text =>
            {
                var parsed = LensValueParser.Material(text);
                if (!parsed.Ok) { _session.ShowToast(parsed.Error!); Refresh(); return; }
                Doc.Edit("Change Material", new LensEdit.SetMaterial(id, parsed.Value).Apply);
            }) { Prompt = "AIR", Owner = id, Suggestions = _session.MaterialNames });
            if (surface.Material is MaterialRef.CatalogGlass glass) rows.AddRange(MeltOffset(sys, surface, glass.Offset));
            if (surface.Material is MaterialRef.ModelGlass mg)
            {
                rows.Add(new NumberRow("medium.nd", "n_d", mg.Nd, v => Edit("Change n_d", id, s => s with { Material = mg with { Nd = v } }))
                { Owner = id, Variable = Variable(sys, new ParamPath.ModelNd(id)), Solve = SolveTitle(sys, new ParamPath.ModelNd(id)) });
                rows.Add(new NumberRow("medium.vd", "V_d", mg.Vd, v => Edit("Change V_d", id, s => s with { Material = mg with { Vd = v } }))
                { Owner = id, Variable = Variable(sys, new ParamPath.ModelVd(id)), Solve = SolveTitle(sys, new ParamPath.ModelVd(id)) });
            }
        }
        return new FormSection("medium", "Spacing and Medium", rows);
    }

    /// <summary>A catalogue glass's melt offset Δn_d and ΔV_d, hidden behind "Add Melt Offset…" while zero.</summary>
    IEnumerable<FormRow> MeltOffset(OpticalSystem sys, Surface surface, MaterialOffset offset)
    {
        var id = surface.Id;
        ParamPath dn = new ParamPath.MaterialDnD(id), dv = new ParamPath.MaterialDvD(id);
        var vars = Doc.Section("variables");
        var showing = _meltExpanded || !offset.IsZero || InspectorVariables.IsVariable(vars, sys, dn) || InspectorVariables.IsVariable(vars, sys, dv);
        void Set(string name, Func<MaterialOffset, MaterialOffset> change) => Edit(name, id, s => InspectorEdits.WithOffset(s, change));
        if (!showing)
        {
            yield return new ButtonRow("melt.add", "Add Melt Offset…", () => { _meltExpanded = true; Refresh(); })
            { Help = "Give the catalog glass a measured melt's Δn_d and ΔV_d" };
            yield break;
        }
        yield return new NumberRow("melt.dn", "Δn_d", offset.DnD, v => Set("Change Δn_d", o => o with { DnD = v }))
        {
            Detail = "melt", Owner = id, Parse = InspectorEdits.OffsetDnD, Variable = Variable(sys, dn), Solve = SolveTitle(sys, dn),
            Help = $"The melt's index offset from the catalog n_d (within ±{InspectorFormat.Short(MaterialOffset.MaxDnD)}); the dispersion keeps its shape.",
        };
        yield return new NumberRow("melt.dv", "ΔV_d", offset.DvD, v => Set("Change ΔV_d", o => o with { DvD = v }))
        {
            Detail = "melt", Owner = id, Parse = InspectorEdits.OffsetDvD, Variable = Variable(sys, dv), Solve = SolveTitle(sys, dv),
            Help = $"The melt's Abbe-number offset from the catalog V_d (within ±{InspectorFormat.Short(MaterialOffset.MaxDvD)}).",
        };
        if (!offset.IsZero)
            yield return new ButtonRow("melt.remove", "Remove Melt Offset", () => { _meltExpanded = false; Set("Remove Melt Offset", _ => MaterialOffset.Zero); })
            { Help = "Back to the catalog glass" };
    }

    FormSection Aperture(OpticalSystem sys, Surface surface, double? autoValue)
    {
        var id = surface.Id;
        var rows = new List<FormRow>
        {
            new ChoiceRow("aperture.mode", "Semi-diameter", [("auto", "Automatic"), ("fixed", "Fixed")],
                surface.SemiDiameter is SemiDiameter.Auto ? "auto" : "fixed",
                v => Edit("Change Semi-Diameter", id, s => s with { SemiDiameter = InspectorEdits.SemiDiameterMode(s, v == "auto", autoValue) })),
        };
        if (surface.SemiDiameter is SemiDiameter.Fixed f)
        {
            rows.Add(new NumberRow("aperture.value", "Value", f.Value, v => Edit("Change Semi-Diameter", id, s => s with { SemiDiameter = new SemiDiameter.Fixed(v) }))
            {
                Unit = "mm", Owner = id, Parse = LensValueParser.PositiveLength,
                Variable = Variable(sys, new ParamPath.SemiDiameter(id)), Solve = SolveTitle(sys, new ParamPath.SemiDiameter(id)),
            });
        }
        else if (autoValue is { } av)
        {
            rows.Add(new InfoRow("aperture.computed", "Computed", InspectorFormat.WithUnit(av, "mm")));
        }
        rows.Add(new ChoiceRow("aperture.clip", "Clipping",
            [("None", "Semi-diameter"), ("Circular", "Circular"), ("Annular", "Annular"), ("Rectangular", "Rectangular"), ("Elliptical", "Elliptical")],
            InspectorEdits.Clip(surface.Aperture).ToString(),
            v => Edit("Change Aperture", id, s => s with { Aperture = InspectorEdits.MakeClip(Enum.Parse<ClipShape>(v), s.SemiDiameter.FixedValue, autoValue) })));
        void Set(ApertureShape a) => Edit("Change Aperture", id, s => s with { Aperture = a });
        NumberRow Length(string key, string title, double value, Func<double, ApertureShape> make) =>
            new("aperture." + key, title, value, v => Set(make(v))) { Unit = "mm", Owner = id, Parse = LensValueParser.PositiveLength };
        switch (surface.Aperture)
        {
            case ApertureShape.Circular c:
                rows.Add(Length("radius", "Radius", c.Radius, v => new ApertureShape.Circular(v)));
                break;
            case ApertureShape.Annular an:
                rows.Add(Length("inner", "Inner radius", an.InnerRadius, v => an with { InnerRadius = v }));
                rows.Add(Length("outer", "Outer radius", an.OuterRadius, v => an with { OuterRadius = v }));
                break;
            case ApertureShape.Rectangular re:
                rows.Add(Length("halfx", "Half width x", re.HalfX, v => re with { HalfX = v }));
                rows.Add(Length("halfy", "Half width y", re.HalfY, v => re with { HalfY = v }));
                break;
            case ApertureShape.Elliptical el:
                rows.Add(Length("semix", "Semi-axis x", el.SemiX, v => el with { SemiX = v }));
                rows.Add(Length("semiy", "Semi-axis y", el.SemiY, v => el with { SemiY = v }));
                break;
        }
        return new FormSection("aperture", "Aperture", rows);
    }

    /// <summary>Decenters and tilts before the surface.</summary>
    FormSection Placement(OpticalSystem sys, Surface surface)
    {
        var id = surface.Id;
        var p = surface.Placement ?? new Placement();
        void Set(Func<Placement, Placement> f) => Edit("Change Placement", id, s => InspectorEdits.WithPlacement(s, f));
        if (!(_placementExpanded || surface.IsCoordinateBreak || !p.IsIdentity))
            return new FormSection("placement", "Placement", [new ButtonRow("placement.add", "Add Decenter or Tilt…", () => { _placementExpanded = true; Refresh(); })]);
        NumberRow Row(string key, string title, double value, string unit, ParamPath path, Func<Placement, double, Placement> with) =>
            new("placement." + key, title, value, v => Set(q => with(q, v))) { Unit = unit, Owner = id, Variable = Variable(sys, path), Solve = SolveTitle(sys, path) };
        return new FormSection("placement", "Placement",
        [
            Row("dx", "Decenter X", p.DecenterX, "mm", new ParamPath.DecenterX(id), (q, v) => q with { DecenterX = v }),
            Row("dy", "Decenter Y", p.DecenterY, "mm", new ParamPath.DecenterY(id), (q, v) => q with { DecenterY = v }),
            Row("tx", "Tilt X", p.TiltX, "°", new ParamPath.TiltX(id), (q, v) => q with { TiltX = v }),
            Row("ty", "Tilt Y", p.TiltY, "°", new ParamPath.TiltY(id), (q, v) => q with { TiltY = v }),
            Row("tz", "Tilt Z", p.TiltZ, "°", new ParamPath.TiltZ(id), (q, v) => q with { TiltZ = v }),
            new ChoiceRow("placement.order", "Order", [("decenter_then_tilt", "Decenter, then tilt"), ("tilt_then_decenter", "Tilt, then decenter")],
                Wire.Name(p.Order), v => Set(q => q with { Order = Wire.Parse<TransformOrder>(v) ?? TransformOrder.DecenterThenTilt })),
            new ToggleRow("placement.restore", "Return after surface", p.RestoreAfter, v => Set(q => q with { RestoreAfter = v })),
        ]);
    }

    /// <summary>Solves on this surface's parameters: list, remove, and the common ones to add.</summary>
    FormSection? Solves(OpticalSystem sys, int index, Surface surface)
    {
        var id = surface.Id;
        var solves = sys.Solves.Select((s, k) => (s, k)).Where(x => x.s.Target.SurfaceId == id).ToList();
        var canAdd = index > 0 && index < sys.ImageIndex;
        if (solves.Count == 0 && !canAdd) return null;
        var rows = new List<FormRow>();
        foreach (var (s, k) in solves)
        {
            var value = s.Kind.Type.ValueLabel() is { } label && s.Kind.MainValue is { } v ? InspectorFormat.WithUnit(v, label.Unit) : "";
            var target = s.Target;
            rows.Add(new InfoRow($"solve.{k}", s.Kind.Type.Title(), value)
            {
                Caption = InspectorEdits.TargetLabel(target),
                Trailing = new TrailingButton("\uE738", "Remove", () => Doc.Edit("Remove Solve", sys2 =>
                    sys2 with { Solves = sys2.Solves.RemoveAll(x => x.Target == target) })),
            });
        }
        if (canAdd)
            rows.Add(new MenuRow("solve.add", "Add Solve",
                InspectorEdits.SolvePresets(surface).Select(p => (p.Title, (Action)(() => Doc.Edit("Add Solve", s => InspectorEdits.AddSolve(s, p.Target, p.Kind))))).ToList()));
        return new FormSection("solves", "Solves", rows);
    }
}

/// <summary>The page's title row: "Surface n", the stop badge, the type menu and the type's summary.</summary>
sealed partial class SurfaceTitle : StackPanel
{
    readonly TextBlock _title = new();
    readonly Border _stop;
    readonly DropDownButton _type;
    readonly TextBlock _summary = new();
    readonly Action<SurfaceTypeChoice> _choose;
    SurfaceTypeChoice? _current;

    public SurfaceTitle(Action<SurfaceTypeChoice> choose)
    {
        _choose = choose;
        Padding = new Thickness(12, 10, 8, 10);
        Spacing = 4;
        _title.Style = InspectorStyles.Get("InspectorTitle");
        var stopText = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 4 };
        stopText.Children.Add(new FontIcon { Glyph = "\uE91F", FontSize = 10 });
        stopText.Children.Add(new TextBlock { Text = "Stop", FontSize = 11, FontWeight = Microsoft.UI.Text.FontWeights.SemiBold });
        _stop = new Border { CornerRadius = new CornerRadius(9), Padding = new Thickness(7, 2, 8, 3), Child = stopText, VerticalAlignment = VerticalAlignment.Center, Margin = new Thickness(8, 2, 0, 0) };
        _type = new DropDownButton { FontSize = 13, Padding = new Thickness(8, 3, 6, 4), VerticalAlignment = VerticalAlignment.Center, HorizontalAlignment = HorizontalAlignment.Right };
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(_type, "Surface type");
        var g = new Grid();
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        g.Children.Add(_title);
        Grid.SetColumn(_stop, 1);
        g.Children.Add(_stop);
        _stop.HorizontalAlignment = HorizontalAlignment.Left;
        Grid.SetColumn(_type, 2);
        g.Children.Add(_type);
        Children.Add(g);
        _summary.Style = InspectorStyles.Get("InspectorCaption");
        _summary.FontSize = 12;
        Children.Add(_summary);
        ActualThemeChanged += (_, _) => Tint();
        Tint();
    }

    void Tint()
    {
        var accent = InspectorColors.Accent(this);
        _stop.Background = new SolidColorBrush(accent.WithAlpha(0.15));
        foreach (var child in ((StackPanel)_stop.Child).Children)
        {
            if (child is FontIcon fi) fi.Foreground = ThemeLookup.Brush(this, "AccentTextFillColorPrimaryBrush");
            if (child is TextBlock tb) tb.Foreground = ThemeLookup.Brush(this, "AccentTextFillColorPrimaryBrush");
        }
    }

    public void Update(string title, bool stop, SurfaceShape? shape, string summary)
    {
        _title.Text = title;
        _stop.Visibility = stop ? Visibility.Visible : Visibility.Collapsed;
        _summary.Text = summary;
        _summary.Visibility = shape is null ? Visibility.Collapsed : Visibility.Visible;
        _type.Visibility = shape is null ? Visibility.Collapsed : Visibility.Visible;
        if (shape is null) return;
        var choice = SurfaceTypeChoice.For(shape);
        if (_current == choice) return;
        _current = choice;
        var entry = SurfaceTypeCatalog.EntryFor(choice);
        var content = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 6 };
        content.Children.Add(new FontIcon { Glyph = entry.Glyph, FontSize = 13 });
        content.Children.Add(new TextBlock { Text = entry.Title });
        _type.Content = content;
        var menu = new MenuFlyout { Placement = Microsoft.UI.Xaml.Controls.Primitives.FlyoutPlacementMode.BottomEdgeAlignedRight };
        SurfaceTypeGroup? group = null;
        foreach (var e in SurfaceTypeCatalog.All)
        {
            if (group is not null && group != e.Group) menu.Items.Add(new MenuFlyoutSeparator());
            group = e.Group;
            var item = new ToggleMenuFlyoutItem { Text = e.Title, IsChecked = e.Choice == choice, Icon = new FontIcon { Glyph = e.Glyph } };
            var c = e.Choice;
            item.Click += (_, _) => { if (c != _current) _choose(c); else item.IsChecked = true; };
            menu.Items.Add(item);
        }
        _type.Flyout = menu;
    }
}

/// <summary>The trace order's efficiency at each system wavelength, as bars in the wavelength colours.</summary>
sealed partial class EfficiencyBars : StackPanel
{
    IReadOnlyList<(double Um, double E)> _rows = [];

    public EfficiencyBars()
    {
        Spacing = 4;
        Padding = new Thickness(12, 8, 12, 10);
        ActualThemeChanged += (_, _) => Build();
    }

    public void Update(IReadOnlyList<(double Um, double E)> rows)
    {
        if (rows.SequenceEqual(_rows)) return;
        _rows = rows;
        Build();
    }

    void Build()
    {
        Children.Clear();
        var dark = ActualTheme == ElementTheme.Dark;
        foreach (var (um, e) in _rows)
        {
            var g = new Grid { ColumnSpacing = 8 };
            g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(62) });
            g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(48) });
            g.Children.Add(new TextBlock { Text = InspectorFormat.Short(um) + " µm", Style = InspectorStyles.Get("InspectorCaption"), VerticalAlignment = VerticalAlignment.Center });
            var track = new Grid { Height = 5, VerticalAlignment = VerticalAlignment.Center };
            track.Children.Add(new Border { CornerRadius = new CornerRadius(2.5), Background = ThemeLookup.Brush(this, "ControlFillColorSecondaryBrush") });
            var fill = new Border { CornerRadius = new CornerRadius(2.5), Background = new SolidColorBrush(Palette.Wavelength(um, dark)), HorizontalAlignment = HorizontalAlignment.Left };
            track.Children.Add(fill);
            track.SizeChanged += (_, s) => fill.Width = Math.Max(2, s.NewSize.Width * (double.IsFinite(e) ? Math.Clamp(e, 0, 1) : 0));
            Grid.SetColumn(track, 1);
            g.Children.Add(track);
            var pct = new TextBlock { Text = InspectorFormat.Percent(e), Style = InspectorStyles.Get("InspectorCaption"), HorizontalAlignment = HorizontalAlignment.Right, VerticalAlignment = VerticalAlignment.Center };
            Grid.SetColumn(pct, 2);
            g.Children.Add(pct);
            Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(g, $"{InspectorFormat.Short(um)} micrometers: {InspectorFormat.Percent(e)}");
            Children.Add(g);
        }
    }
}
