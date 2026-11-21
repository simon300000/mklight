using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.App.ViewModels.Jobs;
using MKLight.Model;

namespace MKLight.App.Views.Jobs;

/// <summary>
/// The design's variables with their bounds (the Mac app's <c>VariablesBoundsList</c>), edited inline: min and max
/// (empty is unbounded) in the parameter's unit, pin (equal bounds: held at its value), remove. Curvatures are
/// bounded as a radius window in mm by default (<see cref="RadiusBounds"/>), or as curvatures. Each variable takes
/// two lines, so the list fits the inspector's width: its name with the pin and remove buttons, then its value now
/// (or its problem) beside the min and max fields and the unit.
/// </summary>
public static class VariablesBoundsList
{
    /// <summary>How curvature bounds are shown (per app run, like the Mac's stored preference).</summary>
    public static CurvatureBoundsUnit CurvatureUnit { get; set; } = CurvatureBoundsUnit.Radius;

    public static StackPanel Build(DesignDocument document, Action refresh)
    {
        var system = document.System;
        var vars = DesignVariables.Of(document);
        var list = new StackPanel { Spacing = 0 };
        if (vars.Any(v => v.Param.IsCurvature))
        {
            var unit = JobUI.Segments([Loc.S("Jobs_Variables_Unit_Radius"), Loc.S("Jobs_Variables_Unit_Curvature")], CurvatureUnit == CurvatureBoundsUnit.Radius ? 0 : 1, k =>
            {
                CurvatureUnit = k == 0 ? CurvatureBoundsUnit.Radius : CurvatureBoundsUnit.Curvature;
                Support.AppSettings.CurvatureBoundsUnit = k == 0 ? "radius" : "curvature";
                refresh();
            });
            ToolTipService.SetToolTip(unit, Loc.S("Jobs_Variables_UnitHelp"));
            var help = new FontIcon { Glyph = "\uE9CE", FontSize = 13, VerticalAlignment = VerticalAlignment.Center };
            JobUI.Bind(help, "TextFillColorSecondaryBrush", b => help.Foreground = b);
            ToolTipService.SetToolTip(help, RadiusBounds.Help);
            var row = JobUI.Row(8, JobUI.Text(Loc.S("Jobs_Variables_CurvaturesAs"), "CaptionText"), unit, help);
            row.Margin = new Thickness(0, 0, 0, 8);
            list.Children.Add(row);
        }
        var header = new Grid { ColumnSpacing = 6, Margin = new Thickness(0, 0, 0, 2) };
        Columns(header);
        header.Children.Add(JobUI.Text(Loc.S("Jobs_Variables_Variable"), "CaptionText"));
        Place(header, JobUI.Text(Loc.S("Jobs_Variables_Min"), "CaptionText"), 1, HorizontalAlignment.Right);
        Place(header, JobUI.Text(Loc.S("Jobs_Variables_Max"), "CaptionText"), 2, HorizontalAlignment.Right);
        Place(header, JobUI.Text(Loc.S("Jobs_Variables_Unit"), "CaptionText"), 3, HorizontalAlignment.Left);
        list.Children.Add(header);
        for (var k = 0; k < vars.Count; k++)
        {
            if (k > 0)
            {
                var rule = new Border { Height = 1, Margin = new Thickness(0, 4, 0, 4) };
                JobUI.Bind(rule, "DividerBrush", b => rule.Background = b);
                list.Children.Add(rule);
            }
            list.Children.Add(Row(document, system, vars[k], k));
        }
        return list;
    }

    static void Columns(Grid g)
    {
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(64) });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(64) });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(34) });
    }

    static void Place(Grid g, FrameworkElement e, int column, HorizontalAlignment align)
    {
        e.HorizontalAlignment = align;
        Grid.SetColumn(e, column);
        g.Children.Add(e);
    }

    static StackPanel Row(DesignDocument document, OpticalSystem system, DesignVariable v, int index)
    {
        var path = v.Param;
        var asRadius = path.IsCurvature && CurvatureUnit == CurvatureBoundsUnit.Radius;
        var label = path.Label(system);
        var unit = asRadius ? "mm" : path.Unit(system);
        var value = system.ValueAt(path);
        var shown = asRadius ? RadiusBounds.Radius(v.Lower, v.Upper) : (v.Lower, v.Upper);
        var problem = DesignVariables.Problem(v, system, asRadius);

        var top = new Grid { ColumnSpacing = 2 };
        top.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        top.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        var name = JobUI.Row(5, JobUI.Text(label));
        if (v.IsPinned)
        {
            var pin = new FontIcon { Glyph = "\uE718", FontSize = 11, VerticalAlignment = VerticalAlignment.Center };
            JobUI.Bind(pin, "AccentTextFillColorPrimaryBrush", b => pin.Foreground = b);
            ToolTipService.SetToolTip(pin, Loc.S("Jobs_Variables_PinnedHelp"));
            name.Children.Add(pin);
        }
        name.VerticalAlignment = VerticalAlignment.Center;
        top.Children.Add(name);
        var buttons = JobUI.Row(0,
            JobUI.Icon(v.IsPinned ? "\uE77A" : "\uE718", Loc.S(v.IsPinned ? "Jobs_Variables_Unpin" : "Jobs_Variables_Pin"),
                () => DesignVariables.TogglePin(document, v), value is not null, 24),
            JobUI.Icon("\uE711", Loc.S("Jobs_Variables_Remove"), () => DesignVariables.Set(document, path, false), true, 24));
        Grid.SetColumn(buttons, 1);
        top.Children.Add(buttons);

        var bottom = new Grid { ColumnSpacing = 6 };
        Columns(bottom);
        var caption = JobUI.Text(problem ?? Caption(value, path, asRadius, shown), "CaptionText");
        caption.TextWrapping = TextWrapping.Wrap;
        caption.TextTrimming = TextTrimming.None;
        caption.VerticalAlignment = VerticalAlignment.Center;
        if (problem is not null) JobUI.Bind(caption, "WarningBrush", b => caption.Foreground = b);
        ToolTipService.SetToolTip(caption, Help(v, path, system, asRadius, unit));
        bottom.Children.Add(caption);

        void Commit(double? min, double? max)
        {
            if (!asRadius)
            {
                DesignVariables.SetBounds(document, path, min, max);
                return;
            }
            var (lo, hi, error) = RadiusBounds.Curvature(min, max);
            if (error is not null)
            {
                caption.Text = error;
                JobUI.Bind(caption, "WarningBrush", b => caption.Foreground = b);
                return;
            }
            DesignVariables.SetBounds(document, path, lo, hi);
        }
        var minField = new NumberField($"bound-min-{index}", () => shown.Item1, x => Commit(x, shown.Item2), 64, "−∞", BoundsFormat.Compact,
            allowsInfinite: asRadius, name: unit.Length == 0 ? Loc.F("Jobs_Variables_MinName", label) : Loc.F("Jobs_Variables_MinNameUnit", label, unit));
        var maxField = new NumberField($"bound-max-{index}", () => shown.Item2, x => Commit(shown.Item1, x), 64, asRadius ? "∞" : "+∞", BoundsFormat.Compact,
            allowsInfinite: asRadius, name: unit.Length == 0 ? Loc.F("Jobs_Variables_MaxName", label) : Loc.F("Jobs_Variables_MaxNameUnit", label, unit));
        Place(bottom, minField, 1, HorizontalAlignment.Stretch);
        Place(bottom, maxField, 2, HorizontalAlignment.Stretch);
        minField.Width = maxField.Width = double.NaN;
        var u = JobUI.Text(unit.Length == 0 ? "–" : unit, "CaptionText");
        u.VerticalAlignment = VerticalAlignment.Center;
        ToolTipService.SetToolTip(u, unit.Length == 0 ? Loc.S("Jobs_Variables_Dimensionless") : Loc.F("Jobs_Variables_BoundsIn", unit));
        Place(bottom, u, 3, HorizontalAlignment.Left);
        return JobUI.Column(2, top, bottom);
    }

    static string Caption(double? value, ParamPath path, bool asRadius, (double? Min, double? Max) shown)
    {
        if (value is not { } v) return "";
        if (!path.IsCurvature) return Loc.F("Jobs_Variables_Now", MeritFormat.Short(v));
        if (!asRadius) return Loc.F("Jobs_Variables_NowCurvature", BoundsFormat.Compact(v));
        var r = MeritFormat.Short(RadiusDisplay.RadiusFromCurvature(v));
        return RadiusBounds.PassesThroughFlat(shown.Min, shown.Max) ? Loc.F("Jobs_Variables_NowRadiusThroughFlat", r) : Loc.F("Jobs_Variables_NowRadius", r);
    }

    static string Help(DesignVariable v, ParamPath path, OpticalSystem system, bool asRadius, string unit)
    {
        if (system.SolveFor(path) is { } s)
            return Loc.F("Jobs_Variables_SolvedHelp", LensText.Solve(s.Kind.Type).ToLowerInvariant());
        var help = asRadius ? RadiusBounds.Help : Loc.S("Jobs_Variables_CurvatureHelp");
        if (asRadius || path.IsCurvature)
            return BoundsFormat.Counterpart(v.Lower, v.Upper, CurvatureUnit) is { } c && path.IsCurvature ? Loc.F("Jobs_Variables_BoundsAs", c, help) : help;
        return unit.Length == 0 ? Loc.S("Jobs_Variables_DimensionlessHelp") : Loc.F("Jobs_Variables_BoundsInHelp", unit);
    }
}
