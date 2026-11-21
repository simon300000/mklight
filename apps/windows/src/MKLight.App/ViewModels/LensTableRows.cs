using MKLight.App.Support;
using MKLight.Model;

namespace MKLight.App.ViewModels;

/// <summary>
/// Columns of the lens data table: the parameters every surface type has (type-specific ones live in
/// the inspector). <see cref="RadiusX"/> and <see cref="ConicX"/> are the x sections of anamorphic
/// surfaces: they share the Radius and Conic cells with the y section, so both show and edit in place.
/// </summary>
public enum LensColumn { Surface, Type, Comment, Radius, RadiusX, Thickness, Material, SemiDiameter, Conic, ConicX }

public static class LensColumns
{
    /// <summary>The table's visible columns, in order.</summary>
    public static readonly LensColumn[] Visible =
        [LensColumn.Surface, LensColumn.Type, LensColumn.Comment, LensColumn.Radius, LensColumn.Thickness, LensColumn.Material, LensColumn.SemiDiameter, LensColumn.Conic];

    /// <summary>Columns a text field edits, in keyboard order (Tab).</summary>
    public static readonly LensColumn[] Editable =
        [LensColumn.Comment, LensColumn.Radius, LensColumn.RadiusX, LensColumn.Thickness, LensColumn.Material, LensColumn.SemiDiameter, LensColumn.Conic, LensColumn.ConicX];

    public static string Title(this LensColumn c) => c switch
    {
        LensColumn.Surface => Loc.S("Lens_Column_Surface"),
        LensColumn.Type => Loc.S("Lens_Column_Type"),
        LensColumn.Comment => Loc.S("Lens_Column_Comment"),
        LensColumn.Radius => Loc.S("Lens_Column_Radius"),
        LensColumn.RadiusX => Loc.S("Lens_Column_RadiusX"),
        LensColumn.Thickness => Loc.S("Lens_Column_Thickness"),
        LensColumn.Material => Loc.S("Lens_Column_Material"),
        LensColumn.SemiDiameter => Loc.S("Lens_Column_SemiDiameter"),
        LensColumn.Conic => Loc.S("Lens_Column_Conic"),
        _ => Loc.S("Lens_Column_ConicX"),
    };

    public static bool IsNumeric(this LensColumn c) =>
        c is LensColumn.Radius or LensColumn.RadiusX or LensColumn.Thickness or LensColumn.SemiDiameter or LensColumn.Conic or LensColumn.ConicX;

    /// <summary>The table column a value shows in (an x section in its y section's).</summary>
    public static LensColumn Base(this LensColumn c) => c switch
    {
        LensColumn.RadiusX => LensColumn.Radius,
        LensColumn.ConicX => LensColumn.Conic,
        _ => c,
    };

    /// <summary>
    /// The model parameter behind the column for this surface (for variable and solve markers): radius and conic
    /// of a toroid or biconic are its y section's, <see cref="LensColumn.RadiusX"/> / <see cref="LensColumn.ConicX"/>
    /// its x section's (a toroid's x section has no conic).
    /// </summary>
    public static ParamPath? Param(this LensColumn c, Surface surface)
    {
        var id = surface.Id;
        var g = surface.Shape.Geometry;
        if (g is ToroidShape or BiconicShape)
        {
            return c switch
            {
                LensColumn.Radius => new ParamPath.Curvature(id),
                LensColumn.RadiusX => new ParamPath.ShapeParam(id, ShapeParameter.CurvatureX),
                LensColumn.Conic => new ParamPath.ShapeParam(id, ShapeParameter.ConicY),
                LensColumn.ConicX when g is BiconicShape => new ParamPath.ShapeParam(id, ShapeParameter.ConicX),
                LensColumn.Thickness => new ParamPath.Thickness(id),
                LensColumn.SemiDiameter => new ParamPath.SemiDiameter(id),
                _ => null,
            };
        }
        return c switch
        {
            LensColumn.Radius => new ParamPath.Curvature(id),
            LensColumn.Thickness => new ParamPath.Thickness(id),
            LensColumn.Conic => new ParamPath.Conic(id),
            LensColumn.SemiDiameter => new ParamPath.SemiDiameter(id),
            _ => null,
        };
    }
}

/// <summary>A variable's bounds: none, a range, or pinned (equal bounds).</summary>
public enum VariableBounds { Free, Bounded, Pinned }

/// <summary>A small inline marker next to a value: a variable (<c>V</c>), or a solve's letter.</summary>
public sealed record CellMarker(bool IsVariable, VariableBounds Bounds, string? SolveLetter, string? SolveTitle)
{
    public bool IsEmpty => !IsVariable && SolveLetter is null;
}

/// <summary>What a row's surface label is.</summary>
public enum LensRowRole { Object, Stop, Image, Surface }

/// <summary>
/// What one table row shows, computed from the design plus the engine's automatic semi-diameters
/// (the Mac app's <c>LensRow</c>). No UI types: the table's view draws it, the tests read it.
/// </summary>
public sealed record LensRow
{
    public required SurfaceID Id { get; init; }
    public required int Index { get; init; }
    public required LensRowRole Role { get; init; }
    public required ShapeKind Kind { get; init; }
    public required string TypeTitle { get; init; }
    public required SurfaceTypeChoice TypeChoice { get; init; }
    public bool IsMirror { get; init; }
    /// <summary>The text at rest, by column.</summary>
    public required IReadOnlyDictionary<LensColumn, string> Cells { get; init; }
    /// <summary>Full-precision text an edit starts from.</summary>
    public required IReadOnlyDictionary<LensColumn, string> Exact { get; init; }
    public required IReadOnlySet<LensColumn> Editable { get; init; }
    public IReadOnlyDictionary<LensColumn, CellMarker> Markers { get; init; } = new Dictionary<LensColumn, CellMarker>();
    public bool SemiDiameterIsAuto { get; init; }
    /// <summary>A short axis hint before a value (<c>y</c>, <c>x</c> for the sections of an anamorphic surface).</summary>
    public IReadOnlyDictionary<LensColumn, string> Hints { get; init; } = new Dictionary<LensColumn, string>();
    /// <summary>A subtle note after a value: a catalogue glass's melt offset (<c>+0.0012/−0.8</c>).</summary>
    public IReadOnlyDictionary<LensColumn, string> Notes { get; init; } = new Dictionary<LensColumn, string>();
    /// <summary>The section with power of a cylinder (its other section is flat, shown dimmed).</summary>
    public CylinderAxis? PoweredAxis { get; init; }

    public string SurfaceText => Role switch
    {
        LensRowRole.Object => "OBJ",
        LensRowRole.Stop => "STO",
        LensRowRole.Image => "IMA",
        _ => Index.ToString(System.Globalization.CultureInfo.InvariantCulture),
    };

    public string Text(LensColumn c) => c switch
    {
        LensColumn.Surface => SurfaceText,
        LensColumn.Type => TypeTitle,
        _ => Cells.TryGetValue(c, out var s) ? s : "",
    };

    /// <summary>The text an edit of <paramref name="c"/> starts from.</summary>
    public string ExactText(LensColumn c) => Exact.TryGetValue(c, out var s) ? s : Text(c);

    /// <summary>
    /// The values shown in a table column's cell: the y and x sections of an anamorphic surface (both, or a
    /// cylinder's powered conic alone), else the column itself.
    /// </summary>
    public IReadOnlyList<LensColumn> Shown(LensColumn column)
    {
        LensColumn[] candidates = column.Base() switch
        {
            LensColumn.Radius => [LensColumn.Radius, LensColumn.RadiusX],
            LensColumn.Conic => [LensColumn.Conic, LensColumn.ConicX],
            _ => [column],
        };
        var present = candidates.Where(Cells.ContainsKey).ToArray();
        return present.Length == 0 ? [column.Base()] : present;
    }

    /// <summary>Whether <paramref name="c"/> is the flat section of a cylinder (dimmed).</summary>
    public bool IsFlatSection(LensColumn c) => PoweredAxis switch
    {
        CylinderAxis.X => c is LensColumn.Radius or LensColumn.Conic,
        CylinderAxis.Y => c is LensColumn.RadiusX or LensColumn.ConicX,
        _ => false,
    };

    /// <summary>Automatic semi-diameters (computed by the engine) and a cylinder's flat section are shown dimmed.</summary>
    public bool IsDimmed(LensColumn c) => (c == LensColumn.SemiDiameter && SemiDiameterIsAuto) || IsFlatSection(c);

    /// <summary>
    /// Parses <paramref name="text"/> typed into <paramref name="column"/> into an edit of this row;
    /// <c>null</c> when the column is not edited here.
    /// </summary>
    public Parsed<LensEdit>? Edit(LensColumn column, string text)
    {
        var anamorphic = Kind is ShapeKind.Toroid or ShapeKind.Biconic;
        return column switch
        {
            LensColumn.Comment => Parsed<LensEdit>.Success(new LensEdit.SetLabel(Id, text)),
            LensColumn.RadiusX when anamorphic =>
                LensValueParser.CurvatureFromRadius(text).Map(c => (LensEdit)new LensEdit.SetShapeParameter(Id, ShapeParameter.CurvatureX, c)),
            LensColumn.Conic when anamorphic =>
                LensValueParser.Finite(text).Map(k => (LensEdit)new LensEdit.SetShapeParameter(Id, ShapeParameter.ConicY, k)),
            LensColumn.ConicX when Kind == ShapeKind.Biconic =>
                LensValueParser.Finite(text).Map(k => (LensEdit)new LensEdit.SetShapeParameter(Id, ShapeParameter.ConicX, k)),
            LensColumn.RadiusX or LensColumn.ConicX => null,
            LensColumn.Radius => LensValueParser.CurvatureFromRadius(text).Map(c => (LensEdit)new LensEdit.SetCurvature(Id, c)),
            LensColumn.Thickness => LensValueParser.Thickness(text, Role == LensRowRole.Object).Map(t => (LensEdit)new LensEdit.SetThickness(Id, t)),
            LensColumn.Conic => LensValueParser.Finite(text).Map(k => (LensEdit)new LensEdit.SetConic(Id, k)),
            LensColumn.SemiDiameter => LensValueParser.SemiDiameter(text).Map(sd => (LensEdit)new LensEdit.SetSemiDiameter(Id, sd)),
            LensColumn.Material => LensValueParser.Material(text).Map(m => (LensEdit)new LensEdit.SetMaterial(Id, m)),
            _ => null,
        };
    }

    /// <summary>
    /// What committing <paramref name="text"/> in <paramref name="column"/> asks for: nothing (the text is the
    /// value's own, or the column is not edited), an edit, or a rejection with its reason.
    /// </summary>
    public Parsed<LensEdit>? Commit(LensColumn column, string text) =>
        ExactText(column) == text || !Editable.Contains(column) ? null : Edit(column, text);
}

/// <summary>Builds the table's rows from the design (the Mac app's <c>LensRow.rows</c>).</summary>
public static class LensRows
{
    static readonly IReadOnlyDictionary<int, double> NoAuto = new Dictionary<int, double>();
    static readonly IReadOnlyDictionary<ParamPath, VariableBounds> NoVariables = new Dictionary<ParamPath, VariableBounds>();

    /// <param name="autoSemiDiameters">The engine's automatic semi-diameters, by surface index.</param>
    /// <param name="variables">The design's variables (from the project's <c>variables</c> section), with their bounds.</param>
    public static IReadOnlyList<LensRow> Build(
        OpticalSystem s, IReadOnlyDictionary<int, double>? autoSemiDiameters = null, IReadOnlyDictionary<ParamPath, VariableBounds>? variables = null)
    {
        autoSemiDiameters ??= NoAuto;
        var vars = (variables ?? NoVariables).GroupBy(kv => s.Canonical(kv.Key)).ToDictionary(g => g.Key, g => g.First().Value);
        var solves = new Dictionary<ParamPath, SolveType>();
        foreach (var solve in s.Solves) solves.TryAdd(s.Canonical(solve.Target), solve.Kind.Type);
        var rows = new List<LensRow>(s.Surfaces.Count);
        for (var i = 0; i < s.Surfaces.Count; i++) rows.Add(Row(s, i, autoSemiDiameters, vars, solves));
        return rows;
    }

    static LensRow Row(
        OpticalSystem s, int i, IReadOnlyDictionary<int, double> autoSemiDiameters, Dictionary<ParamPath, VariableBounds> vars,
        Dictionary<ParamPath, SolveType> solves)
    {
        var sf = s.Surfaces[i];
        var role = i == 0 ? LensRowRole.Object : i == s.ImageIndex ? LensRowRole.Image : i == s.Stop ? LensRowRole.Stop : LensRowRole.Surface;
        var cells = new Dictionary<LensColumn, string> { [LensColumn.Comment] = sf.Label };
        var exact = new Dictionary<LensColumn, string> { [LensColumn.Comment] = sf.Label };
        var editable = new HashSet<LensColumn> { LensColumn.Comment };
        var hints = new Dictionary<LensColumn, string>();
        void Set(LensColumn c, string shown, double value)
        {
            cells[c] = shown;
            exact[c] = NumberDisplay.Exact(value);
            editable.Add(c);
        }
        var g = sf.Shape.Geometry;
        // An object at infinity has no meaningful curvature.
        var hasCurvature = sf.Shape.HasCurvature && !(role == LensRowRole.Object && double.IsInfinity(sf.Thickness));
        var axis = g.PowerAxis;
        switch (g)
        {
            case BiconicShape b:
                // Both sections, y first (the meridional section).
                Set(LensColumn.Radius, LensFormat.Plain(RadiusDisplay.RadiusFromCurvature(b.CurvatureY)), RadiusDisplay.RadiusFromCurvature(b.CurvatureY));
                Set(LensColumn.RadiusX, LensFormat.Plain(RadiusDisplay.RadiusFromCurvature(b.CurvatureX)), RadiusDisplay.RadiusFromCurvature(b.CurvatureX));
                // A cylinder's flat section has no meaningful conic: only the powered one shows.
                if (axis is null or CylinderAxis.Y)
                {
                    Set(LensColumn.Conic, LensFormat.Plain(b.ConicY), b.ConicY);
                    hints[LensColumn.Conic] = "y";
                }
                if (axis is null or CylinderAxis.X)
                {
                    Set(LensColumn.ConicX, LensFormat.Plain(b.ConicX), b.ConicX);
                    hints[LensColumn.ConicX] = "x";
                }
                hints[LensColumn.Radius] = "y";
                hints[LensColumn.RadiusX] = "x";
                break;
            case ToroidShape t:
                Set(LensColumn.Radius, LensFormat.Plain(RadiusDisplay.RadiusFromCurvature(t.CurvatureY)), RadiusDisplay.RadiusFromCurvature(t.CurvatureY));
                Set(LensColumn.RadiusX, LensFormat.Plain(RadiusDisplay.RadiusFromCurvature(t.CurvatureX)), RadiusDisplay.RadiusFromCurvature(t.CurvatureX));
                Set(LensColumn.Conic, LensFormat.Plain(t.ConicY), t.ConicY);
                hints[LensColumn.Radius] = "y";
                hints[LensColumn.RadiusX] = "x";
                hints[LensColumn.Conic] = "y";
                break;
            default:
                if (hasCurvature)
                {
                    Set(LensColumn.Radius, LensFormat.Length(sf.Shape.Radius), sf.Shape.Radius);
                    if (sf.Shape.HasConic) Set(LensColumn.Conic, LensFormat.Plain(sf.Shape.ConicConstant), sf.Shape.ConicConstant);
                }
                else if (sf.Shape is ParaxialShape p)
                {
                    cells[LensColumn.Radius] = "f " + LensFormat.Length(p.FocalLength);
                }
                break;
        }
        if (role != LensRowRole.Image) Set(LensColumn.Thickness, LensFormat.Length(sf.Thickness), sf.Thickness);
        var material = sf.Interaction switch
        {
            Interaction.Reflect => "MIRROR",
            Interaction.Dummy => "DUMMY",
            _ => sf.IsCoordinateBreak || sf.Material.IsAir ? "" : sf.Material.DisplayName,
        };
        cells[LensColumn.Material] = material;
        exact[LensColumn.Material] = material;
        if (role != LensRowRole.Image && !sf.IsCoordinateBreak) editable.Add(LensColumn.Material);
        switch (sf.SemiDiameter)
        {
            case SemiDiameter.Fixed f:
                cells[LensColumn.SemiDiameter] = LensFormat.Length(f.Value, 3);
                exact[LensColumn.SemiDiameter] = NumberDisplay.Exact(f.Value);
                break;
            default:
                cells[LensColumn.SemiDiameter] = autoSemiDiameters.TryGetValue(i, out var auto) ? LensFormat.Length(auto, 3) : "auto";
                exact[LensColumn.SemiDiameter] = "auto";
                break;
        }
        if (!sf.IsCoordinateBreak) editable.Add(LensColumn.SemiDiameter);

        var markers = new Dictionary<LensColumn, CellMarker>();
        foreach (var c in Enum.GetValues<LensColumn>())
        {
            if (c.Param(sf) is not { } path) continue;
            var p = s.Canonical(path);
            var isVariable = vars.TryGetValue(p, out var bounds);
            SolveType? solve = solves.TryGetValue(p, out var k) ? k : null;
            if (c == LensColumn.Radius && solves.TryGetValue(new ParamPath.ParaxialFocalLength(sf.Id), out var f)) solve = f;
            if (isVariable || solve is not null)
                markers[c] = new CellMarker(isVariable, isVariable ? bounds : VariableBounds.Free, solve is { } st ? Letter(st) : null, solve is { } sk ? LensText.Solve(sk) : null);
        }
        var notes = new Dictionary<LensColumn, string>();
        if (sf.Interaction == Interaction.Refract && sf.Material is MaterialRef.CatalogGlass { Offset.IsZero: false } glass)
        {
            notes[LensColumn.Material] = LensFormat.Offset(glass.Offset);
            // Offset variables and solves mark the glass.
            var isVariable = false;
            var bounds = VariableBounds.Free;
            SolveType? solve = null;
            foreach (ParamPath p in new ParamPath[] { new ParamPath.MaterialDnD(sf.Id), new ParamPath.MaterialDvD(sf.Id) })
            {
                if (vars.TryGetValue(p, out var b)) (isVariable, bounds) = (true, b);
                if (solves.TryGetValue(p, out var k)) solve = k;
            }
            if (isVariable || solve is not null)
                markers[LensColumn.Material] = new CellMarker(isVariable, bounds, solve is { } st ? Letter(st) : null, solve is { } sk ? LensText.Solve(sk) : null);
        }
        return new LensRow
        {
            Id = sf.Id,
            Index = i,
            Role = role,
            Kind = sf.Shape.Kind,
            TypeTitle = TypeTitle(sf.Shape),
            TypeChoice = SurfaceTypeChoice.For(sf.Shape),
            IsMirror = sf.Interaction == Interaction.Reflect,
            Cells = cells,
            Exact = exact,
            Editable = editable,
            Markers = markers,
            SemiDiameterIsAuto = sf.SemiDiameter is SemiDiameter.Auto,
            Hints = hints,
            Notes = notes,
            PoweredAxis = axis,
        };
    }

    /// <summary>Short type name for the table: a conic of k = 0 is a sphere.</summary>
    public static string TypeTitle(SurfaceShape shape) => shape.Geometry switch
    {
        _ when shape is DiffractiveShape => Loc.S("Lens_TypeShort_Diffractive"),
        ConicShape c => c.Conic == 0 ? Loc.S("Lens_TypeShort_Sphere") : Loc.S("Lens_TypeShort_Conic"),
        QConShape => "Q-Con",
        QBfsShape => "Q-BFS",
        BiconicShape b when b.IsCylinder => Loc.S("Lens_TypeShort_Cylinder"),
        _ => SurfaceTypeCatalog.Title(shape.Kind),
    };

    /// <summary>One-letter marker shown next to a solved value.</summary>
    public static string Letter(SolveType t) => t switch
    {
        SolveType.MarginalRayHeight or SolveType.MarginalRayAngle => "M",
        SolveType.ChiefRayHeight or SolveType.ChiefRayAngle => "C",
        SolveType.FNumber => "F",
        SolveType.Pickup => "P",
        SolveType.Position => "D",
        SolveType.CenterOfCurvature => "O",
        _ => "E",
    };

    /// <summary>The panel header's caption: <c>11 surfaces · stop on 6 · mm</c> (and the variable count).</summary>
    public static string Caption(OpticalSystem s, int variableCount = 0)
    {
        var n = s.Surfaces.Count - 2;
        var text = Loc.F(n == 1 ? "Lens_Caption_One" : "Lens_Caption_Many", n, s.Stop);
        return variableCount == 0 ? text : Loc.F(variableCount == 1 ? "Lens_Caption_Variables_One" : "Lens_Caption_Variables_Many", text, variableCount);
    }

    /// <summary>
    /// The design's variables from the project's <c>variables</c> section (<c>[{param, lower, upper}]</c>, the
    /// engine's <c>VariableSpec</c>), with their bounds. Entries that do not read are skipped.
    /// </summary>
    public static IReadOnlyDictionary<ParamPath, VariableBounds> Variables(JsonTree section)
    {
        var result = new Dictionary<ParamPath, VariableBounds>();
        if (section.Node is not System.Text.Json.Nodes.JsonArray list) return result;
        foreach (var item in list)
        {
            if (item is not System.Text.Json.Nodes.JsonObject o) continue;
            try
            {
                var p = ParamPath.FromJson(o["param"]);
                double? Bound(string key) => o[key] is System.Text.Json.Nodes.JsonValue v && v.TryGetValue<double>(out var d) ? d : null;
                var (lower, upper) = (Bound("lower"), Bound("upper"));
                result.TryAdd(p, lower is not null && lower == upper ? VariableBounds.Pinned
                    : lower is not null || upper is not null ? VariableBounds.Bounded : VariableBounds.Free);
            }
            catch (Exception e) when (e is System.Text.Json.JsonException or InvalidOperationException or FormatException)
            {
                // A variable the model does not read: the engine reports it.
            }
        }
        return result;
    }
}

/// <summary>Groups of the type menu.</summary>
public enum SurfaceTypeGroup { Spherical, Aspheric, Freeform, Anamorphic, Diffractive, Special }

/// <summary>An entry of the type menu: a model type, or a cylinder (a biconic preset with one plane axis).</summary>
public readonly record struct SurfaceTypeChoice(ShapeKind Kind, CylinderAxis? Cylinder = null)
{
    /// <summary>The menu entry a shape is: a cylinder shows as its preset.</summary>
    public static SurfaceTypeChoice For(SurfaceShape shape) =>
        shape.Geometry is BiconicShape { PowerAxis: { } a } && shape is not DiffractiveShape ? new(ShapeKind.Biconic, a) : new(shape.Kind);
}

/// <summary>The type menu as data (the Mac app's <c>SurfaceTypes.menu</c>): groups in order, each with its types, a title and a Segoe Fluent glyph.</summary>
public static class SurfaceTypeCatalog
{
    /// <param name="Title">The title in the UI language.</param>
    /// <param name="English">The English title (automation sets a type by either).</param>
    public sealed record Entry(SurfaceTypeChoice Choice, string Title, string Glyph, SurfaceTypeGroup Group, string English);

    public static readonly Entry[] All =
    [
        new(new(ShapeKind.Plane), Loc.S("Lens_Type_Plane"), "\uE739", SurfaceTypeGroup.Spherical, "Plane"),
        new(new(ShapeKind.Conic), Loc.S("Lens_Type_Conic"), "\uE7A1", SurfaceTypeGroup.Spherical, "Sphere / Conic"),
        new(new(ShapeKind.EvenAsphere), Loc.S("Lens_Type_EvenAsphere"), "\uE9D9", SurfaceTypeGroup.Aspheric, "Even Asphere"),
        new(new(ShapeKind.OddAsphere), Loc.S("Lens_Type_OddAsphere"), "\uE9D2", SurfaceTypeGroup.Aspheric, "Odd Asphere"),
        new(new(ShapeKind.QCon), Loc.S("Lens_Type_QCon"), "\uEA3F", SurfaceTypeGroup.Aspheric, "Q-Type (Conic)"),
        new(new(ShapeKind.QBfs), Loc.S("Lens_Type_QBfs"), "\uE922", SurfaceTypeGroup.Aspheric, "Q-Type (Best-Fit Sphere)"),
        new(new(ShapeKind.XyPolynomial), Loc.S("Lens_Type_XyPolynomial"), "\uE809", SurfaceTypeGroup.Freeform, "XY Polynomial"),
        new(new(ShapeKind.ZernikeSag), Loc.S("Lens_Type_ZernikeSag"), "\uE8A9", SurfaceTypeGroup.Freeform, "Zernike Sag"),
        new(new(ShapeKind.Toroid), Loc.S("Lens_Type_Toroid"), "\uEA4B", SurfaceTypeGroup.Anamorphic, "Toroid"),
        new(new(ShapeKind.Biconic), Loc.S("Lens_Type_Biconic"), "\uEA3A", SurfaceTypeGroup.Anamorphic, "Biconic"),
        new(new(ShapeKind.Biconic, CylinderAxis.X), Loc.S("Lens_Type_CylinderX"), "\uE8EA", SurfaceTypeGroup.Anamorphic, "Cylinder (power in x)"),
        new(new(ShapeKind.Biconic, CylinderAxis.Y), Loc.S("Lens_Type_CylinderY"), "\uE8EA", SurfaceTypeGroup.Anamorphic, "Cylinder (power in y)"),
        new(new(ShapeKind.Diffractive), Loc.S("Lens_Type_Diffractive"), "\uE790", SurfaceTypeGroup.Diffractive, "Diffractive"),
        new(new(ShapeKind.Paraxial), Loc.S("Lens_Type_IdealLens"), "\uE8A3", SurfaceTypeGroup.Special, "Ideal Lens"),
        new(new(ShapeKind.CoordinateBreak), Loc.S("Lens_Type_CoordinateBreak"), "\uE7C2", SurfaceTypeGroup.Special, "Coordinate Break"),
    ];

    public static string Title(SurfaceTypeGroup g) => LensText.Group(g);

    public static Entry EntryFor(SurfaceTypeChoice choice) => All.FirstOrDefault(e => e.Choice == choice) ?? All.First(e => e.Choice.Kind == choice.Kind);

    public static string Title(ShapeKind kind) => EntryFor(new(kind)).Title;
}
