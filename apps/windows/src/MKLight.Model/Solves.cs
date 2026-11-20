using System.Text.Json.Nodes;
using System.Text.Json.Serialization;

namespace MKLight.Model;

/// <summary>
/// A solve or pickup: <c>Target</c> follows from the rule <c>Kind</c> (mirror of the
/// engine's <c>Solve</c>, docs/SOLVES.md). The stored target value is only the starting
/// point; the engine's solver returns a new system with every target set.
/// </summary>
[JsonConverter(typeof(ModelConverter<Solve>))]
public sealed record Solve(ParamPath Target, SolveKind Kind) : IJsonModel<Solve>
{
    /// <summary>Surfaces the solve refers to (target, pickup source, position range, centre of curvature).</summary>
    public IReadOnlyList<SurfaceID> Surfaces
    {
        get
        {
            var output = new List<SurfaceID>();
            if (Target.SurfaceId is { } t) output.Add(t);
            switch (Kind)
            {
                case SolveKind.Pickup p when p.Source.SurfaceId is { } s: output.Add(s); break;
                case SolveKind.Position p: output.AddRange([p.From, p.To]); break;
                case SolveKind.CenterOfCurvature c: output.Add(c.Surface); break;
            }
            return output;
        }
    }

    public static Solve FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        J.AllowOnly(o, "target", "kind");
        return new Solve(J.Req(o, "target", ParamPath.FromJson), J.Req(o, "kind", SolveKind.FromJson));
    }

    public JsonNode ToJson() => new JsonObject { ["target"] = Target.ToJson(), ["kind"] = Kind.ToJson() };
}

/// <summary>The tag of a <see cref="SolveKind"/>.</summary>
public enum SolveType
{
    [Wire("marginal_ray_height")] MarginalRayHeight,
    [Wire("marginal_ray_angle")] MarginalRayAngle,
    [Wire("chief_ray_height")] ChiefRayHeight,
    [Wire("chief_ray_angle")] ChiefRayAngle,
    [Wire("f_number")] FNumber,
    [Wire("pickup")] Pickup,
    [Wire("position")] Position,
    [Wire("center_of_curvature")] CenterOfCurvature,
    [Wire("edge_thickness")] EdgeThickness,
}

public static class SolveTypeExtensions
{
    public static readonly IReadOnlyList<SolveType> All = Enum.GetValues<SolveType>();

    public static string Title(this SolveType t) => t switch
    {
        SolveType.MarginalRayHeight => "Marginal ray height",
        SolveType.MarginalRayAngle => "Marginal ray angle",
        SolveType.ChiefRayHeight => "Chief ray height",
        SolveType.ChiefRayAngle => "Chief ray angle",
        SolveType.FNumber => "F-number",
        SolveType.Pickup => "Pickup",
        SolveType.Position => "Position",
        SolveType.CenterOfCurvature => "Centre of curvature",
        _ => "Edge thickness",
    };

    /// <summary>Label and unit of the kind's main number. Slopes are "paraxial angle u", dimensionless (CONVENTIONS §8).</summary>
    public static (string Label, string Unit)? ValueLabel(this SolveType t) => t switch
    {
        SolveType.MarginalRayHeight or SolveType.ChiefRayHeight => ("height", "mm"),
        SolveType.MarginalRayAngle or SolveType.ChiefRayAngle => ("paraxial angle u", ""),
        SolveType.FNumber => ("F/#", ""),
        SolveType.Position => ("distance", "mm"),
        SolveType.EdgeThickness => ("edge thickness", "mm"),
        _ => null,
    };

    /// <summary>Whether this kind can control a thickness (else a curvature or focal length, except pickups which take any parameter).</summary>
    public static bool TargetsThickness(this SolveType t) =>
        t is SolveType.MarginalRayHeight or SolveType.ChiefRayHeight or SolveType.Position or SolveType.EdgeThickness;
}

/// <summary>
/// The rule that sets a solve's target (engine <c>SolveKind</c>, tagged by <c>"type"</c>).
/// Angle solves take the paraxial <b>slope</b> <c>u = dy/dz</c>, which is dimensionless:
/// UI labels call it "paraxial angle u" and it is never shown or converted as degrees (CONVENTIONS §8).
/// </summary>
[JsonConverter(typeof(ModelConverter<SolveKind>))]
public abstract record SolveKind : IJsonModel<SolveKind>
{
    /// <summary>On a thickness: marginal-ray height (mm) on the next vertex.</summary>
    public sealed record MarginalRayHeight(double Height) : SolveKind;

    /// <summary>On a curvature or ideal-lens focal length: marginal-ray slope <c>u</c> after the surface.</summary>
    public sealed record MarginalRayAngle(double Slope) : SolveKind;

    public sealed record ChiefRayHeight(double Height) : SolveKind;

    public sealed record ChiefRayAngle(double Slope) : SolveKind;

    /// <summary>Image-space F/# <c>f'/EPD</c>.</summary>
    public sealed record FNumber(double Value) : SolveKind;

    /// <summary><c>target = scale · source + offset</c>.</summary>
    public sealed record Pickup(ParamPath Source, double Scale = 1, double Offset = 0) : SolveKind;

    /// <summary>Axial distance (mm) from the vertex of <c>From</c> to that of <c>To</c>.</summary>
    public sealed record Position(SurfaceID From, SurfaceID To, double Distance) : SolveKind;

    /// <summary>Centre of curvature on the vertex of <c>Surface</c>.</summary>
    public sealed record CenterOfCurvature(SurfaceID Surface) : SolveKind;

    /// <summary>Axial edge thickness (mm) at <c>Radius</c> (<c>null</c>: from the semi-diameters).</summary>
    public sealed record EdgeThickness(double Value, double? Radius = null) : SolveKind;

    public SolveType Type => this switch
    {
        MarginalRayHeight => SolveType.MarginalRayHeight,
        MarginalRayAngle => SolveType.MarginalRayAngle,
        ChiefRayHeight => SolveType.ChiefRayHeight,
        ChiefRayAngle => SolveType.ChiefRayAngle,
        FNumber => SolveType.FNumber,
        Pickup => SolveType.Pickup,
        Position => SolveType.Position,
        CenterOfCurvature => SolveType.CenterOfCurvature,
        _ => SolveType.EdgeThickness,
    };

    /// <summary>The kind's main number (height, slope, F/#, distance, edge thickness).</summary>
    public double? MainValue => this switch
    {
        MarginalRayHeight k => k.Height,
        ChiefRayHeight k => k.Height,
        MarginalRayAngle k => k.Slope,
        ChiefRayAngle k => k.Slope,
        FNumber k => k.Value,
        Position k => k.Distance,
        EdgeThickness k => k.Value,
        _ => null,
    };

    /// <summary>The kind with its main number replaced (unchanged for pickups and centres of curvature).</summary>
    public SolveKind WithMainValue(double v) => this switch
    {
        MarginalRayHeight => new MarginalRayHeight(v),
        ChiefRayHeight => new ChiefRayHeight(v),
        MarginalRayAngle => new MarginalRayAngle(v),
        ChiefRayAngle => new ChiefRayAngle(v),
        FNumber => new FNumber(v),
        Position p => p with { Distance = v },
        EdgeThickness e => e with { Value = v },
        _ => this,
    };

    /// <summary>Every number of the kind (for finiteness checks).</summary>
    internal IEnumerable<double> Numbers => this switch
    {
        Pickup p => [p.Scale, p.Offset],
        CenterOfCurvature => [],
        EdgeThickness e => e.Radius is { } r ? [e.Value, r] : [e.Value],
        _ => MainValue is { } v ? [v] : [],
    };

    public static SolveKind FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        var t = J.Tag(o, "type");
        var type = Wire.Parse<SolveType>(t) ?? throw J.Fail($"unknown solve '{t}'");
        switch (type)
        {
            case SolveType.MarginalRayHeight or SolveType.ChiefRayHeight: J.AllowOnly(o, "type", "height"); break;
            case SolveType.MarginalRayAngle or SolveType.ChiefRayAngle: J.AllowOnly(o, "type", "slope"); break;
            case SolveType.FNumber: J.AllowOnly(o, "type", "value"); break;
            case SolveType.Pickup: J.AllowOnly(o, "type", "source", "scale", "offset"); break;
            case SolveType.Position: J.AllowOnly(o, "type", "from", "to", "distance"); break;
            case SolveType.CenterOfCurvature: J.AllowOnly(o, "type", "surface"); break;
            case SolveType.EdgeThickness: J.AllowOnly(o, "type", "value", "radius"); break;
        }
        return type switch
        {
            SolveType.MarginalRayHeight => new MarginalRayHeight(J.Req(o, "height", J.D)),
            SolveType.MarginalRayAngle => new MarginalRayAngle(J.Req(o, "slope", J.D)),
            SolveType.ChiefRayHeight => new ChiefRayHeight(J.Req(o, "height", J.D)),
            SolveType.ChiefRayAngle => new ChiefRayAngle(J.Req(o, "slope", J.D)),
            SolveType.FNumber => new FNumber(J.Req(o, "value", J.D)),
            SolveType.Pickup => new Pickup(J.Req(o, "source", ParamPath.FromJson), J.Def(o, "scale", J.D, 1), J.Def(o, "offset", J.D, 0)),
            SolveType.Position => new Position(
                J.Req(o, "from", SurfaceID.FromJson), J.Req(o, "to", SurfaceID.FromJson), J.Req(o, "distance", J.D)),
            SolveType.CenterOfCurvature => new CenterOfCurvature(J.Req(o, "surface", SurfaceID.FromJson)),
            _ => new EdgeThickness(J.Req(o, "value", J.D), J.OptVal(o, "radius", J.D)),
        };
    }

    public JsonNode ToJson()
    {
        var o = new JsonObject { ["type"] = Wire.Name(Type) };
        switch (this)
        {
            case MarginalRayHeight k: o["height"] = J.Num(k.Height); break;
            case ChiefRayHeight k: o["height"] = J.Num(k.Height); break;
            case MarginalRayAngle k: o["slope"] = J.Num(k.Slope); break;
            case ChiefRayAngle k: o["slope"] = J.Num(k.Slope); break;
            case FNumber k: o["value"] = J.Num(k.Value); break;
            case Pickup k:
                o["source"] = k.Source.ToJson();
                o["scale"] = J.Num(k.Scale);
                o["offset"] = J.Num(k.Offset);
                break;
            case Position k:
                o["from"] = k.From.ToJson();
                o["to"] = k.To.ToJson();
                o["distance"] = J.Num(k.Distance);
                break;
            case CenterOfCurvature k: o["surface"] = k.Surface.ToJson(); break;
            case EdgeThickness k:
                o["value"] = J.Num(k.Value);
                o["radius"] = k.Radius is { } r ? J.Num(r) : null;
                break;
        }
        return o;
    }
}

public sealed partial record OpticalSystem
{
    /// <summary>Parameters controlled by solves (exclude them from optimisation; the editor marks them).</summary>
    public IReadOnlySet<ParamPath> SolvedParams => Solves.Select(s => Canonical(s.Target)).ToHashSet();

    /// <summary>The solve controlling <paramref name="path"/> (by any path naming the same parameter), if any.</summary>
    public Solve? SolveFor(ParamPath path)
    {
        var p = Canonical(path);
        return Solves.FirstOrDefault(s => Canonical(s.Target) == p);
    }

    /// <summary>Whether a solve sets <paramref name="path"/>. <c>optimize</c> applies solves first and refuses such a parameter as a variable.</summary>
    public bool IsSolved(ParamPath path) => SolveFor(path) is not null;

    /// <summary>
    /// One path per parameter: on a toroid or biconic the <c>curvature</c> path is the y section's
    /// curvature, so <c>shape_param curvature_y</c> there becomes <c>curvature</c>. Every other path is its own.
    /// </summary>
    public ParamPath Canonical(ParamPath path)
    {
        if (path is not ParamPath.ShapeParam { Parameter: ShapeParameter.CurvatureY } p || SurfaceIndexOf(p.Surface) is not { } i) return path;
        return Surfaces[i].Shape is ToroidShape or BiconicShape ? new ParamPath.Curvature(p.Surface) : path;
    }

    /// <summary>
    /// Structural checks of the engine's solve validation that need no optics: every target and source
    /// exists, no parameter has two solves, no pickup of itself, finite numbers, positive F/# and edge
    /// radius. The engine's validation is authoritative (type rules, cycles).
    /// </summary>
    internal List<ModelIssue> SolveIssues()
    {
        var output = new List<ModelIssue>();
        var seen = new HashSet<ParamPath>();
        for (var k = 0; k < Solves.Count; k++)
        {
            var s = Solves[k];
            var name = $"solve {k} ({Wire.Name(s.Kind.Type)})";
            void Error(string message) => output.Add(new ModelIssue(IssueSeverity.Error, $"{name}: {message}"));
            if (!seen.Add(s.Target)) Error($"{s.Target} already has a solve");
            // An automatic semi-diameter can be a target: the solve fixes it.
            var fixesSemiDiameter = s.Target is ParamPath.SemiDiameter;
            if ((!fixesSemiDiameter || (s.Target.SurfaceId is { } ts ? SurfaceIndexOf(ts) : null) is null) && s.Target.UnresolvedReason(this) is { } why)
                Error($"invalid target {s.Target}: {why}");
            if (s.Kind is SolveKind.Pickup pickup)
            {
                if (pickup.Source == s.Target) Error("a pickup cannot read its own target");
                if (SolveFor(pickup.Source) is null && pickup.Source.UnresolvedReason(this) is { } w)
                    Error($"invalid source {pickup.Source}: {w}");
            }
            foreach (var id in s.Surfaces)
            {
                if (SurfaceIndexOf(id) is null) Error($"no surface with id {id.Value}");
            }
            if (s.Kind.Numbers.Any(x => !double.IsFinite(x))) Error("values must be finite");
            if (s.Kind is SolveKind.FNumber f && !(f.Value > 0)) Error("F/# must be > 0");
            if (s.Kind is SolveKind.EdgeThickness { Radius: { } r } && !(r > 0)) Error("edge radius must be > 0");
        }
        return output;
    }
}

/// <summary>How a solve's value was found (engine <c>SolveMethod</c>).</summary>
public enum SolveMethod
{
    [Wire("unchanged")] Unchanged,
    [Wire("closed_form")] ClosedForm,
    [Wire("fixed_point")] FixedPoint,
    [Wire("root_finding")] RootFinding,
}

/// <summary>One solve's outcome (engine <c>SolveOutcome</c>).</summary>
[JsonConverter(typeof(ModelConverter<SolveOutcome>))]
public sealed record SolveOutcome : IJsonModel<SolveOutcome>
{
    /// <summary>Index into <c>OpticalSystem.Solves</c>.</summary>
    public required int Index { get; init; }

    public required string Kind { get; init; }
    public required ParamPath Target { get; init; }

    /// <summary>Value in the input system (<c>null</c> for an automatic semi-diameter).</summary>
    public double? Previous { get; init; }

    public required double Value { get; init; }

    /// <summary><c>measured − goal</c> on the solved system, in the condition's units (mm, slope, F/#, or dimensionless).</summary>
    public required double Residual { get; init; }

    public required SolveMethod Method { get; init; }
    public required int Evaluations { get; init; }

    public static SolveOutcome FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        return new SolveOutcome
        {
            Index = J.Req(o, "index", J.I32),
            Kind = J.Req(o, "kind", J.S),
            Target = J.Req(o, "target", ParamPath.FromJson),
            Previous = J.OptVal(o, "previous", J.D),
            Value = J.Req(o, "value", J.D),
            Residual = J.Req(o, "residual", J.D),
            Method = J.Req(o, "method", n => Wire.Read<SolveMethod>(n, "solve method")),
            Evaluations = J.Req(o, "evaluations", J.I32),
        };
    }

    public JsonNode ToJson() => new JsonObject
    {
        ["index"] = Index,
        ["kind"] = Kind,
        ["target"] = Target.ToJson(),
        ["previous"] = Previous is { } p ? J.Num(p) : null, // explicit null, as serde writes it
        ["value"] = J.Num(Value),
        ["residual"] = J.Num(Residual),
        ["method"] = Wire.Write(Method),
        ["evaluations"] = Evaluations,
    };
}

/// <summary>
/// Result of applying a system's solves (engine <c>SolveReport</c>). The run metadata stays a
/// raw JSON tree here; the engine client decodes it.
/// </summary>
[JsonConverter(typeof(ModelConverter<SolveReport>))]
public sealed record SolveReport : IJsonModel<SolveReport>
{
    public required ValueList<SolveOutcome> Outcomes { get; init; }

    /// <summary>Evaluation order (indices into <c>Solves</c>).</summary>
    public required ValueList<int> Order { get; init; }

    public required int Sweeps { get; init; }
    public required double MaxChange { get; init; }
    public required ValueList<ParamPath> SolvedParams { get; init; }
    public ValueList<string> Warnings { get; init; } = ValueList<string>.Empty;

    /// <summary>
    /// The configuration that was solved (the returned system is that configuration, flattened);
    /// <c>null</c> for the system as given.
    /// </summary>
    public int? Configuration { get; init; }

    public JsonTree RunInfo { get; init; }

    public static SolveReport FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        return new SolveReport
        {
            Outcomes = J.Req(o, "outcomes", n => J.Arr(n, SolveOutcome.FromJson)),
            Order = J.Req(o, "order", n => J.Arr(n, J.I32)),
            Sweeps = J.Req(o, "sweeps", J.I32),
            MaxChange = J.Req(o, "max_change", J.D),
            SolvedParams = J.Req(o, "solved_params", n => J.Arr(n, ParamPath.FromJson)),
            Warnings = J.OptRef(o, "warnings", n => J.Arr(n, J.S)) ?? ValueList<string>.Empty,
            Configuration = J.OptVal(o, "configuration", J.I32),
            RunInfo = J.Req(o, "run_info", n => new JsonTree(n)),
        };
    }

    public JsonNode ToJson() => new JsonObject
    {
        ["outcomes"] = J.Arr(Outcomes, x => x.ToJson()),
        ["order"] = J.Arr(Order, i => (JsonNode)i),
        ["sweeps"] = Sweeps,
        ["max_change"] = J.Num(MaxChange),
        ["solved_params"] = J.Arr(SolvedParams, p => p.ToJson()),
        ["warnings"] = J.Arr(Warnings, w => (JsonNode)w),
        ["configuration"] = Configuration,
        ["run_info"] = RunInfo.Node?.DeepClone(),
    };
}
