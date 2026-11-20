using System.Text.Json.Nodes;
using System.Text.Json.Serialization;

namespace MKLight.Model;

/// <summary>Part of a sag overlay's fringe irregularity.</summary>
public enum IrregularityMode
{
    [Wire("astigmatism")] Astigmatism,
    [Wire("spherical")] Spherical,
}

/// <summary>One component of a rigid <see cref="ElementPlacement"/> motion.</summary>
public enum PlacementComponent
{
    [Wire("decenter_x")] DecenterX,
    [Wire("decenter_y")] DecenterY,
    [Wire("tilt_x")] TiltX,
    [Wire("tilt_y")] TiltY,
    [Wire("tilt_z")] TiltZ,
}

public static class PlacementComponentExtensions
{
    public static string Title(this PlacementComponent c) => c switch
    {
        PlacementComponent.DecenterX => "Decenter X",
        PlacementComponent.DecenterY => "Decenter Y",
        PlacementComponent.TiltX => "Tilt X",
        PlacementComponent.TiltY => "Tilt Y",
        _ => "Tilt Z",
    };

    public static double Get(this PlacementComponent c, Placement p) => c switch
    {
        PlacementComponent.DecenterX => p.DecenterX,
        PlacementComponent.DecenterY => p.DecenterY,
        PlacementComponent.TiltX => p.TiltX,
        PlacementComponent.TiltY => p.TiltY,
        _ => p.TiltZ,
    };

    public static Placement With(this PlacementComponent c, Placement p, double v) => c switch
    {
        PlacementComponent.DecenterX => p with { DecenterX = v },
        PlacementComponent.DecenterY => p with { DecenterY = v },
        PlacementComponent.TiltX => p with { TiltX = v },
        PlacementComponent.TiltY => p with { TiltY = v },
        _ => p with { TiltZ = v },
    };
}

/// <summary>
/// Address of one numeric parameter of an <see cref="OpticalSystem"/> (mirror of the
/// engine's <c>ParamPath</c>). Surfaces are addressed by <see cref="SurfaceID"/>, so paths
/// survive inserts and deletes.
/// </summary>
[JsonConverter(typeof(ModelConverter<ParamPath>))]
public abstract partial record ParamPath : IJsonModel<ParamPath>
{
    private protected ParamPath() { }

    public sealed record Curvature(SurfaceID Surface) : ParamPath;

    public sealed record Thickness(SurfaceID Surface) : ParamPath;

    public sealed record Conic(SurfaceID Surface) : ParamPath;

    /// <summary>Even-asphere coefficient <c>Index</c> (0 → A4).</summary>
    public sealed record Asphere(SurfaceID Surface, int Index) : ParamPath;

    /// <summary>
    /// Coefficient <c>Index</c> (zero-based) of an extended shape's coefficient family
    /// (SURFACES §7). A missing in-range coefficient reads 0.
    /// </summary>
    public sealed record ShapeCoefficient(SurfaceID Surface, CoefficientSet Set, int Index) : ParamPath;

    /// <summary>A scalar parameter of an extended shape. JSON names the selector <c>parameter</c>, since <c>param</c> is the tag.</summary>
    public sealed record ShapeParam(SurfaceID Surface, ShapeParameter Parameter) : ParamPath;

    public sealed record ParaxialFocalLength(SurfaceID Surface) : ParamPath;

    public sealed record ModelNd(SurfaceID Surface) : ParamPath;

    public sealed record ModelVd(SurfaceID Surface) : ParamPath;

    public sealed record ModelDpgf(SurfaceID Surface) : ParamPath;

    /// <summary><c>Δn_d</c> of a catalog glass's <c>MaterialOffset</c>.</summary>
    public sealed record MaterialDnD(SurfaceID Surface) : ParamPath;

    /// <summary><c>ΔV_d</c> of a catalog glass's <c>MaterialOffset</c>.</summary>
    public sealed record MaterialDvD(SurfaceID Surface) : ParamPath;

    public sealed record DecenterX(SurfaceID Surface) : ParamPath;

    public sealed record DecenterY(SurfaceID Surface) : ParamPath;

    public sealed record TiltX(SurfaceID Surface) : ParamPath;

    public sealed record TiltY(SurfaceID Surface) : ParamPath;

    public sealed record TiltZ(SurfaceID Surface) : ParamPath;

    public sealed record SemiDiameter(SurfaceID Surface) : ParamPath;

    public sealed record ApertureValue : ParamPath;

    public sealed record FieldX(int Index) : ParamPath;

    public sealed record FieldY(int Index) : ParamPath;

    public sealed record FieldWeight(int Index) : ParamPath;

    public sealed record WavelengthUm(int Index) : ParamPath;

    public sealed record WavelengthWeight(int Index) : ParamPath;

    public sealed record TemperatureC : ParamPath;

    public sealed record PressureAtm : ParamPath;

    /// <summary>Test fringes of a sag overlay (SURFACES §10); 0 without one.</summary>
    public sealed record Irregularity(SurfaceID Surface, IrregularityMode Mode) : ParamPath;

    /// <summary>
    /// One component of the rigid motion of the range <c>Surface…Last</c> about
    /// <c>Pivot</c> (<c>null</c>: <c>Surface</c>); mm or degrees.
    /// </summary>
    public sealed record ElementPlacement(SurfaceID Surface, SurfaceID Last, SurfaceID? Pivot, PlacementComponent Component) : ParamPath;

    /// <summary>The surface a surface-addressed path names, if any.</summary>
    public SurfaceID? SurfaceId => this switch
    {
        Curvature p => p.Surface,
        Thickness p => p.Surface,
        Conic p => p.Surface,
        Asphere p => p.Surface,
        ShapeCoefficient p => p.Surface,
        ShapeParam p => p.Surface,
        ParaxialFocalLength p => p.Surface,
        ModelNd p => p.Surface,
        ModelVd p => p.Surface,
        ModelDpgf p => p.Surface,
        MaterialDnD p => p.Surface,
        MaterialDvD p => p.Surface,
        DecenterX p => p.Surface,
        DecenterY p => p.Surface,
        TiltX p => p.Surface,
        TiltY p => p.Surface,
        TiltZ p => p.Surface,
        SemiDiameter p => p.Surface,
        Irregularity p => p.Surface,
        ElementPlacement p => p.Surface,
        _ => null,
    };

    private static readonly Dictionary<string, Func<SurfaceID, ParamPath>> SurfaceTags = new()
    {
        ["curvature"] = s => new Curvature(s),
        ["thickness"] = s => new Thickness(s),
        ["conic"] = s => new Conic(s),
        ["paraxial_focal_length"] = s => new ParaxialFocalLength(s),
        ["model_nd"] = s => new ModelNd(s),
        ["model_vd"] = s => new ModelVd(s),
        ["model_dpgf"] = s => new ModelDpgf(s),
        ["material_dn_d"] = s => new MaterialDnD(s),
        ["material_dv_d"] = s => new MaterialDvD(s),
        ["decenter_x"] = s => new DecenterX(s),
        ["decenter_y"] = s => new DecenterY(s),
        ["tilt_x"] = s => new TiltX(s),
        ["tilt_y"] = s => new TiltY(s),
        ["tilt_z"] = s => new TiltZ(s),
        ["semi_diameter"] = s => new SemiDiameter(s),
    };

    private static readonly Dictionary<string, Func<int, ParamPath>> IndexTags = new()
    {
        ["field_x"] = i => new FieldX(i),
        ["field_y"] = i => new FieldY(i),
        ["field_weight"] = i => new FieldWeight(i),
        ["wavelength_um"] = i => new WavelengthUm(i),
        ["wavelength_weight"] = i => new WavelengthWeight(i),
    };

    /// <summary>The JSON <c>param</c> tag and the <c>index</c> of this path.</summary>
    private (string Tag, int? Index) Parts => this switch
    {
        Curvature => ("curvature", null),
        Thickness => ("thickness", null),
        Conic => ("conic", null),
        Asphere p => ("asphere", p.Index),
        ShapeCoefficient p => ("shape_coefficient", p.Index),
        ShapeParam => ("shape_param", null),
        ParaxialFocalLength => ("paraxial_focal_length", null),
        ModelNd => ("model_nd", null),
        ModelVd => ("model_vd", null),
        ModelDpgf => ("model_dpgf", null),
        MaterialDnD => ("material_dn_d", null),
        MaterialDvD => ("material_dv_d", null),
        DecenterX => ("decenter_x", null),
        DecenterY => ("decenter_y", null),
        TiltX => ("tilt_x", null),
        TiltY => ("tilt_y", null),
        TiltZ => ("tilt_z", null),
        SemiDiameter => ("semi_diameter", null),
        ApertureValue => ("aperture_value", null),
        FieldX p => ("field_x", p.Index),
        FieldY p => ("field_y", p.Index),
        FieldWeight p => ("field_weight", p.Index),
        WavelengthUm p => ("wavelength_um", p.Index),
        WavelengthWeight p => ("wavelength_weight", p.Index),
        TemperatureC => ("temperature_c", null),
        PressureAtm => ("pressure_atm", null),
        Irregularity => ("irregularity", null),
        _ => ("element_placement", null),
    };

    public static ParamPath FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        var tag = J.Tag(o, "param");
        if (SurfaceTags.TryGetValue(tag, out var makeSurface))
        {
            J.AllowOnly(o, "param", "surface");
            return makeSurface(J.Req(o, "surface", SurfaceID.FromJson));
        }
        if (IndexTags.TryGetValue(tag, out var makeIndex))
        {
            J.AllowOnly(o, "param", "index");
            return makeIndex(J.Req(o, "index", J.Index));
        }
        switch (tag)
        {
            case "asphere":
                J.AllowOnly(o, "param", "surface", "index");
                return new Asphere(J.Req(o, "surface", SurfaceID.FromJson), J.Req(o, "index", J.Index));
            case "shape_coefficient":
                J.AllowOnly(o, "param", "surface", "set", "index");
                return new ShapeCoefficient(
                    J.Req(o, "surface", SurfaceID.FromJson), J.Req(o, "set", n => Wire.Read<CoefficientSet>(n, "coefficient set")),
                    J.Req(o, "index", J.Index));
            case "shape_param":
                J.AllowOnly(o, "param", "surface", "parameter");
                return new ShapeParam(
                    J.Req(o, "surface", SurfaceID.FromJson), J.Req(o, "parameter", n => Wire.Read<ShapeParameter>(n, "shape parameter")));
            case "irregularity":
                J.AllowOnly(o, "param", "surface", "mode");
                return new Irregularity(
                    J.Req(o, "surface", SurfaceID.FromJson), J.Req(o, "mode", n => Wire.Read<IrregularityMode>(n, "irregularity mode")));
            case "element_placement":
                J.AllowOnly(o, "param", "surface", "last", "pivot", "component");
                return new ElementPlacement(
                    J.Req(o, "surface", SurfaceID.FromJson), J.Req(o, "last", SurfaceID.FromJson),
                    J.OptVal(o, "pivot", SurfaceID.FromJson),
                    J.Req(o, "component", n => Wire.Read<PlacementComponent>(n, "placement component")));
            case "aperture_value" or "temperature_c" or "pressure_atm":
                J.AllowOnly(o, "param");
                return tag == "aperture_value" ? new ApertureValue() : tag == "temperature_c" ? new TemperatureC() : new PressureAtm();
            default: throw J.Fail($"unknown parameter '{tag}'");
        }
    }

    public JsonNode ToJson()
    {
        var (tag, index) = Parts;
        var o = new JsonObject { ["param"] = tag };
        if (SurfaceId is { } surface) o["surface"] = surface.ToJson();
        switch (this)
        {
            case ShapeCoefficient p: o["set"] = Wire.Write(p.Set); break;
            case ShapeParam p: o["parameter"] = Wire.Write(p.Parameter); break;
            case Irregularity p: o["mode"] = Wire.Write(p.Mode); break;
            case ElementPlacement p:
                o["last"] = p.Last.ToJson();
                o["pivot"] = p.Pivot?.ToJson();
                o["component"] = Wire.Write(p.Component);
                break;
        }
        if (index is { } i) o["index"] = i;
        return o;
    }

    /// <summary>Field index of a field-addressed path.</summary>
    public int? FieldIndex => this switch
    {
        FieldX p => p.Index,
        FieldY p => p.Index,
        FieldWeight p => p.Index,
        _ => null,
    };

    /// <summary>Wavelength index of a wavelength-addressed path.</summary>
    public int? WavelengthIndex => this switch
    {
        WavelengthUm p => p.Index,
        WavelengthWeight p => p.Index,
        _ => null,
    };

    /// <summary>The same parameter for another field or wavelength index.</summary>
    internal ParamPath WithIndex(int i) => this switch
    {
        FieldX => new FieldX(i),
        FieldY => new FieldY(i),
        FieldWeight => new FieldWeight(i),
        WavelengthUm => new WavelengthUm(i),
        WavelengthWeight => new WavelengthWeight(i),
        _ => this,
    };
}

/// <summary>A parameter override of a <see cref="Configuration"/>. JSON: <c>[path, value]</c>.</summary>
public readonly record struct ParamOverride(ParamPath Path, double Value)
{
    public static ParamOverride FromJson(JsonNode? node)
    {
        var (a, b) = J.Pair(node);
        return new ParamOverride(ParamPath.FromJson(a), J.D(b));
    }

    public JsonNode ToJson() => J.PairNode(Path.ToJson(), J.Num(Value));
}

/// <summary>One configuration of a multi-configuration system: parameter overrides on top of the base system.</summary>
[JsonConverter(typeof(ModelConverter<Configuration>))]
public sealed record Configuration : IJsonModel<Configuration>
{
    public string Name { get; init; } = "";
    public ValueList<ParamOverride> Overrides { get; init; } = ValueList<ParamOverride>.Empty;

    public static Configuration FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        J.AllowOnly(o, "name", "overrides");
        return new Configuration
        {
            Name = J.Def(o, "name", J.S, ""),
            Overrides = J.Def(o, "overrides", n => J.Arr(n, ParamOverride.FromJson), ValueList<ParamOverride>.Empty),
        };
    }

    public JsonNode ToJson() => new JsonObject { ["name"] = Name, ["overrides"] = J.Arr(Overrides, x => x.ToJson()) };
}
