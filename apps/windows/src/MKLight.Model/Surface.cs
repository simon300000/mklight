using System.Text.Json.Nodes;
using System.Text.Json.Serialization;

namespace MKLight.Model;

/// <summary>
/// Stable identity of a surface, independent of its index. The UI keys
/// selection and view state on it so they survive inserts and deletes.
/// </summary>
[JsonConverter(typeof(ModelConverter<SurfaceID>))]
public readonly record struct SurfaceID(ulong Value) : IJsonModel<SurfaceID>, IComparable<SurfaceID>
{
    public static SurfaceID FromJson(JsonNode? node) => new(J.U64(node));

    public JsonNode ToJson() => JsonValue.Create(Value);

    public int CompareTo(SurfaceID other) => Value.CompareTo(other.Value);

    public override string ToString() => $"#{Value}";
}

/// <summary>What the surface does to light.</summary>
public enum Interaction
{
    [Wire("refract")] Refract,

    /// <summary>Specular mirror; the medium after is the medium before.</summary>
    [Wire("reflect")] Reflect,

    /// <summary>No interaction (dummy/reference surface).</summary>
    [Wire("dummy")] Dummy,
}

public enum TransformOrder
{
    [Wire("decenter_then_tilt")] DecenterThenTilt,
    [Wire("tilt_then_decenter")] TiltThenDecenter,
}

/// <summary>Clear-aperture size.</summary>
[JsonConverter(typeof(ModelConverter<SemiDiameter>))]
public abstract record SemiDiameter : IJsonModel<SemiDiameter>
{
    /// <summary>Computed by the tracer from the largest ray height; does not clip.</summary>
    public sealed record Auto : SemiDiameter;

    /// <summary>User-fixed radius (mm) that clips rays.</summary>
    public sealed record Fixed(double Value) : SemiDiameter;

    public double? FixedValue => this is Fixed f ? f.Value : null;

    public static SemiDiameter FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        switch (J.Tag(o, "mode"))
        {
            case "auto":
                J.AllowOnly(o, "mode");
                return new Auto();
            case "fixed":
                J.AllowOnly(o, "mode", "value");
                return new Fixed(J.Req(o, "value", J.D));
            case var m: throw J.Fail($"unknown semi-diameter mode '{m}'");
        }
    }

    public JsonNode ToJson() => this switch
    {
        Fixed f => new JsonObject { ["mode"] = "fixed", ["value"] = J.Num(f.Value) },
        _ => new JsonObject { ["mode"] = "auto" },
    };
}

/// <summary>Explicit clipping aperture; overrides the circular semi-diameter.</summary>
[JsonConverter(typeof(ModelConverter<ApertureShape>))]
public abstract record ApertureShape : IJsonModel<ApertureShape>
{
    public sealed record Circular(double Radius) : ApertureShape;

    public sealed record Annular(double InnerRadius, double OuterRadius) : ApertureShape;

    public sealed record Rectangular(double HalfX, double HalfY) : ApertureShape;

    public sealed record Elliptical(double SemiX, double SemiY) : ApertureShape;

    /// <summary>Radius of the smallest enclosing circle.</summary>
    public double BoundingRadius => this switch
    {
        Circular c => c.Radius,
        Annular a => a.OuterRadius,
        Rectangular r => double.Hypot(r.HalfX, r.HalfY),
        Elliptical e => Math.Max(e.SemiX, e.SemiY),
        _ => 0,
    };

    public static ApertureShape FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        switch (J.Tag(o, "type"))
        {
            case "circular":
                J.AllowOnly(o, "type", "radius");
                return new Circular(J.Req(o, "radius", J.D));
            case "annular":
                J.AllowOnly(o, "type", "inner_radius", "outer_radius");
                return new Annular(J.Req(o, "inner_radius", J.D), J.Req(o, "outer_radius", J.D));
            case "rectangular":
                J.AllowOnly(o, "type", "half_x", "half_y");
                return new Rectangular(J.Req(o, "half_x", J.D), J.Req(o, "half_y", J.D));
            case "elliptical":
                J.AllowOnly(o, "type", "semi_x", "semi_y");
                return new Elliptical(J.Req(o, "semi_x", J.D), J.Req(o, "semi_y", J.D));
            case var t: throw J.Fail($"unknown aperture '{t}'");
        }
    }

    public JsonNode ToJson() => this switch
    {
        Circular c => new JsonObject { ["type"] = "circular", ["radius"] = J.Num(c.Radius) },
        Annular a => new JsonObject { ["type"] = "annular", ["inner_radius"] = J.Num(a.InnerRadius), ["outer_radius"] = J.Num(a.OuterRadius) },
        Rectangular r => new JsonObject { ["type"] = "rectangular", ["half_x"] = J.Num(r.HalfX), ["half_y"] = J.Num(r.HalfY) },
        Elliptical e => new JsonObject { ["type"] = "elliptical", ["semi_x"] = J.Num(e.SemiX), ["semi_y"] = J.Num(e.SemiY) },
        _ => throw new InvalidOperationException(),
    };
}

/// <summary>
/// Rigid placement change applied before the surface (CONVENTIONS §2):
/// decenters in mm, tilts in degrees about x, then y, then z.
/// </summary>
[JsonConverter(typeof(ModelConverter<Placement>))]
public sealed record Placement : IJsonModel<Placement>
{
    public double DecenterX { get; init; }
    public double DecenterY { get; init; }
    public double TiltX { get; init; }
    public double TiltY { get; init; }
    public double TiltZ { get; init; }
    public TransformOrder Order { get; init; } = TransformOrder.DecenterThenTilt;

    /// <summary>Return to the pre-placement frame after the surface.</summary>
    public bool RestoreAfter { get; init; }

    public bool IsIdentity => DecenterX == 0 && DecenterY == 0 && TiltX == 0 && TiltY == 0 && TiltZ == 0;

    public static Placement FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        J.AllowOnly(o, "decenter_x", "decenter_y", "tilt_x", "tilt_y", "tilt_z", "order", "restore_after");
        return new Placement
        {
            DecenterX = J.Def(o, "decenter_x", J.D, 0),
            DecenterY = J.Def(o, "decenter_y", J.D, 0),
            TiltX = J.Def(o, "tilt_x", J.D, 0),
            TiltY = J.Def(o, "tilt_y", J.D, 0),
            TiltZ = J.Def(o, "tilt_z", J.D, 0),
            Order = J.Def(o, "order", n => Wire.Read<TransformOrder>(n, "transform order"), TransformOrder.DecenterThenTilt),
            RestoreAfter = J.Def(o, "restore_after", J.B, false),
        };
    }

    public JsonNode ToJson() => new JsonObject
    {
        ["decenter_x"] = J.Num(DecenterX),
        ["decenter_y"] = J.Num(DecenterY),
        ["tilt_x"] = J.Num(TiltX),
        ["tilt_y"] = J.Num(TiltY),
        ["tilt_z"] = J.Num(TiltZ),
        ["order"] = Wire.Write(Order),
        ["restore_after"] = RestoreAfter,
    };
}

/// <summary>
/// Rigid motion of the surfaces from this one through <see cref="Last"/>, in this
/// surface's axes, about the <see cref="Pivot"/> vertex (<c>null</c>: this one).
/// Surfaces after the range keep their place (CONVENTIONS §2).
/// <c>Motion.Order</c> and <c>Motion.RestoreAfter</c> have no effect here.
/// </summary>
[JsonConverter(typeof(ModelConverter<ElementPlacement>))]
public sealed record ElementPlacement : IJsonModel<ElementPlacement>
{
    public required SurfaceID Last { get; init; }
    public SurfaceID? Pivot { get; init; }
    public Placement Motion { get; init; } = new();

    public static ElementPlacement FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        J.AllowOnly(o, "last", "pivot", "motion");
        return new ElementPlacement
        {
            Last = J.Req(o, "last", SurfaceID.FromJson),
            Pivot = J.OptVal(o, "pivot", SurfaceID.FromJson),
            Motion = J.Req(o, "motion", Placement.FromJson),
        };
    }

    public JsonNode ToJson() => new JsonObject { ["last"] = Last.ToJson(), ["pivot"] = Pivot?.ToJson(), ["motion"] = Motion.ToJson() };
}

/// <summary>Mechanical description of a lens edge (stray-light geometry).</summary>
[JsonConverter(typeof(ModelConverter<EdgeSpec>))]
public sealed record EdgeSpec : IJsonModel<EdgeSpec>
{
    /// <summary><c>null</c> means the clear aperture plus the default edge margin.</summary>
    public double? MechanicalSemiDiameter { get; init; }

    public ScatterModel? EdgeScatter { get; init; }

    public static EdgeSpec FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        J.AllowOnly(o, "mechanical_semi_diameter", "edge_scatter");
        return new EdgeSpec
        {
            MechanicalSemiDiameter = J.OptVal(o, "mechanical_semi_diameter", J.D),
            EdgeScatter = J.OptRef(o, "edge_scatter", ScatterModel.FromJson),
        };
    }

    public JsonNode ToJson() => new JsonObject
    {
        ["mechanical_semi_diameter"] = MechanicalSemiDiameter is { } m ? J.Num(m) : null,
        ["edge_scatter"] = EdgeScatter?.ToJson(),
    };
}

/// <summary>
/// Scalar order efficiency of a diffractive surface (SURFACES §3): which
/// share of the light goes into each diffraction order. <c>Ideal</c> puts it all
/// into the trace order; a <c>Kinoform</c> (continuous blaze) and a
/// <c>Multilevel</c> staircase of N levels spread it by the thin-element Fourier model.
/// </summary>
[JsonConverter(typeof(ModelConverter<DiffractionEfficiency>))]
public abstract record DiffractionEfficiency : IJsonModel<DiffractionEfficiency>
{
    public sealed record Ideal : DiffractionEfficiency;

    public sealed record Kinoform : DiffractionEfficiency;

    public sealed record Multilevel(int Levels) : DiffractionEfficiency;

    public static DiffractionEfficiency FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        switch (J.Tag(o, "type"))
        {
            case "ideal":
                J.AllowOnly(o, "type");
                return new Ideal();
            case "kinoform":
                J.AllowOnly(o, "type");
                return new Kinoform();
            case "multilevel":
                J.AllowOnly(o, "type", "levels");
                var n = J.Req(o, "levels", J.I32);
                if (n is < 1 or > ushort.MaxValue) throw J.Fail("levels must be 1 … 65535");
                return new Multilevel(n);
            case var t: throw J.Fail($"unknown diffraction efficiency '{t}'");
        }
    }

    public JsonNode ToJson() => this switch
    {
        Multilevel m => new JsonObject { ["type"] = "multilevel", ["levels"] = m.Levels },
        Kinoform => new JsonObject { ["type"] = "kinoform" },
        _ => new JsonObject { ["type"] = "ideal" },
    };

    /// <summary>
    /// The efficiency of <paramref name="order"/> for a relief blazed at <paramref name="blaze"/>
    /// (α·m₀: the design order times the relief's optical path at this wavelength over
    /// its design path, SURFACES §3.1), <paramref name="selected"/> being the trace order.
    /// For the inspector's readout; the engine traces with its own.
    /// </summary>
    public double Efficiency(int order, int selected, double blaze)
    {
        static double Sinc(double x)
        {
            if (x == 0) return 1;
            var r = x - Math.Round(x, MidpointRounding.AwayFromZero);
            return Math.Sin(Math.PI * r) / (Math.PI * x);
        }
        var m = (double)order;
        switch (this)
        {
            case Kinoform: return Math.Pow(Sinc(m - blaze), 2);
            case Multilevel ml:
                var n = (double)ml.Levels;
                var r = (blaze - m) / n;
                r -= Math.Round(r, MidpointRounding.AwayFromZero);
                return Math.Pow(Sinc(m / n) * Sinc(n * r) / Sinc(r), 2);
            default: return order == selected ? 1 : 0;
        }
    }
}

/// <summary>One surface of the sequential train.</summary>
[JsonConverter(typeof(ModelConverter<Surface>))]
public sealed partial record Surface : IJsonModel<Surface>
{
    public required SurfaceID Id { get; init; }
    public string Label { get; init; } = "";
    public SurfaceShape Shape { get; init; } = new PlaneShape();

    /// <summary>Distance to the next vertex (CONVENTIONS §2); may be ±∞ for the object.</summary>
    public double Thickness { get; init; }

    /// <summary>Medium after this surface.</summary>
    public MaterialRef Material { get; init; } = new MaterialRef.Air();

    public Interaction Interaction { get; init; } = Interaction.Refract;
    public SemiDiameter SemiDiameter { get; init; } = new SemiDiameter.Auto();
    public ApertureShape? Aperture { get; init; }
    public Placement? Placement { get; init; }

    /// <summary>Rigid motion of a surface range starting here (tolerancing).</summary>
    public ElementPlacement? ElementPlacement { get; init; }

    /// <summary>Name of a coating in <c>OpticalSystem.Coatings</c>; <c>null</c> is a bare interface.</summary>
    public string? Coating { get; init; }

    public ScatterModel? Scatter { get; init; }

    /// <summary>Order efficiency model; meaningful on diffractive surfaces only.</summary>
    public DiffractionEfficiency DiffractionEfficiency { get; init; } = new DiffractionEfficiency.Ideal();

    public EdgeSpec Edge { get; init; } = new();

    public bool IsCoordinateBreak => Shape is CoordinateBreakShape;

    public static Surface FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        J.AllowOnly(
            o, "id", "label", "shape", "thickness", "material", "interaction", "semi_diameter", "aperture", "placement",
            "element_placement", "coating", "scatter", "diffraction_efficiency", "edge");
        return new Surface
        {
            Id = J.Req(o, "id", SurfaceID.FromJson),
            Label = J.Def(o, "label", J.S, ""),
            Shape = J.Req(o, "shape", SurfaceShape.FromJson),
            Thickness = J.Req(o, "thickness", J.Ext),
            Material = J.Req(o, "material", MaterialRef.FromJson),
            Interaction = J.Def(o, "interaction", n => Wire.Read<Interaction>(n, "interaction"), Interaction.Refract),
            SemiDiameter = J.Def(o, "semi_diameter", Model.SemiDiameter.FromJson, new SemiDiameter.Auto()),
            Aperture = J.OptRef(o, "aperture", ApertureShape.FromJson),
            Placement = J.OptRef(o, "placement", Model.Placement.FromJson),
            ElementPlacement = J.OptRef(o, "element_placement", Model.ElementPlacement.FromJson),
            Coating = J.OptRef(o, "coating", J.S),
            Scatter = J.OptRef(o, "scatter", ScatterModel.FromJson),
            DiffractionEfficiency = J.Def(o, "diffraction_efficiency", Model.DiffractionEfficiency.FromJson, new DiffractionEfficiency.Ideal()),
            Edge = J.Def(o, "edge", EdgeSpec.FromJson, new EdgeSpec()),
        };
    }

    public JsonNode ToJson()
    {
        var o = new JsonObject
        {
            ["id"] = Id.ToJson(),
            ["label"] = Label,
            ["shape"] = Shape.ToJson(),
            ["thickness"] = J.ExtNode(Thickness),
            ["material"] = Material.ToJson(),
            ["interaction"] = Wire.Write(Interaction),
            ["semi_diameter"] = SemiDiameter.ToJson(),
            ["aperture"] = Aperture?.ToJson(),
            ["placement"] = Placement?.ToJson(),
        };
        // The engine omits an absent element placement.
        if (ElementPlacement is { } e) o["element_placement"] = e.ToJson();
        o["coating"] = Coating;
        o["scatter"] = Scatter?.ToJson();
        o["diffraction_efficiency"] = DiffractionEfficiency.ToJson();
        o["edge"] = Edge.ToJson();
        return o;
    }
}
