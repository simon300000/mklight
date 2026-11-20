using System.Text.Json.Nodes;
using System.Text.Json.Serialization;

namespace MKLight.Model;

/// <summary>Surface scatter (BSDF) models; values per unit projected solid angle (sr⁻¹).</summary>
[JsonConverter(typeof(ModelConverter<ScatterModel>))]
public abstract record ScatterModel : IJsonModel<ScatterModel>
{
    /// <summary>Lambertian with total integrated scatter <c>Tis</c>.</summary>
    public sealed record Lambertian(double Tis) : ScatterModel;

    /// <summary>Harvey–Shack ABg: <c>A / (B + |β−β₀|^g)</c>.</summary>
    public sealed record Abg(double A, double B, double G) : ScatterModel;

    /// <summary>Gaussian microfacets with RMS slope (radians).</summary>
    public sealed record Gaussian(double Tis, double SigmaSlope) : ScatterModel;

    public static ScatterModel FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        switch (J.Tag(o, "type"))
        {
            case "lambertian":
                J.AllowOnly(o, "type", "tis");
                return new Lambertian(J.Req(o, "tis", J.D));
            case "abg":
                J.AllowOnly(o, "type", "a", "b", "g");
                return new Abg(J.Req(o, "a", J.D), J.Req(o, "b", J.D), J.Req(o, "g", J.D));
            case "gaussian":
                J.AllowOnly(o, "type", "tis", "sigma_slope");
                return new Gaussian(J.Req(o, "tis", J.D), J.Req(o, "sigma_slope", J.D));
            case var t: throw J.Fail($"unknown scatter model '{t}'");
        }
    }

    public JsonNode ToJson() => this switch
    {
        Lambertian l => new JsonObject { ["type"] = "lambertian", ["tis"] = J.Num(l.Tis) },
        Abg a => new JsonObject { ["type"] = "abg", ["a"] = J.Num(a.A), ["b"] = J.Num(a.B), ["g"] = J.Num(a.G) },
        Gaussian g => new JsonObject { ["type"] = "gaussian", ["tis"] = J.Num(g.Tis), ["sigma_slope"] = J.Num(g.SigmaSlope) },
        _ => throw new InvalidOperationException(),
    };
}

/// <summary>Non-optical geometry for the stray-light solver.</summary>
[JsonConverter(typeof(ModelConverter<MechanicalPart>))]
public abstract record MechanicalPart : IJsonModel<MechanicalPart>
{
    /// <summary>Cylindrical tube between two surfaces' vertex planes.</summary>
    public sealed record Barrel(SurfaceID From, SurfaceID To, double InnerRadius, ScatterModel Finish) : MechanicalPart;

    /// <summary>Flat annular baffle at local <c>z</c> in a surface's frame.</summary>
    public sealed record AnnularBaffle(SurfaceID Surface, double Z, double InnerRadius, double OuterRadius, ScatterModel Finish)
        : MechanicalPart;

    /// <summary>Surfaces the part is attached to.</summary>
    public IReadOnlyList<SurfaceID> Surfaces => this switch
    {
        Barrel b => [b.From, b.To],
        AnnularBaffle a => [a.Surface],
        _ => [],
    };

    public static MechanicalPart FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        switch (J.Tag(o, "type"))
        {
            case "barrel":
                J.AllowOnly(o, "type", "from", "to", "inner_radius", "finish");
                return new Barrel(
                    J.Req(o, "from", SurfaceID.FromJson), J.Req(o, "to", SurfaceID.FromJson), J.Req(o, "inner_radius", J.D),
                    J.Req(o, "finish", ScatterModel.FromJson));
            case "annular_baffle":
                J.AllowOnly(o, "type", "surface", "z", "inner_radius", "outer_radius", "finish");
                return new AnnularBaffle(
                    J.Req(o, "surface", SurfaceID.FromJson), J.Req(o, "z", J.D), J.Req(o, "inner_radius", J.D),
                    J.Req(o, "outer_radius", J.D), J.Req(o, "finish", ScatterModel.FromJson));
            case var t: throw J.Fail($"unknown mechanical part '{t}'");
        }
    }

    public JsonNode ToJson() => this switch
    {
        Barrel b => new JsonObject
        {
            ["type"] = "barrel", ["from"] = b.From.ToJson(), ["to"] = b.To.ToJson(), ["inner_radius"] = J.Num(b.InnerRadius),
            ["finish"] = b.Finish.ToJson(),
        },
        AnnularBaffle a => new JsonObject
        {
            ["type"] = "annular_baffle", ["surface"] = a.Surface.ToJson(), ["z"] = J.Num(a.Z), ["inner_radius"] = J.Num(a.InnerRadius),
            ["outer_radius"] = J.Num(a.OuterRadius), ["finish"] = a.Finish.ToJson(),
        },
        _ => throw new InvalidOperationException(),
    };
}

/// <summary>Image sensor at the image surface.</summary>
[JsonConverter(typeof(ModelConverter<Sensor>))]
public sealed record Sensor : IJsonModel<Sensor>
{
    public required uint PixelsX { get; init; }
    public required uint PixelsY { get; init; }

    /// <summary>Pixel pitch (mm).</summary>
    public required double PitchX { get; init; }

    public required double PitchY { get; init; }

    /// <summary>Largest accepted chief-ray angle (degrees).</summary>
    public double? MaxCraDeg { get; init; }

    /// <summary>Relative spectral response <c>(λ µm, response)</c>; empty means flat.</summary>
    public ValueList<SpectralSample> SpectralResponse { get; init; } = ValueList<SpectralSample>.Empty;

    public double HalfWidth => 0.5 * PixelsX * PitchX;

    public static Sensor FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        J.AllowOnly(o, "pixels_x", "pixels_y", "pitch_x", "pitch_y", "max_cra_deg", "spectral_response");
        return new Sensor
        {
            PixelsX = J.Req(o, "pixels_x", J.U32),
            PixelsY = J.Req(o, "pixels_y", J.U32),
            PitchX = J.Req(o, "pitch_x", J.D),
            PitchY = J.Req(o, "pitch_y", J.D),
            MaxCraDeg = J.OptVal(o, "max_cra_deg", J.D),
            SpectralResponse = J.Def(o, "spectral_response", n => J.Arr(n, SpectralSample.FromJson), ValueList<SpectralSample>.Empty),
        };
    }

    public JsonNode ToJson() => new JsonObject
    {
        ["pixels_x"] = PixelsX,
        ["pixels_y"] = PixelsY,
        ["pitch_x"] = J.Num(PitchX),
        ["pitch_y"] = J.Num(PitchY),
        ["max_cra_deg"] = MaxCraDeg is { } m ? J.Num(m) : null,
        ["spectral_response"] = J.Arr(SpectralResponse, s => s.ToJson()),
    };
}
