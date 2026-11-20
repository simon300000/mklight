using System.Reflection;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;

namespace MKLight.Model;

/// <summary>One of the engine's built-in samples (<c>mklight_core::samples</c>).</summary>
[JsonConverter(typeof(ModelConverter<SampleEntry>))]
public sealed record SampleEntry(string Id, OpticalSystem System) : IJsonModel<SampleEntry>
{
    public static SampleEntry FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        return new SampleEntry(J.Req(o, "id", J.S), J.Req(o, "system", OpticalSystem.FromJson));
    }

    public JsonNode ToJson() => new JsonObject { ["id"] = Id, ["system"] = System.ToJson() };
}

/// <summary>Sample systems for new documents and tests.</summary>
public static class SampleSystems
{
    /// <summary>
    /// The text of the embedded <c>samples.json</c>, which <c>cargo run -p mklight-core --example dump_fixtures</c> writes.
    /// </summary>
    public static string? BundledText { get; } = ReadResource("MKLight.Model.samples.json");

    private static string? ReadResource(string name)
    {
        using var stream = typeof(SampleSystems).Assembly.GetManifestResourceStream(name);
        if (stream is null) return null;
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }

    /// <summary>The engine's built-in samples. They validate in the engine and use only built-in glasses.</summary>
    public static ValueList<SampleEntry> Bundled { get; } =
        BundledText is { } text ? ModelJson.DecodeList<SampleEntry>(text) : ValueList<SampleEntry>.Empty;

    /// <summary>A bundled sample by identifier (e.g. <c>achromat</c>).</summary>
    public static OpticalSystem? Get(string id) => Bundled.FirstOrDefault(e => e.Id == id)?.System;

    /// <summary>The classic Cooke triplet (F/4.7, EFL ≈ 47.1 mm, ±20° field). Identical to the <c>cooke_triplet.json</c> engine fixture.</summary>
    public static OpticalSystem CookeTriplet()
    {
        static Surface S(ulong id, double r, double t, MaterialRef m, bool coated = true) => new()
        {
            Id = new SurfaceID(id),
            Shape = new ConicShape(RadiusDisplay.CurvatureFromRadius(r), 0),
            Thickness = t,
            Material = m,
            Coating = coated ? "AR" : null,
        };
        return new OpticalSystem
        {
            Name = "Cooke triplet",
            Notes = "Classic three-element anastigmat, EFL ≈ 47.1 mm, F/4.7, image at the paraxial focus.",
            Aperture = new ApertureSpec.EntrancePupilDiameter(10),
            Fields = new FieldSet { Kind = FieldKind.ObjectAngle, Fields = [new Field { Y = 0 }, new Field { Y = 14 }, new Field { Y = 20 }] },
            Surfaces =
            [
                new Surface { Id = new SurfaceID(0), Thickness = double.PositiveInfinity },
                S(1, 22.01359, 3.25896, MaterialRef.Glass("N-SK16")),
                S(2, -435.76044, 6.00755, new MaterialRef.Air()),
                S(3, -22.21328, 0.99997, MaterialRef.Glass("F2")),
                S(4, 20.29192, 4.75041, new MaterialRef.Air()),
                new Surface { Id = new SurfaceID(5), Label = "STOP", Thickness = 2.0 },
                S(6, 79.68360, 2.95208, MaterialRef.Glass("N-SK16")),
                S(7, -18.39533, 41.36144, new MaterialRef.Air()),
                new Surface { Id = new SurfaceID(8), Thickness = 0 },
            ],
            Stop = 5,
            Coatings = ValueMap<CoatingDef>.Empty.Set("AR", new CoatingDef.Ideal(0.005, 0.995)),
        };
    }

    /// <summary>A biconvex N-BK7 singlet, f ≈ 100 mm, object at infinity, stop on the first surface.</summary>
    public static OpticalSystem Singlet() => new()
    {
        Name = "Singlet",
        Aperture = new ApertureSpec.EntrancePupilDiameter(20),
        Fields = new FieldSet { Kind = FieldKind.ObjectAngle, Fields = [new Field { Y = 0 }, new Field { Y = 3.5 }, new Field { Y = 5 }] },
        Surfaces =
        [
            new Surface { Id = new SurfaceID(0), Thickness = double.PositiveInfinity },
            new Surface
            {
                Id = new SurfaceID(1), Label = "STOP", Shape = new ConicShape(1 / 103.4, 0), Thickness = 5, Material = MaterialRef.Glass("N-BK7"),
            },
            new Surface { Id = new SurfaceID(2), Shape = new ConicShape(-1 / 103.4, 0), Thickness = 97.5 },
            new Surface { Id = new SurfaceID(3), Thickness = 0 },
        ],
        Stop = 1,
    };
}
