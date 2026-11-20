using System.Globalization;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;

namespace MKLight.Model;

/// <summary>Reference to an optical medium. The engine resolves it to numbers.</summary>
[JsonConverter(typeof(ModelConverter<MaterialRef>))]
public abstract record MaterialRef : IJsonModel<MaterialRef>
{
    /// <summary>Air at the system temperature and pressure (index 1 by convention).</summary>
    public sealed record Air : MaterialRef;

    /// <summary>Vacuum (index <c>1/n_air</c>).</summary>
    public sealed record Vacuum : MaterialRef;

    /// <summary>
    /// A catalog glass; <see cref="Catalog"/> <c>null</c> searches every loaded catalog.
    /// <see cref="Offset"/> shifts its <c>n_d</c> and <c>V_d</c> (docs/MATERIALS.md).
    /// </summary>
    public sealed record CatalogGlass : MaterialRef
    {
        public string? Catalog { get; init; }
        public required string Name { get; init; }
        public MaterialOffset Offset { get; init; }
    }

    /// <summary>Model glass from <c>n_d</c>, <c>V_d</c> and <c>ΔP_g,F</c>.</summary>
    public sealed record ModelGlass(double Nd, double Vd, double Dpgf) : MaterialRef;

    /// <summary>Constant index at every wavelength.</summary>
    public sealed record ConstantIndex(double N) : MaterialRef;

    public static MaterialRef Glass(string name) => new CatalogGlass { Name = name };

    public bool IsAir => this is Air;

    /// <summary>Short name shown in tables (matches the engine's <c>display_name</c>).</summary>
    public string DisplayName => this switch
    {
        Air => "AIR",
        Vacuum => "VACUUM",
        CatalogGlass g => g.Name,
        ModelGlass m => m.Nd.ToString("F5", CultureInfo.InvariantCulture) + "/" + m.Vd.ToString("F2", CultureInfo.InvariantCulture),
        ConstantIndex c => "n=" + NumberDisplay.Significant(c.N),
        _ => "",
    };

    public static MaterialRef FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        switch (J.Tag(o, "type"))
        {
            case "air":
                J.AllowOnly(o, "type");
                return new Air();
            case "vacuum":
                J.AllowOnly(o, "type");
                return new Vacuum();
            case "catalog":
                J.AllowOnly(o, "type", "catalog", "name", "offset");
                return new CatalogGlass
                {
                    Catalog = J.OptRef(o, "catalog", J.S),
                    Name = J.Req(o, "name", J.S),
                    Offset = J.Def(o, "offset", MaterialOffset.FromJson, MaterialOffset.Zero),
                };
            case "model":
                J.AllowOnly(o, "type", "nd", "vd", "dpgf");
                return new ModelGlass(J.Req(o, "nd", J.D), J.Req(o, "vd", J.D), J.Def(o, "dpgf", J.D, 0));
            case "constant":
                J.AllowOnly(o, "type", "n");
                return new ConstantIndex(J.Req(o, "n", J.D));
            case var t: throw J.Fail($"unknown material '{t}'");
        }
    }

    public JsonNode ToJson()
    {
        switch (this)
        {
            case Air: return new JsonObject { ["type"] = "air" };
            case Vacuum: return new JsonObject { ["type"] = "vacuum" };
            case CatalogGlass g:
                var o = new JsonObject { ["type"] = "catalog", ["catalog"] = g.Catalog, ["name"] = g.Name };
                if (!g.Offset.IsZero) o["offset"] = g.Offset.ToJson();
                return o;
            case ModelGlass m:
                return new JsonObject { ["type"] = "model", ["nd"] = J.Num(m.Nd), ["vd"] = J.Num(m.Vd), ["dpgf"] = J.Num(m.Dpgf) };
            case ConstantIndex c: return new JsonObject { ["type"] = "constant", ["n"] = J.Num(c.N) };
            default: throw new InvalidOperationException();
        }
    }
}

/// <summary>
/// Changes <c>Δn_d</c>, <c>ΔV_d</c> to a catalog glass; its dispersion is scaled about
/// d, keeping partial-dispersion ratios. Omitted from JSON when zero.
/// </summary>
public readonly record struct MaterialOffset(double DnD = 0, double DvD = 0)
{
    public static MaterialOffset Zero => default;

    /// <summary>Structural bound on <c>|Δn_d|</c>.</summary>
    public const double MaxDnD = 0.05;

    /// <summary>Structural bound on <c>|ΔV_d|</c>.</summary>
    public const double MaxDvD = 10.0;

    public bool IsZero => DnD == 0 && DvD == 0;

    public static MaterialOffset FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        J.AllowOnly(o, "dn_d", "dv_d");
        return new MaterialOffset(J.Def(o, "dn_d", J.D, 0), J.Def(o, "dv_d", J.D, 0));
    }

    public JsonNode ToJson() => new JsonObject { ["dn_d"] = J.Num(DnD), ["dv_d"] = J.Num(DvD) };
}

/// <summary><c>(wavelength µm, value)</c> sample, as in coating reflectance tables and sensor spectral responses. JSON: <c>[um, value]</c>.</summary>
public readonly record struct SpectralSample(double Um, double Value)
{
    public static SpectralSample FromJson(JsonNode? node)
    {
        var (a, b) = J.Pair(node);
        return new SpectralSample(J.D(a), J.D(b));
    }

    public JsonNode ToJson() => J.PairNode(J.Num(Um), J.Num(Value));
}

/// <summary>Complex refractive index <c>n + ik</c>, serialised as <c>[n, k]</c>.</summary>
public readonly record struct ComplexIndex(double N, double K)
{
    public static ComplexIndex FromJson(JsonNode? node)
    {
        var (a, b) = J.Pair(node);
        return new ComplexIndex(J.D(a), J.D(b));
    }

    public JsonNode ToJson() => J.PairNode(J.Num(N), J.Num(K));
}

/// <summary>One thin-film layer.</summary>
[JsonConverter(typeof(ModelConverter<CoatingLayer>))]
public sealed record CoatingLayer(MaterialRef Material, double ThicknessUm = 0, double? QuarterWaves = null, double Extinction = 0)
    : IJsonModel<CoatingLayer>
{
    public static CoatingLayer FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        J.AllowOnly(o, "material", "thickness_um", "quarter_waves", "extinction");
        return new CoatingLayer(
            J.Req(o, "material", MaterialRef.FromJson),
            J.Def(o, "thickness_um", J.D, 0),
            J.OptVal(o, "quarter_waves", J.D),
            J.Def(o, "extinction", J.D, 0));
    }

    public JsonNode ToJson() => new JsonObject
    {
        ["material"] = Material.ToJson(),
        ["thickness_um"] = J.Num(ThicknessUm),
        ["quarter_waves"] = QuarterWaves is { } q ? J.Num(q) : null,
        ["extinction"] = J.Num(Extinction),
    };
}

/// <summary>
/// Which medium of a refracting surface carries a thin film, fixed once per
/// surface. <c>HigherIndex</c> (the default, omitted from files) is the denser
/// medium at the coating's reference wavelength.
/// </summary>
public enum CoatingSubstrate
{
    [Wire("higher_index")] HigherIndex,
    [Wire("before")] Before,
    [Wire("after")] After,
}

/// <summary>A named coating definition.</summary>
[JsonConverter(typeof(ModelConverter<CoatingDef>))]
public abstract record CoatingDef : IJsonModel<CoatingDef>
{
    /// <summary>Multilayer stack, layers listed outward from the substrate.</summary>
    public sealed record ThinFilm(
        double ReferenceWavelengthUm, ValueList<CoatingLayer> Layers, ComplexIndex? MirrorSubstrate = null,
        CoatingSubstrate Substrate = CoatingSubstrate.HigherIndex) : CoatingDef;

    /// <summary>Fixed reflectance and transmittance; absorption <c>1 − R − T</c>.</summary>
    public sealed record Ideal(double Reflectance, double Transmittance) : CoatingDef;

    /// <summary>Tabulated normal-incidence reflectance <c>(λ µm, R)</c>; <c>T = 1 − R</c>.</summary>
    public sealed record Tabulated(ValueList<SpectralSample> Reflectance) : CoatingDef;

    public static CoatingDef FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        switch (J.Tag(o, "type"))
        {
            case "thin_film":
                J.AllowOnly(o, "type", "reference_wavelength_um", "layers", "mirror_substrate", "substrate");
                return new ThinFilm(
                    J.Req(o, "reference_wavelength_um", J.D),
                    J.Req(o, "layers", n => J.Arr(n, CoatingLayer.FromJson)),
                    J.OptVal(o, "mirror_substrate", ComplexIndex.FromJson),
                    J.OptVal(o, "substrate", n => Wire.Read<CoatingSubstrate>(n, "coating substrate")) ?? CoatingSubstrate.HigherIndex);
            case "ideal":
                J.AllowOnly(o, "type", "reflectance", "transmittance");
                return new Ideal(J.Req(o, "reflectance", J.D), J.Req(o, "transmittance", J.D));
            case "tabulated":
                J.AllowOnly(o, "type", "reflectance");
                return new Tabulated(J.Req(o, "reflectance", n => J.Arr(n, SpectralSample.FromJson)));
            case var t: throw J.Fail($"unknown coating '{t}'");
        }
    }

    public JsonNode ToJson()
    {
        switch (this)
        {
            case ThinFilm f:
                var o = new JsonObject
                {
                    ["type"] = "thin_film",
                    ["reference_wavelength_um"] = J.Num(f.ReferenceWavelengthUm),
                    ["layers"] = J.Arr(f.Layers, l => l.ToJson()),
                    ["mirror_substrate"] = f.MirrorSubstrate?.ToJson(),
                };
                if (f.Substrate != CoatingSubstrate.HigherIndex) o["substrate"] = Wire.Write(f.Substrate);
                return o;
            case Ideal i:
                return new JsonObject { ["type"] = "ideal", ["reflectance"] = J.Num(i.Reflectance), ["transmittance"] = J.Num(i.Transmittance) };
            case Tabulated t:
                return new JsonObject { ["type"] = "tabulated", ["reflectance"] = J.Arr(t.Reflectance, s => s.ToJson()) };
            default: throw new InvalidOperationException();
        }
    }
}
