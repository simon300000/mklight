using System.Text.Json.Nodes;
using System.Text.Json.Serialization;

namespace MKLight.Model;

/// <summary>The kind of an <see cref="ApertureSpec"/>.</summary>
public enum ApertureKind
{
    [Wire("entrance_pupil_diameter")] EntrancePupilDiameter,
    [Wire("image_space_f_number")] ImageSpaceFNumber,
    [Wire("object_space_na")] ObjectSpaceNA,
    [Wire("paraxial_working_f_number")] ParaxialWorkingFNumber,
    [Wire("float_by_stop_size")] FloatByStopSize,
}

public static class ApertureKindExtensions
{
    public static readonly IReadOnlyList<ApertureKind> All = Enum.GetValues<ApertureKind>();

    public static string Title(this ApertureKind k) => k switch
    {
        ApertureKind.EntrancePupilDiameter => "Entrance pupil diameter",
        ApertureKind.ImageSpaceFNumber => "Image-space F/#",
        ApertureKind.ObjectSpaceNA => "Object-space NA",
        ApertureKind.ParaxialWorkingFNumber => "Paraxial working F/#",
        _ => "Float by stop size",
    };

    public static string Unit(this ApertureKind k) => k == ApertureKind.EntrancePupilDiameter ? "mm" : "";
}

/// <summary>How the system aperture is specified.</summary>
[JsonConverter(typeof(ModelConverter<ApertureSpec>))]
public abstract record ApertureSpec : IJsonModel<ApertureSpec>
{
    /// <summary>Entrance pupil diameter (mm).</summary>
    public sealed record EntrancePupilDiameter(double Diameter) : ApertureSpec;

    /// <summary>Paraxial image-space F/# (<c>f'/EPD</c>), object at infinity.</summary>
    public sealed record ImageSpaceFNumber(double FNumber) : ApertureSpec;

    /// <summary>Object-space NA <c>n·sinθ</c> (finite objects).</summary>
    public sealed record ObjectSpaceNA(double NA) : ApertureSpec;

    /// <summary>Paraxial working F/# <c>1/(2 n' sinθ')</c>.</summary>
    public sealed record ParaxialWorkingFNumber(double FNumber) : ApertureSpec;

    /// <summary>The stop's fixed semi-diameter sets the aperture.</summary>
    public sealed record FloatByStopSize : ApertureSpec;

    public static ApertureSpec Default => new EntrancePupilDiameter(10);

    public ApertureKind Kind => this switch
    {
        EntrancePupilDiameter => ApertureKind.EntrancePupilDiameter,
        ImageSpaceFNumber => ApertureKind.ImageSpaceFNumber,
        ObjectSpaceNA => ApertureKind.ObjectSpaceNA,
        ParaxialWorkingFNumber => ApertureKind.ParaxialWorkingFNumber,
        _ => ApertureKind.FloatByStopSize,
    };

    /// <summary>The numeric value, <c>null</c> for <c>FloatByStopSize</c>.</summary>
    public double? Value => this switch
    {
        EntrancePupilDiameter a => a.Diameter,
        ImageSpaceFNumber a => a.FNumber,
        ObjectSpaceNA a => a.NA,
        ParaxialWorkingFNumber a => a.FNumber,
        _ => null,
    };

    public static ApertureSpec Of(ApertureKind kind, double value) => kind switch
    {
        ApertureKind.EntrancePupilDiameter => new EntrancePupilDiameter(value),
        ApertureKind.ImageSpaceFNumber => new ImageSpaceFNumber(value),
        ApertureKind.ObjectSpaceNA => new ObjectSpaceNA(value),
        ApertureKind.ParaxialWorkingFNumber => new ParaxialWorkingFNumber(value),
        _ => new FloatByStopSize(),
    };

    public static ApertureSpec FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        var t = J.Tag(o, "type");
        var kind = Wire.Parse<ApertureKind>(t) ?? throw J.Fail($"unknown aperture spec '{t}'");
        if (kind == ApertureKind.FloatByStopSize)
        {
            J.AllowOnly(o, "type");
            return new FloatByStopSize();
        }
        J.AllowOnly(o, "type", "value");
        return Of(kind, J.Req(o, "value", J.D));
    }

    public JsonNode ToJson()
    {
        var o = new JsonObject { ["type"] = Wire.Name(Kind) };
        if (Value is { } v) o["value"] = J.Num(v);
        return o;
    }
}

public enum FieldKind
{
    /// <summary>Object-space field angle (degrees).</summary>
    [Wire("object_angle")] ObjectAngle,

    /// <summary>Object height (mm), finite objects only.</summary>
    [Wire("object_height")] ObjectHeight,

    /// <summary>Paraxial image height (mm).</summary>
    [Wire("paraxial_image_height")] ParaxialImageHeight,

    /// <summary>Real image height (mm).</summary>
    [Wire("real_image_height")] RealImageHeight,
}

public static class FieldKindExtensions
{
    public static readonly IReadOnlyList<FieldKind> All = Enum.GetValues<FieldKind>();

    public static string Title(this FieldKind k) => k switch
    {
        FieldKind.ObjectAngle => "Object angle",
        FieldKind.ObjectHeight => "Object height",
        FieldKind.ParaxialImageHeight => "Paraxial image height",
        _ => "Real image height",
    };

    public static string Unit(this FieldKind k) => k == FieldKind.ObjectAngle ? "°" : "mm";
}

/// <summary>Vignetting factors (CONVENTIONS §3): <c>Py' = vdy + (1 − vcy)·Py</c>.</summary>
[JsonConverter(typeof(ModelConverter<Vignetting>))]
public sealed record Vignetting(double Vdx = 0, double Vdy = 0, double Vcx = 0, double Vcy = 0) : IJsonModel<Vignetting>
{
    public static Vignetting FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        J.AllowOnly(o, "vdx", "vdy", "vcx", "vcy");
        return new Vignetting(J.Def(o, "vdx", J.D, 0), J.Def(o, "vdy", J.D, 0), J.Def(o, "vcx", J.D, 0), J.Def(o, "vcy", J.D, 0));
    }

    public JsonNode ToJson() => new JsonObject
    {
        ["vdx"] = J.Num(Vdx), ["vdy"] = J.Num(Vdy), ["vcx"] = J.Num(Vcx), ["vcy"] = J.Num(Vcy),
    };
}

[JsonConverter(typeof(ModelConverter<Field>))]
public sealed record Field : IJsonModel<Field>
{
    public double X { get; init; }
    public double Y { get; init; }
    public double Weight { get; init; } = 1;
    public Vignetting Vignetting { get; init; } = new();

    public static Field FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        J.AllowOnly(o, "x", "y", "weight", "vignetting");
        return new Field
        {
            X = J.Req(o, "x", J.D),
            Y = J.Req(o, "y", J.D),
            Weight = J.Def(o, "weight", J.D, 1),
            Vignetting = J.Def(o, "vignetting", Model.Vignetting.FromJson, new Vignetting()),
        };
    }

    public JsonNode ToJson() => new JsonObject
    {
        ["x"] = J.Num(X), ["y"] = J.Num(Y), ["weight"] = J.Num(Weight), ["vignetting"] = Vignetting.ToJson(),
    };
}

[JsonConverter(typeof(ModelConverter<FieldSet>))]
public sealed record FieldSet : IJsonModel<FieldSet>
{
    public FieldKind Kind { get; init; } = FieldKind.ObjectAngle;
    public ValueList<Field> Fields { get; init; } = [new Field()];

    /// <summary>Normaliser for <c>(Hx, Hy)</c>; <c>1</c> when every field is on axis.</summary>
    public double MaxRadial
    {
        get
        {
            var m = Fields.Count == 0 ? 0 : Fields.Max(f => double.Hypot(f.X, f.Y));
            return m > 0 ? m : 1;
        }
    }

    public static FieldSet FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        J.AllowOnly(o, "kind", "fields");
        return new FieldSet
        {
            Kind = J.Req(o, "kind", n => Wire.Read<FieldKind>(n, "field kind")),
            Fields = J.Req(o, "fields", n => J.Arr(n, Field.FromJson)),
        };
    }

    public JsonNode ToJson() => new JsonObject { ["kind"] = Wire.Write(Kind), ["fields"] = J.Arr(Fields, f => f.ToJson()) };
}

[JsonConverter(typeof(ModelConverter<Wavelength>))]
public sealed record Wavelength(double Um, double Weight = 1) : IJsonModel<Wavelength>
{
    public static Wavelength FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        J.AllowOnly(o, "um", "weight");
        return new Wavelength(J.Req(o, "um", J.D), J.Def(o, "weight", J.D, 1));
    }

    public JsonNode ToJson() => new JsonObject { ["um"] = J.Num(Um), ["weight"] = J.Num(Weight) };
}

[JsonConverter(typeof(ModelConverter<WavelengthSet>))]
public sealed record WavelengthSet : IJsonModel<WavelengthSet>
{
    /// <summary>F, d, C with d primary (the engine's default).</summary>
    public ValueList<Wavelength> Wavelengths { get; init; } =
        [new Wavelength(SpectralLine.F.Um()), new Wavelength(SpectralLine.D.Um()), new Wavelength(SpectralLine.C.Um())];

    /// <summary>Index into <see cref="Wavelengths"/>.</summary>
    public int Primary { get; init; } = 1;

    public double? PrimaryUm => Wavelengths.IsValidIndex(Primary) ? Wavelengths[Primary].Um : null;

    public static WavelengthSet FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        J.AllowOnly(o, "wavelengths", "primary");
        return new WavelengthSet
        {
            Wavelengths = J.Req(o, "wavelengths", n => J.Arr(n, Wavelength.FromJson)),
            Primary = J.Req(o, "primary", J.Index),
        };
    }

    public JsonNode ToJson() => new JsonObject { ["wavelengths"] = J.Arr(Wavelengths, w => w.ToJson()), ["primary"] = Primary };
}

[JsonConverter(typeof(ModelConverter<RayAiming>))]
public abstract record RayAiming : IJsonModel<RayAiming>
{
    public const double DefaultTolerance = 1e-10;

    /// <summary>Pupil coordinates map onto the paraxial entrance pupil.</summary>
    public sealed record Off : RayAiming;

    /// <summary>Real-ray aiming onto the stop; tolerance as a fraction of the stop radius.</summary>
    public sealed record Real(double Tolerance = DefaultTolerance) : RayAiming;

    public static RayAiming FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        switch (J.Tag(o, "mode"))
        {
            case "off":
                J.AllowOnly(o, "mode");
                return new Off();
            case "real":
                J.AllowOnly(o, "mode", "tolerance");
                return new Real(J.Def(o, "tolerance", J.D, DefaultTolerance));
            case var m: throw J.Fail($"unknown ray aiming '{m}'");
        }
    }

    public JsonNode ToJson() => this is Real r
        ? new JsonObject { ["mode"] = "real", ["tolerance"] = J.Num(r.Tolerance) }
        : new JsonObject { ["mode"] = "off" };
}

[JsonConverter(typeof(ModelConverter<EnvironmentConditions>))]
public sealed record EnvironmentConditions(double TemperatureC = 20, double PressureAtm = 1) : IJsonModel<EnvironmentConditions>
{
    public static EnvironmentConditions FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        J.AllowOnly(o, "temperature_c", "pressure_atm");
        return new EnvironmentConditions(J.Req(o, "temperature_c", J.D), J.Req(o, "pressure_atm", J.D));
    }

    public JsonNode ToJson() => new JsonObject { ["temperature_c"] = J.Num(TemperatureC), ["pressure_atm"] = J.Num(PressureAtm) };
}

/// <summary>Standard spectral lines, identical to the engine's <c>materials::lines</c>.</summary>
public enum SpectralLine { G, F, E, D, C }

public static class SpectralLineExtensions
{
    /// <summary>Wavelength in µm.</summary>
    public static double Um(this SpectralLine line) => line switch
    {
        SpectralLine.D => 0.5875618,
        SpectralLine.F => 0.4861327,
        SpectralLine.C => 0.6562725,
        SpectralLine.G => 0.4358343,
        _ => 0.5460740,
    };
}
