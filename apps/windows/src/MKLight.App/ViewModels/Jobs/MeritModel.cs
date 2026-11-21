using System.Globalization;
using System.Text.Json.Nodes;
using MKLight.App.Support;
using MKLight.Engine;
using MKLight.Model;

namespace MKLight.App.ViewModels.Jobs;

// The merit function of the Optimize page (the Mac app's MeritModel): custom operands, the default
// generator's settings, the saved `merit` section and the engine's evaluation.

/// <summary>Comparison of an operand with its target.</summary>
public enum MeritComparison { Equal, LessThan, GreaterThan }

public static class MeritComparisons
{
    public static string Wire(this MeritComparison c) => c switch
    {
        MeritComparison.LessThan => "less_than",
        MeritComparison.GreaterThan => "greater_than",
        _ => "equal",
    };

    public static string Symbol(this MeritComparison c) => c switch
    {
        MeritComparison.LessThan => "≤",
        MeritComparison.GreaterThan => "≥",
        _ => "=",
    };

    public static MeritComparison Parse(string? s) => s switch
    {
        "less_than" => MeritComparison.LessThan,
        "greater_than" => MeritComparison.GreaterThan,
        _ => MeritComparison.Equal,
    };
}

/// <summary>
/// One line of a custom merit function: the engine's <c>Operand</c> (docs/OPTIMIZATION.md) with its kind kept as
/// JSON, so parameters the app does not edit (ray groups, detectors, ghost paths) survive a round trip.
/// <see cref="Id"/> is a session identity (not saved) so selection follows an operand across edits.
/// </summary>
public sealed class MeritOperand
{
    public Guid Id { get; set; } = Guid.NewGuid();
    /// <summary><c>{"type": "efl", …}</c>: the operand kind and its parameters.</summary>
    public JsonObject Kind { get; set; }
    public double Target { get; set; }
    public double Weight { get; set; } = 1;
    public MeritComparison Comparison { get; set; }
    /// <summary>An explicit constraint rather than a weighted residual.</summary>
    public bool IsConstraint { get; set; }
    /// <summary><c>{"field": "all"}</c>, <c>{"field": "index", "index": i}</c> or normalised.</summary>
    public JsonObject Field { get; set; } = new() { ["field"] = "all" };
    /// <summary><c>{"wavelength": "primary" | "all" | "index", …}</c>.</summary>
    public JsonObject Wavelength { get; set; } = new() { ["wavelength"] = "primary" };
    public int Config { get; set; }
    public bool Enabled { get; set; } = true;
    public string Label { get; set; } = "";

    public MeritOperand(JsonObject kind)
    {
        Kind = kind;
        if (OperandCatalog.Descriptor(Type) is { SingleField: true }) Field = new JsonObject { ["field"] = "index", ["index"] = 0 };
    }

    public string Type => Kind["type"].StringOrNull() ?? "?";
    public OperandDescriptor? Descriptor => OperandCatalog.Descriptor(Type);

    public MeritOperand Clone(bool newIdentity = false) => new((JsonObject)Kind.DeepClone())
    {
        Id = newIdentity ? Guid.NewGuid() : Id, Target = Target, Weight = Weight, Comparison = Comparison, IsConstraint = IsConstraint,
        Field = (JsonObject)Field.DeepClone(), Wavelength = (JsonObject)Wavelength.DeepClone(), Config = Config, Enabled = Enabled, Label = Label,
    };

    /// <summary>The engine's JSON form.</summary>
    public JsonObject ToJson()
    {
        var o = new JsonObject
        {
            ["kind"] = Kind.DeepClone(), ["target"] = Target, ["weight"] = Weight, ["comparison"] = Comparison.Wire(),
            ["role"] = IsConstraint ? "constraint" : "residual", ["field"] = Field.DeepClone(), ["wavelength"] = Wavelength.DeepClone(),
            ["config"] = Config, ["enabled"] = Enabled,
        };
        if (Label.Length > 0) o["label"] = Label;
        return o;
    }

    /// <summary>Reads an engine operand (a spec's or a result's operand table).</summary>
    public static MeritOperand? FromJson(JsonNode? json)
    {
        if (json?["kind"] is not JsonObject k || k["type"].StringOrNull() is null) return null;
        return new MeritOperand((JsonObject)k.DeepClone())
        {
            Target = json["target"].DoubleValue() ?? 0,
            Weight = json["weight"].DoubleValue() ?? 1,
            Comparison = MeritComparisons.Parse(json["comparison"].StringOrNull()),
            IsConstraint = json["role"].StringOrNull() == "constraint",
            Field = json["field"] is JsonObject f ? (JsonObject)f.DeepClone() : new JsonObject { ["field"] = "all" },
            Wavelength = json["wavelength"] is JsonObject w ? (JsonObject)w.DeepClone() : new JsonObject { ["wavelength"] = "primary" },
            Config = (int)(json["config"].IntOrNull() ?? 0),
            Enabled = json["enabled"].BoolOrNull() ?? true,
            Label = json["label"].StringOrNull() ?? "",
        };
    }

    /// <summary>Problems the app can see before the engine does.</summary>
    public IReadOnlyList<string> LocalIssues(int fieldCount, int wavelengthCount)
    {
        var output = new List<string>();
        if (Descriptor is not { } d) return output;
        if (d.SingleField && Field["field"].StringOrNull() == "all") output.Add(Loc.S("Jobs_Merit_Issue_NeedsOneField"));
        if (Field["field"].StringOrNull() == "index" && Field["index"].IntOrNull() is { } i && i >= fieldCount) output.Add(Loc.F("Jobs_Merit_Issue_NoField", i + 1));
        if (Wavelength["wavelength"].StringOrNull() == "index" && Wavelength["index"].IntOrNull() is { } w && w >= wavelengthCount)
            output.Add(Loc.F("Jobs_Merit_Issue_NoWavelength", w + 1));
        foreach (var p in d.Params.Where(p => p.Required && Kind[p.Key] is null)) output.Add(Loc.F("Jobs_Merit_Issue_Required", EngineText.T(p.Title)));
        if (!double.IsFinite(Weight) || Weight < 0) output.Add(Loc.S("Jobs_Merit_Issue_Weight"));
        return output;
    }

    /// <summary>A short text of the operand's selectors and parameters (field and wavelength first): <c>F2 λ1 axis y px 0 py 1</c>.</summary>
    public string? ParameterSummary()
    {
        var parts = new List<string>();
        switch (Field["field"].StringOrNull())
        {
            case "index": parts.Add($"F{(Field["index"].IntOrNull() ?? 0) + 1}"); break;
            case "normalized": parts.Add($"H({MeritFormat.Short(Field["hx"].DoubleValue() ?? 0)}, {MeritFormat.Short(Field["hy"].DoubleValue() ?? 0)})"); break;
        }
        switch (Wavelength["wavelength"].StringOrNull())
        {
            case "all": parts.Add(Loc.S("Jobs_Merit_Summary_Polychromatic")); break;
            case "index": parts.Add($"λ{(Wavelength["index"].IntOrNull() ?? 0) + 1}"); break;
        }
        foreach (var p in Descriptor?.Params ?? [])
        {
            if (Kind[p.Key] is not { } v) continue;
            var title = EngineText.T(p.Title);
            switch (p.Kind)
            {
                case OperandParamKind.Surface: parts.Add($"{title.ToLowerInvariant()} S{v.IntOrNull()?.ToString(CultureInfo.InvariantCulture) ?? "?"}"); break;
                case OperandParamKind.Wavelength: parts.Add($"{title} λ{(v.IntOrNull() ?? 0) + 1}"); break;
                case OperandParamKind.Choice:
                    var option = v.StringOrNull();
                    parts.Add(option is not null && (p.Options?.Contains(option) ?? false) ? OperandCatalog.OptionTitle(option).ToLowerInvariant()
                        : (option ?? v.ToJsonString()).Replace('_', ' '));
                    break;
                case OperandParamKind.Flag: if (v.BoolOrNull() == true) parts.Add(title.ToLowerInvariant()); break;
                case OperandParamKind.Sampling: parts.Add($"{v["rings"].IntOrNull() ?? 3}×{v["arms"].IntOrNull() ?? 6}"); break;
                case OperandParamKind.OperandList: parts.Add(Loc.F("Jobs_Merit_Summary_Of", string.Join(", ", (v.IntArray() ?? []).Select(k => $"#{k + 1}")))); break;
                default: parts.Add($"{title.ToLowerInvariant()} {(v.DoubleValue() is { } d ? MeritFormat.Short(d) : v.ToJsonString())}"); break;
            }
        }
        return parts.Count == 0 ? null : string.Join(" ", parts);
    }

    /// <summary>The row's detail line: parameters, comparison and target, weight.</summary>
    public string Detail()
    {
        var parts = new List<string>();
        if (ParameterSummary() is { Length: > 0 } s) parts.Add(s);
        parts.Add($"{Comparison.Symbol()} {MeritFormat.Short(Target)}");
        if (!IsConstraint) parts.Add(Loc.F("Jobs_Merit_Detail_Weight", MeritFormat.Short(Weight)));
        return string.Join(" · ", parts);
    }
}

/// <summary>How an operand parameter is edited.</summary>
public enum OperandParamKind { Number, Integer, Surface, Wavelength, Choice, Flag, Sampling, OperandList }

public sealed record OperandParam(string Key, string Title, OperandParamKind Kind, bool Required = false, JsonNode? Default = null,
    string Unit = "", int Minimum = 0, IReadOnlyList<string>? Options = null);

/// <summary>An operand kind of the engine's operand reference.</summary>
public sealed record OperandDescriptor(string Type, string Title, string Category, string Unit, string Summary,
    IReadOnlyList<OperandParam>? Parameters = null, bool SingleField = false)
{
    public IReadOnlyList<OperandParam> Params => Parameters ?? [];

    /// <summary>The kind object for a new operand.</summary>
    public JsonObject NewKind()
    {
        var k = new JsonObject { ["type"] = Type };
        foreach (var p in Params) if (p.Default is { } v) k[p.Key] = v.DeepClone();
        return k;
    }
}

/// <summary>The engine's operands (docs/OPTIMIZATION.md, operand reference), grouped by category for the picker.</summary>
public static class OperandCatalog
{
    public static readonly string[] Categories = ["First order", "Geometry", "Real rays", "Image quality", "Ghosts and stray light", "Manufacturing", "Composition"];

    static OperandParam N(string key, string title, string unit = "", double def = 0, bool required = true) =>
        new(key, title, OperandParamKind.Number, required, required ? JsonValue.Create(def) : null, unit);
    static OperandParam Surf(string key = "surface", string title = "Surface", bool required = false) => new(key, title, OperandParamKind.Surface, required);
    static OperandParam Choice(string key, string title, string[] options, string? def = null) =>
        new(key, title, OperandParamKind.Choice, false, def is null ? null : JsonValue.Create(def), Options: options);
    static OperandParam Wl(string key, string title, int def) => new(key, title, OperandParamKind.Wavelength, true, JsonValue.Create(def));
    static OperandParam Int(string key, string title, int min, int def) => new(key, title, OperandParamKind.Integer, false, JsonValue.Create(def), Minimum: min);
    static OperandParam[] Pupil => [N("px", "Pupil x", def: 0), N("py", "Pupil y", def: 1)];
    static OperandParam Sampling => new("sampling", "Pupil sampling", OperandParamKind.Sampling, false, new JsonObject { ["rings"] = 3, ["arms"] = 6 });
    static OperandDescriptor D(string type, string title, string category, string unit, string summary, OperandParam[]? p = null, bool single = false) =>
        new(type, title, category, unit, summary, p, single);

    public static readonly IReadOnlyList<OperandDescriptor> All =
    [
        // First order
        D("efl", "EFL", "First order", "mm", "Effective focal length (of the y section)"),
        D("efl_x", "EFL x", "First order", "mm", "Focal length of the x section"),
        D("efl_y", "EFL y", "First order", "mm", "Focal length of the y section"),
        D("bfl", "BFL", "First order", "mm", "Back focal length: last surface to paraxial focus"),
        D("bfl_x", "BFL x", "First order", "mm", "Back focus of the x section"),
        D("bfl_y", "BFL y", "First order", "mm", "Back focus of the y section"),
        D("anamorphic_squeeze", "Anamorphic squeeze", "First order", "", "Signed EFL_y / EFL_x"),
        D("f_number", "F-number", "First order", "", "Image-space, working or real working F/#",
            [Choice("kind", "Kind", ["paraxial_image_space", "paraxial_working", "real_working"], "paraxial_image_space")]),
        D("magnification", "Magnification", "First order", "", "Paraxial lateral magnification"),
        D("entrance_pupil_diameter", "Entrance pupil diameter", "First order", "mm", ""),
        D("entrance_pupil_position", "Entrance pupil position", "First order", "mm", "From the first surface"),
        D("exit_pupil_position", "Exit pupil position", "First order", "mm", "From the image surface"),
        D("paraxial_ray", "Paraxial ray", "First order", "mm / – / °", "Marginal or chief ray height, slope or angle after a surface",
            [Choice("ray", "Ray", ["marginal", "chief"], "marginal"), Choice("quantity", "Quantity", ["height", "slope", "angle"], "height"), Surf()]),
        D("petzval_sum", "Petzval sum", "First order", "1/mm", "Σ c (N′ − N)/(N N′)"),
        D("axial_colour", "Axial color", "First order", "mm", "Paraxial focus at short minus long λ", [Wl("short", "Short λ", 0), Wl("long", "Long λ", 2)]),
        // Geometry
        D("total_track", "Total track", "Geometry", "mm", "Sum of thicknesses", [Surf("from", "From"), Surf("to", "To")]),
        D("centre_thickness", "Center thickness", "Geometry", "mm", "Axial gap after a surface", [Surf(required: true), Surf("to", "To")]),
        D("edge_thickness", "Edge thickness", "Geometry", "mm", "Gap at the rim (or a height)",
            [Surf(required: true), Surf("to", "To"), N("height", "Height", "mm", required: false)]),
        D("min_thickness", "Minimum thickness", "Geometry", "mm", "Smallest gap of a medium",
            [Choice("measure", "Measure", ["centre", "edge"], "centre"), Choice("medium", "Medium", ["glass", "air", "any"], "glass"), Surf("from", "From"), Surf("to", "To")]),
        D("max_thickness", "Maximum thickness", "Geometry", "mm", "Largest gap of a medium",
            [Choice("measure", "Measure", ["centre", "edge"], "centre"), Choice("medium", "Medium", ["glass", "air", "any"], "glass"), Surf("from", "From"), Surf("to", "To")]),
        // Real rays
        D("real_ray", "Real ray", "Real rays", "mm / – / °", "Position, direction or angle of one ray",
            [Choice("quantity", "Quantity", ["x", "y", "z", "l", "m", "n", "angle_x", "angle_y", "incidence_angle", "optical_path"], "y"), Surf(), .. Pupil,
             new OperandParam("global", "Global frame", OperandParamKind.Flag)], true),
        D("ray_aberration", "Ray aberration", "Real rays", "mm", "Transverse aberration of one ray",
            [Choice("axis", "Axis", ["x", "y"], "y"), .. Pupil, Choice("reference", "Reference", ["chief", "centroid"], "chief")], true),
        D("wavefront_aberration", "Wavefront aberration", "Real rays", "waves", "OPD of one ray", [.. Pupil, Choice("reference", "Reference", ["chief", "centroid"], "chief")], true),
        D("chief_ray_angle", "Chief ray angle", "Real rays", "°", "Incidence of the chief ray (0 telecentric)", [Surf()], true),
        // Image quality
        D("rms_spot", "RMS spot radius", "Image quality", "mm", "About the centroid or the chief ray", [Choice("reference", "Reference", ["centroid", "chief"], "centroid"), Sampling]),
        D("spot_centroid", "Spot centroid", "Image quality", "mm", "Centroid minus the chief ray", [Choice("axis", "Axis", ["x", "y"], "y")], true),
        D("wavefront_rms", "RMS wavefront", "Image quality", "waves", "Piston (and tilt) removed", [Choice("reference", "Reference", ["centroid", "chief"], "centroid"), Sampling]),
        D("mtf", "MTF", "Image quality", "", "Modulation at one frequency; use ≥ for a lower bound",
            [N("frequency", "Frequency", "cyc/mm", 30), Choice("orientation", "Orientation", ["tangential", "sagittal", "average"], "average"),
             Choice("method", "Method", ["diffraction", "geometric"], "diffraction"), N("defocus_mm", "Defocus", "mm", required: false)]),
        D("distortion", "Distortion", "Image quality", "%", "Real chief ray against the mapping", [Choice("mapping", "Mapping", ["f_tan_theta", "f_theta"], "f_tan_theta")], true),
        D("lateral_colour", "Lateral color", "Image quality", "mm", "Chief height at short minus long λ", [Wl("short", "Short λ", 0), Wl("long", "Long λ", 2)], true),
        D("relative_illumination", "Relative illumination", "Image quality", "", "Against on axis", single: true),
        // Ghosts and stray light
        D("ghost_focus", "Ghost focus distance", "Ghosts and stray light", "mm", "Distance of the strongest ghosts' focus from the image (larger is better)",
            [Int("worst", "Worst ghosts", 1, 1), Choice("metric", "Metric", ["distance", "peak"], "distance")]),
        D("ghost_irradiance", "Ghost irradiance", "Ghosts and stray light", "rel.", "Ghost against primary",
            [Choice("metric", "Metric", ["peak", "mean", "power"], "peak"), Int("max_reflections", "Max reflections", 2, 2)]),
        D("stray_light_power", "Stray light power", "Ghosts and stray light", "rel.", "Monte Carlo fraction; not for least squares",
            [Choice("class", "Class", ["ghost", "scatter", "mechanical", "sneak", "all_non_primary"], "all_non_primary"), Int("rays", "Rays", 1000, 100_000)]),
        // Manufacturing
        D("doe_min_zone_period", "DOE minimum zone period", "Manufacturing", "mm", "", [Surf(required: true)]),
        D("doe_max_phase_slope", "DOE maximum phase slope", "Manufacturing", "rad/mm", "", [Surf(required: true)]),
        D("doe_total_phase", "DOE total phase", "Manufacturing", "rad", "", [Surf(required: true)]),
        D("q_max_departure", "Q-type maximum departure", "Manufacturing", "mm", "From the best-fit sphere", [Surf(required: true)]),
        D("q_max_departure_slope", "Q-type departure slope", "Manufacturing", "mm/mm", "", [Surf(required: true)]),
        D("freeform_max_departure", "Freeform maximum departure", "Manufacturing", "mm", "XY or Zernike sag from its base conic", [Surf(required: true)]),
        // Composition
        D("combine", "Combine", "Composition", "", "Arithmetic on earlier operands",
            [Choice("op", "Operation", ["sum", "difference", "product", "quotient", "abs", "min", "max"], "sum"),
             new OperandParam("args", "Operands", OperandParamKind.OperandList, true, new JsonArray())]),
        D("param_value", "Parameter value", "Composition", "", "Any model parameter"),
        D("tolerance_sensitivity", "Tolerance sensitivity", "Composition", "", "Of a criterion to a parameter"),
    ];

    static readonly Dictionary<string, OperandDescriptor> ByType = All.GroupBy(d => d.Type).ToDictionary(g => g.Key, g => g.First());

    public static OperandDescriptor? Descriptor(string type) => ByType.GetValueOrDefault(type);

    /// <summary>Operands a user adds from the picker (the composition kinds that need a model path or a nested criterion are read and kept, not added).</summary>
    public static IEnumerable<OperandDescriptor> Addable => All.Where(d => d.Type is not ("param_value" or "tolerance_sensitivity"));

    /// <summary>An operand's name as shown (engine text, in the UI language).</summary>
    public static string Title(string type) =>
        EngineText.T(Descriptor(type)?.Title ?? CultureInfo.InvariantCulture.TextInfo.ToTitleCase(type.Replace('_', ' ')));

    /// <summary>A choice parameter's option as shown: <c>real_working</c> is <c>Real Working</c> (engine text, in the UI language).</summary>
    public static string OptionTitle(string option) => EngineText.T(CultureInfo.InvariantCulture.TextInfo.ToTitleCase(option.Replace('_', ' ')));

    /// <summary>Descriptors matching a search (in English or the UI language), by category.</summary>
    public static IReadOnlyList<(string Category, IReadOnlyList<OperandDescriptor> Items)> Search(string text)
    {
        var q = text.Trim().ToLowerInvariant();
        static bool Has(string s, string q) => s.ToLowerInvariant().Contains(q) || EngineText.T(s).ToLowerInvariant().Contains(q);
        var hits = Addable.Where(d => q.Length == 0 || Has(d.Title, q) || d.Type.Contains(q)
            || Has(d.Summary, q) || Has(d.Category, q)).ToList();
        return Categories.Select(c => (c, (IReadOnlyList<OperandDescriptor>)hits.Where(d => d.Category == c).ToList())).Where(g => g.Item2.Count > 0).ToList();
    }
}

/// <summary>Criterion of the default merit generator.</summary>
public enum MeritCriterion { RmsSpot, Wavefront, Mtf }

/// <summary>The engine's default merit generator (<c>DefaultMeritSpec</c>): criterion, pupil sampling, fields and wavelengths, thickness boundaries, squeeze.</summary>
public sealed record DefaultMeritSettings
{
    public MeritCriterion Criterion { get; init; } = MeritCriterion.RmsSpot;
    /// <summary>Spot or wavefront reference: <c>centroid</c> or <c>chief</c>.</summary>
    public string Reference { get; init; } = "centroid";
    /// <summary>Centroid of the white-light spot (<c>polychromatic</c>, the engine's default) or per wavelength.</summary>
    public string Scope { get; init; } = "polychromatic";
    public IReadOnlyList<double> MtfFrequencies { get; init; } = [10, 30];
    public double MtfTarget { get; init; } = 0.5;
    public string MtfMethod { get; init; } = "diffraction";
    public bool WorstField { get; init; }
    public int Rings { get; init; } = 3;
    public int Arms { get; init; } = 6;
    /// <summary>null: every field (or wavelength) with positive weight.</summary>
    public IReadOnlyList<int>? Fields { get; init; }
    public IReadOnlyList<int>? Wavelengths { get; init; }
    public bool Thickness { get; init; } = true;
    public double? GlassMinCentre { get; init; } = 1.0;
    public double? GlassMaxCentre { get; init; } = 30.0;
    public double? GlassMinEdge { get; init; } = 1.0;
    public double? AirMinCentre { get; init; } = 0.1;
    public double? AirMaxCentre { get; init; }
    public double? AirMinEdge { get; init; } = 0.1;
    public double? SqueezeTarget { get; init; }

    public static string Wire(MeritCriterion c) => c switch { MeritCriterion.Wavefront => "wavefront", MeritCriterion.Mtf => "mtf", _ => "rms_spot" };

    public static string Title(MeritCriterion c) => c switch
    {
        MeritCriterion.Wavefront => Loc.S("Jobs_Merit_Criterion_Wavefront"),
        MeritCriterion.Mtf => "MTF",
        _ => Loc.S("Jobs_Merit_Criterion_RmsSpot"),
    };

    /// <summary>The <c>default_merit</c> settings object (laid over the engine's defaults).</summary>
    public JsonObject ToJson()
    {
        var crit = new JsonObject { ["type"] = Wire(Criterion) };
        switch (Criterion)
        {
            case MeritCriterion.RmsSpot:
                crit["reference"] = Reference;
                crit["scope"] = Scope;
                break;
            case MeritCriterion.Wavefront:
                crit["reference"] = Reference;
                break;
            default:
                crit["frequencies"] = new JsonArray([.. MtfFrequencies.Select(f => (JsonNode)f)]);
                crit["target"] = MtfTarget;
                crit["method"] = MtfMethod;
                crit["worst_field"] = WorstField;
                break;
        }
        var o = new JsonObject
        {
            ["criterion"] = crit,
            ["sampling"] = new JsonObject { ["rings"] = Rings, ["arms"] = Arms },
            ["fields"] = Fields is null ? null : new JsonArray([.. Fields.Select(i => (JsonNode)i)]),
            ["wavelengths"] = Wavelengths is null ? null : new JsonArray([.. Wavelengths.Select(i => (JsonNode)i)]),
        };
        static JsonNode? V(double? d) => d is { } x ? JsonValue.Create(x) : null;
        o["thickness"] = Thickness
            ? new JsonObject
            {
                ["glass_min_centre"] = V(GlassMinCentre), ["glass_max_centre"] = V(GlassMaxCentre), ["glass_min_edge"] = V(GlassMinEdge),
                ["air_min_centre"] = V(AirMinCentre), ["air_max_centre"] = V(AirMaxCentre), ["air_min_edge"] = V(AirMinEdge),
            }
            : null;
        if (SqueezeTarget is { } s) o["squeeze_target"] = s;
        return o;
    }

    public static DefaultMeritSettings FromJson(JsonNode? j)
    {
        var d = new DefaultMeritSettings();
        if (j is not JsonObject o) return d;
        var c = o["criterion"];
        var type = c?["type"].StringOrNull();
        d = d with
        {
            Criterion = type == "wavefront" ? MeritCriterion.Wavefront : type == "mtf" ? MeritCriterion.Mtf : MeritCriterion.RmsSpot,
            Reference = c?["reference"].StringOrNull() ?? d.Reference,
            Scope = c?["scope"].StringOrNull() ?? d.Scope,
            MtfFrequencies = c?["frequencies"].NfArray() ?? [.. d.MtfFrequencies],
            MtfTarget = c?["target"].DoubleValue() ?? d.MtfTarget,
            MtfMethod = c?["method"].StringOrNull() ?? d.MtfMethod,
            WorstField = c?["worst_field"].BoolOrNull() ?? false,
            Rings = (int)(o["sampling"]?["rings"].IntOrNull() ?? d.Rings),
            Arms = (int)(o["sampling"]?["arms"].IntOrNull() ?? d.Arms),
            Fields = o["fields"].IntArray()?.Select(i => (int)i).ToList(),
            Wavelengths = o["wavelengths"].IntArray()?.Select(i => (int)i).ToList(),
            SqueezeTarget = Number(o["squeeze_target"]),
        };
        if (o.TryGetPropertyValue("thickness", out var t))
        {
            d = d with { Thickness = t is not null };
            if (t is not null)
                d = d with
                {
                    GlassMinCentre = Number(t["glass_min_centre"]), GlassMaxCentre = Number(t["glass_max_centre"]), GlassMinEdge = Number(t["glass_min_edge"]),
                    AirMinCentre = Number(t["air_min_centre"]), AirMaxCentre = Number(t["air_max_centre"]), AirMinEdge = Number(t["air_min_edge"]),
                };
        }
        return d;
    }

    static double? Number(JsonNode? n) => n.DoubleValue() is { } d && double.IsFinite(d) ? d : null;

    /// <summary>The caption under the default merit's settings.</summary>
    public string Caption => Criterion switch
    {
        MeritCriterion.RmsSpot => Loc.S("Jobs_Merit_Caption_RmsSpot"),
        MeritCriterion.Wavefront => Loc.S("Jobs_Merit_Caption_Wavefront"),
        _ => Loc.S("Jobs_Merit_Caption_Mtf"),
    };

    public bool Equals(DefaultMeritSettings? other) => other is not null && JsonNode.DeepEquals(ToJson(), other.ToJson());
    public override int GetHashCode() => ToJson().ToJsonString().GetHashCode();
}

public enum MeritMode { Default, Custom }

/// <summary>
/// The merit function of a design, saved in the project's <c>merit</c> section: the default generator's settings,
/// or a custom operand list (an engine <c>MeritFunctionSpec</c>), and whether the focal length is held. A working
/// copy: the editor mutates it, then commits <see cref="Section"/> as one undoable step.
/// </summary>
public sealed class MeritSettings
{
    public MeritMode Mode { get; set; } = MeritMode.Default;
    public DefaultMeritSettings Defaults { get; set; } = new();
    public List<MeritOperand> Operands { get; set; } = [];
    public bool HoldEfl { get; set; } = true;
    /// <summary>The custom spec's other keys (<c>name</c>, <c>on_failure</c>, …), kept as the engine wrote them.</summary>
    public JsonObject SpecOptions { get; set; } = [];

    public static MeritSettings FromSection(JsonTree section)
    {
        var m = new MeritSettings();
        if (section.Node is not JsonObject s) return m;
        m.Mode = s["mode"].StringOrNull() == "custom" ? MeritMode.Custom : MeritMode.Default;
        m.Defaults = DefaultMeritSettings.FromJson(s["default_merit"]);
        m.Operands = (s["spec"]?["operands"] as JsonArray ?? []).Select(MeritOperand.FromJson).OfType<MeritOperand>().ToList();
        m.SpecOptions = SpecOptionsOf(s["spec"]);
        m.HoldEfl = s["hold_efl"].BoolOrNull() ?? true;
        return m;
    }

    internal static JsonObject SpecOptionsOf(JsonNode? spec)
    {
        var o = new JsonObject();
        if (spec is JsonObject so)
            foreach (var (k, v) in so)
                if (k is not ("operands" or "schema_version")) o[k] = v?.DeepClone();
        return o;
    }

    public MeritSettings Clone() => new()
    {
        Mode = Mode, Defaults = Defaults, HoldEfl = HoldEfl, SpecOptions = (JsonObject)SpecOptions.DeepClone(),
        Operands = Operands.Select(o => o.Clone()).ToList(),
    };

    public JsonTree Section => new(new JsonObject
    {
        ["mode"] = Mode == MeritMode.Custom ? "custom" : "default", ["default_merit"] = Defaults.ToJson(), ["hold_efl"] = HoldEfl, ["spec"] = Spec(),
    });

    /// <summary>The custom operands as a <c>MeritFunctionSpec</c>.</summary>
    public JsonObject Spec()
    {
        var o = new JsonObject { ["name"] = "" };
        foreach (var (k, v) in SpecOptions) o[k] = v?.DeepClone();
        o["schema_version"] = 1;
        o["operands"] = new JsonArray([.. Operands.Select(op => (JsonNode)op.ToJson())]);
        return o;
    }

    /// <summary>The merit part of <c>optimize</c> / <c>glass_substitution</c> settings.</summary>
    public Dictionary<string, JsonNode?> JobSettings()
    {
        var s = new Dictionary<string, JsonNode?> { ["hold_efl"] = HoldEfl };
        if (Mode == MeritMode.Default) s["default_merit"] = Defaults.ToJson();
        else s["merit"] = Spec();
        return s;
    }

    public bool SameAs(MeritSettings other) => Section == other.Section;

    // ---- Editing (combine references follow their operands) ----

    public Guid Add(OperandDescriptor d, int? after = null)
    {
        var op = new MeritOperand(d.NewKind());
        var at = after is { } i ? Math.Min(i + 1, Operands.Count) : Operands.Count;
        Operands.Insert(at, op);
        RemapCombine(k => k >= at ? k + 1 : k);
        return op.Id;
    }

    public void Delete(IReadOnlySet<Guid> ids)
    {
        var removed = Enumerable.Range(0, Operands.Count).Where(i => ids.Contains(Operands[i].Id)).ToList();
        if (removed.Count == 0) return;
        Operands.RemoveAll(o => ids.Contains(o.Id));
        RemapCombine(old => removed.Contains(old) ? null : old - removed.Count(r => r < old));
    }

    public IReadOnlyList<Guid> Duplicate(IReadOnlySet<Guid> ids)
    {
        var output = new List<Guid>();
        for (var i = Operands.Count - 1; i >= 0; i--)
        {
            if (!ids.Contains(Operands[i].Id)) continue;
            var copy = Operands[i].Clone(newIdentity: true);
            var at = i + 1;
            Operands.Insert(at, copy);
            RemapCombine(k => k >= at ? k + 1 : k);
            output.Add(copy.Id);
        }
        return output;
    }

    /// <summary>Moves the operand at <paramref name="from"/> to index <paramref name="to"/> (after removal), keeping <c>combine</c> references.</summary>
    public void Move(int from, int to)
    {
        if (from < 0 || from >= Operands.Count) return;
        to = Math.Clamp(to, 0, Operands.Count - 1);
        if (from == to) return;
        var before = Operands.Select(o => o.Id).ToList();
        var op = Operands[from];
        Operands.RemoveAt(from);
        Operands.Insert(to, op);
        var newIndex = Operands.Select((o, k) => (o.Id, k)).ToDictionary(x => x.Id, x => x.k);
        RemapCombine(old => old < before.Count ? newIndex[before[old]] : null);
    }

    void RemapCombine(Func<int, int?> f)
    {
        foreach (var op in Operands.Where(o => o.Type == "combine"))
        {
            var args = op.Kind["args"].IntArray() ?? [];
            op.Kind["args"] = new JsonArray([.. args.Select(a => f((int)a)).OfType<int>().Select(a => (JsonNode)a)]);
        }
    }

    /// <summary>The EFL hold the engine adds (<c>hold EFL</c>, or <c>hold EFL (penalty)</c> for the population methods).</summary>
    public static bool IsEflHold(MeritOperand op) => op.Type == "efl" && op.Label.StartsWith("hold EFL", StringComparison.Ordinal);
}

/// <summary>The engine's evaluation status.</summary>
public enum MeritStatus { Ok, Degraded, Failed }

/// <summary>
/// The merit and its operands at the current design, from the engine's <c>merit_evaluation</c>: one evaluation with
/// <c>optimize</c>'s settings (solves applied, the default merit generated, the EFL hold added).
/// </summary>
public sealed record MeritEvaluation(double Merit, double Violation, MeritStatus Status, int Failures, string ReportedOn,
    IReadOnlyList<MeritEvaluation.Row> Rows, JsonNode? Spec)
{
    public sealed record Row(int Index, string Kind, string Label, bool IsConstraint, bool Enabled, double Value, double Target, double Weight,
        double Residual, double Contribution, double Fraction, string? Error);

    public static MeritEvaluation? FromJson(JsonNode? json)
    {
        if (json is not JsonObject o || !o.ContainsKey("merit")) return null;
        var status = o["status"]?["status"].StringOrNull() switch { "degraded" => MeritStatus.Degraded, "failed" => MeritStatus.Failed, _ => MeritStatus.Ok };
        var rows = (o["operands"] as JsonArray ?? []).Select((r, k) => new Row(
            (int)(r?["index"].IntOrNull() ?? k), r?["kind"].StringOrNull() ?? "", r?["label"].StringOrNull() ?? "",
            r?["role"].StringOrNull() == "constraint", r?["enabled"].BoolOrNull() ?? true,
            r?["value"].Nf() ?? double.NaN, r?["target"].Nf() ?? double.NaN, r?["weight"].Nf() ?? double.NaN,
            r?["residual"].Nf() ?? double.NaN, r?["contribution"].Nf() ?? double.NaN, r?["contribution_fraction"].Nf() ?? double.NaN,
            r?["error"].StringOrNull())).ToList();
        return new MeritEvaluation(o["merit"].Nf() ?? double.NaN, o["violation"].Nf() ?? double.NaN, status,
            (int)(o["status"]?["failures"].IntOrNull() ?? 0),
            o["backend"]?["reported_on"].StringOrNull() ?? o["backend"]?["evaluated_on"].StringOrNull() ?? "", rows, o["merit_spec"]?.DeepClone());
    }

    /// <summary>The row of operand <paramref name="index"/>, if it is an operand of type <paramref name="kind"/>.</summary>
    public Row? RowAt(int index, string? kind = null) => Rows.FirstOrDefault(r => r.Index == index && (kind is null || r.Kind == kind));

    /// <summary>The operands of the expanded spec (generated ones and the EFL hold).</summary>
    public IReadOnlyList<MeritOperand> SpecOperands => (Spec?["operands"] as JsonArray ?? []).Select(MeritOperand.FromJson).OfType<MeritOperand>().ToList();

    public JsonObject SpecOptions => MeritSettings.SpecOptionsOf(Spec);

    /// <summary>The readout's tooltip: the evaluator, a violation, a degraded or failed status.</summary>
    public string Help()
    {
        var on = ReportedOn.Length == 0 || ReportedOn.StartsWith("cpu", StringComparison.Ordinal) ? Loc.S("Jobs_Merit_Help_CpuReference") : ReportedOn;
        var s = new List<string> { Loc.F("Jobs_Merit_Help_Merit", MeritFormat.Merit(Merit), on) };
        if (double.IsFinite(Violation) && Violation > 1e-9) s.Add(Loc.F("Jobs_Merit_Help_Violation", MeritFormat.Short(Violation)));
        if (Status == MeritStatus.Degraded) s.Add(Loc.F(Failures == 1 ? "Jobs_Merit_Help_Degraded_One" : "Jobs_Merit_Help_Degraded_Many", Failures));
        if (Status == MeritStatus.Failed) s.Add(Loc.S("Jobs_Merit_Help_Failed"));
        return string.Join(" ", s);
    }
}

/// <summary>Number texts of the job pages (invariant culture, a decimal point, never −0).</summary>
public static class MeritFormat
{
    static readonly CultureInfo C = CultureInfo.InvariantCulture;

    /// <summary>Up to 6 significant digits; scientific for very small and very large values.</summary>
    public static string Short(double v) => BoundsFormat.Short(v);

    /// <summary>A merit value: 4 significant digits, scientific below 0.001 (<c>2.989e-4</c>, <c>0.03512</c>).</summary>
    public static string Merit(double v)
    {
        if (!double.IsFinite(v)) return BoundsFormat.Short(v);
        var a = Math.Abs(v);
        var t = a != 0 && (a < 1e-3 || a >= 1e5) ? v.ToString("0.000e+0", C).Replace("e+", "e") : v.ToString("G4", C);
        return Fix(t);
    }

    /// <summary>A share in percent: <c>18.6 %</c>.</summary>
    public static string Percent(double fraction, int digits = 1) =>
        double.IsFinite(fraction) ? Fix((fraction * 100).ToString("F" + digits, C)) + " %" : "—";

    /// <summary>A share of the merit as the contribution bar labels it.</summary>
    public static string Share(double f) =>
        !double.IsFinite(f) ? "" : f >= 0.995 ? "100%" : f >= 0.1 ? (f * 100).ToString("F0", C) + "%" : f >= 0.001 ? (f * 100).ToString("F1", C) + "%" : f > 0 ? "<0.1%" : "0%";

    /// <summary>A signed change: <c>+0.0050</c>, <c>−0.0270</c>.</summary>
    public static string Signed(double v) => double.IsFinite(v) ? (v >= 0 ? "+" : "") + Short(v) : "—";

    public static string Integer(long v) => v.ToString(C);

    static string Fix(string s) => s.StartsWith('-') ? (s.Skip(1).All(ch => ch is '0' or '.') ? s[1..] : "−" + s[1..]) : s;
}
