using System.Text.Json.Nodes;
using System.Text.Json.Serialization;

namespace MKLight.Model;

/// <summary>
/// The complete physical description of an optical system: a value-type
/// mirror of the engine's <c>OpticalSystem</c>, encoded with exactly the engine's
/// JSON representation. It holds no UI state. It is immutable: an edit produces a
/// new value (<c>with</c>, or the editing methods), so a whole design can be kept for undo.
/// <para>
/// Decoding is strict, like the engine's <c>deny_unknown_fields</c>: an unknown
/// key anywhere in the model is an error rather than silently dropped.
/// </para>
/// </summary>
[JsonConverter(typeof(ModelConverter<OpticalSystem>))]
public sealed partial record OpticalSystem : IJsonModel<OpticalSystem>
{
    public string Name { get; init; } = "";
    public string Notes { get; init; } = "";
    public ApertureSpec Aperture { get; init; } = ApertureSpec.Default;
    public FieldSet Fields { get; init; } = new();
    public WavelengthSet Wavelengths { get; init; } = new();

    /// <summary><c>Surfaces[0]</c> is the object and the last one the image.</summary>
    public required ValueList<Surface> Surfaces { get; init; }

    /// <summary>Index of the aperture stop.</summary>
    public required int Stop { get; init; }

    public RayAiming RayAiming { get; init; } = new RayAiming.Off();
    public EnvironmentConditions Environment { get; init; } = new();

    /// <summary>Named coatings referenced by <c>Surface.Coating</c>.</summary>
    public ValueMap<CoatingDef> Coatings { get; init; } = ValueMap<CoatingDef>.Empty;

    public ValueList<MechanicalPart> Mechanics { get; init; } = ValueList<MechanicalPart>.Empty;
    public Sensor? Sensor { get; init; }
    public ValueList<Configuration> Configurations { get; init; } = ValueList<Configuration>.Empty;

    /// <summary>
    /// Solves and pickups (docs/SOLVES.md). Omitted from the JSON when empty, as the
    /// engine does, so systems without solves hash as before.
    /// </summary>
    public ValueList<Solve> Solves { get; init; } = ValueList<Solve>.Empty;

    /// <summary>The engine's minimal valid system: object at infinity, a plane stop 10 mm before the image.</summary>
    public static OpticalSystem NewEmpty() => new()
    {
        Name = "Untitled",
        Surfaces =
        [
            new Surface { Id = new SurfaceID(0), Thickness = double.PositiveInfinity },
            new Surface { Id = new SurfaceID(1), Label = "STOP", Thickness = 10 },
            new Surface { Id = new SurfaceID(2), Thickness = 0 },
        ],
        Stop = 1,
    };

    public int ImageIndex => Surfaces.Count - 1;

    public bool ObjectAtInfinity => Surfaces.Count > 0 && double.IsInfinity(Surfaces[0].Thickness);

    public int? SurfaceIndexOf(SurfaceID id)
    {
        for (var i = 0; i < Surfaces.Count; i++)
            if (Surfaces[i].Id == id) return i;
        return null;
    }

    /// <summary>A <see cref="SurfaceID"/> no surface uses yet.</summary>
    public SurfaceID NextSurfaceID => new(Surfaces.Count == 0 ? 0 : Surfaces.Max(s => s.Id.Value) + 1);

    /// <summary>The system with surface <paramref name="index"/> replaced by <paramref name="edit"/> of it.</summary>
    public OpticalSystem WithSurface(int index, Func<Surface, Surface> edit) => this with { Surfaces = Surfaces.With(index, edit) };

    /// <summary>The primary wavelength in µm (the d line when the set is invalid).</summary>
    internal double PrimaryWavelengthUm => Wavelengths.PrimaryUm ?? SpectralLine.D.Um();

    public static OpticalSystem FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        J.AllowOnly(
            o, "name", "notes", "aperture", "fields", "wavelengths", "surfaces", "stop", "ray_aiming", "environment",
            "coatings", "mechanics", "sensor", "configurations", "solves");
        return new OpticalSystem
        {
            Name = J.Def(o, "name", J.S, ""),
            Notes = J.Def(o, "notes", J.S, ""),
            Aperture = J.Req(o, "aperture", ApertureSpec.FromJson),
            Fields = J.Req(o, "fields", FieldSet.FromJson),
            Wavelengths = J.Req(o, "wavelengths", WavelengthSet.FromJson),
            Surfaces = J.Req(o, "surfaces", n => J.Arr(n, Surface.FromJson)),
            Stop = J.Req(o, "stop", J.Index),
            RayAiming = J.Def(o, "ray_aiming", Model.RayAiming.FromJson, new RayAiming.Off()),
            Environment = J.Def(o, "environment", EnvironmentConditions.FromJson, new EnvironmentConditions()),
            Coatings = J.Def(o, "coatings", n => J.Map(n, CoatingDef.FromJson), ValueMap<CoatingDef>.Empty),
            Mechanics = J.Def(o, "mechanics", n => J.Arr(n, MechanicalPart.FromJson), ValueList<MechanicalPart>.Empty),
            Sensor = J.OptRef(o, "sensor", Model.Sensor.FromJson),
            Configurations = J.Def(o, "configurations", n => J.Arr(n, Configuration.FromJson), ValueList<Configuration>.Empty),
            Solves = J.Def(o, "solves", n => J.Arr(n, Solve.FromJson), ValueList<Solve>.Empty),
        };
    }

    public JsonNode ToJson()
    {
        var o = new JsonObject
        {
            ["name"] = Name,
            ["notes"] = Notes,
            ["aperture"] = Aperture.ToJson(),
            ["fields"] = Fields.ToJson(),
            ["wavelengths"] = Wavelengths.ToJson(),
            ["surfaces"] = J.Arr(Surfaces, s => s.ToJson()),
            ["stop"] = Stop,
            ["ray_aiming"] = RayAiming.ToJson(),
            ["environment"] = Environment.ToJson(),
            ["coatings"] = J.Map(Coatings, c => c.ToJson()),
            ["mechanics"] = J.Arr(Mechanics, m => m.ToJson()),
            ["sensor"] = Sensor?.ToJson(),
            ["configurations"] = J.Arr(Configurations, c => c.ToJson()),
        };
        if (Solves.Count > 0) o["solves"] = J.Arr(Solves, s => s.ToJson());
        return o;
    }
}

// MARK: - Structural validation

public enum IssueSeverity { Error, Warning }

/// <summary>
/// A structural problem with a model. The engine's validation is authoritative; this
/// mirror lets the editor and an offline engine report the same problems without tracing.
/// Every structural problem is an error, as in the engine; warnings come from the engine's
/// <c>validate</c> analysis.
/// </summary>
public sealed record ModelIssue(IssueSeverity Severity, int? Surface, string Message)
{
    public ModelIssue(IssueSeverity severity, string message) : this(severity, null, message) { }

    public override string ToString() => Surface is { } s ? $"surface {s}: {Message}" : Message;
}


public static class OpticalSystemValidationExtensions
{
    internal const string NonFiniteSuffix = " is not a finite number";

    /// <summary>
    /// The rules of the engine's <c>OpticalSystem::validation_issues</c> (CONVENTIONS §13):
    /// every problem found, each an error. <c>validation_cases.json</c> holds the engine's
    /// verdicts the rules are tested against. Values JSON cannot carry (NaN, and ±∞ outside the
    /// thicknesses that allow it) are errors, of which the JSON check reports the first. Of a
    /// configuration override this checks that its path resolves and that the parameter takes
    /// its value; the engine also validates the system each override configures.
    /// </summary>
    public static IReadOnlyList<ModelIssue> StructuralIssues(this OpticalSystem sys)
    {
        var json = NonRepresentableValueIssues(sys);
        var output = new List<ModelIssue>(json);
        var n = sys.Surfaces.Count;
        if (n < 3)
        {
            output.Add(new ModelIssue(IssueSeverity.Error, "a system needs object, at least one surface, and image"));
            return output;
        }
        if (sys.Stop < 1 || sys.Stop > n - 2)
            output.Add(new ModelIssue(IssueSeverity.Error, $"stop index {sys.Stop} must be between 1 and {n - 2}"));
        var rules = new RuleIssues();
        if (sys.Aperture.Value is { } av) rules.Positive("invalid_aperture_value", "aperture.value", av);
        var wl = sys.Wavelengths;
        if (wl.Wavelengths.Count == 0 || !wl.Wavelengths.IsValidIndex(wl.Primary))
            output.Add(new ModelIssue(IssueSeverity.Error, "wavelength set empty or primary out of range"));
        if (wl.Wavelengths.Any(w => !(w.Um > 0 && double.IsFinite(w.Um)) || w.Weight < 0))
            output.Add(new ModelIssue(IssueSeverity.Error, "wavelengths must be positive and finite with weights ≥ 0"));
        if (sys.Fields.Fields.Count == 0)
            output.Add(new ModelIssue(IssueSeverity.Error, "at least one field is required"));
        if (sys.ObjectAtInfinity && sys.Fields.Kind == FieldKind.ObjectHeight)
            output.Add(new ModelIssue(IssueSeverity.Error, "object-height fields require a finite object distance"));
        if (sys.ObjectAtInfinity && sys.Aperture is ApertureSpec.ObjectSpaceNA)
            output.Add(new ModelIssue(IssueSeverity.Error, "object-space NA requires a finite object distance"));
        if (sys.ObjectAtInfinity && sys.Fields.Kind == FieldKind.ObjectAngle && sys.Fields.Fields.Any(f => double.Hypot(f.X, f.Y) >= 90))
            output.Add(new ModelIssue(IssueSeverity.Error, "field angles must be below 90°"));
        for (var k = 0; k < sys.Fields.Fields.Count; k++)
        {
            var f = sys.Fields.Fields[k];
            var p = $"fields.fields[{k}]";
            rules.NonNegative("negative_weight", $"{p}.weight", f.Weight);
            // `P' = vd + (1 − vc)·P`: a compression of 1 leaves no pupil.
            foreach (var (name, x) in new[] { ("vcx", f.Vignetting.Vcx), ("vcy", f.Vignetting.Vcy) })
            {
                if (double.IsFinite(x) && !(x >= 0 && x < 1))
                    rules.Add("invalid_vignetting", $"{p}.vignetting.{name}", $"a pupil compression must lie in [0, 1) (got {Fmt.D(x)})");
            }
        }
        if (double.IsFinite(sys.Environment.TemperatureC) && sys.Environment.TemperatureC <= -273.15)
            rules.Add("invalid_environment", "environment.temperature_c", "the temperature must be above absolute zero");
        rules.NonNegative("invalid_environment", "environment.pressure_atm", sys.Environment.PressureAtm);
        if (sys.RayAiming is RayAiming.Real real) rules.Positive("invalid_ray_aiming", "ray_aiming.tolerance", real.Tolerance);
        output.AddRange(rules.Take());

        var ids = new HashSet<SurfaceID>();
        for (var i = 0; i < n; i++)
        {
            var s = sys.Surfaces[i];
            if (!ids.Add(s.Id))
                output.Add(new ModelIssue(IssueSeverity.Error, i, $"duplicate surface id {s.Id.Value}"));
            if (i > 0 && !double.IsFinite(s.Thickness) && i != n - 1)
                output.Add(new ModelIssue(IssueSeverity.Error, i, "only the object may have infinite thickness"));
            if (s.Coating is { } c && !sys.Coatings.ContainsKey(c))
                output.Add(new ModelIssue(IssueSeverity.Error, i, $"unknown coating '{c}'"));
            if (s.SemiDiameter is SemiDiameter.Fixed fx && !(fx.Value > 0))
                output.Add(new ModelIssue(IssueSeverity.Error, i, "fixed semi-diameter must be > 0"));
            if ((i == 0 || i == n - 1) && s.IsCoordinateBreak)
                output.Add(new ModelIssue(IssueSeverity.Error, i, "object/image cannot be coordinate breaks"));
            var shape = ShapeIssues(s.Shape, $"surfaces[{i}].shape", i);
            // The JSON check stops at the first non-finite value; the shape
            // rules report each one with its full path, so they replace it.
            if (shape.Any(x => x.Message.EndsWith(NonFiniteSuffix, StringComparison.Ordinal)))
            {
                var prefix = $"surfaces[{i}].shape";
                output.RemoveAll(j => json.Contains(j) && j.Surface == i && j.Message.StartsWith(prefix, StringComparison.Ordinal));
            }
            output.AddRange(shape);
            var path = $"surfaces[{i}]";
            rules.Surface = i;
            rules.Material(s.Material, $"{path}.material");
            if (s.Aperture is { } ap) rules.Aperture(ap, $"{path}.aperture");
            if (s.ElementPlacement is { } e)
            {
                var end = sys.SurfaceIndexOf(e.Last);
                if (i == 0 || (end is { } en ? en < i || en + 1 == n : true))
                    rules.Add("invalid_element_range", $"{path}.element_placement.last", "expected an interior ordered range");
                if (e.Pivot is { } pivot && sys.SurfaceIndexOf(pivot) is null)
                    rules.Add("missing_surface", $"{path}.element_placement.pivot", "missing pivot vertex");
            }
            if (s.Scatter is { } sc) rules.Scatter(sc, $"{path}.scatter");
            if (s.Edge.MechanicalSemiDiameter is { } ms)
                rules.Positive("invalid_mechanical_semi_diameter", $"{path}.edge.mechanical_semi_diameter", ms);
            if (s.Edge.EdgeScatter is { } es) rules.Scatter(es, $"{path}.edge.edge_scatter");
            output.AddRange(rules.Take());
        }
        if (sys.Surfaces[n - 1].Interaction == Interaction.Reflect)
            output.Add(new ModelIssue(IssueSeverity.Error, n - 1, "the image surface cannot be a mirror"));
        rules.Surface = null;
        foreach (var name in sys.Coatings.Keys)
            rules.Coating(sys.Coatings[name], $"coatings[{RuleIssues.Quoted(name)}]");
        for (var k = 0; k < sys.Mechanics.Count; k++)
        {
            var p = $"mechanics[{k}]";
            void Attached(string key, SurfaceID id)
            {
                if (sys.SurfaceIndexOf(id) is null) rules.Add("missing_surface", $"{p}.{key}", $"no surface with id {id.Value}");
            }
            switch (sys.Mechanics[k])
            {
                case MechanicalPart.Barrel b:
                    Attached("from", b.From);
                    Attached("to", b.To);
                    rules.Positive("invalid_mechanics", $"{p}.inner_radius", b.InnerRadius);
                    rules.Scatter(b.Finish, $"{p}.finish");
                    break;
                case MechanicalPart.AnnularBaffle a:
                    Attached("surface", a.Surface);
                    rules.NonNegative("invalid_mechanics", $"{p}.inner_radius", a.InnerRadius);
                    rules.Positive("invalid_mechanics", $"{p}.outer_radius", a.OuterRadius);
                    if (a.InnerRadius >= a.OuterRadius)
                        rules.Add("invalid_mechanics", p, "the inner radius must be below the outer radius");
                    rules.Scatter(a.Finish, $"{p}.finish");
                    break;
            }
        }
        if (sys.Sensor is { } sensor)
        {
            if (sensor.PixelsX == 0 || sensor.PixelsY == 0) rules.Add("invalid_sensor", "sensor", "the sensor needs pixels");
            rules.Positive("invalid_sensor", "sensor.pitch_x", sensor.PitchX);
            rules.Positive("invalid_sensor", "sensor.pitch_y", sensor.PitchY);
            rules.Increasing("invalid_sensor", "sensor.spectral_response", sensor.SpectralResponse);
            for (var k = 0; k < sensor.SpectralResponse.Count; k++)
            {
                rules.Positive("invalid_sensor", $"sensor.spectral_response[{k}][0]", sensor.SpectralResponse[k].Um);
                rules.NonNegative("invalid_sensor", $"sensor.spectral_response[{k}][1]", sensor.SpectralResponse[k].Value);
            }
        }
        output.AddRange(rules.Take());
        for (var k = 0; k < sys.Configurations.Count; k++)
        {
            foreach (var o in sys.Configurations[k].Overrides)
            {
                if (o.Path.UnresolvedReason(sys) is { } why)
                    output.Add(new ModelIssue(IssueSeverity.Error, $"configuration {k}: invalid parameter path {o.Path}: {why}"));
                else if (double.IsFinite(o.Value) && o.Path.ValueRejection(o.Value) is { } rej)
                    output.Add(new ModelIssue(IssueSeverity.Error, $"configuration {k}: invalid value {Fmt.D(o.Value)} for {o.Path}: {rej}"));
            }
        }
        output.AddRange(sys.SolveIssues());
        return output;
    }

    /// <summary>
    /// The engine's structural rules of a surface shape (CONVENTIONS §13, SURFACES §7): finite
    /// values, positive normalisation radius and design wavelength, nonzero design order, a
    /// plane/conic/even-asphere diffractive base beneath any sag overlays, and the coefficient limits.
    /// Messages follow the engine's <c>[code] path: message</c> (non-finite values use the JSON check's wording).
    /// </summary>
    internal static List<ModelIssue> ShapeIssues(SurfaceShape shape, string path, int surface)
    {
        var output = new List<ModelIssue>();
        void Issue(string code, string p, string message) =>
            output.Add(new ModelIssue(IssueSeverity.Error, surface, $"[{code}] {p}: {message}"));
        bool Finite(string name, double v)
        {
            if (double.IsFinite(v)) return true;
            output.Add(new ModelIssue(IssueSeverity.Error, surface, $"{path}.{name} = {Fmt.D(v)}{NonFiniteSuffix}"));
            return false;
        }
        void Positive(string code, string name, double v)
        {
            if (Finite(name, v) && v <= 0) Issue(code, $"{path}.{name}", $"must be > 0 (got {Fmt.D(v)})");
        }
        void Series(string name, ValueList<double> a, int max)
        {
            if (a.Count > max) Issue("coefficient_count", $"{path}.{name}", $"at most {max} coefficients");
            for (var k = 0; k < a.Count; k++) Finite($"{name}[{k}]", a[k]);
        }
        void NormRadius(string name, double r) => Positive("invalid_norm_radius", name, r);

        switch (shape)
        {
            case PlaneShape or CoordinateBreakShape: break;
            case ParaxialShape p:
                if (Finite("focal_length", p.FocalLength) && p.FocalLength == 0)
                    Issue("invalid_focal_length", $"{path}.focal_length", "an ideal lens needs a non-zero focal length");
                break;
            case ConicShape s:
                Finite("curvature", s.VertexCurvature);
                Finite("conic", s.Conic);
                break;
            case EvenAsphereShape s:
                Finite("curvature", s.VertexCurvature);
                Finite("conic", s.Conic);
                Series("coefficients", s.Coefficients, CoefficientLimits.MaxEvenTerms);
                break;
            case OddAsphereShape s:
                Finite("curvature", s.VertexCurvature);
                Finite("conic", s.Conic);
                Series("coefficients", s.Coefficients, CoefficientLimits.MaxOddTerms);
                break;
            case QConShape s:
                Finite("curvature", s.VertexCurvature);
                Finite("conic", s.Conic);
                NormRadius("norm_radius", s.NormRadiusMm);
                Series("coefficients", s.Coefficients, CoefficientLimits.MaxQTerms);
                break;
            case QBfsShape s:
                Finite("curvature", s.VertexCurvature);
                NormRadius("norm_radius", s.NormRadiusMm);
                Series("coefficients", s.Coefficients, CoefficientLimits.MaxQTerms);
                break;
            case XyPolynomialShape s:
                Finite("curvature", s.VertexCurvature);
                Finite("conic", s.Conic);
                NormRadius("norm_radius", s.NormRadiusMm);
                Series("coefficients", s.Coefficients, CoefficientLimits.MaxXYTerms);
                break;
            case ZernikeSagShape s:
                Finite("curvature", s.VertexCurvature);
                Finite("conic", s.Conic);
                NormRadius("norm_radius", s.NormRadiusMm);
                Series("coefficients", s.Coefficients, s.Ordering.MaxTerms());
                break;
            case SagOverlayShape s:
                if (s.Base is CoordinateBreakShape or ParaxialShape or DiffractiveShape)
                    Issue("invalid_sag_overlay_base", $"{path}.base", "expected a geometric sag base");
                output.AddRange(ShapeIssues(s.Base, $"{path}.base", surface));
                NormRadius("norm_radius", s.NormRadiusMm);
                Positive("invalid_wavelength", "test_wavelength_um", s.TestWavelengthUm);
                Finite("astigmatism", s.Astigmatism);
                Finite("astigmatism_angle_deg", s.AstigmatismAngleDeg);
                Finite("spherical", s.Spherical);
                break;
            case DiffractiveShape s:
                var geometry = s.Base;
                while (geometry is SagOverlayShape inner) geometry = inner.Base;
                if (geometry is PlaneShape or ConicShape or EvenAsphereShape) output.AddRange(ShapeIssues(s.Base, $"{path}.base", surface));
                else Issue("invalid_diffractive_base", $"{path}.base", "expected plane, conic or even_asphere");
                NormRadius("phase.norm_radius", s.Phase.NormRadius);
                Positive("invalid_design_wavelength", "phase.design_wavelength_um", s.Phase.DesignWavelengthUm);
                if (s.Phase.DesignOrder == 0)
                    Issue("invalid_diffraction_order", $"{path}.phase.design_order", "design order must be nonzero");
                Series("phase.coefficients", s.Phase.Coefficients, CoefficientLimits.MaxPhaseTerms);
                break;
            case ToroidShape s:
                Finite("curvature_x", s.CurvatureX);
                Finite("curvature_y", s.CurvatureY);
                Finite("conic_y", s.ConicY);
                Series("y_asphere", s.YAsphere, CoefficientLimits.MaxAxisTerms);
                break;
            case BiconicShape s:
                Finite("curvature_x", s.CurvatureX);
                Finite("curvature_y", s.CurvatureY);
                Finite("conic_x", s.ConicX);
                Finite("conic_y", s.ConicY);
                Series("x_asphere", s.XAsphere, CoefficientLimits.MaxAxisTerms);
                Series("y_asphere", s.YAsphere, CoefficientLimits.MaxAxisTerms);
                break;
        }
        return output;
    }

    /// <summary>
    /// The first value that has no JSON form (the encoder stops there), as an error located by
    /// its path in the JSON tree.
    /// </summary>
    internal static List<ModelIssue> NonRepresentableValueIssues(OpticalSystem sys)
    {
        var path = new List<object>();
        var found = FirstNonFinite(sys.ToJson(), path);
        if (found is not { } value) return [];
        int? surface = path.Count >= 2 && path[0] is "surfaces" && path[1] is int s ? s : null;
        var field = string.Concat(path.Select(p => p is int i ? $"[{i}]" : $".{p}")).TrimStart('.');
        return [new ModelIssue(IssueSeverity.Error, surface, $"{field} = {Fmt.D(value)}{NonFiniteSuffix}")];
    }

    private static double? FirstNonFinite(JsonNode? node, List<object> path)
    {
        switch (node)
        {
            case JsonObject o:
                foreach (var (k, v) in o)
                {
                    path.Add(k);
                    if (FirstNonFinite(v, path) is { } hit) return hit;
                    path.RemoveAt(path.Count - 1);
                }
                return null;
            case JsonArray a:
                for (var i = 0; i < a.Count; i++)
                {
                    path.Add(i);
                    if (FirstNonFinite(a[i], path) is { } hit) return hit;
                    path.RemoveAt(path.Count - 1);
                }
                return null;
            case JsonValue v when v.TryGetValue<double>(out var d) && !double.IsFinite(d): return d;
            default: return null;
        }
    }
}

/// <summary>
/// Collects problems of numbers and references in the engine's <c>[code] path: message</c>
/// form, with the engine's codes and JSON paths (<c>validate.rs</c>). A value that is not
/// finite is left to the JSON check.
/// </summary>
internal sealed class RuleIssues
{
    private List<ModelIssue> issues = [];

    /// <summary>Surface index of the issues being added, if they concern one.</summary>
    public int? Surface { get; set; }

    /// <summary>The issues added so far, which it forgets.</summary>
    public List<ModelIssue> Take()
    {
        var taken = issues;
        issues = [];
        return taken;
    }

    public void Add(string code, string path, string message) =>
        issues.Add(new ModelIssue(IssueSeverity.Error, Surface, $"[{code}] {path}: {message}"));

    public void Positive(string code, string path, double v)
    {
        if (double.IsFinite(v) && v <= 0) Add(code, path, $"must be > 0 (got {Fmt.D(v)})");
    }

    public void NonNegative(string code, string path, double v)
    {
        if (double.IsFinite(v) && v < 0) Add(code, path, $"must be ≥ 0 (got {Fmt.D(v)})");
    }

    /// <summary>The wavelengths of a <c>(λ, value)</c> table increase strictly: the engine's interpolants search the table in that order.</summary>
    public void Increasing(string code, string path, ValueList<SpectralSample> table)
    {
        for (var k = 1; k < table.Count; k++)
        {
            if (table[k].Um <= table[k - 1].Um)
                Add(code, $"{path}[{k}][0]", $"wavelengths must increase ({Fmt.D(table[k].Um)} follows {Fmt.D(table[k - 1].Um)})");
        }
    }

    public void Material(MaterialRef m, string path)
    {
        switch (m)
        {
            case MaterialRef.CatalogGlass g:
                foreach (var (name, v, max) in new[] { ("dn_d", g.Offset.DnD, MaterialOffset.MaxDnD), ("dv_d", g.Offset.DvD, MaterialOffset.MaxDvD) })
                {
                    if (double.IsFinite(v) && Math.Abs(v) > max)
                        Add("invalid_material", $"{path}.offset.{name}", $"must be within ±{Fmt.D(max)} (got {Fmt.D(v)})");
                }
                break;
            case MaterialRef.ModelGlass g:
                // Its dispersion is the refractivity `n_d − 1` over `V_d`.
                if (double.IsFinite(g.Nd) && g.Nd <= 1) Add("invalid_material", $"{path}.nd", $"must be > 1 (got {Fmt.D(g.Nd)})");
                Positive("invalid_material", $"{path}.vd", g.Vd);
                break;
            case MaterialRef.ConstantIndex c: Positive("invalid_material", $"{path}.n", c.N); break;
        }
    }

    public void Aperture(ApertureShape a, string path)
    {
        switch (a)
        {
            case ApertureShape.Circular c: Positive("invalid_aperture", $"{path}.radius", c.Radius); break;
            case ApertureShape.Annular x:
                NonNegative("invalid_aperture", $"{path}.inner_radius", x.InnerRadius);
                Positive("invalid_aperture", $"{path}.outer_radius", x.OuterRadius);
                if (x.InnerRadius >= x.OuterRadius)
                    Add("invalid_aperture", path, $"inner radius {Fmt.D(x.InnerRadius)} must be below the outer radius {Fmt.D(x.OuterRadius)}");
                break;
            case ApertureShape.Rectangular r:
                Positive("invalid_aperture", $"{path}.half_x", r.HalfX);
                Positive("invalid_aperture", $"{path}.half_y", r.HalfY);
                break;
            case ApertureShape.Elliptical e:
                Positive("invalid_aperture", $"{path}.semi_x", e.SemiX);
                Positive("invalid_aperture", $"{path}.semi_y", e.SemiY);
                break;
        }
    }

    public void Scatter(ScatterModel m, string path)
    {
        void Tis(double v)
        {
            if (double.IsFinite(v) && !(v >= 0 && v <= 1))
                Add("invalid_scatter", $"{path}.tis", $"a total integrated scatter must lie in [0, 1] (got {Fmt.D(v)})");
        }
        switch (m)
        {
            case ScatterModel.Lambertian l: Tis(l.Tis); break;
            case ScatterModel.Abg a:
                NonNegative("invalid_scatter", $"{path}.a", a.A);
                NonNegative("invalid_scatter", $"{path}.b", a.B);
                break;
            case ScatterModel.Gaussian g:
                Tis(g.Tis);
                Positive("invalid_scatter", $"{path}.sigma_slope", g.SigmaSlope);
                break;
        }
    }

    public void Coating(CoatingDef c, string path)
    {
        switch (c)
        {
            case CoatingDef.ThinFilm f:
                Positive("invalid_coating", $"{path}.reference_wavelength_um", f.ReferenceWavelengthUm);
                for (var j = 0; j < f.Layers.Count; j++)
                {
                    var layer = f.Layers[j];
                    var p = $"{path}.layers[{j}]";
                    Material(layer.Material, $"{p}.material");
                    NonNegative("invalid_coating", $"{p}.thickness_um", layer.ThicknessUm);
                    if (layer.QuarterWaves is { } q) NonNegative("invalid_coating", $"{p}.quarter_waves", q);
                    NonNegative("invalid_coating", $"{p}.extinction", layer.Extinction);
                }
                if (f.MirrorSubstrate is { } sub)
                {
                    Positive("invalid_coating", $"{path}.mirror_substrate[0]", sub.N);
                    NonNegative("invalid_coating", $"{path}.mirror_substrate[1]", sub.K);
                }
                break;
            case CoatingDef.Ideal i:
                if (double.IsFinite(i.Reflectance) && double.IsFinite(i.Transmittance)
                    && !(i.Reflectance >= 0 && i.Transmittance >= 0 && i.Reflectance + i.Transmittance <= 1 + 1e-12))
                    Add("invalid_coating", path, $"R = {Fmt.D(i.Reflectance)} and T = {Fmt.D(i.Transmittance)} must be ≥ 0 with R + T ≤ 1");
                break;
            case CoatingDef.Tabulated t:
                if (t.Reflectance.Count == 0)
                    Add("invalid_coating", $"{path}.reflectance", "a tabulated coating needs at least one sample");
                Increasing("invalid_coating", $"{path}.reflectance", t.Reflectance);
                for (var j = 0; j < t.Reflectance.Count; j++)
                {
                    var sample = t.Reflectance[j];
                    var p = $"{path}.reflectance[{j}]";
                    Positive("invalid_coating", $"{p}[0]", sample.Um);
                    if (double.IsFinite(sample.Value) && !(sample.Value >= 0 && sample.Value <= 1))
                        Add("invalid_coating", p, $"reflectance {Fmt.D(sample.Value)} outside [0, 1]");
                }
                break;
        }
    }

    /// <summary>A map key as the engine's paths write it: in double quotes, with quotes, backslashes, newlines and tabs escaped.</summary>
    public static string Quoted(string key)
    {
        var sb = new System.Text.StringBuilder("\"");
        foreach (var c in key)
        {
            sb.Append(c switch
            {
                '"' => "\\\"",
                '\\' => "\\\\",
                '\n' => "\\n",
                '\r' => "\\r",
                '\t' => "\\t",
                _ => c.ToString(),
            });
        }
        return sb.Append('"').ToString();
    }
}
