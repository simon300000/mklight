namespace MKLight.Model;

/// <summary>
/// Reading and writing the value a <see cref="ParamPath"/> names, with the same rules as
/// the engine's <c>ParamPath::get</c> / <c>set</c> (no optics). Used for variables,
/// solve outcomes and generic parameter forms.
/// </summary>
public sealed partial record OpticalSystem
{
    /// <summary>
    /// The current value of <paramref name="path"/>, or <c>null</c> when the system has no such
    /// parameter (an automatic semi-diameter has no value).
    /// </summary>
    public double? ValueAt(ParamPath path)
    {
        if (!path.Resolves(this)) return null;
        if (path.SurfaceId is { } id)
        {
            if (SurfaceIndexOf(id) is not { } i) return null;
            var s = Surfaces[i];
            switch (path)
            {
                case ParamPath.Curvature: return s.Shape.Curvature;
                case ParamPath.Thickness: return s.Thickness;
                case ParamPath.Conic: return s.Shape.ConicConstant;
                case ParamPath.Asphere a:
                    var asphere = s.Shape.AsphereCoefficients;
                    return asphere.IsValidIndex(a.Index) ? asphere[a.Index] : 0;
                case ParamPath.ShapeCoefficient c:
                    var coefficients = s.Shape.CoefficientsIn(c.Set) ?? [];
                    return coefficients.IsValidIndex(c.Index) ? coefficients[c.Index] : 0;
                case ParamPath.ShapeParam p: return s.Shape.ShapeParameterValue(p.Parameter);
                case ParamPath.ParaxialFocalLength: return s.Shape is ParaxialShape f ? f.FocalLength : null;
                case ParamPath.ModelNd or ParamPath.ModelVd or ParamPath.ModelDpgf:
                    if (s.Material is not MaterialRef.ModelGlass m) return null;
                    return path is ParamPath.ModelNd ? m.Nd : path is ParamPath.ModelVd ? m.Vd : m.Dpgf;
                case ParamPath.MaterialDnD or ParamPath.MaterialDvD:
                    if (s.Material is not MaterialRef.CatalogGlass g) return null;
                    return path is ParamPath.MaterialDnD ? g.Offset.DnD : g.Offset.DvD;
                case ParamPath.DecenterX: return s.Placement?.DecenterX ?? 0;
                case ParamPath.DecenterY: return s.Placement?.DecenterY ?? 0;
                case ParamPath.TiltX: return s.Placement?.TiltX ?? 0;
                case ParamPath.TiltY: return s.Placement?.TiltY ?? 0;
                case ParamPath.TiltZ: return s.Placement?.TiltZ ?? 0;
                case ParamPath.SemiDiameter: return s.SemiDiameter.FixedValue;
                case ParamPath.Irregularity ir:
                    return s.Shape.SagShape is SagOverlayShape o ? (ir.Mode == IrregularityMode.Astigmatism ? o.Astigmatism : o.Spherical) : 0;
                case ParamPath.ElementPlacement e:
                    return s.ElementPlacement is { } ep ? e.Component.Get(ep.Motion) : 0;
                default: return null;
            }
        }
        switch (path)
        {
            case ParamPath.ApertureValue: return Aperture.Value;
            case ParamPath.FieldX p: return Fields.Fields[p.Index].X;
            case ParamPath.FieldY p: return Fields.Fields[p.Index].Y;
            case ParamPath.FieldWeight p: return Fields.Fields[p.Index].Weight;
            case ParamPath.WavelengthUm p: return Wavelengths.Wavelengths[p.Index].Um;
            case ParamPath.WavelengthWeight p: return Wavelengths.Wavelengths[p.Index].Weight;
            case ParamPath.TemperatureC: return Environment.TemperatureC;
            case ParamPath.PressureAtm: return Environment.PressureAtm;
            default: return null;
        }
    }

    /// <summary>
    /// The system with the value <paramref name="path"/> names set, or <c>null</c> when the system has no
    /// such parameter. Like the engine, setting a curvature on a plane makes it a conic (a conic on a
    /// plane is refused), curvature, conic and asphere paths address a diffractive surface's base, a
    /// coefficient past the end extends the list with zeros, an index at or beyond the shape's limit is
    /// refused, and a diffraction order must be an integral <c>i32</c>.
    /// </summary>
    public OpticalSystem? SetValue(ParamPath path, double v)
    {
        if (path.SurfaceId is { } id)
        {
            if (SurfaceIndexOf(id) is not { } i) return null;
            var edited = EditSurface(Surfaces[i], path, v);
            return edited is null ? null : this with { Surfaces = Surfaces.With(i, edited) };
        }
        switch (path)
        {
            case ParamPath.ApertureValue:
                return Aperture.Value is null ? null : this with { Aperture = ApertureSpec.Of(Aperture.Kind, v) };
            case ParamPath.FieldX p: return EditField(p.Index, f => f with { X = v });
            case ParamPath.FieldY p: return EditField(p.Index, f => f with { Y = v });
            case ParamPath.FieldWeight p: return EditField(p.Index, f => f with { Weight = v });
            case ParamPath.WavelengthUm p: return EditWavelength(p.Index, w => w with { Um = v });
            case ParamPath.WavelengthWeight p: return EditWavelength(p.Index, w => w with { Weight = v });
            case ParamPath.TemperatureC: return this with { Environment = Environment with { TemperatureC = v } };
            case ParamPath.PressureAtm: return this with { Environment = Environment with { PressureAtm = v } };
            default: return null;
        }
    }

    private OpticalSystem? EditField(int index, Func<Field, Field> edit) =>
        Fields.Fields.IsValidIndex(index) ? this with { Fields = Fields with { Fields = Fields.Fields.With(index, edit) } } : null;

    private OpticalSystem? EditWavelength(int index, Func<Wavelength, Wavelength> edit) =>
        Wavelengths.Wavelengths.IsValidIndex(index)
            ? this with { Wavelengths = Wavelengths with { Wavelengths = Wavelengths.Wavelengths.With(index, edit) } }
            : null;

    private Surface? EditSurface(Surface s, ParamPath path, double v)
    {
        Surface Placed(Func<Placement, Placement> f) => s with { Placement = f(s.Placement ?? new Placement { RestoreAfter = true }) };
        switch (path)
        {
            case ParamPath.Curvature: return s.Shape.WithCurvature(v) is { } c ? s with { Shape = c } : null;
            case ParamPath.Conic:
                return s.Shape.HasConic && s.Shape.WithConic(v) is { } k ? s with { Shape = k } : null;
            case ParamPath.Thickness: return s with { Thickness = v };
            case ParamPath.Asphere a: return s.Shape.WithAsphereCoefficient(v, a.Index) is { } sh ? s with { Shape = sh } : null;
            case ParamPath.ShapeCoefficient sc:
                if (sc.Set.MaxCount(s.Shape) is not { } max || sc.Index < 0 || sc.Index >= max || s.Shape.CoefficientsIn(sc.Set) is not { } list)
                    return null;
                return s.Shape.WithCoefficients(SurfaceShape.SetPadded(list, sc.Index, v), sc.Set) is { } shaped ? s with { Shape = shaped } : null;
            case ParamPath.ShapeParam p: return s.Shape.WithShapeParameter(p.Parameter, v) is { } sp ? s with { Shape = sp } : null;
            case ParamPath.ParaxialFocalLength: return s.Shape is ParaxialShape ? s with { Shape = new ParaxialShape(v) } : null;
            case ParamPath.ModelNd or ParamPath.ModelVd or ParamPath.ModelDpgf:
                if (s.Material is not MaterialRef.ModelGlass m) return null;
                return s with
                {
                    Material = path is ParamPath.ModelNd ? m with { Nd = v } : path is ParamPath.ModelVd ? m with { Vd = v } : m with { Dpgf = v },
                };
            case ParamPath.MaterialDnD or ParamPath.MaterialDvD:
                if (s.Material is not MaterialRef.CatalogGlass g) return null;
                return s with { Material = g with { Offset = path is ParamPath.MaterialDnD ? g.Offset with { DnD = v } : g.Offset with { DvD = v } } };
            case ParamPath.DecenterX: return Placed(p => p with { DecenterX = v });
            case ParamPath.DecenterY: return Placed(p => p with { DecenterY = v });
            case ParamPath.TiltX: return Placed(p => p with { TiltX = v });
            case ParamPath.TiltY: return Placed(p => p with { TiltY = v });
            case ParamPath.TiltZ: return Placed(p => p with { TiltZ = v });
            case ParamPath.SemiDiameter: return s with { SemiDiameter = new SemiDiameter.Fixed(v) };
            case ParamPath.Irregularity ir:
                var radius = s.Aperture?.BoundingRadius ?? s.SemiDiameter.FixedValue;
                return s.Shape.WithIrregularity(ir.Mode, v, radius, PrimaryWavelengthUm) is { } o ? s with { Shape = o } : null;
            case ParamPath.ElementPlacement e:
                if (!path.Resolves(this)) return null;
                var placement = s.ElementPlacement ?? new ElementPlacement { Last = e.Last, Pivot = e.Pivot };
                return s with { ElementPlacement = placement with { Motion = e.Component.With(placement.Motion, v) } };
            default: return null;
        }
    }
}

public abstract partial record ParamPath
{
    /// <summary>
    /// A short human label, e.g. <c>Radius 3</c>, <c>Thickness 4</c>, <c>A6 on 2</c>, <c>Field 2 y</c>, with the
    /// surface given by its index in <paramref name="system"/>.
    /// </summary>
    public string Label(OpticalSystem system) => Label(system, english: false);

    /// <summary>The label in English whatever the UI language: for what goes to the engine and into exported files.</summary>
    public string EnglishLabel(OpticalSystem system) => Label(system, english: true);

    string Label(OpticalSystem system, bool english)
    {
        var n = SurfaceId is { } sid && system.SurfaceIndexOf(sid) is { } idx ? idx.ToString() : "?";
        switch (this)
        {
            case Curvature c:
                // The y section of a toroid or biconic.
                var anamorphic = system.SurfaceIndexOf(c.Surface) is { } k && system.Surfaces[k].Shape.Kind is ShapeKind.Toroid or ShapeKind.Biconic;
                return anamorphic ? T(english, "RadiusY", "Radius y {0}", n) : T(english, "Radius", "Radius {0}", n);
            case Thickness: return T(english, "Thickness", "Thickness {0}", n);
            case Conic: return T(english, "Conic", "Conic {0}", n);
            case Asphere a: return T(english, "AsphereOn", "A{0} on {1}", 2 * a.Index + 4, n);
            case ShapeCoefficient sc:
                return sc.Set switch
                {
                    CoefficientSet.Odd => T(english, "OddTermOn", "Odd {0} on {1}", sc.Set.TermLabel(sc.Index), n),
                    CoefficientSet.Phase => T(english, "PhaseTermOn", "Phase {0} on {1}", sc.Set.TermLabel(sc.Index), n),
                    _ => T(english, "TermOn", "{0} on {1}", sc.Set.TermLabel(sc.Index), n),
                };
            case ShapeParam p: return T(english, "TermOn", "{0} on {1}", p.Parameter.Title(), n);
            case ParaxialFocalLength: return T(english, "FocalLength", "Focal length {0}", n);
            case ModelNd: return $"n_d {n}";
            case ModelVd: return $"V_d {n}";
            case ModelDpgf: return $"ΔP_g,F {n}";
            case MaterialDnD: return $"Δn_d {n}";
            case MaterialDvD: return $"ΔV_d {n}";
            case DecenterX: return T(english, "DecenterX", "Decenter X {0}", n);
            case DecenterY: return T(english, "DecenterY", "Decenter Y {0}", n);
            case TiltX: return T(english, "TiltX", "Tilt X {0}", n);
            case TiltY: return T(english, "TiltY", "Tilt Y {0}", n);
            case TiltZ: return T(english, "TiltZ", "Tilt Z {0}", n);
            case SemiDiameter: return T(english, "SemiDiameter", "Semi-diameter {0}", n);
            case ApertureValue: return T(english, "Aperture", "Aperture");
            case FieldX f: return T(english, "FieldX", "Field {0} x", f.Index + 1);
            case FieldY f: return T(english, "FieldY", "Field {0} y", f.Index + 1);
            case FieldWeight f: return T(english, "FieldWeight", "Field {0} weight", f.Index + 1);
            case WavelengthUm w: return $"λ{w.Index + 1}";
            case WavelengthWeight w: return T(english, "WavelengthWeight", "λ{0} weight", w.Index + 1);
            case TemperatureC: return T(english, "Temperature", "Temperature");
            case PressureAtm: return T(english, "Pressure", "Pressure");
            case Irregularity ir: return ir.Mode == IrregularityMode.Astigmatism ? T(english, "AstigIrregularity", "Astig. irregularity {0}", n) : T(english, "SphIrregularity", "Sph. irregularity {0}", n);
            case ElementPlacement e:
                var m = system.SurfaceIndexOf(e.Last) is { } li ? li.ToString() : "?";
                return $"{e.Component.Title()} {n}–{m}";
            default: return "";
        }
    }

    /// <summary>A label's format in the UI language (<see cref="ModelText"/>), else the English.</summary>
    static string T(bool inEnglish, string key, string english, params object?[] args) =>
        inEnglish ? string.Format(System.Globalization.CultureInfo.InvariantCulture, english, args) : ModelText.Format("Model_Param_" + key, english, args);

    /// <summary>Whether the value is a curvature <c>c = 1/R</c> (1/mm): shown and bounded as a radius by default.</summary>
    public bool IsCurvature =>
        this is Curvature or ShapeParam { Parameter: ShapeParameter.CurvatureX or ShapeParameter.CurvatureY };

    /// <summary>
    /// The unit of the parameter's value (and of its bounds), <c>""</c> when it is dimensionless:
    /// <c>mm</c>, <c>1/mm</c>, <c>mm⁻³</c> (A₄), <c>°</c>, <c>µm</c>, <c>rad</c>, … Polynomial
    /// coefficients carry the inverse powers that make their term a length (docs/SURFACES.md).
    /// </summary>
    public string Unit(OpticalSystem system)
    {
        switch (this)
        {
            case Curvature: return "1/mm";
            case Thickness or DecenterX or DecenterY or SemiDiameter or ParaxialFocalLength: return "mm";
            case Conic or ModelNd or ModelVd or ModelDpgf or MaterialDnD or MaterialDvD or FieldWeight or WavelengthWeight: return "";
            case Asphere a: return InverseMM(2 * a.Index + 3);
            case ShapeCoefficient c:
                return c.Set switch
                {
                    CoefficientSet.Odd => InverseMM(c.Index),
                    CoefficientSet.XAsphere or CoefficientSet.YAsphere => InverseMM(2 * c.Index + 3),
                    CoefficientSet.Phase => "rad",
                    _ => "mm",
                };
            case ShapeParam p:
                return p.Parameter switch
                {
                    ShapeParameter.CurvatureX or ShapeParameter.CurvatureY => "1/mm",
                    ShapeParameter.NormRadius => "mm",
                    ShapeParameter.AstigmatismAngleDeg => "°",
                    ShapeParameter.DesignWavelengthUm => "µm",
                    _ => "",
                };
            case TiltX or TiltY or TiltZ: return "°";
            case Irregularity: return "fringes";
            case ElementPlacement e: return e.Component is PlacementComponent.DecenterX or PlacementComponent.DecenterY ? "mm" : "°";
            case ApertureValue: return system.Aperture.Kind.Unit();
            case FieldX or FieldY: return system.Fields.Kind.Unit();
            case WavelengthUm: return "µm";
            case TemperatureC: return "°C";
            case PressureAtm: return "atm";
            default: return "";
        }
    }

    /// <summary><c>mm⁻ⁿ</c> with a superscript exponent; <c>""</c> for n = 0, <c>1/mm</c> for n = 1.</summary>
    internal static string InverseMM(int n)
    {
        if (n <= 0) return "";
        if (n == 1) return "1/mm";
        return "mm⁻" + XYTerm.Superscript(n);
    }

    /// <summary>Whether the path names an existing parameter of <paramref name="system"/>.</summary>
    public bool Resolves(OpticalSystem system) => UnresolvedReason(system) is null;

    /// <summary>
    /// Why the path does not name a parameter of <paramref name="system"/> (<c>null</c> when it does).
    /// The same rules and messages as the engine's <c>ParamPath::get</c>, which its <c>validate</c>
    /// applies to every configuration override.
    /// </summary>
    public string? UnresolvedReason(OpticalSystem system)
    {
        if (SurfaceId is { } id)
        {
            if (system.SurfaceIndexOf(id) is not { } i) return $"no surface with id {id.Value}";
            var s = system.Surfaces[i];
            switch (this)
            {
                case Curvature: return s.Shape.HasCurvature ? null : "surface has no curvature";
                case Conic: return s.Shape.HasConic ? null : "surface has no conic";
                case Asphere a:
                    if (!s.Shape.HasAsphereCoefficients) return "surface is not an even asphere";
                    return a.Index < CoefficientLimits.MaxEvenTerms ? null : "[coefficient_count] index exceeds shape limit";
                case ShapeCoefficient c:
                    if (c.Set.MaxCount(s.Shape) is not { } max) return "coefficient set does not belong to shape";
                    return c.Index < max ? null : "[coefficient_count] index exceeds shape limit";
                case ShapeParam p: return s.Shape.ShapeParameterValue(p.Parameter) is null ? "scalar does not belong to shape" : null;
                case ParaxialFocalLength: return s.Shape is ParaxialShape ? null : "surface is not paraxial";
                case ModelNd or ModelVd or ModelDpgf: return s.Material is MaterialRef.ModelGlass ? null : "material is not a model glass";
                case MaterialDnD or MaterialDvD: return s.Material is MaterialRef.CatalogGlass ? null : "material is not a catalogue glass";
                case SemiDiameter: return s.SemiDiameter.FixedValue is null ? "semi-diameter is automatic" : null;
                case ElementPlacement e:
                    var end = system.SurfaceIndexOf(e.Last);
                    if (end is not { } en || i <= 0 || i > en || en + 1 >= system.Surfaces.Count
                        || (e.Pivot is { } pv && system.SurfaceIndexOf(pv) is null))
                        return "expected an interior, ordered range and existing pivot";
                    if (s.ElementPlacement is { } existing && (existing.Last != e.Last || existing.Pivot != e.Pivot))
                        return "another range or pivot starts here";
                    return null;
                default: return null; // thickness, decenters, tilts and irregularity exist on every surface
            }
        }
        switch (this)
        {
            case FieldX or FieldY or FieldWeight:
                return system.Fields.Fields.IsValidIndex(FieldIndex!.Value) ? null : "no such field";
            case WavelengthUm or WavelengthWeight:
                return system.Wavelengths.Wavelengths.IsValidIndex(WavelengthIndex!.Value) ? null : "no such wavelength";
            case ApertureValue: return system.Aperture.Value is null ? "aperture floats by stop size" : null;
            default: return null;
        }
    }

    /// <summary>
    /// Why the engine's <c>ParamPath::set</c> would refuse the finite value <paramref name="v"/> for a path
    /// that resolves: a diffraction order must be an integral <c>i32</c>, a fixed semi-diameter positive.
    /// <c>null</c> when it accepts it.
    /// </summary>
    public string? ValueRejection(double v) => this switch
    {
        ShapeParam p => SurfaceShape.OrderRejection(p.Parameter, v),
        SemiDiameter => v > 0 ? null : "semi-diameter must be finite and > 0",
        _ => null,
    };
}
