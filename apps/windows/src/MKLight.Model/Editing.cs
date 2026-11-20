using System.Text.Json.Nodes;

namespace MKLight.Model;

/// <summary>
/// Structural edits of the surface list. They keep the stop index and surface identities
/// consistent; they never compute optics. Every edit returns the edited system (the receiver
/// is unchanged), or <c>null</c> where the edit is refused.
/// </summary>
public sealed partial record OpticalSystem
{
    /// <summary>
    /// Inserts a plane air surface at <paramref name="index"/> (1…image index; clamped) and returns the new
    /// system with the new surface's id. Surfaces at and after <paramref name="index"/> move down by one.
    /// </summary>
    public (OpticalSystem System, SurfaceID Id) InsertSurface(int index, double thickness = 0)
    {
        var i = Math.Min(Math.Max(index, 1), ImageIndex);
        var id = NextSurfaceID;
        var system = this with
        {
            Surfaces = Surfaces.Insert(i, new Surface { Id = id, Thickness = thickness }),
            Stop = i <= Stop ? Stop + 1 : Stop,
        };
        return (system, id);
    }

    /// <summary>
    /// Removes the surface at <paramref name="index"/>. The object, the image and the last remaining surface
    /// between them cannot be removed (<c>null</c>). Deleting the stop keeps the stop at the same position
    /// (clamped). Configuration overrides, solves and mechanical parts that refer to the removed surface
    /// are removed too.
    /// </summary>
    public OpticalSystem? DeleteSurface(int index)
    {
        if (index <= 0 || index >= ImageIndex || Surfaces.Count <= 3) return null;
        var removed = Surfaces[index].Id;
        var surfaces = Surfaces.RemoveAt(index);
        var stop = index < Stop ? Stop - 1 : Stop;
        stop = Math.Min(Math.Max(stop, 1), surfaces.Count - 2);
        return this with
        {
            Surfaces = surfaces,
            Stop = stop,
            Configurations = Configurations.Select(c => c with { Overrides = c.Overrides.RemoveAll(o => o.Path.SurfaceId == removed) }).ToValueList(),
            Mechanics = Mechanics.RemoveAll(m => m.Surfaces.Contains(removed)),
            Solves = Solves.RemoveAll(s => s.Surfaces.Contains(removed)),
        };
    }

    /// <summary>
    /// Removes field <paramref name="index"/> (at least one field remains; else <c>null</c>). Overrides of that
    /// field are dropped and those of later fields renumbered, so every index-addressed
    /// <see cref="ParamPath"/> keeps naming the same field.
    /// </summary>
    public OpticalSystem? RemoveField(int index)
    {
        if (!Fields.Fields.IsValidIndex(index) || Fields.Fields.Count <= 1) return null;
        var system = this with { Fields = Fields with { Fields = Fields.Fields.RemoveAt(index) } };
        return system.RemapIndexed(index, p => p.FieldIndex);
    }

    /// <summary>
    /// Removes wavelength <paramref name="index"/> (at least one remains; else <c>null</c>), keeping the primary
    /// wavelength and index-addressed overrides pointing at the same entries.
    /// </summary>
    public OpticalSystem? RemoveWavelength(int index)
    {
        if (!Wavelengths.Wavelengths.IsValidIndex(index) || Wavelengths.Wavelengths.Count <= 1) return null;
        var remaining = Wavelengths.Wavelengths.RemoveAt(index);
        var primary = Wavelengths.Primary;
        if (primary > index || (primary == index && index == remaining.Count)) primary--;
        var system = this with { Wavelengths = new WavelengthSet { Wavelengths = remaining, Primary = primary } };
        return system.RemapIndexed(index, p => p.WavelengthIndex);
    }

    private OpticalSystem RemapIndexed(int removed, Func<ParamPath, int?> index)
    {
        ParamPath? Remap(ParamPath p)
        {
            if (index(p) is not { } i) return p;
            if (i == removed) return null;
            return i > removed ? p.WithIndex(i - 1) : p;
        }
        var configurations = Configurations.Select(c => c with
        {
            Overrides = c.Overrides.Select(o => Remap(o.Path) is { } p ? (ParamOverride?)new ParamOverride(p, o.Value) : null)
                .OfType<ParamOverride>().ToValueList(),
        }).ToValueList();
        // Solves whose target or pickup source names the removed entry go; later entries are renumbered.
        var solves = new List<Solve>();
        foreach (var s in Solves)
        {
            if (Remap(s.Target) is not { } target) continue;
            if (s.Kind is SolveKind.Pickup pickup)
            {
                if (Remap(pickup.Source) is not { } source) continue;
                solves.Add(new Solve(target, pickup with { Source = source }));
            }
            else
            {
                solves.Add(new Solve(target, s.Kind));
            }
        }
        return this with { Configurations = configurations, Solves = solves.ToValueList() };
    }

    /// <summary>
    /// Drops configuration overrides, solves and mechanical parts that resolved in <paramref name="previous"/> but
    /// no longer resolve here. Call after an edit (shape, material, semi-diameter or aperture changes can
    /// invalidate them); the edit stays undoable as a whole. References that were already broken in
    /// <paramref name="previous"/> are kept, so <c>StructuralIssues()</c> still reports them.
    /// </summary>
    public OpticalSystem DropReferencesInvalidated(OpticalSystem previous)
    {
        var current = this;
        var configurations = Configurations.Select(c => c with
        {
            Overrides = c.Overrides.RemoveAll(o => o.Path.UnresolvedReason(current) is not null && o.Path.UnresolvedReason(previous) is null),
        }).ToValueList();
        static bool Broken(Solve s, OpticalSystem sys)
        {
            var fixesSemiDiameter = s.Target is ParamPath.SemiDiameter;
            var targetExists = s.Target.SurfaceId is { } t && sys.SurfaceIndexOf(t) is not null;
            if (!(fixesSemiDiameter && targetExists) && s.Target.UnresolvedReason(sys) is not null) return true;
            if (s.Kind is SolveKind.Pickup p && sys.SolveFor(p.Source) is null && p.Source.UnresolvedReason(sys) is not null) return true;
            return s.Surfaces.Any(id => sys.SurfaceIndexOf(id) is null);
        }
        var solves = Solves.RemoveAll(s => Broken(s, current) && !Broken(s, previous));
        var ids = Surfaces.Select(s => s.Id).ToHashSet();
        var before = previous.Surfaces.Select(s => s.Id).ToHashSet();
        var mechanics = Mechanics.RemoveAll(m => m.Surfaces.Any(id => !ids.Contains(id)) && m.Surfaces.All(before.Contains));
        return this with { Configurations = configurations, Solves = solves, Mechanics = mechanics };
    }

    /// <summary>Makes <paramref name="index"/> the aperture stop if it is between object and image; otherwise the system is returned unchanged.</summary>
    public OpticalSystem SetStop(int index) => index > 0 && index < ImageIndex ? this with { Stop = index } : this;

    /// <summary>
    /// Changes a surface's shape type, keeping what carries over: curvature and conic where the new type has
    /// them (a toroid or biconic gets <c>c_x = c_y = c</c>), the normalisation radius between normalised types,
    /// x/y values between toroid and biconic. A diffractive surface wraps the old shape as its base when it is
    /// a plane, conic or even asphere (else a conic with the old c and k), with an empty order +1 phase at the
    /// primary wavelength; converting a diffractive surface uses its base. A new normalisation radius is the fixed
    /// semi-diameter, else the aperture's bounding radius, else 10 mm. Overrides that no longer resolve are dropped.
    /// </summary>
    public OpticalSystem SetShapeKind(ShapeKind kind, int index)
    {
        if (!Surfaces.IsValidIndex(index)) return this;
        var surface = Surfaces[index];
        var old = surface.Shape;
        if (old.Kind != kind || kind == ShapeKind.CoordinateBreak)
        {
            var source = old.Kind == ShapeKind.Diffractive ? old.Geometry : old;
            var c = source.Curvature;
            var k = source.ConicConstant;
            var r = new[] { old.NormRadius, surface.SemiDiameter.FixedValue, surface.Aperture?.BoundingRadius }
                .FirstOrDefault(v => v is { } x && double.IsFinite(x) && x > 0) ?? 10;
            SurfaceShape shape = kind switch
            {
                ShapeKind.Plane => new PlaneShape(),
                ShapeKind.Conic => new ConicShape(c, k),
                ShapeKind.EvenAsphere => new EvenAsphereShape(c, k, source.AsphereCoefficients),
                ShapeKind.OddAsphere => new OddAsphereShape(c, k, []),
                ShapeKind.QCon => new QConShape(c, k, r, []),
                ShapeKind.QBfs => new QBfsShape(c, r, []),
                ShapeKind.Diffractive => new DiffractiveShape(
                    source is PlaneShape or ConicShape or EvenAsphereShape ? source : new ConicShape(c, k),
                    new DiffractivePhase { NormRadius = r, DesignOrder = 1, DesignWavelengthUm = PrimaryWavelengthUm, TraceOrder = 1 }),
                ShapeKind.Toroid => source is BiconicShape b
                    ? new ToroidShape(b.CurvatureX, b.CurvatureY, b.ConicY, b.YAsphere)
                    : new ToroidShape(c, c, k, []),
                ShapeKind.Biconic => source is ToroidShape t
                    ? new BiconicShape(t.CurvatureX, t.CurvatureY, 0, t.ConicY, [], t.YAsphere)
                    : new BiconicShape(c, c, k, k, [], []),
                ShapeKind.XyPolynomial => new XyPolynomialShape(c, k, r, []),
                ShapeKind.ZernikeSag => new ZernikeSagShape(c, k, r, ZernikeOrdering.Fringe, []),
                ShapeKind.Paraxial => new ParaxialShape(old is ParaxialShape p ? p.FocalLength : 100),
                _ => new CoordinateBreakShape(),
            };
            surface = surface with { Shape = shape };
        }
        if (kind == ShapeKind.CoordinateBreak && surface.Placement is null) surface = surface with { Placement = new Placement() };
        var system = this with { Surfaces = Surfaces.With(index, surface) };
        // e.g. a conic override on a surface that became a plane.
        var id = surface.Id;
        return system with
        {
            Configurations = Configurations.Select(cfg => cfg with
            {
                Overrides = cfg.Overrides.RemoveAll(o => o.Path.SurfaceId == id && o.Path.UnresolvedReason(system) is not null),
            }).ToValueList(),
        };
    }

    /// <summary>
    /// Switches the field kind. With <paramref name="conversion"/> (one value per field), every field gets its value in
    /// the new kind, so it keeps its chief ray; weights and vignetting stay. Without, the numbers stay as typed.
    /// </summary>
    public OpticalSystem SetFieldKind(FieldKind kind, SpecConversion? conversion = null)
    {
        if (kind == Fields.Kind) return this;
        var fields = Fields with { Kind = kind };
        if (conversion?.FieldValues is { } values && values.Count == fields.Fields.Count)
        {
            fields = fields with { Fields = fields.Fields.Select((f, k) => f with { X = values[k][0], Y = values[k][1] }).ToValueList() };
        }
        return this with { Fields = fields };
    }

    /// <summary>
    /// Switches the aperture type. With <paramref name="conversion"/>, the new type's value keeps the paraxial
    /// marginal ray; switching to "float by stop size" gives an automatic stop semi-diameter the current beam's
    /// size. Without a converted value, the old number stays (or a default).
    /// </summary>
    public OpticalSystem SetApertureKind(ApertureKind kind, SpecConversion? conversion = null)
    {
        if (kind == Aperture.Kind) return this;
        if (kind == ApertureKind.FloatByStopSize)
        {
            var system = this with { Aperture = new ApertureSpec.FloatByStopSize() };
            if (Surfaces.IsValidIndex(Stop) && Surfaces[Stop].SemiDiameter.FixedValue is null && Surfaces[Stop].Aperture is null
                && conversion?.StopSemiDiameter is { } r && r > 0 && double.IsFinite(r))
            {
                system = system with { Surfaces = Surfaces.With(Stop, s => s with { SemiDiameter = new SemiDiameter.Fixed(r) }) };
            }
            return system;
        }
        if (conversion?.Aperture is { } a && a.Kind == kind) return this with { Aperture = a };
        var v = Aperture.Value ?? (kind == ApertureKind.ObjectSpaceNA ? 0.1 : kind == ApertureKind.EntrancePupilDiameter ? 10 : 4);
        return this with { Aperture = ApertureSpec.Of(kind, v) };
    }
}

/// <summary>
/// The engine's <c>convert_spec</c> result: the system's fields and aperture in another field kind or
/// aperture type, each keeping the system equivalent (docs/RESULTS.md). <c>null</c> members were not asked
/// for or are undefined for the system (e.g. an object height at infinity).
/// </summary>
public sealed record SpecConversion
{
    /// <summary>One <c>(x, y)</c> per system field, in the requested kind.</summary>
    public ValueList<ValueList<double>>? FieldValues { get; init; }

    public ApertureSpec? Aperture { get; init; }

    /// <summary>Paraxial marginal-ray height at the stop (mm).</summary>
    public double? StopSemiDiameter { get; init; }

    /// <summary>Reads the result JSON; <c>null</c> when it is not a <c>convert_spec</c> result.</summary>
    public static SpecConversion? FromJson(JsonNode? json)
    {
        try
        {
            var o = J.Obj(json);
            ValueList<ValueList<double>>? values = null;
            if (J.OptRef(o, "fields", n => J.Req(J.Obj(n), "values", v => J.Arr(v, J.Doubles))) is { } v && v.All(row => row.Count == 2))
                values = v;
            return new SpecConversion
            {
                FieldValues = values,
                Aperture = J.OptRef(o, "aperture", ApertureSpec.FromJson),
                StopSemiDiameter = J.OptVal(o, "stop_semi_diameter", J.D),
            };
        }
        catch (System.Text.Json.JsonException)
        {
            return null;
        }
    }
}
