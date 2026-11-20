namespace MKLight.Model;

public abstract partial record SurfaceShape
{
    /// <summary>The shape carrying the sag: a diffractive surface's base, else itself (the engine's <c>sag_shape</c>).</summary>
    public SurfaceShape SagShape => this is DiffractiveShape d ? d.Base : this;

    /// <summary>
    /// The geometric surface without a diffractive phase or sag overlay (the
    /// engine's <c>geometry</c>). Curvature, conic and asphere paths address it.
    /// </summary>
    public SurfaceShape Geometry => this switch
    {
        DiffractiveShape d => d.Base.Geometry,
        SagOverlayShape o => o.Base.Geometry,
        _ => this,
    };

    /// <summary>
    /// Vertex curvature as the engine's <c>SurfaceShape::curvature()</c>: the y curvature of a
    /// toroid or biconic, the base's for a diffractive surface, 0 for shapes without one.
    /// </summary>
    public double Curvature => this switch
    {
        ConicShape s => s.VertexCurvature,
        EvenAsphereShape s => s.VertexCurvature,
        OddAsphereShape s => s.VertexCurvature,
        QConShape s => s.VertexCurvature,
        QBfsShape s => s.VertexCurvature,
        XyPolynomialShape s => s.VertexCurvature,
        ZernikeSagShape s => s.VertexCurvature,
        DiffractiveShape s => s.Base.Curvature,
        SagOverlayShape s => s.Base.Curvature,
        ToroidShape s => s.CurvatureY,
        BiconicShape s => s.CurvatureY,
        _ => 0,
    };

    /// <summary>
    /// Conic constant as the engine's <c>SurfaceShape::conic()</c>: the y conic of a toroid or
    /// biconic, the base's for a diffractive surface, 0 for shapes without one (including <c>QBfs</c>).
    /// </summary>
    public double ConicConstant => this switch
    {
        ConicShape s => s.Conic,
        EvenAsphereShape s => s.Conic,
        OddAsphereShape s => s.Conic,
        QConShape s => s.Conic,
        XyPolynomialShape s => s.Conic,
        ZernikeSagShape s => s.Conic,
        DiffractiveShape s => s.Base.ConicConstant,
        SagOverlayShape s => s.Base.ConicConstant,
        ToroidShape s => s.ConicY,
        BiconicShape s => s.ConicY,
        _ => 0,
    };

    /// <summary>Even-asphere coefficients of the geometry (A4, A6, …), as the <c>asphere</c> path reads them; empty otherwise.</summary>
    public ValueList<double> AsphereCoefficients => Geometry is EvenAsphereShape e ? e.Coefficients : ValueList<double>.Empty;

    /// <summary>The normalisation radius of a Q, XY, Zernike or diffractive shape.</summary>
    public double? NormRadius => ShapeParameterValue(ShapeParameter.NormRadius);

    /// <summary>
    /// Whether the <c>curvature</c> path resolves (the engine's <c>Curvature</c> get): the geometry is
    /// a plane or has a <c>curvature</c> field, which on a toroid or biconic is the y section.
    /// Paraxial and coordinate break have none.
    /// </summary>
    public bool HasCurvature => Geometry is PlaneShape or ConicShape or EvenAsphereShape or OddAsphereShape or QConShape
        or QBfsShape or XyPolynomialShape or ZernikeSagShape or ToroidShape or BiconicShape;

    /// <summary>Whether the <c>conic</c> path resolves (the engine's <c>Conic</c> get). A plane and <c>QBfs</c> have no conic.</summary>
    public bool HasConic => Geometry is ConicShape or EvenAsphereShape or OddAsphereShape or QConShape or XyPolynomialShape or ZernikeSagShape;

    /// <summary>Whether the <c>asphere</c> path resolves: the geometry is an even asphere.</summary>
    public bool HasAsphereCoefficients => Geometry is EvenAsphereShape;

    /// <summary>
    /// Sets fringe irregularity <paramref name="mode"/>, wrapping the sag shape in a zero overlay over
    /// <paramref name="clearRadius"/> first, like the engine's <c>ParamPath::Irregularity</c> setter.
    /// <c>null</c> for a paraxial lens or coordinate break, or when a new overlay has no clear radius.
    /// </summary>
    public SurfaceShape? WithIrregularity(IrregularityMode mode, double v, double? clearRadius, double testWavelengthUm)
    {
        switch (this)
        {
            case DiffractiveShape d:
                return d.Base.WithIrregularity(mode, v, clearRadius, testWavelengthUm) is { } b ? d with { Base = b } : null;
            case ParaxialShape or CoordinateBreakShape:
                return null;
            case SagOverlayShape o:
                return o with
                {
                    Astigmatism = mode == IrregularityMode.Astigmatism ? v : o.Astigmatism,
                    Spherical = mode == IrregularityMode.Spherical ? v : o.Spherical,
                };
            default:
                if (clearRadius is not { } r) return null;
                return new SagOverlayShape(
                    this, r, testWavelengthUm, mode == IrregularityMode.Astigmatism ? v : 0, 0, mode == IrregularityMode.Spherical ? v : 0);
        }
    }

    /// <summary>Applies <paramref name="edit"/> to the geometry (inside a diffractive phase or overlay).</summary>
    private SurfaceShape? EditGeometry(Func<SurfaceShape, SurfaceShape?> edit) => this switch
    {
        DiffractiveShape d => d.Base.EditGeometry(edit) is { } b ? d with { Base = b } : null,
        SagOverlayShape o => o.Base.EditGeometry(edit) is { } b ? o with { Base = b } : null,
        _ => edit(this),
    };

    /// <summary>
    /// The shape with the curvature of the geometry set, promoting a plane to a sphere, exactly like
    /// the engine's <c>ParamPath::Curvature</c> setter (the y section of a toroid or biconic).
    /// <c>null</c> for shapes without one.
    /// </summary>
    public SurfaceShape? WithCurvature(double c) => EditGeometry(s => s switch
    {
        PlaneShape => new ConicShape(c, 0),
        ConicShape x => x with { VertexCurvature = c },
        EvenAsphereShape x => x with { VertexCurvature = c },
        OddAsphereShape x => x with { VertexCurvature = c },
        QConShape x => x with { VertexCurvature = c },
        QBfsShape x => x with { VertexCurvature = c },
        XyPolynomialShape x => x with { VertexCurvature = c },
        ZernikeSagShape x => x with { VertexCurvature = c },
        ToroidShape x => x with { CurvatureY = c },
        BiconicShape x => x with { CurvatureY = c },
        _ => null,
    });

    /// <summary>
    /// The shape with the conic constant of the geometry set, as the engine's <c>ParamPath::Conic</c>
    /// setter does for shapes that have one. As an editor convenience it also promotes a plane (or a
    /// diffractive plane base) to a conic of zero curvature, which the <c>conic</c> path does not:
    /// <see cref="OpticalSystem.SetValue"/> refuses that, like the engine.
    /// </summary>
    public SurfaceShape? WithConic(double k) => EditGeometry(s => s switch
    {
        PlaneShape => new ConicShape(0, k),
        ConicShape x => x with { Conic = k },
        EvenAsphereShape x => x with { Conic = k },
        OddAsphereShape x => x with { Conic = k },
        QConShape x => x with { Conic = k },
        XyPolynomialShape x => x with { Conic = k },
        ZernikeSagShape x => x with { Conic = k },
        _ => null,
    });

    /// <summary>
    /// The shape with even-asphere coefficient <paramref name="index"/> of the geometry set, extending the
    /// list with zeros; an index at or beyond <see cref="CoefficientLimits.MaxEvenTerms"/> is refused
    /// (the engine's <c>ParamPath::Asphere</c> setter), as is a shape that is no even asphere (<c>null</c>).
    /// </summary>
    public SurfaceShape? WithAsphereCoefficient(double v, int index)
    {
        if (index < 0 || index >= CoefficientLimits.MaxEvenTerms) return null;
        return EditGeometry(s => s is EvenAsphereShape e ? e with { Coefficients = SetPadded(e.Coefficients, index, v) } : null);
    }

    /// <summary>The list with element <paramref name="index"/> set, extending it with zeros.</summary>
    internal static ValueList<double> SetPadded(ValueList<double> a, int index, double v)
    {
        var list = a.ToList();
        while (list.Count <= index) list.Add(0);
        list[index] = v;
        return list.ToValueList();
    }
}
