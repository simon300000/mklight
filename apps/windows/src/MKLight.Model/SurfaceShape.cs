using System.Text.Json.Nodes;
using System.Text.Json.Serialization;

namespace MKLight.Model;

/// <summary>
/// Surface shape in its local frame (CONVENTIONS §2, SURFACES.md). Curvature
/// <c>c = 1/R</c> is stored, never the radius.
/// <para>
/// The JSON form is the engine's serde form: decoding is strict (unknown keys
/// are errors), fields the engine defaults (<c>conic</c>, <c>conic_x</c>, <c>conic_y</c>,
/// <c>x_asphere</c>, <c>y_asphere</c>, <c>ordering</c>) may be absent, and encoding
/// always writes every field.
/// </para>
/// The case records carry the stored values; the computed members of this base
/// (<see cref="Curvature"/>, <see cref="ConicConstant"/>, <see cref="NormRadius"/>, …)
/// follow the engine's accessors across all cases.
/// </summary>
[JsonConverter(typeof(ModelConverter<SurfaceShape>))]
public abstract partial record SurfaceShape : IJsonModel<SurfaceShape>
{
    private protected SurfaceShape() { }

    public static SurfaceShape Plane => new PlaneShape();
    public static SurfaceShape CoordinateBreak => new CoordinateBreakShape();

    public static SurfaceShape FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        var conic = (Func<JsonObject, double>)(x => J.Def(x, "conic", J.D, 0));
        var asphere = (Func<JsonObject, string, ValueList<double>>)((x, k) => J.Def(x, k, J.Doubles, ValueList<double>.Empty));
        switch (J.Tag(o, "type"))
        {
            case "plane":
                J.AllowOnly(o, "type");
                return new PlaneShape();
            case "conic":
                J.AllowOnly(o, "type", "curvature", "conic");
                return new ConicShape(J.Req(o, "curvature", J.D), conic(o));
            case "even_asphere":
                J.AllowOnly(o, "type", "curvature", "conic", "coefficients");
                return new EvenAsphereShape(J.Req(o, "curvature", J.D), conic(o), J.Req(o, "coefficients", J.Doubles));
            case "odd_asphere":
                J.AllowOnly(o, "type", "curvature", "conic", "coefficients");
                return new OddAsphereShape(J.Req(o, "curvature", J.D), conic(o), J.Req(o, "coefficients", J.Doubles));
            case "q_con":
                J.AllowOnly(o, "type", "curvature", "conic", "norm_radius", "coefficients");
                return new QConShape(
                    J.Req(o, "curvature", J.D), conic(o), J.Req(o, "norm_radius", J.D), J.Req(o, "coefficients", J.Doubles));
            case "q_bfs":
                J.AllowOnly(o, "type", "curvature", "norm_radius", "coefficients");
                return new QBfsShape(J.Req(o, "curvature", J.D), J.Req(o, "norm_radius", J.D), J.Req(o, "coefficients", J.Doubles));
            case "diffractive":
                J.AllowOnly(o, "type", "base", "phase");
                return new DiffractiveShape(J.Req(o, "base", FromJson), J.Req(o, "phase", DiffractivePhase.FromJson));
            case "toroid":
                J.AllowOnly(o, "type", "curvature_x", "curvature_y", "conic_y", "y_asphere");
                return new ToroidShape(
                    J.Req(o, "curvature_x", J.D), J.Req(o, "curvature_y", J.D), J.Def(o, "conic_y", J.D, 0), asphere(o, "y_asphere"));
            case "biconic":
                J.AllowOnly(o, "type", "curvature_x", "curvature_y", "conic_x", "conic_y", "x_asphere", "y_asphere");
                return new BiconicShape(
                    J.Req(o, "curvature_x", J.D), J.Req(o, "curvature_y", J.D), J.Def(o, "conic_x", J.D, 0),
                    J.Def(o, "conic_y", J.D, 0), asphere(o, "x_asphere"), asphere(o, "y_asphere"));
            case "xy_polynomial":
                J.AllowOnly(o, "type", "curvature", "conic", "norm_radius", "coefficients");
                return new XyPolynomialShape(
                    J.Req(o, "curvature", J.D), conic(o), J.Req(o, "norm_radius", J.D), J.Req(o, "coefficients", J.Doubles));
            case "zernike_sag":
                J.AllowOnly(o, "type", "curvature", "conic", "norm_radius", "ordering", "coefficients");
                return new ZernikeSagShape(
                    J.Req(o, "curvature", J.D), conic(o), J.Req(o, "norm_radius", J.D),
                    J.Def(o, "ordering", ZernikeOrderingExtensions.FromJson, ZernikeOrdering.Fringe), J.Req(o, "coefficients", J.Doubles));
            case "sag_overlay":
                J.AllowOnly(
                    o, "type", "base", "norm_radius", "test_wavelength_um", "astigmatism", "astigmatism_angle_deg", "spherical");
                return new SagOverlayShape(
                    J.Req(o, "base", FromJson), J.Req(o, "norm_radius", J.D), J.Req(o, "test_wavelength_um", J.D),
                    J.Req(o, "astigmatism", J.D), J.Def(o, "astigmatism_angle_deg", J.D, 0), J.Req(o, "spherical", J.D));
            case "paraxial":
                J.AllowOnly(o, "type", "focal_length");
                return new ParaxialShape(J.Req(o, "focal_length", J.D));
            case "coordinate_break":
                J.AllowOnly(o, "type");
                return new CoordinateBreakShape();
            case var t: throw J.Fail($"unknown surface shape '{t}'");
        }
    }

    public JsonNode ToJson()
    {
        var o = new JsonObject { ["type"] = TypeName };
        switch (this)
        {
            case ConicShape s:
                o["curvature"] = J.Num(s.VertexCurvature);
                o["conic"] = J.Num(s.Conic);
                break;
            case EvenAsphereShape s:
                o["curvature"] = J.Num(s.VertexCurvature);
                o["conic"] = J.Num(s.Conic);
                o["coefficients"] = J.Doubles(s.Coefficients);
                break;
            case OddAsphereShape s:
                o["curvature"] = J.Num(s.VertexCurvature);
                o["conic"] = J.Num(s.Conic);
                o["coefficients"] = J.Doubles(s.Coefficients);
                break;
            case QConShape s:
                o["curvature"] = J.Num(s.VertexCurvature);
                o["conic"] = J.Num(s.Conic);
                o["norm_radius"] = J.Num(s.NormRadiusMm);
                o["coefficients"] = J.Doubles(s.Coefficients);
                break;
            case XyPolynomialShape s:
                o["curvature"] = J.Num(s.VertexCurvature);
                o["conic"] = J.Num(s.Conic);
                o["norm_radius"] = J.Num(s.NormRadiusMm);
                o["coefficients"] = J.Doubles(s.Coefficients);
                break;
            case QBfsShape s:
                o["curvature"] = J.Num(s.VertexCurvature);
                o["norm_radius"] = J.Num(s.NormRadiusMm);
                o["coefficients"] = J.Doubles(s.Coefficients);
                break;
            case DiffractiveShape s:
                o["base"] = s.Base.ToJson();
                o["phase"] = s.Phase.ToJson();
                break;
            case ToroidShape s:
                o["curvature_x"] = J.Num(s.CurvatureX);
                o["curvature_y"] = J.Num(s.CurvatureY);
                o["conic_y"] = J.Num(s.ConicY);
                o["y_asphere"] = J.Doubles(s.YAsphere);
                break;
            case BiconicShape s:
                o["curvature_x"] = J.Num(s.CurvatureX);
                o["curvature_y"] = J.Num(s.CurvatureY);
                o["conic_x"] = J.Num(s.ConicX);
                o["conic_y"] = J.Num(s.ConicY);
                o["x_asphere"] = J.Doubles(s.XAsphere);
                o["y_asphere"] = J.Doubles(s.YAsphere);
                break;
            case ZernikeSagShape s:
                o["curvature"] = J.Num(s.VertexCurvature);
                o["conic"] = J.Num(s.Conic);
                o["norm_radius"] = J.Num(s.NormRadiusMm);
                o["ordering"] = Wire.Write(s.Ordering);
                o["coefficients"] = J.Doubles(s.Coefficients);
                break;
            case SagOverlayShape s:
                o["base"] = s.Base.ToJson();
                o["norm_radius"] = J.Num(s.NormRadiusMm);
                o["test_wavelength_um"] = J.Num(s.TestWavelengthUm);
                o["astigmatism"] = J.Num(s.Astigmatism);
                o["astigmatism_angle_deg"] = J.Num(s.AstigmatismAngleDeg);
                o["spherical"] = J.Num(s.Spherical);
                break;
            case ParaxialShape s:
                o["focal_length"] = J.Num(s.FocalLength);
                break;
        }
        return o;
    }

    /// <summary>The JSON <c>type</c> tag (the engine's <c>SurfaceShape::type_name</c>).</summary>
    public string TypeName => this is SagOverlayShape ? "sag_overlay" : Kind.TypeName();

    /// <summary>
    /// Whether the shape is one of the engine's basic shapes (plane, conic,
    /// even asphere, paraxial, coordinate break; the engine's <c>SurfaceShape::is_legacy</c>).
    /// The others are the extended shapes.
    /// </summary>
    public bool IsLegacy => this is not SagOverlayShape && Kind.IsLegacy();

    /// <summary>The shape type without its parameters; a sag overlay has its base's kind.</summary>
    public ShapeKind Kind => this switch
    {
        PlaneShape => ShapeKind.Plane,
        ConicShape => ShapeKind.Conic,
        EvenAsphereShape => ShapeKind.EvenAsphere,
        OddAsphereShape => ShapeKind.OddAsphere,
        QConShape => ShapeKind.QCon,
        QBfsShape => ShapeKind.QBfs,
        DiffractiveShape => ShapeKind.Diffractive,
        ToroidShape => ShapeKind.Toroid,
        BiconicShape => ShapeKind.Biconic,
        XyPolynomialShape => ShapeKind.XyPolynomial,
        ZernikeSagShape => ShapeKind.ZernikeSag,
        SagOverlayShape o => o.Base.Kind,
        ParaxialShape => ShapeKind.Paraxial,
        _ => ShapeKind.CoordinateBreak,
    };

    // MARK: Cylinder preset

    /// <summary>
    /// A cylinder: the biconic preset with curvature <paramref name="curvature"/> along
    /// <paramref name="axis"/> and a flat other axis (SURFACES §4). <c>X</c> matches the
    /// engine sample's <c>cylinder(curvature_x)</c>.
    /// </summary>
    public static SurfaceShape Cylinder(double curvature, CylinderAxis axis) =>
        new BiconicShape(axis == CylinderAxis.X ? curvature : 0, axis == CylinderAxis.Y ? curvature : 0, 0, 0, [], []);

    /// <summary>
    /// The power axis when the shape is a cylinder: a biconic with one curved
    /// axis and the other axis flat (zero curvature, zero polynomial). The
    /// curved axis may carry a conic and polynomial. <c>null</c> otherwise.
    /// </summary>
    public CylinderAxis? PowerAxis
    {
        get
        {
            if (this is not BiconicShape b) return null;
            var flatX = b.CurvatureX == 0 && b.XAsphere.All(v => v == 0);
            var flatY = b.CurvatureY == 0 && b.YAsphere.All(v => v == 0);
            if (flatY && b.CurvatureX != 0) return CylinderAxis.X;
            if (flatX && b.CurvatureY != 0) return CylinderAxis.Y;
            return null;
        }
    }

    /// <summary>Whether the shape is the cylinder preset of a biconic.</summary>
    public bool IsCylinder => PowerAxis is not null;
}

/// <summary>The axis along which a cylinder has power.</summary>
public enum CylinderAxis
{
    /// <summary>Power in x (curved in the x-z plane, straight along y).</summary>
    X,

    /// <summary>Power in y (curved in the y-z plane, straight along x).</summary>
    Y,
}

/// <summary>Flat surface <c>z = 0</c>.</summary>
public sealed record PlaneShape : SurfaceShape;

/// <summary>Sphere or conic: <c>z = c r² / (1 + √(1 − (1+k) c² r²))</c>.</summary>
public sealed record ConicShape(double VertexCurvature, double Conic) : SurfaceShape;

/// <summary>Conic plus <c>Σ a_i r^(2i+4)</c>; <c>Coefficients[0]</c> is A4.</summary>
public sealed record EvenAsphereShape(double VertexCurvature, double Conic, ValueList<double> Coefficients) : SurfaceShape;

/// <summary>Conic plus <c>Σ β_i r^i</c> from i = 1; <c>Coefficients[0]</c> is β1, in mm^(1−i).</summary>
public sealed record OddAsphereShape(double VertexCurvature, double Conic, ValueList<double> Coefficients) : SurfaceShape;

/// <summary>Forbes conic-based Q departure; coefficients a₀, a₁, … in mm (SURFACES §2).</summary>
public sealed record QConShape(double VertexCurvature, double Conic, double NormRadiusMm, ValueList<double> Coefficients) : SurfaceShape;

/// <summary>Forbes best-fit-sphere Q departure; the curvature is that of the sphere.</summary>
public sealed record QBfsShape(double VertexCurvature, double NormRadiusMm, ValueList<double> Coefficients) : SurfaceShape;

/// <summary>
/// Fixed-period phase structure on a plane, conic or even-asphere base
/// (SURFACES §3). Curvature, conic and asphere parameters address the base.
/// </summary>
public sealed record DiffractiveShape(SurfaceShape Base, DiffractivePhase Phase) : SurfaceShape;

/// <summary>
/// The y profile (conic plus <c>YAsphere</c> from y⁴) rotated about a line
/// parallel to y at the x radius (SURFACES §4).
/// </summary>
public sealed record ToroidShape(double CurvatureX, double CurvatureY, double ConicY, ValueList<double> YAsphere) : SurfaceShape;

/// <summary>
/// Independent x and y conics plus axis polynomials from x⁴ and y⁴ (SURFACES §4).
/// A cylinder is a biconic preset: see <see cref="SurfaceShape.Cylinder"/>.
/// </summary>
public sealed record BiconicShape(
    double CurvatureX, double CurvatureY, double ConicX, double ConicY, ValueList<double> XAsphere, ValueList<double> YAsphere)
    : SurfaceShape;

/// <summary>
/// Conic plus <c>Σ C_j X^m Y^n</c> in normalised coordinates, coefficients in mm;
/// the term order is <see cref="XYTerm"/> (SURFACES §5).
/// </summary>
public sealed record XyPolynomialShape(double VertexCurvature, double Conic, double NormRadiusMm, ValueList<double> Coefficients)
    : SurfaceShape;

/// <summary>
/// Conic plus Zernike sag terms in mm, from Z1 (piston); the term order is
/// <see cref="ZernikeTerm"/> (SURFACES §6).
/// </summary>
public sealed record ZernikeSagShape(
    double VertexCurvature, double Conic, double NormRadiusMm, ZernikeOrdering Ordering, ValueList<double> Coefficients)
    : SurfaceShape;

/// <summary>
/// Fringe irregularity on a geometric base (SURFACES §10): <c>Astigmatism</c> and
/// <c>Spherical</c> in test fringes (λ_test/2 PV sag each) over <c>NormRadiusMm</c> (mm),
/// <c>TestWavelengthUm</c> in µm; the astigmatism axis is <c>AstigmatismAngleDeg</c>
/// from +x toward +y. Its kind is the base's.
/// </summary>
public sealed record SagOverlayShape(
    SurfaceShape Base, double NormRadiusMm, double TestWavelengthUm, double Astigmatism, double AstigmatismAngleDeg, double Spherical)
    : SurfaceShape;

/// <summary>Ideal thin lens of the given focal length (mm).</summary>
public sealed record ParaxialShape(double FocalLength) : SurfaceShape;

/// <summary>Changes the frame only (uses the surface's <c>Placement</c>).</summary>
public sealed record CoordinateBreakShape : SurfaceShape;

/// <summary>The shape types, without their parameters.</summary>
public enum ShapeKind
{
    [Wire("plane")] Plane,
    [Wire("conic")] Conic,
    [Wire("even_asphere")] EvenAsphere,
    [Wire("odd_asphere")] OddAsphere,
    [Wire("q_con")] QCon,
    [Wire("q_bfs")] QBfs,
    [Wire("diffractive")] Diffractive,
    [Wire("toroid")] Toroid,
    [Wire("biconic")] Biconic,
    [Wire("xy_polynomial")] XyPolynomial,
    [Wire("zernike_sag")] ZernikeSag,
    [Wire("paraxial")] Paraxial,
    [Wire("coordinate_break")] CoordinateBreak,
}

public static class ShapeKindExtensions
{
    public static string Title(this ShapeKind k) => k switch
    {
        ShapeKind.Plane => "Plane",
        ShapeKind.Conic => "Standard",
        ShapeKind.EvenAsphere => "Even Asphere",
        ShapeKind.OddAsphere => "Odd Asphere",
        ShapeKind.QCon => "Q-Type (Conic)",
        ShapeKind.QBfs => "Q-Type (Best-Fit Sphere)",
        ShapeKind.Diffractive => "Diffractive",
        ShapeKind.Toroid => "Toroid",
        ShapeKind.Biconic => "Biconic",
        ShapeKind.XyPolynomial => "XY Polynomial",
        ShapeKind.ZernikeSag => "Zernike Sag",
        ShapeKind.Paraxial => "Paraxial",
        _ => "Coord Break",
    };

    /// <summary>The JSON <c>type</c> tag.</summary>
    public static string TypeName(this ShapeKind k) => Wire.Name(k);

    /// <summary>The engine's original solver subset (<c>SurfaceShape::is_legacy</c>).</summary>
    public static bool IsLegacy(this ShapeKind k) =>
        k is ShapeKind.Plane or ShapeKind.Conic or ShapeKind.EvenAsphere or ShapeKind.Paraxial or ShapeKind.CoordinateBreak;
}

/// <summary>
/// Phase lag of a diffractive surface at order +1, <c>Φ = Σ A_i ρ^(2i)</c> in
/// radians with <c>ρ = r / NormRadius</c> (SURFACES §3). The design order and
/// wavelength are metadata; they do not rescale the traced phase.
/// </summary>
[JsonConverter(typeof(ModelConverter<DiffractivePhase>))]
public sealed record DiffractivePhase : IJsonModel<DiffractivePhase>
{
    /// <summary>Normalisation radius (mm), &gt; 0.</summary>
    public required double NormRadius { get; init; }

    /// <summary>A₁, A₂, … (radians), at most <c>CoefficientSet.MaxPhaseTerms</c> (10).</summary>
    public ValueList<double> Coefficients { get; init; } = ValueList<double>.Empty;

    /// <summary>Nonzero signed design order (an <c>i32</c> in the engine).</summary>
    public int DesignOrder { get; init; } = 1;

    /// <summary>Design wavelength λ₀ (µm), &gt; 0.</summary>
    public required double DesignWavelengthUm { get; init; }

    /// <summary>Signed order that is traced; 0 is ordinary refraction or reflection (an <c>i32</c> in the engine).</summary>
    public int TraceOrder { get; init; } = 1;

    public static DiffractivePhase FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        J.AllowOnly(o, "norm_radius", "coefficients", "design_order", "design_wavelength_um", "trace_order");
        // Orders are i32 in the engine: out-of-range values fail to decode.
        return new DiffractivePhase
        {
            NormRadius = J.Req(o, "norm_radius", J.D),
            Coefficients = J.Req(o, "coefficients", J.Doubles),
            DesignOrder = J.Req(o, "design_order", J.I32),
            DesignWavelengthUm = J.Req(o, "design_wavelength_um", J.D),
            TraceOrder = J.Req(o, "trace_order", J.I32),
        };
    }

    public JsonNode ToJson() => new JsonObject
    {
        ["norm_radius"] = J.Num(NormRadius),
        ["coefficients"] = J.Doubles(Coefficients),
        ["design_order"] = DesignOrder,
        ["design_wavelength_um"] = J.Num(DesignWavelengthUm),
        ["trace_order"] = TraceOrder,
    };
}
