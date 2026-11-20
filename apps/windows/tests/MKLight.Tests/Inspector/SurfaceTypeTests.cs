using System.Text.Json.Nodes;
using MKLight.App.ViewModels;
using MKLight.Model;

namespace MKLight.Tests.Inspector;

/// <summary>The data-driven forms of the surface types, and the display sags of their previews against the closed forms in docs/SURFACES.md.</summary>
public class SurfaceTypeTests
{
    public static TheoryData<ShapeKind> Kinds => [.. Enum.GetValues<ShapeKind>()];

    static (OpticalSystem System, Surface Surface) With(ShapeKind kind)
    {
        var system = SampleSystems.Singlet().SetShapeKind(kind, 1);
        if (kind == ShapeKind.Diffractive && system.Surfaces[1].Shape is DiffractiveShape d)
            system = system.WithSurface(1, s => s with { Shape = d with { Base = new ConicShape(0.01, 0) } }); // a conic base, so its parameters show too
        return (system, system.Surfaces[1]);
    }

    /// <summary>
    /// Every parameter and coefficient list of every type addresses a key the model has, writes back through the
    /// model's strict decoder, and its variable path names the value it edits.
    /// </summary>
    [Theory, MemberData(nameof(Kinds))]
    public void DescriptorsAddressTheModel(ShapeKind kind)
    {
        var (system, surface) = With(kind);
        var shape = surface.Shape;
        var type = SurfaceTypes.Descriptor(shape);
        Assert.Equal(kind, type.Kind);
        Assert.Equal(kind.TypeName(), type.Id);
        foreach (var section in type.Sections.Where(s => s.Applies(shape)))
        {
            foreach (var p in section.Params)
            {
                if (SurfaceTypes.Value(shape, p.Key) is not { } v) continue;
                Assert.Equal(shape, SurfaceTypes.WithValue(shape, p.Key, v.DeepClone()));
                if (p.Path?.Invoke(surface.Id) is { } path)
                {
                    Assert.True(path.Resolves(system), $"{kind} {p.Key}: {path}");
                    Assert.Equal(v.GetValue<double>(), system.ValueAt(path));
                }
            }
            foreach (var c in section.Choices) Assert.NotNull(SurfaceTypes.Value(shape, c.Key)?.GetValue<string>());
            if (section.Coefficients is { } list && SurfaceTypes.HasList(list, shape))
            {
                var edited = SurfaceTypes.WithCoefficients(shape, list, [1e-6, -2e-7]);
                Assert.NotNull(edited);
                Assert.Equal([1e-6, -2e-7], SurfaceTypes.Coefficients(list, edited!));
                Assert.True(list.MaxCount(edited!) >= 2);
                if (list.Path?.Invoke(surface.Id, 1) is { } path)
                    Assert.Equal(-2e-7, system.WithSurface(1, s => s with { Shape = edited! }).ValueAt(path));
            }
        }
    }

    [Fact]
    public void UnknownKeysAndWrongTypesAreRefused()
    {
        var conic = new ConicShape(0.02, 0);
        Assert.Null(SurfaceTypes.WithValue(conic, "norm_radius", JsonValue.Create(5.0)));
        Assert.Null(SurfaceTypes.WithValue(conic, "curvature", JsonValue.Create("x")));
        Assert.Null(SurfaceTypes.Value(conic, "phase.norm_radius"));
    }

    [Fact]
    public void RadiiAndOrdersReadAndParseAsShown()
    {
        var radius = SurfaceTypes.Descriptor(ShapeKind.Conic).Sections[0].Params[0];
        Assert.Equal(50, SurfaceTypes.DisplayValue(radius, new ConicShape(0.02, 0))!.Value, 12);
        Assert.Equal(double.PositiveInfinity, SurfaceTypes.DisplayValue(radius, new ConicShape(0, 0)));
        Assert.Equal(0.02, SurfaceTypes.StoredValue(radius, "50").Value, 15);
        Assert.Equal(0, SurfaceTypes.StoredValue(radius, "∞").Value);
        var order = SurfaceTypes.Descriptor(ShapeKind.Diffractive).Sections.Single(s => s.Title == "Orders").Params[0];
        Assert.True(SurfaceTypes.StoredValue(order, "2").Ok);
        Assert.False(SurfaceTypes.StoredValue(order, "1.5").Ok);
        var (_, surface) = With(ShapeKind.Diffractive);
        Assert.Equal(1, SurfaceTypes.DisplayValue(order, surface.Shape));
        var two = SurfaceTypes.WithParam(surface.Shape, order, 2);
        Assert.Equal(2, ((DiffractiveShape)two!).Phase.DesignOrder);
    }

    [Fact]
    public void TermLabelsCarryTheirMeaning()
    {
        var shape = new XyPolynomialShape(0, 0, 5, []);
        var xy = SurfaceTypes.Descriptor(ShapeKind.XyPolynomial).Sections[1].Coefficients!;
        Assert.Equal("C(x²y)", xy.Term(6, shape));
        Assert.Equal("C(x)", xy.Term(0, shape));
        var zernike = SurfaceTypes.Descriptor(ShapeKind.ZernikeSag).Sections[1].Coefficients!;
        var fringe = new ZernikeSagShape(0, 0, 5, ZernikeOrdering.Fringe, []);
        Assert.Equal("Z₇", zernike.Term(6, fringe));
        Assert.Equal("coma x", zernike.Meaning(6, fringe));
        Assert.Equal("spherical", zernike.Meaning(8, fringe));
        Assert.Equal("defocus", zernike.Meaning(3, fringe));
        var standard = fringe with { Ordering = ZernikeOrdering.Standard };
        Assert.Equal("spherical", zernike.Meaning(10, standard)); // Noll Z11
        var q = SurfaceTypes.Descriptor(ShapeKind.QCon).Sections[1].Coefficients!;
        Assert.Equal("a₂", q.Term(2, shape));
        Assert.Equal("u⁴Q₂ᶜᵒⁿ", q.Meaning(2, shape));
        var odd = SurfaceTypes.Descriptor(ShapeKind.OddAsphere).Sections[1].Coefficients!;
        Assert.Equal("β₃", odd.Term(2, shape));
        Assert.Equal("r³", odd.Meaning(2, shape));
        Assert.Equal("mm⁻²", odd.Unit(2));
        var even = SurfaceTypes.Descriptor(ShapeKind.EvenAsphere).Sections[1].Coefficients!;
        Assert.Equal("A₄", even.Term(0, shape));
        Assert.Equal("r⁴", even.Meaning(0, shape));
        Assert.Equal("mm⁻³", even.Unit(0));
    }

    [Fact]
    public void EveryKindHasOneDescriptor()
    {
        Assert.Equal(Enum.GetValues<ShapeKind>().ToHashSet(), SurfaceTypes.All.Select(d => d.Kind).ToHashSet());
        Assert.Equal(SurfaceTypes.All.Count, SurfaceTypes.All.Select(d => d.Kind).Distinct().Count());
    }

    // ---- Coefficient list edits ----

    [Fact]
    public void RemovingTheLastTermDropsItAndAnotherBecomesZero()
    {
        Assert.Equal([1.0, 2], SurfaceTypes.RemoveTerm([1, 2, 3], 2, sparse: false));
        Assert.Equal([1.0, 0, 3], SurfaceTypes.RemoveTerm([1, 2, 3], 1, sparse: false));
        // A sparse list also drops the zero terms before its last, unless they were just added.
        Assert.Equal([1.0], SurfaceTypes.RemoveTerm([1, 0, 0, 3], 3, sparse: true));
        Assert.Equal([1.0, 0, 0], SurfaceTypes.RemoveTerm([1, 0, 0, 3], 3, sparse: true, new HashSet<int> { 2 }));
        Assert.Equal([0.0, 0, 5], SurfaceTypes.AddTerm([], 2).Select((v, k) => k == 2 ? 5 : v));
        Assert.Equal([1.0, 7], SurfaceTypes.SetTerm([1], 1, 7));
    }

    [Fact]
    public void SparseListsShowTheirNonzeroRevealedAndLastTerms()
    {
        double[] v = [0, 1, 0, 0, 2, 0];
        Assert.Equal([1, 4, 5], SurfaceTypes.VisibleTerms(v, sparse: true, showAll: false, new HashSet<int>()));
        Assert.Equal([1, 2, 4, 5], SurfaceTypes.VisibleTerms(v, sparse: true, showAll: false, new HashSet<int> { 2 }));
        Assert.Equal(6, SurfaceTypes.VisibleTerms(v, sparse: true, showAll: true, new HashSet<int>()).Count);
        Assert.Equal(6, SurfaceTypes.VisibleTerms(v, sparse: false, showAll: false, new HashSet<int>()).Count);
    }

    // ---- Display sags ----

    [Fact]
    public void ForbesPolynomialsMatchTheirClosedForms()
    {
        for (var t = 0.0; t <= 1.0; t += 0.125)
        {
            Assert.Equal(6 * t - 5, ShapeSag.QCon([0, 1], t), 12);
            Assert.Equal(28 * t * t - 42 * t + 15, ShapeSag.QCon([0, 0, 1], t), 12);
            Assert.Equal(1, ShapeSag.QBfs([1], t), 12);
            Assert.Equal((13 - 16 * t) / Math.Sqrt(19), ShapeSag.QBfs([0, 1], t), 12);
            Assert.Equal((29 - 100 * t + 76 * t * t) / (Math.Sqrt(190) / 2), ShapeSag.QBfs([0, 0, 1], t), 12);
        }
    }

    [Fact]
    public void ExtendedShapesReduceToTheirBases()
    {
        double c = 1 / 60.0, k = -0.4;
        foreach (var (x, y) in new[] { (0.0, 0.0), (1.5, 0), (0, 2), (1, -2.5), (-3, 1) })
        {
            var conic = ShapeSag.Conic(c, k, x * x + y * y)!.Value;
            var sphere = ShapeSag.Conic(c, 0, x * x + y * y)!.Value;
            Assert.Equal(conic, ShapeSag.Sag(new QConShape(c, k, 5, [0, 0]), x, y)!.Value, 14);
            Assert.Equal(sphere, ShapeSag.Sag(new QBfsShape(c, 5, [0, 0]), x, y)!.Value, 14);
            Assert.Equal(sphere, ShapeSag.Sag(new ToroidShape(c, c, 0, []), x, y)!.Value, 12);
            Assert.Equal(conic, ShapeSag.Sag(new BiconicShape(c, c, k, k, [], []), x, y)!.Value, 14);
            Assert.Equal(conic, ShapeSag.Sag(new XyPolynomialShape(c, k, 5, []), x, y)!.Value, 14);
            Assert.Equal(conic, ShapeSag.Sag(new DiffractiveShape(new ConicShape(c, k), new DiffractivePhase { NormRadius = 5, DesignWavelengthUm = 0.55 }), x, y)!.Value, 14);
        }
        // A cylinder has no sag along its plane axis.
        Assert.Equal(0, ShapeSag.Sag(SurfaceShape.Cylinder(0.02, CylinderAxis.X), 0, 3));
        Assert.True(ShapeSag.Sag(SurfaceShape.Cylinder(0.02, CylinderAxis.X), 3, 0) > 0);
        // Normalised polynomials end at their radius.
        Assert.Null(ShapeSag.Sag(new QConShape(c, 0, 5, [1e-3]), 0, 5.1));
        // An even asphere adds A4 r⁴.
        Assert.Equal(ShapeSag.Conic(c, 0, 4)!.Value + 1e-5 * 16, ShapeSag.Sag(new EvenAsphereShape(c, 0, [1e-5]), 0, 2)!.Value, 14);
    }

    [Fact]
    public void PolynomialTermsFollowTheEngineOrder()
    {
        // XY: index 3 is XY, index 6 is X²Y.
        var xy = new XyPolynomialShape(0, 0, 2, [0, 0, 0, 1e-3, 0, 0, 2e-3]);
        double x = 0.6, y = -1.4, X = x / 2, Y = y / 2;
        Assert.Equal(1e-3 * X * Y + 2e-3 * X * X * Y, ShapeSag.Sag(xy, x, y)!.Value, 15);
        // Fringe Z5 = ρ² cos 2θ (unnormalised); Standard Z4 = √3 (2ρ² − 1); Fringe Z9 = 6ρ⁴ − 6ρ² + 1.
        double rho = 0.7, theta = 0.4;
        Assert.Equal(rho * rho * Math.Cos(2 * theta), ShapeSag.Zernike(ZernikeTerm.FromJ(5, ZernikeOrdering.Fringe)!.Value, ZernikeOrdering.Fringe, rho, theta), 14);
        Assert.Equal(Math.Sqrt(3) * (2 * rho * rho - 1), ShapeSag.Zernike(ZernikeTerm.FromJ(4, ZernikeOrdering.Standard)!.Value, ZernikeOrdering.Standard, rho, theta), 14);
        Assert.Equal(6 * Math.Pow(rho, 4) - 6 * rho * rho + 1, ShapeSag.Zernike(ZernikeTerm.FromJ(9, ZernikeOrdering.Fringe)!.Value, ZernikeOrdering.Fringe, rho, theta), 14);
    }

    /// <summary>φ = −m λ A₁ / (π r_norm²); a quadratic phase of focal length f has the local period λ f / r.</summary>
    [Fact]
    public void DiffractivePowerAndZonePeriod()
    {
        double f = 100, lambda = 0.55, rn = 5;
        var a1 = -Math.PI * rn * rn / (lambda * 1e-3 * f);
        var phase = new DiffractivePhase { NormRadius = rn, Coefficients = [a1], DesignOrder = 1, DesignWavelengthUm = lambda, TraceOrder = 1 };
        Assert.Equal(1 / f, ShapeSag.PhasePower(phase, 1, lambda), 12);
        Assert.Equal(2 / f, ShapeSag.PhasePower(phase, 2, lambda), 12);
        var period = ShapeSag.MinimumPeriod(phase, rn)!.Value;
        Assert.Equal(lambda * 1e-3 * f / rn, period.Period, 9);
        Assert.Equal(rn, period.Radius, 9);
    }

    [Fact]
    public void DepartureMapIsTheFreeformPart()
    {
        var shape = new ZernikeSagShape(0.02, 0, 4, ZernikeOrdering.Fringe, [0, 0, 0, 0, 1e-3]);
        var map = SurfaceTypes.DepartureMap(shape, 4, 20);
        var values = map.Where(double.IsFinite).ToList();
        Assert.Equal(400, map.Length);
        Assert.True(values.Count < 400, "corners outside the disk are empty");
        Assert.True(values.Max() <= 1e-3 + 1e-12 && values.Min() >= -1e-3 - 1e-12);
        Assert.True(values.Max() > 0.8e-3, "astigmatism reaches its amplitude near the edge");
    }

    [Fact]
    public void ProfilesCoverTheSemiDiameterAndSkipTheUndefined()
    {
        var p = SurfaceTypes.Profile(new ConicShape(0.02, 0), 10, samples: 10)!;
        Assert.Equal(11, p.Count);
        Assert.Equal(-10, p[0].T);
        Assert.Equal(10, p[^1].T);
        // A sphere of radius 5 has no real sag beyond ±5.
        Assert.True(SurfaceTypes.Profile(new ConicShape(0.2, 0), 10, samples: 20)!.All(q => Math.Abs(q.T) <= 5));
        Assert.Equal(7.5, SurfaceTypes.PreviewRadius(new ConicShape(0, 0), 7.5));
        Assert.Equal(5, SurfaceTypes.PreviewRadius(new XyPolynomialShape(0, 0, 5, []), null));
        Assert.Equal(10, SurfaceTypes.PreviewRadius(new ConicShape(0, 0), null));
    }
}
