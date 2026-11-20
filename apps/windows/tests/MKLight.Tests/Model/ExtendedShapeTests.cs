using System.Text.Json;
using MKLight.Model;

namespace MKLight.Tests.Model;

/// <summary>The extended surface shapes (docs/SURFACES.md): their JSON form, term numbering, parameter paths, editing and structural validation.</summary>
public class ExtendedShapeTests
{
    private static SurfaceShape Decode(string json) => ModelJson.Decode<SurfaceShape>(json);

    private static JsonTree Encoded(SurfaceShape shape) => JsonTree.Parse(ModelJson.Encode(shape));

    private static OpticalSystem Extended() => Fixtures.Load("extended_shapes");

    private static OpticalSystem WithShape(OpticalSystem s, int i, SurfaceShape? shape) => s.WithSurface(i, x => x with { Shape = shape! });

    // MARK: Fixtures

    [Fact]
    public void ExtendedShapesCoverEveryNewKind()
    {
        var s = Fixtures.RoundTrip<OpticalSystem>("extended_shapes");
        Assert.Empty(s.StructuralIssues());
        var kinds = s.Surfaces.Select(x => x.Shape.Kind).ToHashSet();
        Assert.True(Enum.GetValues<ShapeKind>().Where(k => !k.IsLegacy()).ToHashSet().IsSubsetOf(kinds));
        var bases = s.Surfaces.Select(x => x.Shape).OfType<DiffractiveShape>().Select(d => d.Base.Kind);
        Assert.Equal([ShapeKind.Plane, ShapeKind.Conic, ShapeKind.EvenAsphere], bases);
        var orders = s.Surfaces.Select(x => x.Shape).OfType<DiffractiveShape>().Select(d => (d.Phase.DesignOrder, d.Phase.TraceOrder));
        Assert.Equal([(1, 1), (2, -1), (-1, 0)], orders);
        var orderings = s.Surfaces.Select(x => x.Shape).OfType<ZernikeSagShape>().Select(z => z.Ordering);
        Assert.Equal([ZernikeOrdering.Fringe, ZernikeOrdering.Standard], orderings);
        Assert.Equal([CylinderAxis.X], s.Surfaces.Select(x => x.Shape.PowerAxis).OfType<CylinderAxis>());
        Assert.Equal(SurfaceShape.Cylinder(0.025, CylinderAxis.X), s.Surfaces[9].Shape);
        Assert.Contains(s.Surfaces, x => !x.Shape.IsLegacy);
    }

    /// <summary>The samples the app ships (the engine's, embedded as a resource), read through the model's own loader.</summary>
    [Fact]
    public void BundledSamplesRoundTripAndValidate()
    {
        var text = SampleSystems.BundledText;
        Assert.NotNull(text);
        var entries = Fixtures.RoundTripList<SampleEntry>("samples", text);
        Assert.Equal(entries, SampleSystems.Bundled);
        var ids = entries.Select(e => e.Id).ToList();
        Assert.Equal(16, ids.Count);
        foreach (var id in new[]
                 {
                     "achromat", "hybrid_doe", "anamorphic", "biconic", "toroid", "q_con", "q_bfs", "odd_asphere", "xy_freeform", "zernike_sag",
                 })
            Assert.Contains(id, ids);
        foreach (var e in entries) Assert.True(e.System.StructuralIssues().Count == 0, $"{e.Id}: {string.Join("; ", e.System.StructuralIssues())}");
        Assert.Equal(new ApertureSpec.ImageSpaceFNumber(5), SampleSystems.Get("achromat")?.Aperture);
        Assert.Equal([CylinderAxis.X, CylinderAxis.X], SampleSystems.Get("anamorphic")!.Surfaces.Select(x => x.Shape.PowerAxis).OfType<CylinderAxis>());
        Assert.Null(SampleSystems.Get("nonexistent"));
    }

    // MARK: Strict JSON

    [Fact]
    public void DefaultedFieldsMayBeMissingAndAreAlwaysWritten()
    {
        var toroid = Decode("""{"type":"toroid","curvature_x":0.1,"curvature_y":0.2}""");
        Assert.Equal(new ToroidShape(0.1, 0.2, 0, []), toroid);
        Assert.Equal(JsonTree.Parse("""{"type":"toroid","curvature_x":0.1,"curvature_y":0.2,"conic_y":0,"y_asphere":[]}"""), Encoded(toroid));
        var biconic = Decode("""{"type":"biconic","curvature_x":0.1,"curvature_y":0}""");
        Assert.Equal(SurfaceShape.Cylinder(0.1, CylinderAxis.X), biconic);
        Assert.Equal(
            new ZernikeSagShape(0, 0, 5, ZernikeOrdering.Fringe, [1]),
            Decode("""{"type":"zernike_sag","curvature":0,"norm_radius":5,"coefficients":[1]}"""));
        foreach (var json in new[]
                 {
                     """{"type":"odd_asphere","curvature":0,"coefficients":[]}""",
                     """{"type":"q_con","curvature":0,"norm_radius":1,"coefficients":[]}""",
                     """{"type":"xy_polynomial","curvature":0,"norm_radius":1,"coefficients":[]}""",
                 })
            Assert.Equal(0, Decode(json).ConicConstant);
        // Fields without a serde default are required.
        foreach (var json in new[]
                 {
                     """{"type":"odd_asphere","curvature":0}""",
                     """{"type":"q_bfs","curvature":0,"coefficients":[]}""",
                     """{"type":"toroid","curvature_y":0}""",
                     """{"type":"diffractive","base":{"type":"plane"}}""",
                     """{"type":"diffractive","base":{"type":"plane"},"phase":{"norm_radius":1,"coefficients":[],"design_order":1,"design_wavelength_um":0.5}}""",
                 })
            Assert.Throws<JsonException>(() => Decode(json));
    }

    [Fact]
    public void NollIsAnAliasOfStandard()
    {
        var s = Decode("""{"type":"zernike_sag","curvature":0,"conic":0,"norm_radius":5,"ordering":"noll","coefficients":[]}""");
        Assert.Equal(new ZernikeSagShape(0, 0, 5, ZernikeOrdering.Standard, []), s);
        Assert.Equal("standard", Encoded(s)["ordering"]?.AsString);
        Assert.Throws<JsonException>(() => Decode("""{"type":"zernike_sag","curvature":0,"norm_radius":5,"ordering":"ansi","coefficients":[]}"""));
    }

    [Theory]
    [InlineData("""{"type":"odd_asphere","curvature":0,"conic":0,"coefficients":[],"x":1}""")]
    [InlineData("""{"type":"q_con","curvature":0,"conic":0,"norm_radius":1,"coefficients":[],"x":1}""")]
    [InlineData("""{"type":"q_bfs","curvature":0,"conic":0,"norm_radius":1,"coefficients":[]}""")]
    [InlineData("""{"type":"toroid","curvature_x":0,"curvature_y":0,"conic_x":0}""")]
    [InlineData("""{"type":"biconic","curvature_x":0,"curvature_y":0,"x":1}""")]
    [InlineData("""{"type":"xy_polynomial","curvature":0,"conic":0,"norm_radius":1,"coefficients":[],"ordering":"fringe"}""")]
    [InlineData("""{"type":"zernike_sag","curvature":0,"conic":0,"norm_radius":1,"coefficients":[],"x":1}""")]
    [InlineData("""{"type":"diffractive","base":{"type":"plane"},"phase":{"norm_radius":1,"coefficients":[],"design_order":1,"design_wavelength_um":0.5,"trace_order":1},"x":1}""")]
    [InlineData("""{"type":"diffractive","base":{"type":"plane"},"phase":{"norm_radius":1,"coefficients":[],"design_order":1,"design_wavelength_um":0.5,"trace_order":1,"x":1}}""")]
    [InlineData("""{"type":"diffractive","base":{"type":"plane","x":1},"phase":{"norm_radius":1,"coefficients":[],"design_order":1,"design_wavelength_um":0.5,"trace_order":1}}""")]
    public void UnknownKeysAreRejected(string json) => Assert.Throws<JsonException>(() => Decode(json));

    [Fact]
    public void OrdersAreSignedInt32()
    {
        static string Doe(string design, string trace) =>
            """{"type":"diffractive","base":{"type":"plane"},"phase":{"norm_radius":1,"coefficients":[-1],"design_order":"""
            + design + ""","design_wavelength_um":0.5,"trace_order":""" + trace + "}}";
        var s = Assert.IsType<DiffractiveShape>(Decode(Doe("-2147483648", "2147483647")));
        Assert.Equal(new PlaneShape(), s.Base);
        Assert.True(s.Phase.DesignOrder == int.MinValue && s.Phase.TraceOrder == int.MaxValue);
        Assert.Throws<JsonException>(() => Decode(Doe("1", "2147483648")));
        Assert.Throws<JsonException>(() => Decode(Doe("-2147483649", "1")));
    }

    // MARK: Term numbering

    [Fact]
    public void XyTermsFollowSurfacesTable()
    {
        var table = new[] { (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3) };
        for (var j = 0; j < table.Length; j++) Assert.Equal(new XYTerm(table[j].Item1, table[j].Item2), XYTerm.FromIndex(j));
        Assert.Equal(new XYTerm(0, 10), XYTerm.FromIndex(64));
        Assert.Equal(new XYTerm(10, 0), XYTerm.FromIndex(54));
        Assert.True(XYTerm.FromIndex(65) is null && XYTerm.FromIndex(-1) is null);
        for (var j = 0; j < CoefficientLimits.MaxXYTerms; j++) Assert.Equal(j, XYTerm.FromIndex(j)?.Index);
        Assert.Equal("x²y", XYTerm.FromIndex(6)?.Monomial);
        Assert.Equal("C(xy)", CoefficientSet.Xy.TermLabel(3));
    }

    [Fact]
    public void ZernikeTermsFollowSurfacesTable()
    {
        var fringe = new[] { (0, 0), (1, 1), (1, -1), (2, 0), (2, 2), (2, -2), (3, 1), (3, -1), (4, 0), (3, 3), (3, -3) };
        var standard = new[] { (0, 0), (1, 1), (1, -1), (2, 0), (2, -2), (2, 2), (3, -1), (3, 1), (3, -3), (3, 3), (4, 0) };
        for (var k = 0; k < fringe.Length; k++)
            Assert.Equal(new ZernikeTerm(fringe[k].Item1, fringe[k].Item2), ZernikeTerm.FromJ(k + 1, ZernikeOrdering.Fringe));
        for (var k = 0; k < standard.Length; k++)
            Assert.Equal(new ZernikeTerm(standard[k].Item1, standard[k].Item2), ZernikeTerm.FromJ(k + 1, ZernikeOrdering.Standard));
        Assert.Equal(new ZernikeTerm(10, 0), ZernikeTerm.FromJ(36, ZernikeOrdering.Fringe));
        Assert.Equal(new ZernikeTerm(12, 0), ZernikeTerm.FromJ(37, ZernikeOrdering.Fringe));
        Assert.Equal(new ZernikeTerm(8, -8), ZernikeTerm.FromJ(45, ZernikeOrdering.Standard));
        Assert.True(ZernikeTerm.FromJ(38, ZernikeOrdering.Fringe) is null && ZernikeTerm.FromJ(46, ZernikeOrdering.Standard) is null);
        Assert.Null(ZernikeTerm.FromJ(0, ZernikeOrdering.Fringe));
        foreach (var ordering in Enum.GetValues<ZernikeOrdering>())
        {
            var terms = Enumerable.Range(1, ordering.MaxTerms()).Select(j => ZernikeTerm.FromJ(j, ordering)).OfType<ZernikeTerm>().ToList();
            Assert.True(terms.Count == ordering.MaxTerms() && terms.ToHashSet().Count == terms.Count);
            Assert.All(terms, t => Assert.True(Math.Abs(t.M) <= t.N && (t.N - Math.Abs(t.M)) % 2 == 0));
        }
        // Standard: n ≤ 8 is complete; even j cosine, odd j sine.
        var std = Enumerable.Range(1, 45).Select(j => ZernikeTerm.FromJ(j, ZernikeOrdering.Standard)!.Value).ToList();
        Assert.All(std, t => Assert.True(t.N <= 8));
        for (var k = 0; k < std.Count; k++)
            if (std[k].M != 0) Assert.Equal((k + 1) % 2 == 0, std[k].M > 0);
    }

    // MARK: Parameter paths

    [Fact]
    public void CoefficientPathsReadZeroAndExtendWithZeros()
    {
        var s = Extended();
        var id = Sid.Of(1); // odd asphere [0, 1e-5, -1e-7]
        var p = new ParamPath.ShapeCoefficient(id, CoefficientSet.Odd, 6);
        Assert.Equal<double?>(0, s.ValueAt(p));
        var set = s.SetValue(p, 2e-9);
        Assert.NotNull(set);
        Assert.Equal([0, 1e-5, -1e-7, 0, 0, 0, 2e-9], set.Surfaces[1].Shape.CoefficientsIn(CoefficientSet.Odd));
        Assert.Equal<double?>(1e-5, set.ValueAt(new ParamPath.ShapeCoefficient(id, CoefficientSet.Odd, 1)));
        var beyond = new ParamPath.ShapeCoefficient(id, CoefficientSet.Odd, 20);
        Assert.Equal("[coefficient_count] index exceeds shape limit", beyond.UnresolvedReason(s));
        Assert.Null(s.ValueAt(beyond));
        Assert.Null(s.SetValue(beyond, 1));
        Assert.Null(s.SetValue(new ParamPath.ShapeCoefficient(id, CoefficientSet.Q, 0), 1));
        Assert.Equal("coefficient set does not belong to shape", new ParamPath.ShapeCoefficient(id, CoefficientSet.Xy, 0).UnresolvedReason(s));
        // Limits per shape.
        var limits = s.Surfaces.SelectMany(sf => sf.Shape.CoefficientSets.Select(c => $"{Wire.Name(c)}{c.MaxCount(sf.Shape)}")).ToHashSet();
        Assert.True(limits.SetEquals(["odd20", "q20", "phase10", "y_asphere10", "x_asphere10", "xy65", "zernike37", "zernike45"]));
    }

    [Fact]
    public void ShapeParametersAndOrders()
    {
        var s = Extended();
        var trace = new ParamPath.ShapeParam(Sid.Of(5), ShapeParameter.TraceOrder);
        Assert.Equal<double?>(-1, s.ValueAt(trace));
        Assert.Null(s.SetValue(trace, 0.5));
        Assert.Null(s.SetValue(trace, 3e9));
        Assert.Equal("[invalid_diffraction_order] expected an i32", trace.ValueRejection(0.5));
        s = s.SetValue(trace, -2)!;
        Assert.Equal<double?>(-2, s.ValueAt(trace));
        // Order zero is representable for the design order; validation rejects it.
        s = s.SetValue(new ParamPath.ShapeParam(Sid.Of(5), ShapeParameter.DesignOrder), 0)!;
        Assert.Contains(s.StructuralIssues(), i => i.Message.StartsWith("[invalid_diffraction_order] surfaces[5].shape.phase.design_order"));
        s = s.SetValue(new ParamPath.ShapeParam(Sid.Of(3), ShapeParameter.NormRadius), 4.5)!;
        Assert.Equal<double?>(4.5, s.Surfaces[3].Shape.NormRadius);
        s = s.SetValue(new ParamPath.ShapeParam(Sid.Of(8), ShapeParameter.CurvatureX), 0.03)!;
        s = s.SetValue(new ParamPath.ShapeParam(Sid.Of(8), ShapeParameter.ConicX), -2)!;
        Assert.Equal<double?>(0.03, s.Surfaces[8].Shape.ShapeParameterValue(ShapeParameter.CurvatureX));
        Assert.Equal<double?>(-2, s.Surfaces[8].Shape.ShapeParameterValue(ShapeParameter.ConicX));
        Assert.Equal("scalar does not belong to shape", new ParamPath.ShapeParam(Sid.Of(7), ShapeParameter.ConicX).UnresolvedReason(s));
        Assert.Equal([ShapeParameter.CurvatureX, ShapeParameter.CurvatureY, ShapeParameter.ConicY], s.Surfaces[7].Shape.ShapeParameters);
        Assert.Equal(
            [ShapeParameter.NormRadius, ShapeParameter.DesignOrder, ShapeParameter.DesignWavelengthUm, ShapeParameter.TraceOrder],
            s.Surfaces[4].Shape.ShapeParameters);
    }

    [Fact]
    public void LegacyPathsAddressTheDiffractiveBase()
    {
        var s = Extended();
        // Plane base: curvature promotes it to a sphere; it has no conic.
        Assert.Equal<double?>(0, s.ValueAt(new ParamPath.Curvature(Sid.Of(4))));
        Assert.Equal("surface has no conic", new ParamPath.Conic(Sid.Of(4)).UnresolvedReason(s));
        Assert.Null(s.SetValue(new ParamPath.Conic(Sid.Of(4)), -1));
        s = s.SetValue(new ParamPath.Curvature(Sid.Of(4)), 0.01)!;
        var doe = Assert.IsType<DiffractiveShape>(s.Surfaces[4].Shape);
        Assert.Equal(new ConicShape(0.01, 0), doe.Base);
        Assert.Equal([-20, 0.5], doe.Phase.Coefficients);
        s = s.SetValue(new ParamPath.Conic(Sid.Of(4)), -0.5)!;
        Assert.Equal(-0.5, s.Surfaces[4].Shape.ConicConstant);
        // Even-asphere base.
        var a = new ParamPath.Asphere(Sid.Of(6), 3);
        Assert.Equal<double?>(0, s.ValueAt(a));
        s = s.SetValue(a, 1e-12)!;
        Assert.Equal([1e-6, -2e-9, 0, 1e-12], s.Surfaces[6].Shape.AsphereCoefficients);
        Assert.Equal("surface is not an even asphere", new ParamPath.Asphere(Sid.Of(5), 0).UnresolvedReason(s));
        // Curvature on a toroid or biconic is its y section; QBfs has no conic.
        Assert.Equal(s.Surfaces[7].Shape.ShapeParameterValue(ShapeParameter.CurvatureY), s.ValueAt(new ParamPath.Curvature(Sid.Of(7))));
        Assert.Equal(s.Surfaces[8].Shape.ShapeParameterValue(ShapeParameter.CurvatureY), s.ValueAt(new ParamPath.Curvature(Sid.Of(8))));
        Assert.Equal("surface has no conic", new ParamPath.Conic(Sid.Of(3)).UnresolvedReason(s));
        s = s.SetValue(new ParamPath.Curvature(Sid.Of(3)), 0.03)!;
        Assert.Equal(new QBfsShape(0.03, 8, [0.0005]), s.Surfaces[3].Shape);
        // The engine's curvature()/conic(): y values of a toroid / biconic.
        Assert.True(s.Surfaces[7].Shape.Curvature == 0.01 && s.Surfaces[7].Shape.ConicConstant == -0.3);
        Assert.True(s.Surfaces[8].Shape.Curvature == -0.01 && s.Surfaces[8].Shape.ConicConstant == 0.1);
    }

    [Fact]
    public void Labels()
    {
        var s = Extended();
        var id = Sid.Of(3);
        Assert.Equal("Odd β3 on 3", new ParamPath.ShapeCoefficient(id, CoefficientSet.Odd, 2).Label(s));
        Assert.Equal("Q2 on 3", new ParamPath.ShapeCoefficient(id, CoefficientSet.Q, 2).Label(s));
        Assert.Equal("Phase A1 on 3", new ParamPath.ShapeCoefficient(id, CoefficientSet.Phase, 0).Label(s));
        Assert.Equal("C(x²y) on 3", new ParamPath.ShapeCoefficient(id, CoefficientSet.Xy, 6).Label(s));
        Assert.Equal("Z7 on 3", new ParamPath.ShapeCoefficient(id, CoefficientSet.Zernike, 6).Label(s));
        Assert.Equal("A6y on 3", new ParamPath.ShapeCoefficient(id, CoefficientSet.YAsphere, 1).Label(s));
        Assert.Equal("R_x on 3", new ParamPath.ShapeParam(id, ShapeParameter.CurvatureX).Label(s));
        Assert.Equal("Norm radius on 3", new ParamPath.ShapeParam(id, ShapeParameter.NormRadius).Label(s));
        Assert.Equal("Design order on 3", new ParamPath.ShapeParam(id, ShapeParameter.DesignOrder).Label(s));
        Assert.Equal("Radius y 7", new ParamPath.Curvature(Sid.Of(7)).Label(s));
        Assert.Equal("Field 2 y", new ParamPath.FieldY(1).Label(s));
    }

    [Fact]
    public void UnitsFollowTheParameter()
    {
        var s = SampleSystems.CookeTriplet();
        Assert.Equal("1/mm", new ParamPath.Curvature(Sid.Of(1)).Unit(s));
        Assert.Equal("mm⁻³", new ParamPath.Asphere(Sid.Of(1), 0).Unit(s));
        Assert.Equal("mm⁻¹⁹", new ParamPath.Asphere(Sid.Of(1), 8).Unit(s));
        Assert.Equal("°", new ParamPath.FieldY(1).Unit(s));
        Assert.Equal("mm", new ParamPath.ApertureValue().Unit(s));
        Assert.Equal("rad", new ParamPath.ShapeCoefficient(Sid.Of(1), CoefficientSet.Phase, 0).Unit(s));
        Assert.True(new ParamPath.ShapeParam(Sid.Of(1), ShapeParameter.CurvatureY).IsCurvature);
    }

    // MARK: Editing

    [Fact]
    public void ShapeKindConversionsKeepWhatCarriesOver()
    {
        var s = SampleSystems.CookeTriplet().WithSurface(1, x => x with { SemiDiameter = new SemiDiameter.Fixed(7.5) });
        var c = s.Surfaces[1].Shape.Curvature;
        s = s.SetShapeKind(ShapeKind.QCon, 1);
        Assert.Equal(new QConShape(c, 0, 7.5, []), s.Surfaces[1].Shape);
        s = WithShape(s, 1, s.Surfaces[1].Shape.WithCoefficients([0.001], CoefficientSet.Q))
            .WithSurface(1, x => x with { SemiDiameter = new SemiDiameter.Auto() });
        // Between normalised kinds the radius stays.
        s = s.SetShapeKind(ShapeKind.ZernikeSag, 1);
        Assert.Equal(new ZernikeSagShape(c, 0, 7.5, ZernikeOrdering.Fringe, []), s.Surfaces[1].Shape);
        s = s.SetShapeKind(ShapeKind.QBfs, 1);
        Assert.Equal(new QBfsShape(c, 7.5, []), s.Surfaces[1].Shape);
        s = s.SetShapeKind(ShapeKind.Toroid, 1);
        Assert.Equal(new ToroidShape(c, c, 0, []), s.Surfaces[1].Shape);
        s = WithShape(s, 1, s.Surfaces[1].Shape.WithShapeParameter(ShapeParameter.CurvatureX, 0.5));
        s = WithShape(s, 1, s.Surfaces[1].Shape.WithCoefficients([1e-6], CoefficientSet.YAsphere));
        s = s.SetShapeKind(ShapeKind.Biconic, 1);
        Assert.Equal(new BiconicShape(0.5, c, 0, 0, [], [1e-6]), s.Surfaces[1].Shape);
        s = s.SetShapeKind(ShapeKind.Conic, 1);
        Assert.Equal(new ConicShape(c, 0), s.Surfaces[1].Shape);
        s = WithShape(s, 1, s.Surfaces[1].Shape.WithConic(-0.5));
        s = s.SetShapeKind(ShapeKind.Biconic, 1);
        Assert.Equal(new BiconicShape(c, c, -0.5, -0.5, [], []), s.Surfaces[1].Shape);
        s = s.SetShapeKind(ShapeKind.XyPolynomial, 1);
        // No fixed semi-diameter or aperture: 10 mm.
        Assert.Equal(new XyPolynomialShape(c, -0.5, 10, []), s.Surfaces[1].Shape);
        s = s.SetShapeKind(ShapeKind.OddAsphere, 1);
        Assert.Equal(new OddAsphereShape(c, -0.5, []), s.Surfaces[1].Shape);
        Assert.Empty(s.StructuralIssues());
    }

    [Fact]
    public void DiffractiveWrapsAndUnwrapsItsBase()
    {
        var s = SampleSystems.CookeTriplet();
        s = s with { Wavelengths = s.Wavelengths with { Primary = 0 } };
        var um = s.Wavelengths.Wavelengths[0].Um;
        s = s.WithSurface(2, x => x with { Aperture = new ApertureShape.Rectangular(3, 4), Shape = new EvenAsphereShape(-0.01, -1, [1e-6]) });
        s = s.SetShapeKind(ShapeKind.Diffractive, 2);
        var phase = new DiffractivePhase { NormRadius = 5, DesignOrder = 1, DesignWavelengthUm = um, TraceOrder = 1 };
        Assert.Equal(new DiffractiveShape(new EvenAsphereShape(-0.01, -1, [1e-6]), phase), s.Surfaces[2].Shape);
        // Same kind: unchanged.
        s = WithShape(s, 2, s.Surfaces[2].Shape.WithCoefficients([-3], CoefficientSet.Phase));
        var doe = s.Surfaces[2].Shape;
        s = s.SetShapeKind(ShapeKind.Diffractive, 2);
        Assert.Equal(doe, s.Surfaces[2].Shape);
        // Converting a DOE uses its base.
        s = s.SetShapeKind(ShapeKind.EvenAsphere, 2);
        Assert.Equal(new EvenAsphereShape(-0.01, -1, [1e-6]), s.Surfaces[2].Shape);
        // An unsupported base becomes a conic with the old c and k.
        s = WithShape(s, 2, new QConShape(0.02, -0.5, 6, [0.1])).SetShapeKind(ShapeKind.Diffractive, 2);
        var converted = Assert.IsType<DiffractiveShape>(s.Surfaces[2].Shape);
        Assert.Equal(new ConicShape(0.02, -0.5), converted.Base);
        Assert.Equal(6, converted.Phase.NormRadius);
        // A plane stays a plane base.
        s = WithShape(s, 3, new PlaneShape()).SetShapeKind(ShapeKind.Diffractive, 3);
        Assert.Equal(new PlaneShape(), Assert.IsType<DiffractiveShape>(s.Surfaces[3].Shape).Base);
        Assert.Empty(s.StructuralIssues());
    }

    [Fact]
    public void ShapeChangePrunesOverridesThatNoLongerResolve()
    {
        var s = SampleSystems.CookeTriplet();
        var id = s.Surfaces[1].Id;
        s = s with
        {
            Configurations =
            [
                new Configuration
                {
                    Overrides =
                    [
                        new ParamOverride(new ParamPath.Curvature(id), 0.03), new ParamOverride(new ParamPath.Conic(id), -1),
                        new ParamOverride(new ParamPath.Thickness(id), 3),
                    ],
                },
            ],
        };
        ParamPath[] Paths() => s.Configurations[0].Overrides.Select(o => o.Path).ToArray();
        s = s.SetShapeKind(ShapeKind.QBfs, 1);
        Assert.Equal<ParamPath[]>([new ParamPath.Curvature(id), new ParamPath.Thickness(id)], Paths());
        s = s.SetShapeKind(ShapeKind.Toroid, 1);
        Assert.Equal<ParamPath[]>([new ParamPath.Curvature(id), new ParamPath.Thickness(id)], Paths());
        s = s with
        {
            Configurations = [s.Configurations[0] with { Overrides = s.Configurations[0].Overrides.Add(new ParamOverride(new ParamPath.ShapeParam(id, ShapeParameter.CurvatureX), 0.01)) }],
        };
        s = s.SetShapeKind(ShapeKind.Biconic, 1);
        Assert.Equal(3, s.Configurations[0].Overrides.Count);
        s = s.SetShapeKind(ShapeKind.Conic, 1);
        Assert.Equal<ParamPath[]>([new ParamPath.Curvature(id), new ParamPath.Thickness(id)], Paths());
    }

    [Fact]
    public void CylinderPreset()
    {
        var y = SurfaceShape.Cylinder(-0.02, CylinderAxis.Y);
        Assert.Equal(new BiconicShape(0, -0.02, 0, 0, [], []), y);
        Assert.True(y.PowerAxis == CylinderAxis.Y && y.IsCylinder);
        Assert.Equal(CylinderAxis.X, SurfaceShape.Cylinder(0.1, CylinderAxis.X).PowerAxis);
        Assert.Null(new BiconicShape(0.1, 0.1, 0, 0, [], []).PowerAxis);
        Assert.Null(new BiconicShape(0.1, 0, 0, 0, [], [1e-6]).PowerAxis);
        Assert.Null(SurfaceShape.Cylinder(0, CylinderAxis.X).PowerAxis);
        Assert.False(new ToroidShape(0.1, 0, 0, []).IsCylinder);
        var names = Enum.GetValues<ShapeKind>().Select(k => k.TypeName()).ToList();
        Assert.Equal(names.Count, names.ToHashSet().Count);
        Assert.True(ShapeKind.Biconic.Title() == "Biconic" && ShapeKind.QBfs.Title() == "Q-Type (Best-Fit Sphere)");
    }

    // MARK: Structural validation

    [Fact]
    public void StructuralRulesOfExtendedShapes()
    {
        var s = Extended();
        List<string> Messages(Func<OpticalSystem, OpticalSystem> edit) =>
            edit(s).StructuralIssues().Where(i => i.Severity == IssueSeverity.Error).Select(i => i.Message).ToList();
        OpticalSystem Param(OpticalSystem t, int i, ShapeParameter p, double v) => WithShape(t, i, t.Surfaces[i].Shape.WithShapeParameter(p, v));
        OpticalSystem Coef(OpticalSystem t, int i, CoefficientSet set, int count) =>
            WithShape(t, i, t.Surfaces[i].Shape.WithCoefficients(Enumerable.Repeat(0.0, count).ToValueList(), set));

        Assert.Equal(["[invalid_norm_radius] surfaces[3].shape.norm_radius: must be > 0 (got 0.0)"], Messages(t => Param(t, 3, ShapeParameter.NormRadius, 0)));
        Assert.Equal(["[coefficient_count] surfaces[1].shape.coefficients: at most 20 coefficients"], Messages(t => Coef(t, 1, CoefficientSet.Odd, 21)));
        Assert.Equal(["[coefficient_count] surfaces[12].shape.coefficients: at most 37 coefficients"], Messages(t => Coef(t, 12, CoefficientSet.Zernike, 38)));
        Assert.Empty(Messages(t => Coef(t, 13, CoefficientSet.Zernike, 45)));
        Assert.Equal(["[coefficient_count] surfaces[8].shape.x_asphere: at most 10 coefficients"], Messages(t => Coef(t, 8, CoefficientSet.XAsphere, 11)));
        Assert.Equal(
            ["[invalid_design_wavelength] surfaces[5].shape.phase.design_wavelength_um: must be > 0 (got -0.5)"],
            Messages(t => Param(t, 5, ShapeParameter.DesignWavelengthUm, -0.5)));
        Assert.Equal(
            ["[invalid_diffraction_order] surfaces[5].shape.phase.design_order: design order must be nonzero"],
            Messages(t => Param(t, 5, ShapeParameter.DesignOrder, 0)));
        // Trace order zero is ordinary refraction.
        Assert.Empty(Messages(t => Param(t, 5, ShapeParameter.TraceOrder, 0)));
        Assert.Equal(
            ["[invalid_diffractive_base] surfaces[4].shape.base: expected plane, conic or even_asphere"],
            Messages(t => WithShape(t, 4, new DiffractiveShape(new ToroidShape(0, 0, 0, []), new DiffractivePhase { NormRadius = 1, DesignWavelengthUm = 0.5 }))));
        // A non-finite value is reported once, in the JSON check's words.
        Assert.Equal(["surfaces[7].shape.conic_y = nan is not a finite number"], Messages(t => Param(t, 7, ShapeParameter.ConicY, double.NaN)));
        var inBase = Messages(t => WithShape(t, 6, t.Surfaces[6].Shape.WithAsphereCoefficient(double.PositiveInfinity, 0)));
        Assert.Equal(["surfaces[6].shape.base.coefficients[0] = inf is not a finite number"], inBase);
        s = WithShape(s, 2, new ParaxialShape(0));
        Assert.Equal(
            ["[invalid_focal_length] surfaces[2].shape.focal_length: an ideal lens needs a non-zero focal length"],
            s.StructuralIssues().Select(i => i.Message));
    }

    /// <summary>The astigmatism axis is a shape parameter of the sag shape's overlay, as in the engine, and survives fringe edits.</summary>
    [Fact]
    public void OverlayAxisIsAShapeParameter()
    {
        var conic = new ConicShape(0.02, 0);
        Assert.Null(conic.ShapeParameterValue(ShapeParameter.AstigmatismAngleDeg));
        SurfaceShape overlay = new SagOverlayShape(conic, 5, 0.6328, 0.5, 0, 0);
        Assert.Equal([ShapeParameter.AstigmatismAngleDeg], overlay.ShapeParameters);
        overlay = overlay.WithShapeParameter(ShapeParameter.AstigmatismAngleDeg, 30)!;
        overlay = overlay.WithIrregularity(IrregularityMode.Spherical, 0.25, null, 0.6328)!;
        Assert.Equal(new SagOverlayShape(conic, 5, 0.6328, 0.5, 30, 0.25), overlay);
        SurfaceShape doe = new DiffractiveShape(overlay, new DiffractivePhase { NormRadius = 5, DesignWavelengthUm = 0.55 });
        doe = doe.WithShapeParameter(ShapeParameter.AstigmatismAngleDeg, 45)!;
        Assert.Equal<double?>(45, doe.ShapeParameterValue(ShapeParameter.AstigmatismAngleDeg));
    }
}
