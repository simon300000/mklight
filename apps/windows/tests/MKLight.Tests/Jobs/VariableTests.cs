using MKLight.App.ViewModels;
using MKLight.App.ViewModels.Jobs;
using MKLight.Model;

namespace MKLight.Tests.Jobs;

public class VariableTests
{
    static DesignDocument DoubleGauss() => DesignDocument.FromSample("double_gauss");

    static SurfaceID Id(DesignDocument d, int index) => d.System.Surfaces[index].Id;

    [Fact]
    public void MakeVariableIsOneUndoableStepWithDefaultBounds()
    {
        var doc = DoubleGauss();
        var t = new ParamPath.Thickness(Id(doc, 2)); // an air gap
        Assert.Null(DesignVariables.Set(doc, t, true));
        var v = Assert.Single(DesignVariables.Of(doc));
        Assert.Equal(0, v.Lower);
        Assert.Null(v.Upper);
        Assert.Equal("Make Variable", doc.UndoActionName);
        doc.Undo();
        Assert.Empty(DesignVariables.Of(doc));
    }

    [Fact]
    public void GlassThicknessKeepsAMinimumAndCurvatureIsFree()
    {
        var doc = DoubleGauss();
        Assert.Equal((DesignVariable.GlassMinimum, (double?)null), DesignVariable.DefaultBounds(new ParamPath.Thickness(Id(doc, 1)), doc.System));
        Assert.Equal(((double?)null, (double?)null), DesignVariable.DefaultBounds(new ParamPath.Curvature(Id(doc, 1)), doc.System));
        Assert.Equal((1.4, 2.1), DesignVariable.DefaultBounds(new ParamPath.ModelNd(Id(doc, 1)), doc.System));
    }

    [Fact]
    public void ToggleTwiceRemovesAndMatchesTheCanonicalParameter()
    {
        var doc = DoubleGauss();
        var c = new ParamPath.Curvature(Id(doc, 1));
        DesignVariables.Toggle(doc, c);
        Assert.True(DesignVariables.IsVariable(doc, c));
        DesignVariables.Toggle(doc, c);
        Assert.False(DesignVariables.IsVariable(doc, c));
        Assert.Equal("Remove Variable", doc.UndoActionName);
    }

    [Fact]
    public void SolvedParameterIsRefusedWithTheEnginesReason()
    {
        var doc = DoubleGauss();
        var target = new ParamPath.Thickness(Id(doc, doc.System.ImageIndex - 1));
        doc.Edit("Add Solve", s => s with { Solves = s.Solves.Add(new Solve(target, new SolveKind.MarginalRayHeight(0))) });
        var why = DesignVariables.Set(doc, target, true);
        Assert.NotNull(why);
        Assert.Contains("Marginal ray height", why);
        Assert.Empty(DesignVariables.Of(doc));
        Assert.Equal("Add Solve", doc.UndoActionName);
    }

    [Fact]
    public void BoundsPinAndUnpin()
    {
        var doc = DoubleGauss();
        var t = new ParamPath.Thickness(Id(doc, 2));
        DesignVariables.Set(doc, t, true);
        var v = DesignVariables.Of(doc)[0];
        Assert.True(DesignVariables.TogglePin(doc, v));
        Assert.Equal("Pin Variable", doc.UndoActionName);
        v = DesignVariables.Of(doc)[0];
        Assert.True(v.IsPinned);
        Assert.Equal(doc.System.ValueAt(t), v.Lower);
        DesignVariables.TogglePin(doc, v);
        Assert.Equal("Unpin Variable", doc.UndoActionName);
        Assert.False(DesignVariables.Of(doc)[0].IsPinned);
        Assert.True(DesignVariables.SetBounds(doc, t, 0.1, 2));
        Assert.Equal("Change Bounds", doc.UndoActionName);
        Assert.False(DesignVariables.SetBounds(doc, t, 0.1, 2));
    }

    [Fact]
    public void SectionRoundTripsThroughJson()
    {
        var doc = DoubleGauss();
        var list = new[] { new DesignVariable(new ParamPath.Curvature(Id(doc, 1)), -0.1, 0.1), new DesignVariable(new ParamPath.Thickness(Id(doc, 2))) };
        var tree = DesignVariables.Encode(list);
        Assert.Equal(list, DesignVariables.Read(JsonTree.Parse(tree.ToString())));
        Assert.DoesNotContain("lower", tree.Node![1]!.ToJsonString());
    }

    [Fact]
    public void ReplaceSystemIsOneStepAndDropsVanishedVariables()
    {
        var doc = DoubleGauss();
        var c1 = new ParamPath.Curvature(Id(doc, 1));
        var a = new ParamPath.Asphere(Id(doc, 2), 0); // a sphere has no A4: does not resolve
        doc.EditSection("variables", DesignVariables.Encode([new DesignVariable(c1), new DesignVariable(a)]), "Set Variables");
        var changed = doc.System.SetValue(c1, 0.03)!;
        Assert.True(DesignVariables.ReplaceSystem(doc, changed, "Apply Optimization"));
        Assert.Equal(0.03, doc.System.ValueAt(c1));
        Assert.Equal([c1], DesignVariables.Of(doc).Select(v => v.Param));
        Assert.Equal("Apply Optimization", doc.UndoActionName);
        doc.Undo();
        Assert.Equal(2, DesignVariables.Of(doc).Count);
        Assert.NotEqual(0.03, doc.System.ValueAt(c1));
    }

    [Fact]
    public void ProblemsFlagSolvedAndOutOfBounds()
    {
        var doc = DoubleGauss();
        var t = new ParamPath.Thickness(Id(doc, 2));
        var now = doc.System.ValueAt(t)!.Value;
        Assert.StartsWith("Now ", DesignVariables.Problem(new DesignVariable(t, now + 1, now + 2), doc.System, true));
        Assert.Null(DesignVariables.Problem(new DesignVariable(t, 0, null), doc.System, true));
        Assert.Equal("The minimum is above the maximum.", DesignVariables.Problem(new DesignVariable(t, 3, 2), doc.System, true));
    }
}

public class RadiusBoundsTests
{
    static void Same(double? expected, double? actual)
    {
        if (expected is null) Assert.Null(actual);
        else Assert.Equal(expected.Value, actual!.Value, 12);
    }

    [Theory]
    [InlineData(40.0, 60.0, 1 / 60.0, 1 / 40.0)]
    [InlineData(-60.0, -40.0, -1 / 40.0, -1 / 60.0)]
    [InlineData(100.0, -100.0, -0.01, 0.01)]
    public void WindowsBecomeCurvatureIntervals(double min, double max, double lower, double upper)
    {
        var (lo, hi, error) = RadiusBounds.Curvature(min, max);
        Assert.Null(error);
        Same(lower, lo);
        Same(upper, hi);
    }

    [Fact]
    public void OpenSidesAndZero()
    {
        var (lo, hi, _) = RadiusBounds.Curvature(50, null);
        Same(0, lo);
        Same(0.02, hi);
        (lo, hi, _) = RadiusBounds.Curvature(0, 50);
        Same(0.02, lo);
        Same(null, hi);
        (lo, hi, _) = RadiusBounds.Curvature(double.PositiveInfinity, double.PositiveInfinity);
        Same(0, lo);
        Same(0, hi);
        Assert.Equal((null, null, null), RadiusBounds.Curvature(null, null));
    }

    [Theory]
    [InlineData(-50.0, 50.0)]
    [InlineData(60.0, 40.0)]
    public void WindowsThroughZeroAreRefused(double min, double max) =>
        Assert.Equal(RadiusBounds.ContainsZero, RadiusBounds.Curvature(min, max).Error);

    [Theory]
    [InlineData(40.0, 60.0)]
    [InlineData(-60.0, -40.0)]
    [InlineData(100.0, -100.0)]
    public void RadiusIsTheInverse(double min, double max)
    {
        var (lo, hi, _) = RadiusBounds.Curvature(min, max);
        var (a, b) = RadiusBounds.Radius(lo, hi);
        Same(min, a);
        Same(max, b);
    }

    [Fact]
    public void FlatPinAndWrapCaption()
    {
        Assert.Equal((double.PositiveInfinity, double.PositiveInfinity), RadiusBounds.Radius(0, 0));
        Assert.True(RadiusBounds.PassesThroughFlat(100, -100));
        Assert.Equal("R 100 … ∞ … −100 mm", BoundsFormat.Counterpart(-0.01, 0.01, CurvatureBoundsUnit.Curvature));
        Assert.Equal("c 0.01667 … 0.025 1/mm", BoundsFormat.Counterpart(1 / 60.0, 0.025, CurvatureBoundsUnit.Radius));
    }
}
