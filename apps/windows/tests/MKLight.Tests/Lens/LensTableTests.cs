using MKLight.App.ViewModels;
using MKLight.Model;

namespace MKLight.Tests.Lens;

public class LensValueParserTests
{
    [Theory]
    [InlineData("1.5", 1.5)]
    [InlineData("1,5", 1.5)]
    [InlineData(" -2 ", -2)]
    [InlineData("−3.25", -3.25)]
    [InlineData("1e3", 1000)]
    public void NumbersReadWithPointCommaAndTypographicMinus(string text, double expected) =>
        Assert.Equal(expected, LensValueParser.Number(text));

    [Theory]
    [InlineData("inf")]
    [InlineData("∞")]
    [InlineData("Infinity")]
    public void InfinityIsAccepted(string text) => Assert.Equal(double.PositiveInfinity, LensValueParser.Number(text));

    [Theory]
    [InlineData("")]
    [InlineData("abc")]
    [InlineData("nan")]
    [InlineData("1,000.5")]
    public void NonNumbersAreRefused(string text) => Assert.Null(LensValueParser.Number(text));

    [Theory]
    [InlineData("25 mm", 25)]
    [InlineData("1 in", 25.4)]
    [InlineData("2cm", 20)]
    [InlineData("500 µm", 0.5)]
    public void LengthsTakeUnits(string text, double mm) => Assert.Equal(mm, LensValueParser.Length(text)!.Value, 12);

    [Theory]
    [InlineData("inf", 0)]
    [InlineData("∞", 0)]
    [InlineData("0", 0)]
    [InlineData("50", 0.02)]
    [InlineData("-50", -0.02)]
    public void RadiusTextBecomesCurvature(string text, double c)
    {
        var p = LensValueParser.CurvatureFromRadius(text);
        Assert.True(p.Ok);
        Assert.Equal(c, p.Value, 15);
    }

    [Fact]
    public void ThicknessIsFiniteExceptOnTheObject()
    {
        Assert.False(LensValueParser.Thickness("inf", allowInfinite: false).Ok);
        Assert.Equal(LensValueParser.MustBeFinite, LensValueParser.Thickness("inf", allowInfinite: false).Error);
        Assert.True(LensValueParser.Thickness("inf", allowInfinite: true).Ok);
    }

    [Fact]
    public void SemiDiameterIsAutoOrPositive()
    {
        Assert.IsType<SemiDiameter.Auto>(LensValueParser.SemiDiameter("auto").Value);
        Assert.IsType<SemiDiameter.Auto>(LensValueParser.SemiDiameter("").Value);
        Assert.Equal(new SemiDiameter.Fixed(12.5), LensValueParser.SemiDiameter("12,5").Value);
        Assert.Equal(LensValueParser.MustBePositive, LensValueParser.SemiDiameter("-1").Error);
        Assert.Equal(LensValueParser.MustBeFinite, LensValueParser.SemiDiameter("inf").Error);
        Assert.Contains("is not a number", LensValueParser.SemiDiameter("x").Error);
    }

    [Fact]
    public void MaterialsReadEveryForm()
    {
        Assert.Equal(new MaterialEdit(new MaterialRef.Air(), Interaction.Refract), LensValueParser.Material("").Value);
        Assert.Equal(new MaterialEdit(new MaterialRef.Air(), Interaction.Refract), LensValueParser.Material("air").Value);
        Assert.Equal(new MaterialEdit(null, Interaction.Reflect), LensValueParser.Material("mirror").Value);
        Assert.Equal(new MaterialEdit(new MaterialRef.ModelGlass(1.5, 60, 0), Interaction.Refract), LensValueParser.Material("1,5/60").Value);
        Assert.Equal(new MaterialEdit(new MaterialRef.ConstantIndex(1.7), Interaction.Refract), LensValueParser.Material("n=1.7").Value);
        Assert.Equal(new MaterialEdit(new MaterialRef.ConstantIndex(1.7), Interaction.Refract), LensValueParser.Material("1.7").Value);
        Assert.Equal(MaterialRef.Glass("N-BK7"), LensValueParser.Material("n-bk7").Value.Material);
        Assert.Equal(new MaterialRef.CatalogGlass { Catalog = "SCHOTT", Name = "N-SF6" }, LensValueParser.Material("schott:n-sf6").Value.Material);
        Assert.False(LensValueParser.Material("0.5/60").Ok);
        Assert.False(LensValueParser.Material("bad glass!").Ok);
        Assert.False(LensValueParser.Material(":X").Ok);
    }

    [Fact]
    public void RetypingTheSameGlassKeepsItsMeltOffset()
    {
        var offset = new MaterialOffset(0.001, -0.5);
        var current = new MaterialRef.CatalogGlass { Catalog = "SCHOTT", Name = "N-SK16", Offset = offset };
        Assert.Equal(current, LensValueParser.Replace(current, MaterialRef.Glass("N-SK16")));
        Assert.Equal(MaterialRef.Glass("F2"), LensValueParser.Replace(current, MaterialRef.Glass("F2")));
    }
}

public class LensFormatTests
{
    [Theory]
    [InlineData(28.074, 4, "28.0740")]
    [InlineData(-13.853, 4, "−13.8530")]
    [InlineData(-0.00001, 4, "−1.0000e-05")]
    [InlineData(-0.00004, 3, "−4.0000e-05")]
    [InlineData(-0.0003, 3, "0.000")]
    [InlineData(15.5843, 3, "15.584")]
    [InlineData(double.PositiveInfinity, 4, "∞")]
    [InlineData(double.NegativeInfinity, 4, "−∞")]
    [InlineData(12345678.0, 4, "1.2346e+07")]
    public void LengthsHaveFixedDecimalsAndNoNegativeZero(double v, int decimals, string expected) =>
        Assert.Equal(expected, LensFormat.Length(v, decimals));

    [Fact]
    public void NegativeZeroNeverShows()
    {
        Assert.Equal("0.0000", LensFormat.Length(-0.0, 4));
        Assert.Equal("0", LensFormat.Plain(-0.0));
    }

    [Theory]
    [InlineData(0, "0")]
    [InlineData(-1, "−1")]
    [InlineData(-0.5, "−0.5")]
    [InlineData(1.234567891, "1.23457")]
    [InlineData(2e-4, "2.0000e-04")]
    public void PlainKeepsSixSignificantDigits(double v, string expected) => Assert.Equal(expected, LensFormat.Plain(v));

    [Fact]
    public void MeltOffsetsAreSignedAndTrimmed() =>
        Assert.Equal("+0.0012/−0.8", LensFormat.Offset(new MaterialOffset(0.0012, -0.8)));
}

public class FieldEditTests
{
    static readonly FieldFormat<double> Format = new(
        v => NumberDisplay.Exact(v), v => LensFormat.Length(v), t => LensValueParser.Number(t) is { } v ? new Optional<double>(v) : Optional<double>.None);

    [Fact]
    public void CompactAtRestExactWithFocus()
    {
        var f = new FieldEdit<double>(1.0 / 3, Format);
        Assert.Equal("0.3333", f.Text);
        f.BeginEditing();
        Assert.Equal(NumberDisplay.Exact(1.0 / 3), f.Text);
        f.EndEditing();
        Assert.Equal("0.3333", f.Text);
    }

    [Fact]
    public void UneditedTextIsNeverCommitted()
    {
        var f = new FieldEdit<double>(1.0 / 3, Format);
        f.BeginEditing();
        Assert.IsType<FieldCommit<double>.Unchanged>(f.Submit());
        Assert.IsType<FieldCommit<double>.Unchanged>(f.EndEditing());
        Assert.Equal(1.0 / 3, f.Value);
    }

    [Fact]
    public void AValueEqualToTheCurrentOneIsNotCommitted()
    {
        var f = new FieldEdit<double>(2, Format);
        f.BeginEditing();
        f.Text = "2,0";
        Assert.IsType<FieldCommit<double>.Unchanged>(f.Submit());
        f.Text = "3";
        Assert.Equal(new FieldCommit<double>.Value(3), f.Submit());
    }

    [Fact]
    public void InvalidTextStaysUntilReverted()
    {
        var f = new FieldEdit<double>(2, Format);
        f.BeginEditing();
        f.Text = "abc";
        Assert.Equal(new FieldCommit<double>.Invalid("abc"), f.EndEditing());
        Assert.Equal("abc", f.Text);
        f.Revert();
        Assert.Equal("2.0000", f.Text);
    }

    [Fact]
    public void UneditedTextFollowsTheValueEditedTextStays()
    {
        var f = new FieldEdit<double>(2, Format);
        f.BeginEditing();
        f.ValueChanged(5);
        Assert.Equal("5.0", f.Text);
        f.Text = "7";
        f.ValueChanged(6);
        Assert.Equal("7", f.Text);
        Assert.Equal(new FieldCommit<double>.Value(7), f.Submit());
    }

    [Fact]
    public void TextTypedForOneOwnerGoesToThatOwner()
    {
        var f = new FieldEdit<double>(2, Format, owner: "S1");
        f.BeginEditing();
        f.Text = "9";
        var earlier = f.ValueChanged(4, "S2", out var earlierOwner);
        Assert.Equal(new FieldCommit<double>.Value(9), earlier);
        Assert.Equal("S1", earlierOwner);
        Assert.Equal("S2", f.Owner);
        Assert.Equal(NumberDisplay.Exact(4), f.Text);
    }
}

public class LensRowTests
{
    static OpticalSystem DoubleGauss => SampleSystems.Get("double_gauss")!;

    [Fact]
    public void DoubleGaussRowsReadLikeThePrescription()
    {
        var s = DoubleGauss;
        var rows = LensRows.Build(s, new Dictionary<int, double> { [1] = 15.5843 });
        Assert.Equal(s.Surfaces.Count, rows.Count);
        Assert.Equal("OBJ", rows[0].SurfaceText);
        Assert.Equal("STO", rows[s.Stop].SurfaceText);
        Assert.Equal("IMA", rows[^1].SurfaceText);
        Assert.Equal("1", rows[1].SurfaceText);
        Assert.Equal("Plane", rows[0].TypeTitle);
        Assert.Equal("Sphere", rows[1].TypeTitle);
        Assert.Equal("28.0740", rows[1].Text(LensColumn.Radius));
        Assert.Equal("4.6200", rows[1].Text(LensColumn.Thickness));
        Assert.Equal("N-SK16", rows[1].Text(LensColumn.Material));
        Assert.Equal("15.584", rows[1].Text(LensColumn.SemiDiameter));
        Assert.True(rows[1].SemiDiameterIsAuto);
        Assert.True(rows[1].IsDimmed(LensColumn.SemiDiameter));
        Assert.Equal("auto", rows[2].Text(LensColumn.SemiDiameter));
        Assert.Equal("0", rows[1].Text(LensColumn.Conic));
        Assert.Equal("∞", rows[0].Text(LensColumn.Thickness));
        // The object at infinity has no radius; the image has no thickness and no material edit.
        Assert.False(rows[0].Editable.Contains(LensColumn.Radius));
        Assert.False(rows[^1].Editable.Contains(LensColumn.Thickness));
        Assert.False(rows[^1].Editable.Contains(LensColumn.Material));
        Assert.Contains(rows, r => r.Text(LensColumn.Radius) == "∞" && r.TypeTitle == "Plane");
        Assert.Contains(rows, r => r.Text(LensColumn.Radius).StartsWith('−'));
        Assert.Equal("11 surfaces · stop on 6 · mm", LensRows.Caption(s));
    }

    [Fact]
    public void ExactTextParsesBackToTheValue()
    {
        var row = LensRows.Build(DoubleGauss)[1];
        Assert.Null(row.Commit(LensColumn.Radius, row.ExactText(LensColumn.Radius)));
        Assert.Null(row.Commit(LensColumn.Thickness, row.ExactText(LensColumn.Thickness)));
        Assert.Null(row.Commit(LensColumn.SemiDiameter, "auto"));
    }

    [Fact]
    public void BiconicShowsBothSectionsAndCylinderDimsItsFlatOne()
    {
        var s = SampleSystems.Get("anamorphic")!;
        var rows = LensRows.Build(s);
        var bi = rows.First(r => r.Kind == ShapeKind.Biconic);
        Assert.Equal([LensColumn.Radius, LensColumn.RadiusX], bi.Shown(LensColumn.Radius));
        Assert.Equal("y", bi.Hints[LensColumn.Radius]);
        Assert.Equal("x", bi.Hints[LensColumn.RadiusX]);

        var cyl = s.WithSurface(1, sf => sf with { Shape = SurfaceShape.Cylinder(-0.02, CylinderAxis.X) });
        var row = LensRows.Build(cyl)[1];
        Assert.Equal("Cylinder", row.TypeTitle);
        Assert.Equal("∞", row.Text(LensColumn.Radius));
        Assert.Equal("−50", row.Text(LensColumn.RadiusX));
        Assert.True(row.IsFlatSection(LensColumn.Radius));
        Assert.False(row.IsFlatSection(LensColumn.RadiusX));
        // Only the powered section's conic shows.
        Assert.Equal([LensColumn.ConicX], row.Shown(LensColumn.Conic));
    }

    [Fact]
    public void ToroidEditsBothSections()
    {
        var s = SampleSystems.Get("toroid")!;
        var i = s.Surfaces.ToList().FindIndex(sf => sf.Shape is ToroidShape);
        var row = LensRows.Build(s)[i];
        var edit = row.Commit(LensColumn.RadiusX, "40")!.Value;
        Assert.True(edit.Ok);
        Assert.Equal("Change R_x", edit.Value.ActionName);
        var after = edit.Value.Apply(s)!;
        Assert.Equal(1 / 40.0, ((ToroidShape)after.Surfaces[i].Shape).CurvatureX, 15);
        var conic = row.Commit(LensColumn.Conic, "-1")!.Value.Value.Apply(s)!;
        Assert.Equal(-1, ((ToroidShape)conic.Surfaces[i].Shape).ConicY);
        Assert.Null(row.Edit(LensColumn.ConicX, "1"));
    }

    [Fact]
    public void SolvesAndVariablesMarkTheirCells()
    {
        var s = DoubleGauss;
        var id = s.Surfaces[1].Id;
        s = s with { Solves = [new Solve(new ParamPath.Thickness(s.Surfaces[^2].Id), new SolveKind.MarginalRayHeight(0))] };
        var curvature = new ParamPath.Curvature(id).ToJson().ToJsonString();
        var thickness = new ParamPath.Thickness(id).ToJson().ToJsonString();
        var variables = LensRows.Variables(JsonTree.Parse(
            $$"""[{"param": {{curvature}}, "lower": null, "upper": null}, {"param": {{thickness}}, "lower": 1, "upper": 1}]"""));
        var rows = LensRows.Build(s, variables: variables);
        Assert.Equal("M", rows[^2].Markers[LensColumn.Thickness].SolveLetter);
        Assert.True(rows[1].Markers[LensColumn.Radius].IsVariable);
        Assert.Equal(VariableBounds.Free, rows[1].Markers[LensColumn.Radius].Bounds);
        Assert.Equal(VariableBounds.Pinned, rows[1].Markers[LensColumn.Thickness].Bounds);
        Assert.False(rows[1].Markers.ContainsKey(LensColumn.Conic));
    }

    [Fact]
    public void MeltOffsetShowsAsANote()
    {
        var s = DoubleGauss.WithSurface(1, sf => sf with
        {
            Material = new MaterialRef.CatalogGlass { Name = "N-SK16", Offset = new MaterialOffset(0.0012, -0.8) },
        });
        var row = LensRows.Build(s)[1];
        Assert.Equal("N-SK16", row.Text(LensColumn.Material));
        Assert.Equal("+0.0012/−0.8", row.Notes[LensColumn.Material]);
        // Retyping the glass keeps the offset; the same text is not an edit at all.
        Assert.Null(row.Commit(LensColumn.Material, "N-SK16"));
        var kept = row.Commit(LensColumn.Material, "n-sk16")!.Value.Value.Apply(s)!;
        Assert.Equal(new MaterialOffset(0.0012, -0.8), ((MaterialRef.CatalogGlass)kept.Surfaces[1].Material).Offset);
    }
}

public class LensGridLayoutTests
{
    [Fact]
    public void WidthsFillTheAvailableWidthWithinLimits()
    {
        var columns = LensGridLayout.Columns(hasOffsets: false);
        var ideal = columns.Sum(c => c.Ideal);
        foreach (var total in new[] { LensGridLayout.MinimumWidth(columns), ideal, ideal + 400 })
        {
            var w = LensGridLayout.Widths(columns, total);
            Assert.Equal(total, w.Sum(), 6);
            for (var i = 0; i < w.Length; i++) Assert.InRange(w[i], columns[i].Min, columns[i].Max);
        }
        // Below the minimum, every column stays at its minimum (the grid scrolls sideways).
        Assert.Equal(columns.Select(c => c.Min), LensGridLayout.Widths(columns, 100));
        Assert.True(LensGridLayout.Columns(hasOffsets: true)[5].Min > columns[5].Min);
    }
}

public class LensEditTests
{
    static OpticalSystem DoubleGauss => SampleSystems.Get("double_gauss")!;

    [Fact]
    public void CommitsProduceTheEditedSystem()
    {
        var s = DoubleGauss;
        var row = LensRows.Build(s)[1];
        var radius = row.Commit(LensColumn.Radius, "30")!.Value;
        Assert.Equal("Change Radius", radius.Value.ActionName);
        Assert.Equal(1 / 30.0, radius.Value.Apply(s)!.Surfaces[1].Shape.Curvature, 15);

        var flat = row.Commit(LensColumn.Radius, "inf")!.Value.Value.Apply(s)!;
        Assert.Equal(0, flat.Surfaces[1].Shape.Curvature);

        var t = row.Commit(LensColumn.Thickness, "5,5")!.Value;
        Assert.Equal("Change Thickness", t.Value.ActionName);
        Assert.Equal(5.5, t.Value.Apply(s)!.Surfaces[1].Thickness);

        var sd = row.Commit(LensColumn.SemiDiameter, "12")!.Value.Value.Apply(s)!;
        Assert.Equal(new SemiDiameter.Fixed(12), sd.Surfaces[1].SemiDiameter);

        var mirror = row.Commit(LensColumn.Material, "MIRROR")!.Value;
        Assert.Equal("Change Material", mirror.Value.ActionName);
        Assert.Equal(Interaction.Reflect, mirror.Value.Apply(s)!.Surfaces[1].Interaction);

        var label = row.Commit(LensColumn.Comment, "front")!.Value.Value;
        Assert.Equal("Change Label", label.ActionName);
        Assert.Equal("front", label.Apply(s)!.Surfaces[1].Label);

        var conic = row.Commit(LensColumn.Conic, "-1")!.Value.Value.Apply(s)!;
        Assert.Equal(-1, conic.Surfaces[1].Shape.ConicConstant);
    }

    [Fact]
    public void RejectedInputExplainsWhy()
    {
        var row = LensRows.Build(DoubleGauss)[1];
        var bad = row.Commit(LensColumn.Thickness, "inf")!.Value;
        Assert.False(bad.Ok);
        Assert.Equal(LensValueParser.MustBeFinite, bad.Error);
        Assert.Contains("is not a number", row.Commit(LensColumn.Radius, "r30")!.Value.Error);
    }

    [Fact]
    public void ObjectThicknessMayBeInfinite()
    {
        var row = LensRows.Build(DoubleGauss)[0];
        Assert.Null(row.Commit(LensColumn.Thickness, "inf"));
        Assert.True(row.Commit(LensColumn.Thickness, "1000")!.Value.Ok);
    }

    [Fact]
    public void TypeChangesUseTheModel()
    {
        var s = DoubleGauss;
        var id = s.Surfaces[1].Id;
        var asphere = new LensEdit.SetType(id, new SurfaceTypeChoice(ShapeKind.EvenAsphere)).Apply(s)!;
        Assert.IsType<EvenAsphereShape>(asphere.Surfaces[1].Shape);
        Assert.Equal(s.Surfaces[1].Shape.Curvature, asphere.Surfaces[1].Shape.Curvature);

        var cylinder = new LensEdit.SetType(id, new SurfaceTypeChoice(ShapeKind.Biconic, CylinderAxis.Y)).Apply(s)!;
        Assert.Equal(CylinderAxis.Y, cylinder.Surfaces[1].Shape.PowerAxis);
        Assert.Equal(new SurfaceTypeChoice(ShapeKind.Biconic, CylinderAxis.Y), LensRows.Build(cylinder)[1].TypeChoice);
        Assert.Equal("Change Surface Type", new LensEdit.SetType(id, new(ShapeKind.Plane)).ActionName);
    }

    [Fact]
    public void InsertDeleteAndStop()
    {
        var s = DoubleGauss;
        var before = LensCommands.InsertBefore(s, 3).Apply(s)!;
        Assert.Equal(s.Surfaces.Count + 1, before.Surfaces.Count);
        Assert.Equal(s.Surfaces[3].Id, before.Surfaces[4].Id);
        Assert.Equal(s.Stop + 1, before.Stop);

        var after = LensCommands.InsertAfter(s, s.ImageIndex).Apply(s)!;
        Assert.Equal(s.Surfaces[^1].Id, after.Surfaces[^1].Id);

        Assert.False(LensCommands.CanDelete(s, 0));
        Assert.False(LensCommands.CanDelete(s, s.ImageIndex));
        var deleted = new LensEdit.Delete(2).Apply(s)!;
        Assert.Equal(s.Surfaces.Count - 1, deleted.Surfaces.Count);
        Assert.Equal(2, LensCommands.SelectionAfterDelete(deleted, 2));

        Assert.True(LensCommands.CanMakeStop(s, 2));
        Assert.False(LensCommands.CanMakeStop(s, s.Stop));
        Assert.Equal(2, new LensEdit.MakeStop(2).Apply(s)!.Stop);
        Assert.Equal("Set Stop", new LensEdit.MakeStop(2).ActionName);
    }

    [Fact]
    public void EveryMenuTypeApplies()
    {
        var s = DoubleGauss;
        foreach (var entry in SurfaceTypeCatalog.All)
        {
            var after = new LensEdit.SetType(s.Surfaces[1].Id, entry.Choice).Apply(s)!;
            Assert.Equal(entry.Choice, SurfaceTypeChoice.For(after.Surfaces[1].Shape));
        }
    }
}
