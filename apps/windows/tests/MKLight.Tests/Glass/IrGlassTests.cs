using MKLight.App.ViewModels;
using MKLight.Engine;
using static MKLight.Tests.Engine.EngineTestSupport;

namespace MKLight.Tests.Glass;

/// <summary>
/// Infrared glasses (range without the d line): no n_d / V_d, the catalog's listed values in the
/// tooltip, sorted last in either direction; on the real engine with AGF text written here.
/// </summary>
public sealed class IrGlassTests : IDisposable
{
    readonly string _dir = Path.Combine(Path.GetTempPath(), "mklight-ir-glass-" + Guid.NewGuid().ToString("N"));

    public IrGlassTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    /// <summary>A germanium-like and a chalcogenide-like glass listed as vendors list them (n_d 1, V_d 0), and a visible crown.</summary>
    const string Agf = """
        CC Made-up infrared and visible glasses
        NM GELIKE 6 0 1.000000 0.000000 0 0 0
        ED 5.9 0 5.33 0 0
        CD 9.28156 0 6.7288 0.44105 0.21307 3870.1 0 0 0 0
        LD 2.0 12.0
        NM CHALCO 1 0 1.000000 0.000000 0 3 0
        CD 6.9 -0.0005 0.2 0.01 0 0 0 0 0 0
        LD 1.8 12.0
        NM CROWN 2 517642 1.516800 64.17 0 1 0
        CD 1.03961212 0.00600069867 0.231792344 0.0200179144 1.01046945 103.560653 0 0 0 0
        LD 0.3 2.5
        """;

    static GlassRow Row(string name, double? nd, double? vd, double? catalogNd = null, double? catalogVd = null) => new()
    {
        Catalog = "C", Name = name, Nd = nd, Vd = vd, CatalogNd = catalogNd, CatalogVd = catalogVd, MinUm = 2, MaxUm = 12,
    };

    [Fact]
    public void TooltipNamesWhyAndTheListedValues()
    {
        var ir = Row("IR", null, null, 1, 0);
        Assert.Equal("—", GlassFormat.Nd(ir));
        Assert.Equal("—", GlassFormat.Vd(ir));
        Assert.Equal("The d line (587.6 nm) is outside this glass's range.\nCatalog lists nd 1, Vd 0.", GlassFormat.IrTip(ir));
        Assert.Equal("nd 1, Vd 0", GlassFormat.CatalogListed(ir));
        // A user glass lists nothing (n_d 0).
        Assert.Equal("The d line (587.6 nm) is outside this glass's range.", GlassFormat.IrTip(Row("U", null, null, 0, 0)));
        Assert.Null(GlassFormat.CatalogListed(Row("U", null, null)));
        // The d line inside, F or C outside: n_d but no V_d.
        var nir = Row("NIR", 1.6, null, 1.6, 40);
        Assert.Equal("1.60000", GlassFormat.Nd(nir));
        Assert.StartsWith("The F (486.1 nm) or C (656.3 nm) line", GlassFormat.IrTip(nir));
        Assert.EndsWith("Catalog lists nd 1.6, Vd 40.", GlassFormat.IrTip(nir));
        Assert.Null(GlassFormat.IrTip(Row("G", 1.5168, 64.17)));
    }

    [Fact]
    public void GlassesWithoutAValueSortLastInEitherDirection()
    {
        GlassRow[] rows = [Row("IR1", null, null), Row("A", 1.7, 30), Row("IR2", null, null), Row("B", 1.5, 60)];
        string[] Names(GlassColumn c, bool asc) => GlassBrowserModel.Visible(rows, "", GlassFilter.All, true, c, asc).Select(r => r.Name).ToArray();
        Assert.Equal(["B", "A", "IR1", "IR2"], Names(GlassColumn.Nd, true));
        Assert.Equal(["A", "B", "IR1", "IR2"], Names(GlassColumn.Nd, false));
        Assert.Equal(["A", "B", "IR1", "IR2"], Names(GlassColumn.Vd, true));
        Assert.Equal(["B", "A", "IR1", "IR2"], Names(GlassColumn.Vd, false));
        Assert.Null(GlassBrowserModel.SortKey(rows[0], GlassColumn.Nd));
        // Search and filters do not mind the missing values.
        Assert.Equal(["IR1", "IR2"], GlassBrowserModel.Visible(rows, "ir", GlassFilter.Substitution, true, GlassColumn.Nd, false).Select(r => r.Name));
    }

    [Fact]
    public async Task EngineRowsDetailsAndEditor()
    {
        using var engine = SharedEngine().NewSession();
        var path = Path.Combine(_dir, "IRCAT.agf");
        File.WriteAllText(path, Agf.Replace("\n", "\r\n"));
        var info = await engine.LoadCatalogAsync(path);
        Assert.Empty(info.Warnings);
        var rows = await engine.GlassRowsAsync(info.Name);
        var ge = rows.Single(r => r.Name == "GELIKE");
        Assert.Null(ge.Nd);
        Assert.Null(ge.Vd);
        Assert.Equal((1, 0), (ge.CatalogNd, ge.CatalogVd));
        Assert.EndsWith("Catalog lists nd 1, Vd 0.", GlassFormat.IrTip(ge));
        Assert.Equal(1.5168, rows.Single(r => r.Name == "CROWN").Nd!.Value, 4);
        Assert.Equal(["CROWN", "CHALCO", "GELIKE"], GlassBrowserModel.Visible(rows, "", GlassFilter.All, false, GlassColumn.Nd, true).Select(r => r.Name));

        // Details: no d line, a curve inside the range with a note that it stops short of 12 µm.
        var d = await engine.GlassDetailsAsync(info.Name, "GELIKE");
        Assert.Null(d.ComputedNd);
        Assert.DoesNotContain(d.Lines, l => l.Line == "d");
        Assert.All(d.IndexCurve, p => Assert.InRange(p.Um, 2, 12));
        Assert.All(d.IndexCurve, p => Assert.True(double.IsFinite(p.N) && p.N > 3.5));
        if (d.IndexCurve.Count > 1 && d.IndexCurve[^1].Um < 12) Assert.StartsWith("Drawn for 2.00–", GlassFormat.CurveNote(d));
        Assert.Null(GlassFormat.CurveNote(await engine.GlassDetailsAsync(info.Name, "CROWN")));

        // Duplicate into a user catalog: the formula is kept, the preview has no n_d / V_d, and it saves.
        var lib = new GlassLibrary(Path.Combine(_dir, "lib"));
        lib.CreateUserCatalog("Mine");
        var e = GlassEditorModel.Duplicate(d, "Mine", lib);
        Assert.Equal(GlassEditorModel.Kind.Formula, e.Dispersion);
        Assert.Equal(6, e.Formula);
        Assert.Equal((2.0, 12.0), (e.MinUm, e.MaxUm));
        Assert.NotNull(e.Check(engine));
        Assert.Equal((null, null), e.PreviewNumbers!.Value);
        Assert.Equal("No nd, Vd (the d line is outside the range)", e.PreviewText);
        Assert.True(e.Save(engine));

        // The visible glass's editor shows its numbers.
        var crown = GlassEditorModel.Duplicate(await engine.GlassDetailsAsync(info.Name, "CROWN"), "Mine", lib);
        Assert.NotNull(crown.Check(engine));
        Assert.StartsWith("nd 1.5168", crown.PreviewText);
    }
}
