using System.Text.Json.Nodes;
using MKLight.App.ViewModels;
using MKLight.Engine;
using static MKLight.Tests.Engine.EngineTestSupport;

namespace MKLight.Tests.Glass;

/// <summary>The Glass Catalogs window's model and the library sync, on the real engine with AGF text written here.</summary>
public sealed class GlassBrowserTests : IDisposable
{
    readonly string _dir = Path.Combine(Path.GetTempPath(), "mklight-glass-browser-" + Guid.NewGuid().ToString("N"));

    public GlassBrowserTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    /// <summary>Two made-up Sellmeier glasses (one preferred, one obsolete and excluded) and an IR glass outside the d line.</summary>
    const string Agf = """
        CC Made-up glasses for tests
        NM TK1 2 - 0 0 0 1 0
        CD 1.04 0.006 0.23 0.02 1.01 103.5 0 0 0 0
        OD 1.5 1 2 1 1 0.3 0
        LD 0.35 2.5
        NM TK2 2 - 0 0 1 2 0
        CD 1.35 0.0098 0.21 0.0475 0.87 109 0 0 0 0
        ED 8.2 9.7 3.6 0 0
        LD 0.37 2.5
        NM TIR 2 - 2.4 0 0 0 0
        CD 4 0.15 0.1 0.4 0 1 0 0 0 0
        LD 2 12
        """;

    string WriteAgf(string name, string text = Agf)
    {
        var path = Path.Combine(_dir, name + ".agf");
        File.WriteAllText(path, text.Replace("\n", "\r\n"));
        return path;
    }

    static GlassRow Row(string catalog, string name, double? nd, GlassStatus status, bool excluded = false, double? density = null) => new()
    {
        Catalog = catalog, Name = name, Nd = nd, Vd = nd is null ? null : 50, Status = status, ExcludeSubstitution = excluded, Density = density,
        MinUm = 0.4, MaxUm = 2,
    };

    [Fact]
    public void VisibleRowsSearchFilterAndSort()
    {
        GlassRow[] rows =
        [
            Row("CAT", "G10", 1.7, GlassStatus.Standard, density: 3),
            Row("CAT", "G2", 1.5, GlassStatus.Preferred),
            Row("OTHER", "X1", null, GlassStatus.Obsolete, density: 2),
            Row("OTHER", "S1", 1.6, GlassStatus.Special, excluded: true),
        ];
        string[] Names(IEnumerable<GlassRow> r) => r.Select(x => x.Name).ToArray();
        Assert.Equal(["G2", "G10", "S1", "X1"], Names(GlassBrowserModel.Visible(rows, "", GlassFilter.All, true, GlassColumn.Name, true)));
        Assert.Equal(["X1", "S1", "G10", "G2"], Names(GlassBrowserModel.Visible(rows, "", GlassFilter.All, true, GlassColumn.Name, false)));
        // A missing value sorts last ascending.
        Assert.Equal(["G2", "S1", "G10", "X1"], Names(GlassBrowserModel.Visible(rows, "", GlassFilter.All, true, GlassColumn.Nd, true)));
        Assert.Equal(["X1", "G10", "G2", "S1"], Names(GlassBrowserModel.Visible(rows, "", GlassFilter.All, true, GlassColumn.Density, true)));
        Assert.Equal(["G2", "G10", "S1", "X1"], Names(GlassBrowserModel.Visible(rows, "", GlassFilter.All, true, GlassColumn.Status, true)));
        // The catalog matches only under All Glasses.
        Assert.Equal(["S1", "X1"], Names(GlassBrowserModel.Visible(rows, "other", GlassFilter.All, true, GlassColumn.Name, true)));
        Assert.Empty(GlassBrowserModel.Visible(rows, "other", GlassFilter.All, false, GlassColumn.Name, true));
        Assert.Equal(["G2", "G10"], Names(GlassBrowserModel.Visible(rows, " g ", GlassFilter.All, true, GlassColumn.Name, true)));
        Assert.Equal(["G2", "G10", "S1"], Names(GlassBrowserModel.Visible(rows, "", GlassFilter.Available, true, GlassColumn.Name, true)));
        Assert.Equal(["G2"], Names(GlassBrowserModel.Visible(rows, "", GlassFilter.Preferred, true, GlassColumn.Name, true)));
        Assert.Equal(["G2", "G10"], Names(GlassBrowserModel.Visible(rows, "", GlassFilter.Substitution, true, GlassColumn.Name, true)));
    }

    [Fact]
    public void NumbersUseTheAppFormat()
    {
        Assert.Equal("−0.0040", GlassFormat.Signed(-0.004));
        Assert.Equal("+0.0000", GlassFormat.Signed(-0.00001));
        Assert.Equal("0.00", GlassFormat.Fixed(-0.001, 2));
        Assert.Equal("1.51680", GlassFormat.Fixed(1.5168, 5));
        Assert.Equal("—", GlassFormat.Fixed(null, 2));
        Assert.Equal("—", GlassFormat.Fixed(double.NaN, 2));
        var ir = Row("C", "IR", null, GlassStatus.Standard) with { CatalogNd = 2.4, MinUm = 2, MaxUm = 12 };
        Assert.Equal("—", GlassFormat.Nd(ir));
        Assert.Contains("nd 2.40000", GlassFormat.IrTip(ir));
        Assert.Null(GlassFormat.IrTip(Row("C", "G", 1.5, GlassStatus.Standard)));
        Assert.Equal("2.00–12.00", GlassFormat.Range(ir));
        var change = new CatalogChanges.Change("G", 1.5, 1.50012, 60, 60, ["nd", "density"]);
        Assert.Equal("nd 1.50000 → 1.50012; density", GlassFormat.Describe(change));
    }

    [Fact]
    public void EditorSpecFollowsTheKindAndResizesCoefficients()
    {
        var lib = new GlassLibrary(Path.Combine(_dir, "lib"));
        lib.CreateUserCatalog("Mine");
        var e = GlassEditorModel.New("Mine", lib);
        Assert.Equal("NEWGLASS", e.Name);
        Assert.Equal(1.5168, e.Spec()["model"]!["nd"]!.GetValue<double>());
        e.Dispersion = GlassEditorModel.Kind.Formula;
        e.Formula = 5;
        Assert.Equal(3, e.Coefficients.Length);
        Assert.Equal(3, e.Spec()["coefficients"]!.AsArray().Count);
        Assert.Null(e.Spec()["model"]);
        e.MaxUm = 2;
        Assert.Equal(2, e.Spec()["max_um"]!.GetValue<double>());
        Assert.Equal(13, GlassEditorModel.FormulaNames.Count);
        Assert.All(Enumerable.Range(1, 13), f => Assert.NotEmpty(GlassEditorModel.CoefficientNames(f)));
    }

    [Fact]
    public async Task LibrarySyncLoadsFilesUserGlassesAndOverrides()
    {
        using var engine = SharedEngine().NewSession();
        var lib = new GlassLibrary(Path.Combine(_dir, "lib"));
        lib.Add(WriteAgf("TESTCAT"));
        lib.CreateUserCatalog("Mine");
        var editor = GlassEditorModel.New("Mine", lib);
        editor.Name = "MYGLASS";
        (editor.Nd, editor.Vd) = (1.6, 40);
        Assert.True(editor.Save(engine));
        Assert.Equal(1.6, editor.PreviewNumbers!.Value.Nd, 6);
        lib.SetStatus(GlassStatus.Obsolete, "TESTCAT", "TK1");

        Assert.True(await GlassLibrarySync.SyncAsync(engine, lib));
        Assert.Empty(GlassLibrarySync.Warnings(engine));
        Assert.False(await GlassLibrarySync.SyncAsync(engine, lib));

        var catalogs = await engine.CatalogsAsync();
        Assert.Contains(catalogs, c => c.Name == "TESTCAT" && c.Library && c.GlassCount == 3);
        Assert.Contains(catalogs, c => c.Name == "Mine" && c.Library && c.GlassCount == 1);
        var rows = await engine.GlassRowsAsync("TESTCAT");
        var tk1 = rows.Single(r => r.Name == "TK1");
        Assert.Equal(GlassStatus.Obsolete, tk1.Status);
        Assert.True(tk1.Overridden);
        Assert.True(rows.Single(r => r.Name == "TK2").ExcludeSubstitution);
        var ir = rows.Single(r => r.Name == "TIR");
        // Engines with IR support leave n_d out and list the catalog's value; older ones give the listed value as n_d.
        Assert.Equal(2.4, ir.Nd ?? ir.CatalogNd);
        var mine = Assert.Single(await engine.GlassRowsAsync("Mine"));
        Assert.Equal(1.6, mine.Nd!.Value, 5);

        // The browser on its own session: catalogs by group, scope and selection with details.
        var browser = new GlassBrowserModel(lib, () => engine);
        await browser.RefreshAsync();
        Assert.Contains(browser.LibraryCatalogs, c => c.Name == "TESTCAT");
        Assert.Contains(browser.UserCatalogs, c => c.Name == "Mine");
        Assert.Contains(browser.BuiltinCatalogs, c => c.IsBuiltin);
        Assert.Equal("Mine", browser.DefaultUserCatalog);

        // A status cleared in the library reaches the session at the next sync.
        lib.ClearOverride("TESTCAT", "TK1");
        Assert.True(await GlassLibrarySync.SyncAsync(engine, lib));
        Assert.False((await engine.GlassRowsAsync("TESTCAT")).Single(r => r.Name == "TK1").Overridden);

        // AGF export of the user catalog lists nothing it could not write.
        var export = engine.Agf(lib.UserCatalogNamed("Mine")!.Json);
        Assert.Contains("NM MYGLASS", export.Text);

        // Removing the file from the library drops its catalog.
        lib.Remove(lib.Entries.Single());
        Assert.True(await GlassLibrarySync.SyncAsync(engine, lib));
        Assert.DoesNotContain(await engine.CatalogsAsync(), c => c.Name == "TESTCAT");
    }

    [Fact]
    public async Task EditorReportsWhatTheEngineRefuses()
    {
        using var engine = SharedEngine().NewSession();
        var lib = new GlassLibrary(Path.Combine(_dir, "lib"));
        lib.CreateUserCatalog("Mine");
        var e = GlassEditorModel.New("Mine", lib);
        e.Name = "two words";
        Assert.Null(e.Check(engine));
        Assert.Contains("spaces", e.Problem);
        e.Name = "OK";
        e.Nd = 0.9;
        Assert.Null(e.Check(engine));
        Assert.NotNull(e.Problem);
        e.Nd = double.NaN;
        Assert.Null(e.Check(engine));
        Assert.False(e.Save(engine));
        Assert.Empty(lib.UserCatalogNamed("Mine")!.Glasses);
        await Task.CompletedTask;
    }

    [Fact]
    public async Task ReloadReportsWhatChanged()
    {
        using var engine = SharedEngine().NewSession();
        var lib = new GlassLibrary(Path.Combine(_dir, "lib"));
        var path = WriteAgf("RELOADME");
        lib.Add(path);
        await GlassLibrarySync.SyncAsync(engine, lib);
        var browser = new GlassBrowserModel(lib, () => engine);
        await browser.RefreshAsync();
        WriteAgf("RELOADME", Agf.Replace("CD 1.04 0.006", "CD 1.05 0.006").Replace("NM TIR", "NM TNEW"));
        var changes = await browser.ReloadAsync(browser.Catalog("RELOADME")!);
        Assert.NotNull(changes);
        Assert.Equal(["TNEW"], changes.Added);
        Assert.Equal(["TIR"], changes.Removed);
        Assert.Equal("TK1", Assert.Single(changes.Changed).Name);
    }
}
