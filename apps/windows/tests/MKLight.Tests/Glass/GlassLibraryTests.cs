using System.Text.Json.Nodes;
using MKLight.App.ViewModels;
using MKLight.Engine;
using static MKLight.Tests.Engine.EngineTestSupport;

namespace MKLight.Tests.Glass;

/// <summary>The app-wide glass library: persistence, the file list designs load, overrides and user catalogs.</summary>
public sealed class GlassLibraryTests : IDisposable
{
    readonly string _dir = Path.Combine(Path.GetTempPath(), "mklight-glass-tests-" + Guid.NewGuid().ToString("N"));

    public GlassLibraryTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    string Lib => Path.Combine(_dir, "support");

    string File(string relative, string text = "")
    {
        var path = Path.GetFullPath(Path.Combine(_dir, relative));
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        System.IO.File.WriteAllText(path, text);
        return path;
    }

    static JsonObject Glass(string name, double nd = 1.5) => new() { ["name"] = name, ["nd"] = nd };

    [Fact]
    public void EntriesAndOverridesPersist()
    {
        var file = File("cats/ONE.agf");
        var folder = Path.Combine(_dir, "folder");
        Directory.CreateDirectory(folder);
        var lib = new GlassLibrary(Lib);
        Assert.True(lib.Add(file));
        Assert.False(lib.Add(file.ToUpperInvariant()));
        Assert.True(lib.Add(folder));
        lib.SetStatus(GlassStatus.Preferred, "ONE", "G1");
        lib.SetExcluded(true, "ONE", "G2");

        var again = new GlassLibrary(Lib);
        Assert.Equal([new GlassLibrary.Entry(file, false), new GlassLibrary.Entry(folder, true)], again.Entries);
        Assert.Equal([new GlassOverride("ONE", "G1", GlassStatus.Preferred), new GlassOverride("ONE", "G2", null, true)], again.Overrides);
        Assert.Empty(again.Problems);
    }

    [Fact]
    public void CatalogFilesFollowTheEntriesWithFolderFilesByNameThenUserCatalogs()
    {
        var single = File("x/SINGLE.agf");
        File("f/A10.agf");
        File("f/a2.AGF");
        File("f/notes.txt");
        File("f/B.json");
        var lib = new GlassLibrary(Lib);
        lib.Add(Path.Combine(_dir, "f"));
        lib.Add(single);
        var user = lib.CreateUserCatalog("Mine");
        var names = lib.CatalogFiles.Select(Path.GetFileName).ToArray();
        Assert.Equal(["a2.AGF", "A10.agf", "B.json", "SINGLE.agf", "Mine.json"], names);
        Assert.Equal(user.FilePath, lib.CatalogFiles[^1]);
    }

    [Fact]
    public void EntryContainingFindsTheFileOrItsFolder()
    {
        var file = File("cats/ONE.agf");
        var inFolder = File("folder/TWO.agf");
        var lib = new GlassLibrary(Lib);
        lib.Add(file);
        lib.Add(Path.GetDirectoryName(inFolder)!);
        Assert.Equal(file, lib.EntryContaining(file)?.Path);
        Assert.True(lib.EntryContaining(inFolder)?.IsFolder);
        Assert.Null(lib.EntryContaining(Path.Combine(_dir, "elsewhere", "THREE.agf")));
        lib.Remove(lib.EntryContaining(file)!);
        Assert.Single(lib.Entries);
    }

    [Fact]
    public void OverridesMergeAndClearAndOnlyChangesCount()
    {
        var lib = new GlassLibrary(Lib);
        var revision = lib.Revision;
        lib.SetStatus(GlassStatus.Obsolete, "CAT", "G");
        lib.SetExcluded(true, "cat", "g");
        Assert.Equal([new GlassOverride("CAT", "G", GlassStatus.Obsolete, true)], lib.Overrides);
        Assert.Equal(revision + 2, lib.Revision);
        lib.SetExcluded(true, "CAT", "G");
        Assert.Equal(revision + 2, lib.Revision);
        lib.SetStatus(null, "CAT", "G");
        Assert.Equal([new GlassOverride("CAT", "G", null, true)], lib.Overrides);
        lib.ClearOverride("CAT", "G");
        Assert.Empty(lib.Overrides);
        Assert.Empty(new GlassLibrary(Lib).Overrides);
    }

    [Fact]
    public void UserCatalogsAreCreatedRenamedAndTrashed()
    {
        var trashed = new List<string>();
        var lib = new GlassLibrary(Lib) { Trash = p => { trashed.Add(p); System.IO.File.Delete(p); } };
        Assert.Equal("User Glasses", lib.FreeCatalogName());
        var c = lib.CreateUserCatalog("  User Glasses ");
        Assert.Equal("User Glasses", c.Name);
        Assert.True(System.IO.File.Exists(c.FilePath));
        Assert.Equal("User Glasses 2", lib.FreeCatalogName());
        Assert.Throws<GlassLibrary.LibraryException>(() => lib.CreateUserCatalog("user glasses"));
        Assert.Throws<GlassLibrary.LibraryException>(() => lib.CreateUserCatalog("builtin"));
        Assert.Throws<GlassLibrary.LibraryException>(() => lib.CreateUserCatalog("a/b"));

        lib.PutGlass(Glass("G1"), "User Glasses");
        lib.SetStatus(GlassStatus.Preferred, "User Glasses", "G1");
        lib.RenameUserCatalog("User Glasses", "Mine");
        Assert.False(System.IO.File.Exists(c.FilePath));
        var mine = Assert.Single(lib.UserCatalogs);
        Assert.Equal("Mine", mine.Name);
        Assert.Equal("Mine", JsonNode.Parse(System.IO.File.ReadAllText(mine.FilePath))!["name"]!.GetValue<string>());
        Assert.True(mine.Glasses.ContainsKey("G1"));

        lib.DeleteUserCatalog("mine");
        Assert.Equal([mine.FilePath], trashed);
        Assert.Empty(lib.UserCatalogs);
        Assert.Empty(new GlassLibrary(Lib).UserCatalogs);
    }

    [Fact]
    public void GlassesArePutRenamedAndDeleted()
    {
        var lib = new GlassLibrary(Lib);
        lib.CreateUserCatalog("Mine");
        lib.PutGlass(Glass("G1"), "Mine");
        lib.PutGlass(Glass("G2"), "Mine");
        Assert.Throws<GlassLibrary.LibraryException>(() => lib.PutGlass(Glass("g1"), "Mine"));
        Assert.Throws<GlassLibrary.LibraryException>(() => lib.PutGlass(Glass("G 3"), "Mine"));
        // Replacing a glass under its own name (any case) or renaming it keeps one entry.
        lib.PutGlass(Glass("g1", 1.6), "Mine", replacing: "G1");
        lib.SetExcluded(true, "Mine", "g1");
        lib.PutGlass(Glass("G9", 1.7), "Mine", replacing: "g1");
        var glasses = new GlassLibrary(Lib).UserCatalogNamed("MINE")!.Glasses;
        Assert.Equal(["G2", "G9"], glasses.Select(kv => kv.Key).Order());
        Assert.Equal([new GlassOverride("Mine", "G9", null, true)], lib.Overrides);

        Assert.Equal("G2_2", lib.FreeGlassName("G2", "Mine"));
        Assert.Equal("N-BK7_copy", lib.FreeGlassName("N-BK7 copy", "Mine"));
        lib.DeleteGlass("g9", "Mine");
        Assert.Empty(lib.Overrides);
        Assert.Throws<GlassLibrary.LibraryException>(() => lib.DeleteGlass("G9", "Mine"));
    }

    [Fact]
    public void UnreadableFilesAreProblemsNotFailures()
    {
        File("support/Glass Library.json", "{ not json");
        File("support/User Glasses/Broken.json", "[1, 2");
        File("support/User Glasses/Named.json", """{"glasses": {}}""");
        var lib = new GlassLibrary(Lib);
        Assert.Equal(2, lib.Problems.Count);
        Assert.Equal("Named", Assert.Single(lib.UserCatalogs).Name);
    }

    [Theory]
    [InlineData("", false)]
    [InlineData("N-BK7", true)]
    [InlineData("two words", false)]
    [InlineData("a:b", false)]
    public void GlassNames(string name, bool ok) => Assert.Equal(ok, GlassLibrary.NameProblem(name, glass: true) is null);
}
