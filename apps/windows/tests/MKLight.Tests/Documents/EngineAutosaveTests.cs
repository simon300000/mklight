using MKLight.App.Windows;
using MKLight.Model;
using static MKLight.Tests.Engine.EngineTestSupport;

namespace MKLight.Tests.Documents;

/// <summary>
/// Autosave and recovery through the engine (ABI 1.7, against the real DLL): catalog references survive a
/// catalog that is offline when the design is open, and a project folder that moves before recovery.
/// </summary>
[Collection("FFI engine")]
public sealed class EngineAutosaveTests : IDisposable
{
    readonly string _dir = Path.Combine(Path.GetTempPath(), "mklight-autosave-tests-" + Guid.NewGuid().ToString("N"));

    public EngineAutosaveTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    const string Agf = "CC test\r\nNM CROWN 2 517642 1.516800 64.17 0 1 0\r\n"
        + "CD 1.03961212 0.00600069867 0.231792344 0.0200179144 1.01046945 103.560653 0 0 0 0\r\nLD 0.3 2.5\r\n";

    static string WriteAgf(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, Agf);
        return path;
    }

    [Fact]
    public void ACatalogOfflineWhileTheDesignIsOpenKeepsItsReference()
    {
        // The project references LOCAL (present) and MY on a share that is offline now.
        var project = Path.Combine(_dir, "proj");
        var local = WriteAgf(Path.Combine(project, "glass", "LOCAL.agf"));
        var offline = Path.Combine(_dir, "share", "MY.agf");
        var file = Path.Combine(project, "lens.mklight");
        var doc = new ProjectFile(SampleSystems.Singlet()).WithCatalogs(
        [
            new CatalogReference("LOCAL", local, "glass/LOCAL.agf"),
            new CatalogReference("MY", offline, "../share/MY.agf"),
        ]);
        doc.Write(file);

        // Autosave with the share offline: both references are written, as given.
        var autosave = file + Autosaves.Suffix;
        Autosaves.Write(SharedEngine(), doc, autosave, file);
        var (recovered, warnings) = Autosaves.Read(SharedEngine(), autosave);
        Assert.Equal(["LOCAL", "MY"], recovered.Catalogs.Select(c => c.Name));
        Assert.Equal("../share/MY.agf", recovered.Catalogs[1].RelativePath);
        Assert.Equal(offline, recovered.Catalogs[1].Path);
        Assert.Contains(warnings, w => w.Contains("MY", StringComparison.Ordinal));
        Assert.DoesNotContain(warnings, w => w.Contains("LOCAL", StringComparison.Ordinal));

        // The recovered design, autosaved again, still holds MY; once the share is back it loads.
        Autosaves.Write(SharedEngine(), recovered, autosave, file);
        WriteAgf(offline);
        var (again, after) = Autosaves.Read(SharedEngine(), autosave);
        Assert.Equal(["LOCAL", "MY"], again.Catalogs.Select(c => c.Name));
        Assert.Empty(after);
    }

    [Fact]
    public void AProjectFolderThatMovesBeforeRecoveryStillFindsItsCatalog()
    {
        // A catalog added in the app has an absolute path only; the autosave writes it relative to the design's folder.
        var before = Path.Combine(_dir, "before");
        var catalog = WriteAgf(Path.Combine(before, "glass", "MINE.agf"));
        var file = Path.Combine(before, "lens.mklight");
        var doc = new ProjectFile(SampleSystems.Singlet()).AddCatalog(new CatalogReference("MINE", catalog));
        doc.Write(file);
        Autosaves.Write(SharedEngine(), doc, file + Autosaves.Suffix, file);

        // The folder moves (a renamed synced folder, another drive letter) before recovery.
        var after = Path.Combine(_dir, "after");
        Directory.Move(before, after);
        var (recovered, warnings) = Autosaves.Read(SharedEngine(), Path.Combine(after, "lens.mklight" + Autosaves.Suffix));
        Assert.Empty(warnings);
        var mine = Assert.Single(recovered.Catalogs);
        Assert.Equal("glass/MINE.agf", mine.RelativePath);
    }

    [Fact]
    public void SaveWritesRelativePathsForTheTargetFolder()
    {
        var dir = Path.Combine(_dir, "save");
        var inside = WriteAgf(Path.Combine(dir, "glass", "IN.agf"));
        var outside = WriteAgf(Path.Combine(_dir, "elsewhere", "OUT.agf"));
        var gone = Path.Combine(_dir, "gone", "GONE.agf");
        var p = new ProjectFile(SampleSystems.Singlet()).WithCatalogs(
        [
            new CatalogReference("IN", inside, null),
            new CatalogReference("OUT", outside, "stale/OUT.agf"),
            new CatalogReference("GONE", gone, "../gone/GONE.agf"),
        ]).WithRelativeCatalogPaths(dir);
        // Under the folder: relative; a usable file elsewhere: none; unusable: the old one kept.
        Assert.Equal(["glass/IN.agf", null, "../gone/GONE.agf"], p.Catalogs.Select(c => c.RelativePath));
    }
}
