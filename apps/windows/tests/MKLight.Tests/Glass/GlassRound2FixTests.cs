using MKLight.App.ViewModels;
using MKLight.Engine;

namespace MKLight.Tests.Glass;

/// <summary>Sync retry after a failure, catalog rename and file clashes, and stale catalog refreshes.</summary>
public sealed class GlassRound2FixTests : IDisposable
{
    readonly string _dir = Path.Combine(Path.GetTempPath(), "mklight-glass-fix-" + Guid.NewGuid().ToString("N"));

    public GlassRound2FixTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    sealed class Fake(string name) : IEngineClient
    {
        public bool FailSync;
        public int SyncCalls;
        public TaskCompletionSource<IReadOnlyList<CatalogInfo>>? CatalogsGate;
        public EngineInfo Info => new(name, "0", IsMock: true);
        public Task<ModelSnapshot> OpenSnapshotAsync(string systemJson) => Task.FromResult(new ModelSnapshot(systemJson, new("h")));
        public Task<IReadOnlyList<AnalysisDescriptor>> AnalysisDescriptorsAsync() => Task.FromResult<IReadOnlyList<AnalysisDescriptor>>([]);
        public AnalysisJob Submit(EngineAnalysisRequest request, ModelSnapshot snapshot) =>
            AnalysisJob.Run(request, snapshot.Hash, (_, _) => throw EngineException.Unsupported(request.Analysis));
        public Task<IReadOnlyList<string>> MaterialNamesAsync() => Task.FromResult<IReadOnlyList<string>>([]);
        public Task<LibrarySync> SetLibraryCatalogsAsync(IReadOnlyList<string> paths)
        {
            SyncCalls++;
            return FailSync ? throw EngineException.Unsupported("sync") : Task.FromResult(LibrarySync.Empty);
        }
        public Task<IReadOnlyList<CatalogInfo>> CatalogsAsync() =>
            CatalogsGate?.Task ?? Task.FromResult<IReadOnlyList<CatalogInfo>>([new CatalogInfo(name, "user", null, 0)]);
    }

    [Fact]
    public async Task FailedSyncIsTriedAgain()
    {
        var lib = new GlassLibrary(Path.Combine(_dir, "lib"));
        var engine = new Fake("a") { FailSync = true };
        await Assert.ThrowsAsync<EngineException>(() => GlassLibrarySync.SyncAsync(engine, lib));
        engine.FailSync = false;
        await GlassLibrarySync.SyncAsync(engine, lib);
        Assert.Equal(2, engine.SyncCalls);
        await GlassLibrarySync.SyncAsync(engine, lib);
        Assert.Equal(2, engine.SyncCalls);
    }

    [Fact]
    public void RenameMovesOverrides()
    {
        var lib = new GlassLibrary(Path.Combine(_dir, "lib"));
        lib.CreateUserCatalog("Old");
        lib.SetStatus(GlassStatus.Preferred, "Old", "G1");
        lib.SetExcluded(true, "Old", "G2");
        lib.RenameUserCatalog("Old", "New");
        GlassOverride[] expected = [new("New", "G1", GlassStatus.Preferred), new("New", "G2", null, true)];
        Assert.Equal(expected, lib.Overrides);
        Assert.Equal(expected, new GlassLibrary(Path.Combine(_dir, "lib")).Overrides);
    }

    [Fact]
    public void FileOfAnotherCatalogIsNotOverwritten()
    {
        var lib = new GlassLibrary(Path.Combine(_dir, "lib"));
        var a = lib.CreateUserCatalog("A");
        // A file named "B.json" that holds a catalog called "Other".
        var path = Path.Combine(Path.GetDirectoryName(a.FilePath)!, "B.json");
        File.WriteAllText(path, """{"name":"Other","glasses":{}}""");
        lib = new GlassLibrary(Path.Combine(_dir, "lib"));
        Assert.NotNull(lib.UserCatalogNamed("Other"));
        var before = File.ReadAllText(path);
        var create = Assert.Throws<GlassLibrary.LibraryException>(() => lib.CreateUserCatalog("B"));
        Assert.Contains("Other", create.Message);
        var rename = Assert.Throws<GlassLibrary.LibraryException>(() => lib.RenameUserCatalog("A", "B"));
        Assert.Contains("Other", rename.Message);
        Assert.Equal(before, File.ReadAllText(path));
        Assert.NotNull(lib.UserCatalogNamed("A"));
    }

    [Fact]
    public async Task StaleCatalogRefreshDoesNotOverwriteANewerOne()
    {
        var lib = new GlassLibrary(Path.Combine(_dir, "lib"));
        var a = new Fake("A") { CatalogsGate = new(TaskCreationOptions.RunContinuationsAsynchronously) };
        var b = new Fake("B");
        var model = new GlassBrowserModel(lib);
        static GlassBrowserTarget Target(object key, IEngineClient e) =>
            new(key, () => key.ToString()!, e, () => null, (_, _, _) => { }, _ => throw new NotSupportedException(), _ => { }, () => Task.CompletedTask);
        var first = model.AttachAsync(Target("A", a));
        await model.AttachAsync(Target("B", b));
        Assert.Equal(["B"], model.Catalogs.Select(c => c.Name));
        a.CatalogsGate!.SetResult([new CatalogInfo("A", "user", null, 0)]);
        await first;
        Assert.Equal(["B"], model.Catalogs.Select(c => c.Name));
    }
}
