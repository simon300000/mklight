using System.Text.Json.Nodes;
using MKLight.Engine;

namespace MKLight.Tests.Engine;

/// <summary>Repository paths, sample systems and the shared engine of the engine tests.</summary>
static class EngineTestSupport
{
    public const string MissingLibrary =
        "mklight_ffi.dll not found: run scripts/build-windows.ps1 -EngineOnly, then dotnet test";

    /// <summary>The repository root: the first ancestor of the test binaries with a Cargo.toml.</summary>
    public static readonly string RepoRoot = FindRepoRoot();

    static string FindRepoRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "Cargo.toml"))) return dir.FullName;
        }
        throw new InvalidOperationException("no Cargo.toml above " + AppContext.BaseDirectory);
    }

    public static string RepoPath(params string[] parts) => Path.Combine([RepoRoot, .. parts]);

    /// <summary>The engine's test fixtures (shared with the macOS tests), read in place.</summary>
    public static string Fixture(string name) => RepoPath("apps", "macos", "Tests", "MKLightModelTests", "Fixtures", name);

    static readonly Lazy<Dictionary<string, JsonObject>> samples = new(() =>
        (JsonNode.Parse(File.ReadAllText(RepoPath("apps", "macos", "Sources", "MKLightModel", "Resources", "samples.json")))!.AsArray())
        .ToDictionary(s => s!["id"]!.GetValue<string>(), s => s!["system"]!.AsObject()));

    public static IReadOnlyList<string> SampleIds => samples.Value.Keys.ToArray();

    /// <summary>A bundled sample system as a mutable JSON object (a copy).</summary>
    public static JsonObject Sample(string id) => (JsonObject)samples.Value[id].DeepClone();

    public static string SampleJson(string id) => Sample(id).ToJsonString();

    /// <summary>Skips the calling test, with the build hint, when the engine library is not there.</summary>
    public static void RequireEngine() => Assert.SkipUnless(FFIEngine.IsAvailable, MissingLibrary);

    static readonly Lazy<FFIEngine> shared = new(() =>
        new FFIEngine(new JsonObject { ["worker_threads"] = 2, ["progress_interval_ms"] = 10 }));

    /// <summary>One engine for the tests (two workers, fast progress polling); skips without the library.</summary>
    public static FFIEngine SharedEngine()
    {
        RequireEngine();
        return shared.Value;
    }

    public static Dictionary<string, JsonNode?> Settings(params (string Key, JsonNode? Value)[] pairs) =>
        pairs.ToDictionary(p => p.Key, p => p.Value);

    public static Dictionary<string, JsonNode?> Settings((string Key, JsonNode? Value) pair) => Settings([pair]);
}
