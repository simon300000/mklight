using System.Text.Json.Nodes;
using MKLight.Model;

namespace MKLight.Tests.Model;

/// <summary>
/// The engine-written fixtures, read in place from the macOS test folder (never copied).
/// Decoding them and re-encoding must reproduce the same JSON tree, which proves the C# mirror matches
/// the engine's serde representation field for field.
/// </summary>
internal static class Fixtures
{
    private static readonly string Directory = Find();

    private static string Find()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "Cargo.toml"))) dir = dir.Parent;
        if (dir is null) throw new DirectoryNotFoundException("repository root (Cargo.toml) not found above " + AppContext.BaseDirectory);
        return Path.Combine(dir.FullName, "apps", "macos", "Tests", "MKLightModelTests", "Fixtures");
    }

    public static string Text(string name) => File.ReadAllText(Path.Combine(Directory, name + ".json"));

    public static JsonNode Tree(string name) => JsonNode.Parse(Text(name))!;

    public static OpticalSystem Load(string name) => ModelJson.Decode<OpticalSystem>(Text(name));

    /// <summary>Decode → encode → compare semantically with the original.</summary>
    public static T RoundTrip<T>(string name, string? text = null) where T : IJsonModel<T>
    {
        var original = text ?? Text(name);
        var value = ModelJson.Decode<T>(original);
        var reencoded = ModelJson.Encode(value);
        var diff = JsonTree.Parse(original).Differences(JsonTree.Parse(reencoded));
        Assert.True(diff.Count == 0, $"{name} differs after round trip:\n{string.Join("\n", diff)}");
        // A second decode gives back the same value.
        Assert.Equal(value, ModelJson.Decode<T>(reencoded));
        return value;
    }

    public static ValueList<T> RoundTripList<T>(string name, string? text = null) where T : IJsonModel<T>
    {
        var original = text ?? Text(name);
        var value = ModelJson.DecodeList<T>(original);
        var reencoded = ModelJson.EncodeList(value);
        var diff = JsonTree.Parse(original).Differences(JsonTree.Parse(reencoded));
        Assert.True(diff.Count == 0, $"{name} differs after round trip:\n{string.Join("\n", diff)}");
        Assert.Equal(value, ModelJson.DecodeList<T>(reencoded));
        return value;
    }
}

internal static class Sid
{
    public static SurfaceID Of(ulong v) => new(v);
}
