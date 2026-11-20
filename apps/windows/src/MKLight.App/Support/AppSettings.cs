using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.UI.Xaml;

namespace MKLight.App.Support;

/// <summary>
/// App-wide preferences and the recent files, in %LOCALAPPDATA%\MKLight
/// (the app is unpackaged, so no ApplicationData). Never in projects.
/// </summary>
public static class AppSettings
{
    public static string Directory { get; } = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "MKLight");
    static string FilePath => Path.Combine(Directory, "settings.json");

    static JsonObject Load()
    {
        try { return JsonNode.Parse(File.ReadAllText(FilePath)) as JsonObject ?? []; }
        catch (Exception) { return []; }
    }

    static void Save(JsonObject o)
    {
        try
        {
            System.IO.Directory.CreateDirectory(Directory);
            File.WriteAllText(FilePath, o.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        }
        catch (Exception) { /* preferences are best effort */ }
    }

    /// <summary>A launch option's theme overrides the saved one for this run only.</summary>
    public static ElementTheme? Override { get; set; }

    public static ElementTheme Theme
    {
        get => Override ?? (Load()["theme"]?.GetValue<string>() switch { "light" => ElementTheme.Light, "dark" => ElementTheme.Dark, _ => ElementTheme.Default });
        set
        {
            Override = null;
            var o = Load();
            o["theme"] = value switch { ElementTheme.Light => "light", ElementTheme.Dark => "dark", _ => "system" };
            Save(o);
        }
    }

    public static List<string> Recent
    {
        get => (Load()["recent"] as JsonArray ?? []).Select(n => n?.GetValue<string>()).OfType<string>().ToList();
        set
        {
            var o = Load();
            o["recent"] = new JsonArray(value.Select(v => (JsonNode)v).ToArray());
            Save(o);
        }
    }
}

/// <summary>The ten most recently opened or saved designs.</summary>
public static class RecentFiles
{
    public static IReadOnlyList<string> List => AppSettings.Recent.Where(File.Exists).ToList();

    public static void Add(string path)
    {
        var list = AppSettings.Recent;
        list.RemoveAll(p => string.Equals(p, path, StringComparison.OrdinalIgnoreCase));
        list.Insert(0, path);
        AppSettings.Recent = list.Take(10).ToList();
    }
}
