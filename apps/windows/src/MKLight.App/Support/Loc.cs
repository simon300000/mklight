using System.Globalization;
using System.Xml.Linq;

namespace MKLight.App.Support;

/// <summary>
/// The app's own UI text (docs/WINDOWS.md "Languages"): <c>Strings/&lt;language&gt;/Resources.resw</c>, English
/// the source. <see cref="S"/> looks a key up, <see cref="F"/> fills a format string's <c>{0}</c> placeholders
/// with invariant number formatting. In the app the lookup is MRT Core (<see cref="Provider"/>, set at launch);
/// without one (unit tests, which link this file) it reads the English <c>Resources.resw</c> beside the
/// assembly. A key that is not found shows as itself, so a missing string is visible rather than blank.
/// </summary>
public static class Loc
{
    /// <summary>The lookup for the chosen language (the app sets MRT Core's); null reads the English resw.</summary>
    public static Func<string, string?>? Provider { get; set; }

    static Dictionary<string, string>? _english;

    public static string S(string key) => Provider?.Invoke(key) is { Length: > 0 } s ? s : English(key) ?? key;

    public static string F(string key, params object?[] args) => string.Format(CultureInfo.InvariantCulture, S(key), args);

    /// <summary>The English text of a key (the source language), from the resw beside the assembly.</summary>
    public static string? English(string key)
    {
        _english ??= ReadResw(Path.Combine(AppContext.BaseDirectory, "Strings", "en-US", "Resources.resw"));
        return _english.TryGetValue(key, out var v) ? v : null;
    }

    /// <summary>The name/value pairs of a <c>.resw</c> file (empty when it is missing).</summary>
    public static Dictionary<string, string> ReadResw(string path)
    {
        var d = new Dictionary<string, string>(StringComparer.Ordinal);
        if (!File.Exists(path)) return d;
        foreach (var e in XDocument.Load(path).Root!.Elements("data"))
            if (e.Attribute("name")?.Value is { } name) d[name] = e.Element("value")?.Value ?? "";
        return d;
    }
}
