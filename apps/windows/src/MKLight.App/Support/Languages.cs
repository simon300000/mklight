namespace MKLight.App.Support;

/// <summary>
/// The languages the app ships and how one is chosen (docs/WINDOWS.md "Languages"): the <c>--language</c>
/// launch option, else the saved choice (<c>language</c> in settings.json), else the system's. English is the
/// source; Simplified Chinese is chosen for zh, zh-CN, zh-SG and zh-Hans*; Traditional Chinese (zh-TW, zh-HK,
/// zh-Hant*) stays English for now.
/// </summary>
public static class Languages
{
    public const string English = "en-US";
    public const string SimplifiedChinese = "zh-Hans";
    /// <summary>The saved choice that follows the system.</summary>
    public const string System = "system";

    public static readonly string[] Supported = [English, SimplifiedChinese];

    /// <summary>The language to show for an explicit choice (a launch option or saved setting) and the system's preferred languages, in order.</summary>
    public static string Resolve(string? choice, IEnumerable<string> systemLanguages)
    {
        if (Normalize(choice) is { } chosen) return chosen;
        foreach (var tag in systemLanguages)
            if (FromSystem(tag) is { } l) return l;
        return English;
    }

    /// <summary>A supported language for a choice (<c>en</c>, <c>en-US</c>, <c>zh-Hans</c> …), null for <c>system</c> or nothing.</summary>
    public static string? Normalize(string? choice) => choice?.Trim().ToLowerInvariant() switch
    {
        null or "" or System => null,
        "en" or "en-us" => English,
        "zh-hans" or "zh-cn" => SimplifiedChinese,
        var other => throw new ArgumentException($"unknown language '{other}': use {English}, {SimplifiedChinese} or {System}"),
    };

    /// <summary>A system language tag mapped to a shipped language, null when it does not match one (try the next).</summary>
    public static string? FromSystem(string tag)
    {
        var t = tag.Trim().ToLowerInvariant().Replace('_', '-');
        if (t == "en" || t.StartsWith("en-", StringComparison.Ordinal)) return English;
        if (t == "zh" || t is "zh-cn" or "zh-sg" || t.StartsWith("zh-hans", StringComparison.Ordinal)) return SimplifiedChinese;
        if (t.StartsWith("zh-", StringComparison.Ordinal)) return English; // zh-TW, zh-HK, zh-Hant*: English for now
        return null;
    }
}
