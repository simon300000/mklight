using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace MKLight.Model;

/// <summary>
/// Text that comes from the engine (analysis titles and descriptions, settings descriptions and enum values,
/// operand names, diagnostics, sample names), translated through the shared catalog
/// <c>locales/&lt;language&gt;/engine.json</c> (docs/I18N.md):
/// <code>{ "strings": { "&lt;exact English&gt;": "…" }, "enums": { "&lt;raw enum value&gt;": "…" }, "diagnostics": { "&lt;code&gt;": "&lt;template with {surface} or {path}&gt;" } }</code>
/// A lookup is an exact match; a miss shows the English (for an enum, the app's own formatting). Text is
/// translated only where it is shown: what the app parses, matches or writes back stays the engine's English.
/// English (no catalog) returns the text unchanged.
/// </summary>
public sealed partial class EngineText
{
    readonly Dictionary<string, string> _strings;
    readonly Dictionary<string, string> _diagnostics;
    readonly Dictionary<string, string> _enums;

    public EngineText(IReadOnlyDictionary<string, string> strings, IReadOnlyDictionary<string, string> diagnostics,
        IReadOnlyDictionary<string, string>? enums = null)
    {
        _strings = new(strings, StringComparer.Ordinal);
        _diagnostics = new(diagnostics, StringComparer.Ordinal);
        _enums = enums is null ? new(StringComparer.Ordinal) : new(enums, StringComparer.Ordinal);
    }

    /// <summary>No translations: English as the engine writes it.</summary>
    public static EngineText None { get; } = new(new Dictionary<string, string>(), new Dictionary<string, string>());

    /// <summary>The catalog the app shows (set at launch for the chosen language).</summary>
    public static EngineText Current { get; set; } = None;

    public int Count => _strings.Count + _diagnostics.Count + _enums.Count;

    /// <summary>The label of a raw settings enum value (<c>hexapolar</c>), null on a miss (the app formats the value itself).</summary>
    public string? EnumLabel(string? value) => value is not null && _enums.TryGetValue(value, out var t) && t.Length > 0 ? t : null;

    /// <summary>The translation of an exact English string, else the string itself (null stays null).</summary>
    public string? Translate(string? english) =>
        english is null ? null : _strings.TryGetValue(english, out var t) && t.Length > 0 ? t : english;

    /// <summary>A diagnostic's text: the code's template with its placeholders filled from <paramref name="args"/>,
    /// else the English message (also when a placeholder has no value).</summary>
    public string TranslateDiagnostic(string? code, string englishMessage, IReadOnlyDictionary<string, string>? args = null)
    {
        if (code is null || !_diagnostics.TryGetValue(code, out var template) || template.Length == 0) return englishMessage;
        var missing = false;
        var text = Placeholder().Replace(template, m =>
        {
            if (args is not null && args.TryGetValue(m.Groups[1].Value, out var v)) return v;
            missing = true;
            return m.Value;
        });
        return missing ? englishMessage : text;
    }

    /// <summary>Shorthand for <c>Current.Translate</c>.</summary>
    public static string T(string english) => Current.Translate(english)!;

    /// <summary>Shorthand for <c>Current.TranslateDiagnostic</c>.</summary>
    public static string Diagnostic(string? code, string englishMessage, IReadOnlyDictionary<string, string>? args = null) =>
        Current.TranslateDiagnostic(code, englishMessage, args);

    /// <summary>Shorthand for <c>Current.EnumLabel</c>.</summary>
    public static string? Enum(string? value) => Current.EnumLabel(value);

    /// <summary>
    /// A coded issue as shown: the code's template with <c>{surface}</c> (as the app numbers surfaces) and
    /// <c>{path}</c> filled, else the English message; <c>Detail</c> is the English message when the text is a
    /// translation (it holds the values the template leaves out), else null.
    /// </summary>
    public (string Text, string? Detail) Issue(string? code, string englishMessage, int? surface = null, string? path = null)
    {
        var args = new Dictionary<string, string>(StringComparer.Ordinal);
        if (surface is int s) args["surface"] = s.ToString(System.Globalization.CultureInfo.InvariantCulture);
        if (path is not null) args["path"] = path;
        var text = TranslateDiagnostic(code, englishMessage, args);
        return (text, text == englishMessage ? null : englishMessage);
    }

    /// <summary>A structural issue of an invalid model, <c>[code] path: message</c> (also after <c>invalid model: </c>).</summary>
    public static (string Code, string Path, string Message)? ParseStructural(string issue) =>
        Structural().Match(issue) is { Success: true } m ? (m.Groups[1].Value, m.Groups[2].Value, m.Groups[3].Value) : null;

    /// <summary>An invalid model's issue as shown: the template of its code with its path, else the English (no detail).</summary>
    public (string Text, string? Detail) StructuralIssue(string issue) =>
        ParseStructural(issue) is { } p ? Issue(p.Code, issue, path: p.Path) : (issue, null);

    [GeneratedRegex(@"\[([a-z_]+)\] ([^:]+): (.*)")]
    private static partial Regex Structural();

    /// <summary>Reads a catalog file; a missing or unreadable file is <see cref="None"/> (English).</summary>
    public static EngineText Load(string path)
    {
        try
        {
            if (!File.Exists(path)) return None;
            return Parse(File.ReadAllText(path));
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or System.Text.Json.JsonException or FormatException)
        {
            return None;
        }
    }

    public static EngineText Parse(string json)
    {
        var root = JsonNode.Parse(json) as JsonObject ?? throw new FormatException("engine text catalog: not a JSON object");
        return new EngineText(Pairs(root["strings"]), Pairs(root["diagnostics"]), Pairs(root["enums"]));
    }

    static Dictionary<string, string> Pairs(JsonNode? node)
    {
        var d = new Dictionary<string, string>(StringComparer.Ordinal);
        if (node is JsonObject o)
            foreach (var (k, v) in o)
                if (v is JsonValue jv && jv.TryGetValue<string>(out var s)) d[k] = s;
        return d;
    }

    /// <summary>The catalog path for a language beside the app (<c>locales/&lt;language&gt;/engine.json</c>).</summary>
    public static string PathFor(string baseDirectory, string language) => Path.Combine(baseDirectory, "locales", language, "engine.json");

    [GeneratedRegex(@"\{([A-Za-z_][A-Za-z0-9_]*)\}")]
    private static partial Regex Placeholder();
}
