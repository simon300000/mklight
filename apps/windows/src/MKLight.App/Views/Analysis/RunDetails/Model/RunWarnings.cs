using System.Globalization;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using MKLight.App.Support;
using MKLight.App.Views.Analysis.Results.Readers;
using MKLight.Model;

namespace MKLight.App.Views.Analysis.RunDetails;

/// <summary>What kind of remedy a warning has, when the app can offer one.</summary>
public enum WarningRemedy
{
    None,
    /// <summary>Finer pupil sampling (<c>pupil.grid</c>).</summary>
    PupilSampling,
    /// <summary>A larger image grid around the PSF.</summary>
    ImageGrid,
}

/// <summary>
/// One message of the engine's run warnings, for several fields or
/// wavelengths: the first occurrence's text, capitalised, without its scope,
/// and the scopes it was given for (<c>Field 1</c>, <c>Field 3 · λ2</c>;
/// empty: the whole run).
/// </summary>
public sealed record WarningGroup(string Message, IReadOnlyList<string> Scopes, WarningRemedy Remedy)
{
    /// <summary>One line for tooltips and screen readers: "Field 1, Field 3: message" (the message is engine text).</summary>
    public string Line => Scopes.Count == 0 ? EngineText.T(Message) : Loc.F("Analysis_RunDetails_ScopedWarning", string.Join(", ", Scopes), EngineText.T(Message));
}

/// <summary>
/// The engine's run warnings grouped for reading: the same message for
/// several fields or wavelengths is shown once with its scopes. Engine field
/// and wavelength indices are 0-based; the app numbers them from 1, as everywhere else.
/// </summary>
public static partial class RunWarnings
{
    /// <summary>Groups <paramref name="warnings"/> by their message with the numbers masked, in first-seen order.</summary>
    public static IReadOnlyList<WarningGroup> Group(IEnumerable<string> warnings)
    {
        var order = new List<string>();
        var groups = new Dictionary<string, (string Message, List<string> Scopes, WarningRemedy Remedy)>();
        foreach (var w in warnings)
        {
            var (scope, message) = Split(w);
            var key = Mask(message);
            if (groups.TryGetValue(key, out var g))
            {
                if (scope is not null && !g.Scopes.Contains(scope)) g.Scopes.Add(scope);
            }
            else
            {
                order.Add(key);
                groups[key] = (Capitalised(message), scope is null ? [] : [scope], RemedyOf(message));
            }
        }
        return order.Select(k => new WarningGroup(groups[k].Message, groups[k].Scopes, groups[k].Remedy)).ToList();
    }

    /// <summary>A short summary: the first group's line, and how many more.</summary>
    public static string Summary(IEnumerable<string> warnings)
    {
        var g = Group(warnings);
        if (g.Count == 0) return "";
        return g.Count > 1 ? Loc.F("Analysis_RunDetails_WarningSummary", g[0].Line, g.Count - 1) : g[0].Line;
    }

    /// <summary><c>field 2, wavelength 1: text</c> → (<c>Field 3 · λ2</c>, <c>text</c>); also <c>field (1, -1): text</c>.</summary>
    public static (string? Scope, string Message) Split(string w)
    {
        var t = w.Trim();
        var colon = t.IndexOf(": ", StringComparison.Ordinal);
        if (colon < 0) return (null, t);
        string head = t[..colon], rest = t[(colon + 2)..];
        if (head.StartsWith("field (", StringComparison.Ordinal)) return (Loc.F("Analysis_RunDetails_FieldPoint", head["field ".Length..].Replace('-', '−')), rest);
        var scope = new List<string>();
        foreach (var p in head.Split(',').Select(p => p.Trim()))
        {
            var words = p.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (words.Length != 2 || !int.TryParse(words[1], NumberStyles.None, CultureInfo.InvariantCulture, out var k)) return (null, t);
            switch (words[0])
            {
                case "field": scope.Add(Loc.F("Analysis_RunDetails_Field", k + 1)); break;
                case "wavelength": scope.Add("λ" + (k + 1).ToString(CultureInfo.InvariantCulture)); break;
                case "surface": scope.Add(Loc.F("Analysis_RunDetails_Surface", k)); break;
                default: return (null, t);
            }
        }
        return scope.Count == 0 ? (null, t) : (string.Join(" · ", scope), rest);
    }

    /// <summary>The message with its numbers replaced, so "0.58 waves" and "0.72 waves" group together.</summary>
    public static string Mask(string s) => Number().Replace(s, "#");

    [GeneratedRegex(@"[-+]?[0-9]*\.?[0-9]+(e[-+]?[0-9]+)?")]
    private static partial Regex Number();

    static string Capitalised(string s) => s.Length == 0 ? s : char.ToUpperInvariant(s[0]) + s[1..];

    static WarningRemedy RemedyOf(string s)
    {
        var l = s.ToLowerInvariant();
        if (l.Contains("increase the pupil sampling")) return WarningRemedy.PupilSampling;
        if (l.Contains("enlarge the grid")) return WarningRemedy.ImageGrid;
        return WarningRemedy.None;
    }
}

/// <summary>
/// A setting change that addresses a warning: an explicit coarse pupil
/// grid, image grid or PSF window goes back to the engine's automatic
/// choice, which sizes it from the system. Automatic settings that still
/// warn reached the engine's cap: nothing to offer.
/// </summary>
public sealed record RunRemedy(string Title, string Help, string Key, JsonNode Value)
{
    /// <param name="value">The pane's value of a top-level setting (its own, else the default); null when it has none.</param>
    public static RunRemedy? For(WarningRemedy kind, Func<string, JsonNode?> value)
    {
        string I(int n) => n.ToString(CultureInfo.InvariantCulture);
        // Image simulation keeps its pupil and PSF window in `optics`.
        var optics = value("pupil") is null && value("grid_size") is null ? value("optics")?.DeepClone() as JsonObject : null;
        switch (kind)
        {
            case WarningRemedy.PupilSampling:
                if ((optics?["pupil"] ?? value("pupil"))?.DeepClone() is not JsonObject pupil || Json.Int(pupil["grid"]) is not int grid || grid <= 0) return null;
                pupil["grid"] = 0;
                var title = Loc.F("Analysis_Remedy_PupilTitle", I(grid));
                var help = Loc.F("Analysis_Remedy_PupilHelp", I(grid));
                if (optics is not null)
                {
                    optics["pupil"] = pupil;
                    return new(title, help, "optics", optics);
                }
                return new(title, help, "pupil", pupil);
            case WarningRemedy.ImageGrid:
                if (optics is not null)
                {
                    if (Json.Int(optics["psf_size"]) is not int window || window <= 0) return null;
                    optics["psf_size"] = 0;
                    return new(Loc.F("Analysis_Remedy_PsfWindowTitle", I(window)), Loc.F("Analysis_Remedy_PsfWindowHelp", I(window)),
                        "optics", optics);
                }
                if (Json.Int(value("grid_size")) is not int size || size <= 0) return null;
                return new(Loc.F("Analysis_Remedy_ImageGridTitle", I(size)), Loc.F("Analysis_Remedy_ImageGridHelp", I(size)),
                    "grid_size", 0);
            default:
                return null;
        }
    }
}
