using MKLight.App.Support;
using MKLight.Engine;
using MKLight.Model;

namespace MKLight.App.Windows;

/// <summary>What a group of import warnings is about, most serious first.</summary>
public enum ImportCategory { Problems, Glasses, Surfaces, Dropped, Assumed }

/// <summary>
/// What importing a lens prescription file (<c>.zmx</c>, <c>.seq</c>) left out or changed, grouped by
/// what it means for the design (the Mac app's <c>ImportReport</c>). The design window shows it once,
/// in <see cref="ImportReportDialog"/>, when the imported design opens. Free of WinUI types.
/// </summary>
public sealed class ImportReport
{
    /// <summary>A category's warnings, in the file's order.</summary>
    public sealed record Group(ImportCategory Category, IReadOnlyList<ImportWarning> Warnings);

    /// <param name="lensSurfaces">Surfaces between object and image.</param>
    public ImportReport(string fileName, PrescriptionFormat format, string encoding, int lensSurfaces, IReadOnlyList<ImportWarning> warnings)
    {
        FileName = fileName;
        Format = format;
        Encoding = encoding;
        LensSurfaces = Math.Max(lensSurfaces, 0);
        Warnings = warnings;
    }

    /// <param name="surfaceCount">Surfaces of the imported system, object and image included.</param>
    public static ImportReport From(PrescriptionImport imported, string fileName, int surfaceCount) =>
        new(fileName, imported.Format, imported.Encoding, surfaceCount - 2, imported.Warnings);

    public string FileName { get; }
    public PrescriptionFormat Format { get; }
    public string Encoding { get; }
    public int LensSurfaces { get; }
    public IReadOnlyList<ImportWarning> Warnings { get; }

    public static ImportCategory Category(ImportWarningKind kind) => kind switch
    {
        ImportWarningKind.UnknownGlass => ImportCategory.Glasses,
        ImportWarningKind.UnsupportedSurface or ImportWarningKind.NonSequential => ImportCategory.Surfaces,
        ImportWarningKind.UnsupportedRecord or ImportWarningKind.SolveDropped => ImportCategory.Dropped,
        ImportWarningKind.Approximated or ImportWarningKind.Defaulted => ImportCategory.Assumed,
        _ => ImportCategory.Problems, // malformed, validation, other
    };

    public static string Title(ImportCategory c) => c switch
    {
        ImportCategory.Problems => Loc.S("Shell_Import_Problems"),
        ImportCategory.Glasses => Loc.S("Shell_Import_Glasses"),
        ImportCategory.Surfaces => Loc.S("Shell_Import_Surfaces"),
        ImportCategory.Dropped => Loc.S("Shell_Import_NotImported"),
        _ => Loc.S("Shell_Import_Assumed"),
    };

    /// <summary>The category's Segoe Fluent Icons glyph.</summary>
    public static string Glyph(ImportCategory c) => c switch
    {
        ImportCategory.Problems => "\uE7BA",  // Warning
        ImportCategory.Glasses => "\uEB42",   // Drop
        ImportCategory.Surfaces => "\uE722",  // camera (aperture)
        ImportCategory.Dropped => "\uECC9",   // remove from
        _ => "\uE9E9",                        // equalizer (sliders)
    };

    /// <summary>The warnings by category, in order of seriousness; each group keeps the file's order.</summary>
    public IReadOnlyList<Group> Groups =>
        Warnings.GroupBy(w => Category(w.Kind)).OrderBy(g => g.Key).Select(g => new Group(g.Key, g.ToList())).ToList();

    /// <summary>Glasses in none of the loaded catalogs: loading the right catalog and importing again resolves them.</summary>
    public bool HasUnknownGlasses => Warnings.Any(w => w.Kind == ImportWarningKind.UnknownGlass);

    /// <summary>"6 surfaces · .zmx · UTF-16"</summary>
    public string Summary => string.Join(" · ",
        Loc.F(LensSurfaces == 1 ? "Shell_Import_SurfaceCountOne" : "Shell_Import_SurfaceCountMany", LensSurfaces),
        "." + Format.WireName(), EncodingName(Encoding));

    /// <summary>The explanation under the title.</summary>
    public string Headline => Warnings.Count == 1
        ? Loc.S("Shell_Import_HeadlineOne")
        : Loc.F("Shell_Import_HeadlineMany", Warnings.Count);

    public static string EncodingName(string e) => e switch
    {
        "utf8" or "utf8_bom" => "UTF-8",
        "utf16_le" or "utf16_be" => "UTF-16",
        "latin1" => "Latin-1",
        _ => e,
    };

    /// <summary>The engine's message as a sentence (it starts in lower case), translated when the catalog has it.</summary>
    public static string Sentence(ImportWarning w)
    {
        if (EngineText.T(w.Message) is var t && !string.Equals(t, w.Message, StringComparison.Ordinal)) return t;
        return w.Message.Length == 0 ? w.Message : char.ToUpperInvariant(w.Message[0]) + w.Message[1..];
    }

    /// <summary>"Surface 3 · Line 42", or null for warnings about the whole file.</summary>
    public static string? Location(ImportWarning w)
    {
        var parts = new[]
        {
            w.Surface is { } s ? Loc.F("Shell_Import_SurfaceAt", s) : null,
            w.Line is { } l ? Loc.F("Shell_Import_LineAt", l) : null,
        }.OfType<string>().ToList();
        return parts.Count == 0 ? null : string.Join(" · ", parts);
    }

    /// <summary>The report as plain text, for the clipboard.</summary>
    public string PlainText
    {
        get
        {
            var lines = new List<string> { Loc.F("Shell_Import_PlainHeading", FileName, Summary) };
            foreach (var g in Groups)
            {
                lines.Add("");
                lines.Add(Loc.F("Shell_Import_PlainGroup", Title(g.Category)));
                foreach (var w in g.Warnings) lines.Add("- " + Sentence(w) + (Location(w) is { } at ? $" [{at}]" : ""));
            }
            return string.Join("\n", lines);
        }
    }
}
