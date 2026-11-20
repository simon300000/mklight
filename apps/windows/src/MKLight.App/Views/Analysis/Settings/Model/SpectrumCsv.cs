using System.Globalization;

namespace MKLight.App.Views.Analysis.Settings;

/// <summary>
/// A sensor response table read from CSV text: two columns, wavelength and
/// response; a header line is optional; separators are commas, semicolons or
/// tabs (with semicolons and tabs a decimal comma is accepted); <c>#</c>
/// starts a comment line. Wavelengths in nanometres are detected by
/// magnitude (or a header naming <c>nm</c>) and converted to µm. Rows that
/// cannot be read are reported with their line numbers and left out; the
/// table is sorted by wavelength.
/// </summary>
public sealed record SpectrumCsv(IReadOnlyList<SpectrumSample> Table, bool Nanometres, string? Header, IReadOnlyList<SpectrumCsv.BadRow> BadRows)
{
    /// <param name="Line">1-based line number in the file (0: the file itself).</param>
    public sealed record BadRow(int Line, string Text, string Reason);

    public string Unit => Nanometres ? "nm" : "µm";

    /// <summary>Readable: at least two rows left.</summary>
    public bool IsUsable => Table.Count >= 2;

    /// <summary>The sampling range that covers the table.</summary>
    public (double Lo, double Hi)? RangeUm => Table.Count >= 2 && Table[^1].Um > Table[0].Um ? (Table[0].Um, Table[^1].Um) : null;

    /// <summary>A file that could not be read at all.</summary>
    public static SpectrumCsv Unreadable(string name) => new([], false, null, [new BadRow(0, name, "cannot be read")]);

    static readonly string[] NmWords = ["nm", "nanometer", "nanometers", "nanometre", "nanometres"];
    static readonly string[] UmWords = ["um", "µm", "micron", "microns", "micrometer", "micrometers", "micrometre", "micrometres"];

    public static SpectrumCsv Parse(string text)
    {
        var rows = new List<(int Line, string Text, double Um, double Value)>();
        var bad = new List<BadRow>();
        string? header = null;
        var lines = text.Replace("\r\n", "\n").Replace('\r', '\n').Split('\n');
        for (var k = 0; k < lines.Length; k++)
        {
            var line = lines[k].Trim().Trim('﻿').Trim();
            if (line.Length == 0 || line.StartsWith('#')) continue;
            var separator = line.Contains('\t') ? '\t' : line.Contains(';') ? ';' : ',';
            var decimalComma = separator != ',';
            var cells = line.Split(separator).Select(c => c.Trim().Trim('"')).ToArray();
            double? Number(string s) =>
                double.TryParse(decimalComma ? s.Replace(',', '.') : s, NumberStyles.Float, CultureInfo.InvariantCulture, out var d) ? d : null;
            var a = cells.Length > 0 ? Number(cells[0]) : null;
            var b = cells.Length > 1 ? Number(cells[1]) : null;
            if (a is not double x || b is not double y)
            {
                if (rows.Count == 0 && bad.Count == 0 && header is null && cells.Any(c => Number(c) is null)) header = line;
                else bad.Add(new(k + 1, line, cells.Length < 2 ? "needs two columns" : "not a number"));
                continue;
            }
            if (!(double.IsFinite(x) && x > 0)) bad.Add(new(k + 1, line, "wavelength must be positive"));
            else if (!(double.IsFinite(y) && y >= 0)) bad.Add(new(k + 1, line, "response must not be negative"));
            else rows.Add((k + 1, line, x, y));
        }
        // Units: a header that names them wins; otherwise wavelengths above
        // 50 are nanometres (optical µm values stay far below).
        var words = System.Text.RegularExpressions.Regex.Split((header ?? "").ToLowerInvariant(), @"[^\p{L}µ]+").Where(w => w.Length > 0).ToHashSet();
        var nm = words.Overlaps(NmWords) || (!words.Overlaps(UmWords) && rows.Count > 0 && rows.Max(r => r.Um) > 50);
        var scale = nm ? 0.001 : 1;
        var table = new List<SpectrumSample>();
        foreach (var r in rows.OrderBy(r => r.Um).ThenBy(r => r.Line))
        {
            var um = r.Um * scale;
            if (table.Count > 0 && Math.Abs(table[^1].Um - um) < 1e-12)
            {
                bad.Add(new(r.Line, r.Text, "repeats a wavelength"));
                continue;
            }
            table.Add(new(um, r.Value));
        }
        return new(table, nm, header, bad.OrderBy(b => b.Line).ToList());
    }
}
