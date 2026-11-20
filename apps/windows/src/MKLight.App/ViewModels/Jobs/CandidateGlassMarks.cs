using System.Globalization;
using MKLight.Model;

namespace MKLight.App.ViewModels.Jobs;

/// <summary>A glass to ring on the glass map: its map id (<c>catalog/name</c>), name and surfaces ("S1, S3").</summary>
public sealed record GlassMark(string Id, string Catalog, string Name, IReadOnlyList<int> Surfaces)
{
    public string Label => string.Join(", ", Surfaces.Select(s => "S" + s.ToString(CultureInfo.InvariantCulture)));
}

/// <summary>
/// A glass substitution candidate on the glass map (the Mac app's highlights): the candidate's glasses ringed and
/// labelled with their surfaces, one mark per glass however many lenses use it; and, per surface, the design glass it
/// replaces, so the move from the design's glass to the candidate's reads on the map.
/// </summary>
public static class CandidateGlassMarks
{
    /// <summary>The glass map's id of a catalogue glass.</summary>
    public static string MapId(string catalog, string name) => $"{catalog}/{name}";

    /// <summary>The candidate's glasses grouped by glass, surfaces ascending, ordered by id; surfaces not in the design are left out.</summary>
    public static IReadOnlyList<GlassMark> Marks(IEnumerable<SubstitutedGlass> glasses, Func<SurfaceID, int?> surfaceIndex) =>
        glasses
            .Select(g => (g, Index: surfaceIndex(g.Surface)))
            .Where(x => x.Index is not null)
            .GroupBy(x => MapId(x.g.Catalog, x.g.Name))
            .Select(group => new GlassMark(group.Key, group.First().g.Catalog, group.First().g.Name,
                group.Select(x => x.Index!.Value).Distinct().Order().ToList()))
            .OrderBy(m => m.Id, StringComparer.Ordinal)
            .ToList();

    /// <summary>
    /// The design glass at a surface, from the map's glasses of the design (id and surfaces): an offset or model glass
    /// (its own point) before the catalogue glass of the same lens.
    /// </summary>
    public static string? Original(IEnumerable<(string Id, string Kind, IReadOnlyList<int> Surfaces)> designGlasses, int surface) =>
        designGlasses.Where(g => g.Surfaces.Contains(surface))
            .OrderBy(g => g.Kind == "catalog" ? 1 : 0)
            .Select(g => g.Id)
            .FirstOrDefault();
}
