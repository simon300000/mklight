using System.Text.Json.Nodes;

namespace MKLight.Engine;

/// <summary>
/// A finished analysis: the engine's result (<c>result</c> of the envelope,
/// docs/RESULTS.md) as JSON with its binary buffers, plus the run metadata. Views read
/// both through the tolerant accessors of <see cref="JsonRead"/>.
/// <see cref="Dispose"/> releases the buffers (see <see cref="ResultBuffer"/>).
/// </summary>
public sealed class AnalysisOutput(
    RunInfo runInfo, JsonNode? json, IReadOnlyList<ResultBuffer>? buffers = null, string? libraryHash = null,
    EngineAnalysisRequest? engineRequest = null, BackendReport? backend = null) : IDisposable
{
    public RunInfo RunInfo { get; } = runInfo;
    /// <summary>The analysis-specific result JSON (fully built, so several threads may read it; do not modify it).</summary>
    public JsonNode? Json { get; } = json;
    /// <summary>The result's binary buffers.</summary>
    public IReadOnlyList<ResultBuffer> Buffers { get; } = buffers ?? [];
    /// <summary>The glass library the result was computed with (the envelope's <c>library_hash</c>), if the engine reports one.</summary>
    public string? LibraryHash { get; } = libraryHash;
    /// <summary>
    /// The request that produced this output. Engines that record their own normalised
    /// settings in <see cref="RunInfo.Settings"/> set it, so staleness compares like with like.
    /// </summary>
    public EngineAnalysisRequest? EngineRequest { get; init; } = engineRequest;
    /// <summary>The envelope's backend report (requested, used, fallback reason, cost decision), if the engine sent one.</summary>
    public BackendReport? Backend { get; } = backend;

    /// <summary>The first buffer of that name, or <c>null</c>.</summary>
    public ResultBuffer? Buffer(string name) => Buffers.FirstOrDefault(b => b.Name == name);

    /// <summary>
    /// Whether this output is current for <paramref name="snapshot"/> and
    /// <paramref name="request"/>: computed from the same system <b>and</b> glass library,
    /// with the same request (settings and backend preference). Without the request that
    /// produced it, the engine's recorded settings must contain every requested value.
    /// Otherwise it is stale.
    /// </summary>
    public bool IsCurrent(ModelSnapshot? snapshot, EngineAnalysisRequest request)
    {
        if (snapshot is null || !LibraryMatches(snapshot) || RunInfo.SnapshotHash != snapshot.Hash) return false;
        return EngineRequest is { } own ? own == request : request.Matches(RunInfo);
    }

    bool LibraryMatches(ModelSnapshot snapshot) =>
        LibraryHash is null || snapshot.LibraryHash is null || LibraryHash == snapshot.LibraryHash;

    public void Dispose()
    {
        foreach (var b in Buffers) b.Dispose();
    }
}
