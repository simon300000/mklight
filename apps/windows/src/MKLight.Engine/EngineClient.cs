using System.Text.Json.Nodes;

namespace MKLight.Engine;

/// <summary>Content hash of a model snapshot, as hex. Produced by the engine (BLAKE3 of the canonical JSON); front ends only compare hashes.</summary>
public readonly record struct SnapshotHash(string Hex)
{
    /// <summary>First 12 hex digits, for display.</summary>
    public string Short => Hex.Length > 12 ? Hex[..12] : Hex;

    public override string ToString() => Hex;
}

/// <summary>
/// An immutable, engine-validated model snapshot. Analyses only ever run against
/// snapshots, never against the live document. The system crosses the boundary as the
/// engine's OpticalSystem JSON (the model decodes and encodes it). Two snapshots are equal
/// when both hashes are.
/// </summary>
/// <param name="SystemJson">The system as JSON text.</param>
/// <param name="LibraryHash">
/// Hash of the glass library the snapshot was taken with (the engine's <c>library_hash</c>;
/// <c>null</c> for engines without user catalogs). A result is current only while both hashes match.
/// </param>
public sealed record ModelSnapshot(string SystemJson, SnapshotHash Hash, string? LibraryHash = null)
{
    public bool Equals(ModelSnapshot? other) => other is not null && Hash == other.Hash && LibraryHash == other.LibraryHash;

    public override int GetHashCode() => HashCode.Combine(Hash, LibraryHash);
}

/// <param name="IsMock">True for engines that return synthetic data. Front ends must label such results.</param>
public sealed record EngineInfo(string Name, string Version, bool IsMock);

/// <summary>The outcome of <see cref="IEngineClient.ApplySolvesAsync"/>.</summary>
/// <param name="Report">The <c>solves</c> report (<c>outcomes</c>, <c>warnings</c>, …; docs/RESULTS.md).</param>
/// <param name="Snapshot">The solved system's snapshot.</param>
public sealed record SolveApplication(JsonNode? Report, ModelSnapshot Snapshot);

/// <summary>
/// The boundary between front ends and the optical engine.
/// <para>
/// Every computation goes through this interface; the UI never computes optics itself.
/// The implementation is <see cref="FFIEngine"/> (the Rust engine over its C ABI).
/// Analyses are requested by name with JSON settings, and results come back as the
/// engine's JSON with typed buffers. Systems cross as OpticalSystem JSON text.
/// </para>
/// <para>
/// The members after <see cref="MakeDocumentEngine"/> have default implementations for
/// engines without those capabilities (they answer empty or throw <c>Unsupported</c>).
/// Asynchronous members never block their caller.
/// </para>
/// </summary>
public interface IEngineClient
{
    EngineInfo Info { get; }

    /// <summary>Validates the system and freezes it into a hashed snapshot. Throws <see cref="EngineException"/> (<c>InvalidModel</c>) with the problems found.</summary>
    Task<ModelSnapshot> OpenSnapshotAsync(string systemJson);

    /// <summary>Every analysis the engine knows, with its settings schema.</summary>
    Task<IReadOnlyList<AnalysisDescriptor>> AnalysisDescriptorsAsync();

    /// <summary>
    /// Starts an engine analysis of <paramref name="snapshot"/> by name, with JSON settings.
    /// Returns at once; the job runs asynchronously. Observe its progress and await its
    /// result through the returned handle. Unknown analyses and invalid settings fail the job at once.
    /// </summary>
    AnalysisJob Submit(EngineAnalysisRequest request, ModelSnapshot snapshot);

    /// <summary>Names of the glasses the engine can resolve (for completion and validation in editors).</summary>
    Task<IReadOnlyList<string>> MaterialNamesAsync();

    /// <summary>
    /// An engine for one document: same engine, its own session state (user glass catalogs,
    /// library hash). Engines without per-document state return themselves.
    /// </summary>
    IEngineClient MakeDocumentEngine() => this;

    /// <summary>The engine's validate report (<c>{"ok", "errors", "warnings", "issues"}</c>) of the snapshot's system.</summary>
    async Task<JsonNode?> ValidateAsync(ModelSnapshot snapshot)
    {
        using var output = await Submit(new EngineAnalysisRequest("validate"), snapshot).ResultAsync().ConfigureAwait(false);
        return Json.Clone(output.Json);
    }

    /// <summary>
    /// Solves the snapshot's system (docs/SOLVES.md) and returns the solve report (the
    /// <c>solves</c> result) with the snapshot of the solved system. The <c>solves</c>
    /// analysis is the dry run of this.
    /// </summary>
    Task<SolveApplication> ApplySolvesAsync(IReadOnlyDictionary<string, JsonNode?> settings, ModelSnapshot snapshot) =>
        throw EngineException.Unsupported($"{Info.Name} cannot apply solves");

    /// <summary>The current value of a parameter (a ParamPath as JSON) of the snapshot's system: a number, ±∞, or NaN when it has none.</summary>
    Task<double> GetParamAsync(ModelSnapshot snapshot, string paramPathJson) =>
        throw EngineException.Unsupported($"{Info.Name} cannot read parameters");

    /// <summary>Glass catalogs of this (document) engine, built-in first.</summary>
    Task<IReadOnlyList<CatalogInfo>> CatalogsAsync() => Task.FromResult<IReadOnlyList<CatalogInfo>>([]);

    /// <summary>Loads a user glass catalog (AGF text, or JSON when the file ends in <c>.json</c>), replacing one of the same name. Changes the library hash.</summary>
    Task<CatalogInfo> LoadCatalogAsync(string path) => throw EngineException.Unsupported($"{Info.Name} cannot load glass catalogs");

    /// <summary>Removes a catalog the project records from this session.</summary>
    Task RemoveCatalogAsync(string name) => throw EngineException.Unsupported($"{Info.Name} cannot remove glass catalogs");

    /// <summary>Reads a catalog again from its file and reports what changed.</summary>
    Task<CatalogChanges> ReloadCatalogAsync(string name) => throw EngineException.Unsupported($"{Info.Name} cannot reload glass catalogs");

    /// <summary>Makes the session's app-wide library catalogs exactly these files, in order (unchanged files are not read again).</summary>
    Task<LibrarySync> SetLibraryCatalogsAsync(IReadOnlyList<string> paths) => Task.FromResult(LibrarySync.Empty);

    /// <summary>The user's availability overrides (status, exclusion from glass substitution), applied to the session's glass library.</summary>
    Task SetGlassOverridesAsync(IReadOnlyList<GlassOverride> overrides) => Task.CompletedTask;

    /// <summary>Glass table rows of one catalog, or of every catalog (<paramref name="catalog"/> null).</summary>
    Task<IReadOnlyList<GlassRow>> GlassRowsAsync(string? catalog) => Task.FromResult<IReadOnlyList<GlassRow>>([]);

    /// <summary>Everything about one glass (<paramref name="catalog"/> null: the first that has it).</summary>
    Task<GlassDetails> GlassDetailsAsync(string? catalog, string name) => throw EngineException.Unsupported($"{Info.Name} has no glass data");

    /// <summary>
    /// The library catalogs the design uses that the project does not record yet, as the
    /// engine's JSON array of catalog references (<c>{"name", "path", "relative_path", "format"}</c>,
    /// a model type); a saved project records them.
    /// </summary>
    Task<string> LibraryCatalogsInUseJsonAsync() => Task.FromResult("[]");

    /// <summary>A catalog glass (engine JSON) from an editor's specification (<c>GlassSpec</c>, docs/FFI.md), with n_d and V_d computed.</summary>
    JsonNode? GlassFromSpec(JsonNode spec) => throw EngineException.Unsupported($"{Info.Name} cannot make glasses");

    /// <summary>A catalog in the engine's JSON form as AGF text.</summary>
    AgfExport Agf(JsonNode catalog) => throw EngineException.Unsupported($"{Info.Name} cannot write AGF catalogs");

    /// <summary>
    /// Imports a lens prescription file (<c>.zmx</c> or <c>.seq</c>, any common text
    /// encoding) with the built-in glass catalog. Content that does not map is listed in the
    /// result's warnings; only text without surfaces throws. Synchronous: it blocks the caller.
    /// </summary>
    PrescriptionImport ImportPrescription(ReadOnlySpan<byte> data, PrescriptionFormat format) =>
        throw EngineException.Unsupported($"{Info.Name} cannot import lens prescription files");

    /// <summary>Imports a lens prescription file into this (document) engine's session, resolving glasses against its user catalogs.</summary>
    async Task<PrescriptionImport> ImportPrescriptionAsync(string path)
    {
        var format = PrescriptionFormats.FromExtension(Path.GetExtension(path))
            ?? throw EngineException.Unsupported($"{Path.GetFileName(path)} is not a lens prescription file");
        var data = await File.ReadAllBytesAsync(path).ConfigureAwait(false);
        return ImportPrescription(data, format);
    }

    /// <summary>
    /// Starts creating the GPU's compute kernels in the background (once per engine),
    /// saving the compiled kernels in <paramref name="cacheDirectory"/> so later launches
    /// load them. Returns at once; follow it with <see cref="GetGpuPreparation"/>.
    /// </summary>
    void PrepareGpu(string? cacheDirectory) { }

    /// <summary>The state of <see cref="PrepareGpu"/>.</summary>
    GpuPreparation GetGpuPreparation() => new GpuPreparation.Idle();
}
