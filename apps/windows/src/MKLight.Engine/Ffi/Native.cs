using System.Runtime.InteropServices;

namespace MKLight.Engine;

/// <summary>Job states of the C ABI (<c>MKJobState</c>).</summary>
internal enum JobState
{
    Queued = 0,
    Running = 1,
    Done = 2,
    Failed = 3,
    Cancelled = 4,
}

internal static class JobStateExtensions
{
    public static bool IsTerminal(this JobState s) => s is JobState.Done or JobState.Failed or JobState.Cancelled;
}

/// <summary><c>MKBuffer</c>: a named, row-major typed array owned by its enclosing array.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct MKBuffer
{
    public byte* Name;
    public void* Data;
    public nuint Len;
    public int DType;
    public uint NDim;
    public nuint* Shape;
    public void* Owner;
}

/// <summary>
/// Source-generated P/Invoke declarations of <c>crates/mklight-ffi/include/mklight.h</c>
/// (the functions the app uses). Strings in are UTF-8; strings out are <c>char*</c> the
/// caller frees with <c>mk_string_free</c> (see <see cref="CAbi"/>).
/// </summary>
internal static unsafe partial class Native
{
    public const string Library = "mklight_ffi";

    /// <summary><c>MK_ABI_VERSION_MAJOR</c> this app was built against.</summary>
    public const uint AbiVersionMajor = 1;

    /// <summary>
    /// The oldest minor version this binding needs: it calls functions of ABI 1.4 (GPU kernel
    /// preparation) and 1.5 (glass catalog management). A newer minor only adds.
    /// </summary>
    public const uint AbiVersionMinor = 5;

    /// <summary><c>MK_OK</c>.</summary>
    public const int Ok = 0;

    [LibraryImport(Library)] public static partial uint mk_abi_version();
    [LibraryImport(Library)] public static partial byte* mk_engine_version();
    [LibraryImport(Library)] public static partial byte* mk_last_error();
    [LibraryImport(Library)] public static partial void mk_string_free(nint s);
    [LibraryImport(Library)] public static partial void mk_buffer_free(MKBuffer* buffers, nuint count);

    // Engine
    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int mk_engine_create(string? configJson, out nint engine);
    [LibraryImport(Library)] public static partial void mk_engine_destroy(nint engine);
    [LibraryImport(Library)] public static partial int mk_engine_list_analyses(EngineHandle engine, out nint json);
    [LibraryImport(Library)] public static partial int mk_engine_list_backends(EngineHandle engine, out nint json);
    [LibraryImport(Library)]
    public static partial int mk_engine_set_job_callback(
        EngineHandle engine, delegate* unmanaged[Cdecl]<nint, ulong, int, double, byte*, void> callback, nint userData);
    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int mk_engine_prepare_gpu(EngineHandle engine, string? cacheDir);
    [LibraryImport(Library)] public static partial int mk_engine_gpu_preparation_json(EngineHandle engine, out nint json);

    // Sessions
    [LibraryImport(Library)] public static partial int mk_session_new(EngineHandle engine, out nint session);
    [LibraryImport(Library)] public static partial void mk_session_destroy(nint session);
    [LibraryImport(Library)] public static partial int mk_session_get_system_json(SessionHandle session, out nint json);
    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int mk_session_set_system_json(SessionHandle session, string systemJson, out nint hash);
    [LibraryImport(Library)] public static partial int mk_session_current_snapshot_hash(SessionHandle session, out nint hash);
    [LibraryImport(Library)] public static partial int mk_session_current_library_hash(SessionHandle session, out nint hash);
    [LibraryImport(Library)] public static partial int mk_session_validate_json(SessionHandle session, out nint json);
    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int mk_session_apply_solves(SessionHandle session, string? settingsJson, out nint report, out nint hash);
    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int mk_session_get_param(SessionHandle session, string pathJson, out nint json);
    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int mk_session_autosave(SessionHandle session, string? path, out nint outPath);

    // Glass catalogs
    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int mk_session_load_catalog(SessionHandle session, string path, out nint json);
    [LibraryImport(Library)] public static partial int mk_session_list_catalogs(SessionHandle session, out nint json);
    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int mk_session_list_glasses(SessionHandle session, string? catalog, out nint json);
    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int mk_session_remove_catalog(SessionHandle session, string name, out nint json);
    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int mk_session_reload_catalog(SessionHandle session, string name, out nint json);
    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int mk_session_set_library_catalogs(SessionHandle session, string pathsJson, out nint json);
    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int mk_session_set_glass_overrides(SessionHandle session, string overridesJson);
    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int mk_session_glass_details(SessionHandle session, string? catalog, string name, out nint json);
    [LibraryImport(Library)] public static partial int mk_session_library_catalogs_in_use(SessionHandle session, out nint json);
    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int mk_glass_from_spec(string specJson, out nint json);
    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int mk_catalog_to_agf(string catalogJson, out nint json);

    // Prescription import
    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int mk_import_prescription(byte* bytes, nuint len, string format, out nint json);
    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int mk_session_import_prescription(SessionHandle session, string path, out nint json);

    // Jobs
    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int mk_session_submit_analysis_json(SessionHandle session, string requestJson, out ulong job);
    [LibraryImport(Library)] public static partial int mk_job_status(EngineHandle engine, ulong job, out int state, out double progress);
    [LibraryImport(Library)] public static partial int mk_job_cancel(EngineHandle engine, ulong job);
    [LibraryImport(Library)] public static partial int mk_job_stop(EngineHandle engine, ulong job);
    [LibraryImport(Library)]
    public static partial int mk_job_take_result(EngineHandle engine, ulong job, out nint json, out MKBuffer* buffers, out nuint count);
    [LibraryImport(Library)] public static partial int mk_job_release(EngineHandle engine, ulong job);
}

/// <summary>Owns an <c>MKEngine*</c>. Releasing it destroys the engine (which first clears the job callback), then frees the callback's GC handle.</summary>
internal sealed class EngineHandle : SafeHandle
{
    nint callbackTarget;

    public EngineHandle() : base(0, ownsHandle: true) { }

    internal EngineHandle(nint engine) : this() => SetHandle(engine);

    public override bool IsInvalid => handle == 0;

    /// <summary>The GC handle passed as the job callback's user data, freed after the engine.</summary>
    internal void AttachCallbackTarget(nint gcHandle) => callbackTarget = gcHandle;

    protected override bool ReleaseHandle()
    {
        Native.mk_engine_destroy(handle);
        if (callbackTarget != 0) GCHandle.FromIntPtr(callbackTarget).Free();
        return true;
    }
}

/// <summary>Owns an <c>MKSession*</c> and keeps its engine handle alive until the session is destroyed.</summary>
internal sealed class SessionHandle : SafeHandle
{
    readonly EngineHandle? engine;

    public SessionHandle() : base(0, ownsHandle: true) { }

    internal SessionHandle(nint session, EngineHandle engine) : this()
    {
        bool added = false;
        engine.DangerousAddRef(ref added);
        this.engine = engine;
        SetHandle(session);
    }

    public override bool IsInvalid => handle == 0;

    protected override bool ReleaseHandle()
    {
        Native.mk_session_destroy(handle);
        engine?.DangerousRelease();
        return true;
    }
}
