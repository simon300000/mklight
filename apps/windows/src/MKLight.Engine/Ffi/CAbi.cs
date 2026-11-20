using System.Runtime.InteropServices;

namespace MKLight.Engine;

/// <summary>
/// C#-typed wrappers of every C function the app uses (docs/FFI.md). Each wrapper
/// converts strings, frees what the library allocated, and turns a failed <c>MKStatus</c>
/// into an <see cref="EngineException"/> carrying the engine's message, read from
/// <c>mk_last_error()</c> on the same thread right after the call.
/// </summary>
internal static unsafe class CAbi
{
    internal delegate int StringCall(out nint json);

    static readonly Lazy<string?> loadFailure = new(ProbeLibrary);

    /// <summary>
    /// <c>null</c> when the library loads, its ABI major version matches and its minor version is at least the one the binding needs; else why not.
    /// Never throws.
    /// </summary>
    public static string? LoadFailure => loadFailure.Value;

    static string? ProbeLibrary()
    {
        try
        {
            var version = Native.mk_abi_version();
            uint major = version >> 16, minor = version & 0xffff;
            if (major != Native.AbiVersionMajor)
                return $"The engine DLL has ABI {major}.{minor}; this app needs {Native.AbiVersionMajor}.{Native.AbiVersionMinor} or a later {Native.AbiVersionMajor}.x.";
            if (minor < Native.AbiVersionMinor)
                return $"The engine DLL is too old: ABI {major}.{minor}, this app needs {major}.{Native.AbiVersionMinor} or later. Rebuild it with scripts/build-windows.ps1 -EngineOnly.";
            return null;
        }
        catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException or BadImageFormatException)
        {
            return $"{Native.Library}.dll not found or not loadable ({e.Message}): run scripts/build-windows.ps1 -EngineOnly";
        }
    }

    public static string EngineVersion => Utf8(Native.mk_engine_version()) ?? "";

    static string? Utf8(byte* p) => p is null ? null : Marshal.PtrToStringUTF8((nint)p);

    public static void Check(int status, string function)
    {
        if (status == Native.Ok) return;
        var message = Utf8(Native.mk_last_error()) ?? "";
        if (!Enum.IsDefined((EngineErrorCode)status))
        {
            throw EngineException.Engine(EngineErrorCode.Internal, $"{function}: unknown status {status}: {message}");
        }
        throw EngineException.From((EngineErrorCode)status, message.Length == 0 ? $"{function} failed" : message);
    }

    /// <summary>Takes ownership of a returned string: reads it and frees it.</summary>
    public static string? Take(nint s)
    {
        if (s == 0) return null;
        try
        {
            return Marshal.PtrToStringUTF8(s);
        }
        finally
        {
            Native.mk_string_free(s);
        }
    }

    /// <summary>Calls a function with an out-string, checks the status and takes ownership of the string.</summary>
    public static string String(string function, StringCall call)
    {
        var status = call(out var p);
        // Read the engine's message before freeing (it is valid until the next mk_* call).
        try { Check(status, function); }
        catch { Take(p); throw; }
        return Take(p) ?? "";
    }

    // Engine

    public static EngineHandle EngineCreate(string? config)
    {
        Check(Native.mk_engine_create(config, out var engine), "mk_engine_create");
        if (engine == 0) throw EngineException.Engine(EngineErrorCode.Internal, "mk_engine_create returned no engine");
        return new EngineHandle(engine);
    }

    /// <summary>A session opened from a project file (<c>mk_session_open</c>); unreadable catalogs do not fail it.</summary>
    public static SessionHandle SessionOpen(EngineHandle engine, string path)
    {
        Check(Native.mk_session_open(engine, path, out var session), "mk_session_open");
        if (session == 0) throw EngineException.Engine(EngineErrorCode.Internal, "mk_session_open returned no session");
        return new SessionHandle(session, engine);
    }

    public static SessionHandle SessionNew(EngineHandle engine)
    {
        Check(Native.mk_session_new(engine, out var session), "mk_session_new");
        if (session == 0) throw EngineException.Engine(EngineErrorCode.Internal, "mk_session_new returned no session");
        return new SessionHandle(session, engine);
    }

    /// <summary><c>mk_session_apply_solves</c>: the solve report and the new snapshot hash.</summary>
    public static (string Report, string Hash) ApplySolves(SessionHandle s, string? settings)
    {
        var status = Native.mk_session_apply_solves(s, settings, out var report, out var hash);
        var r = Take(report);
        var h = Take(hash);
        Check(status, "mk_session_apply_solves");
        return (r ?? "{}", h ?? "");
    }

    public static string ImportPrescription(ReadOnlySpan<byte> data, string format)
    {
        fixed (byte* bytes = data)
        {
            var status = Native.mk_import_prescription(bytes, (nuint)data.Length, format, out var json);
            var text = Take(json);
            Check(status, "mk_import_prescription");
            return text ?? "";
        }
    }

    // Jobs

    public static ulong Submit(SessionHandle s, string requestJson)
    {
        Check(Native.mk_session_submit_analysis_json(s, requestJson, out var job), "mk_session_submit_analysis_json");
        return job;
    }

    public static (JobState State, double Progress) JobStatus(EngineHandle e, ulong job)
    {
        Check(Native.mk_job_status(e, job, out var state, out var progress), "mk_job_status");
        return (Enum.IsDefined((JobState)state) ? (JobState)state : JobState.Running, progress);
    }

    public static void JobCancel(EngineHandle e, ulong job) => Check(Native.mk_job_cancel(e, job), "mk_job_cancel");

    /// <summary><c>mk_job_stop</c>: ends a job of a stoppable analysis early, keeping its best result so far.</summary>
    public static void JobStop(EngineHandle e, ulong job) => Check(Native.mk_job_stop(e, job), "mk_job_stop");

    public static void JobRelease(EngineHandle e, ulong job) => Check(Native.mk_job_release(e, job), "mk_job_release");

    /// <summary>
    /// Takes the result of a finished job: the envelope JSON and its buffers, wrapped
    /// without copying (the buffers share one <see cref="BufferArray"/>, which runs
    /// <c>mk_buffer_free</c> when the last of them is released).
    /// </summary>
    public static (string Json, ResultBuffer[] Buffers) JobTakeResult(EngineHandle e, ulong job)
    {
        var status = Native.mk_job_take_result(e, job, out var json, out var array, out var count);
        if (status != Native.Ok)
        {
            // The engine's message is read before any other mk_* call (it is valid until the next one).
            try { Check(status, "mk_job_take_result"); }
            finally
            {
                Native.mk_string_free(json);
                Native.mk_buffer_free(array, count);
            }
        }
        var text = Take(json) ?? "{}";
        if (array is null || count == 0) return (text, []);
        return (text, BufferArray.Wrap(array, count));
    }
}

/// <summary>
/// Owns a buffer array from <c>mk_job_take_result</c>: one reference per
/// <see cref="ResultBuffer"/>; <c>mk_buffer_free</c> runs exactly once, when the last
/// buffer is disposed or, if some never are, when the finalizer runs after all of them
/// became unreachable (each buffer references this owner).
/// </summary>
internal sealed unsafe class BufferArray : BufferOwner
{
    nint array;
    readonly nuint count;
    readonly long bytes;
    int references;

    /// <param name="bytes">The data the array keeps alive: reported to the GC as memory
    /// pressure, so replaced results are collected before native memory piles up.</param>
    BufferArray(MKBuffer* array, nuint count, int references, long bytes)
    {
        this.array = (nint)array;
        this.count = count;
        this.references = references;
        this.bytes = bytes;
        if (bytes > 0) GC.AddMemoryPressure(bytes);
    }

    public static ResultBuffer[] Wrap(MKBuffer* array, nuint count)
    {
        long bytes = 0;
        for (nuint i = 0; i < count; i++)
            if (Enum.IsDefined((BufferDType)array[i].DType)) bytes += (long)array[i].Len * ((BufferDType)array[i].DType).ByteWidth();
        var owner = new BufferArray(array, count, checked((int)count), bytes);
        var buffers = new ResultBuffer[(int)count];
        for (int i = 0; i < buffers.Length; i++)
        {
            var b = array[i];
            var shape = new int[b.NDim];
            for (int k = 0; k < shape.Length; k++) shape[k] = checked((int)b.Shape[k]);
            var known = Enum.IsDefined((BufferDType)b.DType);
            buffers[i] = new ResultBuffer(
                b.Name is null ? $"buffer {i}" : Marshal.PtrToStringUTF8((nint)b.Name) ?? $"buffer {i}",
                known ? (BufferDType)b.DType : BufferDType.U8, shape, known ? checked((int)b.Len) : 0, (byte*)b.Data, owner);
        }
        return buffers;
    }

    /// <summary>Whether the native array has been freed.</summary>
    public bool IsFreed => Volatile.Read(ref array) == 0;

    public override void Release()
    {
        if (Interlocked.Decrement(ref references) == 0) Free();
    }

    void Free()
    {
        var p = Interlocked.Exchange(ref array, 0);
        if (p == 0) return;
        Native.mk_buffer_free((MKBuffer*)p, count);
        if (bytes > 0) GC.RemoveMemoryPressure(bytes);
        GC.SuppressFinalize(this);
    }

    ~BufferArray() => Free();
}
