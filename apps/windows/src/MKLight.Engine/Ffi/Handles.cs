using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Channels;

namespace MKLight.Engine;

/// <summary>One job event from the engine's callback thread.</summary>
internal readonly record struct JobEvent(JobState State, double Progress, string Stage);

/// <summary>
/// Routes job events from the engine's single <c>mklight-events</c> thread to the job
/// that owns the id.
/// <para>
/// Registration happens under the same lock that <see cref="Deliver"/> takes, and
/// <see cref="Submit"/> registers while still holding it, so an event can never arrive for
/// a job before its channel exists. An event for an id that is not registered (its job has
/// ended, or its consumer went away) is dropped. Channels are unbounded: a job produces at
/// most one event per progress interval, and writing never blocks the events thread.
/// The hub holds no reference to the engine, so the GC handle that roots it for the
/// callback does not keep the engine alive.
/// </para>
/// </summary>
internal sealed unsafe class JobEventHub
{
    readonly Lock gate = new();
    readonly Dictionary<ulong, Channel<JobEvent>> streams = [];

    /// <summary>Submits under the registration lock and returns the job id with its event channel, which completes after the terminal event.</summary>
    public (ulong Id, ChannelReader<JobEvent> Events) Submit(Func<ulong> body)
    {
        lock (gate)
        {
            var id = body();
            var channel = Channel.CreateUnbounded<JobEvent>(new UnboundedChannelOptions { SingleReader = true });
            streams[id] = channel;
            return (id, channel.Reader);
        }
    }

    public void Deliver(ulong id, JobEvent e)
    {
        Channel<JobEvent>? channel;
        bool terminal = e.State.IsTerminal();
        lock (gate)
        {
            if (terminal) streams.Remove(id, out channel);
            else streams.TryGetValue(id, out channel);
        }
        if (channel is null) return;
        channel.Writer.TryWrite(e);
        if (terminal) channel.Writer.TryComplete();
    }

    /// <summary>Forgets a job whose consumer went away.</summary>
    public void Remove(ulong id)
    {
        Channel<JobEvent>? channel;
        lock (gate) streams.Remove(id, out channel);
        channel?.Writer.TryComplete();
    }

    /// <summary>
    /// The engine's job callback. Runs on the engine's <c>mklight-events</c> thread with no
    /// engine lock held; it only enqueues, and never lets an exception escape.
    /// </summary>
    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    public static void OnJobEvent(nint userData, ulong job, int state, double progress, byte* stage)
    {
        try
        {
            if (userData == 0 || GCHandle.FromIntPtr(userData).Target is not JobEventHub hub) return;
            var s = Enum.IsDefined((JobState)state) ? (JobState)state : JobState.Running;
            hub.Deliver(job, new JobEvent(s, progress, stage is null ? "" : Marshal.PtrToStringUTF8((nint)stage) ?? ""));
        }
        catch
        {
            // Nothing may unwind into the engine.
        }
    }
}

/// <summary>
/// Owns an <c>MKEngine*</c> (through <see cref="EngineHandle"/>): the engine handle, its
/// job-event hub and the cached analysis list. Thread-safe (the C engine is). The engine
/// is destroyed once this core and every session handle are gone (sessions add a
/// reference to the engine handle); destroying it clears the callback before the hub's
/// GC handle is freed. Sessions and running jobs keep their core alive.
/// </summary>
internal sealed unsafe class FFIEngineCore
{
    readonly Lock gate = new();
    string? gpuPreparationError;

    public EngineHandle Handle { get; }
    public JobEventHub Hub { get; } = new();
    public string Version { get; }
    /// <summary><c>mk_engine_list_analyses</c>, read once (the registry is fixed after creation).</summary>
    public IReadOnlyList<AnalysisDescriptor> Descriptors { get; }
    /// <summary>The analyses whose jobs <c>mk_job_stop</c> can stop: those the engine lists as <c>stoppable</c>.</summary>
    public IReadOnlySet<string> Stoppable { get; }
    /// <summary>The engine's backends (<c>mk_engine_list_backends</c>), read once.</summary>
    public string BackendsJson { get; }
    /// <summary>The name of the first GPU backend the engine lists (<c>gpu:&lt;name&gt;</c>), if any.</summary>
    public string? GpuName { get; }

    /// <summary>Why <c>mk_engine_prepare_gpu</c> refused to start, if it did.</summary>
    public string? GpuPreparationError
    {
        get { lock (gate) return gpuPreparationError; }
        set { lock (gate) gpuPreparationError = value; }
    }

    public FFIEngineCore(string? configuration)
    {
        if (CAbi.LoadFailure is { } failure) throw EngineException.LibraryUnavailable(failure);
        Handle = CAbi.EngineCreate(configuration);
        try
        {
            Version = CAbi.EngineVersion;
            var target = GCHandle.Alloc(Hub);
            Handle.AttachCallbackTarget(GCHandle.ToIntPtr(target));
            CAbi.Check(
                Native.mk_engine_set_job_callback(Handle, &JobEventHub.OnJobEvent, GCHandle.ToIntPtr(target)), "mk_engine_set_job_callback");
            var handle = Handle;
            Descriptors = AnalysisDescriptor.ListFromJson(
                CAbi.String("mk_engine_list_analyses", (out nint p) => Native.mk_engine_list_analyses(handle, out p)));
            Stoppable = Descriptors.Where(d => d.Stoppable).Select(d => d.Name).ToHashSet();
            BackendsJson = CAbi.String("mk_engine_list_backends", (out nint p) => Native.mk_engine_list_backends(handle, out p));
            GpuName = (Json.Parse(BackendsJson, "backend list").ArrayOrNull() ?? [])
                .Select(b => b.At("id").StringOrNull())
                .FirstOrDefault(id => id is not null && id.StartsWith("gpu:", StringComparison.Ordinal))?[4..];
        }
        catch
        {
            Handle.Dispose();
            throw;
        }
    }
}

/// <summary>
/// The engine job id of an <see cref="AnalysisJob"/>, filled in once the job's task has
/// submitted it, whether <c>Cancel()</c> came first, and whether a stop was requested.
/// </summary>
internal sealed class Submission
{
    enum Phase { Pending, Submitted, Cancelled }

    readonly Lock gate = new();
    Phase phase;
    ulong id;
    bool stop;

    /// <summary>Whether <c>Stop()</c> was requested (before or after submission).</summary>
    public bool StopRequested
    {
        get { lock (gate) return stop; }
    }

    public bool IsCancelled
    {
        get { lock (gate) return phase == Phase.Cancelled; }
    }

    /// <summary>Records a stop request; returns the engine job id when submitted.</summary>
    public ulong? RequestStop()
    {
        lock (gate)
        {
            stop = true;
            return phase == Phase.Submitted ? id : null;
        }
    }

    /// <summary>Records the engine job id. <c>false</c> when the job was cancelled first (the caller then cancels the engine job).</summary>
    public bool Record(ulong jobId)
    {
        lock (gate)
        {
            if (phase == Phase.Cancelled) return false;
            phase = Phase.Submitted;
            id = jobId;
            return true;
        }
    }

    /// <summary>Marks the job cancelled; returns the engine job id if it was submitted already.</summary>
    public ulong? Cancel()
    {
        lock (gate)
        {
            var submitted = phase == Phase.Submitted ? id : (ulong?)null;
            phase = Phase.Cancelled;
            return submitted;
        }
    }
}
