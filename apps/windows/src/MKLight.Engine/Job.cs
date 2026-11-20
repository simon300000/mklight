using System.Threading.Channels;

namespace MKLight.Engine;

/// <summary>Progress of a running job: a monotone fraction in 0…1 and a stage label (mirror of the engine's <c>Progress</c>).</summary>
public readonly record struct JobProgress(double Fraction, string Stage);

/// <summary>
/// Handle to one asynchronous analysis job.
/// <para>
/// The job runs independently of whoever awaits it: cancelling the token passed to
/// <see cref="ResultAsync"/> (or abandoning the await) does not cancel the job, only
/// <see cref="Cancel"/> does. That lets a view drop interest in a result without killing
/// a job another view still needs.
/// </para>
/// </summary>
public sealed class AnalysisJob
{
    readonly Task<AnalysisOutput> task;
    readonly CancellationTokenSource cancellation;
    readonly Action onCancel;
    readonly Func<bool>? onStop;
    readonly ChannelReader<JobProgress> progress;

    public Guid Id { get; } = Guid.NewGuid();
    /// <summary>The request the job was started with.</summary>
    public EngineAnalysisRequest EngineRequest { get; }
    public SnapshotHash SnapshotHash { get; }

    AnalysisJob(
        EngineAnalysisRequest request, SnapshotHash snapshotHash, ChannelReader<JobProgress> progress, Task<AnalysisOutput> task,
        CancellationTokenSource cancellation, Action onCancel, Func<bool>? onStop)
    {
        EngineRequest = request;
        SnapshotHash = snapshotHash;
        this.progress = progress;
        this.task = task;
        this.cancellation = cancellation;
        this.onCancel = onCancel;
        this.onStop = onStop;
    }

    /// <summary>The engine's name of the analysis.</summary>
    public string Analysis => EngineRequest.Analysis;

    /// <summary>
    /// Progress updates (the newest 8 are buffered). The sequence ends when the job ends.
    /// It has a single consumer: enumerate it once, from one task. Cancelling
    /// <paramref name="cancellationToken"/> ends the enumeration, not the job.
    /// </summary>
    public IAsyncEnumerable<JobProgress> ProgressAsync(CancellationToken cancellationToken = default) =>
        progress.ReadAllAsync(cancellationToken);

    /// <summary>
    /// Whether the job can be <b>stopped</b>: ended early while keeping its best result so
    /// far (optimisation, tolerancing and glass substitution). Distinct from
    /// <see cref="Cancel"/>, which discards it.
    /// </summary>
    public bool CanStop => onStop is not null;

    /// <summary>
    /// Asks the job to stop and return its best result so far. Returns whether the request
    /// reached the job; the job then ends normally and <see cref="ResultAsync"/> returns the
    /// stopped result.
    /// </summary>
    public bool Stop() => onStop?.Invoke() ?? false;

    /// <summary>Whether the job has ended (with a result, an error or cancelled).</summary>
    public bool IsCompleted => task.IsCompleted;

    /// <summary>
    /// Waits for the result. Throws <see cref="EngineException"/> (kind <c>Cancelled</c>) if
    /// the job was cancelled. Cancelling <paramref name="cancellationToken"/> abandons this
    /// wait (<see cref="OperationCanceledException"/>) and leaves the job running.
    /// </summary>
    public Task<AnalysisOutput> ResultAsync(CancellationToken cancellationToken = default) =>
        cancellationToken.CanBeCanceled ? task.WaitAsync(cancellationToken) : task;

    /// <summary>Requests cooperative cancellation. Idempotent.</summary>
    public void Cancel()
    {
        onCancel();
        try
        {
            cancellation.Cancel();
        }
        catch (ObjectDisposedException)
        {
        }
    }

    /// <summary>
    /// Runs <paramref name="operation"/> as a job on the thread pool. It reports progress
    /// through the reporter and must observe the token at bounded intervals.
    /// <paramref name="onCancel"/> runs on <see cref="Cancel"/>; <paramref name="onStop"/>
    /// makes the job stoppable (<see cref="CanStop"/>). An
    /// <see cref="OperationCanceledException"/> of the job's own token becomes
    /// <see cref="EngineException.Cancelled"/>.
    /// </summary>
    public static AnalysisJob Run(
        EngineAnalysisRequest request, SnapshotHash snapshotHash,
        Func<JobReporter, CancellationToken, Task<AnalysisOutput>> operation, Action? onCancel = null, Func<bool>? onStop = null)
    {
        var channel = Channel.CreateBounded<JobProgress>(new BoundedChannelOptions(8)
        {
            FullMode = BoundedChannelFullMode.DropOldest, SingleReader = true,
        });
        var reporter = new JobReporter(channel.Writer);
        var cts = new CancellationTokenSource();
        var token = cts.Token;
        var task = Task.Run(async () =>
        {
            try
            {
                reporter.Report(0, "Queued");
                var output = await operation(reporter, token).ConfigureAwait(false);
                if (token.IsCancellationRequested)
                {
                    output.Dispose();
                    throw EngineException.Cancelled();
                }
                reporter.Report(1, "Done");
                return output;
            }
            catch (OperationCanceledException) when (token.IsCancellationRequested)
            {
                throw EngineException.Cancelled();
            }
            finally
            {
                reporter.Finish();
            }
        }, CancellationToken.None);
        return new AnalysisJob(request, snapshotHash, channel.Reader, task, cts, onCancel ?? (() => { }), onStop);
    }
}

/// <summary>Written by a running job; forwards monotone progress to the job's progress sequence.</summary>
public sealed class JobReporter
{
    readonly Lock gate = new();
    readonly ChannelWriter<JobProgress> writer;
    JobProgress state = new(0, "");

    internal JobReporter(ChannelWriter<JobProgress> writer) => this.writer = writer;

    /// <summary>The latest progress.</summary>
    public JobProgress Current
    {
        get
        {
            lock (gate) return state;
        }
    }

    /// <summary>
    /// Reports <paramref name="fraction"/> (clamped to 0…1, never decreasing) and an optional
    /// new stage label. Safe to call from concurrent tasks: the update is written while the
    /// lock is held, so the sequence sees the same order as the monotone state (writing to
    /// the dropping channel never blocks).
    /// </summary>
    public void Report(double fraction, string? stage = null)
    {
        lock (gate)
        {
            var f = double.IsFinite(fraction) ? Math.Clamp(fraction, 0, 1) : 0;
            state = new JobProgress(Math.Max(state.Fraction, f), string.IsNullOrEmpty(stage) ? state.Stage : stage);
            writer.TryWrite(state);
        }
    }

    internal void Finish() => writer.TryComplete();
}
