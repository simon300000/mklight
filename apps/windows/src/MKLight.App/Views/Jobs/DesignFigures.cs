using System.Runtime.CompilerServices;
using MKLight.App.ViewModels;
using MKLight.Engine;
using MKLight.Model;

namespace MKLight.App.Views.Jobs;

/// <summary>
/// The engine results the job pages draw of whole designs (the design a job started from, a result or candidate):
/// its layout, its glass map. Each (design, analysis) runs once on the document's engine, read off the UI thread and
/// cached for as long as the design object lives; a request nobody holds any more is cancelled (the selection moved
/// on). A catalogue change empties the cache. Every call is on the UI thread.
/// </summary>
public sealed class DesignFigures
{
    static readonly ConditionalWeakTable<Workspace, DesignFigures> Table = [];
    readonly Workspace _ws;
    ConditionalWeakTable<OpticalSystem, Dictionary<string, Entry>> _entries = [];

    DesignFigures(Workspace ws)
    {
        _ws = ws;
        ws.CatalogsChanged += () => _entries = [];
    }

    public static DesignFigures Of(Workspace ws) => Table.GetValue(ws, x => new DesignFigures(x));

    /// <summary>Requests still computing, over every window (the screenshot launch action waits for none).</summary>
    public static int Pending { get; private set; }

    sealed class Entry
    {
        public object? Value;
        public bool Done;
        public int Users;
        public readonly CancellationTokenSource Cancel = new();
        public event Action<object?>? Ready;
        public void Finish(object? value)
        {
            Value = value;
            Done = true;
            Ready?.Invoke(value);
            Ready = null;
        }
    }

    /// <summary>A hold on a request; disposing it lets an unfinished request be cancelled.</summary>
    public sealed class Lease : IDisposable
    {
        Action? _release;
        internal Lease(Action release) => _release = release;
        public void Dispose() { _release?.Invoke(); _release = null; }
    }

    /// <summary>
    /// The result of <paramref name="analysis"/> on <paramref name="system"/>, read by <paramref name="read"/> on a
    /// worker thread: <paramref name="ready"/> runs on the UI thread with it (at once when cached; null when the engine
    /// failed). The output is disposed after reading, so the reader copies what it keeps.
    /// </summary>
    public Lease Request<T>(OpticalSystem system, string analysis, Func<AnalysisOutput, OpticalSystem, T?> read, Action<T?> ready) where T : class
    {
        var map = _entries.GetValue(system, _ => []);
        if (!map.TryGetValue(analysis, out var e))
        {
            e = new Entry();
            map[analysis] = e;
            _ = Load(e, map, system, analysis, (o, s) => read(o, s));
        }
        if (e.Done)
        {
            ready(e.Value as T);
            return new Lease(() => { });
        }
        void OnReady(object? v) => ready(v as T);
        e.Ready += OnReady;
        e.Users++;
        return new Lease(() =>
        {
            e.Ready -= OnReady;
            if (--e.Users > 0 || e.Done) return;
            e.Cancel.Cancel();
            map.Remove(analysis); // a later request starts again
        });
    }

    async Task Load(Entry e, Dictionary<string, Entry> map, OpticalSystem system, string analysis, Func<AnalysisOutput, OpticalSystem, object?> read)
    {
        Pending++;
        object? value = null;
        var ct = e.Cancel.Token;
        try
        {
            var snap = ReferenceEquals(system, _ws.DesignSystem) && _ws.DesignSnapshot is { } d
                ? d : await _ws.Engine.OpenSnapshotAsync(ModelJson.Encode(system));
            if (ct.IsCancellationRequested) return;
            var job = _ws.Engine.Submit(new EngineAnalysisRequest(analysis), snap);
            using var stop = ct.Register(job.Cancel);
            using var output = await job.ResultAsync();
            if (ct.IsCancellationRequested) return;
            value = await Task.Run(() => read(output, system), CancellationToken.None);
        }
        catch (EngineException x)
        {
            if (x.Kind == EngineErrorKind.Cancelled || ct.IsCancellationRequested) return;
            App.Log($"{analysis} for a job figure: {x.Message}");
        }
        catch (Exception x) when (x is InvalidOperationException or FormatException or IndexOutOfRangeException or ArgumentException)
        {
            App.Log($"{analysis} for a job figure: {x.Message}");
        }
        finally { Pending--; }
        if (ct.IsCancellationRequested) return;
        e.Finish(value);
        if (value is null) map.Remove(analysis); // not cached: a later request tries again
    }
}
