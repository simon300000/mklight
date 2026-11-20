using System.Globalization;
using MKLight.Engine;

namespace MKLight.App.ViewModels;

/// <summary>
/// Follows the engine's preparation of the GPU's compute kernels at launch for the status
/// capsule (the Mac app's GPUPreparationMonitor): the state, polled every 200 ms while
/// preparing for at most 600 s, and what the capsule shows. A preparation that loads from the
/// cache within 400 ms is never shown; one shown is confirmed with "GPU ready" for 3 s.
/// No UI types: the capsule reads <see cref="BadgeText"/>, <see cref="Summary"/> and
/// <see cref="Note"/> on <see cref="Changed"/>.
/// </summary>
public sealed class GpuPreparationMonitor
{
    /// <summary>The app's monitor (one engine, one preparation per run).</summary>
    public static GpuPreparationMonitor Shared { get; } = new();

    /// <summary>What the capsule shows.</summary>
    public enum Display
    {
        Hidden,
        /// <summary>Preparing for longer than a glance: shown with its progress.</summary>
        Preparing,
        /// <summary>Finished after being shown: a short confirmation.</summary>
        Finished,
    }

    public static readonly TimeSpan FollowLimit = TimeSpan.FromSeconds(600);
    public static readonly TimeSpan PollInterval = TimeSpan.FromMilliseconds(200);
    public static readonly TimeSpan ShowAfter = TimeSpan.FromMilliseconds(400);
    public static readonly TimeSpan ReadyFor = TimeSpan.FromSeconds(3);

    public GpuPreparation State { get; private set; } = new GpuPreparation.Idle();
    public Display Shown { get; private set; } = Display.Hidden;

    /// <summary>Raised when <see cref="State"/> or <see cref="Shown"/> changes (on the follower's context).</summary>
    public event Action? Changed;

    bool _started;

    /// <summary>Follows <paramref name="poll"/> once per monitor; later calls do nothing.</summary>
    public Task StartAsync(Func<GpuPreparation> poll)
    {
        if (_started) return Task.CompletedTask;
        _started = true;
        return FollowAsync(poll);
    }

    /// <summary>
    /// Polls until the preparation leaves <c>preparing</c> or <paramref name="limit"/> passes
    /// (the engine keeps preparing; the capsule stops showing it). <paramref name="delay"/> and
    /// <paramref name="elapsed"/> are injectable for tests.
    /// </summary>
    public async Task FollowAsync(
        Func<GpuPreparation> poll, TimeSpan? limit = null, TimeSpan? interval = null,
        Func<TimeSpan, Task>? delay = null, Func<TimeSpan>? elapsed = null)
    {
        limit ??= FollowLimit;
        interval ??= PollInterval;
        delay ??= Task.Delay;
        if (elapsed is null)
        {
            var clock = System.Diagnostics.Stopwatch.StartNew();
            elapsed = () => clock.Elapsed;
        }
        while (true)
        {
            Set(poll(), Shown);
            if (!State.IsPreparing || elapsed() >= limit) break;
            // Loading from the cache takes milliseconds: no flash.
            if (Shown == Display.Hidden && elapsed() > ShowAfter) Set(State, Display.Preparing);
            await delay(interval.Value);
        }
        if (Shown != Display.Preparing) return;
        if (State is GpuPreparation.Ready)
        {
            Set(State, Display.Finished);
            await delay(ReadyFor);
        }
        Set(State, Display.Hidden);
    }

    /// <summary>Shows a state as if the engine reported it (tests and the screenshot launch action).</summary>
    public void Simulate(GpuPreparation state, Display shown) => Set(state, shown);

    void Set(GpuPreparation state, Display shown)
    {
        if (Equals(state, State) && shown == Shown) return;
        State = state;
        Shown = shown;
        Changed?.Invoke();
    }

    /// <summary>The capsule's text: "Preparing GPU 42%", "GPU ready", or null (nothing shown).</summary>
    public static string? BadgeText(GpuPreparation state, Display shown) => shown switch
    {
        Display.Preparing => "Preparing GPU " + Math.Round(state.Fraction * 100).ToString("0", CultureInfo.InvariantCulture) + "%",
        Display.Finished => "GPU ready",
        _ => null,
    };

    public const string BadgeHelp =
        "Compiling the GPU's compute kernels, once after installing or updating MKLight or the graphics driver. A GPU analysis started now waits only for the kernels it needs.";

    /// <summary>The details' "GPU kernels" row; null when there is nothing to say (idle).</summary>
    public static string? Summary(GpuPreparation state) => state switch
    {
        GpuPreparation.Preparing p => p.Total > 0
            ? $"Preparing ({p.Done.ToString(CultureInfo.InvariantCulture)} of {p.Total.ToString(CultureInfo.InvariantCulture)})"
            : "Preparing",
        GpuPreparation.Ready r => r.FromCache == r.Kernels
            ? $"Ready ({r.Kernels.ToString(CultureInfo.InvariantCulture)}, loaded from cache)"
            : $"Ready ({r.Kernels.ToString(CultureInfo.InvariantCulture)}, prepared in {r.Seconds.ToString("0.0", CultureInfo.InvariantCulture)} s)",
        GpuPreparation.Unavailable u => "No GPU: " + u.Reason,
        GpuPreparation.Failed f => $"Compiled on first use ({f.Message})",
        _ => null,
    };

    /// <summary>A second line under the row: the kernels are ready, but the saved ones could not be read or written.</summary>
    public static string? Note(GpuPreparation state) =>
        state is GpuPreparation.Ready { Warning: { } w } ? $"Kernel cache unavailable: the next launch compiles again ({w})" : null;
}
