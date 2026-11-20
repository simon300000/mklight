using System.Text.Json.Nodes;
using MKLight.App.ViewModels;
using MKLight.App.Views;
using MKLight.App.Views.Analysis.Results.Readers;
using MKLight.Engine;
using Display = MKLight.App.ViewModels.GpuPreparationMonitor.Display;

namespace MKLight.Tests.Status;

/// <summary>The GPU kernel preparation as the capsule shows it (docs/FFI.md "GPU kernel preparation").</summary>
public class GpuPreparationTests
{
    static GpuPreparation State(string json) => GpuPreparation.FromJson(json);

    [Fact]
    public void EveryStateInWords()
    {
        Assert.Null(GpuPreparationMonitor.Summary(State("""{"state": "idle"}""")));
        Assert.Equal("Preparing (86 of 205)", GpuPreparationMonitor.Summary(State("""{"state": "preparing", "device": "RTX 4090", "done": 86, "total": 205}""")));
        Assert.Equal("Preparing", GpuPreparationMonitor.Summary(State("""{"state": "preparing", "device": "RTX 4090", "done": 0, "total": 0}""")));
        Assert.Equal("Ready (205, loaded from cache)",
            GpuPreparationMonitor.Summary(State("""{"state": "ready", "device": "RTX 4090", "kernels": 205, "from_cache": 205, "seconds": 0.04}""")));
        Assert.Equal("Ready (205, prepared in 31.4 s)",
            GpuPreparationMonitor.Summary(State("""{"state": "ready", "device": "RTX 4090", "kernels": 205, "from_cache": 3, "seconds": 31.42}""")));
        Assert.Equal("No GPU: no adapter", GpuPreparationMonitor.Summary(State("""{"state": "unavailable", "reason": "no adapter"}""")));
        Assert.Equal("Compiled on first use (driver refused the cache)",
            GpuPreparationMonitor.Summary(State("""{"state": "failed", "device": "RTX 4090", "message": "driver refused the cache"}""")));
        Assert.Null(GpuPreparationMonitor.Summary(State("""{"state": "something newer"}""")));
    }

    [Fact]
    public void CacheUnavailableNote()
    {
        Assert.Equal("Kernel cache unavailable: the next launch compiles again (read-only folder)",
            GpuPreparationMonitor.Note(State("""{"state": "ready", "device": "d", "kernels": 5, "from_cache": 0, "seconds": 2, "warning": "read-only folder"}""")));
        Assert.Null(GpuPreparationMonitor.Note(State("""{"state": "ready", "device": "d", "kernels": 5, "from_cache": 0, "seconds": 2}""")));
        Assert.Null(GpuPreparationMonitor.Note(State("""{"state": "preparing", "device": "d", "done": 1, "total": 5}""")));
    }

    [Fact]
    public void BadgeText()
    {
        var preparing = new GpuPreparation.Preparing("d", 86, 205);
        Assert.Equal("Preparing GPU 42%", GpuPreparationMonitor.BadgeText(preparing, Display.Preparing));
        Assert.Equal("Preparing GPU 0%", GpuPreparationMonitor.BadgeText(new GpuPreparation.Preparing("d", 0, 0), Display.Preparing));
        Assert.Equal("GPU ready", GpuPreparationMonitor.BadgeText(new GpuPreparation.Ready("d", 5, 0, 1), Display.Finished));
        Assert.Null(GpuPreparationMonitor.BadgeText(preparing, Display.Hidden));
    }

    /// <summary>A follow with a fake clock: each poll returns the next state, each delay advances the clock.</summary>
    static async Task<(List<Display> Shown, List<TimeSpan> Delays)> Follow(IEnumerable<GpuPreparation> states, TimeSpan? limit = null)
    {
        var monitor = new GpuPreparationMonitor();
        var now = TimeSpan.Zero;
        var shown = new List<Display>();
        var delays = new List<TimeSpan>();
        monitor.Changed += () => { if (shown.Count == 0 || shown[^1] != monitor.Shown) shown.Add(monitor.Shown); };
        using var e = states.GetEnumerator();
        GpuPreparation last = new GpuPreparation.Idle();
        await monitor.FollowAsync(() => last = e.MoveNext() ? e.Current : last, limit, null,
            d => { delays.Add(d); now += d; return Task.CompletedTask; }, () => now);
        return (shown, delays);
    }

    [Fact]
    public async Task LoadedFromTheCacheAtOnceIsNeverShown()
    {
        var (shown, delays) = await Follow([new GpuPreparation.Preparing("d", 0, 5), new GpuPreparation.Ready("d", 5, 5, 0.01)]);
        Assert.Equal([Display.Hidden], shown);
        Assert.Equal([TimeSpan.FromMilliseconds(200)], delays);
    }

    [Fact]
    public async Task ALongPreparationShowsItsProgressThenReady()
    {
        var states = Enumerable.Range(0, 5).Select(i => (GpuPreparation)new GpuPreparation.Preparing("d", i, 5)).Append(new GpuPreparation.Ready("d", 5, 0, 1));
        var (shown, delays) = await Follow(states);
        // Shown after more than 400 ms (the third poll), "GPU ready" for 3 s, then hidden.
        Assert.Equal([Display.Hidden, Display.Preparing, Display.Finished, Display.Hidden], shown);
        Assert.All(delays.SkipLast(1), d => Assert.Equal(TimeSpan.FromMilliseconds(200), d));
        Assert.Equal(TimeSpan.FromSeconds(3), delays[^1]);
    }

    [Fact]
    public async Task AFailureAfterBeingShownHidesWithoutReady()
    {
        var states = Enumerable.Range(0, 4).Select(i => (GpuPreparation)new GpuPreparation.Preparing("d", i, 9)).Append(new GpuPreparation.Failed("d", "x"));
        var (shown, _) = await Follow(states);
        Assert.Equal([Display.Hidden, Display.Preparing, Display.Hidden], shown);
    }

    [Fact]
    public async Task FollowedForAtMostTheLimit()
    {
        var (shown, delays) = await Follow(Enumerable.Repeat((GpuPreparation)new GpuPreparation.Preparing("d", 1, 9), 100_000), TimeSpan.FromSeconds(2));
        Assert.Equal(10, delays.Count);
        Assert.Equal([Display.Hidden, Display.Preparing, Display.Hidden], shown);
        Assert.Equal(TimeSpan.FromSeconds(600), GpuPreparationMonitor.FollowLimit);
    }
}

/// <summary>The capsule's numbers and details from the engine's <c>first_order</c> (the Mac app's DesignStatus).</summary>
public class StatusCapsuleModelTests
{
    const string Rotational = """
        {"entries": [{"key": "efl", "label": "EFL", "value": 100, "unit": "mm"}, {"key": "bfl", "label": "BFL", "value": 95.5, "unit": "mm"},
                     {"key": "working_f_number", "label": "F/#", "value": 5.0, "unit": ""}, {"key": "total_track", "label": "TT", "value": 112.25, "unit": "mm"}],
         "paraxial_xy": {"symmetry": "rotational", "wavelengths": []}}
        """;

    const string Anamorphic = """
        {"entries": [{"key": "efl", "label": "EFL", "value": 50, "unit": "mm"}, {"key": "bfl", "label": "BFL", "value": 40, "unit": "mm"},
                     {"key": "image_f_number", "label": "F/#", "value": 2.8, "unit": ""}, {"key": "entrance_pupil_diameter", "label": "EPD", "value": 20, "unit": "mm"},
                     {"key": "total_track", "label": "TT", "value": 80, "unit": "mm"}],
         "paraxial_xy": {"symmetry": "meridional", "primary_wavelength": 1, "squeeze_definition": "EFL y / EFL x",
           "warnings": ["the x section has no stop"],
           "wavelengths": [{"wavelength": 0, "wavelength_um": 0.4861, "efl_x": 26, "efl_y": 51, "bfl_x": 21, "bfl_y": 41},
                           {"wavelength": 1, "wavelength_um": 0.5876, "efl_x": 25, "efl_y": 50, "bfl_x": 20, "bfl_y": 40, "squeeze": 2,
                            "doe_power": [{"surface": 3, "power": 0.0012}]}]}}
        """;

    static FirstOrderData Read(string json) => FirstOrderData.Read(JsonNode.Parse(json))!;

    [Fact]
    public void KeyNumbers()
    {
        Assert.Equal([("EFL", Fmt.Value(100, "mm")), ("F/#", Fmt.Value(5, "")), ("TT", Fmt.Value(112.25, "mm"))],
            StatusCapsuleModel.KeyNumbers(Read(Rotational), false));
        Assert.Equal(["EFL y", "Squeeze", "F/#", "TT"], StatusCapsuleModel.KeyNumbers(Read(Anamorphic), true).Select(n => n.Label));
        Assert.Equal("2.00×", StatusCapsuleModel.KeyNumbers(Read(Anamorphic), true)[1].Value);
        Assert.Empty(StatusCapsuleModel.KeyNumbers(null, false));
        // Anamorphic by the surface kinds, without x / y data: the keyed EFL is the y section's.
        var noXy = FirstOrderData.Read(JsonNode.Parse("""{"entries": [{"key": "efl", "value": 50, "unit": "mm"}]}"""));
        Assert.Equal("EFL y", StatusCapsuleModel.KeyNumbers(noXy, true)[0].Label);
    }

    [Fact]
    public void FirstOrderSection()
    {
        var rows = StatusCapsuleModel.FirstOrder(Read(Rotational), false);
        Assert.Equal(["First Order", "Effective focal length", "Back focal length", "Working F/#", "Total track"], rows.Select(r => r.Label));
        Assert.DoesNotContain(rows, r => r.Kind == StatusDetail.Kinds.Xy);

        var ana = StatusCapsuleModel.FirstOrder(Read(Anamorphic), true);
        Assert.Contains(ana, r => r.Label == "Effective focal length (y)");
        Assert.Contains(ana, r => r.Label == "Working F/# (y)");
        Assert.Contains(ana, r => r.Label == "Entrance pupil diameter");
        var efl = Assert.Single(ana, r => r.Kind == StatusDetail.Kinds.Xy && r.Label == "EFL");
        Assert.Equal(("25", "50"), (efl.Value, efl.Value2));
        Assert.Single(ana, r => r.Kind == StatusDetail.Kinds.XyHeader);
        var squeeze = Assert.Single(ana, r => r.Label == "Squeeze");
        Assert.Equal("2.0000×", squeeze.Value);
        Assert.Equal("Squeeze: EFL y / EFL x", squeeze.Help);
        Assert.Equal("Meridional plane", Assert.Single(ana, r => r.Label == "Symmetry").Value);
        Assert.StartsWith("0.0012", Assert.Single(ana, r => r.Label == "DOE power, surface 3").Value);
        Assert.Contains(ana, r => r.Kind == StatusDetail.Kinds.Note && r.Label == "At λ = 0.5876 µm (primary)");
        Assert.Contains(ana, r => r.Kind == StatusDetail.Kinds.Warning && r.Label == "the x section has no stop");

        var noXy = StatusCapsuleModel.FirstOrder(FirstOrderData.Read(JsonNode.Parse("""{"entries": [{"key": "efl", "value": 50, "unit": "mm"}]}""")), true);
        Assert.Contains(noXy, r => r.Kind == StatusDetail.Kinds.Note && r.Label.StartsWith("This design is not rotationally symmetric", StringComparison.Ordinal));
        Assert.Equal("No first-order data yet.", StatusCapsuleModel.FirstOrder(null, false)[1].Label);
    }

    [Fact]
    public void EngineSection()
    {
        var snapshot = new ModelSnapshot("{}", new SnapshotHash("0123456789abcdef0123"), "fedcba9876543210ffff");
        var rows = StatusCapsuleModel.Engine("MKLight engine 0.1.0", snapshot,
            new GpuPreparation.Ready("RTX 4090", 205, 0, 31.4, "read-only folder"));
        Assert.Equal(["Engine", "Engine", "Snapshot", "Glass library", "GPU kernels", "Kernel cache unavailable: the next launch compiles again (read-only folder)"],
            rows.Select(r => r.Label));
        Assert.Equal("0123456789ab", rows[2].Value);
        Assert.Equal("fedcba987654", rows[3].Value);
        Assert.Equal("Ready (205, prepared in 31.4 s)", rows[4].Value);
        Assert.Equal(["Engine", "Engine"], StatusCapsuleModel.Engine("e", null, new GpuPreparation.Idle()).Select(r => r.Label));
    }

    [Fact]
    public void BackendKeptInTheViews()
    {
        var views = JsonNode.Parse("""
            [{"key": "layout", "analysis": "layout", "settings": {}, "ui": {"mode": "section", "arrangement": {"x": 1}}},
             {"key": "p1", "analysis": "spot_diagram", "settings": {}, "ui": {"auto_update": true}}]
            """);
        Assert.Equal(default, StatusCapsuleModel.ReadBackend(views));
        var cuda = new ExecutionSettings(BackendPreference.Gpu, "cuda");
        var kept = StatusCapsuleModel.WithBackend(views, cuda)!;
        Assert.Equal("gpu:cuda", kept[0]!["ui"]!["compute_backend"]!.GetValue<string>());
        Assert.Equal("section", kept[0]!["ui"]!["mode"]!.GetValue<string>());
        Assert.Equal(cuda, StatusCapsuleModel.ReadBackend(kept));
        Assert.Null(views![0]!["ui"]!["compute_backend"]);
        var auto = StatusCapsuleModel.WithBackend(kept, default)!;
        Assert.False(auto[0]!["ui"]!.AsObject().ContainsKey("compute_backend"));
        // No layout entry yet: nothing is written.
        Assert.Null(StatusCapsuleModel.WithBackend(new JsonArray(), cuda));
        Assert.Null(StatusCapsuleModel.WithBackend(null, cuda));
        Assert.Equal(default, StatusCapsuleModel.ReadBackend(null));
    }
}
