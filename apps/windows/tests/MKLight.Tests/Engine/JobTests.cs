// Tests await jobs and their progress without the test's token on purpose: the job's own
// cancellation is what they check.
#pragma warning disable xUnit1051
using System.Diagnostics;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using MKLight.Engine;

namespace MKLight.Tests.Engine;

/// <summary><see cref="AnalysisJob"/> itself, on operations written in the test.</summary>
public class JobTests
{
    static readonly SnapshotHash Hash = new("aa");

    static AnalysisOutput Output(string analysis) => new(
        new RunInfo
        {
            EngineVersion = "test", SnapshotHash = Hash, Analysis = analysis, Settings = new JsonObject(), Backend = "test",
            Precision = Precision.F64Reference,
        },
        new JsonObject { ["ok"] = true });

    static async Task<List<JobProgress>> Drain(AnalysisJob job)
    {
        var updates = new List<JobProgress>();
        await foreach (var p in job.ProgressAsync()) updates.Add(p);
        return updates;
    }

    [Fact]
    public async Task ProgressIsMonotoneAndEndsAtOne()
    {
        var request = new EngineAnalysisRequest("spot_diagram");
        var job = AnalysisJob.Run(request, Hash, async (reporter, token) =>
        {
            for (int k = 1; k <= 4; k++)
            {
                await Task.Delay(5, token);
                reporter.Report(k / 5.0, "Tracing");
                reporter.Report(0.1, null); // never goes back
            }
            return Output("spot_diagram");
        });
        Assert.True(job.Analysis == "spot_diagram" && job.EngineRequest == request && job.SnapshotHash == Hash);
        Assert.False(job.CanStop);
        Assert.False(job.Stop());
        var updates = await Drain(job);
        Assert.True(updates.Count >= 3);
        Assert.True(updates.Zip(updates.Skip(1)).All(p => p.First.Fraction <= p.Second.Fraction), "monotone");
        Assert.Equal(new JobProgress(1, "Done"), updates[^1]);
        Assert.Contains(updates, u => u.Stage == "Tracing");
        var output = await job.ResultAsync();
        Assert.Equal("spot_diagram", output.RunInfo.Analysis);
        Assert.True(job.IsCompleted);
    }

    [Fact]
    public async Task CancellationStopsTheJob()
    {
        var job = AnalysisJob.Run(new EngineAnalysisRequest("spot_diagram"), Hash, async (_, token) =>
        {
            await Task.Delay(TimeSpan.FromSeconds(30), token);
            return Output("spot_diagram");
        });
        var clock = Stopwatch.StartNew();
        await Task.Delay(20);
        job.Cancel();
        job.Cancel();
        var e = await Assert.ThrowsAsync<EngineException>(() => job.ResultAsync());
        Assert.Equal(EngineErrorKind.Cancelled, e.Kind);
        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(5));
        await Drain(job); // the progress sequence ends too
    }

    /// <summary>Abandoning the wait (its token) does not cancel the job.</summary>
    [Fact]
    public async Task AwaiterCancellationDoesNotCancelTheJob()
    {
        var job = AnalysisJob.Run(new EngineAnalysisRequest("first_order"), Hash, async (_, token) =>
        {
            await Task.Delay(50, token);
            return Output("first_order");
        });
        using var waiter = new CancellationTokenSource();
        var wait = job.ResultAsync(waiter.Token);
        waiter.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => wait);
        var output = await job.ResultAsync();
        Assert.Equal("first_order", output.RunInfo.Analysis);
    }

    [Fact]
    public async Task StopIsForwardedWhenTheJobCanStop()
    {
        int stops = 0;
        var job = AnalysisJob.Run(new EngineAnalysisRequest("optimize"), Hash, (_, _) => Task.FromResult(Output("optimize")),
            onStop: () => Interlocked.Increment(ref stops) > 0);
        Assert.True(job.CanStop);
        Assert.True(job.Stop());
        Assert.Equal(1, stops);
        await job.ResultAsync();
    }

    /// <summary>Progress reported from concurrent tasks (how engine jobs fan out over fields and wavelengths) still arrives in non-decreasing order.</summary>
    [Fact]
    public async Task ConcurrentProgressStaysMonotone()
    {
        for (int round = 0; round < 20; round++)
        {
            var job = AnalysisJob.Run(new EngineAnalysisRequest("first_order"), Hash, async (r, _) =>
            {
                await Task.WhenAll(Enumerable.Range(0, 8).Select(child => Task.Run(() =>
                {
                    for (int k = 0; k < 200; k++) r.Report((double)(k * 8 + child) / 1600);
                })));
                return Output("first_order");
            });
            var updates = await Drain(job);
            Assert.True(updates.Zip(updates.Skip(1)).All(p => p.First.Fraction <= p.Second.Fraction));
            await job.ResultAsync();
        }
    }

    [Fact]
    public async Task FailuresPassThrough()
    {
        var job = AnalysisJob.Run(new EngineAnalysisRequest("x"), Hash,
            (_, _) => throw EngineException.Engine(EngineErrorCode.Compute, "diverged"));
        var e = await Assert.ThrowsAsync<EngineException>(() => job.ResultAsync());
        Assert.Equal(EngineErrorCode.Compute, e.Code);
    }
}

/// <summary>The C# constants of the C ABI against <c>crates/mklight-ffi/include/mklight.h</c>.</summary>
public class HeaderConstantsTests
{
    static readonly string Header = File.ReadAllText(EngineTestSupport.RepoPath("crates", "mklight-ffi", "include", "mklight.h"));

    static Dictionary<string, long> Defines() =>
        Regex.Matches(Header, @"^#define\s+(MK_\w+)\s+(\d+)", RegexOptions.Multiline)
            .ToDictionary(m => m.Groups[1].Value, m => long.Parse(m.Groups[2].Value));

    /// <summary><c>InvalidArgument</c> → <c>INVALID_ARGUMENT</c>.</summary>
    static string Snake(string name) => Regex.Replace(name, "(?<=[a-z0-9])([A-Z])", "_$1").ToUpperInvariant();

    static void Matches<T>(string prefix, Dictionary<string, long> defines, IEnumerable<string>? except = null) where T : struct, Enum
    {
        var skip = except?.ToHashSet() ?? [];
        var expected = defines.Where(d => d.Key.StartsWith(prefix, StringComparison.Ordinal) && !skip.Contains(d.Key))
            .ToDictionary(d => d.Key, d => d.Value);
        var actual = Enum.GetValues<T>().ToDictionary(v => prefix + Snake(v.ToString()), v => Convert.ToInt64(v));
        Assert.Equal(expected.OrderBy(kv => kv.Key), actual.OrderBy(kv => kv.Key));
    }

    [Fact]
    public void StatusCodesDTypesAndJobStatesMatchTheHeader()
    {
        var defines = Defines();
        Assert.Equal(Native.AbiVersionMajor, (uint)defines["MK_ABI_VERSION_MAJOR"]);
        Assert.True((uint)defines["MK_ABI_VERSION_MINOR"] >= Native.AbiVersionMinor);
        Assert.Equal(Native.Ok, (int)defines["MK_OK"]);
        Matches<EngineErrorCode>("MK_ERR_", defines);
        Matches<BufferDType>("MK_DTYPE_", defines);
        Matches<JobState>("MK_JOB_", defines);
    }

    /// <summary>Every function bound with <c>LibraryImport</c> is declared in the header.</summary>
    [Fact]
    public void EveryBoundFunctionIsInTheHeader()
    {
        var bound = typeof(Native).GetMethods(System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public)
            .Where(m => m.Name.StartsWith("mk_", StringComparison.Ordinal)).Select(m => m.Name).Distinct().ToArray();
        Assert.True(bound.Length >= 40, $"{bound.Length} functions bound");
        foreach (var name in bound) Assert.Matches(new Regex(@"\b" + name + @"\("), Header);
    }
}
