using MKLight.Engine;
using MKLight.Model;

namespace MKLight.App.Support;

/// <summary>The pure parts of <c>--do bench-panes</c> (BenchPanes.cs): its argument and the unique tiny edit.</summary>
public static class BenchPlan
{
    /// <summary>"10:cuda" → (10, gpu:cuda); "5" → (5, null: the window's backend).</summary>
    public static (int Runs, ExecutionSettings? Backend) Parse(string? arg)
    {
        var parts = (arg ?? "").Split(':', 2);
        var runs = int.TryParse(parts[0], System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out var n) && n > 0 ? n : 5;
        if (parts.Length < 2 || parts[1].Length == 0) return (runs, null);
        var name = parts[1] is "auto" or "cpu_reference" or "gpu" || parts[1].StartsWith("gpu:", StringComparison.Ordinal) ? parts[1] : "gpu:" + parts[1];
        return (runs, ExecutionSettings.FromWireName(name));
    }

    /// <summary>The design with its first curved surface's curvature changed by one to two parts per billion, unique per serial.</summary>
    public static OpticalSystem Edited(OpticalSystem system, int serial)
    {
        for (var i = 1; i < system.Surfaces.Count; i++)
        {
            var r = system.Surfaces[i].Radius;
            if (!double.IsFinite(r) || r == 0) continue;
            var relative = (serial % 2 == 0 ? 1 : -1) * (1 + serial * 1e-4) * 1e-9;
            return system.WithSurface(i, s => s.WithRadius(r / (1 + relative))); // curvature × (1 + relative)
        }
        return system;
    }

    /// <summary>A path argument without the quotes a shell may have passed through: <c>"C:\a b\x.jsonl"</c> becomes <c>C:\a b\x.jsonl</c>.</summary>
    public static string CleanPath(string arg) => arg.Trim().Trim('"', '\'').Trim();
}
