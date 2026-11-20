using System.Globalization;
using System.Runtime.CompilerServices;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Support;
using MKLight.App.Views.Analysis.Settings;

namespace MKLight.App.Views.Analysis.Results;

/// <summary>
/// Launch actions of the image simulation (<c>--do name=arg</c>), for screenshots and tests:
/// <list type="bullet">
/// <item><c>image-simulation-import=&lt;file&gt;</c>: use an image file as the source of every Image Simulation pane (as Choose Image… does).</item>
/// <item><c>image-simulation-fov=&lt;percent&gt;|full</c>, <c>image-simulation-center=&lt;hx&gt;,&lt;hy&gt;</c>, <c>image-simulation-size=&lt;n&gt;</c>: the patch's field of view, centre and resolution.</item>
/// <item><c>image-simulation-zoom=&lt;factor&gt;[,&lt;u&gt;,&lt;v&gt;]</c>: zoom the images (about the image point u, v in 0 … 1 from the top left).</item>
/// </list>
/// </summary>
static class ImageSimulationLaunchActions
{
    static IEnumerable<ViewModels.AnalysisPaneModel> Panes(Windows.DesignWindow w) => w.Session.Workspace.Panes.Where(p => p.Analysis == "image_simulation");

    static double[] Numbers(string? arg) =>
        (arg ?? "").Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries).Select(s => double.Parse(s, CultureInfo.InvariantCulture)).ToArray();

    [ModuleInitializer]
    internal static void Register()
    {
        AppHooks.LaunchActions["image-simulation-import"] = async (w, arg) =>
        {
            foreach (var p in Panes(w).ToList())
                if (!await ImageSimulationFiles.UseAsync(p, arg ?? "")) throw new InvalidOperationException($"{arg} could not be read as an image");
        };
        AppHooks.LaunchActions["image-simulation-fov"] = (w, arg) =>
        {
            var e = arg == "full" ? ImageSimulationSettings.FullExtent : Numbers(arg)[0] / 100;
            foreach (var p in Panes(w).ToList()) p.SetSettings(ImageSimulationSettings.WithFieldExtent(p.Settings, e));
            return Task.CompletedTask;
        };
        AppHooks.LaunchActions["image-simulation-center"] = (w, arg) =>
        {
            var c = Numbers(arg);
            foreach (var p in Panes(w).ToList()) p.SetSettings(ImageSimulationSettings.WithFieldCenter(p.Settings, (c[0], c[1])));
            return Task.CompletedTask;
        };
        AppHooks.LaunchActions["image-simulation-size"] = async (w, arg) =>
        {
            var n = (int)Numbers(arg)[0];
            foreach (var p in Panes(w).ToList()) p.SetSettings(await ImageSimulationFiles.WithSizeAsync(p, n));
        };
        AppHooks.LaunchActions["image-simulation-zoom"] = async (w, arg) =>
        {
            var a = Numbers(arg);
            // The views are made when the results arrive: give the layout a moment.
            await Task.Delay(300);
            foreach (var v in Descendants(w.RootElement).OfType<ImageSimulationView>())
                v.SetZoom(a.Length > 0 ? a[0] : 1, a.Length > 2 ? a[1] : null, a.Length > 2 ? a[2] : null);
        };
    }

    static IEnumerable<DependencyObject> Descendants(DependencyObject root)
    {
        var n = VisualTreeHelper.GetChildrenCount(root);
        for (var i = 0; i < n; i++)
        {
            var c = VisualTreeHelper.GetChild(root, i);
            yield return c;
            foreach (var d in Descendants(c)) yield return d;
        }
    }
}
