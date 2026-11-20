using System.Runtime.CompilerServices;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Support;

namespace MKLight.App.Views.Jobs;

/// <summary>
/// <c>--do job-figures</c>: waits until the job pages' layout thumbnails and glass map (computed asynchronously by
/// <see cref="DesignFigures"/>) are in, so a following <c>--screenshot</c> shows them; <c>--do job-scroll</c> scrolls
/// the open job page to its end.
/// </summary>
static class JobFigureLaunchActions
{
    [ModuleInitializer]
    internal static void Register()
    {
        AppHooks.LaunchActions["job-figures"] = async (_, _) =>
        {
            await Task.Delay(400); // the page rebuilds and asks for its figures on the next UI turns
            var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(60);
            var idle = 0;
            while (DateTime.UtcNow < deadline && idle < 3)
            {
                idle = DesignFigures.Pending == 0 ? idle + 1 : 0;
                await Task.Delay(150);
            }
            await Task.Delay(250); // drawn
        };
        // `job-scroll`: scrolls the open job page to its end (the glass map under a candidate).
        AppHooks.LaunchActions["job-scroll"] = async (w, _) =>
        {
            await Task.Delay(200);
            if (Find<JobPanelFrame>(w.Content) is { } frame) frame.Scroller.ChangeView(null, frame.Scroller.ScrollableHeight, null, disableAnimation: true);
            await Task.Delay(200);
        };
    }

    static T? Find<T>(DependencyObject? root) where T : DependencyObject
    {
        if (root is null) return null;
        if (root is T hit) return hit;
        for (var i = 0; i < VisualTreeHelper.GetChildrenCount(root); i++)
            if (Find<T>(VisualTreeHelper.GetChild(root, i)) is { } found) return found;
        return null;
    }
}
