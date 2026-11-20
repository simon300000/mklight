using System.Runtime.CompilerServices;
using Microsoft.UI.Xaml;

namespace MKLight.App.Support;

/// <summary>
/// Showing and hiding floating labels (the layout's preview label and stale card) with the Mac
/// app's short fade; with Reduce Motion (<see cref="Motion.Enabled"/> false) they appear and go at
/// once. A hidden element is collapsed once its fade has run, so it takes no clicks or tab stops.
/// </summary>
public static class MotionFade
{
    static readonly TimeSpan Duration = TimeSpan.FromMilliseconds(250);
    static readonly ConditionalWeakTable<UIElement, object> Pending = new();

    public static void Show(UIElement e, bool show)
    {
        var animate = Motion.Enabled;
        if (show)
        {
            Pending.Remove(e);
            if (animate && e.Visibility == Visibility.Collapsed) { e.OpacityTransition = null; e.Opacity = 0; }
            e.OpacityTransition = animate ? new ScalarTransition { Duration = Duration } : null;
            e.Visibility = Visibility.Visible;
            e.Opacity = 1;
            return;
        }
        if (e.Visibility == Visibility.Collapsed) return;
        if (!animate)
        {
            e.OpacityTransition = null;
            e.Opacity = 1;
            e.Visibility = Visibility.Collapsed;
            return;
        }
        e.OpacityTransition = new ScalarTransition { Duration = Duration };
        e.Opacity = 0;
        var token = new object();
        Pending.AddOrUpdate(e, token);
        var timer = e.DispatcherQueue.CreateTimer();
        timer.Interval = Duration;
        timer.IsRepeating = false;
        timer.Tick += (_, _) =>
        {
            if (Pending.TryGetValue(e, out var t) && ReferenceEquals(t, token))
            {
                Pending.Remove(e);
                e.Visibility = Visibility.Collapsed;
            }
        };
        timer.Start();
    }

    [ModuleInitializer]
    internal static void Register()
    {
        // `--do reduce-motion`: the app as with Windows' Animation effects off (screenshots, checks);
        // `reduce-motion=system` follows Windows again.
        AppHooks.LaunchActions["reduce-motion"] = (_, arg) =>
        {
            Motion.Override(arg == "system" ? null : false);
            return Task.CompletedTask;
        };
    }
}
