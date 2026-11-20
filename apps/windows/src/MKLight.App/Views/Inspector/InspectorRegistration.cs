using System.Runtime.CompilerServices;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Support;

namespace MKLight.App.Views.Inspector;

/// <summary>
/// Registers the Surface and System pages and their launch actions:
/// <c>--do inspector-surface[=i]</c> (the Surface page, optionally selecting surface i; <c>--select i</c> works too),
/// <c>--do inspector-system</c>, and <c>--do inspector-scroll=px</c> (scrolls the shown page, for screenshots of
/// its lower sections), <c>--do inspector-set=key:value</c> (a scripted edit of one row).
/// </summary>
static class InspectorRegistration
{
    [ModuleInitializer]
    internal static void Register()
    {
        InspectorPages.Add(new("surface", "Surface", "Surface", 10, s => new SurfacePage(s)));
        InspectorPages.Add(new("system", "System", "System", 11, s => new SystemPage(s)));

        AppHooks.LaunchActions["inspector-surface"] = (w, arg) =>
        {
            if (int.TryParse(arg, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out var i))
                w.Session.SelectedSurface = i;
            w.InspectorPanel.Show("surface");
            return Settle();
        };
        AppHooks.LaunchActions["inspector-system"] = (w, _) =>
        {
            w.InspectorPanel.Show("system");
            return Settle();
        };
        // A scripted edit on the shown page: --do inspector-set=medium.thickness:8 (a number row types and enters
        // the text; a choice row takes the option with that id, e.g. aperture.kind:object_space_na).
        AppHooks.LaunchActions["inspector-set"] = async (w, arg) =>
        {
            await Settle();
            var colon = arg?.IndexOf(':') ?? -1;
            if (colon < 0 || w.InspectorPanel.Page is not { } id || w.InspectorPanel.PageView(id) is not { } view) return;
            var form = view switch { SurfacePage s => s.Form, SystemPage s => s.Form, _ => null };
            if (form?.Script(arg![..colon], arg[(colon + 1)..]) != true) App.Log($"inspector-set: no row {arg}");
            await Settle();
        };
        AppHooks.LaunchActions["inspector-scroll"] = async (w, arg) =>
        {
            await Settle();
            if (w.InspectorPanel.Page is { } id && w.InspectorPanel.PageView(id) is { } view && Find<ScrollViewer>(view) is { } scroll
                && double.TryParse(arg, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var y))
                scroll.ChangeView(null, y, null, disableAnimation: true);
            await Settle();
        };
    }

    /// <summary>Lets the page lay out its form (the refresh is queued on the dispatcher).</summary>
    static Task Settle() => Task.Delay(300);

    static T? Find<T>(DependencyObject d) where T : DependencyObject
    {
        if (d is T t) return t;
        for (var i = 0; i < VisualTreeHelper.GetChildrenCount(d); i++)
            if (Find<T>(VisualTreeHelper.GetChild(d, i)) is { } found) return found;
        return null;
    }

    /// <summary>A page that has not arrived yet (the design jobs register theirs).</summary>
    internal static UIElement Placeholder(string title) => new Border
    {
        Style = InspectorStyles.Get("InspectorCard"),
        Margin = new Thickness(12, 8, 12, 12),
        Child = new TextBlock
        {
            Text = title, Style = InspectorStyles.Get("InspectorSecondary"),
            HorizontalAlignment = HorizontalAlignment.Center, VerticalAlignment = VerticalAlignment.Center,
        },
    };
}
