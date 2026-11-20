using Microsoft.UI.Xaml;
using MKLight.App.ViewModels;

namespace MKLight.App.Views.Inspector;

/// <summary>
/// An inspector page: Surface and System, then the design jobs (the Mac
/// app's inspector tabs). <paramref name="Create"/> builds the page for a
/// design window; <paramref name="Order"/> sorts the tab bar.
/// </summary>
public sealed record InspectorPageInfo(string Id, string Title, string Glyph, int Order, Func<DesignSession, UIElement> Create);

/// <summary>
/// The registry of inspector pages. Each feature registers its pages from
/// its own files with a <c>[ModuleInitializer]</c>, so the inspector, the
/// surface and system pages and the design jobs never edit the same file:
/// <code>
/// static class OptimizeRegistration
/// {
///     [ModuleInitializer]
///     internal static void Register() =>
///         InspectorPages.Add(new("optimize", "Optimize", "\uE9D9", 20, s => new OptimizePanel(s)));
/// }
/// </code>
/// Ids in use: <c>surface</c>, <c>system</c> (10, 11), <c>optimize</c>, <c>glass_substitution</c>,
/// <c>tolerance</c>, <c>solves</c> (20…23). <see cref="DesignSession.ShowInspectorPage"/> opens one.
/// </summary>
public static class InspectorPages
{
    static readonly List<InspectorPageInfo> Pages = [];

    public static void Add(InspectorPageInfo page)
    {
        Pages.RemoveAll(p => p.Id == page.Id);
        Pages.Add(page);
    }

    public static IReadOnlyList<InspectorPageInfo> All => Pages.OrderBy(p => p.Order).ToList();
}
