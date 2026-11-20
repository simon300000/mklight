using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Automation.Peers;
using Microsoft.UI.Xaml.Automation.Provider;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Support;
using MKLight.App.ViewModels;

namespace MKLight.App.Views;

/// <summary>
/// Wires the lens data panel to its window's panels (the More menu moves and hides it) and registers its launch
/// actions (UI testing and screenshots):
/// <list type="bullet">
/// <item><c>--do lens-columns=comment:180,radius:140</c>: holds columns at these widths (stable keys); <c>=auto</c> resets them.</item>
/// <item><c>--do lens-fit[=key]</c>: fits one column (or all) to its content.</item>
/// <item><c>--do lens-more</c>: opens the header's More menu.</item>
/// <item><c>--do lens-type-menu=i</c>: opens row i's type menu.</item>
/// <item><c>--do lens-menu=i</c>: opens row i's context menu.</item>
/// <item><c>--do lens-inspect=i</c>: Show in Inspector for row i.</item>
/// <item><c>--do lens-set=i.key=text</c>: sets a cell's value through its automation peer (as UI Automation would).</item>
/// <item><c>--do lens-automation=file</c>: writes every lens cell's, divider's and header's automation id, name and value to a file.</item>
/// </list>
/// </summary>
static class LensTableRegistration
{
    [ModuleInitializer]
    internal static void Register()
    {
        AppHooks.WindowCreated += w => w.Panels.LensData.Panels = w.Panels;
        var a = AppHooks.LaunchActions;
        a["lens-columns"] = (w, arg) =>
        {
            var lens = w.Panels.LensData;
            if (arg is null or "auto") lens.ResetColumnWidths();
            else lens.SetColumnWidths(arg.Split(',').Select(p => p.Split(':')).Where(p => p.Length == 2)
                .Select(p => (p[0].Trim(), double.Parse(p[1], CultureInfo.InvariantCulture))));
            return Settle();
        };
        a["lens-fit"] = (w, arg) =>
        {
            var lens = w.Panels.LensData;
            if (arg is not null && LensColumnWidths.FromKey(arg) is { } c) lens.FitColumn(Array.IndexOf(LensColumns.Visible, c));
            else lens.FitColumns();
            return Settle();
        };
        a["lens-more"] = (w, _) => OpenMenu(w, lens => lens.ShowMoreMenu());
        a["lens-type-menu"] = (w, arg) => OpenMenu(w, lens => lens.ShowTypeMenu(Row(arg)));
        a["lens-menu"] = (w, arg) => OpenMenu(w, lens => lens.ShowRowMenu(Row(arg), LensColumn.Radius));
        a["lens-inspect"] = (w, arg) =>
        {
            w.Panels.LensData.ShowInInspector(Row(arg));
            return Settle();
        };
        a["lens-set"] = (w, arg) =>
        {
            // i.key=text
            var eq = arg?.IndexOf('=') ?? -1;
            if (eq < 0) return Task.CompletedTask;
            var id = $"LensCell.{arg![..eq]}";
            if (Find(w.Panels.LensData, id) is { } cell && FrameworkElementAutomationPeer.CreatePeerForElement(cell) is IValueProvider value)
            {
                try { value.SetValue(arg[(eq + 1)..]); }
                catch (ArgumentException) { /* rejected: the toast says why */ }
            }
            return Settle();
        };
        a["lens-automation"] = (w, arg) =>
        {
            File.WriteAllText(Path.GetFullPath(arg ?? "lens-automation.txt"), Describe(w.Panels.LensData));
            return Task.CompletedTask;
        };
    }

    static int Row(string? arg) => int.TryParse(arg, NumberStyles.Integer, CultureInfo.InvariantCulture, out var i) ? i : 1;

    static Task Settle() => Task.Delay(1000);

    /// <summary>Opens a menu inside the window (so the capture has it), focused as a click leaves it: no keyboard focus ring.</summary>
    static async Task OpenMenu(Windows.DesignWindow w, Action<LensDataPanel> open)
    {
        var lens = w.Panels.LensData;
        lens.MenusInWindow = true;
        open(lens);
        await Task.Delay(300);
        var popups = VisualTreeHelper.GetOpenPopupsForXamlRoot(lens.XamlRoot);
        // The capture waits for the panes: the menu stays open even if another app takes the activation meanwhile.
        foreach (var p in popups) p.IsLightDismissEnabled = false;
        popups.SelectMany(p => p.Child is { } c ? Descendants(c).OfType<Microsoft.UI.Xaml.Controls.Control>() : [])
            .FirstOrDefault(c => c.IsTabStop && c.IsEnabled)?.Focus(FocusState.Pointer);
        await Settle();
    }

    static FrameworkElement? Find(DependencyObject root, string automationId) =>
        Descendants(root).OfType<FrameworkElement>().FirstOrDefault(e => AutomationProperties.GetAutomationId(e) == automationId);

    static IEnumerable<DependencyObject> Descendants(DependencyObject root)
    {
        var n = VisualTreeHelper.GetChildrenCount(root);
        for (var i = 0; i < n; i++)
        {
            var child = VisualTreeHelper.GetChild(root, i);
            yield return child;
            foreach (var d in Descendants(child)) yield return d;
        }
    }

    /// <summary>Every element of the table with an automation id, as its peer exposes it (control type, name, patterns, value).</summary>
    static string Describe(LensDataPanel lens)
    {
        var sb = new StringBuilder();
        foreach (var e in Descendants(lens).OfType<FrameworkElement>())
        {
            var id = AutomationProperties.GetAutomationId(e);
            if (string.IsNullOrEmpty(id)) continue;
            var peer = FrameworkElementAutomationPeer.CreatePeerForElement(e);
            sb.Append(id).Append(" | ").Append(peer?.GetAutomationControlType()).Append(" | ").Append(peer?.GetName());
            if (peer?.GetPattern(PatternInterface.Value) is IValueProvider v) sb.Append(" | value=").Append(v.Value).Append(v.IsReadOnly ? " (read-only)" : "");
            if (peer?.GetPattern(PatternInterface.RangeValue) is IRangeValueProvider r)
                sb.Append(" | range=").Append(r.Value.ToString(CultureInfo.InvariantCulture)).Append(" in ")
                  .Append(r.Minimum.ToString(CultureInfo.InvariantCulture)).Append('…').Append(r.Maximum.ToString(CultureInfo.InvariantCulture));
            if (peer?.GetPattern(PatternInterface.Invoke) is not null) sb.Append(" | invoke");
            if (peer?.GetPattern(PatternInterface.SelectionItem) is ISelectionItemProvider s) sb.Append(" | selected=").Append(s.IsSelected);
            sb.Append('\n');
        }
        return sb.ToString();
    }
}
