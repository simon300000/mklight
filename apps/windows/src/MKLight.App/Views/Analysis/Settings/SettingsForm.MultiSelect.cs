using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using MKLight.App.Support;
using static MKLight.App.Views.Analysis.Settings.FormControls;

namespace MKLight.App.Views.Analysis.Settings;

// Field, wavelength and surface selections as a drop-down menu of check marks.
public sealed partial class SettingsForm
{
    readonly Dictionary<string, DropDownButton> _multiSelects = [];

    const string PresenterStyle = """
        <Style xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation" TargetType="MenuFlyoutPresenter">
            <Setter Property="Background" Value="{ThemeResource AcrylicInAppFillColorDefaultBrush}" />
        </Style>
        """;

    /// <summary>The menu button of a top-level selection setting (launch actions open it for screenshots).</summary>
    public DropDownButton? MultiSelectFor(string setting) => _multiSelects.GetValueOrDefault(setting);

    /// <summary>
    /// A selection of the system's fields, wavelengths or surfaces: a drop-down
    /// button titled "All", "2", "1, 3" or "3 of 5" whose menu has "All" (and
    /// "Primary"), then one check mark per option. Checking options keeps the
    /// menu open, so several are chosen in one go; "All" and "Primary" close it.
    /// Each change is committed at once. Without the design's options it is
    /// a text field of option numbers.
    /// </summary>
    UIElement MultiSelect(Slot slot, IndexDomain domain, SelectionEncoding encoding, bool allowsPrimary)
    {
        var f = slot.Field;
        var options = _ctx.Options(domain);
        // Menus number fields and wavelengths from 1, surfaces by their index.
        var offset = domain == IndexDomain.Surfaces ? 0 : 1;
        var noneHelp = encoding == SelectionEncoding.List ? "\n" + Loc.S("Analysis_Settings_NoneCheckedHelp") : "";
        if (options.Count == 0)
        {
            var box = new CommitBox(
                _ => IndexSelection.Read(slot.Current, encoding).Text(offset),
                text => IndexSelection.Parse(text, allowsPrimary, offset) is { } s ? new Edit(s.Write(encoding)) : null,
                e => slot.Write(e, false), 110, number: false) { PlaceholderText = "all", TextAlignment = TextAlignment.Left };
            ToolTipService.SetToolTip(box, Help(f) + "\n" + Loc.S("Analysis_Settings_NumbersHelp"));
            return box;
        }

        var label = SettingEntry.Label(f);
        var selection = IndexSelection.Read(slot.Current, encoding);
        // In the window, as the form's other popups (and so in window captures), on in-app
        // acrylic: the menu's system backdrop is only drawn behind a popup window of its own.
        var menu = new MenuFlyout
        {
            Placement = FlyoutPlacementMode.BottomEdgeAlignedLeft,
            ShouldConstrainToRootBounds = true,
            MenuFlyoutPresenterStyle = (Style)Microsoft.UI.Xaml.Markup.XamlReader.Load(PresenterStyle),
        };
        var button = Menu("", menu, Help(f) + noneHelp);
        var all = new ToggleMenuFlyoutItem { Text = Loc.S("Analysis_Settings_All") };
        var primary = allowsPrimary ? new ToggleMenuFlyoutItem { Text = Loc.S("Analysis_Settings_Primary") } : null;
        var items = options.Select(o => (o.Index, Item: new ToggleMenuFlyoutItem { Text = o.Title })).ToList();

        void Show()
        {
            all.IsChecked = selection.Kind == SelectionKind.All;
            if (primary is not null) primary.IsChecked = selection.Kind == SelectionKind.Primary;
            foreach (var (k, item) in items) item.IsChecked = selection.IsChecked(k);
            button.Content = selection.Title(options);
            AutomationProperties.SetHelpText(button, Loc.F("Analysis_Settings_SelectionHelp", label, selection.AccessibleValue(options)));
        }

        // An option toggled keeps the menu open: its click comes before the menu's
        // closing requests (more than one), which are refused until the click is handled.
        var keepOpen = false;
        menu.Closing += (_, e) => e.Cancel = keepOpen;
        void Choose(IndexSelection next, bool stayOpen)
        {
            if (stayOpen && !keepOpen)
            {
                keepOpen = true;
                DispatcherQueue.TryEnqueue(Microsoft.UI.Dispatching.DispatcherQueuePriority.Low, () => keepOpen = false);
            }
            selection = next;
            Show();
            slot.Write(new Edit(next.Write(encoding)), false);
        }

        all.Click += (_, _) => Choose(IndexSelection.All, false);
        menu.Items.Add(all);
        if (primary is not null)
        {
            primary.Click += (_, _) => Choose(IndexSelection.Primary, false);
            menu.Items.Add(primary);
        }
        menu.Items.Add(new MenuFlyoutSeparator());
        foreach (var (k, item) in items)
        {
            item.Click += (_, _) => Choose(selection.Toggle(k), true);
            menu.Items.Add(item);
        }
        Microsoft.UI.Xaml.Documents.Typography.SetNumeralAlignment(button, FontNumeralAlignment.Tabular);
        Show();
        _multiSelects[slot.Path] = button;
        return button;
    }
}
