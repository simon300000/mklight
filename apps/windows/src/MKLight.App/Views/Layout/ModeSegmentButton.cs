using Microsoft.UI.Xaml.Automation.Peers;
using Microsoft.UI.Xaml.Automation.Provider;
using Microsoft.UI.Xaml.Controls;

namespace MKLight.App.Views.Layout;

/// <summary>A segment of the layout's view selector: a button that is also a selectable item for automation (the shown view is the selected one).</summary>
sealed partial class ModeSegmentButton(bool selected, Action choose) : Button
{
    public bool Selected { get; } = selected;
    public void Choose() => choose();

    protected override AutomationPeer OnCreateAutomationPeer() => new SegmentPeer(this);

    sealed partial class SegmentPeer(ModeSegmentButton owner) : ButtonAutomationPeer(owner), ISelectionItemProvider
    {
        protected override object? GetPatternCore(PatternInterface pattern) =>
            pattern == PatternInterface.SelectionItem ? this : base.GetPatternCore(pattern);

        public bool IsSelected => owner.Selected;
        public IRawElementProviderSimple? SelectionContainer => null;
        public void Select() => owner.Choose();
        public void AddToSelection() => Select();
        public void RemoveFromSelection() { }
    }
}
