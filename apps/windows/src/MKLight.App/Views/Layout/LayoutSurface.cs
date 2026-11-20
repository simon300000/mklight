using Microsoft.UI.Xaml.Automation.Peers;
using Microsoft.UI.Xaml.Controls;

namespace MKLight.App.Views.Layout;

/// <summary>
/// The layout's drawing surface (the section canvas and the 3-D view, and the input over both).
/// A plain panel has no automation peer, so assistive technology would not see the layout at all:
/// this one is an image named by the scene's summary (<c>AutomationProperties.Name</c>, with the
/// details in <c>HelpText</c>), automation id <c>Layout.Canvas</c>.
/// </summary>
public sealed partial class LayoutSurface : Grid
{
    protected override AutomationPeer OnCreateAutomationPeer() => new Peer(this);

    sealed partial class Peer(LayoutSurface owner) : FrameworkElementAutomationPeer(owner)
    {
        protected override AutomationControlType GetAutomationControlTypeCore() => AutomationControlType.Image;
        protected override string GetLocalizedControlTypeCore() => "layout";
        protected override string GetClassNameCore() => nameof(LayoutSurface);
        protected override bool IsContentElementCore() => true;
        protected override bool IsControlElementCore() => true;
        // The canvas draws; it has no automation children of its own.
        protected override IList<AutomationPeer>? GetChildrenCore() => null;
    }
}
