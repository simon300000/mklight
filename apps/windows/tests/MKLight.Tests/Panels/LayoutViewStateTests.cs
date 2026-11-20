using System.Text.Json.Nodes;
using MKLight.App.ViewModels;

namespace MKLight.Tests.Panels;

/// <summary>The layout viewport's saved state (mode, ray colouring, stretch) in the layout entry's <c>ui</c>.</summary>
public class LayoutViewStateTests
{
    [Fact]
    public void WritesAndReadsBack()
    {
        var ui = new JsonObject { ["other"] = 1 };
        new WorkspaceLayout.ViewState("3D", "wavelength", 5).WriteTo(ui);
        Assert.Equal(1, (int)ui["other"]!);
        var back = WorkspaceLayout.ViewState.Read(ui);
        Assert.Equal(new WorkspaceLayout.ViewState("3D", "wavelength", 5), back);
    }

    [Fact]
    public void DefaultsAndUnchosenStretch()
    {
        Assert.Equal(new WorkspaceLayout.ViewState("2D", "field", null), WorkspaceLayout.ViewState.Read([]));
        var ui = new JsonObject { ["mode"] = "bogus", ["stretch"] = 0.5 };
        Assert.Equal(new WorkspaceLayout.ViewState("2D", "field", null), WorkspaceLayout.ViewState.Read(ui));
        ui["stretch"] = 10;
        new WorkspaceLayout.ViewState("2D-xz", "field", null).WriteTo(ui);
        Assert.False(ui.ContainsKey("stretch"));
        Assert.Equal("2D-xz", (string)ui["mode"]!);
    }

    [Fact]
    public void SettingsAndUiSurviveTheViewsRoundTrip()
    {
        var l = new WorkspaceLayout { LayoutSettings = new JsonObject { ["section"] = "xz", ["rays_per_field"] = 7 } };
        new WorkspaceLayout.ViewState("3D", "field", 2).WriteTo(l.LayoutUi);
        var back = WorkspaceLayout.Read(l.ToViews());
        Assert.Equal(7, (int)back.LayoutSettings["rays_per_field"]!);
        Assert.Equal(new WorkspaceLayout.ViewState("3D", "field", 2), WorkspaceLayout.ViewState.Read(back.LayoutUi));
    }
}
