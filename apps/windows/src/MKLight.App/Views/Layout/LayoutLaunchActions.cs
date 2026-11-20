using System.Globalization;
using System.Runtime.CompilerServices;
using MKLight.App.Support;
using MKLight.App.Views.Viewport3D;

namespace MKLight.App.Views.Layout;

/// <summary>
/// Launch actions of the layout pane (<c>--do name[=arg]</c>): <c>layout-3d</c>, <c>layout-xz</c>,
/// <c>layout-footprints</c>, <c>layout-mesh</c>, <c>layout-stretch=N</c>, <c>layout-yaw=rad</c>,
/// <c>layout-pitch=rad</c>, <c>layout-coloring=wavelength</c> and <c>layout-capture</c> (shows the
/// 3-D frame rendered offscreen, for window captures).
/// </summary>
static class LayoutLaunchActions
{
    [ModuleInitializer]
    internal static void Register()
    {
        var a = AppHooks.LaunchActions;
        a["layout-3d"] = (w, _) => { w.Panels.Layout.Mode = ViewportMode.Orbit3D; return Task.CompletedTask; };
        a["layout-xz"] = (w, _) => { w.Panels.Layout.Mode = ViewportMode.SectionXZ; return Task.CompletedTask; };
        a["layout-yz"] = (w, _) => { w.Panels.Layout.Mode = ViewportMode.SectionYZ; return Task.CompletedTask; };
        a["layout-mesh"] = (w, _) => { w.Panels.Layout.ShowMesh = true; return Task.CompletedTask; };
        a["layout-footprints"] = async (w, _) =>
        {
            w.Panels.Layout.ShowFootprints = true;
            await w.Panels.Layout.FootprintsReady;
        };
        a["layout-stretch"] = (w, arg) =>
        {
            w.Panels.Layout.Stretch = double.Parse(arg ?? "1", CultureInfo.InvariantCulture);
            return Task.CompletedTask;
        };
        a["layout-yaw"] = (w, arg) => Orbit(w.Panels.Layout, yaw: arg);
        a["layout-pitch"] = (w, arg) => Orbit(w.Panels.Layout, pitch: arg);
        a["layout-coloring"] = (w, arg) =>
        {
            w.Panels.Layout.Coloring = arg == "wavelength" ? RayColoring.Wavelength : RayColoring.Field;
            return Task.CompletedTask;
        };
        a["layout-capture"] = async (w, _) =>
        {
            // Let the layout's last result and the 3-D view settle, then freeze a frame.
            await Task.Delay(300);
            w.Panels.Layout.FreezeFrameForCapture();
        };
    }

    static Task Orbit(LayoutPanel layout, string? yaw = null, string? pitch = null)
    {
        var c = layout.Camera;
        if (yaw is not null) c.Yaw = double.Parse(yaw, CultureInfo.InvariantCulture);
        if (pitch is not null) c.Pitch = double.Parse(pitch, CultureInfo.InvariantCulture);
        layout.Camera = c;
        layout.Fit();
        return Task.CompletedTask;
    }
}
