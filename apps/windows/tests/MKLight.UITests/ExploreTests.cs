namespace MKLight.UITests;

/// <summary>Not a check: with MKLIGHT_UI_DUMP=1 as well, writes the app's automation tree and window list to the artefacts folder.</summary>
public sealed class ExploreTests
{
    [Fact]
    public void DumpTree()
    {
        using var app = AppSession.Start();
        if (Environment.GetEnvironmentVariable("MKLIGHT_UI_DUMP") != "1") Assert.Skip("Set MKLIGHT_UI_DUMP=1 to dump the automation tree.");
        Thread.Sleep(3000);
        if (Environment.GetEnvironmentVariable("MKLIGHT_UI_DUMP_ROW") is { Length: > 0 } row)
        {
            Ui.ShowContextMenu(app.Ui.LensRow(int.Parse(row, System.Globalization.CultureInfo.InvariantCulture)));
            Thread.Sleep(1500);
            Directory.CreateDirectory(app.ArtifactsDirectory);
            var pid = app.Window.Properties.ProcessId.Value;
            File.WriteAllText(Path.Combine(app.ArtifactsDirectory, "popups.txt"), string.Join("\n----\n",
                app.Automation.GetDesktop().FindAllChildren(app.Automation.ConditionFactory.ByProcessId(pid)).Select(w => TreeDump.Of(w))));
        }
        Directory.CreateDirectory(app.ArtifactsDirectory);
        File.WriteAllText(Path.Combine(app.ArtifactsDirectory, "tree.txt"), TreeDump.Of(app.Window));
        var windows = Win32.Descendants(app.Hwnd).Select(h =>
        {
            Win32.GetWindowRect(h, out var r);
            return $"{h:X} {Win32.ClassName(h)} visible={Win32.IsWindowVisible(h)} [{r.Left},{r.Top},{r.Right},{r.Bottom}]";
        });
        File.WriteAllLines(Path.Combine(app.ArtifactsDirectory, "windows.txt"), windows);
        app.SaveArtifacts("dump");
    }

    [Fact]
    public void ProbeMouseTargets()
    {
        using var app = AppSession.Start();
        if (Environment.GetEnvironmentVariable("MKLIGHT_UI_DUMP") != "1") Assert.Skip("Set MKLIGHT_UI_DUMP=1 to probe.");
        var log = new List<string>();
        var names = new[] { "ray_fan", "spot_diagram", "footprint", "grid_distortion", "lateral_color" };
        var targets = Win32.Descendants(app.Hwnd).Prepend(app.Hwnd).ToList();
        var k = 0;
        foreach (var h in targets)
            foreach (var mode in new[] { "Post", "Send", "PostPointer", "SendPointer" })
            {
                if (k >= names.Length) break;
                var send = mode.StartsWith("Send");
                var item = app.Ui.SidebarItem(names[k]);
                var r = Ui.Bounds(item);
                var p = new Win32.POINT { X = r.X + r.Width / 2, Y = r.Y + r.Height / 2 };
                var screen = Win32.MakeLParam(p.X, p.Y);
                Win32.ScreenToClient(h, ref p);
                var l = Win32.MakeLParam(p.X, p.Y);
                (uint, nint, nint)[] seq = mode.EndsWith("Pointer")
                    ? [(0x0249u, (nint)(1 | (0x1 | 0x2000) << 16), screen), (0x0245u, (nint)(1 | (0x1 | 0x2000) << 16), screen),
                       (0x0246u, (nint)(1 | (0x1 | 0x4 | 0x10 | 0x2000 | 0x8) << 16), screen), (0x0247u, (nint)(1 | (0x1 | 0x2000) << 16), screen)]
                    : [(Win32.WM_MOUSEMOVE, 0, l), (Win32.WM_LBUTTONDOWN, Win32.MK_LBUTTON, l), (Win32.WM_LBUTTONUP, 0, l)];
                foreach (var (m, w, lp) in seq)
                {
                    if (send) Win32.SendMessage(h, m, w, lp); else Win32.PostMessage(h, m, w, lp);
                    Thread.Sleep(50);
                }
                Thread.Sleep(1500);
                var opened = app.Ui.PaneWithTitle(names[k]) is not null;
                log.Add($"{Win32.ClassName(h)} {mode} →{(opened ? "OPENED" : "nothing")} ({names[k]})");
                if (opened) k++;
            }
        Directory.CreateDirectory(app.ArtifactsDirectory);
        File.WriteAllLines(Path.Combine(app.ArtifactsDirectory, "probe.txt"), log);
    }
}
