using System.Diagnostics;
using System.Runtime.CompilerServices;
using FlaUI.Core;
using FlaUI.Core.AutomationElements;
using FlaUI.UIA3;

namespace MKLight.UITests;

/// <summary>
/// One running app under test: launched with launch options and a scratch support directory,
/// attached through UI Automation, killed at the end. Opt-in: <see cref="Start"/> skips the
/// test unless <c>MKLIGHT_UI_TESTS=1</c>.
/// </summary>
sealed class AppSession : IDisposable
{
    public const string Gate = "MKLIGHT_UI_TESTS";

    readonly Process _process;
    readonly string _scratch;
    readonly string _name;

    public UIA3Automation Automation { get; }
    public Window Window { get; }
    public nint Hwnd { get; }
    public WindowMouse Mouse { get; }
    public Ui Ui { get; }

    AppSession(Process process, string scratch, string name)
    {
        _process = process;
        _scratch = scratch;
        _name = name;
        Automation = new UIA3Automation();
        // By id: FlaUI's Application owns (and disposes) the Process it attaches to.
        var app = Application.Attach(process.Id);
        // UIA calls can time out while the app is still starting (and the machine is busy): try again.
        Window = Ui.Wait(() =>
        {
            try { return app.GetMainWindow(Automation, TimeSpan.FromSeconds(5)); }
            catch (TimeoutException) { return null; }
        }, "the app's main window", TimeSpan.FromSeconds(90));
        Hwnd = Window.Properties.NativeWindowHandle.Value;
        Mouse = new WindowMouse(Hwnd);
        Ui = new Ui(this);
    }

    /// <summary>Skips the test unless UI tests are enabled; otherwise launches the app (double Gauss, 1600×1000, light).</summary>
    public static AppSession Start([CallerMemberName] string name = "", params string[] extra)
    {
        if (Environment.GetEnvironmentVariable(Gate) != "1")
            Assert.Skip($"UI tests drive the running app; set {Gate}=1 to run them (apps/windows/tests/MKLight.UITests/README.md).");
        var exe = AppPath();
        if (!File.Exists(exe)) Assert.Fail($"The app is not built: {exe} (build MKLight.slnx -p:Platform=x64, or set MKLIGHT_APP).");
        var scratch = Path.Combine(Path.GetTempPath(), "mklight-ui-tests", Guid.NewGuid().ToString("N")[..12]);
        Directory.CreateDirectory(scratch);
        // Its output is read here, so the app never holds the test runner's console handles open.
        var info = new ProcessStartInfo(exe)
        {
            UseShellExecute = false, WorkingDirectory = Path.GetDirectoryName(exe)!,
            RedirectStandardOutput = true, RedirectStandardError = true,
        };
        foreach (var a in (string[])["--sample", "double_gauss", "--size", "1600x1000", "--theme", "light", .. extra]) info.ArgumentList.Add(a);
        info.Environment["MKLIGHT_SUPPORT_DIRECTORY"] = scratch;
        var process = Process.Start(info) ?? throw new InvalidOperationException($"Could not start {exe}");
        process.OutputDataReceived += (_, _) => { };
        process.ErrorDataReceived += (_, _) => { };
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        try
        {
            var session = new AppSession(process, scratch, name);
            session.Ui.WaitForLensRows();
            return session;
        }
        catch
        {
            Stop(process);
            throw;
        }
    }

    /// <summary><c>MKLIGHT_APP</c>, or the x64 Debug build next to this checkout.</summary>
    static string AppPath()
    {
        if (Environment.GetEnvironmentVariable("MKLIGHT_APP") is { Length: > 0 } p) return Path.GetFullPath(p);
        // bin/<config>/<tfm>/ of this project → apps/windows
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "MKLight.slnx"))) dir = dir.Parent;
        var root = dir?.FullName ?? throw new InvalidOperationException("apps/windows not found above the test binaries; set MKLIGHT_APP.");
        var debug = Path.Combine(root, "src", "MKLight.App", "bin", "x64", "Debug");
        return Directory.Exists(debug)
            ? Directory.EnumerateFiles(debug, "MKLight.exe", SearchOption.AllDirectories).FirstOrDefault() ?? Path.Combine(debug, "MKLight.exe")
            : Path.Combine(debug, "MKLight.exe");
    }

    /// <summary>The app's top-level windows (design windows, panel windows, popups).</summary>
    public AutomationElement[] TopLevelWindows() =>
        Automation.GetDesktop().FindAllChildren(Automation.ConditionFactory.ByProcessId(Window.Properties.ProcessId.Value));

    /// <summary>Runs a test body; on failure saves a capture of the app window (PrintWindow) and the automation tree.</summary>
    public void Run(Action body)
    {
        try { body(); }
        catch (Exception e) when (e is not Xunit.Sdk.SkipException)
        {
            SaveArtifacts();
            throw;
        }
    }

    public string ArtifactsDirectory =>
        Environment.GetEnvironmentVariable("MKLIGHT_UI_ARTIFACTS") is { Length: > 0 } d ? d : Path.Combine(AppContext.BaseDirectory, "ui-artifacts");

    public void SaveArtifacts(string suffix = "failure")
    {
        try
        {
            Directory.CreateDirectory(ArtifactsDirectory);
            WindowImage.Capture(Hwnd).Save(Path.Combine(ArtifactsDirectory, $"{_name}-{suffix}.png"), System.Drawing.Imaging.ImageFormat.Png);
            File.WriteAllText(Path.Combine(ArtifactsDirectory, $"{_name}-{suffix}-tree.txt"), TreeDump.Of(Window));
        }
        catch (Exception e)
        {
            Console.Error.WriteLine($"Could not save the failure artefacts: {e.Message}");
        }
    }

    static void Stop(Process p)
    {
        try
        {
            if (!p.HasExited) p.Kill(entireProcessTree: true);
            p.WaitForExit(10_000);
        }
        catch (Exception e) when (e is InvalidOperationException or System.ComponentModel.Win32Exception)
        {
            Console.Error.WriteLine($"Could not stop the app (pid {p.Id}): {e.Message}");
        }
    }

    public void Dispose()
    {
        Stop(_process);
        _process.Dispose();
        Automation.Dispose();
        try { Directory.Delete(_scratch, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }
}
