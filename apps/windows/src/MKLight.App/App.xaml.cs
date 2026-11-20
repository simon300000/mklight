using System.Globalization;
using System.Text.Json.Nodes;
using Microsoft.UI.Xaml;
using Microsoft.Windows.Storage.Pickers;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.App.Windows;
using MKLight.Engine;
using MKLight.Model;
using Windows.Graphics;

namespace MKLight.App;

/// <summary>
/// The app: one engine for the process (each design gets its own session),
/// the welcome window, design windows, and the launch options for
/// self-checks (<see cref="LaunchOptions"/>).
/// </summary>
public partial class App : Application
{
    public static IEngineClient? Engine { get; private set; }
    public static List<Window> Windows { get; } = [];
    public static LaunchOptions Options { get; private set; } = new();

    public App()
    {
        // The app's own theme follows the chosen appearance, so theme resources that WinUI resolves
        // at the application level (styles applied in code) match the windows (docs/WINDOWS.md "Look").
        var args = Environment.GetCommandLineArgs();
        var i = Array.IndexOf(args, "--theme");
        var theme = i >= 0 && i + 1 < args.Length ? (args[i + 1] == "dark" ? ElementTheme.Dark : ElementTheme.Light) : AppSettings.Theme;
        if (theme != ElementTheme.Default) RequestedTheme = theme == ElementTheme.Dark ? ApplicationTheme.Dark : ApplicationTheme.Light;
        InitializeComponent();
        UnhandledException += (_, e) =>
        {
            Log($"unhandled: {e.Exception}");
            if (Options.IsScripted) Environment.Exit(3);
        };
    }

    protected override void OnLaunched(LaunchActivatedEventArgs args)
    {
        try { Options = LaunchOptions.Parse(Environment.GetCommandLineArgs().Skip(1).ToList()); }
        catch (Exception e) { Log(e.Message); Environment.Exit(2); }
        if (Options.Theme is { } t) AppSettings.Override = t == "dark" ? ElementTheme.Dark : ElementTheme.Light;

        if (!FFIEngine.IsAvailable)
        {
            Track(new EngineMissingWindow(FFIEngine.UnavailableReason ?? "mklight_ffi.dll was not found.")).Activate();
            return;
        }
        var engine = new FFIEngine(new JsonObject { ["app_name"] = "MKLight for Windows", ["app_version"] = AppVersion });
        Engine = engine;
        // Prepare the GPU's compute kernels once per run, cached between runs.
        try { engine.PrepareGpu(Path.Combine(AppSettings.Directory, "GPU Kernels")); } catch (EngineException) { }

        if (Options.File is { } file) _ = OpenFileAsync(file);
        else if (Options.Sample is { } sample) OpenDocument(DesignDocument.FromSample(sample));
        else ShowWelcome();
    }

    public static string AppVersion => typeof(App).Assembly.GetName().Version?.ToString(3) ?? "0.1.0";

    public static T Track<T>(T window) where T : Window
    {
        Windows.Add(window);
        window.Closed += (_, _) =>
        {
            Windows.Remove(window);
            if (Windows.Count == 0) Current.Exit();
        };
        return window;
    }

    public static void ShowWelcome()
    {
        var existing = Windows.OfType<WelcomeWindow>().FirstOrDefault();
        if (existing is not null) { existing.Activate(); return; }
        var w = Track(new WelcomeWindow());
        Size(w, 1180, 760);
        w.Activate();
        if (Options.Screenshot is { } shot && Options.Sample is null && Options.File is null) _ = CaptureWhenReady(w, null, shot);
    }

    public static DesignWindow OpenDocument(DesignDocument document)
    {
        var session = new DesignSession(document, Engine!);
        var window = Track(new DesignWindow(session));
        Size(window, Options.Size?.W ?? 1600, Options.Size?.H ?? 1000);
        window.Activate();
        if (document.Path is { } p) RecentFiles.Add(p);
        foreach (var w in Windows.OfType<WelcomeWindow>().ToList()) w.Close();
        _ = StartAsync(window, session);
        return window;
    }

    static async Task StartAsync(DesignWindow window, DesignSession session)
    {
        JsonObject? extra = Options.Settings is { } s ? JsonNode.Parse(s) as JsonObject : null;
        await session.StartAsync(Options.Analyses.Count > 0 ? Options.Analyses : null, extra);
        window.BuildSidebar();
        if (Options.Select is int i) session.SelectedSurface = i;
        if (Options.ShowSettings) foreach (var p in session.Workspace.Panes) { p.ShowsSettings = true; p.RaiseChanged(); }
        if (Options.IsScripted || Options.Edits.Count > 0 || Options.SaveAs is not null) await RunScriptAsync(window, session);
    }

    /// <summary>Launch-option edits, undo, save, then the screenshot (and quit).</summary>
    static async Task RunScriptAsync(DesignWindow window, DesignSession session)
    {
        await WaitIdleAsync(session);
        foreach (var edit in Options.Edits)
        {
            if (!ApplyEdit(session.Document, edit)) { Log($"edit refused: {edit}"); continue; }
            await WaitIdleAsync(session);
        }
        if (Options.Undo) { session.Document.Undo(); await WaitIdleAsync(session); }
        if (Options.SaveAs is { } path) await window.SaveAsync(false, path);
        if (Options.Screenshot is { } shot) await CaptureWhenReady(window, session, shot);
    }

    /// <summary>"<c>3.radius=30</c>": surface index, then radius, thickness or conic.</summary>
    static bool ApplyEdit(DesignDocument doc, string edit)
    {
        var (lhs, rhs) = (edit.Split('=')[0], edit.Split('=')[1]);
        var parts = lhs.Split('.');
        var index = int.Parse(parts[0], CultureInfo.InvariantCulture);
        var value = NumberDisplay.Parse(rhs) ?? double.NaN;
        return parts[1] switch
        {
            "radius" => doc.Edit("Change Radius", s => s.WithSurface(index, f => f.WithRadius(value))),
            "thickness" => doc.Edit("Change Thickness", s => s.WithSurface(index, f => f with { Thickness = value })),
            "conic" => doc.Edit("Change Conic", s => s.WithSurface(index, f => f.Shape.WithConic(value) is { } shape ? f with { Shape = shape } : f)),
            _ => false,
        };
    }

    /// <summary>Waits until the snapshot is current and no pane is running (at most 60 s).</summary>
    static async Task WaitIdleAsync(DesignSession session)
    {
        var start = DateTime.UtcNow;
        await Task.Delay(400);
        while (DateTime.UtcNow - start < TimeSpan.FromSeconds(60))
        {
            var ws = session.Workspace;
            var settled = ws.Snapshot is not null && !ws.IsBusy && ws.Layout.Output is not null
                && ws.Panes.All(p => p.Output is not null || p.Phase is PanePhase.Failed or PanePhase.Unavailable || !p.AutoUpdate)
                && !ws.Panes.Any(p => p.IsStale && p.AutoUpdate);
            if (settled) return;
            await Task.Delay(100);
        }
        Log("timed out waiting for the panes");
    }

    static async Task CaptureWhenReady(Window window, DesignSession? session, string path)
    {
        if (session is not null) await WaitIdleAsync(session);
        // Win2D presents a new result on a later frame, and a window larger than its display (--size)
        // composes slowly: give the views time to present, more after scripted edits.
        await Task.Delay(TimeSpan.FromSeconds(2 + (Options.Edits.Count > 0 || Options.Undo ? 4 : 0) + Options.Delay));
        try { WindowCapture.SavePng(window, path); Log($"screenshot: {path}"); }
        catch (Exception e) { Log($"screenshot failed: {e.Message}"); Environment.Exit(4); }
        Environment.Exit(0);
    }

    public static void Size(Window w, int width, int height)
    {
        var scale = GetDpiForWindow(WinRT.Interop.WindowNative.GetWindowHandle(w)) / 96.0;
        // Fit the work area of the window's display (a launch option's size is taken as given).
        var area = Microsoft.UI.Windowing.DisplayArea.GetFromWindowId(w.AppWindow.Id, Microsoft.UI.Windowing.DisplayAreaFallback.Primary).WorkArea;
        var px = new SizeInt32((int)(width * scale), (int)(height * scale));
        if (Options.Size is null) px = new SizeInt32(Math.Min(px.Width, (int)(area.Width * 0.94)), Math.Min(px.Height, (int)(area.Height * 0.94)));
        else WindowSizing.AllowOversize(w);
        w.AppWindow.Resize(px);
        w.AppWindow.Move(new PointInt32(area.X + Math.Max(0, (area.Width - px.Width) / 2), area.Y + Math.Max(0, (area.Height - px.Height) / 2)));
        if (w.AppWindow.Presenter is Microsoft.UI.Windowing.OverlappedPresenter p) p.PreferredMinimumWidth = (int)(900 * scale);
    }

    [System.Runtime.InteropServices.DllImport("user32.dll")]
    static extern uint GetDpiForWindow(IntPtr hwnd);

    // ---- Files ----

    public static async Task OpenWithPickerAsync(Window owner)
    {
        var picker = new FileOpenPicker(owner.AppWindow.Id) { SuggestedStartLocation = PickerLocationId.DocumentsLibrary };
        picker.FileTypeFilter.Add(".mklight");
        picker.FileTypeFilter.Add(".zmx");
        picker.FileTypeFilter.Add(".seq");
        var result = await picker.PickSingleFileAsync();
        if (result?.Path is { } path) await OpenFileAsync(path);
    }

    public static async Task<string?> PickSavePathAsync(Window owner, string suggestedName)
    {
        var picker = new FileSavePicker(owner.AppWindow.Id)
        {
            SuggestedStartLocation = PickerLocationId.DocumentsLibrary,
            SuggestedFileName = string.IsNullOrWhiteSpace(suggestedName) ? "Untitled" : suggestedName,
            DefaultFileExtension = ".mklight",
        };
        picker.FileTypeChoices.Add("MKLight design", [".mklight"]);
        var result = await picker.PickSaveFileAsync();
        return result?.Path;
    }

    /// <summary>A .mklight project, or a .zmx / .seq prescription imported as an untitled design.</summary>
    public static async Task OpenFileAsync(string path)
    {
        var existing = Windows.OfType<DesignWindow>().FirstOrDefault(w => string.Equals(w.Session.Document.Path, path, StringComparison.OrdinalIgnoreCase));
        if (existing is not null) { existing.Activate(); return; }
        try
        {
            var ext = Path.GetExtension(path).ToLowerInvariant();
            if (ext is ".zmx" or ".seq")
            {
                var import = Engine!.ImportPrescription(await File.ReadAllBytesAsync(path), ext == ".zmx" ? PrescriptionFormat.Zmx : PrescriptionFormat.Seq);
                var doc = new DesignDocument(new ProjectFile(ModelJson.Decode<OpticalSystem>(import.SystemJson)));
                var w = OpenDocument(doc);
                if (import.Warnings.Count > 0) w.Session.ShowToast($"Imported with {import.Warnings.Count} note{(import.Warnings.Count == 1 ? "" : "s")}: {import.Warnings[0].Message}");
                return;
            }
            OpenDocument(DesignDocument.Open(path));
        }
        catch (Exception e)
        {
            Log($"open failed: {e.Message}");
            if (Windows.OfType<WelcomeWindow>().FirstOrDefault() is { } welcome) welcome.ShowError($"Could not open {Path.GetFileName(path)}: {e.Message}");
            else if (Windows.OfType<DesignWindow>().FirstOrDefault() is { } dw) dw.Session.ShowToast($"Could not open {Path.GetFileName(path)}: {e.Message}");
            else { ShowWelcome(); Windows.OfType<WelcomeWindow>().First().ShowError($"Could not open {Path.GetFileName(path)}: {e.Message}"); }
        }
    }

    public static void Log(string message)
    {
        try
        {
            Directory.CreateDirectory(AppSettings.Directory);
            File.AppendAllText(Path.Combine(AppSettings.Directory, "mklight.log"), $"{DateTime.Now:O} {message}{Environment.NewLine}");
        }
        catch (Exception) { }
        Console.Error.WriteLine(message);
    }
}
