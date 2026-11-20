namespace MKLight.App.Support;

/// <summary>
/// Command-line options for self-checks and screenshots (the Mac app's debug
/// launch options, docs/WINDOWS.md "Launch options"):
/// <code>
/// MKLight.exe [file.mklight]
///   --sample &lt;id&gt;          open a built-in sample (e.g. double_gauss), no welcome window
///   --analysis &lt;name&gt;      open these analyses as panes (repeat or comma-separate)
///   --theme light|dark      override the system theme
///   --select &lt;i&gt;           select surface i in the lens data
///   --edit &lt;path&gt;=&lt;value&gt;  apply a lens edit after loading (e.g. 1.radius=30), as one undoable step
///   --undo                  undo it again (checks undo end to end)
///   --settings &lt;json&gt;      merge settings into every pane opened with --analysis
///   --show-settings         open the panes' settings forms
///   --do &lt;action&gt;[=&lt;arg&gt;]  run a feature's launch action (AppHooks.LaunchActions), repeatable, in order
///   --size &lt;w&gt;x&lt;h&gt;        window client size in pixels (default 1600x1000)
///   --welcome               show the welcome window (default without a file or sample)
///   --screenshot &lt;file.png&gt; render the window to a PNG once the panes are drawn, then quit
///   --delay &lt;seconds&gt;       extra wait before the screenshot (default 0)
///   --save-as &lt;file&gt;       save the design to this file before the screenshot
/// </code>
/// </summary>
public sealed class LaunchOptions
{
    public string? File;
    public string? Sample;
    public List<string> Analyses = [];
    public string? Theme;
    public int? Select;
    public List<string> Edits = [];
    public bool Undo;
    public string? Settings;
    public bool ShowSettings;
    public List<(string Name, string? Arg)> Actions = [];
    public (int W, int H)? Size;
    public bool Welcome;
    public string? Screenshot;
    public double Delay;
    public string? SaveAs;

    public bool IsScripted => Screenshot is not null;

    public static LaunchOptions Parse(IReadOnlyList<string> args)
    {
        var o = new LaunchOptions();
        for (var i = 0; i < args.Count; i++)
        {
            var a = args[i];
            string Next() => i + 1 < args.Count ? args[++i] : throw new ArgumentException($"{a} needs a value");
            switch (a)
            {
                case "--sample": o.Sample = Next(); break;
                case "--analysis": o.Analyses.AddRange(Next().Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)); break;
                case "--theme": o.Theme = Next(); break;
                case "--select": o.Select = int.Parse(Next(), System.Globalization.CultureInfo.InvariantCulture); break;
                case "--edit": o.Edits.Add(Next()); break;
                case "--undo": o.Undo = true; break;
                case "--show-settings": o.ShowSettings = true; break;
                case "--do":
                    var d = Next();
                    var eq = d.IndexOf('=');
                    o.Actions.Add(eq < 0 ? (d, null) : (d[..eq], d[(eq + 1)..]));
                    break;
                case "--settings": o.Settings = Next(); break;
                case "--welcome": o.Welcome = true; break;
                case "--screenshot": o.Screenshot = Path.GetFullPath(Next()); break;
                case "--delay": o.Delay = double.Parse(Next(), System.Globalization.CultureInfo.InvariantCulture); break;
                case "--save-as": o.SaveAs = Path.GetFullPath(Next()); break;
                case "--size":
                    var p = Next().Split('x');
                    o.Size = (int.Parse(p[0], System.Globalization.CultureInfo.InvariantCulture), int.Parse(p[1], System.Globalization.CultureInfo.InvariantCulture));
                    break;
                default:
                    if (!a.StartsWith("--", StringComparison.Ordinal) && o.File is null) o.File = Path.GetFullPath(a);
                    break;
            }
        }
        return o;
    }
}
