using System.Text.Json.Nodes;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.App.Windows;
using MKLight.Engine;
using MKLight.Model;

namespace MKLight.Tests.Documents;

/// <summary>The import report's grouping and text, autosave paths and recovery, rename checks, launch options.</summary>
public sealed class DocumentsTests : IDisposable
{
    readonly string _dir = Path.Combine(Path.GetTempPath(), "mklight-doc-tests-" + Guid.NewGuid().ToString("N"));

    public DocumentsTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    static ImportWarning W(ImportWarningKind k, string m, int? line = null, int? surface = null) => new(k, line, surface, m);

    static ImportReport Report(params ImportWarning[] w) => new("lens.zmx", PrescriptionFormat.Zmx, "utf16_le", 6, w);

    [Fact]
    public void GroupsBySeriousnessKeepingFileOrder()
    {
        var r = Report(
            W(ImportWarningKind.Defaulted, "a"), W(ImportWarningKind.UnknownGlass, "b"), W(ImportWarningKind.UnsupportedRecord, "c"),
            W(ImportWarningKind.Malformed, "d"), W(ImportWarningKind.UnknownGlass, "e"), W(ImportWarningKind.NonSequential, "f"),
            W(ImportWarningKind.SolveDropped, "g"), W(ImportWarningKind.Other, "h"));
        Assert.Equal([ImportCategory.Problems, ImportCategory.Glasses, ImportCategory.Surfaces, ImportCategory.Dropped, ImportCategory.Assumed],
            r.Groups.Select(g => g.Category));
        Assert.Equal(["d", "h"], r.Groups[0].Warnings.Select(w => w.Message));
        Assert.Equal(["b", "e"], r.Groups[1].Warnings.Select(w => w.Message));
        Assert.Equal(["c", "g"], r.Groups[3].Warnings.Select(w => w.Message));
        Assert.True(r.HasUnknownGlasses);
        Assert.False(Report(W(ImportWarningKind.Approximated, "x")).HasUnknownGlasses);
    }

    [Fact]
    public void SummaryHeadlineAndText()
    {
        var r = Report(W(ImportWarningKind.UnknownGlass, "glass X is missing", 51, 1), W(ImportWarningKind.UnsupportedRecord, "record TOLR dropped", 14));
        Assert.Equal("6 surfaces · .zmx · UTF-16", r.Summary);
        Assert.Equal("2 items could not be imported exactly. Review them before relying on the design.", r.Headline);
        Assert.Equal("1 item could not be imported exactly. Review it before relying on the design.", Report(W(ImportWarningKind.Other, "x")).Headline);
        Assert.Equal("Surface 1 · Line 51", ImportReport.Location(r.Warnings[0]));
        Assert.Null(ImportReport.Location(W(ImportWarningKind.Other, "x")));
        Assert.Equal("Glass X is missing", ImportReport.Sentence(r.Warnings[0]));
        Assert.Equal(
            "Imported lens.zmx (6 surfaces · .zmx · UTF-16)\n\nGlasses:\n- Glass X is missing [Surface 1 · Line 51]\n\nNot Imported:\n- Record TOLR dropped [Line 14]",
            r.PlainText);
        Assert.Equal("1 surface · .seq · Latin-1", new ImportReport("a.seq", PrescriptionFormat.Seq, "latin1", 1, []).Summary);
    }

    [Fact]
    public void AutosavePaths()
    {
        var id = Guid.NewGuid();
        Assert.Equal(@"C:\d\a.mklight.autosave", Autosaves.PathFor(@"C:\d\a.mklight", id, @"C:\s"));
        Assert.Equal(Path.Combine(@"C:\s", "Autosave", id.ToString("N") + ".mklight"), Autosaves.PathFor(null, id, @"C:\s"));
    }

    [Fact]
    public void FindsUntitledAndBesideFileAutosavesAndRecoversOverTheSavedFile()
    {
        var support = Path.Combine(_dir, "support");
        var saved = new ProjectFile(SampleSystems.Singlet());
        var file = Path.Combine(_dir, "lens.mklight");
        saved.Write(file);
        var changed = saved with { System = saved.System with { Name = "Changed" } };
        Autosaves.Write(changed, file + Autosaves.Suffix, "1.0");
        var untitled = Autosaves.PathFor(null, Guid.NewGuid(), support);
        Autosaves.Write(changed, untitled, null);
        File.SetLastWriteTime(untitled, DateTime.Now.AddMinutes(-5));

        var found = Autosaves.Find(support, [file, file, Path.Combine(_dir, "missing.mklight")]);
        Assert.Equal(2, found.Count);
        Assert.Equal(file, found[0].Original); // newest first
        Assert.Equal("lens.mklight", found[0].Title);
        Assert.Null(found[1].Original);

        // Beside a file: undo goes back to the saved file; the recovered state is one undoable step.
        var recovered = ProjectFile.Read(found[0].Path);
        var doc = new DesignDocument(Autosaves.RecoveryBase(found[0], recovered), file);
        Assert.True(doc.EditProject("Recover Unsaved Changes", _ => recovered));
        Assert.True(doc.IsDirty);
        Assert.Equal("Changed", doc.System.Name);
        doc.Undo();
        Assert.Equal(saved.System.Name, doc.System.Name);

        // Untitled: the autosave (stamped with metadata) differs from its bare system, so it shows as changed.
        var u = ProjectFile.Read(found[1].Path);
        var udoc = new DesignDocument(Autosaves.RecoveryBase(found[1], u));
        Assert.True(udoc.EditProject("Recover Unsaved Changes", _ => u));
        Assert.True(udoc.IsDirty);

        Autosaves.Delete(untitled);
        Assert.Single(Autosaves.Find(support, [file]));
    }

    [Fact]
    public void RenameChecks()
    {
        var current = Path.Combine(_dir, "a.mklight");
        File.WriteAllText(current, "");
        File.WriteAllText(Path.Combine(_dir, "b.mklight"), "");
        Assert.Equal(Path.Combine(_dir, "c.mklight"), DocumentFiles.RenamedPath(current, " c "));
        Assert.Null(DocumentFiles.RenameProblem(current, "c"));
        Assert.Null(DocumentFiles.RenameProblem(current, "A")); // case only: the same file
        Assert.NotNull(DocumentFiles.RenameProblem(current, "b"));
        Assert.NotNull(DocumentFiles.RenameProblem(current, " "));
        Assert.NotNull(DocumentFiles.RenameProblem(current, "x/y"));
        Assert.NotNull(DocumentFiles.RenameProblem(current, "x."));
        Assert.True(DocumentFiles.CanOpen(@"C:\a\b.ZMX"));
        Assert.True(DocumentFiles.CanOpen("b.mklight"));
        Assert.False(DocumentFiles.CanOpen("b.agf"));
    }

    [Fact]
    public void LaunchOptionsParse()
    {
        var o = LaunchOptions.Parse(["--sidebar", "hidden", "--show-issues", "--toast", "Hello there", "--support-dir", _dir,
            "--fit", "lens", "--curvature-bounds-unit", "curvature", "--recover-autosaves"]);
        Assert.False(o.Sidebar);
        Assert.True(o.ShowIssues);
        Assert.Equal("Hello there", o.Toast);
        Assert.Equal(Path.GetFullPath(_dir), o.SupportDirectory);
        Assert.Equal("lens", o.Fit);
        Assert.Equal("curvature", o.CurvatureBoundsUnit);
        Assert.True(o.RecoverAutosaves);
        Assert.True(LaunchOptions.Parse(["--sidebar", "shown"]).Sidebar);
        Assert.Null(LaunchOptions.Parse([]).Sidebar);
        // --settings must be a JSON object: quotes a shell stripped fail at once instead of hanging the run.
        Assert.Equal("""{"rings":128}""", LaunchOptions.Parse(["--settings", """{"rings":128}"""]).Settings);
        Assert.Contains("not valid JSON", Assert.Throws<ArgumentException>(() => LaunchOptions.Parse(["--settings", "{rings:128}"])).Message);
        Assert.Contains("JSON object", Assert.Throws<ArgumentException>(() => LaunchOptions.Parse(["--settings", "[1]"])).Message);
        Assert.Throws<ArgumentException>(() => LaunchOptions.Parse(["--sidebar", "maybe"]));
        Assert.Throws<ArgumentException>(() => LaunchOptions.Parse(["--fit", "everything"]));
        Assert.Throws<ArgumentException>(() => LaunchOptions.Parse(["--curvature-bounds-unit", "mm"]));
        Assert.Throws<ArgumentException>(() => LaunchOptions.Parse(["--toast"]));
    }
}
