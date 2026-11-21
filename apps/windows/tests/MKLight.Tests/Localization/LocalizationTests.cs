using System.Text.RegularExpressions;
using MKLight.App.Support;
using MKLight.Model;
using static MKLight.Tests.Engine.EngineTestSupport;

namespace MKLight.Tests.Localization;

/// <summary>
/// The UI text resources (Strings/&lt;language&gt;/Resources.resw): every English key has a Chinese one with the
/// same placeholders, every key the code or XAML uses exists; the language choice; the engine text lookup.
/// </summary>
public sealed partial class LocalizationTests
{
    static string AppDir => RepoPath("apps", "windows", "src", "MKLight.App");
    static Dictionary<string, string> Resw(string language) => Loc.ReadResw(Path.Combine(AppDir, "Strings", language, "Resources.resw"));

    [GeneratedRegex(@"\{(\d+)(?:[,:][^}]*)?\}")] private static partial Regex Placeholder();

    static string[] Placeholders(string s) => Placeholder().Matches(s.Replace("{{", "").Replace("}}", ""))
        .Select(m => m.Groups[1].Value).Distinct().Order(StringComparer.Ordinal).ToArray();

    [Fact]
    public void EveryEnglishKeyHasAChineseOneWithTheSamePlaceholders()
    {
        var en = Resw("en-US");
        var zh = Resw("zh-Hans");
        Assert.NotEmpty(en);
        Assert.Empty(en.Keys.Except(zh.Keys).Order(StringComparer.Ordinal)); // missing in zh-Hans
        Assert.Empty(zh.Keys.Except(en.Keys).Order(StringComparer.Ordinal)); // stale in zh-Hans
        var differ = en.Where(p => zh.TryGetValue(p.Key, out var z) && !Placeholders(p.Value).SequenceEqual(Placeholders(z)))
            .Select(p => p.Key).ToList();
        Assert.Empty(differ);
        Assert.Empty(zh.Where(p => p.Value.Trim().Length == 0).Select(p => p.Key));
    }

    [GeneratedRegex(@"\bLoc\.[SF]\(\s*""([A-Za-z0-9_.]+)""")] private static partial Regex CodeKey();
    [GeneratedRegex(@"x:Uid=""([A-Za-z0-9_]+)""")] private static partial Regex XamlUid();

    [Fact]
    public void EveryKeyTheAppUsesExists()
    {
        var en = Resw("en-US");
        var files = Directory.EnumerateFiles(AppDir, "*.*", SearchOption.AllDirectories)
            .Where(f => !f.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}") && !f.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}"));
        var missing = new List<string>();
        foreach (var f in files)
        {
            if (f.EndsWith(".cs", StringComparison.Ordinal))
                missing.AddRange(CodeKey().Matches(File.ReadAllText(f)).Select(m => m.Groups[1].Value).Where(k => !en.ContainsKey(k)).Select(k => $"{Path.GetFileName(f)}: {k}"));
            else if (f.EndsWith(".xaml", StringComparison.Ordinal))
                missing.AddRange(XamlUid().Matches(File.ReadAllText(f)).Select(m => m.Groups[1].Value)
                    .Where(u => !en.Keys.Any(k => k.StartsWith(u + ".", StringComparison.Ordinal))).Select(u => $"{Path.GetFileName(f)}: x:Uid {u}"));
        }
        Assert.Empty(missing);
    }

    [GeneratedRegex(@"\bT\(english, ""([A-Za-z]+)"", ""([^""]*)""")] private static partial Regex ModelKey();

    /// <summary>The labels MKLight.Model builds (ModelText): each key exists, with the English the code falls back to.</summary>
    [Fact]
    public void ModelLabelsHaveKeysWithTheSameEnglish()
    {
        var en = Resw("en-US");
        var code = File.ReadAllText(RepoPath("apps", "windows", "src", "MKLight.Model", "ParamValues.cs"));
        var found = ModelKey().Matches(code).Select(m => ("Model_Param_" + m.Groups[1].Value, m.Groups[2].Value)).ToList();
        Assert.True(found.Count > 20);
        Assert.Empty(found.Where(f => !en.TryGetValue(f.Item1, out var v) || v != f.Item2).Select(f => f.Item1));
    }

    [Fact]
    public void FormatsWithInvariantNumbers()
    {
        Assert.Equal("System (English)", Loc.F("Shell_Language_System", "English"));
        Assert.Equal("Missing_Key_Shows_Itself", Loc.S("Missing_Key_Shows_Itself"));
        var saved = Loc.Provider;
        try
        {
            Loc.Provider = k => k == "Shell_Language_System" ? "跟随系统（{0}）" : null;
            Assert.Equal("跟随系统（1.5）", Loc.F("Shell_Language_System", 1.5));
            Assert.Equal("Cancel", Loc.S("Common_Cancel")); // a key the provider lacks: English
        }
        finally { Loc.Provider = saved; }
    }

    [Theory]
    [InlineData(null, new[] { "zh-CN" }, "zh-Hans")]
    [InlineData(null, new[] { "zh" }, "zh-Hans")]
    [InlineData(null, new[] { "zh-SG" }, "zh-Hans")]
    [InlineData(null, new[] { "zh-Hans-HK" }, "zh-Hans")]
    [InlineData(null, new[] { "zh-TW" }, "en-US")]
    [InlineData(null, new[] { "zh-HK" }, "en-US")]
    [InlineData(null, new[] { "zh-Hant-TW" }, "en-US")]
    [InlineData(null, new[] { "de-DE", "zh-CN" }, "zh-Hans")]
    [InlineData(null, new[] { "de-DE" }, "en-US")]
    [InlineData(null, new[] { "en-GB", "zh-CN" }, "en-US")]
    [InlineData("system", new[] { "zh-CN" }, "zh-Hans")]
    [InlineData("en-US", new[] { "zh-CN" }, "en-US")]
    [InlineData("zh-Hans", new[] { "en-US" }, "zh-Hans")]
    public void ChoosesTheLanguage(string? choice, string[] system, string expected) =>
        Assert.Equal(expected, Languages.Resolve(choice, system));

    [Fact]
    public void UnknownLanguageChoiceFails()
    {
        Assert.Throws<ArgumentException>(() => Languages.Normalize("fr"));
        Assert.Equal("zh-Hans", LaunchOptions.Parse(["--language", "zh-Hans"]).Language);
        Assert.Equal("system", LaunchOptions.Parse(["--language", "system"]).Language);
        Assert.Throws<ArgumentException>(() => LaunchOptions.Parse(["--language", "xx"]));
    }

    // A small fixture in the shared catalog's format (the real catalog is locales/zh-Hans/engine.json).
    const string Fixture = """
        {
          "strings": { "Spot Diagram": "点列图", "Field": "视场", "Empty": "" },
          "diagnostics": { "thickness_negative": "面 {surface} 的厚度为负（{value}）", "no_args": "系统没有光阑" }
        }
        """;

    [Fact]
    public void EngineTextLooksUpExactEnglishAndFallsBack()
    {
        var t = EngineText.Parse(Fixture);
        Assert.Equal("点列图", t.Translate("Spot Diagram"));
        Assert.Equal("spot diagram", t.Translate("spot diagram")); // exact match only
        Assert.Equal("Empty", t.Translate("Empty")); // an empty translation is a miss
        Assert.Null(t.Translate(null));
        Assert.Equal("面 3 的厚度为负（-1.5）", t.TranslateDiagnostic("thickness_negative", "Surface 3 has negative thickness",
            new Dictionary<string, string> { ["surface"] = "3", ["value"] = "-1.5" }));
        // A placeholder without a value, or an unknown code: the English message.
        Assert.Equal("Surface 3 has negative thickness", t.TranslateDiagnostic("thickness_negative", "Surface 3 has negative thickness"));
        Assert.Equal("other", t.TranslateDiagnostic("unknown", "other"));
        Assert.Equal("系统没有光阑", t.TranslateDiagnostic("no_args", "The system has no stop"));
        Assert.Equal("Spot Diagram", EngineText.None.Translate("Spot Diagram"));
    }

    [Fact]
    public void EngineTextCatalogLoadsFromAFileOrFallsBackToEnglish()
    {
        var dir = Path.Combine(Path.GetTempPath(), "mklight-loc-" + Guid.NewGuid().ToString("N"));
        try
        {
            var path = EngineText.PathFor(dir, "zh-Hans");
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllText(path, Fixture);
            Assert.Equal("视场", EngineText.Load(path).Translate("Field"));
            File.WriteAllText(path, "not json");
            Assert.Same(EngineText.None, EngineText.Load(path));
            Assert.Same(EngineText.None, EngineText.Load(Path.Combine(dir, "missing.json")));
        }
        finally { try { Directory.Delete(dir, true); } catch (IOException) { } }
    }

    /// <summary>The shared catalog (docs/I18N.md) parses, with strings, enums and diagnostics.</summary>
    [Fact]
    public void SharedCatalogParses()
    {
        var t = EngineText.Parse(File.ReadAllText(RepoPath("locales", "zh-Hans", "engine.json")));
        Assert.True(t.Count > 400);
        Assert.Equal("自动", t.EnumLabel("auto"));
        Assert.Null(t.EnumLabel("not_an_enum_value"));
    }

    [Fact]
    public void CodedIssuesUseTheirTemplateAndKeepTheEnglishAsDetail()
    {
        var t = new EngineText(new Dictionary<string, string>(),
            new Dictionary<string, string> { ["missing_glass"] = "第 {surface} 面的玻璃未找到。", ["bad_path"] = "{path} 无效。" },
            new Dictionary<string, string> { ["hexapolar"] = "六角" });
        Assert.Equal(("第 3 面的玻璃未找到。", "Glass N-XYZ on surface 3 not found"), t.Issue("missing_glass", "Glass N-XYZ on surface 3 not found", 3));
        Assert.Equal(("Glass on surface ? not found", (string?)null), t.Issue("missing_glass", "Glass on surface ? not found")); // no surface: English
        Assert.Equal(("other", (string?)null), t.Issue("unknown_code", "other", 1));
        var raw = "invalid model: [bad_path] surfaces[2].radius: must be finite";
        Assert.Equal(("bad_path", "surfaces[2].radius", "must be finite"), EngineText.ParseStructural(raw));
        Assert.Equal(("surfaces[2].radius 无效。", raw), t.StructuralIssue(raw));
        Assert.Equal(("plain error", (string?)null), t.StructuralIssue("plain error"));
        Assert.Equal("六角", t.EnumLabel("hexapolar"));
    }
}
