import Foundation
import Testing

@testable import MKLightEngine
@testable import MKLightModel

/// The string catalog and the engine text catalogs (docs/APP.md, "Languages").
@Suite struct LocalizationTests {
    static let languages = ["zh-Hans"]

    static let repo = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
    static let catalogURL = repo.appending(path: "apps/macos/Sources/MKLightUI/Resources/Localizable.xcstrings")

    static func catalog() throws -> [String: [String: Any]] {
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: catalogURL)) as? [String: Any]
        return try #require(json?["strings"] as? [String: [String: Any]])
    }

    /// Every string unit of a localization: the plain one, or each plural
    /// or device variation.
    static func units(_ localization: [String: Any]) -> [[String: Any]] {
        if let unit = localization["stringUnit"] as? [String: Any] { return [unit] }
        guard let variations = localization["variations"] as? [String: Any] else { return [] }
        return variations.values.flatMap { cases in
            ((cases as? [String: Any]) ?? [:]).values.flatMap { units(($0 as? [String: Any]) ?? [:]) }
        }
    }

    /// Format specifiers in order of argument, `%%` excluded.
    static func specifiers(_ s: String) -> [String] {
        let pattern = /%(?:(\d+)\$)?[-+#0]*\d*(?:\.\d+)?((?:ll|l|hh|h|z|q)?[@dDiuUxXoOfeEgGcCsSpaA])/
        var out: [(Int, String)] = []
        var next = 1
        for m in s.replacing("%%", with: "").matches(of: pattern) {
            let position = m.output.1.flatMap { Int($0) } ?? next
            next += 1
            out.append((position, String(m.output.2)))
        }
        return out.sorted { $0.0 < $1.0 }.map(\.1)
    }

    /// New strings must be translated before they land: run
    /// scripts/sync-strings.sh, translate every new string, delete the stale ones.
    @Test func everyStringIsTranslated() throws {
        var missing: [String] = []
        var stale: [String] = []
        for (key, entry) in try Self.catalog() {
            if entry["extractionState"] as? String == "stale" { stale.append(key) }
            if entry["shouldTranslate"] as? Bool == false { continue }
            let localizations = entry["localizations"] as? [String: Any] ?? [:]
            for language in Self.languages {
                let units = Self.units(localizations[language] as? [String: Any] ?? [:])
                let done = !units.isEmpty && units.allSatisfy {
                    $0["state"] as? String == "translated" && !(($0["value"] as? String) ?? "").isEmpty
                }
                if !done { missing.append("\(language): \(key)") }
            }
        }
        #expect(missing.isEmpty, "untranslated: \(missing.sorted().prefix(40).joined(separator: " | "))")
        #expect(stale.isEmpty, "stale: \(stale.sorted().prefix(40).joined(separator: " | "))")
    }

    /// A translation takes the same arguments as its key.
    @Test func translationsKeepTheFormatArguments() throws {
        var wrong: [String] = []
        for (key, entry) in try Self.catalog() {
            let localizations = entry["localizations"] as? [String: Any] ?? [:]
            let english = Self.units(localizations["en"] as? [String: Any] ?? [:]).first?["value"] as? String ?? key
            let expected = Self.specifiers(english)
            for language in Self.languages {
                let localization = localizations[language] as? [String: Any] ?? [:]
                let isVariation = localization["variations"] != nil
                for unit in Self.units(localization) {
                    let value = unit["value"] as? String ?? ""
                    let got = Self.specifiers(value)
                    // A plural case may leave out the count ("one": "a glass").
                    let ok = got == expected || (isVariation && Set(got).isSubset(of: Set(expected)))
                    if !ok { wrong.append("\(language): \(key) → \(value)") }
                }
            }
        }
        #expect(wrong.isEmpty, "\(wrong.sorted().joined(separator: " | "))")
    }

    @Test func specifierParsing() {
        #expect(Self.specifiers("%lld of %@") == ["lld", "@"])
        #expect(Self.specifiers("%2$@ 的 %1$lld") == ["lld", "@"])
        #expect(Self.specifiers("100%% at %.3f mm") == ["f"])
    }

    /// Chinese for Simplified Chinese and its regions; Traditional Chinese
    /// stays English until it has a translation.
    @Test func languageResolution() {
        let available = ["en"] + Self.languages
        func resolved(_ preferred: String) -> String? {
            Bundle.preferredLocalizations(from: available, forPreferences: [preferred]).first
        }
        for zh in ["zh", "zh-CN", "zh-SG", "zh-Hans", "zh-Hans-CN", "zh-Hans-HK"] { #expect(resolved(zh) == "zh-Hans", "\(zh)") }
        for other in ["zh-Hant", "zh-TW", "zh-HK", "zh-Hant-TW", "en-GB", "de"] { #expect(resolved(other) == "en", "\(other)") }
    }

    /// Numbers read the same in every language: decimal point, no grouping.
    @Test func numbersDoNotChangeWithTheLanguage() {
        let x = 12345.6
        for language in ["en"] + Self.languages {
            let text = NumberDisplay.textLocale(language: language)
            // The views' locale carries the language (SwiftUI's string lookup)...
            #expect(Bundle.preferredLocalizations(from: ["en"] + Self.languages, forPreferences: [text.identifier]).first == language)
            // ...and the numbers keep the app's one format.
            #expect(x.formatted(.number.locale(NumberDisplay.locale)) == "12345.6", "\(language)")
            #expect(12345.formatted(.number.grouping(.never).locale(text)) == "12345", "\(language)")
            #expect(NumberDisplay.format(x, decimals: 1) == "12345.6")
            // Localized text formats its numbers with the number locale.
            #expect(String(localized: "\(x, specifier: "%.1f") mm", locale: NumberDisplay.locale) == "12345.6 mm")
            #expect(String(localized: "\(12345) trials", locale: NumberDisplay.locale) == "12345 trials")
        }
    }

    /// `String(localized:)` formats its arguments with `Locale.current`, the
    /// system region (`12.345,6` in Germany): every interpolated one passes
    /// the number locale.
    @Test func interpolatedStringsUseTheNumberLocale() throws {
        let sources = Self.repo.appending(path: "apps/macos/Sources")
        let files = FileManager.default.enumerator(at: sources, includingPropertiesForKeys: nil)?
            .compactMap { $0 as? URL }.filter { $0.pathExtension == "swift" } ?? []
        var missing: [String] = []
        for f in files {
            let text = try String(contentsOf: f, encoding: .utf8)
            for call in text.ranges(of: "String(localized: \"") {
                // The literal: from its quote to the closing one, skipping interpolations.
                var i = text.index(before: call.upperBound)
                var depth = 0
                var interpolated = false
                i = text.index(after: i)
                while i < text.endIndex {
                    let c = text[i]
                    if depth == 0 && c == "\"" { break }
                    if c == "\\", text.index(after: i) < text.endIndex, text[text.index(after: i)] == "(" {
                        interpolated = true
                        depth += 1
                        i = text.index(i, offsetBy: 2)
                        continue
                    }
                    if depth > 0 && c == "(" { depth += 1 }
                    if depth > 0 && c == ")" { depth -= 1 }
                    if c == "\\" { i = text.index(after: i) }
                    i = text.index(after: i)
                }
                guard interpolated, i < text.endIndex else { continue }
                let rest = text[text.index(after: i)...].prefix(40)
                if !rest.hasPrefix(", locale: NumberDisplay.locale") {
                    missing.append("\(f.lastPathComponent):\(text[..<call.lowerBound].count(where: { $0 == "\n" }) + 1)")
                }
            }
        }
        #expect(missing.isEmpty, "\(missing.joined(separator: ", "))")
    }

    // MARK: Engine text

    @Test func engineTextLookup() {
        let t = EngineText(data: Data(#"""
            { "strings": { "Spot diagram": "点列图", "Empty": "" },
              "diagnostics": { "thickness_sign": "面 {surface} 的厚度为负" } }
            """#.utf8))
        #expect(t.string("Spot diagram") == "点列图")
        #expect(t.string("spot diagram") == "spot diagram")
        #expect(t.string("Empty") == "Empty")
        #expect(t.translation("Ray fan") == nil)
        #expect(t.diagnostic(code: "thickness_sign", message: "negative", values: ["surface": "3"]) == "面 3 的厚度为负")
        #expect(t.diagnostic(code: "thickness_sign", message: "Spot diagram") == "点列图")
        #expect(t.diagnostic(code: "other", message: "the message", values: ["surface": "3"]) == "the message")
        #expect(EngineText(data: Data("not json".utf8)).isEmpty)
        let u = EngineText(data: Data(#"""
            { "strings": {}, "enums": { "hexapolar": "六角极坐标" },
              "diagnostics": { "material": "{path} 的材料无法解析" } }
            """#.utf8))
        #expect(u.enumLabel("hexapolar") == "六角极坐标")
        #expect(u.enumLabel("grid") == nil)
        #expect(u.message("invalid model: [material] surfaces[2].material: unknown material 'X'")
            == "surfaces[2].material 的材料无法解析 (unknown material 'X')")
        #expect(u.message("invalid model: [other] surfaces[2]: m") == "invalid model: [other] surfaces[2]: m")
        #expect(u.structuralIssue("plain text") == nil)
        #expect(EngineText.fill("{a} and {b}", ["a": "1", "b": "2"]) == "1 and 2")
        #expect(EngineText.fill("{a} and {b}", ["a": "1"]) == nil)
        #expect(EngineText.fill("no {placeholder", [:]) == "no {placeholder")
    }

    /// The shared catalogs, when present, are readable and complete.
    @Test func engineCatalogsParse() throws {
        for language in Self.languages {
            let url = Self.repo.appending(path: "locales/\(language)/engine.json")
            guard let data = try? Data(contentsOf: url) else { continue }
            let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
            let strings = try #require(json["strings"] as? [String: String], "\(language): strings")
            #expect(!strings.isEmpty)
            #expect(strings.filter { $0.value.isEmpty }.isEmpty, "\(language): empty translations")
            let t = EngineText(data: data)
            #expect(t.strings.count == strings.count)
        }
    }
}
