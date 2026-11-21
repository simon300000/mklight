import Foundation

/// Translations of text the engine writes in English: analysis titles and
/// descriptions, settings descriptions and choices, operand names, sample
/// names, messages and diagnostics.
///
/// The app's own text is in the string catalog (`Localizable.xcstrings`);
/// engine text comes with the engine and is translated by exact match
/// through the shared catalog `locales/<language>/engine.json` at the
/// repository root:
///
///     { "strings": { "<exact English>": "<translation>" },
///       "diagnostics": { "<code>": "<template with {placeholders}>" } }
///
/// The build script copies it into the app bundle as
/// `Contents/Resources/<language>.lproj/engine.json`, so the language is the
/// app's own (the per-app language setting or the preferred languages, as
/// resolved by `Bundle.main`). A miss, or no catalog for the language,
/// leaves the English. The engine boundary stays English: translate only
/// where text is shown, never before comparing or parsing it.
public struct EngineText: Sendable {
    public let strings: [String: String]
    public let diagnostics: [String: String]

    public static let none = EngineText(strings: [:], diagnostics: [:])

    public init(strings: [String: String], diagnostics: [String: String] = [:]) {
        self.strings = strings
        self.diagnostics = diagnostics
    }

    /// Reads a catalog; anything unreadable gives no translations.
    public init(data: Data?) {
        struct File: Decodable {
            var strings: [String: String]?
            var diagnostics: [String: String]?
        }
        let file = data.flatMap { try? JSONDecoder().decode(File.self, from: $0) }
        self.init(strings: file?.strings ?? [:], diagnostics: file?.diagnostics ?? [:])
    }

    /// The catalog of `bundle`'s preferred localization; none for English.
    public static func load(from bundle: Bundle = .main) -> EngineText {
        guard let language = bundle.preferredLocalizations.first, language != "en",
            let url = bundle.url(forResource: "engine", withExtension: "json", subdirectory: nil, localization: language)
        else { return .none }
        return EngineText(data: try? Data(contentsOf: url))
    }

    /// The app's catalog, read once.
    public static let shared = load()

    /// The translation of `english`, or `nil` when there is none.
    public func translation(_ english: String) -> String? {
        guard let t = strings[english], !t.isEmpty else { return nil }
        return t
    }

    /// `english` translated, or unchanged.
    public func string(_ english: String) -> String { translation(english) ?? english }

    /// A diagnostic by its stable code: the code's template with its
    /// `{placeholders}` filled from `values` (the diagnostic's fields:
    /// `surface`, `path`, `field`, …). Without a template, or when a
    /// placeholder has no value, the message itself by exact match, else the
    /// English message.
    public func diagnostic(code: String, message: String, values: [String: String] = [:]) -> String {
        if let template = diagnostics[code], let filled = Self.fill(template, values) { return filled }
        return string(message)
    }

    /// `template` with every `{name}` replaced; `nil` if a name has no value.
    static func fill(_ template: String, _ values: [String: String]) -> String? {
        var out = ""
        var rest = template[...]
        while let open = rest.firstIndex(of: "{") {
            guard let close = rest[open...].firstIndex(of: "}") else { break }
            out += rest[..<open]
            let name = String(rest[rest.index(after: open)..<close])
            guard let value = values[name] else { return nil }
            out += value
            rest = rest[rest.index(after: close)...]
        }
        return out + rest
    }

    // Shorthands on the app's catalog.

    /// `english` translated through the app's engine catalog, or unchanged.
    public static func string(_ english: String) -> String { shared.string(english) }
    /// The translation through the app's engine catalog, or `nil`.
    public static func translation(_ english: String) -> String? { shared.translation(english) }
    /// A diagnostic through the app's engine catalog (`diagnostic(code:message:values:)`).
    public static func diagnostic(code: String, message: String, values: [String: String] = [:]) -> String {
        shared.diagnostic(code: code, message: message, values: values)
    }
}
