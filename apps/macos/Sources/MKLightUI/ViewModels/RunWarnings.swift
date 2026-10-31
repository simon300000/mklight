import Foundation
import MKLightEngine

/// The engine's run warnings, grouped for reading: the same message for
/// several fields or wavelengths is shown once with its scopes. Engine
/// field and wavelength indices are 0-based; the app numbers them from 1
/// (`Field 1`, `λ1`), as everywhere else.
public struct WarningGroup: Equatable, Sendable, Identifiable {
    /// The message of the first occurrence, capitalised, without its scope.
    public var message: String
    /// `Field 1`, `Field 3 · λ2`, `field (1, −1)`, … (empty: the whole run).
    public var scopes: [String]
    /// What kind of remedy applies, when the app can offer one.
    public var remedy: Remedy?
    public var id: String { message + scopes.joined() }

    public enum Remedy: Equatable, Sendable {
        /// Finer pupil sampling (`pupil.grid`).
        case pupilSampling
        /// A larger image grid around the PSF.
        case imageGrid
    }

    /// One line for help tags and VoiceOver: "Field 1, Field 3: message".
    public var line: String { scopes.isEmpty ? message : scopes.joined(separator: ", ") + ": " + message }
}

public enum RunWarnings {
    /// Groups `warnings` by their message with the numbers masked, keeping
    /// first-seen order.
    public static func group(_ warnings: [String]) -> [WarningGroup] {
        var order: [String] = []
        var groups: [String: WarningGroup] = [:]
        for w in warnings {
            let (scope, message) = split(w)
            let key = mask(message)
            if var g = groups[key] {
                if let scope, !g.scopes.contains(scope) { g.scopes.append(scope) }
                groups[key] = g
            } else {
                order.append(key)
                groups[key] = WarningGroup(message: capitalised(message), scopes: scope.map { [$0] } ?? [], remedy: remedy(message))
            }
        }
        return order.compactMap { groups[$0] }
    }

    /// A short summary: the first group's line, and how many more.
    public static func summary(_ warnings: [String]) -> String {
        let g = group(warnings)
        guard let first = g.first else { return "" }
        return first.line + (g.count > 1 ? " (and \(g.count - 1) more)" : "")
    }

    /// `field 2, wavelength 1: text` → (`Field 3 · λ2`, `text`); also the
    /// engine's `psf_truncated: field [-1.0, -1.0], …` form.
    static func split(_ w: String) -> (String?, String) {
        let t = w.trimmingCharacters(in: .whitespaces)
        guard let colon = t.range(of: ": ") else { return (nil, t) }
        let head = String(t[..<colon.lowerBound]), rest = String(t[colon.upperBound...])
        let parts = head.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        var scope: [String] = []
        if head.hasPrefix("field (") {
            return ("field " + head.dropFirst("field ".count).replacingOccurrences(of: "-", with: "−"), rest)
        }
        for p in parts {
            let words = p.split(separator: " ")
            guard words.count == 2, let k = Int(words[1]) else { return (nil, t) }
            switch words[0] {
            case "field": scope.append("Field \(k + 1)")
            case "wavelength": scope.append("λ\(k + 1)")
            case "surface": scope.append("Surface \(k)")
            default: return (nil, t)
            }
        }
        return scope.isEmpty ? (nil, t) : (scope.joined(separator: " · "), rest)
    }

    /// The message with its numbers replaced, so "0.58 waves" and "0.72
    /// waves" group together.
    static func mask(_ s: String) -> String {
        s.replacingOccurrences(of: "[-+]?[0-9]*\\.?[0-9]+(e[-+]?[0-9]+)?", with: "#", options: .regularExpression)
    }

    static func capitalised(_ s: String) -> String { s.prefix(1).uppercased() + s.dropFirst() }

    static func remedy(_ s: String) -> WarningGroup.Remedy? {
        let l = s.lowercased()
        if l.contains("increase the pupil sampling") { return .pupilSampling }
        if l.contains("enlarge the grid") { return .imageGrid }
        return nil
    }
}
