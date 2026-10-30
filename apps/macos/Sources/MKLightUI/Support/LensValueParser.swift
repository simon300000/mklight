import Foundation
import MKLightModel

/// Parses and validates text typed into the lens editor and inspectors.
/// Pure functions, so the rules are unit-tested independently of the UI.
public enum LensValueParser {
    public enum Failure: Error, Equatable, CustomStringConvertible {
        case notANumber(String)
        case mustBePositive
        case mustBeFinite
        case invalidMaterial(String)

        public var description: String {
            switch self {
            case .notANumber(let s): "'\(s)' is not a number"
            case .mustBePositive: "Value must be greater than zero"
            case .mustBeFinite: "Value must be finite"
            case .invalidMaterial(let s): "'\(s)' is not a material (glass name, AIR, MIRROR, nd/vd, n=…, CATALOG:NAME)"
            }
        }
    }

    /// A number, accepting `∞`/`inf`/`infinity`, a Unicode minus sign and
    /// surrounding whitespace. NaN is rejected.
    public static func number(_ text: String) -> Double? {
        var t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        t = t.replacingOccurrences(of: "\u{2212}", with: "-").replacingOccurrences(of: "∞", with: "inf")
        guard !t.isEmpty, let v = ExtendedDouble.parse(t), !v.isNaN else { return nil }
        return v
    }

    /// Radius text → curvature. `∞`, `inf` and `0` mean a plane (c = 0).
    public static func curvature(fromRadiusText text: String) -> Result<Double, Failure> {
        guard let r = number(text) else { return .failure(.notANumber(text)) }
        let c = RadiusDisplay.curvature(fromRadius: r)
        return c.isFinite ? .success(c) : .failure(.mustBeFinite)
    }

    /// Thickness; only the object surface may be infinite.
    public static func thickness(_ text: String, allowInfinite: Bool) -> Result<Double, Failure> {
        guard let t = number(text) else { return .failure(.notANumber(text)) }
        if t.isInfinite && !allowInfinite { return .failure(.mustBeFinite) }
        return .success(t)
    }

    public static func finite(_ text: String) -> Result<Double, Failure> {
        guard let v = number(text) else { return .failure(.notANumber(text)) }
        return v.isFinite ? .success(v) : .failure(.mustBeFinite)
    }

    public static func positive(_ text: String) -> Result<Double, Failure> {
        switch finite(text) {
        case .success(let v): v > 0 ? .success(v) : .failure(.mustBePositive)
        case .failure(let f): .failure(f)
        }
    }

    /// `auto` (or empty) for an automatic semi-diameter, else a fixed value > 0.
    public static func semiDiameter(_ text: String) -> Result<SemiDiameter, Failure> {
        let t = text.trimmingCharacters(in: .whitespaces).lowercased()
        if t.isEmpty || t == "auto" || t == "a" || t.hasPrefix("auto ") { return .success(.auto) }
        return positive(text).map { .fixed($0) }
    }

    /// A material edit: the new medium (`nil` keeps the current one) and the
    /// surface interaction.
    public struct MaterialEdit: Equatable, Sendable {
        public var material: MaterialRef?
        public var interaction: Interaction
    }

    /// Accepts `AIR`, `VACUUM`, `MIRROR`, `DUMMY`, `nd/vd` (model glass),
    /// `n=1.5` or a bare number (constant index), `CATALOG:NAME`, or a glass
    /// name.
    public static func material(_ text: String) -> Result<MaterialEdit, Failure> {
        let t = text.trimmingCharacters(in: .whitespaces)
        let u = t.uppercased()
        switch u {
        case "", "AIR": return .success(MaterialEdit(material: .air, interaction: .refract))
        case "VACUUM": return .success(MaterialEdit(material: .vacuum, interaction: .refract))
        case "MIRROR": return .success(MaterialEdit(material: nil, interaction: .reflect))
        case "DUMMY": return .success(MaterialEdit(material: nil, interaction: .dummy))
        default: break
        }
        let parts = t.split(separator: "/", omittingEmptySubsequences: false)
        if parts.count == 2 {
            guard let nd = number(String(parts[0])), let vd = number(String(parts[1])), nd >= 1, nd.isFinite, vd > 0, vd.isFinite
            else { return .failure(.invalidMaterial(t)) }
            return .success(MaterialEdit(material: .model(nd: nd, vd: vd, dpgf: 0), interaction: .refract))
        }
        var constant = u
        if constant.hasPrefix("N=") { constant.removeFirst(2) } else if constant.hasPrefix("=") { constant.removeFirst() }
        if let n = number(constant), constant.first.map({ $0.isNumber || $0 == "." }) == true {
            guard n > 0, n.isFinite else { return .failure(.invalidMaterial(t)) }
            return .success(MaterialEdit(material: .constant(n: n), interaction: .refract))
        }
        let colon = t.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: false)
        if colon.count == 2 {
            let cat = colon[0].trimmingCharacters(in: .whitespaces), name = colon[1].trimmingCharacters(in: .whitespaces)
            guard !cat.isEmpty, !name.isEmpty else { return .failure(.invalidMaterial(t)) }
            return .success(MaterialEdit(material: .catalog(catalog: cat.uppercased(), name: name.uppercased()), interaction: .refract))
        }
        guard t.allSatisfy({ $0.isLetter || $0.isNumber || "-_.+".contains($0) }) else { return .failure(.invalidMaterial(t)) }
        return .success(MaterialEdit(material: .glass(u), interaction: .refract))
    }
}
