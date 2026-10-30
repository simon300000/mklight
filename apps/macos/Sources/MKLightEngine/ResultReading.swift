import Foundation
import MKLightModel

/// Tolerant accessors for engine result JSON (docs/RESULTS.md). Readers of
/// analysis results use them so that a missing key or a different type gives
/// `nil` (and the view falls back to its generic form) instead of an error.
///
/// Numbers follow CONVENTIONS §11: `null` is NaN (undefined) and the strings
/// `"inf"`, `"-inf"`, `"nan"` are non-finite values (`nf`).
extension JSONValue {
    public var array: [JSONValue]? { if case .array(let a) = self { a } else { nil } }
    public var object: [String: JSONValue]? { if case .object(let o) = self { o } else { nil } }
    public var string: String? { if case .string(let s) = self { s } else { nil } }
    public var bool: Bool? { if case .bool(let b) = self { b } else { nil } }
    public var isNull: Bool { self == .null }

    /// An `nf` number: a number, `null` (NaN) or one of the strings `inf`,
    /// `-inf`, `nan`. `nil` for anything else.
    public var nf: Double? {
        switch self {
        case .null: .nan
        case .string(let s): ["inf", "-inf", "nan", "+inf"].contains(s.lowercased()) ? ExtendedDouble.parse(s) ?? .nan : nil
        default: doubleValue
        }
    }

    /// An integer value (integral numbers only).
    public var int: Int? {
        switch self {
        case .integer(let i): Int(exactly: i)
        case .unsigned(let u): Int(exactly: u)
        case .number(let d): d.rounded() == d && d.isFinite ? Int(d) : nil
        default: nil
        }
    }

    /// An array of `nf` numbers (`nil` when any element is not one).
    public var nfArray: [Double]? {
        guard let a = array else { return nil }
        var out: [Double] = []
        out.reserveCapacity(a.count)
        for v in a {
            guard let d = v.nf else { return nil }
            out.append(d)
        }
        return out
    }

    /// An array of integers.
    public var intArray: [Int]? {
        guard let a = array else { return nil }
        let out = a.compactMap(\.int)
        return out.count == a.count ? out : nil
    }

    /// Nested lookup: `json[path: "fields", 0, "rms_radius"]`.
    public subscript(path keys: any JSONPathKey...) -> JSONValue? {
        var v: JSONValue? = self
        for k in keys {
            switch (k.jsonKey, v) {
            case (.key(let s), .object(let o)?): v = o[s]
            case (.index(let i), .array(let a)?): v = a.indices.contains(i) ? a[i] : nil
            default: return nil
            }
        }
        return v
    }

    /// Compact text of a scalar for tables (`—` for null).
    public var displayText: String {
        switch self {
        case .null: "—"
        case .bool(let b): b ? "yes" : "no"
        case .integer(let i): "\(i)"
        case .unsigned(let u): "\(u)"
        case .number(let d): NumberDisplay.format(d)
        case .string(let s): s
        case .array(let a): "[\(a.count)]"
        case .object(let o): "{\(o.count)}"
        }
    }
}

/// A key of a nested JSON lookup: a string (object key) or an integer.
public enum JSONPathComponent: Sendable { case key(String), index(Int) }

public protocol JSONPathKey { var jsonKey: JSONPathComponent { get } }
extension String: JSONPathKey { public var jsonKey: JSONPathComponent { .key(self) } }
extension Int: JSONPathKey { public var jsonKey: JSONPathComponent { .index(self) } }

extension ResultBuffer {
    /// Row `i` of a 2-D buffer as `Double`s (`shape [N, k]`).
    public func row(_ i: Int) -> [Double] {
        guard shape.count >= 2 else { return [self[i]] }
        let k = shape.dropFirst().reduce(1, *)
        return (0..<k).map { self[i * k + $0] }
    }

    /// The number of rows (first axis).
    public var rows: Int { shape.first ?? count }
}

/// How the engine computed a result (the envelope's `backend` object,
/// docs/RESULTS.md §5): what was requested, what was used, and why another
/// backend was used, if one was.
public struct BackendReport: Sendable, Hashable {
    public var requested: String
    public var used: String
    public var precision: String
    public var device: String?
    public var fallbackReason: String?

    public init(requested: String, used: String, precision: String, device: String? = nil, fallbackReason: String? = nil) {
        self.requested = requested
        self.used = used
        self.precision = precision
        self.device = device
        self.fallbackReason = fallbackReason
    }

    /// Reads the envelope's `backend` object; `nil` when it is missing.
    public init?(json: JSONValue?) {
        guard let o = json?.object else { return nil }
        requested = o["requested"]?.string ?? ""
        used = o["used"]?.string ?? ""
        precision = o["precision"]?.string ?? ""
        device = o["device"]?.string
        fallbackReason = o["fallback_reason"]?.string
    }

    /// e.g. "CPU reference · f64" or "Metal (Apple M2 Pro) · f32".
    public var summary: String {
        let name: String =
            switch used {
            case "cpu_reference": "CPU reference"
            case let s where s.hasPrefix("gpu:"): s.dropFirst(4).capitalized
            case "mock": "Mock"
            default: used
            }
        let dev = device.map { " (\($0))" } ?? ""
        let prec: String =
            switch precision {
            case "f64_reference": "f64"
            case "f32_gpu": "f32"
            default: precision
            }
        return prec.isEmpty ? name + dev : "\(name)\(dev) · \(prec)"
    }
}
