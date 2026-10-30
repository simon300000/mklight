import Foundation

/// An arbitrary JSON value. Used for settings recorded in run metadata and for
/// semantic comparison of JSON documents (numbers compare by value, so `10`
/// equals `10.0`, and object key order is irrelevant).
public enum JSONValue: Codable, Sendable, Hashable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    public init(from decoder: any Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() {
            self = .null
        } else if let b = try? c.decode(Bool.self) {
            self = .bool(b)
        } else if let d = try? c.decode(Double.self) {
            self = .number(d)
        } else if let s = try? c.decode(String.self) {
            self = .string(s)
        } else if let a = try? c.decode([JSONValue].self) {
            self = .array(a)
        } else {
            self = .object(try c.decode([String: JSONValue].self))
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let b): try c.encode(b)
        case .number(let d): try c.encode(d)
        case .string(let s): try c.encode(s)
        case .array(let a): try c.encode(a)
        case .object(let o): try c.encode(o)
        }
    }

    /// Parses JSON text.
    public static func parse(_ data: Data) throws -> JSONValue {
        try JSONDecoder().decode(JSONValue.self, from: data)
    }

    /// Converts any `Encodable` value into its JSON tree.
    public static func from<T: Encodable>(_ value: T) throws -> JSONValue {
        try parse(MKLightJSON.encode(value))
    }

    public subscript(key: String) -> JSONValue? {
        if case .object(let o) = self { return o[key] }
        return nil
    }

    /// Paths at which two JSON trees differ (empty when equal). For test
    /// diagnostics.
    public func differences(from other: JSONValue, path: String = "$") -> [String] {
        switch (self, other) {
        case (.object(let a), .object(let b)):
            var out: [String] = []
            for k in Set(a.keys).union(b.keys).sorted() {
                switch (a[k], b[k]) {
                case (let x?, let y?): out += x.differences(from: y, path: "\(path).\(k)")
                case (nil, _): out.append("\(path).\(k): missing on left")
                case (_, nil): out.append("\(path).\(k): missing on right")
                }
            }
            return out
        case (.array(let a), .array(let b)):
            guard a.count == b.count else { return ["\(path): array length \(a.count) vs \(b.count)"] }
            return zip(a, b).enumerated().flatMap { i, p in p.0.differences(from: p.1, path: "\(path)[\(i)]") }
        default:
            return self == other ? [] : ["\(path): \(self) vs \(other)"]
        }
    }
}

/// Shared JSON configuration for model and engine documents.
public enum MKLightJSON {
    /// Deterministic output: sorted keys, so equal values give equal bytes.
    public static func encoder(pretty: Bool = false) -> JSONEncoder {
        let e = JSONEncoder()
        e.outputFormatting = pretty ? [.sortedKeys, .prettyPrinted, .withoutEscapingSlashes] : [.sortedKeys, .withoutEscapingSlashes]
        return e
    }
    public static func decoder() -> JSONDecoder { JSONDecoder() }

    public static func encode<T: Encodable>(_ value: T, pretty: Bool = false) throws -> Data {
        try encoder(pretty: pretty).encode(value)
    }
    public static func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
        try decoder().decode(type, from: data)
    }
}
