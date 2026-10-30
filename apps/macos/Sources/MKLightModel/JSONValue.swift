import Foundation

/// An arbitrary JSON value. Used for settings recorded in run metadata and for
/// semantic comparison of JSON documents: object key order is irrelevant and
/// numbers compare by exact value, so `10` equals `10.0`.
///
/// Integers are kept as integers. A `Double` cannot hold every `u64` (Philox
/// seeds are full 64-bit values, CONVENTIONS §6), so integer literals decode to
/// `.integer` / `.unsigned` and only non-integral or out-of-range numbers to
/// `.number`.
public enum JSONValue: Codable, Sendable, Hashable {
    case null
    case bool(Bool)
    /// An integer in `Int64` range.
    case integer(Int64)
    /// An integer above `Int64.max` (decoding never produces it for smaller values).
    case unsigned(UInt64)
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
        } else if let i = try? c.decode(Int64.self) {
            self = .integer(i)
        } else if let u = try? c.decode(UInt64.self) {
            self = .unsigned(u)
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

    /// Exact numeric identity shared by the three number cases.
    private enum NumericKey: Hashable {
        case signed(Int64), unsigned(UInt64), double(Double)
        init(_ d: Double) {
            if let i = Int64(exactly: d) {
                self = .signed(i)
            } else if let u = UInt64(exactly: d) {
                self = .unsigned(u)
            } else {
                self = .double(d)
            }
        }
        init(_ u: UInt64) { self = Int64(exactly: u).map(Self.signed) ?? .unsigned(u) }
    }

    private var numericKey: NumericKey? {
        switch self {
        case .integer(let i): .signed(i)
        case .unsigned(let u): NumericKey(u)
        case .number(let d): NumericKey(d)
        default: nil
        }
    }

    public static func == (a: JSONValue, b: JSONValue) -> Bool {
        if let x = a.numericKey, let y = b.numericKey { return x == y }
        switch (a, b) {
        case (.null, .null): return true
        case (.bool(let x), .bool(let y)): return x == y
        case (.string(let x), .string(let y)): return x == y
        case (.array(let x), .array(let y)): return x == y
        case (.object(let x), .object(let y)): return x == y
        default: return false
        }
    }

    public func hash(into h: inout Hasher) {
        if let k = numericKey {
            h.combine(2)
            h.combine(k)
            return
        }
        switch self {
        case .null: h.combine(0)
        case .bool(let b):
            h.combine(1)
            h.combine(b)
        case .string(let s):
            h.combine(3)
            h.combine(s)
        case .array(let a):
            h.combine(4)
            h.combine(a)
        case .object(let o):
            h.combine(5)
            h.combine(o)
        default: break
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let b): try c.encode(b)
        case .integer(let i): try c.encode(i)
        case .unsigned(let u): try c.encode(u)
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
