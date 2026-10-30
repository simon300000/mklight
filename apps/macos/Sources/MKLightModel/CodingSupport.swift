import Foundation

/// Coding keys built from a string, for tagged enums whose payload keys are
/// shared with the tag in one flat JSON object (serde `#[serde(tag = "...")]`).
struct AnyKey: CodingKey {
    var stringValue: String
    var intValue: Int? { nil }
    init(_ s: String) { stringValue = s }
    init?(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { nil }
}

extension KeyedDecodingContainer where K == AnyKey {
    func value<T: Decodable>(_ key: String, as type: T.Type = T.self) throws -> T {
        try decode(T.self, forKey: AnyKey(key))
    }
    /// A non-optional field with `#[serde(default)]`: a missing key yields
    /// `fallback`, while an explicit `null` is an error, as in serde.
    func value<T: Decodable>(_ key: String, default fallback: T) throws -> T {
        contains(AnyKey(key)) ? try decode(T.self, forKey: AnyKey(key)) : fallback
    }
    /// A `usize` field: a negative value is an error, as in serde.
    func index(_ key: String) throws -> Int {
        let k = AnyKey(key)
        let v = try decode(Int.self, forKey: k)
        guard v >= 0 else {
            throw DecodingError.dataCorruptedError(forKey: k, in: self, debugDescription: "index \(v) is negative")
        }
        return v
    }
    func optional<T: Decodable>(_ key: String, as type: T.Type = T.self) throws -> T? {
        try decodeIfPresent(T.self, forKey: AnyKey(key))
    }
    /// A length that may be `±∞`, written as `"inf"` / `"-inf"` (CONVENTIONS §2).
    func extendedDouble(_ key: String) throws -> Double {
        let k = AnyKey(key)
        if let d = try? decode(Double.self, forKey: k) { return d }
        let s = try decode(String.self, forKey: k)
        guard let v = ExtendedDouble.parse(s) else {
            throw DecodingError.dataCorruptedError(
                forKey: k, in: self, debugDescription: "'\(s)' is not a number or ±inf")
        }
        return v
    }
}

extension KeyedEncodingContainer where K == AnyKey {
    mutating func put<T: Encodable>(_ value: T, _ key: String) throws {
        try encode(value, forKey: AnyKey(key))
    }
    /// Encodes `nil` as an explicit JSON `null`, as the engine's serializer does.
    mutating func putOptional<T: Encodable>(_ value: T?, _ key: String) throws {
        if let value { try encode(value, forKey: AnyKey(key)) } else { try encodeNil(forKey: AnyKey(key)) }
    }
    mutating func putExtendedDouble(_ v: Double, _ key: String) throws {
        if v.isInfinite {
            try encode(v > 0 ? "inf" : "-inf", forKey: AnyKey(key))
        } else {
            try encode(v, forKey: AnyKey(key))
        }
    }
}

/// Parsing of `±∞` length strings, matching the engine's `serde_util::f64_inf`.
/// NaN is rejected: it has no JSON form, so a model holding one could not be
/// saved or sent to the engine.
public enum ExtendedDouble {
    public static func parse(_ text: String) -> Double? {
        switch text.trimmingCharacters(in: .whitespaces).lowercased() {
        case "inf", "+inf", "infinity", "+infinity": return .infinity
        case "-inf", "-infinity": return -.infinity
        case let other:
            guard let v = Double(other), !v.isNaN else { return nil }
            return v
        }
    }
}

/// A two-element JSON array `[a, b]` (the engine serialises Rust tuples this way).
public struct Pair<A: Codable & Sendable & Hashable, B: Codable & Sendable & Hashable>: Codable, Sendable, Hashable {
    public var first: A
    public var second: B
    public init(_ first: A, _ second: B) {
        self.first = first
        self.second = second
    }
    public init(from decoder: any Decoder) throws {
        var c = try decoder.unkeyedContainer()
        first = try c.decode(A.self)
        second = try c.decode(B.self)
        if !c.isAtEnd {
            throw DecodingError.dataCorruptedError(in: c, debugDescription: "expected a 2-element array")
        }
    }
    public func encode(to encoder: any Encoder) throws {
        var c = encoder.unkeyedContainer()
        try c.encode(first)
        try c.encode(second)
    }
}

/// `(wavelength µm, value)` sample, as in coating reflectance tables and sensor
/// spectral responses.
public typealias SpectralSample = Pair<Double, Double>

extension Pair where A == Double, B == Double {
    public var um: Double { first }
    public var value: Double { second }
}
