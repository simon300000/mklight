import Foundation
import MKLightModel

// MARK: - Analysis descriptions

/// One analysis the engine offers (`mk_engine_list_analyses`): its name,
/// availability and settings schema. Unknown keys of the engine's JSON are
/// ignored, so newer engines stay readable.
public struct AnalysisDescriptor: Codable, Sendable, Hashable, Identifiable {
    public var name: String
    public var title: String
    public var category: String
    public var description: String
    public var available: Bool
    public var unavailableReason: String?
    /// JSON-Schema subset (docs/FFI.md): `type`, `properties`, `required`,
    /// `default`, `minimum`, `maximum`, `enum`, `items`, `title`,
    /// `description`, `unit`.
    public var settingsSchema: JSONValue {
        didSet { settingsFields = Self.fields(of: settingsSchema) }
    }
    /// Complete default settings; `null` when some setting is required.
    public var defaultSettings: JSONValue
    /// A running job of the analysis can be stopped early, keeping its best
    /// result so far (`AnalysisJob.stop()`).
    public var stoppable: Bool
    /// The schema's properties, in a stable order (required first, then by
    /// name); read from `settingsSchema` whenever that is set.
    public private(set) var settingsFields: [SettingsField]

    public var id: String { name }

    public init(
        name: String, title: String, category: String, description: String = "", available: Bool,
        unavailableReason: String? = nil, settingsSchema: JSONValue = .object([:]), defaultSettings: JSONValue = .null,
        stoppable: Bool = false
    ) {
        self.name = name
        self.title = title
        self.category = category
        self.description = description
        self.available = available
        self.unavailableReason = unavailableReason
        self.settingsSchema = settingsSchema
        self.defaultSettings = defaultSettings
        self.stoppable = stoppable
        settingsFields = Self.fields(of: settingsSchema)
    }

    private enum CodingKeys: String, CodingKey {
        case name, title, category, description, available
        case unavailableReason = "unavailable_reason", settingsSchema = "settings_schema"
        case defaultSettings = "default_settings", stoppable
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        name = try c.decode(String.self, forKey: .name)
        title = try c.decodeIfPresent(String.self, forKey: .title) ?? name
        category = try c.decodeIfPresent(String.self, forKey: .category) ?? ""
        description = try c.decodeIfPresent(String.self, forKey: .description) ?? ""
        available = try c.decodeIfPresent(Bool.self, forKey: .available) ?? true
        unavailableReason = try c.decodeIfPresent(String.self, forKey: .unavailableReason)
        settingsSchema = try c.decodeIfPresent(JSONValue.self, forKey: .settingsSchema) ?? .object([:])
        defaultSettings = try c.decodeIfPresent(JSONValue.self, forKey: .defaultSettings) ?? .null
        stoppable = try c.decodeIfPresent(Bool.self, forKey: .stoppable) ?? false
        settingsFields = Self.fields(of: settingsSchema)
    }

    private static func fields(of schema: JSONValue) -> [SettingsField] {
        guard case .object(let props)? = schema["properties"] else { return [] }
        let required: Set<String> =
            if case .array(let r)? = schema["required"] {
                Set(r.compactMap { if case .string(let s) = $0 { s } else { nil } })
            } else { [] }
        return SettingsField.fields(props, required: required)
    }

    /// `settings` reduced to the keys the schema declares, with values of an
    /// accepted type. The engine rejects unknown keys, so keys this schema
    /// does not declare are dropped and the engine's defaults apply instead.
    public func acceptedSettings(_ settings: [String: JSONValue]) -> [String: JSONValue] {
        settings.filter { key, value in settingsFields.first { $0.name == key }?.accepts(value) ?? false }
    }
}

/// One property of an analysis settings schema.
public struct SettingsField: Sendable, Hashable, Identifiable {
    public var name: String
    /// JSON types accepted (`number`, `integer`, `string`, `boolean`,
    /// `array`, `object`, `null`); empty means any.
    public var types: [String]
    public var title: String
    public var description: String
    public var unit: String
    public var defaultValue: JSONValue?
    public var minimum: Double?
    public var maximum: Double?
    public var enumValues: [JSONValue]
    public var required: Bool
    /// The schema of an array's elements (`items`), if given.
    public var items: JSONValue?
    /// A fixed value (`const`): the tag of a variant.
    public var constValue: JSONValue?
    /// One alternative of a tagged union (`oneOf`): its title and the
    /// properties of its object, one of which is the tag with a `const`.
    public struct Variant: Sendable, Hashable, Identifiable {
        public var title: String
        public var properties: [SettingsField]
        /// The tag property's name and value (`"type": "grid"`).
        public var tag: (key: String, value: JSONValue)? {
            properties.lazy.compactMap { f in f.constValue.map { (f.name, $0) } }.first
        }
        public var id: String { title }
        public init(title: String, properties: [SettingsField]) {
            self.title = title
            self.properties = properties
        }
        /// The variant's object with every property at its default.
        public var defaultObject: JSONValue {
            .object(Dictionary(uniqueKeysWithValues: properties.compactMap { f in (f.constValue ?? f.defaultValue).map { (f.name, $0) } }))
        }
        public static func == (a: Variant, b: Variant) -> Bool { a.title == b.title && a.properties == b.properties }
        public func hash(into h: inout Hasher) { h.combine(title) }
    }
    /// The alternatives of a tagged union (`oneOf` with `const` tags).
    public var variants: [Variant]
    /// The fields of an object setting: the schema's `properties`, or,
    /// without any, inferred from the keys of the default object (each with
    /// the JSON type of its default). Empty for other types.
    public var properties: [SettingsField]
    public var id: String { name }

    public init(name: String, schema: JSONValue, required: Bool) {
        self.name = name
        self.required = required
        switch schema["type"] {
        case .string(let t)?: types = [t]
        case .array(let a)?: types = a.compactMap { if case .string(let s) = $0 { s } else { nil } }
        default: types = []
        }
        func string(_ k: String) -> String? { if case .string(let s)? = schema[k] { s } else { nil } }
        title = string("title") ?? name
        description = string("description") ?? ""
        unit = string("unit") ?? ""
        defaultValue = schema["default"]
        minimum = schema["minimum"]?.doubleValue
        maximum = schema["maximum"]?.doubleValue
        enumValues = if case .array(let a)? = schema["enum"] { a } else { [] }
        items = schema["items"]
        constValue = schema["const"]
        variants = (schema["oneOf"]?.array ?? []).compactMap { alt in
            guard case .object(let props)? = alt["properties"] else { return nil }
            let fields = Self.fields(props, required: [])
            guard fields.contains(where: { $0.constValue != nil }) else { return nil }
            return Variant(title: alt["title"]?.string ?? "", properties: fields)
        }
        if case .object(let props)? = schema["properties"] {
            let required: Set<String> = Set((schema["required"]?.array ?? []).compactMap(\.string))
            properties = Self.fields(props, required: required)
        } else if variants.isEmpty, case .object(let d)? = schema["default"] {
            // Only the default object says what the keys are: each keeps
            // the type of its default (a null default has no type).
            properties = d.map { key, v in
                SettingsField(name: key, schema: .object(["type": .string(Self.typeName(v)), "default": v]), required: false)
            }
            .sorted { $0.name < $1.name }
        } else {
            properties = []
        }
    }

    /// Properties in a stable order: required first, then by name.
    static func fields(_ props: [String: JSONValue], required: Set<String>) -> [SettingsField] {
        props.map { SettingsField(name: $0.key, schema: $0.value, required: required.contains($0.key)) }
            .sorted { ($0.required ? 0 : 1, $0.name) < ($1.required ? 0 : 1, $1.name) }
    }

    /// The JSON type name of a value.
    public static func typeName(_ v: JSONValue) -> String {
        switch v {
        case .null: "null"
        case .bool: "boolean"
        case .integer, .unsigned: "integer"
        case .number: "number"
        case .string: "string"
        case .array: "array"
        case .object: "object"
        }
    }

    /// Whether `null` is an accepted value.
    public var nullable: Bool { types.contains("null") }
    /// The accepted types other than `null`.
    public var baseTypes: Set<String> { Set(types).subtracting(["null"]) }
    /// The variant of a tagged-union value, by its tag.
    public func variant(of value: JSONValue?) -> Variant? {
        guard let value else { return nil }
        return variants.first { v in v.tag.map { value[$0.key] == $0.value } ?? false }
    }

    /// Whether `value` has one of the accepted types (and, for enums, is a
    /// listed value).
    public func accepts(_ value: JSONValue) -> Bool {
        if !enumValues.isEmpty && !enumValues.contains(value) { return false }
        if types.isEmpty { return true }
        return types.contains { t in
            switch (t, value) {
            case ("null", .null), ("boolean", .bool), ("string", .string), ("array", .array), ("object", .object): true
            case ("integer", .integer), ("integer", .unsigned): true
            case ("integer", .number(let d)): d.rounded() == d
            case ("number", .integer), ("number", .unsigned), ("number", .number): true
            default: false
            }
        }
    }
}

extension JSONValue {
    /// The value as a `Double` for any of the three number cases.
    public var doubleValue: Double? {
        switch self {
        case .integer(let i): Double(i)
        case .unsigned(let u): Double(u)
        case .number(let d): d
        case .string(let s): ExtendedDouble.parse(s)
        default: nil
        }
    }
}

// MARK: - Requests by name

/// Which compute backend a job should use. A preference: the engine falls back
/// to the CPU reference (and says so in `RunInfo.warnings`) when a backend is
/// unavailable or cannot meet the error budget for a design (CONVENTIONS §7).
public enum BackendPreference: String, Codable, Sendable, Hashable, CaseIterable {
    /// The engine chooses.
    case auto
    /// The CPU f64 reference tracer.
    case cpuReference = "cpu_reference"
    /// This platform's GPU backend (Metal on macOS).
    case gpu

    public var title: String {
        switch self {
        case .auto: "Automatic"
        case .cpuReference: "CPU reference"
        case .gpu: "GPU"
        }
    }
}

/// How a job is executed: what a front end chooses of a job's identity
/// `(snapshot, settings, backend, precision, seed)` (ARCHITECTURE §State
/// model) besides the snapshot and the analysis settings. The engine chooses
/// the precision with the backend (`RunInfo.precision` records it), and a
/// seed is the analysis' own `seed` setting. It is part of the request, so
/// it takes part in the staleness check: another backend preference makes
/// the old result stale.
public struct ExecutionSettings: Codable, Sendable, Hashable {
    public var backend = BackendPreference.auto
    public init(backend: BackendPreference = .auto) {
        self.backend = backend
    }
}

/// A request for any engine analysis by name (`{"analysis", "settings",
/// "view_key", "backend", "allow_backend_fallback"}` on the C ABI).
/// `settings` may be partial: the engine fills in defaults and records the
/// complete settings in `RunInfo.settings`.
public struct EngineAnalysisRequest: Codable, Sendable, Hashable {
    public var analysis: String
    public var settings: [String: JSONValue]
    /// The backend preference.
    public var execution: ExecutionSettings

    public init(analysis: String, settings: [String: JSONValue] = [:], execution: ExecutionSettings = ExecutionSettings()) {
        self.analysis = analysis
        self.settings = settings
        self.execution = execution
    }

    /// Whether `info` (normalised by the engine) records these settings: each
    /// requested key has the requested value.
    public func matches(_ info: RunInfo) -> Bool {
        guard info.analysis == analysis else { return false }
        return settings.allSatisfy { info.settings[$0.key] == $0.value }
    }
}

/// A glass catalog of a session (`mk_session_list_catalogs`).
public struct CatalogInfo: Codable, Sendable, Hashable, Identifiable {
    public var name: String
    /// `builtin` or `user`.
    public var source: String
    public var path: String?
    public var glassCount: Int
    public var id: String { name }

    public init(name: String, source: String, path: String?, glassCount: Int) {
        self.name = name
        self.source = source
        self.path = path
        self.glassCount = glassCount
    }

    private enum CodingKeys: String, CodingKey { case name, source, path, glassCount = "glass_count" }
}

/// One glass of a catalog (`mk_session_list_glasses`).
public struct GlassInfo: Codable, Sendable, Hashable {
    public var catalog: String
    public var name: String
    @EngineDouble public var nd: Double
    @EngineDouble public var vd: Double
    @EngineDouble public var minUm: Double
    @EngineDouble public var maxUm: Double

    private enum CodingKeys: String, CodingKey { case catalog, name, nd, vd, minUm = "min_um", maxUm = "max_um" }
}

// MARK: - Binary buffers

/// Element type of a result buffer (`MKDType`).
public enum BufferDType: Int32, Sendable, Hashable, CaseIterable {
    case f64 = 1, f32, i32, u32, i64, u64, u8

    public init?(name: String) {
        guard let t = Self.allCases.first(where: { $0.name == name }) else { return nil }
        self = t
    }

    public var name: String {
        switch self {
        case .f64: "f64"
        case .f32: "f32"
        case .i32: "i32"
        case .u32: "u32"
        case .i64: "i64"
        case .u64: "u64"
        case .u8: "u8"
        }
    }

    public var byteWidth: Int {
        switch self {
        case .f64, .i64, .u64: 8
        case .f32, .i32, .u32: 4
        case .u8: 1
        }
    }
}

/// A named, row-major typed array from an engine result.
///
/// Engine buffers are **not copied**: the buffer points into memory owned
/// by the engine library and keeps its owner (which frees the whole buffer
/// array once every `ResultBuffer` of a result is gone) alive. The memory is
/// immutable, so the type is `Sendable`.
public final class ResultBuffer: @unchecked Sendable, Hashable, CustomStringConvertible {
    public let name: String
    public let dtype: BufferDType
    public let shape: [Int]
    /// Number of elements (product of `shape`).
    public let count: Int
    private let base: UnsafeRawPointer?
    private let owner: AnyObject

    /// Wraps engine memory without copying. `owner` must keep `base` valid
    /// and unchanged for its lifetime.
    public init(name: String, dtype: BufferDType, shape: [Int], count: Int, base: UnsafeRawPointer?, owner: AnyObject) {
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.count = count
        self.base = count > 0 ? base : nil
        self.owner = owner
    }

    /// A buffer holding a copy of `values` (mock engines and tests).
    public convenience init(name: String, doubles values: [Double], shape: [Int]? = nil) {
        let storage = OwnedBytes(values.withUnsafeBytes { Data($0) })
        self.init(
            name: name, dtype: .f64, shape: shape ?? [values.count], count: values.count, base: storage.pointer,
            owner: storage)
    }

    /// The raw bytes (`count × dtype.byteWidth`), valid during `body`.
    public func withUnsafeBytes<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R {
        try body(UnsafeRawBufferPointer(start: base, count: count * dtype.byteWidth))
    }

    /// Typed access without copying; `T` must match `dtype` (else `nil`).
    public func withUnsafeBufferPointer<T, R>(of: T.Type, _ body: (UnsafeBufferPointer<T>) throws -> R) rethrows -> R? {
        guard MemoryLayout<T>.size == dtype.byteWidth, Self.matches(T.self, dtype) else { return nil }
        guard let base else { return try body(UnsafeBufferPointer(start: nil, count: 0)) }
        return try body(UnsafeBufferPointer(start: base.assumingMemoryBound(to: T.self), count: count))
    }

    private static func matches<T>(_ t: T.Type, _ d: BufferDType) -> Bool {
        switch d {
        case .f64: t == Double.self
        case .f32: t == Float.self
        case .i32: t == Int32.self
        case .u32: t == UInt32.self
        case .i64: t == Int64.self
        case .u64: t == UInt64.self
        case .u8: t == UInt8.self
        }
    }

    /// Element `i` converted to `Double`.
    public subscript(i: Int) -> Double {
        precondition(i >= 0 && i < count, "buffer index out of range")
        let p = base!
        return switch dtype {
        case .f64: p.load(fromByteOffset: i * 8, as: Double.self)
        case .f32: Double(p.load(fromByteOffset: i * 4, as: Float.self))
        case .i32: Double(p.load(fromByteOffset: i * 4, as: Int32.self))
        case .u32: Double(p.load(fromByteOffset: i * 4, as: UInt32.self))
        case .i64: Double(p.load(fromByteOffset: i * 8, as: Int64.self))
        case .u64: Double(p.load(fromByteOffset: i * 8, as: UInt64.self))
        case .u8: Double(p.load(fromByteOffset: i, as: UInt8.self))
        }
    }

    /// All elements converted to `Double` (a copy), one pass over the
    /// memory for the buffer's element type.
    public var doubles: [Double] {
        guard let base else { return [] }
        func all<T>(_: T.Type, _ convert: (T) -> Double) -> [Double] {
            UnsafeBufferPointer(start: base.assumingMemoryBound(to: T.self), count: count).map(convert)
        }
        return switch dtype {
        case .f64: Array(UnsafeBufferPointer(start: base.assumingMemoryBound(to: Double.self), count: count))
        case .f32: all(Float.self) { Double($0) }
        case .i32: all(Int32.self) { Double($0) }
        case .u32: all(UInt32.self) { Double($0) }
        case .i64: all(Int64.self) { Double($0) }
        case .u64: all(UInt64.self) { Double($0) }
        case .u8: all(UInt8.self) { Double($0) }
        }
    }

    /// Element at a multi-index (row-major).
    public subscript(index: [Int]) -> Double {
        precondition(index.count == shape.count, "wrong number of indices")
        var flat = 0
        for (i, n) in zip(index, shape) {
            precondition(i >= 0 && i < n, "buffer index out of range")
            flat = flat * n + i
        }
        return self[flat]
    }

    public var description: String {
        "\(name): \(dtype.name)[\(shape.map(String.init).joined(separator: "×"))]"
    }

    public static func == (a: ResultBuffer, b: ResultBuffer) -> Bool {
        a === b
            || (a.name == b.name && a.dtype == b.dtype && a.shape == b.shape
                && a.withUnsafeBytes { x in b.withUnsafeBytes { y in x.elementsEqual(y) } })
    }

    public func hash(into h: inout Hasher) {
        h.combine(name)
        h.combine(dtype)
        h.combine(shape)
    }
}

/// Heap bytes owned by Swift, for buffers built in Swift.
private final class OwnedBytes: @unchecked Sendable {
    let pointer: UnsafeMutableRawPointer
    init(_ data: Data) {
        pointer = UnsafeMutableRawPointer.allocate(byteCount: max(data.count, 1), alignment: 16)
        data.withUnsafeBytes { if let b = $0.baseAddress { pointer.copyMemory(from: b, byteCount: data.count) } }
    }
    deinit { pointer.deallocate() }
}
