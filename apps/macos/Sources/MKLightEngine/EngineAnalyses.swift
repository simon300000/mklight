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
    public var settingsSchema: JSONValue
    /// Complete default settings; `null` when some setting is required.
    public var defaultSettings: JSONValue

    public var id: String { name }

    public init(
        name: String, title: String, category: String, description: String = "", available: Bool,
        unavailableReason: String? = nil, settingsSchema: JSONValue = .object([:]), defaultSettings: JSONValue = .null
    ) {
        self.name = name
        self.title = title
        self.category = category
        self.description = description
        self.available = available
        self.unavailableReason = unavailableReason
        self.settingsSchema = settingsSchema
        self.defaultSettings = defaultSettings
    }

    private enum CodingKeys: String, CodingKey {
        case name, title, category, description, available
        case unavailableReason = "unavailable_reason", settingsSchema = "settings_schema"
        case defaultSettings = "default_settings"
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
    }

    /// The schema's properties, in a stable order (required first, then by
    /// name).
    public var settingsFields: [SettingsField] {
        guard case .object(let props)? = settingsSchema["properties"] else { return [] }
        let required: Set<String> =
            if case .array(let r)? = settingsSchema["required"] {
                Set(r.compactMap { if case .string(let s) = $0 { s } else { nil } })
            } else { [] }
        return props.map { SettingsField(name: $0.key, schema: $0.value, required: required.contains($0.key)) }
            .sorted { ($0.required ? 0 : 1, $0.name) < ($1.required ? 0 : 1, $1.name) }
    }

    /// `settings` reduced to the keys the schema declares, with values of an
    /// accepted type. The engine rejects unknown keys, so requests built by
    /// the front end for an engine whose schema differs (a newer or older
    /// engine) degrade to the engine's defaults instead of failing.
    public func acceptedSettings(_ settings: [String: JSONValue]) -> [String: JSONValue] {
        let fields = Dictionary(uniqueKeysWithValues: settingsFields.map { ($0.name, $0) })
        return settings.filter { key, value in fields[key]?.accepts(value) ?? false }
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

/// A request for any engine analysis by name (`{"analysis", "settings",
/// "view_key", "backend", "allow_backend_fallback"}` on the C ABI).
/// `settings` may be partial: the engine fills in defaults and records the
/// complete settings in `RunInfo.settings`.
public struct EngineAnalysisRequest: Codable, Sendable, Hashable {
    public var analysis: String
    public var settings: [String: JSONValue]
    /// Backend preference and seed. The seed is sent as the `seed` setting
    /// when the analysis declares one.
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

/// One parameter edit (`{"path", "value"}`), applied by the engine.
public struct ParamEdit: Codable, Sendable, Hashable {
    public var path: ParamPath
    /// May be `±∞` for thicknesses (sent as `"inf"` / `"-inf"`).
    @EngineDouble public var value: Double

    public init(_ path: ParamPath, _ value: Double) {
        self.path = path
        self.value = value
    }
    private enum CodingKeys: String, CodingKey { case path, value }
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

// MARK: - Typed analyses on the engine

extension AnalysisKind {
    /// The engine's registry name of this analysis.
    public var engineName: String {
        switch self {
        case .firstOrder: "paraxial"
        default: rawValue
        }
    }
}

extension AnalysisRequest {
    /// The settings in engine form for `descriptor`: the Swift settings
    /// flattened (the execution block contributes `seed`; field and
    /// wavelength selections become indices, or are left to the engine's
    /// default for "all"/"primary"), then reduced to what the schema accepts.
    public func engineSettings(for descriptor: AnalysisDescriptor) -> [String: JSONValue] {
        guard case .object(var o) = (try? settingsJSON()) ?? .null else { return [:] }
        o.removeValue(forKey: "execution")
        o["seed"] = .unsigned(execution.seed)
        if let i = Int64(exactly: execution.seed) { o["seed"] = .integer(i) }
        for (k, v) in o {
            if case .object(let sel) = v, case .string(let t)? = sel["type"] {
                o[k] = t == "index" ? sel["index"] : nil
            }
        }
        return descriptor.acceptedSettings(o)
    }
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

    /// All elements converted to `Double` (a copy).
    public var doubles: [Double] { (0..<count).map { self[$0] } }

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

// MARK: - Generic results

/// An engine result shown generically: the analysis-specific JSON (`result`
/// of the envelope) and its binary buffers. Used for analyses without a
/// typed Swift mirror and for typed analyses whose JSON the mirror does not
/// (yet) decode, so a newer engine never breaks the app.
public struct GenericResult: Sendable, Hashable {
    public var analysis: String
    public var json: JSONValue
    public var buffers: [ResultBuffer]
    /// Why the typed mirror was not used, if a typed decode was attempted.
    public var note: String?

    public init(analysis: String, json: JSONValue, buffers: [ResultBuffer] = [], note: String? = nil) {
        self.analysis = analysis
        self.json = json
        self.buffers = buffers
        self.note = note
    }

    public func buffer(named name: String) -> ResultBuffer? { buffers.first { $0.name == name } }
}

extension AnalysisResult {
    /// Decodes an engine result for `kind` into its typed mirror. When that
    /// fails, first-order data is read from the numeric fields of the JSON
    /// (any engine layout of `{"efl": …, …}` works) and everything else
    /// falls back to `.generic`, never an error.
    public static func fromEngine(kind: AnalysisKind, json: JSONValue, buffers: [ResultBuffer]) -> AnalysisResult {
        func typed<T: Decodable>(_ t: T.Type) -> Result<T, any Error> {
            Result { try MKLightJSON.decode(T.self, from: MKLightJSON.encode(json)) }
        }
        let decoded: Result<AnalysisResult, any Error> =
            switch kind {
            case .layout: typed(LayoutScene.self).map { .layout($0) }
            case .spotDiagram: typed(SpotDiagramResult.self).map { .spotDiagram($0) }
            case .rayFan: typed(FanResult.self).map { .rayFan($0) }
            case .opdFan: typed(FanResult.self).map { .opdFan($0) }
            case .fieldCurvature: typed(FieldCurvatureResult.self).map { .fieldCurvature($0) }
            case .mtf: typed(MTFResult.self).map { .mtf($0) }
            case .psf: typed(PSFResult.self).map { .psf($0) }
            case .firstOrder: typed(FirstOrderResult.self).map { .firstOrder($0) }
            }
        switch decoded {
        case .success(let r): return r
        case .failure(let error):
            if kind == .firstOrder, let fo = FirstOrderResult(engineJSON: json) { return .firstOrder(fo) }
            return .generic(
                GenericResult(
                    analysis: kind.engineName, json: json, buffers: buffers,
                    note: "not in the app's \(kind.title) format (\(error.localizedDescription)); shown as data"))
        }
    }
}

extension FirstOrderResult {
    /// First-order entries from any JSON object: every number (or `±inf`
    /// string) at the top level or one object level down becomes an entry
    /// keyed by its field name (the first occurrence wins). `nil` when there
    /// is none.
    public init?(engineJSON json: JSONValue) {
        guard case .object(let top) = json else { return nil }
        var entries: [Entry] = []
        var seen = Set<String>()
        func add(_ key: String, _ v: JSONValue) {
            guard seen.insert(key).inserted, let d = v.doubleValue else { return }
            if case .string = v, !d.isInfinite { return }
            let label = key.replacingOccurrences(of: "_", with: " ").capitalized
                .replacingOccurrences(of: "Efl", with: "EFL").replacingOccurrences(of: "Bfl", with: "BFL")
            entries.append(Entry(key: key, label: label, value: d, unit: ""))
        }
        for (k, v) in top.sorted(by: { $0.key < $1.key }) where v.doubleValue != nil { add(k, v) }
        for (_, v) in top.sorted(by: { $0.key < $1.key }) {
            if case .object(let inner) = v {
                for (k, w) in inner.sorted(by: { $0.key < $1.key }) { add(k, w) }
            }
        }
        guard !entries.isEmpty else { return nil }
        let wl = top["wavelength_um"]?.doubleValue ?? 0
        self.init(entries: entries.filter { $0.key != "wavelength_um" }, wavelengthUm: wl)
    }
}
