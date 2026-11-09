import Foundation
import MKLightEngine
import MKLightModel

/// The control an analysis setting gets, chosen from its schema (`SettingsField`)
/// alone, so every analysis the engine publishes is editable without typed
/// JSON: scalars become fields, pickers and toggles; selections of fields,
/// wavelengths and surfaces become menus; objects become groups of their
/// properties (optional ones with an on/off switch); tagged unions become a
/// variant picker with the chosen variant's fields; lists become multi-select
/// menus, chips or small tables. A few settings keep a dedicated control
/// (the spectrum, the pupil sampling, the image source). Only settings whose
/// shape has no sensible control are hidden, and those keep their value.
public indirect enum SettingControl: Equatable, Sendable {
    /// Not shown (kept as it is).
    case hidden
    case toggle
    /// A number field; `integer` restricts to whole numbers.
    case number(integer: Bool)
    /// A free text field.
    case text
    /// One of the schema's `enum` values.
    case picker
    /// `{"type": "all" | "index" | "list"}` selections of the system's fields.
    case fieldSelection
    /// `{"type": "all" | "primary" | "index" | "list"}` wavelength selections.
    case wavelengthSelection
    case fieldIndex, wavelengthIndex, surfaceIndex
    /// `[x, y]` of two numbers (`integer` when both defaults are); `nullable`
    /// adds an automatic state.
    case pair(integer: Bool, nullable: Bool)
    /// `[i, …]` of system field, wavelength or surface indices (empty: all).
    case indexList(IndexDomain)
    /// `[x, …]` of numbers, comma separated.
    case numberList
    /// Rows of the same columns: an array of objects with scalar properties,
    /// or of fixed-length numeric arrays (`columns` then carry `"0"`, `"1"`…).
    case table(columns: [SettingsField], rowsAreArrays: Bool)
    /// An object with known properties; `nullable` adds an on/off switch.
    case group(nullable: Bool)
    /// A tagged union: a picker of the variants, then the variant's fields.
    case variants(nullable: Bool)
    // Dedicated controls.
    case spectrum, pupil, frequencies, imageSource, imageOptics, sensorLimit, throughFocus

    public enum IndexDomain: Equatable, Sendable { case fields, wavelengths, surfaces }
}

public enum SettingControls {
    /// The control of `field` in `analysis`. `hasConfigurations` says whether
    /// the system has configurations (the `configuration` index is hidden
    /// otherwise); `pattern` is the analysis' pupil pattern setting, which
    /// decides whether a `seed` matters.
    public static func control(
        for f: SettingsField, analysis: String, hasConfigurations: Bool = false, pattern: String? = nil, nested: Bool = false
    ) -> SettingControl {
        let types = f.baseTypes
        let isSelection = types.contains("object") && f.defaultValue?["type"]?.string != nil && f.variants.isEmpty
        let integer = types.contains("integer")
        if !nested {
            if f.name == "fields" && isSelection { return .fieldSelection }
            if ["wavelengths", "wavelength"].contains(f.name) && isSelection { return .wavelengthSelection }
            if f.name == "field" && integer { return .fieldIndex }
            if ["wavelength", "long_wavelength", "short_wavelength"].contains(f.name) && integer { return .wavelengthIndex }
            if f.name == "surface" && integer { return .surfaceIndex }
            if f.name == "spectrum" && types.contains("object") { return .spectrum }
            if f.name == "pupil" && types.contains("object") && f.defaultValue?["grid"] != nil { return .pupil }
            if f.name == "frequencies" && AutoSettings.sentinel(f.name, analysis: analysis) == .empty { return .frequencies }
            if f.name == "through_focus" && analysis == "spot_diagram" && types == ["object"] { return .throughFocus }
            if f.name == "sensor_limit" && types == ["array"] { return .sensorLimit }
            if f.name == "source" && types == ["object"] && f.defaultValue?["kind"]?.string == "pattern" { return .imageSource }
            if f.name == "optics" && types == ["object"] && f.defaultValue?["psf_size"] != nil { return .imageOptics }
            switch f.name {
            case "configuration": return hasConfigurations ? .number(integer: true) : .hidden
            case "delay_ms", "chunk": return .hidden
            case "seed":
                // A seed matters only for random pupil patterns, where the analysis has one.
                if let pattern, !["random", "dithered"].contains(pattern) { return .hidden }
            default: break
            }
        }
        if !f.enumValues.isEmpty { return .picker }
        if !f.variants.isEmpty { return .variants(nullable: f.nullable) }
        if types == ["boolean"] { return .toggle }
        if !types.isEmpty && types.isSubset(of: ["number", "integer"]) { return .number(integer: integer && !types.contains("number")) }
        if types == ["string"] { return .text }
        if types == ["array"] { return arrayControl(f) }
        if types == ["object"] { return f.properties.isEmpty ? .hidden : .group(nullable: f.nullable) }
        return .hidden
    }

    /// The control of an array setting, from its items and default.
    static func arrayControl(_ f: SettingsField) -> SettingControl {
        let d = f.defaultValue?.array
        let itemType = f.items?["type"]?.string
        // Two numbers: a pair (`[x, y]`), with an automatic state when null is allowed.
        if let d, d.count == 2, d.allSatisfy({ $0.doubleValue != nil }) {
            let integer = d.allSatisfy { if case .integer = $0 { true } else { false } } || itemType == "integer"
            return .pair(integer: integer, nullable: f.nullable)
        }
        if (d == nil || d?.isEmpty == true) && f.nullable && (itemType == "number" || f.items == nil) {
            // `[min, max]` or `[x, y]` ranges that are automatic by default.
            return .pair(integer: false, nullable: true)
        }
        if itemType == "integer" {
            if let domain = indexDomain(f.name) { return .indexList(domain) }
            return .numberList
        }
        if itemType == "number" { return .numberList }
        // Rows of scalars: objects with scalar properties, or fixed-length numeric arrays.
        if let itemProps = f.items.flatMap({ SettingsField(name: f.name, schema: $0, required: false).properties }), !itemProps.isEmpty,
            itemProps.allSatisfy({ $0.baseTypes.isSubset(of: ["number", "integer", "boolean", "string"]) })
        {
            return .table(columns: itemProps, rowsAreArrays: false)
        }
        if let first = d?.first {
            if case .object(let o) = first, o.values.allSatisfy({ $0.doubleValue != nil || $0.bool != nil || $0.string != nil }) {
                let cols = o.keys.sorted().map { k in
                    SettingsField(name: k, schema: .object(["type": .string(SettingsField.typeName(o[k]!)), "default": o[k]!]), required: false)
                }
                return .table(columns: cols, rowsAreArrays: false)
            }
            if case .array(let row) = first, !row.isEmpty, row.allSatisfy({ $0.doubleValue != nil }) {
                let cols = row.indices.map { k in SettingsField(name: "\(k)", schema: .object(["type": .string("number"), "default": row[k]]), required: false) }
                return .table(columns: cols, rowsAreArrays: true)
            }
            return .hidden
        }
        // Empty by default with items undocumented: the description names the
        // row keys in the engine's `[{"hx", "hy"}]` form.
        if let keys = rowKeys(in: f.description) {
            let cols = keys.map { SettingsField(name: $0, schema: .object(["type": .string("number"), "default": .number(0)]), required: false) }
            return .table(columns: cols, rowsAreArrays: false)
        }
        return .hidden
    }

    /// Which indices an index list names, by the setting's name.
    static func indexDomain(_ name: String) -> SettingControl.IndexDomain? {
        switch name {
        case "fields": .fields
        case "wavelengths": .wavelengths
        case "surfaces", "exclude_surfaces": .surfaces
        default: nil
        }
    }

    /// The row keys a description states as `[{"hx", "hy"}]`.
    static func rowKeys(in description: String) -> [String]? {
        guard let r = description.range(of: #"\[\{("[a-z_0-9]+"(,\s*)?)+\}\]"#, options: .regularExpression) else { return nil }
        let keys = description[r].split(whereSeparator: { !$0.isLetter && !$0.isNumber && $0 != "_" }).map(String.init)
        return keys.isEmpty ? nil : keys
    }

    /// The value a control starts from when a nullable setting is switched
    /// on: the field's non-null default, else an object of the properties'
    /// defaults, the first variant, or an empty pair.
    public static func enabledValue(_ f: SettingsField) -> JSONValue {
        if let d = f.defaultValue, d != .null { return d }
        if let v = f.variants.first { return v.defaultObject }
        if !f.properties.isEmpty {
            return .object(Dictionary(uniqueKeysWithValues: f.properties.compactMap { p in p.defaultValue.map { (p.name, $0) } }))
        }
        if f.baseTypes == ["array"] { return .array([.number(0), .number(0)]) }
        if f.baseTypes == ["boolean"] { return .bool(true) }
        if f.baseTypes.contains("string") { return .string("") }
        return .number(0)
    }
}
