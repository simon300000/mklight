import Foundation

/// The engine's `convert_spec` result: the system's fields and aperture in
/// another field kind or aperture type, each keeping the system equivalent
/// (docs/RESULTS.md). `nil` members were not asked for or are undefined for
/// the system (e.g. an object height at infinity).
public struct SpecConversion: Sendable, Hashable {
    /// One `(x, y)` per system field, in the requested kind.
    public var fieldValues: [[Double]]?
    public var aperture: ApertureSpec?
    /// Paraxial marginal-ray height at the stop (mm).
    public var stopSemiDiameter: Double?

    public init(fieldValues: [[Double]]? = nil, aperture: ApertureSpec? = nil, stopSemiDiameter: Double? = nil) {
        self.fieldValues = fieldValues
        self.aperture = aperture
        self.stopSemiDiameter = stopSemiDiameter
    }

    /// Reads the result JSON; `nil` when it is not a `convert_spec` result.
    public init?(json: JSONValue) {
        guard let w = try? json.decode(Wire.self) else { return nil }
        let values = w.fields?.values
        fieldValues = values?.allSatisfy({ $0.count == 2 }) == true ? values : nil
        aperture = w.aperture
        stopSemiDiameter = w.stopSemiDiameter
    }

    private struct Wire: Decodable {
        struct Fields: Decodable { var values: [[Double]] }
        var fields: Fields?
        var aperture: ApertureSpec?
        var stopSemiDiameter: Double?
        enum CodingKeys: String, CodingKey {
            case fields, aperture
            case stopSemiDiameter = "stop_semi_diameter"
        }
    }
}

extension OpticalSystem {
    /// Switches the field kind. With `conversion` (one value per field),
    /// every field gets its value in the new kind, so it keeps its chief
    /// ray; weights and vignetting stay. Without, the numbers stay as typed.
    public mutating func setFieldKind(_ kind: FieldKind, conversion: SpecConversion? = nil) {
        guard kind != fields.kind else { return }
        fields.kind = kind
        guard let values = conversion?.fieldValues, values.count == fields.fields.count else { return }
        for (k, v) in values.enumerated() {
            fields.fields[k].x = v[0]
            fields.fields[k].y = v[1]
        }
    }

    /// Switches the aperture type. With `conversion`, the new type's value
    /// keeps the paraxial marginal ray; switching to "float by stop size"
    /// gives an automatic stop semi-diameter the current beam's size. Without
    /// a converted value, the old number stays (or a default).
    public mutating func setApertureKind(_ kind: ApertureSpec.Kind, conversion: SpecConversion? = nil) {
        guard kind != aperture.kind else { return }
        if kind == .floatByStopSize {
            aperture = .floatByStopSize
            if surfaces.indices.contains(stop), surfaces[stop].semiDiameter == .auto, surfaces[stop].aperture == nil,
                let r = conversion?.stopSemiDiameter, r > 0, r.isFinite
            {
                surfaces[stop].semiDiameter = .fixed(r)
            }
            return
        }
        if let a = conversion?.aperture, a.kind == kind {
            aperture = a
            return
        }
        let v = aperture.value ?? (kind == .objectSpaceNA ? 0.1 : kind == .entrancePupilDiameter ? 10 : 4)
        aperture = ApertureSpec(kind: kind, value: v)
    }
}
