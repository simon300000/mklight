import Foundation
import MKLightEngine
import MKLightModel

/// How a stored shape parameter is shown and entered.
public enum ParamDisplay: Sendable, Equatable {
    /// The stored value as is.
    case value
    /// A stored curvature shown and entered as a radius (∞ for 0).
    case radiusOfCurvature
}

/// One scalar parameter of a surface type, addressed by its key in the
/// shape's JSON object (the engine's model form).
public struct ShapeParam: Identifiable, Sendable {
    public var key: String
    public var title: String
    public var unit: String
    public var display: ParamDisplay = .value
    public var help: String = ""
    /// The `ParamPath` that makes it a variable, if the engine has one.
    public var path: (@Sendable (SurfaceID) -> ParamPath)?
    public var id: String { key }
}

/// A list of coefficients (polynomial terms) with add and remove.
public struct CoefficientList: Sendable {
    public var key: String
    public var title: String
    /// Label of term `i`, e.g. `A4`.
    public var term: @Sendable (Int) -> String
    /// Unit of term `i`, e.g. `mm⁻³`.
    public var unit: @Sendable (Int) -> String
    public var path: (@Sendable (SurfaceID, Int) -> ParamPath)?
    public var maxCount: Int
}

public struct ParamSection: Identifiable, Sendable {
    public var title: String
    public var params: [ShapeParam] = []
    public var coefficients: CoefficientList?
    public var id: String { title }
}

/// A surface type described as data: its sections of parameters, how to
/// convert another shape into it, and its sag for the profile preview.
/// The inspector builds its form from this, so a new surface type needs a
/// model case (its JSON form) and one descriptor here, not a new view.
public struct SurfaceTypeDescriptor: Identifiable, Sendable {
    /// The shape's `type` tag in the model JSON.
    public var id: String
    public var kind: SurfaceShape.Kind
    public var title: String
    public var symbol: String
    public var summary: String
    public var sections: [ParamSection]
    /// Placement (decenters, tilts) is the type's main content.
    public var placementFirst = false
    /// Sag `z(r)` from the shape JSON; `nil` where it has none.
    public var sag: (@Sendable (JSONValue, Double) -> Double?)?
}

public enum SurfaceTypes {
    static let radius = ShapeParam(
        key: "curvature", title: "Radius", unit: "mm", display: .radiusOfCurvature,
        help: "Radius of curvature; ∞ for a plane. Stored as curvature c = 1/R.", path: { .curvature(surface: $0) })
    static let conic = ShapeParam(
        key: "conic", title: "Conic", unit: "", help: "Conic constant k: 0 sphere, −1 paraboloid, < −1 hyperboloid.",
        path: { .conic(surface: $0) })

    /// Conic sag `c r² / (1 + √(1 − (1 + k) c² r²))`.
    static func conicSag(_ c: Double, _ k: Double, _ r: Double) -> Double? {
        let q = 1 - (1 + k) * c * c * r * r
        guard q >= 0 else { return nil }
        return c * r * r / (1 + q.squareRoot())
    }

    public static let all: [SurfaceTypeDescriptor] = [
        SurfaceTypeDescriptor(
            id: "plane", kind: .plane, title: "Plane", symbol: "rectangle.portrait",
            summary: "A flat surface.", sections: [], sag: { _, _ in 0 }),
        SurfaceTypeDescriptor(
            id: "conic", kind: .conic, title: "Sphere / Conic", symbol: "circle.lefthalf.filled",
            summary: "Sphere or conic of revolution.",
            sections: [ParamSection(title: "Shape", params: [radius, conic])],
            sag: { j, r in conicSag(j["curvature"]?.nf ?? 0, j["conic"]?.nf ?? 0, r) }),
        SurfaceTypeDescriptor(
            id: "even_asphere", kind: .evenAsphere, title: "Even Asphere", symbol: "waveform.path",
            summary: "Conic plus even polynomial terms A4 r⁴ + A6 r⁶ + …",
            sections: [
                ParamSection(title: "Base Conic", params: [radius, conic]),
                ParamSection(
                    title: "Polynomial",
                    coefficients: CoefficientList(
                        key: "coefficients", title: "Coefficients", term: { "A\(2 * $0 + 4)" },
                        unit: { "mm⁻\(superscript(2 * $0 + 3))" }, path: { .asphere(surface: $0, index: $1) }, maxCount: 10)),
            ],
            sag: { j, r in
                guard let base = conicSag(j["curvature"]?.nf ?? 0, j["conic"]?.nf ?? 0, r) else { return nil }
                let a = j["coefficients"]?.nfArray ?? []
                return base + a.enumerated().reduce(0) { $0 + $1.element * pow(r, Double(2 * $1.offset + 4)) }
            }),
        SurfaceTypeDescriptor(
            id: "paraxial", kind: .paraxial, title: "Ideal Lens", symbol: "lines.measurement.horizontal",
            summary: "A thin, aberration-free lens of the given focal length.",
            sections: [
                ParamSection(
                    title: "Ideal Lens",
                    params: [
                        ShapeParam(
                            key: "focal_length", title: "Focal length", unit: "mm", help: "Focal length of the ideal lens.",
                            path: { .paraxialFocalLength(surface: $0) })
                    ])
            ]),
        SurfaceTypeDescriptor(
            id: "coordinate_break", kind: .coordinateBreak, title: "Coordinate Break", symbol: "move.3d",
            summary: "Changes the coordinate frame only: decenters, then tilts (or the reverse).", sections: [],
            placementFirst: true),
    ]

    public static func descriptor(for shape: SurfaceShape) -> SurfaceTypeDescriptor { descriptor(for: shape.kind) }

    public static func descriptor(for kind: SurfaceShape.Kind) -> SurfaceTypeDescriptor {
        all.first { $0.kind == kind } ?? all[0]
    }

    /// The shape in its model JSON form.
    public static func json(_ shape: SurfaceShape) -> JSONValue { (try? JSONValue.from(shape)) ?? .null }

    /// The shape with one JSON key replaced, or `nil` when the model does
    /// not accept the result.
    public static func shape(_ shape: SurfaceShape, setting key: String, to value: JSONValue) -> SurfaceShape? {
        guard case .object(var o) = json(shape) else { return nil }
        o[key] = value
        return try? MKLightJSON.decode(SurfaceShape.self, from: MKLightJSON.encode(JSONValue.object(o)))
    }

    /// Display value of a parameter (a radius for a curvature).
    public static func displayValue(_ p: ShapeParam, in shape: SurfaceShape) -> Double? {
        guard let v = json(shape)[p.key]?.nf else { return nil }
        return p.display == .radiusOfCurvature ? RadiusDisplay.radius(fromCurvature: v) : v
    }

    /// Parses entered text for a parameter into its stored JSON value.
    public static func storedValue(_ p: ShapeParam, text: String) -> Result<JSONValue, LensValueParser.Failure> {
        switch p.display {
        case .radiusOfCurvature: LensValueParser.curvature(fromRadiusText: text).map { .number($0) }
        case .value: LensValueParser.finite(text).map { .number($0) }
        }
    }

    static func superscript(_ n: Int) -> String {
        let digits = ["⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹"]
        return String(n).compactMap { $0.wholeNumberValue.map { digits[$0] } }.joined()
    }

    /// Sag profile points `(r, z)` from −h to h for the preview, or `nil`.
    public static func profile(_ shape: SurfaceShape, semiDiameter h: Double, samples: Int = 64) -> [(Double, Double)]? {
        guard let sag = descriptor(for: shape).sag, h > 0, h.isFinite else { return nil }
        let j = json(shape)
        var out: [(Double, Double)] = []
        for k in 0...samples {
            let r = -h + 2 * h * Double(k) / Double(samples)
            guard let z = sag(j, abs(r)), z.isFinite else { continue }
            out.append((r, z))
        }
        return out.count > 2 ? out : nil
    }
}
