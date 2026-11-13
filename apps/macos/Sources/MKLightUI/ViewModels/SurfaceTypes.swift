import Foundation
import MKLightEngine
import MKLightModel

/// How a stored shape parameter is shown and entered.
public enum ParamDisplay: Sendable, Equatable {
    /// The stored value as is.
    case value
    /// A stored curvature shown and entered as a radius (∞ for 0).
    case radiusOfCurvature
    /// A signed integer (diffraction orders).
    case integer
}

/// One scalar parameter of a surface type, addressed by its key path in the
/// shape's JSON object (the engine's model form), e.g. `curvature` or
/// `phase.norm_radius`.
public struct ShapeParam: Identifiable, Sendable {
    public var key: String
    public var title: String
    public var unit: String
    public var display: ParamDisplay = .value
    public var help: String = ""
    /// The `ParamPath` that makes it a variable, if it may be one.
    public var path: (@Sendable (SurfaceID) -> ParamPath)?
    public var id: String { key }
}

/// A parameter with a fixed set of values (e.g. the Zernike ordering).
public struct ShapeChoice: Identifiable, Sendable {
    public var key: String
    public var title: String
    public var options: [(value: String, title: String)]
    public var help: String = ""
    public var id: String { key }
}

/// A list of coefficients (polynomial terms) with add and remove.
public struct CoefficientList: Sendable {
    public var key: String
    public var title: String
    /// Symbol of term `i` on a shape, e.g. `A₄`, `β₃`, `Q₂`, `C(x²y)`, `Z₇`.
    public var term: @Sendable (Int, SurfaceShape) -> String
    /// What term `i` multiplies or means, e.g. `r⁴`, `u⁴Q₂ᶜᵒⁿ`, `coma x`.
    public var meaning: @Sendable (Int, SurfaceShape) -> String
    /// Unit of term `i`, e.g. `mm⁻³`.
    public var unit: @Sendable (Int) -> String
    public var path: (@Sendable (SurfaceID, Int) -> ParamPath)?
    public var maxCount: @Sendable (SurfaceShape) -> Int
    /// Long term lists (XY, Zernike): add any term from a menu and hide
    /// zero terms by default.
    public var sparse = false
}

public struct ParamSection: Identifiable, Sendable {
    public var title: String
    public var params: [ShapeParam] = []
    public var choices: [ShapeChoice] = []
    public var coefficients: CoefficientList?
    public var footer: String = ""
    /// Whether the section applies to this shape (e.g. a DOE's base conic
    /// only when the base is not a plane).
    public var applies: @Sendable (SurfaceShape) -> Bool = { _ in true }
    public var id: String { title }
}

/// Groups of the type menus.
public enum SurfaceTypeGroup: String, CaseIterable, Sendable, Identifiable {
    case spherical, aspheric, freeform, anamorphic, diffractive, special
    public var id: String { rawValue }
    public var title: String {
        switch self {
        case .spherical: "Spherical"
        case .aspheric: "Aspheric"
        case .freeform: "Freeform"
        case .anamorphic: "Anamorphic"
        case .diffractive: "Diffractive"
        case .special: "Special"
        }
    }
}

/// How the inspector previews a type.
public enum ShapePreviewKind: Sendable {
    /// One meridional profile (rotationally symmetric shapes).
    case radial
    /// x and y profiles.
    case anamorphic
    /// x and y profiles and a false-colour departure map.
    case freeform
    /// The base profile plus the phase profile, power and zone period.
    case diffractive
    case idealLens, coordinateBreak
}

/// A surface type described as data: its sections of parameters and its
/// preview. The inspector builds its form from this, so a new surface type
/// needs a model case (its JSON form) and one descriptor here, not a new
/// view. Sags for the previews come from `ShapeSag`.
public struct SurfaceTypeDescriptor: Identifiable, Sendable {
    /// The shape's `type` tag in the model JSON.
    public var id: String
    public var kind: SurfaceShape.Kind
    public var title: String
    public var symbol: String
    public var group: SurfaceTypeGroup
    public var summary: String
    public var sections: [ParamSection]
    /// Placement (decenters, tilts) is the type's main content.
    public var placementFirst = false
    public var preview = ShapePreviewKind.radial
    /// The surface's diffraction order efficiency model (ideal, kinoform,
    /// multilevel) applies: a property of the surface, not of its shape.
    public var orderEfficiency = false
}

/// An entry of the type menus: a model type, or a preset of one (a
/// cylinder is a biconic with one plane axis).
public enum SurfaceTypeChoice: Hashable, Sendable, Identifiable {
    case kind(SurfaceShape.Kind)
    case cylinder(SurfaceShape.CylinderAxis)

    public var id: String {
        switch self {
        case .kind(let k): k.rawValue
        case .cylinder(let a): "cylinder-\(a.rawValue)"
        }
    }
}

public enum SurfaceTypes {
    // MARK: Building blocks

    static func radius(_ key: String = "curvature", _ title: String = "Radius", path: (@Sendable (SurfaceID) -> ParamPath)?) -> ShapeParam {
        ShapeParam(
            key: key, title: title, unit: "mm", display: .radiusOfCurvature,
            help: "Radius of curvature; ∞ for a plane. Stored as curvature c = 1/R.", path: path)
    }
    static let radius = radius(path: { .curvature(surface: $0) })
    static let conic = ShapeParam(
        key: "conic", title: "Conic", unit: "", help: "Conic constant k: 0 sphere, −1 paraboloid, < −1 hyperboloid.",
        path: { .conic(surface: $0) })
    static func normRadius(_ key: String = "norm_radius") -> ShapeParam {
        ShapeParam(
            key: key, title: "Norm radius", unit: "mm",
            help: "Normalization radius r_norm: the polynomial is defined for r ≤ r_norm. Usually the clear semi-diameter.",
            path: { .shapeParam(surface: $0, parameter: .normRadius) })
    }

    static func shapeCoefficient(_ set: CoefficientSet) -> @Sendable (SurfaceID, Int) -> ParamPath {
        { .shapeCoefficient(surface: $0, set: set, index: $1) }
    }
    static func limit(_ set: CoefficientSet) -> @Sendable (SurfaceShape) -> Int {
        { set.maxCount(for: $0) ?? 0 }
    }

    /// Unicode subscript and superscript digits, with a sign.
    static func sub(_ n: Int) -> String { script(n, digits: ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"], minus: "₋") }
    static func superscript(_ n: Int) -> String { script(n, digits: ["⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹"], minus: "⁻") }
    static func script(_ n: Int, digits: [String], minus: String) -> String {
        (n < 0 ? minus : "") + String(abs(n)).compactMap { $0.wholeNumberValue.map { digits[$0] } }.joined()
    }
    static func mmPower(_ p: Int) -> String { p == 0 ? "" : p == 1 ? "mm" : "mm" + superscript(p) }

    /// Even axis polynomial of a toroid or biconic: `A₄ x⁴`, `B₆ y⁶`, …
    static func axisList(_ axis: String, set: CoefficientSet, letter: String) -> CoefficientList {
        CoefficientList(
            key: set.rawValue, title: "\(axis.uppercased()) Polynomial", term: { k, _ in "\(letter)\(sub(2 * k + 4))" },
            meaning: { k, _ in axis + superscript(2 * k + 4) }, unit: { mmPower(-(2 * $0 + 3)) },
            path: shapeCoefficient(set), maxCount: limit(set))
    }

    /// A name for Zernike term `(n, m)`.
    public static func zernikeName(_ t: ZernikeTerm) -> String {
        let m = abs(t.m)
        let axis = t.m > 0 ? " x" : t.m < 0 ? " y" : ""
        let name: String? =
            switch (t.n, m) {
            case (0, 0): "piston"
            case (1, 1): "tilt"
            case (2, 0): "defocus"
            case (2, 2): "astigmatism"
            case (3, 1): "coma"
            case (3, 3): "trefoil"
            case (4, 0): "spherical"
            case (4, 2): "2nd astigmatism"
            case (4, 4): "tetrafoil"
            case (5, 1): "2nd coma"
            case (5, 3): "2nd trefoil"
            case (5, 5): "pentafoil"
            case (6, 0): "2nd spherical"
            case (8, 0): "3rd spherical"
            default: nil
            }
        if let name { return name + (m == 0 ? "" : (t.n == 1 || t.n == 3 || t.n == 5 ? axis : t.m > 0 ? " 0°" : " 45°")) }
        return "n\(t.n) m\(t.m)"
    }

    // MARK: Descriptors

    public static let all: [SurfaceTypeDescriptor] = [
        SurfaceTypeDescriptor(
            id: "plane", kind: .plane, title: "Plane", symbol: "rectangle.portrait", group: .spherical,
            summary: "A flat surface.", sections: []),
        SurfaceTypeDescriptor(
            id: "conic", kind: .conic, title: "Sphere / Conic", symbol: "circle.lefthalf.filled", group: .spherical,
            summary: "Sphere or conic of revolution.",
            sections: [ParamSection(title: "Shape", params: [radius, conic])]),
        SurfaceTypeDescriptor(
            id: "even_asphere", kind: .evenAsphere, title: "Even Asphere", symbol: "waveform.path", group: .aspheric,
            summary: "Conic plus even polynomial terms A₄r⁴ + A₆r⁶ + …",
            sections: [
                ParamSection(title: "Base Conic", params: [radius, conic]),
                ParamSection(
                    title: "Polynomial",
                    coefficients: CoefficientList(
                        key: "coefficients", title: "Coefficients", term: { k, _ in "A\(sub(2 * k + 4))" },
                        meaning: { k, _ in "r" + superscript(2 * k + 4) }, unit: { mmPower(-(2 * $0 + 3)) },
                        path: { .asphere(surface: $0, index: $1) }, maxCount: { _ in 10 })),
            ]),
        SurfaceTypeDescriptor(
            id: "odd_asphere", kind: .oddAsphere, title: "Odd Asphere", symbol: "waveform.path.ecg", group: .aspheric,
            summary: "Conic plus all powers of r: β₁r + β₂r² + β₃r³ + … A nonzero β₁ is a cone at the vertex.",
            sections: [
                ParamSection(title: "Base Conic", params: [radius, conic]),
                ParamSection(
                    title: "Polynomial",
                    coefficients: CoefficientList(
                        key: "coefficients", title: "Coefficients", term: { k, _ in "β\(sub(k + 1))" },
                        meaning: { k, _ in k == 0 ? "r" : "r" + superscript(k + 1) }, unit: { mmPower(-$0) },
                        path: shapeCoefficient(.odd), maxCount: limit(.odd)),
                    footer: "The paraxial curvature is c + 2β₂; β₁ has no first-order meaning."),
            ]),
        SurfaceTypeDescriptor(
            id: "q_con", kind: .qCon, title: "Q-Type (Conic)", symbol: "q.circle", group: .aspheric,
            summary: "Forbes conic-based asphere: conic plus u⁴ Σ aₘ Qₘᶜᵒⁿ(u²), u = r/r_norm. Keeps the vertex curvature.",
            sections: [
                ParamSection(title: "Base Conic", params: [radius, conic, normRadius()]),
                ParamSection(
                    title: "Q Polynomial",
                    coefficients: CoefficientList(
                        key: "coefficients", title: "Coefficients", term: { k, _ in "a\(sub(k))" },
                        meaning: { k, _ in "u⁴Q\(sub(k))ᶜᵒⁿ" }, unit: { _ in "mm" }, path: shapeCoefficient(.q),
                        maxCount: limit(.q))),
            ]),
        SurfaceTypeDescriptor(
            id: "q_bfs", kind: .qBfs, title: "Q-Type (Best-Fit Sphere)", symbol: "q.square", group: .aspheric,
            summary: "Forbes slope-orthogonal departure from a best-fit sphere: u²(1−u²) Σ aₘ Qₘᵇᶠˢ(u²) / √(1−c²r²).",
            sections: [
                ParamSection(title: "Best-Fit Sphere", params: [radius, normRadius()]),
                ParamSection(
                    title: "Q Polynomial",
                    coefficients: CoefficientList(
                        key: "coefficients", title: "Coefficients", term: { k, _ in "a\(sub(k))" },
                        meaning: { k, _ in "Q\(sub(k))ᵇᶠˢ" }, unit: { _ in "mm" }, path: shapeCoefficient(.q),
                        maxCount: limit(.q)),
                    footer: "Vertex curvature c_bfs + 2Σaₘ Qₘ(0)/r_norm²."),
            ]),
        SurfaceTypeDescriptor(
            id: "xy_polynomial", kind: .xyPolynomial, title: "XY Polynomial", symbol: "square.grid.3x3", group: .freeform,
            summary: "Freeform: conic plus Σ C(xᵐyⁿ) Xᵐ Yⁿ with X = x/r_norm, Y = y/r_norm; degrees 1 to 10.",
            sections: [
                ParamSection(title: "Base Conic", params: [radius, conic, normRadius()]),
                ParamSection(
                    title: "XY Terms",
                    coefficients: CoefficientList(
                        key: "coefficients", title: "Coefficients",
                        term: { k, _ in XYTerm(index: k).map { "C(\($0.monomial))" } ?? "C\(k)" },
                        meaning: { k, _ in XYTerm(index: k).map { "degree \($0.degree)" } ?? "" }, unit: { _ in "mm" },
                        path: shapeCoefficient(.xy), maxCount: limit(.xy), sparse: true)),
            ],
            preview: .freeform),
        SurfaceTypeDescriptor(
            id: "zernike_sag", kind: .zernikeSag, title: "Zernike Sag", symbol: "circle.hexagongrid", group: .freeform,
            summary: "Freeform: conic plus Σ cⱼ Zⱼ(ρ, θ), ρ = r/r_norm, θ from +x. Sag coefficients in mm, piston and tilt kept.",
            sections: [
                ParamSection(
                    title: "Base Conic", params: [radius, conic, normRadius()],
                    choices: [
                        ShapeChoice(
                            key: "ordering", title: "Ordering",
                            options: [("fringe", "Fringe (37 terms)"), ("standard", "Standard / Noll (45 terms)")],
                            help: "Fringe terms are unnormalized; Standard (Noll) terms are normalized.")
                    ]),
                ParamSection(
                    title: "Zernike Terms",
                    coefficients: CoefficientList(
                        key: "coefficients", title: "Coefficients", term: { k, _ in "Z\(sub(k + 1))" },
                        meaning: { k, shape in
                            guard case .zernikeSag(_, _, _, let o, _) = shape, let t = ZernikeTerm(j: k + 1, ordering: o) else { return "" }
                            return zernikeName(t)
                        },
                        unit: { _ in "mm" }, path: shapeCoefficient(.zernike), maxCount: limit(.zernike), sparse: true)),
            ],
            preview: .freeform),
        SurfaceTypeDescriptor(
            id: "toroid", kind: .toroid, title: "Toroid", symbol: "capsule", group: .anamorphic,
            summary: "The y profile (conic plus even terms) rotated about an axis parallel to y at the distance R_x.",
            sections: [
                ParamSection(
                    title: "Y Profile",
                    params: [
                        radius("curvature_y", "Radius y", path: { .curvature(surface: $0) }),
                        ShapeParam(
                            key: "conic_y", title: "Conic y", unit: "", help: "Conic constant of the y profile.",
                            path: { .shapeParam(surface: $0, parameter: .conicY) }),
                    ],
                    coefficients: axisList("y", set: .yAsphere, letter: "B")),
                ParamSection(
                    title: "Rotation",
                    params: [radius("curvature_x", "Radius x", path: { .shapeParam(surface: $0, parameter: .curvatureX) })],
                    footer: "Radius x is the distance from the vertex to the axis of rotation; ∞ makes a y cylinder."),
            ],
            preview: .anamorphic),
        SurfaceTypeDescriptor(
            id: "biconic", kind: .biconic, title: "Biconic", symbol: "oval", group: .anamorphic,
            summary: "Independent conics in x and y with even axis polynomials. A cylinder is a biconic with one plane axis.",
            sections: [
                ParamSection(
                    title: "X Section",
                    params: [
                        radius("curvature_x", "Radius x", path: { .shapeParam(surface: $0, parameter: .curvatureX) }),
                        ShapeParam(
                            key: "conic_x", title: "Conic x", unit: "", help: "Conic constant in x.",
                            path: { .shapeParam(surface: $0, parameter: .conicX) }),
                    ],
                    coefficients: axisList("x", set: .xAsphere, letter: "A")),
                ParamSection(
                    title: "Y Section",
                    params: [
                        radius("curvature_y", "Radius y", path: { .curvature(surface: $0) }),
                        ShapeParam(
                            key: "conic_y", title: "Conic y", unit: "", help: "Conic constant in y.",
                            path: { .shapeParam(surface: $0, parameter: .conicY) }),
                    ],
                    coefficients: axisList("y", set: .yAsphere, letter: "B")),
            ],
            preview: .anamorphic),
        SurfaceTypeDescriptor(
            id: "diffractive", kind: .diffractive, title: "Diffractive", symbol: "rainbow", group: .diffractive,
            summary: "A fixed-period phase Φ = Σ Aᵢ ρ²ⁱ (order +1, radians) on a plane, conic or even-asphere base.",
            sections: [
                ParamSection(
                    title: "Base Shape", params: [radius("base.curvature", path: { .curvature(surface: $0) }), conicAt("base.conic")],
                    applies: { if case .diffractive(let b, _) = $0 { return b.kind != .plane }; return false }),
                ParamSection(
                    title: "Base Polynomial",
                    coefficients: CoefficientList(
                        key: "base.coefficients", title: "Coefficients", term: { k, _ in "A\(sub(2 * k + 4))" },
                        meaning: { k, _ in "r" + superscript(2 * k + 4) }, unit: { mmPower(-(2 * $0 + 3)) },
                        path: { .asphere(surface: $0, index: $1) }, maxCount: { _ in 10 }),
                    applies: { if case .diffractive(let b, _) = $0 { return b.kind == .evenAsphere }; return false }),
                ParamSection(
                    title: "Phase",
                    params: [normRadius("phase.norm_radius")],
                    coefficients: CoefficientList(
                        key: "phase.coefficients", title: "Coefficients", term: { k, _ in "A\(sub(k + 1))" },
                        meaning: { k, _ in "ρ" + superscript(2 * (k + 1)) }, unit: { _ in "rad" },
                        path: shapeCoefficient(.phase), maxCount: limit(.phase)),
                    footer: "A focusing phase has negative A₁. Coefficients describe order +1 and stay fixed when the wavelength or order changes."),
                ParamSection(
                    title: "Orders",
                    params: [
                        ShapeParam(
                            key: "phase.design_order", title: "Design order", unit: "", display: .integer,
                            help: "The order the structure is designed for (nonzero). Metadata: it does not rescale the phase."),
                        ShapeParam(
                            key: "phase.design_wavelength_um", title: "Design λ", unit: "µm",
                            help: "Design wavelength λ₀. Metadata: it does not rescale the phase."),
                        ShapeParam(
                            key: "phase.trace_order", title: "Trace order", unit: "", display: .integer,
                            help: "The order rays follow; 0 is ordinary refraction or reflection."),
                    ],
                    footer: "Orders are discrete and never optimization variables."),
            ],
            preview: .diffractive, orderEfficiency: true),
        SurfaceTypeDescriptor(
            id: "paraxial", kind: .paraxial, title: "Ideal Lens", symbol: "lines.measurement.horizontal", group: .special,
            summary: "A thin, aberration-free lens of the given focal length.",
            sections: [
                ParamSection(
                    title: "Ideal Lens",
                    params: [
                        ShapeParam(
                            key: "focal_length", title: "Focal length", unit: "mm", help: "Focal length of the ideal lens.",
                            path: { .paraxialFocalLength(surface: $0) })
                    ])
            ],
            preview: .idealLens),
        SurfaceTypeDescriptor(
            id: "coordinate_break", kind: .coordinateBreak, title: "Coordinate Break", symbol: "move.3d", group: .special,
            summary: "Changes the coordinate frame only: decenters, then tilts (or the reverse).", sections: [],
            placementFirst: true, preview: .coordinateBreak),
    ]

    static func conicAt(_ key: String) -> ShapeParam {
        var c = conic
        c.key = key
        return c
    }

    public static func descriptor(for shape: SurfaceShape) -> SurfaceTypeDescriptor { descriptor(for: shape.kind) }

    public static func descriptor(for kind: SurfaceShape.Kind) -> SurfaceTypeDescriptor {
        all.first { $0.kind == kind } ?? all[0]
    }

    /// The type menu: groups in order, each with its types and presets.
    public static var menu: [(group: SurfaceTypeGroup, choices: [SurfaceTypeChoice])] {
        SurfaceTypeGroup.allCases.map { g in
            var choices = all.filter { $0.group == g }.map { SurfaceTypeChoice.kind($0.kind) }
            if g == .anamorphic { choices += [.cylinder(.x), .cylinder(.y)] }
            return (g, choices)
        }
    }

    public static func title(of choice: SurfaceTypeChoice) -> String {
        switch choice {
        case .kind(let k): descriptor(for: k).title
        case .cylinder(.x): "Cylinder (power in x)"
        case .cylinder(.y): "Cylinder (power in y)"
        }
    }

    public static func symbol(of choice: SurfaceTypeChoice) -> String {
        switch choice {
        case .kind(let k): descriptor(for: k).symbol
        case .cylinder: "cylinder"
        }
    }

    /// The menu entry a shape is: a cylinder shows as its preset.
    public static func choice(for shape: SurfaceShape) -> SurfaceTypeChoice {
        shape.cylinderAxis.map(SurfaceTypeChoice.cylinder) ?? .kind(shape.kind)
    }

    // MARK: JSON access

    /// The shape in its model JSON form.
    public static func json(_ shape: SurfaceShape) -> JSONValue { (try? JSONValue.from(shape)) ?? .null }

    /// The JSON value at a dotted key path (`phase.norm_radius`).
    public static func value(_ shape: SurfaceShape, at key: String) -> JSONValue? {
        key.split(separator: ".").reduce(Optional(json(shape))) { j, k in j?[String(k)] }
    }

    /// The shape with the value at a dotted key path replaced, or `nil` when
    /// the model does not accept the result (unknown key, wrong type).
    public static func shape(_ shape: SurfaceShape, setting key: String, to value: JSONValue) -> SurfaceShape? {
        func set(_ j: JSONValue, _ path: ArraySlice<Substring>) -> JSONValue? {
            guard case .object(var o) = j, let first = path.first else { return nil }
            if path.count == 1 {
                o[String(first)] = value
            } else {
                guard let inner = o[String(first)], let changed = set(inner, path.dropFirst()) else { return nil }
                o[String(first)] = changed
            }
            return .object(o)
        }
        guard let changed = set(json(shape), key.split(separator: ".")[...]) else { return nil }
        return try? changed.decode(SurfaceShape.self)
    }

    /// Display value of a parameter (a radius for a curvature).
    public static func displayValue(_ p: ShapeParam, in shape: SurfaceShape) -> Double? {
        guard let v = value(shape, at: p.key)?.nf else { return nil }
        return p.display == .radiusOfCurvature ? RadiusDisplay.radius(fromCurvature: v) : v
    }

    /// Parses entered text for a parameter into its stored JSON value.
    public static func storedValue(_ p: ShapeParam, text: String) -> Result<JSONValue, LensValueParser.Failure> {
        switch p.display {
        case .radiusOfCurvature: LensValueParser.curvature(fromRadiusText: text).map { .number($0) }
        case .value: LensValueParser.finite(text).map { .number($0) }
        case .integer:
            LensValueParser.finite(text).flatMap { v in
                v.rounded() == v && abs(v) <= Double(Int32.max)
                    ? .success(.integer(Int64(v))) : .failure(.mustBeWhole)
            }
        }
    }

    /// Coefficients of a list on a shape.
    public static func coefficients(_ list: CoefficientList, in shape: SurfaceShape) -> [Double] {
        value(shape, at: list.key)?.nfArray ?? []
    }

    // MARK: Previews

    /// The half-width a preview covers: the clear semi-diameter, else the
    /// normalisation radius, else 10 mm.
    public static func previewRadius(_ shape: SurfaceShape, semiDiameter: Double?) -> Double {
        if let h = semiDiameter, h > 0, h.isFinite { return h }
        if let r = shape.normRadius, r > 0 { return r }
        if case .diffractive(_, let p) = shape, p.normRadius > 0 { return p.normRadius }
        return 10
    }

    /// Sag profile points `(t, z)` from −h to h along x or y, or `nil`.
    public static func profile(
        _ shape: SurfaceShape, semiDiameter h: Double, axis: SurfaceShape.CylinderAxis = .y, samples: Int = 64
    ) -> [(Double, Double)]? {
        guard h > 0, h.isFinite else { return nil }
        var out: [(Double, Double)] = []
        for k in 0...samples {
            let t = -h + 2 * h * Double(k) / Double(samples)
            let z = axis == .y ? ShapeSag.sag(shape, x: 0, y: t) : ShapeSag.sag(shape, x: t, y: 0)
            guard let z, z.isFinite else { continue }
            out.append((t, z))
        }
        return out.count > 2 ? out : nil
    }

    /// A `n × n` grid of the departure from the base conic over the disk of
    /// radius `h` (row-major, y up from the last row); `nil` outside.
    public static func departureMap(_ shape: SurfaceShape, semiDiameter h: Double, n: Int = 40) -> [Double?] {
        (0..<(n * n)).map { k in
            let i = k % n, j = k / n
            let x = -h + 2 * h * (Double(i) + 0.5) / Double(n)
            let y = h - 2 * h * (Double(j) + 0.5) / Double(n)
            guard x * x + y * y <= h * h else { return nil }
            return ShapeSag.departure(shape, x: x, y: y)
        }
    }
}

extension SurfaceShape {
    /// Toroids and biconics: different x and y sections.
    public var isAnamorphic: Bool { kind == .toroid || kind == .biconic }
}
