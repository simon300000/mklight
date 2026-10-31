import Foundation
import MKLightEngine
import MKLightModel

// The tolerance set of a design (docs/OPTIMIZATION.md, `tolerance`):
// rows in the units a lens designer thinks in (fringes of power and
// irregularity, ΔR, arc minutes, Δn, % of V_d), expanded to the engine's
// `ToleranceOperand`s on parameter paths; compensators (`Compensator`);
// the criterion; and the run's mode, trials, seed, percentiles and
// recorded trials. Saved in the project's `tolerances` section.

// MARK: - Kinds and units

/// What a tolerance row perturbs.
public enum ToleranceKind: String, CaseIterable, Sendable, Identifiable {
    /// Surface power: the curvature, in fringes, ΔR or % of R.
    case radius
    /// Surface irregularity in test fringes: astigmatic (x² − y²) or
    /// spherical (the engine's sag overlay).
    case irregularityAstigmatism = "irregularity_astigmatism"
    case irregularitySpherical = "irregularity_spherical"
    /// The axial gap after a surface (glass or air).
    case thickness
    /// Surface decentre in x / y, the rest of the system staying in place.
    case decenterX = "decenter_x", decenterY = "decenter_y"
    /// Surface tilt about x / y (a wedge), the rest staying in place.
    case tiltX = "tilt_x", tiltY = "tilt_y"
    /// A whole element (a surface range, a cemented group together) moved
    /// rigidly about a pivot vertex; the rest stays in place.
    case elementDecenterX = "element_decenter_x", elementDecenterY = "element_decenter_y"
    case elementTiltX = "element_tilt_x", elementTiltY = "element_tilt_y"
    /// The refractive index n_d of the glass after a surface: a model
    /// glass's n_d, a catalogue glass's Δn_d (added to its melt offset).
    case index
    /// The Abbe number V_d of the glass after a surface (model n_d / V_d,
    /// catalogue ΔV_d).
    case abbe
    /// Any other parameter, in its own unit.
    case other

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .radius: "Radius"
        case .irregularityAstigmatism: "Astig. irregularity"
        case .irregularitySpherical: "Sph. irregularity"
        case .thickness: "Thickness"
        case .decenterX: "Surface decentre x"
        case .decenterY: "Surface decentre y"
        case .tiltX: "Surface tilt x"
        case .tiltY: "Surface tilt y"
        case .elementDecenterX: "Element decentre x"
        case .elementDecenterY: "Element decentre y"
        case .elementTiltX: "Element tilt x"
        case .elementTiltY: "Element tilt y"
        case .index: "Index"
        case .abbe: "Abbe"
        case .other: "Parameter"
        }
    }

    public var symbol: String {
        switch self {
        case .radius: "circle.lefthalf.filled"
        case .irregularityAstigmatism, .irregularitySpherical: "water.waves"
        case .thickness: "arrow.left.and.right"
        case .decenterX, .decenterY: "arrow.up.and.down.and.arrow.left.and.right"
        case .tiltX, .tiltY: "rotate.3d"
        case .elementDecenterX, .elementDecenterY: "square.stack.3d.down.right"
        case .elementTiltX, .elementTiltY: "rotate.3d.fill"
        case .index: "n.circle"
        case .abbe: "rainbow"
        case .other: "slider.horizontal.3"
        }
    }

    /// An element's rigid motion (a surface range about a pivot).
    public var isElement: Bool { component != nil }

    /// The rigid-motion component of an element kind.
    public var component: PlacementComponent? {
        switch self {
        case .elementDecenterX: .decenterX
        case .elementDecenterY: .decenterY
        case .elementTiltX: .tiltX
        case .elementTiltY: .tiltY
        default: nil
        }
    }

    /// The element kind of a rigid-motion component (tilt z has none).
    public static func element(_ c: PlacementComponent) -> ToleranceKind? {
        switch c {
        case .decenterX: .elementDecenterX
        case .decenterY: .elementDecenterY
        case .tiltX: .elementTiltX
        case .tiltY: .elementTiltY
        case .tiltZ: nil
        }
    }

    /// The units a row of this kind can be edited in, the first the default.
    public var units: [ToleranceUnit] {
        switch self {
        case .radius: [.fringes, .radiusMM, .percent]
        case .irregularityAstigmatism, .irregularitySpherical: [.fringes]
        case .thickness, .decenterX, .decenterY, .elementDecenterX, .elementDecenterY: [.mm]
        case .tiltX, .tiltY, .elementTiltX, .elementTiltY: [.arcmin, .degrees]
        case .index: [.index]
        case .abbe: [.percent, .abbe]
        case .other: [.native]
        }
    }

    /// Kinds offered when adding a row on a surface (other parameters come
    /// from a list; element kinds from the elements).
    static let addable: [ToleranceKind] = [
        .radius, .irregularityAstigmatism, .irregularitySpherical, .thickness, .decenterX, .decenterY, .tiltX, .tiltY, .index, .abbe,
    ]
    /// Kinds offered when adding a row on an element.
    static let addableElement: [ToleranceKind] = [.elementDecenterX, .elementDecenterY, .elementTiltX, .elementTiltY]
}

/// The unit a row's limits are written in.
public enum ToleranceUnit: String, Sendable, CaseIterable {
    /// Test fringes at the test wavelength: of power over the clear
    /// aperture, or of irregularity.
    case fringes
    /// ΔR in mm.
    case radiusMM = "radius_mm"
    /// Percent (of R, or of V_d).
    case percent
    case mm
    case arcmin
    case degrees
    /// Δn (absolute).
    case index
    /// ΔV_d (absolute).
    case abbe
    /// The parameter's own unit.
    case native

    public var label: String {
        switch self {
        case .fringes: "fr"
        case .radiusMM: "mm"
        case .percent: "%"
        case .mm: "mm"
        case .arcmin: "′"
        case .degrees: "°"
        case .index: "Δn"
        case .abbe: "ΔV"
        case .native: ""
        }
    }

    public var title: String {
        switch self {
        case .fringes: "Test fringes"
        case .radiusMM: "ΔR (mm)"
        case .percent: "Percent"
        case .mm: "mm"
        case .arcmin: "Arc minutes"
        case .degrees: "Degrees"
        case .index: "Δn"
        case .abbe: "ΔV_d"
        case .native: "Parameter units"
        }
    }
}

/// How a trial draws a perturbation within [−minus, +plus] (the engine's
/// `Distribution`).
public enum ToleranceDistribution: Hashable, Sendable {
    case uniform
    /// Normal, truncated at the limits, which lie at ±`sigmas` σ.
    case normal(sigmas: Double)
    /// Exactly −minus or +plus (worst-case style).
    case endPoint

    public var json: JSONValue {
        switch self {
        case .uniform: .object(["type": .string("uniform")])
        case .normal(let k): .object(["type": .string("truncated_normal"), "sigmas": .number(k)])
        case .endPoint: .object(["type": .string("end_point")])
        }
    }

    public init(json: JSONValue?) {
        switch json?["type"]?.string {
        case "truncated_normal": self = .normal(sigmas: json?["sigmas"]?.nf ?? 2)
        case "end_point": self = .endPoint
        default: self = .uniform
        }
    }

    public var title: String {
        switch self {
        case .uniform: "Uniform"
        case .normal(let k): "Normal (limits at ±\(Fmt.short(k)) σ)"
        case .endPoint: "End points"
        }
    }

    static let choices: [ToleranceDistribution] = [.uniform, .normal(sigmas: 2), .normal(sigmas: 3), .endPoint]
}

// MARK: - Rows

/// What a row needs to become an engine operand: the system, the clear
/// semi-diameters (fringes of power are counted over them), the test
/// wavelength and the Abbe numbers of catalogue glasses (for % of V_d).
public struct ToleranceContext: Sendable {
    public var system: OpticalSystem
    /// Clear semi-diameters (mm), fixed or as the engine's layout resolved them.
    public var semiDiameters: [SurfaceID: Double]
    /// Interferometer wavelength for fringes (µm): He–Ne by default.
    public var testWavelengthUm = ToleranceSettings.defaultTestWavelengthUm
    /// V_d of the catalogue glass after a surface, its offset included
    /// (the engine's `glass_info`).
    public var abbeNumbers: [SurfaceID: Double]

    public init(
        system: OpticalSystem, semiDiameters: [SurfaceID: Double] = [:], testWavelengthUm: Double = ToleranceSettings.defaultTestWavelengthUm,
        abbeNumbers: [SurfaceID: Double] = [:]
    ) {
        self.system = system
        self.semiDiameters = semiDiameters
        self.testWavelengthUm = testWavelengthUm
        self.abbeNumbers = abbeNumbers
    }

    /// The clear semi-diameter of a surface: its fixed value, else the
    /// layout's.
    public func semiDiameter(_ id: SurfaceID) -> Double? {
        if let i = system.surfaceIndex(of: id), let v = system.surfaces[i].semiDiameter.fixedValue { return v }
        return semiDiameters[id]
    }

    /// V_d of the glass after a surface: a model glass's own, a catalogue
    /// glass's from the engine.
    public func abbeNumber(_ id: SurfaceID) -> Double? {
        guard let i = system.surfaceIndex(of: id) else { return nil }
        if case .model(_, let vd, _) = system.surfaces[i].material { return vd }
        return abbeNumbers[id]
    }
}

/// One tolerance: a kind on a surface (or an element's surface range, or
/// any parameter), its − and + limits in the row's unit, and the
/// distribution of Monte-Carlo trials.
public struct ToleranceRow: Identifiable, Hashable, Sendable {
    /// Session identity (not saved).
    public var id = UUID()
    public var kind: ToleranceKind
    /// The surface, or an element's first surface.
    public var surface: SurfaceID
    /// An element's last surface (inclusive; `nil`: `surface` alone).
    public var last: SurfaceID?
    /// The vertex an element tilts about (`nil`: its first surface).
    public var pivot: SurfaceID?
    /// The parameter of an `.other` row.
    public var param: ParamPath?
    /// Magnitudes of the limits (≥ 0), in `unit`.
    public var minus: Double
    public var plus: Double
    public var unit: ToleranceUnit
    public var distribution = ToleranceDistribution.uniform

    public init(
        _ kind: ToleranceKind, surface: SurfaceID, minus: Double, plus: Double? = nil, unit: ToleranceUnit? = nil, param: ParamPath? = nil,
        last: SurfaceID? = nil, pivot: SurfaceID? = nil, distribution: ToleranceDistribution = .uniform
    ) {
        self.kind = kind
        self.surface = surface
        self.minus = minus
        self.plus = plus ?? minus
        self.unit = unit ?? kind.units[0]
        self.param = param
        self.last = kind.isElement ? last : nil
        self.pivot = kind.isElement ? pivot : nil
        self.distribution = distribution
    }

    /// Whether `m` is a glass (a medium with n_d and V_d).
    static func hasGlass(_ m: MaterialRef) -> Bool {
        switch m {
        case .model, .catalog: true
        default: false
        }
    }

    /// The engine parameter the row perturbs. Index and Abbe address a
    /// model glass's n_d and V_d, a catalogue glass's offset (added to the
    /// saved melt offset).
    public func path(in system: OpticalSystem) -> ParamPath {
        var catalogue: Bool {
            guard let i = system.surfaceIndex(of: surface), case .catalog = system.surfaces[i].material else { return false }
            return true
        }
        switch kind {
        case .radius: return .curvature(surface: surface)
        case .thickness: return .thickness(surface: surface)
        case .decenterX: return .decenterX(surface: surface)
        case .decenterY: return .decenterY(surface: surface)
        case .tiltX: return .tiltX(surface: surface)
        case .tiltY: return .tiltY(surface: surface)
        case .elementDecenterX, .elementDecenterY, .elementTiltX, .elementTiltY:
            return .elementPlacement(surface: surface, last: last ?? surface, pivot: pivot, component: kind.component ?? .decenterX)
        case .irregularityAstigmatism: return .irregularity(surface: surface, mode: .astigmatism)
        case .irregularitySpherical: return .irregularity(surface: surface, mode: .spherical)
        case .index: return catalogue ? .materialDnD(surface: surface) : .modelNd(surface: surface)
        case .abbe: return catalogue ? .materialDvD(surface: surface) : .modelVd(surface: surface)
        case .other: return param ?? .curvature(surface: surface)
        }
    }

    /// Surface indices that move together (an element's range, inclusive).
    public func movingSurfaces(in system: OpticalSystem) -> [Int] {
        guard let i = system.surfaceIndex(of: surface) else { return [] }
        guard kind.isElement, let l = last.flatMap(system.surfaceIndex(of:)), l >= i else { return [i] }
        return Array(i...l)
    }

    /// The pivot's surface index (the first surface by default).
    public func pivotIndex(in system: OpticalSystem) -> Int? {
        system.surfaceIndex(of: pivot ?? surface)
    }

    /// `S3`, `S3–4` (a thickness between two surfaces, or an element).
    public func surfaceLabel(in system: OpticalSystem) -> String {
        guard let i = system.surfaceIndex(of: surface) else { return "S?" }
        if kind == .thickness { return "S\(i)–\(i + 1)" }
        if kind.isElement {
            let l = last.flatMap(system.surfaceIndex(of:)) ?? i
            return l == i ? "S\(i)" : "S\(i)–\(l)"
        }
        return "S\(i)"
    }

    /// The element (`L2`) whose surfaces this row moves, if it is one.
    public func elementTitle(in system: OpticalSystem) -> String? {
        let moving = movingSurfaces(in: system)
        guard kind.isElement, let f = moving.first, let l = moving.last else { return nil }
        return LensElements.groups(system).first { $0.surfaces.first == f && $0.surfaces.last == l }?.title
    }

    /// The engine's limits `(minus, plus)` in parameter units, or why the
    /// row cannot be expanded.
    public func engineLimits(in c: ToleranceContext) -> Result<(Double, Double), ToleranceIssue> {
        guard minus.isFinite, plus.isFinite, minus >= 0, plus >= 0 else { return .failure(.message("Limits must be zero or positive numbers.")) }
        guard minus > 0 || plus > 0 else { return .failure(.message("Both limits are zero.")) }
        let s = c.system
        guard let i = s.surfaceIndex(of: surface) else { return .failure(.message("The surface no longer exists.")) }
        let sf = s.surfaces[i]
        switch kind {
        case .radius:
            guard let curv = s.value(at: .curvature(surface: surface)) else {
                return .failure(.message("Surface \(i) has no radius."))
            }
            switch unit {
            case .fringes:
                guard let h = c.semiDiameter(surface), h > 0 else {
                    return .failure(.message("Fringes need the clear aperture of surface \(i); run the layout or set a semi-diameter."))
                }
                // N fringes of power: sag change N·λ/2 at the rim, Δc = N·λ/h².
                let k = c.testWavelengthUm * 1e-3 / (h * h)
                return .success((minus * k, plus * k))
            case .radiusMM, .percent:
                guard curv != 0 else { return .failure(.message("Surface \(i) is flat: give its power in fringes.")) }
                let r = 1 / curv
                let dm = unit == .percent ? abs(r) * minus / 100 : minus, dp = unit == .percent ? abs(r) * plus / 100 : plus
                // R in [R − ΔR−, R + ΔR+]: c in [1/(R + ΔR+), 1/(R − ΔR−)].
                let lo = r + dp, hi = r - dm
                guard lo != 0, hi != 0, (lo > 0) == (r > 0), (hi > 0) == (r > 0) else {
                    return .failure(.message("ΔR must stay below |R| = \(Fmt.short(abs(r))) mm."))
                }
                return .success((curv - 1 / lo, 1 / hi - curv))
            default:
                return .failure(.message("Radius tolerances are in fringes, mm or %."))
            }
        case .thickness:
            guard i < s.imageIndex else { return .failure(.message("The image surface has no thickness.")) }
            return .success((minus, plus))
        case .decenterX, .decenterY:
            guard i > 0, i < s.imageIndex else { return .failure(.message("Decentre a surface between the object and the image.")) }
            return .success((minus, plus))
        case .tiltX, .tiltY, .elementTiltX, .elementTiltY, .elementDecenterX, .elementDecenterY:
            if kind.isElement, let why = elementProblem(in: s, first: i) { return .failure(.message(why)) }
            if !kind.isElement, i == 0 || i >= s.imageIndex { return .failure(.message("Tilt a surface between the object and the image.")) }
            let k = unit == .arcmin ? 1.0 / 60 : 1
            return .success((minus * k, plus * k))
        case .irregularityAstigmatism, .irregularitySpherical:
            guard i > 0, i < s.imageIndex, !sf.isCoordinateBreak, sf.shape.kind != .paraxial else {
                return .failure(.message("Surface \(i) has no form to make irregular."))
            }
            return .success((minus, plus))
        case .index, .abbe:
            guard Self.hasGlass(sf.material) else {
                let name = sf.material.isAir ? "air" : sf.material.displayName
                return .failure(.message("Surface \(i) is followed by \(name), not a glass with n_d and V_d."))
            }
            var lim = (minus, plus)
            if kind == .abbe && unit == .percent {
                guard let vd = c.abbeNumber(surface), vd > 0 else {
                    return .failure(.message("The Abbe number of \(sf.material.displayName) is not known yet: give the limit as ΔV_d, or wait for the glass data."))
                }
                lim = (vd * minus / 100, vd * plus / 100)
            }
            if case .catalog(_, _, let o) = sf.material {
                // A tolerance adds to the saved melt offset, which stays within the engine's bounds.
                let (base, cap, what) = kind == .index ? (o.dnD, MaterialOffset.maxDnD, "Δn_d") : (o.dvD, MaterialOffset.maxDvD, "ΔV_d")
                if abs(base - lim.0) > cap + 1e-12 || abs(base + lim.1) > cap + 1e-12 {
                    return .failure(.message("With the melt offset (\(Fmt.short(base))), \(what) would pass ±\(Fmt.short(cap)), the engine's limit."))
                }
            }
            return .success(lim)
        case .other:
            guard let p = param, p.resolves(in: s), s.value(at: p) != nil else {
                return .failure(.message("The parameter no longer exists."))
            }
            return .success((minus, plus))
        }
    }

    /// Why an element row's range or pivot cannot be moved, if it cannot.
    private func elementProblem(in s: OpticalSystem, first i: Int) -> String? {
        guard let l = s.surfaceIndex(of: last ?? surface) else { return "The element's last surface no longer exists." }
        guard i > 0, l >= i, l < s.imageIndex else { return "An element moves surfaces between the object and the image, first to last." }
        if let p = pivot, s.surfaceIndex(of: p) == nil { return "The pivot surface no longer exists." }
        if let e = s.surfaces[i].elementPlacement, e.last != (last ?? surface) || e.pivot != pivot {
            let el = s.surfaceIndex(of: e.last).map(String.init) ?? "?"
            let ep = s.surfaceIndex(of: e.pivot ?? surface).map(String.init) ?? "?"
            return "The design already moves S\(i)–\(el) as an element about S\(ep): use that range and pivot."
        }
        return nil
    }

    /// The engine's `ToleranceOperand`.
    public func operand(in c: ToleranceContext) -> Result<JSONValue, ToleranceIssue> {
        engineLimits(in: c).map { m, p in
            .object([
                "param": (try? JSONValue.from(path(in: c.system))) ?? .null, "minus": .number(m), "plus": .number(p),
                "distribution": distribution.json, "label": .string(label(in: c.system)),
            ])
        }
    }

    /// `Radius S3`, `Thickness S3–4`, `Element tilt x L2`, `Index S5`.
    public func label(in system: OpticalSystem) -> String {
        if kind == .other { return path(in: system).label(in: system) }
        if kind.isElement, let e = elementTitle(in: system) { return "\(kind.title) \(e)" }
        return "\(kind.title) \(surfaceLabel(in: system))"
    }

    /// The limits as text: `±3 fr`, `−0.02 / +0.05 mm`.
    public var limitsText: String {
        let u = unit.label.isEmpty ? "" : (unit == .arcmin ? unit.label : " " + unit.label)
        if minus == plus { return "±" + Fmt.short(minus) + u }
        return "−" + Fmt.short(minus) + " / +" + Fmt.short(plus) + u
    }

    /// One limit with its unit: `+0.8′`, `−0.02 mm`.
    public func limitText(plus: Bool) -> String {
        let u = unit.label.isEmpty ? "" : (unit == .arcmin ? unit.label : " " + unit.label)
        return (plus ? "+" : "−") + Fmt.short(plus ? self.plus : minus) + u
    }

    /// A perturbation in parameter units written in the row's unit, when it
    /// is a plain scale of it (mm, degrees ↔ arc minutes, fringes, Δn,
    /// ΔV_d); `nil` for radii and % (they depend on the design).
    public func displayValue(ofParameterDelta d: Double) -> Double? {
        switch unit {
        case .arcmin: d * 60
        case .degrees, .mm, .fringes, .index, .abbe, .native: d
        case .radiusMM, .percent: nil
        }
    }

    // MARK: Saved form

    public var json: JSONValue {
        var o: [String: JSONValue] = [
            "kind": .string(kind.rawValue), "surface": .integer(Int64(surface.rawValue)), "minus": .number(minus), "plus": .number(plus),
            "unit": .string(unit.rawValue), "distribution": distribution.json,
        ]
        if let param { o["param"] = (try? JSONValue.from(param)) ?? .null }
        if let last { o["last"] = .integer(Int64(last.rawValue)) }
        if let pivot { o["pivot"] = .integer(Int64(pivot.rawValue)) }
        return .object(o)
    }

    public init?(json: JSONValue) {
        guard let k = json["kind"]?.string.flatMap(ToleranceKind.init(rawValue:)), let s = json["surface"]?.int, s >= 0 else { return nil }
        kind = k
        surface = SurfaceID(UInt64(s))
        minus = json["minus"]?.nf ?? 0
        plus = json["plus"]?.nf ?? 0
        unit = json["unit"]?.string.flatMap(ToleranceUnit.init(rawValue:)).flatMap { k.units.contains($0) ? $0 : nil } ?? k.units[0]
        distribution = ToleranceDistribution(json: json["distribution"])
        param = json["param"].flatMap { try? MKLightJSON.decode(ParamPath.self, from: MKLightJSON.encode($0)) }
        if k.isElement {
            last = json["last"]?.int.flatMap { $0 >= 0 ? SurfaceID(UInt64($0)) : nil }
            pivot = json["pivot"]?.int.flatMap { $0 >= 0 ? SurfaceID(UInt64($0)) : nil }
        }
        if k == .other && param == nil { return nil }
    }

    /// A row from an engine operand written elsewhere (Python, CLI): kinds
    /// the app knows in their natural unit (curvature as ΔR), others as
    /// parameters in their own unit.
    public init?(operand o: JSONValue, system: OpticalSystem) {
        guard let p = o["param"].flatMap({ try? MKLightJSON.decode(ParamPath.self, from: MKLightJSON.encode($0)) }), let id = p.surface else { return nil }
        let m = o["minus"]?.nf ?? 0, pl = o["plus"]?.nf ?? 0
        let d = ToleranceDistribution(json: o["distribution"])
        switch p {
        case .curvature:
            guard let c = system.value(at: p), c != 0 else { self.init(.other, surface: id, minus: m, plus: pl, param: p, distribution: d); return }
            // c in [c − m, c + p] → R in [1/(c + p), 1/(c − m)].
            let r = 1 / c
            let dm = r - 1 / (c + pl), dp = 1 / (c - m) - r
            self.init(.radius, surface: id, minus: abs(dm), plus: abs(dp), unit: .radiusMM, distribution: d)
        case .thickness: self.init(.thickness, surface: id, minus: m, plus: pl, distribution: d)
        case .decenterX: self.init(.decenterX, surface: id, minus: m, plus: pl, distribution: d)
        case .decenterY: self.init(.decenterY, surface: id, minus: m, plus: pl, distribution: d)
        case .tiltX: self.init(.tiltX, surface: id, minus: m * 60, plus: pl * 60, distribution: d)
        case .tiltY: self.init(.tiltY, surface: id, minus: m * 60, plus: pl * 60, distribution: d)
        case .modelNd, .materialDnD: self.init(.index, surface: id, minus: m, plus: pl, distribution: d)
        case .modelVd, .materialDvD: self.init(.abbe, surface: id, minus: m, plus: pl, unit: .abbe, distribution: d)
        case .irregularity(_, let mode):
            self.init(mode == .astigmatism ? .irregularityAstigmatism : .irregularitySpherical, surface: id, minus: m, plus: pl, distribution: d)
        case .elementPlacement(_, let last, let pivot, let component):
            guard let k = ToleranceKind.element(component) else {
                self.init(.other, surface: id, minus: m, plus: pl, param: p, distribution: d)
                return
            }
            let tilt = k == .elementTiltX || k == .elementTiltY
            self.init(k, surface: id, minus: tilt ? m * 60 : m, plus: tilt ? pl * 60 : pl, last: last, pivot: pivot, distribution: d)
        default: self.init(.other, surface: id, minus: m, plus: pl, param: p, distribution: d)
        }
    }

    /// Hashable ignores the session id.
    public static func == (a: ToleranceRow, b: ToleranceRow) -> Bool { a.json == b.json }
    public func hash(into h: inout Hasher) { h.combine(json) }
}

/// Why a row cannot run.
public enum ToleranceIssue: Error, Hashable, Sendable {
    case message(String)

    public var text: String {
        switch self {
        case .message(let s): s
        }
    }
}

// MARK: - Default tolerances

/// The "Default tolerances" generator: a preset class of manufacture and
/// its values per kind, expanded over the design's elements. A value of 0
/// leaves its kind out.
public struct ToleranceGenerator: Hashable, Sendable {
    public enum Preset: String, CaseIterable, Sendable, Identifiable {
        case commercial, precision, highPrecision = "high_precision"
        public var id: String { rawValue }
        public var title: String {
            switch self {
            case .commercial: "Commercial"
            case .precision: "Precision"
            case .highPrecision: "High precision"
            }
        }
    }

    /// The preset the values came from (`nil`: edited).
    public var preset: Preset? = .precision
    /// Power, fringes at the test wavelength.
    public var radiusFringes = 3.0
    public var thicknessMM = 0.05
    /// Surface decentre (one surface, the rest staying in place).
    public var decentreMM = 0.025
    /// Surface tilt, a wedge.
    public var tiltArcmin = 1.0
    /// Element decentre: the whole element (a cemented group together).
    public var elementDecentreMM = 0.03
    /// Element tilt about its first vertex.
    public var elementTiltArcmin = 1.0
    public var index = 0.0005
    public var abbePercent = 0.5
    /// Surface irregularity, test fringes: astigmatic and spherical, each.
    public var irregularityFringes = 0.5
    public var distribution = ToleranceDistribution.uniform

    public init() {}

    public init(preset p: Preset) {
        apply(p)
    }

    /// The values of a preset (typical shop classes; one surface's figures).
    public mutating func apply(_ p: Preset) {
        preset = p
        switch p {
        case .commercial:
            (radiusFringes, thicknessMM, decentreMM, tiltArcmin, index, abbePercent, irregularityFringes) = (5, 0.2, 0.1, 3, 0.001, 1, 2)
            (elementDecentreMM, elementTiltArcmin) = (0.1, 3)
        case .precision:
            (radiusFringes, thicknessMM, decentreMM, tiltArcmin, index, abbePercent, irregularityFringes) = (3, 0.05, 0.025, 1, 0.0005, 0.5, 0.5)
            (elementDecentreMM, elementTiltArcmin) = (0.03, 1)
        case .highPrecision:
            (radiusFringes, thicknessMM, decentreMM, tiltArcmin, index, abbePercent, irregularityFringes) = (1, 0.025, 0.01, 0.5, 0.0002, 0.2, 0.25)
            (elementDecentreMM, elementTiltArcmin) = (0.015, 0.5)
        }
    }

    /// The tolerances of `system`: power and irregularity of every
    /// refracting lens surface, every glass thickness and air space between
    /// them (not the back focus), decentre and tilt of every lens surface
    /// and of every element (a cemented group moving as one) in x and y,
    /// and the index and Abbe number of every glass.
    public func rows(for system: OpticalSystem) -> [ToleranceRow] {
        var out: [ToleranceRow] = []
        let lens = LensElements.lensSurfaces(system)
        guard let first = lens.first, let last = lens.last else { return [] }
        let elements = LensElements.groups(system)
        func add(_ k: ToleranceKind, _ id: SurfaceID, _ v: Double, last: SurfaceID? = nil) {
            guard v > 0 else { return }
            out.append(ToleranceRow(k, surface: id, minus: v, last: last, distribution: distribution))
        }
        for i in first...last {
            let s = system.surfaces[i]
            guard !s.isCoordinateBreak else { continue }
            let id = s.id
            if lens.contains(i) && system.value(at: .curvature(surface: id)) != nil {
                add(.radius, id, radiusFringes)
                if s.interaction != .dummy {
                    add(.irregularityAstigmatism, id, irregularityFringes)
                    add(.irregularitySpherical, id, irregularityFringes)
                }
            }
            // Every gap inside the lens, glass or air (the back focus is the compensator).
            if i < last { add(.thickness, id, thicknessMM) }
            if lens.contains(i) {
                for k in [ToleranceKind.decenterX, .decenterY] { add(k, id, decentreMM) }
                for k in [ToleranceKind.tiltX, .tiltY] { add(k, id, tiltArcmin) }
            }
            if i < last, ToleranceRow.hasGlass(system.surfaces[i].material) {
                add(.index, id, index)
                add(.abbe, id, abbePercent)
            }
            // The element's own motion after its last surface's rows.
            if let e = elements.first(where: { $0.surfaces.last == i }), let f = e.surfaces.first, f < i {
                let start = system.surfaces[f].id
                for k in [ToleranceKind.elementDecenterX, .elementDecenterY] { add(k, start, elementDecentreMM, last: id) }
                for k in [ToleranceKind.elementTiltX, .elementTiltY] { add(k, start, elementTiltArcmin, last: id) }
            }
        }
        return out
    }

    public var json: JSONValue {
        .object([
            "preset": preset.map { .string($0.rawValue) } ?? .null, "radius_fringes": .number(radiusFringes), "thickness_mm": .number(thicknessMM),
            "decentre_mm": .number(decentreMM), "tilt_arcmin": .number(tiltArcmin), "element_decentre_mm": .number(elementDecentreMM),
            "element_tilt_arcmin": .number(elementTiltArcmin), "index": .number(index), "abbe_percent": .number(abbePercent),
            "irregularity_fringes": .number(irregularityFringes), "distribution": distribution.json,
        ])
    }

    public init(json: JSONValue?) {
        guard let j = json, j.object != nil else { return }
        preset = j["preset"]?.string.flatMap(Preset.init(rawValue:))
        radiusFringes = j["radius_fringes"]?.nf ?? radiusFringes
        thicknessMM = j["thickness_mm"]?.nf ?? thicknessMM
        decentreMM = j["decentre_mm"]?.nf ?? decentreMM
        tiltArcmin = j["tilt_arcmin"]?.nf ?? tiltArcmin
        elementDecentreMM = j["element_decentre_mm"]?.nf ?? elementDecentreMM
        elementTiltArcmin = j["element_tilt_arcmin"]?.nf ?? elementTiltArcmin
        index = j["index"]?.nf ?? index
        abbePercent = j["abbe_percent"]?.nf ?? abbePercent
        irregularityFringes = j["irregularity_fringes"]?.nf ?? irregularityFringes
        distribution = ToleranceDistribution(json: j["distribution"])
    }
}

// MARK: - Elements

/// The design's lens elements, for grouping tolerances: runs of surfaces
/// joined by glass (a cemented group is one element), numbered L1, L2, …
public enum LensElements {
    public struct Group: Identifiable, Hashable, Sendable {
        /// `L1`, or `S7` for a surface outside any element.
        public var title: String
        /// Surface indices of the group, in order.
        public var surfaces: [Int]
        /// Glass names inside the element.
        public var glasses: [String]
        public var id: String { title }
    }

    /// Whether the medium after surface `i` is a solid (not air or vacuum).
    static func isSolid(_ s: OpticalSystem, _ i: Int) -> Bool {
        let m = s.surfaces[i].material
        return !m.isAir && m != .vacuum
    }

    /// Surface indices that bound or lie inside a lens element (refracting,
    /// with glass on at least one side).
    public static func lensSurfaces(_ s: OpticalSystem) -> [Int] {
        guard s.imageIndex > 1 else { return [] }
        return (1..<s.imageIndex).filter { i in
            !s.surfaces[i].isCoordinateBreak && (isSolid(s, i) || isSolid(s, i - 1))
        }
    }

    /// The elements, then any other surface a tolerance names.
    public static func groups(_ s: OpticalSystem) -> [Group] {
        var out: [Group] = []
        var current: [Int] = []
        for i in lensSurfaces(s) {
            current.append(i)
            if !isSolid(s, i) {
                let glasses = current.dropLast().map { s.surfaces[$0].material.displayName }
                out.append(Group(title: "L\(out.count + 1)", surfaces: current, glasses: glasses))
                current = []
            }
        }
        if !current.isEmpty {
            out.append(Group(title: "L\(out.count + 1)", surfaces: current, glasses: current.map { s.surfaces[$0].material.displayName }))
        }
        return out
    }

    /// The group a row belongs to: its element (an air space goes with the
    /// element before it), else the surface itself.
    public static func groupTitle(of row: ToleranceRow, in s: OpticalSystem, groups: [Group]) -> String {
        guard let i = s.surfaceIndex(of: row.surface) else { return "Other" }
        if let g = groups.first(where: { $0.surfaces.contains(i) }) {
            // The thickness after an element's last surface is the air space after it.
            if row.kind == .thickness, g.surfaces.last == i { return "Air after \(g.title)" }
            return g.title
        }
        return i == s.stop ? "Stop S\(i)" : "S\(i)"
    }
}

// MARK: - Compensators

/// A parameter re-optimised for every trial (the engine's `Compensator`),
/// within ±`range` of its nominal value (`nil`: unbounded).
public struct ToleranceCompensator: Identifiable, Hashable, Sendable {
    public var id = UUID()
    public var param: ParamPath
    public var range: Double?

    public init(_ param: ParamPath, range: Double? = nil) {
        self.param = param
        self.range = range
    }

    /// The back focus: the thickness before the image surface.
    public static func backFocus(_ s: OpticalSystem, range: Double = 0.5) -> ToleranceCompensator? {
        guard s.imageIndex >= 1 else { return nil }
        return ToleranceCompensator(.thickness(surface: s.surfaces[s.imageIndex - 1].id), range: range)
    }

    public static func isBackFocus(_ p: ParamPath, in s: OpticalSystem) -> Bool {
        guard case .thickness(let id) = p, s.imageIndex >= 1 else { return false }
        return s.surfaces[s.imageIndex - 1].id == id
    }

    /// "Back focus" or the parameter's label.
    public func title(in s: OpticalSystem) -> String {
        Self.isBackFocus(param, in: s) ? "Back focus" : param.label(in: s)
    }

    public var json: JSONValue {
        .object(["param": (try? JSONValue.from(param)) ?? .null, "range": range.map(JSONValue.number) ?? .null])
    }

    public init?(json: JSONValue) {
        guard let p = json["param"].flatMap({ try? MKLightJSON.decode(ParamPath.self, from: MKLightJSON.encode($0)) }) else { return nil }
        param = p
        range = json["range"]?.nf.flatMap { $0.isFinite ? $0 : nil }
    }

    public static func == (a: Self, b: Self) -> Bool { a.json == b.json }
    public func hash(into h: inout Hasher) { h.combine(json) }

    /// Parameters a compensator can take: the back focus, every air space,
    /// the design's variables and the decentres, tilts, thicknesses and
    /// curvatures of every surface, in that order, each once.
    public static func candidates(_ s: OpticalSystem, variables: [ParamPath]) -> [(group: String, params: [ParamPath])] {
        var seen = Set<ParamPath>()
        func fresh(_ ps: [ParamPath]) -> [ParamPath] { ps.filter { seen.insert(s.canonical($0)).inserted } }
        let bf = backFocus(s).map { [$0.param] } ?? []
        let air = (1..<max(s.imageIndex - 1, 1)).filter { s.surfaces[$0].material.isAir }.map { ParamPath.thickness(surface: s.surfaces[$0].id) }
        var out: [(String, [ParamPath])] = [("Back focus", fresh(bf)), ("Air spaces", fresh(air)), ("Design variables", fresh(variables))]
        var other: [ParamPath] = []
        for i in 1..<max(s.imageIndex, 1) {
            let id = s.surfaces[i].id
            other += [.decenterX(surface: id), .decenterY(surface: id), .tiltX(surface: id), .tiltY(surface: id), .thickness(surface: id)]
            if s.surfaces[i].shape.hasCurvature { other.append(.curvature(surface: id)) }
        }
        out.append(("Surfaces", fresh(other.filter { s.value(at: $0) != nil })))
        return out.filter { !$0.1.isEmpty }
    }
}

// MARK: - Criterion

/// What a trial is judged by: the RMS spot, the RMS wavefront or the MTF
/// (the engine's default merit for tolerancing, without thickness
/// boundaries), or the Optimize panel's merit function.
public struct ToleranceCriterion: Hashable, Sendable {
    public enum Kind: String, CaseIterable, Sendable, Identifiable {
        case rmsSpot = "rms_spot", wavefront, mtf, designMerit = "design_merit"
        public var id: String { rawValue }
        public var title: String {
            switch self {
            case .rmsSpot: "RMS spot radius"
            case .wavefront: "RMS wavefront"
            case .mtf: "MTF"
            case .designMerit: "Optimizer's merit"
            }
        }
    }

    public var kind = Kind.rmsSpot
    /// Spot and wavefront reference, MTF frequencies and method, pupil
    /// sampling, fields and wavelengths (the merit editor's generator).
    public var defaults: DefaultMeritSettings = {
        var d = DefaultMeritSettings()
        d.thickness = false
        d.mtfFrequencies = [20]
        return d
    }()
    /// The pass limit for the yield, in the criterion's display unit
    /// (µm, waves, MTF or merit); `nil`: no yield.
    public var threshold: Double?

    public init() {}

    /// The `criterion` / `default_merit` settings of the job.
    public func jobSettings(merit: MeritSettings) -> [String: JSONValue] {
        switch kind {
        case .designMerit:
            switch merit.mode {
            case .custom: return ["criterion": merit.spec]
            case .default:
                var d = merit.defaults
                d.thickness = false
                return ["default_merit": d.json]
            }
        default:
            var d = defaults
            d.thickness = false
            d.squeezeTarget = nil
            d.criterion = kind == .wavefront ? .wavefront : kind == .mtf ? .mtf : .rmsSpot
            // The MTF shortfall from 1, so the criterion never saturates at 0.
            if kind == .mtf { d.mtfTarget = 1 }
            return ["default_merit": d.json]
        }
    }

    /// How criterion values (merits) read for this kind.
    public func display(merit: MeritSettings) -> CriterionDisplay {
        switch kind {
        case .rmsSpot: return .spot
        case .wavefront: return .wavefront
        case .mtf: return .mtf
        case .designMerit:
            guard merit.mode == .default else { return .merit }
            switch merit.defaults.criterion {
            case .rmsSpot: return .spot
            case .wavefront: return .wavefront
            case .mtf: return merit.defaults.mtfTarget >= 1 ? .mtf : .merit
            }
        }
    }

    public var json: JSONValue {
        .object(["kind": .string(kind.rawValue), "default_merit": defaults.json, "threshold": threshold.map(JSONValue.number) ?? .null])
    }

    public init(json: JSONValue?) {
        guard let j = json, j.object != nil else { return }
        kind = j["kind"]?.string.flatMap(Kind.init(rawValue:)) ?? .rmsSpot
        if j["default_merit"] != nil { defaults = DefaultMeritSettings(json: j["default_merit"]) }
        threshold = j["threshold"]?.nf.flatMap { $0.isFinite ? $0 : nil }
    }
}

/// Criterion values as a lens designer reads them. The engine's criterion
/// is a merit, the weighted mean of squares: its square root is the RMS
/// spot radius (mm) or wavefront (waves); for the MTF, the root mean square
/// shortfall from 1, so 1 − √c is the mean MTF over fields, directions and
/// frequencies. Lower merits are better in every case.
public enum CriterionDisplay: Sendable, Equatable {
    case spot, wavefront, mtf, merit

    public var title: String {
        switch self {
        case .spot: "RMS spot"
        case .wavefront: "RMS wavefront"
        case .mtf: "MTF"
        case .merit: "Merit"
        }
    }

    public var unit: String {
        switch self {
        case .spot: "µm"
        case .wavefront: "waves"
        case .mtf, .merit: ""
        }
    }

    /// Higher shown values are better (MTF).
    public var higherIsBetter: Bool { self == .mtf }

    public func value(_ c: Double) -> Double {
        guard c.isFinite else { return .nan }
        switch self {
        case .spot: return c.squareRoot() * 1000
        case .wavefront: return c.squareRoot()
        case .mtf: return 1 - c.squareRoot()
        case .merit: return c
        }
    }

    /// The criterion (merit) of a shown value: the yield threshold.
    public func criterion(_ v: Double) -> Double {
        switch self {
        case .spot: return (v / 1000) * (v / 1000)
        case .wavefront: return v * v
        case .mtf: return (1 - v) * (1 - v)
        case .merit: return v
        }
    }

    public func format(_ v: Double) -> String {
        guard v.isFinite else { return "—" }
        switch self {
        case .spot: return String(format: v < 10 ? "%.2f" : "%.1f", v)
        case .wavefront: return String(format: "%.3f", v)
        case .mtf: return String(format: "%.3f", v)
        case .merit: return Fmt.short(v)
        }
    }

    public func formatWithUnit(_ v: Double) -> String { unit.isEmpty ? format(v) : format(v) + " " + unit }

    /// A change in the shown value, signed.
    public func formatChange(_ d: Double) -> String {
        guard d.isFinite else { return "—" }
        let s = format(abs(d))
        if s.allSatisfy({ $0 == "0" || $0 == "." }) { return "\u{2007}" + s }
        return (d < 0 ? "−" : "+") + s
    }
}

// MARK: - The saved set

/// The design's tolerancing: rows, compensators, the generator's values,
/// the criterion and the analysis to run. The project's `tolerances`
/// section holds the engine's `ToleranceSpec` (`schema_version`,
/// `operands`, `compensators`, `criterion`, `monte_carlo`) as last
/// expanded, plus the app's rows under `app`.
public struct ToleranceSettings: Hashable, Sendable {
    public enum Mode: String, CaseIterable, Sendable, Identifiable {
        case sensitivity, monteCarlo = "monte_carlo", both
        public var id: String { rawValue }
        public var title: String {
            switch self {
            case .sensitivity: "Sensitivity"
            case .monteCarlo: "Monte Carlo"
            case .both: "Both"
            }
        }
    }

    /// He–Ne, the usual interferometer.
    public static let defaultTestWavelengthUm = 0.6328
    /// The engine's default levels: the median, 90 % and 97.7 % (2σ one-sided).
    public static let defaultPercentiles: [Double] = [50, 90, 97.7]
    /// The engine keeps at most this many trial records.
    public static let maxRecordTrials = 10_000

    public var rows: [ToleranceRow] = []
    public var compensators: [ToleranceCompensator] = []
    public var generator = ToleranceGenerator()
    public var criterion = ToleranceCriterion()
    public var mode = Mode.both
    public var trials = 200
    public var seed = 0
    /// The interferometer wavelength (µm) fringes are counted at: power
    /// and irregularity (the engine's `irregularity_wavelength_um`).
    public var testWavelengthUm = ToleranceSettings.defaultTestWavelengthUm
    /// Monte-Carlo levels the engine reports, in percent of the builds.
    public var percentiles = ToleranceSettings.defaultPercentiles
    /// Trials whose perturbations and compensators are kept for the list.
    public var recordTrials = 200
    /// Whether the set was ever generated or edited (a new design starts
    /// from the generator's default set).
    public var isEmpty: Bool { rows.isEmpty && compensators.isEmpty }

    public init() {}

    /// The default set of a design: the precision preset and the back
    /// focus as compensator.
    public static func defaults(for s: OpticalSystem) -> ToleranceSettings {
        var t = ToleranceSettings()
        t.rows = t.generator.rows(for: s)
        t.compensators = ToleranceCompensator.backFocus(s).map { [$0] } ?? []
        return t
    }

    public init(section: JSONValue?, system: OpticalSystem) {
        guard let s = section, s.object != nil else { return }
        let app = s["app"]
        if let rows = app?["rows"]?.array {
            self.rows = rows.compactMap(ToleranceRow.init(json:))
        } else {
            // A spec written elsewhere: its operands become rows.
            self.rows = (s["operands"]?.array ?? []).compactMap { ToleranceRow(operand: $0, system: system) }
        }
        compensators = (s["compensators"]?.array ?? []).compactMap(ToleranceCompensator.init(json:))
        generator = ToleranceGenerator(json: app?["generator"])
        criterion = ToleranceCriterion(json: app?["criterion"])
        mode = app?["mode"]?.string.flatMap(Mode.init(rawValue:)) ?? .both
        trials = s[path: "monte_carlo", "trials"]?.int ?? trials
        seed = s[path: "monte_carlo", "seed"]?.int ?? seed
        if let p = s[path: "monte_carlo", "percentiles"]?.nfArray, let v = Self.validPercentiles(p) { percentiles = v }
        recordTrials = s[path: "monte_carlo", "record_trials"]?.int.map { min(max($0, 0), Self.maxRecordTrials) } ?? recordTrials
        if let w = app?["test_wavelength_um"]?.nf, w > 0, w.isFinite { testWavelengthUm = w }
    }

    /// Percent levels in [0, 100], sorted and without repeats; `nil` when
    /// none is left.
    public static func validPercentiles(_ v: [Double]) -> [Double]? {
        let ok = Array(Set(v.filter { $0.isFinite && $0 >= 0 && $0 <= 100 })).sorted()
        return ok.isEmpty ? nil : ok
    }

    /// `context` with this set's test wavelength.
    public func context(_ c: ToleranceContext) -> ToleranceContext {
        var out = c
        out.testWavelengthUm = testWavelengthUm
        return out
    }

    /// Records the engine keeps: at most every trial, at most its cap.
    public var effectiveRecordTrials: Int { min(max(recordTrials, 0), trials, Self.maxRecordTrials) }

    /// The section: the engine's spec as expanded in `context`, and the rows.
    public func section(context: ToleranceContext?, merit: MeritSettings = MeritSettings()) -> JSONValue {
        let ops: [JSONValue] = context.map { c in rows.compactMap { try? $0.operand(in: self.context(c)).get() } } ?? []
        let threshold = criterion.threshold.map { criterion.display(merit: merit).criterion($0) }
        return .object([
            "schema_version": .integer(1),
            "operands": .array(ops),
            "compensators": .array(compensators.map(\.json)),
            "criterion": criterion.jobSettings(merit: merit)["criterion"] ?? .null,
            "monte_carlo": .object([
                "trials": .integer(Int64(trials)), "seed": .integer(Int64(seed)), "batch": .integer(16),
                "threshold": threshold.map(JSONValue.number) ?? .null,
                "percentiles": .array(percentiles.map(JSONValue.number)),
                "record_trials": .integer(Int64(effectiveRecordTrials)),
            ]),
            "app": .object([
                "rows": .array(rows.map(\.json)), "generator": generator.json, "criterion": criterion.json, "mode": .string(mode.rawValue),
                "test_wavelength_um": .number(testWavelengthUm),
            ]),
        ])
    }

    /// Hashable by the saved content (rows' session ids aside).
    public static func == (a: Self, b: Self) -> Bool { a.section(context: nil) == b.section(context: nil) }
    public func hash(into h: inout Hasher) { h.combine(section(context: nil)) }

    // MARK: Editing

    /// Adds rows after `index` (or at the end); returns their ids.
    @discardableResult
    public mutating func add(_ new: [ToleranceRow], after id: UUID? = nil) -> [UUID] {
        let at = id.flatMap { i in rows.firstIndex { $0.id == i }.map { $0 + 1 } } ?? rows.count
        rows.insert(contentsOf: new, at: at)
        // An element row joins the range and pivot of the rows already
        // moving from its first surface.
        let newIDs = Set(new.map(\.id))
        for r in new where r.kind.isElement {
            if let other = rows.first(where: { !newIDs.contains($0.id) && $0.kind.isElement && $0.surface == r.surface }),
                let i = rows.firstIndex(where: { $0.id == r.id })
            {
                rows[i].last = other.last
                rows[i].pivot = other.pivot
            }
        }
        return new.map(\.id)
    }

    public mutating func delete(_ ids: Set<UUID>) { rows.removeAll { ids.contains($0.id) } }

    /// Copies of the rows, each after its original; returns their ids.
    @discardableResult
    public mutating func duplicate(_ ids: Set<UUID>) -> [UUID] {
        var out: [UUID] = []
        for i in rows.indices.reversed() where ids.contains(rows[i].id) {
            var copy = rows[i]
            copy.id = UUID()
            rows.insert(copy, at: i + 1)
            out.append(copy.id)
        }
        return out
    }

    /// Replaces the rows by the generator's set for `system`.
    public mutating func generate(for system: OpticalSystem) {
        rows = generator.rows(for: system)
    }

    /// Moves element row `id` to the range `first…last` about `pivot`
    /// (`nil`: the first vertex). Every element row starting on `first`
    /// takes the same range and pivot, as the engine requires: they move
    /// one rigid group.
    public mutating func setElement(_ id: UUID, first: SurfaceID, last: SurfaceID, pivot: SurfaceID?) {
        guard let k = rows.firstIndex(where: { $0.id == id }), rows[k].kind.isElement else { return }
        rows[k].surface = first
        rows[k].last = last
        rows[k].pivot = pivot == first ? nil : pivot
        for i in rows.indices where rows[i].kind.isElement && rows[i].surface == first {
            rows[i].last = rows[k].last
            rows[i].pivot = rows[k].pivot
        }
    }

    /// The rows of `rows` that cannot run, with why.
    public func issues(in c: ToleranceContext) -> [UUID: ToleranceIssue] {
        let c = context(c)
        var out: [UUID: ToleranceIssue] = [:]
        for r in rows { if case .failure(let e) = r.engineLimits(in: c) { out[r.id] = e } }
        // Element rows that start on one surface share their range and pivot.
        let elements = rows.filter { $0.kind.isElement && out[$0.id] == nil }
        for (first, group) in Dictionary(grouping: elements, by: \.surface) {
            let ranges = Set(group.map { [$0.last ?? first, $0.pivot ?? first] })
            guard ranges.count > 1 else { continue }
            let at = c.system.surfaceIndex(of: first).map { "S\($0)" } ?? "one surface"
            for r in group {
                out[r.id] = .message("Element tolerances starting at \(at) move one group: give them the same last surface and pivot.")
            }
        }
        return out
    }

    // MARK: The run

    /// The job settings in `context`, with the ids of the rows sent, in the
    /// engine's operand order (rows that cannot run are left out).
    public func jobSettings(in c: ToleranceContext, merit: MeritSettings) -> (settings: [String: JSONValue], rowIDs: [UUID]) {
        let c = context(c)
        var ops: [JSONValue] = [], ids: [UUID] = []
        let blocked = issues(in: c)
        for r in rows where blocked[r.id] == nil {
            if case .success(let o) = r.operand(in: c) {
                ops.append(o)
                ids.append(r.id)
            }
        }
        var s: [String: JSONValue] = [
            "tolerances": .array(ops), "compensators": .array(compensators.map(\.json)), "mode": .string(mode.rawValue),
            "trials": .integer(Int64(trials)), "seed": .integer(Int64(seed)),
            "percentiles": .array(percentiles.map(JSONValue.number)),
            "record_trials": .integer(Int64(mode == .sensitivity ? 0 : effectiveRecordTrials)),
            "irregularity_wavelength_um": .number(testWavelengthUm),
        ]
        s.merge(criterion.jobSettings(merit: merit)) { _, b in b }
        if let t = criterion.threshold { s["threshold"] = .number(criterion.display(merit: merit).criterion(t)) }
        return (s, ids)
    }
}