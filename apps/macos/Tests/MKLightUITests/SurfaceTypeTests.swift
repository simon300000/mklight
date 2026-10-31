import Foundation
import MKLightModel
import Testing

@testable import MKLightUI

/// The data-driven forms of the extended surface types, and the display
/// sags of their previews against the closed forms in docs/SURFACES.md.
@Suite struct SurfaceTypeTests {
    /// Every parameter and coefficient list of every type addresses a key
    /// the model has, writes back through the model's strict decoder, and
    /// its variable path names the value it edits.
    @Test(arguments: SurfaceShape.Kind.allCases)
    func descriptorsAddressTheModel(_ kind: SurfaceShape.Kind) throws {
        var system = SampleSystems.singlet()
        system.setShapeKind(kind, at: 1)
        if kind == .diffractive, case .diffractive(_, let p) = system.surfaces[1].shape {
            // A conic base, so the base's parameters show too.
            system.surfaces[1].shape = .diffractive(base: .conic(curvature: 0.01, conic: 0), phase: p)
        }
        let shape = system.surfaces[1].shape
        let id = system.surfaces[1].id
        let type = SurfaceTypes.descriptor(for: shape)
        #expect(type.kind == kind && type.id == kind.typeName)
        for section in type.sections where section.applies(shape) {
            for p in section.params where SurfaceTypes.value(shape, at: p.key) != nil {
                let v = try #require(SurfaceTypes.value(shape, at: p.key), "\(kind) \(p.key)")
                #expect(SurfaceTypes.shape(shape, setting: p.key, to: v) == shape, "\(kind) \(p.key) round-trips")
                if let path = p.path?(id) {
                    #expect(path.resolves(in: system), "\(kind) \(p.key): \(path)")
                }
            }
            for c in section.choices {
                #expect(SurfaceTypes.value(shape, at: c.key)?.string != nil, "\(kind) \(c.key)")
            }
            if let list = section.coefficients, SurfaceTypes.value(shape, at: list.key) != nil {
                let edited = try #require(SurfaceTypes.shape(shape, setting: list.key, to: .array([.number(1e-6), .number(-2e-7)])))
                #expect(SurfaceTypes.coefficients(list, in: edited) == [1e-6, -2e-7], "\(kind) \(list.key)")
                #expect(list.maxCount(edited) >= 2)
                if let path = list.path?(id, 1) {
                    var s = system
                    s.surfaces[1].shape = edited
                    #expect(s.value(at: path) == -2e-7, "\(kind) \(list.key) variable path")
                }
            }
        }
    }

    @Test func termLabelsCarryTheirMeaning() {
        let xy = SurfaceTypes.descriptor(for: .xyPolynomial).sections[1].coefficients!
        let shape = SurfaceShape.xyPolynomial(curvature: 0, conic: 0, normRadius: 5, coefficients: [])
        #expect(xy.term(6, shape) == "C(x²y)" && xy.term(0, shape) == "C(x)")
        let zernike = SurfaceTypes.descriptor(for: .zernikeSag).sections[1].coefficients!
        let fringe = SurfaceShape.zernikeSag(curvature: 0, conic: 0, normRadius: 5, ordering: .fringe, coefficients: [])
        #expect(zernike.term(6, fringe) == "Z₇" && zernike.meaning(6, fringe) == "coma x")
        #expect(zernike.meaning(8, fringe) == "spherical" && zernike.meaning(3, fringe) == "defocus")
        let standard = SurfaceShape.zernikeSag(curvature: 0, conic: 0, normRadius: 5, ordering: .standard, coefficients: [])
        #expect(zernike.meaning(10, standard) == "spherical", "Noll Z11 is spherical")
        let q = SurfaceTypes.descriptor(for: .qCon).sections[1].coefficients!
        #expect(q.term(2, shape) == "a₂" && q.meaning(2, shape) == "u⁴Q₂ᶜᵒⁿ")
        let odd = SurfaceTypes.descriptor(for: .oddAsphere).sections[1].coefficients!
        #expect(odd.term(2, shape) == "β₃" && odd.meaning(2, shape) == "r³" && odd.unit(2) == "mm⁻²")
    }

    @Test func typeMenuGroupsEveryTypeAndTheCylinder() {
        let choices = SurfaceTypes.menu.flatMap(\.choices)
        #expect(Set(choices.compactMap { if case .kind(let k) = $0 { k } else { nil } }) == Set(SurfaceShape.Kind.allCases))
        #expect(choices.contains(.cylinder(.x)) && choices.contains(.cylinder(.y)))
        #expect(SurfaceTypes.menu.map(\.group) == SurfaceTypeGroup.allCases)
        #expect(SurfaceTypes.choice(for: .cylinder(curvature: 0.02, axis: .x)) == .cylinder(.x))
        #expect(SurfaceTypes.choice(for: .conic(curvature: 0.02, conic: 0)) == .kind(.conic))
    }

    // MARK: Display sags

    @Test func forbesPolynomialsMatchTheirClosedForms() {
        for t in stride(from: 0.0, through: 1.0, by: 0.125) {
            #expect(abs(ShapeSag.qCon([0, 1], t) - (6 * t - 5)) < 1e-12)
            #expect(abs(ShapeSag.qCon([0, 0, 1], t) - (28 * t * t - 42 * t + 15)) < 1e-12)
            #expect(abs(ShapeSag.qBfs([1], t) - 1) < 1e-12)
            #expect(abs(ShapeSag.qBfs([0, 1], t) - (13 - 16 * t) / 19.0.squareRoot()) < 1e-12)
            #expect(abs(ShapeSag.qBfs([0, 0, 1], t) - (29 - 100 * t + 76 * t * t) / (190.0.squareRoot() / 2)) < 1e-12)
        }
    }

    @Test func extendedShapesReduceToTheirBases() {
        let c = 1 / 60.0, k = -0.4
        let pts: [(Double, Double)] = [(0, 0), (1.5, 0), (0, 2), (1, -2.5), (-3, 1)]
        for (x, y) in pts {
            let conic = ShapeSag.conic(c, k, x * x + y * y)!
            let sphere = ShapeSag.conic(c, 0, x * x + y * y)!
            #expect(abs(ShapeSag.sag(.qCon(curvature: c, conic: k, normRadius: 5, coefficients: [0, 0]), x: x, y: y)! - conic) < 1e-14)
            #expect(abs(ShapeSag.sag(.qBfs(curvature: c, normRadius: 5, coefficients: [0, 0]), x: x, y: y)! - sphere) < 1e-14)
            #expect(abs(ShapeSag.sag(.toroid(curvatureX: c, curvatureY: c, conicY: 0, yAsphere: []), x: x, y: y)! - sphere) < 1e-12)
            #expect(abs(ShapeSag.sag(.biconic(curvatureX: c, curvatureY: c, conicX: k, conicY: k, xAsphere: [], yAsphere: []), x: x, y: y)! - conic) < 1e-14)
            #expect(abs(ShapeSag.sag(.xyPolynomial(curvature: c, conic: k, normRadius: 5, coefficients: []), x: x, y: y)! - conic) < 1e-14)
        }
        // A cylinder has no sag along its plane axis.
        #expect(ShapeSag.sag(.cylinder(curvature: 0.02, axis: .x), x: 0, y: 3) == 0)
        #expect(ShapeSag.sag(.cylinder(curvature: 0.02, axis: .x), x: 3, y: 0)! > 0)
        // Normalised polynomials end at their radius.
        #expect(ShapeSag.sag(.qCon(curvature: c, conic: 0, normRadius: 5, coefficients: [1e-3]), x: 0, y: 5.1) == nil)
    }

    @Test func polynomialTermsFollowTheEngineOrder() {
        // XY: index 3 is XY, index 6 is X²Y.
        let xy = SurfaceShape.xyPolynomial(curvature: 0, conic: 0, normRadius: 2, coefficients: [0, 0, 0, 1e-3, 0, 0, 2e-3])
        let (x, y) = (0.6, -1.4)
        let X = x / 2, Y = y / 2
        #expect(abs(ShapeSag.sag(xy, x: x, y: y)! - (1e-3 * X * Y + 2e-3 * X * X * Y)) < 1e-15)
        // Fringe Z5 = ρ² cos 2θ (unnormalised); Standard Z4 = √3 (2ρ² − 1).
        let rho = 0.7, theta = 0.4
        let z5 = ShapeSag.zernike(ZernikeTerm(j: 5, ordering: .fringe)!, ordering: .fringe, rho: rho, theta: theta)
        #expect(abs(z5 - rho * rho * cos(2 * theta)) < 1e-14)
        let z4 = ShapeSag.zernike(ZernikeTerm(j: 4, ordering: .standard)!, ordering: .standard, rho: rho, theta: theta)
        #expect(abs(z4 - 3.0.squareRoot() * (2 * rho * rho - 1)) < 1e-14)
        // Fringe Z9 (spherical) = 6ρ⁴ − 6ρ² + 1.
        let z9 = ShapeSag.zernike(ZernikeTerm(j: 9, ordering: .fringe)!, ordering: .fringe, rho: rho, theta: theta)
        #expect(abs(z9 - (6 * pow(rho, 4) - 6 * rho * rho + 1)) < 1e-14)
    }

    /// φ = −m λ A₁ / (π r_norm²); a quadratic phase of focal length f has
    /// the local period λ f / r (the grating equation).
    @Test func diffractivePowerAndZonePeriod() {
        let f = 100.0, lambda = 0.55, rn = 5.0
        let a1 = -Double.pi * rn * rn / (lambda * 1e-3 * f)
        let phase = DiffractivePhase(normRadius: rn, coefficients: [a1], designOrder: 1, designWavelengthUm: lambda, traceOrder: 1)
        #expect(abs(ShapeSag.phasePower(phase, order: 1, wavelengthUm: lambda) - 1 / f) < 1e-12)
        #expect(abs(ShapeSag.phasePower(phase, order: 2, wavelengthUm: lambda) - 2 / f) < 1e-12)
        let period = try! #require(ShapeSag.minimumPeriod(phase, upTo: rn))
        #expect(abs(period.period - lambda * 1e-3 * f / rn) < 1e-9)
        #expect(abs(period.radius - rn) < 1e-9)
    }

    @Test func departureMapIsTheFreeformPart() {
        let shape = SurfaceShape.zernikeSag(curvature: 0.02, conic: 0, normRadius: 4, ordering: .fringe, coefficients: [0, 0, 0, 0, 1e-3])
        let map = SurfaceTypes.departureMap(shape, semiDiameter: 4, n: 20)
        let values = map.compactMap { $0 }
        #expect(map.count == 400 && values.count < 400, "corners outside the disk are empty")
        #expect((values.max() ?? 0) <= 1e-3 + 1e-12 && (values.min() ?? 0) >= -1e-3 - 1e-12)
        #expect((values.max() ?? 0) > 0.8e-3, "astigmatism reaches its amplitude near the edge")
    }

    @Test func anamorphicRowsEditTheirYSection() throws {
        var s = SampleSystems.singlet()
        s.setShapeKind(.toroid, at: 1)
        let rows = LensRow.rows(for: s)
        #expect(rows[1].hints[.radius] == "y" && rows[1].editable.contains(.radius) && rows[1].editable.contains(.conic))
        let edit = try #require(rows[1].edit(.radius, text: "50")).get()
        edit.apply(to: &s)
        #expect(s.surfaces[1].shape.shapeParameter(.curvatureY) == 0.02)
        #expect(LensColumn.radius.param(for: s.surfaces[1]) == .curvature(surface: s.surfaces[1].id), "the curvature path is the y section")
        // A Q-BFS surface has a radius but no conic.
        s.setShapeKind(.qBfs, at: 1)
        let q = LensRow.rows(for: s)[1]
        #expect(q.editable.contains(.radius) && !q.editable.contains(.conic))
    }
}
