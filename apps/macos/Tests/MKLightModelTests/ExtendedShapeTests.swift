import Foundation
import Testing

@testable import MKLightModel

/// The extended surface shapes (docs/SURFACES.md): their JSON form, term
/// numbering, parameter paths, editing and structural validation.
@Suite struct ExtendedShapeTests {
    private func decode(_ json: String) throws -> SurfaceShape {
        try MKLightJSON.decode(SurfaceShape.self, from: Data(json.utf8))
    }

    private func encoded(_ shape: SurfaceShape) throws -> JSONValue {
        try JSONValue.parse(MKLightJSON.encode(shape))
    }

    // MARK: Fixtures

    @Test func extendedShapesCoverEveryNewKind() throws {
        let s = try Fixtures.roundTrip(OpticalSystem.self, "extended_shapes")
        #expect(s.structuralIssues().isEmpty, "\(s.structuralIssues())")
        let kinds = Set(s.surfaces.map(\.shape.kind))
        #expect(Set(SurfaceShape.Kind.allCases.filter { !$0.isLegacy }).isSubset(of: kinds))
        let bases = s.surfaces.compactMap { if case .diffractive(let b, _) = $0.shape { b.kind } else { nil } }
        #expect(bases == [.plane, .conic, .evenAsphere])
        let orders = s.surfaces.compactMap {
            if case .diffractive(_, let p) = $0.shape { Pair(p.designOrder, p.traceOrder) } else { nil }
        }
        #expect(orders == [Pair(1, 1), Pair(2, -1), Pair(-1, 0)])
        let orderings = s.surfaces.compactMap { if case .zernikeSag(_, _, _, let o, _) = $0.shape { o } else { nil } }
        #expect(orderings == [.fringe, .standard])
        #expect(s.surfaces.compactMap(\.shape.cylinderAxis) == [.x])
        #expect(s.surfaces[9].shape == .cylinder(curvature: 0.025, axis: .x))
        #expect(s.surfaces.contains { !$0.shape.isLegacy })
    }

    @Test func bundledSamplesRoundTripAndValidate() throws {
        let entries = try Fixtures.roundTrip([SampleSystems.Entry].self, "samples")
        let ids = entries.map(\.id)
        #expect(ids.count == 16)
        for id in [
            "hybrid_doe", "anamorphic", "biconic", "toroid", "q_con", "q_bfs", "odd_asphere", "xy_freeform", "zernike_sag",
        ] {
            #expect(ids.contains(id), "\(id)")
        }
        for e in entries {
            #expect(e.system.structuralIssues().isEmpty, "\(e.id): \(e.system.structuralIssues())")
        }
        // The app bundle ships the same file.
        #expect(SampleSystems.bundled.map(\.id) == ids)
        #expect(SampleSystems.bundled("anamorphic")?.surfaces.compactMap(\.shape.cylinderAxis) == [.x, .x])
    }

    // MARK: Strict JSON

    @Test func defaultedFieldsMayBeMissingAndAreAlwaysWritten() throws {
        let toroid = try decode(#"{"type":"toroid","curvature_x":0.1,"curvature_y":0.2}"#)
        #expect(toroid == .toroid(curvatureX: 0.1, curvatureY: 0.2, conicY: 0, yAsphere: []))
        #expect(
            try encoded(toroid)
                == JSONValue.parse(
                    Data(#"{"type":"toroid","curvature_x":0.1,"curvature_y":0.2,"conic_y":0,"y_asphere":[]}"#.utf8)))
        let biconic = try decode(#"{"type":"biconic","curvature_x":0.1,"curvature_y":0}"#)
        #expect(biconic == .cylinder(curvature: 0.1, axis: .x))
        #expect(
            try decode(#"{"type":"zernike_sag","curvature":0,"norm_radius":5,"coefficients":[1]}"#)
                == .zernikeSag(curvature: 0, conic: 0, normRadius: 5, ordering: .fringe, coefficients: [1]))
        for json in [
            #"{"type":"odd_asphere","curvature":0,"coefficients":[]}"#,
            #"{"type":"q_con","curvature":0,"norm_radius":1,"coefficients":[]}"#,
            #"{"type":"xy_polynomial","curvature":0,"norm_radius":1,"coefficients":[]}"#,
        ] {
            #expect(try decode(json).conicConstant == 0)
        }
        // Fields without a serde default are required.
        for json in [
            #"{"type":"odd_asphere","curvature":0}"#,
            #"{"type":"q_bfs","curvature":0,"coefficients":[]}"#,
            #"{"type":"toroid","curvature_y":0}"#,
            #"{"type":"diffractive","base":{"type":"plane"}}"#,
            #"{"type":"diffractive","base":{"type":"plane"},"phase":{"norm_radius":1,"coefficients":[],"design_order":1,"design_wavelength_um":0.5}}"#,
        ] {
            #expect(throws: DecodingError.self, "\(json)") { try decode(json) }
        }
    }

    @Test func nollIsAnAliasOfStandard() throws {
        let s = try decode(#"{"type":"zernike_sag","curvature":0,"conic":0,"norm_radius":5,"ordering":"noll","coefficients":[]}"#)
        #expect(s == .zernikeSag(curvature: 0, conic: 0, normRadius: 5, ordering: .standard, coefficients: []))
        #expect(try encoded(s)["ordering"] == .string("standard"))
        #expect(throws: DecodingError.self) {
            try decode(#"{"type":"zernike_sag","curvature":0,"norm_radius":5,"ordering":"ansi","coefficients":[]}"#)
        }
    }

    @Test(arguments: [
        #"{"type":"odd_asphere","curvature":0,"conic":0,"coefficients":[],"x":1}"#,
        #"{"type":"q_con","curvature":0,"conic":0,"norm_radius":1,"coefficients":[],"x":1}"#,
        #"{"type":"q_bfs","curvature":0,"conic":0,"norm_radius":1,"coefficients":[]}"#,
        #"{"type":"toroid","curvature_x":0,"curvature_y":0,"conic_x":0}"#,
        #"{"type":"biconic","curvature_x":0,"curvature_y":0,"x":1}"#,
        #"{"type":"xy_polynomial","curvature":0,"conic":0,"norm_radius":1,"coefficients":[],"ordering":"fringe"}"#,
        #"{"type":"zernike_sag","curvature":0,"conic":0,"norm_radius":1,"coefficients":[],"x":1}"#,
        #"{"type":"diffractive","base":{"type":"plane"},"phase":{"norm_radius":1,"coefficients":[],"design_order":1,"design_wavelength_um":0.5,"trace_order":1},"x":1}"#,
        #"{"type":"diffractive","base":{"type":"plane"},"phase":{"norm_radius":1,"coefficients":[],"design_order":1,"design_wavelength_um":0.5,"trace_order":1,"x":1}}"#,
        #"{"type":"diffractive","base":{"type":"plane","x":1},"phase":{"norm_radius":1,"coefficients":[],"design_order":1,"design_wavelength_um":0.5,"trace_order":1}}"#,
    ])
    func unknownKeysAreRejected(json: String) {
        #expect(throws: DecodingError.self) { try decode(json) }
    }

    @Test func ordersAreSignedInt32() throws {
        func doe(_ design: String, _ trace: String) -> String {
            #"{"type":"diffractive","base":{"type":"plane"},"phase":{"norm_radius":1,"coefficients":[-1],"design_order":"#
                + design + #","design_wavelength_um":0.5,"trace_order":"# + trace + "}}"
        }
        let s = try decode(doe("-2147483648", "2147483647"))
        guard case .diffractive(.plane, let p) = s else { Issue.record("not a DOE"); return }
        #expect(p.designOrder == Int(Int32.min) && p.traceOrder == Int(Int32.max))
        #expect(throws: DecodingError.self) { try decode(doe("1", "2147483648")) }
        #expect(throws: DecodingError.self) { try decode(doe("-2147483649", "1")) }
    }

    // MARK: Term numbering

    @Test func xyTermsFollowSurfacesTable() {
        let table = [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]
        for (j, (m, n)) in table.enumerated() {
            #expect(XYTerm(index: j) == XYTerm(m: m, n: n), "index \(j)")
        }
        #expect(XYTerm(index: 64) == XYTerm(m: 0, n: 10))
        #expect(XYTerm(index: 54) == XYTerm(m: 10, n: 0))
        #expect(XYTerm(index: 65) == nil && XYTerm(index: -1) == nil)
        for j in 0..<CoefficientSet.maxXYTerms { #expect(XYTerm(index: j)?.index == j) }
        #expect(XYTerm(index: 6)?.monomial == "x²y")
        #expect(CoefficientSet.xy.termLabel(3) == "C(xy)")
    }

    @Test func zernikeTermsFollowSurfacesTable() {
        let fringe = [(0, 0), (1, 1), (1, -1), (2, 0), (2, 2), (2, -2), (3, 1), (3, -1), (4, 0), (3, 3), (3, -3)]
        let standard = [(0, 0), (1, 1), (1, -1), (2, 0), (2, -2), (2, 2), (3, -1), (3, 1), (3, -3), (3, 3), (4, 0)]
        for (k, (n, m)) in fringe.enumerated() {
            #expect(ZernikeTerm(j: k + 1, ordering: .fringe) == ZernikeTerm(n: n, m: m), "Fringe Z\(k + 1)")
        }
        for (k, (n, m)) in standard.enumerated() {
            #expect(ZernikeTerm(j: k + 1, ordering: .standard) == ZernikeTerm(n: n, m: m), "Standard Z\(k + 1)")
        }
        #expect(ZernikeTerm(j: 36, ordering: .fringe) == ZernikeTerm(n: 10, m: 0))
        #expect(ZernikeTerm(j: 37, ordering: .fringe) == ZernikeTerm(n: 12, m: 0))
        #expect(ZernikeTerm(j: 45, ordering: .standard) == ZernikeTerm(n: 8, m: -8))
        #expect(ZernikeTerm(j: 38, ordering: .fringe) == nil && ZernikeTerm(j: 46, ordering: .standard) == nil)
        #expect(ZernikeTerm(j: 0, ordering: .fringe) == nil)
        for ordering in ZernikeOrdering.allCases {
            let terms = (1...ordering.maxTerms).compactMap { ZernikeTerm(j: $0, ordering: ordering) }
            #expect(terms.count == ordering.maxTerms && Set(terms).count == terms.count)
            #expect(terms.allSatisfy { abs($0.m) <= $0.n && ($0.n - abs($0.m)) % 2 == 0 })
        }
        // Standard: n ≤ 8 is complete; even j cosine, odd j sine.
        let std = (1...45).compactMap { ZernikeTerm(j: $0, ordering: .standard) }
        #expect(std.allSatisfy { $0.n <= 8 })
        for (k, t) in std.enumerated() where t.m != 0 { #expect((t.m > 0) == ((k + 1) % 2 == 0)) }
    }

    // MARK: Parameter paths

    private func extended() throws -> OpticalSystem {
        try MKLightJSON.decode(OpticalSystem.self, from: Fixtures.data("extended_shapes"))
    }

    @Test func coefficientPathsReadZeroAndExtendWithZeros() throws {
        var s = try extended()
        let id = SurfaceID(1)  // odd asphere [0, 1e-5, -1e-7]
        let p = ParamPath.shapeCoefficient(surface: id, set: .odd, index: 6)
        #expect(s.value(at: p) == 0)
        let ok1 = s.setValue(2e-9, at: p)
        #expect(ok1)
        #expect(s.surfaces[1].shape.coefficients(in: .odd) == [0, 1e-5, -1e-7, 0, 0, 0, 2e-9])
        #expect(s.value(at: .shapeCoefficient(surface: id, set: .odd, index: 1)) == 1e-5)
        let beyond = ParamPath.shapeCoefficient(surface: id, set: .odd, index: 20)
        #expect(beyond.unresolvedReason(in: s) == "[coefficient_count] index exceeds shape limit")
        #expect(s.value(at: beyond) == nil)
        let before = s
        let ok2 = s.setValue(1, at: beyond)
        #expect(!ok2)
        let ok3 = s.setValue(1, at: .shapeCoefficient(surface: id, set: .q, index: 0))
        #expect(!ok3)
        #expect(s == before)
        #expect(
            ParamPath.shapeCoefficient(surface: id, set: .xy, index: 0).unresolvedReason(in: s)
                == "coefficient set does not belong to shape")
        // Limits per shape.
        let limits = s.surfaces.flatMap { sf in sf.shape.coefficientSets.map { "\($0.rawValue)\($0.maxCount(for: sf.shape)!)" } }
        #expect(Set(limits) == ["odd20", "q20", "phase10", "y_asphere10", "x_asphere10", "xy65", "zernike37", "zernike45"])
    }

    @Test func shapeParametersAndOrders() throws {
        var s = try extended()
        let trace = ParamPath.shapeParam(surface: SurfaceID(5), parameter: .traceOrder)
        #expect(s.value(at: trace) == -1)
        let ok4 = s.setValue(0.5, at: trace)
        #expect(!ok4)
        let ok5 = s.setValue(3e9, at: trace)
        #expect(!ok5)
        #expect(trace.valueRejection(0.5) == "[invalid_diffraction_order] expected an i32")
        let ok6 = s.setValue(-2, at: trace)
        #expect(ok6)
        #expect(s.value(at: trace) == -2)
        // Order zero is representable for the design order; validation rejects it.
        let ok7 = s.setValue(0, at: .shapeParam(surface: SurfaceID(5), parameter: .designOrder))
        #expect(ok7)
        #expect(s.structuralIssues().map(\.message).contains { $0.hasPrefix("[invalid_diffraction_order] surfaces[5].shape.phase.design_order") })
        let ok8 = s.setValue(4.5, at: .shapeParam(surface: SurfaceID(3), parameter: .normRadius))
        #expect(ok8)
        #expect(s.surfaces[3].shape.normRadius == 4.5)
        let ok9 = s.setValue(0.03, at: .shapeParam(surface: SurfaceID(8), parameter: .curvatureX))
        #expect(ok9)
        let ok10 = s.setValue(-2, at: .shapeParam(surface: SurfaceID(8), parameter: .conicX))
        #expect(ok10)
        #expect(s.surfaces[8].shape.shapeParameter(.curvatureX) == 0.03)
        #expect(s.surfaces[8].shape.shapeParameter(.conicX) == -2)
        #expect(
            ParamPath.shapeParam(surface: SurfaceID(7), parameter: .conicX).unresolvedReason(in: s)
                == "scalar does not belong to shape")
        #expect(s.surfaces[7].shape.shapeParameters == [.curvatureX, .curvatureY, .conicY])
        #expect(s.surfaces[4].shape.shapeParameters == [.normRadius, .designOrder, .designWavelengthUm, .traceOrder])
    }

    @Test func legacyPathsAddressTheDiffractiveBase() throws {
        var s = try extended()
        // Plane base: curvature promotes it to a sphere; it has no conic.
        #expect(s.value(at: .curvature(surface: SurfaceID(4))) == 0)
        #expect(ParamPath.conic(surface: SurfaceID(4)).unresolvedReason(in: s) == "surface has no conic")
        let ok11 = s.setValue(-1, at: .conic(surface: SurfaceID(4)))
        #expect(!ok11)
        let ok12 = s.setValue(0.01, at: .curvature(surface: SurfaceID(4)))
        #expect(ok12)
        guard case .diffractive(let base, let phase) = s.surfaces[4].shape else { Issue.record("not a DOE"); return }
        #expect(base == .conic(curvature: 0.01, conic: 0) && phase.coefficients == [-20, 0.5])
        let ok13 = s.setValue(-0.5, at: .conic(surface: SurfaceID(4)))
        #expect(ok13)
        #expect(s.surfaces[4].shape.conicConstant == -0.5)
        // Even-asphere base.
        let a = ParamPath.asphere(surface: SurfaceID(6), index: 3)
        #expect(s.value(at: a) == 0)
        let ok14 = s.setValue(1e-12, at: a)
        #expect(ok14)
        #expect(s.surfaces[6].shape.asphereCoefficients == [1e-6, -2e-9, 0, 1e-12])
        #expect(ParamPath.asphere(surface: SurfaceID(5), index: 0).unresolvedReason(in: s) == "surface is not an even asphere")
        // Curvature on a toroid or biconic is its y section; QBfs has no conic.
        #expect(s.value(at: .curvature(surface: SurfaceID(7))) == s.surfaces[7].shape.shapeParameter(.curvatureY))
        #expect(s.value(at: .curvature(surface: SurfaceID(8))) == s.surfaces[8].shape.shapeParameter(.curvatureY))
        #expect(ParamPath.conic(surface: SurfaceID(3)).unresolvedReason(in: s) == "surface has no conic")
        let ok15 = s.setValue(0.03, at: .curvature(surface: SurfaceID(3)))
        #expect(ok15)
        #expect(s.surfaces[3].shape == .qBfs(curvature: 0.03, normRadius: 8, coefficients: [0.0005]))
        // The engine's curvature()/conic(): y values of a toroid / biconic.
        #expect(s.surfaces[7].shape.curvature == 0.01 && s.surfaces[7].shape.conicConstant == -0.3)
        #expect(s.surfaces[8].shape.curvature == -0.01 && s.surfaces[8].shape.conicConstant == 0.1)
    }

    @Test func labels() throws {
        let s = try extended()
        let id = SurfaceID(3)
        #expect(ParamPath.shapeCoefficient(surface: id, set: .odd, index: 2).label(in: s) == "Odd β3 on 3")
        #expect(ParamPath.shapeCoefficient(surface: id, set: .q, index: 2).label(in: s) == "Q2 on 3")
        #expect(ParamPath.shapeCoefficient(surface: id, set: .phase, index: 0).label(in: s) == "Phase A1 on 3")
        #expect(ParamPath.shapeCoefficient(surface: id, set: .xy, index: 6).label(in: s) == "C(x²y) on 3")
        #expect(ParamPath.shapeCoefficient(surface: id, set: .zernike, index: 6).label(in: s) == "Z7 on 3")
        #expect(ParamPath.shapeCoefficient(surface: id, set: .yAsphere, index: 1).label(in: s) == "A6y on 3")
        #expect(ParamPath.shapeParam(surface: id, parameter: .curvatureX).label(in: s) == "R_x on 3")
        #expect(ParamPath.shapeParam(surface: id, parameter: .normRadius).label(in: s) == "Norm radius on 3")
        #expect(ParamPath.shapeParam(surface: id, parameter: .designOrder).label(in: s) == "Design order on 3")
    }

    // MARK: Editing

    @Test func shapeKindConversionsKeepWhatCarriesOver() throws {
        var s = SampleSystems.cookeTriplet()
        s.surfaces[1].semiDiameter = .fixed(7.5)
        let c = s.surfaces[1].shape.curvature
        s.setShapeKind(.qCon, at: 1)
        #expect(s.surfaces[1].shape == .qCon(curvature: c, conic: 0, normRadius: 7.5, coefficients: []))
        s.surfaces[1].shape.setCoefficients([0.001], in: .q)
        s.surfaces[1].semiDiameter = .auto
        // Between normalised kinds the radius stays.
        s.setShapeKind(.zernikeSag, at: 1)
        #expect(s.surfaces[1].shape == .zernikeSag(curvature: c, conic: 0, normRadius: 7.5, ordering: .fringe, coefficients: []))
        s.setShapeKind(.qBfs, at: 1)
        #expect(s.surfaces[1].shape == .qBfs(curvature: c, normRadius: 7.5, coefficients: []))
        s.setShapeKind(.toroid, at: 1)
        #expect(s.surfaces[1].shape == .toroid(curvatureX: c, curvatureY: c, conicY: 0, yAsphere: []))
        s.surfaces[1].shape.setShapeParameter(.curvatureX, 0.5)
        s.surfaces[1].shape.setCoefficients([1e-6], in: .yAsphere)
        s.setShapeKind(.biconic, at: 1)
        #expect(
            s.surfaces[1].shape
                == .biconic(curvatureX: 0.5, curvatureY: c, conicX: 0, conicY: 0, xAsphere: [], yAsphere: [1e-6]))
        s.setShapeKind(.conic, at: 1)
        #expect(s.surfaces[1].shape == .conic(curvature: c, conic: 0))
        s.surfaces[1].shape.setConic(-0.5)
        s.setShapeKind(.biconic, at: 1)
        #expect(
            s.surfaces[1].shape == .biconic(curvatureX: c, curvatureY: c, conicX: -0.5, conicY: -0.5, xAsphere: [], yAsphere: []))
        s.setShapeKind(.xyPolynomial, at: 1)
        // No fixed semi-diameter or aperture: 10 mm.
        #expect(s.surfaces[1].shape == .xyPolynomial(curvature: c, conic: -0.5, normRadius: 10, coefficients: []))
        s.setShapeKind(.oddAsphere, at: 1)
        #expect(s.surfaces[1].shape == .oddAsphere(curvature: c, conic: -0.5, coefficients: []))
        #expect(s.structuralIssues().isEmpty)
    }

    @Test func diffractiveWrapsAndUnwrapsItsBase() throws {
        var s = SampleSystems.cookeTriplet()
        s.wavelengths.primary = 0
        let um = s.wavelengths.wavelengths[0].um
        s.surfaces[2].aperture = .rectangular(halfX: 3, halfY: 4)
        s.surfaces[2].shape = .evenAsphere(curvature: -0.01, conic: -1, coefficients: [1e-6])
        s.setShapeKind(.diffractive, at: 2)
        let phase = DiffractivePhase(normRadius: 5, coefficients: [], designOrder: 1, designWavelengthUm: um, traceOrder: 1)
        #expect(s.surfaces[2].shape == .diffractive(base: .evenAsphere(curvature: -0.01, conic: -1, coefficients: [1e-6]), phase: phase))
        // Same kind: unchanged.
        s.surfaces[2].shape.setCoefficients([-3], in: .phase)
        let doe = s.surfaces[2].shape
        s.setShapeKind(.diffractive, at: 2)
        #expect(s.surfaces[2].shape == doe)
        // Converting a DOE uses its base.
        s.setShapeKind(.evenAsphere, at: 2)
        #expect(s.surfaces[2].shape == .evenAsphere(curvature: -0.01, conic: -1, coefficients: [1e-6]))
        // An unsupported base becomes a conic with the old c and k.
        s.surfaces[2].shape = .qCon(curvature: 0.02, conic: -0.5, normRadius: 6, coefficients: [0.1])
        s.setShapeKind(.diffractive, at: 2)
        guard case .diffractive(let base, let p) = s.surfaces[2].shape else { Issue.record("not a DOE"); return }
        #expect(base == .conic(curvature: 0.02, conic: -0.5) && p.normRadius == 6)
        // A plane stays a plane base.
        s.surfaces[3].shape = .plane
        s.setShapeKind(.diffractive, at: 3)
        if case .diffractive(let b, _) = s.surfaces[3].shape { #expect(b == .plane) } else { Issue.record("not a DOE") }
        #expect(s.structuralIssues().isEmpty)
    }

    @Test func shapeChangePrunesOverridesThatNoLongerResolve() {
        var s = SampleSystems.cookeTriplet()
        let id = s.surfaces[1].id
        s.configurations = [
            Configuration(overrides: [
                Pair(.curvature(surface: id), 0.03), Pair(.conic(surface: id), -1), Pair(.thickness(surface: id), 3),
            ])
        ]
        s.setShapeKind(.qBfs, at: 1)
        #expect(s.configurations[0].overrides.map(\.first) == [.curvature(surface: id), .thickness(surface: id)])
        s.setShapeKind(.toroid, at: 1)
        #expect(s.configurations[0].overrides.map(\.first) == [.curvature(surface: id), .thickness(surface: id)])
        s.configurations[0].overrides.append(Pair(.shapeParam(surface: id, parameter: .curvatureX), 0.01))
        s.setShapeKind(.biconic, at: 1)
        #expect(s.configurations[0].overrides.count == 3)
        s.setShapeKind(.conic, at: 1)
        #expect(s.configurations[0].overrides.map(\.first) == [.curvature(surface: id), .thickness(surface: id)])
    }

    @Test func cylinderPreset() {
        let y = SurfaceShape.cylinder(curvature: -0.02, axis: .y)
        #expect(y == .biconic(curvatureX: 0, curvatureY: -0.02, conicX: 0, conicY: 0, xAsphere: [], yAsphere: []))
        #expect(y.cylinderAxis == .y && y.isCylinder)
        #expect(SurfaceShape.cylinder(curvature: 0.1, axis: .x).cylinderAxis == .x)
        #expect(SurfaceShape.biconic(curvatureX: 0.1, curvatureY: 0.1, conicX: 0, conicY: 0, xAsphere: [], yAsphere: []).cylinderAxis == nil)
        #expect(SurfaceShape.biconic(curvatureX: 0.1, curvatureY: 0, conicX: 0, conicY: 0, xAsphere: [], yAsphere: [1e-6]).cylinderAxis == nil)
        #expect(SurfaceShape.cylinder(curvature: 0, axis: .x).cylinderAxis == nil)
        #expect(SurfaceShape.toroid(curvatureX: 0.1, curvatureY: 0, conicY: 0, yAsphere: []).isCylinder == false)
        #expect(SurfaceShape.Kind.allCases.map(\.typeName).count == Set(SurfaceShape.Kind.allCases.map(\.typeName)).count)
        #expect(SurfaceShape.Kind.biconic.title == "Biconic" && SurfaceShape.Kind.qBfs.title == "Q-Type (Best-Fit Sphere)")
    }

    // MARK: Structural validation

    @Test func structuralRulesOfExtendedShapes() throws {
        var s = try extended()
        func messages(_ edit: (inout OpticalSystem) -> Void) -> [String] {
            var t = s
            edit(&t)
            return t.structuralIssues().filter { $0.severity == .error }.map(\.message)
        }
        #expect(messages { (t: inout OpticalSystem) in t.surfaces[3].shape.setShapeParameter(.normRadius, 0) } == [
            "[invalid_norm_radius] surfaces[3].shape.norm_radius: must be > 0 (got 0.0)"
        ])
        #expect(messages { (t: inout OpticalSystem) in t.surfaces[1].shape.setCoefficients(Array(repeating: 0, count: 21), in: .odd) } == [
            "[coefficient_count] surfaces[1].shape.coefficients: at most 20 coefficients"
        ])
        #expect(messages { (t: inout OpticalSystem) in t.surfaces[12].shape.setCoefficients(Array(repeating: 0, count: 38), in: .zernike) } == [
            "[coefficient_count] surfaces[12].shape.coefficients: at most 37 coefficients"
        ])
        #expect(messages { (t: inout OpticalSystem) in t.surfaces[13].shape.setCoefficients(Array(repeating: 0, count: 45), in: .zernike) }.isEmpty)
        #expect(messages { (t: inout OpticalSystem) in t.surfaces[8].shape.setCoefficients(Array(repeating: 0, count: 11), in: .xAsphere) } == [
            "[coefficient_count] surfaces[8].shape.x_asphere: at most 10 coefficients"
        ])
        #expect(messages { (t: inout OpticalSystem) in t.surfaces[5].shape.setShapeParameter(.designWavelengthUm, -0.5) } == [
            "[invalid_design_wavelength] surfaces[5].shape.phase.design_wavelength_um: must be > 0 (got -0.5)"
        ])
        #expect(messages { (t: inout OpticalSystem) in t.surfaces[5].shape.setShapeParameter(.designOrder, 0) } == [
            "[invalid_diffraction_order] surfaces[5].shape.phase.design_order: design order must be nonzero"
        ])
        // Trace order zero is ordinary refraction.
        #expect(messages { (t: inout OpticalSystem) in t.surfaces[5].shape.setShapeParameter(.traceOrder, 0) }.isEmpty)
        #expect(messages { (t: inout OpticalSystem) in
            t.surfaces[4].shape = .diffractive(
                base: .toroid(curvatureX: 0, curvatureY: 0, conicY: 0, yAsphere: []),
                phase: DiffractivePhase(normRadius: 1, designWavelengthUm: 0.5))
        } == ["[invalid_diffractive_base] surfaces[4].shape.base: expected plane, conic or even_asphere"])
        // A non-finite value is reported once, in the JSON check's words.
        let nan = messages { (t: inout OpticalSystem) in t.surfaces[7].shape.setShapeParameter(.conicY, .nan) }
        #expect(nan == ["surfaces[7].shape.conic_y = nan is not a finite number"])
        let inBase = messages { (t: inout OpticalSystem) in t.surfaces[6].shape.setAsphereCoefficient(.infinity, at: 0) }
        #expect(inBase == ["surfaces[6].shape.base.coefficients[0] = inf is not a finite number"], "\(inBase)")
        s.surfaces[2].shape = .paraxial(focalLength: 0)
        #expect(s.structuralIssues().map(\.message) == [
            "[invalid_focal_length] surfaces[2].shape.focal_length: an ideal lens needs a non-zero focal length"
        ])
    }

    /// The astigmatism axis is a shape parameter of the sag shape's overlay,
    /// as in the engine, and survives fringe edits.
    @Test func overlayAxisIsAShapeParameter() {
        let conic = SurfaceShape.conic(curvature: 0.02, conic: 0)
        #expect(conic.shapeParameter(.astigmatismAngleDeg) == nil)
        var overlay = SurfaceShape.sagOverlay(
            base: conic, normRadius: 5, testWavelengthUm: 0.6328, astigmatism: 0.5, spherical: 0)
        #expect(overlay.shapeParameters == [.astigmatismAngleDeg])
        let setAngle = overlay.setShapeParameter(.astigmatismAngleDeg, 30)
        let setFringes = overlay.setIrregularity(.spherical, 0.25, clearRadius: nil, testWavelengthUm: 0.6328)
        #expect(setAngle && setFringes)
        #expect(
            overlay
                == .sagOverlay(
                    base: conic, normRadius: 5, testWavelengthUm: 0.6328, astigmatism: 0.5, astigmatismAngleDeg: 30,
                    spherical: 0.25))
        var doe = SurfaceShape.diffractive(base: overlay, phase: DiffractivePhase(normRadius: 5, designWavelengthUm: 0.55))
        let setDOE = doe.setShapeParameter(.astigmatismAngleDeg, 45)
        #expect(setDOE && doe.shapeParameter(.astigmatismAngleDeg) == 45)
    }
}
