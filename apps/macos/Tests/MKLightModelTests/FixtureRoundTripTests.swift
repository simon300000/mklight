import Foundation
import Testing

@testable import MKLightModel

/// The fixtures are written by the engine itself
/// (`cargo run -p mklight-core --example dump_fixtures`). Decoding them and
/// re-encoding must reproduce the same JSON tree, which proves the Swift
/// mirror matches the engine's serde representation field for field.
enum Fixtures {
    static func data(_ name: String) throws -> Data {
        let url = try #require(Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Fixtures"))
        return try Data(contentsOf: url)
    }

    /// Decode → encode → compare semantically with the original.
    static func roundTrip<T: Codable & Equatable>(_ type: T.Type, _ name: String) throws -> T {
        try roundTrip(type, name, data: data(name))
    }

    /// The same for `original`, the text of an engine-written file `name`.
    static func roundTrip<T: Codable & Equatable>(_ type: T.Type, _ name: String, data original: Data) throws -> T {
        let value = try MKLightJSON.decode(T.self, from: original)
        let reencoded = try MKLightJSON.encode(value)
        let diff = try JSONValue.parse(original).differences(from: JSONValue.parse(reencoded))
        #expect(diff.isEmpty, "\(name) differs after round trip:\n\(diff.joined(separator: "\n"))")
        // A second decode gives back the same value.
        #expect(try MKLightJSON.decode(T.self, from: reencoded) == value)
        return value
    }
}

@Suite struct FixtureRoundTripTests {
    @Test(arguments: ["empty", "cooke_triplet", "kitchen_sink", "extended_shapes"])
    func systemRoundTrip(name: String) throws {
        let s = try Fixtures.roundTrip(OpticalSystem.self, name)
        #expect(s.structuralIssues().isEmpty, "engine-valid fixture must be structurally valid: \(s.structuralIssues())")
    }

    @Test func variantsRoundTrip() throws {
        let systems = try Fixtures.roundTrip([OpticalSystem].self, "variants")
        #expect(
            systems.map(\.aperture.kind) == [
                .imageSpaceFNumber, .paraxialWorkingFNumber, .floatByStopSize, .entrancePupilDiameter, .entrancePupilDiameter,
            ])
        #expect(systems.map(\.fields.kind) == [.paraxialImageHeight, .realImageHeight, .objectAngle, .objectAngle, .objectAngle])
        #expect(systems[1].rayAiming == .real(tolerance: 1e-10))
        #expect(systems[3].surfaces[0].thickness == -.infinity)
        #expect(systems[3].surfaces[2].thickness == .infinity)
        // A sag overlay keeps its base's kind; the element placement round-trips.
        var moved = systems[4]
        let id = moved.surfaces[1].id
        #expect(moved.surfaces[1].shape.typeName == "sag_overlay" && moved.surfaces[1].shape.kind == .conic)
        #expect(moved.surfaces[1].elementPlacement == ElementPlacement(last: id, motion: Placement(decenterY: 0.02, tiltX: 0.1)))
        #expect(moved.value(at: .irregularity(surface: id, mode: .spherical)) == -0.25)
        #expect(moved.value(at: .curvature(surface: id)) == 0.02)
        let fringes = moved.setValue(1, at: .irregularity(surface: id, mode: .astigmatism))
        let tilt = moved.setValue(0.3, at: .elementPlacement(surface: id, last: id, pivot: nil, component: .tiltY))
        let otherPivot = moved.setValue(0.3, at: .elementPlacement(surface: id, last: id, pivot: id, component: .tiltY))
        #expect(fringes && moved.value(at: .irregularity(surface: id, mode: .astigmatism)) == 1)
        #expect(tilt && moved.surfaces[1].elementPlacement?.motion.tiltY == 0.3)
        #expect(!otherPivot, "one range and pivot per first surface")
    }

    @Test func paramPathsRoundTrip() throws {
        let paths = try Fixtures.roundTrip([ParamPath].self, "param_paths")
        #expect(paths.count == 27 + CoefficientSet.allCases.count + ShapeParameter.allCases.count)
        #expect(paths.contains(.irregularity(surface: SurfaceID(7), mode: .spherical)))
        #expect(paths.contains(.elementPlacement(surface: SurfaceID(7), last: SurfaceID(9), pivot: nil, component: .decenterX)))
        #expect(Set(paths).count == paths.count, "every variant decodes to a distinct case")
        #expect(paths[3] == .asphere(surface: SurfaceID(7), index: 2))
        #expect(paths.contains(.apertureValue) && paths.contains(.pressureAtm))
        for set in CoefficientSet.allCases {
            #expect(paths.contains(.shapeCoefficient(surface: SurfaceID(7), set: set, index: 2)), "\(set)")
        }
        for p in ShapeParameter.allCases {
            #expect(paths.contains(.shapeParam(surface: SurfaceID(7), parameter: p)), "\(p)")
        }
    }

    @Test func emptyMatchesSwiftNewEmpty() throws {
        let rust = try MKLightJSON.decode(OpticalSystem.self, from: Fixtures.data("empty"))
        #expect(rust == OpticalSystem.newEmpty())
    }

    @Test func cookeMatchesSample() throws {
        let rust = try MKLightJSON.decode(OpticalSystem.self, from: Fixtures.data("cooke_triplet"))
        #expect(rust == SampleSystems.cookeTriplet())
    }

    @Test func kitchenSinkCoversEveryVariant() throws {
        let s = try MKLightJSON.decode(OpticalSystem.self, from: Fixtures.data("kitchen_sink"))
        // The extended shapes are in extended_shapes.json.
        #expect(Set(s.surfaces.map(\.shape.kind)) == Set(SurfaceShape.Kind.allCases.filter(\.isLegacy)))
        #expect(Set(s.surfaces.map(\.interaction)) == Set(Interaction.allCases))
        let materials = s.surfaces.map(\.material)
        #expect(materials.contains(.air) && materials.contains(.vacuum))
        #expect(materials.contains(.catalog(catalog: "BUILTIN", name: "N-BK7", offset: MaterialOffset(dnD: 0.0012, dvD: -0.8))))
        #expect(materials.contains(.model(nd: 1.6, vd: 40, dpgf: 0.002)))
        #expect(materials.contains(.constant(n: 1.5)))
        let apertures = s.surfaces.compactMap(\.aperture)
        #expect(apertures.count == 4)
        #expect(s.surfaces[1].semiDiameter == .fixed(12.5))
        #expect(s.surfaces[1].edge.edgeScatter == .gaussian(tis: 0.02, sigmaSlope: 0.1))
        #expect(s.surfaces[5].placement?.order == .tiltThenDecenter)
        #expect(s.surfaces[6].placement?.restoreAfter == true)
        #expect(s.coatings["Silver"] == .thinFilm(referenceWavelengthUm: 0.55, layers: [CoatingLayer(material: .constant(n: 1.38), thicknessUm: 0.1)], mirrorSubstrate: Pair(0.12, 3.4)))
        if case .tabulated(let table) = s.coatings["Measured"] {
            #expect(table.map(\.um) == [0.4, 0.55, 0.7])
        } else {
            Issue.record("tabulated coating missing")
        }
        #expect(s.mechanics.count == 2)
        #expect(s.sensor?.spectralResponse.count == 3)
        #expect(s.sensor?.halfWidth == 0.5 * 4000 * 0.00376)
        #expect(s.configurations[0].overrides[0] == Pair(ParamPath.thickness(surface: SurfaceID(0)), 80.0))
        #expect(s.environment == EnvironmentConditions(temperatureC: 35.5, pressureAtm: 0.75))
        #expect(s.aperture == .objectSpaceNA(0.05))
    }

    @Test func missingDefaultedFieldsDecode() throws {
        // Fields the engine marks `#[serde(default)]` may be absent.
        let json = """
            {"aperture":{"type":"float_by_stop_size"},"fields":{"kind":"object_angle","fields":[{"x":0,"y":0}]},
             "wavelengths":{"wavelengths":[{"um":0.55}],"primary":0},
             "surfaces":[{"id":0,"shape":{"type":"plane"},"thickness":"Infinity","material":{"type":"air"}},
                         {"id":1,"shape":{"type":"conic","curvature":0.01},"thickness":5,"material":{"type":"model","nd":1.5,"vd":60}},
                         {"id":2,"shape":{"type":"plane"},"thickness":0,"material":{"type":"air"}}],
             "stop":1,"ray_aiming":{"mode":"real"}}
            """
        let s = try MKLightJSON.decode(OpticalSystem.self, from: Data(json.utf8))
        #expect(s.name == "" && s.coatings.isEmpty && s.sensor == nil)
        #expect(s.surfaces[0].thickness == .infinity)
        #expect(s.surfaces[1].shape == .conic(curvature: 0.01, conic: 0))
        #expect(s.surfaces[1].material == .model(nd: 1.5, vd: 60, dpgf: 0))
        #expect(s.fields.fields[0].weight == 1)
        #expect(s.wavelengths.wavelengths[0].weight == 1)
        #expect(s.rayAiming == .real(tolerance: RayAiming.defaultTolerance))
        #expect(s.environment == EnvironmentConditions())
    }

    /// `snapshot_hashes.json` holds the engine's BLAKE3 hashes of the system
    /// fixtures. Only an engine binding can reproduce them (the mock hashes
    /// with SHA-256), so the FFI client's tests compare against it; here we
    /// check that it covers exactly the valid system fixtures.
    @Test func snapshotHashFixtureCoversSystemFixtures() throws {
        let hashes = try MKLightJSON.decode([String: String].self, from: Fixtures.data("snapshot_hashes"))
        #expect(
            Set(hashes.keys) == [
                "empty.json", "cooke_triplet.json", "kitchen_sink.json", "solved_triplet.json", "extended_shapes.json",
            ])
        for (file, hex) in hashes {
            #expect(hex.count == 64 && hex.allSatisfy { $0.isHexDigit && !$0.isUppercase })
            _ = try MKLightJSON.decode(OpticalSystem.self, from: Fixtures.data(String(file.dropLast(5))))
        }
        #expect(Set(hashes.values).count == hashes.count)
    }

    @Test func unknownTagIsRejected() {
        let json = #"{"type":"freeform","curvature":0}"#
        #expect(throws: DecodingError.self) { try MKLightJSON.decode(SurfaceShape.self, from: Data(json.utf8)) }
    }
}
