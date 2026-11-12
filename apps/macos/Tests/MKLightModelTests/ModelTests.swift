import Foundation
import Testing

@testable import MKLightModel

@Suite struct DisplayTests {
    @Test func radiusCurvatureConversion() {
        #expect(RadiusDisplay.curvature(fromRadius: 50) == 0.02)
        #expect(RadiusDisplay.curvature(fromRadius: .infinity) == 0)
        #expect(RadiusDisplay.curvature(fromRadius: -.infinity) == 0)
        #expect(RadiusDisplay.curvature(fromRadius: 0) == 0)
        #expect(RadiusDisplay.radius(fromCurvature: 0) == .infinity)
        #expect(RadiusDisplay.radius(fromCurvature: -0.04) == -25)
    }

    @Test func settingRadiusPromotesPlane() {
        var s = Surface(id: SurfaceID(1), shape: .plane, thickness: 1)
        s.radius = 40
        #expect(s.shape == .conic(curvature: 1 / 40.0, conic: 0))
        s.radius = .infinity
        #expect(s.shape == .conic(curvature: 0, conic: 0))
        #expect(s.radius == .infinity)
        var p = Surface(id: SurfaceID(2), shape: .paraxial(focalLength: 10))
        p.radius = 5
        #expect(p.shape == .paraxial(focalLength: 10), "paraxial surfaces have no curvature")
    }

    @Test func extendedDoubleParsing() {
        #expect(ExtendedDouble.parse("inf") == .infinity)
        #expect(ExtendedDouble.parse(" +Infinity ") == .infinity)
        #expect(ExtendedDouble.parse("-inf") == -.infinity)
        #expect(ExtendedDouble.parse("2.5e-3") == 0.0025)
        #expect(ExtendedDouble.parse("abc") == nil)
    }

    @Test func numberFormatting() {
        #expect(NumberDisplay.format(.infinity) == "Infinity")
        #expect(NumberDisplay.format(12.3456789) == "12.3457")
        #expect(NumberDisplay.format(1.5e-5) == "1.5000e-05")
        #expect(NumberDisplay.exact(0.1) == "0.1")
    }

    @Test func materialDisplayNames() {
        #expect(MaterialRef.air.displayName == "AIR")
        #expect(MaterialRef.glass("N-BK7").displayName == "N-BK7")
        #expect(MaterialRef.model(nd: 1.5168, vd: 64.17, dpgf: 0).displayName == "1.51680/64.17")
    }
}

@Suite struct EditingTests {
    @Test func insertBeforeStopShiftsStop() {
        var s = SampleSystems.cookeTriplet()
        let id = s.insertSurface(at: 2)
        #expect(s.surfaces.count == 10)
        #expect(s.surfaces[2].id == id)
        #expect(s.stop == 6)
        #expect(s.surfaces[s.stop].label == "STOP")
        #expect(id == SurfaceID(9), "fresh ids never collide")
        #expect(s.structuralIssues().isEmpty)
    }

    @Test func insertAfterStopKeepsStop() {
        var s = SampleSystems.cookeTriplet()
        s.insertSurface(at: 7)
        #expect(s.stop == 5)
    }

    @Test func insertClampsToValidRange() {
        var s = OpticalSystem.newEmpty()
        s.insertSurface(at: 0)
        #expect(s.surfaces[0].thickness == .infinity, "object stays first")
        s.insertSurface(at: 99)
        #expect(s.surfaces.last?.id == SurfaceID(2), "image stays last")
    }

    @Test func deleteKeepsStopConsistent() {
        var s = SampleSystems.cookeTriplet()
        let first = s.deleteSurface(at: 1)
        #expect(first)
        #expect(s.stop == 4 && s.surfaces[4].label == "STOP")
        let second = s.deleteSurface(at: 4)
        #expect(second, "deleting the stop moves the stop to its neighbour")
        #expect(s.stop == 4)
        let object = s.deleteSurface(at: 0), image = s.deleteSurface(at: s.imageIndex)
        #expect(!object && !image, "object and image are permanent")
        #expect(s.structuralIssues().isEmpty)
    }

    @Test func cannotDeleteLastInnerSurface() {
        var s = OpticalSystem.newEmpty()
        let deleted = s.deleteSurface(at: 1)
        #expect(!deleted)
        #expect(s.surfaces.count == 3)
    }

    @Test func deleteDropsDanglingOverrides() {
        var s = SampleSystems.cookeTriplet()
        s.configurations = [Configuration(name: "c", overrides: [Pair(.thickness(surface: SurfaceID(3)), 1.0), Pair(.temperatureC, 30.0)])]
        s.deleteSurface(at: 3)
        #expect(s.configurations[0].overrides == [Pair(.temperatureC, 30.0)])
    }

    @Test func shapeKindChangeKeepsCurvature() {
        var s = SampleSystems.cookeTriplet()
        let c = s.surfaces[1].shape.curvature
        s.setShapeKind(.evenAsphere, at: 1)
        #expect(s.surfaces[1].shape == .evenAsphere(curvature: c, conic: 0, coefficients: []))
        s.setShapeKind(.coordinateBreak, at: 1)
        #expect(s.surfaces[1].placement == Placement())
    }
}

/// One model and the engine's verdict on it (`validation_cases.json`,
/// written by dump_fixtures from `OpticalSystem::validate`).
private struct EngineVerdict: Decodable {
    var name: String
    var system: OpticalSystem
    /// `nil` when the engine accepts the system, else its error message.
    var error: String?

    static func all() throws -> [EngineVerdict] {
        try MKLightJSON.decode([EngineVerdict].self, from: Fixtures.data("validation_cases"))
    }
}

@Suite struct ValidationTests {
    @Test func detectsEngineValidationErrors() {
        var s = OpticalSystem.newEmpty()
        s.stop = 2
        s.surfaces[1].thickness = .infinity
        s.surfaces[1].coating = "missing"
        s.fields.kind = .objectHeight
        s.surfaces[2].interaction = .reflect
        let messages = s.structuralIssues().map(\.message)
        #expect(messages.contains { $0.contains("stop index 2") })
        #expect(messages.contains { $0.contains("infinite thickness") })
        #expect(messages.contains { $0.contains("unknown coating") })
        #expect(messages.contains { $0.contains("finite object distance") })
        #expect(messages.contains { $0.contains("cannot be a mirror") })
    }

    @Test func detectsDanglingConfigurationOverride() {
        var s = OpticalSystem.newEmpty()
        s.configurations = [Configuration(overrides: [Pair(.fieldY(index: 3), 1)])]
        #expect(s.structuralIssues().count == 1)
    }

    /// Engine verdicts on parameter paths that do or do not fit the surface's
    /// shape, material or semi-diameter mode (the cases with a configuration
    /// override). Swift must agree case by case.
    @Test func agreesWithEngineOnParameterPaths() throws {
        let cases = try EngineVerdict.all().filter { !$0.system.configurations.isEmpty }
        #expect(cases.count >= 70 && cases.contains { $0.error == nil } && cases.contains { $0.error != nil })
        for c in cases {
            let errors = c.system.structuralIssues().filter { $0.severity == .error }
            let override = c.system.configurations[0].overrides[0]
            let path = override.first
            if let engine = c.error {
                #expect(errors.count == 1, "\(c.name): engine rejects (\(engine)), Swift found \(errors)")
                let why = try #require(
                    path.unresolvedReason(in: c.system) ?? path.valueRejection(override.second), "\(c.name)")
                #expect(engine.trimmingCharacters(in: CharacterSet(charactersIn: "'")).hasSuffix(why), "\(c.name): \(engine) vs \(why)")
                // Reading and writing agree with the engine's get/set (whose
                // semi-diameter setter fixes an automatic one).
                if case .semiDiameter = path {} else {
                    var edited = c.system
                    let set = edited.setValue(override.second, at: path)
                    #expect(!set, "\(c.name): Swift set what the engine refuses")
                    #expect(edited == c.system)
                }
            } else {
                #expect(errors.isEmpty, "\(c.name): engine accepts, Swift found \(errors)")
                #expect(c.system.value(at: path) != nil, "\(c.name)")
                var edited = c.system
                let set = edited.setValue(override.second, at: path)
                #expect(set, "\(c.name): Swift refuses what the engine sets")
                #expect(edited.value(at: path) == override.second, "\(c.name)")
            }
        }
    }

    /// Engine verdicts on the structural rules (the cases without a
    /// configuration, each with at most one problem). Swift must accept what
    /// the engine accepts and report the engine's one problem as an error,
    /// with the engine's code and path where its message carries them.
    @Test func agreesWithEngineOnStructuralRules() throws {
        let cases = try EngineVerdict.all().filter { $0.system.configurations.isEmpty }
        #expect(cases.filter { $0.error == nil }.count >= 15 && cases.filter { $0.error != nil }.count >= 60)
        var located = 0
        for c in cases {
            let issues = c.system.structuralIssues()
            #expect(issues.allSatisfy { $0.severity == .error }, "\(c.name)")
            guard let engine = c.error else {
                #expect(issues.isEmpty, "\(c.name): engine accepts, Swift found \(issues)")
                continue
            }
            #expect(issues.count == 1, "\(c.name): engine rejects (\(engine)), Swift found \(issues)")
            // "invalid model: [code] path: message".
            guard let message = issues.first?.message, message.hasPrefix("["), let start = engine.firstIndex(of: "["),
                let end = engine[start...].range(of: ": ")
            else { continue }
            #expect(message.hasPrefix(engine[start..<end.upperBound]), "\(c.name): \(engine) vs \(message)")
            located += 1
        }
        #expect(located >= 50)
    }

    private func messages(_ edit: (inout OpticalSystem) -> Void) -> [String] {
        var s = SampleSystems.cookeTriplet()
        edit(&s)
        let issues = s.structuralIssues()
        #expect(issues.allSatisfy { $0.severity == .error })
        return issues.map(\.message)
    }

    @Test func evenAsphereHoldsTwentyCoefficients() {
        func asphere(_ terms: Int) -> SurfaceShape {
            .evenAsphere(curvature: 0.01, conic: 0, coefficients: Array(repeating: 1e-30, count: terms))
        }
        #expect(messages { $0.surfaces[2].shape = asphere(20) }.isEmpty)
        #expect(messages { $0.surfaces[2].shape = asphere(21) } == [
            "[coefficient_count] surfaces[2].shape.coefficients: at most 20 coefficients"
        ])
        // The parameter path and its setter stop at the same term.
        var s = SampleSystems.cookeTriplet()
        s.surfaces[2].shape = asphere(1)
        let id = s.surfaces[2].id
        let last = s.setValue(1e-40, at: .asphere(surface: id, index: 19))
        #expect(last && s.surfaces[2].shape.asphereCoefficients.count == 20 && s.structuralIssues().isEmpty)
        let beyond = ParamPath.asphere(surface: id, index: 20)
        #expect(beyond.unresolvedReason(in: s) == "[coefficient_count] index exceeds shape limit")
        #expect(s.value(at: beyond) == nil)
        let before = s
        let set = s.setValue(1, at: beyond)
        #expect(!set && s == before)
    }

    @Test func wavelengthTablesIncreaseStrictly() {
        func coating(_ table: [SpectralSample]) -> [String] {
            messages { $0.coatings["T"] = .tabulated(reflectance: table) }
        }
        #expect(coating([Pair(0.4, 0.01), Pair(0.55, 0.03), Pair(0.7, 0.02)]).isEmpty)
        #expect(coating([Pair(0.55, 0.03)]).isEmpty)
        #expect(coating([]) == ["[invalid_coating] coatings[\"T\"].reflectance: a tabulated coating needs at least one sample"])
        #expect(coating([Pair(0.4, 0.01), Pair(0.7, 0.02), Pair(0.55, 0.03)]) == [
            "[invalid_coating] coatings[\"T\"].reflectance[2][0]: wavelengths must increase (0.55 follows 0.7)"
        ])
        #expect(coating([Pair(0.4, 0.01), Pair(0.4, 0.02), Pair(0.55, 0.03)]) == [
            "[invalid_coating] coatings[\"T\"].reflectance[1][0]: wavelengths must increase (0.4 follows 0.4)"
        ])
        func sensor(_ response: [SpectralSample]) -> [String] {
            messages { $0.sensor = Sensor(pixelsX: 10, pixelsY: 10, pitchX: 0.004, pitchY: 0.004, spectralResponse: response) }
        }
        #expect(sensor([]).isEmpty && sensor([Pair(0.45, 0.6), Pair(0.55, 1), Pair(0.65, 0.7)]).isEmpty)
        #expect(sensor([Pair(0.55, 1), Pair(0.45, 0.6), Pair(0.65, 0.7)]) == [
            "[invalid_sensor] sensor.spectral_response[1][0]: wavelengths must increase (0.45 follows 0.55)"
        ])
        #expect(sensor([Pair(0.45, 0.6), Pair(0.55, 1), Pair(0.55, 0.7)]) == [
            "[invalid_sensor] sensor.spectral_response[2][0]: wavelengths must increase (0.55 follows 0.55)"
        ])
    }

    @Test func modelGlassIndexIsAboveOne() {
        for nd in [1, 0.5, 0, -1.5] {
            #expect(messages { $0.surfaces[1].material = .model(nd: nd, vd: 50, dpgf: 0) } == [
                "[invalid_material] surfaces[1].material.nd: must be > 1 (got \(nd))"
            ])
        }
        #expect(messages { $0.surfaces[1].material = .model(nd: 1.0001, vd: 50, dpgf: 0) }.isEmpty)
        // Also the material of a coating layer.
        let layer = CoatingLayer(material: .model(nd: 1, vd: 50, dpgf: 0), quarterWaves: 1)
        #expect(messages { $0.coatings["AR"] = .thinFilm(referenceWavelengthUm: 0.55, layers: [layer], mirrorSubstrate: nil) } == [
            "[invalid_material] coatings[\"AR\"].layers[0].material.nd: must be > 1 (got 1.0)"
        ])
    }

    @Test func detectsValuesJSONCannotCarry() throws {
        var s = SampleSystems.cookeTriplet()
        #expect(s.structuralIssues().isEmpty)
        s.surfaces[1].shape.setCurvature(.infinity)
        let issues = s.structuralIssues()
        #expect(issues.count == 1 && issues[0].surface == 1 && issues[0].message.contains("curvature"))
        s = SampleSystems.cookeTriplet()
        s.fields.fields[1].y = .nan
        #expect(s.structuralIssues().contains { $0.message.contains("fields") })
        s = SampleSystems.cookeTriplet()
        s.surfaces[0].thickness = -.infinity
        #expect(s.structuralIssues().isEmpty, "±∞ object distance has a JSON form")
    }

    /// The engine rejects a part attached to a surface that does not exist.
    @Test func detectsMechanicsOnMissingSurfaces() {
        var s = SampleSystems.cookeTriplet()
        s.mechanics = [
            .annularBaffle(surface: SurfaceID(42), z: 0, innerRadius: 1, outerRadius: 2, finish: .lambertian(tis: 0.1)),
            .barrel(from: SurfaceID(1), to: SurfaceID(43), innerRadius: 9, finish: .lambertian(tis: 0.1)),
        ]
        let issues = s.structuralIssues()
        #expect(issues.map(\.severity) == [.error, .error])
        #expect(issues.map(\.message) == [
            "[missing_surface] mechanics[0].surface: no surface with id 42",
            "[missing_surface] mechanics[1].to: no surface with id 43",
        ])
    }
}

@Suite struct ReferenceMaintenanceTests {
    let baffle = MechanicalPart.annularBaffle(surface: SurfaceID(3), z: 0, innerRadius: 1, outerRadius: 2, finish: .lambertian(tis: 0.1))
    let barrel = MechanicalPart.barrel(from: SurfaceID(1), to: SurfaceID(7), innerRadius: 9, finish: .lambertian(tis: 0.1))

    @Test func deleteRemovesMechanicsOnTheSurface() {
        var s = SampleSystems.cookeTriplet()
        s.mechanics = [baffle, barrel]
        s.deleteSurface(at: 3)
        #expect(s.mechanics == [barrel])
        #expect(s.structuralIssues().isEmpty)
    }

    @Test func removingFieldsAndWavelengthsRenumbersOverrides() {
        var s = SampleSystems.cookeTriplet()
        s.configurations = [Configuration(overrides: [
            Pair(.fieldY(index: 2), 18), Pair(.fieldX(index: 0), 1), Pair(.fieldWeight(index: 1), 2),
            Pair(.wavelengthUm(index: 2), 0.6), Pair(.wavelengthWeight(index: 0), 3),
        ])]
        let field2 = s.fields.fields[2]
        let removedField = s.removeField(at: 0)
        #expect(removedField)
        #expect(s.fields.fields[1] == field2)
        #expect(s.configurations[0].overrides.map(\.first) == [.fieldY(index: 1), .fieldWeight(index: 0), .wavelengthUm(index: 2), .wavelengthWeight(index: 0)])
        let primary = s.wavelengths.wavelengths[s.wavelengths.primary]
        let removedWavelength = s.removeWavelength(at: 0)
        #expect(removedWavelength)
        #expect(s.wavelengths.wavelengths[s.wavelengths.primary] == primary)
        #expect(s.configurations[0].overrides.map(\.first) == [.fieldY(index: 1), .fieldWeight(index: 0), .wavelengthUm(index: 1)])
        #expect(s.structuralIssues().isEmpty)
        // Removing the primary wavelength moves the primary to a neighbour.
        s.wavelengths.primary = s.wavelengths.wavelengths.count - 1
        let removedPrimary = s.removeWavelength(at: s.wavelengths.primary)
        #expect(removedPrimary && s.wavelengths.wavelengths.indices.contains(s.wavelengths.primary))
        let removedLast = s.removeWavelength(at: 0)
        #expect(!removedLast, "the last wavelength stays")
    }

    @Test func shapeChangeDropsOverridesItInvalidates() {
        var s = SampleSystems.cookeTriplet()
        let id = s.surfaces[1].id
        s.configurations = [Configuration(overrides: [Pair(.conic(surface: id), -1), Pair(.thickness(surface: id), 3)])]
        s.setShapeKind(.plane, at: 1)
        #expect(s.configurations[0].overrides == [Pair(.thickness(surface: id), 3)])
        #expect(s.structuralIssues().isEmpty)
    }

    @Test func editsDropOnlyReferencesTheyInvalidate() {
        var before = SampleSystems.cookeTriplet()
        before.surfaces[1].semiDiameter = .fixed(8)
        let broken = Pair(ParamPath.modelNd(surface: SurfaceID(1)), 1.6)  // already invalid: catalog glass
        before.configurations = [Configuration(overrides: [Pair(.semiDiameter(surface: SurfaceID(1)), 9), broken])]
        var after = before
        after.surfaces[1].semiDiameter = .auto
        after.dropReferencesInvalidated(since: before)
        #expect(after.configurations[0].overrides == [broken], "pre-existing problems stay visible")
        #expect(after.structuralIssues().filter { $0.severity == .error }.count == 1)
    }
}

@Suite struct ProjectFileTests {
    @Test func envelopeRoundTrip() throws {
        let p = ProjectFile(system: SampleSystems.cookeTriplet())
        let back = try ProjectFile.decode(p.encoded())
        #expect(back == p)
        let tree = try JSONValue.parse(p.encoded())
        #expect(tree["format"] == .string("mklight-project"))
        #expect(tree["system"]?["surfaces"] != nil)
    }

    @Test func rejectsOtherSchemaVersions() throws {
        let json = #"{"format":"mklight-project","schema_version":99,"system":{}}"#
        #expect(throws: ProjectFile.Error.unsupportedVersion(99)) { try ProjectFile.decode(Data(json.utf8)) }
        let older = #"{"format":"mklight-project","schema_version":0,"system":{}}"#
        #expect(throws: ProjectFile.Error.unsupportedVersion(0)) { try ProjectFile.decode(Data(older.utf8)) }
        let foreign = #"{"format":"something-else","schema_version":1,"system":{}}"#
        #expect(throws: ProjectFile.Error.self) { try ProjectFile.decode(Data(foreign.utf8)) }
        // A bare optical system has no schema version.
        #expect(throws: ProjectFile.Error.notAProject("it has no schema_version")) {
            try ProjectFile.decode(Fixtures.data("kitchen_sink"))
        }
    }

    /// Sections the app does not edit (views, merit, metadata, unknown keys)
    /// survive a round trip unchanged.
    @Test func preservesUnknownSections() throws {
        var tree = try JSONValue.parse(ProjectFile(system: SampleSystems.singlet()).encoded())
        guard case .object(var o) = tree else { Issue.record("not an object"); return }
        o["views"] = .array([.object(["key": .string("w1"), "analysis": .string("system_summary"), "settings": .object([:]), "ui": .object(["zoom": .number(1.5)])])])
        o["future_key"] = .object(["x": .integer(3)])
        o["catalogs"] = .array([.object(["name": .string("MYCAT"), "path": .string("/tmp/my.agf"), "relative_path": .null, "format": .string("agf")])])
        tree = .object(o)
        var p = try ProjectFile.decode(MKLightJSON.encode(tree))
        #expect(p.catalogs == [CatalogReference(name: "MYCAT", path: "/tmp/my.agf")])
        let back = try JSONValue.parse(p.encoded())
        #expect(back.differences(from: tree).isEmpty, "\(back.differences(from: tree))")
        p.addCatalog(CatalogReference(name: "MYCAT", path: "/other/my.agf"))
        p.addCatalog(CatalogReference(name: "SECOND", path: "/x/second.json", format: "json"))
        #expect(p.catalogs.map(\.name) == ["MYCAT", "SECOND"])
        #expect(p.catalogs[0].path == "/other/my.agf")
        p.stampMetadata(appName: "MKLight", appVersion: "1.0", engineVersion: "0.1.0")
        #expect(p.sections["metadata"]?["app_name"] == .string("MKLight"))
        #expect(p.sections["metadata"]?["created"] != nil)
    }

    @Test func encodingIsDeterministic() throws {
        let a = try MKLightJSON.encode(SampleSystems.cookeTriplet())
        let b = try MKLightJSON.encode(SampleSystems.cookeTriplet())
        #expect(a == b)
    }
}

@Suite struct JSONValueTests {
    @Test func numbersCompareByValue() throws {
        let a = try JSONValue.parse(Data(#"{"a": 10.0, "b": [1, 2.5]}"#.utf8))
        let b = try JSONValue.parse(Data(#"{"b": [1.0, 2.5], "a": 10}"#.utf8))
        #expect(a == b)
        #expect(a.differences(from: b).isEmpty)
    }

    @Test func reportsDifferences() throws {
        let a = try JSONValue.parse(Data(#"{"a": 1, "b": null}"#.utf8))
        let b = try JSONValue.parse(Data(#"{"a": 2}"#.utf8))
        #expect(a.differences(from: b) == ["$.a: integer(1) vs integer(2)", "$.b: missing on right"])
    }

    /// Philox seeds are full u64 values (CONVENTIONS §6); a `Double` would
    /// round them above 2^53 and make different seeds compare equal.
    @Test func integersStayExact() throws {
        let big: UInt64 = 12_345_678_901_234_567_891
        let v = try JSONValue.parse(Data("{\"seed\": \(big)}".utf8))
        #expect(v["seed"] == .unsigned(big))
        #expect(String(decoding: try MKLightJSON.encode(v), as: UTF8.self) == "{\"seed\":\(big)}")
        let a = try JSONValue.parse(Data("\((1 << 60) as UInt64)".utf8))
        let b = try JSONValue.parse(Data("\((1 << 60) + 1 as UInt64)".utf8))
        #expect(a != b)
        #expect(try JSONValue.parse(Data("9007199254740993".utf8)) != .number(9_007_199_254_740_992))
        // Mixed representations of the same value are equal and hash alike.
        #expect(JSONValue.integer(6) == .number(6) && JSONValue.integer(6).hashValue == JSONValue.number(6).hashValue)
        #expect(JSONValue.unsigned(1 << 63) == .number(0x1p63))
        #expect(Set<JSONValue>([.integer(-2), .number(-2), .number(2.5)]).count == 2)
    }
}

@Suite struct StrictDecodingTests {
    /// serde rejects these; the mirror must too, or a file the app accepts
    /// would be refused by the engine.
    @Test func rejectsWhatSerdeRejects() {
        #expect(throws: DecodingError.self) {
            try MKLightJSON.decode(ParamPath.self, from: Data(#"{"param":"field_x","index":-1}"#.utf8))
        }
        #expect(throws: DecodingError.self) {
            try MKLightJSON.decode(Configuration.self, from: Data(#"{"name":null,"overrides":[]}"#.utf8))
        }
        #expect(throws: DecodingError.self) {
            try MKLightJSON.decode(WavelengthSet.self, from: Data(#"{"wavelengths":[{"um":0.55}],"primary":-1}"#.utf8))
        }
        var json = try! JSONValue.from(SampleSystems.singlet())
        if case .object(var o) = json, case .array(var surfaces) = o["surfaces"], case .object(var s1) = surfaces[1] {
            s1["thickness"] = .string("nan")
            surfaces[1] = .object(s1)
            o["surfaces"] = .array(surfaces)
            json = .object(o)
        }
        #expect(throws: DecodingError.self) { try MKLightJSON.decode(OpticalSystem.self, from: MKLightJSON.encode(json)) }
        #expect(ExtendedDouble.parse("NaN") == nil && ExtendedDouble.parse("-inf") == -.infinity)
    }

    /// The engine's model types deny unknown fields. The mirror must refuse
    /// them too: it re-encodes every snapshot and save, so an unknown key it
    /// accepted would vanish silently. Every object of the engine-written
    /// fixtures (every model variant) gets an extra key in turn.
    @Test(arguments: [
        ("kitchen_sink", "system"), ("extended_shapes", "system"), ("variants", "systems"), ("solved_triplet", "system"),
        ("param_paths", "paths"), ("solve_kinds", "solves"),
    ])
    func rejectsUnknownKeysEverywhere(fixture: String, kind: String) throws {
        let original = try JSONValue.parse(Fixtures.data(fixture))
        func decode(_ json: JSONValue) throws {
            let data = try MKLightJSON.encode(json)
            switch kind {
            case "system": _ = try MKLightJSON.decode(OpticalSystem.self, from: data)
            case "systems": _ = try MKLightJSON.decode([OpticalSystem].self, from: data)
            case "paths": _ = try MKLightJSON.decode([ParamPath].self, from: data)
            default: _ = try MKLightJSON.decode([Solve].self, from: data)
            }
        }
        try decode(original)
        let paths = Self.objectPaths(in: original)
        #expect(paths.count > 3)
        for path in paths {
            let tampered = Self.inserting("future_field", into: original, at: path[...])
            do {
                try decode(tampered)
                Issue.record("\(fixture): an unknown key at $\(path.map { ".\($0)" }.joined()) was accepted")
            } catch let DecodingError.dataCorrupted(context) {
                #expect(context.debugDescription.contains("unknown field 'future_field'"), "\(context.debugDescription)")
            }
        }
    }

    /// Paths (object keys and array indices) of every JSON object, except
    /// name-keyed maps (`coatings`), whose keys are data.
    private static func objectPaths(in json: JSONValue, at path: [String] = []) -> [[String]] {
        switch json {
        case .object(let o):
            let here = path.last == "coatings" ? [] : [path]
            return here + o.keys.sorted().flatMap { objectPaths(in: o[$0]!, at: path + [$0]) }
        case .array(let a):
            return a.indices.flatMap { objectPaths(in: a[$0], at: path + [String($0)]) }
        default: return []
        }
    }

    private static func inserting(_ key: String, into json: JSONValue, at path: ArraySlice<String>) -> JSONValue {
        guard let head = path.first else {
            guard case .object(var o) = json else { return json }
            o[key] = .integer(1)
            return .object(o)
        }
        switch json {
        case .object(var o):
            o[head] = inserting(key, into: o[head]!, at: path.dropFirst())
            return .object(o)
        case .array(var a):
            let i = Int(head)!
            a[i] = inserting(key, into: a[i], at: path.dropFirst())
            return .array(a)
        default: return json
        }
    }
}

/// Applying a `convert_spec` result to the field kind and aperture type.
@Suite struct SpecConversionTests {
    @Test func fieldValuesReplaceTheNumbersAndKeepWeights() throws {
        var s = SampleSystems.cookeTriplet()
        s.fields.fields[1].weight = 2
        let json = try JSONValue.parse(Data(#"{"fields": {"kind": "paraxial_image_height", "values": [[0, 0], [0, 3.5], [0, 7]]}, "fields_error": null, "aperture": null, "aperture_error": null, "stop_semi_diameter": 4.2}"#.utf8))
        let c = try #require(SpecConversion(json: json))
        #expect(c.stopSemiDiameter == 4.2 && c.aperture == nil)
        var converted = s
        converted.setFieldKind(.paraxialImageHeight, conversion: c)
        #expect(converted.fields.fields.map(\.y) == [0, 3.5, 7] && converted.fields.fields[1].weight == 2)
        // A result for another field count is not applied.
        var other = s
        other.fields.fields.removeLast()
        let kept = other.fields.fields
        other.setFieldKind(.paraxialImageHeight, conversion: c)
        #expect(other.fields.kind == .paraxialImageHeight && other.fields.fields == kept)
    }

    @Test func floatByStopSizeFixesOnlyAnAutomaticStop() {
        var s = SampleSystems.cookeTriplet()
        s.setApertureKind(.floatByStopSize, conversion: SpecConversion(stopSemiDiameter: 4.2))
        #expect(s.aperture == .floatByStopSize && s.surfaces[s.stop].semiDiameter == .fixed(4.2))
        var f = SampleSystems.cookeTriplet()
        f.surfaces[f.stop].semiDiameter = .fixed(3)
        f.setApertureKind(.floatByStopSize, conversion: SpecConversion(stopSemiDiameter: 4.2))
        #expect(f.surfaces[f.stop].semiDiameter == .fixed(3), "a fixed stop is the user's")
        var e = SampleSystems.cookeTriplet()
        e.setApertureKind(.imageSpaceFNumber, conversion: SpecConversion(aperture: .imageSpaceFNumber(4.5)))
        #expect(e.aperture == .imageSpaceFNumber(4.5))
    }
}
