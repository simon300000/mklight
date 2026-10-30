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
    /// shape, material or semi-diameter mode (written by dump_fixtures from
    /// `OpticalSystem::validate`). Swift must agree case by case.
    @Test func agreesWithEngineOnParameterPaths() throws {
        struct Case: Decodable {
            var name: String
            var system: OpticalSystem
            var error: String?
        }
        let cases = try MKLightJSON.decode([Case].self, from: Fixtures.data("validation_cases"))
        #expect(cases.count >= 20 && cases.contains { $0.error == nil } && cases.contains { $0.error != nil })
        for c in cases {
            let errors = c.system.structuralIssues().filter { $0.severity == .error }
            if let engine = c.error {
                #expect(errors.count == 1, "\(c.name): engine rejects (\(engine)), Swift found \(errors)")
                let why = try #require(c.system.configurations[0].overrides[0].first.unresolvedReason(in: c.system))
                #expect(engine.trimmingCharacters(in: CharacterSet(charactersIn: "'")).hasSuffix(why), "\(c.name): \(engine) vs \(why)")
            } else {
                #expect(errors.isEmpty, "\(c.name): engine accepts, Swift found \(errors)")
            }
        }
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

    @Test func detectsMechanicsOnMissingSurfaces() {
        var s = SampleSystems.cookeTriplet()
        s.mechanics = [.annularBaffle(surface: SurfaceID(42), z: 0, innerRadius: 1, outerRadius: 2, finish: .lambertian(tis: 0.1))]
        let issues = s.structuralIssues()
        #expect(issues.count == 1 && issues[0].severity == .warning)
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

    @Test func acceptsBareSystemJSON() throws {
        let p = try ProjectFile.decode(Fixtures.data("kitchen_sink"))
        #expect(p.system.name == "Kitchen sink")
        #expect(p.schemaVersion == ProjectFile.currentVersion)
        #expect(p.catalogs.isEmpty)
    }

    @Test func rejectsNewerSchemaVersion() {
        let json = #"{"format":"mklight-project","schema_version":99,"system":{}}"#
        #expect(throws: ProjectFile.Error.unsupportedVersion(99)) { try ProjectFile.decode(Data(json.utf8)) }
        let foreign = #"{"format":"something-else","schema_version":1,"system":{}}"#
        #expect(throws: ProjectFile.Error.self) { try ProjectFile.decode(Data(foreign.utf8)) }
    }

    /// Files from the app's first envelope (`"version": 1`, no
    /// `schema_version`) migrate like the engine's version-0 drafts.
    @Test func readsEarlyAppEnvelope() throws {
        let early = try JSONValue.object([
            "format": .string("mklight-project"), "version": .integer(1),
            "system": JSONValue.from(SampleSystems.singlet()),
        ])
        let p = try ProjectFile.decode(MKLightJSON.encode(early))
        #expect(p.system == SampleSystems.singlet())
        #expect(p.sections["version"] == nil)
        let tree = try JSONValue.parse(p.encoded())
        #expect(tree["schema_version"] == .integer(1))
        #expect(tree["catalogs"] == .array([]) && tree["merit"] == .null)
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
}
