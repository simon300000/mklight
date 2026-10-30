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
        #expect(p.version == ProjectFile.currentVersion)
    }

    @Test func rejectsNewerFormat() {
        let json = #"{"format":"mklight-project","version":99,"system":{}}"#
        #expect(throws: ProjectFile.Error.self) { try ProjectFile.decode(Data(json.utf8)) }
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
        #expect(a.differences(from: b) == ["$.a: number(1.0) vs number(2.0)", "$.b: missing on right"])
    }
}
