import Foundation
import Testing

@testable import MKLightEngine
@testable import MKLightFFI
@testable import MKLightModel
@testable import MKLightUI

/// Switching the field kind or aperture type converts the numbers on the
/// real engine (`convert_spec`), as one undoable step, and converting back
/// returns the original numbers.
@Suite(.serialized, engineLinked) @MainActor struct FFISpecConversionTests {
    private func engine() throws -> FFIEngine {
        try FFIEngine(configuration: ["worker_threads": .integer(2)])
    }

    private func convert(_ e: FFIEngine, _ system: OpticalSystem, field: FieldKind? = nil, aperture: ApertureSpec.Kind? = nil) async throws -> SpecConversion {
        let snap = try await e.openSnapshot(system)
        let settings: [String: JSONValue] = [
            "field_kind": field.map { .string($0.rawValue) } ?? .null,
            "aperture": aperture.map { .string($0.rawValue) } ?? .null,
        ]
        let out = try await e.submit(EngineAnalysisRequest(analysis: "convert_spec", settings: settings), on: snap).result()
        return try #require(SpecConversion(json: out.json))
    }

    private func close(_ a: Double, _ b: Double) -> Bool { abs(a - b) <= 1e-9 * (1 + max(abs(a), abs(b))) }

    @Test func fieldKindSwitchesConvertAndUndoInOneStep() async throws {
        let e = try engine()
        let original = SampleSystems.cookeTriplet()
        #expect(original.fields.kind == .objectAngle && original.fields.fields.count > 1)
        for kind in [FieldKind.paraxialImageHeight, .realImageHeight] {
            let doc = DesignDocument(system: original)
            let undo = UndoManager()
            let c = try await convert(e, original, field: kind)
            let values = try #require(c.fieldValues)
            #expect(values.count == original.fields.fields.count)
            doc.edit("Change Field Type", undoManager: undo) { $0.setFieldKind(kind, conversion: c) }
            #expect(doc.system.fields.kind == kind)
            #expect(doc.system.fields.fields.map(\.weight) == original.fields.fields.map(\.weight))
            #expect(doc.system.fields.fields.last!.y > 1, "an image height in mm, not the old angle")
            // Back to angles gives the original numbers.
            let back = try #require(try await convert(e, doc.system, field: .objectAngle).fieldValues)
            for (v, f) in zip(back, original.fields.fields) {
                #expect(close(v[0], f.x) && close(v[1], f.y), "\(kind): \(v) vs (\(f.x), \(f.y))")
            }
            undo.undo()
            #expect(doc.system == original, "one undo step restores kind and values")
        }
        // Undefined: an object height at infinity keeps the numbers.
        let c = try await convert(e, original, field: .objectHeight)
        #expect(c.fieldValues == nil)
        var s = original
        s.setFieldKind(.objectHeight, conversion: c)
        #expect(s.fields.kind == .objectHeight && s.fields.fields == original.fields.fields)
    }

    @Test func apertureTypeSwitchesKeepTheEntrancePupil() async throws {
        let e = try engine()
        let original = SampleSystems.cookeTriplet()
        guard case .entrancePupilDiameter(let epd) = original.aperture else {
            Issue.record("expected an EPD sample")
            return
        }
        for kind in [ApertureSpec.Kind.imageSpaceFNumber, .paraxialWorkingFNumber, .floatByStopSize] {
            var s = original
            s.setApertureKind(kind, conversion: try await convert(e, original, aperture: kind))
            #expect(s.aperture.kind == kind)
            if kind == .floatByStopSize {
                guard case .fixed = s.surfaces[s.stop].semiDiameter else {
                    Issue.record("float by stop size fixes the automatic stop semi-diameter")
                    continue
                }
            }
            let back = try #require(try await convert(e, s, aperture: .entrancePupilDiameter).aperture?.value)
            #expect(close(back, epd), "\(kind): EPD \(back) vs \(epd)")
        }
        // Undefined: an NA at infinity keeps the number.
        var s = original
        s.setApertureKind(.objectSpaceNA, conversion: try await convert(e, original, aperture: .objectSpaceNA))
        #expect(s.aperture == .objectSpaceNA(epd))
    }
}
