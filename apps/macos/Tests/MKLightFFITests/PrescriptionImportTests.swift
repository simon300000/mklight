import Foundation
import Testing

@testable import MKLightEngine
@testable import MKLightFFI
@testable import MKLightModel
@testable import MKLightUI

/// The engine's hand-written lens prescription fixtures.
private let prescriptions = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("crates/mklight-core/tests/fixtures/prescriptions")

@Suite(.serialized, engineLinked) struct FFIPrescriptionImportTests {
    @Test func importsUTF16BytesWithoutASession() async throws {
        let engine = try FFIEngine()
        let data = try Data(contentsOf: prescriptions.appendingPathComponent("cooke_triplet.zmx"))
        #expect(data.prefix(2) == Data([0xFF, 0xFE]))
        let r = try engine.importPrescription(data, format: .zmx)
        #expect(r.format == .zmx && r.encoding == "utf16_le")
        #expect(r.system.name == "Cooke triplet f50 F/4")
        #expect(r.system.surfaces.count == 8 && r.system.stop == 4)
        // The imported system is one the engine accepts.
        _ = try await engine.openSnapshot(r.system)
        #expect(throws: EngineError.self) { try engine.importPrescription(Data("hello".utf8), format: .seq) }
    }

    @Test func reportsWhatDidNotMap() throws {
        let engine = try FFIEngine()
        let data = try Data(contentsOf: prescriptions.appendingPathComponent("awkward.zmx"))
        let report = ImportReport(try engine.importPrescription(data, format: .zmx), fileName: "awkward.zmx")
        #expect(report.hasUnknownGlasses)
        #expect(Set(report.groups.map(\.category)).isSuperset(of: [.glasses, .surfaces, .dropped]))
        #expect(!report.warnings.contains { $0.kind == .other })
    }

    @Test func importsIntoTheDocumentSession() async throws {
        let engine = try FFIEngine().makeDocumentEngine()
        let r = try await engine.importPrescription(at: prescriptions.appendingPathComponent("petzval_asphere.seq"))
        #expect(r.format == .seq)
        #expect(r.system.surfaces[6].shape.kind == .evenAsphere)
        #expect(r.warnings.isEmpty)
    }
}
