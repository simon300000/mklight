import Foundation
import Testing
import UniformTypeIdentifiers

@testable import MKLightEngine
@testable import MKLightModel
@testable import MKLightUI

/// An engine whose lens-file import returns a fixed result.
private struct ImportingEngine: EngineClient {
    let result: PrescriptionImport
    private let base = MockEngine(simulatedLatency: .zero)
    var info: EngineInfo { base.info }
    func openSnapshot(_ system: OpticalSystem) async throws -> ModelSnapshot { try await base.openSnapshot(system) }
    func validate(_ system: OpticalSystem) async -> [ModelIssue] { await base.validate(system) }
    func submit(_ request: AnalysisRequest, on snapshot: ModelSnapshot) -> AnalysisJob { base.submit(request, on: snapshot) }
    func materialNames() async -> [String] { await base.materialNames() }
    func importPrescription(_ data: Data, format: PrescriptionFormat) throws -> PrescriptionImport {
        var r = result
        r.format = format
        return r
    }
}

private func warning(_ kind: ImportWarning.Kind, _ message: String, line: Int? = nil, surface: Int? = nil) -> ImportWarning {
    ImportWarning(kind: kind, line: line, surface: surface, message: message)
}

private func sampleImport(_ warnings: [ImportWarning], name: String = "") -> PrescriptionImport {
    var system = OpticalSystem.newEmpty()
    system.name = name
    return PrescriptionImport(format: .zmx, encoding: "utf16_le", system: system, warnings: warnings)
}

/// Serialized: two tests swap the process-wide import engine.
@Suite(.serialized) struct ImportReportTests {
    @Test func groupsByMeaningMostSeriousFirst() {
        let r = ImportReport(
            sampleImport([
                warning(.approximated, "ray aiming settings are not imported", line: 11),
                warning(.unknownGlass, "glass X is not in the loaded catalogs", line: 51, surface: 1),
                warning(.unsupportedRecord, "tolerance data is not imported", line: 34),
                warning(.validation, "surfaces[2].thickness: bad"),
                warning(.unknownGlass, "glass Y is not in the loaded catalogs", line: 60, surface: 3),
                warning(.nonSequential, "non-sequential data is not imported", line: 70),
            ]),
            fileName: "lens.zmx")
        #expect(r.groups.map(\.category) == [.problems, .glasses, .surfaces, .dropped, .assumed])
        #expect(r.groups[1].warnings.map(\.line) == [51, 60])
        // Row identities are unique across groups.
        let ids = r.groups.flatMap { $0.items.map(\.id) }
        #expect(Set(ids).count == r.warnings.count)
        #expect(r.hasUnknownGlasses)
        #expect(r.summary == "1 surface · .zmx · UTF-16")
        #expect(r.headline.hasPrefix("6 items could not be imported exactly."))
        #expect(ImportReport.sentence(r.warnings[1]) == "Glass X is not in the loaded catalogs")
        #expect(ImportReport.location(r.warnings[1]) == "Surface 1 · Line 51")
        #expect(ImportReport.location(r.warnings[3]) == nil)
        let text = r.plainText
        #expect(text.hasPrefix("Imported lens.zmx (1 surface · .zmx · UTF-16)"))
        #expect(text.contains("Glasses:\n- Glass X is not in the loaded catalogs [Surface 1 · Line 51]"))
        #expect(!ImportReport(sampleImport([warning(.defaulted, "x")]), fileName: "a.seq").hasUnknownGlasses)
    }

    @Test func documentTypes() {
        let readable = ProjectDocument.readableContentTypes
        #expect(readable.contains(.zmxPrescription) && readable.contains(.seqPrescription))
        #expect(!ProjectDocument.writableContentTypes.contains(.zmxPrescription))
        #expect(PrescriptionFormat(contentType: .zmxPrescription, fileName: nil) == .zmx)
        #expect(PrescriptionFormat(contentType: .data, fileName: "Lens.SEQ") == .seq)
        #expect(PrescriptionFormat(contentType: .mklightProject, fileName: "a.mklight") == nil)
        for name in ["a.mklight", "b.json", "c.zmx", "d.SEQ"] {
            #expect(ProjectDocument.canOpen(URL(fileURLWithPath: "/tmp/" + name)))
        }
        #expect(!ProjectDocument.canOpen(URL(fileURLWithPath: "/tmp/e.agf")))
    }

    @MainActor @Test func importedDesignKeepsItsReportOnce() throws {
        ProjectDocument.setImportEngine(ImportingEngine(result: sampleImport([warning(.unknownGlass, "glass X …")])))
        defer { ProjectDocument.setImportEngine(MockEngine(simulatedLatency: .zero)) }
        let design = try ProjectDocument.importDesign(Data("SURF 0".utf8), format: .seq, fileName: "Triplet.seq")
        // An unnamed system takes the file's name.
        #expect(design.system.name == "Triplet")
        let report = try #require(design.takeImportReport())
        #expect(report.fileName == "Triplet.seq" && report.format == .seq && report.warnings.count == 1)
        #expect(design.takeImportReport() == nil)
    }

    @Test func engineWithoutImportRefuses() {
        ProjectDocument.setImportEngine(MockEngine(simulatedLatency: .zero))
        #expect(throws: EngineError.self) {
            try ProjectDocument.importDesign(Data(), format: .zmx, fileName: "x.zmx")
        }
    }
}
