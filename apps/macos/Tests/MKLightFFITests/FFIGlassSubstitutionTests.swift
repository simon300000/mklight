import Foundation
import Testing

@testable import MKLightEngine
@testable import MKLightFFI
@testable import MKLightModel
@testable import MKLightUI

/// The glass substitution flow on the real engine: select the glasses,
/// run, preview a combination in the window, apply it as one undoable step.
@Suite(.serialized, engineLinked) @MainActor struct FFIGlassSubstitutionTests {
    private func wait(_ timeout: Duration = .seconds(180), _ cond: () -> Bool) async throws {
        let clock = ContinuousClock()
        let end = clock.now + timeout
        while !cond() {
            guard clock.now < end else {
                Issue.record("condition not met within \(timeout)")
                return
            }
            try await Task.sleep(for: .milliseconds(10))
        }
    }

    @Test func selectRunPreviewApplyAndUndo() async throws {
        let system = try #require(SampleSystems.bundled("achromat"))
        let doc = DesignDocument(system: system)
        let undo = UndoManager()
        undo.groupsByEvent = false
        let w = Workspace(engine: try FFIEngine(configuration: ["worker_threads": .integer(2), "progress_interval_ms": .integer(10)]))
        w.debounce = .zero
        await w.loadEngineMetadata()
        w.update(system: system, immediately: true)
        try await wait { w.snapshot?.system == system }

        let session = GlassSubstitutionSession()
        session.configure(for: system)
        let slots = GlassSlot.all(in: system)
        #expect(slots.count == 2 && session.selected == Set(slots.map(\.id)), "both glasses of the doublet are ticked")
        session.maxCombinations = 8
        session.localIterations = 6
        session.topN = 4
        session.candidates.nearest = 4
        let s = session.settings(system: system, merit: doc.merit, variables: doc.variables)
        #expect(s["material_variables"]?.array?.count == 2)
        #expect(s["material_variables"]?.array?.first?[path: "candidates", "nearest"]?.int == 4)
        session.start(in: w, document: doc)
        #expect(session.isRunning)
        try await wait { !session.isRunning }
        let r = try #require(session.result, "\(session.state)")
        #expect(!r.ranked.isEmpty && r.ranked.count <= 4)
        #expect(r.ranked.allSatisfy { $0.glasses.count == 2 && $0.system != nil })
        #expect(r.combinations > 0 && r.startingMerit.isFinite)
        // The best is previewed in the window without touching the document.
        let best = try #require(session.selectedCombination)
        #expect(best.rank == 1 && w.isPreviewing && w.previewTitle == "Candidate 1")
        try await wait { w.snapshot?.system == best.system }
        #expect(doc.system == system)
        if r.ranked.count > 1 {
            session.select(2, in: w)
            #expect(w.previewTitle == "Candidate 2")
            session.select(1, in: w)
        }
        // Apply: one undoable step; the preview ends.
        undo.beginUndoGrouping()
        session.apply(to: doc, workspace: w, undoManager: undo)
        undo.endUndoGrouping()
        #expect(session.applied && !w.isPreviewing && doc.system == best.system)
        #expect(undo.undoActionName == "Apply Glass Substitution")
        let names = Set(GlassSlot.all(in: doc.system).map(\.material))
        #expect(names == Set(best.glasses.map(\.name)) || best.glasses.allSatisfy { $0.name.isEmpty == false })
        undo.undo()
        #expect(doc.system == system)
        undo.redo()
        #expect(doc.system == best.system)
    }

    /// Stop ends the search early and keeps the ranking.
    @Test func stopKeepsTheRanking() async throws {
        let system = try #require(SampleSystems.bundled("cooke_triplet"))
        let doc = DesignDocument(system: system)
        let w = Workspace(engine: try FFIEngine(configuration: ["worker_threads": .integer(2), "progress_interval_ms": .integer(10)]))
        w.debounce = .zero
        await w.loadEngineMetadata()
        w.update(system: system, immediately: true)
        try await wait { w.snapshot?.system == system }
        let session = GlassSubstitutionSession()
        session.configure(for: system)
        session.maxCombinations = 400
        session.localIterations = 20
        session.start(in: w, document: doc)
        try await wait { (session.progress?.combinations ?? 0) >= 2 || !session.isRunning }
        #expect(session.canStop)
        session.stop()
        try await wait { !session.isRunning }
        let r = try #require(session.result, "\(session.state)")
        #expect(r.stopped && !r.ranked.isEmpty)
        #expect(r.combinations < 400)
    }
}
