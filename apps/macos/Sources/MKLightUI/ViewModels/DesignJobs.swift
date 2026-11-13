import Foundation
import MKLightEngine
import MKLightModel
import Observation

/// One point of a running optimisation (from the job's stage text
/// `"<algorithm> <iteration>: merit <best>"`).
public struct MeritPoint: Hashable, Sendable {
    public var iteration: Int
    public var merit: Double

    /// Parses the engine's progress stage, e.g. `lm 12: merit 3.4e-4`.
    public static func parse(_ stage: String) -> MeritPoint? {
        let parts = stage.split(separator: ":", maxSplits: 1)
        guard parts.count == 2, let it = parts[0].split(separator: " ").last.flatMap({ Int($0) }) else { return nil }
        let rhs = parts[1].trimmingCharacters(in: .whitespaces)
        guard rhs.hasPrefix("merit"), let m = Double(rhs.dropFirst(5).trimmingCharacters(in: .whitespaces)) else { return nil }
        return MeritPoint(iteration: it, merit: m)
    }
}

/// The state shared by the design-job sheets.
public enum JobState: Equatable, Sendable {
    case setup
    case running(JobProgress)
    case finished
    case failed(String)
}

/// The Optimize sheet: settings, a running job with live merit, Stop (keep
/// the best design so far) and Cancel, and the result applied to the design
/// as one undoable step.
@MainActor
@Observable
public final class OptimizeSession {
    public static let algorithms: [(id: String, title: String)] = [
        ("lm", "Damped least squares (LM)"), ("dogleg", "Trust region (dogleg)"), ("de", "Differential evolution"),
        ("cma_es", "CMA-ES"), ("nelder_mead", "Nelder–Mead"), ("hammer", "Basin hopping"),
    ]
    public var algorithm = "lm"
    /// `nil`: the algorithm's default.
    public var maxIterations: Int?
    public var seed = 0
    /// Use the design's variables (else the engine's automatic set: every
    /// curvature and the back focus).
    public var useDesignVariables = true
    /// The merit function being edited: a working copy of the document's
    /// `merit` section whose operands keep their identity while edited.
    public private(set) var merit = MeritSettings()
    /// The merit and every operand at the current design.
    public private(set) var evaluation: MeritEvaluation?
    public private(set) var evaluating = false
    public private(set) var evaluationError: String?
    @ObservationIgnored private var evaluationJob: AnalysisJob?
    @ObservationIgnored private var evaluationTask: Task<Void, Never>?
    @ObservationIgnored private var evaluatedKey: JSONValue?
    /// The request scheduled or running, until it answers.
    @ObservationIgnored private var pendingKey: JSONValue?

    public private(set) var state = JobState.setup
    public private(set) var trace: [MeritPoint] = []
    public private(set) var result: OptimizeResult?
    public private(set) var applied = false
    @ObservationIgnored private var job: AnalysisJob?

    public init() {}

    public var isRunning: Bool { if case .running = state { true } else { false } }
    public var canStop: Bool { isRunning && (job?.canStop ?? false) }

    public func settings(variables: [DesignVariable], system: OpticalSystem? = nil) -> [String: JSONValue] {
        var s: [String: JSONValue] = ["algorithm": .string(algorithm), "seed": .integer(Int64(seed))]
        s.merge(merit.jobSettings) { _, b in b }
        if let maxIterations { s["max_iterations"] = .integer(Int64(maxIterations)) }
        if let v = Self.variableSettings(variables, useDesign: useDesignVariables, system: system) { s["variables"] = v }
        return s
    }

    /// The `variables` setting: the design's variables without the pinned
    /// ones (equal bounds hold them) and those a solve controls (the engine
    /// applies solves first and refuses them), or `nil` for the engine's
    /// automatic set.
    static func variableSettings(_ variables: [DesignVariable], useDesign: Bool, system: OpticalSystem? = nil) -> JSONValue? {
        let free = variables.filter { !$0.isPinned && !(system?.isSolved($0.param) ?? false) }
        guard useDesign, !free.isEmpty else { return nil }
        return (try? JSONValue.from(free)) ?? .array([])
    }

    /// Why the run cannot start with these variables, if it cannot.
    public func variableProblem(_ variables: [DesignVariable], system: OpticalSystem? = nil) -> String? {
        guard useDesignVariables, !variables.isEmpty else { return nil }
        func solved(_ v: DesignVariable) -> Bool { system?.isSolved(v.param) ?? false }
        if variables.allSatisfy({ $0.isPinned || solved($0) }) {
            if !variables.contains(where: solved) { return "Every variable is pinned. Unpin one, or vary the automatic set." }
            return "Every variable is pinned or controlled by a solve. Add a free variable, or vary the automatic set."
        }
        return nil
    }

    // MARK: Merit function

    /// Takes the document's merit function when it changed outside the
    /// editor (open, undo, redo).
    public func load(_ document: DesignDocument) {
        let m = document.merit
        if m != merit { merit = m }
    }

    /// Applies an edit to the merit function as one undoable step.
    public func editMerit(_ document: DesignDocument, _ actionName: String, undoManager: UndoManager?, _ change: (inout MeritSettings) -> Void) {
        load(document)
        var next = merit
        change(&next)
        merit = next
        document.setMerit(next, actionName: actionName, undoManager: undoManager)
    }

    /// The settings of `merit_evaluation`: `optimize`'s merit, variables and
    /// algorithm (which decides how the EFL is held), evaluated once.
    func evaluationSettings(variables: [DesignVariable], merit m: MeritSettings? = nil, system: OpticalSystem? = nil) -> [String: JSONValue] {
        var s = (m ?? merit).jobSettings
        s["algorithm"] = .string(algorithm)
        if let v = Self.variableSettings(variables, useDesign: useDesignVariables, system: system) { s["variables"] = v }
        return s
    }

    /// Evaluates the merit at the current design off the main thread, after
    /// a short pause that newer edits restart, unless the same design and
    /// merit were evaluated already.
    public func scheduleEvaluation(
        in workspace: Workspace, variables: [DesignVariable], system: OpticalSystem? = nil, delay: Duration = .milliseconds(200)
    ) {
        guard !isRunning, let snap = workspace.designSnapshot ?? workspace.snapshot else { return }
        let s = evaluationSettings(variables: variables, system: system ?? snap.system)
        let key = JSONValue.object(["settings": .object(s), "snapshot": .string(snap.hash.hex)])
        guard key != pendingKey else { return }
        evaluationTask?.cancel()
        guard key != evaluatedKey else {
            // Back to what is shown (an edit undone): drop the newer request.
            evaluationJob?.cancel()
            evaluationJob = nil
            pendingKey = nil
            evaluating = false
            return
        }
        pendingKey = key
        // The shown merit is out of date from now on (the readout dims).
        evaluating = true
        evaluationTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: delay)
            guard let self, !Task.isCancelled else { return }
            await self.evaluate(in: workspace, settings: s, key: key)
        }
    }

    func evaluate(in workspace: Workspace, settings s: [String: JSONValue], key: JSONValue) async {
        evaluationJob?.cancel()
        evaluating = true
        var submitted: AnalysisJob?
        do {
            let job = try workspace.submitJob("merit_evaluation", settings: s, key: "merit-evaluation")
            submitted = job
            evaluationJob = job
            let out = try await job.result()
            guard evaluationJob === job else { return }
            evaluation = MeritEvaluation(json: out.json)
            evaluatedKey = key
            pendingKey = nil
            evaluationError = evaluation == nil ? "The engine returned no merit." : nil
            evaluating = false
        } catch EngineError.cancelled {
            // A newer evaluation replaced it.
        } catch {
            if let submitted, evaluationJob !== submitted { return }
            evaluationError = error.localizedDescription
            evaluatedKey = key
            pendingKey = nil
            evaluating = false
        }
    }

    /// Replaces the default merit by the operands the engine generates for
    /// it (`merit_spec`), so they can be edited (one undoable step). The EFL
    /// hold stays with the "Hold the focal length" switch, which adds it
    /// again, so the custom merit equals the default's.
    public func convertToCustom(in workspace: Workspace, document: DesignDocument, variables: [DesignVariable], undoManager: UndoManager?) async {
        load(document)
        guard let snap = workspace.designSnapshot ?? workspace.snapshot else { return }
        var defaults = merit
        defaults.mode = .default
        let s = evaluationSettings(variables: variables, merit: defaults, system: snap.system)
        let key = JSONValue.object(["settings": .object(s), "snapshot": .string(snap.hash.hex)])
        var e = key == evaluatedKey ? evaluation : nil
        if e == nil {
            evaluating = true
            defer { evaluating = false }
            do {
                let out = try await workspace.submitJob("merit_evaluation", settings: s, key: "merit-conversion").result()
                e = MeritEvaluation(json: out.json)
            } catch {
                evaluationError = error.localizedDescription
                return
            }
        }
        let operands = (e?.specOperands ?? []).filter { !Self.isEFLHold($0) || !defaults.holdEFL }
        guard let e, !operands.isEmpty else {
            evaluationError = "The engine generated no operands for this design."
            return
        }
        editMerit(document, "Convert Merit to Custom", undoManager: undoManager) {
            $0.operands = operands
            $0.specOptions = e.specOptions
            $0.mode = .custom
        }
    }

    /// The EFL hold the engine adds (`hold EFL`, or `hold EFL (penalty)` for
    /// the population methods).
    static func isEFLHold(_ op: MeritOperand) -> Bool { op.type == "efl" && op.label.hasPrefix("hold EFL") }

    /// `fresh` (Try Again) recomputes instead of reusing stored engine results.
    public func start(in workspace: Workspace, variables: [DesignVariable], fresh: Bool = false) {
        evaluationTask?.cancel()
        evaluationJob?.cancel()
        pendingKey = nil
        evaluating = false
        let s = settings(variables: variables, system: workspace.designSnapshot?.system ?? workspace.snapshot?.system)
        do {
            let job = try workspace.submitJob("optimize", settings: s, fresh: fresh)
            begin(job)
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    func begin(_ job: AnalysisJob) {
        self.job = job
        trace = []
        result = nil
        applied = false
        state = .running(JobProgress(fraction: 0, stage: "Queued"))
        Task { @MainActor [weak self] in
            for await p in job.progress {
                guard let self, self.job === job else { return }
                // Progress can still be queued after the result arrived;
                // the result's history is final then.
                guard case .running = self.state else { continue }
                self.state = .running(p)
                if let m = MeritPoint.parse(p.stage), m.iteration != self.trace.last?.iteration { self.trace.append(m) }
            }
        }
        Task { @MainActor [weak self] in
            do {
                let out = try await job.result()
                guard let self, self.job === job else { return }
                self.result = OptimizeResult(json: out.json)
                if let h = self.result?.history, !h.isEmpty { self.trace = h }
                self.state = self.result == nil ? .failed("The engine returned no optimized design.") : .finished
            } catch EngineError.cancelled {
                guard let self, self.job === job else { return }
                self.state = .setup
            } catch {
                guard let self, self.job === job else { return }
                self.state = .failed(error.localizedDescription)
            }
        }
    }

    /// Ends the run early, keeping the best design so far.
    public func stop() { job?.stop() }

    /// Ends the run and discards it.
    public func cancel() {
        job?.cancel()
        state = .setup
    }

    /// Applies the optimised design as one undoable step.
    public func apply(to document: DesignDocument, undoManager: UndoManager?) {
        guard let system = result?.system, !applied else { return }
        document.replace(system: system, actionName: "Apply Optimization", undoManager: undoManager)
        applied = true
    }

    public func reset() {
        state = .setup
        result = nil
        trace = []
        applied = false
    }
}

/// The parts of an `optimize` result the sheet shows.
public struct OptimizeResult: Sendable {
    public struct Variable: Identifiable, Sendable {
        public var id: Int
        public var label: String
        public var param: ParamPath?
        public var start: Double
        public var value: Double
        public var atBound: Bool
    }
    public var status: String
    public var stopped: Bool
    public var improved: Bool
    public var startMerit: Double
    public var finalMerit: Double
    public var iterations: Int
    public var evaluations: Int
    public var feasible: Bool
    public var variables: [Variable]
    public var history: [MeritPoint]
    public var system: OpticalSystem?
    public var evaluatedOn: String

    public init?(json: JSONValue) {
        guard json.object != nil else { return nil }
        status = json["status"]?.string ?? ""
        stopped = json["stopped"]?.bool ?? (status == "stopped")
        improved = json["improved"]?.bool ?? false
        startMerit = json[path: "start", "merit"]?.nf ?? .nan
        finalMerit = json[path: "final", "merit"]?.nf ?? .nan
        feasible = json[path: "final", "feasible"]?.bool ?? true
        iterations = json["iterations"]?.int ?? 0
        evaluations = json["evaluations"]?.int ?? 0
        variables = (json["variables"]?.array ?? []).enumerated().map { k, v in
            Variable(
                id: k, label: v["label"]?.string ?? "",
                param: v["param"].flatMap { try? $0.decode(ParamPath.self) },
                start: v["start"]?.nf ?? .nan, value: v["value"]?.nf ?? .nan, atBound: v["at_bound"]?.bool ?? false)
        }
        history = (json["history"]?.array ?? []).compactMap { h in
            guard let it = h["iteration"]?.int, let m = (h["best_merit"] ?? h["merit"])?.nf, m.isFinite else { return nil }
            return MeritPoint(iteration: it, merit: m)
        }
        system = json["system"].flatMap { try? $0.decode(OpticalSystem.self) }
        evaluatedOn = json[path: "backend", "evaluated_on"]?.string ?? ""
    }

    public var improvement: Double { startMerit > 0 ? 1 - finalMerit / startMerit : .nan }
}

/// Apply Solves: a dry run (the `solves` analysis) shown as a report, then
/// an undoable commit of the solved design.
@MainActor
@Observable
public final class SolvesSession {
    public struct Outcome: Identifiable, Sendable {
        public var id: Int
        public var target: String
        public var kind: String
        public var previous: Double?
        public var value: Double
        public var residual: Double
        public var method: String
    }
    public private(set) var state = JobState.setup
    public private(set) var outcomes: [Outcome] = []
    public private(set) var warnings: [String] = []
    public private(set) var committed = false

    public init() {}

    public var isRunning: Bool { if case .running = state { true } else { false } }
    public var changes: Int { outcomes.filter { $0.previous != $0.value }.count }

    /// `fresh`: recompute rather than reuse a stored result (Try Again).
    public func dryRun(in workspace: Workspace, fresh: Bool = false) async {
        state = .running(JobProgress(fraction: 0, stage: "Solving"))
        committed = false
        do {
            let job = try workspace.submitJob("solves", settings: [:], fresh: fresh)
            let report = try await job.result().json
            read(report, system: workspace.snapshot?.system)
            state = .finished
        } catch EngineError.cancelled {
            // A newer dry run replaced this one and reports instead.
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    func read(_ report: JSONValue, system: OpticalSystem?) {
        outcomes = (report["outcomes"]?.array ?? []).enumerated().map { k, o in
            let path = o["target"].flatMap { try? $0.decode(ParamPath.self) }
            let kind = o["kind"]?.string ?? ""
            // Curvatures are shown as radii, like everywhere else.
            var previous = o["previous"]?.nf, value = o["value"]?.nf ?? .nan
            if case .curvature? = path {
                previous = previous.map(RadiusDisplay.radius(fromCurvature:))
                value = RadiusDisplay.radius(fromCurvature: value)
            }
            return Outcome(
                id: o["index"]?.int ?? k,
                target: path.map { p in system.map { p.label(in: $0) } ?? "\(p)" } ?? "?",
                kind: SolveKind.Kind(rawValue: kind)?.title ?? kind, previous: previous,
                value: value, residual: o["residual"]?.nf ?? .nan, method: o["method"]?.string ?? "")
        }
        warnings = (report["warnings"]?.array ?? []).compactMap(\.string)
    }

    /// Solves in the engine and replaces the design (one undoable step).
    public func commit(in workspace: Workspace, document: DesignDocument, undoManager: UndoManager?) async {
        do {
            let applied = try await workspace.applySolves()
            read(applied.report, system: document.system)
            document.replace(system: applied.snapshot.system, actionName: "Apply Solves", undoManager: undoManager)
            committed = true
            state = .finished
        } catch {
            state = .failed(error.localizedDescription)
        }
    }
}
