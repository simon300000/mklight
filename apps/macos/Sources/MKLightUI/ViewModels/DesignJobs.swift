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

    public private(set) var state = JobState.setup
    public private(set) var trace: [MeritPoint] = []
    public private(set) var result: OptimizeResult?
    public private(set) var applied = false
    @ObservationIgnored private var job: AnalysisJob?
    /// The seed and settings the result was computed with.
    public private(set) var settingsUsed: [String: JSONValue] = [:]

    public init() {}

    public var isRunning: Bool { if case .running = state { true } else { false } }
    public var canStop: Bool { isRunning && (job?.canStop ?? false) }

    public func settings(variables: [DesignVariable]) -> [String: JSONValue] {
        var s: [String: JSONValue] = ["algorithm": .string(algorithm), "seed": .integer(Int64(seed))]
        s.merge(merit.jobSettings) { _, b in b }
        if let maxIterations { s["max_iterations"] = .integer(Int64(maxIterations)) }
        if let v = Self.variableSettings(variables, useDesign: useDesignVariables) { s["variables"] = v }
        return s
    }

    /// The `variables` setting: the design's variables without the pinned
    /// ones (equal bounds hold them), or `nil` for the engine's automatic set.
    static func variableSettings(_ variables: [DesignVariable], useDesign: Bool) -> JSONValue? {
        let free = variables.filter { !$0.isPinned }
        guard useDesign, !free.isEmpty else { return nil }
        return (try? JSONValue.from(free)) ?? .array([])
    }

    /// Why the run cannot start with these variables, if it cannot.
    public func variableProblem(_ variables: [DesignVariable]) -> String? {
        guard useDesignVariables, !variables.isEmpty, variables.allSatisfy(\.isPinned) else { return nil }
        return "Every variable is pinned. Unpin one, or vary the automatic set."
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

    /// The settings of a cheap evaluation: one LM iteration, whose `start`
    /// values are the design's merit and operands (the engine has no
    /// evaluate-only call).
    func evaluationSettings(variables: [DesignVariable]) -> [String: JSONValue] {
        var s = merit.jobSettings
        s["algorithm"] = .string("lm")
        s["max_iterations"] = .integer(1)
        if let v = Self.variableSettings(variables, useDesign: useDesignVariables) { s["variables"] = v }
        return s
    }

    /// Evaluates the merit at the current design, after a short pause, unless
    /// the same design and merit were evaluated already.
    public func scheduleEvaluation(in workspace: Workspace, variables: [DesignVariable], delay: Duration = .milliseconds(350)) {
        guard !isRunning, let snap = workspace.designSnapshot ?? workspace.snapshot else { return }
        let s = evaluationSettings(variables: variables)
        let key = JSONValue.object(["settings": .object(s), "snapshot": .string(snap.hash.hex)])
        guard key != evaluatedKey else { return }
        evaluationTask?.cancel()
        evaluationTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: delay)
            guard let self, !Task.isCancelled else { return }
            await self.evaluate(in: workspace, settings: s, key: key)
        }
    }

    func evaluate(in workspace: Workspace, settings s: [String: JSONValue], key: JSONValue) async {
        evaluationJob?.cancel()
        evaluating = true
        do {
            let job = try workspace.submitJob("optimize", settings: s, key: "merit-evaluation")
            evaluationJob = job
            let out = try await job.result()
            guard evaluationJob === job else { return }
            evaluation = MeritEvaluation(json: out.json)
            evaluatedKey = key
            evaluationError = nil
            evaluating = false
        } catch EngineError.cancelled {
            // A newer evaluation replaced it.
        } catch {
            evaluationError = error.localizedDescription
            evaluatedKey = key
            evaluating = false
        }
    }

    /// Replaces the default merit by the operands the engine generates for
    /// it, so they can be edited (one undoable step).
    public func convertToCustom(in workspace: Workspace, document: DesignDocument, variables: [DesignVariable], undoManager: UndoManager?) async {
        load(document)
        var spec = evaluation?.specOperands ?? []
        if spec.isEmpty || evaluatedKey == nil {
            var defaults = merit
            defaults.mode = .default
            let saved = merit
            merit = defaults
            let s = evaluationSettings(variables: variables)
            merit = saved
            evaluating = true
            defer { evaluating = false }
            do {
                let out = try await workspace.submitJob("optimize", settings: s, key: "merit-evaluation").result()
                spec = MeritEvaluation(json: out.json)?.specOperands ?? []
            } catch {
                evaluationError = error.localizedDescription
                return
            }
        }
        guard !spec.isEmpty else {
            evaluationError = "The engine generated no operands for this design."
            return
        }
        editMerit(document, "Convert Merit to Custom", undoManager: undoManager) {
            $0.operands = spec
            $0.mode = .custom
        }
    }

    public func start(in workspace: Workspace, variables: [DesignVariable]) {
        evaluationTask?.cancel()
        evaluationJob?.cancel()
        evaluating = false
        let s = settings(variables: variables)
        do {
            let job = try workspace.submitJob("optimize", settings: s)
            begin(job, settings: s)
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    func begin(_ job: AnalysisJob, settings: [String: JSONValue]) {
        self.job = job
        settingsUsed = settings
        trace = []
        result = nil
        applied = false
        state = .running(JobProgress(fraction: 0, stage: "Queued"))
        Task { @MainActor [weak self] in
            for await p in job.progress {
                guard let self, self.job === job else { return }
                if case .running = self.state { self.state = .running(p) }
                if let m = MeritPoint.parse(p.stage), m.iteration != self.trace.last?.iteration { self.trace.append(m) }
            }
        }
        Task { @MainActor [weak self] in
            do {
                let out = try await job.result()
                guard let self, self.job === job else { return }
                self.result = OptimizeResult(json: out.json)
                if let h = self.result?.history, !h.isEmpty { self.trace = h }
                self.state = self.result == nil ? .failed("The engine returned no optimised design.") : .finished
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
        document.replace(system: system, actionName: "Optimize", undoManager: undoManager)
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
                param: v["param"].flatMap { try? MKLightJSON.decode(ParamPath.self, from: MKLightJSON.encode($0)) },
                start: v["start"]?.nf ?? .nan, value: v["value"]?.nf ?? .nan, atBound: v["at_bound"]?.bool ?? false)
        }
        history = (json["history"]?.array ?? []).compactMap { h in
            guard let it = h["iteration"]?.int, let m = (h["best_merit"] ?? h["merit"])?.nf, m.isFinite else { return nil }
            return MeritPoint(iteration: it, merit: m)
        }
        system = json["system"].flatMap { try? MKLightJSON.decode(OpticalSystem.self, from: MKLightJSON.encode($0)) }
        evaluatedOn = json[path: "backend", "evaluated_on"]?.string ?? ""
    }

    public var improvement: Double { startMerit > 0 ? 1 - finalMerit / startMerit : .nan }
}

/// The Tolerance sheet: sensitivity table and Monte-Carlo statistics.
@MainActor
@Observable
public final class ToleranceSession {
    public var mode = "both"
    public var trials = 100
    public var seed = 0
    public private(set) var state = JobState.setup
    public private(set) var result: ToleranceResult?
    @ObservationIgnored private var job: AnalysisJob?

    public init() {}

    public var isRunning: Bool { if case .running = state { true } else { false } }
    public var canStop: Bool { isRunning && (job?.canStop ?? false) }

    public var settings: [String: JSONValue] {
        ["mode": .string(mode), "trials": .integer(Int64(trials)), "seed": .integer(Int64(seed))]
    }

    public func start(in workspace: Workspace) {
        do {
            let job = try workspace.submitJob("tolerance", settings: settings)
            self.job = job
            result = nil
            state = .running(JobProgress(fraction: 0, stage: "Queued"))
            Task { @MainActor [weak self] in
                for await p in job.progress {
                    guard let self, self.job === job, case .running = self.state else { return }
                    self.state = .running(p)
                }
            }
            Task { @MainActor [weak self] in
                do {
                    let out = try await job.result()
                    guard let self, self.job === job else { return }
                    self.result = ToleranceResult(json: out.json, system: workspace.snapshot?.system)
                    self.state = .finished
                } catch EngineError.cancelled {
                    guard let self, self.job === job else { return }
                    self.state = .setup
                } catch {
                    guard let self, self.job === job else { return }
                    self.state = .failed(error.localizedDescription)
                }
            }
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    public func stop() { job?.stop() }
    public func cancel() {
        job?.cancel()
        state = .setup
    }
}

/// The parts of a `tolerance` result the sheet shows.
public struct ToleranceResult: Sendable {
    public struct Sensitivity: Identifiable, Sendable {
        public var id: Int
        public var label: String
        public var minus: Double
        public var plus: Double
        public var changeMinus: Double
        public var changePlus: Double
        public var worst: Double
    }
    public var status: String
    public var nominal: Double
    public var sensitivities: [Sensitivity]
    public var rssChange: Double
    public var predicted: Double
    public var trials: Int
    public var mean: Double
    public var stdDev: Double
    public var min: Double
    public var max: Double
    public var percentiles: [(Double, Double)]
    public var criteria: [Double]
    public var yield: Double?

    public init(json: JSONValue, system: OpticalSystem?) {
        status = json["status"]?.string ?? ""
        let s = json["sensitivity"]
        nominal = s?[path: "nominal", "criterion"]?.nf ?? .nan
        let ops = (s?["operands"]?.array ?? []).enumerated().map { k, o -> Sensitivity in
            var label = o["label"]?.string ?? ""
            if label.isEmpty, let p = o["param"], let path = try? MKLightJSON.decode(ParamPath.self, from: MKLightJSON.encode(p)),
                let system
            {
                label = path.label(in: system)
            }
            return Sensitivity(
                id: o["index"]?.int ?? k, label: label, minus: o["minus"]?.nf ?? .nan, plus: o["plus"]?.nf ?? .nan,
                changeMinus: o["change_minus"]?.nf ?? .nan, changePlus: o["change_plus"]?.nf ?? .nan,
                worst: o["worst"]?.nf ?? .nan)
        }
        let ranking = s?["ranking"]?.intArray ?? []
        sensitivities = ranking.isEmpty ? ops.sorted { $0.worst > $1.worst } : ranking.compactMap { r in ops.first { $0.id == r } }
        rssChange = s?["rss_change"]?.nf ?? .nan
        predicted = s?["predicted_criterion"]?.nf ?? .nan
        let mc = json["monte_carlo"]
        trials = mc?["trials"]?.int ?? 0
        mean = mc?["mean"]?.nf ?? .nan
        stdDev = mc?["std_dev"]?.nf ?? .nan
        min = mc?["min"]?.nf ?? .nan
        max = mc?["max"]?.nf ?? .nan
        percentiles = (mc?["percentiles"]?.array ?? []).compactMap { p in
            guard let q = p["percentile"]?.nf, let v = p["value"]?.nf else { return nil }
            return (q, v)
        }
        criteria = (mc?["criteria"]?.nfArray ?? []).filter(\.isFinite)
        yield = mc?["yield_fraction"]?.nf.flatMap { $0.isFinite ? $0 : nil }
    }

    public var hasMonteCarlo: Bool { trials > 0 }
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

    public var changes: Int { outcomes.filter { $0.previous != $0.value }.count }

    public func dryRun(in workspace: Workspace) async {
        state = .running(JobProgress(fraction: 0, stage: "Solving"))
        committed = false
        do {
            let job = try workspace.submitJob("solves", settings: [:])
            let report = try await job.result().json
            read(report, system: workspace.snapshot?.system)
            state = .finished
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    func read(_ report: JSONValue, system: OpticalSystem?) {
        outcomes = (report["outcomes"]?.array ?? []).enumerated().map { k, o in
            let path = o["target"].flatMap { try? MKLightJSON.decode(ParamPath.self, from: MKLightJSON.encode($0)) }
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
