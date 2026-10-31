import Foundation
import MKLightEngine
import MKLightModel
import Observation

/// The Tolerance panel: the design's tolerance set being edited (a working
/// copy of the document's `tolerances` section whose rows keep their
/// identity), the run with live progress, Stop (keep the completed
/// operands and trials) and Cancel (discard), the result, and previews of
/// toleranced designs.
@MainActor
@Observable
public final class ToleranceSession {
    /// The tolerance set (rows, compensators, criterion, mode, trials, seed).
    public private(set) var settings = ToleranceSettings()
    public private(set) var state = JobState.setup
    public private(set) var result: ToleranceResult?
    /// The rows the result's operands came from, in the engine's order.
    public private(set) var resultRows: [ToleranceRow] = []
    /// The system the run perturbed (the design with its toleranced
    /// surfaces' return placements), for previews.
    public private(set) var runSystem: OpticalSystem?
    /// The sensitivity row previewed in the window, and at which limit.
    public private(set) var previewed: (row: Int, plus: Bool)?
    /// When the running job started.
    public private(set) var startedAt: Date?
    @ObservationIgnored private var job: AnalysisJob?
    @ObservationIgnored private var apertureCache: (key: String, value: [SurfaceID: Double])?

    /// The clear semi-diameters the layout resolved, read once per layout
    /// result (fringes are counted over them).
    public func semiDiameters(from layout: AnalysisOutput?) -> [SurfaceID: Double] {
        guard let layout else { return [:] }
        let key = "\(layout.runInfo.snapshotHash.hex)-\(layout.runInfo.startedUnixMs)-\(layout.runInfo.durationMs)"
        if let c = apertureCache, c.key == key { return c.value }
        let v = EngineLayout.autoSemiDiameters(EngineLayout.scene(from: layout))
        apertureCache = (key, v)
        return v
    }

    public init() {}

    public var isRunning: Bool { if case .running = state { true } else { false } }
    public var canStop: Bool { isRunning && (job?.canStop ?? false) }

    // MARK: Editing

    /// Takes the document's set when it changed outside the editor (open,
    /// undo, redo) or the design changed.
    public func load(_ document: DesignDocument) {
        let t = document.tolerances
        if t != settings { settings = Self.keepingIDs(t, from: settings) }
    }

    /// Rows keep their session ids where they match by content, so a
    /// selection survives an undo.
    static func keepingIDs(_ new: ToleranceSettings, from old: ToleranceSettings) -> ToleranceSettings {
        var out = new
        // Align the old and new rows (longest common subsequence by
        // content), so duplicates keep their own ids.
        let o = old.rows, n = new.rows
        var lcs = Array(repeating: Array(repeating: 0, count: n.count + 1), count: o.count + 1)
        for i in stride(from: o.count - 1, through: 0, by: -1) {
            for j in stride(from: n.count - 1, through: 0, by: -1) {
                lcs[i][j] = o[i] == n[j] ? lcs[i + 1][j + 1] + 1 : max(lcs[i + 1][j], lcs[i][j + 1])
            }
        }
        var used = Set<Int>(), matched = Set<Int>()
        var i = 0, j = 0
        while i < o.count && j < n.count {
            if o[i] == n[j] && lcs[i][j] == lcs[i + 1][j + 1] + 1 {
                out.rows[j].id = o[i].id
                used.insert(i)
                matched.insert(j)
                i += 1
                j += 1
            } else if lcs[i][j + 1] >= lcs[i + 1][j] {
                j += 1
            } else {
                i += 1
            }
        }
        // A row whose limits changed keeps the id of the row of the same
        // kind and surface it replaces.
        for k in out.rows.indices where !matched.contains(k) {
            if let m = o.indices.first(where: { !used.contains($0) && o[$0].kind == n[k].kind && o[$0].surface == n[k].surface }) {
                out.rows[k].id = o[m].id
                used.insert(m)
            }
        }
        var cpool = old.compensators
        for k in out.compensators.indices {
            if let j = cpool.firstIndex(where: { $0 == out.compensators[k] }) {
                out.compensators[k].id = cpool[j].id
                cpool.remove(at: j)
            }
        }
        return out
    }

    /// Applies an edit as one undoable step, saved in the document.
    public func edit(
        _ document: DesignDocument, _ actionName: String, context: ToleranceContext?, undoManager: UndoManager?,
        _ change: (inout ToleranceSettings) -> Void
    ) {
        load(document)
        var next = settings
        change(&next)
        guard next != settings else { return }
        settings = next
        document.setTolerances(next, context: context, actionName: actionName, undoManager: undoManager)
    }

    // MARK: Running

    /// Why the run cannot start, if it cannot.
    public func runProblem(context: ToleranceContext) -> String? {
        let issues = settings.issues(in: context)
        let blocking = issues.values.filter { !$0.isSkipped }.count
        if blocking > 0 { return blocking == 1 ? "Fix the tolerance marked in red first." : "Fix the \(blocking) tolerances marked in red first." }
        if settings.rows.count == issues.count { return "Add a tolerance first." }
        if settings.mode != .sensitivity && settings.trials < 2 { return "Run at least 2 Monte-Carlo trials." }
        return nil
    }

    /// Starts the run on the design (with return placements on the
    /// decentred and tilted surfaces).
    public func start(in workspace: Workspace, context: ToleranceContext, merit: MeritSettings) {
        let (s, ids) = settings.jobSettings(in: context, merit: merit)
        let system = settings.preparedSystem(context.system)
        let rows = ids.compactMap { id in settings.rows.first { $0.id == id } }
        result = nil
        previewed = nil
        startedAt = Date()
        state = .running(JobProgress(fraction: 0, stage: "Queued"))
        Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                let job = try await workspace.submitJob("tolerance", settings: s, on: system)
                guard case .running = self.state else {
                    job.cancel()
                    return
                }
                self.begin(job, rows: rows, system: system, display: self.settings.criterion.display(merit: merit))
            } catch {
                self.state = .failed(error.localizedDescription)
            }
        }
    }

    func begin(_ job: AnalysisJob, rows: [ToleranceRow], system: OpticalSystem, display: CriterionDisplay) {
        self.job = job
        resultRows = rows
        runSystem = system
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
                self.result = ToleranceResult(json: out.json, system: system, display: display)
                self.state = .finished
            } catch EngineError.cancelled {
                guard let self, self.job === job else { return }
                self.state = .setup
            } catch {
                guard let self, self.job === job else { return }
                self.state = .failed(error.localizedDescription)
            }
        }
    }

    /// Ends the run now and keeps what it completed.
    public func stop() { job?.stop() }

    /// Ends the run and discards it.
    public func cancel() {
        job?.cancel()
        job = nil
        state = .setup
    }

    /// Back to the editor (the result is kept for Show Results).
    public func edit() {
        state = .setup
    }

    /// Shows the last result again.
    public func showResult() {
        if result != nil { state = .finished }
    }

    // MARK: Previews

    /// The design with sensitivity operand `k` at its − or + limit and the
    /// compensators at the values the engine re-optimised for it: exactly
    /// the system that trial evaluated (parameter values only, no optics).
    public func trialSystem(_ k: Int, plus: Bool) -> OpticalSystem? {
        guard let r = result, let s = r.sensitivities.first(where: { $0.id == k }), var sys = runSystem, let p = s.param,
            let v = sys.value(at: p)
        else { return nil }
        guard sys.setValue(v + (plus ? s.plus : -s.minus), at: p) else { return nil }
        let comps = plus ? s.compensatorsPlus : s.compensatorsMinus
        for (c, value) in zip(r.compensators, comps) where value.isFinite { sys.setValue(value, at: c.param) }
        return sys
    }

    /// Previews operand `k` at a limit in the window (`nil` ends it).
    public func preview(_ k: Int?, plus: Bool = true, in workspace: Workspace) {
        guard let k, let s = trialSystem(k, plus: plus), let r = result, let sens = r.sensitivities.first(where: { $0.id == k }) else {
            previewed = nil
            workspace.preview(nil)
            return
        }
        previewed = (k, plus)
        let row = resultRows.indices.contains(k) ? resultRows[k] : nil
        let what = row.map { "\($0.label(in: s)) at \(plus ? "+" : "−")\(Fmt.short(plus ? $0.plus : $0.minus))\($0.unit == .arcmin ? "′" : " " + $0.unit.label)" }
            ?? sens.label
        workspace.preview(s, title: "Tolerance: " + what + (r.compensators.isEmpty ? "" : ", compensated"))
    }
}

/// The parts of a `tolerance` result the panel shows (docs/OPTIMIZATION.md).
public struct ToleranceResult: Sendable {
    public struct Sensitivity: Identifiable, Sendable {
        /// The engine's operand index (= the row's position in the run).
        public var id: Int
        public var label: String
        public var param: ParamPath?
        /// Limits in parameter units.
        public var minus: Double
        public var plus: Double
        public var criterionMinus: Double
        public var criterionPlus: Double
        public var changeMinus: Double
        public var changePlus: Double
        public var worst: Double
        public var ok: Bool
        public var compensatorsMinus: [Double]
        public var compensatorsPlus: [Double]
    }

    public struct Compensator: Sendable {
        public var param: ParamPath
        public var range: Double?
    }

    public var status: String
    public var stopped: Bool
    public var display: CriterionDisplay
    public var compensators: [Compensator]
    // Sensitivity
    public var hasSensitivity: Bool
    public var nominal: Double
    public var nominalCompensators: [Double]
    /// Worst first (the engine's ranking).
    public var sensitivities: [Sensitivity]
    public var rssChange: Double
    public var predicted: Double
    public var evaluations: Int
    public var sensitivityCancelled: Bool
    // Monte Carlo
    public var trials: Int
    public var failures: Int
    public var seed: Int
    public var mean: Double
    public var stdDev: Double
    public var min: Double
    public var max: Double
    /// The engine's percentiles of the criterion (percent, value).
    public var percentiles: [(Double, Double)]
    /// Finite criteria of the trials.
    public var criteria: [Double]
    public var yield: Double?
    public var threshold: Double?
    public var compensatorMean: [Double]
    public var compensatorStd: [Double]
    public var monteCarloCancelled: Bool

    public init(json: JSONValue, system: OpticalSystem?, display: CriterionDisplay = .merit) {
        self.display = display
        status = json["status"]?.string ?? ""
        stopped = json["stopped"]?.bool ?? (status == "stopped")
        compensators = (json["compensators"]?.array ?? []).compactMap { c in
            guard let p = c["param"].flatMap({ try? MKLightJSON.decode(ParamPath.self, from: MKLightJSON.encode($0)) }) else { return nil }
            return Compensator(param: p, range: c["range"]?.nf.flatMap { $0.isFinite ? $0 : nil })
        }
        let s = json["sensitivity"].flatMap { $0 == .null ? nil : $0 }
        hasSensitivity = s != nil
        nominal = s?[path: "nominal", "criterion"]?.nf ?? .nan
        nominalCompensators = s?[path: "nominal", "compensators"]?.nfArray ?? []
        let ops = (s?["operands"]?.array ?? []).enumerated().map { k, o -> Sensitivity in
            let path = o["param"].flatMap { try? MKLightJSON.decode(ParamPath.self, from: MKLightJSON.encode($0)) }
            var label = o["label"]?.string ?? ""
            if label.isEmpty || label.hasPrefix("{"), let path, let system { label = path.label(in: system) }
            return Sensitivity(
                id: o["index"]?.int ?? k, label: label, param: path, minus: o["minus"]?.nf ?? .nan, plus: o["plus"]?.nf ?? .nan,
                criterionMinus: o["criterion_minus"]?.nf ?? .nan, criterionPlus: o["criterion_plus"]?.nf ?? .nan,
                changeMinus: o["change_minus"]?.nf ?? .nan, changePlus: o["change_plus"]?.nf ?? .nan, worst: o["worst"]?.nf ?? .nan,
                ok: o["ok"]?.bool ?? true, compensatorsMinus: o["compensators_minus"]?.nfArray ?? [],
                compensatorsPlus: o["compensators_plus"]?.nfArray ?? [])
        }
        let ranking = s?["ranking"]?.intArray ?? []
        let ranked = ranking.compactMap { r in ops.first { $0.id == r } }
        sensitivities = ranked.count == ops.count ? ranked : ops.sorted { ($0.worst.isFinite ? $0.worst : -.infinity) > ($1.worst.isFinite ? $1.worst : -.infinity) }
        rssChange = s?["rss_change"]?.nf ?? .nan
        predicted = s?["predicted_criterion"]?.nf ?? .nan
        evaluations = s?["evaluations"]?.int ?? 0
        sensitivityCancelled = s?["cancelled"]?.bool ?? false
        let mc = json["monte_carlo"].flatMap { $0 == .null ? nil : $0 }
        trials = mc?["trials"]?.int ?? 0
        failures = mc?["failures"]?.int ?? 0
        seed = mc?["seed"]?.int ?? 0
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
        threshold = mc?["threshold"]?.nf.flatMap { $0.isFinite ? $0 : nil }
        compensatorMean = mc?["compensator_mean"]?.nfArray ?? []
        compensatorStd = mc?["compensator_std"]?.nfArray ?? []
        monteCarloCancelled = mc?["cancelled"]?.bool ?? false
    }

    public var hasMonteCarlo: Bool { trials > 0 }

    /// Operand `k`'s share of the RSS prediction: worst² / Σ worst².
    public func share(_ s: Sensitivity) -> Double {
        let total = sensitivities.map { $0.worst.isFinite ? Swift.max($0.worst, 0) * Swift.max($0.worst, 0) : 0 }.reduce(0, +)
        guard total > 0, s.worst.isFinite else { return 0 }
        return Swift.max(s.worst, 0) * Swift.max(s.worst, 0) / total
    }

    /// The worst offenders: the largest shares that make up half of the
    /// RSS (at most five, each above 5 %).
    public var worstOffenders: Set<Int> {
        var out = Set<Int>(), sum = 0.0
        for s in sensitivities {
            let f = share(s)
            guard f > 0.05, sum < 0.5, out.count < 5 else { break }
            out.insert(s.id)
            sum += f
        }
        return out
    }

    /// The criterion below which `p` percent of the trials lie (linear
    /// between order statistics): the level `p` % of builds reach or beat.
    public func percentile(_ p: Double) -> Double {
        Self.percentile(criteria.sorted(), p)
    }

    static func percentile(_ sorted: [Double], _ p: Double) -> Double {
        guard !sorted.isEmpty else { return .nan }
        let x = Swift.min(Swift.max(p / 100, 0), 1) * Double(sorted.count - 1)
        let i = Int(x.rounded(.down)), f = x - Double(i)
        return i + 1 < sorted.count ? sorted[i] * (1 - f) + sorted[i + 1] * f : sorted[i]
    }

    /// Percent levels shown: the median, 90 % and 97.7 % (2σ one-sided).
    public static let levels: [Double] = [50, 90, 97.7]

    /// The fraction of trials whose criterion is at most `c`.
    public func cumulative(_ c: Double) -> Double {
        guard !criteria.isEmpty else { return .nan }
        return Double(criteria.filter { $0 <= c }.count) / Double(criteria.count)
    }

    /// Range of a compensator over the sensitivity evaluations, relative
    /// to its nominal value: (lowest, highest).
    public func compensatorRange(_ j: Int) -> (Double, Double)? {
        guard nominalCompensators.indices.contains(j) else { return nil }
        let n = nominalCompensators[j]
        let v = sensitivities.flatMap { s in
            [s.compensatorsMinus, s.compensatorsPlus].compactMap { $0.indices.contains(j) ? $0[j] : nil }
        }.filter(\.isFinite)
        guard let lo = v.min(), let hi = v.max() else { return nil }
        return (Swift.min(lo - n, 0), Swift.max(hi - n, 0))
    }
}
