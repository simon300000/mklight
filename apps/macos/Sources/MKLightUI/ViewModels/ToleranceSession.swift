import Foundation
import MKLightEngine
import MKLightModel
import Observation

/// A toleranced build the engine can rebuild (`preview_case`): a
/// Monte-Carlo trial, or one sensitivity operand at its − or + limit.
public enum TolerancePreviewCase: Hashable, Sendable {
    case trial(Int)
    case sensitivity(operand: Int, plus: Bool)

    /// The engine's `preview_case` (zero-based).
    public var json: JSONValue {
        switch self {
        case .trial(let k): .object(["type": .string("trial"), "trial": .integer(Int64(k))])
        case .sensitivity(let k, let plus): .object(["type": .string("sensitivity"), "operand": .integer(Int64(k)), "positive": .bool(plus)])
        }
    }
}

/// The engine's rebuild of a toleranced build: the perturbed system with
/// its compensators re-optimised, the criterion, the compensator values
/// and the perturbation of every operand (parameter units, run order).
public struct TolerancePreview: Sendable {
    public var previewCase: TolerancePreviewCase
    public var system: OpticalSystem
    public var criterion: Double
    public var compensators: [Double]
    public var perturbations: [Double]

    public init?(json: JSONValue?, case c: TolerancePreviewCase) {
        guard let j = json, j.object != nil,
            let s = j["system"].flatMap({ try? MKLightJSON.decode(OpticalSystem.self, from: MKLightJSON.encode($0)) })
        else { return nil }
        previewCase = c
        system = s
        criterion = j["criterion"]?.nf ?? .nan
        compensators = j["compensators"]?.nfArray ?? []
        perturbations = j["perturbations"]?.nfArray ?? []
    }
}

/// The Tolerance panel: the design's tolerance set being edited (a working
/// copy of the document's `tolerances` section whose rows keep their
/// identity), the run with live progress, Stop (keep the completed
/// operands and trials) and Cancel (discard), the result, and previews of
/// toleranced builds rebuilt by the engine.
@MainActor
@Observable
public final class ToleranceSession {
    /// The tolerance set (rows, compensators, criterion, mode, trials, seed).
    public private(set) var settings = ToleranceSettings()
    public private(set) var state = JobState.setup
    public private(set) var result: ToleranceResult?
    /// The rows the result's operands came from, in the engine's order.
    public private(set) var resultRows: [ToleranceRow] = []
    /// The system the run perturbed (the design as it was), for previews.
    public private(set) var runSystem: OpticalSystem?
    /// The job settings of the run, for previews of its builds.
    public private(set) var runSettings: [String: JSONValue]?
    /// The build previewed in the window (being rebuilt while `preview` is nil).
    public private(set) var previewed: TolerancePreviewCase?
    /// The engine's rebuild of `previewed`.
    public private(set) var preview: TolerancePreview?
    /// Why the previewed build could not be rebuilt.
    public private(set) var previewError: String?
    public var previewLoading: Bool { previewed != nil && preview == nil && previewError == nil }
    /// When the running job started, and when its Monte Carlo began.
    public private(set) var startedAt: Date?
    public private(set) var monteCarloStartedAt: Date?
    /// The wall time per trial of the last Monte Carlo, for estimates.
    public private(set) var lastSecondsPerTrial: Double?
    @ObservationIgnored private var job: AnalysisJob?
    @ObservationIgnored private var previewJob: AnalysisJob?
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
        let blocking = settings.issues(in: context).count
        if blocking > 0 { return blocking == 1 ? "Fix the tolerance marked in red first." : "Fix the \(blocking) tolerances marked in red first." }
        if settings.rows.isEmpty { return "Add a tolerance first." }
        if settings.mode != .sensitivity && settings.trials < 2 { return "Run at least 2 Monte-Carlo trials." }
        return nil
    }

    /// Starts the run on the design. A decentre or tilt of one surface
    /// returns to the axis after it, and an element moves its own surfaces
    /// only: the engine does both, so the design is sent as it is.
    public func start(in workspace: Workspace, context: ToleranceContext, merit: MeritSettings) {
        let (s, ids) = settings.jobSettings(in: context, merit: merit)
        let system = context.system
        let rows = ids.compactMap { id in settings.rows.first { $0.id == id } }
        endPreview(in: workspace)
        result = nil
        startedAt = Date()
        monteCarloStartedAt = nil
        state = .running(JobProgress(fraction: 0, stage: "Queued"))
        Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                let job = try await workspace.submitJob("tolerance", settings: s, on: system)
                guard case .running = self.state else {
                    job.cancel()
                    return
                }
                self.begin(job, rows: rows, system: system, settings: s, display: self.settings.criterion.display(merit: merit))
            } catch {
                self.state = .failed(error.localizedDescription)
            }
        }
    }

    func begin(_ job: AnalysisJob, rows: [ToleranceRow], system: OpticalSystem, settings s: [String: JSONValue], display: CriterionDisplay) {
        self.job = job
        resultRows = rows
        runSystem = system
        runSettings = s
        Task { @MainActor [weak self] in
            for await p in job.progress {
                guard let self, self.job === job, case .running = self.state else { return }
                if self.monteCarloStartedAt == nil, p.stage.lowercased().contains("monte carlo") { self.monteCarloStartedAt = Date() }
                self.state = .running(p)
            }
        }
        Task { @MainActor [weak self] in
            do {
                let out = try await job.result()
                guard let self, self.job === job else { return }
                let r = ToleranceResult(json: out.json, system: system, display: display)
                self.result = r
                if let t = r.secondsPerTrial, t > 0 { self.lastSecondsPerTrial = t }
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

    // MARK: Timing

    /// Where a run stands and how long the rest should take, from its
    /// progress: Monte-Carlo trials done, the wall time per trial (measured,
    /// or the last run's, or the sensitivity's time per evaluation) and
    /// the time left.
    public struct Timing: Equatable, Sendable {
        public var trialsDone: Int?
        public var secondsPerTrial: Double?
        public var secondsLeft: Double?
    }

    /// The share of the job's progress the sensitivity takes (the engine
    /// gives it a fifth when both run).
    static func sensitivityShare(_ mode: ToleranceSettings.Mode) -> Double {
        switch mode {
        case .sensitivity: 1
        case .monteCarlo: 0
        case .both: 0.2
        }
    }

    public static func timing(
        elapsed: TimeInterval, monteCarloElapsed: TimeInterval?, fraction: Double, mode: ToleranceSettings.Mode, trials: Int, operands: Int,
        lastSecondsPerTrial: Double?
    ) -> Timing {
        let f = min(max(fraction, 0), 1)
        let split = sensitivityShare(mode)
        var t = Timing()
        if mode != .sensitivity, f > split || monteCarloElapsed != nil {
            let mcf = split < 1 ? min(max((f - split) / (1 - split), 0), 1) : 0
            let done = Int((mcf * Double(trials)).rounded())
            t.trialsDone = done
            if let mce = monteCarloElapsed, done >= 2, mce > 0 {
                t.secondsPerTrial = mce / Double(done)
            } else {
                t.secondsPerTrial = lastSecondsPerTrial
            }
            if let spt = t.secondsPerTrial { t.secondsLeft = Double(max(trials - done, 0)) * spt }
            return t
        }
        // In the sensitivity: its own rest by its rate, then the trials.
        let sf = split > 0 ? f / split : 0
        let evaluations = Double(2 * max(operands, 0) + 1)
        guard sf >= 0.05 else {
            if mode != .sensitivity, let last = lastSecondsPerTrial {
                t.secondsPerTrial = last
                t.secondsLeft = evaluations * last + Double(trials) * last
            }
            return t
        }
        let sensLeft = elapsed * (1 - sf) / sf
        let perEvaluation = elapsed / (sf * evaluations)
        if mode == .sensitivity {
            t.secondsLeft = sensLeft
        } else {
            let spt = lastSecondsPerTrial ?? perEvaluation
            t.secondsPerTrial = spt
            t.secondsLeft = sensLeft + Double(trials) * spt
        }
        return t
    }

    /// The running job's timing now.
    public func timing(at now: Date, progress: JobProgress) -> Timing {
        Self.timing(
            elapsed: startedAt.map { now.timeIntervalSince($0) } ?? 0, monteCarloElapsed: monteCarloStartedAt.map { now.timeIntervalSince($0) },
            fraction: progress.fraction, mode: settings.mode, trials: settings.trials, operands: resultRows.count,
            lastSecondsPerTrial: lastSecondsPerTrial)
    }

    // MARK: Previews

    /// Rebuilds a build of the last run in the engine (`preview_case`) and
    /// previews it in the window, read-only (`nil` ends the preview). The
    /// run's own settings and system are used, so the build is exactly the
    /// one the run evaluated.
    public func preview(_ c: TolerancePreviewCase?, in workspace: Workspace) {
        previewJob?.cancel()
        previewJob = nil
        guard let c, var s = runSettings, let system = runSystem else {
            endPreview(in: workspace)
            return
        }
        previewed = c
        preview = nil
        previewError = nil
        s["preview_case"] = c.json
        s["record_trials"] = .integer(0)
        Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                let job = try await workspace.submitJob("tolerance", settings: s, on: system, key: "tolerance-preview")
                guard self.previewed == c else {
                    job.cancel()
                    return
                }
                self.previewJob = job
                let out = try await job.result()
                guard self.previewed == c else { return }
                guard let p = TolerancePreview(json: out.json["preview"], case: c) else {
                    self.previewError = "The engine could not rebuild this build: no ray reached the image."
                    workspace.preview(nil)
                    return
                }
                self.preview = p
                workspace.preview(p.system, title: self.title(of: c))
            } catch EngineError.cancelled {
                return
            } catch {
                guard self.previewed == c else { return }
                self.previewError = error.localizedDescription
            }
        }
    }

    /// Ends any preview.
    public func endPreview(in workspace: Workspace) {
        previewJob?.cancel()
        previewJob = nil
        let was = previewed != nil
        previewed = nil
        preview = nil
        previewError = nil
        if was || workspace.previewTitle?.hasPrefix("Tolerance") == true { workspace.preview(nil) }
    }

    /// "Tolerance: trial 17, compensated", "Tolerance: Decentre y S3 at +0.02 mm".
    public func title(of c: TolerancePreviewCase) -> String {
        let compensated = (result?.compensators.isEmpty ?? true) ? "" : ", compensated"
        switch c {
        case .trial(let k):
            return "Tolerance: trial \(k + 1)" + compensated
        case .sensitivity(let k, let plus):
            let what: String
            if let system = runSystem, resultRows.indices.contains(k) {
                what = "\(resultRows[k].label(in: system)) at \(resultRows[k].limitText(plus: plus))"
            } else {
                what = result?.sensitivities.first { $0.id == k }?.label ?? "operand \(k + 1)"
            }
            return "Tolerance: " + what + compensated
        }
    }
}

/// One perturbation of a build as its row reads it.
public struct ToleranceBuildChange: Identifiable, Sendable {
    /// The operand (row position in the run).
    public var id: Int
    public var label: String
    /// The perturbation over the limit on its side (−1 … 1).
    public var fraction: Double
    /// `+0.84′`, `−1.9 fr` in the row's unit (a radius in ΔR or % is
    /// scaled from its limit).
    public var text: String
    /// The row's limits, `±1′`.
    public var limits: String
}

extension ToleranceSession {
    /// The engine limits of the run's operands (parameter units).
    public var runLimits: [(minus: Double, plus: Double)] {
        (runSettings?["tolerances"]?.array ?? []).map { ($0["minus"]?.nf ?? .nan, $0["plus"]?.nf ?? .nan) }
    }

    /// A build's perturbations (parameter units, run order) as the rows
    /// read them, largest share of its limit first; zero ones left out.
    public func changes(_ perturbations: [Double]) -> [ToleranceBuildChange] {
        let limits = runLimits
        let system = runSystem
        var out: [ToleranceBuildChange] = []
        for (k, d) in perturbations.enumerated() where d != 0 && d.isFinite {
            guard limits.indices.contains(k), resultRows.indices.contains(k) else { continue }
            let row = resultRows[k]
            let lim = d > 0 ? limits[k].plus : limits[k].minus
            guard lim > 0 else { continue }
            let f = d / lim
            let shown = (d > 0 ? row.plus : row.minus) * abs(f)
            let u = row.unit.label.isEmpty ? "" : (row.unit == .arcmin ? row.unit.label : " " + row.unit.label)
            out.append(
                ToleranceBuildChange(
                    id: k, label: system.map { row.label(in: $0) } ?? row.kind.title, fraction: f,
                    text: (d > 0 ? "+" : "\u{2212}") + Fmt.short(Self.tidy(shown)) + u, limits: row.limitsText))
        }
        return out.sorted { abs($0.fraction) > abs($1.fraction) }
    }

    /// Three significant digits for a perturbation.
    static func tidy(_ v: Double) -> Double {
        guard v != 0, v.isFinite else { return v }
        let q = pow(10, (log10(abs(v))).rounded(.down) - 2)
        return (v / q).rounded() * q
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

        /// The limit whose change is the larger.
        public var worsePlus: Bool { !(changeMinus > changePlus) }
    }

    public struct Compensator: Sendable {
        public var param: ParamPath
        public var range: Double?
    }

    /// One recorded Monte-Carlo trial: its criterion, compensator values
    /// and the perturbation of every operand (parameter units).
    public struct TrialRecord: Identifiable, Sendable {
        public var trial: Int
        public var criterion: Double
        public var ok: Bool
        public var compensators: [Double]
        public var perturbations: [Double]
        public var id: Int { trial }
    }

    public var status: String
    public var stopped: Bool
    public var display: CriterionDisplay
    public var compensators: [Compensator]
    /// The design as it is with the compensators re-optimised (the
    /// sensitivity's, else the Monte Carlo's).
    public var nominal: Double
    public var nominalCompensators: [Double]
    // Sensitivity
    public var hasSensitivity: Bool
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
    public var percentiles: [(percentile: Double, value: Double)]
    /// The criterion of every trial by trial number (not finite: failed).
    public var trialCriteria: [Double]
    /// Finite criteria of the trials.
    public var criteria: [Double]
    public var records: [TrialRecord]
    /// Wall time per completed trial.
    public var secondsPerTrial: Double?
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
        let mc = json["monte_carlo"].flatMap { $0 == .null ? nil : $0 }
        hasSensitivity = s != nil
        let nominalJSON = s?["nominal"] ?? mc?["nominal"]
        nominal = nominalJSON?["criterion"]?.nf ?? .nan
        nominalCompensators = nominalJSON?["compensators"]?.nfArray ?? []
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
        trialCriteria = mc?["criteria"]?.nfArray ?? []
        criteria = trialCriteria.filter(\.isFinite)
        records = (mc?["records"]?.array ?? []).compactMap { r in
            guard let t = r["trial"]?.int else { return nil }
            return TrialRecord(
                trial: t, criterion: r["criterion"]?.nf ?? .nan, ok: r["ok"]?.bool ?? true, compensators: r["compensators"]?.nfArray ?? [],
                perturbations: r["perturbations"]?.nfArray ?? [])
        }
        secondsPerTrial = mc?["seconds_per_trial"]?.nf.flatMap { $0.isFinite && $0 >= 0 ? $0 : nil }
        yield = mc?["yield_fraction"]?.nf.flatMap { $0.isFinite ? $0 : nil }
        threshold = mc?["threshold"]?.nf.flatMap { $0.isFinite ? $0 : nil }
        compensatorMean = mc?["compensator_mean"]?.nfArray ?? []
        compensatorStd = mc?["compensator_std"]?.nfArray ?? []
        monteCarloCancelled = mc?["cancelled"]?.bool ?? false
    }

    public var hasMonteCarlo: Bool { trials > 0 }
    public var hasNominal: Bool { nominal.isFinite }

    /// The record of trial `k`, if it was kept.
    public func record(_ k: Int) -> TrialRecord? { records.first { $0.trial == k } }

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

    /// The fraction of trials whose criterion is at most `c`.
    public func cumulative(_ c: Double) -> Double {
        guard !criteria.isEmpty else { return .nan }
        return Double(criteria.filter { $0 <= c }.count) / Double(criteria.count)
    }

    /// The trials whose shown value lies in `range`, worst first.
    public func trials(shownIn range: ClosedRange<Double>) -> [Int] {
        let ks = trialCriteria.indices.filter { k in
            let v = display.value(trialCriteria[k])
            return v.isFinite && range.contains(v)
        }
        return ks.sorted { worse(trialCriteria[$0], than: trialCriteria[$1]) }
    }

    /// Every completed trial, worst first.
    public var trialsWorstFirst: [Int] {
        trialCriteria.indices.filter { trialCriteria[$0].isFinite }.sorted { worse(trialCriteria[$0], than: trialCriteria[$1]) }
    }

    /// Lower merits are better in every display.
    private func worse(_ a: Double, than b: Double) -> Bool { a > b }

    /// Range of a compensator over the sensitivity evaluations, relative
    /// to its nominal value: (lowest, highest).
    public func compensatorRange(_ j: Int) -> (Double, Double)? {
        guard hasSensitivity, nominalCompensators.indices.contains(j) else { return nil }
        let n = nominalCompensators[j]
        let v = sensitivities.flatMap { s in
            [s.compensatorsMinus, s.compensatorsPlus].compactMap { $0.indices.contains(j) ? $0[j] : nil }
        }.filter(\.isFinite)
        guard let lo = v.min(), let hi = v.max() else { return nil }
        return (Swift.min(lo - n, 0), Swift.max(hi - n, 0))
    }
}
