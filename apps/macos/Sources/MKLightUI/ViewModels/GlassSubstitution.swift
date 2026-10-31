import Foundation
import MKLightEngine
import MKLightModel
import Observation

/// Which catalogue glasses may replace a lens's glass (the engine's
/// `CandidateSet`, docs/OPTIMIZATION.md "Candidates and model-glass
/// snapping"), in plain terms.
public struct GlassCandidates: Hashable, Sendable {
    public enum Rule: String, Hashable, Sendable { case nearest, window }
    public static let allStatuses = ["preferred", "standard", "melt", "special", "obsolete"]

    /// Catalogues to draw from; empty: every loaded catalogue.
    public var catalogs: [String] = []
    public var statuses: Set<String> = ["standard", "preferred", "melt"]
    public var rule = Rule.nearest
    /// The K glasses nearest the current one.
    public var nearest = 10
    /// Half-widths around the current glass.
    public var windowNd = 0.05
    public var windowVd = 5.0
    /// Names or `catalog:name`.
    public var include: [String] = []
    public var exclude: [String] = []
    public var minTransmission: Double?
    public var maxRelativeCost: Double?
    public var maxClimateClass: Double?
    public var maxStainClass: Double?

    public init() {}

    /// The engine's `candidates` object.
    public var json: JSONValue {
        var o: [String: JSONValue] = [
            "catalogs": .array(catalogs.map(JSONValue.string)),
            "statuses": .array(Self.allStatuses.filter(statuses.contains).map(JSONValue.string)),
            "include": .array(include.map(JSONValue.string)), "exclude": .array(exclude.map(JSONValue.string)),
        ]
        switch rule {
        case .nearest:
            o["nearest"] = .integer(Int64(nearest))
            o["window"] = .null
        case .window:
            o["nearest"] = .null
            o["window"] = .object(["nd": .number(windowNd), "vd": .number(windowVd)])
        }
        func put(_ k: String, _ v: Double?) { if let v { o[k] = .number(v) } }
        put("min_transmission", minTransmission)
        put("max_relative_cost", maxRelativeCost)
        put("max_climate_class", maxClimateClass)
        put("max_stain_class", maxStainClass)
        return .object(o)
    }

    /// A short description, e.g. "10 nearest · standard, preferred, melt".
    public var summary: String {
        var parts: [String] = []
        switch rule {
        case .nearest: parts.append("\(nearest) nearest")
        case .window: parts.append("n_d ±\(Fmt.short(windowNd)), V_d ±\(Fmt.short(windowVd))")
        }
        parts.append(catalogs.isEmpty ? "all catalogues" : catalogs.joined(separator: ", "))
        if statuses != ["standard", "preferred", "melt"] { parts.append(Self.allStatuses.filter(statuses.contains).joined(separator: ", ")) }
        if !include.isEmpty { parts.append("+\(include.count)") }
        if !exclude.isEmpty { parts.append("−\(exclude.count)") }
        if [minTransmission, maxRelativeCost, maxClimateClass, maxStainClass].contains(where: { $0 != nil }) { parts.append("limits") }
        return parts.joined(separator: " · ")
    }

    /// Splits "N-BK7, SCHOTT:F2" into names.
    public static func names(_ text: String) -> [String] {
        text.split(whereSeparator: { $0 == "," || $0 == ";" || $0 == "\n" }).map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
    }
}

/// A glass gap of the design that substitution may change: the stable id
/// of the surface in front of the glass, and the glass now there.
public struct GlassSlot: Identifiable, Hashable, Sendable {
    public var id: SurfaceID
    public var index: Int
    public var material: String
    public var isModel: Bool

    /// Every glass gap of a design (refracting surfaces followed by glass).
    public static func all(in system: OpticalSystem) -> [GlassSlot] {
        system.surfaces.enumerated().compactMap { i, s in
            guard i > 0, i < system.imageIndex, !s.isCoordinateBreak, s.interaction == .refract, !s.material.isAir else { return nil }
            if case .model = s.material { return GlassSlot(id: s.id, index: i, material: s.material.displayName, isModel: true) }
            return GlassSlot(id: s.id, index: i, material: s.material.displayName, isModel: false)
        }
    }
}

/// A glass in a candidate combination.
public struct SubstitutedGlass: Hashable, Sendable {
    public var surface: SurfaceID
    public var catalog: String
    public var name: String
    public var nd: Double
    public var vd: Double
    public var dpgf: Double
    /// `catalog/name`, the glass map's id.
    public var mapID: String { "\(catalog)/\(name)" }

    init(json g: JSONValue) {
        surface = SurfaceID(UInt64(max(g["surface"]?.int ?? 0, 0)))
        catalog = g["catalog"]?.string ?? ""
        name = g["name"]?.string ?? "?"
        nd = g["nd"]?.nf ?? .nan
        vd = g["vd"]?.nf ?? .nan
        dpgf = g["dpgf"]?.nf ?? .nan
    }
}

/// One ranked glass combination with its re-optimised design.
public struct GlassCombination: Identifiable, Sendable {
    public struct Change: Sendable, Hashable {
        public var param: ParamPath?
        public var start: Double
        public var value: Double
    }
    public var rank: Int
    public var merit: Double
    public var violation: Double
    public var feasible: Bool
    public var glasses: [SubstitutedGlass]
    public var changes: [Change]
    public var system: OpticalSystem?
    public var cutShort: Bool
    public var id: Int { rank }

    init(json r: JSONValue, fallbackRank: Int) {
        rank = r["rank"]?.int ?? fallbackRank
        merit = r["merit"]?.nf ?? .nan
        violation = r["violation"]?.nf ?? .nan
        feasible = r["feasible"]?.bool ?? true
        glasses = (r["glasses"]?.array ?? []).map(SubstitutedGlass.init(json:))
        changes = (r["variables"]?.array ?? []).map { v in
            Change(
                param: v["param"].flatMap { try? MKLightJSON.decode(ParamPath.self, from: MKLightJSON.encode($0)) },
                start: v["start"]?.nf ?? .nan, value: v["value"]?.nf ?? .nan)
        }
        system = r["system"].flatMap { try? MKLightJSON.decode(OpticalSystem.self, from: MKLightJSON.encode($0)) }
        cutShort = r["local_termination"]?.string == "cancelled"
    }

    /// Variables that moved noticeably.
    public var movedChanges: [Change] {
        changes.filter { c in
            c.value.isFinite && c.start.isFinite && abs(c.value - c.start) > 1e-9 * max(abs(c.start), 1)
        }
    }
}

/// The `glass_substitution` result.
public struct GlassSubstitutionResult: Sendable {
    public var status: String
    public var stopped: Bool
    public var algorithm: String
    public var termination: String
    public var startingMerit: Double
    public var combinations: Int
    public var failed: Int
    public var ranked: [GlassCombination]
    /// The best merit after each re-optimisation.
    public var history: [Double]
    /// Candidate glasses per substituted surface.
    public var candidateCounts: [SurfaceID: Int]

    public init?(json: JSONValue) {
        guard json.object != nil, let list = json["results"]?.array else { return nil }
        status = json["status"]?.string ?? ""
        stopped = json["stopped"]?.bool ?? false
        algorithm = json["algorithm"]?.string ?? ""
        termination = json["termination"]?.string ?? ""
        startingMerit = json["starting_merit"]?.nf ?? .nan
        combinations = json[path: "statistics", "combinations"]?.int ?? 0
        failed = json[path: "statistics", "failed"]?.int ?? 0
        ranked = list.enumerated().map { GlassCombination(json: $0.element, fallbackRank: $0.offset + 1) }
        history = (json["history"]?.array ?? []).compactMap { $0["best_merit"]?.nf }.filter(\.isFinite)
        var counts: [SurfaceID: Int] = [:]
        for c in json["candidate_sets"]?.array ?? [] {
            if let s = c["surface"]?.int { counts[SurfaceID(UInt64(s))] = c["glasses"]?.array?.count ?? 0 }
        }
        candidateCounts = counts
    }
}

/// Live progress of a running substitution, from the engine's stage text
/// `glass_substitution: combinations N; best merit M; glasses [...]`.
public struct GlassProgressInfo: Sendable, Equatable {
    public var combinations: Int
    public var bestMerit: Double
    public var glasses: [String]

    public static func parse(_ stage: String) -> GlassProgressInfo? {
        guard stage.hasPrefix("glass_substitution:") else { return nil }
        var info = GlassProgressInfo(combinations: 0, bestMerit: .nan, glasses: [])
        let body = stage.dropFirst("glass_substitution:".count)
        if let r = body.range(of: "combinations ") {
            info.combinations = Int(body[r.upperBound...].prefix { $0.isNumber }) ?? 0
        }
        if let r = body.range(of: "best merit ") {
            info.bestMerit = Double(body[r.upperBound...].prefix { !$0.isWhitespace && $0 != ";" }) ?? .nan
        }
        if let r = body.range(of: "glasses "), let json = try? JSONValue.parse(Data(body[r.upperBound...].utf8)) {
            info.glasses = (json.array ?? []).map { g in g["name"]?.string ?? "?" }
        }
        return info
    }
}

/// The Glass Substitution panel: which lenses' glasses may change and from
/// which candidates, the search (quick tabu or thorough global), live
/// progress, the ranked combinations (each previewed in the window) and an
/// undoable Apply. The merit and continuous variables are the Optimize
/// panel's.
@MainActor
@Observable
public final class GlassSubstitutionSession {
    public enum Algorithm: String, CaseIterable, Sendable, Identifiable {
        case tabu, hammer
        public var id: String { rawValue }
        public var title: String { self == .tabu ? "Quick (tabu search)" : "Thorough (global search)" }
        public var detail: String {
            self == .tabu
                ? "Tries glass combinations, each re-optimised from the current shape."
                : "Hops over glasses and shape together to escape local minima; slower."
        }
    }

    /// Surfaces whose glass may change. Empty with no glass selected: the
    /// engine snaps every model glass to its nearest real glasses.
    public var selected: Set<SurfaceID> = []
    public var candidates = GlassCandidates()
    /// Per-lens candidate settings that replace `candidates`.
    public var overrides: [SurfaceID: GlassCandidates] = [:]
    public var algorithm = Algorithm.tabu
    public var maxCombinations = 100
    public var maxSeconds: Double?
    public var localIterations = 100
    public var topN = 10
    /// Use the design's variables (else every curvature and the back focus).
    public var useDesignVariables = true

    public private(set) var state = JobState.setup
    public private(set) var progress: GlassProgressInfo?
    public private(set) var trace: [Double] = []
    public private(set) var result: GlassSubstitutionResult?
    /// The previewed combination (its rank).
    public private(set) var selectedRank: Int?
    public private(set) var applied = false
    /// The design's glasses when the run started, for the comparison.
    public private(set) var startGlasses: [SurfaceID: String] = [:]
    @ObservationIgnored private var job: AnalysisJob?
    @ObservationIgnored private var configuredFor: [SurfaceID] = []

    public init() {}

    public var isRunning: Bool { if case .running = state { true } else { false } }
    public var canStop: Bool { isRunning && (job?.canStop ?? false) }
    public var selectedCombination: GlassCombination? { result?.ranked.first { $0.rank == selectedRank } }

    /// Selects every glass of a design the first time (or when its glass
    /// gaps changed).
    public func configure(for system: OpticalSystem) {
        let slots = GlassSlot.all(in: system).map(\.id)
        guard slots != configuredFor else { return }
        configuredFor = slots
        selected = Set(slots).intersection(selected.isEmpty ? Set(slots) : selected)
        overrides = overrides.filter { slots.contains($0.key) }
    }

    /// The `glass_substitution` settings.
    public func settings(system: OpticalSystem, merit: MeritSettings, variables: [DesignVariable]) -> [String: JSONValue] {
        var s: [String: JSONValue] = [
            "algorithm": .string(algorithm.rawValue), "max_combinations": .integer(Int64(maxCombinations)),
            "local_iterations": .integer(Int64(localIterations)), "top_n": .integer(Int64(topN)),
        ]
        if let maxSeconds { s["max_seconds"] = .number(maxSeconds) }
        s.merge(merit.jobSettings) { _, b in b }
        if let v = OptimizeSession.variableSettings(variables, useDesign: useDesignVariables) { s["variables"] = v }
        let order = GlassSlot.all(in: system).map(\.id).filter(selected.contains)
        s["material_variables"] = .array(
            order.map { id in .object(["surface": .integer(Int64(id.rawValue)), "candidates": (overrides[id] ?? candidates).json]) })
        return s
    }

    public func start(in workspace: Workspace, document: DesignDocument) {
        let system = document.system
        let s = settings(system: system, merit: document.merit, variables: document.variables)
        startGlasses = Dictionary(
            uniqueKeysWithValues: GlassSlot.all(in: system).map { ($0.id, $0.material) })
        workspace.preview(nil)
        do {
            let job = try workspace.submitJob("glass_substitution", settings: s)
            begin(job, workspace: workspace)
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    func begin(_ job: AnalysisJob, workspace: Workspace) {
        self.job = job
        progress = nil
        trace = []
        result = nil
        selectedRank = nil
        applied = false
        state = .running(JobProgress(fraction: 0, stage: "Preparing candidates"))
        Task { @MainActor [weak self] in
            for await p in job.progress {
                guard let self, self.job === job, case .running = self.state else { return }
                self.state = .running(p)
                if let info = GlassProgressInfo.parse(p.stage) {
                    self.progress = info
                    if info.bestMerit.isFinite, info.bestMerit != self.trace.last { self.trace.append(info.bestMerit) }
                }
            }
        }
        Task { @MainActor [weak self] in
            do {
                let out = try await job.result()
                guard let self, self.job === job else { return }
                guard let r = GlassSubstitutionResult(json: out.json) else {
                    self.state = .failed("The engine returned no glass combinations.")
                    return
                }
                self.result = r
                if !r.history.isEmpty { self.trace = Self.runningMinimum(r.history) }
                self.state = .finished
                if let best = r.ranked.first { self.select(best.rank, in: workspace) }
            } catch EngineError.cancelled {
                guard let self, self.job === job else { return }
                self.state = .setup
            } catch {
                guard let self, self.job === job else { return }
                self.state = .failed(error.localizedDescription)
            }
        }
    }

    static func runningMinimum(_ v: [Double]) -> [Double] {
        var out: [Double] = []
        for x in v { out.append(min(x, out.last ?? x)) }
        return out
    }

    /// Ends the search now, keeping the ranking so far.
    public func stop() { _ = job?.stop() }

    /// Ends the search and discards it.
    public func cancel() {
        job?.cancel()
        state = .setup
    }

    /// Previews a combination in the window (layout and analyses).
    public func select(_ rank: Int?, in workspace: Workspace) {
        selectedRank = rank
        guard let c = selectedCombination, let s = c.system, !applied else {
            workspace.preview(nil)
            return
        }
        workspace.preview(s, title: "Candidate \(c.rank)")
    }

    /// Applies the previewed combination's design as one undoable step.
    public func apply(to document: DesignDocument, workspace: Workspace, undoManager: UndoManager?) {
        guard let s = selectedCombination?.system, !applied else { return }
        document.replace(system: s, actionName: "Apply Glass Substitution", undoManager: undoManager)
        applied = true
        workspace.preview(nil)
    }

    /// Back to the settings; ends any preview.
    public func reset(_ workspace: Workspace?) {
        workspace?.preview(nil)
        state = .setup
        result = nil
        progress = nil
        trace = []
        selectedRank = nil
        applied = false
    }
}
