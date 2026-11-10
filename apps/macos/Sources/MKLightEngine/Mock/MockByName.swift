import Foundation
import MKLightModel
import Synchronization

// MOCK ENGINE — analyses by name, in the engine's result format
// (docs/RESULTS.md), so front ends that read engine JSON work on the mock
// too (previews, UI tests). The numbers are synthetic (see `MockOptics`).

extension MockEngine {
    /// Analyses the mock runs by name, plus a few listed as unavailable so
    /// front ends can show that state.
    public func analysisDescriptors() async -> [AnalysisDescriptor] {
        Self.mockDescriptors
    }

    static let mockDescriptors: [AnalysisDescriptor] = {
        func sel(_ def: String) -> JSONValue {
            .object(["type": .string("object"), "default": .object(["type": .string(def)]), "title": .string("Selection")])
        }
        func int(_ d: Int, _ lo: Int, _ hi: Int, _ title: String) -> JSONValue {
            .object([
                "type": .string("integer"), "default": .integer(Int64(d)), "minimum": .integer(Int64(lo)),
                "maximum": .integer(Int64(hi)), "title": .string(title),
            ])
        }
        func enm(_ d: String, _ values: [String], _ title: String) -> JSONValue {
            .object([
                "type": .string("string"), "default": .string(d), "enum": .array(values.map(JSONValue.string)),
                "title": .string(title),
            ])
        }
        func schema(_ props: [String: JSONValue]) -> JSONValue {
            .object(["type": .string("object"), "properties": .object(props)])
        }
        func d(_ name: String, _ title: String, _ cat: String, _ props: [String: JSONValue] = [:], available: Bool = true)
            -> AnalysisDescriptor
        {
            AnalysisDescriptor(
                name: name, title: title, category: cat, description: "Mock \(title.lowercased()) (synthetic data).",
                available: available,
                unavailableReason: available ? nil : "the mock engine does not compute \(title.lowercased())",
                settingsSchema: schema(props), defaultSettings: .object([:]),
                stoppable: name == "optimize" || name == "tolerance")
        }
        return [
            d("validate", "Validate system", "model"),
            d("solves", "Solves (preview)", "model"),
            d("first_order", "First-order properties", "geometric", ["wavelength": sel("primary")]),
            d(
                "layout", "Layout", "geometric",
                ["rays_per_field": int(7, 1, 10000, "Rays per field"), "include_3d": .object(["type": .string("boolean"), "default": .bool(true)])]),
            d(
                "spot_diagram", "Spot diagram", "geometric",
                [
                    "fields": sel("all"), "wavelengths": sel("all"), "rings": int(6, 1, 512, "Rings"),
                    "pattern": enm("hexapolar", ["hexapolar", "square", "random"], "Pattern"),
                    "seed": int(1, 0, Int(Int32.max), "Seed"),
                ]),
            d("ray_fan", "Ray aberration fans", "geometric", ["fields": sel("all"), "wavelengths": sel("all"), "samples": int(41, 2, 100001, "Samples")]),
            d("opd_fan", "OPD fans", "geometric", ["fields": sel("all"), "wavelengths": sel("all"), "samples": int(41, 2, 100001, "Samples")]),
            d(
                "field_curvature_distortion", "Field curvature and distortion", "geometric",
                ["field_samples": int(21, 2, 10000, "Field samples"), "wavelengths": sel("all")]),
            d("seidel", "Seidel aberration sums", "geometric", available: false),
            d(
                "optimize", "Optimize", "optimization",
                [
                    "algorithm": enm("lm", ["lm", "dogleg", "de", "cma_es", "nelder_mead", "hammer"], "Algorithm"),
                    "max_iterations": .object(["type": .array([.string("integer"), .string("null")]), "default": .null]),
                    "seed": int(0, 0, Int(Int32.max), "Seed"),
                    "hold_efl": .object(["type": .string("boolean"), "default": .bool(true)]),
                    "variables": .object(["type": .string("array"), "default": .array([])]),
                    "default_merit": .object(["type": .string("object"), "default": .object([:])]),
                ]),
            d(
                "merit_evaluation", "Evaluate merit", "optimization",
                [
                    "algorithm": enm("lm", ["lm", "dogleg", "de", "cma_es", "nelder_mead", "hammer"], "Algorithm"),
                    "hold_efl": .object(["type": .string("boolean"), "default": .bool(true)]),
                    "variables": .object(["type": .string("array"), "default": .array([])]),
                    "default_merit": .object(["type": .string("object"), "default": .object([:])]),
                ]),
            d(
                "tolerance", "Tolerance", "optimization",
                [
                    "mode": enm("both", ["sensitivity", "monte_carlo", "both"], "Mode"),
                    "trials": int(100, 1, 100000, "Trials"), "seed": int(0, 0, Int(Int32.max), "Seed"),
                ]),
            d(
                "glass_info", "Glass data", "materials",
                ["name": .object(["type": .string("string"), "default": .string("")])]),
            d("mtf", "MTF", "wave", available: false),
            d("psf", "Point spread function", "wave", available: false),
            d("wavefront_map", "Wavefront map", "wave", available: false),
        ]
    }()

    public func submit(_ request: EngineAnalysisRequest, on snapshot: ModelSnapshot) -> AnalysisJob {
        let latency = simulatedLatency
        let name = request.analysis
        guard let descriptor = Self.mockDescriptors.first(where: { $0.name == name }) else {
            return AnalysisJob.run(engineRequest: request, snapshot: snapshot) { _ in
                throw EngineError.engine(.notFound, "unknown analysis '\(name)'")
            }
        }
        guard descriptor.available else {
            let why = descriptor.unavailableReason ?? ""
            return AnalysisJob.run(engineRequest: request, snapshot: snapshot) { _ in throw EngineError.unsupported(why) }
        }
        let stop = MockStopFlag()
        let onStop: (@Sendable () -> Bool)? = descriptor.stoppable ? { @Sendable in stop.set() } : nil
        return AnalysisJob.run(engineRequest: request, snapshot: snapshot, onStop: onStop) { reporter in
            let started = Date()
            let optics = MockOptics(system: snapshot.system)
            let (json, buffers): (JSONValue, [ResultBuffer])
            switch name {
            case "optimize":
                (json, buffers) = try await Self.mockOptimize(request.settings, snapshot, optics, reporter, stop, latency)
            case "tolerance":
                (json, buffers) = try await Self.mockTolerance(request.settings, snapshot, reporter, stop, latency)
            case "merit_evaluation":
                if latency > .zero { try await Task.sleep(for: latency / 8) }
                try reporter.checkCancelled()
                (json, buffers) = (try Self.mockMeritEvaluation(request.settings, optics), [])
            default:
                reporter.report(0.2, stage: "Resolving system")
                if latency > .zero { try await Task.sleep(for: latency) }
                try reporter.checkCancelled()
                (json, buffers) = try Self.mockByName(name, request.settings, snapshot, optics)
            }
            var settings = descriptor.defaultSettings.object ?? [:]
            for (k, v) in request.settings { settings[k] = v }
            let seed = request.settings["seed"]?.int.flatMap { UInt64(exactly: $0) }
            let info = RunInfo(
                engineVersion: Self.version, snapshotHash: snapshot.hash, analysis: name, settings: .object(settings),
                backend: "mock", precision: .f64Reference, seed: seed,
                startedUnixMs: UInt64(started.timeIntervalSince1970 * 1000),
                durationMs: Date().timeIntervalSince(started) * 1000, warnings: [Self.mockWarning] + optics.warnings)
            return AnalysisOutput(
                runInfo: info, json: json, buffers: buffers, engineRequest: request,
                backend: BackendReport(requested: "default:mock", used: "mock", precision: "f64_reference"))
        }
    }

    // MARK: Plain analyses

    static func selection(_ v: JSONValue?) -> MockOptics.FieldSelection {
        if v?["type"]?.string == "index", let i = v?["index"]?.int { return .index(i) }
        return .all
    }

    static func wavelengthSelection(_ v: JSONValue?) -> MockOptics.WavelengthSelection {
        switch v?["type"]?.string {
        case "all": .all
        case "primary": .primary
        case "index": v?["index"]?.int.map { .index($0) } ?? .all
        default: .all
        }
    }

    static func pair(_ p: MockOptics.Point2) -> JSONValue { .array([num(p.x), num(p.y)]) }
    static func num(_ d: Double) -> JSONValue {
        d.isNaN ? .null : d.isInfinite ? .string(d > 0 ? "inf" : "-inf") : .number(d)
    }
    static func nums(_ a: [Double?]) -> JSONValue { .array(a.map { num($0 ?? .nan) }) }

    static func mockByName(_ name: String, _ s: [String: JSONValue], _ snap: ModelSnapshot, _ o: MockOptics) throws
        -> (JSONValue, [ResultBuffer])
    {
        let system = snap.system
        let wl = JSONValue.array(system.wavelengths.wavelengths.map { .number($0.um) })
        switch name {
        case "validate":
            let issues = system.structuralIssues()
            let list = issues.map { i -> JSONValue in
                .object([
                    "severity": .string(i.severity == .error ? "error" : "warning"), "code": .string("structure"),
                    "surface": i.surface.map { .integer(Int64($0)) } ?? .null, "message": .string(i.message),
                ])
            }
            let errors = issues.filter { $0.severity == .error }.count
            return (
                .object([
                    "ok": .bool(errors == 0), "errors": .integer(Int64(errors)),
                    "warnings": .integer(Int64(issues.count - errors)), "issues": .array(list),
                ]), []
            )
        case "first_order":
            let fo = o.firstOrder()
            let entries = JSONValue.array(
                fo.entries.map {
                    .object(["key": .string($0.key), "label": .string($0.label), "value": num($0.value), "unit": .string($0.unit)])
                })
            return (.object(["entries": entries, "wavelength_um": .number(fo.wavelengthUm)]), [])
        case "solves":
            return (solvesReport(system), [])
        case "glass_info":
            let name = (s["name"]?.string ?? "").uppercased()
            guard let g = MockOptics.catalog[name] else { throw EngineError.engine(.notFound, "unknown glass '\(name)'") }
            let dn = s["offset"]?["dn_d"]?.nf ?? 0, dv = s["offset"]?["dv_d"]?.nf ?? 0
            return (
                .object([
                    "name": .string(name), "catalog": .string(s["catalog"]?.string ?? "MOCK"), "catalog_nd": .number(g.nd),
                    "catalog_vd": .number(g.vd), "nd": .number(g.nd + dn), "vd": .number(g.vd + dv),
                    "offset": .object(["dn_d": .number(dn), "dv_d": .number(dv)]),
                ]), []
            )
        case "layout":
            var ls = MockOptics.LayoutSettings()
            if let r = s["rays_per_field"]?.int { ls.raysPerField = r }
            return layoutBuffers(try o.layout(ls))
        case "spot_diagram":
            var ss = MockOptics.SpotDiagramSettings()
            ss.fields = selection(s["fields"])
            ss.wavelengths = wavelengthSelection(s["wavelengths"])
            if let r = s["rings"]?.int { ss.rings = r }
            if let p = s["pattern"]?.string, let pat = MockOptics.PupilPattern(rawValue: p) { ss.pattern = pat }
            if let seed = s["seed"]?.int.flatMap({ UInt64(exactly: $0) }) { ss.seed = seed }
            let r = try o.spot(ss)
            var xy: [Double] = [], pw: [Double] = [], pf: [Double] = []
            var fields: [JSONValue] = []
            for (k, f) in r.fields.enumerated() {
                let offset = pw.count
                for p in f.points {
                    xy += [p.x, p.y]
                    pw.append(Double(p.wavelength))
                    pf.append(Double(k))
                }
                fields.append(
                    .object([
                        "field": .integer(Int64(f.field)), "field_value": pair(f.fieldValue),
                        "reference": pair(f.reference), "chief": pair(f.reference), "centroid": pair(f.centroid),
                        "rms_radius": num(f.rmsRadius), "geo_radius": num(f.geoRadius),
                        "point_range": .array([.integer(Int64(offset)), .integer(Int64(f.points.count))]),
                        "polychromatic": .object(["rms_radius_centroid": num(f.rmsRadius), "geo_radius_centroid": num(f.geoRadius)]),
                    ]))
            }
            let wls = ss.wavelengths.indices(in: system.wavelengths)
            return (
                .object([
                    "fields": .array(fields), "airy_radius": num(r.airyRadius), "defocus": .number(0),
                    "wavelengths_um": wl, "wavelengths": .array(wls.map { .integer(Int64($0)) }),
                ]),
                [
                    ResultBuffer(name: "points", doubles: xy, shape: [pw.count, 2]),
                    ResultBuffer(name: "point_field", doubles: pf), ResultBuffer(name: "point_wavelength", doubles: pw),
                ]
            )
        case "ray_fan", "opd_fan":
            var fs = MockOptics.FanSettings()
            fs.fields = selection(s["fields"])
            fs.wavelengths = wavelengthSelection(s["wavelengths"])
            if let n = s["samples"]?.int { fs.samples = n }
            let r = name == "ray_fan" ? try o.rayFan(fs) : try o.opdFan(fs)
            func curves(_ cs: [MockOptics.FanResult.Curve]) -> JSONValue {
                .array(
                    cs.map {
                        .object(["wavelength": .integer(Int64($0.wavelength)), "pupil": nums($0.pupil), "values": nums($0.values)])
                    })
            }
            let fields = r.fields.map { f -> JSONValue in
                .object([
                    "field": .integer(Int64(f.field)), "field_value": pair(f.fieldValue),
                    "tangential": curves(f.tangential), "sagittal": curves(f.sagittal),
                ])
            }
            return (.object(["quantity": .string(r.quantity.rawValue), "fields": .array(fields), "wavelengths_um": wl]), [])
        case "field_curvature_distortion":
            var fs = MockOptics.FieldCurvatureSettings()
            if let n = s["field_samples"]?.int { fs.fieldSamples = n }
            fs.wavelengths = wavelengthSelection(s["wavelengths"])
            let r = try o.fieldCurvature(fs)
            let maxField = r.fieldValues.max() ?? 1
            let curves = r.curves.map { c -> JSONValue in
                .object([
                    "wavelength": .integer(Int64(c.wavelength)),
                    "wavelength_um": .number(system.wavelengths.wavelengths[c.wavelength].um),
                    "tangential": nums(c.tangential), "sagittal": nums(c.sagittal),
                    "distortion_percent": nums(c.distortionPercent),
                ])
            }
            return (
                .object([
                    "field_values": nums(r.fieldValues), "field_unit": .string(r.fieldUnit),
                    "normalized_field": nums(r.fieldValues.map { maxField > 0 ? $0 / maxField : 0 }),
                    "curves": .array(curves), "wavelengths_um": wl,
                ]), []
            )
        default:
            throw EngineError.unsupported("the mock engine does not compute '\(name)'")
        }
    }

    /// Dry run of the solves: pickups are evaluated, every other solve is
    /// reported unchanged with a warning.
    static func solvesReport(_ system: OpticalSystem) -> JSONValue {
        var outcomes: [JSONValue] = []
        var warnings: [JSONValue] = []
        for (k, sv) in system.solves.enumerated() {
            let previous = system.value(at: sv.target)
            var value = previous ?? .nan
            var method = "unchanged"
            if case .pickup(let source, let scale, let offset) = sv.kind, let v = system.value(at: source) {
                value = scale * v + offset
                method = value == previous ? "unchanged" : "closed_form"
            } else {
                warnings.append(.string("mock engine: \(sv.kind.kind.title) solve left unchanged"))
            }
            outcomes.append(
                .object([
                    "index": .integer(Int64(k)), "kind": .string(sv.kind.kind.rawValue),
                    "target": (try? JSONValue.from(sv.target)) ?? .null, "previous": previous.map(num) ?? .null,
                    "value": num(value), "residual": .number(0), "method": .string(method), "evaluations": .integer(1),
                ]))
        }
        return .object([
            "outcomes": .array(outcomes), "order": .array(system.solves.indices.map { .integer(Int64($0)) }),
            "sweeps": .integer(1), "max_change": .number(0), "warnings": .array(warnings),
            "solved_params": .array(system.solves.compactMap { try? JSONValue.from($0.target) }),
            "configuration": .null,
        ])
    }

    /// The mock's scene in the engine's buffer form (`layout`).
    static func layoutBuffers(_ scene: LayoutScene) -> (JSONValue, [ResultBuffer]) {
        var vertices: [Double] = [], offsets: [Double] = [0], kinds: [Double] = []
        var surf: [Double] = [], fld: [Double] = [], wav: [Double] = [], flags: [Double] = []
        for p in scene.polylines {
            for q in p.points { vertices += [q.x, q.y, q.z] }
            offsets.append(Double(vertices.count / 3))
            let kind: Double =
                switch p.kind {
                case .surfaceProfile: 0
                case .surfaceRim: 1
                case .elementEdge: 2
                case .stop: 3
                case .imagePlane: 4
                case .ray: 5
                case .axis: 6
                }
            kinds.append(kind)
            surf.append(Double(p.surface ?? -1))
            fld.append(Double(p.field ?? -1))
            wav.append(Double(p.wavelength ?? -1))
            flags.append((p.inSection ? 1 : 0) + (p.vignetted ? 2 : 0))
        }
        let n = kinds.count, v = vertices.count / 3
        let c = scene.bounds.center
        let surfaces = scene.surfaces.map { s -> JSONValue in
            .object([
                "index": .integer(Int64(s.index)), "id": .integer(Int64(s.id.rawValue)),
                "vertex": .array([num(s.vertex.x), num(s.vertex.y), num(s.vertex.z)]),
                "semi_diameter": num(s.semiDiameter), "semi_diameter_is_auto": .bool(s.semiDiameterIsAuto),
                "is_stop": .bool(false), "is_mirror": .bool(false), "label": .string(""),
            ])
        }
        let elements = scene.elements.map { e -> JSONValue in
            .object(["first_surface": .integer(Int64(e.firstSurface)), "last_surface": .integer(Int64(e.lastSurface))])
        }
        let json = JSONValue.object([
            "bounds": .object([
                "min": .array([num(scene.bounds.min.x), num(scene.bounds.min.y), num(scene.bounds.min.z)]),
                "max": .array([num(scene.bounds.max.x), num(scene.bounds.max.y), num(scene.bounds.max.z)]),
            ]),
            "origin": .array([num(c.x), num(c.y), num(c.z)]),
            "surfaces": .array(surfaces), "elements": .array(elements), "field_count": .integer(Int64(scene.fieldCount)),
            "wavelengths_um": .array(scene.wavelengthsUm.map(JSONValue.number)),
            "polyline_count": .integer(Int64(n)), "vertex_count": .integer(Int64(v)),
            "kind_names": .array(
                ["surface_profile", "surface_rim", "element_edge", "stop", "image_plane", "ray", "axis", "element_outline"].map(JSONValue.string)),
        ])
        return (
            json,
            [
                ResultBuffer(name: "vertices_f64", doubles: vertices, shape: [v, 3]),
                ResultBuffer(name: "offsets", doubles: offsets), ResultBuffer(name: "kinds", doubles: kinds),
                ResultBuffer(name: "surfaces", doubles: surf), ResultBuffer(name: "fields", doubles: fld),
                ResultBuffer(name: "wavelengths", doubles: wav), ResultBuffer(name: "flags", doubles: flags),
            ]
        )
    }

    // MARK: Design jobs

    /// The variables of an `optimize`/`tolerance` request: the listed ones,
    /// or every curvature and the back focus.
    static func variables(_ settings: [String: JSONValue], _ system: OpticalSystem) -> [ParamPath] {
        let listed = (settings["variables"]?.array ?? []).compactMap { v -> ParamPath? in
            guard let p = v["param"] else { return nil }
            return try? p.decode(ParamPath.self)
        }
        if !listed.isEmpty { return listed }
        var out = system.surfaces.dropFirst().dropLast().filter { $0.shape.hasCurvature && $0.shape.curvature != 0 }
            .map { ParamPath.curvature(surface: $0.id) }
        if system.surfaces.count >= 3 { out.append(.thickness(surface: system.surfaces[system.imageIndex - 1].id)) }
        return out
    }

    static func mockOptimize(
        _ s: [String: JSONValue], _ snap: ModelSnapshot, _ o: MockOptics, _ reporter: JobReporter, _ stop: MockStopFlag,
        _ latency: Duration
    ) async throws -> (JSONValue, [ResultBuffer]) {
        let algorithm = s["algorithm"]?.string ?? "lm"
        let iterations = max(s["max_iterations"]?.int ?? 12, 0)
        let vars = variables(s, snap.system)
        var spot = MockOptics.SpotDiagramSettings()
        spot.rings = 3
        let r = try o.spot(spot)
        let start = max(r.fields.map(\.rmsRadius).filter(\.isFinite).reduce(0, +) / Double(max(r.fields.count, 1)), 1e-6)
        var history: [JSONValue] = []
        var flat: [Double] = []
        var merit = start
        var done = 0
        var status = "converged"
        for it in 0..<iterations {
            try reporter.checkCancelled()
            if stop.isSet {
                status = "stopped"
                break
            }
            merit = start * (0.62 + 0.38 * exp(-Double(it + 1) / 4))
            done = it + 1
            history.append(
                .object([
                    "iteration": .integer(Int64(it)), "evaluations": .integer(Int64((it + 1) * (vars.count + 1))),
                    "merit": .number(merit), "best_merit": .number(merit), "violation": .number(0),
                ]))
            flat += [Double(it), Double((it + 1) * (vars.count + 1)), merit, merit]
            reporter.report(Double(it + 1) / Double(iterations), stage: "\(algorithm) \(it + 1): merit \(String(format: "%.4e", merit))")
            if latency > .zero { try await Task.sleep(for: latency / 8) }
        }
        // The "optimised" design: every variable nudged by a small, fixed step.
        var system = snap.system
        var variableRows: [JSONValue] = []
        let fraction = Double(done) / Double(max(iterations, 1))
        for (k, p) in vars.enumerated() {
            guard let v0 = system.value(at: p) else { continue }
            let v1 = v0 * (1 + 0.004 * fraction * (k % 2 == 0 ? 1 : -1))
            system.setValue(v1, at: p)
            variableRows.append(
                .object([
                    "label": .string(p.label(in: system)), "param": (try? JSONValue.from(p)) ?? .null,
                    "start": .number(v0), "value": .number(v1), "change": .number(v1 - v0), "lower": .null, "upper": .null,
                    "at_bound": .bool(false),
                ]))
        }
        let hash = try MockEngine.hash(system)
        let json = JSONValue.object([
            "status": .string(status), "stopped": .bool(status == "stopped"), "termination": .string(status == "stopped" ? "cancelled" : "small_merit_change"),
            "algorithm": .string(algorithm), "iterations": .integer(Int64(done)),
            "evaluations": .integer(Int64(done * (vars.count + 1))), "improved": .bool(done > 0),
            "start": .object(["merit": .number(start), "violation": .number(0)]),
            "final": .object(["merit": .number(done > 0 ? merit : start), "violation": .number(0), "feasible": .bool(true)]),
            "improvement": .number(done > 0 ? 1 - merit / start : 0), "variables": .array(variableRows),
            "operands": .array([]), "history": .array(history),
            "backend": .object(["requested": .string("auto"), "evaluated_on": .string("mock"), "precision": .string("f64_reference")]),
            "system": try JSONValue.from(system), "system_hash": .string(hash.hex),
        ])
        return (json, [ResultBuffer(name: "merit_history", doubles: flat, shape: [done, 4])])
    }

    /// `merit_evaluation`: the default merit is one RMS spot operand per
    /// field (the mean is the merit, as `optimize`'s start); a custom merit
    /// evaluates `efl` from the paraxial trace and every other operand as
    /// that RMS spot. The EFL hold is appended as a constraint.
    static func mockMeritEvaluation(_ s: [String: JSONValue], _ o: MockOptics) throws -> JSONValue {
        var spotSettings = MockOptics.SpotDiagramSettings()
        spotSettings.rings = 3
        let r = try o.spot(spotSettings)
        let spots = r.fields.map(\.rmsRadius).map { $0.isFinite ? $0 : 0 }
        let mean = max(spots.reduce(0, +) / Double(max(spots.count, 1)), 1e-6)
        let efl = o.efl(o.system.wavelengths.primary) ?? .nan
        var ops: [JSONValue]
        if let custom = s["merit"]?["operands"]?.array {
            ops = custom
        } else {
            let w = 1 / Double(max(spots.count, 1))
            ops = spots.indices.map { i in
                .object([
                    "kind": .object(["type": .string("rms_spot")]), "target": .number(0), "weight": .number(w), "comparison": .string("equal"),
                    "role": .string("residual"), "field": .object(["field": .string("index"), "index": .integer(Int64(i))]),
                    "wavelength": .object(["wavelength": .string("primary")]), "config": .integer(0), "enabled": .bool(true),
                ])
            }
        }
        let hasEFL = ops.contains { $0[path: "kind", "type"]?.string == "efl" }
        if s["hold_efl"]?.bool ?? true, !hasEFL, efl.isFinite {
            ops.append(
                .object([
                    "kind": .object(["type": .string("efl")]), "target": .number(efl), "weight": .number(1), "comparison": .string("equal"),
                    "role": .string("constraint"), "field": .object(["field": .string("all")]),
                    "wavelength": .object(["wavelength": .string("primary")]), "config": .integer(0), "enabled": .bool(true),
                    "label": .string("hold EFL"),
                ]))
        }
        var rows: [[String: JSONValue]] = []
        var total = 0.0
        for (k, op) in ops.enumerated() {
            let kind = op[path: "kind", "type"]?.string ?? ""
            let field = op[path: "field", "index"]?.int
            let value = kind == "efl" ? efl : field.map { spots.indices.contains($0) ? spots[$0] : mean } ?? mean
            let target = op["target"]?.nf ?? 0
            let weight = op["weight"]?.nf ?? 1
            let constraint = op["role"]?.string == "constraint"
            let enabled = op["enabled"]?.bool ?? true
            let d = value - target
            let residual: Double =
                switch op["comparison"]?.string {
                case "less_than": max(d, 0)
                case "greater_than": min(d, 0)
                default: d
                }
            let counted = !constraint && enabled
            let c = counted ? weight * residual * residual : .nan
            if counted { total += c }
            var row = op.object ?? [:]
            row["index"] = .integer(Int64(k))
            row["kind"] = .string(kind)
            row["label"] = op["label"] ?? .string("")
            row["start"] = num(value)
            row["value"] = num(value)
            row["residual"] = counted ? num(residual) : .null
            row["contribution"] = counted ? num(c) : .null
            row["error"] = .null
            rows.append(row)
        }
        for i in rows.indices {
            let c = rows[i]["contribution"]?.nf ?? .nan
            rows[i]["contribution_fraction"] = c.isFinite ? num(total > 0 ? c / total : 0) : .null
        }
        let spec = JSONValue.object([
            "schema_version": .integer(1), "name": .string("default"),
            "on_failure": .object(["mode": .string("penalty"), "value": .number(1000)]), "operands": .array(ops),
        ])
        return .object([
            "merit": num(total), "violation": .number(0), "status": .object(["status": .string("ok")]),
            "operands": .array(rows.map(JSONValue.object)), "merit_spec": spec,
            "backend": .object([
                "requested": .string("auto"), "evaluated_on": .string("mock"), "reported_on": .string("mock"), "precision": .string("f64_reference"),
            ]),
        ])
    }

    static func mockTolerance(
        _ s: [String: JSONValue], _ snap: ModelSnapshot, _ reporter: JobReporter, _ stop: MockStopFlag, _ latency: Duration
    ) async throws -> (JSONValue, [ResultBuffer]) {
        let system = snap.system
        let mode = s["mode"]?.string ?? "both"
        let trials = max(s["trials"]?.int ?? 100, 0)
        let seed = s["seed"]?.int.flatMap { UInt64(exactly: $0) } ?? 0
        var rng = SplitMix64(seed: seed &+ 0x9E37_79B9)
        let nominal = 1e-3
        var ops: [JSONValue] = []
        var worst: [(Int, Double)] = []
        // The requested tolerances (else the engine's basic set: every thickness).
        let requested: [(ParamPath, Double, Double, String)] = (s["tolerances"]?.array ?? []).compactMap { o in
            guard let p = o["param"].flatMap({ try? $0.decode(ParamPath.self) }) else { return nil }
            return (p, o["minus"]?.nf ?? 0.05, o["plus"]?.nf ?? 0.05, o["label"]?.string ?? "")
        }
        let params = requested.isEmpty
            ? system.surfaces.dropFirst().dropLast().map { (ParamPath.thickness(surface: $0.id), 0.05, 0.05, "") } : requested
        // Compensators sit at their nominal values and move a little per trial.
        let comps: [(ParamPath, Double)] = (s["compensators"]?.array ?? []).compactMap { c in
            guard let p = c["param"].flatMap({ try? $0.decode(ParamPath.self) }) else { return nil }
            return (p, system.value(at: p) ?? 0)
        }
        let nominalComps = JSONValue.array(comps.map { .number($0.1) })
        // Trial `t`'s perturbations: uniform within the limits, from its own stream.
        func draws(_ t: Int) -> [Double] {
            var r = SplitMix64(seed: seed &+ 0x5EED &+ UInt64(t) &* 0x9E37_79B9)
            return params.map { _, m, p, _ in -m + (m + p) * r.unit() }
        }
        func criterion(_ d: [Double]) -> Double {
            nominal * (1 + zip(d, params).map { x, p in let lim = max(p.1, p.2, 1e-12); return 0.05 * (x / lim) * (x / lim) }.reduce(0, +))
        }
        if let c = s["preview_case"], c != .null {
            let deltas: [Double]
            switch c["type"]?.string {
            case "trial": deltas = draws(max(c["trial"]?.int ?? 0, 0))
            default:
                let k = c["operand"]?.int ?? 0
                guard params.indices.contains(k) else { throw EngineError.invalidSettings(["preview operand out of range"]) }
                deltas = params.indices.map { $0 == k ? (c["positive"]?.bool ?? true ? params[k].2 : -params[k].1) : 0 }
            }
            var built = system
            for (d, p) in zip(deltas, params) where d != 0 {
                if let v = built.value(at: p.0) { built.setValue(v + d, at: p.0) }
            }
            let compensated = comps.map { $0.1 + 0.01 * (deltas.first ?? 0) / max(params.first?.1 ?? 1, 1e-12) }
            for (c, v) in zip(comps, compensated) { built.setValue(v, at: c.0) }
            let preview: JSONValue = .object([
                "system": (try? JSONValue.from(built)) ?? .null, "system_hash": .string("mock"), "criterion": .number(criterion(deltas)),
                "compensators": .array(compensated.map(JSONValue.number)), "perturbations": .array(deltas.map(JSONValue.number)),
            ])
            return (
                .object([
                    "status": .string("finished"), "stopped": .bool(false), "sensitivity": .null, "monte_carlo": .null, "preview": preview,
                    "compensators": s["compensators"] ?? .array([]),
                ]), []
            )
        }
        for (k, (p, minus, plus, label)) in params.enumerated() {
            try reporter.checkCancelled()
            let dm = nominal * 0.05 * rng.unit(), dp = nominal * 0.05 * rng.unit()
            worst.append((k, max(dm, dp)))
            ops.append(
                .object([
                    "index": .integer(Int64(k)), "label": .string(label.isEmpty ? p.label(in: system) : label),
                    "param": (try? JSONValue.from(p)) ?? .null,
                    "minus": .number(minus), "plus": .number(plus), "criterion_minus": .number(nominal + dm),
                    "criterion_plus": .number(nominal + dp), "change_minus": .number(dm), "change_plus": .number(dp),
                    "worst": .number(max(dm, dp)), "ok": .bool(true),
                    "compensators_minus": .array(comps.map { .number($0.1 - 0.01 * Double(k + 1)) }),
                    "compensators_plus": .array(comps.map { .number($0.1 + 0.02 * Double(k + 1)) }),
                ]))
            reporter.report(0.3 * Double(k + 1) / Double(max(params.count, 1)), stage: "tolerance sensitivity")
        }
        let rss = sqrt(worst.map { $0.1 * $0.1 }.reduce(0, +))
        var criteria: [Double] = []
        var records: [JSONValue] = []
        let recordTrials = s["record_trials"]?.int ?? 0
        var stopped = false
        let mcStart = Date()
        if mode != "sensitivity" {
            for t in 0..<trials {
                try reporter.checkCancelled()
                if stop.isSet {
                    stopped = true
                    break
                }
                let d = draws(t)
                criteria.append(criterion(d))
                if t < recordTrials {
                    records.append(
                        .object([
                            "trial": .integer(Int64(t)), "criterion": .number(criteria[t]), "ok": .bool(true),
                            "compensators": .array(comps.map { .number($0.1 + 0.001 * Double(t % 7)) }), "perturbations": .array(d.map(JSONValue.number)),
                        ]))
                }
                if t % 10 == 0 {
                    reporter.report(0.3 + 0.7 * Double(t) / Double(trials), stage: "tolerance monte carlo")
                    if latency > .zero { try await Task.sleep(for: latency / 20) }
                }
            }
        }
        let seconds = criteria.isEmpty ? 0 : Date().timeIntervalSince(mcStart) / Double(criteria.count)
        let sorted = criteria.sorted()
        // Linear between order statistics, as the engine.
        func pct(_ p: Double) -> Double {
            guard !sorted.isEmpty else { return .nan }
            let x = min(max(p / 100, 0), 1) * Double(sorted.count - 1)
            let i = Int(x.rounded(.down)), f = x - Double(i)
            return i + 1 < sorted.count ? sorted[i] * (1 - f) + sorted[i + 1] * f : sorted[i]
        }
        let levels = s["percentiles"]?.nfArray ?? [50, 90, 97.7]
        let mean = criteria.isEmpty ? Double.nan : criteria.reduce(0, +) / Double(criteria.count)
        let sd = criteria.isEmpty ? Double.nan : sqrt(criteria.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(criteria.count))
        let mc: JSONValue =
            mode == "sensitivity"
            ? .null
            : .object([
                "trials": .integer(Int64(criteria.count)), "failures": .integer(0), "seed": .integer(Int64(seed)),
                "nominal": .object(["criterion": .number(nominal), "ok": .bool(true), "compensators": nominalComps]),
                "seconds_per_trial": .number(seconds), "records": .array(records),
                "mean": num(mean), "std_dev": num(sd), "min": num(sorted.first ?? .nan), "max": num(sorted.last ?? .nan),
                "percentiles": .array(levels.map { .object(["percentile": .number($0), "value": num(pct($0))]) }),
                "yield_fraction": .null, "threshold": .null, "cancelled": .bool(stopped), "criteria": nums(criteria),
                "compensator_mean": .array(comps.map { .number($0.1 + 0.01) }), "compensator_std": .array(comps.map { _ in .number(0.05) }),
            ])
        let json = JSONValue.object([
            "status": .string(stopped ? "stopped" : "finished"), "stopped": .bool(stopped),
            "sensitivity": mode == "monte_carlo"
                ? .null
                : .object([
                    "nominal": .object(["criterion": .number(nominal), "ok": .bool(true), "compensators": nominalComps]), "operands": .array(ops),
                    "ranking": .array(worst.sorted { $0.1 > $1.1 }.map { .integer(Int64($0.0)) }),
                    "rss_change": .number(rss), "predicted_criterion": .number(nominal + rss),
                    "evaluations": .integer(Int64(2 * ops.count + 1)), "cancelled": .bool(false),
                ]),
            "monte_carlo": mc, "compensators": s["compensators"] ?? .array([]),
        ])
        return (json, criteria.isEmpty ? [] : [ResultBuffer(name: "monte_carlo_criteria", doubles: criteria)])
    }
}

/// A stop request of a mock design job.
final class MockStopFlag: Sendable {
    private let flag = Mutex(false)
    var isSet: Bool { flag.withLock { $0 } }
    func set() -> Bool {
        flag.withLock { $0 = true }
        return true
    }
}
