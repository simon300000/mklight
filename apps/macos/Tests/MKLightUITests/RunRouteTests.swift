import Foundation
import Testing

@testable import MKLightEngine
@testable import MKLightModel
@testable import MKLightUI

/// The run details' route line, tier and attempt rows, on the engine's
/// `backend.cost_decision` of each kind (docs/RESULTS.md §5) and its
/// `run_info.failed_attempts`.
@Suite struct RunRouteTests {
    private func line(_ json: String, requested: String = "default:auto") throws -> String? {
        RunRoute.line(requested: requested, decision: try MKLightJSON.decode(JSONValue.self, from: Data(json.utf8)))
    }

    /// Decisions of the cost model, as recorded from the engine on an M2 Pro.
    static let costModelCPU = """
        {"advantage":0.006893,"before_device":true,"calibration":{"device_class":"Apple M2 Pro","source":"built-in"},
         "decision":"cpu_reference","predicted_ms":{"cpu":0.7971,"metal":115.6},
         "refinement":{"cpu":{"contention":1.248,"decade":-1,"factor":1.248},"family":"53773f4a64870a1e","metal":{"decade":0,"factor":1}},
         "required":{"gain_ms":5,"ratio":1.3},"rule":"cost model",
         "stages_ms":{"cpu":{"fixed":0.3949,"output":0.03587,"prep":0.1488,"ray":0.05936,"table":0,"transform":0},
                      "metal":{"fixed":1.562,"output":0.0266,"prep":0.4388,"ray":0.04042,"table":0,"transform":0},
                      "metal_fallback_risk":0,"metal_setup":113.6},
         "state":{"aim_traces_per_ray":0,"calls":0,"decided_pupil_grid":null,"device_open":false,"lattice_warm":false,
                  "resident":false,"snapshot_seen":false,"tracer_warm":false,"transient_failure_rate":0},"version":5}
        """
    static func costModelGPU(repeated: Bool = false) -> String {
        """
        {"advantage":1.782,"before_device":false,"calibration":{"device_class":"Apple M2 Pro","source":"built-in"},
         "decision":"qualified_metal","predicted_ms":{"cpu":556.6,"metal":312.3},
         "refinement":{"cpu":{"contention":1.605,"decade":2,"factor":1.605},"family":"53773f4a64870a1e","metal":{"decade":2,"factor":1}},
         \(repeated ? #""repeated":true,"# : "")"required":{"gain_ms":5,"ratio":1.3},"rule":"cost model",
         "stages_ms":{"cpu":{"fixed":0.302,"output":149.6,"prep":0,"ray":196.8,"table":0,"transform":0},
                      "metal":{"fixed":1.562,"output":100.8,"prep":0,"ray":153.1,"table":0,"transform":0},
                      "metal_fallback_risk":0,"metal_setup":56.79},
         "state":{"aim_traces_per_ray":0,"calls":1,"decided_pupil_grid":null,"device_open":false,"lattice_warm":false,
                  "resident":false,"snapshot_seen":true,"tracer_warm":true,"transient_failure_rate":0},"version":5}
        """
    }

    @Test func aCostModelDecisionNamesTheBackendAndBothPredictedTimes() throws {
        #expect(try line(Self.costModelGPU()) == "GPU by predicted time (GPU predicted 312 ms, CPU 557 ms)")
        // A time that rounds to nothing is not "0 ms".
        #expect(try line(Self.costModelCPU) == "CPU by predicted time (GPU predicted 116 ms, CPU 1 ms)")
        #expect(
            try line(#"{"decision":"cpu_reference","rule":"cost model","predicted_ms":{"cpu":0.2,"metal":1840},"version":5}"#)
                == "CPU by predicted time (GPU predicted 1.8 s, CPU <1 ms)")
    }

    /// The GPU is chosen only for a clear advantage (`required`): a GPU
    /// predicted a little faster still leaves the run on the CPU.
    @Test func aGPUGainBelowTheThresholdIsSaidSo() throws {
        let json = #"""
            {"advantage":1.2,"before_device":false,"decision":"cpu_reference","predicted_ms":{"cpu":12,"metal":10},
             "required":{"gain_ms":5,"ratio":1.3},"rule":"cost model","version":5}
            """#
        #expect(try line(json) == "CPU: the GPU's predicted gain is too small (GPU predicted 10 ms, CPU 12 ms)")
    }

    @Test func aRepeatedDecisionSaysTheRouteWasKept() throws {
        #expect(
            try line(Self.costModelGPU(repeated: true))
                == "GPU by predicted time (GPU predicted 312 ms, CPU 557 ms); kept from the first identical request")
    }

    @Test func decisionsWithoutAPredictionSayWhy() throws {
        #expect(
            try line(
                #"{"decision":"cpu_reference","rule":"accuracy exclusion","evidence":"the GPU search returns another finalist glass than the f64 reference (finalist merit 3x outside its gate)","version":5,"before_device":true}"#
            ) == "CPU: this analysis is not accurate enough on the GPU")
        // As the engine sends it for a small `stray_light` run.
        #expect(
            try line(
                #"{"before_device":true,"decision":"cpu_reference","reason":"stray_light: the transport gives the device no source of this run (too few sources, or surfaces it does not intersect in closed form)","rule":"no device share","version":5}"#
            ) == "CPU: the GPU takes no part of a run of this size and shape")
        #expect(
            try line(#"{"decision":"cpu_reference","rule":"known failure","version":5,"before_device":false}"#)
                == "CPU: this request already failed on the GPU")
        #expect(
            try line(#"{"decision":"qualified_metal","reason":"no cost calibration for this device","version":5,"before_device":false}"#)
                == "GPU: no timing data for this device, so no prediction")
        #expect(
            try line(#"{"decision":"qualified_metal","reason":"no calibration for this analysis","version":5,"before_device":false}"#)
                == "GPU: no timing data for this analysis, so no prediction")
        #expect(try line(#"{"decision":"qualified_metal","reason":"no cost decision","version":5}"#) == "GPU, without a prediction")
    }

    /// The route is the router's: an explicit request has none, whatever
    /// the engine records for it.
    @Test func anExplicitRequestHasNoRouteLine() throws {
        let bypassed = #"{"decision":"explicit_backend","reason":"cost rules bypassed"}"#
        #expect(try line(bypassed, requested: "gpu:metal") == nil)
        #expect(try line(bypassed) == nil)
        #expect(try line(Self.costModelGPU(), requested: "gpu:metal") == nil)
        #expect(try line(Self.costModelGPU(), requested: "cpu_reference") == nil)
        #expect(try line(Self.costModelGPU(), requested: "auto") != nil, "auto asked for by name")
    }

    @Test func unknownAndMissingShapesShowNothing() throws {
        #expect(RunRoute.line(requested: "default:auto", decision: nil) == nil)
        #expect(RunRoute.line(requested: "default:auto", decision: .null) == nil)
        #expect(try line("{}") == nil)
        #expect(try line(#"{"rule":"cost model"}"#) == nil)
        #expect(try line(#"{"decision":"qualified_vulkan","rule":"cost model"}"#) == nil)
        #expect(try line(#"{"decision":"cpu_reference","rule":"moon phase"}"#) == nil)
        #expect(try line(#"{"decision":"cpu_reference"}"#) == nil)
        #expect(try line(#"["cpu_reference"]"#) == nil)
        // Without both finite times, the decision alone.
        #expect(try line(#"{"decision":"qualified_metal","rule":"cost model","predicted_ms":{"cpu":3}}"#) == "GPU by predicted time")
        #expect(
            try line(#"{"decision":"cpu_reference","rule":"cost model","predicted_ms":{"cpu":"3","metal":-1}}"#)
                == "CPU by predicted time")
    }

    @Test func onlyATierThatIsNotDesignIsNamed() {
        #expect(RunRoute.tier(.design) == nil)
        #expect(RunRoute.tier(.preview) == "Preview: cheaper sampling, not for final decisions")
    }

    /// One row per attempt, with the engine's reason: a GPU run that failed
    /// and was completed on the CPU reads as that.
    @Test func attemptsReadAsWhatHappened() {
        let rows = RunRoute.attempts([
            BackendAttempt(
                backend: "gpu:metal", outcome: "failed", reason: "native spot_diagram failed after 12.3 ms: device lost", elapsedMs: 12.3),
            BackendAttempt(
                backend: "gpu:metal", outcome: "skipped",
                reason: "native spot_diagram skipped: the same input failed on this device after 12.3 ms (device lost)", elapsedMs: 0),
            BackendAttempt(
                backend: "gpu:metal", outcome: "reference_completion",
                reason: "the native path completed the whole result on the CPU f64 reference", elapsedMs: 1840),
            BackendAttempt(backend: "vulkan", outcome: "out_of_luck", reason: "why", elapsedMs: 3),
        ])
        #expect(
            rows.map(\.title) == [
                "GPU failed after 12 ms; completed on the CPU",
                "GPU not tried: the same input already failed on it",
                "GPU path completed the result on the CPU (1.8 s)",
                "vulkan not used (out of luck)",
            ])
        #expect(rows[0].reason == "native spot_diagram failed after 12.3 ms: device lost")
        #expect(rows.map(\.id) == [0, 1, 2, 3])
        #expect(RunRoute.attempts([]).isEmpty)
        // An attempt that never started has no time.
        #expect(
            RunRoute.attempts([BackendAttempt(backend: "gpu:metal", outcome: "failed", reason: "r", elapsedMs: 0)])[0].title
                == "GPU failed; completed on the CPU")
    }

    @Test func durationsReadAsTheBadgeShowsThem() {
        #expect(Fmt.duration(ms: 4.2) == "4 ms" && Fmt.duration(ms: 999.4) == "999 ms")
        #expect(Fmt.duration(ms: 1000) == "1.0 s" && Fmt.duration(ms: 31_250) == "31.2 s")
    }
}
