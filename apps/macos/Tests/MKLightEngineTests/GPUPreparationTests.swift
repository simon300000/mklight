import Foundation
import Testing

@testable import MKLightEngine

/// `mk_engine_gpu_preparation_json`'s states, in the engine's exact JSON.
@Suite struct GPUPreparationTests {
    private func decode(_ json: String) throws -> GPUPreparation {
        try JSONDecoder().decode(GPUPreparation.self, from: Data(json.utf8))
    }

    @Test func decodesEveryState() throws {
        #expect(try decode(#"{"state":"idle"}"#) == .idle)
        let preparing = try decode(#"{"state":"preparing","device":"Apple M2 Pro","done":41,"total":205}"#)
        #expect(preparing == .preparing(device: "Apple M2 Pro", done: 41, total: 205))
        #expect(preparing.isPreparing)
        #expect(abs(preparing.fraction - 0.2) < 1e-12)
        #expect(try decode(#"{"state":"ready","device":"Apple M2 Pro","kernels":205,"from_cache":205,"seconds":0.005}"#)
            == .ready(device: "Apple M2 Pro", kernels: 205, fromCache: 205, seconds: 0.005))
        #expect(try decode(#"{"state":"unavailable","reason":"no device"}"#) == .unavailable(reason: "no device"))
        #expect(try decode(#"{"state":"failed","device":"Apple M2 Pro","message":"x"}"#)
            == .failed(device: "Apple M2 Pro", message: "x"))
    }

    @Test func totalNotYetKnownIsZeroProgress() throws {
        let s = try decode(#"{"state":"preparing","device":"","done":0,"total":0}"#)
        #expect(s.isPreparing && s.fraction == 0)
    }
}
