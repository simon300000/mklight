import Foundation
import MKLightEngine

#if canImport(CMKLight)
import CMKLight
#endif

/// Job states of the C ABI (`MKJobState`).
enum JobState: Int32, Sendable {
    case queued = 0, running, done, failed, cancelled
    var isTerminal: Bool { self == .done || self == .failed || self == .cancelled }
}

/// Swift-typed wrappers of every C function the app uses (docs/FFI.md).
///
/// Each wrapper converts strings, frees what the library allocated, and turns
/// a failed `MKStatus` into an `EngineError` carrying the engine's message,
/// read from `mk_last_error()` on the same thread right after the call.
/// When the library is not linked (no `Frameworks/libmklight_ffi.a` at build
/// time) the same API exists and every call throws
/// `EngineError.libraryUnavailable`, so code and tests compile either way.
enum CABI {
    /// Thrown by every call when the library is not linked.
    static let notLinked = EngineError.libraryUnavailable(
        "libmklight_ffi.a was not linked into this build; run scripts/build-macos.sh (or --engine-only) and rebuild")

    #if canImport(CMKLight)
    static let isLinked = true

    /// `(major << 16) | minor` of the linked library.
    static var abiVersion: UInt32 { mk_abi_version() }
    /// The major version this app was compiled against.
    static var compiledABIMajor: UInt32 { UInt32(MK_ABI_VERSION_MAJOR) }
    static var engineVersion: String { String(cString: mk_engine_version()) }

    // MARK: Status and strings

    private static func check(_ status: MKStatus, _ function: String) throws {
        guard status != MK_OK else { return }
        let message = String(cString: mk_last_error())
        guard let code = EngineErrorCode(rawValue: status) else {
            throw EngineError.engine(.internal, "\(function): unknown status \(status): \(message)")
        }
        throw EngineError.from(code: code, message: message.isEmpty ? "\(function) failed" : message)
    }

    /// Calls `body` with an out-string, checks the status and takes ownership.
    private static func string(
        _ function: String, _ body: (UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>) -> MKStatus
    ) throws -> String {
        var out: UnsafeMutablePointer<CChar>?
        let status = body(&out)
        defer { mk_string_free(out) }
        try check(status, function)
        guard let out else { return "" }
        return String(cString: out)
    }

    // MARK: Engine

    static func engineCreate(config: String?) throws -> OpaquePointer {
        var engine: OpaquePointer?
        let status = withOptionalCString(config) { mk_engine_create($0, &engine) }
        try check(status, "mk_engine_create")
        guard let engine else { throw EngineError.engine(.internal, "mk_engine_create returned no engine") }
        return engine
    }

    static func engineDestroy(_ e: OpaquePointer) { mk_engine_destroy(e) }

    static func listAnalyses(_ e: OpaquePointer) throws -> String {
        try string("mk_engine_list_analyses") { mk_engine_list_analyses(e, $0) }
    }

    static func listBackends(_ e: OpaquePointer) throws -> String {
        try string("mk_engine_list_backends") { mk_engine_list_backends(e, $0) }
    }

    /// Installs (`hub != nil`) or clears the job callback. Events go to
    /// `JobEventHub.deliver` on the engine's `mklight-events` thread. When
    /// this returns after clearing, no thread is inside the old callback.
    static func setJobCallback(_ e: OpaquePointer, hub: JobEventHub?) throws {
        guard let hub else {
            try check(mk_engine_set_job_callback(e, nil, nil), "mk_engine_set_job_callback")
            return
        }
        let callback: MKJobCallback = { userData, job, state, progress, stage in
            guard let userData else { return }
            let hub = Unmanaged<JobEventHub>.fromOpaque(userData).takeUnretainedValue()
            hub.deliver(
                job,
                JobEvent(
                    state: JobState(rawValue: state) ?? .running, progress: progress,
                    stage: stage.map { String(cString: $0) } ?? ""))
        }
        let user = Unmanaged.passUnretained(hub).toOpaque()
        try check(mk_engine_set_job_callback(e, callback, user), "mk_engine_set_job_callback")
    }

    // MARK: Sessions

    static func sessionNew(_ e: OpaquePointer) throws -> OpaquePointer {
        var s: OpaquePointer?
        try check(mk_session_new(e, &s), "mk_session_new")
        guard let s else { throw EngineError.engine(.internal, "mk_session_new returned no session") }
        return s
    }

    static func sessionOpen(_ e: OpaquePointer, path: String) throws -> OpaquePointer {
        var s: OpaquePointer?
        try check(mk_session_open(e, path, &s), "mk_session_open")
        guard let s else { throw EngineError.engine(.internal, "mk_session_open returned no session") }
        return s
    }

    static func sessionDestroy(_ s: OpaquePointer) { mk_session_destroy(s) }

    static func sessionSave(_ s: OpaquePointer, path: String?) throws {
        try check(withOptionalCString(path) { mk_session_save(s, $0) }, "mk_session_save")
    }

    static func sessionInfo(_ s: OpaquePointer) throws -> String {
        try string("mk_session_info_json") { mk_session_info_json(s, $0) }
    }

    static func sessionProject(_ s: OpaquePointer) throws -> String {
        try string("mk_session_get_project_json") { mk_session_get_project_json(s, $0) }
    }

    static func sessionSystem(_ s: OpaquePointer) throws -> String {
        try string("mk_session_get_system_json") { mk_session_get_system_json(s, $0) }
    }

    /// Returns the new snapshot hash.
    static func sessionSetSystem(_ s: OpaquePointer, json: String) throws -> String {
        try string("mk_session_set_system_json") { mk_session_set_system_json(s, json, $0) }
    }

    /// Returns the new snapshot hash.
    static func sessionApplyEdits(_ s: OpaquePointer, json: String) throws -> String {
        try string("mk_session_apply_param_edit") { mk_session_apply_param_edit(s, json, $0) }
    }

    static func sessionSetSection(_ s: OpaquePointer, section: String, json: String) throws {
        try check(mk_session_set_project_section_json(s, section, json), "mk_session_set_project_section_json")
    }

    static func sessionSnapshotHash(_ s: OpaquePointer) throws -> String {
        try string("mk_session_current_snapshot_hash") { mk_session_current_snapshot_hash(s, $0) }
    }

    static func sessionLibraryHash(_ s: OpaquePointer) throws -> String {
        try string("mk_session_current_library_hash") { mk_session_current_library_hash(s, $0) }
    }

    static func sessionLoadCatalog(_ s: OpaquePointer, path: String) throws -> String {
        try string("mk_session_load_catalog") { mk_session_load_catalog(s, path, $0) }
    }

    static func sessionListCatalogs(_ s: OpaquePointer) throws -> String {
        try string("mk_session_list_catalogs") { mk_session_list_catalogs(s, $0) }
    }

    static func sessionListGlasses(_ s: OpaquePointer, catalog: String?) throws -> String {
        try string("mk_session_list_glasses") { out in withOptionalCString(catalog) { mk_session_list_glasses(s, $0, out) } }
    }

    /// `mk_session_apply_solves`: the solve report and the new snapshot hash.
    static func sessionApplySolves(_ s: OpaquePointer, settings: String?) throws -> (report: String, hash: String) {
        var report: UnsafeMutablePointer<CChar>?
        var hash: UnsafeMutablePointer<CChar>?
        let status = withOptionalCString(settings) { mk_session_apply_solves(s, $0, &report, &hash) }
        defer {
            mk_string_free(report)
            mk_string_free(hash)
        }
        try check(status, "mk_session_apply_solves")
        return (report.map { String(cString: $0) } ?? "{}", hash.map { String(cString: $0) } ?? "")
    }

    // MARK: Jobs

    static func submit(_ s: OpaquePointer, requestJSON: String) throws -> UInt64 {
        var job: UInt64 = 0
        try check(mk_session_submit_analysis_json(s, requestJSON, &job), "mk_session_submit_analysis_json")
        return job
    }

    static func jobStatus(_ e: OpaquePointer, _ job: UInt64) throws -> (state: JobState, progress: Double) {
        var state: MKJobState = 0
        var progress = 0.0
        try check(mk_job_status(e, job, &state, &progress), "mk_job_status")
        return (JobState(rawValue: state) ?? .running, progress)
    }

    static func jobStatusJSON(_ e: OpaquePointer, _ job: UInt64) throws -> String {
        try string("mk_job_status_json") { mk_job_status_json(e, job, $0) }
    }

    static func jobCancel(_ e: OpaquePointer, _ job: UInt64) throws {
        try check(mk_job_cancel(e, job), "mk_job_cancel")
    }

    /// Blocks the calling thread (`timeoutMs < 0`: forever). Never call it
    /// on the main thread.
    static func jobWait(_ e: OpaquePointer, _ job: UInt64, timeoutMs: Int64) throws -> JobState {
        var state: MKJobState = 0
        try check(mk_job_wait(e, job, timeoutMs, &state), "mk_job_wait")
        return JobState(rawValue: state) ?? .running
    }

    /// Takes the result of a finished job: the envelope JSON and its
    /// buffers, wrapped without copying (`ResultBuffer` keeps the array
    /// alive and `mk_buffer_free` runs when the last one goes).
    static func jobTakeResult(_ e: OpaquePointer, _ job: UInt64) throws -> (json: String, buffers: [ResultBuffer]) {
        var json: UnsafeMutablePointer<CChar>?
        var array: UnsafeMutablePointer<MKBuffer>?
        var count = 0
        let status = mk_job_take_result(e, job, &json, &array, &count)
        defer { mk_string_free(json) }
        if status != MK_OK {
            mk_buffer_free(array, count)
            try check(status, "mk_job_take_result")
        }
        let text = json.map { String(cString: $0) } ?? "{}"
        guard let array, count > 0 else { return (text, []) }
        let owner = BufferArray(array, count)
        let buffers = (0..<count).map { i -> ResultBuffer in
            let b = array[i]
            let shape = (0..<Int(b.ndim)).map { b.shape[$0] }
            return ResultBuffer(
                name: b.name.map { String(cString: $0) } ?? "buffer \(i)",
                dtype: BufferDType(rawValue: b.dtype) ?? .u8,
                shape: shape, count: BufferDType(rawValue: b.dtype) == nil ? 0 : b.len,
                base: UnsafeRawPointer(b.data), owner: owner)
        }
        return (text, buffers)
    }

    static func jobRelease(_ e: OpaquePointer, _ job: UInt64) throws {
        try check(mk_job_release(e, job), "mk_job_release")
    }

    /// Owns a buffer array from `mk_job_take_result`.
    private final class BufferArray {
        let array: UnsafeMutablePointer<MKBuffer>
        let count: Int
        init(_ array: UnsafeMutablePointer<MKBuffer>, _ count: Int) {
            self.array = array
            self.count = count
        }
        deinit { mk_buffer_free(array, count) }
    }

    private static func withOptionalCString<R>(_ s: String?, _ body: (UnsafePointer<CChar>?) -> R) -> R {
        guard let s else { return body(nil) }
        return s.withCString { body($0) }
    }

    #else
    static let isLinked = false
    static var abiVersion: UInt32 { 0 }
    static var compiledABIMajor: UInt32 { 0 }
    static var engineVersion: String { "" }

    static func engineCreate(config: String?) throws -> OpaquePointer { throw notLinked }
    static func engineDestroy(_ e: OpaquePointer) {}
    static func listAnalyses(_ e: OpaquePointer) throws -> String { throw notLinked }
    static func listBackends(_ e: OpaquePointer) throws -> String { throw notLinked }
    static func setJobCallback(_ e: OpaquePointer, hub: JobEventHub?) throws { throw notLinked }
    static func sessionNew(_ e: OpaquePointer) throws -> OpaquePointer { throw notLinked }
    static func sessionOpen(_ e: OpaquePointer, path: String) throws -> OpaquePointer { throw notLinked }
    static func sessionDestroy(_ s: OpaquePointer) {}
    static func sessionSave(_ s: OpaquePointer, path: String?) throws { throw notLinked }
    static func sessionInfo(_ s: OpaquePointer) throws -> String { throw notLinked }
    static func sessionProject(_ s: OpaquePointer) throws -> String { throw notLinked }
    static func sessionSystem(_ s: OpaquePointer) throws -> String { throw notLinked }
    static func sessionSetSystem(_ s: OpaquePointer, json: String) throws -> String { throw notLinked }
    static func sessionApplyEdits(_ s: OpaquePointer, json: String) throws -> String { throw notLinked }
    static func sessionSetSection(_ s: OpaquePointer, section: String, json: String) throws { throw notLinked }
    static func sessionSnapshotHash(_ s: OpaquePointer) throws -> String { throw notLinked }
    static func sessionLibraryHash(_ s: OpaquePointer) throws -> String { throw notLinked }
    static func sessionLoadCatalog(_ s: OpaquePointer, path: String) throws -> String { throw notLinked }
    static func sessionListCatalogs(_ s: OpaquePointer) throws -> String { throw notLinked }
    static func sessionListGlasses(_ s: OpaquePointer, catalog: String?) throws -> String { throw notLinked }
    static func submit(_ s: OpaquePointer, requestJSON: String) throws -> UInt64 { throw notLinked }
    static func sessionApplySolves(_ s: OpaquePointer, settings: String?) throws -> (report: String, hash: String) {
        throw notLinked
    }
    static func jobStatus(_ e: OpaquePointer, _ job: UInt64) throws -> (state: JobState, progress: Double) {
        throw notLinked
    }
    static func jobStatusJSON(_ e: OpaquePointer, _ job: UInt64) throws -> String { throw notLinked }
    static func jobCancel(_ e: OpaquePointer, _ job: UInt64) throws { throw notLinked }
    static func jobWait(_ e: OpaquePointer, _ job: UInt64, timeoutMs: Int64) throws -> JobState { throw notLinked }
    static func jobTakeResult(_ e: OpaquePointer, _ job: UInt64) throws -> (json: String, buffers: [ResultBuffer]) {
        throw notLinked
    }
    static func jobRelease(_ e: OpaquePointer, _ job: UInt64) throws { throw notLinked }
    #endif
}
