import Foundation
import MKLightEngine
import MKLightModel

/// Readers of the ghost and stray-light summaries (docs/RESULTS.md) for the
/// table beside their maps. The engine reports a ratio against a primary
/// without light on the map, and the standard errors of a run of a single
/// source ray, as `null`: undefined. Such a value is `nil` here and reads
/// "—" in the table.

/// `ghost_irradiance`: the ghost-to-signal ratios.
public struct GhostIrradianceSummary: Sendable, Equatable {
    public struct Ghost: Sendable, Equatable {
        public var label: String
        public var peakRatio: Double?
    }

    public var totalFluxRatio: Double?
    public var totalPeakRatio: Double?
    /// In the engine's rank, strongest first: ghosts with an undefined
    /// peak ratio lead, by their absolute peak irradiance.
    public var ghosts: [Ghost]

    public init(_ json: JSONValue) {
        totalFluxRatio = json["total_flux_ratio"]?.defined
        totalPeakRatio = json["total_peak_ratio"]?.defined
        ghosts = (json["ghosts"]?.array ?? []).map {
            Ghost(label: $0["label"]?.string ?? "", peakRatio: $0["peak_ratio"]?.defined)
        }
    }

    /// The totals, then the peak ratios of the `limit` strongest ghosts.
    public func rows(limit: Int = 6) -> [(String, String)] {
        [
            ("Ghost / primary flux", Fmt.defined(totalFluxRatio)),
            ("Ghost / primary peak", Fmt.defined(totalPeakRatio)),
        ] + ghosts.prefix(limit).map { ($0.label, Fmt.defined($0.peakRatio)) }
    }
}

/// `stray_light`: the stray total with its standard error and the flux of
/// every path class.
public struct StrayLightSummary: Sendable, Equatable {
    public struct ClassTotal: Sendable, Equatable {
        /// The engine's class name (`primary`, `ghost`, …).
        public var name: String
        public var flux: Double?
        public var stdError: Double?
    }

    public var flux: Double?
    public var stdError: Double?
    /// Stray flux / reference flux; `nil` without a reference flux.
    public var fraction: Double?
    public var totals: [ClassTotal]

    public init(_ json: JSONValue) {
        flux = json["stray_flux"]?.defined
        stdError = json["stray_std_error"]?.defined
        fraction = json["stray_fraction"]?.defined
        totals = (json["totals"]?.array ?? []).map {
            ClassTotal(name: $0["class"]?.string ?? "", flux: $0["flux"]?.defined, stdError: $0["std_error"]?.defined)
        }
    }

    public var rows: [(String, String)] {
        var rows = [
            ("Stray flux", Fmt.defined(flux, unit: "W")),
            ("± standard error", Fmt.defined(stdError, unit: "W")),
        ]
        if let fraction { rows.append(("Stray fraction", Fmt.short(fraction))) }
        return rows + totals.map { ($0.name.capitalized, Fmt.defined($0.flux, unit: "W")) }
    }
}

extension JSONValue {
    /// A number with a value: `nil` for `null`, a non-finite `nf` value or
    /// anything that is not a number.
    var defined: Double? { nf.flatMap { $0.isFinite ? $0 : nil } }
}

extension Fmt {
    /// `value(_:unit:)` of a number that may be undefined: "—" without a
    /// unit for `nil`.
    static func defined(_ v: Double?, unit: String = "") -> String {
        value(v ?? .nan, unit: unit)
    }
}
