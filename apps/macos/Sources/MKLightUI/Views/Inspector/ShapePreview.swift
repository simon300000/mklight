import Charts
import MKLightEngine
import MKLightModel
import SwiftUI

/// The inspector's picture of a surface shape: sag profiles (x and y for
/// anamorphic and freeform shapes), a false-colour departure map for
/// freeforms, and for a diffractive surface its phase profile, paraxial
/// power and smallest zone period. Display geometry from `ShapeSag`; the
/// engine traces.
struct ShapePreview: View {
    let shape: SurfaceShape
    /// Clear semi-diameter (fixed, or the engine's automatic one).
    let semiDiameter: Double?
    /// The system's primary wavelength (µm), for the DOE's power.
    let primaryWavelengthUm: Double

    private var type: SurfaceTypeDescriptor { SurfaceTypes.descriptor(for: shape) }
    private var h: Double { SurfaceTypes.previewRadius(shape, semiDiameter: semiDiameter) }

    var body: some View {
        switch type.preview {
        case .radial:
            profiles([.y])
        case .anamorphic:
            profiles([.x, .y])
        case .freeform:
            VStack(alignment: .leading, spacing: Space.m) {
                profiles([.x, .y])
                DepartureMap(shape: shape, radius: h)
            }
        case .diffractive:
            VStack(alignment: .leading, spacing: Space.m) {
                if case .diffractive(let base, let phase) = shape {
                    PhaseProfile(phase: phase, radius: h)
                    DOEFigures(phase: phase, radius: h, primaryWavelengthUm: primaryWavelengthUm)
                    if base.kind != .plane { profiles([.y], title: "Base sag") }
                }
            }
        case .idealLens, .coordinateBreak:
            EmptyView()
        }
    }

    private func profiles(_ axes: [SurfaceShape.CylinderAxis], title: String = "Sag") -> some View {
        let series: [Series] = axes.compactMap { axis in
            SurfaceTypes.profile(shape, semiDiameter: h, axis: axis, samples: 96).map { pts in
                Series(
                    id: axis.rawValue, label: axis == .x ? "x section" : "y section",
                    color: axis == .x ? Palette.fieldColor(1) : Palette.fieldColor(0), points: pts, dashed: axis == .x)
            }
        }
        return VStack(alignment: .leading, spacing: Space.xs) {
            if series.count > 1 || title != "Sag" {
                HStack(spacing: Space.m) {
                    Text(title).font(.caption.weight(.semibold))
                    Spacer()
                    if series.count > 1 {
                        ForEach(series) { s in LegendItem(color: s.color, label: s.label, dashed: s.dashed) }
                    }
                }
            }
            if series.isEmpty {
                Text("The shape has no real sag over ±\(Fmt.short(h)) mm.").font(.caption).foregroundStyle(.secondary)
            } else {
                LineChart(series: series, xLabel: "Position (mm)", yLabel: "Sag (mm)", xDomain: -h...h)
                    .frame(height: 116)
                    .font(.caption2)
                    .accessibilityLabel(accessibilityProfile(series))
            }
        }
    }

    private func accessibilityProfile(_ series: [Series]) -> String {
        series.map { s in
            let z = s.points.map(\.1)
            return "\(s.label): sag from \(Fmt.short(z.min() ?? 0)) to \(Fmt.short(z.max() ?? 0)) millimetres over ±\(Fmt.short(h)) millimetres"
        }.joined(separator: "; ")
    }
}

/// The departure from the base conic over the clear aperture, blue
/// (negative) to red (positive), with its peak-to-valley.
struct DepartureMap: View {
    let shape: SurfaceShape
    let radius: Double

    var body: some View {
        let n = 64
        let cells = SurfaceTypes.departureMap(shape, semiDiameter: radius, n: n)
        let values = cells.map { $0 ?? .nan }
        // Row 0 of a Grid2D is the bottom; the map's first row is the top.
        let flipped = (0..<n).flatMap { j in values[((n - 1 - j) * n)..<((n - j) * n)] }
        let finite = values.filter(\.isFinite)
        let peak = max(abs(finite.min() ?? 0), abs(finite.max() ?? 0))
        let pv = (finite.max() ?? 0) - (finite.min() ?? 0)
        VStack(alignment: .leading, spacing: Space.xs) {
            HStack {
                Text("Departure from base").font(.caption.weight(.semibold))
                Spacer()
                Text("PV \(Fmt.length(pv))").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            HStack(alignment: .center, spacing: Space.m) {
                HeatMapImage(
                    grid: Grid2D(nx: n, ny: n, values: Array(flipped)), scale: .diverging,
                    range: -max(peak, 1e-15)...max(peak, 1e-15)
                )
                .aspectRatio(1, contentMode: .fit)
                .clipShape(Circle())
                .overlay(Circle().strokeBorder(.separator, lineWidth: 0.5))
                .overlay(alignment: .topLeading) { axisLabel("y ↑") }
                .overlay(alignment: .bottomTrailing) { axisLabel("x →") }
                .frame(maxHeight: 132)
                ColorBar(scale: .diverging, range: -max(peak, 1e-15)...max(peak, 1e-15), unit: "mm")
                    .frame(width: 44, height: 120)
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Departure map from the base conic")
        .accessibilityValue("Peak to valley \(Fmt.length(pv)) over a radius of \(Fmt.short(radius)) millimetres")
    }

    private func axisLabel(_ s: String) -> some View {
        Text(s).font(.caption2).foregroundStyle(.secondary).padding(2)
    }
}

/// Phase Φ(r) of a DOE in waves (Φ/2π) over the clear aperture.
struct PhaseProfile: View {
    let phase: DiffractivePhase
    let radius: Double

    var body: some View {
        let pts = (0...96).map { k -> (Double, Double) in
            let r = radius * Double(k) / 96
            return (r, ShapeSag.phase(phase, r: r) / (2 * .pi))
        }
        VStack(alignment: .leading, spacing: Space.xs) {
            Text("Phase (order +1)").font(.caption.weight(.semibold))
            LineChart(
                series: [Series(id: "phase", label: "Φ / 2π", color: Palette.fieldColor(6), points: pts)],
                xLabel: "Radius (mm)", yLabel: "Φ (waves)", xDomain: 0...radius
            )
            .frame(height: 116)
            .font(.caption2)
            .accessibilityLabel("Phase profile")
            .accessibilityValue("From 0 to \(Fmt.short(pts.last?.1 ?? 0)) waves at a radius of \(Fmt.short(radius)) millimetres")
        }
    }
}

/// The DOE's paraxial power (from A₁, SURFACES §3), its focal length and
/// the smallest zone period over the clear aperture.
struct DOEFigures: View {
    let phase: DiffractivePhase
    let radius: Double
    let primaryWavelengthUm: Double

    var body: some View {
        let trace = ShapeSag.phasePower(phase, order: phase.traceOrder, wavelengthUm: primaryWavelengthUm)
        let design = ShapeSag.phasePower(phase, order: phase.designOrder, wavelengthUm: phase.designWavelengthUm)
        let period = ShapeSag.minimumPeriod(phase, upTo: radius)
        let zones = abs(ShapeSag.phase(phase, r: radius)) / (2 * .pi)
        let sameAsDesign = phase.traceOrder == phase.designOrder && abs(primaryWavelengthUm - phase.designWavelengthUm) < 1e-9
        VStack(alignment: .leading, spacing: Space.s) {
            figure(
                "Paraxial power", "order \(phase.traceOrder), \(Fmt.short(primaryWavelengthUm)) µm",
                trace == 0 ? "0" : "\(Fmt.short(trace)) mm⁻¹", trace == 0 ? nil : "f \(Fmt.short(1 / trace)) mm")
            if !sameAsDesign {
                figure(
                    "At design", "order \(phase.designOrder), \(Fmt.short(phase.designWavelengthUm)) µm",
                    design == 0 ? "0" : "\(Fmt.short(design)) mm⁻¹", design == 0 ? nil : "f \(Fmt.short(1 / design)) mm")
            }
            figure(
                "Smallest zone period", period.map { "at r = \(Fmt.short($0.radius)) mm" } ?? "",
                period.map { Fmt.length($0.period) } ?? "—", nil)
            figure("Zones", "over ±\(Fmt.short(radius)) mm", String(format: "%.1f", zones), nil)
        }
        .textSelection(.enabled)
        .help("Paraxial power φ = −m λ A₁ / (π r_norm²), added to the base's refractive power. A DOE's Abbe number is about −3.45.")
    }

    private func figure(_ title: String, _ detail: String, _ value: String, _ extra: String?) -> some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 0) {
                Text(title).font(.callout)
                if !detail.isEmpty { Text(detail).font(.caption).foregroundStyle(.secondary) }
            }
            Spacer(minLength: Space.s)
            VStack(alignment: .trailing, spacing: 0) {
                Text(value).font(.callout.monospacedDigit().weight(.medium))
                if let extra { Text(extra).font(.caption.monospacedDigit()).foregroundStyle(.secondary) }
            }
            .fixedSize()
        }
        .accessibilityElement(children: .combine)
    }
}
