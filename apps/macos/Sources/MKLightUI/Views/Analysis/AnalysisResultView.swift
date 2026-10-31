import MKLightEngine
import MKLightModel
import SwiftUI

struct KeyNumber {
    var label: String
    var value: String
}

/// Shows a result with the dedicated view of its analysis when there is
/// one and it can read the result; every other result is shown by the
/// schema-agnostic generic view. Never fails: an unreadable result is data.
struct AnalysisResultView: View {
    let analysis: String
    let output: AnalysisOutput
    let context: AnalysisContext

    var body: some View {
        Group {
            if let v = dedicated { v } else { GenericResultView(output: output) }
        }
        // Plots use the same decimal point as the app's number formats.
        .environment(\.locale, NumberDisplay.locale)
    }

    private var dedicated: AnyView? {
        let o = output, c = context
        switch analysis {
        case "first_order":
            return FirstOrderView(output: o, system: c.system).map { AnyView($0) }
        case "validate": return AnyView(ValidateResultView(output: o, context: c))
        case "spot_diagram": return SpotDiagramView(output: o, context: c).map { AnyView($0) }
        case "ray_fan", "opd_fan": return FanView(output: o, context: c).map { AnyView($0) }
        case "field_curvature_distortion": return FieldCurvatureView(output: o, context: c).map { AnyView($0) }
        case "grid_distortion": return GridDistortionView(output: o).map { AnyView($0) }
        case "seidel": return SeidelView(output: o, context: c).map { AnyView($0) }
        case "longitudinal_aberration": return LongitudinalView(output: o, context: c).map { AnyView($0) }
        case "chromatic_focal_shift": return ChromaticFocalShiftView(output: o).map { AnyView($0) }
        case "lateral_colour": return LateralColourView(output: o, context: c).map { AnyView($0) }
        case "relative_illumination": return RelativeIlluminationView(output: o, context: c).map { AnyView($0) }
        case "footprint": return FootprintView(output: o, context: c).map { AnyView($0) }
        case "encircled_energy", "diffraction_encircled_energy": return EncircledEnergyView(output: o, context: c).map { AnyView($0) }
        case "geometric_mtf", "mtf": return MTFView(output: o, context: c).map { AnyView($0) }
        case "mtf_vs_field": return MTFVsFieldView(output: o, context: c).map { AnyView($0) }
        case "through_focus_mtf": return ThroughFocusMTFView(output: o, context: c).map { AnyView($0) }
        case "wavefront_map": return WavefrontView(output: o, context: c).map { AnyView($0) }
        case "psf": return PSFView(output: o, context: c).map { AnyView($0) }
        case "line_spread": return LineSpreadView(output: o, context: c).map { AnyView($0) }
        case "ghost_paths": return GhostPathsView(output: o, context: c).map { AnyView($0) }
        case "ghost_irradiance": return GhostIrradianceView(output: o).map { AnyView($0) }
        case "stray_light": return StrayLightView(output: o).map { AnyView($0) }
        case "polarization_pupil_map": return PolarizationView(output: o).map { AnyView($0) }
        case "system_transmission": return TransmissionView(output: o, context: c).map { AnyView($0) }
        case "best_focus": return BestFocusView(output: o, context: c).map { AnyView($0) }
        case "pupil_aberration": return PupilAberrationView(output: o).map { AnyView($0) }
        case "relative_illumination_map": return RelativeIlluminationMapView(output: o).map { AnyView($0) }
        case "chief_ray_angle": return ChiefRayAngleView(output: o).map { AnyView($0) }
        case "glass_map": return GlassMapView(output: o, context: c).map { AnyView($0) }
        case "image_simulation": return ImageSimulationView(output: o, pane: c.pane).map { AnyView($0) }
        default: return nil
        }
    }

    /// The footer's key numbers for a result.
    static func keyNumbers(analysis: String, output: AnalysisOutput, context: AnalysisContext) -> [KeyNumber] {
        let j = output.json
        switch analysis {
        case "first_order":
            let d = FirstOrderData(j)
            return [("EFL", "efl"), ("F/#", "working_f_number"), ("BFL", "bfl")].compactMap { l, k in
                d?.entry([k]).map { KeyNumber(label: l, value: $0.formatted) }
            }
        case "spot_diagram":
            let rms = (j["fields"]?.array ?? []).compactMap { $0["rms_radius"]?.nf }.filter(\.isFinite)
            var out: [KeyNumber] = []
            if let m = rms.max() { out.append(KeyNumber(label: "Max RMS", value: Fmt.length(m))) }
            if let a = j["airy_radius"]?.nf, a.isFinite { out.append(KeyNumber(label: "Airy", value: Fmt.length(a))) }
            return out
        case "field_curvature_distortion":
            let d = (j["curves"]?.array ?? []).flatMap { $0["distortion_percent"]?.nfArray ?? [] }.filter(\.isFinite)
            return d.isEmpty ? [] : [KeyNumber(label: "Max distortion", value: String(format: "%.3f %%", d.map(abs).max()!))]
        case "grid_distortion":
            return j["max_abs_distortion_percent"]?.nf.map { [KeyNumber(label: "Max distortion", value: String(format: "%.3f %%", $0))] } ?? []
        case "seidel":
            return [("Σ S_I", "s1"), ("Σ S_II", "s2"), ("Σ S_III", "s3")].compactMap { l, k in
                j[path: "sums", k]?.nf.map { KeyNumber(label: l, value: Fmt.short($0)) }
            }
        case "chromatic_focal_shift":
            return j["range"]?.nf.map { [KeyNumber(label: "Range", value: Fmt.length($0))] } ?? []
        case "wavefront_map":
            guard let m = j["maps"]?.array?.first else { return [] }
            return [
                m["rms_waves"]?.nf.map { KeyNumber(label: "RMS", value: String(format: "%.4f λ", $0)) },
                m["pv_waves"]?.nf.map { KeyNumber(label: "P-V", value: String(format: "%.4f λ", $0)) },
                m["strehl_at_reference"]?.nf.map { KeyNumber(label: "Strehl", value: String(format: "%.3f", $0)) },
            ].compactMap { $0 }
        case "psf":
            return (j["psfs"]?.array?.first?["strehl"]?.nf).map { [KeyNumber(label: "Strehl", value: String(format: "%.3f", $0))] } ?? []
        case "ghost_irradiance":
            return j["total_flux_ratio"]?.nf.map { [KeyNumber(label: "Ghost/primary flux", value: Fmt.short($0))] } ?? []
        case "stray_light":
            return j["stray_fraction"]?.nf.map { [KeyNumber(label: "Stray fraction", value: Fmt.short($0))] } ?? []
        case "best_focus":
            return j["defocus"]?.nf.map { [KeyNumber(label: "Best defocus", value: Fmt.length($0))] } ?? []
        case "pupil_aberration", "relative_illumination_map", "chief_ray_angle", "glass_map", "image_simulation":
            return extendedKeyNumbers(analysis: analysis, output: output)
        case "validate":
            return [KeyNumber(label: "Errors", value: "\(j["errors"]?.int ?? 0)"), KeyNumber(label: "Warnings", value: "\(j["warnings"]?.int ?? 0)")]
        default:
            return []
        }
    }
}

/// `first_order`: the keyed report as a two-column table.
struct FirstOrderView: View {
    let data: FirstOrderData
    let json: JSONValue
    let xy: ParaxialXYData?
    /// The design is not rotationally symmetric (the engine's `symmetry`;
    /// the surface kinds of `system` only when it does not say): show its
    /// x / y first-order data.
    let anamorphic: Bool

    init?(output: AnalysisOutput, system: OpticalSystem? = nil) {
        guard let d = FirstOrderData(output.json) else { return nil }
        data = d
        json = output.json
        xy = ParaxialXYData(output.json)
        anamorphic = ParaxialXYData.isAnamorphic(output.json, system: system)
    }

    var body: some View {
        ScrollView {
            if anamorphic {
                VStack(alignment: .leading, spacing: Space.s) {
                    Text("Anamorphic").font(.subheadline.weight(.semibold))
                    AnamorphicSummary(json: json, detailed: true)
                }
                .padding(Space.m)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.fill.quinary, in: .rect(cornerRadius: 10))
                .padding([.horizontal, .top], Space.l)
            } else if let p = xy?.primary, !p.doePower.isEmpty {
                VStack(alignment: .leading, spacing: Space.s) {
                    Text("Diffractive").font(.subheadline.weight(.semibold))
                    DOEPowerGrid(row: p)
                }
                .padding(Space.m)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.fill.quinary, in: .rect(cornerRadius: 10))
                .padding([.horizontal, .top], Space.l)
            }
            Grid(alignment: .leading, horizontalSpacing: Space.xl, verticalSpacing: Space.s) {
                ForEach(data.entries) { e in
                    GridRow {
                        Text(e.label).foregroundStyle(.secondary)
                        Text(Fmt.short(e.value)).numeric().gridColumnAlignment(.trailing)
                        Text(e.unit).foregroundStyle(.secondary)
                    }
                    .accessibilityElement(children: .combine)
                }
            }
            .font(.callout)
            .textSelection(.enabled)
            .padding(Space.l)
            .frame(maxWidth: .infinity, alignment: .leading)
            if let w = data.wavelengthUm {
                Text(String(format: "At λ = %.4f µm", w)).font(.caption).foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, Space.l)
            }
        }
    }
}

/// `validate`: the issues of the analysed snapshot.
struct ValidateResultView: View {
    let output: AnalysisOutput
    let context: AnalysisContext

    var body: some View {
        let issues = ValidationIssue.from(validate: output.json)
        if issues.isEmpty {
            ContentUnavailableView("No Issues", systemImage: "checkmark.seal", description: Text("The design passed every check."))
        } else {
            List(issues) { i in
                Button {
                    if let s = i.surface { context.selectSurface(s) }
                } label: {
                    IssueRow(issue: i, system: context.system)
                }
                .buttonStyle(.plain)
            }
            .listStyle(.plain)
        }
    }
}
