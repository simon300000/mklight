import Foundation
import MKLightEngine

extension AnalysisResultView {
    /// Footer key numbers of the field maps, the glass map and the image
    /// simulation.
    static func extendedKeyNumbers(analysis: String, output: AnalysisOutput) -> [KeyNumber] {
        switch analysis {
        case "pupil_aberration":
            guard let d = FieldMapData(output: output) else { return [] }
            var out: [KeyNumber] = []
            if let x = d.columnIndex("stop_error_x"), let y = d.columnIndex("stop_error_y") {
                let e = (0..<d.count).map { hypot(d.value($0, x), d.value($0, y)) }.filter(\.isFinite)
                if let m = e.max() { out.append(KeyNumber(label: "Max stop error", value: Fmt.length(m))) }
            }
            let v = ["variation_x", "variation_y"].compactMap(d.columnIndex).flatMap { d.column($0) }.filter(\.isFinite)
            if let m = v.max(by: { abs($0) < abs($1) }) { out.append(KeyNumber(label: "Max variation", value: String(format: "%+.2f %%", m * 100))) }
            return out
        case "relative_illumination_map":
            guard let d = FieldMapData(output: output) else { return [] }
            var out: [KeyNumber] = []
            if let c = d.columnIndex("relative_illumination"), let r = d.range(c) {
                out.append(KeyNumber(label: "Min RI", value: String(format: "%.3f", r.lowerBound)))
            }
            if let c = d.columnIndex("vignetting_fraction"), let r = d.range(c), r.upperBound > 0 {
                out.append(KeyNumber(label: "Max vignetting", value: Fmt.percent(r.upperBound)))
            }
            return out
        case "chief_ray_angle":
            guard let d = FieldMapData(output: output), let c = d.columnIndex("incidence"), let k = d.argmax(c) else { return [] }
            var out = [
                KeyNumber(label: "Max CRA", value: String(format: "%.2f°", d.value(k, c))),
                KeyNumber(label: "at", value: Fmt.value(d.imageRadius[k], unit: "mm")),
            ]
            if let m = d.columnIndex("margin"), let r = d.range(m) {
                out.append(KeyNumber(label: "Min margin", value: String(format: "%+.2f°", r.lowerBound)))
            }
            return out
        case "glass_map":
            guard let d = GlassMapData(output: output) else { return [] }
            return [
                KeyNumber(label: "Glasses", value: "\(d.glasses.filter { $0.kind == .catalog }.count)"),
                KeyNumber(label: "In design", value: "\(d.glasses.filter(\.inDesign).count)"),
            ]
        case "image_simulation":
            guard let d = ImageSimulationData(output: output) else { return [] }
            var out: [KeyNumber] = []
            if d.pixelSize.x.isFinite { out.append(KeyNumber(label: "Pixel", value: Fmt.length(d.pixelSize.x))) }
            if let m = d.maxDistortionPercent { out.append(KeyNumber(label: "Max distortion", value: String(format: "%+.3f %%", m))) }
            if let r = d.minRelativeIllumination { out.append(KeyNumber(label: "Min RI", value: String(format: "%.3f", r))) }
            return out
        default:
            return []
        }
    }
}
