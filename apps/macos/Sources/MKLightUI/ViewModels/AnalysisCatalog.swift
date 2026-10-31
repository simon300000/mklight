import Foundation
import MKLightEngine

/// Presentation of engine analyses: short titles, symbols and category
/// names. Analyses the app does not know keep the engine's title and get a
/// generic symbol, so new analyses appear without code changes.
public enum AnalysisCatalog {
    static let titles: [String: String] = [
        "first_order": "First-Order Data", "system_summary": "System Data", "validate": "Validation",
        "spot_diagram": "Spot Diagram", "ray_fan": "Ray Fan", "opd_fan": "OPD Fan",
        "field_curvature_distortion": "Field Curvature & Distortion", "grid_distortion": "Grid Distortion",
        "seidel": "Seidel Coefficients", "longitudinal_aberration": "Longitudinal Aberration",
        "chromatic_focal_shift": "Chromatic Focal Shift", "lateral_colour": "Lateral Color",
        "relative_illumination": "Relative Illumination", "footprint": "Footprint", "encircled_energy": "Encircled Energy",
        "geometric_mtf": "Geometric MTF", "best_focus": "Best Focus", "ray_grid": "Ray Grid", "layout": "Layout",
        "wavefront_map": "Wavefront Map", "psf": "PSF", "mtf": "MTF", "mtf_vs_field": "MTF vs Field",
        "through_focus_mtf": "Through-Focus MTF", "diffraction_encircled_energy": "Diffraction Encircled Energy",
        "line_spread": "Line Spread", "ghost_paths": "Ghost Paths", "ghost_irradiance": "Ghost Irradiance",
        "stray_light": "Stray Light", "polarization_pupil_map": "Polarization Pupil Map",
        "system_transmission": "System Transmission", "glass_info": "Glass Data", "solves": "Solves",
        "optimize": "Optimize", "tolerance": "Tolerance", "rng_stream": "Random Stream",
        "pupil_aberration": "Pupil Aberration", "relative_illumination_map": "Relative Illumination Map",
        "chief_ray_angle": "Chief Ray Angle", "glass_map": "Glass Map", "image_simulation": "Image Simulation",
        "glass_substitution": "Glass Substitution",
    ]

    static let symbols: [String: String] = [
        "first_order": "list.bullet.rectangle", "system_summary": "tablecells", "validate": "checkmark.seal",
        "spot_diagram": "circle.dotted", "ray_fan": "chart.xyaxis.line", "opd_fan": "waveform.path.ecg",
        "field_curvature_distortion": "point.topleft.down.to.point.bottomright.curvepath", "grid_distortion": "grid",
        "seidel": "chart.bar", "longitudinal_aberration": "arrow.left.and.right", "chromatic_focal_shift": "rainbow",
        "lateral_colour": "paintpalette", "relative_illumination": "sun.max", "footprint": "circle.grid.cross",
        "encircled_energy": "circle.circle", "geometric_mtf": "chart.line.downtrend.xyaxis", "best_focus": "scope",
        "ray_grid": "circle.grid.3x3", "wavefront_map": "water.waves", "psf": "sparkle", "mtf": "chart.line.downtrend.xyaxis",
        "mtf_vs_field": "chart.line.flattrend.xyaxis", "through_focus_mtf": "chart.line.uptrend.xyaxis",
        "diffraction_encircled_energy": "circle.circle", "line_spread": "waveform", "ghost_paths": "arrow.triangle.swap",
        "ghost_irradiance": "aqi.medium", "stray_light": "light.max", "polarization_pupil_map": "circle.hexagongrid",
        "system_transmission": "line.diagonal.arrow", "glass_info": "drop", "solves": "function",
        "pupil_aberration": "circle.dashed.inset.filled", "relative_illumination_map": "sun.max.circle",
        "chief_ray_angle": "arrow.down.right.circle", "glass_map": "chart.dots.scatter", "image_simulation": "photo",
    ]

    static let categoryTitles: [String: String] = [
        "model": "System", "materials": "Materials", "geometric": "Geometric", "wave": "Wave Optics",
        "radiometry": "Radiometry", "polarization": "Polarization", "ghost": "Ghosts", "stray": "Stray Light",
        "optimization": "Design", "diagnostic": "Diagnostics",
    ]

    static let categoryOrder = ["geometric", "wave", "model", "radiometry", "polarization", "ghost", "stray", "materials"]

    /// Known analyses have short titles; others keep the engine's, in title case.
    public static func title(of d: AnalysisDescriptor) -> String { titles[d.name] ?? titleCase(d.title) }

    static func titleCase(_ s: String) -> String {
        let small: Set<String> = ["a", "an", "and", "at", "by", "for", "in", "of", "on", "or", "the", "to", "vs"]
        return s.split(separator: " ").enumerated().map { k, w in
            let word = String(w)
            if k > 0 && small.contains(word.lowercased()) { return word.lowercased() }
            return word.prefix(1).uppercased() + word.dropFirst()
        }.joined(separator: " ")
    }
    public static func title(of name: String) -> String { titles[name] ?? name.replacingOccurrences(of: "_", with: " ").capitalized }
    public static func symbol(for name: String) -> String { symbols[name] ?? "chart.dots.scatter" }
    public static func categoryTitle(_ c: String) -> String { categoryTitles[c] ?? c.capitalized }

    /// Analyses grouped by category, known categories first.
    public static func grouped(_ list: [AnalysisDescriptor]) -> [(category: String, items: [AnalysisDescriptor])] {
        let groups = Dictionary(grouping: list, by: \.category)
        let order = categoryOrder + groups.keys.filter { !categoryOrder.contains($0) }.sorted()
        return order.compactMap { c in groups[c].map { (c, $0) } }
    }
}
