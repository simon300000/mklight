import Foundation
import MKLightEngine

/// Presentation of engine analyses: short titles, symbols and category
/// names. Analyses the app does not know keep the engine's title and get a
/// generic symbol, so new analyses appear without code changes.
public enum AnalysisCatalog {
    static let titles: [String: String] = [
        "first_order": String(localized: "First-Order Data"), "system_summary": String(localized: "System Data"), "validate": String(localized: "Validation"),
        "spot_diagram": String(localized: "Spot Diagram"), "ray_fan": String(localized: "Ray Fan"), "opd_fan": String(localized: "OPD Fan"),
        "field_curvature_distortion": String(localized: "Field Curvature & Distortion"), "grid_distortion": String(localized: "Grid Distortion"),
        "seidel": String(localized: "Seidel Coefficients"), "longitudinal_aberration": String(localized: "Longitudinal Aberration"),
        "chromatic_focal_shift": String(localized: "Chromatic Focal Shift"), "lateral_colour": String(localized: "Lateral Color"),
        "relative_illumination": String(localized: "Relative Illumination"), "footprint": String(localized: "Footprint"), "encircled_energy": String(localized: "Encircled Energy"),
        "geometric_mtf": String(localized: "Geometric MTF"), "best_focus": String(localized: "Best Focus"), "ray_grid": String(localized: "Ray Grid"), "layout": String(localized: "Layout"),
        "wavefront_map": String(localized: "Wavefront Map"), "psf": String(localized: "PSF"), "mtf": String(localized: "MTF"), "mtf_vs_field": String(localized: "MTF vs Field"),
        "through_focus_mtf": String(localized: "Through-Focus MTF"), "diffraction_encircled_energy": String(localized: "Diffraction Encircled Energy"),
        "line_spread": String(localized: "Line Spread"), "ghost_paths": String(localized: "Ghost Paths"), "ghost_irradiance": String(localized: "Ghost Irradiance"),
        "stray_light": String(localized: "Stray Light"), "polarization_pupil_map": String(localized: "Polarization Pupil Map"),
        "system_transmission": String(localized: "System Transmission"), "glass_info": String(localized: "Glass Data"), "solves": String(localized: "Solves"),
        "optimize": String(localized: "Optimize"), "tolerance": String(localized: "Tolerance"), "rng_stream": String(localized: "Random Stream"),
        "pupil_aberration": String(localized: "Pupil Aberration"), "relative_illumination_map": String(localized: "Relative Illumination Map"),
        "chief_ray_angle": String(localized: "Chief Ray Angle"), "glass_map": String(localized: "Glass Map"), "image_simulation": String(localized: "Image Simulation"),
        "glass_substitution": String(localized: "Glass Substitution"),
    ]

    /// Abbreviations for narrow panes (the full title stays in the help
    /// and the VoiceOver label).
    static let shortTitles: [String: String] = [
        "first_order": String(localized: "First Order"), "system_summary": String(localized: "System"), "spot_diagram": String(localized: "Spot"),
        "field_curvature_distortion": String(localized: "Field Curv. & Dist."), "grid_distortion": String(localized: "Grid Dist."), "seidel": String(localized: "Seidel"),
        "longitudinal_aberration": String(localized: "Long. Aberration"), "chromatic_focal_shift": String(localized: "Focal Shift"),
        "relative_illumination": String(localized: "Rel. Illumination"), "encircled_energy": String(localized: "Encircled En."), "geometric_mtf": String(localized: "Geom. MTF"),
        "wavefront_map": String(localized: "Wavefront"), "through_focus_mtf": String(localized: "Thru-Focus MTF"), "diffraction_encircled_energy": String(localized: "Diffr. Enc. En."),
        "ghost_irradiance": String(localized: "Ghost Irrad."), "polarization_pupil_map": String(localized: "Pol. Pupil Map"), "system_transmission": String(localized: "Transmission"),
        "pupil_aberration": String(localized: "Pupil Aberr."), "relative_illumination_map": String(localized: "RI Map"), "chief_ray_angle": String(localized: "CRA"),
        "image_simulation": String(localized: "Image Sim."), "glass_substitution": String(localized: "Glass Subst."),
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
        "model": String(localized: "System"), "materials": String(localized: "Materials"), "geometric": String(localized: "Geometric"), "wave": String(localized: "Wave Optics"),
        "radiometry": String(localized: "Radiometry"), "polarization": String(localized: "Polarization"), "ghost": String(localized: "Ghosts"), "stray": String(localized: "Stray Light"),
        "optimization": String(localized: "Design"), "diagnostic": String(localized: "Diagnostics"),
    ]

    static let categoryOrder = ["geometric", "wave", "model", "radiometry", "polarization", "ghost", "stray", "materials"]

    /// Known analyses have short titles; others keep the engine's, in title case.
    public static func title(of d: AnalysisDescriptor) -> String { titles[d.name] ?? EngineText.translation(d.title) ?? titleCase(d.title) }

    static func titleCase(_ s: String) -> String {
        let small: Set<String> = ["a", "an", "and", "at", "by", "for", "in", "of", "on", "or", "the", "to", "vs"]
        return s.split(separator: " ").enumerated().map { k, w in
            let word = String(w)
            if k > 0 && small.contains(word.lowercased()) { return word.lowercased() }
            return word.prefix(1).uppercased() + word.dropFirst()
        }.joined(separator: " ")
    }
    public static func title(of name: String) -> String { titles[name] ?? name.replacingOccurrences(of: "_", with: " ").capitalized }
    /// An abbreviated title for narrow panes; the full title when there is none.
    public static func shortTitle(of name: String) -> String { shortTitles[name] ?? title(of: name) }
    public static func symbol(for name: String) -> String { symbols[name] ?? "chart.dots.scatter" }
    public static func categoryTitle(_ c: String) -> String { categoryTitles[c] ?? c.capitalized }

    /// Analyses grouped by category, known categories first.
    public static func grouped(_ list: [AnalysisDescriptor]) -> [(category: String, items: [AnalysisDescriptor])] {
        let groups = Dictionary(grouping: list, by: \.category)
        let order = categoryOrder + groups.keys.filter { !categoryOrder.contains($0) }.sorted()
        return order.compactMap { c in groups[c].map { (c, $0) } }
    }
}
