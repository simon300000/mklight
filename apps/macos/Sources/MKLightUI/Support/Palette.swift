import AppKit
import SwiftUI

/// Colours for wavelengths, fields and the PSF heat map. Display only.
public enum Palette {
    /// Approximate perceived colour of a wavelength (µm). Visible wavelengths
    /// follow the spectrum; UV and IR get distinct fallback hues.
    public static func wavelengthRGB(_ um: Double) -> SIMD3<Double> {
        let nm = um * 1000
        if nm < 380 { return SIMD3(0.55, 0.25, 0.85) }
        if nm > 780 { return SIMD3(0.55, 0.12, 0.10) }
        var r = 0.0, g = 0.0, b = 0.0
        switch nm {
        case ..<440: (r, g, b) = (-(nm - 440) / 60, 0, 1)
        case ..<490: (r, g, b) = (0, (nm - 440) / 50, 1)
        case ..<510: (r, g, b) = (0, 1, -(nm - 510) / 20)
        case ..<580: (r, g, b) = ((nm - 510) / 70, 1, 0)
        case ..<645: (r, g, b) = (1, -(nm - 645) / 65, 0)
        default: (r, g, b) = (1, 0, 0)
        }
        // Darken towards the ends of the visible range and keep yellows readable.
        let edge: Double = nm < 420 ? 0.3 + 0.7 * (nm - 380) / 40 : (nm > 700 ? 0.3 + 0.7 * (780 - nm) / 80 : 1)
        let v = SIMD3(r, g, b) * edge
        return v * (g > 0.8 && r > 0.6 ? 0.82 : 1)
    }

    public static func wavelengthColor(_ um: Double) -> Color {
        let c = wavelengthRGB(um)
        return Color(.sRGB, red: c.x, green: c.y, blue: c.z)
    }

    /// Categorical colours for fields (distinguishable, colour-blind aware).
    static let fieldRGB: [SIMD3<Double>] = [
        SIMD3(0.00, 0.45, 0.70), SIMD3(0.00, 0.62, 0.45), SIMD3(0.84, 0.37, 0.00),
        SIMD3(0.80, 0.47, 0.65), SIMD3(0.35, 0.70, 0.90), SIMD3(0.90, 0.62, 0.00),
        SIMD3(0.40, 0.40, 0.40), SIMD3(0.60, 0.30, 0.20),
    ]

    public static func fieldRGBValue(_ i: Int) -> SIMD3<Double> { fieldRGB[((i % fieldRGB.count) + fieldRGB.count) % fieldRGB.count] }

    public static func fieldColor(_ i: Int) -> Color {
        let c = fieldRGBValue(i)
        return Color(.sRGB, red: c.x, green: c.y, blue: c.z)
    }

    /// Perceptually uniform heat-map colour for `t` in 0…1 (viridis control
    /// points, linearly interpolated).
    public static func heat(_ t: Double) -> SIMD3<Double> {
        let stops: [SIMD3<Double>] = [
            SIMD3(0.267, 0.005, 0.329), SIMD3(0.283, 0.141, 0.458), SIMD3(0.254, 0.265, 0.530),
            SIMD3(0.207, 0.372, 0.553), SIMD3(0.164, 0.471, 0.558), SIMD3(0.128, 0.567, 0.551),
            SIMD3(0.135, 0.659, 0.518), SIMD3(0.267, 0.749, 0.441), SIMD3(0.478, 0.821, 0.318),
            SIMD3(0.741, 0.873, 0.150), SIMD3(0.993, 0.906, 0.144),
        ]
        let x = (t.isFinite ? min(max(t, 0), 1) : 0) * Double(stops.count - 1)
        let i = min(Int(x), stops.count - 2)
        let f = x - Double(i)
        return stops[i] + (stops[i + 1] - stops[i]) * f
    }
}
