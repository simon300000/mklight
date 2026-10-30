import SwiftUI

#if canImport(AppKit)
import AppKit
#elseif canImport(UIKit)
import UIKit
#endif

/// The small set of platform-specific pieces the UI needs, so views and view
/// models stay the same on macOS and iPadOS.
enum PlatformColor {
    /// A colour that follows the light or dark appearance.
    static func dynamic(light: SIMD3<Double>, dark: SIMD3<Double>) -> Color {
        #if canImport(AppKit)
        let c = NSColor(name: nil) { appearance in
            let v = appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? dark : light
            return NSColor(srgbRed: v.x, green: v.y, blue: v.z, alpha: 1)
        }
        return Color(nsColor: c)
        #elseif canImport(UIKit)
        let c = UIColor { traits in
            let v = traits.userInterfaceStyle == .dark ? dark : light
            return UIColor(red: v.x, green: v.y, blue: v.z, alpha: 1)
        }
        return Color(uiColor: c)
        #else
        return Color(.sRGB, red: light.x, green: light.y, blue: light.z)
        #endif
    }
}

extension View {
    /// The resize pointer over a split divider (macOS); nothing elsewhere.
    @ViewBuilder
    func resizePointer(vertical: Bool) -> some View {
        #if os(macOS)
        self.pointerStyle(vertical ? .rowResize : .columnResize)
        #else
        self
        #endif
    }

    /// Table-like numeric text: monospaced digits.
    func numeric() -> some View { monospacedDigit() }
}

/// Spacing scale used across the app (points).
enum Space {
    static let xxs: CGFloat = 2
    static let xs: CGFloat = 4
    static let s: CGFloat = 6
    static let m: CGFloat = 8
    static let l: CGFloat = 12
    static let xl: CGFloat = 16
    static let xxl: CGFloat = 24
}
