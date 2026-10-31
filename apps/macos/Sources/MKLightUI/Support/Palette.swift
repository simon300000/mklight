import SwiftUI

/// Data colours for plots and the viewport. Display only.
///
/// Series use a fixed, validated categorical order (light and dark steps of
/// the same eight hues; adjacent pairs pass the colour-vision checks), never
/// cycled hues. Wavelengths keep their physical meaning: each maps to the
/// categorical slot nearest its spectral colour, so F is blue, d is aqua and
/// C is red everywhere in the app. Magnitude maps use one hue (blue), light
/// to dark; signed maps (OPD) use blue ↔ neutral ↔ red.
public enum Palette {
    /// Categorical slots: blue, orange, aqua, yellow, magenta, green, violet, red.
    static let lightSlots: [UInt32] = [0x2a78d6, 0xeb6834, 0x1baf7a, 0xeda100, 0xe87ba4, 0x008300, 0x6250d6, 0xe34948]
    static let darkSlots: [UInt32] = [0x3987e5, 0xd95926, 0x199e70, 0xc98500, 0xd55181, 0x008300, 0x9085e9, 0xe66767]

    static func rgb(_ hex: UInt32) -> SIMD3<Double> {
        SIMD3(Double((hex >> 16) & 0xff), Double((hex >> 8) & 0xff), Double(hex & 0xff)) / 255
    }

    static func color(_ c: SIMD3<Double>) -> Color { Color(.sRGB, red: c.x, green: c.y, blue: c.z) }

    /// Categorical slot `i` (clamped into the eight; callers fold more
    /// series into small multiples).
    public static func slotRGB(_ i: Int, dark: Bool) -> SIMD3<Double> {
        let slots = dark ? darkSlots : lightSlots
        return rgb(slots[min(max(i, 0), slots.count - 1)])
    }

    /// Field `i`'s colour (fields take the slots in order).
    public static func fieldRGB(_ i: Int, dark: Bool) -> SIMD3<Double> { slotRGB(i % 8, dark: dark) }

    public static func fieldColor(_ i: Int) -> Color { dynamic { fieldRGB(i, dark: $0) } }

    /// Categorical slot `i` as a colour (6 is violet: DOE marks).
    public static func slotColor(_ i: Int) -> Color { dynamic { slotRGB(i, dark: $0) } }

    /// The slot nearest the wavelength's spectral colour: violet, blue,
    /// aqua, green, yellow, orange, red; infrared and ultraviolet take red
    /// and violet.
    static func wavelengthSlot(_ um: Double) -> Int {
        switch um * 1000 {
        case ..<450: 6  // violet (g line and below)
        case ..<500: 0  // blue (F, F')
        case ..<540: 2  // aqua
        case ..<565: 5  // green (e)
        case ..<600: 3  // yellow (d)
        case ..<630: 1  // orange
        default: 7  // red (C, C') and the infrared
        }
    }

    public static func wavelengthRGB(_ um: Double, dark: Bool) -> SIMD3<Double> {
        slotRGB(wavelengthSlot(um), dark: dark)
    }

    public static func wavelengthColor(_ um: Double) -> Color { dynamic { wavelengthRGB(um, dark: $0) } }

    /// Sequential blue ramp, steps 100…700.
    static let blueRamp: [UInt32] = [0xcde2fb, 0x9ec5f4, 0x6da7ec, 0x3987e5, 0x256abf, 0x184f95, 0x0d366b]

    /// Magnitude `t` in 0…1: near zero recedes into the plot surface.
    public static func sequential(_ t: Double, dark: Bool) -> SIMD3<Double> {
        let surface = dark ? rgb(0x1a1a19) : rgb(0xfcfcfb)
        let stops = [surface] + (dark ? blueRamp.reversed().map(rgb) : blueRamp.map(rgb))
        return interpolate(stops, t)
    }

    /// The perceptual sequential ramp (viridis, from its 0.1 step so the
    /// low end stays apart from a dark background): lightness rises
    /// monotonically from violet through blue and green to yellow, readable
    /// in light and dark appearance and by colour-vision deficiencies.
    static let perceptualRamp: [UInt32] = [
        0x482475, 0x414487, 0x355f8d, 0x2a788e, 0x21918c, 0x22a884, 0x44bf70, 0x7ad151, 0xbddf26, 0xfde725,
    ]

    /// A quantity `t` in 0…1 on the perceptual ramp (same in both appearances).
    public static func perceptual(_ t: Double) -> SIMD3<Double> {
        interpolate(perceptualRamp.map(rgb), t)
    }

    /// Signed value `t` in −1…1: blue (negative) ↔ neutral ↔ red (positive).
    public static func diverging(_ t: Double, dark: Bool) -> SIMD3<Double> {
        let mid = dark ? rgb(0x383835) : rgb(0xf0efec)
        let blue = dark ? [0x3987e5, 0x184f95].map(rgb) : [0x6da7ec, 0x184f95].map(rgb)
        let red = dark ? [0xe66767, 0x9b2a29].map(rgb) : [0xee8f8e, 0xa52b2a].map(rgb)
        let x = t.isFinite ? min(max(t, -1), 1) : 0
        return x < 0 ? interpolate([mid] + blue, -x) : interpolate([mid] + red, x)
    }

    static func interpolate(_ stops: [SIMD3<Double>], _ t: Double) -> SIMD3<Double> {
        let x = (t.isFinite ? min(max(t, 0), 1) : 0) * Double(stops.count - 1)
        let i = min(Int(x), stops.count - 2)
        return stops[i] + (stops[i + 1] - stops[i]) * (x - Double(i))
    }

    /// A colour that follows the light or dark appearance.
    static func dynamic(_ f: @escaping @Sendable (Bool) -> SIMD3<Double>) -> Color {
        PlatformColor.dynamic(light: f(false), dark: f(true))
    }
}
