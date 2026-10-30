import Charts
import CoreGraphics
import MKLightEngine
import SwiftUI

/// PSF heat map (engine-normalised intensities mapped through a perceptual
/// colour map) with a central cross-section.
struct PSFView: View {
    let result: PSFResult
    let logScale: Bool

    var body: some View {
        let halfUm = Double(result.width) * result.pixelPitch * 500
        VStack(spacing: 6) {
            HStack(alignment: .top, spacing: 12) {
                if let image = Self.image(result, logScale: logScale) {
                    Image(decorative: image, scale: 1)
                        .resizable()
                        .interpolation(.medium)
                        .aspectRatio(1, contentMode: .fit)
                        .overlay(alignment: .bottomLeading) {
                            Text(String(format: "±%.1f µm", halfUm)).font(.caption2).padding(3)
                                .background(.black.opacity(0.4)).foregroundStyle(.white)
                        }
                }
                colorBar.frame(width: 14)
            }
            Chart(crossSection) { p in
                LineMark(x: .value("x (µm)", p.x), y: .value("Intensity", p.y))
            }
            .chartYScale(domain: 0...max(1, Double(result.values.max() ?? 1)))
            .chartXAxisLabel("x (µm), central row")
            .frame(height: 110)
            Text(String(format: "Field %d · Strehl ≈ %.3f · pixel %.2f µm%@", result.field + 1, result.strehl, result.pixelPitch * 1000, logScale ? " · log₁₀ scale (6 decades)" : ""))
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(10)
    }

    private struct Sample: Identifiable {
        let id: Int
        let x: Double
        let y: Double
    }

    private var crossSection: [Sample] {
        let row = result.height / 2
        return (0..<result.width).map { c in
            Sample(id: c, x: (Double(c) + 0.5 - Double(result.width) / 2) * result.pixelPitch * 1000, y: Double(result.values[row * result.width + c]))
        }
    }

    private var colorBar: some View {
        LinearGradient(colors: stride(from: 1.0, through: 0, by: -0.1).map { t in
            let c = Palette.heat(t)
            return Color(.sRGB, red: c.x, green: c.y, blue: c.z)
        }, startPoint: .top, endPoint: .bottom)
        .clipShape(RoundedRectangle(cornerRadius: 2))
    }

    /// Maps intensities to RGBA8; row 0 (−y) is drawn at the bottom.
    static func image(_ r: PSFResult, logScale: Bool) -> CGImage? {
        guard r.width > 0, r.height > 0, r.values.count == r.width * r.height else { return nil }
        let peak = Double(r.values.max() ?? 0)
        guard peak > 0 else { return nil }
        var bytes = [UInt8](repeating: 255, count: r.width * r.height * 4)
        for row in 0..<r.height {
            let dst = (r.height - 1 - row) * r.width
            for col in 0..<r.width {
                let v = Double(r.values[row * r.width + col]) / peak
                let t = logScale ? max(0, 1 + log10(max(v, 1e-12)) / 6) : v
                let c = Palette.heat(t)
                let o = (dst + col) * 4
                bytes[o] = UInt8(c.x * 255)
                bytes[o + 1] = UInt8(c.y * 255)
                bytes[o + 2] = UInt8(c.z * 255)
            }
        }
        guard let provider = CGDataProvider(data: Data(bytes) as CFData) else { return nil }
        return CGImage(
            width: r.width, height: r.height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: r.width * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }
}
