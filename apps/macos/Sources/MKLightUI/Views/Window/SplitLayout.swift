import SwiftUI

/// Two panes with a draggable divider whose position is a fraction bound to
/// persisted state (the project's saved layout). Works the same with a
/// mouse and touch; the divider looks like the system's thin split divider.
struct SplitLayout<First: View, Second: View>: View {
    enum Axis { case horizontal, vertical }

    let axis: Axis
    @Binding var fraction: Double
    var showsFirst = true
    var minFirst: CGFloat = 120
    var minSecond: CGFloat = 160
    @ViewBuilder var first: First
    @ViewBuilder var second: Second

    @State private var dragStart: Double?

    var body: some View {
        GeometryReader { geo in
            let total = axis == .horizontal ? geo.size.width : geo.size.height
            let size = firstSize(total)
            if axis == .horizontal {
                HStack(spacing: 0) {
                    if showsFirst {
                        first.frame(width: size)
                        divider(total: total)
                    }
                    second.frame(maxWidth: .infinity)
                }
            } else {
                VStack(spacing: 0) {
                    if showsFirst {
                        first.frame(height: size)
                        divider(total: total)
                    }
                    second.frame(maxHeight: .infinity)
                }
            }
        }
    }

    private func firstSize(_ total: CGFloat) -> CGFloat {
        guard total > minFirst + minSecond else { return max(total * 0.5, 0) }
        return min(max(CGFloat(fraction) * total, minFirst), total - minSecond)
    }

    private func divider(total: CGFloat) -> some View {
        let vertical = axis == .vertical
        return Rectangle()
            .fill(Color(white: 0.5, opacity: 0.001))
            .frame(width: vertical ? nil : 7, height: vertical ? 7 : nil)
            .overlay { Divider() }
            .contentShape(Rectangle())
            .resizePointer(vertical: vertical)
            .gesture(
                DragGesture(minimumDistance: 1, coordinateSpace: .global)
                    .onChanged { g in
                        let start = dragStart ?? fraction
                        if dragStart == nil { dragStart = fraction }
                        let delta = vertical ? g.translation.height : g.translation.width
                        let lo = Double(minFirst / max(total, 1)), hi = Double(1 - minSecond / max(total, 1))
                        fraction = min(max(start + Double(delta / max(total, 1)), lo), max(hi, lo))
                    }
                    .onEnded { _ in dragStart = nil }
            )
            .padding(vertical ? .vertical : .horizontal, -3)
            .zIndex(1)
            .accessibilityElement()
            .accessibilityLabel(vertical ? "Horizontal divider" : "Vertical divider")
            .accessibilityValue("\(Int(fraction * 100)) percent")
            .accessibilityAdjustableAction { dir in
                fraction = min(max(fraction + (dir == .increment ? 0.05 : -0.05), 0.1), 0.9)
            }
    }
}

/// Several panes side by side, each with a resizable width (weights saved
/// per pane key).
struct PaneRow<Item: Identifiable, Content: View>: View where Item.ID == String {
    let items: [Item]
    @Binding var weights: [String: Double]
    var minWidth: CGFloat = 240
    @ViewBuilder var content: (Item) -> Content
    @State private var dragStart: [String: Double]?

    var body: some View {
        GeometryReader { geo in
            let w = widths(geo.size.width)
            HStack(spacing: 0) {
                ForEach(Array(items.enumerated()), id: \.element.id) { k, item in
                    content(item).frame(width: w[k])
                    if k < items.count - 1 { divider(k, total: geo.size.width) }
                }
            }
        }
    }

    private func weight(_ id: String) -> Double { max(weights[id] ?? 1, 0.05) }

    private func widths(_ total: CGFloat) -> [CGFloat] {
        let avail = max(total - CGFloat(max(items.count - 1, 0)), 0)
        let sum = items.map { weight($0.id) }.reduce(0, +)
        return items.map { CGFloat(weight($0.id) / max(sum, 1e-9)) * avail }
    }

    private func divider(_ k: Int, total: CGFloat) -> some View {
        Rectangle()
            .fill(Color(white: 0.5, opacity: 0.001))
            .frame(width: 7)
            .overlay { Divider() }
            .contentShape(Rectangle())
            .resizePointer(vertical: false)
            .padding(.horizontal, -3)
            .zIndex(1)
            .gesture(
                DragGesture(minimumDistance: 1, coordinateSpace: .global)
                    .onChanged { g in
                        let a = items[k].id, b = items[k + 1].id
                        let start = dragStart ?? [a: weight(a), b: weight(b)]
                        if dragStart == nil { dragStart = start }
                        let sum = items.map { weight($0.id) }.reduce(0, +)
                        let pair = start[a]! + start[b]!
                        let dw = Double(g.translation.width / max(total, 1)) * sum
                        let minW = Double(minWidth / max(total, 1)) * sum
                        let na = min(max(start[a]! + dw, minW), pair - minW)
                        guard na > 0, pair - na > 0 else { return }
                        weights[a] = na
                        weights[b] = pair - na
                    }
                    .onEnded { _ in dragStart = nil }
            )
    }
}
