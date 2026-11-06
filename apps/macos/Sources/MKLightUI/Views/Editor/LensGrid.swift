import AppKit
import SwiftUI

/// A column of the lens grid: its title and width limits.
struct LensGridColumn: Equatable {
    var title: String
    var min: CGFloat
    var ideal: CGFloat
    var max: CGFloat = .infinity

    /// The column held at a width the user chose.
    func fixed(at width: CGFloat) -> LensGridColumn {
        let w = Swift.min(Swift.max(width, min), max)
        return LensGridColumn(title: title, min: w, ideal: w, max: w)
    }
}

/// Lays out one row of the lens grid (or its header): one subview per
/// column, side by side. Every row computes the same widths from the
/// proposed width, so columns line up without shared state, and a resize
/// is a layout pass rather than a view update.
struct LensGridLayout: Layout {
    let columns: [LensGridColumn]

    static func minimumWidth(_ columns: [LensGridColumn]) -> CGFloat { columns.reduce(0) { $0 + $1.min } }

    /// Starts from the ideal widths and shares out the difference to the
    /// available width among the columns that can still grow (or shrink).
    static func widths(_ columns: [LensGridColumn], total: CGFloat) -> [CGFloat] {
        var w = columns.map(\.ideal)
        var diff = total - w.reduce(0, +)
        for _ in 0...columns.count where abs(diff) > 0.5 {
            let open = columns.indices.filter { diff > 0 ? w[$0] < columns[$0].max : w[$0] > columns[$0].min }
            guard !open.isEmpty else { break }
            let share = diff / CGFloat(open.count)
            for i in open {
                let n = min(max(w[i] + share, columns[i].min), columns[i].max)
                diff -= n - w[i]
                w[i] = n
            }
        }
        return w
    }

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let w = Self.widths(columns, total: proposal.width ?? columns.reduce(0) { $0 + $1.ideal })
        let h = zip(subviews, w).map { s, cw in s.sizeThatFits(ProposedViewSize(width: cw, height: nil)).height }.max() ?? 0
        return CGSize(width: w.reduce(0, +), height: h)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = bounds.minX
        for (s, cw) in zip(subviews, Self.widths(columns, total: bounds.width)) {
            s.place(at: CGPoint(x: x, y: bounds.midY), anchor: .leading, proposal: ProposedViewSize(width: cw, height: bounds.height))
            x += cw
        }
    }
}

/// The grid's column titles, pinned over the rows; drag a divider to
/// resize the column on its left.
struct LensGridHeader: View {
    let columns: [LensGridColumn]
    @Binding var dragged: [Int: CGFloat]
    @State private var dragStart: CGFloat?

    var body: some View {
        LensGridLayout(columns: columns) {
            ForEach(columns.indices, id: \.self) { k in
                Text(columns[k].title)
                    .font(.subheadline.weight(.semibold))
                    .lineLimit(1)
                    .gridCell()
                    .overlay(alignment: .trailing) {
                        if k < columns.count - 1 { divider(k) }
                    }
            }
        }
        .frame(minHeight: 28)
        .background(Color(nsColor: .controlBackgroundColor))
        .overlay(alignment: .bottom) { Divider() }
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isHeader)
    }

    private func divider(_ k: Int) -> some View {
        Divider()
            .frame(height: 14)
            .frame(width: 7)
            .contentShape(Rectangle())
            .resizePointer(vertical: false)
            .gesture(
                DragGesture(minimumDistance: 1, coordinateSpace: .global)
                    .onChanged { g in
                        let start = dragStart ?? dragged[k] ?? columns[k].ideal
                        if dragStart == nil { dragStart = start }
                        dragged[k] = start + g.translation.width
                    }
                    .onEnded { _ in dragStart = nil }
            )
            .offset(x: 3.5)
            .accessibilityHidden(true)
    }
}

extension View {
    /// The padding of a lens grid cell, its content centred vertically.
    func gridCell() -> some View {
        padding(.horizontal, 6).frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
    }
}
