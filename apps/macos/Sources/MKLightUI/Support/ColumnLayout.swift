import SwiftUI

/// A window column's root: it takes the size it is offered, within fixed
/// bounds, without measuring its content.
///
/// A split view asks each column for its minimum, ideal and maximum size.
/// A plain SwiftUI view answers by measuring its whole tree (the analysis
/// panes, the lens table, the inspector's form), and its answer changes as
/// panels open and close, which made the split view squeeze the inspector
/// to its minimum width. This layout answers from fixed bounds.
struct ColumnLayout: Layout {
    var minSize: CGSize
    var idealSize: CGSize

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let size = proposal.replacingUnspecifiedDimensions(by: idealSize)
        return CGSize(width: max(size.width, minSize.width), height: max(size.height, minSize.height))
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        for subview in subviews {
            subview.place(at: bounds.origin, anchor: .topLeading, proposal: ProposedViewSize(bounds.size))
        }
    }
}

extension View {
    /// Roots the view in a `ColumnLayout`: it fills the offered size and
    /// never reports its content's size to the split view around it.
    func columnLayout(minWidth: CGFloat, minHeight: CGFloat, idealWidth: CGFloat, idealHeight: CGFloat) -> some View {
        ColumnLayout(minSize: CGSize(width: minWidth, height: minHeight), idealSize: CGSize(width: idealWidth, height: idealHeight)) {
            self
        }
    }
}
