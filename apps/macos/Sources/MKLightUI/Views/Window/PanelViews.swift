import MKLightEngine
import MKLightModel
import SwiftUI

/// A window's panels: its tree of splits and tab groups. While a panel is
/// dragged, the place it would land is highlighted.
struct PanelWindowContent: View {
    let session: DesignSession
    let model: PanelWindowModel
    #if os(macOS)
    var onWindow: ((NSWindow) -> Void)?
    #endif

    var body: some View {
        Group {
            if let root = model.root {
                PanelNodeView(node: root, session: session)
            } else {
                EmptyPanelsView(session: session)
            }
        }
        #if os(macOS)
        .background { PanelReporter(role: .content(model.id), drag: session.drag, onWindow: onWindow) }
        #endif
        .overlay { WindowEdgeHighlight(windowID: model.id, drag: session.drag) }
        .environment(\.designSession, session)
    }
}

/// One node of the tree.
struct PanelNodeView: View {
    let node: PanelNode
    let session: DesignSession

    var body: some View {
        switch node {
        case .group(let g): PanelGroupView(group: g, session: session)
        case .split(let s): PanelSplitView(split: s, session: session)
        }
    }
}

// MARK: - Splits

/// Children side by side or stacked, with draggable dividers between them
/// (double-click a divider for equal shares). Only this view reads the
/// shares, so dragging a divider lays out its children without updating
/// the rest of the window.
struct PanelSplitView: View {
    let split: PanelSplit
    let session: DesignSession
    @State private var dragStart: [Double]?

    private var horizontal: Bool { split.axis == .horizontal }
    private var minSize: CGFloat { horizontal ? 200 : 120 }

    var body: some View {
        GeometryReader { geo in
            let total = horizontal ? geo.size.width : geo.size.height
            let sizes = sizes(total)
            let stack = horizontal ? AnyLayout(HStackLayout(spacing: 0)) : AnyLayout(VStackLayout(spacing: 0))
            stack {
                ForEach(Array(split.children.enumerated()), id: \.element.id) { k, child in
                    PanelNodeView(node: child, session: session)
                        .frame(width: horizontal ? sizes[k] : nil, height: horizontal ? nil : sizes[k])
                    if k < split.children.count - 1 { divider(k, total: total) }
                }
            }
        }
    }

    /// The children's sizes: their shares, each at least `minSize` when
    /// the space allows (the others give way in proportion).
    private func sizes(_ total: CGFloat) -> [CGFloat] {
        let n = split.children.count
        let avail = max(total - CGFloat(max(n - 1, 0)), 0)
        var s = split.fractions.map { CGFloat($0) * avail }
        guard avail > minSize * CGFloat(n) else { return s }
        let deficit = s.reduce(0) { $0 + max(minSize - $1, 0) }
        guard deficit > 0 else { return s }
        let spare = s.reduce(0) { $0 + max($1 - minSize, 0) }
        s = s.map { $0 < minSize ? minSize : $0 - deficit * ($0 - minSize) / max(spare, 1) }
        return s
    }

    private func divider(_ k: Int, total: CGFloat) -> some View {
        Rectangle()
            .fill(Color(white: 0.5, opacity: 0.001))
            .frame(width: horizontal ? 7 : nil, height: horizontal ? nil : 7)
            .overlay { Divider() }
            .contentShape(Rectangle())
            .resizePointer(vertical: !horizontal)
            .gesture(
                DragGesture(minimumDistance: 1, coordinateSpace: .global)
                    .onChanged { g in
                        let start = dragStart ?? split.fractions
                        if dragStart == nil { dragStart = start }
                        guard start.indices.contains(k + 1) else { return }
                        let avail = max(total - CGFloat(split.children.count - 1), 1)
                        let delta = Double((horizontal ? g.translation.width : g.translation.height) / avail)
                        let pair = start[k] + start[k + 1]
                        let minF = min(Double(minSize / avail), pair / 2)
                        let a = min(max(start[k] + delta, minF), pair - minF)
                        split.resize(divider: k, fractions: (a, pair - a))
                    }
                    .onEnded { _ in dragStart = nil }
            )
            .onTapGesture(count: 2) { split.equalize() }
            .padding(horizontal ? .horizontal : .vertical, -3)
            .zIndex(1)
            .accessibilityElement()
            .accessibilityLabel(horizontal ? "Vertical divider" : "Horizontal divider")
            .accessibilityValue("\(Int((split.fractions.indices.contains(k) ? split.fractions[k] : 0) * 100)) percent")
            .accessibilityAdjustableAction { dir in
                guard split.fractions.indices.contains(k + 1) else { return }
                let pair = split.fractions[k] + split.fractions[k + 1]
                let a = min(max(split.fractions[k] + (dir == .increment ? 0.05 : -0.05), 0.05), pair - 0.05)
                split.resize(divider: k, fractions: (a, pair - a))
            }
    }
}

// MARK: - Tab groups

/// Panels sharing a place: a tab strip when there are several, then the
/// panel in front.
struct PanelGroupView: View {
    let group: PanelGroup
    let session: DesignSession

    var body: some View {
        let tabbed = group.panels.count > 1
        VStack(spacing: 0) {
            if tabbed {
                PanelTabStrip(group: group, session: session)
                Divider()
            }
            if let panel = group.current {
                PanelContent(panel: panel, session: session)
                    .environment(\.panelTabbed, tabbed)
                    .id(panel)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        #if os(macOS)
        .background { PanelReporter(role: .group(group.id), drag: session.drag) }
        #endif
        .overlay { GroupDropHighlight(groupID: group.id, drag: session.drag) }
        .clipped()
    }
}

/// The panel's view, and the focus following a click into it.
struct PanelContent: View {
    let panel: PanelID
    let session: DesignSession

    var body: some View {
        content
            .simultaneousGesture(TapGesture().onEnded { session.focus(panel) })
            .opacity(session.drag.dragging == panel ? 0.45 : 1)
    }

    @ViewBuilder private var content: some View {
        switch panel {
        case .lensData:
            LensTableView(design: session.design, workspace: session.workspace, window: session.window)
        case .layout:
            LayoutPaneView(workspace: session.workspace, window: session.window, system: session.design.system)
        case .analysis(let id):
            if let pane = session.pane(id) {
                AnalysisPaneView(pane: pane, workspace: session.workspace, window: session.window, system: session.design.system)
            } else {
                Color.clear
            }
        }
    }
}

/// The tabs of a group: each the panel's symbol and title, the front one
/// on glass. Click to switch, drag to move (to another place, another
/// window or out into a new one); the close button shows on hover.
struct PanelTabStrip: View {
    let group: PanelGroup
    let session: DesignSession
    @Namespace private var glass

    var body: some View {
        ScrollView(.horizontal) {
            GlassEffectContainer(spacing: Space.xs) {
                HStack(spacing: Space.xxs) {
                    ForEach(group.panels, id: \.self) { panel in
                        PanelTab(panel: panel, group: group, session: session, glass: glass)
                    }
                }
                .padding(.horizontal, Space.s)
                .padding(.vertical, Space.xs)
            }
        }
        .scrollIndicators(.never)
        .frame(height: 34)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Tabs")
    }
}

private struct PanelTab: View {
    let panel: PanelID
    let group: PanelGroup
    let session: DesignSession
    let glass: Namespace.ID
    @State private var hovering = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        let selected = group.current == panel
        let title = session.title(of: panel)
        HStack(spacing: Space.xs) {
            Image(systemName: session.symbol(of: panel))
                .font(.caption)
                .foregroundStyle(selected ? AnyShapeStyle(.tint) : AnyShapeStyle(.secondary))
            Text(title)
                .font(.callout.weight(selected ? .semibold : .regular))
                .lineLimit(1)
            Button {
                session.close(panel)
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 8, weight: .bold))
                    .frame(width: 14, height: 14)
                    .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            .opacity(hovering || selected ? 1 : 0)
            // Hidden, it takes no clicks: a click near a background tab's
            // end selects the tab instead of closing it.
            .allowsHitTesting(hovering || selected)
            .help(panel.paneID == nil ? "Hide \(title)" : "Close \(title)")
            .accessibilityLabel(panel.paneID == nil ? "Hide \(title)" : "Close \(title)")
        }
        .padding(.leading, Space.m)
        .padding(.trailing, Space.xs)
        .padding(.vertical, Space.xs)
        .background {
            if !selected && hovering { Capsule().fill(.fill.quaternary) }
        }
        .glassEffect(selected ? .regular.interactive() : .identity, in: .capsule)
        .glassEffectID(panel.key, in: glass)
        .contentShape(Capsule())
        .onHover { hovering = $0 }
        .onTapGesture(perform: select)
        .panelHandle(panel)
        #if os(macOS)
        .background { PanelReporter(role: .tab(group.id, panel), drag: session.drag) }
        #endif
        .opacity(session.drag.dragging == panel ? 0.4 : 1)
        .help(title)
        // One element: pressing it selects the tab; closing is an action of its own.
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(title)
        .accessibilityAddTraits(selected ? [.isSelected, .isButton] : .isButton)
        .accessibilityAction(.default, select)
        .accessibilityAction(named: panel.paneID == nil ? "Hide" : "Close") { session.close(panel) }
    }

    private func select() {
        withAnimation(reduceMotion ? nil : .snappy(duration: 0.22)) { group.selected = panel }
        session.focus(panel)
    }
}

// MARK: - Drag sources

extension View {
    /// Makes this view a handle that moves the panel: drag it to another
    /// place, another window or out into a window of its own; its context
    /// menu has the same commands.
    func panelHandle(_ panel: PanelID) -> some View { modifier(PanelHandle(panel: panel)) }
}

private struct PanelHandle: ViewModifier {
    let panel: PanelID
    @Environment(\.designSession) private var session

    func body(content: Content) -> some View {
        if let session {
            content
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 6, coordinateSpace: .global)
                        .onChanged { _ in session.drag.dragChanged(panel) }
                        .onEnded { _ in session.drag.dragEnded() }
                )
                #if os(macOS)
                .pointerStyle(.grabIdle)
                #endif
                .contextMenu { PanelMenuItems(panel: panel, session: session) }
                .accessibilityAction(named: "Move to New Window") { session.moveToNewWindow(panel) }
        } else {
            content
        }
    }
}

/// Moving and closing a panel, for its context menu and More menu.
struct PanelMenuItems: View {
    let panel: PanelID
    let session: DesignSession

    var body: some View {
        if session.isDetached(panel) {
            Button("Move to Main Window", systemImage: "pip.enter") { session.moveToMainWindow(panel) }
        }
        if !session.isAloneInWindow(panel) {
            Button("Move to New Window", systemImage: "pip.exit") { session.moveToNewWindow(panel) }
        }
        Divider()
        if panel.paneID == nil {
            Button("Hide \(session.title(of: panel))", systemImage: "eye.slash") { session.close(panel) }
        } else {
            Button("Close Pane", systemImage: "xmark", role: .destructive) { session.close(panel) }
        }
    }
}

// MARK: - Drop highlights

/// Where a dragged panel would land in this group: the whole group (a tab)
/// or the half on one side (a split).
private struct GroupDropHighlight: View {
    let groupID: UUID
    let drag: PanelDragController
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        GeometryReader { geo in
            if case let (rect, label, symbol)? = zone(in: geo.size) {
                DropZoneShape(label: label, symbol: symbol)
                    .frame(width: rect.width, height: rect.height)
                    .offset(x: rect.minX, y: rect.minY)
                    .transition(.opacity)
            }
        }
        .allowsHitTesting(false)
        .animation(reduceMotion ? nil : .snappy(duration: 0.18), value: drag.target)
    }

    private func zone(in s: CGSize) -> (CGRect, String, String)? {
        let all = CGRect(origin: .zero, size: s)
        switch drag.target {
        case .tab(let g, _) where g == groupID:
            return (all.insetBy(dx: 4, dy: 4), "Add as Tab", "rectangle.stack")
        case .split(let g, let edge) where g == groupID:
            let r: CGRect =
                switch edge {
                case .leading: CGRect(x: 0, y: 0, width: s.width / 2, height: s.height)
                case .trailing: CGRect(x: s.width / 2, y: 0, width: s.width / 2, height: s.height)
                case .top: CGRect(x: 0, y: 0, width: s.width, height: s.height / 2)
                case .bottom: CGRect(x: 0, y: s.height / 2, width: s.width, height: s.height / 2)
                }
            return (r.insetBy(dx: 4, dy: 4), "Split", edge.axis == .horizontal ? "rectangle.split.2x1" : "rectangle.split.1x2")
        default:
            return nil
        }
    }
}

/// Along a window's edge: a band a fifth of the window deep.
private struct WindowEdgeHighlight: View {
    let windowID: UUID
    let drag: PanelDragController
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        GeometryReader { geo in
            if case .windowEdge(let w, let edge) = drag.target, w == windowID {
                let s = geo.size
                let r: CGRect =
                    switch edge {
                    case .leading: CGRect(x: 0, y: 0, width: s.width * 0.22, height: s.height)
                    case .trailing: CGRect(x: s.width * 0.78, y: 0, width: s.width * 0.22, height: s.height)
                    case .top: CGRect(x: 0, y: 0, width: s.width, height: s.height * 0.22)
                    case .bottom: CGRect(x: 0, y: s.height * 0.78, width: s.width, height: s.height * 0.22)
                    }
                DropZoneShape(label: edge.axis == .horizontal ? "Full-Height Column" : "Full-Width Row", symbol: "rectangle.inset.filled")
                    .frame(width: r.width - 8, height: r.height - 8)
                    .offset(x: r.minX + 4, y: r.minY + 4)
                    .transition(.opacity)
            }
        }
        .allowsHitTesting(false)
        .animation(reduceMotion ? nil : .snappy(duration: 0.18), value: drag.target)
    }
}

private struct DropZoneShape: View {
    let label: String
    let symbol: String

    var body: some View {
        RoundedRectangle(cornerRadius: 12, style: .continuous)
            .fill(Color.accentColor.opacity(0.14))
            .overlay {
                RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(Color.accentColor.opacity(0.75), lineWidth: 2)
            }
            .overlay {
                ViewThatFits {
                    Label(label, systemImage: symbol)
                        .font(.callout.weight(.medium))
                        .padding(.horizontal, Space.l)
                        .padding(.vertical, Space.s)
                        .glassEffect(.regular, in: .capsule)
                        .fixedSize()
                    Color.clear
                }
            }
    }
}

// MARK: - Empty window

/// The main window with every panel closed or moved away.
struct EmptyPanelsView: View {
    let session: DesignSession

    var body: some View {
        ContentUnavailableView {
            Label("No Panels", systemImage: "rectangle.dashed")
        } description: {
            Text("Show the lens data or the layout, or add an analysis. Panels moved to other windows can come back here.")
        } actions: {
            HStack {
                Button("Show Lens Data") { session.toggle(.lensData) }
                Button("Show Layout") { session.toggle(.layout) }
                Menu("Add Analysis") { AddPaneMenuItems(workspace: session.workspace) }.fixedSize()
            }
            if !session.arrangement.detached.isEmpty {
                Button("Move All Panels Here") {
                    for w in session.arrangement.detached { session.dock(w) }
                }
                .buttonStyle(.glass)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(.background)
    }
}
