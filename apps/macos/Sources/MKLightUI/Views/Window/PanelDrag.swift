import SwiftUI
#if os(macOS)
import AppKit
#endif

/// Moving panels by dragging their title or tab, within a window, between
/// a design's windows, and out into a window of its own (like a browser
/// tab).
///
/// The drag follows the mouse in screen coordinates, so it works across
/// windows: each tab group, tab and window content area places an
/// invisible AppKit view (`PanelReporter`) whose screen rectangle is read
/// when the mouse moves. Over a group, the drop zone is a tab (the header,
/// or the middle) or a split (near an edge); along a window's edge, a whole
/// row or column; outside the design's windows, a new window, previewed as
/// an outline where it will open. Escape cancels.
@MainActor
@Observable
final class PanelDragController {
    /// The panel being dragged.
    private(set) var dragging: PanelID?
    /// Where it would land now.
    private(set) var target: PanelDropTarget?
    @ObservationIgnored weak var session: DesignSession?

    enum Role: Hashable {
        case group(UUID)
        case tab(UUID, PanelID)
        case content(UUID)
    }

    #if os(macOS)
    /// The locator views by role. A group or tab moved in the tree gets a
    /// new view while the old one may still update once, so a role keeps
    /// every live view and lookups use the one that is in a window.
    @ObservationIgnored private var reporterViews: [Role: [WeakReporter]] = [:]
    @ObservationIgnored private var ghost: PanelDragGhost?
    @ObservationIgnored private var monitor: Any?
    @ObservationIgnored private var sourceSize = CGSize(width: 560, height: 420)
    @ObservationIgnored private var newWindowFrame: CGRect?

    func register(_ view: PanelReporterView, as role: Role) {
        var list = reporterViews[role, default: []].filter { $0.view != nil }
        if !list.contains(where: { $0.view === view }) { list.append(WeakReporter(view)) }
        reporterViews[role] = list
    }

    /// The role's view on screen (the latest one in a window).
    private func reporter(_ role: Role) -> PanelReporterView? {
        reporterViews[role]?.last { $0.view?.window != nil }?.view
    }

    /// Every role with its view on screen.
    private var reporters: [(role: Role, view: PanelReporterView)] {
        reporterViews.compactMap { role, list in list.last { $0.view?.window != nil }?.view.map { (role, $0) } }
    }

    /// The drag source's gesture moved: the first call starts the drag;
    /// later mouse events come from an event monitor, so the drag goes on
    /// even if the source view is rebuilt.
    func dragChanged(_ panel: PanelID) {
        if dragging == nil { begin(panel) } else { update(NSApp.currentEvent) }
    }

    /// The drag source's gesture ended.
    func dragEnded() { finish(drop: true) }

    /// Abandons a drag (Escape, or the window closing).
    func cancel() { finish(drop: false) }

    private func begin(_ panel: PanelID) {
        guard let session else { return }
        dragging = panel
        if case let (_, g)? = session.arrangement.location(of: panel), let r = reporter(.group(g.id))?.screenRect {
            sourceSize = r.size
        }
        ghost = PanelDragGhost(title: session.title(of: panel), symbol: session.symbol(of: panel))
        monitor = NSEvent.addLocalMonitorForEvents(matching: [.leftMouseDragged, .leftMouseUp, .keyDown]) { [weak self] event in
            let consumed: Bool = MainActor.assumeIsolated {
                guard let self else { return false }
                switch event.type {
                case .leftMouseDragged:
                    self.update(event)
                case .leftMouseUp:
                    self.update(event)
                    self.finish(drop: true)
                case .keyDown where event.keyCode == 53:  // Escape
                    self.finish(drop: false)
                    return true
                default:
                    break
                }
                return false
            }
            return consumed ? nil : event
        }
        update(NSApp.currentEvent)
    }

    /// The pointer on screen: where the mouse event happened.
    private func location(_ event: NSEvent?) -> CGPoint {
        guard let event, [.leftMouseDragged, .leftMouseDown, .leftMouseUp].contains(event.type) else { return NSEvent.mouseLocation }
        if let w = event.window { return w.convertPoint(toScreen: event.locationInWindow) }
        return event.locationInWindow
    }

    private func update(_ event: NSEvent? = NSApp.currentEvent) {
        guard let panel = dragging, let session else { return }
        let p = location(event)
        let t = hitTest(p, panel: panel, session: session)
        if t != target { target = t }
        newWindowFrame = t == .newWindow ? frame(at: p) : nil
        ghost?.show(at: p, window: newWindowFrame, dropping: t != nil)
    }

    private func finish(drop: Bool) {
        guard let panel = dragging else { return }
        if let monitor { NSEvent.removeMonitor(monitor) }
        monitor = nil
        let t = drop ? target : nil
        let frame = newWindowFrame
        dragging = nil
        target = nil
        newWindowFrame = nil
        ghost?.close()
        ghost = nil
        if let t { session?.move(panel, to: t, frame: frame) }
    }

    // MARK: Hit testing

    private func hitTest(_ p: CGPoint, panel: PanelID, session: DesignSession) -> PanelDropTarget? {
        // The app's frontmost window under the pointer (the drag preview
        // aside): drops are judged against MKLight's own windows.
        let ghostNumber = ghost?.windowNumber
        let number = NSApp.orderedWindows.first { $0.isVisible && $0.windowNumber != ghostNumber && $0.frame.contains(p) }?.windowNumber ?? -1
        let areas = reporters.compactMap { role, v -> (UUID, PanelReporterView)? in
            guard case .content(let id) = role, v.window?.windowNumber == number else { return nil }
            return (id, v)
        }
        // Not over one of the design's windows: a window of its own.
        guard case let (wid, area)? = areas.first else { return .newWindow }
        guard let rect = area.screenRect, rect.contains(p) else { return nil }

        // Along the window's edges, when it has more than one group: a whole row or column.
        let band: CGFloat = 16
        let edge: PanelEdge? =
            p.x - rect.minX < band ? .leading : rect.maxX - p.x < band ? .trailing : rect.maxY - p.y < band ? .top : p.y - rect.minY < band ? .bottom : nil
        if let edge, (session.arrangement.window(wid)?.groups.count ?? 0) > 1 {
            let t = PanelDropTarget.windowEdge(window: wid, edge: edge)
            if session.arrangement.canMove(panel, to: t) { return t }
        }
        for (role, v) in reporters {
            guard case .group(let gid) = role, v.window?.windowNumber == number, let r = v.screenRect, r.contains(p)
            else { continue }
            let t = zone(p, in: r, group: gid, session: session)
            return session.arrangement.canMove(panel, to: t) ? t : nil
        }
        return nil
    }

    /// The header (top 34 points) and the middle add a tab; the outer
    /// 30 % towards an edge splits the group on that side.
    private func zone(_ p: CGPoint, in r: CGRect, group gid: UUID, session: DesignSession) -> PanelDropTarget {
        let fromTop = r.maxY - p.y
        if fromTop < 34 { return .tab(group: gid, index: tabIndex(p, group: gid, session: session)) }
        let nx = (p.x - r.minX) / max(r.width, 1), ny = fromTop / max(r.height, 1)
        let edges: [(PanelEdge, CGFloat)] = [(.leading, nx), (.trailing, 1 - nx), (.top, ny), (.bottom, 1 - ny)]
        let nearest = edges.min { $0.1 < $1.1 }!
        return nearest.1 < 0.3 ? .split(group: gid, edge: nearest.0) : .tab(group: gid, index: nil)
    }

    /// Over a tab strip: the position between the tabs under the mouse.
    private func tabIndex(_ p: CGPoint, group gid: UUID, session: DesignSession) -> Int? {
        guard let g = session.arrangement.group(gid)?.group, g.panels.count > 1 else { return nil }
        var index = 0
        for (k, panel) in g.panels.enumerated() {
            if let r = reporter(.tab(gid, panel))?.screenRect, p.x > r.midX { index = k + 1 }
        }
        return index
    }

    // MARK: New windows

    /// A new window under the mouse, the size of the group the panel left,
    /// its title bar at the pointer.
    private func frame(at p: CGPoint) -> CGRect {
        let w = min(max(sourceSize.width, 420), 1100), h = min(max(sourceSize.height + 28, 320), 860)
        return onScreen(CGRect(x: p.x - min(160, w / 2), y: p.y + 14 - h, width: w, height: h), near: p)
    }

    /// Move to New Window: beside the panel's place now, a little offset.
    func defaultFrame(for panel: PanelID) -> CGRect {
        guard let session else { return CGRect(x: 200, y: 200, width: 640, height: 480) }
        if case let (_, g)? = session.arrangement.location(of: panel), let r = reporter(.group(g.id))?.screenRect {
            let w = min(max(r.width, 420), 1100), h = min(max(r.height + 28, 320), 860)
            return onScreen(CGRect(x: r.minX + 40, y: r.maxY - 40 - h, width: w, height: h), near: CGPoint(x: r.midX, y: r.midY))
        }
        let main = session.mainWindow?.frame ?? CGRect(x: 200, y: 200, width: 1200, height: 800)
        return onScreen(CGRect(x: main.midX - 300, y: main.midY - 220, width: 600, height: 440), near: CGPoint(x: main.midX, y: main.midY))
    }

    private func onScreen(_ f: CGRect, near p: CGPoint) -> CGRect {
        guard let screen = (NSScreen.screens.first { $0.frame.contains(p) } ?? NSScreen.main)?.visibleFrame else { return f }
        var f = f
        f.size.width = min(f.width, screen.width)
        f.size.height = min(f.height, screen.height)
        f.origin.x = min(max(f.minX, screen.minX), screen.maxX - f.width)
        f.origin.y = min(max(f.minY, screen.minY), screen.maxY - f.height)
        return f
    }
    #else
    func dragChanged(_ panel: PanelID) {}
    func dragEnded() {}
    func cancel() {}
    #endif
}

#if os(macOS)
/// An invisible view whose window and screen rectangle locate a panel
/// group, a tab or a window's content during a drag. It never takes the
/// mouse.
final class PanelReporterView: NSView {
    var onWindow: ((NSWindow) -> Void)?

    override func hitTest(_ point: NSPoint) -> NSView? { nil }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        if let window { onWindow?(window) }
    }

    var screenRect: CGRect? {
        guard let window else { return nil }
        return window.convertToScreen(convert(bounds, to: nil))
    }
}

final class WeakReporter {
    weak var view: PanelReporterView?
    init(_ view: PanelReporterView) { self.view = view }
}

struct PanelReporter: NSViewRepresentable {
    let role: PanelDragController.Role
    let drag: PanelDragController
    var onWindow: ((NSWindow) -> Void)?

    func makeNSView(context: Context) -> PanelReporterView {
        let v = PanelReporterView()
        v.onWindow = onWindow
        drag.register(v, as: role)
        return v
    }

    func updateNSView(_ v: PanelReporterView, context: Context) {
        drag.register(v, as: role)
    }
}

/// What follows the pointer during a drag: the panel's title on a glass
/// capsule, or, outside the design's windows, the outline of the window it
/// would open in. A borderless panel above every window that never takes
/// the mouse.
@MainActor
final class PanelDragGhost {
    @MainActor @Observable final class Model {
        var title = ""
        var symbol = ""
        var windowSize: CGSize?
        var dropping = true
    }

    private let panel: NSPanel
    private let model = Model()
    private static let chipSize = CGSize(width: 280, height: 52)

    init(title: String, symbol: String) {
        model.title = title
        model.symbol = symbol
        panel = NSPanel(
            contentRect: CGRect(origin: .zero, size: Self.chipSize), styleMask: [.borderless, .nonactivatingPanel], backing: .buffered,
            defer: false)
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = false
        panel.ignoresMouseEvents = true
        panel.level = .popUpMenu
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .transient, .ignoresCycle]
        panel.isReleasedWhenClosed = false
        panel.contentView = NSHostingView(rootView: PanelGhostView(model: model))
        panel.orderFrontRegardless()
    }

    var windowNumber: Int { panel.windowNumber }

    func show(at p: CGPoint, window frame: CGRect?, dropping: Bool) {
        if model.dropping != dropping { model.dropping = dropping }
        if let frame {
            if model.windowSize != frame.size { model.windowSize = frame.size }
            if panel.frame.size != frame.size { panel.setFrame(frame, display: true) } else { panel.setFrameOrigin(frame.origin) }
        } else {
            if model.windowSize != nil { model.windowSize = nil }
            let f = CGRect(x: p.x + 8, y: p.y - Self.chipSize.height - 4, width: Self.chipSize.width, height: Self.chipSize.height)
            if panel.frame.size != f.size { panel.setFrame(f, display: true) } else { panel.setFrameOrigin(f.origin) }
        }
    }

    func close() { panel.orderOut(nil) }
}

private struct PanelGhostView: View {
    let model: PanelDragGhost.Model

    var body: some View {
        if model.windowSize != nil {
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color.accentColor.opacity(0.08))
                .overlay {
                    RoundedRectangle(cornerRadius: 16, style: .continuous).strokeBorder(Color.accentColor.opacity(0.8), lineWidth: 2)
                }
                .overlay(alignment: .topLeading) { chip.padding(Space.m) }
                .overlay {
                    Label("New Window", systemImage: "macwindow.badge.plus")
                        .font(.callout.weight(.medium))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, Space.l)
                        .padding(.vertical, Space.s)
                        .glassEffect(.regular, in: .capsule)
                }
                .padding(2)
        } else {
            chip
                .opacity(model.dropping ? 1 : 0.55)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                .padding(Space.xs)
        }
    }

    private var chip: some View {
        Label(model.title, systemImage: model.symbol)
            .font(.callout.weight(.medium))
            .lineLimit(1)
            .padding(.horizontal, Space.l)
            .padding(.vertical, Space.s)
            .glassEffect(.regular, in: .capsule)
            .fixedSize()
    }
}
#endif
