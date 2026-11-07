import CoreGraphics
import Foundation
import MKLightModel
import Observation

// MARK: - Panels

/// Something a window shows in a panel: the lens data editor, the layout
/// viewport, or one analysis pane (by its key). Each is open in at most one
/// place across a design's windows.
public enum PanelID: Hashable, Sendable {
    case lensData
    case layout
    case analysis(String)

    public init(key: String) {
        switch key {
        case "lens_data": self = .lensData
        case "layout": self = .layout
        default: self = .analysis(key)
        }
    }

    /// The key saved in the project (`lens_data`, `layout`, or the pane key).
    public var key: String {
        switch self {
        case .lensData: "lens_data"
        case .layout: "layout"
        case .analysis(let k): k
        }
    }

    public var paneID: String? { if case .analysis(let k) = self { k } else { nil } }
}

/// How a split lays out its children: side by side or stacked.
public enum SplitAxis: String, Sendable {
    case horizontal = "h"
    case vertical = "v"
}

/// A side of a panel group or of a window's content.
public enum PanelEdge: String, Sendable, CaseIterable {
    case leading, trailing, top, bottom

    public var axis: SplitAxis { self == .leading || self == .trailing ? .horizontal : .vertical }
    /// The new panel comes before the existing content (leading or top).
    public var isLeadingSide: Bool { self == .leading || self == .top }
}

/// Where a dragged panel lands.
public enum PanelDropTarget: Hashable, Sendable {
    /// Into a tab group, at a tab position (`nil`: after the last tab).
    case tab(group: UUID, index: Int?)
    /// Beside a tab group, splitting its space.
    case split(group: UUID, edge: PanelEdge)
    /// Along a whole edge of a window's content.
    case windowEdge(window: UUID, edge: PanelEdge)
    /// Into a window of its own.
    case newWindow
}

// MARK: - Tree

/// Panels shown in one place, one at a time; a tab strip switches between
/// them when there are several.
@MainActor
@Observable
public final class PanelGroup: Identifiable {
    public let id = UUID()
    public internal(set) var panels: [PanelID]
    public var selected: PanelID?

    public init(_ panels: [PanelID], selected: PanelID? = nil) {
        self.panels = panels
        self.selected = selected
    }

    /// The panel on screen.
    public var current: PanelID? {
        if let s = selected, panels.contains(s) { return s }
        return panels.first
    }
}

/// Children side by side (horizontal) or stacked (vertical), each with a
/// share of the space.
@MainActor
@Observable
public final class PanelSplit: Identifiable {
    public let id = UUID()
    public let axis: SplitAxis
    public internal(set) var children: [PanelNode]
    /// The children's shares of the space (they sum to 1).
    public internal(set) var fractions: [Double]

    public init(axis: SplitAxis, children: [PanelNode], fractions: [Double]? = nil) {
        self.axis = axis
        self.children = children
        let f = fractions ?? Array(repeating: 1, count: children.count)
        self.fractions = PanelSplit.normalized(f.count == children.count ? f : Array(repeating: 1, count: children.count))
    }

    /// Moves the divider after child `k`: child `k` takes `share` of the
    /// space the pair has together (the divider's drag).
    public func resize(divider k: Int, fractions pair: (Double, Double)) {
        guard fractions.indices.contains(k + 1) else { return }
        var f = fractions
        f[k] = pair.0
        f[k + 1] = pair.1
        if f != fractions { fractions = f }
    }

    /// Equal shares for every child.
    public func equalize() {
        let f = Array(repeating: 1.0 / Double(max(children.count, 1)), count: children.count)
        if f != fractions { fractions = f }
    }

    func set(children: [PanelNode], fractions f: [Double]) {
        let f = PanelSplit.normalized(f)
        if children.map(\.id) != self.children.map(\.id) { self.children = children }
        if f != fractions { fractions = f }
    }

    static func normalized(_ f: [Double]) -> [Double] {
        let clean = f.map { $0.isFinite && $0 > 0 ? $0 : 0 }
        let sum = clean.reduce(0, +)
        guard sum > 0 else { return Array(repeating: 1 / Double(max(f.count, 1)), count: f.count) }
        return clean.map { $0 / sum }
    }
}

/// A node of a window's panel tree.
public enum PanelNode: Identifiable {
    case group(PanelGroup)
    case split(PanelSplit)

    public var id: UUID {
        switch self {
        case .group(let g): g.id
        case .split(let s): s.id
        }
    }

    /// The tab groups, in reading order.
    @MainActor public var groups: [PanelGroup] {
        switch self {
        case .group(let g): [g]
        case .split(let s): s.children.flatMap(\.groups)
        }
    }

    @MainActor public var panels: [PanelID] { groups.flatMap(\.panels) }
}

/// The panels of one window: the design's main window, or a window that
/// panels were moved out to.
@MainActor
@Observable
public final class PanelWindowModel: Identifiable {
    public let id: UUID
    public internal(set) var root: PanelNode?
    /// A separate window's frame on screen (AppKit coordinates, the origin
    /// at the bottom left), saved with the project.
    public var frame: CGRect?

    public init(id: UUID = UUID(), root: PanelNode?, frame: CGRect? = nil) {
        self.id = id
        self.root = root
        self.frame = frame
    }

    public var panels: [PanelID] { root?.panels ?? [] }
    public var groups: [PanelGroup] { root?.groups ?? [] }
}

// MARK: - Arrangement

/// Where a design's panels are: a tree of splits and tab groups for the
/// main window and for each separate window. A panel is in one place at a
/// time; closing one takes it out of the tree (an analysis pane is then
/// removed from the workspace, the lens data and the layout can be shown
/// again).
///
/// Platform-neutral: the views draw the trees, the window code opens a
/// window for each `detached` model.
@MainActor
@Observable
public final class PanelArrangement {
    public let main: PanelWindowModel
    public private(set) var detached: [PanelWindowModel]

    public init(main root: PanelNode?, detached: [PanelWindowModel] = []) {
        main = PanelWindowModel(root: root)
        self.detached = detached
        normalize()
    }

    public var windows: [PanelWindowModel] { [main] + detached }

    public func window(_ id: UUID) -> PanelWindowModel? { windows.first { $0.id == id } }

    public var allPanels: [PanelID] { windows.flatMap(\.panels) }

    public func contains(_ panel: PanelID) -> Bool { location(of: panel) != nil }

    public func location(of panel: PanelID) -> (window: PanelWindowModel, group: PanelGroup)? {
        for w in windows {
            if let g = w.groups.first(where: { $0.panels.contains(panel) }) { return (w, g) }
        }
        return nil
    }

    public func group(_ id: UUID) -> (window: PanelWindowModel, group: PanelGroup)? {
        for w in windows {
            if let g = w.groups.first(where: { $0.id == id }) { return (w, g) }
        }
        return nil
    }

    // MARK: Showing and closing

    /// Brings a panel to the front of its tab group.
    public func select(_ panel: PanelID) {
        guard case let (_, g)? = location(of: panel), g.selected != panel else { return }
        g.selected = panel
    }

    /// Takes a panel out of the tree; a window left empty goes away.
    public func remove(_ panel: PanelID) {
        guard case let (_, g)? = location(of: panel) else { return }
        take(panel, from: g)
        normalize()
    }

    /// Removes a separate window, returning the panels it showed.
    @discardableResult
    public func discard(_ window: PanelWindowModel) -> [PanelID] {
        guard window !== main else { return [] }
        let panels = window.panels
        detached.removeAll { $0 === window }
        return panels
    }

    /// Moves a separate window's panels into the main window, keeping
    /// their arrangement: as more columns of the area under the lens data
    /// (a lone panel goes where its kind belongs, as `returnToMain`).
    public func dock(_ window: PanelWindowModel) {
        guard window !== main, let root = window.root else {
            discard(window)
            return
        }
        if window.panels.count == 1, let panel = window.panels.first {
            returnToMain(panel)
            return
        }
        detached.removeAll { $0 === window }
        window.root = nil
        if let area = lowerArea {
            addColumn(root, after: area, in: main)
        } else {
            attach(root, to: main, edge: main.root.map { $0.panels == [.lensData] } ?? false ? .bottom : .trailing, share: 0.6)
        }
        normalize()
    }

    /// Moves a panel from a separate window back to where its kind belongs
    /// in the main window: an analysis beside the other analyses, the lens
    /// data across the top, the layout at the left.
    public func returnToMain(_ panel: PanelID) {
        guard let loc = location(of: panel), loc.window !== main else { return }
        take(panel, from: loc.group)
        normalize()
        if case .analysis = panel {
            let neighbour = main.panels.last { $0.paneID != nil }
            place(panel, near: neighbour)
        } else {
            show(panel)
        }
        select(panel)
    }

    /// Replaces the whole arrangement with `other`'s main window (Reset
    /// Panel Layout): every panel in one window.
    public func reset(to other: PanelArrangement) {
        detached = []
        main.root = other.main.root
        normalize()
    }

    /// Shows the lens data or the layout where they belong by default: the
    /// lens data across the top of the main window, the layout at the left
    /// of the area under it.
    public func show(_ panel: PanelID) {
        guard !contains(panel) else { return }
        let node = PanelNode.group(PanelGroup([panel]))
        switch panel {
        case .lensData:
            attach(node, to: main, edge: .top, share: 0.36)
        case .layout:
            if let area = lowerArea {
                insert(node, beside: area, in: main, edge: .leading, share: 0.42)
            } else {
                attach(node, to: main, edge: .bottom, share: 0.64)
            }
        case .analysis:
            place(panel, near: nil)
        }
        normalize()
    }

    /// Places a new analysis pane: as another column beside `hint` (the
    /// pane it was added after), else beside the layout, else in the area
    /// under the lens data, else along the main window's trailing edge.
    public func place(_ panel: PanelID, near hint: PanelID?) {
        guard !contains(panel) else { return }
        let node = PanelNode.group(PanelGroup([panel]))
        if let hint, case let (w, g)? = location(of: hint) {
            addColumn(node, after: .group(g), in: w)
        } else if case let (w, g)? = location(of: .layout), w === main {
            addColumn(node, after: .group(g), in: w)
        } else if let area = lowerArea {
            addColumn(node, after: area, in: main)
        } else if main.root.map({ $0.panels == [.lensData] }) ?? false {
            attach(node, to: main, edge: .bottom, share: 0.64)
        } else {
            attach(node, to: main, edge: .trailing, share: 0.4)
        }
        normalize()
    }

    /// Places a new analysis pane in a given window, along its trailing edge.
    public func place(_ panel: PanelID, inWindow id: UUID) {
        guard !contains(panel), let w = window(id) else { return }
        attach(.group(PanelGroup([panel])), to: w, edge: .trailing, share: 1 / Double(w.groups.count + 1))
        normalize()
    }

    // MARK: Moving

    /// Whether dropping `panel` on `target` changes anything. Dropping the
    /// only panel of a group on that group, or the only panel of a window
    /// on that window's edge, does not.
    public func canMove(_ panel: PanelID, to target: PanelDropTarget) -> Bool {
        let source = location(of: panel)
        switch target {
        case .tab(let gid, _):
            guard case let (_, g)? = group(gid) else { return false }
            return g !== source?.group || g.panels.count > 1
        case .split(let gid, _):
            guard case let (_, g)? = group(gid) else { return false }
            return g !== source?.group || g.panels.count > 1
        case .windowEdge(let wid, _):
            guard let w = window(wid) else { return false }
            return w.panels != [panel]
        case .newWindow:
            guard let source else { return true }
            return source.window === main || source.window.panels != [panel]
        }
    }

    /// Moves a panel. A `newWindow` target opens a separate window with
    /// `frame`; it is returned. Windows left empty go away.
    @discardableResult
    public func move(_ panel: PanelID, to target: PanelDropTarget, frame: CGRect? = nil) -> PanelWindowModel? {
        guard canMove(panel, to: target) else { return nil }
        var opened: PanelWindowModel?
        switch target {
        case .tab(let gid, let index):
            guard case let (_, g)? = group(gid) else { return nil }
            if let i = g.panels.firstIndex(of: panel) {
                // Reorder within the group.
                var p = g.panels
                p.remove(at: i)
                let j = min(max((index ?? p.count) - ((index ?? 0) > i ? 1 : 0), 0), p.count)
                p.insert(panel, at: j)
                g.panels = p
            } else {
                if case let (_, from)? = location(of: panel) { take(panel, from: from) }
                g.panels.insert(panel, at: min(max(index ?? g.panels.count, 0), g.panels.count))
            }
            g.selected = panel
        case .split(let gid, let edge):
            if case let (_, from)? = location(of: panel) { take(panel, from: from) }
            normalize()
            guard case let (w, g)? = group(gid) else { return nil }
            insert(.group(PanelGroup([panel])), beside: .group(g), in: w, edge: edge, share: 0.5)
        case .windowEdge(let wid, let edge):
            if case let (_, from)? = location(of: panel) { take(panel, from: from) }
            normalize()
            guard let w = window(wid) else { return nil }
            attach(.group(PanelGroup([panel])), to: w, edge: edge, share: 0.3)
        case .newWindow:
            if case let (_, from)? = location(of: panel) { take(panel, from: from) }
            let w = PanelWindowModel(root: .group(PanelGroup([panel])), frame: frame)
            detached.append(w)
            opened = w
        }
        normalize()
        return opened
    }

    // MARK: Tree editing

    private func take(_ panel: PanelID, from g: PanelGroup) {
        guard let i = g.panels.firstIndex(of: panel) else { return }
        g.panels.remove(at: i)
        if g.selected == panel { g.selected = g.panels.isEmpty ? nil : g.panels[min(i, g.panels.count - 1)] }
    }

    /// The area under the lens data in the main window (the last child of
    /// a vertical root), or the whole content when there is no lens data
    /// on top; `nil` when the main window shows only the lens data.
    private var lowerArea: PanelNode? {
        guard let root = main.root else { return nil }
        if case .split(let s) = root, s.axis == .vertical, let last = s.children.last, s.children.first?.panels.contains(.lensData) == true {
            return last
        }
        if root.panels == [.lensData] { return nil }
        if root.panels.contains(.lensData) { return nil }
        return root
    }

    /// Adds `node` as a column after `anchor`: in a row, an equal share of
    /// it; otherwise splitting the anchor's space.
    private func addColumn(_ node: PanelNode, after anchor: PanelNode, in w: PanelWindowModel) {
        if case let (parent, i)? = parent(of: anchor.id, in: w.root), parent.axis == .horizontal {
            let n = Double(parent.children.count)
            var kids = parent.children, f = parent.fractions.map { $0 * n / (n + 1) }
            kids.insert(node, at: i + 1)
            f.insert(1 / (n + 1), at: i + 1)
            parent.set(children: kids, fractions: f)
        } else if case .split(let s) = anchor, s.axis == .horizontal {
            let n = Double(s.children.count)
            s.set(children: s.children + [node], fractions: s.fractions.map { $0 * n / (n + 1) } + [1 / (n + 1)])
        } else {
            insert(node, beside: anchor, in: w, edge: .trailing, share: 0.5)
        }
    }

    /// Puts `node` beside `anchor`, taking `share` of the anchor's space.
    private func insert(_ node: PanelNode, beside anchor: PanelNode, in w: PanelWindowModel, edge: PanelEdge, share: Double) {
        if case let (parent, i)? = parent(of: anchor.id, in: w.root), parent.axis == edge.axis {
            var kids = parent.children, f = parent.fractions
            let taken = f[i] * share
            f[i] -= taken
            let at = edge.isLeadingSide ? i : i + 1
            kids.insert(node, at: at)
            f.insert(taken, at: at)
            parent.set(children: kids, fractions: f)
            return
        }
        if case .split(let s) = anchor, s.axis == edge.axis {
            let keep = s.fractions.map { $0 * (1 - share) }
            if edge.isLeadingSide {
                s.set(children: [node] + s.children, fractions: [share] + keep)
            } else {
                s.set(children: s.children + [node], fractions: keep + [share])
            }
            return
        }
        let split = PanelSplit(
            axis: edge.axis, children: edge.isLeadingSide ? [node, anchor] : [anchor, node],
            fractions: edge.isLeadingSide ? [share, 1 - share] : [1 - share, share])
        replace(anchor.id, with: .split(split), in: w)
    }

    /// Puts `node` along a whole edge of a window's content.
    private func attach(_ node: PanelNode, to w: PanelWindowModel, edge: PanelEdge, share: Double) {
        guard let root = w.root else {
            w.root = node
            return
        }
        insert(node, beside: root, in: w, edge: edge, share: share)
    }

    private func replace(_ id: UUID, with node: PanelNode, in w: PanelWindowModel) {
        if w.root?.id == id {
            w.root = node
        } else if case let (parent, i)? = parent(of: id, in: w.root) {
            var kids = parent.children
            kids[i] = node
            parent.set(children: kids, fractions: parent.fractions)
        }
    }

    private func parent(of id: UUID, in node: PanelNode?) -> (PanelSplit, Int)? {
        guard case .split(let s) = node else { return nil }
        if let i = s.children.firstIndex(where: { $0.id == id }) { return (s, i) }
        for c in s.children {
            if let found = parent(of: id, in: c) { return found }
        }
        return nil
    }

    /// Drops empty groups, collapses splits of one child, merges a split
    /// into a parent of the same axis, and removes empty separate windows.
    func normalize() {
        for w in windows {
            let n = w.root.flatMap(normalized)
            if n?.id != w.root?.id { w.root = n }
        }
        let empty = detached.filter { $0.root == nil }
        if !empty.isEmpty { detached.removeAll { $0.root == nil } }
    }

    private func normalized(_ node: PanelNode) -> PanelNode? {
        switch node {
        case .group(let g):
            return g.panels.isEmpty ? nil : node
        case .split(let s):
            var kids: [PanelNode] = [], f: [Double] = []
            for (c, share) in zip(s.children, s.fractions) {
                guard let n = normalized(c) else { continue }
                if case .split(let inner) = n, inner.axis == s.axis {
                    kids += inner.children
                    f += inner.fractions.map { $0 * share }
                } else {
                    kids.append(n)
                    f.append(share)
                }
            }
            if kids.isEmpty { return nil }
            if kids.count == 1 { return kids[0] }
            s.set(children: kids, fractions: f)
            return node
        }
    }
}

// MARK: - Defaults and saving

extension PanelArrangement {
    /// The default window: the lens data across the top; under it the
    /// layout and the analysis panes side by side.
    public static func standard(
        panes: [String], showLensData: Bool = true, showLayout: Bool = true, editorFraction: Double = 0.36,
        viewportFraction: Double = 0.42, widths: [String: Double] = [:]
    ) -> PanelArrangement {
        var lower: [PanelNode] = [], f: [Double] = []
        if showLayout {
            lower.append(.group(PanelGroup([.layout])))
            f.append(panes.isEmpty ? 1 : viewportFraction)
        }
        let weights = panes.map { max(widths[$0] ?? 1, 0.05) }
        let total = weights.reduce(0, +)
        for (key, w) in zip(panes, weights) {
            lower.append(.group(PanelGroup([.analysis(key)])))
            f.append((showLayout ? 1 - viewportFraction : 1) * w / max(total, 1e-9))
        }
        let below: PanelNode? = lower.isEmpty ? nil : lower.count == 1 ? lower[0] : .split(PanelSplit(axis: .horizontal, children: lower, fractions: f))
        let top: PanelNode? = showLensData ? .group(PanelGroup([.lensData])) : nil
        let root: PanelNode?
        switch (top, below) {
        case (let t?, let b?): root = .split(PanelSplit(axis: .vertical, children: [t, b], fractions: [editorFraction, 1 - editorFraction]))
        case (let t?, nil): root = t
        case (nil, let b): root = b
        }
        return PanelArrangement(main: root)
    }

    /// The arrangement as saved in the project:
    /// `{"main": node, "windows": [{"frame": [x, y, w, h], "root": node}]}`,
    /// a node being `{"tabs": [keys], "selected": key}` or
    /// `{"axis": "h"|"v", "children": [nodes], "fractions": [shares]}`.
    public var json: JSONValue {
        var out: [String: JSONValue] = [:]
        if let r = main.root { out["main"] = Self.encode(r) }
        out["windows"] = .array(
            detached.compactMap { w in
                guard let r = w.root else { return nil }
                var o: [String: JSONValue] = ["root": Self.encode(r)]
                if let f = w.frame { o["frame"] = .array([f.minX, f.minY, f.width, f.height].map { (v: CGFloat) -> JSONValue in .number((Double(v) * 10).rounded() / 10) }) }
                return .object(o)
            })
        return .object(out)
    }

    /// Reads a saved arrangement, keeping only the panels in `known` (each
    /// once); `nil` when `json` is not one.
    public static func decode(_ json: JSONValue?, known: Set<PanelID>) -> PanelArrangement? {
        guard let json, json.object != nil else { return nil }
        var seen = Set<PanelID>()
        let main = json["main"].flatMap { decode(node: $0, known: known, seen: &seen) }
        var windows: [PanelWindowModel] = []
        for w in json["windows"]?.array ?? [] {
            guard let root = w["root"].flatMap({ decode(node: $0, known: known, seen: &seen) }) else { continue }
            let f = (w["frame"]?.array ?? []).compactMap(\.doubleValue)
            let frame = f.count == 4 && f.allSatisfy(\.isFinite) && f[2] > 100 && f[3] > 100 ? CGRect(x: f[0], y: f[1], width: f[2], height: f[3]) : nil
            windows.append(PanelWindowModel(root: root, frame: frame))
        }
        return PanelArrangement(main: main, detached: windows)
    }

    private static func encode(_ node: PanelNode) -> JSONValue {
        switch node {
        case .group(let g):
            var o: [String: JSONValue] = ["tabs": .array(g.panels.map { .string($0.key) })]
            if let s = g.current, g.panels.count > 1 { o["selected"] = .string(s.key) }
            return .object(o)
        case .split(let s):
            return .object([
                "axis": .string(s.axis.rawValue), "children": .array(s.children.map(encode)),
                "fractions": .array(s.fractions.map { .number(($0 * 10000).rounded() / 10000) }),
            ])
        }
    }

    private static func decode(node: JSONValue, known: Set<PanelID>, seen: inout Set<PanelID>) -> PanelNode? {
        if let tabs = node["tabs"]?.array {
            var panels: [PanelID] = []
            for t in tabs {
                guard let k = t.string else { continue }
                let p = PanelID(key: k)
                if known.contains(p), seen.insert(p).inserted { panels.append(p) }
            }
            guard !panels.isEmpty else { return nil }
            return .group(PanelGroup(panels, selected: node["selected"]?.string.map(PanelID.init(key:))))
        }
        guard let axis = node["axis"]?.string.flatMap(SplitAxis.init(rawValue:)), let children = node["children"]?.array else { return nil }
        let shares = node["fractions"]?.array?.compactMap(\.doubleValue) ?? []
        var kids: [PanelNode] = [], f: [Double] = []
        for (i, c) in children.enumerated() {
            guard let n = decode(node: c, known: known, seen: &seen) else { continue }
            kids.append(n)
            f.append(shares.indices.contains(i) ? shares[i] : 1)
        }
        guard !kids.isEmpty else { return nil }
        return kids.count == 1 ? kids[0] : .split(PanelSplit(axis: axis, children: kids, fractions: f))
    }
}
