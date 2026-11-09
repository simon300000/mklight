import CoreGraphics
import Foundation
import Testing

@testable import MKLightModel
@testable import MKLightUI

/// The panel tree: moving panels into tabs, splits and separate windows,
/// showing and hiding, and saving it with the project.
@MainActor
struct PanelArrangementTests {
    /// The tree as text: `v(...)`/`h(...)` for splits, `[a,b]` for tab groups
    /// (`*` marks the front tab of a group of several).
    func text(_ node: PanelNode?) -> String {
        guard let node else { return "-" }
        switch node {
        case .group(let g):
            return "[" + g.panels.map { p in (g.panels.count > 1 && g.current == p ? "*" : "") + short(p) }.joined(separator: ",") + "]"
        case .split:
            return describe(node)
        }
    }

    func short(_ p: PanelID) -> String {
        switch p {
        case .lensData: "lens"
        case .layout: "layout"
        case .analysis(let k): k
        }
    }

    func tree(_ a: PanelArrangement) -> String { describe(a.main.root) }

    func describe(_ node: PanelNode?) -> String {
        guard let node else { return "-" }
        switch node {
        case .group: return text(node)
        case .split(let s): return (s.axis == .horizontal ? "h(" : "v(") + s.children.map(describe).joined(separator: "|") + ")"
        }
    }

    func group(_ a: PanelArrangement, _ p: PanelID) -> PanelGroup { a.location(of: p)!.group }

    @Test func standardArrangement() {
        let a = PanelArrangement.standard(panes: ["p1", "p2"])
        #expect(tree(a) == "v([lens]|h([layout]|[p1]|[p2]))")
        guard case .split(let root) = a.main.root!, case .split(let lower) = root.children[1] else { Issue.record("not split"); return }
        #expect(abs(root.fractions[0] - 0.36) < 1e-9)
        #expect(abs(lower.fractions[0] - 0.42) < 1e-9 && abs(lower.fractions[1] - 0.29) < 1e-9)
        #expect(tree(.standard(panes: [], showLayout: false)) == "[lens]")
        #expect(tree(.standard(panes: ["p1"], showLensData: false)) == "h([layout]|[p1])")
    }

    @Test func dropOnAGroupMakesTabs() {
        let a = PanelArrangement.standard(panes: ["p1", "p2"])
        let target = group(a, .analysis("p1"))
        a.move(.analysis("p2"), to: .tab(group: target.id, index: nil))
        #expect(tree(a) == "v([lens]|h([layout]|[p1,*p2]))")
        // Reorder within the group, then take a tab out beside the layout.
        a.move(.analysis("p2"), to: .tab(group: target.id, index: 0))
        #expect(tree(a) == "v([lens]|h([layout]|[*p2,p1]))")
        a.move(.analysis("p1"), to: .split(group: group(a, .layout).id, edge: .bottom))
        #expect(tree(a) == "v([lens]|h(v([layout]|[p1])|[p2]))")
    }

    @Test func splitsMergeIntoARowOfTheSameAxis() {
        let a = PanelArrangement.standard(panes: ["p1", "p2"])
        a.move(.analysis("p2"), to: .split(group: group(a, .layout).id, edge: .leading))
        #expect(tree(a) == "v([lens]|h([p2]|[layout]|[p1]))")
        a.move(.lensData, to: .windowEdge(window: a.main.id, edge: .trailing))
        #expect(tree(a) == "h([p2]|[layout]|[p1]|[lens])")
        var shares: [Double] = []
        if case .split(let s)? = a.main.root { shares = s.fractions }
        #expect(abs(shares.reduce(0, +) - 1) < 1e-9 && abs(shares.last! - 0.3) < 1e-9)
    }

    @Test func nothingHappensWhereNothingWouldChange() {
        let a = PanelArrangement.standard(panes: ["p1"])
        let g = group(a, .analysis("p1"))
        #expect(!a.canMove(.analysis("p1"), to: .tab(group: g.id, index: nil)))
        #expect(!a.canMove(.analysis("p1"), to: .split(group: g.id, edge: .leading)))
        #expect(a.canMove(.analysis("p1"), to: .newWindow))
        let w = a.move(.analysis("p1"), to: .newWindow, frame: CGRect(x: 10, y: 20, width: 500, height: 400))!
        // The only panel of a separate window: its window is the panel.
        #expect(!a.canMove(.analysis("p1"), to: .newWindow))
        #expect(!a.canMove(.analysis("p1"), to: .windowEdge(window: w.id, edge: .leading)))
        #expect(a.canMove(.analysis("p1"), to: .windowEdge(window: a.main.id, edge: .trailing)))
    }

    @Test func separateWindowsComeAndGo() {
        let a = PanelArrangement.standard(panes: ["p1", "p2"])
        let w = a.move(.analysis("p1"), to: .newWindow, frame: CGRect(x: 10, y: 20, width: 500, height: 400))!
        #expect(a.detached.count == 1 && w.panels == [.analysis("p1")] && w.frame?.width == 500)
        #expect(tree(a) == "v([lens]|h([layout]|[p2]))")
        // A second panel into the separate window, beside the first.
        a.move(.analysis("p2"), to: .split(group: group(a, .analysis("p1")).id, edge: .trailing))
        #expect(describe(w.root) == "h([p1]|[p2])")
        // Docking puts the window's arrangement back under the lens data.
        a.dock(w)
        #expect(a.detached.isEmpty)
        #expect(tree(a) == "v([lens]|h([layout]|[p1]|[p2]))")
        // A lone panel goes back where its kind belongs.
        a.move(.analysis("p1"), to: .newWindow)
        a.move(.lensData, to: .newWindow)
        a.returnToMain(.analysis("p1"))
        #expect(tree(a) == "h([layout]|[p2]|[p1])")
        a.dock(a.detached[0])
        #expect(tree(a) == "v([lens]|h([layout]|[p2]|[p1]))" && a.detached.isEmpty)
        // Moving a separate window's last panel away closes the window.
        let w2 = a.move(.lensData, to: .newWindow)!
        #expect(a.detached.map(\.id) == [w2.id])
        a.move(.lensData, to: .windowEdge(window: a.main.id, edge: .top))
        #expect(a.detached.isEmpty)
        #expect(a.discard(a.main).isEmpty)
    }

    @Test func showingAndHiding() {
        let a = PanelArrangement.standard(panes: ["p1"])
        a.remove(.layout)
        #expect(tree(a) == "v([lens]|[p1])")
        a.show(.layout)
        #expect(tree(a) == "v([lens]|h([layout]|[p1]))")
        a.remove(.lensData)
        #expect(tree(a) == "h([layout]|[p1])")
        a.show(.lensData)
        #expect(tree(a) == "v([lens]|h([layout]|[p1]))")
        a.remove(.layout)
        a.remove(.analysis("p1"))
        a.remove(.lensData)
        #expect(a.main.root == nil)
        a.show(.layout)
        a.show(.lensData)
        #expect(tree(a) == "v([lens]|[layout])")
    }

    @Test func newPanesGoBesideTheirNeighbour() {
        let a = PanelArrangement.standard(panes: ["p1", "p2"])
        a.place(.analysis("p3"), near: .analysis("p1"))
        #expect(tree(a) == "v([lens]|h([layout]|[p1]|[p3]|[p2]))")
        let a2 = PanelArrangement.standard(panes: [])
        a2.place(.analysis("p1"), near: nil)
        #expect(tree(a2) == "v([lens]|h([layout]|[p1]))")
        let a3 = PanelArrangement.standard(panes: [], showLayout: false)
        a3.place(.analysis("p1"), near: nil)
        #expect(tree(a3) == "v([lens]|[p1])")
        let w = a.move(.analysis("p2"), to: .newWindow)!
        a.place(.analysis("p4"), inWindow: w.id)
        #expect(describe(w.root) == "h([p2]|[p4])")
    }

    @Test func savedWithTheProject() throws {
        let a = PanelArrangement.standard(panes: ["p1", "p2", "p3"])
        a.move(.analysis("p2"), to: .tab(group: group(a, .analysis("p1")).id, index: nil))
        a.move(.analysis("p3"), to: .newWindow, frame: CGRect(x: 100, y: 120, width: 640, height: 480))
        let known = Set<PanelID>([.lensData, .layout, .analysis("p1"), .analysis("p2"), .analysis("p3")])
        let back = try #require(PanelArrangement.decode(a.json, known: known))
        #expect(tree(back) == tree(a))
        #expect(back.detached.count == 1 && back.detached[0].frame == CGRect(x: 100, y: 120, width: 640, height: 480))
        #expect(describe(back.detached[0].root) == "[p3]")
        // Unknown panes (closed since) and duplicates are dropped; the tree tidies up.
        let fewer = try #require(PanelArrangement.decode(a.json, known: [.lensData, .analysis("p1")]))
        #expect(tree(fewer) == "v([lens]|[p1])" && fewer.detached.isEmpty)
        #expect(PanelArrangement.decode(.array([]), known: known) == nil)
    }

    /// Views without an arrangement open in the standard one; a saved
    /// arrangement is kept.
    @Test func workspaceLayoutCarriesTheArrangement() throws {
        let plain: JSONValue = .array([
            .object(["key": .string("layout"), "ui": .object([:])]),
            .object(["key": .string("p1"), "analysis": .string("spot_diagram")]),
            .object(["key": .string("p2"), "analysis": .string("ray_fan")]),
        ])
        let standard = WorkspaceLayout(views: plain).makeArrangement()
        #expect(tree(standard) == "v([lens]|h([layout]|[p1]|[p2]))")
        guard case .split(let root) = standard.main.root!, case .split(let row) = root.children[1] else { Issue.record("not split"); return }
        #expect(abs(root.fractions[0] - 0.36) < 1e-9 && abs(row.fractions[1] - row.fractions[2]) < 1e-9)

        var saved = WorkspaceLayout(views: plain)
        standard.move(.analysis("p2"), to: .newWindow, frame: CGRect(x: 0, y: 0, width: 600, height: 400))
        saved.arrangement = standard.json
        let reread = WorkspaceLayout(views: saved.views)
        #expect(reread.arrangement == standard.json)
        let again = reread.makeArrangement()
        #expect(tree(again) == "v([lens]|h([layout]|[p1]))" && again.detached.count == 1)
    }
}
