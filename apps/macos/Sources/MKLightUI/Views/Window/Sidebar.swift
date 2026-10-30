import MKLightEngine
import MKLightModel
import SwiftUI

/// The sidebar: every analysis the engine offers, by category, searchable.
/// Selecting one shows it in the focused pane; ⌥-click or the context menu
/// opens a new pane. Open analyses carry a dot; unavailable ones say why.
struct AnalysisSidebar: View {
    let workspace: Workspace
    @Bindable var window: WindowState
    @State private var query = ""

    var body: some View {
        List(selection: selection) {
            if !workspace.issues.isEmpty {
                Section("Design") {
                    Button {
                        window.showIssues = true
                    } label: {
                        Label {
                            Text("Issues")
                        } icon: {
                            Image(systemName: worst == .error ? "xmark.octagon" : "exclamationmark.triangle")
                                .foregroundStyle(worst == .error ? .red : .orange)
                        }
                        .badge(workspace.issues.count)
                    }
                    .buttonStyle(.plain)
                }
            }
            ForEach(groups, id: \.category) { group in
                Section(AnalysisCatalog.categoryTitle(group.category)) {
                    ForEach(group.items) { d in row(d).tag(d.name) }
                }
            }
        }
        .listStyle(.sidebar)
        .searchable(text: $query, placement: .sidebar, prompt: "Analyses")
        .overlay {
            if workspace.descriptors.isEmpty {
                ProgressView().controlSize(.small)
            } else if groups.isEmpty {
                ContentUnavailableView.search(text: query)
            }
        }
        .navigationTitle("Analyses")
    }

    private var worst: ValidationIssue.Severity { workspace.issues.map(\.severity).min() ?? .info }

    private var groups: [(category: String, items: [AnalysisDescriptor])] {
        let q = query.trimmingCharacters(in: .whitespaces).lowercased()
        let list = workspace.paneAnalyses.filter {
            q.isEmpty || AnalysisCatalog.title(of: $0).lowercased().contains(q) || $0.name.contains(q)
                || $0.description.lowercased().contains(q)
        }
        return AnalysisCatalog.grouped(list)
    }

    /// The focused pane's analysis is the sidebar selection.
    private var selection: Binding<String?> {
        Binding(
            get: { workspace.focusedPane?.analysis },
            set: { name in
                guard let name else { return }
                #if os(macOS)
                let newPane = NSEvent.modifierFlags.contains(.option)
                #else
                let newPane = false
                #endif
                if workspace.panes.isEmpty { workspace.addPane(name) } else { workspace.open(name, inNewPane: newPane) }
            })
    }

    private func row(_ d: AnalysisDescriptor) -> some View {
        let open = workspace.panes.first { $0.analysis == d.name }
        return HStack(spacing: Space.s) {
            Label(AnalysisCatalog.title(of: d), systemImage: AnalysisCatalog.symbol(for: d.name))
                .foregroundStyle(d.available ? .primary : .tertiary)
            Spacer(minLength: 0)
            if let open, open.isRunning {
                ProgressView().controlSize(.mini)
            } else if open != nil {
                Circle().fill(.secondary).frame(width: 5, height: 5).accessibilityLabel("open")
            }
        }
        .help(d.available ? d.description : (d.unavailableReason ?? "Not available in this engine build"))
        .contextMenu {
            Button("Open in New Pane") { workspace.addPane(d.name) }.disabled(!d.available)
            if let open { Button("Close Pane") { workspace.removePane(open) } }
        }
    }
}

/// Toolbar status of the design's validation, opening the issue list.
struct IssuesButton: View {
    let workspace: Workspace
    @Bindable var window: WindowState

    var body: some View {
        let issues = workspace.issues
        let errors = issues.filter { $0.severity == .error }.count
        Button {
            window.showIssues.toggle()
        } label: {
            if issues.isEmpty {
                Label("No Issues", systemImage: "checkmark.circle").labelStyle(.iconOnly).foregroundStyle(.secondary)
            } else {
                Label("\(issues.count)", systemImage: errors > 0 ? "xmark.octagon.fill" : "exclamationmark.triangle.fill")
                    .labelStyle(.titleAndIcon)
                    .foregroundStyle(errors > 0 ? .red : .orange)
                    .monospacedDigit()
            }
        }
        .accessibilityLabel(issues.isEmpty ? "No issues" : "\(issues.count) issue\(issues.count == 1 ? "" : "s")")
        .help(issues.isEmpty ? "The engine found no problems in the design" : "Show the design's validation issues (⌃⌘E)")
        .popover(isPresented: $window.showIssues, arrowEdge: .bottom) {
            IssuesList(workspace: workspace, window: window).frame(width: 420)
        }
    }
}

/// Validation issues with severity, code and message. Selecting one selects
/// the surface or field it concerns.
struct IssuesList: View {
    let workspace: Workspace
    @Bindable var window: WindowState

    var body: some View {
        let issues = workspace.issues.sorted { ($0.severity, $0.id) < ($1.severity, $1.id) }
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Issues").font(.headline)
                Spacer()
                Text(summary).font(.caption).foregroundStyle(.secondary)
            }
            .padding(Space.l)
            Divider()
            if issues.isEmpty {
                ContentUnavailableView(
                    "No Issues", systemImage: "checkmark.circle",
                    description: Text("The engine checked the design and found nothing to report."))
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: Space.s) {
                        ForEach(issues) { issue in
                            Button {
                                select(issue)
                            } label: {
                                IssueRow(issue: issue, system: workspace.snapshot?.system)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                            .buttonStyle(.plain)
                            .help(issue.surface != nil ? "Select surface \(issue.surface!)" : issue.field != nil ? "Show field \(issue.field! + 1)" : "")
                        }
                    }
                    .padding(Space.l)
                }
                .frame(maxHeight: 360)
                .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var summary: String {
        let e = workspace.issues.filter { $0.severity == .error }.count
        let w = workspace.issues.filter { $0.severity == .warning }.count
        return [e > 0 ? "\(e) error\(e == 1 ? "" : "s")" : nil, w > 0 ? "\(w) warning\(w == 1 ? "" : "s")" : nil]
            .compactMap { $0 }.joined(separator: ", ")
    }

    private func select(_ issue: ValidationIssue) {
        if let s = issue.surface, let sys = workspace.snapshot?.system, sys.surfaces.indices.contains(s) {
            window.selection = sys.surfaces[s].id
            window.inspectorTab = .surface
            window.showInspector = true
        } else if let f = issue.field {
            window.selection = nil
            window.highlightedField = f
            window.inspectorTab = .system
            window.showInspector = true
        }
        window.showIssues = false
    }
}

struct IssueRow: View {
    let issue: ValidationIssue
    let system: OpticalSystem?

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: Space.m) {
            Image(systemName: icon).foregroundStyle(color).accessibilityLabel(issue.severity.rawValue)
            VStack(alignment: .leading, spacing: 2) {
                Text(issue.message).fixedSize(horizontal: false, vertical: true)
                HStack(spacing: Space.s) {
                    if !issue.code.isEmpty { Text(issue.code).monospaced() }
                    if let s = issue.surface { Text("Surface \(s)") }
                    if let f = issue.field { Text("Field \(f + 1)") }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 2)
        .contentShape(Rectangle())
    }

    private var icon: String {
        switch issue.severity {
        case .error: "xmark.octagon.fill"
        case .warning: "exclamationmark.triangle.fill"
        case .info: "info.circle.fill"
        }
    }
    private var color: Color {
        switch issue.severity {
        case .error: .red
        case .warning: .orange
        case .info: .secondary
        }
    }
}
