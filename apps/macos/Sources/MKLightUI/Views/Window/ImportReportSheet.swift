import MKLightEngine
import SwiftUI

/// The sheet a design window shows once after importing a lens prescription
/// file: what was left out or changed, grouped by meaning, each item with
/// its surface and line in the file.
///
/// ```
/// ┌──────────────────────────────────────────────┐
/// │ ⬇︎  Imported “lens.zmx”                        │
/// │    3 items could not be imported exactly. …  │
/// │    6 surfaces · .zmx · UTF-16                │
/// ├──────────────────────────────────────────────┤
/// │ ⚠︎ Glasses                                   2 │
/// │   Glass X is not in the loaded catalogs; …    │
/// │   Surface 1 · Line 51                         │
/// │ …                                             │
/// ├──────────────────────────────────────────────┤
/// │ Copy Report        Load Glass Catalog…  Done │
/// └──────────────────────────────────────────────┘
/// ```
struct ImportReportSheet: View {
    let report: ImportReport
    /// Offered when glasses were not found: load a catalog, then import the
    /// file again. `nil` hides the button.
    var loadCatalog: (() -> Void)?
    @Environment(\.dismiss) private var dismiss
    @State private var copied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
                .padding([.horizontal, .top], Space.xxl)
                .padding(.bottom, Space.l)
            List {
                ForEach(report.groups) { group in
                    Section {
                        ForEach(group.items) { item in
                            row(item.warning)
                        }
                    } header: {
                        HStack {
                            Label(group.category.title, systemImage: group.category.symbol)
                            Spacer()
                            Text("\(group.items.count)").monospacedDigit().foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .listStyle(.inset)
            .scrollEdgeEffectStyle(.soft, for: .top)
            Divider()
            footer.padding(Space.xl)
        }
        .frame(width: 620, height: 480)
    }

    private var header: some View {
        HStack(alignment: .top, spacing: Space.l) {
            Image(systemName: "arrow.down.document")
                .font(.system(size: 34, weight: .regular))
                .foregroundStyle(.tint)
                .symbolRenderingMode(.hierarchical)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: Space.xs) {
                Text("Imported “\(report.fileName)”").font(.title3.weight(.semibold))
                Text(report.headline).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                Text(report.summary).font(.caption).foregroundStyle(.tertiary)
            }
        }
        .accessibilityElement(children: .combine)
    }

    private func row(_ w: ImportWarning) -> some View {
        VStack(alignment: .leading, spacing: Space.xxs) {
            Text(ImportReport.sentence(w))
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
            if let location = ImportReport.location(w) {
                Text(location).font(.caption).foregroundStyle(.secondary).monospacedDigit()
            }
        }
        .padding(.vertical, Space.xxs)
        .accessibilityElement(children: .combine)
    }

    private var footer: some View {
        HStack(spacing: Space.m) {
            Button(copied ? "Copied" : "Copy Report") {
                Clipboard.copy(report.plainText)
                copied = true
            }
            .help("Copies the list as text")
            Spacer()
            if let loadCatalog, report.hasUnknownGlasses {
                Button("Load Glass Catalog…") {
                    loadCatalog()
                    dismiss()
                }
                .help("Loads a glass catalog and imports the file again with it")
            }
            Button("Done") { dismiss() }
                .keyboardShortcut(.defaultAction)
        }
        .controlSize(.large)
    }
}
