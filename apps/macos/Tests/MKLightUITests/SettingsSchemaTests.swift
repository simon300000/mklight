import Foundation
import Testing

@testable import MKLightEngine
@testable import MKLightModel
@testable import MKLightUI

/// The schema-to-control mapping: every shape the engine's settings schemas
/// use gets a real control, never a JSON field.
@Suite @MainActor struct SettingsSchemaTests {
    private func field(_ name: String, _ schema: [String: JSONValue]) -> SettingsField {
        SettingsField(name: name, schema: .object(schema), required: false)
    }

    @Test func scalarsAndSelections() {
        #expect(SettingControls.control(for: field("rings", ["type": .string("integer"), "default": .integer(6)]), analysis: "spot_diagram") == .number(integer: true))
        #expect(SettingControls.control(for: field("defocus", ["type": .string("number"), "default": .number(0)]), analysis: "spot_diagram") == .number(integer: false))
        #expect(SettingControls.control(for: field("lead_in", ["type": .array([.string("number"), .string("null")]), "default": .null]), analysis: "layout") == .number(integer: false))
        #expect(SettingControls.control(for: field("include_points", ["type": .string("boolean"), "default": .bool(true)]), analysis: "spot_diagram") == .toggle)
        #expect(SettingControls.control(for: field("pattern", ["type": .string("string"), "enum": .array([.string("hexapolar"), .string("square")]), "default": .string("hexapolar")]), analysis: "spot_diagram") == .picker)
        #expect(SettingControls.control(for: field("name", ["type": .string("string")]), analysis: "glass_info") == .text)
        #expect(SettingControls.control(for: field("catalog", ["type": .array([.string("string"), .string("null")]), "default": .null]), analysis: "glass_info") == .text)
        let selection: [String: JSONValue] = ["type": .string("object"), "default": .object(["type": .string("all")])]
        #expect(SettingControls.control(for: field("fields", selection), analysis: "spot_diagram") == .fieldSelection)
        #expect(SettingControls.control(for: field("wavelengths", selection), analysis: "spot_diagram") == .wavelengthSelection)
        #expect(SettingControls.control(for: field("field", ["type": .string("integer"), "default": .integer(0)]), analysis: "psf") == .fieldIndex)
        #expect(SettingControls.control(for: field("wavelength", ["type": .array([.string("integer"), .string("null")]), "default": .null]), analysis: "ghost_paths") == .wavelengthIndex)
        #expect(SettingControls.control(for: field("surface", ["type": .string("integer"), "default": .integer(1)]), analysis: "footprint") == .surfaceIndex)
        // A seed is hidden unless the pattern is random.
        let seed = field("seed", ["type": .string("integer"), "default": .integer(1)])
        #expect(SettingControls.control(for: seed, analysis: "spot_diagram", pattern: "hexapolar") == .hidden)
        #expect(SettingControls.control(for: seed, analysis: "spot_diagram", pattern: "random") == .number(integer: true))
        #expect(SettingControls.control(for: field("configuration", ["type": .string("integer"), "default": .integer(0)]), analysis: "mtf") == .hidden)
        #expect(SettingControls.control(for: field("configuration", ["type": .string("integer"), "default": .integer(0)]), analysis: "mtf", hasConfigurations: true) == .number(integer: true))
    }

    @Test func arrays() {
        #expect(SettingControls.control(for: field("grid", ["type": .string("array"), "default": .array([.integer(5), .integer(5)])]), analysis: "chief_ray_angle") == .pair(integer: true, nullable: false))
        #expect(SettingControls.control(for: field("extent", ["type": .string("array"), "default": .array([.number(1), .number(1)])]), analysis: "chief_ray_angle") == .pair(integer: false, nullable: false))
        // `[min, max]` or `[x, y]` that is automatic when null.
        #expect(SettingControls.control(for: field("range", ["type": .array([.string("array"), .string("null")]), "items": .object(["type": .string("number")]), "default": .null]), analysis: "best_focus") == .pair(integer: false, nullable: true))
        #expect(SettingControls.control(for: field("point", ["type": .array([.string("array"), .string("null")]), "items": .object(["type": .string("number")]), "default": .null]), analysis: "polarization_pupil_map") == .pair(integer: false, nullable: true))
        // Index lists name the system's fields, wavelengths or surfaces.
        let ints: [String: JSONValue] = ["type": .string("array"), "items": .object(["type": .string("integer"), "minimum": .integer(0)]), "default": .array([])]
        #expect(SettingControls.control(for: field("wavelengths", ints), analysis: "system_transmission") == .indexList(.wavelengths))
        #expect(SettingControls.control(for: field("fields", ints), analysis: "ghost_irradiance") == .indexList(.fields))
        #expect(SettingControls.control(for: field("exclude_surfaces", ints), analysis: "ghost_paths", nested: true) == .indexList(.surfaces))
        #expect(SettingControls.control(for: field("diffraction_orders", ints), analysis: "ghost_paths", nested: true) == .numberList)
        // Number lists; the MTF frequencies keep their automatic control.
        let numbers: [String: JSONValue] = ["type": .string("array"), "items": .object(["type": .string("number")]), "default": .array([]), "unit": .string("cycles/mm")]
        #expect(SettingControls.control(for: field("frequencies", numbers), analysis: "mtf_vs_field") == .frequencies)
        #expect(SettingControls.control(for: field("percentiles", numbers), analysis: "tolerance") == .numberList)
        #expect(SettingControls.control(for: field("wavelengths_um", ["type": .array([.string("array"), .string("null")]), "items": .object(["type": .string("number")]), "default": .null]), analysis: "glass_info") == .pair(integer: false, nullable: true))
        // Rows of the same columns: objects named by the description, or fixed-length arrays.
        let points = field("fields", ["type": .string("array"), "default": .array([]), "description": .string("Normalised field points [{\"hx\", \"hy\"}]; empty = the system fields.")])
        guard case .table(let columns, let rowsAreArrays) = SettingControls.control(for: points, analysis: "system_transmission") else {
            Issue.record("field points are a table")
            return
        }
        #expect(columns.map(\.name) == ["hx", "hy"] && !rowsAreArrays)
        #expect(SettingControls.control(for: field("sensor_limit", ["type": .string("array"), "default": .array([])]), analysis: "chief_ray_angle") == .sensorLimit)
        let rows = field("table", ["type": .string("array"), "default": .array([.array([.number(0), .number(1)])])])
        guard case .table(let cols2, let arrays) = SettingControls.control(for: rows, analysis: "x") else {
            Issue.record("numeric rows are a table")
            return
        }
        #expect(cols2.count == 2 && arrays)
        #expect(SettingControls.rowKeys(in: "Points [{\"hx\", \"hy\"}] here") == ["hx", "hy"])
        #expect(SettingControls.rowKeys(in: "nothing") == nil)
        // Opaque lists stay out of the form, with their value.
        #expect(SettingControls.control(for: field("metadata", ["type": .string("array"), "default": .array([])]), analysis: "glass_map") == .hidden)
    }

    /// Objects: known properties become a group (optional with a switch),
    /// the keys of a default object are inferred, and tagged unions
    /// (`oneOf` with `const` tags) become variants.
    @Test func objectsAndVariants() {
        let detector = field("detector", [
            "type": .array([.string("object"), .string("null")]), "default": .null,
            "properties": .object([
                "pitch_mm": .object(["type": .string("array"), "default": .array([.number(0.005), .number(0.005)]), "unit": .string("mm")]),
                "fill_factor": .object(["type": .string("number"), "default": .number(1)]),
                "aperture_mm": .object(["type": .array([.string("array"), .string("null")]), "default": .null, "items": .object(["type": .string("number")])]),
            ]),
        ])
        #expect(SettingControls.control(for: detector, analysis: "mtf") == .group(nullable: true))
        #expect(detector.properties.map(\.name) == ["aperture_mm", "fill_factor", "pitch_mm"])
        #expect(SettingControls.control(for: detector.properties[2], analysis: "mtf", nested: true) == .pair(integer: false, nullable: false))
        #expect(SettingControls.control(for: detector.properties[0], analysis: "mtf", nested: true) == .pair(integer: false, nullable: true))
        let enabled = SettingControls.enabledValue(detector)
        #expect(enabled["fill_factor"] == .number(1) && enabled["pitch_mm"]?.array?.count == 2 && enabled["aperture_mm"] == .null)

        // Properties inferred from the default object.
        let huygens = field("huygens", ["type": .string("object"), "default": .object(["grid_size": .integer(128), "padding": .number(4)])])
        #expect(SettingControls.control(for: huygens, analysis: "mtf") == .group(nullable: false))
        #expect(huygens.properties.map(\.name) == ["grid_size", "padding"])
        #expect(huygens.properties[0].types == ["integer"] && huygens.properties[1].types == ["number"])
        #expect(SettingControls.control(for: huygens.properties[0], analysis: "mtf", nested: true) == .number(integer: true))

        // The pupil sampling of the wave analyses keeps its control; nested, a `pupil` object is a group.
        let pupil = field("pupil", ["type": .string("object"), "default": .object(["grid": .integer(0), "rays": .integer(65)])])
        #expect(SettingControls.control(for: pupil, analysis: "psf") == .pupil)
        #expect(SettingControls.control(for: pupil, analysis: "image_simulation", nested: true) == .group(nullable: false))

        // A tagged union.
        let sampling = field("pupil", [
            "type": .string("object"), "default": .object(["type": .string("grid"), "n": .integer(32)]),
            "oneOf": .array([
                .object(["title": .string("Grid"), "properties": .object(["type": .object(["const": .string("grid")]), "n": .object(["type": .string("integer"), "default": .integer(32)])])]),
                .object(["title": .string("Random"), "properties": .object(["type": .object(["const": .string("random")]), "count": .object(["type": .string("integer"), "default": .integer(1024)]), "seed": .object(["type": .string("integer"), "default": .integer(1)])])]),
            ]),
        ])
        #expect(SettingControls.control(for: sampling, analysis: "system_transmission") == .variants(nullable: false))
        #expect(sampling.variants.map(\.title) == ["Grid", "Random"])
        #expect(sampling.variants[0].tag?.key == "type" && sampling.variants[0].tag?.value == .string("grid"))
        #expect(sampling.variants[1].defaultObject == .object(["type": .string("random"), "count": .integer(1024), "seed": .integer(1)]))
        #expect(sampling.variant(of: .object(["type": .string("random"), "count": .integer(5)]))?.title == "Random")
        #expect(sampling.variant(of: .object(["type": .string("other")])) == nil)
        // The selection menus are not variants.
        #expect(sampling.properties.isEmpty)
        let state = field("state", ["type": .array([.string("object"), .string("null")]), "default": .null, "oneOf": sampling.variants.isEmpty ? .array([]) : .array([.object(["title": .string("Unpolarized"), "properties": .object(["type": .object(["const": .string("unpolarized")])])])])])
        #expect(SettingControls.control(for: state, analysis: "x") == .variants(nullable: true))
        #expect(SettingControls.enabledValue(state) == .object(["type": .string("unpolarized")]))
        // An object with no properties at all has no control (its value is kept).
        #expect(SettingControls.control(for: field("options", ["type": .string("object"), "default": .object([:])]), analysis: "optimize") == .hidden)
    }

    /// The spot diagram's through-focus: a dedicated control, automatic
    /// range, and a summary with the range the run used.
    @Test func throughFocus() throws {
        let tf = field("through_focus", [
            "type": .array([.string("object"), .string("null")]), "default": .null,
            "properties": .object([
                "min": .object(["type": .array([.string("number"), .string("null")]), "default": .null]),
                "max": .object(["type": .array([.string("number"), .string("null")]), "default": .null]),
                "steps": .object(["type": .string("integer"), "default": .integer(3)]),
            ]),
        ])
        #expect(SettingControls.control(for: tf, analysis: "spot_diagram") == .throughFocus)
        #expect(SettingControls.control(for: tf, analysis: "other") == .group(nullable: true))
        #expect(AutoSettings.sentinel("through_focus.min", analysis: "spot_diagram") == .null)
        let auto: JSONValue = .object(["min": .null, "max": .null, "steps": .integer(3)])
        #expect(ThroughFocusControl.summary(auto, output: nil) == "Auto range · 3 planes")
        #expect(ThroughFocusControl.summary(.object(["min": .number(-0.1), "max": .number(0.1), "steps": .integer(5)]), output: nil) == "±0.1 mm · 5 planes")
        #expect(ThroughFocusControl.summary(.object(["min": .number(-0.2), "max": .number(0), "steps": .integer(1)]), output: nil) == "-0.2 to 0 mm · 1 plane")
        let out = MapReaderTests.output(
            "spot_diagram",
            json: .object(["fields": .array([.object(["through_focus": .array([-0.08, 0, 0.08].map { .object(["defocus": .number($0)]) })])])]), buffers: [])
        #expect(AutoSettings.used("through_focus", analysis: "spot_diagram", output: out) == "±0.08")
        #expect(ThroughFocusControl.summary(auto, output: out) == "±0.08 mm · 3 planes")
        // A new spot pane samples three planes over the automatic range.
        #expect(Workspace.spotStartingSettings["through_focus"] == auto)
        #expect(WorkspaceLayout.standard.panes[0].settings == Workspace.spotStartingSettings)
        #expect(!WorkspaceLayout.standard.panes[0].showsSettings && AnalysisPane(analysis: "psf").showsSettings == false)
    }
}
