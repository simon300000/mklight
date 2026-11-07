import Foundation
import Testing

@testable import MKLightEngine
@testable import MKLightModel
@testable import MKLightUI

/// The lens designer's review round: radius bounds, anamorphic table
/// cells, number formatting, heat-map scales, run warnings, the section
/// stretch and the antialiased image-simulation patterns.
@Suite @MainActor struct PolishTests {
    private func curvature(_ min: Double?, _ max: Double?) -> (lower: Double?, upper: Double?)? {
        try? RadiusBounds.curvature(minRadius: min, maxRadius: max).get()
    }

    private func close(_ a: Double?, _ b: Double?) -> Bool {
        switch (a, b) {
        case (nil, nil): true
        case (let x?, let y?): x == y || abs(x - y) <= 1e-12 * max(abs(x), abs(y))
        default: false
        }
    }

    // MARK: Radius ↔ curvature bounds

    @Test func positiveAndNegativeRadiusWindows() throws {
        // 40 … 60 mm: c from 1/60 to 1/40.
        let p = try #require(curvature(40, 60))
        #expect(close(p.lower, 1.0 / 60) && close(p.upper, 1.0 / 40))
        // −60 … −40 mm: c from −1/40 to −1/60 (the order flips with the sign).
        let n = try #require(curvature(-60, -40))
        #expect(close(n.lower, -1.0 / 40) && close(n.upper, -1.0 / 60))
        // Both round-trip.
        let rp = RadiusBounds.radius(lower: p.lower, upper: p.upper)
        #expect(close(rp.min, 40) && close(rp.max, 60))
        let rn = RadiusBounds.radius(lower: n.lower, upper: n.upper)
        #expect(close(rn.min, -60) && close(rn.max, -40))
    }

    @Test func aWindowThroughInfinityIsACurvatureIntervalThroughZero() throws {
        // min 100, max −100: |R| ≥ 100, through flat.
        let c = try #require(curvature(100, -100))
        #expect(close(c.lower, -0.01) && close(c.upper, 0.01))
        #expect(RadiusBounds.passesThroughFlat(minRadius: 100, maxRadius: -100))
        let r = RadiusBounds.radius(lower: -0.01, upper: 0.01)
        #expect(close(r.min, 100) && close(r.max, -100))
        // A typed ∞ is flat: 50 … ∞ is c from 0 to 0.02, as is an empty max.
        let open = try #require(curvature(50, .infinity))
        #expect(open.lower == 0 && close(open.upper, 0.02))
        let empty = try #require(curvature(50, nil))
        #expect(empty.lower == 0 && close(empty.upper, 0.02))
        #expect(RadiusBounds.radius(lower: 0, upper: 0.02).max == nil, "c ≥ 0 shows an open max")
        // Negative radii up to −50 (an empty min): c from −0.02 to 0.
        let neg = try #require(curvature(nil, -50))
        #expect(close(neg.lower, -0.02) && neg.upper == 0)
        #expect(neg.upper.map { $0.sign == .plus } == true, "no negative zero is stored")
        // A pinned flat surface shows ∞ … ∞ and stores c = 0 … 0.
        #expect(RadiusBounds.radius(lower: 0, upper: 0) == (.infinity, .infinity))
        let flat = try #require(curvature(.infinity, .infinity))
        #expect(flat.lower == 0 && flat.upper == 0)
    }

    @Test func zeroIsTheUnboundedCurvatureLimit() throws {
        // 0 … 50: R → 0⁺ up to 50, so c ≥ 0.02 and unbounded above.
        let c = try #require(curvature(0, 50))
        #expect(close(c.lower, 0.02) && c.upper == nil)
        let r = RadiusBounds.radius(lower: 0.02, upper: nil)
        #expect(r.min == 0 && close(r.max, 50))
        // Both empty: unbounded.
        let none = try #require(curvature(nil, nil))
        #expect(none.lower == nil && none.upper == nil)
        #expect(RadiusBounds.radius(lower: nil, upper: nil) == (nil, nil))
    }

    @Test func windowsThroughZeroRadiusAreRefused() {
        #expect(RadiusBounds.curvature(minRadius: -50, maxRadius: 50).isFailure)
        // 60 … 40 wraps through ∞ and then through 0.
        #expect(RadiusBounds.curvature(minRadius: 60, maxRadius: 40).isFailure)
        // R ≤ 50 includes every negative radius and R = 0.
        #expect(RadiusBounds.curvature(minRadius: nil, maxRadius: 50).isFailure)
    }

    @Test func boundsShowInTheOtherUnit() {
        #expect(BoundsFormat.counterpart(lower: 1.0 / 60, upper: 1.0 / 40, shownAs: .radius) == "c 0.01667 … 0.025 1/mm")
        #expect(BoundsFormat.counterpart(lower: -0.01, upper: 0.01, shownAs: .curvature) == "R 100 … ∞ … -100 mm")
        #expect(BoundsFormat.counterpart(lower: nil, upper: nil, shownAs: .radius) == nil)
    }

    @Test func everyBoundedQuantityHasItsUnit() {
        let s = SampleSystems.cookeTriplet()
        let id = s.surfaces[1].id
        #expect(ParamPath.curvature(surface: id).unit(in: s) == "1/mm")
        #expect(ParamPath.thickness(surface: id).unit(in: s) == "mm")
        #expect(ParamPath.conic(surface: id).unit(in: s) == "")
        #expect(ParamPath.asphere(surface: id, index: 0).unit(in: s) == "mm⁻³")
        #expect(ParamPath.asphere(surface: id, index: 3).unit(in: s) == "mm⁻⁹")
        #expect(ParamPath.shapeCoefficient(surface: id, set: .odd, index: 0).unit(in: s) == "")
        #expect(ParamPath.shapeCoefficient(surface: id, set: .odd, index: 2).unit(in: s) == "mm⁻²")
        #expect(ParamPath.shapeCoefficient(surface: id, set: .yAsphere, index: 4).unit(in: s) == "mm⁻¹¹")
        #expect(ParamPath.shapeCoefficient(surface: id, set: .xy, index: 4).unit(in: s) == "mm")
        #expect(ParamPath.shapeCoefficient(surface: id, set: .phase, index: 0).unit(in: s) == "rad")
        #expect(ParamPath.modelNd(surface: id).unit(in: s) == "" && ParamPath.modelVd(surface: id).unit(in: s) == "")
        #expect(ParamPath.tiltX(surface: id).unit(in: s) == "°")
        #expect(ParamPath.shapeParam(surface: id, parameter: .curvatureX).isCurvature)
        #expect(!ParamPath.thickness(surface: id).isCurvature)
    }

    // MARK: Anamorphic table cells

    @Test func cylinderRowsShowBothSectionsAndThePoweredOne() throws {
        var s = SampleSystems.singlet()
        s.surfaces[1].shape = .cylinder(curvature: 0.02, axis: .x)
        let row = LensRow.rows(for: s)[1]
        #expect(row.typeTitle == "Cylinder")
        #expect(row.hasSections(.radius))
        #expect(row.text(.radius) == "∞" && row.text(.radiusX) == "50")
        #expect(row.hints[.radius] == "y" && row.hints[.radiusX] == "x")
        #expect(row.poweredAxis == .x && row.isFlatSection(.radius) && !row.isFlatSection(.radiusX))
        // Only the powered section's conic shows.
        #expect(row.cells[.conic] == nil && row.text(.conicX) == "0" && row.hints[.conicX] == "x")
        #expect(!row.hasSections(.conic) && row.shown(.conic) == [.conicX] && row.shown(.radius) == [.radius, .radiusX])
        #expect(LensRow.rows(for: SampleSystems.singlet())[1].shown(.thickness) == [.thickness])
        // Both sections edit in place.
        let edit = try #require(row.edit(.radiusX, text: "-25")).get()
        edit.apply(to: &s)
        #expect(s.surfaces[1].shape.shapeParameter(.curvatureX) == -0.04)
        #expect(LensColumn.radiusX.param(for: s.surfaces[1]) == .shapeParam(surface: s.surfaces[1].id, parameter: .curvatureX))
        #expect(LensColumn.radiusX.base == .radius && LensColumn.conicX.base == .conic)
    }

    @Test func biconicAndToroidRowsShowBothSections() throws {
        var s = SampleSystems.singlet()
        s.surfaces[1].shape = .biconic(curvatureX: 0.01, curvatureY: 0.02, conicX: -1, conicY: 0.5, xAsphere: [], yAsphere: [])
        let b = LensRow.rows(for: s)[1]
        #expect(b.text(.radius) == "50" && b.text(.radiusX) == "100")
        #expect(b.hasSections(.conic) && b.text(.conic) == "0.5" && b.text(.conicX) == "-1")
        #expect(b.poweredAxis == nil && !b.isFlatSection(.radius))
        let k = try #require(b.edit(.conicX, text: "-2")).get()
        k.apply(to: &s)
        #expect(s.surfaces[1].shape.shapeParameter(.conicX) == -2)
        // A toroid: two radii, one (y) conic; its x section has no conic.
        s.surfaces[1].shape = .toroid(curvatureX: 0.04, curvatureY: 0.02, conicY: 0, yAsphere: [])
        let t = LensRow.rows(for: s)[1]
        #expect(t.text(.radius) == "50" && t.text(.radiusX) == "25" && !t.hasSections(.conic) && t.hints[.conic] == "y")
        #expect(LensColumn.conicX.param(for: s.surfaces[1]) == nil)
        #expect(t.edit(.conicX, text: "1") == nil)
        // Rotational rows have no x section.
        let plain = LensRow.rows(for: SampleSystems.singlet())[1]
        #expect(!plain.hasSections(.radius) && plain.cells[.radiusX] == nil)
        #expect(LensColumn.radiusX.param(for: SampleSystems.singlet().surfaces[1]) == nil)
    }

    // MARK: Number formatting

    @Test func numbersUseADecimalPointEverywhere() {
        #expect(MaterialRef.constant(n: 1.5).displayName == "n=1.5")
        #expect(MaterialRef.constant(n: 1.7552).displayName == "n=1.7552")
        #expect(MaterialRef.model(nd: 1.5168, vd: 64.17, dpgf: 0).displayName == "1.51680/64.17")
        #expect(NumberDisplay.significant(12345.678) == "12345.678")
        #expect(NumberDisplay.significant(.infinity) == "∞")
        // The views' locale: a point, no grouping, whatever the region.
        #expect(1234.5.formatted(.number.locale(NumberDisplay.locale)) == "1234.5")
        #expect(Fmt.short(0.5) == "0.5" && Fmt.value(1.25, unit: "mm") == "1.25 mm")
        #expect(RowFormat.compact(-50) == "-50" && RowFormat.compact(.infinity) == "∞")
        #expect(BoundsFormat.compact(1e-5) == "1e-05")
        // Signed columns: no "−0.0000", zero keeps the sign's place.
        #expect(Fmt.length(-1e-7) == "0.00 µm" && Fmt.length(-0.02) == "-20.00 µm")
        #expect(Fmt.signed(-0.00001) == "\u{2007}0.0000" && Fmt.signed(0.0123) == "+0.0123" && Fmt.signed(-2) == "-2.0000")
        // Typed numbers accept a decimal comma too.
        #expect(LensValueParser.number("1,5") == 1.5)
        #expect(LensValueParser.number("−2,25") == -2.25)
        #expect(LensValueParser.number("1,5,3") == nil)
        if case .success(let m) = LensValueParser.material("n=1,6") { #expect(m.material == .constant(n: 1.6)) } else { Issue.record("n=1,6") }
    }

    // MARK: Heat-map scales

    @Test func heatMapsScaleToTheirDataByDefault() {
        // A relative illumination map from 0.85 to 1: its own range, not 0…1.
        #expect(HeatMapScale.range(0.85...1, signed: false) == 0.85...1)
        #expect(HeatMapScale.range(0.85...1, signed: false, mode: .unit, isFraction: true) == 0...1)
        // 0…1 applies to fractions only.
        #expect(HeatMapScale.range(10...20, signed: false, mode: .unit, isFraction: false) == 10...20)
        // Signed data: diverging, symmetric about 0.
        #expect(HeatMapScale.scale(signed: true) == .diverging && HeatMapScale.scale(signed: false) == .perceptual)
        #expect(HeatMapScale.range(-0.2...0.5, signed: true) == -0.5...0.5)
        // A flat map gets a small band around its value.
        let flat = HeatMapScale.range(1...1, signed: false)
        #expect(flat.lowerBound < 1 && flat.upperBound > 1 && flat.upperBound - flat.lowerBound < 0.01)
        #expect(abs(HeatMapImage.normalise(0.925, scale: .perceptual, range: 0.85...1) - 0.5) < 1e-12)
        // The perceptual ramp brightens monotonically (readable as order).
        let luma = stride(from: 0.0, through: 1.0, by: 0.1).map { t -> Double in
            let c = Palette.perceptual(t)
            return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
        }
        #expect(zip(luma, luma.dropFirst()).allSatisfy { $0 < $1 })
        // Iso-illumination levels inside the data's range.
        #expect(Contours.illuminationLevels(in: 0.852...1) == [0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98])
        #expect(Contours.illuminationLevels(in: 0.3...1).first == 0.4)
    }

    // MARK: Run warnings

    @Test func warningsGroupByMessageWithTheirFields() {
        let w = [
            "field 0: wavefront changes by up to 0.58 waves between pupil samples (limit 0.5); increase the pupil sampling",
            "field 2: wavefront changes by up to 0.72 waves between pupil samples (limit 0.5); increase the pupil sampling",
            "field 2: only 89.5 % of the PSF energy lies inside the image grid; the PSF is truncated (enlarge the grid or its spacing)",
            "field 1, wavelength 0: parts of the aperture are 0.044 (pupil units) from the nearest ray",
            "field (-1, -1): 54 of 576 pupil nodes have no ray through the image point",
            "field (1, 1): 54 of 576 pupil nodes have no ray through the image point",
            "psf_truncated: field [0.0, -1.0], sampled energy 0.26641",
        ]
        let g = RunWarnings.group(w)
        #expect(g.count == 5)
        #expect(g[0].scopes == ["Field 1", "Field 3"] && g[0].remedy == .pupilSampling)
        #expect(g[0].message.hasPrefix("Wavefront changes by up to 0.58 waves"))
        #expect(g[1].remedy == .imageGrid)
        #expect(g[2].scopes == ["Field 2 · λ1"])
        #expect(g[3].scopes == ["field (−1, −1)", "field (1, 1)"])
        #expect(g[4].scopes.isEmpty && g[4].message.hasPrefix("Psf_truncated"))
        #expect(RunWarnings.summary(w).hasPrefix("Field 1, Field 3: Wavefront") && RunWarnings.summary(w).hasSuffix("(and 4 more)"))
    }

    // MARK: Section stretch

    @Test func slimSectionsGetAVerticalStretch() {
        #expect(ViewportCamera.automaticStretch(height: 300, viewHeight: 600) == 1)
        #expect(ViewportCamera.automaticStretch(height: 40, viewHeight: 600) == 5)
        #expect(ViewportCamera.automaticStretch(height: 1, viewHeight: 600) == 50)
        var c = ViewportCamera()
        c.stretch = 5
        #expect(c.verticalScale(.section2D) == 5 && c.verticalScale(.sectionXZ) == 5 && c.verticalScale(.orbit3D) == 1)
        // Fitting a stretched section frames the stretched height.
        let pts = [SIMD3(0.0, -1, 0), SIMD3(0.0, 1, 100)]
        let plain = ViewportMath.fit(ViewportCamera(), mode: .section2D, points: pts, size: CGSize(width: 400, height: 400))
        let stretched = ViewportMath.fit(c, mode: .section2D, points: pts, size: CGSize(width: 400, height: 400))
        #expect(abs(plain.pointsPerMM - stretched.pointsPerMM) < 1e-9, "width-limited: the zoom is the same")
        let m = ViewportMath.viewProjection(stretched, mode: .section2D, origin: .zero, size: CGSize(width: 400, height: 400), depthRadius: 100)
        let m1 = ViewportMath.viewProjection(plain, mode: .section2D, origin: .zero, size: CGSize(width: 400, height: 400), depthRadius: 100)
        #expect(abs(m.columns.1.y / m1.columns.1.y - 5) < 1e-4)
    }

    // MARK: Image simulation

    @Test func patternsAreAreaSampled() throws {
        let sharp = try #require(LinearImage.pattern("slanted_edge", width: 64, height: 64))
        let soft = try #require(LinearImage.pattern("slanted_edge", width: 64, height: 64, supersample: 4))
        #expect(Set(sharp.data) == [0, 1], "pixel-centre sampling: staircase")
        #expect(soft.data.contains { $0 > 0 && $0 < 1 }, "edge pixels are partial")
        // The mean stays the pattern's area fraction.
        let mean = soft.data.reduce(0, +) / Double(soft.data.count)
        #expect(abs(mean - 0.5) < 0.01)
        // The run sends the antialiased pattern as a buffer at the run's size.
        let s = ImageSourceImport.engineSettings(
            "image_simulation", ["size": .array([.integer(32), .integer(16)]), "source": .object(["kind": .string("pattern"), "pattern": .string("grid")])],
            defaults: nil)
        let buffer = try #require(LinearImage(source: s["source"]))
        #expect(buffer.width == 32 && buffer.height == 16 && buffer == LinearImage.pattern("grid", width: 32, height: 16, supersample: 4))
        // Other analyses and imported images are left alone.
        #expect(ImageSourceImport.engineSettings("psf", ["size": .integer(3)], defaults: nil) == ["size": .integer(3)])
    }

    /// Image simulation panes saved with the engine's defaults (before the
    /// starting settings existed) take the starting settings and wait for
    /// Update; panes the user changed keep their settings.
    @Test func oldImageSimulationPanesMigrate() async throws {
        let defaults: JSONValue = .object(["size": .array([.integer(64), .integer(64)])])
        #expect(ImageSimulationSettings.isEngineDefault([:], defaults: defaults))
        #expect(ImageSimulationSettings.isEngineDefault(["size": .array([.integer(64), .integer(64)])], defaults: defaults))
        #expect(!ImageSimulationSettings.isEngineDefault(["size": .array([.integer(512), .integer(512)])], defaults: defaults))
        let old = AnalysisPane(id: "old", analysis: "image_simulation", settings: [:], autoUpdate: true)
        let mine = AnalysisPane(id: "mine", analysis: "image_simulation", settings: ["size": .array([.integer(128), .integer(128)])], autoUpdate: true)
        let w = Workspace(engine: MockEngine(simulatedLatency: .zero), panes: [old, mine])
        await w.loadEngineMetadata()
        #expect(old.settings == w.startingSettings("image_simulation") && !old.autoUpdate)
        #expect(mine.settings["size"] == .array([.integer(128), .integer(128)]) && mine.autoUpdate)
    }

    // MARK: Leftovers of the review round

    /// The section stretch is saved in the project's `views`; a stretch
    /// chosen there (even ×1) is kept instead of the automatic one.
    @Test func stretchIsSavedInTheViews() throws {
        let w = Workspace(engine: MockEngine(simulatedLatency: .zero))
        let window = WindowState()
        #expect(WorkspaceLayout(workspace: w, window: window, arrangement: .standard(panes: [])).stretch == nil, "never chosen: the first fit decides")
        window.camera.stretch = 10
        let layout = WorkspaceLayout(workspace: w, window: window, arrangement: .standard(panes: []))
        let back = WorkspaceLayout(views: layout.views)
        #expect(back.stretch == 10 && back == layout)
        let restored = WindowState()
        back.apply(to: restored)
        #expect(restored.camera.stretch == 10 && restored.camera.stretchChosen)
        // Turned off by hand: saved as ×1 and kept.
        restored.camera.stretch = 1
        let off = WorkspaceLayout(views: WorkspaceLayout(workspace: w, window: restored, arrangement: .standard(panes: [])).views)
        #expect(off.stretch == 1)
        let again = WindowState()
        off.apply(to: again)
        #expect(again.camera.stretch == 1 && again.camera.stretchChosen)
    }

    /// Seidel terms and other `_` labels are set with real subscripts.
    @Test func subscriptedLabels() {
        typealias P = Subscripted.Piece
        #expect(Subscripted.pieces("S_III astigmatism") == [P(text: "S", isSubscript: false), P(text: "III", isSubscript: true), P(text: " astigmatism", isSubscript: false)])
        #expect(Subscripted.pieces("Σ S_I") == [P(text: "Σ S", isSubscript: false), P(text: "I", isSubscript: true)])
        #expect(Subscripted.pieces("C_L axial").filter(\.isSubscript).map(\.text) == ["L"])
        #expect(Subscripted.pieces("EFL") == [P(text: "EFL", isSubscript: false)])
        #expect(Subscripted.pieces("_x") == [P(text: "_x", isSubscript: false)], "a leading underscore is text")
        #expect(Subscripted.spoken("Σ S_II") == "Σ S II")
        #expect(SeidelView.terms.map(\.1).allSatisfy { Subscripted.pieces($0).contains(where: \.isSubscript) })
    }
}

extension Result {
    fileprivate var isFailure: Bool { if case .failure = self { true } else { false } }
}
