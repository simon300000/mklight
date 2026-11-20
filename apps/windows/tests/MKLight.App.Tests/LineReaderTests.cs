using MKLight.App.Views.Analysis.Results.Readers;

namespace MKLight.App.Tests;

public class LineReaderTests
{
    static readonly double[] Um = [0.4861, 0.5876, 0.6563];
    static readonly string[] Fields = ["0°", "10°"];

    [Fact]
    public void RayFanPlotsTangentialAndSagittalPerFieldInMicrometres()
    {
        var d = TestData.Result("ray_fan", """
        { "quantity": "transverse_aberration", "wavelengths_um": [0.4861, 0.5876, 0.6563],
          "fields": [
            { "field": 0, "field_value": [0, 0],
              "tangential": [ {"wavelength": 0, "pupil": [-1, 0, 1], "values": [-0.01, 0, 0.02]},
                              {"wavelength": 2, "pupil": [-1, 0, 1], "values": [0.001, 0, null]} ],
              "sagittal": [ {"wavelength": 0, "pupil": [-1, 0, 1], "values": [-0.005, 0, 0.005]} ] },
            { "field": 1, "field_value": [0, 10], "tangential": [], "sagittal": [] } ] }
        """, Um, Fields);
        var m = LineReaders.Read(d)!;
        Assert.Equal(2, m.Groups.Count);
        Assert.Equal("Field 1 · 0°", m.Groups[0].Label);
        Assert.Equal(["Tangential", "Sagittal"], m.Groups[0].Panels.Select(p => p.Title));
        var t = m.Groups[0].Panels[0];
        Assert.Equal([-10.0, 0, 20], t.Series[0].Y.Select(v => Math.Round(v, 9)));
        Assert.True(double.IsNaN(t.Series[1].Y[2]));
        Assert.Equal(ColorRole.Wavelength, t.Series[1].Color.Role);
        Assert.Equal(0.6563, t.Series[1].Color.Um);
        Assert.Equal("εy (µm)", t.YTitle);
        // One symmetric range over every fan.
        Assert.Equal((-21.6, 21.6), (Math.Round(t.YDomain!.Value.Min, 9), Math.Round(t.YDomain!.Value.Max, 9)));
        Assert.Equal(m.Groups[1].Panels[1].YDomain, t.YDomain);
        Assert.Equal(["0.4861 µm", "0.6563 µm"], m.Legend.Select(e => e.Label));
    }

    [Fact]
    public void OpdFanIsInWaves()
    {
        var d = TestData.Result("opd_fan", """
        { "quantity": "opd", "fields": [ { "field": 0, "field_value": [0, 0],
            "tangential": [ {"wavelength": 1, "pupil": [-1, 1], "values": [0.25, -0.25]} ], "sagittal": [] } ] }
        """, Um, Fields);
        var t = LineReaders.Read(d)!.Groups[0].Panels[0];
        Assert.Equal([0.25, -0.25], t.Series[0].Y);
        Assert.Equal("W (waves)", t.YTitle);
        Assert.Equal(0.5876, t.Series[0].Color.Um);
    }

    [Fact]
    public void WaveMtfReadsItsBuffersByNameWithTheDiffractionLimitAndNyquist()
    {
        var d = TestData.Result("mtf", """
        { "frequencies": "frequencies", "nyquist": [100, 100], "units": {"afocal": false},
          "fields": [
            { "field": 0, "field_value": [0, 0], "tangential": "tangential_0", "sagittal": "sagittal_0", "diffraction_limit": "diffraction_limit_0" },
            { "field": 1, "field_value": [0, 10], "tangential": "tangential_1", "sagittal": "sagittal_1", "diffraction_limit": "diffraction_limit_1" } ] }
        """, Um, Fields,
            TestData.Vector("frequencies", 0, 50, 150),
            TestData.Vector("tangential_0", 1, 0.6, 0.2), TestData.Vector("sagittal_0", 1, 0.7, 0.3),
            TestData.Vector("tangential_1", 1, 0.5, 0.1), TestData.Vector("sagittal_1", 1, 0.4, double.NaN),
            TestData.Vector("diffraction_limit_0", 1, 0.8, 0.4), TestData.Vector("diffraction_limit_1", 1, 0.8, 0.4));
        var m = LineReaders.Read(d)!;
        var p = m.Groups.Single().Panels.Single();
        Assert.Equal(5, p.Series.Count); // limit, then T and S per field
        Assert.Equal(ColorRole.Neutral, p.Series[0].Color.Role);
        Assert.False(p.Series[1].Dashed);
        Assert.True(p.Series[2].Dashed);
        Assert.Equal(1, p.Series[3].Color.Index);
        Assert.Equal((0.0, 150.0), p.XDomain);
        Assert.Equal("Spatial frequency (cycles/mm)", p.XTitle);
        Assert.Equal(100, p.Rules.Single().X);
        Assert.Equal("Nyquist", p.Rules.Single().Label);
        Assert.Contains(m.Legend, e => e.Label == "Field 2 · 10°");
        Assert.Contains(m.Legend, e => e.Label == "T solid, S dashed");
    }

    [Fact]
    public void GeometricMtfGroupsCurvesByField()
    {
        var d = TestData.Result("geometric_mtf", """
        { "frequencies": [0, 10, 20], "diffraction_limit": null, "field_values": [[0, 0], [0, 10]],
          "curves": [ {"field": 0, "orientation": "tangential", "modulation": [1, 0.5, 0.2]},
                      {"field": 0, "orientation": "sagittal", "modulation": [1, 0.6, 0.3]},
                      {"field": 1, "orientation": "tangential", "modulation": [1, 0.4, 0.1]} ] }
        """, Um, Fields);
        var p = LineReaders.Read(d)!.Groups[0].Panels[0];
        Assert.Equal(3, p.Series.Count);
        Assert.True(p.Series[1].Dashed);
        Assert.Empty(p.Rules);
    }

    [Fact]
    public void MtfVsFieldAndThroughFocusColourByFrequency()
    {
        var vf = TestData.Result("mtf_vs_field", """
        { "normalized_field": "normalized_field", "frequencies": "frequencies",
          "curves": [ {"frequency": 10, "tangential": "tangential_0", "sagittal": "sagittal_0"},
                      {"frequency": 20, "tangential": "tangential_1", "sagittal": "sagittal_1"} ] }
        """, TestData.Vector("normalized_field", 0, 0.5, 1), TestData.Vector("frequencies", 10, 20),
            TestData.Vector("tangential_0", 0.9, 0.8, 0.7), TestData.Vector("sagittal_0", 0.9, 0.85, 0.8),
            TestData.Vector("tangential_1", 0.7, 0.6, 0.5), TestData.Vector("sagittal_1", 0.7, 0.65, 0.6));
        var m = LineReaders.Read(vf)!;
        Assert.Equal(4, m.Groups[0].Panels[0].Series.Count);
        Assert.Equal("10 cycles/mm", m.Legend[0].Label);
        Assert.Equal(1, m.Groups[0].Panels[0].Series[2].Color.Index);

        var tf = TestData.Result("through_focus_mtf", """
        { "focus_mm": "focus_mm", "frequencies": "frequencies",
          "fields": [ {"field": 0, "field_value": [0, 0], "curves": [ {"frequency": 10, "tangential": "tangential_0_0", "sagittal": "sagittal_0_0"} ]},
                      {"field": 1, "field_value": [0, 10], "curves": [ {"frequency": 10, "tangential": "tangential_1_0", "sagittal": "sagittal_1_0"} ]} ] }
        """, Um, Fields, TestData.Vector("focus_mm", -0.05, 0, 0.05), TestData.Vector("frequencies", 10),
            TestData.Vector("tangential_0_0", 0.5, 0.9, 0.5), TestData.Vector("sagittal_0_0", 0.5, 0.9, 0.5),
            TestData.Vector("tangential_1_0", 0.4, 0.8, 0.4), TestData.Vector("sagittal_1_0", 0.4, 0.8, 0.4));
        var t = LineReaders.Read(tf)!;
        Assert.Equal(2, t.Groups.Count);
        Assert.Equal("Field 2 · 10°", t.Groups[1].Label);
        Assert.Equal((-0.05, 0.05), t.Groups[0].Panels[0].XDomain);
        Assert.Equal("Image", t.Groups[0].Panels[0].Rules.Single().Label);
    }

    [Fact]
    public void FieldCurvatureHasCurvatureAndDistortionPanels()
    {
        var d = TestData.Result("field_curvature_distortion", """
        { "field_unit": "°", "field_values": [0, 5, 10],
          "curves": [ {"wavelength": 1, "wavelength_um": 0.5876, "tangential": [0, -0.01, -0.03], "sagittal": [0, -0.005, -0.01], "distortion_percent": [0, -0.2, -0.9]} ] }
        """, Um, Fields);
        var m = LineReaders.Read(d)!;
        var panels = m.Groups.Single().Panels;
        Assert.Equal(["Field Curvature", "Distortion"], panels.Select(p => p.Title));
        Assert.Equal(2, panels[0].Series.Count);
        Assert.True(panels[0].Series[1].Dashed);
        Assert.Equal([0.0, 5, 10], panels[0].Series[0].Y);
        Assert.Equal("Field (°)", panels[1].YTitle);
        Assert.Equal("Max distortion 0.900 %", KeyNumbers.Text(d));
    }

    [Fact]
    public void EncircledEnergyReadsInlineAndBufferRadii()
    {
        var geo = TestData.Result("encircled_energy", """
        { "radius": [0, 0.01, 0.02], "fields": [ {"field": 0, "field_value": [0, 0], "encircled": [0, 0.6, 1], "r80": 0.015} ] }
        """, Um, Fields);
        var m = LineReaders.Read(geo)!;
        Assert.Equal([0.0, 10, 20], m.Groups[0].Panels[0].Series[0].X.Select(v => Math.Round(v, 9)));
        Assert.Equal("Field 1 · 0° · r80 15.00 µm", m.Legend[0].Label);

        var wave = TestData.Result("diffraction_encircled_energy", """
        { "radius_mm": "radius_mm", "units": {"afocal": false},
          "fields": [ {"field": 0, "field_value": [0, 0], "encircled": "encircled_0", "ensquared": "ensquared_0", "diffraction_limit": "diffraction_limit_0", "r80_mm": 0.004} ] }
        """, Um, Fields, TestData.Vector("radius_mm", 0, 0.005), TestData.Vector("encircled_0", 0, 0.9), TestData.Vector("diffraction_limit_0", 0, 0.95));
        var w = LineReaders.Read(wave)!;
        var p = w.Groups[0].Panels[0];
        Assert.Equal(2, p.Series.Count);
        Assert.True(p.Series[0].Dashed); // the diffraction limit
        Assert.Contains(w.Legend, e => e.Label.EndsWith("r80 4.00 µm"));
    }

    [Fact]
    public void ScalarCurvesAndRules()
    {
        var lc = TestData.Result("lateral_colour", """
        { "field_unit": "mm", "field_values": [0, 1], "airy_radius": 0.003,
          "curves": [ {"wavelength": 0, "wavelength_um": 0.4861, "offset": [0, 0.002]} ] }
        """, Um, Fields);
        var p = LineReaders.Read(lc)!.Groups[0].Panels[0];
        Assert.Equal([3.0, -3.0], p.Rules.Select(r => Math.Round(r.X, 9)));

        var cfs = TestData.Result("chromatic_focal_shift", """{ "wavelengths_um": [0.45, 0.55, 0.65], "focal_shift": [0.01, 0, 0.02], "range": 0.02 }""");
        var c = LineReaders.Read(cfs)!.Groups[0].Panels[0];
        Assert.Equal([10.0, 0, 20], c.Series[0].X.Select(v => Math.Round(v, 9)));
        Assert.Equal("Range 20.00 µm", KeyNumbers.Text(cfs));

        var ri = TestData.Result("relative_illumination", """{ "field_unit": "°", "field_values": [0, 10], "relative_illumination": [1, 0.8], "vignetting_fraction": [0, 0.1] }""");
        Assert.Equal(2, LineReaders.Read(ri)!.Groups[0].Panels[0].Series.Count);

        var bf = TestData.Result("best_focus", """{ "defocus": 0.01, "through_focus_defocus": [-0.1, 0, 0.1], "through_focus_merit": [0.02, 0.01, 0.03] }""");
        Assert.Equal("Best", LineReaders.Read(bf)!.Groups[0].Panels[0].Rules.Single().Label);
        Assert.Null(LineReaders.Read(TestData.Result("best_focus", """{ "defocus": 0.01, "through_focus_defocus": [], "through_focus_merit": [] }""")));
    }

    [Fact]
    public void UnexpectedShapesReadAsNothing()
    {
        Assert.Null(LineReaders.Read(TestData.Result("ray_fan", """{"fields": "nope"}""")));
        Assert.Null(LineReaders.Read(TestData.Result("mtf", """{"frequencies": "missing_buffer", "fields": []}""")));
        Assert.Null(LineReaders.Read(TestData.Result("mtf", "null")));
        Assert.Null(LineReaders.Read(TestData.Result("unknown_analysis", "{}")));
        Assert.Null(LineReaders.Read(TestData.Result("chromatic_focal_shift", """{ "wavelengths_um": [0.5], "focal_shift": [0.1, 0.2] }""")));
    }
}
