using MKLight.App.Views.Analysis;
using MKLight.App.Views.Analysis.Results.Readers;

namespace MKLight.App.Tests.Results;

public class ResultReaderTests
{
    [Fact]
    public void FirstOrderReadsTheReportAndTheXyFirstOrder()
    {
        var d = FirstOrderData.Read(TestData.Obj("""
            { "entries": [ {"key": "efl", "label": "Effective focal length", "unit": "mm", "value": 49.997},
                           {"key": "fno", "label": "F/#", "unit": "", "value": "inf"}, {"key": "bad"} ],
              "wavelength_um": 0.5875618,
              "paraxial_xy": { "symmetry": "none", "primary_wavelength": 1, "warnings": ["w"],
                "wavelengths": [ {"wavelength": 0, "wavelength_um": 0.4861, "efl_x": 40, "efl_y": 50, "bfl_x": 1, "bfl_y": 2, "doe_power": []},
                                 {"wavelength": 1, "wavelength_um": 0.5876, "efl_x": 41, "efl_y": 51, "bfl_x": 1, "bfl_y": 2, "squeeze": 1.25,
                                  "doe_power": [{"surface": 3, "power": 0.001}]} ] } }
            """))!;
        Assert.Equal(["efl", "fno"], d.Entries.Select(e => e.Key));
        Assert.Equal("49.997 mm", d.Entries[0].Formatted);
        Assert.True(double.IsPositiveInfinity(d.Entries[1].Value));
        Assert.True(d.IsAnamorphic);
        Assert.Equal(1, d.Primary!.Wavelength);
        Assert.Equal(1.25, d.Primary.Squeeze);
        Assert.Equal(1.25, d.Others.Single().Squeeze); // EFL y / EFL x when the engine gives no squeeze
        Assert.Equal(3, d.Primary.Doe.Single().Surface);
        Assert.Equal("1.2500×", FirstOrderData.SqueezeText(1.25));
        Assert.Null(FirstOrderData.Read(TestData.Obj("""{"entries": []}""")));
    }

    [Fact]
    public void SeidelReadsSurfacesAndTheSumOrWhyItDoesNotApply()
    {
        var d = SeidelData.Read(TestData.Obj("""
            { "status": "applicable", "petzval_radius": -256.1,
              "surfaces": [ {"surface": 1, "s1": 0.26, "s2": -0.017, "s3": 0.001, "s4": 0.13, "s5": -0.009, "cl": 0.035, "ct": -0.002},
                            {"surface": 2, "s1": 0.003, "s2": null, "s3": 0.09, "s4": -0.04, "s5": -0.25, "cl": 0.004, "ct": -0.022} ],
              "sums": {"s1": 0.263, "s2": -0.017, "s3": 0.091, "s4": 0.09, "s5": -0.259, "cl": 0.039, "ct": -0.024},
              "wavefront_waves": {"w040": 18.67, "w220p": 16.1} }
            """))!;
        Assert.Equal(["1", "2", "Σ"], d.Surfaces);
        Assert.Equal([0.26, 0.003, 0.263], d.Values["s1"]);
        Assert.True(double.IsNaN(d.Values["s2"][1]));
        Assert.Equal([("W_040", 18.67), ("W_220P", 16.1)], d.Wavefront);
        Assert.Equal(-256.1, d.PetzvalRadius);
        Assert.Null(d.NotApplicable);
        var na = SeidelData.Read(TestData.Obj("""{"status": "not_applicable", "reason": "Seidel requires rotational system symmetry."}"""))!;
        Assert.Equal("Seidel requires rotational system symmetry.", na.NotApplicable);
    }

    [Fact]
    public void PsfReadsItsBufferInMicrometresAboutTheGridCentre()
    {
        var d = TestData.Result("psf", """
            { "units": {"afocal": false}, "psfs": [ { "field": 1, "field_value": [0, 10], "strehl": 0.5,
                "psf": {"data": "data_0", "size": 4, "spacing_mm": 0.001, "center_mm": [0, 0], "normalization": "strehl", "wavelength_um": null} } ] }
            """, [0.5876], ["0°", "10°"], TestData.Buffer("data_0", [4, 4], new double[] { 0, 0, 0, 0, 0, 0.5, 0.2, 0, 0, 0.1, 0, 0, 0, 0, 0, 0 }));
        var p = PsfData.Read(d)!;
        var it = Assert.Single(p.Items);
        Assert.Equal("Field 2 · 10° · polychromatic", it.Title);
        Assert.Equal(0.5, it.Peak);
        // Coordinate i is centre + (i − 2) × spacing, in µm.
        Assert.Equal((-2.0, 1.0), (it.Grid.X0, it.Grid.X1));
        Assert.Equal((-2.0, 1.0), (it.Grid.Y0, it.Grid.Y1));
        Assert.Equal("Strehl 0.500", KeyNumbers.Text(TestData.Result("psf", d.Json!.ToJsonString())));
    }

    [Fact]
    public void WavefrontReadsMapsZernikeAndRim()
    {
        var d = TestData.Result("wavefront_map", """
            { "maps": [ { "field": 0, "field_value": [0, 0], "wavelength_um": 0.5876, "opd_waves": "opd_0", "coordinates": "c_0",
                "rms_waves": 0.1, "pv_waves": 0.4, "strehl_at_reference": 0.7, "rim": ["rim_0"], "zernike": {"coefficients": "z_0", "ordering": "fringe"} } ] }
            """, [0.5876], ["0°"],
            TestData.Buffer("opd_0", [2, 2], 0.1, -0.2, double.NaN, 0.3), TestData.Vector("c_0", -0.5, 0.5),
            TestData.Vector("z_0", 0, 0.01, -0.02), TestData.Buffer("rim_0", [3, 2], 1, 0, 0, 1, -1, 0));
        var m = Assert.Single(WavefrontData.Read(d)!.Maps);
        Assert.Equal("Field 1 · 0° · 0.5876 µm", m.Title);
        Assert.Equal((-0.5, 0.5), (m.Grid.X0, m.Grid.X1));
        Assert.Equal([0, 0.01, -0.02], m.Zernike);
        Assert.Equal((0.0, 1.0), m.Rim[0][1]);
        Assert.Equal((0.7, 0.4), (m.Strehl, m.Pv));
    }

    [Fact]
    public void GlassMapReadsCoordinatesStatusesAndTheDesignGlasses()
    {
        var d = TestData.Result("glass_map", """
            { "coordinates": "coordinates", "glasses": [
                {"kind": "catalog", "catalog": "SCHOTT", "name": "N-BK7", "status": "preferred", "surfaces": []},
                {"kind": "catalog", "catalog": "SCHOTT", "name": "F2", "status": {"other": 7}, "surfaces": [4, 7]},
                {"kind": "offset", "catalog": "SCHOTT", "name": "F2", "status": "standard", "surfaces": [4], "offset": {"dn_d": 0.001, "dv_d": -0.2}},
                {"kind": "model", "name": "M", "status": null, "surfaces": [2]} ] }
            """, TestData.Buffer("coordinates", [4, 3], 1.5168, 64.17, 0.5349, 1.62, 36.37, 0.5828, 1.621, 36.17, 0.58, 1.6, 50, double.NaN));
        var g = GlassMapData.Read(d)!;
        Assert.Equal(["SCHOTT/N-BK7", "SCHOTT/F2", "offset:SCHOTT/F2:4", "model:M:2"], g.Glasses.Select(x => x.Id));
        Assert.Equal((1.5168, 64.17), (g.Glasses[0].Nd, g.Glasses[0].Vd));
        Assert.Equal(GlassStatus.Preferred, g.Glasses[0].Status);
        Assert.Equal("Status 7", g.Glasses[1].StatusLabel);
        Assert.False(g.Glasses[0].InDesign);
        Assert.True(g.Glasses[3].InDesign);
        Assert.Equal("S4, S7", g.Glasses[1].SurfaceLabel);
        Assert.Equal("SCHOTT/F2", g.Glasses[2].NominalId);
        Assert.Equal("M (model)", g.Glasses[3].DisplayName);
        Assert.Equal([GlassStatus.Preferred, GlassStatus.Unknown], g.Statuses);
        Assert.Equal(0.6438 - 0.001682 * 64.17, GlassPoint.NormalLine(64.17), 12);
        Assert.Equal("Glasses 2  ·  In design 3", KeyNumbers.Text(d));
    }

    [Fact]
    public void GlassInfoReadsTheGlassAndItsCurve()
    {
        var d = TestData.Result("glass_info", """
            { "name": "N-SK16", "catalog": "BUILTIN", "nd": 1.62041, "vd": 60.32, "ne": 1.6229, "pgf": 0.5412, "catalog_nd": 1.62041, "catalog_vd": 60.32,
              "offset": {"dn_d": 0, "dv_d": 0}, "min_um": 0.31, "max_um": 2.5, "dispersion": {"formula": "sellmeier"}, "has_thermal_data": false,
              "cte": null, "density": 3.58, "comment": " ", "environment": {"temperature_c": 20, "pressure_atm": 1},
              "indices": [ {"wavelength_um": 0.4861, "n": 1.6276, "n_catalog": 1.6276, "absorption_per_mm": 0, "in_range": true} ] }
            """, TestData.Buffer("dispersion_curve", [2, 2], 0.4, 1.63, 0.7, 1.615));
        var g = GlassInfoData.Read(d)!;
        Assert.Equal("Sellmeier", g.Formula);
        Assert.False(g.HasOffset);
        Assert.Null(g.Cte);
        Assert.Equal(3.58, g.Density);
        Assert.Null(g.Comment);
        Assert.Equal([(0.4, 1.63), (0.7, 1.615)], g.Curve);
        Assert.Equal("nd 1.620410  ·  Vd 60.32", KeyNumbers.Text(d));
    }

    [Fact]
    public void ImageSimulationReadsTheImageNodesAndSource()
    {
        var d = TestData.Result("image_simulation", """
            { "image": "image", "pixel_size_mm": [0.01, 0.01], "psf_size": 64, "psf_spacing_mm": 0.0004,
              "field_mapping": [ {"field": [0, 0], "ideal_mm": [0, 0], "real_mm": [0, 0], "relative_illumination": 1},
                                 {"field": [0, 1], "ideal_mm": [0, 10], "real_mm": [0, 9.9], "relative_illumination": 0.9} ] }
            """, TestData.Buffer("image", [2, 2, 1], 0, 0.5, 1, 2));
        var s = ImageSimulationData.Read(d, TestData.Obj("""{"source": {"kind": "pattern", "pattern": "siemens_star"}}"""))!;
        Assert.Equal((2, 2, 1), (s.Image.Width, s.Image.Height, s.Image.Channels));
        Assert.Equal(-1, s.Nodes[1].DistortionPercent, 9);
        Assert.Equal(0.25, s.Image.ClippedFraction());
        Assert.Equal("Siemens star pattern", s.SourceTitle);
        Assert.Equal(2, s.SourceImage()!.Width);
        Assert.Equal((0.0, 0.5), s.FieldCenter);
        // Row 0 (low y) is drawn at the bottom: the first output row is the last input row.
        var px = s.Image.Bgra8();
        Assert.Equal(255, px[0]);   // input (0, 1) = 1 → white
        Assert.Equal(0, px[8]);     // input (0, 0) = 0 → black
        Assert.Equal("Pixel 10.00 µm  ·  Max distortion -1.000 %  ·  Min RI 0.900", KeyNumbers.Text(d));
        Assert.Null(ImageSimulationData.Read(d).SourceImage()); // no settings: unknown source
    }

    [Fact]
    public void SrgbAndPatternsFollowTheEngine()
    {
        Assert.Equal(0, Srgb.Byte(-1));
        Assert.Equal(255, Srgb.Byte(2));
        Assert.Equal(188, Srgb.Byte(0.5));
        Assert.Equal(0.5, Srgb.Decode(Srgb.Encode(0.5)), 12);
        var star = LinearImage.Pattern("siemens_star", 64, 64)!;
        Assert.InRange(star.Data.Average(), 0.45, 0.55);
        Assert.Null(LinearImage.Pattern("nope", 4, 4));
        var edge = LinearImage.Pattern("slanted_edge", 8, 8, 4)!;
        Assert.Equal(0, edge.Data[0]);
        Assert.Equal(1, edge.Data[7]);
    }

    [Fact]
    public void RadiometryReaders()
    {
        var gp = GhostPathsData.Read(TestData.Result("ghost_paths", """
            { "truncated": true, "ghosts": [ {"label": "R11·R9", "path": [11, 9], "relative_irradiance": 2.774e-7, "focus_from_sensor": -18.84,
              "spot_diameter": 16.3, "reflectance_product": 0.00314, "near_focus": true}, {"path": [3, 1]} ] }
            """))!;
        Assert.True(gp.Truncated);
        Assert.Equal("3·1", gp.Ghosts[1].Label);
        Assert.Equal("2.8e-7", GhostPathsData.Sci(2.774e-7));
        Assert.Equal("—", GhostPathsData.Sci(double.NaN));

        var gi = SensorMapsData.GhostIrradiance(TestData.Result("ghost_irradiance", """
            { "map": {"half_size": [2, 1]}, "total_flux_ratio": 0.0147, "total_peak_ratio": null, "ghosts": [ {"label": "R9·R7", "peak_ratio": 1.97e-5} ] }
            """, TestData.Buffer("total_ghost_map", [2, 3], 1, 2, 3, 4, 5, 6), TestData.Buffer("primary_map", [2, 3], new double[6])))!;
        Assert.Equal(["All ghosts", "Primary image"], gi.Maps.Select(m => m.Title));
        Assert.Equal((-2.0, 2.0, -1.0, 1.0), (gi.Maps[0].Grid.X0, gi.Maps[0].Grid.X1, gi.Maps[0].Grid.Y0, gi.Maps[0].Grid.Y1));
        Assert.Equal(("Ghost / primary peak", "—"), gi.Rows[1]);
        Assert.Equal(("R9·R7", "1.970e-05"), gi.Rows[2]);

        var sl = SensorMapsData.StrayLight(TestData.Result("stray_light", """
            { "detector": {"half_x": 1, "half_y": 1}, "map_classes": ["ghost", "scatter"], "stray_flux": 5.4, "stray_std_error": null,
              "stray_fraction": 0.011, "totals": [ {"class": "primary", "flux": 318} ] }
            """, TestData.Buffer("total_irradiance", [2, 2], 1, 2, 3, 4), TestData.Buffer("class_irradiance", [2, 2, 2], 1, 2, 3, 4, 5, 6, 7, 8)))!;
        Assert.Equal(["Stray total", "Ghost", "Scatter"], sl.Maps.Select(m => m.Title));
        Assert.Equal([5.0, 6, 7, 8], sl.Maps[2].Grid.Values);
        Assert.Equal(("± standard error", "—"), sl.Rows[1]);
        Assert.Equal(("Primary", "318 W"), sl.Rows[^1]);
    }

    [Fact]
    public void PolarizationAndTransmission()
    {
        var pol = PolarizationData.Read(TestData.Result("polarization_pupil_map", """{"grid": 3, "mean_intensity": 0.63, "max_retardance": 0.1}""",
            TestData.Buffer("pupil", [3, 2], 0, -1, 0, 0, 1, 0), TestData.Vector("reached", 1, 1, 0),
            TestData.Vector("intensity", 0.6, 0.7, 0.8), TestData.Vector("diattenuation", 0, 0.1, 0.2)))!;
        var m = pol.Map("Intensity")!;
        Assert.Equal(3, m.Nx);
        Assert.Equal(0.6, m.Values[0 * 3 + 1]);
        Assert.Equal(0.7, m.Values[1 * 3 + 1]);
        Assert.True(double.IsNaN(m.Values[1 * 3 + 2])); // did not reach the image
        Assert.Null(pol.Map("Retardance"));
        Assert.Equal(("Max retardance", "0.1 rad"), pol.Stats[2]);

        var t = TransmissionData.Read(TestData.Result("system_transmission", """
            { "fields": [ {"hx": 0, "hy": 0}, {"hx": 0, "hy": 1} ], "polychromatic": [0.63, 0.6],
              "points": [ {"field": 0, "wavelength_um": 0.48, "transmission": 0.62}, {"field": 0, "wavelength_um": 0.59, "transmission": 0.63},
                          {"field": 1, "wavelength_um": 0.48, "transmission": 0.6}, {"field": 1, "wavelength_um": 0.59, "transmission": 0.61} ] }
            """))!;
        Assert.Equal([0, 1], t.Fields);
        Assert.True(t.Spectral);
        Assert.Equal("H (0.00, +1.00)", t.FieldLabels[1]);
    }

    [Fact]
    public void GridDistortionAndFootprint()
    {
        var g = GridDistortionData.Read(TestData.Result("grid_distortion", """{"grid_size": 2, "max_abs_distortion_percent": 0.97}""",
            TestData.Buffer("paraxial", [4, 2], -1, -1, 1, -1, -1, 1, 1, 1), TestData.Buffer("real", [4, 2], -0.99, -0.99, 0.99, -0.99, -0.99, 0.99, double.NaN, double.NaN)))!;
        Assert.Equal(4, g.Lines(g.Real).Count());
        Assert.Equal([(-1.0, -1.0), (1.0, -1.0)], g.Lines(g.Paraxial).First());
        Assert.Null(GridDistortionData.Read(TestData.Result("grid_distortion", """{"grid_size": 3}""", TestData.Buffer("paraxial", [4, 2], new double[8]))));

        var f = FootprintData.Read(TestData.Result("footprint", """
            { "surface": 1, "semi_diameter": 15.6, "sets": [ {"wavelength": 0, "points": [0, 2]}, {"wavelength": 2, "points": [2, 5]} ] }
            """, TestData.Buffer("points", [3, 2], 0, 0, 1, 1, 2, 2), TestData.Buffer("clipped", [0, 2])))!;
        Assert.Equal(3, f.Points.Length); // the second range is clamped to the buffer
        Assert.Equal([0, 2], f.Wavelengths);
        Assert.Empty(f.Clipped);
    }
}
