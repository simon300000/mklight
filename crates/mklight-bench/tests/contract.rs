//! The offline accuracy contract (`mklight_bench::comparison`, GPU §9.4
//! "Contracts"): every declared result field is classed, the classes are at
//! least as strict as the CONVENTIONS §7 budgets they derive from, and the
//! class rules (relative frequencies, outlines by Hausdorff distance,
//! statistical counts, comparison at common sampling coordinates) behave as
//! stated on known answers.
use mklight_bench::comparison::{self, Class, FIELDS, REFERENCE_ONLY, compare, rule};
use mklight_bench::solver::JOBS as SOLVERS;
use mklight_core::JobContext;
use mklight_engine::{
    Engine,
    analysis::AnalysisOutput,
    buffer::{Buffer, BufferData},
};
use serde_json::{Value, json};

fn out(json: Value, buffers: Vec<Buffer>) -> AnalysisOutput {
    AnalysisOutput {
        json,
        buffers,
        ..Default::default()
    }
}

fn run(analysis: &str, a: &AnalysisOutput, b: &AnalysisOutput) -> comparison::Comparison {
    compare(a, b, analysis, &JobContext::new()).unwrap()
}

#[test]
fn every_declared_result_field_of_every_compared_analysis_has_a_class() {
    let analyses: Vec<_> = Engine::default()
        .analyses()
        .into_iter()
        .filter(|a| a.gpu_path && !SOLVERS.contains(&a.name.as_str()))
        .collect();
    let mut missing = Vec::new();
    for info in &analyses {
        assert!(
            FIELDS.iter().any(|(a, _)| *a == info.name),
            "{} has no contract table",
            info.name
        );
        for leaf in comparison::schema_leaves(&info.result_schema) {
            if rule(&info.name, &leaf).is_none() {
                missing.push(format!("{}: {leaf}", info.name));
            }
        }
    }
    assert!(
        missing.is_empty(),
        "unclassed fields:\n{}",
        missing.join("\n")
    );
    for (a, _) in FIELDS {
        assert!(
            analyses.iter().any(|i| i.name == *a),
            "contract table for {a}, which is not a compared analysis"
        );
    }
}

/// Analyses without a GPU path run the reference on every backend request;
/// their results are still gated (a routed result must be the reference's),
/// so every declared field of every one of them is classed too.
#[test]
fn every_declared_result_field_of_every_reference_only_analysis_has_a_class() {
    let analyses: Vec<_> = Engine::default()
        .analyses()
        .into_iter()
        .filter(|a| !a.gpu_path)
        .collect();
    let mut missing = Vec::new();
    for info in &analyses {
        assert!(
            REFERENCE_ONLY.iter().any(|(a, _)| *a == info.name),
            "{} has no contract table",
            info.name
        );
        for leaf in comparison::schema_leaves(&info.result_schema) {
            if rule(&info.name, &leaf).is_none() {
                missing.push(format!("{}: {leaf}", info.name));
            }
        }
    }
    assert!(
        missing.is_empty(),
        "unclassed fields:\n{}",
        missing.join("\n")
    );
    for (a, _) in REFERENCE_ONLY {
        assert!(
            analyses.iter().any(|i| i.name == *a),
            "reference-only table for {a}, which has a GPU path or does not exist"
        );
    }
}

/// First-order entries are classed by key and unit: an F-number is
/// relative (2e-6), a length a position (2e-5 mm), an echoed input exact.
#[test]
fn first_order_entries_are_classed_by_key_and_unit() {
    let entries = |f: f64, efl: f64, field: f64| {
        out(
            json!({"entries":[
                {"key":"image_f_number","label":"","unit":"","value":f},
                {"key":"efl","label":"","unit":"mm","value":efl},
                {"key":"max_field","label":"","unit":"deg","value":field}],
                "wavelength_um":0.55}),
            vec![],
        )
    };
    let a = entries(1.4, 50.0, 20.0);
    assert!(run("first_order", &a, &a).reason.is_none());
    // 5e-6 mm on a 50 mm focal length passes the position gate.
    assert!(
        run("first_order", &a, &entries(1.4, 50.0 + 5e-6, 20.0))
            .reason
            .is_none()
    );
    // 5e-6 relative on an F-number exceeds the F-number gate.
    let c = run("first_order", &a, &entries(1.4 * (1. + 5e-6), 50.0, 20.0));
    assert!(c.reason.unwrap().contains("f_number"));
    // An echoed field definition must be equal.
    let c = run("first_order", &a, &entries(1.4, 50.0, 20.0 + 1e-9));
    assert!(c.reason.unwrap().contains("discrete values differ"));
}

/// Quantity by quantity: the full class budget never exceeds the
/// CONVENTIONS §7 budget of the physical quantity, and the gate is a third
/// of it.
#[test]
fn classes_are_at_least_as_strict_as_the_section_7_budgets() {
    let b = comparison::class_budget;
    // (class, reference value, §7 budget of that quantity)
    let rows = [
        (Class::Position, 10.0, 2e-5),          // ray position
        (Class::Direction, 0.5, 2e-6),          // direction cosine
        (Class::Waves, 3.0, 5e-3),              // OPD
        (Class::Modulation, 0.5, 2e-3),         // MTF
        (Class::Rms, 0.001, 2e-5),              // RMS spot radius: 2e-5 mm
        (Class::Rms, 1.0, 5e-3),                // or 0.5 % of the reference
        (Class::Outline(""), 0.0, 2e-5),        // outline vertices are positions
        (Class::Quantile, 0.01, 2e-5),          // radius on the energy curve
        (Class::PositionDifference, 1.0, 4e-5), // two positions
    ];
    for (class, reference, section7) in rows {
        assert!(
            b(class, reference) <= section7 * (1. + 1e-12),
            "{class:?}: {} > {section7}",
            b(class, reference)
        );
    }
    // A frequency is n sinθ/λ: the direction budget allows a relative error
    // of 2e-6/(n sinθ) ≥ 2e-6, so 2e-6 relative is the strictest reading,
    // for any aperture (here an F/1 image cone and 0.55 µm).
    let (sin, lambda_mm) = (0.5, 0.55e-3);
    let cutoff = sin / lambda_mm;
    assert!(b(Class::Frequency, cutoff) <= 2e-6 / lambda_mm);
    // F-numbers are 1/(2 n sinθ): the same relative bound.
    assert!(b(Class::FNumber, 1.0) / 1.0 <= 2e-6 / sin);
}

#[test]
fn frequencies_are_compared_relatively() {
    let a = out(
        json!({"fields":[{"cutoff":850.0,"field":0,"field_value":[0.0,0.0],"tangential_direction":[0.0,1.0],"pupil_sampling":[]}],"frequencies":"frequencies"}),
        vec![],
    );
    // 2e-8 relative (E2's conditioned-stop difference) passes; the generic
    // 2e-5 mm length tolerance used to fail it.
    let mut b = a.clone();
    b.json["fields"][0]["cutoff"] = json!(850.0 * (1. + 2e-8));
    assert!(run("mtf", &a, &b).reason.is_none());
    // 1e-5 relative exceeds the direction-derived 2e-6 budget.
    b.json["fields"][0]["cutoff"] = json!(850.0 * (1. + 1e-5));
    assert!(run("mtf", &a, &b).reason.is_some());
}

/// Convex hull of a square pattern, counterclockwise, with or without the
/// collinear edge points.
fn hull(collinear: bool, shift: f64) -> Vec<f64> {
    let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
    let mut v = Vec::new();
    for k in 0..4 {
        let (s, e) = (corners[k], corners[(k + 1) % 4]);
        for i in 0..if collinear { 4 } else { 1 } {
            let t = i as f64 / 4.0;
            v.extend([s.0 + t * (e.0 - s.0) + shift, s.1 + t * (e.1 - s.1), 5.0]);
        }
    }
    v
}

fn footprint(outline: Vec<f64>) -> AnalysisOutput {
    let n = outline.len() / 3;
    out(
        json!({"overlays":[{"field":0,"surface":1,"wavelength":0,"outline":[0,n],"points":[0,0]}],"sets":[],"surface":1}),
        vec![
            Buffer::new("global_outlines", vec![n, 3], BufferData::F64(outline)),
            Buffer::new("global_points", vec![0, 3], BufferData::F64(vec![])),
        ],
    )
}

#[test]
fn outlines_are_compared_by_hausdorff_distance_not_vertex_count() {
    let full = footprint(hull(true, 0.));
    let corners = footprint(hull(false, 0.));
    assert_eq!(full.buffers[0].shape[0], 16);
    assert_eq!(corners.buffers[0].shape[0], 4);
    let c = run("footprint", &full, &corners);
    assert!(c.reason.is_none(), "{:?}", c.reason);
    assert!(c.classes["outline"].worst_error < 1e-12);
    // A real displacement of the outline by 1e-4 mm fails the position gate.
    let c = run("footprint", &full, &footprint(hull(false, 1e-4)));
    assert!(c.reason.is_some());
}

#[test]
fn monte_carlo_counts_are_statistical_and_survival_counts_exact() {
    let stray = |events: i64| {
        out(
            json!({"energy":{"emitted":1.0,"events":events,"rays":20000},"emitted":1.0}),
            vec![],
        )
    };
    // Two roulette decisions out of a million events: 0.001 combined SE.
    assert!(
        run("stray_light", &stray(1_000_000), &stray(999_998))
            .reason
            .is_none()
    );
    // 1 % more events is 7 combined standard errors.
    assert!(
        run("stray_light", &stray(1_000_000), &stray(1_010_000))
            .reason
            .is_some()
    );
    let mut other = stray(1_000_000);
    other.json["energy"]["rays"] = json!(19_999);
    assert!(
        run("stray_light", &stray(1_000_000), &other)
            .reason
            .is_some()
    );

    // Class fluxes against their own standard errors (5 combined SE).
    let class = |flux: f64, se: f64| {
        out(
            json!({"energy":{"emitted":1.0},"totals":[{"class":"scatter","flux":flux,"std_error":se,"hits":40}]}),
            vec![],
        )
    };
    assert!(
        run("stray_light", &class(1e-4, 2e-5), &class(1.3e-4, 2e-5))
            .reason
            .is_none()
    );
    assert!(
        run("stray_light", &class(1e-4, 2e-6), &class(1.3e-4, 2e-6))
            .reason
            .is_some()
    );

    // The strongest signatures: a tail signature within 5σ of the other
    // list's cut may swap with its neighbour; a strong one may not go missing.
    let sig = |label: &str, flux: f64| json!({"class":"ghost","label":label,"events":[],"flux":flux,"std_error":1e-6,"hits":9});
    let list = |s: Vec<Value>| out(json!({"energy":{"emitted":1.0},"top_signatures":s}), vec![]);
    let a = list(vec![sig("p", 1e-3), sig("q", 2e-5)]);
    let tail = list(vec![sig("p", 1e-3), sig("r", 2.1e-5)]);
    let c = run("stray_light", &a, &tail);
    assert!(c.reason.is_none(), "{:?}", c.reason);
    assert_eq!(c.undefined, 2);
    let strong = list(vec![sig("q", 2e-5), sig("r", 2.1e-5)]);
    assert!(run("stray_light", &a, &strong).reason.is_some());

    // Fractions of ray counts are survival exact, not float-order exact.
    let spot = |f: f64| {
        out(
            json!({"fields":[{"polychromatic":{"vignetted_fraction":f}}]}),
            vec![],
        )
    };
    let (third, other) = (1.0 / 3.0, 1.0 - 2.0 / 3.0);
    assert_ne!(third, other);
    assert!(
        run("spot_diagram", &spot(third), &spot(other))
            .reason
            .is_none()
    );
    assert!(
        run("spot_diagram", &spot(127.0 / 381.0), &spot(128.0 / 381.0))
            .reason
            .is_some()
    );
}

#[test]
fn unmapped_fields_fail_loudly() {
    let a = out(
        json!({"cutoff_frequency": 100.0, "novel_quantity": 1.0}),
        vec![],
    );
    let c = run("geometric_mtf", &a, &a.clone());
    let reason = c.reason.expect("an unmapped field must fail");
    assert!(
        reason.contains("unmapped result field `novel_quantity`"),
        "{reason}"
    );
}

fn mtf(frequencies: &[f64], error: f64) -> AnalysisOutput {
    let cutoff = 400.0;
    let curve: Vec<f64> = frequencies
        .iter()
        .map(|f| {
            let s = (f / cutoff).min(1.0);
            2.0 / std::f64::consts::PI * (s.acos() - s * (1.0 - s * s).sqrt()) + error
        })
        .collect();
    out(
        json!({"fields":[{"cutoff":cutoff,"field":0,"field_value":[0.0,0.0],"tangential_direction":[0.0,1.0],
            "pupil_sampling":[{"grid":frequencies.len(),"wavelength":0}],
            "tangential":"tangential_0","sagittal":"sagittal_0","diffraction_limit":"diffraction_limit_0"}],
            "frequencies":"frequencies"}),
        vec![
            Buffer::f64_1d("tangential_0", curve.clone()),
            Buffer::f64_1d("sagittal_0", curve.clone()),
            Buffer::f64_1d("diffraction_limit_0", curve),
            Buffer::f64_1d("frequencies", frequencies.to_vec()),
        ],
    )
}

#[test]
fn curves_on_different_axes_are_compared_at_common_frequencies() {
    let fine: Vec<f64> = (0..161).map(|i| i as f64 * 2.5).collect();
    let coarse: Vec<f64> = (0..41).map(|i| i as f64 * 10.0).collect();
    // Same physical curve, different backend sampling: compared on the
    // reference's frequencies after linear resampling; the grid choice is
    // recorded, not gated.
    let c = run("mtf", &mtf(&fine, 0.), &mtf(&coarse, 0.));
    assert!(c.reason.is_none(), "{:?}", c.reason);
    assert!(c.resampled >= 3 && !c.sampling_differences.is_empty());
    // Index-wise comparison would have failed on the lengths alone; a real
    // MTF error at common frequencies still fails.
    assert!(
        run("mtf", &mtf(&fine, 0.), &mtf(&coarse, 3e-3))
            .reason
            .is_some()
    );
    // Equal sampling compares index by index.
    let c = run("mtf", &mtf(&coarse, 0.), &mtf(&coarse, 1e-4));
    assert!(c.reason.is_none() && c.resampled == 0);
}

fn wavefront(n: usize, bump: f64) -> AnalysisOutput {
    let coordinates: Vec<f64> = (0..n)
        .map(|i| -1.0 + (2 * i + 1) as f64 / n as f64)
        .collect();
    let mut opd = Vec::new();
    for v in &coordinates {
        for u in &coordinates {
            let r2 = u * u + v * v;
            opd.push(if r2 <= 0.9 {
                0.25 * (2.0 * r2 - 1.0) + 0.1 * u + bump * (-r2 * 50.0).exp()
            } else {
                f64::NAN
            });
        }
    }
    let amplitude: Vec<f64> = opd
        .iter()
        .map(|o| if o.is_nan() { 0.0 } else { 1.0 })
        .collect();
    out(
        json!({"maps":[{"field":0,"field_value":[0.0,0.0],"grid":n,"wavelength":0,"wavelength_um":0.55,
            "rms_waves":0.1,"opd_waves":"opd_waves_0","coordinates":"coordinates_0","amplitude":"amplitude_0"}]}),
        vec![
            Buffer::f64_1d("coordinates_0", coordinates),
            Buffer::new("opd_waves_0", vec![n, n], BufferData::F64(opd)),
            Buffer::new("amplitude_0", vec![n, n], BufferData::F64(amplitude)),
        ],
    )
}

#[test]
fn pupil_maps_on_different_grids_are_compared_at_common_pupil_coordinates() {
    let a = wavefront(64, 0.);
    let mut b = wavefront(256, 0.);
    // The amplitude mask edge does not resample (a step); compare the OPD.
    b.buffers.retain(|x| x.name != "amplitude_0");
    let mut a2 = a.clone();
    a2.buffers.retain(|x| x.name != "amplitude_0");
    let c = run("wavefront_map", &a2, &b);
    let failures: Vec<_> = c.failures.iter().filter(|f| !f.contains("grid")).collect();
    assert!(failures.is_empty(), "{failures:?}");
    assert!(c.resampled >= 1);
    // A 0.01-wave bump is an error at the same pupil coordinates.
    let mut bumped = wavefront(256, 0.01);
    bumped.buffers.retain(|x| x.name != "amplitude_0");
    assert!(run("wavefront_map", &a2, &bumped).reason.is_some());
}

fn psf(size: usize, spacing: f64, shift: f64) -> AnalysisOutput {
    let mut data = Vec::new();
    let w = 4e-3;
    for j in 0..size {
        for i in 0..size {
            let x = (i as f64 - (size / 2) as f64) * spacing - shift;
            let y = (j as f64 - (size / 2) as f64) * spacing;
            data.push((-(x * x + y * y) / (w * w)).exp());
        }
    }
    out(
        json!({"psfs":[{"field":0,"field_value":[0.0,0.0],"centroid_mm":[0.0,0.0],"peak_mm":[0.0,0.0],
            "reference_mm":[0.0,0.0],"strehl":0.9,"pupil_sampling":[],
            "psf":{"center_mm":[0.0,0.0],"spacing_mm":spacing,"size":size,"data":"data_0",
                "diffraction_peak":1.0,"total_energy":1e-5,"normalization":"strehl"}}]}),
        vec![Buffer::new(
            "data_0",
            vec![size, size],
            BufferData::F64(data),
        )],
    )
}

#[test]
fn psfs_on_different_image_grids_are_compared_at_common_image_coordinates() {
    let c = run("psf", &psf(128, 2.5e-4, 0.), &psf(256, 1.25e-4, 0.));
    assert!(c.reason.is_none(), "{:?}", c.reason);
    assert_eq!(c.resampled, 1);
    // A 1 µm displacement of the PSF is an image error at common coordinates.
    assert!(
        run("psf", &psf(128, 2.5e-4, 0.), &psf(256, 1.25e-4, 1e-3))
            .reason
            .is_some()
    );
}

#[test]
fn fit_condition_numbers_are_diagnostics_and_seidel_axes_of_vanishing_terms_undefined() {
    let map = |condition: f64, angle: f64| {
        out(
            json!({"maps":[{"field":0,"grid":8,"zernike_condition":condition,
                "seidel":{"astigmatism":2e-4,"astigmatism_angle_deg":angle,"coma":0.5,"coma_angle_deg":10.0}}]}),
            vec![],
        )
    };
    // 1e-7 relative on a condition number of ~1.7e4 (E2's near-vignetting row).
    let c = run(
        "wavefront_map",
        &map(17000.0, 30.0),
        &map(17000.0017, -60.0),
    );
    assert!(c.reason.is_none(), "{:?}", c.reason);
    assert_eq!(c.undefined, 1);
    // Orientation is periodic: 0° and 180° astigmatism are the same axis.
    let defined = |angle: f64| {
        let mut m = map(17000.0, angle);
        m.json["maps"][0]["seidel"]["astigmatism"] = json!(0.3);
        m
    };
    assert!(
        run("wavefront_map", &defined(90.0), &defined(-90.0))
            .reason
            .is_none()
    );
    assert!(
        run("wavefront_map", &defined(90.0), &defined(90.1))
            .reason
            .is_some()
    );
}
