//! `analyze_with_signatures`: the same result as `analyze`, plus every
//! tallied (class, signature) flux, whose per-class sums are the class
//! totals (so a front end can re-partition the totals by signature).

mod common;

use std::sync::Arc;

use common::*;
use mklight_core::JobContext;
use mklight_stray::{
    PathClass, StrayAnalysisSettings, StraySource, analyze, analyze_with_signatures,
};

#[test]
fn every_signature_is_returned_and_sums_to_the_class_totals() {
    let (s, opts) = double_gauss_with_barrels();
    let rs = Arc::new(resolve(s));
    let settings = StrayAnalysisSettings {
        scene: opts.scene,
        detector_margin: opts.detector_margin,
        source: StraySource::SystemField {
            index: 0,
            margin: 0.1,
            strength: 1.0,
        },
        rays: 3000,
        seed: 5,
        // No signature report and the ghost class excluded: the full list
        // still holds every class.
        top_signatures: 0,
        exclude_ghost_class: true,
        ..Default::default()
    };
    let plain = analyze(rs.clone(), &settings, &JobContext::new()).unwrap();
    let (r, sigs) = analyze_with_signatures(rs, &settings, &JobContext::new()).unwrap();
    // Forcing signature tallying changes nothing else.
    assert_eq!(r.totals, plain.totals);
    assert_eq!(r.stray_flux, plain.stray_flux);
    assert_eq!(r.total_irradiance, plain.total_irradiance);
    assert!(r.top_signatures.is_empty());
    let mut seen_ghost = false;
    for t in &r.totals {
        let sum: f64 = sigs
            .iter()
            .filter(|f| f.class == t.class)
            .map(|f| f.flux)
            .sum();
        assert!(
            (sum - t.flux).abs() <= 1e-12 * t.flux.max(1e-300),
            "{:?}: signatures {sum} vs total {}",
            t.class,
            t.flux
        );
        seen_ghost |= t.class == PathClass::Ghost && t.flux > 0.0;
    }
    assert!(seen_ghost, "the test needs ghost flux");
    // Largest first.
    assert!(sigs.windows(2).all(|w| w[0].flux >= w[1].flux));
}
