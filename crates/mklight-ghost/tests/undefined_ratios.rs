//! Ghost-to-signal ratios against a primary that puts no light on the map
//! are undefined (`None`, JSON `null`): the ranking then goes by absolute
//! peak irradiance, and the result reads back from its own JSON.

use crate::common;

use common::*;
use mklight_core::{JobContext, samples};
use mklight_ghost::{GhostIrradiance, GhostSettings, IrradianceSettings, ghost_irradiance};
use mklight_trace::FieldPoint;

/// A source at 1.5 field heights of the Double Gauss: its image misses the
/// map, many of its ghosts do not.
fn out_of_field() -> IrradianceSettings {
    IrradianceSettings {
        points: vec![FieldPoint::normalized(0.0, 1.5)],
        wavelengths: vec![1],
        pupil_samples: 24,
        map_bins: [16, 16],
        ..Default::default()
    }
}

#[test]
fn out_of_field_source_ranks_by_absolute_peak() {
    let m = model(samples::double_gauss(), GhostSettings::default());
    let r = ghost_irradiance(&m, &out_of_field(), &JobContext::new()).unwrap();
    assert!(r.primary.fields[0].flux > 0.0, "the primary is detected");
    assert_eq!(r.primary.flux, 0.0, "but not on the map");
    assert_eq!(r.primary.flux_ratio, Some(1.0));
    assert_eq!(r.primary.peak_ratio, Some(1.0));
    assert_eq!(r.total_flux_ratio, None);
    assert_eq!(r.total_peak_ratio, None);

    let lit = r.ghosts.iter().filter(|g| g.flux > 0.0).count();
    assert!(lit > 4 && lit < r.ghosts.len(), "{lit}");
    for (k, g) in r.ghosts.iter().enumerate() {
        if k < lit {
            assert!(g.flux > 0.0 && g.peak() > 0.0, "{}", g.label);
            assert_eq!((g.flux_ratio, g.peak_ratio), (None, None), "{}", g.label);
        } else {
            // No light against no light is the ratio 0.
            assert_eq!(g.flux, 0.0, "{}", g.label);
            assert_eq!((g.flux_ratio, g.peak_ratio), (Some(0.0), Some(0.0)));
        }
    }
    // Strongest absolute peak first, then flux, then path order.
    for w in r.ghosts[..lit].windows(2) {
        let key = |g: &mklight_ghost::PathIrradiance| (g.peak(), g.flux);
        assert!(
            key(&w[0]) >= key(&w[1]),
            "{} before {}",
            w[0].label,
            w[1].label
        );
    }
    assert!(
        r.ghosts[..lit].windows(2).any(|w| w[0].path > w[1].path),
        "not path order"
    );
    assert!(r.ghosts[lit..].windows(2).all(|w| w[0].path < w[1].path));

    // Undefined ratios are `null`, and the result reads its own JSON.
    let json = serde_json::to_string(&r).unwrap();
    assert!(json.contains("\"total_flux_ratio\":null"));
    assert!(json.contains("\"peak_ratio\":null"));
    let back: GhostIrradiance = serde_json::from_str(&json).unwrap();
    assert_eq!(back, r);
}

#[test]
fn undefined_peak_ratios_lead_the_defined_ones() {
    let m = model(samples::double_gauss(), GhostSettings::default());
    let s = IrradianceSettings {
        fields: vec![0],
        ..out_of_field()
    };
    let r = ghost_irradiance(&m, &s, &JobContext::new()).unwrap();
    // The axial field's image is on the map, the point's is not.
    let p = &r.primary.fields;
    assert!(p[0].peak > 0.0 && p[1].peak == 0.0 && p[1].flux > 0.0);
    assert!(r.primary.flux > 0.0);
    assert!(r.total_flux_ratio.unwrap() > 0.0);
    assert!(r.total_peak_ratio.unwrap() > 0.0);

    let undefined = r.ghosts.iter().filter(|g| g.peak_ratio.is_none()).count();
    assert!(undefined > 0 && undefined < r.ghosts.len(), "{undefined}");
    for (k, g) in r.ghosts.iter().enumerate() {
        // The flux ratio is against the whole primary on the map.
        assert_eq!(g.flux_ratio, Some(g.flux / r.primary.flux), "{}", g.label);
        // Undefined exactly where the ghost has light and the primary none.
        assert_eq!(
            g.peak_ratio.is_none(),
            g.fields[1].peak > 0.0,
            "{}",
            g.label
        );
        assert_eq!(g.peak_ratio.is_none(), k < undefined, "{}", g.label);
        if let Some(ratio) = g.peak_ratio {
            assert_eq!(ratio, g.fields[0].peak / p[0].peak, "{}", g.label);
        }
    }
    for w in r.ghosts[..undefined].windows(2) {
        assert!((w[0].peak(), w[0].flux) >= (w[1].peak(), w[1].flux));
    }
    for w in r.ghosts[undefined..].windows(2) {
        assert!(w[0].peak_ratio.unwrap() >= w[1].peak_ratio.unwrap());
    }

    let json = serde_json::to_string(&r).unwrap();
    let back: GhostIrradiance = serde_json::from_str(&json).unwrap();
    assert_eq!(back, r);
}
