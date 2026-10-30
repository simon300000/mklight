//! Small built-in glass catalog so that the engine works without any external
//! data. The coefficients are the manufacturers' published Sellmeier
//! coefficients (and Malitson's for fused silica and CaF₂), which are factual
//! data. Full vendor catalogs are loaded at runtime from AGF-format files the
//! user supplies (see `agf`). Each entry is checked against its catalog
//! `n_d`/`V_d` in the tests below.

use std::collections::BTreeMap;

use super::{Dispersion, Glass, GlassCatalog};

pub const BUILTIN_CATALOG_NAME: &str = "BUILTIN";

fn sellmeier3(
    name: &str,
    b: [f64; 3],
    c: [f64; 3],
    nd: f64,
    vd: f64,
    min_um: f64,
    max_um: f64,
) -> Glass {
    Glass {
        name: name.to_string(),
        dispersion: Dispersion::Sellmeier {
            b: b.to_vec(),
            c: c.to_vec(),
        },
        nd,
        vd,
        min_um,
        max_um,
        thermal: None,
        cte: None,
        density: None,
        transmittance: Vec::new(),
        relative_cost: None,
        substitution: Default::default(),
        comment: String::new(),
    }
}

pub fn catalog() -> GlassCatalog {
    let glasses = vec![
        sellmeier3(
            "N-BK7",
            [1.03961212, 0.231792344, 1.01046945],
            [0.00600069867, 0.0200179144, 103.560653],
            1.51680,
            64.17,
            0.3,
            2.5,
        ),
        sellmeier3(
            "N-BAK1",
            [1.12365662, 0.309276848, 0.881511957],
            [0.00644742752, 0.0222284402, 107.297751],
            1.57250,
            57.55,
            0.3,
            2.5,
        ),
        sellmeier3(
            "N-SK16",
            [1.34317774, 0.241144399, 0.994317969],
            [0.00704687339, 0.0229005, 92.7508526],
            1.62041,
            60.32,
            0.31,
            2.5,
        ),
        sellmeier3(
            "N-LAK9",
            [1.46231905, 0.344399589, 1.15508372],
            [0.00724270156, 0.0243353131, 85.4686868],
            1.69100,
            54.71,
            0.31,
            2.5,
        ),
        sellmeier3(
            "N-BAF10",
            [1.5851495, 0.143559385, 1.08521269],
            [0.00926681282, 0.0424489805, 105.613573],
            1.67003,
            47.11,
            0.35,
            2.5,
        ),
        sellmeier3(
            "N-LAF2",
            [1.80984227, 0.15729555, 1.0930037],
            [0.0101711622, 0.0442431765, 100.687748],
            1.74397,
            44.85,
            0.32,
            2.5,
        ),
        sellmeier3(
            "F2",
            [1.34533359, 0.209073176, 0.937357162],
            [0.00997743871, 0.0470450767, 111.886764],
            1.62004,
            36.37,
            0.32,
            2.5,
        ),
        sellmeier3(
            "SF5",
            [1.52481889, 0.187085527, 1.42729015],
            [0.011254756, 0.0588995392, 129.141675],
            1.67270,
            32.21,
            0.37,
            2.5,
        ),
        sellmeier3(
            "N-SF6",
            [1.77931763, 0.338149866, 2.08734474],
            [0.0133714182, 0.0617533621, 174.01759],
            1.80518,
            25.36,
            0.37,
            2.5,
        ),
        sellmeier3(
            "N-SF11",
            [1.73759695, 0.313747346, 1.89878101],
            [0.013188707, 0.0623068142, 155.23629],
            1.78472,
            25.68,
            0.37,
            2.5,
        ),
        sellmeier3(
            "N-LASF9",
            [2.00029547, 0.298926886, 1.80691843],
            [0.0121426017, 0.0538736236, 156.530829],
            1.85025,
            32.17,
            0.37,
            2.5,
        ),
        sellmeier3(
            "FUSED_SILICA",
            [0.6961663, 0.4079426, 0.8974794],
            [
                0.0684043 * 0.0684043,
                0.1162414 * 0.1162414,
                9.896161 * 9.896161,
            ],
            1.45846,
            67.82,
            0.21,
            3.71,
        ),
        sellmeier3(
            "CAF2",
            [0.5675888, 0.4710914, 3.8484723],
            [
                0.050263605 * 0.050263605,
                0.1003909 * 0.1003909,
                34.649040 * 34.649040,
            ],
            1.43385,
            94.99,
            0.23,
            9.7,
        ),
    ];
    let mut map = BTreeMap::new();
    for g in glasses {
        map.insert(g.name.clone(), g);
    }
    GlassCatalog {
        name: BUILTIN_CATALOG_NAME.into(),
        glasses: map,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::lines;

    /// The dispersion data must reproduce each glass's catalog nd and vd.
    /// This catches mistyped coefficients.
    #[test]
    fn coefficients_consistent_with_nd_vd() {
        for g in catalog().glasses.values() {
            let nd = g.catalog_index(lines::D);
            let vd = (nd - 1.0) / (g.catalog_index(lines::F) - g.catalog_index(lines::C));
            assert!(
                (nd - g.nd).abs() < 5e-5,
                "{}: nd {nd} vs catalog {}",
                g.name,
                g.nd
            );
            assert!(
                (vd - g.vd).abs() < 0.15,
                "{}: vd {vd} vs catalog {}",
                g.name,
                g.vd
            );
        }
    }
}
