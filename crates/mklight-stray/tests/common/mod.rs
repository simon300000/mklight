#![allow(dead_code)]

use std::sync::Arc;

use mklight_core::materials::MaterialLibrary;
use mklight_core::model::*;
use mklight_core::resolved::ResolvedSystem;
use mklight_core::{JobContext, Snapshot};
use mklight_stray::{Scene, SceneOptions, SourceSpec, StrayResult, StraySettings};

pub fn resolve(s: OpticalSystem) -> ResolvedSystem {
    let snap = Snapshot::new(s).unwrap();
    ResolvedSystem::new(&snap, Arc::new(MaterialLibrary::with_builtin())).unwrap()
}

pub fn fixed(s: &mut Surface, r: f64) {
    s.semi_diameter = SemiDiameter::Fixed { value: r };
}

pub fn one_wavelength(s: &mut OpticalSystem) {
    s.wavelengths = WavelengthSet {
        wavelengths: vec![Wavelength {
            um: 0.55,
            weight: 1.0,
        }],
        primary: 0,
    };
}

/// Plane-parallel window of index `n` and thickness `t`, image `back` mm
/// behind it, clear semi-diameters 10 (window) and 15 (image).
pub fn window(n: f64, t: f64, back: f64) -> OpticalSystem {
    let mut s = OpticalSystem::new_empty();
    one_wavelength(&mut s);
    let mut s1 = Surface::new(1, SurfaceShape::Plane, t, MaterialRef::Constant { n });
    let mut s2 = Surface::new(2, SurfaceShape::Plane, back, MaterialRef::Air);
    let mut s3 = Surface::new(3, SurfaceShape::Plane, 0.0, MaterialRef::Air);
    fixed(&mut s1, 10.0);
    fixed(&mut s2, 10.0);
    fixed(&mut s3, 15.0);
    s.surfaces = vec![
        Surface::new(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
        s1,
        s2,
        s3,
    ];
    s.stop = 1;
    s
}

/// Biconvex singlet (n = 1.5) with a barrel from surface 1 to the image.
pub fn barrel_singlet(barrel_finish: ScatterModel) -> OpticalSystem {
    let mut s = OpticalSystem::new_empty();
    one_wavelength(&mut s);
    let mut s1 = Surface::new(
        1,
        SurfaceShape::sphere(40.0),
        4.0,
        MaterialRef::Constant { n: 1.5 },
    );
    let mut s2 = Surface::new(2, SurfaceShape::sphere(-40.0), 35.0, MaterialRef::Air);
    let mut s3 = Surface::new(3, SurfaceShape::Plane, 0.0, MaterialRef::Air);
    fixed(&mut s1, 8.0);
    fixed(&mut s2, 8.0);
    fixed(&mut s3, 10.0);
    s1.edge.mechanical_semi_diameter = Some(9.0);
    s2.edge.mechanical_semi_diameter = Some(9.0);
    s.surfaces = vec![
        Surface::new(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
        s1,
        s2,
        s3,
    ];
    s.stop = 1;
    s.mechanics = vec![
        MechanicalPart::Barrel {
            from: SurfaceId(1),
            to: SurfaceId(3),
            inner_radius: 12.0,
            finish: barrel_finish,
        },
        // A front retaining ring closing the gap between the lens and the barrel.
        MechanicalPart::AnnularBaffle {
            surface: SurfaceId(1),
            z: -0.5,
            inner_radius: 8.0,
            outer_radius: 12.0,
            finish: ScatterModel::Lambertian { tis: 0.02 },
        },
    ];
    s
}

pub fn scene(s: OpticalSystem) -> Scene {
    Scene::from_system(&resolve(s), &SceneOptions::default()).unwrap()
}

pub fn run(scene: &Scene, src: &SourceSpec, set: &StraySettings) -> StrayResult {
    let r = mklight_stray::run(scene, src, set, &JobContext::new()).unwrap();
    assert!(r.energy.is_balanced(), "{}", r.energy.summary());
    r
}

/// Mean and standard error of `f` over independent seeds.
pub fn seeds_mean(k: u64, mut f: impl FnMut(u64) -> f64) -> (f64, f64) {
    let xs: Vec<f64> = (0..k).map(|i| f(1000 + i)).collect();
    let m = xs.iter().sum::<f64>() / k as f64;
    let v = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (k - 1) as f64;
    (m, (v / k as f64).sqrt())
}

use mklight_stray::{StraySystem, SystemOptions};

/// System options with every scatter mechanism switched off: black (fully
/// absorbing, non-scattering) edges, flats, mounts and stop plates.
pub fn no_scatter_options() -> SystemOptions {
    let mut o = SystemOptions::default();
    o.scene.default_edge_finish = ScatterModel::Lambertian { tis: 0.0 };
    o.scene.include_mechanics = false;
    o
}

pub fn stray_system(s: OpticalSystem, opts: &SystemOptions) -> StraySystem {
    StraySystem::new(Arc::new(resolve(s)), opts).unwrap()
}

/// `(a − b)/σ`.
pub fn z(a: f64, b: f64, sigma: f64) -> f64 {
    (a - b) / sigma
}

/// The sample Double Gauss in a black-anodised barrel, with its field
/// limited to ±10° and 0.5 mm mounting rims (at the full ±14° field the
/// automatic clear apertures of L1 and L6 make their surfaces intersect).
/// The barrel runs from the front vertex to the image, 1 mm outside the
/// largest element, and a retaining ring closes it from the first
/// element's clear aperture to the tube.
pub fn double_gauss_with_barrels() -> (OpticalSystem, SystemOptions) {
    let mut base = mklight_core::samples::double_gauss();
    base.fields.fields.truncate(2);
    let mut opts = SystemOptions::default();
    opts.scene.default_edge_margin = 0.5;
    let sys = stray_system(base.clone(), &opts);
    let rmax = sys
        .scene
        .surfaces
        .iter()
        .take(base.surfaces.len() - 1)
        .flatten()
        .map(|g| g.mechanical_semi_diameter)
        .fold(0.0, f64::max);
    let front = sys.scene.surfaces[1].as_ref().unwrap();
    let mut s = base;
    let image = SurfaceId(s.surfaces.last().unwrap().id.0);
    let black = ScatterModel::Lambertian { tis: 0.04 };
    s.mechanics = vec![
        MechanicalPart::Barrel {
            from: SurfaceId(1),
            to: image,
            inner_radius: rmax + 1.0,
            finish: black.clone(),
        },
        MechanicalPart::AnnularBaffle {
            surface: SurfaceId(1),
            z: -0.5,
            inner_radius: front.clear_semi_diameter,
            outer_radius: rmax + 1.0,
            finish: black,
        },
    ];
    (s, opts)
}
