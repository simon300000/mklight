#![allow(dead_code)]

use std::sync::Arc;

use mklight_core::materials::MaterialLibrary;
use mklight_core::{
    ApertureSpec, CoatingDef, Field, FieldKind, FieldSet, Layer, MaterialRef, OpticalSystem,
    ResolvedSystem, Snapshot, Surface, SurfaceShape, Wavelength, WavelengthSet,
};
use mklight_ghost::{ClipMode, GhostModel, GhostSettings};

pub const D_LINE: f64 = 0.5875618;

pub fn resolve_with(sys: OpticalSystem, lib: MaterialLibrary) -> Arc<ResolvedSystem> {
    let snap = Snapshot::new(sys).expect("valid system");
    Arc::new(ResolvedSystem::new(&snap, Arc::new(lib)).expect("resolve"))
}

pub fn resolve(sys: OpticalSystem) -> Arc<ResolvedSystem> {
    resolve_with(sys, MaterialLibrary::with_builtin())
}

pub fn model(sys: OpticalSystem, settings: GhostSettings) -> GhostModel {
    GhostModel::from_system(resolve(sys), settings).expect("ghost model")
}

pub fn unclipped() -> GhostSettings {
    GhostSettings {
        clip: ClipMode::None,
        ..Default::default()
    }
}

pub fn mono() -> WavelengthSet {
    WavelengthSet {
        wavelengths: vec![Wavelength {
            um: D_LINE,
            weight: 1.0,
        }],
        primary: 0,
    }
}

pub fn surf(id: u64, shape: SurfaceShape, t: f64, m: MaterialRef) -> Surface {
    Surface::new(id, shape, t, m)
}

/// Object at infinity, stop plane (air) 5 mm before a plane-parallel plate
/// of index `n` and thickness `d`, image plane `back` mm behind it. Field
/// angles in degrees.
pub fn plate(
    n: f64,
    d: f64,
    back: f64,
    fields_deg: &[f64],
    coating: Option<CoatingDef>,
) -> OpticalSystem {
    let mut s = OpticalSystem::new_empty();
    s.wavelengths = mono();
    s.aperture = ApertureSpec::EntrancePupilDiameter { value: 10.0 };
    s.fields = FieldSet {
        kind: FieldKind::ObjectAngle,
        fields: fields_deg.iter().map(|&y| Field::new(0.0, y)).collect(),
    };
    let mut front = surf(2, SurfaceShape::Plane, d, MaterialRef::Constant { n });
    let mut rear = surf(3, SurfaceShape::Plane, back, MaterialRef::Air);
    if let Some(c) = coating {
        s.coatings.insert("C".into(), c);
        front.coating = Some("C".into());
        rear.coating = Some("C".into());
    }
    s.surfaces = vec![
        surf(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
        surf(1, SurfaceShape::Plane, 5.0, MaterialRef::Air),
        front,
        rear,
        surf(4, SurfaceShape::Plane, 0.0, MaterialRef::Air),
    ];
    s.stop = 1;
    s
}

/// Single-layer quarter-wave coating of index `nf` at `lambda`.
pub fn quarter_wave(nf: f64, lambda: f64) -> CoatingDef {
    CoatingDef::ThinFilm {
        reference_wavelength_um: lambda,
        layers: vec![Layer {
            material: MaterialRef::Constant { n: nf },
            thickness_um: 0.0,
            quarter_waves: Some(1.0),
            extinction: 0.0,
        }],
        mirror_substrate: None,
        substrate: Default::default(),
    }
}

/// Bare normal-incidence Fresnel reflectance.
pub fn r_normal(n1: f64, n2: f64) -> f64 {
    ((n2 - n1) / (n2 + n1)).powi(2)
}
