//! Shared fixtures of the analysis integration tests.
#![allow(dead_code)]

pub use mklight_analysis::tracer_for;
use mklight_core::{
    ApertureSpec, Field, FieldKind, FieldSet, Interaction, MaterialRef, OpticalSystem, Surface,
    SurfaceShape, Wavelength, WavelengthSet,
};
use mklight_trace::SequentialTracer;

pub fn mono() -> WavelengthSet {
    WavelengthSet {
        wavelengths: vec![Wavelength {
            um: 0.5875618,
            weight: 1.0,
        }],
        primary: 0,
    }
}

pub fn surf(id: u64, shape: SurfaceShape, t: f64, m: MaterialRef) -> Surface {
    Surface::new(id, shape, t, m)
}

pub fn angle_fields(ys: &[f64]) -> FieldSet {
    FieldSet {
        kind: FieldKind::ObjectAngle,
        fields: ys.iter().map(|&y| Field::new(0.0, y)).collect(),
    }
}

/// Object at infinity, `body` surfaces (ids from 1), then the image.
pub fn infinite_system(epd: f64, body: Vec<Surface>, stop: usize, fields: &[f64]) -> OpticalSystem {
    let mut s = OpticalSystem::new_empty();
    s.aperture = ApertureSpec::EntrancePupilDiameter { value: epd };
    s.fields = angle_fields(fields);
    s.wavelengths = mono();
    let n = body.len();
    let mut surfaces = vec![surf(
        0,
        SurfaceShape::Plane,
        f64::INFINITY,
        MaterialRef::Air,
    )];
    surfaces.extend(body);
    surfaces.push(surf(
        n as u64 + 1,
        SurfaceShape::Plane,
        0.0,
        MaterialRef::Air,
    ));
    s.surfaces = surfaces;
    s.stop = stop;
    s
}

/// Ideal thin lens of focal length `f` as the stop, image at its focus.
pub fn ideal_lens(f: f64, epd: f64, fields: &[f64]) -> OpticalSystem {
    infinite_system(
        epd,
        vec![surf(
            1,
            SurfaceShape::Paraxial { focal_length: f },
            f,
            MaterialRef::Air,
        )],
        1,
        fields,
    )
}

/// Concave spherical mirror of radius `r` (> 0) with the stop at its centre
/// of curvature, image at the paraxial focus.
pub fn concentric_mirror(r: f64, epd: f64, fields: &[f64]) -> OpticalSystem {
    let mut m = surf(2, SurfaceShape::sphere(-r), -0.5 * r, MaterialRef::Air);
    m.interaction = Interaction::Reflect;
    infinite_system(
        epd,
        vec![surf(1, SurfaceShape::Plane, r, MaterialRef::Air), m],
        1,
        fields,
    )
}

pub fn tracer(sys: OpticalSystem) -> SequentialTracer {
    tracer_for(sys).expect("tracer")
}

/// Every sample system with its tracer.
pub fn samples() -> Vec<(&'static str, SequentialTracer)> {
    mklight_core::samples::all()
        .into_iter()
        // Legacy fixture set for the classical rotational identities; extended
        // samples are exercised independently in extended.rs and golden.rs.
        .filter(|(_, s)| !mklight_core::samples::has_extended_surfaces(s))
        .map(|(n, s)| (n, tracer(s)))
        .collect()
}
