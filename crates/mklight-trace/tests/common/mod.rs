#![allow(dead_code, clippy::needless_range_loop)]

use std::sync::Arc;

use mklight_core::materials::MaterialLibrary;
use mklight_core::{
    ApertureSpec, Field, FieldKind, FieldSet, MaterialRef, OpticalSystem, ResolvedSystem, Snapshot,
    Surface, SurfaceShape, Wavelength, WavelengthSet,
};
use mklight_trace::{SequentialTracer, TracerSettings};

pub const D_LINE: f64 = 0.5875618;

pub fn resolve(sys: OpticalSystem) -> ResolvedSystem {
    let snap = Snapshot::new(sys).expect("valid system");
    ResolvedSystem::new(&snap, Arc::new(MaterialLibrary::with_builtin())).expect("resolve")
}

pub fn tracer(sys: OpticalSystem) -> SequentialTracer {
    SequentialTracer::new(Arc::new(resolve(sys))).expect("tracer")
}

pub fn tracer_with(sys: OpticalSystem, settings: TracerSettings) -> SequentialTracer {
    SequentialTracer::with_settings(Arc::new(resolve(sys)), settings).expect("tracer")
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

pub fn angle_fields(ys: &[f64]) -> FieldSet {
    FieldSet {
        kind: FieldKind::ObjectAngle,
        fields: ys.iter().map(|&y| Field::new(0.0, y)).collect(),
    }
}

/// Monochromatic system at infinity with the given surfaces (object prepended,
/// image appended), stop at surface 1.
pub fn infinite_system(epd: f64, mut body: Vec<Surface>, image_ids_from: u64) -> OpticalSystem {
    let mut s = OpticalSystem::new_empty();
    s.wavelengths = mono();
    s.aperture = ApertureSpec::EntrancePupilDiameter { value: epd };
    let mut surfaces = vec![surf(
        0,
        SurfaceShape::Plane,
        f64::INFINITY,
        MaterialRef::Air,
    )];
    surfaces.append(&mut body);
    surfaces.push(surf(
        image_ids_from,
        SurfaceShape::Plane,
        0.0,
        MaterialRef::Air,
    ));
    s.surfaces = surfaces;
    s.stop = 1;
    s
}

/// Least-squares fit of `y ≈ Σ c_k x^p_k` (normal equations, small systems).
pub fn poly_fit(xs: &[f64], ys: &[f64], powers: &[i32]) -> Vec<f64> {
    let m = powers.len();
    let mut a = vec![vec![0.0; m + 1]; m];
    for (x, y) in xs.iter().zip(ys) {
        let basis: Vec<f64> = powers.iter().map(|&p| x.powi(p)).collect();
        for i in 0..m {
            for j in 0..m {
                a[i][j] += basis[i] * basis[j];
            }
            a[i][m] += basis[i] * y;
        }
    }
    // Gaussian elimination with partial pivoting.
    for c in 0..m {
        let piv = (c..m)
            .max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))
            .unwrap();
        a.swap(c, piv);
        for r in 0..m {
            if r != c {
                let f = a[r][c] / a[c][c];
                for k in c..=m {
                    a[r][k] -= f * a[c][k];
                }
            }
        }
    }
    (0..m).map(|i| a[i][m] / a[i][i]).collect()
}
