//! `ResolvedSystem`: a snapshot with every material and coating resolved and
//! every frame placed. It is the exact numeric input that tracers and the GPU
//! packer consume. Building it does not trace any rays.

use std::sync::Arc;

use crate::error::{MkError, MkResult};
use crate::materials::coating::ResolvedCoating;
use crate::materials::{MaterialLibrary, ResolvedMaterial};
use crate::math::{Frame, Mat3, Vec3};
use crate::model::{
    ApertureShape, Interaction, OpticalSystem, Placement, SemiDiameter, SurfaceId, SurfaceShape,
    TransformOrder,
};
use crate::snapshot::Snapshot;

#[derive(Clone, Debug)]
pub struct ResolvedSurface {
    pub index: usize,
    pub id: SurfaceId,
    pub shape: SurfaceShape,
    pub interaction: Interaction,
    /// Global frame of the surface (vertex and orientation, after placement).
    /// Surface 1's pre-placement frame is the global origin. For an object at
    /// infinity, the object frame is meaningless and is set to identity.
    pub frame: Frame,
    /// Maps this surface's local coordinates into the next surface's local
    /// coordinates. Tracers use only these relative transforms, never global
    /// positions (CONVENTIONS §7.1). `None` for the image surface.
    pub to_next: Option<Frame>,
    /// True when `to_next` is a pure +z translation by `thickness`.
    pub to_next_is_axial: bool,
    pub thickness: f64,
    /// Medium after the surface. A mirror, dummy or coordinate break inherits
    /// the medium before it.
    pub medium_after: ResolvedMaterial,
    /// `n_after[w]` is the index after the surface at system wavelength `w`.
    pub n_after: Vec<f64>,
    /// Internal absorption (1/mm) after the surface, at each system wavelength.
    pub alpha_after: Vec<f64>,
    /// A fixed clear semi-diameter that clips rays (None when automatic).
    pub fixed_semi_diameter: Option<f64>,
    pub aperture: Option<ApertureShape>,
    pub coating: ResolvedCoating,
}

impl ResolvedSurface {
    /// Does a local hit point pass this surface's clipping aperture?
    pub fn passes_aperture(&self, x: f64, y: f64) -> bool {
        if let Some(a) = &self.aperture {
            return a.passes(x, y);
        }
        match self.fixed_semi_diameter {
            Some(radius) => ApertureShape::Circular { radius }.passes(x, y),
            None => true,
        }
    }
    pub fn is_mirror(&self) -> bool {
        self.interaction == Interaction::Reflect
    }
    pub fn is_passive(&self) -> bool {
        matches!(self.interaction, Interaction::Dummy)
            || matches!(self.shape, SurfaceShape::CoordinateBreak)
    }
}

#[derive(Clone, Debug)]
pub struct ResolvedSystem {
    pub snapshot: Snapshot,
    pub surfaces: Vec<ResolvedSurface>,
    pub wavelengths_um: Vec<f64>,
    pub wavelength_weights: Vec<f64>,
    pub primary_wavelength: usize,
    pub stop: usize,
    pub object_at_infinity: bool,
    /// Medium of object space (before surface 1).
    pub object_medium: ResolvedMaterial,
    pub n_object: Vec<f64>,
    /// Problems that do not stop computation, such as wavelengths outside a
    /// glass's data range.
    pub warnings: Vec<String>,
    pub library: Arc<MaterialLibrary>,
}

fn placement_frame(p: &Placement) -> Frame {
    let rot = Mat3::rot_x(p.tilt_x.to_radians())
        .mul_mat(&Mat3::rot_y(p.tilt_y.to_radians()))
        .mul_mat(&Mat3::rot_z(p.tilt_z.to_radians()));
    let dec = Vec3::new(p.decenter_x, p.decenter_y, 0.0);
    match p.order {
        TransformOrder::DecenterThenTilt => Frame { origin: dec, rot },
        TransformOrder::TiltThenDecenter => Frame {
            origin: rot.mul_vec(dec),
            rot,
        },
    }
}

fn translate_z(t: f64) -> Frame {
    Frame {
        origin: Vec3::new(0.0, 0.0, t),
        rot: Mat3::IDENTITY,
    }
}

impl ResolvedSystem {
    pub fn new(snapshot: &Snapshot, library: Arc<MaterialLibrary>) -> MkResult<Self> {
        let sys: &OpticalSystem = snapshot.system();
        let env = sys.environment;
        let wls: Vec<f64> = sys.wavelengths.wavelengths.iter().map(|w| w.um).collect();
        let mut warnings = Vec::new();
        let n = sys.surfaces.len();

        let check_range = |m: &ResolvedMaterial, warnings: &mut Vec<String>, what: &str| {
            if let Some((lo, hi)) = m.range_um() {
                for &w in &wls {
                    if w < lo - 1e-9 || w > hi + 1e-9 {
                        warnings.push(format!(
                            "{what}: wavelength {w} µm outside material data range {lo}–{hi} µm"
                        ));
                    }
                }
            }
        };

        let object_medium = library.resolve(&sys.surfaces[0].material, &env)?;
        check_range(&object_medium, &mut warnings, "object space");

        // --- Frames. Surface 1's pre-placement frame is the global origin.
        let mut frames = vec![Frame::IDENTITY; n];
        let mut running = Frame::IDENTITY;
        #[allow(clippy::needless_range_loop)] // `i` also indexes `frames`
        for i in 1..n {
            let s = &sys.surfaces[i];
            let surf_frame = match &s.placement {
                Some(p) if !p.is_identity() => running.compose(&placement_frame(p)),
                _ => running,
            };
            frames[i] = surf_frame;
            let after = match &s.placement {
                Some(p) if p.restore_after => running,
                _ => surf_frame,
            };
            if i < n - 1 {
                if !s.thickness.is_finite() {
                    return Err(MkError::InvalidModel(format!(
                        "surface {i} has non-finite thickness"
                    )));
                }
                running = after.compose(&translate_z(s.thickness));
            }
        }
        let object_at_infinity = sys.object_at_infinity();
        if !object_at_infinity {
            let s0 = &sys.surfaces[0];
            let mut f = translate_z(-s0.thickness);
            if let Some(p) = &s0.placement
                && !p.is_identity()
            {
                f = f.compose(&placement_frame(p));
            }
            frames[0] = f;
        }

        // Entry and returning placements are evaluated as a frame pair: only
        // the inclusive range moves; the downstream baseline is preserved.
        for (i, sf) in sys.surfaces.iter().enumerate() {
            if let Some(e) = &sf.element_placement {
                if e.motion.is_identity() {
                    continue;
                }
                let end = sys.surfaces.iter().position(|s| s.id == e.last).unwrap();
                let pivot = e
                    .pivot
                    .and_then(|id| sys.surfaces.iter().position(|s| s.id == id))
                    .unwrap_or(i);
                let anchor = frames[i];
                let point = frames[pivot].origin;
                let motion = placement_frame(&e.motion);
                let rotation = anchor
                    .rot
                    .mul_mat(&motion.rot)
                    .mul_mat(&anchor.rot.transpose());
                let shift = anchor.rot.mul_vec(motion.origin);
                for f in &mut frames[i..=end] {
                    f.origin = point + rotation.mul_vec(f.origin - point) + shift;
                    f.rot = rotation.mul_mat(&f.rot);
                }
            }
        }
        // --- Media, indices, coatings.
        let mut surfaces = Vec::with_capacity(n);
        let mut medium_before = object_medium.clone();
        for i in 0..n {
            let s = &sys.surfaces[i];
            let inherit =
                i > 0 && (s.interaction != Interaction::Refract || s.is_coordinate_break());
            let medium_after = if i == 0 {
                object_medium.clone()
            } else if inherit {
                medium_before.clone()
            } else {
                let m = library.resolve(&s.material, &env)?;
                check_range(&m, &mut warnings, &format!("surface {i}"));
                m
            };
            let n_after: Vec<f64> = wls.iter().map(|&w| medium_after.index(w)).collect();
            if n_after.iter().any(|v| !(v.is_finite() && *v > 0.0)) {
                return Err(MkError::InvalidModel(format!(
                    "surface {i}: non-physical refractive index"
                )));
            }
            let alpha_after = wls
                .iter()
                .map(|&w| medium_after.absorption_per_mm(w))
                .collect();
            let coating = ResolvedCoating::resolve(
                s.coating.as_ref().and_then(|c| sys.coatings.get(c)),
                &library,
                &env,
            )?;
            let to_next = if i + 1 < n && i > 0 {
                Some(frames[i + 1].relative_to(&frames[i]))
            } else if i == 0 && !object_at_infinity {
                Some(frames[1].relative_to(&frames[0]))
            } else {
                None
            };
            let to_next_is_axial = to_next
                .is_some_and(|f| f.rot.is_identity() && f.origin.x == 0.0 && f.origin.y == 0.0);
            let fixed_semi_diameter = match s.semi_diameter {
                SemiDiameter::Fixed { value } => Some(value),
                SemiDiameter::Auto => None,
            };
            surfaces.push(ResolvedSurface {
                index: i,
                id: s.id,
                shape: s.shape.clone(),
                interaction: if s.is_coordinate_break() {
                    Interaction::Dummy
                } else {
                    s.interaction
                },
                frame: frames[i],
                to_next,
                to_next_is_axial,
                thickness: s.thickness,
                medium_after: medium_after.clone(),
                n_after,
                alpha_after,
                fixed_semi_diameter,
                aperture: s.aperture.clone(),
                coating,
            });
            medium_before = medium_after;
        }

        Ok(ResolvedSystem {
            snapshot: snapshot.clone(),
            n_object: wls.iter().map(|&w| object_medium.index(w)).collect(),
            object_medium,
            wavelength_weights: sys
                .wavelengths
                .wavelengths
                .iter()
                .map(|w| w.weight)
                .collect(),
            primary_wavelength: sys.wavelengths.primary,
            wavelengths_um: wls,
            stop: sys.stop,
            object_at_infinity,
            surfaces,
            warnings,
            library,
        })
    }

    pub fn system(&self) -> &OpticalSystem {
        self.snapshot.system()
    }
    pub fn num_surfaces(&self) -> usize {
        self.surfaces.len()
    }
    pub fn image_index(&self) -> usize {
        self.surfaces.len() - 1
    }
    /// Index of the medium *before* surface `i` at wavelength `w`.
    pub fn n_before(&self, i: usize, w: usize) -> f64 {
        if i == 0 {
            self.n_object[w]
        } else {
            self.surfaces[i - 1].n_after[w]
        }
    }
    /// Index after surface `i` at an arbitrary wavelength (for polychromatic
    /// integration beyond the system wavelengths).
    pub fn n_after_at(&self, i: usize, lambda_um: f64) -> f64 {
        self.surfaces[i].medium_after.index(lambda_um)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{MaterialRef, Surface};

    fn singlet() -> OpticalSystem {
        let mut s = OpticalSystem::new_empty();
        s.surfaces = vec![
            Surface::new(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
            Surface::new(
                1,
                SurfaceShape::sphere(50.0),
                5.0,
                MaterialRef::glass("N-BK7"),
            ),
            Surface::new(2, SurfaceShape::sphere(-50.0), 45.0, MaterialRef::Air),
            Surface::new(3, SurfaceShape::Plane, 0.0, MaterialRef::Air),
        ];
        s.stop = 1;
        s
    }

    #[test]
    fn frames_and_indices() {
        let snap = Snapshot::new(singlet()).unwrap();
        let r = ResolvedSystem::new(&snap, Arc::new(MaterialLibrary::with_builtin())).unwrap();
        assert_eq!(r.surfaces[2].frame.origin.z, 5.0);
        assert_eq!(r.surfaces[3].frame.origin.z, 50.0);
        assert!(r.surfaces[1].to_next_is_axial);
        assert!((r.surfaces[1].n_after[1] - 1.5168).abs() < 2e-5);
        assert_eq!(r.n_before(2, 1), r.surfaces[1].n_after[1]);
    }

    #[test]
    fn restore_after_keeps_axis() {
        let mut s = singlet();
        s.surfaces[1].placement = Some(Placement {
            tilt_x: 5.0,
            restore_after: true,
            ..Default::default()
        });
        let snap = Snapshot::new(s).unwrap();
        let r = ResolvedSystem::new(&snap, Arc::new(MaterialLibrary::with_builtin())).unwrap();
        assert!(!r.surfaces[1].frame.rot.is_identity());
        assert!(r.surfaces[2].frame.rot.is_identity());
        assert!((r.surfaces[2].frame.origin.z - 5.0).abs() < 1e-15);
    }

    #[test]
    fn mirror_inherits_medium() {
        let mut s = singlet();
        s.surfaces[2].interaction = Interaction::Reflect;
        s.surfaces[2].thickness = -45.0;
        let snap = Snapshot::new(s).unwrap();
        let r = ResolvedSystem::new(&snap, Arc::new(MaterialLibrary::with_builtin())).unwrap();
        assert_eq!(r.surfaces[2].n_after, r.surfaces[1].n_after);
    }
}
