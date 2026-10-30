//! Layout scene geometry in global coordinates.

mod common;

use common::*;
use mklight_analysis::layout::*;
use mklight_core::{SemiDiameter, Vec3, samples};

fn close(a: [f64; 3], b: Vec3, tol: f64) -> bool {
    (a[0] - b.x).abs() < tol && (a[1] - b.y).abs() < tol && (a[2] - b.z).abs() < tol
}

#[test]
fn singlet_scene_geometry() {
    let t = tracer(samples::singlet());
    let set = LayoutSettings::default();
    let s = layout(&t, &set).unwrap();
    let sys = t.system();
    assert_eq!(s.elements.len(), 1);
    let e = &s.elements[0];
    assert_eq!((e.first_surface, e.last_surface), (1, 2));
    assert_eq!(e.outline.first(), e.outline.last());
    for (f, b) in e.front.iter().zip(&e.back) {
        assert!((f[1] - b[1]).abs() < 1e-12, "matching heights");
        assert!(f[2] <= b[2] + 1e-12, "front before back");
    }
    // Profiles: (0, ±sd, z_vertex + sag(sd)) in global coordinates.
    for info in &s.surfaces {
        let j = info.index;
        let prof = s
            .polylines
            .iter()
            .find(|p| p.surface == Some(j) && matches!(p.kind, PolylineKind::SurfaceProfile | PolylineKind::ImagePlane) && p.in_section)
            .unwrap();
        let sd = info.semi_diameter;
        let z = sys.surfaces[j].frame.origin.z + sys.surfaces[j].shape.sag(sd * sd).unwrap();
        assert!(close(*prof.points.last().unwrap(), Vec3::new(0.0, sd, z), 1e-12));
        assert!(close(prof.points[0], Vec3::new(0.0, -sd, z), 1e-12));
        assert!(e.mechanical_semi_diameter >= sd || j == 3);
    }
    // Rays: tangential + sagittal fans per field; each ends at its image hit.
    let rays: Vec<_> = s.polylines.iter().filter(|p| p.kind == PolylineKind::Ray).collect();
    assert_eq!(rays.len(), 3 * (7 + 6));
    for r in &rays {
        assert!(!r.vignetted);
        assert_eq!(r.points.len(), 1 + 3);
        let last = *r.points.last().unwrap();
        assert!((last[2] - sys.surfaces[3].frame.origin.z).abs() < 1e-9);
        // Meridional rays are in section, sagittal ones are not (off axis).
        let tangential = r.points.iter().all(|p| p[0] == 0.0);
        assert_eq!(r.in_section, tangential);
    }
    let chief = t.chief(t.field(2), 1);
    assert!(rays.iter().any(|r| r.field == Some(2) && close(*r.points.last().unwrap(), chief.ray.pos + Vec3::new(0.0, 0.0, sys.surfaces[3].frame.origin.z), 1e-9)));
    // Every point is finite and inside the bounds.
    for p in s.polylines.iter().flat_map(|p| &p.points) {
        for k in 0..3 {
            assert!(p[k].is_finite() && p[k] >= s.bounds.min[k] && p[k] <= s.bounds.max[k]);
        }
    }
    let j = serde_json::to_value(&s).unwrap();
    assert_eq!(j["polylines"][0]["kind"], "surface_profile");
    assert!(j["surfaces"][0]["semi_diameter_is_auto"].is_boolean());
}

#[test]
fn cemented_lenses_form_shared_elements() {
    let t = tracer(samples::double_gauss());
    let s = layout(&t, &LayoutSettings::default()).unwrap();
    let pairs: Vec<_> = s.elements.iter().map(|e| (e.first_surface, e.last_surface)).collect();
    assert_eq!(pairs, vec![(1, 2), (3, 4), (4, 5), (7, 8), (8, 9), (10, 11)]);
    // The pure-aperture stop has markers but no profile.
    assert!(s.polylines.iter().any(|p| p.kind == PolylineKind::Stop && p.surface == Some(6)));
    assert!(!s.polylines.iter().any(|p| p.kind == PolylineKind::SurfaceProfile && p.surface == Some(6)));
}

#[test]
fn folded_newtonian_is_in_global_coordinates() {
    let t = tracer(samples::newtonian_telescope());
    let s = layout(&t, &LayoutSettings::default()).unwrap();
    let sys = t.system();
    let img = sys.image_index();
    let o = sys.surfaces[img].frame.origin;
    // The fold sends the beam along −y: the image is 100 mm below the axis.
    assert!((o.y + 100.0).abs() < 1e-9 && (o.z + 400.0).abs() < 1e-9, "{o:?}");
    assert!(s.surfaces.iter().any(|i| i.index == 1 && i.is_mirror));
    for r in s.polylines.iter().filter(|p| p.kind == PolylineKind::Ray && p.in_section) {
        let last = *r.points.last().unwrap();
        assert!((last[1] - o.y).abs() < 1e-9, "ray ends on the image plane");
    }
    let axis = s.polylines.iter().find(|p| p.kind == PolylineKind::Axis).unwrap();
    assert!(close(*axis.points.last().unwrap(), o, 1e-12));
    assert!(s.elements.is_empty());
}

#[test]
fn vignetted_rays_end_at_the_clipping_surface() {
    let mut sys = ideal_lens(100.0, 20.0, &[0.0]);
    sys.surfaces[1].semi_diameter = SemiDiameter::Fixed { value: 8.0 };
    let t = tracer(sys);
    let s = layout(&t, &LayoutSettings {
        include_3d: false,
        rays_per_field: 5,
        lead_in: Some(30.0),
        ..Default::default()
    })
    .unwrap();
    let rays: Vec<_> = s.polylines.iter().filter(|p| p.kind == PolylineKind::Ray).collect();
    assert_eq!(rays.len(), 5);
    assert_eq!(rays.iter().filter(|r| r.vignetted).count(), 2);
    for r in &rays {
        assert!((r.points[0][2] + 30.0).abs() < 1e-12, "lead-in");
        if r.vignetted {
            assert_eq!(r.points.len(), 2);
            assert!(r.points[1][2].abs() < 1e-12 && (r.points[1][1].abs() - 10.0).abs() < 1e-12);
        } else {
            assert_eq!(r.points.len(), 3);
        }
    }
}

#[test]
fn compact_buffers_round_trip() {
    let t = tracer(samples::cooke_triplet());
    let s = layout(&t, &LayoutSettings::default()).unwrap();
    let b = s.to_buffers();
    let np = s.polylines.len() + s.elements.len();
    assert_eq!(b.polyline_count(), np);
    assert_eq!(b.offsets.len(), np + 1);
    let nv: usize = s.polylines.iter().map(|p| p.points.len()).sum::<usize>()
        + s.elements.iter().map(|e| e.outline.len()).sum::<usize>();
    assert_eq!(*b.offsets.last().unwrap() as usize, nv);
    assert_eq!(b.vertices.len(), 3 * nv);
    for (i, p) in s.polylines.iter().enumerate() {
        let o = b.offsets[i] as usize;
        for (k, q) in p.points.iter().enumerate() {
            for c in 0..3 {
                assert_eq!(b.vertices_f64[3 * (o + k) + c], q[c]);
                let rel = b.vertices[3 * (o + k) + c] as f64 + b.origin[c];
                assert!((rel - q[c]).abs() < 1e-5, "f32 relative to the centre");
            }
        }
        assert_eq!(b.kinds[i], p.kind.code());
        assert_eq!(b.flags[i] & 1 == 1, p.in_section);
    }
    assert_eq!(b.kinds[np - 1], ELEMENT_OUTLINE_KIND);
    let bytes = b.to_le_bytes();
    assert_eq!(&bytes[..4], b"MKLL");
    assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize, np);
    assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize, nv);
    let head = 40 + 4 * (np + 1) + 12 * np + (2 * np).div_ceil(4) * 4;
    assert_eq!(bytes.len(), head + 12 * nv);
}
