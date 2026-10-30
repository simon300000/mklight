//! Energy bookkeeping (T + R + A = 1 shared by primary and ghost paths, the
//! sum over all reflection orders) and reversibility of the backward
//! segments and frame transfers.

mod common;

use std::sync::Arc;

use common::*;
use mklight_core::materials::{GlassCatalog, MaterialLibrary, TransmittancePoint};
use mklight_core::{CoatingDef, Layer, MaterialRef, Placement, Vec3, samples};
use mklight_ghost::{
    Action, Direction, EnumerationSettings, GhostModel, GhostPath, GhostSettings, Step, StepRecord,
    enumerate,
};
use mklight_trace::Ray;

fn absorbing_coating() -> CoatingDef {
    CoatingDef::ThinFilm {
        reference_wavelength_um: D_LINE,
        layers: vec![Layer {
            material: MaterialRef::Constant { n: 1.38 },
            thickness_um: 0.0,
            quarter_waves: Some(1.0),
            extinction: 0.03,
        }],
        mirror_substrate: None,
    }
}

/// Builtin library plus catalog `TEST` with an absorbing copy of N-BK7
/// (internal transmittance 0.8 per 10 mm).
fn absorbing_library() -> MaterialLibrary {
    let mut lib = MaterialLibrary::with_builtin();
    let mut g = (*lib.find_glass(None, "N-BK7").unwrap()).clone();
    g.name = "ABS".into();
    g.transmittance = vec![
        TransmittancePoint {
            um: 0.3,
            tau: 0.8,
            thickness_mm: 10.0,
        },
        TransmittancePoint {
            um: 2.0,
            tau: 0.8,
            thickness_mm: 10.0,
        },
    ];
    let mut cat = GlassCatalog {
        name: "TEST".into(),
        ..Default::default()
    };
    cat.glasses.insert("ABS".into(), g);
    lib.add_catalog(Arc::new(cat));
    lib
}

fn trace_rec(
    m: &GhostModel,
    ray: &Ray,
    path: &GhostPath,
) -> (mklight_ghost::SequenceResult, Vec<StepRecord>) {
    let steps = m.check_path(path).unwrap();
    let mut rec = Vec::new();
    let r = m.trace_steps(ray, 0, &steps, Some(&mut rec));
    (r, rec)
}

/// All alternating reflection sequences over {2, 3} up to `k` reflections,
/// split into detected (even) and returned (odd) paths.
fn plate_paths(k: usize) -> (Vec<GhostPath>, Vec<GhostPath>) {
    let mut even = vec![GhostPath::primary()];
    let mut odd = vec![GhostPath::new(vec![2])];
    let mut seq = Vec::new();
    for i in 0..k {
        seq.push(if i % 2 == 0 { 3 } else { 2 });
        if seq.len() % 2 == 0 {
            even.push(GhostPath::new(seq.clone()));
        } else {
            odd.push(GhostPath::new(seq.clone()));
        }
    }
    (even, odd)
}

#[test]
fn lossless_plate_all_orders_sum_to_one() {
    let n = 1.7;
    let m = model(plate(n, 6.0, 20.0, &[0.0, 25.0], None), unclipped());
    let g = m
        .tracer()
        .generate(m.tracer().field(1), 0, 0.0, 0.3)
        .unwrap();
    let (even, odd) = plate_paths(21);
    let (mut s, mut p) = (0.0, 0.0);
    for path in even.iter().chain(&odd) {
        let (r, _) = trace_rec(&m, &g.ray, path);
        assert!(r.status.is_ok(), "{path:?}: {:?}", r.status);
        s += r.weight_s;
        p += r.weight_p;
    }
    // Rs + Ts²/(1 − Rs) = 1; the tail beyond 21 reflections is ~R^22.
    assert!((s - 1.0).abs() < 1e-12, "s: {s}");
    assert!((p - 1.0).abs() < 1e-12, "p: {p}");
}

#[test]
fn absorbing_plate_bookkeeping() {
    // Absorbing glass (bulk) and an absorbing coating (interfaces) at 20°.
    let mut sys = plate(1.5, 10.0, 20.0, &[0.0, 20.0], Some(absorbing_coating()));
    sys.surfaces[2].material = MaterialRef::Catalog {
        catalog: Some("TEST".into()),
        name: "ABS".into(),
    };
    let m = GhostModel::from_system(resolve_with(sys, absorbing_library()), unclipped()).unwrap();
    let ray = m
        .tracer()
        .generate(m.tracer().field(1), 0, 0.0, 0.0)
        .unwrap()
        .ray;

    let (prim, rp) = trace_rec(&m, &ray, &GhostPath::primary());
    let (_, rg) = trace_rec(&m, &ray, &GhostPath::new(vec![3, 2]));
    let (_, rf) = trace_rec(&m, &ray, &GhostPath::new(vec![2]));
    let tau = prim.internal;
    assert!(tau < 0.999 && tau > 0.5);
    // Splits: F = front from air, B = rear from glass, G = front from glass.
    let split = |r: &[StepRecord], surf: u32, dir: Direction| {
        r.iter()
            .find(|x| x.step.surface == surf && x.step.direction == dir)
            .unwrap()
            .split
    };
    let f = split(&rp, 2, Direction::Forward);
    let b = split(&rp, 3, Direction::Forward);
    let g = split(&rg, 2, Direction::Backward);
    assert_eq!(f, split(&rf, 2, Direction::Forward));
    assert_eq!(
        b,
        split(&rg, 3, Direction::Forward),
        "primary and ghost share the split"
    );
    for (k, x) in [f, b, g].iter().enumerate() {
        let (a_s, a_p) = (1.0 - x.rs - x.ts, 1.0 - x.rp - x.tp);
        assert!(a_s > 1e-4 && a_p > 1e-4, "interface {k} absorbs: {x:?}");
    }

    let (even, odd) = plate_paths(9);
    for pol in 0..2 {
        let pick = |x: &mklight_core::materials::coating::InterfaceRT| {
            if pol == 0 { (x.rs, x.ts) } else { (x.rp, x.tp) }
        };
        let ((rf_, tf), (rb, tb), (rg_, tg)) = (pick(&f), pick(&b), pick(&g));
        let (af, ab, ag) = (1.0 - rf_ - tf, 1.0 - rb - tb, 1.0 - rg_ - tg);
        // Geometric series over the internal round trips.
        let x = tf * tau;
        let l = rb * tau * rg_ * tau;
        let detected = x * tb / (1.0 - l);
        let returned = rf_ + x * rb * tau * tg / (1.0 - l);
        let a_int = af + x * (ab + rb * tau * ag) / (1.0 - l);
        let a_bulk = (1.0 - tau) * tf * (1.0 + tau * rb) / (1.0 - l);
        assert!((detected + returned + a_int + a_bulk - 1.0).abs() < 1e-14);

        let w = |paths: &[GhostPath]| -> f64 {
            paths
                .iter()
                .map(|path| {
                    let (r, _) = trace_rec(&m, &ray, path);
                    assert!(r.status.is_ok());
                    let ch = if pol == 0 { r.weight_s } else { r.weight_p };
                    ch * r.internal
                })
                .sum()
        };
        let (d_sum, r_sum) = (w(&even), w(&odd));
        assert!((d_sum - detected).abs() < 1e-14, "{d_sum} vs {detected}");
        assert!((r_sum - returned).abs() < 1e-14, "{r_sum} vs {returned}");
        // Everything that leaves is at most the input.
        assert!(d_sum + r_sum < 1.0 - a_int - a_bulk + 1e-14);
    }
}

#[test]
fn primary_and_ghost_share_t_r_a_on_double_gauss() {
    // Same ray prefix → same incidence → one coating evaluation: the
    // primary's T and the ghost's R at an interface sum (with A) to 1.
    for coated in [false, true] {
        let mut sys = samples::double_gauss();
        if coated {
            sys.coatings.insert("ABS".into(), absorbing_coating());
            for s in sys.surfaces.iter_mut().skip(1) {
                s.coating = Some("ABS".into());
            }
        }
        let m = model(sys, GhostSettings::default());
        let wl = m.system().primary_wavelength;
        let field = m.tracer().field(2);
        let ray = m.tracer().generate(field, wl, 0.2, 0.3).unwrap().ray;
        let steps = m.check_path(&GhostPath::primary()).unwrap();
        let mut prim = Vec::new();
        assert!(
            m.trace_steps(&ray, wl, &steps, Some(&mut prim))
                .status
                .is_ok()
        );
        let paths = enumerate(&m, &EnumerationSettings::default()).unwrap();
        let mut checked = 0;
        for e in &paths.paths {
            let i = e.path.reflections[0];
            let steps = m.check_path(&e.path).unwrap();
            let mut rec = Vec::new();
            m.trace_steps(&ray, wl, &steps, Some(&mut rec));
            let Some(gr) = rec.iter().find(|r| r.step.action == Action::Reflect) else {
                continue;
            };
            let pr = prim.iter().find(|r| r.step.surface == i).unwrap();
            assert_eq!(gr.point, pr.point);
            assert_eq!(gr.split, pr.split);
            assert_eq!(gr.applied, [pr.split.rs, pr.split.rp]);
            assert_eq!(pr.applied, [pr.split.ts, pr.split.tp]);
            let (a_s, a_p) = (
                1.0 - gr.applied[0] - pr.applied[0],
                1.0 - gr.applied[1] - pr.applied[1],
            );
            if coated {
                assert!(a_s > 0.0 && a_p > 0.0 && a_s < 0.2 && a_p < 0.2);
            } else {
                assert!(a_s.abs() < 1e-15 && a_p.abs() < 1e-15, "bare: {a_s} {a_p}");
            }
            checked += 1;
        }
        assert!(checked >= 30, "{checked}");
    }
}

/// Steps that retrace a recorded forward segment in reverse.
fn reversed(rec: &[StepRecord]) -> Vec<Step> {
    rec.iter()
        .rev()
        .skip(1)
        .map(|r| Step {
            surface: r.step.surface,
            direction: r.step.direction.reversed(),
            action: Action::Pass,
        })
        .collect()
}

fn check_reverse_primary(m: &GhostModel, ray: &Ray, wl: usize, tol: f64) {
    let steps = m.check_path(&GhostPath::primary()).unwrap();
    let mut fwd = Vec::new();
    let r = m.trace_steps(ray, wl, &steps, Some(&mut fwd));
    assert!(r.status.is_ok());
    let img = fwd.last().unwrap();
    let back = Ray::new(img.point, -img.dir_in);
    let mut bwd = Vec::new();
    let image = m.system().image_index();
    let rb = m.trace_from(image, &back, wl, &reversed(&fwd), Some(&mut bwd));
    assert!(rb.status.is_ok(), "{:?}", rb.status);
    for (b, f) in bwd.iter().zip(fwd.iter().rev().skip(1)) {
        assert_eq!(b.step.surface, f.step.surface);
        assert!(
            b.point.max_abs_diff(f.point) < tol,
            "surface {}: {:?} vs {:?}",
            f.step.surface,
            b.point,
            f.point
        );
        assert!(b.dir_out.max_abs_diff(-f.dir_in) < tol);
        // Reciprocity of the split.
        assert!((b.applied[0] - f.applied[0]).abs() < 1e-12);
    }
    // Weights of the two directions agree (Fresnel reciprocity).
    assert!((rb.weight_s - r.weight_s).abs() < 1e-12);
}

#[test]
fn backward_traversal_retraces_forward_rays() {
    let m = model(samples::double_gauss(), GhostSettings::default());
    let wl = m.system().primary_wavelength;
    for f in 0..3 {
        let ray = m
            .tracer()
            .generate(m.tracer().field(f), wl, -0.4, 0.6)
            .unwrap()
            .ray;
        check_reverse_primary(&m, &ray, wl, 1e-10);
    }
    // Tilted and decentered elements: non-axial transfers in both directions.
    let mut sys = plate(1.6, 5.0, 15.0, &[0.0, 5.0], None);
    sys.surfaces[2].placement = Some(Placement {
        tilt_x: 4.0,
        decenter_y: 0.5,
        ..Default::default()
    });
    sys.surfaces[3].placement = Some(Placement {
        tilt_y: -3.0,
        decenter_x: 0.2,
        ..Default::default()
    });
    let m = model(sys, unclipped());
    assert!(!m.system().surfaces[2].to_next_is_axial);
    let ray = m
        .tracer()
        .generate(m.tracer().field(1), 0, 0.3, -0.5)
        .unwrap()
        .ray;
    check_reverse_primary(&m, &ray, 0, 1e-12);
    // transfer_back ∘ transfer_into is the identity at every gap.
    let r0 = Ray::new(
        Vec3::new(1.2, -0.7, 0.3),
        Vec3::new(0.1, 0.2, 1.0).normalized(),
    );
    for i in 2..m.system().num_surfaces() {
        let fwd = mklight_trace::trace::transfer_into(m.system(), i, r0);
        let back = m.transfer_back(i - 1, fwd);
        assert!(back.pos.max_abs_diff(r0.pos) < 1e-14 && back.dir.max_abs_diff(r0.dir) < 1e-15);
    }
}

#[test]
fn ghost_backward_segment_is_reversible() {
    // Reverse the ghost ray just before its second reflection and trace
    // forward: it retraces the backward segment up to the first reflection.
    let m = model(samples::double_gauss(), unclipped());
    let wl = m.system().primary_wavelength;
    let ray = m
        .tracer()
        .generate(m.tracer().field(0), wl, 0.1, 0.2)
        .unwrap()
        .ray;
    let path = GhostPath::new(vec![10, 3]);
    let steps = m.check_path(&path).unwrap();
    let mut rec = Vec::new();
    let r = m.trace_steps(&ray, wl, &steps, Some(&mut rec));
    assert!(rec.len() > 12, "{:?}", r.status);
    let k1 = rec
        .iter()
        .position(|x| x.step.action == Action::Reflect)
        .unwrap();
    let k2 = k1
        + 1
        + rec[k1 + 1..]
            .iter()
            .position(|x| x.step.action == Action::Reflect)
            .unwrap();
    let at = rec[k2].step.surface as usize;
    let back = Ray::new(rec[k2].point, -rec[k2].dir_in);
    let fwd_steps: Vec<Step> = rec[k1..k2]
        .iter()
        .rev()
        .map(|x| Step {
            surface: x.step.surface,
            direction: Direction::Forward,
            action: Action::Pass,
        })
        .collect();
    let mut out = Vec::new();
    let rr = m.trace_from(at, &back, wl, &fwd_steps, Some(&mut out));
    assert!(rr.status.is_ok(), "{:?}", rr.status);
    for (o, x) in out.iter().zip(rec[k1..k2].iter().rev()) {
        assert!(
            o.point.max_abs_diff(x.point) < 1e-10,
            "{:?} {:?}",
            o.point,
            x.point
        );
    }
    // At the first reflection the reversed ray arrives opposite to the
    // reflected direction.
    assert!(out.last().unwrap().dir_in.max_abs_diff(-rec[k1].dir_out) < 1e-12);
}

#[test]
fn sensor_reflection_weight() {
    let rs_sensor = 0.2;
    let settings = GhostSettings {
        sensor_reflectance: Some(rs_sensor),
        ..unclipped()
    };
    let n = 1.5;
    let m = model(plate(n, 5.0, 10.0, &[0.0], None), settings);
    let e = enumerate(&m, &EnumerationSettings::default()).unwrap();
    let labels: Vec<_> = e.paths.iter().map(|p| p.path.reflections.clone()).collect();
    assert_eq!(labels, vec![vec![3, 2], vec![4, 2], vec![4, 3]]);
    let r = r_normal(1.0, n);
    let ray = m
        .tracer()
        .generate(m.tracer().field(0), 0, 0.0, 0.0)
        .unwrap()
        .ray;
    let (res, _) = trace_rec(&m, &ray, &GhostPath::new(vec![4, 3]));
    // Through the plate, sensor, back to the rear face (air side), sensor.
    let expect = (1.0 - r) * (1.0 - r) * rs_sensor * r;
    assert!((res.weight() - expect).abs() < 1e-15);
    assert!((e.paths[2].reflectance_product - rs_sensor * r).abs() < 1e-15);
}
