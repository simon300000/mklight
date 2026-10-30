//! Prints first-order data, ray-failure counts and focus diagnostics for the
//! built-in sample systems: `cargo run -p mklight-trace --example samples_report`.

use std::sync::Arc;

use mklight_core::materials::MaterialLibrary;
use mklight_core::{OpticalSystem, ResolvedSystem, Snapshot, samples};
use mklight_trace::{PupilSampling, SequentialTracer, TraceOptions};

fn tracer(sys: &OpticalSystem) -> SequentialTracer {
    let snap = Snapshot::new(sys.clone()).expect("valid");
    let r = ResolvedSystem::new(&snap, Arc::new(MaterialLibrary::with_builtin())).expect("resolve");
    SequentialTracer::new(Arc::new(r)).expect("tracer")
}

/// Polychromatic RMS spot radius over all fields (field-weighted mean), about
/// each field's polychromatic centroid.
fn rms(sys: &OpticalSystem) -> Option<f64> {
    let t = tracer(sys);
    let pts = PupilSampling::Hexapolar { rings: 8 }.points();
    let opts = TraceOptions::default();
    let mut total = 0.0;
    for f in t.fields() {
        let mut xy = Vec::new();
        for w in 0..t.num_wavelengths() {
            for p in &pts {
                let r = t.trace(f, w, p[0], p[1], &opts);
                if !r.status.is_ok() {
                    return None;
                }
                xy.push((r.ray.pos.x, r.ray.pos.y));
            }
        }
        let n = xy.len() as f64;
        let cx = xy.iter().map(|p| p.0).sum::<f64>() / n;
        let cy = xy.iter().map(|p| p.1).sum::<f64>() / n;
        let v = xy
            .iter()
            .map(|p| (p.0 - cx).powi(2) + (p.1 - cy).powi(2))
            .sum::<f64>()
            / n;
        total += v.sqrt();
    }
    Some(total / t.fields().len() as f64)
}

fn best_focus(sys: &OpticalSystem) -> (f64, f64) {
    let last = sys.surfaces.len() - 2;
    let t0 = sys.surfaces[last].thickness;
    let eval = |t: f64| {
        let mut s = sys.clone();
        s.surfaces[last].thickness = t;
        rms(&s).unwrap_or(f64::INFINITY)
    };
    let (mut a, mut b) = (
        t0 - 2.0 * t0.abs().max(1.0) * 0.02,
        t0 + 2.0 * t0.abs().max(1.0) * 0.02,
    );
    let g = 0.5 * (5f64.sqrt() - 1.0);
    for _ in 0..60 {
        let c = b - g * (b - a);
        let d = a + g * (b - a);
        if eval(c) < eval(d) {
            b = d;
        } else {
            a = c;
        }
    }
    let t = 0.5 * (a + b);
    (t, eval(t))
}

fn main() {
    for (name, sys) in samples::all() {
        let t = tracer(&sys);
        let fo = t.paraxial().primary();
        let opts = TraceOptions::default();
        let mut fails = 0;
        let pts = PupilSampling::Hexapolar { rings: 6 }.points();
        for f in t.fields() {
            for w in 0..t.num_wavelengths() {
                for p in &pts {
                    if !t.trace(f, w, p[0], p[1], &opts).status.is_ok() {
                        fails += 1;
                    }
                }
            }
        }
        let last = sys.surfaces.len() - 2;
        println!(
            "{name:16} EFL {:10.5} BFL {:10.5} img.dist {:10.5} (current {:10.5}) EPD {:8.4} F/# {:6.3} \
             WF/# {:6.3} m {:8.5} ffl {:10.5} EP {:9.4} XP {:9.4} h' {:8.4} fails {fails} rms {:?}",
            fo.efl,
            fo.bfl,
            fo.image_distance,
            sys.surfaces[last].thickness,
            fo.entrance_pupil_diameter,
            fo.image_f_number,
            fo.working_f_number,
            fo.magnification,
            fo.ffl,
            fo.entrance_pupil_position,
            fo.exit_pupil_position,
            fo.paraxial_image_height,
            rms(&sys)
        );
        if std::env::args().any(|a| a == "--focus") {
            let (bt, br) = best_focus(&sys);
            println!("    best focus thickness {bt:.6} rms {br:.6}");
        }
    }
}
