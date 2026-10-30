//! `mklight trace` (one real ray) and `mklight paraxial` (first-order
//! table), computed by the CPU f64 reference (`mklight-trace`) on the
//! session's cached `ResolvedSystem`.

use serde_json::{Value, json};
use std::sync::Arc;

use mklight_core::{ResolvedSystem, Vec3};
use mklight_engine::Session;
use mklight_trace::trace::to_global;
use mklight_trace::{
    FirstOrder, ParaxialRayTable, ParaxialXY, Ray, SequentialTracer, SurfaceRecord, TraceOptions,
    TraceStatus,
};

use super::project::shape_type;
use super::{Ctx, Report, warning_lines};
use crate::cli::{ParaxialArgs, TraceArgs};
use crate::error::{CliError, CliResult};
use crate::output::{Align, Table, fmt_f, fmt_g, key_values, num};

fn tracer_for(session: &Session) -> CliResult<SequentialTracer> {
    let resolved: Arc<ResolvedSystem> = session.resolved()?;
    Ok(SequentialTracer::new(resolved)?)
}

fn v3(v: Vec3) -> Value {
    json!([num(v.x), num(v.y), num(v.z)])
}

/// Parse `px,py`.
pub fn parse_pupil(s: &str) -> CliResult<[f64; 2]> {
    let parts: Vec<&str> = s.split(',').map(str::trim).collect();
    let bad = || CliError::usage(format!("invalid --pupil '{s}' (expected px,py)"));
    if parts.len() != 2 {
        return Err(bad());
    }
    let px: f64 = parts[0].parse().map_err(|_| bad())?;
    let py: f64 = parts[1].parse().map_err(|_| bad())?;
    if !(px.is_finite() && py.is_finite()) {
        return Err(bad());
    }
    // The normalised pupil is the unit disk (a little slack for decimal
    // input such as 0.7071,0.7071).
    if px.hypot(py) > 1.0 + 1e-6 {
        return Err(CliError::usage(format!(
            "invalid --pupil '{s}': the point lies outside the unit pupil (|p| = {} > 1)",
            fmt_g(px.hypot(py), 6)
        )));
    }
    Ok([px, py])
}

pub(crate) fn status_parts(s: &TraceStatus) -> (&'static str, Option<usize>, Value) {
    match *s {
        TraceStatus::Ok => ("ok", None, Value::Null),
        TraceStatus::Miss { surface } => ("miss", Some(surface), Value::Null),
        TraceStatus::Clipped { surface } => ("clipped", Some(surface), Value::Null),
        TraceStatus::TotalInternalReflection { surface } => {
            ("total_internal_reflection", Some(surface), Value::Null)
        }
        TraceStatus::Failed { surface, reason } => (
            "failed",
            Some(surface),
            serde_json::to_value(reason).unwrap_or(Value::Null),
        ),
    }
}

/// Angle of incidence (degrees) from its cosine.
fn aoi_deg(cos_incidence: f64) -> f64 {
    cos_incidence.clamp(-1.0, 1.0).acos().to_degrees()
}

pub fn trace(ctx: &Ctx, a: &TraceArgs) -> CliResult<Report> {
    let pupil = parse_pupil(&a.pupil)?;
    let session = ctx.open(&a.project)?;
    let tracer = tracer_for(&session)?;
    let sys = tracer.system();
    let nf = tracer.fields().len();
    let nw = tracer.num_wavelengths();
    if a.field >= nf {
        return Err(CliError::usage(format!(
            "--field {} out of range (the system has {nf} field(s), indices 0..{})",
            a.field,
            nf - 1
        )));
    }
    let wl = a.wavelength.unwrap_or(sys.primary_wavelength);
    if wl >= nw {
        return Err(CliError::usage(format!(
            "--wavelength {wl} out of range (the system has {nw} wavelength(s), indices 0..{})",
            nw - 1
        )));
    }
    let opts = if a.no_clip {
        TraceOptions::unclipped()
    } else {
        tracer.options()
    };
    let field = tracer.field(a.field);
    let mut rec: Vec<SurfaceRecord> = Vec::new();
    let r = tracer.trace_with(field, wl, pupil[0], pupil[1], &opts, Some(&mut rec));
    let chief = tracer.chief(field, wl);
    let (status, stopped_at, reason) = status_parts(&r.status);
    let ok = r.status.is_ok();
    let image = ok.then_some((r.ray.pos, r.ray.dir));
    let chief_point = chief.status.is_ok().then_some(chief.ray.pos);
    let aberration = match (image, chief_point) {
        (Some((p, _)), Some(c)) => Some([p.x - c.x, p.y - c.y]),
        _ => None,
    };
    let model = &sys.system().surfaces;
    let label = |i: usize| model.get(i).map(|s| s.label.clone()).unwrap_or_default();
    // Surface 0 (the starting point) and coordinate breaks have no
    // intersection and no incidence: their AOI and normal are not reported.
    let has_hit = |i: usize| i > 0 && model.get(i).is_some_and(|s| !s.is_coordinate_break());

    let surfaces: Vec<Value> = rec
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let g = to_global(sys, i, &Ray::new(s.point, s.dir));
            let hit = has_hit(i);
            json!({
                "index": i,
                "label": label(i),
                "type": model.get(i).map_or("", |s| shape_type(&s.shape)),
                "is_coordinate_break": model.get(i).is_some_and(|s| s.is_coordinate_break()),
                "point": v3(s.point),
                "dir": v3(s.dir),
                "normal": if hit { v3(s.normal) } else { Value::Null },
                "aoi_deg": if hit { num(aoi_deg(s.cos_incidence)) } else { Value::Null },
                "opl_mm": num(s.opl),
                "transmittance": num(s.transmittance),
                "global_point": v3(g.pos),
                "global_dir": v3(g.dir),
            })
        })
        .collect();

    let mut json = json!({
        "path": a.project.display().to_string(),
        "field": {"index": a.field, "hx": field.point.hx, "hy": field.point.hy},
        "wavelength": {"index": wl, "um": sys.wavelengths_um[wl]},
        "pupil": pupil,
        "clip": !a.no_clip,
        "status": status,
        "stopped_at_surface": stopped_at,
        "fail_reason": reason,
        "image": image.map_or(Value::Null, |(p, d)| json!({"point": v3(p), "dir": v3(d)})),
        "opl_mm": if ok { num(r.opl.value()) } else { Value::Null },
        "transmittance": if ok { num(r.transmittance) } else { Value::Null },
        "chief_image_point": chief_point.map_or(Value::Null, v3),
        "transverse_aberration": aberration.map_or(Value::Null, |e| json!([num(e[0]), num(e[1])])),
        "warnings": tracer.warnings(),
    });
    if a.surfaces {
        json["surfaces"] = json!(surfaces);
    }

    let mut kv = vec![
        (
            "field",
            format!(
                "{} (Hx {}, Hy {})",
                a.field,
                fmt_g(field.point.hx, 6),
                fmt_g(field.point.hy, 6)
            ),
        ),
        (
            "wavelength",
            format!("{wl} ({} µm)", fmt_f(sys.wavelengths_um[wl], 7)),
        ),
        (
            "pupil",
            format!("({}, {})", fmt_g(pupil[0], 6), fmt_g(pupil[1], 6)),
        ),
        (
            "status",
            match stopped_at {
                None => status.to_string(),
                Some(s) => format!("{status} at surface {s}"),
            },
        ),
    ];
    if let Some((p, d)) = image {
        kv.push((
            "image point",
            format!(
                "x {}  y {}  z {} mm",
                fmt_f(p.x, 6),
                fmt_f(p.y, 6),
                fmt_f(p.z, 6)
            ),
        ));
        kv.push((
            "image dir",
            format!(
                "L {}  M {}  N {}",
                fmt_f(d.x, 8),
                fmt_f(d.y, 8),
                fmt_f(d.z, 8)
            ),
        ));
        kv.push(("OPL", format!("{} mm", fmt_f(r.opl.value(), 6))));
    }
    if let Some(e) = aberration {
        kv.push((
            "ray - chief",
            format!("ex {}  ey {} mm", fmt_f(e[0], 6), fmt_f(e[1], 6)),
        ));
    }
    let mut text = key_values(&kv);
    if a.surfaces {
        let mut t = Table::new(&[
            ("#", Align::Right),
            ("label", Align::Left),
            ("x", Align::Right),
            ("y", Align::Right),
            ("z", Align::Right),
            ("L", Align::Right),
            ("M", Align::Right),
            ("N", Align::Right),
            ("AOI°", Align::Right),
            ("OPL", Align::Right),
        ]);
        for (i, s) in rec.iter().enumerate() {
            t.row(vec![
                i.to_string(),
                label(i),
                fmt_f(s.point.x, 6),
                fmt_f(s.point.y, 6),
                fmt_f(s.point.z, 6),
                fmt_f(s.dir.x, 8),
                fmt_f(s.dir.y, 8),
                fmt_f(s.dir.z, 8),
                if has_hit(i) {
                    fmt_f(aoi_deg(s.cos_incidence), 4)
                } else {
                    "-".into()
                },
                fmt_f(s.opl, 6),
            ]);
        }
        text.push('\n');
        text.push_str(&t.render());
        text.push_str(
            "(coordinates in each surface's local frame, mm; surface 0: the start point)\n",
        );
        if model.iter().any(|s| s.is_coordinate_break()) {
            text.push_str(
                "(coordinate breaks, AOI '-': no intersection, the point is the incoming ray \
                 re-expressed in the break's frame)\n",
            );
        }
    }
    text.push_str(&warning_lines(tracer.warnings()));
    Ok(Report::ok(json, text))
}

// ---------------------------------------------------------------------------
// paraxial

/// The scalar first-order quantities, in table order: (key, label, unit).
const QUANTITIES: &[(&str, &str, &str)] = &[
    ("efl", "effective focal length", "mm"),
    ("bfl", "back focal length", "mm"),
    ("ffl", "front focal length", "mm"),
    ("image_distance", "paraxial image distance", "mm"),
    ("entrance_pupil_diameter", "entrance pupil diameter", "mm"),
    ("entrance_pupil_position", "entrance pupil position", "mm"),
    ("exit_pupil_diameter", "exit pupil diameter", "mm"),
    ("exit_pupil_position", "exit pupil position", "mm"),
    ("image_f_number", "image-space F/#", ""),
    ("working_f_number", "working F/#", ""),
    ("object_na", "object-space NA", ""),
    ("image_na", "image-space NA", ""),
    ("magnification", "paraxial magnification", ""),
    ("lagrange_invariant", "Lagrange invariant", "mm"),
    ("total_track", "total track", "mm"),
    ("stop_semi_diameter", "stop semi-diameter", "mm"),
    ("paraxial_image_height", "paraxial image height", "mm"),
];

fn quantity(f: &FirstOrder, key: &str) -> f64 {
    match key {
        "efl" => f.efl,
        "bfl" => f.bfl,
        "ffl" => f.ffl,
        "image_distance" => f.image_distance,
        "entrance_pupil_diameter" => f.entrance_pupil_diameter,
        "entrance_pupil_position" => f.entrance_pupil_position,
        "exit_pupil_diameter" => f.exit_pupil_diameter,
        "exit_pupil_position" => f.exit_pupil_position,
        "image_f_number" => f.image_f_number,
        "working_f_number" => f.working_f_number,
        "object_na" => f.object_na,
        "image_na" => f.image_na,
        "magnification" => f.magnification,
        "lagrange_invariant" => f.lagrange_invariant,
        "total_track" => f.total_track,
        "stop_semi_diameter" => f.stop_semi_diameter,
        "paraxial_image_height" => f.paraxial_image_height,
        _ => f64::NAN,
    }
}

fn first_order_json(f: &FirstOrder) -> Value {
    let mut m = serde_json::Map::new();
    m.insert("wavelength_um".into(), num(f.wavelength_um));
    for (k, _, _) in QUANTITIES {
        m.insert((*k).into(), num(quantity(f, k)));
    }
    m.insert(
        "image_height_per_tan".into(),
        f.image_height_per_tan.map_or(Value::Null, num),
    );
    m.insert(
        "image_height_per_object_height".into(),
        f.image_height_per_object_height.map_or(Value::Null, num),
    );
    Value::Object(m)
}

fn ray_json(t: &ParaxialRayTable) -> Value {
    let a = |v: &[f64]| v.iter().map(|x| num(*x)).collect::<Vec<_>>();
    json!({"y": a(&t.y), "u": a(&t.u), "nu": a(&t.nu)})
}

pub fn paraxial(ctx: &Ctx, a: &ParaxialArgs) -> CliResult<Report> {
    let session = ctx.open(&a.project)?;
    let tracer = tracer_for(&session)?;
    let p = tracer.paraxial();
    let xy = ParaxialXY::new(tracer.system())?;
    let primary = p.primary();
    let json = json!({
        "path": a.project.display().to_string(),
        "primary": p.primary,
        "wavelengths": p.wavelengths.iter().map(first_order_json).collect::<Vec<_>>(),
        "marginal_ray": ray_json(&primary.marginal),
        "chief_ray": ray_json(&primary.chief),
        "warnings": p.warnings,
        "paraxial_xy": {
            "x": xy.x.iter().map(first_order_json).collect::<Vec<_>>(),
            "y": xy.y.iter().map(first_order_json).collect::<Vec<_>>(),
            "squeeze": xy.squeeze,
        },
    });

    let mut cols: Vec<(String, Align)> = vec![("quantity".into(), Align::Left)];
    for (k, f) in p.wavelengths.iter().enumerate() {
        let star = if k == p.primary { "*" } else { "" };
        cols.push((
            format!("{} µm{star}", fmt_f(f.wavelength_um, 4)),
            Align::Right,
        ));
    }
    cols.push(("unit".into(), Align::Left));
    let col_refs: Vec<(&str, Align)> = cols.iter().map(|(s, al)| (s.as_str(), *al)).collect();
    let mut t = Table::new(&col_refs);
    for (k, label, unit) in QUANTITIES {
        let mut row = vec![(*label).to_string()];
        row.extend(p.wavelengths.iter().map(|f| fmt_g(quantity(f, k), 8)));
        row.push((*unit).into());
        t.row(row);
    }
    let mut text = format!("{}  (* primary)\n", session.system().name);
    text.push_str(&t.render());
    if xy.symmetry != mklight_core::surface_math::Symmetry::Rotational {
        text.push_str("\nAnamorphic first order (x / y)\n");
        let mut axes = Table::new(&[
            ("quantity", Align::Left),
            ("x", Align::Right),
            ("y", Align::Right),
            ("squeeze y/x", Align::Right),
        ]);
        let fx = &xy.x[xy.primary];
        let fy = &xy.y[xy.primary];
        for (key, label, unit) in QUANTITIES {
            axes.row(vec![
                format!("{label} ({unit})"),
                fmt_g(quantity(fx, key), 8),
                fmt_g(quantity(fy, key), 8),
                if *key == "efl" {
                    fmt_g(xy.squeeze[xy.primary], 8)
                } else {
                    String::new()
                },
            ]);
        }
        text.push_str(&axes.render());
    }
    if a.rays {
        let mut r = Table::new(&[
            ("#", Align::Right),
            ("marginal y", Align::Right),
            ("marginal u", Align::Right),
            ("chief y", Align::Right),
            ("chief u", Align::Right),
        ]);
        for j in 0..primary.marginal.y.len() {
            r.row(vec![
                j.to_string(),
                fmt_g(primary.marginal.y[j], 8),
                fmt_g(primary.marginal.u[j], 8),
                fmt_g(primary.chief.y[j], 8),
                fmt_g(primary.chief.u[j], 8),
            ]);
        }
        text.push_str("\nparaxial rays at the primary wavelength (u after each surface)\n");
        text.push_str(&r.render());
    }
    text.push_str(&warning_lines(&p.warnings));
    Ok(Report::ok(json, text))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pupil_parses_pairs_only() {
        assert_eq!(parse_pupil("0,1").unwrap(), [0.0, 1.0]);
        assert_eq!(parse_pupil(" -0.5 , 0.25").unwrap(), [-0.5, 0.25]);
        assert!(parse_pupil("1").is_err());
        assert!(parse_pupil("a,b").is_err());
        assert!(parse_pupil("inf,0").is_err());
        assert!(parse_pupil("3,0").is_err());
        assert!(parse_pupil("0.8,0.7").is_err());
        assert!(parse_pupil("0.7071068,0.7071068").is_ok());
        assert!(parse_pupil("-1,0").is_ok());
    }

    #[test]
    fn aoi_is_zero_at_normal_incidence() {
        assert_eq!(aoi_deg(1.0), 0.0);
        assert!((aoi_deg(0.5) - 60.0).abs() < 1e-12);
    }
}
