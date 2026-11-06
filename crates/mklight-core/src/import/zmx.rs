//! `.zmx` lens files: one record per line, a four-letter keyword and its
//! fields. A surface block starts with `SURF n`; its records are indented.
//! Lengths are in the file's `UNIT` and are converted to mm.

use std::collections::BTreeMap;

use super::{
    Draft, GlassResolver, ImportWarning, ImportWarningKind as K, Units, Warnings,
    curvature_of_radius, finish_system, number, rotational_shape, trim_zeros,
};
use crate::error::MkResult;
use crate::materials::MaterialLibrary;
use crate::model::surface::{DiffractivePhase, ZernikeOrdering};
use crate::model::{
    ApertureShape, ApertureSpec, Environment, Field, FieldKind, FieldSet, Interaction, MaterialRef,
    OpticalSystem, Placement, Surface, SurfaceShape, TransformOrder, Vignetting, Wavelength,
    WavelengthSet,
};
use crate::surface_math::MAX_PHASE_TERMS;

/// Records of a surface block that only affect drawing or bookkeeping.
const SILENT_SURFACE: &[&str] = &["HIDE", "SLAB", "POPS", "MIRR", "MEMA", "FLAP", "BLNK"];
/// Global records that only affect drawing, reports or bookkeeping.
const SILENT_GLOBAL: &[&str] = &[
    "VERS", "PFIL", "LANG", "GFAC", "GLRS", "HIDE", "MIRD", "SLAB", "POPS", "PUSH", "SDMA", "ROPD",
    "PICB", "PUPD", "POLS", "MAZH", "SCOL", "SWID", "TXTR", "COFN", "COFG", "BLNK", "OPDX", "RWRE",
    "EFFL",
];
/// Keywords that belong to a surface even when not indented.
const SURFACE_KEYS: &[&str] = &[
    "TYPE", "CURV", "CONI", "DISZ", "GLAS", "DIAM", "STOP", "PARM", "XDAT", "COMM", "CLAP", "SQAP",
    "ELAP", "OBSC", "COAT",
];

#[derive(Default)]
struct RawSurface {
    line: usize,
    kind: String,
    curv: f64,
    conic: f64,
    thickness: f64,
    /// Name, the numbers after it, and the line.
    glass: Option<(String, Vec<f64>, usize)>,
    diam: Option<f64>,
    stop: bool,
    parm: BTreeMap<i64, f64>,
    xdat: BTreeMap<i64, f64>,
    comment: String,
    aperture: Option<(ApertureShape, usize)>,
}

impl RawSurface {
    fn parm(&self, i: i64) -> f64 {
        self.parm.get(&i).copied().unwrap_or(0.0)
    }
    fn xdat(&self, i: i64) -> f64 {
        self.xdat.get(&i).copied().unwrap_or(0.0)
    }
}

#[derive(Default)]
struct Globals {
    units: Option<Units>,
    name: String,
    notes: Vec<String>,
    aperture: Option<(ApertureSpec, usize)>,
    /// FTYP: field type, number of fields, number of wavelengths.
    ftyp: Option<(i64, Option<usize>, Option<usize>)>,
    xfln: Vec<f64>,
    yfln: Vec<f64>,
    fwgt: Vec<f64>,
    vdx: Vec<f64>,
    vdy: Vec<f64>,
    vcx: Vec<f64>,
    vcy: Vec<f64>,
    wavm: BTreeMap<usize, (f64, f64)>,
    wavl: Vec<f64>,
    wwgt: Vec<f64>,
    pwav: Option<usize>,
    gcat: Vec<String>,
    env: Option<Environment>,
}

fn numbers(fields: &[&str]) -> Vec<f64> {
    fields.iter().map_while(|t| number(t)).collect()
}

/// The text after the keyword, without surrounding quotes.
fn rest_text(line: &str) -> String {
    let t = line.trim_start();
    let rest = t.split_once(char::is_whitespace).map_or("", |(_, r)| r);
    rest.trim().trim_matches('"').trim().to_string()
}

pub(crate) fn parse(
    text: &str,
    library: &MaterialLibrary,
) -> MkResult<(OpticalSystem, Vec<ImportWarning>)> {
    let mut w = Warnings::default();
    let mut g = Globals::default();
    let mut surfaces: Vec<RawSurface> = Vec::new();
    let mut in_surface = false;

    for (i, raw) in text.lines().enumerate() {
        let ln = i + 1;
        let line = raw.trim_end_matches('\r');
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let Some(&key) = tokens.first() else {
            continue;
        };
        let key = key.to_ascii_uppercase();
        let f = &tokens[1..];
        let indented = line.starts_with(char::is_whitespace);

        if key == "SURF" {
            surfaces.push(RawSurface {
                line: ln,
                kind: "STANDARD".into(),
                ..Default::default()
            });
            in_surface = true;
            continue;
        }
        if in_surface && (indented || SURFACE_KEYS.contains(&key.as_str())) {
            let idx = surfaces.len() - 1;
            surface_record(&mut surfaces[idx], &key, f, line, ln, idx, &mut w);
            continue;
        }
        in_surface = false;
        global_record(&mut g, &key, f, line, ln, &mut w);
    }

    let units = g.units.unwrap_or(Units::MM);
    let mut sys = OpticalSystem::new_empty();
    let primary_um = wavelengths(&mut sys, &g, &mut w);
    fields(&mut sys, &g, units, &mut w);
    if let Some(env) = g.env {
        sys.environment = env;
    }

    let resolver = GlassResolver {
        library,
        preferred: g.gcat.clone(),
    };
    let mut out = Vec::with_capacity(surfaces.len());
    let mut medium = MaterialRef::Air;
    let mut stop = None;
    let mut stop_sd = None;
    let last = surfaces.len().saturating_sub(1);
    for (idx, rs) in surfaces.iter().enumerate() {
        let mut s = convert_surface(rs, idx, units, primary_um, &mut w);
        s.thickness = units.len(rs.thickness);
        if idx == 0 && s.thickness.abs() >= 1e10 {
            s.thickness = f64::INFINITY;
        }
        if s.thickness.is_infinite() && idx != 0 {
            w.push(
                K::Malformed,
                Some(rs.line),
                Some(idx),
                "only the object distance can be infinite; 0 is used",
            );
            s.thickness = 0.0;
        }
        let before = medium.clone();
        match &rs.glass {
            _ if s.is_coordinate_break() => s.material = before,
            None => s.material = MaterialRef::Air,
            Some((name, _, _)) if name.eq_ignore_ascii_case("MIRROR") => {
                s.interaction = Interaction::Reflect;
                s.material = before;
            }
            Some((name, nums, gl)) if name.eq_ignore_ascii_case("___BLANK") => {
                let nd = nums.get(2).copied().unwrap_or(0.0);
                let vd = nums.get(3).copied().unwrap_or(0.0);
                if nd > 1.0 && vd > 0.0 {
                    s.material = MaterialRef::Model {
                        nd,
                        vd,
                        dpgf: nums.get(4).copied().unwrap_or(0.0),
                    };
                } else if nd > 0.0 {
                    s.material = MaterialRef::Constant { n: nd };
                } else {
                    w.push(
                        K::Malformed,
                        Some(*gl),
                        Some(idx),
                        "model glass without an index; air is used",
                    );
                    s.material = MaterialRef::Air;
                }
            }
            Some((name, nums, gl)) => {
                let nd_vd = match (nums.get(2), nums.get(3)) {
                    (Some(&nd), Some(&vd)) => Some((nd, vd)),
                    _ => None,
                };
                s.material = resolver.resolve(name, None, nd_vd, Some(*gl), idx, &mut w);
            }
        }
        // The image surface's medium does not matter; keep it air.
        if idx == last && idx > 0 {
            s.material = MaterialRef::Air;
            s.interaction = Interaction::Refract;
        }
        medium = s.material.clone();
        if rs.stop {
            stop = Some(idx);
            stop_sd = rs.diam.map(|d| units.len(d));
        }
        out.push(s);
    }

    let aperture = g.aperture.map(|(a, _)| match a {
        ApertureSpec::EntrancePupilDiameter { value } => ApertureSpec::EntrancePupilDiameter {
            value: units.len(value),
        },
        a => a,
    });
    let draft = Draft {
        name: g.name.clone(),
        notes: g.notes.join("\n"),
        surfaces: out,
        stop,
        aperture,
        stop_semi_diameter: stop_sd,
    };
    let sys = finish_system(sys, draft, &mut w)?;
    Ok((sys, w.finish()))
}

fn surface_record(
    s: &mut RawSurface,
    key: &str,
    f: &[&str],
    line: &str,
    ln: usize,
    idx: usize,
    w: &mut Warnings,
) {
    let first = f.first().and_then(|t| number(t));
    let bad = |w: &mut Warnings| {
        w.push(
            K::Malformed,
            Some(ln),
            Some(idx),
            format!("{key} could not be read; ignored"),
        );
    };
    match key {
        "TYPE" => s.kind = f.first().map_or("STANDARD".into(), |t| t.to_ascii_uppercase()),
        "CURV" => match first {
            Some(c) => s.curv = c,
            None => bad(w),
        },
        "CONI" => match first {
            Some(k) => s.conic = k,
            None => bad(w),
        },
        "DISZ" => match first {
            Some(t) => s.thickness = t,
            None => bad(w),
        },
        "DIAM" => s.diam = first,
        "STOP" => s.stop = true,
        "COMM" => s.comment = rest_text(line),
        "GLAS" => match f.first() {
            Some(name) => s.glass = Some((name.to_string(), numbers(&f[1..]), ln)),
            None => bad(w),
        },
        "PARM" | "XDAT" => {
            let index = f.first().and_then(|t| t.parse::<i64>().ok());
            match (index, f.get(1).and_then(|t| number(t))) {
                (Some(i), Some(v)) => {
                    let map = if key == "PARM" { &mut s.parm } else { &mut s.xdat };
                    map.insert(i, v);
                }
                _ => bad(w),
            }
        }
        "CLAP" => {
            let n = numbers(f);
            match (n.first(), n.get(1)) {
                (Some(&min), Some(&max)) if max > 0.0 => {
                    let shape = if min > 0.0 && min < max {
                        ApertureShape::Annular {
                            inner_radius: min,
                            outer_radius: max,
                        }
                    } else {
                        ApertureShape::Circular { radius: max }
                    };
                    s.aperture = Some((shape, ln));
                }
                _ => bad(w),
            }
        }
        "SQAP" | "ELAP" => {
            let n = numbers(f);
            match (n.first(), n.get(1)) {
                (Some(&x), Some(&y)) if x > 0.0 && y > 0.0 => {
                    let shape = if key == "SQAP" {
                        ApertureShape::Rectangular { half_x: x, half_y: y }
                    } else {
                        ApertureShape::Elliptical { semi_x: x, semi_y: y }
                    };
                    s.aperture = Some((shape, ln));
                }
                _ => bad(w),
            }
        }
        "OBSC" | "OBDC" => w.once(
            key,
            K::UnsupportedRecord,
            Some(ln),
            format!("{key}: obscurations and aperture decentres are not imported"),
        ),
        "COAT" => {
            if f.first().is_some_and(|c| !c.eq_ignore_ascii_case("NULL")) {
                w.once(
                    "COAT",
                    K::UnsupportedRecord,
                    Some(ln),
                    "COAT: coatings are not imported; the surfaces are uncoated",
                );
            }
        }
        k if SILENT_SURFACE.contains(&k) => {}
        k if k.starts_with("NS") => w.once(
            "NSC",
            K::NonSequential,
            Some(ln),
            "non-sequential data is not imported",
        ),
        k => w.once(
            &format!("surface:{k}"),
            K::UnsupportedRecord,
            Some(ln),
            format!("surface record {k} is not supported; dropped (solves, pickups and other surface data are not imported)"),
        ),
    }
}

fn global_record(g: &mut Globals, key: &str, f: &[&str], line: &str, ln: usize, w: &mut Warnings) {
    let first = f.first().and_then(|t| number(t));
    let set_aperture = |g: &mut Globals, a: Option<ApertureSpec>, w: &mut Warnings| match a {
        Some(a) => g.aperture = Some((a, ln)),
        None => w.push(
            K::Malformed,
            Some(ln),
            None,
            format!("{key} could not be read; ignored"),
        ),
    };
    match key {
        "NAME" => g.name = rest_text(line),
        "NOTE" => {
            let t = rest_text(line);
            let t = match t.split_once(char::is_whitespace) {
                Some((n, r)) if n.parse::<i64>().is_ok() => r.trim().to_string(),
                _ if t.parse::<i64>().is_ok() => String::new(),
                _ => t,
            };
            if !t.is_empty() {
                g.notes.push(t);
            }
        }
        "MODE" => {
            if f.first().is_some_and(|m| !m.eq_ignore_ascii_case("SEQ")) {
                w.once(
                    "NSC",
                    K::NonSequential,
                    Some(ln),
                    "the file is in non-sequential mode; only sequential surfaces are imported",
                );
            }
        }
        "UNIT" => {
            let u = f.first().map(|u| u.to_ascii_uppercase());
            g.units = Some(match u.as_deref() {
                Some("MM") => Units::MM,
                Some("CM") => Units(10.0),
                Some("IN") | Some("INCH") => Units(25.4),
                Some("M") | Some("METR") | Some("METER") | Some("METERS") => Units(1000.0),
                other => {
                    w.push(
                        K::Defaulted,
                        Some(ln),
                        None,
                        format!(
                            "unknown lens unit {}; mm assumed",
                            other.unwrap_or("(none)")
                        ),
                    );
                    Units::MM
                }
            });
        }
        "ENPD" => set_aperture(
            g,
            first
                .filter(|v| *v > 0.0)
                .map(|value| ApertureSpec::EntrancePupilDiameter { value }),
            w,
        ),
        "FNUM" => set_aperture(
            g,
            first
                .filter(|v| *v > 0.0)
                .map(|value| ApertureSpec::ImageSpaceFNumber { value }),
            w,
        ),
        "OBNA" => set_aperture(
            g,
            first
                .filter(|v| *v > 0.0 && *v < 1.0)
                .map(|value| ApertureSpec::ObjectSpaceNa { value }),
            w,
        ),
        "FLOA" => g.aperture = Some((ApertureSpec::FloatByStopSize, ln)),
        "FTYP" => {
            let n: Vec<i64> = f.iter().map_while(|t| t.parse().ok()).collect();
            match n.first() {
                Some(&t) => {
                    let count = |i: usize| n.get(i).and_then(|&c| (c > 0).then_some(c as usize));
                    g.ftyp = Some((t, count(2), count(3)));
                }
                None => w.push(
                    K::Malformed,
                    Some(ln),
                    None,
                    "FTYP could not be read; ignored",
                ),
            }
        }
        "XFLN" => g.xfln = numbers(f),
        "YFLN" => g.yfln = numbers(f),
        "FWGN" => g.fwgt = numbers(f),
        "FWGT" => {
            if g.fwgt.is_empty() {
                g.fwgt = numbers(f);
            }
        }
        "VDXN" => g.vdx = numbers(f),
        "VDYN" => g.vdy = numbers(f),
        "VCXN" => g.vcx = numbers(f),
        "VCYN" => g.vcy = numbers(f),
        "WAVM" => {
            let n = numbers(f);
            match (n.first(), n.get(1)) {
                (Some(&i), Some(&um)) if i >= 1.0 && um > 0.0 => {
                    g.wavm
                        .insert(i as usize, (um, n.get(2).copied().unwrap_or(1.0)));
                }
                _ => w.push(
                    K::Malformed,
                    Some(ln),
                    None,
                    "WAVM could not be read; ignored",
                ),
            }
        }
        "WAVL" | "WAVN" => g.wavl = numbers(f),
        "WWGT" | "WWGN" => g.wwgt = numbers(f),
        "PWAV" => g.pwav = first.filter(|p| *p >= 1.0).map(|p| p as usize),
        "GCAT" => g.gcat = f.iter().map(|s| s.to_string()).collect(),
        "ENVD" => {
            let n = numbers(f);
            if let (Some(&t), Some(&p)) = (n.first(), n.get(1))
                && p > 0.0
            {
                g.env = Some(Environment {
                    temperature_c: t,
                    pressure_atm: p,
                });
            }
        }
        "RAIM" => w.once(
            "RAIM",
            K::Approximated,
            Some(ln),
            "ray aiming settings are not imported; ray aiming is off",
        ),
        "TOL" | "TOLE" => w.once(
            "TOL",
            K::UnsupportedRecord,
            Some(ln),
            "tolerance data is not imported",
        ),
        "MNUM" | "MOFF" => w.once(
            "MCE",
            K::UnsupportedRecord,
            Some(ln),
            "multi-configuration data is not imported; the base configuration is used",
        ),
        k if SILENT_GLOBAL.contains(&k) => {}
        k if k.starts_with("NS") => w.once(
            "NSC",
            K::NonSequential,
            Some(ln),
            "non-sequential data is not imported",
        ),
        k => w.once(
            k,
            K::UnsupportedRecord,
            Some(ln),
            format!("record {k} is not supported; dropped"),
        ),
    }
}

/// Fills the wavelengths; returns the primary wavelength (µm).
fn wavelengths(sys: &mut OpticalSystem, g: &Globals, w: &mut Warnings) -> f64 {
    let mut list: Vec<Wavelength> = if !g.wavm.is_empty() {
        let n = g.ftyp.and_then(|f| f.2).unwrap_or(g.wavm.len());
        (1..=n)
            .filter_map(|i| g.wavm.get(&i))
            .map(|&(um, weight)| Wavelength { um, weight })
            .collect()
    } else {
        g.wavl
            .iter()
            .enumerate()
            .filter(|(_, um)| **um > 0.0)
            .map(|(i, &um)| Wavelength {
                um,
                weight: g.wwgt.get(i).copied().unwrap_or(1.0),
            })
            .collect()
    };
    for wl in &mut list {
        if wl.weight.is_nan() || wl.weight < 0.0 {
            wl.weight = 1.0;
        }
    }
    if list.is_empty() {
        w.push(
            K::Defaulted,
            None,
            None,
            "the file lists no wavelengths; F, d and C are used",
        );
        sys.wavelengths = WavelengthSet::default();
    } else {
        let primary = match g.pwav {
            Some(p) if p <= list.len() => p - 1,
            _ => {
                w.push(
                    K::Defaulted,
                    None,
                    None,
                    "no valid primary wavelength; the first one is used",
                );
                0
            }
        };
        sys.wavelengths = WavelengthSet {
            wavelengths: list,
            primary,
        };
    }
    sys.wavelengths.primary_um()
}

fn fields(sys: &mut OpticalSystem, g: &Globals, units: Units, w: &mut Warnings) {
    let (ftype, count) = match g.ftyp {
        Some((t, n, _)) => (Some(t), n),
        None => (None, None),
    };
    let last_set = (0..g.xfln.len().max(g.yfln.len())).rev().find(|&i| {
        g.xfln.get(i).is_some_and(|v| *v != 0.0) || g.yfln.get(i).is_some_and(|v| *v != 0.0)
    });
    let n = count.unwrap_or(last_set.map_or(1, |i| i + 1)).max(1);
    let kind = match ftype {
        Some(0) => FieldKind::ObjectAngle,
        Some(1) => FieldKind::ObjectHeight,
        Some(2) => FieldKind::ParaxialImageHeight,
        Some(3) => FieldKind::RealImageHeight,
        Some(t) => {
            w.push(
                K::Approximated,
                None,
                None,
                format!(
                    "field type {t} has no MKLight equivalent; the values are used as object angles"
                ),
            );
            FieldKind::ObjectAngle
        }
        None => {
            if last_set.is_some() {
                w.push(
                    K::Defaulted,
                    None,
                    None,
                    "the file does not state the field type; object angles are assumed",
                );
            }
            FieldKind::ObjectAngle
        }
    };
    let scale = if kind == FieldKind::ObjectAngle {
        1.0
    } else {
        units.0
    };
    let at = |v: &Vec<f64>, i: usize| v.get(i).copied().unwrap_or(0.0);
    let fields = (0..n)
        .map(|i| Field {
            x: at(&g.xfln, i) * scale,
            y: at(&g.yfln, i) * scale,
            weight: g.fwgt.get(i).copied().filter(|v| *v >= 0.0).unwrap_or(1.0),
            vignetting: Vignetting {
                vdx: at(&g.vdx, i),
                vdy: at(&g.vdy, i),
                vcx: at(&g.vcx, i),
                vcy: at(&g.vcy, i),
            },
        })
        .collect();
    sys.fields = FieldSet { kind, fields };
}

fn convert_surface(
    rs: &RawSurface,
    idx: usize,
    u: Units,
    primary_um: f64,
    w: &mut Warnings,
) -> Surface {
    let c = u.curv(rs.curv);
    let k = rs.conic;
    let parms =
        |range: std::ops::RangeInclusive<i64>| -> Vec<f64> { range.map(|i| rs.parm(i)).collect() };
    // PARM 1…8 on r², r⁴, …, r¹⁶.
    let even = || -> Vec<f64> {
        parms(1..=8)
            .iter()
            .enumerate()
            .map(|(i, a)| u.coef(*a, 2 * (i as i32 + 1)))
            .collect()
    };
    let unsupported = |w: &mut Warnings, what: String| {
        w.push(K::UnsupportedSurface, Some(rs.line), Some(idx), what);
    };
    let mut placement = None;
    let shape = match rs.kind.as_str() {
        "STANDARD" => rotational_shape(c, k, &[]),
        "EVENASPH" => rotational_shape(c, k, &even()),
        "ODDASPHE" => {
            let odd = trim_zeros(
                parms(1..=8)
                    .iter()
                    .enumerate()
                    .map(|(i, a)| u.coef(*a, i as i32 + 1))
                    .collect(),
            );
            if odd.is_empty() {
                rotational_shape(c, k, &[])
            } else {
                SurfaceShape::OddAsphere {
                    curvature: c,
                    conic: k,
                    coefficients: odd,
                }
            }
        }
        "COORDBRK" => {
            placement = Some(Placement {
                decenter_x: u.len(rs.parm(1)),
                decenter_y: u.len(rs.parm(2)),
                tilt_x: rs.parm(3),
                tilt_y: rs.parm(4),
                tilt_z: rs.parm(5),
                order: if rs.parm(6) != 0.0 {
                    TransformOrder::TiltThenDecenter
                } else {
                    TransformOrder::DecenterThenTilt
                },
                restore_after: false,
            });
            SurfaceShape::CoordinateBreak
        }
        "PARAXIAL" => {
            let f = u.len(rs.parm(1));
            if f != 0.0 {
                SurfaceShape::Paraxial { focal_length: f }
            } else {
                unsupported(
                    w,
                    "paraxial surface with zero focal length; replaced by a plane".into(),
                );
                SurfaceShape::Plane
            }
        }
        "TOROIDAL" => {
            if rs.parm(2) != 0.0 {
                unsupported(
                    w,
                    "toroidal y² coefficient dropped (MKLight's y series starts at y⁴)".into(),
                );
            }
            if rs.xdat.values().any(|v| *v != 0.0) {
                unsupported(w, "toroidal Zernike terms dropped".into());
            }
            SurfaceShape::Toroid {
                curvature_x: u.curv(curvature_of_radius(rs.parm(1))),
                curvature_y: c,
                conic_y: k,
                y_asphere: trim_zeros(
                    (3..=8)
                        .map(|i| u.coef(rs.parm(i), 2 * (i as i32 - 1)))
                        .collect(),
                ),
            }
        }
        "BICONICX" => SurfaceShape::Biconic {
            curvature_x: u.curv(curvature_of_radius(rs.parm(1))),
            curvature_y: c,
            conic_x: rs.parm(2),
            conic_y: k,
            x_asphere: Vec::new(),
            y_asphere: Vec::new(),
        },
        "XPOLYNOM" => match polynomial_terms(rs, u, crate::surface_math::MAX_XY_TERMS, w, idx) {
            Some((norm_radius, coefficients)) => SurfaceShape::XYPolynomial {
                curvature: c,
                conic: k,
                norm_radius,
                coefficients,
            },
            None => rotational_shape(c, k, &[]),
        },
        t @ ("SZERNSAG" | "FZERNSAG") => {
            let ordering = if t == "SZERNSAG" {
                ZernikeOrdering::Standard
            } else {
                ZernikeOrdering::Fringe
            };
            if parms(1..=8).iter().any(|v| *v != 0.0) {
                unsupported(
                    w,
                    "Zernike sag surface: its even-asphere terms are dropped".into(),
                );
            }
            match polynomial_terms(rs, u, ordering.max_terms(), w, idx) {
                Some((norm_radius, coefficients)) => SurfaceShape::ZernikeSag {
                    curvature: c,
                    conic: k,
                    norm_radius,
                    ordering,
                    coefficients,
                },
                None => rotational_shape(c, k, &[]),
            }
        }
        "BINARY_2" => {
            let mut e = even();
            if e[0] != 0.0 {
                unsupported(w, "binary-optic base: r² coefficient dropped".into());
                e[0] = 0.0;
            }
            let base = rotational_shape(c, k, &e);
            let order = rs.parm(0).round() as i32;
            let n = (rs.xdat(1).max(0.0) as usize).min(MAX_PHASE_TERMS);
            if rs.xdat(1) as usize > MAX_PHASE_TERMS {
                unsupported(
                    w,
                    format!("binary-optic phase truncated to {MAX_PHASE_TERMS} terms"),
                );
            }
            let norm_radius = u.len(rs.xdat(2));
            if norm_radius > 0.0 {
                SurfaceShape::Diffractive {
                    base: Box::new(base),
                    phase: DiffractivePhase {
                        norm_radius,
                        coefficients: trim_zeros((0..n as i64).map(|i| rs.xdat(3 + i)).collect()),
                        design_order: if order == 0 { 1 } else { order },
                        design_wavelength_um: primary_um,
                        trace_order: order,
                    },
                }
            } else {
                unsupported(
                    w,
                    "binary-optic surface without a normalisation radius; its phase is dropped"
                        .into(),
                );
                base
            }
        }
        "NONSEQCO" => {
            w.once(
                "NSC",
                K::NonSequential,
                Some(rs.line),
                "non-sequential data is not imported",
            );
            w.push(
                K::NonSequential,
                Some(rs.line),
                Some(idx),
                "non-sequential component replaced by a plane",
            );
            SurfaceShape::Plane
        }
        other => {
            unsupported(
                w,
                format!("surface type {other} is not supported; replaced by its base sphere"),
            );
            rotational_shape(c, k, &[])
        }
    };
    let mut s = Surface::new(idx as u64, shape, 0.0, MaterialRef::Air);
    s.label = rs.comment.clone();
    s.placement = placement;
    if let Some((shape, _)) = &rs.aperture {
        s.aperture = Some(scale_aperture(shape, u));
    }
    s
}

/// `XDAT 1` term count, `XDAT 2` normalisation radius, then coefficients
/// (sag in lens units).
fn polynomial_terms(
    rs: &RawSurface,
    u: Units,
    max: usize,
    w: &mut Warnings,
    idx: usize,
) -> Option<(f64, Vec<f64>)> {
    let norm = u.len(rs.xdat(2));
    if norm <= 0.0 {
        w.push(
            K::UnsupportedSurface,
            Some(rs.line),
            Some(idx),
            format!(
                "{} surface without a normalisation radius; replaced by its base sphere",
                rs.kind
            ),
        );
        return None;
    }
    let n = rs.xdat(1).max(0.0) as usize;
    if n > max {
        w.push(
            K::UnsupportedSurface,
            Some(rs.line),
            Some(idx),
            format!(
                "{} surface: only the first {max} terms are imported",
                rs.kind
            ),
        );
    }
    let coefficients = trim_zeros(
        (0..n.min(max) as i64)
            .map(|i| u.len(rs.xdat(3 + i)))
            .collect(),
    );
    Some((norm, coefficients))
}

fn scale_aperture(a: &ApertureShape, u: Units) -> ApertureShape {
    match *a {
        ApertureShape::Circular { radius } => ApertureShape::Circular {
            radius: u.len(radius),
        },
        ApertureShape::Annular {
            inner_radius,
            outer_radius,
        } => ApertureShape::Annular {
            inner_radius: u.len(inner_radius),
            outer_radius: u.len(outer_radius),
        },
        ApertureShape::Rectangular { half_x, half_y } => ApertureShape::Rectangular {
            half_x: u.len(half_x),
            half_y: u.len(half_y),
        },
        ApertureShape::Elliptical { semi_x, semi_y } => ApertureShape::Elliptical {
            semi_x: u.len(semi_x),
            semi_y: u.len(semi_y),
        },
    }
}
