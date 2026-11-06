//! `.seq` lens files: a command script. Commands are separated by `;` or
//! line ends, `!` starts a comment. `SO`, `S` and `SI` add the object, lens
//! and image surfaces; most surface commands act on the latest surface or on
//! the one a qualifier names (`S3`, `SO`, `SI`). Wavelengths are in nm,
//! lengths in the file's `DIM` and are converted to mm.

use super::{
    Draft, GlassResolver, ImportWarning, ImportWarningKind as K, Units, Warnings,
    curvature_of_radius, finish_system, number, rotational_shape,
};
use crate::error::MkResult;
use crate::materials::MaterialLibrary;
use crate::model::{
    ApertureShape, ApertureSpec, Field, FieldKind, FieldSet, Interaction, MaterialRef,
    OpticalSystem, Placement, Solve, SolveKind, Surface, SurfaceId, Vignetting, Wavelength,
    WavelengthSet,
};
use crate::param::ParamPath;

/// Commands that only affect bookkeeping or drawing, or whose meaning the
/// surface data already carries.
const SILENT: &[&str] = &[
    "LEN", "GO", "SPH", "CON", "ASP", "UID", "INI", "COL", "WID", "VIE", "END",
];
/// Optimisation-variable flags.
const VARIABLE_FLAGS: &[&str] = &[
    "CCY", "CCX", "THC", "GLC", "KC", "KCX", "AC", "BC", "CC", "DC", "EC", "FC", "GC", "HC", "JC",
    "XDC", "YDC", "ZDC", "ADC", "BDC", "CDC",
];
/// Solves and pickups.
const SOLVES: &[&str] = &[
    "PUY", "PUX", "PUI", "PCY", "PCX", "PCI", "UMY", "UMX", "UCY", "UCX", "HMY", "HMX", "HCY",
    "HCX", "PIK", "THM", "SLV",
];
/// Asphere coefficient letters, on r⁴, r⁶, … r²⁰.
const ASPHERE_LETTERS: [&str; 9] = ["A", "B", "C", "D", "E", "F", "G", "H", "J"];

struct Command {
    line: usize,
    tokens: Vec<String>,
}

/// Splits the script into commands, keeping quoted text as one token.
fn commands(text: &str) -> Vec<Command> {
    let mut out = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let mut tokens = Vec::new();
        let mut cur = String::new();
        let mut quote: Option<char> = None;
        let flush = |cur: &mut String, tokens: &mut Vec<String>| {
            if !cur.is_empty() {
                tokens.push(std::mem::take(cur));
            }
        };
        for ch in raw.chars() {
            match quote {
                Some(q) if ch == q => {
                    tokens.push(std::mem::take(&mut cur));
                    quote = None;
                }
                Some(_) => cur.push(ch),
                None => match ch {
                    '\'' | '"' => {
                        flush(&mut cur, &mut tokens);
                        quote = Some(ch);
                    }
                    '!' => break,
                    ';' => {
                        flush(&mut cur, &mut tokens);
                        if !tokens.is_empty() {
                            out.push(Command {
                                line: i + 1,
                                tokens: std::mem::take(&mut tokens),
                            });
                        }
                    }
                    c if c.is_whitespace() || c == ',' => flush(&mut cur, &mut tokens),
                    c => cur.push(c),
                },
            }
        }
        flush(&mut cur, &mut tokens);
        if !tokens.is_empty() {
            out.push(Command {
                line: i + 1,
                tokens,
            });
        }
    }
    out
}

#[derive(Default)]
struct SeqSurface {
    line: usize,
    /// In file units.
    curvature: f64,
    thickness: f64,
    conic: f64,
    /// On r⁴ … r²⁰, file units.
    asphere: [f64; 9],
    glass: Option<(String, usize)>,
    reflect: bool,
    special: Option<String>,
    semi_diameter: Option<f64>,
    rect: (Option<f64>, Option<f64>),
    ellipse: (Option<f64>, Option<f64>),
    decenter: Option<(f64, f64)>,
    return_after: bool,
}

#[derive(Default)]
struct State {
    radius_mode: bool,
    units: Option<Units>,
    name: String,
    aperture: Option<ApertureSpec>,
    wavelengths_nm: Vec<f64>,
    weights: Vec<f64>,
    reference: Option<usize>,
    field_kind: Option<FieldKind>,
    fx: Vec<f64>,
    fy: Vec<f64>,
    fw: Vec<f64>,
    vux: Vec<f64>,
    vlx: Vec<f64>,
    vuy: Vec<f64>,
    vly: Vec<f64>,
    surfaces: Vec<SeqSurface>,
    image: Option<usize>,
    current: Option<usize>,
    stop: Option<usize>,
    paraxial_image_solve: bool,
    in_private_catalog: bool,
}

fn nums(t: &[String]) -> Vec<f64> {
    t.iter().map_while(|s| number(s)).collect()
}

pub(crate) fn parse(
    text: &str,
    library: &MaterialLibrary,
) -> MkResult<(OpticalSystem, Vec<ImportWarning>)> {
    let mut w = Warnings::default();
    let mut st = State {
        radius_mode: true,
        ..Default::default()
    };
    for cmd in commands(text) {
        command(&mut st, &cmd, &mut w);
    }
    build(st, library, w)
}

/// The surface a command acts on: a leading `Sk`/`SO`/`SI` qualifier (which
/// is removed from `args`), else the latest surface.
fn target(
    st: &State,
    args: &mut Vec<String>,
    ln: usize,
    key: &str,
    w: &mut Warnings,
) -> Option<usize> {
    let q = args.first().map(|a| a.to_ascii_uppercase());
    let explicit = match q.as_deref() {
        Some("SO") | Some("S0") => Some(Some(0)),
        Some("SI") => Some(st.image.or(st.surfaces.len().checked_sub(1))),
        Some(s)
            if s.len() > 1 && s.starts_with('S') && s[1..].chars().all(|c| c.is_ascii_digit()) =>
        {
            Some(s[1..].parse::<usize>().ok())
        }
        Some("SA") | Some("SL") => {
            w.once(
                &format!("{key}:range"),
                K::UnsupportedRecord,
                Some(ln),
                format!("{key} on a surface range is not supported; dropped"),
            );
            return None;
        }
        _ => None,
    };
    let index = match explicit {
        Some(i) => {
            args.remove(0);
            i
        }
        None => st.current,
    };
    match index {
        Some(i) if i < st.surfaces.len() => Some(i),
        _ => {
            w.push(
                K::Malformed,
                Some(ln),
                None,
                format!("{key}: no such surface; ignored"),
            );
            None
        }
    }
}

fn command(st: &mut State, cmd: &Command, w: &mut Warnings) {
    let ln = cmd.line;
    let key = cmd.tokens[0].to_ascii_uppercase();
    let mut args: Vec<String> = cmd.tokens[1..].to_vec();
    if st.in_private_catalog {
        if key == "END" {
            st.in_private_catalog = false;
        }
        return;
    }
    let bad = |w: &mut Warnings, key: &str| {
        w.push(
            K::Malformed,
            Some(ln),
            None,
            format!("{key} could not be read; ignored"),
        );
    };
    let first = args.first().and_then(|a| number(a));
    match key.as_str() {
        "RDM" => st.radius_mode = !args.first().is_some_and(|a| a.eq_ignore_ascii_case("N")),
        "CUM" if args.is_empty() => st.radius_mode = false,
        "TIT" | "TITLE" => st.name = args.join(" "),
        "DIM" => {
            st.units = Some(match args.first().map(|a| a.to_ascii_uppercase()).as_deref() {
                Some("M") | None => Units::MM,
                Some("C") => Units(10.0),
                Some("I") => Units(25.4),
                Some(other) => {
                    w.push(K::Defaulted, Some(ln), None, format!("unknown unit {other}; mm assumed"));
                    Units::MM
                }
            })
        }
        "EPD" => match first.filter(|v| *v > 0.0) {
            Some(value) => st.aperture = Some(ApertureSpec::EntrancePupilDiameter { value }),
            None => bad(w, &key),
        },
        "FNO" => match first.filter(|v| *v > 0.0) {
            Some(value) => st.aperture = Some(ApertureSpec::ImageSpaceFNumber { value }),
            None => bad(w, &key),
        },
        "NAO" => match first.filter(|v| *v > 0.0 && *v < 1.0) {
            Some(value) => st.aperture = Some(ApertureSpec::ObjectSpaceNa { value }),
            None => bad(w, &key),
        },
        "NA" => match first.filter(|v| *v > 0.0 && *v < 1.0) {
            Some(na) => {
                w.push(
                    K::Approximated,
                    Some(ln),
                    None,
                    "an image-space NA is imported as the paraxial working F/# 1/(2 NA)",
                );
                st.aperture = Some(ApertureSpec::ParaxialWorkingFNumber { value: 0.5 / na });
            }
            None => bad(w, &key),
        },
        "WL" => st.wavelengths_nm = nums(&args),
        "WTW" => st.weights = nums(&args),
        "REF" => st.reference = first.filter(|v| *v >= 1.0).map(|v| v as usize),
        "XAN" | "YAN" | "XOB" | "YOB" | "XIM" | "YIM" | "XRI" | "YRI" => {
            let kind = match &key[1..] {
                "AN" => FieldKind::ObjectAngle,
                "OB" => FieldKind::ObjectHeight,
                "IM" => FieldKind::ParaxialImageHeight,
                _ => FieldKind::RealImageHeight,
            };
            if st.field_kind.is_some_and(|k| k != kind) {
                w.push(K::Approximated, Some(ln), None, format!("{key} changes the field type; the last one given is used"));
            }
            st.field_kind = Some(kind);
            let v = nums(&args);
            if key.starts_with('X') { st.fx = v } else { st.fy = v }
        }
        "WTF" => st.fw = nums(&args),
        "VUX" => st.vux = nums(&args),
        "VLX" => st.vlx = nums(&args),
        "VUY" => st.vuy = nums(&args),
        "VLY" => st.vly = nums(&args),
        "SO" => {
            let n = nums(&args);
            let (r, t) = match n.as_slice() {
                [t] => (0.0, *t),
                [r, t, ..] => (*r, *t),
                [] => (0.0, f64::INFINITY),
            };
            let s = SeqSurface {
                line: ln,
                curvature: if st.radius_mode { curvature_of_radius(r) } else { r },
                thickness: t,
                ..Default::default()
            };
            if st.surfaces.is_empty() {
                st.surfaces.push(s);
            } else {
                st.surfaces[0] = s;
            }
            st.current = Some(0);
        }
        "S" | "SI" => {
            if st.surfaces.is_empty() {
                w.push(K::Defaulted, Some(ln), None, "no object surface (SO) before the first surface; an object at infinity is used");
                st.surfaces.push(SeqSurface {
                    thickness: f64::INFINITY,
                    ..Default::default()
                });
            }
            if st.image.is_some() {
                w.push(K::Malformed, Some(ln), None, "a surface after the image surface; ignored");
                return;
            }
            let n = nums(&args);
            let r = n.first().copied().unwrap_or(0.0);
            let mut s = SeqSurface {
                line: ln,
                curvature: if st.radius_mode { curvature_of_radius(r) } else { r },
                thickness: n.get(1).copied().unwrap_or(0.0),
                ..Default::default()
            };
            if let Some(g) = args.get(n.len()) {
                set_glass(&mut s, g, ln);
            }
            st.surfaces.push(s);
            st.current = Some(st.surfaces.len() - 1);
            if key == "SI" {
                st.image = st.current;
            }
        }
        "STO" => {
            if let Some(i) = target(st, &mut args, ln, &key, w) {
                st.stop = Some(i);
            }
        }
        "RDY" | "CUY" => {
            if let Some(i) = target(st, &mut args, ln, &key, w) {
                match args.first().and_then(|a| number(a)) {
                    Some(v) => st.surfaces[i].curvature = if key == "RDY" { curvature_of_radius(v) } else { v },
                    None => solve_dropped(w, &key, ln),
                }
            }
        }
        "THI" => {
            if let Some(i) = target(st, &mut args, ln, &key, w) {
                match args.first().and_then(|a| number(a)) {
                    Some(v) => st.surfaces[i].thickness = v,
                    None if args.first().is_some_and(|a| a.eq_ignore_ascii_case("PIM")) => {
                        st.paraxial_image_solve = true
                    }
                    None => solve_dropped(w, &key, ln),
                }
            }
        }
        "GLA" => {
            if let Some(i) = target(st, &mut args, ln, &key, w) {
                match args.first() {
                    Some(g) => set_glass(&mut st.surfaces[i], g, ln),
                    None => bad(w, &key),
                }
            }
        }
        "REFL" => {
            if let Some(i) = target(st, &mut args, ln, &key, w) {
                st.surfaces[i].reflect = true;
            }
        }
        "RMD" => {
            if let Some(i) = target(st, &mut args, ln, &key, w) {
                match args.first().map(|a| a.to_ascii_uppercase()).as_deref() {
                    Some("REFL") => st.surfaces[i].reflect = true,
                    Some("REFR") | None => st.surfaces[i].reflect = false,
                    Some(m) => w.once(&format!("RMD {m}"), K::UnsupportedRecord, Some(ln), format!("RMD {m} is not supported; the surface refracts")),
                }
            }
        }
        "K" => {
            if let Some(i) = target(st, &mut args, ln, &key, w) {
                match args.first().and_then(|a| number(a)) {
                    Some(k) => st.surfaces[i].conic = k,
                    None => bad(w, &key),
                }
            }
        }
        k if ASPHERE_LETTERS.contains(&k) => {
            let j = ASPHERE_LETTERS.iter().position(|l| *l == k).expect("listed");
            if let Some(i) = target(st, &mut args, ln, &key, w) {
                match args.first().and_then(|a| number(a)) {
                    Some(v) => st.surfaces[i].asphere[j] = v,
                    None => bad(w, &key),
                }
            }
        }
        "SPS" => {
            if let Some(i) = target(st, &mut args, ln, &key, w) {
                st.surfaces[i].special = Some(args.first().cloned().unwrap_or_default().to_ascii_uppercase());
            }
        }
        "SCO" | "SCC" => {}
        "CIR" | "REX" | "REY" | "ELX" | "ELY" => {
            if let Some(i) = target(st, &mut args, ln, &key, w) {
                let modifier = args.first().filter(|a| number(a).is_none()).map(|a| a.to_ascii_uppercase());
                if modifier.is_some() {
                    args.remove(0);
                }
                match modifier.as_deref() {
                    Some("EDG") => return,
                    Some("HOL") | Some("OBS") => {
                        w.once("obscuration", K::UnsupportedRecord, Some(ln), "holes and obscurations are not imported");
                        return;
                    }
                    _ => {}
                }
                let Some(v) = args.first().and_then(|a| number(a)).filter(|v| *v > 0.0) else {
                    return bad(w, &key);
                };
                let s = &mut st.surfaces[i];
                match key.as_str() {
                    "CIR" => s.semi_diameter = Some(v),
                    "REX" => s.rect.0 = Some(v),
                    "REY" => s.rect.1 = Some(v),
                    "ELX" => s.ellipse.0 = Some(v),
                    _ => s.ellipse.1 = Some(v),
                }
            }
        }
        "XDE" | "YDE" => {
            if let Some(i) = target(st, &mut args, ln, &key, w) {
                match args.first().and_then(|a| number(a)) {
                    Some(v) => {
                        let d = st.surfaces[i].decenter.get_or_insert((0.0, 0.0));
                        if key == "XDE" { d.0 = v } else { d.1 = v }
                    }
                    None => bad(w, &key),
                }
            }
        }
        "DAR" => {
            if let Some(i) = target(st, &mut args, ln, &key, w) {
                st.surfaces[i].return_after = true;
            }
        }
        "ZDE" | "ADE" | "BDE" | "CDE" | "BEN" | "REV" | "RET" | "DEC" => w.once(
            "tilts",
            K::UnsupportedRecord,
            Some(ln),
            format!("{key}: tilts, axial decentres, bends and reversals are not imported; only x/y decentres are"),
        ),
        "PIM" => st.paraxial_image_solve = true,
        "PRV" => {
            st.in_private_catalog = true;
            w.once(
                "PRV",
                K::UnsupportedRecord,
                Some(ln),
                "private glass catalogs (PRV) are not imported; their glasses are unknown",
            );
        }
        k if VARIABLE_FLAGS.contains(&k) => w.once(
            "variables",
            K::SolveDropped,
            Some(ln),
            "optimisation-variable flags are not imported",
        ),
        k if SOLVES.contains(&k) => solve_dropped(w, k, ln),
        k if SILENT.contains(&k) => {}
        k => w.once(
            &format!("cmd:{k}"),
            K::UnsupportedRecord,
            Some(ln),
            format!("command {k} is not supported; dropped"),
        ),
    }
}

fn solve_dropped(w: &mut Warnings, key: &str, ln: usize) {
    w.once(
        &format!("solve:{key}"),
        K::SolveDropped,
        Some(ln),
        format!("{key}: solves and pickups are not imported; the current values are kept"),
    );
}

fn set_glass(s: &mut SeqSurface, g: &str, ln: usize) {
    if g.eq_ignore_ascii_case("REFL") {
        s.reflect = true;
    } else if !g.eq_ignore_ascii_case("AIR") {
        s.glass = Some((g.to_string(), ln));
    }
}

/// A model glass written as `nd:vd` or as a six-digit glass code.
fn model_glass(g: &str) -> Option<(f64, f64)> {
    if let Some((a, b)) = g.split_once(':') {
        return Some((number(a)?, number(b)?));
    }
    let code = g.split('.').next()?;
    if code.len() == 6 && code.chars().all(|c| c.is_ascii_digit()) {
        let n: f64 = code[..3].parse().ok()?;
        let v: f64 = code[3..].parse().ok()?;
        return Some((1.0 + n / 1000.0, v / 10.0));
    }
    None
}

fn build(
    st: State,
    library: &MaterialLibrary,
    mut w: Warnings,
) -> MkResult<(OpticalSystem, Vec<ImportWarning>)> {
    let u = st.units.unwrap_or(Units::MM);
    let mut sys = OpticalSystem::new_empty();

    // Wavelengths (nm in the file).
    let wls: Vec<Wavelength> = st
        .wavelengths_nm
        .iter()
        .enumerate()
        .filter(|(_, nm)| **nm > 0.0)
        .map(|(i, nm)| Wavelength {
            // Through femtometres, so that 587.5618 nm reads as 0.5875618 µm exactly.
            um: (nm * 1e6).round() / 1e9,
            weight: st
                .weights
                .get(i)
                .copied()
                .filter(|v| *v >= 0.0)
                .unwrap_or(1.0),
        })
        .collect();
    if wls.is_empty() {
        w.push(
            K::Defaulted,
            None,
            None,
            "the file lists no wavelengths (WL); F, d and C are used",
        );
    } else {
        let primary = match st.reference {
            Some(r) if r <= wls.len() => r - 1,
            Some(_) => {
                w.push(
                    K::Defaulted,
                    None,
                    None,
                    "REF names no listed wavelength; the first one is used",
                );
                0
            }
            // The usual default reference is the middle wavelength.
            None => wls.len() / 2,
        };
        sys.wavelengths = WavelengthSet {
            wavelengths: wls,
            primary,
        };
    }

    // Fields.
    let kind = st.field_kind.unwrap_or(FieldKind::ObjectAngle);
    let n = st.fx.len().max(st.fy.len()).max(1);
    let scale = if kind == FieldKind::ObjectAngle {
        1.0
    } else {
        u.0
    };
    let at = |v: &Vec<f64>, i: usize| v.get(i).copied().unwrap_or(0.0);
    sys.fields = FieldSet {
        kind,
        fields: (0..n)
            .map(|i| {
                // Upper and lower pupil limits to decentre and compression.
                let (ux, lx, uy, ly) = (
                    at(&st.vux, i),
                    at(&st.vlx, i),
                    at(&st.vuy, i),
                    at(&st.vly, i),
                );
                Field {
                    x: at(&st.fx, i) * scale,
                    y: at(&st.fy, i) * scale,
                    weight: st.fw.get(i).copied().filter(|v| *v >= 0.0).unwrap_or(1.0),
                    vignetting: Vignetting {
                        vdx: 0.5 * (lx - ux),
                        vdy: 0.5 * (ly - uy),
                        vcx: 0.5 * (lx + ux),
                        vcy: 0.5 * (ly + uy),
                    },
                }
            })
            .collect(),
    };

    let resolver = GlassResolver {
        library,
        preferred: Vec::new(),
    };
    let last = st.surfaces.len().saturating_sub(1);
    let mut medium = MaterialRef::Air;
    let mut surfaces = Vec::with_capacity(st.surfaces.len());
    let mut stop_sd = None;
    for (idx, ss) in st.surfaces.iter().enumerate() {
        let c = u.curv(ss.curvature);
        let even: Vec<f64> = std::iter::once(0.0)
            .chain(
                ss.asphere
                    .iter()
                    .enumerate()
                    .map(|(j, a)| u.coef(*a, 2 * j as i32 + 4)),
            )
            .collect();
        if let Some(t) = &ss.special {
            w.push(
                K::UnsupportedSurface,
                Some(ss.line),
                Some(idx),
                format!(
                    "special surface type {t} is not supported; its base sphere or conic is kept"
                ),
            );
        }
        let shape = rotational_shape(c, ss.conic, &even);
        let mut thickness = u.len(ss.thickness);
        if idx == 0 && thickness.abs() >= 1e10 {
            thickness = f64::INFINITY;
        }
        let mut s = Surface::new(idx as u64, shape, thickness, MaterialRef::Air);
        if ss.reflect {
            s.interaction = Interaction::Reflect;
            s.material = medium.clone();
        } else if let Some((g, gl)) = &ss.glass {
            s.material = glass(&resolver, g, *gl, idx, &mut w);
        }
        if idx == last && idx > 0 {
            s.material = MaterialRef::Air;
            s.interaction = Interaction::Refract;
        }
        medium = s.material.clone();
        // One given half-width of a rectangle or ellipse stands for both.
        let pair =
            |(a, b): (Option<f64>, Option<f64>)| a.or(b).map(|x| (u.len(x), u.len(b.unwrap_or(x))));
        s.aperture = match (pair(ss.rect), pair(ss.ellipse)) {
            (Some((half_x, half_y)), _) => Some(ApertureShape::Rectangular { half_x, half_y }),
            (None, Some((semi_x, semi_y))) => Some(ApertureShape::Elliptical { semi_x, semi_y }),
            (None, None) => None,
        };
        if let Some((dx, dy)) = ss.decenter {
            s.placement = Some(Placement {
                decenter_x: u.len(dx),
                decenter_y: u.len(dy),
                restore_after: ss.return_after,
                ..Default::default()
            });
        }
        if st.stop == Some(idx) {
            stop_sd = ss.semi_diameter.map(|v| u.len(v));
        }
        surfaces.push(s);
    }
    if st.image.is_none() && surfaces.len() > 2 {
        w.push(
            K::Defaulted,
            None,
            None,
            "the file has no image surface (SI); the last surface is used as the image",
        );
    }
    if st.paraxial_image_solve && surfaces.len() >= 3 {
        let j = surfaces.len() - 2;
        sys.solves.push(Solve {
            target: ParamPath::Thickness {
                surface: SurfaceId(j as u64),
            },
            kind: SolveKind::MarginalRayHeight { height: 0.0 },
        });
    }
    let aperture = st.aperture.map(|a| match a {
        ApertureSpec::EntrancePupilDiameter { value } => ApertureSpec::EntrancePupilDiameter {
            value: u.len(value),
        },
        a => a,
    });
    let draft = Draft {
        name: st.name,
        notes: String::new(),
        surfaces,
        stop: st.stop,
        aperture,
        stop_semi_diameter: stop_sd,
    };
    let sys = finish_system(sys, draft, &mut w)?;
    Ok((sys, w.finish()))
}

/// Resolves a `.seq` glass: `NAME_VENDOR`, a bare name, `nd:vd` or a glass code.
fn glass(r: &GlassResolver<'_>, g: &str, ln: usize, idx: usize, w: &mut Warnings) -> MaterialRef {
    if let Some((nd, vd)) = model_glass(g) {
        return if nd > 1.0 && vd > 0.0 {
            MaterialRef::Model { nd, vd, dpgf: 0.0 }
        } else {
            w.push(
                K::Malformed,
                Some(ln),
                Some(idx),
                format!("model glass {g} is not valid; air is used"),
            );
            MaterialRef::Air
        };
    }
    // A name that exists as written (underscores included) wins.
    if r.find(None, g).is_some() {
        return r.resolve(g, None, None, Some(ln), idx, w);
    }
    match g.rsplit_once('_') {
        Some((name, vendor))
            if !name.is_empty() && vendor.chars().all(|c| c.is_ascii_alphabetic()) =>
        {
            r.resolve(name, Some(vendor), None, Some(ln), idx, w)
        }
        _ => r.resolve(g, None, None, Some(ln), idx, w),
    }
}
