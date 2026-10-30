//! Structural validation of an [`OpticalSystem`] (no tracing).
//!
//! [`OpticalSystem::validation_issues`] collects every problem as a
//! [`ValidationIssue`] with a stable snake-case `code` and the JSON path of
//! the offending value (`surfaces[3].shape.curvature`,
//! `coatings["AR"].layers[0].thickness_um`, …).
//! [`OpticalSystem::validate`] returns the first one as an error. The codes are
//! listed in `docs/CONVENTIONS.md` §13.
//!
//! Numbers are checked for finiteness everywhere. The only infinities a model
//! may hold are the thickness of the object (object at infinity) and of the
//! image surface (it has no successor), and an override or pickup of the
//! object thickness.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::error::{MkError, MkResult};
use crate::model::{
    ApertureShape, ApertureSpec, CoatingDef, FieldKind, Interaction, MaterialRef, MechanicalPart,
    OpticalSystem, Placement, RayAiming, ScatterModel, SemiDiameter, SurfaceShape,
};
use crate::param::ParamPath;

/// One structural problem of a model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationIssue {
    /// Stable snake-case code (`docs/CONVENTIONS.md` §13).
    pub code: String,
    /// JSON path of the offending value, e.g. `surfaces[3].thickness`.
    pub path: String,
    pub message: String,
}

impl fmt::Display for ValidationIssue {
    /// `[code] path: message`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}: {}", self.code, self.path, self.message)
    }
}

/// Collects issues while walking the model.
#[derive(Default)]
struct Issues(Vec<ValidationIssue>);

impl Issues {
    fn push(&mut self, code: &str, path: impl Into<String>, message: impl Into<String>) {
        self.0.push(ValidationIssue {
            code: code.into(),
            path: path.into(),
            message: message.into(),
        });
    }
    /// `non_finite` unless `v` is a finite number.
    fn finite(&mut self, path: impl Into<String>, v: f64) -> bool {
        if v.is_finite() {
            return true;
        }
        self.push("non_finite", path, format!("{v} is not a finite number"));
        false
    }
    /// Finite and `> 0` (`code` otherwise).
    fn positive(&mut self, code: &str, path: impl Into<String>, v: f64) {
        let path = path.into();
        if self.finite(path.clone(), v) && v <= 0.0 {
            self.push(code, path, format!("must be > 0 (got {v})"));
        }
    }
    /// Finite and `≥ 0` (`code` otherwise).
    fn non_negative(&mut self, code: &str, path: impl Into<String>, v: f64) {
        let path = path.into();
        if self.finite(path.clone(), v) && v < 0.0 {
            self.push(code, path, format!("must be ≥ 0 (got {v})"));
        }
    }
}

fn check_scatter(out: &mut Issues, path: &str, m: &ScatterModel) {
    match *m {
        ScatterModel::Lambertian { tis } => {
            out.non_negative("invalid_scatter", format!("{path}.tis"), tis)
        }
        ScatterModel::Abg { a, b, g } => {
            out.non_negative("invalid_scatter", format!("{path}.a"), a);
            out.non_negative("invalid_scatter", format!("{path}.b"), b);
            out.finite(format!("{path}.g"), g);
        }
        ScatterModel::Gaussian { tis, sigma_slope } => {
            out.non_negative("invalid_scatter", format!("{path}.tis"), tis);
            out.positive(
                "invalid_scatter",
                format!("{path}.sigma_slope"),
                sigma_slope,
            );
        }
    }
}

fn check_material(out: &mut Issues, path: &str, m: &MaterialRef) {
    match *m {
        MaterialRef::Model { nd, vd, dpgf } => {
            out.positive("invalid_material", format!("{path}.nd"), nd);
            out.positive("invalid_material", format!("{path}.vd"), vd);
            out.finite(format!("{path}.dpgf"), dpgf);
        }
        MaterialRef::Constant { n } => out.positive("invalid_material", format!("{path}.n"), n),
        _ => {}
    }
}

fn check_placement(out: &mut Issues, path: &str, p: &Placement) {
    for (k, v) in [
        ("decenter_x", p.decenter_x),
        ("decenter_y", p.decenter_y),
        ("tilt_x", p.tilt_x),
        ("tilt_y", p.tilt_y),
        ("tilt_z", p.tilt_z),
    ] {
        out.finite(format!("{path}.{k}"), v);
    }
}

fn check_aperture_shape(out: &mut Issues, path: &str, a: &ApertureShape) {
    let pos = |out: &mut Issues, k: &str, v: f64| {
        out.positive("invalid_aperture", format!("{path}.{k}"), v)
    };
    match *a {
        ApertureShape::Circular { radius } => pos(out, "radius", radius),
        ApertureShape::Annular {
            inner_radius,
            outer_radius,
        } => {
            out.non_negative(
                "invalid_aperture",
                format!("{path}.inner_radius"),
                inner_radius,
            );
            pos(out, "outer_radius", outer_radius);
            if inner_radius >= outer_radius {
                out.push(
                    "invalid_aperture",
                    path,
                    format!(
                        "inner radius {inner_radius} must be below the outer radius {outer_radius}"
                    ),
                );
            }
        }
        ApertureShape::Rectangular { half_x, half_y } => {
            pos(out, "half_x", half_x);
            pos(out, "half_y", half_y);
        }
        ApertureShape::Elliptical { semi_x, semi_y } => {
            pos(out, "semi_x", semi_x);
            pos(out, "semi_y", semi_y);
        }
    }
}

fn check_coating(out: &mut Issues, path: &str, c: &CoatingDef) {
    match c {
        CoatingDef::ThinFilm {
            reference_wavelength_um,
            layers,
            mirror_substrate,
        } => {
            out.positive(
                "invalid_coating",
                format!("{path}.reference_wavelength_um"),
                *reference_wavelength_um,
            );
            for (j, l) in layers.iter().enumerate() {
                let lp = format!("{path}.layers[{j}]");
                check_material(out, &format!("{lp}.material"), &l.material);
                out.non_negative(
                    "invalid_coating",
                    format!("{lp}.thickness_um"),
                    l.thickness_um,
                );
                if let Some(q) = l.quarter_waves {
                    out.non_negative("invalid_coating", format!("{lp}.quarter_waves"), q);
                }
                out.non_negative("invalid_coating", format!("{lp}.extinction"), l.extinction);
            }
            if let Some((n, k)) = mirror_substrate {
                out.positive("invalid_coating", format!("{path}.mirror_substrate[0]"), *n);
                out.non_negative("invalid_coating", format!("{path}.mirror_substrate[1]"), *k);
            }
        }
        CoatingDef::Ideal {
            reflectance,
            transmittance,
        } => {
            let (r, t) = (*reflectance, *transmittance);
            let ok_r = out.finite(format!("{path}.reflectance"), r);
            let ok_t = out.finite(format!("{path}.transmittance"), t);
            if ok_r && ok_t && !(r >= 0.0 && t >= 0.0 && r + t <= 1.0 + 1e-12) {
                out.push(
                    "invalid_coating",
                    path,
                    format!("R = {r} and T = {t} must be ≥ 0 with R + T ≤ 1"),
                );
            }
        }
        CoatingDef::Tabulated { reflectance } => {
            for (j, &(l, r)) in reflectance.iter().enumerate() {
                let p = format!("{path}.reflectance[{j}]");
                out.positive("invalid_coating", format!("{p}[0]"), l);
                if out.finite(format!("{p}[1]"), r) && !(0.0..=1.0).contains(&r) {
                    out.push(
                        "invalid_coating",
                        p,
                        format!("reflectance {r} outside [0, 1]"),
                    );
                }
            }
        }
    }
}

/// `"solve 3 (pickup): …"` → `solves[3]`, the code and the message.
fn solve_issue(e: &MkError) -> (String, &'static str, String) {
    let msg = match e {
        MkError::InvalidModel(m) | MkError::InvalidParam(m) => m.clone(),
        other => other.to_string(),
    };
    let path = msg
        .strip_prefix("solve ")
        .and_then(|r| r.split_once(' '))
        .and_then(|(k, _)| k.parse::<usize>().ok())
        .map_or_else(|| "solves".to_string(), |k| format!("solves[{k}]"));
    let code = if msg.contains("no surface with id") {
        "missing_surface"
    } else {
        "invalid_solve"
    };
    (path, code, msg)
}

impl OpticalSystem {
    /// Every structural problem of the model (module docs), in a fixed order.
    /// Empty for a valid model.
    pub fn validation_issues(&self) -> Vec<ValidationIssue> {
        let mut out = Issues::default();
        let n = self.surfaces.len();
        if n < 3 {
            out.push(
                "too_few_surfaces",
                "surfaces",
                "a system needs object, at least one surface, and image",
            );
            return out.0;
        }
        if self.stop == 0 || self.stop >= n - 1 {
            out.push(
                "stop_out_of_range",
                "stop",
                format!("stop index {} must be between 1 and {}", self.stop, n - 2),
            );
        }

        // Aperture.
        let aperture_value = match self.aperture {
            ApertureSpec::EntrancePupilDiameter { value }
            | ApertureSpec::ImageSpaceFNumber { value }
            | ApertureSpec::ObjectSpaceNa { value }
            | ApertureSpec::ParaxialWorkingFNumber { value } => Some(value),
            ApertureSpec::FloatByStopSize => None,
        };
        if let Some(v) = aperture_value {
            out.positive("invalid_aperture_value", "aperture.value", v);
        }

        // Wavelengths.
        let wls = &self.wavelengths.wavelengths;
        if wls.is_empty() || self.wavelengths.primary >= wls.len() {
            out.push(
                "invalid_wavelengths",
                "wavelengths",
                "wavelength set empty or primary out of range",
            );
        }
        for (k, w) in wls.iter().enumerate() {
            let p = format!("wavelengths.wavelengths[{k}]");
            out.positive("invalid_wavelength", format!("{p}.um"), w.um);
            out.non_negative("negative_weight", format!("{p}.weight"), w.weight);
        }

        // Fields.
        if self.fields.fields.is_empty() {
            out.push(
                "no_fields",
                "fields.fields",
                "at least one field is required",
            );
        }
        for (k, f) in self.fields.fields.iter().enumerate() {
            let p = format!("fields.fields[{k}]");
            out.finite(format!("{p}.x"), f.x);
            out.finite(format!("{p}.y"), f.y);
            out.non_negative("negative_weight", format!("{p}.weight"), f.weight);
            let v = &f.vignetting;
            for (name, x) in [
                ("vdx", v.vdx),
                ("vdy", v.vdy),
                ("vcx", v.vcx),
                ("vcy", v.vcy),
            ] {
                out.finite(format!("{p}.vignetting.{name}"), x);
            }
        }
        let infinite_object = self.object_at_infinity();
        if infinite_object && matches!(self.fields.kind, FieldKind::ObjectHeight) {
            out.push(
                "object_height_at_infinity",
                "fields.kind",
                "object-height fields require a finite object distance",
            );
        }
        if infinite_object && matches!(self.aperture, ApertureSpec::ObjectSpaceNa { .. }) {
            out.push(
                "object_na_at_infinity",
                "aperture",
                "object-space NA requires a finite object distance",
            );
        }
        if infinite_object && matches!(self.fields.kind, FieldKind::ObjectAngle) {
            for (k, f) in self.fields.fields.iter().enumerate() {
                if f.x.hypot(f.y) >= 90.0 {
                    out.push(
                        "field_angle_too_large",
                        format!("fields.fields[{k}]"),
                        "field angles must be below 90°",
                    );
                }
            }
        }

        // Environment and aiming.
        let env = &self.environment;
        if out.finite("environment.temperature_c", env.temperature_c)
            && env.temperature_c <= -273.15
        {
            out.push(
                "invalid_environment",
                "environment.temperature_c",
                "the temperature must be above absolute zero",
            );
        }
        out.non_negative(
            "invalid_environment",
            "environment.pressure_atm",
            env.pressure_atm,
        );
        if let RayAiming::Real { tolerance } = self.ray_aiming {
            out.positive("invalid_ray_aiming", "ray_aiming.tolerance", tolerance);
        }

        // Surfaces.
        let mut ids = std::collections::BTreeSet::new();
        for (i, s) in self.surfaces.iter().enumerate() {
            let p = format!("surfaces[{i}]");
            if !ids.insert(s.id) {
                out.push(
                    "duplicate_surface_id",
                    format!("{p}.id"),
                    format!("duplicate surface id {}", s.id.0),
                );
            }
            if s.thickness.is_nan() {
                out.finite(format!("{p}.thickness"), s.thickness);
            } else if i > 0 && !s.thickness.is_finite() && i != n - 1 {
                out.push(
                    "infinite_thickness",
                    format!("{p}.thickness"),
                    format!("surface {i}: only the object may have infinite thickness"),
                );
            }
            let sp = format!("{p}.shape");
            match &s.shape {
                SurfaceShape::Conic { curvature, conic } => {
                    out.finite(format!("{sp}.curvature"), *curvature);
                    out.finite(format!("{sp}.conic"), *conic);
                }
                SurfaceShape::EvenAsphere {
                    curvature,
                    conic,
                    coefficients,
                } => {
                    out.finite(format!("{sp}.curvature"), *curvature);
                    out.finite(format!("{sp}.conic"), *conic);
                    for (k, a) in coefficients.iter().enumerate() {
                        out.finite(format!("{sp}.coefficients[{k}]"), *a);
                    }
                }
                SurfaceShape::Paraxial { focal_length } => {
                    if out.finite(format!("{sp}.focal_length"), *focal_length)
                        && *focal_length == 0.0
                    {
                        out.push(
                            "invalid_focal_length",
                            format!("{sp}.focal_length"),
                            "an ideal lens needs a non-zero focal length",
                        );
                    }
                }
                SurfaceShape::Plane | SurfaceShape::CoordinateBreak => {}
            }
            check_material(&mut out, &format!("{p}.material"), &s.material);
            if let Some(c) = &s.coating
                && !self.coatings.contains_key(c)
            {
                out.push(
                    "unknown_coating",
                    format!("{p}.coating"),
                    format!("unknown coating '{c}'"),
                );
            }
            if let SemiDiameter::Fixed { value } = s.semi_diameter
                && !(value > 0.0 && value.is_finite())
            {
                out.push(
                    "invalid_semi_diameter",
                    format!("{p}.semi_diameter.value"),
                    format!(
                        "surface {i}: fixed semi-diameter must be finite and > 0 (got {value})"
                    ),
                );
            }
            if let Some(a) = &s.aperture {
                check_aperture_shape(&mut out, &format!("{p}.aperture"), a);
            }
            if let Some(pl) = &s.placement {
                check_placement(&mut out, &format!("{p}.placement"), pl);
            }
            if let Some(sc) = &s.scatter {
                check_scatter(&mut out, &format!("{p}.scatter"), sc);
            }
            if let Some(m) = s.edge.mechanical_semi_diameter {
                out.positive(
                    "invalid_mechanical_semi_diameter",
                    format!("{p}.edge.mechanical_semi_diameter"),
                    m,
                );
            }
            if let Some(sc) = &s.edge.edge_scatter {
                check_scatter(&mut out, &format!("{p}.edge.edge_scatter"), sc);
            }
            if (i == 0 || i == n - 1) && s.is_coordinate_break() {
                out.push(
                    "coordinate_break_at_end",
                    sp,
                    "object/image cannot be coordinate breaks",
                );
            }
        }
        if self.surfaces[n - 1].interaction == Interaction::Reflect {
            out.push(
                "image_mirror",
                format!("surfaces[{}].interaction", n - 1),
                "the image surface cannot be a mirror",
            );
        }

        // Coatings.
        for (name, c) in &self.coatings {
            check_coating(&mut out, &format!("coatings[{name:?}]"), c);
        }

        // Mechanics.
        let has = |id: crate::model::SurfaceId| self.surface_index(id).is_some();
        for (k, m) in self.mechanics.iter().enumerate() {
            let p = format!("mechanics[{k}]");
            let missing = |out: &mut Issues, key: &str, id: crate::model::SurfaceId| {
                if !has(id) {
                    out.push(
                        "missing_surface",
                        format!("{p}.{key}"),
                        format!("no surface with id {}", id.0),
                    );
                }
            };
            match m {
                MechanicalPart::Barrel {
                    from,
                    to,
                    inner_radius,
                    finish,
                } => {
                    missing(&mut out, "from", *from);
                    missing(&mut out, "to", *to);
                    out.positive(
                        "invalid_mechanics",
                        format!("{p}.inner_radius"),
                        *inner_radius,
                    );
                    check_scatter(&mut out, &format!("{p}.finish"), finish);
                }
                MechanicalPart::AnnularBaffle {
                    surface,
                    z,
                    inner_radius,
                    outer_radius,
                    finish,
                } => {
                    missing(&mut out, "surface", *surface);
                    out.finite(format!("{p}.z"), *z);
                    out.non_negative(
                        "invalid_mechanics",
                        format!("{p}.inner_radius"),
                        *inner_radius,
                    );
                    out.positive(
                        "invalid_mechanics",
                        format!("{p}.outer_radius"),
                        *outer_radius,
                    );
                    if inner_radius >= outer_radius {
                        out.push(
                            "invalid_mechanics",
                            p.clone(),
                            "the inner radius must be below the outer radius",
                        );
                    }
                    check_scatter(&mut out, &format!("{p}.finish"), finish);
                }
            }
        }

        // Sensor.
        if let Some(sn) = &self.sensor {
            if sn.pixels_x == 0 || sn.pixels_y == 0 {
                out.push("invalid_sensor", "sensor", "the sensor needs pixels");
            }
            out.positive("invalid_sensor", "sensor.pitch_x", sn.pitch_x);
            out.positive("invalid_sensor", "sensor.pitch_y", sn.pitch_y);
            if let Some(c) = sn.max_cra_deg {
                out.finite("sensor.max_cra_deg", c);
            }
            for (k, &(l, r)) in sn.spectral_response.iter().enumerate() {
                out.positive(
                    "invalid_sensor",
                    format!("sensor.spectral_response[{k}][0]"),
                    l,
                );
                out.non_negative(
                    "invalid_sensor",
                    format!("sensor.spectral_response[{k}][1]"),
                    r,
                );
            }
        }

        // Configurations: every path resolves, every value is a number the
        // parameter can hold (±∞ only for the object distance).
        for (k, c) in self.configurations.iter().enumerate() {
            for (j, (path, v)) in c.overrides.iter().enumerate() {
                let p = format!("configurations[{k}].overrides[{j}]");
                if let Err(e) = path.get(self) {
                    let code = if e.to_string().contains("no surface with id") {
                        "missing_surface"
                    } else {
                        "invalid_configuration"
                    };
                    out.push(code, p, e.to_string());
                    continue;
                }
                let object_distance = matches!(path, ParamPath::Thickness { surface }
                    if self.surface_index(*surface) == Some(0));
                if v.is_nan() || (v.is_infinite() && !object_distance) {
                    out.push(
                        "non_finite",
                        p,
                        format!("configuration {k}: {v} is not a finite number"),
                    );
                }
            }
        }

        // Solves (structure, targets, sources, cycles). Checked on an
        // otherwise valid model only: their rules assume one.
        if out.0.is_empty()
            && let Err(e) = crate::model::solve::validate_solves(self)
        {
            let (path, code, msg) = solve_issue(&e);
            out.push(code, path, msg);
        }
        out.0
    }

    /// Structural validation (module docs). It does not trace. Solvers can
    /// assume a validated system. Returns the first issue of
    /// [`OpticalSystem::validation_issues`]: `MkError::UnknownCoating` for an
    /// unknown coating name, `MkError::InvalidModel("[code] path: message")`
    /// otherwise.
    pub fn validate(&self) -> MkResult<()> {
        match self.validation_issues().into_iter().next() {
            None => Ok(()),
            Some(i) if i.code == "unknown_coating" => Err(MkError::UnknownCoating(
                self.surfaces
                    .iter()
                    .filter_map(|s| s.coating.clone())
                    .find(|c| !self.coatings.contains_key(c))
                    .unwrap_or_default(),
            )),
            Some(i) => Err(MkError::InvalidModel(i.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        Configuration, Field, Layer, Sensor, Solve, Surface, SurfaceId, Wavelength,
    };
    use crate::samples;

    /// The codes and paths of a system's issues.
    fn issues(s: &OpticalSystem) -> Vec<(String, String)> {
        s.validation_issues()
            .into_iter()
            .map(|i| (i.code, i.path))
            .collect()
    }

    /// `edit` makes the Cooke triplet invalid with exactly this issue.
    fn rejects(code: &str, path: &str, edit: impl FnOnce(&mut OpticalSystem)) {
        let mut s = samples::cooke_triplet();
        assert!(s.validate().is_ok());
        edit(&mut s);
        assert_eq!(
            issues(&s),
            vec![(code.to_string(), path.to_string())],
            "{code} at {path}"
        );
        let e = s.validate().unwrap_err().to_string();
        assert!(e.contains(&format!("[{code}] {path}: ")), "{e}");
    }

    fn baffle(surface: u64, z: f64) -> MechanicalPart {
        MechanicalPart::AnnularBaffle {
            surface: SurfaceId(surface),
            z,
            inner_radius: 1.0,
            outer_radius: 2.0,
            finish: ScatterModel::Lambertian { tis: 0.1 },
        }
    }

    fn override_of(p: ParamPath, v: f64) -> Configuration {
        Configuration {
            name: "c".into(),
            overrides: vec![(p, v)],
        }
    }

    fn coated_cooke() -> OpticalSystem {
        let mut s = samples::cooke_triplet();
        s.coatings.insert(
            "AR".into(),
            CoatingDef::ThinFilm {
                reference_wavelength_um: 0.55,
                layers: vec![Layer {
                    material: MaterialRef::Constant { n: 1.38 },
                    thickness_um: 0.0,
                    quarter_waves: Some(1.0),
                    extinction: 0.0,
                }],
                mirror_substrate: None,
            },
        );
        s.surfaces[1].coating = Some("AR".into());
        s
    }

    #[test]
    fn samples_have_no_issues() {
        for (name, s) in samples::all() {
            assert_eq!(s.validation_issues(), vec![], "{name}");
        }
        assert!(coated_cooke().validate().is_ok());
    }

    #[test]
    fn non_finite_numbers_are_rejected_everywhere() {
        let nan = f64::NAN;
        let inf = f64::INFINITY;
        rejects("non_finite", "surfaces[1].shape.curvature", |s| {
            s.surfaces[1].shape = SurfaceShape::Conic {
                curvature: nan,
                conic: 0.0,
            }
        });
        rejects("non_finite", "surfaces[2].shape.conic", |s| {
            s.surfaces[2].shape = SurfaceShape::Conic {
                curvature: 0.01,
                conic: -inf,
            }
        });
        rejects("non_finite", "surfaces[3].shape.coefficients[1]", |s| {
            s.surfaces[3].shape = SurfaceShape::EvenAsphere {
                curvature: 0.01,
                conic: 0.0,
                coefficients: vec![1e-6, nan],
            }
        });
        rejects("non_finite", "surfaces[2].thickness", |s| {
            s.surfaces[2].thickness = nan
        });
        rejects("non_finite", "surfaces[0].thickness", |s| {
            s.surfaces[0].thickness = nan
        });
        rejects("non_finite", "surfaces[1].shape.focal_length", |s| {
            s.surfaces[1].shape = SurfaceShape::Paraxial { focal_length: inf }
        });
        rejects("non_finite", "surfaces[2].placement.tilt_y", |s| {
            s.surfaces[2].placement = Some(Placement {
                tilt_y: nan,
                ..Default::default()
            })
        });
        rejects("non_finite", "surfaces[2].placement.decenter_x", |s| {
            s.surfaces[2].placement = Some(Placement {
                decenter_x: inf,
                ..Default::default()
            })
        });
        rejects("non_finite", "aperture.value", |s| {
            s.aperture = ApertureSpec::EntrancePupilDiameter { value: nan }
        });
        rejects("non_finite", "fields.fields[1].y", |s| {
            s.fields.fields[1].y = nan
        });
        rejects("non_finite", "fields.fields[2].vignetting.vcy", |s| {
            s.fields.fields[2].vignetting.vcy = inf
        });
        rejects("non_finite", "wavelengths.wavelengths[0].weight", |s| {
            s.wavelengths.wavelengths[0].weight = nan
        });
        rejects("non_finite", "wavelengths.wavelengths[2].um", |s| {
            s.wavelengths.wavelengths[2].um = inf
        });
        rejects("non_finite", "surfaces[1].material.nd", |s| {
            s.surfaces[1].material = MaterialRef::Model {
                nd: nan,
                vd: 50.0,
                dpgf: 0.0,
            }
        });
        rejects("non_finite", "surfaces[2].aperture.radius", |s| {
            s.surfaces[2].aperture = Some(ApertureShape::Circular { radius: inf })
        });
        rejects(
            "non_finite",
            "surfaces[2].edge.mechanical_semi_diameter",
            |s| s.surfaces[2].edge.mechanical_semi_diameter = Some(nan),
        );
        rejects("non_finite", "surfaces[5].scatter.tis", |s| {
            s.surfaces[5].scatter = Some(ScatterModel::Lambertian { tis: nan })
        });
        rejects("non_finite", "environment.temperature_c", |s| {
            s.environment.temperature_c = nan
        });
        rejects("non_finite", "mechanics[0].z", |s| {
            s.mechanics.push(baffle(2, inf))
        });
        rejects("non_finite", "configurations[0].overrides[0]", |s| {
            s.configurations.push(override_of(
                ParamPath::Thickness {
                    surface: SurfaceId(2),
                },
                inf,
            ))
        });
        rejects("non_finite", "sensor.max_cra_deg", |s| {
            s.sensor = Some(Sensor {
                pixels_x: 10,
                pixels_y: 10,
                pitch_x: 0.004,
                pitch_y: 0.004,
                max_cra_deg: Some(nan),
                spectral_response: vec![],
            })
        });
        // A coating layer.
        let mut s = coated_cooke();
        if let Some(CoatingDef::ThinFilm { layers, .. }) = s.coatings.get_mut("AR") {
            layers[0].extinction = nan;
        }
        assert_eq!(
            issues(&s),
            vec![(
                "non_finite".into(),
                "coatings[\"AR\"].layers[0].extinction".into()
            )]
        );
        // The object distance alone may be infinite, also as an override.
        let mut s = samples::finite_relay();
        s.configurations.push(override_of(
            ParamPath::Thickness {
                surface: SurfaceId(0),
            },
            inf,
        ));
        assert_eq!(issues(&s), vec![]);
    }

    #[test]
    fn values_out_of_range() {
        rejects("invalid_aperture_value", "aperture.value", |s| {
            s.aperture = ApertureSpec::EntrancePupilDiameter { value: 0.0 }
        });
        rejects("invalid_aperture_value", "aperture.value", |s| {
            s.aperture = ApertureSpec::ImageSpaceFNumber { value: -2.0 }
        });
        rejects("negative_weight", "fields.fields[1].weight", |s| {
            s.fields.fields[1].weight = -1.0
        });
        rejects(
            "negative_weight",
            "wavelengths.wavelengths[1].weight",
            |s| s.wavelengths.wavelengths[1].weight = -0.5,
        );
        rejects("invalid_wavelength", "wavelengths.wavelengths[0].um", |s| {
            s.wavelengths.wavelengths[0] = Wavelength {
                um: 0.0,
                weight: 1.0,
            }
        });
        rejects(
            "invalid_semi_diameter",
            "surfaces[3].semi_diameter.value",
            |s| s.surfaces[3].semi_diameter = SemiDiameter::Fixed { value: f64::NAN },
        );
        rejects("invalid_aperture", "surfaces[3].aperture", |s| {
            s.surfaces[3].aperture = Some(ApertureShape::Annular {
                inner_radius: 3.0,
                outer_radius: 2.0,
            })
        });
        rejects("invalid_aperture", "surfaces[3].aperture.half_y", |s| {
            s.surfaces[3].aperture = Some(ApertureShape::Rectangular {
                half_x: 3.0,
                half_y: 0.0,
            })
        });
        rejects(
            "invalid_focal_length",
            "surfaces[1].shape.focal_length",
            |s| s.surfaces[1].shape = SurfaceShape::Paraxial { focal_length: 0.0 },
        );
        rejects("invalid_material", "surfaces[1].material.n", |s| {
            s.surfaces[1].material = MaterialRef::Constant { n: -1.5 }
        });
        rejects("invalid_environment", "environment.pressure_atm", |s| {
            s.environment.pressure_atm = -1.0
        });
        rejects("invalid_ray_aiming", "ray_aiming.tolerance", |s| {
            s.ray_aiming = RayAiming::Real { tolerance: 0.0 }
        });
        rejects("invalid_sensor", "sensor.pitch_x", |s| {
            s.sensor = Some(Sensor {
                pixels_x: 10,
                pixels_y: 10,
                pitch_x: 0.0,
                pitch_y: 0.004,
                max_cra_deg: None,
                spectral_response: vec![],
            })
        });
        rejects("invalid_mechanics", "mechanics[0].inner_radius", |s| {
            s.mechanics.push(MechanicalPart::Barrel {
                from: SurfaceId(1),
                to: SurfaceId(6),
                inner_radius: -1.0,
                finish: ScatterModel::Lambertian { tis: 0.1 },
            })
        });
        rejects("invalid_coating", "coatings[\"R\"]", |s| {
            s.coatings.insert(
                "R".into(),
                CoatingDef::Ideal {
                    reflectance: 0.7,
                    transmittance: 0.5,
                },
            );
        });
        rejects(
            "invalid_scatter",
            "surfaces[2].edge.edge_scatter.sigma_slope",
            |s| {
                s.surfaces[2].edge.edge_scatter = Some(ScatterModel::Gaussian {
                    tis: 0.1,
                    sigma_slope: 0.0,
                })
            },
        );
    }

    #[test]
    fn references_to_missing_surfaces() {
        rejects("missing_surface", "mechanics[0].to", |s| {
            s.mechanics.push(MechanicalPart::Barrel {
                from: SurfaceId(1),
                to: SurfaceId(42),
                inner_radius: 10.0,
                finish: ScatterModel::Lambertian { tis: 0.1 },
            })
        });
        rejects("missing_surface", "mechanics[0].surface", |s| {
            s.mechanics.push(baffle(42, 0.0))
        });
        rejects("missing_surface", "configurations[0].overrides[0]", |s| {
            s.configurations.push(override_of(
                ParamPath::Thickness {
                    surface: SurfaceId(42),
                },
                1.0,
            ))
        });
        rejects(
            "invalid_configuration",
            "configurations[0].overrides[0]",
            |s| {
                s.configurations
                    .push(override_of(ParamPath::FieldY { index: 7 }, 1.0))
            },
        );
        let t = |id| ParamPath::Thickness {
            surface: SurfaceId(id),
        };
        rejects("missing_surface", "solves[0]", |s| {
            s.solves.push(Solve::pickup(t(2), t(42), 1.0, 0.0))
        });
        rejects("missing_surface", "solves[0]", |s| {
            s.solves
                .push(Solve::marginal_ray_height(SurfaceId(42), 0.0))
        });
        rejects("invalid_solve", "solves[0]", |s| {
            s.solves.push(Solve::pickup(t(2), t(2), 1.0, 0.0))
        });
    }

    #[test]
    fn structural_rules_keep_their_errors() {
        rejects("stop_out_of_range", "stop", |s| s.stop = 0);
        rejects("infinite_thickness", "surfaces[3].thickness", |s| {
            s.surfaces[3].thickness = f64::INFINITY
        });
        rejects("duplicate_surface_id", "surfaces[2].id", |s| {
            s.surfaces[2].id = SurfaceId(1)
        });
        rejects("image_mirror", "surfaces[7].interaction", |s| {
            s.surfaces[7].interaction = Interaction::Reflect
        });
        rejects("no_fields", "fields.fields", |s| s.fields.fields.clear());
        rejects("field_angle_too_large", "fields.fields[2]", |s| {
            s.fields.fields[2] = Field::new(0.0, 90.0)
        });
        rejects("too_few_surfaces", "surfaces", |s| s.surfaces.truncate(2));
        // Unknown coatings keep their own error variant.
        let mut s = samples::cooke_triplet();
        s.surfaces[2].coating = Some("nope".into());
        assert_eq!(s.validate(), Err(MkError::UnknownCoating("nope".into())));
        // Every issue is reported, not only the first.
        let mut s = samples::cooke_triplet();
        s.surfaces[1].thickness = f64::NAN;
        s.fields.fields[0].weight = -1.0;
        s.surfaces
            .push(Surface::new(99, SurfaceShape::Plane, 0.0, MaterialRef::Air));
        s.surfaces[8].id = SurfaceId(3);
        let codes: Vec<String> = s.validation_issues().into_iter().map(|i| i.code).collect();
        assert_eq!(
            codes,
            ["negative_weight", "non_finite", "duplicate_surface_id"]
        );
    }

    /// The macOS app's JSON fixtures (written by `examples/dump_fixtures`
    /// before unknown keys were rejected) still load, and the engine's
    /// verdicts recorded in `validation_cases.json` still hold.
    #[test]
    fn app_fixtures_still_load() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../apps/macos/Tests/MKLightModelTests/Fixtures");
        let read = |f: &str| std::fs::read_to_string(dir.join(f)).unwrap();
        for f in ["empty.json", "cooke_triplet.json", "kitchen_sink.json"] {
            let s: OpticalSystem = serde_json::from_str(&read(f)).unwrap();
            assert_eq!(s.validation_issues(), vec![], "{f}");
        }
        let v: Vec<OpticalSystem> = serde_json::from_str(&read("variants.json")).unwrap();
        assert!(v.len() >= 4);
        let p: Vec<ParamPath> = serde_json::from_str(&read("param_paths.json")).unwrap();
        assert!(!p.is_empty());
        #[derive(Deserialize)]
        struct Case {
            name: String,
            system: OpticalSystem,
            error: Option<String>,
        }
        let cases: Vec<Case> = serde_json::from_str(&read("validation_cases.json")).unwrap();
        assert!(cases.len() >= 20);
        for c in cases {
            let now = c.system.validate();
            assert_eq!(now.is_err(), c.error.is_some(), "{}: {now:?}", c.name);
            if let (Err(e), Some(old)) = (now, c.error) {
                // Same reason, now with a code and a path.
                let why = old.rsplit(": ").next().unwrap();
                assert!(e.to_string().ends_with(why), "{}: {e} vs {old}", c.name);
            }
        }
    }

    #[test]
    fn unknown_keys_are_rejected_on_load() {
        let good = serde_json::to_value(samples::double_gauss()).unwrap();
        assert!(serde_json::from_value::<OpticalSystem>(good.clone()).is_ok());
        let typo = |edit: &dyn Fn(&mut serde_json::Value)| {
            let mut v = good.clone();
            edit(&mut v);
            serde_json::from_value::<OpticalSystem>(v)
                .unwrap_err()
                .to_string()
        };
        let e = typo(&|v| v["surfaces"][1]["semi_diamter"] = serde_json::json!({"mode": "auto"}));
        assert!(e.contains("unknown field `semi_diamter`"), "{e}");
        let e = typo(&|v| v["surfaces"][1]["shape"]["conik"] = 1.0.into());
        assert!(e.contains("conik"), "{e}");
        let e = typo(&|v| v["fields"]["fields"][0]["wieght"] = 1.0.into());
        assert!(e.contains("wieght"), "{e}");
        let e = typo(&|v| v["aperture"]["vale"] = 1.0.into());
        assert!(e.contains("vale"), "{e}");
        let e = typo(&|v| v["enviroment"] = serde_json::json!({}));
        assert!(e.contains("enviroment"), "{e}");
        let e = typo(&|v| v["surfaces"][1]["material"]["catalogue"] = "X".into());
        assert!(e.contains("catalogue"), "{e}");
        // Tag-only variants take no other keys (serde's `deny_unknown_fields`
        // alone does not cover them): the value would be silently dropped.
        use serde_json::json;
        type Edit<'a> = &'a dyn Fn(&mut serde_json::Value);
        let cases: [(&str, Edit); 8] = [
            ("value", &|v| {
                v["surfaces"][2]["semi_diameter"] = json!({"mode": "auto", "value": 5.0})
            }),
            ("curvature", &|v| {
                v["surfaces"][6]["shape"] = json!({"type": "plane", "curvature": 0.02})
            }),
            ("tilt", &|v| {
                v["surfaces"][6]["shape"] = json!({"type": "coordinate_break", "tilt": 3.0})
            }),
            ("n", &|v| {
                v["surfaces"][2]["material"] = json!({"type": "air", "n": 1.5})
            }),
            ("n", &|v| {
                v["surfaces"][2]["material"] = json!({"type": "vacuum", "n": 1.0})
            }),
            ("tolerance", &|v| {
                v["ray_aiming"] = json!({"mode": "off", "tolerance": 1e-3})
            }),
            ("value", &|v| {
                v["aperture"] = json!({"type": "float_by_stop_size", "value": 4.0})
            }),
            ("surface", &|v| {
                v["configurations"] = json!([{"name": "c", "overrides": [
                    [{"param": "aperture_value", "surface": 3}, 2.0]]}])
            }),
        ];
        for (key, edit) in cases {
            let e = typo(edit);
            assert!(e.contains(&format!("unknown field `{key}`")), "{e}");
        }
        for p in [
            json!({"param": "temperature_c", "index": 0}),
            json!({"param": "pressure_atm", "x": 1}),
        ] {
            assert!(serde_json::from_value::<ParamPath>(p).is_err());
        }
        // The same keys without the extras load, and the JSON form of the
        // tag-only variants is unchanged.
        let mut v = good.clone();
        v["surfaces"][2]["semi_diameter"] = json!({"mode": "auto"});
        v["surfaces"][2]["material"] = json!({"type": "air"});
        v["aperture"] = json!({"type": "float_by_stop_size"});
        v["ray_aiming"] = json!({"mode": "off"});
        let s: OpticalSystem = serde_json::from_value(v.clone()).unwrap();
        assert_eq!(serde_json::to_value(&s).unwrap(), v);
        assert_eq!(
            serde_json::to_value(ParamPath::ApertureValue).unwrap(),
            json!({"param": "aperture_value"})
        );
        // Errors from the strict path still carry the JSON position.
        let text = serde_json::to_string_pretty(&{
            let mut v = good.clone();
            v["ray_aiming"] = json!({"mode": "off", "tolerance": 1e-3});
            v
        })
        .unwrap();
        let e = serde_json::from_str::<OpticalSystem>(&text)
            .unwrap_err()
            .to_string();
        assert!(e.contains("tolerance") && e.contains("line"), "{e}");
    }
}
