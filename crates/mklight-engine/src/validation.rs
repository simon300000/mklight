//! Engine-level model checks, reported by the `validate` analysis next to
//! the structural validation of `mklight-core`, so that a system reported
//! valid can also be traced by every ray-based analysis. Front ends (the
//! CLI's `validate` and `set`, the app, Python) use this one report.
//!
//! | code | severity | check |
//! |---|---|---|
//! | `non_finite` | error | a NaN or infinite number anywhere in the model, except the infinities the model allows (`"inf"` thickness of the object, or of the last surface for an afocal image) |
//! | `aperture_value` | error | the aperture value (EPD, F/#, NA) is not > 0 |
//! | `field_weight` | error | a field weight is negative or NaN |
//! | `wavelength_weight` | error | a wavelength weight is negative or NaN |
//! | `mechanics` | error | a mechanical part references a surface id the system does not have |
//! | `paraxial` | error | the system resolves, but the reference tracer cannot set it up (for example an F/# of 0, or an afocal system with an F/# aperture) |
//! | `field` | warning | a field cannot be resolved |
//! | `chief_ray` | warning | a chief ray (primary wavelength, unclipped) does not reach the image |
//! | `marginal_ray` | warning | the axial marginal ray (field 0, pupil `(0, 1)`, clipped) does not reach the image |
//!
//! Real-ray failures are warnings: a system in the middle of a redesign, or
//! one vignetted on purpose, is still a valid model. When the system does
//! not resolve (unknown glass and similar) only the model checks run: the
//! `validate` analysis has already reported the cause.

use serde::ser::{self, Serialize};
use std::fmt;
use std::sync::Arc;

use mklight_core::{ApertureSpec, MechanicalPart, OpticalSystem, ResolvedSystem};
use mklight_trace::{SequentialTracer, TraceStatus};

/// One validation finding (also the `issues` entries of `validate`).
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Issue {
    /// `error`, `warning` or `info`.
    pub severity: &'static str,
    pub code: &'static str,
    /// Surface index, when the issue belongs to one surface.
    pub surface: Option<usize>,
    pub message: String,
}

impl Issue {
    pub fn new(
        severity: &'static str,
        code: &'static str,
        surface: Option<usize>,
        message: String,
    ) -> Self {
        Issue {
            severity,
            code,
            surface,
            message,
        }
    }
}

/// Most `non_finite` issues listed individually; the rest are summarised.
const MAX_NON_FINITE: usize = 20;

/// Checks on the model alone (no materials, no tracing).
pub fn model_checks(sys: &OpticalSystem) -> Vec<Issue> {
    let mut out = Vec::new();
    let bad = non_finite_paths(sys);
    for p in bad.iter().take(MAX_NON_FINITE) {
        out.push(Issue::new(
            "error",
            "non_finite",
            surface_of_path(p),
            format!("{p} is not a finite number"),
        ));
    }
    if bad.len() > MAX_NON_FINITE {
        out.push(Issue::new(
            "error",
            "non_finite",
            None,
            format!(
                "… and {} more non-finite numbers",
                bad.len() - MAX_NON_FINITE
            ),
        ));
    }
    let aperture_value = match sys.aperture {
        ApertureSpec::EntrancePupilDiameter { value }
        | ApertureSpec::ImageSpaceFNumber { value }
        | ApertureSpec::ObjectSpaceNa { value }
        | ApertureSpec::ParaxialWorkingFNumber { value } => Some(value),
        ApertureSpec::FloatByStopSize => None,
    };
    if let Some(v) = aperture_value
        && !(v > 0.0 && v.is_finite())
    {
        out.push(Issue::new(
            "error",
            "aperture_value",
            None,
            format!("the aperture value must be > 0 (is {v})"),
        ));
    }
    for (k, f) in sys.fields.fields.iter().enumerate() {
        if f.weight.is_nan() || f.weight < 0.0 {
            out.push(Issue::new(
                "error",
                "field_weight",
                None,
                format!("field {k}: weight must be ≥ 0 (is {})", f.weight),
            ));
        }
    }
    for (k, w) in sys.wavelengths.wavelengths.iter().enumerate() {
        if w.weight.is_nan() || w.weight < 0.0 {
            out.push(Issue::new(
                "error",
                "wavelength_weight",
                None,
                format!("wavelength {k}: weight must be ≥ 0 (is {})", w.weight),
            ));
        }
    }
    for (k, m) in sys.mechanics.iter().enumerate() {
        let ids = match m {
            MechanicalPart::Barrel { from, to, .. } => vec![*from, *to],
            MechanicalPart::AnnularBaffle { surface, .. } => vec![*surface],
        };
        for id in ids {
            if sys.surface_index(id).is_none() {
                out.push(Issue::new(
                    "error",
                    "mechanics",
                    None,
                    format!(
                        "mechanical part {k} references surface id {}, which does not exist",
                        id.0
                    ),
                ));
            }
        }
    }
    out
}

/// Checks that need the resolved system: the reference tracer must set the
/// system up, and the chief and marginal rays should reach the image.
pub fn tracer_checks(resolved: Arc<ResolvedSystem>) -> Vec<Issue> {
    let mut out = Vec::new();
    let tracer = match SequentialTracer::new(resolved) {
        Ok(t) => t,
        Err(e) => {
            out.push(Issue::new(
                "error",
                "paraxial",
                None,
                format!("the reference tracer cannot set up the system: {e}"),
            ));
            return out;
        }
    };
    let wl = tracer.system().primary_wavelength;
    for (k, field) in tracer.fields().iter().enumerate() {
        if let Some(e) = &field.error {
            out.push(Issue::new(
                "warning",
                "field",
                None,
                format!("field {k}: {e}"),
            ));
            continue;
        }
        let r = tracer.chief(field, wl);
        if let Some(s) = r.status.surface() {
            out.push(Issue::new(
                "warning",
                "chief_ray",
                Some(s),
                format!(
                    "field {k}: the chief ray is stopped ({}) at surface {s}",
                    status_name(&r.status)
                ),
            ));
        }
    }
    if let Some(field) = tracer.fields().first().filter(|f| f.is_valid()) {
        let r = tracer.trace(field, wl, 0.0, 1.0, &tracer.options());
        if let Some(s) = r.status.surface() {
            out.push(Issue::new(
                "warning",
                "marginal_ray",
                Some(s),
                format!(
                    "field 0: the marginal ray (pupil 0, 1) is stopped ({}) at surface {s}",
                    status_name(&r.status)
                ),
            ));
        }
    }
    out
}

/// Short name of a trace status (`miss`, `clipped`, …).
pub fn status_name(s: &TraceStatus) -> &'static str {
    match s {
        TraceStatus::Ok => "ok",
        TraceStatus::Miss { .. } => "miss",
        TraceStatus::Clipped { .. } => "clipped",
        TraceStatus::TotalInternalReflection { .. } => "total_internal_reflection",
        TraceStatus::Failed { .. } => "failed",
    }
}

/// `surfaces[i]…` → `Some(i)`.
fn surface_of_path(p: &str) -> Option<usize> {
    let rest = p.strip_prefix("surfaces[")?;
    rest[..rest.find(']')?].parse().ok()
}

/// Paths (`surfaces[3].shape.conic`, `fields.fields[1].y`) of every
/// non-finite number in the model, in serialisation order. Infinite
/// thicknesses are written as the strings `"inf"`/`"-inf"` and so are not
/// numbers here (the structural validation decides where they are allowed);
/// a NaN thickness is reported.
pub fn non_finite_paths(sys: &OpticalSystem) -> Vec<String> {
    let mut w = Walker {
        path: Vec::new(),
        found: Vec::new(),
    };
    let _ = sys.serialize(&mut w);
    w.found
}

// --- a serde Serializer that records where non-finite floats are

struct Walker {
    path: Vec<String>,
    found: Vec<String>,
}

#[derive(Debug)]
struct WalkError(String);
impl fmt::Display for WalkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for WalkError {}
impl ser::Error for WalkError {
    fn custom<T: fmt::Display>(m: T) -> Self {
        WalkError(m.to_string())
    }
}

impl Walker {
    fn here(&self) -> String {
        let mut s = String::new();
        for p in &self.path {
            if p.starts_with('[') || s.is_empty() {
                s.push_str(p);
            } else {
                s.push('.');
                s.push_str(p);
            }
        }
        if s.is_empty() { "(root)".into() } else { s }
    }
    fn float(&mut self, v: f64) {
        if !v.is_finite() {
            let h = self.here();
            self.found.push(h);
        }
    }
    fn nested<T: Serialize + ?Sized>(&mut self, seg: String, v: &T) -> Result<(), WalkError> {
        self.path.push(seg);
        let r = v.serialize(&mut *self);
        self.path.pop();
        r
    }
}

/// Collection state: the next element index.
struct Coll<'a> {
    w: &'a mut Walker,
    i: usize,
    key: Option<String>,
}

macro_rules! ignore {
    ($($f:ident: $t:ty),*) => {
        $(fn $f(self, _: $t) -> Result<(), WalkError> { Ok(()) })*
    };
}

impl<'a> ser::Serializer for &'a mut Walker {
    type Ok = ();
    type Error = WalkError;
    type SerializeSeq = Coll<'a>;
    type SerializeTuple = Coll<'a>;
    type SerializeTupleStruct = Coll<'a>;
    type SerializeTupleVariant = Coll<'a>;
    type SerializeMap = Coll<'a>;
    type SerializeStruct = Coll<'a>;
    type SerializeStructVariant = Coll<'a>;

    ignore!(serialize_bool: bool, serialize_i8: i8, serialize_i16: i16, serialize_i32: i32,
            serialize_i64: i64, serialize_u8: u8, serialize_u16: u16, serialize_u32: u32,
            serialize_u64: u64, serialize_char: char, serialize_str: &str,
            serialize_bytes: &[u8], serialize_unit_struct: &'static str);

    fn serialize_f32(self, v: f32) -> Result<(), WalkError> {
        self.float(v as f64);
        Ok(())
    }
    fn serialize_f64(self, v: f64) -> Result<(), WalkError> {
        self.float(v);
        Ok(())
    }
    fn serialize_none(self) -> Result<(), WalkError> {
        Ok(())
    }
    fn serialize_some<T: Serialize + ?Sized>(self, v: &T) -> Result<(), WalkError> {
        v.serialize(self)
    }
    fn serialize_unit(self) -> Result<(), WalkError> {
        Ok(())
    }
    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
    ) -> Result<(), WalkError> {
        Ok(())
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        v: &T,
    ) -> Result<(), WalkError> {
        v.serialize(self)
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        v: &T,
    ) -> Result<(), WalkError> {
        self.nested(variant.to_string(), v)
    }
    fn serialize_seq(self, _: Option<usize>) -> Result<Coll<'a>, WalkError> {
        Ok(Coll {
            w: self,
            i: 0,
            key: None,
        })
    }
    fn serialize_tuple(self, _: usize) -> Result<Coll<'a>, WalkError> {
        Ok(Coll {
            w: self,
            i: 0,
            key: None,
        })
    }
    fn serialize_tuple_struct(self, _: &'static str, _: usize) -> Result<Coll<'a>, WalkError> {
        Ok(Coll {
            w: self,
            i: 0,
            key: None,
        })
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Coll<'a>, WalkError> {
        Ok(Coll {
            w: self,
            i: 0,
            key: None,
        })
    }
    fn serialize_map(self, _: Option<usize>) -> Result<Coll<'a>, WalkError> {
        Ok(Coll {
            w: self,
            i: 0,
            key: None,
        })
    }
    fn serialize_struct(self, _: &'static str, _: usize) -> Result<Coll<'a>, WalkError> {
        Ok(Coll {
            w: self,
            i: 0,
            key: None,
        })
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Coll<'a>, WalkError> {
        Ok(Coll {
            w: self,
            i: 0,
            key: None,
        })
    }
}

impl Coll<'_> {
    fn element<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), WalkError> {
        let seg = format!("[{}]", self.i);
        self.i += 1;
        self.w.nested(seg, v)
    }
}

macro_rules! seq_like {
    ($tr:ident, $m:ident) => {
        impl ser::$tr for Coll<'_> {
            type Ok = ();
            type Error = WalkError;
            fn $m<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), WalkError> {
                self.element(v)
            }
            fn end(self) -> Result<(), WalkError> {
                Ok(())
            }
        }
    };
}
seq_like!(SerializeSeq, serialize_element);
seq_like!(SerializeTuple, serialize_element);
seq_like!(SerializeTupleStruct, serialize_field);
seq_like!(SerializeTupleVariant, serialize_field);

impl ser::SerializeMap for Coll<'_> {
    type Ok = ();
    type Error = WalkError;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, k: &T) -> Result<(), WalkError> {
        self.key = Some(match serde_json::to_value(k) {
            Ok(serde_json::Value::String(s)) => s,
            Ok(v) => v.to_string(),
            Err(_) => "?".into(),
        });
        Ok(())
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), WalkError> {
        let k = self.key.take().unwrap_or_default();
        self.w.nested(k, v)
    }
    fn end(self) -> Result<(), WalkError> {
        Ok(())
    }
}

macro_rules! struct_like {
    ($tr:ident) => {
        impl ser::$tr for Coll<'_> {
            type Ok = ();
            type Error = WalkError;
            fn serialize_field<T: Serialize + ?Sized>(
                &mut self,
                k: &'static str,
                v: &T,
            ) -> Result<(), WalkError> {
                self.w.nested(k.to_string(), v)
            }
            fn end(self) -> Result<(), WalkError> {
                Ok(())
            }
        }
    };
}
struct_like!(SerializeStruct);
struct_like!(SerializeStructVariant);

#[cfg(test)]
mod tests {
    use super::*;
    use mklight_core::{SemiDiameter, SurfaceId, samples};

    fn codes(v: &[Issue]) -> Vec<&'static str> {
        v.iter().map(|i| i.code).collect()
    }

    #[test]
    fn samples_are_clean() {
        for (name, sys) in samples::all() {
            assert!(
                model_checks(&sys).is_empty(),
                "{name}: {:?}",
                model_checks(&sys)
            );
        }
    }

    #[test]
    fn finds_non_finite_numbers_anywhere() {
        let mut s = samples::cooke_triplet();
        s.surfaces[2].shape = mklight_core::SurfaceShape::Conic {
            curvature: 0.01,
            conic: f64::NAN,
        };
        s.surfaces[3].thickness = f64::NAN;
        s.fields.fields[1].y = f64::INFINITY;
        s.surfaces[4].semi_diameter = SemiDiameter::Fixed {
            value: f64::NEG_INFINITY,
        };
        let p = non_finite_paths(&s);
        assert_eq!(p.len(), 4, "{p:?}");
        assert!(
            p.iter()
                .any(|x| x.starts_with("surfaces[2].shape") && x.ends_with("conic")),
            "{p:?}"
        );
        assert!(p.contains(&"surfaces[3].thickness".to_string()), "{p:?}");
        assert!(p.iter().any(|x| x.starts_with("fields.fields[1]")), "{p:?}");
        let issues = model_checks(&s);
        assert_eq!(codes(&issues), vec!["non_finite"; 4]);
        let conic = issues.iter().find(|i| i.message.contains("conic")).unwrap();
        assert_eq!(conic.surface, Some(2));
        // The object's infinite distance is allowed.
        assert!(non_finite_paths(&samples::cooke_triplet()).is_empty());
    }

    #[test]
    fn aperture_weights_and_mechanics() {
        let mut s = samples::cooke_triplet();
        s.aperture = ApertureSpec::ImageSpaceFNumber { value: 0.0 };
        s.fields.fields[0].weight = -1.0;
        s.wavelengths.wavelengths[1].weight = f64::NAN;
        s.mechanics.push(MechanicalPart::AnnularBaffle {
            surface: SurfaceId(999),
            z: 0.0,
            inner_radius: 1.0,
            outer_radius: 2.0,
            finish: mklight_core::ScatterModel::Lambertian { tis: 0.1 },
        });
        let c = codes(&model_checks(&s));
        for k in [
            "aperture_value",
            "field_weight",
            "wavelength_weight",
            "mechanics",
        ] {
            assert!(c.contains(&k), "{k} missing from {c:?}");
        }
    }
}
